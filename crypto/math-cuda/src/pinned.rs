//! Host memory the card reads at DMA speed: jemalloc arenas whose pages are
//! page-locked, for the buffers a group's upload reads.
//!
//! A copy from pageable memory crosses the driver's own staging buffer on the
//! calling thread: on a block's commit that is ≈ 4 GB/s, and a group's upload
//! takes longer than the commit beside it. From page-locked memory the same
//! copy is a DMA the call only queues. A buffer page-locked after it is written
//! pays the lock and the unlock on every use; here the memory is locked once,
//! when an arena of ours takes it from the system, and jemalloc hands it out
//! again and again.
//!
//! With `LAMBDA_VM_PINNED_PRODUCER=1` (read once; off by default), a producer
//! allocates a buffer the card will read inside [`scope`]: the thread is bound
//! to one of [`ARENAS`] arenas of ours for that allocation alone, and the
//! buffer is an ordinary allocation of the global allocator, freed the same
//! way. Under another global allocator the binding moves nothing. Only
//! allocations of at least [`MIN_BYTES`] bind, above the thread cache, so no
//! small block of ours is handed to code outside a scope.
//!
//! The arenas' hooks take each extent from jemalloc's default hooks, touch its
//! pages and page-lock it before jemalloc hands any of it out. They never
//! purge, decommit or unmap it, and never merge two extents locked apart. They
//! allocate nothing (their bookkeeping is atomics in statics), so jemalloc is
//! never re-entered. Page-locking needs the CUDA context: an extent taken
//! before the backend is up stays pageable, which is slower, never wrong.
//!
//! A copy from or into page-locked memory returns before the DMA ends, where
//! one through pageable memory returns after it. The buffers made here are
//! read on the card only by a group's upload, which synchronizes its stream
//! before it returns.

use std::cell::Cell;
use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use cudarc::driver::sys;
use tikv_jemalloc_sys as jem;

/// The arenas the producers share, a thread to each in turn.
pub const ARENAS: usize = 8;

/// The smallest allocation that binds: above any thread cache.
pub const MIN_BYTES: usize = 1 << 20;

/// The most an arena takes from the system at once (`retain_grow_limit`): one
/// page-lock stays short, and an arena holds at most this much it has not
/// handed out.
pub const GROW_LIMIT: usize = 256 << 20;

/// The extents the hooks keep track of; past this many, an extent is left
/// pageable (at [`GROW_LIMIT`] each that is terabytes).
const MAX_CHUNKS: usize = 1 << 14;

/// Whether producers allocate inside the arenas: `LAMBDA_VM_PINNED_PRODUCER=1`.
pub fn enabled() -> bool {
    *ENABLED
        .get_or_init(|| std::env::var("LAMBDA_VM_PINNED_PRODUCER").is_ok_and(|v| v.trim() == "1"))
}

static ENABLED: OnceLock<bool> = OnceLock::new();

/// Turns the arenas on in a test process that has not read the knob yet;
/// whether they are on.
#[doc(hidden)]
pub fn enable_for_tests() -> bool {
    let _ = ENABLED.set(true);
    enabled()
}

/// Whether an allocation of `bytes` binds, with the knob `on`.
fn binds(bytes: usize, on: bool) -> bool {
    on && bytes >= MIN_BYTES
}

/// A thread bound to one of the arenas until it drops; the thread's own arena
/// again after. Inert when the allocation is small, the knob is off or the
/// arenas could not be made.
#[must_use]
pub struct Scope(Option<u32>);

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(previous) = self.0 {
            set_thread_arena(previous);
        }
    }
}

/// Binds this thread to one of the arenas for an allocation of `bytes`.
pub fn scope(bytes: usize) -> Scope {
    if !binds(bytes, enabled()) {
        return Scope(None);
    }
    warm();
    let Some(arenas) = arenas() else {
        return Scope(None);
    };
    if bytes <= arenas.tcache_max {
        return Scope(None);
    }
    let mine = arenas.ids[thread_slot() % ARENAS];
    match thread_arena() {
        Some(previous) if set_thread_arena(mine) => Scope(Some(previous)),
        _ => Scope(None),
    }
}

/// `bytes` from `ptr` going to the card: counted, and counted as page-locked
/// when they lie in one locked extent.
pub fn note_upload(ptr: *const u8, bytes: usize) {
    UPLOAD_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    if enabled() && is_pinned(ptr, bytes) {
        UPLOAD_PINNED.fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

/// Whether `bytes` from `ptr` lie in one extent the hooks page-locked.
pub fn is_pinned(ptr: *const u8, bytes: usize) -> bool {
    let at = ptr as usize;
    chunk_of(at).is_some_and(|k| {
        let chunk = &CHUNKS[k];
        chunk.pinned.load(Ordering::Acquire)
            && at + bytes <= chunk.base.load(Ordering::Relaxed) + chunk.len.load(Ordering::Relaxed)
    })
}

/// Whether the driver has `ptr` as page-locked host memory: its own answer,
/// for a test on a card.
#[doc(hidden)]
pub fn driver_has_locked(ptr: *const u8) -> bool {
    let mut flags = 0u32;
    // SAFETY: the driver only reads `ptr` as an address.
    unsafe {
        sys::cuMemHostGetFlags(&mut flags, ptr.cast_mut().cast()) == sys::CUresult::CUDA_SUCCESS
    }
}

/// What the arenas did so far in this process.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Extents page-locked, and their bytes.
    pub chunks: u64,
    pub pinned_bytes: u64,
    /// Extents left pageable (taken before the backend was up, or refused),
    /// and how many the driver refused.
    pub pageable_bytes: u64,
    pub refused: u64,
    /// Seconds in the hooks: touching the pages and locking them, and the
    /// locking alone.
    pub hook_seconds: f64,
    pub lock_seconds: f64,
    /// Bytes [`note_upload`] saw, and how many of them were page-locked.
    pub upload_bytes: u64,
    pub upload_pinned: u64,
}

pub fn stats() -> Stats {
    Stats {
        chunks: PINNED_CHUNKS.load(Ordering::Relaxed),
        pinned_bytes: PINNED_BYTES.load(Ordering::Relaxed),
        pageable_bytes: PAGEABLE_BYTES.load(Ordering::Relaxed),
        refused: REFUSED.load(Ordering::Relaxed),
        hook_seconds: HOOK_NANOS.load(Ordering::Relaxed) as f64 * 1e-9,
        lock_seconds: LOCK_NANOS.load(Ordering::Relaxed) as f64 * 1e-9,
        upload_bytes: UPLOAD_BYTES.load(Ordering::Relaxed),
        upload_pinned: UPLOAD_PINNED.load(Ordering::Relaxed),
    }
}

static PINNED_CHUNKS: AtomicU64 = AtomicU64::new(0);
static PINNED_BYTES: AtomicU64 = AtomicU64::new(0);
static PAGEABLE_BYTES: AtomicU64 = AtomicU64::new(0);
static REFUSED: AtomicU64 = AtomicU64::new(0);
static HOOK_NANOS: AtomicU64 = AtomicU64::new(0);
static LOCK_NANOS: AtomicU64 = AtomicU64::new(0);
static UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
static UPLOAD_PINNED: AtomicU64 = AtomicU64::new(0);

// ── The arenas ──────────────────────────────────────────────────────────────

struct Arenas {
    ids: [u32; ARENAS],
    /// jemalloc's thread-cache ceiling: an allocation at or under it can come
    /// from a cache of any arena, and never binds.
    tcache_max: usize,
}

/// The arenas, made at the first scope with the knob on; `None` when jemalloc
/// would not make them.
fn arenas() -> Option<&'static Arenas> {
    static ARENAS_MADE: OnceLock<Option<Arenas>> = OnceLock::new();
    ARENAS_MADE.get_or_init(make_arenas).as_ref()
}

fn make_arenas() -> Option<Arenas> {
    // SAFETY: each control is read or written with a value of its own type
    // and size, and every name is NUL-terminated.
    unsafe {
        let mut defaults: *mut jem::extent_hooks_t = std::ptr::null_mut();
        let mut len = std::mem::size_of::<*mut jem::extent_hooks_t>();
        let rc = jem::mallctl(
            c"arena.0.extent_hooks".as_ptr(),
            (&raw mut defaults).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        );
        if rc != 0 || defaults.is_null() {
            return None;
        }
        DEFAULT_HOOKS.store(defaults, Ordering::Release);
        let mut tcache_max = 0usize;
        let mut len = std::mem::size_of::<usize>();
        let rc = jem::mallctl(
            c"arenas.tcache_max".as_ptr(),
            (&raw mut tcache_max).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        );
        if rc != 0 {
            return None;
        }
        let mut ids = [0u32; ARENAS];
        for id in &mut ids {
            let mut hooks = HOOKS.0.get();
            let mut len = std::mem::size_of::<u32>();
            let rc = jem::mallctl(
                c"arenas.create".as_ptr(),
                (id as *mut u32).cast(),
                &mut len,
                (&raw mut hooks).cast(),
                std::mem::size_of::<*mut jem::extent_hooks_t>(),
            );
            if rc != 0 {
                return None;
            }
            // Only where jemalloc retains what it maps (Linux); elsewhere
            // each extent is mapped at its own size and there is no limit.
            let name = std::ffi::CString::new(format!("arena.{id}.retain_grow_limit")).ok()?;
            let mut limit = GROW_LIMIT;
            let _ = jem::mallctl(
                name.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                (&raw mut limit).cast(),
                std::mem::size_of::<usize>(),
            );
            // No decay: a purge the hooks refuse would only be retried.
            for knob in ["dirty_decay_ms", "muzzy_decay_ms"] {
                let name = std::ffi::CString::new(format!("arena.{id}.{knob}")).ok()?;
                let mut never: isize = -1;
                let _ = jem::mallctl(
                    name.as_ptr(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    (&raw mut never).cast(),
                    std::mem::size_of::<isize>(),
                );
            }
        }
        Some(Arenas { ids, tcache_max })
    }
}

/// This thread's arena, as jemalloc has it.
fn thread_arena() -> Option<u32> {
    let mut arena = 0u32;
    let mut len = std::mem::size_of::<u32>();
    // SAFETY: `thread.arena` reads an `unsigned`.
    let rc = unsafe {
        jem::mallctl(
            c"thread.arena".as_ptr(),
            (&raw mut arena).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0).then_some(arena)
}

fn set_thread_arena(mut arena: u32) -> bool {
    // SAFETY: `thread.arena` takes an `unsigned`.
    let rc = unsafe {
        jem::mallctl(
            c"thread.arena".as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            (&raw mut arena).cast(),
            std::mem::size_of::<u32>(),
        )
    };
    rc == 0
}

/// A number for this thread, handed out in turn: which arena it binds to.
fn thread_slot() -> usize {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    thread_local! {
        static SLOT: Cell<u32> = const { Cell::new(u32::MAX) };
    }
    SLOT.with(|slot| {
        if slot.get() == u32::MAX {
            slot.set(NEXT.fetch_add(1, Ordering::Relaxed));
        }
        slot.get() as usize
    })
}

// ── Page-locking ────────────────────────────────────────────────────────────

/// The context page-locking runs under, once [`warm`] found the backend up.
static CONTEXT: AtomicPtr<sys::CUctx_st> = AtomicPtr::new(std::ptr::null_mut());

/// Makes page-locking possible from inside a hook once the backend is up:
/// resolves the driver entry points the hooks call (resolving allocates, and a
/// hook must not), then publishes the context.
fn warm() {
    if !CONTEXT.load(Ordering::Acquire).is_null() {
        return;
    }
    let Some(backend) = crate::device::backend_if_ready() else {
        return;
    };
    let context = backend.ctx.cu_ctx();
    // SAFETY: the context is the backend's, alive for the process; a null
    // pointer of zero bytes is refused by the driver and only resolves the
    // entry point.
    unsafe {
        let _ = sys::cuCtxSetCurrent(context);
        let _ = sys::cuMemHostRegister_v2(std::ptr::null_mut(), 0, 0);
        let _ = sys::cuMemHostUnregister(std::ptr::null_mut());
    }
    CONTEXT.store(context, Ordering::Release);
}

/// How an extent is page-locked: the driver, or a test's stand-in. `None`
/// before the driver can (the backend not up yet), else whether it did.
type Lock = fn(*mut u8, usize) -> Option<bool>;
type Unlock = fn(*mut u8);

static LOCK: AtomicUsize = AtomicUsize::new(0);
static UNLOCK: AtomicUsize = AtomicUsize::new(0);

fn lock(ptr: *mut u8, len: usize) -> Option<bool> {
    match LOCK.load(Ordering::Acquire) {
        0 => driver_lock(ptr, len),
        // SAFETY: only `set_lock` stores here, and it stores a `Lock`.
        f => unsafe { std::mem::transmute::<usize, Lock>(f)(ptr, len) },
    }
}

fn unlock(ptr: *mut u8) {
    match UNLOCK.load(Ordering::Acquire) {
        0 => driver_unlock(ptr),
        // SAFETY: only `set_lock` stores here, and it stores an `Unlock`.
        f => unsafe { std::mem::transmute::<usize, Unlock>(f)(ptr) },
    }
}

/// A stand-in for the driver's page-locking, for tests without a card.
#[doc(hidden)]
pub fn set_lock(lock: Lock, unlock: Unlock) {
    LOCK.store(lock as usize, Ordering::Release);
    UNLOCK.store(unlock as usize, Ordering::Release);
}

fn driver_lock(ptr: *mut u8, len: usize) -> Option<bool> {
    let context = CONTEXT.load(Ordering::Acquire);
    if context.is_null() {
        return None;
    }
    // SAFETY: `ptr..ptr + len` is an extent jemalloc's default hooks just
    // mapped, committed and not yet handed out; the context is the backend's.
    Some(unsafe {
        sys::cuCtxSetCurrent(context) == sys::CUresult::CUDA_SUCCESS
            && sys::cuMemHostRegister_v2(ptr.cast(), len, sys::CU_MEMHOSTREGISTER_PORTABLE)
                == sys::CUresult::CUDA_SUCCESS
    })
}

fn driver_unlock(ptr: *mut u8) {
    let context = CONTEXT.load(Ordering::Acquire);
    // SAFETY: `ptr` is the base of a range `driver_lock` registered.
    unsafe {
        let _ = sys::cuCtxSetCurrent(context);
        let _ = sys::cuMemHostUnregister(ptr.cast());
    }
}

// ── The extents ─────────────────────────────────────────────────────────────

/// An extent the hooks took from the system: where, how long, and whether it
/// is page-locked. `ready` publishes the other three.
struct Chunk {
    base: AtomicUsize,
    len: AtomicUsize,
    pinned: AtomicBool,
    ready: AtomicBool,
}

impl Chunk {
    const fn new() -> Self {
        Self {
            base: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            pinned: AtomicBool::new(false),
            ready: AtomicBool::new(false),
        }
    }
}

static CHUNKS: [Chunk; MAX_CHUNKS] = [const { Chunk::new() }; MAX_CHUNKS];
static NCHUNKS: AtomicUsize = AtomicUsize::new(0);

/// The chunk `at` lies in.
fn chunk_of(at: usize) -> Option<usize> {
    let n = NCHUNKS.load(Ordering::Acquire).min(MAX_CHUNKS);
    (0..n).find(|&k| {
        let chunk = &CHUNKS[k];
        chunk.ready.load(Ordering::Acquire) && {
            let base = chunk.base.load(Ordering::Relaxed);
            (base..base + chunk.len.load(Ordering::Relaxed)).contains(&at)
        }
    })
}

/// Whether two adjacent extents may become one: only inside one chunk, so no
/// extent ever spans two ranges locked apart (or a locked and a pageable one).
fn may_merge(a: usize, b_end: usize) -> bool {
    match (chunk_of(a), chunk_of(b_end - 1)) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

// ── The hooks ───────────────────────────────────────────────────────────────

/// The hooks' table, read by jemalloc for the arenas' lifetime.
struct Hooks(std::cell::UnsafeCell<jem::extent_hooks_t>);

// SAFETY: jemalloc only reads the table, and nothing here writes it.
unsafe impl Sync for Hooks {}

static HOOKS: Hooks = Hooks(std::cell::UnsafeCell::new(jem::extent_hooks_t {
    alloc: Some(hook_alloc),
    dalloc: Some(hook_dalloc),
    destroy: Some(hook_destroy),
    // Committed from the start and never decommitted or purged: a locked
    // page stays the page the driver locked.
    commit: None,
    decommit: None,
    purge_lazy: None,
    purge_forced: None,
    split: Some(hook_split),
    merge: Some(hook_merge),
}));

/// jemalloc's own hooks, which map and unmap for ours.
static DEFAULT_HOOKS: AtomicPtr<jem::extent_hooks_t> = AtomicPtr::new(std::ptr::null_mut());

/// Touching stride: the smallest page any host here uses.
const TOUCH: usize = 4096;

unsafe extern "C" fn hook_alloc(
    _hooks: *mut jem::extent_hooks_t,
    new_addr: *mut c_void,
    size: usize,
    alignment: usize,
    zero: *mut bool,
    commit: *mut bool,
    arena_ind: u32,
) -> *mut c_void {
    let t = Instant::now();
    let defaults = DEFAULT_HOOKS.load(Ordering::Acquire);
    // SAFETY: the default table is jemalloc's static one, and its `alloc` is
    // called with the arguments jemalloc gave this one; a locked page must be
    // committed, so committing is asked for.
    let ptr = unsafe {
        let Some(alloc) = defaults.as_ref().and_then(|d| d.alloc) else {
            return std::ptr::null_mut();
        };
        *commit = true;
        alloc(defaults, new_addr, size, alignment, zero, commit, arena_ind)
    };
    if ptr.is_null() {
        return ptr;
    }
    let k = NCHUNKS.fetch_add(1, Ordering::AcqRel);
    // SAFETY: `commit` is jemalloc's out-parameter, just written.
    let committed = unsafe { *commit };
    let pinned = k < MAX_CHUNKS && committed && {
        // SAFETY: the extent is `size` committed bytes no one else has yet;
        // a byte rewritten with its own value leaves it as it was, zeroed or
        // not.
        unsafe {
            let base = ptr.cast::<u8>();
            for at in (0..size).step_by(TOUCH) {
                let p = base.add(at);
                p.write_volatile(p.read_volatile());
            }
        }
        let l = Instant::now();
        let locked = lock(ptr.cast(), size);
        LOCK_NANOS.fetch_add(l.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if locked == Some(false) {
            REFUSED.fetch_add(1, Ordering::Relaxed);
        }
        locked == Some(true)
    };
    if k < MAX_CHUNKS {
        let chunk = &CHUNKS[k];
        chunk.base.store(ptr as usize, Ordering::Relaxed);
        chunk.len.store(size, Ordering::Relaxed);
        chunk.pinned.store(pinned, Ordering::Relaxed);
        chunk.ready.store(true, Ordering::Release);
    }
    if pinned {
        PINNED_CHUNKS.fetch_add(1, Ordering::Relaxed);
        PINNED_BYTES.fetch_add(size as u64, Ordering::Relaxed);
    } else {
        PAGEABLE_BYTES.fetch_add(size as u64, Ordering::Relaxed);
    }
    HOOK_NANOS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    ptr
}

/// Kept: a locked range is never handed back to the system while the process
/// lives (jemalloc retains it and reuses it).
unsafe extern "C" fn hook_dalloc(
    _hooks: *mut jem::extent_hooks_t,
    _addr: *mut c_void,
    _size: usize,
    _committed: bool,
    _arena_ind: u32,
) -> bool {
    true
}

/// Only an arena's destruction, which nothing here asks for: the chunk is
/// unlocked as a whole before any of it is unmapped.
unsafe extern "C" fn hook_destroy(
    _hooks: *mut jem::extent_hooks_t,
    addr: *mut c_void,
    size: usize,
    committed: bool,
    arena_ind: u32,
) {
    if let Some(k) = chunk_of(addr as usize)
        && CHUNKS[k].pinned.swap(false, Ordering::AcqRel)
    {
        unlock(CHUNKS[k].base.load(Ordering::Relaxed) as *mut u8);
    }
    let defaults = DEFAULT_HOOKS.load(Ordering::Acquire);
    // SAFETY: as for `hook_alloc`'s call of the default table.
    unsafe {
        if let Some(destroy) = defaults.as_ref().and_then(|d| d.destroy) {
            destroy(defaults, addr, size, committed, arena_ind);
        }
    }
}

/// Splitting moves no page: always.
unsafe extern "C" fn hook_split(
    _hooks: *mut jem::extent_hooks_t,
    _addr: *mut c_void,
    _size: usize,
    _size_a: usize,
    _size_b: usize,
    _committed: bool,
    _arena_ind: u32,
) -> bool {
    false
}

/// Refused (`true`) unless both lie in one chunk.
unsafe extern "C" fn hook_merge(
    _hooks: *mut jem::extent_hooks_t,
    addr_a: *mut c_void,
    _size_a: usize,
    addr_b: *mut c_void,
    size_b: usize,
    _committed: bool,
    _arena_ind: u32,
) -> bool {
    !may_merge(addr_a as usize, addr_b as usize + size_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The arena jemalloc says `ptr` belongs to.
    fn arena_of(ptr: *mut c_void) -> u32 {
        let (mut arena, mut len, mut p) = (0u32, std::mem::size_of::<u32>(), ptr);
        // SAFETY: `arenas.lookup` takes a pointer jemalloc allocated and
        // reads back an `unsigned`.
        let rc = unsafe {
            jem::mallctl(
                c"arenas.lookup".as_ptr(),
                (&raw mut arena).cast(),
                &mut len,
                (&raw mut p).cast(),
                std::mem::size_of::<*mut c_void>(),
            )
        };
        assert_eq!(rc, 0, "arenas.lookup");
        arena
    }

    static LOCKED: AtomicU64 = AtomicU64::new(0);

    fn fake_lock(_ptr: *mut u8, len: usize) -> Option<bool> {
        LOCKED.fetch_add(len as u64, Ordering::Relaxed);
        Some(true)
    }

    fn fake_unlock(_ptr: *mut u8) {}

    /// The knob on and the driver stood in for, in this test process.
    fn on() -> &'static Arenas {
        let _ = ENABLED.set(true);
        set_lock(fake_lock, fake_unlock);
        arenas().expect("the arenas")
    }

    /// `bytes` from jemalloc on this thread, as the global allocator would ask.
    fn take(bytes: usize) -> *mut c_void {
        // SAFETY: a plain allocation, freed by `give`.
        let ptr = unsafe { jem::mallocx(bytes, 0) };
        assert!(!ptr.is_null());
        ptr
    }

    fn give(ptr: *mut c_void, bytes: usize) {
        // SAFETY: `ptr` came from `take(bytes)`.
        unsafe { jem::sdallocx(ptr, bytes, 0) };
    }

    #[test]
    fn a_large_allocation_in_a_scope_is_page_locked_and_one_outside_is_not() {
        let arenas = on();
        let bytes = 8 << 20;
        let inside = {
            let _scope = scope(bytes);
            take(bytes)
        };
        let outside = take(bytes);
        assert!(arenas.ids.contains(&arena_of(inside)));
        assert!(is_pinned(inside.cast(), bytes));
        assert!(!arenas.ids.contains(&arena_of(outside)));
        assert!(!is_pinned(outside.cast(), bytes));
        assert!(LOCKED.load(Ordering::Relaxed) >= bytes as u64);
        give(inside, bytes);
        give(outside, bytes);
    }

    #[test]
    fn the_knob_off_binds_nothing() {
        assert!(!binds(1 << 30, false));
        assert!(!binds(MIN_BYTES - 1, true));
        assert!(binds(MIN_BYTES, true));
    }

    #[test]
    fn a_small_allocation_never_binds() {
        let arenas = on();
        let before = thread_arena().expect("the thread's arena");
        let small = {
            let scope = scope(MIN_BYTES - 1);
            assert!(scope.0.is_none());
            assert_eq!(thread_arena(), Some(before));
            take(4096)
        };
        assert!(!arenas.ids.contains(&arena_of(small)));
        give(small, 4096);
    }

    #[test]
    fn a_scope_gives_the_thread_its_arena_back_even_nested() {
        let arenas = on();
        let before = thread_arena().expect("the thread's arena");
        {
            let _outer = scope(MIN_BYTES << 1);
            let bound = thread_arena().expect("bound");
            assert!(arenas.ids.contains(&bound));
            {
                let _inner = scope(MIN_BYTES << 1);
                assert_eq!(thread_arena(), Some(bound));
            }
            assert_eq!(thread_arena(), Some(bound));
        }
        assert_eq!(thread_arena(), Some(before));
    }

    #[test]
    fn a_freed_block_stays_locked_through_a_purge_and_serves_the_next() {
        let arenas = on();
        let bytes = 16 << 20;
        let first = {
            let _scope = scope(bytes);
            take(bytes)
        };
        let arena = arena_of(first);
        give(first, bytes);
        // The hooks refuse every purge: the pages stay locked and mapped.
        let name = std::ffi::CString::new(format!("arena.{arena}.purge")).expect("a name");
        // SAFETY: a control that takes no value.
        let rc = unsafe {
            jem::mallctl(
                name.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(rc, 0, "arena purge");
        let again = {
            let _scope = scope(bytes);
            take(bytes)
        };
        assert!(arenas.ids.contains(&arena_of(again)));
        assert!(is_pinned(again.cast(), bytes));
        give(again, bytes);
    }

    #[test]
    fn merges_stay_inside_one_chunk() {
        let _ = on();
        let bytes = 4 << 20;
        let a = {
            let _scope = scope(bytes);
            take(bytes)
        };
        let k = chunk_of(a as usize).expect("a's chunk");
        let (base, len) = (
            CHUNKS[k].base.load(Ordering::Relaxed),
            CHUNKS[k].len.load(Ordering::Relaxed),
        );
        assert!(may_merge(base, base + len));
        assert!(!may_merge(base, base + len + 1));
        assert!(!may_merge(base - 1, base + 1));
        give(a, bytes);
    }

    #[test]
    fn uploads_are_counted_and_the_locked_ones_apart() {
        let _ = on();
        let bytes = 2 << 20;
        let locked = {
            let _scope = scope(bytes);
            take(bytes)
        };
        let plain = take(bytes);
        let before = stats();
        note_upload(locked.cast(), bytes);
        note_upload(plain.cast(), bytes);
        let after = stats();
        assert!(after.upload_bytes >= before.upload_bytes + 2 * bytes as u64);
        assert!(after.upload_pinned >= before.upload_pinned + bytes as u64);
        give(locked, bytes);
        give(plain, bytes);
    }
}
