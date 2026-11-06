//! Host memory the card reads at DMA speed: a pool of page-locked memory,
//! locked once before any prove, and jemalloc arenas that hand it out for the
//! buffers a group's upload reads.
//!
//! A copy from pageable memory crosses the driver's own staging buffer on the
//! calling thread, and on a block's commit it takes longer than the commit
//! beside it. From page-locked memory the same copy is a DMA the call only
//! queues. Locking memory is serialized across the process and holds every
//! other thread's driver calls while it runs, so it is done once, up front
//! ([`lock_pool`]), and never while a prove runs: the arenas hand out the pool
//! and, once it is spent, plain pageable memory, which is slower and never
//! wrong.
//!
//! With `LAMBDA_VM_PINNED_POOL_GIB=<n>` (read once; unset or 0 is off) and the
//! pool locked, a producer allocates a buffer the card will read inside
//! [`scope`]: the thread is bound to one of [`ARENAS`] arenas of ours for that
//! allocation alone, and the buffer is an ordinary allocation of the global
//! allocator, freed the same way. Under another global allocator the binding
//! moves nothing. Only allocations of at least [`MIN_BYTES`] bind, above the
//! thread cache, so no small block of ours is handed to code outside a scope.
//!
//! The arenas' hooks carve each extent from the pool, or take it from
//! jemalloc's default hooks once the pool is spent. They never purge,
//! decommit or unmap an extent, and never merge two extents that lie in
//! different pieces of the pool (each piece is locked apart). They allocate
//! nothing and lock nothing (their bookkeeping is atomics in statics), so
//! jemalloc is never re-entered and no driver call waits on them.
//!
//! A copy from or into page-locked memory returns before the DMA ends, where
//! one through pageable memory returns after it. The buffers made here are
//! read on the card only by a group's upload, which synchronizes its stream
//! before it returns. Their pages must never be handed back to the OS
//! (`madvise`) while locked: a caller that would checks [`is_pinned`] first.

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

/// The most an arena asks of its hooks at once (`retain_grow_limit`): what a
/// piece can lose at its end to a request that does not fit.
pub const GROW_LIMIT: usize = 64 << 20;

/// The pool is locked a piece at a time; an extent never spans two pieces.
pub const PIECE: usize = 1 << 30;

/// The extents the hooks keep track of; past this many, an extent the
/// default hooks made is not recorded (and can merge with nothing).
const MAX_CHUNKS: usize = 1 << 14;

/// The pool's size, `LAMBDA_VM_PINNED_POOL_GIB` GiB; 0 is off.
pub fn pool_gib() -> usize {
    *POOL_GIB.get_or_init(|| {
        std::env::var("LAMBDA_VM_PINNED_POOL_GIB")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

static POOL_GIB: OnceLock<usize> = OnceLock::new();

/// Whether the pool is asked for.
pub fn enabled() -> bool {
    pool_gib() > 0
}

/// Sets the pool's size in a test process that has not read the knob yet;
/// the size it has.
#[doc(hidden)]
pub fn set_pool_gib_for_tests(gib: usize) -> usize {
    let _ = POOL_GIB.set(gib);
    pool_gib()
}

/// Whether an allocation of `bytes` binds, with the pool `ready`.
fn binds(bytes: usize, ready: bool) -> bool {
    ready && bytes >= MIN_BYTES
}

/// A thread bound to one of the arenas until it drops; the thread's own arena
/// again after. Inert when the allocation is small, the pool is not locked or
/// the arenas could not be made.
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
    if !binds(bytes, POOL_READY.load(Ordering::Acquire)) {
        return Scope(None);
    }
    let Some(arenas) = arenas() else {
        return Scope(None);
    };
    if bytes <= arenas.tcache_max {
        return Scope(None);
    }
    let mine = arenas.ids[thread_slot() % ARENAS];
    match thread_arena() {
        Some(previous) if set_thread_arena(mine) => {
            let now = PENDING.fetch_add(bytes as u64, Ordering::Relaxed) + bytes as u64;
            PENDING_MOST.fetch_max(now, Ordering::Relaxed);
            Scope(Some(previous))
        }
        _ => Scope(None),
    }
}

/// `bytes` from `ptr` going to the card: counted, and counted as page-locked
/// when they lie in the pool.
pub fn note_upload(ptr: *const u8, bytes: usize) {
    UPLOAD_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    if POOL_READY.load(Ordering::Acquire) && is_pinned(ptr, bytes) {
        UPLOAD_PINNED.fetch_add(bytes as u64, Ordering::Relaxed);
        let _ = PENDING.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |p| {
            Some(p.saturating_sub(bytes as u64))
        });
    }
}

/// Whether `bytes` from `ptr` lie in one locked piece of the pool.
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

/// What locking the pool took ([`lock_pool`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct PoolLocked {
    /// Bytes asked for, and bytes of the pieces the driver locked.
    pub bytes: u64,
    pub locked: u64,
    pub pieces: usize,
    /// Pieces the driver refused (left pageable).
    pub refused: usize,
    /// Wall seconds, mapping and locking.
    pub seconds: f64,
}

/// Locks the pool, once: `LAMBDA_VM_PINNED_POOL_GIB` GiB of fresh memory,
/// locked a [`PIECE`] at a time (the driver faults the pages in as it locks
/// them). `None` when the knob is off, the card is not up, or the pool was
/// locked already. Call it before any prove: while it runs, every other
/// thread's driver calls wait.
pub fn lock_pool() -> Option<PoolLocked> {
    let gib = pool_gib();
    if gib == 0 || POOL_TRIED.swap(true, Ordering::AcqRel) {
        return None;
    }
    if LOCK.load(Ordering::Acquire) == 0 {
        let backend = crate::device::backend().ok()?;
        let context = backend.ctx.cu_ctx();
        // SAFETY: the context is the backend's, alive for the process.
        if unsafe { sys::cuCtxSetCurrent(context) } != sys::CUresult::CUDA_SUCCESS {
            return None;
        }
        CONTEXT.store(context, Ordering::Release);
    }
    let t = Instant::now();
    let bytes = gib.checked_mul(1 << 30)?;
    // Mapped here, not taken from the global allocator, so the allocator's
    // own figures count only what the arenas hand out of it, as they would
    // count the same allocations without the pool.
    // SAFETY: a fresh private anonymous mapping, never unmapped (its pieces
    // stay locked and handed out for the process's life).
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if base == libc::MAP_FAILED {
        return None;
    }
    let base = base.cast::<u8>();
    let mut report = PoolLocked {
        bytes: bytes as u64,
        ..PoolLocked::default()
    };
    for at in (0..bytes).step_by(PIECE) {
        let len = PIECE.min(bytes - at);
        // SAFETY: inside the pool's allocation.
        let piece = unsafe { base.add(at) };
        let locked = lock(piece, len);
        let k = NCHUNKS.fetch_add(1, Ordering::AcqRel);
        if k < MAX_CHUNKS {
            let chunk = &CHUNKS[k];
            chunk.base.store(piece as usize, Ordering::Relaxed);
            chunk.len.store(len, Ordering::Relaxed);
            chunk.pinned.store(locked, Ordering::Relaxed);
            chunk.pool.store(true, Ordering::Relaxed);
            chunk.ready.store(true, Ordering::Release);
        }
        report.pieces += 1;
        if locked {
            report.locked += len as u64;
        } else {
            report.refused += 1;
        }
    }
    report.seconds = t.elapsed().as_secs_f64();
    POOL_BASE.store(base as usize, Ordering::Relaxed);
    POOL_LEN.store(bytes, Ordering::Relaxed);
    POOL_LOCKED.store(report.locked, Ordering::Relaxed);
    LOCK_NANOS.store(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    POOL_READY.store(true, Ordering::Release);
    Some(report)
}

/// The pool's bytes no allocation holds: resident from the lock on (the driver
/// faulted them in) and spoken for, the memory a block's later allocations take
/// instead of growing the heap. That is the part never handed out and the
/// part the arenas hold free (handed out, then freed, or kept between their
/// extents). A decision on the host's memory that forecasts that growth takes
/// these off its reading, or it counts them twice. 0 without a pool.
pub fn unused_bytes() -> u64 {
    if !netting() {
        return 0;
    }
    let len = POOL_LEN.load(Ordering::Relaxed) as u64;
    let held = live_bytes().unwrap_or(CURSOR.load(Ordering::Relaxed) as u64);
    len.saturating_sub(held.min(len))
}

/// What the arenas hold of the pool and no allocation does (resident in
/// jemalloc's figures, and no purge returns it: the hooks refuse every one).
/// A reading of the allocator's resident bytes takes it off as the host's
/// readings take [`unused_bytes`] off. 0 without a pool, after
/// [`retire_netting`], or where jemalloc keeps no figures.
pub fn arena_free_bytes() -> u64 {
    if !netting() {
        return 0;
    }
    let (Some(arenas), Some(live)) = (arenas_if_made(), live_bytes()) else {
        return 0;
    };
    let mut resident = 0u64;
    for id in arenas.ids {
        let Ok(name) = std::ffi::CString::new(format!("stats.arenas.{id}.resident")) else {
            return 0;
        };
        let mut bytes = 0usize;
        let mut len = std::mem::size_of::<usize>();
        // SAFETY: the statistic is a `size_t`; the name is NUL-terminated.
        let rc = unsafe {
            jem::mallctl(
                name.as_ptr(),
                (&raw mut bytes).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 {
            return 0;
        }
        resident += bytes as u64;
    }
    resident.saturating_sub(live)
}

/// The block's prove is over: the pool's free bytes stop being room for its
/// allocations (nothing after the base allocates from it), so the host's
/// readings count them again, as the memory they are.
pub fn retire_netting() {
    NETTING_RETIRED.store(true, Ordering::Release);
}

fn netting() -> bool {
    POOL_READY.load(Ordering::Acquire) && !NETTING_RETIRED.load(Ordering::Acquire)
}

static NETTING_RETIRED: AtomicBool = AtomicBool::new(false);

/// What the arenas' allocations hold of the pool now, from jemalloc's own
/// figures (refreshed first); `None` where jemalloc keeps none (built without
/// its statistics) or the arenas were never made.
fn live_bytes() -> Option<u64> {
    let arenas = arenas_if_made()?;
    // SAFETY: `epoch` takes and gives back a `u64`; each statistic is a
    // `size_t`, and every name is NUL-terminated.
    unsafe {
        let mut epoch = 1u64;
        let mut len = std::mem::size_of::<u64>();
        if jem::mallctl(
            c"epoch".as_ptr(),
            (&raw mut epoch).cast(),
            &mut len,
            (&raw mut epoch).cast(),
            len,
        ) != 0
        {
            return None;
        }
        let mut held = 0u64;
        for id in arenas.ids {
            for kind in ["small", "large"] {
                let name =
                    std::ffi::CString::new(format!("stats.arenas.{id}.{kind}.allocated")).ok()?;
                let mut bytes = 0usize;
                let mut len = std::mem::size_of::<usize>();
                if jem::mallctl(
                    name.as_ptr(),
                    (&raw mut bytes).cast(),
                    &mut len,
                    std::ptr::null_mut(),
                    0,
                ) != 0
                {
                    return None;
                }
                held += bytes as u64;
            }
        }
        Some(held)
    }
}

static POOL_TRIED: AtomicBool = AtomicBool::new(false);
static POOL_READY: AtomicBool = AtomicBool::new(false);
static POOL_BASE: AtomicUsize = AtomicUsize::new(0);
static POOL_LEN: AtomicUsize = AtomicUsize::new(0);
/// Bytes of the pool handed out so far: it is handed out front to back, once.
static CURSOR: AtomicUsize = AtomicUsize::new(0);

/// What the pool did so far in this process.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// The pool's bytes, and those the driver locked.
    pub pool_bytes: u64,
    pub locked_bytes: u64,
    /// Seconds locking it took.
    pub lock_seconds: f64,
    /// Bytes of the pool the arenas took, and bytes they took from the default
    /// hooks once it was spent (pageable).
    pub carved_bytes: u64,
    pub spilled_bytes: u64,
    /// Bytes [`note_upload`] saw, and how many of them came from the pool.
    pub upload_bytes: u64,
    pub upload_pinned: u64,
    /// The most bytes allocated in a scope and not yet uploaded at once: about
    /// what a pool that is reused after each upload would have to hold.
    pub pending_most: u64,
    /// What the arenas' allocations hold of the pool now, from jemalloc's own
    /// figures; `None` where it keeps none (then [`unused_bytes`] reads the
    /// pool less what was ever handed out).
    pub held_bytes: Option<u64>,
}

pub fn stats() -> Stats {
    Stats {
        pool_bytes: POOL_LEN.load(Ordering::Relaxed) as u64,
        locked_bytes: POOL_LOCKED.load(Ordering::Relaxed),
        lock_seconds: LOCK_NANOS.load(Ordering::Relaxed) as f64 * 1e-9,
        carved_bytes: CURSOR.load(Ordering::Relaxed) as u64,
        spilled_bytes: SPILLED_BYTES.load(Ordering::Relaxed),
        upload_bytes: UPLOAD_BYTES.load(Ordering::Relaxed),
        upload_pinned: UPLOAD_PINNED.load(Ordering::Relaxed),
        pending_most: PENDING_MOST.load(Ordering::Relaxed),
        held_bytes: live_bytes(),
    }
}

static POOL_LOCKED: AtomicU64 = AtomicU64::new(0);
static LOCK_NANOS: AtomicU64 = AtomicU64::new(0);
static SPILLED_BYTES: AtomicU64 = AtomicU64::new(0);
static UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
static UPLOAD_PINNED: AtomicU64 = AtomicU64::new(0);
static PENDING: AtomicU64 = AtomicU64::new(0);
static PENDING_MOST: AtomicU64 = AtomicU64::new(0);

/// An extent of `size` bytes at `alignment` from the pool, inside one piece;
/// `None` once the pool cannot give it.
fn carve(size: usize, alignment: usize) -> Option<usize> {
    let (base, len) = (
        POOL_BASE.load(Ordering::Relaxed),
        POOL_LEN.load(Ordering::Relaxed),
    );
    if size > PIECE || len == 0 {
        return None;
    }
    let align = alignment.max(1);
    let mut cursor = CURSOR.load(Ordering::Relaxed);
    loop {
        let mut at = (base + cursor).next_multiple_of(align) - base;
        let piece_end = (at / PIECE + 1) * PIECE;
        if at + size > piece_end {
            at = (base + piece_end).next_multiple_of(align) - base;
        }
        if at + size > len {
            return None;
        }
        match CURSOR.compare_exchange_weak(cursor, at + size, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return Some(base + at),
            Err(now) => cursor = now,
        }
    }
}

// ── The arenas ──────────────────────────────────────────────────────────────

struct Arenas {
    ids: [u32; ARENAS],
    /// jemalloc's thread-cache ceiling: an allocation at or under it can come
    /// from a cache of any arena, and never binds.
    tcache_max: usize,
}

/// The arenas, made at the first scope with the pool locked; `None` when
/// jemalloc would not make them.
fn arenas() -> Option<&'static Arenas> {
    ARENAS_MADE.get_or_init(make_arenas).as_ref()
}

/// The arenas when a scope made them already, without making them.
fn arenas_if_made() -> Option<&'static Arenas> {
    ARENAS_MADE.get().and_then(Option::as_ref)
}

static ARENAS_MADE: OnceLock<Option<Arenas>> = OnceLock::new();

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
            // each extent is asked for at its own size and there is no limit.
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

// ── Page-locking (the pool's, once) ─────────────────────────────────────────

/// The context the pool is locked under.
static CONTEXT: AtomicPtr<sys::CUctx_st> = AtomicPtr::new(std::ptr::null_mut());

/// How a piece is page-locked: the driver, or a test's stand-in.
type Lock = fn(*mut u8, usize) -> bool;

static LOCK: AtomicUsize = AtomicUsize::new(0);

fn lock(ptr: *mut u8, len: usize) -> bool {
    match LOCK.load(Ordering::Acquire) {
        0 => driver_lock(ptr, len),
        // SAFETY: only `set_lock` stores here, and it stores a `Lock`.
        f => unsafe { std::mem::transmute::<usize, Lock>(f)(ptr, len) },
    }
}

/// A stand-in for the driver's page-locking, for tests without a card.
#[doc(hidden)]
pub fn set_lock(lock: Lock) {
    LOCK.store(lock as usize, Ordering::Release);
}

fn driver_lock(ptr: *mut u8, len: usize) -> bool {
    let context = CONTEXT.load(Ordering::Acquire);
    // SAFETY: `ptr..ptr + len` is a piece of the pool's own allocation, which
    // is never freed; the context is the backend's.
    unsafe {
        sys::cuCtxSetCurrent(context) == sys::CUresult::CUDA_SUCCESS
            && sys::cuMemHostRegister_v2(ptr.cast(), len, sys::CU_MEMHOSTREGISTER_PORTABLE)
                == sys::CUresult::CUDA_SUCCESS
    }
}

// ── The extents ─────────────────────────────────────────────────────────────

/// An extent the hooks hand jemalloc from: a piece of the pool, or one the
/// default hooks made. `ready` publishes the other four.
struct Chunk {
    base: AtomicUsize,
    len: AtomicUsize,
    pinned: AtomicBool,
    pool: AtomicBool,
    ready: AtomicBool,
}

impl Chunk {
    const fn new() -> Self {
        Self {
            base: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            pinned: AtomicBool::new(false),
            pool: AtomicBool::new(false),
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
/// extent ever spans two pieces locked apart (or the pool and pageable memory).
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

/// jemalloc's own hooks, which map memory once the pool is spent.
static DEFAULT_HOOKS: AtomicPtr<jem::extent_hooks_t> = AtomicPtr::new(std::ptr::null_mut());

unsafe extern "C" fn hook_alloc(
    _hooks: *mut jem::extent_hooks_t,
    new_addr: *mut c_void,
    size: usize,
    alignment: usize,
    zero: *mut bool,
    commit: *mut bool,
    arena_ind: u32,
) -> *mut c_void {
    // An extent where jemalloc asks for it is never in the pool.
    if new_addr.is_null()
        && let Some(at) = carve(size, alignment)
    {
        // SAFETY: jemalloc's out-parameters. The pool's bytes are handed out
        // once, as the driver left them when it faulted them in: zeroed and
        // committed.
        unsafe {
            *zero = true;
            *commit = true;
        }
        return at as *mut c_void;
    }
    let defaults = DEFAULT_HOOKS.load(Ordering::Acquire);
    // SAFETY: the default table is jemalloc's static one, and its `alloc` is
    // called with the arguments jemalloc gave this one; committed memory is
    // asked for, as the hooks never commit later.
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
    SPILLED_BYTES.fetch_add(size as u64, Ordering::Relaxed);
    let k = NCHUNKS.fetch_add(1, Ordering::AcqRel);
    if k < MAX_CHUNKS {
        let chunk = &CHUNKS[k];
        chunk.base.store(ptr as usize, Ordering::Relaxed);
        chunk.len.store(size, Ordering::Relaxed);
        chunk.ready.store(true, Ordering::Release);
    }
    ptr
}

/// Kept: neither the pool nor the default hooks' extents go back to the
/// system while the process lives (jemalloc retains them and reuses them).
unsafe extern "C" fn hook_dalloc(
    _hooks: *mut jem::extent_hooks_t,
    _addr: *mut c_void,
    _size: usize,
    _committed: bool,
    _arena_ind: u32,
) -> bool {
    true
}

/// Only an arena's destruction, which nothing here asks for: the pool stays,
/// and an extent the default hooks made goes back through them.
unsafe extern "C" fn hook_destroy(
    _hooks: *mut jem::extent_hooks_t,
    addr: *mut c_void,
    size: usize,
    committed: bool,
    arena_ind: u32,
) {
    if chunk_of(addr as usize).is_some_and(|k| CHUNKS[k].pool.load(Ordering::Acquire)) {
        return;
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

    fn fake_lock(_ptr: *mut u8, len: usize) -> bool {
        LOCKED.fetch_add(len as u64, Ordering::Relaxed);
        true
    }

    /// A 2 GiB pool locked by a stand-in for the driver, in this test process.
    fn on() -> &'static Arenas {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            set_lock(fake_lock);
            assert_eq!(set_pool_gib_for_tests(2), 2);
            let locked = lock_pool().expect("the pool");
            assert_eq!((locked.pieces, locked.refused), (2, 0));
            assert_eq!(locked.locked, 2 << 30);
        });
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

    const PAGE: usize = 4096;

    #[test]
    fn the_pool_is_locked_once_and_only_when_asked() {
        let _ = on();
        assert_eq!(LOCKED.load(Ordering::Relaxed), 2 << 30);
        assert!(lock_pool().is_none(), "a second lock");
        assert!(!binds(1 << 30, false));
        assert!(!binds(MIN_BYTES - 1, true));
        assert!(binds(MIN_BYTES, true));
    }

    #[test]
    fn a_large_allocation_in_a_scope_comes_from_the_pool_and_one_outside_does_not() {
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
        give(inside, bytes);
        give(outside, bytes);
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
    fn a_freed_block_stays_in_the_pool_through_a_purge_and_serves_the_next() {
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
    fn an_extent_never_spans_two_pieces() {
        let _ = on();
        let base = POOL_BASE.load(Ordering::Relaxed);
        for size in [PAGE, 3 << 20, 64 << 20, 200 << 20] {
            let at = carve(size, PAGE).expect("room in the pool");
            assert_eq!(
                (at - base) / PIECE,
                (at + size - 1 - base) / PIECE,
                "{size}"
            );
            assert!(is_pinned(at as *const u8, size));
        }
        assert!(carve(PIECE + 1, PAGE).is_none(), "larger than a piece");
    }

    #[test]
    fn the_unused_pool_leaves_out_what_an_allocation_holds() {
        let _ = on();
        let len = POOL_LEN.load(Ordering::Relaxed) as u64;
        let bytes = 32 << 20;
        let taken = {
            let _scope = scope(bytes);
            take(bytes)
        };
        // From jemalloc's figures where it keeps them, else from what was
        // handed out; either way the allocation held is not unused. (Other
        // tests allocate beside this one, so only the bound is exact.)
        if let Some(held) = live_bytes() {
            assert!(held >= bytes as u64, "the arenas hold it: {held}");
        }
        assert!(unused_bytes() <= len - bytes as u64);
        give(taken, bytes);
    }

    #[test]
    fn the_netting_retires_with_the_prove() {
        let _ = on();
        let len = POOL_LEN.load(Ordering::Relaxed) as u64;
        assert!(arena_free_bytes() <= len);
        retire_netting();
        // Process-wide: any test reading these after this one reads 0, which
        // every bound they assert allows.
        assert_eq!(unused_bytes(), 0);
        assert_eq!(arena_free_bytes(), 0);
    }

    #[test]
    fn merges_stay_inside_one_piece() {
        let _ = on();
        let base = POOL_BASE.load(Ordering::Relaxed);
        assert!(may_merge(base, base + PIECE));
        assert!(!may_merge(base, base + PIECE + 1));
        assert!(!may_merge(base + PIECE - 1, base + PIECE + 1));
    }

    #[test]
    fn uploads_are_counted_and_the_pools_apart() {
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
        assert!(after.pending_most >= bytes as u64);
        give(locked, bytes);
        give(plain, bytes);
    }
}
