//! Page-locked host slots an epoch's trace columns are written into, so their
//! upload to the card is a DMA with no staging copy (D-TRACE stage 1b).
//!
//! A copy from pageable memory is staged by the driver through its own pinned
//! buffers: 19 GB/s in the proof. From `cuMemHostAlloc` memory the same copies
//! go straight to the card at 57 GB/s. So the producer writes each epoch's
//! columns into one of these slots instead of into `Vec`s, and the committed
//! columns are views into it until the epoch's proof is done.
//!
//! A block of page-locked memory has exactly one owner at every instant:
//! * the pool's free list;
//! * one [`SlotLease`], writable and exclusive, while the producer writes it;
//! * one [`FrozenSlot`], read-only, from [`SlotLease::freeze`] until the last
//!   view of it is gone.
//!
//! Dropping a lease or a frozen slot hands its block back. Nothing here waits
//! for a block: [`SlotPool::try_lease`] answers at once, with a block or with
//! why not, and the caller builds that epoch's columns the old way.

use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use cudarc::driver::CudaContext;

use crate::Result;

/// The size of each of the process pool's blocks, in MiB (3072 by default):
/// the largest epoch of the benchmark block is 2.92 GB.
pub const SLOT_MB_ENV: &str = "LAMBDA_VM_WHIR_PINNED_SLOT_MB";

/// [`SLOT_MB_ENV`]'s default.
pub const DEFAULT_SLOT_MB: usize = 3072;

/// Blocks in the process pool: the epoch being proved and the one being
/// prepared, which is all the producer's hand-off keeps alive.
pub const PROCESS_SLOTS: usize = 2;

/// Bytes per block for [`SLOT_MB_ENV`]'s raw value: a whole number of MiB from
/// 1 up, anything else the default.
pub fn slot_bytes_setting(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&mb| mb > 0 && mb <= 1 << 20)
        .unwrap_or(DEFAULT_SLOT_MB)
        << 20
}

/// Why a lease was not given. Each is counted, and the epoch goes the old way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Miss {
    /// No block is free and not every block exists yet: the allocation has
    /// not run, has not finished, or failed.
    NotReady,
    /// Every block exists and is lent out.
    NoneFree,
    /// The request is bigger than a block.
    TooSmall,
}

/// Where a block's memory came from, which is how it is given back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    /// `cuMemHostAlloc`, flags 0: cacheable, not write-combined, so the
    /// producer's writes and a host fallback's reads run at normal speed.
    Pinned,
    /// The global allocator, for the tests of the ownership rules.
    #[cfg(test)]
    Heap,
}

/// One block of `elems` `u64`s.
struct Block {
    ptr: NonNull<u64>,
    elems: usize,
    /// Its place in allocation order, for the log.
    id: usize,
    source: Source,
}

// SAFETY: a block is memory nothing else points at. Whoever holds it (the free
// list, a lease, a frozen slot) is its only owner, and a frozen one is only
// read.
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

impl Block {
    /// A new pinned block of `elems` zeroed `u64`s. The context must be bound
    /// on this thread.
    fn pinned(elems: usize, id: usize) -> Result<Self> {
        // SAFETY: a plain allocation; flags 0 is the memory type the
        // microbench measured.
        let raw = unsafe { cudarc::driver::result::malloc_host(elems * 8, 0)? };
        let Some(ptr) = NonNull::new(raw.cast::<u64>()) else {
            return Err(cudarc::driver::DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            ));
        };
        // SAFETY: `elems` u64s were just allocated at `ptr`. Zeroed once, so a
        // lease never hands out bytes nobody wrote.
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, elems) };
        Ok(Self {
            ptr,
            elems,
            id,
            source: Source::Pinned,
        })
    }

    #[cfg(test)]
    fn heap(elems: usize, id: usize) -> Self {
        let layout = std::alloc::Layout::array::<u64>(elems.max(1)).expect("a test block");
        // SAFETY: a nonzero layout.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        Self {
            ptr: NonNull::new(raw.cast::<u64>()).expect("a test block"),
            elems,
            id,
            source: Source::Heap,
        }
    }

    /// Gives the memory back. `ctx` is bound first for a pinned block.
    ///
    /// # Safety
    /// No lease or frozen slot may point at the block.
    unsafe fn free(self, ctx: Option<&Arc<CudaContext>>) {
        match self.source {
            Source::Pinned => {
                if let Some(ctx) = ctx {
                    let _ = ctx.bind_to_thread();
                }
                // SAFETY: allocated by `cuMemHostAlloc` in `Block::pinned`,
                // freed once.
                unsafe {
                    let _ = cudarc::driver::result::free_host(self.ptr.as_ptr().cast());
                }
            }
            #[cfg(test)]
            Source::Heap => {
                let layout =
                    std::alloc::Layout::array::<u64>(self.elems.max(1)).expect("a test block");
                // SAFETY: allocated with this layout in `Block::heap`.
                unsafe { std::alloc::dealloc(self.ptr.as_ptr().cast(), layout) };
            }
        }
    }
}

struct FreeList {
    blocks: Vec<Block>,
    /// Blocks that exist, lent out or not.
    created: usize,
}

struct Inner {
    slots: usize,
    block_elems: usize,
    free: Mutex<FreeList>,
    /// One allocator at a time, so two fills cannot overshoot `slots`.
    filling: Mutex<()>,
    /// The context the pinned blocks were allocated in, bound again to free
    /// them.
    ctx: OnceLock<Arc<CudaContext>>,
    leased: AtomicU64,
    not_ready: AtomicU64,
    none_free: AtomicU64,
    too_small: AtomicU64,
}

impl Inner {
    fn list(&self) -> MutexGuard<'_, FreeList> {
        // The list is only pushed and popped under the lock, so a panic
        // elsewhere never leaves it half-changed.
        self.free.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn give_back(&self, block: Block) {
        self.list().blocks.push(block);
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Every lease and frozen slot holds the pool, so by now every block is
        // back in the list.
        let list = self.free.get_mut().unwrap_or_else(PoisonError::into_inner);
        for block in std::mem::take(&mut list.blocks) {
            // SAFETY: in the free list, so nothing points at it.
            unsafe { block.free(self.ctx.get()) };
        }
    }
}

/// A handful of page-locked blocks of one size, lent out one at a time.
///
/// Cloning gives another handle to the same pool.
#[derive(Clone)]
pub struct SlotPool(Arc<Inner>);

/// A pool's counters: blocks that exist and are free now, leases given, and
/// misses by kind, since the pool was made.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlotStats {
    pub slots: usize,
    pub created: usize,
    pub free: usize,
    pub leased: u64,
    pub not_ready: u64,
    pub none_free: u64,
    pub too_small: u64,
}

impl SlotPool {
    /// A pool of `slots` blocks of `block_bytes` bytes (whole `u64`s), none
    /// allocated until [`SlotPool::fill`].
    pub fn new(slots: usize, block_bytes: usize) -> Self {
        Self(Arc::new(Inner {
            slots,
            block_elems: block_bytes / 8,
            free: Mutex::new(FreeList {
                blocks: Vec::with_capacity(slots),
                created: 0,
            }),
            filling: Mutex::new(()),
            ctx: OnceLock::new(),
            leased: AtomicU64::new(0),
            not_ready: AtomicU64::new(0),
            none_free: AtomicU64::new(0),
            too_small: AtomicU64::new(0),
        }))
    }

    /// The process's pool: [`PROCESS_SLOTS`] blocks of [`SLOT_MB_ENV`] MiB,
    /// read once. Nothing is allocated until [`SlotPool::fill`].
    pub fn global() -> &'static SlotPool {
        static POOL: OnceLock<SlotPool> = OnceLock::new();
        POOL.get_or_init(|| {
            let bytes = slot_bytes_setting(std::env::var(SLOT_MB_ENV).ok().as_deref());
            SlotPool::new(PROCESS_SLOTS, bytes)
        })
    }

    pub fn slots(&self) -> usize {
        self.0.slots
    }

    /// The most `u64`s one lease can take.
    pub fn block_elems(&self) -> usize {
        self.0.block_elems
    }

    /// Allocates every block not yet allocated, one after another on this
    /// thread, each lendable as soon as it exists. Returns the seconds each
    /// took; on an error the blocks already made stay, and the rest are
    /// missing, so leases miss with [`Miss::NotReady`] once those are out.
    pub fn fill(&self) -> Result<Vec<f64>> {
        let _one_filler = self
            .0
            .filling
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let be = crate::device::backend()?;
        be.ctx.bind_to_thread()?;
        let _ = self.0.ctx.set(be.ctx.clone());
        let mut took = Vec::new();
        loop {
            let id = self.0.list().created;
            if id >= self.0.slots {
                return Ok(took);
            }
            let started = std::time::Instant::now();
            let block = Block::pinned(self.0.block_elems, id)?;
            took.push(started.elapsed().as_secs_f64());
            let mut list = self.0.list();
            list.blocks.push(block);
            list.created += 1;
        }
    }

    /// [`SlotPool::fill`] from the global allocator instead of the driver, for
    /// the tests of the ownership rules.
    #[cfg(test)]
    fn fill_from_heap(&self) {
        let mut list = self.0.list();
        while list.created < self.0.slots {
            let id = list.created;
            list.blocks.push(Block::heap(self.0.block_elems, id));
            list.created += 1;
        }
    }

    /// A free block for `elems` `u64`s, or why there is none. Never waits:
    /// the free list's lock is held for one pop.
    pub fn try_lease(&self, elems: usize) -> std::result::Result<SlotLease, Miss> {
        let miss = |counter: &AtomicU64, why: Miss| {
            counter.fetch_add(1, Ordering::Relaxed);
            Err(why)
        };
        if elems > self.0.block_elems {
            return miss(&self.0.too_small, Miss::TooSmall);
        }
        let mut list = self.0.list();
        match list.blocks.pop() {
            Some(block) => {
                drop(list);
                self.0.leased.fetch_add(1, Ordering::Relaxed);
                Ok(SlotLease {
                    block: Some(block),
                    elems,
                    pool: self.clone(),
                })
            }
            None if list.created < self.0.slots => miss(&self.0.not_ready, Miss::NotReady),
            None => miss(&self.0.none_free, Miss::NoneFree),
        }
    }

    pub fn stats(&self) -> SlotStats {
        let (created, free) = {
            let list = self.0.list();
            (list.created, list.blocks.len())
        };
        SlotStats {
            slots: self.0.slots,
            created,
            free,
            leased: self.0.leased.load(Ordering::Relaxed),
            not_ready: self.0.not_ready.load(Ordering::Relaxed),
            none_free: self.0.none_free.load(Ordering::Relaxed),
            too_small: self.0.too_small.load(Ordering::Relaxed),
        }
    }
}

/// A block lent out for writing: the only handle to it, so its slice is
/// exclusive. Dropped, the block goes back to the pool.
pub struct SlotLease {
    /// `Some` until [`SlotLease::freeze`] moves it on.
    block: Option<Block>,
    elems: usize,
    pool: SlotPool,
}

impl SlotLease {
    /// The block's place in allocation order.
    pub fn slot(&self) -> usize {
        self.block.as_ref().map_or(0, |block| block.id)
    }

    /// The `u64`s leased, as many as asked for.
    pub fn len(&self) -> usize {
        self.elems
    }

    pub fn is_empty(&self) -> bool {
        self.elems == 0
    }

    /// The leased `u64`s, to write. They hold whatever the block last held:
    /// zeros when new, a previous epoch's columns after.
    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        match &self.block {
            Some(block) => {
                debug_assert!(self.elems <= block.elems);
                // SAFETY: `elems ≤ block.elems` (checked at lease), the memory
                // is initialized (zeroed at allocation, then only written as
                // u64s), and this lease is the block's only owner, borrowed
                // mutably here.
                unsafe { std::slice::from_raw_parts_mut(block.ptr.as_ptr(), self.elems) }
            }
            None => &mut [],
        }
    }

    /// Ends the writing: the block becomes read-only for as long as the frozen
    /// slot lives, and nothing can write it again until the pool lends it anew.
    pub fn freeze(mut self) -> FrozenSlot {
        FrozenSlot {
            block: self.block.take(),
            elems: self.elems,
            pool: self.pool.clone(),
        }
    }
}

impl Drop for SlotLease {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            self.pool.0.give_back(block);
        }
    }
}

/// A written block, read-only from now on. Dropped, the block goes back to
/// the pool, so whatever reads it keeps this alive: a copy to the card that
/// could still be reading when it drops would read memory the producer may be
/// writing again.
pub struct FrozenSlot {
    block: Option<Block>,
    elems: usize,
    pool: SlotPool,
}

impl FrozenSlot {
    /// The block's place in allocation order.
    pub fn slot(&self) -> usize {
        self.block.as_ref().map_or(0, |block| block.id)
    }

    pub fn as_slice(&self) -> &[u64] {
        match &self.block {
            Some(block) => {
                debug_assert!(self.elems <= block.elems);
                // SAFETY: `elems ≤ block.elems`, the memory is initialized, and
                // no `&mut` into it can exist: the lease that wrote it was
                // consumed.
                unsafe { std::slice::from_raw_parts(block.ptr.as_ptr(), self.elems) }
            }
            None => &[],
        }
    }

    pub fn len(&self) -> usize {
        self.elems
    }

    pub fn is_empty(&self) -> bool {
        self.elems == 0
    }
}

impl Drop for FrozenSlot {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            self.pool.0.give_back(block);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(slots: usize, elems: usize) -> SlotPool {
        let pool = SlotPool::new(slots, elems * 8);
        pool.fill_from_heap();
        pool
    }

    #[test]
    fn the_slot_size_knob_takes_whole_mib_only() {
        assert_eq!(slot_bytes_setting(None), 3072 << 20);
        for raw in ["", "0", "-1", "abc", "1.5", "99999999999"] {
            assert_eq!(slot_bytes_setting(Some(raw)), 3072 << 20, "{raw:?}");
        }
        assert_eq!(slot_bytes_setting(Some("1")), 1 << 20);
        assert_eq!(slot_bytes_setting(Some(" 4096 ")), 4096 << 20);
    }

    /// Before any block exists a lease misses as not ready; with every block
    /// lent out, as none free; a request bigger than a block, as too small.
    /// Each miss is counted and nothing waits.
    #[test]
    fn a_lease_misses_at_once_and_says_why() {
        let empty = SlotPool::new(2, 64 * 8);
        assert_eq!(empty.try_lease(8).err(), Some(Miss::NotReady));
        assert_eq!(empty.stats().not_ready, 1);

        let pool = pool(2, 64);
        assert_eq!(pool.try_lease(65).err(), Some(Miss::TooSmall));
        let a = pool.try_lease(64).expect("a free block");
        let b = pool.try_lease(1).expect("the other free block");
        assert_ne!(a.slot(), b.slot());
        let started = std::time::Instant::now();
        assert_eq!(pool.try_lease(1).err(), Some(Miss::NoneFree));
        assert!(
            started.elapsed() < std::time::Duration::from_millis(1),
            "a miss waited"
        );
        assert_eq!(
            pool.stats(),
            SlotStats {
                slots: 2,
                created: 2,
                free: 0,
                leased: 2,
                not_ready: 0,
                none_free: 1,
                too_small: 1,
            }
        );
        drop((a, b));
        assert_eq!(pool.stats().free, 2);
    }

    /// A dropped lease and a dropped frozen slot each give their block back,
    /// and the next lease gets that very block with what was written in it.
    #[test]
    fn a_block_comes_back_when_its_holder_goes() {
        let pool = pool(1, 16);
        let mut lease = pool.try_lease(16).unwrap();
        assert_eq!(lease.as_mut_slice(), &[0u64; 16], "a new block is zeroed");
        lease.as_mut_slice()[3] = 7;
        drop(lease);
        assert_eq!(pool.stats().free, 1);

        let mut lease = pool.try_lease(8).unwrap();
        assert_eq!(lease.len(), 8);
        assert_eq!(lease.as_mut_slice()[3], 7, "the same block came back");
        lease
            .as_mut_slice()
            .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let frozen = lease.freeze();
        assert_eq!(pool.stats().free, 0, "freezing did not give the block back");
        assert_eq!(pool.try_lease(1).err(), Some(Miss::NoneFree));
        assert_eq!(frozen.as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(frozen.len(), 8);

        let shared = Arc::new(frozen);
        let reader = shared.clone();
        drop(shared);
        assert_eq!(pool.stats().free, 0, "a reader still holds the block");
        assert_eq!(reader.as_slice()[7], 8);
        drop(reader);
        assert_eq!(pool.stats().free, 1, "the last reader gave it back");
    }

    /// The two-slot recycling the producer relies on (D-TRACE-1B §2.1): epoch
    /// k's slot is back before k + 2 asks, so every lease after the first two
    /// succeeds, alternating between the blocks.
    #[test]
    fn two_slots_serve_a_pipeline_of_one_waiting_epoch() {
        let pool = pool(2, 32);
        let mut proving: Option<FrozenSlot> = None;
        let mut slots = Vec::new();
        for epoch in 0..10u64 {
            // The producer prepares `epoch` while `proving` is proved.
            let mut lease = pool.try_lease(32).expect("a free slot every epoch");
            lease.as_mut_slice().fill(epoch);
            let waiting = lease.freeze();
            // The prove returns, its columns drop, and the waiting epoch is
            // taken.
            drop(proving.take());
            slots.push(waiting.slot());
            assert!(waiting.as_slice().iter().all(|&v| v == epoch));
            proving = Some(waiting);
        }
        assert_eq!(slots, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
        let stats = pool.stats();
        assert_eq!((stats.leased, stats.none_free, stats.not_ready), (10, 0, 0));
    }

    /// Threads leasing, writing, freezing and dropping at once never share a
    /// block: each finds only its own pattern in what it holds, and every block
    /// is back at the end.
    #[test]
    fn leases_are_exclusive_under_contention() {
        let pool = pool(3, 1024);
        std::thread::scope(|s| {
            for t in 0..8u64 {
                let pool = pool.clone();
                s.spawn(move || {
                    for round in 0..2_000u64 {
                        let Ok(mut lease) = pool.try_lease(1024) else {
                            continue;
                        };
                        let mark = (t << 32) | round;
                        lease.as_mut_slice().fill(mark);
                        std::thread::yield_now();
                        let frozen = lease.freeze();
                        assert!(frozen.as_slice().iter().all(|&v| v == mark), "shared");
                    }
                });
            }
        });
        let stats = pool.stats();
        assert_eq!((stats.created, stats.free), (3, 3));
        assert_eq!(stats.leased + stats.none_free, 16_000);
        assert!(
            stats.leased > 0,
            "no lease was given, so nothing was checked"
        );
        assert_eq!(stats.not_ready, 0);
    }
}
