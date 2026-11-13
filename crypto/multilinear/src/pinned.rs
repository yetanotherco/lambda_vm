//! Buffers the card will read, allocated where it reads them fastest.
//!
//! With the `cuda` feature, `LAMBDA_VM_PINNED_POOL_GIB=<n>` and the pool
//! locked ([`lock_pool`], once, before any prove), an allocation made inside
//! [`with_pinned`] comes from page-locked host memory ([`math_cuda::pinned`]),
//! which the card copies from by DMA while the uploading thread goes on.
//! Otherwise, and for small allocations, it is the allocation `f` makes anyway.
//! Where the bytes live changes no byte of them.

/// `f`, with an allocation of `bytes` it makes coming from the page-locked
/// pool when that is on. `f` should allocate that one buffer and nothing else.
pub fn with_pinned<T>(bytes: usize, f: impl FnOnce() -> T) -> T {
    #[cfg(feature = "cuda")]
    let _scope = math_cuda::pinned::scope(bytes);
    #[cfg(not(feature = "cuda"))]
    let _ = bytes;
    f()
}

/// A list in the page-locked pool gives its capacity past its length back,
/// once nothing grows it: pool it would hold and never write. In place, so it
/// stays page-locked; a list outside the pool is left as it is. The bytes
/// given back; 0 without the card.
pub fn shrink<T>(list: &mut Vec<T>) -> u64 {
    #[cfg(feature = "cuda")]
    return math_cuda::pinned::shrink(list);
    #[cfg(not(feature = "cuda"))]
    {
        let _ = list;
        0
    }
}

/// Whether `bytes` from `ptr` lie in the page-locked pool: memory whose pages
/// must not be handed back to the OS while the driver holds them locked.
pub fn is_pinned(ptr: *const u8, bytes: usize) -> bool {
    #[cfg(feature = "cuda")]
    return math_cuda::pinned::is_pinned(ptr, bytes);
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ptr, bytes);
        false
    }
}

/// The page-locked pool's bytes not handed out yet: resident, and spoken for
/// by the allocations the block will make. A decision that forecasts the
/// host's growth takes them off its reading. 0 without a pool.
pub fn unused_pool_bytes() -> u64 {
    #[cfg(feature = "cuda")]
    return math_cuda::pinned::unused_bytes();
    #[cfg(not(feature = "cuda"))]
    0
}

/// What the page-locked pool's arenas hold free: in the allocator's resident
/// figure, returned by no purge. 0 without a pool.
pub fn arena_free_bytes() -> u64 {
    #[cfg(feature = "cuda")]
    return math_cuda::pinned::arena_free_bytes();
    #[cfg(not(feature = "cuda"))]
    0
}

/// The block's prove is over: from now on the host's readings count the
/// pool's free bytes as the memory they are.
pub fn retire_netting() {
    #[cfg(feature = "cuda")]
    math_cuda::pinned::retire_netting();
}

/// What locking the pool took.
#[derive(Clone, Copy, Debug, Default)]
pub struct PoolLocked {
    /// Bytes asked for, and bytes the driver locked.
    pub bytes: u64,
    pub locked: u64,
    pub pieces: usize,
    /// Pieces the driver refused (left pageable).
    pub refused: usize,
    pub seconds: f64,
}

/// Locks the pool once, when `LAMBDA_VM_PINNED_POOL_GIB` asks for one; `None`
/// otherwise. Before any prove: every other thread's driver calls wait while
/// it runs.
pub fn lock_pool() -> Option<PoolLocked> {
    #[cfg(feature = "cuda")]
    return math_cuda::pinned::lock_pool().map(|p| PoolLocked {
        bytes: p.bytes,
        locked: p.locked,
        pieces: p.pieces,
        refused: p.refused,
        seconds: p.seconds,
    });
    #[cfg(not(feature = "cuda"))]
    None
}

/// What the pool did so far in this process, when it is asked for; `None`
/// without the `cuda` feature or with the knob off.
pub fn stats() -> Option<Stats> {
    #[cfg(feature = "cuda")]
    if math_cuda::pinned::enabled() {
        let s = math_cuda::pinned::stats();
        return Some(Stats {
            pool_bytes: s.pool_bytes,
            locked_bytes: s.locked_bytes,
            lock_seconds: s.lock_seconds,
            carved_bytes: s.carved_bytes,
            spilled_bytes: s.spilled_bytes,
            upload_bytes: s.upload_bytes,
            upload_pinned: s.upload_pinned,
            pending_most: s.pending_most,
            held_bytes: s.held_bytes,
            requested_bytes: s.requested_bytes,
            usable_bytes: s.usable_bytes,
            shrinks: s.shrinks,
            shrunk_bytes: s.shrunk_bytes,
            shrink_moved: s.shrink_moved,
        });
    }
    None
}

/// [`math_cuda::pinned::Stats`], in a crate built without the card too.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// The pool's bytes, and those the driver locked, and the seconds it took.
    pub pool_bytes: u64,
    pub locked_bytes: u64,
    pub lock_seconds: f64,
    /// Bytes handed out from the pool, and pageable bytes handed out once it
    /// was spent.
    pub carved_bytes: u64,
    pub spilled_bytes: u64,
    /// Bytes a group's upload sent, and how many came from the pool.
    pub upload_bytes: u64,
    pub upload_pinned: u64,
    /// The most bytes allocated for the card and not yet uploaded at once.
    pub pending_most: u64,
    /// What allocations hold of the pool now; `None` where the allocator keeps
    /// no figures.
    pub held_bytes: Option<u64>,
    /// Bytes the producers asked of the pool, and the size classes they were
    /// given (lists at their lengths once shrunk).
    pub requested_bytes: u64,
    pub usable_bytes: u64,
    /// Lists that gave their untouched tail back ([`shrink`]), the bytes, and
    /// how many the allocator moved out of the pool instead.
    pub shrinks: u64,
    pub shrunk_bytes: u64,
    pub shrink_moved: u64,
}

/// A buffer page-locked where it lies ([`register_in_place`]); undo it with
/// [`unregister_in_place`] before the buffer is freed or moved.
#[derive(Debug, Default)]
#[must_use = "undo it with `unregister_in_place` before the buffer is freed"]
pub struct InPlace {
    #[cfg(feature = "cuda")]
    inner: math_cuda::registered::Registered,
    /// Bytes locked, pieces the driver refused, and the seconds it took.
    pub bytes: u64,
    pub refused: usize,
    pub seconds: f64,
}

/// Page-locks `bytes` from `ptr` in place, a piece at a time, so an upload
/// from them goes to the card as direct DMAs instead of through the staging
/// buffers ([`math_cuda::registered`]). The bytes uploaded do not change.
/// Without the `cuda` feature or a card, nothing is locked. Every other
/// thread's driver calls wait while a piece is locked.
///
/// The caller keeps the buffer alive and unmoved until it has handed the
/// result to [`unregister_in_place`], and registers no buffer twice.
pub fn register_in_place(ptr: *const u8, bytes: usize) -> InPlace {
    #[cfg(feature = "cuda")]
    {
        let inner = math_cuda::registered::register(ptr, bytes);
        let (bytes, refused, seconds) = (inner.bytes, inner.refused, inner.seconds);
        InPlace {
            inner,
            bytes,
            refused,
            seconds,
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ptr, bytes);
        InPlace::default()
    }
}

/// Undoes [`register_in_place`]; call it once no upload from the buffer is in
/// flight (every upload that reads a registered buffer returns after its DMA).
pub fn unregister_in_place(r: InPlace) {
    #[cfg(feature = "cuda")]
    math_cuda::registered::unregister(r.inner);
    #[cfg(not(feature = "cuda"))]
    let _ = r;
}

/// The in-place registrations' counters ([`math_cuda::registered::Stats`]);
/// all zero without the `cuda` feature.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct InPlaceStats {
    pub registered_bytes: u64,
    pub register_seconds: f64,
    pub refused_pieces: u64,
    pub unregister_seconds: f64,
    /// Bytes uploaded as direct DMAs from registered buffers, and the uploads.
    pub direct_bytes: u64,
    pub direct_calls: u64,
    /// Pieces registered now.
    pub pieces_now: usize,
}

pub fn in_place_stats() -> InPlaceStats {
    #[cfg(feature = "cuda")]
    {
        let s = math_cuda::registered::stats();
        InPlaceStats {
            registered_bytes: s.registered_bytes,
            register_seconds: s.register_seconds,
            refused_pieces: s.refused_pieces,
            unregister_seconds: s.unregister_seconds,
            direct_bytes: s.direct_bytes,
            direct_calls: s.direct_calls,
            pieces_now: s.pieces_now,
        }
    }
    #[cfg(not(feature = "cuda"))]
    InPlaceStats::default()
}

/// Zeroes the in-place registrations' counters (not the registrations).
pub fn reset_in_place_stats() {
    #[cfg(feature = "cuda")]
    math_cuda::registered::reset_stats();
}
