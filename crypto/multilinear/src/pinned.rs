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
}
