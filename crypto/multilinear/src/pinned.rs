//! Buffers the card will read, allocated where it reads them fastest.
//!
//! With the `cuda` feature and `LAMBDA_VM_PINNED_PRODUCER=1`, an allocation
//! made inside [`with_pinned`] comes from page-locked host memory
//! ([`math_cuda::pinned`]), which the card copies from by DMA while the
//! uploading thread goes on. Otherwise, and for small allocations, it is the
//! allocation `f` makes anyway. Where the bytes live changes no byte of them.

/// `f`, with an allocation of `bytes` it makes coming from page-locked memory
/// when that is on. `f` should allocate that one buffer and nothing else.
pub fn with_pinned<T>(bytes: usize, f: impl FnOnce() -> T) -> T {
    #[cfg(feature = "cuda")]
    let _scope = math_cuda::pinned::scope(bytes);
    #[cfg(not(feature = "cuda"))]
    let _ = bytes;
    f()
}

/// What the page-locked arenas did so far in this process, when they are on;
/// `None` without the `cuda` feature or with the knob off.
pub fn stats() -> Option<Stats> {
    #[cfg(feature = "cuda")]
    if math_cuda::pinned::enabled() {
        let s = math_cuda::pinned::stats();
        return Some(Stats {
            chunks: s.chunks,
            pinned_bytes: s.pinned_bytes,
            pageable_bytes: s.pageable_bytes,
            refused: s.refused,
            hook_seconds: s.hook_seconds,
            lock_seconds: s.lock_seconds,
            upload_bytes: s.upload_bytes,
            upload_pinned: s.upload_pinned,
        });
    }
    None
}

/// [`math_cuda::pinned::Stats`], in a crate built without the card too.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Extents page-locked, and their bytes.
    pub chunks: u64,
    pub pinned_bytes: u64,
    /// Extents left pageable, and how many the driver refused.
    pub pageable_bytes: u64,
    pub refused: u64,
    /// Seconds in the arenas' hooks (touching and locking), and locking alone.
    pub hook_seconds: f64,
    pub lock_seconds: f64,
    /// Bytes a group's upload sent, and how many came from page-locked memory.
    pub upload_bytes: u64,
    pub upload_pinned: u64,
}
