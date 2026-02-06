pub mod cpu;
pub mod errors;
#[cfg(feature = "alloc")]
pub mod polynomial;
#[cfg(feature = "alloc")]
pub mod traits;
#[cfg(feature = "alloc")]
pub mod backend;

#[cfg(all(test, feature = "alloc"))]
pub(crate) mod test_helpers;

// Re-exports for convenience
#[cfg(feature = "alloc")]
pub use traits::Fft;
#[cfg(feature = "alloc")]
pub use backend::goldilocks_backend;
#[cfg(feature = "alloc")]
pub use backend::CpuFft;
#[cfg(feature = "metal")]
pub use backend::MetalFftBackend;
#[cfg(feature = "cuda")]
pub use backend::CudaFft;
