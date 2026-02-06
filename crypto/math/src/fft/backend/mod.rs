//! Unified FFT backend selection
//!
//! This module provides automatic backend selection and a unified API
//! for FFT operations on Goldilocks field. The best available backend
//! is selected at runtime based on hardware capabilities and feature flags.
//!
//! # Backend Priority
//!
//! 1. Metal GPU (if `metal` feature enabled and Metal device available)
//! 2. CUDA GPU (if `cuda` feature enabled and CUDA device available)
//! 3. CPU Bowers FFT (always available)
//!
//! # Example
//!
//! ```ignore
//! use math::fft::backend::goldilocks_backend;
//! use math::fft::traits::Fft;
//!
//! let backend = goldilocks_backend();
//! let mut data = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
//! backend.fft(&mut data)?;
//! ```

#[cfg(feature = "alloc")]
pub mod cpu;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(all(test, feature = "alloc"))]
mod tests;

#[cfg(feature = "alloc")]
pub use cpu::CpuFft;

#[cfg(feature = "metal")]
pub use metal::MetalFftBackend;

#[cfg(feature = "cuda")]
pub use cuda::CudaFft;

use crate::fft::traits::Fft;
use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;

/// Returns the best available FFT backend for Goldilocks field.
///
/// Backend selection priority:
/// 1. Metal GPU (on macOS with Metal support)
/// 2. CUDA GPU (on Linux/Windows with NVIDIA GPU)
/// 3. CPU Bowers FFT (fallback)
///
/// # Example
///
/// ```ignore
/// let backend = goldilocks_backend();
/// backend.fft(&mut coefficients)?;
/// ```
#[cfg(all(feature = "alloc", feature = "metal"))]
pub fn goldilocks_backend() -> impl Fft<GoldilocksField> {
    // Try Metal first, fall back to CPU if unavailable
    match MetalFftBackend::try_new() {
        Ok(metal) => BackendChoice::Metal(metal),
        Err(_) => BackendChoice::Cpu(CpuFft::new()),
    }
}

/// Returns the best available FFT backend (CUDA path, no Metal).
#[cfg(all(feature = "alloc", feature = "cuda", not(feature = "metal")))]
pub fn goldilocks_backend() -> impl Fft<GoldilocksField> {
    match CudaFft::try_new() {
        Ok(cuda) => BackendChoice::Cuda(cuda),
        Err(_) => BackendChoice::Cpu(CpuFft::new()),
    }
}

/// Returns the CPU FFT backend for Goldilocks field (no GPU features enabled).
#[cfg(all(feature = "alloc", not(feature = "metal"), not(feature = "cuda")))]
pub fn goldilocks_backend() -> impl Fft<GoldilocksField> {
    CpuFft::new()
}

/// Internal enum for backend selection (used when Metal is enabled)
#[cfg(all(feature = "alloc", feature = "metal"))]
enum BackendChoice {
    Metal(MetalFftBackend),
    Cpu(CpuFft),
}

#[cfg(all(feature = "alloc", feature = "metal"))]
impl Fft<GoldilocksField> for BackendChoice {
    fn fft(
        &self,
        input: &mut [crate::field::element::FieldElement<GoldilocksField>],
    ) -> Result<(), crate::fft::errors::FFTError> {
        match self {
            BackendChoice::Metal(m) => m.fft(input),
            BackendChoice::Cpu(c) => c.fft(input),
        }
    }

    fn ifft(
        &self,
        input: &mut [crate::field::element::FieldElement<GoldilocksField>],
    ) -> Result<(), crate::fft::errors::FFTError> {
        match self {
            BackendChoice::Metal(m) => m.ifft(input),
            BackendChoice::Cpu(c) => c.ifft(input),
        }
    }

    fn batch_fft(
        &self,
        data: &mut [crate::field::element::FieldElement<GoldilocksField>],
        poly_len: usize,
    ) -> Result<(), crate::fft::errors::FFTError> {
        match self {
            BackendChoice::Metal(m) => m.batch_fft(data, poly_len),
            BackendChoice::Cpu(c) => c.batch_fft(data, poly_len),
        }
    }

    fn batch_ifft(
        &self,
        data: &mut [crate::field::element::FieldElement<GoldilocksField>],
        poly_len: usize,
    ) -> Result<(), crate::fft::errors::FFTError> {
        match self {
            BackendChoice::Metal(m) => m.batch_ifft(data, poly_len),
            BackendChoice::Cpu(c) => c.batch_ifft(data, poly_len),
        }
    }
}

/// Internal enum for backend selection (CUDA path, no Metal)
#[cfg(all(feature = "alloc", feature = "cuda", not(feature = "metal")))]
enum BackendChoice {
    Cuda(CudaFft),
    Cpu(CpuFft),
}

#[cfg(all(feature = "alloc", feature = "cuda", not(feature = "metal")))]
impl Fft<GoldilocksField> for BackendChoice {
    fn fft(
        &self,
        input: &mut [crate::field::element::FieldElement<GoldilocksField>],
    ) -> Result<(), crate::fft::errors::FFTError> {
        match self {
            BackendChoice::Cuda(g) => g.fft(input),
            BackendChoice::Cpu(c) => c.fft(input),
        }
    }

    fn ifft(
        &self,
        input: &mut [crate::field::element::FieldElement<GoldilocksField>],
    ) -> Result<(), crate::fft::errors::FFTError> {
        match self {
            BackendChoice::Cuda(g) => g.ifft(input),
            BackendChoice::Cpu(c) => c.ifft(input),
        }
    }

    fn batch_fft(
        &self,
        data: &mut [crate::field::element::FieldElement<GoldilocksField>],
        poly_len: usize,
    ) -> Result<(), crate::fft::errors::FFTError> {
        match self {
            BackendChoice::Cuda(g) => g.batch_fft(data, poly_len),
            BackendChoice::Cpu(c) => c.batch_fft(data, poly_len),
        }
    }

    fn batch_ifft(
        &self,
        data: &mut [crate::field::element::FieldElement<GoldilocksField>],
        poly_len: usize,
    ) -> Result<(), crate::fft::errors::FFTError> {
        match self {
            BackendChoice::Cuda(g) => g.batch_ifft(data, poly_len),
            BackendChoice::Cpu(c) => c.batch_ifft(data, poly_len),
        }
    }
}
