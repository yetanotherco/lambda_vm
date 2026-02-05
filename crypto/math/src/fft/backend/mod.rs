//! Unified FFT backend selection
//!
//! This module provides automatic backend selection and a unified API
//! for FFT operations on Goldilocks field. The best available backend
//! is selected at runtime based on hardware capabilities and feature flags.
//!
//! # Backend Priority
//!
//! 1. Metal GPU (if `metal` feature enabled and Metal device available)
//! 2. CPU Bowers FFT (always available)
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

#[cfg(feature = "alloc")]
pub use cpu::CpuFft;

#[cfg(feature = "metal")]
pub use metal::MetalFftBackend;

use crate::fft::traits::Fft;
use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;

/// Returns the best available FFT backend for Goldilocks field.
///
/// Backend selection priority:
/// 1. Metal GPU (on macOS with Metal support)
/// 2. CPU Bowers FFT (fallback)
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

/// Returns the CPU FFT backend for Goldilocks field (when Metal not enabled).
#[cfg(all(feature = "alloc", not(feature = "metal")))]
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

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::field::element::FieldElement;

    type FE = FieldElement<GoldilocksField>;

    #[test]
    fn test_goldilocks_backend_fft_roundtrip() {
        let backend = goldilocks_backend();
        let original: alloc::vec::Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();

        let mut data = original.clone();
        backend.fft(&mut data).expect("FFT failed");
        backend.ifft(&mut data).expect("IFFT failed");

        assert_eq!(data, original, "FFT/IFFT roundtrip failed");
    }

    #[test]
    fn test_goldilocks_backend_batch_fft() {
        let backend = goldilocks_backend();
        let poly_len = 4;
        let num_polys = 2;
        let original: alloc::vec::Vec<FE> = (1..=(poly_len * num_polys) as u64)
            .map(FE::from)
            .collect();

        let mut data = original.clone();
        backend
            .batch_fft(&mut data, poly_len)
            .expect("Batch FFT failed");
        backend
            .batch_ifft(&mut data, poly_len)
            .expect("Batch IFFT failed");

        assert_eq!(data, original, "Batch FFT/IFFT roundtrip failed");
    }
}
