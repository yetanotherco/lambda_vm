//! CPU FFT backend using Bowers algorithm
//!
//! Wraps the existing `bowers.rs` implementation in the unified `Fft` trait.

use crate::fft::cpu::bowers;
use crate::fft::errors::FFTError;
use crate::fft::traits::Fft;
use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;

/// CPU FFT implementation using Bowers algorithm with 2-layer fusion
///
/// This is the default backend, always available. It provides good performance
/// on CPUs with the cache-friendly LayerTwiddles optimization.
#[derive(Debug, Clone, Default)]
pub struct CpuFft {
    // Currently stateless - twiddles are computed per-call
    // Could be extended to cache twiddles for repeated same-size FFTs
}

impl CpuFft {
    /// Create a new CPU FFT backend
    pub fn new() -> Self {
        Self {}
    }
}

impl Fft<GoldilocksField> for CpuFft {
    fn fft(&self, input: &mut [FieldElement<GoldilocksField>]) -> Result<(), FFTError> {
        bowers::fft::<GoldilocksField>(input)
    }

    fn ifft(&self, input: &mut [FieldElement<GoldilocksField>]) -> Result<(), FFTError> {
        bowers::ifft::<GoldilocksField>(input)
    }

    fn batch_fft(
        &self,
        data: &mut [FieldElement<GoldilocksField>],
        poly_len: usize,
    ) -> Result<(), FFTError> {
        bowers::batch_fft::<GoldilocksField>(data, poly_len)
    }

    fn batch_ifft(
        &self,
        data: &mut [FieldElement<GoldilocksField>],
        poly_len: usize,
    ) -> Result<(), FFTError> {
        bowers::batch_ifft::<GoldilocksField>(data, poly_len)
    }
}
