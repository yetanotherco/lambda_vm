//! CUDA GPU FFT backend (skeleton)
//!
//! Provides the `Fft` trait implementation for CUDA GPU acceleration.
//! This is a skeleton that falls back to the CPU backend; wire in `cudarc`
//! calls to enable actual GPU execution.

use crate::fft::errors::FFTError;
use crate::fft::traits::Fft;
use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;

use super::cpu::CpuFft;

/// CUDA GPU FFT backend for Goldilocks field (skeleton)
///
/// Currently delegates all operations to the CPU backend.
/// Replace individual methods with `cudarc` kernel launches to enable
/// GPU acceleration.
pub struct CudaFft {
    /// CPU fallback used until CUDA kernels are implemented
    cpu_fallback: CpuFft,
}

impl CudaFft {
    /// Try to create a new CUDA FFT backend
    ///
    /// Returns an error if no CUDA device is available.
    pub fn try_new() -> Result<Self, CudaFftError> {
        // TODO: Initialize cudarc device and verify CUDA availability
        //   let device = cudarc::driver::CudaDevice::new(0)
        //       .map_err(|e| CudaFftError::DeviceNotAvailable(e.to_string()))?;
        Ok(Self {
            cpu_fallback: CpuFft::new(),
        })
    }
}

impl Fft<GoldilocksField> for CudaFft {
    fn fft(&self, input: &mut [FieldElement<GoldilocksField>]) -> Result<(), FFTError> {
        // TODO: Launch CUDA NTT kernel via cudarc
        //   1. Copy input to device buffer
        //   2. Launch forward NTT kernel
        //   3. Copy results back
        self.cpu_fallback.fft(input)
    }

    fn ifft(&self, input: &mut [FieldElement<GoldilocksField>]) -> Result<(), FFTError> {
        // TODO: Launch CUDA inverse NTT kernel via cudarc
        self.cpu_fallback.ifft(input)
    }

    fn batch_fft(
        &self,
        data: &mut [FieldElement<GoldilocksField>],
        poly_len: usize,
    ) -> Result<(), FFTError> {
        // TODO: Batch kernel launch — one NTT per polynomial, sharing twiddles
        self.cpu_fallback.batch_fft(data, poly_len)
    }

    fn batch_ifft(
        &self,
        data: &mut [FieldElement<GoldilocksField>],
        poly_len: usize,
    ) -> Result<(), FFTError> {
        // TODO: Batch inverse NTT kernel launch
        self.cpu_fallback.batch_ifft(data, poly_len)
    }
}

/// Errors specific to CUDA backend initialization
#[derive(Debug)]
pub enum CudaFftError {
    DeviceNotAvailable(String),
}

impl core::fmt::Display for CudaFftError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CudaFftError::DeviceNotAvailable(msg) => {
                write!(f, "CUDA device not available: {}", msg)
            }
        }
    }
}

impl std::error::Error for CudaFftError {}
