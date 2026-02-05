//! Metal GPU FFT backend
//!
//! Provides the `Fft` trait implementation using Metal GPU acceleration.
//! Uses zero-copy conversion between `FieldElement<GoldilocksField>` and `u64`.

use crate::fft::errors::FFTError;
use crate::fft::traits::Fft;
use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
use crate::gpu::metal::{MetalError, MetalFft};

/// Metal GPU FFT backend for Goldilocks field
///
/// Uses Apple's Metal framework to accelerate FFT operations on GPU.
/// Falls back gracefully if Metal is unavailable (use `try_new()` to check).
pub struct MetalFftBackend {
    inner: MetalFft,
}

impl MetalFftBackend {
    /// Try to create a new Metal FFT backend
    ///
    /// Returns an error if Metal is not available (no GPU or not on macOS).
    pub fn try_new() -> Result<Self, MetalError> {
        Ok(Self {
            inner: MetalFft::new()?,
        })
    }
}

impl Fft<GoldilocksField> for MetalFftBackend {
    fn fft(&self, input: &mut [FieldElement<GoldilocksField>]) -> Result<(), FFTError> {
        // Zero-copy conversion: FieldElement<GoldilocksField> has the same
        // memory layout as u64 (just a newtype wrapper)
        let data = as_u64_slice_mut(input);
        self.inner.fft_natural_order(data)?;
        Ok(())
    }

    fn ifft(&self, input: &mut [FieldElement<GoldilocksField>]) -> Result<(), FFTError> {
        let n = input.len();
        if n <= 1 {
            return Ok(());
        }

        let data = as_u64_slice_mut(input);
        self.inner.ifft_natural_order(data)?;

        // Apply 1/n scaling (Metal IFFT doesn't include this)
        let n_inv = FieldElement::<GoldilocksField>::from(n as u64)
            .inv()
            .map_err(|_| FFTError::InputError(n))?;

        for val in input.iter_mut() {
            *val = &n_inv * &*val;
        }

        Ok(())
    }

    fn batch_fft(
        &self,
        data: &mut [FieldElement<GoldilocksField>],
        poly_len: usize,
    ) -> Result<(), FFTError> {
        if poly_len <= 1 || data.is_empty() {
            return Ok(());
        }

        let num_polys = data.len() / poly_len;
        if data.len() % poly_len != 0 {
            return Err(FFTError::InputError(data.len()));
        }

        let raw_data = as_u64_slice_mut(data);
        self.inner.batch_fft(raw_data, poly_len, num_polys)?;

        // Apply bit-reversal to each polynomial
        for chunk in raw_data.chunks_mut(poly_len) {
            self.inner.bitrev_permutation_inplace(chunk)?;
        }

        Ok(())
    }

    fn batch_ifft(
        &self,
        data: &mut [FieldElement<GoldilocksField>],
        poly_len: usize,
    ) -> Result<(), FFTError> {
        if poly_len <= 1 || data.is_empty() {
            return Ok(());
        }

        if data.len() % poly_len != 0 {
            return Err(FFTError::InputError(data.len()));
        }

        // Process each polynomial individually since Metal doesn't have batch_ifft
        let n_inv = FieldElement::<GoldilocksField>::from(poly_len as u64)
            .inv()
            .map_err(|_| FFTError::InputError(poly_len))?;

        for chunk in data.chunks_mut(poly_len) {
            let raw_chunk = as_u64_slice_mut(chunk);
            self.inner.ifft_natural_order(raw_chunk)?;
        }

        // Scale all elements by 1/n
        for val in data.iter_mut() {
            *val = &n_inv * &*val;
        }

        Ok(())
    }
}

/// Zero-copy conversion from `&mut [FieldElement<GoldilocksField>]` to `&mut [u64]`
///
/// # Safety
///
/// This is safe because:
/// - `FieldElement<GoldilocksField>` is a transparent newtype over `u64`
/// - Both have the same size and alignment
/// - The conversion is purely a reinterpretation of memory
pub(crate) fn as_u64_slice_mut(data: &mut [FieldElement<GoldilocksField>]) -> &mut [u64] {
    // Compile-time size check
    const _: () = assert!(
        core::mem::size_of::<FieldElement<GoldilocksField>>() == core::mem::size_of::<u64>()
    );
    const _: () = assert!(
        core::mem::align_of::<FieldElement<GoldilocksField>>() == core::mem::align_of::<u64>()
    );

    // SAFETY: FieldElement<GoldilocksField> is a #[repr(transparent)] wrapper
    // around u64, so the memory layout is identical
    unsafe { core::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u64, data.len()) }
}
