//! Unified FFT trait for different backends (CPU, Metal GPU)
//!
//! This module provides a common trait that abstracts over different FFT
//! implementations, allowing transparent switching between CPU and GPU backends.

use crate::fft::errors::FFTError;
use crate::field::element::FieldElement;
use crate::field::traits::IsFFTField;

/// Unified FFT interface for different backends
///
/// Implementations provide in-place FFT operations for field elements.
/// The trait is designed for Goldilocks field but can be extended to other
/// FFT-friendly fields.
///
/// # Example
///
/// ```ignore
/// use math::fft::backend::goldilocks_backend;
/// use math::fft::traits::Fft;
///
/// let backend = goldilocks_backend();
/// let mut data = vec![F::from(1), F::from(2), F::from(3), F::from(4)];
/// backend.fft(&mut data)?;
/// ```
pub trait Fft<F: IsFFTField> {
    /// Forward FFT: coefficients → evaluations in natural order
    ///
    /// # Errors
    ///
    /// - `FFTError::InputError` if length is not a power of two
    /// - `FFTError::RootOfUnityError` if field doesn't support this FFT size
    fn fft(&self, input: &mut [FieldElement<F>]) -> Result<(), FFTError>;

    /// Inverse FFT: evaluations → coefficients in natural order
    ///
    /// Includes the 1/n scaling factor for a complete inverse transform.
    ///
    /// # Errors
    ///
    /// - `FFTError::InputError` if length is not a power of two
    /// - `FFTError::RootOfUnityError` if field doesn't support this FFT size
    fn ifft(&self, input: &mut [FieldElement<F>]) -> Result<(), FFTError>;

    /// Batch FFT for multiple polynomials of the same length
    ///
    /// Polynomials are stored contiguously: `[poly0[0..len], poly1[0..len], ...]`
    /// This is more efficient than calling `fft` repeatedly because twiddle
    /// factors are computed once and reused.
    ///
    /// # Errors
    ///
    /// - `FFTError::InputError` if `poly_len` is not a power of two
    /// - `FFTError::InputError` if `data.len()` is not divisible by `poly_len`
    fn batch_fft(&self, data: &mut [FieldElement<F>], poly_len: usize) -> Result<(), FFTError>;

    /// Batch IFFT for multiple polynomials of the same length
    ///
    /// Includes the 1/n scaling factor for each polynomial.
    ///
    /// # Errors
    ///
    /// - `FFTError::InputError` if `poly_len` is not a power of two
    /// - `FFTError::InputError` if `data.len()` is not divisible by `poly_len`
    fn batch_ifft(&self, data: &mut [FieldElement<F>], poly_len: usize) -> Result<(), FFTError>;
}
