//! Bowers FFT Implementation
//!
//! Optimized FFT using the Bowers G network algorithm with:
//! - **LayerTwiddles**: Cache-friendly twiddle access (10-30% speedup on large inputs)
//! - **2-layer fusion**: Reduces memory traffic by keeping intermediates in registers
//!
//! # Quick Start
//!
//! ```ignore
//! use math::fft::cpu::bowers::{fft, ifft, batch_fft};
//!
//! // Single polynomial FFT
//! let mut coeffs = vec![F::from(1), F::from(2), F::from(3), F::from(4)];
//! fft::<F>(&mut coeffs)?;  // coeffs now contains evaluations
//!
//! // Inverse FFT (includes 1/n scaling)
//! ifft::<F>(&mut coeffs)?;  // coeffs restored to original
//!
//! // Batch FFT for multiple polynomials (same length)
//! let mut batch = vec![...];  // num_polys * poly_len elements
//! batch_fft::<F>(&mut batch, poly_len)?;
//! ```

#[cfg(feature = "alloc")]
use crate::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
#[cfg(feature = "alloc")]
use crate::fft::errors::FFTError;
#[cfg(feature = "alloc")]
use crate::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ============================================================================
// PUBLIC API - Simple, Complete Functions
// ============================================================================

/// Forward FFT: coefficients → evaluations in natural order.
///
/// This is the recommended entry point for single-polynomial FFT.
/// Handles twiddle computation and bit-reversal internally.
///
/// # Example
/// ```ignore
/// let mut coeffs = vec![F::one(), F::from(2), F::from(3), F::from(4)];
/// fft::<F>(&mut coeffs)?;
/// // coeffs now contains [P(w^0), P(w^1), P(w^2), P(w^3)]
/// ```
///
/// # Errors
/// - `FFTError::InputError` if length is not a power of two
/// - `FFTError::RootsOfUnityError` if field doesn't support this FFT size
#[cfg(feature = "alloc")]
pub fn fft<F: IsFFTField>(input: &mut [FieldElement<F>]) -> Result<(), FFTError> {
    fft_with_field::<F, F>(input)
}

/// Forward FFT with explicit subfield for twiddles.
///
/// Use this when computing FFT over an extension field E with twiddles from base field F.
#[cfg(feature = "alloc")]
pub fn fft_with_field<F, E>(input: &mut [FieldElement<E>]) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = input.len();
    if n <= 1 {
        return Ok(());
    }

    let order = n.trailing_zeros() as u64;
    let twiddles = LayerTwiddles::<F>::new(order)
        .ok_or(FFTError::RootOfUnityError(order))?;

    bowers_fft_core(input, &twiddles)?;
    in_place_bit_reverse_permute(input);
    Ok(())
}

/// Inverse FFT: evaluations → coefficients in natural order.
///
/// Includes the 1/n scaling factor for a complete inverse transform.
///
/// # Example
/// ```ignore
/// // After fft(&mut data)
/// ifft::<F>(&mut data)?;
/// // data is restored to original coefficients
/// ```
///
/// # Errors
/// - `FFTError::InputError` if length is not a power of two
/// - `FFTError::RootsOfUnityError` if field doesn't support this FFT size
#[cfg(feature = "alloc")]
pub fn ifft<F: IsFFTField>(input: &mut [FieldElement<F>]) -> Result<(), FFTError> {
    ifft_with_field::<F, F>(input)
}

/// Inverse FFT with explicit subfield for twiddles.
#[cfg(feature = "alloc")]
pub fn ifft_with_field<F, E>(input: &mut [FieldElement<E>]) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = input.len();
    if n <= 1 {
        return Ok(());
    }

    let order = n.trailing_zeros() as u64;
    let inv_twiddles = LayerTwiddles::<F>::new_inverse(order)
        .ok_or(FFTError::RootOfUnityError(order))?;

    // Bit-reverse input, apply inverse butterflies
    in_place_bit_reverse_permute(input);
    bowers_ifft_core(input, &inv_twiddles)?;

    // Scale by 1/n (subfield element multiplied with extension element)
    let n_inv = FieldElement::<F>::from(n as u64)
        .inv()
        .map_err(|_| FFTError::InputError(n))?;
    for val in input.iter_mut() {
        *val = &n_inv * &*val;
    }

    Ok(())
}

/// Batch FFT for multiple polynomials of the same length.
///
/// Polynomials are stored contiguously in `data`:
/// `[poly0[0], poly0[1], ..., poly0[len-1], poly1[0], ...]`
///
/// This is more efficient than calling `fft` repeatedly because
/// twiddle factors are computed once and reused.
///
/// # Example
/// ```ignore
/// // 4 polynomials, each of length 8
/// let mut data = vec![...];  // 32 elements total
/// batch_fft::<F>(&mut data, 8)?;
/// ```
///
/// # Errors
/// - `FFTError::InputError` if `poly_len` is not a power of two
/// - `FFTError::InputError` if `data.len()` is not divisible by `poly_len`
#[cfg(feature = "alloc")]
pub fn batch_fft<F: IsFFTField>(
    data: &mut [FieldElement<F>],
    poly_len: usize,
) -> Result<(), FFTError> {
    batch_fft_with_field::<F, F>(data, poly_len)
}

/// Batch FFT with explicit subfield for twiddles.
#[cfg(feature = "alloc")]
pub fn batch_fft_with_field<F, E>(
    data: &mut [FieldElement<E>],
    poly_len: usize,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    if poly_len <= 1 {
        return Ok(());
    }

    if !poly_len.is_power_of_two() {
        return Err(FFTError::InputError(poly_len));
    }

    if data.len() % poly_len != 0 {
        return Err(FFTError::InputError(data.len()));
    }

    let order = poly_len.trailing_zeros() as u64;
    let twiddles = LayerTwiddles::<F>::new(order)
        .ok_or(FFTError::RootOfUnityError(order))?;

    for chunk in data.chunks_mut(poly_len) {
        bowers_fft_core(chunk, &twiddles)?;
        in_place_bit_reverse_permute(chunk);
    }

    Ok(())
}

/// Batch IFFT for multiple polynomials of the same length.
#[cfg(feature = "alloc")]
pub fn batch_ifft<F: IsFFTField>(
    data: &mut [FieldElement<F>],
    poly_len: usize,
) -> Result<(), FFTError> {
    batch_ifft_with_field::<F, F>(data, poly_len)
}

/// Batch IFFT with explicit subfield for twiddles.
#[cfg(feature = "alloc")]
pub fn batch_ifft_with_field<F, E>(
    data: &mut [FieldElement<E>],
    poly_len: usize,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    if poly_len <= 1 {
        return Ok(());
    }

    if !poly_len.is_power_of_two() {
        return Err(FFTError::InputError(poly_len));
    }

    if data.len() % poly_len != 0 {
        return Err(FFTError::InputError(data.len()));
    }

    let order = poly_len.trailing_zeros() as u64;
    let inv_twiddles = LayerTwiddles::<F>::new_inverse(order)
        .ok_or(FFTError::RootOfUnityError(order))?;

    let n_inv = FieldElement::<F>::from(poly_len as u64)
        .inv()
        .map_err(|_| FFTError::InputError(poly_len))?;

    for chunk in data.chunks_mut(poly_len) {
        in_place_bit_reverse_permute(chunk);
        bowers_ifft_core(chunk, &inv_twiddles)?;
        for val in chunk.iter_mut() {
            *val = &n_inv * &*val;
        }
    }

    Ok(())
}

// ============================================================================
// PARALLEL VARIANTS
// ============================================================================

/// Parallel batch FFT - processes multiple polynomials concurrently.
///
/// Use this for large batches where parallelization overhead is amortized.
#[cfg(all(feature = "alloc", feature = "parallel"))]
pub fn batch_fft_parallel<F, E>(
    data: &mut [FieldElement<E>],
    poly_len: usize,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
    FieldElement<F>: Send + Sync,
    FieldElement<E>: Send + Sync,
{
    if poly_len <= 1 {
        return Ok(());
    }

    if !poly_len.is_power_of_two() {
        return Err(FFTError::InputError(poly_len));
    }

    if data.len() % poly_len != 0 {
        return Err(FFTError::InputError(data.len()));
    }

    let order = poly_len.trailing_zeros() as u64;
    let twiddles = LayerTwiddles::<F>::new(order)
        .ok_or(FFTError::RootOfUnityError(order))?;

    data.par_chunks_mut(poly_len).try_for_each(|chunk| {
        bowers_fft_core(chunk, &twiddles)?;
        in_place_bit_reverse_permute(chunk);
        Ok(())
    })
}

// ============================================================================
// LAYER TWIDDLES
// ============================================================================

/// Pre-computed twiddle factors organized by layer for cache-friendly access.
///
/// Standard FFT implementations access twiddles with strided patterns that cause
/// cache misses. LayerTwiddles reorganizes twiddles so each layer's values are
/// contiguous, achieving O(N) sequential memory access.
///
/// # Reusability
/// Compute once, reuse for multiple FFTs of the same size.
#[cfg(feature = "alloc")]
#[derive(Clone)]
pub struct LayerTwiddles<F: IsField> {
    layers: Vec<Vec<FieldElement<F>>>,
}

#[cfg(feature = "alloc")]
impl<F: IsFFTField> LayerTwiddles<F> {
    /// Create forward twiddles for FFT of size 2^order.
    pub fn new(order: u64) -> Option<Self> {
        let root = F::get_primitive_root_of_unity(order).ok()?;
        Self::from_root(order, root)
    }

    /// Create inverse twiddles for IFFT of size 2^order.
    pub fn new_inverse(order: u64) -> Option<Self> {
        let root = F::get_primitive_root_of_unity(order).ok()?;
        let inv_root = root.inv().ok()?;
        Self::from_root(order, inv_root)
    }

    /// Internal: create twiddles from a given root of unity.
    fn from_root(order: u64, root: FieldElement<F>) -> Option<Self> {
        // Overflow protection
        #[cfg(target_pointer_width = "64")]
        const MAX_ORDER: u64 = 63;
        #[cfg(target_pointer_width = "32")]
        const MAX_ORDER: u64 = 31;

        if order > MAX_ORDER {
            return None;
        }

        let n = 1usize << order;
        let mut layers = Vec::with_capacity(order as usize);

        for layer in 0..order as usize {
            let stride = 1usize << layer;
            let count = n >> (layer + 1);

            let w_stride = root.pow(stride as u64);
            let mut current = FieldElement::<F>::one();
            let mut layer_twiddles = Vec::with_capacity(count);

            for _ in 0..count {
                layer_twiddles.push(current.clone());
                current = &current * &w_stride;
            }

            layers.push(layer_twiddles);
        }

        Some(Self { layers })
    }

    #[inline(always)]
    fn get_layer(&self, layer: usize) -> &[FieldElement<F>] {
        &self.layers[layer]
    }

    /// Number of layers (= FFT order).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ============================================================================
// CORE FFT ALGORITHMS
// ============================================================================

/// Core Bowers FFT with 2-layer fusion.
///
/// Output is in bit-reversed order. Call `in_place_bit_reverse_permute` after.
#[cfg(feature = "alloc")]
fn bowers_fft_core<F, E>(
    input: &mut [FieldElement<E>],
    twiddles: &LayerTwiddles<F>,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = input.len();
    if !n.is_power_of_two() {
        return Err(FFTError::InputError(n));
    }

    if n <= 1 {
        return Ok(());
    }

    let log_n = n.trailing_zeros() as usize;
    let mut layer = 0;

    // Process pairs of layers with 2-layer fusion
    while layer + 1 < log_n {
        let block_size = n >> layer;

        if block_size >= 4 {
            let tw0 = twiddles.get_layer(layer);
            let tw1 = twiddles.get_layer(layer + 1);

            for block_start in (0..n).step_by(block_size) {
                fused_butterfly_block(
                    &mut input[block_start..block_start + block_size],
                    tw0,
                    tw1,
                );
            }
            layer += 2;
        } else {
            break;
        }
    }

    // Process remaining single layer (if odd number of layers)
    while layer < log_n {
        let block_size = n >> layer;
        let half = block_size >> 1;
        let tw = twiddles.get_layer(layer);

        for block_start in (0..n).step_by(block_size) {
            single_butterfly_block(&mut input[block_start..block_start + block_size], tw, half);
        }
        layer += 1;
    }

    Ok(())
}

/// Core Bowers IFFT (inverse butterfly structure).
///
/// Input should be bit-reversed. Output will be in natural order.
/// Does NOT include 1/n scaling.
#[cfg(feature = "alloc")]
fn bowers_ifft_core<F, E>(
    input: &mut [FieldElement<E>],
    twiddles: &LayerTwiddles<F>,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = input.len();
    if !n.is_power_of_two() {
        return Err(FFTError::InputError(n));
    }

    if n <= 1 {
        return Ok(());
    }

    let log_n = n.trailing_zeros() as usize;

    // Process layers in reverse order
    for layer in (0..log_n).rev() {
        let block_size = n >> layer;
        let half = block_size >> 1;
        let tw = twiddles.get_layer(layer);

        for block_start in (0..n).step_by(block_size) {
            for j in 0..half {
                let i0 = block_start + j;
                let i1 = i0 + half;
                let w = &tw[j];

                let bw = w * &input[i1];
                let sum = &input[i0] + &bw;
                let diff = &input[i0] - &bw;

                input[i0] = sum;
                input[i1] = diff;
            }
        }
    }

    Ok(())
}

/// 2-layer fused butterfly (DIF).
#[cfg(feature = "alloc")]
#[inline(always)]
fn fused_butterfly_block<F, E>(
    block: &mut [FieldElement<E>],
    tw0: &[FieldElement<F>],
    tw1: &[FieldElement<F>],
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let quarter = block.len() >> 2;

    for j in 0..quarter {
        let (i0, i1, i2, i3) = (j, j + quarter, j + 2 * quarter, j + 3 * quarter);
        let (w0, w1, w2) = (&tw0[j], &tw0[j + quarter], &tw1[j]);

        // Layer 0 butterflies
        let sum_02 = &block[i0] + &block[i2];
        let diff_02 = &block[i0] - &block[i2];
        let diff_02_w = w0 * &diff_02;

        let sum_13 = &block[i1] + &block[i3];
        let diff_13 = &block[i1] - &block[i3];
        let diff_13_w = w1 * &diff_13;

        // Layer 1 butterflies
        block[i0] = &sum_02 + &sum_13;
        block[i1] = w2 * &(&sum_02 - &sum_13);
        block[i2] = &diff_02_w + &diff_13_w;
        block[i3] = w2 * &(&diff_02_w - &diff_13_w);
    }
}

/// Single-layer butterfly (DIF).
#[cfg(feature = "alloc")]
#[inline(always)]
fn single_butterfly_block<F, E>(
    block: &mut [FieldElement<E>],
    tw: &[FieldElement<F>],
    half: usize,
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    for j in 0..half {
        let w = &tw[j];
        let sum = &block[j] + &block[j + half];
        let diff = &block[j] - &block[j + half];
        block[j] = sum;
        block[j + half] = w * &diff;
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
    use alloc::vec;

    type F = GoldilocksField;
    type FE = FieldElement<F>;

    // ===== Helper: Naive DFT for verification =====

    fn naive_dft(input: &[FE]) -> Vec<FE> {
        let n = input.len();
        if n == 0 {
            return Vec::new();
        }
        let root = F::get_primitive_root_of_unity(n.trailing_zeros() as u64).unwrap();
        let mut result = vec![FE::zero(); n];

        for (k, res) in result.iter_mut().enumerate() {
            for (j, inp) in input.iter().enumerate() {
                *res = &*res + &(inp * &root.pow((j * k) as u64));
            }
        }
        result
    }

    // ===== Correctness Tests =====

    #[test]
    fn test_fft_small_sizes() {
        for order in 0..=4u32 {
            let n = 1 << order;
            let input: Vec<FE> = (0..n).map(|i| FE::from(i as u64)).collect();
            let expected = naive_dft(&input);

            let mut result = input.clone();
            fft::<F>(&mut result).unwrap();

            assert_eq!(result, expected, "FFT failed for size {}", n);
        }
    }

    #[test]
    fn test_fft_medium_sizes() {
        for order in 5..=10u32 {
            let n = 1 << order;
            let input: Vec<FE> = (0..n).map(|i| FE::from(i as u64)).collect();
            let expected = naive_dft(&input);

            let mut result = input.clone();
            fft::<F>(&mut result).unwrap();

            assert_eq!(result, expected, "FFT failed for size {}", n);
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        for order in 1..=8u32 {
            let n = 1 << order;
            let original: Vec<FE> = (0..n).map(|i| FE::from(i as u64 + 1)).collect();

            let mut data = original.clone();
            fft::<F>(&mut data).unwrap();
            ifft::<F>(&mut data).unwrap();

            assert_eq!(data, original, "Roundtrip failed for size {}", n);
        }
    }

    #[test]
    fn test_batch_fft() {
        let poly_len = 8;
        let num_polys = 4;

        // Create test data
        let mut data: Vec<FE> = (0..poly_len * num_polys)
            .map(|i| FE::from(i as u64))
            .collect();

        // Compute expected results
        let expected: Vec<FE> = data
            .chunks(poly_len)
            .flat_map(|chunk| naive_dft(chunk))
            .collect();

        batch_fft::<F>(&mut data, poly_len).unwrap();

        assert_eq!(data, expected);
    }

    #[test]
    fn test_batch_fft_ifft_roundtrip() {
        let poly_len = 16;
        let num_polys = 3;

        let original: Vec<FE> = (0..poly_len * num_polys)
            .map(|i| FE::from(i as u64 + 1))
            .collect();

        let mut data = original.clone();
        batch_fft::<F>(&mut data, poly_len).unwrap();
        batch_ifft::<F>(&mut data, poly_len).unwrap();

        assert_eq!(data, original);
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_fft_size_one() {
        let mut data = vec![FE::from(42u64)];
        fft::<F>(&mut data).unwrap();
        assert_eq!(data, vec![FE::from(42u64)]);
    }

    #[test]
    fn test_fft_all_zeros() {
        let mut data = vec![FE::zero(); 64];
        let original = data.clone();
        fft::<F>(&mut data).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn test_fft_all_ones() {
        let n = 64;
        let mut data = vec![FE::one(); n];
        fft::<F>(&mut data).unwrap();

        // FFT of all ones: first element is n, rest are 0
        assert_eq!(data[0], FE::from(n as u64));
        for i in 1..n {
            assert_eq!(data[i], FE::zero(), "Non-zero at index {}", i);
        }
    }

    #[test]
    fn test_ifft_empty() {
        let mut data: Vec<FE> = vec![];
        ifft::<F>(&mut data).unwrap();
        assert!(data.is_empty());
    }

    // ===== Error Case Tests =====

    #[test]
    fn test_fft_non_power_of_two() {
        let mut data: Vec<FE> = (0..7).map(|i| FE::from(i as u64)).collect();
        let result = fft::<F>(&mut data);
        assert!(matches!(result, Err(FFTError::InputError(7))));
    }

    #[test]
    fn test_fft_non_power_of_two_various() {
        for bad_len in [3, 5, 6, 7, 9, 10, 12, 15, 17, 100] {
            let mut data: Vec<FE> = (0..bad_len).map(|i| FE::from(i as u64)).collect();
            let result = fft::<F>(&mut data);
            assert!(
                matches!(result, Err(FFTError::InputError(n)) if n == bad_len),
                "Expected InputError({}) but got {:?}",
                bad_len,
                result
            );
        }
    }

    #[test]
    fn test_batch_fft_non_power_of_two_poly_len() {
        let mut data: Vec<FE> = (0..21).map(|i| FE::from(i as u64)).collect();
        let result = batch_fft::<F>(&mut data, 7);
        assert!(matches!(result, Err(FFTError::InputError(7))));
    }

    #[test]
    fn test_batch_fft_misaligned_data() {
        let mut data: Vec<FE> = (0..10).map(|i| FE::from(i as u64)).collect();
        // 10 elements is not divisible by 4
        let result = batch_fft::<F>(&mut data, 4);
        assert!(matches!(result, Err(FFTError::InputError(10))));
    }

    #[test]
    fn test_batch_fft_empty() {
        let mut data: Vec<FE> = vec![];
        batch_fft::<F>(&mut data, 8).unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn test_batch_fft_poly_len_one() {
        let mut data: Vec<FE> = vec![FE::from(1u64), FE::from(2u64), FE::from(3u64)];
        let original = data.clone();
        batch_fft::<F>(&mut data, 1).unwrap();
        assert_eq!(data, original);
    }

    // ===== LayerTwiddles Tests =====

    #[test]
    fn test_layer_twiddles_structure() {
        let order = 4u64;
        let twiddles = LayerTwiddles::<F>::new(order).unwrap();

        assert_eq!(twiddles.num_layers(), 4);
        assert_eq!(twiddles.get_layer(0).len(), 8);  // n/2
        assert_eq!(twiddles.get_layer(1).len(), 4);  // n/4
        assert_eq!(twiddles.get_layer(2).len(), 2);  // n/8
        assert_eq!(twiddles.get_layer(3).len(), 1);  // n/16

        // First twiddle of each layer should be 1
        for layer in 0..4 {
            assert_eq!(twiddles.get_layer(layer)[0], FE::one());
        }
    }

    #[test]
    fn test_layer_twiddles_inverse() {
        let order = 4u64;
        let fwd = LayerTwiddles::<F>::new(order).unwrap();
        let inv = LayerTwiddles::<F>::new_inverse(order).unwrap();

        // w * w^(-1) = 1 for corresponding twiddles
        for layer in 0..order as usize {
            let fwd_tw = fwd.get_layer(layer);
            let inv_tw = inv.get_layer(layer);

            for j in 0..fwd_tw.len() {
                let product = &fwd_tw[j] * &inv_tw[j];
                assert_eq!(product, FE::one(), "Layer {} index {}", layer, j);
            }
        }
    }

    #[test]
    fn test_layer_twiddles_overflow_protection() {
        // Order 64 would overflow on 64-bit systems
        let result = LayerTwiddles::<F>::new(64);
        assert!(result.is_none());
    }
}
