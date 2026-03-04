//! Bowers FFT Implementation with Structure of Arrays (SoA) optimization
//!
//! This module implements the Bowers G network FFT algorithm, which provides
//! improved twiddle factor access patterns compared to the standard Cooley-Tukey FFT.
//!
//! # Key optimizations
//!
//! - **Bowers G network**: Improved memory access pattern for twiddle factors
//! - **LayerTwiddles**: Pre-computed twiddles per layer for O(N) sequential access
//!   instead of O(N log N) strided access
//! - **Multi-layer butterfly fusion**: Process 2 layers at once to keep intermediate
//!   values in registers and reduce memory traffic
//! - **Internal parallelization**: Uses rayon to parallelize across blocks when
//!   there are enough blocks (>= 64) to amortize threading overhead
//!
//! # Usage
//!
//! ```ignore
//! use math::fft::cpu::bowers_fft::{LayerTwiddles, bowers_fft_opt_fused};
//!
//! let order = 10u64; // FFT size = 2^10 = 1024
//! let layer_twiddles = LayerTwiddles::<F>::new(order).unwrap();
//!
//! let mut data = vec![...]; // your polynomial coefficients
//! bowers_fft_opt_fused(&mut data, &layer_twiddles);
//! in_place_bit_reverse_permute(&mut data);
//! ```
//!
//! Based on Plonky3's implementation and academic literature on FFT optimization.

#[cfg(feature = "alloc")]
use crate::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
#[cfg(feature = "alloc")]
use crate::fft::errors::FFTError;
#[cfg(feature = "alloc")]
use crate::field::{
    element::FieldElement,
    packed::{HasPacking, PackedField},
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Maximum supported FFT order to prevent integer overflow.
/// With order 63, n = 2^63 which is the largest power of 2 that fits in usize on 64-bit.
/// For 32-bit systems, max order is 31.
#[cfg(all(feature = "alloc", target_pointer_width = "64"))]
const MAX_FFT_ORDER: u64 = 63;
#[cfg(all(feature = "alloc", target_pointer_width = "32"))]
const MAX_FFT_ORDER: u64 = 31;

// =====================================================
// STRUCTURE OF ARRAYS (SoA) FFT
// =====================================================

/// Matrix representation for batch FFT with Structure of Arrays layout
///
/// SoA layout stores multiple polynomials contiguously:
/// ```text
/// [poly0[0], poly0[1], ..., poly0[n-1], poly1[0], poly1[1], ..., poly1[n-1], ...]
/// ```
///
/// This layout provides better cache utilization when processing multiple
/// polynomials simultaneously.
#[cfg(feature = "alloc")]
pub struct FftMatrix<E: IsField> {
    /// Flat storage for all polynomial coefficients
    pub data: Vec<FieldElement<E>>,
    /// Number of columns (polynomial length)
    pub width: usize,
    /// Number of rows (number of polynomials)
    pub height: usize,
}

#[cfg(feature = "alloc")]
impl<E: IsField> FftMatrix<E> {
    /// Create a new FFT matrix from a list of polynomials
    ///
    /// # Panics
    /// Panics if polynomials have different lengths.
    pub fn from_polynomials(polys: Vec<Vec<FieldElement<E>>>) -> Self {
        if polys.is_empty() {
            return Self {
                data: Vec::new(),
                width: 0,
                height: 0,
            };
        }

        let height = polys.len();
        let width = polys[0].len();

        // Flatten in row-major order (SoA layout)
        let mut data = Vec::with_capacity(height * width);
        for poly in polys {
            assert_eq!(poly.len(), width, "All polynomials must have same length");
            data.extend(poly);
        }

        Self {
            data,
            width,
            height,
        }
    }

    /// Get a mutable slice for polynomial at index `row`
    ///
    /// # Panics
    /// Panics if `row >= self.height`.
    pub fn row_mut(&mut self, row: usize) -> &mut [FieldElement<E>] {
        assert!(
            row < self.height,
            "Row index out of bounds: {} >= {}",
            row,
            self.height
        );
        let start = row.checked_mul(self.width).expect("Row index overflow");
        let end = start + self.width;
        &mut self.data[start..end]
    }

    /// Get an immutable slice for polynomial at index `row`
    ///
    /// # Panics
    /// Panics if `row >= self.height`.
    pub fn row(&self, row: usize) -> &[FieldElement<E>] {
        assert!(
            row < self.height,
            "Row index out of bounds: {} >= {}",
            row,
            self.height
        );
        let start = row.checked_mul(self.width).expect("Row index overflow");
        let end = start + self.width;
        &self.data[start..end]
    }

    /// Convert back to list of polynomials
    pub fn to_polynomials(self) -> Vec<Vec<FieldElement<E>>> {
        self.data
            .chunks(self.width)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
}

// =====================================================
// PARALLEL BOWERS FFT
// =====================================================

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Compute adaptive parallelization threshold based on number of available threads.
///
/// The threshold determines the minimum number of independent blocks required before
/// using parallel processing. This function adapts the threshold based on CPU core count:
/// - More cores → lower threshold (can parallelize earlier)
/// - Fewer cores → higher threshold (avoid overhead)
///
/// Formula: `max(num_threads * BLOCKS_PER_THREAD, MIN_BLOCKS)`
/// - Each thread should have at least 4 blocks to process (amortize spawn overhead)
/// - Minimum 16 total blocks to ensure meaningful parallelism
///
/// # Returns
/// Parallelization threshold (number of blocks)
#[cfg(feature = "parallel")]
#[inline]
fn adaptive_parallel_threshold() -> usize {
    const BLOCKS_PER_THREAD: usize = 4;
    const MIN_BLOCKS: usize = 16;

    let num_threads = rayon::current_num_threads();
    num_threads
        .saturating_mul(BLOCKS_PER_THREAD)
        .max(MIN_BLOCKS)
}

/// Optimized Parallel Bowers FFT with 2-layer fusion using LayerTwiddles
///
/// This is the recommended FFT for large inputs (>= 2^16 elements) when
/// the `parallel` feature is enabled. It combines:
///
/// 1. **Sequential twiddle access**: LayerTwiddles stores twiddles per layer
///    for cache-friendly sequential reads instead of strided access
/// 2. **2-layer fusion**: Processes two FFT layers at once, keeping intermediate
///    values in registers to reduce memory traffic
/// 3. **Internal parallelization**: Uses `par_chunks_mut` to process independent
///    blocks in parallel when there are enough blocks (adaptive threshold)
///
/// # Parallelization Strategy
///
/// The FFT is structured in layers, where each layer processes blocks of decreasing size.
/// Early layers have few large blocks (not enough parallelism), while later layers have
/// many small blocks (good parallelism). The parallelization threshold adapts based on
/// CPU core count to ensure threading overhead is amortized.
///
/// # Errors
/// Returns `FFTError::InputError` if:
/// - Input length is not a power of two
/// - Twiddle table size doesn't match input size
#[cfg(all(feature = "alloc", feature = "parallel"))]
#[allow(clippy::needless_range_loop)]
pub fn bowers_fft_opt_fused_parallel<F, E>(
    input: &mut [FieldElement<E>],
    layer_twiddles: &LayerTwiddles<F>,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
    FieldElement<F>: Send + Sync,
    FieldElement<E>: Send + Sync,
{
    // Adaptive threshold based on CPU core count.
    // More cores → lower threshold, can parallelize earlier.
    let parallel_threshold = adaptive_parallel_threshold();

    let n = input.len();
    if !n.is_power_of_two() {
        return Err(FFTError::InputError(n));
    }

    if n <= 1 {
        return Ok(());
    }

    let log_n = n.trailing_zeros() as usize;

    // Validate that twiddle table size matches input size
    if layer_twiddles.num_layers() != log_n {
        return Err(FFTError::InputError(n));
    }

    if n <= 4 {
        return bowers_fft_opt_fused(input, layer_twiddles);
    }

    let mut layer = 0;

    // Process pairs of layers with 2-layer fusion.
    while layer + 1 < log_n {
        let block_size = n >> layer;

        if block_size >= 4 {
            let twiddles_l0 = layer_twiddles.get_layer(layer);
            let twiddles_l1 = layer_twiddles.get_layer(layer + 1);
            let num_blocks = n / block_size;

            if num_blocks >= parallel_threshold {
                input.par_chunks_mut(block_size).for_each(|block| {
                    process_fused_block(block, twiddles_l0, twiddles_l1);
                });
            } else {
                for block_start in (0..n).step_by(block_size) {
                    let block = &mut input[block_start..block_start + block_size];
                    process_fused_block(block, twiddles_l0, twiddles_l1);
                }
            }
            layer += 2;
        } else {
            break;
        }
    }

    // Process remaining single layers (if odd number of layers)
    while layer < log_n {
        let block_size = n >> layer;
        let half_block = block_size >> 1;
        let num_blocks = n / block_size;
        let twiddles = layer_twiddles.get_layer(layer);

        if num_blocks >= parallel_threshold {
            input.par_chunks_mut(block_size).for_each(|block| {
                process_single_layer_block(block, twiddles, half_block);
            });
        } else {
            for block_start in (0..n).step_by(block_size) {
                for j in 0..half_block {
                    let i0 = block_start + j;
                    let i1 = i0 + half_block;
                    let w = &twiddles[j];

                    let sum = &input[i0] + &input[i1];
                    let diff = &input[i0] - &input[i1];
                    let diff_w = w * &diff;

                    input[i0] = sum;
                    input[i1] = diff_w;
                }
            }
        }
        layer += 1;
    }

    Ok(())
}

/// Process a single block with 2-layer fusion (DIF butterfly).
#[cfg(all(feature = "alloc", feature = "parallel"))]
#[inline]
fn process_fused_block<F, E>(
    block: &mut [FieldElement<E>],
    twiddles_l0: &[FieldElement<F>],
    twiddles_l1: &[FieldElement<F>],
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let block_size = block.len();
    let quarter = block_size >> 2;

    // Verify twiddle arrays have sufficient length
    debug_assert!(
        twiddles_l0.len() >= 2 * quarter,
        "twiddles_l0 too short: {} < {}",
        twiddles_l0.len(),
        2 * quarter
    );
    debug_assert!(
        twiddles_l1.len() >= quarter,
        "twiddles_l1 too short: {} < {}",
        twiddles_l1.len(),
        quarter
    );

    for j in 0..quarter {
        let i0 = j;
        let i1 = j + quarter;
        let i2 = j + 2 * quarter;
        let i3 = j + 3 * quarter;

        let w0 = &twiddles_l0[j];
        let w1 = &twiddles_l0[j + quarter];

        // First layer butterflies
        let sum_02 = &block[i0] + &block[i2];
        let diff_02 = &block[i0] - &block[i2];
        let diff_02_w = w0 * &diff_02;

        let sum_13 = &block[i1] + &block[i3];
        let diff_13 = &block[i1] - &block[i3];
        let diff_13_w = w1 * &diff_13;

        let w2 = &twiddles_l1[j];

        // Second layer butterflies
        let final_0 = &sum_02 + &sum_13;
        let diff_sums = &sum_02 - &sum_13;
        let final_1 = w2 * &diff_sums;

        let final_2 = &diff_02_w + &diff_13_w;
        let diff_diffs = &diff_02_w - &diff_13_w;
        let final_3 = w2 * &diff_diffs;

        block[i0] = final_0;
        block[i1] = final_1;
        block[i2] = final_2;
        block[i3] = final_3;
    }
}

/// Process a single layer block (used for remaining odd layer).
#[cfg(all(feature = "alloc", feature = "parallel"))]
#[inline]
fn process_single_layer_block<F, E>(
    block: &mut [FieldElement<E>],
    twiddles: &[FieldElement<F>],
    half_block: usize,
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    debug_assert!(
        twiddles.len() >= half_block,
        "twiddles too short: {} < {}",
        twiddles.len(),
        half_block
    );

    for j in 0..half_block {
        let w = &twiddles[j];

        let sum = &block[j] + &block[j + half_block];
        let diff = &block[j] - &block[j + half_block];
        let diff_w = w * &diff;

        block[j] = sum;
        block[j + half_block] = diff_w;
    }
}

// =====================================================
// LAYER-SPECIFIC TWIDDLE TABLES (OPTIMIZED)
// =====================================================

/// Pre-computed twiddle factors organized by layer for cache-friendly access.
///
/// # Why LayerTwiddles?
///
/// Standard FFT implementations access twiddles with strided patterns like `twiddles[j * 2^layer]`.
/// This causes cache misses because the stride grows exponentially with each layer, leading to
/// random memory access patterns.
///
/// LayerTwiddles reorganizes twiddles so that each layer's values are stored contiguously.
/// During FFT computation, we iterate sequentially through `layer_twiddles[layer][0..count]`,
/// achieving O(N) sequential memory access instead of O(N log N) strided access.
///
/// This optimization can provide 10-30% speedup on large inputs where memory bandwidth
/// is the bottleneck.
///
/// # Memory Layout
///
/// For an FFT of size n = 2^order:
/// - Layer 0: n/2 twiddles (w^0, w^1, w^2, ...)
/// - Layer 1: n/4 twiddles (w^0, w^2, w^4, ...)
/// - Layer k: n/2^(k+1) twiddles (w^0, w^(2^k), w^(2*2^k), ...)
///
/// Total memory: n - 1 twiddles (same as flat storage, but organized for locality).
///
/// # Reusability
///
/// LayerTwiddles can be computed once and reused for multiple FFTs of the same size.
/// This amortizes the precomputation cost when processing many polynomials.
#[cfg(feature = "alloc")]
#[derive(Clone)]
pub struct LayerTwiddles<F: IsField> {
    /// Twiddles organized by layer, stored contiguously for sequential access.
    pub layers: Vec<Vec<FieldElement<F>>>,
}

#[cfg(feature = "alloc")]
impl<F: IsFFTField> LayerTwiddles<F> {
    /// Compute layer-specific twiddles from primitive root of unity.
    ///
    /// For an FFT of size n = 2^order, layer k needs n/2^(k+1) twiddles.
    /// The twiddles for layer k are: w^0, w^(2^k), w^(2*2^k), w^(3*2^k), ...
    ///
    /// # Errors
    /// Returns `None` if:
    /// - `order` exceeds the maximum supported value (would cause integer overflow)
    /// - The field doesn't have a primitive root of unity for the given order
    ///
    /// # Example
    /// ```ignore
    /// let layer_twiddles = LayerTwiddles::<GoldilocksField>::new(10)
    ///     .expect("Failed to create twiddles for order 10");
    /// ```
    pub fn new(order: u64) -> Option<Self> {
        let root = F::get_primitive_root_of_unity(order).ok()?;
        Self::build(order, &root)
    }

    /// Compute layer-specific twiddles from the **inverse** primitive root of unity.
    ///
    /// This is used for the inverse FFT (IFFT). The inverse twiddles are computed
    /// from w^(-1) where w is the primitive root of unity.
    ///
    /// # Errors
    /// Returns `None` if:
    /// - `order` exceeds the maximum supported value (would cause integer overflow)
    /// - The field doesn't have a primitive root of unity for the given order
    ///
    /// # Example
    /// ```ignore
    /// let inv_twiddles = LayerTwiddles::<GoldilocksField>::new_inverse(10)
    ///     .expect("Failed to create inverse twiddles for order 10");
    /// ```
    pub fn new_inverse(order: u64) -> Option<Self> {
        let root = F::get_primitive_root_of_unity(order).ok()?;
        // Primitive roots of unity are always non-zero, so inversion succeeds.
        let inv_root = root.inv().ok()?;
        Self::build(order, &inv_root)
    }

    /// Shared implementation for `new` and `new_inverse`.
    fn build(order: u64, root: &FieldElement<F>) -> Option<Self> {
        if order > MAX_FFT_ORDER {
            return None;
        }

        let n = 1usize << order;
        let mut layers = Vec::with_capacity(order as usize);

        for layer in 0..order as usize {
            debug_assert!(
                layer < usize::BITS as usize,
                "Layer index exceeds shift limit"
            );

            let stride = 1usize << layer;
            let count = n >> (layer + 1);

            let mut layer_twiddles = Vec::with_capacity(count);
            let w_stride = root.pow(stride as u64);
            let mut current = FieldElement::<F>::one();

            for _ in 0..count {
                layer_twiddles.push(current.clone());
                current = &current * &w_stride;
            }

            layers.push(layer_twiddles);
        }

        Some(Self { layers })
    }

    /// Get the twiddles for a specific layer.
    ///
    /// # Panics
    /// Panics if `layer >= self.layers.len()`.
    #[inline(always)]
    pub fn get_layer(&self, layer: usize) -> &[FieldElement<F>] {
        assert!(
            layer < self.layers.len(),
            "Layer index out of bounds: {} >= {}",
            layer,
            self.layers.len()
        );
        &self.layers[layer]
    }

    /// Returns the number of layers (equal to the FFT order).
    #[inline(always)]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// Optimized Bowers IFFT with sequential twiddle access.
///
/// **Note**: This performs the inverse butterfly structure but does NOT apply
/// the 1/n scaling factor. The caller must:
/// 1. Pass inverse twiddles from `LayerTwiddles::new_inverse(order)`
/// 2. Scale results by n^(-1) after the transform
///
/// Using forward twiddles (from `LayerTwiddles::new()`) will produce incorrect results.
///
/// # Example
/// ```ignore
/// let order = 10u64;
/// let n = 1 << order;
///
/// // Create inverse twiddles for IFFT
/// let inv_twiddles = LayerTwiddles::<F>::new_inverse(order).unwrap();
///
/// // Apply inverse FFT (after bit-reversing FFT output)
/// in_place_bit_reverse_permute(&mut data);
/// bowers_ifft_opt(&mut data, &inv_twiddles)?;
///
/// // Scale by 1/n to complete the inverse transform
/// let n_inv = FieldElement::<F>::from(n as u64).inv().unwrap();
/// for val in data.iter_mut() {
///     *val = &*val * &n_inv;
/// }
/// ```
///
/// # Errors
/// Returns `FFTError::InputError` if:
/// - Input length is not a power of two
/// - Twiddle table size doesn't match input size
#[cfg(feature = "alloc")]
#[allow(clippy::needless_range_loop)]
pub fn bowers_ifft_opt<F, E>(
    input: &mut [FieldElement<E>],
    layer_twiddles: &LayerTwiddles<F>,
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

    // Validate that twiddle table size matches input size
    if layer_twiddles.num_layers() != log_n {
        return Err(FFTError::InputError(n));
    }

    for layer in (0..log_n).rev() {
        let block_size = n >> layer;
        let half_block = block_size >> 1;
        let twiddles = layer_twiddles.get_layer(layer);

        for block_start in (0..n).step_by(block_size) {
            for j in 0..half_block {
                let i0 = block_start + j;
                let i1 = i0 + half_block;
                let w = &twiddles[j];

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

/// Parallel Bowers IFFT with adaptive parallelization.
///
/// This is the parallel counterpart to `bowers_ifft_opt`. Like the forward parallel
/// variant, it uses `par_chunks_mut` to process independent blocks in parallel when
/// there are enough blocks to amortize threading overhead.
///
/// **Note**: This performs the inverse butterfly structure but does NOT apply
/// the 1/n scaling factor. The caller must scale results by n^(-1) after the transform.
///
/// # Errors
/// Returns `FFTError::InputError` if input length is not a power of two or if
/// twiddle table size doesn't match input size.
#[cfg(all(feature = "alloc", feature = "parallel"))]
#[allow(clippy::needless_range_loop)]
pub fn bowers_ifft_opt_parallel<F, E>(
    input: &mut [FieldElement<E>],
    layer_twiddles: &LayerTwiddles<F>,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
    FieldElement<F>: Send + Sync,
    FieldElement<E>: Send + Sync,
{
    let parallel_threshold = adaptive_parallel_threshold();

    let n = input.len();
    if !n.is_power_of_two() {
        return Err(FFTError::InputError(n));
    }

    if n <= 1 {
        return Ok(());
    }

    if n <= 4 {
        return bowers_ifft_opt(input, layer_twiddles);
    }

    let log_n = n.trailing_zeros() as usize;

    // Validate twiddle table matches input size
    if layer_twiddles.num_layers() != log_n {
        return Err(FFTError::InputError(n));
    }

    // DIT: iterate layers from bottom (log_n - 1) to top (0)
    for layer in (0..log_n).rev() {
        let block_size = n >> layer;
        let half_block = block_size >> 1;
        let num_blocks = n / block_size;
        let twiddles = layer_twiddles.get_layer(layer);

        if num_blocks >= parallel_threshold {
            input.par_chunks_mut(block_size).for_each(|block| {
                process_ifft_single_layer_block(block, twiddles, half_block);
            });
        } else {
            for block_start in (0..n).step_by(block_size) {
                for j in 0..half_block {
                    let i0 = block_start + j;
                    let i1 = i0 + half_block;
                    let w = &twiddles[j];

                    let bw = w * &input[i1];
                    let sum = &input[i0] + &bw;
                    let diff = &input[i0] - &bw;

                    input[i0] = sum;
                    input[i1] = diff;
                }
            }
        }
    }

    Ok(())
}

/// Process a single IFFT layer block (DIT butterfly: multiply then add/subtract).
#[cfg(all(feature = "alloc", feature = "parallel"))]
#[inline]
fn process_ifft_single_layer_block<F, E>(
    block: &mut [FieldElement<E>],
    twiddles: &[FieldElement<F>],
    half_block: usize,
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    debug_assert!(
        twiddles.len() >= half_block,
        "twiddles too short: {} < {}",
        twiddles.len(),
        half_block
    );

    for j in 0..half_block {
        let w = &twiddles[j];

        let bw = w * &block[j + half_block];
        let sum = &block[j] + &bw;
        let diff = &block[j] - &bw;

        block[j] = sum;
        block[j + half_block] = diff;
    }
}

// =====================================================
// PACKED (SIMD) BUTTERFLY FUNCTIONS
// =====================================================

/// Packed 2-layer fused DIF butterfly.
///
/// Operates on a single block of `block_size` elements, processing `WIDTH` lanes at a time.
/// The block is split into 4 contiguous quarters, each packed into SIMD vectors via
/// zero-cost pointer cast (`pack_slice_with_suffix_mut`).
///
/// Twiddles must be in the same field as the data (base field only).
#[cfg(all(feature = "alloc", feature = "parallel"))]
#[inline]
fn process_fused_block_packed<P: PackedField>(
    block: &mut [FieldElement<P::Scalar>],
    twiddles_l0: &[FieldElement<P::Scalar>],
    twiddles_l1: &[FieldElement<P::Scalar>],
) where
    P::Scalar: IsFFTField,
{
    let block_size = block.len();
    let quarter = block_size >> 2;

    debug_assert!(
        twiddles_l0.len() >= 2 * quarter,
        "twiddles_l0 too short: {} < {}",
        twiddles_l0.len(),
        2 * quarter
    );
    debug_assert!(
        twiddles_l1.len() >= quarter,
        "twiddles_l1 too short: {} < {}",
        twiddles_l1.len(),
        quarter
    );

    // Split into 4 contiguous quarters
    let (q01, q23) = block.split_at_mut(2 * quarter);
    let (q0, q1) = q01.split_at_mut(quarter);
    let (q2, q3) = q23.split_at_mut(quarter);

    // Zero-cost pack each quarter
    let (pq0, _) = P::pack_slice_with_suffix_mut(q0);
    let (pq1, _) = P::pack_slice_with_suffix_mut(q1);
    let (pq2, _) = P::pack_slice_with_suffix_mut(q2);
    let (pq3, _) = P::pack_slice_with_suffix_mut(q3);

    // Pack twiddles (contiguous per-layer)
    let (tw0_first, tw0_second) = twiddles_l0.split_at(quarter);
    let (ptw0_f, _) = P::pack_slice_with_suffix(tw0_first);
    let (ptw0_s, _) = P::pack_slice_with_suffix(tw0_second);
    let (ptw1, _) = P::pack_slice_with_suffix(twiddles_l1);

    // Packed butterfly loop (WIDTH iterations per step)
    for j in 0..pq0.len() {
        let w0 = ptw0_f[j];
        let w1 = ptw0_s[j];

        // First layer butterflies
        let sum_02 = pq0[j] + pq2[j];
        let diff_02 = pq0[j] - pq2[j];
        let diff_02_w = w0 * diff_02;

        let sum_13 = pq1[j] + pq3[j];
        let diff_13 = pq1[j] - pq3[j];
        let diff_13_w = w1 * diff_13;

        let w2 = ptw1[j];

        // Second layer butterflies
        let final_0 = sum_02 + sum_13;
        let diff_sums = sum_02 - sum_13;
        let final_1 = w2 * diff_sums;

        let final_2 = diff_02_w + diff_13_w;
        let diff_diffs = diff_02_w - diff_13_w;
        let final_3 = w2 * diff_diffs;

        pq0[j] = final_0;
        pq1[j] = final_1;
        pq2[j] = final_2;
        pq3[j] = final_3;
    }

    // Scalar tail for remainder (quarter % WIDTH != 0)
    let packed_count = pq0.len() * P::WIDTH;
    for j in packed_count..quarter {
        let w0 = &twiddles_l0[j];
        let w1 = &twiddles_l0[j + quarter];

        let sum_02 = &q0[j] + &q2[j];
        let diff_02 = &q0[j] - &q2[j];
        let diff_02_w = w0 * &diff_02;

        let sum_13 = &q1[j] + &q3[j];
        let diff_13 = &q1[j] - &q3[j];
        let diff_13_w = w1 * &diff_13;

        let w2 = &twiddles_l1[j];

        let final_0 = &sum_02 + &sum_13;
        let diff_sums = &sum_02 - &sum_13;
        let final_1 = w2 * &diff_sums;

        let final_2 = &diff_02_w + &diff_13_w;
        let diff_diffs = &diff_02_w - &diff_13_w;
        let final_3 = w2 * &diff_diffs;

        q0[j] = final_0;
        q1[j] = final_1;
        q2[j] = final_2;
        q3[j] = final_3;
    }
}

/// Packed single-layer DIF butterfly.
///
/// Splits the block into two halves, packs each, and processes WIDTH lanes at a time.
#[cfg(all(feature = "alloc", feature = "parallel"))]
#[inline]
fn process_single_layer_block_packed<P: PackedField>(
    block: &mut [FieldElement<P::Scalar>],
    twiddles: &[FieldElement<P::Scalar>],
    half_block: usize,
) where
    P::Scalar: IsFFTField,
{
    debug_assert!(
        twiddles.len() >= half_block,
        "twiddles too short: {} < {}",
        twiddles.len(),
        half_block
    );

    let (lo, hi) = block.split_at_mut(half_block);

    let (plo, _) = P::pack_slice_with_suffix_mut(lo);
    let (phi, _) = P::pack_slice_with_suffix_mut(hi);
    let (ptw, _) = P::pack_slice_with_suffix(twiddles);

    for j in 0..plo.len() {
        let w = ptw[j];
        let sum = plo[j] + phi[j];
        let diff = plo[j] - phi[j];
        let diff_w = w * diff;

        plo[j] = sum;
        phi[j] = diff_w;
    }

    // Scalar tail
    let packed_count = plo.len() * P::WIDTH;
    for j in packed_count..half_block {
        let w = &twiddles[j];
        let sum = &lo[j] + &hi[j];
        let diff = &lo[j] - &hi[j];
        let diff_w = w * &diff;

        lo[j] = sum;
        hi[j] = diff_w;
    }
}

/// Packed single-layer DIT (inverse) butterfly.
///
/// Splits the block into two halves, packs each, and processes WIDTH lanes at a time.
#[cfg(all(feature = "alloc", feature = "parallel"))]
#[inline]
fn process_ifft_single_layer_block_packed<P: PackedField>(
    block: &mut [FieldElement<P::Scalar>],
    twiddles: &[FieldElement<P::Scalar>],
    half_block: usize,
) where
    P::Scalar: IsFFTField,
{
    debug_assert!(
        twiddles.len() >= half_block,
        "twiddles too short: {} < {}",
        twiddles.len(),
        half_block
    );

    let (lo, hi) = block.split_at_mut(half_block);

    let (plo, _) = P::pack_slice_with_suffix_mut(lo);
    let (phi, _) = P::pack_slice_with_suffix_mut(hi);
    let (ptw, _) = P::pack_slice_with_suffix(twiddles);

    for j in 0..plo.len() {
        let w = ptw[j];
        let bw = w * phi[j];
        let sum = plo[j] + bw;
        let diff = plo[j] - bw;

        plo[j] = sum;
        phi[j] = diff;
    }

    // Scalar tail
    let packed_count = plo.len() * P::WIDTH;
    for j in packed_count..half_block {
        let w = &twiddles[j];
        let bw = w * &hi[j];
        let sum = &lo[j] + &bw;
        let diff = &lo[j] - &bw;

        lo[j] = sum;
        hi[j] = diff;
    }
}

/// Optimized Bowers FFT with 2-layer fusion and sequential twiddle access.
///
/// This is the recommended single-threaded FFT. It combines:
///
/// 1. **Sequential twiddle access**: LayerTwiddles stores twiddles per layer
///    for cache-friendly sequential reads
/// 2. **2-layer fusion**: Processes two FFT layers at once, keeping intermediate
///    values in registers to reduce memory traffic
///
/// For multi-threaded execution, use `bowers_fft_opt_fused_parallel` instead.
///
/// # Errors
/// Returns `FFTError::InputError` if:
/// - Input length is not a power of two
/// - Twiddle table size doesn't match input size
#[cfg(feature = "alloc")]
#[allow(clippy::needless_range_loop)]
pub fn bowers_fft_opt_fused<F, E>(
    input: &mut [FieldElement<E>],
    layer_twiddles: &LayerTwiddles<F>,
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

    // Validate that twiddle table size matches input size
    if layer_twiddles.num_layers() != log_n {
        return Err(FFTError::InputError(n));
    }

    // Handle small sizes with simple sequential processing
    if n <= 4 {
        for layer in 0..log_n {
            let block_size = n >> layer;
            let half_block = block_size >> 1;
            let twiddles = layer_twiddles.get_layer(layer);

            for block_start in (0..n).step_by(block_size) {
                for j in 0..half_block {
                    let i0 = block_start + j;
                    let i1 = i0 + half_block;
                    let w = &twiddles[j];

                    let sum = &input[i0] + &input[i1];
                    let diff = &input[i0] - &input[i1];
                    let diff_w = w * &diff;

                    input[i0] = sum;
                    input[i1] = diff_w;
                }
            }
        }
        return Ok(());
    }

    let mut layer = 0;

    // Process pairs of layers with 2-layer fusion
    while layer + 1 < log_n {
        let block_size = n >> layer;

        if block_size >= 4 {
            let twiddles_l0 = layer_twiddles.get_layer(layer);
            let twiddles_l1 = layer_twiddles.get_layer(layer + 1);

            for block_start in (0..n).step_by(block_size) {
                let quarter = block_size >> 2;
                let block = &mut input[block_start..block_start + block_size];

                // Verify twiddle arrays have sufficient length
                debug_assert!(
                    twiddles_l0.len() >= 2 * quarter,
                    "twiddles_l0 too short: {} < {}",
                    twiddles_l0.len(),
                    2 * quarter
                );
                debug_assert!(
                    twiddles_l1.len() >= quarter,
                    "twiddles_l1 too short: {} < {}",
                    twiddles_l1.len(),
                    quarter
                );

                for j in 0..quarter {
                    let i0 = j;
                    let i1 = j + quarter;
                    let i2 = j + 2 * quarter;
                    let i3 = j + 3 * quarter;

                    let w0 = &twiddles_l0[j];
                    let w1 = &twiddles_l0[j + quarter];

                    // First layer butterflies
                    let sum_02 = &block[i0] + &block[i2];
                    let diff_02 = &block[i0] - &block[i2];
                    let diff_02_w = w0 * &diff_02;

                    let sum_13 = &block[i1] + &block[i3];
                    let diff_13 = &block[i1] - &block[i3];
                    let diff_13_w = w1 * &diff_13;

                    let w2 = &twiddles_l1[j];

                    // Second layer butterflies
                    let final_0 = &sum_02 + &sum_13;
                    let diff_sums = &sum_02 - &sum_13;
                    let final_1 = w2 * &diff_sums;

                    let final_2 = &diff_02_w + &diff_13_w;
                    let diff_diffs = &diff_02_w - &diff_13_w;
                    let final_3 = w2 * &diff_diffs;

                    block[i0] = final_0;
                    block[i1] = final_1;
                    block[i2] = final_2;
                    block[i3] = final_3;
                }
            }
            layer += 2;
        } else {
            break;
        }
    }

    // Process remaining single layers (if odd number of layers)
    while layer < log_n {
        let block_size = n >> layer;
        let half_block = block_size >> 1;
        let twiddles = layer_twiddles.get_layer(layer);

        for block_start in (0..n).step_by(block_size) {
            for j in 0..half_block {
                let i0 = block_start + j;
                let i1 = i0 + half_block;
                let w = &twiddles[j];

                let sum = &input[i0] + &input[i1];
                let diff = &input[i0] - &input[i1];
                let diff_w = w * &diff;

                input[i0] = sum;
                input[i1] = diff_w;
            }
        }
        layer += 1;
    }

    Ok(())
}

/// Batch FFT using optimized Bowers algorithm with LayerTwiddles
///
/// # Errors
/// Returns `FFTError::InputError` if polynomial width is not a power of two.
#[cfg(feature = "alloc")]
pub fn bowers_batch_fft_opt<F, E>(
    matrix: &mut FftMatrix<E>,
    layer_twiddles: &LayerTwiddles<F>,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    if matrix.height == 0 || matrix.width <= 1 {
        return Ok(());
    }

    for row in 0..matrix.height {
        let poly = matrix.row_mut(row);
        bowers_fft_opt_fused(poly, layer_twiddles)?;
        in_place_bit_reverse_permute(poly);
    }

    Ok(())
}

// =====================================================
// PACKED (SIMD) BOWERS FFT
// =====================================================

/// Parallel Bowers FFT with packed (SIMD) inner butterflies.
///
/// This is identical to `bowers_fft_opt_fused_parallel` but uses packed butterfly
/// functions that process `P::WIDTH` lanes simultaneously via SIMD.
///
/// Only works when data and twiddles are in the same field (base field FFT).
/// For extension-field FFT (aux trace), use the scalar variants.
///
/// The parallel outer loop (rayon `par_chunks_mut`) and the packed inner loop
/// compose naturally: each rayon thread processes one block using packed butterflies.
#[cfg(all(feature = "alloc", feature = "parallel"))]
pub fn bowers_fft_opt_fused_packed<F, P>(
    input: &mut [FieldElement<F>],
    layer_twiddles: &LayerTwiddles<F>,
) -> Result<(), FFTError>
where
    F: IsFFTField + HasPacking<Packing = P>,
    P: PackedField<Scalar = F>,
    FieldElement<F>: Send + Sync,
{
    let parallel_threshold = adaptive_parallel_threshold();

    let n = input.len();
    if !n.is_power_of_two() {
        return Err(FFTError::InputError(n));
    }

    if n <= 1 {
        return Ok(());
    }

    let log_n = n.trailing_zeros() as usize;

    if layer_twiddles.num_layers() != log_n {
        return Err(FFTError::InputError(n));
    }

    if n <= 4 {
        return bowers_fft_opt_fused(input, layer_twiddles);
    }

    let mut layer = 0;

    // Process pairs of layers with 2-layer fusion + packed butterflies
    while layer + 1 < log_n {
        let block_size = n >> layer;

        if block_size >= 4 {
            let twiddles_l0 = layer_twiddles.get_layer(layer);
            let twiddles_l1 = layer_twiddles.get_layer(layer + 1);
            let num_blocks = n / block_size;

            if num_blocks >= parallel_threshold {
                input.par_chunks_mut(block_size).for_each(|block| {
                    process_fused_block_packed::<P>(block, twiddles_l0, twiddles_l1);
                });
            } else {
                for block_start in (0..n).step_by(block_size) {
                    let block = &mut input[block_start..block_start + block_size];
                    process_fused_block_packed::<P>(block, twiddles_l0, twiddles_l1);
                }
            }
            layer += 2;
        } else {
            break;
        }
    }

    // Process remaining single layers (if odd number of layers)
    while layer < log_n {
        let block_size = n >> layer;
        let half_block = block_size >> 1;
        let num_blocks = n / block_size;
        let twiddles = layer_twiddles.get_layer(layer);

        if num_blocks >= parallel_threshold {
            input.par_chunks_mut(block_size).for_each(|block| {
                process_single_layer_block_packed::<P>(block, twiddles, half_block);
            });
        } else {
            for block_start in (0..n).step_by(block_size) {
                let block = &mut input[block_start..block_start + block_size];
                process_single_layer_block_packed::<P>(block, twiddles, half_block);
            }
        }
        layer += 1;
    }

    Ok(())
}

/// Parallel Bowers IFFT with packed (SIMD) inner butterflies.
///
/// Counterpart of `bowers_fft_opt_fused_packed` for the inverse transform.
/// Does NOT apply 1/n scaling (caller must scale after).
#[cfg(all(feature = "alloc", feature = "parallel"))]
pub fn bowers_ifft_opt_packed<F, P>(
    input: &mut [FieldElement<F>],
    layer_twiddles: &LayerTwiddles<F>,
) -> Result<(), FFTError>
where
    F: IsFFTField + HasPacking<Packing = P>,
    P: PackedField<Scalar = F>,
    FieldElement<F>: Send + Sync,
{
    let parallel_threshold = adaptive_parallel_threshold();

    let n = input.len();
    if !n.is_power_of_two() {
        return Err(FFTError::InputError(n));
    }

    if n <= 1 {
        return Ok(());
    }

    if n <= 4 {
        return bowers_ifft_opt(input, layer_twiddles);
    }

    let log_n = n.trailing_zeros() as usize;

    if layer_twiddles.num_layers() != log_n {
        return Err(FFTError::InputError(n));
    }

    // DIT: iterate layers from bottom (log_n - 1) to top (0)
    for layer in (0..log_n).rev() {
        let block_size = n >> layer;
        let half_block = block_size >> 1;
        let num_blocks = n / block_size;
        let twiddles = layer_twiddles.get_layer(layer);

        if num_blocks >= parallel_threshold {
            input.par_chunks_mut(block_size).for_each(|block| {
                process_ifft_single_layer_block_packed::<P>(block, twiddles, half_block);
            });
        } else {
            for block_start in (0..n).step_by(block_size) {
                let block = &mut input[block_start..block_start + block_size];
                process_ifft_single_layer_block_packed::<P>(block, twiddles, half_block);
            }
        }
    }

    Ok(())
}
