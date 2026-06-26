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
//! use math::fft::bowers_fft::{LayerTwiddles, bowers_fft_opt_fused};
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
use crate::fft::errors::FFTError;
#[cfg(feature = "alloc")]
use crate::field::{
    element::FieldElement,
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

    // Process triples of layers with 3-layer fusion (radix-8 DIF).
    // Invariant: when layer + 2 < log_n, block_size = n >> layer >= 2^3 = 8.
    while layer + 2 < log_n {
        let block_size = n >> layer;
        debug_assert!(block_size >= 8);
        let tw0 = layer_twiddles.get_layer(layer);
        let tw1 = layer_twiddles.get_layer(layer + 1);
        let tw2 = layer_twiddles.get_layer(layer + 2);
        let num_blocks = n / block_size;

        if num_blocks >= parallel_threshold {
            input.par_chunks_mut(block_size).for_each(|block| {
                process_triple_fused_block(block, tw0, tw1, tw2);
            });
        } else {
            for block_start in (0..n).step_by(block_size) {
                let block = &mut input[block_start..block_start + block_size];
                process_triple_fused_block(block, tw0, tw1, tw2);
            }
        }
        layer += 3;
    }

    // Process remaining pairs of layers with 2-layer fusion.
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
#[cfg(feature = "alloc")]
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

/// Process a single block with 3-layer fusion (DIF radix-8 butterfly).
///
/// Processes 8 elements through 3 DIF butterfly layers at once, keeping all
/// intermediate values in registers. Reduces memory round-trips compared to
/// 2-layer fusion: 8 reads + 8 writes instead of 8+8+8+8 for separate layers.
#[cfg(feature = "alloc")]
#[inline]
#[allow(dead_code)]
fn process_triple_fused_block<F, E>(
    block: &mut [FieldElement<E>],
    twiddles_l0: &[FieldElement<F>],
    twiddles_l1: &[FieldElement<F>],
    twiddles_l2: &[FieldElement<F>],
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let block_size = block.len();
    let eighth = block_size >> 3;

    // Layer 0: half_block = block_size/2, stride between butterfly pairs = block_size/2
    // Layer 1: half_block = block_size/4, stride between butterfly pairs = block_size/4
    // Layer 2: half_block = block_size/8, stride between butterfly pairs = block_size/8

    for j in 0..eighth {
        // 8 input indices within this octant
        let i0 = j;
        let i1 = j + eighth;
        let i2 = j + 2 * eighth;
        let i3 = j + 3 * eighth;
        let i4 = j + 4 * eighth;
        let i5 = j + 5 * eighth;
        let i6 = j + 6 * eighth;
        let i7 = j + 7 * eighth;

        // Layer 0 twiddles: half_block = 4*eighth
        let w0_0 = &twiddles_l0[j];
        let w0_1 = &twiddles_l0[j + eighth];
        let w0_2 = &twiddles_l0[j + 2 * eighth];
        let w0_3 = &twiddles_l0[j + 3 * eighth];

        // Layer 0: DIF butterflies (pairs separated by 4*eighth)
        let s04 = &block[i0] + &block[i4];
        let d04 = w0_0 * &(&block[i0] - &block[i4]);
        let s15 = &block[i1] + &block[i5];
        let d15 = w0_1 * &(&block[i1] - &block[i5]);
        let s26 = &block[i2] + &block[i6];
        let d26 = w0_2 * &(&block[i2] - &block[i6]);
        let s37 = &block[i3] + &block[i7];
        let d37 = w0_3 * &(&block[i3] - &block[i7]);

        // Layer 1 twiddles: half_block = 2*eighth
        let w1_0 = &twiddles_l1[j];
        let w1_1 = &twiddles_l1[j + eighth];

        // Layer 1: DIF butterflies on sums (pairs separated by 2*eighth)
        let ss02 = &s04 + &s26;
        let ds02 = w1_0 * &(&s04 - &s26);
        let ss13 = &s15 + &s37;
        let ds13 = w1_1 * &(&s15 - &s37);

        // Layer 1: DIF butterflies on diffs (pairs separated by 2*eighth)
        let sd02 = &d04 + &d26;
        let dd02 = w1_0 * &(&d04 - &d26);
        let sd13 = &d15 + &d37;
        let dd13 = w1_1 * &(&d15 - &d37);

        // Layer 2 twiddle: half_block = eighth
        let w2 = &twiddles_l2[j];

        // Layer 2: DIF butterflies (pairs separated by eighth)
        block[i0] = &ss02 + &ss13;
        block[i1] = w2 * &(&ss02 - &ss13);
        block[i2] = &ds02 + &ds13;
        block[i3] = w2 * &(&ds02 - &ds13);
        block[i4] = &sd02 + &sd13;
        block[i5] = w2 * &(&sd02 - &sd13);
        block[i6] = &dd02 + &dd13;
        block[i7] = w2 * &(&dd02 - &dd13);
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

/// Process a single block with 2-layer IFFT fusion (DIT butterfly).
///
/// Processes two consecutive IFFT layers in a single pass. The `twiddles_hi` are
/// for the higher-numbered layer (processed first in DIT order) and `twiddles_lo`
/// are for the lower-numbered layer (processed second).
#[cfg(feature = "alloc")]
#[inline]
fn process_ifft_fused_block<F, E>(
    block: &mut [FieldElement<E>],
    twiddles_hi: &[FieldElement<F>],
    twiddles_lo: &[FieldElement<F>],
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let block_size = block.len();
    let quarter = block_size >> 2;

    debug_assert!(
        twiddles_hi.len() >= quarter,
        "twiddles_hi too short: {} < {}",
        twiddles_hi.len(),
        quarter
    );
    debug_assert!(
        twiddles_lo.len() >= 2 * quarter,
        "twiddles_lo too short: {} < {}",
        twiddles_lo.len(),
        2 * quarter
    );

    for j in 0..quarter {
        let i0 = j;
        let i1 = j + quarter;
        let i2 = j + 2 * quarter;
        let i3 = j + 3 * quarter;

        let w_hi = &twiddles_hi[j];

        // Layer hi: first sub-block butterfly (DIT: multiply then add/subtract)
        let bw0 = w_hi * &block[i1];
        let a0 = &block[i0] + &bw0;
        let b0 = &block[i0] - &bw0;

        // Layer hi: second sub-block butterfly
        let bw1 = w_hi * &block[i3];
        let a1 = &block[i2] + &bw1;
        let b1 = &block[i2] - &bw1;

        let w_lo_0 = &twiddles_lo[j];
        let w_lo_1 = &twiddles_lo[j + quarter];

        // Layer lo: butterfly on combined results
        let bw2 = w_lo_0 * &a1;
        block[i0] = &a0 + &bw2;
        block[i2] = &a0 - &bw2;

        let bw3 = w_lo_1 * &b1;
        block[i1] = &b0 + &bw3;
        block[i3] = &b0 - &bw3;
    }
}

/// Process a single block with 3-layer IFFT fusion (DIT radix-8 butterfly).
#[cfg(feature = "alloc")]
#[inline]
#[allow(dead_code)]
fn process_ifft_triple_fused_block<F, E>(
    block: &mut [FieldElement<E>],
    twiddles_hi: &[FieldElement<F>], // innermost layer (highest index)
    twiddles_mid: &[FieldElement<F>], // middle layer
    twiddles_lo: &[FieldElement<F>], // outermost layer (lowest index)
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let block_size = block.len();
    let eighth = block_size >> 3;

    for j in 0..eighth {
        let i0 = j;
        let i1 = j + eighth;
        let i2 = j + 2 * eighth;
        let i3 = j + 3 * eighth;
        let i4 = j + 4 * eighth;
        let i5 = j + 5 * eighth;
        let i6 = j + 6 * eighth;
        let i7 = j + 7 * eighth;

        // Layer hi (innermost): DIT butterflies on consecutive pairs
        // Pairs: (0,1), (2,3), (4,5), (6,7) — half_block = eighth
        let w_hi = &twiddles_hi[j];

        let bw01 = w_hi * &block[i1];
        let a01 = &block[i0] + &bw01;
        let b01 = &block[i0] - &bw01;

        let bw23 = w_hi * &block[i3];
        let a23 = &block[i2] + &bw23;
        let b23 = &block[i2] - &bw23;

        let bw45 = w_hi * &block[i5];
        let a45 = &block[i4] + &bw45;
        let b45 = &block[i4] - &bw45;

        let bw67 = w_hi * &block[i7];
        let a67 = &block[i6] + &bw67;
        let b67 = &block[i6] - &bw67;

        // Layer mid: DIT butterflies on groups of 4
        // Pairs: (a01,a23), (b01,b23), (a45,a67), (b45,b67)
        let w_mid_0 = &twiddles_mid[j];
        let w_mid_1 = &twiddles_mid[j + eighth];

        let bw_m0 = w_mid_0 * &a23;
        let aa0 = &a01 + &bw_m0;
        let ab0 = &a01 - &bw_m0;

        let bw_m1 = w_mid_1 * &b23;
        let ba0 = &b01 + &bw_m1;
        let bb0 = &b01 - &bw_m1;

        let bw_m2 = w_mid_0 * &a67;
        let aa1 = &a45 + &bw_m2;
        let ab1 = &a45 - &bw_m2;

        let bw_m3 = w_mid_1 * &b67;
        let ba1 = &b45 + &bw_m3;
        let bb1 = &b45 - &bw_m3;

        // Layer lo (outermost): DIT butterflies on groups of 8
        let w_lo_0 = &twiddles_lo[j];
        let w_lo_1 = &twiddles_lo[j + eighth];
        let w_lo_2 = &twiddles_lo[j + 2 * eighth];
        let w_lo_3 = &twiddles_lo[j + 3 * eighth];

        let bw_l0 = w_lo_0 * &aa1;
        block[i0] = &aa0 + &bw_l0;
        block[i4] = &aa0 - &bw_l0;

        let bw_l1 = w_lo_1 * &ba1;
        block[i1] = &ba0 + &bw_l1;
        block[i5] = &ba0 - &bw_l1;

        let bw_l2 = w_lo_2 * &ab1;
        block[i2] = &ab0 + &bw_l2;
        block[i6] = &ab0 - &bw_l2;

        let bw_l3 = w_lo_3 * &bb1;
        block[i3] = &bb0 + &bw_l3;
        block[i7] = &bb0 - &bw_l3;
    }
}

/// Optimized Bowers IFFT with 2-layer fusion and sequential twiddle access.
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

    // Handle small sizes with simple sequential processing
    if n <= 4 {
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
        return Ok(());
    }

    // Process pairs of layers with 2-layer fusion (DIT, reverse order)
    let mut layer = log_n;

    while layer >= 2 {
        let layer_hi = layer - 1;
        let layer_lo = layer - 2;
        let block_size = n >> layer_lo;

        debug_assert!(block_size >= 4);
        let twiddles_hi = layer_twiddles.get_layer(layer_hi);
        let twiddles_lo = layer_twiddles.get_layer(layer_lo);

        for block_start in (0..n).step_by(block_size) {
            let block = &mut input[block_start..block_start + block_size];
            process_ifft_fused_block(block, twiddles_hi, twiddles_lo);
        }
        layer -= 2;
    }

    // Process remaining single layer (if odd number of layers)
    if layer >= 1 {
        let remaining_layer = layer - 1;
        let block_size = n >> remaining_layer;
        let half_block = block_size >> 1;
        let twiddles = layer_twiddles.get_layer(remaining_layer);

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

/// Parallel Bowers IFFT with 2-layer fusion and adaptive parallelization.
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

    // Process triples of layers with 3-layer fusion (DIT, reverse order)
    let mut layer = log_n;

    // Invariant: when layer >= 3, block_size = n >> (layer - 3) >= 2^3 = 8.
    while layer >= 3 {
        let layer_hi = layer - 1;
        let layer_mid = layer - 2;
        let layer_lo = layer - 3;
        let block_size = n >> layer_lo;
        debug_assert!(block_size >= 8);

        let tw_hi = layer_twiddles.get_layer(layer_hi);
        let tw_mid = layer_twiddles.get_layer(layer_mid);
        let tw_lo = layer_twiddles.get_layer(layer_lo);
        let num_blocks = n / block_size;

        if num_blocks >= parallel_threshold {
            input.par_chunks_mut(block_size).for_each(|block| {
                process_ifft_triple_fused_block(block, tw_hi, tw_mid, tw_lo);
            });
        } else {
            for block_start in (0..n).step_by(block_size) {
                let block = &mut input[block_start..block_start + block_size];
                process_ifft_triple_fused_block(block, tw_hi, tw_mid, tw_lo);
            }
        }
        layer -= 3;
    }

    // Process remaining pairs of layers with 2-layer fusion (DIT, reverse order)
    while layer >= 2 {
        let layer_hi = layer - 1;
        let layer_lo = layer - 2;
        let block_size = n >> layer_lo;

        debug_assert!(block_size >= 4);
        let twiddles_hi = layer_twiddles.get_layer(layer_hi);
        let twiddles_lo = layer_twiddles.get_layer(layer_lo);
        let num_blocks = n / block_size;

        if num_blocks >= parallel_threshold {
            input.par_chunks_mut(block_size).for_each(|block| {
                process_ifft_fused_block(block, twiddles_hi, twiddles_lo);
            });
        } else {
            for block_start in (0..n).step_by(block_size) {
                let block = &mut input[block_start..block_start + block_size];
                process_ifft_fused_block(block, twiddles_hi, twiddles_lo);
            }
        }
        layer -= 2;
    }

    // Process remaining single layer (if odd number of layers)
    if layer >= 1 {
        let remaining_layer = layer - 1;
        let block_size = n >> remaining_layer;
        let half_block = block_size >> 1;
        let num_blocks = n / block_size;
        let twiddles = layer_twiddles.get_layer(remaining_layer);

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
                let block = &mut input[block_start..block_start + block_size];
                process_fused_block(block, twiddles_l0, twiddles_l1);
            }
            layer += 2;
        } else {
            break;
        }
    }

    // Process remaining single layer (if odd number of layers)
    if layer < log_n {
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

    Ok(())
}
