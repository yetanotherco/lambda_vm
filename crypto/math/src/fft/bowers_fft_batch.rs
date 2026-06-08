//! Batched row-major Bowers FFT.
//!
//! Operates on a flat row-major buffer of size `N * M` where N = FFT size and
//! M = num_cols. Each "row" is M consecutive elements at offset `row * M`.
//!
//! The algorithm mirrors `bowers_fft_opt_fused_parallel` (DIF) and
//! `bowers_ifft_opt_parallel` (DIT) butterfly structure, but for each
//! butterfly pair (i, j) the twiddle is loaded once and applied to the M
//! consecutive elements at offsets [i*M..(i+1)*M] and [j*M..(j+1)*M]. This
//! gives twiddle cache reuse and contiguous-row data locality, which is the
//! key win behind P3's `coset_lde_batch`.
//!
//! This first cut is the unfused / single-layer version: correct but does
//! not yet apply the 2- or 3-layer fusion that the single-column path uses.
//! Fusion is a follow-up.
//!
//! Bit-reversal lives separately in
//! [`in_place_bit_reverse_permute_row_major`]: swap whole rows (M elements)
//! when the row index `i` has a higher bit-reversed partner `j`.

#[cfg(feature = "alloc")]
use crate::fft::errors::FFTError;
#[cfg(feature = "alloc")]
use crate::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

#[cfg(feature = "alloc")]
use crate::fft::bit_reversing::reverse_index;
#[cfg(feature = "alloc")]
use crate::fft::bowers_fft::LayerTwiddles;

#[cfg(all(feature = "alloc", feature = "parallel"))]
use rayon::prelude::*;

/// Threshold for engaging intra-block parallelism inside the butterfly loop.
/// Below this, even Rayon-aware blocks fall back to sequential per-row work.
#[cfg(all(feature = "alloc", feature = "parallel"))]
const INTRA_BLOCK_PAR_THRESHOLD: usize = 64;

/// Process a 2-layer fused DIF block in row-major layout (radix-4 butterfly,
/// 4 sub-blocks of `quarter` rows each).
///
/// Mirrors the single-column `process_fused_block` but operates on M-wide
/// rows. Twiddle factors are loaded once per quartet index `j` and applied
/// to M elements per row.
#[cfg(feature = "alloc")]
#[inline]
fn dif_fused2_block_row_major<F, E>(
    block: &mut [FieldElement<E>],
    num_cols: usize,
    twiddles_l0: &[FieldElement<F>],
    twiddles_l1: &[FieldElement<F>],
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    let m = num_cols;
    let block_size_rows = block.len() / m;
    let quarter_rows = block_size_rows >> 2;
    let q_off = quarter_rows * m;
    let (sub0, rest1) = block.split_at_mut(q_off);
    let (sub1, rest2) = rest1.split_at_mut(q_off);
    let (sub2, sub3) = rest2.split_at_mut(q_off);

    let inner = |j: usize,
                 r0: &mut [FieldElement<E>],
                 r1: &mut [FieldElement<E>],
                 r2: &mut [FieldElement<E>],
                 r3: &mut [FieldElement<E>]| {
        let w0 = &twiddles_l0[j];
        let w1 = &twiddles_l0[j + quarter_rows];
        let w2 = &twiddles_l1[j];
        for k in 0..m {
            let sum_02 = &r0[k] + &r2[k];
            let diff_02 = &r0[k] - &r2[k];
            let diff_02_w = w0 * &diff_02;
            let sum_13 = &r1[k] + &r3[k];
            let diff_13 = &r1[k] - &r3[k];
            let diff_13_w = w1 * &diff_13;
            let final_0 = &sum_02 + &sum_13;
            let diff_sums = &sum_02 - &sum_13;
            let final_1 = w2 * &diff_sums;
            let final_2 = &diff_02_w + &diff_13_w;
            let diff_diffs = &diff_02_w - &diff_13_w;
            let final_3 = w2 * &diff_diffs;
            r0[k] = final_0;
            r1[k] = final_1;
            r2[k] = final_2;
            r3[k] = final_3;
        }
    };

    #[cfg(feature = "parallel")]
    {
        if quarter_rows >= INTRA_BLOCK_PAR_THRESHOLD {
            sub0.par_chunks_exact_mut(m)
                .zip(sub1.par_chunks_exact_mut(m))
                .zip(sub2.par_chunks_exact_mut(m))
                .zip(sub3.par_chunks_exact_mut(m))
                .enumerate()
                .for_each(|(j, (((r0, r1), r2), r3))| inner(j, r0, r1, r2, r3));
            return;
        }
    }
    for j in 0..quarter_rows {
        let r0 = &mut sub0[j * m..j * m + m];
        let r1 = &mut sub1[j * m..j * m + m];
        let r2 = &mut sub2[j * m..j * m + m];
        let r3 = &mut sub3[j * m..j * m + m];
        inner(j, r0, r1, r2, r3);
    }
}

/// Process a 3-layer fused DIF block in row-major layout (radix-8 butterfly,
/// 8 sub-blocks of `eighth` rows each).
#[cfg(feature = "alloc")]
#[inline]
fn dif_fused3_block_row_major<F, E>(
    block: &mut [FieldElement<E>],
    num_cols: usize,
    twiddles_l0: &[FieldElement<F>],
    twiddles_l1: &[FieldElement<F>],
    twiddles_l2: &[FieldElement<F>],
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    let m = num_cols;
    let block_size_rows = block.len() / m;
    let eighth_rows = block_size_rows >> 3;
    let e_off = eighth_rows * m;
    let (s0, r1) = block.split_at_mut(e_off);
    let (s1, r2) = r1.split_at_mut(e_off);
    let (s2, r3) = r2.split_at_mut(e_off);
    let (s3, r4) = r3.split_at_mut(e_off);
    let (s4, r5) = r4.split_at_mut(e_off);
    let (s5, r6) = r5.split_at_mut(e_off);
    let (s6, s7) = r6.split_at_mut(e_off);

    let inner = |j: usize,
                 b0: &mut [FieldElement<E>],
                 b1: &mut [FieldElement<E>],
                 b2: &mut [FieldElement<E>],
                 b3: &mut [FieldElement<E>],
                 b4: &mut [FieldElement<E>],
                 b5: &mut [FieldElement<E>],
                 b6: &mut [FieldElement<E>],
                 b7: &mut [FieldElement<E>]| {
        let w0_0 = &twiddles_l0[j];
        let w0_1 = &twiddles_l0[j + eighth_rows];
        let w0_2 = &twiddles_l0[j + 2 * eighth_rows];
        let w0_3 = &twiddles_l0[j + 3 * eighth_rows];
        let w1_0 = &twiddles_l1[j];
        let w1_1 = &twiddles_l1[j + eighth_rows];
        let w2 = &twiddles_l2[j];
        for k in 0..m {
            let s04 = &b0[k] + &b4[k];
            let d04 = w0_0 * &(&b0[k] - &b4[k]);
            let s15 = &b1[k] + &b5[k];
            let d15 = w0_1 * &(&b1[k] - &b5[k]);
            let s26 = &b2[k] + &b6[k];
            let d26 = w0_2 * &(&b2[k] - &b6[k]);
            let s37 = &b3[k] + &b7[k];
            let d37 = w0_3 * &(&b3[k] - &b7[k]);

            let ss02 = &s04 + &s26;
            let ds02 = w1_0 * &(&s04 - &s26);
            let ss13 = &s15 + &s37;
            let ds13 = w1_1 * &(&s15 - &s37);
            let sd02 = &d04 + &d26;
            let dd02 = w1_0 * &(&d04 - &d26);
            let sd13 = &d15 + &d37;
            let dd13 = w1_1 * &(&d15 - &d37);

            b0[k] = &ss02 + &ss13;
            b1[k] = w2 * &(&ss02 - &ss13);
            b2[k] = &ds02 + &ds13;
            b3[k] = w2 * &(&ds02 - &ds13);
            b4[k] = &sd02 + &sd13;
            b5[k] = w2 * &(&sd02 - &sd13);
            b6[k] = &dd02 + &dd13;
            b7[k] = w2 * &(&dd02 - &dd13);
        }
    };

    #[cfg(feature = "parallel")]
    {
        if eighth_rows >= INTRA_BLOCK_PAR_THRESHOLD {
            s0.par_chunks_exact_mut(m)
                .zip(s1.par_chunks_exact_mut(m))
                .zip(s2.par_chunks_exact_mut(m))
                .zip(s3.par_chunks_exact_mut(m))
                .zip(s4.par_chunks_exact_mut(m))
                .zip(s5.par_chunks_exact_mut(m))
                .zip(s6.par_chunks_exact_mut(m))
                .zip(s7.par_chunks_exact_mut(m))
                .enumerate()
                .for_each(|(j, (((((((b0, b1), b2), b3), b4), b5), b6), b7))| {
                    inner(j, b0, b1, b2, b3, b4, b5, b6, b7)
                });
            return;
        }
    }
    for j in 0..eighth_rows {
        let b0 = &mut s0[j * m..j * m + m];
        let b1 = &mut s1[j * m..j * m + m];
        let b2 = &mut s2[j * m..j * m + m];
        let b3 = &mut s3[j * m..j * m + m];
        let b4 = &mut s4[j * m..j * m + m];
        let b5 = &mut s5[j * m..j * m + m];
        let b6 = &mut s6[j * m..j * m + m];
        let b7 = &mut s7[j * m..j * m + m];
        inner(j, b0, b1, b2, b3, b4, b5, b6, b7);
    }
}

/// Process a 2-layer fused DIT (iFFT) block in row-major layout.
#[cfg(feature = "alloc")]
#[inline]
fn dit_fused2_block_row_major<F, E>(
    block: &mut [FieldElement<E>],
    num_cols: usize,
    twiddles_hi: &[FieldElement<F>],
    twiddles_lo: &[FieldElement<F>],
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    let m = num_cols;
    let block_size_rows = block.len() / m;
    let quarter_rows = block_size_rows >> 2;
    let q_off = quarter_rows * m;
    let (sub0, rest1) = block.split_at_mut(q_off);
    let (sub1, rest2) = rest1.split_at_mut(q_off);
    let (sub2, sub3) = rest2.split_at_mut(q_off);

    let inner = |j: usize,
                 r0: &mut [FieldElement<E>],
                 r1: &mut [FieldElement<E>],
                 r2: &mut [FieldElement<E>],
                 r3: &mut [FieldElement<E>]| {
        let w_hi = &twiddles_hi[j];
        let w_lo_0 = &twiddles_lo[j];
        let w_lo_1 = &twiddles_lo[j + quarter_rows];
        for k in 0..m {
            let bw0 = w_hi * &r1[k];
            let a0 = &r0[k] + &bw0;
            let b0 = &r0[k] - &bw0;
            let bw1 = w_hi * &r3[k];
            let a1 = &r2[k] + &bw1;
            let b1 = &r2[k] - &bw1;
            let bw2 = w_lo_0 * &a1;
            r0[k] = &a0 + &bw2;
            r2[k] = &a0 - &bw2;
            let bw3 = w_lo_1 * &b1;
            r1[k] = &b0 + &bw3;
            r3[k] = &b0 - &bw3;
        }
    };

    #[cfg(feature = "parallel")]
    {
        if quarter_rows >= INTRA_BLOCK_PAR_THRESHOLD {
            sub0.par_chunks_exact_mut(m)
                .zip(sub1.par_chunks_exact_mut(m))
                .zip(sub2.par_chunks_exact_mut(m))
                .zip(sub3.par_chunks_exact_mut(m))
                .enumerate()
                .for_each(|(j, (((r0, r1), r2), r3))| inner(j, r0, r1, r2, r3));
            return;
        }
    }
    for j in 0..quarter_rows {
        let r0 = &mut sub0[j * m..j * m + m];
        let r1 = &mut sub1[j * m..j * m + m];
        let r2 = &mut sub2[j * m..j * m + m];
        let r3 = &mut sub3[j * m..j * m + m];
        inner(j, r0, r1, r2, r3);
    }
}

/// Process a 3-layer fused DIT (iFFT) block in row-major layout.
#[cfg(feature = "alloc")]
#[inline]
fn dit_fused3_block_row_major<F, E>(
    block: &mut [FieldElement<E>],
    num_cols: usize,
    twiddles_hi: &[FieldElement<F>],
    twiddles_mid: &[FieldElement<F>],
    twiddles_lo: &[FieldElement<F>],
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    let m = num_cols;
    let block_size_rows = block.len() / m;
    let eighth_rows = block_size_rows >> 3;
    let e_off = eighth_rows * m;
    let (s0, r1) = block.split_at_mut(e_off);
    let (s1, r2) = r1.split_at_mut(e_off);
    let (s2, r3) = r2.split_at_mut(e_off);
    let (s3, r4) = r3.split_at_mut(e_off);
    let (s4, r5) = r4.split_at_mut(e_off);
    let (s5, r6) = r5.split_at_mut(e_off);
    let (s6, s7) = r6.split_at_mut(e_off);

    let inner = |j: usize,
                 b0: &mut [FieldElement<E>],
                 b1: &mut [FieldElement<E>],
                 b2: &mut [FieldElement<E>],
                 b3: &mut [FieldElement<E>],
                 b4: &mut [FieldElement<E>],
                 b5: &mut [FieldElement<E>],
                 b6: &mut [FieldElement<E>],
                 b7: &mut [FieldElement<E>]| {
        let w_hi = &twiddles_hi[j];
        let w_mid_0 = &twiddles_mid[j];
        let w_mid_1 = &twiddles_mid[j + eighth_rows];
        let w_lo_0 = &twiddles_lo[j];
        let w_lo_1 = &twiddles_lo[j + eighth_rows];
        let w_lo_2 = &twiddles_lo[j + 2 * eighth_rows];
        let w_lo_3 = &twiddles_lo[j + 3 * eighth_rows];
        for k in 0..m {
            let bw01 = w_hi * &b1[k];
            let a01 = &b0[k] + &bw01;
            let bb01 = &b0[k] - &bw01;
            let bw23 = w_hi * &b3[k];
            let a23 = &b2[k] + &bw23;
            let bb23 = &b2[k] - &bw23;
            let bw45 = w_hi * &b5[k];
            let a45 = &b4[k] + &bw45;
            let bb45 = &b4[k] - &bw45;
            let bw67 = w_hi * &b7[k];
            let a67 = &b6[k] + &bw67;
            let bb67 = &b6[k] - &bw67;

            let bw_m0 = w_mid_0 * &a23;
            let aa0 = &a01 + &bw_m0;
            let ab0 = &a01 - &bw_m0;
            let bw_m1 = w_mid_1 * &bb23;
            let ba0 = &bb01 + &bw_m1;
            let bb0 = &bb01 - &bw_m1;
            let bw_m2 = w_mid_0 * &a67;
            let aa1 = &a45 + &bw_m2;
            let ab1 = &a45 - &bw_m2;
            let bw_m3 = w_mid_1 * &bb67;
            let ba1 = &bb45 + &bw_m3;
            let bb1 = &bb45 - &bw_m3;

            let bw_l0 = w_lo_0 * &aa1;
            b0[k] = &aa0 + &bw_l0;
            b4[k] = &aa0 - &bw_l0;
            let bw_l1 = w_lo_1 * &ba1;
            b1[k] = &ba0 + &bw_l1;
            b5[k] = &ba0 - &bw_l1;
            let bw_l2 = w_lo_2 * &ab1;
            b2[k] = &ab0 + &bw_l2;
            b6[k] = &ab0 - &bw_l2;
            let bw_l3 = w_lo_3 * &bb1;
            b3[k] = &bb0 + &bw_l3;
            b7[k] = &bb0 - &bw_l3;
        }
    };

    #[cfg(feature = "parallel")]
    {
        if eighth_rows >= INTRA_BLOCK_PAR_THRESHOLD {
            s0.par_chunks_exact_mut(m)
                .zip(s1.par_chunks_exact_mut(m))
                .zip(s2.par_chunks_exact_mut(m))
                .zip(s3.par_chunks_exact_mut(m))
                .zip(s4.par_chunks_exact_mut(m))
                .zip(s5.par_chunks_exact_mut(m))
                .zip(s6.par_chunks_exact_mut(m))
                .zip(s7.par_chunks_exact_mut(m))
                .enumerate()
                .for_each(|(j, (((((((b0, b1), b2), b3), b4), b5), b6), b7))| {
                    inner(j, b0, b1, b2, b3, b4, b5, b6, b7)
                });
            return;
        }
    }
    for j in 0..eighth_rows {
        let b0 = &mut s0[j * m..j * m + m];
        let b1 = &mut s1[j * m..j * m + m];
        let b2 = &mut s2[j * m..j * m + m];
        let b3 = &mut s3[j * m..j * m + m];
        let b4 = &mut s4[j * m..j * m + m];
        let b5 = &mut s5[j * m..j * m + m];
        let b6 = &mut s6[j * m..j * m + m];
        let b7 = &mut s7[j * m..j * m + m];
        inner(j, b0, b1, b2, b3, b4, b5, b6, b7);
    }
}

/// Process one DIF block in row-major layout (M-wide rows).
///
/// When the parallel feature is enabled and `half_block_rows >= threshold`,
/// the butterflies inside this block run in parallel via Rayon — important
/// for the early FFT layers (large blocks, few blocks) where across-block
/// parallelism is not available.
#[cfg(feature = "alloc")]
#[inline]
fn dif_block_row_major<F, E>(
    block: &mut [FieldElement<E>],
    num_cols: usize,
    twiddles: &[FieldElement<F>],
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    let m = num_cols;
    let block_size_rows = block.len() / m;
    let half_block_rows = block_size_rows >> 1;
    let half_off = half_block_rows * m;
    let (lo_part, hi_part) = block.split_at_mut(half_off);

    #[cfg(feature = "parallel")]
    {
        if half_block_rows >= INTRA_BLOCK_PAR_THRESHOLD {
            lo_part
                .par_chunks_exact_mut(m)
                .zip(hi_part.par_chunks_exact_mut(m))
                .zip(twiddles[..half_block_rows].par_iter())
                .for_each(|((lo_row, hi_row), w)| {
                    for k in 0..m {
                        let sum = &lo_row[k] + &hi_row[k];
                        let diff = &lo_row[k] - &hi_row[k];
                        let diff_w = w * &diff;
                        lo_row[k] = sum;
                        hi_row[k] = diff_w;
                    }
                });
            return;
        }
    }

    for j in 0..half_block_rows {
        let w = &twiddles[j];
        let lo_row = &mut lo_part[j * m..j * m + m];
        let hi_row = &mut hi_part[j * m..j * m + m];
        for k in 0..m {
            let sum = &lo_row[k] + &hi_row[k];
            let diff = &lo_row[k] - &hi_row[k];
            let diff_w = w * &diff;
            lo_row[k] = sum;
            hi_row[k] = diff_w;
        }
    }
}

/// Process one DIT (inverse) block in row-major layout.
#[cfg(feature = "alloc")]
#[inline]
fn dit_block_row_major<F, E>(
    block: &mut [FieldElement<E>],
    num_cols: usize,
    twiddles: &[FieldElement<F>],
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    let m = num_cols;
    let block_size_rows = block.len() / m;
    let half_block_rows = block_size_rows >> 1;
    let half_off = half_block_rows * m;
    let (lo_part, hi_part) = block.split_at_mut(half_off);

    #[cfg(feature = "parallel")]
    {
        if half_block_rows >= INTRA_BLOCK_PAR_THRESHOLD {
            lo_part
                .par_chunks_exact_mut(m)
                .zip(hi_part.par_chunks_exact_mut(m))
                .zip(twiddles[..half_block_rows].par_iter())
                .for_each(|((lo_row, hi_row), w)| {
                    for k in 0..m {
                        let bw = w * &hi_row[k];
                        let sum = &lo_row[k] + &bw;
                        let diff = &lo_row[k] - &bw;
                        lo_row[k] = sum;
                        hi_row[k] = diff;
                    }
                });
            return;
        }
    }

    for j in 0..half_block_rows {
        let w = &twiddles[j];
        let lo_row = &mut lo_part[j * m..j * m + m];
        let hi_row = &mut hi_part[j * m..j * m + m];
        for k in 0..m {
            let bw = w * &hi_row[k];
            let sum = &lo_row[k] + &bw;
            let diff = &lo_row[k] - &bw;
            lo_row[k] = sum;
            hi_row[k] = diff;
        }
    }
}

/// Adaptive parallelism threshold: at least this many independent FFT blocks
/// before we engage Rayon (matches the single-column path).
#[cfg(all(feature = "alloc", feature = "parallel"))]
#[inline]
fn adaptive_parallel_threshold() -> usize {
    const BLOCKS_PER_THREAD: usize = 4;
    const MIN_BLOCKS: usize = 16;
    rayon::current_num_threads()
        .saturating_mul(BLOCKS_PER_THREAD)
        .max(MIN_BLOCKS)
}

/// In-place bit-reverse permutation over rows of a row-major buffer.
///
/// `buf.len()` must equal `n * num_cols` for some power-of-two `n`. Each row
/// is M = `num_cols` consecutive elements. Row `i` is swapped with row
/// `reverse_index(i, n)` when that index is greater (so each pair is swapped
/// exactly once).
///
/// Parallel path: gather all `(i, br(i))` pairs with `br(i) > i` (which are
/// pairwise disjoint by construction) and dispatch each swap to a worker
/// via raw-pointer indexing. Safe because each pair touches two distinct
/// non-overlapping row slices and the pairs themselves don't share rows.
#[cfg(feature = "alloc")]
pub fn in_place_bit_reverse_permute_row_major<E: Send + Sync>(buf: &mut [E], num_cols: usize) {
    if num_cols == 0 || buf.is_empty() {
        return;
    }
    debug_assert!(
        buf.len().is_multiple_of(num_cols),
        "buf.len() must be a multiple of num_cols"
    );
    let n = buf.len() / num_cols;
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "row count must be a power of two");

    #[cfg(feature = "parallel")]
    {
        // For each i in 0..n we check if its bit-reversed partner j has
        // `j > i`; if so we swap rows i and j directly. Bit-reverse is a
        // permutation, so distinct i values map to distinct j values, and
        // (i, j) pairs are pairwise disjoint — safe to dispatch via raw
        // pointer indexing in parallel. No upfront Vec<(usize, usize)>
        // collection (saves ~16 MB at log21 n=64).
        if n >= 2048 {
            use core::sync::atomic::{AtomicPtr, Ordering};
            let raw = AtomicPtr::new(buf.as_mut_ptr());
            (0..n).into_par_iter().for_each(|i| {
                let j = reverse_index(i, n as u64);
                if j > i {
                    let ptr = raw.load(Ordering::Relaxed);
                    let lo = i * num_cols;
                    let hi = j * num_cols;
                    // SAFETY: (lo..lo+M) and (hi..hi+M) point into the same
                    // Vec but are disjoint (lo != hi); the par_iter visits
                    // each unordered pair exactly once (we filter on j > i),
                    // so no two threads touch overlapping ranges.
                    unsafe {
                        let lo_row = core::slice::from_raw_parts_mut(ptr.add(lo), num_cols);
                        let hi_row = core::slice::from_raw_parts_mut(ptr.add(hi), num_cols);
                        lo_row.swap_with_slice(hi_row);
                    }
                }
            });
            return;
        }
    }

    for i in 0..n {
        let j = reverse_index(i, n as u64);
        if j > i {
            let lo = i * num_cols;
            let hi = j * num_cols;
            let (left, right) = buf.split_at_mut(hi);
            left[lo..lo + num_cols].swap_with_slice(&mut right[..num_cols]);
        }
    }
}

/// Batched row-major Bowers DIF FFT (forward).
///
/// Input layout: `buf` of length `n * num_cols`, row-major. Output is in
/// **bit-reversed row order** (the caller is expected to apply
/// [`in_place_bit_reverse_permute_row_major`] if natural order is needed,
/// matching `coset_lde_full_expand`'s convention).
///
/// `layer_twiddles` must have `log2(n)` layers, same as for the
/// single-column FFT.
#[cfg(feature = "alloc")]
pub fn bowers_fft_batch_row_major<F, E>(
    buf: &mut [FieldElement<E>],
    num_cols: usize,
    layer_twiddles: &LayerTwiddles<F>,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    if num_cols == 0 || buf.is_empty() {
        return Ok(());
    }
    let total = buf.len();
    if !total.is_multiple_of(num_cols) {
        return Err(FFTError::InputError(total));
    }
    let n = total / num_cols;
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

    #[cfg(feature = "parallel")]
    let parallel_threshold = adaptive_parallel_threshold();
    let m = num_cols;

    // DIF: layer goes 0..log_n. Mirror the single-column 3-layer fusion
    // structure (radix-8 first, then radix-4, then single-layer fallback).
    // For each fusion mode, blocks are independent and parallelizable.
    let mut layer = 0usize;

    while layer + 2 < log_n {
        let block_size_rows = n >> layer;
        let chunk_bytes = block_size_rows * m;
        let tw0 = layer_twiddles.get_layer(layer);
        let tw1 = layer_twiddles.get_layer(layer + 1);
        let tw2 = layer_twiddles.get_layer(layer + 2);
        let num_blocks = n / block_size_rows;

        #[cfg(feature = "parallel")]
        {
            if num_blocks >= parallel_threshold {
                buf.par_chunks_mut(chunk_bytes).for_each(|block| {
                    dif_fused3_block_row_major::<F, E>(block, m, tw0, tw1, tw2);
                });
                layer += 3;
                continue;
            }
        }
        for block in buf.chunks_mut(chunk_bytes) {
            dif_fused3_block_row_major::<F, E>(block, m, tw0, tw1, tw2);
        }
        layer += 3;
    }

    while layer + 1 < log_n {
        let block_size_rows = n >> layer;
        let chunk_bytes = block_size_rows * m;
        let tw0 = layer_twiddles.get_layer(layer);
        let tw1 = layer_twiddles.get_layer(layer + 1);
        let num_blocks = n / block_size_rows;

        #[cfg(feature = "parallel")]
        {
            if num_blocks >= parallel_threshold {
                buf.par_chunks_mut(chunk_bytes).for_each(|block| {
                    dif_fused2_block_row_major::<F, E>(block, m, tw0, tw1);
                });
                layer += 2;
                continue;
            }
        }
        for block in buf.chunks_mut(chunk_bytes) {
            dif_fused2_block_row_major::<F, E>(block, m, tw0, tw1);
        }
        layer += 2;
    }

    while layer < log_n {
        let block_size_rows = n >> layer;
        let chunk_bytes = block_size_rows * m;
        let twiddles = layer_twiddles.get_layer(layer);
        let num_blocks = n / block_size_rows;

        #[cfg(feature = "parallel")]
        {
            if num_blocks >= parallel_threshold {
                buf.par_chunks_mut(chunk_bytes).for_each(|block| {
                    dif_block_row_major::<F, E>(block, m, twiddles);
                });
                layer += 1;
                continue;
            }
        }
        for block in buf.chunks_mut(chunk_bytes) {
            dif_block_row_major::<F, E>(block, m, twiddles);
        }
        layer += 1;
    }

    Ok(())
}

/// Batched row-major Bowers DIT iFFT (inverse).
///
/// Input layout: `buf` of length `n * num_cols`, row-major in
/// **bit-reversed row order**. Output is in natural row order, matching
/// `bowers_ifft_opt_parallel`'s convention.
///
/// Use `LayerTwiddles::new_inverse(log_n)` as `layer_twiddles`. The final
/// 1/n scaling is the caller's responsibility (matches single-column path,
/// where the coset-weights step folds in the normalization).
#[cfg(feature = "alloc")]
pub fn bowers_ifft_batch_row_major<F, E>(
    buf: &mut [FieldElement<E>],
    num_cols: usize,
    layer_twiddles: &LayerTwiddles<F>,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    if num_cols == 0 || buf.is_empty() {
        return Ok(());
    }
    let total = buf.len();
    if !total.is_multiple_of(num_cols) {
        return Err(FFTError::InputError(total));
    }
    let n = total / num_cols;
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

    #[cfg(feature = "parallel")]
    let parallel_threshold = adaptive_parallel_threshold();
    let m = num_cols;

    // DIT: iterate layer log_n..0 (the "remaining_layer" index). Mirror the
    // single-column 3-layer fusion (radix-8 first, then radix-4, then single).
    let mut layer = log_n;

    while layer >= 3 {
        let layer_hi = layer - 1;
        let layer_mid = layer - 2;
        let layer_lo = layer - 3;
        let block_size_rows = n >> layer_lo;
        let chunk_bytes = block_size_rows * m;
        let tw_hi = layer_twiddles.get_layer(layer_hi);
        let tw_mid = layer_twiddles.get_layer(layer_mid);
        let tw_lo = layer_twiddles.get_layer(layer_lo);
        let num_blocks = n / block_size_rows;

        #[cfg(feature = "parallel")]
        {
            if num_blocks >= parallel_threshold {
                buf.par_chunks_mut(chunk_bytes).for_each(|block| {
                    dit_fused3_block_row_major::<F, E>(block, m, tw_hi, tw_mid, tw_lo);
                });
                layer -= 3;
                continue;
            }
        }
        for block in buf.chunks_mut(chunk_bytes) {
            dit_fused3_block_row_major::<F, E>(block, m, tw_hi, tw_mid, tw_lo);
        }
        layer -= 3;
    }

    while layer >= 2 {
        let layer_hi = layer - 1;
        let layer_lo = layer - 2;
        let block_size_rows = n >> layer_lo;
        let chunk_bytes = block_size_rows * m;
        let tw_hi = layer_twiddles.get_layer(layer_hi);
        let tw_lo = layer_twiddles.get_layer(layer_lo);
        let num_blocks = n / block_size_rows;

        #[cfg(feature = "parallel")]
        {
            if num_blocks >= parallel_threshold {
                buf.par_chunks_mut(chunk_bytes).for_each(|block| {
                    dit_fused2_block_row_major::<F, E>(block, m, tw_hi, tw_lo);
                });
                layer -= 2;
                continue;
            }
        }
        for block in buf.chunks_mut(chunk_bytes) {
            dit_fused2_block_row_major::<F, E>(block, m, tw_hi, tw_lo);
        }
        layer -= 2;
    }

    if layer >= 1 {
        let remaining_layer = layer - 1;
        let block_size_rows = n >> remaining_layer;
        let chunk_bytes = block_size_rows * m;
        let twiddles = layer_twiddles.get_layer(remaining_layer);
        let num_blocks = n / block_size_rows;

        #[cfg(feature = "parallel")]
        {
            if num_blocks >= parallel_threshold {
                buf.par_chunks_mut(chunk_bytes).for_each(|block| {
                    dit_block_row_major::<F, E>(block, m, twiddles);
                });
                return Ok(());
            }
        }
        for block in buf.chunks_mut(chunk_bytes) {
            dit_block_row_major::<F, E>(block, m, twiddles);
        }
    }

    Ok(())
}
