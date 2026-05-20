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
use crate::fft::cpu::bit_reversing::reverse_index;
#[cfg(feature = "alloc")]
use crate::fft::cpu::bowers_fft::LayerTwiddles;

#[cfg(all(feature = "alloc", feature = "parallel"))]
use rayon::prelude::*;

/// Threshold for engaging intra-block parallelism inside the butterfly loop.
/// Below this, even Rayon-aware blocks fall back to sequential per-row work.
#[cfg(all(feature = "alloc", feature = "parallel"))]
const INTRA_BLOCK_PAR_THRESHOLD: usize = 64;

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
    debug_assert!(buf.len() % num_cols == 0, "buf.len() must be a multiple of num_cols");
    let n = buf.len() / num_cols;
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "row count must be a power of two");

    #[cfg(feature = "parallel")]
    {
        // Collect the swap pairs first (each i with br(i) > i appears exactly
        // once). The set of indices touched across all pairs is a disjoint
        // union, so swaps don't race.
        let pairs: alloc::vec::Vec<(usize, usize)> = (0..n)
            .filter_map(|i| {
                let j = reverse_index(i, n as u64);
                (j > i).then_some((i * num_cols, j * num_cols))
            })
            .collect();

        if pairs.len() >= 1024 {
            use core::sync::atomic::{AtomicPtr, Ordering};
            // SAFETY: each (lo, hi) pair is disjoint from every other pair
            // (bit-reverse is a permutation, so distinct sources map to
            // distinct destinations), and lo..lo+M / hi..hi+M never overlap
            // since lo != hi. We share a raw pointer across threads but
            // each thread writes to a unique pair of M-wide row ranges.
            let raw = AtomicPtr::new(buf.as_mut_ptr());
            pairs.par_iter().for_each(|&(lo, hi)| {
                let ptr = raw.load(Ordering::Relaxed);
                unsafe {
                    let lo_row = core::slice::from_raw_parts_mut(ptr.add(lo), num_cols);
                    let hi_row = core::slice::from_raw_parts_mut(ptr.add(hi), num_cols);
                    lo_row.swap_with_slice(hi_row);
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
    if total % num_cols != 0 {
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

    // DIF: layer goes 0..log_n. At layer L there are `n >> L` rows per
    // block and `n / block_size_rows` independent blocks. We chunk the
    // row-major buffer into `block_size_rows * m`-wide slices and run the
    // butterfly per block (parallel when enough blocks).
    for layer in 0..log_n {
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
                continue;
            }
        }
        for block in buf.chunks_mut(chunk_bytes) {
            dif_block_row_major::<F, E>(block, m, twiddles);
        }
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
    if total % num_cols != 0 {
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

    // DIT: layer iterates log_n..0 (the "remaining_layer" index).
    for remaining_layer in (0..log_n).rev() {
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
                continue;
            }
        }
        for block in buf.chunks_mut(chunk_bytes) {
            dit_block_row_major::<F, E>(block, m, twiddles);
        }
    }

    Ok(())
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
    use crate::fft::cpu::bowers_fft::{bowers_fft_opt_fused, bowers_ifft_opt, LayerTwiddles};
    use crate::field::goldilocks::GoldilocksField;

    type F = GoldilocksField;
    type FE = FieldElement<F>;

    /// Build a row-major buffer of size n*m from column-major input.
    fn col_major_to_row_major(cols: &[Vec<FE>]) -> (Vec<FE>, usize) {
        let m = cols.len();
        if m == 0 {
            return (Vec::new(), 0);
        }
        let n = cols[0].len();
        let mut row_major = vec![FE::zero(); n * m];
        for r in 0..n {
            for c in 0..m {
                row_major[r * m + c] = cols[c][r].clone();
            }
        }
        (row_major, m)
    }

    /// Transpose a row-major flat buffer back into M column vectors.
    fn row_major_to_col_major(buf: &[FE], m: usize) -> Vec<Vec<FE>> {
        if m == 0 {
            return Vec::new();
        }
        let n = buf.len() / m;
        let mut cols: Vec<Vec<FE>> = (0..m).map(|_| Vec::with_capacity(n)).collect();
        for r in 0..n {
            for c in 0..m {
                cols[c].push(buf[r * m + c].clone());
            }
        }
        cols
    }

    fn sample_columns(n: usize, m: usize, seed: u64) -> Vec<Vec<FE>> {
        (0..m)
            .map(|c| {
                (0..n)
                    .map(|r| FE::from(seed.wrapping_add((c as u64) * 1_000_003 + r as u64)))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn bit_reverse_row_major_matches_single_column_per_column() {
        for log_n in 1..6 {
            let n = 1usize << log_n;
            for m in [1usize, 2, 3, 4, 5, 8] {
                let cols = sample_columns(n, m, 42);
                // Apply single-column bit-reverse to each column independently.
                let mut expected_cols = cols.clone();
                for c in expected_cols.iter_mut() {
                    in_place_bit_reverse_permute(c);
                }
                // Apply row-major bit-reverse to the flat buffer.
                let (mut row_major, _) = col_major_to_row_major(&cols);
                in_place_bit_reverse_permute_row_major(&mut row_major, m);
                let actual_cols = row_major_to_col_major(&row_major, m);
                assert_eq!(actual_cols, expected_cols, "log_n={log_n} m={m}");
            }
        }
    }

    #[test]
    fn batched_fft_matches_single_column_fft() {
        for log_n in 1..7 {
            let n = 1usize << log_n;
            let tw = LayerTwiddles::<F>::new(log_n as u64).unwrap();
            for m in [1usize, 2, 3, 5, 8] {
                let cols = sample_columns(n, m, 7);
                let mut expected = cols.clone();
                for col in expected.iter_mut() {
                    bowers_fft_opt_fused::<F, F>(col, &tw).unwrap();
                }

                let (mut row_major, _) = col_major_to_row_major(&cols);
                bowers_fft_batch_row_major::<F, F>(&mut row_major, m, &tw).unwrap();
                let actual = row_major_to_col_major(&row_major, m);
                assert_eq!(actual, expected, "log_n={log_n} m={m}");
            }
        }
    }

    #[test]
    fn batched_ifft_matches_single_column_ifft() {
        for log_n in 1..7 {
            let n = 1usize << log_n;
            let tw = LayerTwiddles::<F>::new_inverse(log_n as u64).unwrap();
            for m in [1usize, 2, 3, 5, 8] {
                let cols = sample_columns(n, m, 11);
                let mut expected = cols.clone();
                for col in expected.iter_mut() {
                    bowers_ifft_opt::<F, F>(col, &tw).unwrap();
                }

                let (mut row_major, _) = col_major_to_row_major(&cols);
                bowers_ifft_batch_row_major::<F, F>(&mut row_major, m, &tw).unwrap();
                let actual = row_major_to_col_major(&row_major, m);
                assert_eq!(actual, expected, "log_n={log_n} m={m}");
            }
        }
    }

    #[test]
    fn fft_then_ifft_round_trip_batched() {
        for log_n in 1..7 {
            let n = 1usize << log_n;
            let n_inv = FE::from(n as u64).inv().unwrap();
            let fwd = LayerTwiddles::<F>::new(log_n as u64).unwrap();
            let inv = LayerTwiddles::<F>::new_inverse(log_n as u64).unwrap();

            for m in [1usize, 2, 4] {
                let cols = sample_columns(n, m, 13);
                let original = cols.clone();
                let (mut buf, _) = col_major_to_row_major(&cols);

                // Forward: natural -> bit-reversed
                bowers_fft_batch_row_major::<F, F>(&mut buf, m, &fwd).unwrap();
                in_place_bit_reverse_permute_row_major(&mut buf, m);

                // Inverse: bit-reversed -> natural
                in_place_bit_reverse_permute_row_major(&mut buf, m);
                bowers_ifft_batch_row_major::<F, F>(&mut buf, m, &inv).unwrap();
                for x in buf.iter_mut() {
                    *x = &*x * &n_inv;
                }

                let recovered = row_major_to_col_major(&buf, m);
                assert_eq!(recovered, original, "log_n={log_n} m={m}");
            }
        }
    }
}
