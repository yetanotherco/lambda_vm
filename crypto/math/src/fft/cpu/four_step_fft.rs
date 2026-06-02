//! PROTOTYPE: cache-blocked four-step (six-step) batched forward FFT.
//!
//! Investigation of the trace-LDE FFT gap vs Plonky3 (which is cache-bound:
//! the flat Bowers DIF streams the whole `n·m` buffer with large strides at
//! the early layers, thrashing cache for large `n`). This computes the same
//! forward DFT by decomposing `N = N1·N2` and running each sub-FFT on a
//! cache-resident block (the existing `bowers_fft_batch_row_major` kernel),
//! with three tiled transposes in between. Output is natural order, identical
//! to `bowers_fft_batch_row_major` + `in_place_bit_reverse_permute_row_major`.
//!
//! This is a measurement prototype: if it beats the flat Bowers at large `n`,
//! the cache hypothesis is confirmed and a transpose-free port of P3's
//! two-half (bit-reversal-based) is the production target. Not wired into the
//! prover.

#[cfg(feature = "alloc")]
use crate::fft::cpu::bowers_fft::LayerTwiddles;
#[cfg(feature = "alloc")]
use crate::fft::cpu::bowers_fft_batch::{
    bowers_fft_batch_row_major, in_place_bit_reverse_permute_row_major,
};
#[cfg(feature = "alloc")]
use crate::fft::errors::FFTError;
#[cfg(feature = "alloc")]
use crate::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};
#[cfg(all(feature = "alloc", feature = "parallel"))]
use rayon::prelude::*;

/// Tiled transpose of an `rows × cols` grid whose elements are `m`-vectors,
/// stored row-major. Writes `output[(b*rows + a)] = input[(a*cols + b)]`
/// (per `m`-vector). Blocked by `TILE` rows/cols for cache locality.
#[cfg(feature = "alloc")]
fn transpose_grid_row_major<E: IsField>(
    input: &[FieldElement<E>],
    output: &mut [FieldElement<E>],
    rows: usize,
    cols: usize,
    m: usize,
) {
    const TILE: usize = 16;
    for a0 in (0..rows).step_by(TILE) {
        let a_end = (a0 + TILE).min(rows);
        for b0 in (0..cols).step_by(TILE) {
            let b_end = (b0 + TILE).min(cols);
            for a in a0..a_end {
                for b in b0..b_end {
                    let src = (a * cols + b) * m;
                    let dst = (b * rows + a) * m;
                    output[dst..dst + m].clone_from_slice(&input[src..src + m]);
                }
            }
        }
    }
}

/// Apply a natural-order forward FFT (Bowers + bit-reverse) to each contiguous
/// block of `block_rows` rows. Blocks are independent → parallel over blocks.
#[cfg(feature = "alloc")]
fn fft_natural_blocks<F, E>(
    buf: &mut [FieldElement<E>],
    block_rows: usize,
    m: usize,
    tw: &LayerTwiddles<F>,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    let chunk = block_rows * m;
    #[cfg(feature = "parallel")]
    {
        buf.par_chunks_mut(chunk).try_for_each(|block| {
            bowers_fft_batch_row_major::<F, E>(block, m, tw)?;
            in_place_bit_reverse_permute_row_major(block, m);
            Ok::<(), FFTError>(())
        })?;
    }
    #[cfg(not(feature = "parallel"))]
    {
        for block in buf.chunks_mut(chunk) {
            bowers_fft_batch_row_major::<F, E>(block, m, tw)?;
            in_place_bit_reverse_permute_row_major(block, m);
        }
    }
    Ok(())
}

/// Cache-blocked four-step batched forward FFT. `buf` is `n * num_cols`
/// row-major (`n` rows of `num_cols` consecutive elements). Output is the
/// natural-order forward DFT (matches `bowers_fft_batch_row_major` followed by
/// `in_place_bit_reverse_permute_row_major`).
#[cfg(feature = "alloc")]
pub fn fft_batch_four_step<F, E>(
    buf: &mut [FieldElement<E>],
    num_cols: usize,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    let m = num_cols;
    if m == 0 || buf.is_empty() {
        return Ok(());
    }
    let total = buf.len();
    if !total.is_multiple_of(m) {
        return Err(FFTError::InputError(total));
    }
    let n = total / m;
    if !n.is_power_of_two() {
        return Err(FFTError::InputError(n));
    }
    let log_n = n.trailing_zeros() as usize;
    if log_n <= 1 {
        return Ok(());
    }

    // For small n the flat kernel is already cache-resident; just use it.
    if log_n < 4 {
        let tw = LayerTwiddles::<F>::new(log_n as u64).ok_or(FFTError::InputError(n))?;
        bowers_fft_batch_row_major::<F, E>(buf, m, &tw)?;
        in_place_bit_reverse_permute_row_major(buf, m);
        return Ok(());
    }

    let log_n1 = log_n / 2;
    let log_n2 = log_n - log_n1;
    let n1 = 1usize << log_n1;
    let n2 = 1usize << log_n2;

    let tw_n1 = LayerTwiddles::<F>::new(log_n1 as u64).ok_or(FFTError::InputError(n1))?;
    let tw_n2 = LayerTwiddles::<F>::new(log_n2 as u64).ok_or(FFTError::InputError(n2))?;
    // ω_N, the forward N-th root, for the inter-pass twiddles ω_N^{n1·k2}.
    let omega_n =
        F::get_primitive_root_of_unity(log_n as u64).map_err(|_| FFTError::InputError(n))?;

    let mut scratch: Vec<FieldElement<E>> = vec![FieldElement::<E>::zero(); total];

    // Initial view: A[n2][n1] = x[n2·N1 + n1]  (N2 blocks of N1 contiguous rows).
    // Step 1: transpose A(N2×N1) → scratch B[n1·N2 + n2] = x[n2·N1 + n1].
    transpose_grid_row_major(buf, &mut scratch, n2, n1, m);

    // Step 2: N1 blocks of N2 rows → size-N2 FFT each. scratch[n1·N2 + k2] = C[n1][k2].
    fft_natural_blocks::<F, E>(&mut scratch, n2, m, &tw_n2)?;

    // Step 3: twiddle C[n1][k2] *= ω_N^{n1·k2}. Per n1-block, geometric in k2.
    #[cfg(feature = "parallel")]
    let blocks = scratch.par_chunks_mut(n2 * m).enumerate();
    #[cfg(not(feature = "parallel"))]
    let blocks = scratch.chunks_mut(n2 * m).enumerate();
    blocks.for_each(|(n1_idx, block)| {
        let base = omega_n.pow(n1_idx as u64); // ω_N^{n1}
        let mut tw = FieldElement::<F>::one(); // (ω_N^{n1})^{k2}
        for k2 in 0..n2 {
            if k2 > 0 {
                tw = &tw * &base;
            }
            let row = &mut block[k2 * m..k2 * m + m];
            for x in row.iter_mut() {
                *x = &tw * &*x;
            }
        }
    });

    // Step 4: transpose B(N1×N2) → buf D[k2·N1 + n1] = C[n1][k2].
    transpose_grid_row_major(&scratch, buf, n1, n2, m);

    // Step 5: N2 blocks of N1 rows → size-N1 FFT each.
    //         buf[k2·N1 + k1] = E[k2][k1] = X[N2·k1 + k2].
    fft_natural_blocks::<F, E>(buf, n1, m, &tw_n1)?;

    // Step 6: transpose D'(N2×N1) → scratch[k1·N2 + k2] = X[N2·k1 + k2] (natural).
    transpose_grid_row_major(buf, &mut scratch, n2, n1, m);
    buf.clone_from_slice(&scratch);

    Ok(())
}

// ----------------------------------------------------------------------------
// Transpose-free two-half (P3-style) cache-blocked forward FFT.
// ----------------------------------------------------------------------------

use crate::fft::cpu::bit_reversing::reverse_index;

/// In-place bit-reversal permutation of a flat slice (length a power of two).
#[cfg(feature = "alloc")]
fn bit_reverse_vec<F: IsField>(v: &mut [FieldElement<F>]) {
    let n = v.len();
    for i in 0..n {
        let j = reverse_index(i, n as u64);
        if j > i {
            v.swap(i, j);
        }
    }
}

/// DIT butterfly over two equal-length row-slices, one twiddle for all pairs:
/// `a' = a + tw·b`, `b' = a − tw·b` (element-wise; `tw·b` is the F×E multiply).
#[cfg(feature = "alloc")]
#[inline]
fn dit_butterfly_rows<F, E>(
    lo: &mut [FieldElement<E>],
    hi: &mut [FieldElement<E>],
    tw: &FieldElement<F>,
) where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
        let t = tw * &*b; // F × E → E
        let new_a = &*a + &t;
        *b = &*a - &t;
        *a = new_a;
    }
}

/// First-half DIT layer (per-pair twiddle), applied within one cache-resident
/// row-chunk. `tw` is the flat `[ω^0..ω^(n/2−1)]` array; pair `j` of layer
/// `layer` uses `tw[j · 2^(log_n−1−layer)]`.
#[cfg(feature = "alloc")]
fn dit_first_half_layer<F, E>(
    chunk: &mut [FieldElement<E>],
    m: usize,
    layer: usize,
    log_n: usize,
    tw: &[FieldElement<F>],
) where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let half = 1usize << layer;
    let block_rows = half * 2;
    let step = 1usize << (log_n - 1 - layer);
    for block in chunk.chunks_mut(block_rows * m) {
        let (lows, highs) = block.split_at_mut(half * m);
        for j in 0..half {
            let twj = &tw[j * step];
            dit_butterfly_rows(
                &mut lows[j * m..j * m + m],
                &mut highs[j * m..j * m + m],
                twj,
            );
        }
    }
}

/// Second-half DIT layer (one twiddle per block, bit-reversed twiddle order),
/// applied within one cache-resident row-chunk owned by `thread`.
#[cfg(feature = "alloc")]
fn dit_second_half_layer<F, E>(
    chunk: &mut [FieldElement<E>],
    m: usize,
    layer: usize,
    log_n: usize,
    mid: usize,
    thread: usize,
    bitrev_tw: &[FieldElement<F>],
) where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let half_block = 1usize << (log_n - 1 - layer);
    let block_rows = half_block * 2;
    let first_block = thread << (layer - mid);
    for (b, block) in chunk.chunks_mut(block_rows * m).enumerate() {
        let twb = &bitrev_tw[first_block + b];
        let (lows, highs) = block.split_at_mut(half_block * m);
        dit_butterfly_rows(lows, highs, twb);
    }
}

/// Cache-blocked, transpose-free batched forward FFT (port of Plonky3's
/// two-half `Radix2DitParallel::dft_batch`). Natural-order output (matches
/// `bowers_fft_batch_row_major` + `in_place_bit_reverse_permute_row_major`).
///
/// Bit-reverse → first `mid` DIT layers within `2^mid`-row chunks → bit-reverse
/// → remaining layers within `2^(log_n−mid)`-row chunks → bit-reverse. The
/// bit-reversals turn the large-stride butterflies into chunk-local ones, so
/// every layer stays cache-resident (the win that the flat Bowers misses).
#[cfg(feature = "alloc")]
fn two_half_core<F, E>(
    buf: &mut [FieldElement<E>],
    num_cols: usize,
    inverse: bool,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    let m = num_cols;
    if m == 0 || buf.is_empty() {
        return Ok(());
    }
    let total = buf.len();
    if !total.is_multiple_of(m) {
        return Err(FFTError::InputError(total));
    }
    let n = total / m;
    if !n.is_power_of_two() {
        return Err(FFTError::InputError(n));
    }
    let log_n = n.trailing_zeros() as usize;
    if log_n == 0 {
        return Ok(());
    }

    let fwd = F::get_primitive_root_of_unity(log_n as u64).map_err(|_| FFTError::InputError(n))?;
    let omega = if inverse {
        fwd.inv().map_err(|_| FFTError::InputError(n))?
    } else {
        fwd
    };
    let half = n / 2;
    let mut tw: Vec<FieldElement<F>> = Vec::with_capacity(half);
    let mut cur = FieldElement::<F>::one();
    for _ in 0..half {
        tw.push(cur.clone());
        cur = &cur * &omega;
    }
    let mut bitrev_tw = tw.clone();
    bit_reverse_vec(&mut bitrev_tw);

    let mid = log_n.div_ceil(2);

    // Step 1: bit-reverse rows.
    in_place_bit_reverse_permute_row_major(buf, m);

    // Step 2: first half — layers 0..mid within 2^mid-row chunks (all identical).
    let first_chunk = (1usize << mid) * m;
    #[cfg(feature = "parallel")]
    let it = buf.par_chunks_mut(first_chunk);
    #[cfg(not(feature = "parallel"))]
    let it = buf.chunks_mut(first_chunk);
    it.for_each(|chunk| {
        for layer in 0..mid {
            dit_first_half_layer::<F, E>(chunk, m, layer, log_n, &tw);
        }
    });

    // Step 3: bit-reverse rows.
    in_place_bit_reverse_permute_row_major(buf, m);

    // Step 4: second half — layers mid..log_n within 2^(log_n-mid)-row chunks.
    let second_chunk = (1usize << (log_n - mid)) * m;
    #[cfg(feature = "parallel")]
    let it2 = buf.par_chunks_mut(second_chunk).enumerate();
    #[cfg(not(feature = "parallel"))]
    let it2 = buf.chunks_mut(second_chunk).enumerate();
    it2.for_each(|(thread, chunk)| {
        for layer in mid..log_n {
            dit_second_half_layer::<F, E>(chunk, m, layer, log_n, mid, thread, &bitrev_tw);
        }
    });

    // Step 5: final bit-reverse to natural order.
    in_place_bit_reverse_permute_row_major(buf, m);

    Ok(())
}

/// Cache-blocked, transpose-free batched **forward** FFT (natural-order output;
/// matches `bowers_fft_batch_row_major` + `in_place_bit_reverse_permute_row_major`).
#[cfg(feature = "alloc")]
pub fn fft_batch_two_half<F, E>(
    buf: &mut [FieldElement<E>],
    num_cols: usize,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    two_half_core::<F, E>(buf, num_cols, false)
}

/// Cache-blocked, transpose-free batched **inverse** FFT, WITHOUT the 1/n scale
/// (matches `in_place_bit_reverse_permute_row_major` + `bowers_ifft_batch_row_major`;
/// the 1/n normalization is the caller's responsibility, e.g. folded into the
/// coset-weight pass of the LDE).
#[cfg(feature = "alloc")]
pub fn ifft_batch_two_half<F, E>(
    buf: &mut [FieldElement<E>],
    num_cols: usize,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    two_half_core::<F, E>(buf, num_cols, true)
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::field::goldilocks::GoldilocksField;

    type F = GoldilocksField;

    fn reference_natural_fft(buf: &mut [FieldElement<F>], m: usize, log_n: usize) {
        let tw = LayerTwiddles::<F>::new(log_n as u64).unwrap();
        bowers_fft_batch_row_major::<F, F>(buf, m, &tw).unwrap();
        in_place_bit_reverse_permute_row_major(buf, m);
    }

    // Mirrors the LDE's iFFT: bit-reverse then flat Bowers inverse (no 1/n).
    fn reference_natural_ifft(buf: &mut [FieldElement<F>], m: usize, log_n: usize) {
        use crate::fft::cpu::bowers_fft_batch::bowers_ifft_batch_row_major;
        let tw = LayerTwiddles::<F>::new_inverse(log_n as u64).unwrap();
        in_place_bit_reverse_permute_row_major(buf, m);
        bowers_ifft_batch_row_major::<F, F>(buf, m, &tw).unwrap();
    }

    fn sample(n: usize, m: usize) -> Vec<FieldElement<F>> {
        (0..n * m)
            .map(|i| FieldElement::<F>::from((i as u64).wrapping_mul(2654435761) ^ 0x9e37))
            .collect()
    }

    #[test]
    fn four_step_matches_flat_bowers() {
        for log_n in [2usize, 3, 4, 5, 6, 8, 10] {
            for m in [1usize, 3, 7] {
                let n = 1 << log_n;
                let input = sample(n, m);
                let mut a = input.clone();
                let mut b = input.clone();
                let mut c = input.clone();
                reference_natural_fft(&mut a, m, log_n);
                fft_batch_four_step::<F, F>(&mut b, m).unwrap();
                fft_batch_two_half::<F, F>(&mut c, m).unwrap();
                assert_eq!(a, b, "four_step mismatch at log_n={log_n}, m={m}");
                assert_eq!(a, c, "two_half mismatch at log_n={log_n}, m={m}");

                let mut d = input.clone();
                let mut e = input.clone();
                reference_natural_ifft(&mut d, m, log_n);
                ifft_batch_two_half::<F, F>(&mut e, m).unwrap();
                assert_eq!(d, e, "ifft_two_half mismatch at log_n={log_n}, m={m}");
            }
        }
    }

    /// Timing micro-bench (run with `--release --ignored --nocapture`).
    #[test]
    #[ignore]
    fn bench_four_step_vs_flat() {
        use std::time::Instant;
        let m = 64;
        for log_n in [20usize, 21, 22, 23] {
            let n = 1 << log_n;
            let input = sample(n, m);
            let tw = LayerTwiddles::<F>::new(log_n as u64).unwrap();

            let runs = 5;
            let mut t_flat = f64::INFINITY;
            let mut t_four = f64::INFINITY;
            let mut t_two = f64::INFINITY;
            for _ in 0..runs {
                let mut a = input.clone();
                let s = Instant::now();
                bowers_fft_batch_row_major::<F, F>(&mut a, m, &tw).unwrap();
                in_place_bit_reverse_permute_row_major(&mut a, m);
                t_flat = t_flat.min(s.elapsed().as_secs_f64());

                let mut b = input.clone();
                let s = Instant::now();
                fft_batch_four_step::<F, F>(&mut b, m).unwrap();
                t_four = t_four.min(s.elapsed().as_secs_f64());

                let mut c = input.clone();
                let s = Instant::now();
                fft_batch_two_half::<F, F>(&mut c, m).unwrap();
                t_two = t_two.min(s.elapsed().as_secs_f64());
            }
            println!(
                "log_n={log_n} m={m}: flat={:.4}s four_step={:.4}s two_half={:.4}s  four/flat={:.2}x two/flat={:.2}x",
                t_flat,
                t_four,
                t_two,
                t_flat / t_four,
                t_flat / t_two
            );
        }
    }
}
