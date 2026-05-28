//! Phased Bowers FFT — 2-phase Bailey four-step prototype.
//!
//! This is a probe to evaluate whether a phased / cache-blocked NTT
//! beats the current single-pass `bowers_fft_opt_fused_parallel` for
//! large `N` (especially `log_n >= 22` where the working set spills out
//! of L3 and the naive layer-by-layer Bowers pattern becomes
//! memory-bound).
//!
//! # Approach
//!
//! The standard Bailey four-step. View the input as an `M × K` row-major
//! matrix with `N = M·K` and split log_n roughly in half:
//!
//! 1. Transpose input → `K × M`.
//! 2. FFT_M on each of the K rows (reuses existing Bowers fused
//!    kernels; each row fits in L1).
//! 3. Multiply pointwise by `ω_N^(s·d)` where `s` is the row index
//!    `[0, K)` and `d` the column index `[0, M)`.
//! 4. Transpose → `M × K`.
//! 5. FFT_K on each of the M rows.
//! 6. Final transpose → linear output `Y[0], …, Y[N-1]`.
//!
//! Phase twiddles are computed on-the-fly (no upfront `N`-sized table)
//! — for `log_n = 26` the inter-phase twiddle table would cost ~1 GB
//! otherwise. We do `K` powers `ω_K^s` and step the accumulator
//! inside the row, one mul per element.
//!
//! Output convention: **natural order**. Equivalent to
//! `bowers_fft_opt_fused_parallel(x, ...); in_place_bit_reverse_permute(x)`.
//!
//! # Status
//!
//! Prototype. Correctness verified by equivalence tests against the
//! existing Bowers path; performance measured by `phased_fft_bench`.
//! Not yet wired into the prover / polynomial API.

use crate::fft::bit_reversing::in_place_bit_reverse_permute;
use crate::fft::bowers_fft::{LayerTwiddles, bowers_fft_opt_fused};
use crate::fft::errors::FFTError;
use crate::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};
use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Tile size for the out-of-place transpose. 32 × 32 × 8 B = 8 KB per
/// tile fits the L1 of Goldilocks base-field elements (extension-field
/// tiles are ~3× larger but still fit).
const TRANSPOSE_TILE: usize = 32;

/// Out-of-place tiled transpose.
///
/// `input` is `rows × cols` row-major; `output` becomes `cols × rows`
/// row-major. Tile-based for cache locality: each tile pair is
/// `TRANSPOSE_TILE × TRANSPOSE_TILE` elements, sized to fit L1.
pub fn tiled_transpose<T: Clone>(input: &[T], output: &mut [T], rows: usize, cols: usize) {
    debug_assert_eq!(input.len(), rows * cols);
    debug_assert_eq!(output.len(), rows * cols);

    for tile_r in (0..rows).step_by(TRANSPOSE_TILE) {
        let r_end = (tile_r + TRANSPOSE_TILE).min(rows);
        for tile_c in (0..cols).step_by(TRANSPOSE_TILE) {
            let c_end = (tile_c + TRANSPOSE_TILE).min(cols);
            for r in tile_r..r_end {
                for c in tile_c..c_end {
                    output[c * rows + r] = input[r * cols + c].clone();
                }
            }
        }
    }
}

/// Pre-built phased-FFT twiddle context. Build once per `(log_n)` pair,
/// reuse across many invocations to amortise the LayerTwiddles
/// allocation + the `ω_N` lookup.
pub struct PhasedFftContext<F: IsField + IsFFTField> {
    pub log_n: usize,
    pub log_m: usize,
    pub log_k: usize,
    pub twiddles_m: LayerTwiddles<F>,
    /// `None` when `log_k == log_m` — caller reuses `twiddles_m`.
    pub twiddles_k: Option<LayerTwiddles<F>>,
    pub omega_n: FieldElement<F>,
}

impl<F: IsFFTField> PhasedFftContext<F> {
    pub fn new(log_n: usize) -> Result<Self, FFTError> {
        let n = 1usize << log_n;
        if log_n < 4 {
            return Err(FFTError::InputError(n));
        }
        let log_m = log_n.div_ceil(2);
        let log_k = log_n - log_m;
        let twiddles_m = LayerTwiddles::<F>::new(log_m as u64)
            .ok_or(FFTError::InputError(1usize << log_m))?;
        let twiddles_k = if log_k == log_m {
            None
        } else {
            Some(
                LayerTwiddles::<F>::new(log_k as u64)
                    .ok_or(FFTError::InputError(1usize << log_k))?,
            )
        };
        let omega_n = F::get_primitive_root_of_unity(log_n as u64)
            .map_err(|_| FFTError::InputError(n))?;
        Ok(Self {
            log_n,
            log_m,
            log_k,
            twiddles_m,
            twiddles_k,
            omega_n,
        })
    }

    #[inline]
    fn twiddles_k_ref(&self) -> &LayerTwiddles<F> {
        self.twiddles_k.as_ref().unwrap_or(&self.twiddles_m)
    }
}

/// 2-phase Bailey four-step FFT.
///
/// Splits the `N = 2^log_n` problem into two cache-friendly FFTs of
/// sizes `M = 2^⌈log_n/2⌉` and `K = 2^⌊log_n/2⌋`, with a pointwise
/// phase-twiddle multiply between them.
///
/// Output: natural order (equivalent to
/// `bowers_fft_opt_fused_parallel` followed by
/// `in_place_bit_reverse_permute`).
///
/// For `log_n < 4` falls back to a single-pass Bowers (the phased
/// machinery has fixed overhead that doesn't pay back at tiny sizes).
///
/// Convenience wrapper that builds the `PhasedFftContext` per call.
/// Hot-path callers should prefer [`bowers_phased_fft_with_context`].
///
/// # Errors
/// Returns `FFTError::InputError` if `input.len()` is not a power of
/// two, or if the field lacks a primitive root of unity of the
/// required order.
pub fn bowers_phased_fft<F, E>(input: &mut [FieldElement<E>]) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
    FieldElement<F>: Send + Sync,
    FieldElement<E>: Send + Sync + Clone,
{
    let n = input.len();
    if !n.is_power_of_two() {
        return Err(FFTError::InputError(n));
    }
    if n <= 1 {
        return Ok(());
    }
    let log_n = n.trailing_zeros() as usize;

    if log_n < 4 {
        let twiddles = LayerTwiddles::<F>::new(log_n as u64).ok_or(FFTError::InputError(n))?;
        bowers_fft_opt_fused(input, &twiddles)?;
        in_place_bit_reverse_permute(input);
        return Ok(());
    }

    let ctx = PhasedFftContext::<F>::new(log_n)?;
    bowers_phased_fft_with_context(input, &ctx)
}

/// 2-phase Bailey four-step FFT using a pre-built twiddle context.
///
/// Skips the inner-twiddle and `ω_N` setup that
/// [`bowers_phased_fft`] pays per call. Caller must size `input` to
/// `1 << ctx.log_n`. Allocates an internal `N`-element scratch buffer
/// per call — for repeated calls (multi-column / multi-poly), use
/// [`bowers_phased_fft_with_buf`] instead.
pub fn bowers_phased_fft_with_context<F, E>(
    input: &mut [FieldElement<E>],
    ctx: &PhasedFftContext<F>,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
    FieldElement<F>: Send + Sync,
    FieldElement<E>: Send + Sync + Clone,
{
    let mut buf: Vec<FieldElement<E>> = Vec::new();
    bowers_phased_fft_with_buf(input, ctx, &mut buf)
}

/// 2-phase Bailey four-step FFT, reusing a caller-supplied scratch
/// buffer. The buffer is resized to `1 << ctx.log_n` on demand;
/// repeat callers can hold a long-lived `Vec` and pay only the
/// initial allocation.
pub fn bowers_phased_fft_with_buf<F, E>(
    input: &mut [FieldElement<E>],
    ctx: &PhasedFftContext<F>,
    buf: &mut Vec<FieldElement<E>>,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
    FieldElement<F>: Send + Sync,
    FieldElement<E>: Send + Sync + Clone,
{
    let n = 1usize << ctx.log_n;
    if input.len() != n {
        return Err(FFTError::InputError(input.len()));
    }
    let m = 1usize << ctx.log_m;
    let k = 1usize << ctx.log_k;
    let twiddles_m = &ctx.twiddles_m;
    let twiddles_k_ref = ctx.twiddles_k_ref();
    let omega_n = &ctx.omega_n;

    if buf.len() < n {
        buf.resize(n, FieldElement::<E>::zero());
    }
    let buf: &mut [FieldElement<E>] = &mut buf.as_mut_slice()[..n];

    // ===== Step 1: transpose input (M × K) → buf (K × M) =====
    tiled_transpose(input, buf, m, k);

    // ===== Step 2: FFT_M on each of K rows in buf =====
    #[cfg(feature = "parallel")]
    {
        buf.par_chunks_mut(m).try_for_each(|row| -> Result<(), FFTError> {
            bowers_fft_opt_fused(row, &twiddles_m)?;
            in_place_bit_reverse_permute(row);
            Ok(())
        })?;
    }
    #[cfg(not(feature = "parallel"))]
    {
        for row in buf.chunks_exact_mut(m) {
            bowers_fft_opt_fused(row, &twiddles_m)?;
            in_place_bit_reverse_permute(row);
        }
    }

    // ===== Step 3: phase twiddle multiply =====
    // For row s ∈ [0, K), col d ∈ [0, M):
    //   buf[s·M + d] *= ω_N^(s·d).
    // We compute one row at a time: per row s, ω_s = ω_N^s, and step
    // the accumulator by ω_s as d increases. K rows are independent,
    // parallelise across them.
    apply_phase_twiddles::<F, E>(buf, k, m, omega_n);

    // ===== Step 4: transpose buf (K × M) → input (M × K) =====
    tiled_transpose(buf, input, k, m);

    // ===== Step 5: FFT_K on each of M rows in input =====
    #[cfg(feature = "parallel")]
    {
        input
            .par_chunks_mut(k)
            .try_for_each(|row| -> Result<(), FFTError> {
                bowers_fft_opt_fused(row, twiddles_k_ref)?;
                in_place_bit_reverse_permute(row);
                Ok(())
            })?;
    }
    #[cfg(not(feature = "parallel"))]
    {
        for row in input.chunks_exact_mut(k) {
            bowers_fft_opt_fused(row, twiddles_k_ref)?;
            in_place_bit_reverse_permute(row);
        }
    }

    // ===== Step 6: final transpose input (M × K) → buf (K × M) =====
    // This puts the result in natural order: buf[k_2·M + k_1] holds
    // Y[k_1·K + k_2] = Y[linear index]; verify with the derivation in
    // the module docstring.
    //
    // Equivalent statement: input was M × K with row k_1 containing
    // Y[k_1·K + 0], Y[k_1·K + 1], …, Y[k_1·K + K-1]. We want Y[0],
    // Y[1], …, Y[N-1] contiguous, which is the same data read
    // "down columns then across rows" — i.e., a transpose to K × M.
    //
    // Actually re-deriving from the Bailey identity with our index
    // choice (j = r·K + c, k = u·M + v):
    //   After step 5 row k_1 = v contains Y[u·M + v] for u ∈ [0, K).
    //   That's NOT linear order (linear would put Y[0]..Y[K-1] in row 0
    //   of an M × K matrix). Need one more transpose.
    tiled_transpose(input, buf, m, k);
    input.clone_from_slice(buf);

    Ok(())
}

/// Apply the inter-phase twiddle correction in-place.
///
/// `buf` is `rows × cols` row-major; `omega_n` is the primitive `N`-th
/// root of unity where `N = rows · cols`. Multiplies `buf[s·cols + d]`
/// by `ω_N^(s·d)` for `s ∈ [0, rows)`, `d ∈ [0, cols)`.
///
/// Rows are independent so the loop parallelises across `s`.
fn apply_phase_twiddles<F, E>(
    buf: &mut [FieldElement<E>],
    rows: usize,
    cols: usize,
    omega_n: &FieldElement<F>,
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
    FieldElement<F>: Send + Sync,
    FieldElement<E>: Send + Sync,
{
    debug_assert_eq!(buf.len(), rows * cols);

    #[cfg(feature = "parallel")]
    let row_iter = buf.par_chunks_mut(cols).enumerate();
    #[cfg(not(feature = "parallel"))]
    let row_iter = buf.chunks_exact_mut(cols).enumerate();

    row_iter.for_each(|(s, row)| {
        if s == 0 {
            // ω_N^0 = 1; row 0 is identity — nothing to do.
            return;
        }
        // ω_s = ω_N^s. Step the accumulator by ω_s for each successive d.
        let omega_s = omega_n.pow(s as u64);
        let mut accum = FieldElement::<F>::one();
        for slot in row.iter_mut() {
            *slot = &accum * &*slot;
            accum = &accum * &omega_s;
        }
    });
}


/// Multi-column phased FFT: runs N-point FFTs over many columns in
/// parallel, sharing the PhasedFftContext and amortising the scratch
/// buffer across rayon workers via for_each_init.
///
/// Each column receives the same FFT (output in natural order). Use
/// when you have a batch of independent polynomials to evaluate at the
/// same LDE domain — typical for STARK trace LDE.
///
/// # Memory model
///
/// One scratch buffer of N field elements per active worker thread
/// (= rayon::current_num_threads() typically). Independent of column
/// count: 100 columns processed by 8 workers ⇒ 8 buffers, not 100.
pub fn bowers_phased_fft_multicol<F, E>(
    columns: &mut [&mut [FieldElement<E>]],
    ctx: &PhasedFftContext<F>,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
    FieldElement<F>: Send + Sync,
    FieldElement<E>: Send + Sync + Clone,
{
    let n = 1usize << ctx.log_n;

    #[cfg(feature = "parallel")]
    {
        columns
            .par_iter_mut()
            .try_for_each_init(
                || Vec::<FieldElement<E>>::with_capacity(n),
                |buf, col| -> Result<(), FFTError> {
                    bowers_phased_fft_with_buf(col, ctx, buf)
                },
            )?;
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut buf: Vec<FieldElement<E>> = Vec::with_capacity(n);
        for col in columns.iter_mut() {
            bowers_phased_fft_with_buf(col, ctx, &mut buf)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks::GoldilocksField;

    type F = GoldilocksField;
    type FE = FieldElement<F>;

    /// Reference: existing Bowers + bit_reverse → natural order output.
    fn reference_fft(input: &[FE]) -> Vec<FE> {
        let log_n = input.len().trailing_zeros() as u64;
        let twiddles = LayerTwiddles::<F>::new(log_n).unwrap();
        let mut buf = input.to_vec();
        bowers_fft_opt_fused(&mut buf, &twiddles).unwrap();
        in_place_bit_reverse_permute(&mut buf);
        buf
    }

    fn random_input(log_n: usize, seed: u64) -> Vec<FE> {
        // Deterministic pseudo-random over Goldilocks: chain LCG on u64.
        let n = 1usize << log_n;
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                FE::from(state)
            })
            .collect()
    }

    #[test]
    fn phased_equivalence_log4() {
        let input = random_input(4, 42);
        let expected = reference_fft(&input);
        let mut actual = input;
        bowers_phased_fft::<F, F>(&mut actual).unwrap();
        assert_eq!(actual, expected, "log_n = 4 mismatch");
    }

    #[test]
    fn phased_equivalence_log5_odd_split() {
        // log_n = 5 → log_m = 3, log_k = 2 (uneven split).
        let input = random_input(5, 99);
        let expected = reference_fft(&input);
        let mut actual = input;
        bowers_phased_fft::<F, F>(&mut actual).unwrap();
        assert_eq!(actual, expected, "log_n = 5 mismatch");
    }

    #[test]
    fn phased_equivalence_log_n_4_through_12() {
        for log_n in 4..=12 {
            for seed in [1u64, 7, 12345, 0xdead_beef] {
                let input = random_input(log_n, seed);
                let expected = reference_fft(&input);
                let mut actual = input;
                bowers_phased_fft::<F, F>(&mut actual).unwrap();
                assert_eq!(
                    actual, expected,
                    "log_n = {log_n}, seed = {seed:x}: phased output mismatches reference"
                );
            }
        }
    }

    #[test]
    fn phased_equivalence_log_n_16() {
        let input = random_input(16, 0xcafe_babe);
        let expected = reference_fft(&input);
        let mut actual = input;
        bowers_phased_fft::<F, F>(&mut actual).unwrap();
        assert_eq!(actual, expected, "log_n = 16 mismatch");
    }

    #[test]
    fn phased_equivalence_log_n_18_odd_split() {
        // log_n = 18 → log_m = 9, log_k = 9 (even split, no reuse).
        let input = random_input(18, 7);
        let expected = reference_fft(&input);
        let mut actual = input;
        bowers_phased_fft::<F, F>(&mut actual).unwrap();
        assert_eq!(actual, expected, "log_n = 18 mismatch");
    }

    #[test]
    fn phased_multicol_matches_per_column_reference() {
        let log_n = 14;
        let num_cols = 6;
        let ctx = PhasedFftContext::<F>::new(log_n).expect("ctx valid");

        let mut columns: Vec<Vec<FE>> =
            (0..num_cols).map(|c| random_input(log_n, c as u64 + 1)).collect();
        let expected: Vec<Vec<FE>> = columns.iter().map(|c| reference_fft(c)).collect();

        let mut col_refs: Vec<&mut [FE]> =
            columns.iter_mut().map(|v| v.as_mut_slice()).collect();
        bowers_phased_fft_multicol::<F, F>(&mut col_refs, &ctx).expect("multicol ok");

        for (c, (got, want)) in columns.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, want, "column {c} mismatches reference");
        }
    }

    #[test]
    fn tiled_transpose_2x3() {
        let input: Vec<u64> = vec![1, 2, 3, 4, 5, 6];
        let mut output = vec![0u64; 6];
        tiled_transpose(&input, &mut output, 2, 3);
        // Input (2×3):
        //   1 2 3
        //   4 5 6
        // Expected output (3×2):
        //   1 4
        //   2 5
        //   3 6
        assert_eq!(output, vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn tiled_transpose_roundtrip() {
        let rows = 17;
        let cols = 23;
        let input: Vec<u64> = (0..(rows * cols) as u64).collect();
        let mut t = vec![0u64; rows * cols];
        let mut back = vec![0u64; rows * cols];
        tiled_transpose(&input, &mut t, rows, cols);
        tiled_transpose(&t, &mut back, cols, rows);
        assert_eq!(back, input);
    }
}
