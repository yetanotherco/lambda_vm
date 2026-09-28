//! Cache-blocked, transpose-free batched FFT (port of Plonky3's two-half
//! `Radix2DitParallel::dft_batch`).
//!
//! The flat Bowers DIF streams the whole `n·m` buffer with large strides at the
//! early layers, thrashing cache for large `n`. This kernel keeps every layer
//! cache-resident by interleaving bit-reversals: bit-reverse → first `mid` DIT
//! layers within `2^mid`-row chunks → bit-reverse → remaining layers within
//! `2^(log_n−mid)`-row chunks → bit-reverse. The bit-reversals turn the
//! large-stride butterflies into chunk-local ones — the cache win the flat
//! Bowers misses. Output is natural order, identical to a per-column
//! single-column Bowers FFT followed by `in_place_bit_reverse_permute_row_major`.
//!
//! Twiddles are precomputed once per size in [`TwoHalfTwiddles`] and reused
//! across calls (the trace LDE invokes this once per direction per domain, and
//! the same domain recurs across tables and rounds).

#[cfg(feature = "alloc")]
use crate::fft::bit_reversing::{
    in_place_bit_reverse_permute, in_place_bit_reverse_permute_row_major,
};
#[cfg(feature = "alloc")]
use crate::fft::errors::FFTError;
#[cfg(feature = "alloc")]
use crate::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use core::mem::MaybeUninit;
#[cfg(all(feature = "alloc", feature = "parallel"))]
use rayon::prelude::*;

/// Precomputed twiddles for a size-`2^log_n` two-half FFT in one direction.
///
/// `tw` is the flat geometric array `[ω⁰, ω¹, …, ω^(n/2−1)]` (`ω` the forward
/// root for the forward transform, its inverse for the inverse transform);
/// `bitrev_tw` is its bit-reversal permutation, used by the second-half layers.
/// Build once and share across calls of the same size and direction.
#[cfg(feature = "alloc")]
pub struct TwoHalfTwiddles<F: IsField> {
    log_n: usize,
    tw: Vec<FieldElement<F>>,
    bitrev_tw: Vec<FieldElement<F>>,
}

#[cfg(feature = "alloc")]
impl<F: IsFFTField> TwoHalfTwiddles<F> {
    /// Precompute twiddles for a size-`2^log_n` transform. `inverse = true`
    /// selects the (unscaled) inverse transform (uses `ω⁻¹`); the `1/n`
    /// normalization is the caller's responsibility.
    pub fn new(log_n: usize, inverse: bool) -> Result<Self, FFTError> {
        let n = 1usize << log_n;
        let half = n / 2;
        // `omega` is unused when half == 0 (log_n == 0), so skip the lookup.
        let omega = if half == 0 {
            FieldElement::<F>::one()
        } else {
            let fwd = F::get_primitive_root_of_unity(log_n as u64)
                .map_err(|_| FFTError::InputError(n))?;
            if inverse {
                fwd.inv().map_err(|_| FFTError::InputError(n))?
            } else {
                fwd
            }
        };

        let mut tw: Vec<FieldElement<F>> = Vec::with_capacity(half);
        let mut cur = FieldElement::<F>::one();
        for _ in 0..half {
            tw.push(cur.clone());
            cur = &cur * &omega;
        }
        let mut bitrev_tw = tw.clone();
        in_place_bit_reverse_permute(&mut bitrev_tw);

        Ok(Self {
            log_n,
            tw,
            bitrev_tw,
        })
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

/// Validate a two-half FFT call: returns `Some(log_n)` when there is work to
/// do, `None` for the empty / size-1 no-ops.
#[cfg(feature = "alloc")]
fn two_half_shape<F: IsField>(
    len: usize,
    m: usize,
    tw: &TwoHalfTwiddles<F>,
) -> Result<Option<usize>, FFTError> {
    if m == 0 || len == 0 {
        return Ok(None);
    }
    if !len.is_multiple_of(m) {
        return Err(FFTError::InputError(len));
    }
    let n = len / m;
    if !n.is_power_of_two() {
        return Err(FFTError::InputError(n));
    }
    let log_n = n.trailing_zeros() as usize;
    if log_n != tw.log_n {
        return Err(FFTError::InputError(n));
    }
    Ok((log_n > 0).then_some(log_n))
}

/// Step 2: layers 0..mid on one `2^mid`-row chunk (every chunk is identical).
#[cfg(feature = "alloc")]
fn first_half_chunk<F, E>(
    chunk: &mut [FieldElement<E>],
    m: usize,
    log_n: usize,
    tw: &[FieldElement<F>],
) where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    for layer in 0..log_n.div_ceil(2) {
        dit_first_half_layer::<F, E>(chunk, m, layer, log_n, tw);
    }
}

/// Steps 3–5: bit-reverse, layers mid..log_n within `2^(log_n-mid)`-row
/// chunks, final bit-reverse to natural order.
#[cfg(feature = "alloc")]
fn second_half_to_natural<F, E>(
    buf: &mut [FieldElement<E>],
    m: usize,
    log_n: usize,
    tw: &TwoHalfTwiddles<F>,
) where
    F: IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    let bitrev_tw = &tw.bitrev_tw;
    let mid = log_n.div_ceil(2);

    in_place_bit_reverse_permute_row_major(buf, m);

    let second_chunk = (1usize << (log_n - mid)) * m;
    #[cfg(feature = "parallel")]
    let it2 = buf.par_chunks_mut(second_chunk).enumerate();
    #[cfg(not(feature = "parallel"))]
    let it2 = buf.chunks_mut(second_chunk).enumerate();
    it2.for_each(|(thread, chunk)| {
        for layer in mid..log_n {
            dit_second_half_layer::<F, E>(chunk, m, layer, log_n, mid, thread, bitrev_tw);
        }
    });

    in_place_bit_reverse_permute_row_major(buf, m);
}

/// Cache-blocked, transpose-free batched FFT. `buf` is `n * num_cols` row-major
/// (`n` rows of `num_cols` consecutive elements); `tw` are the precomputed
/// twiddles for size `n` in the desired direction (forward or inverse).
/// Output is the natural-order DFT (matches a per-column single-column Bowers
/// FFT followed by `in_place_bit_reverse_permute_row_major`). Inverse transforms
/// are NOT scaled by `1/n` — that is the caller's responsibility (e.g. folded
/// into the coset-weight pass of the LDE).
#[cfg(feature = "alloc")]
pub fn fft_batch_two_half<F, E>(
    buf: &mut [FieldElement<E>],
    num_cols: usize,
    tw: &TwoHalfTwiddles<F>,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    // Step 1: bit-reverse rows; then the rest of the transform.
    if two_half_shape(buf.len(), num_cols, tw)?.is_some() {
        in_place_bit_reverse_permute_row_major(buf, num_cols);
    }
    fft_batch_two_half_bit_reversed(buf, num_cols, tw)
}

/// [`fft_batch_two_half`] for input whose rows are ALREADY bit-reversed (its
/// step 1 done by the caller, e.g. while copying the input in). Same output.
#[cfg(feature = "alloc")]
pub fn fft_batch_two_half_bit_reversed<F, E>(
    buf: &mut [FieldElement<E>],
    num_cols: usize,
    tw: &TwoHalfTwiddles<F>,
) -> Result<(), FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
    FieldElement<F>: Sync,
    FieldElement<E>: Send + Sync,
{
    let m = num_cols;
    let Some(log_n) = two_half_shape(buf.len(), m, tw)? else {
        return Ok(());
    };

    // Step 2: first half — layers 0..mid within 2^mid-row chunks (all identical).
    let first_chunk = (1usize << log_n.div_ceil(2)) * m;
    #[cfg(feature = "parallel")]
    let it = buf.par_chunks_mut(first_chunk);
    #[cfg(not(feature = "parallel"))]
    let it = buf.chunks_mut(first_chunk);
    it.for_each(|chunk| first_half_chunk::<F, E>(chunk, m, log_n, &tw.tw));

    second_half_to_natural(buf, m, log_n, tw);
    Ok(())
}

/// First-half chunks at or below which [`fft_batch_two_half_expand`] stops its
/// in-place waves: the rest read a copy of their prefix rows and all run in
/// parallel, instead of waves too small to keep every core busy.
#[cfg(feature = "alloc")]
const EXPAND_TAIL_CHUNKS: usize = 32;

/// Load first-half chunk `c` (`rows` rows) of the zero-padded, bit-reversed
/// expansion: row `r` of the expansion is `prefix` row `r >> log_spread` when
/// its low `log_spread` bits are zero, and zero otherwise. `prefix` holds the
/// size-`n` bit-reversed input rows (row `h` at `h * m`), so this is exactly
/// what step 1's bit-reverse of the zero-padded buffer puts in the chunk.
#[cfg(feature = "alloc")]
fn load_spread_chunk<'a, E: IsField>(
    dst: &'a mut [MaybeUninit<FieldElement<E>>],
    prefix: &[FieldElement<E>],
    c: usize,
    rows: usize,
    m: usize,
    log_spread: usize,
) -> &'a mut [FieldElement<E>] {
    let mask = (1usize << log_spread) - 1;
    for (r, row) in dst.chunks_exact_mut(m).enumerate() {
        let src_row = c * rows + r;
        if src_row & mask == 0 {
            let h = src_row >> log_spread;
            for (d, v) in row.iter_mut().zip(&prefix[h * m..h * m + m]) {
                d.write(v.clone());
            }
        } else {
            for d in row.iter_mut() {
                d.write(FieldElement::zero());
            }
        }
    }
    // SAFETY: every element of `dst` was written above; `MaybeUninit<T>` has
    // the same layout as `T`.
    unsafe { &mut *(dst as *mut [MaybeUninit<FieldElement<E>>] as *mut [FieldElement<E>]) }
}

/// Forward two-half FFT of `buf`'s `n = buf.len() / num_cols` rows zero-padded
/// to `lde_rows` rows, growing `buf` to `lde_rows * num_cols`. Same output as
/// `buf.resize(lde_rows * num_cols, zero)` followed by [`fft_batch_two_half`],
/// but the padding is never written as zeros nor permuted: step 1 bit-reverses
/// only the `n` prefix rows, and each first-half chunk loads its zero-padded,
/// bit-reversed rows straight from them (`tw` is for size `lde_rows`).
///
/// In place, like the GPU spread kernels: chunk `c` reads prefix rows
/// `[c*R, (c+1)*R) >> log_spread` (`R = 2^mid` rows per chunk) and writes rows
/// `[c*R, (c+1)*R)`. Chunks run in descending waves `[lo, hi)` with
/// `hi <= lo << log_spread`, so a wave only reads rows below `lo*R` — split off
/// from its destinations with `split_at_mut` — that no earlier wave wrote. The
/// last few chunks read a copy of their prefix rows instead.
#[cfg(feature = "alloc")]
pub fn fft_batch_two_half_expand<F, E>(
    buf: &mut Vec<FieldElement<E>>,
    num_cols: usize,
    lde_rows: usize,
    tw: &TwoHalfTwiddles<F>,
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
    if !buf.len().is_multiple_of(m) {
        return Err(FFTError::InputError(buf.len()));
    }
    let n = buf.len() / m;
    if !n.is_power_of_two() || !lde_rows.is_power_of_two() || lde_rows < n {
        return Err(FFTError::InputError(n));
    }
    let log_n = n.trailing_zeros() as usize;
    let log_lde = lde_rows.trailing_zeros() as usize;
    if log_lde != tw.log_n {
        return Err(FFTError::InputError(lde_rows));
    }
    let log_spread = log_lde - log_n;
    let mid = log_lde.div_ceil(2);
    // The spread load needs padding, and at least one prefix row per chunk.
    if log_spread == 0 || log_spread > mid {
        buf.resize(lde_rows * m, FieldElement::zero());
        return fft_batch_two_half(buf, m, tw);
    }
    let rows = 1usize << mid;
    let chunk = rows * m;
    let num_chunks = lde_rows >> mid;

    // Step 1, prefix only.
    in_place_bit_reverse_permute_row_major(buf.as_mut_slice(), m);
    buf.reserve_exact(lde_rows * m - buf.len());
    let prefix_len = n * m;
    // SAFETY: the capacity covers `lde_rows * m` elements; `MaybeUninit<T>` has
    // `T`'s layout. The first `prefix_len` are initialized; the rest are written
    // (by `load_spread_chunk`) before anything reads them, and `buf` is not
    // touched until `full` is dropped.
    let full: &mut [MaybeUninit<FieldElement<E>>] = unsafe {
        core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<_>, lde_rows * m)
    };

    let mut hi = num_chunks;
    while hi > EXPAND_TAIL_CHUNKS {
        let lo = hi.div_ceil(1 << log_spread);
        let (below, above) = full.split_at_mut(lo * chunk);
        // This wave reads prefix rows below hi*R >> log_spread <= min(lo*R, n):
        // initialized, and not yet overwritten (earlier waves wrote >= hi*R).
        let init = below.len().min(prefix_len);
        // SAFETY: the first `prefix_len` elements are initialized (see above).
        let prefix: &[FieldElement<E>] =
            unsafe { &*(&below[..init] as *const [MaybeUninit<_>] as *const [FieldElement<E>]) };
        #[cfg(feature = "parallel")]
        let it = above[..(hi - lo) * chunk].par_chunks_mut(chunk).enumerate();
        #[cfg(not(feature = "parallel"))]
        let it = above[..(hi - lo) * chunk].chunks_mut(chunk).enumerate();
        it.for_each(|(k, dst)| {
            let dst = load_spread_chunk(dst, prefix, lo + k, rows, m, log_spread);
            first_half_chunk::<F, E>(dst, m, log_lde, &tw.tw);
        });
        hi = lo;
    }
    // Tail chunks [0, hi): their prefix rows, set aside, then all in parallel.
    let tail_len = ((hi * rows) >> log_spread) * m;
    // SAFETY: `tail_len <= prefix_len` (initialized), untouched by the waves.
    let tail_src: Vec<FieldElement<E>> =
        unsafe { &*(&full[..tail_len] as *const [MaybeUninit<_>] as *const [FieldElement<E>]) }
            .to_vec();
    #[cfg(feature = "parallel")]
    let it = full[..hi * chunk].par_chunks_mut(chunk).enumerate();
    #[cfg(not(feature = "parallel"))]
    let it = full[..hi * chunk].chunks_mut(chunk).enumerate();
    it.for_each(|(c, dst)| {
        let dst = load_spread_chunk(dst, &tail_src, c, rows, m, log_spread);
        first_half_chunk::<F, E>(dst, m, log_lde, &tw.tw);
    });

    // SAFETY: every chunk of the `lde_rows * m` elements was written above.
    unsafe { buf.set_len(lde_rows * m) };
    second_half_to_natural(buf, m, log_lde, tw);
    Ok(())
}
