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
    if log_n != tw.log_n {
        return Err(FFTError::InputError(n));
    }
    if log_n == 0 {
        return Ok(());
    }

    let flat_tw = &tw.tw;
    let bitrev_tw = &tw.bitrev_tw;
    let mid = log_n.div_ceil(2);

    in_place_bit_reverse_permute_row_major(buf, m);

    let first_chunk = (1usize << mid) * m;
    #[cfg(feature = "parallel")]
    let it = buf.par_chunks_mut(first_chunk);
    #[cfg(not(feature = "parallel"))]
    let it = buf.chunks_mut(first_chunk);
    it.for_each(|chunk| {
        for layer in 0..mid {
            dit_first_half_layer::<F, E>(chunk, m, layer, log_n, flat_tw);
        }
    });

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

    Ok(())
}
