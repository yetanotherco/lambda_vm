use crate::fft::errors::FFTError;
use crate::field::traits::{IsField, IsSubFieldOf};
use crate::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};
use alloc::{vec, vec::Vec};

use super::cpu::{
    bit_reversing::in_place_bit_reverse_permute,
    bowers_fft::{LayerTwiddles, bowers_fft_opt_fused, bowers_ifft_opt},
    bowers_fft_batch::{
        bowers_fft_batch_row_major, bowers_ifft_batch_row_major,
        in_place_bit_reverse_permute_row_major,
    },
};

#[cfg(feature = "parallel")]
use super::cpu::bowers_fft::{bowers_fft_opt_fused_parallel, bowers_ifft_opt_parallel};

/// Threshold for dispatching to parallel FFT.
/// Below this size, sequential FFT is faster (avoids Rayon overhead).
/// At 2^14 = 16384 elements, the parallel version starts to win
/// because later butterfly layers have enough blocks for effective parallelism.
#[cfg(feature = "parallel")]
const PARALLEL_FFT_THRESHOLD: usize = 1 << 14;

/// Dispatch forward FFT (DIF) to parallel or sequential implementation based on buffer size.
#[inline]
fn dispatch_fft<F: IsFFTField + IsSubFieldOf<E>, E: IsField + Send + Sync>(
    buffer: &mut [FieldElement<E>],
    twiddles: &LayerTwiddles<F>,
) -> Result<(), FFTError> {
    #[cfg(feature = "parallel")]
    {
        if buffer.len() >= PARALLEL_FFT_THRESHOLD {
            return bowers_fft_opt_fused_parallel(buffer, twiddles);
        }
    }
    bowers_fft_opt_fused(buffer, twiddles)
}

/// Dispatch inverse FFT (DIT) to parallel or sequential implementation based on buffer size.
#[inline]
fn dispatch_ifft<F: IsFFTField + IsSubFieldOf<E>, E: IsField + Send + Sync>(
    buffer: &mut [FieldElement<E>],
    twiddles: &LayerTwiddles<F>,
) -> Result<(), FFTError> {
    #[cfg(feature = "parallel")]
    {
        if buffer.len() >= PARALLEL_FFT_THRESHOLD {
            return bowers_ifft_opt_parallel(buffer, twiddles);
        }
    }
    bowers_ifft_opt(buffer, twiddles)
}

impl<E: IsField> Polynomial<FieldElement<E>> {
    /// Returns `N` evaluations of this polynomial using FFT over a domain in a subfield F of E (so the results
    /// are P(w^i), with w being a primitive root of unity).
    /// `N = max(self.coeff_len(), domain_size).next_power_of_two() * blowup_factor`.
    /// If `domain_size` is `None`, it defaults to 0.
    pub fn evaluate_fft<F: IsFFTField + IsSubFieldOf<E>>(
        poly: &Polynomial<FieldElement<E>>,
        blowup_factor: usize,
        domain_size: Option<usize>,
    ) -> Result<Vec<FieldElement<E>>, FFTError>
    where
        E: Send + Sync,
    {
        let domain_size = domain_size.unwrap_or(0);
        let len = core::cmp::max(poly.coeff_len(), domain_size).next_power_of_two() * blowup_factor;
        if len.trailing_zeros() as u64 > F::TWO_ADICITY {
            return Err(FFTError::DomainSizeError(len.trailing_zeros() as usize));
        }
        if poly.coefficients().is_empty() {
            return Ok(vec![FieldElement::zero(); len]);
        }

        let mut coeffs = poly.coefficients().to_vec();
        coeffs.resize(len, FieldElement::zero());
        // padding with zeros will make FFT return more evaluations of the same polynomial.

        evaluate_fft_cpu::<F, E>(&coeffs)
    }

    /// Same as `evaluate_fft` but returns the evaluations in bit-reversed order,
    /// skipping the final natural-order permutation. Use when the consumer expects
    /// bit-reversed input (e.g. FRI commit phase, which pairs consecutive values as
    /// {f(x), f(-x)}).
    pub fn evaluate_fft_bit_reversed<F: IsFFTField + IsSubFieldOf<E>>(
        poly: &Polynomial<FieldElement<E>>,
        blowup_factor: usize,
        domain_size: Option<usize>,
    ) -> Result<Vec<FieldElement<E>>, FFTError>
    where
        E: Send + Sync,
    {
        let domain_size = domain_size.unwrap_or(0);
        let len = core::cmp::max(poly.coeff_len(), domain_size).next_power_of_two() * blowup_factor;
        if len.trailing_zeros() as u64 > F::TWO_ADICITY {
            return Err(FFTError::DomainSizeError(len.trailing_zeros() as usize));
        }
        if poly.coefficients().is_empty() {
            return Ok(vec![FieldElement::zero(); len]);
        }

        let mut coeffs = poly.coefficients().to_vec();
        coeffs.resize(len, FieldElement::zero());

        evaluate_fft_cpu_raw::<F, E>(&coeffs, false)
    }

    /// Returns `N` evaluations with an offset of this polynomial using FFT over a domain in a subfield F of E
    /// (so the results are P(w^i), with w being a primitive root of unity).
    /// `N = max(self.coeff_len(), domain_size).next_power_of_two() * blowup_factor`.
    /// If `domain_size` is `None`, it defaults to 0.
    pub fn evaluate_offset_fft<F: IsFFTField + IsSubFieldOf<E>>(
        poly: &Polynomial<FieldElement<E>>,
        blowup_factor: usize,
        domain_size: Option<usize>,
        offset: &FieldElement<F>,
    ) -> Result<Vec<FieldElement<E>>, FFTError>
    where
        E: Send + Sync,
    {
        let scaled = poly.scale(offset);
        Polynomial::evaluate_fft::<F>(&scaled, blowup_factor, domain_size)
    }

    /// Returns a new polynomial that interpolates `(w^i, fft_evals[i])`, with `w` being a
    /// Nth primitive root of unity in a subfield F of E, and `i in 0..N`, with `N = fft_evals.len()`.
    /// This is considered to be the inverse operation of [Self::evaluate_fft()].
    pub fn interpolate_fft<F: IsFFTField + IsSubFieldOf<E>>(
        fft_evals: &[FieldElement<E>],
    ) -> Result<Self, FFTError>
    where
        E: Send + Sync,
    {
        interpolate_fft_cpu::<F, E>(fft_evals)
    }

    /// Returns a new polynomial that interpolates offset `(w^i, fft_evals[i])`, with `w` being a
    /// Nth primitive root of unity in a subfield F of E, and `i in 0..N`, with `N = fft_evals.len()`.
    /// This is considered to be the inverse operation of [Self::evaluate_offset_fft()].
    pub fn interpolate_offset_fft<F: IsFFTField + IsSubFieldOf<E>>(
        fft_evals: &[FieldElement<E>],
        offset: &FieldElement<F>,
    ) -> Result<Polynomial<FieldElement<E>>, FFTError>
    where
        E: Send + Sync,
    {
        let scaled = Polynomial::interpolate_fft::<F>(fft_evals)?;
        Ok(scaled.scale(&offset.inv().unwrap()))
    }

    /// Compute the coset LDE with pre-computed twiddle factors and pre-computed weights.
    ///
    /// Same as [`coset_lde_with_twiddles`], but also accepts pre-computed `weights[i] = offset^i / n`
    /// so that the scaling step avoids the running product across columns.
    /// Weights are in the base field F — the scaling `w * coeff` uses mixed F×E multiplication.
    pub fn coset_lde_full<F: IsFFTField + IsSubFieldOf<E> + Send + Sync>(
        evals: &[FieldElement<E>],
        blowup_factor: usize,
        weights: &[FieldElement<F>],
        inv_twiddles: &LayerTwiddles<F>,
        fwd_twiddles: &LayerTwiddles<F>,
    ) -> Result<Vec<FieldElement<E>>, FFTError>
    where
        E: Send + Sync,
    {
        let n = evals.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let lde_size = n * blowup_factor;
        let mut buffer = Vec::with_capacity(lde_size);
        Self::coset_lde_full_into(
            evals,
            blowup_factor,
            weights,
            inv_twiddles,
            fwd_twiddles,
            &mut buffer,
        )?;
        Ok(buffer)
    }

    /// Compute the coset LDE into a caller-provided buffer, avoiding allocation when
    /// `buffer.capacity() >= n * blowup_factor`.
    ///
    /// Same as [`coset_lde_full`], but writes into `buffer` instead of allocating a new Vec.
    /// The buffer is cleared and reused: `buffer.clear(); buffer.extend_from_slice(evals);
    /// buffer.resize(lde_size, zero)`. When the capacity is sufficient, no heap allocation occurs.
    /// Weights are in the base field F — the scaling `w * coeff` uses mixed F×E multiplication.
    pub fn coset_lde_full_into<F: IsFFTField + IsSubFieldOf<E> + Send + Sync>(
        evals: &[FieldElement<E>],
        blowup_factor: usize,
        weights: &[FieldElement<F>],
        inv_twiddles: &LayerTwiddles<F>,
        fwd_twiddles: &LayerTwiddles<F>,
        buffer: &mut Vec<FieldElement<E>>,
    ) -> Result<(), FFTError>
    where
        E: Send + Sync,
    {
        let n = evals.len();
        if n == 0 {
            buffer.clear();
            return Ok(());
        }
        if !n.is_power_of_two() {
            return Err(FFTError::InputError(n));
        }
        let lde_size = n * blowup_factor;

        if (lde_size.trailing_zeros() as u64) > F::TWO_ADICITY {
            return Err(FFTError::DomainSizeError(lde_size.trailing_zeros() as usize));
        }

        buffer.clear();
        buffer.extend_from_slice(evals);
        buffer.resize(lde_size, FieldElement::zero());

        in_place_bit_reverse_permute(&mut buffer[..n]);
        dispatch_ifft(&mut buffer[..n], inv_twiddles)?;

        // Scale using pre-computed weights (base field) — F × E → E mixed multiplication.
        for (coeff, w) in buffer[..n].iter_mut().zip(weights.iter()) {
            *coeff = w * &*coeff;
        }

        dispatch_fft(buffer, fwd_twiddles)?;
        in_place_bit_reverse_permute(buffer);

        Ok(())
    }

    /// In-place coset LDE: the buffer already contains N evaluation points at `[0..N]`.
    ///
    /// This expands the buffer from N elements to `N * blowup_factor` by performing:
    /// 1. iFFT on buffer[..N]
    /// 2. Scale by pre-computed weights
    /// 3. Zero-pad to N * blowup_factor
    /// 4. Forward FFT on the full buffer
    ///
    /// Unlike [`coset_lde_full_into`], this skips the `clear + extend_from_slice` step
    /// since data is already in the buffer. Used for transpose elimination: columns are
    /// extracted directly into owned buffers, then expanded in-place.
    pub fn coset_lde_full_expand<F: IsFFTField + IsSubFieldOf<E> + Send + Sync>(
        buffer: &mut Vec<FieldElement<E>>,
        blowup_factor: usize,
        weights: &[FieldElement<F>],
        inv_twiddles: &LayerTwiddles<F>,
        fwd_twiddles: &LayerTwiddles<F>,
    ) -> Result<(), FFTError>
    where
        E: Send + Sync,
    {
        let n = buffer.len();
        if n == 0 {
            return Ok(());
        }
        if !n.is_power_of_two() {
            return Err(FFTError::InputError(n));
        }
        let lde_size = n * blowup_factor;

        if (lde_size.trailing_zeros() as u64) > F::TWO_ADICITY {
            return Err(FFTError::DomainSizeError(lde_size.trailing_zeros() as usize));
        }

        // 1. iFFT on buffer[..n]
        in_place_bit_reverse_permute(&mut buffer[..n]);
        dispatch_ifft(&mut buffer[..n], inv_twiddles)?;

        // 2. Scale using pre-computed weights (base field) — F × E → E mixed multiplication.
        for (coeff, w) in buffer[..n].iter_mut().zip(weights.iter()) {
            *coeff = w * &*coeff;
        }

        // 3. Zero-pad to lde_size
        buffer.resize(lde_size, FieldElement::zero());

        // 4. Forward FFT on the full buffer
        dispatch_fft(buffer, fwd_twiddles)?;
        in_place_bit_reverse_permute(buffer);

        Ok(())
    }

    /// Batched row-major coset LDE expansion.
    ///
    /// `buffer` is the row-major flat layout of `n * num_cols` elements
    /// (input trace evaluations on the natural-order domain, all M columns
    /// interleaved per row). It is expanded in place to length
    /// `n * blowup_factor * num_cols`, also row-major, holding the LDE
    /// evaluations on the coset.
    ///
    /// Pipeline mirrors [`coset_lde_full_expand`] cell-for-cell, just with
    /// the row-major batched FFT primitives so the M columns share twiddle
    /// loads inside each butterfly:
    ///   1. bit-reverse rows
    ///   2. batched iFFT (DIT) over rows[..n]
    ///   3. scale rows[..n] by coset weights (one weight per row, applied to
    ///      all M elements of that row)
    ///   4. zero-pad rows to `n * blowup_factor`
    ///   5. batched forward FFT (DIF)
    ///   6. bit-reverse rows
    ///
    /// `weights` must be `n` base-field elements in natural row order.
    pub fn coset_lde_full_expand_row_major<F: IsFFTField + IsSubFieldOf<E> + Send + Sync>(
        buffer: &mut Vec<FieldElement<E>>,
        num_cols: usize,
        blowup_factor: usize,
        weights: &[FieldElement<F>],
        inv_twiddles: &LayerTwiddles<F>,
        fwd_twiddles: &LayerTwiddles<F>,
    ) -> Result<(), FFTError>
    where
        E: Send + Sync,
    {
        if num_cols == 0 || buffer.is_empty() {
            return Ok(());
        }
        let total = buffer.len();
        if total % num_cols != 0 {
            return Err(FFTError::InputError(total));
        }
        let n = total / num_cols;
        if !n.is_power_of_two() {
            return Err(FFTError::InputError(n));
        }
        let lde_n = n * blowup_factor;
        if (lde_n.trailing_zeros() as u64) > F::TWO_ADICITY {
            return Err(FFTError::DomainSizeError(lde_n.trailing_zeros() as usize));
        }
        if weights.len() < n {
            return Err(FFTError::InputError(weights.len()));
        }

        // 1. iFFT on rows[..n]
        let prefix_len = n * num_cols;
        in_place_bit_reverse_permute_row_major(&mut buffer[..prefix_len], num_cols);
        bowers_ifft_batch_row_major::<F, E>(&mut buffer[..prefix_len], num_cols, inv_twiddles)?;

        // 2. Scale by coset weights — one weight per row, multiply M elements
        //    of that row by it.
        for r in 0..n {
            let w = &weights[r];
            let row = &mut buffer[r * num_cols..(r + 1) * num_cols];
            for x in row.iter_mut() {
                *x = w * &*x;
            }
        }

        // 3. Zero-pad rows to lde_n.
        buffer.resize(lde_n * num_cols, FieldElement::zero());

        // 4. Forward FFT.
        bowers_fft_batch_row_major::<F, E>(buffer, num_cols, fwd_twiddles)?;
        in_place_bit_reverse_permute_row_major(buffer, num_cols);

        Ok(())
    }
}

#[cfg(test)]
pub fn compose_fft<F, E>(
    poly_1: &Polynomial<FieldElement<E>>,
    poly_2: &Polynomial<FieldElement<E>>,
) -> Polynomial<FieldElement<E>>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
{
    let poly_2_evaluations = Polynomial::evaluate_fft::<F>(poly_2, 1, None).unwrap();

    let values: Vec<_> = poly_2_evaluations
        .iter()
        .map(|value| poly_1.evaluate(value))
        .collect();

    Polynomial::interpolate_fft::<F>(values.as_slice()).unwrap()
}

pub fn evaluate_fft_cpu<F, E>(coeffs: &[FieldElement<E>]) -> Result<Vec<FieldElement<E>>, FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
{
    evaluate_fft_cpu_raw::<F, E>(coeffs, true)
}

fn evaluate_fft_cpu_raw<F, E>(
    coeffs: &[FieldElement<E>],
    permute_to_natural: bool,
) -> Result<Vec<FieldElement<E>>, FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
{
    let n = coeffs.len();
    if !n.is_power_of_two() {
        return Err(FFTError::InputError(n));
    }
    let order = n.trailing_zeros() as u64;
    let layer_twiddles =
        LayerTwiddles::<F>::new(order).ok_or(FFTError::DomainSizeError(order as usize))?;

    let mut result = coeffs.to_vec();
    dispatch_fft(&mut result, &layer_twiddles)?;
    if permute_to_natural {
        in_place_bit_reverse_permute(&mut result);
    }
    Ok(result)
}

pub fn interpolate_fft_cpu<F, E>(
    fft_evals: &[FieldElement<E>],
) -> Result<Polynomial<FieldElement<E>>, FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
{
    let n = fft_evals.len();
    if !n.is_power_of_two() {
        return Err(FFTError::InputError(n));
    }
    let order = n.trailing_zeros() as u64;
    let inv_twiddles =
        LayerTwiddles::<F>::new_inverse(order).ok_or(FFTError::DomainSizeError(order as usize))?;

    let mut coeffs = fft_evals.to_vec();
    // Bowers iFFT: bit-reverse first (natural → bit-reversed), then DIT inverse butterflies
    in_place_bit_reverse_permute(&mut coeffs);
    dispatch_ifft(&mut coeffs, &inv_twiddles)?;

    // Scale by 1/n
    let scale_factor = FieldElement::from(n as u64).inv().unwrap();
    Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
}

#[cfg(test)]
impl<E: IsField> Polynomial<FieldElement<E>> {
    /// Multiplies two polynomials using FFT.
    pub fn fast_fft_multiplication<F: IsFFTField + IsSubFieldOf<E>>(
        &self,
        other: &Self,
    ) -> Result<Self, FFTError>
    where
        E: Send + Sync,
    {
        let domain_size = self.degree() + other.degree() + 1;
        let p = Polynomial::evaluate_fft::<F>(self, 1, Some(domain_size))?;
        let q = Polynomial::evaluate_fft::<F>(other, 1, Some(domain_size))?;
        let r = p.into_iter().zip(q).map(|(a, b)| a * b).collect::<Vec<_>>();

        Polynomial::interpolate_fft::<F>(&r)
    }

    /// Divides two polynomials with remainder using FFT.
    pub fn fast_division<F: IsSubFieldOf<E> + IsFFTField>(
        &self,
        divisor: &Self,
    ) -> Result<(Self, Self), FFTError>
    where
        E: Send + Sync,
    {
        use crate::field::errors::FieldError;

        let n = self.degree();
        let m = divisor.degree();
        if divisor.coefficients.is_empty()
            || divisor
                .coefficients
                .iter()
                .all(|c| c == &FieldElement::zero())
        {
            return Err(FieldError::DivisionByZero.into());
        }
        if n < m {
            return Ok((Self::zero(), self.clone()));
        }
        let d = n - m;
        let a_rev = self.reverse(n);
        let b_rev = divisor.reverse(m);
        let inv_b_rev = b_rev.invert_polynomial_mod::<F>(d + 1)?;
        let q = a_rev
            .fast_fft_multiplication::<F>(&inv_b_rev)?
            .truncate(d + 1)
            .reverse(d);

        let r = self - q.fast_fft_multiplication::<F>(divisor)?;
        Ok((q, r))
    }

    /// Computes the inverse of polynomial P modulo x^k using Newton iteration.
    pub fn invert_polynomial_mod<F: IsSubFieldOf<E> + IsFFTField>(
        &self,
        k: usize,
    ) -> Result<Self, FFTError>
    where
        E: Send + Sync,
    {
        use crate::field::errors::FieldError;

        if self.coefficients.is_empty()
            || self.coefficients.iter().all(|c| c == &FieldElement::zero())
        {
            return Err(FieldError::DivisionByZero.into());
        }
        let mut q = Self::new(&[self.coefficients[0].inv()?]);
        let mut current_precision = 1;

        let two = Self::new(&[FieldElement::<F>::one() + FieldElement::one()]);
        while current_precision < k {
            current_precision *= 2;
            let temp = self
                .fast_fft_multiplication::<F>(&q)?
                .truncate(current_precision);
            let correction = &two - temp;
            q = q
                .fast_fft_multiplication::<F>(&correction)?
                .truncate(current_precision);
        }

        Ok(q.truncate(k))
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::field::goldilocks::GoldilocksField;

    type F = GoldilocksField;
    type FE = FieldElement<F>;

    /// Differential test: `coset_lde_full_expand_row_major` on a row-major
    /// buffer holding M columns must produce the same per-cell output as
    /// running `coset_lde_full_expand` on each of those M columns
    /// independently, then transposing the M LDE columns back into row
    /// order. Covers a range of (log_n, M, blowup) so we catch off-by-one
    /// bugs in the M-block bit-reverse and in the row scaling step.
    #[test]
    fn coset_lde_full_expand_row_major_matches_single_column_per_column() {
        use crate::fft::cpu::bowers_fft::LayerTwiddles;

        for log_n in 2..=8 {
            let n = 1usize << log_n;
            for &blowup_factor in &[2usize, 4] {
                let lde_size = n * blowup_factor;
                let inv_tw = LayerTwiddles::<F>::new_inverse(log_n as u64).unwrap();
                let fwd_tw =
                    LayerTwiddles::<F>::new(lde_size.trailing_zeros() as u64).unwrap();

                // Reproduce the weights from the existing coset test.
                let offset = FE::from(3u64);
                let n_inv = FE::from(n as u64).inv().unwrap();
                let mut weights = Vec::with_capacity(n);
                let mut offset_power = n_inv;
                for _ in 0..n {
                    weights.push(offset_power);
                    offset_power = &offset_power * &offset;
                }

                for &m in &[1usize, 2, 3, 5, 8] {
                    // Generate M deterministic columns of length n.
                    let cols: Vec<Vec<FE>> = (0..m)
                        .map(|c| {
                            (0..n)
                                .map(|i| {
                                    FE::from(
                                        (c as u64).wrapping_mul(1_000_003)
                                            + i as u64
                                            + 17,
                                    )
                                })
                                .collect()
                        })
                        .collect();

                    // Reference: run single-column coset_lde_full_expand on each
                    // column independently.
                    let mut expected_cols: Vec<Vec<FE>> = cols
                        .iter()
                        .map(|c| {
                            let mut buf = c.clone();
                            Polynomial::<FE>::coset_lde_full_expand::<F>(
                                &mut buf,
                                blowup_factor,
                                &weights,
                                &inv_tw,
                                &fwd_tw,
                            )
                            .unwrap();
                            buf
                        })
                        .collect();

                    // Subject under test: row-major batched pipeline.
                    let mut row_major: Vec<FE> = Vec::with_capacity(n * m);
                    for r in 0..n {
                        for c in 0..m {
                            row_major.push(cols[c][r].clone());
                        }
                    }
                    Polynomial::<FE>::coset_lde_full_expand_row_major::<F>(
                        &mut row_major,
                        m,
                        blowup_factor,
                        &weights,
                        &inv_tw,
                        &fwd_tw,
                    )
                    .unwrap();
                    assert_eq!(row_major.len(), lde_size * m);

                    // Transpose row-major back to columns and compare.
                    for r in 0..lde_size {
                        for c in 0..m {
                            assert_eq!(
                                row_major[r * m + c],
                                expected_cols[c][r],
                                "log_n={log_n} blowup={blowup_factor} m={m} r={r} c={c}",
                            );
                        }
                    }
                    // Touch expected to silence unused warnings on early exit paths.
                    for col in expected_cols.iter_mut() {
                        col.truncate(lde_size);
                    }
                }
            }
        }
    }

    #[test]
    fn coset_lde_full_into_matches_coset_lde_full() {
        use crate::fft::cpu::bowers_fft::LayerTwiddles;

        let offset = FE::from(3u64);
        let blowup_factor = 2;

        for order in 1..=10 {
            let n = 1usize << order;
            let evals: Vec<FE> = (0..n).map(|i| FE::from((i * 7 + 13) as u64)).collect();

            let lde_size = n * blowup_factor;
            let inv_tw = LayerTwiddles::<F>::new_inverse(n.trailing_zeros() as u64).unwrap();
            let fwd_tw = LayerTwiddles::<F>::new(lde_size.trailing_zeros() as u64).unwrap();

            let n_inv = FE::from(n as u64).inv().unwrap();
            let mut weights = Vec::with_capacity(n);
            let mut offset_power = n_inv;
            for _ in 0..n {
                weights.push(offset_power);
                offset_power = &offset_power * &offset;
            }

            let reference = Polynomial::<FE>::coset_lde_full::<F>(
                &evals,
                blowup_factor,
                &weights,
                &inv_tw,
                &fwd_tw,
            )
            .unwrap();

            // Test with pre-allocated buffer
            let mut buffer = Vec::with_capacity(lde_size);
            Polynomial::<FE>::coset_lde_full_into::<F>(
                &evals,
                blowup_factor,
                &weights,
                &inv_tw,
                &fwd_tw,
                &mut buffer,
            )
            .unwrap();

            assert_eq!(reference, buffer, "Mismatch at order {}", order);
        }
    }

    #[test]
    fn coset_lde_full_into_reuses_buffer() {
        use crate::fft::cpu::bowers_fft::LayerTwiddles;

        let offset = FE::from(5u64);
        let blowup_factor = 2usize;
        let n = 16usize;
        let lde_size = n * blowup_factor;

        let inv_tw = LayerTwiddles::<F>::new_inverse(n.trailing_zeros() as u64).unwrap();
        let fwd_tw = LayerTwiddles::<F>::new(lde_size.trailing_zeros() as u64).unwrap();

        let n_inv = FE::from(n as u64).inv().unwrap();
        let mut weights = Vec::with_capacity(n);
        let mut offset_power = n_inv;
        for _ in 0..n {
            weights.push(offset_power);
            offset_power = &offset_power * &offset;
        }

        // Pre-allocate buffer once, reuse for two different inputs
        let mut buffer = Vec::with_capacity(lde_size);

        for seed in [13u64, 42u64] {
            let evals: Vec<FE> = (0..n).map(|i| FE::from(i as u64 * seed + 1)).collect();

            let reference = Polynomial::<FE>::coset_lde_full::<F>(
                &evals,
                blowup_factor,
                &weights,
                &inv_tw,
                &fwd_tw,
            )
            .unwrap();

            Polynomial::<FE>::coset_lde_full_into::<F>(
                &evals,
                blowup_factor,
                &weights,
                &inv_tw,
                &fwd_tw,
                &mut buffer,
            )
            .unwrap();

            assert_eq!(reference, buffer, "Mismatch for seed {}", seed);
        }
    }
}
