use crate::fft::errors::FFTError;

#[cfg(test)]
use crate::field::errors::FieldError;
use crate::field::traits::{IsField, IsSubFieldOf};
use crate::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};
use alloc::{vec, vec::Vec};

use super::cpu::{
    bit_reversing::in_place_bit_reverse_permute,
    bowers_fft::{LayerTwiddles, bowers_fft_opt_fused, bowers_ifft_opt},
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
    /// extracted directly into pool buffers, then expanded in-place.
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

    /// Multiplies two polynomials using FFT.
    /// It's faster than naive multiplication when the degree of the polynomials is large enough (>=2**6).
    /// This works best with polynomials whose highest degree is equal to a power of 2 - 1.
    /// Will return an error if the degree of the resulting polynomial is greater than 2**63.
    ///
    /// This is an implementation of the fast division algorithm from
    /// [Gathen's book](https://www.cambridge.org/core/books/modern-computer-algebra/DB3563D4013401734851CF683D2F03F0)
    /// chapter 9
    #[cfg(test)]
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

    /// Divides two polynomials with remainder.
    /// This is faster than the naive division if the degree of the divisor
    /// is greater than the degree of the dividend and both degrees are large enough.
    #[cfg(test)]
    pub fn fast_division<F: IsSubFieldOf<E> + IsFFTField>(
        &self,
        divisor: &Self,
    ) -> Result<(Self, Self), FFTError>
    where
        E: Send + Sync,
    {
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
        let d = n - m; // Degree of the quotient
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
    /// P must have an invertible constant term.
    #[cfg(test)]
    pub fn invert_polynomial_mod<F: IsSubFieldOf<E> + IsFFTField>(
        &self,
        k: usize,
    ) -> Result<Self, FFTError>
    where
        E: Send + Sync,
    {
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

        // Final truncation to desired degree k
        Ok(q.truncate(k))
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
    let n = coeffs.len();
    if !n.is_power_of_two() {
        return Err(FFTError::InputError(n));
    }
    let order = n.trailing_zeros() as u64;
    let layer_twiddles =
        LayerTwiddles::<F>::new(order).ok_or(FFTError::DomainSizeError(order as usize))?;

    let mut result = coeffs.to_vec();
    dispatch_fft(&mut result, &layer_twiddles)?;
    in_place_bit_reverse_permute(&mut result);
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

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::field::goldilocks::GoldilocksField;

    type F = GoldilocksField;
    type FE = FieldElement<F>;

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
