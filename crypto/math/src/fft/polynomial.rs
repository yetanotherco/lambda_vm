use crate::fft::errors::FFTError;

use crate::field::errors::FieldError;
use crate::field::traits::{IsField, IsSubFieldOf};
use crate::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, RootsConfig},
    },
    polynomial::Polynomial,
};
use alloc::{vec, vec::Vec};

#[cfg(feature = "cuda")]
use crate::fft::gpu::cuda::polynomial::{evaluate_fft_cuda, interpolate_fft_cuda};

use super::cpu::{
    bit_reversing::in_place_bit_reverse_permute, fft::in_place_nr_2radix_fft, ops,
    roots_of_unity,
};

impl<E: IsField> Polynomial<FieldElement<E>> {
    /// Returns `N` evaluations of this polynomial using FFT over a domain in a subfield F of E (so the results
    /// are P(w^i), with w being a primitive root of unity).
    /// `N = max(self.coeff_len(), domain_size).next_power_of_two() * blowup_factor`.
    /// If `domain_size` is `None`, it defaults to 0.
    pub fn evaluate_fft<F: IsFFTField + IsSubFieldOf<E>>(
        poly: &Polynomial<FieldElement<E>>,
        blowup_factor: usize,
        domain_size: Option<usize>,
    ) -> Result<Vec<FieldElement<E>>, FFTError> {
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

        #[cfg(feature = "cuda")]
        {
            // TODO: support multiple fields with CUDA
            if F::field_name() == "stark256" {
                Ok(evaluate_fft_cuda(&coeffs)?)
            } else {
                evaluate_fft_cpu::<F, E>(&coeffs)
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            evaluate_fft_cpu::<F, E>(&coeffs)
        }
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
    ) -> Result<Vec<FieldElement<E>>, FFTError> {
        let scaled = poly.scale(offset);
        Polynomial::evaluate_fft::<F>(&scaled, blowup_factor, domain_size)
    }

    /// Returns a new polynomial that interpolates `(w^i, fft_evals[i])`, with `w` being a
    /// Nth primitive root of unity in a subfield F of E, and `i in 0..N`, with `N = fft_evals.len()`.
    /// This is considered to be the inverse operation of [Self::evaluate_fft()].
    pub fn interpolate_fft<F: IsFFTField + IsSubFieldOf<E>>(
        fft_evals: &[FieldElement<E>],
    ) -> Result<Self, FFTError> {
        #[cfg(feature = "cuda")]
        {
            if !F::field_name().is_empty() {
                Ok(interpolate_fft_cuda(fft_evals)?)
            } else {
                interpolate_fft_cpu::<F, E>(fft_evals)
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            interpolate_fft_cpu::<F, E>(fft_evals)
        }
    }

    /// Returns a new polynomial that interpolates offset `(w^i, fft_evals[i])`, with `w` being a
    /// Nth primitive root of unity in a subfield F of E, and `i in 0..N`, with `N = fft_evals.len()`.
    /// This is considered to be the inverse operation of [Self::evaluate_offset_fft()].
    pub fn interpolate_offset_fft<F: IsFFTField + IsSubFieldOf<E>>(
        fft_evals: &[FieldElement<E>],
        offset: &FieldElement<F>,
    ) -> Result<Polynomial<FieldElement<E>>, FFTError> {
        let scaled = Polynomial::interpolate_fft::<F>(fft_evals)?;
        Ok(scaled.scale(&offset.inv().unwrap()))
    }

    /// Compute the coset LDE of evaluations on the standard domain.
    ///
    /// Given `n` evaluations `f(ω^i)` on the standard domain `{ω^i}`, returns
    /// `n * blowup_factor` evaluations `f(offset · ω_LDE^j)` on the coset LDE domain,
    /// with a single allocation of the output buffer.
    ///
    /// This fuses the `interpolate_fft` → `scale(offset)` → `evaluate_fft(blowup)` pipeline
    /// into one pass, avoiding 2 intermediate allocations per column.
    pub fn coset_lde<F: IsFFTField + IsSubFieldOf<E>>(
        evals: &[FieldElement<E>],
        blowup_factor: usize,
        offset: &FieldElement<F>,
    ) -> Result<Vec<FieldElement<E>>, FFTError> {
        let n = evals.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        debug_assert!(n.is_power_of_two());
        let lde_size = n * blowup_factor;

        if (lde_size.trailing_zeros() as u64) > F::TWO_ADICITY {
            return Err(FFTError::DomainSizeError(lde_size.trailing_zeros() as usize));
        }

        // 1. Allocate buffer of lde_size, copy evals into first n slots, zero-pad rest.
        let mut buffer = Vec::with_capacity(lde_size);
        buffer.extend_from_slice(evals);
        buffer.resize(lde_size, FieldElement::zero());

        // 2. iFFT in-place on buffer[..n]: NR-DIT with inverse twiddles, then bit-reverse.
        let inv_order = n.trailing_zeros() as u64;
        let inv_twiddles =
            roots_of_unity::get_twiddles::<F>(inv_order, RootsConfig::BitReverseInversed)?;
        in_place_nr_2radix_fft(&mut buffer[..n], &inv_twiddles);
        in_place_bit_reverse_permute(&mut buffer[..n]);

        // 3. Scale by offset^i / n simultaneously (fused inverse-scaling + coset shift).
        //    After iFFT, buffer[i] = (1/n) * Σ_k evals[k] · ω^{-ik} for i=0..n-1.
        //    The scaling transforms coefficients c[i] into c[i] · offset^i,
        //    and the 1/n normalization from iFFT is applied at the same time.
        let n_inv = FieldElement::<F>::from(n as u64).inv().unwrap();
        let mut offset_power = n_inv.clone(); // offset^0 * n_inv = n_inv
        for coeff in buffer[..n].iter_mut() {
            *coeff = &offset_power * &*coeff;
            offset_power = &offset_power * offset;
        }
        // buffer[n..] is already zero from the resize — no scaling needed.

        // 4. Forward FFT in-place on the full buffer: NR-DIT with forward twiddles, then bit-reverse.
        let fwd_order = lde_size.trailing_zeros() as u64;
        let fwd_twiddles =
            roots_of_unity::get_twiddles::<F>(fwd_order, RootsConfig::BitReverse)?;
        in_place_nr_2radix_fft(&mut buffer, &fwd_twiddles);
        in_place_bit_reverse_permute(&mut buffer);

        Ok(buffer)
    }

    /// Multiplies two polynomials using FFT.
    /// It's faster than naive multiplication when the degree of the polynomials is large enough (>=2**6).
    /// This works best with polynomials whose highest degree is equal to a power of 2 - 1.
    /// Will return an error if the degree of the resulting polynomial is greater than 2**63.
    ///
    /// This is an implementation of the fast division algorithm from
    /// [Gathen's book](https://www.cambridge.org/core/books/modern-computer-algebra/DB3563D4013401734851CF683D2F03F0)
    /// chapter 9
    pub fn fast_fft_multiplication<F: IsFFTField + IsSubFieldOf<E>>(
        &self,
        other: &Self,
    ) -> Result<Self, FFTError> {
        let domain_size = self.degree() + other.degree() + 1;
        let p = Polynomial::evaluate_fft::<F>(self, 1, Some(domain_size))?;
        let q = Polynomial::evaluate_fft::<F>(other, 1, Some(domain_size))?;
        let r = p.into_iter().zip(q).map(|(a, b)| a * b).collect::<Vec<_>>();

        Polynomial::interpolate_fft::<F>(&r)
    }

    /// Divides two polynomials with remainder.
    /// This is faster than the naive division if the degree of the divisor
    /// is greater than the degree of the dividend and both degrees are large enough.
    pub fn fast_division<F: IsSubFieldOf<E> + IsFFTField>(
        &self,
        divisor: &Self,
    ) -> Result<(Self, Self), FFTError> {
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
    pub fn invert_polynomial_mod<F: IsSubFieldOf<E> + IsFFTField>(
        &self,
        k: usize,
    ) -> Result<Self, FFTError> {
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

pub fn compose_fft<F, E>(
    poly_1: &Polynomial<FieldElement<E>>,
    poly_2: &Polynomial<FieldElement<E>>,
) -> Polynomial<FieldElement<E>>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
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
    E: IsField,
{
    let order = coeffs.len().trailing_zeros();
    let twiddles = roots_of_unity::get_twiddles::<F>(order.into(), RootsConfig::BitReverse)?;
    // Bit reverse order is needed for NR DIT FFT.
    ops::fft(coeffs, &twiddles)
}

pub fn interpolate_fft_cpu<F, E>(
    fft_evals: &[FieldElement<E>],
) -> Result<Polynomial<FieldElement<E>>, FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let order = fft_evals.len().trailing_zeros();
    let twiddles =
        roots_of_unity::get_twiddles::<F>(order.into(), RootsConfig::BitReverseInversed)?;

    let coeffs = ops::fft(fft_evals, &twiddles)?;

    let scale_factor = FieldElement::from(fft_evals.len() as u64).inv().unwrap();
    Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;

    type F = GoldilocksField;
    type FE = FieldElement<F>;

    #[test]
    fn coset_lde_matches_interpolate_then_evaluate() {
        // Test that coset_lde produces identical results to the
        // interpolate_fft + evaluate_offset_fft pipeline.
        let offset = FE::from(3u64);
        let blowup_factor = 2;

        for order in 1..=10 {
            let n = 1usize << order;

            // Create random-ish evaluations
            let evals: Vec<FE> = (0..n).map(|i| FE::from((i * 7 + 13) as u64)).collect();

            // Reference: interpolate → scale → evaluate (the old pipeline)
            let poly = Polynomial::interpolate_fft::<F>(&evals).unwrap();
            let reference = Polynomial::evaluate_offset_fft::<F>(
                &poly,
                blowup_factor,
                Some(n),
                &offset,
            )
            .unwrap();

            // Fused: coset_lde
            let fused = Polynomial::<FE>::coset_lde::<F>(&evals, blowup_factor, &offset).unwrap();

            assert_eq!(
                reference.len(),
                fused.len(),
                "Length mismatch at order {}",
                order
            );
            assert_eq!(reference, fused, "Value mismatch at order {}", order);
        }
    }

    #[test]
    fn coset_lde_blowup_factor_4() {
        let offset = FE::from(7u64);
        let blowup_factor = 4;
        let n = 16;

        let evals: Vec<FE> = (0..n).map(|i| FE::from((i * 3 + 1) as u64)).collect();

        let poly = Polynomial::interpolate_fft::<F>(&evals).unwrap();
        let reference =
            Polynomial::evaluate_offset_fft::<F>(&poly, blowup_factor, Some(n), &offset).unwrap();

        let fused = Polynomial::<FE>::coset_lde::<F>(&evals, blowup_factor, &offset).unwrap();

        assert_eq!(reference, fused);
    }

    #[test]
    fn coset_lde_empty_input() {
        let offset = FE::from(3u64);
        let result = Polynomial::<FE>::coset_lde::<F>(&[], 2, &offset).unwrap();
        assert!(result.is_empty());
    }
}
