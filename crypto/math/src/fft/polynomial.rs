use crate::fft::errors::FFTError;
use crate::fft::traits::Fft;

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

use super::cpu::{ops, roots_of_unity};

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
        interpolate_fft_cpu::<F, E>(fft_evals)
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

    /// Interpolate evaluations to a polynomial using a unified FFT backend.
    ///
    /// This is the backend-aware version of `interpolate_fft`. It uses the
    /// provided `Fft` backend (CPU or GPU) instead of the hardcoded CPU path.
    /// The field type `F` must match the backend's field (e.g. GoldilocksField).
    pub fn interpolate_fft_with_backend(
        fft_evals: &[FieldElement<E>],
        backend: &dyn Fft<E>,
    ) -> Result<Self, FFTError>
    where
        E: IsFFTField,
    {
        let mut data = fft_evals.to_vec();
        backend.ifft(&mut data)?;
        Ok(Polynomial::new(&data))
    }

    /// Evaluate a polynomial on an FFT domain using a unified FFT backend.
    ///
    /// Returns `N` evaluations where `N = max(coeff_len, domain_size).next_power_of_two() * blowup_factor`.
    pub fn evaluate_fft_with_backend(
        poly: &Polynomial<FieldElement<E>>,
        blowup_factor: usize,
        domain_size: Option<usize>,
        backend: &dyn Fft<E>,
    ) -> Result<Vec<FieldElement<E>>, FFTError>
    where
        E: IsFFTField,
    {
        let domain_size = domain_size.unwrap_or(0);
        let len = core::cmp::max(poly.coeff_len(), domain_size).next_power_of_two() * blowup_factor;
        if len.trailing_zeros() as u64 > E::TWO_ADICITY {
            return Err(FFTError::DomainSizeError(len.trailing_zeros() as usize));
        }
        if poly.coefficients().is_empty() {
            return Ok(vec![FieldElement::zero(); len]);
        }

        let mut coeffs = poly.coefficients().to_vec();
        coeffs.resize(len, FieldElement::zero());
        backend.fft(&mut coeffs)?;
        Ok(coeffs)
    }

    /// Evaluate a polynomial on an offset FFT domain using a unified FFT backend.
    pub fn evaluate_offset_fft_with_backend(
        poly: &Polynomial<FieldElement<E>>,
        blowup_factor: usize,
        domain_size: Option<usize>,
        offset: &FieldElement<E>,
        backend: &dyn Fft<E>,
    ) -> Result<Vec<FieldElement<E>>, FFTError>
    where
        E: IsFFTField,
    {
        let scaled = poly.scale(offset);
        Self::evaluate_fft_with_backend(&scaled, blowup_factor, domain_size, backend)
    }

    /// Interpolate offset evaluations using a unified FFT backend.
    pub fn interpolate_offset_fft_with_backend(
        fft_evals: &[FieldElement<E>],
        offset: &FieldElement<E>,
        backend: &dyn Fft<E>,
    ) -> Result<Polynomial<FieldElement<E>>, FFTError>
    where
        E: IsFFTField,
    {
        let scaled = Self::interpolate_fft_with_backend(fft_evals, backend)?;
        Ok(scaled.scale(&offset.inv().unwrap()))
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
