use super::field::element::FieldElement;
use crate::fft::bit_reversing::in_place_bit_reverse_permute;
use crate::fft::bowers_fft::{LayerTwiddles, bowers_fft_opt_fused, bowers_ifft_opt};
#[cfg(feature = "parallel")]
use crate::fft::bowers_fft::{bowers_fft_opt_fused_parallel, bowers_ifft_opt_parallel};
use crate::fft::errors::FFTError;
use crate::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use alloc::{borrow::ToOwned, vec, vec::Vec};

/// Represents the polynomial c_0 + c_1 * X + c_2 * X^2 + ... + c_n * X^n
/// as a vector of coefficients `[c_0, c_1, ... , c_n]`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Polynomial<FE> {
    pub coefficients: Vec<FE>,
}

impl<F: IsField> Polynomial<FieldElement<F>> {
    /// Creates a new polynomial with the given coefficients
    pub fn new(coefficients: &[FieldElement<F>]) -> Self {
        // Removes trailing zero coefficients at the end
        let mut unpadded_coefficients = coefficients
            .iter()
            .rev()
            .skip_while(|x| **x == FieldElement::zero())
            .cloned()
            .collect::<Vec<FieldElement<F>>>();
        unpadded_coefficients.reverse();
        Polynomial {
            coefficients: unpadded_coefficients,
        }
    }

    /// Creates a new monomial term coefficient*x^degree
    pub fn new_monomial(coefficient: FieldElement<F>, degree: usize) -> Self {
        let mut coefficients = vec![FieldElement::zero(); degree];
        coefficients.push(coefficient);
        Self::new(&coefficients)
    }

    /// Creates the null polynomial
    pub fn zero() -> Self {
        Self::new(&[])
    }

    /// Evaluates a polynomial P(t) at a point x, using Horner's algorithm
    /// Returns y = P(x)
    pub fn evaluate<E>(&self, x: &FieldElement<E>) -> FieldElement<E>
    where
        E: IsField,
        F: IsSubFieldOf<E>,
    {
        self.coefficients
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, coeff| {
                coeff + acc * x.to_owned()
            })
    }

    /// Returns the degree of a polynomial, which corresponds to the highest power of x^d
    /// with non-zero coefficient
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }

    /// Returns coefficients of the polynomial as an array
    /// \[c_0, c_1, c_2, ..., c_n\]
    /// that represents the polynomial
    /// c_0 + c_1 * X + c_2 * X^2 + ... + c_n * X^n
    pub fn coefficients(&self) -> &[FieldElement<F>] {
        &self.coefficients
    }

    /// Returns the length of the vector of coefficients
    pub fn coeff_len(&self) -> usize {
        self.coefficients().len()
    }

    pub fn mul_with_ref(&self, factor: &Self) -> Self {
        let degree = self.degree() + factor.degree();
        let mut coefficients = vec![FieldElement::zero(); degree + 1];

        if self.coefficients.is_empty() || factor.coefficients.is_empty() {
            Polynomial::new(&[FieldElement::zero()])
        } else {
            for i in 0..=factor.degree() {
                if factor.coefficients[i] != FieldElement::zero() {
                    for j in 0..=self.degree() {
                        if self.coefficients[j] != FieldElement::zero() {
                            coefficients[i + j] += &factor.coefficients[i] * &self.coefficients[j];
                        }
                    }
                }
            }
            Polynomial::new(&coefficients)
        }
    }

    /// Scales the coefficients of a polynomial P by a factor
    /// Returns P(factor * x)
    pub fn scale<S: IsSubFieldOf<F>>(&self, factor: &FieldElement<S>) -> Self {
        let scaled_coefficients = self
            .coefficients
            .iter()
            .zip(core::iter::successors(Some(FieldElement::one()), |x| {
                Some(x * factor)
            }))
            .map(|(coeff, power)| power * coeff)
            .collect();
        Self {
            coefficients: scaled_coefficients,
        }
    }

    /// Multiplies all coefficients by a factor
    pub fn scale_coeffs(&self, factor: &FieldElement<F>) -> Self {
        let scaled_coefficients = self
            .coefficients
            .iter()
            .map(|coeff| factor * coeff)
            .collect();
        Self {
            coefficients: scaled_coefficients,
        }
    }

    /// Returns a vector of polynomials [p₀, p₁, ..., p_{d-1}], where d is `number_of_parts`, such that `self` equals
    /// p₀(Xᵈ) + Xp₁(Xᵈ) + ... + X^(d-1)p_{d-1}(Xᵈ).
    ///
    /// Example: if d = 2 and `self` is 3 X^3 + X^2 + 2X + 1, then `poly.break_in_parts(2)`
    /// returns a vector with two polynomials `(p₀, p₁)`, where p₀ = X + 1 and p₁ = 3X + 2.
    pub fn break_in_parts(&self, number_of_parts: usize) -> Vec<Self> {
        let coef = self.coefficients();
        let mut parts: Vec<Self> = Vec::with_capacity(number_of_parts);
        for i in 0..number_of_parts {
            let coeffs: Vec<_> = coef
                .iter()
                .skip(i)
                .step_by(number_of_parts)
                .cloned()
                .collect();
            parts.push(Polynomial::new(&coeffs));
        }
        parts
    }
}

/// Pads a polynomial with zeros until the desired length
/// This function can be useful when evaluating polynomials with the FFT
pub fn pad_with_zero_coefficients_to_length<F: IsField>(
    pa: &mut Polynomial<FieldElement<F>>,
    n: usize,
) {
    pa.coefficients.resize(n, FieldElement::zero());
}

/// Pads polynomial representations with minimum number of zeros to match lengths.
pub fn pad_with_zero_coefficients<L: IsField, F: IsSubFieldOf<L>>(
    pa: &Polynomial<FieldElement<F>>,
    pb: &Polynomial<FieldElement<L>>,
) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<L>>) {
    let mut pa = pa.clone();
    let mut pb = pb.clone();

    if pa.coefficients.len() > pb.coefficients.len() {
        pad_with_zero_coefficients_to_length(&mut pb, pa.coefficients.len());
    } else {
        pad_with_zero_coefficients_to_length(&mut pa, pb.coefficients.len());
    }
    (pa, pb)
}
// ── Barycentric coset interpolation ──────────────────────────────────────
// Four evaluation variants along two axes:
//   - eval field: base field (F) or extension field (E)
//   - g_n_inv:    computed on demand, or precomputed by caller
// Use `_with_g_n_inv` variants when evaluating multiple columns at the
// same coset (g_n_inv is constant across columns).

/// Precompute `1/(z - point_i)` for each coset point, using batch inversion.
///
/// Given an evaluation point `z` (in extension field E) and coset points in base field F,
/// returns the vector of inverse denominators needed for barycentric interpolation.
/// Uses Montgomery's trick: 1 field inversion + O(N) multiplications.
#[cfg(feature = "alloc")]
pub fn barycentric_inv_denoms<F, E>(
    z: &FieldElement<E>,
    coset_points: &[FieldElement<F>],
) -> Vec<FieldElement<E>>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    // z - p where z is in E and p is in F. Since Sub<E> is defined for F: IsSubFieldOf<E>,
    // we compute -(p - z) which equals z - p.
    let mut denoms: Vec<FieldElement<E>> = coset_points.iter().map(|p| -(p - z)).collect();
    FieldElement::inplace_batch_inverse(&mut denoms)
        .expect("z is sampled to avoid coset points, so z - g*w^i is never zero");
    denoms
}

/// Like `interpolate_coset_eval_ext` but takes a precomputed `g_n_inv = (g^N)^{-1}`.
///
/// Both `coset_offset_pow_n` and `g_n_inv` stay in the base field F.
#[cfg(feature = "alloc")]
pub fn interpolate_coset_eval_ext_with_g_n_inv<F, E>(
    z_pow_n: &FieldElement<E>,
    coset_offset_pow_n: &FieldElement<F>,
    n_inv: &FieldElement<F>,
    g_n_inv: &FieldElement<F>,
    coset_points: &[FieldElement<F>],
    evaluations: &[FieldElement<E>],
    inv_denoms: &[FieldElement<E>],
) -> FieldElement<E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    debug_assert_eq!(coset_points.len(), evaluations.len());
    debug_assert_eq!(coset_points.len(), inv_denoms.len());

    // point * eval: F × E → E (mixed multiplication, cheaper than E × E)
    let sum: FieldElement<E> = coset_points
        .iter()
        .zip(evaluations.iter())
        .zip(inv_denoms.iter())
        .fold(FieldElement::<E>::zero(), |acc, ((point, eval), inv_d)| {
            let numerator = point * eval;
            acc + numerator * inv_d
        });

    // All scalar factors in base field F; vanishing via sub_subfield.
    let vanishing = z_pow_n.sub_subfield(coset_offset_pow_n); // E - F → E
    let scalar = n_inv * g_n_inv; // F * F → F
    &scalar * &(vanishing * &sum) // F × E → E
}

// =============================================================================
// FFT-based polynomial methods (merged from the former fft/polynomial.rs)
// =============================================================================

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

        evaluate_fft_cpu_raw::<F, E>(&coeffs, true)
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
        let n = fft_evals.len();
        if !n.is_power_of_two() {
            return Err(FFTError::InputError(n));
        }
        let order = n.trailing_zeros() as u64;
        let inv_twiddles = LayerTwiddles::<F>::new_inverse(order)
            .ok_or(FFTError::DomainSizeError(order as usize))?;

        let mut coeffs = fft_evals.to_vec();
        // Bowers iFFT: bit-reverse first (natural -> bit-reversed), then DIT inverse butterflies
        in_place_bit_reverse_permute(&mut coeffs);
        dispatch_ifft(&mut coeffs, &inv_twiddles)?;

        // Scale by 1/n
        let scale_factor = FieldElement::from(n as u64).inv().unwrap();
        Ok(Polynomial::new(&coeffs).scale_coeffs(&scale_factor))
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
        // A zero coset offset has no inverse; report it instead of panicking.
        let offset_inv = offset.inv().map_err(|_| FFTError::InvalidCosetOffset)?;
        Ok(scaled.scale(&offset_inv))
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
    pub(crate) fn coset_lde_full_into<F: IsFFTField + IsSubFieldOf<E> + Send + Sync>(
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
    /// Unlike `coset_lde_full_into`, this skips the `clear + extend_from_slice` step
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
