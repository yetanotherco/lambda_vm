use super::field::element::FieldElement;
use crate::field::traits::{IsField, IsSubFieldOf};
use alloc::{borrow::ToOwned, vec, vec::Vec};
use core::{fmt::Display, ops, slice};
pub mod dense_multilinear_poly;
mod error;
pub mod sparse_multilinear_poly;

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

    /// Returns a polynomial that interpolates the points with x coordinates and y coordinates given by
    /// `xs` and `ys`.
    /// `xs` and `ys` must be the same length, and `xs` values should be unique. If not, panics.
    /// In short, it finds P(x) such that P(xs[i]) = ys[i]
    pub fn interpolate(
        xs: &[FieldElement<F>],
        ys: &[FieldElement<F>],
    ) -> Result<Self, InterpolateError> {
        // TODO: try to use the type system to avoid this assert
        if xs.len() != ys.len() {
            return Err(InterpolateError::UnequalLengths(xs.len(), ys.len()));
        }
        if xs.is_empty() {
            return Ok(Polynomial::new(&[]));
        }

        let mut denominators = Vec::with_capacity(xs.len() * (xs.len() - 1) / 2);
        let mut indexes = Vec::with_capacity(xs.len());

        let mut idx = 0;

        for (i, xi) in xs.iter().enumerate().skip(1) {
            indexes.push(idx);
            for xj in xs.iter().take(i) {
                if xi == xj {
                    return Err(InterpolateError::NonUniqueXs);
                }
                denominators.push(xi - xj);
                idx += 1;
            }
        }

        FieldElement::inplace_batch_inverse(&mut denominators).unwrap();

        let mut result = Polynomial::zero();

        for (i, y) in ys.iter().enumerate() {
            let mut y_term = Polynomial::new(slice::from_ref(y));
            for (j, x) in xs.iter().enumerate() {
                if i == j {
                    continue;
                }
                let denominator = if i > j {
                    denominators[indexes[i - 1] + j].clone()
                } else {
                    -&denominators[indexes[j - 1] + i]
                };
                let denominator_poly = Polynomial::new(&[denominator]);
                let numerator = Polynomial::new(&[-x, FieldElement::one()]);
                y_term = y_term.mul_with_ref(&(numerator * denominator_poly));
            }
            result = result + y_term;
        }
        Ok(result)
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

    /// Returns the coefficient accompanying x^degree
    pub fn leading_coefficient(&self) -> FieldElement<F> {
        if let Some(coefficient) = self.coefficients.last() {
            coefficient.clone()
        } else {
            FieldElement::zero()
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

    /// Computes quotient with `x - b` in place.
    pub fn ruffini_division_inplace(&mut self, b: &FieldElement<F>) {
        let mut c = FieldElement::zero();
        for coeff in self.coefficients.iter_mut().rev() {
            *coeff = &*coeff + b * &c;
            core::mem::swap(coeff, &mut c);
        }
        self.coefficients.pop();
    }

    /// Computes quotient and remainder of polynomial division.
    ///
    /// Output: (quotient, remainder)
    pub fn long_division_with_remainder(self, dividend: &Self) -> (Self, Self) {
        if dividend.degree() > self.degree() {
            (Polynomial::zero(), self)
        } else {
            let mut n = self;
            let mut q: Vec<FieldElement<F>> = vec![FieldElement::zero(); n.degree() + 1];
            let denominator = dividend.leading_coefficient().inv().unwrap();
            while n != Polynomial::zero() && n.degree() >= dividend.degree() {
                let new_coefficient = n.leading_coefficient() * &denominator;
                q[n.degree() - dividend.degree()] = new_coefficient.clone();
                let d = dividend.mul_with_ref(&Polynomial::new_monomial(
                    new_coefficient,
                    n.degree() - dividend.degree(),
                ));
                n = n - d;
            }
            (Polynomial::new(&q), n)
        }
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

    /// Embeds the coefficients of a polynomial into an extension field
    /// For example, given a polynomial with coefficients in F_p, returns the same
    /// polynomial with its coefficients as elements in F_{p^2}
    pub fn to_extension<L: IsField>(self) -> Polynomial<FieldElement<L>>
    where
        F: IsSubFieldOf<L>,
    {
        Polynomial {
            coefficients: self
                .coefficients
                .into_iter()
                .map(|x| x.to_extension::<L>())
                .collect(),
        }
    }

    pub fn truncate(&self, k: usize) -> Self {
        if k == 0 {
            Self::zero()
        } else {
            Self::new(&self.coefficients[0..k.min(self.coefficients.len())])
        }
    }
    pub fn reverse(&self, d: usize) -> Self {
        let mut coeffs = self.coefficients.clone();
        coeffs.resize(d + 1, FieldElement::zero());
        coeffs.reverse();
        Self::new(&coeffs)
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

// impl Add
impl<F, L> ops::Add<&Polynomial<FieldElement<L>>> for &Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, a_polynomial: &Polynomial<FieldElement<L>>) -> Self::Output {
        let (pa, pb) = pad_with_zero_coefficients(self, a_polynomial);
        let iter_coeff_pa = pa.coefficients.iter();
        let iter_coeff_pb = pb.coefficients.iter();
        let new_coefficients = iter_coeff_pa.zip(iter_coeff_pb).map(|(x, y)| x + y);
        let new_coefficients_vec = new_coefficients.collect::<Vec<FieldElement<L>>>();
        Polynomial::new(&new_coefficients_vec)
    }
}

impl<F, L> ops::Add<Polynomial<FieldElement<L>>> for Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, a_polynomial: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self + &a_polynomial
    }
}

impl<F, L> ops::Add<&Polynomial<FieldElement<L>>> for Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, a_polynomial: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self + a_polynomial
    }
}

impl<F, L> ops::Add<Polynomial<FieldElement<L>>> for &Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, a_polynomial: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        self + &a_polynomial
    }
}

// impl neg, that is, additive inverse for polynomials P(t) + Q(t) = 0
impl<F: IsField> ops::Neg for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn neg(self) -> Polynomial<FieldElement<F>> {
        let neg = self
            .coefficients
            .iter()
            .map(|x| -x)
            .collect::<Vec<FieldElement<F>>>();
        Polynomial::new(&neg)
    }
}

impl<F: IsField> ops::Neg for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn neg(self) -> Polynomial<FieldElement<F>> {
        -&self
    }
}

// impl Sub
impl<F, L> ops::Sub<&Polynomial<FieldElement<L>>> for &Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, substrahend: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        self + (-substrahend)
    }
}

impl<F, L> ops::Sub<Polynomial<FieldElement<L>>> for Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, substrahend: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self - &substrahend
    }
}

impl<F, L> ops::Sub<&Polynomial<FieldElement<L>>> for Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, substrahend: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self - substrahend
    }
}

impl<F, L> ops::Sub<Polynomial<FieldElement<L>>> for &Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, substrahend: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        self - &substrahend
    }
}

impl<F: IsField> ops::Mul<&Polynomial<FieldElement<F>>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;
    fn mul(self, factor: &Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        self.mul_with_ref(factor)
    }
}

impl<F: IsField> ops::Mul<Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;
    fn mul(self, factor: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &self * &factor
    }
}

impl<F: IsField> ops::Mul<Polynomial<FieldElement<F>>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;
    fn mul(self, factor: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        self * &factor
    }
}

impl<F: IsField> ops::Mul<&Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;
    fn mul(self, factor: &Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &self * factor
    }
}

/* Operations between Polynomials and field elements */
/* Multiplication field element at left */
impl<F, L> ops::Mul<FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        let new_coefficients = self
            .coefficients
            .iter()
            .map(|value| &multiplicand * value)
            .collect();
        Polynomial {
            coefficients: new_coefficients,
        }
    }
}

impl<F, L> ops::Mul<&FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        self.clone() * multiplicand.clone()
    }
}

impl<F, L> ops::Mul<FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        self * &multiplicand
    }
}

impl<F, L> ops::Mul<&FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        &self * multiplicand
    }
}

/* Multiplication field element at right */
impl<F, L> ops::Mul<&Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        multiplicand * self
    }
}

impl<F, L> ops::Mul<Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &multiplicand * self
    }
}

impl<F, L> ops::Mul<&Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        multiplicand * self
    }
}

impl<F, L> ops::Mul<Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &multiplicand * &self
    }
}

/* Addition field element at left */
impl<F, L> ops::Add<&FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        Polynomial::new_monomial(other.clone(), 0) + self
    }
}

impl<F, L> ops::Add<FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        &self + &other
    }
}

impl<F, L> ops::Add<FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        self + &other
    }
}

impl<F, L> ops::Add<&FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        &self + other
    }
}

/* Addition field element at right */
impl<F, L> ops::Add<&Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        Polynomial::new_monomial(self.clone(), 0) + other
    }
}

impl<F, L> ops::Add<Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self + &other
    }
}

impl<F, L> ops::Add<Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        self + &other
    }
}

impl<F, L> ops::Add<&Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self + other
    }
}

/* Substraction field element at left */
impl<F, L> ops::Sub<&FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        -Polynomial::new_monomial(other.clone(), 0) + self
    }
}

impl<F, L> ops::Sub<FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        &self - &other
    }
}

impl<F, L> ops::Sub<FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        self - &other
    }
}

impl<F, L> ops::Sub<&FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        &self - other
    }
}

/* Substraction field element at right */
impl<F, L> ops::Sub<&Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        Polynomial::new_monomial(self.clone(), 0) - other
    }
}

impl<F, L> ops::Sub<Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self - &other
    }
}

impl<F, L> ops::Sub<Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        self - &other
    }
}

impl<F, L> ops::Sub<&Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self - other
    }
}

#[derive(Debug)]
pub enum InterpolateError {
    UnequalLengths(usize, usize),
    NonUniqueXs,
}

impl Display for InterpolateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InterpolateError::UnequalLengths(x, y) => {
                write!(f, "xs and ys must be the same length. Got: {x} != {y}")
            }
            InterpolateError::NonUniqueXs => write!(f, "xs values should be unique."),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InterpolateError {}

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
    let mut denoms: Vec<FieldElement<E>> = coset_points
        .iter()
        .map(|p| -(p - z))
        .collect();
    FieldElement::inplace_batch_inverse(&mut denoms)
        .expect("z is sampled to avoid coset points, so z - g*w^i is never zero");
    denoms
}

/// Evaluate a polynomial at point `z` given its evaluations on a coset `{g * w^i}`,
/// using the barycentric interpolation formula.
///
/// Formula: f(z) = (z^N - g^N) / N * sum_{i=0}^{N-1} [ (g*w^i) * f(g*w^i) / (z - g*w^i) ] / g^N
///
/// This variant takes base-field evaluations (main trace columns) and returns an
/// extension-field result.
///
/// # Arguments
/// * `z_pow_n` - z^N where N = coset_points.len()
/// * `coset_offset_pow_n` - g^N where g = coset_offset
/// * `n_inv` - 1/N in the extension field
/// * `coset_points` - the coset {g*w^0, g*w^1, ..., g*w^{N-1}}
/// * `evaluations` - f(g*w^i) for each coset point (base field)
/// * `inv_denoms` - precomputed 1/(z - g*w^i)
#[cfg(feature = "alloc")]
pub fn interpolate_coset_eval<F, E>(
    z_pow_n: &FieldElement<E>,
    coset_offset_pow_n: &FieldElement<E>,
    n_inv: &FieldElement<E>,
    coset_points: &[FieldElement<F>],
    evaluations: &[FieldElement<F>],
    inv_denoms: &[FieldElement<E>],
) -> FieldElement<E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    debug_assert_eq!(coset_points.len(), evaluations.len());
    debug_assert_eq!(coset_points.len(), inv_denoms.len());

    // sum = sum_{i} (g*w^i) * f(g*w^i) / (z - g*w^i)
    // point * eval is in base field F; multiply by inv_d lifts to E
    let sum: FieldElement<E> = coset_points
        .iter()
        .zip(evaluations.iter())
        .zip(inv_denoms.iter())
        .fold(FieldElement::<E>::zero(), |acc, ((point, eval), inv_d)| {
            acc + (point * eval) * inv_d
        });

    // f(z) = (z^N - g^N) / (N * g^N) * sum
    let vanishing = z_pow_n - coset_offset_pow_n;
    // coset_offset_pow_n is g^N where g is the non-zero coset offset
    let g_n_inv = coset_offset_pow_n
        .inv()
        .expect("coset_offset_pow_n is non-zero: g is the coset offset which is always non-zero");
    vanishing * n_inv * &sum * g_n_inv
}

/// Evaluate a polynomial at point `z` given its evaluations on a coset `{g * w^i}`,
/// using the barycentric interpolation formula.
///
/// This variant takes extension-field evaluations (aux trace / composition poly columns)
/// but keeps coset points in the base field to avoid unnecessary extension-field arithmetic.
/// The `point * eval` computation uses mixed F×E multiplication (cheaper than E×E).
#[cfg(feature = "alloc")]
pub fn interpolate_coset_eval_ext<F, E>(
    z_pow_n: &FieldElement<E>,
    coset_offset_pow_n: &FieldElement<E>,
    n_inv: &FieldElement<E>,
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

    let vanishing = z_pow_n - coset_offset_pow_n;
    // coset_offset_pow_n is g^N where g is the non-zero coset offset
    let g_n_inv = coset_offset_pow_n
        .inv()
        .expect("coset_offset_pow_n is non-zero: g is the coset offset which is always non-zero");
    vanishing * n_inv * &sum * g_n_inv
}

#[cfg(test)]
mod barycentric_tests {
    use super::*;
    use crate::fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset;
    use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;

    type FE = FieldElement<GoldilocksField>;

    /// Build a polynomial of degree < n with deterministic coefficients.
    fn test_poly(n: usize) -> Polynomial<FE> {
        let coeffs: Vec<FE> = (0..n).map(|i| FE::from((i * 7 + 3) as u64)).collect();
        Polynomial::new(&coeffs)
    }

    #[test]
    fn barycentric_matches_horner_simple() {
        let n = 8usize;
        let coset_offset = FE::from(3u64);
        let poly = test_poly(n);

        let root_order = n.trailing_zeros() as u64;
        let coset_points =
            get_powers_of_primitive_root_coset(root_order, n, &coset_offset).unwrap();
        let evaluations: Vec<FE> = coset_points.iter().map(|p| poly.evaluate(p)).collect();

        let z = FE::from(42u64);
        let expected = poly.evaluate(&z);

        let z_pow_n = z.pow(n);
        let g_pow_n = coset_offset.pow(n);
        let n_inv = FE::from(n as u64).inv().unwrap();
        let inv_denoms = barycentric_inv_denoms(&z, &coset_points);
        let result = interpolate_coset_eval(
            &z_pow_n,
            &g_pow_n,
            &n_inv,
            &coset_points,
            &evaluations,
            &inv_denoms,
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn barycentric_matches_horner_various_sizes() {
        for log_n in 2..=8 {
            let n = 1 << log_n;
            let coset_offset = FE::from(7u64);
            let poly = test_poly(n);

            let root_order = n.trailing_zeros() as u64;
            let coset_points =
                get_powers_of_primitive_root_coset(root_order, n, &coset_offset).unwrap();
            let evaluations: Vec<FE> = coset_points.iter().map(|p| poly.evaluate(p)).collect();

            let z = FE::from(123u64);
            let expected = poly.evaluate(&z);

            let z_pow_n = z.pow(n);
            let g_pow_n = coset_offset.pow(n);
            let n_inv = FE::from(n as u64).inv().unwrap();
            let inv_denoms = barycentric_inv_denoms(&z, &coset_points);
            let result = interpolate_coset_eval(
                &z_pow_n,
                &g_pow_n,
                &n_inv,
                &coset_points,
                &evaluations,
                &inv_denoms,
            );

            assert_eq!(result, expected, "failed for n={n}");
        }
    }

    #[test]
    fn barycentric_ext_matches_horner() {
        let n = 16usize;
        let coset_offset = FE::from(5u64);
        let poly = test_poly(n);

        let root_order = n.trailing_zeros() as u64;
        let coset_points =
            get_powers_of_primitive_root_coset(root_order, n, &coset_offset).unwrap();
        let evaluations: Vec<FE> = coset_points.iter().map(|p| poly.evaluate(p)).collect();

        let z = FE::from(999u64);
        let expected = poly.evaluate(&z);

        let z_pow_n = z.pow(n);
        let g_pow_n = coset_offset.pow(n);
        let n_inv = FE::from(n as u64).inv().unwrap();
        let inv_denoms = barycentric_inv_denoms(&z, &coset_points);
        let result = interpolate_coset_eval_ext(
            &z_pow_n,
            &g_pow_n,
            &n_inv,
            &coset_points,
            &evaluations,
            &inv_denoms,
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn barycentric_from_lde_stride() {
        // Simulate what the prover does: given an LDE of size n*bf, extract stride-bf
        // evaluations and use barycentric to evaluate at an OOD point
        let n = 8usize;
        let bf = 4usize;
        let coset_offset = FE::from(3u64);
        let poly = test_poly(n);

        let lde_root_order = (n * bf).trailing_zeros() as u64;
        let lde_coset =
            get_powers_of_primitive_root_coset(lde_root_order, n * bf, &coset_offset).unwrap();
        let lde_evals: Vec<FE> = lde_coset.iter().map(|p| poly.evaluate(p)).collect();

        // Extract trace-size coset (stride = bf)
        let trace_coset: Vec<FE> = (0..n).map(|i| lde_coset[i * bf].clone()).collect();
        let trace_evals: Vec<FE> = (0..n).map(|i| lde_evals[i * bf].clone()).collect();

        let z = FE::from(42u64);
        let expected = poly.evaluate(&z);

        let z_pow_n = z.pow(n);
        let g_pow_n = coset_offset.pow(n);
        let n_inv = FE::from(n as u64).inv().unwrap();
        let inv_denoms = barycentric_inv_denoms(&z, &trace_coset);
        let result = interpolate_coset_eval(
            &z_pow_n,
            &g_pow_n,
            &n_inv,
            &trace_coset,
            &trace_evals,
            &inv_denoms,
        );

        assert_eq!(result, expected);
    }
}
