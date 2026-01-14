use super::field::element::FieldElement;
use crate::field::traits::{IsField, IsPrimeField, IsSubFieldOf};
use alloc::string::{String, ToString};
use alloc::{borrow::ToOwned, format, vec, vec::Vec};
use core::{fmt::Display, ops, slice};
pub mod dense_multilinear_poly;
mod error;
pub mod sparse_multilinear_poly;

/// Precomputes powers of x: `[1, x, x^2, x^3, ..., x^(n-1)]`
/// This is useful for batch polynomial evaluation at the same point.
pub fn precompute_powers<F: IsField>(x: &FieldElement<F>, n: usize) -> Vec<FieldElement<F>> {
    if n == 0 {
        return vec![];
    }
    let mut powers = Vec::with_capacity(n);
    powers.push(FieldElement::one());
    for i in 1..n {
        let prev = &powers[i - 1];
        powers.push(prev * x);
    }
    powers
}

/// Evaluates multiple polynomials at the same point using precomputed powers.
/// More efficient than calling `evaluate` on each polynomial individually.
pub fn evaluate_polynomials_at_point<F, E>(
    polynomials: &[Polynomial<FieldElement<F>>],
    x: &FieldElement<E>,
) -> Vec<FieldElement<E>>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    if polynomials.is_empty() {
        return vec![];
    }
    // Find the maximum degree among all polynomials
    let max_len = polynomials.iter().map(|p| p.coeff_len()).max().unwrap_or(0);
    // Precompute powers once
    let powers = precompute_powers(x, max_len);
    // Evaluate each polynomial using the precomputed powers
    polynomials
        .iter()
        .map(|p| p.evaluate_with_powers(&powers))
        .collect()
}

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

    /// Evaluates a polynomial using precomputed powers of x.
    /// The powers slice should contain `[1, x, x^2, ..., x^(n-1)]` where n >= coeff_len().
    /// This is more efficient when evaluating multiple polynomials at the same point.
    pub fn evaluate_with_powers<E>(&self, powers: &[FieldElement<E>]) -> FieldElement<E>
    where
        E: IsField,
        F: IsSubFieldOf<E>,
    {
        debug_assert!(
            powers.len() >= self.coefficients.len(),
            "Not enough powers for polynomial evaluation"
        );
        self.coefficients
            .iter()
            .zip(powers.iter())
            .fold(FieldElement::zero(), |acc, (coeff, power)| {
                acc + coeff * power
            })
    }

    /// Evaluates a polynomial P(t) at a slice of points x
    /// Returns a vector y such that y[i] = P(input[i])
    pub fn evaluate_slice(&self, input: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
        input.iter().map(|x| self.evaluate(x)).collect()
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

    /// Returns the derivative of the polynomial with respect to x.
    pub fn differentiate(&self) -> Self {
        let degree = self.degree();
        if degree == 0 {
            return Polynomial::zero();
        }
        let mut derivative = Vec::with_capacity(degree);
        for (i, coeff) in self.coefficients().iter().enumerate().skip(1) {
            derivative.push(FieldElement::<F>::from(i as u64) * coeff);
        }
        Polynomial::new(&derivative)
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

    /// Computes the quotient of the division of P(x) with x - b using Ruffini's rule
    pub fn ruffini_division<L>(&self, b: &FieldElement<L>) -> Polynomial<FieldElement<L>>
    where
        L: IsField,
        F: IsSubFieldOf<L>,
    {
        if let Some(c) = self.coefficients.last() {
            let mut c = c.clone().to_extension();
            let mut coefficients = Vec::with_capacity(self.degree());
            for coeff in self.coefficients.iter().rev().skip(1) {
                coefficients.push(c.clone());
                c = coeff + c * b;
            }
            coefficients = coefficients.into_iter().rev().collect();
            Polynomial::new(&coefficients)
        } else {
            Polynomial::zero()
        }
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

    /// Extended Euclidean Algorithm for polynomials.
    ///
    /// This method computes the extended greatest common divisor (GCD) of two polynomials `self` and `y`.
    /// It returns a tuple of three elements: `(a, b, g)` such that `a * self + b * y = g`, where `g` is the
    /// greatest common divisor of `self` and `y`.
    pub fn xgcd(&self, y: &Self) -> (Self, Self, Self) {
        let one = Polynomial::new(&[FieldElement::one()]);
        let zero = Polynomial::zero();
        let (mut old_r, mut r) = (self.clone(), y.clone());
        let (mut old_s, mut s) = (one.clone(), zero.clone());
        let (mut old_t, mut t) = (zero.clone(), one.clone());

        while r != Polynomial::zero() {
            let quotient = old_r.clone().div_with_ref(&r);
            old_r = old_r - &quotient * &r;
            core::mem::swap(&mut old_r, &mut r);
            old_s = old_s - &quotient * &s;
            core::mem::swap(&mut old_s, &mut s);
            old_t = old_t - &quotient * &t;
            core::mem::swap(&mut old_t, &mut t);
        }

        let lcinv = old_r.leading_coefficient().inv().unwrap();
        (
            old_s.scale_coeffs(&lcinv),
            old_t.scale_coeffs(&lcinv),
            old_r.scale_coeffs(&lcinv),
        )
    }

    pub fn div_with_ref(self, dividend: &Self) -> Self {
        let (quotient, _remainder) = self.long_division_with_remainder(dividend);
        quotient
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

impl<F: IsPrimeField> Polynomial<FieldElement<F>> {
    // Print the polynomial as a string ready to be used in SageMath, or just for pretty printing.
    pub fn print_as_sage_poly(&self, var_name: Option<char>) -> String {
        let var_name = var_name.unwrap_or('x');
        if self.coefficients.is_empty()
            || self.coefficients.len() == 1 && self.coefficients[0] == FieldElement::zero()
        {
            return String::new();
        }

        let mut string = String::new();
        let zero = FieldElement::<F>::zero();

        for (i, coeff) in self.coefficients.iter().rev().enumerate() {
            if *coeff == zero {
                continue;
            }

            let coeff_str = coeff.representative().to_string();

            if i == self.coefficients.len() - 1 {
                string.push_str(&coeff_str);
            } else if i == self.coefficients.len() - 2 {
                string.push_str(&format!("{coeff_str}*{var_name} + "));
            } else {
                string.push_str(&format!(
                    "{}*{}^{} + ",
                    coeff_str,
                    var_name,
                    self.coefficients.len() - 1 - i
                ));
            }
        }

        string
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

/// Computes the composition of polynomials P1(t) and P2(t), that is P1(P2(t))
/// It uses interpolation to determine the evaluation at points x_i and evaluates
/// P1(P2(x[i])). The interpolation theorem ensures that we can reconstruct the polynomial
/// uniquely by interpolation over a suitable number of points
/// This is an inefficient version, for something more efficient, use FFT for evaluation,
/// provided the field satisfies the necessary traits
pub fn compose<F>(
    poly_1: &Polynomial<FieldElement<F>>,
    poly_2: &Polynomial<FieldElement<F>>,
) -> Polynomial<FieldElement<F>>
where
    F: IsField,
{
    let max_degree: u64 = (poly_1.degree() * poly_2.degree()) as u64;

    let mut interpolation_points = vec![];
    for i in 0_u64..max_degree + 1 {
        interpolation_points.push(FieldElement::<F>::from(i));
    }

    let values: Vec<_> = interpolation_points
        .iter()
        .map(|value| {
            let intermediate_value = poly_2.evaluate(value);
            poly_1.evaluate(&intermediate_value)
        })
        .collect();

    Polynomial::interpolate(interpolation_points.as_slice(), values.as_slice())
        .expect("xs and ys have equal length and xs are unique")
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

impl<F> ops::Div<Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>>
where
    F: IsField,
{
    type Output = Polynomial<FieldElement<F>>;

    fn div(self, dividend: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        self.div_with_ref(&dividend)
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
