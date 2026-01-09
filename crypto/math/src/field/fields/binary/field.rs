use core::cmp::Ordering;
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

// Implementation of binary fields of the form GF(2^{2^n}) (i.e. a finite field of 2^{2^n} elements) by constructing a tower of field extensions.
// The basic idea is to represent an element of each field as a multi-variable polynomial with binary coefficients in GF(2) = {0, 1}.
// The coefficients of each polynomial are stored as bits in a `u128` integer.
// The tower structure is built recursively, with each level representing an extension of the previous field.
// In each level n, polynomials have n variables that satisfy:
// (x_i)² = x_i * x_{i-1} + 1

// For more details, see:
// - Lambdaclass blog post about the use of binary fields in SNARKs: https://blog.lambdaclass.com/snarks-on-binary-fields-binius/
// - Vitalik Buterin's Binius: https://vitalik.eth.limo/general/2024/04/29/binius.html

#[derive(Debug)]
pub enum BinaryFieldError {
    /// Attempt to compute inverse of zero
    InverseOfZero,
}

#[derive(Clone, Copy, Debug)]
/// An element in the tower of binary field extensions from level 0 to level 7.
///
/// Implements arithmetic in finite fields GF(2^{2^n}) where n is the level of the field extension in the tower.
///
/// The internal representation stores polynomial coefficients as bits in a u128 integer.
#[derive(Default)]
pub struct TowerFieldElement {
    /// The value of the element.
    /// The binary expression of this value represents the coefficients of the corresponding polynomial of the element.
    /// For example, if value = 0b1101, then p = xy + y + 1. If value = 0b0110, then p = y + x.
    pub value: u128,
    /// Number of the level in the tower.
    /// It tells us to which field extension the element belongs.
    /// It goes from 0 (representing the base field of two elements) to 7 (representing the field extension of 2^128 elements).
    pub num_level: usize,
}

impl TowerFieldElement {
    /// Constructor that always succeeds by masking the value if it is too big for the given
    /// num_level, and limiting the level so that is not greater than 7.
    pub fn new(val: u128, num_level: usize) -> Self {
        // Limit num_level to a maximum valid value for u128.
        let safe_level = if num_level > 7 { 7 } else { num_level };

        // The number of bits needed for the given level
        let bits = 1 << safe_level;
        let mask = if bits >= 128 {
            u128::MAX
        } else {
            (1 << bits) - 1
        };

        Self {
            // We take just the lsb of val that fit in the extension field we are.
            value: val & mask,
            num_level: safe_level,
        }
    }

    /// Returns true if the element is zero
    pub fn is_zero(&self) -> bool {
        self.value == 0
    }

    /// Returns true if this element is one
    #[inline]
    pub fn is_one(&self) -> bool {
        self.value == 1
    }

    /// Returns the underlying value
    #[inline]
    pub fn value(&self) -> u128 {
        self.value
    }

    /// Returns level number in the tower.
    #[inline]
    pub fn num_level(&self) -> usize {
        self.num_level
    }

    /// Returns the number of bits needed for that level (2^num_levels).
    /// Note that the order of the extension field in that level is 2^num_bits.
    #[inline]
    pub fn num_bits(&self) -> usize {
        1 << self.num_level()
    }

    /// Returns binary string representation
    #[cfg(feature = "std")]
    pub fn to_binary_string(&self) -> String {
        format!("{:0width$b}", self.value, width = self.num_bits())
    }

    /// Splits element into high and low parts.
    /// For example, if a = xy + y + x, then a = (x + 1)y + x and
    /// therefore, a_hi = x + 1 and a_lo = x.
    pub fn split(&self) -> (Self, Self) {
        let half_bits = self.num_bits() / 2;
        let mask = (1 << half_bits) - 1;
        let lo = self.value() & mask;
        let hi = (self.value() >> half_bits) & mask;

        (
            Self::new(hi, self.num_level() - 1),
            Self::new(lo, self.num_level() - 1),
        )
    }

    /// Joins the hi and low part making a new element of a bigger level.
    /// For example, if a_hi = x and a_low = 1
    /// then a = xy + 1.
    pub fn join(&self, low: &Self) -> Self {
        let joined = (self.value() << self.num_bits()) | low.value();
        Self::new(joined, self.num_level() + 1)
    }

    // It embeds an element in an extension changing the level number.
    pub fn extend_num_level(&mut self, new_level: usize) {
        if self.num_level() < new_level {
            self.num_level = new_level;
        }
    }

    /// Create a zero element
    pub fn zero() -> Self {
        Self::new(0, 0)
    }

    /// Create a one element
    pub fn one() -> Self {
        Self::new(1, 0)
    }

    /// Addition between elements of same or different levels.
    fn add_elements(&self, other: &Self) -> Self {
        let num_level = self.num_level().max(other.num_level());
        Self::new(self.value() ^ other.value(), num_level)
    }

    // Multiplies a and b in the following way:
    //
    // - If a and b are from the same level:
    // a = a_hi * x_n + a_lo
    // b = b_hi * x_n + b_lo
    // Then a * b = (b_hi * a_hi * x_{n-1} + b_hi * a_lo + a_hi * b_lo ) * x_n + b_hi * a_hi + a_lo * b_lo.
    // We calculate each product in the equation below using recursion.
    //
    // - if a's level is larger than b's level, we partition a until we have parts of the size of b and
    // multiply each part by b.
    fn mul(self, other: Self) -> Self {
        match self.num_level().cmp(&other.num_level()) {
            Ordering::Greater => {
                // We split a into two parts and call the same method to multiply each part by b.
                let (a_hi, a_lo) = self.split();
                // Join a_hi * b and a_lo * b.
                a_hi.mul(other).join(&a_lo.mul(other))
            }
            Ordering::Less => {
                // If b is larger than a, we swap the arguments and call the same method.
                other.mul(self)
            }
            Ordering::Equal => {
                // Base case:
                if self.num_level() == 0 {
                    // In the binary base field, multiplication is the same as AND operation.
                    return Self::new(self.value() & other.value(), 0);
                }

                // Split both elements into high and low parts
                let (a_high, a_low) = self.split();
                let (b_high, b_low) = other.split();

                // Step 1: Compute sub-products
                let low_product = a_low.mul(b_low); // a_low * b_low
                let high_product = a_high.mul(b_high); // a_high * b_high

                // Step 2: Get the polynomial x_{n-1} value
                let x_value = if self.num_level() == 1 {
                    Self::new(1, 0)
                } else {
                    Self::new(1 << (self.num_bits() / 4), self.num_level() - 1)
                };

                // Step 3: Compute high_product * x_{n-1}
                let shifted_high_product = high_product.mul(x_value);

                // Step 4: Karatsuba optimization for middle term
                // Instead of computing a_high * b_low + a_low * b_high directly,
                // we use (a_low + a_high) * (b_low + b_high) - low_product - high_product
                let sum_product = (a_low + a_high).mul(b_low + b_high);
                let middle_term = sum_product - low_product - high_product;

                // Step 5: Join the parts according to the tower field multiplication formula
                (shifted_high_product + middle_term).join(&(high_product + low_product))
            }
        }
    }

    /// Computes the multiplicative inverse using Fermat's little theorem.
    /// Returns an error if the element is zero.
    // Based on Ingoyama's implementation
    // https://github.com/ingonyama-zk/smallfield-super-sumcheck/blob/a8c61beef39bc0c10a8f68d25eeac0a7190a7289/src/tower_fields/binius.rs#L116C5-L116C6
    pub fn inv(&self) -> Result<Self, BinaryFieldError> {
        if self.is_zero() {
            return Err(BinaryFieldError::InverseOfZero);
        }
        if self.num_level() <= 1 || self.num_bits() <= 4 {
            let exponent = (1 << self.num_bits()) - 2;
            Ok(Self::pow(self, exponent as u32))
        } else {
            let (a_hi, a_lo) = self.split();
            let two_pow_k_minus_one = Self::new(1 << (self.num_bits() / 4), self.num_level() - 1);
            // a = a_hi * x^k + a_lo
            // a_lo_next = a_hi * x^(k-1) + a_lo
            let a_lo_next = a_lo + a_hi * two_pow_k_minus_one;

            // Δ = a_lo * a_lo_next + a_hi^2
            let delta = a_lo * a_lo_next + a_hi * a_hi;

            // Compute inverse of delta recursively
            let delta_inverse = delta.inv()?;

            // Compute parts of the inverse
            let out_hi = delta_inverse * a_hi;
            let out_lo = delta_inverse * a_lo_next;

            // Join the parts to get the final inverse
            Ok(out_hi.join(&out_lo))
        }
    }

    /// Calculate power.
    pub fn pow(&self, exp: u32) -> Self {
        let mut result = Self::one();
        let mut base = *self;
        let mut exp_val = exp;

        while exp_val > 0 {
            if exp_val & 1 == 1 {
                result *= base;
            }
            base = base * base;
            exp_val >>= 1;
        }

        result
    }
}

impl PartialEq<TowerFieldElement> for TowerFieldElement {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl Eq for TowerFieldElement {}

impl Add for TowerFieldElement {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // Use the helper method that takes references
        self.add_elements(&other)
    }
}

impl<'a> Add<&'a TowerFieldElement> for &'a TowerFieldElement {
    type Output = TowerFieldElement;

    fn add(self, other: &'a TowerFieldElement) -> TowerFieldElement {
        // Directly use the helper method
        self.add_elements(other)
    }
}

impl AddAssign for TowerFieldElement {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl Sub for TowerFieldElement {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        // In binary fields, subtraction is the same as addition
        self + other
    }
}

impl Neg for TowerFieldElement {
    type Output = Self;

    fn neg(self) -> Self {
        // In binary fields, negation is the identity
        self
    }
}

impl Mul for TowerFieldElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.mul(other)
    }
}

impl Mul<&TowerFieldElement> for &TowerFieldElement {
    type Output = TowerFieldElement;

    fn mul(self, other: &TowerFieldElement) -> TowerFieldElement {
        <TowerFieldElement as Mul<TowerFieldElement>>::mul(*self, *other)
    }
}

impl MulAssign for TowerFieldElement {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl Product for TowerFieldElement {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl Sum for TowerFieldElement {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl fmt::Display for TowerFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl From<u128> for TowerFieldElement {
    fn from(val: u128) -> Self {
        TowerFieldElement::new(val, 7)
    }
}

impl From<u64> for TowerFieldElement {
    fn from(val: u64) -> Self {
        TowerFieldElement::new(val as u128, 6)
    }
}

impl From<u32> for TowerFieldElement {
    fn from(val: u32) -> Self {
        TowerFieldElement::new(val as u128, 5)
    }
}

impl From<u16> for TowerFieldElement {
    fn from(val: u16) -> Self {
        TowerFieldElement::new(val as u128, 4)
    }
}

impl From<u8> for TowerFieldElement {
    fn from(val: u8) -> Self {
        TowerFieldElement::new(val as u128, 3)
    }
}
