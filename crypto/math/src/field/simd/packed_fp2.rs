//! SIMD-accelerated Goldilocks quadratic extension (Fp2) arithmetic.
//!
//! This module provides `PackedFp2x2`, a type that holds 2 Fp2 elements
//! and performs operations on them in parallel using SIMD operations.
//!
//! # Fp2 Field Structure
//!
//! Fp2 = Fp[x] / (x^2 - W) where W = 7 is the quadratic non-residue.
//! Elements are represented as a0 + a1*w where w^2 = 7.
//!
//! Multiplication: (a0 + a1*w)(b0 + b1*w) = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*w
//!
//! # Memory Layout
//!
//! Uses Struct-of-Arrays (SoA) layout for optimal SIMD utilization:
//! - real: [r0, r1] - real components of two Fp2 elements
//! - imag: [i0, i1] - imaginary components of two Fp2 elements

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::PackedGoldilocks2;
use crate::field::fields::u64_goldilocks_field::Goldilocks64Field;
use crate::field::traits::IsField;

/// The quadratic non-residue W = 7.
/// Fp2 is constructed as Fp[x] / (x^2 - 7)
pub const W: u64 = 7;

/// An Fp2 element represented as (real, imag) where element = real + imag * w.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fp2Raw {
    pub real: u64,
    pub imag: u64,
}

impl Fp2Raw {
    pub const ZERO: Self = Self { real: 0, imag: 0 };
    pub const ONE: Self = Self { real: 1, imag: 0 };

    #[inline(always)]
    pub fn new(real: u64, imag: u64) -> Self {
        Self { real, imag }
    }

    #[inline(always)]
    pub fn add(a: Self, b: Self) -> Self {
        Self {
            real: Goldilocks64Field::add(&a.real, &b.real),
            imag: Goldilocks64Field::add(&a.imag, &b.imag),
        }
    }

    #[inline(always)]
    pub fn sub(a: Self, b: Self) -> Self {
        Self {
            real: Goldilocks64Field::sub(&a.real, &b.real),
            imag: Goldilocks64Field::sub(&a.imag, &b.imag),
        }
    }

    #[inline(always)]
    pub fn neg(a: Self) -> Self {
        Self {
            real: Goldilocks64Field::neg(&a.real),
            imag: Goldilocks64Field::neg(&a.imag),
        }
    }

    /// Multiplies two Fp2 elements using Karatsuba multiplication.
    /// (a0 + a1*w)(b0 + b1*w) = (a0*b0 + W*a1*b1) + (a0*b1 + a1*b0)*w
    #[inline(always)]
    pub fn mul(a: Self, b: Self) -> Self {
        let a0b0 = Goldilocks64Field::mul(&a.real, &b.real);
        let a1b1 = Goldilocks64Field::mul(&a.imag, &b.imag);
        let sum_a = Goldilocks64Field::add(&a.real, &a.imag);
        let sum_b = Goldilocks64Field::add(&b.real, &b.imag);
        let z = Goldilocks64Field::mul(&sum_a, &sum_b);

        // W*a1*b1 = 7*a1*b1
        let w_a1b1 = mul_by_w_scalar(a1b1);

        Self {
            real: Goldilocks64Field::add(&a0b0, &w_a1b1),
            imag: Goldilocks64Field::sub(&Goldilocks64Field::sub(&z, &a0b0), &a1b1),
        }
    }
}

/// Multiply a scalar by W = 7.
#[inline(always)]
fn mul_by_w_scalar(x: u64) -> u64 {
    // 7x = 8x - x
    let x2 = Goldilocks64Field::double(&x);
    let x4 = Goldilocks64Field::double(&x2);
    let x8 = Goldilocks64Field::double(&x4);
    Goldilocks64Field::sub(&x8, &x)
}

/// A packed vector of 2 Fp2 (Goldilocks quadratic extension) elements.
///
/// Uses Struct-of-Arrays layout with two PackedGoldilocks2 vectors:
/// - `real`: real components [a0_real, a1_real]
/// - `imag`: imaginary components [a0_imag, a1_imag]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PackedFp2x2 {
    /// Real (coefficient of w^0) components
    pub real: PackedGoldilocks2,
    /// Imaginary (coefficient of w^1) components
    pub imag: PackedGoldilocks2,
}

impl PackedFp2x2 {
    /// Number of Fp2 elements packed in this type.
    pub const WIDTH: usize = 2;

    /// Zero element (additive identity).
    pub const ZERO: Self = Self {
        real: PackedGoldilocks2::ZERO,
        imag: PackedGoldilocks2::ZERO,
    };

    /// One element (multiplicative identity).
    pub const ONE: Self = Self {
        real: PackedGoldilocks2::ONE,
        imag: PackedGoldilocks2::ZERO,
    };

    /// Creates a new packed Fp2 element with all lanes set to the same value.
    #[inline(always)]
    pub fn broadcast(val: Fp2Raw) -> Self {
        Self {
            real: PackedGoldilocks2::broadcast(val.real),
            imag: PackedGoldilocks2::broadcast(val.imag),
        }
    }

    /// Creates a new packed element from an array of Fp2 elements.
    #[inline(always)]
    pub fn from_array(vals: [Fp2Raw; 2]) -> Self {
        Self {
            real: PackedGoldilocks2::from_array([vals[0].real, vals[1].real]),
            imag: PackedGoldilocks2::from_array([vals[0].imag, vals[1].imag]),
        }
    }

    /// Creates from raw u64 arrays.
    #[inline(always)]
    pub fn from_raw(reals: [u64; 2], imags: [u64; 2]) -> Self {
        Self {
            real: PackedGoldilocks2::from_array(reals),
            imag: PackedGoldilocks2::from_array(imags),
        }
    }

    /// Extracts the elements as an array of Fp2 elements.
    #[inline(always)]
    pub fn to_array(self) -> [Fp2Raw; 2] {
        let reals = self.real.to_array();
        let imags = self.imag.to_array();
        [
            Fp2Raw::new(reals[0], imags[0]),
            Fp2Raw::new(reals[1], imags[1]),
        ]
    }

    /// Extracts raw u64 arrays.
    #[inline(always)]
    pub fn to_raw(self) -> ([u64; 2], [u64; 2]) {
        (self.real.to_array(), self.imag.to_array())
    }

    /// Gets the element at the specified index.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> Fp2Raw {
        debug_assert!(idx < 2);
        let reals = self.real.to_array();
        let imags = self.imag.to_array();
        Fp2Raw::new(reals[idx], imags[idx])
    }

    /// Doubles the element.
    #[inline(always)]
    pub fn double(self) -> Self {
        Self {
            real: self.real.double(),
            imag: self.imag.double(),
        }
    }

    /// Squares the element.
    /// (a0 + a1*w)^2 = (a0^2 + W*a1^2) + 2*a0*a1*w
    #[inline(always)]
    pub fn square(self) -> Self {
        let a0_sq = self.real.square();
        let a1_sq = self.imag.square();
        let a0a1 = self.real * self.imag;

        // W*a1^2 = 7*a1^2
        let w_a1_sq = mul_by_w(a1_sq);

        Self {
            real: a0_sq + w_a1_sq,
            imag: a0a1.double(),
        }
    }

    /// Returns the conjugate: conjugate(a0 + a1*w) = a0 - a1*w
    #[inline(always)]
    pub fn conjugate(self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
        }
    }

    /// Multiplies by a base field element (scalar from Fp).
    #[inline(always)]
    pub fn mul_by_fp(self, scalar: PackedGoldilocks2) -> Self {
        Self {
            real: self.real * scalar,
            imag: self.imag * scalar,
        }
    }
}

/// Multiply a packed element by W = 7.
/// Uses the identity: 7x = 8x - x = (x << 3) - x
#[inline(always)]
fn mul_by_w(x: PackedGoldilocks2) -> PackedGoldilocks2 {
    let x2 = x.double();
    let x4 = x2.double();
    let x8 = x4.double();
    x8 - x
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

impl Add for PackedFp2x2 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real + rhs.real,
            imag: self.imag + rhs.imag,
        }
    }
}

impl AddAssign for PackedFp2x2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for PackedFp2x2 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real - rhs.real,
            imag: self.imag - rhs.imag,
        }
    }
}

impl SubAssign for PackedFp2x2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for PackedFp2x2 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self {
            real: -self.real,
            imag: -self.imag,
        }
    }
}

impl Mul for PackedFp2x2 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        // Karatsuba multiplication for Fp2:
        // (a0 + a1*w)(b0 + b1*w) = (a0*b0 + W*a1*b1) + (a0*b1 + a1*b0)*w
        //
        // Using Karatsuba:
        // z = (a0 + a1)(b0 + b1)
        // c0 = a0*b0 + W*a1*b1
        // c1 = z - a0*b0 - a1*b1 = a0*b1 + a1*b0

        let a0b0 = self.real * rhs.real;
        let a1b1 = self.imag * rhs.imag;
        let sum_a = self.real + self.imag;
        let sum_b = rhs.real + rhs.imag;
        let z = sum_a * sum_b;

        // W*a1*b1 = 7*a1*b1
        let w_a1b1 = mul_by_w(a1b1);

        Self {
            real: a0b0 + w_a1b1,
            imag: z - a0b0 - a1b1,
        }
    }
}

impl MulAssign for PackedFp2x2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl Default for PackedFp2x2 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Debug for PackedFp2x2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elems = self.to_array();
        f.debug_struct("PackedFp2x2")
            .field("0", &elems[0])
            .field("1", &elems[1])
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fp2(r: u64, i: u64) -> Fp2Raw {
        Fp2Raw::new(r, i)
    }

    #[test]
    fn test_from_to_array() {
        let a = fp2(111, 222);
        let b = fp2(333, 444);
        let packed = PackedFp2x2::from_array([a, b]);
        let unpacked = packed.to_array();
        assert_eq!(unpacked[0], a);
        assert_eq!(unpacked[1], b);
    }

    #[test]
    fn test_add() {
        let a0 = fp2(1, 2);
        let a1 = fp2(3, 4);
        let b0 = fp2(5, 6);
        let b1 = fp2(7, 8);

        let packed_a = PackedFp2x2::from_array([a0, a1]);
        let packed_b = PackedFp2x2::from_array([b0, b1]);
        let result = (packed_a + packed_b).to_array();

        assert_eq!(result[0], Fp2Raw::add(a0, b0));
        assert_eq!(result[1], Fp2Raw::add(a1, b1));
    }

    #[test]
    fn test_sub() {
        let a0 = fp2(10, 20);
        let a1 = fp2(30, 40);
        let b0 = fp2(1, 2);
        let b1 = fp2(3, 4);

        let packed_a = PackedFp2x2::from_array([a0, a1]);
        let packed_b = PackedFp2x2::from_array([b0, b1]);
        let result = (packed_a - packed_b).to_array();

        assert_eq!(result[0], Fp2Raw::sub(a0, b0));
        assert_eq!(result[1], Fp2Raw::sub(a1, b1));
    }

    #[test]
    fn test_neg() {
        let a0 = fp2(5, 10);
        let a1 = fp2(15, 20);

        let packed = PackedFp2x2::from_array([a0, a1]);
        let result = (-packed).to_array();

        assert_eq!(result[0], Fp2Raw::neg(a0));
        assert_eq!(result[1], Fp2Raw::neg(a1));
    }

    #[test]
    fn test_mul() {
        let a0 = fp2(2, 3);
        let a1 = fp2(4, 5);
        let b0 = fp2(6, 7);
        let b1 = fp2(8, 9);

        let packed_a = PackedFp2x2::from_array([a0, a1]);
        let packed_b = PackedFp2x2::from_array([b0, b1]);
        let result = (packed_a * packed_b).to_array();

        assert_eq!(result[0], Fp2Raw::mul(a0, b0));
        assert_eq!(result[1], Fp2Raw::mul(a1, b1));
    }

    #[test]
    fn test_mul_large() {
        // Test with larger values
        let p = super::super::packed_goldilocks::P;
        let a0 = fp2(p - 1, p - 2);
        let a1 = fp2(p - 3, p - 4);
        let b0 = fp2(p - 5, p - 6);
        let b1 = fp2(p - 7, p - 8);

        let packed_a = PackedFp2x2::from_array([a0, a1]);
        let packed_b = PackedFp2x2::from_array([b0, b1]);
        let result = (packed_a * packed_b).to_array();

        assert_eq!(result[0], Fp2Raw::mul(a0, b0));
        assert_eq!(result[1], Fp2Raw::mul(a1, b1));
    }

    #[test]
    fn test_square() {
        let a0 = fp2(7, 11);
        let a1 = fp2(13, 17);

        let packed = PackedFp2x2::from_array([a0, a1]);
        let squared = packed.square().to_array();
        let mul_self = (packed * packed).to_array();

        assert_eq!(squared[0], mul_self[0]);
        assert_eq!(squared[1], mul_self[1]);
    }

    #[test]
    fn test_double() {
        let a0 = fp2(5, 10);
        let a1 = fp2(15, 20);

        let packed = PackedFp2x2::from_array([a0, a1]);
        let doubled = packed.double().to_array();
        let added = (packed + packed).to_array();

        assert_eq!(doubled[0], added[0]);
        assert_eq!(doubled[1], added[1]);
    }

    #[test]
    fn test_conjugate() {
        let a0 = fp2(5, 10);
        let a1 = fp2(15, 20);

        let packed = PackedFp2x2::from_array([a0, a1]);
        let conj = packed.conjugate().to_array();

        assert_eq!(conj[0].real, 5);
        assert_eq!(conj[0].imag, Goldilocks64Field::neg(&10));
        assert_eq!(conj[1].real, 15);
        assert_eq!(conj[1].imag, Goldilocks64Field::neg(&20));
    }

    #[test]
    fn test_identity_elements() {
        let a = PackedFp2x2::from_array([fp2(123, 456), fp2(789, 101112)]);

        // Additive identity
        let sum = a + PackedFp2x2::ZERO;
        assert_eq!(sum, a);

        // Multiplicative identity
        let prod = a * PackedFp2x2::ONE;
        assert_eq!(prod.to_array()[0], a.to_array()[0]);
        assert_eq!(prod.to_array()[1], a.to_array()[1]);
    }

    #[test]
    fn test_algebraic_properties() {
        let a = PackedFp2x2::from_array([fp2(123, 456), fp2(789, 101)]);
        let b = PackedFp2x2::from_array([fp2(111, 222), fp2(333, 444)]);
        let c = PackedFp2x2::from_array([fp2(555, 666), fp2(777, 888)]);

        // Commutativity of addition
        assert_eq!((a + b).to_array(), (b + a).to_array());

        // Commutativity of multiplication
        assert_eq!((a * b).to_array(), (b * a).to_array());

        // Associativity of addition
        assert_eq!(((a + b) + c).to_array(), (a + (b + c)).to_array());

        // Associativity of multiplication
        assert_eq!(((a * b) * c).to_array(), (a * (b * c)).to_array());

        // Distributivity
        assert_eq!((a * (b + c)).to_array(), (a * b + a * c).to_array());

        // Additive inverse
        let neg_a = -a;
        let zero = (a + neg_a).to_array();
        assert_eq!(zero[0], Fp2Raw::ZERO);
        assert_eq!(zero[1], Fp2Raw::ZERO);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    const P: u64 = super::super::packed_goldilocks::P;

    fn arb_fp2() -> impl Strategy<Value = Fp2Raw> {
        (0..P, 0..P).prop_map(|(r, i)| Fp2Raw::new(r, i))
    }

    fn arb_packed_fp2() -> impl Strategy<Value = PackedFp2x2> {
        (arb_fp2(), arb_fp2()).prop_map(|(a, b)| PackedFp2x2::from_array([a, b]))
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_add_matches_scalar(
            a0 in arb_fp2(),
            a1 in arb_fp2(),
            b0 in arb_fp2(),
            b1 in arb_fp2()
        ) {
            let packed_a = PackedFp2x2::from_array([a0, a1]);
            let packed_b = PackedFp2x2::from_array([b0, b1]);
            let result = (packed_a + packed_b).to_array();

            let expected0 = Fp2Raw::add(a0, b0);
            let expected1 = Fp2Raw::add(a1, b1);

            prop_assert_eq!(result[0], expected0);
            prop_assert_eq!(result[1], expected1);
        }

        #[test]
        fn prop_sub_matches_scalar(
            a0 in arb_fp2(),
            a1 in arb_fp2(),
            b0 in arb_fp2(),
            b1 in arb_fp2()
        ) {
            let packed_a = PackedFp2x2::from_array([a0, a1]);
            let packed_b = PackedFp2x2::from_array([b0, b1]);
            let result = (packed_a - packed_b).to_array();

            let expected0 = Fp2Raw::sub(a0, b0);
            let expected1 = Fp2Raw::sub(a1, b1);

            prop_assert_eq!(result[0], expected0);
            prop_assert_eq!(result[1], expected1);
        }

        #[test]
        fn prop_mul_matches_scalar(
            a0 in arb_fp2(),
            a1 in arb_fp2(),
            b0 in arb_fp2(),
            b1 in arb_fp2()
        ) {
            let packed_a = PackedFp2x2::from_array([a0, a1]);
            let packed_b = PackedFp2x2::from_array([b0, b1]);
            let result = (packed_a * packed_b).to_array();

            let expected0 = Fp2Raw::mul(a0, b0);
            let expected1 = Fp2Raw::mul(a1, b1);

            prop_assert_eq!(result[0], expected0, "mul lane 0 mismatch");
            prop_assert_eq!(result[1], expected1, "mul lane 1 mismatch");
        }

        #[test]
        fn prop_square_equals_mul_self(a in arb_packed_fp2()) {
            let squared = a.square().to_array();
            let mul_self = (a * a).to_array();
            prop_assert_eq!(squared[0], mul_self[0]);
            prop_assert_eq!(squared[1], mul_self[1]);
        }

        #[test]
        fn prop_double_equals_add_self(a in arb_packed_fp2()) {
            let doubled = a.double().to_array();
            let added = (a + a).to_array();
            prop_assert_eq!(doubled[0], added[0]);
            prop_assert_eq!(doubled[1], added[1]);
        }

        #[test]
        fn prop_add_commutative(a in arb_packed_fp2(), b in arb_packed_fp2()) {
            prop_assert_eq!((a + b).to_array(), (b + a).to_array());
        }

        #[test]
        fn prop_mul_commutative(a in arb_packed_fp2(), b in arb_packed_fp2()) {
            prop_assert_eq!((a * b).to_array(), (b * a).to_array());
        }

        #[test]
        fn prop_distributive(a in arb_packed_fp2(), b in arb_packed_fp2(), c in arb_packed_fp2()) {
            prop_assert_eq!((a * (b + c)).to_array(), (a * b + a * c).to_array());
        }

        #[test]
        fn prop_additive_inverse(a in arb_packed_fp2()) {
            let neg_a = -a;
            let result = (a + neg_a).to_array();
            prop_assert_eq!(result[0], Fp2Raw::ZERO);
            prop_assert_eq!(result[1], Fp2Raw::ZERO);
        }
    }
}
