//! SIMD-accelerated Goldilocks cubic extension (Fp3) arithmetic.
//!
//! This module provides `PackedFp3x2`, a type that holds 2 Fp3 elements
//! and performs operations on them in parallel using SIMD operations.
//!
//! # Fp3 Field Structure
//!
//! Fp3 = Fp[x] / (x^3 - W) where W = 2 is the cubic non-residue.
//! Elements are represented as a0 + a1*w + a2*w^2 where w^3 = 2.
//!
//! # Memory Layout
//!
//! Uses Struct-of-Arrays (SoA) layout for optimal SIMD utilization:
//! - c0: [c0_0, c0_1] - coefficient of w^0 for two Fp3 elements
//! - c1: [c1_0, c1_1] - coefficient of w^1 for two Fp3 elements
//! - c2: [c2_0, c2_1] - coefficient of w^2 for two Fp3 elements

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::PackedGoldilocks2;
use crate::field::fields::u64_goldilocks_field::Goldilocks64Field;
use crate::field::traits::IsField;

/// The cubic non-residue W = 2.
/// Fp3 is constructed as Fp[x] / (x^3 - 2)
pub const W: u64 = 2;

/// An Fp3 element represented as (c0, c1, c2) where element = c0 + c1*w + c2*w^2.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fp3Raw {
    pub c0: u64,
    pub c1: u64,
    pub c2: u64,
}

impl Fp3Raw {
    pub const ZERO: Self = Self { c0: 0, c1: 0, c2: 0 };
    pub const ONE: Self = Self { c0: 1, c1: 0, c2: 0 };

    #[inline(always)]
    pub fn new(c0: u64, c1: u64, c2: u64) -> Self {
        Self { c0, c1, c2 }
    }

    #[inline(always)]
    pub fn add(a: Self, b: Self) -> Self {
        Self {
            c0: Goldilocks64Field::add(&a.c0, &b.c0),
            c1: Goldilocks64Field::add(&a.c1, &b.c1),
            c2: Goldilocks64Field::add(&a.c2, &b.c2),
        }
    }

    #[inline(always)]
    pub fn sub(a: Self, b: Self) -> Self {
        Self {
            c0: Goldilocks64Field::sub(&a.c0, &b.c0),
            c1: Goldilocks64Field::sub(&a.c1, &b.c1),
            c2: Goldilocks64Field::sub(&a.c2, &b.c2),
        }
    }

    #[inline(always)]
    pub fn neg(a: Self) -> Self {
        Self {
            c0: Goldilocks64Field::neg(&a.c0),
            c1: Goldilocks64Field::neg(&a.c1),
            c2: Goldilocks64Field::neg(&a.c2),
        }
    }

    /// Multiplies two Fp3 elements.
    /// (a0 + a1*w + a2*w^2) * (b0 + b1*w + b2*w^2) mod (w^3 - 2)
    ///
    /// Using Karatsuba-like optimization with W=2:
    /// c0 = v0 + 2 * ((a1 + a2)(b1 + b2) - v1 - v2)
    /// c1 = (a0 + a1)(b0 + b1) - v0 - v1 + 2 * v2
    /// c2 = (a0 + a2)(b0 + b2) - v0 + v1 - v2
    #[inline(always)]
    pub fn mul(a: Self, b: Self) -> Self {
        let v0 = Goldilocks64Field::mul(&a.c0, &b.c0);
        let v1 = Goldilocks64Field::mul(&a.c1, &b.c1);
        let v2 = Goldilocks64Field::mul(&a.c2, &b.c2);

        let a1_plus_a2 = Goldilocks64Field::add(&a.c1, &a.c2);
        let b1_plus_b2 = Goldilocks64Field::add(&b.c1, &b.c2);
        let a0_plus_a1 = Goldilocks64Field::add(&a.c0, &a.c1);
        let b0_plus_b1 = Goldilocks64Field::add(&b.c0, &b.c1);
        let a0_plus_a2 = Goldilocks64Field::add(&a.c0, &a.c2);
        let b0_plus_b2 = Goldilocks64Field::add(&b.c0, &b.c2);

        // c0 = v0 + 2 * ((a1 + a2)(b1 + b2) - v1 - v2)
        let cross_12 = Goldilocks64Field::mul(&a1_plus_a2, &b1_plus_b2);
        let cross_12_minus_v1_v2 = Goldilocks64Field::sub(
            &Goldilocks64Field::sub(&cross_12, &v1),
            &v2,
        );
        let c0 = Goldilocks64Field::add(&v0, &Goldilocks64Field::double(&cross_12_minus_v1_v2));

        // c1 = (a0 + a1)(b0 + b1) - v0 - v1 + 2 * v2
        let cross_01 = Goldilocks64Field::mul(&a0_plus_a1, &b0_plus_b1);
        let c1 = Goldilocks64Field::add(
            &Goldilocks64Field::sub(&Goldilocks64Field::sub(&cross_01, &v0), &v1),
            &Goldilocks64Field::double(&v2),
        );

        // c2 = (a0 + a2)(b0 + b2) - v0 + v1 - v2
        let cross_02 = Goldilocks64Field::mul(&a0_plus_a2, &b0_plus_b2);
        let c2 = Goldilocks64Field::sub(
            &Goldilocks64Field::add(&Goldilocks64Field::sub(&cross_02, &v0), &v1),
            &v2,
        );

        Self { c0, c1, c2 }
    }

    /// Multiplies by a scalar from Fp.
    #[inline(always)]
    pub fn mul_by_fp(a: Self, scalar: u64) -> Self {
        Self {
            c0: Goldilocks64Field::mul(&a.c0, &scalar),
            c1: Goldilocks64Field::mul(&a.c1, &scalar),
            c2: Goldilocks64Field::mul(&a.c2, &scalar),
        }
    }
}

/// A packed vector of 2 Fp3 (Goldilocks cubic extension) elements.
///
/// Uses Struct-of-Arrays layout with three PackedGoldilocks2 vectors.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PackedFp3x2 {
    /// Coefficient of w^0
    pub c0: PackedGoldilocks2,
    /// Coefficient of w^1
    pub c1: PackedGoldilocks2,
    /// Coefficient of w^2
    pub c2: PackedGoldilocks2,
}

impl PackedFp3x2 {
    /// Number of Fp3 elements packed in this type.
    pub const WIDTH: usize = 2;

    /// Zero element (additive identity).
    pub const ZERO: Self = Self {
        c0: PackedGoldilocks2::ZERO,
        c1: PackedGoldilocks2::ZERO,
        c2: PackedGoldilocks2::ZERO,
    };

    /// One element (multiplicative identity).
    pub const ONE: Self = Self {
        c0: PackedGoldilocks2::ONE,
        c1: PackedGoldilocks2::ZERO,
        c2: PackedGoldilocks2::ZERO,
    };

    /// Creates a new packed Fp3 element with all lanes set to the same value.
    #[inline(always)]
    pub fn broadcast(val: Fp3Raw) -> Self {
        Self {
            c0: PackedGoldilocks2::broadcast(val.c0),
            c1: PackedGoldilocks2::broadcast(val.c1),
            c2: PackedGoldilocks2::broadcast(val.c2),
        }
    }

    /// Creates a new packed element from an array of Fp3 elements.
    #[inline(always)]
    pub fn from_array(vals: [Fp3Raw; 2]) -> Self {
        Self {
            c0: PackedGoldilocks2::from_array([vals[0].c0, vals[1].c0]),
            c1: PackedGoldilocks2::from_array([vals[0].c1, vals[1].c1]),
            c2: PackedGoldilocks2::from_array([vals[0].c2, vals[1].c2]),
        }
    }

    /// Extracts the elements as an array of Fp3 elements.
    #[inline(always)]
    pub fn to_array(self) -> [Fp3Raw; 2] {
        let c0s = self.c0.to_array();
        let c1s = self.c1.to_array();
        let c2s = self.c2.to_array();
        [
            Fp3Raw::new(c0s[0], c1s[0], c2s[0]),
            Fp3Raw::new(c0s[1], c1s[1], c2s[1]),
        ]
    }

    /// Gets the element at the specified index.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> Fp3Raw {
        debug_assert!(idx < 2);
        let c0s = self.c0.to_array();
        let c1s = self.c1.to_array();
        let c2s = self.c2.to_array();
        Fp3Raw::new(c0s[idx], c1s[idx], c2s[idx])
    }

    /// Doubles the element.
    #[inline(always)]
    pub fn double(self) -> Self {
        Self {
            c0: self.c0.double(),
            c1: self.c1.double(),
            c2: self.c2.double(),
        }
    }

    /// Multiplies by a base field element (scalar from Fp).
    #[inline(always)]
    pub fn mul_by_fp(self, scalar: PackedGoldilocks2) -> Self {
        Self {
            c0: self.c0 * scalar,
            c1: self.c1 * scalar,
            c2: self.c2 * scalar,
        }
    }
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

impl Add for PackedFp3x2 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            c0: self.c0 + rhs.c0,
            c1: self.c1 + rhs.c1,
            c2: self.c2 + rhs.c2,
        }
    }
}

impl AddAssign for PackedFp3x2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for PackedFp3x2 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            c0: self.c0 - rhs.c0,
            c1: self.c1 - rhs.c1,
            c2: self.c2 - rhs.c2,
        }
    }
}

impl SubAssign for PackedFp3x2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for PackedFp3x2 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self {
            c0: -self.c0,
            c1: -self.c1,
            c2: -self.c2,
        }
    }
}

impl Mul for PackedFp3x2 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        // Karatsuba multiplication for Fp3:
        // (a0 + a1*w + a2*w^2) * (b0 + b1*w + b2*w^2) mod (w^3 - 2)
        //
        // v0 = a0*b0, v1 = a1*b1, v2 = a2*b2
        // c0 = v0 + 2 * ((a1 + a2)(b1 + b2) - v1 - v2)
        // c1 = (a0 + a1)(b0 + b1) - v0 - v1 + 2 * v2
        // c2 = (a0 + a2)(b0 + b2) - v0 + v1 - v2

        let v0 = self.c0 * rhs.c0;
        let v1 = self.c1 * rhs.c1;
        let v2 = self.c2 * rhs.c2;

        let a1_plus_a2 = self.c1 + self.c2;
        let b1_plus_b2 = rhs.c1 + rhs.c2;
        let a0_plus_a1 = self.c0 + self.c1;
        let b0_plus_b1 = rhs.c0 + rhs.c1;
        let a0_plus_a2 = self.c0 + self.c2;
        let b0_plus_b2 = rhs.c0 + rhs.c2;

        // c0 = v0 + 2 * ((a1 + a2)(b1 + b2) - v1 - v2)
        let cross_12 = a1_plus_a2 * b1_plus_b2;
        let cross_12_minus_v1_v2 = cross_12 - v1 - v2;
        let c0 = v0 + cross_12_minus_v1_v2.double();

        // c1 = (a0 + a1)(b0 + b1) - v0 - v1 + 2 * v2
        let cross_01 = a0_plus_a1 * b0_plus_b1;
        let c1 = cross_01 - v0 - v1 + v2.double();

        // c2 = (a0 + a2)(b0 + b2) - v0 + v1 - v2
        let cross_02 = a0_plus_a2 * b0_plus_b2;
        let c2 = cross_02 - v0 + v1 - v2;

        Self { c0, c1, c2 }
    }
}

impl MulAssign for PackedFp3x2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl Default for PackedFp3x2 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Debug for PackedFp3x2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elems = self.to_array();
        f.debug_struct("PackedFp3x2")
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

    fn fp3(c0: u64, c1: u64, c2: u64) -> Fp3Raw {
        Fp3Raw::new(c0, c1, c2)
    }

    #[test]
    fn test_from_to_array() {
        let a = fp3(111, 222, 333);
        let b = fp3(444, 555, 666);
        let packed = PackedFp3x2::from_array([a, b]);
        let unpacked = packed.to_array();
        assert_eq!(unpacked[0], a);
        assert_eq!(unpacked[1], b);
    }

    #[test]
    fn test_add() {
        let a0 = fp3(1, 2, 3);
        let a1 = fp3(4, 5, 6);
        let b0 = fp3(7, 8, 9);
        let b1 = fp3(10, 11, 12);

        let packed_a = PackedFp3x2::from_array([a0, a1]);
        let packed_b = PackedFp3x2::from_array([b0, b1]);
        let result = (packed_a + packed_b).to_array();

        assert_eq!(result[0], Fp3Raw::add(a0, b0));
        assert_eq!(result[1], Fp3Raw::add(a1, b1));
    }

    #[test]
    fn test_sub() {
        let a0 = fp3(10, 20, 30);
        let a1 = fp3(40, 50, 60);
        let b0 = fp3(1, 2, 3);
        let b1 = fp3(4, 5, 6);

        let packed_a = PackedFp3x2::from_array([a0, a1]);
        let packed_b = PackedFp3x2::from_array([b0, b1]);
        let result = (packed_a - packed_b).to_array();

        assert_eq!(result[0], Fp3Raw::sub(a0, b0));
        assert_eq!(result[1], Fp3Raw::sub(a1, b1));
    }

    #[test]
    fn test_neg() {
        let a0 = fp3(5, 10, 15);
        let a1 = fp3(20, 25, 30);

        let packed = PackedFp3x2::from_array([a0, a1]);
        let result = (-packed).to_array();

        assert_eq!(result[0], Fp3Raw::neg(a0));
        assert_eq!(result[1], Fp3Raw::neg(a1));
    }

    #[test]
    fn test_mul() {
        let a0 = fp3(2, 3, 4);
        let a1 = fp3(5, 6, 7);
        let b0 = fp3(8, 9, 10);
        let b1 = fp3(11, 12, 13);

        let packed_a = PackedFp3x2::from_array([a0, a1]);
        let packed_b = PackedFp3x2::from_array([b0, b1]);
        let result = (packed_a * packed_b).to_array();

        assert_eq!(result[0], Fp3Raw::mul(a0, b0));
        assert_eq!(result[1], Fp3Raw::mul(a1, b1));
    }

    #[test]
    fn test_mul_large() {
        let p = super::super::packed_goldilocks::P;
        let a0 = fp3(p - 1, p - 2, p - 3);
        let a1 = fp3(p - 4, p - 5, p - 6);
        let b0 = fp3(p - 7, p - 8, p - 9);
        let b1 = fp3(p - 10, p - 11, p - 12);

        let packed_a = PackedFp3x2::from_array([a0, a1]);
        let packed_b = PackedFp3x2::from_array([b0, b1]);
        let result = (packed_a * packed_b).to_array();

        assert_eq!(result[0], Fp3Raw::mul(a0, b0));
        assert_eq!(result[1], Fp3Raw::mul(a1, b1));
    }

    #[test]
    fn test_double() {
        let a0 = fp3(5, 10, 15);
        let a1 = fp3(20, 25, 30);

        let packed = PackedFp3x2::from_array([a0, a1]);
        let doubled = packed.double().to_array();
        let added = (packed + packed).to_array();

        assert_eq!(doubled[0], added[0]);
        assert_eq!(doubled[1], added[1]);
    }

    #[test]
    fn test_mul_by_fp() {
        let a0 = fp3(2, 3, 4);
        let a1 = fp3(5, 6, 7);
        let scalar0 = 10u64;
        let scalar1 = 20u64;

        let packed = PackedFp3x2::from_array([a0, a1]);
        let packed_scalar = PackedGoldilocks2::from_array([scalar0, scalar1]);
        let result = packed.mul_by_fp(packed_scalar).to_array();

        assert_eq!(result[0], Fp3Raw::mul_by_fp(a0, scalar0));
        assert_eq!(result[1], Fp3Raw::mul_by_fp(a1, scalar1));
    }

    #[test]
    fn test_identity_elements() {
        let a = PackedFp3x2::from_array([fp3(123, 456, 789), fp3(111, 222, 333)]);

        // Additive identity
        let sum = a + PackedFp3x2::ZERO;
        assert_eq!(sum, a);

        // Multiplicative identity
        let prod = a * PackedFp3x2::ONE;
        assert_eq!(prod.to_array()[0], a.to_array()[0]);
        assert_eq!(prod.to_array()[1], a.to_array()[1]);
    }

    #[test]
    fn test_algebraic_properties() {
        let a = PackedFp3x2::from_array([fp3(123, 456, 789), fp3(101, 202, 303)]);
        let b = PackedFp3x2::from_array([fp3(111, 222, 333), fp3(444, 555, 666)]);
        let c = PackedFp3x2::from_array([fp3(777, 888, 999), fp3(100, 200, 300)]);

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
        assert_eq!(zero[0], Fp3Raw::ZERO);
        assert_eq!(zero[1], Fp3Raw::ZERO);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    const P: u64 = super::super::packed_goldilocks::P;

    fn arb_fp3() -> impl Strategy<Value = Fp3Raw> {
        (0..P, 0..P, 0..P).prop_map(|(c0, c1, c2)| Fp3Raw::new(c0, c1, c2))
    }

    fn arb_packed_fp3() -> impl Strategy<Value = PackedFp3x2> {
        (arb_fp3(), arb_fp3()).prop_map(|(a, b)| PackedFp3x2::from_array([a, b]))
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn prop_add_matches_scalar(
            a0 in arb_fp3(),
            a1 in arb_fp3(),
            b0 in arb_fp3(),
            b1 in arb_fp3()
        ) {
            let packed_a = PackedFp3x2::from_array([a0, a1]);
            let packed_b = PackedFp3x2::from_array([b0, b1]);
            let result = (packed_a + packed_b).to_array();

            let expected0 = Fp3Raw::add(a0, b0);
            let expected1 = Fp3Raw::add(a1, b1);

            prop_assert_eq!(result[0], expected0);
            prop_assert_eq!(result[1], expected1);
        }

        #[test]
        fn prop_sub_matches_scalar(
            a0 in arb_fp3(),
            a1 in arb_fp3(),
            b0 in arb_fp3(),
            b1 in arb_fp3()
        ) {
            let packed_a = PackedFp3x2::from_array([a0, a1]);
            let packed_b = PackedFp3x2::from_array([b0, b1]);
            let result = (packed_a - packed_b).to_array();

            let expected0 = Fp3Raw::sub(a0, b0);
            let expected1 = Fp3Raw::sub(a1, b1);

            prop_assert_eq!(result[0], expected0);
            prop_assert_eq!(result[1], expected1);
        }

        #[test]
        fn prop_mul_matches_scalar(
            a0 in arb_fp3(),
            a1 in arb_fp3(),
            b0 in arb_fp3(),
            b1 in arb_fp3()
        ) {
            let packed_a = PackedFp3x2::from_array([a0, a1]);
            let packed_b = PackedFp3x2::from_array([b0, b1]);
            let result = (packed_a * packed_b).to_array();

            let expected0 = Fp3Raw::mul(a0, b0);
            let expected1 = Fp3Raw::mul(a1, b1);

            prop_assert_eq!(result[0], expected0, "mul lane 0 mismatch");
            prop_assert_eq!(result[1], expected1, "mul lane 1 mismatch");
        }

        #[test]
        fn prop_mul_by_fp_matches_scalar(
            a0 in arb_fp3(),
            a1 in arb_fp3(),
            s0 in 0..P,
            s1 in 0..P
        ) {
            let packed_a = PackedFp3x2::from_array([a0, a1]);
            let packed_s = PackedGoldilocks2::from_array([s0, s1]);
            let result = packed_a.mul_by_fp(packed_s).to_array();

            let expected0 = Fp3Raw::mul_by_fp(a0, s0);
            let expected1 = Fp3Raw::mul_by_fp(a1, s1);

            prop_assert_eq!(result[0], expected0);
            prop_assert_eq!(result[1], expected1);
        }

        #[test]
        fn prop_double_equals_add_self(a in arb_packed_fp3()) {
            let doubled = a.double().to_array();
            let added = (a + a).to_array();
            prop_assert_eq!(doubled[0], added[0]);
            prop_assert_eq!(doubled[1], added[1]);
        }

        #[test]
        fn prop_add_commutative(a in arb_packed_fp3(), b in arb_packed_fp3()) {
            prop_assert_eq!((a + b).to_array(), (b + a).to_array());
        }

        #[test]
        fn prop_mul_commutative(a in arb_packed_fp3(), b in arb_packed_fp3()) {
            prop_assert_eq!((a * b).to_array(), (b * a).to_array());
        }

        #[test]
        fn prop_distributive(a in arb_packed_fp3(), b in arb_packed_fp3(), c in arb_packed_fp3()) {
            prop_assert_eq!((a * (b + c)).to_array(), (a * b + a * c).to_array());
        }

        #[test]
        fn prop_additive_inverse(a in arb_packed_fp3()) {
            let neg_a = -a;
            let result = (a + neg_a).to_array();
            prop_assert_eq!(result[0], Fp3Raw::ZERO);
            prop_assert_eq!(result[1], Fp3Raw::ZERO);
        }
    }
}
