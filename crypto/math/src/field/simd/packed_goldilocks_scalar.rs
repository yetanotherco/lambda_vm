//! Scalar fallback implementation of PackedGoldilocks2.
//!
//! This module provides a non-SIMD implementation that mimics the SIMD API
//! for use on architectures without ARM NEON support.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::field::element::FieldElement;
use crate::field::fields::u64_goldilocks_field::Goldilocks64Field;
use crate::field::traits::IsField;

/// The Goldilocks prime: p = 2^64 - 2^32 + 1
pub const P: u64 = 0xFFFF_FFFF_0000_0001;

/// EPSILON = 2^32 - 1
pub const EPSILON: u64 = 0xFFFF_FFFF;

/// A "packed" vector of 2 Goldilocks field elements (scalar fallback).
///
/// This implementation processes elements sequentially rather than in parallel,
/// but maintains API compatibility with the SIMD version.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PackedGoldilocks2([u64; 2]);

impl PackedGoldilocks2 {
    /// Number of field elements packed in this type.
    pub const WIDTH: usize = 2;

    /// Zero element (additive identity).
    pub const ZERO: Self = Self([0, 0]);

    /// One element (multiplicative identity).
    pub const ONE: Self = Self([1, 1]);

    /// Creates a new packed element with all lanes set to the same value.
    #[inline(always)]
    pub fn broadcast(val: u64) -> Self {
        debug_assert!(val < P, "value must be in canonical form");
        Self([val, val])
    }

    /// Creates a new packed element from an array of field elements.
    #[inline(always)]
    pub fn from_array(vals: [u64; 2]) -> Self {
        debug_assert!(vals[0] < P && vals[1] < P, "values must be in canonical form");
        Self(vals)
    }

    /// Creates a new packed element from FieldElement array.
    #[inline(always)]
    pub fn from_field_elements(vals: [FieldElement<Goldilocks64Field>; 2]) -> Self {
        Self::from_array([*vals[0].value(), *vals[1].value()])
    }

    /// Extracts the elements as an array.
    #[inline(always)]
    pub fn to_array(self) -> [u64; 2] {
        self.0
    }

    /// Extracts the elements as FieldElements.
    #[inline(always)]
    pub fn to_field_elements(self) -> [FieldElement<Goldilocks64Field>; 2] {
        [
            FieldElement::<Goldilocks64Field>::from_raw(self.0[0]),
            FieldElement::<Goldilocks64Field>::from_raw(self.0[1]),
        ]
    }

    /// Gets the element at the specified index.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> u64 {
        debug_assert!(idx < 2);
        self.0[idx]
    }

    /// Doubles the element.
    #[inline(always)]
    pub fn double(self) -> Self {
        self + self
    }

    /// Squares the element.
    #[inline(always)]
    pub fn square(self) -> Self {
        self * self
    }
}

impl Add for PackedGoldilocks2 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self([
            Goldilocks64Field::add(&self.0[0], &rhs.0[0]),
            Goldilocks64Field::add(&self.0[1], &rhs.0[1]),
        ])
    }
}

impl AddAssign for PackedGoldilocks2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for PackedGoldilocks2 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self([
            Goldilocks64Field::sub(&self.0[0], &rhs.0[0]),
            Goldilocks64Field::sub(&self.0[1], &rhs.0[1]),
        ])
    }
}

impl SubAssign for PackedGoldilocks2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for PackedGoldilocks2 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self([
            Goldilocks64Field::neg(&self.0[0]),
            Goldilocks64Field::neg(&self.0[1]),
        ])
    }
}

impl Mul for PackedGoldilocks2 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self([
            Goldilocks64Field::mul(&self.0[0], &rhs.0[0]),
            Goldilocks64Field::mul(&self.0[1], &rhs.0[1]),
        ])
    }
}

impl MulAssign for PackedGoldilocks2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Default for PackedGoldilocks2 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Debug for PackedGoldilocks2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PackedGoldilocks2")
            .field(&self.0[0])
            .field(&self.0[1])
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        let a = PackedGoldilocks2::from_array([1, 2]);
        let b = PackedGoldilocks2::from_array([3, 4]);

        let sum = a + b;
        assert_eq!(sum.to_array(), [4, 6]);

        let diff = b - a;
        assert_eq!(diff.to_array(), [2, 2]);

        let prod = a * b;
        assert_eq!(prod.to_array(), [3, 8]);
    }
}
