//! Packed cubic extension field Fp3 = Fp[w] / (w^3 - 2).
//!
//! Holds WIDTH independent Fp3 elements across 3 packed base field values.

use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::extensions_goldilocks::Degree3GoldilocksExtensionField;
use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
use crate::field::packed::PackedField;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Packed cubic extension: 3 packed base field values = WIDTH independent Fp3 elements.
#[derive(Copy, Clone, Debug)]
pub struct PackedFp3<P: PackedField<Scalar = GoldilocksField>> {
    pub c0: P,
    pub c1: P,
    pub c2: P,
}

impl<P: PackedField<Scalar = GoldilocksField>> PackedFp3<P> {
    pub fn zero() -> Self {
        Self {
            c0: P::zero(),
            c1: P::zero(),
            c2: P::zero(),
        }
    }

    pub fn one() -> Self {
        Self {
            c0: P::ones(),
            c1: P::zero(),
            c2: P::zero(),
        }
    }

    /// Multiply by a packed base field scalar: (s * c0, s * c1, s * c2)
    /// This is the F x E -> E multiplication critical for constraint evaluation.
    #[inline(always)]
    pub fn mul_scalar(self, s: P) -> Self {
        Self {
            c0: self.c0 * s,
            c1: self.c1 * s,
            c2: self.c2 * s,
        }
    }

    /// Construct from a closure that returns (c0, c1, c2) per lane.
    pub fn from_fn(mut f: impl FnMut(usize) -> [FieldElement<GoldilocksField>; 3]) -> Self {
        // Call f once per lane, collecting into arrays
        assert!(P::WIDTH <= 16, "PackedFp3::from_fn supports WIDTH up to 16");
        let mut c0s = [FieldElement::zero(); 16];
        let mut c1s = [FieldElement::zero(); 16];
        let mut c2s = [FieldElement::zero(); 16];
        for i in 0..P::WIDTH {
            let [a, b, c] = f(i);
            c0s[i] = a;
            c1s[i] = b;
            c2s[i] = c;
        }
        Self {
            c0: P::from_fn(|i| c0s[i]),
            c1: P::from_fn(|i| c1s[i]),
            c2: P::from_fn(|i| c2s[i]),
        }
    }

    /// Broadcast a scalar Fp3 to all WIDTH lanes.
    #[inline(always)]
    pub fn broadcast(val: &FieldElement<Degree3GoldilocksExtensionField>) -> Self {
        let components = val.value();
        Self {
            c0: P::broadcast(components[0]),
            c1: P::broadcast(components[1]),
            c2: P::broadcast(components[2]),
        }
    }

    /// Set a single lane from a scalar Fp3.
    #[inline(always)]
    pub fn set_lane(&mut self, lane: usize, val: &FieldElement<Degree3GoldilocksExtensionField>) {
        let components = val.value();
        self.c0.as_slice_mut()[lane] = components[0];
        self.c1.as_slice_mut()[lane] = components[1];
        self.c2.as_slice_mut()[lane] = components[2];
    }

    /// Extract a single lane as a scalar Fp3.
    #[inline(always)]
    pub fn get_lane(&self, lane: usize) -> FieldElement<Degree3GoldilocksExtensionField> {
        FieldElement::from_raw([
            self.c0.as_slice()[lane],
            self.c1.as_slice()[lane],
            self.c2.as_slice()[lane],
        ])
    }

    /// Optimized squaring: 3 squares + 3 cross-products (no Karatsuba overhead).
    /// From extensions_goldilocks.rs:223-238.
    #[inline(always)]
    pub fn square(&self) -> Self {
        let s0 = self.c0 * self.c0;
        let s1 = self.c1 * self.c1;
        let s2 = self.c2 * self.c2;
        let a01 = self.c0 * self.c1;
        let a02 = self.c0 * self.c2;
        let a12 = self.c1 * self.c2;

        // c0 = s0 + 4 * a12
        // c1 = 2 * a01 + 2 * s2
        // c2 = 2 * a02 + s1
        Self {
            c0: s0 + a12.double().double(),
            c1: a01.double() + s2.double(),
            c2: a02.double() + s1,
        }
    }
}

impl<P: PackedField<Scalar = GoldilocksField>> Add for PackedFp3<P> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 + rhs.c0,
            c1: self.c1 + rhs.c1,
            c2: self.c2 + rhs.c2,
        }
    }
}

impl<P: PackedField<Scalar = GoldilocksField>> Sub for PackedFp3<P> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 - rhs.c0,
            c1: self.c1 - rhs.c1,
            c2: self.c2 - rhs.c2,
        }
    }
}

impl<P: PackedField<Scalar = GoldilocksField>> Neg for PackedFp3<P> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            c0: -self.c0,
            c1: -self.c1,
            c2: -self.c2,
        }
    }
}

impl<P: PackedField<Scalar = GoldilocksField>> Mul for PackedFp3<P> {
    type Output = Self;
    /// Karatsuba-like multiplication modulo w^3 = 2.
    /// Same formula as extensions_goldilocks.rs:206-218.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let v0 = self.c0 * rhs.c0;
        let v1 = self.c1 * rhs.c1;
        let v2 = self.c2 * rhs.c2;

        let t0 = (self.c1 + self.c2) * (rhs.c1 + rhs.c2) - v1 - v2;
        let t1 = (self.c0 + self.c1) * (rhs.c0 + rhs.c1) - v0 - v1;
        let t2 = (self.c0 + self.c2) * (rhs.c0 + rhs.c2) - v0 - v2;

        // residue = 2, so multiply by 2 = double
        Self {
            c0: v0 + t0.double(), // v0 + 2*(cross terms)
            c1: t1 + v2.double(), // cross + 2*v2
            c2: t2 + v1,          // cross + v1
        }
    }
}

impl<P: PackedField<Scalar = GoldilocksField>> AddAssign for PackedFp3<P> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<P: PackedField<Scalar = GoldilocksField>> SubAssign for PackedFp3<P> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<P: PackedField<Scalar = GoldilocksField>> MulAssign for PackedFp3<P> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fields::fft_friendly::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use crate::field::fields::fft_friendly::u64_goldilocks_packed::PackedGoldilocks;
    use crate::field::traits::IsField;

    type FE = FieldElement<GoldilocksField>;
    type Fp3E = FieldElement<Degree3GoldilocksExtensionField>;

    fn random_packed_fp3() -> PackedFp3<PackedGoldilocks> {
        PackedFp3 {
            c0: PackedGoldilocks::from_fn(|i| FE::from((i as u64 + 1) * 111)),
            c1: PackedGoldilocks::from_fn(|i| FE::from((i as u64 + 1) * 222)),
            c2: PackedGoldilocks::from_fn(|i| FE::from((i as u64 + 1) * 333)),
        }
    }

    fn another_packed_fp3() -> PackedFp3<PackedGoldilocks> {
        PackedFp3 {
            c0: PackedGoldilocks::from_fn(|i| FE::from((i as u64 + 3) * 444)),
            c1: PackedGoldilocks::from_fn(|i| FE::from((i as u64 + 3) * 555)),
            c2: PackedGoldilocks::from_fn(|i| FE::from((i as u64 + 3) * 666)),
        }
    }

    fn to_scalar_fp3(p: &PackedFp3<PackedGoldilocks>, lane: usize) -> [FE; 3] {
        [
            p.c0.as_slice()[lane],
            p.c1.as_slice()[lane],
            p.c2.as_slice()[lane],
        ]
    }

    #[test]
    fn test_packed_fp3_add_matches_scalar() {
        let a = random_packed_fp3();
        let b = another_packed_fp3();
        let sum = a + b;
        for i in 0..PackedGoldilocks::WIDTH {
            let a_s = to_scalar_fp3(&a, i);
            let b_s = to_scalar_fp3(&b, i);
            let expected = Degree3GoldilocksExtensionField::add(&a_s, &b_s);
            assert_eq!(to_scalar_fp3(&sum, i), expected, "lane {i}");
        }
    }

    #[test]
    fn test_packed_fp3_sub_matches_scalar() {
        let a = random_packed_fp3();
        let b = another_packed_fp3();
        let diff = a - b;
        for i in 0..PackedGoldilocks::WIDTH {
            let a_s = to_scalar_fp3(&a, i);
            let b_s = to_scalar_fp3(&b, i);
            let expected = Degree3GoldilocksExtensionField::sub(&a_s, &b_s);
            assert_eq!(to_scalar_fp3(&diff, i), expected, "lane {i}");
        }
    }

    #[test]
    fn test_packed_fp3_mul_matches_scalar() {
        let a = random_packed_fp3();
        let b = another_packed_fp3();
        let prod = a * b;
        for i in 0..PackedGoldilocks::WIDTH {
            let a_s = to_scalar_fp3(&a, i);
            let b_s = to_scalar_fp3(&b, i);
            let expected = Degree3GoldilocksExtensionField::mul(&a_s, &b_s);
            assert_eq!(to_scalar_fp3(&prod, i), expected, "lane {i}");
        }
    }

    #[test]
    fn test_packed_fp3_square_matches_scalar() {
        let a = random_packed_fp3();
        let sq = a.square();
        for i in 0..PackedGoldilocks::WIDTH {
            let a_s = to_scalar_fp3(&a, i);
            let expected = Degree3GoldilocksExtensionField::square(&a_s);
            assert_eq!(to_scalar_fp3(&sq, i), expected, "lane {i}");
        }
    }

    #[test]
    fn test_packed_fp3_scalar_mul() {
        let scalar = PackedGoldilocks::from_fn(|i| FE::from((i as u64 + 1) * 42));
        let ext = random_packed_fp3();
        let result = ext.mul_scalar(scalar);
        for i in 0..PackedGoldilocks::WIDTH {
            let s = scalar.as_slice()[i];
            assert_eq!(
                result.c0.as_slice()[i],
                s * ext.c0.as_slice()[i],
                "c0 lane {i}"
            );
            assert_eq!(
                result.c1.as_slice()[i],
                s * ext.c1.as_slice()[i],
                "c1 lane {i}"
            );
            assert_eq!(
                result.c2.as_slice()[i],
                s * ext.c2.as_slice()[i],
                "c2 lane {i}"
            );
        }
    }

    #[test]
    fn test_packed_fp3_mul_identity() {
        let a = random_packed_fp3();
        let one = PackedFp3::one();
        let result = a * one;
        for i in 0..PackedGoldilocks::WIDTH {
            assert_eq!(to_scalar_fp3(&result, i), to_scalar_fp3(&a, i), "lane {i}");
        }
    }

    #[test]
    fn test_packed_fp3_broadcast() {
        let val = Fp3E::from_raw([FE::from(42u64), FE::from(99u64), FE::from(7u64)]);
        let packed = PackedFp3::<PackedGoldilocks>::broadcast(&val);
        for i in 0..PackedGoldilocks::WIDTH {
            assert_eq!(packed.get_lane(i), val, "lane {i}");
        }
    }

    #[test]
    fn test_packed_fp3_set_get_lane() {
        let mut packed = PackedFp3::<PackedGoldilocks>::zero();
        for i in 0..PackedGoldilocks::WIDTH {
            let val = Fp3E::from_raw([
                FE::from((i as u64 + 1) * 11),
                FE::from((i as u64 + 1) * 22),
                FE::from((i as u64 + 1) * 33),
            ]);
            packed.set_lane(i, &val);
        }
        for i in 0..PackedGoldilocks::WIDTH {
            let expected = Fp3E::from_raw([
                FE::from((i as u64 + 1) * 11),
                FE::from((i as u64 + 1) * 22),
                FE::from((i as u64 + 1) * 33),
            ]);
            assert_eq!(packed.get_lane(i), expected, "lane {i}");
        }
    }

    #[test]
    fn test_packed_fp3_distributivity() {
        let a = random_packed_fp3();
        let b = another_packed_fp3();
        let c = PackedFp3 {
            c0: PackedGoldilocks::from_fn(|i| FE::from((i as u64 + 7) * 777)),
            c1: PackedGoldilocks::from_fn(|i| FE::from((i as u64 + 7) * 888)),
            c2: PackedGoldilocks::from_fn(|i| FE::from((i as u64 + 7) * 999)),
        };
        let lhs = a * (b + c);
        let rhs = a * b + a * c;
        for i in 0..PackedGoldilocks::WIDTH {
            assert_eq!(to_scalar_fp3(&lhs, i), to_scalar_fp3(&rhs, i), "lane {i}");
        }
    }
}
