//! Quadratic and cubic extensions of the native Goldilocks field.
//!
//! These extensions use the optimized native Goldilocks implementation
//! instead of Montgomery form for better performance.
//!

use crate::field::{
    element::FieldElement,
    errors::FieldError,
    fields::fft_friendly::u64_goldilocks_native::GoldilocksField,
    traits::{IsField, IsSubFieldOf},
};

// =====================================================
// QUADRATIC EXTENSION (Fp2)
// =====================================================
// The quadratic extension is constructed using x^2 - 7,
// where 7 is a quadratic non-residue in the Goldilocks field.
// This means Fp2 = Fp[x] / (x^2 - 7)
// Elements are represented as a0 + a1*w where w^2 = 7

type FpE = FieldElement<GoldilocksField>;

/// Degree 2 extension field of native Goldilocks
#[derive(Copy, Clone, Debug)]
pub struct Degree2GoldilocksNativeExtensionField;

impl IsField for Degree2GoldilocksNativeExtensionField {
    type BaseType = [FpE; 2];

    /// Returns the component-wise addition of `a` and `b`
    #[inline(always)]
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1]]
    }

    /// Returns the multiplication of `a` and `b`:
    /// (a0 + a1*w) * (b0 + b1*w) = (a0*b0 + W*a1*b1) + (a0*b1 + a1*b0)*w
    /// where w^2 = W = 7
    #[inline(always)]
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let a0b0 = a[0] * b[0];
        let a1b1 = a[1] * b[1];
        let z = (a[0] + a[1]) * (b[0] + b[1]);

        // W * a1b1 = 7 * a1b1
        let w_a1b1 = mul_by_7(&a1b1);

        [a0b0 + w_a1b1, z - a0b0 - a1b1]
    }

    /// Returns the square of `a`:
    /// (a0 + a1*w)^2 = (a0^2 + W*a1^2) + 2*a0*a1*w
    #[inline(always)]
    fn square(a: &Self::BaseType) -> Self::BaseType {
        let a0_sq = a[0].square();
        let a1_sq = a[1].square();
        let a0a1 = a[0] * a[1];

        // W * a1^2 = 7 * a1^2
        let w_a1_sq = mul_by_7(&a1_sq);

        [a0_sq + w_a1_sq, a0a1.double()]
    }

    /// Returns the component-wise subtraction of `a` and `b`
    #[inline(always)]
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] - b[0], a[1] - b[1]]
    }

    /// Returns the component-wise negation of `a`
    #[inline(always)]
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-&a[0], -&a[1]]
    }

    /// Returns the multiplicative inverse of `a`:
    /// (a0 + a1*w)^-1 = (a0 - a1*w) / (a0^2 - W*a1^2)
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let a0_sq = a[0].square();
        let a1_sq = a[1].square();
        let w_a1_sq = mul_by_7(&a1_sq);
        let norm = a0_sq - w_a1_sq;
        let norm_inv = norm.inv()?;
        Ok([a[0] * norm_inv, -a[1] * norm_inv])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = Self::inv(b)?;
        Ok(<Self as IsField>::mul(a, &b_inv))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1]
    }

    fn zero() -> Self::BaseType {
        [FpE::zero(), FpE::zero()]
    }

    fn one() -> Self::BaseType {
        [FpE::one(), FpE::zero()]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [FpE::from(x), FpE::zero()]
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }

    fn double(a: &Self::BaseType) -> Self::BaseType {
        [a[0].double(), a[1].double()]
    }
}

impl IsSubFieldOf<Degree2GoldilocksNativeExtensionField> for GoldilocksField {
    fn mul(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksNativeExtensionField as IsField>::BaseType,
    ) -> <Degree2GoldilocksNativeExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::mul(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::mul(a, b[1].value()));
        [c0, c1]
    }

    fn add(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksNativeExtensionField as IsField>::BaseType,
    ) -> <Degree2GoldilocksNativeExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::add(a, b[0].value()));
        [c0, b[1]]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksNativeExtensionField as IsField>::BaseType,
    ) -> Result<<Degree2GoldilocksNativeExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Degree2GoldilocksNativeExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<
            Degree2GoldilocksNativeExtensionField,
        >>::mul(a, &b_inv))
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksNativeExtensionField as IsField>::BaseType,
    ) -> <Degree2GoldilocksNativeExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::sub(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::neg(b[1].value()));
        [c0, c1]
    }

    fn embed(a: Self::BaseType) -> <Degree2GoldilocksNativeExtensionField as IsField>::BaseType {
        [FpE::from_raw(a), FpE::zero()]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <Degree2GoldilocksNativeExtensionField as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

/// Field element type for the quadratic extension of native Goldilocks
pub type Fp2E = FieldElement<Degree2GoldilocksNativeExtensionField>;

impl Fp2E {
    /// Returns the conjugate of self: conjugate(a0 + a1*w) = a0 - a1*w
    pub fn conjugate(&self) -> Self {
        Self::new([self.value()[0], -self.value()[1]])
    }
}

// =====================================================
// CUBIC EXTENSION (Fp3)
// =====================================================
// The cubic extension is constructed using x^3 - 2,
// where 2 is a cubic non-residue in the Goldilocks field.
// This means Fp3 = Fp[x] / (x^3 - 2)
// Elements are represented as a0 + a1*w + a2*w^2 where w^3 = 2

/// Degree 3 extension field of native Goldilocks
#[derive(Copy, Clone, Debug)]
pub struct Degree3GoldilocksNativeExtensionField;

impl IsField for Degree3GoldilocksNativeExtensionField {
    type BaseType = [FpE; 3];

    /// Returns the component-wise addition of `a` and `b`
    #[inline(always)]
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }

    /// Returns the multiplication of `a` and `b`:
    /// (a0 + a1*w + a2*w^2) * (b0 + b1*w + b2*w^2) mod (w^3 - 2)
    #[inline(always)]
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let v0 = a[0] * b[0];
        let v1 = a[1] * b[1];
        let v2 = a[2] * b[2];

        // c0 = v0 + 2 * ((a1 + a2)(b1 + b2) - v1 - v2)
        // c1 = (a0 + a1)(b0 + b1) - v0 - v1 + 2 * v2
        // c2 = (a0 + a2)(b0 + b2) - v0 + v1 - v2
        let t0 = (a[1] + a[2]) * (b[1] + b[2]) - v1 - v2;
        let t1 = (a[0] + a[1]) * (b[0] + b[1]) - v0 - v1;
        let t2 = (a[0] + a[2]) * (b[0] + b[2]) - v0 - v2;

        [v0 + t0.double(), t1 + v2.double(), t2 + v1]
    }

    /// Returns the square of `a`
    #[inline(always)]
    fn square(a: &Self::BaseType) -> Self::BaseType {
        let s0 = a[0].square();
        let s1 = a[1].square();
        let s2 = a[2].square();
        let a01 = a[0] * a[1];
        let a02 = a[0] * a[2];
        let a12 = a[1] * a[2];

        // c0 = s0 + 4 * a12
        // c1 = 2 * a01 + 2 * s2
        // c2 = 2 * a02 + s1
        [
            s0 + a12.double().double(),
            a01.double() + s2.double(),
            a02.double() + s1,
        ]
    }

    /// Returns the component-wise subtraction of `a` and `b`
    #[inline(always)]
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    /// Returns the component-wise negation of `a`
    #[inline(always)]
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-a[0], -a[1], -a[2]]
    }

    /// Returns the multiplicative inverse of `a`
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let a0_sq = a[0].square();
        let a1_sq = a[1].square();
        let a2_sq = a[2].square();

        // Compute the norm: N = a0^3 + 2*a1^3 + 4*a2^3 - 6*a0*a1*a2
        let a0_cubed = a0_sq * a[0];
        let a1_cubed = a1_sq * a[1];
        let a2_cubed = a2_sq * a[2];
        let a0a1a2 = a[0] * a[1] * a[2];

        // N = a0^3 + 2*a1^3 + 4*a2^3 - 6*a0*a1*a2
        let norm = a0_cubed + a1_cubed.double() + a2_cubed.double().double()
            - (a0a1a2.double() + a0a1a2).double();

        let norm_inv = norm.inv()?;

        // inv[0] = (a0^2 - 2*a1*a2) / N
        // inv[1] = (2*a2^2 - a0*a1) / N
        // inv[2] = (a1^2 - a0*a2) / N
        let a1a2 = a[1] * a[2];
        let a0a1 = a[0] * a[1];
        let a0a2 = a[0] * a[2];

        Ok([
            (a0_sq - a1a2.double()) * norm_inv,
            (a2_sq.double() - a0a1) * norm_inv,
            (a1_sq - a0a2) * norm_inv,
        ])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = Self::inv(b)?;
        Ok(<Self as IsField>::mul(a, &b_inv))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
    }

    fn zero() -> Self::BaseType {
        [FpE::zero(), FpE::zero(), FpE::zero()]
    }

    fn one() -> Self::BaseType {
        [FpE::one(), FpE::zero(), FpE::zero()]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [FpE::from(x), FpE::zero(), FpE::zero()]
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }

    fn double(a: &Self::BaseType) -> Self::BaseType {
        [a[0].double(), a[1].double(), a[2].double()]
    }
}

impl IsSubFieldOf<Degree3GoldilocksNativeExtensionField> for GoldilocksField {
    fn mul(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksNativeExtensionField as IsField>::BaseType,
    ) -> <Degree3GoldilocksNativeExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::mul(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::mul(a, b[1].value()));
        let c2 = FpE::from_raw(<Self as IsField>::mul(a, b[2].value()));
        [c0, c1, c2]
    }

    fn add(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksNativeExtensionField as IsField>::BaseType,
    ) -> <Degree3GoldilocksNativeExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::add(a, b[0].value()));
        [c0, b[1], b[2]]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksNativeExtensionField as IsField>::BaseType,
    ) -> Result<<Degree3GoldilocksNativeExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Degree3GoldilocksNativeExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<
            Degree3GoldilocksNativeExtensionField,
        >>::mul(a, &b_inv))
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksNativeExtensionField as IsField>::BaseType,
    ) -> <Degree3GoldilocksNativeExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::sub(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::neg(b[1].value()));
        let c2 = FpE::from_raw(<Self as IsField>::neg(b[2].value()));
        [c0, c1, c2]
    }

    fn embed(a: Self::BaseType) -> <Degree3GoldilocksNativeExtensionField as IsField>::BaseType {
        [FpE::from_raw(a), FpE::zero(), FpE::zero()]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <Degree3GoldilocksNativeExtensionField as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

/// Field element type for the cubic extension of native Goldilocks
pub type Fp3E = FieldElement<Degree3GoldilocksNativeExtensionField>;

// =====================================================
// HELPER FUNCTIONS
// =====================================================

/// Multiply a field element by 7 (the quadratic non-residue).
/// Uses 7 = 1 + 2 + 4 for efficiency (2 doubles + 2 adds, saves one double vs 8-1).
#[inline(always)]
fn mul_by_7(a: &FpE) -> FpE {
    // 7 * a = a + 2a + 4a
    let a2 = a.double();
    let a4 = a2.double();
    *a + a2 + a4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp2_add() {
        let a = Fp2E::new([FpE::from(3u64), FpE::from(4u64)]);
        let b = Fp2E::new([FpE::from(1u64), FpE::from(2u64)]);
        let c = &a + &b;
        assert_eq!(c.value()[0], FpE::from(4u64));
        assert_eq!(c.value()[1], FpE::from(6u64));
    }

    #[test]
    fn test_fp2_mul() {
        let a = Fp2E::new([FpE::from(3u64), FpE::from(4u64)]);
        let b = Fp2E::new([FpE::from(1u64), FpE::from(2u64)]);
        // (3 + 4w)(1 + 2w) = 3 + 6w + 4w + 8w^2 = 3 + 10w + 8*7 = 59 + 10w
        let c = &a * &b;
        assert_eq!(c.value()[0], FpE::from(59u64));
        assert_eq!(c.value()[1], FpE::from(10u64));
    }

    #[test]
    fn test_fp2_inv() {
        let a = Fp2E::new([FpE::from(3u64), FpE::from(4u64)]);
        let a_inv = a.inv().unwrap();
        let product = &a * &a_inv;
        assert_eq!(product, Fp2E::one());
    }

    #[test]
    fn test_fp3_add() {
        let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
        let b = Fp3E::new([FpE::from(4u64), FpE::from(5u64), FpE::from(6u64)]);
        let c = &a + &b;
        assert_eq!(c.value()[0], FpE::from(5u64));
        assert_eq!(c.value()[1], FpE::from(7u64));
        assert_eq!(c.value()[2], FpE::from(9u64));
    }

    #[test]
    fn test_fp3_mul() {
        let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
        let b = Fp3E::new([FpE::from(4u64), FpE::from(5u64), FpE::from(6u64)]);
        let c = &a * &b;
        // Verify by computing inverse
        let c_div_a = &c * &a.inv().unwrap();
        assert_eq!(c_div_a, b);
    }

    #[test]
    fn test_fp3_inv() {
        let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
        let a_inv = a.inv().unwrap();
        let product = &a * &a_inv;
        assert_eq!(product, Fp3E::one());
    }

    #[test]
    fn test_mul_by_7() {
        let a = FpE::from(5u64);
        let result = mul_by_7(&a);
        assert_eq!(result, FpE::from(35u64));
    }
}
