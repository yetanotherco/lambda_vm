//! Quadratic and cubic extensions of the Goldilocks field.
//!
//! These extensions use the optimized Goldilocks implementation (no Montgomery form).
//!

use crate::field::{
    element::FieldElement,
    errors::FieldError,
    goldilocks::{GOLDILOCKS_PRIME, GoldilocksField, dot_product_2, dot_product_3, mul_by_7_raw},
    traits::{HasDefaultTranscript, IsField, IsSubFieldOf},
};
use crate::traits::{AsBytes, ByteConversion};

impl ByteConversion for [FpE; 2] {
    const BYTE_LEN: usize = 16;

    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        unimplemented!()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        unimplemented!()
    }

    fn from_bytes_be(_bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        unimplemented!()
    }

    fn from_bytes_le(_bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}

impl ByteConversion for [FpE; 3] {
    const BYTE_LEN: usize = 24;

    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        let mut bytes = ByteConversion::to_bytes_be(&self[2]);
        bytes.extend(ByteConversion::to_bytes_be(&self[1]));
        bytes.extend(ByteConversion::to_bytes_be(&self[0]));
        bytes
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        let mut bytes = ByteConversion::to_bytes_le(&self[0]);
        bytes.extend(ByteConversion::to_bytes_le(&self[1]));
        bytes.extend(ByteConversion::to_bytes_le(&self[2]));
        bytes
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const N: usize = 8;
        if bytes.len() < N * 3 {
            return Err(crate::errors::ByteConversionError::FromBEBytesError);
        }
        let x2 = FieldElement::from_bytes_be(&bytes[0..N])?;
        let x1 = FieldElement::from_bytes_be(&bytes[N..N * 2])?;
        let x0 = FieldElement::from_bytes_be(&bytes[N * 2..N * 3])?;
        Ok([x0, x1, x2])
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const N: usize = 8;
        if bytes.len() < N * 3 {
            return Err(crate::errors::ByteConversionError::FromLEBytesError);
        }
        let x0 = FieldElement::from_bytes_le(&bytes[0..N])?;
        let x1 = FieldElement::from_bytes_le(&bytes[N..N * 2])?;
        let x2 = FieldElement::from_bytes_le(&bytes[N * 2..N * 3])?;
        Ok([x0, x1, x2])
    }
}

// =====================================================
// QUADRATIC EXTENSION (Fp2)
// =====================================================
// The quadratic extension is constructed using x^2 - 7,
// where 7 is a quadratic non-residue in the Goldilocks field.
// This means Fp2 = Fp[x] / (x^2 - 7)
// Elements are represented as a0 + a1*w where w^2 = 7

pub(crate) type FpE = FieldElement<GoldilocksField>;

/// Degree 2 extension field of Goldilocks
#[derive(Copy, Clone, Debug)]
pub struct Degree2GoldilocksExtensionField;

impl IsField for Degree2GoldilocksExtensionField {
    type BaseType = [FpE; 2];

    /// Returns the component-wise addition of `a` and `b`
    #[inline(always)]
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1]]
    }

    /// Multiplication using fused dot products for fewer reductions.
    /// (a0 + a1*w) * (b0 + b1*w) = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*w
    ///
    /// Uses dot_product_2 to compute each output component with a single
    /// reduce128 instead of separate mul + reduce per product.
    #[inline(always)]
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let (a0, a1) = (*a[0].value(), *a[1].value());
        let (b0, b1) = (*b[0].value(), *b[1].value());
        let b1_7 = mul_by_7_raw(b1);

        // c0 = a0*b0 + a1*(7*b1)
        let c0 = dot_product_2(a0, b0, a1, b1_7);
        // c1 = a0*b1 + a1*b0
        let c1 = dot_product_2(a0, b1, a1, b0);

        [FpE::from_raw(c0), FpE::from_raw(c1)]
    }

    /// Squaring using fused dot product for the first component.
    /// (a0 + a1*w)^2 = (a0^2 + 7*a1^2) + 2*a0*a1*w
    #[inline(always)]
    fn square(a: &Self::BaseType) -> Self::BaseType {
        let (a0, a1) = (*a[0].value(), *a[1].value());
        let a1_7 = mul_by_7_raw(a1);

        // c0 = a0*a0 + a1*(7*a1) via single-reduction dot product
        let c0 = dot_product_2(a0, a0, a1, a1_7);
        // c1 = 2 * a0 * a1
        let c1 = <GoldilocksField as IsField>::mul(&a0, &a1);
        let c1 = GoldilocksField::double(&c1);

        [FpE::from_raw(c0), FpE::from_raw(c1)]
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

impl IsSubFieldOf<Degree2GoldilocksExtensionField> for GoldilocksField {
    // The base×ext ops run in the constraint-eval hot loop from downstream
    // crates; these impls are concrete (non-generic), so without the
    // attribute they compile as cross-crate calls under the default
    // no-LTO release profile — unlike the #[inline(always)] IsField ops.
    #[inline(always)]
    fn mul(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree2GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::mul(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::mul(a, b[1].value()));
        [c0, c1]
    }

    #[inline(always)]
    fn add(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree2GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::add(a, b[0].value()));
        [c0, b[1]]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksExtensionField as IsField>::BaseType,
    ) -> Result<<Degree2GoldilocksExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Degree2GoldilocksExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<Degree2GoldilocksExtensionField>>::mul(a, &b_inv))
    }

    #[inline(always)]
    fn sub(
        a: &Self::BaseType,
        b: &<Degree2GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree2GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::sub(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::neg(b[1].value()));
        [c0, c1]
    }

    #[inline(always)]
    fn embed(a: Self::BaseType) -> <Degree2GoldilocksExtensionField as IsField>::BaseType {
        [FpE::from_raw(a), FpE::zero()]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <Degree2GoldilocksExtensionField as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

/// Field element type for the quadratic extension of native Goldilocks
pub type Fp2E = FieldElement<Degree2GoldilocksExtensionField>;

impl Fp2E {
    /// Returns the conjugate of self: conjugate(a0 + a1*w) = a0 - a1*w
    pub fn conjugate(&self) -> Self {
        Self::new([self.value()[0], -self.value()[1]])
    }

    /// Create a field element from an i64.
    /// Negative values are converted to their field equivalents: -x becomes p - x.
    pub fn from_i64(value: i64) -> Self {
        Self::from(value)
    }
}

// =====================================================
// CUBIC EXTENSION (Fp3)
// =====================================================
// The cubic extension is constructed using x^3 - 2,
// where 2 is a cubic non-residue in the Goldilocks field.
// This means Fp3 = Fp[x] / (x^3 - 2)
// Elements are represented as a0 + a1*w + a2*w^2 where w^3 = 2

/// Degree 3 extension field of Goldilocks
#[derive(Copy, Clone, Debug)]
pub struct Degree3GoldilocksExtensionField;

impl IsField for Degree3GoldilocksExtensionField {
    type BaseType = [FpE; 3];

    /// Returns the component-wise addition of `a` and `b`
    #[inline(always)]
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }

    /// Multiplication using schoolbook with fused dot products.
    /// (a0 + a1*w + a2*w^2) * (b0 + b1*w + b2*w^2) mod (w^3 - 2)
    ///
    /// Expanding and applying w^3 = 2:
    ///   c0 = a0*b0 + 2*(a1*b2 + a2*b1)
    ///   c1 = a0*b1 + a1*b0 + 2*a2*b2
    ///   c2 = a0*b2 + a1*b1 + a2*b0
    ///
    /// Each component is computed as a single dot_product_3 (9 raw muls,
    /// 3 reduce128 calls) instead of Karatsuba (6 muls, 6 reduce128 + many
    /// add/sub). The reduction savings outweigh the extra multiplications.
    #[inline(always)]
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let (a0, a1, a2) = (*a[0].value(), *a[1].value(), *a[2].value());
        let (b0, b1, b2) = (*b[0].value(), *b[1].value(), *b[2].value());

        // Precompute 2*b1 and 2*b2 for the w^3 = 2 reduction
        let b1_2 = GoldilocksField::double(&b1);
        let b2_2 = GoldilocksField::double(&b2);

        // c0 = a0*b0 + a1*(2*b2) + a2*(2*b1)
        let c0 = dot_product_3(a0, b0, a1, b2_2, a2, b1_2);
        // c1 = a0*b1 + a1*b0 + a2*(2*b2)
        let c1 = dot_product_3(a0, b1, a1, b0, a2, b2_2);
        // c2 = a0*b2 + a1*b1 + a2*b0
        let c2 = dot_product_3(a0, b2, a1, b1, a2, b0);

        [FpE::from_raw(c0), FpE::from_raw(c1), FpE::from_raw(c2)]
    }

    /// Squaring using fused dot products.
    /// (a0 + a1*w + a2*w^2)^2 mod (w^3 - 2):
    ///   c0 = a0^2 + 4*a1*a2
    ///   c1 = 2*a0*a1 + 2*a2^2
    ///   c2 = 2*a0*a2 + a1^2
    #[inline(always)]
    fn square(a: &Self::BaseType) -> Self::BaseType {
        let (a0, a1, a2) = (*a[0].value(), *a[1].value(), *a[2].value());

        let a0_2 = GoldilocksField::double(&a0);
        let a2_4 = GoldilocksField::double(&GoldilocksField::double(&a2));

        // c0 = a0*a0 + a1*(4*a2) — using dot_product_2
        let c0 = dot_product_2(a0, a0, a1, a2_4);
        // c1 = (2*a0)*a1 + (2*a2)*a2 — using dot_product_2
        let a2_2 = GoldilocksField::double(&a2);
        let c1 = dot_product_2(a0_2, a1, a2_2, a2);
        // c2 = a1*a1 + (2*a0)*a2 — using dot_product_2
        let c2 = dot_product_2(a1, a1, a0_2, a2);

        [FpE::from_raw(c0), FpE::from_raw(c1), FpE::from_raw(c2)]
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

impl IsSubFieldOf<Degree3GoldilocksExtensionField> for GoldilocksField {
    // The base×ext ops run in the constraint-eval hot loop from downstream
    // crates (the evaluator's eval·β fold and every LogUp fingerprint term);
    // these impls are concrete (non-generic), so without the attribute they
    // compile as cross-crate calls under the default no-LTO release profile —
    // unlike the #[inline(always)] IsField ops.
    #[inline(always)]
    fn mul(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree3GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::mul(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::mul(a, b[1].value()));
        let c2 = FpE::from_raw(<Self as IsField>::mul(a, b[2].value()));
        [c0, c1, c2]
    }

    #[inline(always)]
    fn add(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree3GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::add(a, b[0].value()));
        [c0, b[1], b[2]]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) -> Result<<Degree3GoldilocksExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Degree3GoldilocksExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<Degree3GoldilocksExtensionField>>::mul(a, &b_inv))
    }

    #[inline(always)]
    fn sub(
        a: &Self::BaseType,
        b: &<Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) -> <Degree3GoldilocksExtensionField as IsField>::BaseType {
        let c0 = FpE::from_raw(<Self as IsField>::sub(a, b[0].value()));
        let c1 = FpE::from_raw(<Self as IsField>::neg(b[1].value()));
        let c2 = FpE::from_raw(<Self as IsField>::neg(b[2].value()));
        [c0, c1, c2]
    }

    #[inline(always)]
    fn embed(a: Self::BaseType) -> <Degree3GoldilocksExtensionField as IsField>::BaseType {
        [FpE::from_raw(a), FpE::zero(), FpE::zero()]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <Degree3GoldilocksExtensionField as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

/// Field element type for the cubic extension of native Goldilocks
pub type Fp3E = FieldElement<Degree3GoldilocksExtensionField>;

impl Fp3E {
    /// Create a field element from an i64.
    /// Negative values are converted to their field equivalents: -x becomes p - x.
    pub fn from_i64(value: i64) -> Self {
        Self::from(value)
    }
}

// =====================================================
// TRAIT IMPLEMENTATIONS FOR PROVER/VERIFIER
// =====================================================

impl ByteConversion for FieldElement<Degree3GoldilocksExtensionField> {
    const BYTE_LEN: usize = 24;

    #[inline(always)]
    fn write_bytes_be(&self, buf: &mut [u8]) {
        debug_assert!(buf.len() >= 24);
        let components = self.value();
        components[0].write_bytes_be(&mut buf[0..8]);
        components[1].write_bytes_be(&mut buf[8..16]);
        components[2].write_bytes_be(&mut buf[16..24]);
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_be(&self.value()[0]);
        byte_slice.extend(ByteConversion::to_bytes_be(&self.value()[1]));
        byte_slice.extend(ByteConversion::to_bytes_be(&self.value()[2]));
        byte_slice
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_le(&self.value()[0]);
        byte_slice.extend(ByteConversion::to_bytes_le(&self.value()[1]));
        byte_slice.extend(ByteConversion::to_bytes_le(&self.value()[2]));
        byte_slice
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const BYTES_PER_FIELD: usize = 8;
        if bytes.len() < BYTES_PER_FIELD * 3 {
            return Err(crate::errors::ByteConversionError::FromBEBytesError);
        }
        let x0 = FieldElement::from_bytes_be(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        let x2 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD * 2..BYTES_PER_FIELD * 3])?;

        Ok(Self::new([x0, x1, x2]))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const BYTES_PER_FIELD: usize = 8;
        if bytes.len() < BYTES_PER_FIELD * 3 {
            return Err(crate::errors::ByteConversionError::FromLEBytesError);
        }
        let x0 = FieldElement::from_bytes_le(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        let x2 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD * 2..BYTES_PER_FIELD * 3])?;

        Ok(Self::new([x0, x1, x2]))
    }
}

#[cfg(feature = "alloc")]
impl AsBytes for FieldElement<Degree3GoldilocksExtensionField> {
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.to_bytes_be()
    }

    #[inline(always)]
    fn stream_bytes(&self, sink: &mut dyn FnMut(&[u8])) {
        let mut buf = [0u8; 24];
        crate::traits::ByteConversion::write_bytes_be(self, &mut buf);
        sink(&buf);
    }
}

impl HasDefaultTranscript for Degree3GoldilocksExtensionField {
    fn get_random_field_element_from_rng(rng: &mut impl rand::Rng) -> FieldElement<Self> {
        let mut sample = [0u8; 8];
        let mut coeffs = [FpE::zero(), FpE::zero(), FpE::zero()];

        for coeff in &mut coeffs {
            loop {
                rng.fill(&mut sample);
                let int_sample = u64::from_be_bytes(sample);
                if int_sample < GOLDILOCKS_PRIME {
                    *coeff = FpE::from(int_sample);
                    break;
                }
            }
        }

        FieldElement::<Self>::new(coeffs)
    }
}

// =====================================================
// HELPER FUNCTIONS
// =====================================================

/// Multiply a field element by 7 (the quadratic non-residue).
/// Wraps the raw u64 implementation for use with FieldElement types.
#[inline(always)]
fn mul_by_7(a: &FpE) -> FpE {
    FpE::from_raw(mul_by_7_raw(*a.value()))
}
