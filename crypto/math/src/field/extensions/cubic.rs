use crate::field::element::FieldElement;
use crate::field::errors::FieldError;
use crate::field::traits::{IsField, IsSubFieldOf};
use crate::traits::ByteConversion;
use core::fmt::Debug;
use core::marker::PhantomData;

/// A general cubic extension field over `F`
/// with cubic non residue `Q::residue()`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CubicExtensionField<F, T> {
    field: PhantomData<F>,
    non_residue: PhantomData<T>,
}

pub type CubicExtensionFieldElement<F, T> = FieldElement<CubicExtensionField<F, T>>;

/// Trait to fix a cubic non residue.
/// Used to construct a cubic extension field by adding
/// a square root of `residue()`.
pub trait HasCubicNonResidue<F: IsField> {
    /// This function must return an element that is not a cube in Fp,
    /// that is, a cubic non-residue.
    fn residue() -> FieldElement<F>;
}

impl<F> ByteConversion for [FieldElement<F>; 3]
where
    F: IsField,
{
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_be(self[0].value());
        byte_slice.extend(ByteConversion::to_bytes_be(self[1].value()));
        byte_slice.extend(ByteConversion::to_bytes_be(self[2].value()));
        byte_slice
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_le(self[0].value());
        byte_slice.extend(ByteConversion::to_bytes_le(self[1].value()));
        byte_slice.extend(ByteConversion::to_bytes_le(self[2].value()));
        byte_slice
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        // Each field element takes 1/3 of the input bytes
        let elem_size = bytes.len() / 3;
        let v0 = F::BaseType::from_bytes_be(&bytes[0..elem_size])?;
        let v1 = F::BaseType::from_bytes_be(&bytes[elem_size..elem_size * 2])?;
        let v2 = F::BaseType::from_bytes_be(&bytes[elem_size * 2..elem_size * 3])?;
        Ok([
            FieldElement::from_raw(v0),
            FieldElement::from_raw(v1),
            FieldElement::from_raw(v2),
        ])
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        // Each field element takes 1/3 of the input bytes
        let elem_size = bytes.len() / 3;
        let v0 = F::BaseType::from_bytes_le(&bytes[0..elem_size])?;
        let v1 = F::BaseType::from_bytes_le(&bytes[elem_size..elem_size * 2])?;
        let v2 = F::BaseType::from_bytes_le(&bytes[elem_size * 2..elem_size * 3])?;
        Ok([
            FieldElement::from_raw(v0),
            FieldElement::from_raw(v1),
            FieldElement::from_raw(v2),
        ])
    }
}

impl<F, Q> IsField for CubicExtensionField<F, Q>
where
    F: IsField,
    Q: Clone + Debug + HasCubicNonResidue<F>,
{
    type BaseType = [FieldElement<F>; 3];

    /// Returns the component wise addition of `a` and `b`
    fn add(a: &[FieldElement<F>; 3], b: &[FieldElement<F>; 3]) -> [FieldElement<F>; 3] {
        [&a[0] + &b[0], &a[1] + &b[1], &a[2] + &b[2]]
    }

    /// Returns the multiplication of `a` and `b` using the following
    /// equation:
    /// (a0 + a1 * t) * (b0 + b1 * t) = a0 * b0 + a1 * b1 * Q::residue() + (a0 * b1 + a1 * b0) * t
    /// where `t.pow(2)` equals `Q::residue()`.
    fn mul(a: &[FieldElement<F>; 3], b: &[FieldElement<F>; 3]) -> [FieldElement<F>; 3] {
        let v0 = &a[0] * &b[0];
        let v1 = &a[1] * &b[1];
        let v2 = &a[2] * &b[2];

        [
            &v0 + Q::residue() * ((&a[1] + &a[2]) * (&b[1] + &b[2]) - &v1 - &v2),
            (&a[0] + &a[1]) * (&b[0] + &b[1]) - &v0 - &v1 + Q::residue() * &v2,
            (&a[0] + &a[2]) * (&b[0] + &b[2]) - v0 + v1 - v2,
        ]
    }

    /// Returns the component wise subtraction of `a` and `b`
    fn sub(a: &[FieldElement<F>; 3], b: &[FieldElement<F>; 3]) -> [FieldElement<F>; 3] {
        [&a[0] - &b[0], &a[1] - &b[1], &a[2] - &b[2]]
    }

    /// Returns the component wise negation of `a`
    fn neg(a: &[FieldElement<F>; 3]) -> [FieldElement<F>; 3] {
        [-&a[0], -&a[1], -&a[2]]
    }

    /// Returns the multiplicative inverse of `a`
    fn inv(a: &[FieldElement<F>; 3]) -> Result<[FieldElement<F>; 3], FieldError> {
        let three = FieldElement::<F>::from(3_u64);
        let d = a[0].pow(3_u64)
            + a[1].pow(3_u64) * Q::residue()
            + a[2].pow(3_u64) * Q::residue().pow(2_u64)
            - three * &a[0] * &a[1] * &a[2] * Q::residue();
        let inv = d.inv()?;
        Ok([
            (a[0].pow(2_u64) - &a[1] * &a[2] * Q::residue()) * &inv,
            (-&a[0] * &a[1] + a[2].pow(2_u64) * Q::residue()) * &inv,
            (-&a[0] * &a[2] + a[1].pow(2_u64)) * &inv,
        ])
    }

    /// Returns the division of `a` and `b`
    fn div(
        a: &[FieldElement<F>; 3],
        b: &[FieldElement<F>; 3],
    ) -> Result<[FieldElement<F>; 3], FieldError> {
        let b_inv = &Self::inv(b).map_err(|_| FieldError::DivisionByZero)?;
        Ok(<Self as IsField>::mul(a, b_inv))
    }

    /// Returns a boolean indicating whether `a` and `b` are equal component wise.
    fn eq(a: &[FieldElement<F>; 3], b: &[FieldElement<F>; 3]) -> bool {
        a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
    }

    /// Returns the additive neutral element of the field extension.
    fn zero() -> [FieldElement<F>; 3] {
        [
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    /// Returns the multiplicative neutral element of the field extension.
    fn one() -> [FieldElement<F>; 3] {
        [
            FieldElement::one(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    fn from_u64(x: u64) -> Self::BaseType {
        [
            FieldElement::from(x),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    /// Note: for this case this is simply the identity, because the components
    /// already have correct representations.
    fn from_base_type(x: [FieldElement<F>; 3]) -> [FieldElement<F>; 3] {
        x
    }
}

impl<F, Q> IsSubFieldOf<CubicExtensionField<F, Q>> for F
where
    F: IsField,
    Q: Clone + Debug + HasCubicNonResidue<F>,
{
    fn mul(
        a: &Self::BaseType,
        b: &<CubicExtensionField<F, Q> as IsField>::BaseType,
    ) -> <CubicExtensionField<F, Q> as IsField>::BaseType {
        let c0 = FieldElement::from_raw(F::mul(a, b[0].value()));
        let c1 = FieldElement::from_raw(F::mul(a, b[1].value()));
        let c2 = FieldElement::from_raw(F::mul(a, b[2].value()));
        [c0, c1, c2]
    }

    fn add(
        a: &Self::BaseType,
        b: &<CubicExtensionField<F, Q> as IsField>::BaseType,
    ) -> <CubicExtensionField<F, Q> as IsField>::BaseType {
        let c0 = FieldElement::from_raw(F::add(a, b[0].value()));
        [c0, b[1].clone(), b[2].clone()]
    }

    fn div(
        a: &Self::BaseType,
        b: &<CubicExtensionField<F, Q> as IsField>::BaseType,
    ) -> Result<<CubicExtensionField<F, Q> as IsField>::BaseType, FieldError> {
        let b_inv = <CubicExtensionField<F, Q> as IsField>::inv(b)?;
        Ok(<Self as IsSubFieldOf<CubicExtensionField<F, Q>>>::mul(
            a, &b_inv,
        ))
    }

    fn sub(
        a: &Self::BaseType,
        b: &<CubicExtensionField<F, Q> as IsField>::BaseType,
    ) -> <CubicExtensionField<F, Q> as IsField>::BaseType {
        let c0 = FieldElement::from_raw(F::sub(a, b[0].value()));
        let c1 = FieldElement::from_raw(F::neg(b[1].value()));
        let c2 = FieldElement::from_raw(F::neg(b[2].value()));
        [c0, c1, c2]
    }

    fn embed(a: Self::BaseType) -> <CubicExtensionField<F, Q> as IsField>::BaseType {
        [
            FieldElement::from_raw(a),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <CubicExtensionField<F, Q> as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}
