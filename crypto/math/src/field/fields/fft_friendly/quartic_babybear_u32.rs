use crate::field::{
    element::FieldElement,
    errors::FieldError,
    fields::fft_friendly::babybear_u32::Babybear31PrimeField,
    traits::{HasDefaultTranscript, IsFFTField, IsField, IsSubFieldOf},
};

use crate::traits::ByteConversion;

#[cfg(feature = "alloc")]
use crate::traits::AsBytes;

/// We are implementig the extension of Baby Bear of degree 4 using the irreducible polynomial x^4 + 11.
/// BETA = 11 and -BETA = -11 is the non-residue.
/// Since `const_from_raw()` doesn't make the montgomery conversion, we calculated it.
/// The montgomery form of a number "a" is a * R mod p.
/// In Baby Bear field, R = 2^32 and p = 2013265921.
/// Then, 939524073 = 11 * 2^32 mod 2013265921.
pub const BETA: FieldElement<Babybear31PrimeField> =
    FieldElement::<Babybear31PrimeField>::const_from_raw(939524073);

#[derive(Copy, Clone, Debug)]
pub struct Degree4BabyBearU32ExtensionField;

/// We implement directly the degree four extension for performance reasons, instead of using
/// the default quadratic extension provided by the library
impl IsField for Degree4BabyBearU32ExtensionField {
    type BaseType = [FieldElement<Babybear31PrimeField>; 4];

    /// Addition of degree four field extension elements
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
    }

    /// Result of multiplying two polynomials a = a0 + a1 * x + a2 * x^2 + a3 * x^3 and
    /// b = b0 + b1 * x + b2 * x^2 + b3 * x^3 by applying distribution and taking
    /// the remainder of the division by x^4 + 11.
    /// Multiplication of two degree four field extension elements
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [
            a[0] * b[0] - BETA * (a[1] * b[3] + a[3] * b[1] + a[2] * b[2]),
            a[0] * b[1] + a[1] * b[0] - BETA * (a[2] * b[3] + a[3] * b[2]),
            a[0] * b[2] + a[2] * b[0] + a[1] * b[1] - BETA * (a[3] * b[3]),
            a[0] * b[3] + a[3] * b[0] + a[1] * b[2] + a[2] * b[1],
        ]
    }

    /// Returns the square of a degree four field extension element
    /// More efficient to use instead of multiplying the element with itself
    fn square(a: &Self::BaseType) -> Self::BaseType {
        [
            a[0].square() - BETA * ((a[1] * a[3]).double() + a[2].square()),
            (a[0] * a[1] - BETA * (a[2] * a[3])).double(),
            (a[0] * a[2]).double() + a[1].square() - BETA * (a[3].square()),
            (a[0] * a[3] + a[1] * a[2]).double(),
        ]
    }

    /// Subtraction of degree four field extension elements
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
    }

    /// Additive inverse of degree four field extension element
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        [-&a[0], -&a[1], -&a[2], -&a[3]]
    }

    /// Return te inverse of a fp4 element if exist.
    /// This algorithm is inspired by Risc0 implementation:
    /// <https://github.com/risc0/risc0/blob/4c41c739779ef2759a01ebcf808faf0fbffe8793/risc0/core/src/field/baby_bear.rs#L460>
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let mut b0 = a[0] * a[0] + BETA * (a[1] * (a[3] + a[3]) - a[2] * a[2]);
        let mut b2 = a[0] * (a[2] + a[2]) - a[1] * a[1] + BETA * (a[3] * a[3]);
        let c = b0.square() + BETA * b2.square();
        let c_inv = c.inv()?;
        b0 *= &c_inv;
        b2 *= &c_inv;
        Ok([
            a[0] * b0 + BETA * a[2] * b2,
            -a[1] * b0 - BETA * a[3] * b2,
            -a[0] * b2 + a[2] * b0,
            a[1] * b2 - a[3] * b0,
        ])
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = &Self::inv(b).map_err(|_| FieldError::DivisionByZero)?;
        Ok(<Self as IsField>::mul(a, b_inv))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]
    }

    fn zero() -> Self::BaseType {
        Self::BaseType::default()
    }

    fn one() -> Self::BaseType {
        [
            FieldElement::one(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    fn from_u64(x: u64) -> Self::BaseType {
        [
            FieldElement::from(x),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    /// It takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    /// Note: for this case this is simply the identity, because the components
    /// already have correct representations.
    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }

    fn double(a: &Self::BaseType) -> Self::BaseType {
        <Degree4BabyBearU32ExtensionField as IsField>::add(a, a)
    }

    fn pow<T>(a: &Self::BaseType, mut exponent: T) -> Self::BaseType
    where
        T: crate::unsigned_integer::traits::IsUnsignedInteger,
    {
        let zero = T::from(0);
        let one = T::from(1);

        if exponent == zero {
            return Self::one();
        }
        if exponent == one {
            return *a;
        }

        let mut result = *a;

        // Fast path for powers of 2
        while exponent & one == zero {
            result = Self::square(&result);
            exponent >>= 1;
            if exponent == zero {
                return result;
            }
        }

        let mut base = result;
        exponent >>= 1;

        while exponent != zero {
            base = Self::square(&base);
            if exponent & one == one {
                result = <Degree4BabyBearU32ExtensionField as IsField>::mul(&result, &base);
            }
            exponent >>= 1;
        }

        result
    }
}

/// Implements efficient operations between a BabyBear element and a degree four extension element
impl IsSubFieldOf<Degree4BabyBearU32ExtensionField> for Babybear31PrimeField {
    fn mul(
        a: &Self::BaseType,
        b: &<Degree4BabyBearU32ExtensionField as IsField>::BaseType,
    ) -> <Degree4BabyBearU32ExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::mul(a, b[0].value()));
        let c1 = FieldElement::from_raw(<Self as IsField>::mul(a, b[1].value()));
        let c2 = FieldElement::from_raw(<Self as IsField>::mul(a, b[2].value()));
        let c3 = FieldElement::from_raw(<Self as IsField>::mul(a, b[3].value()));

        [c0, c1, c2, c3]
    }

    fn add(
        a: &Self::BaseType,
        b: &<Degree4BabyBearU32ExtensionField as IsField>::BaseType,
    ) -> <Degree4BabyBearU32ExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::add(a, b[0].value()));
        let c1 = FieldElement::from_raw(*b[1].value());
        let c2 = FieldElement::from_raw(*b[2].value());
        let c3 = FieldElement::from_raw(*b[3].value());

        [c0, c1, c2, c3]
    }

    fn div(
        a: &Self::BaseType,
        b: &<Degree4BabyBearU32ExtensionField as IsField>::BaseType,
    ) -> Result<<Degree4BabyBearU32ExtensionField as IsField>::BaseType, FieldError> {
        let b_inv = Degree4BabyBearU32ExtensionField::inv(b)?;
        Ok(<Self as IsSubFieldOf<Degree4BabyBearU32ExtensionField>>::mul(a, &b_inv))
    }

    fn sub(
        a: &Self::BaseType,
        b: &<Degree4BabyBearU32ExtensionField as IsField>::BaseType,
    ) -> <Degree4BabyBearU32ExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::sub(a, b[0].value()));
        let c1 = FieldElement::from_raw(<Self as IsField>::neg(b[1].value()));
        let c2 = FieldElement::from_raw(<Self as IsField>::neg(b[2].value()));
        let c3 = FieldElement::from_raw(<Self as IsField>::neg(b[3].value()));
        [c0, c1, c2, c3]
    }

    fn embed(a: Self::BaseType) -> <Degree4BabyBearU32ExtensionField as IsField>::BaseType {
        [
            FieldElement::from_raw(a),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ]
    }

    #[cfg(feature = "alloc")]
    fn to_subfield_vec(
        b: <Degree4BabyBearU32ExtensionField as IsField>::BaseType,
    ) -> alloc::vec::Vec<Self::BaseType> {
        b.into_iter().map(|x| x.to_raw()).collect()
    }
}

impl ByteConversion for [FieldElement<Babybear31PrimeField>; 4] {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_be(&self[0]);
        byte_slice.extend(ByteConversion::to_bytes_be(&self[1]));
        byte_slice.extend(ByteConversion::to_bytes_be(&self[2]));
        byte_slice.extend(ByteConversion::to_bytes_be(&self[3]));
        byte_slice
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_le(&self[0]);
        byte_slice.extend(ByteConversion::to_bytes_le(&self[1]));
        byte_slice.extend(ByteConversion::to_bytes_le(&self[2]));
        byte_slice.extend(ByteConversion::to_bytes_le(&self[3]));
        byte_slice
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const BYTES_PER_FIELD: usize = 32;

        let x0 = FieldElement::from_bytes_be(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        let x2 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD * 2..BYTES_PER_FIELD * 3])?;
        let x3 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD * 3..BYTES_PER_FIELD * 4])?;

        Ok([x0, x1, x2, x3])
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const BYTES_PER_FIELD: usize = 32;

        let x0 = FieldElement::from_bytes_le(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        let x2 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD * 2..BYTES_PER_FIELD * 3])?;
        let x3 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD * 3..BYTES_PER_FIELD * 4])?;

        Ok([x0, x1, x2, x3])
    }
}

impl ByteConversion for FieldElement<Degree4BabyBearU32ExtensionField> {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_be(&self.value()[0]);
        byte_slice.extend(ByteConversion::to_bytes_be(&self.value()[1]));
        byte_slice.extend(ByteConversion::to_bytes_be(&self.value()[2]));
        byte_slice.extend(ByteConversion::to_bytes_be(&self.value()[3]));
        byte_slice
    }
    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        let mut byte_slice = ByteConversion::to_bytes_le(&self.value()[0]);
        byte_slice.extend(ByteConversion::to_bytes_le(&self.value()[1]));
        byte_slice.extend(ByteConversion::to_bytes_le(&self.value()[2]));
        byte_slice.extend(ByteConversion::to_bytes_le(&self.value()[3]));
        byte_slice
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const BYTES_PER_FIELD: usize = 4;
        let x0 = FieldElement::from_bytes_be(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        let x2 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD * 2..BYTES_PER_FIELD * 3])?;
        let x3 = FieldElement::from_bytes_be(&bytes[BYTES_PER_FIELD * 3..BYTES_PER_FIELD * 4])?;

        Ok(Self::new([x0, x1, x2, x3]))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        const BYTES_PER_FIELD: usize = 4;
        let x0 = FieldElement::from_bytes_le(&bytes[0..BYTES_PER_FIELD])?;
        let x1 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD..BYTES_PER_FIELD * 2])?;
        let x2 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD * 2..BYTES_PER_FIELD * 3])?;
        let x3 = FieldElement::from_bytes_le(&bytes[BYTES_PER_FIELD * 3..BYTES_PER_FIELD * 4])?;

        Ok(Self::new([x0, x1, x2, x3]))
    }
}

#[cfg(feature = "alloc")]
impl AsBytes for FieldElement<Degree4BabyBearU32ExtensionField> {
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.to_bytes_be()
    }
}

impl IsFFTField for Degree4BabyBearU32ExtensionField {
    const TWO_ADICITY: u64 = 29;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: Self::BaseType = [
        FieldElement::const_from_raw(0),
        FieldElement::const_from_raw(0),
        FieldElement::const_from_raw(0),
        // We are using the montgomery form of 124907976.
        // That is: 1344142388 = 124907976 * R mod p,
        // where R = 2^32 and p = 2013265921.
        FieldElement::const_from_raw(1344142388),
    ];
}

impl HasDefaultTranscript for Degree4BabyBearU32ExtensionField {
    fn get_random_field_element_from_rng(rng: &mut impl rand::Rng) -> FieldElement<Self> {
        //Babybear Prime p = 2^31 - 2^27 + 1
        const MODULUS: u32 = 2013265921;

        //Babybear prime needs 31 bits and is represented with 32 bits.
        //The mask is used to remove the first bit.
        const MASK: u32 = 0x7FFF_FFFF;

        let mut sample = [0u8; 4];

        let mut coeffs = [
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ];

        for coeff in &mut coeffs {
            loop {
                rng.fill(&mut sample);
                let int_sample = u32::from_be_bytes(sample) & MASK;
                if int_sample < MODULUS {
                    *coeff = FieldElement::from(&int_sample);
                    break;
                }
            }
        }

        FieldElement::<Self>::new(coeffs)
    }
}
