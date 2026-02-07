use crate::cyclic_group::IsGroup;
use crate::errors::ByteConversionError::{FromBEBytesError, FromLEBytesError};
use crate::errors::CreationError;
use crate::errors::DeserializationError;
use crate::field::element::FieldElement;
use crate::field::errors::FieldError;
use crate::field::traits::{HasDefaultTranscript, IsFFTField, IsField, IsPrimeField};
use crate::traits::{ByteConversion, Deserializable};

/// Type representing prime fields over unsigned 64-bit integers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct U64PrimeField<const MODULUS: u64>;
pub type U64FieldElement<const MODULUS: u64> = FieldElement<U64PrimeField<MODULUS>>;

pub type F17 = U64PrimeField<17>;
pub type FE17 = U64FieldElement<17>;

impl IsFFTField for F17 {
    const TWO_ADICITY: u64 = 4;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: u64 = 3;
}

impl<const MODULUS: u64> IsField for U64PrimeField<MODULUS> {
    type BaseType = u64;

    fn add(a: &u64, b: &u64) -> u64 {
        ((*a as u128 + *b as u128) % MODULUS as u128) as u64
    }

    fn sub(a: &u64, b: &u64) -> u64 {
        (((*a as u128 + MODULUS as u128) - *b as u128) % MODULUS as u128) as u64
    }

    fn neg(a: &u64) -> u64 {
        if *a == 0 {
            0
        } else {
            MODULUS - (a % MODULUS)
        }
    }

    fn mul(a: &u64, b: &u64) -> u64 {
        ((*a as u128 * *b as u128) % MODULUS as u128) as u64
    }

    fn div(a: &u64, b: &u64) -> Result<u64, FieldError> {
        let b_inv = &Self::inv(b)?;
        Ok(Self::mul(a, b_inv))
    }

    fn inv(a: &u64) -> Result<u64, FieldError> {
        if *a == 0 {
            return Err(FieldError::InvZeroError);
        }
        Ok(Self::pow(a, MODULUS - 2))
    }

    fn eq(a: &u64, b: &u64) -> bool {
        Self::from_u64(*a) == Self::from_u64(*b)
    }

    fn zero() -> u64 {
        0
    }

    fn one() -> u64 {
        1
    }

    fn from_u64(x: u64) -> u64 {
        x % MODULUS
    }

    fn from_base_type(x: u64) -> u64 {
        Self::from_u64(x)
    }
}

impl<const MODULUS: u64> Copy for U64FieldElement<MODULUS> {}

impl<const MODULUS: u64> IsPrimeField for U64PrimeField<MODULUS> {
    type CanonicalType = u64;

    fn canonical(x: &u64) -> u64 {
        *x
    }

    /// Returns how many bits do you need to represent the biggest field element
    /// It expects the MODULUS to be a Prime
    fn field_bit_size() -> usize {
        ((MODULUS - 1).ilog2() + 1) as usize
    }

    fn from_hex(hex_string: &str) -> Result<Self::BaseType, CreationError> {
        let hex_string = hex_string.strip_prefix("0x").unwrap_or(hex_string);
        u64::from_str_radix(hex_string, 16).map_err(|_| CreationError::InvalidHexString)
    }

    #[cfg(feature = "std")]
    fn to_hex(x: &u64) -> String {
        format!("{x:X}")
    }
}

/// Represents an element in Fp. (E.g: 0, 1, 2 are the elements of F3)
impl<const MODULUS: u64> IsGroup for U64FieldElement<MODULUS> {
    fn neutral_element() -> U64FieldElement<MODULUS> {
        U64FieldElement::zero()
    }

    fn operate_with(&self, other: &Self) -> Self {
        *self + *other
    }

    fn neg(&self) -> Self {
        -self
    }
}

impl<const MODULUS: u64> ByteConversion for U64FieldElement<MODULUS> {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        u64::to_be_bytes(*self.value()).into()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        u64::to_le_bytes(*self.value()).into()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let bytes: [u8; 8] = bytes[0..8].try_into().map_err(|_| FromBEBytesError)?;
        Ok(Self::from(u64::from_be_bytes(bytes)))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let bytes: [u8; 8] = bytes[0..8].try_into().map_err(|_| FromLEBytesError)?;
        Ok(Self::from(u64::from_le_bytes(bytes)))
    }
}

impl<const MODULUS: u64> Deserializable for FieldElement<U64PrimeField<MODULUS>> {
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        Self::from_bytes_be(bytes).map_err(|x| x.into())
    }
}

impl<const MODULUS: u64> HasDefaultTranscript for U64PrimeField<MODULUS> {
    fn get_random_field_element_from_rng(rng: &mut impl rand::Rng) -> FieldElement<Self> {
        let mask = u64::MAX >> MODULUS.leading_zeros();
        let mut sample = [0u8; 8];
        let field;
        loop {
            rng.fill(&mut sample);
            let int_sample = u64::from_be_bytes(sample) & mask;
            if int_sample < MODULUS {
                field = FieldElement::from(int_sample);
                break;
            }
        }
        field
    }
}
