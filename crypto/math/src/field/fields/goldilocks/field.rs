use core::fmt::{self, Display};

use crate::errors::ByteConversionError;
use crate::traits::ByteConversion;
use crate::{
    errors::CreationError,
    field::{
        element::FieldElement,
        errors::FieldError,
        traits::{IsFFTField, IsField, IsPrimeField},
    },
};

/// Goldilocks Prime Field F_p where p = 2^64 - 2^32 + 1
///
/// This is an FFT-friendly field with two-adicity of 32.
/// The implementation uses optimized native u64 arithmetic.
///
/// NOTE: This implementation was inspired by and borrows from the work done by the Plonky3 team
/// https://github.com/Plonky3/Plonky3/blob/main/goldilocks/src/lib.rs
#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Goldilocks64Field;

impl Goldilocks64Field {
    /// The field order: p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
    pub const ORDER: u64 = 0xFFFF_FFFF_0000_0001;

    /// Two's complement of `ORDER` i.e. `2^64 - ORDER = 2^32 - 1`
    pub const NEG_ORDER: u64 = Self::ORDER.wrapping_neg();
}

impl IsField for Goldilocks64Field {
    type BaseType = u64;

    fn add(a: &u64, b: &u64) -> u64 {
        let (sum, over) = a.overflowing_add(*b);
        let (mut sum, over) = sum.overflowing_add(u64::from(over) * Self::NEG_ORDER);
        if over {
            sum += Self::NEG_ORDER
        }
        Self::representative(&sum)
    }

    fn mul(a: &u64, b: &u64) -> u64 {
        Self::representative(&reduce_128(u128::from(*a) * u128::from(*b)))
    }

    fn sub(a: &u64, b: &u64) -> u64 {
        let (diff, under) = a.overflowing_sub(*b);
        let (mut diff, under) = diff.overflowing_sub(u64::from(under) * Self::NEG_ORDER);
        if under {
            diff -= Self::NEG_ORDER;
        }
        Self::representative(&diff)
    }

    fn neg(a: &u64) -> u64 {
        Self::sub(&Self::ORDER, &Self::representative(a))
    }

    /// Returns the multiplicative inverse of `a` using addition chain.
    fn inv(a: &u64) -> Result<u64, FieldError> {
        if *a == Self::zero() || *a == Self::ORDER {
            return Err(FieldError::InvZeroError);
        }

        // a^11
        let t2 = Self::mul(&Self::square(a), a);

        // a^111
        let t3 = Self::mul(&Self::square(&t2), a);

        // compute base^111111 (6 ones)
        let t6 = exp_acc::<3>(&t3, &t3);
        let t60 = Self::square(&t6);
        let t7 = Self::mul(&t60, a);

        // compute base^111111111111 (12 ones)
        let t12 = exp_acc::<5>(&t60, &t6);

        // compute base^111111111111111111111111 (24 ones)
        let t24 = exp_acc::<12>(&t12, &t12);

        // compute base^1111111111111111111111111111111 (31 ones)
        let t31 = exp_acc::<7>(&t24, &t7);

        // compute base^111111111111111111111111111111101111111111111111111111111111111
        let t63 = exp_acc::<32>(&t31, &t31);

        Ok(Self::mul(&Self::square(&t63), a))
    }

    fn div(a: &u64, b: &u64) -> Result<u64, FieldError> {
        let b_inv = &Self::inv(b)?;
        Ok(Self::mul(a, b_inv))
    }

    fn eq(a: &u64, b: &u64) -> bool {
        Self::representative(a) == Self::representative(b)
    }

    fn zero() -> u64 {
        0u64
    }

    fn one() -> u64 {
        1u64
    }

    fn from_u64(x: u64) -> u64 {
        Self::representative(&x)
    }

    fn from_base_type(x: u64) -> u64 {
        Self::representative(&x)
    }
}

impl IsPrimeField for Goldilocks64Field {
    type RepresentativeType = u64;

    fn representative(x: &u64) -> u64 {
        let mut u = *x;
        if u >= Self::ORDER {
            u -= Self::ORDER;
        }
        u
    }

    fn field_bit_size() -> usize {
        64
    }

    fn from_hex(hex_string: &str) -> Result<Self::BaseType, CreationError> {
        let mut hex_string = hex_string;
        // Remove 0x if it's on the string
        let mut char_iterator = hex_string.chars();
        if hex_string.len() > 2
            && char_iterator.next().unwrap() == '0'
            && char_iterator.next().unwrap() == 'x'
        {
            hex_string = &hex_string[2..];
        }
        u64::from_str_radix(hex_string, 16).map_err(|_| CreationError::InvalidHexString)
    }

    #[cfg(feature = "std")]
    fn to_hex(x: &u64) -> String {
        format!("{x:X}")
    }
}

/// IsFFTField implementation for Goldilocks64Field
///
/// The field order is p = 2^64 - 2^32 + 1
/// p - 1 = 2^64 - 2^32 = 2^32 * (2^32 - 1)
/// So the two-adicity is 32.
///
/// The primitive 2^32-th root of unity was taken from Plonky3:
/// https://github.com/Plonky3/Plonky3/blob/main/goldilocks/src/lib.rs
impl IsFFTField for Goldilocks64Field {
    const TWO_ADICITY: u64 = 32;

    /// 2^32-th primitive root of unity: 1753635133440165772
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: u64 = 1753635133440165772;

    fn field_name() -> &'static str {
        "goldilocks64"
    }
}

/// Reduces a 128-bit product to a 64-bit field element
#[inline(always)]
pub fn reduce_128(x: u128) -> u64 {
    let (x_lo, x_hi) = (x as u64, (x >> 64) as u64);
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & Goldilocks64Field::NEG_ORDER;

    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    if borrow {
        t0 -= Goldilocks64Field::NEG_ORDER // Cannot underflow
    }

    let t1 = x_hi_lo * Goldilocks64Field::NEG_ORDER;
    let (res_wrapped, carry) = t0.overflowing_add(t1);
    res_wrapped + Goldilocks64Field::NEG_ORDER * u64::from(carry)
}

#[inline(always)]
fn exp_acc<const N: usize>(base: &u64, tail: &u64) -> u64 {
    Goldilocks64Field::mul(&exp_power_of_2::<N>(base), tail)
}

#[must_use]
fn exp_power_of_2<const POWER_LOG: usize>(base: &u64) -> u64 {
    let mut res = *base;
    for _ in 0..POWER_LOG {
        res = Goldilocks64Field::square(&res);
    }
    res
}

impl ByteConversion for u64 {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized,
    {
        if bytes.len() < 8 {
            return Err(ByteConversionError::FromBEBytesError);
        }
        Ok(u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized,
    {
        if bytes.len() < 8 {
            return Err(ByteConversionError::FromLEBytesError);
        }
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }
}

impl FieldElement<Goldilocks64Field> {
    pub fn to_bytes_le_array(&self) -> [u8; 8] {
        self.representative().to_le_bytes()
    }

    pub fn to_bytes_be_array(&self) -> [u8; 8] {
        self.representative().to_be_bytes()
    }
}

impl ByteConversion for FieldElement<Goldilocks64Field> {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        self.representative().to_be_bytes().to_vec()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        self.representative().to_le_bytes().to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized,
    {
        if bytes.len() < 8 {
            return Err(ByteConversionError::FromBEBytesError);
        }
        let value = u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        Ok(Self::new(Goldilocks64Field::from_base_type(value)))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized,
    {
        if bytes.len() < 8 {
            return Err(ByteConversionError::FromLEBytesError);
        }
        let value = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        Ok(Self::new(Goldilocks64Field::from_base_type(value)))
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for FieldElement<Goldilocks64Field> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.representative().partial_cmp(&other.representative())
    }
}

impl Ord for FieldElement<Goldilocks64Field> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.representative().cmp(&other.representative())
    }
}

impl Display for FieldElement<Goldilocks64Field> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:x}", self.representative())?;
        Ok(())
    }
}
