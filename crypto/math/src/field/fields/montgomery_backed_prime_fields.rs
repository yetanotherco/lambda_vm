use crate::errors::CreationError;
use crate::field::element::FieldElement;
use crate::field::errors::FieldError;
use crate::field::traits::{HasDefaultTranscript, IsPrimeField};
#[cfg(feature = "alloc")]
use crate::traits::AsBytes;
use crate::traits::ByteConversion;
use crate::{
    field::traits::IsField, unsigned_integer::element::UnsignedInteger,
    unsigned_integer::montgomery::MontgomeryAlgorithms,
};

use core::fmt::Debug;
use core::marker::PhantomData;

pub type U384PrimeField<M> = MontgomeryBackendPrimeField<M, 6>;
pub type U256PrimeField<M> = MontgomeryBackendPrimeField<M, 4>;
pub type U64PrimeField<M> = MontgomeryBackendPrimeField<M, 1>;

/// This trait is necessary for us to be able to use unsigned integer types bigger than
/// `u128` (the biggest native `unit`) as constant generics.
/// This trait should be removed when Rust supports this feature.
pub trait IsModulus<U>: Debug {
    const MODULUS: U;
}

#[cfg_attr(
    any(
        feature = "lambdaworks-serde-binary",
        feature = "lambdaworks-serde-string"
    ),
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Clone, Debug, Hash, Copy)]
pub struct MontgomeryBackendPrimeField<M, const NUM_LIMBS: usize> {
    phantom: PhantomData<M>,
}

impl<M, const NUM_LIMBS: usize> MontgomeryBackendPrimeField<M, NUM_LIMBS>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>>,
{
    pub const R2: UnsignedInteger<NUM_LIMBS> = Self::compute_r2_parameter(&M::MODULUS);
    pub const MU: u64 = Self::compute_mu_parameter(&M::MODULUS);
    pub const ZERO: UnsignedInteger<NUM_LIMBS> = UnsignedInteger::from_u64(0);
    pub const ONE: UnsignedInteger<NUM_LIMBS> = MontgomeryAlgorithms::cios(
        &UnsignedInteger::from_u64(1),
        &Self::R2,
        &M::MODULUS,
        &Self::MU,
    );
    const MODULUS_HAS_ONE_SPARE_BIT: bool = Self::modulus_has_one_spare_bit();

    /// Computes `- modulus^{-1} mod 2^{64}`
    /// This algorithm is given  by Dussé and Kaliski Jr. in
    /// "S. R. Dussé and B. S. Kaliski Jr. A cryptographic library for the Motorola
    /// DSP56000. In I. Damgård, editor, Advances in Cryptology – EUROCRYPT’90,
    /// volume 473 of Lecture Notes in Computer Science, pages 230–244. Springer,
    /// Heidelberg, May 1991."
    const fn compute_mu_parameter(modulus: &UnsignedInteger<NUM_LIMBS>) -> u64 {
        let mut y = 1;
        let word_size = 64;
        let mut i: usize = 2;
        while i <= word_size {
            let (_, lo) = UnsignedInteger::mul(modulus, &UnsignedInteger::from_u64(y));
            let least_significant_limb = lo.limbs[NUM_LIMBS - 1];
            if (least_significant_limb << (word_size - i)) >> (word_size - i) != 1 {
                y += 1 << (i - 1);
            }
            i += 1;
        }
        y.wrapping_neg()
    }

    /// Computes 2^{384 * 2} modulo `modulus`
    const fn compute_r2_parameter(
        modulus: &UnsignedInteger<NUM_LIMBS>,
    ) -> UnsignedInteger<NUM_LIMBS> {
        let word_size = 64;
        let mut l: usize = 0;
        let zero = UnsignedInteger::from_u64(0);
        // Define `c` as the largest power of 2 smaller than `modulus`
        while l < NUM_LIMBS * word_size {
            if UnsignedInteger::const_ne(&modulus.const_shr(l), &zero) {
                break;
            }
            l += 1;
        }
        let mut c = UnsignedInteger::from_u64(1).const_shl(l);

        // Double `c` and reduce modulo `modulus` until getting
        // `2^{2 * number_limbs * word_size}` mod `modulus`
        let mut i: usize = 1;
        while i <= 2 * NUM_LIMBS * word_size - l {
            let (double_c, overflow) = UnsignedInteger::add(&c, &c);
            c = if UnsignedInteger::const_le(modulus, &double_c) || overflow {
                UnsignedInteger::sub(&double_c, modulus).0
            } else {
                double_c
            };
            i += 1;
        }
        c
    }

    /// Checks whether the most significant limb of the modulus is at
    /// most `0x7FFFFFFFFFFFFFFE`. This check is useful since special
    /// optimizations exist for this kind of moduli.
    #[inline(always)]
    const fn modulus_has_one_spare_bit() -> bool {
        M::MODULUS.limbs[0] < (1u64 << 63) - 1
    }
}

impl<M, const NUM_LIMBS: usize> IsField for MontgomeryBackendPrimeField<M, NUM_LIMBS>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    type BaseType = UnsignedInteger<NUM_LIMBS>;

    #[inline(always)]
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let (sum, overflow) = UnsignedInteger::add(a, b);
        if Self::MODULUS_HAS_ONE_SPARE_BIT {
            if sum >= M::MODULUS {
                sum - M::MODULUS
            } else {
                sum
            }
        } else if overflow || sum >= M::MODULUS {
            let (diff, _) = UnsignedInteger::sub(&sum, &M::MODULUS);
            diff
        } else {
            sum
        }
    }

    #[inline(always)]
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        if Self::MODULUS_HAS_ONE_SPARE_BIT {
            MontgomeryAlgorithms::cios_optimized_for_moduli_with_one_spare_bit(
                a,
                b,
                &M::MODULUS,
                &Self::MU,
            )
        } else {
            MontgomeryAlgorithms::cios(a, b, &M::MODULUS, &Self::MU)
        }
    }

    #[inline(always)]
    fn square(a: &UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        MontgomeryAlgorithms::sos_square(a, &M::MODULUS, &Self::MU)
    }

    #[inline(always)]
    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        if b <= a { a - b } else { M::MODULUS - (b - a) }
    }

    #[inline(always)]
    fn neg(a: &Self::BaseType) -> Self::BaseType {
        if a == &Self::ZERO { *a } else { M::MODULUS - a }
    }

    #[inline(always)]
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        if a == &Self::ZERO {
            Err(FieldError::InvZeroError)
        } else {
            // Guajardo Kumar Paar Pelzl
            // Efficient Software-Implementation of Finite Fields with Applications to
            // Cryptography
            // Algorithm 16 (BEA for Inversion in Fp)

            //These can be done with const  functions
            let one: UnsignedInteger<NUM_LIMBS> = UnsignedInteger::from_u64(1);
            let modulus: UnsignedInteger<NUM_LIMBS> = M::MODULUS;
            let modulus_has_spare_bits = M::MODULUS.limbs[0] >> 63 == 0;

            let mut u: UnsignedInteger<NUM_LIMBS> = *a;
            let mut v = M::MODULUS;
            let mut b = Self::R2; // Avoids unnecessary reduction step.
            let mut c = Self::zero();

            while u != one && v != one {
                while u.limbs[NUM_LIMBS - 1] & 1 == 0 {
                    u >>= 1;
                    if b.limbs[NUM_LIMBS - 1] & 1 == 0 {
                        b >>= 1;
                    } else {
                        let carry;
                        (b, carry) = UnsignedInteger::<NUM_LIMBS>::add(&b, &modulus);
                        b >>= 1;
                        if !modulus_has_spare_bits && carry {
                            b.limbs[0] |= 1 << 63;
                        }
                    }
                }

                while v.limbs[NUM_LIMBS - 1] & 1 == 0 {
                    v >>= 1;

                    if c.limbs[NUM_LIMBS - 1] & 1 == 0 {
                        c >>= 1;
                    } else {
                        let carry;
                        (c, carry) = UnsignedInteger::<NUM_LIMBS>::add(&c, &modulus);
                        c >>= 1;
                        if !modulus_has_spare_bits && carry {
                            c.limbs[0] |= 1 << 63;
                        }
                    }
                }

                if v <= u {
                    u = u - v;
                    if b < c {
                        b = modulus - c + b;
                    } else {
                        b = b - c;
                    }
                } else {
                    v = v - u;
                    if c < b {
                        c = modulus - b + c;
                    } else {
                        c = c - b;
                    }
                }
            }

            if u == one { Ok(b) } else { Ok(c) }
        }
    }

    #[inline(always)]
    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = &Self::inv(b)?;
        Ok(Self::mul(a, b_inv))
    }

    #[inline(always)]
    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a == b
    }

    #[inline(always)]
    fn zero() -> Self::BaseType {
        Self::ZERO
    }

    #[inline(always)]
    fn one() -> Self::BaseType {
        Self::ONE
    }

    #[inline(always)]
    fn from_u64(x: u64) -> Self::BaseType {
        MontgomeryAlgorithms::cios(
            &UnsignedInteger::from_u64(x),
            &Self::R2,
            &M::MODULUS,
            &Self::MU,
        )
    }

    #[inline(always)]
    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        MontgomeryAlgorithms::cios(&x, &Self::R2, &M::MODULUS, &Self::MU)
    }
}

impl<M, const NUM_LIMBS: usize> IsPrimeField for MontgomeryBackendPrimeField<M, NUM_LIMBS>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    type CanonicalType = Self::BaseType;

    fn canonical(x: &Self::BaseType) -> Self::CanonicalType {
        MontgomeryAlgorithms::cios(x, &UnsignedInteger::from_u64(1), &M::MODULUS, &Self::MU)
    }

    fn field_bit_size() -> usize {
        let mut evaluated_bit = NUM_LIMBS * 64 - 1;
        let max_element = M::MODULUS - UnsignedInteger::<NUM_LIMBS>::from_u64(1);
        let one = UnsignedInteger::from_u64(1);

        while ((max_element >> evaluated_bit) & one) != one {
            evaluated_bit -= 1;
        }

        evaluated_bit + 1
    }

    fn from_hex(hex_string: &str) -> Result<Self::BaseType, CreationError> {
        let integer = Self::BaseType::from_hex(hex_string)?;
        if integer > M::MODULUS {
            return Err(CreationError::RepresentativeOutOfRange);
        }

        Ok(MontgomeryAlgorithms::cios(
            &integer,
            &MontgomeryBackendPrimeField::<M, NUM_LIMBS>::R2,
            &M::MODULUS,
            &MontgomeryBackendPrimeField::<M, NUM_LIMBS>::MU,
        ))
    }

    #[cfg(feature = "std")]
    fn to_hex(x: &Self::BaseType) -> String {
        Self::BaseType::to_hex(x)
    }
}

impl<M, const NUM_LIMBS: usize> FieldElement<MontgomeryBackendPrimeField<M, NUM_LIMBS>> where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug
{
}

impl<M, const NUM_LIMBS: usize> ByteConversion
    for FieldElement<MontgomeryBackendPrimeField<M, NUM_LIMBS>>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        MontgomeryAlgorithms::cios(
            self.value(),
            &UnsignedInteger::from_u64(1),
            &M::MODULUS,
            &MontgomeryBackendPrimeField::<M, NUM_LIMBS>::MU,
        )
        .to_bytes_be()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        MontgomeryAlgorithms::cios(
            self.value(),
            &UnsignedInteger::from_u64(1),
            &M::MODULUS,
            &MontgomeryBackendPrimeField::<M, NUM_LIMBS>::MU,
        )
        .to_bytes_le()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let value = UnsignedInteger::from_bytes_be(bytes)?;
        Ok(Self::new(value))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let value = UnsignedInteger::from_bytes_le(bytes)?;
        Ok(Self::new(value))
    }
}

#[cfg(feature = "alloc")]
impl<M, const NUM_LIMBS: usize> AsBytes for FieldElement<MontgomeryBackendPrimeField<M, NUM_LIMBS>>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.value().to_bytes_be()
    }
}

#[cfg(feature = "alloc")]
impl<M, const NUM_LIMBS: usize> From<FieldElement<MontgomeryBackendPrimeField<M, NUM_LIMBS>>>
    for alloc::vec::Vec<u8>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    fn from(value: FieldElement<MontgomeryBackendPrimeField<M, NUM_LIMBS>>) -> alloc::vec::Vec<u8> {
        value.value().to_bytes_be()
    }
}

impl<M, const NUM_LIMBS: usize> HasDefaultTranscript for MontgomeryBackendPrimeField<M, NUM_LIMBS>
where
    M: IsModulus<UnsignedInteger<NUM_LIMBS>> + Clone + Debug,
{
    /// # Panics
    ///
    /// This function will panic if NUM_LIMBS is greater than 6.
    fn get_random_field_element_from_rng(
        rng: &mut impl rand::Rng,
    ) -> FieldElement<MontgomeryBackendPrimeField<M, NUM_LIMBS>> {
        let mut buffer = [0u8; 6 * 8];
        let first_non_zero_limb_index = M::MODULUS
            .limbs
            .iter()
            .position(|&x| x != 0)
            .expect("modulus should be non-zero");
        let mask = u64::MAX >> M::MODULUS.limbs[first_non_zero_limb_index].leading_zeros();

        let bits_start_idx = first_non_zero_limb_index * 8;
        let bits_end_idx = NUM_LIMBS * 8;
        let mut uint_sample;

        loop {
            let sample_bytes = &mut buffer[bits_start_idx..bits_end_idx];
            rng.fill(sample_bytes);

            uint_sample = UnsignedInteger::from_bytes_be(&buffer).unwrap();

            uint_sample.limbs[first_non_zero_limb_index] &= mask;

            if uint_sample < M::MODULUS {
                break;
            }
        }

        FieldElement::new(MontgomeryBackendPrimeField::<M, NUM_LIMBS>::from_base_type(
            uint_sample,
        ))
    }
}
