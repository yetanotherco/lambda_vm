use crate::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
        traits::IsFFTField,
    },
    unsigned_integer::element::U64,
};

pub type U64MontgomeryBackendPrimeField<T> = MontgomeryBackendPrimeField<T, 1>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MontgomeryConfigU64GoldilocksPrimeField;
impl IsModulus<U64> for MontgomeryConfigU64GoldilocksPrimeField {
    //Babybear Prime p = 2^64 - 2^32 + 1
    const MODULUS: U64 = U64::from_u64(18446744069414584321);
}

pub type U64GoldilocksPrimeField =
    U64MontgomeryBackendPrimeField<MontgomeryConfigU64GoldilocksPrimeField>;

impl FieldElement<U64GoldilocksPrimeField> {
    pub fn to_bytes_le(&self) -> [u8; 8] {
        let limbs = self.representative().limbs;
        limbs[0].to_le_bytes()
    }

    pub fn to_bytes_be(&self) -> [u8; 8] {
        let limbs = self.representative().limbs;
        limbs[0].to_be_bytes()
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for FieldElement<U64GoldilocksPrimeField> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.representative().partial_cmp(&other.representative())
    }
}

impl Ord for FieldElement<U64GoldilocksPrimeField> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.representative().cmp(&other.representative())
    }
}

/// IsFFTField implementation for U64GoldilocksPrimeField
/// The field order is p = 2^64 - 2^32 + 1
/// p - 1 = 2^64 - 2^32 = 2^32 * (2^32 - 1)
/// So the two-adicity is 32.
///
/// The primitive 2^32-th root of unity was taken from Plonky3:
/// https://github.com/Plonky3/Plonky3/blob/main/goldilocks/src/lib.rs
impl IsFFTField for U64GoldilocksPrimeField {
    const TWO_ADICITY: u64 = 32;

    // 2^32-th primitive root of unity in Montgomery form
    // From Plonky3: 1753635133440165772
    // In Montgomery representation for this field
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: U64 = U64::from_u64(1753635133440165772);

    fn field_name() -> &'static str {
        "U64Goldilocks"
    }
}

#[cfg(test)]
mod test_u64_goldilocks_bytes_ops {
    use super::U64GoldilocksPrimeField;
    use crate::{field::element::FieldElement, traits::ByteConversion};

    #[test]
    #[cfg(feature = "alloc")]
    fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_le() {
        let element = FieldElement::<U64GoldilocksPrimeField>::from_hex_unchecked(
            "\
            0123456701234567\
        ",
        );
        let bytes = element.to_bytes_le();
        let expected_bytes: [u8; 8] = ByteConversion::to_bytes_le(&element).try_into().unwrap();
        assert_eq!(bytes, expected_bytes);
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_be() {
        let element = FieldElement::<U64GoldilocksPrimeField>::from_hex_unchecked(
            "\
            0123456701234567\
        ",
        );
        let bytes = element.to_bytes_be();
        let expected_bytes: [u8; 8] = ByteConversion::to_bytes_be(&element).try_into().unwrap();
        assert_eq!(bytes, expected_bytes);
    }

    #[test]

    fn byte_serialization_and_deserialization_works_le() {
        let element = FieldElement::<U64GoldilocksPrimeField>::from_hex_unchecked(
            "\
            7654321076543210\
        ",
        );
        let bytes = element.to_bytes_le();
        let from_bytes = FieldElement::<U64GoldilocksPrimeField>::from_bytes_le(&bytes).unwrap();
        assert_eq!(element, from_bytes);
    }

    #[test]

    fn byte_serialization_and_deserialization_works_be() {
        let element = FieldElement::<U64GoldilocksPrimeField>::from_hex_unchecked(
            "\
            7654321076543210\
        ",
        );
        let bytes = element.to_bytes_be();
        let from_bytes = FieldElement::<U64GoldilocksPrimeField>::from_bytes_be(&bytes).unwrap();
        assert_eq!(element, from_bytes);
    }
}
