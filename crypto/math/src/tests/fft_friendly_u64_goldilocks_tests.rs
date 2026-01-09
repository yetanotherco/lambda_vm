use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks::U64GoldilocksPrimeField;
use crate::field::traits::IsFFTField;
use crate::traits::ByteConversion;

type F = U64GoldilocksPrimeField;
type FE = FieldElement<F>;

#[test]
fn two_adic_primitve_root_of_unity_is_correct() {
    // The primitive root should have order 2^TWO_ADICITY
    let root = F::get_primitive_root_of_unity(F::TWO_ADICITY).unwrap();
    let order = 1u64 << F::TWO_ADICITY;

    // root^(2^TWO_ADICITY) should be 1
    assert_eq!(root.pow(order), FE::one());

    // root^(2^(TWO_ADICITY-1)) should NOT be 1 (it should be -1)
    let half_order = order / 2;
    assert_ne!(root.pow(half_order), FE::one());
}

#[test]
fn primitive_root_of_unity_powers() {
    // Test that we can get roots of unity for various orders
    for order in 1..=16 {
        let root = F::get_primitive_root_of_unity(order).unwrap();
        let n = 1u64 << order;

        // root^n should be 1
        assert_eq!(root.pow(n), FE::one(), "Root of order {} failed", order);

        // root^(n/2) should not be 1 for order > 0
        if order > 0 {
            assert_ne!(
                root.pow(n / 2),
                FE::one(),
                "Root of order {} is not primitive",
                order
            );
        }
    }
}

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
