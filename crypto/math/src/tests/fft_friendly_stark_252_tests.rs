use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use crate::traits::ByteConversion;

#[test]
#[cfg(feature = "alloc")]
fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_le() {
    let element = FieldElement::<Stark252PrimeField>::from_hex_unchecked(
        "\
        0123456701234567\
        0123456701234567\
        0123456701234567\
        0123456701234567\
    ",
    );
    let bytes = element.to_bytes_le();
    let expected_bytes: [u8; 32] = ByteConversion::to_bytes_le(&element).try_into().unwrap();
    assert_eq!(bytes, expected_bytes);
}

#[test]
#[cfg(feature = "alloc")]
fn byte_serialization_for_a_number_matches_with_byte_conversion_implementation_be() {
    let element = FieldElement::<Stark252PrimeField>::from_hex_unchecked(
        "\
        0123456701234567\
        0123456701234567\
        0123456701234567\
        0123456701234567\
    ",
    );
    let bytes = element.to_bytes_be();
    let expected_bytes: [u8; 32] = ByteConversion::to_bytes_be(&element).try_into().unwrap();
    assert_eq!(bytes, expected_bytes);
}

#[test]
fn byte_serialization_and_deserialization_works_le() {
    let element = FieldElement::<Stark252PrimeField>::from_hex_unchecked(
        "\
        0123456701234567\
        7654321076543210\
        7654321076543210\
        7654321076543210\
    ",
    );
    let bytes = element.to_bytes_le();
    let from_bytes = FieldElement::<Stark252PrimeField>::from_bytes_le(&bytes).unwrap();
    assert_eq!(element, from_bytes);
}

#[test]
fn byte_serialization_and_deserialization_works_be() {
    let element = FieldElement::<Stark252PrimeField>::from_hex_unchecked(
        "\
        0123456701234567\
        7654321076543210\
        7654321076543210\
        7654321076543210\
    ",
    );
    let bytes = element.to_bytes_be();
    let from_bytes = FieldElement::<Stark252PrimeField>::from_bytes_be(&bytes).unwrap();
    assert_eq!(element, from_bytes);
}
