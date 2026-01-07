use alloc::vec::Vec;
use math::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    },
    unsigned_integer::element::UnsignedInteger,
};

use crate::hash::{hash_to_field::hash_to_field, sha3::Sha3Hasher};

type F = MontgomeryBackendPrimeField<U64, 1>;

#[derive(Clone, Debug)]
struct U64;
impl IsModulus<UnsignedInteger<1>> for U64 {
    const MODULUS: UnsignedInteger<1> = UnsignedInteger::from_u64(18446744069414584321_u64);
}

#[test]
fn test_same_message_produce_same_field_elements() {
    let input = Sha3Hasher::expand_message(b"helloworld", b"dsttest", 500).unwrap();
    let field_elements: Vec<FieldElement<F>> = hash_to_field(&input, 40);
    let other_field_elements = hash_to_field(&input, 40);
    assert_eq!(field_elements, other_field_elements);
}
