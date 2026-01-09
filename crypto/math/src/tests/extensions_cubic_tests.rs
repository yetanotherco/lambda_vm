use crate::field::element::FieldElement;
use crate::field::extensions::cubic::{CubicExtensionField, HasCubicNonResidue};
use crate::field::fields::u64_prime_field::{U64FieldElement, U64PrimeField};

const ORDER_P: u64 = 13;

#[derive(Debug, Clone)]
struct MyCubicNonResidue;
impl HasCubicNonResidue<U64PrimeField<ORDER_P>> for MyCubicNonResidue {
    fn residue() -> FieldElement<U64PrimeField<ORDER_P>> {
        -FieldElement::from(11)
    }
}

type FE = U64FieldElement<ORDER_P>;
type MyFieldExtensionBackend = CubicExtensionField<U64PrimeField<ORDER_P>, MyCubicNonResidue>;
#[allow(clippy::upper_case_acronyms)]
type FEE = FieldElement<MyFieldExtensionBackend>;

#[test]
fn test_add_1() {
    let a = FEE::new([FE::new(0), FE::new(3), FE::new(5)]);
    let b = FEE::new([-FE::new(2), FE::new(8), FE::new(10)]);
    let expected_result = FEE::new([FE::new(11), FE::new(11), FE::new(15)]);
    assert_eq!(a + b, expected_result);
}

#[test]
fn test_add_2() {
    let a = FEE::new([FE::new(12), FE::new(5), FE::new(3)]);
    let b = FEE::new([-FE::new(4), FE::new(2), FE::new(8)]);
    let expected_result = FEE::new([FE::new(8), FE::new(7), FE::new(11)]);
    assert_eq!(a + b, expected_result);
}

#[test]
fn test_sub_1() {
    let a = FEE::new([FE::new(0), FE::new(3), FE::new(3)]);
    let b = FEE::new([-FE::new(2), FE::new(8), FE::new(2)]);
    let expected_result = FEE::new([FE::new(2), FE::new(8), FE::new(1)]);
    assert_eq!(a - b, expected_result);
}

#[test]
fn test_sub_2() {
    let a = FEE::new([FE::new(12), FE::new(5), FE::new(3)]);
    let b = FEE::new([-FE::new(4), FE::new(2), FE::new(8)]);
    let expected_result = FEE::new([FE::new(16), FE::new(3), FE::new(8)]);
    assert_eq!(a - b, expected_result);
}

#[test]
fn test_mul_1() {
    let a = FEE::new([FE::new(0), FE::new(3), FE::new(5)]);
    let b = FEE::new([-FE::new(2), FE::new(8), FE::new(6)]);
    let expected_result = FEE::new([FE::new(12), FE::new(2), FE::new(1)]);
    assert_eq!(a * b, expected_result);
}

#[test]
fn test_mul_2() {
    let a = FEE::new([FE::new(12), FE::new(5), FE::new(11)]);
    let b = FEE::new([-FE::new(4), FE::new(2), FE::new(15)]);
    let expected_result = FEE::new([FE::new(3), FE::new(9), FE::new(3)]);
    assert_eq!(a * b, expected_result);
}

#[test]
fn test_div_1() {
    let a = FEE::new([FE::new(0), FE::new(3), FE::new(2)]);
    let b = FEE::new([-FE::new(2), FE::new(8), FE::new(5)]);
    let expected_result = FEE::new([FE::new(12), FE::new(6), FE::new(1)]);
    assert_eq!((a / b).unwrap(), expected_result);
}

#[test]
fn test_div_2() {
    let a = FEE::new([FE::new(12), FE::new(5), FE::new(4)]);
    let b = FEE::new([-FE::new(4), FE::new(2), FE::new(2)]);
    let expected_result = FEE::new([FE::new(3), FE::new(8), FE::new(11)]);
    assert_eq!((a / b).unwrap(), expected_result);
}

#[test]
fn test_pow_1() {
    let a = FEE::new([FE::new(0), FE::new(3), FE::new(3)]);
    let b: u64 = 5;
    let expected_result = FEE::new([FE::new(7), FE::new(3), FE::new(1)]);
    assert_eq!(a.pow(b), expected_result);
}

#[test]
fn test_pow_2() {
    let a = FEE::new([FE::new(12), FE::new(5), FE::new(3)]);
    let b: u64 = 8;
    let expected_result = FEE::new([FE::new(5), FE::new(5), FE::new(12)]);
    assert_eq!(a.pow(b), expected_result);
}

#[test]
fn test_inv() {
    let a = FEE::new([FE::new(12), FE::new(5), FE::new(3)]);
    let expected_result = FEE::new([FE::new(2), FE::new(2), FE::new(3)]);
    assert_eq!(a.inv().unwrap(), expected_result);
}

#[test]
fn test_inv_1() {
    let a = FEE::new([FE::new(1), FE::new(0), FE::new(1)]);
    let expected_result = FEE::new([FE::new(8), FE::new(3), FE::new(5)]);
    assert_eq!(a.inv().unwrap(), expected_result);
}

#[test]
fn test_add_as_subfield_1() {
    let a = FE::new(5);
    let b = FEE::new([-FE::new(2), FE::new(8), FE::new(10)]);
    let expected_result = FEE::new([FE::new(3), FE::new(8), FE::new(10)]);
    assert_eq!(a + b, expected_result);
}

#[test]
fn test_add_as_subfield_2() {
    let a = FE::new(12);
    let b = FEE::new([-FE::new(4), FE::new(2), FE::new(8)]);
    let expected_result = FEE::new([FE::new(8), FE::new(2), FE::new(8)]);
    assert_eq!(a + b, expected_result);
}

#[test]
fn test_sub_as_subfield_1() {
    let a = FE::new(3);
    let b = FEE::new([-FE::new(2), FE::new(8), FE::new(2)]);
    let expected_result = FEE::new([FE::new(5), FE::new(5), FE::new(11)]);
    assert_eq!(a - b, expected_result);
}

#[test]
fn test_sub_as_subfield_2() {
    let a = FE::new(12);
    let b = FEE::new([-FE::new(4), FE::new(2), FE::new(3)]);
    let expected_result = FEE::new([FE::new(3), FE::new(11), FE::new(10)]);
    assert_eq!(a - b, expected_result);
}

#[test]
fn test_mul_as_subfield_1() {
    let a = FE::new(5);
    let b = FEE::new([-FE::new(2), FE::new(8), FE::new(6)]);
    let expected_result = FEE::new([FE::new(3), FE::new(1), FE::new(4)]);
    assert_eq!(a * b, expected_result);
}

#[test]
fn test_mul_as_subfield_2() {
    let a = FE::new(11);
    let b = FEE::new([-FE::new(4), FE::new(2), FE::new(15)]);
    let expected_result = FEE::new([FE::new(8), FE::new(9), FE::new(9)]);
    assert_eq!(a * b, expected_result);
}

#[test]
fn test_div_as_subfield_1() {
    let a = FE::new(2);
    let b = FEE::new([-FE::new(2), FE::new(8), FE::new(5)]);
    let expected_result = FEE::new([FE::new(8), FE::new(4), FE::new(10)]);
    assert_eq!((a / b).unwrap(), expected_result);
}

#[test]
fn test_div_as_subfield_2() {
    let a = FE::new(4);
    let b = FEE::new([-FE::new(4), FE::new(2), FE::new(2)]);
    let expected_result = FEE::new([FE::new(3), FE::new(6), FE::new(11)]);
    assert_eq!((a / b).unwrap(), expected_result);
}
