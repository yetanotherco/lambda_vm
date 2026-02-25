use crate::field::element::FieldElement;
use crate::field::extensions::quadratic::{HasQuadraticNonResidue, QuadraticExtensionField};
use crate::field::test_fields::u64_test_field::U64Field;

const ORDER_P: u64 = 59;

#[derive(Debug, Clone)]
struct MyQuadraticNonResidue;
impl HasQuadraticNonResidue<U64Field<ORDER_P>> for MyQuadraticNonResidue {
    fn residue() -> FieldElement<U64Field<ORDER_P>> {
        -FieldElement::one()
    }
}

type FE = FieldElement<U64Field<ORDER_P>>;
type MyFieldExtensionBackend = QuadraticExtensionField<U64Field<ORDER_P>, MyQuadraticNonResidue>;
#[allow(clippy::upper_case_acronyms)]
type FEE = FieldElement<MyFieldExtensionBackend>;

#[test]
fn test_add_1() {
    let a = FEE::new([FE::new(0), FE::new(3)]);
    let b = FEE::new([-FE::new(2), FE::new(8)]);
    let expected_result = FEE::new([FE::new(57), FE::new(11)]);
    assert_eq!(a + b, expected_result);
}

#[test]
fn test_add_2() {
    let a = FEE::new([FE::new(12), FE::new(5)]);
    let b = FEE::new([-FE::new(4), FE::new(2)]);
    let expected_result = FEE::new([FE::new(8), FE::new(7)]);
    assert_eq!(a + b, expected_result);
}

#[test]
fn test_sub_1() {
    let a = FEE::new([FE::new(0), FE::new(3)]);
    let b = FEE::new([-FE::new(2), FE::new(8)]);
    let expected_result = FEE::new([FE::new(2), FE::new(54)]);
    assert_eq!(a - b, expected_result);
}

#[test]
fn test_sub_2() {
    let a = FEE::new([FE::new(12), FE::new(5)]);
    let b = FEE::new([-FE::new(4), FE::new(2)]);
    let expected_result = FEE::new([FE::new(16), FE::new(3)]);
    assert_eq!(a - b, expected_result);
}

#[test]
fn test_mul_1() {
    let a = FEE::new([FE::new(0), FE::new(3)]);
    let b = FEE::new([-FE::new(2), FE::new(8)]);
    let expected_result = FEE::new([FE::new(35), FE::new(53)]);
    assert_eq!(a * b, expected_result);
}

#[test]
fn test_mul_2() {
    let a = FEE::new([FE::new(12), FE::new(5)]);
    let b = FEE::new([-FE::new(4), FE::new(2)]);
    let expected_result = FEE::new([FE::new(1), FE::new(4)]);
    assert_eq!(a * b, expected_result);
}

#[test]
fn test_div_1() {
    let a = FEE::new([FE::new(0), FE::new(3)]);
    let b = FEE::new([-FE::new(2), FE::new(8)]);
    let expected_result = FEE::new([FE::new(42), FE::new(19)]);
    assert_eq!((a / b).unwrap(), expected_result);
}

#[test]
fn test_div_2() {
    let a = FEE::new([FE::new(12), FE::new(5)]);
    let b = FEE::new([-FE::new(4), FE::new(2)]);
    let expected_result = FEE::new([FE::new(4), FE::new(45)]);
    assert_eq!((a / b).unwrap(), expected_result);
}

#[test]
fn test_pow_1() {
    let a = FEE::new([FE::new(0), FE::new(3)]);
    let b: u64 = 5;
    let expected_result = FEE::new([FE::new(0), FE::new(7)]);
    assert_eq!(a.pow(b), expected_result);
}

#[test]
fn test_pow_2() {
    let a = FEE::new([FE::new(12), FE::new(5)]);
    let b: u64 = 8;
    let expected_result = FEE::new([FE::new(52), FE::new(35)]);
    assert_eq!(a.pow(b), expected_result);
}

#[test]
fn test_inv_1() {
    let a = FEE::new([FE::new(0), FE::new(3)]);
    let expected_result = FEE::new([FE::new(0), FE::new(39)]);
    assert_eq!(a.inv().unwrap(), expected_result);
}

#[test]
fn test_inv() {
    let a = FEE::new([FE::new(12), FE::new(5)]);
    let expected_result = FEE::new([FE::new(28), FE::new(8)]);
    assert_eq!(a.inv().unwrap(), expected_result);
}

#[test]
fn test_conjugate() {
    let a = FEE::new([FE::new(12), FE::new(5)]);
    let expected_result = FEE::new([FE::new(12), -FE::new(5)]);
    assert_eq!(a.conjugate(), expected_result);
}

#[test]
fn test_add_as_subfield_1() {
    let a = -FE::new(2);
    let b = FEE::new([FE::new(0), FE::new(3)]);
    let expected_result = FEE::new([FE::new(57), FE::new(3)]);
    assert_eq!(a + b, expected_result);
}

#[test]
fn test_add_as_subfield_2() {
    let a = -FE::new(4);
    let b = FEE::new([FE::new(12), FE::new(5)]);
    let expected_result = FEE::new([FE::new(8), FE::new(5)]);
    assert_eq!(a + b, expected_result);
}

#[test]
fn test_sub_as_subfield_1() {
    let a = FE::new(0);
    let b = FEE::new([-FE::new(2), FE::new(8)]);
    let expected_result = FEE::new([FE::new(2), FE::new(51)]);
    assert_eq!(a - b, expected_result);
}

#[test]
fn test_sub_a_subfield_2() {
    let a = FE::new(12);
    let b = FEE::new([-FE::new(4), -FE::new(2)]);
    let expected_result = FEE::new([FE::new(16), FE::new(2)]);
    assert_eq!(a - b, expected_result);
}

#[test]
fn test_mul_as_subfield_1() {
    let a = FE::new(2);
    let b = FEE::new([-FE::new(2), FE::new(8)]);
    let expected_result = FEE::new([FE::new(55), FE::new(16)]);
    assert_eq!(a * b, expected_result);
}

#[test]
fn test_mul_as_subfield_2() {
    let a = FE::new(12);
    let b = FEE::new([-FE::new(4), FE::new(2)]);
    let expected_result = FEE::new([FE::new(11), FE::new(24)]);
    assert_eq!(a * b, expected_result);
}

#[test]
fn test_div_as_subfield_1() {
    let a = FE::new(3);
    let b = FEE::new([-FE::new(2), FE::new(8)]);
    let expected_result = FEE::new([FE::new(19), FE::new(17)]);
    assert_eq!((a / b).unwrap(), expected_result);
}

#[test]
fn test_div_as_subfield_2() {
    let a = FE::new(22);
    let b = FEE::new([FE::new(4), FE::new(2)]);
    let expected_result = FEE::new([FE::new(28), FE::new(45)]);
    assert_eq!((a / b).unwrap(), expected_result);
}
