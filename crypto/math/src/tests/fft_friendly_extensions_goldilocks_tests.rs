use crate::field::{
    element::FieldElement,
    fields::fft_friendly::{
        extensions_goldilocks::{Fp2E, Fp3E},
        u64_goldilocks::GoldilocksField,
    },
};

type FpE = FieldElement<GoldilocksField>;

// =====================================================
// QUADRATIC EXTENSION TESTS
// =====================================================

#[test]
fn test_fp2_add() {
    let a = Fp2E::new([FpE::from(0u64), FpE::from(3u64)]);
    let b = Fp2E::new([FpE::from(2u64), FpE::from(8u64)]);
    let expected = Fp2E::new([FpE::from(2u64), FpE::from(11u64)]);
    assert_eq!(a + b, expected);
}

#[test]
fn test_fp2_sub() {
    let a = Fp2E::new([FpE::from(10u64), FpE::from(5u64)]);
    let b = Fp2E::new([FpE::from(3u64), FpE::from(2u64)]);
    let expected = Fp2E::new([FpE::from(7u64), FpE::from(3u64)]);
    assert_eq!(a - b, expected);
}

#[test]
fn test_fp2_mul() {
    // (a0 + a1*w) * (b0 + b1*w) = (a0*b0 + a1*b1*7) + (a0*b1 + a1*b0)*w
    let a = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
    let b = Fp2E::new([FpE::from(4u64), FpE::from(5u64)]);
    // c0 = 2*4 + 3*5*7 = 8 + 105 = 113
    // c1 = 2*5 + 3*4 = 10 + 12 = 22
    let expected = Fp2E::new([FpE::from(113u64), FpE::from(22u64)]);
    assert_eq!(a * b, expected);
}

#[test]
fn test_fp2_mul_by_one() {
    let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
    let one = Fp2E::one();
    assert_eq!(a * one, a);
}

#[test]
fn test_fp2_mul_by_zero() {
    let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
    let zero = Fp2E::zero();
    assert_eq!(a * zero, zero);
}

#[test]
fn test_fp2_inv() {
    let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
    let a_inv = a.inv().unwrap();
    assert_eq!(a * a_inv, Fp2E::one());
}

#[test]
fn test_fp2_inv_one() {
    let one = Fp2E::one();
    assert_eq!(one.inv().unwrap(), one);
}

#[test]
fn test_fp2_div() {
    let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
    let b = Fp2E::new([FpE::from(4u64), FpE::from(2u64)]);
    let result = (a / b).unwrap();
    assert_eq!(result * b, a);
}

#[test]
fn test_fp2_pow() {
    let a = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
    let a_squared = a * a;
    let a_cubed = a_squared * a;
    assert_eq!(a.pow(2u64), a_squared);
    assert_eq!(a.pow(3u64), a_cubed);
}

#[test]
fn test_fp2_conjugate() {
    let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
    let expected = Fp2E::new([FpE::from(12u64), -FpE::from(5u64)]);
    assert_eq!(a.conjugate(), expected);
}

#[test]
fn test_fp2_neg() {
    let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
    let neg_a = -a;
    assert_eq!(a + neg_a, Fp2E::zero());
}

#[test]
fn test_fp2_square_equals_mul() {
    let a = Fp2E::new([FpE::from(7u64), FpE::from(11u64)]);
    assert_eq!(a.square(), a * a);
}

#[test]
fn test_fp2_double() {
    let a = Fp2E::new([FpE::from(7u64), FpE::from(11u64)]);
    assert_eq!(a.double(), a + a);
}

// =====================================================
// CUBIC EXTENSION TESTS
// =====================================================

#[test]
fn test_fp3_add() {
    let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
    let b = Fp3E::new([FpE::from(4u64), FpE::from(5u64), FpE::from(6u64)]);
    let expected = Fp3E::new([FpE::from(5u64), FpE::from(7u64), FpE::from(9u64)]);
    assert_eq!(a + b, expected);
}

#[test]
fn test_fp3_sub() {
    let a = Fp3E::new([FpE::from(10u64), FpE::from(8u64), FpE::from(6u64)]);
    let b = Fp3E::new([FpE::from(3u64), FpE::from(2u64), FpE::from(1u64)]);
    let expected = Fp3E::new([FpE::from(7u64), FpE::from(6u64), FpE::from(5u64)]);
    assert_eq!(a - b, expected);
}

#[test]
fn test_fp3_mul_by_one() {
    let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
    let one = Fp3E::one();
    assert_eq!(a * one, a);
}

#[test]
fn test_fp3_mul_by_zero() {
    let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
    let zero = Fp3E::zero();
    assert_eq!(a * zero, zero);
}

#[test]
fn test_fp3_mul_commutativity() {
    let a = Fp3E::new([FpE::from(1u64), FpE::from(2u64), FpE::from(3u64)]);
    let b = Fp3E::new([FpE::from(4u64), FpE::from(5u64), FpE::from(6u64)]);
    assert_eq!(a * b, b * a);
}

#[test]
fn test_fp3_inv() {
    let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
    let a_inv = a.inv().unwrap();
    assert_eq!(a * a_inv, Fp3E::one());
}

#[test]
fn test_fp3_inv_one() {
    let one = Fp3E::one();
    assert_eq!(one.inv().unwrap(), one);
}

#[test]
fn test_fp3_div() {
    let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
    let b = Fp3E::new([FpE::from(4u64), FpE::from(2u64), FpE::from(3u64)]);
    let result = (a / b).unwrap();
    assert_eq!(result * b, a);
}

#[test]
fn test_fp3_pow() {
    let a = Fp3E::new([FpE::from(2u64), FpE::from(3u64), FpE::from(4u64)]);
    let a_squared = a * a;
    let a_cubed = a_squared * a;
    assert_eq!(a.pow(2u64), a_squared);
    assert_eq!(a.pow(3u64), a_cubed);
}

#[test]
fn test_fp3_neg() {
    let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
    let neg_a = -a;
    assert_eq!(a + neg_a, Fp3E::zero());
}

#[test]
fn test_fp3_square_equals_mul() {
    let a = Fp3E::new([FpE::from(7u64), FpE::from(11u64), FpE::from(13u64)]);
    assert_eq!(a.square(), a * a);
}

#[test]
fn test_fp3_double() {
    let a = Fp3E::new([FpE::from(7u64), FpE::from(11u64), FpE::from(13u64)]);
    assert_eq!(a.double(), a + a);
}

// =====================================================
// EMBEDDING TESTS (Base field into extension)
// =====================================================

#[test]
fn test_fp2_from_base() {
    let base = FpE::from(42u64);
    let ext = Fp2E::from(42u64);
    assert_eq!(ext.value()[0], base);
    assert_eq!(ext.value()[1], FpE::zero());
}

#[test]
fn test_fp3_from_base() {
    let base = FpE::from(42u64);
    let ext = Fp3E::from(42u64);
    assert_eq!(ext.value()[0], base);
    assert_eq!(ext.value()[1], FpE::zero());
    assert_eq!(ext.value()[2], FpE::zero());
}

#[test]
fn test_fp2_base_mul() {
    let a = FpE::from(5u64);
    let b = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
    let result = a * b;
    let expected = Fp2E::new([FpE::from(10u64), FpE::from(15u64)]);
    assert_eq!(result, expected);
}

#[test]
fn test_fp3_base_mul() {
    let a = FpE::from(5u64);
    let b = Fp3E::new([FpE::from(2u64), FpE::from(3u64), FpE::from(4u64)]);
    let result = a * b;
    let expected = Fp3E::new([FpE::from(10u64), FpE::from(15u64), FpE::from(20u64)]);
    assert_eq!(result, expected);
}

// =====================================================
// ASSOCIATIVITY AND DISTRIBUTIVITY TESTS
// =====================================================

#[test]
fn test_fp2_mul_associativity() {
    let a = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
    let b = Fp2E::new([FpE::from(4u64), FpE::from(5u64)]);
    let c = Fp2E::new([FpE::from(6u64), FpE::from(7u64)]);
    assert_eq!((a * b) * c, a * (b * c));
}

#[test]
fn test_fp3_mul_associativity() {
    let a = Fp3E::new([FpE::from(2u64), FpE::from(3u64), FpE::from(4u64)]);
    let b = Fp3E::new([FpE::from(5u64), FpE::from(6u64), FpE::from(7u64)]);
    let c = Fp3E::new([FpE::from(8u64), FpE::from(9u64), FpE::from(10u64)]);
    assert_eq!((a * b) * c, a * (b * c));
}

#[test]
fn test_fp2_distributivity() {
    let a = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
    let b = Fp2E::new([FpE::from(4u64), FpE::from(5u64)]);
    let c = Fp2E::new([FpE::from(6u64), FpE::from(7u64)]);
    assert_eq!(a * (b + c), a * b + a * c);
}

#[test]
fn test_fp3_distributivity() {
    let a = Fp3E::new([FpE::from(2u64), FpE::from(3u64), FpE::from(4u64)]);
    let b = Fp3E::new([FpE::from(5u64), FpE::from(6u64), FpE::from(7u64)]);
    let c = Fp3E::new([FpE::from(8u64), FpE::from(9u64), FpE::from(10u64)]);
    assert_eq!(a * (b + c), a * b + a * c);
}

// =====================================================
// LARGE VALUE TESTS
// =====================================================

#[test]
fn test_fp2_large_values() {
    let a = Fp2E::new([
        FpE::from(18446744069414584300u64),
        FpE::from(12345678901234567u64),
    ]);
    let b = Fp2E::new([
        FpE::from(9876543210987654u64),
        FpE::from(11111111111111111u64),
    ]);

    let a_inv = a.inv().unwrap();
    assert_eq!(a * a_inv, Fp2E::one());

    let result = (a / b).unwrap();
    assert_eq!(result * b, a);
}

#[test]
fn test_fp3_large_values() {
    let a = Fp3E::new([
        FpE::from(18446744069414584300u64),
        FpE::from(12345678901234567u64),
        FpE::from(98765432109876543u64),
    ]);
    let b = Fp3E::new([
        FpE::from(9876543210987654u64),
        FpE::from(11111111111111111u64),
        FpE::from(22222222222222222u64),
    ]);

    let a_inv = a.inv().unwrap();
    assert_eq!(a * a_inv, Fp3E::one());

    let result = (a / b).unwrap();
    assert_eq!(result * b, a);
}

// =====================================================
// BATCH INVERSE TESTS
// =====================================================

#[test]
fn test_fp2_batch_inverse() {
    let a = Fp2E::new([FpE::from(2u64), FpE::from(3u64)]);
    let b = Fp2E::new([FpE::from(4u64), FpE::from(5u64)]);
    let c = Fp2E::new([FpE::from(6u64), FpE::from(7u64)]);
    let d = Fp2E::new([FpE::from(8u64), FpE::from(9u64)]);

    let original = [a, b, c, d];
    let mut to_invert = original;

    Fp2E::inplace_batch_inverse(&mut to_invert).unwrap();

    // Verify each element was correctly inverted
    for (inv, orig) in to_invert.iter().zip(original.iter()) {
        assert_eq!(*inv * *orig, Fp2E::one());
    }
}

#[test]
fn test_fp2_batch_inverse_single_element() {
    let a = Fp2E::new([FpE::from(12u64), FpE::from(5u64)]);
    let mut to_invert = [a];

    Fp2E::inplace_batch_inverse(&mut to_invert).unwrap();

    assert_eq!(to_invert[0] * a, Fp2E::one());
    assert_eq!(to_invert[0], a.inv().unwrap());
}

#[test]
fn test_fp2_batch_inverse_empty() {
    let mut empty: [Fp2E; 0] = [];
    assert!(Fp2E::inplace_batch_inverse(&mut empty).is_ok());
}

#[test]
fn test_fp3_batch_inverse() {
    let a = Fp3E::new([FpE::from(2u64), FpE::from(3u64), FpE::from(4u64)]);
    let b = Fp3E::new([FpE::from(5u64), FpE::from(6u64), FpE::from(7u64)]);
    let c = Fp3E::new([FpE::from(8u64), FpE::from(9u64), FpE::from(10u64)]);
    let d = Fp3E::new([FpE::from(11u64), FpE::from(12u64), FpE::from(13u64)]);

    let original = [a, b, c, d];
    let mut to_invert = original;

    Fp3E::inplace_batch_inverse(&mut to_invert).unwrap();

    // Verify each element was correctly inverted
    for (inv, orig) in to_invert.iter().zip(original.iter()) {
        assert_eq!(*inv * *orig, Fp3E::one());
    }
}

#[test]
fn test_fp3_batch_inverse_single_element() {
    let a = Fp3E::new([FpE::from(12u64), FpE::from(5u64), FpE::from(7u64)]);
    let mut to_invert = [a];

    Fp3E::inplace_batch_inverse(&mut to_invert).unwrap();

    assert_eq!(to_invert[0] * a, Fp3E::one());
    assert_eq!(to_invert[0], a.inv().unwrap());
}

#[test]
fn test_fp3_batch_inverse_empty() {
    let mut empty: [Fp3E; 0] = [];
    assert!(Fp3E::inplace_batch_inverse(&mut empty).is_ok());
}

#[test]
fn test_fp2_batch_inverse_large() {
    // Test with larger batch
    let elements: Vec<Fp2E> = (1u64..=20)
        .map(|i| Fp2E::new([FpE::from(i), FpE::from(i + 100)]))
        .collect();

    let original = elements.clone();
    let mut to_invert = elements;

    Fp2E::inplace_batch_inverse(&mut to_invert).unwrap();

    for (inv, orig) in to_invert.iter().zip(original.iter()) {
        assert_eq!(*inv * *orig, Fp2E::one());
    }
}

#[test]
fn test_fp3_batch_inverse_large() {
    // Test with larger batch
    let elements: Vec<Fp3E> = (1u64..=20)
        .map(|i| Fp3E::new([FpE::from(i), FpE::from(i + 100), FpE::from(i + 200)]))
        .collect();

    let original = elements.clone();
    let mut to_invert = elements;

    Fp3E::inplace_batch_inverse(&mut to_invert).unwrap();

    for (inv, orig) in to_invert.iter().zip(original.iter()) {
        assert_eq!(*inv * *orig, Fp3E::one());
    }
}
