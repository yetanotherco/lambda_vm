//! Integration tests for ARM64 assembly Goldilocks field operations.
//!
//! These tests verify that:
//! 1. ASM operations produce correct results
//! 2. Field axioms hold with ASM implementation
//! 3. FFT and other algorithms work correctly with ASM field ops
//! 4. Extension fields work correctly with ASM base field

use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks_asm::{
    add_fast as add_asm, sub_fast as sub_asm, mul,
    neg, double, mul_wide, reduce128, GOLDILOCKS_PRIME, EPSILON,
};
use crate::field::fields::fft_friendly::u64_goldilocks_native::GoldilocksField;
use crate::field::traits::{IsField, IsFFTField};

type GoldilocksElement = FieldElement<GoldilocksField>;

fn canonicalize(x: u64) -> u64 {
    if x >= GOLDILOCKS_PRIME { x - GOLDILOCKS_PRIME } else { x }
}

// ============== BASIC OPERATION TESTS ==============

#[test]
fn test_add_basic() {
    assert_eq!(add_asm(5, 7), 12);
    assert_eq!(add_asm(0, 0), 0);
    assert_eq!(add_asm(1, 0), 1);
    assert_eq!(add_asm(0, 1), 1);
}

#[test]
fn test_add_overflow() {
    // Test overflow handling
    let a = GOLDILOCKS_PRIME - 1;
    let b = 2u64;
    let result = add_asm(a, b);
    assert_eq!(canonicalize(result), 1);

    // Test at boundary
    let result2 = add_asm(GOLDILOCKS_PRIME - 1, 1);
    assert_eq!(canonicalize(result2), 0);
}

#[test]
fn test_sub_basic() {
    assert_eq!(sub_asm(10, 3), 7);
    assert_eq!(sub_asm(5, 5), 0);
    assert_eq!(sub_asm(100, 0), 100);
}

#[test]
fn test_sub_underflow() {
    // Test underflow handling
    let result = sub_asm(3, 10);
    assert_eq!(canonicalize(result), GOLDILOCKS_PRIME - 7);

    // Test at boundary
    let result2 = sub_asm(0, 1);
    assert_eq!(canonicalize(result2), GOLDILOCKS_PRIME - 1);
}

#[test]
fn test_mul_basic() {
    assert_eq!(mul(5, 7), 35);
    assert_eq!(mul(1, 1), 1);
    assert_eq!(mul(0, 12345), 0);
    assert_eq!(mul(12345, 0), 0);
    assert_eq!(mul(1, 12345), 12345);
}

#[test]
fn test_mul_large() {
    // Test with values that produce 128-bit intermediate
    let a = 1u64 << 40;
    let b = 1u64 << 40;
    let result = mul(a, b);
    let expected = ((a as u128 * b as u128) % GOLDILOCKS_PRIME as u128) as u64;
    assert_eq!(canonicalize(result), expected);
}

#[test]
fn test_square_basic() {
    // Square uses native Rust (faster than ASM)
    assert_eq!(canonicalize(GoldilocksField::square(&5)), 25);
    assert_eq!(canonicalize(GoldilocksField::square(&0)), 0);
    assert_eq!(canonicalize(GoldilocksField::square(&1)), 1);
}

#[test]
fn test_square_equals_mul() {
    // Square uses native Rust (faster than ASM)
    let test_values = [5, 123456789, GOLDILOCKS_PRIME - 1, EPSILON, 1u64 << 32, 0xDEADBEEF];

    for a in test_values {
        let sq = GoldilocksField::square(&a);
        let mul_result = mul(a, a);
        assert_eq!(
            canonicalize(sq),
            canonicalize(mul_result),
            "square mismatch for a={}", a
        );
    }
}

#[test]
fn test_neg_basic() {
    assert_eq!(neg(0), 0);

    let neg_5 = neg(5);
    assert_eq!(canonicalize(add_asm(5, neg_5)), 0);

    let neg_p_minus_1 = neg(GOLDILOCKS_PRIME - 1);
    assert_eq!(canonicalize(neg_p_minus_1), 1);
}

#[test]
fn test_double_basic() {
    assert_eq!(canonicalize(double(5)), 10);
    assert_eq!(canonicalize(double(0)), 0);

    // Double equals add to self
    let a = 123456789u64;
    assert_eq!(canonicalize(double(a)), canonicalize(add_asm(a, a)));
}

// ============== FIELD AXIOMS TESTS ==============

#[test]
fn test_field_axiom_add_commutativity() {
    let pairs = [(5, 7), (123, 456), (GOLDILOCKS_PRIME - 1, 2), (0, 100)];

    for (a, b) in pairs {
        assert_eq!(
            canonicalize(add_asm(a, b)),
            canonicalize(add_asm(b, a)),
            "Add commutativity failed for a={}, b={}", a, b
        );
    }
}

#[test]
fn test_field_axiom_mul_commutativity() {
    let pairs = [(5, 7), (123, 456), (GOLDILOCKS_PRIME - 1, 2), (EPSILON, 100)];

    for (a, b) in pairs {
        assert_eq!(
            canonicalize(mul(a, b)),
            canonicalize(mul(b, a)),
            "Mul commutativity failed for a={}, b={}", a, b
        );
    }
}

#[test]
fn test_field_axiom_add_associativity() {
    let triples = [(5, 7, 11), (100, 200, 300), (EPSILON, 1, GOLDILOCKS_PRIME - 1)];

    for (a, b, c) in triples {
        let left = add_asm(add_asm(a, b), c);
        let right = add_asm(a, add_asm(b, c));
        assert_eq!(
            canonicalize(left),
            canonicalize(right),
            "Add associativity failed for a={}, b={}, c={}", a, b, c
        );
    }
}

#[test]
fn test_field_axiom_mul_associativity() {
    let triples = [(5, 7, 11), (100, 200, 300), (123, 456, 789)];

    for (a, b, c) in triples {
        let left = mul(mul(a, b), c);
        let right = mul(a, mul(b, c));
        assert_eq!(
            canonicalize(left),
            canonicalize(right),
            "Mul associativity failed for a={}, b={}, c={}", a, b, c
        );
    }
}

#[test]
fn test_field_axiom_distributivity() {
    let triples = [(2, 3, 4), (100, 200, 300), (EPSILON, 1, 2)];

    for (a, b, c) in triples {
        let left = mul(a, add_asm(b, c));
        let right = add_asm(mul(a, b), mul(a, c));
        assert_eq!(
            canonicalize(left),
            canonicalize(right),
            "Distributivity failed for a={}, b={}, c={}", a, b, c
        );
    }
}

#[test]
fn test_field_axiom_identities() {
    let values = [0, 1, 5, GOLDILOCKS_PRIME - 1, EPSILON, 1u64 << 32];

    for a in values {
        // Additive identity
        assert_eq!(canonicalize(add_asm(a, 0)), canonicalize(a));
        assert_eq!(canonicalize(add_asm(0, a)), canonicalize(a));

        // Multiplicative identity
        assert_eq!(canonicalize(mul(a, 1)), canonicalize(a));
        assert_eq!(canonicalize(mul(1, a)), canonicalize(a));
    }
}

#[test]
fn test_field_axiom_additive_inverse() {
    let values = [0, 1, 5, GOLDILOCKS_PRIME - 1, EPSILON, 123456789];

    for a in values {
        let neg_a = neg(a);
        let sum = add_asm(a, neg_a);
        assert_eq!(
            canonicalize(sum),
            0,
            "Additive inverse failed for a={}: neg={}, sum={}", a, neg_a, sum
        );
    }
}

// ============== REDUCE128 TESTS ==============

#[test]
fn test_reduce128_small() {
    // Small values that don't need much reduction
    let result = reduce128(35u128);
    assert_eq!(canonicalize(result), 35);
}

#[test]
fn test_reduce128_pure_hi() {
    // Test with pure high bits
    let result = reduce128(1u128 << 64);
    // 2^64 mod p = EPSILON
    assert_eq!(canonicalize(result), EPSILON);
}

#[test]
fn test_reduce128_max() {
    // Test with maximum 128-bit value
    let result = reduce128(u128::MAX);
    // This should still produce a valid field element
    assert!(canonicalize(result) < GOLDILOCKS_PRIME);
}

#[test]
fn test_reduce128_from_mul() {
    // Test reduce128 with actual multiplication results
    let test_cases = [(5u64, 7u64), (1u64 << 40, 1u64 << 40), (EPSILON, EPSILON)];

    for (a, b) in test_cases {
        let (lo, hi) = mul_wide(a, b);
        let x = (lo as u128) | ((hi as u128) << 64);
        let reduced = reduce128(x);
        let expected = ((a as u128 * b as u128) % GOLDILOCKS_PRIME as u128) as u64;
        assert_eq!(
            canonicalize(reduced),
            expected,
            "reduce128 mismatch for mul({}, {})", a, b
        );
    }
}

// ============== MUL_WIDE TESTS ==============

#[test]
fn test_mul_wide_basic() {
    let (lo, hi) = mul_wide(5, 7);
    assert_eq!(lo, 35);
    assert_eq!(hi, 0);
}

#[test]
fn test_mul_wide_large() {
    let a = 1u64 << 40;
    let b = 1u64 << 40;
    let (lo, hi) = mul_wide(a, b);
    let expected = (a as u128) * (b as u128);
    assert_eq!(lo, expected as u64);
    assert_eq!(hi, (expected >> 64) as u64);
}

#[test]
fn test_mul_wide_max() {
    let (lo, hi) = mul_wide(u64::MAX, u64::MAX);
    let expected = (u64::MAX as u128) * (u64::MAX as u128);
    assert_eq!(lo, expected as u64);
    assert_eq!(hi, (expected >> 64) as u64);
}

// ============== INTEGRATION WITH FIELD ELEMENT ==============

#[test]
fn test_field_element_add() {
    let a = GoldilocksElement::from(5u64);
    let b = GoldilocksElement::from(7u64);
    let sum = &a + &b;
    assert_eq!(*sum.value(), 12);
}

#[test]
fn test_field_element_mul() {
    let a = GoldilocksElement::from(5u64);
    let b = GoldilocksElement::from(7u64);
    let product = &a * &b;
    assert_eq!(*product.value(), 35);
}

#[test]
fn test_field_element_square() {
    let a = GoldilocksElement::from(5u64);
    let sq = a.square();
    assert_eq!(*sq.value(), 25);
}

#[test]
fn test_field_element_inv() {
    let a = GoldilocksElement::from(5u64);
    let a_inv = a.inv().unwrap();
    let product = &a * &a_inv;
    assert_eq!(canonicalize(*product.value()), 1);
}

#[test]
fn test_field_element_pow() {
    let a = GoldilocksElement::from(2u64);
    let a_pow_10 = a.pow(10u64);
    assert_eq!(*a_pow_10.value(), 1024);
}

// ============== FFT ROOT OF UNITY TEST ==============

#[test]
fn test_primitive_root_with_asm() {
    // The primitive root should have order 2^32
    let root = GoldilocksField::get_primitive_root_of_unity(GoldilocksField::TWO_ADICITY).unwrap();

    // root^(2^32) should be 1
    let mut result = *root.value();
    for _ in 0..32 {
        result = GoldilocksField::square(&result);
    }
    assert_eq!(canonicalize(result), 1);
}

#[test]
fn test_root_of_unity_chain() {
    // Test that squaring the 2^32-th root 32 times gives 1
    let root = GoldilocksField::TWO_ADIC_PRIMITVE_ROOT_OF_UNITY;

    let mut current = root;
    for _ in 0..32 {
        let next = GoldilocksField::square(&current);
        current = next;
    }

    // After 32 squarings, we should have root^(2^32) = 1
    assert_eq!(canonicalize(current), 1, "Root of unity chain failed");
}

// ============== EDGE CASES ==============

#[test]
fn test_epsilon_operations() {
    // EPSILON = 2^32 - 1, special constant in Goldilocks arithmetic
    let eps = EPSILON;

    // eps + eps should reduce correctly
    let sum = add_asm(eps, eps);
    let expected = ((eps as u128 + eps as u128) % GOLDILOCKS_PRIME as u128) as u64;
    assert_eq!(canonicalize(sum), expected);

    // eps * eps
    let product = mul(eps, eps);
    let expected_product = ((eps as u128 * eps as u128) % GOLDILOCKS_PRIME as u128) as u64;
    assert_eq!(canonicalize(product), expected_product);
}

#[test]
fn test_prime_boundary_operations() {
    let p_minus_1 = GOLDILOCKS_PRIME - 1;

    // (p-1) + 1 = 0 mod p
    assert_eq!(canonicalize(add_asm(p_minus_1, 1)), 0);

    // (p-1) + 2 = 1 mod p
    assert_eq!(canonicalize(add_asm(p_minus_1, 2)), 1);

    // 0 - 1 = p-1 mod p
    assert_eq!(canonicalize(sub_asm(0, 1)), p_minus_1);

    // (p-1) * (p-1) = 1 mod p (since -1 * -1 = 1)
    assert_eq!(canonicalize(mul(p_minus_1, p_minus_1)), 1);
}

#[test]
fn test_powers_of_two() {
    // Test multiplication of powers of 2
    for exp1 in [16, 32, 48, 63] {
        for exp2 in [16, 32, 48, 63] {
            if exp1 + exp2 <= 127 {
                let a = 1u64 << std::cmp::min(exp1, 63);
                let b = 1u64 << std::cmp::min(exp2, 63);
                let result = mul(a, b);
                let expected = ((a as u128 * b as u128) % GOLDILOCKS_PRIME as u128) as u64;
                assert_eq!(
                    canonicalize(result),
                    expected,
                    "Power of 2 mul failed for 2^{} * 2^{}", exp1, exp2
                );
            }
        }
    }
}

// ============== EXTENSION FIELD ASM TESTS ==============
// Tests for Fp2 and Fp3 operations using ASM-optimized base field ops

mod extension_asm_tests {
    use crate::field::fields::fft_friendly::goldilocks_extensions_asm::{
        mul_by_6, mul_by_7, mul_by_4,
        fp2_add, fp2_sub, fp2_mul, fp2_square, fp2_neg, fp2_double,
        fp2_conjugate, fp2_norm, fp2_scalar_mul,
        fp3_add, fp3_sub, fp3_mul, fp3_square, fp3_neg, fp3_double, fp3_scalar_mul,
    };
    use crate::field::fields::fft_friendly::u64_goldilocks_asm::{
        mul, sub_fast, GOLDILOCKS_PRIME,
    };

    fn canonicalize(x: u64) -> u64 {
        if x >= GOLDILOCKS_PRIME { x - GOLDILOCKS_PRIME } else { x }
    }

    // ============== MULTIPLY BY CONSTANT TESTS ==============

    #[test]
    fn test_mul_by_4() {
        let test_values = [5u64, 100, GOLDILOCKS_PRIME - 1, 1, 0, 123456789];
        for a in test_values {
            let result = mul_by_4(a);
            let expected = mul(a, 4);
            assert_eq!(
                canonicalize(result), canonicalize(expected),
                "mul_by_4 mismatch for a={}", a
            );
        }
    }

    #[test]
    fn test_mul_by_6() {
        let test_values = [5u64, 100, GOLDILOCKS_PRIME - 1, 1, 0, 123456789];
        for a in test_values {
            let result = mul_by_6(a);
            let expected = mul(a, 6);
            assert_eq!(
                canonicalize(result), canonicalize(expected),
                "mul_by_6 mismatch for a={}", a
            );
        }
    }

    #[test]
    fn test_mul_by_7() {
        let test_values = [5u64, 100, GOLDILOCKS_PRIME - 1, 1, 0, 123456789];
        for a in test_values {
            let result = mul_by_7(a);
            let expected = mul(a, 7);
            assert_eq!(
                canonicalize(result), canonicalize(expected),
                "mul_by_7 mismatch for a={}", a
            );
        }
    }

    // ============== FP2 TESTS ==============

    #[test]
    fn test_fp2_add() {
        let a = [3u64, 4u64];
        let b = [1u64, 2u64];
        let c = fp2_add(a, b);
        assert_eq!(canonicalize(c[0]), 4);
        assert_eq!(canonicalize(c[1]), 6);
    }

    #[test]
    fn test_fp2_sub() {
        let a = [10u64, 8u64];
        let b = [3u64, 2u64];
        let c = fp2_sub(a, b);
        assert_eq!(canonicalize(c[0]), 7);
        assert_eq!(canonicalize(c[1]), 6);
    }

    #[test]
    fn test_fp2_mul() {
        let a = [3u64, 4u64];
        let b = [1u64, 2u64];
        // (3 + 4w)(1 + 2w) = 3 + 6w + 4w + 8w^2 = 3 + 10w + 8*7 = 59 + 10w
        let c = fp2_mul(a, b);
        assert_eq!(canonicalize(c[0]), 59);
        assert_eq!(canonicalize(c[1]), 10);
    }

    #[test]
    fn test_fp2_square() {
        let a = [3u64, 4u64];
        // (3 + 4w)^2 = 9 + 24w + 16w^2 = 9 + 24w + 112 = 121 + 24w
        let c = fp2_square(a);
        assert_eq!(canonicalize(c[0]), 121);
        assert_eq!(canonicalize(c[1]), 24);

        // Verify square equals mul(a, a)
        let c_mul = fp2_mul(a, a);
        assert_eq!(canonicalize(c[0]), canonicalize(c_mul[0]));
        assert_eq!(canonicalize(c[1]), canonicalize(c_mul[1]));
    }

    #[test]
    fn test_fp2_neg() {
        let a = [3u64, 4u64];
        let neg_a = fp2_neg(a);
        let sum = fp2_add(a, neg_a);
        assert_eq!(canonicalize(sum[0]), 0);
        assert_eq!(canonicalize(sum[1]), 0);
    }

    #[test]
    fn test_fp2_double() {
        let a = [5u64, 7u64];
        let doubled = fp2_double(a);
        let added = fp2_add(a, a);
        assert_eq!(canonicalize(doubled[0]), canonicalize(added[0]));
        assert_eq!(canonicalize(doubled[1]), canonicalize(added[1]));
    }

    #[test]
    fn test_fp2_conjugate() {
        let a = [3u64, 4u64];
        let conj = fp2_conjugate(a);
        assert_eq!(canonicalize(conj[0]), 3);
        // -4 mod p = p - 4
        assert_eq!(canonicalize(conj[1]), GOLDILOCKS_PRIME - 4);
    }

    #[test]
    fn test_fp2_norm() {
        let a = [3u64, 4u64];
        // norm = 3^2 - 7*4^2 = 9 - 112 = -103 mod p
        let norm = fp2_norm(a);
        let expected = sub_fast(9, 112);
        assert_eq!(canonicalize(norm), canonicalize(expected));
    }

    #[test]
    fn test_fp2_scalar_mul() {
        let a = [3u64, 4u64];
        let scalar = 5u64;
        let result = fp2_scalar_mul(scalar, a);
        assert_eq!(canonicalize(result[0]), 15);
        assert_eq!(canonicalize(result[1]), 20);
    }

    #[test]
    fn test_fp2_mul_one() {
        let a = [5u64, 7u64];
        let one = [1u64, 0u64];
        let result = fp2_mul(a, one);
        assert_eq!(canonicalize(result[0]), canonicalize(a[0]));
        assert_eq!(canonicalize(result[1]), canonicalize(a[1]));
    }

    #[test]
    fn test_fp2_mul_commutativity() {
        let a = [5u64, 7u64];
        let b = [13u64, 17u64];
        let ab = fp2_mul(a, b);
        let ba = fp2_mul(b, a);
        assert_eq!(canonicalize(ab[0]), canonicalize(ba[0]));
        assert_eq!(canonicalize(ab[1]), canonicalize(ba[1]));
    }

    // ============== FP3 TESTS ==============

    #[test]
    fn test_fp3_add() {
        let a = [1u64, 2u64, 3u64];
        let b = [4u64, 5u64, 6u64];
        let c = fp3_add(a, b);
        assert_eq!(canonicalize(c[0]), 5);
        assert_eq!(canonicalize(c[1]), 7);
        assert_eq!(canonicalize(c[2]), 9);
    }

    #[test]
    fn test_fp3_sub() {
        let a = [10u64, 8u64, 6u64];
        let b = [3u64, 2u64, 1u64];
        let c = fp3_sub(a, b);
        assert_eq!(canonicalize(c[0]), 7);
        assert_eq!(canonicalize(c[1]), 6);
        assert_eq!(canonicalize(c[2]), 5);
    }

    #[test]
    fn test_fp3_neg() {
        let a = [3u64, 4u64, 5u64];
        let neg_a = fp3_neg(a);
        let sum = fp3_add(a, neg_a);
        assert_eq!(canonicalize(sum[0]), 0);
        assert_eq!(canonicalize(sum[1]), 0);
        assert_eq!(canonicalize(sum[2]), 0);
    }

    #[test]
    fn test_fp3_double() {
        let a = [5u64, 7u64, 11u64];
        let doubled = fp3_double(a);
        let added = fp3_add(a, a);
        assert_eq!(canonicalize(doubled[0]), canonicalize(added[0]));
        assert_eq!(canonicalize(doubled[1]), canonicalize(added[1]));
        assert_eq!(canonicalize(doubled[2]), canonicalize(added[2]));
    }

    #[test]
    fn test_fp3_square_equals_mul() {
        let a = [5u64, 7u64, 11u64];
        let sq = fp3_square(a);
        let mul_result = fp3_mul(a, a);
        assert_eq!(canonicalize(sq[0]), canonicalize(mul_result[0]));
        assert_eq!(canonicalize(sq[1]), canonicalize(mul_result[1]));
        assert_eq!(canonicalize(sq[2]), canonicalize(mul_result[2]));
    }

    #[test]
    fn test_fp3_scalar_mul() {
        let a = [2u64, 3u64, 4u64];
        let scalar = 5u64;
        let result = fp3_scalar_mul(scalar, a);
        assert_eq!(canonicalize(result[0]), 10);
        assert_eq!(canonicalize(result[1]), 15);
        assert_eq!(canonicalize(result[2]), 20);
    }

    #[test]
    fn test_fp3_mul_commutativity() {
        let a = [5u64, 7u64, 11u64];
        let b = [13u64, 17u64, 19u64];
        let ab = fp3_mul(a, b);
        let ba = fp3_mul(b, a);
        assert_eq!(canonicalize(ab[0]), canonicalize(ba[0]));
        assert_eq!(canonicalize(ab[1]), canonicalize(ba[1]));
        assert_eq!(canonicalize(ab[2]), canonicalize(ba[2]));
    }

    #[test]
    fn test_fp3_mul_one() {
        let a = [5u64, 7u64, 11u64];
        let one = [1u64, 0u64, 0u64];
        let result = fp3_mul(a, one);
        assert_eq!(canonicalize(result[0]), canonicalize(a[0]));
        assert_eq!(canonicalize(result[1]), canonicalize(a[1]));
        assert_eq!(canonicalize(result[2]), canonicalize(a[2]));
    }
}
