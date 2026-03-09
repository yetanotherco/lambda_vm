//! Fuzzer for native Goldilocks field operations against a reference implementation.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsField;

/// Input for fuzzing field operations
#[derive(Debug, Arbitrary)]
struct FuzzInput {
    a: u64,
    b: u64,
}

/// Goldilocks prime constant for canonicalization
const GOLDILOCKS_PRIME: u64 = 0xFFFF_FFFF_0000_0001;

/// Canonicalize a u64 to [0, p) for comparison
#[inline]
fn canonicalize(x: u64) -> u64 {
    if x >= GOLDILOCKS_PRIME {
        x - GOLDILOCKS_PRIME
    } else {
        x
    }
}

#[inline]
fn mod_add(a: u64, b: u64) -> u64 {
    ((a as u128 + b as u128) % GOLDILOCKS_PRIME as u128) as u64
}

#[inline]
fn mod_sub(a: u64, b: u64) -> u64 {
    ((a as u128 + GOLDILOCKS_PRIME as u128 - b as u128) % GOLDILOCKS_PRIME as u128) as u64
}

#[inline]
fn mod_mul(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % GOLDILOCKS_PRIME as u128) as u64
}

#[inline]
fn mod_neg(a: u64) -> u64 {
    if a == 0 { 0 } else { GOLDILOCKS_PRIME - a }
}

fn mod_pow(mut base: u64, mut exp: u64) -> u64 {
    let mut result = 1u64;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base);
        }
        base = mod_mul(base, base);
        exp >>= 1;
    }
    result
}

#[inline]
fn mod_inv(a: u64) -> u64 {
    mod_pow(a, GOLDILOCKS_PRIME - 2)
}

fuzz_target!(|input: FuzzInput| {
    // Reduce inputs to canonical range for fair comparison
    let a_native = input.a % GOLDILOCKS_PRIME;
    let b_native = input.b % GOLDILOCKS_PRIME;

    // Test addition
    let sum_native = GoldilocksField::add(&a_native, &b_native);
    assert_eq!(
        canonicalize(sum_native),
        mod_add(a_native, b_native),
        "Addition mismatch: native={}, ref={}, a={}, b={}",
        canonicalize(sum_native),
        mod_add(a_native, b_native),
        a_native,
        b_native
    );

    // Test subtraction
    let diff_native = GoldilocksField::sub(&a_native, &b_native);
    assert_eq!(
        canonicalize(diff_native),
        mod_sub(a_native, b_native),
        "Subtraction mismatch: native={}, ref={}, a={}, b={}",
        canonicalize(diff_native),
        mod_sub(a_native, b_native),
        a_native,
        b_native
    );

    // Test multiplication
    let prod_native = GoldilocksField::mul(&a_native, &b_native);
    assert_eq!(
        canonicalize(prod_native),
        mod_mul(a_native, b_native),
        "Multiplication mismatch: native={}, ref={}, a={}, b={}",
        canonicalize(prod_native),
        mod_mul(a_native, b_native),
        a_native,
        b_native
    );

    // Test squaring
    let sq_native = GoldilocksField::square(&a_native);
    assert_eq!(
        canonicalize(sq_native),
        mod_mul(a_native, a_native),
        "Squaring mismatch: native={}, ref={}, a={}",
        canonicalize(sq_native),
        mod_mul(a_native, a_native),
        a_native
    );

    // Test negation
    let neg_native = GoldilocksField::neg(&a_native);
    assert_eq!(
        canonicalize(neg_native),
        mod_neg(a_native),
        "Negation mismatch: native={}, ref={}, a={}",
        canonicalize(neg_native),
        mod_neg(a_native),
        a_native
    );

    // Test inversion (skip zero)
    if a_native != 0 {
        let inv_native = GoldilocksField::inv(&a_native).unwrap();
        assert_eq!(
            canonicalize(inv_native),
            mod_inv(a_native),
            "Inversion mismatch: native={}, ref={}, a={}",
            canonicalize(inv_native),
            mod_inv(a_native),
            a_native
        );

        // Verify a * a^-1 = 1 for native implementation
        let should_be_one = GoldilocksField::mul(&a_native, &inv_native);
        assert_eq!(
            canonicalize(should_be_one),
            1,
            "Native inversion verification failed: a * a^-1 = {}, expected 1, a={}",
            canonicalize(should_be_one),
            a_native
        );
    }

    // Test division (skip if b is zero)
    if b_native != 0 {
        let div_native = GoldilocksField::div(&a_native, &b_native).unwrap();
        assert_eq!(
            canonicalize(div_native),
            mod_mul(a_native, mod_inv(b_native)),
            "Division mismatch: native={}, ref={}, a={}, b={}",
            canonicalize(div_native),
            mod_mul(a_native, mod_inv(b_native)),
            a_native,
            b_native
        );
    }

    // Test doubling
    let double_native = GoldilocksField::double(&a_native);
    assert_eq!(
        canonicalize(double_native),
        mod_add(a_native, a_native),
        "Doubling mismatch: native={}, ref={}, a={}",
        canonicalize(double_native),
        mod_add(a_native, a_native),
        a_native
    );
});
