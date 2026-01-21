//! Differential fuzzer for Goldilocks field implementations.
//!
//! Compares:
//! - Native Goldilocks (u64_goldilocks_native) operations
//! - Montgomery Goldilocks (u64_goldilocks) operations as reference
//!
//! With asm-arm64 feature enabled, this also validates that ASM
//! implementations produce identical results to native Rust.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use math::field::{
    element::FieldElement,
    fields::fft_friendly::{
        u64_goldilocks::U64GoldilocksPrimeField, u64_goldilocks_native::GoldilocksField,
    },
    traits::IsField,
};

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

fuzz_target!(|input: FuzzInput| {
    // Reduce inputs to canonical range for fair comparison
    let a_native = input.a % GOLDILOCKS_PRIME;
    let b_native = input.b % GOLDILOCKS_PRIME;

    // Create Montgomery field elements from canonical values
    let a_mont = FieldElement::<U64GoldilocksPrimeField>::from(a_native);
    let b_mont = FieldElement::<U64GoldilocksPrimeField>::from(b_native);

    // Test addition
    let sum_native = GoldilocksField::add(&a_native, &b_native);
    let sum_mont = &a_mont + &b_mont;
    let sum_mont_canonical = sum_mont.representative().limbs[0];
    assert_eq!(
        canonicalize(sum_native),
        sum_mont_canonical,
        "Addition mismatch: native={}, mont={}, a={}, b={}",
        canonicalize(sum_native),
        sum_mont_canonical,
        a_native,
        b_native
    );

    // Test subtraction
    let diff_native = GoldilocksField::sub(&a_native, &b_native);
    let diff_mont = &a_mont - &b_mont;
    let diff_mont_canonical = diff_mont.representative().limbs[0];
    assert_eq!(
        canonicalize(diff_native),
        diff_mont_canonical,
        "Subtraction mismatch: native={}, mont={}, a={}, b={}",
        canonicalize(diff_native),
        diff_mont_canonical,
        a_native,
        b_native
    );

    // Test multiplication
    let prod_native = GoldilocksField::mul(&a_native, &b_native);
    let prod_mont = &a_mont * &b_mont;
    let prod_mont_canonical = prod_mont.representative().limbs[0];
    assert_eq!(
        canonicalize(prod_native),
        prod_mont_canonical,
        "Multiplication mismatch: native={}, mont={}, a={}, b={}",
        canonicalize(prod_native),
        prod_mont_canonical,
        a_native,
        b_native
    );

    // Test squaring
    let sq_native = GoldilocksField::square(&a_native);
    let sq_mont = &a_mont * &a_mont;
    let sq_mont_canonical = sq_mont.representative().limbs[0];
    assert_eq!(
        canonicalize(sq_native),
        sq_mont_canonical,
        "Squaring mismatch: native={}, mont={}, a={}",
        canonicalize(sq_native),
        sq_mont_canonical,
        a_native
    );

    // Test negation
    let neg_native = GoldilocksField::neg(&a_native);
    let neg_mont = -&a_mont;
    let neg_mont_canonical = neg_mont.representative().limbs[0];
    assert_eq!(
        canonicalize(neg_native),
        neg_mont_canonical,
        "Negation mismatch: native={}, mont={}, a={}",
        canonicalize(neg_native),
        neg_mont_canonical,
        a_native
    );

    // Test inversion (skip zero)
    if a_native != 0 {
        let inv_native = GoldilocksField::inv(&a_native).unwrap();
        let inv_mont = a_mont.inv().unwrap();
        let inv_mont_canonical = inv_mont.representative().limbs[0];
        assert_eq!(
            canonicalize(inv_native),
            inv_mont_canonical,
            "Inversion mismatch: native={}, mont={}, a={}",
            canonicalize(inv_native),
            inv_mont_canonical,
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
        let div_mont = (&a_mont / &b_mont).unwrap();
        let div_mont_canonical = div_mont.representative().limbs[0];
        assert_eq!(
            canonicalize(div_native),
            div_mont_canonical,
            "Division mismatch: native={}, mont={}, a={}, b={}",
            canonicalize(div_native),
            div_mont_canonical,
            a_native,
            b_native
        );
    }

    // Test doubling
    let double_native = GoldilocksField::double(&a_native);
    let double_mont = &a_mont + &a_mont;
    let double_mont_canonical = double_mont.representative().limbs[0];
    assert_eq!(
        canonicalize(double_native),
        double_mont_canonical,
        "Doubling mismatch: native={}, mont={}, a={}",
        canonicalize(double_native),
        double_mont_canonical,
        a_native
    );
});
