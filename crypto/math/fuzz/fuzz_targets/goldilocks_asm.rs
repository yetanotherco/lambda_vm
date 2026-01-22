//! Differential fuzzing for ARM64 assembly Goldilocks field operations.
//!
//! This fuzzer compares ARM64 assembly implementations against native Rust
//! implementations to ensure correctness across a wide range of inputs.
//!
//! Run with: cargo +nightly fuzz run goldilocks_asm

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
use math::field::fields::fft_friendly::u64_goldilocks_asm;

#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
use math::field::fields::fft_friendly::goldilocks_extensions_asm;

use math::field::fields::fft_friendly::u64_goldilocks::GOLDILOCKS_PRIME;

/// EPSILON = 2^32 - 1
const EPSILON: u64 = 0xFFFF_FFFF;

/// Input for fuzzing operations
#[derive(Debug, Arbitrary)]
struct FuzzInput {
    op_type: u8, // Operation type selector
    a: u64,
    b: u64,
    c: u64,
}

/// Canonicalize a value to [0, p)
#[inline]
fn canonicalize(x: u64) -> u64 {
    if x >= GOLDILOCKS_PRIME {
        x - GOLDILOCKS_PRIME
    } else {
        x
    }
}

// Native reference implementations
fn native_mul(a: u64, b: u64) -> u64 {
    let product = (a as u128) * (b as u128);
    native_reduce128(product)
}

fn native_reduce128(x: u128) -> u64 {
    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & EPSILON;

    let (t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    let t0 = if borrow { t0.wrapping_sub(EPSILON) } else { t0 };

    let t1 = x_hi_lo.wrapping_mul(EPSILON);

    let (result, carry) = t0.overflowing_add(t1);
    if carry {
        result.wrapping_add(EPSILON)
    } else {
        result
    }
}

fn native_add(a: u64, b: u64) -> u64 {
    let (sum, over) = a.overflowing_add(b);
    let (sum, over2) = sum.overflowing_add((over as u64) * EPSILON);
    if over2 {
        sum.wrapping_add(EPSILON)
    } else {
        sum
    }
}

fn native_sub(a: u64, b: u64) -> u64 {
    let (diff, under) = a.overflowing_sub(b);
    let (diff, under2) = diff.overflowing_sub((under as u64) * EPSILON);
    if under2 {
        diff.wrapping_sub(EPSILON)
    } else {
        diff
    }
}

fn native_double(a: u64) -> u64 {
    native_add(a, a)
}

// Native Fp2 implementations
fn native_fp2_add(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    [native_add(a[0], b[0]), native_add(a[1], b[1])]
}

fn native_fp2_mul(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    let a0b0 = native_mul(a[0], b[0]);
    let a1b1 = native_mul(a[1], b[1]);
    let a0b1 = native_mul(a[0], b[1]);
    let a1b0 = native_mul(a[1], b[0]);

    // 7 * a1b1
    let a1b1_2 = native_double(a1b1);
    let a1b1_4 = native_double(a1b1_2);
    let temp = native_add(a1b1, a1b1_2);
    let w_a1b1 = native_add(temp, a1b1_4);

    [native_add(a0b0, w_a1b1), native_add(a0b1, a1b0)]
}

fn native_fp2_square(a: [u64; 2]) -> [u64; 2] {
    let a0_sq = native_mul(a[0], a[0]);
    let a1_sq = native_mul(a[1], a[1]);
    let a0a1 = native_mul(a[0], a[1]);

    // 7 * a1_sq
    let a1_sq_2 = native_double(a1_sq);
    let a1_sq_4 = native_double(a1_sq_2);
    let temp = native_add(a1_sq, a1_sq_2);
    let w_a1_sq = native_add(temp, a1_sq_4);

    [native_add(a0_sq, w_a1_sq), native_double(a0a1)]
}

// Native Fp3 implementations
fn native_fp3_add(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
    [
        native_add(a[0], b[0]),
        native_add(a[1], b[1]),
        native_add(a[2], b[2]),
    ]
}

fn native_fp3_mul(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
    let v0 = native_mul(a[0], b[0]);
    let v1 = native_mul(a[1], b[1]);
    let v2 = native_mul(a[2], b[2]);

    let a1_plus_a2 = native_add(a[1], a[2]);
    let b1_plus_b2 = native_add(b[1], b[2]);
    let temp0 = native_mul(a1_plus_a2, b1_plus_b2);
    let temp0 = native_sub(temp0, v1);
    let t0 = native_sub(temp0, v2);

    let a0_plus_a1 = native_add(a[0], a[1]);
    let b0_plus_b1 = native_add(b[0], b[1]);
    let temp1 = native_mul(a0_plus_a1, b0_plus_b1);
    let temp1 = native_sub(temp1, v0);
    let t1 = native_sub(temp1, v1);

    let a0_plus_a2 = native_add(a[0], a[2]);
    let b0_plus_b2 = native_add(b[0], b[2]);
    let temp2 = native_mul(a0_plus_a2, b0_plus_b2);
    let temp2 = native_sub(temp2, v0);
    let t2 = native_sub(temp2, v2);

    [
        native_add(v0, native_double(t0)),
        native_add(t1, native_double(v2)),
        native_add(t2, v1),
    ]
}

fn native_fp3_square(a: [u64; 3]) -> [u64; 3] {
    let s0 = native_mul(a[0], a[0]);
    let s1 = native_mul(a[1], a[1]);
    let s2 = native_mul(a[2], a[2]);
    let a01 = native_mul(a[0], a[1]);
    let a02 = native_mul(a[0], a[2]);
    let a12 = native_mul(a[1], a[2]);

    let a12_2 = native_double(a12);
    let a12_4 = native_double(a12_2);
    let c0 = native_add(s0, a12_4);

    let c1 = native_add(native_double(a01), native_double(s2));
    let c2 = native_add(native_double(a02), s1);

    [c0, c1, c2]
}

#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
fuzz_target!(|input: FuzzInput| {
    // Reduce inputs to canonical range
    let a = input.a % GOLDILOCKS_PRIME;
    let b = input.b % GOLDILOCKS_PRIME;
    let c = input.c % GOLDILOCKS_PRIME;

    // Test different operations based on op_type
    match input.op_type % 10 {
        // Base field operations
        0 => {
            // Test add
            let asm_result = u64_goldilocks_asm::add_fast(a, b);
            let native_result = native_add(a, b);
            assert_eq!(
                canonicalize(asm_result),
                canonicalize(native_result),
                "ADD mismatch: a={}, b={}, asm={}, native={}",
                a,
                b,
                asm_result,
                native_result
            );
        }
        1 => {
            // Test sub
            let asm_result = u64_goldilocks_asm::sub_fast(a, b);
            let native_result = native_sub(a, b);
            assert_eq!(
                canonicalize(asm_result),
                canonicalize(native_result),
                "SUB mismatch: a={}, b={}, asm={}, native={}",
                a,
                b,
                asm_result,
                native_result
            );
        }
        2 => {
            // Test mul
            let asm_result = u64_goldilocks_asm::mul(a, b);
            let native_result = native_mul(a, b);
            assert_eq!(
                canonicalize(asm_result),
                canonicalize(native_result),
                "MUL mismatch: a={}, b={}, asm={}, native={}",
                a,
                b,
                asm_result,
                native_result
            );
        }
        3 => {
            // Test reduce128
            let x = (a as u128) * (b as u128);
            let asm_result = u64_goldilocks_asm::reduce128(x);
            let native_result = native_reduce128(x);
            assert_eq!(
                canonicalize(asm_result),
                canonicalize(native_result),
                "REDUCE128 mismatch"
            );
        }
        // Fp2 operations
        4 => {
            let a_fp2 = [a, b];
            let b_fp2 = [c, (c.wrapping_add(1)) % GOLDILOCKS_PRIME];
            let asm_result = goldilocks_extensions_asm::fp2_add(a_fp2, b_fp2);
            let native_result = native_fp2_add(a_fp2, b_fp2);
            assert_eq!(
                [canonicalize(asm_result[0]), canonicalize(asm_result[1])],
                [
                    canonicalize(native_result[0]),
                    canonicalize(native_result[1])
                ],
                "FP2_ADD mismatch"
            );
        }
        5 => {
            let a_fp2 = [a, b];
            let b_fp2 = [c, (c.wrapping_add(1)) % GOLDILOCKS_PRIME];
            let asm_result = goldilocks_extensions_asm::fp2_mul(a_fp2, b_fp2);
            let native_result = native_fp2_mul(a_fp2, b_fp2);
            assert_eq!(
                [canonicalize(asm_result[0]), canonicalize(asm_result[1])],
                [
                    canonicalize(native_result[0]),
                    canonicalize(native_result[1])
                ],
                "FP2_MUL mismatch"
            );
        }
        6 => {
            let a_fp2 = [a, b];
            let asm_result = goldilocks_extensions_asm::fp2_square(a_fp2);
            let native_result = native_fp2_square(a_fp2);
            assert_eq!(
                [canonicalize(asm_result[0]), canonicalize(asm_result[1])],
                [
                    canonicalize(native_result[0]),
                    canonicalize(native_result[1])
                ],
                "FP2_SQUARE mismatch"
            );
        }
        // Fp3 operations
        7 => {
            let a_fp3 = [a, b, c];
            let b_fp3 = [
                (a.wrapping_add(1)) % GOLDILOCKS_PRIME,
                (b.wrapping_add(2)) % GOLDILOCKS_PRIME,
                (c.wrapping_add(3)) % GOLDILOCKS_PRIME,
            ];
            let asm_result = goldilocks_extensions_asm::fp3_add(a_fp3, b_fp3);
            let native_result = native_fp3_add(a_fp3, b_fp3);
            assert_eq!(
                [
                    canonicalize(asm_result[0]),
                    canonicalize(asm_result[1]),
                    canonicalize(asm_result[2])
                ],
                [
                    canonicalize(native_result[0]),
                    canonicalize(native_result[1]),
                    canonicalize(native_result[2])
                ],
                "FP3_ADD mismatch"
            );
        }
        8 => {
            let a_fp3 = [a, b, c];
            let b_fp3 = [
                (a.wrapping_add(1)) % GOLDILOCKS_PRIME,
                (b.wrapping_add(2)) % GOLDILOCKS_PRIME,
                (c.wrapping_add(3)) % GOLDILOCKS_PRIME,
            ];
            let asm_result = goldilocks_extensions_asm::fp3_mul(a_fp3, b_fp3);
            let native_result = native_fp3_mul(a_fp3, b_fp3);
            assert_eq!(
                [
                    canonicalize(asm_result[0]),
                    canonicalize(asm_result[1]),
                    canonicalize(asm_result[2])
                ],
                [
                    canonicalize(native_result[0]),
                    canonicalize(native_result[1]),
                    canonicalize(native_result[2])
                ],
                "FP3_MUL mismatch"
            );
        }
        9 => {
            let a_fp3 = [a, b, c];
            let asm_result = goldilocks_extensions_asm::fp3_square(a_fp3);
            let native_result = native_fp3_square(a_fp3);
            assert_eq!(
                [
                    canonicalize(asm_result[0]),
                    canonicalize(asm_result[1]),
                    canonicalize(asm_result[2])
                ],
                [
                    canonicalize(native_result[0]),
                    canonicalize(native_result[1]),
                    canonicalize(native_result[2])
                ],
                "FP3_SQUARE mismatch"
            );
        }
        _ => unreachable!(),
    }
});

#[cfg(not(all(feature = "asm-arm64", target_arch = "aarch64")))]
fuzz_target!(|_input: FuzzInput| {
    // ASM not available on this platform
});
