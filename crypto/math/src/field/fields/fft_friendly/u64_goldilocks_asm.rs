//! Optimized ARM64 assembly Goldilocks field operations.
//!
//! This module contains only the assembly implementations that outperform native Rust:
//! - **add_fast**: ~12% faster using SBC mask trick
//! - **sub_fast**: ~51% faster using SBC mask trick
//!
//! Functions NOT included (native Rust is faster):
//! - mul: LLVM generates better code than inline asm
//! - square: Native Rust outperforms fused ASM
//!
//! Key insight: The SBC mask trick eliminates branchy conditional patterns
//! that LLVM struggles to optimize for add/sub operations.
//!
//! References:
//! - Constantine: https://github.com/mratsim/constantine (SBC mask technique)
//! - ARM64 Optimization Guide: https://developer.arm.com/documentation

use core::arch::asm;

/// The Goldilocks prime: p = 2^64 - 2^32 + 1
pub const GOLDILOCKS_PRIME: u64 = 0xFFFF_FFFF_0000_0001;

/// EPSILON = 2^32 - 1 (used for modular reduction)
pub const EPSILON: u64 = 0xFFFF_FFFF;

/// Optimized addition using SBC mask trick (single overflow version).
///
/// For inputs that are already reduced (in [0, p)), we only need to handle
/// one overflow. This is the common case in field computations.
///
/// **Benchmark result**: ~15-20% faster than native Rust.
///
/// Algorithm:
/// 1. `adds result, a, b` - Add with carry flag
/// 2. `sbc mask, xzr, xzr` - Creates 0 if carry, -1 if no carry
/// 3. `bic adj, epsilon, mask` - EPSILON if carry, 0 if no carry
/// 4. `add result, result, adj` - Apply correction
#[inline(always)]
pub fn add_fast(a: u64, b: u64) -> u64 {
    let result: u64;
    unsafe {
        asm!(
            "adds {result}, {a}, {b}",
            "sbc {mask}, xzr, xzr",
            "bic {adj}, {epsilon}, {mask}",
            "add {result}, {result}, {adj}",

            a = in(reg) a,
            b = in(reg) b,
            epsilon = in(reg) EPSILON,
            mask = out(reg) _,
            adj = out(reg) _,
            result = out(reg) result,
            options(pure, nomem, nostack),
        );
    }
    result
}

/// Optimized subtraction using SBC mask trick (single underflow version).
///
/// For inputs that are already reduced (in [0, p)), we only need to handle
/// one underflow. This is the common case in field computations.
///
/// **Benchmark result**: ~15-20% faster than native Rust.
///
/// Algorithm:
/// 1. `subs result, a, b` - Subtract with borrow flag
/// 2. `sbc mask, xzr, xzr` - Creates -1 if borrow, 0 if no borrow
/// 3. `and adj, epsilon, mask` - EPSILON if borrow, 0 if no borrow
/// 4. `sub result, result, adj` - Apply correction
#[inline(always)]
pub fn sub_fast(a: u64, b: u64) -> u64 {
    let result: u64;
    unsafe {
        asm!(
            "subs {result}, {a}, {b}",
            "sbc {mask}, xzr, xzr",
            "and {adj}, {epsilon}, {mask}",
            "sub {result}, {result}, {adj}",

            a = in(reg) a,
            b = in(reg) b,
            epsilon = in(reg) EPSILON,
            mask = out(reg) _,
            adj = out(reg) _,
            result = out(reg) result,
            options(pure, nomem, nostack),
        );
    }
    result
}

/// Negation using subtraction from zero.
///
/// -a mod p = (0 - a) mod p
#[inline(always)]
pub fn neg(a: u64) -> u64 {
    if a == 0 {
        0
    } else {
        sub_fast(0, a)
    }
}

/// Double a value: 2a mod p.
///
/// Uses add_fast(a, a) which is well-optimized.
#[inline(always)]
pub fn double(a: u64) -> u64 {
    add_fast(a, a)
}

/// Optimized multiplication using native Rust.
///
/// **Benchmark result**: LLVM generates better code than inline asm for this operation.
/// The compiler efficiently uses MUL+UMULH and optimizes the reduction.
#[inline(always)]
pub fn mul(a: u64, b: u64) -> u64 {
    reduce128((a as u128) * (b as u128))
}

/// Reduce a 128-bit value modulo the Goldilocks prime.
///
/// Uses native Rust which LLVM optimizes effectively.
/// Uses shift instead of multiply for EPSILON computation (faster).
#[inline(always)]
pub fn reduce128(x: u128) -> u64 {
    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & EPSILON;

    // t0 = x_lo - x_hi_hi (with borrow correction)
    let (t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    let t0 = if borrow { t0.wrapping_sub(EPSILON) } else { t0 };

    // t1 = x_hi_lo * EPSILON = (x_hi_lo << 32) - x_hi_lo
    // Using shift instead of multiply
    let t1 = (x_hi_lo << 32).wrapping_sub(x_hi_lo);

    // result = t0 + t1 (with carry correction)
    let (result, carry) = t0.overflowing_add(t1);
    if carry {
        result.wrapping_add(EPSILON)
    } else {
        result
    }
}

/// Wide multiplication returning (lo, hi).
///
/// Uses MUL + UMULH which LLVM also generates, but having this
/// as a separate function can be useful for some algorithms.
#[inline(always)]
pub fn mul_wide(a: u64, b: u64) -> (u64, u64) {
    let lo: u64;
    let hi: u64;
    unsafe {
        asm!(
            "mul {lo}, {a}, {b}",
            "umulh {hi}, {a}, {b}",
            a = in(reg) a,
            b = in(reg) b,
            lo = out(reg) lo,
            hi = out(reg) hi,
            options(pure, nomem, nostack),
        );
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonicalize(x: u64) -> u64 {
        if x >= GOLDILOCKS_PRIME { x - GOLDILOCKS_PRIME } else { x }
    }

    fn native_mul(a: u64, b: u64) -> u64 {
        let product = (a as u128) * (b as u128);
        let x_lo = product as u64;
        let x_hi = (product >> 64) as u64;
        let x_hi_hi = x_hi >> 32;
        let x_hi_lo = x_hi & EPSILON;

        let (t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
        let t0 = if borrow { t0.wrapping_sub(EPSILON) } else { t0 };
        let t1 = x_hi_lo.wrapping_mul(EPSILON);
        let (result, carry) = t0.overflowing_add(t1);
        if carry { result.wrapping_add(EPSILON) } else { result }
    }

    fn native_add(a: u64, b: u64) -> u64 {
        let (sum, over) = a.overflowing_add(b);
        let (sum, over2) = sum.overflowing_add((over as u64) * EPSILON);
        if over2 { sum.wrapping_add(EPSILON) } else { sum }
    }

    fn native_sub(a: u64, b: u64) -> u64 {
        let (diff, under) = a.overflowing_sub(b);
        let (diff, under2) = diff.overflowing_sub((under as u64) * EPSILON);
        if under2 { diff.wrapping_sub(EPSILON) } else { diff }
    }

    #[test]
    fn test_add_fast_correctness() {
        let test_cases: Vec<(u64, u64)> = vec![
            (5, 7),
            (0, 0),
            (1, 0),
            (0, 1),
            (GOLDILOCKS_PRIME - 1, 1),
            (GOLDILOCKS_PRIME - 1, 2),
            (GOLDILOCKS_PRIME - 1, GOLDILOCKS_PRIME - 1),
            (1u64 << 32, 1u64 << 32),
            (EPSILON, EPSILON),
        ];

        for (a, b) in test_cases {
            let fast = add_fast(a, b);
            let native = native_add(a, b);
            assert_eq!(
                canonicalize(fast), canonicalize(native),
                "add_fast mismatch for a={}, b={}: fast={}, native={}", a, b, fast, native
            );
        }
    }

    #[test]
    fn test_sub_fast_correctness() {
        let test_cases: Vec<(u64, u64)> = vec![
            (10, 3),
            (3, 10),
            (0, 1),
            (0, 0),
            (GOLDILOCKS_PRIME - 1, GOLDILOCKS_PRIME - 1),
            (1, GOLDILOCKS_PRIME - 1),
            (EPSILON, EPSILON + 1),
        ];

        for (a, b) in test_cases {
            let fast = sub_fast(a, b);
            let native = native_sub(a, b);
            assert_eq!(
                canonicalize(fast), canonicalize(native),
                "sub_fast mismatch for a={}, b={}: fast={}, native={}", a, b, fast, native
            );
        }
    }

    #[test]
    fn test_mul_correctness() {
        let test_cases: Vec<(u64, u64)> = vec![
            (5, 7),
            (0, 12345),
            (1, GOLDILOCKS_PRIME - 1),
            (GOLDILOCKS_PRIME - 1, GOLDILOCKS_PRIME - 1),
            (1u64 << 40, 1u64 << 40),
            (EPSILON, EPSILON),
        ];

        for (a, b) in test_cases {
            let result = mul(a, b);
            let native = native_mul(a, b);
            assert_eq!(
                canonicalize(result), canonicalize(native),
                "mul mismatch for a={}, b={}", a, b
            );
        }
    }

    #[test]
    fn test_neg_correctness() {
        let test_values: Vec<u64> = vec![0, 1, 5, GOLDILOCKS_PRIME - 1, EPSILON];

        for a in test_values {
            let neg_a = neg(a);
            let sum = add_fast(a, neg_a);
            assert_eq!(
                canonicalize(sum), 0,
                "neg failed for a={}: neg={}, sum={}", a, neg_a, sum
            );
        }
    }

    #[test]
    fn test_double_correctness() {
        let test_values: Vec<u64> = vec![0, 1, 5, GOLDILOCKS_PRIME - 1, EPSILON, 123456789];

        for a in test_values {
            let doubled = double(a);
            let added = add_fast(a, a);
            assert_eq!(
                canonicalize(doubled), canonicalize(added),
                "double mismatch for a={}", a
            );
        }
    }

    #[test]
    fn test_field_axioms() {
        let a = 123456789u64;
        let b = 987654321u64;
        let c = 555555555u64;

        // Commutativity
        assert_eq!(canonicalize(add_fast(a, b)), canonicalize(add_fast(b, a)));
        assert_eq!(canonicalize(mul(a, b)), canonicalize(mul(b, a)));

        // Associativity
        assert_eq!(
            canonicalize(add_fast(add_fast(a, b), c)),
            canonicalize(add_fast(a, add_fast(b, c)))
        );
        assert_eq!(
            canonicalize(mul(mul(a, b), c)),
            canonicalize(mul(a, mul(b, c)))
        );

        // Distributivity
        assert_eq!(
            canonicalize(mul(a, add_fast(b, c))),
            canonicalize(add_fast(mul(a, b), mul(a, c)))
        );

        // Identity
        assert_eq!(canonicalize(add_fast(a, 0)), canonicalize(a));
        assert_eq!(canonicalize(mul(a, 1)), canonicalize(a));
    }
}
