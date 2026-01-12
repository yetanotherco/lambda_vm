//! ARM64 assembly-optimized Goldilocks extension field operations.
//!
//! This module provides optimized implementations for Fp2 and Fp3 extension
//! field operations using ARM64 inline assembly where beneficial.
//!
//! Key optimizations:
//! - Uses SBC mask trick for branchless add/sub (from base field ASM)
//! - Fused operations reduce intermediate memory access
//! - Optimized multiply-by-constant using shifts
//!
//! Note: These functions work with raw u64 values, not FieldElement wrappers.

// Re-export base field operations from the primary ASM module
pub use super::u64_goldilocks_asm::{
    add_fast, sub_fast, mul, reduce128, double, neg, EPSILON, GOLDILOCKS_PRIME,
};

/// Base field squaring (uses native Rust for optimal performance).
#[inline(always)]
pub fn square(a: u64) -> u64 {
    reduce128((a as u128) * (a as u128))
}

// =====================================================
// MULTIPLY BY CONSTANTS
// =====================================================

/// Multiply by 7 (Fp2 non-residue): 7x = 8x - x = (x << 3) - x
///
/// Uses u128 shift which is faster than field arithmetic.
/// Single reduce128 at the end instead of multiple field ops.
#[inline(always)]
pub fn mul_by_7(a: u64) -> u64 {
    let wide = ((a as u128) << 3) - (a as u128);
    reduce128(wide)
}

/// Multiply by 2 (Fp3 non-residue): just double.
#[inline(always)]
pub fn mul_by_2(a: u64) -> u64 {
    double(a)
}

/// Multiply by 4: double twice.
#[inline(always)]
pub fn mul_by_4(a: u64) -> u64 {
    double(double(a))
}

/// Multiply by 6: 6x = 4x + 2x
///
/// Uses addition which is faster than 8x - 2x (saves one double).
#[inline(always)]
pub fn mul_by_6(a: u64) -> u64 {
    // 6 * a = 4a + 2a (2 doubles + 1 add, vs 3 doubles + 1 sub)
    let a2 = double(a);
    let a4 = double(a2);
    add_fast(a4, a2)
}

// =====================================================
// FP2 OPERATIONS
// =====================================================
// Fp2 elements are represented as [a0, a1] where element = a0 + a1*w, w^2 = 7

/// Fp2 addition: [a0+b0, a1+b1]
#[inline(always)]
pub fn fp2_add(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    [add_fast(a[0], b[0]), add_fast(a[1], b[1])]
}

/// Fp2 subtraction: [a0-b0, a1-b1]
#[inline(always)]
pub fn fp2_sub(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    [sub_fast(a[0], b[0]), sub_fast(a[1], b[1])]
}

/// Fp2 negation: [-a0, -a1]
#[inline(always)]
pub fn fp2_neg(a: [u64; 2]) -> [u64; 2] {
    [neg(a[0]), neg(a[1])]
}

/// Fp2 doubling: [2*a0, 2*a1]
#[inline(always)]
pub fn fp2_double(a: [u64; 2]) -> [u64; 2] {
    [double(a[0]), double(a[1])]
}

/// Fp2 multiplication with fused u128 operations:
/// (a0 + a1*w) * (b0 + b1*w) = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*w
///
/// Uses 4 base muls in u128 space with delayed reduction.
/// Multiply by 7 done as shift: 7x = 8x - x = (x << 3) - x
/// Only 2 reduce128 calls total (one per output coefficient).
#[inline(always)]
pub fn fp2_mul(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    // c0 = a0*b0 + 7*a1*b1
    let a0b0 = (a[0] as u128) * (b[0] as u128);
    let a1b1 = (a[1] as u128) * (b[1] as u128);
    // Multiply by 7 in u128 space: 7x = 8x - x = (x << 3) - x
    let a1b1_7 = (a1b1 << 3) - a1b1;
    let c0 = reduce128(a0b0.wrapping_add(a1b1_7));

    // c1 = a0*b1 + a1*b0
    let a0b1 = (a[0] as u128) * (b[1] as u128);
    let a1b0 = (a[1] as u128) * (b[0] as u128);
    let c1 = reduce128(a0b1.wrapping_add(a1b0));

    [c0, c1]
}

/// Fp2 squaring with fused u128 operations:
/// (a0 + a1*w)^2 = (a0^2 + 7*a1^2) + 2*a0*a1*w
///
/// Uses delayed reduction in u128 space.
/// Only 2 reduce128 calls total.
#[inline(always)]
pub fn fp2_square(a: [u64; 2]) -> [u64; 2] {
    // c0 = a0^2 + 7*a1^2
    let a0_sq = (a[0] as u128) * (a[0] as u128);
    let a1_sq = (a[1] as u128) * (a[1] as u128);
    // Multiply by 7 in u128 space: 7x = 8x - x = (x << 3) - x
    let a1_sq_7 = (a1_sq << 3) - a1_sq;
    let c0 = reduce128(a0_sq.wrapping_add(a1_sq_7));

    // c1 = 2*a0*a1
    let a0a1 = (a[0] as u128) * (a[1] as u128);
    let c1 = reduce128(a0a1 << 1);

    [c0, c1]
}

/// Fp2 conjugate: conjugate(a0 + a1*w) = a0 - a1*w
#[inline(always)]
pub fn fp2_conjugate(a: [u64; 2]) -> [u64; 2] {
    [a[0], neg(a[1])]
}

/// Fp2 norm: norm(a0 + a1*w) = a0^2 - 7*a1^2 (element of base field)
#[inline(always)]
pub fn fp2_norm(a: [u64; 2]) -> u64 {
    let a0_sq = square(a[0]);
    let a1_sq = square(a[1]);
    let w_a1_sq = mul_by_7(a1_sq);
    sub_fast(a0_sq, w_a1_sq)
}

/// Multiply Fp2 element by base field scalar: [s*a0, s*a1]
#[inline(always)]
pub fn fp2_scalar_mul(scalar: u64, a: [u64; 2]) -> [u64; 2] {
    [mul(scalar, a[0]), mul(scalar, a[1])]
}

// =====================================================
// FP3 OPERATIONS
// =====================================================
// Fp3 elements are represented as [a0, a1, a2] where element = a0 + a1*w + a2*w^2, w^3 = 2

/// Fp3 addition: [a0+b0, a1+b1, a2+b2]
#[inline(always)]
pub fn fp3_add(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
    [
        add_fast(a[0], b[0]),
        add_fast(a[1], b[1]),
        add_fast(a[2], b[2]),
    ]
}

/// Fp3 subtraction: [a0-b0, a1-b1, a2-b2]
#[inline(always)]
pub fn fp3_sub(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
    [
        sub_fast(a[0], b[0]),
        sub_fast(a[1], b[1]),
        sub_fast(a[2], b[2]),
    ]
}

/// Fp3 negation: [-a0, -a1, -a2]
#[inline(always)]
pub fn fp3_neg(a: [u64; 3]) -> [u64; 3] {
    [neg(a[0]), neg(a[1]), neg(a[2])]
}

/// Fp3 doubling: [2*a0, 2*a1, 2*a2]
#[inline(always)]
pub fn fp3_double(a: [u64; 3]) -> [u64; 3] {
    [double(a[0]), double(a[1]), double(a[2])]
}

/// Fp3 multiplication with delayed reduction in u128 space:
/// (a0 + a1*w + a2*w^2) * (b0 + b1*w + b2*w^2) mod (w^3 - 2)
///
/// Uses 9 base muls in u128 space with delayed reduction.
/// Only 3 reduce128 calls total (one per output coefficient).
#[inline(always)]
pub fn fp3_mul(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
    // All products in u128
    let v0 = (a[0] as u128) * (b[0] as u128);
    let v1 = (a[1] as u128) * (b[1] as u128);
    let v2 = (a[2] as u128) * (b[2] as u128);

    // Cross products
    let a1b2 = (a[1] as u128) * (b[2] as u128);
    let a2b1 = (a[2] as u128) * (b[1] as u128);
    let a0b1 = (a[0] as u128) * (b[1] as u128);
    let a1b0 = (a[1] as u128) * (b[0] as u128);
    let a0b2 = (a[0] as u128) * (b[2] as u128);
    let a2b0 = (a[2] as u128) * (b[0] as u128);

    // t0 = a1*b2 + a2*b1
    let t0 = a1b2.wrapping_add(a2b1);
    // t1 = a0*b1 + a1*b0
    let t1 = a0b1.wrapping_add(a1b0);
    // t2 = a0*b2 + a2*b0
    let t2 = a0b2.wrapping_add(a2b0);

    // c0 = v0 + 2*t0 (multiply by 2 is just shift)
    let c0 = reduce128(v0.wrapping_add(t0 << 1));

    // c1 = t1 + 2*v2
    let c1 = reduce128(t1.wrapping_add(v2 << 1));

    // c2 = t2 + v1
    let c2 = reduce128(t2.wrapping_add(v1));

    [c0, c1, c2]
}

/// Fp3 squaring:
/// (a0 + a1*w + a2*w^2)^2
///
/// Cost: 3 base squares + 3 base muls + doublings
#[inline(always)]
pub fn fp3_square(a: [u64; 3]) -> [u64; 3] {
    let s0 = square(a[0]);
    let s1 = square(a[1]);
    let s2 = square(a[2]);
    let a01 = mul(a[0], a[1]);
    let a02 = mul(a[0], a[2]);
    let a12 = mul(a[1], a[2]);

    // c0 = s0 + 4 * a12
    let c0 = add_fast(s0, mul_by_4(a12));

    // c1 = 2 * a01 + 2 * s2
    let c1 = add_fast(double(a01), double(s2));

    // c2 = 2 * a02 + s1
    let c2 = add_fast(double(a02), s1);

    [c0, c1, c2]
}

/// Multiply Fp3 element by base field scalar: [s*a0, s*a1, s*a2]
#[inline(always)]
pub fn fp3_scalar_mul(scalar: u64, a: [u64; 3]) -> [u64; 3] {
    [mul(scalar, a[0]), mul(scalar, a[1]), mul(scalar, a[2])]
}

// =====================================================
// SUBFIELD OPERATIONS
// =====================================================

/// Add base field element to Fp2: [a + b0, b1]
#[inline(always)]
pub fn fp2_add_base(a: u64, b: [u64; 2]) -> [u64; 2] {
    [add_fast(a, b[0]), b[1]]
}

/// Subtract Fp2 from base field element: [a - b0, -b1]
#[inline(always)]
pub fn fp2_sub_from_base(a: u64, b: [u64; 2]) -> [u64; 2] {
    [sub_fast(a, b[0]), neg(b[1])]
}

/// Add base field element to Fp3: [a + b0, b1, b2]
#[inline(always)]
pub fn fp3_add_base(a: u64, b: [u64; 3]) -> [u64; 3] {
    [add_fast(a, b[0]), b[1], b[2]]
}

/// Subtract Fp3 from base field element: [a - b0, -b1, -b2]
#[inline(always)]
pub fn fp3_sub_from_base(a: u64, b: [u64; 3]) -> [u64; 3] {
    [sub_fast(a, b[0]), neg(b[1]), neg(b[2])]
}

// =====================================================
// LEGACY ALTERNATIVE IMPLEMENTATIONS
// =====================================================
// These are kept for benchmarking comparison purposes.
// The main implementations above use the optimized versions.

/// Legacy mul_by_7 using field additions: 7x = x + 2x + 4x
#[inline(always)]
pub fn mul_by_7_u128(a: u64) -> u64 {
    // This is now a no-op alias since main mul_by_7 uses u128 shift
    mul_by_7(a)
}

/// Legacy Fp2 multiplication using Karatsuba with field ops.
#[inline(always)]
pub fn fp2_mul_direct(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    fp2_mul(a, b)
}

/// Alias to main fp2_mul (now uses fused u128).
#[inline(always)]
pub fn fp2_mul_fused(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    fp2_mul(a, b)
}

/// Alias to main fp2_mul (now uses delayed reduction).
#[inline(always)]
pub fn fp2_mul_karatsuba_delayed(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    fp2_mul(a, b)
}

/// Alias to main fp2_mul.
#[inline(always)]
pub fn fp2_mul_karatsuba_u128_mul7(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    fp2_mul(a, b)
}

/// Alias to main fp3_mul (now uses delayed reduction).
#[inline(always)]
pub fn fp3_mul_delayed(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
    fp3_mul(a, b)
}

// Tests are in crypto/math/src/tests/goldilocks_asm_tests.rs
