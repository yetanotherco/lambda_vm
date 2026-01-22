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
    EPSILON, GOLDILOCKS_PRIME, add_fast, double, mul, neg, reduce128, sub_fast,
};

/// Base field squaring (uses native Rust for optimal performance).
#[inline(always)]
pub fn square(a: u64) -> u64 {
    reduce128((a as u128) * (a as u128))
}

// =====================================================
// MULTIPLY BY CONSTANTS
// =====================================================

/// Multiply by 7 (Fp2 non-residue): 7a = a + 2a + 4a
///
/// Uses field operations to avoid overflow issues.
#[inline(always)]
pub fn mul_by_7(a: u64) -> u64 {
    let a2 = double(a);
    let a4 = double(a2);
    let temp = add_fast(a, a2);
    add_fast(temp, a4)
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

/// Fp2 multiplication:
/// (a0 + a1*w) * (b0 + b1*w) = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*w
///
/// Performs field multiplications first, then multiply by 7 in the field.
#[inline(always)]
pub fn fp2_mul(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    // Perform field multiplications
    let a0b0 = mul(a[0], b[0]);
    let a1b1 = mul(a[1], b[1]);
    let a0b1 = mul(a[0], b[1]);
    let a1b0 = mul(a[1], b[0]);

    // c0 = a0*b0 + 7*a1*b1
    let w_a1b1 = mul_by_7(a1b1);
    let c0 = add_fast(a0b0, w_a1b1);

    // c1 = a0*b1 + a1*b0
    let c1 = add_fast(a0b1, a1b0);

    [c0, c1]
}

/// Fp2 squaring:
/// (a0 + a1*w)^2 = (a0^2 + 7*a1^2) + 2*a0*a1*w
///
/// Performs field operations first, then multiply by constants in the field.
#[inline(always)]
pub fn fp2_square(a: [u64; 2]) -> [u64; 2] {
    // Perform field operations
    let a0_sq = square(a[0]);
    let a1_sq = square(a[1]);
    let a0a1 = mul(a[0], a[1]);

    // c0 = a0^2 + 7*a1^2
    let w_a1_sq = mul_by_7(a1_sq);
    let c0 = add_fast(a0_sq, w_a1_sq);

    // c1 = 2*a0*a1
    let c1 = double(a0a1);

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

/// Fp3 multiplication using Karatsuba-like algorithm:
/// (a0 + a1*w + a2*w^2) * (b0 + b1*w + b2*w^2) mod (w^3 - 2)
///
/// Uses 6 field multiplications instead of 9.
#[inline(always)]
pub fn fp3_mul(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
    // Three direct products
    let v0 = mul(a[0], b[0]);
    let v1 = mul(a[1], b[1]);
    let v2 = mul(a[2], b[2]);

    // Karatsuba terms to save multiplications
    // t0 = (a1 + a2)(b1 + b2) - v1 - v2
    let a1_plus_a2 = add_fast(a[1], a[2]);
    let b1_plus_b2 = add_fast(b[1], b[2]);
    let temp0 = mul(a1_plus_a2, b1_plus_b2);
    let temp0 = sub_fast(temp0, v1);
    let t0 = sub_fast(temp0, v2);

    // t1 = (a0 + a1)(b0 + b1) - v0 - v1
    let a0_plus_a1 = add_fast(a[0], a[1]);
    let b0_plus_b1 = add_fast(b[0], b[1]);
    let temp1 = mul(a0_plus_a1, b0_plus_b1);
    let temp1 = sub_fast(temp1, v0);
    let t1 = sub_fast(temp1, v1);

    // t2 = (a0 + a2)(b0 + b2) - v0 - v2
    let a0_plus_a2 = add_fast(a[0], a[2]);
    let b0_plus_b2 = add_fast(b[0], b[2]);
    let temp2 = mul(a0_plus_a2, b0_plus_b2);
    let temp2 = sub_fast(temp2, v0);
    let t2 = sub_fast(temp2, v2);

    // Final combination with w^3 = 2
    // c0 = v0 + 2*t0
    let c0 = add_fast(v0, double(t0));
    // c1 = t1 + 2*v2
    let c1 = add_fast(t1, double(v2));
    // c2 = t2 + v1
    let c2 = add_fast(t2, v1);

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
// SUBFIELD OPERATIONS (Fp2)
// =====================================================
// All operations support both argument orders with zero overhead.

/// Add base field element to Fp2: base + ext = [a + b0, b1]
#[inline(always)]
pub fn fp2_add_base(a: u64, b: [u64; 2]) -> [u64; 2] {
    [add_fast(a, b[0]), b[1]]
}

/// Add Fp2 to base field element: ext + base = [a0 + b, a1]
/// Same as fp2_add_base due to commutativity.
#[inline(always)]
pub fn fp2_ext_add_base(a: [u64; 2], b: u64) -> [u64; 2] {
    [add_fast(a[0], b), a[1]]
}

/// Subtract Fp2 from base field element: base - ext = [a - b0, -b1]
#[inline(always)]
pub fn fp2_sub_from_base(a: u64, b: [u64; 2]) -> [u64; 2] {
    [sub_fast(a, b[0]), neg(b[1])]
}

/// Subtract base field element from Fp2: ext - base = [a0 - b, a1]
#[inline(always)]
pub fn fp2_ext_sub_base(a: [u64; 2], b: u64) -> [u64; 2] {
    [sub_fast(a[0], b), a[1]]
}

/// Multiply base field element by Fp2: base * ext = [a*b0, a*b1]
#[inline(always)]
pub fn fp2_mul_ext(a: u64, b: [u64; 2]) -> [u64; 2] {
    [mul(a, b[0]), mul(a, b[1])]
}

/// Multiply Fp2 by base field element: ext * base = [a0*b, a1*b]
/// Same as fp2_mul_ext due to commutativity.
#[inline(always)]
pub fn fp2_ext_mul_base(a: [u64; 2], b: u64) -> [u64; 2] {
    [mul(a[0], b), mul(a[1], b)]
}

// =====================================================
// SUBFIELD OPERATIONS (Fp3)
// =====================================================
// All operations support both argument orders with zero overhead.

/// Add base field element to Fp3: base + ext = [a + b0, b1, b2]
#[inline(always)]
pub fn fp3_add_base(a: u64, b: [u64; 3]) -> [u64; 3] {
    [add_fast(a, b[0]), b[1], b[2]]
}

/// Add Fp3 to base field element: ext + base = [a0 + b, a1, a2]
/// Same as fp3_add_base due to commutativity.
#[inline(always)]
pub fn fp3_ext_add_base(a: [u64; 3], b: u64) -> [u64; 3] {
    [add_fast(a[0], b), a[1], a[2]]
}

/// Subtract Fp3 from base field element: base - ext = [a - b0, -b1, -b2]
#[inline(always)]
pub fn fp3_sub_from_base(a: u64, b: [u64; 3]) -> [u64; 3] {
    [sub_fast(a, b[0]), neg(b[1]), neg(b[2])]
}

/// Subtract base field element from Fp3: ext - base = [a0 - b, a1, a2]
#[inline(always)]
pub fn fp3_ext_sub_base(a: [u64; 3], b: u64) -> [u64; 3] {
    [sub_fast(a[0], b), a[1], a[2]]
}

/// Multiply base field element by Fp3: base * ext = [a*b0, a*b1, a*b2]
#[inline(always)]
pub fn fp3_mul_ext(a: u64, b: [u64; 3]) -> [u64; 3] {
    [mul(a, b[0]), mul(a, b[1]), mul(a, b[2])]
}

/// Multiply Fp3 by base field element: ext * base = [a0*b, a1*b, a2*b]
/// Same as fp3_mul_ext due to commutativity.
#[inline(always)]
pub fn fp3_ext_mul_base(a: [u64; 3], b: u64) -> [u64; 3] {
    [mul(a[0], b), mul(a[1], b), mul(a[2], b)]
}
