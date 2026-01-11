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
/// Uses shifts and subtract which is faster than general multiplication.
#[inline(always)]
pub fn mul_by_7(a: u64) -> u64 {
    // 7 * a = (a << 3) - a = 8a - a
    // But we need to be careful about overflow
    // a << 3 might overflow, so we compute it as:
    // 8a mod p = double(double(double(a)))
    let a2 = double(a);
    let a4 = double(a2);
    let a8 = double(a4);
    sub_fast(a8, a)
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

/// Multiply by 6: 6x = 8x - 2x = (x << 3) - (x << 1)
#[inline(always)]
pub fn mul_by_6(a: u64) -> u64 {
    let a2 = double(a);
    let a4 = double(a2);
    let a8 = double(a4);
    sub_fast(a8, a2)
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

/// Fp2 multiplication using Karatsuba:
/// (a0 + a1*w) * (b0 + b1*w) = (a0*b0 + 7*a1*b1) + (a0*b1 + a1*b0)*w
///
/// Using Karatsuba: (a0+a1)(b0+b1) - a0*b0 - a1*b1 = a0*b1 + a1*b0
/// Cost: 3 base muls + adds/subs + mul_by_7
#[inline(always)]
pub fn fp2_mul(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    let a0b0 = mul(a[0], b[0]);
    let a1b1 = mul(a[1], b[1]);

    // (a0 + a1) * (b0 + b1)
    let a_sum = add_fast(a[0], a[1]);
    let b_sum = add_fast(b[0], b[1]);
    let z = mul(a_sum, b_sum);

    // c0 = a0*b0 + 7*a1*b1
    let w_a1b1 = mul_by_7(a1b1);
    let c0 = add_fast(a0b0, w_a1b1);

    // c1 = z - a0*b0 - a1*b1 = a0*b1 + a1*b0
    let c1 = sub_fast(sub_fast(z, a0b0), a1b1);

    [c0, c1]
}

/// Fp2 squaring:
/// (a0 + a1*w)^2 = (a0^2 + 7*a1^2) + 2*a0*a1*w
///
/// Cost: 2 base squares + 1 base mul + mul_by_7 + double
#[inline(always)]
pub fn fp2_square(a: [u64; 2]) -> [u64; 2] {
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
/// Cost: 6 base muls + adds/subs + doublings
#[inline(always)]
pub fn fp3_mul(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
    let v0 = mul(a[0], b[0]);
    let v1 = mul(a[1], b[1]);
    let v2 = mul(a[2], b[2]);

    // t0 = (a1 + a2)(b1 + b2) - v1 - v2
    let a12 = add_fast(a[1], a[2]);
    let b12 = add_fast(b[1], b[2]);
    let t0 = sub_fast(sub_fast(mul(a12, b12), v1), v2);

    // t1 = (a0 + a1)(b0 + b1) - v0 - v1
    let a01 = add_fast(a[0], a[1]);
    let b01 = add_fast(b[0], b[1]);
    let t1 = sub_fast(sub_fast(mul(a01, b01), v0), v1);

    // t2 = (a0 + a2)(b0 + b2) - v0 - v2
    let a02 = add_fast(a[0], a[2]);
    let b02 = add_fast(b[0], b[2]);
    let t2 = sub_fast(sub_fast(mul(a02, b02), v0), v2);

    // c0 = v0 + 2 * t0
    let c0 = add_fast(v0, double(t0));

    // c1 = t1 + 2 * v2
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

#[cfg(test)]
mod tests {
    use super::*;

    fn canonicalize(x: u64) -> u64 {
        if x >= GOLDILOCKS_PRIME { x - GOLDILOCKS_PRIME } else { x }
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
    fn test_fp3_square_equals_mul() {
        let a = [5u64, 7u64, 11u64];
        let sq = fp3_square(a);
        let mul_result = fp3_mul(a, a);
        assert_eq!(canonicalize(sq[0]), canonicalize(mul_result[0]));
        assert_eq!(canonicalize(sq[1]), canonicalize(mul_result[1]));
        assert_eq!(canonicalize(sq[2]), canonicalize(mul_result[2]));
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
    fn test_fp2_mul_one() {
        let a = [5u64, 7u64];
        let one = [1u64, 0u64];
        let result = fp2_mul(a, one);
        assert_eq!(canonicalize(result[0]), canonicalize(a[0]));
        assert_eq!(canonicalize(result[1]), canonicalize(a[1]));
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
