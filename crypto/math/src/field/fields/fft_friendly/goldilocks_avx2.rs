//! AVX2-accelerated Goldilocks field arithmetic operating on 4 packed u64 elements.
//!
//! Each `__m256i` holds 4 Goldilocks field elements (4 × u64).
//! All functions require the `avx2` target feature.
//!
//! The reduction strategy mirrors the scalar `u64_goldilocks.rs` implementation,
//! but the multiplication uses 32-bit schoolbook decomposition (since x86 lacks
//! a 64×64→128 SIMD multiply) following the approach from Plonky3.

use core::arch::x86_64::*;

use super::u64_goldilocks::GOLDILOCKS_PRIME;

const EPSILON: u64 = 0xFFFF_FFFF;

/// Add 4 packed Goldilocks elements: a + b mod p.
///
/// If a + b overflows u64, we add EPSILON (since 2^64 ≡ EPSILON mod p).
///
/// # Safety
/// Caller must ensure AVX2 is available (`is_x86_feature_detected!("avx2")`).
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn add4(a: __m256i, b: __m256i) -> __m256i {
    let eps = _mm256_set1_epi64x(EPSILON as i64);
    let sign_bit = _mm256_set1_epi64x(i64::MIN);

    let sum = _mm256_add_epi64(a, b);
    // Unsigned overflow detection via signed compare with flipped sign bits.
    let overflow = _mm256_cmpgt_epi64(
        _mm256_xor_si256(a, sign_bit),
        _mm256_xor_si256(sum, sign_bit),
    );
    let result = _mm256_add_epi64(sum, _mm256_and_si256(overflow, eps));
    // Handle rare second overflow
    let overflow2 = _mm256_cmpgt_epi64(
        _mm256_xor_si256(sum, sign_bit),
        _mm256_xor_si256(result, sign_bit),
    );
    _mm256_add_epi64(result, _mm256_and_si256(overflow2, eps))
}

/// Subtract 4 packed Goldilocks elements: a - b mod p.
///
/// If a - b underflows, we subtract EPSILON (since -2^64 ≡ -EPSILON mod p).
///
/// # Safety
/// Caller must ensure AVX2 is available (`is_x86_feature_detected!("avx2")`).
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn sub4(a: __m256i, b: __m256i) -> __m256i {
    let eps = _mm256_set1_epi64x(EPSILON as i64);
    let sign_bit = _mm256_set1_epi64x(i64::MIN);

    let diff = _mm256_sub_epi64(a, b);
    // Underflow where b > a (unsigned)
    let underflow =
        _mm256_cmpgt_epi64(_mm256_xor_si256(b, sign_bit), _mm256_xor_si256(a, sign_bit));
    let result = _mm256_sub_epi64(diff, _mm256_and_si256(underflow, eps));
    // Handle rare second underflow
    let underflow2 = _mm256_cmpgt_epi64(
        _mm256_xor_si256(result, sign_bit),
        _mm256_xor_si256(diff, sign_bit),
    );
    _mm256_sub_epi64(result, _mm256_and_si256(underflow2, eps))
}

/// Negate 4 packed Goldilocks elements: p - a mod p.
///
/// # Safety
/// Caller must ensure AVX2 is available (`is_x86_feature_detected!("avx2")`).
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn neg4(a: __m256i) -> __m256i {
    unsafe { sub4(_mm256_set1_epi64x(GOLDILOCKS_PRIME as i64), a) }
}

/// Multiply 4 packed Goldilocks elements using 32-bit schoolbook decomposition.
///
/// Since AVX2 has no 64×64→128 multiply, we decompose each u64 into two u32 halves
/// and use `_mm256_mul_epu32` (32×32→64 unsigned multiply).
///
/// # Safety
/// Caller must ensure AVX2 is available (`is_x86_feature_detected!("avx2")`).
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn mul4(a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let sign_bit = _mm256_set1_epi64x(i64::MIN);

        let a_hi = _mm256_srli_epi64(a, 32);
        let b_hi = _mm256_srli_epi64(b, 32);

        // Four 32×32→64 products
        let ll = _mm256_mul_epu32(a, b);
        let lh = _mm256_mul_epu32(a, b_hi);
        let hl = _mm256_mul_epu32(a_hi, b);
        let hh = _mm256_mul_epu32(a_hi, b_hi);

        // Middle sum with carry detection
        let mid = _mm256_add_epi64(lh, hl);
        let mid_carry = _mm256_cmpgt_epi64(
            _mm256_xor_si256(lh, sign_bit),
            _mm256_xor_si256(mid, sign_bit),
        );
        let mid_carry_val = _mm256_srli_epi64(mid_carry, 63);

        // Split middle into low 32 and high 32
        let mid_lo = _mm256_slli_epi64(mid, 32);
        let mid_hi = _mm256_srli_epi64(mid, 32);

        // result_lo = ll + (mid_lo_32 << 32), with carry detection
        let res_lo = _mm256_add_epi64(ll, mid_lo);
        let lo_carry = _mm256_cmpgt_epi64(
            _mm256_xor_si256(ll, sign_bit),
            _mm256_xor_si256(res_lo, sign_bit),
        );
        let lo_carry_val = _mm256_srli_epi64(lo_carry, 63);

        // result_hi = hh + mid_hi + lo_carry + (mid_carry << 32)
        let mid_carry_shifted = _mm256_slli_epi64(mid_carry_val, 32);
        let res_hi = _mm256_add_epi64(
            _mm256_add_epi64(hh, mid_hi),
            _mm256_add_epi64(lo_carry_val, mid_carry_shifted),
        );

        reduce(res_lo, res_hi)
    }
}

/// Square 4 packed Goldilocks elements. Saves one `_mm256_mul_epu32` vs `mul4(a, a)`
/// since the cross term `a_lo * a_hi` only needs computing once (then doubled).
///
/// # Safety
/// Caller must ensure AVX2 is available (`is_x86_feature_detected!("avx2")`).
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn square4(a: __m256i) -> __m256i {
    let sign_bit = _mm256_set1_epi64x(i64::MIN);

    let a_hi = _mm256_srli_epi64(a, 32);

    // Three 32×32→64 products (vs four for mul4)
    let ll = _mm256_mul_epu32(a, a);
    let lh = _mm256_mul_epu32(a, a_hi);
    let hh = _mm256_mul_epu32(a_hi, a_hi);

    // mid = 2 * lh (since lh == hl for squaring)
    let mid = _mm256_add_epi64(lh, lh);
    // Carry from doubling: top bit of lh before the shift
    let mid_carry_val = _mm256_srli_epi64(lh, 63);

    // Split middle into low 32 and high 32
    let mid_lo = _mm256_slli_epi64(mid, 32);
    let mid_hi = _mm256_srli_epi64(mid, 32);

    // result_lo = ll + (mid_lo_32 << 32), with carry detection
    let res_lo = _mm256_add_epi64(ll, mid_lo);
    let lo_carry = _mm256_cmpgt_epi64(
        _mm256_xor_si256(ll, sign_bit),
        _mm256_xor_si256(res_lo, sign_bit),
    );
    let lo_carry_val = _mm256_srli_epi64(lo_carry, 63);

    // result_hi = hh + mid_hi + lo_carry + (mid_carry << 32)
    let mid_carry_shifted = _mm256_slli_epi64(mid_carry_val, 32);
    let res_hi = _mm256_add_epi64(
        _mm256_add_epi64(hh, mid_hi),
        _mm256_add_epi64(lo_carry_val, mid_carry_shifted),
    );

    // SAFETY: reduce is unsafe because it calls add4/sub4 which require AVX2.
    unsafe { reduce(res_lo, res_hi) }
}

/// Compute multiplicative inverse of 4 packed Goldilocks elements via the
/// addition chain for x^(p-2), matching `u64_goldilocks::inv_addition_chain`.
///
/// # Safety
/// Caller must ensure AVX2 is available (`is_x86_feature_detected!("avx2")`).
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn inv4(a: __m256i) -> __m256i {
    /// Square `base` n times, then multiply by `tail`.
    #[inline(always)]
    unsafe fn exp_acc4(base: __m256i, tail: __m256i, n: u32) -> __m256i {
        unsafe {
            let mut result = base;
            for _ in 0..n {
                result = square4(result);
            }
            mul4(result, tail)
        }
    }

    unsafe {
        let x = a;
        let x2 = square4(x);
        let x3 = mul4(x2, x);
        let x7 = exp_acc4(x3, x, 1);
        let x63 = exp_acc4(x7, x7, 3);
        let x12m1 = exp_acc4(x63, x63, 6);
        let x24m1 = exp_acc4(x12m1, x12m1, 12);
        let x30m1 = exp_acc4(x24m1, x63, 6);
        let x31m1 = exp_acc4(x30m1, x, 1);
        let x32m1 = exp_acc4(x31m1, x, 1);

        let mut t = x31m1;
        for _ in 0..33 {
            t = square4(t);
        }
        mul4(t, x32m1)
    }
}

/// Reduce a 128-bit value (lo + hi * 2^64) to a Goldilocks field element.
///
/// hi = hi_hi * 2^32 + hi_lo
/// lo + hi * 2^64 ≡ lo + hi_lo * EPSILON - hi_hi (mod p)
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn reduce(lo: __m256i, hi: __m256i) -> __m256i {
    unsafe {
        let lo_mask = _mm256_set1_epi64x(0xFFFFFFFF);
        let hi_hi = _mm256_srli_epi64(hi, 32);
        let hi_lo = _mm256_and_si256(hi, lo_mask);

        // t0 = lo - hi_hi
        let t0 = sub4(lo, hi_hi);

        // t1 = hi_lo * EPSILON = (hi_lo << 32) - hi_lo
        let t1 = _mm256_sub_epi64(_mm256_slli_epi64(hi_lo, 32), hi_lo);

        add4(t0, t1)
    }
}

// =====================================================
// CUBIC EXTENSION AVX2 OPS
// =====================================================
// A cubic extension element across 4 independent points is stored as
// `Ext3x4 = (__m256i, __m256i, __m256i)` — one register per component
// (c0, c1, c2), each holding 4 base field values from 4 different points.

/// 4-wide packed cubic extension element: (c0, c1, c2) where each is 4 × u64.
pub type Ext3x4 = (__m256i, __m256i, __m256i);

/// Add 4 packed cubic extension elements component-wise.
///
/// # Safety
/// Caller must ensure AVX2 is available (`is_x86_feature_detected!("avx2")`).
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn ext3_add4(a: Ext3x4, b: Ext3x4) -> Ext3x4 {
    unsafe { (add4(a.0, b.0), add4(a.1, b.1), add4(a.2, b.2)) }
}

/// Subtract 4 packed cubic extension elements component-wise.
///
/// # Safety
/// Caller must ensure AVX2 is available (`is_x86_feature_detected!("avx2")`).
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn ext3_sub4(a: Ext3x4, b: Ext3x4) -> Ext3x4 {
    unsafe { (sub4(a.0, b.0), sub4(a.1, b.1), sub4(a.2, b.2)) }
}

/// Multiply 4 packed cubic extension elements using Karatsuba with residue w^3 = 2.
/// Uses 6 base-field `mul4` calls (vs 9 for schoolbook).
///
/// # Safety
/// Caller must ensure AVX2 is available (`is_x86_feature_detected!("avx2")`).
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn ext3_mul4(a: Ext3x4, b: Ext3x4) -> Ext3x4 {
    unsafe {
        let v0 = mul4(a.0, b.0);
        let v1 = mul4(a.1, b.1);
        let v2 = mul4(a.2, b.2);

        let t0 = sub4(sub4(mul4(add4(a.1, a.2), add4(b.1, b.2)), v1), v2);
        let t1 = sub4(sub4(mul4(add4(a.0, a.1), add4(b.0, b.1)), v0), v1);
        let t2 = sub4(sub4(mul4(add4(a.0, a.2), add4(b.0, b.2)), v0), v2);

        // c0 = v0 + 2*t0, c1 = t1 + 2*v2, c2 = t2 + v1
        (add4(v0, add4(t0, t0)), add4(t1, add4(v2, v2)), add4(t2, v1))
    }
}

/// Multiply 4 packed cubic extension elements by a packed base field scalar.
/// F x E multiplication: 3 base-field `mul4` calls.
///
/// # Safety
/// Caller must ensure AVX2 is available (`is_x86_feature_detected!("avx2")`).
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn ext3_scalar_mul4(scalar: __m256i, ext: Ext3x4) -> Ext3x4 {
    unsafe {
        (
            mul4(scalar, ext.0),
            mul4(scalar, ext.1),
            mul4(scalar, ext.2),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::element::FieldElement;
    use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
    use crate::field::traits::IsField;

    type FpE = FieldElement<GoldilocksField>;

    unsafe fn load4(a: [u64; 4]) -> __m256i {
        unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) }
    }

    unsafe fn store4(v: __m256i) -> [u64; 4] {
        let mut out = [0u64; 4];
        unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v) };
        out
    }

    fn canon(x: u64) -> u64 {
        if x >= GOLDILOCKS_PRIME {
            x - GOLDILOCKS_PRIME
        } else {
            x
        }
    }

    #[test]
    fn test_add4() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let cases: [(u64, u64); 5] = [
            (5, 7),
            (GOLDILOCKS_PRIME - 1, 2),
            (GOLDILOCKS_PRIME - 1, GOLDILOCKS_PRIME - 1),
            (0, 0),
            (1 << 40, 1 << 40),
        ];
        for (a_val, b_val) in cases {
            let expected = canon(GoldilocksField::add(&a_val, &b_val));
            unsafe {
                let a = _mm256_set1_epi64x(a_val as i64);
                let b = _mm256_set1_epi64x(b_val as i64);
                let result = store4(add4(a, b));
                for (lane, &val) in result.iter().enumerate() {
                    assert_eq!(
                        canon(val),
                        expected,
                        "add4 mismatch for a={a_val}, b={b_val}, lane={lane}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_sub4() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let cases: [(u64, u64); 5] = [
            (10, 3),
            (3, 10),
            (0, GOLDILOCKS_PRIME - 1),
            (GOLDILOCKS_PRIME - 1, 0),
            (1 << 40, 1 << 40),
        ];
        for (a_val, b_val) in cases {
            let expected = canon(GoldilocksField::sub(&a_val, &b_val));
            unsafe {
                let a = _mm256_set1_epi64x(a_val as i64);
                let b = _mm256_set1_epi64x(b_val as i64);
                let result = store4(sub4(a, b));
                for (lane, &val) in result.iter().enumerate() {
                    assert_eq!(
                        canon(val),
                        expected,
                        "sub4 mismatch for a={a_val}, b={b_val}, lane={lane}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_neg4() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let cases: [u64; 5] = [0, 1, 5, GOLDILOCKS_PRIME - 1, 0xDEADBEEF];
        for a_val in cases {
            let expected = canon(GoldilocksField::neg(&a_val));
            unsafe {
                let a = _mm256_set1_epi64x(a_val as i64);
                let result = store4(neg4(a));
                for (lane, &val) in result.iter().enumerate() {
                    assert_eq!(
                        canon(val),
                        expected,
                        "neg4 mismatch for a={a_val}, lane={lane}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_mul4() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let cases: [(u64, u64); 8] = [
            (5, 7),
            (0, 12345),
            (1, GOLDILOCKS_PRIME - 1),
            (GOLDILOCKS_PRIME - 1, GOLDILOCKS_PRIME - 1),
            (1 << 40, 1 << 40),
            (0xDEADBEEF, 0xCAFEBABE),
            (GOLDILOCKS_PRIME - 1, 2),
            (0x1234_5678_9ABC_DEF0, 0xFEDC_BA98_7654_3210),
        ];
        for (a_val, b_val) in cases {
            let expected = canon(GoldilocksField::mul(&a_val, &b_val));
            unsafe {
                let a = _mm256_set1_epi64x(a_val as i64);
                let b = _mm256_set1_epi64x(b_val as i64);
                let result = store4(mul4(a, b));
                for (lane, &val) in result.iter().enumerate() {
                    assert_eq!(
                        canon(val),
                        expected,
                        "mul4 mismatch for a={a_val:#x}, b={b_val:#x}, lane={lane}: got {val:#x}, expected {expected:#x}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_mul4_heterogeneous_lanes() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let a_vals = [5u64, 1 << 40, GOLDILOCKS_PRIME - 1, 0xDEADBEEF];
        let b_vals = [7u64, 1 << 40, 2, 0xCAFEBABE];
        unsafe {
            let a = load4(a_vals);
            let b = load4(b_vals);
            let result = store4(mul4(a, b));
            for (lane, &val) in result.iter().enumerate() {
                let expected = canon(GoldilocksField::mul(&a_vals[lane], &b_vals[lane]));
                assert_eq!(
                    canon(val),
                    expected,
                    "mul4 heterogeneous mismatch lane={lane}"
                );
            }
        }
    }

    #[test]
    fn test_square4() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let cases: [u64; 8] = [
            0,
            1,
            5,
            GOLDILOCKS_PRIME - 1,
            1 << 40,
            0xDEADBEEF,
            0xCAFEBABE,
            0x1234_5678_9ABC_DEF0,
        ];
        for a_val in cases {
            let expected = canon(GoldilocksField::square(&a_val));
            unsafe {
                let a = _mm256_set1_epi64x(a_val as i64);
                let result = store4(square4(a));
                for i in 0..4 {
                    assert_eq!(
                        canon(result[i]),
                        expected,
                        "square4 mismatch for a={a_val:#x}, lane={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_square4_matches_mul4() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let vals = [5u64, 1 << 40, GOLDILOCKS_PRIME - 1, 0xDEADBEEF];
        unsafe {
            let a = load4(vals);
            let sq = store4(square4(a));
            let mm = store4(mul4(a, a));
            for i in 0..4 {
                assert_eq!(
                    canon(sq[i]),
                    canon(mm[i]),
                    "square4 vs mul4 mismatch lane={i}"
                );
            }
        }
    }

    #[test]
    fn test_inv4() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use super::super::u64_goldilocks::inv_addition_chain;
        let cases: [u64; 6] = [1, 5, GOLDILOCKS_PRIME - 1, 0xDEADBEEF, 1 << 40, 0xCAFEBABE];
        for a_val in cases {
            let expected = canon(inv_addition_chain(a_val));
            unsafe {
                let a = _mm256_set1_epi64x(a_val as i64);
                let result = store4(inv4(a));
                for i in 0..4 {
                    assert_eq!(
                        canon(result[i]),
                        expected,
                        "inv4 mismatch for a={a_val:#x}, lane={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_inv4_heterogeneous() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use super::super::u64_goldilocks::inv_addition_chain;
        let vals = [5u64, 1 << 40, GOLDILOCKS_PRIME - 1, 0xDEADBEEF];
        unsafe {
            let a = load4(vals);
            let result = store4(inv4(a));
            for i in 0..4 {
                let expected = canon(inv_addition_chain(vals[i]));
                assert_eq!(
                    canon(result[i]),
                    expected,
                    "inv4 heterogeneous mismatch lane={i}"
                );
            }
        }
    }

    #[test]
    fn test_inv4_times_original_is_one() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let vals = [5u64, 1 << 40, GOLDILOCKS_PRIME - 1, 0xDEADBEEF];
        unsafe {
            let a = load4(vals);
            let inv = inv4(a);
            let product = store4(mul4(a, inv));
            for i in 0..4 {
                assert_eq!(canon(product[i]), 1, "a * inv(a) != 1, lane={i}");
            }
        }
    }

    #[test]
    fn test_ext3_add4() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use super::super::extensions_goldilocks::Degree3GoldilocksExtensionField;
        use crate::field::traits::IsField as _;

        let a = [[1u64, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
        let b = [[13u64, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]];
        unsafe {
            let pa = (
                load4([a[0][0], a[1][0], a[2][0], a[3][0]]),
                load4([a[0][1], a[1][1], a[2][1], a[3][1]]),
                load4([a[0][2], a[1][2], a[2][2], a[3][2]]),
            );
            let pb = (
                load4([b[0][0], b[1][0], b[2][0], b[3][0]]),
                load4([b[0][1], b[1][1], b[2][1], b[3][1]]),
                load4([b[0][2], b[1][2], b[2][2], b[3][2]]),
            );
            let (r0, r1, r2) = ext3_add4(pa, pb);
            let c0 = store4(r0);
            let c1 = store4(r1);
            let c2 = store4(r2);
            for i in 0..4 {
                type F = Degree3GoldilocksExtensionField;
                let ea = a[i].map(FpE::from);
                let eb = b[i].map(FpE::from);
                let expected = F::add(&ea, &eb);
                assert_eq!(
                    canon(c0[i]),
                    canon(*expected[0].value()),
                    "ext3_add4 c0 lane={i}"
                );
                assert_eq!(
                    canon(c1[i]),
                    canon(*expected[1].value()),
                    "ext3_add4 c1 lane={i}"
                );
                assert_eq!(
                    canon(c2[i]),
                    canon(*expected[2].value()),
                    "ext3_add4 c2 lane={i}"
                );
            }
        }
    }

    #[test]
    fn test_ext3_mul4() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use super::super::extensions_goldilocks::Degree3GoldilocksExtensionField;
        use crate::field::traits::IsField as _;

        let a = [
            [1u64, 2, 3],
            [100, 200, 300],
            [GOLDILOCKS_PRIME - 1, 42, 7],
            [0xDEADBEEF, 0xCAFEBABE, 12345],
        ];
        let b = [
            [4u64, 5, 6],
            [7, 8, 9],
            [10, GOLDILOCKS_PRIME - 1, 3],
            [999, 888, 777],
        ];
        unsafe {
            let pa = (
                load4([a[0][0], a[1][0], a[2][0], a[3][0]]),
                load4([a[0][1], a[1][1], a[2][1], a[3][1]]),
                load4([a[0][2], a[1][2], a[2][2], a[3][2]]),
            );
            let pb = (
                load4([b[0][0], b[1][0], b[2][0], b[3][0]]),
                load4([b[0][1], b[1][1], b[2][1], b[3][1]]),
                load4([b[0][2], b[1][2], b[2][2], b[3][2]]),
            );
            let (r0, r1, r2) = ext3_mul4(pa, pb);
            let c0 = store4(r0);
            let c1 = store4(r1);
            let c2 = store4(r2);
            for i in 0..4 {
                type F = Degree3GoldilocksExtensionField;
                let ea = a[i].map(FpE::from);
                let eb = b[i].map(FpE::from);
                let expected = F::mul(&ea, &eb);
                assert_eq!(
                    canon(c0[i]),
                    canon(*expected[0].value()),
                    "ext3_mul4 c0 lane={i}"
                );
                assert_eq!(
                    canon(c1[i]),
                    canon(*expected[1].value()),
                    "ext3_mul4 c1 lane={i}"
                );
                assert_eq!(
                    canon(c2[i]),
                    canon(*expected[2].value()),
                    "ext3_mul4 c2 lane={i}"
                );
            }
        }
    }

    #[test]
    fn test_ext3_scalar_mul4() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use super::super::extensions_goldilocks::Degree3GoldilocksExtensionField;
        use super::super::u64_goldilocks::GoldilocksField;
        use crate::field::traits::{IsField as _, IsSubFieldOf};

        let scalars = [5u64, 100, GOLDILOCKS_PRIME - 1, 0xDEADBEEF];
        let ext = [[1u64, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
        unsafe {
            let s = load4(scalars);
            let pe = (
                load4([ext[0][0], ext[1][0], ext[2][0], ext[3][0]]),
                load4([ext[0][1], ext[1][1], ext[2][1], ext[3][1]]),
                load4([ext[0][2], ext[1][2], ext[2][2], ext[3][2]]),
            );
            let (r0, r1, r2) = ext3_scalar_mul4(s, pe);
            let c0 = store4(r0);
            let c1 = store4(r1);
            let c2 = store4(r2);
            for i in 0..4 {
                let ee = ext[i].map(FpE::from);
                let expected =
                    <GoldilocksField as IsSubFieldOf<Degree3GoldilocksExtensionField>>::mul(
                        &scalars[i],
                        &ee,
                    );
                assert_eq!(
                    canon(c0[i]),
                    canon(*expected[0].value()),
                    "ext3_scalar c0 lane={i}"
                );
                assert_eq!(
                    canon(c1[i]),
                    canon(*expected[1].value()),
                    "ext3_scalar c1 lane={i}"
                );
                assert_eq!(
                    canon(c2[i]),
                    canon(*expected[2].value()),
                    "ext3_scalar c2 lane={i}"
                );
            }
        }
    }
}
