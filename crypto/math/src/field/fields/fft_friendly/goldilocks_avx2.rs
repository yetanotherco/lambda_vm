//! AVX2-accelerated Goldilocks field arithmetic.
//!
//! Provides `PackedGoldilocks4`, holding 4 Goldilocks field elements with
//! parallel AVX2 operations. Used by the FFT butterfly kernels.
//!
//! **Important**: `_mm256_cmpgt_epi64` is signed. For unsigned overflow
//! detection we compare `(a_signed > sum_signed)` which works because
//! wrapping addition flips the sign bit on overflow.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::ops::{Add, Mul, Sub};

use super::u64_goldilocks::GOLDILOCKS_PRIME;

const P: u64 = GOLDILOCKS_PRIME;
const EPSILON: u64 = 0xFFFF_FFFF;

/// 4-wide packed Goldilocks field elements using AVX2 `__m256i`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PackedGoldilocks4(__m256i);

impl PackedGoldilocks4 {
    #[inline(always)]
    #[target_feature(enable = "avx2")]
    pub unsafe fn load(slice: &[u64]) -> Self {
        debug_assert!(slice.len() >= 4);
        Self(_mm256_loadu_si256(slice.as_ptr() as *const __m256i))
    }

    #[inline(always)]
    #[target_feature(enable = "avx2")]
    pub unsafe fn store(self, slice: &mut [u64]) {
        debug_assert!(slice.len() >= 4);
        _mm256_storeu_si256(slice.as_mut_ptr() as *mut __m256i, self.0)
    }

    #[inline(always)]
    #[target_feature(enable = "avx2")]
    pub unsafe fn broadcast(val: u64) -> Self {
        Self(_mm256_set1_epi64x(val as i64))
    }

    #[inline(always)]
    #[target_feature(enable = "avx2")]
    pub unsafe fn square(self) -> Self {
        self * self
    }
}

impl Add for PackedGoldilocks4 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { add_avx2(self, rhs) }
    }
}

impl Sub for PackedGoldilocks4 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { sub_avx2(self, rhs) }
    }
}

impl Mul for PackedGoldilocks4 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { mul_avx2(self, rhs) }
    }
}

#[inline(always)]
#[target_feature(enable = "avx2")]
unsafe fn add_avx2(a: PackedGoldilocks4, b: PackedGoldilocks4) -> PackedGoldilocks4 {
    let epsilon = _mm256_set1_epi64x(EPSILON as i64);
    let sum = _mm256_add_epi64(a.0, b.0);
    // Unsigned overflow: sum < a (signed cmpgt detects wrapping)
    let overflow = _mm256_cmpgt_epi64(a.0, sum);
    let correction = _mm256_and_si256(overflow, epsilon);
    let result = _mm256_add_epi64(sum, correction);
    // Second overflow from adding epsilon
    let overflow2 = _mm256_cmpgt_epi64(sum, result);
    let correction2 = _mm256_and_si256(overflow2, epsilon);
    PackedGoldilocks4(_mm256_add_epi64(result, correction2))
}

#[inline(always)]
#[target_feature(enable = "avx2")]
unsafe fn sub_avx2(a: PackedGoldilocks4, b: PackedGoldilocks4) -> PackedGoldilocks4 {
    let epsilon = _mm256_set1_epi64x(EPSILON as i64);
    let diff = _mm256_sub_epi64(a.0, b.0);
    // Unsigned underflow: b > a (signed cmpgt)
    let underflow = _mm256_cmpgt_epi64(b.0, a.0);
    let correction = _mm256_and_si256(underflow, epsilon);
    let result = _mm256_sub_epi64(diff, correction);
    // Second underflow from subtracting epsilon
    let underflow2 = _mm256_cmpgt_epi64(diff, result);
    let correction2 = _mm256_and_si256(underflow2, epsilon);
    PackedGoldilocks4(_mm256_sub_epi64(result, correction2))
}

#[inline(always)]
#[target_feature(enable = "avx2")]
unsafe fn mul_avx2(a: PackedGoldilocks4, b: PackedGoldilocks4) -> PackedGoldilocks4 {
    let low_mask = _mm256_set1_epi64x(0xFFFF_FFFF_i64);

    let a_lo = _mm256_and_si256(a.0, low_mask);
    let a_hi = _mm256_srli_epi64(a.0, 32);
    let b_lo = _mm256_and_si256(b.0, low_mask);
    let b_hi = _mm256_srli_epi64(b.0, 32);

    let p_ll = _mm256_mul_epu32(a_lo, b_lo);
    let p_lh = _mm256_mul_epu32(a_lo, b_hi);
    let p_hl = _mm256_mul_epu32(a_hi, b_lo);
    let p_hh = _mm256_mul_epu32(a_hi, b_hi);

    // Middle term
    let p_mid = _mm256_add_epi64(p_lh, p_hl);
    let p_mid_lo = _mm256_and_si256(p_mid, low_mask);
    let p_mid_hi = _mm256_srli_epi64(p_mid, 32);

    // Detect carry from p_lh + p_hl overflow (at bit 64 of mid, i.e. bit 96 total)
    // When p_mid < p_lh, overflow happened
    let mid_overflow = _mm256_cmpgt_epi64(p_lh, p_mid);
    let mid_carry = _mm256_srli_epi64(mid_overflow, 63); // -1 >> 63 = 1 on overflow lanes

    // lo = p_ll + (p_mid_lo << 32)
    let mid_lo_shifted = _mm256_slli_epi64(p_mid_lo, 32);
    let lo = _mm256_add_epi64(p_ll, mid_lo_shifted);
    let lo_overflow = _mm256_cmpgt_epi64(p_ll, lo);
    let lo_carry = _mm256_srli_epi64(lo_overflow, 63);

    // hi = p_hh + p_mid_hi + mid_carry_shifted + lo_carry
    let mid_carry_shifted = _mm256_slli_epi64(mid_carry, 32);
    let hi = _mm256_add_epi64(
        _mm256_add_epi64(p_hh, p_mid_hi),
        _mm256_add_epi64(mid_carry_shifted, lo_carry),
    );

    reduce_128_avx2(lo, hi)
}

#[inline(always)]
#[target_feature(enable = "avx2")]
unsafe fn reduce_128_avx2(lo: __m256i, hi: __m256i) -> PackedGoldilocks4 {
    let epsilon = _mm256_set1_epi64x(EPSILON as i64);
    let low_mask = epsilon; // same bit pattern

    let hi_lo = _mm256_and_si256(hi, low_mask);
    let hi_hi = _mm256_srli_epi64(hi, 32);

    // t0 = lo - hi_hi
    let borrow = _mm256_cmpgt_epi64(hi_hi, lo);
    let t0 = _mm256_sub_epi64(lo, hi_hi);
    let borrow_correction = _mm256_and_si256(borrow, epsilon);
    let t0 = _mm256_sub_epi64(t0, borrow_correction);

    // t1 = hi_lo * EPSILON = (hi_lo << 32) - hi_lo
    let hi_lo_shifted = _mm256_slli_epi64(hi_lo, 32);
    let t1 = _mm256_sub_epi64(hi_lo_shifted, hi_lo);

    // result = t0 + t1
    let sum = _mm256_add_epi64(t0, t1);
    let carry = _mm256_cmpgt_epi64(t0, sum);
    let carry_correction = _mm256_and_si256(carry, epsilon);
    PackedGoldilocks4(_mm256_add_epi64(sum, carry_correction))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
    use crate::field::traits::IsField;

    unsafe fn to_array(p: PackedGoldilocks4) -> [u64; 4] {
        let mut out = [0u64; 4];
        p.store(&mut out);
        out
    }

    #[test]
    fn add_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let cases: [(u64, u64); 4] = [(1, 3), (P - 1, 2), (P - 1, P - 1), (0, 0)];
        for (a, b) in cases {
            unsafe {
                let pa = PackedGoldilocks4::broadcast(a);
                let pb = PackedGoldilocks4::broadcast(b);
                let out = to_array(pa + pb);
                assert_eq!(out[0], GoldilocksField::add(&a, &b), "add({a}, {b})");
                assert_eq!(out[3], GoldilocksField::add(&a, &b));
            }
        }
    }

    #[test]
    fn sub_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let cases: [(u64, u64); 4] = [(5, 3), (3, 5), (0, P - 1), (P - 1, P - 1)];
        for (a, b) in cases {
            unsafe {
                let pa = PackedGoldilocks4::broadcast(a);
                let pb = PackedGoldilocks4::broadcast(b);
                let out = to_array(pa - pb);
                assert_eq!(out[0], GoldilocksField::sub(&a, &b), "sub({a}, {b})");
            }
        }
    }

    #[test]
    fn mul_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let cases: [(u64, u64); 6] = [
            (0, 0),
            (1, 1),
            (2, 3),
            (EPSILON, EPSILON),
            (P - 1, 2),
            (P - 1, P - 1),
        ];
        for (a, b) in cases {
            unsafe {
                let pa = PackedGoldilocks4::broadcast(a);
                let pb = PackedGoldilocks4::broadcast(b);
                let out = to_array(pa * pb);
                assert_eq!(out[0], GoldilocksField::mul(&a, &b), "mul({a}, {b})");
            }
        }
    }

    #[test]
    fn load_store_roundtrip() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let data = [42u64, 99, 7, 13];
        unsafe {
            let p = PackedGoldilocks4::load(&data);
            let out = to_array(p);
            assert_eq!(out, data);
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
    use crate::field::traits::IsField;
    use proptest::prelude::*;

    fn arb_fe() -> impl Strategy<Value = u64> {
        0..P
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10000))]

        #[test]
        fn prop_add(a in arb_fe(), b in arb_fe()) {
            if !is_x86_feature_detected!("avx2") { return Ok(()); }
            unsafe {
                let pa = PackedGoldilocks4::broadcast(a);
                let pb = PackedGoldilocks4::broadcast(b);
                let mut out = [0u64; 4];
                (pa + pb).store(&mut out);
                prop_assert_eq!(out[0], GoldilocksField::add(&a, &b));
            }
        }

        #[test]
        fn prop_sub(a in arb_fe(), b in arb_fe()) {
            if !is_x86_feature_detected!("avx2") { return Ok(()); }
            unsafe {
                let pa = PackedGoldilocks4::broadcast(a);
                let pb = PackedGoldilocks4::broadcast(b);
                let mut out = [0u64; 4];
                (pa - pb).store(&mut out);
                prop_assert_eq!(out[0], GoldilocksField::sub(&a, &b));
            }
        }

        #[test]
        fn prop_mul(a in arb_fe(), b in arb_fe()) {
            if !is_x86_feature_detected!("avx2") { return Ok(()); }
            unsafe {
                let pa = PackedGoldilocks4::broadcast(a);
                let pb = PackedGoldilocks4::broadcast(b);
                let mut out = [0u64; 4];
                (pa * pb).store(&mut out);
                prop_assert_eq!(out[0], GoldilocksField::mul(&a, &b));
            }
        }
    }
}
