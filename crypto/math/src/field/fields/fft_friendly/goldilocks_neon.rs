//! SIMD-accelerated Goldilocks field arithmetic using ARM NEON.
//!
//! Provides `PackedGoldilocks2`, holding 2 Goldilocks field elements with
//! parallel NEON operations. Used by the FFT butterfly kernels.

use core::arch::aarch64::*;
use core::ops::{Add, Mul, Neg, Sub};

use super::u64_goldilocks::GOLDILOCKS_PRIME;

const P: u64 = GOLDILOCKS_PRIME;
const EPSILON: u64 = 0xFFFF_FFFF;

/// 2-wide packed Goldilocks field elements using NEON `uint64x2_t`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PackedGoldilocks2(uint64x2_t);

impl PackedGoldilocks2 {
    #[inline(always)]
    pub fn load(slice: &[u64]) -> Self {
        debug_assert!(slice.len() >= 2);
        unsafe { Self(vld1q_u64(slice.as_ptr())) }
    }

    #[inline(always)]
    pub fn store(self, slice: &mut [u64]) {
        debug_assert!(slice.len() >= 2);
        unsafe { vst1q_u64(slice.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    pub fn broadcast(val: u64) -> Self {
        unsafe { Self(vdupq_n_u64(val)) }
    }

    #[inline(always)]
    pub fn square(self) -> Self {
        self * self
    }

    /// Reduce values in [0, 2P) to canonical [0, P).
    #[inline(always)]
    fn reduce(self) -> Self {
        unsafe {
            let p = vdupq_n_u64(P);
            let reduced = vsubq_u64(self.0, p);
            let mask = vcgeq_u64(self.0, p);
            Self(vbslq_u64(mask, reduced, self.0))
        }
    }
}

impl Add for PackedGoldilocks2 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            let sum = vaddq_u64(self.0, rhs.0);
            let overflow_mask = vcltq_u64(sum, self.0);
            let epsilon = vdupq_n_u64(EPSILON);
            let correction = vandq_u64(overflow_mask, epsilon);
            let result = vaddq_u64(sum, correction);
            Self(result).reduce()
        }
    }
}

impl Sub for PackedGoldilocks2 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            let underflow_mask = vcltq_u64(self.0, rhs.0);
            let diff = vsubq_u64(self.0, rhs.0);
            let p = vdupq_n_u64(P);
            let correction = vandq_u64(underflow_mask, p);
            Self(vaddq_u64(diff, correction))
        }
    }
}

impl Neg for PackedGoldilocks2 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            let p = vdupq_n_u64(P);
            let zero = vdupq_n_u64(0);
            let is_zero = vceqq_u64(self.0, zero);
            let neg = vsubq_u64(p, self.0);
            Self(vbslq_u64(is_zero, zero, neg))
        }
    }
}

impl Mul for PackedGoldilocks2 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let a_32: uint32x4_t = vreinterpretq_u32_u64(self.0);
            let b_32: uint32x4_t = vreinterpretq_u32_u64(rhs.0);

            let a_lo_32 = vuzp1q_u32(a_32, a_32);
            let a_hi_32 = vuzp2q_u32(a_32, a_32);
            let b_lo_32 = vuzp1q_u32(b_32, b_32);
            let b_hi_32 = vuzp2q_u32(b_32, b_32);

            let a_lo = vget_low_u32(a_lo_32);
            let a_hi = vget_low_u32(a_hi_32);
            let b_lo = vget_low_u32(b_lo_32);
            let b_hi = vget_low_u32(b_hi_32);

            let p_ll: uint64x2_t = vmull_u32(a_lo, b_lo);
            let p_lh: uint64x2_t = vmull_u32(a_lo, b_hi);
            let p_hl: uint64x2_t = vmull_u32(a_hi, b_lo);
            let p_hh: uint64x2_t = vmull_u32(a_hi, b_hi);

            let p_mid = vaddq_u64(p_lh, p_hl);
            let mid_overflow = vcltq_u64(p_mid, p_lh);
            let mid_carry = vandq_u64(mid_overflow, vdupq_n_u64(1));

            let mid_lo_shifted = vshlq_n_u64(p_mid, 32);
            let lo = vaddq_u64(p_ll, mid_lo_shifted);
            let lo_overflow = vcltq_u64(lo, p_ll);
            let lo_carry = vandq_u64(lo_overflow, vdupq_n_u64(1));

            let mid_hi = vshrq_n_u64(p_mid, 32);
            let mid_carry_shifted = vshlq_n_u64(mid_carry, 32);
            let hi = vaddq_u64(
                vaddq_u64(vaddq_u64(p_hh, mid_hi), mid_carry_shifted),
                lo_carry,
            );

            reduce_128_neon(lo, hi)
        }
    }
}

#[inline(always)]
unsafe fn reduce_128_neon(lo: uint64x2_t, hi: uint64x2_t) -> PackedGoldilocks2 {
    unsafe {
        let epsilon = vdupq_n_u64(EPSILON);

        let hi_lo = vandq_u64(hi, epsilon);
        let hi_hi = vshrq_n_u64(hi, 32);

        // t0 = lo - hi_hi
        let borrow_mask = vcltq_u64(lo, hi_hi);
        let t0 = vsubq_u64(lo, hi_hi);
        let borrow_correction = vandq_u64(borrow_mask, epsilon);
        let t0 = vsubq_u64(t0, borrow_correction);

        // t1 = hi_lo * EPSILON = (hi_lo << 32) - hi_lo
        let hi_lo_shifted = vshlq_n_u64(hi_lo, 32);
        let t1 = vsubq_u64(hi_lo_shifted, hi_lo);

        // result = t0 + t1
        let sum = vaddq_u64(t0, t1);
        let carry_mask = vcltq_u64(sum, t0);
        let carry_correction = vandq_u64(carry_mask, epsilon);
        let result = vaddq_u64(sum, carry_correction);

        PackedGoldilocks2(result).reduce()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
    use crate::field::traits::{IsField, IsPrimeField};

    /// Canonicalize to [0, P) for comparison. Scalar ops may return non-canonical values.
    fn canon(x: u64) -> u64 {
        GoldilocksField::canonical(&x)
    }

    fn scalar_add(a: u64, b: u64) -> u64 {
        GoldilocksField::add(&a, &b)
    }
    fn scalar_sub(a: u64, b: u64) -> u64 {
        GoldilocksField::sub(&a, &b)
    }
    fn scalar_mul(a: u64, b: u64) -> u64 {
        GoldilocksField::mul(&a, &b)
    }
    fn scalar_neg(a: u64) -> u64 {
        GoldilocksField::neg(&a)
    }

    #[test]
    fn add_matches_scalar() {
        let cases: [(u64, u64); 4] = [(1, 3), (P - 1, 2), (P - 1, P - 1), (0, 0)];
        for (a, b) in cases {
            let pa = PackedGoldilocks2::broadcast(a);
            let pb = PackedGoldilocks2::broadcast(b);
            let mut out = [0u64; 2];
            (pa + pb).store(&mut out);
            assert_eq!(canon(out[0]), canon(scalar_add(a, b)), "add({a}, {b})");
        }
    }

    #[test]
    fn sub_matches_scalar() {
        let cases: [(u64, u64); 4] = [(5, 3), (3, 5), (0, P - 1), (P - 1, P - 1)];
        for (a, b) in cases {
            let pa = PackedGoldilocks2::broadcast(a);
            let pb = PackedGoldilocks2::broadcast(b);
            let mut out = [0u64; 2];
            (pa - pb).store(&mut out);
            assert_eq!(canon(out[0]), canon(scalar_sub(a, b)), "sub({a}, {b})");
        }
    }

    #[test]
    fn neg_matches_scalar() {
        for a in [0u64, 1, 5, P - 1, EPSILON] {
            let pa = PackedGoldilocks2::broadcast(a);
            let mut out = [0u64; 2];
            (-pa).store(&mut out);
            assert_eq!(canon(out[0]), canon(scalar_neg(a)), "neg({a})");
        }
    }

    #[test]
    fn mul_matches_scalar() {
        let cases: [(u64, u64); 6] = [
            (0, 0),
            (1, 1),
            (2, 3),
            (EPSILON, EPSILON),
            (P - 1, 2),
            (P - 1, P - 1),
        ];
        for (a, b) in cases {
            let pa = PackedGoldilocks2::broadcast(a);
            let pb = PackedGoldilocks2::broadcast(b);
            let mut out = [0u64; 2];
            (pa * pb).store(&mut out);
            assert_eq!(canon(out[0]), canon(scalar_mul(a, b)), "mul({a}, {b})");
        }
    }

    #[test]
    fn load_store_roundtrip() {
        let data = [42u64, 99];
        let p = PackedGoldilocks2::load(&data);
        let mut out = [0u64; 2];
        p.store(&mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn square_matches_mul() {
        for a in [7u64, P - 1, EPSILON, 1 << 32] {
            let pa = PackedGoldilocks2::broadcast(a);
            let mut sq = [0u64; 2];
            let mut ml = [0u64; 2];
            pa.square().store(&mut sq);
            (pa * pa).store(&mut ml);
            assert_eq!(sq, ml, "square({a})");
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
    use crate::field::traits::{IsField, IsPrimeField};
    use proptest::prelude::*;

    fn canon(x: u64) -> u64 {
        GoldilocksField::canonical(&x)
    }

    fn arb_fe() -> impl Strategy<Value = u64> {
        0..P
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10000))]

        #[test]
        fn prop_add(a0 in arb_fe(), a1 in arb_fe(), b0 in arb_fe(), b1 in arb_fe()) {
            let a = PackedGoldilocks2::load(&[a0, a1]);
            let b = PackedGoldilocks2::load(&[b0, b1]);
            let mut out = [0u64; 2];
            (a + b).store(&mut out);
            prop_assert_eq!(canon(out[0]), canon(GoldilocksField::add(&a0, &b0)));
            prop_assert_eq!(canon(out[1]), canon(GoldilocksField::add(&a1, &b1)));
        }

        #[test]
        fn prop_sub(a0 in arb_fe(), a1 in arb_fe(), b0 in arb_fe(), b1 in arb_fe()) {
            let a = PackedGoldilocks2::load(&[a0, a1]);
            let b = PackedGoldilocks2::load(&[b0, b1]);
            let mut out = [0u64; 2];
            (a - b).store(&mut out);
            prop_assert_eq!(canon(out[0]), canon(GoldilocksField::sub(&a0, &b0)));
            prop_assert_eq!(canon(out[1]), canon(GoldilocksField::sub(&a1, &b1)));
        }

        #[test]
        fn prop_mul(a0 in arb_fe(), a1 in arb_fe(), b0 in arb_fe(), b1 in arb_fe()) {
            let a = PackedGoldilocks2::load(&[a0, a1]);
            let b = PackedGoldilocks2::load(&[b0, b1]);
            let mut out = [0u64; 2];
            (a * b).store(&mut out);
            prop_assert_eq!(canon(out[0]), canon(GoldilocksField::mul(&a0, &b0)));
            prop_assert_eq!(canon(out[1]), canon(GoldilocksField::mul(&a1, &b1)));
        }
    }
}
