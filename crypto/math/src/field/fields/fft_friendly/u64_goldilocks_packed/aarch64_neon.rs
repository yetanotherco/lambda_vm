//! NEON packed Goldilocks field arithmetic (WIDTH=2).
//!
//! Uses 128-bit uint64x2_t registers holding 2 x 64-bit field elements.
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::aarch64::*;
use core::mem::transmute;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks::{GoldilocksField, GOLDILOCKS_PRIME};
use crate::field::packed::PackedField;

const WIDTH: usize = 2;
const EPSILON: u64 = 0xFFFF_FFFF;

/// Packed Goldilocks field element holding 2 elements in a NEON register.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct PackedGoldilocksNeon(pub [FieldElement<GoldilocksField>; WIDTH]);

impl Default for PackedGoldilocksNeon {
    fn default() -> Self {
        Self([FieldElement::zero(); WIDTH])
    }
}

impl PartialEq for PackedGoldilocksNeon {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for PackedGoldilocksNeon {}

impl PackedGoldilocksNeon {
    #[inline(always)]
    fn to_vector(self) -> uint64x2_t {
        unsafe { transmute(self) }
    }

    #[inline(always)]
    fn from_vector(v: uint64x2_t) -> Self {
        unsafe { transmute(v) }
    }
}

// ---- Constants ----

const SIGN_BIT: uint64x2_t = unsafe { transmute([i64::MIN as u64; WIDTH]) };
const FIELD_ORDER_VEC: uint64x2_t = unsafe { transmute([GOLDILOCKS_PRIME; WIDTH]) };
const SHIFTED_FIELD_ORDER: uint64x2_t =
    unsafe { transmute([GOLDILOCKS_PRIME ^ (i64::MIN as u64); WIDTH]) };
const EPSILON_VEC: uint64x2_t = unsafe { transmute([EPSILON; WIDTH]) };

// ---- Shifted-representation helpers (same approach as AVX2) ----

#[inline(always)]
unsafe fn shift(x: uint64x2_t) -> uint64x2_t {
    veorq_u64(x, SIGN_BIT)
}

#[inline(always)]
unsafe fn canonicalize_s(x_s: uint64x2_t) -> uint64x2_t {
    // mask = all-ones if P > x (signed compare in shifted domain)
    let mask: uint64x2_t = transmute(vcgtq_s64(
        transmute(SHIFTED_FIELD_ORDER),
        transmute(x_s),
    ));
    // wrapback = EPSILON if x >= P, else 0
    let wrapback = vbicq_u64(EPSILON_VEC, mask);
    vaddq_u64(x_s, wrapback)
}

#[inline(always)]
unsafe fn add_no_double_overflow_s(x: uint64x2_t, y_s: uint64x2_t) -> uint64x2_t {
    let res_s = vaddq_u64(x, y_s);
    let mask: uint64x2_t = transmute(vcgtq_s64(transmute(y_s), transmute(res_s)));
    let correction = vshrq_n_u64::<32>(mask);
    vaddq_u64(res_s, correction)
}

/// Packed modular addition
#[inline(always)]
unsafe fn add_neon(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    let b_s = canonicalize_s(shift(b));
    let res_s = add_no_double_overflow_s(a, b_s);
    shift(res_s)
}

/// Packed modular subtraction
#[inline(always)]
unsafe fn sub_neon(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    let a_s = shift(a);
    let b_s = canonicalize_s(shift(b));
    let mask: uint64x2_t = transmute(vcgtq_s64(transmute(b_s), transmute(a_s)));
    let correction = vshrq_n_u64::<32>(mask);
    let res = vsubq_u64(a_s, b_s);
    // No shift here: (a XOR S) - (b XOR S) = a - b (shifts cancel in subtraction)
    vsubq_u64(res, correction)
}

/// Packed modular negation
#[inline(always)]
unsafe fn neg_neon(a: uint64x2_t) -> uint64x2_t {
    let a_canon = shift(canonicalize_s(shift(a)));
    sub_neon(FIELD_ORDER_VEC, a_canon)
}

// ---- Multiply helpers ----

#[inline(always)]
unsafe fn sub_small_s(x_s: uint64x2_t, y_small: uint64x2_t) -> uint64x2_t {
    let res_s = vsubq_u64(x_s, y_small);
    let mask: uint64x2_t = transmute(vcgtq_s64(transmute(res_s), transmute(x_s)));
    let correction = vshrq_n_u64::<32>(mask);
    vsubq_u64(res_s, correction)
}

#[inline(always)]
unsafe fn add_small_s(x_s: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    let res_s = vaddq_u64(x_s, y);
    let mask: uint64x2_t = transmute(vcgtq_s64(transmute(x_s), transmute(res_s)));
    let correction = vshrq_n_u64::<32>(mask);
    vaddq_u64(res_s, correction)
}

/// Reduce 128-bit (hi, lo) to Goldilocks element.
/// x mod P = lo - hi_hi + hi_lo * EPSILON
#[inline(always)]
unsafe fn reduce128_neon(hi: uint64x2_t, lo: uint64x2_t) -> uint64x2_t {
    let lo_s = shift(lo);
    let hi_hi = vshrq_n_u64::<32>(hi);
    let lo1_s = sub_small_s(lo_s, hi_hi);
    // hi_lo * EPSILON = hi_lo * (2^32 - 1) = (hi_lo << 32) - hi_lo
    // hi_lo < 2^32, so hi_lo << 32 < 2^64 (no overflow)
    let hi_lo = vandq_u64(hi, EPSILON_VEC);
    let t1 = vsubq_u64(vshlq_n_u64::<32>(hi_lo), hi_lo);
    let lo2_s = add_small_s(lo1_s, t1);
    shift(lo2_s)
}

/// NEON multiply using inline assembly for native 64x64->128 bit multiply.
/// AArch64 has `mul` (low 64) and `umulh` (high 64) which together give 128-bit result.
#[inline(always)]
unsafe fn mul_neon(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    // Extract lanes
    let a0 = vgetq_lane_u64::<0>(a);
    let a1 = vgetq_lane_u64::<1>(a);
    let b0 = vgetq_lane_u64::<0>(b);
    let b1 = vgetq_lane_u64::<1>(b);

    let lo0: u64;
    let hi0: u64;
    let lo1: u64;
    let hi1: u64;

    // Native 64x64->128 multiply using inline assembly
    core::arch::asm!(
        "mul {lo0}, {a0}, {b0}",
        "umulh {hi0}, {a0}, {b0}",
        "mul {lo1}, {a1}, {b1}",
        "umulh {hi1}, {a1}, {b1}",
        a0 = in(reg) a0,
        b0 = in(reg) b0,
        a1 = in(reg) a1,
        b1 = in(reg) b1,
        lo0 = out(reg) lo0,
        hi0 = out(reg) hi0,
        lo1 = out(reg) lo1,
        hi1 = out(reg) hi1,
        options(pure, nomem, nostack),
    );

    let lo_vec = vcombine_u64(vcreate_u64(lo0), vcreate_u64(lo1));
    let hi_vec = vcombine_u64(vcreate_u64(hi0), vcreate_u64(hi1));
    reduce128_neon(hi_vec, lo_vec)
}

/// NEON square using inline assembly.
#[inline(always)]
unsafe fn square_neon(a: uint64x2_t) -> uint64x2_t {
    mul_neon(a, a)
}

// ---- Operator impls ----

impl Add for PackedGoldilocksNeon {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_vector(unsafe { add_neon(self.to_vector(), rhs.to_vector()) })
    }
}

impl Sub for PackedGoldilocksNeon {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_vector(unsafe { sub_neon(self.to_vector(), rhs.to_vector()) })
    }
}

impl Neg for PackedGoldilocksNeon {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_vector(unsafe { neg_neon(self.to_vector()) })
    }
}

impl Mul for PackedGoldilocksNeon {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self::from_vector(mul_neon(self.to_vector(), rhs.to_vector())) }
    }
}

impl AddAssign for PackedGoldilocksNeon {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for PackedGoldilocksNeon {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for PackedGoldilocksNeon {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// ---- PackedField impl ----

unsafe impl PackedField for PackedGoldilocksNeon {
    type Scalar = GoldilocksField;
    const WIDTH: usize = WIDTH;

    #[inline(always)]
    fn from_fn(mut f: impl FnMut(usize) -> FieldElement<GoldilocksField>) -> Self {
        Self([f(0), f(1)])
    }

    #[inline(always)]
    fn from_slice(slice: &[FieldElement<GoldilocksField>]) -> Self {
        Self([slice[0], slice[1]])
    }

    #[inline(always)]
    fn as_slice(&self) -> &[FieldElement<GoldilocksField>] {
        &self.0
    }

    #[inline(always)]
    fn as_slice_mut(&mut self) -> &mut [FieldElement<GoldilocksField>] {
        &mut self.0
    }

    #[inline(always)]
    fn zero() -> Self {
        Self([FieldElement::zero(); WIDTH])
    }

    #[inline(always)]
    fn ones() -> Self {
        Self([FieldElement::one(); WIDTH])
    }

    #[inline(always)]
    fn broadcast(value: FieldElement<GoldilocksField>) -> Self {
        Self([value; WIDTH])
    }

    fn square(&self) -> Self {
        Self::from_vector(unsafe { square_neon(self.to_vector()) })
    }

    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        unsafe {
            let a = self.to_vector();
            let b = other.to_vector();
            match block_len {
                1 => {
                    // [a0,a1] x [b0,b1] -> [a0,b0], [a1,b1]
                    let lo = vzip1q_u64(a, b);
                    let hi = vzip2q_u64(a, b);
                    (Self::from_vector(lo), Self::from_vector(hi))
                }
                2 => (*self, other), // WIDTH = identity
                _ => panic!("block_len must be 1 or 2 for WIDTH=2"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type FE = FieldElement<GoldilocksField>;

    fn random_packed() -> PackedGoldilocksNeon {
        PackedGoldilocksNeon::from_fn(|i| FE::from((i as u64 + 1) * 0x123456789ABCDEFu64))
    }

    #[test]
    fn test_packed_add_matches_scalar() {
        let a = random_packed();
        let b = PackedGoldilocksNeon::from_fn(|i| FE::from((i as u64 + 5) * 0xFEDCBA987654321u64));
        let packed_sum = a + b;
        let scalar_sum = PackedGoldilocksNeon::from_fn(|i| a.as_slice()[i] + b.as_slice()[i]);
        assert_eq!(packed_sum.as_slice(), scalar_sum.as_slice());
    }

    #[test]
    fn test_packed_sub_matches_scalar() {
        let a = random_packed();
        let b = PackedGoldilocksNeon::from_fn(|i| FE::from((i as u64 + 5) * 0xFEDCBA987654321u64));
        let packed_diff = a - b;
        let scalar_diff = PackedGoldilocksNeon::from_fn(|i| a.as_slice()[i] - b.as_slice()[i]);
        assert_eq!(packed_diff.as_slice(), scalar_diff.as_slice());
    }

    #[test]
    fn test_packed_neg_matches_scalar() {
        let a = random_packed();
        let packed_neg = -a;
        let scalar_neg = PackedGoldilocksNeon::from_fn(|i| -a.as_slice()[i]);
        assert_eq!(packed_neg.as_slice(), scalar_neg.as_slice());
    }

    #[test]
    fn test_packed_add_overflow() {
        let a = PackedGoldilocksNeon::broadcast(FE::from(GOLDILOCKS_PRIME - 1));
        let b = PackedGoldilocksNeon::broadcast(FE::from(2u64));
        let sum = a + b;
        assert_eq!(sum.as_slice()[0], FE::from(1u64));
    }

    #[test]
    fn test_packed_sub_underflow() {
        let a = PackedGoldilocksNeon::broadcast(FE::from(1u64));
        let b = PackedGoldilocksNeon::broadcast(FE::from(3u64));
        let diff = a - b;
        assert_eq!(diff.as_slice()[0], FE::from(GOLDILOCKS_PRIME - 2));
    }

    #[test]
    fn test_packed_mul_matches_scalar() {
        let a = random_packed();
        let b = PackedGoldilocksNeon::from_fn(|i| FE::from((i as u64 + 5) * 0xFEDCBA987654321u64));
        let packed_prod = a * b;
        let scalar_prod = PackedGoldilocksNeon::from_fn(|i| a.as_slice()[i] * b.as_slice()[i]);
        assert_eq!(packed_prod.as_slice(), scalar_prod.as_slice());
    }

    #[test]
    fn test_packed_square_matches_mul() {
        let a = random_packed();
        let packed_sq = a.square();
        let packed_mul = a * a;
        assert_eq!(packed_sq.as_slice(), packed_mul.as_slice());
    }

    #[test]
    fn test_packed_mul_identity() {
        let a = random_packed();
        let one = PackedGoldilocksNeon::ones();
        assert_eq!((a * one).as_slice(), a.as_slice());
    }

    #[test]
    fn test_packed_mul_zero() {
        let a = random_packed();
        let zero = PackedGoldilocksNeon::zero();
        let result = a * zero;
        for lane in result.as_slice() {
            assert_eq!(*lane, FE::zero());
        }
    }

    #[test]
    fn test_packed_mul_large_values() {
        let a = PackedGoldilocksNeon::broadcast(FE::from(GOLDILOCKS_PRIME - 1));
        let b = PackedGoldilocksNeon::broadcast(FE::from(GOLDILOCKS_PRIME - 1));
        let result = a * b;
        assert_eq!(result.as_slice()[0], FE::from(1u64));
    }

    #[test]
    fn test_packed_distributivity() {
        let a = random_packed();
        let b = PackedGoldilocksNeon::from_fn(|i| FE::from((i as u64 + 5) * 0xFEDCBA987654321u64));
        let c = PackedGoldilocksNeon::from_fn(|i| FE::from((i as u64 + 9).wrapping_mul(0xABCDEF0123456789u64)));
        let lhs = a * (b + c);
        let rhs = a * b + a * c;
        assert_eq!(lhs.as_slice(), rhs.as_slice());
    }

    #[test]
    fn test_pack_slice_roundtrip() {
        let values: Vec<FE> = (0..6).map(|i| FE::from(i as u64 + 100)).collect();
        let (packed, suffix) = PackedGoldilocksNeon::pack_slice_with_suffix(&values);
        assert_eq!(packed.len(), 3);
        assert_eq!(suffix.len(), 0);
        assert_eq!(packed[0].as_slice(), &values[0..2]);
        assert_eq!(packed[1].as_slice(), &values[2..4]);
    }

    #[test]
    fn test_interleave_block1() {
        let a = PackedGoldilocksNeon::from_fn(|i| FE::from(i as u64));
        let b = PackedGoldilocksNeon::from_fn(|i| FE::from(10 + i as u64));
        let (lo, hi) = a.interleave(b, 1);
        // [0,1] x [10,11] -> [0,10], [1,11]
        assert_eq!(lo.as_slice()[0], FE::from(0u64));
        assert_eq!(lo.as_slice()[1], FE::from(10u64));
        assert_eq!(hi.as_slice()[0], FE::from(1u64));
        assert_eq!(hi.as_slice()[1], FE::from(11u64));
    }
}
