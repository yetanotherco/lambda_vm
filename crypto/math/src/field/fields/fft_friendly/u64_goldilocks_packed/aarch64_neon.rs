//! NEON packed Goldilocks field arithmetic (WIDTH=2).
//!
//! Uses 128-bit uint64x2_t registers holding 2 x 64-bit field elements.
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::aarch64::*;
use core::mem::transmute;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks::{GOLDILOCKS_PRIME, GoldilocksField};
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
    let mask: uint64x2_t = vcgtq_s64(
        transmute::<uint64x2_t, int64x2_t>(SHIFTED_FIELD_ORDER),
        transmute::<uint64x2_t, int64x2_t>(x_s),
    );
    // wrapback = EPSILON if x >= P, else 0
    let wrapback = vbicq_u64(EPSILON_VEC, mask);
    vaddq_u64(x_s, wrapback)
}

#[inline(always)]
unsafe fn add_no_double_overflow_s(x: uint64x2_t, y_s: uint64x2_t) -> uint64x2_t {
    let res_s = vaddq_u64(x, y_s);
    let mask: uint64x2_t = vcgtq_s64(
        transmute::<uint64x2_t, int64x2_t>(y_s),
        transmute::<uint64x2_t, int64x2_t>(res_s),
    );
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
    let mask: uint64x2_t = vcgtq_s64(
        transmute::<uint64x2_t, int64x2_t>(b_s),
        transmute::<uint64x2_t, int64x2_t>(a_s),
    );
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

/// Interleaved dual-lane multiply + reduce in a single asm block.
/// Both lanes' mul and reduce are interleaved for maximum ILP on AArch64.
/// Uses native carry flags (subs/csetm/adds/csetm) instead of NEON compare+mask.
/// Adapted from Plonky3's mul_reduce_dual_asm.
#[inline(always)]
unsafe fn mul_neon(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    let a0 = vgetq_lane_u64::<0>(a);
    let a1 = vgetq_lane_u64::<1>(a);
    let b0 = vgetq_lane_u64::<0>(b);
    let b1 = vgetq_lane_u64::<1>(b);

    let result0: u64;
    let result1: u64;

    core::arch::asm!(
        // Compute both 128-bit products (interleaved for ILP)
        "mul   {lo0}, {a0}, {b0}",
        "mul   {lo1}, {a1}, {b1}",
        "umulh {hi0}, {a0}, {b0}",
        "umulh {hi1}, {a1}, {b1}",

        // hi_hi = hi >> 32
        "lsr   {hi_hi0}, {hi0}, #32",
        "lsr   {hi_hi1}, {hi1}, #32",

        // tmp = lo - hi_hi (with borrow via carry flag)
        "subs  {tmp0}, {lo0}, {hi_hi0}",
        "csetm {adj0:w}, cc",
        "subs  {tmp1}, {lo1}, {hi_hi1}",
        "csetm {adj1:w}, cc",
        "sub   {tmp0}, {tmp0}, {adj0}",
        "sub   {tmp1}, {tmp1}, {adj1}",

        // hi_lo = hi & EPSILON
        "and   {hi_lo0}, {hi0}, {epsilon}",
        "and   {hi_lo1}, {hi1}, {epsilon}",

        // hi_lo_eps = (hi_lo << 32) - hi_lo  (avoids multiply by EPSILON)
        "lsl   {t0}, {hi_lo0}, #32",
        "lsl   {t1}, {hi_lo1}, #32",
        "sub   {hle0}, {t0}, {hi_lo0}",
        "sub   {hle1}, {t1}, {hi_lo1}",

        // result = tmp + hi_lo_eps (with overflow via carry flag)
        "adds  {result0}, {tmp0}, {hle0}",
        "csetm {adj0:w}, cs",
        "adds  {result1}, {tmp1}, {hle1}",
        "csetm {adj1:w}, cs",
        "add   {result0}, {result0}, {adj0}",
        "add   {result1}, {result1}, {adj1}",

        a0 = in(reg) a0,
        b0 = in(reg) b0,
        a1 = in(reg) a1,
        b1 = in(reg) b1,
        epsilon = in(reg) EPSILON,
        lo0 = out(reg) _,
        lo1 = out(reg) _,
        hi0 = out(reg) _,
        hi1 = out(reg) _,
        hi_hi0 = out(reg) _,
        hi_hi1 = out(reg) _,
        tmp0 = out(reg) _,
        tmp1 = out(reg) _,
        hi_lo0 = out(reg) _,
        hi_lo1 = out(reg) _,
        t0 = out(reg) _,
        t1 = out(reg) _,
        hle0 = out(reg) _,
        hle1 = out(reg) _,
        adj0 = out(reg) _,
        adj1 = out(reg) _,
        result0 = out(reg) result0,
        result1 = out(reg) result1,
        options(pure, nomem, nostack),
    );

    vcombine_u64(vcreate_u64(result0), vcreate_u64(result1))
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
        let c = PackedGoldilocksNeon::from_fn(|i| {
            FE::from((i as u64 + 9).wrapping_mul(0xABCDEF0123456789u64))
        });
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

    // ---- Proptest randomized tests ----
    use proptest::prelude::*;

    fn arb_packed() -> impl Strategy<Value = PackedGoldilocksNeon> {
        prop::array::uniform2(0u64..GOLDILOCKS_PRIME)
            .prop_map(|arr| PackedGoldilocksNeon::from_fn(|i| FE::from(arr[i])))
    }

    proptest! {
        #[test]
        fn prop_add_commutative(a in arb_packed(), b in arb_packed()) {
            let ab = a + b;
            let ba = b + a;
            prop_assert_eq!(ab.as_slice(), ba.as_slice());
        }

        #[test]
        fn prop_mul_commutative(a in arb_packed(), b in arb_packed()) {
            let ab = a * b;
            let ba = b * a;
            prop_assert_eq!(ab.as_slice(), ba.as_slice());
        }

        #[test]
        fn prop_add_matches_scalar(
            a_vals in prop::array::uniform2(0u64..GOLDILOCKS_PRIME),
            b_vals in prop::array::uniform2(0u64..GOLDILOCKS_PRIME),
        ) {
            let a = PackedGoldilocksNeon::from_fn(|i| FE::from(a_vals[i]));
            let b = PackedGoldilocksNeon::from_fn(|i| FE::from(b_vals[i]));
            let packed = a + b;
            for i in 0..2 {
                let scalar = FE::from(a_vals[i]) + FE::from(b_vals[i]);
                prop_assert_eq!(packed.as_slice()[i], scalar);
            }
        }

        #[test]
        fn prop_sub_matches_scalar(
            a_vals in prop::array::uniform2(0u64..GOLDILOCKS_PRIME),
            b_vals in prop::array::uniform2(0u64..GOLDILOCKS_PRIME),
        ) {
            let a = PackedGoldilocksNeon::from_fn(|i| FE::from(a_vals[i]));
            let b = PackedGoldilocksNeon::from_fn(|i| FE::from(b_vals[i]));
            let packed = a - b;
            for i in 0..2 {
                let scalar = FE::from(a_vals[i]) - FE::from(b_vals[i]);
                prop_assert_eq!(packed.as_slice()[i], scalar);
            }
        }

        #[test]
        fn prop_mul_matches_scalar(
            a_vals in prop::array::uniform2(0u64..GOLDILOCKS_PRIME),
            b_vals in prop::array::uniform2(0u64..GOLDILOCKS_PRIME),
        ) {
            let a = PackedGoldilocksNeon::from_fn(|i| FE::from(a_vals[i]));
            let b = PackedGoldilocksNeon::from_fn(|i| FE::from(b_vals[i]));
            let packed = a * b;
            for i in 0..2 {
                let scalar = FE::from(a_vals[i]) * FE::from(b_vals[i]);
                prop_assert_eq!(packed.as_slice()[i], scalar);
            }
        }

        #[test]
        fn prop_sub_is_add_neg(a in arb_packed(), b in arb_packed()) {
            let sub_result = a - b;
            let add_neg_result = a + (-b);
            prop_assert_eq!(sub_result.as_slice(), add_neg_result.as_slice());
        }

        #[test]
        fn prop_square_matches_mul(a in arb_packed()) {
            let sq = a.square();
            let mul = a * a;
            prop_assert_eq!(sq.as_slice(), mul.as_slice());
        }

        #[test]
        fn prop_distributivity(a in arb_packed(), b in arb_packed(), c in arb_packed()) {
            let lhs = a * (b + c);
            let rhs = a * b + a * c;
            prop_assert_eq!(lhs.as_slice(), rhs.as_slice());
        }
    }
}
