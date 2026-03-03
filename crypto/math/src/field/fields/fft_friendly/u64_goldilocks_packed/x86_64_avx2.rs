//! AVX2 packed Goldilocks field arithmetic (WIDTH=4).
//!
//! Uses 256-bit __m256i registers holding 4 x 64-bit field elements.
//! Modular arithmetic uses the shifted-representation trick for unsigned
//! comparison emulation (XOR with 2^63 converts to signed domain).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::mem::transmute;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks::{GoldilocksField, GOLDILOCKS_PRIME};
use crate::field::packed::PackedField;

const WIDTH: usize = 4;
const EPSILON: u64 = 0xFFFF_FFFF; // 2^32 - 1 = 2^64 mod P

/// Packed Goldilocks field element holding 4 elements in an AVX2 register.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct PackedGoldilocksAVX2(pub [FieldElement<GoldilocksField>; WIDTH]);

impl Default for PackedGoldilocksAVX2 {
    fn default() -> Self {
        Self([FieldElement::zero(); WIDTH])
    }
}

impl PartialEq for PackedGoldilocksAVX2 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for PackedGoldilocksAVX2 {}

impl PackedGoldilocksAVX2 {
    #[inline(always)]
    fn to_vector(self) -> __m256i {
        unsafe { transmute(self) }
    }

    #[inline(always)]
    fn from_vector(v: __m256i) -> Self {
        unsafe { transmute(v) }
    }
}

// ---- Constants ----

const SIGN_BIT: __m256i = unsafe { transmute([i64::MIN; WIDTH]) };
const FIELD_ORDER: __m256i = unsafe { transmute([GOLDILOCKS_PRIME; WIDTH]) };
const SHIFTED_FIELD_ORDER: __m256i =
    unsafe { transmute([GOLDILOCKS_PRIME ^ (i64::MIN as u64); WIDTH]) };
const EPSILON_VEC: __m256i = unsafe { transmute([EPSILON; WIDTH]) };

// ---- Shifted-representation helpers ----

/// XOR with 2^63 to convert unsigned -> signed comparison domain.
#[inline(always)]
unsafe fn shift(x: __m256i) -> __m256i {
    _mm256_xor_si256(x, SIGN_BIT)
}

/// Canonicalize a shifted value to [0, P) in the shifted domain.
#[inline(always)]
unsafe fn canonicalize_s(x_s: __m256i) -> __m256i {
    let mask = _mm256_cmpgt_epi64(SHIFTED_FIELD_ORDER, x_s);
    let wrapback = _mm256_andnot_si256(mask, EPSILON_VEC);
    _mm256_add_epi64(x_s, wrapback)
}

/// Add x (non-shifted) + y_s (shifted, canonical) -> result in shifted domain.
#[inline(always)]
unsafe fn add_no_double_overflow_s(x: __m256i, y_s: __m256i) -> __m256i {
    let res_s = _mm256_add_epi64(x, y_s);
    let mask = _mm256_cmpgt_epi64(y_s, res_s);
    let correction = _mm256_srli_epi64::<32>(mask); // 0xFFFFFFFF = EPSILON
    _mm256_add_epi64(res_s, correction)
}

/// Packed modular addition: (a + b) mod P
#[inline(always)]
unsafe fn add_avx2(a: __m256i, b: __m256i) -> __m256i {
    let b_s = canonicalize_s(shift(b));
    let res_s = add_no_double_overflow_s(a, b_s);
    shift(res_s)
}

/// Packed modular subtraction: (a - b) mod P
#[inline(always)]
unsafe fn sub_avx2(a: __m256i, b: __m256i) -> __m256i {
    let a_s = shift(a);
    let b_s = canonicalize_s(shift(b));
    let mask = _mm256_cmpgt_epi64(b_s, a_s);
    let correction = _mm256_srli_epi64::<32>(mask); // EPSILON
    let res = _mm256_sub_epi64(a_s, b_s);
    // No shift here: (a XOR S) - (b XOR S) = a - b (shifts cancel in subtraction)
    _mm256_sub_epi64(res, correction)
}

/// Packed modular negation: P - a (or 0 if a == 0)
#[inline(always)]
unsafe fn neg_avx2(a: __m256i) -> __m256i {
    let a_canon = shift(canonicalize_s(shift(a)));
    sub_avx2(FIELD_ORDER, a_canon)
}

// ---- Multiply helpers ----

/// Subtract a "small" value (< 2^32) from a shifted value.
#[inline(always)]
unsafe fn sub_small_s(x_s: __m256i, y_small: __m256i) -> __m256i {
    let res_s = _mm256_sub_epi64(x_s, y_small);
    let mask = _mm256_cmpgt_epi64(res_s, x_s);
    let correction = _mm256_srli_epi64::<32>(mask);
    _mm256_sub_epi64(res_s, correction)
}

/// Add a "small" value (< 2^64) to a shifted value.
#[inline(always)]
unsafe fn add_small_s(x_s: __m256i, y: __m256i) -> __m256i {
    let res_s = _mm256_add_epi64(x_s, y);
    let mask = _mm256_cmpgt_epi64(x_s, res_s);
    let correction = _mm256_srli_epi64::<32>(mask);
    _mm256_add_epi64(res_s, correction)
}

/// Multiply two packed 64-bit values, producing (hi, lo) 128-bit results.
/// Uses four 32x32->64 sub-multiplications via _mm256_mul_epu32.
#[inline(always)]
unsafe fn mul64_64(x: __m256i, y: __m256i) -> (__m256i, __m256i) {
    // Extract high 32-bit halves. movehdup_ps runs on port 5,
    // avoiding contention with mul_epu32 on ports 0/1.
    let x_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x)));
    let y_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(y)));

    let mul_ll = _mm256_mul_epu32(x, y);
    let mul_lh = _mm256_mul_epu32(x, y_hi);
    let mul_hl = _mm256_mul_epu32(x_hi, y);
    let mul_hh = _mm256_mul_epu32(x_hi, y_hi);

    let mul_ll_hi = _mm256_srli_epi64::<32>(mul_ll);
    let t0 = _mm256_add_epi64(mul_hl, mul_ll_hi);
    let t0_lo = _mm256_and_si256(t0, EPSILON_VEC);
    let t0_hi = _mm256_srli_epi64::<32>(t0);
    let t1 = _mm256_add_epi64(mul_lh, t0_lo);
    let t1_hi = _mm256_srli_epi64::<32>(t1);
    let res_hi = _mm256_add_epi64(_mm256_add_epi64(mul_hh, t0_hi), t1_hi);

    let t1_lo_shifted = _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(t1)));
    let res_lo = _mm256_blend_epi32::<0b10101010>(mul_ll, t1_lo_shifted);

    (res_hi, res_lo)
}

/// Reduce a 128-bit packed value (hi, lo) to 64-bit Goldilocks elements.
///
/// For x = hi * 2^64 + lo:
///   x mod P = lo - hi_hi + hi_lo * EPSILON
///
/// Uses: 2^96 = -1 (mod P) and 2^64 = EPSILON (mod P).
#[inline(always)]
unsafe fn reduce128_avx2(hi: __m256i, lo: __m256i) -> __m256i {
    let lo_s = shift(lo);
    let hi_hi = _mm256_srli_epi64::<32>(hi);
    let lo1_s = sub_small_s(lo_s, hi_hi);
    let t1 = _mm256_mul_epu32(hi, EPSILON_VEC);
    let lo2_s = add_small_s(lo1_s, t1);
    shift(lo2_s)
}

/// Packed modular multiplication: (a * b) mod P
#[inline(always)]
unsafe fn mul_avx2(a: __m256i, b: __m256i) -> __m256i {
    let (hi, lo) = mul64_64(a, b);
    reduce128_avx2(hi, lo)
}

/// Packed modular squaring: a^2 mod P (3 sub-products instead of 4)
#[inline(always)]
unsafe fn square_avx2(a: __m256i) -> __m256i {
    let a_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(a)));

    let mul_ll = _mm256_mul_epu32(a, a);
    let mul_lh = _mm256_mul_epu32(a, a_hi);
    let mul_hh = _mm256_mul_epu32(a_hi, a_hi);

    // Double the cross term (shift left by 33 instead of 32 to account for 2x)
    let mul_ll_hi = _mm256_srli_epi64::<33>(mul_ll);
    let t0 = _mm256_add_epi64(mul_lh, mul_ll_hi);
    let t0_hi = _mm256_srli_epi64::<31>(t0);
    let res_hi = _mm256_add_epi64(mul_hh, t0_hi);

    let t0_lo_shifted = _mm256_slli_epi64::<33>(t0);
    let mul_ll_lo = _mm256_and_si256(mul_ll, _mm256_set1_epi64x(1));
    let res_lo = _mm256_or_si256(t0_lo_shifted, mul_ll_lo);

    reduce128_avx2(res_hi, res_lo)
}

// ---- Operator impls ----

impl Add for PackedGoldilocksAVX2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_vector(unsafe { add_avx2(self.to_vector(), rhs.to_vector()) })
    }
}

impl Sub for PackedGoldilocksAVX2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_vector(unsafe { sub_avx2(self.to_vector(), rhs.to_vector()) })
    }
}

impl Neg for PackedGoldilocksAVX2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_vector(unsafe { neg_avx2(self.to_vector()) })
    }
}

impl Mul for PackedGoldilocksAVX2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::from_vector(unsafe { mul_avx2(self.to_vector(), rhs.to_vector()) })
    }
}

impl AddAssign for PackedGoldilocksAVX2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for PackedGoldilocksAVX2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for PackedGoldilocksAVX2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// ---- PackedField impl ----

unsafe impl PackedField for PackedGoldilocksAVX2 {
    type Scalar = GoldilocksField;
    const WIDTH: usize = WIDTH;

    #[inline(always)]
    fn from_fn(mut f: impl FnMut(usize) -> FieldElement<GoldilocksField>) -> Self {
        Self([f(0), f(1), f(2), f(3)])
    }

    #[inline(always)]
    fn from_slice(slice: &[FieldElement<GoldilocksField>]) -> Self {
        Self([slice[0], slice[1], slice[2], slice[3]])
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
        Self::from_vector(unsafe { square_avx2(self.to_vector()) })
    }

    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        unsafe {
            let a = self.to_vector();
            let b = other.to_vector();
            match block_len {
                1 => {
                    let lo = _mm256_unpacklo_epi64(a, b);
                    let hi = _mm256_unpackhi_epi64(a, b);
                    (Self::from_vector(lo), Self::from_vector(hi))
                }
                2 => {
                    let t = _mm256_permute2x128_si256::<0x21>(a, b);
                    let lo = _mm256_blend_epi32::<0b11110000>(a, t);
                    let hi = _mm256_blend_epi32::<0b11110000>(t, b);
                    (Self::from_vector(lo), Self::from_vector(hi))
                }
                4 => (*self, other),
                _ => panic!("block_len must be 1, 2, or 4 for WIDTH=4"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type FE = FieldElement<GoldilocksField>;

    fn random_packed() -> PackedGoldilocksAVX2 {
        PackedGoldilocksAVX2::from_fn(|i| FE::from((i as u64 + 1) * 0x123456789ABCDEFu64))
    }

    #[test]
    fn test_packed_add_matches_scalar() {
        let a = random_packed();
        let b = PackedGoldilocksAVX2::from_fn(|i| FE::from((i as u64 + 5) * 0xFEDCBA987654321u64));
        let packed_sum = a + b;
        let scalar_sum =
            PackedGoldilocksAVX2::from_fn(|i| a.as_slice()[i] + b.as_slice()[i]);
        assert_eq!(packed_sum.as_slice(), scalar_sum.as_slice());
    }

    #[test]
    fn test_packed_sub_matches_scalar() {
        let a = random_packed();
        let b = PackedGoldilocksAVX2::from_fn(|i| FE::from((i as u64 + 5) * 0xFEDCBA987654321u64));
        let packed_diff = a - b;
        let scalar_diff =
            PackedGoldilocksAVX2::from_fn(|i| a.as_slice()[i] - b.as_slice()[i]);
        assert_eq!(packed_diff.as_slice(), scalar_diff.as_slice());
    }

    #[test]
    fn test_packed_neg_matches_scalar() {
        let a = random_packed();
        let packed_neg = -a;
        let scalar_neg = PackedGoldilocksAVX2::from_fn(|i| -a.as_slice()[i]);
        assert_eq!(packed_neg.as_slice(), scalar_neg.as_slice());
    }

    #[test]
    fn test_packed_add_overflow() {
        let a = PackedGoldilocksAVX2::broadcast(FE::from(GOLDILOCKS_PRIME - 1));
        let b = PackedGoldilocksAVX2::broadcast(FE::from(2u64));
        let sum = a + b;
        assert_eq!(sum.as_slice()[0], FE::from(1u64));
    }

    #[test]
    fn test_packed_sub_underflow() {
        let a = PackedGoldilocksAVX2::broadcast(FE::from(1u64));
        let b = PackedGoldilocksAVX2::broadcast(FE::from(3u64));
        let diff = a - b;
        assert_eq!(diff.as_slice()[0], FE::from(GOLDILOCKS_PRIME - 2));
    }

    #[test]
    fn test_pack_slice_roundtrip() {
        let values: Vec<FE> = (0..8).map(|i| FE::from(i as u64 + 100)).collect();
        let (packed, suffix) = PackedGoldilocksAVX2::pack_slice_with_suffix(&values);
        assert_eq!(packed.len(), 2);
        assert_eq!(suffix.len(), 0);
        assert_eq!(packed[0].as_slice(), &values[0..4]);
        assert_eq!(packed[1].as_slice(), &values[4..8]);
    }

    #[test]
    fn test_packed_mul_matches_scalar() {
        let a = random_packed();
        let b = PackedGoldilocksAVX2::from_fn(|i| FE::from((i as u64 + 5) * 0xFEDCBA987654321u64));
        let packed_prod = a * b;
        let scalar_prod =
            PackedGoldilocksAVX2::from_fn(|i| a.as_slice()[i] * b.as_slice()[i]);
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
        let one = PackedGoldilocksAVX2::ones();
        assert_eq!((a * one).as_slice(), a.as_slice());
    }

    #[test]
    fn test_packed_mul_zero() {
        let a = random_packed();
        let zero = PackedGoldilocksAVX2::zero();
        let result = a * zero;
        for lane in result.as_slice() {
            assert_eq!(*lane, FE::zero());
        }
    }

    #[test]
    fn test_packed_mul_large_values() {
        let a = PackedGoldilocksAVX2::broadcast(FE::from(GOLDILOCKS_PRIME - 1));
        let b = PackedGoldilocksAVX2::broadcast(FE::from(GOLDILOCKS_PRIME - 1));
        let result = a * b;
        // (p-1)^2 mod p = 1
        assert_eq!(result.as_slice()[0], FE::from(1u64));
    }

    #[test]
    fn test_packed_distributivity() {
        let a = random_packed();
        let b = PackedGoldilocksAVX2::from_fn(|i| FE::from((i as u64 + 5) * 0xFEDCBA987654321u64));
        let c = PackedGoldilocksAVX2::from_fn(|i| FE::from((i as u64 + 9).wrapping_mul(0xABCDEF0123456789u64)));
        let lhs = a * (b + c);
        let rhs = a * b + a * c;
        assert_eq!(lhs.as_slice(), rhs.as_slice());
    }

    #[test]
    fn test_interleave_block1() {
        let a = PackedGoldilocksAVX2::from_fn(|i| FE::from(i as u64));
        let b = PackedGoldilocksAVX2::from_fn(|i| FE::from(10 + i as u64));
        let (lo, hi) = a.interleave(b, 1);
        // [0,1,2,3] x [10,11,12,13] -> [0,10,2,12], [1,11,3,13]
        assert_eq!(lo.as_slice()[0], FE::from(0u64));
        assert_eq!(lo.as_slice()[1], FE::from(10u64));
        assert_eq!(lo.as_slice()[2], FE::from(2u64));
        assert_eq!(lo.as_slice()[3], FE::from(12u64));
        assert_eq!(hi.as_slice()[0], FE::from(1u64));
        assert_eq!(hi.as_slice()[1], FE::from(11u64));
        assert_eq!(hi.as_slice()[2], FE::from(3u64));
        assert_eq!(hi.as_slice()[3], FE::from(13u64));
    }

    // ---- Proptests ----

    use proptest::prelude::*;

    fn arb_packed() -> impl Strategy<Value = PackedGoldilocksAVX2> {
        prop::array::uniform4(0u64..GOLDILOCKS_PRIME)
            .prop_map(|arr| PackedGoldilocksAVX2::from_fn(|i| FE::from(arr[i])))
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
            a_vals in prop::array::uniform4(0u64..GOLDILOCKS_PRIME),
            b_vals in prop::array::uniform4(0u64..GOLDILOCKS_PRIME),
        ) {
            let a = PackedGoldilocksAVX2::from_fn(|i| FE::from(a_vals[i]));
            let b = PackedGoldilocksAVX2::from_fn(|i| FE::from(b_vals[i]));
            let packed = a + b;
            for i in 0..4 {
                let scalar = FE::from(a_vals[i]) + FE::from(b_vals[i]);
                prop_assert_eq!(packed.as_slice()[i], scalar);
            }
        }

        #[test]
        fn prop_sub_matches_scalar(
            a_vals in prop::array::uniform4(0u64..GOLDILOCKS_PRIME),
            b_vals in prop::array::uniform4(0u64..GOLDILOCKS_PRIME),
        ) {
            let a = PackedGoldilocksAVX2::from_fn(|i| FE::from(a_vals[i]));
            let b = PackedGoldilocksAVX2::from_fn(|i| FE::from(b_vals[i]));
            let packed = a - b;
            for i in 0..4 {
                let scalar = FE::from(a_vals[i]) - FE::from(b_vals[i]);
                prop_assert_eq!(packed.as_slice()[i], scalar);
            }
        }

        #[test]
        fn prop_mul_matches_scalar(
            a_vals in prop::array::uniform4(0u64..GOLDILOCKS_PRIME),
            b_vals in prop::array::uniform4(0u64..GOLDILOCKS_PRIME),
        ) {
            let a = PackedGoldilocksAVX2::from_fn(|i| FE::from(a_vals[i]));
            let b = PackedGoldilocksAVX2::from_fn(|i| FE::from(b_vals[i]));
            let packed = a * b;
            for i in 0..4 {
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
