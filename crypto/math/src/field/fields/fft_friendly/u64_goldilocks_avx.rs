//! AVX2-optimized Goldilocks field multiplication.
//!
//! This module provides SIMD-accelerated multiplication for the Goldilocks prime field
//! p = 2^64 - 2^32 + 1. It processes 4 field elements in parallel using AVX2 instructions.
//!
//! # Architecture Support
//!
//! This module is only compiled on x86/x86_64 architectures with AVX2 support.
//! Runtime feature detection is used to ensure AVX2 is available.
//!
//! # Performance
//!
//! The AVX2 implementation achieves approximately 4x throughput compared to scalar
//! operations when processing batches of field elements.
//!
//! # References
//!
//! - Goldilocks prime structure: p = 2^64 - 2^32 + 1
//! - EPSILON = 2^32 - 1 = -2^64 mod p

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::u64_goldilocks::{GoldilocksElement, GoldilocksField};
use crate::field::element::FieldElement;
use crate::field::traits::IsField;

/// EPSILON = 2^32 - 1 = p - 2^64 (used for fast reduction)
const EPSILON: u64 = 0xFFFF_FFFF;

/// Check if AVX2 is available at runtime.
#[inline]
pub fn is_avx2_available() -> bool {
    is_x86_feature_detected!("avx2")
}

/// A packed representation of 4 Goldilocks field elements for SIMD operations.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct PackedGoldilocks {
    /// The 4 field elements stored as u64 values
    pub values: [u64; 4],
}

impl PackedGoldilocks {
    /// Create a new PackedGoldilocks from an array of 4 field elements.
    #[inline]
    pub fn new(values: [u64; 4]) -> Self {
        Self { values }
    }

    /// Create a PackedGoldilocks with all elements set to zero.
    #[inline]
    pub fn zero() -> Self {
        Self { values: [0; 4] }
    }

    /// Create a PackedGoldilocks with all elements set to one.
    #[inline]
    pub fn one() -> Self {
        Self { values: [1; 4] }
    }

    /// Create a PackedGoldilocks from field elements.
    #[inline]
    pub fn from_field_elements(elements: [GoldilocksElement; 4]) -> Self {
        Self {
            values: [
                *elements[0].value(),
                *elements[1].value(),
                *elements[2].value(),
                *elements[3].value(),
            ],
        }
    }

    /// Convert to an array of field elements.
    #[inline]
    pub fn to_field_elements(self) -> [GoldilocksElement; 4] {
        [
            FieldElement::from(self.values[0]),
            FieldElement::from(self.values[1]),
            FieldElement::from(self.values[2]),
            FieldElement::from(self.values[3]),
        ]
    }

    /// Multiply two packed Goldilocks values using AVX2.
    ///
    /// # Safety
    ///
    /// This function requires AVX2 support. Use `is_avx2_available()` to check
    /// before calling, or use `mul_packed` which falls back to scalar operations.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn mul_avx2(self, other: Self) -> Self {
        // SAFETY: Caller guarantees AVX2 is available
        unsafe { mul_packed_avx2(self, other) }
    }

    /// Multiply two packed Goldilocks values, with automatic fallback to scalar.
    #[inline]
    pub fn mul_packed(self, other: Self) -> Self {
        if is_avx2_available() {
            // SAFETY: We just checked that AVX2 is available
            unsafe { self.mul_avx2(other) }
        } else {
            self.mul_scalar(other)
        }
    }

    /// Scalar fallback for multiplication.
    #[inline]
    fn mul_scalar(self, other: Self) -> Self {
        Self {
            values: [
                GoldilocksField::mul(&self.values[0], &other.values[0]),
                GoldilocksField::mul(&self.values[1], &other.values[1]),
                GoldilocksField::mul(&self.values[2], &other.values[2]),
                GoldilocksField::mul(&self.values[3], &other.values[3]),
            ],
        }
    }

    /// Add two packed Goldilocks values using AVX2.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn add_avx2(self, other: Self) -> Self {
        // SAFETY: Caller guarantees AVX2 is available
        unsafe { add_packed_avx2(self, other) }
    }

    /// Add two packed Goldilocks values with automatic fallback.
    #[inline]
    pub fn add_packed(self, other: Self) -> Self {
        if is_avx2_available() {
            // SAFETY: We just checked that AVX2 is available
            unsafe { self.add_avx2(other) }
        } else {
            self.add_scalar(other)
        }
    }

    /// Scalar fallback for addition.
    #[inline]
    fn add_scalar(self, other: Self) -> Self {
        Self {
            values: [
                GoldilocksField::add(&self.values[0], &other.values[0]),
                GoldilocksField::add(&self.values[1], &other.values[1]),
                GoldilocksField::add(&self.values[2], &other.values[2]),
                GoldilocksField::add(&self.values[3], &other.values[3]),
            ],
        }
    }

    /// Subtract two packed Goldilocks values using AVX2.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn sub_avx2(self, other: Self) -> Self {
        // SAFETY: Caller guarantees AVX2 is available
        unsafe { sub_packed_avx2(self, other) }
    }

    /// Subtract two packed Goldilocks values with automatic fallback.
    #[inline]
    pub fn sub_packed(self, other: Self) -> Self {
        if is_avx2_available() {
            // SAFETY: We just checked that AVX2 is available
            unsafe { self.sub_avx2(other) }
        } else {
            self.sub_scalar(other)
        }
    }

    /// Scalar fallback for subtraction.
    #[inline]
    fn sub_scalar(self, other: Self) -> Self {
        Self {
            values: [
                GoldilocksField::sub(&self.values[0], &other.values[0]),
                GoldilocksField::sub(&self.values[1], &other.values[1]),
                GoldilocksField::sub(&self.values[2], &other.values[2]),
                GoldilocksField::sub(&self.values[3], &other.values[3]),
            ],
        }
    }

    /// Square all elements in the packed value.
    #[inline]
    pub fn square_packed(self) -> Self {
        self.mul_packed(self)
    }
}

/// AVX2 implementation of packed Goldilocks multiplication.
///
/// This implements schoolbook multiplication with Goldilocks reduction.
/// For a * b where both are 64-bit:
///   - Split each into high and low 32-bit parts: a = a_hi * 2^32 + a_lo
///   - Compute partial products and reduce using the Goldilocks identity
///
/// The Goldilocks reduction uses: 2^64 ≡ 2^32 - 1 (mod p) and 2^96 ≡ -1 (mod p)
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mul_packed_avx2(a: PackedGoldilocks, b: PackedGoldilocks) -> PackedGoldilocks {
    // SAFETY: All operations require AVX2, which is guaranteed by the target_feature attribute
    unsafe {
        // Load the packed values
        let a_vec = _mm256_loadu_si256(a.values.as_ptr() as *const __m256i);
        let b_vec = _mm256_loadu_si256(b.values.as_ptr() as *const __m256i);

        // Constants
        let epsilon = _mm256_set1_epi64x(EPSILON as i64);
        let low_mask = _mm256_set1_epi64x(0xFFFFFFFF_i64);

        // Split a and b into high and low 32-bit parts
        let a_lo = _mm256_and_si256(a_vec, low_mask);
        let a_hi = _mm256_srli_epi64(a_vec, 32);
        let b_lo = _mm256_and_si256(b_vec, low_mask);
        let b_hi = _mm256_srli_epi64(b_vec, 32);

        // Compute the four 32x32 -> 64 partial products
        // a * b = (a_hi * 2^32 + a_lo) * (b_hi * 2^32 + b_lo)
        //       = a_hi * b_hi * 2^64 + (a_hi * b_lo + a_lo * b_hi) * 2^32 + a_lo * b_lo

        // p0 = a_lo * b_lo (contributes to low 64 bits)
        let p0 = _mm256_mul_epu32(a_lo, b_lo);

        // p1 = a_lo * b_hi (contributes at bit position 32)
        let p1 = _mm256_mul_epu32(a_lo, b_hi);

        // p2 = a_hi * b_lo (contributes at bit position 32)
        let p2 = _mm256_mul_epu32(a_hi, b_lo);

        // p3 = a_hi * b_hi (contributes at bit position 64)
        let p3 = _mm256_mul_epu32(a_hi, b_hi);

        // Now we need to combine these into a 128-bit result and reduce.
        // The 128-bit result is: result = p0 + (p1 + p2) << 32 + p3 << 64
        //
        // For Goldilocks reduction: x = x_lo + x_hi * 2^64
        // where x_hi = x_hi_hi * 2^32 + x_hi_lo
        // result ≡ x_lo + x_hi_lo * EPSILON - x_hi_hi (mod p)

        // Step 1: Compute the middle term (p1 + p2)
        let p_mid = _mm256_add_epi64(p1, p2);
        let p_mid_lo = _mm256_and_si256(p_mid, low_mask);
        let p_mid_hi = _mm256_srli_epi64(p_mid, 32);

        // Step 2: Add p_mid_lo << 32 to p0 (this forms the lower 64 bits plus carry)
        let p_mid_shifted = _mm256_slli_epi64(p_mid_lo, 32);
        let lo_sum = _mm256_add_epi64(p0, p_mid_shifted);

        // Check for carry: if lo_sum < p0, there was a carry
        let carry_lo = _mm256_cmpgt_epi64(p0, lo_sum);
        let carry_lo_val = _mm256_and_si256(carry_lo, _mm256_set1_epi64x(1));

        // Step 3: Compute high 64 bits: p3 + p_mid_hi + carry
        let hi = _mm256_add_epi64(p3, p_mid_hi);
        let hi = _mm256_sub_epi64(hi, carry_lo_val); // carry was negative due to cmpgt

        // Step 4: Apply Goldilocks reduction
        // hi = hi_hi * 2^32 + hi_lo
        // result = lo - hi_hi + hi_lo * EPSILON
        let hi_lo = _mm256_and_si256(hi, low_mask);
        let hi_hi = _mm256_srli_epi64(hi, 32);

        // Compute hi_lo * EPSILON = (hi_lo << 32) - hi_lo
        let hi_lo_shifted = _mm256_slli_epi64(hi_lo, 32);
        let epsilon_term = _mm256_sub_epi64(hi_lo_shifted, hi_lo);

        // Step 5: result = lo - hi_hi
        // Handle underflow: if lo < hi_hi, we need to subtract EPSILON
        let underflow = _mm256_cmpgt_epi64(hi_hi, lo_sum);
        let underflow_correction = _mm256_and_si256(underflow, epsilon);
        let result = _mm256_sub_epi64(lo_sum, hi_hi);
        let result = _mm256_sub_epi64(result, underflow_correction);

        // Step 6: result = result + epsilon_term
        // Handle overflow
        let result = _mm256_add_epi64(result, epsilon_term);
        let overflow = _mm256_cmpgt_epi64(
            epsilon_term,
            _mm256_sub_epi64(_mm256_set1_epi64x(-1), result),
        );
        let overflow_correction = _mm256_and_si256(overflow, epsilon);
        let result = _mm256_add_epi64(result, overflow_correction);

        // Store result
        let mut output = PackedGoldilocks::zero();
        _mm256_storeu_si256(output.values.as_mut_ptr() as *mut __m256i, result);
        output
    }
}

/// AVX2 implementation of packed Goldilocks addition.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn add_packed_avx2(a: PackedGoldilocks, b: PackedGoldilocks) -> PackedGoldilocks {
    // SAFETY: All operations require AVX2, which is guaranteed by the target_feature attribute
    unsafe {
        let a_vec = _mm256_loadu_si256(a.values.as_ptr() as *const __m256i);
        let b_vec = _mm256_loadu_si256(b.values.as_ptr() as *const __m256i);
        let epsilon = _mm256_set1_epi64x(EPSILON as i64);

        // Add the values
        let sum = _mm256_add_epi64(a_vec, b_vec);

        // Check for overflow (sum < a indicates overflow in unsigned arithmetic)
        let overflow = _mm256_cmpgt_epi64(a_vec, sum);
        let correction = _mm256_and_si256(overflow, epsilon);
        let result = _mm256_add_epi64(sum, correction);

        // Second overflow check for the correction
        let overflow2 = _mm256_cmpgt_epi64(sum, result);
        let correction2 = _mm256_and_si256(overflow2, epsilon);
        let result = _mm256_add_epi64(result, correction2);

        let mut output = PackedGoldilocks::zero();
        _mm256_storeu_si256(output.values.as_mut_ptr() as *mut __m256i, result);
        output
    }
}

/// AVX2 implementation of packed Goldilocks subtraction.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sub_packed_avx2(a: PackedGoldilocks, b: PackedGoldilocks) -> PackedGoldilocks {
    // SAFETY: All operations require AVX2, which is guaranteed by the target_feature attribute
    unsafe {
        let a_vec = _mm256_loadu_si256(a.values.as_ptr() as *const __m256i);
        let b_vec = _mm256_loadu_si256(b.values.as_ptr() as *const __m256i);
        let epsilon = _mm256_set1_epi64x(EPSILON as i64);

        // Subtract the values
        let diff = _mm256_sub_epi64(a_vec, b_vec);

        // Check for underflow (a < b indicates underflow)
        let underflow = _mm256_cmpgt_epi64(b_vec, a_vec);
        let correction = _mm256_and_si256(underflow, epsilon);
        let result = _mm256_sub_epi64(diff, correction);

        // Second underflow check
        let underflow2 = _mm256_cmpgt_epi64(diff, result);
        let correction2 = _mm256_and_si256(underflow2, epsilon);
        let result = _mm256_sub_epi64(result, correction2);

        let mut output = PackedGoldilocks::zero();
        _mm256_storeu_si256(output.values.as_mut_ptr() as *mut __m256i, result);
        output
    }
}

/// Multiply a slice of Goldilocks field elements by another slice using AVX2.
///
/// Both slices must have the same length. The length must be a multiple of 4.
/// Returns a new vector with the element-wise products.
#[cfg(feature = "alloc")]
pub fn mul_slice_avx2(
    a: &[GoldilocksElement],
    b: &[GoldilocksElement],
) -> alloc::vec::Vec<GoldilocksElement> {
    assert_eq!(a.len(), b.len(), "Slices must have the same length");
    assert!(a.len() % 4 == 0, "Length must be a multiple of 4");

    let mut result = alloc::vec::Vec::with_capacity(a.len());

    for i in (0..a.len()).step_by(4) {
        let packed_a =
            PackedGoldilocks::from_field_elements([a[i], a[i + 1], a[i + 2], a[i + 3]]);
        let packed_b =
            PackedGoldilocks::from_field_elements([b[i], b[i + 1], b[i + 2], b[i + 3]]);

        let packed_result = packed_a.mul_packed(packed_b);
        let elements = packed_result.to_field_elements();
        result.extend_from_slice(&elements);
    }

    result
}

/// Multiply a slice of Goldilocks field elements by a scalar using AVX2.
#[cfg(feature = "alloc")]
pub fn mul_scalar_slice_avx2(
    a: &[GoldilocksElement],
    scalar: GoldilocksElement,
) -> alloc::vec::Vec<GoldilocksElement> {
    assert!(a.len() % 4 == 0, "Length must be a multiple of 4");

    let packed_scalar = PackedGoldilocks::new([*scalar.value(); 4]);
    let mut result = alloc::vec::Vec::with_capacity(a.len());

    for i in (0..a.len()).step_by(4) {
        let packed_a =
            PackedGoldilocks::from_field_elements([a[i], a[i + 1], a[i + 2], a[i + 3]]);
        let packed_result = packed_a.mul_packed(packed_scalar);
        let elements = packed_result.to_field_elements();
        result.extend_from_slice(&elements);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::u64_goldilocks::GOLDILOCKS_PRIME;

    /// Helper function to canonicalize a value
    fn canonicalize(x: u64) -> u64 {
        if x >= GOLDILOCKS_PRIME {
            x - GOLDILOCKS_PRIME
        } else {
            x
        }
    }

    #[test]
    fn test_avx2_available() {
        // Just print whether AVX2 is available for debugging
        println!("AVX2 available: {}", is_avx2_available());
    }

    #[test]
    fn test_packed_mul_basic() {
        let a = PackedGoldilocks::new([5, 7, 11, 13]);
        let b = PackedGoldilocks::new([3, 5, 7, 11]);
        let result = a.mul_packed(b);

        // Verify against scalar multiplication
        for i in 0..4 {
            let expected = GoldilocksField::mul(&a.values[i], &b.values[i]);
            let expected_canonical = canonicalize(expected);
            let result_canonical = canonicalize(result.values[i]);
            assert_eq!(
                result_canonical, expected_canonical,
                "Mismatch at index {}: expected {}, got {}",
                i, expected_canonical, result_canonical
            );
        }
    }

    #[test]
    fn test_packed_mul_large_values() {
        let a = PackedGoldilocks::new([
            0xFFFFFFFF_00000000, // Near prime
            1u64 << 40,
            GOLDILOCKS_PRIME - 1,
            0xDEADBEEF_CAFEBABE,
        ]);
        let b = PackedGoldilocks::new([
            0x12345678_9ABCDEF0,
            1u64 << 40,
            2,
            0xFEEDFACE_DEADBEEF,
        ]);
        let result = a.mul_packed(b);

        // Verify against scalar multiplication
        for i in 0..4 {
            let expected = GoldilocksField::mul(&a.values[i], &b.values[i]);
            let expected_canonical = canonicalize(expected);
            let result_canonical = canonicalize(result.values[i]);
            assert_eq!(
                result_canonical, expected_canonical,
                "Mismatch at index {}: expected {}, got {}",
                i, expected_canonical, result_canonical
            );
        }
    }

    #[test]
    fn test_packed_add() {
        let a = PackedGoldilocks::new([5, 7, 11, GOLDILOCKS_PRIME - 1]);
        let b = PackedGoldilocks::new([3, 5, 7, 2]);
        let result = a.add_packed(b);

        // Verify against scalar addition
        for i in 0..4 {
            let expected = GoldilocksField::add(&a.values[i], &b.values[i]);
            let expected_canonical = canonicalize(expected);
            let result_canonical = canonicalize(result.values[i]);
            assert_eq!(
                result_canonical, expected_canonical,
                "Mismatch at index {}: expected {}, got {}",
                i, expected_canonical, result_canonical
            );
        }
    }

    #[test]
    fn test_packed_sub() {
        let a = PackedGoldilocks::new([5, 7, 11, 2]);
        let b = PackedGoldilocks::new([3, 5, 7, 5]); // Last one causes underflow
        let result = a.sub_packed(b);

        // Verify against scalar subtraction
        for i in 0..4 {
            let expected = GoldilocksField::sub(&a.values[i], &b.values[i]);
            let expected_canonical = canonicalize(expected);
            let result_canonical = canonicalize(result.values[i]);
            assert_eq!(
                result_canonical, expected_canonical,
                "Mismatch at index {}: expected {}, got {}",
                i, expected_canonical, result_canonical
            );
        }
    }

    #[test]
    fn test_packed_square() {
        let a = PackedGoldilocks::new([5, 123456789, GOLDILOCKS_PRIME - 1, 0xDEADBEEF]);
        let result = a.square_packed();

        // Verify against scalar squaring
        for i in 0..4 {
            let expected = GoldilocksField::square(&a.values[i]);
            let expected_canonical = canonicalize(expected);
            let result_canonical = canonicalize(result.values[i]);
            assert_eq!(
                result_canonical, expected_canonical,
                "Mismatch at index {}: expected {}, got {}",
                i, expected_canonical, result_canonical
            );
        }
    }

    #[test]
    fn test_packed_identity() {
        let a = PackedGoldilocks::new([5, 7, 11, 13]);
        let one = PackedGoldilocks::one();

        // a * 1 = a
        let result = a.mul_packed(one);
        for i in 0..4 {
            assert_eq!(
                canonicalize(result.values[i]),
                canonicalize(a.values[i]),
                "Multiplication by one failed at index {}",
                i
            );
        }

        // a + 0 = a
        let zero = PackedGoldilocks::zero();
        let result = a.add_packed(zero);
        for i in 0..4 {
            assert_eq!(
                canonicalize(result.values[i]),
                canonicalize(a.values[i]),
                "Addition by zero failed at index {}",
                i
            );
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_mul_slice() {
        let a: alloc::vec::Vec<GoldilocksElement> =
            (0..8).map(|i| FieldElement::from(i as u64 + 1)).collect();
        let b: alloc::vec::Vec<GoldilocksElement> =
            (0..8).map(|i| FieldElement::from(i as u64 + 5)).collect();

        let result = mul_slice_avx2(&a, &b);

        for i in 0..8 {
            let expected = a[i] * b[i];
            assert_eq!(result[i], expected, "Mismatch at index {}", i);
        }
    }
}
