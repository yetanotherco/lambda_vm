//! Optimized Goldilocks field implementation using native u64 arithmetic.
//!
//! This implementation uses direct u64 representation (no Montgomery form) and exploits
//! the special structure of the Goldilocks prime p = 2^64 - 2^32 + 1 for fast reduction.
//! # Key Properties
//!
//! - EPSILON = 2^32 - 1 = -2^64 mod p
//! - φ = 2^32 is a 6th root of unity (φ^3 = -1, φ^6 = 1)
//! - 2^96 ≡ -1 (mod p)
//!
//! # References
//!
//! - Plonky3: <https://github.com/Plonky3/Plonky3>
//! - Plonky2: <https://github.com/0xPolygonZero/plonky2>
//! - Remco Bloemen: <https://xn--2-umb.com/23/gold-reduce/>

use crate::field::traits::HasDefaultTranscript;
use crate::field::{element::FieldElement, errors::FieldError, traits::IsField};
use crate::traits::{AsBytes, ByteConversion};

/// The Goldilocks prime: p = 2^64 - 2^32 + 1
pub const GOLDILOCKS_PRIME: u64 = 0xFFFF_FFFF_0000_0001;

/// EPSILON = 2^32 - 1 = p - 2^64 (i.e., -2^64 mod p)
/// This is the key constant for fast reduction.
const EPSILON: u64 = 0xFFFF_FFFF;

/// Native Goldilocks field using direct u64 representation.
///
/// Values are stored as u64 in the range [0, 2^64), not necessarily canonical.
/// Canonicalization to [0, p) happens only when needed (comparison, serialization).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GoldilocksField;

impl IsField for GoldilocksField {
    type BaseType = u64;

    /// Addition with overflow handling.
    /// If a + b overflows, we add EPSILON (since 2^64 ≡ EPSILON mod p)
    #[inline(always)]
    fn add(a: &u64, b: &u64) -> u64 {
        let (sum, over) = a.overflowing_add(*b);
        let (sum, over2) = sum.overflowing_add((over as u64) * EPSILON);
        // Second overflow is rare but possible
        if over2 {
            sum.wrapping_add(EPSILON)
        } else {
            sum
        }
    }

    /// Subtraction with underflow handling.
    /// If a - b underflows, we subtract EPSILON (since -2^64 ≡ -EPSILON mod p)
    #[inline(always)]
    fn sub(a: &u64, b: &u64) -> u64 {
        let (diff, under) = a.overflowing_sub(*b);
        let (diff, under2) = diff.overflowing_sub((under as u64) * EPSILON);
        // Second underflow is rare but possible
        if under2 {
            diff.wrapping_sub(EPSILON)
        } else {
            diff
        }
    }

    /// Multiplication using 128-bit intermediate and fast reduction.
    /// LLVM generates optimal MUL+UMULH code on ARM64.
    #[inline(always)]
    fn mul(a: &u64, b: &u64) -> u64 {
        let product = (*a as u128) * (*b as u128);
        reduce128(product)
    }

    /// Squaring using 128-bit intermediate and fast reduction.
    #[inline(always)]
    fn square(a: &u64) -> u64 {
        let a_val = *a;
        let product = (a_val as u128) * (a_val as u128);
        reduce128(product)
    }

    /// Negation: -a = p - a (or 0 if a = 0)
    #[inline(always)]
    fn neg(a: &u64) -> u64 {
        let canonical = Self::canonical(a);
        if canonical == 0 {
            0
        } else {
            GOLDILOCKS_PRIME - canonical
        }
    }

    /// Multiplicative inverse using Fermat's little theorem: a^(-1) = a^(p-2)
    fn inv(a: &u64) -> Result<u64, FieldError> {
        let canonical = Self::canonical(a);
        if canonical == 0 {
            return Err(FieldError::InvZeroError);
        }
        Ok(exp_p_minus_2(canonical))
    }

    fn div(a: &u64, b: &u64) -> Result<u64, FieldError> {
        let b_inv = Self::inv(b)?;
        Ok(Self::mul(a, &b_inv))
    }

    #[inline(always)]
    fn eq(a: &u64, b: &u64) -> bool {
        Self::canonical(a) == Self::canonical(b)
    }

    #[inline(always)]
    fn zero() -> u64 {
        0
    }

    #[inline(always)]
    fn one() -> u64 {
        1
    }

    #[inline(always)]
    fn from_u64(x: u64) -> u64 {
        // For values >= p, we need to reduce
        if x >= GOLDILOCKS_PRIME {
            x - GOLDILOCKS_PRIME
        } else {
            x
        }
    }

    #[inline(always)]
    fn from_base_type(x: u64) -> u64 {
        x
    }

    #[inline(always)]
    fn double(a: &u64) -> u64 {
        Self::add(a, a)
    }
}

/// Reduce a 128-bit value to a 64-bit Goldilocks field element.
///
/// Uses the identity: 2^64 ≡ 2^32 - 1 (mod p)
/// and: 2^96 ≡ -1 (mod p)
///
/// For x = x_lo + x_hi * 2^64, where x_hi = x_hi_hi * 2^32 + x_hi_lo:
/// x ≡ x_lo + x_hi_lo * EPSILON - x_hi_hi (mod p)
///
/// **Optimization**: Uses shift instead of multiply for EPSILON computation:
/// x_hi_lo * EPSILON = x_hi_lo * (2^32 - 1) = (x_hi_lo << 32) - x_hi_lo
/// Benchmarks show this is ~10% faster than using multiply.
#[inline(always)]
fn reduce128(x: u128) -> u64 {
    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & EPSILON;

    // Step 1: t0 = x_lo - x_hi_hi
    let (t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    let t0 = if borrow { t0.wrapping_sub(EPSILON) } else { t0 };

    // Step 2: t1 = x_hi_lo * EPSILON = (x_hi_lo << 32) - x_hi_lo
    // Using shift is ~10% faster than multiply
    let t1 = (x_hi_lo << 32).wrapping_sub(x_hi_lo);

    // Step 3: result = t0 + t1
    let (result, carry) = t0.overflowing_add(t1);
    if carry {
        result.wrapping_add(EPSILON)
    } else {
        result
    }
}

/// Inversion using optimized addition chain for a^(p-2).
/// Based on Plonky2's approach.
///
/// p - 2 = 0xFFFFFFFE_FFFFFFFF = 2^64 - 2^32 - 1
/// Binary structure: 32 ones, one zero, 31 ones
///
/// This uses approximately 72 multiplications (vs ~96 for binary exp).
#[inline(never)]
pub fn inv_addition_chain(base: u64) -> u64 {
    // Helper: square n times then multiply by tail
    #[inline(always)]
    fn exp_acc(base: u64, tail: u64, n: u32) -> u64 {
        let mut result = base;
        for _ in 0..n {
            result = GoldilocksField::square(&result);
        }
        GoldilocksField::mul(&result, &tail)
    }

    let x = base;
    let x2 = GoldilocksField::square(&x);

    // x^(2^2 - 1) = x^3
    let x3 = GoldilocksField::mul(&x2, &x);

    // x^(2^3 - 1) = x^7 = (x^3)^2 * x
    let x7 = exp_acc(x3, x, 1);

    // x^(2^6 - 1) = x^63 = (x^7)^8 * x^7
    let x63 = exp_acc(x7, x7, 3);

    // x^(2^12 - 1) = (x^63)^64 * x^63
    let x12m1 = exp_acc(x63, x63, 6);

    // x^(2^24 - 1) = (x^(2^12-1))^4096 * x^(2^12-1)
    let x24m1 = exp_acc(x12m1, x12m1, 12);

    // x^(2^30 - 1) = (x^(2^24-1))^64 * x^63
    let x30m1 = exp_acc(x24m1, x63, 6);

    // x^(2^31 - 1) = (x^(2^30-1))^2 * x
    let x31m1 = exp_acc(x30m1, x, 1);

    // Now we need x^(2^64 - 2^32 - 1)
    // = x^(2^64 - 1) / x^(2^32)
    // = x^((2^63-1)*2 + 1) / x^(2^32)
    //
    // Alternative decomposition:
    // p - 2 = 0xFFFFFFFE_FFFFFFFF
    // = (2^32 - 2) * 2^32 + (2^32 - 1)
    // = (2^31 - 1) * 2^33 + (2^32 - 1)
    //
    // x^(p-2) = x^((2^31-1) * 2^33) * x^(2^32-1)
    //         = (x^(2^31-1))^(2^33) * x^(2^32-1)

    // x^(2^32 - 1) = (x^(2^31-1))^2 * x
    let x32m1 = exp_acc(x31m1, x, 1);

    // (x^(2^31-1))^(2^33) = square x31m1 33 times
    let mut t = x31m1;
    for _ in 0..33 {
        t = GoldilocksField::square(&t);
    }

    // Final result: t * x^(2^32-1)
    GoldilocksField::mul(&t, &x32m1)
}

/// Compute a^(p-2) for field inversion using the optimized addition chain.
#[inline(always)]
fn exp_p_minus_2(base: u64) -> u64 {
    inv_addition_chain(base)
}

/// Type alias for Goldilocks field elements
pub type GoldilocksElement = FieldElement<GoldilocksField>;

impl GoldilocksElement {
    /// Create a new field element from a u64.
    pub fn from_canonical_u64(n: u64) -> Self {
        Self::from(n)
    }

    /// Get the canonical u64 representation in [0, p).
    pub fn canonical_u64(&self) -> u64 {
        GoldilocksField::canonical(self.value())
    }

    /// Convert to little-endian bytes.
    pub fn to_bytes_le(&self) -> [u8; 8] {
        self.canonical_u64().to_le_bytes()
    }

    /// Convert to big-endian bytes.
    pub fn to_bytes_be(&self) -> [u8; 8] {
        self.canonical_u64().to_be_bytes()
    }

    /// Create a field element from an i64.
    /// Negative values are converted to their field equivalents: -x becomes p - x.
    pub fn from_i64(value: i64) -> Self {
        Self::from(value)
    }
}

// =====================================================
// TRAIT IMPLEMENTATIONS FOR PROVER/VERIFIER
// =====================================================

impl ByteConversion for FieldElement<GoldilocksField> {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        self.canonical_u64().to_be_bytes().to_vec()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        self.canonical_u64().to_le_bytes().to_vec()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        let needed_bytes = bytes
            .get(0..8)
            .ok_or(crate::errors::ByteConversionError::FromBEBytesError)?;
        let value = u64::from_be_bytes(
            needed_bytes
                .try_into()
                .map_err(|_| crate::errors::ByteConversionError::FromBEBytesError)?,
        );
        Ok(Self::from(value))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError>
    where
        Self: Sized,
    {
        let needed_bytes = bytes
            .get(0..8)
            .ok_or(crate::errors::ByteConversionError::FromLEBytesError)?;
        let value = u64::from_le_bytes(
            needed_bytes
                .try_into()
                .map_err(|_| crate::errors::ByteConversionError::FromLEBytesError)?,
        );
        Ok(Self::from(value))
    }
}

#[cfg(feature = "alloc")]
impl AsBytes for FieldElement<GoldilocksField> {
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        ByteConversion::to_bytes_be(self)
    }
}

// Implement IsPrimeField for the native Goldilocks
use crate::errors::CreationError;
use crate::field::traits::IsPrimeField;

impl IsPrimeField for GoldilocksField {
    type CanonicalType = u64;

    #[inline(always)]
    fn canonical(a: &Self::BaseType) -> Self::CanonicalType {
        if *a >= GOLDILOCKS_PRIME {
            *a - GOLDILOCKS_PRIME
        } else {
            *a
        }
    }

    fn from_hex(hex_string: &str) -> Result<Self::BaseType, CreationError> {
        let hex = hex_string.strip_prefix("0x").unwrap_or(hex_string);
        u64::from_str_radix(hex, 16)
            .map(Self::from_u64)
            .map_err(|_| CreationError::InvalidHexString)
    }

    #[cfg(feature = "std")]
    fn to_hex(a: &Self::BaseType) -> String {
        format!("{:x}", Self::canonical(a))
    }

    fn field_bit_size() -> usize {
        64 // Goldilocks uses 64-bit representation
    }
}

// Implement IsFFTField for the native Goldilocks
use crate::field::traits::IsFFTField;

impl IsFFTField for GoldilocksField {
    /// Two-adicity of Goldilocks: p - 1 = 2^32 * (2^32 - 1)
    const TWO_ADICITY: u64 = 32;

    /// Primitive 2^32-th root of unity.
    /// This is the same value used in Plonky3.
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: u64 = 1753635133440165772;

    fn field_name() -> &'static str {
        "Goldilocks"
    }
}

use crate::field::fields::fft_friendly::u64_goldilocks_packed::PackedGoldilocks;
use crate::field::packed::HasPacking;

impl HasPacking for GoldilocksField {
    type Packing = PackedGoldilocks;
}

impl HasDefaultTranscript for GoldilocksField {
    fn get_random_field_element_from_rng(rng: &mut impl rand::Rng) -> FieldElement<Self> {
        let mut sample = [0u8; 8];
        loop {
            rng.fill(&mut sample);
            let int_sample = u64::from_be_bytes(sample);
            if int_sample < GOLDILOCKS_PRIME {
                return FieldElement::from(int_sample);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_basic() {
        let a = 5u64;
        let b = 7u64;
        assert_eq!(GoldilocksField::add(&a, &b), 12);
    }

    #[test]
    fn test_add_overflow() {
        let a = GOLDILOCKS_PRIME - 1;
        let b = 2u64;
        let result = GoldilocksField::add(&a, &b);
        assert_eq!(GoldilocksField::canonical(&result), 1);
    }

    #[test]
    fn test_sub_basic() {
        let a = 10u64;
        let b = 3u64;
        assert_eq!(GoldilocksField::sub(&a, &b), 7);
    }

    #[test]
    fn test_sub_underflow() {
        let a = 3u64;
        let b = 10u64;
        let result = GoldilocksField::sub(&a, &b);
        assert_eq!(GoldilocksField::canonical(&result), GOLDILOCKS_PRIME - 7);
    }

    #[test]
    fn test_mul_basic() {
        let a = 5u64;
        let b = 7u64;
        assert_eq!(GoldilocksField::mul(&a, &b), 35);
    }

    #[test]
    fn test_mul_large() {
        // Test with values that produce 128-bit result
        let a = 1u64 << 40;
        let b = 1u64 << 40;
        let result = GoldilocksField::mul(&a, &b);
        // (2^40)^2 = 2^80 mod p
        // 2^80 = 2^64 * 2^16 ≡ EPSILON * 2^16 (mod p)
        let expected = ((a as u128 * b as u128) % GOLDILOCKS_PRIME as u128) as u64;
        assert_eq!(GoldilocksField::canonical(&result), expected);
    }

    #[test]
    fn test_inv() {
        let a = 5u64;
        let a_inv = GoldilocksField::inv(&a).unwrap();
        let product = GoldilocksField::mul(&a, &a_inv);
        assert_eq!(GoldilocksField::canonical(&product), 1);
    }

    #[test]
    fn test_inv_larger() {
        let a = 123456789u64;
        let a_inv = GoldilocksField::inv(&a).unwrap();
        let product = GoldilocksField::mul(&a, &a_inv);
        assert_eq!(GoldilocksField::canonical(&product), 1);
    }

    #[test]
    fn test_zero_inv() {
        assert!(GoldilocksField::inv(&0).is_err());
    }

    #[test]
    fn test_neg() {
        let a = 5u64;
        let neg_a = GoldilocksField::neg(&a);
        let sum = GoldilocksField::add(&a, &neg_a);
        assert_eq!(GoldilocksField::canonical(&sum), 0);
    }

    #[test]
    fn test_primitive_root() {
        // The primitive root should have order 2^32
        let root =
            GoldilocksField::get_primitive_root_of_unity(GoldilocksField::TWO_ADICITY).unwrap();

        // root^(2^32) should be 1
        let mut result = *root.value();
        for _ in 0..32 {
            result = GoldilocksField::square(&result);
        }
        assert_eq!(GoldilocksField::canonical(&result), 1);
    }

    #[test]
    fn test_inv_addition_chain() {
        // Test addition chain inversion
        for a in [5u64, 123456789, GOLDILOCKS_PRIME - 1, 0xDEADBEEF, 1, 2] {
            let a_inv = inv_addition_chain(a);
            let product = GoldilocksField::mul(&a, &a_inv);
            assert_eq!(
                GoldilocksField::canonical(&product),
                1,
                "Failed for a = {}",
                a
            );
        }
    }

    #[test]
    fn test_square() {
        // Test that square matches mul(a, a)
        for a in [5u64, 123456789, GOLDILOCKS_PRIME - 1, 0xDEADBEEF, 1, 2] {
            let sq = GoldilocksField::square(&a);
            let mul = GoldilocksField::mul(&a, &a);
            assert_eq!(
                GoldilocksField::canonical(&sq),
                GoldilocksField::canonical(&mul),
                "Square mismatch for a = {}",
                a
            );
        }
    }

    // =========================================================================
    // Tests for From<i64> implementation
    // =========================================================================

    #[test]
    fn test_from_i64_positive() {
        // Positive i64 values should work like u64
        let fe_from_i64 = GoldilocksElement::from(42i64);
        let fe_from_u64 = GoldilocksElement::from(42u64);
        assert_eq!(fe_from_i64, fe_from_u64);
    }

    #[test]
    fn test_from_i64_zero() {
        let fe = GoldilocksElement::from(0i64);
        assert_eq!(fe, GoldilocksElement::zero());
    }

    #[test]
    fn test_from_i64_negative_one() {
        // -1 should equal p - 1
        let fe = GoldilocksElement::from(-1i64);
        let expected = GoldilocksElement::from(GOLDILOCKS_PRIME - 1);
        assert_eq!(fe, expected);

        // Also verify: -1 + 1 = 0
        let one = GoldilocksElement::one();
        assert_eq!(fe + one, GoldilocksElement::zero());
    }

    #[test]
    fn test_from_i64_negative_values() {
        // -x should equal p - x, which means x + (-x) = 0
        for x in [1i64, 5, 100, 1000, 123456789] {
            let pos = GoldilocksElement::from(x);
            let neg = GoldilocksElement::from(-x);
            assert_eq!(pos + neg, GoldilocksElement::zero(), "Failed for x = {}", x);
        }
    }

    #[test]
    fn test_from_i64_negative_equals_negation() {
        // FE::from(-x) should equal -FE::from(x)
        for x in [1i64, 42, 1000, 999999] {
            let from_neg = GoldilocksElement::from(-x);
            let neg_from = -GoldilocksElement::from(x);
            assert_eq!(from_neg, neg_from, "Failed for x = {}", x);
        }
    }

    #[test]
    fn test_from_i64_arithmetic() {
        // Test that arithmetic works correctly with i64-created elements
        // 5 - 10 = -5
        let five = GoldilocksElement::from(5i64);
        let ten = GoldilocksElement::from(10i64);
        let minus_five = GoldilocksElement::from(-5i64);

        assert_eq!(five - ten, minus_five);
    }

    #[test]
    fn test_from_i64_large_negative() {
        // Test with larger negative values
        let large_neg = GoldilocksElement::from(-1_000_000_000i64);
        let large_pos = GoldilocksElement::from(1_000_000_000i64);

        // They should sum to zero
        assert_eq!(large_neg + large_pos, GoldilocksElement::zero());

        // And the negative should equal the negation of positive
        assert_eq!(large_neg, -large_pos);
    }

    #[test]
    fn test_from_i64_min_value() {
        // Test with i64::MIN - this is a valid field element
        let min_val = GoldilocksElement::from(i64::MIN);
        // i64::MIN = -2^63, so this should be p - 2^63
        let expected_val = GOLDILOCKS_PRIME - (1u64 << 63);
        let expected = GoldilocksElement::from(expected_val);
        assert_eq!(min_val, expected);
    }

    #[test]
    fn test_from_i64_max_value() {
        // Test with i64::MAX
        let max_val = GoldilocksElement::from(i64::MAX);
        let expected = GoldilocksElement::from(i64::MAX as u64);
        assert_eq!(max_val, expected);
    }
}
