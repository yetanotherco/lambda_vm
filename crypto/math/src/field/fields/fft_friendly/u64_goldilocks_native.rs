//! Optimized Goldilocks field implementation using native u64 arithmetic.
//!
//! This implementation uses direct u64 representation (no Montgomery form) and exploits
//! the special structure of the Goldilocks prime p = 2^64 - 2^32 + 1 for fast reduction.
//!
//! Key properties:
//! - EPSILON = 2^32 - 1 = -2^64 mod p
//! - φ = 2^32 is a 6th root of unity (φ^3 = -1, φ^6 = 1)
//! - 2^96 ≡ -1 (mod p)
//!
//! Based on techniques from:
//! - Plonky3: <https://github.com/Plonky3/Plonky3>
//! - Plonky2: <https://github.com/0xPolygonZero/plonky2>
//! - Remco Bloemen: <https://xn--2-umb.com/23/gold-reduce/>

use crate::field::{element::FieldElement, errors::FieldError, traits::IsField};

#[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
use super::u64_goldilocks_asm;


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
    ///
    /// With asm-arm64 feature: Uses optimized SBC mask trick (~10% faster than native).
    #[inline(always)]
    fn add(a: &u64, b: &u64) -> u64 {
        #[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
        {
            u64_goldilocks_asm::add_fast(*a, *b)
        }
        #[cfg(not(all(feature = "asm-arm64", target_arch = "aarch64")))]
        {
            let (sum, over) = a.overflowing_add(*b);
            let (sum, over2) = sum.overflowing_add((over as u64) * EPSILON);
            // Second overflow is rare but possible
            if over2 {
                sum.wrapping_add(EPSILON)
            } else {
                sum
            }
        }
    }

    /// Subtraction with underflow handling.
    /// If a - b underflows, we subtract EPSILON (since -2^64 ≡ -EPSILON mod p)
    ///
    /// With asm-arm64 feature: Uses optimized SBC mask trick (~10% faster than native).
    #[inline(always)]
    fn sub(a: &u64, b: &u64) -> u64 {
        #[cfg(all(feature = "asm-arm64", target_arch = "aarch64"))]
        {
            u64_goldilocks_asm::sub_fast(*a, *b)
        }
        #[cfg(not(all(feature = "asm-arm64", target_arch = "aarch64")))]
        {
            let (diff, under) = a.overflowing_sub(*b);
            let (diff, under2) = diff.overflowing_sub((under as u64) * EPSILON);
            // Second underflow is rare but possible
            if under2 {
                diff.wrapping_sub(EPSILON)
            } else {
                diff
            }
        }
    }

    /// Multiplication using 128-bit intermediate and fast reduction.
    ///
    /// Note: Benchmarks show LLVM generates better code for this operation
    /// than inline assembly, so we use native Rust even with asm-arm64 feature.
    #[inline(always)]
    fn mul(a: &u64, b: &u64) -> u64 {
        // LLVM generates optimal MUL+UMULH code for this, so we use native Rust
        let product = (*a as u128) * (*b as u128);
        reduce128(product)
    }

    /// Optimized squaring.
    ///
    /// Note: Benchmarks show native Rust outperforms inline ASM for squaring,
    /// so we always use native implementation.
    #[inline(always)]
    fn square(a: &u64) -> u64 {
        let a_val = *a;
        let product = (a_val as u128) * (a_val as u128);
        reduce128(product)
    }

    /// Negation: -a = p - a (or 0 if a = 0)
    #[inline(always)]
    fn neg(a: &u64) -> u64 {
        if *a == 0 {
            0
        } else {
            // First canonicalize, then negate
            let canonical = canonicalize(*a);
            if canonical == 0 {
                0
            } else {
                GOLDILOCKS_PRIME - canonical
            }
        }
    }

    /// Multiplicative inverse using Fermat's little theorem: a^(-1) = a^(p-2)
    fn inv(a: &u64) -> Result<u64, FieldError> {
        let canonical = canonicalize(*a);
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
        canonicalize(*a) == canonicalize(*b)
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
    let t0 = if borrow {
        t0.wrapping_sub(EPSILON)
    } else {
        t0
    };

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

/// Canonicalize a field element to [0, p).
/// This is needed for comparisons and serialization.
#[inline(always)]
fn canonicalize(x: u64) -> u64 {
    // Since values can be up to 2^64 - 1, we may need multiple subtractions
    // But in practice, after proper reduction, at most one subtraction is needed
    if x >= GOLDILOCKS_PRIME {
        x - GOLDILOCKS_PRIME
    } else {
        x
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
    pub fn to_canonical_u64(&self) -> u64 {
        canonicalize(*self.value())
    }

    /// Convert to little-endian bytes.
    pub fn to_bytes_le(&self) -> [u8; 8] {
        self.to_canonical_u64().to_le_bytes()
    }

    /// Convert to big-endian bytes.
    pub fn to_bytes_be(&self) -> [u8; 8] {
        self.to_canonical_u64().to_be_bytes()
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
        "GoldilocksNative"
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
        assert_eq!(canonicalize(result), 1);
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
        assert_eq!(canonicalize(result), GOLDILOCKS_PRIME - 7);
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
        assert_eq!(canonicalize(result), expected);
    }

    #[test]
    fn test_inv() {
        let a = 5u64;
        let a_inv = GoldilocksField::inv(&a).unwrap();
        let product = GoldilocksField::mul(&a, &a_inv);
        assert_eq!(canonicalize(product), 1);
    }

    #[test]
    fn test_inv_larger() {
        let a = 123456789u64;
        let a_inv = GoldilocksField::inv(&a).unwrap();
        let product = GoldilocksField::mul(&a, &a_inv);
        assert_eq!(canonicalize(product), 1);
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
        assert_eq!(canonicalize(sum), 0);
    }

    #[test]
    fn test_primitive_root() {
        // The primitive root should have order 2^32
        let root = GoldilocksField::get_primitive_root_of_unity(GoldilocksField::TWO_ADICITY).unwrap();

        // root^(2^32) should be 1
        let mut result = *root.value();
        for _ in 0..32 {
            result = GoldilocksField::square(&result);
        }
        assert_eq!(canonicalize(result), 1);
    }

    #[test]
    fn test_inv_addition_chain() {
        // Test addition chain inversion
        for a in [5u64, 123456789, GOLDILOCKS_PRIME - 1, 0xDEADBEEF, 1, 2] {
            let a_inv = inv_addition_chain(a);
            let product = GoldilocksField::mul(&a, &a_inv);
            assert_eq!(canonicalize(product), 1, "Failed for a = {}", a);
        }
    }

    #[test]
    fn test_square() {
        // Test that square matches mul(a, a)
        for a in [5u64, 123456789, GOLDILOCKS_PRIME - 1, 0xDEADBEEF, 1, 2] {
            let sq = GoldilocksField::square(&a);
            let mul = GoldilocksField::mul(&a, &a);
            assert_eq!(canonicalize(sq), canonicalize(mul), "Square mismatch for a = {}", a);
        }
    }
}
