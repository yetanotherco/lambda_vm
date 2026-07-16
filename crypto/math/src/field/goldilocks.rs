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

use core::hint::unreachable_unchecked;

use crate::field::traits::HasDefaultTranscript;
use crate::field::{element::FieldElement, errors::FieldError, traits::IsField};
use crate::traits::{AsBytes, ByteConversion};

// =====================================================
// COMPILER HINTS (inspired by Plonky3)
// =====================================================

/// Hint to the compiler that a branch is unlikely to be taken.
/// The empty asm block acts as a barrier that prevents the compiler from
/// converting the branch into a conditional move, which is slower when
/// the branch is highly predictable.
#[inline(always)]
fn branch_hint() {
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "x86",
        target_arch = "x86_64",
    ))]
    unsafe {
        core::arch::asm!("", options(nomem, nostack, preserves_flags));
    }
}

/// Inform the compiler that a condition is always true.
///
/// # Safety
/// The caller must guarantee that `p` is true.
#[inline(always)]
const unsafe fn assume(p: bool) {
    debug_assert!(p);
    if !p {
        unsafe {
            unreachable_unchecked();
        }
    }
}

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

    /// Addition with branch hint for rare double-overflow.
    /// Compiles to a 3-instruction common path (add + csel + adds on ARM)
    /// with a predicted-not-taken branch for the exceedingly rare double overflow.
    #[inline(always)]
    fn add(a: &u64, b: &u64) -> u64 {
        let (sum, over) = a.overflowing_add(*b);
        let (mut sum, over) = sum.overflowing_add((over as u64) * EPSILON);
        if over {
            // Double overflow requires a + b >= 2^65 - EPSILON.
            // If either input <= p, then a + b <= p + (2^64-1) = 2^65 - EPSILON - 1,
            // which is below the threshold. Therefore both must exceed p.
            unsafe {
                assume(*a > GOLDILOCKS_PRIME && *b > GOLDILOCKS_PRIME);
            }
            branch_hint();
            // After double overflow, sum < EPSILON, so sum + EPSILON < 2^64.
            sum += EPSILON;
        }
        sum
    }

    /// Subtraction with branch hint for rare double-underflow.
    #[inline(always)]
    fn sub(a: &u64, b: &u64) -> u64 {
        let (diff, under) = a.overflowing_sub(*b);
        let (mut diff, under) = diff.overflowing_sub((under as u64) * EPSILON);
        if under {
            // Double underflow requires a - b + 2^64 < EPSILON, i.e., b > a + p.
            // Since b < 2^64 = p + EPSILON, we get a < EPSILON - 1.
            // At a = EPSILON - 1, the minimum b for first underflow already gives
            // diff1 = EPSILON, so diff1 - EPSILON = 0 (no second underflow).
            unsafe {
                assume(*a < EPSILON - 1 && *b > GOLDILOCKS_PRIME);
            }
            branch_hint();
            diff -= EPSILON;
        }
        diff
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
/// Uses the identities: 2^64 ≡ 2^32 - 1 (mod p), 2^96 ≡ -1 (mod p).
/// Branch hints mark rare borrow/carry paths for better branch prediction.
#[inline(always)]
fn reduce128(x: u128) -> u64 {
    let (x_lo, x_hi) = (x as u64, (x >> 64) as u64);
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & EPSILON;

    // 2^96 ≡ -1 (mod p), so x_hi_hi * 2^96 becomes -x_hi_hi
    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    if borrow {
        branch_hint();
        t0 -= EPSILON; // Cannot underflow
    }

    // 2^64 ≡ EPSILON (mod p), so x_hi_lo * 2^64 = x_hi_lo * EPSILON
    // Compute as (x_hi_lo << 32) - x_hi_lo to avoid a multiply
    let t1 = (x_hi_lo << 32).wrapping_sub(x_hi_lo);

    // Final addition with overflow correction
    // Safety: t0 + t1 < 2^64 + ORDER
    unsafe { add_no_canonicalize_trashing_input(t0, t1) }
}

/// Fast modular addition: returns (x + y) mod p, assuming x + y < 2^64 + ORDER.
/// On x86_64, uses inline asm (add + sbb trick) for 2-instruction modular add.
/// On other architectures, uses portable overflowing_add + conditional correction.
///
/// # Safety
/// Caller must ensure x + y < 2^64 + ORDER.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn add_no_canonicalize_trashing_input(x: u64, y: u64) -> u64 {
    let res_wrapped: u64;
    let adjustment: u64;
    unsafe {
        core::arch::asm!(
            "add {0}, {1}",
            // sbb {1:e}, {1:e} sets the low 32 bits to 0xFFFFFFFF on carry (= NEG_ORDER),
            // or 0 otherwise. The high 32 bits are zeroed by the 32-bit register write.
            "sbb {1:e}, {1:e}",
            inlateout(reg) x => res_wrapped,
            inlateout(reg) y => adjustment,
            options(pure, nomem, nostack),
        );
    }
    res_wrapped + adjustment
}

#[inline(always)]
#[cfg(not(target_arch = "x86_64"))]
unsafe fn add_no_canonicalize_trashing_input(x: u64, y: u64) -> u64 {
    let (res_wrapped, carry) = x.overflowing_add(y);
    res_wrapped.wrapping_add(EPSILON * (carry as u64))
}

// =====================================================
// FUSED MULTIPLY-ACCUMULATE (inspired by Plonky3)
// =====================================================

/// Compute a0*b0 + a1*b1 mod p in a single reduction pass.
///
/// Instead of reducing each product separately (2 reduce128 calls),
/// this sums the u128 products and reduces once. When the sum overflows u128,
/// we correct by adding 2^128 mod p = EPSILON^2 = (2^32 - 1)^2.
///
/// This is the critical building block for extension field multiplication:
/// each Fp2 mul needs two dot products instead of three separate mul+reduce.
#[inline(always)]
pub(crate) fn dot_product_2(a0: u64, b0: u64, a1: u64, b1: u64) -> u64 {
    let prod0 = (a0 as u128) * (b0 as u128);
    let prod1 = (a1 as u128) * (b1 as u128);
    let (sum, overflow) = prod0.overflowing_add(prod1);

    let reduced = reduce128(sum);

    if overflow {
        // True value is sum + 2^128. Since 2^128 mod p = EPSILON^2,
        // add EPSILON^2 = (2^32-1)^2 = 2^64 - 2^33 + 1.
        // Safety: reduced < 2^64 (it's a u64), EPSILON_SQ < p,
        // so reduced + EPSILON_SQ < 2^64 + p, satisfying add_no_canonicalize's precondition.
        branch_hint();
        const EPSILON_SQ: u64 = EPSILON.wrapping_mul(EPSILON);
        unsafe { add_no_canonicalize_trashing_input(reduced, EPSILON_SQ) }
    } else {
        reduced
    }
}

/// Compute a0*b0 + a1*b1 + a2*b2 mod p in a single reduction pass.
///
/// Accumulates three u128 products, tracking overflow count (at most 2).
/// Each overflow adds 2^128 mod p = EPSILON^2 to the result.
/// This is the critical building block for Fp3 multiplication (the extension
/// field used by the VM's STARK prover).
#[inline(always)]
pub(crate) fn dot_product_3(a0: u64, b0: u64, a1: u64, b1: u64, a2: u64, b2: u64) -> u64 {
    let prod0 = (a0 as u128) * (b0 as u128);
    let prod1 = (a1 as u128) * (b1 as u128);
    let prod2 = (a2 as u128) * (b2 as u128);

    let (sum01, over1) = prod0.overflowing_add(prod1);
    let (sum012, over2) = sum01.overflowing_add(prod2);
    let overflow_count = (over1 as u64) + (over2 as u64);

    let mut reduced = reduce128(sum012);

    if overflow_count > 0 {
        // Each overflow represents +2^128 to the true sum.
        // 2^128 mod p = EPSILON^2 = (2^32 - 1)^2 = 2^64 - 2^33 + 1.
        // Safety: reduced < 2^64, EPSILON_SQ < p, so sum < 2^64 + p.
        branch_hint();
        const EPSILON_SQ: u64 = EPSILON.wrapping_mul(EPSILON);
        reduced = unsafe { add_no_canonicalize_trashing_input(reduced, EPSILON_SQ) };
        if overflow_count > 1 {
            branch_hint();
            reduced = unsafe { add_no_canonicalize_trashing_input(reduced, EPSILON_SQ) };
        }
    }

    reduced
}

/// Multiply a raw u64 field element by 7 (the Fp2 non-residue).
/// Uses 7 = 8 - 1 for a straight-line computation.
#[inline(always)]
pub(crate) fn mul_by_7_raw(a: u64) -> u64 {
    let a2 = GoldilocksField::double(&a);
    let a4 = GoldilocksField::double(&a2);
    let a8 = GoldilocksField::double(&a4);
    GoldilocksField::sub(&a8, &a)
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
    const BYTE_LEN: usize = 8;

    #[inline(always)]
    fn write_bytes_be(&self, buf: &mut [u8]) {
        debug_assert!(buf.len() >= 8);
        buf[..8].copy_from_slice(&self.canonical_u64().to_be_bytes());
    }

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

    #[inline(always)]
    fn stream_bytes(&self, sink: &mut dyn FnMut(&[u8])) {
        sink(&self.canonical_u64().to_be_bytes());
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
