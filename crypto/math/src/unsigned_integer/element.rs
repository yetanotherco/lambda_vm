use core::cmp::Ordering;
use core::convert::From;
use core::ops::{
    Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul, Shl, Shr, ShrAssign,
    Sub,
};

#[cfg(feature = "proptest")]
use proptest::{
    arbitrary::Arbitrary,
    prelude::any,
    strategy::{SBoxedStrategy, Strategy},
};

use crate::errors::ByteConversionError;
use crate::errors::CreationError;
#[cfg(feature = "alloc")]
use crate::traits::AsBytes;
use crate::traits::ByteConversion;
use crate::unsigned_integer::traits::IsUnsignedInteger;

use core::fmt::{self, Debug, Display, LowerHex, UpperHex};

pub type U384 = UnsignedInteger<6>;
pub type U256 = UnsignedInteger<4>;
pub type U128 = UnsignedInteger<2>;
pub type U64 = UnsignedInteger<1>;

/// A big unsigned integer in base 2^{64} represented
/// as fixed-size array `limbs` of `u64` components.
/// The most significant bit is in the left-most position.
/// That is, the array `[a_n, ..., a_0]` represents the
/// integer 2^{64 * n} * a_n + ... + 2^{64} * a_1 + a_0.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct UnsignedInteger<const NUM_LIMBS: usize> {
    pub limbs: [u64; NUM_LIMBS],
}

impl<const NUM_LIMBS: usize> Default for UnsignedInteger<NUM_LIMBS> {
    fn default() -> Self {
        Self {
            limbs: [0; NUM_LIMBS],
        }
    }
}

// NOTE: manually implementing `PartialOrd` may seem unorthodox, but the
// derived implementation had terrible performance.
#[allow(clippy::non_canonical_partial_ord_impl)]
impl<const NUM_LIMBS: usize> PartialOrd for UnsignedInteger<NUM_LIMBS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let mut i = 0;
        while i < NUM_LIMBS {
            if self.limbs[i] != other.limbs[i] {
                return Some(self.limbs[i].cmp(&other.limbs[i]));
            }
            i += 1;
        }
        Some(Ordering::Equal)
    }
}

// NOTE: because we implemented `PartialOrd`, clippy asks us to implement
// this manually too.
impl<const NUM_LIMBS: usize> Ord for UnsignedInteger<NUM_LIMBS> {
    fn cmp(&self, other: &Self) -> Ordering {
        let mut i = 0;
        while i < NUM_LIMBS {
            if self.limbs[i] != other.limbs[i] {
                return self.limbs[i].cmp(&other.limbs[i]);
            }
            i += 1;
        }
        Ordering::Equal
    }
}

/// impl from u128 for UnSignedInteger
impl<const NUM_LIMBS: usize> From<u128> for UnsignedInteger<NUM_LIMBS> {
    fn from(value: u128) -> Self {
        let mut limbs = [0u64; NUM_LIMBS];
        limbs[NUM_LIMBS - 1] = value as u64;
        limbs[NUM_LIMBS - 2] = (value >> 64) as u64;
        UnsignedInteger { limbs }
    }
}

/// impl from u64 for UnSignedInteger
impl<const NUM_LIMBS: usize> From<u64> for UnsignedInteger<NUM_LIMBS> {
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

/// impl from u16 for UnSignedInteger
impl<const NUM_LIMBS: usize> From<u16> for UnsignedInteger<NUM_LIMBS> {
    fn from(value: u16) -> Self {
        let mut limbs = [0u64; NUM_LIMBS];
        limbs[NUM_LIMBS - 1] = value as u64;
        UnsignedInteger { limbs }
    }
}

impl<const NUM_LIMBS: usize> From<&str> for UnsignedInteger<NUM_LIMBS> {
    fn from(hex_str: &str) -> Self {
        Self::from_hex_unchecked(hex_str)
    }
}

impl<const NUM_LIMBS: usize> Display for UnsignedInteger<NUM_LIMBS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x")?;

        let mut limbs_iterator = self.limbs.iter().skip_while(|limb| **limb == 0).peekable();

        if limbs_iterator.peek().is_none() {
            write!(f, "0")?;
        } else {
            if let Some(most_significant_limb) = limbs_iterator.next() {
                write!(f, "{most_significant_limb:x}")?;
            }

            for limb in limbs_iterator {
                write!(f, "{limb:016x}")?;
            }
        }

        Ok(())
    }
}

impl<const NUM_LIMBS: usize> LowerHex for UnsignedInteger<NUM_LIMBS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "0x")?;
        }

        let mut limbs_iterator = self.limbs.iter().skip_while(|limb| **limb == 0).peekable();

        if limbs_iterator.peek().is_none() {
            write!(f, "0")?;
        } else {
            if let Some(most_significant_limb) = limbs_iterator.next() {
                write!(f, "{most_significant_limb:x}")?;
            }

            for limb in limbs_iterator {
                write!(f, "{limb:016x}")?;
            }
        }

        Ok(())
    }
}

impl<const NUM_LIMBS: usize> UpperHex for UnsignedInteger<NUM_LIMBS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "0X")?;
        }

        let mut limbs_iterator = self.limbs.iter().skip_while(|limb| **limb == 0).peekable();

        if limbs_iterator.peek().is_none() {
            write!(f, "0")?;
        } else {
            if let Some(most_significant_limb) = limbs_iterator.next() {
                write!(f, "{most_significant_limb:X}")?;
            }

            for limb in limbs_iterator {
                write!(f, "{limb:016X}")?;
            }
        }

        Ok(())
    }
}

// impl Add for both references and variables

impl<const NUM_LIMBS: usize> Add<&UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn add(self, other: &UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        let (result, overflow) = UnsignedInteger::add(self, other);
        debug_assert!(!overflow, "UnsignedInteger addition overflow.");
        result
    }
}

impl<const NUM_LIMBS: usize> Add<UnsignedInteger<NUM_LIMBS>> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn add(self, other: UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        &self + &other
    }
}

impl<const NUM_LIMBS: usize> Add<&UnsignedInteger<NUM_LIMBS>> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn add(self, other: &Self) -> Self {
        &self + other
    }
}

impl<const NUM_LIMBS: usize> Add<UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn add(self, other: UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        self + &other
    }
}

// impl Sub

impl<const NUM_LIMBS: usize> Sub<&UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn sub(self, other: &UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        let (result, overflow) = UnsignedInteger::sub(self, other);
        debug_assert!(!overflow, "UnsignedInteger subtraction overflow.");
        result
    }
}

impl<const NUM_LIMBS: usize> Sub<UnsignedInteger<NUM_LIMBS>> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn sub(self, other: UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        &self - &other
    }
}

impl<const NUM_LIMBS: usize> Sub<&UnsignedInteger<NUM_LIMBS>> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    fn sub(self, other: &Self) -> Self {
        &self - other
    }
}

impl<const NUM_LIMBS: usize> Sub<UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn sub(self, other: UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        self - &other
    }
}

/// Multi-precision multiplication.
/// Algorithm 14.12 of "Handbook of Applied Cryptography" (https://cacr.uwaterloo.ca/hac/)
impl<const NUM_LIMBS: usize> Mul<&UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;

    #[inline(always)]
    fn mul(self, other: &UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        let (mut n, mut t) = (0, 0);
        for i in (0..NUM_LIMBS).rev() {
            if self.limbs[i] != 0u64 {
                n = NUM_LIMBS - 1 - i;
            }
            if other.limbs[i] != 0u64 {
                t = NUM_LIMBS - 1 - i;
            }
        }
        debug_assert!(
            n + t < NUM_LIMBS,
            "UnsignedInteger multiplication overflow."
        );

        // 1.
        let mut limbs = [0u64; NUM_LIMBS];
        // 2.
        let mut carry = 0u128;
        for i in 0..=t {
            // 2.2
            for j in 0..=n {
                let uv = (limbs[NUM_LIMBS - 1 - (i + j)] as u128)
                    + (self.limbs[NUM_LIMBS - 1 - j] as u128)
                        * (other.limbs[NUM_LIMBS - 1 - i] as u128)
                    + carry;
                carry = uv >> 64;
                limbs[NUM_LIMBS - 1 - (i + j)] = uv as u64;
            }
            if i + n + 1 < NUM_LIMBS {
                // 2.3
                limbs[NUM_LIMBS - 1 - (i + n + 1)] = carry as u64;
                carry = 0;
            }
        }
        assert_eq!(carry, 0, "UnsignedInteger multiplication overflow.");
        // 3.
        Self::Output { limbs }
    }
}

impl<const NUM_LIMBS: usize> Mul<UnsignedInteger<NUM_LIMBS>> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn mul(self, other: UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        &self * &other
    }
}

impl<const NUM_LIMBS: usize> Mul<&UnsignedInteger<NUM_LIMBS>> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn mul(self, other: &Self) -> Self {
        &self * other
    }
}

impl<const NUM_LIMBS: usize> Mul<UnsignedInteger<NUM_LIMBS>> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn mul(self, other: UnsignedInteger<NUM_LIMBS>) -> UnsignedInteger<NUM_LIMBS> {
        self * &other
    }
}

impl<const NUM_LIMBS: usize> Shl<usize> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn shl(self, times: usize) -> UnsignedInteger<NUM_LIMBS> {
        self.const_shl(times)
    }
}

impl<const NUM_LIMBS: usize> Shl<usize> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn shl(self, times: usize) -> UnsignedInteger<NUM_LIMBS> {
        &self << times
    }
}

// impl Shr

impl<const NUM_LIMBS: usize> Shr<usize> for &UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn shr(self, times: usize) -> UnsignedInteger<NUM_LIMBS> {
        self.const_shr(times)
    }
}

impl<const NUM_LIMBS: usize> Shr<usize> for UnsignedInteger<NUM_LIMBS> {
    type Output = UnsignedInteger<NUM_LIMBS>;
    #[inline(always)]
    fn shr(self, times: usize) -> UnsignedInteger<NUM_LIMBS> {
        &self >> times
    }
}

impl<const NUM_LIMBS: usize> ShrAssign<usize> for UnsignedInteger<NUM_LIMBS> {
    fn shr_assign(&mut self, times: usize) {
        debug_assert!(
            times < 64 * NUM_LIMBS,
            "UnsignedInteger shift left overflows."
        );

        let (a, b) = (times / 64, times % 64);

        if b == 0 {
            self.limbs.copy_within(..NUM_LIMBS - a, a);
        } else {
            for i in (a + 1..NUM_LIMBS).rev() {
                self.limbs[i] = (self.limbs[i - a] >> b) | (self.limbs[i - a - 1] << (64 - b));
            }
            self.limbs[a] = self.limbs[0] >> b;
        }

        for limb in self.limbs.iter_mut().take(a) {
            *limb = 0;
        }
    }
}

/// Impl BitAnd
impl<const NUM_LIMBS: usize> BitAnd for UnsignedInteger<NUM_LIMBS> {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        let mut result = self;
        result &= rhs;
        result
    }
}

impl<const NUM_LIMBS: usize> BitAndAssign for UnsignedInteger<NUM_LIMBS> {
    fn bitand_assign(&mut self, rhs: Self) {
        for (a_i, b_i) in self.limbs.iter_mut().zip(rhs.limbs.iter()) {
            *a_i &= b_i;
        }
    }
}

/// Impl BitOr
impl<const NUM_LIMBS: usize> BitOr for UnsignedInteger<NUM_LIMBS> {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        let mut result = self;
        result |= rhs;
        result
    }
}

impl<const NUM_LIMBS: usize> BitOrAssign for UnsignedInteger<NUM_LIMBS> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        for (a_i, b_i) in self.limbs.iter_mut().zip(rhs.limbs.iter()) {
            *a_i |= b_i;
        }
    }
}

/// Impl BitXor
impl<const NUM_LIMBS: usize> BitXor for UnsignedInteger<NUM_LIMBS> {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut result = self;
        result ^= rhs;
        result
    }
}

impl<const NUM_LIMBS: usize> BitXorAssign for UnsignedInteger<NUM_LIMBS> {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        for (a_i, b_i) in self.limbs.iter_mut().zip(rhs.limbs.iter()) {
            *a_i ^= b_i;
        }
    }
}

impl<const NUM_LIMBS: usize> UnsignedInteger<NUM_LIMBS> {
    pub const fn from_limbs(limbs: [u64; NUM_LIMBS]) -> Self {
        Self { limbs }
    }

    #[inline(always)]
    pub const fn from_u64(value: u64) -> Self {
        let mut limbs = [0u64; NUM_LIMBS];
        limbs[NUM_LIMBS - 1] = value;
        UnsignedInteger { limbs }
    }

    #[inline(always)]
    pub const fn from_u128(value: u128) -> Self {
        let mut limbs = [0u64; NUM_LIMBS];
        limbs[NUM_LIMBS - 1] = value as u64;
        limbs[NUM_LIMBS - 2] = (value >> 64) as u64;
        UnsignedInteger { limbs }
    }

    #[inline(always)]
    const fn is_hex_string(string: &str) -> bool {
        let len: usize = string.len();
        let bytes = string.as_bytes();
        let mut i = 0;

        while i < len {
            match bytes[i] {
                b'0'..=b'9' => (),
                b'a'..=b'f' => (),
                b'A'..=b'F' => (),
                _ => return false,
            }
            i += 1;
        }

        true
    }

    /// Creates an `UnsignedInteger` from a hexstring. It can contain `0x` or not.
    /// Returns an `CreationError::InvalidHexString`if the value is not a hexstring.
    /// Returns a `CreationError::EmptyString` if the input string is empty.
    /// Returns a `CreationError::HexStringIsTooBig` if the the input hex string is bigger
    /// than the maximum amount of characters for this element.
    pub fn from_hex(value: &str) -> Result<Self, CreationError> {
        let mut string = value;
        let mut char_iterator = value.chars();
        if string.len() > 2
            && char_iterator.next().unwrap() == '0'
            && char_iterator.next().unwrap() == 'x'
        {
            string = &string[2..];
        }
        if string.is_empty() {
            return Err(CreationError::EmptyString);
        }
        if !Self::is_hex_string(string) {
            return Err(CreationError::InvalidHexString);
        }

        // Limbs are of 64 bits - 8 bytes
        // We have 16 nibbles per bytes
        let max_amount_of_hex_chars = NUM_LIMBS * 16;
        if string.len() > max_amount_of_hex_chars {
            return Err(CreationError::HexStringIsTooBig);
        }

        Ok(Self::from_hex_unchecked(string))
    }

    /// Creates an `UnsignedInteger` from a hexstring
    /// # Panics
    /// Panics if value is not a hexstring. It can contain `0x` or not.
    pub const fn from_hex_unchecked(value: &str) -> Self {
        let mut result = [0u64; NUM_LIMBS];
        let mut limb = 0;
        let mut limb_index = NUM_LIMBS - 1;
        let mut shift = 0;

        let value_bytes = value.as_bytes();

        // Remove "0x" if it's at the beginning of the string
        let mut i = 0;
        if value_bytes.len() > 2 && value_bytes[0] == b'0' && value_bytes[1] == b'x' {
            i = 2;
        }

        let mut j = value_bytes.len();
        while j > i {
            j -= 1;
            limb |= match value_bytes[j] {
                c @ b'0'..=b'9' => (c as u64 - b'0' as u64) << shift,
                c @ b'a'..=b'f' => (c as u64 - b'a' as u64 + 10) << shift,
                c @ b'A'..=b'F' => (c as u64 - b'A' as u64 + 10) << shift,
                _ => panic!("Malformed hex expression."),
            };
            shift += 4;
            if shift == 64 && limb_index > 0 {
                result[limb_index] = limb;
                limb = 0;
                limb_index -= 1;
                shift = 0;
            }
        }

        result[limb_index] = limb;
        UnsignedInteger { limbs: result }
    }

    /// Creates a hexstring from a `FieldElement` without `0x`.
    #[cfg(feature = "std")]
    pub fn to_hex(&self) -> String {
        let mut hex_string = String::new();
        for &limb in self.limbs.iter() {
            hex_string.push_str(&format!("{limb:016X}"));
        }
        hex_string.trim_start_matches('0').to_string()
    }

    pub const fn const_ne(a: &UnsignedInteger<NUM_LIMBS>, b: &UnsignedInteger<NUM_LIMBS>) -> bool {
        let mut i = 0;
        while i < NUM_LIMBS {
            if a.limbs[i] != b.limbs[i] {
                return true;
            }
            i += 1;
        }
        false
    }

    pub const fn const_le(a: &UnsignedInteger<NUM_LIMBS>, b: &UnsignedInteger<NUM_LIMBS>) -> bool {
        let mut i = 0;
        while i < NUM_LIMBS {
            if a.limbs[i] < b.limbs[i] {
                return true;
            } else if a.limbs[i] > b.limbs[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    pub const fn const_shl(self, times: usize) -> Self {
        debug_assert!(
            times < 64 * NUM_LIMBS,
            "UnsignedInteger shift left overflows."
        );
        let mut limbs = [0u64; NUM_LIMBS];
        let (a, b) = (times / 64, times % 64);

        if b == 0 {
            let mut i = 0;
            while i < NUM_LIMBS - a {
                limbs[i] = self.limbs[a + i];
                i += 1;
            }
            Self { limbs }
        } else {
            limbs[NUM_LIMBS - 1 - a] = self.limbs[NUM_LIMBS - 1] << b;
            let mut i = a + 1;
            while i < NUM_LIMBS {
                limbs[NUM_LIMBS - 1 - i] = (self.limbs[NUM_LIMBS - 1 - i + a] << b)
                    | (self.limbs[NUM_LIMBS - i + a] >> (64 - b));
                i += 1;
            }
            Self { limbs }
        }
    }

    pub const fn const_shr(self, times: usize) -> UnsignedInteger<NUM_LIMBS> {
        debug_assert!(
            times < 64 * NUM_LIMBS,
            "UnsignedInteger shift right overflows."
        );

        let mut limbs = [0u64; NUM_LIMBS];
        let (a, b) = (times / 64, times % 64);

        if b == 0 {
            let mut i = 0;
            while i < NUM_LIMBS - a {
                limbs[a + i] = self.limbs[i];
                i += 1;
            }
            Self { limbs }
        } else {
            limbs[a] = self.limbs[0] >> b;
            let mut i = a + 1;
            while i < NUM_LIMBS {
                limbs[i] = (self.limbs[i - a - 1] << (64 - b)) | (self.limbs[i - a] >> b);
                i += 1;
            }
            Self { limbs }
        }
    }

    pub const fn add(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
    ) -> (UnsignedInteger<NUM_LIMBS>, bool) {
        let mut limbs = [0u64; NUM_LIMBS];
        let mut carry = 0u64;
        let mut i = NUM_LIMBS;
        while i > 0 {
            let (x, cb) = a.limbs[i - 1].overflowing_add(b.limbs[i - 1]);
            let (x, cc) = x.overflowing_add(carry);
            limbs[i - 1] = x;
            carry = (cb | cc) as u64;
            i -= 1;
        }
        (UnsignedInteger { limbs }, carry > 0)
    }

    pub fn double(a: &UnsignedInteger<NUM_LIMBS>) -> (UnsignedInteger<NUM_LIMBS>, bool) {
        let mut cloned = *a;
        let overflow = cloned.double_in_place();
        (cloned, overflow)
    }

    pub fn double_in_place(&mut self) -> bool {
        let mut msb_of_previous_limb = 0;
        for i in (0..NUM_LIMBS).rev() {
            let limb_ref = &mut self.limbs[i];
            let msb_of_current_limb = *limb_ref >> 63;
            *limb_ref <<= 1;
            *limb_ref |= msb_of_previous_limb;
            msb_of_previous_limb = msb_of_current_limb;
        }
        msb_of_previous_limb != 0
    }

    /// Multi-precision subtraction.
    /// Adapted from Algorithm 14.9 of "Handbook of Applied Cryptography" (https://cacr.uwaterloo.ca/hac/)
    /// Returns the results and a flag that is set if the substraction underflowed
    #[inline(always)]
    pub const fn sub(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
    ) -> (UnsignedInteger<NUM_LIMBS>, bool) {
        let mut limbs = [0u64; NUM_LIMBS];
        // 1.
        let mut carry = false;
        // 2.
        let mut i: usize = NUM_LIMBS;
        while i > 0 {
            i -= 1;
            let (x, cb) = a.limbs[i].overflowing_sub(b.limbs[i]);
            let (x, cc) = x.overflowing_sub(carry as u64);
            // Casting i128 to u64 drops the most significant bits of i128,
            // which effectively computes residue modulo 2^{64}
            // 2.1
            limbs[i] = x;
            // 2.2
            carry = cb | cc;
        }
        // 3.
        (Self { limbs }, carry)
    }

    /// Multi-precision multiplication.
    /// Adapted from Algorithm 14.12 of "Handbook of Applied Cryptography" (https://cacr.uwaterloo.ca/hac/)
    pub const fn mul(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
    ) -> (UnsignedInteger<NUM_LIMBS>, UnsignedInteger<NUM_LIMBS>) {
        // 1.
        let mut hi = [0u64; NUM_LIMBS];
        let mut lo = [0u64; NUM_LIMBS];
        // Const functions don't support for loops so we use whiles
        // this is equivalent to:
        // for i in (0..NUM_LIMBS).rev()
        // 2.
        let mut i = NUM_LIMBS;
        while i > 0 {
            i -= 1;
            // 2.1
            let mut carry = 0u128;
            let mut j = NUM_LIMBS;
            // 2.2
            while j > 0 {
                j -= 1;
                let mut k = i + j;
                if k >= NUM_LIMBS - 1 {
                    k -= NUM_LIMBS - 1;
                    let uv = (lo[k] as u128) + (a.limbs[j] as u128) * (b.limbs[i] as u128) + carry;
                    carry = uv >> 64;
                    // Casting u128 to u64 takes modulo 2^{64}
                    lo[k] = uv as u64;
                } else {
                    let uv =
                        (hi[k + 1] as u128) + (a.limbs[j] as u128) * (b.limbs[i] as u128) + carry;
                    carry = uv >> 64;
                    // Casting u128 to u64 takes modulo 2^{64}
                    hi[k + 1] = uv as u64;
                }
            }
            // 2.3
            hi[i] = carry as u64;
        }
        // 3.
        (Self { limbs: hi }, Self { limbs: lo })
    }

    #[inline(always)]
    pub fn square(
        a: &UnsignedInteger<NUM_LIMBS>,
    ) -> (UnsignedInteger<NUM_LIMBS>, UnsignedInteger<NUM_LIMBS>) {
        // NOTE: we use explicit `while` loops in this function because profiling pointed
        // at iterators of the form `(<x>..<y>).rev()` as the main performance bottleneck.

        let mut hi = Self {
            limbs: [0u64; NUM_LIMBS],
        };
        let mut lo = Self {
            limbs: [0u64; NUM_LIMBS],
        };

        // Compute products between a[i] and a[j] when i != j.
        // The variable `index` below is the index of `lo` or
        // `hi` to update
        let mut i = NUM_LIMBS;
        while i > 1 {
            i -= 1;
            let mut c: u128 = 0;
            let mut j = i;
            while j > 0 {
                j -= 1;
                let k = i + j;
                if k >= NUM_LIMBS - 1 {
                    let index = k + 1 - NUM_LIMBS;
                    let cs = lo.limbs[index] as u128 + a.limbs[i] as u128 * a.limbs[j] as u128 + c;
                    c = cs >> 64;
                    lo.limbs[index] = cs as u64;
                } else {
                    let index = k + 1;
                    let cs = hi.limbs[index] as u128 + a.limbs[i] as u128 * a.limbs[j] as u128 + c;
                    c = cs >> 64;
                    hi.limbs[index] = cs as u64;
                }
            }
            hi.limbs[i] = c as u64;
        }

        // All these terms should appear twice each,
        // so we have to multiply what we got so far by two.
        let carry = lo.limbs[0] >> 63;
        lo = lo << 1;
        hi = hi << 1;
        hi.limbs[NUM_LIMBS - 1] |= carry;

        // Add the only remaning terms, which are the squares a[i] * a[i].
        // The variable `index` below is the index of `lo` or
        // `hi` to update
        let mut c = 0;
        let mut i = NUM_LIMBS;
        while i > 0 {
            i -= 1;
            if NUM_LIMBS - 1 <= i * 2 {
                let index = 2 * i + 1 - NUM_LIMBS;
                let cs = lo.limbs[index] as u128 + a.limbs[i] as u128 * a.limbs[i] as u128 + c;
                c = cs >> 64;
                lo.limbs[index] = cs as u64;
            } else {
                let index = 2 * i + 1;
                let cs = hi.limbs[index] as u128 + a.limbs[i] as u128 * a.limbs[i] as u128 + c;
                c = cs >> 64;
                hi.limbs[index] = cs as u64;
            }
            if NUM_LIMBS - 1 < i * 2 {
                let index = 2 * i - NUM_LIMBS;
                let cs = lo.limbs[index] as u128 + c;
                c = cs >> 64;
                lo.limbs[index] = cs as u64;
            } else {
                let index = 2 * i;
                let cs = hi.limbs[index] as u128 + c;
                c = cs >> 64;
                hi.limbs[index] = cs as u64;
            }
        }
        debug_assert_eq!(c, 0);
        (hi, lo)
    }

    #[inline(always)]
    /// Returns the number of bits needed to represent the number (0 for zero).
    /// If nonzero, this is equivalent to one plus the floored log2 of the number.
    pub const fn bits(&self) -> u32 {
        let mut i = NUM_LIMBS;
        while i > 0 {
            if self.limbs[i - 1] != 0 {
                return i as u32 * u64::BITS - self.limbs[i - 1].leading_zeros();
            }
            i -= 1;
        }
        0
    }

    /// Returns the truthy value if `self != 0` and the falsy value otherwise.
    #[inline]
    const fn ct_is_nonzero(ct: u64) -> u64 {
        Self::ct_from_lsb((ct | ct.wrapping_neg()) >> (u64::BITS - 1))
    }

    /// Returns the truthy value if `value == 1`, and the falsy value if `value == 0`.
    /// Panics for other values.
    const fn ct_from_lsb(value: u64) -> u64 {
        debug_assert!(value == 0 || value == 1);
        value.wrapping_neg()
    }

    /// Return `b` if `c` is truthy, otherwise return `a`.
    #[inline]
    const fn ct_select_limb(a: u64, b: u64, ct: u64) -> u64 {
        a ^ (ct & (a ^ b))
    }

    /// Return `b` if `c` is truthy, otherwise return `a`.
    #[inline]
    const fn ct_select(a: &Self, b: &Self, c: u64) -> Self {
        let mut limbs = [0_u64; NUM_LIMBS];

        let mut i = 0;
        while i < NUM_LIMBS {
            limbs[i] = Self::ct_select_limb(a.limbs[i], b.limbs[i], c);
            i += 1;
        }

        Self { limbs }
    }

    /// Computes `self - (rhs + borrow)`, returning the result along with the new borrow.
    #[inline(always)]
    const fn sbb_limbs(lhs: u64, rhs: u64, borrow: u64) -> (u64, u64) {
        let a = lhs as u128;
        let b = rhs as u128;
        let borrow = (borrow >> (u64::BITS - 1)) as u128;
        let ret = a.wrapping_sub(b + borrow);
        (ret as u64, (ret >> u64::BITS) as u64)
    }

    #[inline(always)]
    /// Computes `a - (b + borrow)`, returning the result along with the new borrow.
    pub fn sbb(&self, rhs: &Self, mut borrow: u64) -> (Self, u64) {
        let mut limbs = [0; NUM_LIMBS];

        for i in (0..NUM_LIMBS).rev() {
            let (w, b) = Self::sbb_limbs(self.limbs[i], rhs.limbs[i], borrow);
            limbs[i] = w;
            borrow = b;
        }

        (Self { limbs }, borrow)
    }

    #[inline(always)]
    /// Returns the number of bits needed to represent the number as little endian
    pub const fn bits_le(&self) -> usize {
        let mut i = 0;
        while i < NUM_LIMBS {
            if self.limbs[i] != 0 {
                return u64::BITS as usize * (NUM_LIMBS - i)
                    - self.limbs[i].leading_zeros() as usize;
            }
            i += 1;
        }
        0
    }

    /// Computes self / rhs, returns the quotient, remainder.
    pub fn div_rem(&self, rhs: &Self) -> (Self, Self) {
        debug_assert!(
            *rhs != UnsignedInteger::from_u64(0),
            "Attempted to divide by zero"
        );
        let mb = rhs.bits_le();
        let mut bd = (NUM_LIMBS * u64::BITS as usize) - mb;
        let mut rem = *self;
        let mut quo = Self::from_u64(0);
        let mut c = rhs.shl(bd);

        loop {
            let (mut r, borrow) = rem.sbb(&c, 0);
            debug_assert!(borrow == 0 || borrow == u64::MAX);
            rem = Self::ct_select(&r, &rem, borrow);
            r = quo.bitor(Self::from_u64(1));
            quo = Self::ct_select(&r, &quo, borrow);
            if bd == 0 {
                break;
            }
            bd -= 1;
            c = c.shr(1);
            quo = quo.shl(1);
        }

        let is_some = Self::ct_is_nonzero(mb as u64);
        quo = Self::ct_select(&Self::from_u64(0), &quo, is_some);
        (quo, rem)
    }

    /// Convert from a decimal string.
    pub fn from_dec_str(value: &str) -> Result<Self, CreationError> {
        if value.is_empty() {
            return Err(CreationError::InvalidDecString);
        }
        let mut res = Self::from_u64(0);
        for b in value.bytes().map(|b| b.wrapping_sub(b'0')) {
            if b > 9 {
                return Err(CreationError::InvalidDecString);
            }
            let (high, low) = Self::mul(&res, &Self::from(10_u64));
            if high > Self::from_u64(0) {
                return Err(CreationError::InvalidDecString);
            }
            res = low + Self::from(b as u64);
        }
        Ok(res)
    }

    #[cfg(feature = "proptest")]
    pub fn nonzero_uint() -> impl Strategy<Value = UnsignedInteger<NUM_LIMBS>> {
        any_uint::<NUM_LIMBS>().prop_filter("is_zero", |&x| x != UnsignedInteger::from_u64(0))
    }
}

impl<const NUM_LIMBS: usize> IsUnsignedInteger for UnsignedInteger<NUM_LIMBS> {}

impl<const NUM_LIMBS: usize> ByteConversion for UnsignedInteger<NUM_LIMBS> {
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        self.limbs
            .iter()
            .flat_map(|limb| limb.to_be_bytes())
            .collect()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        self.limbs
            .iter()
            .rev()
            .flat_map(|limb| limb.to_le_bytes())
            .collect()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        // We cut off extra bytes, this is useful when you use this function to generate the element from randomness
        // In the future with the right algorithm this shouldn't be needed

        let needed_bytes = bytes
            .get(0..NUM_LIMBS * 8)
            .ok_or(ByteConversionError::FromBEBytesError)?;

        let mut limbs: [u64; NUM_LIMBS] = [0; NUM_LIMBS];

        needed_bytes
            .chunks_exact(8)
            .enumerate()
            .try_for_each(|(i, chunk)| {
                let limb = u64::from_be_bytes(
                    chunk
                        .try_into()
                        .map_err(|_| ByteConversionError::FromBEBytesError)?,
                );
                limbs[i] = limb;
                Ok::<_, ByteConversionError>(())
            })?;

        Ok(Self { limbs })
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError> {
        let needed_bytes = bytes
            .get(0..NUM_LIMBS * 8)
            .ok_or(ByteConversionError::FromBEBytesError)?;

        let mut limbs: [u64; NUM_LIMBS] = [0; NUM_LIMBS];

        needed_bytes
            .chunks_exact(8)
            .rev()
            .enumerate()
            .try_for_each(|(i, chunk)| {
                let limb = u64::from_le_bytes(
                    chunk
                        .try_into()
                        .map_err(|_| ByteConversionError::FromLEBytesError)?,
                );
                limbs[i] = limb;
                Ok::<_, ByteConversionError>(())
            })?;

        Ok(Self { limbs })
    }
}

impl<const NUM_LIMBS: usize> From<UnsignedInteger<NUM_LIMBS>> for u16 {
    fn from(value: UnsignedInteger<NUM_LIMBS>) -> Self {
        value.limbs[NUM_LIMBS - 1] as u16
    }
}

#[cfg(feature = "alloc")]
impl<const NUM_LIMBS: usize> AsBytes for UnsignedInteger<NUM_LIMBS> {
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.limbs
            .into_iter()
            .fold(alloc::vec::Vec::new(), |mut acc, limb| {
                acc.extend_from_slice(&limb.as_bytes());
                acc
            })
    }
}

#[cfg(feature = "alloc")]
impl<const NUM_LIMBS: usize> From<UnsignedInteger<NUM_LIMBS>> for alloc::vec::Vec<u8> {
    fn from(val: UnsignedInteger<NUM_LIMBS>) -> Self {
        val.as_bytes()
    }
}

#[cfg(feature = "proptest")]
fn any_uint<const NUM_LIMBS: usize>() -> impl Strategy<Value = UnsignedInteger<NUM_LIMBS>> {
    any::<[u64; NUM_LIMBS]>().prop_map(UnsignedInteger::from_limbs)
}

#[cfg(feature = "proptest")]
impl<const NUM_LIMBS: usize> Arbitrary for UnsignedInteger<NUM_LIMBS> {
    type Parameters = ();

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        any_uint::<NUM_LIMBS>().sboxed()
    }

    type Strategy = SBoxedStrategy<Self>;
}
