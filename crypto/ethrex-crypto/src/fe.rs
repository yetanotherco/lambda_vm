//! secp256k1 base-field arithmetic on four 64-bit limbs, for the ECSM-accelerated
//! recovery.
//!
//! k256's `FieldElement` keeps five 52-bit limbs so that sums need no carries; on a
//! 64-bit RISC-V guest its product is 317 instructions and its square 224. Here an
//! element is four little-endian `u64` limbs holding any value below 2^256 that is
//! congruent to it modulo p = 2^256 − C, C = 2^32 + 977 (a "weak" form: values in
//! [p, 2^256) are allowed and mean value − p). Products fold the high half back with
//! C, sums and differences fold their carry or borrow the same way, and only
//! comparisons and encodings bring a value to its canonical form below p. The limbs
//! are also the precompile's little-endian encoding, so its inputs and outputs need
//! no byte reversal.
//!
//! Every operation is checked against k256 in `tests/fe_tests.rs`.

// Some conversions are only reached from the guest path and the tests.
#![cfg_attr(not(any(target_arch = "riscv64", test)), allow(dead_code))]

use core::ops::{Add, Mul, Neg, Sub};

use k256::FieldElement;

/// 2^256 − p.
const C: u64 = 0x1_0000_03D1;

/// An element of the secp256k1 base field; see the module documentation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Fe([u64; 4]);

#[inline(always)]
fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = a as u128 + b as u128 + carry as u128;
    (t as u64, (t >> 64) as u64)
}

#[inline(always)]
fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let t = (a as u128).wrapping_sub(b as u128 + borrow as u128);
    (t as u64, (t >> 127) as u64)
}

impl Fe {
    pub(crate) const ZERO: Fe = Fe([0; 4]);
    pub(crate) const ONE: Fe = Fe([1, 0, 0, 0]);
    pub(crate) const SEVEN: Fe = Fe([7, 0, 0, 0]);

    /// The value with these little-endian limbs (any value below 2^256).
    pub(crate) const fn from_limbs(limbs: [u64; 4]) -> Fe {
        Fe(limbs)
    }

    /// The canonical value's limbs, below p.
    #[inline]
    fn canonical(self) -> [u64; 4] {
        // self >= p exactly when self + C carries out of 2^256; the wrapped sum is then
        // self − p.
        let (r0, c) = adc(self.0[0], C, 0);
        let (r1, c) = adc(self.0[1], 0, c);
        let (r2, c) = adc(self.0[2], 0, c);
        let (r3, c) = adc(self.0[3], 0, c);
        if c != 0 {
            [r0, r1, r2, r3]
        } else {
            self.0
        }
    }

    /// The element of 32 little-endian bytes, or `None` if they encode p or more
    /// (the same rule as `FieldElement::from_bytes`).
    pub(crate) fn from_le_bytes(bytes: &[u8; 32]) -> Option<Fe> {
        let mut limbs = [0u64; 4];
        for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(8)) {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            *limb = u64::from_le_bytes(word);
        }
        let fe = Fe(limbs);
        (fe.canonical() == limbs).then_some(fe)
    }

    /// The element of 32 big-endian bytes, or `None` if they encode p or more.
    pub(crate) fn from_be_bytes(bytes: &[u8; 32]) -> Option<Fe> {
        let mut limbs = [0u64; 4];
        for (limb, chunk) in limbs.iter_mut().zip(bytes.rchunks_exact(8)) {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            *limb = u64::from_be_bytes(word);
        }
        let fe = Fe(limbs);
        (fe.canonical() == limbs).then_some(fe)
    }

    /// The canonical value as 32 little-endian bytes.
    pub(crate) fn to_le_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (chunk, limb) in out.chunks_exact_mut(8).zip(self.canonical()) {
            chunk.copy_from_slice(&limb.to_le_bytes());
        }
        out
    }

    /// The canonical value as 32 big-endian bytes.
    pub(crate) fn to_be_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (chunk, limb) in out.rchunks_exact_mut(8).zip(self.canonical()) {
            chunk.copy_from_slice(&limb.to_be_bytes());
        }
        out
    }

    /// The same element as k256's type.
    pub(crate) fn from_k256(x: &FieldElement) -> Fe {
        // `to_bytes` is canonical, so every limb pattern it gives is below p.
        let bytes: [u8; 32] = x.to_bytes().into();
        let mut limbs = [0u64; 4];
        for (limb, chunk) in limbs.iter_mut().zip(bytes.rchunks_exact(8)) {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            *limb = u64::from_be_bytes(word);
        }
        Fe(limbs)
    }

    /// The same element as k256's type (always `Some`: the encoding is canonical).
    pub(crate) fn to_k256(self) -> Option<FieldElement> {
        Option::from(FieldElement::from_bytes(&self.to_be_bytes().into()))
    }

    /// Whether the element is zero.
    pub(crate) fn is_zero(self) -> bool {
        self.canonical() == [0; 4]
    }

    /// Whether the two values are the same element.
    pub(crate) fn same(self, other: Fe) -> bool {
        self.canonical() == other.canonical()
    }

    /// Whether the canonical value is odd.
    pub(crate) fn is_odd(self) -> bool {
        self.canonical()[0] & 1 == 1
    }

    pub(crate) fn double(self) -> Fe {
        self + self
    }

    /// The square, with the cross products computed once and doubled.
    pub(crate) fn square(self) -> Fe {
        let a = self.0;
        let mut r = [0u64; 8];
        for i in 0..4 {
            let mut carry: u128 = 0;
            for j in (i + 1)..4 {
                let t = r[i + j] as u128 + (a[i] as u128) * (a[j] as u128) + carry;
                r[i + j] = t as u64;
                carry = t >> 64;
            }
            r[i + 4] = carry as u64;
        }
        let mut top = 0u64;
        for limb in r.iter_mut().skip(1) {
            let next = *limb >> 63;
            *limb = (*limb << 1) | top;
            top = next;
        }
        let mut carry: u128 = 0;
        for i in 0..4 {
            let d = (a[i] as u128) * (a[i] as u128);
            let t = r[2 * i] as u128 + (d as u64) as u128 + carry;
            r[2 * i] = t as u64;
            let t2 = r[2 * i + 1] as u128 + (d >> 64) + (t >> 64);
            r[2 * i + 1] = t2 as u64;
            carry = t2 >> 64;
        }
        reduce(r)
    }
}

/// The 512-bit `r` modulo p, in the weak form: the high half folded in with C twice.
#[inline(always)]
fn reduce(r: [u64; 8]) -> Fe {
    let mut out = [0u64; 4];
    let mut carry: u128 = 0;
    for i in 0..4 {
        let t = r[i] as u128 + (r[i + 4] as u128) * (C as u128) + carry;
        out[i] = t as u64;
        carry = t >> 64;
    }
    // carry < 2^34, carry · C < 2^67.
    let t = out[0] as u128 + carry * (C as u128);
    out[0] = t as u64;
    let mut c = (t >> 64) as u64;
    for limb in out.iter_mut().skip(1) {
        if c == 0 {
            return Fe(out);
        }
        let (v, next) = adc(*limb, c, 0);
        *limb = v;
        c = next;
    }
    if c != 0 {
        // Past 2^256 once more: the value is now below 2^67, so adding C cannot carry.
        out[0] += C;
    }
    Fe(out)
}

impl Add for Fe {
    type Output = Fe;

    fn add(self, other: Fe) -> Fe {
        let (r0, c) = adc(self.0[0], other.0[0], 0);
        let (r1, c) = adc(self.0[1], other.0[1], c);
        let (r2, c) = adc(self.0[2], other.0[2], c);
        let (r3, c) = adc(self.0[3], other.0[3], c);
        if c == 0 {
            return Fe([r0, r1, r2, r3]);
        }
        // The sum wrapped 2^256 ≡ C: add C back.
        let (r0, c) = adc(r0, C, 0);
        let (r1, c) = adc(r1, 0, c);
        let (r2, c) = adc(r2, 0, c);
        let (r3, c) = adc(r3, 0, c);
        if c == 0 {
            return Fe([r0, r1, r2, r3]);
        }
        // Wrapped again: the value is now below C, so adding C cannot carry.
        Fe([r0 + C, r1, r2, r3])
    }
}

impl Sub for Fe {
    type Output = Fe;

    fn sub(self, other: Fe) -> Fe {
        let (r0, b) = sbb(self.0[0], other.0[0], 0);
        let (r1, b) = sbb(self.0[1], other.0[1], b);
        let (r2, b) = sbb(self.0[2], other.0[2], b);
        let (r3, b) = sbb(self.0[3], other.0[3], b);
        if b == 0 {
            return Fe([r0, r1, r2, r3]);
        }
        // The difference wrapped by +2^256 ≡ +C: take C off.
        let (r0, b) = sbb(r0, C, 0);
        let (r1, b) = sbb(r1, 0, b);
        let (r2, b) = sbb(r2, 0, b);
        let (r3, b) = sbb(r3, 0, b);
        if b == 0 {
            return Fe([r0, r1, r2, r3]);
        }
        // Wrapped again (the value was below C): now at least 2^256 − C, so taking C
        // off once more cannot borrow.
        Fe([r0 - C, r1, r2, r3])
    }
}

impl Neg for Fe {
    type Output = Fe;

    fn neg(self) -> Fe {
        Fe::ZERO - self
    }
}

impl Mul for Fe {
    type Output = Fe;

    fn mul(self, other: Fe) -> Fe {
        let (a, b) = (self.0, other.0);
        let mut r = [0u64; 8];
        for i in 0..4 {
            let mut carry: u128 = 0;
            for j in 0..4 {
                let t = r[i + j] as u128 + (a[i] as u128) * (b[j] as u128) + carry;
                r[i + j] = t as u64;
                carry = t >> 64;
            }
            r[i + 4] = carry as u64;
        }
        reduce(r)
    }
}

#[cfg(test)]
impl Fe {
    /// The raw limbs, for the tests.
    pub(crate) fn limbs(self) -> [u64; 4] {
        self.0
    }
}
