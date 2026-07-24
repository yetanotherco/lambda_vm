use core::{
    fmt::{Display, LowerHex, UpperHex},
    ops::{Add, BitAnd, Shr, ShrAssign},
};

pub trait IsUnsignedInteger:
    Shr<usize, Output = Self>
    + ShrAssign<usize>
    + BitAnd<Output = Self>
    + Eq
    + Ord
    + From<u16>
    + Copy
    + Display
    + LowerHex
    + UpperHex
    + Add<Self, Output = Self>
{
    /// The low 64 bits of `self` (i.e. `self as u64`): a widening for the
    /// narrower types and a plain truncation for `u128`. Lets generic code hand
    /// an exponent to a `u64` ABI without the trait requiring `Into<u64>`.
    fn low_u64(self) -> u64;
}

impl IsUnsignedInteger for u128 {
    fn low_u64(self) -> u64 {
        self as u64
    }
}
impl IsUnsignedInteger for u64 {
    fn low_u64(self) -> u64 {
        self
    }
}
impl IsUnsignedInteger for u32 {
    fn low_u64(self) -> u64 {
        self as u64
    }
}
impl IsUnsignedInteger for u16 {
    fn low_u64(self) -> u64 {
        self as u64
    }
}
impl IsUnsignedInteger for usize {
    fn low_u64(self) -> u64 {
        self as u64
    }
}

/// The low 64 bits of an [`IsUnsignedInteger`] exponent, for the sim/27
/// `SIM_POW` overrides which must hand the exponent to the host ecall as a
/// `u64` (the recursion verifier's `pow` exponents — trace length, part counts,
/// coset indices — all fit in 64 bits). A direct [`IsUnsignedInteger::low_u64`]
/// cast; byte-identical to the previous O(bits) bit-by-bit reconstruction, which
/// cost ~30x the actual ecall it was marshaling for. Guest-only to avoid a
/// dead-code warning on the host.
#[cfg(all(target_arch = "riscv64", feature = "sim-pow"))]
pub(crate) fn exp_to_u64<T: IsUnsignedInteger>(exponent: T) -> u64 {
    exponent.low_u64()
}
