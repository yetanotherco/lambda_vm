//! Little-endian limb decomposition of 64-bit values for trace columns.
//!
//! VM tables repeatedly split a `u64` operand into 16-bit or 32-bit limbs — to
//! range-check each piece and feed it to the bus. These helpers centralize the
//! shift/mask arithmetic (and the field-element column writes) so the exact same
//! decomposition isn't open-coded in every table.
//!
//! Limb order is always little-endian: limb 0 is the least-significant chunk.

use super::types::FE;

/// Mask selecting one 16-bit limb.
const LIMB16_MASK: u64 = 0xFFFF;
/// Mask selecting one 32-bit limb.
const LIMB32_MASK: u64 = 0xFFFF_FFFF;

/// The four little-endian 16-bit limbs of `x`:
/// `[x[0..16], x[16..32], x[32..48], x[48..64]]`.
#[inline]
pub const fn limbs_16(x: u64) -> [u64; 4] {
    [
        x & LIMB16_MASK,
        (x >> 16) & LIMB16_MASK,
        (x >> 32) & LIMB16_MASK,
        (x >> 48) & LIMB16_MASK,
    ]
}

/// The two little-endian 32-bit limbs of `x`: `[lo, hi]`.
#[inline]
pub const fn limbs_32(x: u64) -> [u64; 2] {
    [x & LIMB32_MASK, x >> 32]
}

/// The `i`-th little-endian 16-bit limb of `x` (`i` in `0..4`).
#[inline]
pub const fn limb_16(x: u64, i: u32) -> u64 {
    (x >> (16 * i)) & LIMB16_MASK
}

/// Write the four 16-bit limbs of `x` as field elements into the four
/// consecutive columns `data[col..col + 4]` (little-endian limb order).
#[inline]
pub fn set_limbs_16(data: &mut [FE], col: usize, x: u64) {
    let limbs = limbs_16(x);
    data[col] = FE::from(limbs[0]);
    data[col + 1] = FE::from(limbs[1]);
    data[col + 2] = FE::from(limbs[2]);
    data[col + 3] = FE::from(limbs[3]);
}

/// Write the two 32-bit limbs of `x` as field elements into the two consecutive
/// columns `data[col..col + 2]` (`[lo, hi]`).
#[inline]
pub fn set_limbs_32(data: &mut [FE], col: usize, x: u64) {
    let limbs = limbs_32(x);
    data[col] = FE::from(limbs[0]);
    data[col + 1] = FE::from(limbs[1]);
}
