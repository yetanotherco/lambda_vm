//! Tests for the little-endian limb-decomposition helpers.

use crate::tables::limbs::*;
use crate::tables::types::FE;

const X: u64 = 0xDEAD_BEEF_1234_5678;
const SHIFT_BOUNDARY: u64 = 0xFFFF_0000_FFFF_0000;

#[test]
fn limbs_16_is_little_endian_and_reconstructs() {
    assert_eq!(limbs_16(X), [0x5678, 0x1234, 0xBEEF, 0xDEAD]);
    let [a, b, c, d] = limbs_16(X);
    assert_eq!(a | (b << 16) | (c << 32) | (d << 48), X);
}

#[test]
fn limbs_32_is_little_endian_and_reconstructs() {
    assert_eq!(limbs_32(X), [0x1234_5678, 0xDEAD_BEEF]);
    let [lo, hi] = limbs_32(X);
    assert_eq!(lo | (hi << 32), X);
}

#[test]
fn limb_16_matches_limbs_16() {
    for i in 0..4 {
        assert_eq!(limb_16(X, i), limbs_16(X)[i as usize]);
    }
}

#[test]
fn limbs_match_open_coded_form() {
    // Guards against drift from the idioms these helpers replace.
    for x in [0u64, 1, u64::MAX, X, 0xFFFF, 0x1_0000, SHIFT_BOUNDARY] {
        assert_eq!(
            limbs_16(x),
            [
                x & 0xFFFF,
                (x >> 16) & 0xFFFF,
                (x >> 32) & 0xFFFF,
                (x >> 48) & 0xFFFF
            ]
        );
        assert_eq!(limbs_32(x), [x & 0xFFFF_FFFF, x >> 32]);
    }
}

#[test]
fn set_limbs_write_consecutive_columns() {
    let mut data = vec![FE::from(0u64); 6];
    set_limbs_16(&mut data, 0, X);
    set_limbs_32(&mut data, 4, X);
    assert_eq!(
        data,
        vec![
            FE::from(0x5678u64),
            FE::from(0x1234u64),
            FE::from(0xBEEFu64),
            FE::from(0xDEADu64),
            FE::from(0x1234_5678u64),
            FE::from(0xDEAD_BEEFu64),
        ]
    );
}
