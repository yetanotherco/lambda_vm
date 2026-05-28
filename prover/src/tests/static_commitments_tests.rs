//! Drift-detection tests for hardcoded preprocessed-table commitments.
//!
//! Each test recomputes the commitment for a given `blowup_factor` and
//! asserts it matches the value pinned in `HARDCODED_PREPROCESSED_COMMITMENTS`.
//! If anyone changes the AIR, FFT pipeline, or `coset_offset`, these tests
//! fail loudly — regenerate via
//! `cargo run --bin compute_static_commitments --release`.

use stark::proof::options::GoldilocksCubicProofOptions;

use crate::tables::{bitwise, keccak_rc};

fn find_hardcoded(table: &[(u8, [u8; 32])], blowup: u8) -> [u8; 32] {
    table
        .iter()
        .find(|(b, _)| *b == blowup)
        .map(|(_, commitment)| *commitment)
        .unwrap_or_else(|| panic!("no hardcoded commitment for blowup_factor={blowup}"))
}

fn assert_bitwise_matches(blowup: u8) {
    let options = GoldilocksCubicProofOptions::with_blowup(blowup)
        .expect("blowup must be a valid power of 2");
    let computed = bitwise::compute_preprocessed_commitment(&options);
    let hardcoded = find_hardcoded(bitwise::HARDCODED_PREPROCESSED_COMMITMENTS, blowup);
    assert_eq!(
        computed, hardcoded,
        "bitwise commitment drifted for blowup={blowup}; regenerate constants via \
         `cargo run --bin compute_static_commitments --release`",
    );
}

fn assert_keccak_rc_matches(blowup: u8) {
    let options = GoldilocksCubicProofOptions::with_blowup(blowup)
        .expect("blowup must be a valid power of 2");
    let computed = keccak_rc::compute_preprocessed_commitment(&options);
    let hardcoded = find_hardcoded(keccak_rc::HARDCODED_PREPROCESSED_COMMITMENTS, blowup);
    assert_eq!(
        computed, hardcoded,
        "keccak_rc commitment drifted for blowup={blowup}; regenerate constants via \
         `cargo run --bin compute_static_commitments --release`",
    );
}

#[test]
fn bitwise_hardcoded_matches_recompute_blowup_2() {
    assert_bitwise_matches(2);
}

#[test]
fn bitwise_hardcoded_matches_recompute_blowup_4() {
    assert_bitwise_matches(4);
}

#[test]
fn bitwise_hardcoded_matches_recompute_blowup_8() {
    assert_bitwise_matches(8);
}

#[test]
fn keccak_rc_hardcoded_matches_recompute_blowup_2() {
    assert_keccak_rc_matches(2);
}

#[test]
fn keccak_rc_hardcoded_matches_recompute_blowup_4() {
    assert_keccak_rc_matches(4);
}

#[test]
fn keccak_rc_hardcoded_matches_recompute_blowup_8() {
    assert_keccak_rc_matches(8);
}
