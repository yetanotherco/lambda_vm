//! Drift-detection tests for hardcoded preprocessed-table commitments.
//!
//! Each test recomputes the commitment for a given `blowup_factor` and
//! asserts it matches the value pinned in `HARDCODED_PREPROCESSED_COMMITMENTS`.
//! If anyone changes the AIR, FFT pipeline, or `coset_offset`, these tests
//! fail loudly — regenerate via
//! `cargo run --bin compute_static_commitments --release`.

use stark::proof::options::GoldilocksCubicProofOptions;

use crate::tables::{bitwise, keccak_rc};

fn lookup(table: &[(u8, [u8; 32])], blowup: u8) -> [u8; 32] {
    table
        .iter()
        .find(|(b, _)| *b == blowup)
        .map(|(_, commitment)| *commitment)
        .unwrap_or_else(|| panic!("no hardcoded commitment for blowup_factor={blowup}"))
}

#[test]
fn bitwise_hardcoded_matches_recompute_blowup_2() {
    let options = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid");
    let computed = bitwise::compute_preprocessed_commitment(&options);
    let hardcoded = lookup(
        bitwise::HARDCODED_PREPROCESSED_COMMITMENTS,
        options.blowup_factor,
    );
    assert_eq!(
        computed, hardcoded,
        "bitwise commitment drifted for blowup=2; regenerate constants via \
         `cargo run --bin compute_static_commitments --release`",
    );
}

#[test]
fn keccak_rc_hardcoded_matches_recompute_blowup_2() {
    let options = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid");
    let computed = keccak_rc::compute_preprocessed_commitment(&options);
    let hardcoded = lookup(
        keccak_rc::HARDCODED_PREPROCESSED_COMMITMENTS,
        options.blowup_factor,
    );
    assert_eq!(
        computed, hardcoded,
        "keccak_rc commitment drifted for blowup=2; regenerate constants via \
         `cargo run --bin compute_static_commitments --release`",
    );
}
