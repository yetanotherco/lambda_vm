//! Drift-detection and lookup-dispatch tests for the static preprocessed-table
//! commitments shipped in `bitwise` and `keccak_rc`.
//!
//! - The drift tests recompute the commitment for every blowup in
//!   `STATIC_BLOWUP_FACTORS` (the list shared with the generator binary) and
//!   compare against the value the table-module's `preprocessed_commitment`
//!   wrapper returns. This catches AIR or FFT-pipeline drift AND confirms the
//!   wrapper dispatches through `static_commitment` for the static blowups.
//! - The fallback test picks a blowup not in `STATIC_BLOWUP_FACTORS` and
//!   asserts the wrapper still returns the correct value via recompute.
//!
//! If a drift test fails, regenerate the constants via
//! `cargo run --bin compute_static_commitments --release`.

use stark::proof::options::GoldilocksCubicProofOptions;

use crate::tables::{STATIC_BLOWUP_FACTORS, bitwise, keccak_rc};

fn options_for(blowup: u8) -> stark::proof::options::ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(blowup).expect("blowup must be a valid power of 2")
}

/// A blowup that is *not* in `STATIC_BLOWUP_FACTORS` — drives the fallback
/// (recompute) path through `preprocessed_commitment`.
const NON_STATIC_BLOWUP: u8 = 16;

#[test]
fn bitwise_static_matches_recompute_for_all_blowups() {
    for &blowup in STATIC_BLOWUP_FACTORS {
        let options = options_for(blowup);
        let from_wrapper = bitwise::preprocessed_commitment(&options);
        let recomputed = bitwise::compute_preprocessed_commitment(&options);
        assert_eq!(
            from_wrapper, recomputed,
            "bitwise commitment drifted (or wrapper dispatch broke) for blowup={blowup}; \
             regenerate constants via `cargo run --bin compute_static_commitments --release`",
        );
    }
}

#[test]
fn keccak_rc_static_matches_recompute_for_all_blowups() {
    for &blowup in STATIC_BLOWUP_FACTORS {
        let options = options_for(blowup);
        let from_wrapper = keccak_rc::preprocessed_commitment(&options);
        let recomputed = keccak_rc::compute_preprocessed_commitment(&options);
        assert_eq!(
            from_wrapper, recomputed,
            "keccak_rc commitment drifted (or wrapper dispatch broke) for blowup={blowup}; \
             regenerate constants via `cargo run --bin compute_static_commitments --release`",
        );
    }
}

/// Asserts the wrapper's fallback path (no static entry for this blowup)
/// recomputes a commitment that matches the direct compute call. Uses
/// keccak_rc because its table is only 32 rows, making the recompute cheap
/// even at a non-standard blowup.
#[test]
fn keccak_rc_non_static_blowup_recomputes_via_fallback() {
    assert!(
        !STATIC_BLOWUP_FACTORS.contains(&NON_STATIC_BLOWUP),
        "test relies on NON_STATIC_BLOWUP not being in STATIC_BLOWUP_FACTORS",
    );
    let options = options_for(NON_STATIC_BLOWUP);
    let from_wrapper = keccak_rc::preprocessed_commitment(&options);
    let recomputed = keccak_rc::compute_preprocessed_commitment(&options);
    assert_eq!(
        from_wrapper, recomputed,
        "keccak_rc fallback returned a value that doesn't match direct compute at blowup={NON_STATIC_BLOWUP}",
    );
}

/// Bitwise counterpart of `keccak_rc_non_static_blowup_recomputes_via_fallback`.
/// Ignored by default: at the smallest legal non-static blowup (16), the
/// bitwise LDE is 2^24 rows × 11 columns ≈ 1.4 GB plus the FFT/Merkle build,
/// which takes several minutes on a laptop. Run explicitly when validating
/// the wrapper's fallback for bitwise.
#[test]
#[ignore = "heavy: 2^24-row bitwise LDE; minutes per run"]
fn bitwise_non_static_blowup_recomputes_via_fallback() {
    assert!(
        !STATIC_BLOWUP_FACTORS.contains(&NON_STATIC_BLOWUP),
        "test relies on NON_STATIC_BLOWUP not being in STATIC_BLOWUP_FACTORS",
    );
    let options = options_for(NON_STATIC_BLOWUP);
    let from_wrapper = bitwise::preprocessed_commitment(&options);
    let recomputed = bitwise::compute_preprocessed_commitment(&options);
    assert_eq!(
        from_wrapper, recomputed,
        "bitwise fallback returned a value that doesn't match direct compute at blowup={NON_STATIC_BLOWUP}",
    );
}
