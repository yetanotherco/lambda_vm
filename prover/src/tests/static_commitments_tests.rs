//! Drift-detection and lookup-dispatch tests for the static preprocessed-table
//! commitments shipped in `bitwise` and `keccak_rc`.
//!
//! - The drift tests recompute the commitment for every blowup in
//!   `STATIC_BLOWUP_FACTORS` (the list shared with the generator binary) and
//!   compare against the value the table-module's `preprocessed_commitment`
//!   wrapper returns. This catches AIR or FFT-pipeline drift AND confirms the
//!   wrapper dispatches through `static_commitment` for the static blowups.
//! - The non-static-blowup fallback test picks a blowup not in
//!   `STATIC_BLOWUP_FACTORS` and asserts the wrapper still returns the correct
//!   value via recompute.
//! - The coset-mismatch tests use `coset_offset != 3` and assert the wrapper
//!   takes the recompute path (rather than silently returning the coset-3
//!   static bytes); they're the regression test for the
//!   `options.coset_offset == 3` gate in `preprocessed_commitment`.
//!
//! If a drift test fails, regenerate the constants via
//! `cargo run --bin compute_static_commitments --release`.

use stark::proof::options::GoldilocksCubicProofOptions;

use crate::tables::{STATIC_BLOWUP_FACTORS, bitwise, keccak_rc, page};

fn options_for(blowup: u8) -> stark::proof::options::ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(blowup).expect("blowup must be a valid power of 2")
}

fn options_with_coset(blowup: u8, coset_offset: u64) -> stark::proof::options::ProofOptions {
    let mut options = options_for(blowup);
    options.coset_offset = coset_offset;
    options
}

/// A blowup that is *not* in `STATIC_BLOWUP_FACTORS` — drives the fallback
/// (recompute) path through `preprocessed_commitment`.
const NON_STATIC_BLOWUP: u8 = 16;

/// The coset offset every in-tree `ProofOptions` constructor pins, and the
/// one the static commitment bytes were generated for.
const STANDARD_COSET: u64 = 3;

/// A coset offset different from the one used to generate the static
/// commitments. Picked to match `test_multi_prove_mixed_coset_offsets` in
/// the stark crate so we exercise a configuration the rest of the system
/// supports.
const NON_STANDARD_COSET: u64 = 7;

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

/// Drift / dispatch test for the zero-init PAGE static commitments. For every
/// blowup in `STATIC_BLOWUP_FACTORS`, builds a synthetic zero-init page at
/// `DEFAULT_PAGE_SIZE` (page_base = 0 — the value doesn't affect the
/// commitment since OFFSET is page-relative and INIT is uniformly zero) and
/// asserts that `preprocessed_commitment` returns the same value as a
/// direct `compute_precomputed_commitment`. Catches AIR / FFT-pipeline drift
/// AND confirms the wrapper dispatches through `static_zero_page_commitment`
/// for the static blowups.
#[test]
fn zero_page_static_matches_recompute_for_all_blowups() {
    let zero_page_config = page::PageConfig::zero_init(0);
    for &blowup in STATIC_BLOWUP_FACTORS {
        let options = options_for(blowup);
        let from_wrapper = page::preprocessed_commitment(&zero_page_config, &options);
        let recomputed = page::compute_precomputed_commitment(&zero_page_config, &options);
        assert_eq!(
            from_wrapper, recomputed,
            "zero-init page commitment drifted (or wrapper dispatch broke) for \
             blowup={blowup}; regenerate constants via \
             `cargo run --bin compute_static_commitments --release`",
        );
    }
}

/// Asserts the page wrapper's fallback path (no static entry for this
/// blowup) recomputes a commitment that matches the direct compute call.
/// Ignored by default: at NON_STATIC_BLOWUP=16, the page LDE is 2^22 rows ×
/// 2 cols plus the FFT/Merkle build, which takes minutes per run. Run
/// explicitly when validating the wrapper's fallback for page.
#[test]
#[ignore = "heavy: page LDE at NON_STATIC_BLOWUP=16 is 2^22 rows × 2 cols; minutes per run"]
fn page_non_static_blowup_recomputes_via_fallback() {
    assert!(
        !STATIC_BLOWUP_FACTORS.contains(&NON_STATIC_BLOWUP),
        "test relies on NON_STATIC_BLOWUP not being in STATIC_BLOWUP_FACTORS",
    );
    let zero_page_config = page::PageConfig::zero_init(0);
    let options = options_for(NON_STATIC_BLOWUP);
    let from_wrapper = page::preprocessed_commitment(&zero_page_config, &options);
    let recomputed = page::compute_precomputed_commitment(&zero_page_config, &options);
    assert_eq!(
        from_wrapper, recomputed,
        "page fallback returned a value that doesn't match direct compute at blowup={NON_STATIC_BLOWUP}",
    );
}

/// Regression test for the `options.coset_offset == 3` gate in
/// `page::preprocessed_commitment`. With a non-3 coset offset, the wrapper
/// must NOT return a static value — it must recompute (matching direct
/// compute) and must NOT equal the coset-3 static commitment. Ignored by
/// default: each blowup at DEFAULT_PAGE_SIZE builds a 2^19-row × 2-col page
/// LDE, multiple seconds per blowup. Run explicitly when validating the
/// coset-3 gate for page.
#[test]
#[ignore = "heavy: 2^19-row page LDE per blowup; tens of seconds total"]
fn page_non_three_coset_recomputes_and_differs_from_static() {
    let zero_page_config = page::PageConfig::zero_init(0);
    for &blowup in STATIC_BLOWUP_FACTORS {
        let opts_coset3 = options_with_coset(blowup, STANDARD_COSET);
        let opts_coset7 = options_with_coset(blowup, NON_STANDARD_COSET);

        let from_wrapper_7 = page::preprocessed_commitment(&zero_page_config, &opts_coset7);
        let recomputed_7 = page::compute_precomputed_commitment(&zero_page_config, &opts_coset7);
        let from_wrapper_3 = page::preprocessed_commitment(&zero_page_config, &opts_coset3);

        assert_eq!(
            from_wrapper_7, recomputed_7,
            "page wrapper at coset {NON_STANDARD_COSET} must take the recompute path \
             (blowup={blowup})",
        );
        assert_ne!(
            from_wrapper_7, from_wrapper_3,
            "page commitment at coset {NON_STANDARD_COSET} must differ from coset \
             {STANDARD_COSET} static value (blowup={blowup})",
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

/// Regression test for the `options.coset_offset == 3` gate in
/// `keccak_rc::preprocessed_commitment`. With a non-3 coset offset, the
/// wrapper must NOT return a static value — it must recompute (matching
/// direct compute) and must NOT equal the coset-3 static commitment.
/// Cheap because keccak_rc is only 32 rows.
#[test]
fn keccak_rc_non_three_coset_recomputes_and_differs_from_static() {
    for &blowup in STATIC_BLOWUP_FACTORS {
        let opts_coset3 = options_with_coset(blowup, STANDARD_COSET);
        let opts_coset7 = options_with_coset(blowup, NON_STANDARD_COSET);

        let from_wrapper_7 = keccak_rc::preprocessed_commitment(&opts_coset7);
        let recomputed_7 = keccak_rc::compute_preprocessed_commitment(&opts_coset7);
        let from_wrapper_3 = keccak_rc::preprocessed_commitment(&opts_coset3);

        assert_eq!(
            from_wrapper_7, recomputed_7,
            "keccak_rc wrapper at coset {NON_STANDARD_COSET} must take the recompute path \
             (blowup={blowup})",
        );
        assert_ne!(
            from_wrapper_7, from_wrapper_3,
            "keccak_rc commitment at coset {NON_STANDARD_COSET} must differ from coset \
             {STANDARD_COSET} static value (blowup={blowup})",
        );
    }
}

/// Bitwise counterpart of
/// `keccak_rc_non_three_coset_recomputes_and_differs_from_static`. Ignored
/// by default: a 2^20-row × 11-column bitwise LDE at the static blowups
/// takes tens of seconds per blowup. Run explicitly when validating the
/// coset-3 gate for bitwise.
#[test]
#[ignore = "heavy: 2^20-row bitwise LDE per blowup; tens of seconds total"]
fn bitwise_non_three_coset_recomputes_and_differs_from_static() {
    for &blowup in STATIC_BLOWUP_FACTORS {
        let opts_coset3 = options_with_coset(blowup, STANDARD_COSET);
        let opts_coset7 = options_with_coset(blowup, NON_STANDARD_COSET);

        let from_wrapper_7 = bitwise::preprocessed_commitment(&opts_coset7);
        let recomputed_7 = bitwise::compute_preprocessed_commitment(&opts_coset7);
        let from_wrapper_3 = bitwise::preprocessed_commitment(&opts_coset3);

        assert_eq!(
            from_wrapper_7, recomputed_7,
            "bitwise wrapper at coset {NON_STANDARD_COSET} must take the recompute path \
             (blowup={blowup})",
        );
        assert_ne!(
            from_wrapper_7, from_wrapper_3,
            "bitwise commitment at coset {NON_STANDARD_COSET} must differ from coset \
             {STANDARD_COSET} static value (blowup={blowup})",
        );
    }
}
