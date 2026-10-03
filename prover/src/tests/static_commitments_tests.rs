//! Drift-detection and lookup-dispatch tests for the static preprocessed-table
//! commitments shipped in `bitwise`, `keccak_rc`, and `page` (the shared
//! zero-init page commitment).
//!
//! - The drift tests recompute the commitment for every blowup in
//!   `STATIC_BLOWUP_FACTORS` (the list shared with the generator binary) and
//!   compare against the value the table-module's wrapper returns
//!   (`preprocessed_commitment` for `bitwise`/`keccak_rc`,
//!   `zero_init_preprocessed_commitment` for `page`). This catches AIR or
//!   FFT-pipeline drift; the page test additionally pins the static bytes
//!   against the recompute directly.
//! - The non-static-blowup fallback tests pick a blowup not in
//!   `STATIC_BLOWUP_FACTORS` and assert the wrapper still returns the correct
//!   value via recompute.
//! - The coset-mismatch tests use `coset_offset != 3` and assert the wrapper
//!   takes the recompute path (rather than silently returning the coset-3
//!   static bytes); they're the regression test for the
//!   `options.coset_offset == 3` gate in the wrappers.
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
/// commitment since OFFSET is page-relative and INIT is uniformly zero),
/// recomputes the commitment from scratch, and checks two things:
///
/// - the static bytes equal the recompute, read directly from
///   `static_zero_page_commitment` (panicking if the match arm is missing) —
///   so byte drift can't be masked by a broken dispatch gate;
/// - `zero_init_preprocessed_commitment` equals the recompute — the wrapper
///   returns a correct value whichever path it takes. (The coset-3 gate
///   itself is covered by the ignored
///   `page_non_three_coset_recomputes_and_differs_from_static` test.)
#[test]
fn zero_page_static_matches_recompute_for_all_blowups() {
    let zero_page_config = page::PageConfig::zero_init(0);
    for &blowup in STATIC_BLOWUP_FACTORS {
        let options = options_for(blowup);
        let recomputed = page::compute_precomputed_commitment(&zero_page_config, &options);
        let Some(static_bytes) = page::static_zero_page_commitment(blowup) else {
            panic!("no static zero-page match arm shipped for blowup={blowup}");
        };
        assert_eq!(
            static_bytes, recomputed,
            "static zero-init page commitment drifted for blowup={blowup}; \
             regenerate constants via \
             `cargo run --bin compute_static_commitments --release`",
        );
        let from_wrapper = page::zero_init_preprocessed_commitment(&options);
        assert_eq!(
            from_wrapper, recomputed,
            "zero_init_preprocessed_commitment returned a wrong value for blowup={blowup}",
        );
    }
}

/// Same drift guard for the private-input page's OFFSET-only commitment — the
/// verifier's compiled-in anchor for every private page, and the thing that
/// stops a prover repointing those rows at arbitrary addresses. Also asserts it
/// DIFFERS from the zero-init commitment: the two cover different column sets
/// (OFFSET alone vs OFFSET+INIT), so equal bytes would mean one of the two
/// call sites is committing the wrong number of columns.
#[test]
fn private_page_static_matches_recompute_for_all_blowups() {
    for &blowup in STATIC_BLOWUP_FACTORS {
        let options = options_for(blowup);
        let recomputed = page::compute_offset_only_commitment(&options);
        let Some(static_bytes) = page::static_private_page_commitment(blowup) else {
            panic!("no static private-page match arm shipped for blowup={blowup}");
        };
        assert_eq!(
            static_bytes, recomputed,
            "static private-page (OFFSET-only) commitment drifted for blowup={blowup}; \
             regenerate constants via \
             `cargo run --bin compute_static_commitments --release`",
        );
        let from_wrapper = page::private_page_preprocessed_commitment(&options);
        assert_eq!(
            from_wrapper, recomputed,
            "private_page_preprocessed_commitment returned a wrong value for blowup={blowup}",
        );
        assert_ne!(
            recomputed,
            page::compute_precomputed_commitment(&page::PageConfig::zero_init(0), &options),
            "OFFSET-only and OFFSET+INIT commitments must differ (blowup={blowup}); \
             equality would mean a call site commits the wrong column count",
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
    let from_wrapper = page::zero_init_preprocessed_commitment(&options);
    let recomputed = page::compute_precomputed_commitment(&zero_page_config, &options);
    assert_eq!(
        from_wrapper, recomputed,
        "page fallback returned a value that doesn't match direct compute at blowup={NON_STATIC_BLOWUP}",
    );
}

/// Regression test for the `options.coset_offset == 3` gate in
/// `page::zero_init_preprocessed_commitment`. With a non-3 coset offset, the
/// wrapper must NOT return a static value — it must recompute (matching direct
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

        let from_wrapper_7 = page::zero_init_preprocessed_commitment(&opts_coset7);
        let recomputed_7 = page::compute_precomputed_commitment(&zero_page_config, &opts_coset7);
        let from_wrapper_3 = page::zero_init_preprocessed_commitment(&opts_coset3);

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

// =========================================================================
// One-row (S2) twins: a missing twin is a hard miss, never a recompute
// =========================================================================
//
// Each static table ships a SECOND match table for the one-row leaf layout
// (`*_one_row`), generated by `compute_static_commitments --layout row` for
// `STATIC_BLOWUP_FACTORS_ONE_ROW`. The row-pair tests above are untouched;
// these pin the twins the same way, and pin that a missing twin is a hard
// miss (`None`, the prover's `PrecomputedCommitmentMissing`), never a
// recompute.

use stark::leaf_layout::LeafLayout;

use crate::tables::STATIC_BLOWUP_FACTORS_ONE_ROW;

#[test]
fn bitwise_one_row_static_matches_recompute() {
    for &blowup in STATIC_BLOWUP_FACTORS_ONE_ROW {
        let options = options_for(blowup);
        let recomputed = bitwise::compute_preprocessed_commitment_with(&options, LeafLayout::Row);
        assert_eq!(
            bitwise::static_commitment_one_row(blowup),
            Some(recomputed),
            "bitwise one-row commitment drifted for blowup={blowup}; regenerate via \
             `cargo run --bin compute_static_commitments --release -- --layout row`",
        );
        assert_eq!(
            bitwise::preprocessed_commitment_for(&options, LeafLayout::Row),
            Some(recomputed)
        );
        assert_ne!(
            recomputed,
            bitwise::preprocessed_commitment(&options),
            "the two layouts commit different bytes"
        );
    }
}

#[test]
fn keccak_rc_one_row_static_matches_recompute() {
    for &blowup in STATIC_BLOWUP_FACTORS_ONE_ROW {
        let options = options_for(blowup);
        let recomputed = keccak_rc::compute_preprocessed_commitment_with(&options, LeafLayout::Row);
        assert_eq!(
            keccak_rc::static_commitment_one_row(blowup),
            Some(recomputed),
            "keccak_rc one-row commitment drifted for blowup={blowup}"
        );
        assert_eq!(
            keccak_rc::preprocessed_commitment_for(&options, LeafLayout::Row),
            Some(recomputed)
        );
        assert_ne!(recomputed, keccak_rc::preprocessed_commitment(&options));
    }
}

#[test]
fn pages_one_row_static_match_recompute() {
    let zero_page_config = page::PageConfig::zero_init(0);
    for &blowup in STATIC_BLOWUP_FACTORS_ONE_ROW {
        let options = options_for(blowup);
        let zero =
            page::compute_precomputed_commitment_with(&zero_page_config, &options, LeafLayout::Row);
        assert_eq!(
            page::static_zero_page_commitment_one_row(blowup),
            Some(zero)
        );
        assert_eq!(
            page::zero_init_preprocessed_commitment_for(&options, LeafLayout::Row),
            Some(zero)
        );
        assert_ne!(zero, page::zero_init_preprocessed_commitment(&options));
        let private = page::compute_offset_only_commitment_with(&options, LeafLayout::Row);
        assert_eq!(
            page::static_private_page_commitment_one_row(blowup),
            Some(private)
        );
        assert_eq!(
            page::private_page_preprocessed_commitment_for(&options, LeafLayout::Row),
            Some(private)
        );
        assert_ne!(private, zero, "OFFSET alone vs OFFSET+INIT");
    }
}

/// Under one row, a blowup with no twin and a non-3 coset are
/// HARD MISSES — `None`, never the recompute the row-pair wrappers fall back
/// to (which would silently rebuild a 2^20-row BITWISE LDE and tree). The
/// row-pair layout keeps today's answers.
#[test]
fn a_missing_one_row_twin_is_a_hard_miss() {
    for blowup in [2u8, 8, NON_STATIC_BLOWUP] {
        assert!(!STATIC_BLOWUP_FACTORS_ONE_ROW.contains(&blowup));
        let options = options_for(blowup);
        assert_eq!(bitwise::static_commitment_one_row(blowup), None);
        assert_eq!(
            bitwise::preprocessed_commitment_for(&options, LeafLayout::Row),
            None
        );
        assert_eq!(
            keccak_rc::preprocessed_commitment_for(&options, LeafLayout::Row),
            None
        );
        assert_eq!(
            page::zero_init_preprocessed_commitment_for(&options, LeafLayout::Row),
            None
        );
        assert_eq!(
            page::private_page_preprocessed_commitment_for(&options, LeafLayout::Row),
            None
        );
    }
    for &blowup in STATIC_BLOWUP_FACTORS_ONE_ROW {
        let options = options_with_coset(blowup, NON_STANDARD_COSET);
        assert_eq!(
            bitwise::preprocessed_commitment_for(&options, LeafLayout::Row),
            None
        );
        assert_eq!(
            keccak_rc::preprocessed_commitment_for(&options, LeafLayout::Row),
            None
        );
        assert_eq!(
            page::zero_init_preprocessed_commitment_for(&options, LeafLayout::Row),
            None
        );
        assert_eq!(
            page::private_page_preprocessed_commitment_for(&options, LeafLayout::Row),
            None
        );
    }
    // Row pairs: unchanged (the static root at a shipped blowup).
    let options = options_for(2);
    assert_eq!(
        keccak_rc::preprocessed_commitment_for(&options, LeafLayout::RowPair),
        Some(keccak_rc::preprocessed_commitment(&options))
    );
}

/// The lazy commitment sources the AIRs are built with serve both layouts:
/// today's root for row pairs (unchanged) and the twin for one row.
#[test]
fn the_air_commitment_sources_serve_both_layouts() {
    let options = options_for(4);
    let k = keccak_rc::lazy_commitment(&options);
    assert_eq!(
        k.get_for(LeafLayout::RowPair),
        Some(keccak_rc::preprocessed_commitment(&options))
    );
    assert_eq!(
        k.get_for(LeafLayout::Row),
        keccak_rc::static_commitment_one_row(4)
    );
    let p = page::private_page_lazy_commitment(&options);
    assert_eq!(
        p.get_for(LeafLayout::Row),
        page::static_private_page_commitment_one_row(4)
    );
    // A data page: computed on demand, per layout.
    let mut config = page::PageConfig::zero_init(0x1000);
    config.init_values = Some((0..64u8).collect());
    let d = page::data_page_lazy_commitment(&config, &options, None);
    assert_eq!(
        d.get_for(LeafLayout::Row),
        Some(page::compute_precomputed_commitment_with(
            &config,
            &options,
            LeafLayout::Row
        ))
    );
    assert_eq!(
        d.get_for(LeafLayout::RowPair),
        Some(page::compute_precomputed_commitment(&config, &options))
    );
    // A SUPPLIED row-pair root is never handed out for the other layout.
    let supplied = page::data_page_lazy_commitment(&config, &options, Some([7u8; 32]));
    assert_eq!(supplied.get_for(LeafLayout::RowPair), Some([7u8; 32]));
    assert_ne!(supplied.get_for(LeafLayout::Row), Some([7u8; 32]));
    // REGISTER (program-dependent): both computed.
    let init: Vec<u32> = (0..crate::tables::register::NUM_REGISTER_ADDRESSES as u32).collect();
    let r = crate::tables::register::lazy_commitment(&options, &init);
    assert_eq!(
        r.get_for(LeafLayout::Row),
        Some(
            crate::tables::register::compute_precomputed_commitment_with(
                &options,
                &init,
                LeafLayout::Row
            )
        )
    );
}
