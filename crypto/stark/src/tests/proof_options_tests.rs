use crate::proof::options::{GoldilocksCubicProofOptions, ProofOptions, ProofOptionsError};

#[test]
fn jbr_queries_match_expected_values() {
    // These counts were verified against zisk's pil2-proofman-js security
    // calculator, which runs at the historical Johnson gap of 1/300.
    // ⚠ Blowups 2 and 4 now use a re-tuned gap, so their agreement with that
    // calculator is no longer the reason they hold — the (gap, budget) pair is
    // chosen to keep the count fixed, and `the_production_posture_is_pinned`
    // is what states that intent. Blowups 8, 32 and 64 are still on 1/300 and
    // still match the calculator directly.
    assert_eq!(
        GoldilocksCubicProofOptions::with_blowup(2)
            .unwrap()
            .fri_number_of_queries,
        219
    );
    assert_eq!(
        GoldilocksCubicProofOptions::with_blowup(4)
            .unwrap()
            .fri_number_of_queries,
        110
    );
    assert_eq!(
        GoldilocksCubicProofOptions::with_blowup(8)
            .unwrap()
            .fri_number_of_queries,
        73
    );
    // with_params allows custom grinding — zisk uses 22 for final layers
    assert_eq!(
        GoldilocksCubicProofOptions::with_params(32, 128, 22)
            .unwrap()
            .fri_number_of_queries,
        43
    );
    assert_eq!(
        GoldilocksCubicProofOptions::with_params(64, 128, 22)
            .unwrap()
            .fri_number_of_queries,
        36
    );
    // default grinding=20 gives slightly more queries
    assert_eq!(
        GoldilocksCubicProofOptions::with_blowup(32)
            .unwrap()
            .fri_number_of_queries,
        44
    );
    assert_eq!(
        GoldilocksCubicProofOptions::with_blowup(64)
            .unwrap()
            .fri_number_of_queries,
        37
    );
}

#[test]
fn default_grinding_is_20() {
    assert_eq!(
        GoldilocksCubicProofOptions::with_blowup(4)
            .unwrap()
            .grinding_factor,
        20
    );
    assert_eq!(
        GoldilocksCubicProofOptions::with_blowup(64)
            .unwrap()
            .grinding_factor,
        20
    );
}

#[test]
fn custom_grinding() {
    let opts = GoldilocksCubicProofOptions::with_params(4, 128, 22).unwrap();
    assert_eq!(opts.grinding_factor, 22);
    // More grinding → fewer queries needed, HOLDING THE BUDGET FIXED. Both
    // sides must pass the same `security_bits`: `with_blowup` carries a
    // per-blowup budget, so comparing against it would attribute a budget
    // difference to grinding and read backwards the moment the two diverge.
    assert!(
        opts.fri_number_of_queries
            < GoldilocksCubicProofOptions::with_params(4, 128, 20)
                .unwrap()
                .fri_number_of_queries
    );
}

/// ★ The posture pin. Every one of these numbers is a deliberate choice, and
/// the defect it guards is a constant drifting silently: the Johnson gap sat at
/// 1/300 for a year while nothing recomputed what it cost.
///
/// ⚠ QUERY COUNTS MUST NOT MOVE. The re-tune is free precisely because it holds
/// 219 / 110 / 73 — the gap feeds both `eps_C` and `bits_per_query`, so the
/// budget moves with it to keep the count fixed. A query count changing here
/// means the (gap, budget) pair is wrong, not that this test is stale.
#[test]
fn the_production_posture_is_pinned() {
    // (blowup, security_bits, grinding, queries)
    let expected = [(2u8, 118u8, 20u8, 219usize), (4, 120, 20, 110)];
    for (blowup, bits, grinding, queries) in expected {
        let opts = GoldilocksCubicProofOptions::with_blowup(blowup).expect("valid blowup");
        assert_eq!(
            opts.fri_number_of_queries, queries,
            "blowup {blowup}: query count moved — the re-tune is only free if it does not"
        );
        assert_eq!(opts.grinding_factor, grinding, "blowup {blowup}: grinding");
        // The budget is not stored on ProofOptions, so pin it the only way it
        // is observable: passing the same budget explicitly must reproduce the
        // count `with_blowup` produces.
        assert_eq!(
            GoldilocksCubicProofOptions::with_params(blowup, bits, grinding)
                .expect("valid params")
                .fri_number_of_queries,
            queries,
            "blowup {blowup}: with_blowup must be passing security_bits {bits}"
        );
    }

    // ⚠ blowup 8 is NOT re-tuned: its delivered bits are not computed, so it
    // keeps the historical gap and the 128 budget rather than inheriting a
    // neighbour's constant. 73 queries either way — the claim is what differs.
    assert_eq!(
        GoldilocksCubicProofOptions::with_blowup(8)
            .unwrap()
            .fri_number_of_queries,
        73
    );
    assert_eq!(
        GoldilocksCubicProofOptions::with_params(8, 128, 20)
            .unwrap()
            .fri_number_of_queries,
        73,
        "blowup 8 must still be on the 128 budget"
    );
}

#[test]
fn higher_blowup_means_fewer_queries() {
    let q2 = GoldilocksCubicProofOptions::with_blowup(2)
        .unwrap()
        .fri_number_of_queries;
    let q4 = GoldilocksCubicProofOptions::with_blowup(4)
        .unwrap()
        .fri_number_of_queries;
    let q8 = GoldilocksCubicProofOptions::with_blowup(8)
        .unwrap()
        .fri_number_of_queries;
    assert!(q2 > q4 && q4 > q8);
}

#[test]
fn rejects_non_power_of_two() {
    assert!(matches!(
        GoldilocksCubicProofOptions::with_blowup(3),
        Err(ProofOptionsError::InvalidBlowup(3))
    ));
}

#[test]
fn rejects_blowup_one() {
    assert!(matches!(
        GoldilocksCubicProofOptions::with_blowup(1),
        Err(ProofOptionsError::InvalidBlowup(1))
    ));
}

#[test]
fn rejects_security_below_grinding() {
    assert!(matches!(
        GoldilocksCubicProofOptions::with_params(4, 10, 20),
        Err(ProofOptionsError::SecurityTooLow { .. })
    ));
}

#[test]
fn test_options_unchanged() {
    let opts = ProofOptions::default_test_options();
    assert_eq!(opts.blowup_factor, 2);
    assert_eq!(opts.fri_number_of_queries, 3);
    assert_eq!(opts.grinding_factor, 1);
    assert_eq!(opts.fri_final_poly_log_degree, 7);
}

#[test]
fn with_blowup_sets_default_final_poly_log_degree() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("valid blowup");
    assert_eq!(opts.fri_final_poly_log_degree, 7);
}

#[test]
fn default_test_options_sets_final_poly_log_degree() {
    assert_eq!(
        ProofOptions::default_test_options().fri_final_poly_log_degree,
        7
    );
}
