use crate::proof::options::{GoldilocksCubicProofOptions, ProofOptions, ProofOptionsError};

#[test]
fn jbr_queries_match_expected_values() {
    // Verified against zisk's pil2-proofman-js security calculator
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
    // More grinding → fewer queries needed
    assert!(
        opts.fri_number_of_queries
            < GoldilocksCubicProofOptions::with_blowup(4)
                .unwrap()
                .fri_number_of_queries
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
