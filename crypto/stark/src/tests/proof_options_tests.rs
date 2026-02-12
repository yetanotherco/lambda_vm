use crate::proof::options::{GoldilocksCubicProofOptions, ProofOptions};

#[test]
fn jbr_queries_match_expected_values() {
    // Verified against zisk's pil2-proofman-js security calculator
    assert_eq!(GoldilocksCubicProofOptions::new(2).fri_number_of_queries, 219);
    assert_eq!(GoldilocksCubicProofOptions::new(4).fri_number_of_queries, 110);
    assert_eq!(GoldilocksCubicProofOptions::new(8).fri_number_of_queries, 73);
    // with_params allows custom grinding — zisk uses 22 for final layers
    assert_eq!(GoldilocksCubicProofOptions::with_params(32, 128, 22).fri_number_of_queries, 43);
    assert_eq!(GoldilocksCubicProofOptions::with_params(64, 128, 22).fri_number_of_queries, 36);
    // default grinding=20 gives slightly more queries
    assert_eq!(GoldilocksCubicProofOptions::new(32).fri_number_of_queries, 44);
    assert_eq!(GoldilocksCubicProofOptions::new(64).fri_number_of_queries, 37);
}

#[test]
fn default_grinding_is_20() {
    assert_eq!(GoldilocksCubicProofOptions::new(4).grinding_factor, 20);
    assert_eq!(GoldilocksCubicProofOptions::new(64).grinding_factor, 20);
}

#[test]
fn custom_grinding() {
    let opts = GoldilocksCubicProofOptions::with_params(4, 128, 22);
    assert_eq!(opts.grinding_factor, 22);
    // More grinding → fewer queries needed
    assert!(opts.fri_number_of_queries < GoldilocksCubicProofOptions::new(4).fri_number_of_queries);
}

#[test]
fn higher_blowup_means_fewer_queries() {
    let q2 = GoldilocksCubicProofOptions::new(2).fri_number_of_queries;
    let q4 = GoldilocksCubicProofOptions::new(4).fri_number_of_queries;
    let q8 = GoldilocksCubicProofOptions::new(8).fri_number_of_queries;
    assert!(q2 > q4 && q4 > q8);
}

#[test]
#[should_panic(expected = "blowup_factor must be a power of 2")]
fn rejects_non_power_of_two() {
    GoldilocksCubicProofOptions::new(3);
}

#[test]
#[should_panic(expected = "blowup_factor must be a power of 2")]
fn rejects_blowup_one() {
    GoldilocksCubicProofOptions::new(1);
}

#[test]
fn test_options_unchanged() {
    let opts = ProofOptions::default_test_options();
    assert_eq!(opts.blowup_factor, 4);
    assert_eq!(opts.fri_number_of_queries, 3);
    assert_eq!(opts.grinding_factor, 1);
}
