//! A build of the prover library that is not its own unit-test binary proves
//! and verifies at production, with or without the `test-security` feature:
//! run it as `cargo test -p lambda-vm-prover --features test-security --test
//! security_posture` to see the feature alone reach nothing.

use lambda_vm_prover::security_posture;
use lambda_vm_prover::zf_format::ZfFormat;

#[test]
fn a_build_outside_the_libs_own_tests_is_production_whatever_its_features() {
    // `black_box`: the constant IS what is checked (which value this build compiled).
    assert!(!std::hint::black_box(security_posture::TEST_POSTURE));
    assert_eq!(security_posture::name(), "production");
    assert_eq!(security_posture::banner(), None);
    assert_eq!(security_posture::chain_security_bits(), 128);
    assert_eq!(security_posture::whir_grind_bits(18), 18);
    let chain =
        lambda_vm_prover::multilinear_prove::chain_config_under(&ZfFormat::DEFAULT, &[(1, 25)]);
    assert_eq!((chain.num_queries, chain.grind.query), (229, 18));
    let wrap = lambda_vm_prover::lfm::proof::aggregation_wrap_options_under(&ZfFormat::DEFAULT);
    assert_eq!(
        (wrap.fri_number_of_queries, wrap.grinding_factor),
        (112, 20)
    );
    let base = lambda_vm_prover::lfm::proof::block_base_options_under(&ZfFormat::DEFAULT);
    assert_eq!(
        (base.fri_number_of_queries, base.grinding_factor),
        (110, 20)
    );
}
