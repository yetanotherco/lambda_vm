//! Tests for FRI optimizations: early stopping (last_layer_degree_bound)
//! and higher folding factor (fri_folding_factor).
//!
//! These tests verify that non-default FRI parameters produce valid proofs
//! that pass verification.

use math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

use crate::{
    Felt252,
    examples::simple_fibonacci::{self, FibonacciAIR, FibonacciPublicInputs},
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    traits::AIR,
    transcript::StoneProverTranscript,
    verifier::{IsStarkVerifier, Verifier},
};

fn prove_and_verify_fib(trace_length: usize, proof_options: ProofOptions) {
    let mut trace =
        simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], trace_length);

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let air = FibonacciAIR::<Stark252PrimeField>::new(&proof_options);

    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut StoneProverTranscript::new(&[]),
    )
    .unwrap();

    assert!(
        Verifier::verify(&proof, &air, &mut StoneProverTranscript::new(&[])),
        "Verification failed with options: blowup={}, folding_factor={}, degree_bound={}, trace_len={}",
        proof_options.blowup_factor,
        proof_options.fri_folding_factor,
        proof_options.fri_last_layer_degree_bound,
        trace_length,
    );
}

fn options_with_fri(folding_factor: usize, degree_bound: usize) -> ProofOptions {
    ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 3,
        coset_offset: 3,
        grinding_factor: 0,
        fri_last_layer_degree_bound: degree_bound,
        fri_folding_factor: folding_factor,
    }
}

// --- Optimization 1: Early stopping (last_layer_degree_bound) ---

#[test_log::test]
fn test_fri_early_stop_degree_bound_1() {
    prove_and_verify_fib(64, options_with_fri(2, 1));
}

#[test_log::test]
fn test_fri_early_stop_degree_bound_3() {
    prove_and_verify_fib(64, options_with_fri(2, 3));
}

#[test_log::test]
fn test_fri_early_stop_degree_bound_7() {
    prove_and_verify_fib(64, options_with_fri(2, 7));
}

#[test_log::test]
fn test_fri_early_stop_degree_bound_15() {
    prove_and_verify_fib(256, options_with_fri(2, 15));
}

// --- Optimization 2: Higher folding factor ---

#[test_log::test]
fn test_fri_folding_factor_4() {
    prove_and_verify_fib(64, options_with_fri(4, 0));
}

#[test_log::test]
fn test_fri_folding_factor_8() {
    prove_and_verify_fib(256, options_with_fri(8, 0));
}

// --- Both optimizations combined ---

#[test_log::test]
fn test_fri_folding_4_degree_bound_3() {
    prove_and_verify_fib(64, options_with_fri(4, 3));
}

#[test_log::test]
fn test_fri_folding_4_degree_bound_7() {
    prove_and_verify_fib(256, options_with_fri(4, 7));
}

#[test_log::test]
fn test_fri_folding_8_degree_bound_7() {
    prove_and_verify_fib(256, options_with_fri(8, 7));
}

// --- Default parameters (backward compatibility sanity check) ---

#[test_log::test]
fn test_fri_default_options_64() {
    prove_and_verify_fib(64, options_with_fri(2, 0));
}

#[test_log::test]
fn test_fri_default_options_256() {
    prove_and_verify_fib(256, options_with_fri(2, 0));
}

// --- Extension field test: exercises ff>2 path where Field != FieldExtension ---

#[test_log::test]
fn test_fri_extension_field_folding_4_degree_bound_3() {
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::{
        element::FieldElement,
        fields::fft_friendly::{
            extensions_goldilocks::Degree3GoldilocksExtensionField,
            u64_goldilocks::GoldilocksField as GoldilocksBaseField,
        },
    };

    use crate::examples::fibonacci_multi_column::{self, FibonacciMultiColumnAIR};

    type GoldilocksField = GoldilocksBaseField;
    type GoldilocksExt = Degree3GoldilocksExtensionField;
    type GoldilocksFE = FieldElement<GoldilocksField>;

    let proof_options = ProofOptions {
        blowup_factor: 4,
        fri_number_of_queries: 3,
        coset_offset: 3,
        grinding_factor: 0,
        fri_last_layer_degree_bound: 3,
        fri_folding_factor: 4,
    };
    let num_columns = 2;
    let trace_length = 16;

    let initial_values: Vec<(GoldilocksFE, GoldilocksFE)> = (0..num_columns)
        .map(|i| {
            (
                GoldilocksFE::from((i + 1) as u64),
                GoldilocksFE::from((i + 2) as u64),
            )
        })
        .collect();

    let mut trace = fibonacci_multi_column::compute_trace::<GoldilocksField, GoldilocksExt>(
        &initial_values,
        trace_length,
    );
    let pub_inputs = fibonacci_multi_column::create_public_inputs(initial_values);
    let air = FibonacciMultiColumnAIR::<GoldilocksField, GoldilocksExt>::with_num_columns(
        &proof_options,
        num_columns,
    );

    let proof = Prover::<GoldilocksField, GoldilocksExt, _>::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<GoldilocksExt>::new(&[]),
    )
    .unwrap();

    assert!(
        Verifier::<GoldilocksField, GoldilocksExt, _>::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksExt>::new(&[])
        ),
        "Extension field verification failed with ff=4, degree_bound=3"
    );
}
