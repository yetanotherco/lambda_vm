//! Integration tests for FRI configurations: higher-arity folding and early termination.
//!
//! Tests prove and verify with various (log_arity, log_final_poly_len) configurations
//! to validate correctness of the higher-arity FRI implementation.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, u64_goldilocks::GoldilocksField,
};

use crate::{
    examples::fibonacci_multi_column::{self, FibonacciMultiColumnAIR},
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    verifier::{IsStarkVerifier, Verifier},
};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

fn prove_and_verify(log_arity: u8, log_final_poly_len: u8) {
    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 3,
        coset_offset: 3,
        grinding_factor: 1,
        fri_log_arity: log_arity,
        fri_log_final_poly_len: log_final_poly_len,
    };
    let num_columns = 2;
    let trace_length = 16;

    let initial_values: Vec<(FE, FE)> = (0..num_columns)
        .map(|i| (FE::from((i + 1) as u64), FE::from((i + 2) as u64)))
        .collect();

    let mut trace = fibonacci_multi_column::compute_trace::<F, E>(&initial_values, trace_length);
    let pub_inputs = fibonacci_multi_column::create_public_inputs(initial_values);
    let air = FibonacciMultiColumnAIR::<F, E>::with_num_columns(&proof_options, num_columns);

    let proof = Prover::<F, E, _>::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .unwrap();

    assert!(
        Verifier::<F, E, _>::verify(&proof, &air, &mut DefaultTranscript::<E>::new(&[]),),
        "Verification failed for log_arity={log_arity}, log_final_poly_len={log_final_poly_len}"
    );
}

// --- Baseline: arity-2, fold to constant (current behavior) ---

#[test]
fn test_fri_arity2_final0() {
    prove_and_verify(1, 0);
}

// --- Early termination only (arity-2) ---

#[test]
fn test_fri_arity2_final2() {
    prove_and_verify(1, 2);
}

#[test]
fn test_fri_arity2_final3() {
    prove_and_verify(1, 3);
}

// --- Higher arity only (fold to constant) ---

#[test]
fn test_fri_arity4_final0() {
    prove_and_verify(2, 0);
}

#[test]
fn test_fri_arity8_final0() {
    prove_and_verify(3, 0);
}

// --- Both optimizations ---

#[test]
fn test_fri_arity4_final2() {
    prove_and_verify(2, 2);
}

#[test]
fn test_fri_arity4_final3() {
    prove_and_verify(2, 3);
}

#[test]
fn test_fri_arity8_final2() {
    prove_and_verify(3, 2);
}
