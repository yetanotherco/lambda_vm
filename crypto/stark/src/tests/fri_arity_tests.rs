//! Higher-arity FRI prove/verify roundtrip tests.
//!
//! These tests verify that the prover and verifier work correctly with
//! arity-4 (fri_log_arity=2) and arity-8 (fri_log_arity=3) FRI folding,
//! and that higher arity produces fewer Merkle roots (fewer FRI layers).

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::{element::FieldElement, goldilocks::GoldilocksField};

use crate::{
    examples::simple_fibonacci::{FibonacciAIR, FibonacciPublicInputs, fibonacci_trace},
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    traits::AIR,
    verifier::{IsStarkVerifier, Verifier},
};

type F = GoldilocksField;
type Felt = FieldElement<GoldilocksField>;

/// Prove and verify Fibonacci AIR with arity-4 FRI (fri_log_arity = 2).
///
/// Each FRI round folds 4× instead of 2×, so we expect half as many
/// Merkle roots compared to arity-2 with the same trace.
#[test_log::test]
fn test_fri_arity_4_prove_verify() {
    let mut trace = fibonacci_trace([Felt::from(1), Felt::from(1)], 64);

    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 3,
        coset_offset: 3,
        grinding_factor: 1,
        fri_log_arity: 2, // arity-4
        fri_log_final_poly_len: 0,
    };

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    };

    let air = FibonacciAIR::<F>::new(&proof_options);

    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .expect("Proving failed for arity-4 FRI");

    assert!(
        Verifier::verify(&proof, &air, &mut DefaultTranscript::<F>::new(&[])),
        "Verification failed for arity-4 FRI"
    );
}

/// Prove and verify Fibonacci AIR with arity-8 FRI (fri_log_arity = 3).
///
/// Each FRI round folds 8× instead of 2×, so we expect roughly one-third
/// as many Merkle roots compared to arity-2 with the same trace.
#[test_log::test]
fn test_fri_arity_8_prove_verify() {
    let mut trace = fibonacci_trace([Felt::from(1), Felt::from(1)], 64);

    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 3,
        coset_offset: 3,
        grinding_factor: 1,
        fri_log_arity: 3, // arity-8
        fri_log_final_poly_len: 0,
    };

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt::one(),
        a1: Felt::one(),
    };

    let air = FibonacciAIR::<F>::new(&proof_options);

    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .expect("Proving failed for arity-8 FRI");

    assert!(
        Verifier::verify(&proof, &air, &mut DefaultTranscript::<F>::new(&[])),
        "Verification failed for arity-8 FRI"
    );
}

/// Verify that arity-4 FRI produces strictly fewer Merkle roots than arity-2.
///
/// More folds per round → fewer total rounds → fewer committed Merkle trees.
#[test_log::test]
fn test_fri_arity_4_has_fewer_merkle_roots_than_arity_2() {
    let trace_length = 64;

    // Arity-2 proof (default)
    let proof_arity2 = {
        let mut trace = fibonacci_trace([Felt::from(1), Felt::from(1)], trace_length);
        let proof_options = ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 3,
            coset_offset: 3,
            grinding_factor: 1,
            fri_log_arity: 1, // arity-2
            fri_log_final_poly_len: 0,
        };
        let pub_inputs = FibonacciPublicInputs {
            a0: Felt::one(),
            a1: Felt::one(),
        };
        let air = FibonacciAIR::<F>::new(&proof_options);
        Prover::prove(
            &air,
            &mut trace,
            &pub_inputs,
            &mut DefaultTranscript::<F>::new(&[]),
        )
        .expect("Proving failed for arity-2 FRI")
    };

    // Arity-4 proof
    let proof_arity4 = {
        let mut trace = fibonacci_trace([Felt::from(1), Felt::from(1)], trace_length);
        let proof_options = ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 3,
            coset_offset: 3,
            grinding_factor: 1,
            fri_log_arity: 2, // arity-4
            fri_log_final_poly_len: 0,
        };
        let pub_inputs = FibonacciPublicInputs {
            a0: Felt::one(),
            a1: Felt::one(),
        };
        let air = FibonacciAIR::<F>::new(&proof_options);
        Prover::prove(
            &air,
            &mut trace,
            &pub_inputs,
            &mut DefaultTranscript::<F>::new(&[]),
        )
        .expect("Proving failed for arity-4 FRI")
    };

    let roots_arity2 = proof_arity2.fri_layers_merkle_roots.len();
    let roots_arity4 = proof_arity4.fri_layers_merkle_roots.len();

    assert!(
        roots_arity4 < roots_arity2,
        "Expected arity-4 to have fewer Merkle roots than arity-2, \
         but got arity-2={roots_arity2}, arity-4={roots_arity4}"
    );
}
