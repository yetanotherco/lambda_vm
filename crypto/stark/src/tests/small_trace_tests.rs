//! Tests for STARK proving/verification with small traces (1-2 rows).
//! These tests verify that the FRI protocol handles 0 FRI layers correctly.

use math::field::{element::FieldElement, goldilocks::GoldilocksField};

use crypto::fiat_shamir::default_transcript::DefaultTranscript;

use crate::{
    examples::simple_addition::{
        SimpleAdditionAIR, SimpleAdditionPublicInputs, simple_addition_trace,
    },
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    traits::AIR,
    verifier::{IsStarkVerifier, Verifier},
};

type Felt = FieldElement<GoldilocksField>;

/// Test STARK prove/verify with a single-row trace.
/// This exercises the FRI protocol with 0 FRI layers (trace_length=1, number_layers=0).
#[test_log::test]
fn test_prove_verify_single_row() {
    let mut trace = simple_addition_trace::<GoldilocksField>(1);

    let proof_options = ProofOptions::default_test_options();

    // For row 0: col0=1, col1=2, col2=3 (1+2=3)
    let pub_inputs = SimpleAdditionPublicInputs {
        a: Felt::from(1u64),
        b: Felt::from(2u64),
    };

    let air = SimpleAdditionAIR::<GoldilocksField>::new(&proof_options);

    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<GoldilocksField>::new(&[]),
    )
    .unwrap();

    assert!(
        Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verification failed for single-row trace"
    );
}

/// Test STARK prove/verify with a two-row trace.
/// This exercises the FRI protocol with 0 FRI layers (trace_length=2, number_layers=1).
#[test_log::test]
fn test_prove_verify_two_rows() {
    let mut trace = simple_addition_trace::<GoldilocksField>(2);

    let proof_options = ProofOptions::default_test_options();

    // For row 0: col0=1, col1=2, col2=3 (1+2=3)
    let pub_inputs = SimpleAdditionPublicInputs {
        a: Felt::from(1u64),
        b: Felt::from(2u64),
    };

    let air = SimpleAdditionAIR::<GoldilocksField>::new(&proof_options);

    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<GoldilocksField>::new(&[]),
    )
    .unwrap();

    assert!(
        Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verification failed for two-row trace"
    );
}

/// Test that verification fails when using wrong public inputs.
/// This ensures the boundary constraints are actually enforced.
#[test_log::test]
fn test_verify_fails_with_wrong_inputs() {
    let mut trace = simple_addition_trace::<GoldilocksField>(2);

    let proof_options = ProofOptions::default_test_options();

    // Correct public inputs for proving
    let correct_pub_inputs = SimpleAdditionPublicInputs {
        a: Felt::from(1u64),
        b: Felt::from(2u64),
    };

    let air = SimpleAdditionAIR::<GoldilocksField>::new(&proof_options);

    let mut proof = Prover::prove(
        &air,
        &mut trace,
        &correct_pub_inputs,
        &mut DefaultTranscript::<GoldilocksField>::new(&[]),
    )
    .unwrap();

    // Tamper with the proof's public inputs
    proof.public_inputs = SimpleAdditionPublicInputs {
        a: Felt::from(99u64), // Wrong value - doesn't match trace
        b: Felt::from(2u64),
    };

    // Verification should fail because boundary constraint col0[0]=99 doesn't match trace
    assert!(
        !Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verification should fail with tampered public inputs"
    );
}
