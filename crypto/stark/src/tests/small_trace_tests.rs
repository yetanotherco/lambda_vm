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
    tests::trace_test_helpers::make_valid_simple_proof,
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
    let (air, proof) = make_valid_simple_proof();

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
    let (air, mut proof) = make_valid_simple_proof();

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

/// A malformed proof that drops entries from
/// `composition_poly_parts_ood_evaluation` so the verifier indexes past the
/// end during deep composition. The `.get(j)?` bounds check must cause the
/// verifier to return `false` instead of panicking.
#[test_log::test]
fn test_verify_rejects_truncated_composition_poly_parts_ood() {
    let (air, mut proof) = make_valid_simple_proof();

    assert!(
        !proof.composition_poly_parts_ood_evaluation.is_empty(),
        "test precondition: a valid proof has at least one composition poly part",
    );
    // Drop one entry so the per-query opening has more parts than the header.
    proof.composition_poly_parts_ood_evaluation.pop();

    assert!(
        !Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verifier must reject when composition_poly_parts_ood_evaluation is truncated"
    );
}

/// A malformed proof whose `deep_poly_openings` Vec is shorter than the FRI
/// query count. `reconstruct_deep_composition_poly_evaluations_for_all_queries`
/// indexes `deep_poly_openings[i]` for every query index, and this Vec's length
/// is not otherwise bound (the `query_list.len()` guard checks a different
/// field), so a truncated `deep_poly_openings` must make the verifier return
/// `false` instead of panicking with an out-of-bounds index in release builds.
#[test_log::test]
fn test_verify_rejects_truncated_deep_poly_openings() {
    let (air, mut proof) = make_valid_simple_proof();

    assert!(
        proof.deep_poly_openings.len() >= 2,
        "test precondition: a valid proof has one deep-poly opening per FRI query",
    );
    // Drop the last opening so the Vec is shorter than `fri_number_of_queries`;
    // the query loop would then index past the end.
    proof.deep_poly_openings.pop();

    assert!(
        !Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verifier must reject when deep_poly_openings is shorter than the query count"
    );
}

/// A malformed proof whose deep-poly opening `evaluations` slice has the
/// wrong number of columns. The runtime width-mismatch guard added in this
/// PR must cause the verifier to return `false` instead of indexing past
/// the end of `lde_trace_aux_evaluations` and panicking in release builds.
#[test_log::test]
fn test_verify_rejects_opening_column_count_mismatch() {
    let (air, mut proof) = make_valid_simple_proof();

    // Append a phantom extra evaluation column to the first query's
    // main-trace opening so the (base + aux) count exceeds `ood_evaluations_table_width`.
    if let Some(opening) = proof.deep_poly_openings.first_mut() {
        let extra = opening
            .main_trace_polys
            .evaluations
            .last()
            .cloned()
            .unwrap_or_else(Felt::zero);
        opening.main_trace_polys.evaluations.push(extra);
        opening.main_trace_polys.evaluations_sym.push(extra);
    } else {
        panic!("test precondition: a valid proof has at least one deep poly opening");
    }

    assert!(
        !Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verifier must reject when an opening's column count does not match the OOD table width"
    );
}
