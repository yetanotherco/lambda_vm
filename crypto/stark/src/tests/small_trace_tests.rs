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
    tests::trace_test_helpers::{make_valid_logup_proof, make_valid_simple_proof},
    traits::AIR,
    verifier::{IsStarkVerifier, Verifier},
};
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;

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

/// Prove + verify with DEFAULT options (K=7) and a trace large enough that FRI
/// actually folds (trace_bits = 10 > 7). This exercises the full early-termination
/// path: committed FRI layers, a final fold, and terminal-codeword reconstruction
/// from the emitted final-polynomial coefficients.
#[test_log::test]
fn test_prove_verify_folding_default_options() {
    let mut trace = simple_addition_trace::<GoldilocksField>(1024);
    let proof_options = ProofOptions::default_test_options();
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
        "Verification failed for a folding trace under default options (K=7)"
    );
}

/// Prove + verify with DEFAULT options (K=7) and a tiny trace (trace_bits = 3 <= 7)
/// so the FRI final-polynomial degree is clamped (`expected_k = min(k, trace_bits)`)
/// and no folding happens (`total_folds == 0`). The terminal codeword is the deep
/// composition codeword itself and the verifier checks the deep evaluations against
/// it directly.
#[test_log::test]
fn test_prove_verify_tiny_trace_clamp() {
    let mut trace = simple_addition_trace::<GoldilocksField>(8);
    let proof_options = ProofOptions::default_test_options();
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
        "Verification failed for a clamped tiny trace under default options (K=7)"
    );
}

/// Prove + verify with DEFAULT options (K=7) and a 256-row trace (trace_bits=8).
/// With blowup=2 (blowup_log=1): expected_k = min(7,8) = 7, total_folds = 8-7 = 1.
/// This exercises the single-fold path: zero committed FRI layers, one final fold,
/// and the `fri_layers_merkle_roots.is_empty() && !zetas.is_empty()` branch in
/// `verify_query_and_sym_openings`.
#[test_log::test]
fn test_prove_verify_single_fold() {
    let mut trace = simple_addition_trace::<GoldilocksField>(256);
    let proof_options = ProofOptions::default_test_options();
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
    .expect("Failed to generate proof for single-fold trace");

    assert!(
        Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verification failed for single-fold trace (256 rows, total_folds=1)"
    );
}

/// Prove + verify with k=0: FRI folds all the way down to a single-coefficient
/// terminal polynomial (the closest analog to the old fold-to-constant behavior).
/// With a 1024-row trace: expected_k=0, total_folds=10, fri_final_poly_coeffs.len()=1,
/// fri_layers_merkle_roots.len()=9. Exercises the maximal-fold path.
#[test_log::test]
fn test_prove_verify_k0() {
    let mut trace = simple_addition_trace::<GoldilocksField>(1024);
    let mut proof_options = ProofOptions::default_test_options();
    proof_options.fri_final_poly_log_degree = 0;
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
    .expect("prover must succeed with k=0");

    assert_eq!(
        proof.fri_final_poly_coeffs.len(),
        1,
        "k=0 must emit a single terminal coefficient"
    );
    assert!(
        Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "k=0 proof must verify"
    );
}

/// Prove + verify with an oversized `fri_final_poly_log_degree`. A `k` this large
/// used to overflow `2^(blowup_log + k)` to 0 in the prover, dividing by zero.
/// It must instead clamp to no early termination (terminal_len == initial_len,
/// total_folds == 0) and still verify, mirroring the verifier's `min(k, root_order)`.
#[test_log::test]
fn test_prove_verify_oversized_k_clamps() {
    let mut trace = simple_addition_trace::<GoldilocksField>(1024);
    let mut proof_options = ProofOptions::default_test_options();
    proof_options.fri_final_poly_log_degree = 63;
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
    .expect("prover must clamp an oversized k instead of overflowing");

    assert!(
        proof.fri_layers_merkle_roots.is_empty(),
        "an oversized k must clamp to no early termination (no committed layers)"
    );
    assert!(
        Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "clamped oversized-k proof must verify"
    );
}

/// Prove + verify with `blowup_factor = 4` (blowup_log = 2). This is the only
/// test exercising a blowup > 2, so the terminal-codeword decimation
/// (`step_by(blowup)`) and the coset FFT run with a non-trivial blowup factor.
#[test_log::test]
fn test_prove_verify_blowup4() {
    let mut trace = simple_addition_trace::<GoldilocksField>(1024);
    let mut proof_options = ProofOptions::default_test_options();
    proof_options.blowup_factor = 4;
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
    .expect("prover must succeed with blowup_factor=4");

    assert!(
        Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "blowup_factor=4 proof must verify"
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

/// A malformed proof whose symmetric opening splits its columns between the
/// base and auxiliary segments differently from the regular opening, while
/// keeping the same total column count.
///
/// `reconstruct_deep_composition_poly_evaluation_pair` bounds its base-column
/// branch by the REGULAR `num_base` but resolves the symmetric column through
/// `base_at_sym`, which indexes the SYMMETRIC slices. The
/// `num_base != num_base_sym` guard is the only thing standing between this
/// proof and an out-of-bounds index panic in release builds, and it runs in
/// step 3 — before step 4's Merkle openings could reject the tampering.
///
/// The total width is deliberately preserved (one column moved from the
/// symmetric main trace to the symmetric aux trace) so that the width guards
/// cannot fire: only the base/aux split guard can reject this.
#[test_log::test]
fn test_verify_rejects_asymmetric_base_aux_split() {
    let (air, mut proof) = make_valid_logup_proof();

    // Sanity: the unmodified proof must verify first.
    assert!(
        Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<Degree3GoldilocksExtensionField>::new(&[])
        ),
        "precondition: valid LogUp proof must verify"
    );

    let opening = proof
        .deep_poly_openings
        .first_mut()
        .expect("test precondition: a valid proof has at least one deep poly opening");
    let aux = opening
        .aux_trace_polys
        .as_mut()
        .expect("test precondition: the LogUp RAP has an auxiliary trace segment");

    // Move one column from the symmetric main trace into the symmetric aux
    // trace: `num_base_sym` drops by one while `num_base_sym + aux_sym.len()`
    // still equals the OOD table width.
    let moved = opening
        .main_trace_polys
        .evaluations_sym
        .pop()
        .expect("test precondition: the main trace opening has at least one column");
    aux.evaluations_sym.push(moved.to_extension());

    assert!(
        !Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<Degree3GoldilocksExtensionField>::new(&[])
        ),
        "Verifier must reject when the regular and symmetric base-column counts disagree"
    );
}

/// A malformed proof whose symmetric composition-poly opening is shorter than
/// the proof-level `number_of_parts`.
///
/// The composition loop in `reconstruct_deep_composition_poly_evaluation_pair`
/// is bounded by `number_of_parts` (derived from
/// `composition_poly_parts_ood_evaluation`) and indexes both the regular and
/// symmetric per-query slices with it, so a short symmetric slice would index
/// out of bounds and panic in release builds. The length guard is the sole
/// protection and runs before step 4's Merkle openings.
///
/// Uses the LogUp RAP (`composition_poly_degree_bound == 2 * trace_length`) so
/// the multi-part loop is exercised rather than the degenerate one-part case.
#[test_log::test]
fn test_verify_rejects_truncated_symmetric_composition_opening() {
    let (air, mut proof) = make_valid_logup_proof();

    // Sanity: the unmodified proof must verify first.
    assert!(
        Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<Degree3GoldilocksExtensionField>::new(&[])
        ),
        "precondition: valid LogUp proof must verify"
    );
    assert!(
        proof.composition_poly_parts_ood_evaluation.len() > 1,
        "test precondition: the LogUp RAP must have more than one composition part, \
         so the multi-part loop is covered"
    );

    // Drop one symmetric part so the per-query opening is shorter than the
    // proof-level part count the loop is bounded by.
    proof.deep_poly_openings[0]
        .composition_poly
        .evaluations_sym
        .pop()
        .expect("test precondition: the composition opening has at least one part");

    assert!(
        !Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<Degree3GoldilocksExtensionField>::new(&[])
        ),
        "Verifier must reject when a symmetric composition opening is shorter than number_of_parts"
    );
}

// ---------------------------------------------------------------------------
// Helpers shared by the FRI early-termination soundness tests below.
// ---------------------------------------------------------------------------

/// Build a valid proof over a 1024-row trace (trace_bits=10) using the
/// default options (k=7, blowup=2).  With these parameters:
///   expected_k  = min(7, 10) = 7
///   total_folds = 10 - 7    = 3
///   fri_final_poly_coeffs.len() = 2^7 = 128
///   fri_layers_merkle_roots.len() = total_folds - 1 = 2
fn make_valid_folding_proof() -> (
    SimpleAdditionAIR<GoldilocksField>,
    crate::proof::stark::StarkProof<
        GoldilocksField,
        GoldilocksField,
        SimpleAdditionPublicInputs<GoldilocksField>,
    >,
) {
    let mut trace = simple_addition_trace::<GoldilocksField>(1024);
    let proof_options = ProofOptions::default_test_options();
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
    .expect("Prover failed to generate 1024-row folding proof");
    (air, proof)
}

// ---------------------------------------------------------------------------
// FRI early-termination soundness negative tests (Task 9)
// ---------------------------------------------------------------------------

/// Soundness: mutating one element of `fri_final_poly_coeffs` must cause
/// verification to fail.  The verifier absorbs every coefficient into the
/// Fiat-Shamir transcript before sampling query indices, so any modification
/// shifts all query challenges and invalidates the FRI openings.
#[test_log::test]
fn tampered_final_coeff_is_rejected() {
    let (air, mut proof) = make_valid_folding_proof();

    // Sanity: the unmodified proof must verify first.
    assert!(
        Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "precondition: valid folding proof must verify"
    );

    // Corrupt the first coefficient by adding 1.
    proof.fri_final_poly_coeffs[0] += Felt::one();

    assert!(
        !Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verifier must reject a proof with a tampered FRI final-poly coefficient"
    );
}

/// Soundness: pushing an extra element so `fri_final_poly_coeffs.len() > 2^k`
/// must be rejected by the structural degree check and must NOT panic.
/// The length check `len != 1 << expected_k` fires before the helper that
/// asserts a power-of-two length, so no assert is reachable.
#[test_log::test]
fn over_length_final_poly_is_rejected() {
    let (air, mut proof) = make_valid_folding_proof();

    // Sanity: the unmodified proof must verify first.
    assert!(
        Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "precondition: valid folding proof must verify"
    );

    // Extend to length 129 (not equal to 128 = 2^7).
    proof.fri_final_poly_coeffs.push(Felt::zero());

    assert!(
        !Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verifier must reject when fri_final_poly_coeffs is longer than 2^k (over-length)"
    );
}

/// Soundness: removing one element so `fri_final_poly_coeffs.len() < 2^k`
/// must be rejected and must NOT panic.  The verifier's length check
/// (`len != 1 << expected_k`) fires before `terminal_codeword_from_coeffs`
/// (which asserts power-of-two length), so no assert is triggered.
/// If this test panics instead of returning false, that is a real verifier bug.
#[test_log::test]
fn truncated_final_poly_is_rejected() {
    let (air, mut proof) = make_valid_folding_proof();

    // Sanity: the unmodified proof must verify first.
    assert!(
        Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "precondition: valid folding proof must verify"
    );

    // Shorten to length 127 (not equal to 128 = 2^7).
    proof.fri_final_poly_coeffs.pop();

    assert!(
        !Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verifier must reject when fri_final_poly_coeffs is shorter than 2^k (truncated)"
    );
}

/// Soundness: emptying every per-query FRI decommitment must be rejected.
///
/// In the multi-fold path, `verify_query_and_sym_openings` folds the query value
/// through a loop that `zip`s the (trusted-length) committed layer roots against
/// the per-query `layers_auth_paths` / `layers_evaluations_sym`. Those vecs come
/// from the untrusted proof and are NOT absorbed into the Fiat-Shamir transcript,
/// so emptying them (`zip` truncates to 0) would make the fold run zero iterations
/// and return `true` — no Merkle openings, no terminal low-degree check — bypassing
/// FRI. The per-query decommitment length check in `step_3_verify_fri` must reject
/// this before the fold loop runs.
#[test_log::test]
fn empty_fri_decommitment_is_rejected() {
    let (air, mut proof) = make_valid_folding_proof();

    // Sanity: the unmodified multi-fold proof must verify first.
    assert!(
        Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "precondition: valid folding proof must verify"
    );

    // Sanity: this is genuinely the multi-fold regime (num_committed >= 1).
    assert!(
        !proof.query_list[0].layers_evaluations_sym.is_empty(),
        "precondition: multi-fold proof must have at least one committed layer"
    );

    // Drop every per-query decommitment layer.
    for decommitment in proof.query_list.iter_mut() {
        decommitment.layers_auth_paths.clear();
        decommitment.layers_evaluations_sym.clear();
    }

    assert!(
        !Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verifier must reject a proof whose per-query FRI decommitment layers are empty"
    );
}

/// Soundness: padding every per-query FRI decommitment by one layer must be
/// rejected. With `layers_evaluations_sym.len() == num_committed + 1`, the fold
/// loop's last-iteration guard `i < layers_evaluations_sym.len() - 1` stays true,
/// so the terminal low-degree check in the `else` branch is never executed. The
/// per-query decommitment length check must reject this before the loop runs.
#[test_log::test]
fn padded_fri_decommitment_is_rejected() {
    let (air, mut proof) = make_valid_folding_proof();

    // Sanity: the unmodified multi-fold proof must verify first.
    assert!(
        Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "precondition: valid folding proof must verify"
    );

    // Append one junk layer (copied from the first real layer) to every query.
    let junk_eval = proof.query_list[0].layers_evaluations_sym[0];
    let junk_path = proof.query_list[0].layers_auth_paths[0].clone();
    for decommitment in proof.query_list.iter_mut() {
        decommitment.layers_evaluations_sym.push(junk_eval);
        decommitment.layers_auth_paths.push(junk_path.clone());
    }

    assert!(
        !Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verifier must reject a proof whose per-query FRI decommitment layers are padded"
    );
}

/// Soundness (single-fold regime, total_folds=1 ⇒ num_committed=0): the honest
/// decommitment carries zero layers. Padding it must be rejected by the per-query
/// decommitment length check, which runs for every regime — not only multi-fold.
#[test_log::test]
fn padded_decommitment_rejected_single_fold() {
    let mut trace = simple_addition_trace::<GoldilocksField>(256);
    let proof_options = ProofOptions::default_test_options();
    let pub_inputs = SimpleAdditionPublicInputs {
        a: Felt::from(1u64),
        b: Felt::from(2u64),
    };
    let air = SimpleAdditionAIR::<GoldilocksField>::new(&proof_options);
    let mut proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<GoldilocksField>::new(&[]),
    )
    .expect("prover must succeed (single-fold)");

    assert!(
        Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "precondition: single-fold proof must verify"
    );
    assert!(
        proof.query_list[0].layers_evaluations_sym.is_empty(),
        "precondition: single-fold has zero committed layers"
    );

    for decommitment in proof.query_list.iter_mut() {
        decommitment.layers_evaluations_sym.push(Felt::zero());
    }

    assert!(
        !Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verifier must reject a padded decommitment in the single-fold regime"
    );
}

/// Soundness (no-fold/clamp regime, total_folds=0 ⇒ num_committed=0): same as
/// above but on the clamped tiny-trace path, which also has zero committed layers.
#[test_log::test]
fn padded_decommitment_rejected_no_fold() {
    let mut trace = simple_addition_trace::<GoldilocksField>(8);
    let proof_options = ProofOptions::default_test_options();
    let pub_inputs = SimpleAdditionPublicInputs {
        a: Felt::from(1u64),
        b: Felt::from(2u64),
    };
    let air = SimpleAdditionAIR::<GoldilocksField>::new(&proof_options);
    let mut proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<GoldilocksField>::new(&[]),
    )
    .expect("prover must succeed (clamp/no-fold)");

    assert!(
        Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "precondition: clamped proof must verify"
    );
    assert!(
        proof.query_list[0].layers_evaluations_sym.is_empty(),
        "precondition: no-fold has zero committed layers"
    );

    for decommitment in proof.query_list.iter_mut() {
        decommitment.layers_evaluations_sym.push(Felt::zero());
    }

    assert!(
        !Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verifier must reject a padded decommitment in the no-fold regime"
    );
}

/// Soundness: a proof generated under k=7 must NOT verify when the verifier
/// uses k=6.  The verifier reads `fri_final_poly_log_degree` from the AIR it
/// is given, so constructing a fresh AIR with k=6 is sufficient to switch the
/// expected degree.
///
/// With a 1024-row trace (trace_bits=10):
///   Prover   (k=7): expected_k=7, total_folds=3, merkle_roots.len()=2
///   Verifier (k=6): expected_k=6, total_folds=4, expects merkle_roots.len()=3
/// The committed-layer count mismatch (2 vs 3) causes `step_3_verify_fri` to
/// return false immediately, before any transcript-dependent checks.
#[test_log::test]
fn cross_k_proof_does_not_verify() {
    let (air_k7, proof) = make_valid_folding_proof();

    // Sanity: the proof verifies under the matching k=7 AIR.
    assert!(
        Verifier::verify(
            &proof,
            &air_k7,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "precondition: valid folding proof must verify with k=7"
    );

    // Build a verifier AIR that expects k=6.
    let mut options_k6 = ProofOptions::default_test_options();
    options_k6.fri_final_poly_log_degree = 6;
    let air_k6 = SimpleAdditionAIR::<GoldilocksField>::new(&options_k6);

    assert!(
        !Verifier::verify(
            &proof,
            &air_k6,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verifier with k=6 must reject a proof generated with k=7 (cross-k mismatch)"
    );
}
