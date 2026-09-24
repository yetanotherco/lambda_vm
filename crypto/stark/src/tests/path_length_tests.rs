//! Exact authentication-path lengths at the default format.
//!
//! Every tree's depth is a verifier constant: `log2(lde) − 1` for the trace,
//! precomputed, aux and composition trees (a leaf is a row pair), and
//! `log2(lde) − i − 2` for committed FRI layer `i` (pair leaves over
//! `lde / 2^(i+1)` values). The verifier used to fold a path of any length and
//! compare the result with the root; it now requires the exact length
//! (design/CAP.md §9.4, commit C1b). These tests pin that honest proofs meet
//! the lengths exactly and that a path one node short or long is rejected, for
//! each tree class the verifier walks.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;

use crate::examples::simple_addition::{
    SimpleAdditionAIR, SimpleAdditionPublicInputs, simple_addition_trace,
};
use crate::proof::options::ProofOptions;
use crate::proof::stark::StarkProof;
use crate::prover::{IsStarkProver, Prover};
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type FE = FieldElement<F>;
type PI = SimpleAdditionPublicInputs<F>;

/// 1024 rows at blowup 2: `lde = 2048`, so the trace trees are 10 deep and
/// FRI commits layers (final degree 2^7 < 1024).
const TRACE_ROWS: usize = 1024;
const LDE_LOG: usize = 11;

fn prove() -> (SimpleAdditionAIR<F>, StarkProof<F, F, PI>) {
    let options = ProofOptions::default_test_options();
    let air = SimpleAdditionAIR::<F>::new(&options);
    let pub_inputs = SimpleAdditionPublicInputs {
        a: FE::from(1u64),
        b: FE::from(2u64),
    };
    let mut trace = simple_addition_trace::<F>(TRACE_ROWS);
    let proof = Prover::prove(
        &air,
        &mut trace,
        &pub_inputs,
        &mut DefaultTranscript::<F>::new(&[]),
    )
    .expect("proving must succeed");
    (air, proof)
}

fn verifies(air: &SimpleAdditionAIR<F>, proof: &StarkProof<F, F, PI>) -> bool {
    Verifier::verify(proof, air, &mut DefaultTranscript::<F>::new(&[]))
}

#[test]
fn honest_paths_have_exactly_the_verifier_depths_and_verify() {
    let (air, proof) = prove();
    assert_eq!(
        air.options().blowup_factor as usize * TRACE_ROWS,
        1 << LDE_LOG
    );
    assert!(
        !proof.fri_layers_merkle_roots.is_empty(),
        "the trace must fold, or the FRI arm is vacuous"
    );
    for opening in &proof.deep_poly_openings {
        assert_eq!(
            opening.main_trace_polys.proof.merkle_path.len(),
            LDE_LOG - 1
        );
        assert_eq!(
            opening.composition_poly.proof.merkle_path.len(),
            LDE_LOG - 1
        );
    }
    for query in &proof.query_list {
        for (i, path) in query.layers_auth_paths.iter().enumerate() {
            assert_eq!(path.merkle_path.len(), LDE_LOG - i - 2, "layer {i}");
        }
    }
    assert!(verifies(&air, &proof), "an honest proof must verify");
}

/// One node short, one node long: both rejected.
fn assert_both_lengths_rejected(
    air: &SimpleAdditionAIR<F>,
    honest: &StarkProof<F, F, PI>,
    what: &str,
    path_of: impl Fn(&mut StarkProof<F, F, PI>) -> &mut Vec<[u8; 32]>,
) {
    let mut short = honest.clone();
    let path = path_of(&mut short);
    assert!(!path.is_empty(), "{what}: precondition, a non-empty path");
    path.pop();
    assert!(
        !verifies(air, &short),
        "{what}: a path one node short must be rejected"
    );

    let mut long = honest.clone();
    let path = path_of(&mut long);
    let extra = path[0];
    path.push(extra);
    assert!(
        !verifies(air, &long),
        "{what}: a path one node long must be rejected"
    );
}

#[test]
fn a_main_trace_path_of_the_wrong_length_is_rejected() {
    let (air, honest) = prove();
    assert_both_lengths_rejected(&air, &honest, "main, query 0", |p| {
        &mut p.deep_poly_openings[0].main_trace_polys.proof.merkle_path
    });
    let last = honest.deep_poly_openings.len() - 1;
    assert_both_lengths_rejected(&air, &honest, "main, last query", move |p| {
        &mut p.deep_poly_openings[last]
            .main_trace_polys
            .proof
            .merkle_path
    });
}

#[test]
fn a_composition_path_of_the_wrong_length_is_rejected() {
    let (air, honest) = prove();
    assert_both_lengths_rejected(&air, &honest, "composition, query 0", |p| {
        &mut p.deep_poly_openings[0].composition_poly.proof.merkle_path
    });
}

#[test]
fn a_fri_layer_path_of_the_wrong_length_is_rejected() {
    let (air, honest) = prove();
    let layers = honest.fri_layers_merkle_roots.len();
    for layer in [0, layers - 1] {
        assert_both_lengths_rejected(&air, &honest, &format!("FRI layer {layer}"), move |p| {
            &mut p.query_list[0].layers_auth_paths[layer].merkle_path
        });
    }
}
