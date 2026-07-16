//! Soundness negatives for the batched (unified-shard) verifier
//! `Verifier::batched_multi_verify`.
//!
//! Each test builds ONE valid `BatchedMultiProof` (three all-padding, bus-balanced
//! tables — the simplest valid multi-table epoch, Σ table_contribution = 0) and then
//! tampers a single component of the proof, asserting the verifier rejects. Together
//! they exercise every batched-specific verification path: the three shared
//! mixed-height MMCS openings (value + width binding), the fold-and-inject FRI query
//! check (last value, layer root, layer sym), the shared OOD/transcript replay, the
//! grinding nonce, the query-count guard, and the bus-balance check.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::proof::options::ProofOptions;
use crate::proof::stark::BatchedMultiProof;
use crate::table::Table;
use crate::test_utils::multi_prove_batched_ram;
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

/// Build a valid batched proof over three all-padding (bus-balanced) tables:
/// CPU (5 main columns) + ADD + MUL (4 main columns each), all 4 rows of zeros.
fn valid_padding_proof() -> BatchedMultiProof<F, E, ()> {
    let mut cpu_trace = TraceTable::from_columns_main(vec![vec![FE::zero(); 4]; 5], 1);
    let mut add_trace = TraceTable::from_columns_main(vec![vec![FE::zero(); 4]; 4], 1);
    let mut mul_trace = TraceTable::from_columns_main(vec![vec![FE::zero(); 4]; 4], 1);

    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    multi_prove_batched_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap()
}

/// Verify a batched proof with a fresh verifier (AIRs reconstructed, as a real
/// verifier would) against `expected_bus_balance`.
fn batched_verify(
    proof: &BatchedMultiProof<F, E, ()>,
    expected_bus_balance: FieldElement<E>,
) -> bool {
    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];
    Verifier::batched_multi_verify(
        &airs,
        proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &expected_bus_balance,
    )
}

/// Sanity anchor: the untampered proof verifies (guards against a false-reject
/// regression that would make every negative below pass vacuously).
#[test_log::test]
fn batched_valid_padding_proof_verifies() {
    let proof = valid_padding_proof();
    assert!(
        batched_verify(&proof, FieldElement::<E>::zero()),
        "a valid all-padding batched proof must verify"
    );
}

/// Tampering an opened MAIN-trace evaluation breaks the shared main MMCS auth path.
#[test_log::test]
fn batched_rejects_tampered_main_trace_opening() {
    let mut proof = valid_padding_proof();
    proof.deep_poly_openings[0].main.per_matrix[0].evaluations[0] += FE::one();
    assert!(!batched_verify(&proof, FieldElement::<E>::zero()));
}

/// Tampering an opened COMPOSITION evaluation breaks the shared composition MMCS auth path.
#[test_log::test]
fn batched_rejects_tampered_composition_opening() {
    let mut proof = valid_padding_proof();
    proof.deep_poly_openings[0].composition.per_matrix[0].evaluations[0] +=
        FieldElement::<E>::one();
    assert!(!batched_verify(&proof, FieldElement::<E>::zero()));
}

/// Tampering an opened AUX evaluation breaks the shared aux MMCS auth path.
#[test_log::test]
fn batched_rejects_tampered_aux_opening() {
    let mut proof = valid_padding_proof();
    proof.deep_poly_openings[0]
        .aux
        .as_mut()
        .expect("padding tables carry an aux (LogUp) trace")
        .per_matrix[0]
        .evaluations[0] += FieldElement::<E>::one();
    assert!(!batched_verify(&proof, FieldElement::<E>::zero()));
}

/// Width-binding: shrinking a per-matrix opening's column count (without changing the
/// flat leaf bytes it would concatenate into) must be caught by the MMCS width check.
#[test_log::test]
fn batched_rejects_main_opening_width_mismatch() {
    let mut proof = valid_padding_proof();
    // Drop one column from the opened main row → evaluations.len() != committed width.
    proof.deep_poly_openings[0].main.per_matrix[0]
        .evaluations
        .pop();
    assert!(!batched_verify(&proof, FieldElement::<E>::zero()));
}

/// Tampering the final FRI value breaks the fold-and-inject terminal check.
#[test_log::test]
fn batched_rejects_tampered_fri_last_value() {
    let mut proof = valid_padding_proof();
    proof.fri_last_value += FieldElement::<E>::one();
    assert!(!batched_verify(&proof, FieldElement::<E>::zero()));
}

/// Tampering a committed FRI layer root diverges the derived fold challenges and
/// invalidates that layer's opening.
#[test_log::test]
fn batched_rejects_tampered_fri_layer_root() {
    let mut proof = valid_padding_proof();
    assert!(
        !proof.fri_layers_merkle_roots.is_empty(),
        "expect >= 1 FRI layer"
    );
    proof.fri_layers_merkle_roots[0][0] ^= 1;
    assert!(!batched_verify(&proof, FieldElement::<E>::zero()));
}

/// Tampering a per-query symmetric layer evaluation breaks that FRI layer's opening.
#[test_log::test]
fn batched_rejects_tampered_layer_evaluation_sym() {
    let mut proof = valid_padding_proof();
    assert!(
        !proof.query_list[0].layers_evaluations_sym.is_empty(),
        "expect >= 1 FRI layer eval"
    );
    proof.query_list[0].layers_evaluations_sym[0] += FieldElement::<E>::one();
    assert!(!batched_verify(&proof, FieldElement::<E>::zero()));
}

/// Tampering a per-table composition-parts OOD value breaks the step-2 composition claim.
#[test_log::test]
fn batched_rejects_tampered_composition_ood() {
    let mut proof = valid_padding_proof();
    proof.per_table[1].composition_poly_parts_ood_evaluation[0] += FieldElement::<E>::one();
    assert!(!batched_verify(&proof, FieldElement::<E>::zero()));
}

/// Tampering a current-row trace-OOD value (the g·z-split current block) desyncs the
/// shared transcript replay / step-2 composition claim → rejected.
#[test_log::test]
fn batched_rejects_tampered_trace_ood_current() {
    let mut proof = valid_padding_proof();
    let orig = &proof.per_table[0].trace_ood_evaluations;
    let mut data = orig.row_major_data().to_vec();
    data[0] += FieldElement::<E>::one();
    proof.per_table[0].trace_ood_evaluations = Table::new(data, orig.width);
    assert!(!batched_verify(&proof, FieldElement::<E>::zero()));
}

/// Shape guard (I3): the current-row OOD block width is the physical trace width
/// (main + aux), fixed by the AIR. A too-narrow block is rejected before any row
/// access, mirroring the non-batched `ood_blocks_well_formed` guard.
#[test_log::test]
fn batched_rejects_malformed_current_row_ood_block() {
    let mut proof = valid_padding_proof();
    let orig = &proof.per_table[0].trace_ood_evaluations;
    let (w, h) = (orig.width, orig.height);
    assert!(w >= 2, "the CPU table has more than one trace column");
    // Drop the last column: width w-1, same row count.
    let mut data = Vec::with_capacity((w - 1) * h);
    for r in 0..h {
        data.extend_from_slice(&orig.get_row(r)[..w - 1]);
    }
    proof.per_table[0].trace_ood_evaluations = Table::new(data, w - 1);
    assert!(!batched_verify(&proof, FieldElement::<E>::zero()));
}

/// Shape guard (I3): the pruned next-row OOD block width is the transition window
/// (here the LogUp accumulator, width 1). Emptying it — a prover trying to drop the
/// surviving next-row opening — is rejected on shape before Round 3 absorbs it.
#[test_log::test]
fn batched_rejects_malformed_next_row_ood_block() {
    let mut proof = valid_padding_proof();
    assert!(
        proof.per_table[0].trace_ood_next_evaluations.width >= 1,
        "bus tables open the accumulator column at the next row"
    );
    proof.per_table[0].trace_ood_next_evaluations = Table::new(Vec::new(), 0);
    assert!(!batched_verify(&proof, FieldElement::<E>::zero()));
}

/// Dropping a query opening leaves fewer than `fri_number_of_queries` → rejected.
#[test_log::test]
fn batched_rejects_dropped_query_opening() {
    let mut proof = valid_padding_proof();
    proof.deep_poly_openings.pop();
    assert!(!batched_verify(&proof, FieldElement::<E>::zero()));
}

/// Removing the grinding nonce (with grinding_factor > 0) fails the proof-of-work check.
#[test_log::test]
fn batched_rejects_missing_nonce() {
    let mut proof = valid_padding_proof();
    proof.nonce = None;
    assert!(!batched_verify(&proof, FieldElement::<E>::zero()));
}

/// A valid Σ = 0 proof must be rejected when a NONZERO bus balance is expected.
#[test_log::test]
fn batched_rejects_wrong_expected_bus_balance() {
    let proof = valid_padding_proof();
    assert!(!batched_verify(&proof, FieldElement::<E>::one()));
}
