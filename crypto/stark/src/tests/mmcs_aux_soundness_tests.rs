//! Soundness tests for the shared AUX-trace MMCS path (mirror of
//! `mmcs_soundness_tests.rs`). Uses two `LogReadOnlyRAP` AIRs so both
//! tables have an aux trace and therefore both participate in the shared
//! aux MMCS — the only path that produces `AuxTraceOpening::Mmcs` data.
//!
//! Each test tampers with a single field on the aux MMCS path and
//! asserts the verifier rejects.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::merkle_tree::mmcs::MatrixTag;
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField,
    goldilocks::GoldilocksField,
};

use crate::examples::read_only_memory_logup::{
    LogReadOnlyPublicInputs, LogReadOnlyRAP, read_only_logup_trace,
};
use crate::proof::options::ProofOptions;
use crate::proof::stark::{AuxTraceOpening, MultiProof};
use crate::test_utils::{multi_prove_ram, multi_verify_ram};
use crate::traits::AIR;

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;

#[allow(clippy::type_complexity)]
fn baseline_proof() -> (
    LogReadOnlyRAP<F, E>,
    LogReadOnlyRAP<F, E>,
    MultiProof<F, E, LogReadOnlyPublicInputs<F>>,
) {
    let proof_options = ProofOptions::default_test_options();
    let air_1 = LogReadOnlyRAP::<F, E>::new(&proof_options);
    let air_2 = LogReadOnlyRAP::<F, E>::new(&proof_options);

    let address_col_1 = vec![
        FieldElement::<F>::from(3),
        FieldElement::<F>::from(2),
        FieldElement::<F>::from(2),
        FieldElement::<F>::from(3),
        FieldElement::<F>::from(4),
        FieldElement::<F>::from(5),
        FieldElement::<F>::from(1),
        FieldElement::<F>::from(3),
    ];
    let value_col_1 = vec![
        FieldElement::<F>::from(30),
        FieldElement::<F>::from(20),
        FieldElement::<F>::from(20),
        FieldElement::<F>::from(30),
        FieldElement::<F>::from(40),
        FieldElement::<F>::from(50),
        FieldElement::<F>::from(10),
        FieldElement::<F>::from(30),
    ];
    let address_col_2 = vec![
        FieldElement::<F>::from(15),
        FieldElement::<F>::from(12),
        FieldElement::<F>::from(17),
        FieldElement::<F>::from(10),
        FieldElement::<F>::from(14),
        FieldElement::<F>::from(11),
        FieldElement::<F>::from(16),
        FieldElement::<F>::from(13),
    ];
    let value_col_2 = vec![
        FieldElement::<F>::from(150),
        FieldElement::<F>::from(120),
        FieldElement::<F>::from(170),
        FieldElement::<F>::from(100),
        FieldElement::<F>::from(140),
        FieldElement::<F>::from(110),
        FieldElement::<F>::from(160),
        FieldElement::<F>::from(130),
    ];
    let pub_inputs_1 = LogReadOnlyPublicInputs {
        a0: FieldElement::<F>::from(3),
        v0: FieldElement::<F>::from(30),
        a_sorted_0: FieldElement::<F>::from(1),
        v_sorted_0: FieldElement::<F>::from(10),
        m0: FieldElement::<F>::from(1),
    };
    let pub_inputs_2 = LogReadOnlyPublicInputs {
        a0: FieldElement::<F>::from(15),
        v0: FieldElement::<F>::from(150),
        a_sorted_0: FieldElement::<F>::from(10),
        v_sorted_0: FieldElement::<F>::from(100),
        m0: FieldElement::<F>::from(1),
    };

    let mut trace_1 = read_only_logup_trace(address_col_1, value_col_1);
    let mut trace_2 = read_only_logup_trace(address_col_2, value_col_2);
    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = LogReadOnlyPublicInputs<F>>,
        &mut _,
        &_,
    )> = vec![
        (&air_1, &mut trace_1, &pub_inputs_1),
        (&air_2, &mut trace_2, &pub_inputs_2),
    ];
    let proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).expect("prove");
    (air_1, air_2, proof)
}

fn verify(
    airs: &[&dyn AIR<Field = F, FieldExtension = E, PublicInputs = LogReadOnlyPublicInputs<F>>],
    proof: &MultiProof<F, E, LogReadOnlyPublicInputs<F>>,
) -> bool {
    multi_verify_ram(airs, proof, &mut DefaultTranscript::<E>::new(&[]), &FieldElement::zero())
}

fn first_aux_mmcs_opening_mut(
    proof: &mut MultiProof<F, E, LogReadOnlyPublicInputs<F>>,
) -> &mut AuxTraceOpening<E> {
    proof.proofs[0].deep_poly_openings[0]
        .aux_trace_polys
        .as_mut()
        .expect("baseline must have aux openings")
}

/// First chunk index whose aux MMCS root is `Some`.
fn first_populated_aux_chunk(proof: &MultiProof<F, E, LogReadOnlyPublicInputs<F>>) -> usize {
    proof
        .aux_mmcs_roots
        .iter()
        .position(|r| r.is_some())
        .expect("at least one chunk must have an aux MMCS root in this baseline")
}

#[test_log::test]
fn baseline_two_rap_tables_verify() {
    let (air_1, air_2, proof) = baseline_proof();
    assert!(
        proof.aux_mmcs_roots.iter().any(|r| r.is_some()),
        "at least one chunk's aux MMCS must be present"
    );
    assert!(
        proof
            .aux_mmcs_specs
            .iter()
            .map(|s| s.len())
            .sum::<usize>()
            == 2,
        "both AIRs contribute aux"
    );
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = LogReadOnlyPublicInputs<F>>> =
        vec![&air_1, &air_2];
    assert!(verify(&airs, &proof), "baseline aux proof must verify");
}

#[test_log::test]
fn tampered_aux_mmcs_root_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = LogReadOnlyPublicInputs<F>>> =
        vec![&air_1, &air_2];
    let chunk_idx = first_populated_aux_chunk(&proof);
    let root = proof.aux_mmcs_roots[chunk_idx]
        .as_mut()
        .expect("populated");
    root[0] ^= 1;
    assert!(!verify(&airs, &proof));
}

#[test_log::test]
fn missing_aux_mmcs_root_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = LogReadOnlyPublicInputs<F>>> =
        vec![&air_1, &air_2];
    let chunk_idx = first_populated_aux_chunk(&proof);
    proof.aux_mmcs_roots[chunk_idx] = None;
    assert!(
        !verify(&airs, &proof),
        "aux_mmcs_root=None while chunk has aux tables must be rejected"
    );
}

#[test_log::test]
fn tampered_aux_mmcs_spec_height_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = LogReadOnlyPublicInputs<F>>> =
        vec![&air_1, &air_2];
    let chunk_idx = first_populated_aux_chunk(&proof);
    proof.aux_mmcs_specs[chunk_idx][0].1 /= 2;
    assert!(!verify(&airs, &proof));
}

#[test_log::test]
fn tampered_aux_mmcs_spec_tag_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = LogReadOnlyPublicInputs<F>>> =
        vec![&air_1, &air_2];
    let chunk_idx = first_populated_aux_chunk(&proof);
    proof.aux_mmcs_specs[chunk_idx][0].0 = MatrixTag::new([0xFF; 8]);
    assert!(!verify(&airs, &proof));
}

#[test_log::test]
fn tampered_aux_mmcs_opening_leaf_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = LogReadOnlyPublicInputs<F>>> =
        vec![&air_1, &air_2];
    let AuxTraceOpening::Mmcs { mmcs_opening, .. } = first_aux_mmcs_opening_mut(&mut proof);
    mmcs_opening.matrix_leaves[0].1[0] ^= 1;
    assert!(!verify(&airs, &proof));
}

#[test_log::test]
fn tampered_aux_mmcs_opening_global_index_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = LogReadOnlyPublicInputs<F>>> =
        vec![&air_1, &air_2];
    let AuxTraceOpening::Mmcs { mmcs_opening, .. } = first_aux_mmcs_opening_mut(&mut proof);
    mmcs_opening.global_index ^= 0b10;
    assert!(!verify(&airs, &proof));
}

#[test_log::test]
fn tampered_aux_mmcs_opening_sibling_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = LogReadOnlyPublicInputs<F>>> =
        vec![&air_1, &air_2];
    let AuxTraceOpening::Mmcs { mmcs_opening, .. } = first_aux_mmcs_opening_mut(&mut proof);
    assert!(!mmcs_opening.siblings.is_empty());
    mmcs_opening.siblings[0][0] ^= 1;
    assert!(!verify(&airs, &proof));
}

#[test_log::test]
fn tampered_aux_evaluations_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = LogReadOnlyPublicInputs<F>>> =
        vec![&air_1, &air_2];
    let AuxTraceOpening::Mmcs { evaluations, .. } = first_aux_mmcs_opening_mut(&mut proof);
    assert!(!evaluations.is_empty());
    evaluations[0] += FieldElement::<E>::one();
    assert!(!verify(&airs, &proof));
}
