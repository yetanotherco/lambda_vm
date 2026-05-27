//! Soundness tests for the shared main-trace MMCS path.
//!
//! All tests use a multi-table proof over non-preprocessed AIRs (so every
//! table's main slice lives in `MainTraceOpening::Mmcs`). The preprocessed
//! per-table-tree path is exercised end-to-end by lambda-vm-prover's
//! `bitwise_tests` (the bitwise AIR is preprocessed).
//!
//! Each test starts from a baseline-valid multi-proof, tampers with a
//! single field on the MMCS path, and asserts the verifier rejects.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::merkle_tree::mmcs::MatrixTag;
use math::field::{element::FieldElement, goldilocks::GoldilocksField};

use crate::examples::{
    bit_flags::{self, BitFlagsAIR},
    dummy_air::{self, DummyAIR},
};
use crate::proof::options::ProofOptions;
use crate::proof::stark::{MainTraceOpening, MultiProof};
use crate::test_utils::{multi_prove_ram, multi_verify_ram, synth_main_tags};
use crate::traits::AIR;

type F = GoldilocksField;

/// Build a baseline multi-proof over (DummyAIR, BitFlagsAIR). Both are
/// non-preprocessed → every main opening is `MainTraceOpening::Mmcs`.
#[allow(clippy::type_complexity)]
fn baseline_proof() -> (
    DummyAIR,
    BitFlagsAIR,
    MultiProof<F, F, ()>,
) {
    let proof_options = ProofOptions::default_test_options();
    let air_1 = DummyAIR::new(&proof_options);
    let air_2 = BitFlagsAIR::new(&proof_options);
    let mut trace_1 = dummy_air::dummy_trace::<F>(16);
    let mut trace_2 = bit_flags::bit_prefix_flag_trace(32);
    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>,
        &mut _,
        &_,
    )> = vec![
        (&air_1, &mut trace_1, &()),
        (&air_2, &mut trace_2, &()),
    ];
    let proof =
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<F>::new(&[])).unwrap();
    (air_1, air_2, proof)
}

fn verify(airs: &[&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>], proof: &MultiProof<F, F, ()>) -> bool {
    multi_verify_ram(
        airs,
        proof,
        &mut DefaultTranscript::<F>::new(&[]),
        &FieldElement::zero(),
    )
}

/// First-iota opening for the first table in the multi-proof, in the Mmcs
/// variant. Helper for tests that need a mutable handle into the per-query
/// MMCS opening fields.
fn first_mmcs_opening_mut(
    proof: &mut MultiProof<F, F, ()>,
) -> &mut MainTraceOpening<F> {
    &mut proof.proofs[0].deep_poly_openings[0].main_trace_polys
}

#[test_log::test]
fn baseline_two_table_proof_verifies() {
    let (air_1, air_2, proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    assert!(verify(&airs, &proof), "baseline proof must verify");
}

#[test_log::test]
fn tampered_main_mmcs_root_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    proof.main_mmcs_root[0] ^= 1;
    assert!(
        !verify(&airs, &proof),
        "tampered main MMCS root must be rejected"
    );
}

#[test_log::test]
fn tampered_main_mmcs_spec_height_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    let height = &mut proof.main_mmcs_spec[0].1;
    *height /= 2;
    assert!(
        !verify(&airs, &proof),
        "spec height mismatch must be rejected"
    );
}

#[test_log::test]
fn tampered_main_mmcs_spec_tag_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    proof.main_mmcs_spec[0].0 = MatrixTag::new([0xFF; 8]);
    assert!(
        !verify(&airs, &proof),
        "spec tag mismatch must be rejected"
    );
}

#[test_log::test]
fn tampered_mmcs_opening_leaf_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    match first_mmcs_opening_mut(&mut proof) {
        MainTraceOpening::Mmcs { mmcs_opening, .. } => {
            mmcs_opening.matrix_leaves[0].1[0] ^= 1;
        }
        MainTraceOpening::Tree(_) => panic!("baseline must produce Mmcs variant"),
    }
    assert!(
        !verify(&airs, &proof),
        "tampered matrix-leaf digest must be rejected"
    );
}

#[test_log::test]
fn tampered_mmcs_opening_leaf_tag_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    match first_mmcs_opening_mut(&mut proof) {
        MainTraceOpening::Mmcs { mmcs_opening, .. } => {
            mmcs_opening.matrix_leaves[0].0 = MatrixTag::new([0xCC; 8]);
        }
        MainTraceOpening::Tree(_) => panic!("baseline must produce Mmcs variant"),
    }
    assert!(
        !verify(&airs, &proof),
        "tampered matrix-leaf tag must be rejected"
    );
}

#[test_log::test]
fn tampered_mmcs_opening_global_index_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    match first_mmcs_opening_mut(&mut proof) {
        MainTraceOpening::Mmcs { mmcs_opening, .. } => {
            mmcs_opening.global_index ^= 0b10;
        }
        MainTraceOpening::Tree(_) => panic!("baseline must produce Mmcs variant"),
    }
    assert!(
        !verify(&airs, &proof),
        "tampered MMCS global_index must be rejected"
    );
}

#[test_log::test]
fn tampered_mmcs_opening_sibling_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    match first_mmcs_opening_mut(&mut proof) {
        MainTraceOpening::Mmcs { mmcs_opening, .. } => {
            assert!(!mmcs_opening.siblings.is_empty());
            mmcs_opening.siblings[0][0] ^= 1;
        }
        MainTraceOpening::Tree(_) => panic!("baseline must produce Mmcs variant"),
    }
    assert!(
        !verify(&airs, &proof),
        "tampered MMCS sibling must be rejected"
    );
}

#[test_log::test]
fn tampered_evaluations_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    match first_mmcs_opening_mut(&mut proof) {
        MainTraceOpening::Mmcs { evaluations, .. } => {
            assert!(!evaluations.is_empty());
            evaluations[0] += FieldElement::<F>::one();
        }
        MainTraceOpening::Tree(_) => panic!("baseline must produce Mmcs variant"),
    }
    assert!(
        !verify(&airs, &proof),
        "tampered row evaluations must be rejected (rehash mismatch)"
    );
}

#[test_log::test]
fn swapped_main_tags_at_verifier_rejected() {
    // The verifier reproduces `main_tags` from `synth_main_tags(num_airs)`
    // inside `multi_verify_ram`. To simulate a verifier that "lies" about
    // tag ordering we call `multi_verify` directly with a permuted slice.
    use crate::verifier::{IsStarkVerifier, Verifier};
    let (air_1, air_2, proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];

    // Sanity: with the correct (synth) tag order it passes.
    let correct = synth_main_tags(airs.len());
    assert!(
        Verifier::multi_verify(
            &airs,
            &proof,
            &correct,
            &mut DefaultTranscript::<F>::new(&[]),
            &FieldElement::zero(),
        ),
        "baseline must verify with correct tags"
    );

    // Swap the two tags — the spec sort order is now wrong relative to the
    // prover's commitments, so the spec match check must reject.
    let mut swapped = correct.clone();
    swapped.swap(0, 1);
    assert!(
        !Verifier::multi_verify(
            &airs,
            &proof,
            &swapped,
            &mut DefaultTranscript::<F>::new(&[]),
            &FieldElement::zero(),
        ),
        "swapped main_tags must be rejected"
    );
}
