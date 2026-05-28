//! Phase D — per-(chunk, lde_size) batched FRI soundness tests.
//!
//! Every test starts from a baseline-valid multi-proof, then tampers
//! with a single field on the bucket-FRI path inside `MultiProof::
//! fri_chunk_buckets` and asserts the verifier rejects. Pre-existing
//! main / aux / composition MMCS path soundness is covered by
//! `mmcs_soundness_tests`, `mmcs_aux_soundness_tests`, and the
//! composition tests inside `mmcs_soundness_tests`.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::{element::FieldElement, goldilocks::GoldilocksField};

use crate::examples::{
    bit_flags::{self, BitFlagsAIR},
    dummy_air::{self, DummyAIR},
};
use crate::proof::options::ProofOptions;
use crate::proof::stark::MultiProof;
use crate::test_utils::{multi_prove_ram, multi_verify_ram};
use crate::traits::AIR;

type F = GoldilocksField;

#[allow(clippy::type_complexity)]
fn baseline_proof() -> (DummyAIR, BitFlagsAIR, MultiProof<F, F, ()>) {
    let proof_options = ProofOptions::default_test_options();
    let air_1 = DummyAIR::new(&proof_options);
    let air_2 = BitFlagsAIR::new(&proof_options);
    let mut trace_1 = dummy_air::dummy_trace::<F>(16);
    let mut trace_2 = bit_flags::bit_prefix_flag_trace(32);
    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>,
        &mut _,
        &_,
    )> = vec![(&air_1, &mut trace_1, &()), (&air_2, &mut trace_2, &())];
    let proof = multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<F>::new(&[])).unwrap();
    (air_1, air_2, proof)
}

fn verify(
    airs: &[&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>],
    proof: &MultiProof<F, F, ()>,
) -> bool {
    multi_verify_ram(
        airs,
        proof,
        &mut DefaultTranscript::<F>::new(&[]),
        &FieldElement::zero(),
    )
}

/// Locate the first chunk whose `fri_chunk_buckets` is non-empty and the
/// first bucket inside it. Used by tampering tests to find a real bucket
/// to mutate.
fn first_bucket_mut(
    proof: &mut MultiProof<F, F, ()>,
) -> (usize, usize) {
    let chunk_idx = proof
        .fri_chunk_buckets
        .iter()
        .position(|c| !c.is_empty())
        .expect("baseline has at least one non-empty fri_chunk_buckets entry");
    (chunk_idx, 0)
}

#[test_log::test]
fn baseline_phase_d_proof_verifies() {
    let (air_1, air_2, proof) = baseline_proof();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    assert!(verify(&airs, &proof), "baseline Phase D proof must verify");
    // Sanity: fri_chunk_buckets is parallel to per-chunk MMCS vecs.
    assert_eq!(proof.fri_chunk_buckets.len(), proof.main_mmcs_roots.len());
    // Every populated bucket must have non-empty members + at least one
    // decommitment per fri query.
    for chunk in &proof.fri_chunk_buckets {
        for bucket in chunk {
            assert!(!bucket.members.is_empty());
            assert!(!bucket.decommitments.is_empty());
        }
    }
}

#[test_log::test]
fn tampered_bucket_last_value_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let (ci, bi) = first_bucket_mut(&mut proof);
    proof.fri_chunk_buckets[ci][bi].last_value =
        &proof.fri_chunk_buckets[ci][bi].last_value + FieldElement::<F>::one();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    assert!(!verify(&airs, &proof));
}

#[test_log::test]
fn tampered_bucket_layer_root_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let (ci, bi) = first_bucket_mut(&mut proof);
    if proof.fri_chunk_buckets[ci][bi].layer_roots.is_empty() {
        // Trivially-small LDE: no committed FRI layers to tamper with;
        // tampering last_value above already covers that case.
        return;
    }
    proof.fri_chunk_buckets[ci][bi].layer_roots[0][0] ^= 0xFF;
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    assert!(!verify(&airs, &proof));
}

#[test_log::test]
fn truncated_bucket_decommitments_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let (ci, bi) = first_bucket_mut(&mut proof);
    assert!(!proof.fri_chunk_buckets[ci][bi].decommitments.is_empty());
    proof.fri_chunk_buckets[ci][bi].decommitments.pop();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    assert!(!verify(&airs, &proof));
}

#[test_log::test]
fn missing_chunk_buckets_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    // Wipe a chunk's bucket list; verifier checks bucket count matches
    // the lde-size grouping expected from the AIRs in the chunk.
    let (ci, _) = first_bucket_mut(&mut proof);
    proof.fri_chunk_buckets[ci].clear();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    assert!(!verify(&airs, &proof));
}

#[test_log::test]
fn wrong_bucket_lde_size_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    let (ci, bi) = first_bucket_mut(&mut proof);
    let actual = proof.fri_chunk_buckets[ci][bi].lde_size;
    // Bump to a different power of two — verifier reconstructs expected
    // lde_size from per-AIR blowup × trace_length and rejects mismatch.
    proof.fri_chunk_buckets[ci][bi].lde_size = actual.wrapping_mul(2);
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    assert!(!verify(&airs, &proof));
}

#[test_log::test]
fn swapped_member_order_rejected() {
    let (air_1, air_2, mut proof) = baseline_proof();
    // Find a bucket with ≥ 2 members and swap their order. The verifier
    // requires bucket members in canonical chunk-local-index order so a
    // tag swap shifts δ_fri^i powers and rejects the combined FRI.
    let target = proof
        .fri_chunk_buckets
        .iter_mut()
        .enumerate()
        .find_map(|(ci, c)| c.iter_mut().enumerate().find_map(|(bi, b)| {
            if b.members.len() >= 2 { Some((ci, bi)) } else { None }
        }));
    let Some((ci, bi)) = target else {
        // Single-table-per-bucket baseline — swap is not applicable; in
        // practice every chunk-mate becomes its own singleton bucket here.
        return;
    };
    proof.fri_chunk_buckets[ci][bi].members.swap(0, 1);
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = F, PublicInputs = ()>> =
        vec![&air_1, &air_2];
    assert!(!verify(&airs, &proof));
}
