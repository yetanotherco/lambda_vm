//! Negative tests for the row-pair trace opening verification
//! (`verifier::verify_opening_pair`). The row pair `(2·iota, 2·iota+1)` is
//! committed as a single Merkle leaf, so one `proof` authenticates both
//! `evaluations` and `evaluations_sym`. Removing the old separate `proof_sym`
//! opening deleted the "symmetric opening mismatch" rejection class; these
//! tests restore it — an implementation that ignored `evaluations_sym` or the
//! authentication path would otherwise pass every other test.

use super::small_trace_tests::make_valid_simple_proof;
use crate::verifier::{IsStarkVerifier, Verifier};
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::{element::FieldElement, goldilocks::GoldilocksField};

type Felt = FieldElement<GoldilocksField>;

/// Tampering the value at the symmetric LDE position must break verification:
/// the committed leaf hashed `evaluations ‖ evaluations_sym`, so a perturbed
/// `evaluations_sym` no longer reconstructs the committed leaf.
#[test_log::test]
fn test_verify_rejects_tampered_main_trace_evaluations_sym() {
    let (air, mut proof) = make_valid_simple_proof();

    let opening = proof
        .deep_poly_openings
        .first_mut()
        .expect("test precondition: a valid proof has at least one deep poly opening");
    assert!(
        !opening.main_trace_polys.evaluations_sym.is_empty(),
        "test precondition: the main-trace opening has at least one symmetric evaluation",
    );
    // Perturb (not resize) the first symmetric evaluation.
    opening.main_trace_polys.evaluations_sym[0] =
        &opening.main_trace_polys.evaluations_sym[0] + Felt::one();

    assert!(
        !Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verifier must reject a tampered symmetric trace evaluation"
    );
}

/// The row-pair Merkle authentication path itself must be checked. Corrupting a
/// node in `main_trace_polys.proof.merkle_path` is caught ONLY by
/// `verify_opening_pair` (the deep-composition reconstruction does not touch the
/// auth path), so this proves the single row-pair path is actually authenticated
/// against the committed root rather than ignored.
#[test_log::test]
fn test_verify_rejects_tampered_main_trace_merkle_path() {
    let (air, mut proof) = make_valid_simple_proof();

    let opening = proof
        .deep_poly_openings
        .first_mut()
        .expect("test precondition: a valid proof has at least one deep poly opening");
    let path = &mut opening.main_trace_polys.proof.merkle_path;
    assert!(
        !path.is_empty(),
        "test precondition: the row-pair trace tree has a non-trivial authentication path",
    );
    path[0][0] ^= 0x01;

    assert!(
        !Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<GoldilocksField>::new(&[])
        ),
        "Verifier must reject a corrupted main-trace Merkle authentication path"
    );
}
