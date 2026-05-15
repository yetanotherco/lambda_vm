//! Phase 2 tests for `proof::quotient_chunks::QuotientChunksCommitments`.
//!
//! Validates:
//!   1. serde CBOR round-trip preserves the structure,
//!   2. honest chunk openings produced by the prover kernel verify against
//!      the directly-evaluated `H(z)`,
//!   3. tampering any chunk OOD evaluation or chunk root makes verification
//!      fail.
//!
//! The verifier-side path (`StarkProof` + `IsStarkVerifier::verify`) is not
//! changed in this phase — only the standalone chunks data shape and its
//! consistency check at the out-of-domain point are exercised.

use math::{
    field::{element::FieldElement, goldilocks::GoldilocksField, traits::IsFFTField},
    polynomial::Polynomial,
};

use crate::{
    domain::{Domain, QuotientDomain},
    proof::quotient_chunks::QuotientChunksCommitments,
    prover::{IsStarkProver, Prover},
};

type Felt = FieldElement<GoldilocksField>;

/// Build a minimal `Domain` for tests without going through an AIR.
/// Same pattern as the Phase 1.2/1.3/1.4 tests in `prover_tests.rs`.
fn make_test_domain(trace_length: usize, blowup_factor: usize, coset_offset: Felt) -> Domain<GoldilocksField> {
    let root_order = trace_length.trailing_zeros();
    let trace_primitive_root =
        GoldilocksField::get_primitive_root_of_unity(root_order as u64).unwrap();
    let lde_root_order = (trace_length * blowup_factor).trailing_zeros();
    let lde_roots_of_unity_coset =
        math::fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset(
            lde_root_order as u64,
            trace_length * blowup_factor,
            &coset_offset,
        )
        .unwrap();
    let trace_roots_of_unity = math::fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset(
        root_order as u64,
        trace_length,
        &Felt::one(),
    )
    .unwrap();
    Domain::<GoldilocksField> {
        root_order,
        lde_roots_of_unity_coset,
        trace_primitive_root,
        trace_roots_of_unity,
        coset_offset,
        blowup_factor,
        interpolation_domain_size: trace_length,
    }
}

/// Produce honest `QuotientChunksCommitments` for a known polynomial `H` and
/// out-of-domain point `z`. Returns the commitments together with the
/// ground-truth `H(z)` so callers can verify or tamper.
fn build_honest_chunks(
    d_max: usize,
    z: &Felt,
) -> (
    QuotientDomain<GoldilocksField>,
    QuotientChunksCommitments<GoldilocksField>,
    Felt,
) {
    let trace_length: usize = 8;
    let blowup_factor: usize = 2;
    let coset_offset = Felt::from(3u64);
    let domain = make_test_domain(trace_length, blowup_factor, coset_offset);
    let qd = QuotientDomain::new(&domain, d_max);

    let h_coeffs: Vec<Felt> = (0..qd.size).map(|i| Felt::from((11 * i + 17) as u64)).collect();
    let h_poly = Polynomial::new(&h_coeffs);
    let h_evals: Vec<Felt> = (0..qd.size).map(|i| h_poly.evaluate(qd.point_at(i))).collect();
    let chunks = qd.split_evals_interleaved(&h_evals);

    let results = Prover::<GoldilocksField, GoldilocksField, ()>::lde_and_commit_quotient_chunks(
        &qd, &domain, &chunks,
    );
    let chunk_roots: Vec<_> = results.iter().map(|(_, _, r)| r.clone()).collect();

    let chunk_ood_evaluations: Vec<Felt> = chunks
        .iter()
        .enumerate()
        .map(|(i, chunk)| {
            let (sub_offset, _) = qd.chunk_subdomain(i);
            let q_i =
                Polynomial::interpolate_offset_fft::<GoldilocksField>(chunk, &sub_offset).unwrap();
            q_i.evaluate(z)
        })
        .collect();

    let h_at_z = h_poly.evaluate(z);
    (
        qd,
        QuotientChunksCommitments {
            chunk_roots,
            chunk_ood_evaluations,
        },
        h_at_z,
    )
}

#[test]
fn serde_cbor_round_trip() {
    let z = Felt::from(12345u64);
    let (_qd, commitments, _h_at_z) = build_honest_chunks(3, &z);

    let bytes = serde_cbor::to_vec(&commitments).expect("CBOR serialize should succeed");
    let decoded: QuotientChunksCommitments<GoldilocksField> =
        serde_cbor::from_slice(&bytes).expect("CBOR deserialize should succeed");

    assert_eq!(decoded, commitments);
}

#[test]
fn verify_at_ood_accepts_honest_chunks() {
    for &d_max in &[1usize, 2, 3] {
        let z = Felt::from(98765u64);
        let (qd, commitments, h_at_z) = build_honest_chunks(d_max, &z);
        assert!(
            commitments.verify_at_ood(&qd, &z, &h_at_z),
            "d_max={d_max}: honest chunk openings should verify at z",
        );
    }
}

#[test]
fn verify_at_ood_rejects_tampered_chunk_ood_eval() {
    for &d_max in &[1usize, 2, 3] {
        let z = Felt::from(98765u64);
        let (qd, mut commitments, h_at_z) = build_honest_chunks(d_max, &z);

        // Flip the last chunk's claimed OOD evaluation by adding one.
        let last = commitments.chunk_ood_evaluations.len() - 1;
        commitments.chunk_ood_evaluations[last] =
            &commitments.chunk_ood_evaluations[last] + Felt::one();

        assert!(
            !commitments.verify_at_ood(&qd, &z, &h_at_z),
            "d_max={d_max}: tampered chunk OOD evaluation must not verify",
        );
    }
}

#[test]
fn verify_at_ood_rejects_wrong_expected_h_z() {
    let z = Felt::from(98765u64);
    let (qd, commitments, h_at_z) = build_honest_chunks(2, &z);
    let bad_h_at_z = &h_at_z + Felt::one();
    assert!(
        !commitments.verify_at_ood(&qd, &z, &bad_h_at_z),
        "verify_at_ood should reject when the verifier's expected H(z) is wrong",
    );
}

#[test]
fn verify_at_ood_rejects_length_mismatch() {
    let z = Felt::from(98765u64);
    let (qd, mut commitments, h_at_z) = build_honest_chunks(3, &z);

    // Drop one chunk OOD evaluation — length now disagrees with quotient_domain.num_chunks.
    commitments.chunk_ood_evaluations.pop();
    assert!(
        !commitments.verify_at_ood(&qd, &z, &h_at_z),
        "verify_at_ood should reject when chunk_ood_evaluations.len() != num_chunks",
    );
}
