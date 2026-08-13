//! `ResidencyMode` equivalence: dropping each main LDE after Round 1 and
//! recomputing it inside the table's fused task changes nothing a verifier can
//! see.
//!
//! The oracle is the whole proof, not just the roots. Comparing serialized
//! proof bytes is normally avoided — a committed golden blob turns every
//! legitimate format change into a test failure — but there is no golden blob
//! here: both sides are produced in this process from the same traces, and the
//! only difference between them is the mode. That makes byte equality the
//! sharpest available statement of "invisible to the proof", and it is the same
//! oracle the closed streaming-prover PR (#647) used for the same change.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::proof::options::ProofOptions;
use crate::proof::stark::MultiProof;
use crate::prover::{IsStarkProver, Prover};
use crate::residency_mode::ResidencyMode;
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

/// The bus-balanced CPU/ADD/MUL instance from the completeness tests. Rebuilt
/// per prove because `multi_prove` writes the LogUp aux columns into the caller's
/// traces — and under `RecomputeLde` frees them again.
fn traces() -> (TraceTable<F, E>, TraceTable<F, E>, TraceTable<F, E>) {
    let cpu = TraceTable::from_columns_main(
        vec![
            vec![
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::one(),
                FE::zero(),
                FE::zero(),
            ],
            vec![
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::zero(),
                FE::one(),
                FE::one(),
            ],
            (1..=8).map(FE::from).collect(),
            (1..=8).map(|i| FE::from(i * 10)).collect(),
            vec![
                FE::from(11),
                FE::from(40),
                FE::from(33),
                FE::from(160),
                FE::from(55),
                FE::from(66),
                FE::from(490),
                FE::from(640),
            ],
        ],
        1,
    );
    let add = TraceTable::from_columns_main(
        vec![
            vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)],
            vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)],
            vec![FE::from(11), FE::from(33), FE::from(55), FE::from(66)],
            vec![FE::one(); 4],
        ],
        1,
    );
    let mul = TraceTable::from_columns_main(
        vec![
            vec![FE::from(2), FE::from(4), FE::from(7), FE::from(8)],
            vec![FE::from(20), FE::from(40), FE::from(70), FE::from(80)],
            vec![FE::from(40), FE::from(160), FE::from(490), FE::from(640)],
            vec![FE::one(); 4],
        ],
        1,
    );
    (cpu, add, mul)
}

fn prove_under(residency: ResidencyMode) -> MultiProof<F, E, ()> {
    let (mut cpu_trace, mut add_trace, mut mul_trace) = traces();
    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);

    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];

    Prover::multi_prove(
        pairs,
        &mut DefaultTranscript::<E>::new(&[]),
        #[cfg(feature = "disk-spill")]
        crate::storage_mode::StorageMode::Ram,
        residency,
    )
    .unwrap()
}

fn verifies(proof: &MultiProof<F, E, ()>) -> bool {
    let proof_options = ProofOptions::default_test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];
    Verifier::multi_verify(
        &airs,
        proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

/// Every commitment root is unchanged. This is the load-bearing half: the root
/// that entered the transcript at Round 1 is the root the recomputed LDE's
/// openings are checked against, so if the recompute produced different values
/// the openings would be answered against a tree that no longer matches them.
#[test_log::test]
fn recompute_lde_preserves_every_commitment_root() {
    let retained = prove_under(ResidencyMode::Retain);
    let recomputed = prove_under(ResidencyMode::RecomputeLde);

    assert_eq!(retained.proofs.len(), recomputed.proofs.len());
    for (idx, (a, b)) in retained
        .proofs
        .iter()
        .zip(recomputed.proofs.iter())
        .enumerate()
    {
        assert_eq!(
            a.lde_trace_main_merkle_root, b.lde_trace_main_merkle_root,
            "table {idx}: main root moved"
        );
        assert_eq!(
            a.lde_trace_aux_merkle_root, b.lde_trace_aux_merkle_root,
            "table {idx}: aux root moved"
        );
        assert_eq!(
            a.lde_trace_precomputed_merkle_root, b.lde_trace_precomputed_merkle_root,
            "table {idx}: precomputed root moved"
        );
        assert_eq!(
            a.composition_poly_root, b.composition_poly_root,
            "table {idx}: composition root moved"
        );
        assert_eq!(
            a.fri_layers_merkle_roots, b.fri_layers_merkle_roots,
            "table {idx}: FRI roots moved"
        );
    }
}

/// The whole proof, byte for byte — openings, FRI decommitments, grinding nonce
/// and all.
#[test_log::test]
fn recompute_lde_produces_byte_identical_proofs() {
    let retained = bincode::serialize(&prove_under(ResidencyMode::Retain)).unwrap();
    let recomputed = bincode::serialize(&prove_under(ResidencyMode::RecomputeLde)).unwrap();
    assert_eq!(
        retained.len(),
        recomputed.len(),
        "proof size moved between residency modes"
    );
    assert!(
        retained == recomputed,
        "proof bytes moved between residency modes"
    );
}

/// A proof made under `RecomputeLde` verifies with the standard verifier. Near
/// tautological given the byte equality above — and that is the point: the mode
/// has no wire presence for a verifier to know about.
#[test_log::test]
fn recompute_lde_proofs_verify() {
    assert!(verifies(&prove_under(ResidencyMode::RecomputeLde)));
    assert!(verifies(&prove_under(ResidencyMode::Retain)));
}

/// The caller-visible half of the `RecomputeLde` contract: the aux columns
/// `multi_prove` wrote into the caller's traces are gone when it returns, and
/// under `Retain` they are still there. Pins the documented difference so a
/// caller that needs the aux columns after proving finds out here.
#[test_log::test]
fn recompute_lde_releases_aux_columns_and_retain_keeps_them() {
    for (residency, expect_aux_rows) in [
        (ResidencyMode::Retain, true),
        (ResidencyMode::RecomputeLde, false),
    ] {
        let (mut cpu_trace, mut add_trace, mut mul_trace) = traces();
        let proof_options = ProofOptions::default_test_options();
        let cpu_air = new_cpu_air_with_lookup(&proof_options);
        let add_air = new_add_air_with_lookup(&proof_options);
        let mul_air = new_mul_air_with_lookup(&proof_options);

        {
            let pairs: Vec<(
                &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
                _,
                _,
            )> = vec![
                (&cpu_air, &mut cpu_trace, &()),
                (&add_air, &mut add_trace, &()),
                (&mul_air, &mut mul_trace, &()),
            ];
            Prover::multi_prove(
                pairs,
                &mut DefaultTranscript::<E>::new(&[]),
                #[cfg(feature = "disk-spill")]
                crate::storage_mode::StorageMode::Ram,
                residency,
            )
            .unwrap();
        }

        for (name, trace) in [
            ("cpu", &cpu_trace),
            ("add", &add_trace),
            ("mul", &mul_trace),
        ] {
            let has_rows = trace.aux_table.height > 0;
            assert_eq!(
                has_rows, expect_aux_rows,
                "{name} trace aux residency wrong under {residency:?}"
            );
            // The declared width survives either way — only the data is freed.
            assert!(trace.aux_table.width > 0, "{name} lost its aux width");
        }
    }
}
