//! LogUp-GKR mode: end-to-end completeness and soundness tests.
//!
//! The fixture mirrors `completeness_tests::test_multi_table_proof` (a CPU
//! table dispatching to ADD and MUL over buses), with every AIR switched to
//! [`LogUpMode::Gkr`]: proofs carry a batch GKR proof + column claims instead
//! of bus public inputs, and the verifier replays the batch GKR before the
//! per-table rounds.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use crate::constraints::builder::EmptyConstraints;
use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::lookup::{AirWithBuses, LogUpMode, NullBoundaryConstraintBuilder};
use crate::proof::options::ProofOptions;
use crate::proof::stark::GkrMultiProof;
use crate::test_utils::multi_prove_gkr_ram;
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;
type GkrAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints>;

fn gkr_airs() -> (GkrAir, GkrAir, GkrAir) {
    let proof_options = ProofOptions::default_test_options();
    (
        new_cpu_air_with_lookup(&proof_options).with_logup_mode(LogUpMode::Gkr),
        new_add_air_with_lookup(&proof_options).with_logup_mode(LogUpMode::Gkr),
        new_mul_air_with_lookup(&proof_options).with_logup_mode(LogUpMode::Gkr),
    )
}

fn fixture_traces() -> (TraceTable<F, E>, TraceTable<F, E>, TraceTable<F, E>) {
    // CPU (8 rows) dispatches 4 additions and 4 multiplications; ADD and MUL
    // (4 rows each) receive them with multiplicity 1. Mixed trace lengths
    // exercise the batch GKR's staggered instance activation.
    let cpu_trace = TraceTable::from_columns_main(
        vec![
            // add_flag
            [1u64, 0, 1, 0, 1, 1, 0, 0]
                .iter()
                .map(|&v| FE::from(v))
                .collect(),
            // mul_flag
            [0u64, 1, 0, 1, 0, 0, 1, 1]
                .iter()
                .map(|&v| FE::from(v))
                .collect(),
            (1..=8).map(|v| FE::from(v as u64)).collect(),
            (1..=8).map(|v| FE::from(10 * v as u64)).collect(),
            [11u64, 40, 33, 160, 55, 66, 490, 640]
                .iter()
                .map(|&v| FE::from(v))
                .collect(),
        ],
        1,
    );
    let add_trace = TraceTable::from_columns_main(
        vec![
            [1u64, 3, 5, 6].iter().map(|&v| FE::from(v)).collect(),
            [10u64, 30, 50, 60].iter().map(|&v| FE::from(v)).collect(),
            [11u64, 33, 55, 66].iter().map(|&v| FE::from(v)).collect(),
            vec![FE::one(); 4],
        ],
        1,
    );
    let mul_trace = TraceTable::from_columns_main(
        vec![
            [2u64, 4, 7, 8].iter().map(|&v| FE::from(v)).collect(),
            [20u64, 40, 70, 80].iter().map(|&v| FE::from(v)).collect(),
            [40u64, 160, 490, 640]
                .iter()
                .map(|&v| FE::from(v))
                .collect(),
            vec![FE::one(); 4],
        ],
        1,
    );
    (cpu_trace, add_trace, mul_trace)
}

/// Prove the fixture in GKR mode.
fn prove_fixture() -> (GkrAir, GkrAir, GkrAir, GkrMultiProof<F, E, ()>) {
    let (cpu_air, add_air, mul_air) = gkr_airs();
    let (mut cpu_trace, mut add_trace, mut mul_trace) = fixture_traces();
    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];
    let proof = multi_prove_gkr_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[]))
        .expect("GKR prove succeeds");
    (cpu_air, add_air, mul_air, proof)
}

fn verify(
    cpu_air: &GkrAir,
    add_air: &GkrAir,
    mul_air: &GkrAir,
    proof: &GkrMultiProof<F, E, ()>,
) -> bool {
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![cpu_air, add_air, mul_air];
    Verifier::multi_verify_gkr(
        &airs,
        proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

/// Valid GKR-mode multi-table proof is accepted.
#[test_log::test]
fn test_gkr_multi_table_prove_verify() {
    let (cpu_air, add_air, mul_air, proof) = prove_fixture();
    assert_eq!(proof.multi.proofs.len(), 3);
    // GKR proofs carry no bus public inputs; the wrapper carries claims for
    // every (interacting) table.
    assert!(
        proof
            .multi
            .proofs
            .iter()
            .all(|p| p.bus_public_inputs.is_none())
    );
    assert!(proof.column_claims_by_table.iter().all(|c| c.is_some()));
    assert!(verify(&cpu_air, &add_air, &mul_air, &proof));
}

/// The GKR wrapper survives an rkyv roundtrip and still verifies.
#[test_log::test]
fn test_gkr_proof_rkyv_roundtrip() {
    let (cpu_air, add_air, mul_air, proof) = prove_fixture();
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).unwrap();
    let deserialized: GkrMultiProof<F, E, ()> =
        rkyv::from_bytes::<_, rkyv::rancor::Error>(&bytes).unwrap();
    assert!(verify(&cpu_air, &add_air, &mul_air, &deserialized));
}

/// A single-table GKR proof roundtrips (the batch degenerates to one instance).
#[test_log::test]
fn test_gkr_single_table_prove_verify() {
    // A self-balancing table: one sender and one receiver interaction over
    // the same bus and columns cancel row-by-row.
    let proof_options = ProofOptions::default_test_options();
    let air = {
        use crate::lookup::{AuxiliaryTraceBuildData, BusInteraction, Multiplicity, Packing};
        AirWithBuses::<F, E, NullBoundaryConstraintBuilder, (), _>::new(
            2,
            AuxiliaryTraceBuildData {
                interactions: vec![
                    BusInteraction::sender(1u64, Multiplicity::One, Packing::Direct.columns(&[0])),
                    BusInteraction::receiver(
                        1u64,
                        Multiplicity::One,
                        Packing::Direct.columns(&[0]),
                    ),
                ],
            },
            &proof_options,
            1,
            EmptyConstraints,
        )
        .with_logup_mode(LogUpMode::Gkr)
    };
    let mut trace = TraceTable::from_columns_main(
        vec![
            (1..=8).map(|v| FE::from(v as u64)).collect(),
            (11..=18).map(|v| FE::from(v as u64)).collect(),
        ],
        1,
    );
    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![(&air, &mut trace, &())];
    let proof =
        multi_prove_gkr_ram(pairs, &mut DefaultTranscript::<E>::new(&[])).expect("prove succeeds");
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![&air];
    assert!(Verifier::multi_verify_gkr(
        &airs,
        &proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    ));
}

/// Tampering a batch-GKR root claim (the table contribution) is rejected —
/// this is the GKR analogue of tampering `table_contribution`.
#[test_log::test]
fn test_gkr_tampered_root_claim_rejected() {
    let (cpu_air, add_air, mul_air, mut proof) = prove_fixture();
    proof.batch_gkr_proof.root_claims[0].0 += FieldElement::<E>::one();
    assert!(!verify(&cpu_air, &add_air, &mul_air, &proof));
}

/// Tampering a column claim is rejected (the claims are transcript-bound
/// before γ is sampled, and the bridge pins them to the committed trace).
#[test_log::test]
fn test_gkr_tampered_column_claim_rejected() {
    let (cpu_air, add_air, mul_air, mut proof) = prove_fixture();
    let claims = proof.column_claims_by_table[1]
        .as_mut()
        .expect("ADD table has claims");
    claims[0].1 += FieldElement::<E>::one();
    assert!(!verify(&cpu_air, &add_air, &mul_air, &proof));
}

/// Dropping a column claim (or the whole table entry) is rejected by the
/// exact index-set check.
#[test_log::test]
fn test_gkr_missing_column_claims_rejected() {
    let (cpu_air, add_air, mul_air, proof) = prove_fixture();

    let mut truncated = proof.clone();
    truncated.column_claims_by_table[1]
        .as_mut()
        .expect("ADD table has claims")
        .pop();
    assert!(!verify(&cpu_air, &add_air, &mul_air, &truncated));

    let mut missing = proof;
    missing.column_claims_by_table[1] = None;
    assert!(!verify(&cpu_air, &add_air, &mul_air, &missing));
}

/// Tampering a layer proof's child claims breaks the transcript-bound GKR
/// replay and is rejected.
#[test_log::test]
fn test_gkr_tampered_child_claims_rejected() {
    let (cpu_air, add_air, mul_air, mut proof) = prove_fixture();
    let layer = proof
        .batch_gkr_proof
        .layer_proofs
        .last_mut()
        .expect("batch proof has layers");
    layer.child_claims_by_instance[0][0] += FieldElement::<E>::one();
    assert!(!verify(&cpu_air, &add_air, &mul_air, &proof));
}

/// Tampering the σ (bridge running sum) next-row OOD evaluation is rejected —
/// σ is the only next-row read, so this is the pruned block's single column.
#[test_log::test]
fn test_gkr_tampered_sigma_ood_rejected() {
    let (cpu_air, add_air, mul_air, mut proof) = prove_fixture();
    let add_proof = &mut proof.multi.proofs[1];
    assert_eq!(
        add_proof.trace_ood_next_evaluations.width, 1,
        "pruned next-row block is exactly the σ column"
    );
    let corrupted = *add_proof.trace_ood_next_evaluations.get(0, 0) + FieldElement::one();
    add_proof.trace_ood_next_evaluations.set(0, 0, corrupted);
    assert!(!verify(&cpu_air, &add_air, &mul_air, &proof));
}

/// Tampering the Lagrange-kernel current-row OOD evaluation is rejected (the
/// kernel is bound by its boundary constraint and the bridge's γ^K·l² term).
#[test_log::test]
fn test_gkr_tampered_kernel_ood_rejected() {
    let (cpu_air, add_air, mul_air, mut proof) = prove_fixture();
    let add_proof = &mut proof.multi.proofs[1];
    // ADD has 4 main columns; aux col 0 (kernel) is full-width index 4.
    let kernel_idx = 4usize;
    let corrupted = *add_proof.trace_ood_evaluations.get(0, kernel_idx) + FieldElement::one();
    add_proof
        .trace_ood_evaluations
        .set(0, kernel_idx, corrupted);
    assert!(!verify(&cpu_air, &add_air, &mul_air, &proof));
}

/// Smuggling bus public inputs into a GKR-mode proof is rejected (they would
/// otherwise be absorbed into the fork transcript).
#[test_log::test]
fn test_gkr_smuggled_bus_public_inputs_rejected() {
    use crate::lookup::BusPublicInputs;
    let (cpu_air, add_air, mul_air, mut proof) = prove_fixture();
    proof.multi.proofs[0].bus_public_inputs =
        Some(BusPublicInputs::from_contribution(FieldElement::zero()));
    assert!(!verify(&cpu_air, &add_air, &mul_air, &proof));
}

/// Mode mismatches fail closed in both directions: GKR-mode AIRs refuse the
/// standard entry point, and standard-mode AIRs refuse the GKR entry point.
#[test_log::test]
fn test_gkr_mode_mismatch_rejected() {
    let (cpu_air, add_air, mul_air, proof) = prove_fixture();

    // GKR-mode AIRs through the standard verifier (with the inner MultiProof).
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];
    assert!(
        !Verifier::multi_verify(
            &airs,
            &proof.multi,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "GKR-mode AIRs must be rejected by the standard entry point"
    );

    // Standard-mode AIRs through the GKR verifier.
    let proof_options = ProofOptions::default_test_options();
    let std_cpu = new_cpu_air_with_lookup(&proof_options);
    let std_add = new_add_air_with_lookup(&proof_options);
    let std_mul = new_mul_air_with_lookup(&proof_options);
    let std_airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&std_cpu, &std_add, &std_mul];
    assert!(
        !Verifier::multi_verify_gkr(
            &std_airs,
            &proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "standard-mode AIRs must be rejected by the GKR entry point"
    );
}

/// A wrong expected bus balance is rejected (the fixture balances at zero).
#[test_log::test]
fn test_gkr_wrong_expected_balance_rejected() {
    let (cpu_air, add_air, mul_air, proof) = prove_fixture();
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];
    assert!(!Verifier::multi_verify_gkr(
        &airs,
        &proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::one(),
    ));
}
