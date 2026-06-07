//! Roundtrip tests: proof serialization and prover-verifier separation.
//!
//! These tests verify that proofs survive serialization/deserialization
//! and can be verified independently from the prover.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use crate::constraints::transition::TransitionConstraintEvaluator;
use crate::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use crate::proof::options::ProofOptions;
use crate::proof::stark::MultiProof;
use crate::traits::AIR;
use crate::{
    prover::{IsStarkProver, Prover},
    verifier::{IsStarkVerifier, Verifier},
};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

/// Bus IDs for multi-table tests
#[repr(u64)]
enum BusId {
    Add,
    Mul,
}

impl From<BusId> for u64 {
    fn from(id: BusId) -> u64 {
        id as u64
    }
}

/// Test that verifies multi-table LogUp proofs can be serialized, transmitted,
/// and verified by a verifier who never ran the prover.
/// Builds the deterministic 3-table (CPU/ADD/MUL via LogUp) demo proof shared by
/// the roundtrip test and the byte-identical-proof oracle.
fn build_demo_multi_proof() -> MultiProof<F, E, ()> {
    {
        // CPU Trace (8 rows, 5 main columns)
        let add_column = vec![
            FE::one(),
            FE::zero(),
            FE::one(),
            FE::zero(),
            FE::one(),
            FE::one(),
            FE::zero(),
            FE::zero(),
        ];
        let mul_column = vec![
            FE::zero(),
            FE::one(),
            FE::zero(),
            FE::one(),
            FE::zero(),
            FE::zero(),
            FE::one(),
            FE::one(),
        ];
        let a_column = vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
        ];
        let b_column = vec![
            FE::from(10),
            FE::from(20),
            FE::from(30),
            FE::from(40),
            FE::from(50),
            FE::from(60),
            FE::from(70),
            FE::from(80),
        ];
        let c_column = vec![
            FE::from(11),  // 1 + 10
            FE::from(40),  // 2 * 20
            FE::from(33),  // 3 + 30
            FE::from(160), // 4 * 40
            FE::from(55),  // 5 + 50
            FE::from(66),  // 6 + 60
            FE::from(490), // 7 * 70
            FE::from(640), // 8 * 80
        ];
        let cpu_main_columns = vec![add_column, mul_column, a_column, b_column, c_column];
        let mut cpu_trace = crate::trace::TraceTable::from_columns_main(cpu_main_columns, 1);

        // ADD Trace (4 rows, 4 main columns)
        let add_a = vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)];
        let add_b = vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)];
        let add_c = vec![FE::from(11), FE::from(33), FE::from(55), FE::from(66)];
        let add_m = vec![FE::one(), FE::one(), FE::one(), FE::one()];
        let mut add_trace =
            crate::trace::TraceTable::from_columns_main(vec![add_a, add_b, add_c, add_m], 1);

        // MUL Trace (4 rows, 4 main columns)
        let mul_a = vec![FE::from(2), FE::from(4), FE::from(7), FE::from(8)];
        let mul_b = vec![FE::from(20), FE::from(40), FE::from(70), FE::from(80)];
        let mul_c = vec![FE::from(40), FE::from(160), FE::from(490), FE::from(640)];
        let mul_m = vec![FE::one(), FE::one(), FE::one(), FE::one()];
        let mut mul_trace =
            crate::trace::TraceTable::from_columns_main(vec![mul_a, mul_b, mul_c, mul_m], 1);

        let proof_options = ProofOptions::default_test_options();

        // Create AIRs - prover passes num_main_columns from trace
        let cpu_air = create_cpu_air(&proof_options);
        let add_air = create_add_air(&proof_options);
        let mul_air = create_mul_air(&proof_options);

        // Generate proofs
        #[allow(clippy::type_complexity)]
        let air_trace_pairs: Vec<(
            &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
            &mut crate::trace::TraceTable<F, E>,
            &(),
        )> = vec![
            (&cpu_air, &mut cpu_trace, &()),
            (&add_air, &mut add_trace, &()),
            (&mul_air, &mut mul_trace, &()),
        ];

        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap()
    }
}

#[test_log::test]
fn test_verify_serialized_multi_table_proofs() {
    // PROVER SIDE - generate the demo proof.
    let proofs = build_demo_multi_proof();

    // NETWORK TRANSMISSION - serialize/deserialize (CBOR binary format).
    let serialized = serde_cbor::to_vec(&proofs).expect("Failed to serialize proofs");
    let received_proofs: MultiProof<F, E, ()> =
        serde_cbor::from_slice(&serialized).expect("Failed to deserialize proofs");

    // VERIFIER SIDE - reconstruct AIRs (the verifier knows the structure) and verify.
    let proof_options = ProofOptions::default_test_options();
    let cpu_air = create_cpu_air(&proof_options);
    let add_air = create_add_air(&proof_options);
    let mul_air = create_mul_air(&proof_options);
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        Verifier::multi_verify(
            &airs,
            &received_proofs,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Verification should succeed for valid proofs"
    );
}

/// Phase 0 (Track B) — byte-identical-proof oracle.
///
/// The deterministic demo proof (Fiat-Shamir) must serialize to a fixed digest.
/// This is the tripwire for the bit-reversed-LDE wiring: that rework is meant to
/// be *proof-invariant* (same committed roots/openings, only the prover's
/// evaluation order changes), so any change that perturbs the committed proof
/// fails here — flagging an index-map bug. (`test_verify_serialized_multi_table_proofs`
/// covers that the same proof still verifies.)
#[test]
fn track_b_byte_identical_proof_oracle() {
    let proofs = build_demo_multi_proof();

    // FNV-1a-64 over the CBOR bytes: stable across platforms/Rust versions, no
    // extra deps. Golden captured on the natural-order baseline (1f2cebbb).
    let bytes = serde_cbor::to_vec(&proofs).expect("serialize proofs");
    let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
    for b in &bytes {
        digest ^= *b as u64;
        digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
    }
    const GOLDEN_LEN: usize = 24321;
    const GOLDEN_DIGEST: u64 = 0xe466_7f95_8994_7065;
    assert_eq!(
        (bytes.len(), digest),
        (GOLDEN_LEN, GOLDEN_DIGEST),
        "proof changed — len={} digest={:#018x} (update golden ONLY if intentional)",
        bytes.len(),
        digest,
    );
}

// =============================================================================
// Helper functions to create AIRs
// =============================================================================

fn create_cpu_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            BusInteraction::sender(
                BusId::Add,
                Multiplicity::Column(0),
                Packing::Direct.columns(&[2, 3, 4]),
            ),
            BusInteraction::sender(
                BusId::Mul,
                Multiplicity::Column(1),
                Packing::Direct.columns(&[2, 3, 4]),
            ),
        ],
    };
    AirWithBuses::new(
        5, // CPU: add_flag, mul_flag, a, b, c
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

fn create_add_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![BusInteraction::receiver(
            BusId::Add,
            Multiplicity::Column(3),
            Packing::Direct.columns(&[0, 1, 2]),
        )],
    };
    AirWithBuses::new(
        4, // ADD: a, b, c, multiplicity
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

fn create_mul_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![BusInteraction::receiver(
            BusId::Mul,
            Multiplicity::Column(3),
            Packing::Direct.columns(&[0, 1, 2]),
        )],
    };
    AirWithBuses::new(
        4, // MUL: a, b, c, multiplicity
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}
