//! Tests that verify the prover-verifier separation works correctly.
//!
//! These tests simulate a realistic scenario where:
//! 1. The prover generates proofs and serializes them
//! 2. The proofs are "transmitted" (serialized/deserialized)
//! 3. The verifier creates the AIR from scratch (without the prover's trace)
//! 4. The verifier deserializes the proofs and verifies them

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::{
    babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
};

use crate::constraints::transition::TransitionConstraint;
use crate::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusType, NullBoundaryConstraintBuilder,
    TableInteraction,
};
use crate::proof::options::ProofOptions;
use crate::proof::stark::MultiProof;
use crate::traits::AIR;
use crate::{
    prover::{IsStarkProver, Prover},
    verifier::{IsStarkVerifier, Verifier},
};

type F = Babybear31PrimeField;
type E = Degree4BabyBearExtensionField;
type FE = FieldElement<F>;

/// Test that verifies multi-table LogUp proofs can be serialized, transmitted,
/// and verified by a verifier who never ran the prover.
#[test_log::test]
fn test_verify_serialized_multi_table_proofs() {
    // =========================================================================
    // PROVER SIDE - Generate proofs
    // =========================================================================

    let proofs = {
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
    };

    // =========================================================================
    // NETWORK TRANSMISSION - Serialize and deserialize (using CBOR binary format)
    // =========================================================================

    let serialized = serde_cbor::to_vec(&proofs).expect("Failed to serialize proofs");

    // At this point, the prover's data is dropped (out of scope above)
    // The verifier only has the serialized data

    let received_proofs: MultiProof<F, E, ()> =
        serde_cbor::from_slice(&serialized).expect("Failed to deserialize proofs");

    // =========================================================================
    // VERIFIER SIDE - Reconstruct AIRs and verify
    // =========================================================================
    // The verifier knows the AIR structure (columns, interactions) as part of
    // the protocol definition.

    let proof_options = ProofOptions::default_test_options();

    // Reconstruct AIRs - verifier knows the structure
    let cpu_air = create_cpu_air(&proof_options);
    let add_air = create_add_air(&proof_options);
    let mul_air = create_mul_air(&proof_options);

    // Verify the proofs
    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&cpu_air, &add_air, &mul_air];

    assert!(
        Verifier::multi_verify(
            &airs,
            &received_proofs,
            &mut DefaultTranscript::<E>::new(&[]),
        ),
        "Verification should succeed for valid proofs"
    );
}

// =============================================================================
// Helper functions to create AIRs
// =============================================================================

fn create_cpu_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            TableInteraction::sender(
                Some(0),
                vec![BusType::Single.at(2), BusType::Single.at(3), BusType::Single.at(4)],
            ),
            TableInteraction::sender(
                Some(1),
                vec![BusType::Single.at(2), BusType::Single.at(3), BusType::Single.at(4)],
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
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![TableInteraction::receiver(
            Some(3),
            vec![BusType::Single.at(0), BusType::Single.at(1), BusType::Single.at(2)],
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
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![TableInteraction::receiver(
            Some(3),
            vec![BusType::Single.at(0), BusType::Single.at(1), BusType::Single.at(2)],
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
