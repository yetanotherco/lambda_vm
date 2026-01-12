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
    AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder, TableInteraction,
};
use crate::proof::options::ProofOptions;
use crate::proof::stark::MultiProof;
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::{
    prover::{IsStarkProver, Prover},
    verifier::{IsStarkVerifier, Verifier},
};

type F = Babybear31PrimeField;
type E = Degree4BabyBearExtensionField;
type FE = FieldElement<F>;
type ExtFE = FieldElement<E>;

/// Test that verifies multi-table LogUp proofs can be serialized, transmitted,
/// and verified by a verifier who never ran the prover.
///
/// This simulates the real-world scenario where:
/// - Prover runs on one machine, generates proofs
/// - Proofs are serialized and sent over network
/// - Verifier runs on different machine, receives proofs
/// - Verifier reconstructs AIRs from public data and verifies
///
/// The proofs are self-contained - they include bus_interactions (initial/final
/// aux column values) needed for verification. The verifier only needs to know
/// the AIR structure (which chips, their columns, interactions) which is part
/// of the protocol definition.
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
        let cpu_aux_columns = vec![
            vec![ExtFE::zero(); 8],
            vec![ExtFE::zero(); 8],
            vec![ExtFE::zero(); 8],
        ];
        let mut cpu_trace = TraceTable::from_columns(cpu_main_columns, cpu_aux_columns, 1);

        // ADD Trace (4 rows, 4 main columns)
        let add_a = vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)];
        let add_b = vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)];
        let add_c = vec![FE::from(11), FE::from(33), FE::from(55), FE::from(66)];
        let add_m = vec![FE::one(), FE::one(), FE::one(), FE::one()];
        let mut add_trace = TraceTable::from_columns(
            vec![add_a, add_b, add_c, add_m],
            vec![vec![ExtFE::zero(); 4]],
            1,
        );

        // MUL Trace (4 rows, 4 main columns)
        let mul_a = vec![FE::from(2), FE::from(4), FE::from(7), FE::from(8)];
        let mul_b = vec![FE::from(20), FE::from(40), FE::from(70), FE::from(80)];
        let mul_c = vec![FE::from(40), FE::from(160), FE::from(490), FE::from(640)];
        let mul_m = vec![FE::one(), FE::one(), FE::one(), FE::one()];
        let mut mul_trace = TraceTable::from_columns(
            vec![mul_a, mul_b, mul_c, mul_m],
            vec![vec![ExtFE::zero(); 4]],
            1,
        );

        let proof_options = ProofOptions::default_test_options();

        // Create AIRs for proving
        let cpu_air = create_cpu_air_for_prover(&cpu_trace, &proof_options);
        let add_air = create_add_air_for_prover(&add_trace, &proof_options);
        let mul_air = create_mul_air_for_prover(&mul_trace, &proof_options);

        // Generate proofs
        let airs: Vec<(
            &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
            &mut TraceTable<F, E>,
        )> = vec![
            (&cpu_air, &mut cpu_trace),
            (&add_air, &mut add_trace),
            (&mul_air, &mut mul_trace),
        ];

        Prover::multi_prove(airs, &mut DefaultTranscript::<E>::new(&[])).unwrap()
    };

    // =========================================================================
    // NETWORK TRANSMISSION - Serialize and deserialize (using CBOR binary format)
    // =========================================================================

    let serialized = serde_cbor::to_vec(&proofs).expect("Failed to serialize proofs");

    // At this point, the prover's data is dropped (out of scope above)
    // The verifier only has the serialized data

    let received_proofs: MultiProof<F, E> =
        serde_cbor::from_slice(&serialized).expect("Failed to deserialize proofs");

    // =========================================================================
    // VERIFIER SIDE - Reconstruct AIRs and verify
    // =========================================================================
    // The verifier knows the AIR structure (which chips, their columns, interactions)
    // as part of the protocol definition. Only trace_length varies per proof.

    let proof_options = ProofOptions::default_test_options();

    // Reconstruct AIRs - verifier knows the structure, only needs trace_length from proof
    let cpu_air = create_cpu_air_for_verifier(
        received_proofs.proofs[0].trace_length,
        5, // CPU has 5 main columns - verifier knows this
        &proof_options,
    );
    let add_air = create_add_air_for_verifier(
        received_proofs.proofs[1].trace_length,
        4, // ADD has 4 main columns - verifier knows this
        &proof_options,
    );
    let mul_air = create_mul_air_for_verifier(
        received_proofs.proofs[2].trace_length,
        4, // MUL has 4 main columns - verifier knows this
        &proof_options,
    );

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

/// Creates a CPU AIR for the prover (needs trace to extract public inputs)
fn create_cpu_air_for_prover(
    trace: &TraceTable<F, E>,
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            TableInteraction {
                flag_columns: vec![0],
                value_columns: vec![2, 3, 4],
                is_sender: true,
            },
            TableInteraction {
                flag_columns: vec![1],
                value_columns: vec![2, 3, 4],
                is_sender: true,
            },
        ],
    };
    AirWithBuses::create(
        trace,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
        (),
    )
}

/// Creates a CPU AIR for the verifier (no trace needed)
fn create_cpu_air_for_verifier(
    trace_length: usize,
    num_main_columns: usize,
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            TableInteraction {
                flag_columns: vec![0],
                value_columns: vec![2, 3, 4],
                is_sender: true,
            },
            TableInteraction {
                flag_columns: vec![1],
                value_columns: vec![2, 3, 4],
                is_sender: true,
            },
        ],
    };
    AirWithBuses::create_for_verification(
        trace_length,
        num_main_columns,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
        (),
    )
}

/// Creates an ADD AIR for the prover
fn create_add_air_for_prover(
    trace: &TraceTable<F, E>,
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![TableInteraction {
            flag_columns: vec![3],
            value_columns: vec![0, 1, 2],
            is_sender: false,
        }],
    };
    AirWithBuses::create(
        trace,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
        (),
    )
}

/// Creates an ADD AIR for the verifier
fn create_add_air_for_verifier(
    trace_length: usize,
    num_main_columns: usize,
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![TableInteraction {
            flag_columns: vec![3],
            value_columns: vec![0, 1, 2],
            is_sender: false,
        }],
    };
    AirWithBuses::create_for_verification(
        trace_length,
        num_main_columns,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
        (),
    )
}

/// Creates a MUL AIR for the prover
fn create_mul_air_for_prover(
    trace: &TraceTable<F, E>,
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![TableInteraction {
            flag_columns: vec![3],
            value_columns: vec![0, 1, 2],
            is_sender: false,
        }],
    };
    AirWithBuses::create(
        trace,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
        (),
    )
}

/// Creates a MUL AIR for the verifier
fn create_mul_air_for_verifier(
    trace_length: usize,
    num_main_columns: usize,
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![TableInteraction {
            flag_columns: vec![3],
            value_columns: vec![0, 1, 2],
            is_sender: false,
        }],
    };
    AirWithBuses::create_for_verification(
        trace_length,
        num_main_columns,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
        (),
    )
}
