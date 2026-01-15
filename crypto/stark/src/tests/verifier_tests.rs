//! Tests that verify the prover-verifier separation works correctly.
//!
//! These tests simulate a realistic scenario where:
//! 1. The prover generates proofs and serializes them
//! 2. The proofs are "transmitted" (serialized/deserialized)
//! 3. The verifier creates the AIR from scratch (without the prover's trace)
//! 4. The verifier deserializes the proofs and verifies them

#![allow(clippy::type_complexity)]

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

/// Creates the standard CPU trace for tests (8 rows, 5 main columns + 3 aux).
fn create_cpu_trace() -> TraceTable<F, E> {
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
    let main_columns = vec![add_column, mul_column, a_column, b_column, c_column];
    let aux_columns = vec![
        vec![ExtFE::zero(); 8],
        vec![ExtFE::zero(); 8],
        vec![ExtFE::zero(); 8],
    ];
    TraceTable::from_columns(main_columns, aux_columns, 1)
}

/// Creates the standard ADD trace for tests (4 rows, 4 main columns + 2 aux).
fn create_add_trace() -> TraceTable<F, E> {
    let add_a = vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)];
    let add_b = vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)];
    let add_c = vec![FE::from(11), FE::from(33), FE::from(55), FE::from(66)];
    let add_m = vec![FE::one(); 4];
    TraceTable::from_columns(
        vec![add_a, add_b, add_c, add_m],
        vec![vec![ExtFE::zero(); 4], vec![ExtFE::zero(); 4]],
        1,
    )
}

/// Creates the standard MUL trace for tests (4 rows, 4 main columns + 2 aux).
fn create_mul_trace() -> TraceTable<F, E> {
    let mul_a = vec![FE::from(2), FE::from(4), FE::from(7), FE::from(8)];
    let mul_b = vec![FE::from(20), FE::from(40), FE::from(70), FE::from(80)];
    let mul_c = vec![FE::from(40), FE::from(160), FE::from(490), FE::from(640)];
    let mul_m = vec![FE::one(); 4];
    TraceTable::from_columns(
        vec![mul_a, mul_b, mul_c, mul_m],
        vec![vec![ExtFE::zero(); 4], vec![ExtFE::zero(); 4]],
        1,
    )
}

fn create_cpu_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![
            TableInteraction {
                multiplicity_column: Some(0),
                value_columns: vec![2, 3, 4],
                is_sender: true,
            },
            TableInteraction {
                multiplicity_column: Some(1),
                value_columns: vec![2, 3, 4],
                is_sender: true,
            },
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
        interactions: vec![TableInteraction {
            multiplicity_column: Some(3),
            value_columns: vec![0, 1, 2],
            is_sender: false,
        }],
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
        interactions: vec![TableInteraction {
            multiplicity_column: Some(3),
            value_columns: vec![0, 1, 2],
            is_sender: false,
        }],
    };
    AirWithBuses::new(
        4, // MUL: a, b, c, multiplicity
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

/// Test that verifies multi-table LogUp proofs can be serialized, transmitted,
/// and verified by a verifier who never ran the prover.
#[test_log::test]
fn test_verify_serialized_multi_table_proofs() {
    // PROVER SIDE - Generate proofs
    let proofs = {
        let mut cpu_trace = create_cpu_trace();
        let mut add_trace = create_add_trace();
        let mut mul_trace = create_mul_trace();

        let proof_options = ProofOptions::default_test_options();
        let cpu_air = create_cpu_air(&proof_options);
        let add_air = create_add_air(&proof_options);
        let mul_air = create_mul_air(&proof_options);

        let air_trace_pairs: Vec<(
            &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
            &mut TraceTable<F, E>,
            &(),
        )> = vec![
            (&cpu_air, &mut cpu_trace, &()),
            (&add_air, &mut add_trace, &()),
            (&mul_air, &mut mul_trace, &()),
        ];

        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap()
    };

    // NETWORK TRANSMISSION - Serialize and deserialize (using CBOR binary format)
    let serialized = serde_cbor::to_vec(&proofs).expect("Failed to serialize proofs");
    let received_proofs: MultiProof<F, E, ()> =
        serde_cbor::from_slice(&serialized).expect("Failed to deserialize proofs");

    // VERIFIER SIDE - Reconstruct AIRs and verify
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
        ),
        "Verification should succeed for valid proofs"
    );
}
