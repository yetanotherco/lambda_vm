//! Tests for Multiplicity variants.
//!
//! These tests verify that all Multiplicity variants (One, Column, Sum, Negated, Diff, Sum3)
//! work correctly for computing bus interaction multiplicities.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use crate::constraints::transition::TransitionConstraint;
use crate::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use crate::proof::options::ProofOptions;
use crate::prover::{IsStarkProver, Prover};
use crate::trace::TraceTable;
use crate::traits::AIR;
use crate::verifier::{IsStarkVerifier, Verifier};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

/// Bus ID for multiplicity tests (single bus)
const TEST_BUS: u64 = 0;

// =============================================================================
// Multiplicity::One tests
// =============================================================================

/// Test Multiplicity::One: every row contributes with multiplicity 1.
/// Sender sends 4 values (one per row), receiver receives each once.
#[test_log::test]
fn test_multiplicity_one() {
    fn sender_air(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
        let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // Multiplicity::One means every row sends with multiplicity 1
                BusInteraction::sender(
                    TEST_BUS,
                    Multiplicity::One,
                    Packing::Direct.columns(&[0, 1]),
                ),
            ],
        };
        AirWithBuses::new(
            2, // columns: a, b
            auxiliary_trace_build_data,
            proof_options,
            1,
            transition_constraints,
        )
    }

    fn receiver_air(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
        let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // Receiver also uses Multiplicity::One
                BusInteraction::receiver(
                    TEST_BUS,
                    Multiplicity::One,
                    Packing::Direct.columns(&[0, 1]),
                ),
            ],
        };
        AirWithBuses::new(
            2, // columns: a, b
            auxiliary_trace_build_data,
            proof_options,
            1,
            transition_constraints,
        )
    }

    // Sender trace: 4 rows, each sends (a, b) with multiplicity 1
    let mut sender_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(10), FE::from(20), FE::from(30), FE::from(40)], // a
            vec![FE::from(100), FE::from(200), FE::from(300), FE::from(400)], // b
        ],
        1,
    );

    // Receiver trace: same 4 rows, each receives with multiplicity 1
    let mut receiver_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(10), FE::from(20), FE::from(30), FE::from(40)], // a
            vec![FE::from(100), FE::from(200), FE::from(300), FE::from(400)], // b
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let sender = sender_air(&proof_options);
    let receiver = receiver_air(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&sender, &mut sender_trace, &()),
        (&receiver, &mut receiver_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender, &receiver];

    assert!(
        Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Multiplicity::One should work for matching sender/receiver"
    );
}

// =============================================================================
// Multiplicity::Sum tests
// =============================================================================

/// Test Multiplicity::Sum: multiplicity is col_a + col_b.
/// Sender has two flag columns, receiver uses their sum as multiplicity.
#[test_log::test]
fn test_multiplicity_sum() {
    fn sender_air(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
        let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // Multiplicity::Sum(0, 1) means multiplicity = col[0] + col[1]
                BusInteraction::sender(
                    TEST_BUS,
                    Multiplicity::Sum(0, 1),
                    Packing::Direct.columns(&[2, 3]),
                ),
            ],
        };
        AirWithBuses::new(
            4, // columns: flag_a, flag_b, value_a, value_b
            auxiliary_trace_build_data,
            proof_options,
            1,
            transition_constraints,
        )
    }

    fn receiver_air(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
        let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // Receiver uses Column(2) as multiplicity
                BusInteraction::receiver(
                    TEST_BUS,
                    Multiplicity::Column(2),
                    Packing::Direct.columns(&[0, 1]),
                ),
            ],
        };
        AirWithBuses::new(
            3, // columns: value_a, value_b, multiplicity
            auxiliary_trace_build_data,
            proof_options,
            1,
            transition_constraints,
        )
    }

    // Sender trace:
    // Row 0: flag_a=1, flag_b=0, sends (10, 100) with mult=1
    // Row 1: flag_a=0, flag_b=1, sends (20, 200) with mult=1
    // Row 2: flag_a=1, flag_b=1, sends (30, 300) with mult=2
    // Row 3: flag_a=0, flag_b=0, sends (40, 400) with mult=0 (no contribution)
    let mut sender_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::zero(), FE::one(), FE::zero()], // flag_a
            vec![FE::zero(), FE::one(), FE::one(), FE::zero()], // flag_b
            vec![FE::from(10), FE::from(20), FE::from(30), FE::from(40)], // value_a
            vec![FE::from(100), FE::from(200), FE::from(300), FE::from(400)], // value_b
        ],
        1,
    );

    // Receiver trace: receives the values with matching multiplicities
    // (10, 100) with mult=1, (20, 200) with mult=1, (30, 300) with mult=2
    let mut receiver_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(10), FE::from(20), FE::from(30), FE::zero()], // value_a
            vec![FE::from(100), FE::from(200), FE::from(300), FE::zero()], // value_b
            vec![FE::one(), FE::one(), FE::from(2), FE::zero()],        // multiplicity
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let sender = sender_air(&proof_options);
    let receiver = receiver_air(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&sender, &mut sender_trace, &()),
        (&receiver, &mut receiver_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender, &receiver];

    assert!(
        Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Multiplicity::Sum should work for matching sender/receiver"
    );
}

// =============================================================================
// Multiplicity::Negated tests
// =============================================================================

/// Test Multiplicity::Negated: multiplicity is 1 - col_value.
/// Useful for "all rows except those marked by this flag".
#[test_log::test]
fn test_multiplicity_negated() {
    fn sender_air(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
        let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // Multiplicity::Negated(0) means multiplicity = 1 - col[0]
                // When col[0]=0, multiplicity=1; when col[0]=1, multiplicity=0
                BusInteraction::sender(
                    TEST_BUS,
                    Multiplicity::Negated(0),
                    Packing::Direct.columns(&[1, 2]),
                ),
            ],
        };
        AirWithBuses::new(
            3, // columns: skip_flag, value_a, value_b
            auxiliary_trace_build_data,
            proof_options,
            1,
            transition_constraints,
        )
    }

    fn receiver_air(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
        let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![BusInteraction::receiver(
                TEST_BUS,
                Multiplicity::Column(2),
                Packing::Direct.columns(&[0, 1]),
            )],
        };
        AirWithBuses::new(
            3, // columns: value_a, value_b, multiplicity
            auxiliary_trace_build_data,
            proof_options,
            1,
            transition_constraints,
        )
    }

    // Sender trace:
    // Row 0: skip_flag=0, sends (10, 100) with mult=1-0=1
    // Row 1: skip_flag=1, sends (20, 200) with mult=1-1=0 (skipped!)
    // Row 2: skip_flag=0, sends (30, 300) with mult=1-0=1
    // Row 3: skip_flag=0, sends (40, 400) with mult=1-0=1
    let mut sender_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::zero(), FE::one(), FE::zero(), FE::zero()], // skip_flag
            vec![FE::from(10), FE::from(20), FE::from(30), FE::from(40)], // value_a
            vec![FE::from(100), FE::from(200), FE::from(300), FE::from(400)], // value_b
        ],
        1,
    );

    // Receiver trace: only receives rows where skip_flag=0
    // (10, 100), (30, 300), (40, 400) each with multiplicity 1
    // Row 1 is padding (not used)
    let mut receiver_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(10), FE::from(30), FE::from(40), FE::zero()], // value_a
            vec![FE::from(100), FE::from(300), FE::from(400), FE::zero()], // value_b
            vec![FE::one(), FE::one(), FE::one(), FE::zero()],          // multiplicity
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let sender = sender_air(&proof_options);
    let receiver = receiver_air(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&sender, &mut sender_trace, &()),
        (&receiver, &mut receiver_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender, &receiver];

    assert!(
        Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Multiplicity::Negated should work for skipping flagged rows"
    );
}

// =============================================================================
// Multiplicity::Diff tests
// =============================================================================

/// Test Multiplicity::Diff: multiplicity is col_a - col_b.
/// Sender uses difference of two columns as multiplicity.
#[test_log::test]
fn test_multiplicity_diff() {
    fn sender_air(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
        let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // Multiplicity::Diff(0, 1) means multiplicity = col[0] - col[1]
                BusInteraction::sender(
                    TEST_BUS,
                    Multiplicity::Diff(0, 1),
                    Packing::Direct.columns(&[2, 3]),
                ),
            ],
        };
        AirWithBuses::new(
            4, // columns: total, used, value_a, value_b
            auxiliary_trace_build_data,
            proof_options,
            1,
            transition_constraints,
        )
    }

    fn receiver_air(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
        let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![BusInteraction::receiver(
                TEST_BUS,
                Multiplicity::Column(2),
                Packing::Direct.columns(&[0, 1]),
            )],
        };
        AirWithBuses::new(
            3, // columns: value_a, value_b, multiplicity
            auxiliary_trace_build_data,
            proof_options,
            1,
            transition_constraints,
        )
    }

    // Sender trace:
    // Row 0: total=3, used=1, sends (10, 100) with mult=3-1=2
    // Row 1: total=5, used=4, sends (20, 200) with mult=5-4=1
    // Row 2: total=2, used=2, sends (30, 300) with mult=2-2=0 (no contribution)
    // Row 3: total=0, used=0, sends (40, 400) with mult=0-0=0 (no contribution)
    let mut sender_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(3), FE::from(5), FE::from(2), FE::zero()], // total
            vec![FE::one(), FE::from(4), FE::from(2), FE::zero()],   // used
            vec![FE::from(10), FE::from(20), FE::from(30), FE::from(40)], // value_a
            vec![FE::from(100), FE::from(200), FE::from(300), FE::from(400)], // value_b
        ],
        1,
    );

    // Receiver trace: receives (10, 100) with mult=2, (20, 200) with mult=1
    let mut receiver_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(10), FE::from(20), FE::zero(), FE::zero()], // value_a
            vec![FE::from(100), FE::from(200), FE::zero(), FE::zero()], // value_b
            vec![FE::from(2), FE::one(), FE::zero(), FE::zero()],     // multiplicity
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let sender = sender_air(&proof_options);
    let receiver = receiver_air(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&sender, &mut sender_trace, &()),
        (&receiver, &mut receiver_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender, &receiver];

    assert!(
        Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Multiplicity::Diff should work for difference of two columns"
    );
}

// =============================================================================
// Multiplicity::Sum3 tests
// =============================================================================

/// Test Multiplicity::Sum3: multiplicity is col_a + col_b + col_c.
/// Sender uses sum of three flag columns as multiplicity.
#[test_log::test]
fn test_multiplicity_sum3() {
    fn sender_air(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
        let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![
                // Multiplicity::Sum3(0, 1, 2) means multiplicity = col[0] + col[1] + col[2]
                BusInteraction::sender(
                    TEST_BUS,
                    Multiplicity::Sum3(0, 1, 2),
                    Packing::Direct.columns(&[3, 4]),
                ),
            ],
        };
        AirWithBuses::new(
            5, // columns: flag_a, flag_b, flag_c, value_a, value_b
            auxiliary_trace_build_data,
            proof_options,
            1,
            transition_constraints,
        )
    }

    fn receiver_air(
        proof_options: &ProofOptions,
    ) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
        let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];
        let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
            interactions: vec![BusInteraction::receiver(
                TEST_BUS,
                Multiplicity::Column(2),
                Packing::Direct.columns(&[0, 1]),
            )],
        };
        AirWithBuses::new(
            3, // columns: value_a, value_b, multiplicity
            auxiliary_trace_build_data,
            proof_options,
            1,
            transition_constraints,
        )
    }

    // Sender trace:
    // Row 0: flags=(1,1,1), sends (10, 100) with mult=3
    // Row 1: flags=(1,0,0), sends (20, 200) with mult=1
    // Row 2: flags=(0,1,1), sends (30, 300) with mult=2
    // Row 3: flags=(0,0,0), sends (40, 400) with mult=0 (no contribution)
    let mut sender_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::one(), FE::one(), FE::zero(), FE::zero()], // flag_a
            vec![FE::one(), FE::zero(), FE::one(), FE::zero()], // flag_b
            vec![FE::one(), FE::zero(), FE::one(), FE::zero()], // flag_c
            vec![FE::from(10), FE::from(20), FE::from(30), FE::from(40)], // value_a
            vec![FE::from(100), FE::from(200), FE::from(300), FE::from(400)], // value_b
        ],
        1,
    );

    // Receiver trace: receives with matching multiplicities
    // (10, 100) with mult=3, (20, 200) with mult=1, (30, 300) with mult=2
    let mut receiver_trace = TraceTable::from_columns_main(
        vec![
            vec![FE::from(10), FE::from(20), FE::from(30), FE::zero()], // value_a
            vec![FE::from(100), FE::from(200), FE::from(300), FE::zero()], // value_b
            vec![FE::from(3), FE::one(), FE::from(2), FE::zero()],      // multiplicity
        ],
        1,
    );

    let proof_options = ProofOptions::default_test_options();
    let sender = sender_air(&proof_options);
    let receiver = receiver_air(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&sender, &mut sender_trace, &()),
        (&receiver, &mut receiver_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender, &receiver];

    assert!(
        Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[])),
        "Multiplicity::Sum3 should work for sum of three columns"
    );
}
