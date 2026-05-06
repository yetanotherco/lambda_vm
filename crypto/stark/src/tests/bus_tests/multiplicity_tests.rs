//! Tests for Multiplicity variants.
//!
//! These tests verify that all Multiplicity variants (One, Column, Sum, Negated)
//! work correctly for computing bus interaction multiplicities.

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
        let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];
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
        let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];
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
        Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
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
        let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];
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
        let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];
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
        Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
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
        let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];
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
        let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];
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
        Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Multiplicity::Negated should work for skipping flagged rows"
    );
}

// =============================================================================
// Multiplicity::OpMatch tests
// =============================================================================
//
// OpMatch(match_column) reads multiplicity from a witness column that the
// AIR is responsible for constraining (caller-provided via transition
// constraints — boolean and `match_col * (op - target) = 0`). Functionally
// identical to `Multiplicity::Column`; the tests document the dispatch
// pattern and ensure the variant routes via the named contract.
//
// We exercise the variant standalone here. Phase 1 of the CPU refactor
// adds the witness column and its sibling constraints in the prover crate,
// where the op range and target encoding live.

/// Common helper: build a sender AIR whose single bus interaction uses
/// `OpMatch { match_column: 1 }`. Trace shape: column 0 is `value`,
/// column 1 is the per-row indicator bit (1 on rows that should dispatch).
fn op_match_sender_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![BusInteraction::sender(
            TEST_BUS,
            Multiplicity::OpMatch { match_column: 1 },
            Packing::Direct.columns(&[0]),
        )],
    };
    AirWithBuses::new(
        2, // value, match
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

/// Receiver AIR: receives `value` rows with multiplicity from column 1.
fn op_match_receiver_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![BusInteraction::receiver(
            TEST_BUS,
            Multiplicity::Column(1),
            Packing::Direct.columns(&[0]),
        )],
    };
    AirWithBuses::new(
        2, // value, mu
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

/// Run prove + verify for an `OpMatch` sender plus matching receiver. The
/// sender trace has 4 rows; only row index `match_row` has its match bit
/// set. The receiver receives the corresponding `value` once.
fn run_op_match_roundtrip(match_row: usize, value: u64) -> bool {
    let mut val_col = vec![FE::zero(); 4];
    let mut match_col = vec![FE::zero(); 4];
    val_col[match_row] = FE::from(value);
    match_col[match_row] = FE::one();

    let mut recv_val = vec![FE::zero(); 4];
    let mut recv_mu = vec![FE::zero(); 4];
    recv_val[0] = FE::from(value);
    recv_mu[0] = FE::one();

    let mut sender_trace = TraceTable::from_columns_main(vec![val_col, match_col], 1);
    let mut receiver_trace = TraceTable::from_columns_main(vec![recv_val, recv_mu], 1);

    let proof_options = ProofOptions::default_test_options();
    let sender = op_match_sender_air(&proof_options);
    let receiver = op_match_receiver_air(&proof_options);

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

    Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

/// Smoke test: a single matching row sends; receiver picks it up.
#[test_log::test]
fn test_multiplicity_op_match_single_dispatch() {
    assert!(
        run_op_match_roundtrip(2, 42),
        "OpMatch must dispatch the matching row's value to the receiver"
    );
}

/// Boundary: match on row 0 (start of trace).
#[test_log::test]
fn test_multiplicity_op_match_first_row() {
    assert!(
        run_op_match_roundtrip(0, 1234),
        "OpMatch must work when the matching row is row 0"
    );
}

/// Boundary: match on the last row (last of a power-of-two-sized trace).
#[test_log::test]
fn test_multiplicity_op_match_last_row() {
    assert!(
        run_op_match_roundtrip(3, 5678),
        "OpMatch must work when the matching row is the final row"
    );
}

/// Soundness: a row whose match bit is 0 must NOT send. We construct a
/// receiver expecting a value that only the would-be-matched (but not
/// dispatched) row carries. Bus must fail to balance.
#[test_log::test]
fn test_multiplicity_op_match_unmatched_row_does_not_send() {
    // Row 1 has value=99 but match bit = 0 → must not send.
    let val_col: Vec<FE> = vec![FE::zero(), FE::from(99u64), FE::zero(), FE::zero()];
    let match_col: Vec<FE> = vec![FE::zero(); 4];

    // Receiver demands one copy of value=99.
    let mut recv_val = vec![FE::zero(); 4];
    let mut recv_mu = vec![FE::zero(); 4];
    recv_val[0] = FE::from(99u64);
    recv_mu[0] = FE::one();

    let mut sender_trace = TraceTable::from_columns_main(vec![val_col, match_col], 1);
    let mut receiver_trace = TraceTable::from_columns_main(vec![recv_val, recv_mu], 1);

    let proof_options = ProofOptions::default_test_options();
    let sender = op_match_sender_air(&proof_options);
    let receiver = op_match_receiver_air(&proof_options);

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
        !Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Verifier must reject when match bit is 0 yet receiver expected the value"
    );
}

/// Forged dispatch: prover sets match=1 on a row whose value differs from
/// what the receiver demands. Bus balance must reject. Concretely
/// demonstrates that the OpMatch contract relies on AIR-level constraints
/// (`match_column * (op - target) = 0`) to prevent the prover from routing
/// the wrong row's data — the LogUp framework alone does not stop this.
#[test_log::test]
fn test_multiplicity_op_match_forged_dispatch_rejected() {
    // Sender row 0 carries value=11, row 1 carries value=22, only row 0 has
    // match=1. The prover legitimately dispatches value=11.
    let val_col: Vec<FE> = vec![FE::from(11u64), FE::from(22u64), FE::zero(), FE::zero()];
    let match_col: Vec<FE> = vec![FE::one(), FE::zero(), FE::zero(), FE::zero()];

    // Receiver expects value=22 (the other row's value). With no constraint
    // tying match to the row's data, this is what a "wrong target" forgery
    // would look like — the prover sent value=11, the bus has no copy of
    // value=22, balance fails.
    let mut recv_val = vec![FE::zero(); 4];
    let mut recv_mu = vec![FE::zero(); 4];
    recv_val[0] = FE::from(22u64);
    recv_mu[0] = FE::one();

    let mut sender_trace = TraceTable::from_columns_main(vec![val_col, match_col], 1);
    let mut receiver_trace = TraceTable::from_columns_main(vec![recv_val, recv_mu], 1);

    let proof_options = ProofOptions::default_test_options();
    let sender = op_match_sender_air(&proof_options);
    let receiver = op_match_receiver_air(&proof_options);

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
        !Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Verifier must reject when the sent value does not match the receiver's demand"
    );
}

/// Multiple matching rows: every row's match bit fires; each row's value
/// reaches the receiver with multiplicity 1.
#[test_log::test]
fn test_multiplicity_op_match_all_rows_dispatch() {
    let val_col: Vec<FE> = (1u64..=4).map(FE::from).collect();
    let match_col: Vec<FE> = vec![FE::one(); 4];

    let recv_val = val_col.clone();
    let recv_mu: Vec<FE> = vec![FE::one(); 4];

    let mut sender_trace = TraceTable::from_columns_main(vec![val_col, match_col], 1);
    let mut receiver_trace = TraceTable::from_columns_main(vec![recv_val, recv_mu], 1);

    let proof_options = ProofOptions::default_test_options();
    let sender = op_match_sender_air(&proof_options);
    let receiver = op_match_receiver_air(&proof_options);

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
        Verifier::multi_verify(
            &airs,
            &multi_proof,
            &mut DefaultTranscript::<E>::new(&[]),
            &FieldElement::zero(),
        ),
        "Verifier must accept when every row dispatches and the receiver matches"
    );
}
