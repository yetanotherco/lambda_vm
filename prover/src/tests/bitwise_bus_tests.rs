//! Soundness and completeness tests for BITWISE table bus interactions.
//!
//! These tests verify that:
//! - Completeness: Valid lookups to BITWISE are accepted
//! - Soundness: Invalid lookups to BITWISE are rejected

use crypto::fiat_shamir::default_transcript::DefaultTranscript;

use stark::constraints::transition::TransitionConstraint;
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::proof::options::ProofOptions;
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables64::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

type F = GoldilocksField;
type E = GoldilocksExtension;

// =============================================================================
// Column indices (simplified for testing)
// =============================================================================

/// Sender table columns
mod sender_cols {
    pub const X: usize = 0;
    pub const Y: usize = 1;
    pub const AND_RESULT: usize = 2;
    pub const MU_AND: usize = 3;
    pub const NUM_COLUMNS: usize = 4;
}

/// Receiver (BITWISE-like) table columns
mod receiver_cols {
    pub const X: usize = 0;
    pub const Y: usize = 1;
    pub const AND: usize = 2;
    pub const MU_AND: usize = 3;
    pub const NUM_COLUMNS: usize = 4;
}

// =============================================================================
// AIR definitions
// =============================================================================

fn new_sender_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![BusInteraction::sender(
            BusId::AndByte,
            Multiplicity::Column(sender_cols::MU_AND),
            vec![
                BusValue::Packed {
                    start_column: sender_cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::Y,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::AND_RESULT,
                    packing: Packing::Direct,
                },
            ],
        )],
    };

    AirWithBuses::new(
        sender_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

fn new_receiver_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, ()> {
    let transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![BusInteraction::receiver(
            BusId::AndByte,
            Multiplicity::Column(receiver_cols::MU_AND),
            vec![
                BusValue::Packed {
                    start_column: receiver_cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: receiver_cols::Y,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: receiver_cols::AND,
                    packing: Packing::Direct,
                },
            ],
        )],
    };

    AirWithBuses::new(
        receiver_cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        transition_constraints,
    )
}

// =============================================================================
// Helper functions
// =============================================================================

fn create_sender_trace(lookups: &[(u8, u8, u8)]) -> TraceTable<F, E> {
    let num_rows = lookups.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * sender_cols::NUM_COLUMNS];

    for (i, &(x, y, result)) in lookups.iter().enumerate() {
        let base = i * sender_cols::NUM_COLUMNS;
        data[base + sender_cols::X] = FE::from(x as u64);
        data[base + sender_cols::Y] = FE::from(y as u64);
        data[base + sender_cols::AND_RESULT] = FE::from(result as u64);
        data[base + sender_cols::MU_AND] = FE::one();
    }

    TraceTable::new_main(data, sender_cols::NUM_COLUMNS, 1)
}

fn create_receiver_trace(lookups: &[(u8, u8)]) -> TraceTable<F, E> {
    let num_rows = lookups.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * receiver_cols::NUM_COLUMNS];

    for (i, &(x, y)) in lookups.iter().enumerate() {
        let base = i * receiver_cols::NUM_COLUMNS;
        data[base + receiver_cols::X] = FE::from(x as u64);
        data[base + receiver_cols::Y] = FE::from(y as u64);
        data[base + receiver_cols::AND] = FE::from((x & y) as u64); // Correct AND result
        data[base + receiver_cols::MU_AND] = FE::one();
    }

    TraceTable::new_main(data, receiver_cols::NUM_COLUMNS, 1)
}

fn prove_and_verify(sender_lookups: &[(u8, u8, u8)], receiver_lookups: &[(u8, u8)]) -> bool {
    let mut sender_trace = create_sender_trace(sender_lookups);
    let mut receiver_trace = create_receiver_trace(receiver_lookups);

    let proof_options = ProofOptions::default_test_options();
    let sender_air = new_sender_air(&proof_options);
    let receiver_air = new_receiver_air(&proof_options);

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&sender_air, &mut sender_trace, &()),
        (&receiver_air, &mut receiver_trace, &()),
    ];

    let multi_proof =
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender_air, &receiver_air];

    Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[]))
}

// =============================================================================
// Completeness Tests - Valid lookups should be ACCEPTED
// =============================================================================

#[test]
fn test_completeness_and_byte_simple() {
    // Sender: AND_BYTE[5, 3] = 1 (correct: 5 & 3 = 1)
    // Receiver: has row (5, 3) with AND = 1
    let sender = vec![(5u8, 3u8, 1u8)];
    let receiver = vec![(5u8, 3u8)];

    assert!(prove_and_verify(&sender, &receiver));
}

#[test]
fn test_completeness_and_byte_zero_result() {
    // 0xAA & 0x55 = 0 (alternating bits)
    let sender = vec![(0xAAu8, 0x55u8, 0x00u8)];
    let receiver = vec![(0xAAu8, 0x55u8)];

    assert!(prove_and_verify(&sender, &receiver));
}

#[test]
fn test_completeness_and_byte_max() {
    // 0xFF & 0xFF = 0xFF
    let sender = vec![(0xFFu8, 0xFFu8, 0xFFu8)];
    let receiver = vec![(0xFFu8, 0xFFu8)];

    assert!(prove_and_verify(&sender, &receiver));
}

#[test]
fn test_completeness_multiple_lookups() {
    let sender = vec![
        (0xFFu8, 0xFFu8, 0xFFu8), // FF & FF = FF
        (0xAAu8, 0x55u8, 0x00u8), // AA & 55 = 00
        (0x0Fu8, 0xF0u8, 0x00u8), // 0F & F0 = 00
        (0x12u8, 0x34u8, 0x10u8), // 12 & 34 = 10
    ];
    let receiver: Vec<(u8, u8)> = sender.iter().map(|&(x, y, _)| (x, y)).collect();

    assert!(prove_and_verify(&sender, &receiver));
}

// =============================================================================
// Soundness Tests - Invalid lookups should be REJECTED
// =============================================================================

#[test]
fn test_soundness_wrong_result() {
    // Sender claims AND_BYTE[5, 3] = 99 (WRONG! Should be 1)
    let sender = vec![(5u8, 3u8, 99u8)];
    let receiver = vec![(5u8, 3u8)]; // Has correct value 1

    assert!(!prove_and_verify(&sender, &receiver));
}

#[test]
fn test_soundness_off_by_one() {
    // Sender claims AND_BYTE[0xFF, 0xFF] = 0xFE (WRONG! Should be 0xFF)
    let sender = vec![(0xFFu8, 0xFFu8, 0xFEu8)];
    let receiver = vec![(0xFFu8, 0xFFu8)];

    assert!(!prove_and_verify(&sender, &receiver));
}

#[test]
fn test_soundness_multiplicity_mismatch() {
    // Sender sends 2 lookups for same value, receiver expects 1
    let sender = vec![
        (5u8, 3u8, 1u8),
        (5u8, 3u8, 1u8), // Duplicate!
    ];
    let receiver = vec![(5u8, 3u8)]; // Only 1 multiplicity

    assert!(!prove_and_verify(&sender, &receiver));
}

#[test]
fn test_soundness_missing_receiver_row() {
    // Sender looks up (100, 200), but receiver only has (5, 3)
    let sender = vec![(100u8, 200u8, 64u8)]; // 100 & 200 = 64
    let receiver = vec![(5u8, 3u8)]; // Different row!

    assert!(!prove_and_verify(&sender, &receiver));
}

#[test]
fn test_soundness_swapped_inputs() {
    // Sender: AND_BYTE[3, 5] = 1
    // Receiver: has (5, 3) not (3, 5) - order matters!
    let sender = vec![(3u8, 5u8, 1u8)]; // Note: swapped order
    let receiver = vec![(5u8, 3u8)]; // Different input order

    assert!(!prove_and_verify(&sender, &receiver));
}

#[test]
fn test_soundness_extra_sender_lookup() {
    // Sender sends 2 lookups, receiver only provides 1
    let sender = vec![
        (5u8, 3u8, 1u8),
        (10u8, 6u8, 2u8), // Extra lookup not in receiver
    ];
    let receiver = vec![(5u8, 3u8)];

    assert!(!prove_and_verify(&sender, &receiver));
}
