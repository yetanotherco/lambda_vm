//! Soundness and completeness tests for BITWISE table bus interactions.
//!
//! These tests verify that:
//! - Completeness: Valid lookups to BITWISE are accepted
//! - Soundness: Invalid lookups to BITWISE are rejected

use std::collections::HashMap;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;

use stark::constraints::transition::TransitionConstraintEvaluator;
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::test_utils::multi_prove_ram;

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
    /// Flag indicating this row performs an AND operation (0 or 1)
    pub const AND: usize = 3;
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
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];

    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![BusInteraction::sender(
            BusId::AndByte,
            Multiplicity::Column(sender_cols::AND),
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
    let transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = vec![];

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
        data[base + sender_cols::AND] = FE::one(); // Flag: this row performs AND
    }

    TraceTable::new_main(data, sender_cols::NUM_COLUMNS, 1)
}

/// Creates a receiver (BITWISE-like) trace with precomputed values.
///
/// This function mimics the real BITWISE table behavior:
/// 1. Precomputes the AND result for each unique (X, Y) combination
/// 2. Updates multiplicities based on how many times each tuple is received from sender
///
/// # Arguments
/// * `sender_lookups` - Tuples (X, Y, claimed_result) sent by the sender table
fn create_receiver_trace(sender_lookups: &[(u8, u8, u8)]) -> TraceTable<F, E> {
    // Count multiplicities for each unique (X, Y) pair from sender lookups
    let mut multiplicities: HashMap<(u8, u8), u32> = HashMap::new();
    for &(x, y, _) in sender_lookups {
        *multiplicities.entry((x, y)).or_insert(0) += 1;
    }

    // Build precomputed table with AND results and multiplicities
    let unique_rows: Vec<((u8, u8), u32)> = multiplicities.into_iter().collect();
    let num_rows = unique_rows.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * receiver_cols::NUM_COLUMNS];

    for (i, &((x, y), multiplicity)) in unique_rows.iter().enumerate() {
        let base = i * receiver_cols::NUM_COLUMNS;
        // Precomputed values
        data[base + receiver_cols::X] = FE::from(x as u64);
        data[base + receiver_cols::Y] = FE::from(y as u64);
        data[base + receiver_cols::AND] = FE::from((x & y) as u64); // Precomputed AND
        // Multiplicity updated from sender lookups
        data[base + receiver_cols::MU_AND] = FE::from(multiplicity as u64);
    }

    TraceTable::new_main(data, receiver_cols::NUM_COLUMNS, 1)
}

/// Proves and verifies a sender-receiver interaction.
///
/// The receiver trace is automatically built from sender lookups, mimicking
/// how the BITWISE table works: precomputed values with multiplicities
/// updated based on received lookups.
fn prove_and_verify(sender_lookups: &[(u8, u8, u8)]) -> bool {
    let mut sender_trace = create_sender_trace(sender_lookups);
    let mut receiver_trace = create_receiver_trace(sender_lookups);

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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender_air, &receiver_air];

    Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

// =============================================================================
// Completeness Tests - Valid lookups should be ACCEPTED
// =============================================================================

#[test]
fn test_completeness_and_byte_simple() {
    // Sender: AND_BYTE[5, 3] = 1 (correct: 5 & 3 = 1)
    // Receiver: precomputed table has row (5, 3) with AND = 1, multiplicity = 1
    let sender = vec![(5u8, 3u8, 1u8)];

    assert!(prove_and_verify(&sender));
}

#[test]
fn test_completeness_and_byte_zero_result() {
    // 0xAA & 0x55 = 0 (alternating bits)
    let sender = vec![(0xAAu8, 0x55u8, 0x00u8)];

    assert!(prove_and_verify(&sender));
}

#[test]
fn test_completeness_and_byte_max() {
    // 0xFF & 0xFF = 0xFF
    let sender = vec![(0xFFu8, 0xFFu8, 0xFFu8)];

    assert!(prove_and_verify(&sender));
}

#[test]
fn test_completeness_multiple_lookups() {
    // Multiple different lookups - receiver precomputes each and sets multiplicity = 1
    let sender = vec![
        (0xFFu8, 0xFFu8, 0xFFu8), // FF & FF = FF
        (0xAAu8, 0x55u8, 0x00u8), // AA & 55 = 00
        (0x0Fu8, 0xF0u8, 0x00u8), // 0F & F0 = 00
        (0x12u8, 0x34u8, 0x10u8), // 12 & 34 = 10
    ];

    assert!(prove_and_verify(&sender));
}

#[test]
fn test_completeness_duplicate_lookups() {
    // Same lookup multiple times - receiver precomputes once with multiplicity = 3
    let sender = vec![
        (5u8, 3u8, 1u8), // First lookup
        (5u8, 3u8, 1u8), // Duplicate
        (5u8, 3u8, 1u8), // Duplicate
    ];

    assert!(prove_and_verify(&sender));
}

// =============================================================================
// Soundness Tests - Invalid lookups should be REJECTED
// =============================================================================

/// Creates a custom receiver trace for soundness tests.
/// Allows specifying exact (X, Y, AND, multiplicity) tuples.
fn create_custom_receiver_trace(rows: &[(u8, u8, u8, u32)]) -> TraceTable<F, E> {
    let num_rows = rows.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * receiver_cols::NUM_COLUMNS];

    for (i, &(x, y, and_result, multiplicity)) in rows.iter().enumerate() {
        let base = i * receiver_cols::NUM_COLUMNS;
        data[base + receiver_cols::X] = FE::from(x as u64);
        data[base + receiver_cols::Y] = FE::from(y as u64);
        data[base + receiver_cols::AND] = FE::from(and_result as u64);
        data[base + receiver_cols::MU_AND] = FE::from(multiplicity as u64);
    }

    TraceTable::new_main(data, receiver_cols::NUM_COLUMNS, 1)
}

/// Proves and verifies with a custom receiver trace (for soundness tests).
fn prove_and_verify_custom(
    sender_lookups: &[(u8, u8, u8)],
    receiver_rows: &[(u8, u8, u8, u32)],
) -> bool {
    let mut sender_trace = create_sender_trace(sender_lookups);
    let mut receiver_trace = create_custom_receiver_trace(receiver_rows);

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
        multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender_air, &receiver_air];

    Verifier::multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

#[test]
fn test_soundness_wrong_result() {
    // Sender claims AND_BYTE[5, 3] = 99 (WRONG! Should be 1)
    // Receiver has precomputed correct value 1, so verification should fail
    let sender = vec![(5u8, 3u8, 99u8)];

    assert!(!prove_and_verify(&sender));
}

#[test]
fn test_soundness_off_by_one() {
    // Sender claims AND_BYTE[0xFF, 0xFF] = 0xFE (WRONG! Should be 0xFF)
    let sender = vec![(0xFFu8, 0xFFu8, 0xFEu8)];

    assert!(!prove_and_verify(&sender));
}

#[test]
fn test_soundness_multiplicity_mismatch() {
    // Sender sends 2 lookups for same value, but receiver has multiplicity 1
    let sender = vec![
        (5u8, 3u8, 1u8),
        (5u8, 3u8, 1u8), // Duplicate!
    ];
    // Custom receiver with wrong multiplicity (1 instead of 2)
    let receiver = vec![(5u8, 3u8, 1u8, 1u32)]; // multiplicity = 1, should be 2

    assert!(!prove_and_verify_custom(&sender, &receiver));
}

#[test]
fn test_soundness_missing_receiver_row() {
    // Sender looks up (100, 200), but receiver only has (5, 3)
    let sender = vec![(100u8, 200u8, 64u8)]; // 100 & 200 = 64
    // Custom receiver with different row
    let receiver = vec![(5u8, 3u8, 1u8, 1u32)]; // Different row!

    assert!(!prove_and_verify_custom(&sender, &receiver));
}

#[test]
fn test_soundness_swapped_inputs() {
    // Sender: AND_BYTE[3, 5] = 1
    // Receiver: has (5, 3) not (3, 5) - order matters!
    let sender = vec![(3u8, 5u8, 1u8)]; // Note: X=3, Y=5
    // Custom receiver with swapped inputs
    let receiver = vec![(5u8, 3u8, 1u8, 1u32)]; // X=5, Y=3 - different input order

    assert!(!prove_and_verify_custom(&sender, &receiver));
}

#[test]
fn test_soundness_extra_sender_lookup() {
    // Sender sends 2 different lookups, but receiver only provides 1 row
    let sender = vec![
        (5u8, 3u8, 1u8),
        (10u8, 6u8, 2u8), // Extra lookup not in receiver
    ];
    // Custom receiver missing the (10, 6) row
    let receiver = vec![(5u8, 3u8, 1u8, 1u32)];

    assert!(!prove_and_verify_custom(&sender, &receiver));
}
