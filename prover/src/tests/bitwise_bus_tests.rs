//! Soundness and completeness tests for BITWISE table bus interactions.
//!
//! These tests verify that:
//! - Completeness: Valid lookups to BITWISE are accepted
//! - Soundness: Invalid lookups to BITWISE are rejected

use std::collections::HashMap;

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

use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

type F = GoldilocksField;
type E = GoldilocksExtension;

// =============================================================================
// Column indices (simplified for testing)
// =============================================================================

/// Sender table columns
mod sender_cols {
    pub const X: usize = 0;
    pub const Y: usize = 1;
    pub const RESULT: usize = 2;
    /// Flag indicating this row performs a bitwise operation (0 or 1)
    pub const FLAG: usize = 3;
    /// Opcode: 0=AND, 1=OR, 2=XOR
    pub const OPCODE: usize = 4;
    pub const NUM_COLUMNS: usize = 5;
}

/// Receiver (BITWISE-like) table columns
mod receiver_cols {
    pub const X: usize = 0;
    pub const Y: usize = 1;
    /// Opcode (Z column): 0=AND, 1=OR, 2=XOR
    pub const OPCODE: usize = 2;
    /// Precomputed result: AND when opcode=0, OR when opcode=1, XOR when opcode=2
    pub const BITWISE_RESULT: usize = 3;
    pub const MU: usize = 4;
    pub const NUM_COLUMNS: usize = 5;
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
            BusId::BitwiseByte,
            Multiplicity::Column(sender_cols::FLAG),
            vec![
                BusValue::Packed {
                    start_column: sender_cols::OPCODE,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::Y,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::RESULT,
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
            BusId::BitwiseByte,
            Multiplicity::Column(receiver_cols::MU),
            vec![
                BusValue::Packed {
                    start_column: receiver_cols::OPCODE,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: receiver_cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: receiver_cols::Y,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: receiver_cols::BITWISE_RESULT,
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

/// Creates a sender trace. Each tuple is (opcode, x, y, result).
fn create_sender_trace(lookups: &[(u8, u8, u8, u8)]) -> TraceTable<F, E> {
    let num_rows = lookups.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * sender_cols::NUM_COLUMNS];

    for (i, &(opcode, x, y, result)) in lookups.iter().enumerate() {
        let base = i * sender_cols::NUM_COLUMNS;
        data[base + sender_cols::X] = FE::from(x as u64);
        data[base + sender_cols::Y] = FE::from(y as u64);
        data[base + sender_cols::RESULT] = FE::from(result as u64);
        data[base + sender_cols::FLAG] = FE::one();
        data[base + sender_cols::OPCODE] = FE::from(opcode as u64);
    }

    TraceTable::new_main(data, sender_cols::NUM_COLUMNS, 1)
}

/// Creates a receiver (BITWISE-like) trace with precomputed values.
///
/// This function mimics the real BITWISE table behavior:
/// 1. Precomputes the result for each unique (opcode, X, Y) combination
/// 2. Updates multiplicities based on how many times each tuple is received from sender
///
/// # Arguments
/// * `sender_lookups` - Tuples (opcode, X, Y, claimed_result) sent by the sender table
fn create_receiver_trace(sender_lookups: &[(u8, u8, u8, u8)]) -> TraceTable<F, E> {
    // Count multiplicities for each unique (opcode, X, Y) triple from sender lookups
    let mut multiplicities: HashMap<(u8, u8, u8), u32> = HashMap::new();
    for &(opcode, x, y, _) in sender_lookups {
        *multiplicities.entry((opcode, x, y)).or_insert(0) += 1;
    }

    // Build precomputed table with results and multiplicities
    let unique_rows: Vec<((u8, u8, u8), u32)> = multiplicities.into_iter().collect();
    let num_rows = unique_rows.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * receiver_cols::NUM_COLUMNS];

    for (i, &((opcode, x, y), multiplicity)) in unique_rows.iter().enumerate() {
        let base = i * receiver_cols::NUM_COLUMNS;
        let result = match opcode {
            0 => x & y,
            1 => x | y,
            2 => x ^ y,
            _ => panic!("invalid opcode"),
        };
        data[base + receiver_cols::X] = FE::from(x as u64);
        data[base + receiver_cols::Y] = FE::from(y as u64);
        data[base + receiver_cols::OPCODE] = FE::from(opcode as u64);
        data[base + receiver_cols::BITWISE_RESULT] = FE::from(result as u64);
        data[base + receiver_cols::MU] = FE::from(multiplicity as u64);
    }

    TraceTable::new_main(data, receiver_cols::NUM_COLUMNS, 1)
}

/// Proves and verifies a sender-receiver interaction.
///
/// The receiver trace is automatically built from sender lookups, mimicking
/// how the BITWISE table works: precomputed values with multiplicities
/// updated based on received lookups.
fn prove_and_verify(sender_lookups: &[(u8, u8, u8, u8)]) -> bool {
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
    // Sender: BitwiseByte[AND, 5, 3] = 1 (correct: 5 & 3 = 1)
    let sender = vec![(0u8, 5u8, 3u8, 1u8)];

    assert!(prove_and_verify(&sender));
}

#[test]
fn test_completeness_or_byte_simple() {
    // Sender: BitwiseByte[OR, 5, 3] = 7 (correct: 5 | 3 = 7)
    let sender = vec![(1u8, 5u8, 3u8, 7u8)];

    assert!(prove_and_verify(&sender));
}

#[test]
fn test_completeness_xor_byte_simple() {
    // Sender: BitwiseByte[XOR, 5, 3] = 6 (correct: 5 ^ 3 = 6)
    let sender = vec![(2u8, 5u8, 3u8, 6u8)];

    assert!(prove_and_verify(&sender));
}

#[test]
fn test_completeness_and_byte_zero_result() {
    // 0xAA & 0x55 = 0 (alternating bits)
    let sender = vec![(0u8, 0xAAu8, 0x55u8, 0x00u8)];

    assert!(prove_and_verify(&sender));
}

#[test]
fn test_completeness_and_byte_max() {
    // 0xFF & 0xFF = 0xFF
    let sender = vec![(0u8, 0xFFu8, 0xFFu8, 0xFFu8)];

    assert!(prove_and_verify(&sender));
}

#[test]
fn test_completeness_multiple_lookups() {
    // Multiple different lookups with different opcodes
    let sender = vec![
        (0u8, 0xFFu8, 0xFFu8, 0xFFu8), // AND: FF & FF = FF
        (1u8, 0xAAu8, 0x55u8, 0xFFu8), // OR:  AA | 55 = FF
        (2u8, 0x0Fu8, 0xF0u8, 0xFFu8), // XOR: 0F ^ F0 = FF
        (0u8, 0x12u8, 0x34u8, 0x10u8), // AND: 12 & 34 = 10
    ];

    assert!(prove_and_verify(&sender));
}

#[test]
fn test_completeness_duplicate_lookups() {
    // Same lookup multiple times - receiver precomputes once with multiplicity = 3
    let sender = vec![
        (0u8, 5u8, 3u8, 1u8), // First lookup
        (0u8, 5u8, 3u8, 1u8), // Duplicate
        (0u8, 5u8, 3u8, 1u8), // Duplicate
    ];

    assert!(prove_and_verify(&sender));
}

// =============================================================================
// Soundness Tests - Invalid lookups should be REJECTED
// =============================================================================

/// Creates a custom receiver trace for soundness tests.
/// Allows specifying exact (opcode, X, Y, result, multiplicity) tuples.
fn create_custom_receiver_trace(rows: &[(u8, u8, u8, u8, u32)]) -> TraceTable<F, E> {
    let num_rows = rows.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * receiver_cols::NUM_COLUMNS];

    for (i, &(opcode, x, y, result, multiplicity)) in rows.iter().enumerate() {
        let base = i * receiver_cols::NUM_COLUMNS;
        data[base + receiver_cols::X] = FE::from(x as u64);
        data[base + receiver_cols::Y] = FE::from(y as u64);
        data[base + receiver_cols::OPCODE] = FE::from(opcode as u64);
        data[base + receiver_cols::BITWISE_RESULT] = FE::from(result as u64);
        data[base + receiver_cols::MU] = FE::from(multiplicity as u64);
    }

    TraceTable::new_main(data, receiver_cols::NUM_COLUMNS, 1)
}

/// Proves and verifies with a custom receiver trace (for soundness tests).
fn prove_and_verify_custom(
    sender_lookups: &[(u8, u8, u8, u8)],
    receiver_rows: &[(u8, u8, u8, u8, u32)],
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
        Prover::multi_prove(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender_air, &receiver_air];

    Verifier::multi_verify(&airs, &multi_proof, &mut DefaultTranscript::<E>::new(&[]))
}

#[test]
fn test_soundness_wrong_result() {
    // Sender claims BitwiseByte[AND, 5, 3] = 99 (WRONG! Should be 1)
    // Receiver has precomputed correct value 1, so verification should fail
    let sender = vec![(0u8, 5u8, 3u8, 99u8)];

    assert!(!prove_and_verify(&sender));
}

#[test]
fn test_soundness_off_by_one() {
    // Sender claims BitwiseByte[AND, 0xFF, 0xFF] = 0xFE (WRONG! Should be 0xFF)
    let sender = vec![(0u8, 0xFFu8, 0xFFu8, 0xFEu8)];

    assert!(!prove_and_verify(&sender));
}

#[test]
fn test_soundness_wrong_opcode() {
    // Sender claims AND result but with OR opcode - should fail
    let sender = vec![(1u8, 5u8, 3u8, 1u8)]; // opcode=OR, but result=1 (AND result, not OR)

    assert!(!prove_and_verify(&sender));
}

#[test]
fn test_soundness_multiplicity_mismatch() {
    // Sender sends 2 lookups for same value, but receiver has multiplicity 1
    let sender = vec![
        (0u8, 5u8, 3u8, 1u8),
        (0u8, 5u8, 3u8, 1u8), // Duplicate!
    ];
    // Custom receiver with wrong multiplicity (1 instead of 2)
    let receiver = vec![(0u8, 5u8, 3u8, 1u8, 1u32)]; // multiplicity = 1, should be 2

    assert!(!prove_and_verify_custom(&sender, &receiver));
}

#[test]
fn test_soundness_missing_receiver_row() {
    // Sender looks up (AND, 100, 200), but receiver only has (AND, 5, 3)
    let sender = vec![(0u8, 100u8, 200u8, 64u8)]; // 100 & 200 = 64
    // Custom receiver with different row
    let receiver = vec![(0u8, 5u8, 3u8, 1u8, 1u32)]; // Different row!

    assert!(!prove_and_verify_custom(&sender, &receiver));
}

#[test]
fn test_soundness_swapped_inputs() {
    // Sender: BitwiseByte[AND, 3, 5] = 1
    // Receiver: has (AND, 5, 3) not (AND, 3, 5) - order matters!
    let sender = vec![(0u8, 3u8, 5u8, 1u8)]; // Note: X=3, Y=5
    // Custom receiver with swapped inputs
    let receiver = vec![(0u8, 5u8, 3u8, 1u8, 1u32)]; // X=5, Y=3 - different input order

    assert!(!prove_and_verify_custom(&sender, &receiver));
}

#[test]
fn test_soundness_extra_sender_lookup() {
    // Sender sends 2 different lookups, but receiver only provides 1 row
    let sender = vec![
        (0u8, 5u8, 3u8, 1u8),
        (0u8, 10u8, 6u8, 2u8), // Extra lookup not in receiver
    ];
    // Custom receiver missing the (AND, 10, 6) row
    let receiver = vec![(0u8, 5u8, 3u8, 1u8, 1u32)];

    assert!(!prove_and_verify_custom(&sender, &receiver));
}
