//! Soundness and completeness tests for LT table bus interactions.
//!
//! These tests verify that:
//! - Completeness: Valid lookups to LT are accepted
//! - Soundness: Invalid lookups to LT are rejected
//! - Padding: Auto-padding to power of 2 works correctly
//! - Border cases: Edge values (0, MAX, signed boundaries) work

use stark::constraints::builder::EmptyConstraints;
use std::collections::HashMap;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;

use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::lt::{LtOperation, cols, generate_lt_trace};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::test_utils::{multi_prove_batched_ram, multi_prove_ram};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Signed comparison flag
const SIGNED: bool = true;
/// Unsigned comparison flag
const UNSIGNED: bool = false;

// =============================================================================
// Column indices for sender (CPU-like) table
// =============================================================================

mod sender_cols {
    /// lhs[0]: Word (bits 0-31)
    pub const LHS_0: usize = 0;
    /// lhs[1]: Half (bits 32-47)
    pub const LHS_1: usize = 1;
    /// lhs[2]: Half (bits 48-63)
    pub const LHS_2: usize = 2;
    /// rhs[0]: Word (bits 0-31)
    pub const RHS_0: usize = 3;
    /// rhs[1]: Half (bits 32-47)
    pub const RHS_1: usize = 4;
    /// rhs[2]: Half (bits 48-63)
    pub const RHS_2: usize = 5;
    /// signed flag
    pub const SIGNED: usize = 6;
    /// lt result
    pub const LT: usize = 7;
    /// multiplicity (1 = active row)
    pub const MU: usize = 8;
    pub const NUM_COLUMNS: usize = 9;
}

// =============================================================================
// AIR definitions
// =============================================================================

fn new_sender_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![BusInteraction::sender(
            BusId::Alu,
            Multiplicity::Column(sender_cols::MU),
            vec![
                BusValue::Packed {
                    start_column: sender_cols::LHS_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::LHS_1,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::LHS_2,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::RHS_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::RHS_1,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::RHS_2,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::SIGNED,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::LT,
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
        EmptyConstraints,
    )
}

fn new_receiver_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
    // Use the same bus interaction as the LT table
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![BusInteraction::receiver(
            BusId::Alu,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: cols::LHS_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::LHS_1,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::LHS_2,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::RHS_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::RHS_1,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::RHS_2,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::SIGNED,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::LT,
                    packing: Packing::Direct,
                },
            ],
        )],
    };

    AirWithBuses::new(
        cols::NUM_COLUMNS,
        auxiliary_trace_build_data,
        proof_options,
        1,
        EmptyConstraints,
    )
}

// =============================================================================
// Helper functions
// =============================================================================

/// Create a sender trace from LT operations.
fn create_sender_trace(ops: &[LtOperation]) -> TraceTable<F, E> {
    let num_rows = ops.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * sender_cols::NUM_COLUMNS];

    for (i, op) in ops.iter().enumerate() {
        let base = i * sender_cols::NUM_COLUMNS;

        // Extract lhs as DWordHHW
        let lhs_0 = (op.lhs & 0xFFFF_FFFF) as u32;
        let lhs_1 = ((op.lhs >> 32) & 0xFFFF) as u16;
        let lhs_2 = ((op.lhs >> 48) & 0xFFFF) as u16;

        // Extract rhs as DWordHHW
        let rhs_0 = (op.rhs & 0xFFFF_FFFF) as u32;
        let rhs_1 = ((op.rhs >> 32) & 0xFFFF) as u16;
        let rhs_2 = ((op.rhs >> 48) & 0xFFFF) as u16;

        data[base + sender_cols::LHS_0] = FE::from(lhs_0 as u64);
        data[base + sender_cols::LHS_1] = FE::from(lhs_1 as u64);
        data[base + sender_cols::LHS_2] = FE::from(lhs_2 as u64);
        data[base + sender_cols::RHS_0] = FE::from(rhs_0 as u64);
        data[base + sender_cols::RHS_1] = FE::from(rhs_1 as u64);
        data[base + sender_cols::RHS_2] = FE::from(rhs_2 as u64);
        data[base + sender_cols::SIGNED] = FE::from(if op.signed { 1u64 } else { 0u64 });
        data[base + sender_cols::LT] = FE::from(if op.compute_lt() { 1u64 } else { 0u64 });
        data[base + sender_cols::MU] = FE::one();
    }

    TraceTable::new_main(data, sender_cols::NUM_COLUMNS, 1)
}

/// Create a receiver trace that matches sender operations (for completeness tests).
fn create_receiver_trace(ops: &[LtOperation]) -> TraceTable<F, E> {
    // Count multiplicities for each unique operation
    let mut multiplicities: HashMap<(u64, u64, bool), u32> = HashMap::new();
    for op in ops {
        *multiplicities
            .entry((op.lhs, op.rhs, op.signed))
            .or_insert(0) += 1;
    }

    // Build operations with correct multiplicities
    let unique_ops: Vec<(LtOperation, u32)> = multiplicities
        .into_iter()
        .map(|((lhs, rhs, signed), mult)| (LtOperation::new(lhs, rhs, signed), mult))
        .collect();

    let num_rows = unique_ops.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (i, (op, mult)) in unique_ops.iter().enumerate() {
        let base = i * cols::NUM_COLUMNS;

        // Extract lhs as DWordHHW
        let lhs_0 = (op.lhs & 0xFFFF_FFFF) as u32;
        let lhs_1 = ((op.lhs >> 32) & 0xFFFF) as u16;
        let lhs_2 = ((op.lhs >> 48) & 0xFFFF) as u16;

        // Extract rhs as DWordHHW
        let rhs_0 = (op.rhs & 0xFFFF_FFFF) as u32;
        let rhs_1 = ((op.rhs >> 32) & 0xFFFF) as u16;
        let rhs_2 = ((op.rhs >> 48) & 0xFFFF) as u16;

        // Compute lhs_sub_rhs
        let lhs_sub_rhs = op.lhs.wrapping_sub(op.rhs);
        let sub_0 = (lhs_sub_rhs & 0xFFFF) as u16;
        let sub_1 = ((lhs_sub_rhs >> 16) & 0xFFFF) as u16;
        let sub_2 = ((lhs_sub_rhs >> 32) & 0xFFFF) as u16;
        let sub_3 = ((lhs_sub_rhs >> 48) & 0xFFFF) as u16;

        // Compute MSBs
        let lhs_msb = (op.lhs >> 63) & 1;
        let rhs_msb = (op.rhs >> 63) & 1;

        // Store columns
        data[base + cols::LHS_0] = FE::from(lhs_0 as u64);
        data[base + cols::LHS_1] = FE::from(lhs_1 as u64);
        data[base + cols::LHS_2] = FE::from(lhs_2 as u64);
        data[base + cols::RHS_0] = FE::from(rhs_0 as u64);
        data[base + cols::RHS_1] = FE::from(rhs_1 as u64);
        data[base + cols::RHS_2] = FE::from(rhs_2 as u64);
        data[base + cols::SIGNED] = FE::from(if op.signed { 1u64 } else { 0u64 });
        data[base + cols::LT] = FE::from(if op.compute_lt() { 1u64 } else { 0u64 });
        data[base + cols::LHS_SUB_RHS_0] = FE::from(sub_0 as u64);
        data[base + cols::LHS_SUB_RHS_1] = FE::from(sub_1 as u64);
        data[base + cols::LHS_SUB_RHS_2] = FE::from(sub_2 as u64);
        data[base + cols::LHS_SUB_RHS_3] = FE::from(sub_3 as u64);
        data[base + cols::LHS_MSB] = FE::from(lhs_msb);
        data[base + cols::RHS_MSB] = FE::from(rhs_msb);
        data[base + cols::MU] = FE::from(*mult as u64);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

/// Proves and verifies sender-receiver interaction.
fn prove_and_verify(ops: &[LtOperation]) -> bool {
    let mut sender_trace = create_sender_trace(ops);
    let mut receiver_trace = create_receiver_trace(ops);

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
        multi_prove_batched_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender_air, &receiver_air];

    Verifier::batched_multi_verify(
        &airs,
        &multi_proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    )
}

/// Create a custom receiver trace for soundness tests.
struct CustomLtRow {
    lhs: u64,
    rhs: u64,
    signed: bool,
    lt: bool, // Can be wrong for soundness tests
    multiplicity: u32,
}

fn create_custom_receiver_trace(rows: &[CustomLtRow]) -> TraceTable<F, E> {
    let num_rows = rows.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (i, row) in rows.iter().enumerate() {
        let base = i * cols::NUM_COLUMNS;

        let lhs_0 = (row.lhs & 0xFFFF_FFFF) as u32;
        let lhs_1 = ((row.lhs >> 32) & 0xFFFF) as u16;
        let lhs_2 = ((row.lhs >> 48) & 0xFFFF) as u16;

        let rhs_0 = (row.rhs & 0xFFFF_FFFF) as u32;
        let rhs_1 = ((row.rhs >> 32) & 0xFFFF) as u16;
        let rhs_2 = ((row.rhs >> 48) & 0xFFFF) as u16;

        let lhs_sub_rhs = row.lhs.wrapping_sub(row.rhs);
        let sub_0 = (lhs_sub_rhs & 0xFFFF) as u16;
        let sub_1 = ((lhs_sub_rhs >> 16) & 0xFFFF) as u16;
        let sub_2 = ((lhs_sub_rhs >> 32) & 0xFFFF) as u16;
        let sub_3 = ((lhs_sub_rhs >> 48) & 0xFFFF) as u16;

        let lhs_msb = (row.lhs >> 63) & 1;
        let rhs_msb = (row.rhs >> 63) & 1;

        data[base + cols::LHS_0] = FE::from(lhs_0 as u64);
        data[base + cols::LHS_1] = FE::from(lhs_1 as u64);
        data[base + cols::LHS_2] = FE::from(lhs_2 as u64);
        data[base + cols::RHS_0] = FE::from(rhs_0 as u64);
        data[base + cols::RHS_1] = FE::from(rhs_1 as u64);
        data[base + cols::RHS_2] = FE::from(rhs_2 as u64);
        data[base + cols::SIGNED] = FE::from(if row.signed { 1u64 } else { 0u64 });
        data[base + cols::LT] = FE::from(if row.lt { 1u64 } else { 0u64 });
        data[base + cols::LHS_SUB_RHS_0] = FE::from(sub_0 as u64);
        data[base + cols::LHS_SUB_RHS_1] = FE::from(sub_1 as u64);
        data[base + cols::LHS_SUB_RHS_2] = FE::from(sub_2 as u64);
        data[base + cols::LHS_SUB_RHS_3] = FE::from(sub_3 as u64);
        data[base + cols::LHS_MSB] = FE::from(lhs_msb);
        data[base + cols::RHS_MSB] = FE::from(rhs_msb);
        data[base + cols::MU] = FE::from(row.multiplicity as u64);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

fn prove_and_verify_custom(ops: &[LtOperation], receiver_rows: &[CustomLtRow]) -> bool {
    let mut sender_trace = create_sender_trace(ops);
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

// =============================================================================
// Padding Tests
// =============================================================================

#[test]
fn test_padding_single_row() {
    let ops = vec![LtOperation::new(1, 2, UNSIGNED)];
    let trace = generate_lt_trace(&ops);
    // 1 row -> pads to 4 (minimum for FRI)
    assert_eq!(trace.main_table.height, 4);
}

#[test]
fn test_padding_three_rows() {
    let ops = vec![
        LtOperation::new(1, 2, UNSIGNED),
        LtOperation::new(3, 4, UNSIGNED),
        LtOperation::new(5, 6, UNSIGNED),
    ];
    let trace = generate_lt_trace(&ops);
    // 3 rows -> pads to 4
    assert_eq!(trace.main_table.height, 4);
}

#[test]
fn test_padding_five_rows() {
    let ops = vec![
        LtOperation::new(1, 2, UNSIGNED),
        LtOperation::new(3, 4, UNSIGNED),
        LtOperation::new(5, 6, UNSIGNED),
        LtOperation::new(7, 8, UNSIGNED),
        LtOperation::new(9, 10, UNSIGNED),
    ];
    let trace = generate_lt_trace(&ops);
    // 5 rows -> pads to 8
    assert_eq!(trace.main_table.height, 8);
}

#[test]
fn test_padding_rows_have_zero_multiplicity() {
    let ops = vec![LtOperation::new(100, 200, UNSIGNED)];
    let trace = generate_lt_trace(&ops);

    // First row has MU = 1
    let row0 = trace.main_table.get_row(0);
    assert_eq!(row0[cols::MU], FE::one());

    // Padding row has MU = 0
    let row1 = trace.main_table.get_row(1);
    assert_eq!(row1[cols::MU], FE::zero());
}

// =============================================================================
// Border Case Tests
// =============================================================================

#[test]
fn test_border_zero_values() {
    let ops = vec![
        LtOperation::new(0, 0, UNSIGNED), // 0 < 0 = false
        LtOperation::new(0, 1, UNSIGNED), // 0 < 1 = true
        LtOperation::new(1, 0, UNSIGNED), // 1 < 0 = false
    ];

    assert!(!ops[0].compute_lt());
    assert!(ops[1].compute_lt());
    assert!(!ops[2].compute_lt());

    assert!(prove_and_verify(&ops));
}

#[test]
fn test_border_max_unsigned() {
    let max = u64::MAX;
    let ops = vec![
        LtOperation::new(max, max, UNSIGNED),     // MAX < MAX = false
        LtOperation::new(max - 1, max, UNSIGNED), // MAX-1 < MAX = true
        LtOperation::new(max, max - 1, UNSIGNED), // MAX < MAX-1 = false
        LtOperation::new(max, 0, UNSIGNED),       // MAX < 0 = false (unsigned)
        LtOperation::new(0, max, UNSIGNED),       // 0 < MAX = true (unsigned)
    ];

    assert!(!ops[0].compute_lt());
    assert!(ops[1].compute_lt());
    assert!(!ops[2].compute_lt());
    assert!(!ops[3].compute_lt());
    assert!(ops[4].compute_lt());

    assert!(prove_and_verify(&ops));
}

#[test]
fn test_border_signed_boundaries() {
    let min_signed = i64::MIN as u64; // 0x8000_0000_0000_0000
    let max_signed = i64::MAX as u64; // 0x7FFF_FFFF_FFFF_FFFF

    let ops = vec![
        LtOperation::new(min_signed, max_signed, SIGNED), // MIN < MAX = true
        LtOperation::new(max_signed, min_signed, SIGNED), // MAX < MIN = false
        LtOperation::new(min_signed, min_signed, SIGNED), // MIN < MIN = false
        LtOperation::new(max_signed, max_signed, SIGNED), // MAX < MAX = false
        LtOperation::new((-1i64) as u64, 0, SIGNED),      // -1 < 0 = true
        LtOperation::new(0, (-1i64) as u64, SIGNED),      // 0 < -1 = false
    ];

    assert!(ops[0].compute_lt());
    assert!(!ops[1].compute_lt());
    assert!(!ops[2].compute_lt());
    assert!(!ops[3].compute_lt());
    assert!(ops[4].compute_lt());
    assert!(!ops[5].compute_lt());

    assert!(prove_and_verify(&ops));
}

#[test]
fn test_border_32bit_boundary() {
    // Test around 2^32 boundary
    let boundary = 1u64 << 32;
    let ops = vec![
        LtOperation::new(boundary - 1, boundary, UNSIGNED), // 2^32-1 < 2^32 = true
        LtOperation::new(boundary, boundary - 1, UNSIGNED), // 2^32 < 2^32-1 = false
        LtOperation::new(boundary, boundary, UNSIGNED),     // 2^32 < 2^32 = false
    ];

    assert!(ops[0].compute_lt());
    assert!(!ops[1].compute_lt());
    assert!(!ops[2].compute_lt());

    assert!(prove_and_verify(&ops));
}

// =============================================================================
// Completeness Tests - Valid lookups should be ACCEPTED
// =============================================================================

#[test]
fn test_completeness_simple_unsigned() {
    let ops = vec![LtOperation::new(5, 10, UNSIGNED)]; // 5 < 10 = true
    assert!(prove_and_verify(&ops));
}

#[test]
fn test_completeness_simple_signed() {
    let ops = vec![LtOperation::new((-5i64) as u64, 5, SIGNED)]; // -5 < 5 = true
    assert!(prove_and_verify(&ops));
}

#[test]
fn test_completeness_multiple_lookups() {
    let ops = vec![
        LtOperation::new(1, 2, UNSIGNED),
        LtOperation::new(100, 50, UNSIGNED),
        LtOperation::new((-10i64) as u64, (-5i64) as u64, SIGNED),
        LtOperation::new(0, 0, UNSIGNED),
    ];
    assert!(prove_and_verify(&ops));
}

#[test]
fn test_completeness_duplicate_lookups() {
    let ops = vec![
        LtOperation::new(42, 100, UNSIGNED),
        LtOperation::new(42, 100, UNSIGNED), // Duplicate
        LtOperation::new(42, 100, UNSIGNED), // Duplicate
    ];
    assert!(prove_and_verify(&ops));
}

// =============================================================================
// Soundness Tests - Invalid lookups should be REJECTED
// =============================================================================

#[test]
fn test_soundness_wrong_lt_result() {
    // Sender claims 5 < 10 = false (WRONG! Should be true)
    let ops = vec![LtOperation::new(5, 10, UNSIGNED)];

    // Custom receiver with wrong LT result
    let receiver = vec![CustomLtRow {
        lhs: 5,
        rhs: 10,
        signed: UNSIGNED,
        lt: false, // WRONG - should be true
        multiplicity: 1,
    }];

    assert!(!prove_and_verify_custom(&ops, &receiver));
}

#[test]
fn test_soundness_wrong_signed_result() {
    // Sender: -5 < 5 (signed) = true
    let lhs = (-5i64) as u64;
    let ops = vec![LtOperation::new(lhs, 5, SIGNED)];

    // Custom receiver claims false
    let receiver = vec![CustomLtRow {
        lhs,
        rhs: 5,
        signed: SIGNED,
        lt: false, // WRONG - should be true
        multiplicity: 1,
    }];

    assert!(!prove_and_verify_custom(&ops, &receiver));
}

#[test]
fn test_soundness_multiplicity_mismatch() {
    // Sender sends 2 lookups, receiver has multiplicity 1
    let ops = vec![
        LtOperation::new(5, 10, UNSIGNED),
        LtOperation::new(5, 10, UNSIGNED), // Duplicate
    ];

    let receiver = vec![CustomLtRow {
        lhs: 5,
        rhs: 10,
        signed: UNSIGNED,
        lt: true,
        multiplicity: 1, // WRONG - should be 2
    }];

    assert!(!prove_and_verify_custom(&ops, &receiver));
}

#[test]
fn test_soundness_missing_receiver_row() {
    // Sender looks up (100, 200), receiver only has (5, 10)
    let ops = vec![LtOperation::new(100, 200, UNSIGNED)];

    let receiver = vec![CustomLtRow {
        lhs: 5,
        rhs: 10,
        signed: UNSIGNED,
        lt: true,
        multiplicity: 1,
    }];

    assert!(!prove_and_verify_custom(&ops, &receiver));
}

#[test]
fn test_soundness_swapped_operands() {
    // Sender: 5 < 10, Receiver has 10 < 5
    let ops = vec![LtOperation::new(5, 10, UNSIGNED)];

    let receiver = vec![CustomLtRow {
        lhs: 10, // Swapped!
        rhs: 5,  // Swapped!
        signed: UNSIGNED,
        lt: false, // 10 < 5 = false (correct for swapped)
        multiplicity: 1,
    }];

    assert!(!prove_and_verify_custom(&ops, &receiver));
}

#[test]
fn test_soundness_wrong_signed_flag() {
    // Sender: unsigned comparison, receiver has signed
    let ops = vec![LtOperation::new(5, 10, UNSIGNED)];

    let receiver = vec![CustomLtRow {
        lhs: 5,
        rhs: 10,
        signed: SIGNED, // WRONG - should be UNSIGNED
        lt: true,
        multiplicity: 1,
    }];

    assert!(!prove_and_verify_custom(&ops, &receiver));
}
