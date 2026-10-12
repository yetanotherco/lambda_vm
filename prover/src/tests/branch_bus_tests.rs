//! Soundness and completeness tests for BRANCH table bus interactions.
//!
//! These tests verify that:
//! - Completeness: Valid lookups to BRANCH are accepted
//! - Soundness: Invalid lookups to BRANCH are rejected
//! - Padding: Auto-padding to power of 2 works correctly
//! - Border cases: Edge values (0, MAX, signed boundaries) work

use math::field::element::FieldElement;
use stark::constraints::builder::EmptyConstraints;
use std::collections::HashMap;

use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, LinearTerm, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::IsStarkVerifier;

use crate::tables::branch::{BranchOperation, cols, generate_branch_trace};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::test_utils::multi_prove_ram;

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Constants for shifts
const SHIFT_8: u64 = 1 << 8;
const SHIFT_16: u64 = 1 << 16;

// =============================================================================
// Column indices for sender (CPU-like) table
// =============================================================================

mod sender_cols {
    /// next_pc[0]: Word (bits 0-31)
    pub const NEXT_PC_0: usize = 0;
    /// next_pc[1]: Word (bits 32-63)
    pub const NEXT_PC_1: usize = 1;
    /// pc[0]: Word (bits 0-31)
    pub const PC_0: usize = 2;
    /// pc[1]: Word (bits 32-63)
    pub const PC_1: usize = 3;
    /// offset[0]: Word (bits 0-31)
    pub const OFFSET_0: usize = 4;
    /// offset[1]: Word (bits 32-63)
    pub const OFFSET_1: usize = 5;
    /// register[0]: Word (bits 0-31)
    pub const REGISTER_0: usize = 6;
    /// register[1]: Word (bits 32-63)
    pub const REGISTER_1: usize = 7;
    /// JALR flag
    pub const JALR: usize = 8;
    /// multiplicity (1 = active row)
    pub const MU: usize = 9;
    pub const NUM_COLUMNS: usize = 10;
}

// =============================================================================
// AIR definitions
// =============================================================================

fn new_sender_air(
    proof_options: &ProofOptions,
) -> AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints> {
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![BusInteraction::sender(
            BusId::Branch,
            Multiplicity::Column(sender_cols::MU),
            vec![
                // next_pc as DWordWL (2 words)
                BusValue::Packed {
                    start_column: sender_cols::NEXT_PC_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::NEXT_PC_1,
                    packing: Packing::Direct,
                },
                // pc as DWordWL
                BusValue::Packed {
                    start_column: sender_cols::PC_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::PC_1,
                    packing: Packing::Direct,
                },
                // offset as DWordWL
                BusValue::Packed {
                    start_column: sender_cols::OFFSET_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::OFFSET_1,
                    packing: Packing::Direct,
                },
                // register as DWordWL
                BusValue::Packed {
                    start_column: sender_cols::REGISTER_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sender_cols::REGISTER_1,
                    packing: Packing::Direct,
                },
                // JALR flag
                BusValue::Packed {
                    start_column: sender_cols::JALR,
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
    // Use the same bus interaction format as the BRANCH table receiver
    let auxiliary_trace_build_data = AuxiliaryTraceBuildData {
        interactions: vec![BusInteraction::receiver(
            BusId::Branch,
            Multiplicity::Column(cols::MU),
            vec![
                // next_pc as DWordWL (computed from high/low columns)
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::NEXT_PC_LOW_0,
                    },
                    LinearTerm::Column {
                        coefficient: SHIFT_8 as i64,
                        column: cols::NEXT_PC_LOW_1,
                    },
                    LinearTerm::Column {
                        coefficient: SHIFT_16 as i64,
                        column: cols::NEXT_PC_HIGH_0,
                    },
                ]),
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::NEXT_PC_HIGH_1,
                    },
                    LinearTerm::Column {
                        coefficient: SHIFT_16 as i64,
                        column: cols::NEXT_PC_HIGH_2,
                    },
                ]),
                // pc as DWordWL
                BusValue::Packed {
                    start_column: cols::PC_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::PC_1,
                    packing: Packing::Direct,
                },
                // offset as DWordWL
                BusValue::Packed {
                    start_column: cols::OFFSET_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::OFFSET_1,
                    packing: Packing::Direct,
                },
                // register as DWordWL
                BusValue::Packed {
                    start_column: cols::REGISTER_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::REGISTER_1,
                    packing: Packing::Direct,
                },
                // JALR flag
                BusValue::Packed {
                    start_column: cols::JALR,
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

/// Create a sender trace from BRANCH operations.
fn create_sender_trace(ops: &[BranchOperation]) -> TraceTable<F, E> {
    let num_rows = ops.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * sender_cols::NUM_COLUMNS];

    for (i, op) in ops.iter().enumerate() {
        let base = i * sender_cols::NUM_COLUMNS;

        // Compute next_pc
        let next_pc = op.compute_next_pc();
        let next_pc_0 = (next_pc & 0xFFFF_FFFF) as u32;
        let next_pc_1 = (next_pc >> 32) as u32;

        // Extract pc as DWordWL
        let pc_0 = (op.pc & 0xFFFF_FFFF) as u32;
        let pc_1 = (op.pc >> 32) as u32;

        // Extract offset as DWordWL
        let offset_0 = (op.offset & 0xFFFF_FFFF) as u32;
        let offset_1 = (op.offset >> 32) as u32;

        // Extract register as DWordWL
        let register_0 = (op.register & 0xFFFF_FFFF) as u32;
        let register_1 = (op.register >> 32) as u32;

        data[base + sender_cols::NEXT_PC_0] = FE::from(next_pc_0 as u64);
        data[base + sender_cols::NEXT_PC_1] = FE::from(next_pc_1 as u64);
        data[base + sender_cols::PC_0] = FE::from(pc_0 as u64);
        data[base + sender_cols::PC_1] = FE::from(pc_1 as u64);
        data[base + sender_cols::OFFSET_0] = FE::from(offset_0 as u64);
        data[base + sender_cols::OFFSET_1] = FE::from(offset_1 as u64);
        data[base + sender_cols::REGISTER_0] = FE::from(register_0 as u64);
        data[base + sender_cols::REGISTER_1] = FE::from(register_1 as u64);
        data[base + sender_cols::JALR] = FE::from(if op.jalr { 1u64 } else { 0u64 });
        data[base + sender_cols::MU] = FE::one();
    }

    TraceTable::new_main(data, sender_cols::NUM_COLUMNS, 1)
}

/// Create a receiver trace that matches sender operations (for completeness tests).
fn create_receiver_trace(ops: &[BranchOperation]) -> TraceTable<F, E> {
    // Count multiplicities for each unique operation
    let mut multiplicities: HashMap<(u64, u64, u64, bool), u32> = HashMap::new();
    for op in ops {
        *multiplicities
            .entry((op.pc, op.offset, op.register, op.jalr))
            .or_insert(0) += 1;
    }

    // Build operations with correct multiplicities
    let unique_ops: Vec<(BranchOperation, u32)> = multiplicities
        .into_iter()
        .map(|((pc, offset, register, jalr), mult)| {
            (BranchOperation::new(pc, offset, register, jalr), mult)
        })
        .collect();

    let num_rows = unique_ops.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (i, (op, mult)) in unique_ops.iter().enumerate() {
        let base = i * cols::NUM_COLUMNS;

        // Extract pc as DWordWL
        let pc_0 = (op.pc & 0xFFFF_FFFF) as u32;
        let pc_1 = (op.pc >> 32) as u32;

        // Extract offset as DWordWL
        let offset_0 = (op.offset & 0xFFFF_FFFF) as u32;
        let offset_1 = (op.offset >> 32) as u32;

        // Extract register as DWordWL
        let register_0 = (op.register & 0xFFFF_FFFF) as u32;
        let register_1 = (op.register >> 32) as u32;

        // Compute next_pc
        let next_pc_unmasked = op.compute_next_pc_unmasked();
        let next_pc = op.compute_next_pc();

        // Extract next_pc components
        let unmasked_low_byte = (next_pc_unmasked & 0xFF) as u8;
        let next_pc_low_0 = (next_pc & 0xFF) as u8;
        let next_pc_low_1 = ((next_pc >> 8) & 0xFF) as u8;
        let next_pc_high_0 = ((next_pc >> 16) & 0xFFFF) as u16;
        let next_pc_high_1 = ((next_pc >> 32) & 0xFFFF) as u16;
        let next_pc_high_2 = ((next_pc >> 48) & 0xFFFF) as u16;

        // Store columns
        data[base + cols::PC_0] = FE::from(pc_0 as u64);
        data[base + cols::PC_1] = FE::from(pc_1 as u64);
        data[base + cols::OFFSET_0] = FE::from(offset_0 as u64);
        data[base + cols::OFFSET_1] = FE::from(offset_1 as u64);
        data[base + cols::REGISTER_0] = FE::from(register_0 as u64);
        data[base + cols::REGISTER_1] = FE::from(register_1 as u64);
        data[base + cols::JALR] = FE::from(if op.jalr { 1u64 } else { 0u64 });
        data[base + cols::NEXT_PC_HIGH_0] = FE::from(next_pc_high_0 as u64);
        data[base + cols::NEXT_PC_HIGH_1] = FE::from(next_pc_high_1 as u64);
        data[base + cols::NEXT_PC_HIGH_2] = FE::from(next_pc_high_2 as u64);
        data[base + cols::NEXT_PC_LOW_0] = FE::from(next_pc_low_0 as u64);
        data[base + cols::NEXT_PC_LOW_1] = FE::from(next_pc_low_1 as u64);
        data[base + cols::UNMASKED_LOW_BYTE] = FE::from(unmasked_low_byte as u64);
        data[base + cols::MU] = FE::from(*mult as u64);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

/// Proves and verifies sender-receiver interaction.
fn prove_and_verify(ops: &[BranchOperation]) -> bool {
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
        multi_prove_ram(air_trace_pairs, &mut crate::hash_pin::block_transcript(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender_air, &receiver_air];

    crate::hash_pin::BlockVerifier::multi_verify(
        &airs,
        &multi_proof,
        &mut crate::hash_pin::block_transcript(&[]),
        &FieldElement::zero(),
    )
}

/// Create a custom receiver trace for soundness tests.
struct CustomBranchRow {
    pc: u64,
    offset: u64,
    register: u64,
    jalr: bool,
    next_pc: u64, // Can be wrong for soundness tests
    multiplicity: u32,
}

fn create_custom_receiver_trace(rows: &[CustomBranchRow]) -> TraceTable<F, E> {
    let num_rows = rows.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (i, row) in rows.iter().enumerate() {
        let base = i * cols::NUM_COLUMNS;

        let pc_0 = (row.pc & 0xFFFF_FFFF) as u32;
        let pc_1 = (row.pc >> 32) as u32;

        let offset_0 = (row.offset & 0xFFFF_FFFF) as u32;
        let offset_1 = (row.offset >> 32) as u32;

        let register_0 = (row.register & 0xFFFF_FFFF) as u32;
        let register_1 = (row.register >> 32) as u32;

        // Use the (possibly wrong) next_pc value
        let next_pc = row.next_pc;

        // For unmasked_low_byte, compute what it should be for correct trace
        let op = BranchOperation::new(row.pc, row.offset, row.register, row.jalr);
        let next_pc_unmasked = op.compute_next_pc_unmasked();
        let unmasked_low_byte = (next_pc_unmasked & 0xFF) as u8;

        let next_pc_low_0 = (next_pc & 0xFF) as u8;
        let next_pc_low_1 = ((next_pc >> 8) & 0xFF) as u8;
        let next_pc_high_0 = ((next_pc >> 16) & 0xFFFF) as u16;
        let next_pc_high_1 = ((next_pc >> 32) & 0xFFFF) as u16;
        let next_pc_high_2 = ((next_pc >> 48) & 0xFFFF) as u16;

        data[base + cols::PC_0] = FE::from(pc_0 as u64);
        data[base + cols::PC_1] = FE::from(pc_1 as u64);
        data[base + cols::OFFSET_0] = FE::from(offset_0 as u64);
        data[base + cols::OFFSET_1] = FE::from(offset_1 as u64);
        data[base + cols::REGISTER_0] = FE::from(register_0 as u64);
        data[base + cols::REGISTER_1] = FE::from(register_1 as u64);
        data[base + cols::JALR] = FE::from(if row.jalr { 1u64 } else { 0u64 });
        data[base + cols::NEXT_PC_HIGH_0] = FE::from(next_pc_high_0 as u64);
        data[base + cols::NEXT_PC_HIGH_1] = FE::from(next_pc_high_1 as u64);
        data[base + cols::NEXT_PC_HIGH_2] = FE::from(next_pc_high_2 as u64);
        data[base + cols::NEXT_PC_LOW_0] = FE::from(next_pc_low_0 as u64);
        data[base + cols::NEXT_PC_LOW_1] = FE::from(next_pc_low_1 as u64);
        data[base + cols::UNMASKED_LOW_BYTE] = FE::from(unmasked_low_byte as u64);
        data[base + cols::MU] = FE::from(row.multiplicity as u64);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

fn prove_and_verify_custom(ops: &[BranchOperation], receiver_rows: &[CustomBranchRow]) -> bool {
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
        multi_prove_ram(air_trace_pairs, &mut crate::hash_pin::block_transcript(&[])).unwrap();

    let airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> =
        vec![&sender_air, &receiver_air];

    crate::hash_pin::BlockVerifier::multi_verify(
        &airs,
        &multi_proof,
        &mut crate::hash_pin::block_transcript(&[]),
        &FieldElement::zero(),
    )
}

// =============================================================================
// Padding Tests
// =============================================================================

#[test]
fn test_padding_single_row() {
    let ops = vec![BranchOperation::new(0x1000, 4, 0, false)];
    let trace = generate_branch_trace(&ops);
    // 1 row -> pads to 4 (minimum for FRI)
    assert_eq!(trace.main_table.height, 4);
}

#[test]
fn test_padding_three_rows() {
    let ops = vec![
        BranchOperation::new(0x1000, 4, 0, false),
        BranchOperation::new(0x1004, 8, 0, false),
        BranchOperation::new(0x1008, 12, 0, false),
    ];
    let trace = generate_branch_trace(&ops);
    // 3 rows -> pads to 4
    assert_eq!(trace.main_table.height, 4);
}

#[test]
fn test_padding_five_rows() {
    let ops = vec![
        BranchOperation::new(0x1000, 4, 0, false),
        BranchOperation::new(0x1004, 8, 0, false),
        BranchOperation::new(0x1008, 12, 0, false),
        BranchOperation::new(0x100C, 16, 0, false),
        BranchOperation::new(0x1010, 20, 0, false),
    ];
    let trace = generate_branch_trace(&ops);
    // 5 rows -> pads to 8
    assert_eq!(trace.main_table.height, 8);
}

#[test]
fn test_padding_rows_have_zero_multiplicity() {
    let ops = vec![BranchOperation::new(0x1000, 4, 0, false)];
    let trace = generate_branch_trace(&ops);

    // Check that padding rows have mu = 0
    for row_idx in 1..4 {
        assert_eq!(*trace.get_main(row_idx, cols::MU), FE::zero());
    }
}

// =============================================================================
// Border Case Tests
// =============================================================================

#[test]
fn test_border_zero_values() {
    // pc = 0, offset = 0, not JALR -> next_pc = 0
    let ops = vec![BranchOperation::new(0, 0, 0, false)];
    assert!(prove_and_verify(&ops));
}

#[test]
fn test_border_max_pc() {
    // Large PC value with small positive offset
    let ops = vec![BranchOperation::new(0xFFFF_FFFF_FFFF_FFF0, 4, 0, false)];
    assert!(prove_and_verify(&ops));
}

#[test]
fn test_border_negative_offset() {
    // PC + negative offset (sign-extended to 64-bit)
    let negative_4: u64 = (-4i64) as u64; // 0xFFFFFFFF_FFFFFFFC
    let ops = vec![BranchOperation::new(0x1000, negative_4, 0, false)];
    // Expected: next_pc = 0x1000 + (-4) = 0x0FFC, masked to even = 0x0FFC
    assert!(prove_and_verify(&ops));
}

#[test]
fn test_border_jalr_operation() {
    // JALR: uses register as base instead of pc
    let ops = vec![BranchOperation::new(0x1000, 8, 0x2000, true)];
    // Expected: next_pc = 0x2000 + 8 = 0x2008, masked to even = 0x2008
    assert!(prove_and_verify(&ops));
}

#[test]
fn test_border_jalr_with_negative_offset() {
    // JALR with negative offset (sign-extended to 64-bit)
    let negative_16: u64 = (-16i64) as u64;
    let ops = vec![BranchOperation::new(0x1000, negative_16, 0x3000, true)];
    // Expected: next_pc = 0x3000 + (-16) = 0x2FF0, masked to even = 0x2FF0
    assert!(prove_and_verify(&ops));
}

#[test]
fn test_border_odd_result_gets_masked() {
    // Verify LSB masking: offset = 5 should result in even next_pc
    let ops = vec![BranchOperation::new(0x1000, 5, 0, false)];
    // Expected: unmasked = 0x1005, masked = 0x1004
    let op = &ops[0];
    let next_pc = op.compute_next_pc();
    assert_eq!(next_pc & 1, 0, "LSB should be masked to 0");
    assert!(prove_and_verify(&ops));
}

// =============================================================================
// Completeness Tests (valid lookups should be accepted)
// =============================================================================

#[test]
fn test_completeness_simple_branch() {
    let ops = vec![BranchOperation::new(0x1000, 0x100, 0, false)];
    assert!(prove_and_verify(&ops));
}

#[test]
fn test_completeness_simple_jalr() {
    let ops = vec![BranchOperation::new(0x1000, 0x20, 0x5000, true)];
    assert!(prove_and_verify(&ops));
}

#[test]
fn test_completeness_multiple_lookups() {
    let ops = vec![
        BranchOperation::new(0x1000, 4, 0, false),
        BranchOperation::new(0x2000, 8, 0, false),
        BranchOperation::new(0x3000, 12, 0x4000, true),
    ];
    assert!(prove_and_verify(&ops));
}

#[test]
fn test_completeness_duplicate_lookups() {
    // Same operation multiple times
    let ops = vec![
        BranchOperation::new(0x1000, 100, 0, false),
        BranchOperation::new(0x1000, 100, 0, false),
        BranchOperation::new(0x1000, 100, 0, false),
    ];
    assert!(prove_and_verify(&ops));
}

#[test]
fn test_completeness_mixed_jalr_and_branch() {
    let ops = vec![
        BranchOperation::new(0x1000, 16, 0, false),     // branch
        BranchOperation::new(0x2000, 32, 0x8000, true), // jalr
        BranchOperation::new(0x3000, (-8i64) as u64, 0, false), // branch with negative offset
        BranchOperation::new(0x4000, (-16i64) as u64, 0xA000, true), // jalr with negative offset
    ];
    assert!(prove_and_verify(&ops));
}

// =============================================================================
// Soundness Tests (invalid lookups should be rejected)
// =============================================================================

#[test]
fn test_soundness_wrong_next_pc() {
    // Sender expects next_pc = 0x1004, but receiver has wrong value
    let ops = vec![BranchOperation::new(0x1000, 4, 0, false)];

    let receiver_rows = vec![CustomBranchRow {
        pc: 0x1000,
        offset: 4,
        register: 0,
        jalr: false,
        next_pc: 0x1008, // Wrong! Should be 0x1004
        multiplicity: 1,
    }];

    assert!(!prove_and_verify_custom(&ops, &receiver_rows));
}

#[test]
fn test_soundness_wrong_jalr_flag() {
    // Sender expects JALR=false, receiver has JALR=true
    let ops = vec![BranchOperation::new(0x1000, 4, 0, false)];

    // Receiver claims it's a JALR instruction with register=0x1000
    // This would give the same result but wrong flag
    let receiver_rows = vec![CustomBranchRow {
        pc: 0x1000,
        offset: 4,
        register: 0x1000, // Use register that would give same result
        jalr: true,       // Wrong flag!
        next_pc: 0x1004,
        multiplicity: 1,
    }];

    assert!(!prove_and_verify_custom(&ops, &receiver_rows));
}

#[test]
fn test_soundness_multiplicity_mismatch() {
    // Sender has 2 lookups, receiver only has 1
    let ops = vec![
        BranchOperation::new(0x1000, 4, 0, false),
        BranchOperation::new(0x1000, 4, 0, false),
    ];

    let op = &ops[0];
    let receiver_rows = vec![CustomBranchRow {
        pc: op.pc,
        offset: op.offset,
        register: op.register,
        jalr: op.jalr,
        next_pc: op.compute_next_pc(),
        multiplicity: 1, // Should be 2!
    }];

    assert!(!prove_and_verify_custom(&ops, &receiver_rows));
}

#[test]
fn test_soundness_missing_receiver_row() {
    // Sender requests two different operations, receiver only has one
    let ops = vec![
        BranchOperation::new(0x1000, 4, 0, false),
        BranchOperation::new(0x2000, 8, 0, false),
    ];

    let op1 = &ops[0];
    // Only provide receiver row for first operation
    let receiver_rows = vec![CustomBranchRow {
        pc: op1.pc,
        offset: op1.offset,
        register: op1.register,
        jalr: op1.jalr,
        next_pc: op1.compute_next_pc(),
        multiplicity: 1,
    }];

    assert!(!prove_and_verify_custom(&ops, &receiver_rows));
}

#[test]
fn test_soundness_wrong_offset() {
    // Sender expects offset=4, receiver has different offset
    let ops = vec![BranchOperation::new(0x1000, 4, 0, false)];

    let receiver_rows = vec![CustomBranchRow {
        pc: 0x1000,
        offset: 8, // Wrong offset!
        register: 0,
        jalr: false,
        next_pc: 0x1004, // Correct next_pc for offset=4
        multiplicity: 1,
    }];

    assert!(!prove_and_verify_custom(&ops, &receiver_rows));
}
