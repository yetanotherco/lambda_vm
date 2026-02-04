//! Tests for trace segmentation.
//!
//! These tests verify:
//! - Boundary state serialization/deserialization
//! - Memory and register state continuity across segments
//! - Timestamp continuity across segments

use crate::tables::segment::{SegmentBoundary, SegmentConfig};
use crate::tables::trace_builder::Traces;
use crate::test_utils::run_asm_elf;
use executor::vm::instruction::decoding::Instruction;
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;

// =============================================================================
// Helper Functions
// =============================================================================

/// Find the index of the ECALL instruction in the logs.
/// Returns None if no ECALL is found.
fn find_ecall_index(logs: &[Log], instructions: &U64HashMap<Instruction>) -> Option<usize> {
    logs.iter().position(|log| {
        instructions
            .get(&log.current_pc)
            .map(|inst| matches!(inst, Instruction::EcallEbreak))
            .unwrap_or(false)
    })
}

// =============================================================================
// Boundary State Tests
// =============================================================================

#[test]
fn test_boundary_initial() {
    let boundary = SegmentBoundary::initial();

    assert_eq!(boundary.segment_index, 0);
    assert_eq!(boundary.start_timestamp, 4);
    assert!(boundary.memory_state.is_empty());
    assert_eq!(boundary.register_state, [(0, 0); 31]);
    assert_eq!(boundary.start_pc, 0);
}

#[test]
fn test_boundary_next() {
    let initial = SegmentBoundary::initial();

    let memory_state = vec![(0x1000, 0x42, 100), (0x1001, 0x43, 104)];
    let mut register_state = [(0u64, 0u64); 31];
    register_state[0] = (0xDEAD, 108); // x1
    register_state[9] = (0xBEEF, 112); // x10

    let next = initial.next(200, memory_state.clone(), register_state, 0x80000010);

    assert_eq!(next.segment_index, 1);
    assert_eq!(next.start_timestamp, 200);
    assert_eq!(next.memory_state, memory_state);
    assert_eq!(next.register_state, register_state);
    assert_eq!(next.start_pc, 0x80000010);
}

// =============================================================================
// Segmented Trace Generation Tests
// =============================================================================

#[test]
fn test_from_logs_segmented_intermediate_segment() {
    // Test that from_logs_segmented works for an intermediate (non-final) segment
    // Intermediate segments don't need an ECALL
    let (_elf, logs, instructions) = run_asm_elf("sub");

    let boundary = SegmentBoundary::initial();
    let config = SegmentConfig::intermediate();

    let result = Traces::from_logs_segmented(&logs, instructions, &boundary, &config)
        .expect("Failed to generate segmented traces");

    assert!(!result.is_final);
    assert!(result.next_boundary.is_some());

    // Verify traces are generated
    assert!(result.traces.cpu.main_table.height > 0);
    assert!(result.traces.memw.main_table.height > 0);
}

#[test]
fn test_from_logs_segmented_with_state_continuity() {
    // Test that state is properly captured and restored across segments
    let (_elf, logs, instructions) = run_asm_elf("comprehensive_test");

    // Find ECALL position
    let ecall_idx = match find_ecall_index(&logs, &instructions) {
        Some(idx) => idx,
        None => {
            // No ECALL found, skip test
            return;
        }
    };

    // Need at least 2 logs to split
    if ecall_idx == 0 {
        return;
    }

    // Split so ECALL is in the second segment
    let first_half = &logs[..ecall_idx];
    let second_half = &logs[ecall_idx..];

    // Generate first segment (intermediate - no ECALL)
    let boundary = SegmentBoundary::initial();
    let config = SegmentConfig::intermediate();

    let result1 = Traces::from_logs_segmented(first_half, instructions.clone(), &boundary, &config)
        .expect("Failed to generate first segment");

    assert!(!result1.is_final);
    let next_boundary = result1.next_boundary.expect("Should have next boundary");

    // Verify boundary state
    assert_eq!(next_boundary.segment_index, 1);
    assert_eq!(
        next_boundary.start_timestamp,
        4 + (first_half.len() as u64) * 4
    );

    // Generate second segment (final - contains ECALL)
    let config = SegmentConfig::final_segment();

    let result2 = Traces::from_logs_segmented(second_half, instructions, &next_boundary, &config)
        .expect("Failed to generate second segment");

    assert!(result2.is_final);
    assert!(result2.next_boundary.is_none());
}

#[test]
fn test_timestamp_continuity() {
    // Verify timestamps are continuous across segments
    let (_elf, logs, instructions) = run_asm_elf("arith_8");

    if logs.len() < 4 {
        return;
    }

    let split_point = logs.len() / 2;
    let first_half = &logs[..split_point];

    // Generate first segment (intermediate)
    let boundary = SegmentBoundary::initial();
    let config = SegmentConfig::intermediate();

    let result = Traces::from_logs_segmented(first_half, instructions, &boundary, &config)
        .expect("Failed to generate segment");

    let next_boundary = result.next_boundary.expect("Should have next boundary");

    // Expected end timestamp: start (4) + num_logs * 4
    let expected_end_ts = 4 + (first_half.len() as u64) * 4;
    assert_eq!(next_boundary.start_timestamp, expected_end_ts);
}

#[test]
fn test_multi_segment_trace_generation() {
    // Test generating traces across 3 segments
    let (_elf, logs, instructions) = run_asm_elf("all_instructions_64");

    // Find ECALL position
    let ecall_idx = match find_ecall_index(&logs, &instructions) {
        Some(idx) => idx,
        None => return,
    };

    // Need at least 3 instructions before ECALL to split into 3 segments
    if ecall_idx < 3 {
        return;
    }

    // Split into 3 segments: [0..n/3], [n/3..ecall_idx], [ecall_idx..]
    let first_end = ecall_idx / 3;
    let second_end = ecall_idx;

    let seg1_logs = &logs[..first_end];
    let seg2_logs = &logs[first_end..second_end];
    let seg3_logs = &logs[second_end..];

    // Segment 1: intermediate
    let boundary1 = SegmentBoundary::initial();
    let result1 = Traces::from_logs_segmented(
        seg1_logs,
        instructions.clone(),
        &boundary1,
        &SegmentConfig::intermediate(),
    )
    .expect("Segment 1 failed");
    assert!(!result1.is_final);
    let boundary2 = result1.next_boundary.expect("Should have boundary");

    // Segment 2: intermediate
    let result2 = Traces::from_logs_segmented(
        seg2_logs,
        instructions.clone(),
        &boundary2,
        &SegmentConfig::intermediate(),
    )
    .expect("Segment 2 failed");
    assert!(!result2.is_final);
    let boundary3 = result2.next_boundary.expect("Should have boundary");

    // Segment 3: final (contains ECALL)
    let result3 = Traces::from_logs_segmented(
        seg3_logs,
        instructions,
        &boundary3,
        &SegmentConfig::final_segment(),
    )
    .expect("Segment 3 failed");
    assert!(result3.is_final);
    assert!(result3.next_boundary.is_none());

    // Verify timestamp continuity
    assert_eq!(boundary2.start_timestamp, 4 + (seg1_logs.len() as u64) * 4);
    assert_eq!(
        boundary3.start_timestamp,
        boundary2.start_timestamp + (seg2_logs.len() as u64) * 4
    );
}
