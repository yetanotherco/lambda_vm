//! Tests for trace segmentation.
//!
//! These tests verify:
//! - Single-segment behavior when trace fits in one segment
//! - Multi-segment splitting with correct op slicing
//! - ECALL lands in final segment naturally
//! - Segment traces are valid (non-empty, correct structure)

use crate::tables::trace_builder::Traces;
use executor::vm::instruction::decoding::{ArithOp, Instruction};
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;

// =============================================================================
// Helper Functions
// =============================================================================

fn make_add_log(pc: u64, rs1_val: u64, rs2_val: u64, dst_val: u64) -> Log {
    Log {
        current_pc: pc,
        next_pc: pc + 4,
        src1_val: rs1_val,
        src2_val: rs2_val,
        dst_val,
    }
}

fn make_instructions(logs: &[Log], instrs: &[Instruction]) -> U64HashMap<Instruction> {
    let mut map = U64HashMap::default();
    for (log, instr) in logs.iter().zip(instrs.iter()) {
        map.insert(log.current_pc, *instr);
    }
    map
}

fn append_ecall(logs: &mut Vec<Log>, instrs: &mut Vec<Instruction>) {
    let last_pc = logs.last().map(|l| l.current_pc + 4).unwrap_or(0x1000);
    logs.push(Log {
        current_pc: last_pc,
        next_pc: 0,
        src1_val: 0,
        src2_val: 0,
        dst_val: 0,
    });
    instrs.push(Instruction::EcallEbreak);
}

/// Build N ADD instructions followed by ECALL.
fn make_test_program(n: usize) -> (Vec<Log>, U64HashMap<Instruction>) {
    let mut logs: Vec<Log> = (0..n)
        .map(|i| make_add_log(0x1000 + (i as u64) * 4, i as u64, i as u64, (i * 2) as u64))
        .collect();
    let mut instrs: Vec<Instruction> = (0..n)
        .map(|_| Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        })
        .collect();
    append_ecall(&mut logs, &mut instrs);
    let instructions = make_instructions(&logs, &instrs);
    (logs, instructions)
}

// =============================================================================
// Single-segment tests
// =============================================================================

#[test]
fn test_single_segment_when_trace_fits() {
    let (logs, instructions) = make_test_program(5);

    let results =
        Traces::from_logs_segmented(&logs, instructions).expect("Failed to generate segments");

    assert_eq!(results.len(), 1);
    assert!(results[0].is_final);
    assert_eq!(results[0].segment_index, 0);
    assert!(results[0].traces.cpu.main_table.height > 0);
}

#[test]
fn test_segmented_matches_unsegmented() {
    let (logs, instructions) = make_test_program(10);

    let unsegmented = Traces::from_logs(&logs, instructions.clone()).expect("from_logs failed");
    let segmented =
        Traces::from_logs_segmented(&logs, instructions).expect("from_logs_segmented failed");

    assert_eq!(segmented.len(), 1);
    assert_eq!(
        segmented[0].traces.cpu.main_table.height,
        unsegmented.cpu.main_table.height
    );
    assert_eq!(
        segmented[0].traces.memw.main_table.height,
        unsegmented.memw.main_table.height
    );
    assert_eq!(
        segmented[0].traces.lt.main_table.height,
        unsegmented.lt.main_table.height
    );
}

// =============================================================================
// Multi-segment tests
// =============================================================================

#[test]
fn test_multi_segment_with_small_max_size() {
    // 20 ADDs + 1 ECALL = 21 ops, max_size=4 → 6 segments
    let (logs, instructions) = make_test_program(20);

    let results = Traces::from_logs_segmented_with_max_size(&logs, instructions, 4)
        .expect("Failed to generate segments");

    assert!(
        results.len() > 1,
        "Expected multiple segments, got {}",
        results.len()
    );

    // Only the last segment should be final
    for (i, result) in results.iter().enumerate() {
        assert_eq!(result.segment_index, i);
        if i < results.len() - 1 {
            assert!(!result.is_final, "Segment {i} should not be final");
        } else {
            assert!(result.is_final, "Last segment should be final");
        }
    }

    // All segments should have non-empty CPU traces
    for (i, result) in results.iter().enumerate() {
        assert!(
            result.traces.cpu.main_table.height > 0,
            "Segment {i} has empty CPU trace"
        );
    }
}

#[test]
fn test_segment_cpu_row_counts() {
    let max_size = 8;
    // 30 ADDs + 1 ECALL = 31 ops
    let (logs, instructions) = make_test_program(30);
    let num_ops = logs.len(); // 31

    let results = Traces::from_logs_segmented_with_max_size(&logs, instructions, max_size)
        .expect("Failed to generate segments");

    let expected_segments = num_ops.div_ceil(max_size);
    assert_eq!(results.len(), expected_segments);
}

#[test]
fn test_intermediate_segments_have_dummy_halt() {
    // 16 ADDs + 1 ECALL = 17 ops, max_size=4 → 5 segments
    let (logs, instructions) = make_test_program(16);

    let results = Traces::from_logs_segmented_with_max_size(&logs, instructions, 4)
        .expect("Failed to generate segments");

    assert!(results.len() > 1);

    // All segments should have halt trace height 1 (it's always a single-row table)
    for result in &results {
        assert_eq!(result.traces.halt.main_table.height, 1);
    }
}

#[test]
fn test_ecall_in_final_segment() {
    // Verify ECALL naturally lands in the final segment
    // 7 ADDs + 1 ECALL = 8 ops, max_size=3 → 3 segments: [0..3], [3..6], [6..8]
    let (logs, instructions) = make_test_program(7);

    let results = Traces::from_logs_segmented_with_max_size(&logs, instructions, 3)
        .expect("Failed to generate segments");

    assert_eq!(results.len(), 3);
    assert!(results[2].is_final);
    assert!(!results[0].is_final);
    assert!(!results[1].is_final);
}
