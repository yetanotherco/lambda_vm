//! Tests for the trace builder module.

use crate::tables::bitwise;
use crate::tables::cpu::cols;
use crate::tables::lt;
use crate::tables::trace_builder::Traces;
use crate::tables::types::FE;
use executor::vm::execution::{InstructionCache, InstructionSegment};
use executor::vm::instruction::decoding::{ArithOp, Comparison, Instruction};
use executor::vm::logs::Log;

/// A single step in a test program, containing both the instruction and runtime values.
struct ProgramStep {
    instruction: Instruction,
    src1_val: u64,
    src2_val: u64,
    dst_val: u64,
    taken: bool,
    offset: i32,
}

impl ProgramStep {
    fn add(dst: u32, src1: u32, src2: u32, src1_val: u64, src2_val: u64, dst_val: u64) -> Self {
        Self {
            instruction: Instruction::Arith {
                dst,
                src1,
                src2,
                op: ArithOp::Add,
            },
            src1_val,
            src2_val,
            dst_val,
            taken: false,
            offset: 0,
        }
    }

    fn slt(dst: u32, src1: u32, src2: u32, src1_val: u64, src2_val: u64, result: u64) -> Self {
        Self {
            instruction: Instruction::Arith {
                dst,
                src1,
                src2,
                op: ArithOp::SetLessThan,
            },
            src1_val,
            src2_val,
            dst_val: result,
            taken: false,
            offset: 0,
        }
    }

    fn and(dst: u32, src1: u32, src2: u32, src1_val: u64, src2_val: u64, result: u64) -> Self {
        Self {
            instruction: Instruction::Arith {
                dst,
                src1,
                src2,
                op: ArithOp::And,
            },
            src1_val,
            src2_val,
            dst_val: result,
            taken: false,
            offset: 0,
        }
    }

    fn blt(src1: u32, src2: u32, src1_val: u64, src2_val: u64, taken: bool, offset: i32) -> Self {
        Self {
            instruction: Instruction::Branch {
                src1,
                src2,
                cond: Comparison::LessThan,
                offset,
            },
            src1_val,
            src2_val,
            dst_val: 0,
            taken,
            offset,
        }
    }
}

/// Build test data from program steps.
/// Returns logs and instruction cache with sequential PCs starting at base_pc.
fn build_test_data(base_pc: u64, steps: Vec<ProgramStep>) -> (Vec<Log>, InstructionCache) {
    let mut logs = Vec::with_capacity(steps.len());
    let mut instructions = Vec::with_capacity(steps.len());

    for (i, step) in steps.iter().enumerate() {
        let pc = base_pc + (i as u64 * 4);
        let next_pc = if step.taken {
            (pc as i64 + step.offset as i64) as u64
        } else {
            pc + 4
        };
        logs.push(Log {
            current_pc: pc,
            next_pc,
            src1_val: step.src1_val,
            src2_val: step.src2_val,
            dst_val: step.dst_val,
        });
        instructions.push(step.instruction);
    }

    let cache = InstructionCache {
        segments: vec![InstructionSegment {
            base_addr: base_pc,
            instructions,
        }],
    };

    (logs, cache)
}

#[test]
#[should_panic(expected = "CPU trace requires at least 4 operations")]
fn test_empty_logs() {
    // CPU trace cannot be padded - caller must provide valid power-of-2 operations
    let _traces = Traces::from_logs(&[], &InstructionCache::new(&[]).unwrap()).unwrap();
}

#[test]
#[should_panic(expected = "CPU trace requires at least 4 operations")]
fn test_single_log() {
    // CPU trace cannot be padded - caller must provide valid power-of-2 operations
    let (logs, cache) = build_test_data(0x1000, vec![ProgramStep::add(1, 2, 3, 10, 20, 30)]);
    let _traces = Traces::from_logs(&logs, &cache).unwrap();
}

#[test]
fn test_power_of_two_logs() {
    let (logs, cache) = build_test_data(
        0x1000,
        (0..4)
            .map(|i| ProgramStep::add(1, 2, 3, i, i, i * 2))
            .collect(),
    );

    let traces = Traces::from_logs(&logs, &cache).unwrap();
    assert_eq!(traces.cpu.main_table.height, 4);
}

#[test]
#[should_panic(expected = "CPU trace requires power-of-2 operations")]
fn test_padding_to_power_of_two() {
    // CPU trace cannot be padded - caller must provide valid power-of-2 operations
    let (logs, cache) = build_test_data(
        0x1000,
        (0..5)
            .map(|i| ProgramStep::add(1, 2, 3, i, i, i * 2))
            .collect(),
    );

    let _traces = Traces::from_logs(&logs, &cache).unwrap();
}

#[test]
fn test_lt_operations_collected() {
    let (logs, cache) = build_test_data(
        0x1000,
        vec![
            ProgramStep::slt(1, 2, 3, 5, 10, 1),
            ProgramStep::slt(1, 2, 3, 10, 5, 0),
            ProgramStep::add(1, 2, 3, 1, 2, 3),
            ProgramStep::blt(2, 3, 3, 7, true, 8),
        ],
    );

    let traces = Traces::from_logs(&logs, &cache).unwrap();

    // LT trace should have rows (2 SLT + 1 BLT = 3 ops, deduplicated)
    assert!(traces.lt.main_table.height >= 2);
}

#[test]
fn test_lt_deduplication() {
    let (logs, cache) = build_test_data(
        0x1000,
        vec![
            ProgramStep::slt(1, 2, 3, 5, 10, 1),
            ProgramStep::slt(1, 2, 3, 5, 10, 1), // duplicate
            ProgramStep::slt(1, 2, 3, 5, 10, 1), // duplicate
            ProgramStep::add(1, 2, 3, 0, 0, 0),  // padding to 4
        ],
    );

    let traces = Traces::from_logs(&logs, &cache).unwrap();

    // Should have 1 unique LT op with multiplicity 3
    assert_eq!(traces.lt.main_table.height, 4); // 1 op padded to 4 (minimum for FRI)
    let row = traces.lt.main_table.get_row(0);
    assert_eq!(row[lt::cols::MU], FE::from(3u64));
}

#[test]
fn test_bitwise_lookups_collected() {
    let (logs, cache) = build_test_data(
        0x1000,
        vec![
            ProgramStep::and(1, 2, 3, 0x12, 0x34, 0x10),
            ProgramStep::add(1, 2, 3, 0, 0, 0),
            ProgramStep::add(1, 2, 3, 0, 0, 0),
            ProgramStep::add(1, 2, 3, 0, 0, 0),
        ],
    );

    let traces = Traces::from_logs(&logs, &cache).unwrap();

    // Check AND multiplicity was updated for (0x12, 0x34, 0)
    let row_idx = bitwise::row_index(0x12, 0x34, 0);
    let row = traces.bitwise.main_table.get_row(row_idx);
    assert_eq!(row[bitwise::cols::MU_AND], FE::one());
}

#[test]
fn test_cpu_timestamps() {
    let (logs, cache) = build_test_data(
        0x1000,
        vec![
            ProgramStep::add(1, 2, 3, 1, 2, 3),
            ProgramStep::add(1, 2, 3, 4, 5, 6),
            ProgramStep::add(1, 2, 3, 7, 8, 9),
            ProgramStep::add(1, 2, 3, 10, 11, 12),
        ],
    );

    let traces = Traces::from_logs(&logs, &cache).unwrap();

    // Check timestamps are 0, 4, 8, 12
    for i in 0..4 {
        let row = traces.cpu.main_table.get_row(i);
        assert_eq!(row[cols::TIMESTAMP], FE::from((i * 4) as u64));
    }
}

#[test]
fn test_mixed_instructions() {
    let (logs, cache) = build_test_data(
        0x1000,
        vec![
            ProgramStep::add(1, 2, 3, 10, 20, 30),
            ProgramStep::slt(1, 2, 3, 5, 10, 1),
            ProgramStep::and(1, 2, 3, 0xFF, 0xF0, 0xF0),
            ProgramStep::blt(2, 3, 1, 2, true, 8),
        ],
    );

    let traces = Traces::from_logs(&logs, &cache).unwrap();

    assert_eq!(traces.cpu.main_table.height, 4);
    assert_eq!(traces.bitwise.main_table.height, bitwise::NUM_ROWS);
    // 1 SLT + 1 BLT = 2 LT ops
    assert!(traces.lt.main_table.height >= 2);
}
