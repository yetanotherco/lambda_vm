//! Tests for the trace builder module.

use crate::tables::bitwise;
use crate::tables::cpu::cols;
use crate::tables::lt;
use crate::tables::trace_builder::Traces;
use crate::tables::types::FE;
use executor::vm::instruction::decoding::{ArithOp, Comparison, Instruction};
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;

fn make_log(pc: u64, rs1_val: u64, rs2_val: u64, dst_val: u64, taken: bool, offset: i32) -> Log {
    Log {
        current_pc: pc,
        next_pc: if taken { (pc as i64 + offset as i64) as u64 } else { pc + 4 },
        src1_val: rs1_val,
        src2_val: rs2_val,
        dst_val,
    }
}

fn make_add_log(pc: u64, rs1_val: u64, rs2_val: u64, dst_val: u64) -> Log {
    make_log(pc, rs1_val, rs2_val, dst_val, false, 0)
}

fn make_slt_log(pc: u64, rs1_val: u64, rs2_val: u64, result: u64) -> Log {
    make_log(pc, rs1_val, rs2_val, result, false, 0)
}

fn make_blt_log(pc: u64, rs1_val: u64, rs2_val: u64, taken: bool) -> Log {
    make_log(pc, rs1_val, rs2_val, 0, taken, 8)
}

fn make_and_log(pc: u64, rs1_val: u64, rs2_val: u64, result: u64) -> Log {
    make_log(pc, rs1_val, rs2_val, result, false, 0)
}

/// Build instructions map for test logs
fn make_instructions(logs: &[Log], instrs: &[Instruction]) -> U64HashMap<Instruction> {
    let mut map = U64HashMap::default();
    for (log, instr) in logs.iter().zip(instrs.iter()) {
        map.insert(log.current_pc, *instr);
    }
    map
}

#[test]
#[should_panic(expected = "CPU trace requires at least 4 operations")]
fn test_empty_logs() {
    // CPU trace cannot be padded - caller must provide valid power-of-2 operations
    let _traces = Traces::from_logs(&[], U64HashMap::default()).unwrap();
}

#[test]
#[should_panic(expected = "CPU trace requires at least 4 operations")]
fn test_single_log() {
    // CPU trace cannot be padded - caller must provide valid power-of-2 operations
    let logs = vec![make_add_log(0x1000, 10, 20, 30)];
    let instrs = vec![Instruction::Arith {
        dst: 1,
        src1: 2,
        src2: 3,
        op: ArithOp::Add,
    }];
    let instructions = make_instructions(&logs, &instrs);
    let _traces = Traces::from_logs(&logs, instructions).unwrap();
}

#[test]
fn test_power_of_two_logs() {
    let logs: Vec<Log> = (0..4)
        .map(|i| make_add_log(0x1000 + i * 4, i, i, i * 2))
        .collect();
    let instrs: Vec<Instruction> = (0..4)
        .map(|_| Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        })
        .collect();
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions).unwrap();
    assert_eq!(traces.cpu.main_table.height, 4);
}

#[test]
#[should_panic(expected = "CPU trace requires power-of-2 operations")]
fn test_padding_to_power_of_two() {
    // CPU trace cannot be padded - caller must provide valid power-of-2 operations
    let logs: Vec<Log> = (0..5)
        .map(|i| make_add_log(0x1000 + i * 4, i, i, i * 2))
        .collect();
    let instrs: Vec<Instruction> = (0..5)
        .map(|_| Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        })
        .collect();
    let instructions = make_instructions(&logs, &instrs);

    let _traces = Traces::from_logs(&logs, instructions).unwrap();
}

#[test]
fn test_lt_operations_collected() {
    let logs = vec![
        make_slt_log(0x1000, 5, 10, 1),
        make_slt_log(0x1004, 10, 5, 0),
        make_add_log(0x1008, 1, 2, 3),
        make_blt_log(0x100c, 3, 7, true),
    ];
    let instrs = vec![
        Instruction::Arith { dst: 1, src1: 2, src2: 3, op: ArithOp::SetLessThan },
        Instruction::Arith { dst: 1, src1: 2, src2: 3, op: ArithOp::SetLessThan },
        Instruction::Arith { dst: 1, src1: 2, src2: 3, op: ArithOp::Add },
        Instruction::Branch { src1: 2, src2: 3, cond: Comparison::LessThan, offset: 8 },
    ];
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions).unwrap();

    // LT trace should have rows (2 SLT + 1 BLT = 3 ops, deduplicated)
    assert!(traces.lt.main_table.height >= 2);
}

#[test]
fn test_lt_deduplication() {
    let logs = vec![
        make_slt_log(0x1000, 5, 10, 1),
        make_slt_log(0x1004, 5, 10, 1), // duplicate
        make_slt_log(0x1008, 5, 10, 1), // duplicate
        make_add_log(0x100c, 0, 0, 0),  // padding to 4
    ];
    let instrs = vec![
        Instruction::Arith { dst: 1, src1: 2, src2: 3, op: ArithOp::SetLessThan },
        Instruction::Arith { dst: 1, src1: 2, src2: 3, op: ArithOp::SetLessThan },
        Instruction::Arith { dst: 1, src1: 2, src2: 3, op: ArithOp::SetLessThan },
        Instruction::Arith { dst: 1, src1: 2, src2: 3, op: ArithOp::Add },
    ];
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions).unwrap();

    // Should have 1 unique LT op with multiplicity 3
    assert_eq!(traces.lt.main_table.height, 2); // 1 op padded to 2
    let row = traces.lt.main_table.get_row(0);
    assert_eq!(row[lt::cols::MU], FE::from(3u64));
}

#[test]
fn test_bitwise_lookups_collected() {
    let logs = vec![
        make_and_log(0x1000, 0x12, 0x34, 0x10),
        make_add_log(0x1004, 0, 0, 0),
        make_add_log(0x1008, 0, 0, 0),
        make_add_log(0x100c, 0, 0, 0),
    ];
    let instrs = vec![
        Instruction::Arith { dst: 1, src1: 2, src2: 3, op: ArithOp::And },
        Instruction::Arith { dst: 1, src1: 2, src2: 3, op: ArithOp::Add },
        Instruction::Arith { dst: 1, src1: 2, src2: 3, op: ArithOp::Add },
        Instruction::Arith { dst: 1, src1: 2, src2: 3, op: ArithOp::Add },
    ];
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions).unwrap();

    // Check AND multiplicity was updated for (0x12, 0x34, 0)
    let row_idx = bitwise::row_index(0x12, 0x34, 0);
    let row = traces.bitwise.main_table.get_row(row_idx);
    assert_eq!(row[bitwise::cols::MU_AND], FE::one());
}

#[test]
fn test_cpu_timestamps() {
    let logs = vec![
        make_add_log(0x1000, 1, 2, 3),
        make_add_log(0x1004, 4, 5, 6),
        make_add_log(0x1008, 7, 8, 9),
        make_add_log(0x100c, 10, 11, 12),
    ];
    let instrs: Vec<Instruction> = (0..4)
        .map(|_| Instruction::Arith { dst: 1, src1: 2, src2: 3, op: ArithOp::Add })
        .collect();
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions).unwrap();

    // Check timestamps are 0, 4, 8, 12
    for i in 0..4 {
        let row = traces.cpu.main_table.get_row(i);
        assert_eq!(row[cols::TIMESTAMP], FE::from((i * 4) as u64));
    }
}

#[test]
fn test_mixed_instructions() {
    let logs = vec![
        make_add_log(0x1000, 10, 20, 30),
        make_slt_log(0x1004, 5, 10, 1),
        make_and_log(0x1008, 0xFF, 0xF0, 0xF0),
        make_blt_log(0x100c, 1, 2, true),
    ];
    let instrs = vec![
        Instruction::Arith { dst: 1, src1: 2, src2: 3, op: ArithOp::Add },
        Instruction::Arith { dst: 1, src1: 2, src2: 3, op: ArithOp::SetLessThan },
        Instruction::Arith { dst: 1, src1: 2, src2: 3, op: ArithOp::And },
        Instruction::Branch { src1: 2, src2: 3, cond: Comparison::LessThan, offset: 8 },
    ];
    let instructions = make_instructions(&logs, &instrs);

    let traces = Traces::from_logs(&logs, instructions).unwrap();

    assert_eq!(traces.cpu.main_table.height, 4);
    assert_eq!(traces.bitwise.main_table.height, bitwise::NUM_ROWS);
    // 1 SLT + 1 BLT = 2 LT ops
    assert!(traces.lt.main_table.height >= 2);
}
