//! Tests for the CPU table.
//!
//! Unit tests for the reworked `CpuOperation::from_log` (arg2 multiplex, res,
//! rvd, branch decision, word-instruction delegation), `generate_cpu_trace`
//! (column layout, padding, word-row masking), and `collect_bitwise_ops`.

use crate::tables::cpu::{CPU_PADDING_PC, CpuOperation, cols, generate_cpu_trace};
use crate::tables::types::DecodeEntry;

use executor::vm::{
    instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth},
    logs::Log,
};

const PC: u64 = 0x1000;

/// Build a CpuOperation from an instruction + register values.
fn op_of(instr: Instruction, src1: u64, src2: u64, dst: u64, next_pc: u64) -> CpuOperation {
    let decode = DecodeEntry::from_instruction(PC, instr, 4);
    let log = Log {
        current_pc: PC,
        next_pc,
        src1_val: src1,
        src2_val: src2,
        dst_val: dst,
    };
    CpuOperation::from_log(&log, 4, decode)
}

// =========================================================================
// from_log: arg2 multiplex, res, rvd, branch decision
// =========================================================================

#[test]
fn test_from_log_add_reg_reg() {
    let op = op_of(
        Instruction::Arith {
            dst: 3,
            src1: 1,
            src2: 2,
            op: ArithOp::Add,
        },
        10,
        20,
        30,
        PC + 4,
    );
    assert_eq!(op.rv1, 10);
    assert_eq!(op.rv2, 20);
    assert_eq!(op.arg2, 20, "reg-reg: arg2 = rv2 (imm = 0)");
    assert_eq!(op.res, 30, "res = rv1 + arg2");
    assert_eq!(op.rvd, 30, "rvd = res (not memory)");
    assert_eq!(op.next_pc, PC + 4);
    assert!(!op.branch_cond);
}

#[test]
fn test_from_log_addi() {
    let op = op_of(
        Instruction::ArithImm {
            dst: 3,
            src: 1,
            imm: 5,
            op: ArithOp::Add,
        },
        10,
        0,
        15,
        PC + 4,
    );
    assert_eq!(op.arg2, 5, "reg-imm: arg2 = imm (rv2 = 0)");
    assert_eq!(op.res, 15);
    assert_eq!(op.rvd, 15);
}

#[test]
fn test_from_log_sub() {
    let op = op_of(
        Instruction::Arith {
            dst: 3,
            src1: 1,
            src2: 2,
            op: ArithOp::Sub,
        },
        30,
        20,
        10,
        PC + 4,
    );
    assert_eq!(op.res, 10, "res = rv1 - arg2");
    assert_eq!(op.rvd, 10);
}

#[test]
fn test_from_log_beq_taken() {
    let op = op_of(
        Instruction::Branch {
            src1: 1,
            src2: 2,
            cond: Comparison::Equal,
            offset: 8,
        },
        5,
        5,
        0,
        PC + 8,
    );
    assert!(op.branch_cond, "BEQ with equal operands is taken");
    assert_eq!(op.arg2, 5, "conditional branch: arg2 = rv2");
    assert_eq!(op.res, 1, "EQ result on the ALU bus is 1 when taken");
    assert_eq!(op.next_pc, PC + 8, "taken branch uses the executor next_pc");
}

#[test]
fn test_from_log_beq_not_taken() {
    let op = op_of(
        Instruction::Branch {
            src1: 1,
            src2: 2,
            cond: Comparison::Equal,
            offset: 8,
        },
        5,
        6,
        0,
        PC + 4,
    );
    assert!(!op.branch_cond);
    assert_eq!(op.res, 0);
    assert_eq!(
        op.next_pc,
        PC + 4,
        "untaken branch falls through to pc + len"
    );
}

#[test]
fn test_from_log_bne_taken() {
    let op = op_of(
        Instruction::Branch {
            src1: 1,
            src2: 2,
            cond: Comparison::NotEqual,
            offset: 8,
        },
        5,
        6,
        0,
        PC + 8,
    );
    assert!(
        op.branch_cond,
        "BNE with differing operands is taken (invert)"
    );
    assert_eq!(op.res, 1);
}

#[test]
fn test_from_log_load() {
    let op = op_of(
        Instruction::Load {
            dst: 3,
            offset: 4,
            base: 1,
            width: LoadStoreWidth::Word,
        },
        0x100,
        0,
        0xDEAD,
        PC + 4,
    );
    assert_eq!(op.res, 0x104, "load address = rv1 + imm");
    assert_eq!(op.rvd, 0xDEAD, "load rvd = the loaded value");
}

#[test]
fn test_from_log_store() {
    let op = op_of(
        Instruction::Store {
            src: 2,
            offset: 8,
            base: 1,
            width: LoadStoreWidth::Word,
        },
        0x100,
        0xAB,
        0,
        PC + 4,
    );
    assert_eq!(op.res, 0x108, "store address = rv1 + imm");
    assert_eq!(op.rv2, 0xAB, "store value comes from rs2");
    assert_eq!(op.rvd, 0, "store writes nothing back to rd");
}

#[test]
fn test_from_log_word_carries_real_register_values() {
    let op = op_of(
        Instruction::ArithW {
            dst: 3,
            src1: 1,
            src2: 2,
            op: ArithOp::Add,
        },
        10,
        20,
        30,
        PC + 4,
    );
    assert!(op.decode.fields.word_instr);
    // The delegate CpuOperation carries the real values for CPU32/register ops.
    assert_eq!(op.rv1, 10);
    assert_eq!(op.rv2, 20);
    assert_eq!(op.rvd, 30);
    assert_eq!(op.res, 0, "the main CPU delegate row computes no result");
    assert_eq!(op.next_pc, PC + 4);
}

// =========================================================================
// generate_cpu_trace
// =========================================================================

fn ops4(instr: Instruction) -> Vec<CpuOperation> {
    (0..4)
        .map(|i| {
            let decode = DecodeEntry::from_instruction(PC + i * 4, instr, 4);
            let log = Log {
                current_pc: PC + i * 4,
                next_pc: PC + i * 4 + 4,
                src1_val: 10,
                src2_val: 20,
                dst_val: 30,
            };
            CpuOperation::from_log(&log, i * 4 + 4, decode)
        })
        .collect()
}

#[test]
fn test_trace_width_and_real_row() {
    let ops = ops4(Instruction::Arith {
        dst: 3,
        src1: 1,
        src2: 2,
        op: ArithOp::Add,
    });
    let trace = generate_cpu_trace(&ops);
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);
    assert_eq!(cols::NUM_COLUMNS, 38);
    assert_eq!(trace.main_table.height, 4);
    let row = trace.main_table.get_row(0);
    assert_eq!(row[cols::PC_0], (PC).into());
    assert_eq!(row[cols::ADD], 1u64.into(), "ADD fast-path flag set");
    assert_eq!(row[cols::RES_0], 30u64.into());
}

#[test]
fn test_trace_padding_row() {
    // One real op → padded to 4 rows; rows 1..4 are padding.
    let ops = vec![
        ops4(Instruction::Arith {
            dst: 3,
            src1: 1,
            src2: 2,
            op: ArithOp::Add,
        })
        .remove(0),
    ];
    let trace = generate_cpu_trace(&ops);
    let pad = trace.main_table.get_row(1);
    assert_eq!(
        pad[cols::PC_0],
        CPU_PADDING_PC.into(),
        "padding pc = 1 (odd)"
    );
    assert_eq!(
        pad[cols::NEXT_PC_0],
        CPU_PADDING_PC.into(),
        "next_pc = pc (half_instruction_length = 0)"
    );
    assert_eq!(pad[cols::HALF_INSTRUCTION_LENGTH], 0u64.into());
    assert_eq!(pad[cols::WORD_INSTR], 0u64.into());
}

#[test]
fn test_trace_word_row_columns_masked() {
    let ops = ops4(Instruction::ArithW {
        dst: 3,
        src1: 1,
        src2: 2,
        op: ArithOp::Add,
    });
    let trace = generate_cpu_trace(&ops);
    let row = trace.main_table.get_row(0);
    // Delegate row: word_instr set, but all operational columns masked to 0.
    assert_eq!(row[cols::WORD_INSTR], 1u64.into());
    assert_eq!(row[cols::HALF_INSTRUCTION_LENGTH], 2u64.into());
    assert_eq!(
        row[cols::RV1_0],
        0u64.into(),
        "rv1 column masked on word row"
    );
    assert_eq!(row[cols::READ_REGISTER1], 0u64.into());
    assert_eq!(row[cols::ADD], 0u64.into());
    assert_eq!(row[cols::RVD_0], 0u64.into());
}

// =========================================================================
// collect_bitwise_ops
// =========================================================================

#[test]
fn test_collect_bitwise_ops_shape() {
    use crate::tables::bitwise::BitwiseOperationType;
    let op = op_of(
        Instruction::Arith {
            dst: 3,
            src1: 1,
            src2: 2,
            op: ArithOp::Add,
        },
        10,
        20,
        30,
        PC + 4,
    );
    let ops = op.collect_bitwise_ops();
    assert_eq!(ops.len(), 7, "3 ARE_BYTES + 4 IS_HALF");
    assert!(
        ops[0..3]
            .iter()
            .all(|o| o.lookup_type == BitwiseOperationType::AreBytes)
    );
    assert!(
        ops[3..7]
            .iter()
            .all(|o| o.lookup_type == BitwiseOperationType::IsHalf)
    );
    // First ARE_BYTES is (rs1, rs2) = (1, 2).
    assert_eq!(ops[0].x, 1);
    assert_eq!(ops[0].y, 2);
}

#[test]
fn test_collect_bitwise_ops_word_row_zeroed() {
    let op = op_of(
        Instruction::ArithW {
            dst: 3,
            src1: 1,
            src2: 2,
            op: ArithOp::Add,
        },
        10,
        20,
        30,
        PC + 4,
    );
    let ops = op.collect_bitwise_ops();
    // On a word delegate row the CPU zeroes rs1/rs2/rd/alu_flags/mem_flags/res,
    // but half_instruction_length stays (it is set unconditionally in the trace).
    assert_eq!(ops[0].x, 0, "rs1 zeroed");
    assert_eq!(ops[0].y, 0, "rs2 zeroed");
    assert_eq!(ops[1].x, 0, "rd zeroed");
    assert_eq!(ops[1].y, 2, "half_instruction_length retained");
}
