//! Tests for the CPU table.
//!
//! This module contains:
//! - Unit tests for CpuOperation struct and its methods
//! - Trace generation tests
//! - Integration tests for CpuOperation::from_log (ELF execution)

use crate::tables::cpu::{CpuOperation, bus_interactions, cols, generate_cpu_trace};
use crate::tables::trace_builder::Traces;
use crate::tables::types::{DecodeEntry, FE};

use executor::{
    elf::Elf,
    vm::{execution::Executor, instruction::decoding::Instruction, memory::U64HashMap},
};

/// Helper to create 4 operations from a template (required for power-of-2 trace).
fn ops4(op: CpuOperation) -> Vec<CpuOperation> {
    (0..4)
        .map(|i| {
            let mut new_op = op.clone();
            new_op.timestamp = (i as u64) * 4 + 4;
            new_op.decode.pc = op.decode.pc + (i as u64) * 4;
            new_op.next_pc = op.decode.pc + (i as u64) * 4 + 4;
            new_op
        })
        .collect()
}

#[test]
fn test_cpu_operation_default() {
    let op = CpuOperation::new();
    assert_eq!(op.timestamp, 0);
    assert_eq!(op.decode.pc, 0);
    assert!(!op.decode.op_add);
    assert!(!op.branch_cond);
}

#[test]
fn test_cpu_operation_compute_arg1_no_extension() {
    let mut op = CpuOperation::new();
    op.rv1 = 0x1234_5678_9ABC_DEF0;
    op.decode.word_instr = false;

    assert_eq!(op.compute_arg1(), 0x1234_5678_9ABC_DEF0);
}

#[test]
fn test_cpu_operation_compute_arg1_word_zero_extend() {
    let mut op = CpuOperation::new();
    op.rv1 = 0x1234_5678_9ABC_DEF0;
    op.decode.word_instr = true;
    op.decode.signed = false;

    // Should zero-extend from lower 32 bits
    assert_eq!(op.compute_arg1(), 0x9ABC_DEF0);
}

#[test]
fn test_cpu_operation_compute_arg1_word_sign_extend_positive() {
    let mut op = CpuOperation::new();
    op.rv1 = 0x1234_5678_1ABC_DEF0; // Positive 32-bit value
    op.decode.word_instr = true;
    op.decode.signed = true;

    // Bit 31 is 0, so sign extension keeps it positive
    assert_eq!(op.compute_arg1(), 0x1ABC_DEF0);
}

#[test]
fn test_cpu_operation_compute_arg1_word_sign_extend_negative() {
    let mut op = CpuOperation::new();
    op.rv1 = 0x1234_5678_8000_0001; // Negative when viewed as 32-bit signed
    op.decode.word_instr = true;
    op.decode.signed = true;

    // Per spec constraint: arg1[4:] = (2^32-1) * rv1_sign_bit * signed
    // For signed word instructions with sign bit set, arg1 is sign-extended.
    assert_eq!(op.compute_arg1(), 0xFFFF_FFFF_8000_0001);
}

#[test]
fn test_cpu_operation_compute_arg2_store() {
    let mut op = CpuOperation::new();
    op.rv2 = 0xDEAD_BEEF;
    op.decode.imm = 0x1234;
    op.decode.op_store = true;

    // STORE: arg2 = rv2 (the data being stored)
    // Address is computed separately as res = arg1 + imm
    assert_eq!(op.compute_arg2(), 0xDEAD_BEEF);
}

#[test]
fn test_cpu_operation_compute_arg2_load() {
    let mut op = CpuOperation::new();
    op.rv2 = 0xDEAD_BEEF;
    op.decode.imm = 0x1234;
    op.decode.op_load = true;

    // LOAD uses imm for address calculation (addr = rv1 + imm)
    assert_eq!(op.compute_arg2(), 0x1234);
}

#[test]
fn test_cpu_operation_compute_arg2_beq() {
    let mut op = CpuOperation::new();
    op.rv2 = 0xCAFE_BABE;
    op.decode.imm = 0x5678;
    op.decode.op_beq = true;

    // BEQ uses rv2
    assert_eq!(op.compute_arg2(), 0xCAFE_BABE);
}

#[test]
fn test_cpu_operation_compute_arg2_add_with_imm() {
    let mut op = CpuOperation::new();
    op.rv2 = 0;
    op.decode.rs2 = 0; // rs2 = 0 means use immediate
    op.decode.imm = 0x1234_5678;
    op.decode.op_add = true;

    // ADD with rs2=0 uses imm
    assert_eq!(op.compute_arg2(), 0x1234_5678);
}

#[test]
fn test_cpu_operation_compute_arg2_add_with_rs2() {
    let mut op = CpuOperation::new();
    op.rv2 = 0xABCD_EF00;
    op.decode.rs2 = 5; // Non-zero rs2
    op.decode.imm = 0; // Per CPU-A2: when rs2 != 0, imm must be 0
    op.decode.op_add = true;

    // ADD with rs2 != 0: arg2 = rv2 + imm = rv2 + 0 = rv2
    assert_eq!(op.compute_arg2(), 0xABCD_EF00);
}

#[test]
fn test_sign_bit_32_positive() {
    assert!(!CpuOperation::sign_bit_32(0x7FFF_FFFF));
    assert!(!CpuOperation::sign_bit_32(0x0000_0000));
    assert!(!CpuOperation::sign_bit_32(0x1234_5678));
}

#[test]
fn test_sign_bit_32_negative() {
    assert!(CpuOperation::sign_bit_32(0x8000_0000));
    assert!(CpuOperation::sign_bit_32(0xFFFF_FFFF));
    assert!(CpuOperation::sign_bit_32(0x8000_0001));
}

#[test]
fn test_trace_generation_basic() {
    let ops = ops4(CpuOperation {
        decode: DecodeEntry {
            pc: 0x1000,
            rs1: 1,
            rs2: 2,
            rd: 3,
            write_register: true,
            op_add: true,
            ..Default::default()
        },
        rv1: 10,
        rv2: 20,
        res: 30,
        rvd: 30,
        ..Default::default()
    });

    let trace = generate_cpu_trace(&ops);

    assert_eq!(trace.main_table.height, 4);
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);

    // Check first row values
    let row0 = trace.main_table.get_row(0);
    assert_eq!(row0[cols::TIMESTAMP], FE::from(4u64));
    assert_eq!(row0[cols::PC_0], FE::from(0x1000u64));
    assert_eq!(row0[cols::PC_1], FE::zero());
    assert_eq!(row0[cols::RS1], FE::from(1u64));
    assert_eq!(row0[cols::RS2], FE::from(2u64));
    assert_eq!(row0[cols::RD], FE::from(3u64));
    assert_eq!(row0[cols::WRITE_REGISTER], FE::one());
    assert_eq!(row0[cols::ADD], FE::one());
    assert_eq!(row0[cols::SUB], FE::zero());
}

#[test]
fn test_trace_generation_64bit_pc() {
    let ops = ops4(CpuOperation {
        decode: DecodeEntry {
            pc: 0x8000_0000_1234_5678,
            op_add: true,
            ..Default::default()
        },
        ..Default::default()
    });

    let trace = generate_cpu_trace(&ops);
    let row0 = trace.main_table.get_row(0);

    // Check 64-bit PC is split correctly
    assert_eq!(row0[cols::PC_0], FE::from(0x1234_5678u64));
    assert_eq!(row0[cols::PC_1], FE::from(0x8000_0000u64));
    // next_pc set by ops4 helper
    assert_eq!(row0[cols::NEXT_PC_0], FE::from(0x1234_567Cu64));
    assert_eq!(row0[cols::NEXT_PC_1], FE::from(0x8000_0000u64));
}

#[test]
fn test_trace_generation_rv1_dwordwhh() {
    let ops = ops4(CpuOperation {
        decode: DecodeEntry {
            op_add: true,
            ..Default::default()
        },
        rv1: 0xFFFF_EEEE_DDDD_CCCCu64,
        ..Default::default()
    });

    let trace = generate_cpu_trace(&ops);
    let row0 = trace.main_table.get_row(0);

    // rv1 stored as DWordWHH: [Half, Half, Word] - Word is MSB
    assert_eq!(row0[cols::RV1_0], FE::from(0xCCCCu64)); // bits 0-15 (Half)
    assert_eq!(row0[cols::RV1_1], FE::from(0xDDDDu64)); // bits 16-31 (Half)
    assert_eq!(row0[cols::RV1_2], FE::from(0xFFFF_EEEEu64)); // bits 32-63 (Word)
}

#[test]
fn test_trace_generation_arg1_dwordbl() {
    let ops = ops4(CpuOperation {
        decode: DecodeEntry {
            word_instr: false,
            op_add: true,
            ..Default::default()
        },
        rv1: 0x0807_0605_0403_0201u64,
        ..Default::default()
    });

    let trace = generate_cpu_trace(&ops);
    let row0 = trace.main_table.get_row(0);

    // arg1 stored as DWordBL: 8 bytes
    assert_eq!(row0[cols::ARG1_0], FE::from(0x01u64));
    assert_eq!(row0[cols::ARG1_1], FE::from(0x02u64));
    assert_eq!(row0[cols::ARG1_2], FE::from(0x03u64));
    assert_eq!(row0[cols::ARG1_3], FE::from(0x04u64));
    assert_eq!(row0[cols::ARG1_4], FE::from(0x05u64));
    assert_eq!(row0[cols::ARG1_5], FE::from(0x06u64));
    assert_eq!(row0[cols::ARG1_6], FE::from(0x07u64));
    assert_eq!(row0[cols::ARG1_7], FE::from(0x08u64));
}

#[test]
fn test_trace_generation_res_dwordbl() {
    // For op_add, compute_res() calculates arg1 + arg2 (not using self.res directly).
    // Set rv1 to the desired result value since arg1 = rv1 when word_instr=false,
    // and arg2 = 0 (imm default) when rs2=0.
    let ops = ops4(CpuOperation {
        decode: DecodeEntry {
            op_add: true,
            ..Default::default()
        },
        rv1: 0xFEDC_BA98_7654_3210u64,
        ..Default::default()
    });

    let trace = generate_cpu_trace(&ops);
    let row0 = trace.main_table.get_row(0);

    // res = arg1 + arg2 = rv1 + 0 = 0xFEDC_BA98_7654_3210
    // Stored as DWordBL: 8 bytes (little-endian)
    assert_eq!(row0[cols::RES_0], FE::from(0x10u64));
    assert_eq!(row0[cols::RES_1], FE::from(0x32u64));
    assert_eq!(row0[cols::RES_2], FE::from(0x54u64));
    assert_eq!(row0[cols::RES_3], FE::from(0x76u64));
    assert_eq!(row0[cols::RES_4], FE::from(0x98u64));
    assert_eq!(row0[cols::RES_5], FE::from(0xBAu64));
    assert_eq!(row0[cols::RES_6], FE::from(0xDCu64));
    assert_eq!(row0[cols::RES_7], FE::from(0xFEu64));
}

#[test]
fn test_trace_generation_ext_bits() {
    let ops = ops4(CpuOperation {
        decode: DecodeEntry {
            word_instr: true,
            op_add: true,
            ..Default::default()
        },
        rv1: 0x0000_0000_8000_0000u64, // bit 31 set
        res: 0x0000_0000_8000_0000u64, // bit 31 set
        ..Default::default()
    });

    let trace = generate_cpu_trace(&ops);
    let row0 = trace.main_table.get_row(0);

    assert_eq!(row0[cols::RV1_EXT_BIT], FE::one());
    assert_eq!(row0[cols::RES_EXT_BIT], FE::one());
}

#[test]
fn test_bus_interactions_count() {
    let interactions = bus_interactions();

    // Expected interactions:
    // - 8 AND_BYTE
    // - 8 OR_BYTE
    // - 8 XOR_BYTE
    // - 2 MSB16 (rv1_sign_bit, arg2_sign_bit)
    // - 1 MSB8 (res_sign_bit)
    // - 1 ZERO (is_equal for BEQ)
    // - 1 LT (less-than comparison)
    // - 1 M1 (MEMW read rs1 register)
    // - 1 M3 (MEMW read rs2 register)
    // - 1 M5 (MEMW write rd register)
    // - 1 M6 (LOAD from memory)
    // - 1 M7 (STORE to memory)
    // - 4 inline PC (2 reads + 2 writes to Memory bus for x255)
    // - 1 DECODE (instruction fetch)
    // - 1 MUL (multiplication)
    // - 1 DVRM (division/remainder)
    // - 1 SHIFT (shift operations)
    // - 1 BRANCH (branch/jump target calculation)
    // - 1 ECALL (shared bus for HALT, COMMIT, and KECCAK, mult = ECALL)
    // - 1 ARE_BYTES for (RS1, RS2) paired
    // - 1 ARE_BYTES for (RD, 0)
    // - 12 ARE_BYTES (ARG1/ARG2/RES byte pairs: 4 pairs × 3 arrays)
    // Inline PC replaces CM54: -1 CM54, +4 inline PC → net +3 vs pre-PR main.
    // Total: 8 + 8 + 8 + 2 + 1 + 1 + 1 + 1 + 5 + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 12 = 58
    assert_eq!(interactions.len(), 58);
}

#[test]
fn test_column_count() {
    assert_eq!(cols::NUM_COLUMNS, 76);
}

#[test]
fn test_column_arrays() {
    // Verify ARG1, ARG2, RES arrays are correct
    assert_eq!(cols::ARG1.len(), 8);
    assert_eq!(cols::ARG2.len(), 8);
    assert_eq!(cols::RES.len(), 8);

    // Check they're consecutive
    for i in 0..7 {
        assert_eq!(cols::ARG1[i + 1], cols::ARG1[i] + 1);
        assert_eq!(cols::ARG2[i + 1], cols::ARG2[i] + 1);
        assert_eq!(cols::RES[i + 1], cols::RES[i] + 1);
    }
}

// =============================================================================
// ELF execution helpers and from_log tests
// =============================================================================

/// Helper to run an ELF and return the logs and instructions
fn run_elf(path: &str) -> (Vec<executor::vm::logs::Log>, U64HashMap<Instruction>) {
    let elf_data = std::fs::read(path).expect("Failed to read ELF");
    let program = Elf::load(&elf_data).expect("Failed to load ELF");
    let executor = Executor::new(&program, vec![]).expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");
    (result.logs, result.instructions)
}

/// Helper to run an ELF from the program_artifacts directory
fn run_asm_elf(name: &str) -> (Vec<executor::vm::logs::Log>, U64HashMap<Instruction>) {
    run_elf(&format!(
        "{}/executor/program_artifacts/asm/{}.elf",
        env!("CARGO_MANIFEST_DIR").replace("/prover", ""),
        name
    ))
}

#[test]
fn test_trace_from_logs_subw() {
    // subw test - 4 steps (power of 2, works without padding)
    let (logs, instructions) = run_asm_elf("subw");
    let traces = Traces::from_logs(&logs, instructions, &Default::default()).unwrap();

    // Should have SUB instruction with word_instr flag
    let has_sub =
        (0..logs.len()).any(|i| traces.cpus[0].main_table.get_row(i)[cols::SUB] == FE::one());
    assert!(has_sub, "subw.elf should have SUB instruction");
}

#[test]
fn test_cpu_operation_from_log_arith() {
    use executor::vm::instruction::decoding::ArithOp;
    use executor::vm::logs::Log;

    let instruction = Instruction::Arith {
        dst: 10,
        src1: 11,
        src2: 12,
        op: ArithOp::Add,
    };

    let log = Log {
        current_pc: 0x1000,
        next_pc: 0x1004,
        src1_val: 100,
        src2_val: 200,
        dst_val: 300,
    };

    let op = CpuOperation::from_log_and_instruction(&log, 0, instruction);

    assert_eq!(op.decode.pc, 0x1000);
    assert_eq!(op.next_pc, 0x1004);
    assert_eq!(op.decode.rd, 10);
    assert_eq!(op.decode.rs1, 11);
    assert_eq!(op.decode.rs2, 12);
    assert!(op.decode.op_add);
    assert!(op.decode.write_register);
    assert_eq!(op.rv1, 100);
    assert_eq!(op.rv2, 200);
    assert_eq!(op.res, 300);
}

#[test]
fn test_cpu_operation_from_log_branch() {
    use executor::vm::instruction::decoding::Comparison;
    use executor::vm::logs::Log;

    let instruction = Instruction::Branch {
        src1: 5,
        src2: 6,
        cond: Comparison::LessThan,
        offset: 8,
    };

    let log = Log {
        current_pc: 0x2000,
        next_pc: 0x2008, // Branch taken
        src1_val: 10,
        src2_val: 20,
        dst_val: 0,
    };

    let op = CpuOperation::from_log_and_instruction(&log, 4, instruction);

    assert_eq!(op.timestamp, 4);
    assert_eq!(op.decode.pc, 0x2000);
    assert!(op.decode.op_blt);
    assert!(op.decode.signed);
    assert!(op.branch_cond); // 10 < 20
    // For BLT, res is the comparison result (0 or 1), not subtraction
    // res[0] = 1 if arg1 < arg2, res[1..7] = 0 (enforced by SLT res zero constraint)
    assert_eq!(op.res, 1); // 10 < 20 = true
}

#[test]
fn test_cpu_operation_from_log_word_instr() {
    use executor::vm::instruction::decoding::ArithOp;
    use executor::vm::logs::Log;

    let instruction = Instruction::ArithW {
        dst: 1,
        src1: 2,
        src2: 3,
        op: ArithOp::Add,
    };

    let log = Log {
        current_pc: 0x3000,
        next_pc: 0x3004,
        src1_val: 0xFFFF_FFFF_8000_0000, // Would be negative as 32-bit
        src2_val: 1,
        dst_val: 0xFFFF_FFFF_8000_0001, // Result sign-extended
    };

    let op = CpuOperation::from_log_and_instruction(&log, 8, instruction);

    assert!(op.decode.word_instr);
    assert!(op.decode.op_add);
}
