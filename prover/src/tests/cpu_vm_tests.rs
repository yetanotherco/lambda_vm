//! Integration tests for the CPU table with real VM execution.
//!
//! These tests verify that:
//! - CpuOperation::from_log correctly converts executor logs
//! - generate_cpu_trace_from_logs produces valid traces
//! - Traces from real program executions have correct values

use executor::{elf::Elf, vm::execution::run_program};

use crate::tables64::cpu::{CpuOperation, cols, generate_cpu_trace_from_logs};
use crate::tables64::types::FE;

/// Helper to run an ELF and return the logs
fn run_elf(path: &str) -> Vec<executor::vm::logs::Log> {
    let elf_data = std::fs::read(path).expect("Failed to read ELF");
    let program = Elf::load(&elf_data).expect("Failed to load ELF");
    let (_results, logs) =
        run_program(program.image, program.entry_point, vec![]).expect("Failed to run program");
    logs
}

/// Helper to run an ELF from the program_artifacts directory
fn run_asm_elf(name: &str) -> Vec<executor::vm::logs::Log> {
    run_elf(&format!(
        "{}/executor/program_artifacts/asm/{}.elf",
        env!("CARGO_MANIFEST_DIR").replace("/prover", ""),
        name
    ))
}

// =============================================================================
// Basic trace generation tests
// =============================================================================

#[test]
fn test_trace_from_logs_subw() {
    // subw test - 4 steps (power of 2, works without padding)
    let logs = run_asm_elf("subw");
    assert_eq!(logs.len(), 4, "subw.elf should have 4 steps");

    let trace = generate_cpu_trace_from_logs(&logs);

    assert_eq!(trace.main_table.height, 4);

    // Should have SUB instruction with word_instr flag
    let has_sub = (0..logs.len()).any(|i| trace.main_table.get_row(i)[cols::SUB] == FE::one());
    assert!(has_sub, "subw.elf should have SUB instruction");
}

// =============================================================================
// CpuOperation::from_log unit tests
// =============================================================================

#[test]
fn test_cpu_operation_from_log_arith() {
    use executor::vm::instruction::decoding::{ArithOp, Instruction};
    use executor::vm::logs::Log;

    let log = Log {
        instruction: Instruction::Arith {
            dst: 10,
            src1: 11,
            src2: 12,
            op: ArithOp::Add,
        },
        current_pc: 0x1000,
        next_pc: 0x1004,
        src1_val: 100,
        src2_val: 200,
        dst_val: 300,
    };

    let op = CpuOperation::from_log(&log, 0);

    assert_eq!(op.pc, 0x1000);
    assert_eq!(op.next_pc, 0x1004);
    assert_eq!(op.rd, 10);
    assert_eq!(op.rs1, 11);
    assert_eq!(op.rs2, 12);
    assert!(op.op_add);
    assert!(op.write_register);
    assert_eq!(op.rv1, 100);
    assert_eq!(op.rv2, 200);
    assert_eq!(op.res, 300);
}

#[test]
fn test_cpu_operation_from_log_branch() {
    use executor::vm::instruction::decoding::{Comparison, Instruction};
    use executor::vm::logs::Log;

    let log = Log {
        instruction: Instruction::Branch {
            src1: 5,
            src2: 6,
            cond: Comparison::LessThan,
            offset: 8,
        },
        current_pc: 0x2000,
        next_pc: 0x2008, // Branch taken
        src1_val: 10,
        src2_val: 20,
        dst_val: 0,
    };

    let op = CpuOperation::from_log(&log, 4);

    assert_eq!(op.timestamp, 4);
    assert_eq!(op.pc, 0x2000);
    assert!(op.op_blt);
    assert!(op.signed);
    assert!(op.branch_cond); // 10 < 20
    // For BLT, res is the comparison result (0 or 1), not subtraction
    // res[0] = 1 if arg1 < arg2, res[1..7] = 0 (enforced by SLT res zero constraint)
    assert_eq!(op.res, 1); // 10 < 20 = true
}

#[test]
fn test_cpu_operation_from_log_word_instr() {
    use executor::vm::instruction::decoding::{ArithOp, Instruction};
    use executor::vm::logs::Log;

    let log = Log {
        instruction: Instruction::ArithW {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
        current_pc: 0x3000,
        next_pc: 0x3004,
        src1_val: 0xFFFF_FFFF_8000_0000, // Would be negative as 32-bit
        src2_val: 1,
        dst_val: 0xFFFF_FFFF_8000_0001, // Result sign-extended
    };

    let op = CpuOperation::from_log(&log, 8);

    assert!(op.word_instr);
    assert!(op.op_add);
}
