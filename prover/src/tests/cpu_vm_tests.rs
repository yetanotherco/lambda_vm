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

/// TODO: Re-enable when we support padding or have ELF with 4 steps
#[test]
#[ignore]
fn test_trace_from_logs_lui() {
    // LUI: 2 steps - requires padding support
    let logs = run_asm_elf("lui");
    assert_eq!(logs.len(), 2, "lui.elf should have 2 steps");

    let trace = generate_cpu_trace_from_logs(&logs);
    assert_eq!(trace.main_table.height, 4); // min 4 rows for FRI
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);

    // Check first instruction is ADD (LUI is implemented as ADD with imm)
    let row0 = trace.main_table.get_row(0);
    assert_eq!(row0[cols::ADD], FE::one(), "LUI should set ADD flag");
}

/// TODO: Re-enable when we support padding or have ELF with power-of-2 steps
#[test]
#[ignore]
fn test_trace_from_logs_beq() {
    // BEQ test - 6 steps, requires padding support
    let logs = run_asm_elf("beq");

    let trace = generate_cpu_trace_from_logs(&logs);

    // Trace height should be power of 2 >= logs.len()
    assert!(trace.main_table.height.is_power_of_two());
    assert!(trace.main_table.height >= logs.len());

    // Should have BEQ instruction
    let has_beq = (0..logs.len()).any(|i| trace.main_table.get_row(i)[cols::BEQ] == FE::one());
    assert!(has_beq, "beq.elf should have BEQ instruction");
}

/// TODO: Re-enable when we support padding
#[test]
#[ignore]
fn test_trace_from_logs_add_64bit() {
    // add_64bit: 6 steps - requires padding support
    let logs = run_asm_elf("add_64bit");
    assert_eq!(logs.len(), 6, "add_64bit.elf should have 6 steps");

    let trace = generate_cpu_trace_from_logs(&logs);
    // 6 logs -> padded to 8
    assert_eq!(trace.main_table.height, 8);
}

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
// Instruction-specific tests
// =============================================================================

/// TODO: Re-enable when we support padding or have ELF with power-of-2 steps
#[test]
#[ignore]
fn test_from_log_shift_left() {
    // slli_64: tests 64-bit shift left - requires padding support
    let logs = run_asm_elf("slli_64");

    let trace = generate_cpu_trace_from_logs(&logs);

    // Find shift instruction
    let shift_row = (0..trace.main_table.height)
        .find(|&i| trace.main_table.get_row(i)[cols::SHIFT] == FE::one());

    assert!(
        shift_row.is_some(),
        "Should have at least one SHIFT instruction"
    );
}

/// TODO: Re-enable when we support padding or have ELF with power-of-2 steps
#[test]
#[ignore]
fn test_from_log_store_load() {
    // ld_sd: tests load/store double word - requires padding support
    let logs = run_asm_elf("ld_sd");

    let trace = generate_cpu_trace_from_logs(&logs);

    // Should have STORE and LOAD instructions
    let has_store =
        (0..trace.main_table.height).any(|i| trace.main_table.get_row(i)[cols::STORE] == FE::one());
    let has_load =
        (0..trace.main_table.height).any(|i| trace.main_table.get_row(i)[cols::LOAD] == FE::one());

    assert!(has_store, "ld_sd should have STORE instruction");
    assert!(has_load, "ld_sd should have LOAD instruction");
}

/// TODO: Re-enable when we support padding or have ELF with power-of-2 steps
#[test]
#[ignore]
fn test_from_log_branch() {
    // blt: tests branch less than - requires padding support
    let logs = run_asm_elf("blt");

    let trace = generate_cpu_trace_from_logs(&logs);

    // Should have BLT instruction
    let has_blt =
        (0..trace.main_table.height).any(|i| trace.main_table.get_row(i)[cols::BLT] == FE::one());

    assert!(has_blt, "blt should have BLT instruction");
}

/// TODO: Re-enable when we support padding or have ELF with power-of-2 steps
#[test]
#[ignore]
fn test_from_log_mul() {
    // mulw: tests 32-bit multiplication - requires padding support
    let logs = run_asm_elf("mulw");

    let trace = generate_cpu_trace_from_logs(&logs);

    // Should have MUL instruction
    let has_mul = (0..logs.len()).any(|i| trace.main_table.get_row(i)[cols::MUL] == FE::one());

    assert!(has_mul, "mulw should have MUL instruction");
}

/// TODO: Re-enable when we support padding or have ELF with power-of-2 steps
#[test]
#[ignore]
fn test_from_log_div() {
    // divw: tests 32-bit division - requires padding support
    let logs = run_asm_elf("divw");

    let trace = generate_cpu_trace_from_logs(&logs);

    // Should have DIVREM instruction
    let has_divrem = (0..trace.main_table.height)
        .any(|i| trace.main_table.get_row(i)[cols::DIVREM] == FE::one());

    assert!(has_divrem, "divw should have DIVREM instruction");
}

// =============================================================================
// Timestamp tests
// =============================================================================

/// TODO: Re-enable when we support padding or have ELF with power-of-2 steps
#[test]
#[ignore]
fn test_timestamps_sequential() {
    let logs = run_asm_elf("beq");
    let trace = generate_cpu_trace_from_logs(&logs);

    // Timestamps should be 0, 4, 8, 12 (increment by 4)
    for i in 0..logs.len() {
        let row = trace.main_table.get_row(i);
        let expected_timestamp = (i as u64) * 4;
        assert_eq!(
            row[cols::TIMESTAMP],
            FE::from(expected_timestamp),
            "Timestamp at row {} should be {}",
            i,
            expected_timestamp
        );
    }
}

// =============================================================================
// PC and next_pc tests
// =============================================================================

/// TODO: Re-enable when we support padding or have ELF with power-of-2 steps
#[test]
#[ignore]
fn test_pc_values_match_logs() {
    let logs = run_asm_elf("lui");
    let trace = generate_cpu_trace_from_logs(&logs);

    for (i, log) in logs.iter().enumerate() {
        let row = trace.main_table.get_row(i);

        // PC should match log (split into low and high words)
        let expected_pc_low = log.current_pc & 0xFFFF_FFFF;
        let expected_pc_high = log.current_pc >> 32;
        assert_eq!(
            row[cols::PC_0],
            FE::from(expected_pc_low),
            "PC_0 at row {} should match log",
            i
        );
        assert_eq!(
            row[cols::PC_1],
            FE::from(expected_pc_high),
            "PC_1 at row {} should match log",
            i
        );

        // next_pc should match log
        let expected_next_pc_low = log.next_pc & 0xFFFF_FFFF;
        let expected_next_pc_high = log.next_pc >> 32;
        assert_eq!(
            row[cols::NEXT_PC_0],
            FE::from(expected_next_pc_low),
            "NEXT_PC_0 at row {} should match log",
            i
        );
        assert_eq!(
            row[cols::NEXT_PC_1],
            FE::from(expected_next_pc_high),
            "NEXT_PC_1 at row {} should match log",
            i
        );
    }
}

// =============================================================================
// Register value tests
// =============================================================================

/// TODO: Re-enable when we support padding or have ELF with power-of-2 steps
#[test]
#[ignore]
fn test_register_values_from_logs() {
    let logs = run_asm_elf("add_64bit");
    let trace = generate_cpu_trace_from_logs(&logs);

    for (i, log) in logs.iter().enumerate() {
        let row = trace.main_table.get_row(i);

        // rv1 stored as DWordWHH: [Half, Half, Word] - Word is MSB
        let expected_rv1_0 = log.src1_val & 0xFFFF; // bits 0-15 (Half)
        let expected_rv1_1 = (log.src1_val >> 16) & 0xFFFF; // bits 16-31 (Half)
        let expected_rv1_2 = log.src1_val >> 32; // bits 32-63 (Word)

        assert_eq!(
            row[cols::RV1_0],
            FE::from(expected_rv1_0),
            "RV1_0 at row {} should match log.src1_val bits 0-15",
            i
        );
        assert_eq!(
            row[cols::RV1_1],
            FE::from(expected_rv1_1),
            "RV1_1 at row {} should match log.src1_val bits 16-31",
            i
        );
        assert_eq!(
            row[cols::RV1_2],
            FE::from(expected_rv1_2),
            "RV1_2 at row {} should match log.src1_val bits 32-63",
            i
        );
    }
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
    assert_eq!(op.res, 10u64.wrapping_sub(20)); // Subtraction result
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
