//! Tests for the DECODE table.

use executor::elf::{Elf, Segment};
use executor::vm::instruction::decoding::{ArithOp, Instruction};
use executor::vm::memory::U64HashMap;
use math::field::element::FieldElement;

use stark::proof::options::GoldilocksCubicProofOptions;

use crate::tables::decode::{
    DecodeEntry, bus_interactions, cols, commitment_from_elf, generate_decode_trace,
    instructions_from_elf, tables_from_elf, update_multiplicities,
};
use crate::tables::trace_builder::Traces;
use crate::tables::types::{FE, packed_decode as bits};
use crate::test_utils::asm_elf_bytes;
use crate::test_utils::multi_prove_ram;
use crate::test_utils::run_asm_elf;
use crate::{prove, verify_with_options};

// =========================================================================
// Packed decode tests
// =========================================================================

#[test]
fn test_packed_decode_flags() {
    // Test each control flag individually using the constants from packed_decode module.
    // This validates that the constants match the actual bit packing logic.
    let mut entry = DecodeEntry::new();

    // READ_REG1: excludes x0 and x255, so we need rs1 != 0 && rs1 != 255
    entry.read_register1 = true;
    entry.rs1 = 1;
    assert_eq!(
        entry.packed_decode() & (1 << bits::READ_REG1),
        1 << bits::READ_REG1
    );
    entry.read_register1 = false;
    entry.rs1 = 0;

    // READ_REG2: excludes x0, so we need rs2 != 0
    entry.read_register2 = true;
    entry.rs2 = 1;
    assert_eq!(
        entry.packed_decode() & (1 << bits::READ_REG2),
        1 << bits::READ_REG2
    );
    entry.read_register2 = false;
    entry.rs2 = 0;

    // WRITE_REG: excludes x0, so we need rd != 0
    entry.write_register = true;
    entry.rd = 1;
    assert_eq!(
        entry.packed_decode() & (1 << bits::WRITE_REG),
        1 << bits::WRITE_REG
    );
    entry.write_register = false;
    entry.rd = 0;

    // MEMORY_2BYTES
    entry.memory_2bytes = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::MEMORY_2BYTES),
        1 << bits::MEMORY_2BYTES
    );
    entry.memory_2bytes = false;

    // MEMORY_4BYTES
    entry.memory_4bytes = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::MEMORY_4BYTES),
        1 << bits::MEMORY_4BYTES
    );
    entry.memory_4bytes = false;

    // MEMORY_8BYTES
    entry.memory_8bytes = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::MEMORY_8BYTES),
        1 << bits::MEMORY_8BYTES
    );
    entry.memory_8bytes = false;

    // C_TYPE
    entry.c_type = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::C_TYPE),
        1 << bits::C_TYPE
    );
    entry.c_type = false;

    // SIGNED
    entry.signed = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::SIGNED),
        1 << bits::SIGNED
    );
    entry.signed = false;

    // MP_SELECTOR
    entry.mp_selector = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::MP_SELECTOR),
        1 << bits::MP_SELECTOR
    );
    entry.mp_selector = false;

    // MULDIV_SELECTOR
    entry.muldiv_selector = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::MULDIV_SELECTOR),
        1 << bits::MULDIV_SELECTOR
    );
    entry.muldiv_selector = false;

    // WORD_INSTR
    entry.word_instr = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::WORD_INSTR),
        1 << bits::WORD_INSTR
    );
}

#[test]
fn test_packed_decode_alu_flags() {
    // ALU flags - using constants to validate they match the packing logic
    let mut entry = DecodeEntry::new();

    entry.op_add = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::OP_ADD),
        1 << bits::OP_ADD
    );
    entry.op_add = false;

    entry.op_sub = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::OP_SUB),
        1 << bits::OP_SUB
    );
    entry.op_sub = false;

    entry.op_slt = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::OP_SLT),
        1 << bits::OP_SLT
    );
    entry.op_slt = false;

    entry.op_and = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::OP_AND),
        1 << bits::OP_AND
    );
    entry.op_and = false;

    entry.op_or = true;
    assert_eq!(entry.packed_decode() & (1 << bits::OP_OR), 1 << bits::OP_OR);
    entry.op_or = false;

    entry.op_xor = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::OP_XOR),
        1 << bits::OP_XOR
    );
    entry.op_xor = false;

    entry.op_shift = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::OP_SHIFT),
        1 << bits::OP_SHIFT
    );
    entry.op_shift = false;

    entry.op_jalr = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::OP_JALR),
        1 << bits::OP_JALR
    );
    entry.op_jalr = false;

    entry.op_beq = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::OP_BEQ),
        1 << bits::OP_BEQ
    );
    entry.op_beq = false;

    entry.op_blt = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::OP_BLT),
        1 << bits::OP_BLT
    );
    entry.op_blt = false;

    entry.op_load = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::OP_LOAD),
        1 << bits::OP_LOAD
    );
    entry.op_load = false;

    entry.op_store = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::OP_STORE),
        1 << bits::OP_STORE
    );
    entry.op_store = false;

    entry.op_mul = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::OP_MUL),
        1 << bits::OP_MUL
    );
    entry.op_mul = false;

    entry.op_divrem = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::OP_DIVREM),
        1 << bits::OP_DIVREM
    );
    entry.op_divrem = false;

    entry.op_ecall = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::OP_ECALL),
        1 << bits::OP_ECALL
    );
    entry.op_ecall = false;

    entry.op_ebreak = true;
    assert_eq!(
        entry.packed_decode() & (1 << bits::OP_EBREAK),
        1 << bits::OP_EBREAK
    );
}

#[test]
fn test_packed_decode_registers() {
    // Register positions - using constants
    let mut entry = DecodeEntry::new();

    // rs1
    entry.rs1 = 0b10101010;
    let packed = entry.packed_decode();
    let rs1_extracted = (packed >> bits::RS1) & 0xFF;
    assert_eq!(rs1_extracted, 0b10101010);
    entry.rs1 = 0;

    // rs2
    entry.rs2 = 0b11001100;
    let packed = entry.packed_decode();
    let rs2_extracted = (packed >> bits::RS2) & 0xFF;
    assert_eq!(rs2_extracted, 0b11001100);
    entry.rs2 = 0;

    // rd
    entry.rd = 0b11110000;
    let packed = entry.packed_decode();
    let rd_extracted = (packed >> bits::RD) & 0xFF;
    assert_eq!(rd_extracted, 0b11110000);
}

#[test]
fn test_packed_decode_combined() {
    // Test with realistic ADD instruction: rd=10, rs1=5, rs2=6
    // Per decode.md spec: read_register1 at bit 0, read_register2 at bit 1,
    // write_register at bit 2, op_add at bit 11
    let entry = DecodeEntry {
        pc: 0x1000,
        rs1: 5,
        rs2: 6,
        rd: 10,
        read_register1: true,
        read_register2: true,
        write_register: true,
        op_add: true,
        ..Default::default()
    };

    let packed = entry.packed_decode();

    // Verify flags per spec
    assert_eq!(
        packed & (1 << 0),
        1 << 0,
        "read_register1 should be set at bit 0"
    );
    assert_eq!(
        packed & (1 << 1),
        1 << 1,
        "read_register2 should be set at bit 1"
    );
    assert_eq!(
        packed & (1 << 2),
        1 << 2,
        "write_register should be set at bit 2"
    );
    assert_eq!(
        packed & (1 << 11),
        1 << 11,
        "op_add should be set at bit 11"
    );

    // Verify registers per spec: rs1 at bits 27-34, rs2 at bits 35-42, rd at bits 43-50
    assert_eq!((packed >> 27) & 0xFF, 5, "rs1 should be 5");
    assert_eq!((packed >> 35) & 0xFF, 6, "rs2 should be 6");
    assert_eq!((packed >> 43) & 0xFF, 10, "rd should be 10");
}

// =========================================================================
// Padding entry tests
// =========================================================================

#[test]
fn test_padding_entry() {
    let padding = DecodeEntry::padding_entry();

    assert_eq!(padding.pc, 7, "Padding entry should have pc=7");
    assert!(padding.op_ebreak, "Padding entry should have EBREAK=1");

    // All other flags should be false
    assert!(!padding.read_register1);
    assert!(!padding.read_register2);
    assert!(!padding.write_register);
    assert!(!padding.op_add);
    assert!(!padding.op_sub);
    assert_eq!(padding.rs1, 0);
    assert_eq!(padding.rs2, 0);
    assert_eq!(padding.rd, 0);
    assert_eq!(padding.imm, 0);
}

// =========================================================================
// from_instruction tests
// =========================================================================

#[test]
fn test_from_instruction_arith() {
    // ADD x10, x5, x6
    let instr = Instruction::Arith {
        dst: 10,
        src1: 5,
        src2: 6,
        op: ArithOp::Add,
    };

    let entry = DecodeEntry::from_instruction(0x1000, instr);

    assert_eq!(entry.pc, 0x1000);
    assert_eq!(entry.rd, 10);
    assert_eq!(entry.rs1, 5);
    assert_eq!(entry.rs2, 6);
    assert!(entry.read_register1);
    assert!(entry.read_register2);
    assert!(entry.write_register);
    assert!(entry.op_add);
}

#[test]
fn test_from_instruction_arith_imm() {
    // ADDI x10, x5, 100
    let instr = Instruction::ArithImm {
        dst: 10,
        src: 5,
        imm: 100,
        op: ArithOp::Add,
    };

    let entry = DecodeEntry::from_instruction(0x1000, instr);

    assert_eq!(entry.pc, 0x1000);
    assert_eq!(entry.rd, 10);
    assert_eq!(entry.rs1, 5);
    assert_eq!(entry.rs2, 0);
    assert_eq!(entry.imm, 100);
    assert!(entry.read_register1);
    assert!(!entry.read_register2);
    assert!(entry.write_register);
    assert!(entry.op_add);
}

// =========================================================================
// Trace generation tests
// =========================================================================

#[test]
fn test_trace_generation_basic() {
    let mut instructions = U64HashMap::default();
    instructions.insert(
        0x1000,
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
    );
    instructions.insert(
        0x1004,
        Instruction::Arith {
            dst: 4,
            src1: 5,
            src2: 6,
            op: ArithOp::Sub,
        },
    );

    let (trace, _pc_to_row) = generate_decode_trace(&instructions);

    // 2 instructions + 1 CPU padding entry = 3, padded to power of 2 = 4
    assert_eq!(trace.main_table.height, 4);
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);
}

#[test]
fn test_trace_multiplicities() {
    let mut instructions = U64HashMap::default();
    instructions.insert(
        0x1000,
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
    );

    let (mut trace, pc_to_row) = generate_decode_trace(&instructions);

    // PC 0x1000 executed 5 times
    let lookups = vec![0x1000, 0x1000, 0x1000, 0x1000, 0x1000];
    update_multiplicities(&mut trace, &pc_to_row, &lookups);

    // Should be padded to 2 (1 entry -> next power of 2)
    assert_eq!(trace.main_table.height, 2);

    // Find the row with pc=0x1000
    let mut found = false;
    for row_idx in 0..trace.main_table.height {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::PC_0] == FE::from(0x1000u64) {
            assert_eq!(row[cols::MU], FE::from(5u64), "Multiplicity should be 5");
            found = true;
        }
    }
    assert!(found, "Row with pc=0x1000 not found");
}

#[test]
fn test_trace_multiple_instructions_different_multiplicities() {
    let mut instructions = U64HashMap::default();
    instructions.insert(
        0x1000,
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
    );
    instructions.insert(
        0x1004,
        Instruction::Arith {
            dst: 4,
            src1: 5,
            src2: 6,
            op: ArithOp::Sub,
        },
    );

    let (mut trace, pc_to_row) = generate_decode_trace(&instructions);

    // 0x1000 executed 3 times, 0x1004 executed 7 times
    let lookups = vec![
        0x1000, 0x1004, 0x1000, 0x1004, 0x1004, 0x1000, 0x1004, 0x1004, 0x1004, 0x1004,
    ];
    update_multiplicities(&mut trace, &pc_to_row, &lookups);

    // 2 instructions + 1 CPU padding entry = 3, padded to 4
    assert_eq!(trace.main_table.height, 4);

    let mut mu_1000 = None;
    let mut mu_1004 = None;

    for row_idx in 0..trace.main_table.height {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::PC_0] == FE::from(0x1000u64) {
            mu_1000 = Some(row[cols::MU]);
        }
        if row[cols::PC_0] == FE::from(0x1004u64) {
            mu_1004 = Some(row[cols::MU]);
        }
    }

    assert_eq!(mu_1000, Some(FE::from(3u64)), "PC 0x1000 should have mu=3");
    assert_eq!(mu_1004, Some(FE::from(7u64)), "PC 0x1004 should have mu=7");
}

#[test]
fn test_trace_padding_to_power_of_two() {
    let mut instructions = U64HashMap::default();
    instructions.insert(
        0x1000,
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
    );
    instructions.insert(
        0x1004,
        Instruction::Arith {
            dst: 4,
            src1: 5,
            src2: 6,
            op: ArithOp::Sub,
        },
    );
    instructions.insert(
        0x1008,
        Instruction::Arith {
            dst: 7,
            src1: 8,
            src2: 9,
            op: ArithOp::Add,
        },
    );

    let (trace, _pc_to_row) = generate_decode_trace(&instructions);

    // 3 instructions + 1 CPU padding entry = 4, already power of 2
    assert_eq!(
        trace.main_table.height, 4,
        "3 instructions + 1 CPU padding entry = 4 rows"
    );

    // Verify the CPU padding row has pc=1 and all flags=0
    let mut found_cpu_padding = false;
    for row_idx in 0..trace.main_table.height {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::PC_0] == FE::from(1u64) {
            assert_eq!(
                row[cols::PACKED_DECODE],
                FE::zero(),
                "CPU padding entry should have all flags=0"
            );
            assert_eq!(
                row[cols::MU],
                FE::zero(),
                "CPU padding entry should have mu=0"
            );
            found_cpu_padding = true;
        }
    }
    assert!(found_cpu_padding, "CPU padding row with pc=1 not found");
}

#[test]
fn test_trace_dword_encoding() {
    // Test 64-bit PC and immediate encoding as DWordWL
    let mut instructions = U64HashMap::default();
    instructions.insert(
        0xDEAD_BEEF_1234_5678,
        Instruction::ArithImm {
            dst: 1,
            src: 2,
            imm: 0x8765_4321u32 as i32, // Will be sign-extended
            op: ArithOp::Add,
        },
    );

    let (trace, _pc_to_row) = generate_decode_trace(&instructions);

    // Find the row (could be row 0 or 1 due to HashMap ordering)
    let mut found = false;
    for row_idx in 0..trace.main_table.height {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::PC_0] == FE::from(0x1234_5678u64) {
            // PC low word
            assert_eq!(row[cols::PC_0], FE::from(0x1234_5678u64));
            // PC high word
            assert_eq!(row[cols::PC_1], FE::from(0xDEAD_BEEFu64));
            found = true;
        }
    }
    assert!(found, "Row with expected PC not found");
}

// =========================================================================
// Bus interaction tests
// =========================================================================

#[test]
fn test_bus_interactions_count() {
    let interactions = bus_interactions();

    // DECODE table should have exactly 1 interaction (receiver for DECODE bus)
    assert_eq!(
        interactions.len(),
        1,
        "DECODE should have 1 bus interaction"
    );
}

#[test]
fn test_bus_interactions_is_receiver() {
    let interactions = bus_interactions();

    // The single interaction should be a receiver (is_sender = false)
    assert!(
        !interactions[0].is_sender,
        "DECODE should be a receiver, not sender"
    );
}

// =========================================================================
// Precomputed commitment tests
// =========================================================================

#[test]
fn test_compute_precomputed_commitment_deterministic() {
    use crate::tables::decode::compute_precomputed_commitment;
    use stark::proof::options::ProofOptions;

    // Same instructions should produce same commitment
    let mut instructions = U64HashMap::default();
    instructions.insert(
        0x1000,
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
    );
    instructions.insert(
        0x1004,
        Instruction::Arith {
            dst: 4,
            src1: 5,
            src2: 6,
            op: ArithOp::Sub,
        },
    );

    let options = ProofOptions::default_test_options();

    let commitment1 = compute_precomputed_commitment(&instructions, &options);
    let commitment2 = compute_precomputed_commitment(&instructions, &options);

    assert_eq!(
        commitment1, commitment2,
        "Same instructions should produce same commitment"
    );
}

#[test]
fn test_compute_precomputed_commitment_different_programs() {
    use crate::tables::decode::compute_precomputed_commitment;
    use stark::proof::options::ProofOptions;

    let options = ProofOptions::default_test_options();

    // Program A: ADD instruction
    let mut program_a = U64HashMap::default();
    program_a.insert(
        0x1000,
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
    );

    // Program B: SUB instruction (different from A)
    let mut program_b = U64HashMap::default();
    program_b.insert(
        0x1000,
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Sub, // Different operation
        },
    );

    let commitment_a = compute_precomputed_commitment(&program_a, &options);
    let commitment_b = compute_precomputed_commitment(&program_b, &options);

    assert_ne!(
        commitment_a, commitment_b,
        "Different programs should produce different commitments"
    );
}

#[test]
fn test_compute_precomputed_commitment_different_pc() {
    use crate::tables::decode::compute_precomputed_commitment;
    use stark::proof::options::ProofOptions;

    let options = ProofOptions::default_test_options();

    // Program A: instruction at PC 0x1000
    let mut program_a = U64HashMap::default();
    program_a.insert(
        0x1000,
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
    );

    // Program B: same instruction at different PC
    let mut program_b = U64HashMap::default();
    program_b.insert(
        0x2000, // Different PC
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
    );

    let commitment_a = compute_precomputed_commitment(&program_a, &options);
    let commitment_b = compute_precomputed_commitment(&program_b, &options);

    assert_ne!(
        commitment_a, commitment_b,
        "Programs with different PCs should produce different commitments"
    );
}

// =========================================================================
// instructions_from_elf tests (verifier vs executor consistency)
// =========================================================================

/// Test that instructions_from_elf produces the same result as the executor.
#[test]
fn test_instructions_from_elf_matches_executor() {
    // Run executor to get instructions
    let (_elf, _logs, executor_instructions) = run_asm_elf("arith_8");

    // Load the same ELF and extract instructions directly
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let elf_path = manifest_dir
        .parent()
        .unwrap()
        .join("executor/program_artifacts/asm/arith_8.elf");
    let elf_bytes = std::fs::read(&elf_path).expect("Failed to read ELF file");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");

    let verifier_instructions =
        instructions_from_elf(&elf).expect("Failed to extract instructions");

    // Compare via DecodeEntry (what matters for the DECODE table)
    for (pc, executor_instr) in executor_instructions.iter() {
        let verifier_instr = verifier_instructions
            .get(pc)
            .unwrap_or_else(|| panic!("Verifier missing instruction at PC {:#x}", pc));

        // Compare by converting to DecodeEntry - this is what the DECODE table uses
        let executor_entry = DecodeEntry::from_instruction(*pc, *executor_instr);
        let verifier_entry = DecodeEntry::from_instruction(*pc, *verifier_instr);

        assert_eq!(
            executor_entry.packed_decode(),
            verifier_entry.packed_decode(),
            "packed_decode mismatch at PC {:#x}",
            pc
        );
        assert_eq!(
            executor_entry.imm, verifier_entry.imm,
            "imm mismatch at PC {:#x}",
            pc
        );
    }

    // Verifier may have more instructions (all executable code vs only executed code)
    // but every executed instruction must match
    assert!(
        verifier_instructions.len() >= executor_instructions.len(),
        "Verifier should have at least as many instructions as executor"
    );
}

/// Test instructions_from_elf with a more complex program.
#[test]
fn test_instructions_from_elf_matches_executor_complex() {
    let (_elf, _logs, executor_instructions) = run_asm_elf("all_instructions_64");

    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let elf_path = manifest_dir
        .parent()
        .unwrap()
        .join("executor/program_artifacts/asm/all_instructions_64.elf");
    let elf_bytes = std::fs::read(&elf_path).expect("Failed to read ELF file");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");

    let verifier_instructions =
        instructions_from_elf(&elf).expect("Failed to extract instructions");

    // Every executed instruction must be present and match
    for (pc, executor_instr) in executor_instructions.iter() {
        let verifier_instr = verifier_instructions
            .get(pc)
            .unwrap_or_else(|| panic!("Verifier missing instruction at PC {:#x}", pc));

        // Compare via DecodeEntry
        let executor_entry = DecodeEntry::from_instruction(*pc, *executor_instr);
        let verifier_entry = DecodeEntry::from_instruction(*pc, *verifier_instr);

        assert_eq!(
            executor_entry.packed_decode(),
            verifier_entry.packed_decode(),
            "packed_decode mismatch at PC {:#x}",
            pc
        );
        assert_eq!(
            executor_entry.imm, verifier_entry.imm,
            "imm mismatch at PC {:#x}",
            pc
        );
    }
}

/// Test that instructions_from_elf includes all executable instructions,
/// not just the ones that were executed.
#[test]
fn test_instructions_from_elf_includes_all_executable() {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let elf_path = manifest_dir
        .parent()
        .unwrap()
        .join("executor/program_artifacts/asm/all_branches_16.elf");
    let elf_bytes = std::fs::read(&elf_path).expect("Failed to read ELF file");
    let elf = Elf::load(&elf_bytes).expect("Failed to load ELF");

    let instructions = instructions_from_elf(&elf).expect("Failed to extract instructions");

    // Should have decoded all executable code
    assert!(
        !instructions.is_empty(),
        "Should have extracted some instructions"
    );

    // All PCs should be 4-byte aligned
    for (pc, _) in instructions.iter() {
        assert_eq!(pc % 4, 0, "PC {:#x} is not 4-byte aligned", pc);
    }
}

// =========================================================================
// Soundness tests (prover/verifier decoupling)
// =========================================================================

/// SECURITY TEST: Verifier with different ELF rejects proof.
///
/// This test proves the security model works:
/// - Prover runs program A, generates proof with DECODE commitment from ELF A
/// - Verifier has ELF B, computes DECODE commitment from ELF B
/// - Commitments differ → Fiat-Shamir challenges differ → verification FAILS
///
/// This demonstrates that a verifier who independently has the correct ELF
/// will reject proofs from a prover who ran a different program.
#[test]
fn test_decode_soundness_different_elf_rejected() {
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use stark::proof::options::ProofOptions;
    use stark::traits::AIR;
    use stark::verifier::{IsStarkVerifier, Verifier};

    use crate::tables::decode::{self, commitment_from_elf};
    use crate::tables::trace_builder::Traces;
    use crate::tables::types::{GoldilocksExtension, GoldilocksField};
    use crate::test_utils::{
        create_bitwise_air, create_branch_air, create_cpu_air, create_decode_air, create_halt_air,
        create_load_air, create_lt_air, create_memw_air,
    };

    type F = GoldilocksField;
    type E = GoldilocksExtension;

    let proof_options = ProofOptions::default_test_options();

    // Load two DIFFERENT ELF files
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let elf_path_a = manifest_dir
        .parent()
        .unwrap()
        .join("executor/program_artifacts/asm/arith_8.elf");
    let elf_path_b = manifest_dir
        .parent()
        .unwrap()
        .join("executor/program_artifacts/asm/test_sub_8.elf");

    let elf_bytes_a = std::fs::read(&elf_path_a).expect("Failed to read ELF A");
    let elf_bytes_b = std::fs::read(&elf_path_b).expect("Failed to read ELF B");

    let elf_a = Elf::load(&elf_bytes_a).expect("Failed to load ELF A");
    let elf_b = Elf::load(&elf_bytes_b).expect("Failed to load ELF B");

    // Verify the two programs produce different commitments
    let commitment_a = commitment_from_elf(&elf_a, &proof_options).expect("commitment A");
    let commitment_b = commitment_from_elf(&elf_b, &proof_options).expect("commitment B");
    assert_ne!(
        commitment_a, commitment_b,
        "Test requires two different programs with different commitments"
    );

    // =========================================================================
    // PROVER: Runs program A, builds traces, generates proof
    // =========================================================================
    let executor_a =
        executor::vm::execution::Executor::new(&elf_a, vec![]).expect("Failed to create executor");
    let result_a = executor_a.run().expect("Failed to run program A");

    let mut traces =
        Traces::from_logs_minimal(&result_a.logs, result_a.instructions, &Default::default())
            .unwrap();

    // Prover builds AIRs with commitment from ELF A
    let prover_cpu_air = create_cpu_air(&proof_options);
    let prover_bitwise_air = create_bitwise_air(&proof_options);
    let prover_lt_air = create_lt_air(&proof_options);
    let prover_memw_air = create_memw_air(&proof_options);
    let prover_load_air = create_load_air(&proof_options);
    let prover_branch_air = create_branch_air(&proof_options);
    let prover_halt_air = create_halt_air(&proof_options);
    let prover_decode_air = create_decode_air(&proof_options).with_preprocessed(
        commitment_a, // Prover uses commitment from ELF A
        decode::NUM_PRECOMPUTED_COLS,
    );

    let air_trace_pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&prover_cpu_air, &mut traces.cpus[0], &()),
        (&prover_bitwise_air, &mut traces.bitwise, &()),
        (&prover_lt_air, &mut traces.lts[0], &()),
        (&prover_memw_air, &mut traces.memws[0], &()),
        (&prover_load_air, &mut traces.loads[0], &()),
        (&prover_branch_air, &mut traces.branches[0], &()),
        (&prover_halt_air, &mut traces.halt, &()),
        (&prover_decode_air, &mut traces.decode, &()),
    ];

    let proof = multi_prove_ram(air_trace_pairs, &mut DefaultTranscript::<E>::new(&[]))
        .expect("Prover failed to generate proof");

    // =========================================================================
    // VERIFIER: Has ELF B (different program!), computes commitment from it
    // =========================================================================
    let verifier_cpu_air = create_cpu_air(&proof_options);
    let verifier_bitwise_air = create_bitwise_air(&proof_options);
    let verifier_lt_air = create_lt_air(&proof_options);
    let verifier_memw_air = create_memw_air(&proof_options);
    let verifier_load_air = create_load_air(&proof_options);
    let verifier_branch_air = create_branch_air(&proof_options);
    let verifier_halt_air = create_halt_air(&proof_options);
    let verifier_decode_air = create_decode_air(&proof_options).with_preprocessed(
        commitment_b, // Verifier uses commitment from ELF B (DIFFERENT!)
        decode::NUM_PRECOMPUTED_COLS,
    );

    let verifier_airs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = vec![
        &verifier_cpu_air,
        &verifier_bitwise_air,
        &verifier_lt_air,
        &verifier_memw_air,
        &verifier_load_air,
        &verifier_branch_air,
        &verifier_halt_air,
        &verifier_decode_air,
    ];

    let result = Verifier::multi_verify(
        &verifier_airs,
        &proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &FieldElement::zero(),
    );

    // With different ELFs, verification should FAIL (secure!)
    assert!(
        !result,
        "Verifier with different ELF should REJECT the proof"
    );
}

/// SECURITY TEST: Verifier with same ELF accepts proof.
///
/// Complementary test: when prover and verifier have the SAME ELF,
/// verification should succeed.
#[test]
fn test_decode_soundness_same_elf_accepted() {
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use stark::proof::options::ProofOptions;
    use stark::verifier::{IsStarkVerifier, Verifier};

    use crate::VmAirs;
    use crate::tables::types::GoldilocksExtension;

    type E = GoldilocksExtension;

    let proof_options = ProofOptions::default_test_options();

    // Load the SAME ELF for both prover and verifier
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let elf_path = manifest_dir
        .parent()
        .unwrap()
        .join("executor/program_artifacts/asm/arith_8.elf");

    let elf_bytes = std::fs::read(&elf_path).expect("Failed to read ELF");

    // Prover loads ELF
    let prover_elf = Elf::load(&elf_bytes).expect("Prover: failed to load ELF");
    // Verifier loads ELF independently (same bytes)
    let verifier_elf = Elf::load(&elf_bytes).expect("Verifier: failed to load ELF");

    // =========================================================================
    // PROVER: Runs program, builds traces, generates proof
    // =========================================================================
    let executor = executor::vm::execution::Executor::new(&prover_elf, vec![])
        .expect("Failed to create executor");
    let result = executor.run().expect("Failed to run program");

    let mut traces = Traces::from_elf_and_logs(
        &prover_elf,
        &result.logs,
        &Default::default(),
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .unwrap();
    let table_counts = traces.table_counts();
    let prover_airs = VmAirs::new(
        &prover_elf,
        &proof_options,
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
    );

    let proof = multi_prove_ram(
        prover_airs.air_trace_pairs(&mut traces),
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("Prover failed to generate proof");
    // =========================================================================
    // VERIFIER: Loads same ELF independently, verifies proof
    // =========================================================================
    let verifier_airs = VmAirs::new(
        &verifier_elf,
        &proof_options,
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
    );
    let verifier_air_refs = verifier_airs.air_refs();
    let mut replay_transcript = DefaultTranscript::<E>::new(&[]);
    let expected_bus_balance = crate::compute_expected_commit_bus_balance(
        &verifier_air_refs,
        &proof,
        &traces.public_output_bytes,
        &mut replay_transcript,
    )
    .expect("fingerprint collision in test");

    let result = Verifier::multi_verify(
        &verifier_air_refs,
        &proof,
        &mut DefaultTranscript::<E>::new(&[]),
        &expected_bus_balance,
    );

    // With same ELF, verification should SUCCEED
    assert!(result, "Verifier with same ELF should ACCEPT the proof");
}

#[test]
fn test_tables_from_elf_single_executable_segment() {
    // ADDI x1, x0, 42  (opcode: 0x02a00093)
    // ADDI x2, x1, 10  (opcode: 0x00a08113)
    let elf = Elf {
        entry_point: 0x1000,
        data: vec![Segment {
            base_addr: 0x1000,
            values: vec![0x02a00093, 0x00a08113],
            is_executable: true,
        }],
    };

    let tables = tables_from_elf(&elf).unwrap();

    // Check DECODE table
    assert_eq!(tables.pc_to_row.len(), 3); // 2 instructions + CPU padding
    assert!(tables.pc_to_row.contains_key(&0x1000));
    assert!(tables.pc_to_row.contains_key(&0x1004));
    assert!(
        tables
            .pc_to_row
            .contains_key(&crate::tables::cpu::CPU_PADDING_PC)
    );
}

#[test]
fn test_tables_from_elf_mixed_segments() {
    // Executable segment with instructions
    // Data segment with data (not included in DECODE)
    let elf = Elf {
        entry_point: 0x1000,
        data: vec![
            Segment {
                base_addr: 0x1000,
                values: vec![0x02a00093], // ADDI instruction
                is_executable: true,
            },
            Segment {
                base_addr: 0x2000,
                values: vec![0xDEADBEEF, 0xCAFEBABE], // Data
                is_executable: false,
            },
        ],
    };

    let tables = tables_from_elf(&elf).unwrap();

    // DECODE: only executable segment (1 instruction + CPU padding)
    assert_eq!(tables.pc_to_row.len(), 2);
    assert!(tables.pc_to_row.contains_key(&0x1000));
    assert!(!tables.pc_to_row.contains_key(&0x2000)); // Data not in decode
}

#[test]
fn test_tables_from_elf_empty() {
    let elf = Elf {
        entry_point: 0x1000,
        data: vec![],
    };

    let tables = tables_from_elf(&elf).unwrap();

    // DECODE: only CPU padding entry
    assert_eq!(tables.pc_to_row.len(), 1);
    assert!(
        tables
            .pc_to_row
            .contains_key(&crate::tables::cpu::CPU_PADDING_PC)
    );
}

// =========================================================================
// verify_with_options: optional decode_commitment parameter
// =========================================================================

#[test]
fn decode_commitment_some_matches_default_path() {
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("prove failed");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let options = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    let decode_c = commitment_from_elf(&elf, &options).expect("decode commitment");

    let default_ok = verify_with_options(&vm_proof, &elf_bytes, &options, None)
        .expect("verify with None should not error");
    let explicit_ok = verify_with_options(&vm_proof, &elf_bytes, &options, Some(decode_c))
        .expect("verify with Some(correct) should not error");

    assert!(default_ok, "default path must accept the proof");
    assert!(
        explicit_ok,
        "Some(correct_commitment) must accept the proof"
    );
}

#[test]
fn decode_commitment_wrong_value_rejects() {
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("prove failed");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let options = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    // Flip a byte in the correct commitment so the Fiat-Shamir transcripts diverge.
    let mut wrong = commitment_from_elf(&elf, &options).expect("decode commitment");
    wrong[0] ^= 0xFF;

    let result = verify_with_options(&vm_proof, &elf_bytes, &options, Some(wrong))
        .expect("verify must not return Err — Fiat-Shamir mismatch is Ok(false)");
    assert!(
        !result,
        "tampered decode commitment must cause Fiat-Shamir rejection",
    );
}

#[test]
fn decode_commitment_zero_bytes_rejects() {
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("prove failed");
    let options = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    // [0u8; 32] is the most plausible accidental default — passing it must
    // not pass verification.
    let result = verify_with_options(&vm_proof, &elf_bytes, &options, Some([0u8; 32]))
        .expect("verify must not return Err — Fiat-Shamir mismatch is Ok(false)");
    assert!(
        !result,
        "all-zero decode commitment must cause Fiat-Shamir rejection",
    );
}

/// DECODE preprocessed commitment for the `sub` asm test ELF at blowup=2,
/// computed offline once. Mirrors how the recursion guest embeds the
/// commitment as a compile-time constant for its inner program. If the
/// AIR or FFT pipeline changes, this drifts and the test fails —
/// regenerate via the `print_decode_commitment_for_sub` helper below.
const SUB_DECODE_COMMITMENT_BLOWUP_2: [u8; 32] = [
    0x00, 0x83, 0x59, 0xa3, 0x34, 0x5f, 0x86, 0x79, 0x59, 0x71, 0xc8, 0x71, 0x54, 0x2c, 0xc4, 0xac,
    0x8b, 0x9c, 0x48, 0x9b, 0x25, 0xa3, 0x6a, 0xc7, 0x48, 0xee, 0x71, 0xe6, 0x77, 0xfb, 0x59, 0xfa,
];

#[test]
fn decode_commitment_compile_time_const_accepts() {
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("prove failed");
    let options = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    // Pass the OFFLINE-COMPUTED const directly — mimics the recursion guest's
    // workflow where the value lives in the caller's compiled binary.
    let result = verify_with_options(
        &vm_proof,
        &elf_bytes,
        &options,
        Some(SUB_DECODE_COMMITMENT_BLOWUP_2),
    )
    .expect("verify must not return Err");
    assert!(
        result,
        "verifier must accept the offline-computed decode commitment",
    );
}

#[test]
#[ignore = "prints decode commitment for the sub asm ELF so SUB_DECODE_COMMITMENT_BLOWUP_2 \
            can be regenerated; run with --ignored --nocapture"]
fn print_decode_commitment_for_sub() {
    let elf_bytes = asm_elf_bytes("sub");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let options = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");
    let c = commitment_from_elf(&elf, &options).expect("decode commitment");
    eprintln!("SUB_DECODE_COMMITMENT_BLOWUP_2 (sub.elf, blowup=2):");
    eprintln!("{c:02x?}");
}
