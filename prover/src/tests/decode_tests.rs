//! Tests for the DECODE table.

use executor::vm::instruction::decoding::{ArithOp, Instruction};
use executor::vm::memory::U64HashMap;

use crate::tables::decode::{
    DecodeEntry, bus_interactions, cols, generate_decode_trace, update_multiplicities,
};
use crate::tables::types::FE;

// =========================================================================
// Packed decode tests
// =========================================================================

#[test]
fn test_packed_decode_flags() {
    // Test each control flag individually
    // Format matches decode.md spec
    let mut entry = DecodeEntry::new();

    // Bit 0: read_register1
    // Note: packed_decode excludes x0 and x255, so we need rs1 != 0 && rs1 != 255
    entry.read_register1 = true;
    entry.rs1 = 1; // Use a valid register (not x0 or x255)
    assert_eq!(entry.packed_decode() & (1 << 0), 1 << 0);
    entry.read_register1 = false;
    entry.rs1 = 0;

    // Bit 1: read_register2
    // Note: packed_decode excludes x0, so we need rs2 != 0
    entry.read_register2 = true;
    entry.rs2 = 1; // Use a valid register (not x0)
    assert_eq!(entry.packed_decode() & (1 << 1), 1 << 1);
    entry.read_register2 = false;
    entry.rs2 = 0;

    // Bit 2: write_register
    // Note: packed_decode excludes x0, so we need rd != 0
    entry.write_register = true;
    entry.rd = 1; // Use a valid register (not x0)
    assert_eq!(entry.packed_decode() & (1 << 2), 1 << 2);
    entry.write_register = false;
    entry.rd = 0;

    // Bit 3: memory_2bytes
    entry.memory_2bytes = true;
    assert_eq!(entry.packed_decode() & (1 << 3), 1 << 3);
    entry.memory_2bytes = false;

    // Bit 4: memory_4bytes
    entry.memory_4bytes = true;
    assert_eq!(entry.packed_decode() & (1 << 4), 1 << 4);
    entry.memory_4bytes = false;

    // Bit 5: memory_8bytes
    entry.memory_8bytes = true;
    assert_eq!(entry.packed_decode() & (1 << 5), 1 << 5);
    entry.memory_8bytes = false;

    // Bit 6: c_type
    entry.c_type = true;
    assert_eq!(entry.packed_decode() & (1 << 6), 1 << 6);
    entry.c_type = false;

    // Bit 7: signed
    entry.signed = true;
    assert_eq!(entry.packed_decode() & (1 << 7), 1 << 7);
    entry.signed = false;

    // Bit 8: mp_selector
    entry.mp_selector = true;
    assert_eq!(entry.packed_decode() & (1 << 8), 1 << 8);
    entry.mp_selector = false;

    // Bit 9: muldiv_selector
    entry.muldiv_selector = true;
    assert_eq!(entry.packed_decode() & (1 << 9), 1 << 9);
    entry.muldiv_selector = false;

    // Bit 10: word_instr
    entry.word_instr = true;
    assert_eq!(entry.packed_decode() & (1 << 10), 1 << 10);
    entry.word_instr = false;
}

#[test]
fn test_packed_decode_alu_flags() {
    // ALU flags at bits 11-26 per decode.md spec
    let mut entry = DecodeEntry::new();

    // Bit 11: ADD
    entry.op_add = true;
    assert_eq!(entry.packed_decode() & (1 << 11), 1 << 11);
    entry.op_add = false;

    // Bit 12: SUB
    entry.op_sub = true;
    assert_eq!(entry.packed_decode() & (1 << 12), 1 << 12);
    entry.op_sub = false;

    // Bit 13: SLT
    entry.op_slt = true;
    assert_eq!(entry.packed_decode() & (1 << 13), 1 << 13);
    entry.op_slt = false;

    // Bit 14: AND
    entry.op_and = true;
    assert_eq!(entry.packed_decode() & (1 << 14), 1 << 14);
    entry.op_and = false;

    // Bit 15: OR
    entry.op_or = true;
    assert_eq!(entry.packed_decode() & (1 << 15), 1 << 15);
    entry.op_or = false;

    // Bit 16: XOR
    entry.op_xor = true;
    assert_eq!(entry.packed_decode() & (1 << 16), 1 << 16);
    entry.op_xor = false;

    // Bit 17: SHIFT
    entry.op_shift = true;
    assert_eq!(entry.packed_decode() & (1 << 17), 1 << 17);
    entry.op_shift = false;

    // Bit 18: JALR
    entry.op_jalr = true;
    assert_eq!(entry.packed_decode() & (1 << 18), 1 << 18);
    entry.op_jalr = false;

    // Bit 19: BEQ
    entry.op_beq = true;
    assert_eq!(entry.packed_decode() & (1 << 19), 1 << 19);
    entry.op_beq = false;

    // Bit 20: BLT
    entry.op_blt = true;
    assert_eq!(entry.packed_decode() & (1 << 20), 1 << 20);
    entry.op_blt = false;

    // Bit 21: LOAD
    entry.op_load = true;
    assert_eq!(entry.packed_decode() & (1 << 21), 1 << 21);
    entry.op_load = false;

    // Bit 22: STORE
    entry.op_store = true;
    assert_eq!(entry.packed_decode() & (1 << 22), 1 << 22);
    entry.op_store = false;

    // Bit 23: MUL
    entry.op_mul = true;
    assert_eq!(entry.packed_decode() & (1 << 23), 1 << 23);
    entry.op_mul = false;

    // Bit 24: DIVREM
    entry.op_divrem = true;
    assert_eq!(entry.packed_decode() & (1 << 24), 1 << 24);
    entry.op_divrem = false;

    // Bit 25: ECALL
    entry.op_ecall = true;
    assert_eq!(entry.packed_decode() & (1 << 25), 1 << 25);
    entry.op_ecall = false;

    // Bit 26: EBREAK
    entry.op_ebreak = true;
    assert_eq!(entry.packed_decode() & (1 << 26), 1 << 26);
    entry.op_ebreak = false;
}

#[test]
fn test_packed_decode_registers() {
    // Register positions per decode.md spec:
    // rs1 at bits [27:35), rs2 at bits [35:43), rd at bits [43:51)
    let mut entry = DecodeEntry::new();

    // rs1 at bits [27:35)
    entry.rs1 = 0b10101010;
    let packed = entry.packed_decode();
    let rs1_extracted = (packed >> 27) & 0xFF;
    assert_eq!(rs1_extracted, 0b10101010);
    entry.rs1 = 0;

    // rs2 at bits [35:43)
    entry.rs2 = 0b11001100;
    let packed = entry.packed_decode();
    let rs2_extracted = (packed >> 35) & 0xFF;
    assert_eq!(rs2_extracted, 0b11001100);
    entry.rs2 = 0;

    // rd at bits [43:51)
    entry.rd = 0b11110000;
    let packed = entry.packed_decode();
    let rd_extracted = (packed >> 43) & 0xFF;
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

    // Should be padded to power of 2
    assert_eq!(trace.main_table.height, 2);
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

    assert_eq!(trace.main_table.height, 2);

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

    assert_eq!(trace.main_table.height, 4, "3 entries should pad to 4 rows");

    // Verify the padding row has pc=7 and EBREAK flag
    let padding = DecodeEntry::padding_entry();
    let padding_packed = padding.packed_decode();

    // Find a row with pc=7 (padding)
    let mut found_padding = false;
    for row_idx in 0..trace.main_table.height {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::PC_0] == FE::from(7u64) {
            assert_eq!(
                row[cols::PACKED_DECODE],
                FE::from(padding_packed),
                "Padding should have EBREAK set"
            );
            assert_eq!(row[cols::MU], FE::zero(), "Padding should have mu=0");
            found_padding = true;
        }
    }
    assert!(found_padding, "Padding row with pc=7 not found");
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
