//! Tests for the `packed_decode` layout.
//!
//! These validate the single source of truth (`types::packed_decode_shrunk`,
//! `build_alu_flags`/`build_mem_flags`, `ShrunkDecode`) before it is wired into
//! the DECODE/CPU tables in Phase 2+3 of the rework.

use crate::tables::types::{
    ShrunkDecode, alu_op, build_alu_flags, build_mem_flags, packed_decode_shrunk as bits,
};
use executor::vm::instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth};

#[test]
fn test_build_alu_flags_matches_spec_formula() {
    // alu_flags = alu_op + 32·signed + 64·(signed2|invert) + 128·muldiv
    assert_eq!(build_alu_flags(alu_op::AND, false, false, false), 0);
    assert_eq!(build_alu_flags(alu_op::OR, false, false, false), 1);
    assert_eq!(build_alu_flags(alu_op::XOR, false, false, false), 2);
    // SLT (signed less-than)
    assert_eq!(build_alu_flags(alu_op::LT, true, false, false), 4 + 32);
    // SLTU (unsigned)
    assert_eq!(build_alu_flags(alu_op::LT, false, false, false), 4);
    // SRL (logical right shift): invert set
    assert_eq!(build_alu_flags(alu_op::SHIFT, false, true, false), 5 + 64);
    // SRA (arithmetic right shift): signed + invert
    assert_eq!(
        build_alu_flags(alu_op::SHIFT, true, true, false),
        5 + 32 + 64
    );
    // MUL: signed + signed2
    assert_eq!(build_alu_flags(alu_op::MUL, true, true, false), 7 + 32 + 64);
    // MULH: signed + signed2 + muldiv
    assert_eq!(
        build_alu_flags(alu_op::MUL, true, true, true),
        7 + 32 + 64 + 128
    );
    // MULHU: muldiv only
    assert_eq!(build_alu_flags(alu_op::MUL, false, false, true), 7 + 128);
    // REM (signed): DIVREM + signed + muldiv
    assert_eq!(
        build_alu_flags(alu_op::DIVREM, true, false, true),
        8 + 32 + 128
    );
}

#[test]
fn test_build_mem_flags_matches_spec_formula() {
    // mem_flags = jalr_or_op + 2·signed + 4·2B + 8·4B + 16·8B
    // LB (signed byte load): mem_signed only
    assert_eq!(build_mem_flags(false, true, false, false, false), 2);
    // LBU (unsigned byte load): nothing
    assert_eq!(build_mem_flags(false, false, false, false, false), 0);
    // LH (signed halfword): signed + 2B
    assert_eq!(build_mem_flags(false, true, true, false, false), 2 + 4);
    // LW (signed word): signed + 4B
    assert_eq!(build_mem_flags(false, true, false, true, false), 2 + 8);
    // LD (doubleword, always full): 8B
    assert_eq!(build_mem_flags(false, false, false, false, true), 16);
    // SB (store byte): memory_op bit
    assert_eq!(build_mem_flags(true, false, false, false, false), 1);
    // SD (store doubleword): memory_op + 8B
    assert_eq!(build_mem_flags(true, false, false, false, true), 1 + 16);
}

#[test]
fn test_field_placement() {
    // Each field lands at its declared offset and nowhere else.
    assert_eq!(
        ShrunkDecode {
            memory: true,
            ..Default::default()
        }
        .pack(),
        1 << bits::MEMORY
    );
    assert_eq!(
        ShrunkDecode {
            rs1: 0xFF,
            ..Default::default()
        }
        .pack(),
        0xFF << bits::RS1
    );
    assert_eq!(
        ShrunkDecode {
            rd: 0xAB,
            ..Default::default()
        }
        .pack(),
        0xAB << bits::RD
    );
    assert_eq!(
        ShrunkDecode {
            half_instruction_length: 2,
            ..Default::default()
        }
        .pack(),
        2 << bits::HALF_INSTRUCTION_LENGTH
    );
    assert_eq!(
        ShrunkDecode {
            alu_flags: 0xFF,
            ..Default::default()
        }
        .pack(),
        0xFF << bits::ALU_FLAGS
    );
    assert_eq!(
        ShrunkDecode {
            mem_flags: 0xFF,
            ..Default::default()
        }
        .pack(),
        0xFF << bits::MEM_FLAGS
    );
}

#[test]
fn test_fields_are_disjoint_and_fit_in_58_bits() {
    // All fields maxed out.
    let full = ShrunkDecode {
        read_register1: true,
        read_register2: true,
        write_register: true,
        word_instr: true,
        alu: true,
        add: true,
        sub: true,
        memory: true,
        branch: true,
        ecall: true,
        rs1: 0xFF,
        rs2: 0xFF,
        rd: 0xFF,
        half_instruction_length: 0xFF,
        alu_flags: 0xFF,
        mem_flags: 0xFF,
    };
    let packed = full.pack();

    // Fits in 58 bits (mem_flags ends at bit 50+8 = 58).
    assert_eq!(packed >> 58, 0, "packed_decode must fit in 58 bits");

    // Disjointness: with no overlap, summing each field's individual pack
    // equals the combined pack (OR == sum iff masks are disjoint).
    let individual_sum: u64 = [
        ShrunkDecode {
            read_register1: true,
            ..Default::default()
        },
        ShrunkDecode {
            read_register2: true,
            ..Default::default()
        },
        ShrunkDecode {
            write_register: true,
            ..Default::default()
        },
        ShrunkDecode {
            word_instr: true,
            ..Default::default()
        },
        ShrunkDecode {
            alu: true,
            ..Default::default()
        },
        ShrunkDecode {
            add: true,
            ..Default::default()
        },
        ShrunkDecode {
            sub: true,
            ..Default::default()
        },
        ShrunkDecode {
            memory: true,
            ..Default::default()
        },
        ShrunkDecode {
            branch: true,
            ..Default::default()
        },
        ShrunkDecode {
            ecall: true,
            ..Default::default()
        },
        ShrunkDecode {
            rs1: 0xFF,
            ..Default::default()
        },
        ShrunkDecode {
            rs2: 0xFF,
            ..Default::default()
        },
        ShrunkDecode {
            rd: 0xFF,
            ..Default::default()
        },
        ShrunkDecode {
            half_instruction_length: 0xFF,
            ..Default::default()
        },
        ShrunkDecode {
            alu_flags: 0xFF,
            ..Default::default()
        },
        ShrunkDecode {
            mem_flags: 0xFF,
            ..Default::default()
        },
    ]
    .iter()
    .map(ShrunkDecode::pack)
    .sum();

    assert_eq!(
        individual_sum, packed,
        "packed_decode fields must be disjoint"
    );
}

#[test]
fn test_pack_unpack_round_trip() {
    let entries = [
        ShrunkDecode::default(),
        // An ALU register op: ADD rd, rs1, rs2
        ShrunkDecode {
            read_register1: true,
            read_register2: true,
            write_register: true,
            add: true,
            rs1: 0x11,
            rs2: 0x22,
            rd: 0x33,
            half_instruction_length: 2,
            ..Default::default()
        },
        // A signed word ALU op going through the ALU bus (e.g. SRAW)
        ShrunkDecode {
            read_register1: true,
            write_register: true,
            word_instr: true,
            alu: true,
            rs1: 7,
            rd: 9,
            half_instruction_length: 2,
            alu_flags: build_alu_flags(alu_op::SHIFTW, true, true, false),
            ..Default::default()
        },
        // A load: LW rd, imm(rs1)
        ShrunkDecode {
            read_register1: true,
            write_register: true,
            memory: true,
            rs1: 5,
            rd: 6,
            half_instruction_length: 2,
            mem_flags: build_mem_flags(false, true, false, true, false),
            ..Default::default()
        },
        // Fully saturated.
        ShrunkDecode {
            read_register1: true,
            read_register2: true,
            write_register: true,
            word_instr: true,
            alu: true,
            add: true,
            sub: true,
            memory: true,
            branch: true,
            ecall: true,
            rs1: 0xFF,
            rs2: 0xFF,
            rd: 0xFF,
            half_instruction_length: 0xFF,
            alu_flags: 0xFF,
            mem_flags: 0xFF,
        },
    ];

    for entry in entries {
        assert_eq!(ShrunkDecode::unpack(entry.pack()), entry);
    }
}

#[test]
fn test_from_instruction_arith_ops() {
    // ADD rd=3, rs1=1, rs2=2 → ADD fast-path (ALU not set), all reg flags on.
    let d = ShrunkDecode::from_instruction(
        Instruction::Arith {
            dst: 3,
            src1: 1,
            src2: 2,
            op: ArithOp::Add,
        },
        4,
    );
    assert!(d.add && !d.alu && !d.sub);
    assert!(d.read_register1 && d.read_register2 && d.write_register);
    assert_eq!(
        (d.rs1, d.rs2, d.rd, d.half_instruction_length),
        (1, 2, 3, 2)
    );
    assert_eq!(d.alu_flags, 0);

    // AND → ALU path, alu_flags = AND.
    let d = ShrunkDecode::from_instruction(
        Instruction::Arith {
            dst: 5,
            src1: 6,
            src2: 7,
            op: ArithOp::And,
        },
        4,
    );
    assert!(d.alu && !d.add && !d.sub);
    assert_eq!(
        d.alu_flags,
        build_alu_flags(alu_op::AND, false, false, false)
    );

    // SUB → SUB fast-path.
    let d = ShrunkDecode::from_instruction(
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Sub,
        },
        4,
    );
    assert!(d.sub && !d.add && !d.alu);

    // SLT (signed) → ALU, LT signed.
    let d = ShrunkDecode::from_instruction(
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::SetLessThan,
        },
        4,
    );
    assert_eq!(d.alu_flags, build_alu_flags(alu_op::LT, true, false, false));

    // x0 operands/dest → no read/write flags.
    let d = ShrunkDecode::from_instruction(
        Instruction::Arith {
            dst: 0,
            src1: 0,
            src2: 0,
            op: ArithOp::Add,
        },
        4,
    );
    assert!(!d.write_register && !d.read_register1 && !d.read_register2);
}

#[test]
fn test_from_instruction_word_shifts() {
    // SRAW → word_instr, SHIFTW, signed + invert.
    let d = ShrunkDecode::from_instruction(
        Instruction::ArithW {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::ShiftRightArith,
        },
        4,
    );
    assert!(d.word_instr && d.alu);
    assert_eq!(
        d.alu_flags,
        build_alu_flags(alu_op::SHIFTW, true, true, false)
    );

    // SLL (non-word) → SHIFT, no invert.
    let d = ShrunkDecode::from_instruction(
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::ShiftLeftLogical,
        },
        4,
    );
    assert!(!d.word_instr);
    assert_eq!(
        d.alu_flags,
        build_alu_flags(alu_op::SHIFT, false, false, false)
    );
}

#[test]
fn test_from_instruction_mul_div() {
    // MULHU → unsigned, muldiv.
    let d = ShrunkDecode::from_instruction(
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::MulHighUnsigned,
        },
        4,
    );
    assert_eq!(
        d.alu_flags,
        build_alu_flags(alu_op::MUL, false, false, true)
    );

    // REM (signed) → DIVREM, signed, muldiv.
    let d = ShrunkDecode::from_instruction(
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Remainder,
        },
        4,
    );
    assert_eq!(
        d.alu_flags,
        build_alu_flags(alu_op::DIVREM, true, false, true)
    );
}

#[test]
fn test_from_instruction_branches_set_branch_and_alu() {
    // Q3: conditional branches set BRANCH ∧ ALU; mem_flags = 0 (not JALR); no rd write.
    let d = ShrunkDecode::from_instruction(
        Instruction::Branch {
            src1: 1,
            src2: 2,
            cond: Comparison::Equal,
            offset: 16,
        },
        4,
    );
    assert!(d.branch && d.alu && !d.write_register);
    assert_eq!(
        d.alu_flags,
        build_alu_flags(alu_op::EQ, false, false, false)
    );
    assert_eq!(d.mem_flags, 0);

    // BNE → EQ inverted.
    let d = ShrunkDecode::from_instruction(
        Instruction::Branch {
            src1: 1,
            src2: 2,
            cond: Comparison::NotEqual,
            offset: 16,
        },
        4,
    );
    assert_eq!(d.alu_flags, build_alu_flags(alu_op::EQ, false, true, false));

    // BGE → LT signed inverted.
    let d = ShrunkDecode::from_instruction(
        Instruction::Branch {
            src1: 1,
            src2: 2,
            cond: Comparison::GreaterOrEqual,
            offset: 16,
        },
        4,
    );
    assert_eq!(d.alu_flags, build_alu_flags(alu_op::LT, true, true, false));
}

#[test]
fn test_from_instruction_jumps() {
    // JAL → BRANCH + JALR bit, no ALU op, rs1 = x255.
    let d = ShrunkDecode::from_instruction(Instruction::JumpAndLink { dst: 1, offset: 32 }, 4);
    assert!(d.branch && d.write_register && d.read_register1);
    assert!(!d.add && !d.sub && !d.alu);
    assert_eq!(d.rs1, 255);
    assert_eq!(
        d.mem_flags,
        build_mem_flags(true, false, false, false, false)
    );

    // JALR → BRANCH + JALR bit, no ALU op, rs1 = base.
    let d = ShrunkDecode::from_instruction(
        Instruction::JumpAndLinkRegister {
            base: 9,
            dst: 1,
            offset: 0,
        },
        4,
    );
    assert!(d.branch);
    assert!(!d.add && !d.sub && !d.alu);
    assert_eq!(d.rs1, 9);
    assert_eq!(d.mem_flags & 1, 1);
}

#[test]
fn test_from_instruction_load_store() {
    // LW (signed) → ADD + MEMORY, mem_signed + mem_4B.
    let d = ShrunkDecode::from_instruction(
        Instruction::Load {
            dst: 1,
            offset: 0,
            base: 2,
            width: LoadStoreWidth::Word,
        },
        4,
    );
    assert!(d.add && d.memory && d.write_register);
    assert_eq!(
        d.mem_flags,
        build_mem_flags(false, true, false, true, false)
    );

    // LBU → no signed, no width bits.
    let d = ShrunkDecode::from_instruction(
        Instruction::Load {
            dst: 1,
            offset: 0,
            base: 2,
            width: LoadStoreWidth::ByteUnsigned,
        },
        4,
    );
    assert_eq!(d.mem_flags, 0);

    // SD → ADD + MEMORY, memory_op + mem_8B, no rd write.
    let d = ShrunkDecode::from_instruction(
        Instruction::Store {
            src: 3,
            offset: 0,
            base: 2,
            width: LoadStoreWidth::DoubleWord,
        },
        4,
    );
    assert!(d.add && d.memory && !d.write_register);
    assert_eq!(
        d.mem_flags,
        build_mem_flags(true, false, false, false, true)
    );
}

#[test]
fn test_from_instruction_system() {
    // ECALL → ECALL, rs1 = x17 (a7).
    let d = ShrunkDecode::from_instruction(Instruction::EcallEbreak, 4);
    assert!(d.ecall && d.read_register1);
    assert_eq!(d.rs1, 17);

    // LUI → ADD, rs1 = x0.
    let d = ShrunkDecode::from_instruction(
        Instruction::LoadUpperImm {
            dst: 5,
            imm: 0x1000,
        },
        4,
    );
    assert!(d.add && d.write_register);
    assert_eq!(d.rs1, 0);

    // AUIPC → ADD, rs1 = x255.
    let d = ShrunkDecode::from_instruction(
        Instruction::AddUpperImmToPc {
            dst: 5,
            imm: 0x1000,
        },
        4,
    );
    assert!(d.add && d.read_register1);
    assert_eq!(d.rs1, 255);

    // FENCE → ADD no-op.
    let d = ShrunkDecode::from_instruction(Instruction::Fence, 4);
    assert!(d.add);

    // Compressed instruction length (2 bytes) propagates as half = 1.
    let d = ShrunkDecode::from_instruction(
        Instruction::Arith {
            dst: 1,
            src1: 2,
            src2: 3,
            op: ArithOp::Add,
        },
        2,
    );
    assert_eq!(d.half_instruction_length, 1);
}
