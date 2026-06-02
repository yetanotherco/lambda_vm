//! Tests for the DECODE table.
//!
//! `decode_layout_tests` covers the `ShrunkDecode` pack/unpack/from_instruction
//! bit layout in isolation; here we test the `DecodeEntry` wrapper (pc/imm
//! extraction, padding) and the DECODE *table* generation (`generate_decode_trace`):
//! the per-instruction rows, the `pc = 1` padding entry, and the `pc_to_row` map.

use crate::tables::cpu::CPU_PADDING_PC;
use crate::tables::decode::{cols, generate_decode_trace};
use crate::tables::types::DecodeEntry;

use executor::vm::instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth};
use executor::vm::memory::U64HashMap;

// =========================================================================
// DecodeEntry
// =========================================================================

#[test]
fn test_decode_entry_default_and_padding() {
    let d = DecodeEntry::new();
    assert_eq!(d.pc, 0);
    assert_eq!(d.imm, 0);
    assert_eq!(d.packed_decode(), 0);

    let pad = DecodeEntry::padding_entry();
    assert_eq!(pad.pc, CPU_PADDING_PC, "padding sits at the odd address 1");
    assert_eq!(pad.imm, 0);
    assert_eq!(pad.packed_decode(), 0, "padding has all flags zero");
}

#[test]
fn test_decode_entry_packed_decode_matches_fields() {
    let d = DecodeEntry::from_instruction(
        0x2000,
        Instruction::Arith {
            dst: 3,
            src1: 1,
            src2: 2,
            op: ArithOp::Add,
        },
        4,
    );
    assert_eq!(d.packed_decode(), d.fields.pack());
    assert!(d.fields.add, "ADD is a fast-path flag");
    assert_eq!(d.fields.half_instruction_length, 2);
}

#[test]
fn test_decode_entry_imm_extraction() {
    let add = DecodeEntry::from_instruction(
        0,
        Instruction::Arith {
            dst: 3,
            src1: 1,
            src2: 2,
            op: ArithOp::Add,
        },
        4,
    );
    assert_eq!(add.imm, 0, "reg-reg has no immediate");

    let addi = DecodeEntry::from_instruction(
        0,
        Instruction::ArithImm {
            dst: 3,
            src: 1,
            imm: 5,
            op: ArithOp::Add,
        },
        4,
    );
    assert_eq!(addi.imm, 5);

    let beq = DecodeEntry::from_instruction(
        0,
        Instruction::Branch {
            src1: 1,
            src2: 2,
            cond: Comparison::Equal,
            offset: 8,
        },
        4,
    );
    assert_eq!(beq.imm, 8, "branch offset");

    let lw = DecodeEntry::from_instruction(
        0,
        Instruction::Load {
            dst: 3,
            offset: 16,
            base: 1,
            width: LoadStoreWidth::Word,
        },
        4,
    );
    assert_eq!(lw.imm, 16, "load offset");
}

#[test]
fn test_decode_entry_negative_imm_sign_extended() {
    let addi = DecodeEntry::from_instruction(
        0,
        Instruction::ArithImm {
            dst: 3,
            src: 1,
            imm: -1,
            op: ArithOp::Add,
        },
        4,
    );
    assert_eq!(
        addi.imm,
        u64::MAX,
        "-1 sign-extends to the full 64-bit word"
    );
}

// =========================================================================
// generate_decode_trace
// =========================================================================

const TEST_PC: u64 = 0x1000;

fn test_instr() -> Instruction {
    Instruction::ArithImm {
        dst: 3,
        src: 1,
        imm: 7,
        op: ArithOp::Add,
    }
}

#[test]
fn test_decode_table_instruction_row() {
    let entry = DecodeEntry::from_instruction(TEST_PC, test_instr(), 4);
    let mut instrs: U64HashMap<Instruction> = U64HashMap::default();
    instrs.insert(TEST_PC, test_instr());
    let (trace, pc_to_row) = generate_decode_trace(&instrs);

    let row = trace.main_table.get_row(pc_to_row[&TEST_PC]);
    assert_eq!(row[cols::PC_0], (TEST_PC & 0xFFFF_FFFF).into());
    assert_eq!(row[cols::PACKED_DECODE], entry.packed_decode().into());
    assert_eq!(row[cols::IMM_0], (entry.imm & 0xFFFF_FFFF).into());
}

#[test]
fn test_decode_table_padding_row() {
    let mut instrs: U64HashMap<Instruction> = U64HashMap::default();
    instrs.insert(TEST_PC, test_instr());
    let (trace, pc_to_row) = generate_decode_trace(&instrs);

    let row = trace.main_table.get_row(pc_to_row[&CPU_PADDING_PC]);
    assert_eq!(row[cols::PC_0], CPU_PADDING_PC.into());
    assert_eq!(
        row[cols::PACKED_DECODE],
        0u64.into(),
        "padding entry has packed_decode = 0"
    );
    assert_eq!(row[cols::IMM_0], 0u64.into());
}

#[test]
fn test_decode_table_is_power_of_two() {
    let mut instrs: U64HashMap<Instruction> = U64HashMap::default();
    instrs.insert(TEST_PC, test_instr());
    let (trace, _) = generate_decode_trace(&instrs);
    assert!(
        trace.main_table.height.is_power_of_two(),
        "decode table is padded to a power of two"
    );
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);
}
