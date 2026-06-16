//! Tests for the DECODE table.
//!
//! `decode_layout_tests` covers the `ShrunkDecode` pack/unpack/from_instruction
//! bit layout in isolation; here we test the `DecodeEntry` wrapper (pc/imm
//! extraction, padding) and the DECODE *table* generation (`generate_decode_trace`):
//! the per-instruction rows, the `pc = 1` padding entry, and the `pc_to_row` map.

use crate::tables::cpu::CPU_PADDING_PC;
use crate::tables::decode::{cols, commitment_from_elf, generate_decode_trace};
use crate::tables::types::DecodeEntry;
use crate::test_utils::asm_elf_bytes;
use crate::{prove, verify_with_options};

use executor::elf::Elf;
use executor::vm::instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth};
use executor::vm::memory::U64HashMap;
use stark::proof::options::GoldilocksCubicProofOptions;

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

// =========================================================================
// verify_with_options: optional decode_commitment parameter (#640)
// =========================================================================

#[test]
fn decode_commitment_some_matches_default_path() {
    let elf_bytes = asm_elf_bytes("sub");
    let vm_proof = prove(&elf_bytes).expect("prove failed");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let options = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    let decode_c = commitment_from_elf(&elf, &options).expect("decode commitment");

    let default_ok = verify_with_options(&vm_proof, &elf_bytes, &options, None, None)
        .expect("verify with None should not error");
    let explicit_ok = verify_with_options(&vm_proof, &elf_bytes, &options, Some(decode_c), None)
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

    let result = verify_with_options(&vm_proof, &elf_bytes, &options, Some(wrong), None)
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
    let result = verify_with_options(&vm_proof, &elf_bytes, &options, Some([0u8; 32]), None)
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
    0x60, 0x66, 0x0b, 0x18, 0x0d, 0x41, 0x08, 0xb3, 0x3a, 0x03, 0x99, 0x03, 0x8c, 0x9d, 0x12, 0x57,
    0x68, 0x8d, 0xed, 0x13, 0x60, 0xeb, 0x1d, 0x2b, 0xa8, 0xea, 0x1c, 0x76, 0xc9, 0xdd, 0x25, 0xaf,
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
        None,
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
