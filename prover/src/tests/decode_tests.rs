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
/// computed offline once UNDER THE BLOCK PIN (`hash_pin::BLOCK_STARK_HASH`,
/// RPX256 here), over rows in PC ORDER. Mirrors how the recursion guest embeds
/// the commitment as a compile-time constant for its inner program.
///
/// ⚠ A fifth blessed constant, outside the four families
/// `compute_static_commitments` regenerates: it moves with the pin exactly as
/// they do, and `HASH-PINNING.md` lists it with them. If the pin, the AIR or the
/// FFT pipeline changes, this drifts and the test fails — regenerate via the
/// `print_decode_commitment_for_sub` helper below (`--ignored --nocapture`).
///
/// ⚠⚠ **THREE earlier values exist and none of them is this one.** The two
/// inputs to this merge each carried their own, and the merged branch needs a
/// fourth because it combines both changes:
///
/// | value | row order | commit hash |
/// |---|---|---|
/// | `e97168d6…` | `instructions.iter()` | stark's default aliases |
/// | `0a710a9c…` | **pc order** | stark's default aliases |
/// | `e6a99f70…` | `instructions.iter()` | **the block pin, RPX256** |
/// | `858a103e…` (this one) | **pc order** | **the block pin, RPX256** |
///
/// Taking either side's constant would have failed, and for a reason the other
/// side could not see. Regenerated here rather than copied.
///
/// ★ **Run the helper TWICE and diff its output against what you wrote.** The
/// first transcription of this value read `0x55` where the helper had printed
/// `85`; the second run is what caught it. A one-nibble error here is a
/// constant that is wrong in a way no reasoning finds — the test fails, the
/// value looks plausible, and the obvious conclusion is that the code drifted
/// rather than that the pin was mistyped. The second run costs nothing and is
/// the only check that covers the transcription step at all.
///
/// The pc sort is why the row order half can no longer move for a reason nobody
/// chose: it was hashbrown's — a function of the hasher, the map's capacity and
/// the insertion sequence rather than of the ELF. Under the sort it is a
/// function of the ELF alone, which is what a constant pinned in a guest has to
/// be. See [`the_decode_trace_does_not_depend_on_the_map_that_carried_it`].
const SUB_DECODE_COMMITMENT_BLOWUP_2: [u8; 32] = [
    0x85, 0x8a, 0x10, 0x3e, 0xe4, 0xe2, 0xe7, 0x9b, 0x08, 0x0b, 0x67, 0xf1, 0xa6, 0x38, 0x5a, 0xfe,
    0x6d, 0xd9, 0x55, 0x7b, 0x5a, 0x0e, 0xac, 0xde, 0xa0, 0x9b, 0xd5, 0xe3, 0x99, 0xb6, 0xe7, 0x74,
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

// =========================================================================
// Row order is a function of the ELF alone
// =========================================================================

/// A distinct instruction per pc, so a permuted trace cannot match a sorted one
/// by accident — every row differs from every other in PACKED_DECODE and IMM.
fn instr_for(pc: u64) -> Instruction {
    Instruction::ArithImm {
        dst: ((pc / 4) % 30) as u32 + 1,
        src: ((pc / 4) % 7) as u32 + 1,
        imm: (pc % 2048) as i32 - 1024,
        op: ArithOp::Add,
    }
}

/// The same instruction set, reached through two independently-constructed
/// maps: ascending with no reserve, descending with a large one.
///
/// Different insertion order and different capacity means a different hashbrown
/// bucket layout, hence a different `iter()` order — which is exactly the
/// variation a hashbrown version bump, an added `reserve`, or a change of hasher
/// would introduce, expressed as something a test can construct today.
fn two_maps_of(n: u64) -> (U64HashMap<Instruction>, U64HashMap<Instruction>) {
    let pcs: Vec<u64> = (0..n).map(|i| 0x1000 + i * 4).collect();

    let mut ascending: U64HashMap<Instruction> = U64HashMap::default();
    for &pc in &pcs {
        ascending.insert(pc, instr_for(pc));
    }

    let mut descending: U64HashMap<Instruction> = U64HashMap::default();
    descending.reserve(1024);
    for &pc in pcs.iter().rev() {
        descending.insert(pc, instr_for(pc));
    }

    (ascending, descending)
}

/// ★★★ THE PROPERTY: the DECODE trace is a function of the instruction SET,
/// not of the map that carried it.
///
/// Sorting by pc is the mechanism; this is the thing that must be true, and it
/// is stated that way on purpose. A test asserting "the rows are sorted" would
/// pass on any total order and would not say why the order matters — whereas a
/// root pinned as a program constant needs exactly this: two parties holding
/// the same ELF, and nothing else in common, produce the same rows.
///
/// ⚠ This is what fails without the sort. The two maps differ in insertion
/// order and capacity, so `instructions.iter()` walks them differently and the
/// traces come out permuted. Nothing in the system notices today, because
/// prover and verifier both build their map through `instructions_from_elf` and
/// so make the same arbitrary choice — they agree on a construction procedure
/// rather than on the ELF. A hashbrown bump breaks that agreement with no ELF
/// change and no failing test.
#[test]
fn the_decode_trace_does_not_depend_on_the_map_that_carried_it() {
    let (ascending, descending) = two_maps_of(300);

    // The premise: the two maps really are walked differently. If hashbrown ever
    // made iteration order insertion- and capacity-independent, this test would
    // still pass below while testing nothing, so the premise is asserted.
    let order_a: Vec<u64> = ascending.iter().map(|(&pc, _)| pc).collect();
    let order_b: Vec<u64> = descending.iter().map(|(&pc, _)| pc).collect();
    assert_ne!(
        order_a, order_b,
        "the two maps iterate identically, so this test cannot detect a \
         map-dependent trace — rebuild the maps so they differ"
    );

    let (trace_a, pc_to_row_a) = generate_decode_trace(&ascending);
    let (trace_b, pc_to_row_b) = generate_decode_trace(&descending);

    assert_eq!(trace_a.num_rows(), trace_b.num_rows());
    for row in 0..trace_a.num_rows() {
        assert_eq!(
            trace_a.main_table.get_row(row),
            trace_b.main_table.get_row(row),
            "row {row} differs between two maps holding the same instructions"
        );
    }

    // …and the index agrees too, or `update_multiplicities` would write the
    // right counts to the wrong rows.
    for (&pc, &row) in pc_to_row_a.iter() {
        assert_eq!(
            pc_to_row_b.get(&pc),
            Some(&row),
            "pc {pc:#x} maps to a different row in the two maps"
        );
    }
}

/// ★ THE MECHANISM, pinned separately so the reason stays visible.
///
/// The property above holds for any canonical order; this says which one, so a
/// future change that keeps determinism but moves the rows has to come here and
/// re-baseline the pins rather than sliding past.
#[test]
fn decode_rows_are_in_ascending_pc_order() {
    let (ascending, _) = two_maps_of(64);
    let (trace, _) = generate_decode_trace(&ascending);

    // The instruction rows come first, then the CPU padding row, then zeroed
    // padding to the next power of two — so only the first `n` are ordered.
    let pcs: Vec<u64> = (0..64)
        .map(|row| {
            let lo = *trace.main_table.get(row, cols::PC_0).value();
            let hi = *trace.main_table.get(row, cols::PC_1).value();
            lo | (hi << 32)
        })
        .collect();

    let mut sorted = pcs.clone();
    sorted.sort_unstable();
    assert_eq!(
        pcs, sorted,
        "the instruction rows are not in ascending pc order"
    );
    assert_eq!(pcs[0], 0x1000, "the first row is not the lowest pc");
}

/// Prints DECODE's row count and the W1-B group shape for a named ELF.
///
/// The five preprocessed columns stack into one polynomial, so the group's
/// `n_stack` is `ceil_log2(5 * rows)` and its codeword is `n_stack + log_blowup`
/// — which is the residency budget W1-B has to reserve. One power of two either
/// way doubles it, so it is measured from the ELF rather than quoted.
///
/// ```text
/// cargo test -p lambda-vm-prover --release --lib print_decode_shape_for \
///     -- --ignored --nocapture
/// ```
#[test]
#[ignore = "prints the DECODE group shape for the bench ELFs; run with --ignored --nocapture"]
fn print_decode_shape_for_the_bench_elfs() {
    for name in ["ethrex", "sub"] {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .join("executor/program_artifacts");
        let bytes = ["rust", "asm"]
            .iter()
            .find_map(|d| std::fs::read(root.join(d).join(format!("{name}.elf"))).ok());
        let Some(bytes) = bytes else {
            eprintln!("{name}: no ELF, skipped");
            continue;
        };
        let elf = Elf::load(&bytes).expect("ELF load");
        let instructions =
            crate::tables::decode::instructions_from_elf(&elf).expect("instructions");
        // +1 for the CPU padding entry, then the next power of two, min 2.
        let rows = (instructions.len() + 1).next_power_of_two().max(2);
        let cells = 5 * rows;
        let n_stack = cells.next_power_of_two().trailing_zeros();
        eprintln!(
            "{name}: {} instructions -> {rows} rows (2^{}), 5 cols = {cells} cells, \
             n_stack = {n_stack}, codeword at blowup 4 = 2^{} = {} MiB",
            instructions.len(),
            rows.trailing_zeros(),
            n_stack + 2,
            ((1u64 << (n_stack + 2)) * 8) / (1024 * 1024),
        );
    }
}
