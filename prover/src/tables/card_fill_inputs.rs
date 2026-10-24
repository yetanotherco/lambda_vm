//! The inputs of the card fill's tests: the records the card's fill kernels
//! read (`math_cuda::fill`), encoded from the prover's own; seeded synthetic
//! sets of each family's; the host fill's words in the card's layout; and the
//! vectors of the kernels' host known-answer harness
//! (`crypto/math-cuda/tests/host_kat/fill_kat_vectors.h`).
//!
//! The synthetic sets (splitmix64, seeds [`SEEDS`]):
//! - **MEMW_R:** 2^21 register rows. Address < 256, timestamp in [8, 2^34),
//!   the old timestamp's low word 1..=2^16 below the timestamp's with no 32-bit
//!   wrap, random values and read flags.
//! - **CPU:** 2^21 logs over a program's decode table from cycle 7·2^21. Each
//!   log draws a decode row uniformly (the decode gathers' worst case) and takes
//!   its pc. The next pc is pc + length, or a random aligned target on a branch
//!   row. Register values are random, and one log in four has src2 = src1, so
//!   conditional branches go both ways.
//! - **KECCAK_RND:** 2,732 keccak operations on random states, the output their
//!   permutation, timestamps increasing across 2^32. The table is 2^16 rows cut
//!   16 rows into the first op, so the first and the last op are both cut.

use executor::elf::Elf;
use executor::vm::instruction::decoding::Instruction;
use executor::vm::instruction::execution::keccak_f1600;
use executor::vm::logs::Log;
use executor::vm::memory::U64HashMap;
use math_cuda::fill::{AlignedRowC, DecodeEntryC, DecodeRunC, KeccakOpC, LoadOpC, LogC};

use super::cpu::{CpuOperation, generate_cpu_trace_as};
use super::decode::{DecodeTable, instructions_from_elf};
use super::gpack::{Trace, TraceForm};
use super::keccak_rnd::{self, KeccakRoundOperation, generate_keccak_rnd_rows_as};
use super::load::{self, LoadOperation};
use super::memw_aligned::{self, AlignedRow};
use super::memw_register::{RegRow, generate_memw_register_trace_from_rows_as};

/// The synthetic sets' seeds.
pub(crate) const SEEDS: [u64; 3] = [1, 2, 3];
/// Rows of a streamed CPU or MEMW_R chunk.
pub(crate) const CHUNK_ROWS: usize = 1 << 21;
/// The synthetic CPU chunk's first cycle.
pub(crate) const CPU_FIRST_CYCLE: u64 = 7 << 21;
/// Rows of a KECCAK_RND table.
pub(crate) const KECCAK_RND_ROWS: usize = 1 << 16;
/// The synthetic KECCAK_RND table's ops and its cut into the first.
pub(crate) const KECCAK_RND_OPS: usize = 2732;
pub(crate) const KECCAK_RND_SKIP: usize = 16;
// The table's rows cut into the first and the last op, and the ops reach its
// end.
const _: () = {
    assert!(!KECCAK_RND_SKIP.is_multiple_of(24));
    assert!(!(KECCAK_RND_SKIP + KECCAK_RND_ROWS).is_multiple_of(24));
    assert!(KECCAK_RND_OPS * 24 >= KECCAK_RND_SKIP + KECCAK_RND_ROWS);
};

/// splitmix64: the inputs are a function of the seed alone.
pub(crate) fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// =========================================================================
// The records
// =========================================================================

/// A log as the card's CPU fill reads it, field for field.
pub(crate) fn log_record(log: &Log) -> LogC {
    LogC {
        current_pc: log.current_pc,
        next_pc: log.next_pc,
        src1_val: log.src1_val,
        src2_val: log.src2_val,
        dst_val: log.dst_val,
    }
}

/// `decode`'s entries and runs as the card's CPU fill reads them.
pub(crate) fn decode_records(decode: &DecodeTable) -> (Vec<DecodeEntryC>, Vec<DecodeRunC>) {
    let entries = decode
        .entries()
        .iter()
        .map(|e| {
            let f = &e.fields;
            DecodeEntryC {
                pc: e.pc,
                imm: e.imm,
                read_register1: u8::from(f.read_register1),
                read_register2: u8::from(f.read_register2),
                write_register: u8::from(f.write_register),
                word_instr: u8::from(f.word_instr),
                alu: u8::from(f.alu),
                add: u8::from(f.add),
                sub: u8::from(f.sub),
                memory: u8::from(f.memory),
                branch: u8::from(f.branch),
                ecall: u8::from(f.ecall),
                rs1: f.rs1,
                rs2: f.rs2,
                rd: f.rd,
                half_instruction_length: f.half_instruction_length,
                alu_flags: f.alu_flags,
                mem_flags: f.mem_flags,
            }
        })
        .collect();
    let runs = decode
        .runs()
        .iter()
        .map(|&(first_pc, row, len)| DecodeRunC { first_pc, row, len })
        .collect();
    (entries, runs)
}

/// A keccak operation as the card's KECCAK_RND fill reads it.
pub(crate) fn keccak_record(op: &KeccakRoundOperation) -> KeccakOpC {
    KeccakOpC {
        timestamp: op.timestamp,
        input: op.input,
        output: op.output,
    }
}

// =========================================================================
// The synthetic sets
// =========================================================================

/// `n` register rows from `seed` (the module docs).
pub(crate) fn memw_r_rows(seed: u64, n: usize) -> Vec<RegRow> {
    let mut rng = seed;
    (0..n)
        .map(|_| {
            // The low word above 2^16, so the old one is below it with no wrap.
            let timestamp = loop {
                let t = 8 + splitmix(&mut rng) % ((1 << 34) - 8);
                if t & 0xFFFF_FFFF > 1 << 16 {
                    break t;
                }
            };
            let delta = 1 + splitmix(&mut rng) % (1 << 16);
            let old_timestamp = timestamp - delta;
            let address = splitmix(&mut rng) % 256;
            let [val0, val1, old0, old1] = std::array::from_fn(|_| splitmix(&mut rng) as u32);
            let is_read = splitmix(&mut rng) & 1 == 1;
            RegRow::new(
                2 * address,
                timestamp,
                val0,
                val1,
                old0,
                old1,
                old_timestamp,
                is_read,
            )
        })
        .collect()
}

/// A log of the instruction at `entry` of `decode`, from `rng`; `equal` sets
/// src2 = src1.
fn cpu_log(rng: &mut u64, decode: &DecodeTable, entry: usize, equal: bool) -> Log {
    let e = &decode.entries()[entry];
    let length = 2 * u64::from(e.fields.half_instruction_length);
    let src1_val = splitmix(rng);
    let src2_val = if equal { src1_val } else { splitmix(rng) };
    let next_pc = if e.fields.branch {
        splitmix(rng) & !3
    } else {
        e.pc.wrapping_add(length)
    };
    Log {
        current_pc: e.pc,
        next_pc,
        src1_val,
        src2_val,
        dst_val: splitmix(rng),
    }
}

/// `n` logs over `decode` from `seed` (the module docs).
pub(crate) fn cpu_logs(seed: u64, n: usize, decode: &DecodeTable) -> Vec<Log> {
    let mut rng = seed;
    let entries = decode.entries().len() as u64;
    (0..n)
        .map(|i| {
            let entry = (splitmix(&mut rng) % entries) as usize;
            cpu_log(&mut rng, decode, entry, i % 4 == 0)
        })
        .collect()
}

/// The CPU ops of `logs` from cycle `first_cycle`, as the walk makes them
/// (`collect_cpu_ops_into`: timestamp `4·cycle + 4`, the op read with its
/// DECODE row).
pub(crate) fn cpu_ops(logs: &[Log], first_cycle: u64, decode: &DecodeTable) -> Vec<CpuOperation> {
    logs.iter()
        .enumerate()
        .map(|(i, log)| {
            let row = decode
                .row(log.current_pc)
                .expect("every log's pc is the program's");
            CpuOperation::from_log(
                log,
                (first_cycle + i as u64) * 4 + 4,
                decode.entry(row),
                row,
            )
        })
        .collect()
}

/// `n` keccak operations from `seed` (the module docs).
pub(crate) fn keccak_ops(seed: u64, n: usize) -> Vec<KeccakRoundOperation> {
    let mut rng = seed;
    // 1,024 an op below 2^32, half the increments' mean (2,050): the run
    // crosses 2^32 near its middle.
    let mut timestamp = (1u64 << 32) - 1024 * n as u64;
    (0..n)
        .map(|_| {
            timestamp += 4 * (1 + splitmix(&mut rng) % 1024);
            let input: [u64; 25] = std::array::from_fn(|_| splitmix(&mut rng));
            let mut output = input;
            keccak_f1600(&mut output);
            KeccakRoundOperation {
                timestamp,
                input,
                output,
            }
        })
        .collect()
}

/// The RV64 major opcodes a decode table with no guest draws from: LOAD,
/// OP-IMM, AUIPC, OP-IMM-32, STORE, OP, LUI, OP-32, BRANCH, JALR, JAL.
const OPCODES: [u32; 11] = [
    0x03, 0x13, 0x17, 0x1b, 0x23, 0x33, 0x37, 0x3b, 0x63, 0x67, 0x6f,
];
/// The BRANCH opcode and its six conditions (BEQ, BNE, BLT, BGE, BLTU, BGEU).
const BRANCH: u32 = 0x63;
const BRANCH_FUNCT3: [u32; 6] = [0, 1, 4, 5, 6, 7];
/// ECALL's word.
const ECALL: u32 = 0x0000_0073;
/// Where the runs of a decode table with no guest start: gaps between them,
/// so the CPU fill's search is across runs.
const RUN_BASES: [u64; 3] = [0x1_0000, 0x40_0000, 0x8000_0000];

/// A seeded random word of `opcode` (and `funct3`, when given) that parses.
fn random_instruction(rng: &mut u64, opcode: u32, funct3: Option<u32>) -> Instruction {
    loop {
        let mut word = (splitmix(rng) as u32 & !0x7f) | opcode;
        if let Some(f) = funct3 {
            word = (word & !(0x7 << 12)) | (f << 12);
        }
        if let Ok(instruction) = Instruction::parse(word) {
            return instruction;
        }
    }
}

/// A decode table with no guest: three runs of `per_run` instructions 4
/// apart, each a seeded random word of a random major opcode that parses, one
/// in 64 an ECALL.
pub(crate) fn parsed_decode(seed: u64, per_run: usize) -> DecodeTable {
    let mut rng = seed;
    let mut instructions = U64HashMap::default();
    for base in RUN_BASES {
        for i in 0..per_run {
            let instruction = if i % 64 == 63 {
                Instruction::parse(ECALL).expect("ECALL parses")
            } else {
                let opcode = OPCODES[(splitmix(&mut rng) % OPCODES.len() as u64) as usize];
                random_instruction(&mut rng, opcode, None)
            };
            instructions.insert(base + 4 * i as u64, instruction);
        }
    }
    DecodeTable::from_instructions(&instructions)
}

/// The decode table the CPU sets are drawn over: the guest's at
/// `BLOCK_WHIR_ELF` (the median block's guest, on the box), or else
/// [`parsed_decode`]'s.
pub(crate) fn program_decode() -> DecodeTable {
    match std::env::var("BLOCK_WHIR_ELF") {
        Ok(path) => {
            let bytes =
                std::fs::read(&path).unwrap_or_else(|e| panic!("BLOCK_WHIR_ELF {path}: {e}"));
            let elf = Elf::load(&bytes).expect("the guest ELF loads");
            DecodeTable::from_instructions(
                &instructions_from_elf(&elf).expect("the guest's instructions parse"),
            )
        }
        Err(_) => parsed_decode(SEEDS[0], 4096),
    }
}

// =========================================================================
// The host fills, in the card's layout
// =========================================================================

/// An aligned row's record: the card's struct, field for field.
pub(crate) fn aligned_record(row: &AlignedRow) -> AlignedRowC {
    let r = &memw_aligned::card_rows(std::slice::from_ref(row))[0];
    AlignedRowC {
        base_address: r.base_address,
        timestamp: r.timestamp,
        old_timestamp: r.old_timestamp,
        value: r.value,
        old: r.old,
        width: r.width,
        is_read: r.is_read,
        is_register: r.is_register,
    }
}

/// A load op's record: the card's struct, field for field.
pub(crate) fn load_record(op: &LoadOperation) -> LoadOpC {
    let r = &load::card_ops(std::slice::from_ref(op))[0];
    LoadOpC {
        base_address: r.base_address,
        timestamp: r.timestamp,
        width: r.width,
        is_signed: r.signed,
        res: r.res,
    }
}

/// `n` aligned rows from `seed`: memory and register accesses, reads and
/// writes, of widths 1, 2, 4 and 8 in turn, the values random (a register
/// access's two 32-bit halves, a memory access's eight bytes).
pub(crate) fn memw_a_rows(seed: u64, n: usize) -> Vec<AlignedRow> {
    let mut rng = seed;
    (0..n)
        .map(|i| {
            let width = 1u8 << (i % 4);
            let is_register = i % 3 == 1;
            let timestamp = 8 + splitmix(&mut rng) % (1 << 40);
            let old_timestamp = splitmix(&mut rng) % timestamp;
            AlignedRow::from_parts(
                splitmix(&mut rng) & !7,
                timestamp,
                old_timestamp,
                splitmix(&mut rng),
                splitmix(&mut rng),
                width,
                splitmix(&mut rng) & 1 == 1,
                is_register,
            )
        })
        .collect()
}

/// `n` load ops from `seed`: widths 1, 2, 4 and 8 in turn, signed or not, the
/// result's bytes random (so each width's sign bit is set in some op and clear
/// in another).
pub(crate) fn load_ops(seed: u64, n: usize) -> Vec<LoadOperation> {
    let mut rng = seed;
    (0..n)
        .map(|i| {
            LoadOperation::new(
                splitmix(&mut rng),
                8 + splitmix(&mut rng) % (1 << 40),
                1 << (i % 4),
                (i / 4) % 2 == 1,
                splitmix(&mut rng).to_le_bytes(),
            )
        })
        .collect()
}

/// MEMW_A's host fill of `rows`, in `form`.
pub(crate) fn memw_a_host(rows: &[AlignedRow], form: TraceForm) -> Trace {
    memw_aligned::generate_memw_aligned_trace_segments_as(&[rows], form)
}

/// LOAD's host fill of `ops`, in `form`.
pub(crate) fn load_host(ops: &[LoadOperation], form: TraceForm) -> Trace {
    load::generate_load_trace_segments_as(&[ops], form)
}

/// MEMW_R's host fill of `rows`, in `form`.
pub(crate) fn memw_r_host(rows: &[RegRow], form: TraceForm) -> Trace {
    generate_memw_register_trace_from_rows_as(rows, form)
}

/// CPU's host fill of `ops`, in `form`.
pub(crate) fn cpu_host(ops: &[CpuOperation], decode: &DecodeTable, form: TraceForm) -> Trace {
    generate_cpu_trace_as(ops, decode, form)
}

/// KECCAK_RND's host fill of rows `[skip, skip + rows)` of `ops`' table, in
/// `form`.
pub(crate) fn keccak_rnd_host(
    ops: &[KeccakRoundOperation],
    skip: usize,
    rows: usize,
    form: TraceForm,
) -> Trace {
    generate_keccak_rnd_rows_as(ops, skip, rows, form)
}

/// The words of `trace`'s main table column-major (`col · rows + row`), the
/// card's layout, whether it is held packed or at 8 bytes a cell.
pub(crate) fn col_major(trace: &Trace) -> Vec<u64> {
    let (rows, cols) = (trace.num_rows(), trace.main_table.width);
    if let Some(packed) = trace.narrow_main() {
        return (0..cols).flat_map(|c| packed.column(c)).collect();
    }
    // A block of rows at a time, so its rows stay in cache across the columns.
    const BLOCK_ROWS: usize = 64;
    let mut words = vec![0u64; rows * cols];
    for first in (0..rows).step_by(BLOCK_ROWS) {
        let end = (first + BLOCK_ROWS).min(rows);
        let block: Vec<&[super::types::FE]> =
            (first..end).map(|r| trace.main_table.get_row(r)).collect();
        for c in 0..cols {
            for (k, row) in block.iter().enumerate() {
                words[c * rows + first + k] = *row[c].value();
            }
        }
    }
    words
}

/// Where two column-major tables of `rows` rows differ: how many words, and
/// the first (column, row, wanted, got).
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Mismatches {
    pub(crate) count: usize,
    pub(crate) first: Option<(usize, usize, u64, u64)>,
}

/// [`Mismatches`] of `got` against `want`; a length difference counts each
/// word past the shorter.
pub(crate) fn mismatches(want: &[u64], got: &[u64], rows: usize) -> Mismatches {
    let mut out = Mismatches {
        count: want.len().abs_diff(got.len()),
        first: None,
    };
    for (i, (&w, &g)) in want.iter().zip(got).enumerate() {
        if w != g {
            out.count += 1;
            out.first
                .get_or_insert((i / rows.max(1), i % rows.max(1), w, g));
        }
    }
    out
}

// =========================================================================
// The host known-answer harness's vectors
// =========================================================================

/// The harness's vectors, relative to the workspace root.
pub(crate) const HOST_KAT_VECTORS: &str = "crypto/math-cuda/tests/host_kat/fill_kat_vectors.h";

/// The known-answer MEMW_R table: rows, then zero rows.
const KAT_MEMW_R_ROWS: usize = 13;
const KAT_MEMW_R_HEIGHT: usize = 16;
/// The known-answer CPU table: three runs of eight instructions, every op
/// class among them, and logs that take each in turn, then padding rows. The
/// ops carry the run's timestamps from a cycle well into the run; the table's
/// own are its rows' `4·(row + 1)` whatever the ops carry.
const KAT_CPU_LOGS: usize = 29;
const KAT_CPU_HEIGHT: usize = 32;
const KAT_CPU_FIRST_CYCLE: u64 = (1 << 30) - 20;
/// The known-answer MEMW_A and LOAD tables: rows, then zero rows.
const KAT_MEMW_A_ROWS: usize = 13;
const KAT_MEMW_A_HEIGHT: usize = 16;
const KAT_LOAD_OPS: usize = 19;
const KAT_LOAD_HEIGHT: usize = 32;
/// The known-answer KECCAK_RND tables, `(first op, ops, skip, rows)` of
/// [`KAT_KECCAK_OPS`] ops: cut into the first op, every round of the second,
/// zero rows after; and cut at both ends.
const KAT_KECCAK_OPS: usize = 4;
const KAT_KECCAK_CASES: [(usize, usize, usize, usize); 2] = [(0, 2, 21, 32), (2, 2, 10, 20)];

/// The known-answer decode table (the CPU table's docs).
fn kat_decode() -> DecodeTable {
    let mut rng = 0xCA4D_F111;
    let mut program: Vec<Instruction> = Vec::new();
    for opcode in OPCODES.into_iter().filter(|&o| o != BRANCH) {
        program.push(random_instruction(&mut rng, opcode, None));
    }
    for funct3 in BRANCH_FUNCT3 {
        program.push(random_instruction(&mut rng, BRANCH, Some(funct3)));
    }
    program.push(Instruction::parse(ECALL).expect("ECALL parses"));
    while program.len() < 24 {
        let opcode = OPCODES[(splitmix(&mut rng) % OPCODES.len() as u64) as usize];
        program.push(random_instruction(&mut rng, opcode, None));
    }
    let mut instructions = U64HashMap::default();
    for (i, instruction) in program.into_iter().enumerate() {
        instructions.insert(RUN_BASES[i / 8] + 4 * (i % 8) as u64, instruction);
    }
    DecodeTable::from_instructions(&instructions)
}

/// FNV-1a over `words`' little-endian bytes: the harness's digest of a row's
/// run of KECCAK_RND columns.
pub(crate) fn fnv1a(words: impl IntoIterator<Item = u64>) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for w in words {
        for b in w.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

/// The KECCAK_RND column groups' ends, in column order (`keccak_rnd::cols`).
fn keccak_group_ends() -> [usize; 15] {
    use keccak_rnd::cols;
    [
        cols::ROUND,
        cols::START,
        cols::CXZ,
        cols::CXZ_LEFT,
        cols::CXZ_RIGHT,
        cols::DXZ,
        cols::THETA,
        cols::ROT_LEFT,
        cols::ROT_RIGHT,
        cols::CHI_ANDS,
        cols::CHI,
        cols::RC,
        cols::IOTA,
        cols::MU,
        cols::NUM_COLUMNS,
    ]
}

/// `values` as a C initializer list, `per_line` to a line.
fn c_list(values: &[String], per_line: usize) -> String {
    values
        .chunks(per_line)
        .map(|line| format!("    {},\n", line.join(", ")))
        .collect()
}

fn u64s(values: impl IntoIterator<Item = u64>) -> Vec<String> {
    values.into_iter().map(|v| format!("{v}ull")).collect()
}

/// The harness's vectors (`HOST_KAT_VECTORS`): the records and the host
/// fill's words of the known-answer tables, as C++.
pub(crate) fn host_kat_vectors() -> String {
    let mut s = String::from(
        "// Known-answer vectors for the table fills (`kernels/fill.cu`), from the prover's host\n\
         // fill. GENERATED by `tables::card_fill_inputs` (prover); do not edit. Regenerate with\n\
         //   CARD_FILL_KAT_WRITE=1 cargo test -p lambda-vm-prover --lib \\\n\
         //     tables::card_fill_inputs::tests::the_host_kat_vectors_are_the_host_fills\n\
         // Include after `fill.cu` (it declares the record structs). The CPU table's decode is\n\
         // seeded random instruction words (no guest ELF): these are host-generated vectors,\n\
         // not a guest's rows.\n\
         #pragma once\n#include <cstdint>\n\n",
    );

    // MEMW_R.
    let rows = memw_r_rows(0x4D45_4D57, KAT_MEMW_R_ROWS);
    let words = col_major(&memw_r_host(&rows, TraceForm::Wide));
    assert_eq!(
        words.len(),
        KAT_MEMW_R_HEIGHT * math_cuda::fill::MEMW_R_COLUMNS
    );
    s += &format!(
        "// MEMW_R: {KAT_MEMW_R_ROWS} rows, then zero rows to {KAT_MEMW_R_HEIGHT}.\n\
         inline constexpr uint64_t KAT_MEMW_R_N = {KAT_MEMW_R_ROWS};\n\
         inline constexpr uint64_t KAT_MEMW_R_HEIGHT = {KAT_MEMW_R_HEIGHT};\n\
         inline const RegRowC KAT_MEMW_R_ROWS[KAT_MEMW_R_N] = {{\n"
    );
    for r in &rows {
        let c = r.card_record();
        s += &format!(
            "    {{{}ull, {}u, {}u, {}u, {}u, {}u, {}, {}, {}}},\n",
            c.timestamp, c.val0, c.val1, c.old0, c.old1, c.old_ts_lo, c.address, c.is_read, c.pad
        );
    }
    s += "};\n// Column-major: column c's rows from c * KAT_MEMW_R_HEIGHT.\n";
    s += "inline const uint64_t KAT_MEMW_R_WORDS[] = {\n";
    s += &c_list(&u64s(words), 6);
    s += "};\n\n";

    // MEMW_A.
    let rows = memw_a_rows(0x4D45_4D41, KAT_MEMW_A_ROWS);
    let words = col_major(&memw_a_host(&rows, TraceForm::Wide));
    assert_eq!(
        words.len(),
        KAT_MEMW_A_HEIGHT * math_cuda::fill::MEMW_A_COLUMNS
    );
    s += &format!(
        "// MEMW_A: {KAT_MEMW_A_ROWS} rows, then zero rows to {KAT_MEMW_A_HEIGHT}.\n\
         inline constexpr uint64_t KAT_MEMW_A_N = {KAT_MEMW_A_ROWS};\n\
         inline constexpr uint64_t KAT_MEMW_A_HEIGHT = {KAT_MEMW_A_HEIGHT};\n\
         inline const AlignedRowC KAT_MEMW_A_ROWS[KAT_MEMW_A_N] = {{\n"
    );
    for c in rows.iter().map(aligned_record) {
        s += &format!(
            "    {{{}ull, {}ull, {}ull, {}ull, {}ull, {}, {}, {}}},\n",
            c.base_address,
            c.timestamp,
            c.old_timestamp,
            c.value,
            c.old,
            c.width,
            c.is_read,
            c.is_register
        );
    }
    s += "};\n// Column-major: column c's rows from c * KAT_MEMW_A_HEIGHT.\n";
    s += "inline const uint64_t KAT_MEMW_A_WORDS[] = {\n";
    s += &c_list(&u64s(words), 6);
    s += "};\n\n";

    // LOAD.
    let ops = load_ops(0x4C4F_4144, KAT_LOAD_OPS);
    let words = col_major(&load_host(&ops, TraceForm::Wide));
    assert_eq!(words.len(), KAT_LOAD_HEIGHT * math_cuda::fill::LOAD_COLUMNS);
    s += &format!(
        "// LOAD: {KAT_LOAD_OPS} ops, then zero rows to {KAT_LOAD_HEIGHT}.\n\
         inline constexpr uint64_t KAT_LOAD_N = {KAT_LOAD_OPS};\n\
         inline constexpr uint64_t KAT_LOAD_HEIGHT = {KAT_LOAD_HEIGHT};\n\
         inline const LoadOpC KAT_LOAD_OPS[KAT_LOAD_N] = {{\n"
    );
    for c in ops.iter().map(load_record) {
        s += &format!(
            "    {{{}ull, {}ull, {}, {}, {{{}}}}},\n",
            c.base_address,
            c.timestamp,
            c.width,
            c.is_signed,
            c.res.map(|b| b.to_string()).join(", ")
        );
    }
    s += "};\n// Column-major: column c's rows from c * KAT_LOAD_HEIGHT.\n";
    s += "inline const uint64_t KAT_LOAD_WORDS[] = {\n";
    s += &c_list(&u64s(words), 6);
    s += "};\n\n";

    // CPU.
    let decode = kat_decode();
    let (entries, runs) = decode_records(&decode);
    let mut rng = 0x4350_5521u64;
    let logs: Vec<Log> = (0..KAT_CPU_LOGS)
        .map(|i| cpu_log(&mut rng, &decode, i % entries.len(), i % 3 == 0))
        .collect();
    let ops = cpu_ops(&logs, KAT_CPU_FIRST_CYCLE, &decode);
    let words = col_major(&cpu_host(&ops, &decode, TraceForm::Wide));
    assert_eq!(words.len(), KAT_CPU_HEIGHT * math_cuda::fill::CPU_COLUMNS);
    let absent = runs[0].first_pc + 2;
    assert_eq!(decode.row(absent), None);
    s += &format!(
        "// CPU: {} decode entries in {} runs; {KAT_CPU_LOGS} logs (their ops from cycle\n\
         // {KAT_CPU_FIRST_CYCLE}), each entry in turn, then padding rows to {KAT_CPU_HEIGHT}; every row's\n\
         // timestamp the table's own 4·(row + 1). KAT_CPU_ABSENT_PC is no entry's.\n\
         inline constexpr uint64_t KAT_CPU_N = {KAT_CPU_LOGS};\n\
         inline constexpr uint64_t KAT_CPU_HEIGHT = {KAT_CPU_HEIGHT};\n\
         inline constexpr uint64_t KAT_CPU_ABSENT_PC = {absent}ull;\n\
         inline constexpr uint32_t KAT_CPU_ENTRIES_N = {};\n\
         inline constexpr uint32_t KAT_CPU_RUNS_N = {};\n\
         inline const DecodeEntryC KAT_CPU_ENTRIES[KAT_CPU_ENTRIES_N] = {{\n",
        entries.len(),
        runs.len(),
        entries.len(),
        runs.len()
    );
    for e in &entries {
        s += &format!(
            "    {{{}ull, {}ull, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}}},\n",
            e.pc,
            e.imm,
            e.read_register1,
            e.read_register2,
            e.write_register,
            e.word_instr,
            e.alu,
            e.add,
            e.sub,
            e.memory,
            e.branch,
            e.ecall,
            e.rs1,
            e.rs2,
            e.rd,
            e.half_instruction_length,
            e.alu_flags,
            e.mem_flags
        );
    }
    s += "};\ninline const DecodeRunC KAT_CPU_RUNS[KAT_CPU_RUNS_N] = {\n";
    for r in &runs {
        s += &format!("    {{{}ull, {}u, {}u}},\n", r.first_pc, r.row, r.len);
    }
    s += "};\ninline const LogC KAT_CPU_LOGS[KAT_CPU_N] = {\n";
    for l in logs.iter().map(log_record) {
        s += &format!(
            "    {{{}ull, {}ull, {}ull, {}ull, {}ull}},\n",
            l.current_pc, l.next_pc, l.src1_val, l.src2_val, l.dst_val
        );
    }
    s += "};\n// Column-major: column c's rows from c * KAT_CPU_HEIGHT.\n";
    s += "inline const uint64_t KAT_CPU_WORDS[] = {\n";
    s += &c_list(&u64s(words), 6);
    s += "};\n\n";

    // KECCAK_RND.
    let ops = keccak_ops(0x4B45_4343, KAT_KECCAK_OPS);
    let ends = keccak_group_ends();
    s += &format!(
        "// KECCAK_RND: {KAT_KECCAK_OPS} ops; each case a table of rows [skip, skip + rows) of ops\n\
         // [first_op, first_op + nops); KAT_KECCAK_DIGESTS: per case, per row, per column group\n\
         // (columns [KAT_KECCAK_GROUP_END[g - 1], KAT_KECCAK_GROUP_END[g])), FNV-1a of the words'\n\
         // little-endian bytes.\n\
         inline constexpr int KAT_KECCAK_GROUPS = {};\n\
         inline const uint32_t KAT_KECCAK_GROUP_END[KAT_KECCAK_GROUPS] = {{{}}};\n\
         struct KatKeccakCase {{\n    uint32_t first_op;\n    uint32_t nops;\n    uint64_t skip;\n    uint64_t rows;\n}};\n\
         inline constexpr int KAT_KECCAK_CASES_N = {};\n\
         inline const KatKeccakCase KAT_KECCAK_CASES[KAT_KECCAK_CASES_N] = {{\n",
        ends.len(),
        ends.map(|e| e.to_string()).join(", "),
        KAT_KECCAK_CASES.len()
    );
    for (first, nops, skip, rows) in KAT_KECCAK_CASES {
        s += &format!("    {{{first}, {nops}, {skip}, {rows}}},\n");
    }
    s += &format!("}};\ninline const KeccakOpC KAT_KECCAK_OPS[{KAT_KECCAK_OPS}] = {{\n");
    for op in ops.iter().map(keccak_record) {
        s += &format!(
            "    {{{}ull,\n     {{{}}},\n     {{{}}}}},\n",
            op.timestamp,
            u64s(op.input).join(", "),
            u64s(op.output).join(", ")
        );
    }
    s += "};\ninline const uint64_t KAT_KECCAK_DIGESTS[] = {\n";
    let mut digests = Vec::new();
    for (first, nops, skip, rows) in KAT_KECCAK_CASES {
        let words = col_major(&keccak_rnd_host(
            &ops[first..first + nops],
            skip,
            rows,
            TraceForm::Wide,
        ));
        for row in 0..rows {
            let mut start = 0;
            for &end in &ends {
                digests.push(fnv1a((start..end).map(|c| words[c * rows + row])));
                start = end;
            }
        }
    }
    s += &c_list(&u64s(digests), 4);
    s += "};\n";
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::{cpu, memw_register};

    /// The card's column counts are the prover's tables', and a log is the
    /// record it uploads as.
    #[test]
    fn the_card_layouts_are_the_prover_tables() {
        use math_cuda::fill as f;
        assert_eq!(f::MEMW_R_COLUMNS, memw_register::cols::NUM_COLUMNS);
        assert_eq!(f::MEMW_A_COLUMNS, memw_aligned::cols::NUM_COLUMNS);
        assert_eq!(f::LOAD_COLUMNS, load::cols::NUM_COLUMNS);
        assert_eq!(f::CPU_COLUMNS, cpu::cols::NUM_COLUMNS);
        assert_eq!(f::KECCAK_RND_COLUMNS, keccak_rnd::cols::NUM_COLUMNS);
        assert_eq!(std::mem::size_of::<Log>(), std::mem::size_of::<LogC>());
        assert_eq!(std::mem::size_of::<f::RegRowC>(), 32);
        assert_eq!(std::mem::size_of::<KeccakOpC>(), 408);
    }

    /// The bytes of `value` as the card receives them.
    fn bytes_of<T: Copy>(value: &T) -> Vec<u8> {
        // SAFETY: `T` is one of the `#[repr(C)]` records, plain data with no
        // padding (asserted in `math_cuda::fill`).
        unsafe {
            std::slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>())
        }
        .to_vec()
    }

    /// The first `n` bytes of `value` as the card receives them: `T`'s fields
    /// up to its trailing padding.
    fn field_bytes<T>(value: &T, n: usize) -> Vec<u8> {
        assert!(n <= std::mem::size_of::<T>());
        // SAFETY: `T` is `#[repr(C)]` and its first `n` bytes are fields with
        // no padding between them (asserted by the callers' offsets), so every
        // byte read is initialized.
        unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), n) }.to_vec()
    }

    /// A register row's record: its fields, and its 32 bytes as uploaded.
    #[test]
    fn a_register_row_encodes_field_for_field() {
        let row = RegRow::new(
            2 * 37,
            0x3_0000_0010,
            0x1111_2222,
            0x3333_4444,
            0x5555_6666,
            0x7777_8888,
            0x3_0000_0001,
            true,
        );
        let c = row.card_record();
        let want = math_cuda::fill::RegRowC {
            timestamp: 0x3_0000_0010,
            val0: 0x1111_2222,
            val1: 0x3333_4444,
            old0: 0x5555_6666,
            old1: 0x7777_8888,
            old_ts_lo: 1,
            address: 37,
            is_read: 1,
            pad: 0,
        };
        assert_eq!(c, want);
        #[rustfmt::skip]
        let bytes = [
            0x10, 0, 0, 0, 3, 0, 0, 0,
            0x22, 0x22, 0x11, 0x11, 0x44, 0x44, 0x33, 0x33,
            0x66, 0x66, 0x55, 0x55, 0x88, 0x88, 0x77, 0x77,
            1, 0, 0, 0, 37, 0, 1, 0,
        ];
        assert_eq!(bytes_of(&c), bytes);
        let write = RegRow::new(0, 9, 0, 0, 0, 0, 8, false).card_record();
        assert_eq!((write.is_read, write.address, write.old_ts_lo), (0, 0, 8));
    }

    /// An aligned row's record: its fields, and its first 43 bytes as
    /// uploaded (the rest is padding no kernel reads).
    #[test]
    fn an_aligned_row_encodes_field_for_field() {
        let row = AlignedRow::from_parts(
            0x1122_3344_5566_7788,
            0x3_0000_0010,
            0x2_FFFF_FFF0,
            0x0102_0304_0506_0708,
            u64::MAX,
            4,
            true,
            false,
        );
        let c = aligned_record(&row);
        let want = AlignedRowC {
            base_address: 0x1122_3344_5566_7788,
            timestamp: 0x3_0000_0010,
            old_timestamp: 0x2_FFFF_FFF0,
            value: 0x0102_0304_0506_0708,
            old: u64::MAX,
            width: 4,
            is_read: 1,
            is_register: 0,
        };
        assert_eq!(c, want);
        let bytes = field_bytes(&row, 43);
        let words: Vec<u64> = bytes[..40]
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(
            words,
            [
                0x1122_3344_5566_7788,
                0x3_0000_0010,
                0x2_FFFF_FFF0,
                0x0102_0304_0506_0708,
                u64::MAX
            ]
        );
        assert_eq!(bytes[40..43], [4, 1, 0]);
        let register = AlignedRow::from_parts(0, 9, 8, 0, 0, 8, false, true);
        assert_eq!(field_bytes(&register, 43)[40..43], [8, 0, 1]);
    }

    /// A load op's record: its fields, and its first 26 bytes as uploaded
    /// (the rest is padding no kernel reads).
    #[test]
    fn a_load_op_encodes_field_for_field() {
        let op = LoadOperation::new(
            0xFF00_0000_0000_0008,
            0x1_0000_0004,
            2,
            true,
            [9, 8, 7, 6, 5, 4, 3, 2],
        );
        let c = load_record(&op);
        assert_eq!(
            c,
            LoadOpC {
                base_address: 0xFF00_0000_0000_0008,
                timestamp: 0x1_0000_0004,
                width: 2,
                is_signed: 1,
                res: [9, 8, 7, 6, 5, 4, 3, 2],
            }
        );
        let bytes = field_bytes(&op, 26);
        assert_eq!(bytes[..8], 0xFF00_0000_0000_0008u64.to_le_bytes());
        assert_eq!(bytes[8..16], 0x1_0000_0004u64.to_le_bytes());
        assert_eq!(bytes[16..26], [2, 1, 9, 8, 7, 6, 5, 4, 3, 2]);
        let unsigned = LoadOperation::new(0, 8, 1, false, [0; 8]);
        assert_eq!(field_bytes(&unsigned, 26)[16..18], [1, 0]);
    }

    /// A log's record: its five words in order.
    #[test]
    fn a_log_encodes_field_for_field() {
        let log = Log {
            current_pc: 0x1000,
            next_pc: 0x1004,
            src1_val: u64::MAX,
            src2_val: 7,
            dst_val: 0x0102_0304_0506_0708,
        };
        let c = log_record(&log);
        assert_eq!(
            (c.current_pc, c.next_pc, c.src1_val, c.src2_val, c.dst_val),
            (0x1000, 0x1004, u64::MAX, 7, 0x0102_0304_0506_0708)
        );
        let words: Vec<u64> = bytes_of(&c)
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(words, [0x1000, 0x1004, u64::MAX, 7, 0x0102_0304_0506_0708]);
    }

    /// The known-answer decode table's records: every entry's fields, the
    /// runs, and every pc found where the prover finds it (and the pcs around
    /// them nowhere).
    #[test]
    fn a_decode_table_encodes_row_for_row() {
        let decode = kat_decode();
        let (entries, runs) = decode_records(&decode);
        assert_eq!(entries.len(), 24);
        assert_eq!(
            runs,
            RUN_BASES.map(|first_pc| DecodeRunC {
                first_pc,
                row: entries.iter().position(|e| e.pc == first_pc).unwrap() as u32,
                len: 8,
            })
        );
        for (row, (e, c)) in decode.entries().iter().zip(&entries).enumerate() {
            let f = &e.fields;
            assert_eq!((c.pc, c.imm), (e.pc, e.imm), "row {row}");
            let flags = [
                f.read_register1,
                f.read_register2,
                f.write_register,
                f.word_instr,
                f.alu,
                f.add,
                f.sub,
                f.memory,
                f.branch,
                f.ecall,
            ]
            .map(u8::from);
            let bytes = bytes_of(c);
            assert_eq!(bytes[16..26], flags, "row {row}");
            assert_eq!(
                bytes[26..32],
                [
                    f.rs1,
                    f.rs2,
                    f.rd,
                    f.half_instruction_length,
                    f.alu_flags,
                    f.mem_flags
                ],
                "row {row}"
            );
            assert_eq!(DecodeRunC::find(&runs, e.pc), Some(row as u32), "row {row}");
            for miss in [e.pc + 2, e.pc.wrapping_sub(2)] {
                assert_eq!(
                    DecodeRunC::find(&runs, miss),
                    decode.row(miss),
                    "pc {miss:#x}"
                );
            }
        }
        for pc in [0, RUN_BASES[0] - 4, RUN_BASES[1] + 32, !3] {
            assert_eq!(DecodeRunC::find(&runs, pc), None, "pc {pc:#x}");
            assert_eq!(decode.row(pc), None, "pc {pc:#x}");
        }
        // Every op class the CPU fill branches on is in the table.
        let has = |p: &dyn Fn(&DecodeEntryC) -> bool| entries.iter().any(p);
        assert!(has(&|e| e.word_instr == 1));
        assert!(has(&|e| e.memory == 1 && e.mem_flags & 1 == 1), "a store");
        assert!(has(&|e| e.memory == 1 && e.mem_flags & 1 == 0), "a load");
        assert!(has(&|e| e.branch == 1 && e.mem_flags & 1 == 1), "a JALR");
        assert!(has(&|e| e.rs1 == 255), "a pc read");
        assert!(has(&|e| e.ecall == 1));
        assert!(has(&|e| e.add == 1) && has(&|e| e.sub == 1) && has(&|e| e.alu == 1));
    }

    /// A keccak operation's record: the timestamp, then input and output.
    #[test]
    fn a_keccak_op_encodes_field_for_field() {
        let op = &keccak_ops(9, 1)[0];
        let c = keccak_record(op);
        assert_eq!(
            (c.timestamp, c.input, c.output),
            (op.timestamp, op.input, op.output)
        );
        let words: Vec<u64> = bytes_of(&c)
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(words[0], op.timestamp);
        assert_eq!(words[1..26], op.input);
        assert_eq!(words[26..51], op.output);
    }

    /// The synthetic sets are what the module docs say.
    #[test]
    fn the_synthetic_sets_have_their_shapes() {
        for r in memw_r_rows(SEEDS[0], 1 << 12)
            .iter()
            .map(RegRow::card_record)
        {
            let ts_lo = (r.timestamp & 0xFFFF_FFFF) as u32;
            assert!((8..1 << 34).contains(&r.timestamp));
            assert!(ts_lo > r.old_ts_lo && ts_lo - r.old_ts_lo <= 1 << 16);
        }
        let decode = parsed_decode(SEEDS[0], 256);
        assert_eq!(decode.runs().len(), RUN_BASES.len());
        let logs = cpu_logs(SEEDS[0], 1 << 12, &decode);
        assert!(logs.iter().all(|l| decode.row(l.current_pc).is_some()));
        let ops = keccak_ops(SEEDS[0], KECCAK_RND_OPS);
        assert!(ops.windows(2).all(|w| w[0].timestamp < w[1].timestamp));
        assert!(ops[0].timestamp < 1 << 32 && ops[KECCAK_RND_OPS - 1].timestamp > 1 << 32);
    }

    /// [`mismatches`] of `packed`'s words against `wide`'s, a column at a
    /// time (a 2^16-row KECCAK_RND table is 776 MB at 8 bytes a cell).
    fn packed_and_wide(packed: &Trace, wide: &Trace) -> Mismatches {
        let rows = wide.num_rows();
        let want = col_major(wide);
        let Some(columns) = packed.narrow_main() else {
            // A build that holds no table packed (`debug-checks`).
            return mismatches(&want, &col_major(packed), rows);
        };
        let mut out = Mismatches {
            count: want.len().abs_diff(rows * columns.cols()),
            first: None,
        };
        for c in 0..columns.cols().min(wide.main_table.width) {
            let m = mismatches(&want[c * rows..(c + 1) * rows], &columns.column(c), rows);
            out.count += m.count;
            if out.first.is_none() {
                out.first = m.first.map(|(_, row, w, g)| (c, row, w, g));
            }
        }
        out
    }

    /// Every synthetic input's host fill is the same words packed as at 8
    /// bytes a cell (G-pack's invariant, which the card's identity rests on).
    #[test]
    fn every_synthetic_input_is_the_same_words_wide_and_packed() {
        let decode = program_decode();
        for seed in SEEDS {
            let rows = memw_r_rows(seed, CHUNK_ROWS);
            let m = packed_and_wide(
                &memw_r_host(&rows, TraceForm::Narrow),
                &memw_r_host(&rows, TraceForm::Wide),
            );
            assert_eq!(m, Mismatches::default(), "MEMW_R seed {seed}");

            let ops = cpu_ops(
                &cpu_logs(seed, CHUNK_ROWS, &decode),
                CPU_FIRST_CYCLE,
                &decode,
            );
            let m = packed_and_wide(
                &cpu_host(&ops, &decode, TraceForm::Narrow),
                &cpu_host(&ops, &decode, TraceForm::Wide),
            );
            assert_eq!(m, Mismatches::default(), "CPU seed {seed}");

            let ops = keccak_ops(seed, KECCAK_RND_OPS);
            let m = packed_and_wide(
                &keccak_rnd_host(&ops, KECCAK_RND_SKIP, KECCAK_RND_ROWS, TraceForm::Narrow),
                &keccak_rnd_host(&ops, KECCAK_RND_SKIP, KECCAK_RND_ROWS, TraceForm::Wide),
            );
            assert_eq!(m, Mismatches::default(), "KECCAK_RND seed {seed}");
        }
    }

    /// The comparison the identity checks rest on finds one changed word,
    /// where it is.
    #[test]
    fn one_changed_word_is_one_mismatch() {
        let rows = memw_r_rows(SEEDS[1], 100);
        let want = col_major(&memw_r_host(&rows, TraceForm::Wide));
        let height = 128;
        let mut got = col_major(&memw_r_host(&rows, TraceForm::Narrow));
        assert_eq!(mismatches(&want, &got, height), Mismatches::default());
        let at = memw_register::cols::OLD_1 * height + 41;
        got[at] ^= 1 << 7;
        assert_eq!(
            mismatches(&want, &got, height),
            Mismatches {
                count: 1,
                first: Some((memw_register::cols::OLD_1, 41, want[at], got[at])),
            }
        );
        assert_eq!(mismatches(&want, &got[..got.len() - 3], height).count, 4);
    }

    /// The harness's vectors are the host fills' of today: regenerated and
    /// compared with the committed file (`CARD_FILL_KAT_WRITE=1` writes it).
    #[test]
    fn the_host_kat_vectors_are_the_host_fills() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the workspace root")
            .join(HOST_KAT_VECTORS);
        let text = host_kat_vectors();
        if std::env::var("CARD_FILL_KAT_WRITE").is_ok_and(|v| v == "1") {
            std::fs::write(&path, &text).expect("write the vectors");
        }
        let committed = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            committed == text,
            "{} is not the host fills' vectors; regenerate it (its header says how)",
            path.display()
        );
    }
}
