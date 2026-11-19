//! DECODE table for instruction decoding.
//!
//! The DECODE table contains all decoded instructions from the program.
//! It receives lookups from the CPU table to verify instruction decoding.
//!
//! ## Columns (Compressed Form)
//!
//! - `pc`: DWordWL (2 cols) - program counter
//! - `packed_decode`: BaseField (1 col) - packed flags and register indices
//! - `imm`: DWordWL (2 cols) - fully extended 64-bit immediate
//! - `μ`: BaseField (1 col) - multiplicity
//!
//! ## packed_decode Format
//!
//! A single base-field element packing the control flags, register indices, and
//! the `alu_flags`/`mem_flags` bytes. The authoritative bit layout lives in
//! `packed_decode_shrunk` and is produced by `ShrunkDecode::pack` (both in
//! `tables/types.rs`) — consult those for the exact bit position of every field.
//! Summary (low → high bits):
//!
//! ```text
//! Bits [0..10]:  read_register1, read_register2, write_register, word_instr,
//!                ALU, ADD, SUB, MEMORY, BRANCH, ECALL (one bit each)
//! Bits [10..34]: rs1, rs2, rd (8 bits each)
//! Bits [34..42]: half_instruction_length (Byte: byte length / 2)
//! Bits [42..50]: alu_flags (Byte: alu_op in bits 0-4, then signed / signed2|invert / muldiv)
//! Bits [50..58]: mem_flags (Byte: JALR|memory_op, signed, 2B, 4B, 8B)
//! ```
//!
//! ## Bus Interactions
//!
//! - **Receiver**: DECODE bus - receives lookups from CPU table

use executor::elf::Elf;
use executor::vm::instruction::decoding::{Instruction, InstructionError};
use executor::vm::memory::U64HashMap;
use math::polynomial::Polynomial;
use stark::config::Commitment;
use stark::leaf_layout::LeafLayout;
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};

// Re-export DecodeEntry from types for backwards compatibility
pub use super::types::DecodeEntry;

// =========================================================================
// Column indices for DECODE table
// =========================================================================

/// Column definitions for the DECODE table.
pub mod cols {
    // PC as DWordWL (2 columns)
    /// pc[0]: Program counter (low word, bits 0-31)
    pub const PC_0: usize = 0;
    /// pc[1]: Program counter (high word, bits 32-63)
    pub const PC_1: usize = 1;

    // packed_decode (1 column)
    /// packed_decode: All flags and register indices packed into single field element
    pub const PACKED_DECODE: usize = 2;

    // imm as DWordWL (2 columns)
    /// imm[0]: Immediate value (low word, bits 0-31)
    pub const IMM_0: usize = 3;
    /// imm[1]: Immediate value (high word, bits 32-63)
    pub const IMM_1: usize = 4;

    // Multiplicity column
    /// μ: Multiplicity for bus interactions
    pub const MU: usize = 5;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 6;
}

/// Number of precomputed columns (PC_0, PC_1, PACKED_DECODE, IMM_0, IMM_1).
/// The remaining column (MU) is the multiplicity column that varies per execution.
pub const NUM_PRECOMPUTED_COLS: usize = 5;

/// The DECODE table under the lean CPU row
/// ([`CpuRowForm::Lean`](super::cpu::CpuRowForm::Lean)): [`cols`]' five
/// precomputed columns, unchanged, then the six decode bytes, each its own
/// precomputed column, then `MU`. The receiver's flags element is
/// `PACKED_DECODE` less the six bytes at their offsets — the ten flag bits.
pub mod lean_cols {
    pub use super::cols::{IMM_0, IMM_1, PACKED_DECODE, PC_0, PC_1};
    /// `rs1`, then `rs2`, `rd`, `half_instruction_length`, `alu_flags` and
    /// `mem_flags` at the next five columns (the CPU's
    /// [`DECODE_BYTE_COLUMNS`](crate::tables::cpu::DECODE_BYTE_COLUMNS) order).
    pub const RS1: usize = 5;
    pub const RS2: usize = 6;
    pub const RD: usize = 7;
    pub const HALF_INSTRUCTION_LENGTH: usize = 8;
    pub const ALU_FLAGS: usize = 9;
    pub const MEM_FLAGS: usize = 10;
    /// μ: multiplicity.
    pub const MU: usize = 11;
    pub const NUM_COLUMNS: usize = 12;
}

/// [`lean_cols`]' precomputed columns: all but `MU`.
pub const NUM_PRECOMPUTED_COLS_LEAN: usize = 11;

/// The decode bytes' offsets in `PACKED_DECODE`, in [`lean_cols`]' order.
const BYTE_OFFSETS: [u32; 6] = {
    use super::types::packed_decode_shrunk as pd;
    [
        pd::RS1,
        pd::RS2,
        pd::RD,
        pd::HALF_INSTRUCTION_LENGTH,
        pd::ALU_FLAGS,
        pd::MEM_FLAGS,
    ]
};

/// DECODE's columns under `form`.
pub const fn num_columns(form: super::cpu::CpuRowForm) -> usize {
    if form.decode_bytes_apart() {
        lean_cols::NUM_COLUMNS
    } else {
        cols::NUM_COLUMNS
    }
}

/// DECODE's precomputed columns under `form`.
pub const fn num_precomputed_cols(form: super::cpu::CpuRowForm) -> usize {
    if form.decode_bytes_apart() {
        NUM_PRECOMPUTED_COLS_LEAN
    } else {
        NUM_PRECOMPUTED_COLS
    }
}

/// A DECODE trace in [`cols`]' layout, its counts made, in `form`'s: itself
/// under [`CpuRowForm::Full`](super::cpu::CpuRowForm::Full), and under
/// [`CpuRowForm::Lean`](super::cpu::CpuRowForm::Lean) each row's entry as the
/// lean row holds it ([`entry_under`]: an ECALL's with its ADD flag) and the
/// six decode bytes, read off its `PACKED_DECODE`, inserted before `MU`.
///
/// # Panics
/// On a trace that is not [`cols`]' width.
pub fn trace_under(
    trace: TraceTable<GoldilocksField, GoldilocksExtension>,
    form: super::cpu::CpuRowForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    if !form.decode_bytes_apart() {
        return trace;
    }
    let num_rows = trace.num_rows();
    assert_eq!(
        trace.main_table.width,
        cols::NUM_COLUMNS,
        "a DECODE trace in today's layout"
    );
    let mut lean = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * lean_cols::NUM_COLUMNS),
        lean_cols::NUM_COLUMNS,
        1,
    );
    let table = &mut lean.main_table;
    for row in 0..num_rows {
        for col in 0..NUM_PRECOMPUTED_COLS {
            table.set_fe(row, col, *trace.main_table.get(row, col));
        }
        let packed = packed_under(
            trace.main_table.get(row, cols::PACKED_DECODE).canonical(),
            form,
        );
        table.set_u64(row, lean_cols::PACKED_DECODE, packed);
        for (k, offset) in BYTE_OFFSETS.iter().enumerate() {
            table.set_u64(row, lean_cols::RS1 + k, (packed >> offset) & 0xFF);
        }
        table.set_fe(row, lean_cols::MU, *trace.main_table.get(row, cols::MU));
    }
    lean
}

// =========================================================================
// Trace generation
// =========================================================================

/// Map from PC to row index in the DECODE trace table.
pub type PcToRow = U64HashMap<usize>;

/// Generates the DECODE trace table from the instructions map.
///
/// Returns the trace table and a map from PC to row index for use with
/// `update_multiplicities`. All multiplicities are initialized to 0.
///
/// ## Padding
///
/// Empty rows use pc=7 with EBREAK=1, which makes them unprovable
/// since CPU asserts EBREAK=0.
pub fn generate_decode_trace(
    instructions: &U64HashMap<Instruction>,
) -> (TraceTable<GoldilocksField, GoldilocksExtension>, PcToRow) {
    // ★★ ROWS GO IN PC ORDER, and the sort is the whole point of this block.
    //
    // The rows used to come out of `instructions.iter()`, so their order was
    // hashbrown's: a function of the hasher, the capacity the map happened to
    // grow to, and the insertion sequence. Every one of those is stable for a
    // given binary, which is why nothing has ever failed — prover and verifier
    // both reach this through `instructions_from_elf`, so they agree with each
    // other. What they agree on is a CONSTRUCTION PROCEDURE, not the ELF.
    //
    // That distinction is about to start mattering. These five columns are
    // ELF-derived and their Merkle root is on its way to being a program
    // constant pinned in a recursion guest (W1-B). A pinned root must be a
    // function of the ELF ALONE: sorted by pc it is, and a hashbrown version
    // bump, a capacity change or a reserve added upstream cannot move it.
    // Unsorted it is not, and the failure mode is the bad kind — a toolchain
    // update silently invalidates the pin, with no ELF change, no code change
    // and no test that fails until a verifier rejects a valid proof.
    //
    // pc is unique (it is the map's key), so the order is total and the sort is
    // not merely deterministic but canonical.
    let mut entries: Vec<(u64, Instruction)> = instructions
        .iter()
        .map(|(&pc, &instr)| (pc, instr))
        .collect();
    entries.sort_unstable_by_key(|(pc, _)| *pc);

    let mut pc_to_row = PcToRow::default();
    pc_to_row.reserve(instructions.len() + 1);
    let entries: Vec<_> = entries
        .into_iter()
        .enumerate()
        .map(|(row_idx, (pc, instr))| {
            pc_to_row.insert(pc, row_idx);
            // instruction_length = 4 (RV64C compressed decode is a separate workstream).
            DecodeEntry::from_instruction(pc, instr, 4)
        })
        .collect();

    // Add the CPU padding entry: pc=CPU_PADDING_PC, all flags=0 (per spec, decode must
    // include this). This row is looked up by CPU padding rows. Its MU will be set by
    // update_multiplicities.
    let cpu_padding_row = entries.len();
    pc_to_row.insert(super::cpu::CPU_PADDING_PC, cpu_padding_row);
    let cpu_padding_entry = DecodeEntry {
        pc: super::cpu::CPU_PADDING_PC,
        ..Default::default()
    };

    // Pad to next power of 2, minimum 2
    // +1 for the CPU padding entry
    let num_entries = entries.len() + 1;
    let num_rows = num_entries.next_power_of_two().max(2);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    // Fill actual entries (MU = 0 initially)
    for (row_idx, entry) in entries.iter().enumerate() {
        // PC as DWordWL
        table.set_dword_wl(row_idx, cols::PC_0, entry.pc);

        // packed_decode
        table.set_u64(row_idx, cols::PACKED_DECODE, entry.packed_decode());

        // imm as DWordWL
        table.set_dword_wl(row_idx, cols::IMM_0, entry.imm);

        // MU = 0 (already zero from vec initialization)
    }

    // Write CPU padding entry (pc=1, all flags=0)
    {
        table.set_dword_wl(cpu_padding_row, cols::PC_0, cpu_padding_entry.pc);
        table.set_u64(
            cpu_padding_row,
            cols::PACKED_DECODE,
            cpu_padding_entry.packed_decode(),
        );
        table.set_dword_wl(cpu_padding_row, cols::IMM_0, cpu_padding_entry.imm);
    }

    // Fill padding rows with the DECODE padding pattern: odd pc=1, all flags 0
    // (unprovable as a fetch target; same row the CPU pads to).
    let padding_entry = DecodeEntry::padding_entry();
    for row_idx in num_entries..num_rows {
        table.set_dword_wl(row_idx, cols::PC_0, padding_entry.pc);
        table.set_u64(row_idx, cols::PACKED_DECODE, padding_entry.packed_decode());
        table.set_dword_wl(row_idx, cols::IMM_0, padding_entry.imm);
        // MU = 0 for padding rows (already zero from vec initialization)
    }

    (trace, pc_to_row)
}

/// Every instruction of a program decoded once, in the DECODE table's row
/// order: the entry of row `r` is [`generate_decode_trace`]'s row `r` (both
/// sort the pcs), and the CPU padding entry's row follows the last
/// instruction's. A CPU op keeps its DECODE row instead of a copy of the entry
/// ([`super::cpu::CpuOperation`]), so its decode is one indexed read here and
/// its DECODE lookup is counted by row, with no pc search.
#[derive(Debug)]
pub struct DecodeTable {
    /// The entries, in pc order.
    entries: Vec<DecodeEntry>,
    /// `(first pc, its row, number of pcs)` of each run of pcs 4 apart,
    /// ascending: a pc's row is found by run, not by hashing.
    runs: Vec<(u64, u32, u32)>,
}

#[cfg(test)]
impl DecodeTable {
    /// The entries, in pc order: entry `r` is DECODE row `r`.
    pub(crate) fn entries(&self) -> &[DecodeEntry] {
        &self.entries
    }

    /// The runs of pcs 4 apart, ascending: `(first pc, its row, number of pcs)`.
    pub(crate) fn runs(&self) -> &[(u64, u32, u32)] {
        &self.runs
    }
}

/// `entry` as DECODE holds it under the CPU row's `form`: under
/// [`CpuRowForm::Lean`](super::cpu::CpuRowForm::Lean) an ECALL's entry sets
/// the ADD flag too, so its row's `res = rv1 + arg2 = a7` — `arg2` is
/// `rv2 + imm`, both zero on an ECALL (no `rs2` read, no immediate) — and
/// `rvd = res`: the merged memory dispatch carries the syscall number as its
/// address and result, pinned by the ADD fast path's identity. Every other
/// entry, and every entry under today's row, is unchanged.
pub fn entry_under(mut entry: DecodeEntry, form: super::cpu::CpuRowForm) -> DecodeEntry {
    if form.ecall_on_memory_op() && entry.fields.ecall {
        entry.fields.add = true;
    }
    entry
}

/// A `PACKED_DECODE` word under the CPU row's `form` ([`entry_under`]): the
/// ADD bit set on an ECALL's under the lean row.
fn packed_under(packed: u64, form: super::cpu::CpuRowForm) -> u64 {
    use super::types::packed_decode_shrunk as pd;
    if form.ecall_on_memory_op() && (packed >> pd::ECALL) & 1 == 1 {
        packed | (1 << pd::ADD)
    } else {
        packed
    }
}

/// The entries the lean row's ECALL argument covers ([`entry_under`]): every
/// ECALL entry reads `rs1 = 17` (a7) and nothing else, has no immediate, and
/// sets no flag but `read_register1` and ECALL (no rs2 read, no register
/// write, no ALU, ADD, SUB, MEMORY, BRANCH or word flag, both flag bytes 0).
/// Then the ADD fast path gives `res = rv1 + rv2 + imm = a7`. An entry outside
/// that set refuses the lean row instead of giving it a table its argument
/// does not cover. `Ok` under today's row whatever the entries.
pub fn check_lean_ecall_entries<'a>(
    entries: impl IntoIterator<Item = &'a DecodeEntry>,
    form: super::cpu::CpuRowForm,
) -> Result<(), String> {
    if !form.ecall_on_memory_op() {
        return Ok(());
    }
    for e in entries {
        let f = &e.fields;
        if !f.ecall {
            continue;
        }
        let covered = e.imm == 0
            && f.read_register1
            && f.rs1 == 17
            && !f.read_register2
            && !f.write_register
            && !f.word_instr
            && !f.alu
            && !f.add
            && !f.sub
            && !f.memory
            && !f.branch
            && f.alu_flags == 0
            && f.mem_flags == 0;
        if !covered {
            return Err(format!(
                "the ECALL at pc {:#x} is not the entry the lean CPU row covers \
                 (imm {}, fields {:?})",
                e.pc, e.imm, f
            ));
        }
    }
    Ok(())
}

impl DecodeTable {
    /// Every entry, in pc order (the CPU padding entry not among them).
    pub fn entries_all(&self) -> &[DecodeEntry] {
        &self.entries
    }

    /// This table under the CPU row's `form` ([`entry_under`] of each entry;
    /// the rows and runs are the same), or a refusal of an ECALL entry the
    /// lean row does not cover ([`check_lean_ecall_entries`]).
    pub fn under(&self, form: super::cpu::CpuRowForm) -> Result<Self, String> {
        check_lean_ecall_entries(&self.entries, form)?;
        Ok(Self {
            entries: self
                .entries
                .iter()
                .map(|e| entry_under(e.clone(), form))
                .collect(),
            runs: self.runs.clone(),
        })
    }

    /// The table of `instructions`, each decoded as the DECODE trace decodes
    /// it (length 4).
    ///
    /// # Panics
    /// If the program has 2^32 − 1 instructions or more (rows are `u32`).
    pub fn from_instructions(instructions: &U64HashMap<Instruction>) -> Self {
        let mut pcs: Vec<u64> = instructions.keys().copied().collect();
        // The order of `generate_decode_trace`'s rows.
        pcs.sort_unstable();
        assert!(
            pcs.len() < u32::MAX as usize,
            "{} instructions do not fit u32 DECODE rows",
            pcs.len()
        );
        let mut entries = Vec::with_capacity(pcs.len());
        let mut runs: Vec<(u64, u32, u32)> = Vec::new();
        for pc in pcs {
            let row = entries.len() as u32;
            entries.push(DecodeEntry::from_instruction(pc, instructions[&pc], 4));
            match runs.last_mut() {
                Some((first, _, len)) if first.checked_add(4 * *len as u64) == Some(pc) => {
                    *len += 1
                }
                _ => runs.push((pc, row, 1)),
            }
        }
        Self { entries, runs }
    }

    /// The DECODE row of the instruction at `pc`, if the program has one.
    #[inline]
    pub fn row(&self, pc: u64) -> Option<u32> {
        self.runs.iter().find_map(|&(first, row, len)| {
            let offset = pc.wrapping_sub(first);
            (offset % 4 == 0 && offset / 4 < len as u64).then(|| row + (offset / 4) as u32)
        })
    }

    /// The table as the card's CPU fill reads it
    /// (`multilinear::records::DecodeRecords`): each entry's pc, immediate and
    /// decoded fields as bytes, in row order, and the runs. `None` for a
    /// program with no instruction.
    pub(crate) fn card_records(&self) -> Option<multilinear::records::DecodeRecords> {
        use multilinear::records::{DecodeRecords, DecodeRow, DecodeRun};
        let entries = self
            .entries
            .iter()
            .map(|e| {
                let f = &e.fields;
                DecodeRow {
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
        let runs = self
            .runs
            .iter()
            .map(|&(first_pc, row, len)| DecodeRun { first_pc, row, len })
            .collect();
        DecodeRecords::new(entries, runs)
    }

    /// The entry of DECODE row `row` (a row [`Self::row`] returned).
    #[inline]
    pub fn entry(&self, row: u32) -> &DecodeEntry {
        &self.entries[row as usize]
    }

    /// The DECODE row of the CPU padding entry (`pc = CPU_PADDING_PC`).
    pub fn padding_row(&self) -> usize {
        self.entries.len()
    }

    /// `op` with its decode.
    #[inline]
    pub fn op<'a>(&'a self, op: &'a super::cpu::CpuOperation) -> super::cpu::CpuOp<'a> {
        super::cpu::CpuOp::new(self.entry(op.decode_row), op)
    }
}

/// The BITWISE lookups today's CPU row and CPU32 send and the lean row
/// ([`CpuRowForm::Lean`](super::cpu::CpuRowForm::Lean)) does not, for DECODE
/// counts `counts` (row `r` of `table` looked up `counts[r]` times, the CPU
/// padding row included): each with its total.
///
/// They follow from DECODE's counts alone. Every CPU operation looks DECODE up
/// once — through the CPU row, or through its CPU32 row on a word instruction
/// — and every CPU padding row once, at the padding entry; and the checks that
/// go read only the entry's bytes:
/// - a row that is not a word instruction (the padding row too):
///   `ARE_BYTES[rs1, rs2]`, `ARE_BYTES[rd, half_instruction_length]`,
///   `ARE_BYTES[alu_flags, mem_flags]`;
/// - a word instruction: its CPU row's `ARE_BYTES[0, 0]` twice and
///   `ARE_BYTES[0, half_instruction_length]` (the CPU zeroes the other bytes
///   there, [`super::cpu::CpuOp::collect_bitwise_ops`]), and its CPU32 row's
///   `ARE_BYTES[x, 0]` on `half_instruction_length`, `alu_flags`, `rs1`, `rs2`
///   and `rd`.
pub(crate) fn lean_removed_bitwise(
    table: &DecodeTable,
    counts: &[u64],
) -> Vec<(super::bitwise::BitwiseOperation, u64)> {
    use super::bitwise::{BitwiseOperation, BitwiseOperationType::AreBytes};
    let padding = DecodeEntry {
        pc: super::cpu::CPU_PADDING_PC,
        ..Default::default()
    };
    let mut out = Vec::new();
    for (row, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let entry = if row == table.padding_row() {
            &padding
        } else {
            table.entry(row as u32)
        };
        let f = &entry.fields;
        if f.word_instr {
            out.push((BitwiseOperation::byte_op(AreBytes, 0, 0), 2 * count));
            out.push((
                BitwiseOperation::byte_op(AreBytes, 0, f.half_instruction_length),
                count,
            ));
            for byte in [f.half_instruction_length, f.alu_flags, f.rs1, f.rs2, f.rd] {
                out.push((BitwiseOperation::single_byte(AreBytes, byte), count));
            }
        } else {
            out.push((BitwiseOperation::byte_op(AreBytes, f.rs1, f.rs2), count));
            out.push((
                BitwiseOperation::byte_op(AreBytes, f.rd, f.half_instruction_length),
                count,
            ));
            out.push((
                BitwiseOperation::byte_op(AreBytes, f.alu_flags, f.mem_flags),
                count,
            ));
        }
    }
    out
}

/// DECODE's counts so far in `trace`'s `MU` column ([`cols`]' layout), plus
/// one for each of `rows` and `num_padding_rows` at the CPU padding row: the
/// counts [`count_rows`] would leave, read without writing them.
pub(crate) fn counts_with(
    trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
    table: &DecodeTable,
    rows: impl IntoIterator<Item = usize>,
    num_padding_rows: usize,
) -> Vec<u64> {
    let mut counts: Vec<u64> = (0..trace.num_rows())
        .map(|row| trace.main_table.get(row, cols::MU).canonical())
        .collect();
    for row in rows {
        counts[row] += 1;
    }
    counts[table.padding_row()] += num_padding_rows as u64;
    counts
}

/// Counts one DECODE lookup per row of `rows` into the MU column (rows a
/// [`DecodeTable`] gives).
pub fn count_rows(
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    rows: impl IntoIterator<Item = usize>,
) {
    for row in rows {
        let current = trace.main_table.get(row, cols::MU);
        trace.main_table.set_fe(row, cols::MU, current + FE::one());
    }
}

/// Adds `counts[row]` DECODE lookups to each row's MU column at once: the
/// column [`count_rows`] leaves after `counts[row]` lookups of that row, the
/// field's sums being exact. Rows with no count are not touched. The first
/// counted row the trace does not have is the error.
pub fn add_row_counts(
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    counts: &[u64],
) -> Result<(), usize> {
    for (row, &n) in counts.iter().enumerate() {
        if n == 0 {
            continue;
        }
        if row >= trace.main_table.height {
            return Err(row);
        }
        let current = trace.main_table.get(row, cols::MU);
        trace
            .main_table
            .set_fe(row, cols::MU, current + FE::from(n));
    }
    Ok(())
}

/// Updates multiplicities in the DECODE trace table.
///
/// For each PC in `lookups`, increments the MU column in the corresponding row.
pub fn update_multiplicities(
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    pc_to_row: &PcToRow,
    lookups: &[u64],
) {
    for &pc in lookups {
        if let Some(&row_idx) = pc_to_row.get(&pc) {
            let current = trace.main_table.get(row_idx, cols::MU);
            trace
                .main_table
                .set_fe(row_idx, cols::MU, current + FE::one());
        }
    }
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the DECODE table.
///
/// The DECODE table is a **receiver** that accepts lookups from the CPU table.
/// Per spec (cpu.toml): input = ["pc", "imm", "packed_decode"]
pub fn bus_interactions() -> Vec<BusInteraction> {
    bus_interactions_for(super::cpu::CpuRowForm::Full)
}

/// [`bus_interactions`] under `form`: under
/// [`CpuRowForm::Lean`](super::cpu::CpuRowForm::Lean)
/// `DECODE[pc, imm, flags, rs1, rs2, rd, half_instruction_length, alu_flags,
/// mem_flags]` at [`lean_cols`], `flags` being `PACKED_DECODE` less the six
/// bytes at their offsets.
pub fn bus_interactions_for(form: super::cpu::CpuRowForm) -> Vec<BusInteraction> {
    if form.decode_bytes_apart() {
        let mut flags = vec![stark::lookup::LinearTerm::Column {
            coefficient: 1,
            column: lean_cols::PACKED_DECODE,
        }];
        flags.extend(BYTE_OFFSETS.iter().enumerate().map(|(k, &offset)| {
            stark::lookup::LinearTerm::Column {
                coefficient: -(1i64 << offset),
                column: lean_cols::RS1 + k,
            }
        }));
        let mut values = vec![
            BusValue::Packed {
                start_column: lean_cols::PC_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: lean_cols::IMM_0,
                packing: Packing::DWordWL,
            },
            BusValue::linear(flags),
        ];
        values.extend((0..BYTE_OFFSETS.len()).map(|k| BusValue::Packed {
            start_column: lean_cols::RS1 + k,
            packing: Packing::Direct,
        }));
        return vec![BusInteraction::receiver(
            BusId::Decode,
            Multiplicity::Column(lean_cols::MU),
            values,
        )];
    }
    vec![
        // DECODE[pc, imm, packed_decode] - receiver from CPU
        BusInteraction::receiver(
            BusId::Decode,
            Multiplicity::Column(cols::MU),
            vec![
                // pc as DWordWL (2 bus elements)
                BusValue::Packed {
                    start_column: cols::PC_0,
                    packing: Packing::DWordWL,
                },
                // imm as DWordWL (2 bus elements)
                BusValue::Packed {
                    start_column: cols::IMM_0,
                    packing: Packing::DWordWL,
                },
                // packed_decode as Direct (1 bus element)
                BusValue::Packed {
                    start_column: cols::PACKED_DECODE,
                    packing: Packing::Direct,
                },
            ],
        ),
    ]
}

// =========================================================================
// Precomputed commitment
// =========================================================================

/// The precomputed columns themselves, `0..NUM_PRECOMPUTED_COLS` of the DECODE
/// trace: the program's instruction table.
///
/// The multilinear path checks a proof's claimed openings against these instead
/// of comparing a commitment, so it needs the values and not just their root.
/// This is what binds a proof to the program it claims to run.
pub fn preprocessed_columns(instructions: &U64HashMap<Instruction>) -> Vec<Vec<FE>> {
    preprocessed_columns_for(instructions, super::cpu::CpuRowForm::Full)
}

/// [`preprocessed_columns`] under `form` ([`num_precomputed_cols`] of them).
pub fn preprocessed_columns_for(
    instructions: &U64HashMap<Instruction>,
    form: super::cpu::CpuRowForm,
) -> Vec<Vec<FE>> {
    // MU=0: only the precomputed columns are wanted.
    let (trace, _pc_to_row) = generate_decode_trace(instructions);
    let trace = trace_under(trace, form);
    let num_rows = trace.num_rows();
    (0..num_precomputed_cols(form))
        .map(|col_idx| {
            (0..num_rows)
                .map(|row_idx| *trace.main_table.get(row_idx, col_idx))
                .collect()
        })
        .collect()
}

/// Computes the LDE commitment for DECODE precomputed columns.
///
/// This builds a Merkle tree over the LDE (Low Degree Extension) of the precomputed
/// columns (PC_0, PC_1, PACKED_DECODE, IMM_0, IMM_1), matching exactly how the prover
/// commits to traces.
///
/// Used by both prover (sanity check) and verifier (soundness check). Pure
/// library function — no caching, no side effects. Callers manage their own
/// caching, hardcoding, or recomputation policy as needed:
///
/// * **Always recompute**: call this function (or [`commitment_from_elf`])
///   on every verify. Simple and slow.
/// * **Cache once per process**: wrap the call in a `OnceLock` /
///   `HashMap<elf_hash, Commitment>` at the caller site. Useful for native
///   verifiers that check many proofs of the same ELF in one process.
/// * **Compile-time constant**: call this function once offline (e.g. from
///   a one-off test in the consumer crate that prints the result), then
///   store the resulting bytes as a `const [u8; 32]` in the caller's
///   source. Useful for the recursion guest where in-VM recomputation is
///   too expensive.
///
/// ## Arguments
/// * `instructions` - The program's instruction map (PC → Instruction)
/// * `options` - Proof options containing blowup factor and coset offset
///
/// ## Returns
/// The Merkle root commitment over the LDE of precomputed columns.
pub fn compute_precomputed_commitment(
    instructions: &U64HashMap<Instruction>,
    options: &ProofOptions,
) -> Commitment {
    compute_precomputed_commitment_with(instructions, options, LeafLayout::RowPair)
}

/// [`compute_precomputed_commitment`] under an explicit trace-tree leaf layout
/// (S2). DECODE is program-dependent, so a one-row DECODE root is computed at
/// run time, like the row-pair one.
pub fn compute_precomputed_commitment_with(
    instructions: &U64HashMap<Instruction>,
    options: &ProofOptions,
    layout: LeafLayout,
) -> Commitment {
    compute_precomputed_commitment_under(
        instructions,
        options,
        layout,
        super::cpu::CpuRowForm::Full,
    )
}

/// [`compute_precomputed_commitment_with`] over `form`'s precomputed columns.
pub fn compute_precomputed_commitment_under(
    instructions: &U64HashMap<Instruction>,
    options: &ProofOptions,
    layout: LeafLayout,
    form: super::cpu::CpuRowForm,
) -> Commitment {
    let columns = preprocessed_columns_for(instructions, form);
    let num_rows = columns[0].len();

    // Interpolate each column to a polynomial
    let polys: Vec<Polynomial<FE>> = columns
        .iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for decode column")
        })
        .collect();

    // Evaluate polynomials on LDE domain (N * blowup_factor points)
    let blowup_factor = options.blowup_factor as usize;
    let coset_offset = FE::from(options.coset_offset);
    let lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, num_rows, &coset_offset)
                .expect("LDE evaluation failed for decode polynomial")
        })
        .collect();

    // ★ Through the LFM commit helper, which commits under the BLOCK PATH's pin
    // rather than `stark`'s default aliases. This root is a PREPROCESSED
    // commitment the prover recomputes and compares against, so building it with
    // a different hash than the path commits under fails at prove time with
    // `PrecomputedCommitmentMismatch` — which is exactly how it was found.
    crate::lfm::commit::commit_lde_columns_with(&lde_columns, layout)
}

/// DECODE's commitment source for both leaf layouts: the row-pair root
/// `supplied` by the caller (the recursion guest's) or computed on first use,
/// and the one-row root computed on first use (program-dependent: no static
/// twin; a supplied root never stands in for the other layout).
pub fn lazy_commitment(
    instructions: std::sync::Arc<U64HashMap<Instruction>>,
    options: &ProofOptions,
    supplied: Option<Commitment>,
) -> stark::lookup::LazyCommitment {
    let base = match supplied {
        Some(c) => stark::lookup::LazyCommitment::ready(c),
        None => {
            let (instructions, options) = (instructions.clone(), options.clone());
            stark::lookup::LazyCommitment::deferred(move || {
                compute_precomputed_commitment(&instructions, &options)
            })
        }
    };
    let options = options.clone();
    base.with_one_row(move || {
        Some(compute_precomputed_commitment_with(
            &instructions,
            &options,
            LeafLayout::Row,
        ))
    })
}

/// [`lazy_commitment`] over `form`'s precomputed columns. A supplied root is
/// [`CpuRowForm::Full`](super::cpu::CpuRowForm::Full)'s, so under the lean
/// form both roots are computed on first use.
pub fn lazy_commitment_for(
    instructions: std::sync::Arc<U64HashMap<Instruction>>,
    options: &ProofOptions,
    supplied: Option<Commitment>,
    form: super::cpu::CpuRowForm,
) -> stark::lookup::LazyCommitment {
    if !form.decode_bytes_apart() {
        return lazy_commitment(instructions, options, supplied);
    }
    let base = {
        let (instructions, options) = (instructions.clone(), options.clone());
        stark::lookup::LazyCommitment::deferred(move || {
            compute_precomputed_commitment_under(&instructions, &options, LeafLayout::RowPair, form)
        })
    };
    let options = options.clone();
    base.with_one_row(move || {
        Some(compute_precomputed_commitment_under(
            &instructions,
            &options,
            LeafLayout::Row,
            form,
        ))
    })
}

// =========================================================================
// ELF to Instructions (for verifier)
// =========================================================================

/// Extract instructions from an ELF without running the executor.
///
/// This is the minimal computation needed for verifier to compute
/// the DECODE commitment from the program.
pub fn instructions_from_elf(elf: &Elf) -> Result<U64HashMap<Instruction>, InstructionError> {
    let mut map = U64HashMap::default();
    for seg in elf.data.iter().filter(|s| s.is_executable) {
        for (i, &word) in seg.values.iter().enumerate() {
            let pc = seg.base_addr + (i as u64 * 4);
            map.insert(pc, Instruction::parse(word)?);
        }
    }
    Ok(map)
}

/// Compute DECODE commitment directly from an ELF.
///
/// Thin convenience wrapper around [`instructions_from_elf`] + [`compute_precomputed_commitment`].
/// Pure library function — no caching, always recomputes. Callers that need
/// caching, hardcoding, or a different policy should wrap this call at their
/// site (see [`compute_precomputed_commitment`] for the policy options).
///
/// This is what the verifier uses — no executor needed.
pub fn commitment_from_elf(
    elf: &Elf,
    options: &ProofOptions,
) -> Result<Commitment, InstructionError> {
    let instructions = instructions_from_elf(elf)?;
    Ok(compute_precomputed_commitment(&instructions, options))
}

// =========================================================================
// Combined ELF processing (DECODE only)
// =========================================================================

/// Result of ELF processing for DECODE table.
pub struct ElfTables {
    /// DECODE trace table
    pub decode: TraceTable<GoldilocksField, GoldilocksExtension>,
    /// PC to row mapping for DECODE multiplicities
    pub pc_to_row: PcToRow,
}

/// Process ELF to generate DECODE table from executable segments.
///
/// ## Returns
///
/// - `decode`: DECODE trace with all instructions from executable segments
/// - `pc_to_row`: Map from PC to row index for DECODE multiplicity updates
///
/// Table has multiplicities initialized to 0.
pub fn tables_from_elf(elf: &Elf) -> Result<ElfTables, InstructionError> {
    let mut decode_entries = Vec::new();
    let mut pc_to_row = PcToRow::default();
    pc_to_row.reserve(elf.data.iter().map(|s| s.values.len()).sum());

    // Process all ELF segments for DECODE (only executable segments)
    for segment in &elf.data {
        if segment.is_executable {
            for (i, &word) in segment.values.iter().enumerate() {
                let addr = segment.base_addr + (i as u64 * 4);
                let instruction = Instruction::parse(word)?;
                pc_to_row.insert(addr, decode_entries.len());
                decode_entries.push(DecodeEntry::from_instruction(addr, instruction, 4));
            }
        }
    }

    // Build DECODE table
    let decode = build_decode_table(decode_entries, &mut pc_to_row);

    Ok(ElfTables { decode, pc_to_row })
}

/// Build DECODE trace table from entries.
fn build_decode_table(
    entries: Vec<DecodeEntry>,
    pc_to_row: &mut PcToRow,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    // Add CPU padding entry
    let cpu_padding_row = entries.len();
    pc_to_row.insert(super::cpu::CPU_PADDING_PC, cpu_padding_row);
    let cpu_padding_entry = DecodeEntry {
        pc: super::cpu::CPU_PADDING_PC,
        ..Default::default()
    };

    // Pad to next power of 2, minimum 2
    let num_entries = entries.len() + 1;
    let num_rows = num_entries.next_power_of_two().max(2);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    // Fill actual entries
    for (row_idx, entry) in entries.iter().enumerate() {
        table.set_dword_wl(row_idx, cols::PC_0, entry.pc);
        table.set_u64(row_idx, cols::PACKED_DECODE, entry.packed_decode());
        table.set_dword_wl(row_idx, cols::IMM_0, entry.imm);
    }

    // Write CPU padding entry
    {
        table.set_dword_wl(cpu_padding_row, cols::PC_0, cpu_padding_entry.pc);
        table.set_u64(
            cpu_padding_row,
            cols::PACKED_DECODE,
            cpu_padding_entry.packed_decode(),
        );
        table.set_dword_wl(cpu_padding_row, cols::IMM_0, cpu_padding_entry.imm);
    }

    // Fill padding rows with DECODE padding pattern
    let padding_entry = DecodeEntry::padding_entry();
    for row_idx in num_entries..num_rows {
        table.set_dword_wl(row_idx, cols::PC_0, padding_entry.pc);
        table.set_u64(row_idx, cols::PACKED_DECODE, padding_entry.packed_decode());
        table.set_dword_wl(row_idx, cols::IMM_0, padding_entry.imm);
    }

    trace
}
