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
    // Build entries and PC-to-row mapping
    let mut pc_to_row = PcToRow::default();
    pc_to_row.reserve(instructions.len() + 1);
    let entries: Vec<_> = instructions
        .iter()
        .enumerate()
        .map(|(row_idx, (&pc, &instr))| {
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
    // Step 1: Generate trace (MU=0, we only need precomputed columns)
    let (trace, _pc_to_row) = generate_decode_trace(instructions);
    let num_rows = trace.num_rows();

    // Step 2: Extract precomputed columns (0..NUM_PRECOMPUTED_COLS)
    let columns: Vec<Vec<FE>> = (0..NUM_PRECOMPUTED_COLS)
        .map(|col_idx| {
            (0..num_rows)
                .map(|row_idx| *trace.main_table.get(row_idx, col_idx))
                .collect()
        })
        .collect();

    // Step 3: Interpolate each column to a polynomial
    let polys: Vec<Polynomial<FE>> = columns
        .iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for decode column")
        })
        .collect();

    // Step 4: Evaluate polynomials on LDE domain (N * blowup_factor points)
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
    // commitment the block path's verifier absorbs and the prover re-derives, so
    // the hash that BUILDS it must be the hash that path commits under — on a
    // branch pinning an algebraic hash, a root left on the alias is a BLAKE3
    // artifact inside an RPO proof, and the prover rejects it at commit time
    // with `PrecomputedCommitmentMismatch`.
    crate::lfm::commit::commit_lde_columns(&lde_columns)
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
