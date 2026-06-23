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
//! ## packed_decode Format (51 bits)
//!
//! ```text
//! Bits [0]:     read_register1
//! Bits [1]:     read_register2
//! Bits [2]:     write_register
//! Bits [3]:     memory_2bytes
//! Bits [4]:     memory_4bytes
//! Bits [5]:     memory_8bytes
//! Bits [6]:     c_type
//! Bits [7]:     signed
//! Bits [8]:     mp_selector
//! Bits [9]:     muldiv_selector
//! Bits [10]:    word_instr
//! Bits [11-26]: ALU flags (ADD, SUB, SLT, AND, OR, XOR, SHIFT, JALR,
//!               BEQ, BLT, LOAD, STORE, MUL, DIVREM, ECALL, EBREAK)
//! Bits [27:35]: rs1 (8 bits)
//! Bits [35:43]: rs2 (8 bits)
//! Bits [43:51]: rd (8 bits)
//! ```
//!
//! ## Bus Interactions
//!
//! - **Receiver**: DECODE bus - receives lookups from CPU table

use alloc::vec;
use alloc::vec::Vec;
use executor::elf::Elf;
use executor::vm::instruction::decoding::{Instruction, InstructionError};
use executor::vm::memory::U64HashMap;
use math::fft::bit_reversing::in_place_bit_reverse_permute;
use math::polynomial::Polynomial;
use smallvec::smallvec;
use stark::config::{BatchedMerkleTree, Commitment};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::{TraceTable, columns2rows};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

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

use hashbrown::HashMap;

/// Map from PC to row index in the DECODE trace table.
pub type PcToRow = HashMap<u64, usize>;

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
    let mut pc_to_row = HashMap::with_capacity(instructions.len());
    let entries: Vec<_> = instructions
        .iter()
        .enumerate()
        .map(|(row_idx, (&pc, &instr))| {
            pc_to_row.insert(pc, row_idx);
            DecodeEntry::from_instruction(pc, instr)
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
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    // Fill actual entries (MU = 0 initially)
    for (row_idx, entry) in entries.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;

        // PC as DWordWL
        data[base + cols::PC_0] = FE::from(entry.pc & 0xFFFF_FFFF);
        data[base + cols::PC_1] = FE::from(entry.pc >> 32);

        // packed_decode
        data[base + cols::PACKED_DECODE] = FE::from(entry.packed_decode());

        // imm as DWordWL
        data[base + cols::IMM_0] = FE::from(entry.imm & 0xFFFF_FFFF);
        data[base + cols::IMM_1] = FE::from(entry.imm >> 32);

        // MU = 0 (already zero from vec initialization)
    }

    // Write CPU padding entry (pc=1, all flags=0)
    {
        let base = cpu_padding_row * cols::NUM_COLUMNS;
        data[base + cols::PC_0] = FE::from(cpu_padding_entry.pc & 0xFFFF_FFFF);
        data[base + cols::PC_1] = FE::from(cpu_padding_entry.pc >> 32);
        data[base + cols::PACKED_DECODE] = FE::from(cpu_padding_entry.packed_decode());
        data[base + cols::IMM_0] = FE::from(cpu_padding_entry.imm & 0xFFFF_FFFF);
        data[base + cols::IMM_1] = FE::from(cpu_padding_entry.imm >> 32);
    }

    // Fill padding rows with DECODE padding pattern: pc=7, EBREAK=1
    let padding_entry = DecodeEntry::padding_entry();
    for row_idx in num_entries..num_rows {
        let base = row_idx * cols::NUM_COLUMNS;

        data[base + cols::PC_0] = FE::from(padding_entry.pc & 0xFFFF_FFFF);
        data[base + cols::PC_1] = FE::from(padding_entry.pc >> 32);
        data[base + cols::PACKED_DECODE] = FE::from(padding_entry.packed_decode());
        data[base + cols::IMM_0] = FE::from(padding_entry.imm & 0xFFFF_FFFF);
        data[base + cols::IMM_1] = FE::from(padding_entry.imm >> 32);
        // MU = 0 for padding rows (already zero from vec initialization)
    }

    (TraceTable::new_main(data, cols::NUM_COLUMNS, 1), pc_to_row)
}

/// Updates multiplicities in the DECODE trace table.
///
/// For each PC in `lookups`, increments the MU column in the corresponding row.
#[cfg(feature = "prove")]
pub fn update_multiplicities(
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    pc_to_row: &PcToRow,
    lookups: &[u64],
) {
    for &pc in lookups {
        if let Some(&row_idx) = pc_to_row.get(&pc) {
            let current = trace.main_table.get(row_idx, cols::MU);
            trace.main_table.set(row_idx, cols::MU, current + FE::one());
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
            smallvec![
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
/// Used by both prover (sanity check) and verifier (soundness check). The verifier
/// computes this from the program and checks that the proof's commitment matches.
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
    let mut lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, num_rows, &coset_offset)
                .expect("LDE evaluation failed for decode polynomial")
        })
        .collect();

    // Step 5: Bit-reverse permute (same as prover)
    for col in lde_columns.iter_mut() {
        in_place_bit_reverse_permute(col);
    }

    // Step 6: Convert columns to rows for Merkle tree
    let lde_rows = columns2rows(lde_columns);

    // Step 7: Build Merkle tree over LDE (N * blowup leaves)
    let tree = BatchedMerkleTree::<GoldilocksField>::build(&lde_rows)
        .expect("Failed to build Merkle tree for decode LDE");

    tree.root
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
/// This is what the verifier uses - no executor needed.
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
#[cfg(feature = "prove")]
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
#[cfg(feature = "prove")]
pub fn tables_from_elf(elf: &Elf) -> Result<ElfTables, InstructionError> {
    let mut decode_entries = Vec::new();
    let mut pc_to_row = HashMap::with_capacity(elf.data.iter().map(|s| s.values.len()).sum());

    // Process all ELF segments for DECODE (only executable segments)
    for segment in &elf.data {
        if segment.is_executable {
            for (i, &word) in segment.values.iter().enumerate() {
                let addr = segment.base_addr + (i as u64 * 4);
                let instruction = Instruction::parse(word)?;
                pc_to_row.insert(addr, decode_entries.len());
                decode_entries.push(DecodeEntry::from_instruction(addr, instruction));
            }
        }
    }

    // Build DECODE table
    let decode = build_decode_table(decode_entries, &mut pc_to_row);

    Ok(ElfTables { decode, pc_to_row })
}

/// Build DECODE trace table from entries.
#[cfg(feature = "prove")]
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
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    // Fill actual entries
    for (row_idx, entry) in entries.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;
        data[base + cols::PC_0] = FE::from(entry.pc & 0xFFFF_FFFF);
        data[base + cols::PC_1] = FE::from(entry.pc >> 32);
        data[base + cols::PACKED_DECODE] = FE::from(entry.packed_decode());
        data[base + cols::IMM_0] = FE::from(entry.imm & 0xFFFF_FFFF);
        data[base + cols::IMM_1] = FE::from(entry.imm >> 32);
    }

    // Write CPU padding entry
    {
        let base = cpu_padding_row * cols::NUM_COLUMNS;
        data[base + cols::PC_0] = FE::from(cpu_padding_entry.pc & 0xFFFF_FFFF);
        data[base + cols::PC_1] = FE::from(cpu_padding_entry.pc >> 32);
        data[base + cols::PACKED_DECODE] = FE::from(cpu_padding_entry.packed_decode());
        data[base + cols::IMM_0] = FE::from(cpu_padding_entry.imm & 0xFFFF_FFFF);
        data[base + cols::IMM_1] = FE::from(cpu_padding_entry.imm >> 32);
    }

    // Fill padding rows with DECODE padding pattern
    let padding_entry = DecodeEntry::padding_entry();
    for row_idx in num_entries..num_rows {
        let base = row_idx * cols::NUM_COLUMNS;
        data[base + cols::PC_0] = FE::from(padding_entry.pc & 0xFFFF_FFFF);
        data[base + cols::PC_1] = FE::from(padding_entry.pc >> 32);
        data[base + cols::PACKED_DECODE] = FE::from(padding_entry.packed_decode());
        data[base + cols::IMM_0] = FE::from(padding_entry.imm & 0xFFFF_FFFF);
        data[base + cols::IMM_1] = FE::from(padding_entry.imm >> 32);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "prove")]
    use executor::elf::Segment;

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
                .contains_key(&super::super::cpu::CPU_PADDING_PC)
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
                .contains_key(&super::super::cpu::CPU_PADDING_PC)
        );
    }
}
