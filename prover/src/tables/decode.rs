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

use executor::vm::instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth};
use executor::vm::memory::U64HashMap;
use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::polynomial::Polynomial;
use stark::config::{BatchedMerkleTree, Commitment};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::{TraceTable, columns2rows};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

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
// DecodeEntry (uncompressed representation)
// =========================================================================

/// A single decoded instruction entry.
///
/// This is the uncompressed representation used internally for easy manipulation.
/// It gets packed into the compressed trace format during trace generation.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Default)]
pub struct DecodeEntry {
    // Program counter
    /// Program counter (64-bit)
    pub pc: u64,

    // Register indices (8 bits each)
    /// Source register 1 index
    pub rs1: u8,
    /// Source register 2 index
    pub rs2: u8,
    /// Destination register index
    pub rd: u8,

    // Control flags
    /// Whether to read from rs1
    pub read_register1: bool,
    /// Whether to read from rs2
    pub read_register2: bool,
    /// Whether to write to rd
    pub write_register: bool,
    /// Memory access is 2 bytes
    pub memory_2bytes: bool,
    /// Memory access is 4 bytes
    pub memory_4bytes: bool,
    /// Memory access is 8 bytes
    pub memory_8bytes: bool,
    /// Compressed instruction (2 bytes instead of 4)
    pub c_type: bool,
    /// Signed operation
    pub signed: bool,
    /// Multi-purpose selector (shift direction, branch invert, etc.)
    pub mp_selector: bool,
    /// MUL/DIV output selector
    pub muldiv_selector: bool,
    /// Word instruction (32-bit with sign extension)
    pub word_instr: bool,

    // ALU selector flags (one-hot)
    /// ADD operation
    pub op_add: bool,
    /// SUB operation
    pub op_sub: bool,
    /// SLT (Set Less Than) operation
    pub op_slt: bool,
    /// AND operation
    pub op_and: bool,
    /// OR operation
    pub op_or: bool,
    /// XOR operation
    pub op_xor: bool,
    /// SHIFT operation
    pub op_shift: bool,
    /// JALR operation
    pub op_jalr: bool,
    /// BEQ (Branch if Equal) operation
    pub op_beq: bool,
    /// BLT (Branch if Less Than) operation
    pub op_blt: bool,
    /// LOAD operation
    pub op_load: bool,
    /// STORE operation
    pub op_store: bool,
    /// MUL operation
    pub op_mul: bool,
    /// DIVREM operation
    pub op_divrem: bool,
    /// ECALL operation
    pub op_ecall: bool,
    /// EBREAK operation
    pub op_ebreak: bool,

    // Immediate value
    /// Fully extended 64-bit immediate
    pub imm: u64,
}

impl DecodeEntry {
    /// Creates a new empty DecodeEntry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates the special padding entry for DECODE table.
    ///
    /// Uses pc=7 with EBREAK=1 flag set. This makes padding rows
    /// unprovable since CPU asserts EBREAK=0.
    pub fn padding_entry() -> Self {
        Self {
            pc: 7,
            op_ebreak: true,
            ..Default::default()
        }
    }

    /// Packs all flags and register indices into a single 49-bit value.
    ///
    /// This matches the spec's packed_decode format (cpu.toml):
    /// - bit 0:      write_register
    /// - bit 1:      memory_2bytes
    /// - bit 2:      memory_4bytes
    /// - bit 3:      memory_8bytes
    /// - bit 4:      c_type
    /// - bit 5:      signed
    /// - bit 6:      mp_selector
    /// - bit 7:      muldiv_selector
    /// - bit 8:      word_instr
    /// - bit 9:      ADD
    /// - bit 10:     SUB
    /// - bit 11:     SLT
    /// - bit 12:     AND
    /// - bit 13:     OR
    /// - bit 14:     XOR
    /// - bit 15:     SHIFT
    /// - bit 16:     JALR
    /// - bit 17:     BEQ
    /// - bit 18:     BLT
    /// - bit 19:     LOAD
    /// - bit 20:     STORE
    /// - bit 21:     MUL
    /// - bit 22:     DIVREM
    /// - bit 23:     ECALL
    /// - bit 24:     EBREAK
    /// - bits 25-32: rs1 (8 bits)
    /// - bits 33-40: rs2 (8 bits)
    /// - bits 41-48: rd (8 bits)
    pub fn packed_decode(&self) -> u64 {
        let mut packed: u64 = 0;

        // Control flags (bits 0-8)
        packed |= self.write_register as u64;
        packed |= (self.memory_2bytes as u64) << 1;
        packed |= (self.memory_4bytes as u64) << 2;
        packed |= (self.memory_8bytes as u64) << 3;
        packed |= (self.c_type as u64) << 4;
        packed |= (self.signed as u64) << 5;
        packed |= (self.mp_selector as u64) << 6;
        packed |= (self.muldiv_selector as u64) << 7;
        packed |= (self.word_instr as u64) << 8;

        // ALU flags (bits 9-24)
        packed |= (self.op_add as u64) << 9;
        packed |= (self.op_sub as u64) << 10;
        packed |= (self.op_slt as u64) << 11;
        packed |= (self.op_and as u64) << 12;
        packed |= (self.op_or as u64) << 13;
        packed |= (self.op_xor as u64) << 14;
        packed |= (self.op_shift as u64) << 15;
        packed |= (self.op_jalr as u64) << 16;
        packed |= (self.op_beq as u64) << 17;
        packed |= (self.op_blt as u64) << 18;
        packed |= (self.op_load as u64) << 19;
        packed |= (self.op_store as u64) << 20;
        packed |= (self.op_mul as u64) << 21;
        packed |= (self.op_divrem as u64) << 22;
        packed |= (self.op_ecall as u64) << 23;
        packed |= (self.op_ebreak as u64) << 24;

        // Register indices (bits 25-48)
        packed |= (self.rs1 as u64) << 25;
        packed |= (self.rs2 as u64) << 33;
        packed |= (self.rd as u64) << 41;

        packed
    }

    /// Creates a DecodeEntry from a PC and Instruction.
    ///
    /// Extracts all decode-time information: pc, registers, flags, immediate.
    pub fn from_instruction(pc: u64, instruction: Instruction) -> Self {
        let mut entry = Self {
            pc,
            ..Default::default()
        };

        match instruction {
            Instruction::Arith {
                dst,
                src1,
                src2,
                op,
            } => {
                entry.rd = dst as u8;
                entry.rs1 = src1 as u8;
                entry.rs2 = src2 as u8;
                entry.read_register1 = src1 != 0;
                entry.read_register2 = src2 != 0;
                if dst != 0 {
                    entry.write_register = true;
                }
                Self::set_arith_op(&mut entry, op, false);
            }

            Instruction::ArithImm { dst, src, imm, op } => {
                entry.rd = dst as u8;
                entry.rs1 = src as u8;
                entry.rs2 = 0;
                entry.imm = imm as i64 as u64; // Sign extend
                entry.read_register1 = src != 0;
                if dst != 0 {
                    entry.write_register = true;
                }
                Self::set_arith_op(&mut entry, op, false);
            }

            Instruction::ArithW {
                dst,
                src1,
                src2,
                op,
            } => {
                entry.rd = dst as u8;
                entry.rs1 = src1 as u8;
                entry.rs2 = src2 as u8;
                entry.word_instr = true;
                entry.read_register1 = src1 != 0;
                entry.read_register2 = src2 != 0;
                if dst != 0 {
                    entry.write_register = true;
                }
                Self::set_arith_op(&mut entry, op, true);
            }

            Instruction::ArithImmW { dst, src, imm, op } => {
                entry.rd = dst as u8;
                entry.rs1 = src as u8;
                entry.rs2 = 0;
                entry.imm = imm as i64 as u64; // Sign extend
                entry.word_instr = true;
                entry.read_register1 = src != 0;
                if dst != 0 {
                    entry.write_register = true;
                }
                Self::set_arith_op(&mut entry, op, true);
            }

            Instruction::JumpAndLink { dst, offset } => {
                entry.op_jalr = true;
                entry.rd = dst as u8;
                entry.rs1 = 255; // x255 holds PC for JAL
                entry.imm = offset as i64 as u64;
                entry.read_register1 = true;
                if dst != 0 {
                    entry.write_register = true;
                }
            }

            Instruction::JumpAndLinkRegister { base, dst, offset } => {
                entry.op_jalr = true;
                entry.rd = dst as u8;
                entry.rs1 = base as u8;
                entry.imm = offset as i64 as u64;
                entry.read_register1 = base != 0;
                if dst != 0 {
                    entry.write_register = true;
                }
            }

            Instruction::Store {
                src,
                offset,
                base,
                width,
            } => {
                entry.op_store = true;
                entry.rs1 = base as u8;
                entry.rs2 = src as u8;
                entry.imm = offset as i64 as u64;
                entry.read_register1 = base != 0;
                entry.read_register2 = src != 0;
                // write_register = false for STORE
                Self::set_memory_width(&mut entry, width);
            }

            Instruction::Load {
                dst,
                offset,
                base,
                width,
            } => {
                entry.op_load = true;
                entry.rd = dst as u8;
                entry.rs1 = base as u8;
                entry.imm = offset as i64 as u64;
                entry.read_register1 = base != 0;
                if dst != 0 {
                    entry.write_register = true;
                }
                Self::set_memory_width(&mut entry, width);
                // Set signed flag for sign-extending loads
                match width {
                    LoadStoreWidth::Byte | LoadStoreWidth::Half | LoadStoreWidth::Word => {
                        entry.signed = true;
                    }
                    _ => {}
                }
            }

            Instruction::Branch {
                src1,
                src2,
                cond,
                offset,
            } => {
                entry.rs1 = src1 as u8;
                entry.rs2 = src2 as u8;
                entry.imm = offset as i64 as u64;
                entry.read_register1 = src1 != 0;
                entry.read_register2 = src2 != 0;

                match cond {
                    Comparison::Equal => {
                        entry.op_beq = true;
                    }
                    Comparison::NotEqual => {
                        entry.op_beq = true;
                        entry.mp_selector = true; // Inverted
                    }
                    Comparison::LessThan => {
                        entry.op_blt = true;
                        entry.signed = true;
                    }
                    Comparison::LessThanUnsigned => {
                        entry.op_blt = true;
                    }
                    Comparison::GreaterOrEqual => {
                        entry.op_blt = true;
                        entry.signed = true;
                        entry.mp_selector = true; // Inverted
                    }
                    Comparison::GreaterOrEqualUnsigned => {
                        entry.op_blt = true;
                        entry.mp_selector = true; // Inverted
                    }
                }
            }

            Instruction::LoadUpperImm { dst, imm } => {
                entry.op_add = true;
                entry.rd = dst as u8;
                entry.rs1 = 0;
                entry.rs2 = 0;
                // LUI immediate is sign-extended to 64 bits
                entry.imm = (imm as i32) as i64 as u64;
                if dst != 0 {
                    entry.write_register = true;
                }
            }

            Instruction::AddUpperImmToPc { dst, imm } => {
                entry.op_add = true;
                entry.rd = dst as u8;
                entry.rs1 = 255; // x255 holds PC for AUIPC
                // AUIPC immediate is sign-extended to 64 bits
                entry.imm = (imm as i32) as i64 as u64;
                entry.read_register1 = true;
                if dst != 0 {
                    entry.write_register = true;
                }
            }

            Instruction::CSR { .. } => {
                // CSR instructions not yet supported in prover
            }

            Instruction::EcallEbreak => {
                // Determine if ECALL or EBREAK based on context
                // For now, default to ECALL
                entry.op_ecall = true;
                // ECALL uses: rs1=x17 (syscall number), rs2=x10 (arg), rd=x10 (result)
                entry.rs1 = 17;
                entry.rs2 = 10;
                entry.rd = 10;
                entry.read_register1 = true;
                entry.read_register2 = true;
                entry.write_register = true;
            }

            Instruction::Fence => {
                // FENCE is a memory barrier - in single-threaded, in-order execution it's a no-op
                // No operation flags needed, just advance PC (handled by default)
            }
        }

        entry
    }

    /// Helper to set ALU operation flags based on ArithOp.
    fn set_arith_op(entry: &mut Self, arith_op: ArithOp, is_word: bool) {
        match arith_op {
            ArithOp::Add => {
                entry.op_add = true;
            }
            ArithOp::Sub => {
                entry.op_sub = true;
            }
            ArithOp::Xor => entry.op_xor = true,
            ArithOp::Or => entry.op_or = true,
            ArithOp::And => entry.op_and = true,
            ArithOp::ShiftLeftLogical => {
                entry.op_shift = true;
                // mp_selector = 0 for left shift
            }
            ArithOp::ShiftRightLogical => {
                entry.op_shift = true;
                entry.mp_selector = true; // Right shift
            }
            ArithOp::ShiftRightArith => {
                entry.op_shift = true;
                entry.mp_selector = true;
                entry.signed = true;
            }
            ArithOp::SetLessThan => {
                entry.op_slt = true;
                entry.signed = true;
            }
            ArithOp::SetLessThanU => {
                entry.op_slt = true;
            }
            ArithOp::Mul => {
                entry.op_mul = true;
                entry.mp_selector = true;
                if !is_word {
                    entry.signed = true;
                }
            }
            ArithOp::MulHigh => {
                entry.op_mul = true;
                entry.muldiv_selector = true;
                entry.signed = true;
            }
            ArithOp::MulHighSignedUnsigned => {
                entry.op_mul = true;
                entry.muldiv_selector = true;
                entry.mp_selector = true;
                entry.signed = true;
            }
            ArithOp::MulHighUnsigned => {
                entry.op_mul = true;
                entry.muldiv_selector = true;
            }
            ArithOp::Div => {
                entry.op_divrem = true;
                entry.signed = true;
            }
            ArithOp::DivUnsigned => {
                entry.op_divrem = true;
            }
            ArithOp::Remainder => {
                entry.op_divrem = true;
                entry.muldiv_selector = true;
                entry.signed = true;
            }
            ArithOp::RemainderUnsigned => {
                entry.op_divrem = true;
                entry.muldiv_selector = true;
            }
        }
    }

    /// Helper to set memory width flags.
    fn set_memory_width(entry: &mut Self, width: LoadStoreWidth) {
        match width {
            LoadStoreWidth::Byte | LoadStoreWidth::ByteUnsigned => {
                // 1 byte - no flags set
            }
            LoadStoreWidth::Half | LoadStoreWidth::HalfUnsigned => {
                entry.memory_2bytes = true;
            }
            LoadStoreWidth::Word | LoadStoreWidth::WordUnsigned => {
                entry.memory_4bytes = true;
            }
            LoadStoreWidth::DoubleWord => {
                entry.memory_8bytes = true;
            }
        }
    }
}

// =========================================================================
// Trace generation
// =========================================================================

use std::collections::HashMap;

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

    // Pad to next power of 2, minimum 2
    let num_rows = entries.len().next_power_of_two().max(2);
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

    // Fill padding rows with DECODE padding pattern: pc=7, EBREAK=1
    let padding_entry = DecodeEntry::padding_entry();
    for row_idx in entries.len()..num_rows {
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
