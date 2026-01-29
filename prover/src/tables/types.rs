//! Core types for the 64-bit VM prover tables.
//!
//! This module defines the bus IDs, field types, and shared structures used across all 64-bit tables.
//!
//! ## Field Choice
//!
//! For the 64-bit VM prover, we use the Goldilocks field:
//! - Prime: p = 2^64 - 2^32 + 1
//! - Two-adicity: 32 (supports FFT up to 2^32 rows)
//! - Extension: Degree 3 (cubic extension with w³ = 2, provides 192-bit security)
//!
//! ## DecodeEntry
//!
//! The `DecodeEntry` struct represents decoded instruction information shared between
//! the CPU and DECODE tables. It contains all static decode-time information extracted
//! from an instruction, excluding runtime values like register contents.

use executor::vm::instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth};
use math::field::element::FieldElement;
use math::field::fields::fft_friendly::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::fields::fft_friendly::u64_goldilocks::GoldilocksField as GoldilocksBaseField;

/// Base field type: Goldilocks prime field (p = 2^64 - 2^32 + 1)
pub type GoldilocksField = GoldilocksBaseField;

/// Extension field type: Degree 3 extension of Goldilocks (w³ = 2)
pub type GoldilocksExtension = Degree3GoldilocksExtensionField;

/// Field element in the base Goldilocks field
pub type FE = FieldElement<GoldilocksField>;

/// Field element in the Goldilocks extension field
pub type FEE = FieldElement<GoldilocksExtension>;

/// Bus identifiers for LogUp interactions between tables.
///
/// Each bus connects senders (tables that produce values) with receivers
/// (tables that consume/verify those values). For the bus to balance,
/// the sum of sender multiplicities must equal the sum of receiver multiplicities.
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusId {
    // =========================================================================
    // Range checks (BITWISE table provides)
    // =========================================================================
    /// Range check: value is a valid byte [0, 256)
    IsByte = 0,
    /// Range check: value is a valid halfword [0, 2^16)
    IsHalfword,
    /// Range check: value is a 20-bit value [0, 2^20)
    IsB20,

    // =========================================================================
    // Bitwise operations (BITWISE table provides)
    // =========================================================================
    /// Bitwise AND of two bytes: AND_BYTE[X, Y] -> X & Y
    AndByte,
    /// Bitwise OR of two bytes: OR_BYTE[X, Y] -> X | Y
    OrByte,
    /// Bitwise XOR of two bytes: XOR_BYTE[X, Y] -> X ^ Y
    XorByte,
    /// Most significant bit of a byte: MSB8[X] -> (X >> 7) & 1
    Msb8,
    /// Most significant bit of a halfword: MSB16[X] -> (X >> 15) & 1
    Msb16,
    /// Check if value is zero: ZERO[X] -> X == 0 ? 1 : 0
    Zero,

    // =========================================================================
    // Shift helpers (BITWISE table provides)
    // =========================================================================
    /// Halfword shift left: HWSL[X, Z] -> (X << Z) & 0xFFFF
    Hwsl,
    /// Halfword shift left carry: HWSLC[X, Z] -> X >> (16 - Z)
    Hwslc,

    // =========================================================================
    // Arithmetic operations (separate tables)
    // =========================================================================
    /// Less-than comparison: LT[lhs, rhs, signed] -> lhs < rhs
    Lt,
    /// Multiplication: MUL[lhs, lhs_signed, rhs, rhs_signed, hi] -> product
    Mul,
    /// Shift operation: SHIFT[in, shift, dir, signed, word] -> out
    Shift,

    // =========================================================================
    // Memory/Control
    // =========================================================================
    /// Memory word read/write with timestamps (lookup bus from CPU)
    Memw,
    /// Memory load with sign/zero extension (lookup bus from CPU)
    Load,
    /// Internal memory consistency bus: memory[is_register, address, timestamp, value]
    /// Used for read/write pairing in MEMW table (M1-M8 in spec)
    Memory,
    /// Branch target computation
    Branch,

    // =========================================================================
    // System (specs not yet defined)
    // =========================================================================
    /// Instruction decode lookup
    Decode,
    /// System call handling
    Ecall,
}

impl From<BusId> for u64 {
    fn from(id: BusId) -> u64 {
        id as u64
    }
}

// =========================================================================
// Constants for 64-bit arithmetic
// =========================================================================

/// 2^16 for halfword combining
pub const SHIFT_16: u64 = 1 << 16;

/// 2^32 for word combining
pub const SHIFT_32: u64 = 1 << 32;

/// 2^(-32) mod p for carry extraction in 64-bit addition
/// In Babybear field: p = 2^31 - 2^27 + 1
/// We need to find x such that x * 2^32 ≡ 1 (mod p)
pub const INV_2_32: u64 = {
    // 2^32 mod p = 2^32 mod (2^31 - 2^27 + 1)
    // 2^32 = 2 * 2^31 = 2 * (p + 2^27 - 1) = 2p + 2^28 - 2 ≡ 2^28 - 2 (mod p)
    // We need the inverse of (2^28 - 2) mod p
    // For now, this is a placeholder - compute at runtime or verify
    1
};

/// 2^(-16) mod p for halfword operations
pub const INV_2_16: u64 = {
    // Placeholder - compute at runtime or verify
    1
};

// =========================================================================
// Column index helpers
// =========================================================================

/// Helper to define column ranges for different 64-bit representations.
pub mod columns {
    /// A DWordWL (64-bit as 2 words) spans 2 columns
    pub const DWORD_WL_SIZE: usize = 2;

    /// A DWordHL (64-bit as 4 halves) spans 4 columns
    pub const DWORD_HL_SIZE: usize = 4;

    /// A DWordBL (64-bit as 8 bytes) spans 8 columns
    pub const DWORD_BL_SIZE: usize = 8;

    /// A DWordHHW (64-bit as Word, Half, Half) spans 3 columns
    pub const DWORD_HHW_SIZE: usize = 3;

    /// A QuadWL (128-bit as 4 words) spans 4 columns
    pub const QUAD_WL_SIZE: usize = 4;
}

// =========================================================================
// DecodeEntry - Shared decode information for CPU and DECODE tables
// =========================================================================

/// A single decoded instruction entry.
///
/// This struct contains all static decode-time information extracted from an instruction.
/// It is shared between the CPU table (which uses it for execution) and the DECODE table
/// (which provides it as a lookup table).
///
/// ## Usage
///
/// - **CPU table**: `CpuOperation` contains a `DecodeEntry` plus runtime values (rv1, rv2, etc.)
/// - **DECODE table**: Stores `DecodeEntry` directly, with multiplicity tracking
///
/// ## packed_decode Format (51 bits)
///
/// ```text
/// Bits [0]:     read_register1
/// Bits [1]:     read_register2
/// Bits [2]:     write_register
/// Bits [3]:     memory_2bytes
/// Bits [4]:     memory_4bytes
/// Bits [5]:     memory_8bytes
/// Bits [6]:     c_type
/// Bits [7]:     signed
/// Bits [8]:     mp_selector
/// Bits [9]:     muldiv_selector
/// Bits [10]:    word_instr
/// Bits [11-26]: ALU flags (ADD, SUB, SLT, AND, OR, XOR, SHIFT, JALR,
///               BEQ, BLT, LOAD, STORE, MUL, DIVREM, ECALL, EBREAK)
/// Bits [27:35]: rs1 (8 bits)
/// Bits [35:43]: rs2 (8 bits)
/// Bits [43:51]: rd (8 bits)
/// ```
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

    /// Packs all flags and register indices into a single 51-bit value.
    ///
    /// This matches the spec's packed_decode format (decode.md).
    ///
    /// Note: The register flags (read_register1, read_register2, write_register)
    /// are adjusted to exclude x0 (hardwired zero) and x255 (virtual PC for AUIPC/JAL).
    /// This matches the CPU trace columns and ensures the DECODE bus balances.
    pub fn packed_decode(&self) -> u64 {
        let mut packed: u64 = 0;

        // Control flags (bits 0-10)
        // Note: Register flags exclude x0 and x255 (virtual PC) to match CPU trace
        let read_reg1_physical = self.read_register1 && self.rs1 != 0 && self.rs1 != 255;
        let read_reg2_physical = self.read_register2 && self.rs2 != 0;
        let write_reg_physical = self.write_register && self.rd != 0;
        packed |= read_reg1_physical as u64;
        packed |= (read_reg2_physical as u64) << 1;
        packed |= (write_reg_physical as u64) << 2;
        packed |= (self.memory_2bytes as u64) << 3;
        packed |= (self.memory_4bytes as u64) << 4;
        packed |= (self.memory_8bytes as u64) << 5;
        packed |= (self.c_type as u64) << 6;
        packed |= (self.signed as u64) << 7;
        packed |= (self.mp_selector as u64) << 8;
        packed |= (self.muldiv_selector as u64) << 9;
        packed |= (self.word_instr as u64) << 10;

        // ALU flags (bits 11-26)
        packed |= (self.op_add as u64) << 11;
        packed |= (self.op_sub as u64) << 12;
        packed |= (self.op_slt as u64) << 13;
        packed |= (self.op_and as u64) << 14;
        packed |= (self.op_or as u64) << 15;
        packed |= (self.op_xor as u64) << 16;
        packed |= (self.op_shift as u64) << 17;
        packed |= (self.op_jalr as u64) << 18;
        packed |= (self.op_beq as u64) << 19;
        packed |= (self.op_blt as u64) << 20;
        packed |= (self.op_load as u64) << 21;
        packed |= (self.op_store as u64) << 22;
        packed |= (self.op_mul as u64) << 23;
        packed |= (self.op_divrem as u64) << 24;
        packed |= (self.op_ecall as u64) << 25;
        packed |= (self.op_ebreak as u64) << 26;

        // Register indices (bits 27-50)
        packed |= (self.rs1 as u64) << 27;
        packed |= (self.rs2 as u64) << 35;
        packed |= (self.rd as u64) << 43;

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
                // Per spec: JAL is represented as JALR rd, x255, imm
                // x255 is the virtual register holding PC
                entry.rs1 = 255;
                entry.read_register1 = true; // rs1 ≠ 0
                entry.imm = offset as i64 as u64;
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
                // Per spec: AUIPC is represented as ADDI rd, x255, imm
                // x255 is the virtual register holding PC
                entry.rs1 = 255;
                entry.read_register1 = true; // rs1 ≠ 0
                // AUIPC immediate is sign-extended to 64 bits
                entry.imm = (imm as i32) as i64 as u64;
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

    /// Helper to set memory width flags (exclusive encoding per spec).
    ///
    /// Memory width uses exclusive flags ("exactly N bytes"):
    /// - 1 byte: no flags
    /// - 2 bytes: memory_2bytes = true
    /// - 4 bytes: memory_4bytes = true
    /// - 8 bytes: memory_8bytes = true
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
