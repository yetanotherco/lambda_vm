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
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField as GoldilocksBaseField;

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
    /// `ARE_BYTES[X, Y]`: range check that both X and Y are valid bytes [0, 256).
    /// Single-byte checks (spec template `IS_BYTE<X>`) send the second value as 0.
    AreBytes = 0,
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
    /// Halfword shift left: HWSL[X, Z] -> [(X << Z) & 0xFFFF, X >> (16 - Z)]
    Hwsl,

    // =========================================================================
    // Arithmetic operations (separate tables)
    // =========================================================================
    /// Less-than comparison: LT[lhs, rhs, signed] -> lhs < rhs
    Lt,
    /// Multiplication: MUL[lhs, lhs_signed, rhs, rhs_signed, hi] -> product
    Mul,
    /// Division/Remainder: DVRM[result; n, d, signed, muldiv_selector]
    Dvrm,
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
    /// System call handling (CPU → HALT/COMMIT for all ECALLs)
    Ecall,
    /// COMMIT self-referencing recursive bus (row N → row N+1)
    CommitNextByte,
    /// COMMIT output bus: verifier computes the receiver contribution externally
    /// from `VmProof.public_output` using the shared LogUp challenges
    Commit,
    /// Keccak core ↔ round chip: (timestamp, round, state[200 bytes])
    Keccak,
    /// Keccak round ↔ RC lookup: (round, rc[8 bytes])
    KeccakRc,
}

impl BusId {
    /// Human-readable name for debug output.
    pub fn name(&self) -> &'static str {
        match self {
            BusId::AreBytes => "AreBytes",
            BusId::IsHalfword => "IsHalfword",
            BusId::IsB20 => "IsB20",
            BusId::AndByte => "AndByte",
            BusId::OrByte => "OrByte",
            BusId::XorByte => "XorByte",
            BusId::Msb8 => "Msb8",
            BusId::Msb16 => "Msb16",
            BusId::Zero => "Zero",
            BusId::Hwsl => "Hwsl",
            BusId::Lt => "Lt",
            BusId::Mul => "Mul",
            BusId::Shift => "Shift",
            BusId::Memw => "Memw",
            BusId::Load => "Load",
            BusId::Memory => "Memory",
            BusId::Branch => "Branch",
            BusId::Decode => "Decode",
            BusId::Ecall => "Ecall",
            BusId::Dvrm => "Dvrm",
            BusId::CommitNextByte => "CommitNextByte",
            BusId::Commit => "Commit",
            BusId::Keccak => "Keccak",
            BusId::KeccakRc => "KeccakRc",
        }
    }
}

impl TryFrom<u64> for BusId {
    type Error = u64;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(BusId::AreBytes),
            1 => Ok(BusId::IsHalfword),
            2 => Ok(BusId::IsB20),
            3 => Ok(BusId::AndByte),
            4 => Ok(BusId::OrByte),
            5 => Ok(BusId::XorByte),
            6 => Ok(BusId::Msb8),
            7 => Ok(BusId::Msb16),
            8 => Ok(BusId::Zero),
            9 => Ok(BusId::Hwsl),
            10 => Ok(BusId::Lt),
            11 => Ok(BusId::Mul),
            12 => Ok(BusId::Dvrm),
            13 => Ok(BusId::Shift),
            14 => Ok(BusId::Memw),
            15 => Ok(BusId::Load),
            16 => Ok(BusId::Memory),
            17 => Ok(BusId::Branch),
            18 => Ok(BusId::Decode),
            19 => Ok(BusId::Ecall),
            20 => Ok(BusId::CommitNextByte),
            21 => Ok(BusId::Commit),
            22 => Ok(BusId::Keccak),
            23 => Ok(BusId::KeccakRc),
            other => Err(other),
        }
    }
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

// Field inverses for Goldilocks prime p = 2^64 - 2^32 + 1
// Used for virtual carry computation in MUL table
//
// Only the constants actually used in mul.rs are defined here:
// - Positive: INV_2_32, INV_2_64, INV_2_96, INV_2_128 (for raw_product terms)
// - Negative: NEG_INV_2_* (for lo/hi terms which are subtracted)

/// 2^(-32) mod p
pub const INV_2_32: u64 = 18446744065119617026;
/// 2^(-64) mod p
pub const INV_2_64: u64 = 18446744065119617025;
/// 2^(-96) mod p
pub const INV_2_96: u64 = 18446744069414584320;
/// 2^(-128) mod p
pub const INV_2_128: u64 = 4294967295;

/// -(2^-16) mod p = p - 2^(-16)
pub const NEG_INV_2_16: u64 = 281474976645120;
/// -(2^-32) mod p = p - 2^(-32)
pub const NEG_INV_2_32: u64 = 4294967295;
/// -(2^-48) mod p = p - 2^(-48)
pub const NEG_INV_2_48: u64 = 281474976710656;
/// -(2^-64) mod p = p - 2^(-64)
pub const NEG_INV_2_64: u64 = 4294967296;
/// -(2^-80) mod p = p - 2^(-80)
pub const NEG_INV_2_80: u64 = 65536;
/// -(2^-96) mod p = p - 2^(-96)
pub const NEG_INV_2_96: u64 = 1;
/// -(2^-112) mod p = p - 2^(-112)
pub const NEG_INV_2_112: u64 = 18446462594437939201;
/// -(2^-128) mod p = p - 2^(-128)
pub const NEG_INV_2_128: u64 = 18446744065119617026;

// =========================================================================
// packed_decode bit positions (shared between CPU and DECODE tables)
// =========================================================================

/// Bit positions for the packed_decode field.
///
/// This is the single source of truth for how decode fields are packed into
/// a 51-bit value. Used by:
/// - `DecodeEntry::packed_decode()` - packs fields into a u64
/// - CPU table bus interaction - builds LinearTerm coefficients
///
/// ## Format (51 bits total)
///
/// ```text
/// Bits [0-10]:  Control flags (read_reg1, read_reg2, write_reg, memory_*, etc.)
/// Bits [11-26]: ALU operation flags (ADD, SUB, SLT, AND, OR, XOR, etc.)
/// Bits [27-34]: rs1 register index (8 bits)
/// Bits [35-42]: rs2 register index (8 bits)
/// Bits [43-50]: rd register index (8 bits)
/// ```
pub mod packed_decode {
    // Control flags (bits 0-10)
    pub const READ_REG1: u32 = 0;
    pub const READ_REG2: u32 = 1;
    pub const WRITE_REG: u32 = 2;
    pub const MEMORY_2BYTES: u32 = 3;
    pub const MEMORY_4BYTES: u32 = 4;
    pub const MEMORY_8BYTES: u32 = 5;
    pub const C_TYPE: u32 = 6;
    pub const SIGNED: u32 = 7;
    pub const MP_SELECTOR: u32 = 8;
    pub const MULDIV_SELECTOR: u32 = 9;
    pub const WORD_INSTR: u32 = 10;

    // ALU operation flags (bits 11-26)
    pub const OP_ADD: u32 = 11;
    pub const OP_SUB: u32 = 12;
    pub const OP_SLT: u32 = 13;
    pub const OP_AND: u32 = 14;
    pub const OP_OR: u32 = 15;
    pub const OP_XOR: u32 = 16;
    pub const OP_SHIFT: u32 = 17;
    pub const OP_JALR: u32 = 18;
    pub const OP_BEQ: u32 = 19;
    pub const OP_BLT: u32 = 20;
    pub const OP_LOAD: u32 = 21;
    pub const OP_STORE: u32 = 22;
    pub const OP_MUL: u32 = 23;
    pub const OP_DIVREM: u32 = 24;
    pub const OP_ECALL: u32 = 25;
    pub const OP_EBREAK: u32 = 26;

    // Register indices (bits 27-50)
    pub const RS1: u32 = 27;
    pub const RS2: u32 = 35;
    pub const RD: u32 = 43;
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
    /// Bit positions are defined in the `packed_decode` module.
    ///
    /// Note: The register flags (read_register1, read_register2, write_register)
    /// are adjusted to exclude x0 (hardwired zero) and x255 (virtual PC for AUIPC/JAL).
    /// This matches the CPU trace columns and ensures the DECODE bus balances.
    pub fn packed_decode(&self) -> u64 {
        use crate::tables::types::packed_decode as bits;

        let mut packed: u64 = 0;

        // Control flags (bits 0-10)
        // x0 is hardwired to zero and never physically read.
        // x255 is the register where the pc is stored (per spec decode.md),
        // so read_register1=1 for rs1=255.
        let read_reg1_physical = self.read_register1 && self.rs1 != 0;
        let read_reg2_physical = self.read_register2 && self.rs2 != 0;
        let write_reg_physical = self.write_register && self.rd != 0;
        packed |= (read_reg1_physical as u64) << bits::READ_REG1;
        packed |= (read_reg2_physical as u64) << bits::READ_REG2;
        packed |= (write_reg_physical as u64) << bits::WRITE_REG;
        packed |= (self.memory_2bytes as u64) << bits::MEMORY_2BYTES;
        packed |= (self.memory_4bytes as u64) << bits::MEMORY_4BYTES;
        packed |= (self.memory_8bytes as u64) << bits::MEMORY_8BYTES;
        packed |= (self.c_type as u64) << bits::C_TYPE;
        packed |= (self.signed as u64) << bits::SIGNED;
        packed |= (self.mp_selector as u64) << bits::MP_SELECTOR;
        packed |= (self.muldiv_selector as u64) << bits::MULDIV_SELECTOR;
        packed |= (self.word_instr as u64) << bits::WORD_INSTR;

        // ALU flags (bits 11-26)
        packed |= (self.op_add as u64) << bits::OP_ADD;
        packed |= (self.op_sub as u64) << bits::OP_SUB;
        packed |= (self.op_slt as u64) << bits::OP_SLT;
        packed |= (self.op_and as u64) << bits::OP_AND;
        packed |= (self.op_or as u64) << bits::OP_OR;
        packed |= (self.op_xor as u64) << bits::OP_XOR;
        packed |= (self.op_shift as u64) << bits::OP_SHIFT;
        packed |= (self.op_jalr as u64) << bits::OP_JALR;
        packed |= (self.op_beq as u64) << bits::OP_BEQ;
        packed |= (self.op_blt as u64) << bits::OP_BLT;
        packed |= (self.op_load as u64) << bits::OP_LOAD;
        packed |= (self.op_store as u64) << bits::OP_STORE;
        packed |= (self.op_mul as u64) << bits::OP_MUL;
        packed |= (self.op_divrem as u64) << bits::OP_DIVREM;
        packed |= (self.op_ecall as u64) << bits::OP_ECALL;
        packed |= (self.op_ebreak as u64) << bits::OP_EBREAK;

        // Register indices (bits 27-50)
        packed |= (self.rs1 as u64) << bits::RS1;
        packed |= (self.rs2 as u64) << bits::RS2;
        packed |= (self.rd as u64) << bits::RD;

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
                Self::set_arith_op(&mut entry, op);
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
                Self::set_arith_op(&mut entry, op);
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
                Self::set_arith_op(&mut entry, op);
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
                Self::set_arith_op(&mut entry, op);
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
                // CSR instructions are executed as no-ops by the VM (see
                // executor Instruction::CSR arm returning dst_val: 0,
                // src1/2_val: 0). Mirror that here by treating them as
                // `ADDI x0, x0, 0` — same pattern as `Fence`. This sets
                // `op_add=true` so CM54's multiplicity is non-zero and the
                // CPU's PC-update Memw sender fires.
                entry.op_add = true;
            }

            Instruction::EcallEbreak => {
                entry.op_ecall = true;
                entry.rs1 = 17; // a7 (syscall number)
                entry.read_register1 = true; // M1 reads a7 → rv1 = syscall number
                // rs2 and rd default to 0 per spec; read_register2 and write_register remain false.
                // HALT/COMMIT chips access registers via direct MEMW interactions.
            }

            Instruction::Fence => {
                // Per spec, FENCE is a no-op interpreted as ADDI x0, x0, 0.
                entry.op_add = true;
            }
        }

        entry
    }

    /// Helper to set ALU operation flags based on ArithOp.
    fn set_arith_op(entry: &mut Self, arith_op: ArithOp) {
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
                entry.signed = true;
            }
            ArithOp::MulHigh => {
                entry.op_mul = true;
                entry.muldiv_selector = true;
                entry.mp_selector = true; // both operands signed for MULH
                entry.signed = true;
            }
            ArithOp::MulHighSignedUnsigned => {
                entry.op_mul = true;
                entry.muldiv_selector = true;
                // mp_selector = false (default): rhs is unsigned for MULHSU
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
