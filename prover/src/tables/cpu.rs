//! CPU table for the 64-bit VM.
//!
//! The CPU table is the central execution table that:
//! - Fetches instructions via DECODE interaction
//! - Dispatches ALU operations to specialized tables (ADD, SUB, LT, BITWISE, SHIFT, MUL, DIVREM)
//! - Handles memory operations (LOAD, STORE, register read/write)
//! - Computes branch conditions and next_pc
//!
//! ## Column Layout
//!
//! ### Input (from DECODE)
//! - `timestamp`: Timestamp (1 col)
//! - `pc`: DWordWL (2 cols) - program counter
//! - `rs1`, `rs2`, `rd`: Byte (3 cols) - register indices
//! - Flags: `write_register`, `memory_2bytes`, `memory_4bytes`, `memory_8bytes`,
//!   `c_type_instruction`, `signed`, `mp_selector`, `muldiv_selector`, `word_instr`
//! - `imm`: DWordWL (2 cols) - fully extended immediate
//! - ALU selectors: `ADD`, `SUB`, `SLT`, `AND`, `OR`, `XOR`, `SHIFT`, `JALR`,
//!   `BEQ`, `BLT`, `LOAD`, `STORE`, `MUL`, `DIVREM`, `ECALL`, `EBREAK`
//!
//! ### Output
//! - `next_pc`: DWordWL (2 cols)
//! - `rvd`: DWordWL (2 cols) - value to write to destination register
//!
//! ### Auxiliary
//! - `rv1`: DWordWHH (3 cols) - value of register rs1
//! - `rv2`: DWordWHH (3 cols) - value of register rs2
//! - `rv1_ext_bit`, `rv2_ext_bit`, `res_ext_bit`: Bit (for word instruction extension)
//! - `arg1`: DWordBL (8 cols) - extended rv1
//! - `arg2`: DWordBL (8 cols) - multiplexed rv2/imm
//! - `res`: DWordBL (8 cols) - ALU result
//! - `is_equal`: Bit - whether arg1 == arg2
//! - `branch_cond`: Bit - whether branch is taken
//!
//! ## Bus Interactions
//!
//! ### Senders (CPU sends to other tables)
//! - DECODE: instruction fetch
//! - IS_BYTE: range checks for rs1, rs2, rd, and arg1/arg2/res byte pairs
//! - IS_BIT: range checks for flags (via templates)
//! - ADD: for ADD, LOAD, JALR operations
//! - STORE ADD: for STORE (res = arg1 + imm, separate from main ADD)
//! - SUB: for SUB, BEQ operations
//! - LT: for SLT, BLT operations
//! - AND_BYTE, OR_BYTE, XOR_BYTE: for bitwise operations (×8 each)
//! - SHIFT: for shift operations
//! - MUL: for multiplication
//! - DIVREM: for division/remainder
//! - MEMW: for register and memory access
//! - MSB16: for sign/extension bit extraction (rv1, rv2, res)
//! - ZERO: for equality check
//! - BRANCH: for branch target calculation
//! - ECALL: for system calls

use super::types::{BusId, DecodeEntry, FE, GoldilocksExtension, GoldilocksField};
use crate::Error;
use alloc::vec;
use alloc::vec::Vec;
#[cfg(feature = "prove")]
use executor::vm::{
    instruction::{decoding::Instruction, execution::SyscallNumbers},
    logs::Log,
    memory::U64HashMap,
};
use smallvec::smallvec;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

/// PC value used for CPU padding rows. Per spec, this is an odd address (unreachable
/// during normal execution) with all flags=0. The DECODE table must contain a
/// corresponding entry at this PC.
pub const CPU_PADDING_PC: u64 = 1;

// =========================================================================
// Column indices for CPU table
// =========================================================================

/// Column definitions for the CPU table.
pub mod cols {
    // -------------------------------------------------------------------------
    // Input columns (from DECODE)
    // -------------------------------------------------------------------------

    /// timestamp: Timestamp for memory argument coordination
    pub const TIMESTAMP: usize = 0;

    /// pc[0]: Program counter (low word)
    pub const PC_0: usize = 1;
    /// pc[1]: Program counter (high word)
    pub const PC_1: usize = 2;

    /// rs1: Source register 1 index (Byte)
    pub const RS1: usize = 3;
    /// rs2: Source register 2 index (Byte)
    pub const RS2: usize = 4;
    /// rd: Destination register index (Byte)
    pub const RD: usize = 5;

    /// read_register1: Whether to read from rs1 (Bit)
    pub const READ_REGISTER1: usize = 6;
    /// read_register2: Whether to read from rs2 (Bit)
    pub const READ_REGISTER2: usize = 7;
    /// write_register: Whether to write back to rd (Bit)
    pub const WRITE_REGISTER: usize = 8;
    /// memory_2bytes: Memory access is 2 bytes (Bit)
    pub const MEMORY_2BYTES: usize = 9;
    /// memory_4bytes: Memory access is 4 bytes (Bit)
    pub const MEMORY_4BYTES: usize = 10;
    /// memory_8bytes: Memory access is 8 bytes (Bit)
    pub const MEMORY_8BYTES: usize = 11;
    /// c_type_instruction: Instruction is 2 bytes (compressed) instead of 4 (Bit)
    pub const C_TYPE_INSTRUCTION: usize = 12;

    /// imm[0]: Immediate value (low word)
    pub const IMM_0: usize = 13;
    /// imm[1]: Immediate value (high word)
    pub const IMM_1: usize = 14;

    /// signed: Signed operation flag (Bit)
    pub const SIGNED: usize = 15;
    /// mp_selector: Multi-purpose selector (branch invert, shift direction, MUL variant)
    pub const MP_SELECTOR: usize = 16;
    /// muldiv_selector: Select MUL/DIV output variant
    pub const MULDIV_SELECTOR: usize = 17;
    /// word_instr: 32-bit word instruction (requires sign extension)
    pub const WORD_INSTR: usize = 18;

    // ALU selector flags (one-hot encoded)
    /// ADD operation
    pub const ADD: usize = 19;
    /// SUB operation
    pub const SUB: usize = 20;
    /// SLT (Set Less Than) operation
    pub const SLT: usize = 21;
    /// AND operation
    pub const AND: usize = 22;
    /// OR operation
    pub const OR: usize = 23;
    /// XOR operation
    pub const XOR: usize = 24;
    /// SHIFT operation
    pub const SHIFT: usize = 25;
    /// JALR (Jump And Link Register)
    pub const JALR: usize = 26;
    /// BEQ (Branch if Equal)
    pub const BEQ: usize = 27;
    /// BLT (Branch if Less Than)
    pub const BLT: usize = 28;
    /// LOAD operation
    pub const LOAD: usize = 29;
    /// STORE operation
    pub const STORE: usize = 30;
    /// MUL operation
    pub const MUL: usize = 31;
    /// DIVREM (Division/Remainder) operation
    pub const DIVREM: usize = 32;
    /// ECALL (Environment Call)
    pub const ECALL: usize = 33;
    /// EBREAK (Environment Break)
    pub const EBREAK: usize = 34;

    // -------------------------------------------------------------------------
    // Output columns
    // -------------------------------------------------------------------------

    /// next_pc[0]: Next program counter (low word)
    pub const NEXT_PC_0: usize = 35;
    /// next_pc[1]: Next program counter (high word)
    pub const NEXT_PC_1: usize = 36;

    /// rvd[0]: Value to write to destination register (low word)
    pub const RVD_0: usize = 37;
    /// rvd[1]: Value to write to destination register (high word)
    pub const RVD_1: usize = 38;

    // -------------------------------------------------------------------------
    // Auxiliary columns
    // -------------------------------------------------------------------------

    /// rv1[0]: Register rs1 value (Half - bits 0-15) [DWordWHH]
    pub const RV1_0: usize = 39;
    /// rv1[1]: Register rs1 value (Half - bits 16-31) [DWordWHH]
    pub const RV1_1: usize = 40;
    /// rv1[2]: Register rs1 value (Word - bits 32-63) [DWordWHH]
    pub const RV1_2: usize = 41;

    /// rv2[0]: Register rs2 value (Half - bits 0-15) [DWordWHH]
    pub const RV2_0: usize = 42;
    /// rv2[1]: Register rs2 value (Half - bits 16-31) [DWordWHH]
    pub const RV2_1: usize = 43;
    /// rv2[2]: Register rs2 value (Word - bits 32-63) [DWordWHH]
    pub const RV2_2: usize = 44;

    /// rv1_ext_bit: Sign bit of rv1 as 32-bit word (for word_instr sign extension)
    pub const RV1_EXT_BIT: usize = 45;

    /// arg1[0..8]: Extended rv1 as DWordBL (8 bytes)
    pub const ARG1_0: usize = 46;
    pub const ARG1_1: usize = 47;
    pub const ARG1_2: usize = 48;
    pub const ARG1_3: usize = 49;
    pub const ARG1_4: usize = 50;
    pub const ARG1_5: usize = 51;
    pub const ARG1_6: usize = 52;
    pub const ARG1_7: usize = 53;

    /// rv2_ext_bit: Sign bit of rv2 as 32-bit word (bit 31 of rv2; used for arg2 sign extension)
    pub const RV2_EXT_BIT: usize = 54;

    /// arg2[0..8]: Extended rv2/imm as DWordBL (8 bytes)
    pub const ARG2_0: usize = 55;
    pub const ARG2_1: usize = 56;
    pub const ARG2_2: usize = 57;
    pub const ARG2_3: usize = 58;
    pub const ARG2_4: usize = 59;
    pub const ARG2_5: usize = 60;
    pub const ARG2_6: usize = 61;
    pub const ARG2_7: usize = 62;

    /// res_ext_bit: Sign bit of res as 32-bit word (for rvd sign extension)
    pub const RES_EXT_BIT: usize = 63;

    /// res[0..8]: ALU result as DWordBL (8 bytes)
    pub const RES_0: usize = 64;
    pub const RES_1: usize = 65;
    pub const RES_2: usize = 66;
    pub const RES_3: usize = 67;
    pub const RES_4: usize = 68;
    pub const RES_5: usize = 69;
    pub const RES_6: usize = 70;
    pub const RES_7: usize = 71;

    /// is_equal: Whether rv1 == arg2 (for BEQ)
    pub const IS_EQUAL: usize = 72;

    /// branch_cond: Whether branch is taken
    pub const BRANCH_COND: usize = 73;

    /// prev_pc_timestamp_borrow: Borrow bit for the 32-bit subtraction timestamp_lo - 3
    /// in the inline PC prev_ts formula. Fires only when timestamp_lo < 3 and
    /// pc_double_read = 0 (i.e. after timestamp wraps past 2^32 into values 0..2).
    pub const PREV_PC_TIMESTAMP_BORROW: usize = 74;

    /// pc_double_read: Whether PC is read as rs1 this cycle (AUIPC/JAL)
    pub const PC_DOUBLE_READ: usize = 75;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 76;

    // -------------------------------------------------------------------------
    // Helper ranges for iteration
    // -------------------------------------------------------------------------

    /// ARG1 byte columns as array
    pub const ARG1: [usize; 8] = [
        ARG1_0, ARG1_1, ARG1_2, ARG1_3, ARG1_4, ARG1_5, ARG1_6, ARG1_7,
    ];

    /// ARG2 byte columns as array
    pub const ARG2: [usize; 8] = [
        ARG2_0, ARG2_1, ARG2_2, ARG2_3, ARG2_4, ARG2_5, ARG2_6, ARG2_7,
    ];

    /// RES byte columns as array
    pub const RES: [usize; 8] = [RES_0, RES_1, RES_2, RES_3, RES_4, RES_5, RES_6, RES_7];
}

// =========================================================================
// CPU Operation (for trace generation)
// =========================================================================

/// A single CPU cycle to be added to the trace.
///
/// Contains static decode information (from DecodeEntry) plus runtime values
/// from execution (register values, computed results, etc.).
#[derive(Debug, Clone, Default)]
pub struct CpuOperation {
    /// Static decode information (shared with DECODE table)
    pub decode: DecodeEntry,

    /// Timestamp for memory argument coordination
    pub timestamp: u64,

    /// Next program counter (from execution)
    pub next_pc: u64,

    /// Value to write to destination register (from execution)
    pub rvd: u64,

    /// Value of register rs1 (from execution)
    pub rv1: u64,

    /// Value of register rs2 (from execution)
    pub rv2: u64,

    /// ALU result or memory address (computed)
    pub res: u64,

    /// Whether rv1 == rv2 (for BEQ)
    pub is_equal: bool,

    /// Whether branch is taken
    pub branch_cond: bool,

    /// Whether this ECALL is a Commit syscall
    pub ecall_commit: bool,

    /// For Commit ECALLs: buffer address from x11
    pub commit_buf_addr: u64,

    /// For Commit ECALLs: byte count from x12
    pub commit_count: u64,

    /// Whether this ECALL is a KeccakPermute syscall
    pub ecall_keccak: bool,

    /// For KeccakPermute ECALLs: state address from x10
    pub keccak_state_addr: u64,

    /// Whether this ECALL is a Fp3Mul syscall
    pub ecall_fp3_mul: bool,

    /// For Fp3Mul ECALLs: result pointer from x10 (a0)
    pub fp3_mul_result_ptr: u64,
}

impl CpuOperation {
    /// Creates a new CPU operation with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    // =========================================================================
    // Convenience accessors for decode fields (reduces verbosity)
    // =========================================================================

    #[inline]
    pub fn pc(&self) -> u64 {
        self.decode.pc
    }
    #[inline]
    pub fn rs1(&self) -> u8 {
        self.decode.rs1
    }
    #[inline]
    pub fn rs2(&self) -> u8 {
        self.decode.rs2
    }
    #[inline]
    pub fn rd(&self) -> u8 {
        self.decode.rd
    }
    #[inline]
    pub fn imm(&self) -> u64 {
        self.decode.imm
    }
    #[inline]
    pub fn word_instr(&self) -> bool {
        self.decode.word_instr
    }
    #[inline]
    pub fn signed(&self) -> bool {
        self.decode.signed
    }

    // =========================================================================
    // Computation methods
    // =========================================================================

    /// Compute arg1 from rv1 based on word_instr and signed flags.
    ///
    /// Per spec constraint: arg1[4:] = rv1[2] * (1 - word_instr) + (2^32 - 1) * rv1_ext_bit * signed
    ///
    /// For 64-bit instructions: pass through full rv1
    /// For unsigned word instructions: zero-extend from 32 bits
    /// For signed word instructions: sign-extend from 32 bits
    pub fn compute_arg1(&self) -> u64 {
        if self.decode.word_instr {
            let lower_32 = self.rv1 & 0xFFFF_FFFF;
            if self.decode.signed && Self::sign_bit_32(self.rv1) {
                // Sign extend: set upper 32 bits to all 1s
                lower_32 | (0xFFFF_FFFF_u64 << 32)
            } else {
                // Zero extend: upper 32 bits are 0
                lower_32
            }
        } else {
            self.rv1
        }
    }

    /// Compute arg2 following the spec formula exactly (CPU-CE62/CE63).
    ///
    /// arg2[:4] = (1-LOAD)*rv2[:2] + (1-BEQ-BLT-STORE)*imm[0]
    /// arg2[4:] = (1-LOAD)*((1-word_instr)*rv2[2] + signed*rv2_ext_bit*(2^32-1))
    ///            + (1-BEQ-BLT-STORE)*imm[1]
    ///
    /// Per CPU-A2, the decode guarantees that at most one of rv2/imm is non-zero
    /// when STORE+LOAD+BEQ+BLT=0, so the addition acts as a selection.
    pub fn compute_arg2(&self) -> u64 {
        let d = &self.decode;

        // rv2 contribution: zeroed when LOAD (spec: (1-LOAD) factor)
        let rv2_extended = if d.op_load {
            0
        } else if d.word_instr {
            // Word-instruction sign/zero extension on upper 32 bits
            let lower_32 = self.rv2 & 0xFFFF_FFFF;
            if d.signed && Self::sign_bit_32(self.rv2) {
                lower_32 | (0xFFFF_FFFF_u64 << 32)
            } else {
                lower_32
            }
        } else {
            self.rv2
        };

        // imm contribution: zeroed when BEQ, BLT, or STORE (spec: (1-BEQ-BLT-STORE) factor)
        let imm_contrib = if d.op_beq || d.op_blt || d.op_store {
            0
        } else {
            d.imm
        };

        rv2_extended.wrapping_add(imm_contrib)
    }

    /// Extract sign bit of a 32-bit word (bit 31).
    pub fn sign_bit_32(val: u64) -> bool {
        (val >> 31) & 1 == 1
    }

    /// Compute rvd (destination register value) based on res and word_instr.
    ///
    /// According to spec constraints:
    /// - rvd[0] = res[:4] (lower 32 bits of res)
    /// - rvd[1] = (1 - word_instr) * res[4:] + res_ext_bit * (2^32 - 1)
    ///
    /// For LOAD: rvd comes from the executor (loaded value), not this method.
    /// For all other operations: rvd is computed from res with sign extension.
    pub fn compute_rvd(&self) -> u64 {
        let res = self.compute_res();
        let res_lo = res & 0xFFFF_FFFF;

        if self.decode.word_instr {
            // Sign extend from 32 bits
            let res_ext_bit = Self::sign_bit_32(res);
            if res_ext_bit {
                // Upper 32 bits = 0xFFFF_FFFF (sign extension)
                res_lo | (0xFFFF_FFFF_u64 << 32)
            } else {
                // Upper 32 bits = 0 (zero extension)
                res_lo
            }
        } else {
            // rvd = res (full 64-bit value)
            res
        }
    }

    /// Compute the result based on operation type.
    ///
    /// For ADD: res = arg1 + arg2 (64-bit wrapping)
    /// For SUB: res = arg1 - arg2 (64-bit wrapping)
    /// For SHIFT: res = raw 64-bit shift of arg1 by arg2 (no word sign extension;
    ///            rvd handles sign extension for word instructions)
    /// For SLT: res = 0 or 1 (comparison result from executor)
    /// For other operations: uses the executor's result (self.res)
    ///
    /// This ensures the ADD/SUB constraints are satisfied.
    /// The rvd column holds the actual sign-extended result for word instructions.
    pub fn compute_res(&self) -> u64 {
        let arg1 = self.compute_arg1();
        let arg2 = self.compute_arg2();

        if self.decode.op_add || self.decode.op_load {
            // ADD constraint: arg1 + arg2 = res
            // For ADD: computes arithmetic result
            // For LOAD: computes memory address (rv1 + imm)
            arg1.wrapping_add(arg2)
        } else if self.decode.op_store {
            // STORE: res = arg1 + imm (address), not arg1 + arg2 (which is now rv2)
            arg1.wrapping_add(self.decode.imm)
        } else if self.decode.op_sub {
            // SUB constraint checks: res + arg2 = arg1, so res = arg1 - arg2
            arg1.wrapping_sub(arg2)
        } else if self.decode.op_shift {
            // SHIFT: raw 64-bit shift matching the SHIFT chip's computation.
            // The SHIFT chip shifts the full 64-bit arg1 by (shift mod 32*(2-word_instr)).
            // Sign extension for word instructions is handled by rvd, not res.
            let shift = (arg2 & 0xFF) as u32;
            let modulus = if self.decode.word_instr { 32 } else { 64 };
            let effective = shift % modulus;
            if !self.decode.mp_selector {
                // Left shift
                arg1.wrapping_shl(effective)
            } else if !self.decode.signed {
                // Logical right shift
                arg1.wrapping_shr(effective)
            } else {
                // Arithmetic right shift
                (arg1 as i64).wrapping_shr(effective) as u64
            }
        } else {
            // For SLT and other operations, use the executor's result
            // SLT res is 0 or 1, verified by SltResZeroConstraint
            self.res
        }
    }

    /// Collects CPU range-check lookups for register indices and byte pairs.
    ///
    /// The CPU sends:
    /// - 1 IS_BYTE lookup for (RS1, RS2) batched as a pair
    /// - 1 IS_BYTE lookup for RD encoded as (RD, 0)
    /// - 12 IS_BYTE lookups for adjacent byte pairs in ARG1, ARG2, and RES
    pub fn collect_byte_check_ops(&self) -> Vec<super::bitwise::BitwiseOperation> {
        use super::bitwise::{BitwiseOperation, BitwiseOperationType};

        let arg1 = self.compute_arg1();
        let arg2 = self.compute_arg2();
        let res = self.compute_res();

        let mut ops = Vec::with_capacity(14);

        // Batch RS1+RS2 as a pair; RD stays single with Y=0.
        ops.push(BitwiseOperation::byte_op(
            BitwiseOperationType::IsByte,
            self.decode.rs1,
            self.decode.rs2,
        ));
        ops.push(BitwiseOperation::single_byte(
            BitwiseOperationType::IsByte,
            self.decode.rd,
        ));

        // 12 IS_BYTE lookups for ARG1/ARG2/RES byte pairs
        // Each pair sends [lo, hi] as two separate bus values, so the LogUp
        // fingerprint forces each byte to match individually against BITWISE X, Y.
        for value in [arg1, arg2, res] {
            for i in 0..4 {
                let lo = ((value >> (i * 16)) & 0xFF) as u8;
                let hi = ((value >> (i * 16 + 8)) & 0xFF) as u8;
                ops.push(BitwiseOperation::byte_op(
                    BitwiseOperationType::IsByte,
                    lo,
                    hi,
                ));
            }
        }

        ops
    }

    /// Collects Bitwise table lookups generated by this CPU operation.
    pub fn collect_bitwise_ops(&self) -> Vec<super::bitwise::BitwiseOperation> {
        use super::bitwise::{BitwiseOperation, BitwiseOperationType};
        let mut lookups = Vec::new();

        // Range checks: 14 IS_BYTE ops (RS1+RS2 paired, RD single with Y=0,
        // plus 12 ARG1/ARG2/RES byte pairs).
        lookups.extend(self.collect_byte_check_ops());

        // MSB16 lookups for sign bit extraction (when word_instr=1)
        if self.decode.word_instr {
            // rv1[1] is bits 16-31, extract as halfword for MSB16 lookup
            let rv1_half = ((self.rv1 >> 16) & 0xFFFF) as u16;
            let lo = (rv1_half & 0xFF) as u8;
            let hi = ((rv1_half >> 8) & 0xFF) as u8;
            lookups.push(BitwiseOperation::halfword(
                BitwiseOperationType::Msb16,
                lo,
                hi,
            ));

            // rv2[1] for rv2_ext_bit
            let rv2_half = ((self.rv2 >> 16) & 0xFFFF) as u16;
            let lo = (rv2_half & 0xFF) as u8;
            let hi = ((rv2_half >> 8) & 0xFF) as u8;
            lookups.push(BitwiseOperation::halfword(
                BitwiseOperationType::Msb16,
                lo,
                hi,
            ));

            // res::DWordHL[1] for res_ext_bit (MSB16 on half at bits 16-31)
            let res_half = ((self.res >> 16) & 0xFFFF) as u16;
            lookups.push(BitwiseOperation::halfword(
                BitwiseOperationType::Msb16,
                (res_half & 0xFF) as u8,
                (res_half >> 8) as u8,
            ));
        }

        // ZERO lookup for is_equal (when BEQ=1)
        if self.decode.op_beq {
            // Sum of all result bytes
            let mut sum: u64 = 0;
            for i in 0..8 {
                sum += (self.res >> (i * 8)) & 0xFF;
            }
            // Sum fits in 11 bits (max 8 * 255 = 2040), well within ZERO's 20-bit range
            lookups.push(BitwiseOperation::zero(sum as u32));
        }

        // AND/OR/XOR lookups (×8 each for each byte)
        let arg1 = self.compute_arg1();
        let arg2 = self.compute_arg2();

        if self.decode.op_and {
            for i in 0..8 {
                let a = ((arg1 >> (i * 8)) & 0xFF) as u8;
                let b = ((arg2 >> (i * 8)) & 0xFF) as u8;
                lookups.push(BitwiseOperation::byte_op(
                    BitwiseOperationType::AndByte,
                    a,
                    b,
                ));
            }
        }

        if self.decode.op_or {
            for i in 0..8 {
                let a = ((arg1 >> (i * 8)) & 0xFF) as u8;
                let b = ((arg2 >> (i * 8)) & 0xFF) as u8;
                lookups.push(BitwiseOperation::byte_op(
                    BitwiseOperationType::OrByte,
                    a,
                    b,
                ));
            }
        }

        if self.decode.op_xor {
            for i in 0..8 {
                let a = ((arg1 >> (i * 8)) & 0xFF) as u8;
                let b = ((arg2 >> (i * 8)) & 0xFF) as u8;
                lookups.push(BitwiseOperation::byte_op(
                    BitwiseOperationType::XorByte,
                    a,
                    b,
                ));
            }
        }

        lookups
    }

    /// Creates a CpuOperation from an executor Log and DecodeEntry.
    ///
    /// The DecodeEntry contains static instruction information. This method
    /// adds runtime values from the Log (register values, branch decisions, etc.).
    #[cfg(feature = "prove")]
    pub fn from_log(log: &Log, timestamp: u64, decode: DecodeEntry) -> Self {
        let ecall_commit = decode.op_ecall && log.src1_val == SyscallNumbers::Commit as u64;
        let (commit_buf_addr, commit_count) = if ecall_commit {
            (log.src2_val, log.dst_val)
        } else {
            (0, 0)
        };
        let ecall_keccak =
            decode.op_ecall && log.src1_val == executor::constants::KECCAK_SYSCALL_NUMBER;
        let keccak_state_addr = if ecall_keccak { log.src2_val } else { 0 };
        let ecall_fp3_mul =
            decode.op_ecall && log.src1_val == executor::constants::FP3_MUL_SYSCALL_NUMBER;
        // The executor sets src2_val = result_ptr for Fp3Mul (see execution.rs).
        let fp3_mul_result_ptr = if ecall_fp3_mul { log.src2_val } else { 0 };
        // CM50: (1 - read_register2) * rv2[i] = 0. When read_register2=0, rv2 must be 0.
        // For example, ECALL has read_register2=0 (rs2 defaults to 0). The commit buf_addr is
        // carried separately in commit_buf_addr and does not go through rv2.
        let rv2 = if !decode.read_register2 {
            0
        } else {
            log.src2_val
        };

        let mut op = Self {
            decode,
            timestamp,
            next_pc: log.next_pc,
            rv1: log.src1_val,
            rv2,
            rvd: log.dst_val,
            res: log.dst_val, // Default: result is destination value
            is_equal: false,
            branch_cond: false,
            ecall_commit,
            commit_buf_addr,
            commit_count,
            ecall_keccak,
            keccak_state_addr,
            ecall_fp3_mul,
            fp3_mul_result_ptr,
        };

        // Compute runtime-specific values based on instruction type
        op.compute_runtime_values(log);
        op
    }

    /// Creates a CpuOperation from Log and Instruction (convenience method).
    ///
    /// This creates the DecodeEntry internally. Use `from_log` with a pre-built
    /// DecodeEntry when possible to avoid redundant decoding.
    #[cfg(feature = "prove")]
    pub fn from_log_and_instruction(log: &Log, timestamp: u64, instruction: Instruction) -> Self {
        let decode = DecodeEntry::from_instruction(log.current_pc, instruction);
        Self::from_log(log, timestamp, decode)
    }

    /// Computes runtime-specific values based on the instruction type.
    ///
    /// This handles:
    /// - Memory address computation for LOAD/STORE
    /// - Branch condition and result computation for BEQ/BLT
    /// - AUIPC special case (rv1 = current_pc)
    /// - JALR branch_cond = true
    #[cfg(feature = "prove")]
    fn compute_runtime_values(&mut self, log: &Log) {
        // JALR: always jumps
        if self.decode.op_jalr {
            self.branch_cond = true;
        }

        // LOAD/STORE: res = memory address = rv1 + imm
        if self.decode.op_load || self.decode.op_store {
            self.res = (log.src1_val as i64 + self.decode.imm as i64) as u64;
        }

        // BEQ: res = rv1 - rv2, branch if equal (or not equal for BNE)
        if self.decode.op_beq {
            self.is_equal = log.src1_val == log.src2_val;
            self.res = log.src1_val.wrapping_sub(log.src2_val);
            // mp_selector inverts the condition (BNE vs BEQ)
            self.branch_cond = if self.decode.mp_selector {
                log.src1_val != log.src2_val
            } else {
                log.src1_val == log.src2_val
            };
        }

        // BLT: res = comparison result (0 or 1)
        if self.decode.op_blt {
            self.is_equal = log.src1_val == log.src2_val;
            let lt_result = if self.decode.signed {
                (log.src1_val as i64) < (log.src2_val as i64)
            } else {
                log.src1_val < log.src2_val
            };
            self.res = lt_result as u64;
            // mp_selector inverts the condition (BGE/BGEU vs BLT/BLTU)
            self.branch_cond = if self.decode.mp_selector {
                !lt_result
            } else {
                lt_result
            };
        }

        // AUIPC/JAL: rv1 should be current_pc (special case)
        // Per spec, these instructions use rs1=255 (virtual PC register)
        if self.decode.rs1 == 255 {
            self.rv1 = log.current_pc;
        }

        // ECALL: Per spec constraint CO69, next_pc = pc + instr_size for all instructions,
        // including ECALL. The CPU transition constraint enforces next_pc = pc + 4 on every
        // row, so the trace must satisfy this even though the executor sets next_pc=0 to
        // signal halt. The HALT table separately proves program termination via the ECALL bus.
        if self.decode.op_ecall {
            self.next_pc = self.decode.pc + 4;
        }
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the CPU trace table from a list of operations.
///
/// Each operation becomes one row in the table. The table is then
/// padded to the next power of 2.
pub fn generate_cpu_trace(
    operations: &[CpuOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = operations.len();

    let num_rows = n.next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, op) in operations.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;
        let d = &op.decode; // Shorthand for decode fields

        // Input columns (from decode)
        data[base + cols::TIMESTAMP] = FE::from(op.timestamp);
        data[base + cols::PC_0] = FE::from(d.pc & 0xFFFF_FFFF);
        data[base + cols::PC_1] = FE::from(d.pc >> 32);
        data[base + cols::RS1] = FE::from(d.rs1 as u64);
        data[base + cols::RS2] = FE::from(d.rs2 as u64);
        data[base + cols::RD] = FE::from(d.rd as u64);
        // Skip x0 (hardwired zero). x255 is the register where the pc is stored
        // (per spec decode.md). read_register1=1 for rs1=255 ensures the CM47 MEMW
        // interaction is sent and rv1 is not forced to zero by CM48.
        data[base + cols::READ_REGISTER1] = FE::from((d.read_register1 && d.rs1 != 0) as u64);
        data[base + cols::READ_REGISTER2] = FE::from((d.read_register2 && d.rs2 != 0) as u64);
        data[base + cols::WRITE_REGISTER] = FE::from((d.write_register && d.rd != 0) as u64);
        data[base + cols::MEMORY_2BYTES] = FE::from(d.memory_2bytes as u64);
        data[base + cols::MEMORY_4BYTES] = FE::from(d.memory_4bytes as u64);
        data[base + cols::MEMORY_8BYTES] = FE::from(d.memory_8bytes as u64);
        data[base + cols::C_TYPE_INSTRUCTION] = FE::from(d.c_type as u64);
        data[base + cols::IMM_0] = FE::from(d.imm & 0xFFFF_FFFF);
        data[base + cols::IMM_1] = FE::from(d.imm >> 32);
        data[base + cols::SIGNED] = FE::from(d.signed as u64);
        data[base + cols::MP_SELECTOR] = FE::from(d.mp_selector as u64);
        data[base + cols::MULDIV_SELECTOR] = FE::from(d.muldiv_selector as u64);
        data[base + cols::WORD_INSTR] = FE::from(d.word_instr as u64);

        // ALU selector flags
        data[base + cols::ADD] = FE::from(d.op_add as u64);
        data[base + cols::SUB] = FE::from(d.op_sub as u64);
        data[base + cols::SLT] = FE::from(d.op_slt as u64);
        data[base + cols::AND] = FE::from(d.op_and as u64);
        data[base + cols::OR] = FE::from(d.op_or as u64);
        data[base + cols::XOR] = FE::from(d.op_xor as u64);
        data[base + cols::SHIFT] = FE::from(d.op_shift as u64);
        data[base + cols::JALR] = FE::from(d.op_jalr as u64);
        data[base + cols::BEQ] = FE::from(d.op_beq as u64);
        data[base + cols::BLT] = FE::from(d.op_blt as u64);
        data[base + cols::LOAD] = FE::from(d.op_load as u64);
        data[base + cols::STORE] = FE::from(d.op_store as u64);
        data[base + cols::MUL] = FE::from(d.op_mul as u64);
        data[base + cols::DIVREM] = FE::from(d.op_divrem as u64);
        data[base + cols::ECALL] = FE::from(d.op_ecall as u64);
        data[base + cols::EBREAK] = FE::from(d.op_ebreak as u64);

        // Output columns
        data[base + cols::NEXT_PC_0] = FE::from(op.next_pc & 0xFFFF_FFFF);
        data[base + cols::NEXT_PC_1] = FE::from(op.next_pc >> 32);

        // rvd: For LOAD, use the executor's loaded value (op.rvd).
        // For all other operations (including STORE), compute from res with sign extension.
        // This satisfies spec constraint: (1-LOAD) * (rvd - res_extended) = 0
        let rvd = if d.op_load {
            op.rvd // Loaded value from executor
        } else {
            op.compute_rvd() // res with sign extension for word instructions
        };
        data[base + cols::RVD_0] = FE::from(rvd & 0xFFFF_FFFF);
        data[base + cols::RVD_1] = FE::from(rvd >> 32);

        // Auxiliary: rv1 as DWordWHH [Half, Half, Word] - Word is MSB (bits 32-63)
        data[base + cols::RV1_0] = FE::from(op.rv1 & 0xFFFF); // bits 0-15 (Half)
        data[base + cols::RV1_1] = FE::from((op.rv1 >> 16) & 0xFFFF); // bits 16-31 (Half)
        data[base + cols::RV1_2] = FE::from(op.rv1 >> 32); // bits 32-63 (Word)

        // Auxiliary: rv2 as DWordWHH [Half, Half, Word] - Word is MSB (bits 32-63)
        data[base + cols::RV2_0] = FE::from(op.rv2 & 0xFFFF); // bits 0-15 (Half)
        data[base + cols::RV2_1] = FE::from((op.rv2 >> 16) & 0xFFFF); // bits 16-31 (Half)
        data[base + cols::RV2_2] = FE::from(op.rv2 >> 32); // bits 32-63 (Word)

        // Extension bits - only set when word_instr=1, per SIGN template
        // The constraint enforces: (1 - word_instr) * ext_bit = 0 for each ext bit
        let rv1_ext_bit = d.word_instr && CpuOperation::sign_bit_32(op.rv1);
        data[base + cols::RV1_EXT_BIT] = FE::from(rv1_ext_bit as u64);

        // Compute and store arg1 as DWordBL (8 bytes)
        let arg1 = op.compute_arg1();
        for i in 0..8 {
            data[base + cols::ARG1[i]] = FE::from((arg1 >> (i * 8)) & 0xFF);
        }

        // Compute and store arg2
        let arg2 = op.compute_arg2();
        let rv2_ext_bit = d.word_instr && CpuOperation::sign_bit_32(op.rv2);
        data[base + cols::RV2_EXT_BIT] = FE::from(rv2_ext_bit as u64);
        for i in 0..8 {
            data[base + cols::ARG2[i]] = FE::from((arg2 >> (i * 8)) & 0xFF);
        }

        // Result - computed from arg1/arg2 for ADD/SUB to satisfy constraints
        let res = op.compute_res();
        let res_ext_bit = d.word_instr && CpuOperation::sign_bit_32(res);
        data[base + cols::RES_EXT_BIT] = FE::from(res_ext_bit as u64);
        for i in 0..8 {
            data[base + cols::RES[i]] = FE::from((res >> (i * 8)) & 0xFF);
        }

        // Branch columns
        data[base + cols::IS_EQUAL] = FE::from(op.is_equal as u64);
        data[base + cols::BRANCH_COND] = FE::from(op.branch_cond as u64);

        // Inline PC columns
        let pc_double_read = (d.read_register1 && d.rs1 == 255) as u64;
        let ts_lo = op.timestamp & 0xFFFF_FFFF;
        let prev_pc_ts_borrow = if pc_double_read == 0 && ts_lo < 3 {
            1u64
        } else {
            0u64
        };
        data[base + cols::PC_DOUBLE_READ] = FE::from(pc_double_read);
        data[base + cols::PREV_PC_TIMESTAMP_BORROW] = FE::from(prev_pc_ts_borrow);
    }

    // Padding rows: per spec, padding uses pc=1 (odd address, unreachable during
    // normal execution) with all flags=0, so pad=1 and no bus interactions fire.
    // next_pc=5 satisfies the NextPcAdd constraint: carry=(1+4-5)/2^32=0.
    // The DECODE table must contain a corresponding entry at pc=1.
    for row_idx in n..num_rows {
        let base = row_idx * cols::NUM_COLUMNS;
        data[base + cols::PC_0] = FE::from(CPU_PADDING_PC);
        data[base + cols::NEXT_PC_0] = FE::from(CPU_PADDING_PC + 4);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

/// Generates the CPU trace table directly from executor logs.
///
/// This is a convenience function that converts logs to CpuOperations
/// and then generates the trace.
///
/// Returns an error if an instruction is not found for a PC.
/// Panics if logs.len() is not a power of 2 >= 4.
#[cfg(feature = "prove")]
pub fn generate_cpu_trace_from_logs(
    logs: &[Log],
    instructions: &U64HashMap<Instruction>,
) -> Result<TraceTable<GoldilocksField, GoldilocksExtension>, Error> {
    let mut operations = Vec::with_capacity(logs.len());
    for (i, log) in logs.iter().enumerate() {
        let instruction = *instructions
            .get(&log.current_pc)
            .ok_or(Error::MissingInstruction(log.current_pc))?;
        operations.push(CpuOperation::from_log_and_instruction(
            log,
            (i as u64) * 4 + 4,
            instruction,
        ));
    }
    Ok(generate_cpu_trace(&operations))
}

/// Collects all Bitwise lookups from a list of CPU operations.
pub fn collect_bitwise_ops(operations: &[CpuOperation]) -> Vec<super::bitwise::BitwiseOperation> {
    operations
        .iter()
        .flat_map(|op| op.collect_bitwise_ops())
        .collect()
}

/// Collects all Bitwise lookups from executor logs.
///
/// Convenience function that converts logs to operations and collects lookups.
#[cfg(feature = "prove")]
pub fn collect_bitwise_ops_from_logs(
    logs: &[Log],
    instructions: &U64HashMap<Instruction>,
) -> Result<Vec<super::bitwise::BitwiseOperation>, Error> {
    let mut operations = Vec::with_capacity(logs.len());
    for (i, log) in logs.iter().enumerate() {
        let instruction = *instructions
            .get(&log.current_pc)
            .ok_or(Error::MissingInstruction(log.current_pc))?;
        operations.push(CpuOperation::from_log_and_instruction(
            log,
            (i as u64) * 4 + 4,
            instruction,
        ));
    }
    Ok(collect_bitwise_ops(&operations))
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Helper to create a LinearTerm with coefficient 2^bit for a column.
fn linear_term(bit: u32, column: usize) -> LinearTerm {
    LinearTerm::Column {
        coefficient: 1 << bit,
        column,
    }
}

/// Returns the bus interactions for the CPU table.
///
/// The CPU table sends to:
/// - DECODE: instruction fetch (every row)
/// - AND_BYTE, OR_BYTE, XOR_BYTE: for bitwise operations (×8 each)
///
/// Note: LT interaction is TODO - needs proper DWordHHW packing to match LT table receiver.
pub fn bus_interactions() -> Vec<BusInteraction> {
    use super::types::packed_decode as bits;

    let mut interactions = Vec::new();

    // -------------------------------------------------------------------------
    // DECODE interaction (instruction fetch)
    // -------------------------------------------------------------------------
    // Every CPU row looks up the DECODE table once to verify instruction decoding.
    // Format: DECODE[pc::DWordWL, imm::DWordWL, packed_decode]
    //
    // packed_decode is computed as a linear combination of all decode columns.
    // Bit positions are defined in types::packed_decode (single source of truth).
    interactions.push(BusInteraction::sender(
        BusId::Decode,
        Multiplicity::One, // Every row sends exactly once
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
            // packed_decode as linear combination of decode columns
            BusValue::linear(vec![
                // Control flags (bits 0-10)
                linear_term(bits::READ_REG1, cols::READ_REGISTER1),
                linear_term(bits::READ_REG2, cols::READ_REGISTER2),
                linear_term(bits::WRITE_REG, cols::WRITE_REGISTER),
                linear_term(bits::MEMORY_2BYTES, cols::MEMORY_2BYTES),
                linear_term(bits::MEMORY_4BYTES, cols::MEMORY_4BYTES),
                linear_term(bits::MEMORY_8BYTES, cols::MEMORY_8BYTES),
                linear_term(bits::C_TYPE, cols::C_TYPE_INSTRUCTION),
                linear_term(bits::SIGNED, cols::SIGNED),
                linear_term(bits::MP_SELECTOR, cols::MP_SELECTOR),
                linear_term(bits::MULDIV_SELECTOR, cols::MULDIV_SELECTOR),
                linear_term(bits::WORD_INSTR, cols::WORD_INSTR),
                // ALU selector flags (bits 11-26)
                linear_term(bits::OP_ADD, cols::ADD),
                linear_term(bits::OP_SUB, cols::SUB),
                linear_term(bits::OP_SLT, cols::SLT),
                linear_term(bits::OP_AND, cols::AND),
                linear_term(bits::OP_OR, cols::OR),
                linear_term(bits::OP_XOR, cols::XOR),
                linear_term(bits::OP_SHIFT, cols::SHIFT),
                linear_term(bits::OP_JALR, cols::JALR),
                linear_term(bits::OP_BEQ, cols::BEQ),
                linear_term(bits::OP_BLT, cols::BLT),
                linear_term(bits::OP_LOAD, cols::LOAD),
                linear_term(bits::OP_STORE, cols::STORE),
                linear_term(bits::OP_MUL, cols::MUL),
                linear_term(bits::OP_DIVREM, cols::DIVREM),
                linear_term(bits::OP_ECALL, cols::ECALL),
                linear_term(bits::OP_EBREAK, cols::EBREAK),
                // Register indices (bits 27-50)
                linear_term(bits::RS1, cols::RS1),
                linear_term(bits::RS2, cols::RS2),
                linear_term(bits::RD, cols::RD),
            ]),
        ],
    ));

    // -------------------------------------------------------------------------
    // LT interaction (for SLT, BLT) - TODO: Re-add when properly implemented
    // -------------------------------------------------------------------------
    // The LT table receiver expects: lhs (DWordHHW: 3 cols), rhs (DWordHHW: 3 cols), signed, lt
    // The CPU has arg1/arg2 as DWordBL (8 bytes), needs Linear bus values to repack to HHW format
    // For now, commented out until we implement the proper packing.
    //
    // interactions.push(BusInteraction::sender(
    //     BusId::Lt,
    //     Multiplicity::Column(cols::SLT),
    //     vec![...], // Need Linear to repack DWordBL -> DWordHHW
    // ));

    // -------------------------------------------------------------------------
    // AND_BYTE interactions (×8 for each byte)
    // -------------------------------------------------------------------------
    for i in 0..8 {
        interactions.push(BusInteraction::sender(
            BusId::AndByte,
            Multiplicity::Column(cols::AND),
            smallvec![
                BusValue::Packed {
                    start_column: cols::ARG1[i],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ARG2[i],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::RES[i],
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // -------------------------------------------------------------------------
    // OR_BYTE interactions (×8)
    // -------------------------------------------------------------------------
    for i in 0..8 {
        interactions.push(BusInteraction::sender(
            BusId::OrByte,
            Multiplicity::Column(cols::OR),
            smallvec![
                BusValue::Packed {
                    start_column: cols::ARG1[i],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ARG2[i],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::RES[i],
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // -------------------------------------------------------------------------
    // XOR_BYTE interactions (×8)
    // -------------------------------------------------------------------------
    for i in 0..8 {
        interactions.push(BusInteraction::sender(
            BusId::XorByte,
            Multiplicity::Column(cols::XOR),
            smallvec![
                BusValue::Packed {
                    start_column: cols::ARG1[i],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ARG2[i],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::RES[i],
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // -------------------------------------------------------------------------
    // SIGN template: MSB16 interactions for extension bit extraction
    // -------------------------------------------------------------------------
    // SIGN(rv1[1], word_instr) -> rv1_ext_bit
    // rv1[1] is a Half (bits 16-31), MSB16 extracts bit 31
    interactions.push(BusInteraction::sender(
        BusId::Msb16,
        Multiplicity::Column(cols::WORD_INSTR),
        smallvec![
            BusValue::Packed {
                start_column: cols::RV1_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RV1_EXT_BIT,
                packing: Packing::Direct,
            },
        ],
    ));

    // SIGN(rv2[1], word_instr) -> rv2_ext_bit
    interactions.push(BusInteraction::sender(
        BusId::Msb16,
        Multiplicity::Column(cols::WORD_INSTR),
        smallvec![
            BusValue::Packed {
                start_column: cols::RV2_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RV2_EXT_BIT,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // MSB16 interaction for res extension bit extraction
    // -------------------------------------------------------------------------
    // MSB16[res::DWordHL[1]] -> res_ext_bit, multiplicity = word_instr
    // res::DWordHL[1] is the half at bits 16-31 = res[2] + 256*res[3]
    interactions.push(BusInteraction::sender(
        BusId::Msb16,
        Multiplicity::Column(cols::WORD_INSTR),
        smallvec![
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::RES[2],
                },
                LinearTerm::Column {
                    coefficient: 256,
                    column: cols::RES[3],
                },
            ]),
            BusValue::Packed {
                start_column: cols::RES_EXT_BIT,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // ZERO interaction for is_equal (BEQ)
    // -------------------------------------------------------------------------
    // ZERO[sum(res[0..7])] -> is_equal, multiplicity = BEQ
    // If all 8 bytes of res are zero, sum = 0, is_equal = 1
    interactions.push(BusInteraction::sender(
        BusId::Zero,
        Multiplicity::Column(cols::BEQ),
        smallvec![
            // Sum of all 8 result bytes as linear combination
            BusValue::linear(vec![
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::RES[0],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::RES[1],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::RES[2],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::RES[3],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::RES[4],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::RES[5],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::RES[6],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::RES[7],
                },
            ]),
            BusValue::Packed {
                start_column: cols::IS_EQUAL,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // LT interaction (for SLT, BLT)
    // -------------------------------------------------------------------------
    // LT[arg1, arg2, signed] -> res[0]
    // multiplicity = SLT + BLT
    //
    // LT bus uses 2 elements per 64-bit operand: [lo32, hi32]
    // arg1/arg2 are DWordBL (8 bytes) - use Packing::DWordBL to produce 2 elements
    interactions.push(BusInteraction::sender(
        BusId::Lt,
        // SLT + BLT using Multiplicity::Sum
        Multiplicity::Sum(cols::SLT, cols::BLT),
        smallvec![
            // arg1 as DWordBL (8 bytes → 2 elements: [lo32, hi32])
            BusValue::Packed {
                start_column: cols::ARG1[0],
                packing: Packing::DWordBL,
            },
            // arg2 as DWordBL (8 bytes → 2 elements: [lo32, hi32])
            BusValue::Packed {
                start_column: cols::ARG2[0],
                packing: Packing::DWordBL,
            },
            // signed flag
            BusValue::Packed {
                start_column: cols::SIGNED,
                packing: Packing::Direct,
            },
            // lt result (res[0])
            BusValue::Packed {
                start_column: cols::RES[0],
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // MUL interaction (for MUL, MULH, MULHSU, MULHU)
    // -------------------------------------------------------------------------
    // MUL[arg1, signed, arg2, mp_selector, rvd, muldiv_selector] per spec CPU-CA44
    // multiplicity = MUL
    //
    // The MUL table expects DWordHL (4 halfwords), but CPU has DWordBL (8 bytes).
    // Both pack to 2 words (lo32, hi32), so the signatures match for the same values.
    //
    // rhs_signed = mp_selector per spec:
    // - MUL/MULH: mp_selector=1 (both operands signed)
    // - MULHU/MULHSU: mp_selector=0 (rhs unsigned)
    //
    // muldiv_selector distinguishes lo (0) from hi (1) result
    interactions.push(BusInteraction::sender(
        BusId::Mul,
        Multiplicity::Column(cols::MUL),
        smallvec![
            // arg1 (lhs) as DWordBL (8 bytes → 2 elements)
            BusValue::Packed {
                start_column: cols::ARG1[0],
                packing: Packing::DWordBL,
            },
            // lhs_signed = signed
            BusValue::Packed {
                start_column: cols::SIGNED,
                packing: Packing::Direct,
            },
            // arg2 (rhs) as DWordBL (8 bytes → 2 elements)
            BusValue::Packed {
                start_column: cols::ARG2[0],
                packing: Packing::DWordBL,
            },
            // rhs_signed = mp_selector
            BusValue::Packed {
                start_column: cols::MP_SELECTOR,
                packing: Packing::Direct,
            },
            // result (res) as DWordBL (8 bytes → 2 elements) per spec CPU-CA44.
            // Must send res (raw MUL output), not rvd. For MULW, rvd = sign_extend(res[31:0]),
            // which can differ from res when bits [63:32] ≠ sign_extend(bit31) of res.
            BusValue::Packed {
                start_column: cols::RES[0],
                packing: Packing::DWordBL,
            },
            // muldiv_selector: 0=lo (MUL), 1=hi (MULH/MULHSU/MULHU)
            BusValue::Packed {
                start_column: cols::MULDIV_SELECTOR,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // DVRM interaction (for DIV, DIVU, REM, REMU) — CPU-CA45
    // -------------------------------------------------------------------------
    // DVRM[rvd; arg1, arg2, signed, muldiv_selector]
    // multiplicity = DIVREM
    interactions.push(BusInteraction::sender(
        BusId::Dvrm,
        Multiplicity::Column(cols::DIVREM),
        smallvec![
            // arg1 (numerator n) as DWordBL (8 bytes → 2 elements)
            BusValue::Packed {
                start_column: cols::ARG1[0],
                packing: Packing::DWordBL,
            },
            // arg2 (denominator d) as DWordBL (8 bytes → 2 elements)
            BusValue::Packed {
                start_column: cols::ARG2[0],
                packing: Packing::DWordBL,
            },
            // signed
            BusValue::Packed {
                start_column: cols::SIGNED,
                packing: Packing::Direct,
            },
            // result (res) as DWordBL (8 bytes → 2 elements) per spec CPU-CA45.
            // Must send res (raw DVRM output), not rvd. For DIVW/REMW, rvd = sign_extend(res[31:0]),
            // which can differ from res when bits [63:32] ≠ sign_extend(bit31) of res.
            BusValue::Packed {
                start_column: cols::RES[0],
                packing: Packing::DWordBL,
            },
            // muldiv_selector: 0=quotient (DIV), 1=remainder (REM)
            BusValue::Packed {
                start_column: cols::MULDIV_SELECTOR,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // SHIFT interaction (for SLL, SRL, SRA) — CPU-CA43
    // -------------------------------------------------------------------------
    // SHIFT[res::DWordWL; arg1::DWordHL, arg2[0], mp_selector, signed, word_instr]
    // multiplicity = SHIFT
    interactions.push(BusInteraction::sender(
        BusId::Shift,
        Multiplicity::Column(cols::SHIFT),
        smallvec![
            // res (result) as DWordBL (8 bytes → 2 elements, same as DWordWL)
            BusValue::Packed {
                start_column: cols::RES[0],
                packing: Packing::DWordBL,
            },
            // arg1 (input) as DWordBL (8 bytes → 2 elements)
            BusValue::Packed {
                start_column: cols::ARG1[0],
                packing: Packing::DWordBL,
            },
            // arg2[0] (shift amount byte)
            BusValue::Packed {
                start_column: cols::ARG2[0],
                packing: Packing::Direct,
            },
            // mp_selector (direction: 0=left, 1=right)
            BusValue::Packed {
                start_column: cols::MP_SELECTOR,
                packing: Packing::Direct,
            },
            // signed
            BusValue::Packed {
                start_column: cols::SIGNED,
                packing: Packing::Direct,
            },
            // word_instr
            BusValue::Packed {
                start_column: cols::WORD_INSTR,
                packing: Packing::Direct,
            },
        ],
    ));

    // =========================================================================
    // MEMW and LOAD bus interactions (M1, M3, M5, M6, M7)
    // =========================================================================
    // M1 and M3: Register read interactions (CPU → MEMW μ_read)
    // -------------------------------------------------------------------------
    // M1: MEMW[rv1; 1, 2*rs1, rv1, timestamp+0, 1, 0, 0] | read_register1
    // -------------------------------------------------------------------------
    // Read from rs1 register via MEMW. Format: 24 elements
    // [old[8], is_register, base_addr[2], value[8], timestamp[2], write2, write4, write8]
    //
    // Registers are stored as WL (2 words), remaining 6 values are unconstrained (zeros).
    // rv1 is DWordWHH (3 cols: Half, Half, Word) -> pack as WL: lo32 = rv1[0] + 2^16*rv1[1], hi32 = rv1[2]
    interactions.push(BusInteraction::sender(
        BusId::Memw,
        Multiplicity::Column(cols::READ_REGISTER1),
        smallvec![
            // old[0] = lo32 = RV1_0 + 2^16 * RV1_1
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::RV1_0,
                },
                LinearTerm::Column {
                    coefficient: 65536,
                    column: cols::RV1_1,
                },
            ]),
            // old[1] = hi32 = RV1_2
            BusValue::Packed {
                start_column: cols::RV1_2,
                packing: Packing::Direct,
            },
            // old[2..7] = 0 (unconstrained for registers)
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // is_register = 1
            BusValue::constant(1),
            // base_address[0] = 2 * rs1
            BusValue::linear(vec![LinearTerm::Column {
                coefficient: 2,
                column: cols::RS1,
            }]),
            // base_address[1] = 0
            BusValue::constant(0),
            // value[0..7] = same as old (rv1 as WL + 6 zeros)
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::RV1_0,
                },
                LinearTerm::Column {
                    coefficient: 65536,
                    column: cols::RV1_1,
                },
            ]),
            BusValue::Packed {
                start_column: cols::RV1_2,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // timestamp[0] = timestamp, timestamp[1] = 0
            BusValue::Packed {
                start_column: cols::TIMESTAMP,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
            // write2=1, write4=0, write8=0 (register access = 2 Words / 64 bits)
            BusValue::constant(1),
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    ));

    // -------------------------------------------------------------------------
    // M3: MEMW[rv2; 1, 2*rs2, rv2, timestamp+1, 0, 0, 1] | read_register2
    // -------------------------------------------------------------------------
    // Same pattern as M1 but with RV2 and timestamp+1
    interactions.push(BusInteraction::sender(
        BusId::Memw,
        Multiplicity::Column(cols::READ_REGISTER2),
        smallvec![
            // old[0] = lo32 = RV2_0 + 2^16 * RV2_1
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::RV2_0,
                },
                LinearTerm::Column {
                    coefficient: 65536,
                    column: cols::RV2_1,
                },
            ]),
            // old[1] = hi32 = RV2_2
            BusValue::Packed {
                start_column: cols::RV2_2,
                packing: Packing::Direct,
            },
            // old[2..7] = 0
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // is_register = 1
            BusValue::constant(1),
            // base_address[0] = 2 * rs2
            BusValue::linear(vec![LinearTerm::Column {
                coefficient: 2,
                column: cols::RS2,
            }]),
            // base_address[1] = 0
            BusValue::constant(0),
            // value[0..7] = rv2 as WL + 6 zeros
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::RV2_0,
                },
                LinearTerm::Column {
                    coefficient: 65536,
                    column: cols::RV2_1,
                },
            ]),
            BusValue::Packed {
                start_column: cols::RV2_2,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // timestamp[0] = timestamp + 1, timestamp[1] = 0
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::TIMESTAMP,
                },
                LinearTerm::Constant(1),
            ]),
            BusValue::constant(0),
            // write2=1, write4=0, write8=0 (register access = 2 Words / 64 bits)
            BusValue::constant(1),
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    ));

    // -------------------------------------------------------------------------
    // M5: MEMW[1, 2*rd, rvd, timestamp+2, 0, 0, 1] | write_register
    // -------------------------------------------------------------------------
    // Write to rd register via MEMW. Format: 16 elements (write, no old)
    // [is_register, base_addr[2], value[8], timestamp[2], write2, write4, write8]
    //
    // rvd is DWordWL (2 cols: Word, Word)
    // MEMW uses EXCLUSIVE encoding for write flags: (0, 0, 1) for 8-byte access
    // ("exactly N bytes" semantics, not "at least N bytes")
    interactions.push(BusInteraction::sender(
        BusId::Memw,
        Multiplicity::Column(cols::WRITE_REGISTER),
        smallvec![
            // is_register = 1
            BusValue::constant(1),
            // base_address[0] = 2 * rd
            BusValue::linear(vec![LinearTerm::Column {
                coefficient: 2,
                column: cols::RD,
            }]),
            // base_address[1] = 0
            BusValue::constant(0),
            // value[0] = rvd_lo = RVD_0
            BusValue::Packed {
                start_column: cols::RVD_0,
                packing: Packing::Direct,
            },
            // value[1] = rvd_hi = RVD_1
            BusValue::Packed {
                start_column: cols::RVD_1,
                packing: Packing::Direct,
            },
            // value[2..7] = 0
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // timestamp[0] = timestamp + 2, timestamp[1] = 0
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::TIMESTAMP,
                },
                LinearTerm::Constant(2),
            ]),
            BusValue::constant(0),
            // write2=1, write4=0, write8=0 (EXCLUSIVE encoding for 2-Word register access)
            BusValue::constant(1),
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    ));

    // -------------------------------------------------------------------------
    // M6: LOAD[rvd; base_address, timestamp, read2, read4, read8, signed] | LOAD
    // -------------------------------------------------------------------------
    // LOAD receiver expects: [res::DWordBL(2), base_address::DWordWL(2), timestamp::DWordWL(2), flags(3), signed(1)] = 10 elements
    //
    // For CPU LOAD:
    // - rvd (the loaded result) corresponds to res
    // - res (computed address = rv1 + imm) corresponds to base_address
    // - memory_Xbytes flags use EXCLUSIVE encoding per spec ("exactly N bytes")
    interactions.push(BusInteraction::sender(
        BusId::Load,
        Multiplicity::Column(cols::LOAD),
        smallvec![
            // rvd as DWordWL (2 words) - this is the loaded value
            // CPU RVD is already WL format
            BusValue::Packed {
                start_column: cols::RVD_0,
                packing: Packing::DWordWL,
            },
            // base_address = res (computed address) as DWordBL (8 bytes → 2 elements)
            BusValue::Packed {
                start_column: cols::RES[0],
                packing: Packing::DWordBL,
            },
            // timestamp as DWordWL: [timestamp, 0]
            BusValue::Packed {
                start_column: cols::TIMESTAMP,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
            // read flags: exclusive encoding (pass through directly)
            BusValue::Packed {
                start_column: cols::MEMORY_2BYTES,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::MEMORY_4BYTES,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::MEMORY_8BYTES,
                packing: Packing::Direct,
            },
            // signed flag
            BusValue::Packed {
                start_column: cols::SIGNED,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // M7: MEMW[0, res, rv2, timestamp+1, memory_2bytes, memory_4bytes, memory_8bytes] | STORE
    // -------------------------------------------------------------------------
    // Write to memory via MEMW. Format: 16 elements
    // [is_register, base_addr[2], value[8], timestamp[2], write2, write4, write8]
    //
    // For STORE:
    // - is_register = 0 (memory access)
    // - base_address = res (computed address = rv1 + imm)
    // - value = rv2 (the value being stored)
    interactions.push(BusInteraction::sender(
        BusId::Memw,
        Multiplicity::Column(cols::STORE),
        smallvec![
            // is_register = 0 (memory access)
            BusValue::constant(0),
            // base_address = res as DWordBL → 2 elements [lo32, hi32]
            BusValue::Packed {
                start_column: cols::RES[0],
                packing: Packing::DWordBL,
            },
            // value[0..7] = arg2 bytes (8 individual Direct elements)
            BusValue::Packed {
                start_column: cols::ARG2[0],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::ARG2[1],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::ARG2[2],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::ARG2[3],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::ARG2[4],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::ARG2[5],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::ARG2[6],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::ARG2[7],
                packing: Packing::Direct,
            },
            // timestamp[0] = timestamp + 1, timestamp[1] = 0
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::TIMESTAMP,
                },
                LinearTerm::Constant(1),
            ]),
            BusValue::constant(0),
            // write flags: exclusive encoding (pass through directly)
            BusValue::Packed {
                start_column: cols::MEMORY_2BYTES,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::MEMORY_4BYTES,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::MEMORY_8BYTES,
                packing: Packing::Direct,
            },
        ],
    ));

    // =========================================================================
    // Inline PC memory interactions (replaces CM54 MEMW interaction)
    // =========================================================================
    // CPU directly talks to the low-level memory bus for PC register (x255,
    // addresses 510 and 511), bypassing MEMW_R.

    // Non-padding multiplicity: sum of all ALU selector flags
    let non_pad_mult = Multiplicity::Linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::ADD,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::SUB,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::SLT,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::AND,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::OR,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::XOR,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::SHIFT,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::JALR,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::BEQ,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::BLT,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::LOAD,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::STORE,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::MUL,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::DIVREM,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::ECALL,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::EBREAK,
        },
    ]);

    // prev_ts_lo = timestamp - 3*(1 - pc_double_read) + 2^32 * borrow
    //            = timestamp - 3 + 3*pc_double_read + 2^32 * borrow
    let prev_ts_lo = BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::TIMESTAMP,
        },
        LinearTerm::Constant(-3),
        LinearTerm::Column {
            coefficient: 3,
            column: cols::PC_DOUBLE_READ,
        },
        LinearTerm::Column {
            coefficient: 1i64 << 32,
            column: cols::PREV_PC_TIMESTAMP_BORROW,
        },
    ]);

    // prev_ts_hi = 0 - borrow
    // The -1 cancels the +2^32 added to prev_ts_lo when borrow fires, keeping the
    // 64-bit timestamp correct: (prev_ts_hi * 2^32 + prev_ts_lo) = timestamp - 3.
    let prev_ts_hi = BusValue::linear(vec![LinearTerm::Column {
        coefficient: -1,
        column: cols::PREV_PC_TIMESTAMP_BORROW,
    }]);

    for i in 0..2u64 {
        // PC read (sender, +1): consume old token
        // memory[1, 510+i, 0, prev_ts_lo, prev_ts_hi, pc[i]]
        interactions.push(BusInteraction::sender(
            BusId::Memory,
            non_pad_mult.clone(),
            vec![
                BusValue::constant(1),
                BusValue::constant(510 + i),
                BusValue::constant(0),
                prev_ts_lo.clone(),
                prev_ts_hi.clone(),
                BusValue::Packed {
                    start_column: if i == 0 { cols::PC_0 } else { cols::PC_1 },
                    packing: Packing::Direct,
                },
            ],
        ));

        // PC write (receiver, -1): emit new token
        // memory[1, 510+i, 0, timestamp+1, 0, next_pc[i]]
        interactions.push(BusInteraction::receiver(
            BusId::Memory,
            non_pad_mult.clone(),
            vec![
                BusValue::constant(1),
                BusValue::constant(510 + i),
                BusValue::constant(0),
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::TIMESTAMP,
                    },
                    LinearTerm::Constant(1),
                ]),
                BusValue::constant(0),
                BusValue::Packed {
                    start_column: if i == 0 {
                        cols::NEXT_PC_0
                    } else {
                        cols::NEXT_PC_1
                    },
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // -------------------------------------------------------------------------
    // BRANCH interaction (for branch/jump target calculation)
    // -------------------------------------------------------------------------
    // CPU-CO68: BRANCH[next_pc; pc, imm, arg1::DWordWL, JALR] | branch_cond
    //
    // Sends to BRANCH table when branch_cond is true.
    // Bus signature: [next_pc[0], next_pc[1], pc[0], pc[1], offset[0], offset[1], register[0], register[1], JALR]
    // - next_pc: DWordWL (2 words) from NEXT_PC_0, NEXT_PC_1
    // - pc: DWordWL (2 words) from PC_0, PC_1
    // - offset: DWordWL (2 words) from IMM_0, IMM_1 (already sign-extended)
    // - register: DWordWL (2 words) - arg1 (DWordBL: 8 bytes) repacked as 2 words
    // - JALR: Bit flag
    interactions.push(BusInteraction::sender(
        BusId::Branch,
        Multiplicity::Column(cols::BRANCH_COND),
        smallvec![
            // next_pc[0] (Word) - low 32 bits
            BusValue::Packed {
                start_column: cols::NEXT_PC_0,
                packing: Packing::Direct,
            },
            // next_pc[1] (Word) - high 32 bits
            BusValue::Packed {
                start_column: cols::NEXT_PC_1,
                packing: Packing::Direct,
            },
            // pc[0] (Word)
            BusValue::Packed {
                start_column: cols::PC_0,
                packing: Packing::Direct,
            },
            // pc[1] (Word)
            BusValue::Packed {
                start_column: cols::PC_1,
                packing: Packing::Direct,
            },
            // offset[0] = imm[0] (Word) - low 32 bits of immediate
            BusValue::Packed {
                start_column: cols::IMM_0,
                packing: Packing::Direct,
            },
            // offset[1] = imm[1] (Word) - high 32 bits of immediate (sign-extended)
            BusValue::Packed {
                start_column: cols::IMM_1,
                packing: Packing::Direct,
            },
            // register[0] = arg1[0..4] repacked as Word
            // arg1_word0 = arg1[0] + 2^8*arg1[1] + 2^16*arg1[2] + 2^24*arg1[3]
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::ARG1[0],
                },
                LinearTerm::Column {
                    coefficient: 256,
                    column: cols::ARG1[1],
                },
                LinearTerm::Column {
                    coefficient: 65536,
                    column: cols::ARG1[2],
                },
                LinearTerm::Column {
                    coefficient: 16777216,
                    column: cols::ARG1[3],
                },
            ]),
            // register[1] = arg1[4..8] repacked as Word
            // arg1_word1 = arg1[4] + 2^8*arg1[5] + 2^16*arg1[6] + 2^24*arg1[7]
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::ARG1[4],
                },
                LinearTerm::Column {
                    coefficient: 256,
                    column: cols::ARG1[5],
                },
                LinearTerm::Column {
                    coefficient: 65536,
                    column: cols::ARG1[6],
                },
                LinearTerm::Column {
                    coefficient: 16777216,
                    column: cols::ARG1[7],
                },
            ]),
            // JALR flag
            BusValue::Packed {
                start_column: cols::JALR,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // Range checks (14 total):
    // CPU-CR29: IS_BYTE[rs1, rs2], CPU-CR30: IS_BYTE[rd, 0]
    // CPU-CR31.i: IS_BYTE[arg1[2i], arg1[2i+1]] (i=0..3)
    // CPU-CR32.i: IS_BYTE[arg2[2i], arg2[2i+1]] (i=0..3)
    // CPU-CR33.i: IS_BYTE[res[2i], res[2i+1]] (i=0..3)
    // -------------------------------------------------------------------------
    // RS1 and RS2 share one IS_BYTE check; RD uses 0 as the second argument.
    // ARG1/ARG2/RES are 8-byte little-endian values — adjacent byte pairs are
    // batched into IS_BYTE checks. Each pair sends two separate bus values
    // [lo, hi], so the LogUp fingerprint forces each byte to match individually
    // against the BITWISE table's X in [0,255] and Y in [0,255].
    // Every CPU row (including padding) sends with Multiplicity::One.
    interactions.push(BusInteraction::sender(
        BusId::IsByte,
        Multiplicity::One,
        smallvec![
            BusValue::Packed {
                start_column: cols::RS1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RS2,
                packing: Packing::Direct,
            },
        ],
    ));
    interactions.push(BusInteraction::sender(
        BusId::IsByte,
        Multiplicity::One,
        smallvec![
            BusValue::Packed {
                start_column: cols::RD,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
        ],
    ));
    for arr in [&cols::ARG1, &cols::ARG2, &cols::RES] {
        for i in 0..4 {
            interactions.push(BusInteraction::sender(
                BusId::IsByte,
                Multiplicity::One,
                smallvec![
                    BusValue::Packed {
                        start_column: arr[2 * i],
                        packing: Packing::Direct,
                    },
                    BusValue::Packed {
                        start_column: arr[2 * i + 1],
                        packing: Packing::Direct,
                    },
                ],
            ));
        }
    }

    // ECALL interaction (shared bus for HALT, COMMIT, and KECCAK)
    // -------------------------------------------------------------------------
    // multiplicity = ECALL (all ECALLs, each receiver matches on syscall number)
    interactions.push(BusInteraction::sender(
        BusId::Ecall,
        Multiplicity::Column(cols::ECALL),
        smallvec![
            BusValue::Packed {
                start_column: cols::TIMESTAMP,
                packing: Packing::Direct,
            },
            BusValue::constant(0), // timestamp_hi = 0 (CPU timestamps fit in u32)
            // cast(rv1, DWordWL)[0] = rv1_lo32 = RV1_0 + 2^16 * RV1_1
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::RV1_0,
                },
                LinearTerm::Column {
                    coefficient: 65536,
                    column: cols::RV1_1,
                },
            ]),
            // cast(rv1, DWordWL)[1] = rv1_hi32 = RV1_2
            BusValue::Packed {
                start_column: cols::RV1_2,
                packing: Packing::Direct,
            },
        ],
    ));

    interactions
}

// =========================================================================
// Constraints (placeholder - will be implemented in constraints/)
// =========================================================================

// The CPU constraints include:
// 1. Range checks (IS_BIT) for all bit flags - via templates
// 2. ALU dispatch constraints (conditional on selector flags)
// 3. Extension constraints (arg1, arg2, rvd from rv1, rv2, res)
// 4. Branch condition computation
// 5. next_pc computation (increment or branch target)
//
// These will be implemented using:
// - IsBitConstraint template for flags
// - AddConstraint template for ADD, SUB, next_pc
// - Custom constraints for extension logic
