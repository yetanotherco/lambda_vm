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
//! - `rv1_sign_bit`, `arg2_sign_bit`, `res_sign_bit`: Bit (for word instruction extension)
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
//! - IS_BYTE: range checks for rs1, rs2, rd, arg1[i], arg2[i], res[i]
//! - IS_BIT: range checks for flags (via templates)
//! - ADD: for ADD, LOAD, STORE, JALR operations
//! - SUB: for SUB, BEQ operations
//! - LT: for SLT, BLT operations
//! - AND_BYTE, OR_BYTE, XOR_BYTE: for bitwise operations (×8 each)
//! - SHIFT: for shift operations
//! - MUL: for multiplication
//! - DIVREM: for division/remainder
//! - MEMW: for register and memory access
//! - MSB16: for sign bit extraction
//! - MSB8: for 32-bit sign bit extraction
//! - ZERO: for equality check
//! - BRANCH: for branch target calculation
//! - ECALL: for system calls

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use executor::vm::{
    instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth},
    logs::Log,
};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

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

    /// write_register: Whether to write back to rd (Bit)
    pub const WRITE_REGISTER: usize = 6;
    /// memory_2bytes: Memory access is 2 bytes (Bit)
    pub const MEMORY_2BYTES: usize = 7;
    /// memory_4bytes: Memory access is 4 bytes (Bit)
    pub const MEMORY_4BYTES: usize = 8;
    /// memory_8bytes: Memory access is 8 bytes (Bit)
    pub const MEMORY_8BYTES: usize = 9;
    /// c_type_instruction: Instruction is 2 bytes (compressed) instead of 4 (Bit)
    pub const C_TYPE_INSTRUCTION: usize = 10;

    /// imm[0]: Immediate value (low word)
    pub const IMM_0: usize = 11;
    /// imm[1]: Immediate value (high word)
    pub const IMM_1: usize = 12;

    /// signed: Signed operation flag (Bit)
    pub const SIGNED: usize = 13;
    /// mp_selector: Multi-purpose selector (branch invert, shift direction, MUL variant)
    pub const MP_SELECTOR: usize = 14;
    /// muldiv_selector: Select MUL/DIV output variant
    pub const MULDIV_SELECTOR: usize = 15;
    /// word_instr: 32-bit word instruction (requires sign extension)
    pub const WORD_INSTR: usize = 16;

    // ALU selector flags (one-hot encoded)
    /// ADD operation
    pub const ADD: usize = 17;
    /// SUB operation
    pub const SUB: usize = 18;
    /// SLT (Set Less Than) operation
    pub const SLT: usize = 19;
    /// AND operation
    pub const AND: usize = 20;
    /// OR operation
    pub const OR: usize = 21;
    /// XOR operation
    pub const XOR: usize = 22;
    /// SHIFT operation
    pub const SHIFT: usize = 23;
    /// JALR (Jump And Link Register)
    pub const JALR: usize = 24;
    /// BEQ (Branch if Equal)
    pub const BEQ: usize = 25;
    /// BLT (Branch if Less Than)
    pub const BLT: usize = 26;
    /// LOAD operation
    pub const LOAD: usize = 27;
    /// STORE operation
    pub const STORE: usize = 28;
    /// MUL operation
    pub const MUL: usize = 29;
    /// DIVREM (Division/Remainder) operation
    pub const DIVREM: usize = 30;
    /// ECALL (Environment Call)
    pub const ECALL: usize = 31;
    /// EBREAK (Environment Break)
    pub const EBREAK: usize = 32;

    // -------------------------------------------------------------------------
    // Output columns
    // -------------------------------------------------------------------------

    /// next_pc[0]: Next program counter (low word)
    pub const NEXT_PC_0: usize = 33;
    /// next_pc[1]: Next program counter (high word)
    pub const NEXT_PC_1: usize = 34;

    /// rvd[0]: Value to write to destination register (low word)
    pub const RVD_0: usize = 35;
    /// rvd[1]: Value to write to destination register (high word)
    pub const RVD_1: usize = 36;

    // -------------------------------------------------------------------------
    // Auxiliary columns
    // -------------------------------------------------------------------------

    /// rv1[0]: Register rs1 value (Half - bits 0-15) [DWordWHH]
    pub const RV1_0: usize = 37;
    /// rv1[1]: Register rs1 value (Half - bits 16-31) [DWordWHH]
    pub const RV1_1: usize = 38;
    /// rv1[2]: Register rs1 value (Word - bits 32-63) [DWordWHH]
    pub const RV1_2: usize = 39;

    /// rv2[0]: Register rs2 value (Half - bits 0-15) [DWordWHH]
    pub const RV2_0: usize = 40;
    /// rv2[1]: Register rs2 value (Half - bits 16-31) [DWordWHH]
    pub const RV2_1: usize = 41;
    /// rv2[2]: Register rs2 value (Word - bits 32-63) [DWordWHH]
    pub const RV2_2: usize = 42;

    /// rv1_sign_bit: Sign bit of rv1 as 32-bit word (for word_instr extension)
    pub const RV1_SIGN_BIT: usize = 43;

    /// arg1[0..8]: Extended rv1 as DWordBL (8 bytes)
    pub const ARG1_0: usize = 44;
    pub const ARG1_1: usize = 45;
    pub const ARG1_2: usize = 46;
    pub const ARG1_3: usize = 47;
    pub const ARG1_4: usize = 48;
    pub const ARG1_5: usize = 49;
    pub const ARG1_6: usize = 50;
    pub const ARG1_7: usize = 51;

    /// arg2_sign_bit: Sign bit of arg2 as 32-bit word
    pub const ARG2_SIGN_BIT: usize = 52;

    /// arg2[0..8]: Extended rv2/imm as DWordBL (8 bytes)
    pub const ARG2_0: usize = 53;
    pub const ARG2_1: usize = 54;
    pub const ARG2_2: usize = 55;
    pub const ARG2_3: usize = 56;
    pub const ARG2_4: usize = 57;
    pub const ARG2_5: usize = 58;
    pub const ARG2_6: usize = 59;
    pub const ARG2_7: usize = 60;

    /// res_sign_bit: Sign bit of res as 32-bit word
    pub const RES_SIGN_BIT: usize = 61;

    /// res[0..8]: ALU result as DWordBL (8 bytes)
    pub const RES_0: usize = 62;
    pub const RES_1: usize = 63;
    pub const RES_2: usize = 64;
    pub const RES_3: usize = 65;
    pub const RES_4: usize = 66;
    pub const RES_5: usize = 67;
    pub const RES_6: usize = 68;
    pub const RES_7: usize = 69;

    /// is_equal: Whether rv1 == arg2 (for BEQ)
    pub const IS_EQUAL: usize = 70;

    /// branch_cond: Whether branch is taken
    pub const BRANCH_COND: usize = 71;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 72;

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
/// This is a high-level representation that will be converted to
/// the flat column format during trace generation.
#[derive(Debug, Clone, Default)]
pub struct CpuOperation {
    // Input (from DECODE)
    pub timestamp: u64,
    pub pc: u64,
    pub rs1: u8,
    pub rs2: u8,
    pub rd: u8,
    pub write_register: bool,
    pub memory_2bytes: bool,
    pub memory_4bytes: bool,
    pub memory_8bytes: bool,
    pub c_type_instruction: bool,
    pub imm: u64,
    pub signed: bool,
    pub mp_selector: bool,
    pub muldiv_selector: bool,
    pub word_instr: bool,

    // ALU selector (exactly one should be true)
    pub op_add: bool,
    pub op_sub: bool,
    pub op_slt: bool,
    pub op_and: bool,
    pub op_or: bool,
    pub op_xor: bool,
    pub op_shift: bool,
    pub op_jalr: bool,
    pub op_beq: bool,
    pub op_blt: bool,
    pub op_load: bool,
    pub op_store: bool,
    pub op_mul: bool,
    pub op_divrem: bool,
    pub op_ecall: bool,
    pub op_ebreak: bool,

    // Output
    pub next_pc: u64,
    pub rvd: u64,

    // Auxiliary (computed from register file and ALU)
    pub rv1: u64,
    pub rv2: u64,
    pub res: u64,
    pub is_equal: bool,
    pub branch_cond: bool,
}

impl CpuOperation {
    /// Creates a new CPU operation with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute arg1 from rv1 based on word_instr and signed flags.
    ///
    /// Per spec constraint: arg1[4:] = rv1[2] * (1 - word_instr) + (2^32 - 1) * rv1_sign_bit * signed
    ///
    /// For 64-bit instructions: pass through full rv1
    /// For unsigned word instructions: zero-extend from 32 bits
    /// For signed word instructions: sign-extend from 32 bits
    pub fn compute_arg1(&self) -> u64 {
        if self.word_instr {
            let lower_32 = self.rv1 & 0xFFFF_FFFF;
            if self.signed && Self::sign_bit_32(self.rv1) {
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

    /// Compute arg2 based on instruction type.
    ///
    /// Per spec constraint for arg2[4:]:
    /// (1-STORE-LOAD) * ((1-word_instr)*rv2[2] + signed*arg2_sign_bit*(2^32-1)) + (1-BEQ-BLT)*imm[1]
    ///
    /// For LOAD/STORE: uses imm (full 64-bit, for address calculation)
    /// For BEQ/BLT: uses rv2 (full 64-bit, comparing register values)
    /// Otherwise: uses imm (when rs2=0) or rv2, with sign extension for signed word instructions
    pub fn compute_arg2(&self) -> u64 {
        if self.op_load || self.op_store {
            // LOAD/STORE: address = rv1 + imm, use full imm
            self.imm
        } else if self.op_beq || self.op_blt {
            // BEQ/BLT: compare rv1 vs rv2, use full rv2
            self.rv2
        } else {
            // For other ops, use imm (when rs2=0) or rv2
            let base = if self.rs2 == 0 { self.imm } else { self.rv2 };

            // For word instructions, apply sign/zero extension based on signed flag
            if self.word_instr {
                let lower_32 = base & 0xFFFF_FFFF;
                if self.signed && Self::sign_bit_32(base) {
                    // Sign extend: set upper 32 bits to all 1s
                    lower_32 | (0xFFFF_FFFF_u64 << 32)
                } else {
                    // Zero extend: upper 32 bits are 0
                    lower_32
                }
            } else {
                base
            }
        }
    }

    /// Extract sign bit of a 32-bit word (bit 31).
    pub fn sign_bit_32(val: u64) -> bool {
        (val >> 31) & 1 == 1
    }

    /// Compute rvd (destination register value) based on res and word_instr.
    ///
    /// According to spec constraints:
    /// - rvd[0] = res[:4] (lower 32 bits of res)
    /// - rvd[1] = (1 - word_instr) * res[4:] + res_sign_bit * (2^32 - 1)
    ///
    /// For LOAD: rvd comes from the executor (loaded value), not this method.
    /// For all other operations: rvd is computed from res with sign extension.
    pub fn compute_rvd(&self) -> u64 {
        let res = self.compute_res();
        let res_lo = res & 0xFFFF_FFFF;

        if self.word_instr {
            // Sign extend from 32 bits
            let res_sign_bit = Self::sign_bit_32(res);
            if res_sign_bit {
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
    /// For SLT: res = 0 or 1 (comparison result from executor)
    /// For other operations: uses the executor's result (self.res)
    ///
    /// This ensures the ADD/SUB constraints are satisfied.
    /// The rvd column holds the actual sign-extended result for word instructions.
    pub fn compute_res(&self) -> u64 {
        let arg1 = self.compute_arg1();
        let arg2 = self.compute_arg2();

        if self.op_add || self.op_load || self.op_store {
            // ADD constraint: arg1 + arg2 = res
            // For ADD: computes arithmetic result
            // For LOAD/STORE: computes memory address (rv1 + imm)
            arg1.wrapping_add(arg2)
        } else if self.op_sub {
            // SUB constraint checks: res + arg2 = arg1, so res = arg1 - arg2
            arg1.wrapping_sub(arg2)
        } else {
            // For SLT and other operations, use the executor's result
            // SLT res is 0 or 1, verified by SltResZeroConstraint
            self.res
        }
    }

    /// Collects Bitwise table lookups generated by this CPU operation.
    ///
    /// Returns a vector of (BitwiseLookup, x, y, z) tuples to pass to
    /// `bitwise::update_multiplicities`.
    pub fn collect_bitwise_lookups(&self) -> Vec<(super::bitwise::BitwiseLookup, u8, u8, u8)> {
        use super::bitwise::BitwiseLookup;
        let mut lookups = Vec::new();

        // MSB16 lookups for sign bit extraction (when word_instr=1)
        if self.word_instr {
            // rv1[1] is bits 16-31, extract as halfword for MSB16 lookup
            let rv1_half = ((self.rv1 >> 16) & 0xFFFF) as u16;
            let x = (rv1_half & 0xFF) as u8;
            let y = ((rv1_half >> 8) & 0xFF) as u8;
            lookups.push((BitwiseLookup::Msb16, x, y, 0));

            // rv2[1] for arg2_sign_bit
            let rv2_half = ((self.rv2 >> 16) & 0xFFFF) as u16;
            let x = (rv2_half & 0xFF) as u8;
            let y = ((rv2_half >> 8) & 0xFF) as u8;
            lookups.push((BitwiseLookup::Msb16, x, y, 0));

            // res[3] for res_sign_bit (MSB8 on byte at bits 24-31)
            let res_byte = ((self.res >> 24) & 0xFF) as u8;
            lookups.push((BitwiseLookup::Msb8, res_byte, 0, 0));
        }

        // ZERO lookup for is_equal (when BEQ=1)
        if self.op_beq {
            // Sum of all result bytes
            let mut sum: u64 = 0;
            for i in 0..8 {
                sum += (self.res >> (i * 8)) & 0xFF;
            }
            // Sum fits in 16 bits (max 8 * 255 = 2040)
            let x = (sum & 0xFF) as u8;
            let y = ((sum >> 8) & 0xFF) as u8;
            lookups.push((BitwiseLookup::Zero, x, y, 0));
        }

        // AND/OR/XOR lookups (×8 each for each byte)
        let arg1 = self.compute_arg1();
        let arg2 = self.compute_arg2();

        if self.op_and {
            for i in 0..8 {
                let a = ((arg1 >> (i * 8)) & 0xFF) as u8;
                let b = ((arg2 >> (i * 8)) & 0xFF) as u8;
                lookups.push((BitwiseLookup::AndByte, a, b, 0));
            }
        }

        if self.op_or {
            for i in 0..8 {
                let a = ((arg1 >> (i * 8)) & 0xFF) as u8;
                let b = ((arg2 >> (i * 8)) & 0xFF) as u8;
                lookups.push((BitwiseLookup::OrByte, a, b, 0));
            }
        }

        if self.op_xor {
            for i in 0..8 {
                let a = ((arg1 >> (i * 8)) & 0xFF) as u8;
                let b = ((arg2 >> (i * 8)) & 0xFF) as u8;
                lookups.push((BitwiseLookup::XorByte, a, b, 0));
            }
        }

        lookups
    }

    /// Creates a CpuOperation from an executor Log.
    ///
    /// This is the main integration point between the executor and the prover.
    pub fn from_log(log: &Log, timestamp: u64) -> Self {
        let mut op = Self {
            timestamp,
            pc: log.current_pc,
            next_pc: log.next_pc,
            rv1: log.src1_val,
            rv2: log.src2_val,
            rvd: log.dst_val,
            res: log.dst_val, // Default: result is destination value
            ..Default::default()
        };

        match log.instruction {
            Instruction::Arith {
                dst,
                src1,
                src2,
                op: arith_op,
            } => {
                op.rd = dst as u8;
                op.rs1 = src1 as u8;
                op.rs2 = src2 as u8;
                if dst != 0 {
                    op.write_register = true;
                }
                Self::set_arith_op(&mut op, arith_op, false);
            }

            Instruction::ArithImm {
                dst,
                src,
                imm,
                op: arith_op,
            } => {
                op.rd = dst as u8;
                op.rs1 = src as u8;
                op.rs2 = 0;
                op.imm = imm as i64 as u64; // Sign extend
                if dst != 0 {
                    op.write_register = true;
                }
                Self::set_arith_op(&mut op, arith_op, false);
            }

            Instruction::ArithW {
                dst,
                src1,
                src2,
                op: arith_op,
            } => {
                op.rd = dst as u8;
                op.rs1 = src1 as u8;
                op.rs2 = src2 as u8;
                op.word_instr = true;
                if dst != 0 {
                    op.write_register = true;
                }
                Self::set_arith_op(&mut op, arith_op, true);
            }

            Instruction::ArithImmW {
                dst,
                src,
                imm,
                op: arith_op,
            } => {
                op.rd = dst as u8;
                op.rs1 = src as u8;
                op.rs2 = 0;
                op.imm = imm as i64 as u64; // Sign extend
                op.word_instr = true;
                if dst != 0 {
                    op.write_register = true;
                }
                Self::set_arith_op(&mut op, arith_op, true);
            }

            Instruction::JumpAndLink { dst, offset } => {
                op.op_jalr = true;
                op.rd = dst as u8;
                op.imm = offset as i64 as u64;
                op.branch_cond = true;
                if dst != 0 {
                    op.write_register = true;
                }
            }

            Instruction::JumpAndLinkRegister { base, dst, offset } => {
                op.op_jalr = true;
                op.rd = dst as u8;
                op.rs1 = base as u8;
                op.imm = offset as i64 as u64;
                op.branch_cond = true;
                if dst != 0 {
                    op.write_register = true;
                }
            }

            Instruction::Store {
                src,
                offset,
                base,
                width,
            } => {
                op.op_store = true;
                op.rs1 = base as u8;
                op.rs2 = src as u8;
                op.imm = offset as i64 as u64;
                // res is the memory address = base + offset
                op.res = (log.src1_val as i64 + offset as i64) as u64;
                Self::set_memory_width(&mut op, width);
            }

            Instruction::Load {
                dst,
                offset,
                base,
                width,
            } => {
                op.op_load = true;
                op.rd = dst as u8;
                op.rs1 = base as u8;
                op.imm = offset as i64 as u64;
                // res is the memory address = base + offset
                op.res = (log.src1_val as i64 + offset as i64) as u64;
                if dst != 0 {
                    op.write_register = true;
                }
                Self::set_memory_width(&mut op, width);
                // Set signed flag for sign-extending loads
                match width {
                    LoadStoreWidth::Byte | LoadStoreWidth::Half | LoadStoreWidth::Word => {
                        op.signed = true;
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
                op.rs1 = src1 as u8;
                op.rs2 = src2 as u8;
                op.imm = offset as i64 as u64;
                op.is_equal = log.src1_val == log.src2_val;

                match cond {
                    Comparison::Equal => {
                        op.op_beq = true;
                        // BEQ uses SUB constraint: res = arg1 - arg2
                        op.res = log.src1_val.wrapping_sub(log.src2_val);
                        op.branch_cond = log.src1_val == log.src2_val;
                    }
                    Comparison::NotEqual => {
                        op.op_beq = true;
                        op.mp_selector = true; // Inverted
                        // BNE uses SUB constraint: res = arg1 - arg2
                        op.res = log.src1_val.wrapping_sub(log.src2_val);
                        op.branch_cond = log.src1_val != log.src2_val;
                    }
                    Comparison::LessThan => {
                        op.op_blt = true;
                        op.signed = true;
                        // BLT uses LT table: res[0] = comparison result, res[1..7] = 0
                        let lt_result = (log.src1_val as i64) < (log.src2_val as i64);
                        op.res = lt_result as u64;
                        op.branch_cond = lt_result;
                    }
                    Comparison::LessThanUnsigned => {
                        op.op_blt = true;
                        // BLTU uses LT table: res[0] = comparison result, res[1..7] = 0
                        let lt_result = log.src1_val < log.src2_val;
                        op.res = lt_result as u64;
                        op.branch_cond = lt_result;
                    }
                    Comparison::GreaterOrEqual => {
                        op.op_blt = true;
                        op.signed = true;
                        op.mp_selector = true; // Inverted
                        // BGE uses LT table with inversion: res[0] = arg1 < arg2
                        let lt_result = (log.src1_val as i64) < (log.src2_val as i64);
                        op.res = lt_result as u64;
                        op.branch_cond = !lt_result; // Inverted: >= means NOT <
                    }
                    Comparison::GreaterOrEqualUnsigned => {
                        op.op_blt = true;
                        op.mp_selector = true; // Inverted
                        // BGEU uses LT table with inversion: res[0] = arg1 < arg2
                        let lt_result = log.src1_val < log.src2_val;
                        op.res = lt_result as u64;
                        op.branch_cond = !lt_result; // Inverted: >= means NOT <
                    }
                }
            }

            Instruction::LoadUpperImm { dst, imm } => {
                op.op_add = true;
                op.rd = dst as u8;
                op.rs1 = 0;
                op.rs2 = 0;
                // LUI immediate must be sign-extended to 64 bits (matches executor)
                op.imm = (imm as i32) as i64 as u64;
                if dst != 0 {
                    op.write_register = true;
                }
            }

            Instruction::AddUpperImmToPc { dst, imm } => {
                op.op_add = true;
                op.rd = dst as u8;
                // AUIPC immediate must be sign-extended to 64 bits (matches executor)
                op.imm = (imm as i32) as i64 as u64;
                op.rv1 = log.current_pc;
                if dst != 0 {
                    op.write_register = true;
                }
            }

            Instruction::CSR { .. } => {
                // CSR instructions not yet supported in prover
                // TODO: Add CSR support
            }

            Instruction::EcallEbreak => {
                // Determine if ECALL or EBREAK based on context
                // For now, default to ECALL
                op.op_ecall = true;
            }
        }

        op
    }

    /// Helper to set ALU operation flags based on ArithOp.
    fn set_arith_op(op: &mut Self, arith_op: ArithOp, is_word: bool) {
        match arith_op {
            ArithOp::Add => {
                op.op_add = true;
            }
            ArithOp::Sub => {
                op.op_sub = true;
            }
            ArithOp::Xor => op.op_xor = true,
            ArithOp::Or => op.op_or = true,
            ArithOp::And => op.op_and = true,
            ArithOp::ShiftLeftLogical => {
                op.op_shift = true;
                // mp_selector = 0 for left shift
            }
            ArithOp::ShiftRightLogical => {
                op.op_shift = true;
                op.mp_selector = true; // mp_selector = 1 for right shift
            }
            ArithOp::ShiftRightArith => {
                op.op_shift = true;
                op.mp_selector = true;
                op.signed = true;
            }
            ArithOp::SetLessThan => {
                op.op_slt = true;
                op.signed = true;
            }
            ArithOp::SetLessThanU => {
                op.op_slt = true;
            }
            ArithOp::Mul => {
                op.op_mul = true;
                op.mp_selector = true;
                if !is_word {
                    op.signed = true;
                }
            }
            ArithOp::MulHigh => {
                op.op_mul = true;
                op.muldiv_selector = true;
                op.signed = true;
            }
            ArithOp::MulHighSignedUnsigned => {
                op.op_mul = true;
                op.muldiv_selector = true;
                op.mp_selector = true;
                op.signed = true;
            }
            ArithOp::MulHighUnsigned => {
                op.op_mul = true;
                op.muldiv_selector = true;
            }
            ArithOp::Div => {
                op.op_divrem = true;
                op.signed = true;
            }
            ArithOp::DivUnsigned => {
                op.op_divrem = true;
            }
            ArithOp::Remainder => {
                op.op_divrem = true;
                op.muldiv_selector = true;
                op.signed = true;
            }
            ArithOp::RemainderUnsigned => {
                op.op_divrem = true;
                op.muldiv_selector = true;
            }
        }
    }

    /// Helper to set memory width flags.
    fn set_memory_width(op: &mut Self, width: LoadStoreWidth) {
        match width {
            LoadStoreWidth::Byte | LoadStoreWidth::ByteUnsigned => {
                // 1 byte - no flags set
            }
            LoadStoreWidth::Half | LoadStoreWidth::HalfUnsigned => {
                op.memory_2bytes = true;
            }
            LoadStoreWidth::Word | LoadStoreWidth::WordUnsigned => {
                op.memory_2bytes = true;
                op.memory_4bytes = true;
            }
            LoadStoreWidth::DoubleWord => {
                op.memory_2bytes = true;
                op.memory_4bytes = true;
                op.memory_8bytes = true;
            }
        }
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the CPU trace table from a list of operations.
///
/// Each operation becomes one row in the table. The table is padded
/// to the next power of 2 (minimum 4 rows) with zero-filled rows.
pub fn generate_cpu_trace(
    operations: &[CpuOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = operations.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, op) in operations.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;

        // Input columns
        data[base + cols::TIMESTAMP] = FE::from(op.timestamp);
        data[base + cols::PC_0] = FE::from(op.pc & 0xFFFF_FFFF);
        data[base + cols::PC_1] = FE::from(op.pc >> 32);
        data[base + cols::RS1] = FE::from(op.rs1 as u64);
        data[base + cols::RS2] = FE::from(op.rs2 as u64);
        data[base + cols::RD] = FE::from(op.rd as u64);
        data[base + cols::WRITE_REGISTER] = FE::from(op.write_register as u64);
        data[base + cols::MEMORY_2BYTES] = FE::from(op.memory_2bytes as u64);
        data[base + cols::MEMORY_4BYTES] = FE::from(op.memory_4bytes as u64);
        data[base + cols::MEMORY_8BYTES] = FE::from(op.memory_8bytes as u64);
        data[base + cols::C_TYPE_INSTRUCTION] = FE::from(op.c_type_instruction as u64);
        data[base + cols::IMM_0] = FE::from(op.imm & 0xFFFF_FFFF);
        data[base + cols::IMM_1] = FE::from(op.imm >> 32);
        data[base + cols::SIGNED] = FE::from(op.signed as u64);
        data[base + cols::MP_SELECTOR] = FE::from(op.mp_selector as u64);
        data[base + cols::MULDIV_SELECTOR] = FE::from(op.muldiv_selector as u64);
        data[base + cols::WORD_INSTR] = FE::from(op.word_instr as u64);

        // ALU selector flags
        data[base + cols::ADD] = FE::from(op.op_add as u64);
        data[base + cols::SUB] = FE::from(op.op_sub as u64);
        data[base + cols::SLT] = FE::from(op.op_slt as u64);
        data[base + cols::AND] = FE::from(op.op_and as u64);
        data[base + cols::OR] = FE::from(op.op_or as u64);
        data[base + cols::XOR] = FE::from(op.op_xor as u64);
        data[base + cols::SHIFT] = FE::from(op.op_shift as u64);
        data[base + cols::JALR] = FE::from(op.op_jalr as u64);
        data[base + cols::BEQ] = FE::from(op.op_beq as u64);
        data[base + cols::BLT] = FE::from(op.op_blt as u64);
        data[base + cols::LOAD] = FE::from(op.op_load as u64);
        data[base + cols::STORE] = FE::from(op.op_store as u64);
        data[base + cols::MUL] = FE::from(op.op_mul as u64);
        data[base + cols::DIVREM] = FE::from(op.op_divrem as u64);
        data[base + cols::ECALL] = FE::from(op.op_ecall as u64);
        data[base + cols::EBREAK] = FE::from(op.op_ebreak as u64);

        // Output columns
        data[base + cols::NEXT_PC_0] = FE::from(op.next_pc & 0xFFFF_FFFF);
        data[base + cols::NEXT_PC_1] = FE::from(op.next_pc >> 32);

        // rvd: For LOAD, use the executor's loaded value (op.rvd).
        // For all other operations (including STORE), compute from res with sign extension.
        // This satisfies spec constraint: (1-LOAD) * (rvd - res_extended) = 0
        let rvd = if op.op_load {
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

        // Sign bits - only set when word_instr=1, per spec constraint ext_sign_bits
        // The constraint enforces: (rv1_sign_bit + arg2_sign_bit + res_sign_bit) * (1 - word_instr) = 0
        let rv1_sign_bit = op.word_instr && CpuOperation::sign_bit_32(op.rv1);
        data[base + cols::RV1_SIGN_BIT] = FE::from(rv1_sign_bit as u64);

        // Compute and store arg1 as DWordBL (8 bytes)
        let arg1 = op.compute_arg1();
        for i in 0..8 {
            data[base + cols::ARG1[i]] = FE::from((arg1 >> (i * 8)) & 0xFF);
        }

        // Compute and store arg2
        let arg2 = op.compute_arg2();
        let arg2_sign_bit = op.word_instr && CpuOperation::sign_bit_32(arg2);
        data[base + cols::ARG2_SIGN_BIT] = FE::from(arg2_sign_bit as u64);
        for i in 0..8 {
            data[base + cols::ARG2[i]] = FE::from((arg2 >> (i * 8)) & 0xFF);
        }

        // Result - computed from arg1/arg2 for ADD/SUB to satisfy constraints
        let res = op.compute_res();
        let res_sign_bit = op.word_instr && CpuOperation::sign_bit_32(res);
        data[base + cols::RES_SIGN_BIT] = FE::from(res_sign_bit as u64);
        for i in 0..8 {
            data[base + cols::RES[i]] = FE::from((res >> (i * 8)) & 0xFF);
        }

        // Branch columns
        data[base + cols::IS_EQUAL] = FE::from(op.is_equal as u64);
        data[base + cols::BRANCH_COND] = FE::from(op.branch_cond as u64);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Returns the bus interactions for the CPU table.
///
/// The CPU table sends to:
/// - AND_BYTE, OR_BYTE, XOR_BYTE: for bitwise operations (×8 each)
///
/// Note: LT interaction is TODO - needs proper DWordHHW packing to match LT table receiver.
/// Note: IS_BYTE, MSB8, ZERO, BRANCH interactions are TODO for later.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::new();

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
            vec![
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
            vec![
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
            vec![
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
    // MSB16 interactions for sign bit extraction
    // -------------------------------------------------------------------------
    // MSB16[rv1[1]] -> rv1_sign_bit, multiplicity = word_instr
    // rv1[1] is a Half (bits 16-31), containing the sign bit at position 31
    interactions.push(BusInteraction::sender(
        BusId::Msb16,
        Multiplicity::Column(cols::WORD_INSTR),
        vec![
            BusValue::Packed {
                start_column: cols::RV1_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RV1_SIGN_BIT,
                packing: Packing::Direct,
            },
        ],
    ));

    // MSB16[rv2[1]] -> arg2_sign_bit, multiplicity = word_instr
    interactions.push(BusInteraction::sender(
        BusId::Msb16,
        Multiplicity::Column(cols::WORD_INSTR),
        vec![
            BusValue::Packed {
                start_column: cols::RV2_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::ARG2_SIGN_BIT,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // MSB8 interaction for res sign bit extraction
    // -------------------------------------------------------------------------
    // MSB8[res[3]] -> res_sign_bit, multiplicity = word_instr
    // res[3] is the byte at bits 24-31, containing the sign bit at position 31
    interactions.push(BusInteraction::sender(
        BusId::Msb8,
        Multiplicity::Column(cols::WORD_INSTR),
        vec![
            BusValue::Packed {
                start_column: cols::RES[3],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES_SIGN_BIT,
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
        vec![
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
    // LT[arg1::DWordHHW, arg2::DWordHHW, signed] -> res[0]
    // multiplicity = SLT + BLT
    //
    // DWordHHW format: [Word(0-31), Half(32-47), Half(48-63)]
    // arg1/arg2 are DWordBL (8 bytes), need to repack:
    //   Word = byte[0] + 2^8*byte[1] + 2^16*byte[2] + 2^24*byte[3]
    //   Half1 = byte[4] + 2^8*byte[5]
    //   Half2 = byte[6] + 2^8*byte[7]
    interactions.push(BusInteraction::sender(
        BusId::Lt,
        // SLT + BLT using Multiplicity::Sum
        Multiplicity::Sum(cols::SLT, cols::BLT),
        vec![
            // arg1[0]: Word (lower 32 bits)
            BusValue::linear(vec![
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::ARG1[0],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1 << 8,
                    column: cols::ARG1[1],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1 << 16,
                    column: cols::ARG1[2],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1 << 24,
                    column: cols::ARG1[3],
                },
            ]),
            // arg1[1]: Half (bits 32-47)
            BusValue::linear(vec![
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::ARG1[4],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1 << 8,
                    column: cols::ARG1[5],
                },
            ]),
            // arg1[2]: Half (bits 48-63)
            BusValue::linear(vec![
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::ARG1[6],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1 << 8,
                    column: cols::ARG1[7],
                },
            ]),
            // arg2[0]: Word (lower 32 bits)
            BusValue::linear(vec![
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::ARG2[0],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1 << 8,
                    column: cols::ARG2[1],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1 << 16,
                    column: cols::ARG2[2],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1 << 24,
                    column: cols::ARG2[3],
                },
            ]),
            // arg2[1]: Half (bits 32-47)
            BusValue::linear(vec![
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::ARG2[4],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1 << 8,
                    column: cols::ARG2[5],
                },
            ]),
            // arg2[2]: Half (bits 48-63)
            BusValue::linear(vec![
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::ARG2[6],
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 1 << 8,
                    column: cols::ARG2[7],
                },
            ]),
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

    interactions
}

// =========================================================================
// Constraints (placeholder - will be implemented in constraints64/)
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
