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

    /// rv1[0]: Register rs1 value (Word - bits 0-31)
    pub const RV1_0: usize = 37;
    /// rv1[1]: Register rs1 value (Half - bits 32-47)
    pub const RV1_1: usize = 38;
    /// rv1[2]: Register rs1 value (Half - bits 48-63)
    pub const RV1_2: usize = 39;

    /// rv2[0]: Register rs2 value (Word - bits 0-31)
    pub const RV2_0: usize = 40;
    /// rv2[1]: Register rs2 value (Half - bits 32-47)
    pub const RV2_1: usize = 41;
    /// rv2[2]: Register rs2 value (Half - bits 48-63)
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
    /// For word instructions, arg1 is sign/zero extended from the lower 32 bits.
    pub fn compute_arg1(&self) -> u64 {
        if self.word_instr && self.signed {
            // Sign extend from 32 bits
            let lower = (self.rv1 & 0xFFFF_FFFF) as i32;
            lower as i64 as u64
        } else if self.word_instr {
            // Zero extend from 32 bits
            self.rv1 & 0xFFFF_FFFF
        } else {
            self.rv1
        }
    }

    /// Compute arg2 based on instruction type.
    ///
    /// For STORE/LOAD: uses rv2
    /// For BEQ/BLT: uses rv2
    /// Otherwise: uses imm (when rs2=0) or rv2
    pub fn compute_arg2(&self) -> u64 {
        if self.op_store || self.op_load {
            self.rv2
        } else if self.op_beq || self.op_blt {
            self.rv2
        } else {
            // For other ops, use imm (the spec assumes rs2=0 or imm=0)
            if self.rs2 == 0 {
                self.imm
            } else {
                self.rv2
            }
        }
    }

    /// Extract sign bit of a 32-bit word (bit 31).
    pub fn sign_bit_32(val: u64) -> bool {
        (val >> 31) & 1 == 1
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
    let num_rows = operations.len().next_power_of_two().max(2);
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
        data[base + cols::RVD_0] = FE::from(op.rvd & 0xFFFF_FFFF);
        data[base + cols::RVD_1] = FE::from(op.rvd >> 32);

        // Auxiliary: rv1 as DWordWHH [Word, Half, Half]
        data[base + cols::RV1_0] = FE::from(op.rv1 & 0xFFFF_FFFF);
        data[base + cols::RV1_1] = FE::from((op.rv1 >> 32) & 0xFFFF);
        data[base + cols::RV1_2] = FE::from((op.rv1 >> 48) & 0xFFFF);

        // Auxiliary: rv2 as DWordWHH [Word, Half, Half]
        data[base + cols::RV2_0] = FE::from(op.rv2 & 0xFFFF_FFFF);
        data[base + cols::RV2_1] = FE::from((op.rv2 >> 32) & 0xFFFF);
        data[base + cols::RV2_2] = FE::from((op.rv2 >> 48) & 0xFFFF);

        // Sign bits
        let rv1_sign_bit = CpuOperation::sign_bit_32(op.rv1);
        data[base + cols::RV1_SIGN_BIT] = FE::from(rv1_sign_bit as u64);

        // Compute and store arg1 as DWordBL (8 bytes)
        let arg1 = op.compute_arg1();
        for i in 0..8 {
            data[base + cols::ARG1[i]] = FE::from((arg1 >> (i * 8)) & 0xFF);
        }

        // Compute and store arg2
        let arg2 = op.compute_arg2();
        let arg2_sign_bit = CpuOperation::sign_bit_32(arg2);
        data[base + cols::ARG2_SIGN_BIT] = FE::from(arg2_sign_bit as u64);
        for i in 0..8 {
            data[base + cols::ARG2[i]] = FE::from((arg2 >> (i * 8)) & 0xFF);
        }

        // Result
        let res_sign_bit = CpuOperation::sign_bit_32(op.res);
        data[base + cols::RES_SIGN_BIT] = FE::from(res_sign_bit as u64);
        for i in 0..8 {
            data[base + cols::RES[i]] = FE::from((op.res >> (i * 8)) & 0xFF);
        }

        // Branch columns
        data[base + cols::IS_EQUAL] = FE::from(op.is_equal as u64);
        data[base + cols::BRANCH_COND] = FE::from(op.branch_cond as u64);
    }

    // Padding rows are already zeros (no operations, no multiplicities)

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Returns the bus interactions for the CPU table.
///
/// The CPU table is primarily a sender to other tables.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::new();

    // -------------------------------------------------------------------------
    // IS_BYTE range checks for register indices
    // -------------------------------------------------------------------------
    for &col in &[cols::RS1, cols::RS2, cols::RD] {
        interactions.push(BusInteraction::sender(
            BusId::IsByte,
            Multiplicity::One,
            vec![BusValue::Packed {
                start_column: col,
                packing: Packing::Direct,
            }],
        ));
    }

    // -------------------------------------------------------------------------
    // IS_BYTE range checks for arg1, arg2, res bytes
    // -------------------------------------------------------------------------
    for &col in &cols::ARG1 {
        interactions.push(BusInteraction::sender(
            BusId::IsByte,
            Multiplicity::One,
            vec![BusValue::Packed {
                start_column: col,
                packing: Packing::Direct,
            }],
        ));
    }

    for &col in &cols::ARG2 {
        interactions.push(BusInteraction::sender(
            BusId::IsByte,
            Multiplicity::One,
            vec![BusValue::Packed {
                start_column: col,
                packing: Packing::Direct,
            }],
        ));
    }

    for &col in &cols::RES {
        interactions.push(BusInteraction::sender(
            BusId::IsByte,
            Multiplicity::One,
            vec![BusValue::Packed {
                start_column: col,
                packing: Packing::Direct,
            }],
        ));
    }

    // -------------------------------------------------------------------------
    // LT interaction (for SLT, BLT)
    // -------------------------------------------------------------------------
    // For LT we need to send: lhs (DWordHHW), rhs (DWordHHW), signed, lt output
    // The CPU has arg1/arg2 as DWordBL, so we need to pack them
    // For now, using a simplified version with direct column refs
    // Full implementation would use Linear to repack bytes to HHW format
    interactions.push(BusInteraction::sender(
        BusId::Lt,
        Multiplicity::Column(cols::SLT), // Simplified: only SLT (would add BLT)
        vec![
            // For a proper implementation, these would be Linear combinations
            // that repack the bytes. For now, placeholder with first columns.
            BusValue::Packed {
                start_column: cols::ARG1_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::ARG2_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::SIGNED,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES_0,
                packing: Packing::Direct,
            },
        ],
    ));

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
    // MSB8 interaction for res sign bit (for word_instr)
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Msb8,
        Multiplicity::Column(cols::WORD_INSTR),
        vec![
            BusValue::Packed {
                start_column: cols::RES_3, // res[3] contains bits 24-31, including bit 31
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
    // For ZERO check, we need to check if all res bytes sum to zero
    // This requires a Linear bus value
    interactions.push(BusInteraction::sender(
        BusId::Zero,
        Multiplicity::Column(cols::BEQ),
        vec![
            // Sum of all res bytes (simplified - using res[0] for now)
            // Full implementation would use BusValue::Linear
            BusValue::Packed {
                start_column: cols::RES_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::IS_EQUAL,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // BRANCH interaction (for branch target calculation)
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Branch,
        Multiplicity::Column(cols::BRANCH_COND),
        vec![
            // pc
            BusValue::Packed {
                start_column: cols::PC_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::PC_1,
                packing: Packing::Direct,
            },
            // imm[0] (branch offset)
            BusValue::Packed {
                start_column: cols::IMM_0,
                packing: Packing::Direct,
            },
            // JALR flag (for register-based branch)
            BusValue::Packed {
                start_column: cols::JALR,
                packing: Packing::Direct,
            },
            // next_pc output
            BusValue::Packed {
                start_column: cols::NEXT_PC_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::NEXT_PC_1,
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
