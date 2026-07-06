//! CPU32 table.
//!
//! Handles all 32-bit word (`*W`) instructions delegated by the main CPU via
//! the `CPU32[timestamp, pc, half_instruction_length]` interaction. All `*W`
//! instructions are ALU-only, so there is no BRANCH/MEMORY/ECALL path. The chip
//! does its own DECODE lookup, reads the registers, sign-extends the inputs to
//! 64 bits, runs the ALU (or the ADD/SUB fast-path) and sign-extends the 32-bit
//! result back to 64 bits before writing `rd`.
//!
//! Spec: `spec/src/cpu32.toml`.
//!
//! ## Sign extension
//! `*W` instructions operate on the low 32 bits of the registers and produce a
//! sign-extended 64-bit result. `signed` (extracted from `alu_flags` bit 5)
//! selects sign- vs zero-extension of the inputs; the output `rvd` is always
//! sign-extended (RV64 `*W` semantics).
//!
//! Register reads use the cast-to-`DWordWL` encoding.

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{
    BusId, FE, GoldilocksExtension, GoldilocksField, SHIFT_16, VmTable, alu_op,
    packed_decode_shrunk,
};
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

use crate::constraints::templates::{AddOperand, emit_add_pair, emit_is_bit};

// =========================================================================
// Column indices for CPU32 table
// =========================================================================

/// Column definitions for the CPU32 table.
pub mod cols {
    // Inputs (from the CPU32 interaction)
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;
    pub const PC_0: usize = 2;
    pub const PC_1: usize = 3;

    // rs1 read
    pub const RS1: usize = 4;
    pub const READ_REGISTER1: usize = 5;
    // rv1: DWordWHH = [Half, Half, Word] (low word as 2 halves + high word)
    pub const RV1_0: usize = 6;
    pub const RV1_1: usize = 7;
    pub const RV1_2: usize = 8;
    pub const RV1_SIGN: usize = 9;
    // arg1: DWordWL = sign/zero-extended low word of rv1
    pub const ARG1_0: usize = 10;
    pub const ARG1_1: usize = 11;

    // rs2 read
    pub const RS2: usize = 12;
    pub const READ_REGISTER2: usize = 13;
    pub const RV2_0: usize = 14;
    pub const RV2_1: usize = 15;
    pub const RV2_2: usize = 16;
    pub const RV2_SIGN: usize = 17;
    // imm: DWordWL (fully sign-extended immediate)
    pub const IMM_0: usize = 18;
    pub const IMM_1: usize = 19;
    // arg2: DWordWL = ext(rv2) or imm
    pub const ARG2_0: usize = 20;
    pub const ARG2_1: usize = 21;

    // res: DWordHL = ALU result (4 halves)
    pub const RES_0: usize = 22;
    pub const RES_1: usize = 23;
    pub const RES_2: usize = 24;
    pub const RES_3: usize = 25;
    pub const RES_SIGN: usize = 26;

    // rd write
    pub const RD: usize = 27;
    pub const WRITE_REGISTER: usize = 28;
    // rvd: DWordWL = sign-extended low word of res
    pub const RVD_0: usize = 29;
    pub const RVD_1: usize = 30;

    // ALU control
    pub const ALU: usize = 31;
    pub const ALU_FLAGS: usize = 32;
    pub const ADD: usize = 33;
    pub const SUB: usize = 34;
    /// half the byte length (1 or 2); real length = `2 * half`.
    pub const HALF_INSTRUCTION_LENGTH: usize = 35;
    /// signed: extracted from `alu_flags` bit 5 (via BYTE_ALU[AND, 32, alu_flags]).
    pub const SIGNED: usize = 36;

    /// μ: multiplicity
    pub const MU: usize = 37;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 38;
}

/// Mask selecting `signed` from the `alu_flags` byte (bit 5).
const SIGNED_MASK: u64 = 1 << packed_decode_shrunk::ALU_FLAGS_SIGNED;
/// `2^32 - 1`, the sign-extension fill for the high word.
const HI_FILL: u64 = 0xFFFF_FFFF;

// =========================================================================
// Trace generation
// =========================================================================

/// A single CPU32 operation (a delegated `*W` instruction).
///
/// `res` is the raw 64-bit ALU result (computed by the executor); `rvd` is
/// derived from it by sign-extending the low 32 bits.
#[derive(Debug, Clone, Default, Hash, PartialEq, Eq)]
pub struct Cpu32Operation {
    pub timestamp: u64,
    pub pc: u64,
    pub rs1: u8,
    pub read_register1: bool,
    pub rv1: u64,
    pub rs2: u8,
    pub read_register2: bool,
    pub rv2: u64,
    pub imm: u64,
    /// Raw 64-bit ALU result.
    pub res: u64,
    pub rd: u8,
    pub write_register: bool,
    pub alu: bool,
    pub alu_flags: u8,
    pub add: bool,
    pub sub: bool,
    pub half_instruction_length: u8,
}

/// Derived auxiliary values for a CPU32 row.
pub struct Cpu32Aux {
    pub signed: bool,
    pub rv1_sign: bool,
    pub arg1: u64,
    pub rv2_sign: bool,
    pub arg2: u64,
    pub res_sign: bool,
    pub rvd: u64,
}

impl Cpu32Operation {
    /// Whether the inputs are sign-extended (`alu_flags` bit 5).
    pub fn signed(&self) -> bool {
        (self.alu_flags as u64 & SIGNED_MASK) != 0
    }

    /// Computes the derived auxiliary values (signs, extended args, rvd).
    pub fn compute_aux(&self) -> Cpu32Aux {
        let signed = self.signed();

        // Sign bits via `SIGN(·, gate)`: `rv1`/`rv2` are gated by `signed` (the
        // column is the MSB only when sign-extending, else 0 — matching the spec's
        // `SIGN(rv·[1], signed)`); `res` is gated by `μ` (the `*W` result is always
        // sign-extended).
        let rv1_sign = signed && (self.rv1 >> 31) & 1 == 1;
        let rv2_sign = signed && (self.rv2 >> 31) & 1 == 1;
        let res_sign = (self.res >> 31) & 1 == 1;

        // arg1 = ext(rv1 low word): low word as-is, high word = (2^32-1) when
        // rv1_sign (which already folds in `signed`), else 0.
        let arg1_hi = if rv1_sign { HI_FILL } else { 0 };
        let arg1 = (self.rv1 & 0xFFFF_FFFF) | (arg1_hi << 32);

        // arg2 = ext(rv2 low word) + imm. By the decoding assumption exactly one
        // of rv2 / imm is non-zero, so the per-word sums never overflow.
        let arg2_lo = (self.rv2 & 0xFFFF_FFFF) + (self.imm & 0xFFFF_FFFF);
        let arg2_hi = if rv2_sign { HI_FILL } else { 0 } + (self.imm >> 32);
        let arg2 = (arg2_lo & 0xFFFF_FFFF) | (arg2_hi << 32);

        // rvd = sign-extend(res low word) — always sign-extended for *W.
        let rvd_hi = if res_sign { HI_FILL } else { 0 };
        let rvd = (self.res & 0xFFFF_FFFF) | (rvd_hi << 32);

        Cpu32Aux {
            signed,
            rv1_sign,
            arg1,
            rv2_sign,
            arg2,
            res_sign,
            rvd,
        }
    }
}

/// Generates the CPU32 trace from a list of operations.
///
/// Each operation occupies its own row (μ = 1); the table is padded to the next
/// power of two (minimum 4).
pub fn generate_cpu32_trace(
    operations: &[Cpu32Operation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = operations.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        vec![FE::zero(); num_rows * cols::NUM_COLUMNS],
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, op) in operations.iter().enumerate() {
        let aux = op.compute_aux();

        // Inputs
        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);
        table.set_dword_wl(row_idx, cols::PC_0, op.pc);

        // rv1 as DWordWHH: [Half, Half, Word]
        table.set_byte(row_idx, cols::RS1, op.rs1);
        table.set_bool(row_idx, cols::READ_REGISTER1, op.read_register1);
        table.set_dword_whh(row_idx, cols::RV1_0, op.rv1);
        table.set_bool(row_idx, cols::RV1_SIGN, aux.rv1_sign);
        table.set_dword_wl(row_idx, cols::ARG1_0, aux.arg1);

        // rv2 as DWordWHH
        table.set_byte(row_idx, cols::RS2, op.rs2);
        table.set_bool(row_idx, cols::READ_REGISTER2, op.read_register2);
        table.set_dword_whh(row_idx, cols::RV2_0, op.rv2);
        table.set_bool(row_idx, cols::RV2_SIGN, aux.rv2_sign);
        table.set_dword_wl(row_idx, cols::IMM_0, op.imm);
        table.set_dword_wl(row_idx, cols::ARG2_0, aux.arg2);

        // res as DWordHL: 4 halves
        table.set_dword_hl(row_idx, cols::RES_0, op.res);
        table.set_bool(row_idx, cols::RES_SIGN, aux.res_sign);

        // rd write
        table.set_byte(row_idx, cols::RD, op.rd);
        table.set_bool(row_idx, cols::WRITE_REGISTER, op.write_register);
        table.set_dword_wl(row_idx, cols::RVD_0, aux.rvd);

        // ALU control
        table.set_bool(row_idx, cols::ALU, op.alu);
        table.set_byte(row_idx, cols::ALU_FLAGS, op.alu_flags);
        table.set_bool(row_idx, cols::ADD, op.add);
        table.set_bool(row_idx, cols::SUB, op.sub);
        table.set_byte(
            row_idx,
            cols::HALF_INSTRUCTION_LENGTH,
            op.half_instruction_length,
        );
        table.set_bool(row_idx, cols::SIGNED, aux.signed);

        table.set_fe(row_idx, cols::MU, FE::one());
    }

    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

/// 2^16, to combine two halves into a word.
const HALF_SHIFT: i64 = 1 << 16;

/// The 8-element MEMW value/old for a register read: `[lo_word, hi_word, 0×6]`
/// where `lo_word = lo0 + 2^16·lo1` (Q9: cast `DWordWHH` → `DWordWL`).
fn register_dword(lo0: usize, lo1: usize, hi: usize) -> Vec<BusValue> {
    let mut v = vec![
        BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: lo0,
            },
            LinearTerm::Column {
                coefficient: HALF_SHIFT,
                column: lo1,
            },
        ]),
        BusValue::Packed {
            start_column: hi,
            packing: Packing::Direct,
        },
    ];
    v.extend(std::iter::repeat_n(BusValue::constant(0), 6));
    v
}

/// `timestamp + offset` as DWordWL: `[TIMESTAMP_0 + offset, TIMESTAMP_1]`.
fn timestamp_plus(offset: i64) -> Vec<BusValue> {
    vec![
        BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::TIMESTAMP_0,
            },
            LinearTerm::Constant(offset),
        ]),
        BusValue::Packed {
            start_column: cols::TIMESTAMP_1,
            packing: Packing::Direct,
        },
    ]
}

/// MEMW register **read** (24 elements: `old == value`, `is_register=1`, `write2=1`).
fn reg_read(
    rs: usize,
    lo0: usize,
    lo1: usize,
    hi: usize,
    ts_offset: i64,
    mult: usize,
) -> BusInteraction {
    let mut values = register_dword(lo0, lo1, hi); // old
    values.push(BusValue::constant(1)); // is_register
    values.push(BusValue::linear(vec![LinearTerm::Column {
        coefficient: 2,
        column: rs,
    }])); // base_address[0] = 2*rs
    values.push(BusValue::constant(0)); // base_address[1]
    values.extend(register_dword(lo0, lo1, hi)); // value
    values.extend(timestamp_plus(ts_offset));
    values.push(BusValue::constant(1)); // write2 = 1 (register = 2 words)
    values.push(BusValue::constant(0)); // write4
    values.push(BusValue::constant(0)); // write8
    BusInteraction::sender(BusId::Memw, Multiplicity::Column(mult), values)
}

/// MEMW register **write** (16 elements: `value = [val_lo, val_hi, 0×6]`, `write2=1`).
fn reg_write(
    rd: usize,
    val_lo: usize,
    val_hi: usize,
    ts_offset: i64,
    mult: usize,
) -> BusInteraction {
    let mut values = vec![
        BusValue::constant(1), // is_register
        BusValue::linear(vec![LinearTerm::Column {
            coefficient: 2,
            column: rd,
        }]), // base_address[0] = 2*rd
        BusValue::constant(0), // base_address[1]
        BusValue::Packed {
            start_column: val_lo,
            packing: Packing::Direct,
        },
        BusValue::Packed {
            start_column: val_hi,
            packing: Packing::Direct,
        },
    ];
    values.extend(std::iter::repeat_n(BusValue::constant(0), 6)); // value[2..8]
    values.extend(timestamp_plus(ts_offset));
    values.push(BusValue::constant(1)); // write2 = 1
    values.push(BusValue::constant(0)); // write4
    values.push(BusValue::constant(0)); // write8
    BusInteraction::sender(BusId::Memw, Multiplicity::Column(mult), values)
}

/// All bus interactions for the CPU32 table.
pub fn bus_interactions() -> Vec<BusInteraction> {
    use packed_decode_shrunk as pd;
    let mut interactions = Vec::new();

    // DECODE[pc, imm, packed_decode] (sender, mult μ); word_instr is constant 1,
    // and there are no MEMORY/BRANCH/ECALL/mem_flags terms (CPU32 is ALU-only).
    interactions.push(BusInteraction::sender(
        BusId::Decode,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::Packed {
                start_column: cols::PC_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::IMM_0,
                packing: Packing::DWordWL,
            },
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1 << pd::READ_REG1,
                    column: cols::READ_REGISTER1,
                },
                LinearTerm::Column {
                    coefficient: 1 << pd::READ_REG2,
                    column: cols::READ_REGISTER2,
                },
                LinearTerm::Column {
                    coefficient: 1 << pd::WRITE_REG,
                    column: cols::WRITE_REGISTER,
                },
                LinearTerm::Constant(1 << pd::WORD_INSTR), // word_instr = 1
                LinearTerm::Column {
                    coefficient: 1 << pd::ALU,
                    column: cols::ALU,
                },
                LinearTerm::Column {
                    coefficient: 1 << pd::ADD,
                    column: cols::ADD,
                },
                LinearTerm::Column {
                    coefficient: 1 << pd::SUB,
                    column: cols::SUB,
                },
                LinearTerm::Column {
                    coefficient: 1 << pd::RS1,
                    column: cols::RS1,
                },
                LinearTerm::Column {
                    coefficient: 1 << pd::RS2,
                    column: cols::RS2,
                },
                LinearTerm::Column {
                    coefficient: 1 << pd::RD,
                    column: cols::RD,
                },
                LinearTerm::Column {
                    coefficient: 1 << pd::HALF_INSTRUCTION_LENGTH,
                    column: cols::HALF_INSTRUCTION_LENGTH,
                },
                LinearTerm::Column {
                    coefficient: 1 << pd::ALU_FLAGS,
                    column: cols::ALU_FLAGS,
                },
            ]),
        ],
    ));

    // Byte range checks: ARE_BYTES[x, 0].
    for col in [
        cols::HALF_INSTRUCTION_LENGTH,
        cols::ALU_FLAGS,
        cols::RS1,
        cols::RS2,
        cols::RD,
    ] {
        interactions.push(BusInteraction::sender(
            BusId::AreBytes,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: col,
                    packing: Packing::Direct,
                },
                BusValue::constant(0),
            ],
        ));
    }

    // IS_HALF for the rv1/rv2 low-word halves and the res halves.
    for col in [
        cols::RV1_0,
        cols::RV1_1,
        cols::RV2_0,
        cols::RV2_1,
        cols::RES_0,
        cols::RES_1,
        cols::RES_2,
        cols::RES_3,
    ] {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: col,
                packing: Packing::Direct,
            }],
        ));
    }

    // Register reads (rv1 @ ts+0, rv2 @ ts+1) and write (rvd @ ts+2).
    interactions.push(reg_read(
        cols::RS1,
        cols::RV1_0,
        cols::RV1_1,
        cols::RV1_2,
        0,
        cols::READ_REGISTER1,
    ));
    interactions.push(reg_read(
        cols::RS2,
        cols::RV2_0,
        cols::RV2_1,
        cols::RV2_2,
        1,
        cols::READ_REGISTER2,
    ));
    interactions.push(reg_write(
        cols::RD,
        cols::RVD_0,
        cols::RVD_1,
        2,
        cols::WRITE_REGISTER,
    ));

    // ALU[arg1, arg2, alu_flags] -> res (sender, mult ALU). res is DWordHL cast to DWordWL.
    interactions.push(BusInteraction::sender(
        BusId::Alu,
        Multiplicity::Column(cols::ALU),
        vec![
            BusValue::Packed {
                start_column: cols::ARG1_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::ARG2_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::ALU_FLAGS,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES_0,
                packing: Packing::DWordHL,
            },
        ],
    ));

    // BYTE_ALU[AND, 32, alu_flags] -> 32·signed (extracts the signed bit).
    interactions.push(BusInteraction::sender(
        BusId::ByteAlu,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::constant(alu_op::AND as u64),
            BusValue::constant(1u64 << pd::ALU_FLAGS_SIGNED), // 32
            BusValue::Packed {
                start_column: cols::ALU_FLAGS,
                packing: Packing::Direct,
            },
            BusValue::linear(vec![LinearTerm::Column {
                coefficient: 1 << pd::ALU_FLAGS_SIGNED,
                column: cols::SIGNED,
            }]),
        ],
    ));

    // MSB16 sign extraction (high half of each low word).
    // `rv1`/`rv2`: `SIGN(rv·[1], signed)` — the MSB16 is gated by `signed`, so the
    // sign is only looked up when the inputs are sign-extended (unsigned ops send
    // nothing and the `(1-signed)·rv·_sign = 0` arith forces the sign to 0).
    // `res`: `SIGN(res[1], μ)` — the `*W` result is always sign-extended, so it is
    // gated by `μ` (every active row) instead.
    let msb16 = |half_col: usize, sign_col: usize, mult: usize| {
        BusInteraction::sender(
            BusId::Msb16,
            Multiplicity::Column(mult),
            vec![
                BusValue::Packed {
                    start_column: half_col,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: sign_col,
                    packing: Packing::Direct,
                },
            ],
        )
    };
    interactions.push(msb16(cols::RV1_1, cols::RV1_SIGN, cols::SIGNED));
    interactions.push(msb16(cols::RV2_1, cols::RV2_SIGN, cols::SIGNED));
    interactions.push(msb16(cols::RES_1, cols::RES_SIGN, cols::MU));

    // CPU32[timestamp, pc, half_instruction_length] (receiver from the main CPU).
    interactions.push(BusInteraction::receiver(
        BusId::Cpu32,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::PC_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::HALF_INSTRUCTION_LENGTH,
                packing: Packing::Direct,
            },
        ],
    ));

    interactions
}

// =========================================================================
// Single-source constraint set (ConstraintBuilder front-end)
// =========================================================================

/// The CPU32 table's 32 transition constraints as a single [`ConstraintSet`]:
/// - idx 0-6:   `IS_BIT` on `read_register1/2`, `write_register`, `alu`, `add`,
///   `sub`, `μ`;
/// - idx 7,8:   `ADD` pair `arg1 + arg2 = res` (gated on `add`);
/// - idx 9,10:  `ADD` pair `arg2 + res = arg1` (gated on `sub`);
/// - idx 11-16: `(1 − read)·value` for the six `rv1/rv2` limbs;
/// - idx 17-22: sign-extension arithmetic `Arg1Lo/Hi`, `Arg2Lo/Hi`, `RvdLo/Hi`;
/// - idx 23,24: `(1 − signed)·rv·_sign` (sign zero when unsigned);
/// - idx 25,26: `read_register2·imm[i]` (arg2 exclusivity);
/// - idx 27-31: `(1 − μ)·flag` for `read_register1/2`, `write_register`,
///   `signed`, `res_sign`.
pub struct Cpu32Constraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for Cpu32Constraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0-6: IS_BIT on the flag columns and the multiplicity.
        for (i, &col) in [
            cols::READ_REGISTER1,
            cols::READ_REGISTER2,
            cols::WRITE_REGISTER,
            cols::ALU,
            cols::ADD,
            cols::SUB,
            cols::MU,
        ]
        .iter()
        .enumerate()
        {
            emit_is_bit(b, i, col, None);
        }

        // idx 7,8: ADD fast-path arg1 + arg2 = res (cond = ADD).
        emit_add_pair(
            b,
            7,
            &[cols::ADD],
            &AddOperand::dword(cols::ARG1_0),
            &AddOperand::dword(cols::ARG2_0),
            &AddOperand::from_dword_hl(cols::RES_0),
        );

        // idx 9,10: SUB fast-path arg2 + res = arg1 (cond = SUB).
        emit_add_pair(
            b,
            9,
            &[cols::SUB],
            &AddOperand::dword(cols::ARG2_0),
            &AddOperand::from_dword_hl(cols::RES_0),
            &AddOperand::dword(cols::ARG1_0),
        );

        // idx 11-16: unread register limbs are zero: (1 - read)·value.
        let mut idx = 11;
        for (read_col, value_col) in [
            (cols::READ_REGISTER1, cols::RV1_0),
            (cols::READ_REGISTER1, cols::RV1_1),
            (cols::READ_REGISTER1, cols::RV1_2),
            (cols::READ_REGISTER2, cols::RV2_0),
            (cols::READ_REGISTER2, cols::RV2_1),
            (cols::READ_REGISTER2, cols::RV2_2),
        ] {
            let one = b.one();
            let read = b.main(0, read_col);
            let value = b.main(0, value_col);
            b.emit_base(idx, (one - read) * value);
            idx += 1;
        }

        // idx 17-22: sign-extension (`ext`) arithmetic for arg1, arg2, rvd.
        let shift16 = b.const_base(SHIFT_16);
        let hi_fill = b.const_base(HI_FILL);

        // Arg1Lo: arg1_0 - rv1_0 - shift16·rv1_1
        let e = b.main(0, cols::ARG1_0)
            - b.main(0, cols::RV1_0)
            - shift16.clone() * b.main(0, cols::RV1_1);
        b.emit_base(17, e);
        // Arg1Hi: arg1_1 - hi_fill·rv1_sign
        let e = b.main(0, cols::ARG1_1) - hi_fill.clone() * b.main(0, cols::RV1_SIGN);
        b.emit_base(18, e);
        // Arg2Lo: arg2_0 - rv2_0 - shift16·rv2_1 - imm_0
        let e = b.main(0, cols::ARG2_0)
            - b.main(0, cols::RV2_0)
            - shift16.clone() * b.main(0, cols::RV2_1)
            - b.main(0, cols::IMM_0);
        b.emit_base(19, e);
        // Arg2Hi: arg2_1 - hi_fill·rv2_sign - imm_1
        let e = b.main(0, cols::ARG2_1)
            - hi_fill.clone() * b.main(0, cols::RV2_SIGN)
            - b.main(0, cols::IMM_1);
        b.emit_base(20, e);
        // RvdLo: rvd_0 - res_0 - shift16·res_1
        let e = b.main(0, cols::RVD_0) - b.main(0, cols::RES_0) - shift16 * b.main(0, cols::RES_1);
        b.emit_base(21, e);
        // RvdHi: rvd_1 - hi_fill·res_sign
        let e = b.main(0, cols::RVD_1) - hi_fill * b.main(0, cols::RES_SIGN);
        b.emit_base(22, e);

        // idx 23,24: (1 - signed)·sign — sign bit is zero when unsigned.
        for (i, sign_col) in [cols::RV1_SIGN, cols::RV2_SIGN].into_iter().enumerate() {
            let one = b.one();
            let signed = b.main(0, cols::SIGNED);
            let sign = b.main(0, sign_col);
            b.emit_base(23 + i, (one - signed) * sign);
        }

        // idx 25,26: read_register2·imm[i] (arg2 multiplex exclusivity).
        for (i, imm_col) in [cols::IMM_0, cols::IMM_1].into_iter().enumerate() {
            let rr2 = b.main(0, cols::READ_REGISTER2);
            let imm = b.main(0, imm_col);
            b.emit_base(25 + i, rr2 * imm);
        }

        // idx 27-31: (1 - μ)·flag — flag must be 0 on padding rows.
        for (i, flag_col) in [
            cols::READ_REGISTER1,
            cols::READ_REGISTER2,
            cols::WRITE_REGISTER,
            cols::SIGNED,
            cols::RES_SIGN,
        ]
        .into_iter()
        .enumerate()
        {
            let one = b.one();
            let mu = b.main(0, cols::MU);
            let flag = b.main(0, flag_col);
            b.emit_base(27 + i, (one - mu) * flag);
        }
    }
}
