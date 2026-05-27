//! CPU32 table — shrink-cpu rework.
//!
//! Handles all 32-bit word (`*W`) instructions delegated by the main CPU via
//! the `CPU32[timestamp, pc, instruction_length]` interaction. All `*W`
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
//! NOTE: this module currently provides the column layout and trace generation.
//! Bus interactions and constraints are added in a follow-up step; the chip is
//! not yet registered in `VmAirs`.

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::table::TableView;
use stark::trace::TraceTable;

use super::types::{FE, GoldilocksExtension, GoldilocksField, SHIFT_16, packed_decode_shrunk};
use crate::constraints::templates::{AddConstraint, AddOperand, new_is_bit_constraints};

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
    pub const INSTRUCTION_LENGTH: usize = 35;
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
    pub instruction_length: u8,
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
    /// Computes the derived auxiliary values (signs, extended args, rvd).
    pub fn compute_aux(&self) -> Cpu32Aux {
        let signed = (self.alu_flags as u64 & SIGNED_MASK) != 0;

        // Sign bits = MSB (bit 31) of the low word of each value.
        let rv1_sign = (self.rv1 >> 31) & 1 == 1;
        let rv2_sign = (self.rv2 >> 31) & 1 == 1;
        let res_sign = (self.res >> 31) & 1 == 1;

        // arg1 = ext(rv1 low word): low word as-is, high word = (2^32-1) if
        // (signed AND rv1_sign) else 0.
        let arg1_hi = if signed && rv1_sign { HI_FILL } else { 0 };
        let arg1 = (self.rv1 & 0xFFFF_FFFF) | (arg1_hi << 32);

        // arg2 = ext(rv2 low word) + imm. By the decoding assumption exactly one
        // of rv2 / imm is non-zero, so the per-word sums never overflow.
        let arg2_lo = (self.rv2 & 0xFFFF_FFFF) + (self.imm & 0xFFFF_FFFF);
        let arg2_hi = if signed && rv2_sign { HI_FILL } else { 0 } + (self.imm >> 32);
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
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, op) in operations.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;
        let aux = op.compute_aux();

        // Inputs
        data[base + cols::TIMESTAMP_0] = FE::from(op.timestamp & 0xFFFF_FFFF);
        data[base + cols::TIMESTAMP_1] = FE::from(op.timestamp >> 32);
        data[base + cols::PC_0] = FE::from(op.pc & 0xFFFF_FFFF);
        data[base + cols::PC_1] = FE::from(op.pc >> 32);

        // rv1 as DWordWHH: [Half, Half, Word]
        data[base + cols::RS1] = FE::from(op.rs1 as u64);
        data[base + cols::READ_REGISTER1] = FE::from(op.read_register1 as u64);
        data[base + cols::RV1_0] = FE::from(op.rv1 & 0xFFFF);
        data[base + cols::RV1_1] = FE::from((op.rv1 >> 16) & 0xFFFF);
        data[base + cols::RV1_2] = FE::from(op.rv1 >> 32);
        data[base + cols::RV1_SIGN] = FE::from(aux.rv1_sign as u64);
        data[base + cols::ARG1_0] = FE::from(aux.arg1 & 0xFFFF_FFFF);
        data[base + cols::ARG1_1] = FE::from(aux.arg1 >> 32);

        // rv2 as DWordWHH
        data[base + cols::RS2] = FE::from(op.rs2 as u64);
        data[base + cols::READ_REGISTER2] = FE::from(op.read_register2 as u64);
        data[base + cols::RV2_0] = FE::from(op.rv2 & 0xFFFF);
        data[base + cols::RV2_1] = FE::from((op.rv2 >> 16) & 0xFFFF);
        data[base + cols::RV2_2] = FE::from(op.rv2 >> 32);
        data[base + cols::RV2_SIGN] = FE::from(aux.rv2_sign as u64);
        data[base + cols::IMM_0] = FE::from(op.imm & 0xFFFF_FFFF);
        data[base + cols::IMM_1] = FE::from(op.imm >> 32);
        data[base + cols::ARG2_0] = FE::from(aux.arg2 & 0xFFFF_FFFF);
        data[base + cols::ARG2_1] = FE::from(aux.arg2 >> 32);

        // res as DWordHL: 4 halves
        data[base + cols::RES_0] = FE::from(op.res & 0xFFFF);
        data[base + cols::RES_1] = FE::from((op.res >> 16) & 0xFFFF);
        data[base + cols::RES_2] = FE::from((op.res >> 32) & 0xFFFF);
        data[base + cols::RES_3] = FE::from((op.res >> 48) & 0xFFFF);
        data[base + cols::RES_SIGN] = FE::from(aux.res_sign as u64);

        // rd write
        data[base + cols::RD] = FE::from(op.rd as u64);
        data[base + cols::WRITE_REGISTER] = FE::from(op.write_register as u64);
        data[base + cols::RVD_0] = FE::from(aux.rvd & 0xFFFF_FFFF);
        data[base + cols::RVD_1] = FE::from(aux.rvd >> 32);

        // ALU control
        data[base + cols::ALU] = FE::from(op.alu as u64);
        data[base + cols::ALU_FLAGS] = FE::from(op.alu_flags as u64);
        data[base + cols::ADD] = FE::from(op.add as u64);
        data[base + cols::SUB] = FE::from(op.sub as u64);
        data[base + cols::INSTRUCTION_LENGTH] = FE::from(op.instruction_length as u64);
        data[base + cols::SIGNED] = FE::from(aux.signed as u64);

        data[base + cols::MU] = FE::one();
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Constraints
// =========================================================================

/// Arithmetic constraints for CPU32: the sign-extension `ext` group plus the
/// register-zero checks. (`IS_BIT` flags and the ADD/SUB carries are produced
/// by the template helpers in [`cpu32_constraints`].)
pub struct Cpu32Constraint {
    constraint_idx: usize,
    kind: Cpu32ConstraintKind,
}

#[derive(Debug, Clone, Copy)]
pub enum Cpu32ConstraintKind {
    /// `arg1[0] = rv1[0] + 2^16·rv1[1]` (low word of `arg1`).
    Arg1Lo,
    /// `arg1[1] = (2^32-1)·signed·rv1_sign` (sign/zero extension of the high word).
    Arg1Hi,
    /// `arg2[0] = rv2[0] + 2^16·rv2[1] + imm[0]`.
    Arg2Lo,
    /// `arg2[1] = (2^32-1)·signed·rv2_sign + imm[1]`.
    Arg2Hi,
    /// `rvd[0] = res[0] + 2^16·res[1]`.
    RvdLo,
    /// `rvd[1] = (2^32-1)·res_sign` (the `*W` result is always sign-extended).
    RvdHi,
    /// `(1 - read_col)·value_col = 0` (an unread register half is zero).
    RegZero { read_col: usize, value_col: usize },
}

impl Cpu32Constraint {
    pub fn new(kind: Cpu32ConstraintKind, constraint_idx: usize) -> Self {
        Self {
            constraint_idx,
            kind,
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for Cpu32Constraint {
    fn degree(&self) -> usize {
        match self.kind {
            Cpu32ConstraintKind::Arg1Lo
            | Cpu32ConstraintKind::Arg2Lo
            | Cpu32ConstraintKind::RvdLo
            | Cpu32ConstraintKind::RvdHi => 1,
            // signed·sign (degree 2) and (1-read)·value (degree 2)
            Cpu32ConstraintKind::Arg1Hi
            | Cpu32ConstraintKind::Arg2Hi
            | Cpu32ConstraintKind::RegZero { .. } => 2,
        }
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let get = |c: usize| step.get_main_evaluation_element(0, c).clone();
        let shift16 = FieldElement::<F>::from(SHIFT_16);
        let hi_fill = FieldElement::<F>::from(HI_FILL);
        let one = FieldElement::<F>::one();

        match self.kind {
            Cpu32ConstraintKind::Arg1Lo => {
                get(cols::ARG1_0) - get(cols::RV1_0) - &shift16 * get(cols::RV1_1)
            }
            Cpu32ConstraintKind::Arg1Hi => {
                get(cols::ARG1_1) - hi_fill * get(cols::SIGNED) * get(cols::RV1_SIGN)
            }
            Cpu32ConstraintKind::Arg2Lo => {
                get(cols::ARG2_0)
                    - get(cols::RV2_0)
                    - &shift16 * get(cols::RV2_1)
                    - get(cols::IMM_0)
            }
            Cpu32ConstraintKind::Arg2Hi => {
                get(cols::ARG2_1)
                    - hi_fill * get(cols::SIGNED) * get(cols::RV2_SIGN)
                    - get(cols::IMM_1)
            }
            Cpu32ConstraintKind::RvdLo => {
                get(cols::RVD_0) - get(cols::RES_0) - &shift16 * get(cols::RES_1)
            }
            Cpu32ConstraintKind::RvdHi => get(cols::RVD_1) - hi_fill * get(cols::RES_SIGN),
            Cpu32ConstraintKind::RegZero {
                read_col,
                value_col,
            } => (one - get(read_col)) * get(value_col),
        }
    }
}

/// Creates all transition constraints for the CPU32 table:
/// `IS_BIT` on the flag columns, the `ADD`/`SUB` fast-path carries, the
/// register-zero checks, and the sign-extension `ext` arithmetic.
pub fn cpu32_constraints(
    constraint_idx_start: usize,
) -> (
    Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>>,
    usize,
) {
    let mut constraints: Vec<
        Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>,
    > = Vec::new();

    // IS_BIT on the flag columns.
    let (is_bit, mut idx) = new_is_bit_constraints(
        &[
            cols::READ_REGISTER1,
            cols::READ_REGISTER2,
            cols::WRITE_REGISTER,
            cols::ALU,
            cols::ADD,
            cols::SUB,
        ],
        constraint_idx_start,
    );
    for c in is_bit {
        constraints.push(c.boxed());
    }

    // ADD fast-path: arg1 + arg2 = res (cond = ADD).
    let (add_lo, add_hi) = AddConstraint::new_pair(
        vec![cols::ADD],
        AddOperand::dword(cols::ARG1_0),
        AddOperand::dword(cols::ARG2_0),
        AddOperand::from_dword_hl(cols::RES_0),
        idx,
    );
    idx += 2;
    constraints.push(add_lo.boxed());
    constraints.push(add_hi.boxed());

    // SUB fast-path: res = arg1 - arg2, encoded as arg2 + res = arg1 (cond = SUB).
    let (sub_lo, sub_hi) = AddConstraint::new_pair(
        vec![cols::SUB],
        AddOperand::dword(cols::ARG2_0),
        AddOperand::from_dword_hl(cols::RES_0),
        AddOperand::dword(cols::ARG1_0),
        idx,
    );
    idx += 2;
    constraints.push(sub_lo.boxed());
    constraints.push(sub_hi.boxed());

    // Unread register halves are zero.
    for (read_col, value_col) in [
        (cols::READ_REGISTER1, cols::RV1_0),
        (cols::READ_REGISTER1, cols::RV1_1),
        (cols::READ_REGISTER2, cols::RV2_0),
        (cols::READ_REGISTER2, cols::RV2_1),
    ] {
        constraints.push(
            Cpu32Constraint::new(
                Cpu32ConstraintKind::RegZero {
                    read_col,
                    value_col,
                },
                idx,
            )
            .boxed(),
        );
        idx += 1;
    }

    // Sign-extension (`ext`) arithmetic for arg1, arg2, rvd.
    for kind in [
        Cpu32ConstraintKind::Arg1Lo,
        Cpu32ConstraintKind::Arg1Hi,
        Cpu32ConstraintKind::Arg2Lo,
        Cpu32ConstraintKind::Arg2Hi,
        Cpu32ConstraintKind::RvdLo,
        Cpu32ConstraintKind::RvdHi,
    ] {
        constraints.push(Cpu32Constraint::new(kind, idx).boxed());
        idx += 1;
    }

    (constraints, idx)
}
