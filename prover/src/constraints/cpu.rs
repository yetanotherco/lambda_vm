//! CPU table constraints for the 64-bit VM.
//!
//! Translates the `cpu.toml` constraint groups onto the shrunk CPU layout
//! (`tables::cpu::cols`). Byte/half range checks (`IS_BYTE`/`IS_HALF`) and all
//! lookups (`DECODE`/`ALU`/`MEMORY`/`CPU32`/`MEMW`/`BRANCH`/`ECALL`) live in
//! `tables::cpu::bus_interactions`; this module holds only the algebraic
//! (transition) constraints:
//!
//! - **decode**: `word_instr · {MEMORY,BRANCH,ECALL} = 0` mutex.
//! - **range**: `IS_BIT` for the flag columns + the inline-PC bits + `non_padding`.
//! - **alu**: `arg2` multiplex, `ADD`/`SUB` fast-path templates on `rv1`/`arg2`.
//! - **mem**: `¬read_registerN ⇒ rvN = 0`, `¬MEMORY ⇒ rvd = cast(res, WL)`.
//! - **branch**: `branch_cond = BRANCH·(JALR + (1−JALR)·res[0])`, `next_pc = pc + len`.
//!
//! `JALR` is the `mem_flags` byte read directly: under `BRANCH` only the JALR bit
//! of `mem_flags` can be set, so `mem_flags ∈ {0,1} = JALR` there.

use crate::tables::cpu::cols;
use crate::tables::types::{GoldilocksExtension, GoldilocksField, SHIFT_16};

use super::templates::AddOperand;

// =========================================================================
// Range: IS_BIT flag columns
// =========================================================================

/// Bit columns that need `IS_BIT` (`x·(x−1) = 0`) constraints.
pub const BIT_FLAG_COLUMNS: &[usize] = &[
    cols::READ_REGISTER1,
    cols::READ_REGISTER2,
    cols::WRITE_REGISTER,
    cols::WORD_INSTR,
    cols::ALU,
    cols::ADD,
    cols::SUB,
    cols::MEMORY,
    cols::BRANCH,
    cols::ECALL,
    cols::PC_DOUBLE_READ,
    cols::PREV_PC_TIMESTAMP_BORROW,
];

// =========================================================================
// Assembly
// =========================================================================

/// Total number of CPU transition constraints (excludes bus lookups):
/// - IS_BIT: 12
/// - decode mutex: 6 (`word_instr · {MEMORY, BRANCH, ECALL, WRITE_REGISTER,
///   READ_REGISTER1, READ_REGISTER2}`)
/// - ADD pair: 2, SUB pair: 2
/// - arg2 multiplex: 2
/// - register zero-forcing: 4 (`rv1[0..1]`, `rv2[0..1]`)
/// - rvd = res: 2
/// - branch rvd (`pc + len`): 2
/// - branch_cond: 1
/// - next_pc: 2
/// - assumptions: 4 (MEMORY·BRANCH mutex 1 + arg2 exclusivity 2 + mem_flags IS_BIT 1)
pub const NUM_CPU_CONSTRAINTS: usize = 12 + 6 + 2 + 2 + 2 + 4 + 2 + 2 + 1 + 2 + 4;

// =========================================================================
// Single-body emit functions (ConstraintBuilder front-end)
// =========================================================================
//
// One body per constraint against the generic `ConstraintBuilder` serves the
// compiled prover folder, the verifier folder and IR capture. All constraints
// here use the default zerofier shape (every row, no exemptions).

use stark::constraints::builder::{ConstraintBuilder, ConstraintMeta, ConstraintSet};

use super::templates::{INV_SHIFT_32, add_pair_meta, emit_add_pair, emit_is_bit, is_bit_meta};

/// `col_a · col_b = 0`.
pub fn emit_product_zero<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &mut B,
    idx: usize,
    col_a: usize,
    col_b: usize,
) {
    let root = b.main(0, col_a) * b.main(0, col_b);
    b.emit_base(idx, root);
}

/// Metadata for [`emit_product_zero`].
pub fn product_zero_meta(idx: usize) -> ConstraintMeta {
    ConstraintMeta::base(idx, 2)
}

/// `(1 − MEMORY − BRANCH) · read_register2 · imm[i] = 0`.
pub fn emit_arg2_exclusive<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &mut B,
    idx: usize,
    imm_col: usize,
) {
    let one = b.one();
    let memory = b.main(0, cols::MEMORY);
    let branch = b.main(0, cols::BRANCH);
    let rr2 = b.main(0, cols::READ_REGISTER2);
    let imm = b.main(0, imm_col);
    b.emit_base(idx, (one - memory - branch) * rr2 * imm);
}

/// Metadata for [`emit_arg2_exclusive`].
pub fn arg2_exclusive_meta(idx: usize) -> ConstraintMeta {
    ConstraintMeta::base(idx, 3)
}

/// `(1 − MEMORY) · mem_flags · (1 − mem_flags) = 0`.
pub fn emit_mem_flags_bit<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &mut B,
    idx: usize,
) {
    let one = b.one();
    let memory = b.main(0, cols::MEMORY);
    let mem_flags = b.main(0, cols::MEM_FLAGS);
    b.emit_base(
        idx,
        (one.clone() - memory) * mem_flags.clone() * (one - mem_flags),
    );
}

/// Metadata for [`emit_mem_flags_bit`].
pub fn mem_flags_bit_meta(idx: usize) -> ConstraintMeta {
    ConstraintMeta::base(idx, 3)
}

/// `(1 − flag) · value = 0`.
pub fn emit_reg_not_read_is_zero<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &mut B,
    idx: usize,
    flag_col: usize,
    value_col: usize,
) {
    let one = b.one();
    let flag = b.main(0, flag_col);
    let value = b.main(0, value_col);
    b.emit_base(idx, (one - flag) * value);
}

/// Metadata for [`emit_reg_not_read_is_zero`].
pub fn reg_not_read_is_zero_meta(idx: usize) -> ConstraintMeta {
    ConstraintMeta::base(idx, 2)
}

/// `arg2` multiplex for word index `word_idx ∈ {0, 1}`:
///
/// ```text
/// arg2[i] − (MEMORY·imm[i] + BRANCH·rv2[i] + (1−MEMORY−BRANCH)·(rv2[i] + imm[i]))
/// ```
pub fn emit_arg2<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &mut B,
    idx: usize,
    word_idx: usize,
) {
    let (arg2_col, imm_col, rv2_col) = if word_idx == 0 {
        (cols::ARG2_0, cols::IMM_0, cols::RV2_0)
    } else {
        (cols::ARG2_1, cols::IMM_1, cols::RV2_1)
    };
    let one = b.one();
    let arg2 = b.main(0, arg2_col);
    let imm = b.main(0, imm_col);
    let rv2 = b.main(0, rv2_col);
    let memory = b.main(0, cols::MEMORY);
    let branch = b.main(0, cols::BRANCH);

    let expected = memory.clone() * imm.clone()
        + branch.clone() * rv2.clone()
        + (one - memory - branch) * (rv2 + imm);
    b.emit_base(idx, arg2 - expected);
}

/// Metadata for [`emit_arg2`] (degree 2 relies on the live `MEMORY·BRANCH = 0`
/// mutex).
pub fn arg2_meta(idx: usize) -> ConstraintMeta {
    ConstraintMeta::base(idx, 2)
}

/// `cast(res, DWordWL)` word from the four `res` halves (DWordHL).
fn res_word_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    high: bool,
) -> B::Expr {
    let (lo_col, hi_col) = if high {
        (cols::RES_2, cols::RES_3)
    } else {
        (cols::RES_0, cols::RES_1)
    };
    b.main(0, lo_col) + b.main(0, hi_col) * b.const_base(SHIFT_16)
}

/// `(1 − MEMORY − BRANCH) · (rvd[i] − cast(res, WL)[i]) = 0`.
pub fn emit_rvd_eq_res<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &mut B,
    idx: usize,
    word_idx: usize,
) {
    let high = word_idx == 1;
    let rvd_col = if high { cols::RVD_1 } else { cols::RVD_0 };
    let one = b.one();
    let memory = b.main(0, cols::MEMORY);
    let branch = b.main(0, cols::BRANCH);
    let rvd = b.main(0, rvd_col);
    let res_w = res_word_expr(b, high);
    b.emit_base(idx, (one - memory - branch) * (rvd - res_w));
}

/// Metadata for [`emit_rvd_eq_res`].
pub fn rvd_eq_res_meta(idx: usize) -> ConstraintMeta {
    ConstraintMeta::base(idx, 2)
}

/// The `pc + instruction_length` carry pair against a destination dword
/// (`rvd` or `next_pc`), gated by `gate`; shared body of
/// [`emit_branch_rvd_pair`] and [`emit_next_pc_add_pair`]:
///
/// ```text
/// carry_0 = (pc[0] + 2·half_len − dst[0])·2⁻³²
/// carry_1 = (pc[1] + carry_0 − dst[1])·2⁻³²
/// emit:     gate·carry_i·(1 − carry_i)     at idx, idx+1
/// ```
fn emit_pc_len_add_pair<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &mut B,
    idx: usize,
    dst_lo_col: usize,
    dst_hi_col: usize,
    gate: fn(&B) -> B::Expr,
) {
    let inv_2_32 = b.const_base(INV_SHIFT_32);
    let pc_lo = b.main(0, cols::PC_0);
    let pc_hi = b.main(0, cols::PC_1);
    let dst_lo = b.main(0, dst_lo_col);
    let dst_hi = b.main(0, dst_hi_col);
    let half_len = b.main(0, cols::HALF_INSTRUCTION_LENGTH);
    let instr_len = half_len.clone() + half_len; // real byte length = 2 · half
    let carry_0 = (pc_lo + instr_len - dst_lo) * inv_2_32.clone();
    let carry_1 = (pc_hi + carry_0.clone() - dst_hi) * inv_2_32;

    let one = b.one();
    let g = gate(b);
    b.emit_base(idx, g * carry_0.clone() * (one - carry_0));
    let one = b.one();
    let g = gate(b);
    b.emit_base(idx + 1, g * carry_1.clone() * (one - carry_1));
}

/// `BRANCH · carry · (1 − carry) = 0` for `rvd = pc + instruction_length`
/// (two instances at `idx`, `idx + 1`).
pub fn emit_branch_rvd_pair<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &mut B,
    idx: usize,
) {
    emit_pc_len_add_pair(b, idx, cols::RVD_0, cols::RVD_1, |b| {
        b.main(0, cols::BRANCH)
    });
}

/// Metadata for [`emit_branch_rvd_pair`].
pub fn branch_rvd_meta(idx: usize) -> [ConstraintMeta; 2] {
    [
        ConstraintMeta::base(idx, 3),
        ConstraintMeta::base(idx + 1, 3),
    ]
}

/// `branch_cond − (BRANCH·JALR + BRANCH·(1−JALR)·res[0])`.
pub fn emit_branch_cond<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &mut B,
    idx: usize,
) {
    let one = b.one();
    let branch = b.main(0, cols::BRANCH);
    let jalr = b.main(0, cols::MEM_FLAGS);
    let res0 = b.main(0, cols::RES_0);
    let branch_cond = b.main(0, cols::BRANCH_COND);

    let expected = branch.clone() * jalr.clone() + branch * (one - jalr) * res0;
    b.emit_base(idx, branch_cond - expected);
}

/// Metadata for [`emit_branch_cond`].
pub fn branch_cond_meta(idx: usize) -> ConstraintMeta {
    ConstraintMeta::base(idx, 3)
}

/// `(1 − branch_cond) · carry · (1 − carry) = 0` for
/// `next_pc = pc + instruction_length` (two instances at `idx`, `idx + 1`).
pub fn emit_next_pc_add_pair<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &mut B,
    idx: usize,
) {
    emit_pc_len_add_pair(b, idx, cols::NEXT_PC_0, cols::NEXT_PC_1, |b| {
        let one = b.one();
        one - b.main(0, cols::BRANCH_COND)
    });
}

/// Metadata for [`emit_next_pc_add_pair`].
pub fn next_pc_add_meta(idx: usize) -> [ConstraintMeta; 2] {
    [
        ConstraintMeta::base(idx, 3),
        ConstraintMeta::base(idx + 1, 3),
    ]
}

// =========================================================================
// Single-source constraint set (ConstraintBuilder front-end)
// =========================================================================

/// The CPU table's transition constraints as a single [`ConstraintSet`]
/// ([`NUM_CPU_CONSTRAINTS`] = 39 constraints, all base-field):
/// - idx 0..11:  IS_BIT (unconditional) on each of [`BIT_FLAG_COLUMNS`];
/// - idx 12,13:  ADD fast-path carry pair (conditional on `ADD`);
/// - idx 14,15:  SUB fast-path carry pair (conditional on `SUB`);
/// - idx 16..21: `word_instr · {MEMORY, BRANCH, ECALL, WRITE_REGISTER,
///   READ_REGISTER1, READ_REGISTER2} = 0`;
/// - idx 22,23:  `arg2` multiplex (words 0, 1);
/// - idx 24..27: register zero-forcing (`rv1[0..1]`, `rv2[0..1]`);
/// - idx 28,29:  `rvd = cast(res, WL)` (words 0, 1);
/// - idx 30,31:  BRANCH ⇒ `rvd = pc + instruction_length` carry pair;
/// - idx 32:     `branch_cond`;
/// - idx 33,34:  `next_pc = pc + instruction_length` carry pair;
/// - idx 35:     `MEMORY · BRANCH = 0`;
/// - idx 36,37:  `arg2` exclusivity (`imm_0`, `imm_1`);
/// - idx 38:     `IS_BIT(mem_flags)` on non-MEMORY rows.
pub struct CpuConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for CpuConstraints {
    fn meta(&self) -> Vec<ConstraintMeta> {
        let mut m = Vec::with_capacity(NUM_CPU_CONSTRAINTS);
        // idx 0..11: IS_BIT on each BIT_FLAG_COLUMNS entry (unconditional).
        for i in 0..BIT_FLAG_COLUMNS.len() {
            m.push(is_bit_meta(i, false));
        }
        let mut idx = BIT_FLAG_COLUMNS.len();
        // idx 12,13: ADD pair (conditional on ADD).
        m.extend(add_pair_meta(idx, true));
        idx += 2;
        // idx 14,15: SUB pair (conditional on SUB).
        m.extend(add_pair_meta(idx, true));
        idx += 2;
        // idx 16..21: word_instr mutexes + register-read gates.
        for _ in 0..6 {
            m.push(product_zero_meta(idx));
            idx += 1;
        }
        // idx 22,23: arg2 multiplex.
        m.push(arg2_meta(idx));
        idx += 1;
        m.push(arg2_meta(idx));
        idx += 1;
        // idx 24..27: register zero-forcing.
        for _ in 0..4 {
            m.push(reg_not_read_is_zero_meta(idx));
            idx += 1;
        }
        // idx 28,29: rvd = cast(res, WL).
        m.push(rvd_eq_res_meta(idx));
        idx += 1;
        m.push(rvd_eq_res_meta(idx));
        idx += 1;
        // idx 30,31: branch rvd = pc + len.
        m.extend(branch_rvd_meta(idx));
        idx += 2;
        // idx 32: branch_cond.
        m.push(branch_cond_meta(idx));
        idx += 1;
        // idx 33,34: next_pc = pc + len.
        m.extend(next_pc_add_meta(idx));
        idx += 2;
        // idx 35: MEMORY · BRANCH = 0.
        m.push(product_zero_meta(idx));
        idx += 1;
        // idx 36,37: arg2 exclusivity.
        m.push(arg2_exclusive_meta(idx));
        idx += 1;
        m.push(arg2_exclusive_meta(idx));
        idx += 1;
        // idx 38: IS_BIT(mem_flags) on non-MEMORY rows.
        m.push(mem_flags_bit_meta(idx));
        idx += 1;
        debug_assert_eq!(idx, NUM_CPU_CONSTRAINTS);
        debug_assert_eq!(m.len(), NUM_CPU_CONSTRAINTS);
        m
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0..11: IS_BIT on each BIT_FLAG_COLUMNS entry (unconditional).
        for (i, &col) in BIT_FLAG_COLUMNS.iter().enumerate() {
            emit_is_bit(b, i, col, None);
        }
        let mut idx = BIT_FLAG_COLUMNS.len();

        // idx 12,13: ADD fast-path (cond = ADD), rv1 + arg2 = cast(res, WL).
        emit_add_pair(
            b,
            idx,
            &[cols::ADD],
            &AddOperand::dword(cols::RV1_0),
            &AddOperand::dword(cols::ARG2_0),
            &AddOperand::from_dword_hl(cols::RES_0),
        );
        idx += 2;

        // idx 14,15: SUB fast-path (cond = SUB), arg2 + res = rv1.
        emit_add_pair(
            b,
            idx,
            &[cols::SUB],
            &AddOperand::dword(cols::ARG2_0),
            &AddOperand::from_dword_hl(cols::RES_0),
            &AddOperand::dword(cols::RV1_0),
        );
        idx += 2;

        // idx 16..21: word_instr mutexes + register-read gates.
        for &col in &[
            cols::MEMORY,
            cols::BRANCH,
            cols::ECALL,
            cols::WRITE_REGISTER,
            cols::READ_REGISTER1,
            cols::READ_REGISTER2,
        ] {
            emit_product_zero(b, idx, cols::WORD_INSTR, col);
            idx += 1;
        }

        // idx 22,23: arg2 multiplex (low, high words).
        emit_arg2(b, idx, 0);
        idx += 1;
        emit_arg2(b, idx, 1);
        idx += 1;

        // idx 24..27: register zero-forcing (rv1/rv2 are DWordWL → 2 words each).
        for &value_col in &[cols::RV1_0, cols::RV1_1] {
            emit_reg_not_read_is_zero(b, idx, cols::READ_REGISTER1, value_col);
            idx += 1;
        }
        for &value_col in &[cols::RV2_0, cols::RV2_1] {
            emit_reg_not_read_is_zero(b, idx, cols::READ_REGISTER2, value_col);
            idx += 1;
        }

        // idx 28,29: ¬MEMORY ∧ ¬BRANCH ⇒ rvd = cast(res, WL).
        emit_rvd_eq_res(b, idx, 0);
        idx += 1;
        emit_rvd_eq_res(b, idx, 1);
        idx += 1;

        // idx 30,31: BRANCH ⇒ rvd = pc + instruction_length.
        emit_branch_rvd_pair(b, idx);
        idx += 2;

        // idx 32: branch_cond.
        emit_branch_cond(b, idx);
        idx += 1;

        // idx 33,34: next_pc = pc + instruction_length.
        emit_next_pc_add_pair(b, idx);
        idx += 2;

        // idx 35: MEMORY · BRANCH = 0.
        emit_product_zero(b, idx, cols::MEMORY, cols::BRANCH);
        idx += 1;

        // idx 36,37: arg2 exclusivity.
        for &imm_col in &[cols::IMM_0, cols::IMM_1] {
            emit_arg2_exclusive(b, idx, imm_col);
            idx += 1;
        }

        // idx 38: IS_BIT(mem_flags) on non-MEMORY rows.
        emit_mem_flags_bit(b, idx);
        idx += 1;

        debug_assert_eq!(idx, NUM_CPU_CONSTRAINTS);
    }
}
