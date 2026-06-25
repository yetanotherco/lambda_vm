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

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::constraints::builder::{
    ConstraintBuilder, ConstraintContext, ProverConstraintBuilder, TableConstraints,
    VerifierConstraintBuilder,
};
use stark::table::TableView;

use crate::tables::cpu::cols;
use crate::tables::types::{GoldilocksExtension, GoldilocksField, SHIFT_16};

use super::templates::{
    add_pair_fold, is_bit_fold, AddConstraint, AddOperand, IsBitConstraint, INV_SHIFT_32,
};

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

/// Creates all IS_BIT constraints for CPU flag columns.
pub fn create_is_bit_constraints(constraint_idx_start: usize) -> (Vec<IsBitConstraint>, usize) {
    super::templates::new_is_bit_constraints(BIT_FLAG_COLUMNS, constraint_idx_start)
}

// =========================================================================
// Generic helpers
// =========================================================================

/// `cast(res, DWordWL)` low/high words from the four `res` halves (DWordHL).
#[inline]
fn res_word<F, E>(step: &TableView<F, E>, high: bool) -> FieldElement<F>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let (lo_col, hi_col) = if high {
        (cols::RES_2, cols::RES_3)
    } else {
        (cols::RES_0, cols::RES_1)
    };
    let shift_16: FieldElement<F> = FieldElement::from(SHIFT_16);
    step.get_main_evaluation_element(0, lo_col)
        + step.get_main_evaluation_element(0, hi_col) * shift_16
}

// =========================================================================
// decode group: word_instr mutex
// =========================================================================

/// Constraint `col_a · col_b = 0`. Used for the decode mutexes
/// `word_instr · {MEMORY, BRANCH, ECALL} = 0`.
pub struct ProductZeroConstraint {
    col_a: usize,
    col_b: usize,
    constraint_idx: usize,
}

impl ProductZeroConstraint {
    pub fn new(col_a: usize, col_b: usize, constraint_idx: usize) -> Self {
        Self {
            col_a,
            col_b,
            constraint_idx,
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for ProductZeroConstraint {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        step.get_main_evaluation_element(0, self.col_a)
            * step.get_main_evaluation_element(0, self.col_b)
    }
}

/// `(1 - MEMORY - BRANCH) · read_register2 · imm[i] = 0`: when neither MEMORY nor
/// BRANCH is set, the `arg2` multiplex needs at most one of `rv2`/`imm` nonzero.
/// Decoding already guarantees this; a spec defense-in-depth assumption.
pub struct Arg2ExclusiveConstraint {
    imm_col: usize,
    constraint_idx: usize,
}

impl Arg2ExclusiveConstraint {
    pub fn new(imm_col: usize, constraint_idx: usize) -> Self {
        Self {
            imm_col,
            constraint_idx,
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for Arg2ExclusiveConstraint {
    fn degree(&self) -> usize {
        3
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let one = FieldElement::<F>::one();
        let memory = step.get_main_evaluation_element(0, cols::MEMORY).clone();
        let branch = step.get_main_evaluation_element(0, cols::BRANCH).clone();
        let rr2 = step.get_main_evaluation_element(0, cols::READ_REGISTER2);
        let imm = step.get_main_evaluation_element(0, self.imm_col);
        (one - memory - branch) * rr2 * imm
    }
}

/// `IS_BIT<mem_flags>` on non-MEMORY rows: `(1 - MEMORY) · mem_flags · (1 - mem_flags) = 0`.
/// On non-memory rows `mem_flags` carries only the JALR bit, so it must be 0/1.
/// A spec defense-in-depth assumption (the DECODE lookup already enforces it).
pub struct MemFlagsBitConstraint {
    constraint_idx: usize,
}

impl MemFlagsBitConstraint {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for MemFlagsBitConstraint {
    fn degree(&self) -> usize {
        3
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let one = FieldElement::<F>::one();
        let memory = step.get_main_evaluation_element(0, cols::MEMORY).clone();
        let mem_flags = step.get_main_evaluation_element(0, cols::MEM_FLAGS).clone();
        (one.clone() - memory) * &mem_flags * (one - &mem_flags)
    }
}

// =========================================================================
// mem group: register zero-forcing
// =========================================================================

/// Constraint `(1 − flag) · value = 0`: when `flag = 0`, `value` must be 0.
/// Used for `¬read_registerN ⇒ rvN[i] = 0`.
pub struct RegNotReadIsZeroConstraint {
    flag_col: usize,
    value_col: usize,
    constraint_idx: usize,
}

impl RegNotReadIsZeroConstraint {
    pub fn new(flag_col: usize, value_col: usize, constraint_idx: usize) -> Self {
        Self {
            flag_col,
            value_col,
            constraint_idx,
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for RegNotReadIsZeroConstraint {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let one = FieldElement::<F>::one();
        let flag = step.get_main_evaluation_element(0, self.flag_col).clone();
        let value = step.get_main_evaluation_element(0, self.value_col);
        (one - flag) * value
    }
}

// =========================================================================
// alu group: arg2 multiplex
// =========================================================================

/// `arg2` multiplex (`cpu.toml` CPU-A1), for word index
/// `word_idx ∈ {0,1}`:
///
/// ```text
/// arg2[i] = MEMORY·imm[i]
///         + BRANCH·rv2[i]
///         + (1−MEMORY−BRANCH)·(rv2[i] + imm[i])
/// ```
///
/// For BRANCH rows `arg2 = rv2` (JAL/JALR read no rs2, so `rv2 = 0`; conditional
/// branches feed `rv2` to the EQ/LT comparison). The final `rv2 + imm` term has
/// no inter-word carry because decode assumption A2 guarantees at most one of
/// `rv2`/`imm` is nonzero when `MEMORY+BRANCH = 0`. `MEMORY` and `BRANCH` are
/// mutually exclusive (enforced by the live `MEMORY·BRANCH = 0` constraint), so
/// `1−MEMORY−BRANCH ∈ {0,1}` and matches the degree-2 spec form.
pub struct Arg2Constraint {
    /// 0 = low word, 1 = high word.
    word_idx: usize,
    constraint_idx: usize,
}

impl Arg2Constraint {
    pub fn new(word_idx: usize, constraint_idx: usize) -> Self {
        Self {
            word_idx,
            constraint_idx,
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for Arg2Constraint {
    fn degree(&self) -> usize {
        // (1 - MEMORY - BRANCH) [deg 1] · (rv2 + imm) [deg 1] = 2. The degree-2
        // form relies on the live MEMORY·BRANCH = 0 mutex.
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let (arg2_col, imm_col, rv2_col) = if self.word_idx == 0 {
            (cols::ARG2_0, cols::IMM_0, cols::RV2_0)
        } else {
            (cols::ARG2_1, cols::IMM_1, cols::RV2_1)
        };

        let one = FieldElement::<F>::one();
        let arg2 = step.get_main_evaluation_element(0, arg2_col).clone();
        let imm = step.get_main_evaluation_element(0, imm_col).clone();
        let rv2 = step.get_main_evaluation_element(0, rv2_col).clone();
        let memory = step.get_main_evaluation_element(0, cols::MEMORY).clone();
        let branch = step.get_main_evaluation_element(0, cols::BRANCH).clone();

        // MEMORY · imm
        let mut expected = &memory * &imm;
        // BRANCH · rv2
        expected += &branch * &rv2;
        // (1 - MEMORY - BRANCH) · (rv2 + imm)
        expected += (&one - &memory - &branch) * (&rv2 + &imm);

        arg2 - expected
    }
}

// =========================================================================
// mem group: ¬MEMORY ∧ ¬JALR ⇒ rvd = cast(res, WL)
// =========================================================================

/// `(1 − MEMORY − BRANCH) · (rvd[i] − cast(res, WL)[i]) = 0` (`cpu.toml` CPU-M*).
///
/// On plain ALU rows `rvd = res`. BRANCH rows are exempt: their `rvd` is the
/// return address `pc + instruction_length`, pinned by [`BranchRvdConstraint`].
/// `MEMORY` and `BRANCH` are mutually exclusive (decode assumption), so
/// `1 − MEMORY − BRANCH ∈ {0,1}`. For LOAD/STORE `rvd` comes from the MEMORY bus.
pub struct RvdEqResConstraint {
    /// 0 = low word, 1 = high word.
    word_idx: usize,
    constraint_idx: usize,
}

impl RvdEqResConstraint {
    pub fn new(word_idx: usize, constraint_idx: usize) -> Self {
        Self {
            word_idx,
            constraint_idx,
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for RvdEqResConstraint {
    fn degree(&self) -> usize {
        // (1 - MEMORY - BRANCH) [deg 1] · (rvd - cast(res, WL)) [deg 1] = 2.
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let high = self.word_idx == 1;
        let rvd_col = if high { cols::RVD_1 } else { cols::RVD_0 };
        let one = FieldElement::<F>::one();
        let memory = step.get_main_evaluation_element(0, cols::MEMORY).clone();
        let branch = step.get_main_evaluation_element(0, cols::BRANCH).clone();
        let rvd = step.get_main_evaluation_element(0, rvd_col).clone();
        let res_w = res_word(step, high);
        (&one - &memory - &branch) * (rvd - res_w)
    }
}

// =========================================================================
// branch group: BRANCH ⇒ rvd = pc + instruction_length
// =========================================================================

/// `BRANCH · carry · (1 − carry) = 0` for the 64-bit addition
/// `rvd = pc + instruction_length` (the JAL/JALR return address), in two
/// instances (`carry_0` / `carry_1`). Mirrors [`NextPcAddConstraint`] so the
/// low→high carry is propagated: the spec computes `rvd` with the same
/// carry-correct `ADD` template as `next_pc` (`cpu.toml` branch group), so the
/// high word must include the carry out of `pc[0] + instruction_length`.
///
/// On every BRANCH row `rvd` holds the return address `pc + instruction_length`
/// (written to `rd` only by JAL/JALR; conditional branches compute it but never
/// write it). See [`RvdEqResConstraint`] for the complementary
/// `¬MEMORY ∧ ¬BRANCH ⇒ rvd = res` case.
pub struct BranchRvdConstraint {
    /// 0 = low-word carry, 1 = high-word carry.
    carry_idx: usize,
    constraint_idx: usize,
}

impl BranchRvdConstraint {
    pub fn new(carry_idx: usize, constraint_idx: usize) -> Self {
        assert!(carry_idx <= 1);
        Self {
            carry_idx,
            constraint_idx,
        }
    }

    pub fn new_pair(constraint_idx_start: usize) -> (Self, Self) {
        (
            Self::new(0, constraint_idx_start),
            Self::new(1, constraint_idx_start + 1),
        )
    }

    fn compute_carry_0<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let pc_lo = step.get_main_evaluation_element(0, cols::PC_0).clone();
        let rvd_lo = step.get_main_evaluation_element(0, cols::RVD_0).clone();
        let half_len = step
            .get_main_evaluation_element(0, cols::HALF_INSTRUCTION_LENGTH)
            .clone();
        let instr_len = &half_len + &half_len; // real byte length = 2 * half
        let inv_2_32 = FieldElement::<F>::from(super::templates::INV_SHIFT_32);
        (pc_lo + instr_len - rvd_lo) * inv_2_32
    }

    fn compute_carry_1<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let pc_hi = step.get_main_evaluation_element(0, cols::PC_1).clone();
        let rvd_hi = step.get_main_evaluation_element(0, cols::RVD_1).clone();
        let carry_0 = self.compute_carry_0(step);
        let inv_2_32 = FieldElement::<F>::from(super::templates::INV_SHIFT_32);
        (pc_hi + carry_0 - rvd_hi) * inv_2_32
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for BranchRvdConstraint {
    fn degree(&self) -> usize {
        // BRANCH (deg 1) · carry · (1 − carry) = 3.
        3
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let one = FieldElement::<F>::one();
        let branch = step.get_main_evaluation_element(0, cols::BRANCH).clone();
        let carry = match self.carry_idx {
            0 => self.compute_carry_0(step),
            1 => self.compute_carry_1(step),
            _ => unreachable!("carry_idx validated <= 1 at construction"),
        };
        branch * &carry * (&one - &carry)
    }
}

// =========================================================================
// branch group: branch_cond
// =========================================================================

/// `branch_cond = BRANCH·JALR + BRANCH·(1−JALR)·res[0]` (`cpu.toml` CPU-B1).
/// `JALR = mem_flags` (bit, under BRANCH); `res[0]` is the low half of `res`.
pub struct BranchCondConstraint {
    constraint_idx: usize,
}

impl BranchCondConstraint {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for BranchCondConstraint {
    fn degree(&self) -> usize {
        3
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let one = FieldElement::<F>::one();
        let branch = step.get_main_evaluation_element(0, cols::BRANCH).clone();
        let jalr = step.get_main_evaluation_element(0, cols::MEM_FLAGS).clone();
        let res0 = step.get_main_evaluation_element(0, cols::RES_0).clone();
        let branch_cond = step
            .get_main_evaluation_element(0, cols::BRANCH_COND)
            .clone();

        let expected = &branch * &jalr + &branch * (&one - &jalr) * res0;
        branch_cond - expected
    }
}

// =========================================================================
// branch group: next_pc = pc + instruction_length (when not branching)
// =========================================================================

/// `(1 − branch_cond) · carry · (1 − carry) = 0` for the 64-bit addition
/// `next_pc = pc + instruction_length`. Two instances (carry_0/carry_1).
pub struct NextPcAddConstraint {
    carry_idx: usize,
    constraint_idx: usize,
}

impl NextPcAddConstraint {
    pub fn new(carry_idx: usize, constraint_idx: usize) -> Self {
        assert!(carry_idx <= 1);
        Self {
            carry_idx,
            constraint_idx,
        }
    }

    pub fn new_pair(constraint_idx_start: usize) -> (Self, Self) {
        (
            Self::new(0, constraint_idx_start),
            Self::new(1, constraint_idx_start + 1),
        )
    }

    fn compute_carry_0<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let pc_lo = step.get_main_evaluation_element(0, cols::PC_0).clone();
        let next_pc_lo = step.get_main_evaluation_element(0, cols::NEXT_PC_0).clone();
        let half_len = step
            .get_main_evaluation_element(0, cols::HALF_INSTRUCTION_LENGTH)
            .clone();
        let instr_len = &half_len + &half_len; // real byte length = 2 * half
        let inv_2_32 = FieldElement::<F>::from(super::templates::INV_SHIFT_32);
        (pc_lo + instr_len - next_pc_lo) * inv_2_32
    }

    fn compute_carry_1<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let pc_hi = step.get_main_evaluation_element(0, cols::PC_1).clone();
        let next_pc_hi = step.get_main_evaluation_element(0, cols::NEXT_PC_1).clone();
        let carry_0 = self.compute_carry_0(step);
        let inv_2_32 = FieldElement::<F>::from(super::templates::INV_SHIFT_32);
        (pc_hi + carry_0 - next_pc_hi) * inv_2_32
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for NextPcAddConstraint {
    fn degree(&self) -> usize {
        3
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let branch_cond = step
            .get_main_evaluation_element(0, cols::BRANCH_COND)
            .clone();
        let one = FieldElement::<F>::one();
        let not_branch = &one - branch_cond;
        let carry = match self.carry_idx {
            0 => self.compute_carry_0(step),
            1 => self.compute_carry_1(step),
            _ => unreachable!("carry_idx validated <= 1 at construction"),
        };
        not_branch * &carry * (one - carry)
    }
}

// =========================================================================
// alu group: ADD / SUB fast-path templates
// =========================================================================

/// ADD fast-path: `cond = ADD`, `rv1 + arg2 = cast(res, WL)`. Covers ADD, LOAD,
/// STORE and JAL(R) (all set `ADD`).
pub fn create_add_constraints(constraint_idx_start: usize) -> (Vec<AddConstraint>, usize) {
    let lhs = AddOperand::dword(cols::RV1_0);
    let rhs = AddOperand::dword(cols::ARG2_0);
    let sum = AddOperand::from_dword_hl(cols::RES_0);
    let (c0, c1) = AddConstraint::new_pair(vec![cols::ADD], lhs, rhs, sum, constraint_idx_start);
    (vec![c0, c1], constraint_idx_start + 2)
}

/// SUB fast-path: `cond = SUB`, `res = rv1 − arg2`, verified as `arg2 + res = rv1`.
pub fn create_sub_constraints(constraint_idx_start: usize) -> (Vec<AddConstraint>, usize) {
    let lhs = AddOperand::dword(cols::ARG2_0);
    let rhs = AddOperand::from_dword_hl(cols::RES_0);
    let sum = AddOperand::dword(cols::RV1_0);
    let (c0, c1) = AddConstraint::new_pair(vec![cols::SUB], lhs, rhs, sum, constraint_idx_start);
    (vec![c0, c1], constraint_idx_start + 2)
}

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

/// Creates all CPU transition constraints.
///
/// Returns `(is_bit_constraints, add_constraints, other_constraints, next_idx)`.
#[allow(clippy::type_complexity)]
pub fn create_all_cpu_constraints() -> (
    Vec<IsBitConstraint>,
    Vec<AddConstraint>,
    Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>>,
    usize,
) {
    let mut next_idx = 0;

    // range: IS_BIT
    let (is_bit, next) = create_is_bit_constraints(next_idx);
    next_idx = next;

    // alu: ADD + SUB fast-paths
    let (mut add_constraints, next) = create_add_constraints(next_idx);
    next_idx = next;
    let (sub, next) = create_sub_constraints(next_idx);
    next_idx = next;
    add_constraints.extend(sub);

    let mut other: Vec<
        Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>,
    > = Vec::new();

    // decode: word_instr mutex with MEMORY / BRANCH / ECALL, plus word_instr ⇒
    // {write,read1,read2}_register = 0 (word instructions are delegated to CPU32
    // and must not touch the main register file — leaving these free is unsound).
    // The register-read gates are spec-mandated ("out of caution").
    for &col in &[
        cols::MEMORY,
        cols::BRANCH,
        cols::ECALL,
        cols::WRITE_REGISTER,
        cols::READ_REGISTER1,
        cols::READ_REGISTER2,
    ] {
        other.push(ProductZeroConstraint::new(cols::WORD_INSTR, col, next_idx).boxed());
        next_idx += 1;
    }

    // alu: arg2 multiplex (low, high words)
    other.push(Arg2Constraint::new(0, next_idx).boxed());
    next_idx += 1;
    other.push(Arg2Constraint::new(1, next_idx).boxed());
    next_idx += 1;

    // mem: register zero-forcing (rv1/rv2 are DWordWL → 2 words each)
    for &value_col in &[cols::RV1_0, cols::RV1_1] {
        other.push(
            RegNotReadIsZeroConstraint::new(cols::READ_REGISTER1, value_col, next_idx).boxed(),
        );
        next_idx += 1;
    }
    for &value_col in &[cols::RV2_0, cols::RV2_1] {
        other.push(
            RegNotReadIsZeroConstraint::new(cols::READ_REGISTER2, value_col, next_idx).boxed(),
        );
        next_idx += 1;
    }

    // mem: ¬MEMORY ∧ ¬BRANCH ⇒ rvd = cast(res, WL)
    other.push(RvdEqResConstraint::new(0, next_idx).boxed());
    next_idx += 1;
    other.push(RvdEqResConstraint::new(1, next_idx).boxed());
    next_idx += 1;

    // branch: BRANCH ⇒ rvd = pc + instruction_length (JAL/JALR return), carry-aware
    let (branch_rvd_0, branch_rvd_1) = BranchRvdConstraint::new_pair(next_idx);
    other.push(branch_rvd_0.boxed());
    other.push(branch_rvd_1.boxed());
    next_idx += 2;

    // branch: branch_cond + next_pc
    other.push(BranchCondConstraint::new(next_idx).boxed());
    next_idx += 1;
    let (next_pc_0, next_pc_1) = NextPcAddConstraint::new_pair(next_idx);
    other.push(next_pc_0.boxed());
    other.push(next_pc_1.boxed());
    next_idx += 2;

    // assumptions (spec defense-in-depth, redundant with the DECODE lookup):
    // MEMORY/BRANCH mutex, arg2 multiplex exclusivity, and IS_BIT<mem_flags> on
    // non-memory rows.
    other.push(ProductZeroConstraint::new(cols::MEMORY, cols::BRANCH, next_idx).boxed());
    next_idx += 1;
    for &imm_col in &[cols::IMM_0, cols::IMM_1] {
        other.push(Arg2ExclusiveConstraint::new(imm_col, next_idx).boxed());
        next_idx += 1;
    }
    other.push(MemFlagsBitConstraint::new(next_idx).boxed());
    next_idx += 1;

    (is_bit, add_constraints, other, next_idx)
}

// =========================================================================
// ConstraintBuilder path: cpu_domain_eval
// =========================================================================

/// Folds every CPU transition constraint produced by [`create_all_cpu_constraints`]
/// into a [`ConstraintBuilder`], in the exact assembly (constraint_idx) order:
/// (1) the unconditional IS_BIT flag-column constraints, (2) the ADD then SUB
/// fast-path carry pairs, (3) the custom `other` constraints. Each residual is
/// field-exact to the corresponding boxed `evaluate` / `compute`.
pub fn cpu_domain_eval<CB: ConstraintBuilder>(cb: &mut CB) {
    // (1) range: IS_BIT (unconditional) for each flag column, in order.
    for &col in BIT_FLAG_COLUMNS {
        is_bit_fold(cb, None, col);
    }

    // (2) alu: ADD then SUB fast-path carry pairs (same AddOperands as
    // create_add_constraints / create_sub_constraints).
    // ADD: cond = ADD, rv1 + arg2 = cast(res, WL).
    let add_lhs = AddOperand::dword(cols::RV1_0);
    let add_rhs = AddOperand::dword(cols::ARG2_0);
    let add_sum = AddOperand::from_dword_hl(cols::RES_0);
    add_pair_fold(cb, &[cols::ADD], &add_lhs, &add_rhs, &add_sum);
    // SUB: cond = SUB, arg2 + res = rv1.
    let sub_lhs = AddOperand::dword(cols::ARG2_0);
    let sub_rhs = AddOperand::from_dword_hl(cols::RES_0);
    let sub_sum = AddOperand::dword(cols::RV1_0);
    add_pair_fold(cb, &[cols::SUB], &sub_lhs, &sub_rhs, &sub_sum);

    // (3) other custom constraints, in push order.
    let one = FieldElement::<CB::F>::one();

    // decode: word_instr * {MEMORY, BRANCH, ECALL, WRITE_REGISTER,
    // READ_REGISTER1, READ_REGISTER2} = 0 (ProductZeroConstraint).
    let word_instr = cb.main(cols::WORD_INSTR).clone();
    for &col in &[
        cols::MEMORY,
        cols::BRANCH,
        cols::ECALL,
        cols::WRITE_REGISTER,
        cols::READ_REGISTER1,
        cols::READ_REGISTER2,
    ] {
        let c = cb.main(col).clone();
        cb.fold(&word_instr * &c);
    }

    // alu: arg2 multiplex (low word then high word) -- Arg2Constraint.
    for &(arg2_col, imm_col, rv2_col) in &[
        (cols::ARG2_0, cols::IMM_0, cols::RV2_0),
        (cols::ARG2_1, cols::IMM_1, cols::RV2_1),
    ] {
        let arg2 = cb.main(arg2_col).clone();
        let imm = cb.main(imm_col).clone();
        let rv2 = cb.main(rv2_col).clone();
        let memory = cb.main(cols::MEMORY).clone();
        let branch = cb.main(cols::BRANCH).clone();
        // MEMORY*imm + BRANCH*rv2 + (1 - MEMORY - BRANCH)*(rv2 + imm).
        let mut expected = &memory * &imm;
        expected += &branch * &rv2;
        expected += (&one - &memory - &branch) * (&rv2 + &imm);
        cb.fold(arg2 - expected);
    }

    // mem: not read_registerN => rvN[i] = 0 (rv1 lo/hi then rv2 lo/hi) --
    // RegNotReadIsZeroConstraint.
    for &(flag_col, value_col) in &[
        (cols::READ_REGISTER1, cols::RV1_0),
        (cols::READ_REGISTER1, cols::RV1_1),
        (cols::READ_REGISTER2, cols::RV2_0),
        (cols::READ_REGISTER2, cols::RV2_1),
    ] {
        let flag = cb.main(flag_col).clone();
        let value = cb.main(value_col).clone();
        cb.fold((&one - &flag) * &value);
    }

    // mem: not MEMORY and not BRANCH => rvd = cast(res, WL) (low then high) --
    // RvdEqResConstraint. cast(res, WL) word = res_lo + res_hi*2^16.
    for high in [false, true] {
        let rvd_col = if high { cols::RVD_1 } else { cols::RVD_0 };
        let (lo_col, hi_col) = if high {
            (cols::RES_2, cols::RES_3)
        } else {
            (cols::RES_0, cols::RES_1)
        };
        let memory = cb.main(cols::MEMORY).clone();
        let branch = cb.main(cols::BRANCH).clone();
        let rvd = cb.main(rvd_col).clone();
        let res_lo = cb.main(lo_col).clone();
        let res_hi = cb.main(hi_col).clone();
        let shift_16 = FieldElement::<CB::F>::from(SHIFT_16);
        let res_w = &res_lo + &(&res_hi * &shift_16);
        cb.fold((&one - &memory - &branch) * (rvd - res_w));
    }

    // branch: BRANCH => rvd = pc + instruction_length (carry_0 then carry_1) --
    // BranchRvdConstraint. carry_1 reuses carry_0.
    {
        let pc_lo = cb.main(cols::PC_0).clone();
        let rvd_lo = cb.main(cols::RVD_0).clone();
        let pc_hi = cb.main(cols::PC_1).clone();
        let rvd_hi = cb.main(cols::RVD_1).clone();
        let half_len = cb.main(cols::HALF_INSTRUCTION_LENGTH).clone();
        let branch = cb.main(cols::BRANCH).clone();
        let inv = FieldElement::<CB::F>::from(INV_SHIFT_32);
        let instr_len = &half_len + &half_len; // real byte length = 2 * half
        let carry_0 = (&pc_lo + &instr_len - &rvd_lo) * &inv;
        let carry_1 = (&pc_hi + &carry_0 - &rvd_hi) * &inv;
        cb.fold(&branch * &carry_0 * (&one - &carry_0));
        cb.fold(&branch * &carry_1 * (&one - &carry_1));
    }

    // branch: branch_cond = BRANCH*JALR + BRANCH*(1 - JALR)*res[0] --
    // BranchCondConstraint (JALR = mem_flags under BRANCH).
    {
        let branch = cb.main(cols::BRANCH).clone();
        let jalr = cb.main(cols::MEM_FLAGS).clone();
        let res0 = cb.main(cols::RES_0).clone();
        let branch_cond = cb.main(cols::BRANCH_COND).clone();
        let expected = &branch * &jalr + &branch * (&one - &jalr) * &res0;
        cb.fold(branch_cond - expected);
    }

    // branch: (1 - branch_cond)*carry*(1 - carry) for next_pc = pc +
    // instruction_length (carry_0 then carry_1) -- NextPcAddConstraint.
    {
        let pc_lo = cb.main(cols::PC_0).clone();
        let next_pc_lo = cb.main(cols::NEXT_PC_0).clone();
        let pc_hi = cb.main(cols::PC_1).clone();
        let next_pc_hi = cb.main(cols::NEXT_PC_1).clone();
        let half_len = cb.main(cols::HALF_INSTRUCTION_LENGTH).clone();
        let branch_cond = cb.main(cols::BRANCH_COND).clone();
        let inv = FieldElement::<CB::F>::from(INV_SHIFT_32);
        let instr_len = &half_len + &half_len; // real byte length = 2 * half
        let carry_0 = (&pc_lo + &instr_len - &next_pc_lo) * &inv;
        let carry_1 = (&pc_hi + &carry_0 - &next_pc_hi) * &inv;
        let not_branch = &one - &branch_cond;
        cb.fold(&not_branch * &carry_0 * (&one - &carry_0));
        cb.fold(&not_branch * &carry_1 * (&one - &carry_1));
    }

    // assumptions: MEMORY*BRANCH mutex (ProductZeroConstraint).
    {
        let memory = cb.main(cols::MEMORY).clone();
        let branch = cb.main(cols::BRANCH).clone();
        cb.fold(&memory * &branch);
    }
    // assumptions: arg2 exclusivity (1 - MEMORY - BRANCH)*read_register2*imm[i]
    // for imm_0 then imm_1 (Arg2ExclusiveConstraint).
    for &imm_col in &[cols::IMM_0, cols::IMM_1] {
        let memory = cb.main(cols::MEMORY).clone();
        let branch = cb.main(cols::BRANCH).clone();
        let rr2 = cb.main(cols::READ_REGISTER2).clone();
        let imm = cb.main(imm_col).clone();
        cb.fold((&one - &memory - &branch) * &rr2 * &imm);
    }
    // assumptions: IS_BIT<mem_flags> on non-memory rows
    // (1 - MEMORY)*mem_flags*(1 - mem_flags) -- MemFlagsBitConstraint.
    {
        let memory = cb.main(cols::MEMORY).clone();
        let mem_flags = cb.main(cols::MEM_FLAGS).clone();
        cb.fold((&one - &memory) * &mem_flags * (&one - &mem_flags));
    }
}

pub struct CpuDomain;
impl TableConstraints<GoldilocksField, GoldilocksExtension> for CpuDomain {
    fn eval_prover(
        &self,
        cb: &mut ProverConstraintBuilder<GoldilocksField, GoldilocksExtension>,
        _ctx: &ConstraintContext<GoldilocksField, GoldilocksExtension>,
    ) {
        cpu_domain_eval(cb);
    }
    fn eval_verifier(
        &self,
        cb: &mut VerifierConstraintBuilder<GoldilocksExtension>,
        _ctx: &ConstraintContext<GoldilocksExtension, GoldilocksExtension>,
    ) {
        cpu_domain_eval(cb);
    }
}
