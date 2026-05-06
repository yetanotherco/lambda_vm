//! CPU table constraints for the 64-bit VM.
//!
//! This module defines the constraints for the CPU table, including:
//! - Range checks (IS_BIT) for all flag columns
//! - ALU constraints (ADD, SUB templates)
//! - Extension constraints (arg1, arg2, rvd computation)
//! - Branch condition computation
//! - next_pc computation
//!
//! ## Constraint Groups (from spec)
//!
//! 1. **Range checks**: IS_BIT for all bit flags (~25 constraints)
//! 2. **ALU**: ADD/SUB templates conditional on selectors
//! 3. **Extension**: arg1/arg2/rvd from rv1/rv2/res with sign extension
//! 4. **Misc**: branch_cond, next_pc computation

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::table::TableView;

use crate::tables::cpu::cols;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

use super::templates::{AddConstraint, AddLinearTerm, AddOperand, IsBitConstraint};

// =========================================================================
// CPU Constraint Collection
// =========================================================================

/// All bit flag columns that need IS_BIT constraints.
pub const BIT_FLAG_COLUMNS: &[usize] = &[
    cols::READ_REGISTER1,
    cols::READ_REGISTER2,
    cols::WRITE_REGISTER,
    cols::MEMORY_2BYTES,
    cols::MEMORY_4BYTES,
    cols::MEMORY_8BYTES,
    cols::C_TYPE_INSTRUCTION,
    cols::SIGNED,
    cols::MP_SELECTOR,
    cols::MULDIV_SELECTOR,
    cols::WORD_INSTR,
    // ALU selectors
    cols::ADD,
    cols::SUB,
    cols::SLT,
    cols::AND,
    cols::OR,
    cols::XOR,
    cols::SHIFT,
    cols::JALR,
    cols::BEQ,
    cols::BLT,
    cols::LOAD,
    cols::STORE,
    cols::MUL,
    cols::DIVREM,
    cols::ECALL,
    cols::EBREAK,
    // Sign bits
    cols::RV1_EXT_BIT,
    cols::RV2_EXT_BIT,
    cols::RES_EXT_BIT,
    // Computed flags
    cols::IS_EQUAL,
    cols::BRANCH_COND,
    // Inline PC columns
    cols::PREV_PC_TIMESTAMP_BORROW,
    cols::PC_DOUBLE_READ,
];

/// Creates all IS_BIT constraints for CPU flag columns.
///
/// Returns the constraints and the next available constraint index.
pub fn create_is_bit_constraints(constraint_idx_start: usize) -> (Vec<IsBitConstraint>, usize) {
    super::templates::new_is_bit_constraints(BIT_FLAG_COLUMNS, constraint_idx_start)
}

// =========================================================================
// ALU ADD Constraints
// =========================================================================

/// Creates ADD constraints for the CPU table.
///
/// ADD template is used when: ADD + LOAD + STORE > 0
/// - ADD: arg1 + arg2 = res (arithmetic addition)
/// - LOAD/STORE: base_address + offset = effective_address (in res)
///
/// Returns the constraints and the next available constraint index.
pub fn create_add_constraints(constraint_idx_start: usize) -> (Vec<AddConstraint>, usize) {
    // For ADD/LOAD operations, we compute: arg1 + arg2 = res
    // Phase 2: operands are now DWordWL on the u32 limb cols (ARG1_LO/HI etc).
    // The carry chain shrinks from 8 byte carries to 2 u32 carries — handled
    // automatically by AddConstraint::new_pair on DWordWL inputs.

    let lhs = AddOperand::dword(cols::ARG1_LO);
    let rhs = AddOperand::dword(cols::ARG2_LO);
    let sum = AddOperand::dword(cols::RES_LO);

    // Condition: ADD + LOAD (active when any of these flags is set)
    let cond_cols = vec![cols::ADD, cols::LOAD];

    let (add_c0, add_c1) = AddConstraint::new_pair(cond_cols, lhs, rhs, sum, constraint_idx_start);

    // STORE: res = arg1 + imm (separate ADD, because arg2 now holds rv2)
    let store_lhs = AddOperand::dword(cols::ARG1_LO);
    let store_rhs = AddOperand::dword(cols::IMM_0);
    let store_sum = AddOperand::dword(cols::RES_LO);
    let store_cond = vec![cols::STORE];
    let (store_c0, store_c1) = AddConstraint::new_pair(
        store_cond,
        store_lhs,
        store_rhs,
        store_sum,
        constraint_idx_start + 2,
    );

    (
        vec![add_c0, add_c1, store_c0, store_c1],
        constraint_idx_start + 4,
    )
}

// =========================================================================
// Branch Condition Constraint
// =========================================================================

/// Constraint for branch_cond computation.
///
/// From spec:
/// branch_cond = JALR
///             + BLT * (res[0] XOR mp_selector)
///             + BEQ * (is_equal XOR mp_selector)
///
/// Where XOR is computed as: a XOR b = a + b - 2*a*b
///
/// Phase 2 step 9: `res[0]` is treated as a bit (0 or 1) here — it's the
/// LT outcome from the SLT/BLT bus. `SltResLoUpperBytesZeroConstraint`
/// guarantees `RES_LO ∈ {0, 1}` and `RES_HI = 0` under SLT/BLT, so
/// `RES_LO == res[0]` and we can read the u32 limb directly.
pub struct BranchCondConstraint {
    constraint_idx: usize,
}

impl BranchCondConstraint {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let jalr = step.get_main_evaluation_element(0, cols::JALR).clone();
        let blt = step.get_main_evaluation_element(0, cols::BLT).clone();
        let beq = step.get_main_evaluation_element(0, cols::BEQ).clone();
        let mp_selector = step
            .get_main_evaluation_element(0, cols::MP_SELECTOR)
            .clone();
        let res_lo = step.get_main_evaluation_element(0, cols::RES_LO).clone();
        let is_equal = step.get_main_evaluation_element(0, cols::IS_EQUAL).clone();
        let branch_cond = step
            .get_main_evaluation_element(0, cols::BRANCH_COND)
            .clone();

        let two = FieldElement::<F>::from(2u64);

        // XOR computation: a XOR b = a + b - 2*a*b
        // res_lo XOR mp_selector  (res_lo is 0 or 1 under BLT)
        let res_xor_mp = &res_lo + &mp_selector - &two * &res_lo * &mp_selector;
        // is_equal XOR mp_selector
        let eq_xor_mp = &is_equal + &mp_selector - &two * &is_equal * &mp_selector;

        // branch_cond = JALR + BLT * res_xor_mp + BEQ * eq_xor_mp
        let expected = jalr + &blt * res_xor_mp + &beq * eq_xor_mp;

        // Constraint: branch_cond - expected = 0
        branch_cond - expected
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for BranchCondConstraint {
    fn degree(&self) -> usize {
        // BLT * res_0 * mp_selector has degree 3
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
        self.compute(step)
    }
}

// =========================================================================
// BEQ zero-test constraints (replaces ZERO bus sender, Phase 2 step 9A)
// =========================================================================
//
// `is_equal` must equal `(res == 0)` on BEQ rows. The prior implementation
// sent `(sum(res[0..7]), is_equal)` to the BITWISE-hosted ZERO bus, which
// only covers a 20-bit input space. To migrate off the byte decomposition
// without widening the preprocessed table, we use a witness-inverse
// pattern over `sum = RES_LO + RES_HI` (each limb in [0, 2^32), so
// sum == 0 iff res == 0):
//
//   sum_constraint:    BEQ * IS_EQUAL * (RES_LO + RES_HI) = 0
//   witness_constraint: BEQ * (1 - IS_EQUAL - (RES_LO + RES_HI) * RES_INV) = 0
//
// On non-BEQ rows both vanish (BEQ = 0), leaving IS_EQUAL unconstrained
// — same as before. On BEQ rows:
//   * sum = 0:     is_equal must be 1 (sum_constraint trivially holds;
//                  witness_constraint forces 1 - is_equal = 0).
//   * sum != 0:    is_equal must be 0 (sum_constraint forces it; the
//                  prover witnesses RES_INV = 1/sum to satisfy
//                  witness_constraint).
// IS_EQUAL is already IS_BIT-constrained globally, so 0/1 valuation is
// unconditionally enforced.

/// `BEQ * IS_EQUAL * (RES_LO + RES_HI) = 0`
pub struct BeqIsEqualSumZeroConstraint {
    constraint_idx: usize,
}

impl BeqIsEqualSumZeroConstraint {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let beq = step.get_main_evaluation_element(0, cols::BEQ).clone();
        let is_equal = step.get_main_evaluation_element(0, cols::IS_EQUAL).clone();
        let res_lo = step.get_main_evaluation_element(0, cols::RES_LO).clone();
        let res_hi = step.get_main_evaluation_element(0, cols::RES_HI).clone();

        beq * is_equal * (res_lo + res_hi)
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for BeqIsEqualSumZeroConstraint {
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
        self.compute(step)
    }
}

/// `BEQ * (1 - IS_EQUAL - (RES_LO + RES_HI) * RES_INV) = 0`
pub struct BeqIsEqualWitnessConstraint {
    constraint_idx: usize,
}

impl BeqIsEqualWitnessConstraint {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let beq = step.get_main_evaluation_element(0, cols::BEQ).clone();
        let is_equal = step.get_main_evaluation_element(0, cols::IS_EQUAL).clone();
        let res_lo = step.get_main_evaluation_element(0, cols::RES_LO).clone();
        let res_hi = step.get_main_evaluation_element(0, cols::RES_HI).clone();
        let res_inv = step.get_main_evaluation_element(0, cols::RES_INV).clone();

        let one = FieldElement::<F>::one();
        beq * (one - is_equal - (res_lo + res_hi) * res_inv)
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for BeqIsEqualWitnessConstraint {
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
        self.compute(step)
    }
}

// =========================================================================
// EBREAK Constraint
// =========================================================================

/// Constraint that EBREAK must be 0 (unprovable trap).
///
/// From spec: !EBREAK (we treat EBREAK as an unprovable trap)
pub struct EbreakConstraint {
    constraint_idx: usize,
}

impl EbreakConstraint {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        // EBREAK must be 0
        step.get_main_evaluation_element(0, cols::EBREAK).clone()
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for EbreakConstraint {
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        self.compute(step)
    }
}

// =========================================================================
// Extension Constraints
// =========================================================================

/// Constraint: arg1[0:4] = rv1[0:2] (lower 32 bits match)
///
/// arg1 is DWordBL (8 bytes), rv1 is DWordWHH [Half, Half, Word]
/// arg1[:4] as word = rv1[0] + rv1[1] * 2^16 (two halves make a word)
///
/// Spec (CPU-CE54): arg1::DWordWL[0] - rv1::DWordWL[0] = 0
pub struct Arg1LowerConstraint {
    constraint_idx: usize,
}

impl Arg1LowerConstraint {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        // Phase 2: u32 limb cells directly.
        let arg1_lo = step.get_main_evaluation_element(0, cols::ARG1_LO).clone();

        // rv1 is DWordWHH: [Half(0-15), Half(16-31), Word(32-63)]
        // rv1::DWordWL[0] = rv1[0] + rv1[1] * 2^16
        let rv1_0 = step.get_main_evaluation_element(0, cols::RV1_0);
        let rv1_1 = step.get_main_evaluation_element(0, cols::RV1_1);
        let shift_16: FieldElement<F> = FieldElement::from(1u64 << 16);
        let rv1_lower = rv1_0 + rv1_1 * shift_16;

        // Constraint: arg1_lo - rv1_lower = 0
        arg1_lo - rv1_lower
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for Arg1LowerConstraint {
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        self.compute(step)
    }
}

/// Constraint: arg1[4:8] = rv1[2] * (1 - word_instr) + (2^32 - 1) * rv1_ext_bit * signed
///
/// Upper 32 bits of arg1 depends on word_instr and sign extension.
pub struct Arg1UpperConstraint {
    constraint_idx: usize,
}

impl Arg1UpperConstraint {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        // Phase 2: u32 limb cells directly.
        let arg1_hi = step.get_main_evaluation_element(0, cols::ARG1_HI).clone();

        // rv1 is DWordWHH: rv1[2] IS the upper 32 bits directly (Word)
        let rv1_upper = step.get_main_evaluation_element(0, cols::RV1_2);

        let word_instr = step
            .get_main_evaluation_element(0, cols::WORD_INSTR)
            .clone();
        let signed = step.get_main_evaluation_element(0, cols::SIGNED).clone();
        let rv1_ext_bit = step
            .get_main_evaluation_element(0, cols::RV1_EXT_BIT)
            .clone();

        let one = FieldElement::<F>::one();
        let mask_32: FieldElement<F> = FieldElement::from((1u64 << 32) - 1); // 2^32 - 1

        // Expected: rv1_upper * (1 - word_instr) + mask_32 * rv1_ext_bit * signed
        let expected = rv1_upper * (one - &word_instr) + mask_32 * rv1_ext_bit * signed;

        // Constraint: arg1_hi - expected = 0
        arg1_hi - expected
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for Arg1UpperConstraint {
    fn degree(&self) -> usize {
        // rv1_ext_bit * signed * word_instr has degree 3
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
        self.compute(step)
    }
}

// =========================================================================
// SLT/BLT Zero Upper Bytes Constraints (u32-limb form)
// =========================================================================

/// Constraint: when SLT + BLT = 1, RES_LO is 0 or 1.
///
/// LT outputs a single bit. Combined with `SltResHiZeroConstraint` this
/// guarantees the full 64-bit `res` is in `{0, 1}` under SLT/BLT.
///
/// Phase 2 step 9: prior form was `(SLT + BLT) * (RES_LO - RES[0]) = 0`,
/// which read the now-removed byte cell `RES[0]`. The boolean check
/// `RES_LO * (RES_LO - 1) = 0` is the direct u32-limb expression of the
/// same invariant.
pub struct SltResLoUpperBytesZeroConstraint {
    constraint_idx: usize,
}

impl SltResLoUpperBytesZeroConstraint {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let slt = step.get_main_evaluation_element(0, cols::SLT).clone();
        let blt = step.get_main_evaluation_element(0, cols::BLT).clone();
        let res_lo = step.get_main_evaluation_element(0, cols::RES_LO).clone();

        let one = FieldElement::<F>::one();
        // (SLT + BLT) * RES_LO * (RES_LO - 1) = 0
        (slt + blt) * &res_lo * (res_lo - one)
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension>
    for SltResLoUpperBytesZeroConstraint
{
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
        self.compute(step)
    }
}

/// Constraint: when SLT + BLT = 1, the entire `res_hi` is zero.
///
/// Equivalent to `(SLT + BLT) * RES_HI = 0`.
pub struct SltResHiZeroConstraint {
    constraint_idx: usize,
}

impl SltResHiZeroConstraint {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let slt = step.get_main_evaluation_element(0, cols::SLT).clone();
        let blt = step.get_main_evaluation_element(0, cols::BLT).clone();
        let res_hi = step.get_main_evaluation_element(0, cols::RES_HI).clone();

        // (SLT + BLT) * RES_HI = 0
        (slt + blt) * res_hi
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for SltResHiZeroConstraint {
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
        self.compute(step)
    }
}

/// Creates the two SLT/BLT zero constraints over u32 limbs.
///
/// Returns boxed evaluators so the two distinct constraint types can be
/// stored in the same `other` constraint vector.
pub fn create_slt_res_zero_constraints(
    constraint_idx_start: usize,
) -> (
    Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>>,
    usize,
) {
    let constraints: Vec<
        Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>,
    > = vec![
        SltResLoUpperBytesZeroConstraint::new(constraint_idx_start).boxed(),
        SltResHiZeroConstraint::new(constraint_idx_start + 1).boxed(),
    ];

    (constraints, constraint_idx_start + 2)
}

// =========================================================================
// Extension Bit Constraints (SIGN template from spec)
// =========================================================================

/// Constraint: ext_bit must be zero when word_instr = 0
///
/// (1 - word_instr) * ext_bit = 0
///
/// One instance per extension bit (rv1_ext_bit, rv2_ext_bit, res_ext_bit).
pub struct ExtBitZeroConstraint {
    constraint_idx: usize,
    ext_bit_col: usize,
}

impl ExtBitZeroConstraint {
    pub fn new(constraint_idx: usize, ext_bit_col: usize) -> Self {
        Self {
            constraint_idx,
            ext_bit_col,
        }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let ext_bit = step
            .get_main_evaluation_element(0, self.ext_bit_col)
            .clone();
        let word_instr = step
            .get_main_evaluation_element(0, cols::WORD_INSTR)
            .clone();

        let one = FieldElement::<F>::one();

        // (1 - word_instr) * ext_bit = 0
        (one - word_instr) * ext_bit
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for ExtBitZeroConstraint {
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
        self.compute(step)
    }
}

// =========================================================================
// Next PC (Non-Branching) Constraint
// =========================================================================

/// Constraint: when branch_cond = 0, next_pc = pc + instr_size
///
/// where instr_size = 4 - 2 * c_type_instruction
/// (4 bytes for normal instructions, 2 bytes for compressed)
///
/// Uses the same carry-based approach as AddConstraint but with
/// condition `(1 - branch_cond)` instead of a column value.
pub struct NextPcAddConstraint {
    /// Which carry constraint this is (0 or 1)
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

    /// Creates constraints for both carries.
    pub fn new_pair(constraint_idx_start: usize) -> (Self, Self) {
        (
            Self::new(0, constraint_idx_start),
            Self::new(1, constraint_idx_start + 1),
        )
    }

    /// Compute carry_0 = (pc_lo + instr_size - next_pc_lo) / 2^32
    fn compute_carry_0<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let pc_lo = step.get_main_evaluation_element(0, cols::PC_0).clone();
        let next_pc_lo = step.get_main_evaluation_element(0, cols::NEXT_PC_0).clone();
        let c_type = step
            .get_main_evaluation_element(0, cols::C_TYPE_INSTRUCTION)
            .clone();

        // instr_size = 4 - 2 * c_type_instruction
        let four: FieldElement<F> = FieldElement::from(4u64);
        let two: FieldElement<F> = FieldElement::from(2u64);
        let instr_size = four - two * c_type;

        // carry_0 = (pc_lo + instr_size - next_pc_lo) * 2^(-32)
        let inv_2_32 = FieldElement::<F>::from(super::templates::INV_SHIFT_32);
        (pc_lo + instr_size - next_pc_lo) * inv_2_32
    }

    /// Compute carry_1 = (pc_hi + carry_0 - next_pc_hi) / 2^32
    fn compute_carry_1<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let pc_hi = step.get_main_evaluation_element(0, cols::PC_1).clone();
        let next_pc_hi = step.get_main_evaluation_element(0, cols::NEXT_PC_1).clone();
        let carry_0 = self.compute_carry_0(step);

        // rhs_hi = 0 (instruction size fits in low word)
        // carry_1 = (pc_hi + 0 + carry_0 - next_pc_hi) * 2^(-32)
        let inv_2_32 = FieldElement::<F>::from(super::templates::INV_SHIFT_32);
        (pc_hi + carry_0 - next_pc_hi) * inv_2_32
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
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
            _ => panic!("Invalid carry index"),
        };

        // (1 - branch_cond) * carry * (1 - carry)
        not_branch * &carry * (one - carry)
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for NextPcAddConstraint {
    fn degree(&self) -> usize {
        // (1 - branch_cond) * carry * (1 - carry) has degree 3
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
        self.compute(step)
    }
}

// =========================================================================
// Arg2 Constraints
// =========================================================================

/// Constraint: arg2[:4] = (1-LOAD)*rv2[:2] + (1-BEQ-BLT-STORE)*imm[0]
///
/// arg2 lower 32 bits comes from either rv2 or imm depending on instruction type.
pub struct Arg2LowerConstraint {
    constraint_idx: usize,
}

impl Arg2LowerConstraint {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        // Phase 2: u32 limb cells directly.
        let arg2_lo = step.get_main_evaluation_element(0, cols::ARG2_LO).clone();

        // rv2 is DWordWHH: rv2[:2] = rv2[0] + rv2[1] * 2^16
        let rv2_0 = step.get_main_evaluation_element(0, cols::RV2_0);
        let rv2_1 = step.get_main_evaluation_element(0, cols::RV2_1);
        let shift_16: FieldElement<F> = FieldElement::from(1u64 << 16);
        let rv2_lower = rv2_0 + rv2_1 * shift_16;

        // imm[0] is lower word of immediate
        let imm_0 = step.get_main_evaluation_element(0, cols::IMM_0);

        // Selectors
        let store = step.get_main_evaluation_element(0, cols::STORE);
        let load = step.get_main_evaluation_element(0, cols::LOAD);
        let beq = step.get_main_evaluation_element(0, cols::BEQ);
        let blt = step.get_main_evaluation_element(0, cols::BLT);

        let one = FieldElement::<F>::one();

        // (1-LOAD) * rv2_lower + (1-BEQ-BLT-STORE) * imm[0]
        // STORE now gets rv2 (via rv2_lower), not imm
        let expected = (&one - load) * rv2_lower + (&one - beq - blt - store) * imm_0;

        // Constraint: arg2_lo - expected = 0
        arg2_lo - expected
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for Arg2LowerConstraint {
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
        self.compute(step)
    }
}

/// Constraint: arg2[4:] = (1-LOAD)*((1-word_instr)*rv2[2] + signed*rv2_ext_bit*(2^32-1)) + (1-BEQ-BLT-STORE)*imm[1]
///
/// arg2 upper 32 bits with sign extension logic.
pub struct Arg2UpperConstraint {
    constraint_idx: usize,
}

impl Arg2UpperConstraint {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        // Phase 2: u32 limb cells directly.
        let arg2_hi = step.get_main_evaluation_element(0, cols::ARG2_HI).clone();

        // rv2 is DWordWHH: rv2[2] IS the upper 32 bits directly (Word)
        let rv2_upper = step.get_main_evaluation_element(0, cols::RV2_2);

        // imm[1] is upper word of immediate
        let imm_1 = step.get_main_evaluation_element(0, cols::IMM_1);

        // Flags
        let store = step.get_main_evaluation_element(0, cols::STORE);
        let load = step.get_main_evaluation_element(0, cols::LOAD);
        let beq = step.get_main_evaluation_element(0, cols::BEQ);
        let blt = step.get_main_evaluation_element(0, cols::BLT);
        let word_instr = step.get_main_evaluation_element(0, cols::WORD_INSTR);
        let signed = step.get_main_evaluation_element(0, cols::SIGNED);
        let rv2_ext_bit = step.get_main_evaluation_element(0, cols::RV2_EXT_BIT);

        let one = FieldElement::<F>::one();
        let mask_32: FieldElement<F> = FieldElement::from((1u64 << 32) - 1);

        // rv2_term = (1 - word_instr) * rv2[2] + signed * rv2_ext_bit * (2^32 - 1)
        let rv2_term = (&one - word_instr) * rv2_upper + signed * rv2_ext_bit * &mask_32;

        // expected = (1-LOAD) * rv2_term + (1-BEQ-BLT-STORE) * imm[1]
        // STORE now gets rv2_term (with sign extension), not imm
        let expected = (&one - load) * rv2_term + (&one - beq - blt - store) * imm_1;

        // Constraint: arg2_hi - expected = 0
        arg2_hi - expected
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for Arg2UpperConstraint {
    fn degree(&self) -> usize {
        // (1-LOAD) * signed * rv2_ext_bit has degree 3
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
        self.compute(step)
    }
}

// =========================================================================
// RVD Constraints
// =========================================================================

/// Constraint: (1-LOAD) * (rvd[0] - res[:4]) = 0
///
/// When not LOAD, rvd lower 32 bits equals res lower 32 bits.
/// For LOAD: rvd is the loaded value, not res (which is the address).
/// For non-LOAD ops (including STORE): rvd must equal res in the trace.
pub struct RvdLowerConstraint {
    constraint_idx: usize,
}

impl RvdLowerConstraint {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        // rvd[0] is lower word
        let rvd_0 = step.get_main_evaluation_element(0, cols::RVD_0);

        // Phase 2: u32 limb cells directly.
        let res_lo = step.get_main_evaluation_element(0, cols::RES_LO).clone();

        let load = step.get_main_evaluation_element(0, cols::LOAD);
        let one = FieldElement::<F>::one();

        // (1 - LOAD) * (rvd[0] - res_lo) = 0
        (one - load) * (rvd_0 - res_lo)
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for RvdLowerConstraint {
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
        self.compute(step)
    }
}

/// Constraint: (1-LOAD) * (rvd[1] - ((1-word_instr)*res[4:] + res_ext_bit*(2^32-1))) = 0
///
/// When not LOAD, rvd upper 32 bits equals res upper with sign extension.
/// For LOAD: rvd is the loaded value, not res (which is the address).
/// For non-LOAD ops (including STORE): rvd must equal res in the trace.
pub struct RvdUpperConstraint {
    constraint_idx: usize,
}

impl RvdUpperConstraint {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        // rvd[1] is upper word
        let rvd_1 = step.get_main_evaluation_element(0, cols::RVD_1);

        // Phase 2: u32 limb cells directly.
        let res_hi = step.get_main_evaluation_element(0, cols::RES_HI).clone();

        let load = step.get_main_evaluation_element(0, cols::LOAD);
        let word_instr = step.get_main_evaluation_element(0, cols::WORD_INSTR);
        let res_ext_bit = step.get_main_evaluation_element(0, cols::RES_EXT_BIT);

        let one = FieldElement::<F>::one();
        let mask_32: FieldElement<F> = FieldElement::from((1u64 << 32) - 1);

        // expected = (1 - word_instr) * res_hi + res_ext_bit * (2^32 - 1)
        let expected = (&one - word_instr) * res_hi + res_ext_bit * mask_32;

        // (1 - LOAD) * (rvd[1] - expected) = 0
        (one - load) * (rvd_1 - expected)
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for RvdUpperConstraint {
    fn degree(&self) -> usize {
        // (1-LOAD) * (1-word_instr) * res_hi has degree 3
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
        self.compute(step)
    }
}

// =========================================================================
// read_register - register Constraints (CM48, CM50)
// =========================================================================

/// Constraint: `(1 - flag_col) * value_col = 0`
///
/// Forces `value_col` to zero whenever `flag_col` is 0.
///
/// Used for:
/// - CPU-CM48.i: `(1 - read_register1) * rv1[i] = 0` for i ∈ [0, 2]
///   When read_register1 = 0 (rs1 is x0), rv1 is not loaded from memory,
///   so it must be forced to zero by a polynomial constraint.
/// - CPU-CM50.i: `(1 - read_register2) * rv2[i] = 0` for i ∈ [0, 2]
///   Same logic for rv2 when read_register2 = 0 (I-type instructions).
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

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let flag = step.get_main_evaluation_element(0, self.flag_col).clone();
        let value = step.get_main_evaluation_element(0, self.value_col).clone();
        let one = FieldElement::<F>::one();
        // (1 - flag) * value = 0
        (one - flag) * value
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
        self.compute(step)
    }
}

// =========================================================================
// SUB Constraints
// =========================================================================

/// Creates SUB constraints for the CPU table.
///
/// SUB template is used when: SUB + BEQ > 0
/// - SUB: res = arg1 - arg2
/// - BEQ: computes arg1 - arg2 to check equality (res = 0 means equal)
///
/// Verifies: arg2 + res = arg1 (subtraction expressed as addition)
///
/// Returns the constraints and the next available constraint index.
pub fn create_sub_constraints(constraint_idx_start: usize) -> (Vec<AddConstraint>, usize) {
    // SUB is verified as: arg2 + res = arg1.
    // Phase 2: u32-limb operands; AddConstraint::new_pair emits 2 carry
    // constraints instead of 8 byte carries.
    let lhs = AddOperand::dword(cols::ARG2_LO);
    let rhs = AddOperand::dword(cols::RES_LO);
    let sum = AddOperand::dword(cols::ARG1_LO);

    // Condition: SUB + BEQ (active when either flag is set)
    let cond_cols = vec![cols::SUB, cols::BEQ];

    let (sub_c0, sub_c1) = AddConstraint::new_pair(cond_cols, lhs, rhs, sum, constraint_idx_start);

    (vec![sub_c0, sub_c1], constraint_idx_start + 2)
}

// =========================================================================
// JALR Result Constraint
// =========================================================================

/// Creates JALR result constraints using the ADD template.
///
/// JALR: res = pc + instr_size (return address)
/// where instr_size = 4 - 2 * c_type_instruction
///
/// This uses proper 64-bit addition with carry handling.
pub fn create_jalr_constraints(constraint_idx_start: usize) -> (Vec<AddConstraint>, usize) {
    // pc is stored as DWordWL (2 consecutive columns)
    let pc = AddOperand::dword(cols::PC_0);

    // instr_size = 4 - 2 * c_type_instruction
    // This is a linear expression with only a low word (hi = 0)
    let instr_size = AddOperand::linear(
        vec![
            AddLinearTerm::Constant(4),
            AddLinearTerm::Column {
                coefficient: -2,
                column: cols::C_TYPE_INSTRUCTION,
            },
        ],
        vec![], // hi = 0
    );

    // res is stored as u32 limbs (Phase 2)
    let res = AddOperand::dword(cols::RES_LO);

    // Condition: JALR
    let cond_cols = vec![cols::JALR];

    let (jalr_c0, jalr_c1) =
        AddConstraint::new_pair(cond_cols, pc, instr_size, res, constraint_idx_start);

    (vec![jalr_c0, jalr_c1], constraint_idx_start + 2)
}

// =========================================================================
// Inline PC Constraints
// =========================================================================
//
// Per spec/cpu.typ: "Constraints on `pc_double_read` corresponding to an `AUIPC`
// instruction are not necessary, as regardless of its value, the old timestamp is
// guaranteed smaller than the new timestamp, and the integrity of the memory
// argument therefore ensures the correctness of this bit."
//
// The IS_BIT constraints on PC_DOUBLE_READ and PREV_PC_TIMESTAMP_BORROW are
// sufficient; no extra algebraic constraints linking them to rs1/read_register1
// or to each other are required.

// =========================================================================
// Constraint Summary
// =========================================================================

/// Total number of CPU constraints.
///
/// - IS_BIT: 32 (all bit flags, including read_register1/2)
/// - ADD carry: 2 (for ADD + LOAD)
/// - STORE ADD carry: 2 (for STORE: res = arg1 + imm)
/// - SUB carry: 2 (for SUB + BEQ)
/// - JALR carry: 2 (res = pc + instr_size)
/// - Branch cond: 1
/// - EBREAK: 1
/// - Arg1 lower: 1
/// - Arg1 upper: 1
/// - Arg2 lower: 1
/// - Arg2 upper: 1
/// - Rvd lower: 1
/// - Rvd upper: 1
/// - SLT res zero: 2 (u32-limb form: res_lo upper bytes + res_hi)
/// - Ext bit zero (SIGN template): 3 (rv1_ext_bit, rv2_ext_bit, res_ext_bit)
/// - rv1 zero-forcing (CM48): 3 (rv1[0..2] when read_register1 = 0)
/// - rv2 zero-forcing (CM50): 3 (rv2[0..2] when read_register2 = 0)
/// - Next PC (non-branching): 2
///
/// Total: 65 constraints (34 IS_BIT + 8 ADD + 23 other)
/// (The inline PC columns PC_DOUBLE_READ and PREV_PC_TIMESTAMP_BORROW are
/// IS_BIT-constrained; per spec/cpu.typ no additional algebraic constraints
/// are required.)
///
/// "other" includes: branch_cond (1), BEQ zero-test (2 — sum and witness),
/// ebreak (1), arg1/arg2/rvd lower+upper (6), SLT res zero (2), ext bit
/// zero (3), rv1/rv2 zero-forcing (6), next_pc (2).
pub const NUM_CPU_CONSTRAINTS: usize = 34   // IS_BIT
    + 2 + 2 + 2 + 2                          // ADD: ADD/LOAD + STORE addr + SUB/BEQ + JALR
    + 1                                      // branch_cond
    + 2                                      // BEQ zero-test (sum + witness)
    + 1                                      // ebreak
    + 3 + 3                                  // rv1/rv2 zero-forcing
    + 1 + 1                                  // arg1 lower/upper
    + 1 + 1                                  // arg2 lower/upper
    + 1 + 1                                  // rvd lower/upper
    + 2                                      // SLT res zero (lo bit + hi zero)
    + 3                                      // ext bit zero (rv1, rv2, res)
    + 2; // next_pc (cond + uncond)

/// Creates all CPU constraints.
///
/// Returns a tuple of (is_bit_constraints, add_constraints, other_constraints, next_idx)
#[allow(clippy::type_complexity)]
pub fn create_all_cpu_constraints() -> (
    Vec<IsBitConstraint>,
    Vec<AddConstraint>,
    Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>>,
    usize,
) {
    let mut next_idx = 0;

    // IS_BIT constraints
    let (is_bit, next) = create_is_bit_constraints(next_idx);
    next_idx = next;

    // ADD constraints (for ADD + LOAD + STORE)
    let (mut add_constraints, next) = create_add_constraints(next_idx);
    next_idx = next;

    // SUB constraints (for SUB + BEQ)
    let (sub, next) = create_sub_constraints(next_idx);
    next_idx = next;
    add_constraints.extend(sub);

    // JALR constraints (res = pc + instr_size)
    let (jalr, next) = create_jalr_constraints(next_idx);
    next_idx = next;
    add_constraints.extend(jalr);

    // Other constraints
    let mut other: Vec<
        Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>,
    > = Vec::new();

    // Branch condition
    other.push(BranchCondConstraint::new(next_idx).boxed());
    next_idx += 1;

    // BEQ zero-test (replaces ZERO bus, Phase 2 step 9A)
    other.push(BeqIsEqualSumZeroConstraint::new(next_idx).boxed());
    next_idx += 1;
    other.push(BeqIsEqualWitnessConstraint::new(next_idx).boxed());
    next_idx += 1;

    // EBREAK
    other.push(EbreakConstraint::new(next_idx).boxed());
    next_idx += 1;

    // rv1 zero-forcing (CM48): (1 - read_register1) * rv1[i] = 0 for i ∈ [0, 2]
    for &value_col in &[cols::RV1_0, cols::RV1_1, cols::RV1_2] {
        other.push(
            RegNotReadIsZeroConstraint::new(cols::READ_REGISTER1, value_col, next_idx).boxed(),
        );
        next_idx += 1;
    }

    // rv2 zero-forcing (CM50): (1 - read_register2) * rv2[i] = 0 for i ∈ [0, 2]
    for &value_col in &[cols::RV2_0, cols::RV2_1, cols::RV2_2] {
        other.push(
            RegNotReadIsZeroConstraint::new(cols::READ_REGISTER2, value_col, next_idx).boxed(),
        );
        next_idx += 1;
    }

    // Arg1 constraints
    other.push(Arg1LowerConstraint::new(next_idx).boxed());
    next_idx += 1;
    other.push(Arg1UpperConstraint::new(next_idx).boxed());
    next_idx += 1;

    // Arg2 constraints
    other.push(Arg2LowerConstraint::new(next_idx).boxed());
    next_idx += 1;
    other.push(Arg2UpperConstraint::new(next_idx).boxed());
    next_idx += 1;

    // Rvd constraints
    other.push(RvdLowerConstraint::new(next_idx).boxed());
    next_idx += 1;
    other.push(RvdUpperConstraint::new(next_idx).boxed());
    next_idx += 1;

    // SLT res zero constraints
    let (slt_zero, next) = create_slt_res_zero_constraints(next_idx);
    next_idx = next;
    other.extend(slt_zero);

    // Extension bit zero constraints (SIGN template: !word_instr => ext_bit = 0)
    other.push(ExtBitZeroConstraint::new(next_idx, cols::RV1_EXT_BIT).boxed());
    next_idx += 1;
    other.push(ExtBitZeroConstraint::new(next_idx, cols::RV2_EXT_BIT).boxed());
    next_idx += 1;
    other.push(ExtBitZeroConstraint::new(next_idx, cols::RES_EXT_BIT).boxed());
    next_idx += 1;

    // Next PC (non-branching) constraints
    let (next_pc_0, next_pc_1) = NextPcAddConstraint::new_pair(next_idx);
    other.push(next_pc_0.boxed());
    other.push(next_pc_1.boxed());
    next_idx += 2;

    (is_bit, add_constraints, other, next_idx)
}
