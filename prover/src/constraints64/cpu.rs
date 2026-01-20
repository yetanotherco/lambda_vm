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
use stark::constraints::transition::TransitionConstraint;
use stark::table::TableView;
use stark::traits::TransitionEvaluationContext;

use crate::tables64::cpu::cols;
use crate::tables64::types::{GoldilocksExtension, GoldilocksField};

use super::templates::{AddConstraint, AddOperand, IsBitConstraint};

// =========================================================================
// CPU Constraint Collection
// =========================================================================

/// All bit flag columns that need IS_BIT constraints.
pub const BIT_FLAG_COLUMNS: &[usize] = &[
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
    cols::RV1_SIGN_BIT,
    cols::ARG2_SIGN_BIT,
    cols::RES_SIGN_BIT,
    // Computed flags
    cols::IS_EQUAL,
    cols::BRANCH_COND,
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
/// ADD is used when: ADD + LOAD + STORE = 1
/// - ADD: arg1 + arg2 = res
/// - LOAD/STORE: base_address + offset = effective_address (in res)
///
/// Returns the constraints and the next available constraint index.
pub fn create_add_constraints(constraint_idx_start: usize) -> (Vec<AddConstraint>, usize) {
    // For ADD operations, we compute: arg1 + arg2 = res
    // All operands are DWordBL (8 bytes), need to cast to DWordWL (2 words)

    // Condition: ADD + LOAD + STORE
    // We need a virtual column for this, or we handle it differently.
    // For now, we'll create separate constraints for each.

    // ADD constraint: when ADD=1, arg1 + arg2 = res
    let lhs = AddOperand::from_dword_bl(cols::ARG1_0);
    let rhs = AddOperand::from_dword_bl(cols::ARG2_0);
    let sum = AddOperand::from_dword_bl(cols::RES_0);

    let (add_c0, add_c1) = AddConstraint::new_pair(cols::ADD, lhs, rhs, sum, constraint_idx_start);

    (vec![add_c0, add_c1], constraint_idx_start + 2)
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
        let res_0 = step.get_main_evaluation_element(0, cols::RES_0).clone();
        let is_equal = step.get_main_evaluation_element(0, cols::IS_EQUAL).clone();
        let branch_cond = step
            .get_main_evaluation_element(0, cols::BRANCH_COND)
            .clone();

        let two = FieldElement::<F>::from(2u64);

        // XOR computation: a XOR b = a + b - 2*a*b
        // res[0] XOR mp_selector
        let res_xor_mp = &res_0 + &mp_selector - &two * &res_0 * &mp_selector;
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

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<GoldilocksField, GoldilocksExtension>,
        transition_evaluations: &mut [FieldElement<GoldilocksExtension>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _,
                rap_challenges: _,
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _,
                rap_challenges: _,
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value;
            }
        }
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

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<GoldilocksField, GoldilocksExtension>,
        transition_evaluations: &mut [FieldElement<GoldilocksExtension>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _,
                rap_challenges: _,
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _,
                rap_challenges: _,
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value;
            }
        }
    }
}

// =========================================================================
// Extension Constraints
// =========================================================================

/// Constraint: arg1[0:4] = rv1[0:2] (lower 32 bits match)
///
/// arg1 is DWordBL (8 bytes), rv1 is DWordWHH (word + 2 halves)
/// arg1[:4] as word = rv1[0] (word)
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
        // arg1[0:4] as DWordWL[0] = sum of bytes
        let arg1_0 = step.get_main_evaluation_element(0, cols::ARG1_0).clone();
        let arg1_1 = step.get_main_evaluation_element(0, cols::ARG1_1).clone();
        let arg1_2 = step.get_main_evaluation_element(0, cols::ARG1_2).clone();
        let arg1_3 = step.get_main_evaluation_element(0, cols::ARG1_3).clone();

        let shift_8: FieldElement<F> = FieldElement::from(1u64 << 8);
        let shift_16: FieldElement<F> = FieldElement::from(1u64 << 16);
        let shift_24: FieldElement<F> = FieldElement::from(1u64 << 24);

        let arg1_lo = arg1_0 + arg1_1 * shift_8 + arg1_2 * shift_16 + arg1_3 * shift_24;

        // rv1[0] is already a word
        let rv1_0 = step.get_main_evaluation_element(0, cols::RV1_0).clone();

        // Constraint: arg1_lo - rv1[0] = 0
        arg1_lo - rv1_0
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for Arg1LowerConstraint {
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<GoldilocksField, GoldilocksExtension>,
        transition_evaluations: &mut [FieldElement<GoldilocksExtension>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _,
                rap_challenges: _,
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _,
                rap_challenges: _,
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value;
            }
        }
    }
}

/// Constraint: arg1[4:8] = rv1[2] * (1 - word_instr) + (2^32 - 1) * rv1_sign_bit * signed
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
        // arg1[4:8] as DWordWL[1]
        let arg1_4 = step.get_main_evaluation_element(0, cols::ARG1_4).clone();
        let arg1_5 = step.get_main_evaluation_element(0, cols::ARG1_5).clone();
        let arg1_6 = step.get_main_evaluation_element(0, cols::ARG1_6).clone();
        let arg1_7 = step.get_main_evaluation_element(0, cols::ARG1_7).clone();

        let shift_8: FieldElement<F> = FieldElement::from(1u64 << 8);
        let shift_16: FieldElement<F> = FieldElement::from(1u64 << 16);
        let shift_24: FieldElement<F> = FieldElement::from(1u64 << 24);

        let arg1_hi = arg1_4 + arg1_5 * shift_8 + arg1_6 * shift_16.clone() + arg1_7 * shift_24;

        // rv1 upper part: rv1[1] (half) + rv1[2] (half) * 2^16
        let rv1_1 = step.get_main_evaluation_element(0, cols::RV1_1).clone();
        let rv1_2 = step.get_main_evaluation_element(0, cols::RV1_2).clone();
        let rv1_upper = rv1_1 + rv1_2 * shift_16;

        let word_instr = step
            .get_main_evaluation_element(0, cols::WORD_INSTR)
            .clone();
        let signed = step.get_main_evaluation_element(0, cols::SIGNED).clone();
        let rv1_sign_bit = step
            .get_main_evaluation_element(0, cols::RV1_SIGN_BIT)
            .clone();

        let one = FieldElement::<F>::one();
        let mask_32: FieldElement<F> = FieldElement::from((1u64 << 32) - 1); // 2^32 - 1

        // Expected: rv1_upper * (1 - word_instr) + mask_32 * rv1_sign_bit * signed
        let expected = rv1_upper * (one - &word_instr) + mask_32 * rv1_sign_bit * signed;

        // Constraint: arg1_hi - expected = 0
        arg1_hi - expected
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for Arg1UpperConstraint {
    fn degree(&self) -> usize {
        // rv1_sign_bit * signed * word_instr has degree 3
        3
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<GoldilocksField, GoldilocksExtension>,
        transition_evaluations: &mut [FieldElement<GoldilocksExtension>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _,
                rap_challenges: _,
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _,
                rap_challenges: _,
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value;
            }
        }
    }
}

// =========================================================================
// SLT/BLT Zero Upper Bytes Constraint
// =========================================================================

/// Constraint: when SLT + BLT = 1, res[i] = 0 for i in 1..8
///
/// The LT result is a single bit stored in res[0], upper bytes must be zero.
pub struct SltResZeroConstraint {
    /// Which byte index (1-7) this constraint applies to
    byte_idx: usize,
    constraint_idx: usize,
}

impl SltResZeroConstraint {
    pub fn new(byte_idx: usize, constraint_idx: usize) -> Self {
        assert!(byte_idx >= 1 && byte_idx <= 7);
        Self {
            byte_idx,
            constraint_idx,
        }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let slt = step.get_main_evaluation_element(0, cols::SLT).clone();
        let blt = step.get_main_evaluation_element(0, cols::BLT).clone();
        let res_i = step
            .get_main_evaluation_element(0, cols::RES[self.byte_idx])
            .clone();

        // (SLT + BLT) * res[i] = 0
        (slt + blt) * res_i
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for SltResZeroConstraint {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<GoldilocksField, GoldilocksExtension>,
        transition_evaluations: &mut [FieldElement<GoldilocksExtension>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _,
                rap_challenges: _,
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _,
                rap_challenges: _,
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value;
            }
        }
    }
}

/// Creates all SLT/BLT zero constraints for res[1..8].
pub fn create_slt_res_zero_constraints(
    constraint_idx_start: usize,
) -> (Vec<SltResZeroConstraint>, usize) {
    let constraints: Vec<_> = (1..8)
        .enumerate()
        .map(|(i, byte_idx)| SltResZeroConstraint::new(byte_idx, constraint_idx_start + i))
        .collect();

    (constraints, constraint_idx_start + 7)
}

// =========================================================================
// Sign Bit Constraints
// =========================================================================

/// Constraint: sign bits are zero when word_instr = 0
///
/// (rv1_sign_bit + arg2_sign_bit + res_sign_bit) * (1 - word_instr) = 0
pub struct SignBitZeroConstraint {
    constraint_idx: usize,
}

impl SignBitZeroConstraint {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let rv1_sign_bit = step
            .get_main_evaluation_element(0, cols::RV1_SIGN_BIT)
            .clone();
        let arg2_sign_bit = step
            .get_main_evaluation_element(0, cols::ARG2_SIGN_BIT)
            .clone();
        let res_sign_bit = step
            .get_main_evaluation_element(0, cols::RES_SIGN_BIT)
            .clone();
        let word_instr = step
            .get_main_evaluation_element(0, cols::WORD_INSTR)
            .clone();

        let one = FieldElement::<F>::one();

        // (sum of sign bits) * (1 - word_instr) = 0
        (rv1_sign_bit + arg2_sign_bit + res_sign_bit) * (one - word_instr)
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for SignBitZeroConstraint {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<GoldilocksField, GoldilocksExtension>,
        transition_evaluations: &mut [FieldElement<GoldilocksExtension>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _,
                rap_challenges: _,
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _,
                rap_challenges: _,
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value;
            }
        }
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
        let inv_2_32: FieldElement<F> = FieldElement::from(super::templates::SHIFT_32)
            .inv()
            .unwrap();
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
        let inv_2_32: FieldElement<F> = FieldElement::from(super::templates::SHIFT_32)
            .inv()
            .unwrap();
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
        let not_branch = one - branch_cond;

        let carry = match self.carry_idx {
            0 => self.compute_carry_0(step),
            1 => self.compute_carry_1(step),
            _ => panic!("Invalid carry index"),
        };

        // (1 - branch_cond) * carry * (1 - carry)
        not_branch * &carry * (FieldElement::<F>::one() - carry)
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

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<GoldilocksField, GoldilocksExtension>,
        transition_evaluations: &mut [FieldElement<GoldilocksExtension>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _,
                rap_challenges: _,
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _,
                rap_challenges: _,
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value;
            }
        }
    }
}

// =========================================================================
// Constraint Summary
// =========================================================================

/// Total number of CPU constraints.
///
/// - IS_BIT: 30 (all bit flags)
/// - ADD carry: 2
/// - Branch cond: 1
/// - EBREAK: 1
/// - Arg1 lower: 1
/// - Arg1 upper: 1
/// - SLT res zero: 7 (bytes 1-7)
/// - Sign bit zero: 1
/// - Next PC (non-branching): 2
///
/// Total: 46 constraints
pub const NUM_CPU_CONSTRAINTS: usize = 30 + 2 + 1 + 1 + 1 + 1 + 7 + 1 + 2;

/// Creates all CPU constraints.
///
/// Returns a tuple of (is_bit_constraints, add_constraints, other_constraints, next_idx)
pub fn create_all_cpu_constraints() -> (
    Vec<IsBitConstraint>,
    Vec<AddConstraint>,
    Vec<Box<dyn TransitionConstraint<GoldilocksField, GoldilocksExtension>>>,
    usize,
) {
    let mut next_idx = 0;

    // IS_BIT constraints
    let (is_bit, next) = create_is_bit_constraints(next_idx);
    next_idx = next;

    // ADD constraints
    let (add, next) = create_add_constraints(next_idx);
    next_idx = next;

    // Other constraints
    let mut other: Vec<Box<dyn TransitionConstraint<GoldilocksField, GoldilocksExtension>>> =
        Vec::new();

    // Branch condition
    other.push(Box::new(BranchCondConstraint::new(next_idx)));
    next_idx += 1;

    // EBREAK
    other.push(Box::new(EbreakConstraint::new(next_idx)));
    next_idx += 1;

    // Arg1 constraints
    other.push(Box::new(Arg1LowerConstraint::new(next_idx)));
    next_idx += 1;
    other.push(Box::new(Arg1UpperConstraint::new(next_idx)));
    next_idx += 1;

    // SLT res zero constraints
    let (slt_zero, next) = create_slt_res_zero_constraints(next_idx);
    next_idx = next;
    for c in slt_zero {
        other.push(Box::new(c));
    }

    // Sign bit zero constraint
    other.push(Box::new(SignBitZeroConstraint::new(next_idx)));
    next_idx += 1;

    // Next PC (non-branching) constraints
    let (next_pc_0, next_pc_1) = NextPcAddConstraint::new_pair(next_idx);
    other.push(Box::new(next_pc_0));
    other.push(Box::new(next_pc_1));
    next_idx += 2;

    (is_bit, add, other, next_idx)
}
