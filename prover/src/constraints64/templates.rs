//! Constraint templates for the 64-bit VM prover.
//!
//! This module provides reusable constraint templates using the Goldilocks field.
//!
//! ## Templates
//!
//! - **IS_BIT**: Enforces that a value is binary (0 or 1)
//!   - Constraint: `cond * X * (1-X) = 0`
//!
//! - **ADD**: 64-bit addition with embedded virtual carry columns
//!   - lhs, rhs, sum: DWordWL (2 × 32-bit words)
//!   - Embeds carry constraints inline

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::{
    constraints::transition::TransitionConstraint, table::TableView,
    traits::TransitionEvaluationContext,
};

use crate::tables64::types::{GoldilocksExtension, GoldilocksField};

// =========================================================================
// Constants
// =========================================================================

/// 2^32 for word combining and carry extraction
pub const SHIFT_32: u64 = 1u64 << 32;

// =========================================================================
// IS_BIT Template
// =========================================================================

/// Enforces that a value is binary (0 or 1) when a condition is active.
///
/// Constraint: `cond * X * (1-X) = 0`
///
/// This constraint evaluates to 0 in three cases:
/// - cond = 0 (constraint inactive)
/// - X = 0 (valid binary value)
/// - X = 1 (valid binary value)
///
/// Degree: 3 (cubic)
pub struct IsBitConstraint {
    /// Column index for the condition (cond)
    cond_col: usize,
    /// Column index for the value to check (X)
    value_col: usize,
    /// Unique constraint identifier
    constraint_idx: usize,
}

impl IsBitConstraint {
    /// Creates a new IS_BIT constraint.
    ///
    /// # Arguments
    /// * `cond_col` - Column index containing the condition flag
    /// * `value_col` - Column index containing the value to check
    /// * `constraint_idx` - Unique constraint identifier
    pub fn new(cond_col: usize, value_col: usize, constraint_idx: usize) -> Self {
        Self {
            cond_col,
            value_col,
            constraint_idx,
        }
    }

    /// Creates an unconditional IS_BIT constraint (always active).
    ///
    /// Uses the value column itself as condition (since 0 or 1 as condition
    /// still satisfies the constraint when the value is correct).
    ///
    /// For truly unconditional constraints, use a column that's always 1.
    pub fn unconditional(value_col: usize, always_one_col: usize, constraint_idx: usize) -> Self {
        Self {
            cond_col: always_one_col,
            value_col,
            constraint_idx,
        }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let cond = step.get_main_evaluation_element(0, self.cond_col).clone();
        let x = step.get_main_evaluation_element(0, self.value_col).clone();
        let one = FieldElement::<F>::one();

        // cond * X * (1 - X)
        &cond * &x * (one - x)
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for IsBitConstraint {
    fn degree(&self) -> usize {
        3 // cubic: cond * X * (1-X)
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
// ADD Template (Embedded Carry Approach)
// =========================================================================

/// 64-bit addition constraint with embedded carry.
///
/// Enforces: `lhs + rhs = sum (mod 2^64)`
///
/// Uses DWordWL representation (2 × 32-bit words):
/// - lhs = [lhs_lo, lhs_hi]
/// - rhs = [rhs_lo, rhs_hi]
/// - sum = [sum_lo, sum_hi]
///
/// Embeds virtual carry columns inline:
/// - carry_0 = (lhs_lo + rhs_lo - sum_lo) / 2^32
/// - carry_1 = (lhs_hi + rhs_hi + carry_0 - sum_hi) / 2^32
///
/// Constraints:
/// - carry_0 is a bit: cond * carry_0 * (1 - carry_0) = 0
/// - carry_1 is a bit: cond * carry_1 * (1 - carry_1) = 0
///
/// Assumptions (must be verified via bus lookups):
/// - lhs_lo, lhs_hi, rhs_lo, rhs_hi, sum_lo, sum_hi are all valid 32-bit words
pub struct AddConstraint {
    /// Column index for condition flag
    cond_col: usize,
    /// Starting column index for lhs (2 consecutive columns)
    lhs_start_col: usize,
    /// Starting column index for rhs (2 consecutive columns)
    rhs_start_col: usize,
    /// Starting column index for sum (2 consecutive columns)
    sum_start_col: usize,
    /// Which carry constraint this is (0 or 1)
    carry_idx: usize,
    /// Unique constraint identifier
    constraint_idx: usize,
}

impl AddConstraint {
    /// Creates ADD constraints for both carries.
    ///
    /// Returns two constraints: one for carry_0 and one for carry_1.
    pub fn new_pair(
        cond_col: usize,
        lhs_start_col: usize,
        rhs_start_col: usize,
        sum_start_col: usize,
        constraint_idx_start: usize,
    ) -> (Self, Self) {
        let carry_0 = Self {
            cond_col,
            lhs_start_col,
            rhs_start_col,
            sum_start_col,
            carry_idx: 0,
            constraint_idx: constraint_idx_start,
        };

        let carry_1 = Self {
            cond_col,
            lhs_start_col,
            rhs_start_col,
            sum_start_col,
            carry_idx: 1,
            constraint_idx: constraint_idx_start + 1,
        };

        (carry_0, carry_1)
    }

    /// Compute carry_0 inline from trace values.
    fn compute_carry_0<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let lhs_lo = step.get_main_evaluation_element(0, self.lhs_start_col);
        let rhs_lo = step.get_main_evaluation_element(0, self.rhs_start_col);
        let sum_lo = step.get_main_evaluation_element(0, self.sum_start_col);

        // carry_0 = (lhs_lo + rhs_lo - sum_lo) * 2^(-32)
        let inv_2_32: FieldElement<F> = FieldElement::from(SHIFT_32).inv().unwrap();
        (lhs_lo + rhs_lo - sum_lo) * inv_2_32
    }

    /// Compute carry_1 inline from trace values.
    fn compute_carry_1<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let lhs_hi = step.get_main_evaluation_element(0, self.lhs_start_col + 1);
        let rhs_hi = step.get_main_evaluation_element(0, self.rhs_start_col + 1);
        let sum_hi = step.get_main_evaluation_element(0, self.sum_start_col + 1);
        let carry_0 = self.compute_carry_0(step);

        // carry_1 = (lhs_hi + rhs_hi + carry_0 - sum_hi) * 2^(-32)
        let inv_2_32: FieldElement<F> = FieldElement::from(SHIFT_32).inv().unwrap();
        (lhs_hi + rhs_hi + carry_0 - sum_hi) * inv_2_32
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let cond = step.get_main_evaluation_element(0, self.cond_col);
        let one = FieldElement::<F>::one();

        let carry = match self.carry_idx {
            0 => self.compute_carry_0(step),
            1 => self.compute_carry_1(step),
            _ => panic!("Invalid carry index"),
        };

        // cond * carry * (1 - carry)
        cond * &carry * (one - carry)
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for AddConstraint {
    fn degree(&self) -> usize {
        // The constraint is cond * carry * (1 - carry)
        // where carry involves division by 2^32 (degree 1 in trace elements)
        // So total degree: 1 (cond) * 1 (carry) * 1 (1-carry) = 3
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
// Helper Functions
// =========================================================================

/// Creates multiple IS_BIT constraints for consecutive columns.
///
/// # Arguments
/// * `cond_col` - Column index for the condition flag
/// * `value_cols` - Slice of column indices to constrain
/// * `constraint_idx_start` - Starting index for constraint numbering
///
/// # Returns
/// Vector of IS_BIT constraints and the next available constraint index.
pub fn new_is_bit_constraints(
    cond_col: usize,
    value_cols: &[usize],
    constraint_idx_start: usize,
) -> (Vec<IsBitConstraint>, usize) {
    let constraints = value_cols
        .iter()
        .enumerate()
        .map(|(i, &col)| IsBitConstraint::new(cond_col, col, constraint_idx_start + i))
        .collect();

    (constraints, constraint_idx_start + value_cols.len())
}
