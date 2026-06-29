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
use stark::constraint_ir::{Capture, Expr, IrBuilder};
use stark::{constraints::transition::TransitionConstraint, table::TableView};

use crate::tables::types::{GoldilocksExtension, GoldilocksField};

// =========================================================================
// Constants
// =========================================================================

/// 2^32 for word combining and carry extraction
pub const SHIFT_32: u64 = 1u64 << 32;

/// Precomputed: (2^32)^(-1) mod p where p = 2^64 - 2^32 + 1.
/// Avoids ~72 multiplications per inv() call in constraint hot loops.
/// Verify: INV_SHIFT_32 * SHIFT_32 ≡ 1 (mod p)
pub const INV_SHIFT_32: u64 = 18446744065119617026;

/// 2^(-32) in the field, used for carry extraction.
#[inline]
fn inv_2_32<F: IsField>() -> FieldElement<F> {
    FieldElement::from(INV_SHIFT_32)
}

// =========================================================================
// IS_BIT Template
// =========================================================================

/// Enforces that a value is binary (0 or 1).
///
/// Two modes:
/// - Conditional: `cond * X * (1-X) = 0` (degree 3)
/// - Unconditional: `X * (1-X) = 0` (degree 2)
pub struct IsBitConstraint {
    /// Column index for the condition (None = unconditional)
    cond_col: Option<usize>,
    /// Column index for the value to check (X)
    value_col: usize,
    /// Unique constraint identifier
    constraint_idx: usize,
}

impl IsBitConstraint {
    /// Creates a conditional IS_BIT constraint.
    ///
    /// Constraint: `cond * X * (1-X) = 0`
    pub fn new(cond_col: usize, value_col: usize, constraint_idx: usize) -> Self {
        Self {
            cond_col: Some(cond_col),
            value_col,
            constraint_idx,
        }
    }

    /// Creates an unconditional IS_BIT constraint.
    ///
    /// Constraint: `X * (1-X) = 0`
    pub fn unconditional(value_col: usize, constraint_idx: usize) -> Self {
        Self {
            cond_col: None,
            value_col,
            constraint_idx,
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for IsBitConstraint {
    fn degree(&self) -> usize {
        match self.cond_col {
            Some(_) => 3, // cubic: cond * X * (1-X)
            None => 2,    // quadratic: X * (1-X)
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
        let x = step.get_main_evaluation_element(0, self.value_col).clone();
        let one = FieldElement::<F>::one();

        match self.cond_col {
            Some(cond_col) => {
                let cond = step.get_main_evaluation_element(0, cond_col).clone();
                &cond * &x * (one - x)
            }
            None => &x * (one - &x),
        }
    }
}

impl Capture for IsBitConstraint {
    fn capture(&self, b: &mut IrBuilder) {
        // Mirrors `evaluate`: x = main(value_col), one - x, then the product.
        let x = b.main(0, self.value_col);
        let one = b.one();
        let one_minus_x = b.sub(one, x);

        let root = match self.cond_col {
            Some(cond_col) => {
                // cond * x * (1 - x), left-associated like `&cond * &x * (one - x)`.
                let cond = b.main(0, cond_col);
                let cond_x = b.mul(cond, x);
                b.mul(cond_x, one_minus_x)
            }
            // x * (1 - x)
            None => b.mul(x, one_minus_x),
        };

        b.emit(self.constraint_idx, root);
    }
}

// =========================================================================
// ADD Template (Embedded Carry Approach)
// =========================================================================

// -------------------------------------------------------------------------
// AddOperand - Flexible operand representation for ADD constraints
// -------------------------------------------------------------------------

/// A term in a linear combination: coeff * column OR constant.
///
/// Uses i64 for coefficients to support negative values (e.g., `4 - 2*c`).
/// Converted to FieldElement in eval().
#[derive(Debug, Clone)]
pub enum AddLinearTerm {
    /// coefficient * column_value
    Column {
        /// Coefficient (can be negative, e.g., -2 for subtraction)
        coefficient: i64,
        /// Column index to read from
        column: usize,
    },
    /// A constant value
    Constant(i64),
}

/// An ADD operand representing a 64-bit value as [lo, hi] words.
///
/// Supports various representations:
/// - DWordWL: 2 consecutive columns (most common case)
/// - Linear: Arbitrary linear combinations for lo and hi limbs
///
/// The Linear variant handles:
/// - Constants: `AddOperand::constant(42)` → lo=42, hi=0
/// - Word → DWordWL: `AddOperand::from_word(col)` → lo=col, hi=0
/// - DWordHL → DWordWL: `AddOperand::from_dword_hl(col)` → repack 4 halves
/// - DWordBL → DWordWL: `AddOperand::from_dword_bl(col)` → repack 8 bytes
/// - Expressions: `AddOperand::linear(...)` → arbitrary linear combinations
#[derive(Debug, Clone)]
pub enum AddOperand {
    /// Two consecutive columns (DWordWL): evaluates to [col, col+1]
    DWordWL { start_column: usize },

    /// Linear combination for lo and hi limbs.
    /// Handles: constants, single columns, expressions, and virtual columns.
    Linear {
        /// Terms for the low 32-bit word
        lo: Vec<AddLinearTerm>,
        /// Terms for the high 32-bit word (empty = zero)
        hi: Vec<AddLinearTerm>,
    },
}

impl AddLinearTerm {
    /// Evaluate this term using values from the trace.
    fn eval<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        match self {
            AddLinearTerm::Column {
                coefficient,
                column,
            } => {
                let col_val = step.get_main_evaluation_element(0, *column);
                col_val * FieldElement::<F>::from(*coefficient)
            }
            AddLinearTerm::Constant(value) => FieldElement::<F>::from(*value),
        }
    }

    /// Capture this term into builder nodes, mirroring [`Self::eval`].
    fn capture(&self, b: &mut IrBuilder) -> Expr {
        match self {
            AddLinearTerm::Column {
                coefficient,
                column,
            } => {
                // `col_val * FieldElement::from(coeff)`: column on the left.
                let col = b.main(0, *column);
                let coeff = b.const_signed(*coefficient);
                b.mul(col, coeff)
            }
            AddLinearTerm::Constant(value) => b.const_signed(*value),
        }
    }
}

/// Evaluate a slice of terms as a sum.
fn eval_terms<F, E>(terms: &[AddLinearTerm], step: &TableView<F, E>) -> FieldElement<F>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    if terms.is_empty() {
        FieldElement::zero()
    } else {
        terms
            .iter()
            .map(|t| t.eval(step))
            .fold(FieldElement::zero(), |acc, x| acc + x)
    }
}

/// Capture a slice of terms as a sum, mirroring [`eval_terms`].
///
/// Empty -> `0`; otherwise `0 + t0 + t1 + ...` (same fold seed and order as
/// `eval_terms`, so the captured node tree matches bit-for-bit).
fn capture_terms(terms: &[AddLinearTerm], b: &mut IrBuilder) -> Expr {
    let zero = b.const_base(0);
    if terms.is_empty() {
        zero
    } else {
        let mut acc = zero;
        for t in terms {
            let term = t.capture(b);
            acc = b.add(acc, term);
        }
        acc
    }
}

impl AddOperand {
    /// Get the low word value from the trace.
    pub fn eval_lo<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        match self {
            AddOperand::DWordWL { start_column } => {
                step.get_main_evaluation_element(0, *start_column).clone()
            }
            AddOperand::Linear { lo, .. } => eval_terms(lo, step),
        }
    }

    /// Get the high word value from the trace.
    pub fn eval_hi<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        match self {
            AddOperand::DWordWL { start_column } => step
                .get_main_evaluation_element(0, *start_column + 1)
                .clone(),
            AddOperand::Linear { hi, .. } => eval_terms(hi, step),
        }
    }

    /// Capture the low word, mirroring [`Self::eval_lo`].
    pub fn capture_lo(&self, b: &mut IrBuilder) -> Expr {
        match self {
            AddOperand::DWordWL { start_column } => b.main(0, *start_column),
            AddOperand::Linear { lo, .. } => capture_terms(lo, b),
        }
    }

    /// Capture the high word, mirroring [`Self::eval_hi`].
    pub fn capture_hi(&self, b: &mut IrBuilder) -> Expr {
        match self {
            AddOperand::DWordWL { start_column } => b.main(0, *start_column + 1),
            AddOperand::Linear { hi, .. } => capture_terms(hi, b),
        }
    }

    // -------------------------------------------------------------------------
    // Convenience constructors for common cast types
    // -------------------------------------------------------------------------

    /// DWordWL: 2 consecutive columns [col, col+1].
    pub fn dword(start_column: usize) -> Self {
        AddOperand::DWordWL { start_column }
    }

    /// Constant: single value, zero-extended to 64 bits.
    /// hi = 0 (since constants fit in 32 bits for VM use cases).
    pub fn constant(value: i64) -> Self {
        AddOperand::Linear {
            lo: vec![AddLinearTerm::Constant(value)],
            hi: vec![],
        }
    }

    /// Word → DWordWL: single column, zero-extended.
    /// hi = 0.
    pub fn from_word(col: usize) -> Self {
        AddOperand::Linear {
            lo: vec![AddLinearTerm::Column {
                coefficient: 1,
                column: col,
            }],
            hi: vec![],
        }
    }

    /// DWordHL → DWordWL: repack 4 half-words into 2 words.
    /// lo = h[0] + 2^16 * h[1]
    /// hi = h[2] + 2^16 * h[3]
    pub fn from_dword_hl(start_column: usize) -> Self {
        AddOperand::Linear {
            lo: vec![
                AddLinearTerm::Column {
                    coefficient: 1,
                    column: start_column,
                },
                AddLinearTerm::Column {
                    coefficient: 1 << 16,
                    column: start_column + 1,
                },
            ],
            hi: vec![
                AddLinearTerm::Column {
                    coefficient: 1,
                    column: start_column + 2,
                },
                AddLinearTerm::Column {
                    coefficient: 1 << 16,
                    column: start_column + 3,
                },
            ],
        }
    }

    /// DWordBL → DWordWL: repack 8 bytes into 2 words.
    /// lo = b[0] + 2^8*b[1] + 2^16*b[2] + 2^24*b[3]
    /// hi = b[4] + 2^8*b[5] + 2^16*b[6] + 2^24*b[7]
    pub fn from_dword_bl(start_column: usize) -> Self {
        AddOperand::Linear {
            lo: vec![
                AddLinearTerm::Column {
                    coefficient: 1,
                    column: start_column,
                },
                AddLinearTerm::Column {
                    coefficient: 1 << 8,
                    column: start_column + 1,
                },
                AddLinearTerm::Column {
                    coefficient: 1 << 16,
                    column: start_column + 2,
                },
                AddLinearTerm::Column {
                    coefficient: 1 << 24,
                    column: start_column + 3,
                },
            ],
            hi: vec![
                AddLinearTerm::Column {
                    coefficient: 1,
                    column: start_column + 4,
                },
                AddLinearTerm::Column {
                    coefficient: 1 << 8,
                    column: start_column + 5,
                },
                AddLinearTerm::Column {
                    coefficient: 1 << 16,
                    column: start_column + 6,
                },
                AddLinearTerm::Column {
                    coefficient: 1 << 24,
                    column: start_column + 7,
                },
            ],
        }
    }

    /// Creates a Linear operand from explicit lo/hi term lists.
    /// Use this for complex expressions like `4 - 2*c` or virtual columns.
    pub fn linear(lo: Vec<AddLinearTerm>, hi: Vec<AddLinearTerm>) -> Self {
        AddOperand::Linear { lo, hi }
    }
}

// -------------------------------------------------------------------------
// AddConstraint
// -------------------------------------------------------------------------

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
    /// Column indices for condition flags (constraint active when sum > 0)
    cond_cols: Vec<usize>,
    /// Left-hand side operand (flexible representation)
    lhs: AddOperand,
    /// Right-hand side operand (flexible representation)
    rhs: AddOperand,
    /// Sum/output operand (flexible representation)
    sum: AddOperand,
    /// Which carry constraint this is (0 or 1)
    carry_idx: usize,
    /// Unique constraint identifier
    constraint_idx: usize,
}

impl AddConstraint {
    /// Creates ADD constraints for both carries.
    ///
    /// Returns two constraints: one for carry_0 and one for carry_1.
    ///
    /// # Arguments
    /// * `cond_cols` - Column indices for condition flags (constraint active when sum > 0)
    /// * `lhs` - Left-hand side operand (flexible representation)
    /// * `rhs` - Right-hand side operand (flexible representation)
    /// * `sum` - Sum/output operand (flexible representation)
    /// * `constraint_idx_start` - Starting constraint index (uses 2 consecutive indices)
    pub fn new_pair(
        cond_cols: Vec<usize>,
        lhs: AddOperand,
        rhs: AddOperand,
        sum: AddOperand,
        constraint_idx_start: usize,
    ) -> (Self, Self) {
        let carry_0 = Self {
            cond_cols: cond_cols.clone(),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            sum: sum.clone(),
            carry_idx: 0,
            constraint_idx: constraint_idx_start,
        };

        let carry_1 = Self {
            cond_cols,
            lhs,
            rhs,
            sum,
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
        let lhs_lo = self.lhs.eval_lo(step);
        let rhs_lo = self.rhs.eval_lo(step);
        let sum_lo = self.sum.eval_lo(step);

        // carry_0 = (lhs_lo + rhs_lo - sum_lo) * 2^(-32)
        (lhs_lo + rhs_lo - sum_lo) * inv_2_32::<F>()
    }

    /// Compute carry_1 inline from trace values.
    fn compute_carry_1<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let lhs_hi = self.lhs.eval_hi(step);
        let rhs_hi = self.rhs.eval_hi(step);
        let sum_hi = self.sum.eval_hi(step);
        let carry_0 = self.compute_carry_0(step);

        // carry_1 = (lhs_hi + rhs_hi + carry_0 - sum_hi) * 2^(-32)
        (lhs_hi + rhs_hi + carry_0 - sum_hi) * inv_2_32::<F>()
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let one = FieldElement::<F>::one();

        let carry = match self.carry_idx {
            0 => self.compute_carry_0(step),
            1 => self.compute_carry_1(step),
            _ => unreachable!("carry_idx validated <= 1 at construction"),
        };

        if self.cond_cols.is_empty() {
            // Unconditional: carry * (1 - carry)
            &carry * (one - &carry)
        } else {
            // Conditional: cond * carry * (1 - carry)
            let cond = self
                .cond_cols
                .iter()
                .map(|&col| step.get_main_evaluation_element(0, col).clone())
                .fold(FieldElement::<F>::zero(), |acc, x| acc + x);
            cond * &carry * (one - carry)
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for AddConstraint {
    fn degree(&self) -> usize {
        if self.cond_cols.is_empty() { 2 } else { 3 }
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

impl AddConstraint {
    /// Capture carry_0, mirroring [`Self::compute_carry_0`].
    fn capture_carry_0(&self, b: &mut IrBuilder) -> Expr {
        let lhs_lo = self.lhs.capture_lo(b);
        let rhs_lo = self.rhs.capture_lo(b);
        let sum_lo = self.sum.capture_lo(b);
        let inv = b.const_base(INV_SHIFT_32);

        // ((lhs_lo + rhs_lo) - sum_lo) * inv_2_32
        let s = b.add(lhs_lo, rhs_lo);
        let s = b.sub(s, sum_lo);
        b.mul(s, inv)
    }

    /// Capture carry_1, mirroring [`Self::compute_carry_1`].
    fn capture_carry_1(&self, b: &mut IrBuilder) -> Expr {
        let lhs_hi = self.lhs.capture_hi(b);
        let rhs_hi = self.rhs.capture_hi(b);
        let sum_hi = self.sum.capture_hi(b);
        let carry_0 = self.capture_carry_0(b);
        let inv = b.const_base(INV_SHIFT_32);

        // (((lhs_hi + rhs_hi) + carry_0) - sum_hi) * inv_2_32
        let s = b.add(lhs_hi, rhs_hi);
        let s = b.add(s, carry_0);
        let s = b.sub(s, sum_hi);
        b.mul(s, inv)
    }
}

impl Capture for AddConstraint {
    fn capture(&self, b: &mut IrBuilder) {
        let one = b.one();

        let carry = match self.carry_idx {
            0 => self.capture_carry_0(b),
            1 => self.capture_carry_1(b),
            _ => unreachable!("carry_idx validated <= 1 at construction"),
        };

        let root = if self.cond_cols.is_empty() {
            // Unconditional: carry * (1 - carry)
            let one_minus_carry = b.sub(one, carry);
            b.mul(carry, one_minus_carry)
        } else {
            // Conditional: cond * carry * (1 - carry), left-associated like
            // `cond * &carry * (one - carry)`.
            // cond = fold over cond_cols starting from zero: 0 + col0 + col1 + ...
            let mut cond = b.const_base(0);
            for &col in &self.cond_cols {
                let c = b.main(0, col);
                cond = b.add(cond, c);
            }
            let one_minus_carry = b.sub(one, carry);
            let cond_carry = b.mul(cond, carry);
            b.mul(cond_carry, one_minus_carry)
        };

        b.emit(self.constraint_idx, root);
    }
}

// =========================================================================
// Helper Functions
// =========================================================================

/// Creates multiple unconditional IS_BIT constraints for the given columns.
///
/// # Arguments
/// * `value_cols` - Slice of column indices to constrain
/// * `constraint_idx_start` - Starting index for constraint numbering
///
/// # Returns
/// Vector of IS_BIT constraints and the next available constraint index.
pub fn new_is_bit_constraints(
    value_cols: &[usize],
    constraint_idx_start: usize,
) -> (Vec<IsBitConstraint>, usize) {
    let constraints = value_cols
        .iter()
        .enumerate()
        .map(|(i, &col)| IsBitConstraint::unconditional(col, constraint_idx_start + i))
        .collect();

    (constraints, constraint_idx_start + value_cols.len())
}
