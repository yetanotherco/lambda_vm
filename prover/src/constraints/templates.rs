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

impl AddOperand {
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

// =========================================================================
// Single-body emit functions (ConstraintBuilder front-end)
// =========================================================================
//
// Non-destructive twins of the boxed constraint structs above, written once
// against the generic `ConstraintBuilder` so one body serves the compiled
// prover folder, the verifier folder and IR capture. The structs stay for
// now (they are the differential oracle for the emit functions); the table
// conversion deletes them.
//
// Each `emit_*` takes the constraint index it emits at; the matching
// `*_meta` returns the idx-ordered metadata (declared degree; default
// zerofier shape — none of these templates override period/offset/
// exemptions).

use stark::constraints::builder::{ConstraintBuilder, ConstraintMeta};

/// IS_BIT: `x·(1−x)`, optionally gated by a condition column:
/// `cond·x·(1−x)`. Twin of `IsBitConstraint`.
pub fn emit_is_bit<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &mut B,
    idx: usize,
    value_col: usize,
    cond_col: Option<usize>,
) {
    let x = b.main(0, value_col);
    let one = b.one();
    let root = match cond_col {
        Some(c) => {
            let cond = b.main(0, c);
            cond * x.clone() * (one - x)
        }
        None => x.clone() * (one - x),
    };
    b.emit_base(idx, root);
}

/// Metadata for [`emit_is_bit`]: degree 3 gated, 2 ungated.
pub fn is_bit_meta(idx: usize, conditional: bool) -> ConstraintMeta {
    ConstraintMeta::base(idx, if conditional { 3 } else { 2 })
}

/// One [`AddLinearTerm`]: `column · coefficient` or a constant
/// (matches `AddLinearTerm::eval`'s operand order).
fn add_term_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    t: &AddLinearTerm,
) -> B::Expr {
    match t {
        AddLinearTerm::Column {
            coefficient,
            column,
        } => b.main(0, *column) * b.const_signed(*coefficient),
        AddLinearTerm::Constant(v) => b.const_signed(*v),
    }
}

/// Sum of terms, from zero (matches `eval_terms`).
fn add_terms_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    terms: &[AddLinearTerm],
) -> B::Expr {
    let mut acc = b.zero();
    for t in terms {
        acc = acc + add_term_expr(b, t);
    }
    acc
}

/// An operand's low word (matches `AddOperand::eval_lo`).
fn add_operand_lo<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    op: &AddOperand,
) -> B::Expr {
    match op {
        AddOperand::DWordWL { start_column } => b.main(0, *start_column),
        AddOperand::Linear { lo, .. } => add_terms_expr(b, lo),
    }
}

/// An operand's high word (matches `AddOperand::eval_hi`).
fn add_operand_hi<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    op: &AddOperand,
) -> B::Expr {
    match op {
        AddOperand::DWordWL { start_column } => b.main(0, *start_column + 1),
        AddOperand::Linear { hi, .. } => add_terms_expr(b, hi),
    }
}

/// The ADD carry pair, emitted from ONE body at `idx` and `idx + 1`
/// (twin of `AddConstraint::new_pair`):
///
/// ```text
/// carry_0 = (lhs.lo + rhs.lo − sum.lo)·2⁻³²
/// carry_1 = (lhs.hi + rhs.hi + carry_0 − sum.hi)·2⁻³²
/// emit:     [cond·] carry_i·(1 − carry_i)      at idx, idx+1
/// ```
///
/// `cond` is the sum of the `cond_cols` flags (empty = unconditional).
pub fn emit_add_pair<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &mut B,
    idx: usize,
    cond_cols: &[usize],
    lhs: &AddOperand,
    rhs: &AddOperand,
    sum: &AddOperand,
) {
    let inv_2_32 = b.const_base(INV_SHIFT_32);
    let carry_0 = (add_operand_lo(b, lhs) + add_operand_lo(b, rhs) - add_operand_lo(b, sum))
        * inv_2_32.clone();
    let carry_1 = (add_operand_hi(b, lhs) + add_operand_hi(b, rhs) + carry_0.clone()
        - add_operand_hi(b, sum))
        * inv_2_32;

    let cond = |b: &B| -> Option<B::Expr> {
        if cond_cols.is_empty() {
            None
        } else {
            let mut acc = b.zero();
            for &c in cond_cols {
                acc = acc + b.main(0, c);
            }
            Some(acc)
        }
    };
    let bit = |b: &B, cond: Option<B::Expr>, carry: B::Expr| -> B::Expr {
        let one = b.one();
        match cond {
            Some(c) => c * carry.clone() * (one - carry),
            None => carry.clone() * (one - carry),
        }
    };

    let c0 = cond(b);
    let root_0 = bit(b, c0, carry_0);
    b.emit_base(idx, root_0);
    let c1 = cond(b);
    let root_1 = bit(b, c1, carry_1);
    b.emit_base(idx + 1, root_1);
}

/// Metadata for [`emit_add_pair`]: two constraints at `idx`, `idx + 1`;
/// degree 3 gated, 2 ungated.
pub fn add_pair_meta(idx: usize, conditional: bool) -> [ConstraintMeta; 2] {
    let degree = if conditional { 3 } else { 2 };
    [
        ConstraintMeta::base(idx, degree),
        ConstraintMeta::base(idx + 1, degree),
    ]
}
