//! LT (Less-Than) comparison table.
//!
//! This table computes whether `lhs < rhs` for 64-bit values, supporting both
//! signed and unsigned comparisons.
//!
//! ## Inputs
//! - `lhs`: DWordHHW (64-bit as [Word, Half, Half])
//! - `rhs`: DWordHHW (64-bit as [Word, Half, Half])
//! - `signed`: Bit (whether to interpret as signed)
//!
//! ## Output
//! - `lt`: Bit (1 if lhs < rhs, 0 otherwise)
//!
//! ## Auxiliary
//! - `lhs_sub_rhs`: DWordHL (lhs - rhs as 4 halfwords)
//! - `lhs_msb`: Bit (MSB of lhs)
//! - `rhs_msb`: Bit (MSB of rhs)
//!
//! ## Virtual (embedded in constraints)
//! - `carry[0]`, `carry[1]`: Carries from verifying lhs = rhs + lhs_sub_rhs
//! - `unsigned_lt`: equals carry[1] (borrow from subtraction)
//!
//! ## Bus Interactions
//! - Sender: MSB16 (×2 for lhs_msb, rhs_msb)
//! - Sender: IS_HALFWORD (×6: ×4 for lhs_sub_rhs, ×1 for lhs[1], ×1 for rhs[1])
//! - Receiver: ALU (all less-than lookups — CPU SLT/BLT/BGE dispatch and the
//!   internal `memw`/`memw_aligned`/`dvrm` timestamp / |r|<|d| checks)

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::TransitionConstraint;
use stark::constraints::builder::{
    ConstraintBuilder, ConstraintContext, ProverConstraintBuilder, TableConstraints,
    VerifierConstraintBuilder,
};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, SHIFT_16, VmTable, alu_op};

// =========================================================================
// Column indices for LT table
// =========================================================================

/// Column definitions for the LT table.
pub mod cols {
    // Input columns: lhs (DWordHHW = Word + Half + Half = 3 columns)
    /// lhs[0]: Word (lower 32 bits)
    pub const LHS_0: usize = 0;
    /// lhs[1]: Half (bits 32-47)
    pub const LHS_1: usize = 1;
    /// lhs[2]: Half (bits 48-63, contains MSB)
    pub const LHS_2: usize = 2;

    // Input columns: rhs (DWordHHW = 3 columns)
    /// rhs[0]: Word (lower 32 bits)
    pub const RHS_0: usize = 3;
    /// rhs[1]: Half (bits 32-47)
    pub const RHS_1: usize = 4;
    /// rhs[2]: Half (bits 48-63, contains MSB)
    pub const RHS_2: usize = 5;

    // Input column: signed flag
    /// signed: Bit (1 for signed comparison, 0 for unsigned)
    pub const SIGNED: usize = 6;

    // Output column
    /// lt: Bit (result: 1 if lhs < rhs, 0 otherwise)
    pub const LT: usize = 7;

    // Auxiliary columns: lhs_sub_rhs (DWordHL = 4 halfwords)
    /// lhs_sub_rhs[0]: Half (bits 0-15 of lhs - rhs)
    pub const LHS_SUB_RHS_0: usize = 8;
    /// lhs_sub_rhs[1]: Half (bits 16-31)
    pub const LHS_SUB_RHS_1: usize = 9;
    /// lhs_sub_rhs[2]: Half (bits 32-47)
    pub const LHS_SUB_RHS_2: usize = 10;
    /// lhs_sub_rhs[3]: Half (bits 48-63)
    pub const LHS_SUB_RHS_3: usize = 11;

    // Auxiliary columns: MSB extractions
    /// lhs_msb: Bit (MSB of lhs, i.e., bit 63)
    pub const LHS_MSB: usize = 12;
    /// rhs_msb: Bit (MSB of rhs, i.e., bit 63)
    pub const RHS_MSB: usize = 13;

    // Every LT lookup (CPU SLT/BLT/BGE dispatch and the internal
    // memw/memw_aligned/dvrm comparisons) goes through the unified `ALU` bus,
    // so one multiplicity column suffices.
    /// invert: Bit — invert the comparison (BGE/BGEU); `out = lt XOR invert`.
    pub const INVERT: usize = 14;
    /// out: the ALU result `lt XOR invert` (the low word; high word is 0).
    pub const OUT: usize = 15;
    /// μ: multiplicity for the `ALU` bus receiver.
    pub const MU: usize = 16;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 17;
}

// =========================================================================
// Trace generation
// =========================================================================

/// A single LT operation to be added to the trace.
///
/// Every operation is dispatched on the unified `ALU` bus; the `invert` flag
/// distinguishes plain less-than (memw/dvrm internal checks, CPU `SLT[U]`/`BLT[U]`)
/// from the inverted form (`BGE[U]`).
///
/// Derives Hash and Eq so it can be used as a HashMap key for deduplication.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LtOperation {
    /// Left operand (64-bit value)
    pub lhs: u64,
    /// Right operand (64-bit value)
    pub rhs: u64,
    /// Whether to do signed comparison
    pub signed: bool,
    /// Whether to invert the result (`out = lt XOR invert`); used for BGE/BGEU.
    pub invert: bool,
}

impl LtOperation {
    /// Create a new LT operation with `invert = false` (plain less-than).
    pub fn new(lhs: u64, rhs: u64, signed: bool) -> Self {
        Self {
            lhs,
            rhs,
            signed,
            invert: false,
        }
    }

    /// Create a new LT operation with an explicit `invert` flag (BGE/BGEU dispatch).
    pub fn new_with_invert(lhs: u64, rhs: u64, signed: bool, invert: bool) -> Self {
        Self {
            lhs,
            rhs,
            signed,
            invert,
        }
    }

    /// Compute the raw less-than result (before inversion).
    pub fn compute_lt(&self) -> bool {
        if self.signed {
            (self.lhs as i64) < (self.rhs as i64)
        } else {
            self.lhs < self.rhs
        }
    }

    /// The ALU output: `lt XOR invert`.
    pub fn compute_out(&self) -> bool {
        self.compute_lt() ^ self.invert
    }
}

/// Generates the LT trace table from a list of operations.
///
/// Duplicate operations (same lhs, rhs, signed) are merged into a single row
/// with their multiplicities summed. The table is then padded to the next power of 2.
pub fn generate_lt_trace(
    operations: &[LtOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    use std::collections::HashMap;

    // Deduplicate operations: (lhs, rhs, signed) -> multiplicity
    let mut op_map: HashMap<LtOperation, u64> = HashMap::new();
    for op in operations {
        *op_map.entry(op.clone()).or_insert(0) += 1;
    }

    let unique_ops: Vec<_> = op_map.into_iter().collect();
    let num_rows = unique_ops.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        vec![FE::zero(); num_rows * cols::NUM_COLUMNS],
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, (op, multiplicity)) in unique_ops.iter().enumerate() {
        // Store input columns
        table.set_dword_hhw(row_idx, cols::LHS_0, op.lhs);
        table.set_dword_hhw(row_idx, cols::RHS_0, op.rhs);
        table.set_bool(row_idx, cols::SIGNED, op.signed);

        // Compute lt result
        let lt = op.compute_lt();
        table.set_bool(row_idx, cols::LT, lt);

        // Compute lhs_sub_rhs = lhs - rhs (wrapping)
        // Note: We compute this as a 64-bit wrapping subtraction
        let lhs_sub_rhs = op.lhs.wrapping_sub(op.rhs);

        // Store lhs_sub_rhs as DWordHL: [Half, Half, Half, Half]
        table.set_dword_hl(row_idx, cols::LHS_SUB_RHS_0, lhs_sub_rhs);

        // Compute MSBs (bit 63 of each value)
        let lhs_msb = (op.lhs >> 63) & 1;
        let rhs_msb = (op.rhs >> 63) & 1;
        table.set_u64(row_idx, cols::LHS_MSB, lhs_msb);
        table.set_u64(row_idx, cols::RHS_MSB, rhs_msb);

        // ALU-bus fields: invert + the inverted output.
        table.set_bool(row_idx, cols::INVERT, op.invert);
        table.set_bool(row_idx, cols::OUT, op.compute_out());

        // All LT lookups go through the unified ALU bus → single multiplicity.
        table.set_u64(row_idx, cols::MU, *multiplicity);
    }

    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the LT table.
///
/// The LT table:
/// - **Sends** MSB16 lookups for lhs_msb and rhs_msb extraction
/// - **Sends** IS_HALFWORD lookups for lhs_sub_rhs range checks
/// - **Receives** LT lookups from other tables (CPU)
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // MSB16[lhs[2]] -> lhs_msb
        // Input: lhs[2] (half containing MSB)
        // Output: lhs_msb
        BusInteraction::sender(
            BusId::Msb16,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: cols::LHS_2,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::LHS_MSB,
                    packing: Packing::Direct,
                },
            ],
        ),
        // MSB16[rhs[2]] -> rhs_msb
        BusInteraction::sender(
            BusId::Msb16,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: cols::RHS_2,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::RHS_MSB,
                    packing: Packing::Direct,
                },
            ],
        ),
        // IS_HALFWORD[lhs_sub_rhs[0]]
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::LHS_SUB_RHS_0,
                packing: Packing::Direct,
            }],
        ),
        // IS_HALFWORD[lhs_sub_rhs[1]]
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::LHS_SUB_RHS_1,
                packing: Packing::Direct,
            }],
        ),
        // IS_HALFWORD[lhs_sub_rhs[2]]
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::LHS_SUB_RHS_2,
                packing: Packing::Direct,
            }],
        ),
        // IS_HALFWORD[lhs_sub_rhs[3]]
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::LHS_SUB_RHS_3,
                packing: Packing::Direct,
            }],
        ),
        // IS_HALFWORD[lhs[1]]
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::LHS_1,
                packing: Packing::Direct,
            }],
        ),
        // IS_HALFWORD[rhs[1]]
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::RHS_1,
                packing: Packing::Direct,
            }],
        ),
        // ALU[lhs, rhs, opsel(LT) + 32*signed + 64*invert] -> out  (receiver).
        // Every LT lookup arrives here: the CPU dispatches SLT/BLT/BGE on the
        // unified ALU bus, and the internal memw/memw_aligned/dvrm comparisons
        // (timestamps and |r|<|d|) encode `signed=0, invert=0`. lhs/rhs are
        // packed DWordHHW -> [lo32, hi32] (matching DWordWL senders); the
        // output is [out, 0] (a comparison result fits in the low word).
        BusInteraction::receiver(
            BusId::Alu,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: cols::LHS_0,
                    packing: Packing::DWordHHW,
                },
                BusValue::Packed {
                    start_column: cols::RHS_0,
                    packing: Packing::DWordHHW,
                },
                BusValue::linear(vec![
                    LinearTerm::Constant(alu_op::LT as i64),
                    LinearTerm::Column {
                        coefficient: 32,
                        column: cols::SIGNED,
                    },
                    LinearTerm::Column {
                        coefficient: 64,
                        column: cols::INVERT,
                    },
                ]),
                BusValue::Packed {
                    start_column: cols::OUT,
                    packing: Packing::Direct,
                },
                BusValue::constant(0),
            ],
        ),
    ]
}

// =========================================================================
// Constraints
// =========================================================================

/// LT table constraint for virtual carry IS_BIT checks and the LT formula.
///
/// This constraint embeds the virtual carry computations and verifies:
/// 1. IS_BIT<carry[0]> and IS_BIT<carry[1]> (carry values are 0 or 1)
/// 2. LT formula: lt = signed * (A*(1-B) + A*C + (1-B)*C) + (1-signed) * unsigned_lt
///
/// Where A = lhs_msb, B = rhs_msb, C = carry[1], unsigned_lt = carry[1]
pub struct LtConstraint {
    /// Unique constraint identifier
    constraint_idx: usize,
    /// Which constraint to check (0 = carry[0] IS_BIT, 1 = carry[1] IS_BIT, 2 = LT formula)
    kind: LtConstraintKind,
}

/// Kind of LT constraint.
#[derive(Debug, Clone, Copy)]
pub enum LtConstraintKind {
    /// IS_BIT constraint on virtual carry[0]
    Carry0IsBit,
    /// IS_BIT constraint on virtual carry[1]
    Carry1IsBit,
    /// LT formula constraint
    LtFormula,
    /// `out = lt XOR invert`, i.e. `out - (lt + invert - 2*lt*invert) = 0`
    /// (`lt.toml:159`). The ALU bus consumes `out`, while `LtFormula` only binds
    /// `lt` — without this the `out` column (used for BGE/BGEU via `invert`) is
    /// free and any comparison result can be forged.
    OutXorInvert,
    /// IS_BIT constraint on `invert` (`lt:c:range_invert`).
    InvertIsBit,
    /// IS_BIT constraint on `signed` (`lt:c:range_signed`).
    SignedIsBit,
}

impl LtConstraint {
    /// Creates a new LT constraint.
    pub fn new(kind: LtConstraintKind, constraint_idx: usize) -> Self {
        Self {
            constraint_idx,
            kind,
        }
    }

    /// Compute virtual carry[0] from the addition check.
    ///
    /// carry[0] = 2^(-32) * (rhs[0] + cast(lhs_sub_rhs, DWordWL)[0] - lhs[0])
    ///
    /// Where cast(lhs_sub_rhs, DWordWL)[0] = lhs_sub_rhs[0] + 2^16 * lhs_sub_rhs[1]
    fn compute_carry_0<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let lhs_0 = step.get_main_evaluation_element(0, cols::LHS_0).clone();
        let rhs_0 = step.get_main_evaluation_element(0, cols::RHS_0).clone();
        let sub_0 = step
            .get_main_evaluation_element(0, cols::LHS_SUB_RHS_0)
            .clone();
        let sub_1 = step
            .get_main_evaluation_element(0, cols::LHS_SUB_RHS_1)
            .clone();

        // cast(lhs_sub_rhs, DWordWL)[0] = sub_0 + 2^16 * sub_1
        let shift_16 = FieldElement::<F>::from(SHIFT_16);
        let sub_lo = &sub_0 + &sub_1 * &shift_16;

        // carry[0] = (rhs[0] + sub_lo - lhs[0]) / 2^32
        let inv_2_32 = FieldElement::<F>::from(crate::constraints::templates::INV_SHIFT_32);
        (&rhs_0 + &sub_lo - &lhs_0) * &inv_2_32
    }

    /// Compute virtual carry[1] from the addition check.
    ///
    /// carry[1] = 2^(-32) * (cast(rhs, DWordWL)[1] + cast(lhs_sub_rhs, DWordWL)[1] + carry[0] - cast(lhs, DWordWL)[1])
    ///
    /// Where:
    /// - cast(rhs, DWordWL)[1] = rhs[1] + 2^16 * rhs[2]
    /// - cast(lhs_sub_rhs, DWordWL)[1] = lhs_sub_rhs[2] + 2^16 * lhs_sub_rhs[3]
    /// - cast(lhs, DWordWL)[1] = lhs[1] + 2^16 * lhs[2]
    fn compute_carry_1<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let lhs_1 = step.get_main_evaluation_element(0, cols::LHS_1).clone();
        let lhs_2 = step.get_main_evaluation_element(0, cols::LHS_2).clone();
        let rhs_1 = step.get_main_evaluation_element(0, cols::RHS_1).clone();
        let rhs_2 = step.get_main_evaluation_element(0, cols::RHS_2).clone();
        let sub_2 = step
            .get_main_evaluation_element(0, cols::LHS_SUB_RHS_2)
            .clone();
        let sub_3 = step
            .get_main_evaluation_element(0, cols::LHS_SUB_RHS_3)
            .clone();

        let shift_16 = FieldElement::<F>::from(SHIFT_16);

        // cast(lhs, DWordWL)[1] = lhs[1] + 2^16 * lhs[2]
        let lhs_hi = &lhs_1 + &lhs_2 * &shift_16;

        // cast(rhs, DWordWL)[1] = rhs[1] + 2^16 * rhs[2]
        let rhs_hi = &rhs_1 + &rhs_2 * &shift_16;

        // cast(lhs_sub_rhs, DWordWL)[1] = sub_2 + 2^16 * sub_3
        let sub_hi = &sub_2 + &sub_3 * &shift_16;

        // carry[0]
        let carry_0 = self.compute_carry_0(step);

        // carry[1] = (rhs_hi + sub_hi + carry_0 - lhs_hi) / 2^32
        let inv_2_32 = FieldElement::<F>::from(crate::constraints::templates::INV_SHIFT_32);
        (&rhs_hi + &sub_hi + &carry_0 - &lhs_hi) * &inv_2_32
    }

    /// Compute the constraint value.
    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let one = FieldElement::<F>::one();

        match self.kind {
            LtConstraintKind::Carry0IsBit => {
                // IS_BIT<carry[0]>: carry[0] * (1 - carry[0]) = 0
                let c0 = self.compute_carry_0(step);
                &c0 * (one - &c0)
            }
            LtConstraintKind::Carry1IsBit => {
                // IS_BIT<carry[1]>: carry[1] * (1 - carry[1]) = 0
                let c1 = self.compute_carry_1(step);
                &c1 * (one - &c1)
            }
            LtConstraintKind::LtFormula => {
                // LT formula:
                // lt = signed * (A*(1-B) + A*C + (1-B)*C) + (1-signed) * unsigned_lt
                // Where A = lhs_msb, B = rhs_msb, C = carry[1], unsigned_lt = carry[1]
                let lt = step.get_main_evaluation_element(0, cols::LT).clone();
                let signed = step.get_main_evaluation_element(0, cols::SIGNED).clone();
                let a = step.get_main_evaluation_element(0, cols::LHS_MSB).clone();
                let b = step.get_main_evaluation_element(0, cols::RHS_MSB).clone();
                let c = self.compute_carry_1(step);

                // unsigned_lt = carry[1]
                let unsigned_lt = c.clone();

                // signed_lt = A*(1-B) + A*C + (1-B)*C
                // = A - A*B + A*C + C - B*C
                // = A*(1-B+C) + C*(1-B)
                let one_minus_b = &one - &b;
                let signed_lt = &a * &one_minus_b + &a * &c + &one_minus_b * &c;

                // lt = signed * signed_lt + (1 - signed) * unsigned_lt
                let expected_lt = &signed * &signed_lt + (&one - &signed) * &unsigned_lt;

                // Constraint: lt - expected_lt = 0
                lt - expected_lt
            }
            LtConstraintKind::OutXorInvert => {
                // out = lt XOR invert = lt + invert - 2*lt*invert
                let out = step.get_main_evaluation_element(0, cols::OUT).clone();
                let lt = step.get_main_evaluation_element(0, cols::LT).clone();
                let invert = step.get_main_evaluation_element(0, cols::INVERT).clone();
                let two = FieldElement::<F>::from(2u64);
                out - (&lt + &invert - two * &lt * &invert)
            }
            LtConstraintKind::InvertIsBit => {
                // invert * (1 - invert) = 0
                let invert = step.get_main_evaluation_element(0, cols::INVERT).clone();
                &invert * (one - &invert)
            }
            LtConstraintKind::SignedIsBit => {
                // signed * (1 - signed) = 0
                let signed = step.get_main_evaluation_element(0, cols::SIGNED).clone();
                &signed * (one - &signed)
            }
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for LtConstraint {
    fn degree(&self) -> usize {
        match self.kind {
            // IS_BIT on virtual carry involves computing carry (degree 1) then X*(1-X) (degree 2)
            LtConstraintKind::Carry0IsBit => 2,
            LtConstraintKind::Carry1IsBit => 2,
            // LT formula involves products like signed * A * (1-B)
            LtConstraintKind::LtFormula => 3,
            // out - (lt + invert - 2*lt*invert): the lt*invert product is degree 2
            LtConstraintKind::OutXorInvert => 2,
            // X*(1-X)
            LtConstraintKind::InvertIsBit => 2,
            LtConstraintKind::SignedIsBit => 2,
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
        self.compute(step)
    }
}

/// Creates all constraints for the LT table.
///
/// Returns: (constraints, next_constraint_idx)
pub fn lt_constraints(constraint_idx_start: usize) -> (Vec<LtConstraint>, usize) {
    let mut idx = constraint_idx_start;
    let constraints = vec![
        LtConstraint::new(LtConstraintKind::Carry0IsBit, {
            let i = idx;
            idx += 1;
            i
        }),
        LtConstraint::new(LtConstraintKind::Carry1IsBit, {
            let i = idx;
            idx += 1;
            i
        }),
        LtConstraint::new(LtConstraintKind::LtFormula, {
            let i = idx;
            idx += 1;
            i
        }),
        // out = lt XOR invert (binds the ALU-bus-consumed `out` column).
        LtConstraint::new(LtConstraintKind::OutXorInvert, {
            let i = idx;
            idx += 1;
            i
        }),
        // Range-check the boolean flags that drive the formula / bus.
        LtConstraint::new(LtConstraintKind::InvertIsBit, {
            let i = idx;
            idx += 1;
            i
        }),
        LtConstraint::new(LtConstraintKind::SignedIsBit, {
            let i = idx;
            idx += 1;
            i
        }),
    ];
    (constraints, idx)
}

/// LT's migrated domain constraints, folded in the same order as
/// [`lt_constraints`] (each residual field-exactly equal to the boxed
/// `LtConstraint::compute`).
///
/// Order: Carry0IsBit, Carry1IsBit, LtFormula, OutXorInvert, InvertIsBit,
/// SignedIsBit. The two virtual carries are computed once and shared with the
/// LT formula (which reads `carry[1]`).
pub fn lt_domain_eval<CB: ConstraintBuilder>(cb: &mut CB) {
    let one = FieldElement::<CB::F>::one();
    let shift_16 = FieldElement::<CB::F>::from(SHIFT_16);
    let inv_2_32 = FieldElement::<CB::F>::from(crate::constraints::templates::INV_SHIFT_32);

    // Read-only phase: pull every needed cell into owned locals.
    let lhs_0 = cb.main(cols::LHS_0).clone();
    let lhs_1 = cb.main(cols::LHS_1).clone();
    let lhs_2 = cb.main(cols::LHS_2).clone();
    let rhs_0 = cb.main(cols::RHS_0).clone();
    let rhs_1 = cb.main(cols::RHS_1).clone();
    let rhs_2 = cb.main(cols::RHS_2).clone();
    let signed = cb.main(cols::SIGNED).clone();
    let lt = cb.main(cols::LT).clone();
    let sub_0 = cb.main(cols::LHS_SUB_RHS_0).clone();
    let sub_1 = cb.main(cols::LHS_SUB_RHS_1).clone();
    let sub_2 = cb.main(cols::LHS_SUB_RHS_2).clone();
    let sub_3 = cb.main(cols::LHS_SUB_RHS_3).clone();
    let a = cb.main(cols::LHS_MSB).clone();
    let b = cb.main(cols::RHS_MSB).clone();
    let invert = cb.main(cols::INVERT).clone();
    let out = cb.main(cols::OUT).clone();

    // Virtual carry[0] = (rhs[0] + cast(lhs_sub_rhs)[0] - lhs[0]) / 2^32.
    let sub_lo = &sub_0 + &sub_1 * &shift_16;
    let carry_0 = (&rhs_0 + &sub_lo - &lhs_0) * &inv_2_32;

    // Virtual carry[1] = (rhs_hi + sub_hi + carry_0 - lhs_hi) / 2^32.
    let lhs_hi = &lhs_1 + &lhs_2 * &shift_16;
    let rhs_hi = &rhs_1 + &rhs_2 * &shift_16;
    let sub_hi = &sub_2 + &sub_3 * &shift_16;
    let carry_1 = (&rhs_hi + &sub_hi + &carry_0 - &lhs_hi) * &inv_2_32;

    // 0: IS_BIT<carry[0]>.
    cb.fold(&carry_0 * (&one - &carry_0));
    // 1: IS_BIT<carry[1]>.
    cb.fold(&carry_1 * (&one - &carry_1));

    // 2: LT formula. C = carry[1], unsigned_lt = carry[1].
    let c = carry_1;
    let one_minus_b = &one - &b;
    let signed_lt = &a * &one_minus_b + &a * &c + &one_minus_b * &c;
    let expected_lt = &signed * &signed_lt + (&one - &signed) * &c;
    cb.fold(&lt - &expected_lt);

    // 3: out = lt XOR invert = lt + invert - 2*lt*invert.
    let two = FieldElement::<CB::F>::from(2u64);
    cb.fold(out - (&lt + &invert - &two * &lt * &invert));

    // 4: IS_BIT<invert>.
    cb.fold(&invert * (&one - &invert));
    // 5: IS_BIT<signed>.
    cb.fold(&signed * (&one - &signed));
}

/// LT's migrated domain constraints as an object-safe `TableConstraints`.
pub struct LtDomain;

impl TableConstraints<GoldilocksField, GoldilocksExtension> for LtDomain {
    fn eval_prover(
        &self,
        cb: &mut ProverConstraintBuilder<GoldilocksField, GoldilocksExtension>,
        _ctx: &ConstraintContext<GoldilocksField, GoldilocksExtension>,
    ) {
        lt_domain_eval(cb);
    }

    fn eval_verifier(
        &self,
        cb: &mut VerifierConstraintBuilder<GoldilocksExtension>,
        _ctx: &ConstraintContext<GoldilocksExtension, GoldilocksExtension>,
    ) {
        lt_domain_eval(cb);
    }
}
