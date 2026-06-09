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
//! - Receiver: LT (provides less-than results to other tables)

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::TransitionConstraint;
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;

use super::types::{BusId, FE, FxHashMap, GoldilocksExtension, GoldilocksField, SHIFT_16};

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

    // Multiplicity column
    /// μ: multiplicity for bus interactions
    pub const MU: usize = 14;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 15;
}

// =========================================================================
// Trace generation
// =========================================================================

/// A single LT operation to be added to the trace.
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
}

impl LtOperation {
    /// Create a new LT operation.
    pub fn new(lhs: u64, rhs: u64, signed: bool) -> Self {
        Self { lhs, rhs, signed }
    }

    /// Compute the less-than result.
    pub fn compute_lt(&self) -> bool {
        if self.signed {
            (self.lhs as i64) < (self.rhs as i64)
        } else {
            self.lhs < self.rhs
        }
    }
}

/// Generates the LT trace table from a list of operations.
///
/// Duplicate operations (same lhs, rhs, signed) are merged into a single row
/// with their multiplicities summed. The table is then padded to the next power of 2.
pub fn generate_lt_trace(
    operations: &[LtOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    // Deduplicate operations: (lhs, rhs, signed) -> multiplicity
    let mut op_map: FxHashMap<LtOperation, u64> = FxHashMap::default();
    for op in operations {
        *op_map.entry(op.clone()).or_insert(0) += 1;
    }

    let unique_ops: Vec<_> = op_map.into_iter().collect();
    let num_rows = unique_ops.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, (op, multiplicity)) in unique_ops.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;

        // Extract lhs as DWordHHW: [Word, Half, Half]
        let lhs_0 = (op.lhs & 0xFFFF_FFFF) as u32; // bits 0-31
        let lhs_1 = ((op.lhs >> 32) & 0xFFFF) as u16; // bits 32-47
        let lhs_2 = ((op.lhs >> 48) & 0xFFFF) as u16; // bits 48-63

        // Extract rhs as DWordHHW: [Word, Half, Half]
        let rhs_0 = (op.rhs & 0xFFFF_FFFF) as u32; // bits 0-31
        let rhs_1 = ((op.rhs >> 32) & 0xFFFF) as u16; // bits 32-47
        let rhs_2 = ((op.rhs >> 48) & 0xFFFF) as u16; // bits 48-63

        // Store input columns
        data[base + cols::LHS_0] = FE::from(lhs_0 as u64);
        data[base + cols::LHS_1] = FE::from(lhs_1 as u64);
        data[base + cols::LHS_2] = FE::from(lhs_2 as u64);
        data[base + cols::RHS_0] = FE::from(rhs_0 as u64);
        data[base + cols::RHS_1] = FE::from(rhs_1 as u64);
        data[base + cols::RHS_2] = FE::from(rhs_2 as u64);
        data[base + cols::SIGNED] = FE::from(if op.signed { 1u64 } else { 0u64 });

        // Compute lt result
        let lt = op.compute_lt();
        data[base + cols::LT] = FE::from(if lt { 1u64 } else { 0u64 });

        // Compute lhs_sub_rhs = lhs - rhs (wrapping)
        // Note: We compute this as a 64-bit wrapping subtraction
        let lhs_sub_rhs = op.lhs.wrapping_sub(op.rhs);

        // Store lhs_sub_rhs as DWordHL: [Half, Half, Half, Half]
        let sub_0 = (lhs_sub_rhs & 0xFFFF) as u16;
        let sub_1 = ((lhs_sub_rhs >> 16) & 0xFFFF) as u16;
        let sub_2 = ((lhs_sub_rhs >> 32) & 0xFFFF) as u16;
        let sub_3 = ((lhs_sub_rhs >> 48) & 0xFFFF) as u16;
        data[base + cols::LHS_SUB_RHS_0] = FE::from(sub_0 as u64);
        data[base + cols::LHS_SUB_RHS_1] = FE::from(sub_1 as u64);
        data[base + cols::LHS_SUB_RHS_2] = FE::from(sub_2 as u64);
        data[base + cols::LHS_SUB_RHS_3] = FE::from(sub_3 as u64);

        // Compute MSBs (bit 63 of each value)
        let lhs_msb = (op.lhs >> 63) & 1;
        let rhs_msb = (op.rhs >> 63) & 1;
        data[base + cols::LHS_MSB] = FE::from(lhs_msb);
        data[base + cols::RHS_MSB] = FE::from(rhs_msb);

        // Multiplicity: aggregated count of this operation
        data[base + cols::MU] = FE::from(*multiplicity);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
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
        // LT[lhs, rhs, signed] -> lt (receiver)
        // lhs is DWordHHW, rhs is DWordHHW, signed is Bit, lt is Bit
        // Uses DWordHHW packing: reads 3 columns (Word, Half, Half), produces 2 bus elements [lo32, hi32]
        // This allows DWordWL senders (like MEMW timestamps) to match via Packing::DWordWL
        BusInteraction::receiver(
            BusId::Lt,
            Multiplicity::Column(cols::MU),
            vec![
                // lhs as DWordHHW (reads 3 columns: Word, Half, Half; produces 2 elements: [lo32, hi32])
                BusValue::Packed {
                    start_column: cols::LHS_0,
                    packing: Packing::DWordHHW,
                },
                // rhs as DWordHHW (reads 3 columns, produces 2 elements)
                BusValue::Packed {
                    start_column: cols::RHS_0,
                    packing: Packing::DWordHHW,
                },
                // signed
                BusValue::Packed {
                    start_column: cols::SIGNED,
                    packing: Packing::Direct,
                },
                // lt (output)
                BusValue::Packed {
                    start_column: cols::LT,
                    packing: Packing::Direct,
                },
            ],
        ),
    ]
}

/// Compute virtual carry[0] and carry[1] for the addition rhs + lhs_sub_rhs = lhs
///
/// From spec:
/// carry[0] = 2^(-32) * (rhs[0] + cast(lhs_sub_rhs, DWordWL)[0] - lhs[0])
/// carry[1] = 2^(-32) * (cast(rhs, DWordWL)[1] + cast(lhs_sub_rhs, DWordWL)[1] + carry[0] - cast(lhs, DWordWL)[1])
///
/// Note: carry[1] = 1 means lhs < rhs (unsigned), because the subtraction borrowed
pub fn compute_carries(lhs: u64, rhs: u64, lhs_sub_rhs: u64) -> (u64, u64) {
    // Cast to DWordWL format (2 words)
    let lhs_lo = lhs & 0xFFFF_FFFF;
    let lhs_hi = lhs >> 32;

    let rhs_lo = rhs & 0xFFFF_FFFF;
    let rhs_hi = rhs >> 32;

    let sub_lo = lhs_sub_rhs & 0xFFFF_FFFF;
    let sub_hi = lhs_sub_rhs >> 32;

    // carry[0] = (rhs_lo + sub_lo - lhs_lo) / 2^32
    // This should be 0 or 1 (or -1 in some representations)
    let sum_lo = rhs_lo + sub_lo;
    let carry_0 = if sum_lo >= lhs_lo {
        (sum_lo - lhs_lo) >> 32
    } else {
        // This shouldn't happen if lhs_sub_rhs is computed correctly
        0
    };

    // carry[1] = (rhs_hi + sub_hi + carry_0 - lhs_hi) / 2^32
    let sum_hi = rhs_hi + sub_hi + carry_0;
    let carry_1 = if sum_hi >= lhs_hi {
        (sum_hi - lhs_hi) >> 32
    } else {
        // This indicates lhs < rhs (unsigned)
        // In field arithmetic, this would be handled differently
        1
    };

    (carry_0, carry_1)
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
    ];
    (constraints, idx)
}
