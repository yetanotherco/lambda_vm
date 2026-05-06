//! BinaryAdd AIR — proves `lhs + rhs = sum (mod 2^64)` for ADD-style ops
//! that the CPU dispatches via [`BusId::BinaryAdd`].
//!
//! ## Phase 2 progress
//!
//! - **Step 1** (✓): skeleton AIR + bus ID + wiring.
//! - **Step 2** (this commit): carry-chain transition constraints, halfword
//!   range-check senders, and the `Multiplicity::Sum(MU_ADD, MU_SUB)`
//!   receiver on [`BusId::BinaryAdd`]. Trace-builder collects ADD/LOAD
//!   ops from CPU and emits one row per unique `(lhs, rhs, sum)` tuple
//!   with `MU_ADD` set; `MU_SUB` stays at 0 until step 3.
//! - **Step 3**: trace-builder also collects STORE/SUB/BEQ/JALR.
//! - **Step 4**: drop the now-redundant inline carry constraints from CPU.
//!
//! ## Column layout (14 total)
//!
//! Operands are stored as 4 halfwords each (DWordHL). Each halfword is
//! range-checked via an `IS_HALFWORD` sender; the carry chain itself uses
//! [`AddConstraint`]'s virtual-carry trick (no committed carry columns).
//!
//! | Range | Cols | Description |
//! |---|---:|---|
//! | `LHS_0..3` | 4 | lhs as DWordHL halfwords |
//! | `RHS_0..3` | 4 | rhs as DWordHL halfwords |
//! | `SUM_0..3` | 4 | `lhs + rhs (mod 2^64)` as DWordHL halfwords |
//! | `MU_ADD`   | 1 | multiplicity for forward-add (ADD/LOAD/STORE/JALR) |
//! | `MU_SUB`   | 1 | multiplicity for reverse-add (SUB/BEQ) |

use std::collections::HashMap;

use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use crate::constraints::templates::{AddConstraint, AddOperand};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column layout
// =========================================================================

/// Column definitions for the BinaryAdd table.
pub mod cols {
    /// lhs halfword 0 (bits 0..16)
    pub const LHS_0: usize = 0;
    /// lhs halfword 1 (bits 16..32)
    pub const LHS_1: usize = 1;
    /// lhs halfword 2 (bits 32..48)
    pub const LHS_2: usize = 2;
    /// lhs halfword 3 (bits 48..64)
    pub const LHS_3: usize = 3;

    pub const RHS_0: usize = 4;
    pub const RHS_1: usize = 5;
    pub const RHS_2: usize = 6;
    pub const RHS_3: usize = 7;

    pub const SUM_0: usize = 8;
    pub const SUM_1: usize = 9;
    pub const SUM_2: usize = 10;
    pub const SUM_3: usize = 11;

    /// Multiplicity for forward-add (ADD/LOAD/STORE/JALR).
    pub const MU_ADD: usize = 12;
    /// Multiplicity for reverse-add (SUB/BEQ): the row absorbs CPU senders that
    /// supply `(arg2, res, arg1)` as `(lhs, rhs, sum)` to prove `arg2+res=arg1`.
    pub const MU_SUB: usize = 13;

    pub const NUM_COLUMNS: usize = 14;
}

// =========================================================================
// BinaryAddOperation — input to trace generation
// =========================================================================

/// A single (lhs, rhs, sum) triple with `lhs + rhs = sum (mod 2^64)`.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BinaryAddOperation {
    pub lhs: u64,
    pub rhs: u64,
    pub sum: u64,
}

impl BinaryAddOperation {
    pub fn new(lhs: u64, rhs: u64, sum: u64) -> Self {
        debug_assert_eq!(
            lhs.wrapping_add(rhs),
            sum,
            "BinaryAddOperation invariant: lhs + rhs == sum (mod 2^64)"
        );
        Self { lhs, rhs, sum }
    }

    /// Build the ADD-flavour op for an ADD/LOAD CPU row (`res = arg1 + arg2`).
    pub fn for_add(arg1: u64, arg2: u64) -> Self {
        let sum = arg1.wrapping_add(arg2);
        Self {
            lhs: arg1,
            rhs: arg2,
            sum,
        }
    }

    /// Build the SUB-flavour op for a SUB/BEQ CPU row. The CPU's row has
    /// `res = arg1 - arg2 (mod 2^64)`; the BinaryAdd row's `(lhs, rhs, sum)`
    /// is `(arg2, res, arg1)` so that `lhs + rhs = sum` proves the SUB.
    pub fn for_sub(arg1: u64, arg2: u64) -> Self {
        let res = arg1.wrapping_sub(arg2);
        Self {
            lhs: arg2,
            rhs: res,
            sum: arg1,
        }
    }
}

/// Which receiver flavour absorbs a given CPU op. Forwards (ADD/LOAD/STORE/JALR)
/// dispatch with `(lhs, rhs, sum) = (arg1, arg2, res)`. Reverses (SUB/BEQ)
/// dispatch with `(lhs, rhs, sum) = (arg2, res, arg1)`, proving `arg2+res=arg1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryAddFlavour {
    Add,
    Sub,
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the BinaryAdd trace from a list of (op, flavour) pairs.
///
/// Operations are deduplicated by `(lhs, rhs, sum)`. Each unique row tracks
/// separate multiplicities for the ADD and SUB flavours.
pub fn generate_binary_add_trace(
    operations: &[(BinaryAddOperation, BinaryAddFlavour)],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let mut multiplicities: HashMap<BinaryAddOperation, (u64, u64)> = HashMap::new();
    for (op, flavour) in operations {
        let entry = multiplicities.entry(*op).or_insert((0, 0));
        match flavour {
            BinaryAddFlavour::Add => entry.0 += 1,
            BinaryAddFlavour::Sub => entry.1 += 1,
        }
    }

    let unique_ops: Vec<_> = multiplicities.into_iter().collect();
    let num_rows = unique_ops.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, (op, (mu_add, mu_sub))) in unique_ops.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;
        write_op_halfwords(&mut data, base + cols::LHS_0, op.lhs);
        write_op_halfwords(&mut data, base + cols::RHS_0, op.rhs);
        write_op_halfwords(&mut data, base + cols::SUM_0, op.sum);
        data[base + cols::MU_ADD] = FE::from(*mu_add);
        data[base + cols::MU_SUB] = FE::from(*mu_sub);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

#[inline]
fn write_op_halfwords(data: &mut [FE], start: usize, value: u64) {
    data[start] = FE::from(value & 0xFFFF);
    data[start + 1] = FE::from((value >> 16) & 0xFFFF);
    data[start + 2] = FE::from((value >> 32) & 0xFFFF);
    data[start + 3] = FE::from((value >> 48) & 0xFFFF);
}

// =========================================================================
// Constraints
// =========================================================================

/// Returns the BinaryAdd transition constraints.
///
/// Two unconditional carry-chain constraints (from `AddConstraint`)
/// enforce `LHS + RHS = SUM (mod 2^64)` on every row. On padding rows
/// (all zeros) the constraint trivially evaluates to `0 = 0`.
pub fn binary_add_constraints()
-> Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>> {
    let lhs = AddOperand::from_dword_hl(cols::LHS_0);
    let rhs = AddOperand::from_dword_hl(cols::RHS_0);
    let sum = AddOperand::from_dword_hl(cols::SUM_0);
    let (carry_0, carry_1) = AddConstraint::new_pair(Vec::new(), lhs, rhs, sum, 0);
    vec![carry_0.boxed(), carry_1.boxed()]
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Returns the BinaryAdd bus interactions.
///
/// **Senders (12)**: one `IS_HALFWORD` per halfword column with multiplicity
/// `Sum(MU_ADD, MU_SUB)`. Range-checks each halfword to `[0, 2^16)`, which
/// the carry constraints rely on for soundness. The `Sum` is safe because
/// both `MU_*` columns are anchored by their respective receivers below —
/// a malicious prover cannot inflate one to cancel the other without
/// breaking bus balance on `BusId::BinaryAdd`.
///
/// **Receivers (2)** on `BusId::BinaryAdd`, both carrying
/// `(lhs::DWordHL, rhs::DWordHL, sum::DWordHL)`:
/// - `Multiplicity::Column(MU_ADD)` — absorbs forward-flavour CPU senders
///   (ADD/LOAD/STORE/JALR), where `(lhs, rhs, sum) = (arg1, arg2, res)` or
///   `(arg1, imm, res)` or `(pc, instr_size, res)` per family.
/// - `Multiplicity::Column(MU_SUB)` — absorbs reverse-flavour CPU senders
///   (SUB/BEQ) which transmit `(arg2, res, arg1)` so the row's `lhs+rhs=sum`
///   proves `arg2 + res = arg1`, i.e. `res = arg1 - arg2`.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(13);

    let halfword_cols = [
        cols::LHS_0,
        cols::LHS_1,
        cols::LHS_2,
        cols::LHS_3,
        cols::RHS_0,
        cols::RHS_1,
        cols::RHS_2,
        cols::RHS_3,
        cols::SUM_0,
        cols::SUM_1,
        cols::SUM_2,
        cols::SUM_3,
    ];
    for col in halfword_cols {
        // Step 3: now safe to use Sum(MU_ADD, MU_SUB) because both columns
        // are anchored by their respective BusId::BinaryAdd receivers below.
        // A malicious prover cannot set MU_SUB = p - MU_ADD without breaking
        // bus balance on BusId::BinaryAdd's MU_SUB receiver.
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Sum(cols::MU_ADD, cols::MU_SUB),
            vec![BusValue::Packed {
                start_column: col,
                packing: Packing::Direct,
            }],
        ));
    }

    // Forward-flavour receiver (ADD/LOAD/STORE/JALR senders absorbed here).
    // CPU sends `(arg1, arg2, res)`; the row stores `(lhs, rhs, sum)` with the
    // same numeric values, validated by the carry-chain constraints.
    interactions.push(BusInteraction::receiver(
        BusId::BinaryAdd,
        Multiplicity::Column(cols::MU_ADD),
        vec![
            BusValue::Packed {
                start_column: cols::LHS_0,
                packing: Packing::DWordHL,
            },
            BusValue::Packed {
                start_column: cols::RHS_0,
                packing: Packing::DWordHL,
            },
            BusValue::Packed {
                start_column: cols::SUM_0,
                packing: Packing::DWordHL,
            },
        ],
    ));

    // Reverse-flavour receiver (SUB/BEQ senders absorbed here). CPU swaps
    // operands on send: it transmits `(arg2, res, arg1)` so this row's
    // `(lhs, rhs, sum)` again satisfies `lhs + rhs = sum` — proving
    // `arg2 + res = arg1`, i.e. SUB semantics. Same row data shape as the
    // forward receiver; only the multiplicity column differs.
    interactions.push(BusInteraction::receiver(
        BusId::BinaryAdd,
        Multiplicity::Column(cols::MU_SUB),
        vec![
            BusValue::Packed {
                start_column: cols::LHS_0,
                packing: Packing::DWordHL,
            },
            BusValue::Packed {
                start_column: cols::RHS_0,
                packing: Packing::DWordHL,
            },
            BusValue::Packed {
                start_column: cols::SUM_0,
                packing: Packing::DWordHL,
            },
        ],
    ));

    interactions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_invariant_holds_for_for_add() {
        let op = BinaryAddOperation::for_add(5, 7);
        assert_eq!(op.lhs, 5);
        assert_eq!(op.rhs, 7);
        assert_eq!(op.sum, 12);
    }

    #[test]
    fn op_invariant_wraps_at_u64() {
        let op = BinaryAddOperation::for_add(u64::MAX, 1);
        assert_eq!(op.sum, 0);
    }

    #[test]
    fn trace_dedupes_by_operand_tuple() {
        let op = BinaryAddOperation::for_add(3, 4);
        let ops = vec![
            (op, BinaryAddFlavour::Add),
            (op, BinaryAddFlavour::Add),
            (op, BinaryAddFlavour::Sub),
        ];
        let trace = generate_binary_add_trace(&ops);
        // First (and only meaningful) row.
        let row = trace.main_table.get_row(0);
        assert_eq!(*row[cols::MU_ADD].value(), 2);
        assert_eq!(*row[cols::MU_SUB].value(), 1);
    }

    #[test]
    fn trace_columns_match_layout() {
        let op = BinaryAddOperation::for_add(0x1122_3344_5566_7788, 0x1);
        let trace = generate_binary_add_trace(&[(op, BinaryAddFlavour::Add)]);
        let row = trace.main_table.get_row(0);
        assert_eq!(*row[cols::LHS_0].value(), 0x7788);
        assert_eq!(*row[cols::LHS_1].value(), 0x5566);
        assert_eq!(*row[cols::LHS_2].value(), 0x3344);
        assert_eq!(*row[cols::LHS_3].value(), 0x1122);
        assert_eq!(*row[cols::RHS_0].value(), 0x1);
        assert_eq!(*row[cols::SUM_0].value(), 0x7789);
    }

    #[test]
    fn empty_ops_produces_padded_trace() {
        let trace = generate_binary_add_trace(&[]);
        assert_eq!(trace.main_table.height, 4);
    }
}
