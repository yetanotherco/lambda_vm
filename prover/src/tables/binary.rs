//! Binary AIR — proves `lhs op rhs = res` for `op ∈ {AND, OR, XOR}` on
//! 64-bit operands. CPU dispatches every AND/OR/XOR row here via
//! [`BusId::Binary`]; the per-byte AND_BYTE/OR_BYTE/XOR_BYTE senders that
//! used to live in CPU are now this AIR's responsibility, paying their
//! 24 bus-interaction cost only on rows that actually fire.
//!
//! ## Phase 2 step 5
//!
//! - **Step 1**: skeleton (no constraints, no buses).
//! - **Step 5** (this commit): full implementation. CPU drops its 24
//!   per-byte AND_BYTE/OR_BYTE/XOR_BYTE senders and replaces them with a
//!   single sender on `BusId::Binary`. This AIR stores `(lhs, rhs, res)`
//!   as 24 byte cols + 3 op-selector helper bits, validates `lhs op rhs
//!   = res` byte-by-byte via the BITWISE table, and absorbs the CPU's
//!   single bus message per AND/OR/XOR row.
//! - **Step 6**: shrink CPU's byte cols to word-pair cols. The CPU
//!   sender on `BusId::Binary` will switch from `Packing::DWordBL`
//!   (8-byte storage) to `Packing::DWordWL` (2-word storage); Binary's
//!   own byte cols stay because the per-byte BITWISE bus needs them.
//!
//! ## Column layout (27 total)
//!
//! Byte storage is required: the per-byte AND/OR/XOR BITWISE bus needs
//! byte access, and bus-balance against BITWISE's `[0,256)²` row set
//! transitively range-checks each byte (no separate IS_BYTE needed).
//!
//! | Range | Cols | Description |
//! |---|---:|---|
//! | `LHS[0..8]` | 8 | lhs as 8 little-endian bytes |
//! | `RHS[0..8]` | 8 | rhs as 8 little-endian bytes |
//! | `RES[0..8]` | 8 | `lhs op rhs` as 8 little-endian bytes |
//! | `IS_AND`    | 1 | bit, 1 iff this row processes AND |
//! | `IS_OR`     | 1 | bit, 1 iff this row processes OR  |
//! | `IS_XOR`    | 1 | bit, 1 iff this row processes XOR |
//!
//! At-most-one selector activity is enforced by **three explicit
//! algebraic constraints**: `IS_AND·IS_OR = 0`, `IS_AND·IS_XOR = 0`,
//! `IS_OR·IS_XOR = 0`. Bus balance alone is *not* sufficient — for the
//! degenerate input `lhs == rhs`, both `lhs & rhs` and `lhs | rhs`
//! equal `lhs`, so a confused row with two selectors set could absorb
//! both AND_BYTE and OR_BYTE traffic without imbalance. The mutex
//! constraints close that soundness gap. Each is degree 2, matching
//! the IsBit constraints — no constraint-degree budget impact.

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;

use crate::constraints::templates::IsBitConstraint;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column layout
// =========================================================================

pub mod cols {
    pub const LHS_0: usize = 0;
    pub const LHS_1: usize = 1;
    pub const LHS_2: usize = 2;
    pub const LHS_3: usize = 3;
    pub const LHS_4: usize = 4;
    pub const LHS_5: usize = 5;
    pub const LHS_6: usize = 6;
    pub const LHS_7: usize = 7;

    pub const RHS_0: usize = 8;
    pub const RHS_1: usize = 9;
    pub const RHS_2: usize = 10;
    pub const RHS_3: usize = 11;
    pub const RHS_4: usize = 12;
    pub const RHS_5: usize = 13;
    pub const RHS_6: usize = 14;
    pub const RHS_7: usize = 15;

    pub const RES_0: usize = 16;
    pub const RES_1: usize = 17;
    pub const RES_2: usize = 18;
    pub const RES_3: usize = 19;
    pub const RES_4: usize = 20;
    pub const RES_5: usize = 21;
    pub const RES_6: usize = 22;
    pub const RES_7: usize = 23;

    pub const IS_AND: usize = 24;
    pub const IS_OR: usize = 25;
    pub const IS_XOR: usize = 26;

    pub const NUM_COLUMNS: usize = 27;

    pub const LHS: [usize; 8] = [LHS_0, LHS_1, LHS_2, LHS_3, LHS_4, LHS_5, LHS_6, LHS_7];
    pub const RHS: [usize; 8] = [RHS_0, RHS_1, RHS_2, RHS_3, RHS_4, RHS_5, RHS_6, RHS_7];
    pub const RES: [usize; 8] = [RES_0, RES_1, RES_2, RES_3, RES_4, RES_5, RES_6, RES_7];
}

// =========================================================================
// BinaryOp — input to trace generation
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    And,
    Or,
    Xor,
}

impl BinaryOp {
    /// Encoded value used in the `BusId::Binary` bus message
    /// (`0` for AND, `1` for OR, `2` for XOR — same encoding the CPU
    /// sender produces from its per-op selectors).
    pub const fn bus_encoding(self) -> u64 {
        match self {
            BinaryOp::And => 0,
            BinaryOp::Or => 1,
            BinaryOp::Xor => 2,
        }
    }

    pub fn apply(self, lhs: u64, rhs: u64) -> u64 {
        match self {
            BinaryOp::And => lhs & rhs,
            BinaryOp::Or => lhs | rhs,
            BinaryOp::Xor => lhs ^ rhs,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BinaryOperation {
    pub op: BinaryOp,
    pub lhs: u64,
    pub rhs: u64,
}

impl BinaryOperation {
    pub fn new(op: BinaryOp, lhs: u64, rhs: u64) -> Self {
        Self { op, lhs, rhs }
    }

    pub fn res(&self) -> u64 {
        self.op.apply(self.lhs, self.rhs)
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// Build the Binary trace from collected AND/OR/XOR ops.
///
/// Operations are deduplicated by `(op, lhs, rhs)`. Each unique row sets
/// the matching `IS_*` selector to 1; the others stay 0. The receiver
/// multiplicity on `BusId::Binary` is `Sum3(IS_AND, IS_OR, IS_XOR)` —
/// summed selectors equal the count of CPU senders for this row's op.
///
/// **Multi-occurrence note**: unlike MUL/BinaryAdd, Binary's per-row
/// multiplicity here is just `0` or `1`. If two CPU rows share the same
/// `(op, lhs, rhs)`, the dedup collapses them to one row but bus balance
/// requires multiplicity = 2 — which a single bit selector cannot
/// express. To keep the design simple, this implementation emits **one
/// Binary row per CPU op** (no dedup). Future iterations can add a
/// dedicated `MU` column if profiling shows worthwhile collision
/// counts in real workloads.
pub fn generate_binary_trace(
    operations: &[BinaryOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = operations.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, op) in operations.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;
        let res = op.res();
        for byte_idx in 0..8 {
            data[base + cols::LHS[byte_idx]] = FE::from((op.lhs >> (byte_idx * 8)) & 0xFF);
            data[base + cols::RHS[byte_idx]] = FE::from((op.rhs >> (byte_idx * 8)) & 0xFF);
            data[base + cols::RES[byte_idx]] = FE::from((res >> (byte_idx * 8)) & 0xFF);
        }
        match op.op {
            BinaryOp::And => data[base + cols::IS_AND] = FE::one(),
            BinaryOp::Or => data[base + cols::IS_OR] = FE::one(),
            BinaryOp::Xor => data[base + cols::IS_XOR] = FE::one(),
        }
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Constraints
// =========================================================================

/// Pairwise mutex constraint: `col_a * col_b = 0`. Combined with the
/// IsBit constraints on each operand it forces at-most-one of the two
/// to be 1 on every row. Degree 2.
struct OpSelectorMutex {
    col_a: usize,
    col_b: usize,
    constraint_idx: usize,
}

impl OpSelectorMutex {
    fn new(col_a: usize, col_b: usize, constraint_idx: usize) -> Self {
        Self {
            col_a,
            col_b,
            constraint_idx,
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for OpSelectorMutex {
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
        let a = step.get_main_evaluation_element(0, self.col_a).clone();
        let b = step.get_main_evaluation_element(0, self.col_b).clone();
        a * b
    }
}

/// Returns the Binary transition constraints.
///
/// - 3 unconditional IsBit constraints (one per op selector), degree 2.
/// - 3 pairwise mutex constraints `IS_X * IS_Y = 0`, degree 2 — these
///   close the `lhs == rhs` soundness gap where bus balance alone
///   cannot reject a row with two selectors set simultaneously
///   (`a & a == a | a == a` allows both byte buses to absorb).
///
/// `lhs op rhs = res` correctness is enforced at the bus level: every
/// active row sends 8 per-byte `AND_BYTE`/`OR_BYTE`/`XOR_BYTE` lookups
/// to BITWISE, which only contains rows with `(X, Y, X op Y)` for valid
/// bytes. A malformed row's bus message has no matching BITWISE row, so
/// LogUp imbalance rejects the proof.
pub fn binary_constraints()
-> Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>> {
    vec![
        IsBitConstraint::unconditional(cols::IS_AND, 0).boxed(),
        IsBitConstraint::unconditional(cols::IS_OR, 1).boxed(),
        IsBitConstraint::unconditional(cols::IS_XOR, 2).boxed(),
        OpSelectorMutex::new(cols::IS_AND, cols::IS_OR, 3).boxed(),
        OpSelectorMutex::new(cols::IS_AND, cols::IS_XOR, 4).boxed(),
        OpSelectorMutex::new(cols::IS_OR, cols::IS_XOR, 5).boxed(),
    ]
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Returns the Binary bus interactions (25 total).
///
/// **Senders (24)**: per-byte `AND_BYTE`/`OR_BYTE`/`XOR_BYTE` to BITWISE
/// (8 each), gated by the matching op selector. These take over from
/// CPU's deleted per-byte senders — the BITWISE multiplicity columns
/// see the same overall traffic.
///
/// **Receiver (1)**: `BusId::Binary` with multiplicity
/// `Sum3(IS_AND, IS_OR, IS_XOR)` (= 1 on active rows, 0 on padding).
/// Bus payload `(op, lhs, rhs, res)` mirrors CPU's sender:
/// - `op` is `0·IS_AND + 1·IS_OR + 2·IS_XOR` (linear, matches CPU
///   sender's `0·AND + 1·OR + 2·XOR`).
/// - `lhs`, `rhs`, `res` are sent via `Packing::DWordBL` on the
///   8-byte storage cols, producing `[lo32, hi32]` matching CPU.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(25);

    for i in 0..8 {
        interactions.push(per_byte_sender(BusId::AndByte, cols::IS_AND, i));
    }
    for i in 0..8 {
        interactions.push(per_byte_sender(BusId::OrByte, cols::IS_OR, i));
    }
    for i in 0..8 {
        interactions.push(per_byte_sender(BusId::XorByte, cols::IS_XOR, i));
    }

    interactions.push(BusInteraction::receiver(
        BusId::Binary,
        Multiplicity::Sum3(cols::IS_AND, cols::IS_OR, cols::IS_XOR),
        vec![
            // op encoding (matches CPU sender): 0·IS_AND + 1·IS_OR + 2·IS_XOR.
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::IS_OR,
                },
                LinearTerm::Column {
                    coefficient: 2,
                    column: cols::IS_XOR,
                },
            ]),
            BusValue::Packed {
                start_column: cols::LHS_0,
                packing: Packing::DWordBL,
            },
            BusValue::Packed {
                start_column: cols::RHS_0,
                packing: Packing::DWordBL,
            },
            BusValue::Packed {
                start_column: cols::RES_0,
                packing: Packing::DWordBL,
            },
        ],
    ));

    interactions
}

fn per_byte_sender(bus: BusId, mu_col: usize, byte_idx: usize) -> BusInteraction {
    BusInteraction::sender(
        bus,
        Multiplicity::Column(mu_col),
        vec![
            BusValue::Packed {
                start_column: cols::LHS[byte_idx],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RHS[byte_idx],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[byte_idx],
                packing: Packing::Direct,
            },
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_apply_matches_native() {
        assert_eq!(BinaryOp::And.apply(0xFF00, 0x0F0F), 0x0F00);
        assert_eq!(BinaryOp::Or.apply(0xFF00, 0x0F0F), 0xFF0F);
        assert_eq!(BinaryOp::Xor.apply(0xFF00, 0x0F0F), 0xF00F);
    }

    #[test]
    fn trace_layout_matches_op_bytes() {
        let op = BinaryOperation::new(BinaryOp::And, 0xAABBCCDDEEFF0011, 0xFF);
        let trace = generate_binary_trace(&[op]);
        let row = trace.main_table.get_row(0);
        assert_eq!(*row[cols::LHS_0].value(), 0x11);
        assert_eq!(*row[cols::LHS_7].value(), 0xAA);
        assert_eq!(*row[cols::RHS_0].value(), 0xFF);
        assert_eq!(*row[cols::RHS_1].value(), 0);
        assert_eq!(*row[cols::RES_0].value(), 0x11);
        assert_eq!(*row[cols::IS_AND].value(), 1);
        assert_eq!(*row[cols::IS_OR].value(), 0);
        assert_eq!(*row[cols::IS_XOR].value(), 0);
    }

    #[test]
    fn empty_ops_padded_to_four_rows() {
        let trace = generate_binary_trace(&[]);
        assert_eq!(trace.main_table.height, 4);
    }

    #[test]
    fn bus_encoding_uses_zero_one_two() {
        assert_eq!(BinaryOp::And.bus_encoding(), 0);
        assert_eq!(BinaryOp::Or.bus_encoding(), 1);
        assert_eq!(BinaryOp::Xor.bus_encoding(), 2);
    }
}
