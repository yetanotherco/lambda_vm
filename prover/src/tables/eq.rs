//! EQ (equality) comparison table.
//!
//! Computes `res = (a == b) XOR invert` for 64-bit `a`, `b`. Used by `BEQ`
//! (`invert = 0`) and `BNE` (`invert = 1`); the CPU dispatches to it on the
//! unified `ALU` bus with `alu_flags = opsel(EQ) + 64*invert`.
//!
//! Spec: `spec/src/eq.toml`.
//!
//! ## Columns
//! - `a`: DWordWL (2 words)         — first input
//! - `b`: DWordWL (2 words)         — second input
//! - `invert`: Bit                  — invert the result
//! - `res`: Bit                     — output, `(a == b) XOR invert`
//! - `diff`: DWordHL (4 halves)     — `a - b` (aux)
//! - `eq`: Bit                      — `a == b` (aux)
//! - `μ`: multiplicity
//!
//! ## Method
//! `diff = a - b` is enforced via the `ADD` template (`b + diff = a`), its
//! halves range-checked via `IS_HALF`. Then `eq = ZERO[Σ diff[i]]` (the sum of
//! four range-checked halves is `0` iff `diff == 0` iff `a == b`), and
//! `res = eq XOR invert`.

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

use super::types::{BusId, GoldilocksExtension, GoldilocksField, VmTable, alu_op};
use crate::constraints::templates::{AddOperand, emit_add_pair, emit_is_bit};

// =========================================================================
// Column indices for EQ table
// =========================================================================

/// Column definitions for the EQ table.
pub mod cols {
    // Input: a (DWordWL = 2 words)
    pub const A_0: usize = 0;
    pub const A_1: usize = 1;
    // Input: b (DWordWL = 2 words)
    pub const B_0: usize = 2;
    pub const B_1: usize = 3;
    /// invert: Bit
    pub const INVERT: usize = 4;
    /// res: Bit (output) = (a == b) XOR invert
    pub const RES: usize = 5;
    // Auxiliary: diff (DWordHL = 4 halves) = a - b
    pub const DIFF_0: usize = 6;
    pub const DIFF_1: usize = 7;
    pub const DIFF_2: usize = 8;
    pub const DIFF_3: usize = 9;
    /// eq: Bit (auxiliary) = (a == b)
    pub const EQ: usize = 10;
    /// μ: multiplicity
    pub const MU: usize = 11;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 12;
}

// =========================================================================
// Trace generation
// =========================================================================

/// A single EQ operation.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct EqOperation {
    /// First operand (64-bit)
    pub a: u64,
    /// Second operand (64-bit)
    pub b: u64,
    /// Whether to invert the equality result
    pub invert: bool,
}

impl EqOperation {
    /// Create a new EQ operation.
    pub fn new(a: u64, b: u64, invert: bool) -> Self {
        Self { a, b, invert }
    }

    /// `a == b` (before inversion).
    pub fn compute_eq(&self) -> bool {
        self.a == self.b
    }

    /// The output: `(a == b) XOR invert`.
    pub fn compute_res(&self) -> bool {
        self.compute_eq() ^ self.invert
    }

    /// The BITWISE lookups this op sends (4× `IS_HALF` on the `diff` halves and
    /// one `ZERO` on their sum), for the BITWISE table's multiplicity bookkeeping.
    pub fn collect_bitwise_ops(&self) -> Vec<super::bitwise::BitwiseOperation> {
        use super::bitwise::{BitwiseOperation, BitwiseOperationType};
        let diff = self.a.wrapping_sub(self.b);
        let mut ops = Vec::with_capacity(5);
        let mut sum = 0u32;
        for i in 0..4 {
            let half = ((diff >> (i * 16)) & 0xFFFF) as u32;
            sum += half;
            ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (half & 0xFF) as u8,
                (half >> 8) as u8,
            ));
        }
        ops.push(BitwiseOperation::zero(sum));
        ops
    }
}

/// Generates the EQ trace from a list of operations.
///
/// Duplicate operations are merged into a single row with summed multiplicities,
/// then padded to the next power of two (minimum 4).
pub fn generate_eq_trace(
    operations: &[EqOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    use std::collections::HashMap;

    let mut op_map: HashMap<EqOperation, u64> = HashMap::new();
    for op in operations {
        *op_map.entry(op.clone()).or_insert(0) += 1;
    }

    let unique_ops: Vec<_> = op_map.into_iter().collect();
    let num_rows = unique_ops.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, (op, multiplicity)) in unique_ops.iter().enumerate() {
        // a, b as DWordWL (2 words each)
        table.set_dword_wl(row_idx, cols::A_0, op.a);
        table.set_dword_wl(row_idx, cols::B_0, op.b);

        table.set_bool(row_idx, cols::INVERT, op.invert);
        table.set_bool(row_idx, cols::RES, op.compute_res());

        // diff = a - b (wrapping) as DWordHL (4 halves)
        let diff = op.a.wrapping_sub(op.b);
        table.set_dword_hl(row_idx, cols::DIFF_0, diff);

        table.set_bool(row_idx, cols::EQ, op.compute_eq());
        table.set_u64(row_idx, cols::MU, *multiplicity);
    }

    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

/// All bus interactions for the EQ table:
/// - **Sends** `IS_HALF[diff[i]]` (×4) to range-check the difference halves.
/// - **Sends** `ZERO[Σ diff[i]] -> eq`.
/// - **Receives** `ALU[a, b, opsel(EQ) + 64*invert] -> res`.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(6);

    // IS_HALF[diff[i]] for i in 0..3
    for diff_col in [cols::DIFF_0, cols::DIFF_1, cols::DIFF_2, cols::DIFF_3] {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: diff_col,
                packing: Packing::Direct,
            }],
        ));
    }

    // ZERO[diff[0] + diff[1] + diff[2] + diff[3]] -> eq
    // The sum of four range-checked halves is in [0, 2^18) < 2^20, so it is 0
    // iff diff == 0 iff a == b. Matches the BITWISE ZERO lookup domain.
    interactions.push(BusInteraction::sender(
        BusId::Zero,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::DIFF_0,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::DIFF_1,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::DIFF_2,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::DIFF_3,
                },
            ]),
            BusValue::Packed {
                start_column: cols::EQ,
                packing: Packing::Direct,
            },
        ],
    ));

    // ALU[a, b, opsel(EQ) + 64*invert] -> res  (receiver).
    // The ALU output is DWordWL (2 elements); for a comparison it is [res, 0]
    // (the bit in the low word, 0 in the high word).
    interactions.push(BusInteraction::receiver(
        BusId::Alu,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::Packed {
                start_column: cols::A_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::B_0,
                packing: Packing::DWordWL,
            },
            BusValue::linear(vec![
                LinearTerm::Constant(alu_op::EQ as i64),
                LinearTerm::Column {
                    coefficient: 64,
                    column: cols::INVERT,
                },
            ]),
            // out = [res, 0] (DWordWL)
            BusValue::Packed {
                start_column: cols::RES,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
        ],
    ));

    interactions
}

// =========================================================================
// Single-source constraint set (ConstraintBuilder front-end)
// =========================================================================

/// The EQ table's transition constraints as a single [`ConstraintSet`]:
/// - idx 0,1: `ADD` pair `b + diff = a` (unconditional);
/// - idx 2:   `IS_BIT(invert)` (unconditional);
/// - idx 3:   `res = eq XOR invert`.
pub struct EqConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for EqConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // diff = a - b, encoded as b + diff = a (unconditional).
        emit_add_pair(
            b,
            0,
            &[],
            &AddOperand::dword(cols::B_0),
            &AddOperand::from_dword_hl(cols::DIFF_0),
            &AddOperand::dword(cols::A_0),
        );
        // IS_BIT(invert)
        emit_is_bit(b, 2, cols::INVERT, None);
        // res = eq XOR invert = eq + invert - 2*eq*invert
        let res = b.main(0, cols::RES);
        let eq = b.main(0, cols::EQ);
        let invert = b.main(0, cols::INVERT);
        let two = b.const_base(2);
        b.emit_base(3, res - (eq.clone() + invert.clone() - two * eq * invert));
    }
}
