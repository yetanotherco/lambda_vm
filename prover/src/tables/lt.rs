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

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use std::collections::HashMap;

use super::types::{BusId, GoldilocksExtension, GoldilocksField, SHIFT_16, VmTable, alu_op};

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
    // Deduplicate operations: (lhs, rhs, signed) -> multiplicity
    let mut op_map: HashMap<LtOperation, u64> = HashMap::new();
    for op in operations {
        *op_map.entry(op.clone()).or_insert(0) += 1;
    }

    // Canonical row order: HashMap iteration order is per-process random, so
    // sort to keep the committed trace deterministic across runs.
    let mut unique_ops: Vec<_> = op_map.into_iter().collect();
    unique_ops.sort_unstable_by_key(|(op, _)| (op.lhs, op.rhs, op.signed, op.invert));
    let num_rows = unique_ops.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
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
// Single-body constraint set (ConstraintSet front-end)
// =========================================================================
//
// One body against the generic `ConstraintBuilder` serves the compiled prover
// folder, the verifier folder and IR capture. Constraint indices 0..6.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

/// LT table constraints as a single-source [`ConstraintSet`]. No column
/// configuration is needed (the LT layout is fixed via `cols`).
pub struct LtConstraints;

impl LtConstraints {
    /// `cast(lhs_sub_rhs, DWordWL)[0] = sub_0 + 2^16 · sub_1`.
    fn carry_0<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(b: &B) -> B::Expr {
        let lhs_0 = b.main(0, cols::LHS_0);
        let rhs_0 = b.main(0, cols::RHS_0);
        let sub_0 = b.main(0, cols::LHS_SUB_RHS_0);
        let sub_1 = b.main(0, cols::LHS_SUB_RHS_1);
        let shift_16 = b.const_base(SHIFT_16);
        let sub_lo = sub_0 + sub_1 * shift_16;
        // carry[0] = (rhs[0] + sub_lo - lhs[0]) / 2^32
        let inv_2_32 = b.const_base(crate::constraints::templates::INV_SHIFT_32);
        (rhs_0 + sub_lo - lhs_0) * inv_2_32
    }

    /// carry[1] = (rhs_hi + sub_hi + carry_0 - lhs_hi) / 2^32.
    ///
    /// Known redundancy: this rebuilds [`Self::carry_0`], which idx 0 also
    /// computes. Threading the value through was tried and showed no
    /// measurable speedup (ABBA), so the helpers stay self-contained.
    fn carry_1<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(b: &B) -> B::Expr {
        let lhs_1 = b.main(0, cols::LHS_1);
        let lhs_2 = b.main(0, cols::LHS_2);
        let rhs_1 = b.main(0, cols::RHS_1);
        let rhs_2 = b.main(0, cols::RHS_2);
        let sub_2 = b.main(0, cols::LHS_SUB_RHS_2);
        let sub_3 = b.main(0, cols::LHS_SUB_RHS_3);
        let shift_16 = b.const_base(SHIFT_16);
        // cast(lhs, DWordWL)[1] = lhs[1] + 2^16 * lhs[2]
        let lhs_hi = lhs_1 + lhs_2 * shift_16.clone();
        // cast(rhs, DWordWL)[1] = rhs[1] + 2^16 * rhs[2]
        let rhs_hi = rhs_1 + rhs_2 * shift_16.clone();
        // cast(lhs_sub_rhs, DWordWL)[1] = sub_2 + 2^16 * sub_3
        let sub_hi = sub_2 + sub_3 * shift_16;
        let carry_0 = Self::carry_0(b);
        let inv_2_32 = b.const_base(crate::constraints::templates::INV_SHIFT_32);
        (rhs_hi + sub_hi + carry_0 - lhs_hi) * inv_2_32
    }
}

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for LtConstraints {
    // The LT formula (idx 2) is degree 3; the rest are degree 2.
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0: IS_BIT<carry[0]>: carry[0] * (1 - carry[0])
        let c0 = Self::carry_0(b);
        let one = b.one();
        b.emit_base(0, c0.clone() * (one - c0));

        // idx 1: IS_BIT<carry[1]>: carry[1] * (1 - carry[1])
        let c1 = Self::carry_1(b);
        let one = b.one();
        b.emit_base(1, c1.clone() * (one - c1));

        // idx 2: LT formula: lt - (signed*signed_lt + (1-signed)*unsigned_lt)
        // signed_lt = A*(1-B) + A*C + (1-B)*C; unsigned_lt = C = carry[1]
        let lt = b.main(0, cols::LT);
        let signed = b.main(0, cols::SIGNED);
        let a = b.main(0, cols::LHS_MSB);
        let bb = b.main(0, cols::RHS_MSB);
        let c = Self::carry_1(b);
        let unsigned_lt = c.clone();
        let one = b.one();
        let one_minus_b = one - bb;
        let signed_lt = a.clone() * one_minus_b.clone() + a * c.clone() + one_minus_b * c;
        let one = b.one();
        let expected_lt = signed.clone() * signed_lt + (one - signed) * unsigned_lt;
        b.emit_base(2, lt - expected_lt);

        // idx 3: out = lt XOR invert = lt + invert - 2*lt*invert
        let out = b.main(0, cols::OUT);
        let lt = b.main(0, cols::LT);
        let invert = b.main(0, cols::INVERT);
        let two = b.const_base(2);
        b.emit_base(3, out - (lt.clone() + invert.clone() - two * lt * invert));

        // idx 4: invert * (1 - invert)
        let invert = b.main(0, cols::INVERT);
        let one = b.one();
        b.emit_base(4, invert.clone() * (one - invert));

        // idx 5: signed * (1 - signed)
        let signed = b.main(0, cols::SIGNED);
        let one = b.one();
        b.emit_base(5, signed.clone() * (one - signed));
    }
}
