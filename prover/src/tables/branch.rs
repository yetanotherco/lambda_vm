//! BRANCH table for computing next program counter for branch/jump instructions.
//!
//! This table computes the target address for branch and jump instructions,
//! supporting both PC-relative branches and register-based JALR jumps.
//!
//! ## Inputs
//! - `pc`: DWordWL (64-bit as [Word, Word]) - current program counter
//! - `offset`: DWordWL (64-bit as [Word, Word]) - already sign-extended from CPU's imm
//! - `register`: DWordWL (64-bit) - base address for JALR
//! - `JALR`: Bit - selector between pc (0) and register (1) as base
//!
//! ## Output
//! - `next_pc_high`: Half[3] - upper 48 bits split into 3 halfwords
//! - `next_pc_low`: Byte[2] - lower 16 bits split into 2 bytes (LSB masked)
//!
//! ## Auxiliary
//! - `unmasked_low_byte`: Byte - low byte before LSB masking (for ADD constraint)
//!
//! ## Virtual (embedded in constraints)
//! - `next_pc_unmasked`: The raw addition result before masking LSB
//! - `carry[0]`, `carry[1]`: Carries from 64-bit addition
//!
//! ## Bus Interactions
//! - Sender: ARE_BYTES (×1 for `[next_pc_low[1], 0]`, spec template `IS_BYTE<next_pc_low[1]>`)
//! - Sender: BYTE_ALU[AND] (×1 for masking LSB)
//! - Sender: IS_HALFWORD (×3 for next_pc_high[0..3])
//! - Receiver: BRANCH (provides branch targets to CPU)

use stark::constraints::builder::{ConstraintBuilder, ConstraintMeta, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use std::collections::HashMap;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, SHIFT_16, VmTable, alu_op};

// =========================================================================
// Column indices for BRANCH table
// =========================================================================

/// Column definitions for the BRANCH table.
pub mod cols {
    // Input columns: pc (DWordWL = 2 Words)
    /// pc[0]: Word (bits 0-31)
    pub const PC_0: usize = 0;
    /// pc[1]: Word (bits 32-63)
    pub const PC_1: usize = 1;

    // Input columns: offset (DWordWL = 2 Words, sign-extended from CPU's imm)
    // NOTE: This offset representation differs from the one used in the spec.
    /// offset[0]: Word (bits 0-31)
    pub const OFFSET_0: usize = 2;
    /// offset[1]: Word (bits 32-63)
    pub const OFFSET_1: usize = 3;

    // Input columns: register (DWordWL = 2 Words)
    /// register[0]: Word (bits 0-31)
    pub const REGISTER_0: usize = 4;
    /// register[1]: Word (bits 32-63)
    pub const REGISTER_1: usize = 5;

    // Input column: JALR flag
    /// JALR: Bit (1 = use register as base, 0 = use pc as base)
    pub const JALR: usize = 6;

    // Output columns: next_pc_high (Half[3])
    /// next_pc_high[0]: Half (bits 16-31 of next_pc)
    pub const NEXT_PC_HIGH_0: usize = 7;
    /// next_pc_high[1]: Half (bits 32-47 of next_pc)
    pub const NEXT_PC_HIGH_1: usize = 8;
    /// next_pc_high[2]: Half (bits 48-63 of next_pc)
    pub const NEXT_PC_HIGH_2: usize = 9;

    // Output columns: next_pc_low (Byte[2])
    /// next_pc_low[0]: Byte (bits 0-7 of next_pc, with LSB masked to 0)
    pub const NEXT_PC_LOW_0: usize = 10;
    /// next_pc_low[1]: Byte (bits 8-15 of next_pc)
    pub const NEXT_PC_LOW_1: usize = 11;

    // Auxiliary columns
    /// unmasked_low_byte: Byte (bits 0-7 before LSB masking)
    pub const UNMASKED_LOW_BYTE: usize = 12;

    // Multiplicity column
    /// μ: multiplicity for bus interactions
    pub const MU: usize = 13;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 14;
}

// =========================================================================
// Constants
// =========================================================================

/// 2^8 for byte combining
const SHIFT_8: u64 = 1 << 8;

/// Mask value 254 = 0xFE (clears LSB)
const MASK_254: u64 = 254;

// =========================================================================
// Trace generation
// =========================================================================

/// A single BRANCH operation to be added to the trace.
///
/// Derives Hash and Eq so it can be used as a HashMap key for deduplication.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BranchOperation {
    /// Current program counter (64-bit)
    pub pc: u64,
    /// Offset from base address (64-bit DWordWL, already sign-extended from CPU's imm)
    pub offset: u64,
    /// Register value for JALR (64-bit)
    pub register: u64,
    /// Whether this is a JALR instruction (uses register instead of pc)
    pub jalr: bool,
}

impl BranchOperation {
    /// Create a new BRANCH operation.
    pub fn new(pc: u64, offset: u64, register: u64, jalr: bool) -> Self {
        Self {
            pc,
            offset,
            register,
            jalr,
        }
    }

    /// Compute the next program counter.
    ///
    /// For regular branches: next_pc = pc + offset
    /// For JALR: next_pc = (register + offset) & !1
    ///
    /// The LSB is always masked to 0 per RISC-V ISA.
    /// Note: offset is already sign-extended to 64-bit by the CPU.
    pub fn compute_next_pc(&self) -> u64 {
        let base = if self.jalr { self.register } else { self.pc };
        let unmasked = base.wrapping_add(self.offset);
        // Mask LSB to 0 (RISC-V requirement)
        unmasked & !1u64
    }

    /// Compute the unmasked next_pc (before LSB masking).
    pub fn compute_next_pc_unmasked(&self) -> u64 {
        let base = if self.jalr { self.register } else { self.pc };
        base.wrapping_add(self.offset)
    }
}

/// Generates the BRANCH trace table from a list of operations.
///
/// Duplicate operations (same pc, offset, register, jalr) are merged into a single row
/// with their multiplicities summed. The table is then padded to the next power of 2.
pub fn generate_branch_trace(
    operations: &[BranchOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    // Deduplicate operations: (pc, offset, register, jalr) -> multiplicity
    let mut op_map: HashMap<BranchOperation, u64> = HashMap::new();
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
        // Compute next_pc
        let next_pc_unmasked = op.compute_next_pc_unmasked();
        let next_pc = op.compute_next_pc();

        // Extract next_pc components
        // next_pc_low[0]: bits 0-7 (masked)
        // next_pc_low[1]: bits 8-15
        // next_pc_high[0]: bits 16-31
        // next_pc_high[1]: bits 32-47
        // next_pc_high[2]: bits 48-63
        let unmasked_low_byte = (next_pc_unmasked & 0xFF) as u8;
        let next_pc_low_0 = (next_pc & 0xFF) as u8; // = unmasked_low_byte & 0xFE
        let next_pc_low_1 = ((next_pc >> 8) & 0xFF) as u8;
        let next_pc_high_0 = ((next_pc >> 16) & 0xFFFF) as u16;
        let next_pc_high_1 = ((next_pc >> 32) & 0xFFFF) as u16;
        let next_pc_high_2 = ((next_pc >> 48) & 0xFFFF) as u16;

        // Store columns
        table.set_dword_wl(row_idx, cols::PC_0, op.pc);
        table.set_dword_wl(row_idx, cols::OFFSET_0, op.offset);
        table.set_dword_wl(row_idx, cols::REGISTER_0, op.register);
        table.set_bool(row_idx, cols::JALR, op.jalr);
        table.set_halves(
            row_idx,
            cols::NEXT_PC_HIGH_0,
            &[next_pc_high_0, next_pc_high_1, next_pc_high_2],
        );
        table.set_bytes(
            row_idx,
            cols::NEXT_PC_LOW_0,
            &[next_pc_low_0, next_pc_low_1],
        );
        table.set_byte(row_idx, cols::UNMASKED_LOW_BYTE, unmasked_low_byte);
        table.set_u64(row_idx, cols::MU, *multiplicity);
    }

    trace
}

// =========================================================================
// Bus Interactions
// =========================================================================

/// Creates all bus interactions for the BRANCH table.
///
/// The BRANCH table:
/// - **Sends** ARE_BYTES lookup for next_pc_low[1] range check (Y=0)
/// - **Sends** BYTE_ALU[AND] lookup for LSB masking
///   (next_pc_low[0] = unmasked_low_byte & 254)
/// - **Sends** IS_HALFWORD lookups for next_pc_high[0..3] range checks
/// - **Receives** BRANCH lookups from CPU table
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // ARE_BYTES[next_pc_low[1], 0] - range check bits 8-15
        BusInteraction::sender(
            BusId::AreBytes,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: cols::NEXT_PC_LOW_1,
                    packing: Packing::Direct,
                },
                BusValue::constant(0),
            ],
        ),
        // BYTE_ALU[next_pc_low[0]; AND, unmasked_low_byte, 254]
        // Verifies: next_pc_low[0] = unmasked_low_byte & 0xFE
        BusInteraction::sender(
            BusId::ByteAlu,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::constant(alu_op::AND as u64),
                BusValue::Packed {
                    start_column: cols::UNMASKED_LOW_BYTE,
                    packing: Packing::Direct,
                },
                BusValue::constant(MASK_254),
                BusValue::Packed {
                    start_column: cols::NEXT_PC_LOW_0,
                    packing: Packing::Direct,
                },
            ],
        ),
        // IS_HALFWORD[next_pc_high[0]]
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::NEXT_PC_HIGH_0,
                packing: Packing::Direct,
            }],
        ),
        // IS_HALFWORD[next_pc_high[1]]
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::NEXT_PC_HIGH_1,
                packing: Packing::Direct,
            }],
        ),
        // IS_HALFWORD[next_pc_high[2]]
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::NEXT_PC_HIGH_2,
                packing: Packing::Direct,
            }],
        ),
        // BRANCH[next_pc; pc, offset, register, JALR] (receiver)
        // Signature: [next_pc (DWordWL), pc (DWordWL), offset (DWordWL), register (DWordWL), JALR (Bit)]
        BusInteraction::receiver(
            BusId::Branch,
            Multiplicity::Column(cols::MU),
            vec![
                // next_pc as DWordWL (2 words)
                // next_pc[0] = 2^16 * next_pc_high[0] + 2^8 * next_pc_low[1] + next_pc_low[0]
                // next_pc[1] = 2^16 * next_pc_high[2] + next_pc_high[1]
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::NEXT_PC_LOW_0,
                    },
                    LinearTerm::Column {
                        coefficient: SHIFT_8 as i64,
                        column: cols::NEXT_PC_LOW_1,
                    },
                    LinearTerm::Column {
                        coefficient: SHIFT_16 as i64,
                        column: cols::NEXT_PC_HIGH_0,
                    },
                ]),
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::NEXT_PC_HIGH_1,
                    },
                    LinearTerm::Column {
                        coefficient: SHIFT_16 as i64,
                        column: cols::NEXT_PC_HIGH_2,
                    },
                ]),
                // pc as DWordWL
                BusValue::Packed {
                    start_column: cols::PC_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::PC_1,
                    packing: Packing::Direct,
                },
                // offset as DWordWL
                BusValue::Packed {
                    start_column: cols::OFFSET_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::OFFSET_1,
                    packing: Packing::Direct,
                },
                // register as DWordWL
                BusValue::Packed {
                    start_column: cols::REGISTER_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::REGISTER_1,
                    packing: Packing::Direct,
                },
                // JALR flag
                BusValue::Packed {
                    start_column: cols::JALR,
                    packing: Packing::Direct,
                },
            ],
        ),
    ]
}

// =========================================================================
// Single-source constraint set (ConstraintBuilder front-end)
// =========================================================================

/// `(unmasked_0, unmasked_1)` — the next-pc value repacked into two words,
/// as builder expressions (twin of [`BranchConstraint::compute_next_pc_unmasked`]):
///
/// ```text
/// unmasked_0 = unmasked_low_byte + next_pc_low_1·2⁸ + next_pc_high_0·2¹⁶
/// unmasked_1 = next_pc_high_1 + next_pc_high_2·2¹⁶
/// ```
fn next_pc_unmasked_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
) -> (B::Expr, B::Expr) {
    let shift_8 = b.const_base(SHIFT_8);
    let shift_16 = b.const_base(SHIFT_16);
    let unmasked_0 = b.main(0, cols::UNMASKED_LOW_BYTE)
        + b.main(0, cols::NEXT_PC_LOW_1) * shift_8
        + b.main(0, cols::NEXT_PC_HIGH_0) * shift_16.clone();
    let unmasked_1 = b.main(0, cols::NEXT_PC_HIGH_1) + b.main(0, cols::NEXT_PC_HIGH_2) * shift_16;
    (unmasked_0, unmasked_1)
}

/// `carry_0 = (base_0 + offset_0 − unmasked_0)·2⁻³²` (twin of
/// [`BranchConstraint::compute_carry_0_for`]). Takes `unmasked_0` from
/// [`next_pc_unmasked_expr`] so the body computes the repack once per row
/// (it is shared by the pc and register paths).
fn carry_0_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    base_col_0: usize,
    unmasked_0: B::Expr,
) -> B::Expr {
    let inv_2_32 = b.const_base(crate::constraints::templates::INV_SHIFT_32);
    (b.main(0, base_col_0) + b.main(0, cols::OFFSET_0) - unmasked_0) * inv_2_32
}

/// `carry_1 = (base_1 + offset_1 + carry_0 − unmasked_1)·2⁻³²` (twin of
/// [`BranchConstraint::compute_carry_1_for`]). Takes the path's `carry_0`
/// and the shared `unmasked_1` so neither is recomputed.
fn carry_1_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    base_col_1: usize,
    carry_0: B::Expr,
    unmasked_1: B::Expr,
) -> B::Expr {
    let inv_2_32 = b.const_base(crate::constraints::templates::INV_SHIFT_32);
    (b.main(0, base_col_1) + b.main(0, cols::OFFSET_1) + carry_0 - unmasked_1) * inv_2_32
}

/// The BRANCH table's transition constraints as a single [`ConstraintSet`],
/// mirroring `branch_constraints` index-for-index (5 constraints):
/// - idx 0: `(1 − JALR)·carry_0·(1 − carry_0)` on the pc path (degree 3);
/// - idx 1: `(1 − JALR)·carry_1·(1 − carry_1)` on the pc path (degree 3);
/// - idx 2: `JALR·carry_0·(1 − carry_0)` on the register path (degree 3);
/// - idx 3: `JALR·carry_1·(1 − carry_1)` on the register path (degree 3);
/// - idx 4: `JALR·(1 − JALR)` (degree 2).
pub struct BranchConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for BranchConstraints {
    fn meta(&self) -> Vec<ConstraintMeta> {
        vec![
            ConstraintMeta::base(0, 3), // PcCarry0IsBit
            ConstraintMeta::base(1, 3), // PcCarry1IsBit
            ConstraintMeta::base(2, 3), // RegCarry0IsBit
            ConstraintMeta::base(3, 3), // RegCarry1IsBit
            ConstraintMeta::base(4, 2), // JalrIsBit
        ]
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // The unmasked next-pc repack and each path's carry_0 are shared
        // across the four carry constraints — computed once per row.
        let (unmasked_0, unmasked_1) = next_pc_unmasked_expr(b);
        let pc_c0 = carry_0_expr(b, cols::PC_0, unmasked_0.clone());
        let pc_c1 = carry_1_expr(b, cols::PC_1, pc_c0.clone(), unmasked_1.clone());
        let reg_c0 = carry_0_expr(b, cols::REGISTER_0, unmasked_0);
        let reg_c1 = carry_1_expr(b, cols::REGISTER_1, reg_c0.clone(), unmasked_1);

        // idx 0: (1 - JALR) * carry_0(pc) * (1 - carry_0)
        let one = b.one();
        let cond = one - b.main(0, cols::JALR);
        let one = b.one();
        b.emit_base(0, cond * pc_c0.clone() * (one - pc_c0));

        // idx 1: (1 - JALR) * carry_1(pc) * (1 - carry_1)
        let one = b.one();
        let cond = one - b.main(0, cols::JALR);
        let one = b.one();
        b.emit_base(1, cond * pc_c1.clone() * (one - pc_c1));

        // idx 2: JALR * carry_0(register) * (1 - carry_0)
        let cond = b.main(0, cols::JALR);
        let one = b.one();
        b.emit_base(2, cond * reg_c0.clone() * (one - reg_c0));

        // idx 3: JALR * carry_1(register) * (1 - carry_1)
        let cond = b.main(0, cols::JALR);
        let one = b.one();
        b.emit_base(3, cond * reg_c1.clone() * (one - reg_c1));

        // idx 4: JALR * (1 - JALR)
        let one = b.one();
        let jalr = b.main(0, cols::JALR);
        b.emit_base(4, jalr.clone() * (one - jalr));
    }
}

// =========================================================================
// Helper functions for computing carries (used by trace generator and tests)
// =========================================================================

/// Compute virtual carry[0] and carry[1] for the branch addition.
///
/// This computes the carries for: base + offset = next_pc_unmasked
/// where offset is already sign-extended to 64 bits as DWordWL.
///
/// Returns (carry_0, carry_1) where both should be 0 or 1.
pub fn compute_carries(base: u64, offset: u64) -> (u64, u64) {
    // Split into DWordWL format
    let base_lo = base & 0xFFFF_FFFF;
    let base_hi = base >> 32;

    let offset_lo = offset & 0xFFFF_FFFF;
    let offset_hi = offset >> 32;

    // carry[0] = (base_lo + offset_lo) >> 32
    let carry_0 = (base_lo + offset_lo) >> 32;

    // carry[1] = (base_hi + offset_hi + carry_0) >> 32
    let carry_1 = (base_hi + offset_hi + carry_0) >> 32;

    (carry_0, carry_1)
}
