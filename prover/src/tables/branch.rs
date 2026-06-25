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

use crate::constraints::templates::is_bit_fold;

use super::types::{
    BusId, FE, FxHashMap, GoldilocksExtension, GoldilocksField, SHIFT_16, VmTable, alu_op,
};

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
    let mut op_map: FxHashMap<BranchOperation, u64> = FxHashMap::default();
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
// Constraints
// =========================================================================

/// BRANCH table conditional ADD constraint.
///
/// Implements two conditional ADD templates per the spec:
/// - `ADD(pc, offset) = next_pc_unmasked` conditioned on `(1 - JALR)`
/// - `ADD(register, offset) = next_pc_unmasked` conditioned on `JALR`
///
/// Each ADD template produces two carry IS_BIT constraints (carry_0 and carry_1),
/// for a total of 4 constraints, all at degree 3:
///   `cond * carry * (1 - carry) = 0`
///
/// The carries are computed from degree-1 operands (pc or register, not both),
/// so carry is degree 1 and the full constraint is degree 3.
pub struct BranchConstraint {
    /// Unique constraint identifier
    constraint_idx: usize,
    /// Which constraint to check
    kind: BranchConstraintKind,
}

/// Kind of BRANCH constraint.
///
/// Four variants: two carries × two conditions (pc-path and register-path).
#[derive(Debug, Clone, Copy)]
pub enum BranchConstraintKind {
    /// `(1 - JALR) * carry_0_pc * (1 - carry_0_pc) = 0`
    /// where carry_0_pc = (pc[0] + offset[0] - next_pc_unmasked[0]) / 2^32
    PcCarry0IsBit,
    /// `(1 - JALR) * carry_1_pc * (1 - carry_1_pc) = 0`
    /// where carry_1_pc = (pc[1] + offset[1] + carry_0_pc - next_pc_unmasked[1]) / 2^32
    PcCarry1IsBit,
    /// `IS_BIT<JALR>`: `JALR * (1 - JALR) = 0` (spec defense-in-depth assumption)
    JalrIsBit,
    /// `JALR * carry_0_reg * (1 - carry_0_reg) = 0`
    /// where carry_0_reg = (register[0] + offset[0] - next_pc_unmasked[0]) / 2^32
    RegCarry0IsBit,
    /// `JALR * carry_1_reg * (1 - carry_1_reg) = 0`
    /// where carry_1_reg = (register[1] + offset[1] + carry_0_reg - next_pc_unmasked[1]) / 2^32
    RegCarry1IsBit,
}

impl BranchConstraint {
    /// Creates a new BRANCH constraint.
    pub fn new(kind: BranchConstraintKind, constraint_idx: usize) -> Self {
        Self {
            constraint_idx,
            kind,
        }
    }

    /// Compute virtual next_pc_unmasked as DWordWL.
    ///
    /// next_pc_unmasked[0] = unmasked_low_byte + 2^8 * next_pc_low[1] + 2^16 * next_pc_high[0]
    /// next_pc_unmasked[1] = next_pc_high[1] + 2^16 * next_pc_high[2]
    fn compute_next_pc_unmasked<F, E>(step: &TableView<F, E>) -> (FieldElement<F>, FieldElement<F>)
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let unmasked_low_byte = step
            .get_main_evaluation_element(0, cols::UNMASKED_LOW_BYTE)
            .clone();
        let next_pc_low_1 = step
            .get_main_evaluation_element(0, cols::NEXT_PC_LOW_1)
            .clone();
        let next_pc_high_0 = step
            .get_main_evaluation_element(0, cols::NEXT_PC_HIGH_0)
            .clone();
        let next_pc_high_1 = step
            .get_main_evaluation_element(0, cols::NEXT_PC_HIGH_1)
            .clone();
        let next_pc_high_2 = step
            .get_main_evaluation_element(0, cols::NEXT_PC_HIGH_2)
            .clone();

        let shift_8 = FieldElement::<F>::from(SHIFT_8);
        let shift_16 = FieldElement::<F>::from(SHIFT_16);

        let unmasked_0 =
            &unmasked_low_byte + &next_pc_low_1 * &shift_8 + &next_pc_high_0 * &shift_16;
        let unmasked_1 = &next_pc_high_1 + &next_pc_high_2 * &shift_16;

        (unmasked_0, unmasked_1)
    }

    /// Compute carry_0 for a given base column pair.
    ///
    /// carry_0 = (base[0] + offset[0] - next_pc_unmasked[0]) / 2^32
    fn compute_carry_0_for<F, E>(base_col_0: usize, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let base_0 = step.get_main_evaluation_element(0, base_col_0).clone();
        let offset_0 = step.get_main_evaluation_element(0, cols::OFFSET_0).clone();
        let (unmasked_0, _) = Self::compute_next_pc_unmasked(step);

        let inv_2_32 = FieldElement::<F>::from(crate::constraints::templates::INV_SHIFT_32);
        (base_0 + offset_0 - unmasked_0) * inv_2_32
    }

    /// Compute carry_1 for a given base column pair.
    ///
    /// carry_1 = (base[1] + offset[1] + carry_0 - next_pc_unmasked[1]) / 2^32
    fn compute_carry_1_for<F, E>(
        base_col_0: usize,
        base_col_1: usize,
        step: &TableView<F, E>,
    ) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let base_1 = step.get_main_evaluation_element(0, base_col_1).clone();
        let offset_1 = step.get_main_evaluation_element(0, cols::OFFSET_1).clone();
        let carry_0 = Self::compute_carry_0_for(base_col_0, step);
        let (_, unmasked_1) = Self::compute_next_pc_unmasked(step);

        let inv_2_32 = FieldElement::<F>::from(crate::constraints::templates::INV_SHIFT_32);
        (base_1 + offset_1 + carry_0 - unmasked_1) * inv_2_32
    }

    /// Compute the constraint value: `cond * carry * (1 - carry)`.
    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let jalr = step.get_main_evaluation_element(0, cols::JALR).clone();
        let one = FieldElement::<F>::one();

        match self.kind {
            BranchConstraintKind::JalrIsBit => &jalr * (&one - &jalr),
            BranchConstraintKind::PcCarry0IsBit => {
                let cond = &one - &jalr;
                let c = Self::compute_carry_0_for(cols::PC_0, step);
                cond * &c * (&one - c)
            }
            BranchConstraintKind::PcCarry1IsBit => {
                let cond = &one - &jalr;
                let c = Self::compute_carry_1_for(cols::PC_0, cols::PC_1, step);
                cond * &c * (&one - c)
            }
            BranchConstraintKind::RegCarry0IsBit => {
                let cond = jalr;
                let c = Self::compute_carry_0_for(cols::REGISTER_0, step);
                cond * &c * (&one - c)
            }
            BranchConstraintKind::RegCarry1IsBit => {
                let cond = jalr;
                let c = Self::compute_carry_1_for(cols::REGISTER_0, cols::REGISTER_1, step);
                cond * &c * (&one - c)
            }
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for BranchConstraint {
    fn degree(&self) -> usize {
        match self.kind {
            // JALR * (1 - JALR) = degree 2
            BranchConstraintKind::JalrIsBit => 2,
            // cond (degree 1) * carry (degree 1) * (1 - carry) (degree 1) = degree 3
            _ => 3,
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

/// Creates all constraints for the BRANCH table.
///
/// Returns 5 constraints (two conditional ADD templates × 2 carries each, plus
/// the `IS_BIT<JALR>` defense-in-depth assumption):
/// - PcCarry0IsBit:  `(1 - JALR) * carry_0 * (1 - carry_0) = 0`  (pc path)
/// - PcCarry1IsBit:  `(1 - JALR) * carry_1 * (1 - carry_1) = 0`  (pc path)
/// - RegCarry0IsBit: `JALR * carry_0 * (1 - carry_0) = 0`        (register path)
/// - RegCarry1IsBit: `JALR * carry_1 * (1 - carry_1) = 0`        (register path)
/// - JalrIsBit:      `JALR * (1 - JALR) = 0`
pub fn branch_constraints(constraint_idx_start: usize) -> (Vec<BranchConstraint>, usize) {
    let mut idx = constraint_idx_start;
    let mut next = || {
        let i = idx;
        idx += 1;
        i
    };
    let constraints = vec![
        BranchConstraint::new(BranchConstraintKind::PcCarry0IsBit, next()),
        BranchConstraint::new(BranchConstraintKind::PcCarry1IsBit, next()),
        BranchConstraint::new(BranchConstraintKind::RegCarry0IsBit, next()),
        BranchConstraint::new(BranchConstraintKind::RegCarry1IsBit, next()),
        BranchConstraint::new(BranchConstraintKind::JalrIsBit, next()),
    ];
    (constraints, idx)
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

pub fn branch_domain_eval<CB: ConstraintBuilder>(cb: &mut CB) {
    let one = FieldElement::<CB::F>::one();
    let inv_2_32 = FieldElement::<CB::F>::from(crate::constraints::templates::INV_SHIFT_32);
    let shift_8 = FieldElement::<CB::F>::from(SHIFT_8);
    let shift_16 = FieldElement::<CB::F>::from(SHIFT_16);

    let unmasked_low_byte = cb.main(cols::UNMASKED_LOW_BYTE).clone();
    let next_pc_low_1 = cb.main(cols::NEXT_PC_LOW_1).clone();
    let next_pc_high_0 = cb.main(cols::NEXT_PC_HIGH_0).clone();
    let next_pc_high_1 = cb.main(cols::NEXT_PC_HIGH_1).clone();
    let next_pc_high_2 = cb.main(cols::NEXT_PC_HIGH_2).clone();
    let unmasked_0 = &unmasked_low_byte + &next_pc_low_1 * &shift_8 + &next_pc_high_0 * &shift_16;
    let unmasked_1 = &next_pc_high_1 + &next_pc_high_2 * &shift_16;

    let offset_0 = cb.main(cols::OFFSET_0).clone();
    let offset_1 = cb.main(cols::OFFSET_1).clone();
    let jalr = cb.main(cols::JALR).clone();
    let pc_0 = cb.main(cols::PC_0).clone();
    let pc_1 = cb.main(cols::PC_1).clone();
    let reg_0 = cb.main(cols::REGISTER_0).clone();
    let reg_1 = cb.main(cols::REGISTER_1).clone();

    let cond_pc = &one - &jalr;
    let cond_reg = jalr;

    // Constraints are folded in the exact order of `branch_constraints(0)`:
    //   0 PcCarry0IsBit, 1 PcCarry1IsBit, 2 RegCarry0IsBit, 3 RegCarry1IsBit, 4 JalrIsBit.

    // PC path (carry_0 shared by both pc constraints).
    let pc_c0 = (&pc_0 + &offset_0 - &unmasked_0) * &inv_2_32;
    let pc_c1 = (&pc_1 + &offset_1 + &pc_c0 - &unmasked_1) * &inv_2_32;
    cb.fold(&cond_pc * &pc_c0 * (&one - &pc_c0));
    cb.fold(&cond_pc * &pc_c1 * (&one - &pc_c1));

    // REGISTER path.
    let reg_c0 = (&reg_0 + &offset_0 - &unmasked_0) * &inv_2_32;
    let reg_c1 = (&reg_1 + &offset_1 + &reg_c0 - &unmasked_1) * &inv_2_32;
    cb.fold(&cond_reg * &reg_c0 * (&one - &reg_c0));
    cb.fold(&cond_reg * &reg_c1 * (&one - &reg_c1));

    // IS_BIT<JALR> defense-in-depth: `JALR * (1 - JALR) = 0` (unconditional IS_BIT).
    is_bit_fold(cb, None, cols::JALR);
}

/// BRANCH's migrated domain constraints as an object-safe `TableConstraints`.
pub struct BranchDomain;

impl TableConstraints<GoldilocksField, GoldilocksExtension> for BranchDomain {
    fn eval_prover(
        &self,
        cb: &mut ProverConstraintBuilder<GoldilocksField, GoldilocksExtension>,
        _ctx: &ConstraintContext<GoldilocksField, GoldilocksExtension>,
    ) {
        branch_domain_eval(cb);
    }

    fn eval_verifier(
        &self,
        cb: &mut VerifierConstraintBuilder<GoldilocksExtension>,
        _ctx: &ConstraintContext<GoldilocksExtension, GoldilocksExtension>,
    ) {
        branch_domain_eval(cb);
    }
}
