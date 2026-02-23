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
//! - Sender: IS_BYTE (×1 for next_pc_low[1])
//! - Sender: AND_BYTE (×1 for masking LSB)
//! - Sender: IS_HALFWORD (×3 for next_pc_high[0..3])
//! - Receiver: BRANCH (provides branch targets to CPU)

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::TransitionConstraint;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, SHIFT_16, SHIFT_32};

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
    use std::collections::HashMap;

    // Deduplicate operations: (pc, offset, register, jalr) -> multiplicity
    let mut op_map: HashMap<BranchOperation, u64> = HashMap::new();
    for op in operations {
        *op_map.entry(op.clone()).or_insert(0) += 1;
    }

    let unique_ops: Vec<_> = op_map.into_iter().collect();
    let num_rows = unique_ops.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, (op, multiplicity)) in unique_ops.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;

        // Extract pc as DWordWL: [Word, Word]
        let pc_0 = (op.pc & 0xFFFF_FFFF) as u32;
        let pc_1 = (op.pc >> 32) as u32;

        // Extract offset as DWordWL: [Word, Word]
        let offset_0 = (op.offset & 0xFFFF_FFFF) as u32;
        let offset_1 = (op.offset >> 32) as u32;

        // Extract register as DWordWL: [Word, Word]
        let register_0 = (op.register & 0xFFFF_FFFF) as u32;
        let register_1 = (op.register >> 32) as u32;

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
        data[base + cols::PC_0] = FE::from(pc_0 as u64);
        data[base + cols::PC_1] = FE::from(pc_1 as u64);
        data[base + cols::OFFSET_0] = FE::from(offset_0 as u64);
        data[base + cols::OFFSET_1] = FE::from(offset_1 as u64);
        data[base + cols::REGISTER_0] = FE::from(register_0 as u64);
        data[base + cols::REGISTER_1] = FE::from(register_1 as u64);
        data[base + cols::JALR] = FE::from(if op.jalr { 1u64 } else { 0u64 });
        data[base + cols::NEXT_PC_HIGH_0] = FE::from(next_pc_high_0 as u64);
        data[base + cols::NEXT_PC_HIGH_1] = FE::from(next_pc_high_1 as u64);
        data[base + cols::NEXT_PC_HIGH_2] = FE::from(next_pc_high_2 as u64);
        data[base + cols::NEXT_PC_LOW_0] = FE::from(next_pc_low_0 as u64);
        data[base + cols::NEXT_PC_LOW_1] = FE::from(next_pc_low_1 as u64);
        data[base + cols::UNMASKED_LOW_BYTE] = FE::from(unmasked_low_byte as u64);
        data[base + cols::MU] = FE::from(*multiplicity);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus Interactions
// =========================================================================

/// Creates all bus interactions for the BRANCH table.
///
/// The BRANCH table:
/// - **Sends** IS_BYTE lookup for next_pc_low[1] range check
/// - **Sends** AND_BYTE lookup for LSB masking (next_pc_low[0] = unmasked_low_byte & 254)
/// - **Sends** IS_HALFWORD lookups for next_pc_high[0..3] range checks
/// - **Receives** BRANCH lookups from CPU table
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // IS_BYTE[next_pc_low[1]] - range check bits 8-15
        BusInteraction::sender(
            BusId::IsByte,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::NEXT_PC_LOW_1,
                packing: Packing::Direct,
            }],
        ),
        // AND_BYTE[next_pc_low[0]; unmasked_low_byte, 254]
        // Verifies: next_pc_low[0] = unmasked_low_byte & 0xFE
        BusInteraction::sender(
            BusId::AndByte,
            Multiplicity::Column(cols::MU),
            vec![
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

/// BRANCH table constraint for ADD template verification.
///
/// The ADD template verifies: base + offset_ext = next_pc_unmasked
/// where base is either pc (when JALR=0) or register (when JALR=1).
///
/// This constraint embeds virtual carry computations and verifies:
/// 1. IS_BIT<carry[0]> - carry from low word addition is 0 or 1
/// 2. IS_BIT<carry[1]> - carry from high word addition is 0 or 1
///
/// The ADD template is computed as:
/// - iter=0: base[0] + offset_ext[0] = next_pc_unmasked[0] + carry[0] * 2^32
/// - iter=1: base[1] + offset_ext[1] + carry[0] = next_pc_unmasked[1] + carry[1] * 2^32
///
/// Where offset_ext is the 64-bit sign extension of the 32-bit offset.
pub struct BranchConstraint {
    /// Unique constraint identifier
    constraint_idx: usize,
    /// Which constraint to check
    kind: BranchConstraintKind,
}

/// Kind of BRANCH constraint.
#[derive(Debug, Clone, Copy)]
pub enum BranchConstraintKind {
    /// IS_BIT constraint on virtual carry[0]
    Carry0IsBit,
    /// IS_BIT constraint on virtual carry[1]
    Carry1IsBit,
}

impl BranchConstraint {
    /// Creates a new BRANCH constraint.
    pub fn new(kind: BranchConstraintKind, constraint_idx: usize) -> Self {
        Self {
            constraint_idx,
            kind,
        }
    }

    /// Compute the effective base address: (1-JALR)*pc + JALR*register
    fn compute_base<F, E>(&self, step: &TableView<F, E>) -> (FieldElement<F>, FieldElement<F>)
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let pc_0 = step.get_main_evaluation_element(0, cols::PC_0).clone();
        let pc_1 = step.get_main_evaluation_element(0, cols::PC_1).clone();
        let register_0 = step
            .get_main_evaluation_element(0, cols::REGISTER_0)
            .clone();
        let register_1 = step
            .get_main_evaluation_element(0, cols::REGISTER_1)
            .clone();
        let jalr = step.get_main_evaluation_element(0, cols::JALR).clone();

        let one = FieldElement::<F>::one();
        let one_minus_jalr = &one - &jalr;

        // base[0] = (1-JALR) * pc[0] + JALR * register[0]
        let base_0 = &one_minus_jalr * &pc_0 + &jalr * &register_0;
        // base[1] = (1-JALR) * pc[1] + JALR * register[1]
        let base_1 = &one_minus_jalr * &pc_1 + &jalr * &register_1;

        (base_0, base_1)
    }

    /// Compute virtual next_pc_unmasked as DWordWL.
    ///
    /// next_pc_unmasked[0] = 2^16 * next_pc_high[0] + 2^8 * next_pc_low[1] + unmasked_low_byte
    /// next_pc_unmasked[1] = 2^16 * next_pc_high[2] + next_pc_high[1]
    fn compute_next_pc_unmasked<F, E>(
        &self,
        step: &TableView<F, E>,
    ) -> (FieldElement<F>, FieldElement<F>)
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

        // next_pc_unmasked[0] = unmasked_low_byte + 2^8 * next_pc_low[1] + 2^16 * next_pc_high[0]
        let unmasked_0 =
            &unmasked_low_byte + &next_pc_low_1 * &shift_8 + &next_pc_high_0 * &shift_16;

        // next_pc_unmasked[1] = next_pc_high[1] + 2^16 * next_pc_high[2]
        let unmasked_1 = &next_pc_high_1 + &next_pc_high_2 * &shift_16;

        (unmasked_0, unmasked_1)
    }

    /// Compute virtual carry[0] from the addition check.
    ///
    /// From ADD template iter=0:
    /// base[0] + offset[0] = next_pc_unmasked[0] + carry[0] * 2^32
    ///
    /// Therefore:
    /// carry[0] = (base[0] + offset[0] - next_pc_unmasked[0]) / 2^32
    fn compute_carry_0<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let (base_0, _base_1) = self.compute_base(step);
        let offset_0 = step.get_main_evaluation_element(0, cols::OFFSET_0).clone();
        let (unmasked_0, _unmasked_1) = self.compute_next_pc_unmasked(step);

        // carry[0] = (base[0] + offset[0] - next_pc_unmasked[0]) / 2^32
        // Compute carry_0 as (base + offset - result) / 2^32 in the field.
        // This works because: base + offset = result + carry_0 * 2^32 (mod field)
        // Rearranging: carry_0 = (base + offset - result) * (2^32)^-1 (mod field)
        let inv_2_32 = FieldElement::<F>::from(SHIFT_32)
            .inv()
            .expect("2^32 must be invertible in field");
        (&base_0 + &offset_0 - &unmasked_0) * &inv_2_32
    }

    /// Compute virtual carry[1] from the addition check.
    ///
    /// From ADD template iter=1:
    /// base[1] + offset[1] + carry[0] = next_pc_unmasked[1] + carry[1] * 2^32
    ///
    /// With offset as DWordWL, offset[1] directly contains the sign-extended high word.
    ///
    /// carry[1] = (base[1] + offset[1] + carry[0] - next_pc_unmasked[1]) / 2^32
    fn compute_carry_1<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let (_base_0, base_1) = self.compute_base(step);
        let (_unmasked_0, unmasked_1) = self.compute_next_pc_unmasked(step);
        let carry_0 = self.compute_carry_0(step);

        // offset[1] already contains the sign-extended high word
        let offset_1 = step.get_main_evaluation_element(0, cols::OFFSET_1).clone();

        // carry[1] = (base[1] + offset[1] + carry[0] - unmasked[1]) / 2^32
        let inv_2_32 = FieldElement::<F>::from(SHIFT_32)
            .inv()
            .expect("2^32 must be invertible in field");
        (&base_1 + &offset_1 + &carry_0 - &unmasked_1) * &inv_2_32
    }

    /// Compute the constraint value.
    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let one = FieldElement::<F>::one();

        match self.kind {
            BranchConstraintKind::Carry0IsBit => {
                // IS_BIT<carry[0]>: carry[0] * (1 - carry[0]) = 0
                let c0 = self.compute_carry_0(step);
                &c0 * (&one - &c0)
            }
            BranchConstraintKind::Carry1IsBit => {
                // IS_BIT<carry[1]>: carry[1] * (1 - carry[1]) = 0
                let c1 = self.compute_carry_1(step);
                &c1 * (&one - &c1)
            }
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for BranchConstraint {
    fn degree(&self) -> usize {
        match self.kind {
            // IS_BIT on virtual carry: X*(1-X) is degree 2 in the carry
            // But carry itself is degree 2 due to JALR * register term
            // So total is degree 4 for carry^2 terms... but simplified is degree 2
            // Actually: base = (1-JALR)*pc + JALR*reg has degree 2 (JALR * something)
            // carry = (base + offset - unmasked) / 2^32 has degree 2
            // carry * (1 - carry) has degree 4
            BranchConstraintKind::Carry0IsBit => 4,
            BranchConstraintKind::Carry1IsBit => 4,
        }
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<GoldilocksField, GoldilocksExtension>,
        transition_evaluations: &mut [FieldElement<GoldilocksExtension>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _,
                rap_challenges: _,
                ..
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value.to_extension();
            }
            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _,
                rap_challenges: _,
                ..
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value;
            }
        }
    }
}

/// Creates all constraints for the BRANCH table.
///
/// Returns: (constraints, next_constraint_idx)
pub fn branch_constraints(constraint_idx_start: usize) -> (Vec<BranchConstraint>, usize) {
    let mut idx = constraint_idx_start;
    let constraints = vec![
        BranchConstraint::new(BranchConstraintKind::Carry0IsBit, {
            let i = idx;
            idx += 1;
            i
        }),
        BranchConstraint::new(BranchConstraintKind::Carry1IsBit, {
            let i = idx;
            idx += 1;
            i
        }),
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
pub fn compute_carries(base: u64, offset: u64, _next_pc_unmasked: u64) -> (u64, u64) {
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
