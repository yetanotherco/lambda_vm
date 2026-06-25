//! MEMW_A (Memory Write/Read — Aligned) table.
//!
//! Fast path for aligned memory/register accesses where all bytes share the
//! same old_timestamp. Most operations (aligned memory + all register accesses)
//! route here instead of the heavier MEMW table.
//!
//! ## Column layout (29 columns)
//!
//! - `is_register`: Bit
//! - `base_address[3]`: DWordWHH
//!   - `base_address[0]`: Half (low 16 bits)
//!   - `base_address[1]`: Half (mid 16 bits)
//!   - `base_address[2]`: Word (high 32 bits)
//! - `value[8]`: BaseField[8]
//! - `timestamp`: DWordWL (2 cols)
//! - `write2/4/8`: Bit (access width flags)
//! - `old[8]`: BaseField[8]
//! - `old_timestamp`: DWordWL (2 cols — single, not 8!)
//! - `mu_read`, `mu_write`: multiplicity columns
//!
//! ## Bus Interactions (20)
//! - 1 IS_HALF[base_address[0] + mask] (range check: address span fits in 16 bits)
//! - 1 ALU[old_timestamp, timestamp, opsel(LT), 1, 0] → asserts old_ts < ts
//! - 16 Memory bus tokens
//! - 2 MEMW output interactions (read + write)
//!
//! ## Constraints (4 total)
//! - IS_BIT<μ_sum> (1)
//! - w2 => μ_sum (1)
//! - IS_BIT<μ_read> (1)
//! - IS_BIT<μ_write> (1)
//!
//! ## Assumptions (caller's responsibility, not enforced here)
//! - IS_HALF[base_address[i]] for i ∈ [0, 1]
//! - IS_WORD[base_address[2]]

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::constraints::builder::{
    ConstraintBuilder, ConstraintContext, ProverConstraintBuilder, TableConstraints,
    VerifierConstraintBuilder,
};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;

use super::memw::MemwOperation;
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};
use crate::constraints::templates::IsBitConstraint;

/// Maximum number of rows per MEMW_A table chunk.
pub const MAX_ROWS: usize = super::max_rows::MEMW_A;

// =========================================================================
// Column indices (29 columns)
// =========================================================================

pub mod cols {
    pub const IS_REGISTER: usize = 0;

    /// base_address: DWordWHH (3 columns)
    /// base_address[0] = low half (bits 0-15)
    /// base_address[1] = mid half (bits 16-31)
    /// base_address[2] = high word (bits 32-63)
    pub const BASE_ADDRESS: [usize; 3] = [1, 2, 3];

    pub const VALUE: [usize; 8] = [4, 5, 6, 7, 8, 9, 10, 11];

    pub const TIMESTAMP_0: usize = 12;
    pub const TIMESTAMP_1: usize = 13;

    pub const WRITE2: usize = 14;
    pub const WRITE4: usize = 15;
    pub const WRITE8: usize = 16;

    pub const OLD: [usize; 8] = [17, 18, 19, 20, 21, 22, 23, 24];

    /// Single old_timestamp (shared across all bytes, since they're aligned)
    pub const OLD_TIMESTAMP_0: usize = 25;
    pub const OLD_TIMESTAMP_1: usize = 26;

    pub const MU_READ: usize = 27;
    pub const MU_WRITE: usize = 28;

    pub const NUM_COLUMNS: usize = 29;
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the MEMW_A trace table from aligned operations.
///
/// Reuses `MemwOperation` — the trace generator uses `old_timestamp[0]`
/// (verified equal for all accessed bytes by the routing logic).
pub fn generate_memw_aligned_trace(
    operations: &[MemwOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = operations.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        vec![FE::zero(); num_rows * cols::NUM_COLUMNS],
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, op) in operations.iter().enumerate() {
        table.set_bool(row_idx, cols::IS_REGISTER, op.is_register);

        table.set_dword_whh(row_idx, cols::BASE_ADDRESS[0], op.base_address);

        for i in 0..8 {
            table.set_u64(row_idx, cols::VALUE[i], op.value[i]);
        }

        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);

        let (w2, w4, w8) = op.write_flags();
        table.set_bool(row_idx, cols::WRITE2, w2);
        table.set_bool(row_idx, cols::WRITE4, w4);
        table.set_bool(row_idx, cols::WRITE8, w8);

        for i in 0..8 {
            table.set_u64(row_idx, cols::OLD[i], op.old[i]);
        }

        // Single old_timestamp (from old_timestamp[0], verified equal for all bytes)
        table.set_dword_wl(row_idx, cols::OLD_TIMESTAMP_0, op.old_timestamp[0]);

        table.set_bool(row_idx, cols::MU_READ, op.is_read);
        table.set_bool(row_idx, cols::MU_WRITE, !op.is_read);
    }

    trace
}

// =========================================================================
// Bus interactions (20 total)
// =========================================================================

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(20);

    let mu_sum = Multiplicity::Sum(cols::MU_READ, cols::MU_WRITE);

    // -------------------------------------------------------------------------
    // IS_HALF[base_address[0] + write2 + 3*write4 + 7*write8] with μ_sum
    // Range check: ensures base_address[0] + mask fits in 16 bits, so the
    // byte-address span of the access doesn't overflow the low-half field element.
    // Alignment itself is the caller's (CPU's) responsibility — see Assumptions above.
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::IsHalfword,
        mu_sum.clone(),
        vec![BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::BASE_ADDRESS[0],
            },
            LinearTerm::Column {
                coefficient: 1,
                column: cols::WRITE2,
            },
            LinearTerm::Column {
                coefficient: 3,
                column: cols::WRITE4,
            },
            LinearTerm::Column {
                coefficient: 7,
                column: cols::WRITE8,
            },
        ])],
    ));

    // -------------------------------------------------------------------------
    // ALU[old_timestamp, timestamp, opsel(LT), 1, 0] → asserts old_ts < ts.
    // (Every LT lookup goes through the unified ALU bus with
    // signed=0/invert=0; there is no dedicated `Lt` bus.)
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Alu,
        mu_sum.clone(),
        vec![
            BusValue::Packed {
                start_column: cols::OLD_TIMESTAMP_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::DWordWL,
            },
            BusValue::constant(alu_op::LT as u64),
            BusValue::constant(1),
            BusValue::constant(0),
        ],
    ));

    // -------------------------------------------------------------------------
    // Memory bus interactions (16 total)
    // -------------------------------------------------------------------------
    // base_address as DWordWL:
    //   lo32 = base_address[0] + 2^16 * base_address[1]
    //   hi32 = base_address[2]
    // For aligned accesses, address for byte i: lo32 + i (no carry since aligned)

    let base_addr_lo = BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::BASE_ADDRESS[0],
        },
        LinearTerm::Column {
            coefficient: 1 << 16,
            column: cols::BASE_ADDRESS[1],
        },
    ]);

    let base_addr_hi = BusValue::Packed {
        start_column: cols::BASE_ADDRESS[2],
        packing: Packing::Direct,
    };

    // CM16: memory[is_register, base_address, old_timestamp, old[0]] with +μ_sum
    interactions.push(BusInteraction::sender(
        BusId::Memory,
        mu_sum.clone(),
        vec![
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            base_addr_lo.clone(),
            base_addr_hi.clone(),
            BusValue::Packed {
                start_column: cols::OLD_TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD_TIMESTAMP_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[0],
                packing: Packing::Direct,
            },
        ],
    ));

    // CM17: memory[is_register, base_address, timestamp, value[0]] with -μ_sum
    interactions.push(BusInteraction::receiver(
        BusId::Memory,
        mu_sum,
        vec![
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            base_addr_lo.clone(),
            base_addr_hi.clone(),
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[0],
                packing: Packing::Direct,
            },
        ],
    ));

    // w2 multiplicity: write2 + write4 + write8
    let w2_mult = Multiplicity::Sum3(cols::WRITE2, cols::WRITE4, cols::WRITE8);

    // CM18/19: byte 1 with w2
    // For aligned accesses, adding 1 to lo32 never overflows (alignment guarantees it).
    let addr_1_lo = BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::BASE_ADDRESS[0],
        },
        LinearTerm::Column {
            coefficient: 1 << 16,
            column: cols::BASE_ADDRESS[1],
        },
        LinearTerm::Constant(1),
    ]);

    interactions.push(BusInteraction::sender(
        BusId::Memory,
        w2_mult.clone(),
        vec![
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            addr_1_lo.clone(),
            base_addr_hi.clone(),
            BusValue::Packed {
                start_column: cols::OLD_TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD_TIMESTAMP_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[1],
                packing: Packing::Direct,
            },
        ],
    ));

    interactions.push(BusInteraction::receiver(
        BusId::Memory,
        w2_mult,
        vec![
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            addr_1_lo,
            base_addr_hi.clone(),
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[1],
                packing: Packing::Direct,
            },
        ],
    ));

    // CM20/21: bytes 2-3 with w4
    for i in 2..=3 {
        let addr_i_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::BASE_ADDRESS[0],
            },
            LinearTerm::Column {
                coefficient: 1 << 16,
                column: cols::BASE_ADDRESS[1],
            },
            LinearTerm::Constant(i as i64),
        ]);

        interactions.push(BusInteraction::sender(
            BusId::Memory,
            Multiplicity::Sum(cols::WRITE4, cols::WRITE8),
            vec![
                BusValue::Packed {
                    start_column: cols::IS_REGISTER,
                    packing: Packing::Direct,
                },
                addr_i_lo.clone(),
                base_addr_hi.clone(),
                BusValue::Packed {
                    start_column: cols::OLD_TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::OLD_TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::OLD[i],
                    packing: Packing::Direct,
                },
            ],
        ));

        interactions.push(BusInteraction::receiver(
            BusId::Memory,
            Multiplicity::Sum(cols::WRITE4, cols::WRITE8),
            vec![
                BusValue::Packed {
                    start_column: cols::IS_REGISTER,
                    packing: Packing::Direct,
                },
                addr_i_lo,
                base_addr_hi.clone(),
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::VALUE[i],
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // CM22/23: bytes 4-7 with write8
    for i in 4..=7 {
        let addr_i_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::BASE_ADDRESS[0],
            },
            LinearTerm::Column {
                coefficient: 1 << 16,
                column: cols::BASE_ADDRESS[1],
            },
            LinearTerm::Constant(i as i64),
        ]);

        interactions.push(BusInteraction::sender(
            BusId::Memory,
            Multiplicity::Column(cols::WRITE8),
            vec![
                BusValue::Packed {
                    start_column: cols::IS_REGISTER,
                    packing: Packing::Direct,
                },
                addr_i_lo.clone(),
                base_addr_hi.clone(),
                BusValue::Packed {
                    start_column: cols::OLD_TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::OLD_TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::OLD[i],
                    packing: Packing::Direct,
                },
            ],
        ));

        interactions.push(BusInteraction::receiver(
            BusId::Memory,
            Multiplicity::Column(cols::WRITE8),
            vec![
                BusValue::Packed {
                    start_column: cols::IS_REGISTER,
                    packing: Packing::Direct,
                },
                addr_i_lo,
                base_addr_hi.clone(),
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::VALUE[i],
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // -------------------------------------------------------------------------
    // CO24: Read receiver (from CPU)
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::receiver(
        BusId::Memw,
        Multiplicity::Column(cols::MU_READ),
        vec![
            // old[8]
            BusValue::Packed {
                start_column: cols::OLD[0],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[1],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[2],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[3],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[4],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[5],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[6],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[7],
                packing: Packing::Direct,
            },
            // is_register
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            // base_address as DWordWL: [lo32, hi32]
            base_addr_lo.clone(),
            base_addr_hi.clone(),
            // value[8]
            BusValue::Packed {
                start_column: cols::VALUE[0],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[1],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[2],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[3],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[4],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[5],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[6],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[7],
                packing: Packing::Direct,
            },
            // timestamp
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            // write flags
            BusValue::Packed {
                start_column: cols::WRITE2,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::WRITE4,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::WRITE8,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // CO25: Write receiver (from CPU)
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::receiver(
        BusId::Memw,
        Multiplicity::Column(cols::MU_WRITE),
        vec![
            // is_register
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            // base_address as DWordWL: [lo32, hi32]
            base_addr_lo,
            base_addr_hi,
            // value[8]
            BusValue::Packed {
                start_column: cols::VALUE[0],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[1],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[2],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[3],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[4],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[5],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[6],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VALUE[7],
                packing: Packing::Direct,
            },
            // timestamp
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            // write flags
            BusValue::Packed {
                start_column: cols::WRITE2,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::WRITE4,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::WRITE8,
                packing: Packing::Direct,
            },
        ],
    ));

    interactions
}

// =========================================================================
// Constraints (4 total)
// =========================================================================

/// MEMW_A constraint kinds.
#[derive(Debug, Clone, Copy)]
pub enum MemwAlignedConstraintKind {
    /// IS_BIT<μ_sum>: multiplicity sum is 0 or 1
    MuSumIsBit,
    /// w2 => μ_sum: if accessing 2+ bytes, must be active row
    W2ImpliesMuSum,
    /// IS_BIT<write2 + write4 + write8>: the width-sum is 0 or 1 (spec assumption).
    WidthSumIsBit,
}

pub struct MemwAlignedConstraint {
    constraint_idx: usize,
    kind: MemwAlignedConstraintKind,
}

impl MemwAlignedConstraint {
    pub fn new(kind: MemwAlignedConstraintKind, constraint_idx: usize) -> Self {
        Self {
            constraint_idx,
            kind,
        }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let one = FieldElement::<F>::one();
        let mu_read = step.get_main_evaluation_element(0, cols::MU_READ).clone();
        let mu_write = step.get_main_evaluation_element(0, cols::MU_WRITE).clone();
        let mu_sum = &mu_read + &mu_write;

        match self.kind {
            MemwAlignedConstraintKind::MuSumIsBit => &mu_sum * (&one - &mu_sum),
            MemwAlignedConstraintKind::W2ImpliesMuSum => {
                let write2 = step.get_main_evaluation_element(0, cols::WRITE2).clone();
                let write4 = step.get_main_evaluation_element(0, cols::WRITE4).clone();
                let write8 = step.get_main_evaluation_element(0, cols::WRITE8).clone();
                let w2 = write2 + write4 + write8;
                &w2 * (&one - &mu_sum)
            }
            MemwAlignedConstraintKind::WidthSumIsBit => {
                let write2 = step.get_main_evaluation_element(0, cols::WRITE2).clone();
                let write4 = step.get_main_evaluation_element(0, cols::WRITE4).clone();
                let write8 = step.get_main_evaluation_element(0, cols::WRITE8).clone();
                let w2 = write2 + write4 + write8;
                &w2 * (&one - &w2)
            }
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for MemwAlignedConstraint {
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
        self.compute(step)
    }
}

/// Creates all constraints for the MEMW_A table (8 total). The last four are the
/// spec's defense-in-depth width-flag assumptions.
pub fn constraints()
-> Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>> {
    vec![
        MemwAlignedConstraint::new(MemwAlignedConstraintKind::MuSumIsBit, 0).boxed(),
        MemwAlignedConstraint::new(MemwAlignedConstraintKind::W2ImpliesMuSum, 1).boxed(),
        IsBitConstraint::unconditional(cols::MU_READ, 2).boxed(),
        IsBitConstraint::unconditional(cols::MU_WRITE, 3).boxed(),
        IsBitConstraint::unconditional(cols::WRITE2, 4).boxed(),
        IsBitConstraint::unconditional(cols::WRITE4, 5).boxed(),
        IsBitConstraint::unconditional(cols::WRITE8, 6).boxed(),
        MemwAlignedConstraint::new(MemwAlignedConstraintKind::WidthSumIsBit, 7).boxed(),
    ]
}

pub fn memw_aligned_domain_eval<CB: ConstraintBuilder>(cb: &mut CB) {
    let one = FieldElement::<CB::F>::one();
    let mu_read = cb.main(cols::MU_READ).clone();
    let mu_write = cb.main(cols::MU_WRITE).clone();
    let mu_sum = &mu_read + &mu_write;
    // idx 0: MuSumIsBit — mu_sum * (1 - mu_sum)
    cb.fold(&mu_sum * &(&one - &mu_sum));
    // idx 1: W2ImpliesMuSum — (write2+write4+write8) * (1 - mu_sum)
    let w2 = cb.main(cols::WRITE2) + cb.main(cols::WRITE4) + cb.main(cols::WRITE8);
    cb.fold(&w2 * &(&one - &mu_sum));
    // idx 2: IS_BIT<mu_read>
    cb.fold(&mu_read * &(&one - &mu_read));
    // idx 3: IS_BIT<mu_write>
    cb.fold(&mu_write * &(&one - &mu_write));
    // idx 4-6: IS_BIT<write2/write4/write8>
    for &col in &[cols::WRITE2, cols::WRITE4, cols::WRITE8] {
        let v = cb.main(col).clone();
        cb.fold(&v * &(&one - &v));
    }
    // idx 7: WidthSumIsBit — w * (1 - w)
    let w = cb.main(cols::WRITE2) + cb.main(cols::WRITE4) + cb.main(cols::WRITE8);
    cb.fold(&w * &(&one - &w));
}

/// MEMW_A's migrated domain constraints as an object-safe `TableConstraints`.
pub struct MemwAlignedDomain;

impl TableConstraints<GoldilocksField, GoldilocksExtension> for MemwAlignedDomain {
    fn eval_prover(
        &self,
        cb: &mut ProverConstraintBuilder<GoldilocksField, GoldilocksExtension>,
        _ctx: &ConstraintContext<GoldilocksField, GoldilocksExtension>,
    ) {
        memw_aligned_domain_eval(cb);
    }

    fn eval_verifier(
        &self,
        cb: &mut VerifierConstraintBuilder<GoldilocksExtension>,
        _ctx: &ConstraintContext<GoldilocksExtension, GoldilocksExtension>,
    ) {
        memw_aligned_domain_eval(cb);
    }
}
