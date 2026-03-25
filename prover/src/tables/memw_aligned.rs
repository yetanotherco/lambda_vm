//! MEMW_A (Memory Write/Read — Aligned) table.
//!
//! Fast path for aligned memory/register accesses where all bytes share the
//! same old_timestamp. Most operations (aligned memory + all register accesses)
//! route here instead of the heavier MEMW table.
//!
//! ## Column layout (30 columns)
//!
//! - `is_register`: Bit
//! - `base_address_high`: Word (32-bit high word)
//! - `base_address_mid`: Half (16-bit mid)
//! - `base_address_low[2]`: Bytes (low 2 bytes)
//! - `value[8]`: BaseField[8]
//! - `timestamp`: DWordWL (2 cols)
//! - `write2/4/8`: Bit (access width flags)
//! - `old[8]`: BaseField[8]
//! - `old_timestamp`: DWordWL (2 cols — single, not 8!)
//! - `mu_read`, `mu_write`: multiplicity columns
//!
//! ## Bus Interactions (20)
//! - 1 AND_BYTE[base_address_low[0], mask] → 0 (alignment check)
//! - 1 LT[old_timestamp, timestamp, 0] → 1
//! - 16 Memory bus tokens
//! - 2 MEMW output interactions (read + write)
//!
//! ## Assumptions (caller's responsibility, not enforced here)
//! - MEMW_A-A2: IS_HALF[base_address_mid]
//! - MEMW_A-A3.i: IS_BYTE[base_address_low[i]] for i ∈ [0, 1]

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::TransitionConstraint;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

use super::memw::MemwOperation;
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

/// Maximum number of rows per MEMW_A table chunk.
pub const MAX_ROWS: usize = super::max_rows::MEMW_A;

// =========================================================================
// Column indices (30 columns)
// =========================================================================

pub mod cols {
    pub const IS_REGISTER: usize = 0;

    /// base_address decomposed: high = addr >> 32 (Word, 32-bit)
    pub const BASE_ADDRESS_HIGH: usize = 1;
    /// base_address decomposed: mid = (addr >> 16) & 0xFFFF (Half, 16-bit)
    pub const BASE_ADDRESS_MID: usize = 2;
    /// base_address decomposed: low bytes
    /// low[0] = addr & 0xFF, low[1] = (addr >> 8) & 0xFF
    pub const BASE_ADDRESS_LOW: [usize; 2] = [3, 4];

    pub const VALUE: [usize; 8] = [5, 6, 7, 8, 9, 10, 11, 12];

    pub const TIMESTAMP_0: usize = 13;
    pub const TIMESTAMP_1: usize = 14;

    pub const WRITE2: usize = 15;
    pub const WRITE4: usize = 16;
    pub const WRITE8: usize = 17;

    pub const OLD: [usize; 8] = [18, 19, 20, 21, 22, 23, 24, 25];

    /// Single old_timestamp (shared across all bytes, since they're aligned)
    pub const OLD_TIMESTAMP_0: usize = 26;
    pub const OLD_TIMESTAMP_1: usize = 27;

    pub const MU_READ: usize = 28;
    pub const MU_WRITE: usize = 29;

    pub const NUM_COLUMNS: usize = 30;
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
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, op) in operations.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;

        data[base + cols::IS_REGISTER] = FE::from(op.is_register as u64);

        // Decompose base_address
        let addr = op.base_address;
        let high = addr >> 32;
        let mid = (addr >> 16) & 0xFFFF;
        let low_1 = (addr >> 8) & 0xFF;
        let low_0 = addr & 0xFF;

        data[base + cols::BASE_ADDRESS_HIGH] = FE::from(high);
        data[base + cols::BASE_ADDRESS_MID] = FE::from(mid);
        data[base + cols::BASE_ADDRESS_LOW[0]] = FE::from(low_0);
        data[base + cols::BASE_ADDRESS_LOW[1]] = FE::from(low_1);

        for i in 0..8 {
            data[base + cols::VALUE[i]] = FE::from(op.value[i]);
        }

        data[base + cols::TIMESTAMP_0] = FE::from(op.timestamp & 0xFFFF_FFFF);
        data[base + cols::TIMESTAMP_1] = FE::from(op.timestamp >> 32);

        let (w2, w4, w8) = op.write_flags();
        data[base + cols::WRITE2] = FE::from(w2 as u64);
        data[base + cols::WRITE4] = FE::from(w4 as u64);
        data[base + cols::WRITE8] = FE::from(w8 as u64);

        for i in 0..8 {
            data[base + cols::OLD[i]] = FE::from(op.old[i]);
        }

        // Single old_timestamp (from old_timestamp[0], verified equal for all bytes)
        data[base + cols::OLD_TIMESTAMP_0] = FE::from(op.old_timestamp[0] & 0xFFFF_FFFF);
        data[base + cols::OLD_TIMESTAMP_1] = FE::from(op.old_timestamp[0] >> 32);

        data[base + cols::MU_READ] = FE::from(op.is_read as u64);
        data[base + cols::MU_WRITE] = FE::from(!op.is_read as u64);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions (20 total)
// =========================================================================

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(20);

    let mu_sum = Multiplicity::Sum(cols::MU_READ, cols::MU_WRITE);

    // -------------------------------------------------------------------------
    // AND_BYTE[base_address_low[0], mask] → 0 with μ_sum
    // mask = write2*1 + write4*3 + write8*7
    // This implicitly range-checks low[0] to [0, 256) AND checks alignment.
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::AndByte,
        mu_sum.clone(),
        vec![
            // x = base_address_low[0]
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_LOW[0],
                packing: Packing::Direct,
            },
            // y = mask = write2*1 + write4*3 + write8*7
            BusValue::linear(vec![
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
            ]),
            // result = 0 (alignment constraint: low bits must be 0)
            BusValue::constant(0),
        ],
    ));

    // -------------------------------------------------------------------------
    // LT[old_timestamp, timestamp, 0] → 1 with μ_sum
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Lt,
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
            BusValue::constant(0),
            BusValue::constant(1),
        ],
    ));

    // -------------------------------------------------------------------------
    // Memory bus interactions (16 total)
    // -------------------------------------------------------------------------
    // For aligned accesses, address for byte i:
    //   lo = 2^16 * MID + 2^8 * LOW[1] + LOW[0] + i
    //   hi = HIGH
    // All old_timestamp references use the single old_timestamp columns.

    // Virtual base_address_lo = 2^16 * MID + 2^8 * LOW[1] + LOW[0]
    // For byte 0, the address is exactly this.
    let base_addr_lo = BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1 << 16,
            column: cols::BASE_ADDRESS_MID,
        },
        LinearTerm::Column {
            coefficient: 1 << 8,
            column: cols::BASE_ADDRESS_LOW[1],
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::BASE_ADDRESS_LOW[0],
        },
    ]);

    let base_addr_hi = BusValue::Packed {
        start_column: cols::BASE_ADDRESS_HIGH,
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
    // For aligned accesses, adding 1 to the low byte never overflows to hi word
    // (since alignment guarantees base_address_lo + width-1 < 2^32).
    let addr_1_lo = BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1 << 16,
            column: cols::BASE_ADDRESS_MID,
        },
        LinearTerm::Column {
            coefficient: 1 << 8,
            column: cols::BASE_ADDRESS_LOW[1],
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::BASE_ADDRESS_LOW[0],
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
                coefficient: 1 << 16,
                column: cols::BASE_ADDRESS_MID,
            },
            LinearTerm::Column {
                coefficient: 1 << 8,
                column: cols::BASE_ADDRESS_LOW[1],
            },
            LinearTerm::Column {
                coefficient: 1,
                column: cols::BASE_ADDRESS_LOW[0],
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
                coefficient: 1 << 16,
                column: cols::BASE_ADDRESS_MID,
            },
            LinearTerm::Column {
                coefficient: 1 << 8,
                column: cols::BASE_ADDRESS_LOW[1],
            },
            LinearTerm::Column {
                coefficient: 1,
                column: cols::BASE_ADDRESS_LOW[0],
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
    // The MEMW output bus fingerprint uses base_address as [lo32, hi32].
    // Reconstruct: lo32 = 2^16*MID + 2^8*LOW[1] + LOW[0], hi32 = HIGH
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
            // base_address reconstructed as [lo32, hi32]
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
            // base_address reconstructed
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
// Constraints (2 algebraic)
// =========================================================================

/// MEMW_A constraint kinds.
#[derive(Debug, Clone, Copy)]
pub enum MemwAlignedConstraintKind {
    /// IS_BIT<μ_sum>: multiplicity sum is 0 or 1
    MuSumIsBit,
    /// w2 => μ_sum: if accessing 2+ bytes, must be active row
    W2ImpliesMuSum,
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
                let v = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = v.to_extension();
            }
            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _,
                rap_challenges: _,
                ..
            } => {
                let v = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = v;
            }
        }
    }
}

/// Creates all constraints for the MEMW_A table (2 total).
pub fn constraints() -> Vec<Box<dyn TransitionConstraint<GoldilocksField, GoldilocksExtension>>> {
    vec![
        Box::new(MemwAlignedConstraint::new(
            MemwAlignedConstraintKind::MuSumIsBit,
            0,
        )),
        Box::new(MemwAlignedConstraint::new(
            MemwAlignedConstraintKind::W2ImpliesMuSum,
            1,
        )),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memw_aligned_trace_generation() {
        let ops = vec![
            MemwOperation::new(true, 4, [42, 0, 0, 0, 0, 0, 0, 0], 100, 2, true)
                .with_old([42, 0, 0, 0, 0, 0, 0, 0], [50, 50, 0, 0, 0, 0, 0, 0]),
            MemwOperation::new(false, 0x1000, [1, 2, 3, 4, 0, 0, 0, 0], 200, 4, false)
                .with_old([0; 8], [100; 8]),
        ];

        let trace = generate_memw_aligned_trace(&ops);
        assert_eq!(trace.num_cols(), cols::NUM_COLUMNS);
        assert!(trace.num_rows() >= 2);

        // Check address decomposition for op[1]: addr = 0x1000
        // high = 0, mid = 0, low[1] = 0x10, low[0] = 0x00
        assert_eq!(*trace.get_main(1, cols::BASE_ADDRESS_HIGH), FE::from(0u64));
        assert_eq!(*trace.get_main(1, cols::BASE_ADDRESS_MID), FE::from(0u64));
        assert_eq!(
            *trace.get_main(1, cols::BASE_ADDRESS_LOW[1]),
            FE::from(0x10u64)
        );
        assert_eq!(
            *trace.get_main(1, cols::BASE_ADDRESS_LOW[0]),
            FE::from(0x00u64)
        );
    }
}
