//! MEMW_R (Memory Write/Read -- Register) table.
//!
//! Ultra-slim fast path for register accesses. Registers are always 2 words
//! (DWordWL), always aligned, and `is_register=1`, so this table strips out
//! all memory-specific columns (address decomposition, alignment mask, width
//! flags, per-byte old_timestamps).
//!
//! ## Timestamp ordering: IS_HALF instead of LT
//!
//! The general MEMW table proves `old_timestamp < timestamp` by routing through
//! the LT table, which requires extra LT trace rows and bus interactions.
//! MEMW_R instead checks `IS_HALF[timestamp[0] - old_timestamp[0] - 1]`,
//! which proves the delta is in `[1, 2^16]` in a single lookup. This is safe
//! because registers are accessed very frequently — their timestamp deltas are
//! almost always small — and the routing predicate (`is_register_op`) enforces
//! the delta fits before admitting an op into this table.
//!
//! ## Column layout (10 columns)
//!
//! - `ADDRESS`:          Byte  (register index 0-31)
//! - `TIMESTAMP_0`:      Word  (low 32 bits)
//! - `TIMESTAMP_1`:      Word  (high 32 bits)
//! - `VAL_0`:            Word  (low 32 bits of register value)
//! - `VAL_1`:            Word  (high 32 bits of register value)
//! - `OLD_0`:            Word  (low 32 bits of previous value)
//! - `OLD_1`:            Word  (high 32 bits of previous value)
//! - `OLD_TIMESTAMP_LO`: Word  (low 32 bits of old timestamp; upper limb = TIMESTAMP_1)
//! - `MU_READ`:          Bit
//! - `MU_WRITE`:         Bit
//!
//! ## Virtual
//!
//! - `old_timestamp = [OLD_TIMESTAMP_LO, TIMESTAMP_1]` (shares upper limb!)
//! - `mu_sum = MU_READ + MU_WRITE`
//!
//! ## Bus Interactions (7)
//! - 1 IS_HALFWORD[timestamp_0 - old_timestamp_lo - 1]
//! - 4 Memory bus tokens (read-old + write-new, per word)
//! - 2 MEMW output interactions (read + write, from CPU)

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use smallvec::smallvec;
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;

use super::memw::MemwOperation;
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices (10 columns)
// =========================================================================

pub mod cols {
    /// Register index (0-31). CPU sends base_address = 2*reg_index.
    pub const ADDRESS: usize = 0;

    /// Timestamp low 32 bits
    pub const TIMESTAMP_0: usize = 1;
    /// Timestamp high 32 bits
    pub const TIMESTAMP_1: usize = 2;

    /// Register value low 32 bits
    pub const VAL_0: usize = 3;
    /// Register value high 32 bits
    pub const VAL_1: usize = 4;

    /// Previous value low 32 bits
    pub const OLD_0: usize = 5;
    /// Previous value high 32 bits
    pub const OLD_1: usize = 6;

    /// Old timestamp low 32 bits (upper limb shared with TIMESTAMP_1)
    pub const OLD_TIMESTAMP_LO: usize = 7;

    /// Read multiplicity
    pub const MU_READ: usize = 8;
    /// Write multiplicity
    pub const MU_WRITE: usize = 9;

    pub const NUM_COLUMNS: usize = 10;
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the MEMW_R trace table from register operations.
///
/// Reuses `MemwOperation` -- the trace generator divides `base_address` by 2
/// to recover the register index (CPU sends `2 * register_index`).
pub fn generate_memw_register_trace(
    operations: &[MemwOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = operations.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, op) in operations.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;

        debug_assert_eq!(
            op.base_address % 2,
            0,
            "register base_address must be even (got {})",
            op.base_address
        );
        // Both register words must have been last accessed at the same timestamp.
        // MEMW_R stores a single old_timestamp_lo and shares TIMESTAMP_1 as the
        // upper limb, so if the two words differ, the wrong token would be sent
        // to the memory bus. The routing predicate enforces this before dispatch.
        debug_assert_eq!(
            op.old_timestamp[0], op.old_timestamp[1],
            "register words must share old_timestamp ({} != {})",
            op.old_timestamp[0], op.old_timestamp[1]
        );

        // ADDRESS = base_address / 2 (CPU sends 2 * register_index)
        data[base + cols::ADDRESS] = FE::from(op.base_address / 2);

        // Timestamp split into lo/hi 32-bit words
        data[base + cols::TIMESTAMP_0] = FE::from(op.timestamp & 0xFFFF_FFFF);
        data[base + cols::TIMESTAMP_1] = FE::from(op.timestamp >> 32);

        // Value: registers are DWordWL = 2 words
        data[base + cols::VAL_0] = FE::from(op.value[0]);
        data[base + cols::VAL_1] = FE::from(op.value[1]);

        // Old value
        data[base + cols::OLD_0] = FE::from(op.old[0]);
        data[base + cols::OLD_1] = FE::from(op.old[1]);

        // Old timestamp low (upper limb shared with TIMESTAMP_1)
        data[base + cols::OLD_TIMESTAMP_LO] = FE::from(op.old_timestamp[0] & 0xFFFF_FFFF);

        // Multiplicity
        data[base + cols::MU_READ] = FE::from(op.is_read as u64);
        data[base + cols::MU_WRITE] = FE::from(!op.is_read as u64);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions (7 total)
// =========================================================================

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(7);

    let mu_sum = Multiplicity::Sum(cols::MU_READ, cols::MU_WRITE);

    // -------------------------------------------------------------------------
    // IS_HALFWORD[timestamp_0 - old_timestamp_lo - 1] with mu_sum
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::IsHalfword,
        mu_sum.clone(),
        vec![BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::TIMESTAMP_0,
            },
            LinearTerm::Column {
                coefficient: -1,
                column: cols::OLD_TIMESTAMP_LO,
            },
            LinearTerm::Constant(-1),
        ])],
    ));

    // -------------------------------------------------------------------------
    // Memory bus read-old (sender, for i=0,1)
    // memory[is_register=1, addr_lo=2*ADDRESS+i, addr_hi=0,
    //        OLD_TIMESTAMP_LO, TIMESTAMP_1, OLD[i]]
    // -------------------------------------------------------------------------
    for i in 0..2 {
        let addr_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 2,
                column: cols::ADDRESS,
            },
            LinearTerm::Constant(i as i64),
        ]);

        interactions.push(BusInteraction::sender(
            BusId::Memory,
            mu_sum.clone(),
            vec![
                BusValue::constant(1),
                addr_lo,
                BusValue::constant(0),
                BusValue::Packed {
                    start_column: cols::OLD_TIMESTAMP_LO,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: if i == 0 { cols::OLD_0 } else { cols::OLD_1 },
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // -------------------------------------------------------------------------
    // Memory bus write-new (receiver, for i=0,1)
    // memory[is_register=1, addr_lo=2*ADDRESS+i, addr_hi=0,
    //        TIMESTAMP_0, TIMESTAMP_1, VAL[i]]
    // -------------------------------------------------------------------------
    for i in 0..2 {
        let addr_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 2,
                column: cols::ADDRESS,
            },
            LinearTerm::Constant(i as i64),
        ]);

        interactions.push(BusInteraction::receiver(
            BusId::Memory,
            mu_sum.clone(),
            vec![
                BusValue::constant(1),
                addr_lo,
                BusValue::constant(0),
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: if i == 0 { cols::VAL_0 } else { cols::VAL_1 },
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // -------------------------------------------------------------------------
    // CO24: MEMW read receiver (from CPU M1/M3 sender)
    // -------------------------------------------------------------------------
    let addr_lo_linear = BusValue::linear(vec![LinearTerm::Column {
        coefficient: 2,
        column: cols::ADDRESS,
    }]);

    interactions.push(BusInteraction::receiver(
        BusId::Memw,
        Multiplicity::Column(cols::MU_READ),
        smallvec![
            // old[0..8]
            BusValue::Packed {
                start_column: cols::OLD_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD_1,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // is_register = 1
            BusValue::constant(1),
            // base_address = [2*ADDRESS, 0]
            addr_lo_linear.clone(),
            BusValue::constant(0),
            // value[0..8]
            BusValue::Packed {
                start_column: cols::VAL_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VAL_1,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // timestamp
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            // write flags: write2=1, write4=0, write8=0 (registers are always 2 words)
            BusValue::constant(1),
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    ));

    // -------------------------------------------------------------------------
    // CO25: MEMW write receiver (from CPU M5 sender — register write to rd)
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::receiver(
        BusId::Memw,
        Multiplicity::Column(cols::MU_WRITE),
        smallvec![
            // is_register = 1
            BusValue::constant(1),
            // base_address = [2*ADDRESS, 0]
            addr_lo_linear,
            BusValue::constant(0),
            // value[0..8]
            BusValue::Packed {
                start_column: cols::VAL_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::VAL_1,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // timestamp
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            // write flags: write2=1, write4=0, write8=0
            BusValue::constant(1),
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    ));

    interactions
}

// =========================================================================
// Constraints (3 algebraic)
// =========================================================================

/// MEMW_R constraint: IS_BIT(mu_sum) = (mu_read + mu_write) * (1 - mu_read - mu_write) = 0
pub struct MemwRegisterMuSumIsBit {
    constraint_idx: usize,
}

impl MemwRegisterMuSumIsBit {
    pub fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
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
        &mu_sum * (&one - &mu_sum)
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for MemwRegisterMuSumIsBit {
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

/// Creates all constraints for the MEMW_R table (3 total).
///
/// - IS_BIT(MU_READ) -- unconditional
/// - IS_BIT(MU_WRITE) -- unconditional
/// - IS_BIT(mu_sum) = (mu_read + mu_write) * (1 - mu_read - mu_write) = 0
pub fn constraints()
-> Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>> {
    use crate::constraints::templates::IsBitConstraint;

    vec![
        IsBitConstraint::unconditional(cols::MU_READ, 0).boxed(),
        IsBitConstraint::unconditional(cols::MU_WRITE, 1).boxed(),
        MemwRegisterMuSumIsBit::new(2).boxed(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memw_register_trace_generation() {
        // Create a simple register op (reg x1 = address 1, so base_address = 2)
        let ops = vec![
            MemwOperation::new(
                true, // is_register
                2,    // base_address = 2 * register_index (reg x1)
                [42, 7, 0, 0, 0, 0, 0, 0],
                100,
                2, // width = 2 words (registers are DWordWL)
                true,
            )
            .with_old([10, 3, 0, 0, 0, 0, 0, 0], [50, 50, 0, 0, 0, 0, 0, 0]),
        ];

        let trace = generate_memw_register_trace(&ops);
        assert_eq!(trace.num_cols(), cols::NUM_COLUMNS);
        assert!(trace.num_rows() >= 4); // minimum 4 rows

        // ADDRESS = base_address / 2 = 2 / 2 = 1
        assert_eq!(*trace.get_main(0, cols::ADDRESS), FE::from(1u64));

        // TIMESTAMP split
        assert_eq!(*trace.get_main(0, cols::TIMESTAMP_0), FE::from(100u64));
        assert_eq!(*trace.get_main(0, cols::TIMESTAMP_1), FE::from(0u64));

        // Values
        assert_eq!(*trace.get_main(0, cols::VAL_0), FE::from(42u64));
        assert_eq!(*trace.get_main(0, cols::VAL_1), FE::from(7u64));

        // Old values
        assert_eq!(*trace.get_main(0, cols::OLD_0), FE::from(10u64));
        assert_eq!(*trace.get_main(0, cols::OLD_1), FE::from(3u64));

        // Old timestamp lo
        assert_eq!(*trace.get_main(0, cols::OLD_TIMESTAMP_LO), FE::from(50u64));

        // Multiplicity: is_read = true => MU_READ=1, MU_WRITE=0
        assert_eq!(*trace.get_main(0, cols::MU_READ), FE::from(1u64));
        assert_eq!(*trace.get_main(0, cols::MU_WRITE), FE::from(0u64));
    }

    #[test]
    fn test_memw_register_trace_generation_write_op() {
        // Write op: is_read = false => MU_WRITE=1, MU_READ=0
        let ops = vec![
            MemwOperation::new(
                true, // is_register
                4,    // base_address = 2 * register_index (reg x2)
                [99, 55, 0, 0, 0, 0, 0, 0],
                200,
                2,     // width = 2 words
                false, // is_read = false (write)
            )
            .with_old([11, 22, 0, 0, 0, 0, 0, 0], [180, 180, 0, 0, 0, 0, 0, 0]),
        ];

        let trace = generate_memw_register_trace(&ops);

        // ADDRESS = base_address / 2 = 4 / 2 = 2
        assert_eq!(*trace.get_main(0, cols::ADDRESS), FE::from(2u64));

        // Values
        assert_eq!(*trace.get_main(0, cols::VAL_0), FE::from(99u64));
        assert_eq!(*trace.get_main(0, cols::VAL_1), FE::from(55u64));

        // Old values
        assert_eq!(*trace.get_main(0, cols::OLD_0), FE::from(11u64));
        assert_eq!(*trace.get_main(0, cols::OLD_1), FE::from(22u64));

        // Old timestamp lo
        assert_eq!(*trace.get_main(0, cols::OLD_TIMESTAMP_LO), FE::from(180u64));

        // Multiplicity: is_read = false => MU_WRITE=1, MU_READ=0
        assert_eq!(*trace.get_main(0, cols::MU_READ), FE::from(0u64));
        assert_eq!(*trace.get_main(0, cols::MU_WRITE), FE::from(1u64));
    }
}
