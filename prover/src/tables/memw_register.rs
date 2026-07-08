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
//! MEMW_R instead checks `IS_HALF[timestamp - old_timestamp - 1]`,
//! which proves the delta is in `[1, 2^16]` in a single lookup. This is safe
//! because registers are accessed very frequently — their timestamp deltas are
//! almost always small — and the routing predicate (`is_register_op`) enforces
//! the delta fits before admitting an op into this table.
//!
//! ## Column layout (9 columns)
//!
//! - `ADDRESS`:          Byte  (register index 0-31)
//! - `TIMESTAMP`:        Word  (32-bit timestamp)
//! - `VAL_0`:            Word  (low 32 bits of register value)
//! - `VAL_1`:            Word  (high 32 bits of register value)
//! - `OLD_0`:            Word  (low 32 bits of previous value)
//! - `OLD_1`:            Word  (high 32 bits of previous value)
//! - `OLD_TIMESTAMP`:    Word  (32-bit timestamp at which this register was last accessed)
//! - `MU_READ`:          Bit
//! - `MU_WRITE`:         Bit
//!
//! ## Virtual
//!
//! - `mu_sum = MU_READ + MU_WRITE`
//!
//! ## Bus Interactions (7)
//! - 1 IS_HALFWORD[timestamp - old_timestamp - 1]
//! - 4 Memory bus tokens (read-old + write-new, per word)
//! - 2 MEMW output interactions (read + write, from CPU)

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

use super::memw::MemwOperation;
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};
use crate::constraints::templates::emit_is_bit;

// =========================================================================
// Column indices (9 columns)
// =========================================================================

pub mod cols {
    /// Register index (0-31). CPU sends base_address = 2*reg_index.
    pub const ADDRESS: usize = 0;

    /// Timestamp: Word (32 bits)
    pub const TIMESTAMP: usize = 1;

    /// Register value low 32 bits
    pub const VAL_0: usize = 2;
    /// Register value high 32 bits
    pub const VAL_1: usize = 3;

    /// Previous value low 32 bits
    pub const OLD_0: usize = 4;
    /// Previous value high 32 bits
    pub const OLD_1: usize = 5;

    /// Old timestamp: Word (32 bits)
    pub const OLD_TIMESTAMP: usize = 6;

    /// Read multiplicity
    pub const MU_READ: usize = 7;
    /// Write multiplicity
    pub const MU_WRITE: usize = 8;

    pub const NUM_COLUMNS: usize = 9;
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
    let mut trace = TraceTable::new_main(
        vec![FE::zero(); num_rows * cols::NUM_COLUMNS],
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, op) in operations.iter().enumerate() {
        debug_assert_eq!(
            op.base_address % 2,
            0,
            "register base_address must be even (got {})",
            op.base_address
        );
        // Both register words must have been last accessed at the same timestamp.
        // MEMW_R stores a single old_timestamp shared by both words, so if the two
        // words differ, the wrong token would be sent to the memory bus. The routing
        // predicate enforces this before dispatch.
        debug_assert_eq!(
            op.old_timestamp[0], op.old_timestamp[1],
            "register words must share old_timestamp ({} != {})",
            op.old_timestamp[0], op.old_timestamp[1]
        );

        // ADDRESS = base_address / 2 (CPU sends 2 * register_index)
        table.set_u64(row_idx, cols::ADDRESS, op.base_address / 2);

        // Timestamp: single Word
        table.set_u64(row_idx, cols::TIMESTAMP, op.timestamp);

        // Value: registers are DWordWL = 2 words
        table.set_u64(row_idx, cols::VAL_0, op.value[0]);
        table.set_u64(row_idx, cols::VAL_1, op.value[1]);

        // Old value
        table.set_u64(row_idx, cols::OLD_0, op.old[0]);
        table.set_u64(row_idx, cols::OLD_1, op.old[1]);

        // Old timestamp: single Word
        table.set_u64(row_idx, cols::OLD_TIMESTAMP, op.old_timestamp[0]);

        // Multiplicity
        table.set_bool(row_idx, cols::MU_READ, op.is_read);
        table.set_bool(row_idx, cols::MU_WRITE, !op.is_read);
    }

    trace
}

// =========================================================================
// Bus interactions (7 total)
// =========================================================================

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(7);

    let mu_sum = Multiplicity::Sum(cols::MU_READ, cols::MU_WRITE);

    // -------------------------------------------------------------------------
    // IS_HALFWORD[timestamp - old_timestamp - 1] with mu_sum
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::IsHalfword,
        mu_sum.clone(),
        vec![BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::TIMESTAMP,
            },
            LinearTerm::Column {
                coefficient: -1,
                column: cols::OLD_TIMESTAMP,
            },
            LinearTerm::Constant(-1),
        ])],
    ));

    // -------------------------------------------------------------------------
    // Memory bus read-old (sender, for i=0,1)
    // memory[is_register=1, addr_lo=2*ADDRESS+i, addr_hi=0,
    //        OLD_TIMESTAMP, OLD[i]]
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
                    start_column: cols::OLD_TIMESTAMP,
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
    //        TIMESTAMP, VAL[i]]
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
                    start_column: cols::TIMESTAMP,
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
        vec![
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
                start_column: cols::TIMESTAMP,
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
        vec![
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
                start_column: cols::TIMESTAMP,
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
// Single-source constraint set (ConstraintBuilder front-end)
// =========================================================================

/// The MEMW_R table's 3 transition constraints as a single [`ConstraintSet`]:
/// - idx 0,1: `IS_BIT` on `μ_read`, `μ_write`;
/// - idx 2:   `IS_BIT<μ_sum>` with `μ_sum = μ_read + μ_write`.
pub struct MemwRegisterConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for MemwRegisterConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0,1: IS_BIT<μ_read>, IS_BIT<μ_write>
        emit_is_bit(b, 0, cols::MU_READ, None);
        emit_is_bit(b, 1, cols::MU_WRITE, None);

        // idx 2: IS_BIT<μ_sum> = μ_sum * (1 - μ_sum), μ_sum = μ_read + μ_write
        let one = b.one();
        let mu_sum = b.main(0, cols::MU_READ) + b.main(0, cols::MU_WRITE);
        b.emit_base(2, mu_sum.clone() * (one - mu_sum));
    }
}
