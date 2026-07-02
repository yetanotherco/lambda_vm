//! MEMW (Memory Write/Read) table — unaligned / split-timestamp path.
//!
//! This table handles memory and register read/write operations where bytes may
//! have different old_timestamps or the access is unaligned.
//!
//! ## Column layout (49 columns)
//!
//! - `is_register`: Bit (1 = register access, 0 = memory access)
//! - `base_address`: DWordWL (64-bit address, 2 cols)
//! - `value[8]`: BaseField[8] (8 bytes to write)
//! - `timestamp`: DWordWL (64-bit timestamp, 2 cols)
//! - `write2/4/8`: Bit (access width flags)
//! - `old[8]`: BaseField[8] (previous values at address)
//! - `carry[7]`: Bit[7] (carry flags for base_address + i)
//! - `old_timestamp[8]`: DWordWL[8] (previous timestamps, 16 cols)
//! - `mu_read`, `mu_write`: multiplicity columns
//!
//! ## Virtual (computed inline)
//! - `address_add[i]` = (base_address_0 + i+1 - 2^32 * carry[i], base_address_1 + carry[i])
//! - `w2`: write2 + write4 + write8 (writing at least 2 bytes)
//! - `w4`: write4 + write8 (writing at least 4 bytes)
//! - `μ_sum`: μ_read + μ_write
//!
//! ## Bus Interactions (26)
//! - 8 ALU lookups for timestamp ordering (old_timestamp[i] < timestamp,
//!   dispatched as `ALU[old_ts, ts, opsel(LT), 1, 0]` on the unified bus)
//! - 16 Memory bus tokens (read old + write new, per byte)
//! - 2 MEMW output interactions (read + write, from CPU)
//!
//! ## Constraints (11 total: 2 custom + 2 IS_BIT for multiplicities + 7 IS_BIT for carry)

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use stark::constraints::builder::{ConstraintBuilder, ConstraintMeta, ConstraintSet};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};
use crate::constraints::templates::emit_is_bit;

/// Maximum number of rows per MEMW table chunk.
/// If operations exceed this, the trace is split into multiple tables.
pub const MAX_ROWS: usize = super::max_rows::MEMW;

// =========================================================================
// Column indices for MEMW table (49 columns)
// =========================================================================

/// Column definitions for the MEMW table.
pub mod cols {
    // Input columns
    /// is_register: Bit (1 = register, 0 = memory)
    pub const IS_REGISTER: usize = 0;

    /// base_address: DWordWL (2 words = 2 columns)
    pub const BASE_ADDRESS_0: usize = 1;
    pub const BASE_ADDRESS_1: usize = 2;

    /// value[8]: 8 BaseField columns
    pub const VALUE: [usize; 8] = [3, 4, 5, 6, 7, 8, 9, 10];

    /// timestamp: DWordWL (2 words = 2 columns)
    pub const TIMESTAMP_0: usize = 11;
    pub const TIMESTAMP_1: usize = 12;

    /// write2, write4, write8: access width flags
    pub const WRITE2: usize = 13;
    pub const WRITE4: usize = 14;
    pub const WRITE8: usize = 15;

    // Output columns
    /// old[8]: 8 BaseField columns for previous values
    pub const OLD: [usize; 8] = [16, 17, 18, 19, 20, 21, 22, 23];

    // Auxiliary columns
    /// carry[7]: Bit columns indicating carry when adding i+1 to base_address_0
    pub const CARRY: [usize; 7] = [24, 25, 26, 27, 28, 29, 30];

    /// old_timestamp[8]: each is DWordWL (2 words = 2 columns)
    /// Total: 8 * 2 = 16 columns
    pub const OLD_TIMESTAMP_START: usize = 31;

    // Multiplicity columns
    /// μ_read: Whether we are performing a read
    pub const MU_READ: usize = 47;
    /// μ_write: Whether we are performing a write
    pub const MU_WRITE: usize = 48;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 49;

    /// Helper to get old_timestamp[i] column indices (2 words each)
    pub fn old_timestamp(i: usize) -> [usize; 2] {
        let base = OLD_TIMESTAMP_START + i * 2;
        [base, base + 1]
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// A single MEMW operation to be added to the trace.
#[derive(Debug, Clone)]
pub struct MemwOperation {
    /// Whether this is a register access (true) or memory access (false)
    pub is_register: bool,
    /// Base address (64-bit)
    pub base_address: u64,
    /// Values to write (8 bytes)
    pub value: [u64; 8],
    /// Timestamp of this access
    pub timestamp: u64,
    /// Access width: 1, 2, 4, or 8 bytes
    pub width: u8,
    /// Whether this is a read (true) or write (false)
    pub is_read: bool,
    /// Previous values at the addresses (filled by memory model)
    pub old: [u64; 8],
    /// Previous timestamps at the addresses (filled by memory model)
    pub old_timestamp: [u64; 8],
}

impl MemwOperation {
    /// Create a new MEMW operation.
    pub fn new(
        is_register: bool,
        base_address: u64,
        value: [u64; 8],
        timestamp: u64,
        width: u8,
        is_read: bool,
    ) -> Self {
        Self {
            is_register,
            base_address,
            value,
            timestamp,
            width,
            is_read,
            old: [0; 8],
            old_timestamp: [0; 8],
        }
    }

    /// Set the old values (from memory model).
    pub fn with_old(mut self, old: [u64; 8], old_timestamp: [u64; 8]) -> Self {
        self.old = old;
        self.old_timestamp = old_timestamp;
        self
    }

    /// Convert access width to the spec's flag representation (write2, write4, write8).
    ///
    /// | Width | write2 | write4 | write8 |
    /// |-------|--------|--------|--------|
    /// |   1   |   0    |   0    |   0    |
    /// |   2   |   1    |   0    |   0    |
    /// |   4   |   0    |   1    |   0    |
    /// |   8   |   0    |   0    |   1    |
    pub fn write_flags(&self) -> (bool, bool, bool) {
        match self.width {
            1 => (false, false, false),
            2 => (true, false, false),
            4 => (false, true, false),
            8 => (false, false, true),
            _ => (false, false, false),
        }
    }
}

/// Generates the MEMW trace table from a list of operations.
pub fn generate_memw_trace(
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
        // Input columns
        table.set_bool(row_idx, cols::IS_REGISTER, op.is_register);

        // base_address as DWordWL (2 words)
        let base_addr_lo = op.base_address & 0xFFFF_FFFF;
        table.set_dword_wl(row_idx, cols::BASE_ADDRESS_0, op.base_address);

        // value[8]
        for i in 0..8 {
            table.set_u64(row_idx, cols::VALUE[i], op.value[i]);
        }

        // timestamp as DWordWL (2 words)
        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);

        // write flags
        let (w2, w4, w8) = op.write_flags();
        table.set_bool(row_idx, cols::WRITE2, w2);
        table.set_bool(row_idx, cols::WRITE4, w4);
        table.set_bool(row_idx, cols::WRITE8, w8);

        // Output: old[8]
        for i in 0..8 {
            table.set_u64(row_idx, cols::OLD[i], op.old[i]);
        }

        // Auxiliary: carry[7]
        // carry[i] = 1 if (base_address_lo + i+1) >= 2^32
        for i in 0..7 {
            let overflows = base_addr_lo + (i as u64 + 1) >= (1u64 << 32);
            table.set_bool(row_idx, cols::CARRY[i], overflows);
        }

        // Auxiliary: old_timestamp[8] - each as DWordWL (2 words)
        for i in 0..8 {
            let cols_i = cols::old_timestamp(i);
            table.set_dword_wl(row_idx, cols_i[0], op.old_timestamp[i]);
        }

        // Multiplicity
        table.set_bool(row_idx, cols::MU_READ, op.is_read);
        table.set_bool(row_idx, cols::MU_WRITE, !op.is_read);
    }

    trace
}

// =========================================================================
// Bus interactions (26 total)
// =========================================================================

/// Creates all bus interactions for the MEMW table.
///
/// 26 interactions:
/// - 8 LT timestamp ordering checks
/// - 16 Memory bus tokens (read old + write new per byte)
/// - 2 MEMW output interactions (read + write from CPU)
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(26);

    // -------------------------------------------------------------------------
    // Memory bus interactions (16 total)
    // -------------------------------------------------------------------------
    // address_add[i] is VIRTUAL:
    //   lo = base_address_0 + (i+1) - 2^32 * carry[i]
    //   hi = base_address_1 + carry[i]
    //
    // Safety: `hi` is at most `base_address_1 + 1`. This never reaches 2^32
    // because the CPU table splits addresses into (lo, hi) with both halves
    // in [0, 2^32), and the Memw bus ties MEMW's base_address to the CPU's
    // value. MEMW only receives accesses where base_address_1 <= 0xFFFF_FFFE
    // (addresses near u64::MAX are rejected by the executor before proving).
    // Consequently, `carry[i]` is implicitly correct: a wrong carry bit
    // produces a memory token at a wrong address that has no matching
    // PAGE/REGISTER token, causing multiset imbalance and an invalid proof.

    // CM8: memory[is_register, base_address, old_timestamp[0], old[0]] with +μ_sum
    interactions.push(BusInteraction::sender(
        BusId::Memory,
        Multiplicity::Sum(cols::MU_READ, cols::MU_WRITE),
        vec![
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::old_timestamp(0)[0],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::old_timestamp(0)[1],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[0],
                packing: Packing::Direct,
            },
        ],
    ));

    // CM9: memory[is_register, base_address, timestamp, value[0]] with -μ_sum
    interactions.push(BusInteraction::receiver(
        BusId::Memory,
        Multiplicity::Sum(cols::MU_READ, cols::MU_WRITE),
        vec![
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_1,
                packing: Packing::Direct,
            },
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

    // CM10/11: byte 1, multiplicity w2 = write2 + write4 + write8
    // address_add[0] is virtual: lo = base_address_0 + 1 - 2^32 * carry[0]
    //                            hi = base_address_1 + carry[0]
    let addr_add_0_lo = BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::BASE_ADDRESS_0,
        },
        LinearTerm::Constant(1),
        LinearTerm::Column {
            coefficient: -(1i64 << 32),
            column: cols::CARRY[0],
        },
    ]);
    let addr_add_0_hi = BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::BASE_ADDRESS_1,
        },
        LinearTerm::Column {
            coefficient: 1,
            column: cols::CARRY[0],
        },
    ]);

    // CM10: send old token for byte 1
    interactions.push(BusInteraction::sender(
        BusId::Memory,
        Multiplicity::Sum3(cols::WRITE2, cols::WRITE4, cols::WRITE8),
        vec![
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            addr_add_0_lo.clone(),
            addr_add_0_hi.clone(),
            BusValue::Packed {
                start_column: cols::old_timestamp(1)[0],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::old_timestamp(1)[1],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::OLD[1],
                packing: Packing::Direct,
            },
        ],
    ));

    // CM11: receive new token for byte 1
    interactions.push(BusInteraction::receiver(
        BusId::Memory,
        Multiplicity::Sum3(cols::WRITE2, cols::WRITE4, cols::WRITE8),
        vec![
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            addr_add_0_lo,
            addr_add_0_hi,
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

    // CM12/13: bytes 2-3 with multiplicity w4 = write4 + write8
    for i in 2..=3 {
        let overflow_col = cols::CARRY[i - 1];
        let addr_add_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::BASE_ADDRESS_0,
            },
            LinearTerm::Constant(i as i64),
            LinearTerm::Column {
                coefficient: -(1i64 << 32),
                column: overflow_col,
            },
        ]);
        let addr_add_hi = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::BASE_ADDRESS_1,
            },
            LinearTerm::Column {
                coefficient: 1,
                column: overflow_col,
            },
        ]);

        // send old token
        interactions.push(BusInteraction::sender(
            BusId::Memory,
            Multiplicity::Sum(cols::WRITE4, cols::WRITE8),
            vec![
                BusValue::Packed {
                    start_column: cols::IS_REGISTER,
                    packing: Packing::Direct,
                },
                addr_add_lo.clone(),
                addr_add_hi.clone(),
                BusValue::Packed {
                    start_column: cols::old_timestamp(i)[0],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::old_timestamp(i)[1],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::OLD[i],
                    packing: Packing::Direct,
                },
            ],
        ));

        // receive new token
        interactions.push(BusInteraction::receiver(
            BusId::Memory,
            Multiplicity::Sum(cols::WRITE4, cols::WRITE8),
            vec![
                BusValue::Packed {
                    start_column: cols::IS_REGISTER,
                    packing: Packing::Direct,
                },
                addr_add_lo,
                addr_add_hi,
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

    // CM14/15: bytes 4-7 with multiplicity write8
    for i in 4..=7 {
        let overflow_col = cols::CARRY[i - 1];
        let addr_add_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::BASE_ADDRESS_0,
            },
            LinearTerm::Constant(i as i64),
            LinearTerm::Column {
                coefficient: -(1i64 << 32),
                column: overflow_col,
            },
        ]);
        let addr_add_hi = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::BASE_ADDRESS_1,
            },
            LinearTerm::Column {
                coefficient: 1,
                column: overflow_col,
            },
        ]);

        // send old token
        interactions.push(BusInteraction::sender(
            BusId::Memory,
            Multiplicity::Column(cols::WRITE8),
            vec![
                BusValue::Packed {
                    start_column: cols::IS_REGISTER,
                    packing: Packing::Direct,
                },
                addr_add_lo.clone(),
                addr_add_hi.clone(),
                BusValue::Packed {
                    start_column: cols::old_timestamp(i)[0],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::old_timestamp(i)[1],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::OLD[i],
                    packing: Packing::Direct,
                },
            ],
        ));

        // receive new token
        interactions.push(BusInteraction::receiver(
            BusId::Memory,
            Multiplicity::Column(cols::WRITE8),
            vec![
                BusValue::Packed {
                    start_column: cols::IS_REGISTER,
                    packing: Packing::Direct,
                },
                addr_add_lo,
                addr_add_hi,
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
    // CO16: Read receiver (from CPU)
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
            // base_address
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_1,
                packing: Packing::Direct,
            },
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
    // CO17: Write receiver (from CPU)
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
            // base_address
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_1,
                packing: Packing::Direct,
            },
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
    // ALU interactions for timestamp ordering (MEMW-C4 through C7).
    // Each lookup is dispatched on the unified ALU bus as
    // `[old_ts, ts, opsel(LT), 1, 0]` (signed=0, invert=0, asserting
    // old_ts < ts); there is no dedicated `Lt` bus.
    // -------------------------------------------------------------------------

    // MEMW-C4: old_timestamp[0] < timestamp with μ_sum
    interactions.push(BusInteraction::sender(
        BusId::Alu,
        Multiplicity::Sum(cols::MU_READ, cols::MU_WRITE),
        vec![
            BusValue::Packed {
                start_column: cols::old_timestamp(0)[0],
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

    // MEMW-C5: old_timestamp[1] < timestamp with w2
    interactions.push(BusInteraction::sender(
        BusId::Alu,
        Multiplicity::Sum3(cols::WRITE2, cols::WRITE4, cols::WRITE8),
        vec![
            BusValue::Packed {
                start_column: cols::old_timestamp(1)[0],
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

    // MEMW-C6: old_timestamp[i] < timestamp for i ∈ [2,3] with w4
    for i in 2..4 {
        interactions.push(BusInteraction::sender(
            BusId::Alu,
            Multiplicity::Sum(cols::WRITE4, cols::WRITE8),
            vec![
                BusValue::Packed {
                    start_column: cols::old_timestamp(i)[0],
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
    }

    // MEMW-C7: old_timestamp[i] < timestamp for i ∈ [4,7] with write8
    for i in 4..8 {
        interactions.push(BusInteraction::sender(
            BusId::Alu,
            Multiplicity::Column(cols::WRITE8),
            vec![
                BusValue::Packed {
                    start_column: cols::old_timestamp(i)[0],
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
    }

    interactions
}

// =========================================================================
// Single-source constraint set (ConstraintBuilder front-end)
// =========================================================================

/// `μ_sum = μ_read + μ_write` as a builder expression.
fn mu_sum_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(b: &B) -> B::Expr {
    b.main(0, cols::MU_READ) + b.main(0, cols::MU_WRITE)
}

/// `w2 = write2 + write4 + write8` as a builder expression.
fn w2_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(b: &B) -> B::Expr {
    b.main(0, cols::WRITE2) + b.main(0, cols::WRITE4) + b.main(0, cols::WRITE8)
}

/// The MEMW table's transition constraints as a single [`ConstraintSet`],
/// mirroring `constraints` index-for-index (15 constraints):
/// - idx 0:     `IS_BIT<μ_sum>`;
/// - idx 1:     `w2 ⇒ μ_sum` (`w2·(1 − μ_sum)`);
/// - idx 2,3:   `IS_BIT` on `μ_read`, `μ_write`;
/// - idx 4-10:  `IS_BIT` on `carry[0..6]`;
/// - idx 11-13: `IS_BIT` on `write2`, `write4`, `write8`;
/// - idx 14:    `IS_BIT<w2>` (width sum is a bit).
pub struct MemwConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for MemwConstraints {
    fn meta(&self) -> Vec<ConstraintMeta> {
        let mut m = Vec::with_capacity(15);
        for i in 0..15 {
            m.push(ConstraintMeta::base(i, 2));
        }
        m
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0: IS_BIT<μ_sum> = μ_sum * (1 - μ_sum)
        let one = b.one();
        let mu_sum = mu_sum_expr(b);
        b.emit_base(0, mu_sum.clone() * (one - mu_sum));

        // idx 1: w2 ⇒ μ_sum = w2 * (1 - μ_sum)
        let one = b.one();
        let w2 = w2_expr(b);
        let mu_sum = mu_sum_expr(b);
        b.emit_base(1, w2 * (one - mu_sum));

        // idx 2,3: IS_BIT<μ_read>, IS_BIT<μ_write>
        emit_is_bit(b, 2, cols::MU_READ, None);
        emit_is_bit(b, 3, cols::MU_WRITE, None);

        // idx 4-10: IS_BIT for carry[0..6]
        let mut idx = 4;
        for &col in &cols::CARRY {
            emit_is_bit(b, idx, col, None);
            idx += 1;
        }

        // idx 11-13: IS_BIT on the width flags
        for &col in &[cols::WRITE2, cols::WRITE4, cols::WRITE8] {
            emit_is_bit(b, idx, col, None);
            idx += 1;
        }

        // idx 14: IS_BIT<w2> = w2 * (1 - w2)
        let one = b.one();
        let w2 = w2_expr(b);
        b.emit_base(idx, w2.clone() * (one - w2));
    }
}
