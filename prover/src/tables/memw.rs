//! MEMW (Memory Write/Read) table.
//!
//! This table handles memory and register read/write operations with timestamp-based
//! consistency checking.
//!
//! ## Inputs
//! - `is_register`: Bit (1 = register access, 0 = memory access)
//! - `base_address`: DWordWL (64-bit address)
//! - `value[8]`: BaseField[8] (8 bytes to write)
//! - `timestamp`: DWordWL (64-bit timestamp)
//! - `write2/4/8`: Bit (access width flags)
//!
//! ## Output
//! - `old[8]`: BaseField[8] (previous values at address)
//!
//! ## Auxiliary
//! - `address_add[7]`: DWordHL[7] (base_address + 1..7)
//! - `old_timestamp[8]`: DWordWL[8] (previous timestamps)
//!
//! ## Virtual (computed inline)
//! - `w2`: write2 + write4 + write8 (writing at least 2 bytes)
//! - `w4`: write4 + write8 (writing at least 4 bytes)
//! - `μ_sum`: μ_read + μ_write
//!
//! ## Bus Interactions
//! - Receiver: MEMW (from CPU for LOAD/STORE operations)
//! - Sender: IS_HALFWORD (range checks for address_add)
//! - Sender: LT (timestamp ordering checks)
//! - Sender/Receiver: Memory bus (internal read/write consistency)

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::TransitionConstraint;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::constraints::templates::{AddConstraint, AddOperand};

/// Maximum number of rows per MEMW table chunk.
/// If operations exceed this, the trace is split into multiple tables.
pub const MAX_ROWS: usize = super::max_rows::MEMW;

// =========================================================================
// Column indices for MEMW table
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
    /// address_add[7]: each is DWordHL (4 halfwords = 4 columns)
    /// Total: 7 * 4 = 28 columns
    pub const ADDRESS_ADD_START: usize = 24;
    // address_add[i] uses columns ADDRESS_ADD_START + i*4 .. ADDRESS_ADD_START + i*4 + 4

    /// old_timestamp[8]: each is DWordWL (2 words = 2 columns)
    /// Total: 8 * 2 = 16 columns
    pub const OLD_TIMESTAMP_START: usize = 52; // 24 + 28
    // old_timestamp[i] uses columns OLD_TIMESTAMP_START + i*2 .. OLD_TIMESTAMP_START + i*2 + 2

    // Multiplicity columns
    /// μ_read: Whether we are performing a read
    pub const MU_READ: usize = 68; // 52 + 16
    /// μ_write: Whether we are performing a write
    pub const MU_WRITE: usize = 69;

    /// Total number of columns
    /// Note: w2, w4, μ_sum are now computed inline via Multiplicity::Linear/Sum
    pub const NUM_COLUMNS: usize = 70;

    /// Helper to get address_add[i] column indices (4 halfwords each)
    pub fn address_add(i: usize) -> [usize; 4] {
        let base = ADDRESS_ADD_START + i * 4;
        [base, base + 1, base + 2, base + 3]
    }

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
    /// The spec uses three flags to encode access width:
    /// - `write2`: set if accessing 2+ bytes (width >= 2)
    /// - `write4`: set if accessing 4+ bytes (width >= 4)
    /// - `write8`: set if accessing 8 bytes (width == 8)
    ///
    /// This encoding allows computing "at least N bytes" predicates:
    /// - w2 (at least 2) = write2 + write4 + write8
    /// - w4 (at least 4) = write4 + write8
    ///
    /// | Width | write2 | write4 | write8 |
    /// |-------|--------|--------|--------|
    /// |   1   |   0    |   0    |   0    |
    /// |   2   |   1    |   0    |   0    |
    /// |   4   |   0    |   1    |   0    |
    /// |   8   |   0    |   0    |   1    |
    ///
    /// Note: These are "exactly N" semantics per spec, not cumulative.
    /// Virtual columns w2 = write2 + write4 + write8 and w4 = write4 + write8
    /// compute "at least N" from these.
    pub fn write_flags(&self) -> (bool, bool, bool) {
        match self.width {
            1 => (false, false, false),
            2 => (true, false, false),
            4 => (false, true, false),
            8 => (false, false, true),
            _ => (false, false, false),
        }
    }

    /// Collect LT operations for timestamp ordering and overflow checking.
    ///
    /// Per spec constraints #7-10 and R1-R3:
    /// - #7: old_timestamp[0] < timestamp (for all accesses)
    /// - #8: old_timestamp[1] < timestamp (for width >= 2)
    /// - #9: old_timestamp[2,3] < timestamp (for width >= 4)
    /// - #10: old_timestamp[4..7] < timestamp (for width == 8)
    /// - R1: base_address < base_address + 1 (overflow check for width >= 2)
    /// - R2: base_address < base_address + 3 (overflow check for width >= 4)
    /// - R3: base_address < base_address + 7 (overflow check for width == 8)
    pub fn collect_lt_lookups(&self) -> Vec<super::lt::LtOperation> {
        use super::lt::LtOperation;

        let mut ops = Vec::new();

        // Constraint 7: old_timestamp[0] < timestamp (always, for any access)
        ops.push(LtOperation::new(
            self.old_timestamp[0],
            self.timestamp,
            false,
        ));

        // Constraint 8: old_timestamp[1] < timestamp (for width >= 2)
        if self.width >= 2 {
            ops.push(LtOperation::new(
                self.old_timestamp[1],
                self.timestamp,
                false,
            ));
        }

        // Constraint 9: old_timestamp[2,3] < timestamp (for width >= 4)
        if self.width >= 4 {
            ops.push(LtOperation::new(
                self.old_timestamp[2],
                self.timestamp,
                false,
            ));
            ops.push(LtOperation::new(
                self.old_timestamp[3],
                self.timestamp,
                false,
            ));
        }

        // Constraint 10: old_timestamp[4..7] < timestamp (for width == 8)
        if self.width == 8 {
            for i in 4..8 {
                ops.push(LtOperation::new(
                    self.old_timestamp[i],
                    self.timestamp,
                    false,
                ));
            }
        }

        // Overflow checks R1-R3: base_address < base_address + offset
        // Always generate the LT operation - if overflow occurred, LT will return 0
        // and the constraint (expecting result=1) will fail, rejecting the proof.
        // R1: for width == 2, check base_address < base_address + 1
        if self.width == 2 {
            let addr_plus_1 = self.base_address.wrapping_add(1);
            ops.push(LtOperation::new(self.base_address, addr_plus_1, false));
        }

        // R2: for width == 4, check base_address < base_address + 3
        if self.width == 4 {
            let addr_plus_3 = self.base_address.wrapping_add(3);
            ops.push(LtOperation::new(self.base_address, addr_plus_3, false));
        }

        // R3: for width == 8, check base_address < base_address + 7
        if self.width == 8 {
            let addr_plus_7 = self.base_address.wrapping_add(7);
            ops.push(LtOperation::new(self.base_address, addr_plus_7, false));
        }

        ops
    }
}

/// Generates the MEMW trace table from a list of operations.
pub fn generate_memw_trace(
    operations: &[MemwOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = operations.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, op) in operations.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;

        // Input columns
        data[base + cols::IS_REGISTER] = FE::from(op.is_register as u64);

        // base_address as DWordWL (2 words)
        data[base + cols::BASE_ADDRESS_0] = FE::from(op.base_address & 0xFFFF_FFFF);
        data[base + cols::BASE_ADDRESS_1] = FE::from(op.base_address >> 32);

        // value[8]
        for i in 0..8 {
            data[base + cols::VALUE[i]] = FE::from(op.value[i]);
        }

        // timestamp as DWordWL (2 words)
        data[base + cols::TIMESTAMP_0] = FE::from(op.timestamp & 0xFFFF_FFFF);
        data[base + cols::TIMESTAMP_1] = FE::from(op.timestamp >> 32);

        // write flags
        let (w2, w4, w8) = op.write_flags();
        data[base + cols::WRITE2] = FE::from(w2 as u64);
        data[base + cols::WRITE4] = FE::from(w4 as u64);
        data[base + cols::WRITE8] = FE::from(w8 as u64);

        // Output: old[8]
        for i in 0..8 {
            data[base + cols::OLD[i]] = FE::from(op.old[i]);
        }

        // Auxiliary: address_add[7] - each as DWordHL (4 halfwords)
        for i in 0..7 {
            let addr = op.base_address.wrapping_add(i as u64 + 1);
            let cols_i = cols::address_add(i);
            data[base + cols_i[0]] = FE::from(addr & 0xFFFF);
            data[base + cols_i[1]] = FE::from((addr >> 16) & 0xFFFF);
            data[base + cols_i[2]] = FE::from((addr >> 32) & 0xFFFF);
            data[base + cols_i[3]] = FE::from((addr >> 48) & 0xFFFF);
        }

        // Auxiliary: old_timestamp[8] - each as DWordWL (2 words)
        for i in 0..8 {
            let cols_i = cols::old_timestamp(i);
            data[base + cols_i[0]] = FE::from(op.old_timestamp[i] & 0xFFFF_FFFF);
            data[base + cols_i[1]] = FE::from(op.old_timestamp[i] >> 32);
        }

        // Multiplicity
        data[base + cols::MU_READ] = FE::from(op.is_read as u64);
        data[base + cols::MU_WRITE] = FE::from(!op.is_read as u64);
        // Note: w2, w4, μ_sum are computed inline via Multiplicity::Linear/Sum
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the MEMW table.
///
/// The MEMW table:
/// - **Receives** MEMW lookups from CPU (for LOAD/STORE operations)
/// - **Sends** IS_HALFWORD lookups for address_add range checks
/// - **Sends** LT lookups for timestamp ordering (old_timestamp < timestamp)
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::new();

    // -------------------------------------------------------------------------
    // IS_HALFWORD range checks for address_add[i][j]
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // IsHalfword range checks for address_add columns
    // -------------------------------------------------------------------------
    // Each address_add[i] is 4 halfwords (DWordHL packing), need to range check all.
    // Only check when row is active (μ_read + μ_write > 0).
    for i in 0..7 {
        let cols_i = cols::address_add(i);
        for &col in &cols_i {
            interactions.push(BusInteraction::sender(
                BusId::IsHalfword,
                // Only range check when row is active
                Multiplicity::Sum(cols::MU_READ, cols::MU_WRITE),
                vec![BusValue::Packed {
                    start_column: col,
                    packing: Packing::Direct,
                }],
            ));
        }
    }

    // -------------------------------------------------------------------------
    // Memory bus interactions (M1-M8 from spec)
    // -------------------------------------------------------------------------
    // DISABLED: Memory bus requires initialization and finalization:
    // - Initialization: For each address accessed, an initial row at timestamp=0
    //   with the starting value must exist so the first read has a matching write.
    // - Finalization: Final values must be consumed to balance the bus.
    // Without these, the bus won't balance.
    // -------------------------------------------------------------------------
    // These ensure read/write consistency:
    // - Read old value at old_timestamp (+multiplicity)
    // - Write new value at current timestamp (-multiplicity)
    //
    // Memory bus format: memory[is_register, address, timestamp_lo, timestamp_hi, value]
    //
    // Register tokens (is_register=1) are balanced by the REGISTER table.
    // Memory tokens (is_register=0) are balanced by PAGE tables.

    // -------------------------------------------------------------------------
    // Memory bus interactions per spec CM16-CM23
    // -------------------------------------------------------------------------
    // Token format: memory[is_register, address_lo, address_hi, ts_lo, ts_hi, value]
    //
    // For registers (is_register=1): value is a Word (32-bit), address is Word-indexed
    // For memory (is_register=0): value is a Byte (8-bit), address is byte-indexed
    //
    // Multiplicities per spec:
    // - CM16/17 (index 0): μ_sum
    // - CM18/19 (index 1): w2 = write2 + write4 + write8
    // - CM20/21 (indices 2-3): w4 = write4 + write8
    // - CM22/23 (indices 4-7): write8

    // CM16: memory[is_register, base_address, old_timestamp[0], old[0]] with +μ_sum
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

    // CM17: memory[is_register, base_address, timestamp, value[0]] with -μ_sum
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

    // Helper: address_add[0] = base_address + 1, stored as DWordHL (4 halfwords)
    // Use Word2L to combine each pair of halfwords into a word
    let addr_add_0_lo = BusValue::Packed {
        start_column: cols::address_add(0)[0],
        packing: Packing::Word2L,
    };
    let addr_add_0_hi = BusValue::Packed {
        start_column: cols::address_add(0)[2],
        packing: Packing::Word2L,
    };

    // CM18: memory[is_register, address_add[0], old_timestamp[1], old[1]] with +w2
    // w2 = write2 + write4 + write8
    interactions.push(BusInteraction::sender(
        BusId::Memory,
        Multiplicity::Linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::WRITE2,
            },
            LinearTerm::Column {
                coefficient: 1,
                column: cols::WRITE4,
            },
            LinearTerm::Column {
                coefficient: 1,
                column: cols::WRITE8,
            },
        ]),
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

    // CM19: memory[is_register, address_add[0], timestamp, value[1]] with -w2
    interactions.push(BusInteraction::receiver(
        BusId::Memory,
        Multiplicity::Linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::WRITE2,
            },
            LinearTerm::Column {
                coefficient: 1,
                column: cols::WRITE4,
            },
            LinearTerm::Column {
                coefficient: 1,
                column: cols::WRITE8,
            },
        ]),
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

    // CM20/21: indices 2-3 with multiplicity w4 = write4 + write8
    for i in 2..=3 {
        let addr_add_lo = BusValue::Packed {
            start_column: cols::address_add(i - 1)[0],
            packing: Packing::Word2L,
        };
        let addr_add_hi = BusValue::Packed {
            start_column: cols::address_add(i - 1)[2],
            packing: Packing::Word2L,
        };

        // CM22.i: send old token
        interactions.push(BusInteraction::sender(
            BusId::Memory,
            Multiplicity::Linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::WRITE4,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::WRITE8,
                },
            ]),
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

        // CM23.i: receive new token
        interactions.push(BusInteraction::receiver(
            BusId::Memory,
            Multiplicity::Linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::WRITE4,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::WRITE8,
                },
            ]),
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

    // CM22/23: indices 4-7 with multiplicity write8
    for i in 4..=7 {
        let addr_add_lo = BusValue::Packed {
            start_column: cols::address_add(i - 1)[0],
            packing: Packing::Word2L,
        };
        let addr_add_hi = BusValue::Packed {
            start_column: cols::address_add(i - 1)[2],
            packing: Packing::Word2L,
        };

        // CM22.i: send old token
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

        // CM23.i: receive new token
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
    // CO24: Read receiver (unified for register and memory operations)
    // -------------------------------------------------------------------------
    // OLD and VALUE are 8 individual BaseField elements (Direct packing).
    // For registers: [lo32_word, hi32_word, 0, 0, 0, 0, 0, 0]
    // For memory: [byte0, byte1, ..., byte7]
    // Both match sender format since bus compares field elements directly.
    interactions.push(BusInteraction::receiver(
        BusId::Memw,
        Multiplicity::Column(cols::MU_READ),
        vec![
            // old[8] - output for reads (words)
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
            // base_address (DWordWL = 2 words)
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_1,
                packing: Packing::Direct,
            },
            // value[8] - direct reads (words)
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
            // timestamp (DWordWL = 2 words)
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
    // CO25: Write receiver (unified for register and memory operations)
    // -------------------------------------------------------------------------
    // VALUE is 8 individual BaseField elements (Direct packing).
    // For registers: [lo32_word, hi32_word, 0, 0, 0, 0, 0, 0]
    // For memory: [byte0, byte1, ..., byte7]
    // Both match sender format since bus compares field elements directly.
    interactions.push(BusInteraction::receiver(
        BusId::Memw,
        Multiplicity::Column(cols::MU_WRITE),
        vec![
            // is_register
            BusValue::Packed {
                start_column: cols::IS_REGISTER,
                packing: Packing::Direct,
            },
            // base_address (DWordWL = 2 words)
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_1,
                packing: Packing::Direct,
            },
            // value[8] - direct reads (words)
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
            // timestamp (DWordWL = 2 words)
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
    // LT interactions for timestamp ordering (constraints 7-10)
    // -------------------------------------------------------------------------
    // Verify old_timestamp[i] < timestamp for each accessed byte.
    // LT bus uses 2 elements per 64-bit operand: [lo32, hi32]
    // Both old_timestamp and timestamp are DWordWL, so use Packing::DWordWL.

    // Constraint 7: LT[1; old_timestamp[0], timestamp] with μ_sum
    interactions.push(BusInteraction::sender(
        BusId::Lt,
        Multiplicity::Sum(cols::MU_READ, cols::MU_WRITE),
        vec![
            // lhs = old_timestamp[0] as DWordWL (2 elements: [lo32, hi32])
            BusValue::Packed {
                start_column: cols::old_timestamp(0)[0],
                packing: Packing::DWordWL,
            },
            // rhs = timestamp as DWordWL (2 elements: [lo32, hi32])
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::DWordWL,
            },
            // signed = 0 (unsigned comparison)
            BusValue::constant(0),
            // lt = 1 (expected result: old_timestamp < timestamp)
            BusValue::constant(1),
        ],
    ));

    // Constraint 8: LT[1; old_timestamp[1], timestamp] with w2
    interactions.push(BusInteraction::sender(
        BusId::Lt,
        Multiplicity::Linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::WRITE2,
            },
            LinearTerm::Column {
                coefficient: 1,
                column: cols::WRITE4,
            },
            LinearTerm::Column {
                coefficient: 1,
                column: cols::WRITE8,
            },
        ]),
        vec![
            BusValue::Packed {
                start_column: cols::old_timestamp(1)[0],
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

    // Constraint 9: LT[1; old_timestamp[i], timestamp] for i ∈ [2,3] with w4
    for i in 2..4 {
        interactions.push(BusInteraction::sender(
            BusId::Lt,
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
                BusValue::constant(0),
                BusValue::constant(1),
            ],
        ));
    }

    // Constraint 10: LT[1; old_timestamp[i], timestamp] for i ∈ [4,7] with write8
    for i in 4..8 {
        interactions.push(BusInteraction::sender(
            BusId::Lt,
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
                BusValue::constant(0),
                BusValue::constant(1),
            ],
        ));
    }

    // -------------------------------------------------------------------------
    // LT interactions for overflow checking (constraints R1-R3)
    // -------------------------------------------------------------------------
    // Verify base_address < address_add[i] (no overflow when adding offset).
    // base_address is DWordWL, address_add[i] is DWordHL (cast to DWordWL).
    // Both packings produce 2 elements [lo32, hi32].

    // R1: LT[1; base_address, address_add[0]] with write2
    // This checks for no overflow when accessing byte 1 (width == 2)
    interactions.push(BusInteraction::sender(
        BusId::Lt,
        Multiplicity::Column(cols::WRITE2),
        vec![
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::address_add(0)[0],
                packing: Packing::DWordHL,
            },
            BusValue::constant(0), // unsigned
            BusValue::constant(1), // lt = 1
        ],
    ));

    // R2: LT[1; base_address, address_add[2]] with write4
    // This checks for no overflow when accessing byte 3 (width == 4)
    interactions.push(BusInteraction::sender(
        BusId::Lt,
        Multiplicity::Column(cols::WRITE4),
        vec![
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::address_add(2)[0],
                packing: Packing::DWordHL,
            },
            BusValue::constant(0),
            BusValue::constant(1),
        ],
    ));

    // R3: LT[1; base_address, address_add[6]] with write8
    interactions.push(BusInteraction::sender(
        BusId::Lt,
        Multiplicity::Column(cols::WRITE8),
        vec![
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::address_add(6)[0],
                packing: Packing::DWordHL,
            },
            BusValue::constant(0),
            BusValue::constant(1),
        ],
    ));

    interactions
}

// =========================================================================
// Virtual column computations
// =========================================================================

/// Compute virtual w2 = write2 + write4 + write8
fn compute_w2<F, E>(step: &TableView<F, E>) -> FieldElement<F>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let write2 = step.get_main_evaluation_element(0, cols::WRITE2).clone();
    let write4 = step.get_main_evaluation_element(0, cols::WRITE4).clone();
    let write8 = step.get_main_evaluation_element(0, cols::WRITE8).clone();
    write2 + write4 + write8
}

/// Compute virtual w4 = write4 + write8
#[allow(dead_code)]
fn compute_w4<F, E>(step: &TableView<F, E>) -> FieldElement<F>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let write4 = step.get_main_evaluation_element(0, cols::WRITE4).clone();
    let write8 = step.get_main_evaluation_element(0, cols::WRITE8).clone();
    write4 + write8
}

/// Compute virtual μ_sum = μ_read + μ_write
fn compute_mu_sum<F, E>(step: &TableView<F, E>) -> FieldElement<F>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let mu_read = step.get_main_evaluation_element(0, cols::MU_READ).clone();
    let mu_write = step.get_main_evaluation_element(0, cols::MU_WRITE).clone();
    mu_read + mu_write
}

// =========================================================================
// Constraints
// =========================================================================

/// MEMW table constraint kinds.
#[derive(Debug, Clone, Copy)]
pub enum MemwConstraintKind {
    /// IS_BIT<μ_sum>: multiplicity sum is 0 or 1
    MuSumIsBit,
    /// w2 => μ_sum: if accessing 2+ bytes, must be active row
    W2ImpliesMuSum,
}

/// MEMW table constraint.
pub struct MemwConstraint {
    constraint_idx: usize,
    kind: MemwConstraintKind,
}

impl MemwConstraint {
    pub fn new(kind: MemwConstraintKind, constraint_idx: usize) -> Self {
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

        match self.kind {
            MemwConstraintKind::MuSumIsBit => {
                // IS_BIT<μ_sum>: μ_sum * (1 - μ_sum) = 0
                let mu_sum = compute_mu_sum(step);
                &mu_sum * (&one - &mu_sum)
            }
            MemwConstraintKind::W2ImpliesMuSum => {
                // w2 * (1 - μ_sum) = 0
                let w2 = compute_w2(step);
                let mu_sum = compute_mu_sum(step);
                &w2 * (&one - &mu_sum)
            }
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for MemwConstraint {
    fn degree(&self) -> usize {
        match self.kind {
            MemwConstraintKind::MuSumIsBit => 2,
            MemwConstraintKind::W2ImpliesMuSum => 2,
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
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value.to_extension();
            }
            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _,
                rap_challenges: _,
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value;
            }
        }
    }
}

/// Creates all constraints for the MEMW table.
pub fn constraints() -> Vec<Box<dyn TransitionConstraint<GoldilocksField, GoldilocksExtension>>> {
    let mut constraints: Vec<Box<dyn TransitionConstraint<GoldilocksField, GoldilocksExtension>>> =
        Vec::new();

    let mut idx = 0;

    // IS_BIT<μ_sum>
    constraints.push(Box::new(MemwConstraint::new(
        MemwConstraintKind::MuSumIsBit,
        idx,
    )));
    idx += 1;

    // w2 => μ_sum
    constraints.push(Box::new(MemwConstraint::new(
        MemwConstraintKind::W2ImpliesMuSum,
        idx,
    )));
    idx += 1;

    // ADD constraints for address_add[0..6]
    // Each ADD constraint verifies: address_add[i] = base_address + (i+1)
    // Multiplicities per spec:
    //   C3: address_add[0] with w2 (write2 + write4 + write8)
    //   C4: address_add[1..2] with w4 (write4 + write8)
    //   C5: address_add[3..6] with write8
    for i in 0..7 {
        let lhs = AddOperand::dword(cols::BASE_ADDRESS_0);
        let rhs = AddOperand::constant((i + 1) as i64);
        let sum = AddOperand::from_dword_hl(cols::address_add(i)[0]);

        // Select multiplicity based on which address_add we're constraining
        let condition = match i {
            0 => vec![cols::WRITE2, cols::WRITE4, cols::WRITE8], // w2
            1 | 2 => vec![cols::WRITE4, cols::WRITE8],           // w4
            _ => vec![cols::WRITE8],                             // write8 (i = 3..6)
        };

        // ADD constraint produces 2 constraints (carry_0, carry_1)
        let (c0, c1) = AddConstraint::new_pair(condition, lhs, rhs, sum, idx);
        constraints.push(Box::new(c0));
        constraints.push(Box::new(c1));
        idx += 2;
    }

    constraints
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memw_trace_generation() {
        let ops = vec![
            MemwOperation::new(false, 0x1000, [1, 2, 3, 4, 5, 6, 7, 8], 100, 8, false)
                .with_old([0; 8], [50; 8]),
            MemwOperation::new(true, 5, [42, 0, 0, 0, 0, 0, 0, 0], 200, 1, true)
                .with_old([10, 0, 0, 0, 0, 0, 0, 0], [150, 0, 0, 0, 0, 0, 0, 0]),
        ];

        let trace = generate_memw_trace(&ops);
        assert_eq!(trace.num_cols(), cols::NUM_COLUMNS);
        assert!(trace.num_rows() >= 2);
    }

    #[test]
    fn test_write_flags() {
        // "Exactly N" semantics per spec
        let op1 = MemwOperation::new(false, 0, [0; 8], 0, 1, false);
        assert_eq!(op1.write_flags(), (false, false, false)); // no flags for 1 byte

        let op2 = MemwOperation::new(false, 0, [0; 8], 0, 2, false);
        assert_eq!(op2.write_flags(), (true, false, false)); // write2 only

        let op4 = MemwOperation::new(false, 0, [0; 8], 0, 4, false);
        assert_eq!(op4.write_flags(), (false, true, false)); // write4 only

        let op8 = MemwOperation::new(false, 0, [0; 8], 0, 8, false);
        assert_eq!(op8.write_flags(), (false, false, true)); // write8 only
    }
}
