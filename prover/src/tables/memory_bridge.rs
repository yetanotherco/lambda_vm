//! MemoryBridge table for segment boundary state in continuation proofs.
//!
//! When execution is split into segments, each segment's MEMW and PAGE tables
//! have a bus mismatch: MEMW's first access sends old state from the checkpoint
//! (previous segment's final state), but PAGE-C3 receives the ELF initial state.
//!
//! The MemoryBridge table resolves this by providing two interactions per modified
//! address:
//!
//! - **SENDER** on Memory bus: `[is_register, addr, 0, 0, init_val]` — matches
//!   PAGE-C3's receive of initial ELF state
//! - **RECEIVER** on Memory bus: `[is_register, addr, ckpt_ts, ckpt_val]` — matches
//!   MEMW's send of old state from checkpoint (or PAGE-C4's send for unaccessed addresses)
//!
//! ## When to include
//!
//! Only for segments >0. Segment 0 uses ELF state directly and doesn't need bridging.
//! Rows are generated for every byte address and register word where the checkpoint
//! state differs from the ELF/default initial state.
//!
//! ## Columns
//!
//! | Column          | Type     | Description                              |
//! |-----------------|----------|------------------------------------------|
//! | is_register     | Bit      | 1 = register, 0 = memory                |
//! | addr_lo         | Word     | Address low 32 bits                      |
//! | addr_hi         | Word     | Address high 32 bits                     |
//! | init_val        | BaseField| ELF/default initial value                |
//! | checkpoint_ts_lo| Word     | Checkpoint timestamp low 32 bits         |
//! | checkpoint_ts_hi| Word     | Checkpoint timestamp high 32 bits        |
//! | checkpoint_val  | BaseField| Checkpoint value                         |
//! | mu              | Bit      | 1 for real rows, 0 for padding           |
//!
//! ## Bus Interactions
//!
//! | Tag | Bus    | Signature                                     | Multiplicity |
//! |-----|--------|-----------------------------------------------|--------------|
//! | B1  | Memory | `[is_register, addr, 0, 0, init_val]`         | +mu (sender) |
//! | B2  | Memory | `[is_register, addr, ckpt_ts, ckpt_val]`      | -mu (receiver)|

use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices
// =========================================================================

/// Column definitions for the MemoryBridge table.
pub mod cols {
    /// is_register: 1 for register, 0 for memory
    pub const IS_REGISTER: usize = 0;
    /// Address low 32 bits
    pub const ADDR_LO: usize = 1;
    /// Address high 32 bits
    pub const ADDR_HI: usize = 2;
    /// Initial value from ELF (memory) or default (register)
    pub const INIT_VAL: usize = 3;
    /// Checkpoint timestamp low 32 bits
    pub const CHECKPOINT_TS_LO: usize = 4;
    /// Checkpoint timestamp high 32 bits
    pub const CHECKPOINT_TS_HI: usize = 5;
    /// Checkpoint value
    pub const CHECKPOINT_VAL: usize = 6;
    /// Multiplicity: 1 for real rows, 0 for padding
    pub const MU: usize = 7;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 8;
}

// =========================================================================
// Bridge entry (public interface for trace generation)
// =========================================================================

/// A single bridge entry representing one modified address.
///
/// Computed by comparing checkpoint state with ELF/default initial state.
#[derive(Debug, Clone)]
pub struct BridgeEntry {
    /// 1 for register, 0 for memory
    pub is_register: bool,
    /// Address (byte address for memory, word address for registers)
    pub addr: u64,
    /// Initial value from ELF (for memory bytes) or default (for register words)
    pub init_val: u64,
    /// Checkpoint timestamp
    pub checkpoint_ts: u64,
    /// Checkpoint value
    pub checkpoint_val: u64,
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the MemoryBridge trace table from bridge entries.
///
/// Each entry becomes one row with `mu=1`. Remaining rows are padding with `mu=0`.
/// The table is padded to at least 4 rows (power of 2) for FRI.
pub fn generate_memory_bridge_trace(
    entries: &[BridgeEntry],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_real_rows = entries.len();
    let num_rows = num_real_rows.max(4).next_power_of_two();
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row, entry) in entries.iter().enumerate() {
        let base = row * cols::NUM_COLUMNS;
        let addr_lo = entry.addr & 0xFFFF_FFFF;
        let addr_hi = entry.addr >> 32;
        let ts_lo = entry.checkpoint_ts & 0xFFFF_FFFF;
        let ts_hi = entry.checkpoint_ts >> 32;

        data[base + cols::IS_REGISTER] = FE::from(entry.is_register as u64);
        data[base + cols::ADDR_LO] = FE::from(addr_lo);
        data[base + cols::ADDR_HI] = FE::from(addr_hi);
        data[base + cols::INIT_VAL] = FE::from(entry.init_val);
        data[base + cols::CHECKPOINT_TS_LO] = FE::from(ts_lo);
        data[base + cols::CHECKPOINT_TS_HI] = FE::from(ts_hi);
        data[base + cols::CHECKPOINT_VAL] = FE::from(entry.checkpoint_val);
        data[base + cols::MU] = FE::one();
    }
    // Padding rows remain all zeros (mu=0), producing no bus interactions.

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the MemoryBridge table.
///
/// - B1 (SENDER): Memory[is_register, addr_lo, addr_hi, 0, 0, init_val]
///   Balances PAGE-C3 / REGISTER's receive of ELF/default initial state.
///
/// - B2 (RECEIVER): Memory[is_register, addr_lo, addr_hi, ckpt_ts_lo, ckpt_ts_hi, ckpt_val]
///   Balances MEMW's send of old state from checkpoint, or PAGE-C4's send
///   for addresses not accessed in this segment.
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // B1: Send initial ELF/default state token → balances PAGE-C3 / REGISTER receive
        BusInteraction::sender(
            BusId::Memory,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: cols::IS_REGISTER,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ADDR_LO,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ADDR_HI,
                    packing: Packing::Direct,
                },
                // timestamp = 0 (initial state)
                BusValue::constant(0),
                BusValue::constant(0),
                // value = init_val (from ELF or default)
                BusValue::Packed {
                    start_column: cols::INIT_VAL,
                    packing: Packing::Direct,
                },
            ],
        ),
        // B2: Receive checkpoint state token → balances MEMW's old-state send or PAGE-C4 send
        BusInteraction::receiver(
            BusId::Memory,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: cols::IS_REGISTER,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ADDR_LO,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ADDR_HI,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::CHECKPOINT_TS_LO,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::CHECKPOINT_TS_HI,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::CHECKPOINT_VAL,
                    packing: Packing::Direct,
                },
            ],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bus_interactions_count() {
        let interactions = bus_interactions();
        assert_eq!(interactions.len(), 2); // B1 (sender) + B2 (receiver)
    }

    #[test]
    fn test_generate_empty_trace() {
        let trace = generate_memory_bridge_trace(&[]);
        assert_eq!(trace.num_rows(), 4); // Minimum for FRI
        // All mu=0 (padding)
        assert_eq!(*trace.main_table.get(0, cols::MU), FE::zero());
    }

    #[test]
    fn test_generate_trace_with_entries() {
        let entries = vec![
            BridgeEntry {
                is_register: false,
                addr: 0x1000,
                init_val: 0xAA,
                checkpoint_ts: 100,
                checkpoint_val: 0xBB,
            },
            BridgeEntry {
                is_register: true,
                addr: 4, // register x2 lo word
                init_val: 0x1234,
                checkpoint_ts: 50,
                checkpoint_val: 0x5678,
            },
        ];

        let trace = generate_memory_bridge_trace(&entries);
        assert_eq!(trace.num_rows(), 4); // 2 entries padded to 4

        // Check first entry (memory)
        assert_eq!(*trace.main_table.get(0, cols::IS_REGISTER), FE::zero());
        assert_eq!(
            *trace.main_table.get(0, cols::ADDR_LO),
            FE::from(0x1000u64)
        );
        assert_eq!(*trace.main_table.get(0, cols::INIT_VAL), FE::from(0xAAu64));
        assert_eq!(
            *trace.main_table.get(0, cols::CHECKPOINT_TS_LO),
            FE::from(100u64)
        );
        assert_eq!(
            *trace.main_table.get(0, cols::CHECKPOINT_VAL),
            FE::from(0xBBu64)
        );
        assert_eq!(*trace.main_table.get(0, cols::MU), FE::one());

        // Check second entry (register)
        assert_eq!(*trace.main_table.get(1, cols::IS_REGISTER), FE::one());
        assert_eq!(*trace.main_table.get(1, cols::MU), FE::one());

        // Check padding rows
        assert_eq!(*trace.main_table.get(2, cols::MU), FE::zero());
        assert_eq!(*trace.main_table.get(3, cols::MU), FE::zero());
    }
}
