//! HALT (ECALL) table for program termination.
//!
//! This is a single-row table that handles program termination via the `ecall`
//! instruction with syscall number 93 (`sys_exit`).
//!
//! ## Columns
//! - `timestamp`: DWordWL (2 columns) - timestamp at which to halt the program
//!
//! ## Bus Interactions
//! - **Receiver**: ECALL bus - receives `[timestamp_lo, timestamp_hi]` from CPU
//!   when the ECALL flag is set
//!
//! ## Memory Interactions (deferred)
//! The spec requires 33 memory interactions (1 read for x10, 31 writes for
//! other registers, 1 write for pc), all at timestamp `2^64-1`. These are
//! deferred until the Memory bus is fully implemented (requires memory_init
//! and memory_final tables).
//!
//! ## Padding
//! Single-row table (2^0 = 1), no padding needed.

use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices for HALT table
// =========================================================================

/// Column definitions for the HALT table.
pub mod cols {
    /// timestamp[0]: Word (lower 32 bits of halt timestamp)
    pub const TIMESTAMP_0: usize = 0;
    /// timestamp[1]: Word (upper 32 bits of halt timestamp)
    pub const TIMESTAMP_1: usize = 1;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 2;
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the HALT trace table from the halt timestamp.
///
/// This produces a single-row table with the timestamp split into DWordWL format.
/// The HALT table expects exactly one ECALL per execution: the executor stops on the
/// first ECALL, so a valid trace always contains exactly one. If a program had multiple
/// ECALLs, the CPU would send multiple bus interactions but HALT only receives one,
/// causing a bus imbalance and proof failure.
pub fn generate_halt_trace(timestamp: u64) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    // CPU timestamps must fit in u32 (timestamp_hi should be 0)
    debug_assert!(
        timestamp <= u32::MAX as u64,
        "HALT timestamp {timestamp} exceeds u32 range"
    );
    let timestamp_lo = timestamp & 0xFFFF_FFFF;
    let timestamp_hi = timestamp >> 32;

    let data = vec![FE::from(timestamp_lo), FE::from(timestamp_hi)];

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the HALT table.
///
/// The HALT table:
/// - **Receives** ECALL from CPU with `[timestamp_lo, timestamp_hi]`
///
/// Memory interactions (33 total: read x10, write x0-x9/x11-x31, write pc)
/// are deferred until the Memory bus is fully implemented.
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // ECALL receiver: receives [timestamp] from CPU when ECALL flag is set
        BusInteraction::receiver(
            BusId::Ecall,
            Multiplicity::One,
            vec![
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
            ],
        ),
    ]
}
