//! HALT (ECALL) table for program termination.
//!
//! This is a single-row table that handles program termination via the `ecall`
//! instruction with syscall number 93 (`sys_exit`).
//!
//! ## Columns
//! - `timestamp`: DWordWL (2 columns) - timestamp at which to halt the program
//!
//! ## Bus Interactions
//! - **Receiver**: ECALL bus - receives `[timestamp_lo, timestamp_hi, 93, 0]` from CPU
//! - **Senders**: 33 MEMW bus interactions for register finalization at timestamp 2^64-1
//!   - 30 writes: registers x1-x9, x11-x31 (value = 0)
//!   - 1 read: register x10/a0 (asserts exit code = 0)
//!   - 1 write: register x255/PC (value = 1)
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
// Constants
// =========================================================================

/// Timestamp 2^64 - 1 split into two 32-bit words.
const TS_MAX_LO: u64 = 0xFFFF_FFFF;
const TS_MAX_HI: u64 = 0xFFFF_FFFF;

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

/// Creates a MEMW read sender (24 elements) for register finalization.
///
/// Format: [old[8], is_register, base_addr_lo, base_addr_hi, value[8], ts_lo, ts_hi, write2, write4, write8]
///
/// The `old` values assert what the register's previous value was.
/// Used for x10 (a0) to enforce exit code = 0.
fn memw_register_read(base_addr_lo: u64, old_lo: u64, old_hi: u64, value_lo: u64, value_hi: u64) -> BusInteraction {
    BusInteraction::sender(
        BusId::Memw,
        Multiplicity::One,
        vec![
            // old[0..7] (asserted previous value)
            BusValue::constant(old_lo),
            BusValue::constant(old_hi),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // is_register = 1
            BusValue::constant(1),
            // base_address[0], base_address[1]
            BusValue::constant(base_addr_lo),
            BusValue::constant(0),
            // value[0..7]
            BusValue::constant(value_lo),
            BusValue::constant(value_hi),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // timestamp[0], timestamp[1]
            BusValue::constant(TS_MAX_LO),
            BusValue::constant(TS_MAX_HI),
            // write2=1, write4=0, write8=0 (register access = 2 Words)
            BusValue::constant(1),
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    )
}

/// Creates a MEMW write sender (16 elements) for register finalization.
///
/// Format: [is_register, base_addr_lo, base_addr_hi, value[8], ts_lo, ts_hi, write2, write4, write8]
fn memw_register_write(base_addr_lo: u64, value_lo: u64, value_hi: u64) -> BusInteraction {
    BusInteraction::sender(
        BusId::Memw,
        Multiplicity::One,
        vec![
            // is_register = 1
            BusValue::constant(1),
            // base_address[0], base_address[1]
            BusValue::constant(base_addr_lo),
            BusValue::constant(0),
            // value[0..7]
            BusValue::constant(value_lo),
            BusValue::constant(value_hi),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::constant(0),
            // timestamp[0], timestamp[1]
            BusValue::constant(TS_MAX_LO),
            BusValue::constant(TS_MAX_HI),
            // write2=1, write4=0, write8=0 (register access = 2 Words)
            BusValue::constant(1),
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    )
}

/// Creates all bus interactions for the HALT table.
///
/// The HALT table:
/// - **Receives** ECALL from CPU with `[timestamp_lo, timestamp_hi, 93, 0]`
/// - **Sends** 33 MEMW register finalization interactions at timestamp 2^64-1:
///   - x1-x9: write value=0 (9 writes, 16 elements each)
///   - x10: read with old=0 (asserts exit code = 0, 24 elements)
///   - x11-x31: write value=0 (21 writes, 16 elements each)
///   - x255 (PC): write value=1 (1 write, 16 elements)
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(34);

    // ECALL receiver: receives [timestamp_lo, timestamp_hi, 93] from CPU
    interactions.push(BusInteraction::receiver(
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
            BusValue::constant(93), // syscall number (sys_exit) — DWordWL[0]
            BusValue::constant(0),  // syscall number hi — DWordWL[1]
        ],
    ));

    // 31 MEMW senders for register finalization at timestamp 2^64-1
    // (x0 is excluded per spec — hardwired to zero, never written by CPU)

    // halt:c:zeroize_registers_lo — x1 through x9: write value=0
    for reg in 1u64..10 {
        interactions.push(memw_register_write(2 * reg, 0, 0));
    }

    // halt:c:read_zero_exit_code — x10 (a0): read asserting old=0 (exit code must be 0)
    interactions.push(memw_register_read(20, 0, 0, 0, 0));

    // halt:c:zeroize_registers_hi — x11 through x31: write value=0
    for reg in 11u64..32 {
        interactions.push(memw_register_write(2 * reg, 0, 0));
    }

    // x255 (PC register): write value=1 at addr 510
    interactions.push(memw_register_write(510, 1, 0));

    interactions
}
