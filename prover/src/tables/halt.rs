//! HALT (ECALL) table for program termination.
//!
//! This is a single-row table that handles program termination via the `ecall`
//! instruction with syscall number 93 (`sys_exit`).
//!
//! ## Columns
//! - `timestamp`: DWordWL (2 columns) - timestamp at which to halt the program
//!
//! ## Bus Interactions
//! - **Receiver**: ECALL bus - receives `[timestamp, cast(rv1, DWordWL)]` from CPU
//!   when the ECALL flag is set (rv1 must be 93 = sys_exit)
//! - **Sender**: MEMW bus - 32 register finalization interactions at `ts = 2^64-1`:
//!   - x1-x9: write 0 (zeroize lo GPRs)
//!   - x10: read with old=0 (enforce exit_code=0; non-zero → bus imbalance → proof failure)
//!   - x11-x31: write 0 (zeroize hi GPRs)
//!   - x255: write 1 (PC halted sentinel)
//!
//! All MEMW interactions use constant values only (no additional columns needed).
//! Corresponding MEMW table rows are generated in trace_builder.
//!
//! ## Padding
//! Single-row table (2^0 = 1), no padding needed.

use alloc::vec;
use alloc::vec::Vec;
use smallvec::smallvec;
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

/// Returns the 24-element MEMW read bus values for x10 exit code verification.
///
/// Format matches CO24 (read receiver): `[old[0..7], is_register, base_addr[0..1],
/// value[0..7], timestamp[0..1], write2, write4, write8]`.
/// old=0 enforces that x10 was 0 at halt time.
fn halt_read_bus_values(base_addr: u64) -> Vec<BusValue> {
    vec![
        // old[0..7] = 0 (enforces exit code = 0)
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        // input (same 16 elements as write format)
        BusValue::constant(1),           // is_register = 1
        BusValue::constant(base_addr),   // base_address[0]
        BusValue::constant(0),           // base_address[1]
        BusValue::constant(0),           // value[0] = 0
        BusValue::constant(0),           // value[1]
        BusValue::constant(0),           // value[2]
        BusValue::constant(0),           // value[3]
        BusValue::constant(0),           // value[4]
        BusValue::constant(0),           // value[5]
        BusValue::constant(0),           // value[6]
        BusValue::constant(0),           // value[7]
        BusValue::constant(0xFFFF_FFFF), // timestamp[0] = lo(2^64-1)
        BusValue::constant(0xFFFF_FFFF), // timestamp[1] = hi(2^64-1)
        BusValue::constant(1),           // write2 = 1
        BusValue::constant(0),           // write4 = 0
        BusValue::constant(0),           // write8 = 0
    ]
}

/// Returns the 16-element MEMW write bus values for a register finalization.
///
/// Format matches CO25 (write receiver): `[is_register, base_addr[0..1], value[0..7],
/// timestamp[0..1], write2, write4, write8]`.
fn halt_write_bus_values(base_addr: u64, value_lo: u64) -> Vec<BusValue> {
    vec![
        BusValue::constant(1),           // is_register = 1
        BusValue::constant(base_addr),   // base_address[0]
        BusValue::constant(0),           // base_address[1]
        BusValue::constant(value_lo),    // value[0]
        BusValue::constant(0),           // value[1]
        BusValue::constant(0),           // value[2]
        BusValue::constant(0),           // value[3]
        BusValue::constant(0),           // value[4]
        BusValue::constant(0),           // value[5]
        BusValue::constant(0),           // value[6]
        BusValue::constant(0),           // value[7]
        BusValue::constant(0xFFFF_FFFF), // timestamp[0] = lo(2^64-1)
        BusValue::constant(0xFFFF_FFFF), // timestamp[1] = hi(2^64-1)
        BusValue::constant(1),           // write2 = 1
        BusValue::constant(0),           // write4 = 0
        BusValue::constant(0),           // write8 = 0
    ]
}

/// Creates all bus interactions for the HALT table.
///
/// - **ECALL receiver**: receives `[timestamp, cast(rv1, DWordWL)]` from CPU
/// - **MEMW senders** (32 total): register finalization at `ts = 2^64-1`
///   - x1-x9: write 0 (zeroize lo GPRs)
///   - x10: read with old=0 (enforce exit_code=0)
///   - x11-x31: write 0 (zeroize hi GPRs)
///   - x255: write 1 (PC halted sentinel)
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(33);

    // ECALL receiver: receives [timestamp, cast(rv1, DWordWL)] from CPU
    // rv1 must be 93 (sys_exit) for bus to balance; otherwise proof fails.
    interactions.push(BusInteraction::receiver(
        BusId::Ecall,
        Multiplicity::One,
        smallvec![
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            BusValue::constant(93), // syscall number lo = sys_exit (93)
            BusValue::constant(0),  // syscall number hi = 0
        ],
    ));

    // x1-x9: write 0 at ts=2^64-1 (zeroize lo registers)
    for i in 1..=9u64 {
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::One,
            halt_write_bus_values(2 * i, 0),
        ));
    }

    // x10: read with old=0 at ts=2^64-1 (verify exit code = 0)
    // Per spec halt:c:read_zero_exit_code: enforces that x10 was 0 at halt.
    // Non-zero exit code → bus imbalance → proof failure.
    interactions.push(BusInteraction::sender(
        BusId::Memw,
        Multiplicity::One,
        halt_read_bus_values(20),
    ));

    // x11-x31: write 0 at ts=2^64-1 (zeroize hi registers)
    for i in 11..=31u64 {
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::One,
            halt_write_bus_values(2 * i, 0),
        ));
    }

    // x255 (PC): write 1 at ts=2^64-1 (halted sentinel)
    interactions.push(BusInteraction::sender(
        BusId::Memw,
        Multiplicity::One,
        halt_write_bus_values(510, 1),
    ));

    interactions
}
