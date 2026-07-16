//! HALT (ECALL) table for program termination.
//!
//! This is a single-row table that handles program termination via the `ecall`
//! instruction with syscall number 93 (`sys_exit`).
//!
//! ## Columns
//! - `timestamp`: Word (1 column) - timestamp at which to halt the program
//! - `pc`: DWordWL (2 columns) - the `next_pc` the CPU wrote during the halting
//!   instruction (consumed off the `memory` bus and replaced by the padding PC=1)
//!
//! ## Bus Interactions
//! - **Receiver**: ECALL bus - receives `[timestamp, cast(rv1, DWordWL)]` from CPU
//!   when the ECALL flag is set (rv1 must be 93 = sys_exit)
//! - **Sender**: MEMW bus - 31 register finalization interactions at `ts = 2^32-1`:
//!   - x1-x9: write 0 (zeroize lo GPRs)
//!   - x10: read with old=0 (enforce exit_code=0; non-zero → bus imbalance → proof failure)
//!   - x11-x31: write 0 (zeroize hi GPRs)
//! - **`memory` bus (PC finalization, per spec halt:c:consume_pc/emit_pc)**: at
//!   `ts = timestamp + 1` the chip *consumes* the real `next_pc` the CPU wrote for
//!   the halting instruction and *re-emits* `pc = 1`. This bridges the last real PC
//!   write to the CPU padding rows (which all carry PC=1); the padding chain then
//!   carries PC=1 to the REGISTER table's final token. x255 is therefore NOT
//!   finalized via MEMW at `2^32-1` anymore.
//!
//! Corresponding MEMW table rows are generated in trace_builder.
//!
//! ## Padding
//! Single-row table (2^0 = 1), no padding needed.

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, GoldilocksExtension, GoldilocksField, VmTable};

// =========================================================================
// Column indices for HALT table
// =========================================================================

/// Column definitions for the HALT table.
pub mod cols {
    /// timestamp: Word (32-bit halt timestamp)
    pub const TIMESTAMP: usize = 0;

    /// pc[0]: Word (lower 32 bits of the halting instruction's next_pc)
    pub const PC_0: usize = 1;
    /// pc[1]: Word (upper 32 bits of the halting instruction's next_pc)
    pub const PC_1: usize = 2;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 3;
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the HALT trace table from the halt timestamp.
///
/// This produces a single-row table with the timestamp stored as a single Word.
/// The HALT table expects exactly one ECALL per execution: the executor stops on the
/// first ECALL, so a valid trace always contains exactly one. If a program had multiple
/// ECALLs, the CPU would send multiple bus interactions but HALT only receives one,
/// causing a bus imbalance and proof failure.
pub fn generate_halt_trace(
    timestamp: u64,
    next_pc: u64,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    // CPU timestamps must fit in a Word (u32 range)
    debug_assert!(
        timestamp <= u32::MAX as u64,
        "HALT timestamp {timestamp} exceeds u32 range"
    );
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    table.set_u64(0, cols::TIMESTAMP, timestamp);
    table.set_dword_wl(0, cols::PC_0, next_pc);

    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Returns the 23-element MEMW read bus values for x10 exit code verification.
///
/// Format matches CO24 (read receiver): `[old[0..7], is_register, base_addr[0..1],
/// value[0..7], timestamp, write2, write4, write8]`.
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
        // input (same 15 elements as write format)
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
        BusValue::constant(0xFFFF_FFFF), // timestamp = 2^32-1
        BusValue::constant(1),           // write2 = 1
        BusValue::constant(0),           // write4 = 0
        BusValue::constant(0),           // write8 = 0
    ]
}

/// Returns the 15-element MEMW write bus values for a register finalization.
///
/// Format matches CO25 (write receiver): `[is_register, base_addr[0..1], value[0..7],
/// timestamp, write2, write4, write8]`.
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
        BusValue::constant(0xFFFF_FFFF), // timestamp = 2^32-1
        BusValue::constant(1),           // write2 = 1
        BusValue::constant(0),           // write4 = 0
        BusValue::constant(0),           // write8 = 0
    ]
}

/// Creates all bus interactions for the HALT table.
///
/// - **ECALL receiver**: receives `[timestamp, cast(rv1, DWordWL)]` from CPU
/// - **MEMW senders** (31 total): register finalization at `ts = 2^32-1`
///   - x1-x9: write 0 (zeroize lo GPRs)
///   - x10: read with old=0 (enforce exit_code=0)
///   - x11-x31: write 0 (zeroize hi GPRs)
/// - **`memory` bus (4 total)**: consume_pc (x2) + emit_pc (x2) at `ts = timestamp+1`,
///   bridging the last real PC write to the PC=1 padding chain.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(36);

    // ECALL receiver: receives [timestamp, cast(rv1, DWordWL)] from CPU
    // rv1 must be 93 (sys_exit) for bus to balance; otherwise proof fails.
    interactions.push(BusInteraction::receiver(
        BusId::Ecall,
        Multiplicity::One,
        vec![
            BusValue::Packed {
                start_column: cols::TIMESTAMP,
                packing: Packing::Direct,
            },
            BusValue::constant(93), // syscall number lo = sys_exit (93)
            BusValue::constant(0),  // syscall number hi = 0
        ],
    ));

    // x1-x9: write 0 at ts=2^32-1 (zeroize lo registers)
    for i in 1..=9u64 {
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::One,
            halt_write_bus_values(2 * i, 0),
        ));
    }

    // x10: read with old=0 at ts=2^32-1 (verify exit code = 0)
    // Per spec halt:c:read_zero_exit_code: enforces that x10 was 0 at halt.
    // Non-zero exit code → bus imbalance → proof failure.
    interactions.push(BusInteraction::sender(
        BusId::Memw,
        Multiplicity::One,
        halt_read_bus_values(20),
    ));

    // x11-x31: write 0 at ts=2^32-1 (zeroize hi registers)
    for i in 11..=31u64 {
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::One,
            halt_write_bus_values(2 * i, 0),
        ));
    }

    // PC finalization on the low-level `memory` token bus at ts = timestamp + 1
    // (per spec halt:c:consume_pc / halt:c:emit_pc). The CPU's halting row wrote
    // its real `next_pc` to x255 (addresses 510/511) at this same timestamp; we
    // consume it (sender, +1) and re-emit pc=1 (receiver, -1) so the CPU padding
    // rows — which all carry pc=1 — chain cleanly to the REGISTER final token.
    // `value` layout on the bus: [is_register, addr_lo, addr_hi, timestamp, value].
    let ts_plus_one = || {
        BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::TIMESTAMP,
            },
            LinearTerm::Constant(1),
        ])
    };
    for (addr, pc_col) in [(510u64, cols::PC_0), (511u64, cols::PC_1)] {
        // consume_pc (sender, +1): consume the real next_pc the CPU wrote.
        interactions.push(BusInteraction::sender(
            BusId::Memory,
            Multiplicity::One,
            vec![
                BusValue::constant(1),
                BusValue::constant(addr),
                BusValue::constant(0),
                ts_plus_one(),
                BusValue::Packed {
                    start_column: pc_col,
                    packing: Packing::Direct,
                },
            ],
        ));
    }
    for (addr, value) in [(510u64, 1u64), (511u64, 0u64)] {
        // emit_pc (receiver, -1): re-emit pc = 1 (value [1, 0]).
        interactions.push(BusInteraction::receiver(
            BusId::Memory,
            Multiplicity::One,
            vec![
                BusValue::constant(1),
                BusValue::constant(addr),
                BusValue::constant(0),
                ts_plus_one(),
                BusValue::constant(value),
            ],
        ));
    }

    interactions
}
