//! HINT table — receiver for the non-constraining `hint` ecall (BENCH ONLY).
//!
//! The `hint` ecall (syscall `u64::MAX - 20`) lets the executor hand the guest a
//! value that is expensive to compute but cheap to verify (modular inverse, sqrt,
//! …); the guest verifies it with ordinary constrained instructions. Unlike a
//! normal `STORE`, the ecall writes the 32-byte output to guest memory *directly*
//! (not through the CPU load/store decode), so those writes are invisible to the
//! CPU op stream — this table is what puts them into the memory argument.
//!
//! The table therefore does exactly two things, and constrains **nothing** about
//! the hinted value (that is the point — soundness lives in the guest's verify):
//!
//! 1. **Receives** the `Hint` ecall on the `Ecall` bus (balances the CPU's send;
//!    a syscall with no receiver leaves the LogUp argument unbalanced).
//! 2. **Sends** the four 8-byte MEMW writes of the output at `out_addr` +0/8/16/24
//!    (received by the MEMW table). Without these the output's initial→final
//!    memory chain is unexplained and the memory argument fails to balance.
//!
//! The input read (the ecall also reads `in_addr`) is intentionally **not** modeled:
//! a read leaves the value unchanged, the guest supplies the input via ordinary
//! stores, and nothing depends on the ecall having re-read it — so omitting it is
//! sound and avoids the mixed-timestamp bookkeeping of a partial-buffer read.
//!
//! ## Columns (37)
//! - `timestamp[0..1]` (DWordWL): the ecall timestamp `T`
//! - `out_addr[0..1]` (DWordWL): base address of the 32-byte output buffer
//! - `out_bytes[0..31]`: the 32 output bytes (the hint) — **unconstrained**
//! - `mu`: multiplicity flag (1 = real hint call, 0 = padding) — gates every bus

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};

/// `hint` ecall syscall number — must match
/// `executor::vm::instruction::execution::HINT_SYSCALL_NUMBER`.
const HINT_SYSCALL_NUMBER: u64 = u64::MAX - 20;

pub mod cols {
    /// timestamp[0]: lower 32 bits of the ecall timestamp
    pub const TIMESTAMP_0: usize = 0;
    /// timestamp[1]: upper 32 bits (always 0 — timestamps fit u32)
    pub const TIMESTAMP_1: usize = 1;
    /// out_addr[0]: lower 32 bits of the output base address
    pub const ADDR_OUT_0: usize = 2;
    /// out_addr[1]: upper 32 bits of the output base address
    pub const ADDR_OUT_1: usize = 3;
    /// out_bytes[0..31]: the 32 output bytes, one per column
    pub const OUT: usize = 4;
    /// multiplicity flag (1 = real hint call, 0 = padding)
    pub const MU: usize = 36;

    pub const NUM_COLUMNS: usize = 37;

    /// Column of output byte `i` (0..32).
    #[inline]
    pub const fn out(i: usize) -> usize {
        OUT + i
    }
}

/// One `hint` ecall: the timestamp, the output base address, and the 32 output
/// bytes the executor wrote to guest memory (recomputed by the trace builder).
#[derive(Debug, Clone)]
pub struct HintOperation {
    pub timestamp: u64,
    pub out_addr: u64,
    pub out_bytes: [u8; 32],
}

/// Generates the HINT trace: one row per hint-ecall call (in program order),
/// `mu = 1`; padding rows are all-zero (`mu = 0`, inert on the bus). Empty (all
/// padding) for programs that make no hint calls.
pub fn generate_hint_trace(
    ops: &[HintOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = ops.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row, op) in ops.iter().enumerate() {
        debug_assert!(
            op.timestamp <= u32::MAX as u64,
            "HINT timestamp {} exceeds u32",
            op.timestamp
        );
        table.set_dword_wl(row, cols::TIMESTAMP_0, op.timestamp);
        table.set_dword_wl(row, cols::ADDR_OUT_0, op.out_addr);
        table.set_bytes(row, cols::OUT, &op.out_bytes);
        table.set_fe(row, cols::MU, FE::one());
    }

    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

fn packed(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    }
}

/// The eight output bytes of doubleword `chunk` (`out_bytes[8*chunk .. 8*chunk+7]`)
/// as MEMW value elements.
fn out_dword_bytes(chunk: usize) -> [BusValue; 8] {
    std::array::from_fn(|b| packed(cols::out(8 * chunk + b)))
}

/// A 16-element MEMW **write** tuple (CO25): `[is_register=0, base_lo, base_hi,
/// value[8], ts_lo, ts_hi, w2=0, w4=0, w8=1]`. The MEMW table supplies `old`.
fn memw_write(value: [BusValue; 8], base_lo: BusValue, base_hi: BusValue) -> Vec<BusValue> {
    let mut v = Vec::with_capacity(16);
    v.push(BusValue::constant(0)); // is_register = 0 (memory)
    v.push(base_lo);
    v.push(base_hi);
    v.extend(value);
    v.push(packed(cols::TIMESTAMP_0)); // ts_lo
    v.push(packed(cols::TIMESTAMP_1)); // ts_hi
    v.push(BusValue::constant(0)); // w2
    v.push(BusValue::constant(0)); // w4
    v.push(BusValue::constant(1)); // w8 = 1 (8-byte write)
    v
}

/// Bus interactions:
/// - **`Ecall` receiver** (mult `mu`): `[timestamp, cast(HINT_SYSCALL_NUMBER,
///   DWordWL)]` — HALT-shaped, balances the CPU's ECALL send.
/// - **MEMW write senders** (mult `mu`, ×4): the four 8-byte writes of the output
///   at `out_addr` +0/8/16/24, timestamp `T`. Received by the MEMW table.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mu = || Multiplicity::Column(cols::MU);
    let mut out = Vec::with_capacity(5);

    // ECALL receiver: [ts_lo, ts_hi, syscall_lo32, syscall_hi32].
    out.push(BusInteraction::receiver(
        BusId::Ecall,
        mu(),
        vec![
            packed(cols::TIMESTAMP_0),
            packed(cols::TIMESTAMP_1),
            BusValue::constant(HINT_SYSCALL_NUMBER & 0xFFFF_FFFF),
            BusValue::constant(HINT_SYSCALL_NUMBER >> 32),
        ],
    ));

    // write output: 4 doublewords at out_addr + 8i (timestamp T).
    for i in 0..4 {
        let base_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::ADDR_OUT_0,
            },
            LinearTerm::Constant((8 * i) as i64),
        ]);
        out.push(BusInteraction::sender(
            BusId::Memw,
            mu(),
            memw_write(out_dword_bytes(i), base_lo, packed(cols::ADDR_OUT_1)),
        ));
    }

    out
}
