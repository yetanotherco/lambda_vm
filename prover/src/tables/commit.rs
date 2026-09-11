//! COMMIT (ECALL) table for writing bytes to stdout.
//!
//! This table handles the `write` syscall (ECALL #64): writing bytes from a memory
//! buffer to stdout. It is **one row per ECALL** — it accepts the syscall number,
//! reads the operand registers and advances the committed-length register, then
//! defers the byte copying itself to the MEMMOVE chip over `BusId::CommitDefer`.
//!
//! The per-byte recursion this table used to run, and its self-referencing
//! `CommitNextByte` bus, are gone: MEMMOVE walks the buffer instead, and it — not
//! this table — is what sends the committed bytes on `BusId::Commit`, as eight
//! `(index, value)` pairs per row. That is the fact to keep in mind when reasoning
//! about the verifier, which rebuilds that bus from `public_output`
//! (`compute_commit_bus_offset`).
//!
//! ## Columns (8 total)
//! - `timestamp`: DWordWL (2 cols) — timestamp of the ECALL
//! - `index`: BaseField (1 col) — global byte index the committed range starts at
//! - `address`: DWordWL (2 cols) — buffer address the committed range starts at
//! - `count`: DWordWL (2 cols) — number of bytes this ECALL commits
//! - `mu`: Bit — multiplicity (1 for real rows, 0 for padding)
//!
//! There is no `first` column: one row per ECALL means a real row is always the first
//! row of its commit, so `first` was identically `mu` and every multiplicity reads
//! `mu` instead. `address_incr`, `count_decr`, `end` and `value` modelled the per-byte
//! sequence and went with it.
//!
//! ## Bus Interactions (6 total)
//! - **Receiver**: Ecall bus — receives `[timestamp_lo, timestamp_hi, constant(64), constant(0)]` from CPU (mult = mu)
//! - **Sender**: CommitDefer bus — hands the byte loop to MEMMOVE (mult = mu)
//! - **Sender**: Memw bus — read+write x10 register (fd=1→count) at ts (mult = mu)
//! - **Sender**: Memw bus — read x11 register (buf_addr) at ts (mult = mu)
//! - **Sender**: Memw bus — read x12 register (count) at ts (mult = mu)
//! - **Sender**: Memw bus — read+write x254 commit index at ts (mult = mu)
//!
//! The per-byte `Memw` read and the `Commit` `(index, value)` sender moved to MEMMOVE,
//! which sends the committed bytes itself. The eight `IsHalfword` range checks and the
//! `Zero` end-detection went with the columns they checked. `CommitNextByte` is retired
//! (bus id 20 is now a reserved hole). The count is pinned by
//! `commit_tests::test_bus_interactions_count`.
//!
//! ## Constraints (1 total)
//! - `range_mu`: mu * (1 - mu) = 0 (degree 2)
//!
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

use crate::constraints::templates::emit_is_bit;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};

// =========================================================================
// Column indices for COMMIT table
// =========================================================================

/// Column definitions for the COMMIT table.
///
/// Layout follows the spec order: timestamp, index, address, address_incr,
/// count, count_decr, first, end, value, mu.
pub mod cols {
    // Timestamp (DWordWL: 2 cols)
    /// timestamp[0]: low 32 bits
    pub const TIMESTAMP_0: usize = 0;
    /// timestamp[1]: high 32 bits
    pub const TIMESTAMP_1: usize = 1;

    // Commit index (BaseField: 1 col)
    /// index: global byte index the committed range starts at
    pub const INDEX: usize = 2;

    // Buffer address (DWordWL: 2 cols)
    /// address[0]: low 32 bits
    pub const ADDRESS_0: usize = 3;
    /// address[1]: high 32 bits
    pub const ADDRESS_1: usize = 4;

    // Byte count (DWordWL: 2 cols)
    /// count[0]: low 32 bits
    pub const COUNT_0: usize = 5;
    /// count[1]: high 32 bits
    pub const COUNT_1: usize = 6;

    /// mu: multiplicity bit (1 for real rows, 0 for padding)
    ///
    /// There is no `first` column any more. This table is one row per ECALL, so a
    /// real row is always the first row of its commit, and `first` was identically
    /// `mu`; every multiplicity that used to read `first` reads `mu` instead.
    pub const MU: usize = 7;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 8;
}

// =========================================================================
// Operation type
// =========================================================================

/// A single row in the COMMIT table.
///
/// One row per commit ECALL. It accepts the syscall number, reads the operands and
/// advances the committed-length register; MEMMOVE walks the buffer.
#[derive(Debug, Clone)]
pub struct CommitOperation {
    /// Timestamp of the originating ECALL
    pub timestamp: u64,
    /// Global commit index the committed range starts at
    pub index: u64,
    /// Buffer address the committed range starts at
    pub address: u64,
    /// Number of bytes this ECALL commits
    pub count: u64,
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the COMMIT trace table from a list of operations.
///
/// Each operation becomes one row. The table is padded to the next power of 2 (min 4).
/// Padding rows use spec-defined values: count=1, address_incr=[1,0,0,0] to satisfy
/// the unconditional ADD/SUB template constraints.
pub fn generate_commit_trace(
    ops: &[CommitOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, op) in ops.iter().enumerate() {
        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);
        table.set_u64(row_idx, cols::INDEX, op.index);
        table.set_dword_wl(row_idx, cols::ADDRESS_0, op.address);
        table.set_dword_wl(row_idx, cols::COUNT_0, op.count);
        table.set_fe(row_idx, cols::MU, FE::one());
    }

    // Padding rows are all-zero. The ADD/SUB templates that used to force a
    // non-zero padding row went with `address_incr` and `count_decr`; the one
    // surviving constraint is `IS_BIT(mu)`, which zero satisfies.

    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the COMMIT table (6 total).
///
/// The COMMIT table:
/// - **Receives** Ecall from CPU with `[timestamp_lo, timestamp_hi, constant(64), constant(0)]` (mult = mu)
/// - **Sends** to CommitDefer, handing the byte loop to MEMMOVE (mult = mu)
/// - **Sends** to Memw for register accesses (×4, mult = mu)
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // 1. Receive ECALL from CPU (mult = first)
        // Payload: [timestamp_lo, timestamp_hi, syscall_lo32, syscall_hi32]
        BusInteraction::receiver(
            BusId::Ecall,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                BusValue::constant(64), // syscall number lo32 = Commit (64)
                BusValue::constant(0),  // syscall number hi32 = 0
            ],
        ),
        // 2. Defer the byte loop to the MEMMOVE chip. COMMIT keeps the sys_write
        //    ecall number and the register-254 update; the copying is handed over.
        BusInteraction::sender(
            BusId::CommitDefer,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ADDRESS_0,
                    packing: Packing::DWordWL,
                },
                // `dst` on the MEMMOVE side is a DWordWL, i.e. two bus elements; the
                // COMMIT-domain address is the index, whose high word is always zero.
                BusValue::linear(vec![LinearTerm::Column {
                    coefficient: 1,
                    column: cols::INDEX,
                }]),
                BusValue::constant(0),
                BusValue::Packed {
                    start_column: cols::COUNT_0,
                    packing: Packing::DWordWL,
                },
            ],
        ),
        // 13. MEMW read+write x10 (fd=1 → count) at ts (mult = first)
        // CO24 format: [old[8], is_register, base_addr[2], value[8], ts[2], w2, w4, w8]
        // old = [1,0,...,0] (asserts x10=1=fd), value = [count_0, count_1, 0,...,0] (writes count)
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::MU),
            vec![
                // old[0..7] = [1, 0, 0, 0, 0, 0, 0, 0]
                BusValue::constant(1),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                // is_register = 1
                BusValue::constant(1),
                // base_address = [20, 0] (x10 → addr 2*10 = 20)
                BusValue::constant(20),
                BusValue::constant(0),
                // value[0..7] = [COUNT_0, COUNT_1, 0, 0, 0, 0, 0, 0]
                BusValue::Packed {
                    start_column: cols::COUNT_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::COUNT_1,
                    packing: Packing::Direct,
                },
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                // timestamp = [TIMESTAMP_0, TIMESTAMP_1]
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                // w2=1, w4=0, w8=0 (register = 2 words)
                BusValue::constant(1),
                BusValue::constant(0),
                BusValue::constant(0),
            ],
        ),
        // 14. MEMW read x11 (buf_addr) at ts (mult = first)
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::MU),
            vec![
                // old[0..7] = [ADDRESS_0, ADDRESS_1, 0, 0, 0, 0, 0, 0]
                BusValue::Packed {
                    start_column: cols::ADDRESS_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ADDRESS_1,
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
                // base_address = [22, 0] (x11 → addr 2*11 = 22)
                BusValue::constant(22),
                BusValue::constant(0),
                // value[0..7] = same as old (read)
                BusValue::Packed {
                    start_column: cols::ADDRESS_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ADDRESS_1,
                    packing: Packing::Direct,
                },
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                // timestamp = [TIMESTAMP_0, TIMESTAMP_1]
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                // w2=1, w4=0, w8=0
                BusValue::constant(1),
                BusValue::constant(0),
                BusValue::constant(0),
            ],
        ),
        // 15. MEMW read x12 (count) at ts (mult = first)
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::MU),
            vec![
                // old[0..7] = [COUNT_0, COUNT_1, 0, 0, 0, 0, 0, 0]
                BusValue::Packed {
                    start_column: cols::COUNT_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::COUNT_1,
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
                // base_address = [24, 0] (x12 → addr 2*12 = 24)
                BusValue::constant(24),
                BusValue::constant(0),
                // value[0..7] = same as old (read)
                BusValue::Packed {
                    start_column: cols::COUNT_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::COUNT_1,
                    packing: Packing::Direct,
                },
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                // timestamp = [TIMESTAMP_0, TIMESTAMP_1]
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                // w2=1, w4=0, w8=0
                BusValue::constant(1),
                BusValue::constant(0),
                BusValue::constant(0),
            ],
        ),
        // 16. MEMW x254 read+write commit index at ts (mult = first)
        // Single-word synthetic register per spec: width=1, base address 508.
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::MU),
            vec![
                // old[0..7] = [INDEX, 0, 0, 0, 0, 0, 0, 0]
                BusValue::Packed {
                    start_column: cols::INDEX,
                    packing: Packing::Direct,
                },
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                // is_register = 1
                BusValue::constant(1),
                // base_address = [508, 0]
                BusValue::constant(508),
                BusValue::constant(0),
                // value[0..7] = [INDEX + cast(count, BaseField), 0, ...]
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::INDEX,
                    },
                    LinearTerm::ColumnUnsigned {
                        coefficient: 1,
                        column: cols::COUNT_0,
                    },
                    LinearTerm::ColumnUnsigned {
                        coefficient: super::types::SHIFT_32,
                        column: cols::COUNT_1,
                    },
                ]),
                // value[1..7] = 0
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                // timestamp = [TIMESTAMP_0, TIMESTAMP_1]
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                // w2=0, w4=0, w8=0 (single-word access)
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
            ],
        ),
    ]
}

// =========================================================================
// Single-source constraint set (ConstraintBuilder front-end)
// =========================================================================

/// The COMMIT table's 8 transition constraints as a single [`ConstraintSet`]:
/// - idx 0-2: `IS_BIT` on `first`, `end`, `μ`;
/// - idx 3:   `(first + end)·(1 − μ) = 0` (first/end ⇒ μ);
/// - idx 4,5: `ADD` pair `address + 1 = address_incr` (unconditional);
/// - idx 6,7: `ADD` pair `count_decr + 1 = count` (unconditional).
#[derive(Clone, Copy)]
pub struct CommitConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for CommitConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // One constraint is all that is left. This table is one row per ECALL: it
        // accepts the syscall number, reads the operands, advances x254 and hands the
        // byte loop to MEMMOVE. Everything that modelled a per-byte sequence went with
        // the loop — `first` (identically `mu` now), `end` and its `Zero` detection,
        // and the `address_incr`/`count_decr` ADD pairs with their range checks.
        emit_is_bit(b, 0, cols::MU, None);
    }
}
