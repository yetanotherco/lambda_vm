//! COMMIT (ECALL) table for writing bytes to stdout.
//!
//! This table handles the `write` syscall (ECALL #64): writing bytes from a memory
//! buffer to stdout. It uses a **recursive design** — each row commits one byte,
//! and rows are linked via a self-referencing "CommitNextByte" bus.
//!
//! Only the first row of each commit sequence receives from the CPU's ECALL bus;
//! subsequent rows receive from the previous commit row via the CommitNextByte bus.
//!
//! ## Columns (19 total)
//! - `timestamp`: Word (1 col) — timestamp of the ECALL
//! - `index`: BaseField (1 col) — global byte index for this committed value
//! - `address`: DWordWL (2 cols) — current buffer address
//! - `address_incr`: DWordHL (4 cols) — address + 1, as 4 halfwords
//! - `count`: DWordWL (2 cols) — remaining byte count
//! - `count_decr`: DWordHL (4 cols) — count - 1 as 4 halfwords (or all 0xFFFF when count=0)
//! - `first`: Bit — first row in a commit sequence
//! - `end`: Bit — last row (count was 0)
//! - `value`: Byte — the byte being committed
//! - `mu`: Bit — multiplicity (1 for real rows, 0 for padding)
//!
//! ## Bus Interactions (18 total)
//! - **Receiver**: Ecall bus — receives `[timestamp, constant(64), constant(0)]` from CPU (mult = first)
//! - **Sender**: CommitNextByte bus — sends to next row (mult = mu - end)
//! - **Receiver**: CommitNextByte bus — receives from prev row (mult = mu - first)
//! - **Sender**: IsHalfword bus — range checks for count_decr halfwords (×4, mult = mu)
//! - **Sender**: IsHalfword bus — range checks for address_incr halfwords (×4, mult = mu)
//! - **Sender**: Zero bus — end detection via count_decr (mult = mu)
//! - **Sender**: Memw bus — read+write x10 register (fd=1→count) at ts (mult = first)
//! - **Sender**: Memw bus — read x11 register (buf_addr) at ts (mult = first)
//! - **Sender**: Memw bus — read x12 register (count) at ts (mult = first)
//! - **Sender**: Memw bus — read+write x254 commit index at ts (mult = first)
//! - **Sender**: Memw bus — read memory byte at ts (mult = mu - end)
//! - **Sender**: Commit bus — sends committed `(index, value)` pairs (mult = mu - end)
//!
//! ## Constraints (8 total)
//! - `range_first`: first * (1 - first) = 0 (degree 2)
//! - `range_end`: end * (1 - end) = 0 (degree 2)
//! - `range_mu`: mu * (1 - mu) = 0 (degree 2)
//! - `first_or_end_implies_mu`: (first + end) * (1 - mu) = 0 (degree 2)
//! - `address_incr_carry_0`: ADD template carry_0 for address + 1 = address_incr (degree 2)
//! - `address_incr_carry_1`: ADD template carry_1 for address + 1 = address_incr (degree 2)
//! - `count_decr_carry_0`: SUB template carry_0 for count_decr + 1 = count (degree 2)
//! - `count_decr_carry_1`: SUB template carry_1 for count_decr + 1 = count (degree 2)
//!
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

use crate::constraints::templates::{AddOperand, emit_add_pair, emit_is_bit};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};

// =========================================================================
// Column indices for COMMIT table
// =========================================================================

/// Column definitions for the COMMIT table.
///
/// Layout follows the spec order: timestamp, index, address, address_incr,
/// count, count_decr, first, end, value, mu.
pub mod cols {
    // Timestamp (Word: 1 col)
    /// timestamp: 32-bit timestamp of the originating ECALL
    pub const TIMESTAMP: usize = 0;

    // Commit index (BaseField: 1 col)
    /// index: global byte index of the committed value
    pub const INDEX: usize = 1;

    // Buffer address (DWordWL: 2 cols)
    /// address[0]: low 32 bits
    pub const ADDRESS_0: usize = 2;
    /// address[1]: high 32 bits
    pub const ADDRESS_1: usize = 3;

    // address + 1 (DWordHL: 4 halfword cols)
    /// address_incr[0]: halfword 0 (bits 0-15)
    pub const ADDRESS_INCR_0: usize = 4;
    /// address_incr[1]: halfword 1 (bits 16-31)
    pub const ADDRESS_INCR_1: usize = 5;
    /// address_incr[2]: halfword 2 (bits 32-47)
    pub const ADDRESS_INCR_2: usize = 6;
    /// address_incr[3]: halfword 3 (bits 48-63)
    pub const ADDRESS_INCR_3: usize = 7;

    // Remaining byte count (DWordWL: 2 cols)
    /// count[0]: low 32 bits
    pub const COUNT_0: usize = 8;
    /// count[1]: high 32 bits
    pub const COUNT_1: usize = 9;

    // count - 1 (DWordHL: 4 halfword cols)
    // When count > 0: count_decr = count - 1
    // When count = 0: count_decr = 0xFFFF_FFFF_FFFF_FFFF (all halfwords = 0xFFFF)
    /// count_decr[0]: halfword 0 (bits 0-15)
    pub const COUNT_DECR_0: usize = 10;
    /// count_decr[1]: halfword 1 (bits 16-31)
    pub const COUNT_DECR_1: usize = 11;
    /// count_decr[2]: halfword 2 (bits 32-47)
    pub const COUNT_DECR_2: usize = 12;
    /// count_decr[3]: halfword 3 (bits 48-63)
    pub const COUNT_DECR_3: usize = 13;

    // Control bits
    /// first: 1 if this is the first row of a commit sequence
    pub const FIRST: usize = 14;
    /// end: 1 if this is the last row (count was 0)
    pub const END: usize = 15;

    // Byte value being committed
    /// value: the byte [0, 256) being committed at this row
    pub const VALUE: usize = 16;

    /// mu: multiplicity bit (1 for real rows, 0 for padding)
    pub const MU: usize = 17;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 18;
}

// =========================================================================
// Operation type
// =========================================================================

/// A single row in the COMMIT table.
///
/// Each row represents one byte being committed from a buffer. Rows are linked
/// via the CommitNextByte bus to form a chain for each commit ECALL.
#[derive(Debug, Clone)]
pub struct CommitOperation {
    /// Timestamp of the originating ECALL
    pub timestamp: u64,
    /// Global commit index for this byte
    pub index: u64,
    /// Current buffer address for this byte
    pub address: u64,
    /// Remaining byte count (including this byte, 0 on end row)
    pub count: u64,
    /// Whether this is the first row of a commit sequence
    pub first: bool,
    /// Whether this is the end row (count was 0, no byte committed)
    pub end: bool,
    /// The byte value being committed (0 on end row)
    pub value: u8,
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
        vec![FE::zero(); num_rows * cols::NUM_COLUMNS],
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, op) in ops.iter().enumerate() {
        // Timestamp (Word)
        table.set_u64(row_idx, cols::TIMESTAMP, op.timestamp);

        // Index (BaseField)
        table.set_u64(row_idx, cols::INDEX, op.index);

        // Address (DWordWL)
        table.set_dword_wl(row_idx, cols::ADDRESS_0, op.address);

        // address_incr = address + 1 (DWordHL: 4 halfwords)
        let address_incr = op.address.wrapping_add(1);
        table.set_dword_hl(row_idx, cols::ADDRESS_INCR_0, address_incr);

        // Count (DWordWL)
        table.set_dword_wl(row_idx, cols::COUNT_0, op.count);

        // count_decr: if count == 0, use 0xFFFF_FFFF_FFFF_FFFF; else count - 1
        let count_decr = if op.count == 0 {
            u64::MAX
        } else {
            op.count - 1
        };
        table.set_dword_hl(row_idx, cols::COUNT_DECR_0, count_decr);

        // Control bits
        table.set_bool(row_idx, cols::FIRST, op.first);
        table.set_bool(row_idx, cols::END, op.end);

        // Value
        table.set_byte(row_idx, cols::VALUE, op.value);

        // mu = 1 for all real rows (first, middle, and end rows)
        table.set_fe(row_idx, cols::MU, FE::one());
    }

    // Padding rows: spec requires count=1 and address_incr=[1,0,0,0] so
    // the unconditional ADD/SUB templates have valid carry values.
    // count=1 → count_decr=0 (all halfwords zero), address=0 → address_incr=1.
    for row_idx in n..num_rows {
        // count = 1 (low word)
        table.set_fe(row_idx, cols::COUNT_0, FE::one());
        // address_incr halfword 0 = 1 (address=0, so address+1 = 1)
        table.set_fe(row_idx, cols::ADDRESS_INCR_0, FE::one());
        // All other fields remain zero: timestamp=0, address=0, count_1=0,
        // count_decr=[0,0,0,0], first=0, end=0, value=0, mu=0,
        // address_incr_1..3=0
    }

    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the COMMIT table (18 total).
///
/// The COMMIT table:
/// - **Receives** Ecall from CPU with `[timestamp, constant(64), constant(0)]` (mult = first)
/// - **Sends** to CommitNextByte with `[timestamp, index + 1, address_incr, count_decr]` (mult = mu - end)
/// - **Receives** from CommitNextByte with `[timestamp, index, address, count]` (mult = mu - first)
/// - **Sends** to IsHalfword for count_decr range checks (×4, mult = mu)
/// - **Sends** to IsHalfword for address_incr range checks (×4, mult = mu)
/// - **Sends** to Zero for end detection (mult = mu)
/// - **Sends** to Memw for register/memory accesses (×5, mult varies)
pub fn bus_interactions() -> Vec<BusInteraction> {
    // Reusable multiplicity expressions
    let mu_minus_end = Multiplicity::Diff(cols::MU, cols::END);
    let mu_minus_first = Multiplicity::Diff(cols::MU, cols::FIRST);

    vec![
        // 1. Receive ECALL from CPU (mult = first)
        // Payload: [timestamp, syscall_lo32, syscall_hi32]
        BusInteraction::receiver(
            BusId::Ecall,
            Multiplicity::Column(cols::FIRST),
            vec![
                BusValue::Packed {
                    start_column: cols::TIMESTAMP,
                    packing: Packing::Direct,
                },
                BusValue::constant(64), // syscall number lo32 = Commit (64)
                BusValue::constant(0),  // syscall number hi32 = 0
            ],
        ),
        // 2. Send to CommitNextByte (mult = mu - end)
        // Sends: [timestamp, index + 1, address_incr(as DWordWL), count_decr(as DWordWL)]
        BusInteraction::sender(
            BusId::CommitNextByte,
            mu_minus_end.clone(),
            vec![
                // timestamp (Word)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP,
                    packing: Packing::Direct,
                },
                // index + 1 (BaseField)
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::INDEX,
                    },
                    LinearTerm::Constant(1),
                ]),
                // address_incr (DWordHL → 2 bus elements via DWordHL packing)
                BusValue::Packed {
                    start_column: cols::ADDRESS_INCR_0,
                    packing: Packing::DWordHL,
                },
                // count_decr (DWordHL → 2 bus elements via DWordHL packing)
                BusValue::Packed {
                    start_column: cols::COUNT_DECR_0,
                    packing: Packing::DWordHL,
                },
            ],
        ),
        // 3. Receive from CommitNextByte (mult = mu - first)
        // Receives: [timestamp, index, address, count]
        BusInteraction::receiver(
            BusId::CommitNextByte,
            mu_minus_first,
            vec![
                // timestamp (Word)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP,
                    packing: Packing::Direct,
                },
                // index (BaseField)
                BusValue::Packed {
                    start_column: cols::INDEX,
                    packing: Packing::Direct,
                },
                // address (DWordWL)
                BusValue::Packed {
                    start_column: cols::ADDRESS_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ADDRESS_1,
                    packing: Packing::Direct,
                },
                // count (DWordWL → 2 bus elements)
                BusValue::Packed {
                    start_column: cols::COUNT_0,
                    packing: Packing::DWordWL,
                },
            ],
        ),
        // 4-7. IsHalfword for count_decr (×4, mult = mu)
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::COUNT_DECR_0,
                packing: Packing::Direct,
            }],
        ),
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::COUNT_DECR_1,
                packing: Packing::Direct,
            }],
        ),
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::COUNT_DECR_2,
                packing: Packing::Direct,
            }],
        ),
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::COUNT_DECR_3,
                packing: Packing::Direct,
            }],
        ),
        // 8-11. IsHalfword for address_incr (×4, mult = mu)
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::ADDRESS_INCR_0,
                packing: Packing::Direct,
            }],
        ),
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::ADDRESS_INCR_1,
                packing: Packing::Direct,
            }],
        ),
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::ADDRESS_INCR_2,
                packing: Packing::Direct,
            }],
        ),
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::ADDRESS_INCR_3,
                packing: Packing::Direct,
            }],
        ),
        // 12. ZERO bus for end detection (mult = mu)
        // Input: (65535 - cd_0) + (65535 - cd_1) + (65535 - cd_2) + (65535 - cd_3)
        // Output: end (1 when all count_decr halfwords are 0xFFFF, i.e., count was 0)
        BusInteraction::sender(
            BusId::Zero,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::linear(vec![
                    LinearTerm::Constant(4 * 65535),
                    LinearTerm::Column {
                        coefficient: -1,
                        column: cols::COUNT_DECR_0,
                    },
                    LinearTerm::Column {
                        coefficient: -1,
                        column: cols::COUNT_DECR_1,
                    },
                    LinearTerm::Column {
                        coefficient: -1,
                        column: cols::COUNT_DECR_2,
                    },
                    LinearTerm::Column {
                        coefficient: -1,
                        column: cols::COUNT_DECR_3,
                    },
                ]),
                BusValue::Packed {
                    start_column: cols::END,
                    packing: Packing::Direct,
                },
            ],
        ),
        // 13. MEMW read+write x10 (fd=1 → count) at ts (mult = first)
        // CO24 format: [old[8], is_register, base_addr[2], value[8], ts[2], w2, w4, w8]
        // old = [1,0,...,0] (asserts x10=1=fd), value = [count_0, count_1, 0,...,0] (writes count)
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST),
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
                // timestamp (Word)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP,
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
            Multiplicity::Column(cols::FIRST),
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
                // timestamp (Word)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP,
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
            Multiplicity::Column(cols::FIRST),
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
                // timestamp (Word)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP,
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
            Multiplicity::Column(cols::FIRST),
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
                // timestamp (Word)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP,
                    packing: Packing::Direct,
                },
                // w2=0, w4=0, w8=0 (single-word access)
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
            ],
        ),
        // 17. MEMW read byte at ts (mult = mu - end)
        BusInteraction::sender(
            BusId::Memw,
            mu_minus_end.clone(),
            vec![
                // old[0..7] = [VALUE, 0, 0, 0, 0, 0, 0, 0]
                BusValue::Packed {
                    start_column: cols::VALUE,
                    packing: Packing::Direct,
                },
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                // is_register = 0
                BusValue::constant(0),
                // base_address = [ADDRESS_0, ADDRESS_1]
                BusValue::Packed {
                    start_column: cols::ADDRESS_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ADDRESS_1,
                    packing: Packing::Direct,
                },
                // value[0..7] = [VALUE, 0, 0, 0, 0, 0, 0, 0] (read: same as old)
                BusValue::Packed {
                    start_column: cols::VALUE,
                    packing: Packing::Direct,
                },
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                // timestamp (Word)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP,
                    packing: Packing::Direct,
                },
                // w2=0, w4=0, w8=0 (width=1 byte)
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
            ],
        ),
        // 18. COMMIT[index, value] (mult = mu - end)
        BusInteraction::sender(
            BusId::Commit,
            mu_minus_end,
            vec![
                BusValue::Packed {
                    start_column: cols::INDEX,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::VALUE,
                    packing: Packing::Direct,
                },
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
pub struct CommitConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for CommitConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0-2: IS_BIT for first, end, mu
        emit_is_bit(b, 0, cols::FIRST, None);
        emit_is_bit(b, 1, cols::END, None);
        emit_is_bit(b, 2, cols::MU, None);

        // idx 3: (first + end) * (1 - mu)
        let one = b.one();
        let first = b.main(0, cols::FIRST);
        let end = b.main(0, cols::END);
        let mu = b.main(0, cols::MU);
        b.emit_base(3, (first + end) * (one - mu));

        // idx 4,5: ADD template for address + 1 = address_incr (unconditional)
        emit_add_pair(
            b,
            4,
            &[],
            &AddOperand::dword(cols::ADDRESS_0),
            &AddOperand::constant(1),
            &AddOperand::from_dword_hl(cols::ADDRESS_INCR_0),
        );

        // idx 6,7: SUB via ADD: count_decr + 1 = count (unconditional)
        emit_add_pair(
            b,
            6,
            &[],
            &AddOperand::from_dword_hl(cols::COUNT_DECR_0),
            &AddOperand::constant(1),
            &AddOperand::dword(cols::COUNT_0),
        );
    }
}
