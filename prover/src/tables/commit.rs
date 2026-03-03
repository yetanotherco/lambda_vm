//! COMMIT (ECALL) table for writing bytes to stdout.
//!
//! This table handles the `write` syscall (ECALL #3): writing bytes from a memory
//! buffer to stdout. It uses a **recursive design** — each row commits one byte,
//! and rows are linked via a self-referencing "CommitNextByte" bus.
//!
//! Only the first row of each commit sequence receives from the CPU's ECALL bus;
//! subsequent rows receive from the previous commit row via the CommitNextByte bus.
//!
//! ## Columns (24 total)
//! - `timestamp`: DWordWL (2 cols) — timestamp of the ECALL
//! - `address`: DWordWL (2 cols) — current buffer address
//! - `count`: DWordWL (2 cols) — remaining byte count
//! - `first`: Bit — first row in a commit sequence
//! - `end`: Bit — last row (count was 0)
//! - `mu`: Bit — multiplicity (1 for real rows, 0 for padding)
//! - `value`: Byte — the byte being committed
//! - `index`: DWordWL (2 cols) — global commit index
//! - `address_incr`: DWordWL (2 cols) — address + 1
//! - `count_decr`: DWordHL (4 cols) — count - 1 as 4 halfwords (or all 0xFFFF when count=0)
//! - `carry`: Bit — carry from low 32-bit addition of address + 1
//! - `address_incr_hl`: DWordHL (4 cols) — halfword decomposition of address_incr for range checks
//! - `borrow`: Bit — borrow from low 32-bit subtraction of count - 1
//!
//! ## Bus Interactions (18 total)
//! - **Receiver**: EcallCommit bus — receives `[timestamp_lo, timestamp_hi]` from CPU (mult = first)
//! - **Sender**: CommitNextByte bus — sends to next row (mult = mu - end)
//! - **Receiver**: CommitNextByte bus — receives from prev row (mult = mu - first)
//! - **Sender**: IsHalfword bus — range checks for count_decr halfwords (×4, mult = mu)
//! - **Sender**: IsByte bus — range check for value (mult = mu)
//! - **Sender**: IsHalfword bus — range checks for address_incr halfwords (×4, mult = mu - end)
//! - **Sender**: Memw bus — read x10 register (fd=1 assertion) at ts+1 (mult = first)
//! - **Sender**: Memw bus — read x11 register (buf_addr) at ts+1 (mult = first)
//! - **Sender**: Memw bus — read x12 register (count) at ts+1 (mult = first)
//! - **Sender**: Memw bus — write x10 register (return value = count) at ts+2 (mult = first)
//! - **Sender**: Memw bus — read memory byte at ts+3 (mult = mu - end)
//!
//! ## Constraints (13 total)
//! - `range_first`: first * (1 - first) = 0
//! - `range_end`: end * (1 - end) = 0
//! - `range_mu`: mu * (1 - mu) = 0
//! - `first_or_end_implies_mu`: (first + end - first*end) * (1 - mu) = 0
//! - `end_detection`: end * ((65535 - count_decr_0) + ...count_decr_3) = 0
//! - `carry_is_bit`: carry * (1 - carry) = 0
//! - `address_incr_lo`: (mu - end) * (address_incr_0 + carry * 2^32 - address_0 - 1) = 0
//! - `address_incr_hi`: (mu - end) * (address_incr_1 - address_1 - carry) = 0
//! - `address_incr_decomp_lo`: (mu - end) * (address_incr_0 - hl_0 - hl_1 * 65536) = 0
//! - `address_incr_decomp_hi`: (mu - end) * (address_incr_1 - hl_2 - hl_3 * 65536) = 0
//! - `borrow_is_bit`: borrow * (1 - borrow) = 0
//! - `count_decr_lo`: (mu - end) * (count_decr_0 + count_decr_1*65536 + 1 - count_0 - borrow*2^32) = 0
//! - `count_decr_hi`: (mu - end) * (count_decr_2 + count_decr_3*65536 - count_1 + borrow) = 0
//!
//! ## Deferred
//! - x254 register (global commit index) — executor doesn't track this yet
//! - Commit output bus — no consumer table exists yet

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices for COMMIT table
// =========================================================================

/// Column definitions for the COMMIT table.
pub mod cols {
    // Timestamp (DWordWL: 2 cols)
    /// timestamp[0]: low 32 bits
    pub const TIMESTAMP_0: usize = 0;
    /// timestamp[1]: high 32 bits
    pub const TIMESTAMP_1: usize = 1;

    // Buffer address (DWordWL: 2 cols)
    /// address[0]: low 32 bits
    pub const ADDRESS_0: usize = 2;
    /// address[1]: high 32 bits
    pub const ADDRESS_1: usize = 3;

    // Remaining byte count (DWordWL: 2 cols)
    /// count[0]: low 32 bits
    pub const COUNT_0: usize = 4;
    /// count[1]: high 32 bits
    pub const COUNT_1: usize = 5;

    // Control bits
    /// first: 1 if this is the first row of a commit sequence
    pub const FIRST: usize = 6;
    /// end: 1 if this is the last row (count was 0)
    pub const END: usize = 7;
    /// mu: multiplicity bit (1 for real rows, 0 for padding)
    pub const MU: usize = 8;

    // Byte value being committed
    /// value: the byte [0, 256) being committed at this row
    pub const VALUE: usize = 9;

    // Global commit index (DWordWL: 2 cols)
    /// index[0]: low 32 bits of global commit index
    pub const INDEX_0: usize = 10;
    /// index[1]: high 32 bits of global commit index
    pub const INDEX_1: usize = 11;

    // address + 1 result (DWordWL: 2 cols)
    /// address_incr[0]: low 32 bits of (address + 1)
    pub const ADDRESS_INCR_0: usize = 12;
    /// address_incr[1]: high 32 bits of (address + 1)
    pub const ADDRESS_INCR_1: usize = 13;

    // count - 1 result (DWordHL: 4 halfword cols)
    // When count > 0: count_decr = count - 1, decomposed into 4 halfwords
    // When count = 0: count_decr = 0xFFFF_FFFF_FFFF_FFFF (all halfwords = 0xFFFF)
    /// count_decr[0]: halfword 0 (bits 0-15)
    pub const COUNT_DECR_0: usize = 14;
    /// count_decr[1]: halfword 1 (bits 16-31)
    pub const COUNT_DECR_1: usize = 15;
    /// count_decr[2]: halfword 2 (bits 32-47)
    pub const COUNT_DECR_2: usize = 16;
    /// count_decr[3]: halfword 3 (bits 48-63)
    pub const COUNT_DECR_3: usize = 17;

    // Carry bit for address + 1 computation
    /// carry: 1 if low 32 bits of address overflow when adding 1
    pub const CARRY: usize = 18;

    // Halfword decomposition of address_incr (for IsHalfword range checks)
    /// address_incr_hl[0]: bits 0-15 of address_incr
    pub const ADDRESS_INCR_HL_0: usize = 19;
    /// address_incr_hl[1]: bits 16-31 of address_incr
    pub const ADDRESS_INCR_HL_1: usize = 20;
    /// address_incr_hl[2]: bits 32-47 of address_incr
    pub const ADDRESS_INCR_HL_2: usize = 21;
    /// address_incr_hl[3]: bits 48-63 of address_incr
    pub const ADDRESS_INCR_HL_3: usize = 22;

    // Borrow bit for count - 1 computation
    /// borrow: 1 if low 32 bits of count are 0 (borrow from high word)
    pub const BORROW: usize = 23;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 24;
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
    /// Global commit index (accumulated across all ECALLs)
    pub index: u64,
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the COMMIT trace table from a list of operations.
///
/// Each operation becomes one row. The table is padded to the next power of 2 (min 4).
/// Padding rows have all zeros (first=0, end=0, mu=0).
pub fn generate_commit_trace(
    ops: &[CommitOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, op) in ops.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;

        // Timestamp (DWordWL)
        data[base + cols::TIMESTAMP_0] = FE::from(op.timestamp & 0xFFFF_FFFF);
        data[base + cols::TIMESTAMP_1] = FE::from(op.timestamp >> 32);

        // Address (DWordWL)
        data[base + cols::ADDRESS_0] = FE::from(op.address & 0xFFFF_FFFF);
        data[base + cols::ADDRESS_1] = FE::from(op.address >> 32);

        // Count (DWordWL)
        data[base + cols::COUNT_0] = FE::from(op.count & 0xFFFF_FFFF);
        data[base + cols::COUNT_1] = FE::from(op.count >> 32);

        // Control bits
        data[base + cols::FIRST] = FE::from(op.first as u64);
        data[base + cols::END] = FE::from(op.end as u64);
        // mu = 1 for all real rows (first, middle, and end rows)
        data[base + cols::MU] = FE::one();

        // Value
        data[base + cols::VALUE] = FE::from(op.value as u64);

        // Index (DWordWL)
        data[base + cols::INDEX_0] = FE::from(op.index & 0xFFFF_FFFF);
        data[base + cols::INDEX_1] = FE::from(op.index >> 32);

        // address_incr = address + 1 (wrapping)
        let address_incr = op.address.wrapping_add(1);
        let ai_lo = address_incr & 0xFFFF_FFFF;
        let ai_hi = address_incr >> 32;
        data[base + cols::ADDRESS_INCR_0] = FE::from(ai_lo);
        data[base + cols::ADDRESS_INCR_1] = FE::from(ai_hi);

        // Carry for address + 1: overflow of low 32-bit word
        let carry = if (op.address & 0xFFFF_FFFF) + 1 > 0xFFFF_FFFF {
            1u64
        } else {
            0u64
        };
        data[base + cols::CARRY] = FE::from(carry);

        // Halfword decomposition of address_incr (for IsHalfword range checks)
        data[base + cols::ADDRESS_INCR_HL_0] = FE::from(ai_lo & 0xFFFF);
        data[base + cols::ADDRESS_INCR_HL_1] = FE::from((ai_lo >> 16) & 0xFFFF);
        data[base + cols::ADDRESS_INCR_HL_2] = FE::from(ai_hi & 0xFFFF);
        data[base + cols::ADDRESS_INCR_HL_3] = FE::from((ai_hi >> 16) & 0xFFFF);

        // Borrow for count - 1: needed when low 32 bits of count are 0
        let borrow = if (op.count & 0xFFFF_FFFF) == 0 {
            1u64
        } else {
            0u64
        };
        data[base + cols::BORROW] = FE::from(borrow);

        // count_decr: if count == 0, use 0xFFFF_FFFF_FFFF_FFFF; else count - 1
        let count_decr = if op.count == 0 {
            u64::MAX
        } else {
            op.count - 1
        };
        data[base + cols::COUNT_DECR_0] = FE::from(count_decr & 0xFFFF);
        data[base + cols::COUNT_DECR_1] = FE::from((count_decr >> 16) & 0xFFFF);
        data[base + cols::COUNT_DECR_2] = FE::from((count_decr >> 32) & 0xFFFF);
        data[base + cols::COUNT_DECR_3] = FE::from((count_decr >> 48) & 0xFFFF);
    }

    // Padding rows are already zero (first=0, end=0, mu=0)

    TraceTable::new_main(data, cols::NUM_COLUMNS, num_rows)
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the COMMIT table.
///
/// The COMMIT table:
/// - **Receives** EcallCommit from CPU with `[timestamp_lo, timestamp_hi]` (mult = first)
/// - **Sends** to CommitNextByte with `[timestamp, address_incr, count_decr]` (mult = mu - end)
/// - **Receives** from CommitNextByte with `[timestamp, address, count]` (mult = mu - first)
/// - **Sends** to IsHalfword for count_decr range checks (×4, mult = mu)
/// - **Sends** to IsByte for value range check (mult = mu)
///
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // 1. Receive ECALL from CPU (mult = first)
        // Only the first row of each commit sequence receives from the CPU.
        BusInteraction::receiver(
            BusId::EcallCommit,
            Multiplicity::Column(cols::FIRST),
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
        // 2. Send to CommitNextByte (mult = mu - end)
        // Non-end rows send their successor's expected values.
        // Sends: [timestamp, address_incr, count_decr]
        BusInteraction::sender(
            BusId::CommitNextByte,
            Multiplicity::Linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::MU,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::END,
                },
            ]),
            vec![
                // timestamp (DWordWL: 2 Direct elements)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                // address_incr (DWordWL: 2 Direct elements)
                BusValue::Packed {
                    start_column: cols::ADDRESS_INCR_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ADDRESS_INCR_1,
                    packing: Packing::Direct,
                },
                // count_decr (DWordHL: 4 halfwords → 2 bus elements)
                BusValue::Packed {
                    start_column: cols::COUNT_DECR_0,
                    packing: Packing::DWordHL,
                },
            ],
        ),
        // 3. Receive from CommitNextByte (mult = mu - first)
        // Non-first rows receive their values from the previous row's send.
        // Receives: [timestamp, address, count] — must match sender's format
        BusInteraction::receiver(
            BusId::CommitNextByte,
            Multiplicity::Linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::MU,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::FIRST,
                },
            ]),
            vec![
                // timestamp (DWordWL: 2 Direct elements)
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                // address (DWordWL: 2 Direct elements)
                BusValue::Packed {
                    start_column: cols::ADDRESS_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ADDRESS_1,
                    packing: Packing::Direct,
                },
                // count (DWordWL: 2 Direct → 2 bus elements)
                // DWordWL produces same 2 bus elements as DWordHL when values match
                BusValue::Packed {
                    start_column: cols::COUNT_0,
                    packing: Packing::DWordWL,
                },
            ],
        ),
        // 4. Range checks: IsHalfword for count_decr components (×4, mult = mu)
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
        // 5. IsByte for value (mult = mu)
        BusInteraction::sender(
            BusId::IsByte,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: cols::VALUE,
                packing: Packing::Direct,
            }],
        ),
        // 6-9. IsHalfword for address_incr halfwords (×4, mult = mu - end)
        // End rows don't need a valid address_incr (the CNB sender has mult=mu-end),
        // so we only range-check address_incr on non-end rows.
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::MU,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::END,
                },
            ]),
            vec![BusValue::Packed {
                start_column: cols::ADDRESS_INCR_HL_0,
                packing: Packing::Direct,
            }],
        ),
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::MU,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::END,
                },
            ]),
            vec![BusValue::Packed {
                start_column: cols::ADDRESS_INCR_HL_1,
                packing: Packing::Direct,
            }],
        ),
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::MU,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::END,
                },
            ]),
            vec![BusValue::Packed {
                start_column: cols::ADDRESS_INCR_HL_2,
                packing: Packing::Direct,
            }],
        ),
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::MU,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::END,
                },
            ]),
            vec![BusValue::Packed {
                start_column: cols::ADDRESS_INCR_HL_3,
                packing: Packing::Direct,
            }],
        ),
        // 10. MEMW read x10 (fd=1) at ts+1 (mult = first)
        // 24-element read format: [old[8], is_register, base_addr[2], value[8], ts[2], w2, w4, w8]
        // The fd=1 assertion is inherent: if x10 ≠ 1, the MEMW bus won't balance.
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST),
            vec![
                // old[0..7] = [1, 0, 0, 0, 0, 0, 0, 0] (x10 holds fd=1)
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
                // value[0..7] = [1, 0, 0, 0, 0, 0, 0, 0] (read: same as old)
                BusValue::constant(1),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
                // timestamp = [TIMESTAMP + 1, 0]
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::TIMESTAMP_0,
                    },
                    LinearTerm::Constant(1),
                ]),
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                // w2=1, w4=0, w8=0 (register access = 2 Words)
                BusValue::constant(1),
                BusValue::constant(0),
                BusValue::constant(0),
            ],
        ),
        // 11. MEMW read x11 (buf_addr) at ts+1 (mult = first)
        // x11 holds buf_addr as [ADDRESS_0, ADDRESS_1, 0, 0, 0, 0, 0, 0]
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
                // timestamp = [TIMESTAMP + 1, 0]
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::TIMESTAMP_0,
                    },
                    LinearTerm::Constant(1),
                ]),
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
        // 12. MEMW read x12 (count) at ts+1 (mult = first)
        // x12 holds count as [COUNT_0, COUNT_1, 0, 0, 0, 0, 0, 0]
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
                // timestamp = [TIMESTAMP + 1, 0]
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::TIMESTAMP_0,
                    },
                    LinearTerm::Constant(1),
                ]),
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
        // 13. MEMW write x10 (return value = count) at ts+2 (mult = first)
        // 16-element write format: [is_register, base_addr[2], value[8], ts[2], w2, w4, w8]
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST),
            vec![
                // is_register = 1
                BusValue::constant(1),
                // base_address = [20, 0] (x10)
                BusValue::constant(20),
                BusValue::constant(0),
                // value[0..7] = [COUNT_0, COUNT_1, 0, 0, 0, 0, 0, 0] (writes count as return)
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
                // timestamp = [TIMESTAMP + 2, 0]
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::TIMESTAMP_0,
                    },
                    LinearTerm::Constant(2),
                ]),
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
        // 14. MEMW read byte at ts+3 (mult = mu - end)
        // 24-element read format for 1-byte memory access
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::MU,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::END,
                },
            ]),
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
                // timestamp = [TIMESTAMP + 3, 0]
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::TIMESTAMP_0,
                    },
                    LinearTerm::Constant(3),
                ]),
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                // w2=0, w4=0, w8=0 (width=1 byte)
                BusValue::constant(0),
                BusValue::constant(0),
                BusValue::constant(0),
            ],
        ),
    ]
}

// =========================================================================
// Constraints
// =========================================================================

/// Creates all constraints for the COMMIT table.
///
/// Returns the constraint objects and the next available constraint index.
///
/// Constraints:
/// 0. `range_first`: first * (1 - first) = 0
/// 1. `range_end`: end * (1 - end) = 0
/// 2. `range_mu`: mu * (1 - mu) = 0
/// 3. `first_or_end_implies_mu`: (first + end - first*end) * (1 - mu) = 0
/// 4. `end_detection`: end * ((65535 - count_decr_0) + (65535 - count_decr_1)
///   + (65535 - count_decr_2) + (65535 - count_decr_3)) = 0
/// 5. `carry_is_bit`: carry * (1 - carry) = 0
/// 6. `address_incr_lo`: (mu - end) * (address_incr_0 + carry * 2^32 - address_0 - 1) = 0
/// 7. `address_incr_hi`: (mu - end) * (address_incr_1 - address_1 - carry) = 0
/// 8. `address_incr_decomp_lo`: mu * (address_incr_0 - hl_0 - hl_1 * 65536) = 0
/// 9. `address_incr_decomp_hi`: mu * (address_incr_1 - hl_2 - hl_3 * 65536) = 0
pub fn create_constraints(constraint_idx_start: usize) -> (Vec<CommitConstraint>, usize) {
    let constraints = vec![
        CommitConstraint {
            kind: CommitConstraintKind::RangeFirst,
            constraint_idx: constraint_idx_start,
        },
        CommitConstraint {
            kind: CommitConstraintKind::RangeEnd,
            constraint_idx: constraint_idx_start + 1,
        },
        CommitConstraint {
            kind: CommitConstraintKind::RangeMu,
            constraint_idx: constraint_idx_start + 2,
        },
        CommitConstraint {
            kind: CommitConstraintKind::FirstOrEndImpliesMu,
            constraint_idx: constraint_idx_start + 3,
        },
        CommitConstraint {
            kind: CommitConstraintKind::EndDetection,
            constraint_idx: constraint_idx_start + 4,
        },
        CommitConstraint {
            kind: CommitConstraintKind::CarryIsBit,
            constraint_idx: constraint_idx_start + 5,
        },
        CommitConstraint {
            kind: CommitConstraintKind::AddressIncrLo,
            constraint_idx: constraint_idx_start + 6,
        },
        CommitConstraint {
            kind: CommitConstraintKind::AddressIncrHi,
            constraint_idx: constraint_idx_start + 7,
        },
        CommitConstraint {
            kind: CommitConstraintKind::AddressIncrDecompLo,
            constraint_idx: constraint_idx_start + 8,
        },
        CommitConstraint {
            kind: CommitConstraintKind::AddressIncrDecompHi,
            constraint_idx: constraint_idx_start + 9,
        },
        CommitConstraint {
            kind: CommitConstraintKind::BorrowIsBit,
            constraint_idx: constraint_idx_start + 10,
        },
        CommitConstraint {
            kind: CommitConstraintKind::CountDecrLo,
            constraint_idx: constraint_idx_start + 11,
        },
        CommitConstraint {
            kind: CommitConstraintKind::CountDecrHi,
            constraint_idx: constraint_idx_start + 12,
        },
    ];
    let next_idx = constraint_idx_start + constraints.len();
    (constraints, next_idx)
}

/// The kind of COMMIT constraint.
#[derive(Debug, Clone, Copy)]
enum CommitConstraintKind {
    /// first * (1 - first) = 0
    RangeFirst,
    /// end * (1 - end) = 0
    RangeEnd,
    /// mu * (1 - mu) = 0
    RangeMu,
    /// (first + end - first*end) * (1 - mu) = 0
    FirstOrEndImpliesMu,
    /// end * ((65535 - count_decr_0) + (65535 - count_decr_1) + (65535 - count_decr_2) + (65535 - count_decr_3)) = 0
    EndDetection,
    /// carry * (1 - carry) = 0
    CarryIsBit,
    /// (mu - end) * (address_incr_0 + carry * 2^32 - address_0 - 1) = 0
    AddressIncrLo,
    /// (mu - end) * (address_incr_1 - address_1 - carry) = 0
    AddressIncrHi,
    /// (mu - end) * (address_incr_0 - hl_0 - hl_1 * 65536) = 0
    AddressIncrDecompLo,
    /// (mu - end) * (address_incr_1 - hl_2 - hl_3 * 65536) = 0
    AddressIncrDecompHi,
    /// borrow * (1 - borrow) = 0
    BorrowIsBit,
    /// (mu - end) * (count_decr_0 + count_decr_1 * 65536 + 1 - count_0 - borrow * 2^32) = 0
    CountDecrLo,
    /// (mu - end) * (count_decr_2 + count_decr_3 * 65536 - count_1 + borrow) = 0
    CountDecrHi,
}

/// A constraint for the COMMIT table.
pub struct CommitConstraint {
    kind: CommitConstraintKind,
    constraint_idx: usize,
}

impl CommitConstraint {
    fn compute<F, E>(
        &self,
        step: &stark::table::TableView<F, E>,
    ) -> math::field::element::FieldElement<F>
    where
        F: math::field::traits::IsSubFieldOf<E>,
        E: math::field::traits::IsField,
    {
        let one = math::field::element::FieldElement::<F>::one();

        match self.kind {
            CommitConstraintKind::RangeFirst => {
                let first = step.get_main_evaluation_element(0, cols::FIRST).clone();
                // first * (1 - first)
                &first * (&one - &first)
            }
            CommitConstraintKind::RangeEnd => {
                let end = step.get_main_evaluation_element(0, cols::END).clone();
                // end * (1 - end)
                &end * (&one - &end)
            }
            CommitConstraintKind::RangeMu => {
                let mu = step.get_main_evaluation_element(0, cols::MU).clone();
                // mu * (1 - mu)
                &mu * (&one - &mu)
            }
            CommitConstraintKind::FirstOrEndImpliesMu => {
                let first = step.get_main_evaluation_element(0, cols::FIRST).clone();
                let end = step.get_main_evaluation_element(0, cols::END).clone();
                let mu = step.get_main_evaluation_element(0, cols::MU).clone();
                // (first + end - first*end) * (1 - mu)
                let first_or_end = &first + &end - &first * &end;
                first_or_end * (one - mu)
            }
            CommitConstraintKind::EndDetection => {
                let end = step.get_main_evaluation_element(0, cols::END).clone();
                let c0 = step
                    .get_main_evaluation_element(0, cols::COUNT_DECR_0)
                    .clone();
                let c1 = step
                    .get_main_evaluation_element(0, cols::COUNT_DECR_1)
                    .clone();
                let c2 = step
                    .get_main_evaluation_element(0, cols::COUNT_DECR_2)
                    .clone();
                let c3 = step
                    .get_main_evaluation_element(0, cols::COUNT_DECR_3)
                    .clone();
                let max_half = math::field::element::FieldElement::<F>::from(65535u64);
                // end * ((65535 - c0) + (65535 - c1) + (65535 - c2) + (65535 - c3))
                let sum =
                    (&max_half - &c0) + (&max_half - &c1) + (&max_half - &c2) + (max_half - c3);
                end * sum
            }
            CommitConstraintKind::CarryIsBit => {
                let carry = step.get_main_evaluation_element(0, cols::CARRY).clone();
                // carry * (1 - carry)
                &carry * (&one - &carry)
            }
            CommitConstraintKind::AddressIncrLo => {
                let mu = step.get_main_evaluation_element(0, cols::MU).clone();
                let end = step.get_main_evaluation_element(0, cols::END).clone();
                let addr0 = step.get_main_evaluation_element(0, cols::ADDRESS_0).clone();
                let incr0 = step
                    .get_main_evaluation_element(0, cols::ADDRESS_INCR_0)
                    .clone();
                let carry = step.get_main_evaluation_element(0, cols::CARRY).clone();
                let two_32 = math::field::element::FieldElement::<F>::from(1u64 << 32);
                // (mu - end) * (address_incr_0 + carry * 2^32 - address_0 - 1)
                (&mu - &end) * (&incr0 + &carry * &two_32 - &addr0 - &one)
            }
            CommitConstraintKind::AddressIncrHi => {
                let mu = step.get_main_evaluation_element(0, cols::MU).clone();
                let end = step.get_main_evaluation_element(0, cols::END).clone();
                let addr1 = step.get_main_evaluation_element(0, cols::ADDRESS_1).clone();
                let incr1 = step
                    .get_main_evaluation_element(0, cols::ADDRESS_INCR_1)
                    .clone();
                let carry = step.get_main_evaluation_element(0, cols::CARRY).clone();
                // (mu - end) * (address_incr_1 - address_1 - carry)
                (&mu - &end) * (incr1 - &addr1 - carry)
            }
            CommitConstraintKind::AddressIncrDecompLo => {
                let mu = step.get_main_evaluation_element(0, cols::MU).clone();
                let end = step.get_main_evaluation_element(0, cols::END).clone();
                let incr0 = step
                    .get_main_evaluation_element(0, cols::ADDRESS_INCR_0)
                    .clone();
                let hl0 = step
                    .get_main_evaluation_element(0, cols::ADDRESS_INCR_HL_0)
                    .clone();
                let hl1 = step
                    .get_main_evaluation_element(0, cols::ADDRESS_INCR_HL_1)
                    .clone();
                let c65536 = math::field::element::FieldElement::<F>::from(65536u64);
                // (mu - end) * (address_incr_0 - hl_0 - hl_1 * 65536)
                (&mu - &end) * (incr0 - &hl0 - hl1 * c65536)
            }
            CommitConstraintKind::AddressIncrDecompHi => {
                let mu = step.get_main_evaluation_element(0, cols::MU).clone();
                let end = step.get_main_evaluation_element(0, cols::END).clone();
                let incr1 = step
                    .get_main_evaluation_element(0, cols::ADDRESS_INCR_1)
                    .clone();
                let hl2 = step
                    .get_main_evaluation_element(0, cols::ADDRESS_INCR_HL_2)
                    .clone();
                let hl3 = step
                    .get_main_evaluation_element(0, cols::ADDRESS_INCR_HL_3)
                    .clone();
                let c65536 = math::field::element::FieldElement::<F>::from(65536u64);
                // (mu - end) * (address_incr_1 - hl_2 - hl_3 * 65536)
                (&mu - &end) * (incr1 - &hl2 - hl3 * c65536)
            }
            CommitConstraintKind::BorrowIsBit => {
                let borrow = step.get_main_evaluation_element(0, cols::BORROW).clone();
                // borrow * (1 - borrow)
                &borrow * (&one - &borrow)
            }
            CommitConstraintKind::CountDecrLo => {
                let mu = step.get_main_evaluation_element(0, cols::MU).clone();
                let end = step.get_main_evaluation_element(0, cols::END).clone();
                let count0 = step.get_main_evaluation_element(0, cols::COUNT_0).clone();
                let cd0 = step
                    .get_main_evaluation_element(0, cols::COUNT_DECR_0)
                    .clone();
                let cd1 = step
                    .get_main_evaluation_element(0, cols::COUNT_DECR_1)
                    .clone();
                let borrow = step.get_main_evaluation_element(0, cols::BORROW).clone();
                let c65536 = math::field::element::FieldElement::<F>::from(65536u64);
                let two_32 = math::field::element::FieldElement::<F>::from(1u64 << 32);
                // (mu - end) * (count_decr_0 + count_decr_1 * 65536 + 1 - count_0 - borrow * 2^32)
                (&mu - &end) * (&cd0 + &cd1 * &c65536 + &one - &count0 - borrow * two_32)
            }
            CommitConstraintKind::CountDecrHi => {
                let mu = step.get_main_evaluation_element(0, cols::MU).clone();
                let end = step.get_main_evaluation_element(0, cols::END).clone();
                let count1 = step.get_main_evaluation_element(0, cols::COUNT_1).clone();
                let cd2 = step
                    .get_main_evaluation_element(0, cols::COUNT_DECR_2)
                    .clone();
                let cd3 = step
                    .get_main_evaluation_element(0, cols::COUNT_DECR_3)
                    .clone();
                let borrow = step.get_main_evaluation_element(0, cols::BORROW).clone();
                let c65536 = math::field::element::FieldElement::<F>::from(65536u64);
                // (mu - end) * (count_decr_2 + count_decr_3 * 65536 - count_1 + borrow)
                (&mu - &end) * (cd2 + cd3 * c65536 - count1 + borrow)
            }
        }
    }
}

use math::field::element::FieldElement;
use stark::constraints::transition::TransitionConstraint;
use stark::traits::TransitionEvaluationContext;

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for CommitConstraint {
    fn degree(&self) -> usize {
        match self.kind {
            CommitConstraintKind::RangeFirst
            | CommitConstraintKind::RangeEnd
            | CommitConstraintKind::RangeMu
            | CommitConstraintKind::CarryIsBit
            | CommitConstraintKind::EndDetection
            | CommitConstraintKind::AddressIncrLo
            | CommitConstraintKind::AddressIncrHi
            | CommitConstraintKind::AddressIncrDecompLo
            | CommitConstraintKind::AddressIncrDecompHi
            | CommitConstraintKind::BorrowIsBit
            | CommitConstraintKind::CountDecrLo
            | CommitConstraintKind::CountDecrHi => 2,
            CommitConstraintKind::FirstOrEndImpliesMu => 3,
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
                ..
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value.to_extension();
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _,
                rap_challenges: _,
                ..
            } => {
                let constraint_value = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = constraint_value;
            }
        }
    }
}
