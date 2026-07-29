//! DMA memcpy table — proves a `memcpy(dst, src, n)` off the CPU execution trace.
//!
//! The guest's strong `memcpy` symbol (see `syscalls/src/syscalls.rs`)
//! dispatches bulk copies to the DMA ecall (`DMA_MEMCPY_SYSCALL_NUMBER`); this table
//! proves the copy so the per-byte load/store loop leaves the CPU trace.
//!
//! **Recursive/streaming design, cloned from COMMIT** (`commit.rs`): a row copies
//! eight bytes while `count >= 8`, otherwise one byte. The LT table pins that choice,
//! so the prover cannot select a convenient partition. Rows chain through `DmaNext`;
//! each call ends with one terminal row where `count == 0`.
//!
//! Data rows emit a MEMW read at `T+1` and a MEMW write at `T+2`. All reads precede
//! all writes in trace generation, which gives overlapping regions well-defined
//! snapshot/memmove semantics. The same eight value columns feed both tuples, making
//! copied-value equality structural.
//!
//! ## Columns (32 total)
//! - `timestamp`: DWordWL (2) — the ECALL timestamp
//! - `src`: DWordWL (2) — current source byte address
//! - `src_incr`: DWordHL (4) — src + selected width
//! - `dst`: DWordWL (2) — current destination byte address
//! - `dst_incr`: DWordHL (4) — dst + selected width
//! - `count`: DWordWL (2) — remaining byte count (including this byte; 0 on the end row)
//! - `count_decr`: DWordHL (4) — count - width (all 0xFFFF when count == 0, since
//!   the terminal row is a one-byte row and `0 - 1` wraps every halfword to 0xFFFF)
//! - `first`: Bit — first row of a copy
//! - `end`: Bit — last row (count was 0)
//! - `tail`: Bit — `count < 8`; selects a 1-byte rather than 8-byte row
//! - `value[8]`: bytes being copied (bytes 1..7 are zero on tail rows)
//! - `mu`: Bit — multiplicity (1 real, 0 padding)
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use crate::constraints::templates::{
    AddLinearTerm, AddOperand, emit_add_pair, emit_add_pair_no_overflow, emit_is_bit,
};

use executor::vm::instruction::execution::{
    DMA_MEMCPY_MAX_BYTES as EXECUTOR_DMA_MEMCPY_MAX_BYTES, DMA_MEMCPY_SYSCALL_NUMBER,
};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};

/// DMA memcpy syscall value, split into 32-bit limbs for the Ecall bus.
const DMA_MEMCPY_LO32: u64 = DMA_MEMCPY_SYSCALL_NUMBER & 0xFFFF_FFFF;
const DMA_MEMCPY_HI32: u64 = DMA_MEMCPY_SYSCALL_NUMBER >> 32;
/// Maximum bytes represented by one DMA ecall, taken from the executor so the
/// bound the AIR proves cannot drift from the bound execution enforces. The
/// guest stub chunks larger copies.
pub const DMA_MEMCPY_MAX_BYTES: u64 = EXECUTOR_DMA_MEMCPY_MAX_BYTES;

pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;

    pub const SRC_0: usize = 2;
    pub const SRC_1: usize = 3;

    pub const SRC_INCR_0: usize = 4;
    pub const SRC_INCR_1: usize = 5;
    pub const SRC_INCR_2: usize = 6;
    pub const SRC_INCR_3: usize = 7;

    pub const DST_0: usize = 8;
    pub const DST_1: usize = 9;

    pub const DST_INCR_0: usize = 10;
    pub const DST_INCR_1: usize = 11;
    pub const DST_INCR_2: usize = 12;
    pub const DST_INCR_3: usize = 13;

    pub const COUNT_0: usize = 14;
    pub const COUNT_1: usize = 15;

    pub const COUNT_DECR_0: usize = 16;
    pub const COUNT_DECR_1: usize = 17;
    pub const COUNT_DECR_2: usize = 18;
    pub const COUNT_DECR_3: usize = 19;

    pub const FIRST: usize = 20;
    pub const END: usize = 21;
    pub const TAIL: usize = 22;
    pub const VALUE_0: usize = 23;
    pub const VALUE: [usize; 8] = [
        VALUE_0,
        VALUE_0 + 1,
        VALUE_0 + 2,
        VALUE_0 + 3,
        VALUE_0 + 4,
        VALUE_0 + 5,
        VALUE_0 + 6,
        VALUE_0 + 7,
    ];
    pub const MU: usize = 31;

    pub const NUM_COLUMNS: usize = 32;
}

/// One row of the DMA memcpy table: eight bytes, one tail byte, or the terminal row.
#[derive(Debug, Clone)]
pub struct DmaOperation {
    pub timestamp: u64,
    pub src: u64,
    pub dst: u64,
    /// Remaining byte count (including this byte; 0 on the end row).
    pub count: u64,
    pub first: bool,
    pub end: bool,
    /// Copied bytes, zero-padded after the selected width.
    pub value: [u8; 8],
}

/// Generates the DMA trace. One row per operation; padded to the next power of two
/// (min 4). Padding rows model an inactive one-byte step so unconditional constraints hold.
pub fn generate_dma_trace(
    ops: &[DmaOperation],
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
        let tail = op.count < 8;
        let width = if tail { 1 } else { 8 };
        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);

        table.set_dword_wl(row_idx, cols::SRC_0, op.src);
        table.set_dword_hl(row_idx, cols::SRC_INCR_0, op.src.wrapping_add(width));

        table.set_dword_wl(row_idx, cols::DST_0, op.dst);
        table.set_dword_hl(row_idx, cols::DST_INCR_0, op.dst.wrapping_add(width));

        table.set_dword_wl(row_idx, cols::COUNT_0, op.count);
        let count_decr = op.count.wrapping_sub(width);
        table.set_dword_hl(row_idx, cols::COUNT_DECR_0, count_decr);

        table.set_bool(row_idx, cols::FIRST, op.first);
        table.set_bool(row_idx, cols::END, op.end);
        table.set_bool(row_idx, cols::TAIL, tail);
        for (column, &byte) in cols::VALUE.iter().zip(&op.value) {
            table.set_byte(row_idx, *column, byte);
        }
        table.set_fe(row_idx, cols::MU, FE::one());
    }

    for row_idx in n..num_rows {
        table.set_fe(row_idx, cols::COUNT_0, FE::one());
        table.set_fe(row_idx, cols::SRC_INCR_0, FE::one());
        table.set_fe(row_idx, cols::DST_INCR_0, FE::one());
        table.set_fe(row_idx, cols::TAIL, FE::one());
    }

    trace
}

/// Helper: a MEMW register read (CO24, is_register=1, width2), value == old == the
/// register's two 32-bit limbs. Binds `x{reg}` to `(lo_col, hi_col)` at the ecall ts.
fn memw_register_read(reg_addr: u64, lo_col: usize, hi_col: usize) -> Vec<BusValue> {
    vec![
        // old[0..7] = [lo, hi, 0,0,0,0,0,0]
        BusValue::Packed {
            start_column: lo_col,
            packing: Packing::Direct,
        },
        BusValue::Packed {
            start_column: hi_col,
            packing: Packing::Direct,
        },
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(1),        // is_register = 1
        BusValue::constant(reg_addr), // base_address lo = 2*reg
        BusValue::constant(0),        // base_address hi
        // value[0..7] = same as old (a read leaves the value unchanged)
        BusValue::Packed {
            start_column: lo_col,
            packing: Packing::Direct,
        },
        BusValue::Packed {
            start_column: hi_col,
            packing: Packing::Direct,
        },
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        // timestamp
        BusValue::Packed {
            start_column: cols::TIMESTAMP_0,
            packing: Packing::Direct,
        },
        BusValue::Packed {
            start_column: cols::TIMESTAMP_1,
            packing: Packing::Direct,
        },
        BusValue::constant(1), // w2 = 1 (register = 2 words)
        BusValue::constant(0),
        BusValue::constant(0),
    ]
}

fn timestamp_with_offset(offset: i64) -> BusValue {
    BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::TIMESTAMP_0,
        },
        LinearTerm::Constant(offset),
    ])
}

fn value_columns() -> Vec<BusValue> {
    cols::VALUE
        .iter()
        .map(|&column| BusValue::Packed {
            start_column: column,
            packing: Packing::Direct,
        })
        .collect()
}

/// DMA memcpy bus interactions (23 total).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mu_minus_end = Multiplicity::Diff(cols::MU, cols::END);
    let mu_minus_first = Multiplicity::Diff(cols::MU, cols::FIRST);

    vec![
        // 1. Receive ECALL from CPU (mult = first).
        BusInteraction::receiver(
            BusId::Ecall,
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
                BusValue::constant(DMA_MEMCPY_LO32),
                BusValue::constant(DMA_MEMCPY_HI32),
            ],
        ),
        // 2. Send to DmaNext (mult = mu - end): [ts, src_incr, dst_incr, count_decr].
        BusInteraction::sender(
            BusId::DmaNext,
            mu_minus_end.clone(),
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
                    start_column: cols::SRC_INCR_0,
                    packing: Packing::DWordHL,
                },
                BusValue::Packed {
                    start_column: cols::DST_INCR_0,
                    packing: Packing::DWordHL,
                },
                BusValue::Packed {
                    start_column: cols::COUNT_DECR_0,
                    packing: Packing::DWordHL,
                },
            ],
        ),
        // 3. Receive from DmaNext (mult = mu - first): [ts, src, dst, count].
        BusInteraction::receiver(
            BusId::DmaNext,
            mu_minus_first,
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
                    start_column: cols::SRC_0,
                    packing: Packing::DWordWL,
                },
                BusValue::Packed {
                    start_column: cols::DST_0,
                    packing: Packing::DWordWL,
                },
                BusValue::Packed {
                    start_column: cols::COUNT_0,
                    packing: Packing::DWordWL,
                },
            ],
        ),
        // 4-7. IsHalfword: count_decr (mult = mu).
        halfword(cols::COUNT_DECR_0),
        halfword(cols::COUNT_DECR_1),
        halfword(cols::COUNT_DECR_2),
        halfword(cols::COUNT_DECR_3),
        // 8-11. IsHalfword: src_incr (mult = mu).
        halfword(cols::SRC_INCR_0),
        halfword(cols::SRC_INCR_1),
        halfword(cols::SRC_INCR_2),
        halfword(cols::SRC_INCR_3),
        // 12-15. IsHalfword: dst_incr (mult = mu).
        halfword(cols::DST_INCR_0),
        halfword(cols::DST_INCR_1),
        halfword(cols::DST_INCR_2),
        halfword(cols::DST_INCR_3),
        // 16. ZERO bus end detection: end == 1 iff all count_decr halfwords are 0xFFFF.
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
        // 17-19. Register reads (mult = first): x10 = dst, x11 = src, x12 = count.
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST),
            memw_register_read(20, cols::DST_0, cols::DST_1),
        ),
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST),
            memw_register_read(22, cols::SRC_0, cols::SRC_1),
        ),
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST),
            memw_register_read(24, cols::COUNT_0, cols::COUNT_1),
        ),
        // 20. ALU LT pins `tail = (count < 8)`.
        BusInteraction::sender(
            BusId::Alu,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: cols::COUNT_0,
                    packing: Packing::DWordWL,
                },
                BusValue::constant(8),
                BusValue::constant(0),
                BusValue::constant(alu_op::LT as u64),
                BusValue::Packed {
                    start_column: cols::TAIL,
                    packing: Packing::Direct,
                },
                BusValue::constant(0),
            ],
        ),
        // 21. The first row proves `count <= DMA_MEMCPY_MAX_BYTES`.
        BusInteraction::sender(
            BusId::Alu,
            Multiplicity::Column(cols::FIRST),
            vec![
                BusValue::Packed {
                    start_column: cols::COUNT_0,
                    packing: Packing::DWordWL,
                },
                BusValue::constant(DMA_MEMCPY_MAX_BYTES + 1),
                BusValue::constant(0),
                BusValue::constant(alu_op::LT as u64),
                BusValue::constant(1),
                BusValue::constant(0),
            ],
        ),
        // 22. MEMW read from src at T+1. `w8 = 1-tail`; old == value.
        BusInteraction::sender(BusId::Memw, mu_minus_end.clone(), {
            let mut values = value_columns();
            let mut tuple = Vec::with_capacity(24);
            tuple.extend(values.iter().cloned()); // old[8]
            tuple.push(BusValue::constant(0)); // is_register
            tuple.push(BusValue::Packed {
                start_column: cols::SRC_0,
                packing: Packing::Direct,
            });
            tuple.push(BusValue::Packed {
                start_column: cols::SRC_1,
                packing: Packing::Direct,
            });
            tuple.append(&mut values); // value[8]
            tuple.push(timestamp_with_offset(1));
            tuple.push(BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            });
            tuple.push(BusValue::constant(0)); // w2
            tuple.push(BusValue::constant(0)); // w4
            tuple.push(BusValue::linear(vec![
                LinearTerm::Constant(1),
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::TAIL,
                },
            ])); // w8 = 1-tail
            tuple
        }),
        // 23. MEMW write to dst at T+2, with the same value columns.
        BusInteraction::sender(BusId::Memw, mu_minus_end, {
            let mut tuple = Vec::with_capacity(16);
            tuple.push(BusValue::constant(0)); // is_register
            tuple.push(BusValue::Packed {
                start_column: cols::DST_0,
                packing: Packing::Direct,
            });
            tuple.push(BusValue::Packed {
                start_column: cols::DST_1,
                packing: Packing::Direct,
            });
            tuple.extend(value_columns());
            tuple.push(timestamp_with_offset(2));
            tuple.push(BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            });
            tuple.push(BusValue::constant(0)); // w2
            tuple.push(BusValue::constant(0)); // w4
            tuple.push(BusValue::linear(vec![
                LinearTerm::Constant(1),
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::TAIL,
                },
            ])); // w8
            tuple
        }),
    ]
}

/// An `IsHalfword` range-check sender for one halfword column (mult = mu).
fn halfword(column: usize) -> BusInteraction {
    BusInteraction::sender(
        BusId::IsHalfword,
        Multiplicity::Column(cols::MU),
        vec![BusValue::Packed {
            start_column: column,
            packing: Packing::Direct,
        }],
    )
}

/// The DMA table constraints:
/// - bitness for `first`, `end`, `tail`, `mu`;
/// - active first/end rows;
/// - `step = 8 - 7*tail` address/count arithmetic;
/// - unused bytes are zero on one-byte tail rows.
pub struct DmaConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for DmaConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::FIRST, None);
        emit_is_bit(b, 1, cols::END, None);
        emit_is_bit(b, 2, cols::TAIL, None);
        emit_is_bit(b, 3, cols::MU, None);

        let one = b.one();
        let first = b.main(0, cols::FIRST);
        let end = b.main(0, cols::END);
        let mu = b.main(0, cols::MU);
        b.emit_base(4, (first + end) * (one - mu));

        let step = AddOperand::linear(
            &[
                AddLinearTerm::Constant(8),
                AddLinearTerm::Column {
                    coefficient: -7,
                    column: cols::TAIL,
                },
            ],
            &[],
        );

        emit_add_pair_no_overflow(
            b,
            5,
            cols::MU,
            cols::END,
            &AddOperand::dword(cols::SRC_0),
            &step,
            &AddOperand::from_dword_hl(cols::SRC_INCR_0),
        );
        emit_add_pair_no_overflow(
            b,
            7,
            cols::MU,
            cols::END,
            &AddOperand::dword(cols::DST_0),
            &step,
            &AddOperand::from_dword_hl(cols::DST_INCR_0),
        );
        emit_add_pair(
            b,
            9,
            &[],
            &AddOperand::from_dword_hl(cols::COUNT_DECR_0),
            &step,
            &AddOperand::dword(cols::COUNT_0),
        );

        let tail = b.main(0, cols::TAIL);
        for (i, &column) in cols::VALUE.iter().enumerate().skip(1) {
            b.emit_base(11 + i - 1, tail.clone() * b.main(0, column));
        }
    }
}
