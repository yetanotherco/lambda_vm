//! DMA memset table — proves a `memset(dst, fill, n)` off the CPU execution trace.
//!
//! The guest's strong `memset` symbol (see `syscalls/src/syscalls.rs`) dispatches
//! bulk fills to the DMA memset ecall (`DMA_MEMSET_SYSCALL_NUMBER`); this table
//! proves the fill so the per-byte store loop leaves the CPU trace.
//!
//! Same streaming shape as the memcpy table (`dma.rs`): a row writes eight bytes
//! while `count >= 8`, otherwise one byte, and rows chain through `DmaSetNext`
//! until a terminal row where `count == 0`. The LT table pins that choice, so the
//! prover cannot select a convenient partition.
//!
//! Two things make this cheaper than memcpy rather than a copy of it:
//!
//! * **No source.** There is nothing to read, so a row emits one MEMW *write* at
//!   `T+1` and no read at all — half the memory traffic per byte. There is also
//!   no `src`/`src_incr` pair to carry or range-check.
//! * **No value lanes.** Every byte written is the same constant, so one `fill`
//!   column replaces memcpy's eight value columns. `fill_wide` is `fill` on
//!   eight-byte rows and zero on one-byte tail rows, which is what lets the same
//!   write tuple serve both widths without per-lane constraints.
//!
//! The result is 20 columns against memcpy's 32, and 18 bus interactions against
//! 23. `fill <= 255` is proven on the first row, mirroring how `dma.rs` proves
//! the per-ecall byte bound: the executor rejects a wider value, so an honest
//! guest (whose stub masks `a1`) never trips it.
//!
//! ## Columns (20 total)
//! - `timestamp`: DWordWL (2) — the ECALL timestamp
//! - `dst`: DWordWL (2) — current destination byte address
//! - `dst_incr`: DWordHL (4) — dst + selected width
//! - `count`: DWordWL (2) — remaining byte count (including this byte; 0 on the end row)
//! - `count_decr`: DWordHL (4) — count - width (all 0xFFFF when count == 0)
//! - `fill`: byte being written
//! - `fill_wide`: `fill` on eight-byte rows, 0 on one-byte tail rows
//! - `first`: Bit — first row of a fill
//! - `end`: Bit — last row (count was 0)
//! - `tail`: Bit — `count < 8`; selects a 1-byte rather than 8-byte row
//! - `mu`: Bit — multiplicity (1 real, 0 padding)
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use crate::constraints::templates::{
    AddLinearTerm, AddOperand, emit_add_pair, emit_add_pair_no_overflow, emit_is_bit,
};

use executor::vm::instruction::execution::{
    DMA_MEMCPY_MAX_BYTES as EXECUTOR_DMA_MEMCPY_MAX_BYTES,
    DMA_MEMSET_MAX_FILL as EXECUTOR_DMA_MEMSET_MAX_FILL, DMA_MEMSET_SYSCALL_NUMBER,
};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};

/// DMA memset syscall value, split into 32-bit limbs for the Ecall bus.
const DMA_MEMSET_LO32: u64 = DMA_MEMSET_SYSCALL_NUMBER & 0xFFFF_FFFF;
const DMA_MEMSET_HI32: u64 = DMA_MEMSET_SYSCALL_NUMBER >> 32;
/// Per-ecall byte bound, shared with memcpy so both stubs chunk identically.
pub const DMA_MEMSET_MAX_BYTES: u64 = EXECUTOR_DMA_MEMCPY_MAX_BYTES;
/// Largest accepted fill value, taken from the executor so the bound the AIR
/// proves cannot drift from the bound execution enforces.
pub const DMA_MEMSET_MAX_FILL: u64 = EXECUTOR_DMA_MEMSET_MAX_FILL;

pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;

    pub const DST_0: usize = 2;
    pub const DST_1: usize = 3;

    pub const DST_INCR_0: usize = 4;
    pub const DST_INCR_1: usize = 5;
    pub const DST_INCR_2: usize = 6;
    pub const DST_INCR_3: usize = 7;

    pub const COUNT_0: usize = 8;
    pub const COUNT_1: usize = 9;

    pub const COUNT_DECR_0: usize = 10;
    pub const COUNT_DECR_1: usize = 11;
    pub const COUNT_DECR_2: usize = 12;
    pub const COUNT_DECR_3: usize = 13;

    pub const FILL: usize = 14;
    pub const FILL_WIDE: usize = 15;

    pub const FIRST: usize = 16;
    pub const END: usize = 17;
    pub const TAIL: usize = 18;
    pub const MU: usize = 19;

    pub const NUM_COLUMNS: usize = 20;
}

/// One row of the DMA memset table: eight bytes, one tail byte, or the terminal row.
#[derive(Debug, Clone)]
pub struct DmaSetOperation {
    pub timestamp: u64,
    pub dst: u64,
    /// Remaining byte count (including this byte; 0 on the end row).
    pub count: u64,
    pub fill: u8,
    pub first: bool,
    pub end: bool,
}

/// Generates the DMA memset trace. One row per operation; padded to the next
/// power of two (min 4). Padding rows model an inactive one-byte step so the
/// unconditional `count_decr + step == count` relation still holds.
pub fn generate_dma_set_trace(
    ops: &[DmaSetOperation],
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

        table.set_dword_wl(row_idx, cols::DST_0, op.dst);
        table.set_dword_hl(row_idx, cols::DST_INCR_0, op.dst.wrapping_add(width));

        table.set_dword_wl(row_idx, cols::COUNT_0, op.count);
        table.set_dword_hl(row_idx, cols::COUNT_DECR_0, op.count.wrapping_sub(width));

        table.set_byte(row_idx, cols::FILL, op.fill);
        // Zero on tail rows so the shared write tuple narrows to a single byte.
        table.set_byte(row_idx, cols::FILL_WIDE, if tail { 0 } else { op.fill });

        table.set_bool(row_idx, cols::FIRST, op.first);
        table.set_bool(row_idx, cols::END, op.end);
        table.set_bool(row_idx, cols::TAIL, tail);
        table.set_fe(row_idx, cols::MU, FE::one());
    }

    for row_idx in n..num_rows {
        table.set_fe(row_idx, cols::COUNT_0, FE::one());
        table.set_fe(row_idx, cols::DST_INCR_0, FE::one());
        table.set_fe(row_idx, cols::TAIL, FE::one());
    }

    trace
}

/// Helper: a MEMW register read (CO24, is_register=1, width2), value == old == the
/// register's two 32-bit limbs. Binds `x{reg}` to `(lo_col, hi_col)` at the ecall ts.
fn memw_register_read(reg_addr: u64, lo_col: usize, hi_col: usize) -> Vec<BusValue> {
    let limb = |c: usize| BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    };
    vec![
        limb(lo_col),
        limb(hi_col),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(1),        // is_register = 1
        BusValue::constant(reg_addr), // base_address lo = 2*reg
        BusValue::constant(0),        // base_address hi
        limb(lo_col),
        limb(hi_col),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        BusValue::constant(0),
        limb(cols::TIMESTAMP_0),
        limb(cols::TIMESTAMP_1),
        BusValue::constant(1), // w2 = 1 (register = 2 words)
        BusValue::constant(0),
        BusValue::constant(0),
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

/// DMA memset bus interactions (18 total).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mu_minus_end = Multiplicity::Diff(cols::MU, cols::END);
    let mu_minus_first = Multiplicity::Diff(cols::MU, cols::FIRST);
    let direct = |c: usize| BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    };

    vec![
        // 1. Receive ECALL from CPU (mult = first).
        BusInteraction::receiver(
            BusId::Ecall,
            Multiplicity::Column(cols::FIRST),
            vec![
                direct(cols::TIMESTAMP_0),
                direct(cols::TIMESTAMP_1),
                BusValue::constant(DMA_MEMSET_LO32),
                BusValue::constant(DMA_MEMSET_HI32),
            ],
        ),
        // 2. Send to DmaSetNext (mult = mu - end): [ts, dst_incr, count_decr, fill].
        // `fill` rides the chain so every row of one call writes the same byte.
        BusInteraction::sender(
            BusId::DmaSetNext,
            mu_minus_end.clone(),
            vec![
                direct(cols::TIMESTAMP_0),
                direct(cols::TIMESTAMP_1),
                BusValue::Packed {
                    start_column: cols::DST_INCR_0,
                    packing: Packing::DWordHL,
                },
                BusValue::Packed {
                    start_column: cols::COUNT_DECR_0,
                    packing: Packing::DWordHL,
                },
                direct(cols::FILL),
            ],
        ),
        // 3. Receive from DmaSetNext (mult = mu - first): [ts, dst, count, fill].
        BusInteraction::receiver(
            BusId::DmaSetNext,
            mu_minus_first,
            vec![
                direct(cols::TIMESTAMP_0),
                direct(cols::TIMESTAMP_1),
                BusValue::Packed {
                    start_column: cols::DST_0,
                    packing: Packing::DWordWL,
                },
                BusValue::Packed {
                    start_column: cols::COUNT_0,
                    packing: Packing::DWordWL,
                },
                direct(cols::FILL),
            ],
        ),
        // 4-7. IsHalfword: count_decr (mult = mu).
        halfword(cols::COUNT_DECR_0),
        halfword(cols::COUNT_DECR_1),
        halfword(cols::COUNT_DECR_2),
        halfword(cols::COUNT_DECR_3),
        // 8-11. IsHalfword: dst_incr (mult = mu).
        halfword(cols::DST_INCR_0),
        halfword(cols::DST_INCR_1),
        halfword(cols::DST_INCR_2),
        halfword(cols::DST_INCR_3),
        // 12. ZERO bus end detection: end == 1 iff all count_decr halfwords are 0xFFFF.
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
                direct(cols::END),
            ],
        ),
        // 13-15. Register reads (mult = first): x10 = dst, x11 = fill, x12 = count.
        // x11's high limb is pinned to 0 by the constant below, so a fill wider
        // than 32 bits cannot be smuggled past the `fill <= 255` check.
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST),
            memw_register_read(20, cols::DST_0, cols::DST_1),
        ),
        BusInteraction::sender(BusId::Memw, Multiplicity::Column(cols::FIRST), {
            let mut tuple = memw_register_read(22, cols::FILL, cols::FILL);
            // x11 = (fill, 0): overwrite both high-limb slots with the constant 0.
            tuple[1] = BusValue::constant(0);
            tuple[12] = BusValue::constant(0);
            tuple
        }),
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST),
            memw_register_read(24, cols::COUNT_0, cols::COUNT_1),
        ),
        // 16. ALU LT pins `tail = (count < 8)`.
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
                direct(cols::TAIL),
                BusValue::constant(0),
            ],
        ),
        // 17. The first row proves `count <= DMA_MEMSET_MAX_BYTES`.
        BusInteraction::sender(
            BusId::Alu,
            Multiplicity::Column(cols::FIRST),
            vec![
                BusValue::Packed {
                    start_column: cols::COUNT_0,
                    packing: Packing::DWordWL,
                },
                BusValue::constant(DMA_MEMSET_MAX_BYTES + 1),
                BusValue::constant(0),
                BusValue::constant(alu_op::LT as u64),
                BusValue::constant(1),
                BusValue::constant(0),
            ],
        ),
        // 18. The first row proves `fill <= DMA_MEMSET_MAX_FILL`, so the byte the
        // write tuple broadcasts really is a byte.
        BusInteraction::sender(
            BusId::Alu,
            Multiplicity::Column(cols::FIRST),
            vec![
                // The ALU bus takes its left operand as two 32-bit limbs; `fill`
                // is a single byte column, so the high limb is a literal zero.
                direct(cols::FILL),
                BusValue::constant(0),
                BusValue::constant(DMA_MEMSET_MAX_FILL + 1),
                BusValue::constant(0),
                BusValue::constant(alu_op::LT as u64),
                BusValue::constant(1),
                BusValue::constant(0),
            ],
        ),
        // 19. MEMW write to dst at T+1. `w8 = 1-tail`; lanes 1..7 carry `fill_wide`,
        // which the constraints force to 0 exactly on one-byte tail rows.
        BusInteraction::sender(BusId::Memw, mu_minus_end, {
            let mut tuple = Vec::with_capacity(16);
            tuple.push(BusValue::constant(0)); // is_register
            tuple.push(direct(cols::DST_0));
            tuple.push(direct(cols::DST_1));
            tuple.push(direct(cols::FILL));
            for _ in 1..8 {
                tuple.push(direct(cols::FILL_WIDE));
            }
            tuple.push(BusValue::linear(vec![
                LinearTerm::Constant(1),
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::TIMESTAMP_0,
                },
            ]));
            tuple.push(direct(cols::TIMESTAMP_1));
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
    ]
}

/// The DMA memset constraints:
/// - bitness for `first`, `end`, `tail`, `mu`;
/// - active first/end rows;
/// - `step = 8 - 7*tail` address/count arithmetic;
/// - `fill_wide` equals `fill` on wide rows and 0 on tail rows.
#[derive(Clone, Copy)]
pub struct DmaSetConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for DmaSetConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::FIRST, None);
        emit_is_bit(b, 1, cols::END, None);
        emit_is_bit(b, 2, cols::TAIL, None);
        emit_is_bit(b, 3, cols::MU, None);

        let one = b.one();
        let first = b.main(0, cols::FIRST);
        let end = b.main(0, cols::END);
        let mu = b.main(0, cols::MU);
        b.emit_base(4, (first + end) * (one.clone() - mu));

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
            &AddOperand::dword(cols::DST_0),
            &step,
            &AddOperand::from_dword_hl(cols::DST_INCR_0),
        );
        emit_add_pair(
            b,
            7,
            &[],
            &AddOperand::from_dword_hl(cols::COUNT_DECR_0),
            &step,
            &AddOperand::dword(cols::COUNT_0),
        );

        // fill_wide == (1 - tail) * fill, expressed as the two cases so the
        // degree stays at 2: zero on tail rows, equal to fill otherwise.
        let tail = b.main(0, cols::TAIL);
        let fill = b.main(0, cols::FILL);
        let fill_wide = b.main(0, cols::FILL_WIDE);
        b.emit_base(9, tail.clone() * fill_wide.clone());
        b.emit_base(10, (one - tail) * (fill_wide - fill));
    }
}
