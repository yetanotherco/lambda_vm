//! COPY table — one streaming copy primitive for `memcpy`/`memmove`, `memset`
//! and the `write` syscall.
//!
//! Replaces the COMMIT table. A row copies `1` or `8`
//! bytes from `src` to `dst` and chains through [`BusId::CopyNext`] until the row
//! where `count == step`.
//!
//! ## The three functionalities
//!
//! One-hot over `is_set` and `is_write`; `is_cpy = mu - is_set - is_write` is linear
//! and costs no column. Neither the memory domain nor the timestamp order is chosen
//! by the caller: both are *derived* from the selector, and the selector is pinned to
//! the ecall the row receives, because the ECALL number is itself a linear function
//! of the two flags.
//!
//! | op | ecall | domains | timestamp order | arguments |
//! |---|---|---|---|---|
//! | memcpy / memmove | `-30` | RAM → RAM | read `T`, write `T+1` | `x10`, `x11`, `x12` |
//! | memset | `-31` | RAM → RAM | **write `T`, read `T+1`** | `x10`, `x11`, `x12` |
//! | write | `64` | RAM → COMMIT | read `T`, write `T+1` | `x10`, `x11`, `x12`, `x254` |
//!
//! ## Timestamp order
//!
//! ```text
//! read_ts  = T + is_set
//! write_ts = T + 1 - is_set
//! ```
//!
//! Both are linear in a bit column. With the normal order every read of a call
//! happens at one timestamp and every write at a later one, so a whole chunk is a
//! snapshot — that is what gives `memmove` its overlap semantics for free. With the
//! order inverted, a row's read observes the *previous* row's write, so a self-copy
//! propagates its first bytes across the range: `memset`. `old_ts < ts` holds strictly
//! either way, so the memory argument is undisturbed.
//!
//! `memset` reuses the copy path, but not without a constraint of its own. Its stub
//! seeds the first eight bytes with an ordinary store and calls with
//! `(dst = seed_end, src = seed_start, count = n - 8)`, so the chip sees an
//! overlapping memmove -- and `copy:c:set_gap` is what pins that overlap to exactly
//! `dst = src + 8`. Drop it and `dst == src` becomes provable, where a row's read and
//! its write hit the same cell at adjacent timestamps and all eight lanes go free:
//! an arbitrary memory write. See the comment on `set_gap` in `eval`.
//!
//! ## Return value
//!
//! All four operations share the `write(2)` ABI: `x10`, `x11`, `x12` hold the three
//! operands and `x10` receives the return value. `copy:c:read_dst` therefore reads
//! `dst` from `x10 + 244·is_write` — `x10` for the `mem*` calls, `x254` (the running
//! commit index) for `write` — and writes `dst_res = dst + count·is_write` back. One
//! interaction covers both the `void*` the `mem*` calls return and the commit-index
//! update `write` needs. `copy:c:read_fd` additionally pins `x10 == 1` on the first
//! row of a `write` (only `stdout` is supported) and returns `count` there.
//!
//! ## Width is chosen per row
//!
//! `single` is free upward: a one-byte row is legal at any count. Downward it is
//! pinned exactly, and locally. A wide row with `count < 8` borrows out of
//! `count - step`, and `copy:c:borrow_implies_count_is_zero` then demands
//! `count == 0` — a contradiction. So a chain can never overshoot, and the prover is
//! *forced* to take narrow rows for the last seven bytes. That is what replaced the
//! `count < 8` ALU lookup, and it is stronger than the lookup was.
//!
//! The `count <= 256` lookup went for an unrelated reason: the spec places no bound
//! on `count` and leaves the chain length to the prover. The executor still refuses
//! an ecall above 256 bytes, so it is now strictly stricter than this chip — an
//! intentional acceptance-set gap, benign (the same bytes, in fewer ecalls), with
//! the executor's limit the side to relax.
//!
//! ## Terminating the chain
//!
//! There is no separate terminal row. The chain stops on the row that moves the last
//! bytes, and `end` is pinned to `count == step or (count == 0 and first)` by the
//! `restrict_end` group, using the borrow out of `count - step` as the proxy for the
//! second (zero-length) state. The old `ZERO` lookup over `count_decr` is gone.
//!
//! ## Columns (41)
//!
//! - `timestamp` DWordWL (2), `value` DWordBL (8)
//! - `src` DWordWL (2), `src_incr` DWordHL (4)
//! - `dst` DWordWL (2) — for `write` this is the COMMIT-domain address, i.e. the
//!   running global byte index — `dst_incr` DWordHL (4)
//! - `count` DWordWL (2), `count_decr` DWordHL (4)
//! - `dst_res` DWordHL (4)
//! - `first`, `end`, `single`, `is_set`, `is_write`, `first_write`, `mu`
//! - `mu_com = (mu - count_borrows[1]) * is_write`,
//!   `mu_com_wide = mu_com * (1 - single)` — multiplicities are strictly linear in
//!   this framework, so an op-specific gate that is not already linear needs a column
//!   and a degree-2 constraint. The RAM write needs none: it rides
//!   `mu - count_borrows[1] - mu_com` directly, which is the same product expanded.
//!
//! Three divergences in the columns from the spec's 34, all representational and all
//! recorded in `shape_tests`: the spec types `timestamp` as a `Word` where this code uses a
//! `DWordWL`; it types the three increments and `dst_res` as `DWordWHH`
//! (`Half, Half, Word`) where this code uses `DWordHL` (four `Half`s). Four `IS_HALF`
//! lookups prove the same bound `IS_WHH` would, at one extra column each; moving these
//! four to `IS_WHH` and the `IS_WH` table that KECCAK's `state_ptr` already uses would
//! drop those columns. `mu_com`/`mu_com_wide` are the COMMIT-bus gate the spec does not
//! need because it writes the committed bytes through `MEMW` in domain `2·is_write`.
//!
//! A fourth divergence is not representational: constraints 37, 38 and 39 pin the
//! carry out of `src + step`, `dst + step` and the `memset` gap `src + 8` to zero.
//! The spec has none of the three -- its `ADD` template range-checks both carries as
//! bits and stops there, and the `ADDNW` template that used to say otherwise is gone
//! from the spec entirely -- and without them the chip accepts operands that wrap the
//! address space, which the executor refuses outright. Extra constraints only ever
//! narrow the acceptance set, so this is soundness-safe; it narrows it to the
//! executor's.
//!
//! One consequence of that last column divergence is worth stating here rather than
//! leaving for a reader to find: the `Commit` bus carries the index as `dst[0]`
//! alone, while the verifier enumerates `start_index + i` over a `u64`. The two
//! agree only while the running commit index stays below `2^32`, and `x254`'s high
//! word is pinned to zero by the trace builder rather than by a constraint.
//!
//! What keeps that safe is not the prover's own range check -- that is a backstop
//! for a bug, and it is unreachable. It is the executor: `commit_public_output`
//! caps the total committed output at `MAX_PUBLIC_OUTPUT_TOTAL_SIZE`, one mebibyte,
//! so the index of an honest run stays four orders of magnitude below `2^32`. A run
//! that tried to pass the cap fails in the executor, long before any trace exists.
//! Raising that cap is what would make this a live question, and the domain-2 write
//! is what retires it.
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use crate::constraints::templates::{
    AddLinearTerm, AddOperand, add_pair_carry_hi, emit_add_pair, emit_is_bit,
};

use executor::vm::instruction::execution::{
    DMA_MEMCPY_SYSCALL_NUMBER, DMA_MEMSET_GAP, DMA_MEMSET_SYSCALL_NUMBER,
};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};

/// `write(2)`, the one ECALL number this chip serves that is not an extension.
const WRITE_SYSCALL_NUMBER: u64 = 64;

const MEMCPY_LO32: i64 = (DMA_MEMCPY_SYSCALL_NUMBER & 0xFFFF_FFFF) as i64;
const MEMCPY_HI32: i64 = (DMA_MEMCPY_SYSCALL_NUMBER >> 32) as i64;
const MEMSET_LO32: i64 = (DMA_MEMSET_SYSCALL_NUMBER & 0xFFFF_FFFF) as i64;
const MEMSET_HI32: i64 = (DMA_MEMSET_SYSCALL_NUMBER >> 32) as i64;
const WRITE_LO32: i64 = (WRITE_SYSCALL_NUMBER & 0xFFFF_FFFF) as i64;
const WRITE_HI32: i64 = (WRITE_SYSCALL_NUMBER >> 32) as i64;

/// `x10 + 244·is_write`: the register `dst` is read from, and `dst_res` written to.
const DST_REGISTER_BASE: u8 = 10;
const DST_REGISTER_WRITE_OFFSET: i64 = 244;

pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;

    pub const VALUE_0: usize = 2;
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

    pub const SRC_0: usize = 10;
    pub const SRC_1: usize = 11;
    /// `src + step`, four halfwords. The `ADD` leaves the carry out of the high limb
    /// free; constraints 37 and 38 pin it to zero on every row that moves a byte, so
    /// a chain cannot walk past the top of the address space. See the end of `eval`.
    pub const SRC_INCR_0: usize = 12;

    pub const DST_0: usize = 16;
    pub const DST_1: usize = 17;
    /// `dst + step`; see [`SRC_INCR_0`].
    pub const DST_INCR_0: usize = 18;

    pub const COUNT_0: usize = 22;
    pub const COUNT_1: usize = 23;
    /// `count - step`, four halfwords. The zero-length call relies on the wraparound
    /// to `-1`, and that borrow is `count_borrows[1]`, the proxy for `end`.
    pub const COUNT_DECR_0: usize = 24;

    /// `dst + count · is_write`, four halfwords: the value written back to
    /// `x10 + 244·is_write`.
    pub const DST_RES_0: usize = 28;

    pub const FIRST: usize = 32;
    pub const END: usize = 33;
    /// `1` when this row moves a single byte rather than eight.
    pub const SINGLE: usize = 34;

    /// Decoded functionality. `is_cpy = mu - is_set - is_write` is implied.
    pub const IS_SET: usize = 35;
    pub const IS_WRITE: usize = 36;
    /// `first · is_write` — the `x10` file-descriptor check.
    pub const FIRST_WRITE: usize = 37;

    pub const MU: usize = 38;

    /// `(mu - count_borrows[1]) · is_write` — the COMMIT-domain write.
    pub const MU_COM: usize = 39;
    /// `mu_com · (1 - single)` — lanes 1..7 of the COMMIT-domain write. Without it a
    /// one-byte `write` row would send seven spurious `(index, 0)` pairs and corrupt
    /// the public-output fingerprint.
    pub const MU_COM_WIDE: usize = 40;

    pub const NUM_COLUMNS: usize = 41;
}

/// Which functionality a row is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Functionality {
    /// `memcpy` / `memmove`: RAM → RAM, snapshot order.
    Copy,
    /// `memset`: RAM → RAM, inverted order, so the fill propagates.
    Set,
    /// `write`: RAM → COMMIT domain, snapshot order.
    Write,
}

/// One row: `1` or `8` bytes, or the degenerate zero-length row.
#[derive(Debug, Clone)]
pub struct CopyOperation {
    pub timestamp: u64,
    pub src: u64,
    /// For `Write` this is the COMMIT-domain address (the global byte index).
    pub dst: u64,
    /// Remaining byte count including this row's bytes; `0` only on a zero-length call.
    pub count: u64,
    pub width: u8,
    pub first: bool,
    pub end: bool,
    pub functionality: Functionality,
    /// The bytes moved, zero-padded past `width`.
    pub value: [u8; 8],
}

impl CopyOperation {
    /// `read_ts = T + is_set`, `write_ts = T + 1 - is_set`.
    pub fn read_timestamp(&self) -> u64 {
        self.timestamp + u64::from(self.functionality == Functionality::Set)
    }

    pub fn write_timestamp(&self) -> u64 {
        self.timestamp + 1 - u64::from(self.functionality == Functionality::Set)
    }

    /// `dst + count · is_write`, the value written back to `x10`/`x254`.
    pub fn dst_res(&self) -> u64 {
        match self.functionality {
            Functionality::Write => self.dst.wrapping_add(self.count),
            _ => self.dst,
        }
    }
}

/// Generates the COPY trace. One row per operation, padded to the next power of
/// two (min 4). Padding rows carry `count = 8` and `src_incr = dst_incr = 8`, which
/// is what keeps the unconditional address/count relations and both borrows at zero.
pub fn generate_copy_trace(
    ops: &[CopyOperation],
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
        let width = u64::from(op.width);
        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);

        for (column, &byte) in cols::VALUE.iter().zip(&op.value) {
            table.set_byte(row_idx, *column, byte);
        }

        table.set_dword_wl(row_idx, cols::SRC_0, op.src);
        table.set_dword_hl(row_idx, cols::SRC_INCR_0, op.src.wrapping_add(width));

        table.set_dword_wl(row_idx, cols::DST_0, op.dst);
        table.set_dword_hl(row_idx, cols::DST_INCR_0, op.dst.wrapping_add(width));

        table.set_dword_wl(row_idx, cols::COUNT_0, op.count);
        table.set_dword_hl(row_idx, cols::COUNT_DECR_0, op.count.wrapping_sub(width));

        table.set_dword_hl(row_idx, cols::DST_RES_0, op.dst_res());

        table.set_bool(row_idx, cols::FIRST, op.first);
        table.set_bool(row_idx, cols::END, op.end);
        table.set_bool(row_idx, cols::SINGLE, op.width == 1);
        table.set_fe(row_idx, cols::MU, FE::one());

        let is_set = op.functionality == Functionality::Set;
        let is_write = op.functionality == Functionality::Write;
        table.set_bool(row_idx, cols::IS_SET, is_set);
        table.set_bool(row_idx, cols::IS_WRITE, is_write);
        table.set_bool(row_idx, cols::FIRST_WRITE, op.first && is_write);

        // `mu - count_borrows[1]`: the row copies unless it is the degenerate
        // zero-length row, which is the only one that borrows out of `count - step`.
        let copies = op.count != 0;
        table.set_bool(row_idx, cols::MU_COM, copies && is_write);
        table.set_bool(
            row_idx,
            cols::MU_COM_WIDE,
            copies && is_write && op.width == 8,
        );
    }

    for row_idx in n..num_rows {
        // `single = 0` on padding, so `step = 8` and the three unconditional ADD
        // relations close with `count = src_incr = dst_incr = 8` and everything else
        // zero. Both borrows are then zero, so no padding row touches the memory bus.
        table.set_fe(row_idx, cols::COUNT_0, FE::from(8u64));
        table.set_fe(row_idx, cols::SRC_INCR_0, FE::from(8u64));
        table.set_fe(row_idx, cols::DST_INCR_0, FE::from(8u64));
    }

    trace
}

/// `-1 / 2^32`, `-1 / 2^16` and friends, for the two borrows of `count - step`.
///
/// `count_borrows` is virtual in the spec — an expression, not a column — and the
/// read/write multiplicity `mu - count_borrows[1]` therefore has to be written out
/// as a linear form over the columns it is built from.
mod borrow_consts {
    use crate::constraints::templates::INV_SHIFT_32;
    const P: u128 = 0xFFFF_FFFF_0000_0001;
    const fn neg(x: u64) -> u64 {
        (P - x as u128) as u64
    }
    const fn mul(a: u64, b: u64) -> u64 {
        ((a as u128 * b as u128) % P) as u64
    }
    /// `2^-32`
    pub const INV_2_32: u64 = INV_SHIFT_32;
    /// `2^-16` (as `2^16 · 2^-32`)
    pub const INV_2_16: u64 = mul(1 << 16, INV_2_32);
    /// `2^-64`
    pub const INV_2_64: u64 = mul(INV_2_32, INV_2_32);
    /// `2^-48`
    pub const INV_2_48: u64 = mul(INV_2_16, INV_2_32);
    pub const NEG_INV_2_32: u64 = neg(INV_2_32);
    pub const NEG_INV_2_64: u64 = neg(INV_2_64);
    pub const NEG_SEVEN_INV_2_64: u64 = neg(mul(7, INV_2_64));
    pub const EIGHT_INV_2_64: u64 = mul(8, INV_2_64);
    /// `8 / 2^64 = -2^35` in the field: `2^-64 = -2^32`, since `2^96 = -1`.
    pub const EIGHT_INV_2_64_SIGNED: i64 = -(1 << 35);
}

/// `count_borrows[1]`, the borrow out of `count - step`, as a linear form.
///
/// ```text
/// borrow_0 = (count_decr_lo + step - count_0) / 2^32
/// borrow_1 = (count_decr_hi + borrow_0 - count_1) / 2^32
/// ```
///
/// expanded over the four `count_decr` halfwords, the two `count` words and `single`.
/// `8 / 2^64` is `-2^35` in the field, which is how a term that looks like it could
/// not fit an `i64` constant fits one comfortably. On a padding row it cancels
/// against `-count_0 / 2^64 = 2^35` exactly, which is what the `count = 8` padding
/// is for.
fn count_borrow_1_terms() -> Vec<LinearTerm> {
    use borrow_consts::*;
    vec![
        LinearTerm::ColumnUnsigned {
            coefficient: INV_2_32,
            column: cols::COUNT_DECR_0 + 2,
        },
        LinearTerm::ColumnUnsigned {
            coefficient: INV_2_16,
            column: cols::COUNT_DECR_0 + 3,
        },
        LinearTerm::ColumnUnsigned {
            coefficient: NEG_INV_2_32,
            column: cols::COUNT_1,
        },
        LinearTerm::ColumnUnsigned {
            coefficient: INV_2_64,
            column: cols::COUNT_DECR_0,
        },
        LinearTerm::ColumnUnsigned {
            coefficient: INV_2_48,
            column: cols::COUNT_DECR_0 + 1,
        },
        LinearTerm::ColumnUnsigned {
            coefficient: NEG_INV_2_64,
            column: cols::COUNT_0,
        },
        LinearTerm::ColumnUnsigned {
            coefficient: NEG_SEVEN_INV_2_64,
            column: cols::SINGLE,
        },
        LinearTerm::Constant(EIGHT_INV_2_64_SIGNED),
    ]
}

/// `mu - count_borrows[1]`, the multiplicity of the two memory accesses.
fn mu_minus_borrow() -> Multiplicity {
    let mut terms = vec![LinearTerm::Column {
        coefficient: 1,
        column: cols::MU,
    }];
    terms.extend(negate(count_borrow_1_terms()));
    Multiplicity::Linear(terms)
}

fn negate(terms: Vec<LinearTerm>) -> Vec<LinearTerm> {
    const P: u128 = 0xFFFF_FFFF_0000_0001;
    terms
        .into_iter()
        .map(|term| match term {
            LinearTerm::Column {
                coefficient,
                column,
            } => LinearTerm::Column {
                coefficient: -coefficient,
                column,
            },
            LinearTerm::ColumnUnsigned {
                coefficient,
                column,
            } => LinearTerm::ColumnUnsigned {
                coefficient: (P - coefficient as u128) as u64,
                column,
            },
            LinearTerm::Constant(c) => LinearTerm::Constant(-c),
        })
        .collect()
}

/// A MEMW register access (`is_register = 1`, width 2) on the 24-element tuple:
/// `[old[8], is_register, address[2], value[8], timestamp[2], w2, w4, w8]`.
///
/// `address` is a `BusValue` rather than a constant because `copy:c:read_dst`
/// addresses `x10` or `x254` depending on `is_write`.
fn memw_register(address: BusValue, old: [BusValue; 2], value: [BusValue; 2]) -> Vec<BusValue> {
    let pad6 = || (0..6).map(|_| BusValue::constant(0));
    let mut tuple = Vec::with_capacity(24);
    tuple.extend(old);
    tuple.extend(pad6());
    tuple.push(BusValue::constant(1)); // is_register
    tuple.push(address);
    tuple.push(BusValue::constant(0)); // address high limb
    tuple.extend(value);
    tuple.extend(pad6());
    tuple.push(BusValue::Packed {
        start_column: cols::TIMESTAMP_0,
        packing: Packing::Direct,
    });
    tuple.push(BusValue::Packed {
        start_column: cols::TIMESTAMP_1,
        packing: Packing::Direct,
    });
    tuple.push(BusValue::constant(1)); // w2
    tuple.push(BusValue::constant(0)); // w4
    tuple.push(BusValue::constant(0)); // w8
    tuple
}

fn direct(column: usize) -> BusValue {
    BusValue::Packed {
        start_column: column,
        packing: Packing::Direct,
    }
}

/// The two 32-bit words of a `DWordHL` quadruple, as the `DWordWL` the bus wants.
fn word2l_pair(start_column: usize) -> [BusValue; 2] {
    [
        BusValue::Packed {
            start_column,
            packing: Packing::Word2L,
        },
        BusValue::Packed {
            start_column: start_column + 2,
            packing: Packing::Word2L,
        },
    ]
}

/// `T + offset + coefficient · is_set`, the timestamp-order customisation.
fn timestamp_with_order(offset: i64, is_set_coefficient: i64) -> BusValue {
    BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::TIMESTAMP_0,
        },
        LinearTerm::Column {
            coefficient: is_set_coefficient,
            column: cols::IS_SET,
        },
        LinearTerm::Constant(offset),
    ])
}

fn value_columns() -> Vec<BusValue> {
    cols::VALUE.iter().map(|&column| direct(column)).collect()
}

fn halfword(column: usize, multiplicity: Multiplicity) -> BusInteraction {
    BusInteraction::sender(BusId::IsHalfword, multiplicity, vec![direct(column)])
}

/// The COPY bus interactions.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mu_minus_end = Multiplicity::Diff(cols::MU, cols::END);
    let mu_minus_first = Multiplicity::Diff(cols::MU, cols::FIRST);
    let w8 = || {
        BusValue::linear(vec![
            LinearTerm::Constant(1),
            LinearTerm::Column {
                coefficient: -1,
                column: cols::SINGLE,
            },
        ])
    };
    let reg = |index: u8| BusValue::constant(2 * index as u64);

    let mut interactions = vec![
        // 1. Receive the ECALL. The syscall number is a linear function of the two
        //    selectors, so the decoded functionality is pinned to the ecall the guest
        //    actually made: `64` for `write`, `-30` for memcpy/memmove, `-31` for
        //    memset, and nothing else.
        BusInteraction::receiver(
            BusId::Ecall,
            Multiplicity::Column(cols::FIRST),
            vec![
                direct(cols::TIMESTAMP_0),
                direct(cols::TIMESTAMP_1),
                BusValue::linear(vec![
                    LinearTerm::Constant(MEMCPY_LO32),
                    LinearTerm::Column {
                        coefficient: MEMSET_LO32 - MEMCPY_LO32,
                        column: cols::IS_SET,
                    },
                    LinearTerm::Column {
                        coefficient: WRITE_LO32 - MEMCPY_LO32,
                        column: cols::IS_WRITE,
                    },
                ]),
                BusValue::linear(vec![
                    LinearTerm::Constant(MEMCPY_HI32),
                    LinearTerm::Column {
                        coefficient: MEMSET_HI32 - MEMCPY_HI32,
                        column: cols::IS_SET,
                    },
                    LinearTerm::Column {
                        coefficient: WRITE_HI32 - MEMCPY_HI32,
                        column: cols::IS_WRITE,
                    },
                ]),
            ],
        ),
        // 2. Read `src` from x11.
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST),
            memw_register(
                reg(11),
                [direct(cols::SRC_0), direct(cols::SRC_1)],
                [direct(cols::SRC_0), direct(cols::SRC_1)],
            ),
        ),
        // 3. Read `count` from x12.
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST),
            memw_register(
                reg(12),
                [direct(cols::COUNT_0), direct(cols::COUNT_1)],
                [direct(cols::COUNT_0), direct(cols::COUNT_1)],
            ),
        ),
        // 4. Read `dst` from x10 (the `mem*` destination) or x254 (the running commit
        //    index), and write `dst_res` back: the `void*` those calls return, and the
        //    advanced index a `write` needs. One interaction, selected by `is_write`.
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST),
            memw_register(
                BusValue::linear(vec![
                    LinearTerm::Constant(2 * DST_REGISTER_BASE as i64),
                    LinearTerm::Column {
                        coefficient: 2 * DST_REGISTER_WRITE_OFFSET,
                        column: cols::IS_WRITE,
                    },
                ]),
                [direct(cols::DST_0), direct(cols::DST_1)],
                word2l_pair(cols::DST_RES_0),
            ),
        ),
        // 5. On the first row of a `write`, x10 holds the file descriptor and must be
        //    1 (only stdout is supported); the same access returns `count`, the number
        //    of bytes written, because this chip always writes all of them.
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST_WRITE),
            memw_register(
                reg(10),
                [BusValue::constant(1), BusValue::constant(0)],
                [direct(cols::COUNT_0), direct(cols::COUNT_1)],
            ),
        ),
        // 6. Chain forward. The selectors ride inside the tuple, so a chain cannot
        //    change functionality half way through it.
        BusInteraction::sender(
            BusId::CopyNext,
            mu_minus_end.clone(),
            vec![
                direct(cols::TIMESTAMP_0),
                direct(cols::TIMESTAMP_1),
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
                direct(cols::IS_SET),
                direct(cols::IS_WRITE),
            ],
        ),
        // 7. Chain backward.
        BusInteraction::receiver(
            BusId::CopyNext,
            mu_minus_first,
            vec![
                direct(cols::TIMESTAMP_0),
                direct(cols::TIMESTAMP_1),
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
                direct(cols::IS_SET),
                direct(cols::IS_WRITE),
            ],
        ),
    ];

    // 8-19. `IS_WHH` on the three chained dwords, as four `IS_HALF` each.
    for start in [cols::SRC_INCR_0, cols::DST_INCR_0, cols::COUNT_DECR_0] {
        for k in 0..4 {
            interactions.push(halfword(start + k, Multiplicity::Column(cols::MU)));
        }
    }
    // 20-23. `IS_WHH[dst_res]`, which only the `ADD` of interaction 4 needs, so it is
    // gated on `is_write` exactly as the spec gates it. On a non-write row `dst_res`
    // is pinned equal to `dst` instead.
    for k in 0..4 {
        interactions.push(halfword(
            cols::DST_RES_0 + k,
            Multiplicity::Column(cols::IS_WRITE),
        ));
    }

    interactions.extend([
        // 24. Read the source at `T + is_set`.
        BusInteraction::sender(BusId::Memw, mu_minus_borrow(), {
            let mut values = value_columns();
            let mut tuple = Vec::with_capacity(24);
            tuple.extend(values.iter().cloned());
            tuple.push(BusValue::constant(0)); // is_register
            tuple.push(direct(cols::SRC_0));
            tuple.push(direct(cols::SRC_1));
            tuple.append(&mut values);
            tuple.push(timestamp_with_order(0, 1));
            tuple.push(direct(cols::TIMESTAMP_1));
            tuple.push(BusValue::constant(0)); // w2
            tuple.push(BusValue::constant(0)); // w4
            tuple.push(w8());
            tuple
        }),
        // 25. Write the destination at `T + 1 - is_set`, RAM domain only. The spec
        //     sends this one interaction with domain `2 · is_write`; this prover has
        //     no domain-2 endpoint on the memory bus yet, so a `write` row takes the
        //     COMMIT bus below instead and this send is gated off for it. The gate is
        //     `mu - count_borrows[1] - mu_com`, which is the same product expanded.
        BusInteraction::sender(
            BusId::Memw,
            {
                let Multiplicity::Linear(mut terms) = mu_minus_borrow() else {
                    unreachable!("mu_minus_borrow is Linear")
                };
                terms.push(LinearTerm::Column {
                    coefficient: -1,
                    column: cols::MU_COM,
                });
                Multiplicity::Linear(terms)
            },
            {
                let mut tuple = Vec::with_capacity(16);
                tuple.push(BusValue::constant(0)); // is_register
                tuple.push(direct(cols::DST_0));
                tuple.push(direct(cols::DST_1));
                tuple.extend(value_columns());
                tuple.push(timestamp_with_order(1, -1));
                tuple.push(direct(cols::TIMESTAMP_1));
                tuple.push(BusValue::constant(0)); // w2
                tuple.push(BusValue::constant(0)); // w4
                tuple.push(w8());
                tuple
            },
        ),
    ]);

    // 26-33. Write the destination in the COMMIT domain, one `(index, value)` pair per
    // byte. A row still carries eight bytes, but it sends them as eight separate pairs
    // at `dst`, `dst + 1`, ..., rather than as one eight-lane tuple.
    //
    // The arity is what matters here, not the row width. The verifier rebuilds this bus
    // from `public_output` alone, and it does not know where one `write` ECALL ended and
    // the next began — it sees only the concatenation. With one tuple per row the
    // verifier would have to reproduce the prover's row schedule exactly, which it
    // cannot: the schedule restarts at every ECALL, so a guest committing 4 bytes and
    // then 4 more sends eight one-byte rows where the verifier, chunking the eight
    // bytes it sees, would expect a single eight-byte row. An honest proof would be
    // rejected. Addressing every byte by its own global index removes the grouping, and
    // with it anything for the two sides to disagree about.
    //
    // Lane 0 is sent whenever the row copies (`mu_com`); lanes 1..7 only on an
    // eight-byte row (`mu_com_wide`), so a one-byte row does not send seven spurious
    // `(index, 0)` pairs.
    let commit_pair = |lane: usize| {
        vec![
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::DST_0,
                },
                LinearTerm::Constant(lane as i64),
            ]),
            direct(cols::VALUE[lane]),
        ]
    };
    interactions.push(BusInteraction::sender(
        BusId::Commit,
        Multiplicity::Column(cols::MU_COM),
        commit_pair(0),
    ));
    for lane in 1..8 {
        interactions.push(BusInteraction::sender(
            BusId::Commit,
            Multiplicity::Column(cols::MU_COM_WIDE),
            commit_pair(lane),
        ));
    }

    interactions
}

/// The COPY constraints.
#[derive(Clone, Copy)]
pub struct CopyConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for CopyConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // `bits`: copy:c:range_first / range_end / range_single / range_mu, and
        // `flags`: copy:c:range_is_set / range_is_write.
        emit_is_bit(b, 0, cols::FIRST, None);
        emit_is_bit(b, 1, cols::END, None);
        emit_is_bit(b, 2, cols::SINGLE, None);
        emit_is_bit(b, 3, cols::MU, None);
        emit_is_bit(b, 4, cols::IS_SET, None);
        emit_is_bit(b, 5, cols::IS_WRITE, None);

        let one = b.one();
        let first = b.main(0, cols::FIRST);
        let end = b.main(0, cols::END);
        let mu = b.main(0, cols::MU);
        let single = b.main(0, cols::SINGLE);
        let is_set = b.main(0, cols::IS_SET);
        let is_write = b.main(0, cols::IS_WRITE);

        // copy:c:first_or_end_implies_mu. An active row is implied by first or end,
        // which keeps the chain multiplicities `mu - first` and `mu - end` binary.
        b.emit_base(
            6,
            (first.clone() + end.clone()) * (one.clone() - mu.clone()),
        );
        // copy:c:one_hot, copy:c:functionality_implies_mu, copy:c:first_write. The
        // functionality is one-hot and only set on active rows.
        b.emit_base(7, is_set.clone() * is_write.clone());
        b.emit_base(
            8,
            (is_set.clone() + is_write.clone()) * (one.clone() - mu.clone()),
        );
        b.emit_base(
            9,
            b.main(0, cols::FIRST_WRITE) - first.clone() * is_write.clone(),
        );

        let step = AddOperand::linear(
            &[
                AddLinearTerm::Constant(8),
                AddLinearTerm::Column {
                    coefficient: -7,
                    column: cols::SINGLE,
                },
            ],
            &[],
        );

        // copy:c:src_incr and copy:c:dst_incr, plain `ADD`. The carry out of the high
        // limb is free here, exactly as the spec's `ADD` leaves it; the two
        // constraints at the end of `eval` are what forbid it. Nothing else does:
        // `count` decrements independently of the two positions, so a wrapping chain
        // terminates perfectly well.
        emit_add_pair(
            b,
            10,
            &[],
            &AddOperand::dword(cols::SRC_0),
            &step,
            &AddOperand::from_dword_hl(cols::SRC_INCR_0),
        );
        emit_add_pair(
            b,
            12,
            &[],
            &AddOperand::dword(cols::DST_0),
            &step,
            &AddOperand::from_dword_hl(cols::DST_INCR_0),
        );
        // copy:c:count_borrows: `count_decr + step = count`, the `SUB` the spec inlines.
        // The two carries are `count_borrows`, the borrows
        // out of `count - step`; `count_borrows[1]` is what `restrict_end` reads.
        emit_add_pair(
            b,
            14,
            &[],
            &AddOperand::from_dword_hl(cols::COUNT_DECR_0),
            &step,
            &AddOperand::dword(cols::COUNT_0),
        );

        // copy:c:dst_res_is_write and copy:c:dst_res_non_write.
        emit_add_pair(
            b,
            16,
            &[cols::IS_WRITE],
            &AddOperand::dword(cols::DST_0),
            &AddOperand::dword(cols::COUNT_0),
            &AddOperand::from_dword_hl(cols::DST_RES_0),
        );
        let dst_res_lo =
            b.main(0, cols::DST_RES_0) + b.const_base(1 << 16) * b.main(0, cols::DST_RES_0 + 1);
        let dst_res_hi =
            b.main(0, cols::DST_RES_0 + 2) + b.const_base(1 << 16) * b.main(0, cols::DST_RES_0 + 3);
        b.emit_base(
            18,
            (one.clone() - is_write.clone()) * (dst_res_lo - b.main(0, cols::DST_0)),
        );
        b.emit_base(
            19,
            (one.clone() - is_write) * (dst_res_hi - b.main(0, cols::DST_1)),
        );

        // copy:c:set_gap -- memset's operand contract: `dst = src + 8`.
        //
        // This is what pins `value` on an `is_set` row. The inverted order puts the
        // write at `T` and the read at `T+1`, so a row's read observes writes this
        // call already made -- that is the propagation. It pins the copied value only
        // while the read resolves to a *different* address, already written, with the
        // recursion bottoming out in memory the call never wrote. `dst == src` is the
        // one degenerate case: read and write address the same cell at adjacent
        // timestamps, the memory argument closes on `value == value`, and all eight
        // lanes become free field elements -- unconstrained RAM, chosen by the prover.
        // Nothing else touches them (the lane constraints below only *zero* lanes 1..7
        // on narrow rows, and no `AreBytes` reaches them), so this is the only thing
        // standing between the chip and an arbitrary memory write.
        //
        // `is_set` alone is the correct gate. `step` advances `src` and `dst` together,
        // so `dst - src` is invariant along a chain and the relation holds on every row
        // of it; padding rows leave `is_set = 0`.
        //
        // Pinning the exact gap rather than merely `src != dst` also settles the
        // direction: `dst < src` is sound but propagates the wrong way, so the AIR
        // would otherwise admit traces the executor's forward byte walk never produces.
        //
        // What these do NOT give on their own is the gap over the integers: getting
        // from "gap of 8 in F" to "gap of 8 in Z" needs both limbs canonical, and COPY
        // range-checks none of `SRC_0/SRC_1/DST_0/DST_1` -- the spec lists those as
        // assumptions. Canonicality comes from the far end of the `Memory` bus instead:
        // PAGE builds `address_lo` as `page_base_lo + OFFSET` from a preprocessed
        // offset, and in continuations L2G must chain back to a GLOBAL_MEMORY genesis
        // token of the same shape, so a non-canonical limb pair has no receiver. Worth
        // knowing before adding another `Memw` producer or a non-PAGE `Memory`
        // endpoint -- either would weaken this to the field statement.
        //
        // There is no executor backstop for the limb boundary any more: `set_gap` is
        // one 64-bit `ADD`, carries and all, so a range crossing 2^32 is provable and
        // the guard that used to refuse it is gone. The PAGE argument above carries
        // this alone.
        emit_add_pair(
            b,
            20,
            &[cols::IS_SET],
            &AddOperand::dword(cols::SRC_0),
            &AddOperand::constant(DMA_MEMSET_GAP as i64),
            &AddOperand::dword(cols::DST_0),
        );

        // copy:c:single_lanes: unused lanes are zero on one-byte rows.
        for (i, &column) in cols::VALUE.iter().enumerate().skip(1) {
            b.emit_base(21 + i, single.clone() * b.main(0, column));
        }

        // `restrict_end`: `end <=> count == step or (count == 0 and first)`, i.e.
        // copy:c:borrow_implies_first, borrow_implies_count_is_zero,
        // end_implies_count_is_step_or_zc_upper / _lower and borrow_implies_end.
        //
        // The second state is the zero-length call, and `count_borrows[1]` -- the
        // borrow out of `count - step` -- is its proxy: a borrow can only happen when
        // `count < step <= 8`, and the two implications below pin that to `count == 0`
        // on a `first` row. The reverse of `end <= count == step` needs no constraint:
        // not setting `end` there produces a successor row with `count == 0` and
        // `first == 0`, which `borrow_implies_first` rejects.
        let borrow_1 = {
            use borrow_consts::*;
            let decr = |k: usize| b.main(0, cols::COUNT_DECR_0 + k);
            b.const_base(INV_2_32) * decr(2)
                + b.const_base(INV_2_16) * decr(3)
                + b.const_base(NEG_INV_2_32) * b.main(0, cols::COUNT_1)
                + b.const_base(INV_2_64) * decr(0)
                + b.const_base(INV_2_48) * decr(1)
                + b.const_base(NEG_INV_2_64) * b.main(0, cols::COUNT_0)
                + b.const_base(NEG_SEVEN_INV_2_64) * single.clone()
                + b.const_base(EIGHT_INV_2_64)
        };
        let not_borrow = one.clone() - borrow_1.clone();
        b.emit_base(29, borrow_1.clone() * (one.clone() - first));
        b.emit_base(30, borrow_1.clone() * b.main(0, cols::COUNT_0));
        b.emit_base(31, borrow_1.clone() * b.main(0, cols::COUNT_1));
        b.emit_base(
            32,
            end.clone() * not_borrow.clone() * b.main(0, cols::COUNT_1),
        );
        let step_expr = b.const_base(8) - b.const_base(7) * single.clone();
        b.emit_base(
            33,
            end.clone() * not_borrow * (b.main(0, cols::COUNT_0) - step_expr),
        );
        b.emit_base(34, borrow_1.clone() * (one.clone() - end));

        // The two COMMIT-bus gates. `mu - count_borrows[1]` is the copy multiplicity;
        // `mu_com` restricts it to `write` rows and `mu_com_wide` to wide ones.
        b.emit_base(
            35,
            b.main(0, cols::MU_COM) - (mu.clone() - borrow_1.clone()) * b.main(0, cols::IS_WRITE),
        );
        let mu_com = b.main(0, cols::MU_COM);
        b.emit_base(36, b.main(0, cols::MU_COM_WIDE) - mu_com * (one - single));

        // Nothing a row computes may wrap the address space. All three `ADD`s in this
        // chip leave the carry out of the high limb free -- that is the spec's `ADD`
        // template, and the `ADDNW` that used to carry this is gone from the spec
        // entirely. Unconstrained, a chain walks off the top of the address space and
        // resumes at 0: every row still reads and writes PAGE-backed cells at
        // consistent timestamps, so the memory argument balances and no byte value is
        // forged, but the verifier accepts a copy the executor refuses outright
        // (`src.checked_add(n)` / `dst.checked_add(n)` for the positions,
        // `src.checked_add(DMA_MEMSET_GAP) != Some(dst)` for the gap, all in
        // `execution.rs`). An acceptance set wider than the VM's is the one thing a
        // proof of execution may not have.
        //
        // The gate is `mu - count_borrows[1]`, not `mu`: the borrow is set exactly on
        // the zero-length row, where no byte moves and `src_incr`/`dst_incr` go
        // nowhere -- `copy:c:send_next_chunk` carries `mu - end` and that row is
        // `end`. Gating on plain `mu` would reject a zero-length call whose pointer
        // sits at the very top of memory, which the executor accepts. On every row
        // that moves a byte the two gates agree, including the last one, which is
        // what keeps the parity with `checked_add`: a copy may not finish on `2^64`
        // either. Padding rows carry `mu = 0` and are unaffected.
        //
        // None of the three is in the spec; see the divergence note at the top.
        let active = mu - borrow_1;
        let src_carry_hi = add_pair_carry_hi(
            b,
            &AddOperand::dword(cols::SRC_0),
            &step,
            &AddOperand::from_dword_hl(cols::SRC_INCR_0),
        );
        b.emit_base(37, active.clone() * src_carry_hi);
        let dst_carry_hi = add_pair_carry_hi(
            b,
            &AddOperand::dword(cols::DST_0),
            &step,
            &AddOperand::from_dword_hl(cols::DST_INCR_0),
        );
        b.emit_base(38, active * dst_carry_hi);
        // The gap of `copy:c:set_gap` may not wrap either, and neither constraint
        // above catches it: one-byte rows never push `src` or `dst` past `2^64`, so a
        // `memset` with `src = 2^64 - 4`, `dst = 4` walks a legal chain over an
        // illegal operand pair. `is_set` is the right gate -- it is what turns the
        // `ADD` itself on.
        let gap_carry_hi = add_pair_carry_hi(
            b,
            &AddOperand::dword(cols::SRC_0),
            &AddOperand::constant(DMA_MEMSET_GAP as i64),
            &AddOperand::dword(cols::DST_0),
        );
        b.emit_base(39, is_set * gap_carry_hi);
    }

    /// `restrict_end` multiplies `end`, `1 - count_borrows[1]` and a `count` limb,
    /// and the two conditional `ADD`s multiply their gate by a carry bit. Both are
    /// degree 3; everything else in this chip is degree 2.
    fn max_degree(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod shape_tests {
    /// Pins the committed width of the table.
    ///
    /// **The spec says 34 and this says 41, and the gap is three deliberate
    /// representation choices:**
    ///
    /// - `timestamp` is a `Word` in the spec and a `DWordWL` here (+1). The high limb
    ///   is provably zero (the CPU sends `constant(0)` in the `Ecall` tuple and
    ///   `COPY_NEXT` propagates it), so it carries no information.
    /// - `src_incr`, `dst_incr`, `count_decr` and `dst_res` are `DWordWHH`
    ///   (`Half, Half, Word`) in the spec and `DWordHL` (four `Half`s) here (+4).
    ///   The spec range-checks them with `IS_WHH`, which sends their top 48 bits to
    ///   the `IS_WH` table. This chip keeps four `IS_HALF` lookups instead, which prove
    ///   the same bound at one extra column each; moving them to `IS_WHH`, as KECCAK's
    ///   `state_ptr` already is, would drop those four columns.
    /// - `mu_com` and `mu_com_wide` (+2) gate the COMMIT bus. The spec has no such
    ///   bus: it writes committed bytes through `MEMW` in domain `2 · is_write` and
    ///   lets the verifier initialise and finalise that domain in the memory
    ///   argument. Moving to that is a change to the memory subsystem, not to this
    ///   chip.
    #[test]
    fn the_committed_shape_is_pinned() {
        assert_eq!(super::cols::NUM_COLUMNS, 41, "COPY columns (spec: 34 + 7)");

        let n = super::bus_interactions().len();
        println!(
            "COPY: {} columns, {} bus interactions, aux {} -> weight {}",
            super::cols::NUM_COLUMNS,
            n,
            n.div_ceil(2),
            super::cols::NUM_COLUMNS + 3 * n.div_ceil(2)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(functionality: Functionality, dst: u64, count: u64, width: u8) -> CopyOperation {
        CopyOperation {
            timestamp: 100,
            src: 0x1000,
            dst,
            count,
            width,
            first: false,
            end: false,
            functionality,
            value: [1, 2, 3, 4, 5, 6, 7, 8],
        }
    }

    #[test]
    fn timestamp_order_is_inverted_only_for_memset() {
        let copy = op(Functionality::Copy, 0x2000, 64, 8);
        assert_eq!(copy.read_timestamp(), 100);
        assert_eq!(copy.write_timestamp(), 101);

        let write = op(Functionality::Write, 0, 64, 8);
        assert_eq!(write.read_timestamp(), 100);
        assert_eq!(write.write_timestamp(), 101);

        // memset writes first, so a row's read observes the previous row's write and
        // the seeded bytes propagate across the range.
        let set = op(Functionality::Set, 0x2008, 64, 8);
        assert_eq!(set.write_timestamp(), 100);
        assert_eq!(set.read_timestamp(), 101);
    }

    #[test]
    fn dst_res_advances_only_for_write() {
        assert_eq!(op(Functionality::Copy, 0x2000, 64, 8).dst_res(), 0x2000);
        assert_eq!(op(Functionality::Set, 0x2008, 64, 8).dst_res(), 0x2008);
        assert_eq!(op(Functionality::Write, 40, 64, 8).dst_res(), 104);
    }

    #[test]
    fn gate_columns_follow_the_functionality() {
        let rows = [
            op(Functionality::Copy, 0x2000, 64, 8),
            op(Functionality::Write, 0, 64, 8),
            op(Functionality::Set, 0x2008, 64, 8),
        ];
        let trace = generate_copy_trace(&rows);
        let table = &trace.main_table;
        let get = |row: usize, column: usize| *table.get(row, column);

        // The RAM write rides `mu - count_borrows[1] - mu_com` and has no column, so
        // `mu_com` is what says which domain a row writes to.
        assert_eq!(get(0, cols::MU_COM), FE::zero());
        assert_eq!(get(1, cols::MU_COM), FE::one());
        assert_eq!(get(1, cols::MU_COM_WIDE), FE::one());
        assert_eq!(get(2, cols::MU_COM), FE::zero());
        assert_eq!(get(2, cols::IS_SET), FE::one());
    }

    #[test]
    fn a_one_byte_write_row_closes_the_wide_lanes() {
        // Otherwise it would send seven spurious `(index, 0)` pairs on the COMMIT bus
        // and corrupt the public-output fingerprint.
        let rows = [op(Functionality::Write, 40, 3, 1)];
        let trace = generate_copy_trace(&rows);
        assert_eq!(*trace.main_table.get(0, cols::MU_COM), FE::one());
        assert_eq!(*trace.main_table.get(0, cols::MU_COM_WIDE), FE::zero());
    }

    #[test]
    fn the_schedule_is_wide_until_the_tail_whatever_the_alignment() {
        use super::super::trace_builder::copy_row_width_for_test as w;
        // Alignment no longer enters into it: matched residues, mismatched residues
        // and both-aligned all take eight bytes while eight remain.
        for (src, dst) in [(0x1005u64, 0x2005u64), (0x1002, 0x2005), (0x1000, 0x2000)] {
            assert_eq!(w(src, dst, 0, 24, false), 8, "src {src:#x} dst {dst:#x}");
            assert_eq!(w(src, dst, 8, 16, false), 8, "src {src:#x} dst {dst:#x}");
        }
        // A short remainder is one byte a row, which is the only case that narrows.
        assert_eq!(w(0x1000, 0x2000, 16, 5, false), 1);
        assert_eq!(w(0x1005, 0x2005, 0, 7, false), 1);
        // `write` is unchanged.
        assert_eq!(w(0x1002, 0x2005, 0, 24, true), 8);
    }
}
