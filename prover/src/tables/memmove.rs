//! MEMMOVE table — one streaming copy primitive for `memcpy`/`memmove`, `memset`
//! and the byte loop of `commit`.
//!
//! Replaces the separate DMA and DMA_SET tables and takes over COMMIT's looping.
//! A row copies `1` or `8` bytes from `src` to `dst` and chains through
//! [`BusId::MemmoveNext`] until a terminal row where `count == 0`.
//!
//! ## The three functionalities
//!
//! One-hot over `is_set` and `is_commit`; `is_cpy = mu - is_set - is_commit` is
//! linear and costs no column. Neither the memory domain nor the timestamp order
//! is chosen by the caller: both are *derived* from the selector, and the selector
//! is pinned to the ecall the row receives.
//!
//! | op | domains | timestamp order | arguments |
//! |---|---|---|---|
//! | memcpy / memmove | RAM → RAM | read `T+1`, write `T+2` | `x10`, `x11`, `x12` |
//! | memset | RAM → RAM | **write `T+1`, read `T+2`** | `x10`, `x11`, `x12` |
//! | commit | RAM → COMMIT | read `T+1`, write `T+2` | the COMMIT chip's defer bus |
//!
//! ## Timestamp order
//!
//! ```text
//! read_ts  = T + 1 + is_set
//! write_ts = T + 2 - is_set
//! ```
//!
//! Both are linear in a bit column. With the normal order every read of a call
//! happens at one timestamp and every write at a later one, so a whole chunk is a
//! snapshot — that is what gives `memmove` its overlap semantics for free. With the
//! order inverted, a row's read observes the *previous* row's write, so a self-copy
//! propagates its first bytes across the range: `memset`. `old_ts < ts` holds strictly
//! either way, so the memory argument is undisturbed.
//!
//! `memset` needs no special handling here at all. Its stub seeds the first eight
//! bytes with an ordinary store and calls with `(dst = seed_end, src = seed_start,
//! count = n - 8)`, so the chip sees a plain overlapping memmove.
//!
//! ## Width is chosen per row
//!
//! `tail` is free except that an eight-byte row is illegal when fewer than eight
//! bytes remain (`(1 - tail) * lt8 = 0`, with `lt8` pinned by the ALU). A schedule can
//! therefore walk one-byte rows until `dst` is eight-aligned and take eight-byte rows
//! through the body, which keeps those rows in MEMW_A rather than MEMW.
//!
//! ## Columns (38)
//!
//! - `timestamp` DWordWL (2), `src` DWordWL (2), `src_incr` DWordHL (4)
//! - `dst` DWordWL (2) — for `commit` this is the COMMIT-domain address, i.e. the
//!   running global byte index — `dst_incr` DWordHL (4)
//! - `count` DWordWL (2), `count_decr` DWordHL (4)
//! - `first`, `end`, `tail`, `value[8]`, `mu`
//! - `is_set`, `is_commit` — the decoded functionality
//! - `lt8` — `count < 8`, pinned by the ALU
//! - `f_ncommit = first * (1 - is_commit)`, `mu_ram = (mu - end) * (1 - is_commit)`,
//!   `mu_com = (mu - end) * is_commit` — multiplicities are strictly linear in this
//!   framework, so each op-specific gate needs a column and a degree-2 constraint.
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use crate::constraints::templates::{
    AddLinearTerm, AddOperand, emit_add_pair, emit_add_pair_no_overflow, emit_is_bit,
};

use executor::vm::instruction::execution::{
    DMA_MEMCPY_MAX_BYTES as EXECUTOR_MAX_BYTES, DMA_MEMCPY_SYSCALL_NUMBER,
    DMA_MEMSET_SYSCALL_NUMBER,
};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};

const MEMCPY_LO32: u64 = DMA_MEMCPY_SYSCALL_NUMBER & 0xFFFF_FFFF;
const MEMCPY_HI32: u64 = DMA_MEMCPY_SYSCALL_NUMBER >> 32;
const MEMSET_LO32: u64 = DMA_MEMSET_SYSCALL_NUMBER & 0xFFFF_FFFF;
const MEMSET_HI32: u64 = DMA_MEMSET_SYSCALL_NUMBER >> 32;

/// Maximum bytes one ecall may move, taken from the executor so the bound the AIR
/// proves cannot drift from the bound execution enforces.
pub const MEMMOVE_MAX_BYTES: u64 = EXECUTOR_MAX_BYTES;

pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;

    pub const SRC_0: usize = 2;
    pub const SRC_1: usize = 3;

    pub const SRC_INCR_0: usize = 4;

    pub const DST_0: usize = 8;
    pub const DST_1: usize = 9;

    pub const DST_INCR_0: usize = 10;

    pub const COUNT_0: usize = 14;
    pub const COUNT_1: usize = 15;

    pub const COUNT_DECR_0: usize = 16;

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

    /// Decoded functionality. `is_cpy = mu - is_set - is_commit` is implied.
    pub const IS_SET: usize = 32;
    pub const IS_COMMIT: usize = 33;
    /// `count < 8`, pinned by the ALU; blocks an eight-byte row on a short count.
    pub const LT8: usize = 34;
    /// `first * (1 - is_commit)` — the ecall receive and the register reads.
    pub const F_NCOMMIT: usize = 35;
    /// `(mu - end) * (1 - is_commit)` — the RAM write.
    pub const MU_RAM: usize = 36;
    /// `(mu - end) * is_commit` — the COMMIT-domain write.
    pub const MU_COM: usize = 37;
    /// `mu_com * (1 - tail)` — lanes 1..7 of the COMMIT-domain write. Without it a
    /// one-byte commit row would send seven spurious `(index, 0)` pairs and corrupt
    /// the public-output fingerprint.
    pub const MU_COM_WIDE: usize = 38;

    pub const NUM_COLUMNS: usize = 39;
}

/// Which functionality a row is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Functionality {
    /// `memcpy` / `memmove`: RAM → RAM, snapshot order.
    Copy,
    /// `memset`: RAM → RAM, inverted order, so the fill propagates.
    Set,
    /// `commit`: RAM → COMMIT domain, snapshot order.
    Commit,
}

/// One row: `1` or `8` bytes, or the terminal row.
#[derive(Debug, Clone)]
pub struct MemmoveOperation {
    pub timestamp: u64,
    pub src: u64,
    /// For `Commit` this is the COMMIT-domain address (the global byte index).
    pub dst: u64,
    /// Remaining byte count including this row's bytes; `0` on the terminal row.
    pub count: u64,
    pub width: u8,
    pub first: bool,
    pub end: bool,
    pub functionality: Functionality,
    /// The bytes moved, zero-padded past `width`.
    pub value: [u8; 8],
}

impl MemmoveOperation {
    /// `read_ts = T + 1 + is_set`, `write_ts = T + 2 - is_set`.
    pub fn read_timestamp(&self) -> u64 {
        self.timestamp + 1 + u64::from(self.functionality == Functionality::Set)
    }

    pub fn write_timestamp(&self) -> u64 {
        self.timestamp + 2 - u64::from(self.functionality == Functionality::Set)
    }
}

/// Generates the MEMMOVE trace. One row per operation, padded to the next power of
/// two (min 4). Padding rows model an inactive one-byte copy so the unconditional
/// address/count relations still hold.
pub fn generate_memmove_trace(
    ops: &[MemmoveOperation],
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

        table.set_dword_wl(row_idx, cols::SRC_0, op.src);
        table.set_dword_hl(row_idx, cols::SRC_INCR_0, op.src.wrapping_add(width));

        table.set_dword_wl(row_idx, cols::DST_0, op.dst);
        table.set_dword_hl(row_idx, cols::DST_INCR_0, op.dst.wrapping_add(width));

        table.set_dword_wl(row_idx, cols::COUNT_0, op.count);
        table.set_dword_hl(row_idx, cols::COUNT_DECR_0, op.count.wrapping_sub(width));

        table.set_bool(row_idx, cols::FIRST, op.first);
        table.set_bool(row_idx, cols::END, op.end);
        table.set_bool(row_idx, cols::TAIL, op.width == 1);
        for (column, &byte) in cols::VALUE.iter().zip(&op.value) {
            table.set_byte(row_idx, *column, byte);
        }
        table.set_fe(row_idx, cols::MU, FE::one());

        let is_set = op.functionality == Functionality::Set;
        let is_commit = op.functionality == Functionality::Commit;
        table.set_bool(row_idx, cols::IS_SET, is_set);
        table.set_bool(row_idx, cols::IS_COMMIT, is_commit);
        table.set_bool(row_idx, cols::LT8, op.count < 8);
        table.set_bool(row_idx, cols::F_NCOMMIT, op.first && !is_commit);
        table.set_bool(row_idx, cols::MU_RAM, !op.end && !is_commit);
        table.set_bool(row_idx, cols::MU_COM, !op.end && is_commit);
        table.set_bool(
            row_idx,
            cols::MU_COM_WIDE,
            !op.end && is_commit && op.width == 8,
        );
    }

    for row_idx in n..num_rows {
        table.set_fe(row_idx, cols::COUNT_0, FE::one());
        table.set_fe(row_idx, cols::SRC_INCR_0, FE::one());
        table.set_fe(row_idx, cols::DST_INCR_0, FE::one());
        table.set_fe(row_idx, cols::TAIL, FE::one());
        table.set_fe(row_idx, cols::LT8, FE::one());
    }

    trace
}

/// A MEMW register read (CO24, `is_register = 1`, width 2): `value == old ==` the
/// register's two 32-bit limbs, binding `x{reg}` to `(lo_col, hi_col)` at the ecall.
fn memw_register_read(reg_addr: u64, lo_col: usize, hi_col: usize) -> Vec<BusValue> {
    let limbs = || {
        vec![
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
        ]
    };
    let mut tuple = limbs();
    tuple.push(BusValue::constant(1)); // is_register
    tuple.push(BusValue::constant(reg_addr));
    tuple.push(BusValue::constant(0));
    tuple.extend(limbs());
    tuple.push(BusValue::Packed {
        start_column: cols::TIMESTAMP_0,
        packing: Packing::Direct,
    });
    tuple.push(BusValue::Packed {
        start_column: cols::TIMESTAMP_1,
        packing: Packing::Direct,
    });
    tuple.push(BusValue::constant(1)); // w2
    tuple.push(BusValue::constant(0));
    tuple.push(BusValue::constant(0));
    tuple
}

/// `T + offset + coefficient * is_set`, the timestamp-order customisation.
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
    cols::VALUE
        .iter()
        .map(|&column| BusValue::Packed {
            start_column: column,
            packing: Packing::Direct,
        })
        .collect()
}

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

/// The MEMMOVE bus interactions.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mu_minus_end = Multiplicity::Diff(cols::MU, cols::END);
    let mu_minus_first = Multiplicity::Diff(cols::MU, cols::FIRST);
    // first * is_commit, without a column: first - f_ncommit.
    let f_commit = Multiplicity::Diff(cols::FIRST, cols::F_NCOMMIT);
    let w8 = || {
        BusValue::linear(vec![
            LinearTerm::Constant(1),
            LinearTerm::Column {
                coefficient: -1,
                column: cols::TAIL,
            },
        ])
    };

    let mut interactions = vec![
        // 1. Receive the ECALL for the two RAM-to-RAM functionalities. The syscall
        //    number is a linear function of the selector, so the decoded functionality
        //    is pinned to the ecall the guest actually made.
        BusInteraction::receiver(
            BusId::Ecall,
            Multiplicity::Column(cols::F_NCOMMIT),
            vec![
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::TIMESTAMP_1,
                    packing: Packing::Direct,
                },
                BusValue::linear(vec![
                    LinearTerm::Constant(MEMCPY_LO32 as i64),
                    LinearTerm::Column {
                        coefficient: MEMSET_LO32 as i64 - MEMCPY_LO32 as i64,
                        column: cols::IS_SET,
                    },
                ]),
                BusValue::linear(vec![
                    LinearTerm::Constant(MEMCPY_HI32 as i64),
                    LinearTerm::Column {
                        coefficient: MEMSET_HI32 as i64 - MEMCPY_HI32 as i64,
                        column: cols::IS_SET,
                    },
                ]),
            ],
        ),
        // 2. Receive the deferred loop from the COMMIT chip.
        BusInteraction::receiver(
            BusId::CommitDefer,
            f_commit,
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
        // 3. Chain forward. The selectors ride inside the tuple, so a chain cannot
        //    change functionality half way through it.
        BusInteraction::sender(
            BusId::MemmoveNext,
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
                BusValue::Packed {
                    start_column: cols::IS_SET,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::IS_COMMIT,
                    packing: Packing::Direct,
                },
            ],
        ),
        // 4. Chain backward.
        BusInteraction::receiver(
            BusId::MemmoveNext,
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
                BusValue::Packed {
                    start_column: cols::IS_SET,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::IS_COMMIT,
                    packing: Packing::Direct,
                },
            ],
        ),
        // 5-16. Halfword range checks.
        halfword(cols::COUNT_DECR_0),
        halfword(cols::COUNT_DECR_0 + 1),
        halfword(cols::COUNT_DECR_0 + 2),
        halfword(cols::COUNT_DECR_0 + 3),
        halfword(cols::SRC_INCR_0),
        halfword(cols::SRC_INCR_0 + 1),
        halfword(cols::SRC_INCR_0 + 2),
        halfword(cols::SRC_INCR_0 + 3),
        halfword(cols::DST_INCR_0),
        halfword(cols::DST_INCR_0 + 1),
        halfword(cols::DST_INCR_0 + 2),
        halfword(cols::DST_INCR_0 + 3),
        // 17. `end == 1` iff every count_decr halfword is 0xFFFF.
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
                        column: cols::COUNT_DECR_0 + 1,
                    },
                    LinearTerm::Column {
                        coefficient: -1,
                        column: cols::COUNT_DECR_0 + 2,
                    },
                    LinearTerm::Column {
                        coefficient: -1,
                        column: cols::COUNT_DECR_0 + 3,
                    },
                ]),
                BusValue::Packed {
                    start_column: cols::END,
                    packing: Packing::Direct,
                },
            ],
        ),
        // 18-20. Register reads, only for the ecall-driven functionalities.
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::F_NCOMMIT),
            memw_register_read(20, cols::DST_0, cols::DST_1),
        ),
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::F_NCOMMIT),
            memw_register_read(22, cols::SRC_0, cols::SRC_1),
        ),
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::F_NCOMMIT),
            memw_register_read(24, cols::COUNT_0, cols::COUNT_1),
        ),
        // 21. `lt8 = (count < 8)`. Width is otherwise the prover's choice.
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
                    start_column: cols::LT8,
                    packing: Packing::Direct,
                },
                BusValue::constant(0),
            ],
        ),
        // 22. The first row of an ecall-driven call proves `count <= MEMMOVE_MAX_BYTES`.
        //     Commit is excluded: it arrives over CommitDefer and the guest does not
        //     chunk it, so its length is bounded by the COMMIT chip instead.
        BusInteraction::sender(
            BusId::Alu,
            Multiplicity::Column(cols::F_NCOMMIT),
            vec![
                BusValue::Packed {
                    start_column: cols::COUNT_0,
                    packing: Packing::DWordWL,
                },
                BusValue::constant(MEMMOVE_MAX_BYTES + 1),
                BusValue::constant(0),
                BusValue::constant(alu_op::LT as u64),
                BusValue::constant(1),
                BusValue::constant(0),
            ],
        ),
        // 23. Read the source at `T + 1 + is_set`.
        BusInteraction::sender(BusId::Memw, mu_minus_end.clone(), {
            let mut values = value_columns();
            let mut tuple = Vec::with_capacity(24);
            tuple.extend(values.iter().cloned());
            tuple.push(BusValue::constant(0)); // is_register
            tuple.push(BusValue::Packed {
                start_column: cols::SRC_0,
                packing: Packing::Direct,
            });
            tuple.push(BusValue::Packed {
                start_column: cols::SRC_1,
                packing: Packing::Direct,
            });
            tuple.append(&mut values);
            tuple.push(timestamp_with_order(1, 1));
            tuple.push(BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            });
            tuple.push(BusValue::constant(0)); // w2
            tuple.push(BusValue::constant(0)); // w4
            tuple.push(w8());
            tuple
        }),
        // 24. Write the destination at `T + 2 - is_set`, RAM domain only.
        BusInteraction::sender(BusId::Memw, Multiplicity::Column(cols::MU_RAM), {
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
            tuple.push(timestamp_with_order(2, -1));
            tuple.push(BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            });
            tuple.push(BusValue::constant(0)); // w2
            tuple.push(BusValue::constant(0)); // w4
            tuple.push(w8());
            tuple
        }),
    ];

    // 25-32. Write the destination in the COMMIT domain: one `(index, value)` pair
    // per byte moved. `dst` is the running global byte index there.
    for (k, &value_column) in cols::VALUE.iter().enumerate() {
        let lane_mult = if k == 0 {
            Multiplicity::Column(cols::MU_COM)
        } else {
            Multiplicity::Column(cols::MU_COM_WIDE)
        };
        interactions.push(BusInteraction::sender(
            BusId::Commit,
            lane_mult,
            vec![
                BusValue::linear(vec![
                    LinearTerm::Column {
                        coefficient: 1,
                        column: cols::DST_0,
                    },
                    LinearTerm::Constant(k as i64),
                ]),
                BusValue::Packed {
                    start_column: value_column,
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    interactions
}

/// The MEMMOVE constraints.
#[derive(Clone, Copy)]
pub struct MemmoveConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for MemmoveConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::FIRST, None);
        emit_is_bit(b, 1, cols::END, None);
        emit_is_bit(b, 2, cols::TAIL, None);
        emit_is_bit(b, 3, cols::MU, None);
        emit_is_bit(b, 4, cols::IS_SET, None);
        emit_is_bit(b, 5, cols::IS_COMMIT, None);
        emit_is_bit(b, 6, cols::LT8, None);
        emit_is_bit(b, 7, cols::F_NCOMMIT, None);
        emit_is_bit(b, 8, cols::MU_RAM, None);
        emit_is_bit(b, 9, cols::MU_COM, None);
        emit_is_bit(b, 10, cols::MU_COM_WIDE, None);

        let one = b.one();
        let first = b.main(0, cols::FIRST);
        let end = b.main(0, cols::END);
        let mu = b.main(0, cols::MU);
        let tail = b.main(0, cols::TAIL);
        let lt8 = b.main(0, cols::LT8);
        let is_set = b.main(0, cols::IS_SET);
        let is_commit = b.main(0, cols::IS_COMMIT);

        // An active row is implied by first or end.
        b.emit_base(
            11,
            (first.clone() + end.clone()) * (one.clone() - mu.clone()),
        );
        // The functionality is one-hot and only set on active rows.
        b.emit_base(12, is_set.clone() * is_commit.clone());
        b.emit_base(
            13,
            (is_set.clone() + is_commit.clone()) * (one.clone() - mu.clone()),
        );
        // An eight-byte row is illegal when fewer than eight bytes remain.
        b.emit_base(14, (one.clone() - tail.clone()) * lt8);

        // The three gate columns.
        b.emit_base(
            15,
            b.main(0, cols::F_NCOMMIT) - first.clone() * (one.clone() - is_commit.clone()),
        );
        b.emit_base(
            16,
            b.main(0, cols::MU_RAM)
                - (mu.clone() - end.clone()) * (one.clone() - is_commit.clone()),
        );
        b.emit_base(
            17,
            b.main(0, cols::MU_COM) - (mu.clone() - end.clone()) * is_commit,
        );
        let mu_com = b.main(0, cols::MU_COM);
        b.emit_base(
            18,
            b.main(0, cols::MU_COM_WIDE) - mu_com * (one.clone() - tail.clone()),
        );

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
            19,
            cols::MU,
            cols::END,
            &AddOperand::dword(cols::SRC_0),
            &step,
            &AddOperand::from_dword_hl(cols::SRC_INCR_0),
        );
        emit_add_pair_no_overflow(
            b,
            21,
            cols::MU,
            cols::END,
            &AddOperand::dword(cols::DST_0),
            &step,
            &AddOperand::from_dword_hl(cols::DST_INCR_0),
        );
        emit_add_pair(
            b,
            23,
            &[],
            &AddOperand::from_dword_hl(cols::COUNT_DECR_0),
            &step,
            &AddOperand::dword(cols::COUNT_0),
        );

        // Unused lanes are zero on one-byte rows.
        for (i, &column) in cols::VALUE.iter().enumerate().skip(1) {
            b.emit_base(25 + i - 1, tail.clone() * b.main(0, column));
        }
    }
}

#[cfg(test)]
mod shape_tests {
    #[test]
    fn reports_the_merged_shape() {
        let n = super::bus_interactions().len();
        println!(
            "MEMMOVE: {} columns, {} bus interactions, aux {} -> weight {}",
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

    fn op(functionality: Functionality, dst: u64, count: u64, width: u8) -> MemmoveOperation {
        MemmoveOperation {
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
        assert_eq!(copy.read_timestamp(), 101);
        assert_eq!(copy.write_timestamp(), 102);

        let commit = op(Functionality::Commit, 0, 64, 8);
        assert_eq!(commit.read_timestamp(), 101);
        assert_eq!(commit.write_timestamp(), 102);

        // memset writes first, so a row's read observes the previous row's write and
        // the seeded bytes propagate across the range.
        let set = op(Functionality::Set, 0x2008, 64, 8);
        assert_eq!(set.write_timestamp(), 101);
        assert_eq!(set.read_timestamp(), 102);
    }

    #[test]
    fn gate_columns_follow_the_functionality() {
        let rows = [
            op(Functionality::Copy, 0x2000, 64, 8),
            op(Functionality::Commit, 0, 64, 8),
            op(Functionality::Set, 0x2008, 64, 8),
        ];
        let trace = generate_memmove_trace(&rows);
        let table = &trace.main_table;
        let get = |row: usize, column: usize| *table.get(row, column);

        // Copy: RAM write on, COMMIT write off.
        assert_eq!(get(0, cols::MU_RAM), FE::one());
        assert_eq!(get(0, cols::MU_COM), FE::zero());
        // Commit: the mirror image, and the wide lanes are open on an eight-byte row.
        assert_eq!(get(1, cols::MU_RAM), FE::zero());
        assert_eq!(get(1, cols::MU_COM), FE::one());
        assert_eq!(get(1, cols::MU_COM_WIDE), FE::one());
        // Set is a RAM-to-RAM copy like memcpy; only the order differs.
        assert_eq!(get(2, cols::MU_RAM), FE::one());
        assert_eq!(get(2, cols::IS_SET), FE::one());
    }

    #[test]
    fn a_one_byte_commit_row_closes_the_wide_lanes() {
        // Otherwise it would send seven spurious `(index, 0)` pairs on the COMMIT bus
        // and corrupt the public-output fingerprint.
        let rows = [op(Functionality::Commit, 40, 3, 1)];
        let trace = generate_memmove_trace(&rows);
        assert_eq!(*trace.main_table.get(0, cols::MU_COM), FE::one());
        assert_eq!(*trace.main_table.get(0, cols::MU_COM_WIDE), FE::zero());
    }

    #[test]
    fn the_schedule_aligns_both_ends_or_neither() {
        use super::super::trace_builder::memmove_row_width_for_test as w;
        // Matched residues (both 5 mod 8): one-byte rows until alignment, then wide.
        assert_eq!(w(0x1005, 0x2005, 0, 24), 1);
        assert_eq!(w(0x1005, 0x2005, 3, 21), 8);
        // Mismatched: aligning `dst` would misalign `src`, so do not split at all.
        assert_eq!(w(0x1002, 0x2005, 0, 24), 8);
        // A short remainder always falls back to one byte a row.
        assert_eq!(w(0x1000, 0x2000, 16, 5), 1);
    }
}
