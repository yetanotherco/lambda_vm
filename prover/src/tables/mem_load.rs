//! MEM_LOAD: a CPU load checked in one row with its memory tokens, under
//! [`MemoryFormat::Fused`](super::fused_memory::MemoryFormat::Fused).
//!
//! The row receives the CPU's dispatch exactly as LOAD does
//! ([`super::load::bus_interactions`]), so a dispatch is received by a LOAD row
//! or a MEM_LOAD row, never both, and holds MEMW_A's memory check in the
//! limbs' form ([`super::fused_memory::memory_check_interactions`]). A read
//! leaves the bytes as they are, so the consumed and the produced tokens carry
//! the same bytes, the result's: below the width `RES[k]` is the byte read,
//! and a byte past the width has no token.
//!
//! ## Columns (21)
//!
//! `BASE_ADDRESS[3]` (`DWordWHH`: half, half, word, as MEMW_A), `TIMESTAMP_0`,
//! `TIMESTAMP_1`, `READ2`, `READ4`, `READ8`, `SIGNED`, `RES[8]`, `SIGN_BIT`,
//! `STEP_0`, `STEP_1` (`ts − old_ts − 1` in 16-bit limbs), `MU`.
//!
//! ## Interactions (23)
//!
//! - the memory check: the span's `IS_HALF`, the limbs' two, the 16 byte tokens
//!   (old = new = `RES[k]`), on `MU` and the width flags;
//! - LOAD's three `MSB8` sends for the sign bit;
//! - LOAD's `MemoryOp` receive: `(timestamp, address, 0, 0, 2·SIGNED + 4·READ2
//!   + 8·READ4 + 16·READ8, RES as two words)`, on `MU`.
//!
//! ## Constraints (14, degree 3)
//!
//! LOAD's ([`super::load::LoadConstraints`]): the flags and their sum bits,
//! the sum ⇒ `MU`, the seven extension constraints, `MU` a bit. They are
//! MEMW_A's eight too, a load row's `μ_read` being `MU` and its `μ_write` 0.

// No walk routes an access to the fused tables yet: until one does, only the
// tests reach the records, the fills and the lookup counts.
#![cfg_attr(not(test), allow(dead_code))]

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::bitwise::{BitwiseHistogram, BitwiseOperation, BitwiseOperationType};
use super::fused_memory::{
    CheckColumns, count_memory_check_into, fused_step, memory_check_interactions, step_limbs,
    width_flags,
};
use super::gpack::{TraceForm, WidthHint, generate_main};
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};

/// Column definitions for the MEM_LOAD table.
pub mod cols {
    /// The address as `DWordWHH`: low half, mid half, high word.
    pub const BASE_ADDRESS: [usize; 3] = [0, 1, 2];
    pub const TIMESTAMP_0: usize = 3;
    pub const TIMESTAMP_1: usize = 4;
    /// The width flags, LOAD's: exactly 2, 4 or 8 bytes (none: 1 byte).
    pub const READ2: usize = 5;
    pub const READ4: usize = 6;
    pub const READ8: usize = 7;
    /// 1: sign-extend, 0: zero-extend.
    pub const SIGNED: usize = 8;
    /// The result's eight bytes: the bytes read below the width, the extension
    /// past it.
    pub const RES: [usize; 8] = [9, 10, 11, 12, 13, 14, 15, 16];
    /// The most significant bit of the highest byte read.
    pub const SIGN_BIT: usize = 17;
    /// `ts − old_ts − 1`, low 16 bits.
    pub const STEP_0: usize = 18;
    /// `ts − old_ts − 1`, high 16 bits.
    pub const STEP_1: usize = 19;
    /// Whether the row is a load.
    pub const MU: usize = 20;
    pub const NUM_COLUMNS: usize = 21;
}

/// The columns MEM_LOAD's memory check reads: a read's old and new bytes are
/// both the result's.
const CHECK: CheckColumns = CheckColumns {
    base: cols::BASE_ADDRESS,
    ts: [cols::TIMESTAMP_0, cols::TIMESTAMP_1],
    flags: [cols::READ2, cols::READ4, cols::READ8],
    old: cols::RES,
    new: cols::RES,
    step: [cols::STEP_0, cols::STEP_1],
    mu: cols::MU,
};

// =========================================================================
// The record
// =========================================================================

/// One MEM_LOAD row as the walk keeps it: 32 bytes, where a split load keeps
/// a 32-byte LOAD op and a 48-byte MEMW_A row. `#[repr(C)]`, its layout fixed
/// (asserted below): a chunk's rows are the records a card fill reads.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MemLoadRow {
    address: u64,
    timestamp: u64,
    /// `timestamp − old_timestamp − 1`.
    step: u32,
    /// The result's eight bytes, extended.
    res: [u8; 8],
    /// 1, 2, 4 or 8.
    width: u8,
    signed: bool,
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<MemLoadRow>() == 32);
    assert!(offset_of!(MemLoadRow, address) == 0);
    assert!(offset_of!(MemLoadRow, timestamp) == 8);
    assert!(offset_of!(MemLoadRow, step) == 16);
    assert!(offset_of!(MemLoadRow, res) == 20);
    assert!(offset_of!(MemLoadRow, width) == 28);
    assert!(offset_of!(MemLoadRow, signed) == 29);
};

impl MemLoadRow {
    /// The row of a `width`-byte load at `address` and `timestamp`, its bytes
    /// last touched at `old_timestamp`, its result `res` (extended), or the
    /// reason no fused row takes it ([`fused_step`]).
    pub(crate) fn new(
        address: u64,
        timestamp: u64,
        old_timestamp: u64,
        width: u8,
        signed: bool,
        res: [u8; 8],
    ) -> Result<Self, String> {
        let step = fused_step(address, width, old_timestamp, timestamp)?;
        Ok(Self {
            address,
            timestamp,
            step,
            res,
            width,
            signed,
        })
    }

    #[inline]
    pub(crate) fn address(&self) -> u64 {
        self.address
    }

    #[inline]
    pub(crate) fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// The accessed bytes' shared old timestamp.
    #[inline]
    pub(crate) fn old_timestamp(&self) -> u64 {
        self.timestamp - 1 - u64::from(self.step)
    }

    #[inline]
    pub(crate) fn width(&self) -> u8 {
        self.width
    }

    #[inline]
    pub(crate) fn signed(&self) -> bool {
        self.signed
    }

    #[inline]
    pub(crate) fn res(&self) -> [u8; 8] {
        self.res
    }

    /// `timestamp − old_timestamp − 1`.
    #[inline]
    pub(crate) fn step(&self) -> u32 {
        self.step
    }

    /// The sign bit: the most significant bit of the highest byte read.
    #[inline]
    fn sign_bit(&self) -> bool {
        self.res[usize::from(self.width.max(1)) - 1] >> 7 == 1
    }

    /// The row's BITWISE lookups counted into `histogram`: the sign bit's
    /// `MSB8` of a load narrower than 8 bytes (LOAD's), the span's and the two
    /// limbs' `IS_HALF`.
    #[inline]
    pub(crate) fn count_bitwise_into(&self, histogram: &mut BitwiseHistogram) {
        if self.width < 8 {
            histogram.bump(BitwiseOperation::single_byte(
                BitwiseOperationType::Msb8,
                self.res[usize::from(self.width.max(1)) - 1],
            ));
        }
        count_memory_check_into(self.address, self.width, self.step, histogram);
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// The widths MEM_LOAD traces needed so far in this process (`tables::gpack`).
pub(crate) static WIDTHS: WidthHint = WidthHint::new();

/// The MEM_LOAD trace of `segments`' rows, one after another, then zero rows
/// to a power of two of at least four, in `form`.
pub(crate) fn generate_mem_load_trace_segments_as(
    segments: &[&[MemLoadRow]],
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let len: usize = segments.iter().map(|s| s.len()).sum();
    let num_rows = len.next_power_of_two().max(4);
    generate_main!(form, &WIDTHS, num_rows, cols::NUM_COLUMNS, |table| {
        for (row, r) in segments.iter().flat_map(|s| s.iter()).enumerate() {
            table.set_dword_whh(row, cols::BASE_ADDRESS[0], r.address);
            table.set_dword_wl(row, cols::TIMESTAMP_0, r.timestamp);
            let (r2, r4, r8) = width_flags(r.width);
            table.set_bool(row, cols::READ2, r2);
            table.set_bool(row, cols::READ4, r4);
            table.set_bool(row, cols::READ8, r8);
            table.set_bool(row, cols::SIGNED, r.signed);
            table.set_bytes(row, cols::RES[0], &r.res);
            table.set_bool(row, cols::SIGN_BIT, r.sign_bit());
            let (d0, d1) = step_limbs(r.step);
            table.set_half(row, cols::STEP_0, d0);
            table.set_half(row, cols::STEP_1, d1);
            table.set_fe(row, cols::MU, FE::one());
        }
    })
}

// =========================================================================
// Bus interactions
// =========================================================================

fn column(c: usize) -> BusValue {
    BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    }
}

/// MEM_LOAD's 23 interactions (module docs): the memory check, LOAD's three
/// `MSB8` sends, LOAD's `MemoryOp` receive.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = memory_check_interactions(&CHECK);

    // MSB8[RES[k]] → SIGN_BIT on the highest byte read: byte 0 at
    // `μ − read2 − read4 − read8` (exactly one byte), byte 1 at `read2`,
    // byte 3 at `read4`; an 8-byte load extends nothing.
    let read1 = Multiplicity::Linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::MU,
        },
        LinearTerm::Column {
            coefficient: -1,
            column: cols::READ2,
        },
        LinearTerm::Column {
            coefficient: -1,
            column: cols::READ4,
        },
        LinearTerm::Column {
            coefficient: -1,
            column: cols::READ8,
        },
    ]);
    for (multiplicity, byte) in [
        (read1, 0),
        (Multiplicity::Column(cols::READ2), 1),
        (Multiplicity::Column(cols::READ4), 3),
    ] {
        interactions.push(BusInteraction::sender(
            BusId::Msb8,
            multiplicity,
            vec![column(cols::RES[byte]), column(cols::SIGN_BIT)],
        ));
    }

    // MEMORY[timestamp, address, value = 0, mem_flags] → RES, LOAD's tuple:
    // mem_flags = 2·signed + 4·read2 + 8·read4 + 16·read8 (memory_op 0).
    interactions.push(BusInteraction::receiver(
        BusId::MemoryOp,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::DWordWL,
            },
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS[0],
                packing: Packing::DWordWHH,
            },
            BusValue::constant(0),
            BusValue::constant(0),
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 2,
                    column: cols::SIGNED,
                },
                LinearTerm::Column {
                    coefficient: 4,
                    column: cols::READ2,
                },
                LinearTerm::Column {
                    coefficient: 8,
                    column: cols::READ4,
                },
                LinearTerm::Column {
                    coefficient: 16,
                    column: cols::READ8,
                },
            ]),
            BusValue::Packed {
                start_column: cols::RES[0],
                packing: Packing::DWordBL,
            },
        ],
    ));

    interactions
}

// =========================================================================
// Constraints
// =========================================================================

/// MEM_LOAD's 14 constraints, LOAD's on MEM_LOAD's columns, in LOAD's order:
/// - idx 0–3: `IS_BIT` on `SIGNED`, `READ2`, `READ4`, `READ8`;
/// - idx 4: the width flags' sum a bit;
/// - idx 5: the sum ⇒ `MU`;
/// - idx 6–9: `(1 − read8)·(RES[i] − signed·sign_bit·255)`, `i` in 4..8;
/// - idx 10, 11: `(1 − read4 − read8)·(RES[i] − …)`, `i` in 2..4;
/// - idx 12: `(1 − read2 − read4 − read8)·(RES[1] − …)`;
/// - idx 13: `IS_BIT` on `MU`. A free `MU` on a byte load lets a `−1 / +1`
///   pair of rows with one result byte and opposite sign bits cancel on every
///   bus but `MSB8`, where it cancels another row's forged sign bit.
#[derive(Clone, Copy)]
pub struct MemLoadConstraints;

impl MemLoadConstraints {
    fn is_bit<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        col: usize,
    ) -> B::Expr {
        let x = b.main(0, col);
        let one = b.one();
        x.clone() * (one - x)
    }

    /// `signed · sign_bit · 255`: a byte past the width.
    fn extended<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(b: &B) -> B::Expr {
        b.main(0, cols::SIGNED) * b.main(0, cols::SIGN_BIT) * b.const_base(255)
    }
}

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for MemLoadConstraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        for (idx, col) in [cols::SIGNED, cols::READ2, cols::READ4, cols::READ8]
            .into_iter()
            .enumerate()
        {
            let root = Self::is_bit(b, col);
            b.emit_base(idx, root);
        }

        let sum = b.main(0, cols::READ2) + b.main(0, cols::READ4) + b.main(0, cols::READ8);
        let one = b.one();
        b.emit_base(4, sum.clone() * (one - sum.clone()));
        let one = b.one();
        b.emit_base(5, sum * (one - b.main(0, cols::MU)));

        for (offset, i) in (4..8).enumerate() {
            let one = b.one();
            let past = one - b.main(0, cols::READ8);
            let root = past * (b.main(0, cols::RES[i]) - Self::extended(b));
            b.emit_base(6 + offset, root);
        }
        for (offset, i) in (2..4).enumerate() {
            let one = b.one();
            let past = one - b.main(0, cols::READ4) - b.main(0, cols::READ8);
            let root = past * (b.main(0, cols::RES[i]) - Self::extended(b));
            b.emit_base(10 + offset, root);
        }
        let one = b.one();
        let past = one - b.main(0, cols::READ2) - b.main(0, cols::READ4) - b.main(0, cols::READ8);
        let root = past * (b.main(0, cols::RES[1]) - Self::extended(b));
        b.emit_base(12, root);

        let root = Self::is_bit(b, cols::MU);
        b.emit_base(13, root);
    }
}
