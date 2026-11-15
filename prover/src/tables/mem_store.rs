//! MEM_STORE: a CPU store checked in one row with its memory tokens, under
//! [`MemoryFormat::Fused`](super::fused_memory::MemoryFormat::Fused).
//!
//! The row receives the CPU's dispatch exactly as STORE does
//! ([`super::store::bus_interactions`]), so a dispatch is received by a STORE
//! row or a MEM_STORE row, never both, and holds MEMW_A's memory check in the
//! limbs' form ([`super::fused_memory::memory_check_interactions`]): each byte
//! below the width consumes its old token (`OLD[k]`) and produces the stored
//! byte's (`VALUE[k]`).
//!
//! ## Columns (27)
//!
//! `BASE_ADDRESS[3]` (`DWordWHH`), `TIMESTAMP_0`, `TIMESTAMP_1`, `WRITE2`,
//! `WRITE4`, `WRITE8`, `VALUE[8]` (the register's eight bytes, as the dispatch
//! carries them), `OLD[8]` (the memory's bytes before the store; past the width
//! in no interaction), `STEP_0`, `STEP_1`, `MU`.
//!
//! ## Interactions (24)
//!
//! - the memory check: the span's `IS_HALF`, the limbs' two, the 16 byte
//!   tokens, on `MU` and the width flags;
//! - STORE's `MemoryOp` receive: `(timestamp, address, VALUE as two words, 1
//!   + 4·WRITE2 + 8·WRITE4 + 16·WRITE8, 0, 0)`, on `MU`;
//! - `ARE_BYTES[VALUE[2j], VALUE[2j + 1]]` for `j` in 0..4, on `MU`: the same
//!   eight bytes STORE checks one at a time.
//!
//! ## Constraints (6, degree 2)
//!
//! STORE's ([`super::store::StoreConstraints`]): the flags and `MU` bits, the
//! flags' sum a bit, the sum ⇒ `MU`. They are MEMW_A's eight too, a store
//! row's `μ_write` being `MU` and its `μ_read` 0.

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
use crate::constraints::templates::emit_is_bit;

/// Column definitions for the MEM_STORE table.
pub mod cols {
    /// The address as `DWordWHH`: low half, mid half, high word.
    pub const BASE_ADDRESS: [usize; 3] = [0, 1, 2];
    pub const TIMESTAMP_0: usize = 3;
    pub const TIMESTAMP_1: usize = 4;
    /// The width flags, STORE's: exactly 2, 4 or 8 bytes (none: 1 byte).
    pub const WRITE2: usize = 5;
    pub const WRITE4: usize = 6;
    pub const WRITE8: usize = 7;
    /// The register's eight bytes, little-endian: the bytes stored below the
    /// width.
    pub const VALUE: [usize; 8] = [8, 9, 10, 11, 12, 13, 14, 15];
    /// The memory's bytes before the store.
    pub const OLD: [usize; 8] = [16, 17, 18, 19, 20, 21, 22, 23];
    /// `ts − old_ts − 1`, low 16 bits.
    pub const STEP_0: usize = 24;
    /// `ts − old_ts − 1`, high 16 bits.
    pub const STEP_1: usize = 25;
    /// Whether the row is a store.
    pub const MU: usize = 26;
    pub const NUM_COLUMNS: usize = 27;
}

/// The columns MEM_STORE's memory check reads.
const CHECK: CheckColumns = CheckColumns {
    base: cols::BASE_ADDRESS,
    ts: [cols::TIMESTAMP_0, cols::TIMESTAMP_1],
    flags: [cols::WRITE2, cols::WRITE4, cols::WRITE8],
    old: cols::OLD,
    new: cols::VALUE,
    step: [cols::STEP_0, cols::STEP_1],
    mu: cols::MU,
};

// =========================================================================
// The record
// =========================================================================

/// One MEM_STORE row as the walk keeps it: 40 bytes, where a split store
/// keeps a 32-byte STORE op and a 48-byte MEMW_A row. `#[repr(C)]`, its layout
/// fixed (asserted below): a chunk's rows are the records a card fill reads.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MemStoreRow {
    address: u64,
    timestamp: u64,
    /// `timestamp − old_timestamp − 1`.
    step: u32,
    /// 1, 2, 4 or 8.
    width: u8,
    /// The register's value, all eight bytes.
    value: u64,
    /// The memory's eight bytes at the address before the store.
    old: u64,
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<MemStoreRow>() == 40);
    assert!(offset_of!(MemStoreRow, address) == 0);
    assert!(offset_of!(MemStoreRow, timestamp) == 8);
    assert!(offset_of!(MemStoreRow, step) == 16);
    assert!(offset_of!(MemStoreRow, width) == 20);
    assert!(offset_of!(MemStoreRow, value) == 24);
    assert!(offset_of!(MemStoreRow, old) == 32);
};

impl MemStoreRow {
    /// The row of a `width`-byte store of the register value `value` at
    /// `address` and `timestamp`, over the memory's bytes `old` last touched at
    /// `old_timestamp`, or the reason no fused row takes it ([`fused_step`]).
    pub(crate) fn new(
        address: u64,
        timestamp: u64,
        old_timestamp: u64,
        width: u8,
        value: u64,
        old: u64,
    ) -> Result<Self, String> {
        let step = fused_step(address, width, old_timestamp, timestamp)?;
        Ok(Self {
            address,
            timestamp,
            step,
            width,
            value,
            old,
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
    pub(crate) fn value(&self) -> u64 {
        self.value
    }

    #[inline]
    pub(crate) fn old(&self) -> u64 {
        self.old
    }

    /// `timestamp − old_timestamp − 1`.
    #[inline]
    pub(crate) fn step(&self) -> u32 {
        self.step
    }

    /// The row's BITWISE lookups counted into `histogram`: the four
    /// `ARE_BYTES` pairs of the value's bytes, the span's and the two limbs'
    /// `IS_HALF`.
    #[inline]
    pub(crate) fn count_bitwise_into(&self, histogram: &mut BitwiseHistogram) {
        let bytes = self.value.to_le_bytes();
        for pair in bytes.chunks_exact(2) {
            histogram.bump(BitwiseOperation::byte_op(
                BitwiseOperationType::AreBytes,
                pair[0],
                pair[1],
            ));
        }
        count_memory_check_into(self.address, self.width, self.step, histogram);
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// The widths MEM_STORE traces needed so far in this process (`tables::gpack`).
pub(crate) static WIDTHS: WidthHint = WidthHint::new();

/// The MEM_STORE trace of `segments`' rows, one after another, then zero rows
/// to a power of two of at least four, in `form`.
pub(crate) fn generate_mem_store_trace_segments_as(
    segments: &[&[MemStoreRow]],
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let len: usize = segments.iter().map(|s| s.len()).sum();
    let num_rows = len.next_power_of_two().max(4);
    generate_main!(form, &WIDTHS, num_rows, cols::NUM_COLUMNS, |table| {
        for (row, r) in segments.iter().flat_map(|s| s.iter()).enumerate() {
            table.set_dword_whh(row, cols::BASE_ADDRESS[0], r.address);
            table.set_dword_wl(row, cols::TIMESTAMP_0, r.timestamp);
            let (w2, w4, w8) = width_flags(r.width);
            table.set_bool(row, cols::WRITE2, w2);
            table.set_bool(row, cols::WRITE4, w4);
            table.set_bool(row, cols::WRITE8, w8);
            table.set_dword_bl(row, cols::VALUE[0], r.value);
            table.set_dword_bl(row, cols::OLD[0], r.old);
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

/// MEM_STORE's 24 interactions (module docs): the memory check, STORE's
/// `MemoryOp` receive, the four `ARE_BYTES` pairs.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = memory_check_interactions(&CHECK);

    // MEMORY[timestamp, address, value, flags] → 0, STORE's tuple: flags =
    // 1 + 4·write2 + 8·write4 + 16·write8, the 1 its memory_op.
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
            BusValue::Packed {
                start_column: cols::VALUE[0],
                packing: Packing::DWordBL,
            },
            BusValue::linear(vec![
                LinearTerm::Constant(1),
                LinearTerm::Column {
                    coefficient: 4,
                    column: cols::WRITE2,
                },
                LinearTerm::Column {
                    coefficient: 8,
                    column: cols::WRITE4,
                },
                LinearTerm::Column {
                    coefficient: 16,
                    column: cols::WRITE8,
                },
            ]),
            BusValue::constant(0),
            BusValue::constant(0),
        ],
    ));

    // ARE_BYTES[VALUE[2j], VALUE[2j + 1]]: the value's eight bytes, two a
    // lookup.
    for pair in cols::VALUE.chunks_exact(2) {
        interactions.push(BusInteraction::sender(
            BusId::AreBytes,
            Multiplicity::Column(cols::MU),
            vec![column(pair[0]), column(pair[1])],
        ));
    }

    interactions
}

// =========================================================================
// Constraints
// =========================================================================

/// MEM_STORE's 6 constraints, STORE's on MEM_STORE's columns, in STORE's
/// order:
/// - idx 0–3: `IS_BIT` on `WRITE2`, `WRITE4`, `WRITE8`, `MU`;
/// - idx 4: the width flags' sum a bit;
/// - idx 5: the sum ⇒ `MU`.
#[derive(Clone, Copy)]
pub struct MemStoreConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for MemStoreConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::WRITE2, None);
        emit_is_bit(b, 1, cols::WRITE4, None);
        emit_is_bit(b, 2, cols::WRITE8, None);
        emit_is_bit(b, 3, cols::MU, None);

        let sum = b.main(0, cols::WRITE2) + b.main(0, cols::WRITE4) + b.main(0, cols::WRITE8);
        let one = b.one();
        b.emit_base(4, sum.clone() * (one - sum.clone()));
        let one = b.one();
        b.emit_base(5, sum * (one - b.main(0, cols::MU)));
    }
}
