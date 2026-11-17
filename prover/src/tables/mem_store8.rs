//! MEM_STORE8: a CPU store of one whole aligned word (SD), checked in one row
//! with the word's token, under
//! [`MemoryFormat::Words`](super::fused_memory::MemoryFormat::Words).
//!
//! The row receives the CPU's dispatch with STORE's tuple, its flags the
//! constant 17 (a store of eight bytes), consumes the word's token with its
//! old value and produces the next with the register's value: the new word is
//! the dispatch's `rv2`, both halves. A token exists only at an aligned
//! address (`super::word_cpu`).
//!
//! The new word is held as four 16-bit halves, each range-checked, so the
//! word the row writes is canonical: each 32-bit half below `2^32`. The CPU
//! does not keep every register word canonical (a jump's return address is
//! held by its carries alone), so a store that wrote the dispatch's words as
//! they come would let such a word into memory, and back into any register
//! by a load. The split and fused formats check a stored value's bytes for
//! the same reason.
//!
//! ## Columns (12)
//!
//! `ADDRESS_LO`, `ADDRESS_HI`, `TIMESTAMP`, `OLD_LO`, `OLD_HI`, `NEW[4]`
//! (16-bit halves, low first), `STEP_0`, `STEP_1`, `MU`.
//!
//! ## Interactions (9)
//!
//! - the word check at `MU`: the limbs' two `IS_HALF`, the token consumed at
//!   `(ADDRESS_LO, ADDRESS_HI)` with `(OLD_LO, OLD_HI)` and produced with
//!   `(NEW_0 + 2^16·NEW_1, NEW_2 + 2^16·NEW_3)`;
//! - `IS_HALF` on each `NEW` half, at `MU`;
//! - the dispatch at `MU`: `(TIMESTAMP, 0, ADDRESS_LO, ADDRESS_HI, NEW_0 +
//!   2^16·NEW_1, NEW_2 + 2^16·NEW_3, 17, 0, 0)`.
//!
//! ## Constraints (1, degree 2)
//!
//! `MU` a bit.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, Multiplicity};

use super::bitwise::BitwiseOperationType;
use super::types::BusId;
use super::word_memory::term;
use stark::trace::TraceTable;

use super::bitwise::{BitwiseHistogram, BitwiseOperation};
use super::gpack::{TraceForm, WidthHint, generate_main};
use super::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};
use super::word_cpu::{FLAG_STORE, FLAG_W8, WordClass, dispatch, limb_lookups, word_class};
use super::word_memory::{column, halves, step_less_one, step_limbs, word_check_of};
use crate::constraints::templates::emit_is_bit;

/// Column definitions for the MEM_STORE8 table.
pub mod cols {
    /// The address: its low and its high 32-bit word.
    pub const ADDRESS_LO: usize = 0;
    pub const ADDRESS_HI: usize = 1;
    /// The dispatch's timestamp.
    pub const TIMESTAMP: usize = 2;
    /// The word before the store, low and high halves.
    pub const OLD_LO: usize = 3;
    pub const OLD_HI: usize = 4;
    /// The word stored, the register's value: four 16-bit halves, low first.
    pub const NEW: [usize; 4] = [5, 6, 7, 8];
    /// `ts − old_ts − 1`, low and high 16 bits.
    pub const STEP_0: usize = 9;
    pub const STEP_1: usize = 10;
    /// Whether the row is a store.
    pub const MU: usize = 11;
    pub const NUM_COLUMNS: usize = 12;
}

// =========================================================================
// The record
// =========================================================================

/// One MEM_STORE8 row as the walk keeps it: 40 bytes, `#[repr(C)]`, its
/// layout fixed (asserted below).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Store8Row {
    address: u64,
    timestamp: u64,
    /// `timestamp − old_timestamp − 1`.
    step: u32,
    /// The word before the store.
    old: u64,
    /// The register's value, the word after it.
    new: u64,
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<Store8Row>() == 40);
    assert!(offset_of!(Store8Row, address) == 0);
    assert!(offset_of!(Store8Row, timestamp) == 8);
    assert!(offset_of!(Store8Row, step) == 16);
    assert!(offset_of!(Store8Row, old) == 24);
    assert!(offset_of!(Store8Row, new) == 32);
};

#[cfg_attr(not(test), allow(dead_code))]
impl Store8Row {
    /// The row of an SD of the register value `new` at `address` and
    /// `timestamp`, over the word `old` last touched at `old_timestamp`; or the
    /// reason no MEM_STORE8 row takes it.
    pub(crate) fn new(
        address: u64,
        timestamp: u64,
        old_timestamp: u64,
        old: u64,
        new: u64,
    ) -> Result<Self, String> {
        if word_class(address, 8)? != WordClass::Whole {
            return Err(format!(
                "MEM_STORE8 takes an aligned word, not {address:#x}"
            ));
        }
        let step = step_less_one(old_timestamp, timestamp)?;
        Ok(Self {
            address,
            timestamp,
            step,
            old,
            new,
        })
    }

    /// The row's BITWISE lookups: the limbs' `IS_HALF`, then the new word's
    /// four halves' `IS_HALF`.
    #[inline]
    pub(crate) fn bitwise_ops(&self) -> impl Iterator<Item = BitwiseOperation> {
        limb_lookups(self.step)
            .into_iter()
            .chain(quarters(self.new).into_iter().map(|half| {
                BitwiseOperation::halfword(
                    BitwiseOperationType::IsHalf,
                    half as u8,
                    (half >> 8) as u8,
                )
            }))
    }

    /// [`Self::bitwise_ops`] counted into `histogram`.
    #[inline]
    pub(crate) fn count_bitwise_into(&self, histogram: &mut BitwiseHistogram) {
        for op in self.bitwise_ops() {
            histogram.bump(op);
        }
    }
}

/// The record's fields, for the tests that read rows back.
#[cfg(test)]
impl Store8Row {
    #[inline]
    pub(crate) fn address(&self) -> u64 {
        self.address
    }

    #[inline]
    pub(crate) fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// The word's timestamp before the store.
    #[inline]
    pub(crate) fn old_timestamp(&self) -> u64 {
        self.timestamp - 1 - u64::from(self.step)
    }

    /// The word after the store.
    #[inline]
    pub(crate) fn new_word(&self) -> u64 {
        self.new
    }
}

/// A word's four 16-bit halves, low first.
#[inline]
fn quarters(word: u64) -> [u16; 4] {
    [
        word as u16,
        (word >> 16) as u16,
        (word >> 32) as u16,
        (word >> 48) as u16,
    ]
}

// =========================================================================
// Trace generation
// =========================================================================

/// The widths MEM_STORE8 traces needed so far in this process.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) static WIDTHS: WidthHint = WidthHint::new();

/// The MEM_STORE8 trace of `segments`' rows, one after another, then zero
/// rows to a power of two of at least four, in `form`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn generate_mem_store8_trace_segments_as(
    segments: &[&[Store8Row]],
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let len: usize = segments.iter().map(|s| s.len()).sum();
    let num_rows = len.next_power_of_two().max(4);
    generate_main!(form, &WIDTHS, num_rows, cols::NUM_COLUMNS, |table| {
        for (row, r) in segments.iter().flat_map(|s| s.iter()).enumerate() {
            let (address_lo, address_hi) = halves(r.address);
            table.set_word(row, cols::ADDRESS_LO, address_lo);
            table.set_word(row, cols::ADDRESS_HI, address_hi);
            table.set_u64(row, cols::TIMESTAMP, r.timestamp);
            let (old_lo, old_hi) = halves(r.old);
            table.set_word(row, cols::OLD_LO, old_lo);
            table.set_word(row, cols::OLD_HI, old_hi);
            table.set_halves(row, cols::NEW[0], &quarters(r.new));
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

/// MEM_STORE8's 9 interactions (module docs).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let [n0, n1, n2, n3] = cols::NEW;
    let new = || {
        [
            BusValue::linear(vec![term(1, n0), term(1 << 16, n1)]),
            BusValue::linear(vec![term(1, n2), term(1 << 16, n3)]),
        ]
    };
    let mut interactions = word_check_of(
        [column(cols::ADDRESS_LO), column(cols::ADDRESS_HI)],
        cols::TIMESTAMP,
        [column(cols::OLD_LO), column(cols::OLD_HI)],
        new(),
        [cols::STEP_0, cols::STEP_1],
        Multiplicity::Column(cols::MU),
    );
    interactions.extend(cols::NEW.map(|half| {
        BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![column(half)],
        )
    }));
    interactions.push(dispatch(
        cols::TIMESTAMP,
        [cols::ADDRESS_LO, cols::ADDRESS_HI],
        new(),
        BusValue::constant((FLAG_STORE + FLAG_W8) as u64),
        [BusValue::constant(0), BusValue::constant(0)],
        cols::MU,
    ));
    interactions
}

// =========================================================================
// Constraints
// =========================================================================

/// MEM_STORE8's constraint: idx 0, `MU` a bit.
#[derive(Clone, Copy)]
pub struct MemStore8Constraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for MemStore8Constraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::MU, None);
    }
}

#[cfg(test)]
impl Store8Row {
    /// A test's fault: the same row, its timestamp `cycles` later (a row the
    /// CPU never dispatched).
    pub(crate) fn later(mut self, cycles: u64) -> Self {
        self.timestamp += cycles;
        self
    }
}
