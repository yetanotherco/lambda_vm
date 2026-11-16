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
//! ## Columns (10)
//!
//! `ADDRESS_LO`, `ADDRESS_HI`, `TIMESTAMP`, `OLD_LO`, `OLD_HI`, `NEW_LO`,
//! `NEW_HI`, `STEP_0`, `STEP_1`, `MU`.
//!
//! ## Interactions (5)
//!
//! - the word check at `MU`: the limbs' two `IS_HALF`, the token consumed at
//!   `(ADDRESS_LO, ADDRESS_HI)` with `(OLD_LO, OLD_HI)` and produced with
//!   `(NEW_LO, NEW_HI)`;
//! - the dispatch at `MU`: `(TIMESTAMP, 0, ADDRESS_LO, ADDRESS_HI, NEW_LO,
//!   NEW_HI, 17, 0, 0)`.
//!
//! ## Constraints (1, degree 2)
//!
//! `MU` a bit.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, Multiplicity};
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
    /// The word stored: the register's value, low and high halves.
    pub const NEW_LO: usize = 5;
    pub const NEW_HI: usize = 6;
    /// `ts − old_ts − 1`, low and high 16 bits.
    pub const STEP_0: usize = 7;
    pub const STEP_1: usize = 8;
    /// Whether the row is a store.
    pub const MU: usize = 9;
    pub const NUM_COLUMNS: usize = 10;
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

    /// The row's BITWISE lookups: the limbs' `IS_HALF`.
    #[inline]
    pub(crate) fn bitwise_ops(&self) -> impl Iterator<Item = BitwiseOperation> {
        limb_lookups(self.step).into_iter()
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
            let (new_lo, new_hi) = halves(r.new);
            table.set_word(row, cols::NEW_LO, new_lo);
            table.set_word(row, cols::NEW_HI, new_hi);
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

/// MEM_STORE8's 5 interactions (module docs).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let new = || [column(cols::NEW_LO), column(cols::NEW_HI)];
    let mut interactions = word_check_of(
        [column(cols::ADDRESS_LO), column(cols::ADDRESS_HI)],
        cols::TIMESTAMP,
        [column(cols::OLD_LO), column(cols::OLD_HI)],
        new(),
        [cols::STEP_0, cols::STEP_1],
        Multiplicity::Column(cols::MU),
    );
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
