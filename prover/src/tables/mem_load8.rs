//! MEM_LOAD8: a CPU load of one whole aligned word (LD), checked in one row
//! with the word's token, under
//! [`MemoryFormat::Words`](super::fused_memory::MemoryFormat::Words).
//!
//! The row receives the CPU's dispatch with LOAD's tuple, its flags the
//! constant 16 (LD's: eight bytes, zero-extended), and passes the word's token
//! through at the dispatched address: the word read is the result. A token
//! exists only at an aligned address, so a row at any other address names a
//! word nothing minted (`super::word_cpu`).
//!
//! ## Columns (8)
//!
//! `ADDRESS_LO`, `ADDRESS_HI`, `TIMESTAMP`, `LO`, `HI` (the word),
//! `STEP_0`, `STEP_1` (`ts − old_ts − 1` in 16-bit limbs), `MU`.
//!
//! ## Interactions (5)
//!
//! - the word check at `MU`: the limbs' two `IS_HALF`, the token consumed and
//!   produced at `(ADDRESS_LO, ADDRESS_HI)`, the word `(LO, HI)` both times;
//! - the dispatch at `MU`: `(TIMESTAMP, 0, ADDRESS_LO, ADDRESS_HI, 0, 0, 16,
//!   LO, HI)`.
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
use super::word_cpu::{FLAG_W8, WordClass, dispatch, limb_lookups, word_class};
use super::word_memory::{column, halves, step_less_one, step_limbs, word_check_of};
use crate::constraints::templates::emit_is_bit;

/// Column definitions for the MEM_LOAD8 table.
pub mod cols {
    /// The address: its low and its high 32-bit word.
    pub const ADDRESS_LO: usize = 0;
    pub const ADDRESS_HI: usize = 1;
    /// The dispatch's timestamp.
    pub const TIMESTAMP: usize = 2;
    /// The word read, low and high halves: the result.
    pub const LO: usize = 3;
    pub const HI: usize = 4;
    /// `ts − old_ts − 1`, low and high 16 bits.
    pub const STEP_0: usize = 5;
    pub const STEP_1: usize = 6;
    /// Whether the row is a load.
    pub const MU: usize = 7;
    pub const NUM_COLUMNS: usize = 8;
}

// =========================================================================
// The record
// =========================================================================

/// One MEM_LOAD8 row as the walk keeps it: 32 bytes, `#[repr(C)]`, its layout
/// fixed (asserted below).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Load8Row {
    address: u64,
    timestamp: u64,
    /// `timestamp − old_timestamp − 1`.
    step: u32,
    /// The word read.
    word: u64,
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<Load8Row>() == 32);
    assert!(offset_of!(Load8Row, address) == 0);
    assert!(offset_of!(Load8Row, timestamp) == 8);
    assert!(offset_of!(Load8Row, step) == 16);
    assert!(offset_of!(Load8Row, word) == 24);
};

#[cfg_attr(not(test), allow(dead_code))]
impl Load8Row {
    /// The row of an LD at `address` and `timestamp` of the word `word`, last
    /// touched at `old_timestamp`; or the reason no MEM_LOAD8 row takes it (an
    /// unaligned address, a step the limbs cannot hold).
    pub(crate) fn new(
        address: u64,
        timestamp: u64,
        old_timestamp: u64,
        word: u64,
    ) -> Result<Self, String> {
        if word_class(address, 8)? != WordClass::Whole {
            return Err(format!("MEM_LOAD8 takes an aligned word, not {address:#x}"));
        }
        let step = step_less_one(old_timestamp, timestamp)?;
        Ok(Self {
            address,
            timestamp,
            step,
            word,
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
impl Load8Row {
    #[inline]
    pub(crate) fn address(&self) -> u64 {
        self.address
    }

    #[inline]
    pub(crate) fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// The word's timestamp before the load.
    #[inline]
    pub(crate) fn old_timestamp(&self) -> u64 {
        self.timestamp - 1 - u64::from(self.step)
    }

    #[inline]
    pub(crate) fn word(&self) -> u64 {
        self.word
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// The widths MEM_LOAD8 traces needed so far in this process.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) static WIDTHS: WidthHint = WidthHint::new();

/// The MEM_LOAD8 trace of `segments`' rows, one after another, then zero rows
/// to a power of two of at least four, in `form`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn generate_mem_load8_trace_segments_as(
    segments: &[&[Load8Row]],
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
            let (lo, hi) = halves(r.word);
            table.set_word(row, cols::LO, lo);
            table.set_word(row, cols::HI, hi);
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

/// MEM_LOAD8's 5 interactions (module docs).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let address = [cols::ADDRESS_LO, cols::ADDRESS_HI];
    let word = || [column(cols::LO), column(cols::HI)];
    let mut interactions = word_check_of(
        [column(cols::ADDRESS_LO), column(cols::ADDRESS_HI)],
        cols::TIMESTAMP,
        word(),
        word(),
        [cols::STEP_0, cols::STEP_1],
        Multiplicity::Column(cols::MU),
    );
    interactions.push(dispatch(
        cols::TIMESTAMP,
        address,
        [BusValue::constant(0), BusValue::constant(0)],
        BusValue::constant(FLAG_W8 as u64),
        word(),
        cols::MU,
    ));
    interactions
}

// =========================================================================
// Constraints
// =========================================================================

/// MEM_LOAD8's constraint: idx 0, `MU` a bit.
#[derive(Clone, Copy)]
pub struct MemLoad8Constraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for MemLoad8Constraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::MU, None);
    }
}

#[cfg(test)]
impl Load8Row {
    /// A test's fault: the same row, its timestamp `cycles` later (a row the
    /// CPU never dispatched).
    pub(crate) fn later(mut self, cycles: u64) -> Self {
        self.timestamp += cycles;
        self
    }
}
