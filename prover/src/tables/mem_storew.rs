//! MEM_STOREW: a CPU store of 1, 2 or 4 bytes inside one word (SB, SH, SW),
//! checked in one row with the word's token, under
//! [`MemoryFormat::Words`](super::fused_memory::MemoryFormat::Words).
//!
//! The row holds the word's eight old bytes, all checked, and the register's
//! low four bytes, checked; it consumes the old word's token at
//! `(ADDRESS_LO − o, ADDRESS_HI)` and produces the new word's, the old one with
//! bytes `o..o + w` replaced by the register's (`super::word_cpu`).
//!
//! ## Columns (31)
//!
//! `ADDRESS_LO`, `ADDRESS_HI`, `TIMESTAMP`, `OFF[8]`, `WRITE2`, `WRITE4`,
//! `OLD[8]` (the word before), `VALUE[4]` (the register's low bytes),
//! `VALUE_HI` (its high word), `NEW_LO`, `NEW_HI` (the word after), `STEP_0`,
//! `STEP_1`, `MU`.
//!
//! ## Interactions (11)
//!
//! - the word check at `MU`: key `(ADDRESS_LO − Σ k·OFF_k, ADDRESS_HI)`, the
//!   old word `OLD` as two halves consumed, `(NEW_LO, NEW_HI)` produced;
//! - `ARE_BYTES` on the four pairs of `OLD` and the two of `VALUE`, at `MU`;
//! - the dispatch at `MU`: `(TIMESTAMP, 0, ADDRESS_LO, ADDRESS_HI,
//!   Σ VALUE_j·2^(8j), VALUE_HI, 1 + 4·WRITE2 + 8·WRITE4, 0, 0)`.
//!
//! ## Constraints (18, degree 3)
//!
//! - idx 0–8: the one-hot;
//! - idx 9, 10: `WRITE2`, `WRITE4` bits; 11: their sum a bit; 12: it implies
//!   `MU`;
//! - idx 13, 14: the window inside the word;
//! - idx 15, 16: `NEW_LO`, `NEW_HI` the old halves with byte `o + j` replaced
//!   by `VALUE_j` for `j` below the width;
//! - idx 17: `MU` a bit.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, Multiplicity};
use stark::trace::TraceTable;

use super::bitwise::{BitwiseHistogram, BitwiseOperation};
use super::gpack::{TraceForm, WidthHint, generate_main};
use super::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};
use super::word_cpu::{
    FLAG_STORE, FLAG_W2, FLAG_W4, Lookups, WordClass, bit, dispatch, flags, limb_lookups,
    one_hot_constraints, pair_lookups, store_into, stored_half, word_class, word_key,
};
use super::word_memory::{
    WORD_BYTES, byte_pair_checks, checked_word, column, halves, step_less_one, step_limbs,
    word_check_of, word_of_bytes,
};

/// Column definitions for the MEM_STOREW table.
pub mod cols {
    /// The address: its low and its high 32-bit word.
    pub const ADDRESS_LO: usize = 0;
    pub const ADDRESS_HI: usize = 1;
    /// The dispatch's timestamp.
    pub const TIMESTAMP: usize = 2;
    /// The address's offset in its word, one-hot.
    pub const OFF: [usize; 8] = [3, 4, 5, 6, 7, 8, 9, 10];
    /// The width flags, STORE's: 2 or 4 bytes (neither: 1 byte).
    pub const WRITE2: usize = 11;
    pub const WRITE4: usize = 12;
    /// The word's eight bytes before the store, little-endian.
    pub const OLD: [usize; 8] = [13, 14, 15, 16, 17, 18, 19, 20];
    /// The register's low four bytes, little-endian.
    pub const VALUE: [usize; 4] = [21, 22, 23, 24];
    /// The register's high 32-bit word (not stored).
    pub const VALUE_HI: usize = 25;
    /// The word after the store, low and high halves.
    pub const NEW_LO: usize = 26;
    pub const NEW_HI: usize = 27;
    /// `ts − old_ts − 1`, low and high 16 bits.
    pub const STEP_0: usize = 28;
    pub const STEP_1: usize = 29;
    /// Whether the row is a store.
    pub const MU: usize = 30;
    pub const NUM_COLUMNS: usize = 31;
}

// =========================================================================
// The record
// =========================================================================

/// One MEM_STOREW row as the walk keeps it: 40 bytes, `#[repr(C)]`, its
/// layout fixed (asserted below).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StoreWRow {
    address: u64,
    timestamp: u64,
    /// `timestamp − old_timestamp − 1`.
    step: u32,
    /// 1, 2 or 4.
    width: u8,
    /// The register's value, all eight bytes.
    value: u64,
    /// The word holding the bytes stored, before the store.
    old: u64,
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<StoreWRow>() == 40);
    assert!(offset_of!(StoreWRow, address) == 0);
    assert!(offset_of!(StoreWRow, timestamp) == 8);
    assert!(offset_of!(StoreWRow, step) == 16);
    assert!(offset_of!(StoreWRow, width) == 20);
    assert!(offset_of!(StoreWRow, value) == 24);
    assert!(offset_of!(StoreWRow, old) == 32);
};

#[cfg_attr(not(test), allow(dead_code))]
impl StoreWRow {
    /// The row of a `width`-byte store of the register value `value` at
    /// `address` and `timestamp`, over the word `old` (the word holding
    /// `address`) last touched at `old_timestamp`; or the reason no MEM_STOREW
    /// row takes it.
    pub(crate) fn new(
        address: u64,
        timestamp: u64,
        old_timestamp: u64,
        width: u8,
        value: u64,
        old: u64,
    ) -> Result<Self, String> {
        if word_class(address, width)? != WordClass::Within {
            return Err(format!(
                "MEM_STOREW takes fewer than 8 bytes inside a word, not {width} at {address:#x}"
            ));
        }
        let step = step_less_one(old_timestamp, timestamp)?;
        Ok(Self {
            address,
            timestamp,
            step,
            width,
            value,
            old,
        })
    }

    /// The address's offset in its word.
    #[inline]
    fn offset(&self) -> usize {
        (self.address % WORD_BYTES) as usize
    }

    /// The word after the store.
    #[inline]
    pub(crate) fn new_word(&self) -> u64 {
        let mut bytes = self.old.to_le_bytes();
        store_into(&mut bytes, self.offset(), self.width, self.value);
        u64::from_le_bytes(bytes)
    }

    /// The row's BITWISE lookups: the old word's four pairs, the register's
    /// two low pairs, the limbs' `IS_HALF`.
    #[inline]
    pub(crate) fn bitwise_ops(&self) -> impl Iterator<Item = BitwiseOperation> {
        let old = self.old.to_le_bytes();
        let value = self.value.to_le_bytes();
        let mut ops: Vec<BitwiseOperation> = pair_lookups(&old).collect();
        ops.extend(pair_lookups(&value[..4]));
        ops.into_iter().chain(limb_lookups(self.step))
    }

    /// The same lookups in a fixed-capacity list for the walker's count.
    #[inline]
    pub(crate) fn bitwise_ops_direct(&self) -> Lookups<8> {
        let old = self.old.to_le_bytes();
        let value = self.value.to_le_bytes();
        let mut ops = Lookups::new();
        ops.extend(pair_lookups(&old));
        ops.extend(pair_lookups(&value[..4]));
        ops.extend(limb_lookups(self.step));
        ops
    }

    #[inline]
    pub(crate) fn count_bitwise_into(&self, histogram: &mut BitwiseHistogram) {
        for op in self.bitwise_ops_direct() {
            histogram.bump(op);
        }
    }

    /// [`Self::bitwise_ops`] counted into `histogram`.
    #[inline]
    pub(crate) fn count_bitwise_into_listed(&self, histogram: &mut BitwiseHistogram) {
        for op in self.bitwise_ops() {
            histogram.bump(op);
        }
    }
}

/// The record's fields, for the tests that read rows back.
#[cfg(test)]
impl StoreWRow {
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
}

// =========================================================================
// Trace generation
// =========================================================================

/// The widths MEM_STOREW traces needed so far in this process.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) static WIDTHS: WidthHint = WidthHint::new();

/// The MEM_STOREW trace of `segments`' rows, one after another, then zero rows
/// to a power of two of at least four, in `form`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn generate_mem_storew_trace_segments_as(
    segments: &[&[StoreWRow]],
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
            table.set_bool(row, cols::OFF[r.offset()], true);
            table.set_bool(row, cols::WRITE2, r.width == 2);
            table.set_bool(row, cols::WRITE4, r.width == 4);
            table.set_bytes(row, cols::OLD[0], &r.old.to_le_bytes());
            table.set_bytes(row, cols::VALUE[0], &r.value.to_le_bytes()[..4]);
            table.set_word(row, cols::VALUE_HI, halves(r.value).1);
            let (new_lo, new_hi) = halves(r.new_word());
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

/// MEM_STOREW's 11 interactions (module docs).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let address = [cols::ADDRESS_LO, cols::ADDRESS_HI];
    let (old, checks) = checked_word(&cols::OLD, Multiplicity::Column(cols::MU));
    let mut interactions = word_check_of(
        word_key(address, cols::OFF, 0),
        cols::TIMESTAMP,
        old,
        [column(cols::NEW_LO), column(cols::NEW_HI)],
        [cols::STEP_0, cols::STEP_1],
        Multiplicity::Column(cols::MU),
    );
    interactions.extend(checks);
    interactions.extend(byte_pair_checks(
        &cols::VALUE,
        Multiplicity::Column(cols::MU),
    ));
    interactions.push(dispatch(
        cols::TIMESTAMP,
        address,
        [word_of_bytes(&cols::VALUE), column(cols::VALUE_HI)],
        flags(
            FLAG_STORE,
            &[(FLAG_W2, cols::WRITE2), (FLAG_W4, cols::WRITE4)],
        ),
        [BusValue::constant(0), BusValue::constant(0)],
        cols::MU,
    ));
    interactions
}

// =========================================================================
// Constraints
// =========================================================================

/// MEM_STOREW's 18 constraints (module docs).
#[derive(Clone, Copy)]
pub struct MemStoreWConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for MemStoreWConstraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        one_hot_constraints(b, 0, cols::OFF, cols::MU);
        let w2 = b.main(0, cols::WRITE2);
        let w4 = b.main(0, cols::WRITE4);
        let root = bit(b, w2.clone());
        b.emit_base(9, root);
        let root = bit(b, w4.clone());
        b.emit_base(10, root);
        let width = w2.clone() + w4.clone();
        let root = bit(b, width.clone());
        b.emit_base(11, root);
        let root = width.clone() * (b.one() - b.main(0, cols::MU));
        b.emit_base(12, root);
        // The window: `o + w ≤ 8`.
        let root = w2 * b.main(0, cols::OFF[7]);
        b.emit_base(13, root);
        let past = b.main(0, cols::OFF[5]) + b.main(0, cols::OFF[6]) + b.main(0, cols::OFF[7]);
        b.emit_base(14, w4.clone() * past);
        // The new word: byte `o + j` written for `j` below the width (byte 0
        // always; byte 1 from two bytes; bytes 2 and 3 at four).
        let act = [None, Some(width), Some(w4.clone()), Some(w4)];
        for (idx, (half, new)) in [(15, (0, cols::NEW_LO)), (16, (1, cols::NEW_HI))] {
            let root = stored_half(b, new, half, cols::OFF, &cols::OLD, &cols::VALUE, &act);
            b.emit_base(idx, root);
        }
        let root = bit(b, b.main(0, cols::MU));
        b.emit_base(17, root);
    }
}
