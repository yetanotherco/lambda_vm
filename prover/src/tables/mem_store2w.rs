//! MEM_STORE2W: a CPU store of 2, 4 or 8 bytes across two words (SH, SW,
//! SD), checked in one row with both words' tokens, under
//! [`MemoryFormat::Words`](super::fused_memory::MemoryFormat::Words).
//!
//! The row holds both words' sixteen old bytes and the register's eight bytes,
//! all checked; it consumes each old word's token and produces the new word's,
//! the first at `(ADDRESS_LO − o, ADDRESS_HI)` and the second at
//! `(ADDRESS_LO − o + 8 − 2^32·CARRY, ADDRESS_HI + CARRY)`, each with its own
//! old timestamp. `CARRY` is 1 exactly when the first word is the last of its
//! low range, `2^32 − 8` ([`super::word_cpu::second_word_key`]). The new words
//! are the old ones with bytes `o..o + w` of the sixteen replaced by the
//! register's (`super::word_cpu`).
//!
//! ## Columns (48)
//!
//! `ADDRESS_LO`, `ADDRESS_HI`, `TIMESTAMP`, `OFF[8]`, `WRITE2`, `WRITE4`,
//! `WRITE8`, `OLD[16]`, `VALUE[8]`, `NEW1_LO`, `NEW1_HI`, `NEW2_LO`,
//! `NEW2_HI`, `STEP_0`, `STEP_1`, `STEP2_0`, `STEP2_1`, `MU`, `CARRY`.
//!
//! ## Interactions (21)
//!
//! - the first word's check (old `OLD[0..8]`, new `(NEW1_LO, NEW1_HI)`) and
//!   the second's (`OLD[8..16]`, `(NEW2_LO, NEW2_HI)`), at `MU`;
//! - `ARE_BYTES` on the eight pairs of `OLD` and the four of `VALUE`, at `MU`;
//! - the dispatch at `MU`: `(TIMESTAMP, 0, ADDRESS_LO, ADDRESS_HI, VALUE as
//!   two words, 1 + 4·WRITE2 + 8·WRITE4 + 16·WRITE8, 0, 0)`.
//!
//! ## Constraints (23, degree 3)
//!
//! - idx 0–8: the one-hot;
//! - idx 9–11: `WRITE2`, `WRITE4`, `WRITE8` bits; 12: their sum `MU`;
//! - idx 13–15: the access crosses;
//! - idx 16–19: the four new halves, the old ones with byte `o + j` replaced
//!   by `VALUE_j` for `j` below the width;
//! - idx 20: `MU` a bit;
//! - idx 21: `CARRY` a bit; 22: `CARRY·(ADDRESS_LO − Σ k·OFF_k − (2^32 − 8))
//!   = 0`, a carry only from a low range's last word
//!   ([`super::word_cpu::carry_constraints`]).

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, Multiplicity};
use stark::trace::TraceTable;

use super::bitwise::{BitwiseHistogram, BitwiseOperation};
use super::gpack::{TraceForm, WidthHint, generate_main};
use super::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};
use super::word_cpu::{
    FLAG_STORE, FLAG_W2, FLAG_W4, FLAG_W8, Lookups, WordClass, bit, carries, carry_constraints,
    dispatch, flags, limb_lookups, one_hot_constraints, pair_lookups, second_word_key, store_into,
    stored_half, sum, word_class, word_key,
};
use super::word_memory::{
    WORD_BYTES, byte_pair_checks, checked_word, column, halves, step_less_one, step_limbs,
    word_check_of, word_of_bytes,
};

/// Column definitions for the MEM_STORE2W table.
pub mod cols {
    /// The address: its low and its high 32-bit word.
    pub const ADDRESS_LO: usize = 0;
    pub const ADDRESS_HI: usize = 1;
    /// The dispatch's timestamp.
    pub const TIMESTAMP: usize = 2;
    /// The address's offset in its first word, one-hot.
    pub const OFF: [usize; 8] = [3, 4, 5, 6, 7, 8, 9, 10];
    /// The width flags, STORE's: 2, 4 or 8 bytes.
    pub const WRITE2: usize = 11;
    pub const WRITE4: usize = 12;
    pub const WRITE8: usize = 13;
    /// The two words' sixteen bytes before the store, the first word's first.
    pub const OLD: [usize; 16] = [
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    ];
    /// The register's eight bytes, little-endian.
    pub const VALUE: [usize; 8] = [30, 31, 32, 33, 34, 35, 36, 37];
    /// The first word after the store, low and high halves.
    pub const NEW1_LO: usize = 38;
    pub const NEW1_HI: usize = 39;
    /// The second word after the store.
    pub const NEW2_LO: usize = 40;
    pub const NEW2_HI: usize = 41;
    /// The first word's `ts − old_ts − 1`, low and high 16 bits.
    pub const STEP_0: usize = 42;
    pub const STEP_1: usize = 43;
    /// The second word's.
    pub const STEP2_0: usize = 44;
    pub const STEP2_1: usize = 45;
    /// Whether the row is a store.
    pub const MU: usize = 46;
    /// 1 when the first word is the last of its low range: the second word's
    /// key carries into the high word.
    pub const CARRY: usize = 47;
    pub const NUM_COLUMNS: usize = 48;
}

// =========================================================================
// The record
// =========================================================================

/// One MEM_STORE2W row as the walk keeps it: 56 bytes, `#[repr(C)]`, its
/// layout fixed (asserted below).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Store2WRow {
    address: u64,
    timestamp: u64,
    /// Each word's `timestamp − old_timestamp − 1`.
    steps: [u32; 2],
    /// The register's value, all eight bytes.
    value: u64,
    /// The two words before the store, the one holding `address` first.
    old: [u64; 2],
    /// 2, 4 or 8.
    width: u8,
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<Store2WRow>() == 56);
    assert!(offset_of!(Store2WRow, address) == 0);
    assert!(offset_of!(Store2WRow, timestamp) == 8);
    assert!(offset_of!(Store2WRow, steps) == 16);
    assert!(offset_of!(Store2WRow, value) == 24);
    assert!(offset_of!(Store2WRow, old) == 32);
    assert!(offset_of!(Store2WRow, width) == 48);
};

#[cfg_attr(not(test), allow(dead_code))]
impl Store2WRow {
    /// The row of a `width`-byte store of the register value `value` at
    /// `address` and `timestamp` across the words `old` (the one holding
    /// `address`, then the next) last touched at `old_timestamps`; or the reason
    /// no MEM_STORE2W row takes it.
    pub(crate) fn new(
        address: u64,
        timestamp: u64,
        old_timestamps: [u64; 2],
        width: u8,
        value: u64,
        old: [u64; 2],
    ) -> Result<Self, String> {
        if word_class(address, width)? != WordClass::Crossing {
            return Err(format!(
                "MEM_STORE2W takes an access across two words, not {width} at {address:#x}"
            ));
        }
        let steps = [
            step_less_one(old_timestamps[0], timestamp)?,
            step_less_one(old_timestamps[1], timestamp)?,
        ];
        Ok(Self {
            address,
            timestamp,
            steps,
            value,
            old,
            width,
        })
    }

    /// The address's offset in its first word.
    #[inline]
    fn offset(&self) -> usize {
        (self.address % WORD_BYTES) as usize
    }

    /// The two old words' sixteen bytes.
    #[inline]
    fn old_bytes(&self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&self.old[0].to_le_bytes());
        bytes[8..].copy_from_slice(&self.old[1].to_le_bytes());
        bytes
    }

    /// The two words after the store.
    #[inline]
    pub(crate) fn new_words(&self) -> [u64; 2] {
        let mut bytes = self.old_bytes();
        store_into(&mut bytes, self.offset(), self.width, self.value);
        let word = |w: usize| u64::from_le_bytes(bytes[8 * w..8 * w + 8].try_into().unwrap());
        [word(0), word(1)]
    }

    /// The row's BITWISE lookups: the old words' eight pairs, the register's
    /// four, both words' limbs' `IS_HALF`.
    #[inline]
    pub(crate) fn bitwise_ops(&self) -> impl Iterator<Item = BitwiseOperation> {
        let old = self.old_bytes();
        let value = self.value.to_le_bytes();
        let mut ops: Vec<BitwiseOperation> = pair_lookups(&old).collect();
        ops.extend(pair_lookups(&value));
        ops.extend(limb_lookups(self.steps[0]));
        ops.extend(limb_lookups(self.steps[1]));
        ops.into_iter()
    }

    /// The same lookups in a fixed-capacity list for the walker's count.
    #[inline]
    pub(crate) fn bitwise_ops_direct(&self) -> Lookups<16> {
        let old = self.old_bytes();
        let value = self.value.to_le_bytes();
        let mut ops = Lookups::new();
        ops.extend(pair_lookups(&old));
        ops.extend(pair_lookups(&value));
        ops.extend(limb_lookups(self.steps[0]));
        ops.extend(limb_lookups(self.steps[1]));
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
impl Store2WRow {
    #[inline]
    pub(crate) fn address(&self) -> u64 {
        self.address
    }

    #[inline]
    pub(crate) fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Each word's timestamp before the store.
    #[inline]
    pub(crate) fn old_timestamps(&self) -> [u64; 2] {
        self.steps.map(|s| self.timestamp - 1 - u64::from(s))
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// The widths MEM_STORE2W traces needed so far in this process.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) static WIDTHS: WidthHint = WidthHint::new();

/// The MEM_STORE2W trace of `segments`' rows, one after another, then zero
/// rows to a power of two of at least four, in `form`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn generate_mem_store2w_trace_segments_as(
    segments: &[&[Store2WRow]],
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
            table.set_bool(row, cols::WRITE8, r.width == 8);
            table.set_bytes(row, cols::OLD[0], &r.old_bytes());
            table.set_bytes(row, cols::VALUE[0], &r.value.to_le_bytes());
            let [first, second] = r.new_words();
            for (word, (lo, hi)) in [first, second].into_iter().zip([
                (cols::NEW1_LO, cols::NEW1_HI),
                (cols::NEW2_LO, cols::NEW2_HI),
            ]) {
                let (w_lo, w_hi) = halves(word);
                table.set_word(row, lo, w_lo);
                table.set_word(row, hi, w_hi);
            }
            for (step, (c0, c1)) in r
                .steps
                .into_iter()
                .zip([(cols::STEP_0, cols::STEP_1), (cols::STEP2_0, cols::STEP2_1)])
            {
                let (d0, d1) = step_limbs(step);
                table.set_half(row, c0, d0);
                table.set_half(row, c1, d1);
            }
            table.set_fe(row, cols::MU, FE::one());
            table.set_bool(row, cols::CARRY, carries(r.address));
        }
    })
}

// =========================================================================
// Bus interactions
// =========================================================================

/// MEM_STORE2W's 21 interactions (module docs).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let address = [cols::ADDRESS_LO, cols::ADDRESS_HI];
    let words = [&cols::OLD[..8], &cols::OLD[8..]]
        .map(|bytes| checked_word(bytes, Multiplicity::Column(cols::MU)));
    let mut interactions = Vec::with_capacity(21);
    for (w, new, step) in [
        (
            0usize,
            [cols::NEW1_LO, cols::NEW1_HI],
            [cols::STEP_0, cols::STEP_1],
        ),
        (
            1,
            [cols::NEW2_LO, cols::NEW2_HI],
            [cols::STEP2_0, cols::STEP2_1],
        ),
    ] {
        let key = if w == 0 {
            word_key(address, cols::OFF, 0)
        } else {
            second_word_key(address, cols::OFF, cols::CARRY)
        };
        interactions.extend(word_check_of(
            key,
            cols::TIMESTAMP,
            words[w].0.clone(),
            [column(new[0]), column(new[1])],
            step,
            Multiplicity::Column(cols::MU),
        ));
    }
    for (_, checks) in words {
        interactions.extend(checks);
    }
    interactions.extend(byte_pair_checks(
        &cols::VALUE,
        Multiplicity::Column(cols::MU),
    ));
    interactions.push(dispatch(
        cols::TIMESTAMP,
        address,
        [
            word_of_bytes(&cols::VALUE[..4]),
            word_of_bytes(&cols::VALUE[4..]),
        ],
        flags(
            FLAG_STORE,
            &[
                (FLAG_W2, cols::WRITE2),
                (FLAG_W4, cols::WRITE4),
                (FLAG_W8, cols::WRITE8),
            ],
        ),
        [BusValue::constant(0), BusValue::constant(0)],
        cols::MU,
    ));
    interactions
}

// =========================================================================
// Constraints
// =========================================================================

/// MEM_STORE2W's 23 constraints (module docs).
#[derive(Clone, Copy)]
pub struct MemStore2WConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for MemStore2WConstraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        one_hot_constraints(b, 0, cols::OFF, cols::MU);
        for (idx, c) in [(9, cols::WRITE2), (10, cols::WRITE4), (11, cols::WRITE8)] {
            let root = bit(b, b.main(0, c));
            b.emit_base(idx, root);
        }
        let w2 = b.main(0, cols::WRITE2);
        let w4 = b.main(0, cols::WRITE4);
        let w8 = b.main(0, cols::WRITE8);
        let root = w2.clone() + w4.clone() + w8.clone() - b.main(0, cols::MU);
        b.emit_base(12, root);
        // The access crosses: `o + w > 8`.
        let root = w2 * sum(b, &cols::OFF[..7]);
        b.emit_base(13, root);
        let root = w4.clone() * sum(b, &cols::OFF[..5]);
        b.emit_base(14, root);
        let root = w8.clone() * b.main(0, cols::OFF[0]);
        b.emit_base(15, root);
        // The new halves: bytes 0 and 1 written at every width, 2 and 3 from
        // four bytes, 4 to 7 at eight.
        let four = w4 + w8.clone();
        let act = [
            None,
            None,
            Some(four.clone()),
            Some(four),
            Some(w8.clone()),
            Some(w8.clone()),
            Some(w8.clone()),
            Some(w8),
        ];
        for (idx, (half, new)) in [
            (16, (0, cols::NEW1_LO)),
            (17, (1, cols::NEW1_HI)),
            (18, (2, cols::NEW2_LO)),
            (19, (3, cols::NEW2_HI)),
        ] {
            let root = stored_half(b, new, half, cols::OFF, &cols::OLD, &cols::VALUE, &act);
            b.emit_base(idx, root);
        }
        let root = bit(b, b.main(0, cols::MU));
        b.emit_base(20, root);
        carry_constraints(b, 21, cols::ADDRESS_LO, cols::OFF, cols::CARRY);
    }
}
