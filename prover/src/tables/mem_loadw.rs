//! MEM_LOADW: a CPU load of 1, 2 or 4 bytes inside one word (LB, LBU, LH,
//! LHU, LW, LWU), checked in one row with the word's token, under
//! [`MemoryFormat::Words`](super::fused_memory::MemoryFormat::Words).
//!
//! The row holds the word's eight bytes, all checked, and passes its token
//! through at `(ADDRESS_LO − o, ADDRESS_HI)`, `o` the one-hot offset; the
//! tokens force `o` to the address's offset in its word (`super::word_cpu`).
//! The result's bytes are the word's bytes from `o`, the width's worth (placed
//! by the one-hot), then LOAD's extension: `255·SIGNED·SIGN_BIT` with
//! `SIGN_BIT` the top bit of the highest byte read, by LOAD's `MSB8` sends.
//!
//! ## Columns (31)
//!
//! `ADDRESS_LO`, `ADDRESS_HI`, `TIMESTAMP`, `OFF[8]`, `READ2`, `READ4`,
//! `SIGNED`, `BYTES[8]` (the word), `RES[4]` (the result's low bytes),
//! `SIGN_BIT`, `EXT` (`SIGNED·SIGN_BIT`), `STEP_0`, `STEP_1`, `MU`.
//!
//! ## Interactions (12)
//!
//! - the word check at `MU`: key `(ADDRESS_LO − Σ k·OFF_k, ADDRESS_HI)`, the
//!   word `BYTES` as two halves consumed and produced;
//! - `ARE_BYTES` on the four pairs of `BYTES`, at `MU`;
//! - `MSB8[RES[0]] → SIGN_BIT` at `MU − READ2 − READ4`, `MSB8[RES[1]]` at
//!   `READ2`, `MSB8[RES[3]]` at `READ4`;
//! - the dispatch at `MU`: `(TIMESTAMP, 0, ADDRESS_LO, ADDRESS_HI, 0, 0,
//!   2·SIGNED + 4·READ2 + 8·READ4, Σ RES_j·2^(8j), 0xFFFF_FFFF·EXT)`.
//!
//! ## Constraints (25, degree 3)
//!
//! - idx 0–8: the one-hot (each `OFF_k` a bit, their sum `MU`);
//! - idx 9–11: `READ2`, `READ4`, `SIGNED` bits; 12: `READ2 + READ4` a bit;
//!   13: it implies `MU`;
//! - idx 14, 15: the window inside the word, `READ2·OFF_7 = 0`,
//!   `READ4·(OFF_5 + OFF_6 + OFF_7) = 0`;
//! - idx 16–19: the placement, `RES_j = Σ_k OFF_k·BYTES_(k+j)` for `j` below
//!   the width (`j = 0` always, `j = 1` under `READ2 + READ4`, `j = 2, 3`
//!   under `READ4`);
//! - idx 20–22: the extension, `RES_j = 255·EXT` for `j` from the width;
//! - idx 23: `EXT = SIGNED·SIGN_BIT`; 24: `MU` a bit.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity};
use stark::trace::TraceTable;

use super::bitwise::{BitwiseHistogram, BitwiseOperation, BitwiseOperationType};
use super::gpack::{TraceForm, WidthHint, generate_main};
use super::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};
use super::word_cpu::{
    FLAG_SIGNED, FLAG_W2, FLAG_W4, WordClass, bit, dispatch, flags, limb_lookups, load_result,
    msb8, one_hot_constraints, pair_lookups, placed, word_class, word_key,
};
use super::word_memory::{
    WORD_BYTES, byte_pair_checks, halves, step_less_one, step_limbs, term, word_check_of,
    word_of_bytes,
};

/// Column definitions for the MEM_LOADW table.
pub mod cols {
    /// The address: its low and its high 32-bit word.
    pub const ADDRESS_LO: usize = 0;
    pub const ADDRESS_HI: usize = 1;
    /// The dispatch's timestamp.
    pub const TIMESTAMP: usize = 2;
    /// The address's offset in its word, one-hot: `OFF[k] = 1` at offset `k`.
    pub const OFF: [usize; 8] = [3, 4, 5, 6, 7, 8, 9, 10];
    /// The width flags, LOAD's: 2 or 4 bytes (neither: 1 byte).
    pub const READ2: usize = 11;
    pub const READ4: usize = 12;
    /// 1: sign-extend, 0: zero-extend.
    pub const SIGNED: usize = 13;
    /// The word's eight bytes, little-endian.
    pub const BYTES: [usize; 8] = [14, 15, 16, 17, 18, 19, 20, 21];
    /// The result's low four bytes: the bytes read, then the extension.
    pub const RES: [usize; 4] = [22, 23, 24, 25];
    /// The top bit of the highest byte read.
    pub const SIGN_BIT: usize = 26;
    /// `SIGNED·SIGN_BIT`: a byte past the width is `255·EXT`.
    pub const EXT: usize = 27;
    /// `ts − old_ts − 1`, low and high 16 bits.
    pub const STEP_0: usize = 28;
    pub const STEP_1: usize = 29;
    /// Whether the row is a load.
    pub const MU: usize = 30;
    pub const NUM_COLUMNS: usize = 31;
}

// =========================================================================
// The record
// =========================================================================

/// One MEM_LOADW row as the walk keeps it: 32 bytes, `#[repr(C)]`, its layout
/// fixed (asserted below).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LoadWRow {
    address: u64,
    timestamp: u64,
    /// `timestamp − old_timestamp − 1`.
    step: u32,
    /// 1, 2 or 4.
    width: u8,
    signed: bool,
    /// The word holding the bytes read.
    word: u64,
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<LoadWRow>() == 32);
    assert!(offset_of!(LoadWRow, address) == 0);
    assert!(offset_of!(LoadWRow, timestamp) == 8);
    assert!(offset_of!(LoadWRow, step) == 16);
    assert!(offset_of!(LoadWRow, width) == 20);
    assert!(offset_of!(LoadWRow, signed) == 21);
    assert!(offset_of!(LoadWRow, word) == 24);
};

#[cfg_attr(not(test), allow(dead_code))]
impl LoadWRow {
    /// The row of a `width`-byte load at `address` and `timestamp` from the
    /// word `word` (the word holding `address`), last touched at
    /// `old_timestamp`; or the reason no MEM_LOADW row takes it (an access that
    /// is not inside one word, a step the limbs cannot hold).
    pub(crate) fn new(
        address: u64,
        timestamp: u64,
        old_timestamp: u64,
        width: u8,
        signed: bool,
        word: u64,
    ) -> Result<Self, String> {
        if word_class(address, width)? != WordClass::Within {
            return Err(format!(
                "MEM_LOADW takes fewer than 8 bytes inside a word, not {width} at {address:#x}"
            ));
        }
        let step = step_less_one(old_timestamp, timestamp)?;
        Ok(Self {
            address,
            timestamp,
            step,
            width,
            signed,
            word,
        })
    }

    /// The address's offset in its word.
    #[inline]
    fn offset(&self) -> usize {
        (self.address % WORD_BYTES) as usize
    }

    /// The result (extended) and its sign bit.
    #[inline]
    pub(crate) fn result(&self) -> ([u8; 8], bool) {
        load_result(
            &self.word.to_le_bytes(),
            self.offset(),
            self.width,
            self.signed,
        )
    }

    /// The row's BITWISE lookups: the word's four pairs, the sign bit's
    /// `MSB8`, the limbs' `IS_HALF`.
    #[inline]
    pub(crate) fn bitwise_ops(&self) -> impl Iterator<Item = BitwiseOperation> {
        let bytes = self.word.to_le_bytes();
        let top = bytes[self.offset() + usize::from(self.width) - 1];
        let pairs: Vec<BitwiseOperation> = pair_lookups(&bytes).collect();
        pairs
            .into_iter()
            .chain(std::iter::once(BitwiseOperation::single_byte(
                BitwiseOperationType::Msb8,
                top,
            )))
            .chain(limb_lookups(self.step))
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
impl LoadWRow {
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

/// The widths MEM_LOADW traces needed so far in this process.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) static WIDTHS: WidthHint = WidthHint::new();

/// The MEM_LOADW trace of `segments`' rows, one after another, then zero rows
/// to a power of two of at least four, in `form`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn generate_mem_loadw_trace_segments_as(
    segments: &[&[LoadWRow]],
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
            table.set_bool(row, cols::READ2, r.width == 2);
            table.set_bool(row, cols::READ4, r.width == 4);
            table.set_bool(row, cols::SIGNED, r.signed);
            table.set_bytes(row, cols::BYTES[0], &r.word.to_le_bytes());
            let (res, sign_bit) = r.result();
            table.set_bytes(row, cols::RES[0], &res[..4]);
            table.set_bool(row, cols::SIGN_BIT, sign_bit);
            table.set_bool(row, cols::EXT, r.signed && sign_bit);
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

/// MEM_LOADW's 12 interactions (module docs).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let address = [cols::ADDRESS_LO, cols::ADDRESS_HI];
    let word = || {
        [
            word_of_bytes(&cols::BYTES[..4]),
            word_of_bytes(&cols::BYTES[4..]),
        ]
    };
    let mut interactions = word_check_of(
        word_key(address, cols::OFF, 0),
        cols::TIMESTAMP,
        word(),
        word(),
        [cols::STEP_0, cols::STEP_1],
        Multiplicity::Column(cols::MU),
    );
    interactions.extend(byte_pair_checks(
        &cols::BYTES,
        Multiplicity::Column(cols::MU),
    ));
    // MSB8 on the highest byte read: byte 0 at `MU − READ2 − READ4` (one
    // byte), byte 1 at `READ2`, byte 3 at `READ4`.
    let read1 = Multiplicity::Linear(vec![
        term(1, cols::MU),
        term(-1, cols::READ2),
        term(-1, cols::READ4),
    ]);
    interactions.push(msb8(cols::RES[0], cols::SIGN_BIT, read1));
    interactions.push(msb8(
        cols::RES[1],
        cols::SIGN_BIT,
        Multiplicity::Column(cols::READ2),
    ));
    interactions.push(msb8(
        cols::RES[3],
        cols::SIGN_BIT,
        Multiplicity::Column(cols::READ4),
    ));
    interactions.push(dispatch(
        cols::TIMESTAMP,
        address,
        [BusValue::constant(0), BusValue::constant(0)],
        flags(
            0,
            &[
                (FLAG_SIGNED, cols::SIGNED),
                (FLAG_W2, cols::READ2),
                (FLAG_W4, cols::READ4),
            ],
        ),
        [
            word_of_bytes(&cols::RES),
            BusValue::linear(vec![LinearTerm::Column {
                coefficient: 0xFFFF_FFFF,
                column: cols::EXT,
            }]),
        ],
        cols::MU,
    ));
    interactions
}

// =========================================================================
// Constraints
// =========================================================================

/// MEM_LOADW's 25 constraints (module docs).
#[derive(Clone, Copy)]
pub struct MemLoadWConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for MemLoadWConstraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        let mut idx = one_hot_constraints(b, 0, cols::OFF, cols::MU);
        for c in [cols::READ2, cols::READ4, cols::SIGNED] {
            let root = bit(b, b.main(0, c));
            b.emit_base(idx, root);
            idx += 1;
        }
        let r2 = b.main(0, cols::READ2);
        let r4 = b.main(0, cols::READ4);
        let width = r2.clone() + r4.clone();
        let root = bit(b, width.clone());
        b.emit_base(12, root);
        let root = width.clone() * (b.one() - b.main(0, cols::MU));
        b.emit_base(13, root);
        // The window: `o + w ≤ 8`.
        let root = r2 * b.main(0, cols::OFF[7]);
        b.emit_base(14, root);
        let past = b.main(0, cols::OFF[5]) + b.main(0, cols::OFF[6]) + b.main(0, cols::OFF[7]);
        b.emit_base(15, r4.clone() * past);
        // The placement, under each byte's width gate.
        let root = b.main(0, cols::RES[0]) - placed(b, cols::OFF, &cols::BYTES, 0);
        b.emit_base(16, root);
        let root =
            width.clone() * (b.main(0, cols::RES[1]) - placed(b, cols::OFF, &cols::BYTES, 1));
        b.emit_base(17, root);
        for (k, j) in [(18, 2), (19, 3)] {
            let root =
                r4.clone() * (b.main(0, cols::RES[j]) - placed(b, cols::OFF, &cols::BYTES, j));
            b.emit_base(k, root);
        }
        // The extension, from the width.
        let ext = b.const_base(255) * b.main(0, cols::EXT);
        let root = (b.one() - width) * (b.main(0, cols::RES[1]) - ext.clone());
        b.emit_base(20, root);
        for (k, j) in [(21, 2), (22, 3)] {
            let root = (b.one() - r4.clone()) * (b.main(0, cols::RES[j]) - ext.clone());
            b.emit_base(k, root);
        }
        let root = b.main(0, cols::EXT) - b.main(0, cols::SIGNED) * b.main(0, cols::SIGN_BIT);
        b.emit_base(23, root);
        let root = bit(b, b.main(0, cols::MU));
        b.emit_base(24, root);
    }
}
