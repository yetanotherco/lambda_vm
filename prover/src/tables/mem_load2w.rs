//! MEM_LOAD2W: a CPU load of 2, 4 or 8 bytes across two words (LH, LHU, LW,
//! LWU, LD), checked in one row with both words' tokens, under
//! [`MemoryFormat::Words`](super::fused_memory::MemoryFormat::Words).
//!
//! The row holds both words' sixteen bytes, all checked, and passes each
//! word's token through, the first at `(ADDRESS_LO − o, ADDRESS_HI)` and the
//! second at `(ADDRESS_LO − o + 8 − 2^32·CARRY, ADDRESS_HI + CARRY)`, each
//! with its own old timestamp; the access crosses (`o + w > 8`). `CARRY` is 1
//! exactly when the first word is the last of its low range, `2^32 − 8`: the
//! second word is then low word 0 of the next high word
//! ([`super::word_cpu::second_word_key`]). The result's bytes are the sixteen
//! from `o`, the width's worth, then LOAD's extension (`super::word_cpu`).
//!
//! ## Columns (47)
//!
//! `ADDRESS_LO`, `ADDRESS_HI`, `TIMESTAMP`, `OFF[8]`, `READ2`, `READ4`,
//! `READ8`, `SIGNED`, `BYTES[16]` (the first word's, then the second's),
//! `RES[8]`, `SIGN_BIT`, `EXT`, `STEP_0`, `STEP_1` (the first word's step),
//! `STEP2_0`, `STEP2_1` (the second's), `MU`, `CARRY`.
//!
//! ## Interactions (19)
//!
//! - the first word's check and the second's at `MU`, old = new = the word;
//! - `ARE_BYTES` on the eight pairs of `BYTES`, at `MU`;
//! - `MSB8[RES[1]] → SIGN_BIT` at `READ2`, `MSB8[RES[3]]` at `READ4` (an
//!   8-byte load extends nothing);
//! - the dispatch at `MU`: `(TIMESTAMP, 0, ADDRESS_LO, ADDRESS_HI, 0, 0,
//!   2·SIGNED + 4·READ2 + 8·READ4 + 16·READ8, RES as two words)`.
//!
//! ## Constraints (35, degree 3)
//!
//! - idx 0–8: the one-hot;
//! - idx 9–12: `READ2`, `READ4`, `READ8`, `SIGNED` bits; 13: exactly one width
//!   on a row, `READ2 + READ4 + READ8 = MU` (one byte never crosses);
//! - idx 14–16: the access crosses, `READ2·Σ_(k<7) OFF_k = 0`,
//!   `READ4·Σ_(k<5) OFF_k = 0`, `READ8·OFF_0 = 0`;
//! - idx 17–24: the placement, `RES_j = Σ_k OFF_k·BYTES_(k+j)` for `j` below
//!   the width (`j < 2` always, `j < 4` under `READ4 + READ8`, `j < 8` under
//!   `READ8`);
//! - idx 25–30: the extension, `RES_j = 255·EXT` for `j` from the width;
//! - idx 31: `EXT = SIGNED·SIGN_BIT`; 32: `MU` a bit;
//! - idx 33: `CARRY` a bit; 34: `CARRY·(ADDRESS_LO − Σ k·OFF_k − (2^32 − 8))
//!   = 0`, a carry only from a low range's last word
//!   ([`super::word_cpu::carry_constraints`]).

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, Multiplicity};
use stark::trace::TraceTable;

use super::bitwise::{BitwiseHistogram, BitwiseOperation, BitwiseOperationType};
use super::gpack::{TraceForm, WidthHint, generate_main};
use super::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};
use super::word_cpu::{
    FLAG_SIGNED, FLAG_W2, FLAG_W4, FLAG_W8, WordClass, bit, carries, carry_constraints, dispatch,
    flags, limb_lookups, load_result, msb8, one_hot_constraints, pair_lookups, placed,
    second_word_key, sum, word_class, word_key,
};
use super::word_memory::{
    WORD_BYTES, checked_word, halves, step_less_one, step_limbs, word_check_of, word_of_bytes,
};

/// Column definitions for the MEM_LOAD2W table.
pub mod cols {
    /// The address: its low and its high 32-bit word.
    pub const ADDRESS_LO: usize = 0;
    pub const ADDRESS_HI: usize = 1;
    /// The dispatch's timestamp.
    pub const TIMESTAMP: usize = 2;
    /// The address's offset in its first word, one-hot.
    pub const OFF: [usize; 8] = [3, 4, 5, 6, 7, 8, 9, 10];
    /// The width flags, LOAD's: 2, 4 or 8 bytes.
    pub const READ2: usize = 11;
    pub const READ4: usize = 12;
    pub const READ8: usize = 13;
    /// 1: sign-extend, 0: zero-extend.
    pub const SIGNED: usize = 14;
    /// The two words' sixteen bytes, little-endian, the first word's first.
    pub const BYTES: [usize; 16] = [
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    ];
    /// The result's eight bytes: the bytes read, then the extension.
    pub const RES: [usize; 8] = [31, 32, 33, 34, 35, 36, 37, 38];
    /// The top bit of the highest byte read.
    pub const SIGN_BIT: usize = 39;
    /// `SIGNED·SIGN_BIT`.
    pub const EXT: usize = 40;
    /// The first word's `ts − old_ts − 1`, low and high 16 bits.
    pub const STEP_0: usize = 41;
    pub const STEP_1: usize = 42;
    /// The second word's.
    pub const STEP2_0: usize = 43;
    pub const STEP2_1: usize = 44;
    /// Whether the row is a load.
    pub const MU: usize = 45;
    /// 1 when the first word is the last of its low range: the second word's
    /// key carries into the high word.
    pub const CARRY: usize = 46;
    pub const NUM_COLUMNS: usize = 47;
}

// =========================================================================
// The record
// =========================================================================

/// One MEM_LOAD2W row as the walk keeps it: 48 bytes, `#[repr(C)]`, its
/// layout fixed (asserted below).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Load2WRow {
    address: u64,
    timestamp: u64,
    /// Each word's `timestamp − old_timestamp − 1`.
    steps: [u32; 2],
    /// The two words, the one holding `address` first.
    words: [u64; 2],
    /// 2, 4 or 8.
    width: u8,
    signed: bool,
}

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<Load2WRow>() == 48);
    assert!(offset_of!(Load2WRow, address) == 0);
    assert!(offset_of!(Load2WRow, timestamp) == 8);
    assert!(offset_of!(Load2WRow, steps) == 16);
    assert!(offset_of!(Load2WRow, words) == 24);
    assert!(offset_of!(Load2WRow, width) == 40);
    assert!(offset_of!(Load2WRow, signed) == 41);
};

#[cfg_attr(not(test), allow(dead_code))]
impl Load2WRow {
    /// The row of a `width`-byte load at `address` and `timestamp` across the
    /// words `words` (the one holding `address`, then the next), last touched
    /// at `old_timestamps`; or the reason no MEM_LOAD2W row takes it.
    pub(crate) fn new(
        address: u64,
        timestamp: u64,
        old_timestamps: [u64; 2],
        width: u8,
        signed: bool,
        words: [u64; 2],
    ) -> Result<Self, String> {
        if word_class(address, width)? != WordClass::Crossing {
            return Err(format!(
                "MEM_LOAD2W takes an access across two words, not {width} at {address:#x}"
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
            words,
            width,
            signed,
        })
    }

    /// The address's offset in its first word.
    #[inline]
    fn offset(&self) -> usize {
        (self.address % WORD_BYTES) as usize
    }

    /// The two words' sixteen bytes.
    #[inline]
    fn bytes(&self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&self.words[0].to_le_bytes());
        bytes[8..].copy_from_slice(&self.words[1].to_le_bytes());
        bytes
    }

    /// The result (extended) and its sign bit.
    #[inline]
    pub(crate) fn result(&self) -> ([u8; 8], bool) {
        load_result(&self.bytes(), self.offset(), self.width, self.signed)
    }

    /// The row's BITWISE lookups: the eight pairs, the sign bit's `MSB8` (a
    /// load narrower than 8 bytes), both words' limbs' `IS_HALF`.
    #[inline]
    pub(crate) fn bitwise_ops(&self) -> impl Iterator<Item = BitwiseOperation> {
        let bytes = self.bytes();
        let mut ops: Vec<BitwiseOperation> = pair_lookups(&bytes).collect();
        if self.width < 8 {
            let top = bytes[self.offset() + usize::from(self.width) - 1];
            ops.push(BitwiseOperation::single_byte(
                BitwiseOperationType::Msb8,
                top,
            ));
        }
        ops.extend(limb_lookups(self.steps[0]));
        ops.extend(limb_lookups(self.steps[1]));
        ops.into_iter()
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
impl Load2WRow {
    #[inline]
    pub(crate) fn address(&self) -> u64 {
        self.address
    }

    #[inline]
    pub(crate) fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Each word's timestamp before the load.
    #[inline]
    pub(crate) fn old_timestamps(&self) -> [u64; 2] {
        self.steps.map(|s| self.timestamp - 1 - u64::from(s))
    }

    #[inline]
    pub(crate) fn words(&self) -> [u64; 2] {
        self.words
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// The widths MEM_LOAD2W traces needed so far in this process.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) static WIDTHS: WidthHint = WidthHint::new();

/// The MEM_LOAD2W trace of `segments`' rows, one after another, then zero
/// rows to a power of two of at least four, in `form`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn generate_mem_load2w_trace_segments_as(
    segments: &[&[Load2WRow]],
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
            table.set_bool(row, cols::READ8, r.width == 8);
            table.set_bool(row, cols::SIGNED, r.signed);
            table.set_bytes(row, cols::BYTES[0], &r.bytes());
            let (res, sign_bit) = r.result();
            table.set_bytes(row, cols::RES[0], &res);
            table.set_bool(row, cols::SIGN_BIT, sign_bit);
            table.set_bool(row, cols::EXT, r.signed && sign_bit);
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

/// MEM_LOAD2W's 19 interactions (module docs).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let address = [cols::ADDRESS_LO, cols::ADDRESS_HI];
    let words = [&cols::BYTES[..8], &cols::BYTES[8..]]
        .map(|bytes| checked_word(bytes, Multiplicity::Column(cols::MU)));
    let mut interactions = Vec::with_capacity(19);
    for (w, step) in [
        (0, [cols::STEP_0, cols::STEP_1]),
        (1, [cols::STEP2_0, cols::STEP2_1]),
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
            words[w].0.clone(),
            step,
            Multiplicity::Column(cols::MU),
        ));
    }
    for (_, checks) in words {
        interactions.extend(checks);
    }
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
                (FLAG_W8, cols::READ8),
            ],
        ),
        [
            word_of_bytes(&cols::RES[..4]),
            word_of_bytes(&cols::RES[4..]),
        ],
        cols::MU,
    ));
    interactions
}

// =========================================================================
// Constraints
// =========================================================================

/// MEM_LOAD2W's 35 constraints (module docs).
#[derive(Clone, Copy)]
pub struct MemLoad2WConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for MemLoad2WConstraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        one_hot_constraints(b, 0, cols::OFF, cols::MU);
        for (idx, c) in [
            (9, cols::READ2),
            (10, cols::READ4),
            (11, cols::READ8),
            (12, cols::SIGNED),
        ] {
            let root = bit(b, b.main(0, c));
            b.emit_base(idx, root);
        }
        let r2 = b.main(0, cols::READ2);
        let r4 = b.main(0, cols::READ4);
        let r8 = b.main(0, cols::READ8);
        let root = r2.clone() + r4.clone() + r8.clone() - b.main(0, cols::MU);
        b.emit_base(13, root);
        // The access crosses: `o + w > 8`.
        let root = r2 * sum(b, &cols::OFF[..7]);
        b.emit_base(14, root);
        let root = r4.clone() * sum(b, &cols::OFF[..5]);
        b.emit_base(15, root);
        let root = r8.clone() * b.main(0, cols::OFF[0]);
        b.emit_base(16, root);
        // The placement: bytes 0 and 1 at every width, 2 and 3 from four
        // bytes, 4 to 7 at eight.
        let four = r4 + r8.clone();
        for j in 0..8 {
            let diff = b.main(0, cols::RES[j]) - placed(b, cols::OFF, &cols::BYTES, j);
            let root = match j {
                0 | 1 => diff,
                2 | 3 => four.clone() * diff,
                _ => r8.clone() * diff,
            };
            b.emit_base(17 + j, root);
        }
        // The extension from the width.
        let ext = b.const_base(255) * b.main(0, cols::EXT);
        for j in 2..8 {
            let below = if j < 4 {
                b.one() - four.clone()
            } else {
                b.one() - r8.clone()
            };
            let root = below * (b.main(0, cols::RES[j]) - ext.clone());
            b.emit_base(23 + j, root);
        }
        let root = b.main(0, cols::EXT) - b.main(0, cols::SIGNED) * b.main(0, cols::SIGN_BIT);
        b.emit_base(31, root);
        let root = bit(b, b.main(0, cols::MU));
        b.emit_base(32, root);
        carry_constraints(b, 33, cols::ADDRESS_LO, cols::OFF, cols::CARRY);
    }
}

#[cfg(test)]
impl Load2WRow {
    /// A test's fault: the same row, its timestamp `cycles` later (a row the
    /// CPU never dispatched).
    pub(crate) fn later(mut self, cycles: u64) -> Self {
        self.timestamp += cycles;
        self
    }
}
