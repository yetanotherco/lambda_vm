//! COPY_W: the COPY chip (memcpy and memmove, memset, the commit syscall)
//! under [`MemoryFormat::Words`](super::fused_memory::MemoryFormat::Words),
//! where RAM is checked by aligned 8-byte words ([`super::word_memory`]).
//!
//! The three functionalities, their ECALL numbers, their operand registers
//! and their timestamp order are COPY's ([`super::copy`]): a copy reads at
//! `T` and writes at `T + 1`, a memset writes at `T` and reads at `T + 1`
//! (`dst = src + 8`, so each row's read sees the previous row's write and the
//! seed propagates), and a commit reads at `T` and sends its bytes on the
//! `Commit` bus by index. What changes is the unit: a word token covers eight
//! bytes and a word takes at most one access a timestamp, so a call can no
//! longer touch a byte at a time.
//!
//! ## A row is one destination word
//!
//! Row `k` of a call writes the aligned destination word at `DST_KEY + 8k`.
//! Its *window*, the bytes `START..=LAST` of that word, is what the call
//! writes there: from `dst`'s offset on the first row, to the word's end on
//! every row but the last, and up to the last byte on the last row. Bytes
//! outside the window keep their old value: the row reads the word and writes
//! it back (one access message, old and new word).
//!
//! The bytes come from the source words `A` (key `SRC_KEY`) and `B` (key
//! `SRC_KEY + 8`): window byte `j` is byte `j + SHIFT` of `A ‖ B`, with
//! `SHIFT = (src − dst) mod 8` fixed for the call. `B` is the next row's `A`:
//! the chain ([`BusId::CopyWordNext`]) carries its eight bytes, so a source
//! word is read once. A row reads `A` only on the first row and only when its
//! window uses it (`READ_A`), and `B` on every row but the last, and on the
//! last when its window reaches it (`READ_B`): exactly the words the call's
//! range covers.
//!
//! A commit has no destination word: its "destination" is the running commit
//! index (x254). Its rows read the source in place (`SHIFT = 0`, the window
//! from `src`'s offset), and send byte `j` of `A` as the pair
//! `(DST_KEY + j − 8, A_j)` on the `Commit` bus; `DST_KEY` is the index less
//! the offset plus 8, so it never goes below zero.
//!
//! ## What pins what
//!
//! - **Keys.** `DST_KEY`'s and `SRC_KEY`'s low words are `IS_HALF` limbs and
//!   step by 8 along the chain; their high words ride the chain unchanged from
//!   the first row, where the operand registers pin them. A word token exists
//!   only at an aligned PAGE_W key, so the keys a row reads and writes are
//!   aligned, and the one-hot `START` (the destination's offset) and `SHIFT`
//!   are the true ones: x10 reads `DST_KEY + START`, x11 reads
//!   `SRC_KEY + START + SHIFT`.
//! - **The window.** `START` and `LAST` are one-hot, `LAST` never before
//!   `START`; every row but the first starts at byte 0 and every row but the
//!   last ends at byte 7; the count drops by the window's width and the last
//!   row leaves zero. So the windows cover exactly the call's bytes.
//! - **The shift and the carry** ride the chain: a row cannot change the shift
//!   mid-call, or take an `A` other than the previous row's `B`.
//! - **Bytes.** `A` is range-checked when it is read, `B` when it is read (so
//!   every carried `A` was checked), the old destination word whenever it is
//!   written. The new word is the old one with the window's bytes replaced, a
//!   degree-3 identity per half: a sum of checked bytes.
//! - **Reads.** `READ_A` and `READ_B` are fixed by the window and the shift:
//!   a word whose bytes the window places is always read.
//! - **memset** keeps `DST_KEY = SRC_KEY + 8` and `SHIFT = 0`: `dst = src + 8`
//!   exactly, as the executor demands. A commit keeps `SHIFT = 0`.
//!
//! ## Columns (77)
//!
//! `TIMESTAMP`; `COUNT_DECR` (two halves: the count left after the row);
//! `SRC_KEY_LO` (two halves), `SRC_KEY_HI`; `DST_KEY_LO` (two halves),
//! `DST_KEY_HI`; `FIRST`, `END`, `MU`, `ZERO` (a zero-length call's one row),
//! `IS_SET`, `IS_WRITE`, `FIRST_WRITE`; `READ_A`, `READ_B`, `MU_W` (the
//! destination access); `START[8]`, `LAST[8]`, `SHIFT[8]`; `A[8]`, `B[8]`,
//! `OLD[8]`; `NEW_LO`, `NEW_HI`; `COMMIT[8]` (byte `j` committed).
//!
//! ## Interactions (37)
//!
//! The ECALL; x11, x12, x10 (a copy or memset) or x254 and x10 (a commit);
//! the chain forward and back; six `IS_HALF`; the three word messages; eight
//! commit pairs; twelve `ARE_BYTES` pairs.
//!
//! ## Limits (completeness only)
//!
//! A call whose words would cross a 4 GiB boundary (a key's low word past
//! `2^32 − 8`, or `src` less its offset and shift below its high word's start),
//! or whose count is `2^32` or more, has no rows: the walk refuses it. The
//! executor caps a copy or memset at 256 bytes and a run's commits at one
//! mebibyte.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity};
use stark::trace::TraceTable;

use executor::vm::instruction::execution::{
    DMA_MEMCPY_SYSCALL_NUMBER, DMA_MEMSET_GAP, DMA_MEMSET_SYSCALL_NUMBER,
};

use super::bitwise::{BitwiseHistogram, BitwiseOperation, BitwiseOperationType};
use super::copy::Functionality;
use super::gpack::{TraceForm, WidthHint, generate_main};
use super::memw_w::{MemwWordRow, MessageKind};
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};
use super::word_cpu::{bit, placed};
use super::word_memory::{
    WORD_BYTES, access_message, byte_pair_checks, column, term, word_of_bytes,
};
use crate::constraints::templates::emit_is_bit;

/// The commit syscall's number (ecall 64).
const COMMIT_SYSCALL_NUMBER: u64 = 64;

const MEMCPY_LO32: i64 = (DMA_MEMCPY_SYSCALL_NUMBER & 0xFFFF_FFFF) as i64;
const MEMCPY_HI32: i64 = (DMA_MEMCPY_SYSCALL_NUMBER >> 32) as i64;
const MEMSET_LO32: i64 = (DMA_MEMSET_SYSCALL_NUMBER & 0xFFFF_FFFF) as i64;
const MEMSET_HI32: i64 = (DMA_MEMSET_SYSCALL_NUMBER >> 32) as i64;
const COMMIT_LO32: i64 = (COMMIT_SYSCALL_NUMBER & 0xFFFF_FFFF) as i64;
const COMMIT_HI32: i64 = (COMMIT_SYSCALL_NUMBER >> 32) as i64;

/// How far a commit's `DST_KEY` sits above the commit index less the offset.
const COMMIT_KEY_BIAS: i64 = WORD_BYTES as i64;

/// Column definitions for the COPY_W table.
pub mod cols {
    /// The call's timestamp `T`.
    pub const TIMESTAMP: usize = 0;
    /// The count left after this row, two 16-bit halves.
    pub const COUNT_DECR: [usize; 2] = [1, 2];
    /// The source word `A`'s key: its low word as two 16-bit halves, its high
    /// word.
    pub const SRC_KEY_LO: [usize; 2] = [3, 4];
    pub const SRC_KEY_HI: usize = 5;
    /// The destination word's key (a commit's: the index less the offset,
    /// plus 8), likewise.
    pub const DST_KEY_LO: [usize; 2] = [6, 7];
    pub const DST_KEY_HI: usize = 8;
    /// The call's first row, its last row, an active row.
    pub const FIRST: usize = 9;
    pub const END: usize = 10;
    pub const MU: usize = 11;
    /// The one row of a zero-length call: no window, no access.
    pub const ZERO: usize = 12;
    /// The functionality: memset, commit (neither: a copy).
    pub const IS_SET: usize = 13;
    pub const IS_WRITE: usize = 14;
    /// `FIRST · IS_WRITE`.
    pub const FIRST_WRITE: usize = 15;
    /// Whether the row reads `A`, and `B`.
    pub const READ_A: usize = 16;
    pub const READ_B: usize = 17;
    /// Whether the row writes its destination word: `(MU − ZERO)·(1 − IS_WRITE)`.
    pub const MU_W: usize = 18;
    /// The window's first and last byte, and the shift, each one-hot.
    pub const START: [usize; 8] = [19, 20, 21, 22, 23, 24, 25, 26];
    pub const LAST: [usize; 8] = [27, 28, 29, 30, 31, 32, 33, 34];
    pub const SHIFT: [usize; 8] = [35, 36, 37, 38, 39, 40, 41, 42];
    /// The source words `A` and `B`, little-endian bytes.
    pub const A: [usize; 8] = [43, 44, 45, 46, 47, 48, 49, 50];
    pub const B: [usize; 8] = [51, 52, 53, 54, 55, 56, 57, 58];
    /// The destination word before the row, little-endian bytes.
    pub const OLD: [usize; 8] = [59, 60, 61, 62, 63, 64, 65, 66];
    /// The destination word after the row, low and high halves.
    pub const NEW_LO: usize = 67;
    pub const NEW_HI: usize = 68;
    /// Whether a commit row commits its byte `j`: `IS_WRITE` on the window.
    pub const COMMIT: [usize; 8] = [69, 70, 71, 72, 73, 74, 75, 76];
    pub const NUM_COLUMNS: usize = 77;

    /// `A ‖ B`: the sixteen bytes a window places from.
    pub const AB: [usize; 16] = [
        43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    ];
}

/// The name the AIR set gives a COPY_W table: `COPY_W[i]`.
pub const NAME: &str = "COPY_W";

// =========================================================================
// The record
// =========================================================================

pub(crate) const FLAG_FIRST: u8 = 1;
pub(crate) const FLAG_END: u8 = 1 << 1;
pub(crate) const FLAG_ZERO: u8 = 1 << 2;
pub(crate) const FLAG_SET: u8 = 1 << 3;
pub(crate) const FLAG_WRITE: u8 = 1 << 4;
pub(crate) const FLAG_READ_A: u8 = 1 << 5;
pub(crate) const FLAG_READ_B: u8 = 1 << 6;

/// One COPY_W row: a call's addresses and window ([`call_rows`]), then its
/// words, which the walk reads ([`Self::set_words`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CopyWordRow {
    /// The call's timestamp `T`.
    pub(crate) timestamp: u64,
    /// `A`'s key and the destination word's (a commit's: the index less the
    /// offset, plus 8).
    pub(crate) src_key: u64,
    pub(crate) dst_key: u64,
    /// The count left after the row.
    pub(crate) count_decr: u32,
    /// The window's first and last byte, and the shift.
    pub(crate) start: u8,
    pub(crate) last: u8,
    pub(crate) shift: u8,
    /// `FIRST`, `END`, `ZERO`, `IS_SET`, `IS_WRITE`, `READ_A`, `READ_B`.
    pub(crate) flags: u8,
    /// The source words and the destination's old word.
    pub(crate) a: u64,
    pub(crate) b: u64,
    pub(crate) old: u64,
}

impl CopyWordRow {
    #[inline]
    fn flag(&self, f: u8) -> bool {
        self.flags & f != 0
    }

    pub(crate) fn timestamp(&self) -> u64 {
        self.timestamp
    }

    pub(crate) fn src_key(&self) -> u64 {
        self.src_key
    }

    pub(crate) fn dst_key(&self) -> u64 {
        self.dst_key
    }

    pub(crate) fn count_decr(&self) -> u32 {
        self.count_decr
    }

    pub(crate) fn first(&self) -> bool {
        self.flag(FLAG_FIRST)
    }

    pub(crate) fn end(&self) -> bool {
        self.flag(FLAG_END)
    }

    pub(crate) fn zero(&self) -> bool {
        self.flag(FLAG_ZERO)
    }

    pub(crate) fn functionality(&self) -> Functionality {
        if self.flag(FLAG_SET) {
            Functionality::Set
        } else if self.flag(FLAG_WRITE) {
            Functionality::Write
        } else {
            Functionality::Copy
        }
    }

    pub(crate) fn reads_a(&self) -> bool {
        self.flag(FLAG_READ_A)
    }

    pub(crate) fn reads_b(&self) -> bool {
        self.flag(FLAG_READ_B)
    }

    /// Whether the row writes its destination word.
    pub(crate) fn writes(&self) -> bool {
        !self.zero() && !self.flag(FLAG_WRITE)
    }

    /// The window's first and last byte (none on a zero-length call's row).
    pub(crate) fn window(&self) -> Option<(u8, u8)> {
        (!self.zero()).then_some((self.start, self.last))
    }

    pub(crate) fn shift(&self) -> u8 {
        self.shift
    }

    /// The bytes the row moves.
    pub(crate) fn width(&self) -> u64 {
        self.window()
            .map_or(0, |(start, last)| u64::from(last - start) + 1)
    }

    /// Whether byte `j` of the destination word is in the window.
    #[inline]
    pub(crate) fn in_window(&self, j: usize) -> bool {
        self.window()
            .is_some_and(|(start, last)| usize::from(start) <= j && j <= usize::from(last))
    }

    pub(crate) fn a(&self) -> u64 {
        self.a
    }

    pub(crate) fn b(&self) -> u64 {
        self.b
    }

    pub(crate) fn old(&self) -> u64 {
        self.old
    }

    /// The words the walk read: `A`, `B` and the destination's old word (zero
    /// for one the row does not use).
    pub(crate) fn set_words(&mut self, a: u64, b: u64, old: u64) {
        self.a = a;
        self.b = b;
        self.old = old;
    }

    /// Byte `j + SHIFT` of `A ‖ B`: what window byte `j` takes.
    #[inline]
    pub(crate) fn placed(&self, j: usize) -> u8 {
        let k = j + usize::from(self.shift);
        let word = if k < 8 { self.a } else { self.b };
        (word >> (8 * (k % 8))) as u8
    }

    /// The destination word after the row: the old word with the window's
    /// bytes placed.
    pub(crate) fn new_word(&self) -> u64 {
        (0..8).fold(0u64, |word, j| {
            let byte = if self.in_window(j) {
                self.placed(j)
            } else {
                (self.old >> (8 * j)) as u8
            };
            word | u64::from(byte) << (8 * j)
        })
    }

    /// The source words' timestamp, `T + IS_SET`.
    pub(crate) fn read_timestamp(&self) -> u64 {
        self.timestamp + u64::from(self.flag(FLAG_SET))
    }

    /// The destination word's timestamp, `T + 1 − IS_SET`.
    pub(crate) fn write_timestamp(&self) -> u64 {
        self.timestamp + 1 - u64::from(self.flag(FLAG_SET))
    }

    /// `A`'s key, `B`'s key.
    pub(crate) fn a_key(&self) -> u64 {
        self.src_key
    }

    pub(crate) fn b_key(&self) -> u64 {
        self.src_key.wrapping_add(WORD_BYTES)
    }

    /// The commit pairs a commit row sends: `(index, byte)` for each window
    /// byte.
    pub(crate) fn commit_pairs(&self) -> impl Iterator<Item = (u64, u8)> + '_ {
        let commit = self.flag(FLAG_WRITE);
        (0..8)
            .filter(move |&j| commit && self.in_window(j))
            .map(|j| {
                (
                    (self.dst_key as u32 as u64 + j as u64).wrapping_sub(COMMIT_KEY_BIAS as u64),
                    (self.a >> (8 * j)) as u8,
                )
            })
    }

    /// The row's BITWISE lookups: `IS_HALF` on the count and the keys' low
    /// halves, then `ARE_BYTES` on `A` (read), `B` (read) and the old
    /// destination word (written).
    pub(crate) fn bitwise_ops(&self) -> impl Iterator<Item = BitwiseOperation> {
        let halves = [
            self.count_decr,
            self.src_key as u32,
            self.dst_key as u32,
        ]
        .into_iter()
        .flat_map(|w| [w as u16, (w >> 16) as u16])
        .map(|h| BitwiseOperation::halfword(BitwiseOperationType::IsHalf, h as u8, (h >> 8) as u8));
        let pairs = |on: bool, word: u64| {
            let bytes = word.to_le_bytes();
            (0..4)
                .filter(move |_| on)
                .map(move |p| {
                    BitwiseOperation::byte_op(
                        BitwiseOperationType::AreBytes,
                        bytes[2 * p],
                        bytes[2 * p + 1],
                    )
                })
        };
        halves
            .chain(pairs(self.reads_a(), self.a))
            .chain(pairs(self.reads_b(), self.b))
            .chain(pairs(self.writes(), self.old))
    }

    /// [`Self::bitwise_ops`] counted into `histogram`.
    #[inline]
    pub(crate) fn count_bitwise_into(&self, histogram: &mut BitwiseHistogram) {
        for op in self.bitwise_ops() {
            histogram.bump(op);
        }
    }
}

/// `x >> 32 == y >> 32`.
#[inline]
fn same_high(x: u64, y: u64) -> bool {
    x >> 32 == y >> 32
}

/// The rows of one call, `(functionality, timestamp, src, dst, count)` as the
/// walk reads them from x10, x11 and x12 (a commit's `dst` is x254, the
/// commit index), their words zero; or the reason no rows take the call.
pub(crate) fn call_rows(
    functionality: Functionality,
    timestamp: u64,
    src: u64,
    dst: u64,
    count: u64,
) -> Result<Vec<CopyWordRow>, String> {
    let kind = match functionality {
        Functionality::Copy => 0,
        Functionality::Set => FLAG_SET,
        Functionality::Write => FLAG_WRITE,
    };
    let commit = functionality == Functionality::Write;
    let bias = if commit { COMMIT_KEY_BIAS as u64 } else { 0 };
    if count >= 1 << 32 {
        return Err(format!(
            "a copy of {count:#x} bytes: COPY_W counts below 2^32"
        ));
    }
    if commit && dst.checked_add(count).is_none_or(|end| end >= 1 << 32) {
        return Err(format!(
            "a commit from index {dst:#x} of {count:#x} bytes passes 2^32"
        ));
    }
    if functionality == Functionality::Set && dst != src.wrapping_add(DMA_MEMSET_GAP) {
        return Err(format!(
            "a memset with dst {dst:#x} not src {src:#x} + {DMA_MEMSET_GAP}"
        ));
    }
    if count == 0 {
        // One row, first and last, moving nothing: the operands as they come
        // (no window, so no offset or shift).
        let dst_key = dst.wrapping_add(bias);
        if !same_high(dst_key, dst) {
            return Err(format!("a commit index {dst:#x} at the top of its range"));
        }
        return Ok(vec![CopyWordRow {
            timestamp,
            src_key: src,
            dst_key,
            flags: FLAG_FIRST | FLAG_END | FLAG_ZERO | kind,
            ..CopyWordRow::default()
        }]);
    }
    // The window's grid: the destination's words, or a commit's source's.
    let (offset, shift, src_key, dst_key, grid) = if commit {
        let offset = src % WORD_BYTES;
        (offset, 0, src - offset, dst + bias - offset, src)
    } else {
        let offset = dst % WORD_BYTES;
        let shift = src.wrapping_sub(dst) % WORD_BYTES;
        let src_key = src.checked_sub(offset + shift).filter(|k| same_high(*k, src));
        let Some(src_key) = src_key else {
            return Err(format!(
                "a copy from {src:#x} (offset {offset}, shift {shift}) whose first source \
                 word is below its high word's range"
            ));
        };
        (offset, shift, src_key, dst - offset, dst)
    };
    let (_, words) = executor::vm::history::covering_words(grid, count);
    let last_byte = (grid.wrapping_add(count - 1) % WORD_BYTES) as u8;
    let mut rows = Vec::with_capacity(words as usize);
    let mut remaining = count;
    for k in 0..words {
        let first = k == 0;
        let end = k + 1 == words;
        let start = if first { offset as u8 } else { 0 };
        let last = if end { last_byte } else { 7 };
        let width = u64::from(last - start) + 1;
        remaining -= width;
        let reads_a = first && u64::from(start) + shift < WORD_BYTES;
        let reads_b = !end || u64::from(last) + shift >= WORD_BYTES;
        let row_src = src_key.wrapping_add(WORD_BYTES * k);
        let row_dst = dst_key.wrapping_add(WORD_BYTES * k);
        // Every key a row names, and the next row's, stays inside its high
        // word: the low words are halves and step by 8, the high words ride
        // the chain unchanged.
        let src_top = row_src.wrapping_add(if reads_b || !end { WORD_BYTES } else { 0 });
        let dst_top = row_dst.wrapping_add(if end { 0 } else { WORD_BYTES });
        if !same_high(row_src, src_key)
            || !same_high(src_top, src_key)
            || !same_high(row_dst, dst_key)
            || !same_high(dst_top, dst_key)
        {
            return Err(format!(
                "a copy from {src:#x} to {dst:#x} of {count:#x} bytes crosses a 4 GiB boundary"
            ));
        }
        let mut flags = kind;
        for (on, f) in [
            (first, FLAG_FIRST),
            (end, FLAG_END),
            (reads_a, FLAG_READ_A),
            (reads_b, FLAG_READ_B),
        ] {
            if on {
                flags |= f;
            }
        }
        rows.push(CopyWordRow {
            timestamp,
            src_key: row_src,
            dst_key: row_dst,
            count_decr: remaining as u32,
            start,
            last,
            shift: shift as u8,
            flags,
            ..CopyWordRow::default()
        });
    }
    debug_assert_eq!(remaining, 0, "the windows cover the count");
    if functionality == Functionality::Set
        && rows.iter().any(|r| !same_high(r.src_key, r.dst_key))
    {
        return Err(format!(
            "a memset from {src:#x} whose source and destination words straddle a 4 GiB boundary"
        ));
    }
    Ok(rows)
}

/// The words a call's accesses read and write: the walk's memory, or a
/// test's model of it.
pub(crate) trait CallMemory {
    /// The word at `key` (a multiple of 8): its value and the timestamp of
    /// its last access; `None` when the memory does not hold it.
    fn word(&mut self, key: u64) -> Option<(u64, u64)>;
    /// The call accessed the word at `key` at `timestamp`, leaving `value`;
    /// `touched`, `(first address, length)` inside the word and never empty,
    /// is what the call reads or writes there.
    fn set(&mut self, key: u64, timestamp: u64, value: u64, touched: (u64, u8));
}

/// The bytes a call's range (`len` bytes from `start`) holds in the word at
/// `key`: `(first address, length)`.
fn inside(key: u64, (start, len): (u64, u64)) -> (u64, u8) {
    let lo = u128::from(key).max(u128::from(start));
    let hi = (u128::from(key) + u128::from(WORD_BYTES)).min(u128::from(start) + u128::from(len));
    (lo as u64, hi.saturating_sub(lo) as u8)
}

/// Runs a call's accesses on `memory` in their order, fills `rows`' words
/// ([`call_rows`] of the same call) and returns each access as its MEMW_W
/// row; `None` when `memory` lacks a word (nothing is changed then).
///
/// A copy or commit reads its source words at `T` (`A` on the first row when
/// its window uses it, each `B` that it reads), then a copy writes each
/// destination word at `T + 1`. A memset writes its destination words at `T`,
/// each from the previous one (the first from the word before them), then
/// reads at `T + 1`. A read leaves the word as it was; every access stamps
/// the bytes of the call's range it touches.
pub(crate) fn run_call(
    rows: &mut [CopyWordRow],
    src: u64,
    dst: u64,
    count: u64,
    memory: &mut impl CallMemory,
) -> Result<Option<Vec<MemwWordRow>>, String> {
    let Some(head) = rows.first().copied() else {
        return Ok(Some(Vec::new()));
    };
    if head.zero() {
        return Ok(Some(Vec::new()));
    }
    let functionality = head.functionality();
    let (read_ts, write_ts) = (head.read_timestamp(), head.write_timestamp());
    let (source, destination) = ((src, count), (dst, count));
    // Every word first, so a missing one changes nothing.
    for r in rows.iter() {
        let keys = [
            (r.reads_a(), r.a_key()),
            (r.reads_b(), r.b_key()),
            (r.writes(), r.dst_key),
        ];
        for (on, key) in keys {
            if on && memory.word(key).is_none() {
                return Ok(None);
            }
        }
    }
    let mut out = Vec::with_capacity(3 * rows.len());
    match functionality {
        Functionality::Copy | Functionality::Write => {
            for k in 0..rows.len() {
                let a = if rows[k].reads_a() {
                    read(&mut out, memory, rows[k].a_key(), read_ts, source)?
                } else if k > 0 {
                    rows[k - 1].b
                } else {
                    0
                };
                let b = if rows[k].reads_b() {
                    read(&mut out, memory, rows[k].b_key(), read_ts, source)?
                } else {
                    0
                };
                rows[k].set_words(a, b, 0);
            }
            if functionality == Functionality::Copy {
                for r in rows.iter_mut() {
                    r.old = held(memory, r.dst_key).0;
                    write(&mut out, memory, r.dst_key, write_ts, r.new_word(), destination)?;
                }
            }
        }
        Functionality::Set => {
            // The word before the destination: read at T + 1, untouched by the
            // writes at T, so its value is the one the first row places.
            let a0 = if head.reads_a() {
                held(memory, head.a_key()).0
            } else {
                0
            };
            for k in 0..rows.len() {
                let a = if k == 0 { a0 } else { rows[k - 1].new_word() };
                let key = rows[k].dst_key;
                let old = held(memory, key).0;
                rows[k].set_words(a, 0, old);
                write(&mut out, memory, key, write_ts, rows[k].new_word(), destination)?;
            }
            for k in 0..rows.len() {
                if rows[k].reads_a() {
                    let value = read(&mut out, memory, rows[k].a_key(), read_ts, source)?;
                    if value != rows[k].a {
                        return Err(format!(
                            "a memset's first source word at {:#x} changed under its writes",
                            rows[k].a_key()
                        ));
                    }
                }
                if rows[k].reads_b() {
                    rows[k].b = read(&mut out, memory, rows[k].b_key(), read_ts, source)?;
                    if rows.get(k + 1).is_some_and(|next| next.a != rows[k].b) {
                        return Err(format!(
                            "a memset's word at {:#x} reads back other than it wrote",
                            rows[k].b_key()
                        ));
                    }
                }
            }
        }
    }
    Ok(Some(out))
}

/// The word at `key`, which [`run_call`] checked is held.
fn held(memory: &mut impl CallMemory, key: u64) -> (u64, u64) {
    memory
        .word(key)
        .expect("every word the call touches is held, checked first")
}

/// A read at `ts` of the word at `key`: its MEMW_W row (old = new), the
/// call's bytes in it stamped; its value.
fn read(
    out: &mut Vec<MemwWordRow>,
    memory: &mut impl CallMemory,
    key: u64,
    ts: u64,
    range: (u64, u64),
) -> Result<u64, String> {
    let (value, old_ts) = held(memory, key);
    out.push(MemwWordRow::new(
        MessageKind::Access,
        key,
        ts,
        old_ts,
        value,
        value,
    )?);
    memory.set(key, ts, value, inside(key, range));
    Ok(value)
}

/// A write at `ts` of `new` to the word at `key`: its MEMW_W row (an access:
/// old and new), the call's bytes in it written.
fn write(
    out: &mut Vec<MemwWordRow>,
    memory: &mut impl CallMemory,
    key: u64,
    ts: u64,
    new: u64,
    range: (u64, u64),
) -> Result<(), String> {
    let (old, old_ts) = held(memory, key);
    out.push(MemwWordRow::new(
        MessageKind::Access,
        key,
        ts,
        old_ts,
        old,
        new,
    )?);
    memory.set(key, ts, new, inside(key, range));
    Ok(())
}

// =========================================================================
// Trace generation
// =========================================================================

/// The widths COPY_W traces needed so far in this process (`tables::gpack`).
pub(crate) static WIDTHS: WidthHint = WidthHint::new();

/// The COPY_W trace of `rows`, then zero rows to a power of two of at least
/// four, in `form`.
pub(crate) fn generate_copy_w_trace_as(
    rows: &[CopyWordRow],
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    generate_copy_w_rows_as(rows, rows.len().next_power_of_two().max(4), form)
}

/// A COPY_W table of `num_rows` rows in `form`: `rows` (at most `num_rows`),
/// then zero rows. A row reads its own record alone, so any cut of a call's
/// rows into tables is the same rows: a call's rows chain only through the
/// chain bus.
pub(crate) fn generate_copy_w_rows_as(
    rows: &[CopyWordRow],
    num_rows: usize,
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    debug_assert!(rows.len() <= num_rows);
    generate_main!(form, &WIDTHS, num_rows, cols::NUM_COLUMNS, |table| {
        for (row, r) in rows.iter().enumerate() {
            table.set_u64(row, cols::TIMESTAMP, r.timestamp);
            table.set_half(row, cols::COUNT_DECR[0], r.count_decr as u16);
            table.set_half(row, cols::COUNT_DECR[1], (r.count_decr >> 16) as u16);
            for (lo, hi, key) in [
                (cols::SRC_KEY_LO, cols::SRC_KEY_HI, r.src_key),
                (cols::DST_KEY_LO, cols::DST_KEY_HI, r.dst_key),
            ] {
                table.set_half(row, lo[0], key as u16);
                table.set_half(row, lo[1], (key >> 16) as u16);
                table.set_word(row, hi, (key >> 32) as u32);
            }
            let commit = r.flag(FLAG_WRITE);
            table.set_bool(row, cols::FIRST, r.first());
            table.set_bool(row, cols::END, r.end());
            table.set_fe(row, cols::MU, FE::one());
            table.set_bool(row, cols::ZERO, r.zero());
            table.set_bool(row, cols::IS_SET, r.flag(FLAG_SET));
            table.set_bool(row, cols::IS_WRITE, commit);
            table.set_bool(row, cols::FIRST_WRITE, r.first() && commit);
            table.set_bool(row, cols::READ_A, r.reads_a());
            table.set_bool(row, cols::READ_B, r.reads_b());
            table.set_bool(row, cols::MU_W, r.writes());
            if let Some((start, last)) = r.window() {
                table.set_bool(row, cols::START[usize::from(start)], true);
                table.set_bool(row, cols::LAST[usize::from(last)], true);
                table.set_bool(row, cols::SHIFT[usize::from(r.shift)], true);
            }
            table.set_bytes(row, cols::A[0], &r.a.to_le_bytes());
            table.set_bytes(row, cols::B[0], &r.b.to_le_bytes());
            table.set_bytes(row, cols::OLD[0], &r.old.to_le_bytes());
            let new = r.new_word();
            table.set_word(row, cols::NEW_LO, new as u32);
            table.set_word(row, cols::NEW_HI, (new >> 32) as u32);
            for (j, &c) in cols::COMMIT.iter().enumerate() {
                table.set_bool(row, c, commit && r.in_window(j));
            }
        }
    })
}

// =========================================================================
// Bus interactions
// =========================================================================

/// `IS_HALF` limbs `[lo, hi]` as one 32-bit word, plus `constant`.
fn limbs(halves: [usize; 2], constant: i64) -> Vec<LinearTerm> {
    let mut terms = vec![term(1, halves[0]), term(1 << 16, halves[1])];
    if constant != 0 {
        terms.push(LinearTerm::Constant(constant));
    }
    terms
}

/// `Σ k·one_hot_k`: a one-hot's value.
fn one_hot_value(one_hot: [usize; 8], sign: i64) -> Vec<LinearTerm> {
    (1..8).map(|k| term(sign * k as i64, one_hot[k])).collect()
}

/// The window's width: `MU − ZERO + Σ k·LAST_k − Σ k·START_k`.
fn width_terms() -> Vec<LinearTerm> {
    let mut terms = vec![term(1, cols::MU), term(-1, cols::ZERO)];
    terms.extend(one_hot_value(cols::LAST, 1));
    terms.extend(one_hot_value(cols::START, -1));
    terms
}

/// The count the row starts from: `COUNT_DECR + width`.
fn count_terms() -> Vec<LinearTerm> {
    let mut terms = limbs(cols::COUNT_DECR, 0);
    terms.extend(width_terms());
    terms
}

/// The destination operand: `DST_KEY_LO + START` (x10), less the commit
/// bias on a commit row (x254).
fn dst_operand_terms(commit: bool) -> Vec<LinearTerm> {
    let mut terms = limbs(
        cols::DST_KEY_LO,
        if commit { -COMMIT_KEY_BIAS } else { 0 },
    );
    terms.extend(one_hot_value(cols::START, 1));
    terms
}

/// A MEMW register access (`is_register = 1`, width 2) on the 24-value tuple
/// `[old[8], is_register, address[2], value[8], timestamp[2], w2, w4, w8]`,
/// at the call's timestamp.
fn memw_register(address: u64, old: [BusValue; 2], value: [BusValue; 2]) -> Vec<BusValue> {
    let pad6 = || (0..6).map(|_| BusValue::constant(0));
    let mut tuple = Vec::with_capacity(24);
    tuple.extend(old);
    tuple.extend(pad6());
    tuple.push(BusValue::constant(1)); // is_register
    tuple.push(BusValue::constant(address));
    tuple.push(BusValue::constant(0)); // address high word
    tuple.extend(value);
    tuple.extend(pad6());
    tuple.push(column(cols::TIMESTAMP));
    tuple.push(BusValue::constant(0)); // timestamp high word
    tuple.push(BusValue::constant(1)); // w2
    tuple.push(BusValue::constant(0)); // w4
    tuple.push(BusValue::constant(0)); // w8
    tuple
}

/// A register's address on the MEMW bus: `2·index`.
const fn reg(index: u64) -> u64 {
    2 * index
}

/// `T + IS_SET` (the reads) or `T + 1 − IS_SET` (the write).
fn timestamp(write: bool) -> BusValue {
    BusValue::linear(if write {
        vec![
            term(1, cols::TIMESTAMP),
            term(-1, cols::IS_SET),
            LinearTerm::Constant(1),
        ]
    } else {
        vec![term(1, cols::TIMESTAMP), term(1, cols::IS_SET)]
    })
}

/// A word held as eight byte columns, as two halves.
fn word(bytes: [usize; 8]) -> [BusValue; 2] {
    [word_of_bytes(&bytes[..4]), word_of_bytes(&bytes[4..])]
}

/// The chain's tuple: the timestamp, the destination key, the source key,
/// the count, the functionality, the shift, the carried source word.
/// `forward` is the sender's side (the next row's keys and count, `B`),
/// else the receiver's (its own keys and count, `A`).
fn chain_tuple(forward: bool) -> Vec<BusValue> {
    let step = if forward { WORD_BYTES as i64 } else { 0 };
    let mut tuple = vec![
        column(cols::TIMESTAMP),
        BusValue::linear(limbs(cols::DST_KEY_LO, step)),
        column(cols::DST_KEY_HI),
        BusValue::linear(limbs(cols::SRC_KEY_LO, step)),
        column(cols::SRC_KEY_HI),
        BusValue::linear(if forward {
            limbs(cols::COUNT_DECR, 0)
        } else {
            count_terms()
        }),
        column(cols::IS_SET),
        column(cols::IS_WRITE),
        BusValue::linear(one_hot_value(cols::SHIFT, 1)),
    ];
    let carried = if forward { cols::B } else { cols::A };
    tuple.extend(carried.iter().map(|&c| column(c)));
    tuple
}

/// COPY_W's 37 interactions (module docs).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let first = || Multiplicity::Column(cols::FIRST);
    let first_write = || Multiplicity::Column(cols::FIRST_WRITE);
    let src_key = |step: i64| {
        [
            BusValue::linear(limbs(cols::SRC_KEY_LO, step)),
            column(cols::SRC_KEY_HI),
        ]
    };
    let src_operand = {
        let mut terms = limbs(cols::SRC_KEY_LO, 0);
        terms.extend(one_hot_value(cols::START, 1));
        terms.extend(one_hot_value(cols::SHIFT, 1));
        [BusValue::linear(terms), column(cols::SRC_KEY_HI)]
    };
    let count = [BusValue::linear(count_terms()), BusValue::constant(0)];
    let dst_operand = [
        BusValue::linear(dst_operand_terms(false)),
        column(cols::DST_KEY_HI),
    ];
    let index = [
        BusValue::linear(dst_operand_terms(true)),
        column(cols::DST_KEY_HI),
    ];
    let index_after = {
        let mut terms = dst_operand_terms(true);
        terms.extend(count_terms());
        [BusValue::linear(terms), column(cols::DST_KEY_HI)]
    };
    let mut interactions = vec![
        // The ECALL: its number is a line in the two functionality flags, so
        // a row runs the call the guest made and nothing else.
        BusInteraction::receiver(
            BusId::Ecall,
            first(),
            vec![
                column(cols::TIMESTAMP),
                BusValue::constant(0),
                BusValue::linear(vec![
                    LinearTerm::Constant(MEMCPY_LO32),
                    term(MEMSET_LO32 - MEMCPY_LO32, cols::IS_SET),
                    term(COMMIT_LO32 - MEMCPY_LO32, cols::IS_WRITE),
                ]),
                BusValue::linear(vec![
                    LinearTerm::Constant(MEMCPY_HI32),
                    term(MEMSET_HI32 - MEMCPY_HI32, cols::IS_SET),
                    term(COMMIT_HI32 - MEMCPY_HI32, cols::IS_WRITE),
                ]),
            ],
        ),
        // x11, the source: `SRC_KEY + START + SHIFT`, read back unchanged.
        BusInteraction::sender(
            BusId::Memw,
            first(),
            memw_register(reg(11), src_operand.clone(), src_operand),
        ),
        // x12, the count, read back unchanged.
        BusInteraction::sender(
            BusId::Memw,
            first(),
            memw_register(reg(12), count.clone(), count.clone()),
        ),
        // x10, a copy's or memset's destination `DST_KEY + START`, read back
        // unchanged (the `void*` the call returns).
        BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Diff(cols::FIRST, cols::FIRST_WRITE),
            memw_register(reg(10), dst_operand.clone(), dst_operand),
        ),
        // x254, a commit's index `DST_KEY + START − 8`, advanced by the count.
        BusInteraction::sender(
            BusId::Memw,
            first_write(),
            memw_register(
                super::register::register_base_address(254),
                index,
                index_after,
            ),
        ),
        // x10 on a commit: the file descriptor 1 in, the count out.
        BusInteraction::sender(
            BusId::Memw,
            first_write(),
            memw_register(
                reg(10),
                [BusValue::constant(1), BusValue::constant(0)],
                count,
            ),
        ),
        // The chain, forward and back.
        BusInteraction::sender(
            BusId::CopyWordNext,
            Multiplicity::Diff(cols::MU, cols::END),
            chain_tuple(true),
        ),
        BusInteraction::receiver(
            BusId::CopyWordNext,
            Multiplicity::Diff(cols::MU, cols::FIRST),
            chain_tuple(false),
        ),
    ];
    for limb in cols::COUNT_DECR
        .into_iter()
        .chain(cols::SRC_KEY_LO)
        .chain(cols::DST_KEY_LO)
    {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![column(limb)],
        ));
    }
    let (a, b, old) = (word(cols::A), word(cols::B), word(cols::OLD));
    interactions.extend([
        BusInteraction::sender(
            BusId::MemWordMsg,
            Multiplicity::Column(cols::READ_A),
            access_message(src_key(0), timestamp(false), a.clone(), a),
        ),
        BusInteraction::sender(
            BusId::MemWordMsg,
            Multiplicity::Column(cols::READ_B),
            access_message(
                src_key(WORD_BYTES as i64),
                timestamp(false),
                b.clone(),
                b,
            ),
        ),
        BusInteraction::sender(
            BusId::MemWordMsg,
            Multiplicity::Column(cols::MU_W),
            access_message(
                [
                    BusValue::linear(limbs(cols::DST_KEY_LO, 0)),
                    column(cols::DST_KEY_HI),
                ],
                timestamp(true),
                old,
                [column(cols::NEW_LO), column(cols::NEW_HI)],
            ),
        ),
    ]);
    // A commit's bytes, each its own `(index, byte)` pair: the verifier
    // rebuilds this bus from the public output alone.
    for (j, &c) in cols::COMMIT.iter().enumerate() {
        interactions.push(BusInteraction::sender(
            BusId::Commit,
            Multiplicity::Column(c),
            vec![
                BusValue::linear(limbs(cols::DST_KEY_LO, j as i64 - COMMIT_KEY_BIAS)),
                column(cols::A[j]),
            ],
        ));
    }
    interactions.extend(byte_pair_checks(
        &cols::A,
        Multiplicity::Column(cols::READ_A),
    ));
    interactions.extend(byte_pair_checks(
        &cols::B,
        Multiplicity::Column(cols::READ_B),
    ));
    interactions.extend(byte_pair_checks(
        &cols::OLD,
        Multiplicity::Column(cols::MU_W),
    ));
    interactions
}

// =========================================================================
// Constraints
// =========================================================================

type Expr<B> = <B as ConstraintBuilder<GoldilocksField, GoldilocksExtension>>::Expr;

/// The sum of `cols`.
fn sum<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    cols: &[usize],
) -> Expr<B> {
    cols.iter().fold(b.zero(), |acc, &c| acc + b.main(0, c))
}

/// `Σ k·cols_k`.
fn weighted<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    cols: [usize; 8],
) -> Expr<B> {
    (1..8).fold(b.zero(), |acc, k| {
        acc + b.const_base(k as u64) * b.main(0, cols[k])
    })
}

/// Window byte `j`'s mask, `Σ_(i ≤ j) START_i − Σ_(i < j) LAST_i`: 1 inside
/// the window, 0 outside, once `START` and `LAST` are one-hot and in order.
fn mask<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(b: &B, j: usize) -> Expr<B> {
    sum(b, &cols::START[..=j]) - sum(b, &cols::LAST[..j])
}

/// `Σ one_hot_a·one_hot_b` over the pairs `(i, s)` with `keep(i + s)`.
fn pairs<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    x: [usize; 8],
    y: [usize; 8],
    keep: impl Fn(usize) -> bool,
) -> Expr<B> {
    let mut acc = b.zero();
    for (i, &xi) in x.iter().enumerate() {
        for (s, &ys) in y.iter().enumerate() {
            if keep(i + s) {
                acc = acc + b.main(0, xi) * b.main(0, ys);
            }
        }
    }
    acc
}

/// `[lo, hi]` limbs as one word.
fn limbs_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    halves: [usize; 2],
) -> Expr<B> {
    b.main(0, halves[0]) + b.const_base(1 << 16) * b.main(0, halves[1])
}

/// COPY_W's 59 constraints, degree 3:
/// - idx 0–5: `FIRST`, `END`, `MU`, `ZERO`, `IS_SET`, `IS_WRITE` bits;
/// - idx 6: `FIRST` or `END` only on an active row; 7, 8: one functionality,
///   only on an active row; 9: `FIRST_WRITE = FIRST·IS_WRITE`; 10, 11: a
///   zero-length row is first and last;
/// - idx 12–20, 21–29, 30–38: `START`, `LAST`, `SHIFT` each eight bits summing
///   to `MU − ZERO`;
/// - idx 39: no `LAST` before `START`; 40: a row after the first starts at
///   byte 0; 41: a row before the last ends at byte 7; 42: the last row leaves
///   no count;
/// - idx 43: `READ_A = FIRST·[START + SHIFT < 8]`; 44: `READ_B = (MU − END) +
///   END·[LAST + SHIFT ≥ 8]`; 45: `MU_W = (MU − ZERO)·(1 − IS_WRITE)`;
/// - idx 46–53: `COMMIT_j = IS_WRITE·mask_j`;
/// - idx 54, 55: the new word's halves, the old word with the window placed;
/// - idx 56: a memset or commit does not shift; 57, 58: a memset's
///   destination key is its source key plus 8.
#[derive(Clone, Copy)]
pub struct CopyWordConstraints;

/// How many constraints [`CopyWordConstraints`] emits.
pub const NUM_CONSTRAINTS: usize = 59;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for CopyWordConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        for (idx, c) in [
            cols::FIRST,
            cols::END,
            cols::MU,
            cols::ZERO,
            cols::IS_SET,
            cols::IS_WRITE,
        ]
        .into_iter()
        .enumerate()
        {
            emit_is_bit(b, idx, c, None);
        }
        let one = b.one();
        let (first, end, mu, zero) = (b.main(0, cols::FIRST), b.main(0, cols::END), b.main(0, cols::MU), b.main(0, cols::ZERO));
        let (is_set, is_write) = (b.main(0, cols::IS_SET), b.main(0, cols::IS_WRITE));
        b.emit_base(
            6,
            (first.clone() + end.clone()) * (one.clone() - mu.clone()),
        );
        b.emit_base(7, is_set.clone() * is_write.clone());
        b.emit_base(
            8,
            (is_set.clone() + is_write.clone()) * (one.clone() - mu.clone()),
        );
        b.emit_base(
            9,
            b.main(0, cols::FIRST_WRITE) - first.clone() * is_write.clone(),
        );
        b.emit_base(10, zero.clone() * (one.clone() - first.clone()));
        b.emit_base(11, zero.clone() * (one.clone() - end.clone()));

        let window = mu.clone() - zero.clone();
        for (base, one_hot) in [(12, cols::START), (21, cols::LAST), (30, cols::SHIFT)] {
            for (k, &c) in one_hot.iter().enumerate() {
                let root = bit(b, b.main(0, c));
                b.emit_base(base + k, root);
            }
            let root = sum(b, &one_hot) - window.clone();
            b.emit_base(base + 8, root);
        }

        // No LAST before START: Σ_j START_j · Σ_(i < j) LAST_i.
        let mut order = b.zero();
        for j in 1..8 {
            order = order + b.main(0, cols::START[j]) * sum(b, &cols::LAST[..j]);
        }
        b.emit_base(39, order);
        b.emit_base(
            40,
            (mu.clone() - first.clone()) * (one.clone() - b.main(0, cols::START[0])),
        );
        b.emit_base(
            41,
            (mu.clone() - end.clone()) * (one.clone() - b.main(0, cols::LAST[7])),
        );
        b.emit_base(
            42,
            end.clone() * (b.main(0, cols::COUNT_DECR[0]) + b.main(0, cols::COUNT_DECR[1])),
        );

        let need_a = pairs(b, cols::START, cols::SHIFT, |k| k < 8);
        b.emit_base(43, b.main(0, cols::READ_A) - first.clone() * need_a);
        let need_b = pairs(b, cols::LAST, cols::SHIFT, |k| k >= 8);
        b.emit_base(
            44,
            b.main(0, cols::READ_B) - (mu.clone() - end.clone()) - end.clone() * need_b,
        );
        b.emit_base(
            45,
            b.main(0, cols::MU_W) - window.clone() * (one.clone() - is_write.clone()),
        );

        for (j, &c) in cols::COMMIT.iter().enumerate() {
            let root = b.main(0, c) - is_write.clone() * mask(b, j);
            b.emit_base(46 + j, root);
        }

        // The new word: byte j is placed inside the window, kept outside it.
        for (idx, half, bytes) in [(54, cols::NEW_LO, 0..4), (55, cols::NEW_HI, 4..8)] {
            let mut acc = b.zero();
            for j in bytes.clone() {
                let mask_j = mask(b, j);
                let byte = mask_j.clone() * placed(b, cols::SHIFT, &cols::AB, j)
                    + (one.clone() - mask_j) * b.main(0, cols::OLD[j]);
                acc = acc + b.const_base(1 << (8 * (j - bytes.start))) * byte;
            }
            b.emit_base(idx, b.main(0, half) - acc);
        }

        b.emit_base(56, (is_set.clone() + is_write) * weighted(b, cols::SHIFT));
        let gap = b.const_base(DMA_MEMSET_GAP);
        b.emit_base(
            57,
            is_set.clone()
                * (limbs_expr(b, cols::DST_KEY_LO) - limbs_expr(b, cols::SRC_KEY_LO) - gap),
        );
        b.emit_base(
            58,
            is_set * (b.main(0, cols::DST_KEY_HI) - b.main(0, cols::SRC_KEY_HI)),
        );
    }

    fn max_degree(&self) -> usize {
        3
    }
}
