//! KECCAK_W: the KECCAK core chip's Keccak-256 of a buffer (`keccak256`) under
//! [`MemoryFormat::Words`](super::fused_memory::MemoryFormat::Words), where RAM
//! is checked by aligned 8-byte words ([`super::word_memory`]).
//!
//! One row a permutation, as KECCAK's sponge mode ([`super::keccak`]): row `j`
//! of a call absorbs block `j` (136 bytes from `src + 136·j`) into the state
//! the previous row's permutation ended in, asks the round chip for the
//! permutation on the `Keccak` bus, and the last row writes the digest (the
//! output's first 32 bytes) to `out`. The rows are linked on
//! [`BusId::KeccakSponge`]. The mask `s` (`FULL`, `PART`), the padding, the
//! absorb XORs, the count and the zero first state are KECCAK's, unchanged.
//! KECCAK's permute mode is not here: under the word format the walk refuses a
//! permute call.
//!
//! ## The block's words
//!
//! A word token covers eight bytes and takes one access a timestamp, so a
//! block is read as the aligned words it covers, `W_0 … W_17` at
//! `SRC_KEY + 8w` (`SRC_KEY = SRC_LO − o`, `o = src mod 8` one-hot in `OFF`,
//! fixed for the call), and block byte `i` is byte `i + o` of `W` (`MSG_i =
//! s_i · Σ_k OFF_k · W[i + k]`). Block `j`'s `W_17` is block `j + 1`'s `W_0`
//! (136 is a multiple of 8): the link carries its eight bytes, so a word is
//! read once a call. A row reads `W_w` (`READ_w`) exactly when its window
//! holds an input byte there: `W_0` when the block holds one (only on the
//! first row, or when `o = 0`; otherwise it is the carried word), `W_w` for
//! `1 ≤ w ≤ 16` when byte `8w − o` is input, `W_17` when `o > 0` and byte
//! `136 − o` is input — the input's covering words, each once.
//!
//! ## The digest
//!
//! The 32 bytes `D` (the output's first four lanes) go to `out`, at offset `q`
//! (one-hot `Q`) in its word: the words at `OUT_KEY + 8v` (`OUT_KEY = OUT_LO −
//! q`), `v = 0 … 4`. Words 1–3 are whole (write messages); word 0 keeps its
//! bytes below `q` and word 4 (written only when `q > 0`, `W4`) those from
//! `q` on: read-modify-writes whose old bytes (`OLD0`, `OLD4`) are checked.
//! Each new word is a pair of halves (`NEW`) fixed by a degree-2 identity to
//! the old word with `D`'s bytes placed; `D`'s bytes are checked here too.
//!
//! ## What pins what
//!
//! - Only PAGE_W keys hold tokens, so the keys a row reads and writes are
//!   aligned and `OFF`, `Q` are the true offsets; the keys' low words are
//!   `IS_HALF` limbs, their high words the operand registers' (`SRC_HI`,
//!   `OUT_HI` ride the link).
//! - `OFF` rides the link (`Σ k·OFF_k`) with the carried word: a row cannot
//!   change the offset mid-call, or take a `W_0` other than the previous
//!   `W_17`.
//! - `READ_w` is fixed by the mask and the offset: a word a row places input
//!   bytes from is always read.
//!
//! ## Columns (1,031)
//!
//! KECCAK's sponge columns (timestamp, input and output states, `MU`, `FIRST`,
//! `LAST`, `COUNT`, `SRC`, `OUT`, `SRC_END`, `OUT_END`, `PREV`, `MSG`,
//! `FULL`, `PART`), then `OFF[8]`, `W[144]`, `READ[18]`, `SRC_KEY[2]`,
//! `Q[8]`, `OUT_KEY[2]`, `OLD0[8]`, `OLD4[8]`, `NEW[10]`, `W4`.
//!
//! ## Interactions (271)
//!
//! The ECALL, x10–x12, the link both ways, the round chip both ways, 136
//! absorb XORs, eight `IS_HALF`, 18 word reads and their 72 `ARE_BYTES`
//! pairs, five digest messages and 24 `ARE_BYTES` pairs (the edges' old
//! words, `D`).

use executor::vm::instruction::execution::KECCAK256_SYSCALL_NUMBER;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity};
use stark::trace::TraceTable;

use super::bitwise::{BitwiseOperation, BitwiseOperationType};
use super::gpack::{TraceForm, WidthHint, generate_main};
use super::keccak::{KeccakOperation, SpongeRow};
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};
use super::word_memory::{
    access_message, byte_pair_checks, column, term, word_of_bytes, write_message,
};
use crate::constraints::templates::emit_is_bit;

/// Bytes of a block (the rate), lanes of a block.
pub const RATE: usize = 136;
pub const RATE_LANES: usize = 17;
/// The words a block covers: 17, and one more when it is not aligned.
pub const BLOCK_WORDS: usize = 18;
/// The digest's bytes, and the words it covers.
pub const DIGEST: usize = 32;
pub const DIGEST_WORDS: usize = 5;

/// Column definitions for the KECCAK_W table.
pub mod cols {
    use super::{BLOCK_WORDS, DIGEST_WORDS, RATE, RATE_LANES};

    /// `(T, j)`: the call's timestamp, the block index.
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;
    /// The permutation's input and output, 200 bytes each (lane-major).
    pub const INPUT_STATE: usize = 2;
    pub const OUTPUT_STATE: usize = INPUT_STATE + 200;
    /// A row (one permutation).
    pub const MU: usize = OUTPUT_STATE + 200;
    pub const FIRST: usize = MU + 1;
    pub const LAST: usize = FIRST + 1;
    /// Input bytes left before this row's block.
    pub const COUNT: usize = LAST + 1;
    /// The block's address and the digest's, low and high words.
    pub const SRC_LO: usize = COUNT + 1;
    pub const SRC_HI: usize = SRC_LO + 1;
    pub const OUT_LO: usize = SRC_HI + 1;
    pub const OUT_HI: usize = OUT_LO + 1;
    /// `SRC_LO + 135` and `OUT_LO + 31`, two halves each.
    pub const SRC_END: usize = OUT_HI + 1;
    pub const OUT_END: usize = SRC_END + 2;
    /// The previous permutation's rate, the block's bytes (0 past the input).
    pub const PREV: usize = OUT_END + 2;
    pub const MSG: usize = PREV + RATE;
    /// The mask: lane `k` all input (`FULL`), byte `b < 7` of a lane the
    /// input ends inside (`PART`).
    pub const FULL: usize = MSG + RATE;
    pub const PART: usize = FULL + RATE_LANES;
    /// The block's offset in its first word, one-hot.
    pub const OFF: usize = PART + RATE_LANES * 7;
    /// The covering words' bytes, `W_w` at `W + 8w`.
    pub const W: usize = OFF + 8;
    /// Whether the row reads `W_w`.
    pub const READ: usize = W + 8 * BLOCK_WORDS;
    /// `SRC_LO − o`, two halves.
    pub const SRC_KEY: usize = READ + BLOCK_WORDS;
    /// The digest's offset in its first word, one-hot.
    pub const Q: usize = SRC_KEY + 2;
    /// `OUT_LO − q`, two halves.
    pub const OUT_KEY: usize = Q + 8;
    /// The digest's edge words before the write: word 0's, word 4's bytes.
    pub const OLD0: usize = OUT_KEY + 2;
    pub const OLD4: usize = OLD0 + 8;
    /// The digest's words after the write, `(lo, hi)` at `NEW + 2v`.
    pub const NEW: usize = OLD4 + 8;
    /// Whether the fifth digest word is written: `LAST·(1 − Q_0)`.
    pub const W4: usize = NEW + 2 * DIGEST_WORDS;
    pub const NUM_COLUMNS: usize = W4 + 1;

    #[inline]
    pub const fn input_byte(i: usize) -> usize {
        INPUT_STATE + i
    }
    #[inline]
    pub const fn output_byte(i: usize) -> usize {
        OUTPUT_STATE + i
    }
    #[inline]
    pub const fn prev(i: usize) -> usize {
        PREV + i
    }
    #[inline]
    pub const fn msg(i: usize) -> usize {
        MSG + i
    }
    #[inline]
    pub const fn full(lane: usize) -> usize {
        FULL + lane
    }
    #[inline]
    pub const fn part(lane: usize, byte: usize) -> usize {
        PART + lane * 7 + byte
    }
    #[inline]
    pub const fn off(k: usize) -> usize {
        OFF + k
    }
    /// Byte `b` of covering word `w` (`w·8 + b` of the flat `W`).
    #[inline]
    pub const fn w(i: usize) -> usize {
        W + i
    }
    #[inline]
    pub const fn read(w: usize) -> usize {
        READ + w
    }
    #[inline]
    pub const fn q(k: usize) -> usize {
        Q + k
    }
    #[inline]
    pub const fn new(v: usize, half: usize) -> usize {
        NEW + 2 * v + half
    }
}

/// The name the AIR set gives a KECCAK_W table: `KECCAK_W[i]`.
pub const NAME: &str = "KECCAK_W";

// =========================================================================
// The record
// =========================================================================

/// What a KECCAK_W row holds beyond KECCAK's sponge row: the block's words
/// as the walk read them, and on the last row the digest's edge words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpongeWords {
    /// `src mod 8`, the call's.
    pub offset: u8,
    /// `W_0 … W_17`: the words at `src − offset + 136·j + 8w` (a word the row
    /// neither reads nor carries is any value; the walk's are the memory's or
    /// zero).
    pub words: [u64; BLOCK_WORDS],
    /// On the last row: the digest's offset and its edge words before it.
    pub digest: Option<DigestWords>,
}

/// The digest's offset `q = out mod 8` and its edge words before the write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DigestWords {
    pub q: u8,
    pub old0: u64,
    pub old4: u64,
}

/// Which covering words a row reads: `W_0` when the block holds an input
/// byte and the row is the call's first or the block is aligned (else it is
/// the previous row's `W_17`, carried); `W_w` (`1 ≤ w ≤ 16`) when block byte
/// `8w − o` is input; `W_17` when `o > 0` and byte `136 − o` is.
pub fn reads(mask: &[bool; RATE], offset: u8, first: bool) -> [bool; BLOCK_WORDS] {
    let o = usize::from(offset);
    std::array::from_fn(|w| match w {
        0 => mask[0] && (first || o == 0),
        17 => o > 0 && mask[RATE - o],
        _ => mask[8 * w - o],
    })
}

/// The digest's five words after the write: `D` (32 bytes) at offset `q`,
/// word 0 keeping `old0`'s bytes below `q`, word 4 `old4`'s from `q` on
/// (and, at `q = 0`, all of them: not written).
pub fn digest_words(digest: &[u8; DIGEST], q: u8, old0: u64, old4: u64) -> [u64; DIGEST_WORDS] {
    let q = usize::from(q);
    std::array::from_fn(|v| {
        let old = match v {
            0 => old0,
            4 => old4,
            _ => 0,
        };
        (0..8).fold(0u64, |word, j| {
            let at = 8 * v + j;
            let byte = if (q..q + DIGEST).contains(&at) {
                digest[at - q]
            } else {
                (old >> (8 * j)) as u8
            };
            word | u64::from(byte) << (8 * j)
        })
    })
}

/// The digest's bytes: the output's first four lanes.
pub fn digest_of(output: &[u64; 25]) -> [u8; DIGEST] {
    std::array::from_fn(|i| (output[i / 8] >> (8 * (i % 8))) as u8)
}

/// The BITWISE lookups a KECCAK_W row sends beyond KECCAK's sponge row's
/// (the absorb XORs and the end halves): `IS_HALF` on its keys, `ARE_BYTES`
/// on each word it reads, and on the last row on the edges' old words (word
/// 4's only when written) and on the digest.
pub fn extra_bitwise_ops(
    kop: &KeccakOperation,
    sponge: &SpongeRow,
    words: &SpongeWords,
    emit: &mut impl FnMut(BitwiseOperation),
) {
    let half = |value: u64, emit: &mut dyn FnMut(BitwiseOperation)| {
        for h in [value & 0xFFFF, (value >> 16) & 0xFFFF] {
            emit(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                h as u8,
                (h >> 8) as u8,
            ));
        }
    };
    let pairs = |bytes: &[u8], emit: &mut dyn FnMut(BitwiseOperation)| {
        for p in bytes.chunks_exact(2) {
            emit(BitwiseOperation::byte_op(
                BitwiseOperationType::AreBytes,
                p[0],
                p[1],
            ));
        }
    };
    let o = u64::from(words.offset);
    half((sponge.src & 0xFFFF_FFFF).wrapping_sub(o), &mut *emit);
    for (w, read) in reads(&sponge.mask, words.offset, sponge.first)
        .into_iter()
        .enumerate()
    {
        if read {
            pairs(&words.words[w].to_le_bytes(), &mut *emit);
        }
    }
    if let Some(d) = &words.digest {
        half(
            (sponge.out & 0xFFFF_FFFF).wrapping_sub(u64::from(d.q)),
            &mut *emit,
        );
        pairs(&d.old0.to_le_bytes(), &mut *emit);
        if d.q != 0 {
            pairs(&d.old4.to_le_bytes(), &mut *emit);
        }
        pairs(&digest_of(&kop.output), &mut *emit);
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// The widths KECCAK_W traces needed so far in this process (`tables::gpack`).
pub(crate) static WIDTHS: WidthHint = WidthHint::new();

/// Whether [`generate_keccak_w_rows_as`] in `form` writes its rows packed
/// (`tables::gpack`).
pub(crate) fn rows_written_packed(form: TraceForm) -> bool {
    matches!(
        super::gpack::Plan::new(form, &WIDTHS, cols::NUM_COLUMNS),
        super::gpack::Plan::Write(_)
    )
}

/// The KECCAK_W trace of `ops` (sponge rows with their words), padded to a
/// power of two of at least four.
pub fn generate_keccak_w_trace_as(
    ops: &[KeccakOperation],
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    generate_keccak_w_rows_as(ops, ops.len().next_power_of_two().max(4), form)
}

/// A KECCAK_W table of `num_rows` rows in `form`: `ops` (at most `num_rows`,
/// each a sponge row with its words), then zero rows. A row reads its own op
/// alone, so any cut of the rows is the same rows.
pub fn generate_keccak_w_rows_as(
    ops: &[KeccakOperation],
    num_rows: usize,
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    debug_assert!(ops.len() <= num_rows);
    generate_main!(form, &WIDTHS, num_rows, cols::NUM_COLUMNS, |table| {
        for (row, op) in ops.iter().enumerate() {
            let sponge = op.sponge.as_ref().expect("a KECCAK_W row is a sponge row");
            let words = sponge
                .words
                .as_deref()
                .expect("a KECCAK_W row holds its words");
            table.set_dword_wl(row, cols::TIMESTAMP_0, op.timestamp);
            for lane in 0..25 {
                table.set_dword_bl(row, cols::input_byte(8 * lane), op.input[lane]);
                table.set_dword_bl(row, cols::output_byte(8 * lane), op.output[lane]);
            }
            table.set_fe(row, cols::MU, FE::one());
            table.set_bool(row, cols::FIRST, sponge.first);
            table.set_bool(row, cols::LAST, sponge.last);
            table.set_u64(row, cols::COUNT, sponge.count);
            table.set_dword_wl(row, cols::SRC_LO, sponge.src);
            table.set_dword_wl(row, cols::OUT_LO, sponge.out);
            let src_end = (sponge.src & 0xFFFF_FFFF) + 135;
            table.set_u64(row, cols::SRC_END, src_end & 0xFFFF);
            table.set_u64(row, cols::SRC_END + 1, src_end >> 16);
            table.set_bytes(row, cols::PREV, &op.sponge_prev(sponge));
            table.set_bytes(row, cols::MSG, &sponge.msg);
            for lane in 0..RATE_LANES {
                table.set_bool(row, cols::full(lane), sponge.full(lane));
                for byte in 0..7 {
                    table.set_bool(row, cols::part(lane, byte), sponge.part(lane, byte));
                }
            }
            let o = words.offset;
            table.set_bool(row, cols::off(usize::from(o)), true);
            for (w, word) in words.words.iter().enumerate() {
                table.set_bytes(row, cols::w(8 * w), &word.to_le_bytes());
            }
            for (w, read) in reads(&sponge.mask, o, sponge.first).into_iter().enumerate() {
                table.set_bool(row, cols::read(w), read);
            }
            let key = (sponge.src & 0xFFFF_FFFF).wrapping_sub(u64::from(o));
            table.set_half(row, cols::SRC_KEY, key as u16);
            table.set_half(row, cols::SRC_KEY + 1, (key >> 16) as u16);
            if let Some(d) = &words.digest {
                let out_end = (sponge.out & 0xFFFF_FFFF) + 31;
                table.set_u64(row, cols::OUT_END, out_end & 0xFFFF);
                table.set_u64(row, cols::OUT_END + 1, out_end >> 16);
                table.set_bool(row, cols::q(usize::from(d.q)), true);
                let key = (sponge.out & 0xFFFF_FFFF).wrapping_sub(u64::from(d.q));
                table.set_half(row, cols::OUT_KEY, key as u16);
                table.set_half(row, cols::OUT_KEY + 1, (key >> 16) as u16);
                table.set_bytes(row, cols::OLD0, &d.old0.to_le_bytes());
                table.set_bytes(row, cols::OLD4, &d.old4.to_le_bytes());
                let new = digest_words(&digest_of(&op.output), d.q, d.old0, d.old4);
                for (v, word) in new.iter().enumerate() {
                    table.set_word(row, cols::new(v, 0), *word as u32);
                    table.set_word(row, cols::new(v, 1), (*word >> 32) as u32);
                }
                table.set_bool(row, cols::W4, d.q != 0);
            }
        }
    })
}

// =========================================================================
// Bus interactions
// =========================================================================

/// `column + constant`.
fn offset(column: usize, constant: i64) -> BusValue {
    BusValue::linear(vec![term(1, column), LinearTerm::Constant(constant)])
}

/// The mask `s_i` as linear terms at `coefficient` (`s_{8k+b} = FULL_k +
/// PART_{k,b}` for `b < 7`, `s_{8k+7} = FULL_k`; `s_{−1} = 1`).
fn mask_terms(i: isize, coefficient: i64) -> Vec<LinearTerm> {
    if i < 0 {
        return vec![LinearTerm::Constant(coefficient)];
    }
    let i = i as usize;
    let (lane, byte) = (i / 8, i % 8);
    let mut terms = vec![term(coefficient, cols::full(lane))];
    if byte < 7 {
        terms.push(term(coefficient, cols::part(lane, byte)));
    }
    terms
}

/// The padded message byte `m'_i = MSG_i + (s_{i−1} − s_i) + [i = 135]·0x80·LAST`.
fn padded_byte(i: usize) -> BusValue {
    let mut terms = vec![term(1, cols::msg(i))];
    terms.extend(mask_terms(i as isize - 1, 1));
    terms.extend(mask_terms(i as isize, -1));
    if i == RATE - 1 {
        terms.push(term(0x80, cols::LAST));
    }
    BusValue::linear(terms)
}

/// A register read of `x{reg}` at `(T, 0)` binding `(lo, hi)`.
fn register_read(reg: u64, lo: BusValue, hi: BusValue) -> Vec<BusValue> {
    let zeros = || std::iter::repeat_n(BusValue::constant(0), 6);
    let mut v = Vec::with_capacity(24);
    v.extend([lo.clone(), hi.clone()]);
    v.extend(zeros());
    v.push(BusValue::constant(1)); // is_register
    v.push(BusValue::constant(2 * reg));
    v.push(BusValue::constant(0));
    v.extend([lo, hi]);
    v.extend(zeros());
    v.push(column(cols::TIMESTAMP_0));
    v.push(BusValue::constant(0));
    v.push(BusValue::constant(1)); // w2: a register
    v.push(BusValue::constant(0));
    v.push(BusValue::constant(0));
    v
}

/// `Σ k·one_hot_k`.
fn one_hot_value(start: usize) -> BusValue {
    BusValue::linear((1..8).map(|k| term(k as i64, start + k)).collect())
}

/// Two `IS_HALF` limbs as one 32-bit word plus `constant`.
fn limbs(start: usize, constant: i64) -> BusValue {
    let mut terms = vec![term(1, start), term(1 << 16, start + 1)];
    if constant != 0 {
        terms.push(LinearTerm::Constant(constant));
    }
    BusValue::linear(terms)
}

/// Eight byte columns from `start` as a word's two halves.
fn word(start: usize) -> [BusValue; 2] {
    let b: Vec<usize> = (start..start + 8).collect();
    [word_of_bytes(&b[..4]), word_of_bytes(&b[4..])]
}

/// The memory timestamp `T + plus`.
fn at(plus: i64) -> BusValue {
    if plus == 0 {
        column(cols::TIMESTAMP_0)
    } else {
        offset(cols::TIMESTAMP_0, plus)
    }
}

/// KECCAK_W's 271 interactions (module docs).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let first = || Multiplicity::Column(cols::FIRST);
    let last = || Multiplicity::Column(cols::LAST);
    let mu = || Multiplicity::Column(cols::MU);
    let mut out = Vec::with_capacity(271);

    out.push(BusInteraction::receiver(
        BusId::Ecall,
        first(),
        vec![
            column(cols::TIMESTAMP_0),
            BusValue::constant(0),
            BusValue::constant(KECCAK256_SYSCALL_NUMBER & 0xFFFF_FFFF),
            BusValue::constant(KECCAK256_SYSCALL_NUMBER >> 32),
        ],
    ));
    for (reg, lo, hi) in [
        (10, column(cols::SRC_LO), column(cols::SRC_HI)),
        (11, column(cols::COUNT), BusValue::constant(0)),
        (12, column(cols::OUT_LO), column(cols::OUT_HI)),
    ] {
        out.push(BusInteraction::sender(
            BusId::Memw,
            first(),
            register_read(reg, lo, hi),
        ));
    }

    // The link: (T, j + 1, COUNT − 136, SRC_END + 1, SRC_HI, OUT_LO, OUT_HI,
    // OUTPUT_STATE, the offset, W_17) to the next row, which receives its own
    // (T, j, COUNT, SRC_LO, SRC_HI, OUT_LO, OUT_HI, PREV, capacity, the
    // offset, W_0).
    let mut forth = vec![
        column(cols::TIMESTAMP_0),
        offset(cols::TIMESTAMP_1, 1),
        offset(cols::COUNT, -(RATE as i64)),
        BusValue::linear(vec![
            term(1, cols::SRC_END),
            term(1 << 16, cols::SRC_END + 1),
            LinearTerm::Constant(1),
        ]),
        column(cols::SRC_HI),
        column(cols::OUT_LO),
        column(cols::OUT_HI),
    ];
    forth.extend((0..200).map(|i| column(cols::output_byte(i))));
    forth.push(one_hot_value(cols::OFF));
    forth.extend((0..8).map(|b| column(cols::w(8 * 17 + b))));
    out.push(BusInteraction::sender(
        BusId::KeccakSponge,
        Multiplicity::Diff(cols::MU, cols::LAST),
        forth,
    ));
    let mut back = vec![
        column(cols::TIMESTAMP_0),
        column(cols::TIMESTAMP_1),
        column(cols::COUNT),
        column(cols::SRC_LO),
        column(cols::SRC_HI),
        column(cols::OUT_LO),
        column(cols::OUT_HI),
    ];
    back.extend((0..RATE).map(|i| column(cols::prev(i))));
    back.extend((RATE..200).map(|i| column(cols::input_byte(i))));
    back.push(one_hot_value(cols::OFF));
    back.extend((0..8).map(|b| column(cols::w(b))));
    out.push(BusInteraction::receiver(
        BusId::KeccakSponge,
        Multiplicity::Diff(cols::MU, cols::FIRST),
        back,
    ));

    // The round chip: (T, j, 0, input) out, (T, j, 24, output) back.
    for (round, start, send) in [
        (0u64, cols::INPUT_STATE, true),
        (24, cols::OUTPUT_STATE, false),
    ] {
        let mut v = vec![
            column(cols::TIMESTAMP_0),
            column(cols::TIMESTAMP_1),
            BusValue::constant(round),
        ];
        // KECCAK's order: x major, then y, then the lane's bytes.
        for x in 0..5 {
            for y in 0..5 {
                v.extend((0..8).map(|b| column(start + 8 * (x + 5 * y) + b)));
            }
        }
        out.push(if send {
            BusInteraction::sender(BusId::Keccak, mu(), v)
        } else {
            BusInteraction::receiver(BusId::Keccak, mu(), v)
        });
    }

    // The absorb: INPUT_i = PREV_i XOR m'_i over the rate.
    for i in 0..RATE {
        out.push(BusInteraction::sender(
            BusId::ByteAlu,
            mu(),
            vec![
                BusValue::constant(alu_op::XOR as u64),
                column(cols::prev(i)),
                padded_byte(i),
                column(cols::input_byte(i)),
            ],
        ));
    }

    for (start, m) in [
        (cols::SRC_END, mu()),
        (cols::OUT_END, last()),
        (cols::SRC_KEY, mu()),
        (cols::OUT_KEY, last()),
    ] {
        for half in 0..2 {
            out.push(BusInteraction::sender(
                BusId::IsHalfword,
                m.clone(),
                vec![column(start + half)],
            ));
        }
    }

    // The block's words at T.
    for w in 0..BLOCK_WORDS {
        let address = [limbs(cols::SRC_KEY, 8 * w as i64), column(cols::SRC_HI)];
        let bytes = word(cols::w(8 * w));
        out.push(BusInteraction::sender(
            BusId::MemWordMsg,
            Multiplicity::Column(cols::read(w)),
            access_message(address, at(0), bytes.clone(), bytes),
        ));
    }
    for w in 0..BLOCK_WORDS {
        let bytes: Vec<usize> = (0..8).map(|b| cols::w(8 * w + b)).collect();
        out.extend(byte_pair_checks(
            &bytes,
            Multiplicity::Column(cols::read(w)),
        ));
    }

    // The digest's words at T + 1.
    let out_address = |v: usize| [limbs(cols::OUT_KEY, 8 * v as i64), column(cols::OUT_HI)];
    let new = |v: usize| [column(cols::new(v, 0)), column(cols::new(v, 1))];
    out.push(BusInteraction::sender(
        BusId::MemWordMsg,
        last(),
        access_message(out_address(0), at(1), word(cols::OLD0), new(0)),
    ));
    for v in 1..4 {
        out.push(BusInteraction::sender(
            BusId::MemWordMsg,
            last(),
            write_message(out_address(v), at(1), new(v)),
        ));
    }
    out.push(BusInteraction::sender(
        BusId::MemWordMsg,
        Multiplicity::Column(cols::W4),
        access_message(out_address(4), at(1), word(cols::OLD4), new(4)),
    ));
    let old0: Vec<usize> = (0..8).map(|b| cols::OLD0 + b).collect();
    let old4: Vec<usize> = (0..8).map(|b| cols::OLD4 + b).collect();
    let digest: Vec<usize> = (0..DIGEST).map(cols::output_byte).collect();
    out.extend(byte_pair_checks(&old0, last()));
    out.extend(byte_pair_checks(&old4, Multiplicity::Column(cols::W4)));
    out.extend(byte_pair_checks(&digest, last()));
    out
}

// =========================================================================
// Constraints
// =========================================================================

type Expr<B> = <B as ConstraintBuilder<GoldilocksField, GoldilocksExtension>>::Expr;

/// How many constraints [`KeccakWordConstraints`] emits.
pub const NUM_CONSTRAINTS: usize = 4 // modes
    + RATE_LANES * 8 // FULL, PART bits
    + RATE_LANES * 7 // FULL · PART
    + (RATE - 1) // prefix
    + 4 // no read off a row, whole unless last, last not whole, count
    + 1 + RATE + (200 - RATE) // first row: block 0, zero state
    + 2 // the ends' halves
    + 9 + 9 // OFF, Q one-hot
    + 2 // the keys
    + BLOCK_WORDS // reads
    + RATE // placement
    + 1 // W4
    + 2 * DIGEST_WORDS; // the digest's halves

/// The first index of the read rules (`READ_w`, `w = 0 … 17`).
pub const READ_RULES: usize = NUM_CONSTRAINTS - BLOCK_WORDS - RATE - 1 - 2 * DIGEST_WORDS;
/// The first index of the placement (`MSG_i`, `i = 0 … 135`).
pub const PLACEMENT: usize = READ_RULES + BLOCK_WORDS;
/// The index of `W4`'s rule.
pub const W4_RULE: usize = PLACEMENT + RATE;
/// The first index of the digest's halves (`NEW` word `v` half `h` at
/// `DIGEST_HALVES + 2v + h`).
pub const DIGEST_HALVES: usize = W4_RULE + 1;

/// KECCAK_W's constraints, degree 3 (indices in order):
/// - the row and its flags: `MU`, `FIRST`, `LAST` bits; `FIRST`, `LAST` only
///   on a row;
/// - KECCAK's sponge rules: the mask's bits, one read a byte, a prefix, no
///   input off a row, whole blocks but the last, the last below 136 bytes and
///   `COUNT` of them, the first row block 0 from the zero state, `SRC_END`
///   and `OUT_END` the ends' halves;
/// - `OFF` one-hot on a row, `Q` one-hot on the last;
/// - `SRC_KEY = SRC_LO − o` on a row, `OUT_KEY = OUT_LO − q` on the last;
/// - `READ_w` fixed by the mask, the offset and `FIRST`;
/// - `MSG_i = s_i · Σ_k OFF_k · W[i + k]`;
/// - `W4 = LAST·(1 − Q_0)`;
/// - each digest word's halves the old word with `D` placed.
#[derive(Clone, Copy)]
pub struct KeccakWordConstraints;

/// `s_i` as an expression.
fn mask<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(b: &B, i: usize) -> Expr<B> {
    let (lane, byte) = (i / 8, i % 8);
    let f = b.main(0, cols::full(lane));
    if byte < 7 {
        f + b.main(0, cols::part(lane, byte))
    } else {
        f
    }
}

/// `Σ k·cols[start + k]`.
fn weighted<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    start: usize,
) -> Expr<B> {
    (1..8).fold(b.zero(), |acc, k| {
        acc + b.const_base(k as u64) * b.main(0, start + k)
    })
}

/// Two limbs as a word.
fn limbs_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    start: usize,
) -> Expr<B> {
    b.main(0, start) + b.const_base(1 << 16) * b.main(0, start + 1)
}

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for KeccakWordConstraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        emit_is_bit(b, 0, cols::MU, None);
        emit_is_bit(b, 1, cols::FIRST, None);
        emit_is_bit(b, 2, cols::LAST, None);
        let one = b.one();
        let (mu, first, last) = (
            b.main(0, cols::MU),
            b.main(0, cols::FIRST),
            b.main(0, cols::LAST),
        );
        b.emit_base(
            3,
            (first.clone() + last.clone()) * (one.clone() - mu.clone()),
        );
        let mut idx = 4;
        let mut emit = |b: &mut B, e: Expr<B>| {
            b.emit_base(idx, e);
            idx += 1;
        };

        // KECCAK's sponge rules.
        for lane in 0..RATE_LANES {
            let f = b.main(0, cols::full(lane));
            let e = f.clone() * (b.one() - f);
            emit(b, e);
            for byte in 0..7 {
                let p = b.main(0, cols::part(lane, byte));
                let e = p.clone() * (b.one() - p);
                emit(b, e);
            }
        }
        for lane in 0..RATE_LANES {
            for byte in 0..7 {
                let e = b.main(0, cols::full(lane)) * b.main(0, cols::part(lane, byte));
                emit(b, e);
            }
        }
        for i in 0..RATE - 1 {
            let e = (b.one() - mask(b, i)) * mask(b, i + 1);
            emit(b, e);
        }
        let e = (b.one() - mu.clone()) * mask(b, 0);
        emit(b, e);
        let full_last = b.main(0, cols::full(RATE_LANES - 1));
        let e = (mu.clone() - last.clone()) * (b.one() - full_last.clone());
        emit(b, e);
        emit(b, last.clone() * full_last);
        let taken = (0..RATE).fold(b.zero(), |acc, i| acc + mask(b, i));
        let e = last.clone() * (b.main(0, cols::COUNT) - taken);
        emit(b, e);
        let e = first.clone() * b.main(0, cols::TIMESTAMP_1);
        emit(b, e);
        for i in 0..RATE {
            let e = first.clone() * b.main(0, cols::prev(i));
            emit(b, e);
        }
        for i in RATE..200 {
            let e = first.clone() * b.main(0, cols::input_byte(i));
            emit(b, e);
        }
        let e = mu.clone()
            * (b.main(0, cols::SRC_LO) + b.const_base(135) - limbs_expr(b, cols::SRC_END));
        emit(b, e);
        let e = last.clone()
            * (b.main(0, cols::OUT_LO) + b.const_base(31) - limbs_expr(b, cols::OUT_END));
        emit(b, e);

        // The offsets, one-hot: the block's on a row, the digest's on the last.
        for (start, on) in [(cols::OFF, mu.clone()), (cols::Q, last.clone())] {
            for k in 0..8 {
                let x = b.main(0, start + k);
                let e = x.clone() * (b.one() - x);
                emit(b, e);
            }
            let sum = (0..8).fold(b.zero(), |acc, k| acc + b.main(0, start + k));
            emit(b, sum - on);
        }
        // The keys.
        let e = mu.clone()
            * (b.main(0, cols::SRC_LO) - weighted(b, cols::OFF) - limbs_expr(b, cols::SRC_KEY));
        emit(b, e);
        let e = last.clone()
            * (b.main(0, cols::OUT_LO) - weighted(b, cols::Q) - limbs_expr(b, cols::OUT_KEY));
        emit(b, e);

        // The reads.
        for w in 0..BLOCK_WORDS {
            let need = match w {
                0 => {
                    mask(b, 0)
                        * (first.clone() + (b.one() - first.clone()) * b.main(0, cols::off(0)))
                }
                17 => (1..8).fold(b.zero(), |acc, k| {
                    acc + b.main(0, cols::off(k)) * mask(b, RATE - k)
                }),
                _ => (0..8).fold(b.zero(), |acc, k| {
                    acc + b.main(0, cols::off(k)) * mask(b, 8 * w - k)
                }),
            };
            let e = b.main(0, cols::read(w)) - need;
            emit(b, e);
        }
        // The placement: MSG_i = s_i · Σ_k OFF_k · W[i + k].
        for i in 0..RATE {
            let placed = (0..8).fold(b.zero(), |acc, k| {
                acc + b.main(0, cols::off(k)) * b.main(0, cols::w(i + k))
            });
            let e = b.main(0, cols::msg(i)) - mask(b, i) * placed;
            emit(b, e);
        }
        let e = b.main(0, cols::W4) - last.clone() * (b.one() - b.main(0, cols::q(0)));
        emit(b, e);

        // The digest's words: byte j of word v is D[8v + j − q] when that is
        // a digest byte, else the old word's (word 0, word 4).
        for v in 0..DIGEST_WORDS {
            for half in 0..2 {
                let mut acc = b.zero();
                for jj in 0..4 {
                    let j = 4 * half + jj;
                    let at = 8 * v + j;
                    // Σ_q Q_q·D[at − q] over the q that land inside D.
                    let placed = (0..8)
                        .filter(|&q| at >= q && at - q < DIGEST)
                        .fold(b.zero(), |acc, q| {
                            acc + b.main(0, cols::q(q)) * b.main(0, cols::output_byte(at - q))
                        });
                    // 1 − Σ (the q that land inside D): the old byte's weight.
                    let kept = match v {
                        0 => Some((cols::OLD0, (0..8).filter(|&q| q > j).collect::<Vec<_>>())),
                        4 => Some((cols::OLD4, (0..8).filter(|&q| q <= j).collect::<Vec<_>>())),
                        _ => None,
                    };
                    let byte = match kept {
                        Some((old, qs)) => {
                            let weight = qs
                                .iter()
                                .fold(b.zero(), |acc, &q| acc + b.main(0, cols::q(q)));
                            placed + weight * b.main(0, old + j)
                        }
                        None => placed,
                    };
                    acc = acc + b.const_base(1 << (8 * jj)) * byte;
                }
                let e = b.main(0, cols::new(v, half)) - acc;
                emit(b, e);
            }
        }
        debug_assert_eq!(idx, NUM_CONSTRAINTS);
    }
}
