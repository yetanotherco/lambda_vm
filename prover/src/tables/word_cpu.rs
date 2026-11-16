//! The CPU's loads and stores under
//! [`MemoryFormat::Words`](super::fused_memory::MemoryFormat::Words): six
//! tables by the words an access touches, and the pieces they share.
//!
//! ## Which table takes an access
//!
//! A load or store of `w` bytes at address `A` (low 32-bit word `A_lo`) with
//! offset `o = A mod 8` in its word ([`word_class`]):
//! - `w = 8`, `o = 0`: one whole word, MEM_LOAD8 ([`super::mem_load8`]) or
//!   MEM_STORE8 ([`super::mem_store8`]);
//! - `o + w ≤ 8`, `w < 8`: inside one word, MEM_LOADW ([`super::mem_loadw`])
//!   or MEM_STOREW ([`super::mem_storew`]);
//! - `o + w > 8`: across two words, MEM_LOAD2W ([`super::mem_load2w`]) or
//!   MEM_STORE2W ([`super::mem_store2w`]).
//!
//! Every access is one of these, and each table refuses the others' accesses,
//! so the walk's routing is exact.
//!
//! ## What a row checks
//!
//! - **The dispatch.** A row receives the CPU's memory dispatch on
//!   [`BusId::MemoryOp`] with LOAD's or STORE's tuple exactly ([`dispatch`]):
//!   `(timestamp, 0, A_lo, A_hi, value_lo, value_hi, flags, result_lo,
//!   result_hi)`, at its multiplicity `MU`, a bit. The CPU sends each access
//!   once, so one row of one table takes it. The flags are the table's widths'
//!   only, every flag a bit and their sum a bit.
//! - **The words.** A row passes each word it touches through
//!   ([`super::word_memory::word_check_of`]): it consumes the word's token at
//!   an earlier timestamp and produces the next at the dispatch's. The key's
//!   low element is `A_lo − o` (`+ 8` for the second word), `o` the row's
//!   one-hot `OFF` ([`one_hot_constraints`]), and the high element is `A_hi`;
//!   a second word past the last word of a low range is keyed with the carry
//!   taken into the high word, `(A_lo − o + 8 − 2^32, A_hi + 1)`
//!   ([`second_word_key`], [`carry_constraints`]).
//!   Tokens exist only at keys PAGE_W mints, integer multiples of 8 below
//!   `2^32` (`super::word_memory`); `A_lo` is an integer below `2^32` (the
//!   CPU's `IS_HALF` on each half of its address); and `o` is in `0..8`. So
//!   `A_lo − o` is a key only when `o = A mod 8`: the tokens force the offset,
//!   and an 8-byte table's `A` aligned.
//! - **The window.** A narrow table's window stays inside its word
//!   (`o + w ≤ 8`), a two-word table's crosses (`o + w > 8`): each a
//!   polynomial identity on the one-hot and the width flags.
//! - **The bytes.** A table that uses single bytes holds every byte of each
//!   word it touches and checks all eight with `ARE_BYTES` pairs, so a placed
//!   byte is the memory's byte; a store checks the register bytes it places.
//!   A load's result bytes are the word's bytes at `o..o + w` (placement
//!   identities) and its extension past the width; a store's new word is the
//!   old one with bytes `o..o + w` replaced by the register's low bytes.

use stark::constraints::builder::ConstraintBuilder;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity};

use super::bitwise::{BitwiseOperation, BitwiseOperationType};
use super::types::{BusId, GoldilocksExtension, GoldilocksField};
use super::word_memory::{WORD_BYTES, column, step_limbs, term};

/// The word memory format's table families, in the block's AIR order and
/// its proof's word counts' (`block_whir::WORD_TABLES` re-exports it): the
/// CPU's loads and stores by the words they touch (one whole word, inside one
/// word, across two), then the chips' word table.
pub const WORD_TABLES: [&str; 7] = [
    "MEM_LOAD8",
    "MEM_STORE8",
    "MEM_LOADW",
    "MEM_STOREW",
    "MEM_LOAD2W",
    "MEM_STORE2W",
    "MEMW_W",
];

/// A load's flags (LOAD's `mem_flags`): `2·signed + 4·w2 + 8·w4 + 16·w8`.
pub(crate) const FLAG_SIGNED: i64 = 2;
pub(crate) const FLAG_W2: i64 = 4;
pub(crate) const FLAG_W4: i64 = 8;
pub(crate) const FLAG_W8: i64 = 16;
/// A store's flags add its memory-op bit, 1 (STORE's `mem_flags`).
pub(crate) const FLAG_STORE: i64 = 1;

/// The words an access touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WordClass {
    /// Eight bytes, aligned: one whole word.
    Whole,
    /// Fewer than eight bytes inside one word.
    Within,
    /// Bytes in two words.
    Crossing,
}

/// The class of a `width`-byte access at `address`, or the reason no word
/// table takes it: a width the CPU never accesses, or a crossing access whose
/// second word would wrap past `2^64` to address 0 (a two-word table's carry
/// keys it at a high word of `2^32`, which no PAGE_W key has).
pub(crate) fn word_class(address: u64, width: u8) -> Result<WordClass, String> {
    if !matches!(width, 1 | 2 | 4 | 8) {
        return Err(format!("a CPU access is 1, 2, 4 or 8 bytes, not {width}"));
    }
    let offset = address % WORD_BYTES;
    if offset + u64::from(width) <= WORD_BYTES {
        return Ok(if width == 8 {
            WordClass::Whole
        } else {
            WordClass::Within
        });
    }
    if address - offset == u64::MAX - (WORD_BYTES - 1) {
        return Err(format!(
            "an access at {address:#x} crosses past the top of the address space"
        ));
    }
    Ok(WordClass::Crossing)
}

/// The low word of a low range's last aligned word, `2^32 − 8`: a crossing
/// access whose first word is there finds its second word at low word 0 of
/// the next high word ([`second_word_key`]).
pub(crate) const LAST_WORD_LO: u64 = (1 << 32) - WORD_BYTES;

/// Whether a crossing access at `address` carries into the next high word:
/// its first word is the last of its low range.
#[inline]
pub(crate) fn carries(address: u64) -> bool {
    (address & 0xFFFF_FFFF) - address % WORD_BYTES == LAST_WORD_LO
}

/// The dispatch a word table receives at `mu`: the CPU's `MemoryOp` tuple
/// `(timestamp, 0, address_lo, address_hi, value_lo, value_hi, flags,
/// result_lo, result_hi)`, as LOAD and STORE receive it.
pub(crate) fn dispatch(
    timestamp: usize,
    address: [usize; 2],
    value: [BusValue; 2],
    flags: BusValue,
    result: [BusValue; 2],
    mu: usize,
) -> BusInteraction {
    let [value_lo, value_hi] = value;
    let [result_lo, result_hi] = result;
    BusInteraction::receiver(
        BusId::MemoryOp,
        Multiplicity::Column(mu),
        vec![
            column(timestamp),
            BusValue::constant(0),
            column(address[0]),
            column(address[1]),
            value_lo,
            value_hi,
            flags,
            result_lo,
            result_hi,
        ],
    )
}

/// A dispatch's flags: `constant + Σ coefficient·flag`.
pub(crate) fn flags(constant: i64, flags: &[(i64, usize)]) -> BusValue {
    let mut terms: Vec<LinearTerm> = flags.iter().map(|&(c, col)| term(c, col)).collect();
    if constant != 0 {
        terms.push(LinearTerm::Constant(constant));
    }
    BusValue::linear(terms)
}

/// A word's key `(address_lo − Σ k·OFF_k + 8·word, address_hi)`: the word
/// `word` (0 or 1) of an access at `address` with the one-hot offset `off`.
/// The two-word tables key their second word with [`second_word_key`].
pub(crate) fn word_key(address: [usize; 2], off: [usize; 8], word: i64) -> [BusValue; 2] {
    let mut lo = vec![term(1, address[0])];
    lo.extend((1..8).map(|k| term(-(k as i64), off[k])));
    if word != 0 {
        lo.push(LinearTerm::Constant(WORD_BYTES as i64 * word));
    }
    [BusValue::linear(lo), column(address[1])]
}

/// The second word's key of an access across two words, at `address` with the
/// one-hot offset `off` and the carry `carry`: `(address_lo − Σ k·OFF_k + 8 −
/// 2^32·CARRY, address_hi + CARRY)`. `CARRY` is a bit that is 1 only on the
/// last word of a low range ([`carry_constraints`]): there the next word is
/// low word 0 of the next high word, the address plus 8 with the carry taken
/// into the high word. A key's low element is a PAGE_W key's, an integer below
/// `2^32`, only with the carry the address makes (0 there gives `2^32`), so
/// the word bus refuses a wrong carry as well.
pub(crate) fn second_word_key(address: [usize; 2], off: [usize; 8], carry: usize) -> [BusValue; 2] {
    let mut lo = vec![term(1, address[0])];
    lo.extend((1..8).map(|k| term(-(k as i64), off[k])));
    lo.push(LinearTerm::Constant(WORD_BYTES as i64));
    lo.push(term(-(1 << 32), carry));
    [
        BusValue::linear(lo),
        BusValue::linear(vec![term(1, address[1]), term(1, carry)]),
    ]
}

/// The carry's two constraints, from index `idx`: `CARRY` a bit, and
/// `CARRY·(ADDRESS_LO − Σ k·OFF_k − (2^32 − 8)) = 0`, a carry only from the
/// last word of a low range (degree 2). Returns the next index.
pub(crate) fn carry_constraints<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &mut B,
    idx: usize,
    address_lo: usize,
    off: [usize; 8],
    carry: usize,
) -> usize {
    let root = bit(b, b.main(0, carry));
    b.emit_base(idx, root);
    let first_word = (1..8).fold(b.main(0, address_lo), |acc, k| {
        acc - b.const_base(k as u64) * b.main(0, off[k])
    });
    let root = b.main(0, carry) * (first_word - b.const_base(LAST_WORD_LO));
    b.emit_base(idx + 1, root);
    idx + 2
}

/// An `MSB8` send of `(byte, sign_bit)` at `multiplicity`: `sign_bit` is the
/// top bit of the byte (LOAD's sign bit).
pub(crate) fn msb8(byte: usize, sign_bit: usize, multiplicity: Multiplicity) -> BusInteraction {
    BusInteraction::sender(
        BusId::Msb8,
        multiplicity,
        vec![column(byte), column(sign_bit)],
    )
}

/// Word family `k`'s main columns and interactions
/// (`block_whir::WORD_TABLES`' order: the six CPU tables, then MEMW_W).
pub(crate) fn word_table_shape(k: usize) -> (usize, usize) {
    use super::{mem_load2w, mem_load8, mem_loadw, mem_store2w, mem_store8, mem_storew, memw_w};
    match k {
        0 => (
            mem_load8::cols::NUM_COLUMNS,
            mem_load8::bus_interactions().len(),
        ),
        1 => (
            mem_store8::cols::NUM_COLUMNS,
            mem_store8::bus_interactions().len(),
        ),
        2 => (
            mem_loadw::cols::NUM_COLUMNS,
            mem_loadw::bus_interactions().len(),
        ),
        3 => (
            mem_storew::cols::NUM_COLUMNS,
            mem_storew::bus_interactions().len(),
        ),
        4 => (
            mem_load2w::cols::NUM_COLUMNS,
            mem_load2w::bus_interactions().len(),
        ),
        5 => (
            mem_store2w::cols::NUM_COLUMNS,
            mem_store2w::bus_interactions().len(),
        ),
        _ => (memw_w::cols::NUM_COLUMNS, memw_w::bus_interactions().len()),
    }
}

// =========================================================================
// Constraints
// =========================================================================

type Expr<B> = <B as ConstraintBuilder<GoldilocksField, GoldilocksExtension>>::Expr;

/// `x·(1 − x)`.
pub(crate) fn bit<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    x: Expr<B>,
) -> Expr<B> {
    let one = b.one();
    x.clone() * (one - x)
}

/// The sum of `cols`.
pub(crate) fn sum<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    cols: &[usize],
) -> Expr<B> {
    cols.iter().fold(b.zero(), |acc, &c| acc + b.main(0, c))
}

/// The one-hot offset: each `OFF_k` a bit (8 constraints) and their sum `MU`
/// (1), from index `idx`. Returns the next index.
pub(crate) fn one_hot_constraints<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &mut B,
    idx: usize,
    off: [usize; 8],
    mu: usize,
) -> usize {
    for (k, &c) in off.iter().enumerate() {
        let root = bit(b, b.main(0, c));
        b.emit_base(idx + k, root);
    }
    let root = sum(b, &off) - b.main(0, mu);
    b.emit_base(idx + 8, root);
    idx + 9
}

/// `Σ_k OFF_k·bytes[k + j]` over the `k` with `k + j` inside `bytes`: the
/// byte `j` places from the offset (degree 2).
pub(crate) fn placed<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    off: [usize; 8],
    bytes: &[usize],
    j: usize,
) -> Expr<B> {
    (0..8)
        .filter(|k| k + j < bytes.len())
        .fold(b.zero(), |acc, k| {
            acc + b.main(0, off[k]) * b.main(0, bytes[k + j])
        })
}

/// A store's new half `half` (bytes `4·half..4·half + 4` of `old`) as the
/// identity `NEW − old half − Σ_j act_j·Σ_k [k + j in the half]
/// OFF_k·2^(8(k + j − 4·half))·(value_j − old_(k + j))` (degree 3): the old
/// half with each byte `o + j` the store writes (`j < w`, `act_j` its width
/// gate, `None` for every width) replaced by `value_j`.
pub(crate) fn stored_half<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    new: usize,
    half: usize,
    off: [usize; 8],
    old: &[usize],
    value: &[usize],
    act: &[Option<Expr<B>>],
) -> Expr<B> {
    let lo = 4 * half;
    let old_half = (0..4).fold(b.zero(), |acc, p| {
        acc + b.const_base(1u64 << (8 * p)) * b.main(0, old[lo + p])
    });
    let mut written = b.zero();
    for (j, gate) in act.iter().enumerate() {
        let mut row = b.zero();
        for (k, &o) in off.iter().enumerate() {
            let at = k + j;
            if !(lo..lo + 4).contains(&at) || at >= old.len() {
                continue;
            }
            let scale = b.const_base(1u64 << (8 * (at - lo)));
            row = row + scale * b.main(0, o) * (b.main(0, value[j]) - b.main(0, old[at]));
        }
        written = written
            + match gate {
                Some(g) => g.clone() * row,
                None => row,
            };
    }
    b.main(0, new) - old_half - written
}

// =========================================================================
// Rows
// =========================================================================

/// A row's two step limbs' `IS_HALF` lookups.
#[inline]
pub(crate) fn limb_lookups(step: u32) -> [BitwiseOperation; 2] {
    let (d0, d1) = step_limbs(step);
    [d0, d1].map(|half| {
        BitwiseOperation::halfword(BitwiseOperationType::IsHalf, half as u8, (half >> 8) as u8)
    })
}

/// `ARE_BYTES` lookups on `bytes`, two a lookup.
#[inline]
pub(crate) fn pair_lookups(bytes: &[u8]) -> impl Iterator<Item = BitwiseOperation> + '_ {
    bytes
        .chunks_exact(2)
        .map(|pair| BitwiseOperation::byte_op(BitwiseOperationType::AreBytes, pair[0], pair[1]))
}

/// A load's result as LOAD makes it: the `width` bytes from `offset` of
/// `bytes`, sign- or zero-extended to eight; and its sign bit (the top bit of
/// the highest byte read).
pub(crate) fn load_result(bytes: &[u8], offset: usize, width: u8, signed: bool) -> ([u8; 8], bool) {
    let width = usize::from(width);
    let sign_bit = bytes[offset + width - 1] >> 7 == 1;
    let fill = if signed && sign_bit { 0xFF } else { 0 };
    let mut res = [fill; 8];
    res[..width].copy_from_slice(&bytes[offset..offset + width]);
    (res, sign_bit)
}

/// `old` (little-endian bytes) with the `width` low bytes of `value` written
/// from `offset`.
pub(crate) fn store_into(old: &mut [u8], offset: usize, width: u8, value: u64) {
    let width = usize::from(width);
    old[offset..offset + width].copy_from_slice(&value.to_le_bytes()[..width]);
}
