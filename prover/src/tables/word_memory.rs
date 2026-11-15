//! Memory as 8-byte words: how a block checks RAM under
//! [`MemoryFormat::Words`](super::fused_memory::MemoryFormat::Words), and the
//! pieces the word tables share.
//!
//! ## The token
//!
//! One token an aligned 8-byte RAM word on [`BusId::MemWord`]:
//! `(address_lo, address_hi, timestamp, lo, hi)`. The address is two field
//! elements, its low and high 32-bit words, and is never packed into one: in
//! one element `lo + 2^32·hi` the top of the address space wraps onto low
//! addresses (`2^64 − 2^32 ≡ −1`), so a stack word would answer for an
//! unaligned low pointer. The timestamp is one element, as the register
//! tokens' on [`BusId::Reg`]. `lo` and `hi` are the word's two 32-bit halves,
//! little-endian; every table that produces a token keeps them canonical.
//!
//! ## Who makes tokens
//!
//! - **PAGE_W** ([`super::page_w`]) is the only table that mints: a row
//!   produces a word's first token, at timestamp 0, and consumes its last. Its
//!   address is `page_base_lo + 8·OFFSET` with `OFFSET` preprocessed and the
//!   base page-aligned, so every minted token sits at an integer address that is
//!   a multiple of 8 and below `2^32`.
//! - **Every other table passes a token through**: a row consumes the token at
//!   a key and produces the next one at the same key, both at one multiplicity,
//!   that multiplicity a bit ([`word_check_interactions`]).
//! - **Steps are strict.** A row consumes the token made at
//!   `timestamp − 1 − STEP_0 − 2^16·STEP_1`, its two limbs each `IS_HALF`, so
//!   the old timestamp is strictly earlier, and the timestamps a row produces
//!   are the clock's (the row's message or dispatch carries them).
//!
//! So a balanced set of accesses at a key needs a chain back to a PAGE_W row
//! at that key: an access at an address that is not a multiple of 8, or past
//! `2^32`, consumes a token nothing produced, and the bus does not balance.
//! Alignment is a property of the keys PAGE_W mints; no word table checks it
//! with a lookup on an address half.
//!
//! ## The chips' messages
//!
//! A chip accesses a word through MEMW_W ([`super::memw_w`]) with a message on
//! [`BusId::MemWordMsg`], led by a kind constant:
//! - a write [`MESSAGE_WRITE`]: `(1, address_lo, address_hi, timestamp,
//!   new_lo, new_hi)`; the word's old value is MEMW_W's own, bound by the token
//!   it consumes;
//! - an access [`MESSAGE_ACCESS`]: `(0, address_lo, address_hi, timestamp,
//!   old_lo, old_hi, new_lo, new_hi)`; a read sends old = new.
//!
//! The kind is part of the message, not a column of the receiver: the
//! fingerprint has no length term, so without it a write of `n` would equal an
//! access of old `n` and new `(0, 0)`.
//!
//! ## Bytes
//!
//! A table that uses a word's single bytes holds all eight and checks every
//! one ([`byte_pair_checks`]); the word is their little-endian sum
//! ([`word_of_bytes`]). Eight bytes decompose a canonical word one way only.

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};

use super::types::BusId;

/// The kind constant that leads a chip's access message (old and new word).
pub(crate) const MESSAGE_ACCESS: u64 = 0;
/// The kind constant that leads a chip's write message (new word only).
pub(crate) const MESSAGE_WRITE: u64 = 1;

/// A word's bytes: one word token covers this many.
pub const WORD_BYTES: u64 = 8;

pub(crate) fn column(c: usize) -> BusValue {
    BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    }
}

pub(crate) fn term(coefficient: i64, column: usize) -> LinearTerm {
    LinearTerm::Column {
        coefficient,
        column,
    }
}

/// A word token's five values: the key's two elements, the timestamp, the
/// word's two halves.
pub(crate) fn token(
    address: [BusValue; 2],
    timestamp: BusValue,
    word: [BusValue; 2],
) -> Vec<BusValue> {
    let [address_lo, address_hi] = address;
    let [lo, hi] = word;
    vec![address_lo, address_hi, timestamp, lo, hi]
}

/// A chip's write message: the kind [`MESSAGE_WRITE`], the address, the
/// timestamp, the new word.
pub(crate) fn write_message(
    address: [BusValue; 2],
    timestamp: BusValue,
    new: [BusValue; 2],
) -> Vec<BusValue> {
    let [address_lo, address_hi] = address;
    let [new_lo, new_hi] = new;
    vec![
        BusValue::constant(MESSAGE_WRITE),
        address_lo,
        address_hi,
        timestamp,
        new_lo,
        new_hi,
    ]
}

/// A chip's access message: the kind [`MESSAGE_ACCESS`], the address, the
/// timestamp, the old word and the new one (equal for a read).
pub(crate) fn access_message(
    address: [BusValue; 2],
    timestamp: BusValue,
    old: [BusValue; 2],
    new: [BusValue; 2],
) -> Vec<BusValue> {
    let [address_lo, address_hi] = address;
    let [old_lo, old_hi] = old;
    let [new_lo, new_hi] = new;
    vec![
        BusValue::constant(MESSAGE_ACCESS),
        address_lo,
        address_hi,
        timestamp,
        old_lo,
        old_hi,
        new_lo,
        new_hi,
    ]
}

/// The columns a pass-through word check reads ([`word_check_interactions`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct WordCheck {
    /// The word's address: its low and its high 32-bit word.
    pub(crate) address: [usize; 2],
    /// The access's timestamp.
    pub(crate) timestamp: usize,
    /// The word the consumed token carries.
    pub(crate) old: [usize; 2],
    /// The word the produced token carries.
    pub(crate) new: [usize; 2],
    /// `timestamp − old_timestamp − 1`, as two 16-bit limbs.
    pub(crate) step: [usize; 2],
}

/// A pass-through word check, every interaction at `mu`: the two limbs'
/// `IS_HALF`, then the consumed token at
/// `(address, timestamp − 1 − STEP_0 − 2^16·STEP_1, old)` and the produced one
/// at `(address, timestamp, new)`. The two tokens share the key expression and
/// the multiplicity, so a row moves a token and never mints one.
pub(crate) fn word_check_interactions(c: &WordCheck, mu: Multiplicity) -> Vec<BusInteraction> {
    let address = || [column(c.address[0]), column(c.address[1])];
    let old_timestamp = BusValue::linear(vec![
        term(1, c.timestamp),
        term(-1, c.step[0]),
        term(-(1 << 16), c.step[1]),
        LinearTerm::Constant(-1),
    ]);
    let mut interactions = Vec::with_capacity(4);
    for limb in c.step {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            mu.clone(),
            vec![column(limb)],
        ));
    }
    interactions.push(BusInteraction::sender(
        BusId::MemWord,
        mu.clone(),
        token(
            address(),
            old_timestamp,
            [column(c.old[0]), column(c.old[1])],
        ),
    ));
    interactions.push(BusInteraction::receiver(
        BusId::MemWord,
        mu,
        token(
            address(),
            column(c.timestamp),
            [column(c.new[0]), column(c.new[1])],
        ),
    ));
    interactions
}

/// `timestamp − old_timestamp − 1` of a word access, which two 16-bit limbs
/// hold: the old timestamp strictly earlier and at most `2^32` before. Anything
/// else is refused with the reason, never laid out with wrapped limbs.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn step_less_one(old_timestamp: u64, timestamp: u64) -> Result<u32, String> {
    if old_timestamp >= timestamp {
        return Err(format!(
            "a word access consumes an earlier token: {old_timestamp} → {timestamp}"
        ));
    }
    u32::try_from(timestamp - old_timestamp - 1).map_err(|_| {
        format!("a word access steps at most 2^32 timestamps: {old_timestamp} → {timestamp}")
    })
}

/// The limbs `(STEP_0, STEP_1)` of a step less one.
#[inline]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn step_limbs(step: u32) -> (u16, u16) {
    (step as u16, (step >> 16) as u16)
}

/// A word's half from four byte columns, little-endian:
/// `b[0] + 2^8·b[1] + 2^16·b[2] + 2^24·b[3]`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn word_of_bytes(bytes: [usize; 4]) -> BusValue {
    BusValue::linear(
        bytes
            .iter()
            .enumerate()
            .map(|(k, &b)| term(1 << (8 * k), b))
            .collect(),
    )
}

/// `ARE_BYTES` on a word's eight byte columns, two a lookup, all at `mu`:
/// with all eight bytes checked, their sum is the word's one decomposition.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn byte_pair_checks(bytes: [usize; 8], mu: Multiplicity) -> Vec<BusInteraction> {
    bytes
        .chunks_exact(2)
        .map(|pair| {
            BusInteraction::sender(
                BusId::AreBytes,
                mu.clone(),
                vec![column(pair[0]), column(pair[1])],
            )
        })
        .collect()
}

/// A 64-bit word's two 32-bit halves, low first.
#[inline]
pub(crate) fn halves(word: u64) -> (u32, u32) {
    (word as u32, (word >> 32) as u32)
}
