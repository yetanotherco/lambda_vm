//! `B3R` — a lean, NARROW BLAKE3 compression chip: one round a row.
//!
//! Test-only. This is the chip D-B3-3 §2.2 designs, built to turn its model
//! into a census and a card measurement (thoughts/zf/gap3/I-B3CHIP.md). It is
//! not registered in the LFM's AIR set and nothing outside the probes reads it.
//!
//! # Shape
//!
//! Three AIRs prove one compression:
//!
//! - [`round`] (`B3R_ROUND`): seven rows, one a round, and **no transition
//!   constraints**. A row receives `(tag, r, state, m)` on [`ROUND_BUS`] and
//!   sends `(tag, r + 1, state', P(m))`, the way `KECCAK_RND` chains its 24
//!   rounds (`chunking.rs`). So the table splits into chunks with no pairing
//!   logic, and an all-zero padding row satisfies every constraint.
//! - [`head`] (`B3R_HEAD`): one row a compression. It reads the chaining value
//!   and the message block from `LfmMem`, range-checks them, sends the round-0
//!   token, receives the round-7 token and computes the feed-forward of a
//!   **256-bit digest** (`out[i] = v[i] ⊕ v[i+8]`, `i < 8`), which it writes
//!   back to `LfmMem`.
//! - [`table`] (`B3R_XOR`): the fixed 2^16-row byte table every lookup closes
//!   against — XOR, XOR split into nibbles, byte pair, 16-bit value.
//!
//! # The G-function, 40 cells and 16 lookups
//!
//! Limbs are bytes. The four XORs are lookups; every rotation is linear wiring:
//!
//! | step | cells | how |
//! |---|---|---|
//! | `a' = a + b + m₀` | 4 | the carry is implicit: `s·(s − 2³²)·(s − 2·2³²) = 0`, `s = a + b + m − a'` |
//! | `d' = (d ⊕ a') ⋙ 16` | 4 | four XOR lookups; the rotation is a byte relabel |
//! | `c' = c + d'` | 4 | implicit carry, quadratic |
//! | `b' = (b ⊕ c') ⋙ 12` | 8 | the SPLIT lookup returns each XOR byte as (low, high) nibbles; `b'` bytes are linear in them |
//! | `a'' = a' + b' + m₁` | 4 | as `a'` |
//! | `d'' = (d' ⊕ a'') ⋙ 8` | 4 | relabel |
//! | `c'' = c' + d''` | 4 | as `c'` |
//! | `b'' = (b' ⊕ c'') ⋙ 7` | 8 | `z = b' ⊕ c''`, then `⋙7 = (⋙8)⋘1`: byte `i` is `2·y_i − 256·t_i + t_{i−1}` with `y = z ⋙ 8` and four committed top bits |
//!
//! # Why the implicit carries are exact
//!
//! `s = a + b + m − out` is an integer identity only when every operand is
//! below 2³²: then `a + b + m < 3·2³² < p`, so `s ∈ {0, 2³², 2·2³²}` as field
//! elements means exactly the true carry. Every operand is bounded by a lookup:
//!
//! - every add OUTPUT is an XOR operand in the same G (`a'` in `x1`, `c'` in
//!   `x2`, `a''` in `x3`, `c''` in `z`);
//! - `b'`'s bytes are sums of table nibbles; `d'`, `d''` are table outputs;
//! - `b''`'s bytes are in range only for the true top bits (the bits are
//!   boolean, and a wrong bit pushes its byte below 0 or above 255), and every
//!   `b''` is an XOR operand — in the next G of its row, in the next round's row
//!   (through the token, which forces the receiving columns equal), or in the
//!   head's feed-forward;
//! - the message words and `h[0..4]` are range-checked in the head; `h[4..8]`
//!   are round 0's `x2` operands; `IV` is constant.
//!
//! # What crosses a row boundary, and what it costs
//!
//! A bus value reads only its own row, so the next round's row cannot look up
//! the previous row's bytes. Each G is two XOR chains (`d → x1 → x3`,
//! `b → x2 → z`), so the round's input `b`, `d` words come in as bytes and the
//! add-chain words `a`, `c` as one cell each — 40 cells a row, for the four
//! column G's only: the diagonal G's read the column G's outputs in the same
//! row. The message travels as 16 word cells a row. That is what the narrow
//! layout pays over one row a compression (I-B3CHIP §1.1).

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity};
use stark::trace::TraceTable;

use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField, zeroed_fe_vec};

use super::blake3::{BLAKE3_IV, BLAKE3_MSG_PERMUTATION, blake3_compress_rounds};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Standard BLAKE3: seven rounds, whatever the build's `blake3-6round` says.
pub const ROUNDS: usize = 7;

/// The fixed byte table's bus. Outside `BusId`'s range: this chip is not in the
/// machine, and its buses must not alias one that is.
pub const XOR_BUS: u64 = 0xB3_01;
/// The round chain's bus: `(tag, round, state, message)`.
pub const ROUND_BUS: u64 = 0xB3_02;

/// `XOR_BUS` op tags, the tuple's first element.
pub const OP_XOR: i64 = 0;
pub const OP_SPLIT: i64 = 1;
pub const OP_PAIR: i64 = 2;
pub const OP_R16: i64 = 3;

/// `chunk_rows`' default ceiling: `KECCAK_RND`'s chunk cap in the machine.
pub const MAX_CHUNK_LOG: u32 = 19;
/// The smallest chunk a remainder is padded up to.
pub const MIN_CHUNK_LOG: u32 = 3;

/// The (a, b, c, d) state indices of a round's 8 G-calls: 4 column mixes, then
/// 4 diagonal mixes (BLAKE3 spec §2.1).
pub const G_INDICES: [(usize, usize, usize, usize); 8] = [
    (0, 4, 8, 12),
    (1, 5, 9, 13),
    (2, 6, 10, 14),
    (3, 7, 11, 15),
    (0, 5, 10, 15),
    (1, 6, 11, 12),
    (2, 7, 8, 13),
    (3, 4, 9, 14),
];

const TWO_32: i64 = 1 << 32;

// =========================================================================
// Linear forms
// =========================================================================

/// A linear form over one row's columns: `Σ coeff·col + constant`.
///
/// Every value a constraint or a bus tuple reads is one of these, so the two
/// cannot read different wirings.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lin {
    pub terms: Vec<(usize, i64)>,
    pub constant: i64,
}

impl Lin {
    pub fn col(c: usize) -> Self {
        Lin {
            terms: vec![(c, 1)],
            constant: 0,
        }
    }

    pub fn konst(v: i64) -> Self {
        Lin {
            terms: Vec::new(),
            constant: v,
        }
    }

    pub fn scaled(&self, k: i64) -> Self {
        Lin {
            terms: self.terms.iter().map(|&(c, a)| (c, a * k)).collect(),
            constant: self.constant * k,
        }
    }

    pub fn plus(mut self, other: &Lin) -> Self {
        self.terms.extend_from_slice(&other.terms);
        self.constant += other.constant;
        self
    }

    /// The terms with each column once and no zero coefficient: a word of
    /// `b''` bytes names each top bit twice, and the two mostly cancel.
    pub fn merged(&self) -> Vec<(usize, i64)> {
        let mut t = self.terms.clone();
        t.sort_unstable_by_key(|&(c, _)| c);
        let mut out: Vec<(usize, i64)> = Vec::with_capacity(t.len());
        for (c, k) in t {
            match out.last_mut() {
                Some(last) if last.0 == c => last.1 += k,
                _ => out.push((c, k)),
            }
        }
        out.retain(|&(_, k)| k != 0);
        out
    }

    pub fn bus(&self) -> BusValue {
        let mut terms: Vec<LinearTerm> = self
            .merged()
            .into_iter()
            .map(|(column, coefficient)| LinearTerm::Column {
                coefficient,
                column,
            })
            .collect();
        if self.constant != 0 || terms.is_empty() {
            terms.push(LinearTerm::Constant(self.constant));
        }
        BusValue::Linear(terms)
    }

    pub fn expr<B: ConstraintBuilder<F, E>>(&self, b: &B) -> B::Expr {
        let mut acc = b.const_signed(self.constant);
        for (c, k) in self.merged() {
            acc = acc + b.main(0, c) * b.const_signed(k);
        }
        acc
    }
}

/// Four byte forms, least significant first.
pub type Bytes = [Lin; 4];

pub fn cols4(start: usize) -> Bytes {
    core::array::from_fn(|k| Lin::col(start + k))
}

pub fn word_of(bytes: &Bytes) -> Lin {
    bytes
        .iter()
        .enumerate()
        .fold(Lin::default(), |acc, (k, b)| {
            acc.plus(&b.scaled(1 << (8 * k)))
        })
}

/// `x ⋙ 8·n` as a byte relabel: output byte `j` is input byte `j + n`.
pub fn rotr_bytes(x: &Bytes, n: usize) -> Bytes {
    core::array::from_fn(|j| x[(j + n) % 4].clone())
}

fn konst_bytes(w: u32) -> Bytes {
    core::array::from_fn(|k| Lin::konst(((w >> (8 * k)) & 0xFF) as i64))
}

// =========================================================================
// The round row
// =========================================================================

pub mod round {
    //! `B3R_ROUND` columns: preprocessed `[MU, TAG, RND]`, then the round's
    //! input state, its message, and 8 G-blocks of 40.

    pub const MU: usize = 0;
    pub const TAG: usize = 1;
    pub const RND: usize = 2;
    pub const PREP_WIDTH: usize = 3;

    /// `v[0..4]`, one word cell each (the `a` operands of the column G's).
    pub const IA: usize = PREP_WIDTH;
    /// `v[4..8]`, four bytes each.
    pub const IB: usize = IA + 4;
    /// `v[8..12]`, one word cell each.
    pub const IC: usize = IB + 16;
    /// `v[12..16]`, four bytes each.
    pub const ID: usize = IC + 4;
    /// This round's message words, in its schedule order.
    pub const M: usize = ID + 16;
    pub const G: usize = M + 16;
    pub const G_SIZE: usize = 40;
    pub const NUM_COLUMNS: usize = G + 8 * G_SIZE;
    pub const MAIN_COLUMNS: usize = NUM_COLUMNS - PREP_WIDTH;

    // Offsets inside a G-block.
    pub const A1: usize = 0;
    pub const X1: usize = 4;
    pub const C1: usize = 8;
    /// `x2`'s nibbles: low of byte `k` at `P2 + 2k`, high at `P2 + 2k + 1`.
    pub const P2: usize = 12;
    pub const A2: usize = 20;
    pub const X3: usize = 24;
    pub const C2: usize = 28;
    pub const Z: usize = 32;
    /// The top bit of `z` byte `k`.
    pub const T: usize = 36;

    pub const fn g_base(g: usize) -> usize {
        G + g * G_SIZE
    }
}

/// One state word as wiring: its word form, and its bytes where it has them
/// (a row input `a` or `c` is a single cell).
#[derive(Clone, Debug)]
pub struct SRef {
    pub word: Lin,
    pub bytes: Option<Bytes>,
}

impl SRef {
    fn of_bytes(bytes: Bytes) -> Self {
        SRef {
            word: word_of(&bytes),
            bytes: Some(bytes),
        }
    }

    fn bytes(&self) -> &Bytes {
        self.bytes
            .as_ref()
            .expect("a b or d operand always has bytes: a row input or a G output")
    }
}

/// One G-call's wiring.
#[derive(Clone, Debug)]
pub struct GWire {
    pub base: usize,
    pub a: Lin,
    pub b: Bytes,
    pub c: Lin,
    pub d: Bytes,
    pub mx: Lin,
    pub my: Lin,
    /// `d' = x1 ⋙ 16`.
    pub d1: Bytes,
    /// `b' = x2 ⋙ 12`, from the nibbles.
    pub b1: Bytes,
    /// `d'' = x3 ⋙ 8`.
    pub d2: Bytes,
    /// `b'' = z ⋙ 7`, from `z` and its top bits.
    pub b2: Bytes,
}

/// The whole row's wiring: the 8 G's in order and the state after the round.
#[derive(Clone, Debug)]
pub struct RoundWires {
    pub gs: Vec<GWire>,
    pub out: Vec<SRef>,
}

/// `b' = x2 ⋙ 12`: byte `i` is `hi(x_{i+1}) + 16·lo(x_{i+2})`.
fn rotr12_from_nibbles(base: usize) -> Bytes {
    let lo = |k: usize| Lin::col(base + round::P2 + 2 * (k % 4));
    let hi = |k: usize| Lin::col(base + round::P2 + 2 * (k % 4) + 1);
    core::array::from_fn(|i| hi(i + 1).plus(&lo(i + 2).scaled(16)))
}

/// `b'' = z ⋙ 7 = (z ⋙ 8) ⋘ 1`: byte `i` is `2·z_{i+1} − 256·t_{i+1} + t_i`,
/// with `t_k` the top bit of `z_k`.
fn rotr7_from_top_bits(base: usize) -> Bytes {
    let z = |k: usize| Lin::col(base + round::Z + k % 4);
    let t = |k: usize| Lin::col(base + round::T + k % 4);
    core::array::from_fn(|i| z(i + 1).scaled(2).plus(&t(i + 1).scaled(-256)).plus(&t(i)))
}

pub fn round_wires() -> RoundWires {
    let mut state: Vec<SRef> = (0..16)
        .map(|i| match i {
            0..=3 => SRef {
                word: Lin::col(round::IA + i),
                bytes: None,
            },
            4..=7 => SRef::of_bytes(cols4(round::IB + 4 * (i - 4))),
            8..=11 => SRef {
                word: Lin::col(round::IC + (i - 8)),
                bytes: None,
            },
            _ => SRef::of_bytes(cols4(round::ID + 4 * (i - 12))),
        })
        .collect();
    let mut gs = Vec::with_capacity(8);
    for (j, &(ia, ib, ic, id)) in G_INDICES.iter().enumerate() {
        let base = round::g_base(j);
        let wire = GWire {
            base,
            a: state[ia].word.clone(),
            b: state[ib].bytes().clone(),
            c: state[ic].word.clone(),
            d: state[id].bytes().clone(),
            mx: Lin::col(round::M + 2 * j),
            my: Lin::col(round::M + 2 * j + 1),
            d1: rotr_bytes(&cols4(base + round::X1), 2),
            b1: rotr12_from_nibbles(base),
            d2: rotr_bytes(&cols4(base + round::X3), 1),
            b2: rotr7_from_top_bits(base),
        };
        state[ia] = SRef::of_bytes(cols4(base + round::A2));
        state[ib] = SRef::of_bytes(wire.b2.clone());
        state[ic] = SRef::of_bytes(cols4(base + round::C2));
        state[id] = SRef::of_bytes(wire.d2.clone());
        gs.push(wire);
    }
    RoundWires { gs, out: state }
}

/// The round-chain token: `[tag, round, v0..v3 words, v4..v7 bytes, v8..v11
/// words, v12..v15 bytes, m0..m15]` — 58 elements. One function for every
/// sender and receiver, so their element orders cannot differ.
pub fn round_token(
    tag: Lin,
    rnd: Lin,
    words: [&Lin; 8],
    bytes: [&Bytes; 8],
    msg: &[Lin; 16],
) -> Vec<BusValue> {
    let mut v = vec![tag.bus(), rnd.bus()];
    v.extend(words[..4].iter().map(|w| w.bus()));
    for b in &bytes[..4] {
        v.extend(b.iter().map(Lin::bus));
    }
    v.extend(words[4..].iter().map(|w| w.bus()));
    for b in &bytes[4..] {
        v.extend(b.iter().map(Lin::bus));
    }
    v.extend(msg.iter().map(Lin::bus));
    v
}

fn xor_tuple(a: &Lin, b: &Lin, c: &Lin) -> Vec<BusValue> {
    vec![Lin::konst(OP_XOR).bus(), a.bus(), b.bus(), c.bus()]
}

fn split_tuple(a: &Lin, b: &Lin, lo: &Lin, hi: &Lin) -> Vec<BusValue> {
    vec![
        Lin::konst(OP_SPLIT).bus(),
        a.bus(),
        b.bus(),
        lo.bus(),
        hi.bus(),
    ]
}

pub fn round_interactions() -> Vec<BusInteraction> {
    let w = round_wires();
    let mu = || Multiplicity::Column(round::MU);
    let mut out = Vec::with_capacity(130);

    // The round chain: receive this round's input, send the next round's.
    let in_words: [Lin; 8] = core::array::from_fn(|i| {
        if i < 4 {
            Lin::col(round::IA + i)
        } else {
            Lin::col(round::IC + i - 4)
        }
    });
    let in_bytes: [Bytes; 8] = core::array::from_fn(|i| {
        if i < 4 {
            cols4(round::IB + 4 * i)
        } else {
            cols4(round::ID + 4 * (i - 4))
        }
    });
    let msg: [Lin; 16] = core::array::from_fn(|k| Lin::col(round::M + k));
    out.push(BusInteraction::receiver(
        ROUND_BUS,
        mu(),
        round_token(
            Lin::col(round::TAG),
            Lin::col(round::RND),
            core::array::from_fn(|i| &in_words[i]),
            core::array::from_fn(|i| &in_bytes[i]),
            &msg,
        ),
    ));
    let out_words: [Lin; 8] =
        core::array::from_fn(|i| w.out[if i < 4 { i } else { i + 4 }].word.clone());
    let out_bytes: [Bytes; 8] =
        core::array::from_fn(|i| w.out[if i < 4 { 4 + i } else { 8 + i }].bytes().clone());
    let next_msg: [Lin; 16] =
        core::array::from_fn(|i| Lin::col(round::M + BLAKE3_MSG_PERMUTATION[i]));
    out.push(BusInteraction::sender(
        ROUND_BUS,
        mu(),
        round_token(
            Lin::col(round::TAG),
            Lin::col(round::RND).plus(&Lin::konst(1)),
            core::array::from_fn(|i| &out_words[i]),
            core::array::from_fn(|i| &out_bytes[i]),
            &next_msg,
        ),
    ));

    // The 16 lookups of each G, in step order.
    for g in &w.gs {
        let a1 = cols4(g.base + round::A1);
        let x1 = cols4(g.base + round::X1);
        let c1 = cols4(g.base + round::C1);
        let a2 = cols4(g.base + round::A2);
        let x3 = cols4(g.base + round::X3);
        let c2 = cols4(g.base + round::C2);
        let z = cols4(g.base + round::Z);
        for k in 0..4 {
            out.push(BusInteraction::sender(
                XOR_BUS,
                mu(),
                xor_tuple(&g.d[k], &a1[k], &x1[k]),
            ));
        }
        for (k, (bk, ck)) in g.b.iter().zip(&c1).enumerate() {
            out.push(BusInteraction::sender(
                XOR_BUS,
                mu(),
                split_tuple(
                    bk,
                    ck,
                    &Lin::col(g.base + round::P2 + 2 * k),
                    &Lin::col(g.base + round::P2 + 2 * k + 1),
                ),
            ));
        }
        for k in 0..4 {
            out.push(BusInteraction::sender(
                XOR_BUS,
                mu(),
                xor_tuple(&g.d1[k], &a2[k], &x3[k]),
            ));
        }
        for k in 0..4 {
            out.push(BusInteraction::sender(
                XOR_BUS,
                mu(),
                xor_tuple(&g.b1[k], &c2[k], &z[k]),
            ));
        }
    }
    out
}

/// Constraint groups a test can switch off, to show each is load-bearing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Drops {
    /// The cubic implicit-carry identity of every add3.
    pub add3_carry: bool,
    /// The top bits' booleanity.
    pub top_bits: bool,
}

/// `B3R_ROUND`'s constraints, ungated (a zero row satisfies all): per G, two
/// implicit add3 carries (degree 3), two add2 carries (degree 2) and four
/// top-bit booleanities (degree 2). 64 a row.
#[derive(Clone, Copy, Debug, Default)]
pub struct RoundConstraints {
    pub drops: Drops,
}

pub const ROUND_CONSTRAINTS: usize = 8 * 8;

impl ConstraintSet<F, E> for RoundConstraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let w = round_wires();
        let mut idx = 0usize;
        let add3 = |b: &mut B, idx: &mut usize, s: Lin, drop: bool| {
            let s = s.expr(b);
            let e = if drop {
                b.zero()
            } else {
                s.clone() * (s.clone() - b.const_signed(TWO_32)) * (s - b.const_signed(2 * TWO_32))
            };
            b.emit_base(*idx, e);
            *idx += 1;
        };
        let add2 = |b: &mut B, idx: &mut usize, s: Lin| {
            let s = s.expr(b);
            let e = s.clone() * (s - b.const_signed(TWO_32));
            b.emit_base(*idx, e);
            *idx += 1;
        };
        for g in &w.gs {
            let a1 = word_of(&cols4(g.base + round::A1));
            let c1 = word_of(&cols4(g.base + round::C1));
            let a2 = word_of(&cols4(g.base + round::A2));
            let c2 = word_of(&cols4(g.base + round::C2));
            // a' = a + b + m0, c' = c + d', a'' = a' + b' + m1, c'' = c' + d''.
            let s1 =
                g.a.clone()
                    .plus(&word_of(&g.b))
                    .plus(&g.mx)
                    .plus(&a1.scaled(-1));
            add3(b, &mut idx, s1, self.drops.add3_carry);
            add2(
                b,
                &mut idx,
                g.c.clone().plus(&word_of(&g.d1)).plus(&c1.scaled(-1)),
            );
            let s3 = a1.plus(&word_of(&g.b1)).plus(&g.my).plus(&a2.scaled(-1));
            add3(b, &mut idx, s3, self.drops.add3_carry);
            add2(b, &mut idx, c1.plus(&word_of(&g.d2)).plus(&c2.scaled(-1)));
            for k in 0..4 {
                let t = b.main(0, g.base + round::T + k);
                let e = if self.drops.top_bits {
                    b.zero()
                } else {
                    t.clone() * (t - b.one())
                };
                b.emit_base(idx, e);
                idx += 1;
            }
        }
        debug_assert_eq!(idx, ROUND_CONSTRAINTS);
    }
}

// =========================================================================
// The head row
// =========================================================================

pub mod head {
    //! `B3R_HEAD` columns: a preprocessed instruction group, then the block,
    //! the chaining value, the final state, the digest and the canonicity
    //! witnesses.

    pub const MU: usize = 0;
    pub const TAG: usize = 1;
    /// `h` is the IV (else a chaining value read from memory).
    pub const IVF: usize = 2;
    /// The message is 16 `u32` lanes of 4 memory words (a Merkle parent).
    pub const MDIG: usize = 3;
    /// The message is 8 big-endian felts of 2 memory words (a leaf block).
    pub const MFELT: usize = 4;
    pub const BLEN: usize = 5;
    pub const FLAGS: usize = 6;
    pub const HADDR: usize = 7; // ..9
    pub const MADDR: usize = 9; // ..13
    pub const OADDR: usize = 13; // ..15
    pub const OCNT: usize = 15; // ..17
    pub const PREP_WIDTH: usize = 17;

    /// The 64 block bytes, in message order.
    pub const MB: usize = PREP_WIDTH;
    /// `h[0..4]` as 16-bit limbs: `h[i] = HL[2i] + 2^16·HL[2i+1]`.
    pub const HL: usize = MB + 64;
    /// `h[4..8]` as bytes.
    pub const HB: usize = HL + 8;
    /// The state after 7 rounds, 16 words of 4 bytes.
    pub const FS: usize = HB + 16;
    /// The digest, 8 words of 4 bytes.
    pub const OB: usize = FS + 64;
    /// Per felt: the "high half is maximal" flag, then the inverse witness.
    pub const CZ: usize = OB + 32;
    pub const CG: usize = CZ + 8;
    pub const NUM_COLUMNS: usize = CG + 8;
    pub const MAIN_COLUMNS: usize = NUM_COLUMNS - PREP_WIDTH;
}

const MAX_HALF: i64 = 0xFFFF_FFFF;

/// Message word `i` (little-endian over the block bytes).
fn m_word(i: usize) -> Lin {
    word_of(&cols4(head::MB + 4 * i))
}

/// Felt `f` of a leaf block: its 8 bytes big-endian.
fn felt_value(f: usize) -> Lin {
    (0..8).fold(Lin::default(), |acc, k| {
        acc.plus(&Lin::col(head::MB + 8 * f + k).scaled(1i64 << (8 * (7 - k))))
    })
}

/// The high (`half = 0`) or low (`half = 1`) `u32` of leaf felt `f`.
fn felt_half(f: usize, half: usize) -> Lin {
    (0..4).fold(Lin::default(), |acc, k| {
        acc.plus(&Lin::col(head::MB + 8 * f + 4 * half + k).scaled(1i64 << (8 * (3 - k))))
    })
}

fn h_word(i: usize) -> Lin {
    if i < 4 {
        Lin::col(head::HL + 2 * i).plus(&Lin::col(head::HL + 2 * i + 1).scaled(1 << 16))
    } else {
        word_of(&cols4(head::HB + 4 * (i - 4)))
    }
}

fn fs_bytes(i: usize) -> Bytes {
    cols4(head::FS + 4 * i)
}

/// `Q_r`: round `r`'s schedule as indices into the block's message,
/// `m_r[i] = m[Q_r[i]]` (`m_{r+1}[i] = m_r[P[i]]`).
pub fn schedule(r: usize) -> [usize; 16] {
    let mut q: [usize; 16] = core::array::from_fn(|i| i);
    for _ in 0..r {
        let prev = q;
        for (i, &p) in BLAKE3_MSG_PERMUTATION.iter().enumerate() {
            q[i] = prev[p];
        }
    }
    q
}

fn mem_token(addr: usize, lanes: [Lin; 4]) -> Vec<BusValue> {
    let mut v = vec![Lin::col(addr).bus()];
    v.extend(lanes.iter().map(Lin::bus));
    v
}

pub fn head_interactions() -> Vec<BusInteraction> {
    let mu = || Multiplicity::Column(head::MU);
    let mut out = Vec::with_capacity(84);

    // The round-0 token: h, IV, (t = 0, block_len, flags), the block.
    let h_words: [Lin; 8] = core::array::from_fn(|i| {
        if i < 4 {
            h_word(i)
        } else {
            Lin::konst(BLAKE3_IV[i - 4] as i64)
        }
    });
    let mut v12: [Bytes; 4] = core::array::from_fn(|_| konst_bytes(0));
    v12[2][0] = Lin::col(head::BLEN);
    v12[3][0] = Lin::col(head::FLAGS);
    let h_bytes: [Bytes; 8] = core::array::from_fn(|i| {
        if i < 4 {
            cols4(head::HB + 4 * i)
        } else {
            v12[i - 4].clone()
        }
    });
    let msg0: [Lin; 16] = core::array::from_fn(m_word);
    out.push(BusInteraction::sender(
        ROUND_BUS,
        mu(),
        round_token(
            Lin::col(head::TAG),
            Lin::konst(0),
            core::array::from_fn(|i| &h_words[i]),
            core::array::from_fn(|i| &h_bytes[i]),
            &msg0,
        ),
    ));
    // The round-7 token: the final state, the message after seven permutations.
    let fs_words: [Lin; 8] =
        core::array::from_fn(|i| word_of(&fs_bytes(if i < 4 { i } else { i + 4 })));
    let fs_b: [Bytes; 8] = core::array::from_fn(|i| fs_bytes(if i < 4 { 4 + i } else { 8 + i }));
    let q7 = schedule(ROUNDS);
    let msg7: [Lin; 16] = core::array::from_fn(|i| m_word(q7[i]));
    out.push(BusInteraction::receiver(
        ROUND_BUS,
        mu(),
        round_token(
            Lin::col(head::TAG),
            Lin::konst(ROUNDS as i64),
            core::array::from_fn(|i| &fs_words[i]),
            core::array::from_fn(|i| &fs_b[i]),
            &msg7,
        ),
    ));

    // Range: the 64 block bytes in pairs; `h[0..4]`'s limbs as 16-bit values.
    for j in 0..32 {
        out.push(BusInteraction::sender(
            XOR_BUS,
            mu(),
            vec![
                Lin::konst(OP_PAIR).bus(),
                Lin::col(head::MB + 2 * j).bus(),
                Lin::col(head::MB + 2 * j + 1).bus(),
            ],
        ));
    }
    for k in 0..8 {
        out.push(BusInteraction::sender(
            XOR_BUS,
            mu(),
            vec![Lin::konst(OP_R16).bus(), Lin::col(head::HL + k).bus()],
        ));
    }
    // The feed-forward of a 256-bit digest.
    for i in 0..8 {
        let (lo, hi, o) = (fs_bytes(i), fs_bytes(i + 8), cols4(head::OB + 4 * i));
        for k in 0..4 {
            out.push(BusInteraction::sender(
                XOR_BUS,
                mu(),
                xor_tuple(&lo[k], &hi[k], &o[k]),
            ));
        }
    }

    // Memory: the chaining value (unless IV), the block in one of its two
    // modes, and the digest's two words.
    for w in 0..2 {
        out.push(BusInteraction::receiver(
            BusId::LfmMem,
            Multiplicity::Diff(head::MU, head::IVF),
            mem_token(head::HADDR + w, core::array::from_fn(|l| h_word(4 * w + l))),
        ));
    }
    for w in 0..4 {
        out.push(BusInteraction::receiver(
            BusId::LfmMem,
            Multiplicity::Column(head::MDIG),
            mem_token(head::MADDR + w, core::array::from_fn(|l| m_word(4 * w + l))),
        ));
    }
    for w in 0..2 {
        out.push(BusInteraction::receiver(
            BusId::LfmMem,
            Multiplicity::Column(head::MFELT),
            mem_token(
                head::MADDR + w,
                core::array::from_fn(|l| felt_value(4 * w + l)),
            ),
        ));
    }
    for w in 0..2 {
        out.push(BusInteraction::sender(
            BusId::LfmMem,
            Multiplicity::Column(head::OCNT + w),
            mem_token(
                head::OADDR + w,
                core::array::from_fn(|l| word_of(&cols4(head::OB + 4 * (4 * w + l)))),
            ),
        ));
    }
    out
}

/// `B3R_HEAD`'s constraints: the IV under its flag (24, degree 2) and leaf
/// felt canonicity under the leaf mode (24, degree 3).
#[derive(Clone, Copy, Debug, Default)]
pub struct HeadConstraints;

pub const HEAD_CONSTRAINTS: usize = 24 + 24;

impl ConstraintSet<F, E> for HeadConstraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let mut idx = 0usize;
        let ivf = || Lin::col(head::IVF);
        for (i, &iv) in BLAKE3_IV[..4].iter().enumerate() {
            for (k, limb) in [iv & 0xFFFF, iv >> 16].into_iter().enumerate() {
                let d = Lin::col(head::HL + 2 * i + k).plus(&Lin::konst(-(limb as i64)));
                let e = ivf().expr(b) * d.expr(b);
                b.emit_base(idx, e);
                idx += 1;
            }
        }
        for i in 0..4 {
            let iv = BLAKE3_IV[4 + i];
            for k in 0..4 {
                let byte = ((iv >> (8 * k)) & 0xFF) as i64;
                let d = Lin::col(head::HB + 4 * i + k).plus(&Lin::konst(-byte));
                let e = ivf().expr(b) * d.expr(b);
                b.emit_base(idx, e);
                idx += 1;
            }
        }
        // `v < p ⟺ ¬(hi = 2³²−1 ∧ lo ≠ 0)` for halves already known u32:
        // the socket's Z/GINV idiom (`blake3_socket.rs`, LEAF mode).
        for f in 0..8 {
            let mode = b.main(0, head::MFELT);
            let z = b.main(0, head::CZ + f);
            let ginv = b.main(0, head::CG + f);
            let g = Lin::konst(MAX_HALF)
                .plus(&felt_half(f, 0).scaled(-1))
                .expr(b);
            let lo = felt_half(f, 1).expr(b);
            b.emit_base(idx, mode.clone() * z.clone() * g.clone());
            let one = b.one();
            b.emit_base(idx + 1, mode.clone() * (one - z.clone() - g * ginv));
            b.emit_base(idx + 2, mode * z * lo);
            idx += 3;
        }
        debug_assert_eq!(idx, HEAD_CONSTRAINTS);
    }
}

// =========================================================================
// The fixed table
// =========================================================================

pub mod table {
    //! `B3R_XOR`: row `a + 256·b`, preprocessed `[A, B, X, LO, HI]`, then one
    //! multiplicity column per op.
    pub const A: usize = 0;
    pub const B: usize = 1;
    pub const X: usize = 2;
    pub const LO: usize = 3;
    pub const HI: usize = 4;
    pub const PREP_WIDTH: usize = 5;
    pub const M_XOR: usize = 5;
    pub const M_SPLIT: usize = 6;
    pub const M_PAIR: usize = 7;
    pub const M_R16: usize = 8;
    pub const NUM_COLUMNS: usize = 9;
    pub const ROWS: usize = 1 << 16;
}

pub fn table_interactions() -> Vec<BusInteraction> {
    let c = Lin::col;
    vec![
        BusInteraction::receiver(
            XOR_BUS,
            Multiplicity::Column(table::M_XOR),
            xor_tuple(&c(table::A), &c(table::B), &c(table::X)),
        ),
        BusInteraction::receiver(
            XOR_BUS,
            Multiplicity::Column(table::M_SPLIT),
            split_tuple(&c(table::A), &c(table::B), &c(table::LO), &c(table::HI)),
        ),
        BusInteraction::receiver(
            XOR_BUS,
            Multiplicity::Column(table::M_PAIR),
            vec![
                Lin::konst(OP_PAIR).bus(),
                c(table::A).bus(),
                c(table::B).bus(),
            ],
        ),
        BusInteraction::receiver(
            XOR_BUS,
            Multiplicity::Column(table::M_R16),
            vec![
                Lin::konst(OP_R16).bus(),
                c(table::A).plus(&c(table::B).scaled(256)).bus(),
            ],
        ),
    ]
}

/// Per-row lookup counts for the table, by op.
#[derive(Clone)]
pub struct TableHist {
    pub counts: [Vec<u64>; 4],
}

impl Default for TableHist {
    fn default() -> Self {
        TableHist {
            counts: core::array::from_fn(|_| vec![0u64; table::ROWS]),
        }
    }
}

impl TableHist {
    #[inline]
    fn add(&mut self, op: usize, a: u32, b: u32) {
        self.counts[op][(a as usize) | ((b as usize) << 8)] += 1;
    }

    pub fn merge(mut self, other: &TableHist) -> Self {
        for (mine, theirs) in self.counts.iter_mut().zip(&other.counts) {
            for (m, t) in mine.iter_mut().zip(theirs) {
                *m += t;
            }
        }
        self
    }

    pub fn total(&self) -> u64 {
        self.counts.iter().flatten().sum()
    }
}

pub fn table_trace(hist: &TableHist) -> TraceTable<F, E> {
    let mut data = zeroed_fe_vec(table::ROWS * table::NUM_COLUMNS);
    for r in 0..table::ROWS {
        let (a, b) = ((r & 0xFF) as u64, (r >> 8) as u64);
        let x = a ^ b;
        let row = &mut data[r * table::NUM_COLUMNS..(r + 1) * table::NUM_COLUMNS];
        row[table::A] = FE::from(a);
        row[table::B] = FE::from(b);
        row[table::X] = FE::from(x);
        row[table::LO] = FE::from(x & 15);
        row[table::HI] = FE::from(x >> 4);
        for op in 0..4 {
            row[table::M_XOR + op] = FE::from(hist.counts[op][r]);
        }
    }
    TraceTable::new_main(data, table::NUM_COLUMNS, 1)
}

pub fn table_prep_columns() -> Vec<Vec<FE>> {
    (0..table::PREP_WIDTH)
        .map(|c| {
            (0..table::ROWS)
                .map(|r| {
                    let (a, b) = ((r & 0xFF) as u64, (r >> 8) as u64);
                    let x = a ^ b;
                    FE::from([a, b, x, x & 15, x >> 4][c])
                })
                .collect()
        })
        .collect()
}

// =========================================================================
// Operations and the witness
// =========================================================================

/// One compression as a program would issue it.
#[derive(Clone, Debug)]
pub struct LeanOp {
    /// The chaining value; ignored (and the IV used) when `iv`.
    pub h: [u32; 8],
    pub iv: bool,
    /// The block, zero-padded.
    pub block: [u8; 64],
    pub block_len: u32,
    pub flags: u32,
    /// The block is 8 big-endian felts read as 2 memory words, else 16 `u32`
    /// lanes read as 4.
    pub felt_mode: bool,
    pub h_addr: [u64; 2],
    pub m_addr: [u64; 4],
    pub out_addr: [u64; 2],
    pub out_cnt: [u64; 2],
}

impl LeanOp {
    pub fn chaining_value(&self) -> [u32; 8] {
        if self.iv { BLAKE3_IV } else { self.h }
    }

    pub fn message(&self) -> [u32; 16] {
        core::array::from_fn(|i| {
            u32::from_le_bytes(self.block[4 * i..4 * i + 4].try_into().expect("4 bytes"))
        })
    }

    /// The 256-bit digest: the compression's low eight output words.
    pub fn digest(&self) -> [u32; 8] {
        let out = blake3_compress_rounds(
            &self.chaining_value(),
            &self.message(),
            0,
            self.block_len,
            self.flags,
            ROUNDS,
        );
        core::array::from_fn(|i| out[i])
    }

    /// Leaf mode's 8 felts, as the memory words carry them.
    pub fn felts(&self) -> [u64; 8] {
        core::array::from_fn(|f| {
            u64::from_be_bytes(self.block[8 * f..8 * f + 8].try_into().expect("8 bytes"))
        })
    }
}

fn set_bytes(row: &mut [FE], col: usize, w: u32) {
    for k in 0..4 {
        row[col + k] = FE::from(u64::from((w >> (8 * k)) & 0xFF));
    }
}

/// A deliberate fault one witness generator commits, at one G of one round:
/// the negative tests' compressions. Each is CONSISTENT downstream — every
/// later value is computed from the faulty one — so the only thing it breaks
/// is the one relation it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultKind {
    /// `b' = x2 ⋙ 13` where BLAKE3 rotates by 12.
    Rot13,
    /// `b'' = z ⋙ 6` where BLAKE3 rotates by 7.
    Rot6,
    /// `a' = a + b + m₀ + 1`: an add3 whose implied carry is not 0, 1 or 2.
    Carry3,
    /// `c' = c + d' + 1`: an add2 whose implied carry is not 0 or 1.
    Carry2,
    /// `b''` with its low bit flipped, its "top bits" solved as field elements
    /// so that every rebuilt byte is the forged one: what booleanity stops.
    ForgedB2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fault {
    pub round: usize,
    pub g: usize,
    pub kind: FaultKind,
}

/// The four "top bits" that rebuild `forged` from `z` through
/// `b''_i = 2·z_{i+1} − 256·t_{i+1} + t_i`, as field elements. For the honest
/// `b''` they are the true bits (the system has one solution).
pub fn solve_top_bits(z: u32, forged: u32) -> [FE; 4] {
    let byte = |w: u32, k: usize| FE::from(u64::from((w >> (8 * (k % 4))) & 0xFF));
    let c: [FE; 4] = core::array::from_fn(|i| byte(forged, i) - FE::from(2u64) * byte(z, i + 1));
    let k256 = FE::from(256u64);
    let num = c[0] + k256 * (c[1] + k256 * (c[2] + k256 * c[3]));
    let t0 = num
        * (FE::one() - FE::from(1u64 << 32))
            .inv()
            .expect("1 − 2^32 ≠ 0");
    let t3 = c[3] + k256 * t0;
    let t2 = c[2] + k256 * t3;
    let t1 = c[1] + k256 * t2;
    [t0, t1, t2, t3]
}

/// Write one G's 40 cells and return `(a'', b'', c'', d'')`, recording its 16
/// lookups.
#[allow(clippy::too_many_arguments)]
fn fill_g(
    row: &mut [FE],
    base: usize,
    (a, b, c, d): (u32, u32, u32, u32),
    mx: u32,
    my: u32,
    fault: Option<FaultKind>,
    hist: &mut TableHist,
) -> (u32, u32, u32, u32) {
    let is = |k: FaultKind| fault == Some(k);
    let a1 = a
        .wrapping_add(b)
        .wrapping_add(mx)
        .wrapping_add(u32::from(is(FaultKind::Carry3)));
    let x1 = d ^ a1;
    let d1 = x1.rotate_right(16);
    let c1 = c
        .wrapping_add(d1)
        .wrapping_add(u32::from(is(FaultKind::Carry2)));
    let x2 = b ^ c1;
    let b1 = x2.rotate_right(if is(FaultKind::Rot13) { 13 } else { 12 });
    let a2 = a1.wrapping_add(b1).wrapping_add(my);
    let x3 = d1 ^ a2;
    let d2 = x3.rotate_right(8);
    let c2 = c1.wrapping_add(d2);
    let z = b1 ^ c2;
    let mut b2 = z.rotate_right(if is(FaultKind::Rot6) { 6 } else { 7 });

    set_bytes(row, base + round::A1, a1);
    set_bytes(row, base + round::X1, x1);
    set_bytes(row, base + round::C1, c1);
    for k in 0..4 {
        let byte = (x2 >> (8 * k)) & 0xFF;
        row[base + round::P2 + 2 * k] = FE::from(u64::from(byte & 15));
        row[base + round::P2 + 2 * k + 1] = FE::from(u64::from(byte >> 4));
    }
    set_bytes(row, base + round::A2, a2);
    set_bytes(row, base + round::X3, x3);
    set_bytes(row, base + round::C2, c2);
    set_bytes(row, base + round::Z, z);
    for k in 0..4 {
        row[base + round::T + k] = FE::from(u64::from((z >> (8 * k + 7)) & 1));
    }
    if is(FaultKind::ForgedB2) {
        b2 ^= 1;
        for (k, t) in solve_top_bits(z, b2).into_iter().enumerate() {
            row[base + round::T + k] = t;
        }
    }

    let byte = |w: u32, k: usize| (w >> (8 * k)) & 0xFF;
    for k in 0..4 {
        hist.add(OP_XOR as usize, byte(d, k), byte(a1, k));
    }
    for k in 0..4 {
        hist.add(OP_SPLIT as usize, byte(b, k), byte(c1, k));
    }
    for k in 0..4 {
        hist.add(OP_XOR as usize, byte(d1, k), byte(a2, k));
    }
    for k in 0..4 {
        hist.add(OP_XOR as usize, byte(b1, k), byte(c2, k));
    }
    (a2, b2, c2, d2)
}

/// Fill one compression's `rounds` round rows (`rows`, row-major,
/// `round::NUM_COLUMNS` wide, the preprocessed prefix included) and return the
/// final state. `rounds` is [`ROUNDS`] for every compression but a negative
/// test's.
pub fn fill_rounds(
    rows: &mut [FE],
    tag: u64,
    op: &LeanOp,
    fault: Option<Fault>,
    rounds: usize,
    hist: &mut TableHist,
) -> [u32; 16] {
    let h = op.chaining_value();
    let mut v: [u32; 16] = [
        h[0],
        h[1],
        h[2],
        h[3],
        h[4],
        h[5],
        h[6],
        h[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        0,
        0,
        op.block_len,
        op.flags,
    ];
    let m = op.message();
    for r in 0..rounds {
        let row = &mut rows[r * round::NUM_COLUMNS..(r + 1) * round::NUM_COLUMNS];
        row[round::MU] = FE::one();
        row[round::TAG] = FE::from(tag);
        row[round::RND] = FE::from(r as u64);
        for i in 0..4 {
            row[round::IA + i] = FE::from(u64::from(v[i]));
            set_bytes(row, round::IB + 4 * i, v[4 + i]);
            row[round::IC + i] = FE::from(u64::from(v[8 + i]));
            set_bytes(row, round::ID + 4 * i, v[12 + i]);
        }
        let q = schedule(r);
        let mr: [u32; 16] = core::array::from_fn(|i| m[q[i]]);
        for (k, &w) in mr.iter().enumerate() {
            row[round::M + k] = FE::from(u64::from(w));
        }
        for (j, &(ia, ib, ic, id)) in G_INDICES.iter().enumerate() {
            let here = fault.filter(|f| f.round == r && f.g == j).map(|f| f.kind);
            let (a, b, c, d) = fill_g(
                row,
                round::g_base(j),
                (v[ia], v[ib], v[ic], v[id]),
                mr[2 * j],
                mr[2 * j + 1],
                here,
                hist,
            );
            v[ia] = a;
            v[ib] = b;
            v[ic] = c;
            v[id] = d;
        }
    }
    v
}

/// Fill one head row from the op and its final state; records the head's
/// lookups.
pub fn fill_head(row: &mut [FE], tag: u64, op: &LeanOp, fs: &[u32; 16], hist: &mut TableHist) {
    row[head::MU] = FE::one();
    row[head::TAG] = FE::from(tag);
    row[head::IVF] = FE::from(u64::from(op.iv));
    row[head::MDIG] = FE::from(u64::from(!op.felt_mode));
    row[head::MFELT] = FE::from(u64::from(op.felt_mode));
    row[head::BLEN] = FE::from(u64::from(op.block_len));
    row[head::FLAGS] = FE::from(u64::from(op.flags));
    for w in 0..2 {
        row[head::HADDR + w] = FE::from(op.h_addr[w]);
        row[head::OADDR + w] = FE::from(op.out_addr[w]);
        row[head::OCNT + w] = FE::from(op.out_cnt[w]);
    }
    for w in 0..4 {
        row[head::MADDR + w] = FE::from(op.m_addr[w]);
    }
    for (k, &byte) in op.block.iter().enumerate() {
        row[head::MB + k] = FE::from(u64::from(byte));
    }
    let h = op.chaining_value();
    for i in 0..4 {
        row[head::HL + 2 * i] = FE::from(u64::from(h[i] & 0xFFFF));
        row[head::HL + 2 * i + 1] = FE::from(u64::from(h[i] >> 16));
        set_bytes(row, head::HB + 4 * i, h[4 + i]);
    }
    for (i, &w) in fs.iter().enumerate() {
        set_bytes(row, head::FS + 4 * i, w);
    }
    for i in 0..8 {
        set_bytes(row, head::OB + 4 * i, fs[i] ^ fs[i + 8]);
    }
    for f in 0..8 {
        let hi = u32::from_be_bytes(op.block[8 * f..8 * f + 4].try_into().expect("4 bytes"));
        let g = u64::from(0xFFFF_FFFFu32 - hi);
        if g == 0 {
            row[head::CZ + f] = FE::one();
        } else {
            row[head::CG + f] = FE::from(g).inv().expect("nonzero inverts");
        }
    }

    for j in 0..32 {
        hist.add(
            OP_PAIR as usize,
            u32::from(op.block[2 * j]),
            u32::from(op.block[2 * j + 1]),
        );
    }
    for &hi in &h[..4] {
        for limb in [hi & 0xFFFF, hi >> 16] {
            hist.add(OP_R16 as usize, limb & 0xFF, limb >> 8);
        }
    }
    for i in 0..8 {
        for k in 0..4 {
            hist.add(
                OP_XOR as usize,
                (fs[i] >> (8 * k)) & 0xFF,
                (fs[i + 8] >> (8 * k)) & 0xFF,
            );
        }
    }
}

/// Chunk sizes for `rows` round rows: the largest power of two at or under the
/// remainder, capped at `2^max_log`; a remainder under `2^min_log` is padded up
/// to it. So `7·N` rows split with no padding at all whenever `N`'s binary
/// expansion allows (`7 · 2^k = 2^(k+2) + 2^(k+1) + 2^k`).
pub fn chunk_rows(rows: usize, max_log: u32, min_log: u32) -> Vec<usize> {
    let mut left = rows;
    let mut out = Vec::new();
    while left > 0 {
        let p = (usize::BITS - 1 - left.leading_zeros()).min(max_log);
        let size = if p < min_log { 1 << min_log } else { 1 << p };
        out.push(size);
        left = left.saturating_sub(size);
    }
    out
}
