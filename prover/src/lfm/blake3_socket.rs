//! The BLAKE3 arm of `LFM_HASH` — the Option-A 2-to-1 compress socket.
//!
//! This is Route A of `thoughts/shared/lfm-real-hash/PLAN.md` §3: BLAKE3 hosted
//! *behind* the frozen `LFM_HASH` socket, exactly the way Poseidon is. The chip
//! count stays 14 and the 28-column shared value prefix keeps its offsets, so
//! the `LFM_HASH` tuple contract is untouched and everything BLAKE3 witnesses is
//! appended after the prefix. `PREP_WIDTH` is 13 — the transcript selector
//! (option B1) took it from 11 to 12 and the leaf selector (option C) to 13,
//! each moving every preprocessed root and every registered program's digest
//! once, in one re-bless.
//!
//! # What one row proves
//!
//! One row = one compression, in one of THREE domains, specified byte-level in
//! `thoughts/blake3/socket-kats/SOCKET.md` §2.1 and word-level in §2.2, at the
//! leaf RATE of `block-compression/commit-spec/COMMIT.md` §1.2:
//!
//! ```text
//! msg    = LE32(lane0..lane11) ‖ tag                        (52 bytes)
//! digest = BLAKE3(msg)[0..16]                               (128 bits, 1 cell)
//! ```
//!
//! | tag | row | the twelve lanes are |
//! |---|---|---|
//! | `"LFMC"` | Merkle parent / 2-to-1 compress | two digest cells, then four zeros |
//! | `"LFMT"` | a Fiat–Shamir transcript step | state ‖ operand ‖ four zeros |
//! | `"LFML"` | a **leaf** over four field elements | the chaining accumulator, then the felts' `lo`/`hi` halves |
//!
//! 52 bytes being one block, that is exactly one compression with `h = IV` (all
//! eight words), `m[0..12] = the lanes`, `m[12] = tag` as a little-endian `u32`,
//! `m[13..16] = 0`, `t = 0`, `block_len = 52`,
//! `flags = CHUNK_START|CHUNK_END|ROOT`, and the digest the LOW four output
//! words. **The three domains differ in `m[12]` and in nothing else**, so one
//! mixing core and one column layout serve all three.
//!
//! # The leaf RATE — why twelve lanes and not eight
//!
//! A leaf row absorbs **four felts and chains an accumulator in ONE
//! compression** (COMMIT.md §1.2): the accumulator cell rides in the message
//! rather than in `h`, so there is no separate fold. That is 4 felts per
//! compression against the 2 the accumulator-free row reached once its digest
//! had to be folded into a chain by an `"LFMC"` parent, and leaf absorption is
//! ~70% of a recursion tower node's bill.
//!
//! The lanes it costs are free of witness columns on the digest modes: lanes
//! 8–11 read `IN8..IN12`, the THIRD input cell, which
//! `chips::hash::emit_unread_input_pins` already pins to zero on every row that
//! does not read it. So a compress row's four new message words are forced to
//! zero by constraints that were already there — see the lane block in
//! [`eval`], and note that twelve is the last lane count for which this holds
//! (at thirteen, `IN0 + 12` is `S8` and the identity would start reading the
//! capacity state as an input felt).
//!
//! **At [`SOCKET_ROUNDS`] = 7 that is literally `blake3::hash(lanes ‖ tag)`,**
//! so the socket has a direct external anchor and needs no oracle in the chain —
//! and the transcript and leaf domains inherit that anchor unchanged, because
//! the tag is the only thing that moved. That is the whole reason the domain tag lives in the
//! *message* rather than in `flags`, `t` or `h`: a tag anywhere else would make
//! even the 7-round socket a nonstandard invocation of `f` that no library
//! computes, throwing the anchor away for nothing (SOCKET.md §2.3).
//!
//! The tag word is a linear form over the three PREPROCESSED mode columns rather than
//! a compile-time constant, which keeps it prover-unchosen and free — see
//! [`TAG_SELECTOR`].
//!
//! # The LEAF mode, and the one thing not to conclude from it
//!
//! A leaf row reads TWO cells: a chaining accumulator, which is an ordinary
//! digest cell and fills lanes 0–3, and four arbitrary Goldilocks elements, each
//! split into a `lo`/`hi` `u32` pair so that eight halves fill lanes 4–11.
//! `p − 1 = 0xFFFFFFFF_00000000`, so for halves already known to be `u32`:
//!
//! ```text
//! v < p   <==>   NOT( hi = 2^32−1  AND  lo >= 1 )
//! ```
//!
//! — "if `hi` is maximal then `lo` is zero", which is two witness columns and
//! four constraints per felt rather than a 64-bit decomposition. Without it one
//! field element would have TWO half-encodings and therefore two leaf digests,
//! which is a collision in the felt→digest map and exactly what a Merkle tree
//! must not have.
//!
//! ⚠ **The canonicity block ASSUMES the `u32` bound; it does not ESTABLISH it.**
//! `lo` and `hi` are ordinary input lanes, so the bound comes from the same O1
//! machinery as every other lane: byte columns plus the `AreBytes` sends. That
//! is the whole reason this mode is cheap, and it is stated here because the
//! shape invites two opposite mistakes — adding a redundant range check on the
//! halves, or (far worse) **removing the lane identity or the `AreBytes` sends
//! on the theory that canonicity subsumes them. It does not.** With unbounded
//! halves, `hi = 2^32−1` stops being a reachable-and-detectable case and the
//! predicate above stops meaning `v < p` at all.
//!
//! # Why the socket is so much cheaper than the standalone chip
//!
//! [`super::blake3_chip`] is the syscall-shaped chip: 28 input `u32` words and
//! all 16 output words are committed columns. Here `h`, `t`, `block_len`,
//! `flags` and every message word above the lanes are **compile-time
//! constants**, and the truncation
//! window means only 4 of the 16 output words are ever built. What is left as
//! witness is 8 input lanes, the mixing core, and 4 output words.
//!
//! # The two soundness obligations this module discharges
//!
//! - **O1 — input lanes carry a committed byte decomposition.** A digest cell's
//!   lane is a Goldilocks felt over `[0, p)` with `p ≈ 2^64`, and
//!   `edsl::merkle_walk` feeds `compress` *arena-hinted* — that is,
//!   prover-chosen — sibling cells. The contract has two halves and **they buy
//!   different things**; conflating them is easy and is why this is spelled out.
//!
//!   *The mu-gated linear identity* `IN_lane = Σ MB[k]·2^{8k}` ties the felt to
//!   the bytes. Note what follows from it: the mixing core reads **the same
//!   linear form** as the message word (`message_word_ref`), so `IN_lane` and
//!   `m[lane]` are the same field element by construction. A lane therefore
//!   cannot be hashed as anything but itself, and the textbook alias — `v` and
//!   `v + 2^32` hashing alike — is **unconstructible here**, not merely
//!   prevented. (It is real for a chip that derives the message bytes by
//!   reduction mod 2^32 instead of by a checked decomposition, which is why the
//!   identity is the right shape; it is not what the `AreBytes` sends buy.)
//!
//!   *The `AreBytes` sends* are the message words' **only** range check —
//!   `m` reaches `add3` and nothing else, never an XOR, so unlike almost every
//!   other word in this design it gets no free byte bound from a lookup that
//!   consumes it. And `add3`'s exactness needs `m < 2^32`: in round 0 the `a`
//!   and `b` operands are compile-time constants and the output `s` is
//!   byte-bounded by the XOR that consumes it, so an unbounded `m` lets a
//!   prover solve `m ≡ s + 2^32·k − a − b (mod p)` for any chosen `s`, put the
//!   whole value in `MB[0]` with the other three bytes zero — satisfying the
//!   identity, since nothing bounds them — and hint the sibling cell to match.
//!   The first `add3`'s output, and hence the entire compression, would be
//!   prover-chosen.
//!
//!   Stated as the one mechanism, since this is the spot the argument keeps
//!   drifting: what the sends do is **transfer a bound onto the lane**. Without
//!   them the identity is satisfiable for *every* felt `IN_lane` — put the whole
//!   value in `MB[0]` — so it bounds nothing. With them the four bytes sum to
//!   less than `2^32`, so it is satisfiable exactly when `IN_lane < 2^32`, and
//!   then the decomposition is unique.
//!
//!   So: neither half alone suffices, and `blake3_socket_tests::
//!   the_lane_range_check_is_load_bearing_on_its_own` pins the separation by
//!   exhibiting a witness the eval set cannot see at all.
//! - **O3 — `compress_iv()` does not participate.** The IV enters through `h`,
//!   all eight words, not through the state's capacity lanes, so this arm
//!   overrides `compress` (and [`LfmHasher::compress_out`]) rather than
//!   inheriting the trait's permute-and-truncate default.
//!
//! # ✓ O5 — RETIRED, and enforced by the tag rather than by review
//!
//! The obligation was: leaves and parents must be domain-separated, or a
//! variable-depth tree admits the classic Merkle second-preimage confusion — an
//! internal node replayed as a leaf. It is now discharged **mechanically**. A
//! leaf digest is `BLAKE3(…‖"LFML")` and a parent is `BLAKE3(…‖"LFMC")`, so an
//! internal node cannot be replayed as a leaf whatever the tree's shape, and the
//! tag is selected by a preprocessed column the prover does not choose.
//!
//! What this replaced is worth recording, because it was weaker than it looked.
//! Programs formed leaf digests by compressing raw data rows under the SAME
//! `"LFMC"` tag as parents, so leaves and parents were not separated at all;
//! that was sound only because every eDSL circuit is fixed-shape at build time,
//! so a node at one level could not be replayed at another. Fixed depth remains
//! true of every current program and remains worth having, but **it is no longer
//! load-bearing for second-preimage resistance.**
//!
//! The reviewer's job shrinks accordingly: from "is this a leaf path, and does
//! the tree have fixed depth?" to *"is this row's mode right?"* — which the
//! registrar's one-hot check and controls M9/M10 answer.
//!
//! (BLAKE3's own `PARENT` flag was rejected for the split: it cannot be reused
//! without leaving the standard-hash framing that makes the crate a direct KAT.)
//!
//! Equally on the record: the digest is 128 bits, so this socket offers
//! **64-bit collision resistance** by the birthday bound. That follows from
//! `HASH_DIGEST_FELTS = 4` and the machine's declared 128-bit target — it is
//! not introduced by BLAKE3 or by the truncation window.
//!
//! # ✗ There is no `permute` socket, and there never will be
//!
//! `LFM_HASH` has four modes and this arm implements **three**. The `permute`
//! socket — 12 felts in, 12 out — is unspecified: it has no mapping decision,
//! no KATs, and its security argument is not the same argument as `compress`'s
//! (SOCKET.md §7). Rather than invent one, the AIR forces `MODE_P = 0`, so a
//! program containing a `permute` is *unprovable* under BLAKE3, and
//! [`Blake3Permutation`] rejects one at execution with a message saying why.
//!
//! Option B1 (ratified 2026-08-11) made that permanent by removing the only
//! reason to want one: the Fiat–Shamir sponge is a **compress chain**, not a
//! permutation duplex, so `edsl::SpongeVar` runs on this socket like everything
//! else and `MODE_P` stays pinned forever. Option C then gave leaves their own
//! mode on the same socket rather than a second one. The tag `"LFMP"` that was
//! reserved for the permute socket is retired unused.

use stark::constraints::builder::ConstraintBuilder;
use stark::lookup::{BusInteraction, BusValue, Multiplicity};

use crate::tables::bitwise::{BitwiseOperation, BitwiseOperationType};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField, alu_op};

use super::blake3::{BLAKE3_IV, BLAKE3_ROUNDS, blake3_compress_rounds};
use super::blake3_chip::{
    Add2Wire, Add3Wire, Blake3Flow, ByteRef, FlowConfig, ROT_SHIFT_R, RotWire, ValueFlow, WireFlow,
    WordRef, XorWire, half_expr, run_flow, word_cols, word_expr,
};
use super::chips::hash::NUM_UNREAD_INPUT_PINS;
use super::hash::{HASH_DIGEST_FELTS, HASH_STATE_FELTS, LfmHasher};
use super::instr::HashMode;
use super::word::LfmWord;

type F = GoldilocksField;
type E = GoldilocksExtension;

// =========================================================================
// The framing (SOCKET.md §2.2) — every one of these is a way to be wrong
// =========================================================================

/// Rounds the `LFM_HASH` BLAKE3 arm is compiled for.
///
/// An alias for [`BLAKE3_ROUNDS`] — 7 (standard BLAKE3) by default, 6 under the
/// `blake3-6round` feature. The socket and the standalone `LFM_BLAKE3` probe
/// share ONE knob on purpose: two would let a sweep leave the machine's hash and
/// the chip it is priced against describing different functions, and the whole
/// value of the probe is that the two are comparable.
pub const SOCKET_ROUNDS: usize = BLAKE3_ROUNDS;

/// Tripwire for the single-knob invariant.
///
/// Trivially true while [`SOCKET_ROUNDS`] is an alias — which is the point. The
/// invariant is enforced by that one `=` and by nothing else: `NUM_G == 8 *
/// SOCKET_ROUNDS` and `cols::OUT - cols::G == 60 * NUM_G` are each internally
/// consistent and would pass happily with the socket and the standalone probe
/// compiled at different round counts. That is not hypothetical — it is the
/// shape this tree had before the A6R flip, and re-introducing a second `cfg`
/// pair here is a silent pricing lie: the probe would measure one hash and the
/// machine would use another. This assertion is what fails instead.
const _: () = assert!(SOCKET_ROUNDS == BLAKE3_ROUNDS);

/// G-instances per compression: 8 per round.
pub const NUM_G: usize = SOCKET_ROUNDS * 8;

/// The domain tag `"LFMC"`, read as one little-endian `u32` — the message word
/// straight after the lanes, `m[NUM_LANES]`.
///
/// A tag is never reused for a second purpose, for the same reason
/// `HasherKind::as_tag` never reuses a discriminant. `"LFMT"` is the transcript
/// domain, `"LFML"` is the LEAF domain (live since option C), and `"LFMP"` is
/// RETIRED UNUSED — it was reserved for a permute socket that option B1 decided
/// never to build. Retired rather than deleted: freeing the value would let a
/// later allocation reuse it and create a domain nobody analysed.
pub const TAG_LFMC: u32 = u32::from_le_bytes(*b"LFMC");

/// The domain tag `"LFML"` — a Merkle LEAF over four field elements.
///
/// The third live domain, and the one that retires obligation O5: a leaf digest
/// and a parent digest are different functions of the same bits, so an internal
/// node cannot be replayed as a leaf regardless of tree depth. That previously
/// rested on every eDSL circuit being fixed-shape — true, but enforced by
/// nothing.
///
/// The message is the felts' checked `u32` halves, so its byte layout is
/// identical to a digest-mode compress and the crate anchor survives: at
/// [`SOCKET_ROUNDS`] = 7 a leaf is
/// `blake3::hash(LE32(lo0)‖LE32(hi0)‖…‖LE32(hi3)‖"LFML")` truncated.
pub const TAG_LFML: u32 = u32::from_le_bytes(*b"LFML");

/// The domain tag `"LFMT"` — one step of the Fiat–Shamir transcript chain.
///
/// The transcript step is this socket in every respect except this word: same
/// `h = IV`, same `m[0..4] = state`, `m[4..8] = operand`, same `t`,
/// `block_len` and `flags`, same four-word truncation. So at
/// [`SOCKET_ROUNDS`] = 7 a transcript step is literally
/// `blake3::hash(state ‖ operand ‖ "LFMT")` truncated to 128 bits, and it
/// inherits the compress socket's external anchor unchanged — which is the
/// whole point of building the transcript out of this socket rather than out of
/// a second one.
pub const TAG_LFMT: u32 = u32::from_le_bytes(*b"LFMT");

/// `CHUNK_START | CHUNK_END | ROOT` — the flags a one-block, one-chunk,
/// root-position BLAKE3 hash uses. Matching the tree hasher exactly is what
/// keeps §2.1's byte-level form a plain library call.
pub const FLAGS_LFMC: u32 = 0x0B;

/// The message length in bytes: one 4-byte word per lane, plus the tag.
///
/// 52 at [`cols::NUM_LANES`] = 12. **Derived, never written as a literal**: it
/// is `v[14]`, hence the `vd` operand of round-0 G #2 and from there an XOR
/// operand, so it cannot be mode-dependent (`WordRef::byte` panics on a
/// `ModeSelected`) — all three domains move together and a hand-written 52 that
/// disagreed with the lane count would desynchronise the wire interpretation
/// from the host reference (COMMIT.md §1.4.4 **H9**).
///
/// 52 < 64 keeps a row ONE BLAKE3 block, which is what keeps the crate-KAT
/// anchor: at [`SOCKET_ROUNDS`] = 7 a row is still a plain `blake3::hash` call.
pub const BLOCK_LEN_LFMC: u32 = 4 * (cols::NUM_LANES as u32 + 1);

/// The counter. Zero: one block, one chunk, chunk index 0.
pub const COUNTER_LFMC: u64 = 0;

/// The truncation window: the digest is the LOW four of the 16 output words.
pub const OUT_WINDOW: usize = HASH_DIGEST_FELTS;

/// The dataflow framing [`run_flow`] itself decides — the round count and the
/// truncation window. One value, used by the wire interpretation, the value
/// interpretation and the trace filler alike, so they cannot desynchronise.
pub(crate) const FLOW: FlowConfig = FlowConfig {
    rounds: SOCKET_ROUNDS,
    out_window: OUT_WINDOW,
    full_output: false,
};

/// The 16 message words of the socket's 52-byte block, under domain `tag`.
///
/// The lanes first, the tag straight after them, zeros above — so the byte
/// string is `LE32(lanes) ‖ tag` whatever the mode, and the tag stays LAST as
/// COMMIT.md §1.2 specifies it.
pub fn socket_message(lanes: &[u32; cols::NUM_LANES], tag: u32) -> [u32; 16] {
    let mut m = [0u32; 16];
    m[..cols::NUM_LANES].copy_from_slice(lanes);
    m[cols::NUM_LANES] = tag;
    m
}

/// **The reference the chip is checked against**: the socket's 2-to-1 step,
/// word-level, at an explicit round count and in an explicit domain.
///
/// `rounds` is an argument rather than [`SOCKET_ROUNDS`] so the KATs can pin
/// both variants in one test run; the chip itself is compiled for exactly one.
/// `tag` is an argument for the same reason it is a column on the chip: two
/// domains, one function.
pub fn socket_digest_rounds_tagged(
    a: &[u32; 4],
    b: &[u32; 4],
    rounds: usize,
    tag: u32,
) -> [u32; 4] {
    socket_digest_lanes(&digest_row_lanes(a, b), rounds, tag)
}

/// A digest row's twelve lanes: the two cells it reads, then the four zeros the
/// third input cell's pins force.
///
/// Written once, so the host reference cannot disagree with the AIR about what a
/// compress row's new lanes hold.
pub fn digest_row_lanes(a: &[u32; 4], b: &[u32; 4]) -> [u32; cols::NUM_LANES] {
    let mut lanes = [0u32; cols::NUM_LANES];
    lanes[0..4].copy_from_slice(a);
    lanes[4..8].copy_from_slice(b);
    lanes
}

/// The socket over twelve explicit lanes — the one place the framing is applied.
pub fn socket_digest_lanes(lanes: &[u32; cols::NUM_LANES], rounds: usize, tag: u32) -> [u32; 4] {
    let out = blake3_compress_rounds(
        &BLAKE3_IV,
        &socket_message(lanes, tag),
        COUNTER_LFMC,
        BLOCK_LEN_LFMC,
        FLAGS_LFMC,
        rounds,
    );
    [out[0], out[1], out[2], out[3]]
}

/// [`socket_digest_rounds_tagged`] in the MERKLE domain.
pub fn socket_digest_rounds(a: &[u32; 4], b: &[u32; 4], rounds: usize) -> [u32; 4] {
    socket_digest_rounds_tagged(a, b, rounds, TAG_LFMC)
}

/// [`socket_digest_rounds`] at the compiled-in round count — what a `Compress`
/// row proves, and what [`Blake3Permutation::compress`] computes.
pub fn socket_digest(a: &[u32; 4], b: &[u32; 4]) -> [u32; 4] {
    socket_digest_rounds(a, b, SOCKET_ROUNDS)
}

/// One transcript step at an explicit round count — the `"LFMT"` domain.
pub fn transcript_digest_rounds(state: &[u32; 4], operand: &[u32; 4], rounds: usize) -> [u32; 4] {
    socket_digest_rounds_tagged(state, operand, rounds, TAG_LFMT)
}

/// [`transcript_digest_rounds`] at the compiled-in round count — what a
/// `Transcript` row proves, and what [`Blake3Permutation::transcript`]
/// computes.
pub fn transcript_digest(state: &[u32; 4], operand: &[u32; 4]) -> [u32; 4] {
    transcript_digest_rounds(state, operand, SOCKET_ROUNDS)
}

/// The domain tag a row in `mode` hashes under, or `None` for a mode this arm
/// has no socket for.
///
/// One function, so the executor, the trace filler, the multiplicity histogram
/// and the KATs cannot disagree about which tag a row carries. The AIR gets the
/// same mapping through [`TAG_SELECTOR`], written the one other way it has to
/// be written — as a linear form over the mode columns.
pub const fn tag_for_mode(mode: HashMode) -> Option<u32> {
    match mode {
        HashMode::Compress => Some(TAG_LFMC),
        HashMode::Transcript => Some(TAG_LFMT),
        HashMode::Leaf => Some(TAG_LFML),
        HashMode::Permute => None,
    }
}

// =========================================================================
// The felt boundary (the LEAF mode) — host side
// =========================================================================

/// Field elements one leaf row hashes — the leaf **RATE** (COMMIT.md §1.4.1).
///
/// Four felts = eight halves, which with the four accumulator lanes fill the
/// socket's twelve message lanes. It is one whole machine cell, which is the
/// property the rate was chosen for: the leaf program reads its felt stream in
/// the natural 4-per-cell layout with no re-packing pass.
pub const FELTS_PER_LEAF: usize = 4;

/// Goldilocks `p = 2^64 − 2^32 + 1`, as the halves see it: `p − 1` is
/// `hi = 2^32−1`, `lo = 0`.
const MAX_HALF: u32 = u32::MAX;

/// The chip's canonicity predicate, stated exactly as its constraints do.
///
/// For halves already known to be `u32`, `v = lo + 2^32·hi < p` **iff** NOT
/// (`hi` maximal AND `lo ≥ 1`) — because `p − 1 = 0xFFFFFFFF_00000000`. That one
/// line is the whole reason this mode is cheap: it costs two witness columns per
/// felt instead of a 64-bit decomposition.
///
/// ⚠ It **assumes** the `u32` bound rather than establishing it — see the
/// module docs.
pub const fn is_canonical(lo: u32, hi: u32) -> bool {
    !(hi == MAX_HALF && lo >= 1)
}

/// `v → (lo, hi)`, or `None` when `v` is not a canonical Goldilocks element.
///
/// **REJECTS, never reduces.** A non-canonical value has no satisfying witness,
/// so its row is unprovable; a host that wrapped instead would claim a digest no
/// proof can produce. Same shape as obligation O1's own reject-don't-reduce
/// rule, and the reason is the same.
pub fn felt_halves(v: u64) -> Option<(u32, u32)> {
    let (lo, hi) = (v as u32, (v >> 32) as u32);
    is_canonical(lo, hi).then_some((lo, hi))
}

/// Four felts → the eight message lanes ABOVE the accumulator,
/// `[lo0, hi0, …, lo3, hi3]` — the row's lanes 4–11.
///
/// A felt's halves are ADJACENT, which is load-bearing: it lets the canonicity
/// gate read one pair of neighbouring lanes instead of reaching across the row.
pub fn leaf_lanes(felts: &LfmWord) -> Option<[u32; 2 * FELTS_PER_LEAF]> {
    use math::field::traits::IsPrimeField;
    let mut lanes = [0u32; 2 * FELTS_PER_LEAF];
    for (i, f) in felts.iter().enumerate() {
        let (lo, hi) = felt_halves(GoldilocksField::canonical(f.value()))?;
        lanes[2 * i] = lo;
        lanes[2 * i + 1] = hi;
    }
    Some(lanes)
}

/// A leaf row's twelve lanes: the accumulator cell, then the felts' halves.
///
/// `None` if the accumulator is not four `u32` lanes (it is a previous digest,
/// so it is by construction) or if a felt is not canonical.
///
/// The split is what makes the row a HYBRID and it is the reason this exists as
/// one function: the accumulator is read as digest lanes and the felts as
/// halves, on the same row, and the trace filler, the BITWISE histogram and the
/// host reference must all split it identically (COMMIT.md §1.4.4 **H5**).
pub fn leaf_row_lanes(acc: &LfmWord, felts: &LfmWord) -> Option<[u32; cols::NUM_LANES]> {
    let mut lanes = [0u32; cols::NUM_LANES];
    lanes[..cols::NUM_ACC_LANES].copy_from_slice(&lanes_of(acc)?);
    lanes[cols::NUM_ACC_LANES..].copy_from_slice(&leaf_lanes(felts)?);
    Some(lanes)
}

/// One leaf row at an explicit round count — the `"LFML"` domain over the
/// accumulator and the felts' halves.
///
/// **The accumulator rides in the message, so there is no separate fold**: this
/// one compression both absorbs `felts` and chains `acc` (COMMIT.md §1.2).
pub fn leaf_digest_rounds(acc: &LfmWord, felts: &LfmWord, rounds: usize) -> Option<[u32; 4]> {
    Some(socket_digest_lanes(
        &leaf_row_lanes(acc, felts)?,
        rounds,
        TAG_LFML,
    ))
}

/// [`leaf_digest_rounds`] at the compiled-in round count — what a `Leaf` row
/// proves, and what [`Blake3Permutation::leaf`] computes.
pub fn leaf_digest(acc: &LfmWord, felts: &LfmWord) -> Option<[u32; 4]> {
    leaf_digest_rounds(acc, felts, SOCKET_ROUNDS)
}

// =========================================================================
// The lane boundary (obligation O1), host side
// =========================================================================

/// A digest cell's four lanes as `u32`s, or `None` if any lane is out of range.
///
/// **`None` must never be turned into a reduction.** The host and the chip have
/// to agree about what was hashed; a felt outside `[0, 2^32)` has no byte
/// decomposition the chip can commit, so reducing it here would make a
/// host-side assertion pass while the chip proved something else. Rejecting is
/// also what keeps the two consistent in the other direction: the chip refuses
/// such a lane (no byte string satisfies both the identity and `AreBytes`), so
/// a host that reduced would claim a digest no proof can produce.
pub fn lanes_of(word: &LfmWord) -> Option<[u32; 4]> {
    use math::field::traits::IsPrimeField;
    let mut out = [0u32; 4];
    for (o, felt) in out.iter_mut().zip(word.iter()) {
        *o = u32::try_from(GoldilocksField::canonical(felt.value())).ok()?;
    }
    Some(out)
}

/// A digest cell built from four `u32` lanes — the inverse of [`lanes_of`], and
/// the `keccak_host` convention (one felt = one `u32` = four little-endian
/// bytes), NOT `word::pack_digest`'s eight-bytes-per-lane serialisation.
pub fn word_of(lanes: &[u32; 4]) -> LfmWord {
    core::array::from_fn(|i| FE::from(u64::from(lanes[i])))
}

// =========================================================================
// The host-side hasher
// =========================================================================

/// BLAKE3 behind `LFM_HASH`, `compress` only.
///
/// The trait's `permute` is **partial** for lane-restricted hashers, and this is
/// one: it has no permute socket at all. [`LfmHasher::admits`] is what makes the
/// partiality a rejection rather than a wrong answer.
pub struct Blake3Permutation;

impl LfmHasher for Blake3Permutation {
    /// ✗ Unreachable by construction: [`LfmHasher::admits`] rejects a `Permute`
    /// row before the executor gets here, and the AIR forces `MODE_P = 0`.
    ///
    /// It panics rather than returning something, because every value it could
    /// return would be a hash the chip does not prove.
    fn permute(&self, _state: [FE; HASH_STATE_FELTS]) -> [FE; HASH_STATE_FELTS] {
        panic!(
            "BLAKE3 has no LFM_HASH permute socket: 12-felt permute is unspecified \
             (thoughts/blake3/socket-kats/SOCKET.md §7). Use compress, or select \
             another hasher."
        )
    }

    /// `BLAKE3_IV[0..4]`, so the capacity columns carry something meaningful if
    /// read — but it is **not** part of the compress framing (obligation O3).
    /// The IV enters through `h`, all eight words, and this arm overrides
    /// `compress`/`compress_out` rather than inheriting the trait's
    /// permute-a‖b‖IV default.
    fn compress_iv(&self) -> LfmWord {
        core::array::from_fn(|i| FE::from(u64::from(BLAKE3_IV[i])))
    }

    fn compress(&self, a: &LfmWord, b: &LfmWord) -> LfmWord {
        self.step(a, b, TAG_LFMC)
    }

    /// The digest in lanes 0–3 and zeros above, which is exactly what the chip's
    /// `OUT` columns carry: `MULT1`/`MULT2` are zero on a Compress row, so the
    /// upper eight are sent nowhere, and the AIR pins them to zero.
    fn compress_out(&self, a: &LfmWord, b: &LfmWord) -> [FE; HASH_STATE_FELTS] {
        Self::widen(self.compress(a, b))
    }

    /// The same socket under the TRANSCRIPT tag — this is where BLAKE3 stops
    /// inheriting the trait's single-domain default, and it is the only thing
    /// that makes a transcript step un-replayable as a Merkle parent.
    fn transcript(&self, a: &LfmWord, b: &LfmWord) -> LfmWord {
        self.step(a, b, TAG_LFMT)
    }

    fn transcript_out(&self, a: &LfmWord, b: &LfmWord) -> [FE; HASH_STATE_FELTS] {
        Self::widen(self.transcript(a, b))
    }

    /// The LEAF domain, and the one override that is an ENCODING rather than a
    /// tag: the four felts become eight checked `u32` halves before they reach
    /// the socket. This is what lets arbitrary Goldilocks data be hashed at all
    /// — obligation O1 restricts the *lanes*, and a leaf row satisfies it by
    /// construction rather than by luck.
    fn leaf(&self, acc: &LfmWord, felts: &LfmWord) -> LfmWord {
        word_of(&leaf_digest(acc, felts).expect(
            "leaf accumulator lane is not a u32, or a leaf felt is not canonical — admits() \
             should have rejected it (reject, never reduce)",
        ))
    }

    fn leaf_out(&self, acc: &LfmWord, felts: &LfmWord) -> [FE; HASH_STATE_FELTS] {
        Self::widen(self.leaf(acc, felts))
    }

    fn admits(&self, mode: HashMode, state: &[FE; HASH_STATE_FELTS]) -> Result<(), &'static str> {
        match mode {
            HashMode::Permute => Err(
                "BLAKE3 has no LFM_HASH permute socket (SOCKET.md §7); its AIR forces MODE_P = 0",
            ),
            // ★ A leaf row is a HYBRID and both halves have to be checked, in
            // the cells the AIR actually reads them from: the ACCUMULATOR in
            // cell 0 is an ordinary digest and carries the full O1 `u32`
            // restriction, while the FELTS in cell 1 have none — that is the
            // entire point of the mode, since they are split into checked halves
            // inside the socket.
            //
            // ⚠ Checking the wrong cell is a prover PANIC rather than a clean
            // rejection: a non-canonical felt would pass here and blow up later
            // in the witness filler (COMMIT.md §1.4.4 **H7**). The house rule is
            // reject, never reduce — and never panic where a rejection is
            // available.
            HashMode::Leaf => {
                let acc: LfmWord = core::array::from_fn(|i| state[i]);
                let felts: LfmWord = core::array::from_fn(|i| state[4 + i]);
                if lanes_of(&acc).is_none() {
                    return Err(
                        "BLAKE3 leaf accumulator lane is not a u32 (SOCKET.md obligation O1)",
                    );
                }
                if leaf_lanes(&felts).is_none() {
                    return Err(
                        "BLAKE3 leaf felt is not a canonical Goldilocks element (LEAF.md §1.1)",
                    );
                }
                Ok(())
            }
            HashMode::Compress | HashMode::Transcript => {
                let (a, b): (LfmWord, LfmWord) = (
                    core::array::from_fn(|i| state[i]),
                    core::array::from_fn(|i| state[4 + i]),
                );
                if lanes_of(&a).is_none() || lanes_of(&b).is_none() {
                    // Obligation O1, host side, and it binds both two-to-one
                    // modes: a transcript step is the same socket over the same
                    // lane columns, so it inherits the same restriction.
                    // Rejecting rather than reducing is the point: reduction is
                    // the collision. Data that cannot satisfy this belongs in a
                    // LEAF row, which is what that mode exists for.
                    return Err(
                        "BLAKE3 compress input lane is not a u32 (SOCKET.md obligation O1)",
                    );
                }
                Ok(())
            }
        }
    }
}

impl Blake3Permutation {
    /// The socket, once, in the named domain — the one place the host computes
    /// it, so `compress` and `transcript` cannot drift into different framings.
    fn step(&self, a: &LfmWord, b: &LfmWord, tag: u32) -> LfmWord {
        let (a, b) = (
            lanes_of(a).expect("socket lane is not a u32 — admits() should have rejected it"),
            lanes_of(b).expect("socket lane is not a u32 — admits() should have rejected it"),
        );
        word_of(&socket_digest_rounds_tagged(&a, &b, SOCKET_ROUNDS, tag))
    }

    fn widen(digest: LfmWord) -> [FE; HASH_STATE_FELTS] {
        let mut out = [FE::zero(); HASH_STATE_FELTS];
        out[0..HASH_DIGEST_FELTS].clone_from_slice(&digest);
        out
    }
}

// =========================================================================
// Column layout
// =========================================================================

/// The BLAKE3 arm's columns.
///
/// The frozen prefix (`IN0..12`, `S8..12`, `OUT0..12`) keeps the offsets
/// `chips::hash::cols` gives it, so the `LFM_HASH` tuple contract stays
/// literally frozen and every existing `edsl::merkle_walk` caller works
/// unchanged. Everything BLAKE3 additionally witnesses is appended from
/// [`LANES`] on.
///
/// Width, in blocks: 28 shared + 32 lane bytes + `NUM_G · 60` mixing + 16
/// output bytes.
pub mod cols {
    pub use crate::lfm::chips::hash::cols::{
        IN_ADDR0, IN_ADDR1, IN_ADDR2, IN0, MODE_C, MODE_L, MODE_P, MODE_T, MULT0, MULT1, MULT2,
        OUT_ADDR0, OUT_ADDR1, OUT_ADDR2, OUT0, PREP_WIDTH, S8, SHARED_VALUE_COLUMNS,
    };

    use super::{FELTS_PER_LEAF, HASH_DIGEST_FELTS, NUM_G, OUT_WINDOW};

    /// The is-real flag every constraint is gated by and every send's
    /// multiplicity: `MODE_C + MODE_T + MODE_L`, the three modes this arm has a
    /// socket for. `MODE_P` is pinned to zero, so the sum is a bit on every row
    /// and zero on padding.
    ///
    /// All three are *preprocessed* columns, so a prover chooses neither the
    /// gate nor — through the same columns — the domain tag it selects.
    pub const MU_COLUMNS: [usize; 3] = [MODE_C, MODE_T, MODE_L];

    /// The modes whose message lanes above the accumulator ARE the `IN` lanes —
    /// the digest modes. A leaf row's lanes 4–11 are its felts' halves instead,
    /// so the lane identity is gated on this above [`NUM_ACC_LANES`] rather than
    /// on the full mu. Lanes 0–3 are a digest cell in EVERY mode and take the
    /// full mu; see [`super::eval`] and COMMIT.md §1.4.4 **H6**.
    pub const DIGEST_MODE_COLUMNS: [usize; 2] = [MODE_C, MODE_T];

    /// First appended witness column: the byte decomposition of the input lanes,
    /// 4 bytes each, little-endian (`lane_byte`).
    pub const LANES: usize = PREP_WIDTH + SHARED_VALUE_COLUMNS;

    /// Lanes carrying a cell that is a DIGEST under every mode: `a` on a digest
    /// row, the chaining accumulator on a leaf row. One identity serves both
    /// readings, which is why they can share a gate.
    pub const NUM_ACC_LANES: usize = HASH_DIGEST_FELTS;

    /// Input lanes that carry message words.
    ///
    /// The accumulator cell plus one felt cell's halves — the leaf RATE decides
    /// this number, and the digest modes inherit it: `a ‖ b` fills the first
    /// eight and the last four are the third input cell, which the unread-`IN`
    /// pins force to zero (COMMIT.md §1.4.1).
    pub const NUM_LANES: usize = NUM_ACC_LANES + 2 * FELTS_PER_LEAF;

    /// The mixing core: one 60-cell block per G-instance, laid out exactly as
    /// `blake3_chip::cols` lays one out (56 byte cells + 4 carry bits).
    pub const G: usize = LANES + 4 * NUM_LANES;
    pub const G_SIZE: usize = 60;

    /// Feed-forward output bytes — only the truncation window's four words.
    pub const OUTW: usize = G + NUM_G * G_SIZE;

    /// The LEAF mode's canonicity witnesses: `Z_i` and `GINV_i` per felt.
    ///
    /// `LFM_BITDEC`'s own `Z`/`GINV` idiom, applied to two halves instead of 64
    /// bits — the machine's established canonicity shape, not a new invention.
    /// Two columns and four constraints per felt, **zero extra sends**.
    ///
    /// They exist on EVERY row, leaf or not, because a chip has one width. That
    /// is the mode's whole marginal cost: +8 value cells per compress row.
    pub const CANON: usize = OUTW + 4 * OUT_WINDOW;

    pub const NUM_COLUMNS: usize = CANON + 2 * FELTS_PER_LEAF;

    // Offsets inside one G block, shared verbatim with `blake3_chip::cols` so
    // the two chips' blocks are the same shape and the wire interpretation
    // below is the same code with different bases.
    pub use crate::lfm::blake3_chip::cols::{
        G_A1, G_A1_C, G_A2, G_A2_C, G_C1, G_C2, G_R1, G_R2, G_X1, G_X2, G_X3, G_X4,
    };

    /// Byte `b` of input lane `lane` (0..[`NUM_LANES`]).
    #[inline]
    pub const fn lane_byte(lane: usize, b: usize) -> usize {
        LANES + 4 * lane + b
    }

    /// Base column of G-block `g`.
    #[inline]
    pub const fn g_base(g: usize) -> usize {
        G + g * G_SIZE
    }

    /// Byte `b` of digest word `i` (0..4).
    #[inline]
    pub const fn out_byte(i: usize, b: usize) -> usize {
        OUTW + 4 * i + b
    }

    /// Felt `i`'s canonicity flag: 1 exactly when its high half is maximal.
    #[inline]
    pub const fn canon_z(i: usize) -> usize {
        CANON + 2 * i
    }

    /// Felt `i`'s inverse witness for `(2^32 − 1) − hi`, zero when that is zero.
    #[inline]
    pub const fn canon_ginv(i: usize) -> usize {
        CANON + 2 * i + 1
    }

    /// The `IN` column of leaf felt `i` — the SECOND input cell.
    ///
    /// The felts sit above the accumulator, which is what the halves binding and
    /// [`super::lanes_from_cells`] must both read (COMMIT.md §1.4.4 **H4**).
    #[inline]
    pub const fn leaf_felt(i: usize) -> usize {
        IN0 + NUM_ACC_LANES + i
    }

    /// Message lane carrying felt `i`'s LOW half. Halves are adjacent, so the
    /// canonicity gate reads neighbours rather than reaching across the row;
    /// they start above the accumulator lanes.
    #[inline]
    pub const fn leaf_lo_lane(i: usize) -> usize {
        NUM_ACC_LANES + 2 * i
    }

    /// Message lane carrying felt `i`'s HIGH half.
    #[inline]
    pub const fn leaf_hi_lane(i: usize) -> usize {
        NUM_ACC_LANES + 2 * i + 1
    }
}

/// Value columns the census counts: everything past the preprocessed prefix.
pub const MAIN_COLUMNS: usize = cols::NUM_COLUMNS - cols::PREP_WIDTH;

// =========================================================================
// Wire interpretation — the socket's framing over the shared dataflow
// =========================================================================

/// `m[8] = MODE_C·"LFMC" + MODE_T·"LFMT"` — the row's domain tag.
///
/// **Why this is not prover-chosen.** `MODE_C` and `MODE_T` are preprocessed
/// columns: a row's mode is fixed by its position in the preprocessed trace,
/// that trace is fixed by its commitment, and the commitment is folded into
/// `lfm_program_id`. The prover chooses neither, which is the same argument
/// that already makes the mu gate trustworthy. Two constraints make it bite —
/// the mode-sum booleanity (idx 4) forces at most one tag to be selected, and
/// `MODE_T` being preprocessed is what stops the selector itself being chosen.
/// Controls M5 and M6 in `blake3_socket_tests` are what make each of those
/// dependencies a checked claim rather than an assertion.
const TAG_SELECTOR: &[(usize, u32)] = &[
    (cols::MODE_C, TAG_LFMC),
    (cols::MODE_T, TAG_LFMT),
    (cols::MODE_L, TAG_LFML),
    // ✗ `MODE_P` is deliberately absent, not forgotten: there is no permute
    // socket and idx 5 pins the column to zero, so a term for it would be
    // identically zero and would suggest a domain that does not exist.
];

/// The message word at schedule index `i`, as wiring.
///
/// `i < NUM_LANES` are the input lanes' byte columns; `m[NUM_LANES]` is the
/// domain tag and everything above it is zero. None of them is a witness column,
/// which is what makes the domain separation free (no cells, no range checks,
/// SOCKET.md §2.3) — the tag went from a constant to a linear form over
/// preprocessed columns and kept that property, because a preprocessed column is
/// not a witness.
///
/// **The tag sits immediately after the lanes, not at a fixed `m[8]`.** That is
/// what makes the message the byte string `LE32(lanes) ‖ tag` at any lane count,
/// which is the form COMMIT.md §1.2 specifies and the form the KATs pin. A
/// `ModeSelected` word is legal at any index for the same reason a `Const` one
/// is: message words reach `add3` and nothing else.
fn message_word_ref(i: usize) -> WordRef {
    match i {
        i if i < cols::NUM_LANES => WordRef::Cols(word_cols(cols::lane_byte(i, 0))),
        i if i == cols::NUM_LANES => WordRef::ModeSelected(TAG_SELECTOR),
        _ => WordRef::Const(0),
    }
}

/// The socket's wire interpretation: same [`run_flow`], different framing and
/// different column bases.
struct SocketWire(WireFlow);

impl Blake3Flow for SocketWire {
    type Word = WordRef;

    /// `h = IV`, all eight words — so the entire initial state is constant and
    /// the socket costs zero input-state columns.
    fn input_h(&mut self, i: usize) -> WordRef {
        WordRef::Const(BLAKE3_IV[i])
    }

    /// `v[12..16] = t_lo, t_hi, block_len, flags` — all constants here.
    fn input_v12(&mut self, j: usize) -> WordRef {
        WordRef::Const(
            [
                COUNTER_LFMC as u32,
                (COUNTER_LFMC >> 32) as u32,
                BLOCK_LEN_LFMC,
                FLAGS_LFMC,
            ][j],
        )
    }

    fn iv_const(&mut self, i: usize) -> WordRef {
        WordRef::Const(BLAKE3_IV[i])
    }

    fn add3(&mut self, g: usize, half: usize, a: WordRef, b: WordRef, m_idx: usize) -> WordRef {
        let base = cols::g_base(g);
        let s = word_cols(base + if half == 0 { cols::G_A1 } else { cols::G_A2 });
        let cbase = base
            + if half == 0 {
                cols::G_A1_C
            } else {
                cols::G_A2_C
            };
        self.0.add3s.push(Add3Wire {
            a,
            b,
            m: message_word_ref(m_idx),
            s,
            c1: cbase,
            c2: cbase + 1,
        });
        WordRef::Cols(s)
    }

    fn add2(&mut self, g: usize, half: usize, a: WordRef, b: WordRef) -> WordRef {
        let s = word_cols(cols::g_base(g) + if half == 0 { cols::G_C1 } else { cols::G_C2 });
        self.0.add2s.push(Add2Wire { a, b, s });
        WordRef::Cols(s)
    }

    fn xor(&mut self, g: usize, slot: usize, a: WordRef, b: WordRef) -> WordRef {
        let off = match slot {
            0 => cols::G_X1,
            1 => cols::G_X2,
            2 => cols::G_X3,
            _ => cols::G_X4,
        };
        let out = word_cols(cols::g_base(g) + off);
        self.0.xors.push(XorWire { a, b, out });
        WordRef::Cols(out)
    }

    fn rotr16(&mut self, w: WordRef) -> WordRef {
        w.rotr_bytes(2)
    }

    fn rotr8(&mut self, w: WordRef) -> WordRef {
        w.rotr_bytes(1)
    }

    fn rot_shift(&mut self, g: usize, half: usize, w: WordRef) -> WordRef {
        let base = cols::g_base(g) + if half == 0 { cols::G_R1 } else { cols::G_R2 };
        let y = word_cols(base + 8);
        self.0.rots.push(RotWire {
            input: w,
            sll_lo: [base, base + 1],
            sllc_lo: [base + 2, base + 3],
            sll_hi: [base + 4, base + 5],
            sllc_hi: [base + 6, base + 7],
            y,
            r: ROT_SHIFT_R[half],
        });
        WordRef::Cols(y)
    }

    fn feed_forward_low(&mut self, i: usize, vi: WordRef, vi8: WordRef) {
        self.0.xors.push(XorWire {
            a: vi,
            b: vi8,
            out: word_cols(cols::out_byte(i, 0)),
        });
    }

    /// ✗ Never called: [`FLOW`] has `full_output = false`. `out[i+8]` is not
    /// part of a truncated 128-bit digest, and never building those twelve
    /// words is where most of the socket's saving comes from.
    fn feed_forward_high(&mut self, _i: usize, _vi8: WordRef, _hi: WordRef) {
        unreachable!("the socket's truncation window produces no high output words")
    }
}

/// The socket's full wiring, in canonical order. Built from the single
/// dataflow, so the senders below and the witness written by
/// [`fill_socket_witness`] cannot drift apart.
fn socket_wires() -> WireFlow {
    let mut w = SocketWire(WireFlow {
        add3s: Vec::with_capacity(NUM_G * 2),
        add2s: Vec::with_capacity(NUM_G * 2),
        xors: Vec::with_capacity(NUM_G * 4 + OUT_WINDOW),
        rots: Vec::with_capacity(NUM_G * 2),
    });
    run_flow(&mut w, FLOW);
    w.0
}

/// The value interpretation of the same dataflow, for one row's twelve message
/// lanes in one domain.
///
/// Lanes rather than cells because a LEAF row's lanes are not cells throughout —
/// lanes 0–3 are its accumulator and lanes 4–11 are four felts' halves. The
/// mixing core does not care which; it sees twelve `u32`s either way, and that is
/// exactly why the leaf mode needs no new layout.
///
/// The tag is an input because it is a message word: it enters the very first
/// round's `add3` and every value downstream of it, so a row's witness and its
/// BITWISE lookups both depend on which domain the row hashes in.
fn socket_values(lanes: &[u32; cols::NUM_LANES], tag: u32) -> ValueFlow {
    ValueFlow::compute_with(
        &BLAKE3_IV,
        &socket_message(lanes, tag),
        COUNTER_LFMC,
        BLOCK_LEN_LFMC,
        FLAGS_LFMC,
        FLOW,
    )
}

// =========================================================================
// Bus interactions — the BITWISE half of `chips::hash::bus_interactions`
// =========================================================================

fn direct(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: stark::lookup::Packing::Direct,
    }
}

fn byte_bus_value(b: ByteRef) -> BusValue {
    match b {
        ByteRef::Col(c) => direct(c),
        ByteRef::Const(v) => BusValue::constant(u64::from(v)),
    }
}

/// The BITWISE lookups the BLAKE3 arm adds to `LFM_HASH`'s six `LfmMem` tuples.
///
/// Three groups, in canonical [`socket_wires`] order:
///
/// 1. `ByteAlu[XOR]` per XOR byte — the mixing core and the feed-forward. The
///    lookup pins the output *and* byte-range-checks both operands, which is
///    why nearly every word in this design needs no explicit `AreBytes`.
/// 2. `AreBytes` on the four shift halfwords of each rotation. The `SLL` bound
///    is tight and load-bearing: with `2^16` invertible mod `p` it is what pins
///    `SLL = (x · 2^r) mod 2^16` uniquely.
/// 3. `AreBytes` on the input lanes' bytes — obligation O1. These are the only
///    bytes with no XOR consumer, exactly as `m`'s are in `blake3_chip`.
pub fn bitwise_interactions() -> Vec<BusInteraction> {
    let wires = socket_wires();
    let mut interactions =
        Vec::with_capacity(4 * wires.xors.len() + 4 * wires.rots.len() + 2 * cols::NUM_LANES);
    let mu = || {
        Multiplicity::Sum3(
            cols::MU_COLUMNS[0],
            cols::MU_COLUMNS[1],
            cols::MU_COLUMNS[2],
        )
    };

    for xw in &wires.xors {
        for b in 0..4 {
            interactions.push(BusInteraction::sender(
                BusId::ByteAlu,
                mu(),
                vec![
                    BusValue::constant(alu_op::XOR as u64),
                    byte_bus_value(xw.a.byte(b)),
                    byte_bus_value(xw.b.byte(b)),
                    direct(xw.out[b]),
                ],
            ));
        }
    }

    for rw in &wires.rots {
        for pair in [rw.sll_lo, rw.sllc_lo, rw.sll_hi, rw.sllc_hi] {
            interactions.push(BusInteraction::sender(
                BusId::AreBytes,
                mu(),
                vec![direct(pair[0]), direct(pair[1])],
            ));
        }
    }

    for lane in 0..cols::NUM_LANES {
        for p in 0..2 {
            interactions.push(BusInteraction::sender(
                BusId::AreBytes,
                mu(),
                vec![
                    direct(cols::lane_byte(lane, 2 * p)),
                    direct(cols::lane_byte(lane, 2 * p + 1)),
                ],
            ));
        }
    }

    interactions
}

/// The BITWISE lookups [`bitwise_interactions`] sends, mirrored send for send,
/// for the multiplicity histogram. Enumeration order is the senders' own, via
/// the shared [`ValueFlow`].
///
/// Each row is `(lanes, tag)`: the domain reaches the histogram because it
/// reaches the tag word, and every XOR byte downstream of round 0 differs between the
/// domains. A histogram built with the wrong tag balances against nothing.
pub fn bitwise_ops_for(rows: &[([u32; cols::NUM_LANES], u32)]) -> Vec<BitwiseOperation> {
    let mut out = Vec::with_capacity(
        rows.len() * (4 * (NUM_G * 4 + OUT_WINDOW) + 4 * NUM_G * 2 + 2 * cols::NUM_LANES),
    );

    for (lanes, tag) in rows {
        let flow = socket_values(lanes, *tag);
        for &(x, y, _out) in &flow.xors {
            for byte in 0..4 {
                out.push(BitwiseOperation::byte_op(
                    BitwiseOperationType::ByteAluXor,
                    ((x >> (8 * byte)) & 0xFF) as u8,
                    ((y >> (8 * byte)) & 0xFF) as u8,
                ));
            }
        }
        for &(sll_lo, sllc_lo, sll_hi, sllc_hi, _y) in &flow.rots {
            for hw in [sll_lo, sllc_lo, sll_hi, sllc_hi] {
                out.push(BitwiseOperation::byte_op(
                    BitwiseOperationType::AreBytes,
                    (hw & 0xFF) as u8,
                    (hw >> 8) as u8,
                ));
            }
        }
        for &lane in lanes.iter() {
            for p in 0..2 {
                out.push(BitwiseOperation::byte_op(
                    BitwiseOperationType::AreBytes,
                    ((lane >> (16 * p)) & 0xFF) as u8,
                    ((lane >> (16 * p + 8)) & 0xFF) as u8,
                ));
            }
        }
    }

    out
}

// =========================================================================
// Trace
// =========================================================================

#[inline]
fn set_word_bytes(row: &mut [FE], col: usize, w: u32) {
    for b in 0..4 {
        row[col + b] = FE::from(u64::from((w >> (8 * b)) as u8));
    }
}

/// Writes the BLAKE3 witness into a hash row whose `IN`/`S`/`OUT` columns are
/// already filled.
///
/// The two input cells are read back out of the row's own `IN0..8` columns —
/// the exact cells the lane-decomposition constraints read — rather than from
/// the executor record, so the witness cannot describe a different input than
/// the one the AIR constrains. That is the discipline `fill_poseidon_witness`
/// established and it matters more here, because the lane boundary is where the
/// only new soundness surface lives.
///
/// The DOMAIN is read back out of the row's own mode columns for the same
/// reason, and it is the half that matters most: the tag word is a linear form over
/// exactly those columns, so a witness built from them cannot describe a
/// different domain than the one the AIR evaluates. Taking the tag as an
/// argument — as this did at first — left a filler that could be handed the
/// wrong domain for a row whose selectors said otherwise.
///
/// # Panics
///
/// If a lane is not a `u32`, or if the row selects no domain this arm has a
/// socket for. `LfmHasher::admits` rejects both at execution, so reaching here
/// means the executor and the trace filler disagree.
pub fn fill_socket_witness(row: &mut [FE]) {
    let tag = tag_from_row(row);
    fill_socket_witness_tagged(row, tag);
}

/// The row's domain tag, read off its preprocessed mode columns — the machine
/// side of [`TAG_SELECTOR`], and the same value the tag word evaluates to.
///
/// # Panics
///
/// If the row selects neither two-to-one domain. A padding row never reaches
/// the filler (`chip_trace` fills only real rows) and a permute row is
/// unprovable here, so either is a caller bug rather than a case to handle.
fn tag_from_row(row: &[FE]) -> u32 {
    let one = FE::one();
    let set: Vec<u32> = TAG_SELECTOR
        .iter()
        .filter(|(col, _)| row[*col] == one)
        .map(|(_, tag)| *tag)
        .collect();
    match set[..] {
        [tag] => tag,
        _ => panic!(
            "a BLAKE3 hash row must select EXACTLY ONE of the domains this arm \
             has a socket for (MODE_C, MODE_T, MODE_L). None set means a permute \
             or padding row reached the socket witness filler; more than one is a \
             row the registrar's one-hot check should already have refused. \
             Either way its AIR cannot prove the row."
        ),
    }
}

/// The twelve message lanes of a row that reads `cells`, split the way its mode
/// reads them.
///
/// ★ **The row is a HYBRID and this is the only place the split lives.** Lanes
/// 0–3 are `cells[0]` read as digest lanes under EVERY mode — `a` on a digest
/// row, the chaining accumulator on a leaf row. Above that the readings differ:
///
/// - **digest modes** — lanes 4–11 are `cells[1]` and `cells[2]`, two more cells
///   of four `u32` lanes. The third is unread by these modes and the AIR pins it
///   to zero, so those four lanes are zero on every honest digest row.
/// - **leaf mode** — lanes 4–11 are `cells[1]` read as four FELTS and split into
///   `lo`/`hi` halves.
///
/// The trace filler and the BITWISE histogram both come through here, so a
/// witness and the multiplicities it must balance against cannot split a row
/// differently (COMMIT.md §1.4.4 **H5**).
pub fn lanes_from_cells(is_leaf: bool, cells: &[LfmWord; 3]) -> [u32; cols::NUM_LANES] {
    let mut lanes = [0u32; cols::NUM_LANES];
    lanes[..cols::NUM_ACC_LANES]
        .copy_from_slice(&lanes_of(&cells[0]).expect("socket lane is not a u32 (O1)"));
    if is_leaf {
        lanes[cols::NUM_ACC_LANES..].copy_from_slice(
            &leaf_lanes(&cells[1]).expect("leaf felt is not canonical (LEAF.md §1.1)"),
        );
    } else {
        for (k, cell) in cells[1..].iter().enumerate() {
            let base = cols::NUM_ACC_LANES + 4 * k;
            lanes[base..base + 4]
                .copy_from_slice(&lanes_of(cell).expect("compress lane is not a u32 (O1)"));
        }
    }
    lanes
}

/// [`lanes_from_cells`] for a trace row, reading the input cells and the mode
/// off the row itself.
///
/// Keyed on `MODE_L` rather than on a tag, so it is total for any row a control
/// can build — including one whose mode columns are fractional.
fn lanes_from_row(row: &[FE]) -> [u32; cols::NUM_LANES] {
    let cell = |base: usize| -> LfmWord { core::array::from_fn(|i| row[base + i]) };
    lanes_from_cells(
        row[cols::MODE_L] == FE::one(),
        &[cell(cols::IN0), cell(cols::IN0 + 4), cell(cols::IN0 + 8)],
    )
}

/// The canonicity witnesses for one leaf row's four felts.
///
/// `Z_i = 1` exactly when felt `i`'s high half is maximal; `GINV_i` inverts
/// `G_i = (2^32 − 1) − hi_i` when that is nonzero and is zero when it is not.
/// The same `Z`/`GINV` pair `LFM_BITDEC` uses for its own canonicity check.
fn fill_canonicity_witness(row: &mut [FE], lanes: &[u32; cols::NUM_LANES]) {
    for i in 0..FELTS_PER_LEAF {
        let hi = lanes[cols::leaf_hi_lane(i)];
        let g = u64::from(MAX_HALF - hi);
        let (z, ginv) = if g == 0 {
            (FE::one(), FE::zero())
        } else {
            (
                FE::zero(),
                FE::from(g).inv().expect("a nonzero field element inverts"),
            )
        };
        row[cols::canon_z(i)] = z;
        row[cols::canon_ginv(i)] = ginv;
    }
}

/// [`fill_socket_witness`] under an EXPLICIT domain.
///
/// Exists for the negative controls (M1/M2), which have to build a row whose
/// witness and whose mode columns deliberately disagree — the forgery the
/// domain separation is supposed to reject. Production goes through
/// [`fill_socket_witness`], which cannot construct that.
pub(crate) fn fill_socket_witness_tagged(row: &mut [FE], tag: u32) {
    // The lane READING is a property of the row's mode; the `tag` argument is
    // only the hash DOMAIN. Keeping them separate is what lets the mode-
    // confusion controls build a row that reads its input correctly and hashes
    // it in the wrong domain — which is the forgery, and it would be
    // unconstructible if one argument decided both.
    let lanes = lanes_from_row(row);

    for (lane, &v) in lanes.iter().enumerate() {
        set_word_bytes(row, cols::lane_byte(lane, 0), v);
    }
    // The canonicity witnesses are filled on EVERY row, not only leaf rows: the
    // columns exist chip-wide, and a digest row's felts are its lanes, whose
    // high halves are never maximal-and-nonzero-low in a way that matters
    // because the constraints are `MODE_L`-gated. Filling them uniformly keeps
    // the filler branch-free and leaves no uninitialised witness anywhere.
    fill_canonicity_witness(row, &lanes);

    let flow = socket_values(&lanes, tag);
    let mut a3 = flow.add3s.iter();
    let mut a2 = flow.add2s.iter();
    let mut xo = flow.xors.iter();
    let mut ro = flow.rots.iter();
    for g in 0..NUM_G {
        let base = cols::g_base(g);
        for half in 0..2 {
            let (s_off, c_off, x_off, c2_off, x2_off, r_off) = if half == 0 {
                (
                    cols::G_A1,
                    cols::G_A1_C,
                    cols::G_X1,
                    cols::G_C1,
                    cols::G_X2,
                    cols::G_R1,
                )
            } else {
                (
                    cols::G_A2,
                    cols::G_A2_C,
                    cols::G_X3,
                    cols::G_C2,
                    cols::G_X4,
                    cols::G_R2,
                )
            };
            let &(s, c1, c2) = a3.next().expect("add3 count");
            set_word_bytes(row, base + s_off, s);
            row[base + c_off] = FE::from(u64::from(c1));
            row[base + c_off + 1] = FE::from(u64::from(c2));

            let &(_, _, x) = xo.next().expect("xor count");
            set_word_bytes(row, base + x_off, x);

            let &c = a2.next().expect("add2 count");
            set_word_bytes(row, base + c2_off, c);

            let &(_, _, x2) = xo.next().expect("xor count");
            set_word_bytes(row, base + x2_off, x2);

            let &(sll_lo, sllc_lo, sll_hi, sllc_hi, y) = ro.next().expect("rot count");
            let hw = |v: u16, k: usize| FE::from(u64::from((v >> (8 * k)) as u8));
            for (k, v) in [sll_lo, sllc_lo, sll_hi, sllc_hi].into_iter().enumerate() {
                row[base + r_off + 2 * k] = hw(v, 0);
                row[base + r_off + 2 * k + 1] = hw(v, 1);
            }
            set_word_bytes(row, base + r_off + 8, y);
        }
    }

    for i in 0..OUT_WINDOW {
        set_word_bytes(row, cols::out_byte(i, 0), flow.out[i]);
        debug_assert_eq!(
            row[cols::OUT0 + i],
            FE::from(u64::from(flow.out[i])),
            "the digest lane the executor wrote is the one the mixing core produced"
        );
    }
}

// =========================================================================
// Constraints
// =========================================================================

/// Constraints the BLAKE3 arm emits.
///
/// The framing block — 4 capacity copies, the mode-sum booleanity, the
/// `MODE_P = 0` pin, [`cols::NUM_LANES`] lane decompositions, 8 unused-output
/// pins, 4 digest recompositions, [`NUM_UNREAD_INPUT_PINS`] unread-`IN` pins and
/// 16 leaf felt/canonicity constraints — plus 16 per G-instance: per G, two
/// add3s (a sum identity and two carry booleanities each), two add2 carry
/// booleanities, and two rotations (two shift identities and two recombines
/// each).
pub const NUM_CONSTRAINTS: usize = CORE_IDX + 16 * NUM_G;

/// First mixing-core constraint index — everything below it is framing.
const CORE_IDX: usize = LEAF_IDX + LEAF_CONSTRAINTS_PER_FELT * FELTS_PER_LEAF;

/// First lane-decomposition index: after the capacity copies, the mode-sum
/// booleanity and the `MODE_P` pin.
///
/// Public so the gate suite can name the identity for a specific lane rather
/// than locate it by a literal — the point of the H6 controls is that the
/// violated constraint IS the lane's own identity.
pub const LANE_IDX: usize = 6;

/// First unused-output pin index.
///
/// ★ **DERIVED FROM [`cols::NUM_LANES`], never written as a literal.** The lane
/// block grew from 8 to 12 with the leaf RATE, and the constraint COUNT did not
/// move — the unread-`IN` pins lost exactly the four the lanes gained — so a
/// hardcoded `14` here would have silently overwritten the first four output
/// pins with lane identities, leaving lanes 8–11 with no identity at all.
/// `EmitTracker`'s duplicate assert is `#[cfg(debug_assertions)]` and the house
/// convention runs the suite in release, so nothing would have failed
/// (COMMIT.md §1.4.4 **H1**). `blake3_socket_tests::
/// every_hash_candidate_emits_each_constraint_index_exactly_once` is the
/// release-visible guard that would catch it if these ever go back to literals.
const OUT_PIN_IDX: usize = LANE_IDX + cols::NUM_LANES;

/// First digest-recomposition index.
const DIGEST_IDX: usize = OUT_PIN_IDX + 8;

/// First unread-`IN` pin index.
///
/// Public so the controls can name the pins rather than locate them by a
/// literal — the point of those tests is that the violated set IS the pins.
pub const UNREAD_IDX: usize = DIGEST_IDX + OUT_WINDOW;

/// First LEAF constraint index: four per felt (the halves binding and the three
/// canonicity constraints), after the shared unread-`IN` pins.
///
/// Public so the controls can locate a specific canonicity constraint by name
/// rather than by a literal that silently rots when the framing grows.
pub const LEAF_IDX: usize = UNREAD_IDX + NUM_UNREAD_INPUT_PINS;

/// Constraints per leaf felt: the halves binding, then `canon-a/b/c`.
pub const LEAF_CONSTRAINTS_PER_FELT: usize = 4;

/// The BLAKE3 arm of `HashConstraints::eval`.
///
/// Every constraint is mu-gated on `MU = MODE_C + MODE_T + MODE_L` and every
/// bus send carries the same sum, so an all-zero padding row satisfies the set
/// vacuously and emits nothing. Max degree is 3, reached by the mu-gated carry
/// booleanities and by the leaf canonicity block — the wrap's blowup 2 depends
/// on that staying 3, which is why the 3-operand add uses two summed carry BITS
/// rather than one ternary carry (`k(k−1)(k−2) = 0` is already degree 3, and
/// mu-gating would push it to 4).
pub fn eval<B: ConstraintBuilder<F, E>>(b: &mut B) {
    let mu = |b: &B| {
        let [c, t, l] = cols::MU_COLUMNS;
        b.main(0, c) + b.main(0, t) + b.main(0, l)
    };
    let digest_mu = |b: &B| {
        let [c, t] = cols::DIGEST_MODE_COLUMNS;
        b.main(0, c) + b.main(0, t)
    };
    let mode_c = b.main(0, cols::MODE_C);
    let mode_t = b.main(0, cols::MODE_T);
    let mode_l = b.main(0, cols::MODE_L);
    let mode_p = b.main(0, cols::MODE_P);

    // idx 0–3: capacity-state copy, in the same shape every other arm uses —
    // `S_i = MODE_P·IN_i + (MODE_C + MODE_T + MODE_L)·IV_i`. A transcript row
    // and a leaf row are both compresses in framing, so their capacity prefix is
    // still the IV; only the selector widens. With MODE_P pinned to zero below
    // this reduces to `S_i = MU·IV_i`; it is written in the general form so the
    // shared prefix means the same thing under every hasher.
    for (k, iv) in BLAKE3_IV.iter().take(4).enumerate() {
        let s = b.main(0, cols::S8 + k);
        let in_i = b.main(0, cols::IN0 + 8 + k);
        let iv_i = b.const_base(u64::from(*iv));
        let m = mu(b);
        b.emit_base(k, s - (mode_p.clone() * in_i + m * iv_i));
    }

    // idx 4: mode sum-boolean (exactly-one-of is the registrar's). This is what
    // excludes two selectors both being 1 — which would sum BOTH domain tags
    // into the tag word — since the sum would be 2 and 2·(1−2) ≠ 0. ⚠ It does NOT
    // force each selector to a bit: a fractional split still satisfies it and
    // blends the tags, which is what control M5/M6 demonstrates and what the
    // registrar's one-hot check is the actual answer to.
    let mode_sum = mode_c + mode_t + mode_l.clone() + mode_p.clone();
    let one = b.one();
    b.emit_base(4, mode_sum.clone() * (one - mode_sum));

    // idx 5: ✗ no permute socket, PERMANENTLY. Pinning the preprocessed mode
    // selector makes a program containing a `permute` unprovable under BLAKE3
    // rather than silently proved against a framing nobody specified. Option B1
    // decided no permute socket is ever built, so this pin is not a placeholder
    // waiting to be deleted — it is the decision, written down as a constraint.
    b.emit_base(5, mode_p);

    // THE LANE BOUNDARY (obligation O1). One linear identity per input lane; the
    // matching `AreBytes` sends are in `bitwise_interactions`. NEITHER ALONE
    // SUFFICES, and the two buy DIFFERENT things — see the module docs. This
    // identity makes `IN_lane` and `m[lane]` the same field element, because the
    // core reads the same linear form; the sends bound the bytes, and are the
    // message words' ONLY range check, which is what `add3`'s exactness needs.
    // With both, the sum of four bytes weighted by 2^{8k} is < 2^32 ≪ p, so it
    // cannot wrap and the lane is forced below 2^32.
    //
    // ★ THE GATE IS PER LANE RANGE, and getting it wrong in the permissive
    // direction is a soundness break (COMMIT.md §1.4.4 **H6**):
    //
    // - **lanes 0–3 → the full mu.** They are `a` on a digest row and the
    //   chaining ACCUMULATOR on a leaf row, and the same identity is correct for
    //   both readings, so one gate serves. This is also the only thing that
    //   range-checks the accumulator: identity + `AreBytes` is what makes "the
    //   accumulator is a previous digest, hence `u32`" a constraint rather than
    //   a hope. Gate these on `digest_mu` and a leaf row's accumulator carries
    //   no identity at all — the prover picks the chain's message words freely
    //   and the whole leaf chain unbinds.
    // - **lanes 4–11 → the digest modes only.** On a LEAF row they are four
    //   felts' halves, so `IN_lane` and `m[lane]` are deliberately NOT the same
    //   field element — the leaf block below states the relation those rows do
    //   satisfy. Gating these on mu instead would make every leaf row
    //   unprovable.
    //
    // ★ On a digest row lanes 8–11 read the THIRD input cell, which the unread-
    // `IN` pins force to zero, so the identity reads `0 = Σ bytes·2^{8k}` and
    // with the `AreBytes` bound in hand forces all sixteen bytes to zero. That
    // is what keeps the four message words the leaf RATE added out of the
    // prover's hands on a Merkle parent — free, but only because these
    // identities exist.
    for lane in 0..cols::NUM_LANES {
        let felt = b.main(0, cols::IN0 + lane);
        let bytes = word_expr(b, &WordRef::Cols(word_cols(cols::lane_byte(lane, 0))));
        let m = if lane < cols::NUM_ACC_LANES {
            mu(b)
        } else {
            digest_mu(b)
        };
        b.emit_base(LANE_IDX + lane, m * (felt - bytes));
    }

    // The digest is ONE cell, so the upper eight `OUT` lanes carry nothing.
    // `MULT1`/`MULT2` are zero on a Compress row so they reach no bus, but
    // pinning them costs eight degree-1 constraints and removes the question
    // entirely. Ungated: they are zero on padding rows too.
    for j in 0..8 {
        let out = b.main(0, cols::OUT0 + HASH_DIGEST_FELTS + j);
        b.emit_base(OUT_PIN_IDX + j, out);
    }

    // The digest lanes. No range check is needed on `OUTW`'s bytes — they are
    // `ByteAlu[XOR]` outputs, hence already bytes — and the sum is < 2^32 ≪ p,
    // so `OUT_i` is forced to the honest u32. That is why the socket's OUTPUT
    // always satisfies O1 (obligation O2) and only leaf digests and
    // prover-hinted siblings need the input check.
    for i in 0..OUT_WINDOW {
        let felt = b.main(0, cols::OUT0 + i);
        let bytes = word_expr(b, &WordRef::Cols(word_cols(cols::out_byte(i, 0))));
        let m = mu(b);
        b.emit_base(DIGEST_IDX + i, m * (felt - bytes));
    }

    // The input cells this row's mode does not read.
    //
    // ⚠ **LOAD-BEARING, and not only here.** On THIS arm the unread columns
    // reach no constraint, so the pin is what keeps them from being an open
    // question. On an arm whose constraints read `IN` — `Test` and `Poseidon`
    // both do, `A_i = IN_i` for `i < 8` — the same pin is the difference between
    // a leaf digest that is a function of its input and one carrying four free
    // prover-chosen felts. It shipped missing there once. That is why this is
    // `chips::hash`'s single derivation from `HashMode::num_input_cells` and not
    // four lines written out per arm.
    let next = crate::lfm::chips::hash::emit_unread_input_pins(b, UNREAD_IDX);
    debug_assert_eq!(next, LEAF_IDX);

    // ★ THE LEAF MODE. Per felt: the halves binding, then the three canonicity
    // constraints. The felts are the SECOND input cell — the first is the
    // chaining accumulator, whose lanes the identity block above binds.
    //
    // `v = lo + 2^32·hi` with `lo, hi < 2^32` is a decomposition, not yet a
    // canonical one: `p − 1 = 0xFFFFFFFF_00000000`, so the pairs with `hi`
    // maximal and `lo ≥ 1` encode field elements that ALSO have an ordinary
    // encoding. Without the canonicity block one felt would have two half-pairs,
    // hence two leaf digests — a collision in the felt→digest map, which is
    // exactly what a Merkle tree must not have.
    //
    // `Z`/`GINV` is `LFM_BITDEC`'s own idiom over two halves instead of 64 bits:
    // `canon_a` gives `G ≠ 0 ⇒ Z = 0`, `canon_b` gives `G = 0 ⇒ Z = 1`, and
    // `canon_c` then reads "hi maximal ⇒ lo zero".
    let two_32_leaf = b.const_base(1u64 << 32);
    let max_half = b.const_base(u64::from(MAX_HALF));
    for i in 0..FELTS_PER_LEAF {
        let lo = word_expr(
            b,
            &WordRef::Cols(word_cols(cols::lane_byte(cols::leaf_lo_lane(i), 0))),
        );
        let hi = word_expr(
            b,
            &WordRef::Cols(word_cols(cols::lane_byte(cols::leaf_hi_lane(i), 0))),
        );
        let v = b.main(0, cols::leaf_felt(i));
        let z = b.main(0, cols::canon_z(i));
        let ginv = b.main(0, cols::canon_ginv(i));
        let g = max_half.clone() - hi.clone();
        let base = LEAF_IDX + LEAF_CONSTRAINTS_PER_FELT * i;

        // binding: the felt IS its two halves.
        b.emit_base(
            base,
            mode_l.clone() * (v - lo.clone() - hi * two_32_leaf.clone()),
        );
        // canon-a: G ≠ 0 ⇒ Z = 0.
        b.emit_base(base + 1, mode_l.clone() * z.clone() * g.clone());
        // canon-b: G = 0 ⇒ Z = 1.
        let one = b.one();
        b.emit_base(base + 2, mode_l.clone() * (one - z.clone() - g * ginv));
        // canon-c: hi maximal ⇒ lo zero. THE constraint; the two above exist to
        // make `Z` mean what this one needs it to mean.
        b.emit_base(base + 3, mode_l.clone() * z * lo);
    }

    // The mixing core, from the single dataflow.
    let wires = socket_wires();
    let mut idx = CORE_IDX;

    let two_32 = b.const_base(1u64 << 32);
    let inv_2_32 = b.const_base(crate::constraints::templates::INV_SHIFT_32);

    // add3: μ·(a + b + m − s − 2^32·(c1+c2)) = 0; μ·ci·(1−ci) = 0.
    for aw in &wires.add3s {
        let a = word_expr(b, &aw.a);
        let bb = word_expr(b, &aw.b);
        let m_w = word_expr(b, &aw.m);
        let s = word_expr(b, &WordRef::Cols(aw.s));
        let c1 = b.main(0, aw.c1);
        let c2 = b.main(0, aw.c2);
        let sum_id = a + bb + m_w - s - (c1.clone() + c2.clone()) * two_32.clone();
        let m = mu(b);
        b.emit_base(idx, m * sum_id);
        idx += 1;
        for c in [c1, c2] {
            let one = b.one();
            let m = mu(b);
            b.emit_base(idx, m * c.clone() * (one - c));
            idx += 1;
        }
    }

    // add2: the carry is an EXPRESSION, `(a + b − s)·2^−32`, not a column —
    // `μ·carry·(1−carry) = 0` says `a + b − s ∈ {0, 2^32}`, which is the sum
    // identity and the carry's booleanity in one degree-3 constraint. One
    // column and one constraint per add2 cheaper than witnessing the carry, and
    // exactly as strong.
    for aw in &wires.add2s {
        let a = word_expr(b, &aw.a);
        let bb = word_expr(b, &aw.b);
        let s = word_expr(b, &WordRef::Cols(aw.s));
        let carry = (a + bb - s) * inv_2_32.clone();
        let one = b.one();
        let m = mu(b);
        b.emit_base(idx, m * carry.clone() * (one - carry));
        idx += 1;
    }

    // Rotations: two shift identities and two recombines each. Soundness needs
    // 2^16 invertible mod p, which is a FIELD fact a bitvector model cannot
    // see — it is what makes the tight `AreBytes` bound on `SLL` load-bearing.
    for rw in &wires.rots {
        let (xlo, xhi) = match &rw.input {
            WordRef::Cols(c) => (half_expr(b, &[c[0], c[1]]), half_expr(b, &[c[2], c[3]])),
            WordRef::Const(_) | WordRef::ModeSelected(_) => {
                unreachable!("shift inputs are always committed XOR outputs")
            }
        };
        let sll_lo = half_expr(b, &rw.sll_lo);
        let sllc_lo = half_expr(b, &rw.sllc_lo);
        let sll_hi = half_expr(b, &rw.sll_hi);
        let sllc_hi = half_expr(b, &rw.sllc_hi);
        let ylo = half_expr(b, &[rw.y[0], rw.y[1]]);
        let yhi = half_expr(b, &[rw.y[2], rw.y[3]]);
        let two_r = b.const_base(1u64 << rw.r);
        let two_16 = b.const_base(65536);

        let m = mu(b);
        b.emit_base(
            idx,
            m * (xlo * two_r.clone() - sllc_lo.clone() * two_16.clone() - sll_lo.clone()),
        );
        idx += 1;
        let m = mu(b);
        b.emit_base(
            idx,
            m * (xhi * two_r - sllc_hi.clone() * two_16 - sll_hi.clone()),
        );
        idx += 1;
        let m = mu(b);
        b.emit_base(idx, m * (ylo - sll_hi - sllc_lo));
        idx += 1;
        let m = mu(b);
        b.emit_base(idx, m * (yhi - sll_lo - sllc_hi));
        idx += 1;
    }

    debug_assert_eq!(
        idx, NUM_CONSTRAINTS,
        "every declared constraint index must be emitted exactly once"
    );
}
