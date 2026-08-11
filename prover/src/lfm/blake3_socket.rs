//! The BLAKE3 arm of `LFM_HASH` — the Option-A 2-to-1 compress socket.
//!
//! This is Route A of `thoughts/shared/lfm-real-hash/PLAN.md` §3: BLAKE3 hosted
//! *behind* the frozen `LFM_HASH` socket, exactly the way Poseidon is. The chip
//! count stays 14, `PREP_WIDTH` stays 11, the 28-column shared value prefix
//! keeps its offsets, and everything BLAKE3 witnesses is appended after it — so
//! no preprocessed root moves and no registered program's digest moves. Only a
//! program that opts into [`HasherKind::Blake3`] gets a different identity, and
//! it gets it deliberately, through the hasher tag `lfm_program_id` binds.
//!
//! # What one row proves
//!
//! One row = one 2-to-1 compress, specified byte-level in
//! `thoughts/blake3/socket-kats/SOCKET.md` §2.1 and word-level in §2.2:
//!
//! ```text
//! msg    = LE32(a0..a3) ‖ LE32(b0..b3) ‖ "LFMC"            (36 bytes)
//! digest = BLAKE3(msg)[0..16]                              (128 bits, 1 cell)
//! ```
//!
//! which, 36 bytes being one block, is exactly one compression with `h = IV`
//! (all eight words), `m[0..4] = a`, `m[4..8] = b`, `m[8] = "LFMC"` as a
//! little-endian `u32`, `m[9..16] = 0`, `t = 0`, `block_len = 36`,
//! `flags = CHUNK_START|CHUNK_END|ROOT`, and the digest the LOW four output
//! words.
//!
//! **At [`SOCKET_ROUNDS`] = 7 that is literally `blake3::hash(a ‖ b ‖ "LFMC")`,**
//! so the socket has a direct external anchor and needs no oracle in the chain.
//! That is the whole reason the domain tag lives in the *message* rather than in
//! `flags`, `t` or `h`: a tag anywhere else would make even the 7-round socket a
//! nonstandard invocation of `f` that no library computes, throwing the anchor
//! away for nothing (SOCKET.md §2.3).
//!
//! # Why the socket is so much cheaper than the standalone chip
//!
//! [`super::blake3_chip`] is the syscall-shaped chip: 28 input `u32` words and
//! all 16 output words are committed columns. Here `h`, `t`, `block_len`,
//! `flags` and `m[8..16]` are **compile-time constants**, and the truncation
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
//! # ✓ DECIDED — O5: leaves get the `"LFML"` tag
//!
//! This socket has **one** tag, so it separates LFM compressions from other
//! BLAKE3 uses but **not leaves from parents within a tree**. If leaves ever
//! enter a tree as raw cells rather than through a distinct domain, a
//! variable-depth tree admits the classic Merkle second-preimage confusion: an
//! internal node replayed as a leaf. Decided 2026-08-10: any future
//! leaf-hashing path MUST use the reserved `"LFML"` tag — the RFC 6962
//! leaf/parent split expressed in the tag scheme, keeping both domains direct
//! `blake3::hash` KATs. (BLAKE3's own `PARENT` flag was rejected: it cannot be
//! reused without leaving the standard-hash framing that makes the crate a
//! direct KAT. A fixed-depth-only policy was rejected as an invariant no
//! mechanism enforces.)
//!
//! Nothing implements `"LFML"` yet, and what makes that safe is **fixed depth
//! alone** — not any absence of leaf hashing. Programs already form leaf
//! digests by compressing raw data rows under the same `"LFMC"` tag
//! (`programs.rs` FriToyV0: `leaf = compress(row_even, row_odd)` before each
//! `merkle_walk`), so leaves and parents are NOT domain-separated today. That
//! is sound only because every current tree is a fixed-depth static circuit:
//! the eDSL builder fixes the program's shape at build time — hints supply
//! values, never structure — so a node at one level cannot be replayed at
//! another. The obligation binds review, not this code: a change adding
//! variable-depth trees, or a leaf-hashing API meant to coexist with them,
//! without `"LFML"` is rejected on O5 (`gate-oracle/ORACLE.md` §7).
//!
//! Equally on the record: the digest is 128 bits, so this socket offers
//! **64-bit collision resistance** by the birthday bound. That follows from
//! `HASH_DIGEST_FELTS = 4` and the machine's declared 128-bit target — it is
//! not introduced by BLAKE3 or by the truncation window.
//!
//! # ✗ There is no `permute` socket
//!
//! `LFM_HASH` has two modes and this arm implements **one**. The `permute`
//! socket — 12 felts in, 12 out — is unspecified: it has no mapping decision,
//! no KATs, and its security argument is not the same argument as `compress`'s
//! (SOCKET.md §7). Rather than invent one, the AIR forces `MODE_P = 0`, so a
//! program containing a `permute` is *unprovable* under BLAKE3, and
//! [`Blake3Permutation`] rejects one at execution with a message saying why.
//! The practical consequence is that `edsl::merkle_walk` works under BLAKE3 and
//! `edsl::SpongeVar` does not, so this retires half of the F3.4 disclosure.

use stark::constraints::builder::ConstraintBuilder;
use stark::lookup::{BusInteraction, BusValue, Multiplicity};

use crate::tables::bitwise::{BitwiseOperation, BitwiseOperationType};
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField, alu_op};

use super::blake3::{BLAKE3_IV, BLAKE3_ROUNDS, blake3_compress_rounds};
use super::blake3_chip::{
    Add2Wire, Add3Wire, Blake3Flow, ByteRef, FlowConfig, ROT_SHIFT_R, RotWire, ValueFlow, WireFlow,
    WordRef, XorWire, half_expr, run_flow, word_cols, word_expr,
};
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

/// The domain tag `"LFMC"`, read as one little-endian `u32` — `m[8]`.
///
/// A tag is never reused for a second purpose, for the same reason
/// `HasherKind::as_tag` never reuses a discriminant. `"LFMP"` is reserved for
/// the (unspecified) permute socket and `"LFML"` for a leaf domain.
pub const TAG_LFMC: u32 = u32::from_le_bytes(*b"LFMC");

/// `CHUNK_START | CHUNK_END | ROOT` — the flags a one-block, one-chunk,
/// root-position BLAKE3 hash uses. Matching the tree hasher exactly is what
/// keeps §2.1's byte-level form a plain library call.
pub const FLAGS_LFMC: u32 = 0x0B;

/// The message length in bytes: 16 (`a`) + 16 (`b`) + 4 (tag).
pub const BLOCK_LEN_LFMC: u32 = 36;

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

/// The 16 message words of the socket's 36-byte block.
pub fn socket_message(a: &[u32; 4], b: &[u32; 4]) -> [u32; 16] {
    let mut m = [0u32; 16];
    m[0..4].copy_from_slice(a);
    m[4..8].copy_from_slice(b);
    m[8] = TAG_LFMC;
    m
}

/// **The reference the chip is checked against**: the socket's 2-to-1 compress,
/// word-level, at an explicit round count.
///
/// `rounds` is an argument rather than [`SOCKET_ROUNDS`] so the KATs can pin
/// both variants in one test run; the chip itself is compiled for exactly one.
pub fn socket_digest_rounds(a: &[u32; 4], b: &[u32; 4], rounds: usize) -> [u32; 4] {
    let out = blake3_compress_rounds(
        &BLAKE3_IV,
        &socket_message(a, b),
        COUNTER_LFMC,
        BLOCK_LEN_LFMC,
        FLAGS_LFMC,
        rounds,
    );
    [out[0], out[1], out[2], out[3]]
}

/// [`socket_digest_rounds`] at the compiled-in round count — what the chip
/// proves, and what [`Blake3Permutation::compress`] computes.
pub fn socket_digest(a: &[u32; 4], b: &[u32; 4]) -> [u32; 4] {
    socket_digest_rounds(a, b, SOCKET_ROUNDS)
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
        let (a, b) = (
            lanes_of(a).expect("compress lane is not a u32 — admits() should have rejected it"),
            lanes_of(b).expect("compress lane is not a u32 — admits() should have rejected it"),
        );
        word_of(&socket_digest(&a, &b))
    }

    /// The digest in lanes 0–3 and zeros above, which is exactly what the chip's
    /// `OUT` columns carry: `MULT1`/`MULT2` are zero on a Compress row, so the
    /// upper eight are sent nowhere, and the AIR pins them to zero.
    fn compress_out(&self, a: &LfmWord, b: &LfmWord) -> [FE; HASH_STATE_FELTS] {
        let digest = self.compress(a, b);
        let mut out = [FE::zero(); HASH_STATE_FELTS];
        out[0..HASH_DIGEST_FELTS].clone_from_slice(&digest);
        out
    }

    fn admits(&self, mode: HashMode, state: &[FE; HASH_STATE_FELTS]) -> Result<(), &'static str> {
        if mode == HashMode::Permute {
            return Err(
                "BLAKE3 has no LFM_HASH permute socket (SOCKET.md §7); its AIR forces MODE_P = 0",
            );
        }
        let (a, b): (LfmWord, LfmWord) = (
            core::array::from_fn(|i| state[i]),
            core::array::from_fn(|i| state[4 + i]),
        );
        if lanes_of(&a).is_none() || lanes_of(&b).is_none() {
            // Obligation O1, host side. Rejecting rather than reducing is the
            // point: reduction is the collision.
            return Err("BLAKE3 compress input lane is not a u32 (SOCKET.md obligation O1)");
        }
        Ok(())
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
        IN_ADDR0, IN_ADDR1, IN_ADDR2, IN0, MODE_C, MODE_P, MULT0, MULT1, MULT2, OUT_ADDR0,
        OUT_ADDR1, OUT_ADDR2, OUT0, PREP_WIDTH, S8, SHARED_VALUE_COLUMNS,
    };

    use super::{NUM_G, OUT_WINDOW};

    /// The is-real flag every constraint is gated by and every send's
    /// multiplicity: `MODE_C`, because this arm proves only Compress rows and
    /// pins `MODE_P` to zero. It is a *preprocessed* column, so a prover cannot
    /// choose it.
    pub const MU: usize = MODE_C;

    /// First appended witness column: the byte decomposition of the 8 input
    /// lanes, 4 bytes each, little-endian (`lane_byte`).
    pub const LANES: usize = PREP_WIDTH + SHARED_VALUE_COLUMNS;
    /// Input lanes that carry message words: `a[0..4] ‖ b[0..4]`.
    pub const NUM_LANES: usize = 8;

    /// The mixing core: one 60-cell block per G-instance, laid out exactly as
    /// `blake3_chip::cols` lays one out (56 byte cells + 4 carry bits).
    pub const G: usize = LANES + 4 * NUM_LANES;
    pub const G_SIZE: usize = 60;

    /// Feed-forward output bytes — only the truncation window's four words.
    pub const OUTW: usize = G + NUM_G * G_SIZE;

    pub const NUM_COLUMNS: usize = OUTW + 4 * OUT_WINDOW;

    // Offsets inside one G block, shared verbatim with `blake3_chip::cols` so
    // the two chips' blocks are the same shape and the wire interpretation
    // below is the same code with different bases.
    pub use crate::lfm::blake3_chip::cols::{
        G_A1, G_A1_C, G_A2, G_A2_C, G_C1, G_C2, G_R1, G_R2, G_X1, G_X2, G_X3, G_X4,
    };

    /// Byte `b` of input lane `lane` (0..8).
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
}

/// Value columns the census counts: everything past the preprocessed prefix.
pub const MAIN_COLUMNS: usize = cols::NUM_COLUMNS - cols::PREP_WIDTH;

// =========================================================================
// Wire interpretation — the socket's framing over the shared dataflow
// =========================================================================

/// The message word at schedule index `i`, as wiring.
///
/// `i < 8` are the input lanes' byte columns; `m[8]` is the domain tag and
/// `m[9..16]` are zero — **constants, not columns**, which is what makes the
/// domain separation free (no cells, no range checks, SOCKET.md §2.3).
fn message_word_ref(i: usize) -> WordRef {
    match i {
        0..=7 => WordRef::Cols(word_cols(cols::lane_byte(i, 0))),
        8 => WordRef::Const(TAG_LFMC),
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
        match w {
            WordRef::Cols([b0, b1, b2, b3]) => WordRef::Cols([b2, b3, b0, b1]),
            WordRef::Const(v) => WordRef::Const(v.rotate_right(16)),
        }
    }

    fn rotr8(&mut self, w: WordRef) -> WordRef {
        match w {
            WordRef::Cols([b0, b1, b2, b3]) => WordRef::Cols([b1, b2, b3, b0]),
            WordRef::Const(v) => WordRef::Const(v.rotate_right(8)),
        }
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

/// The value interpretation of the same dataflow, for one `(a, b)` pair.
fn socket_values(a: &[u32; 4], b: &[u32; 4]) -> ValueFlow {
    ValueFlow::compute_with(
        &BLAKE3_IV,
        &socket_message(a, b),
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
/// 3. `AreBytes` on the 8 input lanes' 32 bytes — obligation O1. These are the
///    only bytes with no XOR consumer, exactly as `m`'s are in
///    `blake3_chip`.
pub fn bitwise_interactions() -> Vec<BusInteraction> {
    let wires = socket_wires();
    let mut interactions =
        Vec::with_capacity(4 * wires.xors.len() + 4 * wires.rots.len() + 2 * cols::NUM_LANES);

    for xw in &wires.xors {
        for b in 0..4 {
            interactions.push(BusInteraction::sender(
                BusId::ByteAlu,
                Multiplicity::Column(cols::MU),
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
                Multiplicity::Column(cols::MU),
                vec![direct(pair[0]), direct(pair[1])],
            ));
        }
    }

    for lane in 0..cols::NUM_LANES {
        for p in 0..2 {
            interactions.push(BusInteraction::sender(
                BusId::AreBytes,
                Multiplicity::Column(cols::MU),
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
pub fn bitwise_ops_for(rows: &[([u32; 4], [u32; 4])]) -> Vec<BitwiseOperation> {
    let mut out =
        Vec::with_capacity(rows.len() * (4 * (NUM_G * 4 + OUT_WINDOW) + 4 * NUM_G * 2 + 16));

    for (a, b) in rows {
        let flow = socket_values(a, b);
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
        for &lane in a.iter().chain(b.iter()) {
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
/// # Panics
///
/// If a lane is not a `u32`. `LfmHasher::admits` rejects that at execution, so
/// reaching here means the executor and the trace filler disagree.
pub fn fill_socket_witness(row: &mut [FE]) {
    let cell = |base: usize| -> LfmWord { core::array::from_fn(|i| row[base + i]) };
    let a = lanes_of(&cell(cols::IN0)).expect("compress lane is not a u32 (O1)");
    let b = lanes_of(&cell(cols::IN0 + 4)).expect("compress lane is not a u32 (O1)");

    for (lane, &v) in a.iter().chain(b.iter()).enumerate() {
        set_word_bytes(row, cols::lane_byte(lane, 0), v);
    }

    let flow = socket_values(&a, &b);
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
/// `26` framing constraints (4 capacity copies, the mode-sum booleanity, the
/// `MODE_P = 0` pin, 8 lane decompositions, 8 unused-output pins, 4 digest
/// recompositions) plus 16 per G-instance: per G, two add3s (a sum identity and
/// two carry booleanities each), two add2 carry booleanities, and two rotations
/// (two shift identities and two recombines each).
pub const NUM_CONSTRAINTS: usize = 26 + 16 * NUM_G;

/// First mixing-core constraint index — everything below it is framing.
const CORE_IDX: usize = 26;

/// The BLAKE3 arm of `HashConstraints::eval`.
///
/// Every constraint is mu-gated on `MU = MODE_C` and every bus send carries
/// `Multiplicity::Column(MU)`, so an all-zero padding row satisfies the set
/// vacuously and emits nothing. Max degree is 3, reached by the mu-gated carry
/// booleanities — the wrap's blowup 2 depends on that staying 3, which is why
/// the 3-operand add uses two summed carry BITS rather than one ternary carry
/// (`k(k−1)(k−2) = 0` is already degree 3, and mu-gating would push it to 4).
pub fn eval<B: ConstraintBuilder<F, E>>(b: &mut B) {
    let mu = |b: &B| b.main(0, cols::MU);
    let mode_c = b.main(0, cols::MODE_C);
    let mode_p = b.main(0, cols::MODE_P);

    // idx 0–3: capacity-state copy, in the same shape every other arm uses —
    // `S_i = MODE_P·IN_i + MODE_C·IV_i`. With MODE_P pinned to zero below it
    // reduces to `S_i = MODE_C·IV_i`; it is written in the general form so the
    // shared prefix means the same thing under every hasher.
    for (k, iv) in BLAKE3_IV.iter().take(4).enumerate() {
        let s = b.main(0, cols::S8 + k);
        let in_i = b.main(0, cols::IN0 + 8 + k);
        let iv_i = b.const_base(u64::from(*iv));
        b.emit_base(k, s - (mode_p.clone() * in_i + mode_c.clone() * iv_i));
    }

    // idx 4: mode sum-boolean (exactly-one-of is the registrar's).
    let mode_sum = mode_c + mode_p.clone();
    let one = b.one();
    b.emit_base(4, mode_sum.clone() * (one - mode_sum));

    // idx 5: ✗ no permute socket. Pinning the preprocessed mode selector makes
    // a program containing a `permute` unprovable under BLAKE3 rather than
    // silently proved against a framing nobody specified.
    b.emit_base(5, mode_p);

    // idx 6–13: THE LANE BOUNDARY (obligation O1). One mu-gated linear identity
    // per input lane; the matching `AreBytes` sends are in
    // `bitwise_interactions`. NEITHER ALONE SUFFICES, and the two buy DIFFERENT
    // things — see the module docs. This identity makes `IN_lane` and `m[lane]`
    // the same field element, because the core reads the same linear form; the
    // sends bound the bytes, and are the message words' ONLY range check, which
    // is what `add3`'s exactness needs. With both, the sum of four bytes
    // weighted by 2^{8k} is < 2^32 ≪ p, so it cannot wrap and the lane is
    // forced below 2^32.
    for lane in 0..cols::NUM_LANES {
        let felt = b.main(0, cols::IN0 + lane);
        let bytes = word_expr(b, &WordRef::Cols(word_cols(cols::lane_byte(lane, 0))));
        let m = mu(b);
        b.emit_base(6 + lane, m * (felt - bytes));
    }

    // idx 14–21: the digest is ONE cell, so the upper eight `OUT` lanes carry
    // nothing. `MULT1`/`MULT2` are zero on a Compress row so they reach no bus,
    // but pinning them costs eight degree-1 constraints and removes the
    // question entirely. Ungated: they are zero on padding rows too.
    for j in 0..8 {
        let out = b.main(0, cols::OUT0 + HASH_DIGEST_FELTS + j);
        b.emit_base(14 + j, out);
    }

    // idx 22–25: the digest lanes. No range check is needed on `OUTW`'s bytes —
    // they are `ByteAlu[XOR]` outputs, hence already bytes — and the sum is
    // < 2^32 ≪ p, so `OUT_i` is forced to the honest u32. That is why the
    // socket's OUTPUT always satisfies O1 (obligation O2) and only leaf digests
    // and prover-hinted siblings need the input check.
    for i in 0..OUT_WINDOW {
        let felt = b.main(0, cols::OUT0 + i);
        let bytes = word_expr(b, &WordRef::Cols(word_cols(cols::out_byte(i, 0))));
        let m = mu(b);
        b.emit_base(22 + i, m * (felt - bytes));
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
            WordRef::Const(_) => unreachable!("shift inputs are always committed XOR outputs"),
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
