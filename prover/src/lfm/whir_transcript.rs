//! The WHIR Fiat–Shamir transcript replayed inside the machine.
//!
//! The mirror is `DefaultTranscript<E, RpxTranscriptHash>`
//! (`crypto/crypto/src/fiat_shamir/default_transcript.rs`): a BUFFERING sponge
//! — every absorb appends to a byte buffer, and a squeeze hashes the whole
//! accumulated buffer with `rpx::sponge_leaf_bytes`, clears it, and re-absorbs
//! the digest to advance the chain.
//!
//! Like [`super::transcript_replay`] this is an eDSL library rather than a chip:
//! ordinary Rust tracking the transcript's state AT EMIT TIME and emitting the
//! instructions that reproduce its VALUES at run time. Which squeeze a challenge
//! comes from, where a refill lands, which absorb invalidates the output buffer
//! — all decided by the emitter and baked into the program's shape.
//!
//! # Why this replay is nearly free
//!
//! Three facts, each read off the host rather than assumed.
//!
//! 1. **The squeeze is not reversed and needs no rejection.**
//!    `RpxTranscriptHash` sets `REVERSES_SQUEEZE = false` and
//!    `CANDIDATES_PER_COORDINATE = Some(1)` (`transcript_hash.rs:169-180`): a
//!    squeeze is four canonical felts and one candidate per coordinate is
//!    structural, not probabilistic. The transcript never leaves the field.
//!
//! 2. **An absorbed value IS its felts.** The cubic extension streams 24 bytes
//!    big-endian in coordinate order (`extensions_goldilocks.rs:566-572`), the
//!    base field streams `canonical_u64().to_be_bytes()`, and
//!    `sponge_leaf_bytes` reads 8-byte groups with `u64::from_be_bytes`
//!    (`rpx/mod.rs:404`). So three felts per extension element and four per
//!    32-byte commitment, with no byte work — provided a runtime value starts at
//!    an offset that is a multiple of eight.
//!
//! 3. **The machine's leaf hash IS the host's sponge.**
//!    `WrapHash::algebraic_leaf_hash` (`edsl.rs:676-708`) is `rpx::sponge_leaf`
//!    structurally: rate-8 overwrite, `leaf_capacity` with the padding flag in
//!    lane 0, only the capacity carried between blocks, the digest from the last
//!    permutation, the same empty case. The replay needs no new hash operation.
//!
//! # Constants are free; runtime values must be aligned
//!
//! A run of constant bytes is packed into felts at emit time by exactly
//! `sponge_leaf_bytes`' rule — 8-byte big-endian groups, the trailing partial
//! group zero-extended on the LOW side — at any length. So the epoch statement's
//! fixed part costs no rows and imposes no alignment.
//!
//! What needs alignment is the first RUNTIME value after such a run, which in
//! the epoch statement is the roots. [`WhirTranscript::absorb_felts`] REFUSES an
//! unaligned runtime absorb rather than emitting a byte shift: a value straddling
//! two felts would have to be bit-decomposed, and a shape-dependent shift
//! computed from three lengths is the bug class that is correct on a fixture and
//! wrong on a block. The statement padding that makes the offset a multiple of
//! eight is W1's, computed from the accumulated length.

use crate::tables::types::{FE, FEE};

use super::builder::{Bit, Cell, Ext, Felt, LfmBuilder};
use super::edsl::WrapHash;

/// Felts a squeeze hands out before it must refill — `SQUEEZE_LEN / 8` in the
/// host's bytes (`default_transcript.rs:19`).
pub const CANDIDATES_PER_SQUEEZE: usize = 4;

/// Coordinates in a cubic extension element, one candidate each.
pub const COORDINATES_PER_EXT: usize = 3;

/// Bytes one felt occupies in the sponge's stream.
const BYTES_PER_FELT: usize = 8;

/// The WHIR transcript, replayed.
#[derive(Default)]
pub struct WhirTranscript {
    /// Constant bytes absorbed but not yet packed into felts. Only a trailing
    /// run of constants can be pending: any runtime absorb flushes.
    pending: Vec<u8>,
    /// Felts absorbed since the last squeeze, in order.
    buf: Vec<Felt>,
    /// The last squeeze's four lanes, and how many have been handed out.
    /// `out_pos == CANDIDATES_PER_SQUEEZE` means "empty, squeeze to refill",
    /// which is also what every absorb resets it to.
    out: Option<[Felt; CANDIDATES_PER_SQUEEZE]>,
    out_pos: usize,
    /// Squeezes emitted, for the cost pins.
    squeezes: usize,
}

impl WhirTranscript {
    /// `DefaultTranscript::new(&[])` — the empty seed is absorbed, which
    /// appends nothing and invalidates the (already empty) output buffer.
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            buf: Vec::new(),
            out: None,
            out_pos: CANDIDATES_PER_SQUEEZE,
            squeezes: 0,
        }
    }

    /// `append_bytes` of a PROGRAM CONSTANT: free, and unconstrained in length.
    pub fn absorb_const_bytes(&mut self, bytes: &[u8]) {
        self.pending.extend_from_slice(bytes);
        self.invalidate();
    }

    /// `append_bytes` / `append_field_element` of RUNTIME felts.
    ///
    /// Panics when the constant run before it does not end on a felt boundary:
    /// see the module header.
    pub fn absorb_felts(&mut self, b: &mut LfmBuilder, felts: &[Felt]) {
        assert_eq!(
            self.pending.len() % BYTES_PER_FELT,
            0,
            "a runtime value must start on a felt boundary — {} constant bytes are pending, \
             which is {} short. The statement's computed padding is what guarantees this; \
             emitting a byte shift here instead is the bug class that is correct on a fixture \
             and wrong on a block.",
            self.pending.len(),
            BYTES_PER_FELT - self.pending.len() % BYTES_PER_FELT
        );
        self.flush_pending(b);
        self.buf.extend_from_slice(felts);
        self.invalidate();
    }

    /// `append_field_element` of a cubic extension element: its three
    /// coordinates, big-endian and in order, are three felts of the stream.
    pub fn absorb_ext(&mut self, b: &mut LfmBuilder, value: Ext) {
        let lanes = b.unpack(value.as_cell());
        self.absorb_felts(b, &lanes[..COORDINATES_PER_EXT]);
    }

    /// `append_bytes(root, 32)`: a commitment is four canonical felts, big-endian
    /// (`digest_to_commitment`, `rpx/mod.rs:428`).
    pub fn absorb_digest(&mut self, b: &mut LfmBuilder, digest: Cell) {
        let lanes = b.unpack(digest);
        self.absorb_felts(b, &lanes);
    }

    /// `DefaultTranscript::sample()` — hash the accumulated buffer, clear it,
    /// re-absorb the digest, and refill the output buffer.
    ///
    /// ⚠ The re-absorb is the chain advance, and it is what makes consecutive
    /// draws differ. Dropping it leaves every draw after the first identical,
    /// which a single pinned value cannot see.
    pub fn squeeze(&mut self, b: &mut LfmBuilder) -> [Felt; CANDIDATES_PER_SQUEEZE] {
        let felts = self.felts_now(b);
        let digest = WrapHash::Algebraic.leaf_hash(b, &felts);
        let lanes = b.unpack(digest.cells()[0]);
        self.pending.clear();
        self.buf = lanes.to_vec();
        self.out = Some(lanes);
        self.out_pos = 0;
        self.squeezes += 1;
        lanes
    }

    /// `next_sample_u64` — one candidate, refilling with a squeeze when the
    /// buffer is spent. A candidate is a squeeze lane: `digest_to_commitment`
    /// writes felt `i` as the big-endian bytes `8i..8i+8` and the sampler reads
    /// exactly those with `from_be_bytes`, so the two conversions cancel.
    pub fn next_candidate(&mut self, b: &mut LfmBuilder) -> Felt {
        if self.out.is_none() || self.out_pos >= CANDIDATES_PER_SQUEEZE {
            self.squeeze(b);
        }
        let lane = self.out.expect("a squeeze just filled the buffer")[self.out_pos];
        self.out_pos += 1;
        lane
    }

    /// `sample_field_element` on the cubic extension: three candidates, no
    /// rejection.
    ///
    /// Every candidate is a canonical felt of an RPX squeeze, so
    /// `candidate_in_range` holds by construction and the host's single-draw
    /// schedule (`CANDIDATES_PER_COORDINATE = Some(1)`) has no rejected branch
    /// to emit.
    pub fn sample_ext(&mut self, b: &mut LfmBuilder) -> Ext {
        let a0 = self.next_candidate(b);
        let a1 = self.next_candidate(b);
        let a2 = self.next_candidate(b);
        b.pack_ext(a0, a1, a2)
    }

    /// `sample_u64(2^nbits)`, as the BITS the Merkle walk consumes.
    ///
    /// The bound is always a power of two here — `domain.size() >> k` — so
    /// `threshold = (-2^k) mod 2^k = 0` and no candidate is ever rejected: the
    /// schedule is fixed and there is no branch to emit. The answer is
    /// `candidate mod 2^nbits`, the low `nbits` bits, which is one `BitDec` row
    /// and needs no recomposition because its consumer wants bits.
    pub fn sample_u64_pow2(&mut self, b: &mut LfmBuilder, nbits: usize) -> Vec<Bit> {
        let candidate = self.next_candidate(b);
        b.bit_dec(candidate, nbits)
    }

    /// `state()` — the digest of everything absorbed so far, WITHOUT advancing
    /// the chain: production finalizes a clone, with no reset and no re-absorb
    /// (`default_transcript.rs:236-239`). Only grinding needs it.
    ///
    /// Neither the buffer nor the output position moves, so a later squeeze
    /// hashes the same bytes again. The packing is therefore emitted twice,
    /// which is redundant work and never a different value.
    pub fn state(&mut self, b: &mut LfmBuilder) -> Cell {
        let felts = self.felts_now(b);
        WrapHash::Algebraic.leaf_hash(b, &felts).cells()[0]
    }

    /// Squeezes emitted so far.
    pub fn squeezes(&self) -> usize {
        self.squeezes
    }

    /// Emit-time output position, for tests that pin the consumption schedule.
    pub fn out_pos(&self) -> usize {
        self.out_pos
    }

    /// Felts currently in the sponge's stream: what has been absorbed so far,
    /// then the trailing constant run packed by `sponge_leaf_bytes`' rule —
    /// pending constants come AFTER the felts, never before. Does not mutate, so
    /// `state` can call it without disturbing a partial constant group that a
    /// later absorb will continue.
    fn felts_now(&self, b: &mut LfmBuilder) -> Vec<Felt> {
        let mut felts = self.buf.clone();
        felts.extend(pack_const_bytes(b, &self.pending));
        felts
    }

    /// Moves the pending constants into the felt buffer. Only called where the
    /// run is known to end on a felt boundary, so no group is left partial.
    fn flush_pending(&mut self, b: &mut LfmBuilder) {
        if self.pending.is_empty() {
            return;
        }
        let packed = pack_const_bytes(b, &self.pending);
        self.pending.clear();
        self.buf.extend(packed);
    }

    /// Every absorb drops the buffered squeeze output, so a challenge can never
    /// predate the input it must depend on (`default_transcript.rs:206-210`).
    fn invalidate(&mut self) {
        self.out_pos = CANDIDATES_PER_SQUEEZE;
    }
}

/// Constant bytes as the felts `sponge_leaf_bytes` would read: 8-byte
/// BIG-endian groups, the trailing partial group zero-extended on the LOW side
/// (`rpx/mod.rs:396-406` writes the available bytes at the FRONT of a zeroed
/// eight-byte buffer, so the missing ones are the low bytes).
fn pack_const_bytes(b: &mut LfmBuilder, bytes: &[u8]) -> Vec<Felt> {
    bytes
        .chunks(BYTES_PER_FELT)
        .map(|group| {
            let mut whole = [0u8; BYTES_PER_FELT];
            whole[..group.len()].copy_from_slice(group);
            b.felt_const(FE::from(u64::from_be_bytes(whole)))
        })
        .collect()
}

/// INSTRUCTIONS a squeeze emits over `felts` buffered felts.
///
/// `ceil(felts/4)` `Pack` rows to build the sponge's words, `ceil(felts/8)`
/// permutations (a block is two words), and one `Unpack` to read the digest's
/// four lanes. An empty buffer never permutes — `sponge_leaf_bytes` returns the
/// zero digest — so only the `Unpack` remains.
pub const fn squeeze_rows(felts: usize) -> usize {
    felts.div_ceil(4) + felts.div_ceil(8) + 1
}

/// INSTRUCTIONS a `state()` emits over `felts` buffered felts: the same hash
/// without the digest `Unpack`, because the grind consumes the digest as a word.
pub const fn state_rows(felts: usize) -> usize {
    felts.div_ceil(4) + felts.div_ceil(8)
}

/// PERMUTATIONS a transcript hash over `felts` buffered felts costs, whether it
/// is a squeeze or a `state()`: one per rate-8 block, and NONE for an empty
/// buffer, which returns the zero digest without permuting
/// (`edsl.rs:694-697`).
///
/// The rate is [`super::whir_open::RATE_FELTS`] rather than a literal, because
/// it is the same sponge the leaves are hashed with.
pub const fn sponge_perms(felts: usize) -> usize {
    felts.div_ceil(super::whir_open::RATE_FELTS)
}

/// INSTRUCTIONS an absorbed extension element or commitment emits: the one
/// `Unpack` that turns the cell into lanes.
pub const fn absorb_unpack_rows() -> usize {
    1
}

/// INSTRUCTIONS `sample_ext` emits beyond its squeezes: one `Pack`.
pub const fn sample_ext_rows() -> usize {
    1
}

/// INSTRUCTIONS `sample_u64_pow2` emits beyond its squeezes: one `BitDec`.
pub const fn sample_u64_rows() -> usize {
    1
}

/// The cubic extension element a squeeze's lanes would produce, for tests that
/// need the host's value beside the machine's.
pub fn ext_from_lanes(a0: FE, a1: FE, a2: FE) -> FEE {
    FEE::new([a0, a1, a2])
}

/// ★ `whir_chain::check_grind` (`crypto/multilinear/src/whir_chain.rs:125-139`),
/// emitted — and it is a REUSE, not a port.
///
/// The check is
/// `is_valid_nonce::<GrindingDigest<H>>(&transcript.state(), nonce, bits)`, and
/// `GrindingDigest<H>` is the TRANSCRIPT's digest (`whir_hash.rs:54`), which on
/// this arm is `Rpx256Digest` — the same leaf-domain sponge the replay above
/// uses. So `H(H(PREFIX ‖ seed ‖ factor) ‖ nonce_be)` is exactly what
/// [`super::epoch::emit_grinding_check`]'s algebraic arm already computes: both
/// preimages are already felts (41 bytes is six big-endian groups, 40 is five),
/// the inner digest cell IS the outer preimage's first rate cell so nothing
/// repacks it, and the range check is one `BitDec` whose top `bits` bits are
/// asserted zero. ⚠ My own sizing note called this "a port, not a reuse"; that
/// was wrong, and reusing it is also what keeps ONE definition of the lane
/// placement rather than two.
///
/// `bits == 0` returns immediately, as the host does — with NO state read and
/// NO nonce absorb. Both are observable: a state read is counted separately
/// from a squeeze, and an absorb would move every later challenge.
///
/// ⚠ A nonce at or above `p` reduces, on both sides: the host's sponge reads the
/// eight big-endian bytes through `Fp::from(u64)` exactly as the machine holds
/// the felt. It is a completeness restriction, not a disagreement — and search
/// returns small nonces, so it is unreachable in practice.
pub fn emit_grind_check(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    bits: u8,
    nonce: Felt,
) {
    if bits == 0 {
        return;
    }
    let seed = transcript.state(b);
    super::epoch::emit_grinding_check(b, super::edsl::WrapDigest::from_cell(seed), nonce, bits);
    // `append_bytes(&nonce.to_be_bytes())`: eight big-endian bytes are one felt.
    transcript.absorb_felts(b, &[nonce]);
}

/// INSTRUCTIONS [`emit_grind_check`] emits beyond the transcript state's own
/// hash, which costs [`state_rows`] of whatever the buffer holds.
///
/// One `Unpack` of the seed; two `Pack`s and one permutation for the inner
/// hash; one `Pack` and one permutation for the outer hash; one `Unpack` of the
/// result; and the range check, one `BitDec` plus two rows per asserted zero
/// bit. The nonce absorb is free — it is already a felt. So `8 + 2·bits`.
///
/// ★ The inner digest is NEVER unpacked, and that is the shape's one real
/// saving: it IS the outer preimage's first rate cell, so nothing repacks it
/// (`epoch.rs:624-626`). An earlier version of this form charged an `Unpack`
/// for it, and F1 said so.
pub const fn grind_check_rows(bits: usize) -> usize {
    if bits == 0 {
        return 0;
    }
    let seed_unpack = 1;
    let inner = 2 + 1;
    let outer = 1 + 1;
    let digest_unpack = 1;
    let range = 1 + 2 * bits;
    seed_unpack + inner + outer + digest_unpack + range
}

/// `LFM_CONST` rows the grind interns BEYOND the capacity words of its two
/// hashes (which are `leaf_capacity(6)` for the 41-byte inner preimage and
/// `leaf_capacity(5)` for the 40-byte outer one, and are shared with any other
/// hash of those widths): the PREFIX felt and the factor felt.
pub const fn grind_check_const_felts() -> usize {
    2
}
