//! [`Blake3Chain`] — the byte hash the RV64 prover's commitments are built from.
//!
//! **Specified in `thoughts/shared/block-compression/PA-PLAN.md` §1.7, which is
//! the normative text; this is the implementation of it.** The construction is
//! DRAFT: it is the working default by standing decision, and formally awaits
//! ratification. §1.7.3 lists the forks that are still open.
//!
//! # The construction, in one sentence
//!
//! **Standard BLAKE3 restricted to a single chunk that never ends.** The message
//! is split into 64-byte blocks, the last zero-padded; the chaining value starts
//! at [`BLAKE3_IV`] and each block compresses it forward with `t = 0`; the first
//! block carries `CHUNK_START`, the last carries `CHUNK_END | ROOT` and the true
//! byte count as its `block_len`. The digest is the low 8 output words,
//! little-endian.
//!
//! # Why this shape
//!
//! It is chosen so that two other things are true by construction rather than by
//! agreement, which is the whole reason to prefer it to a bare chain with one
//! flag constant:
//!
//! - **For any message of at most 1024 bytes, at 7 rounds, this IS
//!   `blake3::hash`.** Standard BLAKE3's first chunk is exactly this chain, and a
//!   message of at most one chunk has that chunk's output as its root — so `ROOT`
//!   lands on the same compression. The official crate is therefore a direct
//!   known-answer test for the *framing*, not merely for the round function, over
//!   the entire range that matters: leaves, FRI pairs and parents are all far
//!   inside it. `seven_round_chain_is_the_blake3_crate` is that test.
//! - **A 64-byte message is exactly a Merkle parent.** One block, first and last,
//!   so `flags = 0x0B`, `block_len = 64`, `h = IV`, `t = 0` — which is what
//!   `hash_new_parent` compresses and what the device kernel
//!   (`math-cuda/kernels/blake3.cu`) implements. So the `StarkHash` invariant
//!   that `Batched::hash_data(&vec![a, b]) == Pair::hash_data(&[a, b])` holds
//!   because both are the same 64 bytes through this one function.
//!
//! Above 1024 bytes it deliberately leaves the standard: BLAKE3 would start a
//! second chunk (`t = 1`, chaining value reset to `IV`) and build a tree over
//! chunk chaining values. Keeping one unbounded chunk costs nothing at 6 rounds,
//! where no external verifier exists in any case, and saves both a chunk-tree
//! state machine in every CUDA kernel and the same state machine again in the
//! wrap's eDSL emitter. `past_one_chunk_leaves_the_blake3_crate` pins that the
//! divergence is real, so the claim is falsifiable rather than decorative.

use digest::{FixedOutput, FixedOutputReset, HashMarker, Output, OutputSizeUser, Reset, Update};

use super::{BLAKE3_IV, BLAKE3_ROUNDS, blake3_compress_rounds};

/// Bytes in one BLAKE3 message block.
pub const BLOCK_LEN: usize = 64;

/// This block begins the chunk. Set on the first block only.
const CHUNK_START: u32 = 1;
/// This block ends the chunk. Set on the last block only.
const CHUNK_END: u32 = 2;
/// This compression produces the root output. Set on the last block only —
/// a single-chunk message's chunk output *is* its root output.
const ROOT: u32 = 8;

/// The flags of a message that is one block long: first and last at once. Equal
/// to `CHUNK_START | CHUNK_END | ROOT`, and the framing every Merkle parent uses.
pub const FLAGS_ONE_BLOCK: u32 = CHUNK_START | CHUNK_END | ROOT;

/// [`Blake3Chain`] as a one-shot over a byte slice.
///
/// The streaming type and this agree by construction — this *is* the streaming
/// type, fed once.
pub fn blake3_chain(data: &[u8]) -> [u8; 32] {
    blake3_chain_rounds(data, BLAKE3_ROUNDS)
}

/// [`blake3_chain`] with the round count as an argument.
///
/// The round count is the only parameter, exactly as in
/// [`blake3_compress_rounds`] and for the same reason: at
/// [`BLAKE3_STANDARD_ROUNDS`](super::BLAKE3_STANDARD_ROUNDS) the result is the
/// `blake3` crate's, so that arm certifies this whole code path — the block
/// splitting, the padding, the flag schedule, the `block_len` — and the 6-round
/// arm differs from it by a loop bound alone.
pub fn blake3_chain_rounds(data: &[u8], rounds: usize) -> [u8; 32] {
    let mut chain = Blake3Chain::with_rounds(rounds);
    chain.update(data);
    chain.finalize_digest()
}

/// The single-chunk BLAKE3 chain as an incremental hasher.
///
/// Implements the `digest` traits, so it drops into the Merkle backends and the
/// transcript anywhere a `D: Digest` is expected — the same way
/// [`PlatformKeccak256`](crate::hash::platform_keccak::PlatformKeccak256) does.
/// That is what lets the batched and paired backends be one hash rather than two
/// implementations that have to be shown to coincide.
///
/// A full block is held rather than compressed until more input arrives, because
/// the final block's flags and `block_len` differ from every other block's and
/// whether a block is final is not known until the message ends.
#[derive(Clone)]
pub struct Blake3Chain {
    /// The chaining value: `IV`, then the truncated output of each compressed
    /// block. Never reset — that is the "single chunk" of the construction.
    cv: [u32; 8],
    /// The pending block, zero-padded. Zeroing on reset is what pads the final
    /// partial block.
    block: [u8; BLOCK_LEN],
    /// Bytes of `block` that are message, `0..=BLOCK_LEN`.
    block_len: usize,
    /// Whether any block has been compressed yet — i.e. whether the pending
    /// block still carries `CHUNK_START`.
    started: bool,
    /// Rounds. [`BLAKE3_ROUNDS`] for every production instance; see
    /// [`Self::with_rounds`].
    rounds: usize,
}

impl Default for Blake3Chain {
    fn default() -> Self {
        Self::with_rounds(BLAKE3_ROUNDS)
    }
}

impl Blake3Chain {
    /// A hasher at the crate-global [`BLAKE3_ROUNDS`]. The only constructor any
    /// production path uses; [`Default`] and `Digest::new` are this.
    pub fn new() -> Self {
        Self::default()
    }

    /// A hasher at an explicit round count.
    ///
    /// **For anchoring and known-answer tests only.** The production round count
    /// is a compile-time crate-global on purpose — a per-instance one would let a
    /// single build commit under two different hashes, which is the failure the
    /// `SOCKET_ROUNDS == BLAKE3_ROUNDS` assertion in `prover` exists to prevent.
    /// It is exposed because the 7-round arm is the external anchor for the
    /// 6-round one, so both must be reachable from one build's tests.
    pub fn with_rounds(rounds: usize) -> Self {
        Self {
            cv: BLAKE3_IV,
            block: [0u8; BLOCK_LEN],
            block_len: 0,
            started: false,
            rounds,
        }
    }

    /// The pending block as 16 little-endian message words.
    fn block_words(&self) -> [u32; 16] {
        core::array::from_fn(|i| {
            u32::from_le_bytes([
                self.block[4 * i],
                self.block[4 * i + 1],
                self.block[4 * i + 2],
                self.block[4 * i + 3],
            ])
        })
    }

    /// The pending block's flags. `CHUNK_START` while nothing has been
    /// compressed yet; `CHUNK_END | ROOT` when this is the message's last block.
    fn flags(&self, is_final: bool) -> u32 {
        let start = if self.started { 0 } else { CHUNK_START };
        let end = if is_final { CHUNK_END | ROOT } else { 0 };
        start | end
    }

    /// Fold the pending block — known not to be the last — into the chaining
    /// value, and clear the block so the next one is zero-padded.
    fn compress_pending(&mut self) {
        let out = blake3_compress_rounds(
            &self.cv,
            &self.block_words(),
            0,
            BLOCK_LEN as u32,
            self.flags(false),
            self.rounds,
        );
        self.cv.copy_from_slice(&out[..8]);
        self.block = [0u8; BLOCK_LEN];
        self.block_len = 0;
        self.started = true;
    }

    /// Absorb more message. Identical results for any split of the same bytes —
    /// `streaming_splits_agree_with_one_shot`.
    pub fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            // Only now is the pending block known not to be the last one.
            if self.block_len == BLOCK_LEN {
                self.compress_pending();
            }
            let take = (BLOCK_LEN - self.block_len).min(input.len());
            self.block[self.block_len..self.block_len + take].copy_from_slice(&input[..take]);
            self.block_len += take;
            input = &input[take..];
        }
    }

    /// The 32-byte digest: one final compression over the pending block, with
    /// the true byte count as `block_len` and `CHUNK_END | ROOT` set.
    ///
    /// The empty message takes this path with an all-zero block and
    /// `block_len = 0`, which is one compression, not zero — and is what
    /// `blake3::hash(b"")` is at 7 rounds.
    pub fn finalize_digest(&self) -> [u8; 32] {
        let out = blake3_compress_rounds(
            &self.cv,
            &self.block_words(),
            0,
            self.block_len as u32,
            self.flags(true),
            self.rounds,
        );
        let mut digest = [0u8; 32];
        for i in 0..8 {
            digest[4 * i..4 * i + 4].copy_from_slice(&out[i].to_le_bytes());
        }
        digest
    }
}

impl HashMarker for Blake3Chain {}

impl OutputSizeUser for Blake3Chain {
    type OutputSize = digest::typenum::U32;
}

impl Update for Blake3Chain {
    fn update(&mut self, data: &[u8]) {
        Blake3Chain::update(self, data);
    }
}

impl FixedOutput for Blake3Chain {
    fn finalize_into(self, out: &mut Output<Self>) {
        out.copy_from_slice(&self.finalize_digest());
    }
}

impl Reset for Blake3Chain {
    fn reset(&mut self) {
        *self = Self::with_rounds(self.rounds);
    }
}

impl FixedOutputReset for Blake3Chain {
    fn finalize_into_reset(&mut self, out: &mut Output<Self>) {
        out.copy_from_slice(&self.finalize_digest());
        Reset::reset(self);
    }
}

/// The message the KAT table is taken over, at a given length.
///
/// Byte `i` is `37i + 11 (mod 256)`: every length is a different message, no
/// byte value repeats within a block, and it is the same generator the existing
/// compression-level anchor in `prover::lfm::blake3` uses.
pub const fn kat_message_byte(i: usize) -> u8 {
    (i as u8).wrapping_mul(37).wrapping_add(11)
}

/// The lengths [`CHAIN_KAT_6ROUND`] covers, in order. PA-PLAN §1.7.4 says what
/// each one discriminates: the empty message is one block (0); `block_len` is
/// the true length and the tail is zero-padded (1, 31, 63); a 64-byte message is
/// the parent form (64); the chain's first step moves `CHUNK_END | ROOT` off
/// block 0 (65); an exact multiple of 64 emits no spurious final block (128);
/// interior blocks carry no flags (192, 256, 1024); and 1088 is the first length
/// past one chunk, where this construction leaves standard BLAKE3 (1088).
pub const CHAIN_KAT_LENS: [usize; 12] = [0, 1, 31, 63, 64, 65, 127, 128, 192, 256, 1024, 1088];

/// [`blake3_chain`] at **6 rounds** over `kat_message_byte` messages of each
/// [`CHAIN_KAT_LENS`] length.
///
/// # What this table is, and how strong its provenance actually is
///
/// It is a regression pin — generated from this implementation and committed, so
/// a later refactor cannot change the construction silently. But it is more than
/// that, and the difference is worth stating precisely because the compression
/// vectors it sits next to are weaker.
///
/// Every entry from length 0 to 1024 was **independently reproduced** by #903's
/// Python oracle (`thoughts/blake3/blake3-oracle/blake3_ref.py`) evaluated at
/// `rounds = 6`, on 2026-08-14. That oracle is a full standard-BLAKE3
/// implementation with the round count as a parameter, written by another author
/// for a different purpose, and at `rounds = 7` it reproduces the official
/// `blake3` package bit-for-bit at every length checked — including the
/// multi-chunk ones. So for the whole ≤1-chunk range these digests are not a
/// self-consistency check: two implementations that share no code agree, and
/// the conventions they agree on are pinned to the published hash from outside.
///
/// Length 1088 is where they part, and that is the point of including it: the
/// oracle stays standard past one chunk and this construction does not (P3).
/// Being able to say the divergence is *the chunking* rather than the round
/// count needs a reference that is standard at 6 rounds too, which is exactly
/// what the oracle is.
///
/// ⚠ The oracle survives only as `__pycache__` bytecode in an untracked
/// directory; its `.py` source is gone. The cross-check is recorded in PA-PLAN
/// §1.7.5 with the digests, so the result outlives the artifact even though
/// re-running it may not be possible.
///
/// The 7-round arm remains the primary anchor and needs none of this:
/// `blake3_chain_rounds(m, 7)` is checked directly against the `blake3` crate
/// over all 1025 lengths, with no table in between.
pub const CHAIN_KAT_6ROUND: [[u8; 32]; 12] = [
    // len 0
    [
        0x3C, 0x3B, 0xBB, 0x1F, 0x33, 0x5A, 0x31, 0xEA, 0x86, 0x46, 0x4B, 0x65, 0x1C, 0x02, 0x06,
        0xFC, 0x81, 0xD3, 0x32, 0x62, 0xAE, 0x00, 0xEA, 0x1A, 0x65, 0xF3, 0xD1, 0xD0, 0x4A, 0xFA,
        0xEF, 0xC9,
    ],
    // len 1
    [
        0x2A, 0x50, 0xE4, 0x5B, 0x89, 0x21, 0xF9, 0xEF, 0xA0, 0x08, 0xD9, 0xF3, 0x9F, 0x71, 0x65,
        0x60, 0x0C, 0xF4, 0x8A, 0x7F, 0x0E, 0x85, 0x9C, 0x21, 0x22, 0xE3, 0xCC, 0xB6, 0xB9, 0x67,
        0x7E, 0xE5,
    ],
    // len 31
    [
        0xC3, 0x8B, 0xF6, 0x2F, 0x50, 0x60, 0x40, 0xB2, 0x60, 0x02, 0x73, 0x77, 0x8D, 0x28, 0x1B,
        0x89, 0x43, 0x62, 0x1E, 0x2B, 0x8A, 0x9F, 0x59, 0xE2, 0x37, 0x9F, 0x8F, 0xD7, 0xE5, 0xC8,
        0x51, 0x25,
    ],
    // len 63
    [
        0xC3, 0x73, 0xF5, 0x1A, 0x5E, 0xB8, 0xB2, 0x7E, 0xA0, 0x5B, 0xB1, 0xF6, 0xF4, 0xE6, 0x2E,
        0x92, 0x4F, 0xF4, 0xD8, 0xA2, 0x79, 0xF0, 0xD0, 0x5A, 0xFA, 0x5C, 0xD5, 0x19, 0x39, 0x1D,
        0x63, 0x89,
    ],
    // len 64
    [
        0x59, 0x00, 0xA1, 0xE3, 0x98, 0xBB, 0x2B, 0xF6, 0xD3, 0xBA, 0x7F, 0x1A, 0x29, 0x19, 0x7B,
        0x79, 0xC8, 0x6B, 0x71, 0xAD, 0x2C, 0x26, 0x31, 0xF4, 0xAC, 0x73, 0x6C, 0x82, 0xDB, 0x04,
        0x3C, 0xB5,
    ],
    // len 65
    [
        0x53, 0x95, 0x3F, 0xCA, 0xDC, 0x39, 0xB8, 0x62, 0x39, 0x01, 0xAF, 0x7B, 0x53, 0x4F, 0x2F,
        0x69, 0x33, 0xE3, 0x12, 0xF5, 0x02, 0x99, 0x33, 0x13, 0x34, 0xE6, 0xC0, 0xA7, 0xC9, 0xDB,
        0xC2, 0xBE,
    ],
    // len 127
    [
        0x9E, 0x0D, 0xD8, 0x16, 0x8D, 0x19, 0x9A, 0x04, 0x59, 0x0C, 0x2C, 0xBA, 0x43, 0x9B, 0x27,
        0x07, 0x76, 0xE4, 0x27, 0x15, 0xD5, 0x18, 0xF6, 0x86, 0x55, 0xE5, 0x66, 0x92, 0x48, 0x3E,
        0x50, 0x5E,
    ],
    // len 128
    [
        0x5C, 0xAF, 0xFC, 0x87, 0x84, 0xE8, 0x17, 0xBB, 0xBA, 0x99, 0x1B, 0x21, 0x08, 0xC2, 0x6A,
        0x3D, 0xFD, 0xF8, 0x04, 0x24, 0x5E, 0xF6, 0x3A, 0xE1, 0x04, 0x0A, 0x3C, 0x34, 0xF1, 0xB3,
        0x62, 0xFF,
    ],
    // len 192
    [
        0x39, 0x9D, 0x6B, 0x9A, 0xDE, 0xB2, 0xF8, 0x84, 0x50, 0x77, 0x5F, 0x77, 0x3E, 0x9D, 0xEC,
        0x08, 0x83, 0x6C, 0x13, 0x57, 0x13, 0xC2, 0xC5, 0xDD, 0x09, 0xF4, 0xCE, 0xCE, 0xB0, 0xED,
        0x38, 0x88,
    ],
    // len 256
    [
        0xFB, 0xCA, 0xB3, 0x69, 0x9A, 0x49, 0x59, 0xFA, 0x37, 0x19, 0x0E, 0x98, 0xCA, 0x51, 0x42,
        0xDD, 0xBC, 0x88, 0x33, 0x0F, 0x2E, 0x7D, 0x12, 0x33, 0x5D, 0xB9, 0xC6, 0xC8, 0x88, 0x1A,
        0x0B, 0x87,
    ],
    // len 1024
    [
        0xF3, 0x95, 0xE7, 0xE2, 0x15, 0x03, 0x63, 0xB6, 0xD2, 0x00, 0x48, 0x75, 0x15, 0x42, 0x5B,
        0x02, 0x04, 0xEE, 0xA4, 0x24, 0x07, 0x21, 0x83, 0xB7, 0x01, 0x17, 0x6E, 0xCC, 0xBE, 0x0F,
        0xFE, 0x1B,
    ],
    // len 1088
    [
        0xB4, 0x73, 0x8E, 0xDE, 0x77, 0xA6, 0xEC, 0x16, 0x6E, 0xE9, 0x76, 0x67, 0x11, 0x8D, 0x47,
        0x93, 0xCB, 0xF2, 0xB0, 0x8B, 0x45, 0xAA, 0xC7, 0xC6, 0xD5, 0x29, 0x43, 0xB5, 0xD2, 0x98,
        0xC6, 0x88,
    ],
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::blake3::{BLAKE3_SIX_ROUNDS, BLAKE3_STANDARD_ROUNDS};
    use alloc::vec::Vec;

    fn message(len: usize) -> Vec<u8> {
        (0..len).map(kat_message_byte).collect()
    }

    /// ★ **The external anchor.** At 7 rounds this construction is the `blake3`
    /// crate's hash, for every message length up to one full chunk — no oracle,
    /// no JSON, no transcription.
    ///
    /// The range is what makes it worth more than a compression-level anchor:
    /// it pins the block splitting, the zero padding, the `block_len` of the
    /// final block, the `CHUNK_START`/`CHUNK_END`/`ROOT` schedule and the
    /// little-endian digest read-back, at every boundary those can be wrong at.
    #[test]
    fn seven_round_chain_is_the_blake3_crate() {
        for len in 0..=1024usize {
            let msg = message(len);
            assert_eq!(
                blake3_chain_rounds(&msg, BLAKE3_STANDARD_ROUNDS),
                *blake3::hash(&msg).as_bytes(),
                "the 7-round chain must equal the blake3 crate at length {len}"
            );
        }
    }

    /// NEGATIVE CONTROL for the anchor: at 6 rounds it must not match, or the
    /// test above would pass just as well with `rounds` ignored — the one bug
    /// that would make the whole external-anchor argument vacuous.
    #[test]
    fn six_round_chain_is_not_the_blake3_crate() {
        for len in [0usize, 1, 64, 65, 128, 1024] {
            let msg = message(len);
            assert_ne!(
                blake3_chain_rounds(&msg, BLAKE3_SIX_ROUNDS),
                *blake3::hash(&msg).as_bytes(),
                "length {len}"
            );
        }
    }

    /// ★ **P3, stated as a test.** Past one chunk the construction deliberately
    /// leaves standard BLAKE3 — the standard would start chunk 1 and build a
    /// tree, this keeps chaining. Without this, "we implement the single-chunk
    /// chain" would be an unfalsifiable claim: the anchor above would pass
    /// identically if we had implemented the whole chunk tree instead.
    ///
    /// 1024 is the last length where they agree and 1088 the first block past
    /// it, so the two assertions together locate the divergence exactly.
    #[test]
    fn past_one_chunk_leaves_the_blake3_crate() {
        let last_agreeing = message(1024);
        assert_eq!(
            blake3_chain_rounds(&last_agreeing, BLAKE3_STANDARD_ROUNDS),
            *blake3::hash(&last_agreeing).as_bytes(),
            "1024 bytes is still one chunk and must agree"
        );
        for len in [1025usize, 1088, 2048] {
            let msg = message(len);
            assert_ne!(
                blake3_chain_rounds(&msg, BLAKE3_STANDARD_ROUNDS),
                *blake3::hash(&msg).as_bytes(),
                "past one chunk the constructions must differ, at length {len}"
            );
        }
    }

    /// ★ **P2** — a 64-byte message is exactly the Merkle parent form: one
    /// compression, `h = IV`, `t = 0`, `block_len = 64`, `flags = 0x0B`.
    ///
    /// This is what makes the `StarkHash` two-element invariant hold by
    /// construction, and it is the framing the device kernel implements. Written
    /// out as an explicit compression rather than as "whatever the code does",
    /// so it fails if the flag schedule or the counter moves.
    #[test]
    fn a_sixty_four_byte_message_is_the_parent_form() {
        let left: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(7));
        let right: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(31).wrapping_add(3));
        let mut msg = [0u8; 64];
        msg[..32].copy_from_slice(&left);
        msg[32..].copy_from_slice(&right);

        for rounds in [BLAKE3_SIX_ROUNDS, BLAKE3_STANDARD_ROUNDS] {
            let words: [u32; 16] = core::array::from_fn(|i| {
                u32::from_le_bytes(msg[4 * i..4 * i + 4].try_into().unwrap())
            });
            let out = blake3_compress_rounds(&BLAKE3_IV, &words, 0, 64, FLAGS_ONE_BLOCK, rounds);
            let mut expected = [0u8; 32];
            for i in 0..8 {
                expected[4 * i..4 * i + 4].copy_from_slice(&out[i].to_le_bytes());
            }
            assert_eq!(
                blake3_chain_rounds(&msg, rounds),
                expected,
                "a 64-byte message must be one parent compression, at {rounds} rounds"
            );
        }
    }

    /// The `Update` contract: the digest depends on the bytes, not on how they
    /// were handed over. Splits are taken at and either side of every block
    /// boundary, which is where a mis-set `CHUNK_START` or a prematurely
    /// compressed final block would show.
    #[test]
    fn streaming_splits_agree_with_one_shot() {
        for len in [0usize, 1, 63, 64, 65, 127, 128, 129, 200] {
            let msg = message(len);
            let want = blake3_chain(&msg);
            for split in 0..=len {
                let mut chain = Blake3Chain::new();
                chain.update(&msg[..split]);
                chain.update(&msg[split..]);
                assert_eq!(
                    chain.finalize_digest(),
                    want,
                    "length {len} split at {split}"
                );
            }
            // Byte at a time, which crosses every boundary in the smallest
            // possible increments.
            let mut chain = Blake3Chain::new();
            for b in &msg {
                chain.update(&[*b]);
            }
            assert_eq!(chain.finalize_digest(), want, "length {len} byte at a time");
        }
    }

    /// The committed 6-round regression pin. See [`CHAIN_KAT_6ROUND`] for what
    /// this does and does not establish.
    #[test]
    fn six_round_chain_matches_the_committed_table() {
        for (i, &len) in CHAIN_KAT_LENS.iter().enumerate() {
            assert_eq!(
                blake3_chain_rounds(&message(len), BLAKE3_SIX_ROUNDS),
                CHAIN_KAT_6ROUND[i],
                "6-round chain KAT at length {len}"
            );
        }
    }

    /// NEGATIVE CONTROL for the table: the entries must be distinct data, or a
    /// generation bug that wrote one digest twelve times would leave the test
    /// above passing and pinning nothing.
    #[test]
    fn the_committed_table_entries_are_distinct() {
        for (i, a) in CHAIN_KAT_6ROUND.iter().enumerate() {
            for (j, b) in CHAIN_KAT_6ROUND.iter().enumerate().skip(i + 1) {
                assert_ne!(a, b, "KAT entries {i} and {j} are the same digest");
            }
        }
    }

    /// **P4** in its cheapest observable form: lengths that share a padded block
    /// must not share a digest. A construction that ignored `block_len` would
    /// collide 31 with 32, and one that ignored the flag schedule would collide
    /// 64 with 65's first block.
    #[test]
    fn lengths_sharing_a_padded_block_do_not_collide() {
        let mut seen: Vec<[u8; 32]> = Vec::new();
        for len in 0..=130usize {
            let digest = blake3_chain(&message(len));
            assert!(
                !seen.contains(&digest),
                "length {len} collides with a shorter message"
            );
            seen.push(digest);
        }
    }

    /// `Reset` really returns to the initial state, including the pending block
    /// and the `CHUNK_START` flag — a reset that kept `started` set would hash
    /// the next message under the wrong flags.
    #[test]
    fn reset_returns_to_the_initial_state() {
        let mut chain = Blake3Chain::new();
        chain.update(&message(100));
        Reset::reset(&mut chain);
        chain.update(&message(7));
        assert_eq!(chain.finalize_digest(), blake3_chain(&message(7)));
    }

    /// The `digest` route and the free function are the same hash — the backends
    /// reach this type through `Digest`, the KATs above through `blake3_chain`.
    #[test]
    fn the_digest_trait_route_agrees_with_the_free_function() {
        use digest::Digest;
        for len in [0usize, 1, 64, 65, 200] {
            let msg = message(len);
            let mut hasher = <Blake3Chain as Digest>::new();
            Digest::update(&mut hasher, &msg);
            let via_digest: [u8; 32] = Digest::finalize(hasher).into();
            assert_eq!(via_digest, blake3_chain(&msg), "length {len}");
        }
    }
}
