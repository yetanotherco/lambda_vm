//! The BLAKE3 compression function with the round count as a parameter, and the
//! byte hash the RV64 prover's commitments are built from.
//!
//! # Why this lives in `crypto`
//!
//! It has three callers that cannot share a copy any other way. The Merkle
//! backends in [`crate::merkle_tree::backends`] are in this crate; the
//! `LFM_BLAKE3` chip and the `LFM_HASH` socket are in `prover`, which depends on
//! this crate; and the CUDA kernels are checked against it from `math-cuda`,
//! which `prover` depends on. `crypto` is the only place all three can reach, so
//! the compression function is defined here once and re-exported upward —
//! `prover::lfm::blake3` is a re-export of this module, not a second
//! implementation. A chip and a commitment backend that hash identically because
//! they call one function is a different claim from two that agree today.
//!
//! # The round count
//!
//! [`BLAKE3_ROUNDS`] is 7 — standard BLAKE3 — unless the crate's
//! `blake3-6round` feature is on, and then it is 6. The polarity is deliberate
//! and must not be inverted: every existing measurement and sign-off reads
//! "7-round instantiated baseline, 6 behind the feature".
//!
//! The knob is crate-global rather than a generic parameter so that one build
//! cannot produce two hashes. Crates above re-export it rather than defining
//! their own, and `prover`'s `blake3-6round` feature forwards to this one, so
//! the chip's round count and the commitment's round count are the same symbol.
//! `math-cuda` necessarily has its own (it compiles a cubin), which is why it
//! exports the compiled-in count for a caller to assert instead of discover.
//!
//! # Provenance of the primitive, and why no external KAT exists at 6 rounds
//!
//! Vendored from PR #903 (`yetanotherco/lambda_vm`, head
//! `89aeeb8c2b0389e9d21a861c9e3a10a7b1b5704e`). Standing-decisions rule 9
//! requires pinning a new primitive against an external known-answer vector that
//! nothing in this repository produced. That is *impossible in the usual form*
//! for the 6-round variant: it is not standard BLAKE3, so no published vector
//! and no crate exposes it. The provenance chain #903 supplies instead:
//!
//! 1. A z3-proved model of the compression dataflow
//!    (`thoughts/blake3/blake3-chip/z3_blake_verify.py`).
//! 2. A Python oracle (`thoughts/blake3/blake3-oracle/blake3_ref.py`) whose
//!    **7-round** instantiation is pinned against the official `blake3` crate's
//!    published test vectors — so the oracle's G-function, message schedule,
//!    counter split and feed-forward are all externally validated; only the
//!    round count is varied.
//! 3. That oracle at `rounds = 6` emitted the 10 canonical vectors in
//!    [`CANONICAL_VECTORS`], which pin this port.
//!
//! So the external anchor is one step removed: the *conventions* are pinned by
//! the official crate through the oracle, and the round count is the single
//! degree of freedom the canonical vectors add. That is weaker than a direct
//! KAT and is recorded as such — but [`CANONICAL_VECTORS`] still discriminates
//! every convention a wrong port could get wrong, which the falsification tests
//! in `prover::lfm::blake3` demonstrate one convention at a time.
//!
//! [`chain`] extends the anchor considerably: at 7 rounds [`Blake3Chain`] is the
//! `blake3` crate's full hash for every message up to 1024 bytes, so the framing
//! — not just the round function — is externally checked.
//!
//! ⚠ Security assumption **A6R**: collision resistance of the 6-round variant
//! is a named, unratified assumption (#903's `IMPLEMENTATION.md`). Nothing here
//! ratifies it.

pub mod chain;
mod vectors;

pub use chain::{Blake3Chain, blake3_chain};
pub use vectors::{CANONICAL_OUT_7ROUND, CANONICAL_VECTORS, Vector};

/// The BLAKE3 IV (identical to SHA-256's initial state). `IV[0..4]` seeds
/// `v[8..12]` of the compression working state.
pub const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// The BLAKE3 message-schedule permutation, applied between rounds
/// (`m'[i] = m[MSG_PERMUTATION[i]]`).
pub const BLAKE3_MSG_PERMUTATION: [usize; 16] =
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// Rounds of *standard* BLAKE3. At this value [`blake3_compress_rounds`] is
/// bit-for-bit the published compression function — the property the whole
/// external-anchor argument rests on, pinned by
/// `tests::seven_rounds_is_the_blake3_crate`.
pub const BLAKE3_STANDARD_ROUNDS: usize = 7;

/// Rounds of the 6-round internal variant. Reachable only through the
/// `blake3-6round` feature; [`CANONICAL_VECTORS`] pin it unconditionally.
pub const BLAKE3_SIX_ROUNDS: usize = 6;

/// The round count every BLAKE3 chip in this tree is compiled for — the
/// standalone `LFM_BLAKE3` probe and the `LFM_HASH` socket arm alike. They share
/// one knob deliberately: two would let a sweep leave the two chips describing
/// different hashes.
///
/// **7 by default**, i.e. standard BLAKE3, which is what the A6R sign-off
/// instantiates. At 7 rounds the `blake3` crate is a direct known-answer test
/// for the primitive *and* for the socket, and no unratified assumption is
/// carried. `--features blake3-6round` selects the 6-round internal variant:
/// the measured performance variant, resting on **A6R**.
///
/// It is a compile-time constant rather than a parameter because both chips'
/// column layouts are `8 · rounds` G-blocks wide and their width functions are
/// `const fn`. The round count is the ONLY thing it varies — the G function, the
/// message schedule, the counter split and the feed-forward are fixed — which is
/// what lets the 7-round anchor certify the whole code path rather than a
/// separate 7-round copy of it.
#[cfg(not(feature = "blake3-6round"))]
pub const BLAKE3_ROUNDS: usize = BLAKE3_STANDARD_ROUNDS;
#[cfg(feature = "blake3-6round")]
pub const BLAKE3_ROUNDS: usize = BLAKE3_SIX_ROUNDS;

/// The BLAKE3 quarter-round G (spec §2.1).
#[inline]
fn blake3_g(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(mx);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(12);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(my);
    v[d] = (v[d] ^ v[a]).rotate_right(8);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(7);
}

/// The BLAKE3 compression function `f` at 6 rounds (spec §2.2, oracle §2.4).
///
/// State init: `v[0..8] = h`, `v[8..12] = IV[0..4]`, `v[12] = t as u32`,
/// `v[13] = (t >> 32) as u32`, `v[14] = block_len`, `v[15] = flags`. Six rounds
/// of 8 G-calls (4 columns then 4 diagonals), permuting the message schedule
/// between rounds (`r < rounds - 1`, i.e. 5 permutes — the trailing permute is
/// never consumed). Feed-forward: `out[i] = v[i] ^ v[i+8]`,
/// `out[i+8] = v[i+8] ^ h[i]`. The truncated chaining value is `out[0..8]`.
pub fn blake3_compress_6round(
    h: &[u32; 8],
    m: &[u32; 16],
    t: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 16] {
    blake3_compress_rounds(h, m, t, block_len, flags, BLAKE3_SIX_ROUNDS)
}

/// [`blake3_compress_6round`] with the round count as an argument.
///
/// The round count is the *only* parameter: everything else — the G function,
/// the message schedule, the counter split, the feed-forward — is fixed. That
/// is what makes `rounds = BLAKE3_STANDARD_ROUNDS` an external anchor for the
/// whole code path rather than for a separate 7-round copy of it, and it is why
/// this is one function with a loop bound instead of two functions.
pub fn blake3_compress_rounds(
    h: &[u32; 8],
    m: &[u32; 16],
    t: u64,
    block_len: u32,
    flags: u32,
    rounds: usize,
) -> [u32; 16] {
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
        t as u32,
        (t >> 32) as u32,
        block_len,
        flags,
    ];

    let mut m = *m;
    for r in 0..rounds {
        // Mix the columns.
        blake3_g(&mut v, 0, 4, 8, 12, m[0], m[1]);
        blake3_g(&mut v, 1, 5, 9, 13, m[2], m[3]);
        blake3_g(&mut v, 2, 6, 10, 14, m[4], m[5]);
        blake3_g(&mut v, 3, 7, 11, 15, m[6], m[7]);
        // Mix the diagonals.
        blake3_g(&mut v, 0, 5, 10, 15, m[8], m[9]);
        blake3_g(&mut v, 1, 6, 11, 12, m[10], m[11]);
        blake3_g(&mut v, 2, 7, 8, 13, m[12], m[13]);
        blake3_g(&mut v, 3, 4, 9, 14, m[14], m[15]);
        // Permute between rounds; the permute after the last round is never
        // consumed (oracle: `r < rounds - 1`).
        if r < rounds - 1 {
            let prev = m;
            for (i, &p) in BLAKE3_MSG_PERMUTATION.iter().enumerate() {
                m[i] = prev[p];
            }
        }
    }

    let mut out = [0u32; 16];
    for i in 0..8 {
        out[i] = v[i] ^ v[i + 8];
        out[i + 8] = v[i + 8] ^ h[i];
    }
    out
}

/// The 16-word output of `CANONICAL_VECTORS[i]` at the compiled-in
/// [`BLAKE3_ROUNDS`] — what a chip built from this module must produce.
pub const fn canonical_expected_out(i: usize) -> [u32; 16] {
    if BLAKE3_ROUNDS == BLAKE3_STANDARD_ROUNDS {
        CANONICAL_OUT_7ROUND[i]
    } else {
        CANONICAL_VECTORS[i].out
    }
}
