//! The hash a Fiat-Shamir transcript runs on.
//!
//! [`DefaultTranscript`](super::default_transcript::DefaultTranscript) is a thin
//! `digest::Digest` wrapper, so swapping the hash is a type substitution. What
//! this trait adds beyond naming a digest is that the name travels with the
//! proof: a transcript is part of what a replaying verifier has to reproduce,
//! so the configuration has to be something a call site can state rather than
//! something a default decides.
//!
//! ★ **The two constants, and why they arrived together.** A straight-line
//! machine replaying the transcript needs a draw count that does not depend on
//! the data, which is what [`TranscriptHash::CANDIDATES_PER_COORDINATE`] states.
//! For an algebraic sponge the count is one, because a squeeze is already field
//! elements — but only if the sampler sees them as the sponge produced them.
//!
//! An earlier revision of this file argued the opposite, and the argument was
//! correct at the time: [`DefaultTranscript::sample`] reversed all 32 bytes of
//! every squeeze, so the first eight bytes a sampler read were the LAST felt's
//! canonical bytes backwards — a number with no canonicality property, for
//! which a one-candidate schedule would have been a claim no test could reach.
//! [`TranscriptHash::REVERSES_SQUEEZE`] is what removed that obstacle, so the
//! two constants are one change: the schedule is a consequence of the byte
//! order, not an independent decision.

use digest::{Digest, FixedOutputReset, OutputSizeUser, typenum::U32};

use crate::hash::platform_keccak::PlatformKeccak256;
use crate::hash::rpx::Rpx256Digest;

/// One Fiat-Shamir configuration: the digest the sponge runs on.
pub trait TranscriptHash: 'static {
    /// The sponge's hash.
    ///
    /// `Clone` because the transcript is snapshotted (the GPU FRI fallback
    /// restores it) and because `state()` finalizes a clone. `FixedOutputReset`
    /// because the squeeze is `finalize_reset`. The 32-byte output size is
    /// pinned rather than left associated: `state()` returns `[u8; 32]`, and
    /// that is what seeds grinding, so a configuration with a different digest
    /// width would not be a drop-in anywhere it is consumed. `'static` because
    /// the GPU grinding dispatch keys the device search on the concrete digest
    /// by `TypeId`, the way the Merkle backends' keccak fast paths do.
    type Digest: Digest + FixedOutputReset + OutputSizeUser<OutputSize = U32> + Clone + 'static;

    /// Name for KATs, banners and diagnostics.
    const NAME: &'static str;

    /// ★ Whether [`DefaultTranscript::sample`] reverses the 32 bytes of a
    /// squeeze before handing them out and chaining them back in.
    ///
    /// A byte convention, not a security parameter: reversing 32 bytes is a
    /// bijection, so the distribution a sampler draws from is the digest's
    /// either way. What it decides is *which* bijection sits between the digest
    /// and the sampler, and for an algebraic sponge that is the whole question
    /// — see [`CANDIDATES_PER_COORDINATE`](Self::CANDIDATES_PER_COORDINATE).
    ///
    /// ⚠ It is `true` for keccak because that is the convention this system's
    /// proofs have always been produced under, and moving it would change every
    /// keccak proof on this branch for no gain. It is not `true` for any reason
    /// a new configuration should copy.
    const REVERSES_SQUEEZE: bool;

    /// ★ How many `u64` candidates a base-field coordinate needs, when that is
    /// a fixed number.
    ///
    /// `None` means unbounded: the sampler rejects candidates `>= p` and draws
    /// again, however many times that takes. That is the honest description of
    /// a byte-oriented hash, whose squeeze is 32 uniform bytes with no relation
    /// to the field — there is no bound, only a probability.
    ///
    /// `Some(n)` is a promise that `n` candidates always suffice, and it exists
    /// for a replaying verifier that cannot branch on how many it needed. A
    /// configuration may only claim it if the claim is structural. `Some(1)`
    /// here means a squeeze IS field elements, canonically encoded, so the
    /// rejection test is unreachable rather than merely unlikely.
    ///
    /// ⚠ Scope: this governs `sample_field_element` alone. Query indices come
    /// from `sample_u64`, whose rejection tests `candidate >= 2^64 mod bound` —
    /// a different test, about which canonicality says nothing. Those draws are
    /// single-candidate on this branch for an unrelated reason (every WHIR
    /// query bound is a power of two, making that threshold zero), which is
    /// hash-independent and pinned separately. Two facts, two reasons; a
    /// verifier that needs both must not take this constant as evidence of the
    /// other.
    const CANDIDATES_PER_COORDINATE: Option<usize>;
}

/// The keccak-256 configuration — what every `DefaultTranscript` is unless a
/// caller says otherwise, and byte-for-byte the transcript this system has
/// always produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeccakTranscriptHash;

impl TranscriptHash for KeccakTranscriptHash {
    type Digest = PlatformKeccak256;

    const NAME: &'static str = "keccak256";

    /// The convention every keccak proof on this branch was produced under.
    const REVERSES_SQUEEZE: bool = true;

    /// Unbounded, and it has to be: a keccak squeeze is 32 uniform bytes, so a
    /// candidate lands in `[p, 2^64)` with probability about `2^-32` per draw
    /// and the number of draws has no ceiling. Small is not fixed.
    const CANDIDATES_PER_COORDINATE: Option<usize> = None;
}

/// The RPX256 configuration — the algebraic sponge, for a transcript a
/// field-native verifier has to replay.
///
/// ★ **Why this one does not reverse, and what that buys.** A squeeze here is
/// `digest_to_commitment(sponge_leaf_bytes(..))` — four canonical felts, each
/// eight big-endian bytes. Handed out in that order, every 8-byte group a
/// sampler reads is a field element by construction, so the rejection test is
/// unreachable and one candidate per coordinate is exact rather than typical.
/// Reversed, the first group is the LAST felt's bytes backwards, a number with
/// no canonicality property at all — which is why an earlier revision of this
/// file argued `Some(1)` could not be claimed. It could not, then.
///
/// The reversal had no security role to lose: it is a bijection on 32 bytes, so
/// challenges are the digest's distribution before and after. What it cost was
/// the LFM replay — four byte-reversals and four modular reductions per squeeze
/// that a field-native verifier has to pay in rows to undo an encoding the host
/// had no reason to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RpxTranscriptHash;

impl TranscriptHash for RpxTranscriptHash {
    type Digest = Rpx256Digest;

    const NAME: &'static str = "rpx256";

    const REVERSES_SQUEEZE: bool = false;

    /// Structural, not probabilistic: see the type's documentation.
    const CANDIDATES_PER_COORDINATE: Option<usize> = Some(1);
}
