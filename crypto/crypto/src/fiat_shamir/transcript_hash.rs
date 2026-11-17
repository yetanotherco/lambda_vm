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

use core::num::NonZeroUsize;
use digest::{Digest, FixedOutputReset, OutputSizeUser, typenum::U32};

use crate::hash::blake3::chain::{Blake3Chain, Blake3Std};
use crate::hash::platform_keccak::PlatformKeccak256;
use crate::hash::rpx::Rpx256Digest;

/// ★★ Which [`TranscriptHash`] a concrete transcript type is running on.
///
/// The type-level answer to "what hash is this transcript", so a caller that
/// needs a transcript to match something else can say so in a `where` clause
/// and have the compiler check it.
///
/// # Why this exists
///
/// `DefaultTranscript`'s hash parameter has a default, so `DefaultTranscript::<E>`
/// is a keccak transcript and looks like it names no hash at all. Every WHIR
/// call site wrote exactly that, under a dispatch that selects the hash for the
/// Merkle backend and the grind — so the RPX configuration ran an RPX backend,
/// an RPX grind and a KECCAK transcript, for four measured A/Bs, without one
/// instrument disagreeing. Nothing failed: the proofs were valid and the two
/// arms genuinely differed.
///
/// Naming the hash at every call site would not have prevented it — a site can
/// name the wrong one as easily as it can take a default. What prevents it is
/// an equality the compiler checks, which is what this trait makes sayable:
/// `T: HasTranscriptHash<Hash = <H as WhirHash>::Transcript>` on the WHIR entry
/// points turns a mismatched transcript into a build error, and leaves the
/// several hundred STARK call sites — for which keccak is not a default but the
/// answer — untouched.
pub trait HasTranscriptHash {
    /// The configuration this transcript's sponge runs on.
    type Hash: TranscriptHash;
}

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
    /// ⚠ `GrindDigest` is part of the bound, not an afterthought. The
    /// proof-of-work hash IS this digest, and which device arm it takes has to
    /// travel with it: a configuration that could not say would fall to the
    /// host search silently, which is worth thousands of seconds a block and
    /// fails nothing. Requiring it here is what makes "a configuration with no
    /// declared arm" unconstructible rather than merely unlikely.
    type Digest: Digest
        + FixedOutputReset
        + OutputSizeUser<OutputSize = U32>
        + Clone
        + crate::grinding::GrindDigest
        + 'static;

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
    /// ⚠ `NonZeroUsize`, not `usize`: `Some(0)` would be a schedule that draws
    /// no candidates and returns an uninitialised one, and the representation
    /// that cannot spell it is better than the check that rejects it.
    const CANDIDATES_PER_COORDINATE: Option<NonZeroUsize>;

    /// ★ Bytes one squeeze hands to the sampler: [`SQUEEZE_BYTES`] (the
    /// digest), or [`WIDE_SQUEEZE_BYTES`] for a configuration whose hash's
    /// final compression has 64 bytes of output
    /// ([`Self::finalize_squeeze_reset`]).
    ///
    /// Only the first 32 bytes are re-absorbed to advance the chain, whatever
    /// this says: the second 32 are output the same compression already
    /// produced, so a wide squeeze hands out eight candidates for the cost of
    /// one compression where a narrow one hands out four.
    const SQUEEZE_LEN: usize = SQUEEZE_BYTES;

    /// Finalize `hasher` into the first [`Self::SQUEEZE_LEN`] bytes of `out`,
    /// UNREVERSED, and reset it. The first 32 bytes are the digest
    /// (`finalize_reset`); a wide configuration writes the 32 its final
    /// compression produces past them.
    fn finalize_squeeze_reset(hasher: &mut Self::Digest, out: &mut [u8; WIDE_SQUEEZE_BYTES]) {
        let digest: [u8; 32] = hasher.finalize_reset().into();
        out[..SQUEEZE_BYTES].copy_from_slice(&digest);
    }
}

/// Bytes of a squeeze that is the digest alone.
pub const SQUEEZE_BYTES: usize = 32;

/// Bytes of a squeeze that is a final compression's whole output.
pub const WIDE_SQUEEZE_BYTES: usize = 64;

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
    const CANDIDATES_PER_COORDINATE: Option<NonZeroUsize> = None;
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
    const CANDIDATES_PER_COORDINATE: Option<NonZeroUsize> = NonZeroUsize::new(1);
}

/// The BLAKE3 configuration — `Blake3Chain` over the same sponge, with rider
/// 1's constant-consumption sampling adopted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blake3TranscriptHash;

impl TranscriptHash for Blake3TranscriptHash {
    type Digest = Blake3Chain;

    const NAME: &'static str = "blake3-chain";

    /// ⚠ **`true`, and it is not a preference.** Every BLAKE3 proof this
    /// repository has produced came from a `sample()` that reversed
    /// unconditionally. The reversal became a per-configuration choice so the
    /// WHIR RPX transcript could drop it; dropping it anywhere else would move
    /// the challenges of a pipeline that is not changing, which is the one
    /// thing this merge must not do.
    const REVERSES_SQUEEZE: bool = true;

    /// Two candidates per coordinate.
    ///
    /// One would be free — it is what the current schedule costs in the modal
    /// case — but a single candidate that misses has nowhere to go, so the tail
    /// would sit at ≈ 2⁻³² per coordinate, i.e. once in a few hundred thousand
    /// proofs at production draw counts. That is not negligible enough to call
    /// the schedule fixed. Two puts the tail at ≈ 2⁻⁶⁴ per coordinate, for one
    /// extra candidate per coordinate — see the cost note in PA-PLAN §2.3.
    ///
    /// ⚠ Unlike the RPX configuration's `Some(1)`, this is probabilistic: a
    /// BLAKE3 squeeze is uniform bytes, so both candidates can miss. Two makes
    /// the tail negligible; it does not make it unreachable.
    const CANDIDATES_PER_COORDINATE: Option<NonZeroUsize> = NonZeroUsize::new(2);
}

/// ★ [`Blake3TranscriptHash`] at **7 rounds**: the same sponge, the same
/// squeeze reversal and the same two-candidate sampling, over [`Blake3Std`]
/// rather than the build's [`Blake3Chain`], so no feature moves its rounds.
///
/// The transcript of the WHIR base proof's BLAKE3 arm
/// (`multilinear::whir_hash::Blake3Whir`). [`Blake3TranscriptHash`] itself is
/// left as it is: it is the STARK pipeline's, at that build's round count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blake3StdTranscriptHash;

impl TranscriptHash for Blake3StdTranscriptHash {
    type Digest = Blake3Std;

    const NAME: &'static str = "blake3-std";

    /// [`Blake3TranscriptHash`]'s convention, so the two differ by the round
    /// count alone.
    const REVERSES_SQUEEZE: bool = true;

    /// [`Blake3TranscriptHash`]'s schedule, for the same reason.
    const CANDIDATES_PER_COORDINATE: Option<NonZeroUsize> = NonZeroUsize::new(2);
}

/// ★ [`Blake3StdTranscriptHash`] with **64-byte squeezes**: a squeeze hands out
/// the final compression's whole output — the digest, then the 32 bytes
/// BLAKE3's own extended output puts after it ([`Blake3Std::finalize_xof64`])
/// — so eight candidates a squeeze instead of four, at no extra compression.
///
/// Everything else is [`Blake3StdTranscriptHash`]'s: the hasher, the grind
/// (`state()` is the 32-byte digest), the re-absorb of the 32 digest bytes
/// that advances the chain, the reversal and the two-candidate schedule. The
/// second 32 bytes are reversed as the first are, so every candidate is one
/// little-endian `u64` of output words, as today's first four are, and a
/// squeeze's first four candidates are the narrow configuration's.
///
/// The extended bytes are never absorbed: the next squeeze's input is the
/// digest and what is appended after it, exactly as under
/// [`Blake3StdTranscriptHash`]. They are pseudorandom output of the same
/// compression whose first half the chain continues from, so a challenge
/// drawn from them is as bound to the transcript so far as one drawn from the
/// first half.
///
/// Opt-in: a proof under it is another proof format (its configuration is a
/// verifier constant, named in the statement that binds it), and is refused by
/// a verifier under [`Blake3StdTranscriptHash`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blake3StdSqueeze64TranscriptHash;

impl TranscriptHash for Blake3StdSqueeze64TranscriptHash {
    type Digest = Blake3Std;

    const NAME: &'static str = "blake3-std-squeeze64";

    /// [`Blake3StdTranscriptHash`]'s, for both halves.
    const REVERSES_SQUEEZE: bool = true;

    /// [`Blake3StdTranscriptHash`]'s schedule.
    const CANDIDATES_PER_COORDINATE: Option<NonZeroUsize> = NonZeroUsize::new(2);

    const SQUEEZE_LEN: usize = WIDE_SQUEEZE_BYTES;

    fn finalize_squeeze_reset(hasher: &mut Self::Digest, out: &mut [u8; WIDE_SQUEEZE_BYTES]) {
        *out = hasher.finalize_xof64();
        digest::Reset::reset(hasher);
    }
}
