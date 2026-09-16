//! The hash a Fiat-Shamir transcript runs on, and the sampling schedule that
//! travels with it.
//!
//! `DefaultTranscript` is a thin `digest::Digest` wrapper, so swapping the hash
//! is a type substitution. What this trait adds beyond the digest is the
//! *challenge-consumption schedule*, because the two are decided together: a
//! proof's transcript is named by one configuration, and the schedule is part of
//! what a replaying verifier — host or in-machine — has to reproduce.

use core::num::NonZeroUsize;
use digest::{Digest, FixedOutputReset, OutputSizeUser, typenum::U32};

use crate::hash::blake3::chain::Blake3Chain;
use crate::hash::platform_keccak::PlatformKeccak256;
use crate::hash::rpx::Rpx256Digest;

/// One Fiat-Shamir configuration: the digest the sponge runs on, plus how many
/// candidates a field-coordinate draw consumes.
pub trait TranscriptHash: 'static {
    /// The sponge's hash.
    ///
    /// `Clone` because the transcript is snapshotted (the GPU FRI path restores
    /// it) and because `state()` finalizes a clone. `FixedOutputReset` because
    /// the squeeze is `finalize_reset`. The 32-byte output size is pinned rather
    /// than left associated: `state()` returns `[u8; 32]`, and that is what
    /// seeds grinding, so a configuration with a different digest width would
    /// not be a drop-in anywhere it is consumed. `'static` because the GPU
    /// grinding dispatch keys the device search on the concrete digest by
    /// `TypeId`, like the merkle backends' keccak fast paths.
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

    /// How many 64-bit candidates one *base coordinate* draws.
    ///
    /// `None` — draw until one lands in the field's canonical range. The
    /// expected cost is one candidate (rejection probability ≈ 2⁻³²), but the
    /// count is data-dependent.
    ///
    /// `Some(n)` — always draw exactly `n` and take the first in range. This is
    /// the property a straight-line machine needs: the LFM transcript replay
    /// encodes one consumption schedule, and a transcript whose draw count
    /// varies is unprovable against it (`SOUNDNESS.md` §6.3, and
    /// `others/lfm-migration-riders.md` rider 1).
    ///
    /// ⚠ `Some(n)` is constant-consumption *up to a tail*: if all `n` candidates
    /// miss — probability ≈ 2⁻³²ⁿ per coordinate — the draw continues rather
    /// than failing. Failing would make challenge sampling fallible on the
    /// verifier's replay path, which the no-panic policy forbids and which would
    /// make `sample_field_element` return an `Option` everywhere. Continuing
    /// keeps the distribution *exactly* uniform (no modular-reduction bias,
    /// which at 2⁻³² per draw would dominate the proof system's soundness
    /// error), and leaves a fixed schedule that holds except on that tail.
    const CANDIDATES_PER_COORDINATE: Option<NonZeroUsize>;

    /// Name for KATs and diagnostics.
    const NAME: &'static str;
}

/// The keccak-256 configuration — what every `DefaultTranscript` is unless a
/// caller says otherwise, and byte-for-byte the transcript this system has
/// always produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeccakTranscriptHash;

impl TranscriptHash for KeccakTranscriptHash {
    type Digest = PlatformKeccak256;

    /// Deliberately `None`. Rider 1 is adopted for the BLAKE3 configuration
    /// only: changing the keccak schedule would move every existing proof's
    /// challenges, which is the one thing P-a's staging keeps still until the
    /// flip.
    const CANDIDATES_PER_COORDINATE: Option<NonZeroUsize> = None;

    const NAME: &'static str = "keccak256";
}

/// The BLAKE3 configuration — `Blake3Chain` over the same sponge, with rider
/// 1's constant-consumption sampling adopted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blake3TranscriptHash;

impl TranscriptHash for Blake3TranscriptHash {
    type Digest = Blake3Chain;

    /// Two candidates per coordinate.
    ///
    /// One would be free — it is what the current schedule costs in the modal
    /// case — but a single candidate that misses has nowhere to go, so the tail
    /// would sit at ≈ 2⁻³² per coordinate, i.e. once in a few hundred thousand
    /// proofs at production draw counts. That is not negligible enough to call
    /// the schedule fixed. Two puts the tail at ≈ 2⁻⁶⁴ per coordinate, for one
    /// extra candidate per coordinate — see the cost note in PA-PLAN §2.3.
    const CANDIDATES_PER_COORDINATE: Option<NonZeroUsize> = NonZeroUsize::new(2);

    const NAME: &'static str = "blake3-chain";
}

/// The RPX256 configuration — the algebraic sponge, for a transcript a
/// field-native verifier has to replay.
///
/// ⚠ **Why this has no fixed schedule, when the per-table branch's RPX
/// transcript sets `CANDIDATES_PER_COORDINATE` to `Some(1)`.** That branch's
/// argument is that a squeeze yields four felts which are canonical by
/// construction, so a single `u64` candidate can never miss. The argument does
/// not survive this transcript's plumbing: [`DefaultTranscript::sample`]
/// REVERSES all 32 bytes of the squeeze before handing them out
/// (`default_transcript.rs`, `result_hash.reverse()`), so the first eight bytes
/// a sampler reads are the LAST felt's canonical bytes in reverse order — a
/// number with no canonicality property at all. Adopting `Some(1)` here would
/// have been a constant whose stated justification is false and whose failure
/// mode (a rejected candidate with nowhere to go) no test in this workspace
/// could reach. The fixed schedule is an LFM-replay requirement; it belongs
/// with the emitter that needs it, alongside whatever makes the canonicality
/// argument true again.
///
/// [`DefaultTranscript::sample`]: super::default_transcript::DefaultTranscript
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RpxTranscriptHash;

impl TranscriptHash for RpxTranscriptHash {
    type Digest = Rpx256Digest;

    /// `None` — the unbounded rejection schedule, which is what this
    /// transcript already did before the trait gained the constant. Choosing it
    /// explicitly is what keeps the WHIR path's bytes where they are; the byte
    /// gate in `prover/src/tests/whir_byte_gate.rs` is what says so.
    const CANDIDATES_PER_COORDINATE: Option<NonZeroUsize> = None;

    const NAME: &'static str = "rpx256";
}
