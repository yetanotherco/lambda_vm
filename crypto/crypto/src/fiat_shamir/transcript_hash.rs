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
    type Digest: Digest + FixedOutputReset + OutputSizeUser<OutputSize = U32> + Clone + 'static;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
