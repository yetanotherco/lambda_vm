//! The hash a Fiat-Shamir transcript runs on.
//!
//! [`DefaultTranscript`](super::default_transcript::DefaultTranscript) is a thin
//! `digest::Digest` wrapper, so swapping the hash is a type substitution. What
//! this trait adds beyond naming a digest is that the name travels with the
//! proof: a transcript is part of what a replaying verifier has to reproduce,
//! so the configuration has to be something a call site can state rather than
//! something a default decides.
//!
//! ⚠ **What is deliberately NOT here: a challenge-consumption schedule.** The
//! sibling of this trait on the per-table branch carries a
//! `CANDIDATES_PER_COORDINATE` constant, because a straight-line machine
//! replaying the transcript needs a draw count that does not depend on the
//! data. Nothing in this workspace replays a WHIR transcript yet, so the
//! constant would be a knob no test here could observe. There is a second
//! reason, recorded because it is easy to get backwards: the canonicality
//! argument that makes a one-candidate schedule safe for an algebraic sponge
//! does not survive
//! [`DefaultTranscript::sample`](super::default_transcript::DefaultTranscript::sample)'s
//! byte reversal, so adopting the constant here would have been a check that
//! cannot fail.

use digest::{Digest, FixedOutputReset, OutputSizeUser, typenum::U32};

use crate::hash::platform_keccak::PlatformKeccak256;

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
}

/// The keccak-256 configuration — what every `DefaultTranscript` is unless a
/// caller says otherwise, and byte-for-byte the transcript this system has
/// always produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeccakTranscriptHash;

impl TranscriptHash for KeccakTranscriptHash {
    type Digest = PlatformKeccak256;

    const NAME: &'static str = "keccak256";
}
