//! ★ The hash the WHIR path runs on — the one name a proof's Merkle trees, its
//! Fiat-Shamir sponge and its proof-of-work all answer to.
//!
//! # Why one trait and not three parameters
//!
//! A WHIR proof has three hash consumers: the Merkle backend that builds its
//! roots, the transcript sponge that draws its challenges, and the grind that
//! gates each redrawable challenge. Parameterising them separately would make
//! the **half-flip** spellable — one hash's trees under another hash's sponge.
//! That configuration is self-consistent between prover and verifier, so it
//! verifies, so nothing fails; it is silent by construction, and the only thing
//! wrong with it is that no single name describes the proof. Here there is one
//! name to write, so there is nothing to assert against: the bad state is
//! unreachable rather than checked.
//!
//! # What a hash may NOT change
//!
//! [`Commitment`] stays `[u8; 32]` for every implementation. A keccak digest is
//! 32 bytes and an algebraic digest is four canonical Goldilocks felts, which is
//! also 32 bytes — so every proof type on this path (`ChainProof`,
//! `StackedProof`, `CosetOpening`, `Proof<Commitment>`, and `MultiProof` above
//! them) keeps its layout, its rkyv derives and its serialized length. **A hash
//! swap is not a proof-format change**, and `stacked_eval`'s
//! `the_two_hashes_serialize_to_the_same_length` is what holds that to it.
//!
//! # The device, and what is NOT here yet
//!
//! Under `cuda` the leaf and parent hashing happens in kernels, and the host
//! backend is only the label on the tree they built — so a second hash needs a
//! device dispatch key on this trait, and the `math-cuda` entry points need to
//! read it. That arrives with the kernels themselves (H2). It is deliberately
//! absent here: a dispatch key with one variant that nothing branches on is a
//! knob no test can observe, and the guard and the thing it guards belong in
//! one commit.

use crypto::fiat_shamir::transcript_hash::{KeccakTranscriptHash, TranscriptHash};
use crypto::merkle_tree::backends::types::BatchKeccak256Backend;
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use math::field::{element::FieldElement, traits::IsField};
use math::traits::AsBytes;

use crate::whir_commit::Commitment;

/// The digest a configuration grinds over: its transcript's hash, because the
/// grinding seed is `transcript.state()`.
pub type GrindingDigest<H> = <<H as WhirHash>::Transcript as TranscriptHash>::Digest;

/// ★ One WHIR hash configuration.
///
/// Implementing it is the whole of adding a hash to this path: a unit struct, a
/// Merkle backend and a Fiat-Shamir configuration. Nothing in `whir_commit`,
/// `whir_round`, `whir_chain` or `stacked_eval` knows which one it has.
pub trait WhirHash: Copy + Clone + Default + Send + Sync + 'static {
    /// The name a proof's roots may be called by — for banners, KATs and
    /// diagnostics. Never absorbed into the transcript: the sponge IS this
    /// hash, so two configurations' challenge streams diverge at the first
    /// squeeze and a tag would separate nothing that is not already separate.
    const NAME: &'static str;

    /// The Fiat-Shamir configuration this commitment hash is paired with — the
    /// sponge's hash, and the one the proof-of-work grind computes over.
    type Transcript: TranscriptHash;

    /// The Merkle backend: one leaf per fold block, 32-byte nodes.
    type Backend<F>: IsMerkleTreeBackend<Node = Commitment, Data = Vec<FieldElement<F>>>
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;
}

/// The keccak-256 configuration — the default everywhere on this path, and
/// byte-for-byte what PR #988 produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeccakWhir;

impl WhirHash for KeccakWhir {
    const NAME: &'static str = "keccak256";

    type Transcript = KeccakTranscriptHash;

    type Backend<F>
        = BatchKeccak256Backend<F>
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;
}
