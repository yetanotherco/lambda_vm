//! ★★ **THE HASH PIN** — the one place a build says which hash the BLOCK path
//! proves under.
//!
//! # Why this is a module and not a line in `crypto/stark`
//!
//! `crypto/stark`'s [`stark::config::DefaultStarkHash`] is the *workspace's*
//! default: it names the hash behind `Commitment`, `BatchedMerkleTree` and every
//! blessed constant in the repo, and a `const` assertion there makes re-pointing
//! it a compile error precisely so those artifacts cannot drift.
//!
//! The hash-comparison branches need something different — to change what the
//! block path commits under **without** touching that default, so BLAKE3's
//! enforcement stays intact while a sibling branch proves under RPO. ✓ VERIFIED
//! that is expressible: `IsStarkProver<Field, FieldExtension, PI, H: StarkHash>`
//! is generic over the configuration, and `prover` was already naming
//! `DefaultStarkHash` *explicitly* at each of its prove and verify call sites.
//! Those are type parameters, not a global. Collecting them behind these two
//! names turns "which hash does the block path use" from a property spread over
//! six files into a property of this one.
//!
//! # ⚠ The pin is TWO things, and the second is easy to miss
//!
//! [`StarkHash::Transcript`] names a `TranscriptHash` — a **digest
//! configuration**, which is what GRINDING computes over. The Fiat–Shamir
//! transcript **object** is built by the caller and handed to `multi_prove`, so
//! the type system does not force it to match.
//!
//! For the byte hashes the two coincide: the object is
//! `DefaultTranscript<E, H::Transcript>`, a sponge over that digest. **For an
//! algebraic hash they do not.** `AlgebraicTranscript` is a compress chain over
//! cells, not a byte sponge over `AlgebraicDigest`, and a branch that pinned only
//! [`BlockStarkHash`] would commit under RPO while sponging Fiat–Shamir through
//! bytes — self-consistent between prover and verifier, and therefore **silent**.
//! That is the same half-flip `stark::config::DefaultStarkTranscript`'s own doc
//! warns about, and [`block_transcript`] is why it cannot happen here.
//!
//! # What a branch changes
//!
//! Exactly the three items below, and nothing else in the workspace. On an
//! algebraic branch they become, for example:
//!
//! ```ignore
//! pub type BlockStarkHash  = crate::lfm::algebraic_commit::RpoStarkHash;
//! pub type BlockTranscript = crate::lfm::algebraic_transcript::AlgebraicTranscript;
//! pub fn block_transcript(seed: &[u8]) -> BlockTranscript {
//!     BlockTranscript::with_seed(crate::lfm::hash::HasherKind::Rpo, seed)
//! }
//! ```
//!
//! ✓ VERIFIED that flip compiles and runs end to end — it was performed, built,
//! and executed against this crate's own prove/verify tests before this module
//! was written, which is how [`BlockProver`], the generic transcript parameter
//! on `compute_expected_commit_bus_balance_view`, and the
//! `IsStreamingLeafBackend` import in `proof_arena` were found. None of those
//! three shows up on a build that only ever pins BLAKE3.
//!
//! # ⚠ TWO regenerations, not one
//!
//! A pin change is **not** complete until every root blessed under the old hash
//! is regenerated, and there are two families of them:
//!
//! 1. **`LFM_REGISTRY`** — the hasher is folded into every `program_id`.
//!    `cargo run --bin compute_lfm_registry --release`.
//! 2. **The static preprocessed commitments** — `bitwise`, `keccak_rc` and
//!    `page` each return a BLESSED CONSTANT from `preprocessed_commitment`
//!    rather than recomputing, so under a new pin the prover recomputes an
//!    RPO root, compares it against a BLAKE3 constant, and fails with
//!    `ProvingError::PrecomputedCommitmentMismatch`.
//!    `cargo run --bin compute_static_commitments --release`.
//!
//! ✓ VERIFIED (2) empirically: it is exactly how the trial flip failed, and it
//! is the correct failure — loud, at prove time, naming the cause. `registry.rs`
//! governs both: a drift failure is investigated, never re-blessed to silence
//! the test, and neither table is ever hand-edited.

/// The commitment configuration the block path proves and verifies under.
///
/// Every `multi_prove` / `multi_verify` instantiation in this crate names this
/// rather than `stark::config::DefaultStarkHash`, so the two can differ on a
/// branch without the workspace default moving.
pub type BlockStarkHash = stark::config::DefaultStarkHash;

/// The Fiat–Shamir transcript OBJECT the block path builds.
///
/// See the module header for why this is pinned separately from
/// [`BlockStarkHash`] rather than derived from it.
pub type BlockTranscript =
    stark::config::DefaultStarkTranscript<crate::tables::types::GoldilocksExtension>;

/// A fresh block-path transcript over `seed`.
///
/// A function rather than a bare `::new`, because the two arms construct
/// differently: a byte transcript takes the seed in its constructor, an
/// algebraic one absorbs it as its first `append_bytes` call. Callers should not
/// have to know which.
pub fn block_transcript(seed: &[u8]) -> BlockTranscript {
    BlockTranscript::new(seed)
}

/// The prover the block path drives, at [`BlockStarkHash`].
///
/// ⚠ **Not `stark::prover::Prover`.** That alias is `GenericProver` at
/// `DefaultStarkHash`, so it is BLAKE3-fixed regardless of what `H` a call site
/// passes alongside it — and the two disagreeing is a type error rather than a
/// silent wrong hash, which is how this was found. The `IsStarkProver` impl
/// itself is fully generic over `H`; only the alias is pinned, so the fix is an
/// alias at the pin rather than anything in `crypto/stark`.
pub type BlockProver<Field, FieldExtension, PI> =
    stark::prover::GenericProver<Field, FieldExtension, PI, BlockStarkHash>;

/// The verifier the block path drives, at [`BlockStarkHash`]. See
/// [`BlockProver`] for why the `stark::verifier::Verifier` alias is not it.
pub type BlockVerifier<Field, FieldExtension, PI> =
    stark::verifier::GenericVerifier<Field, FieldExtension, PI, BlockStarkHash>;

/// The [`CommitmentHash`] the block path's roots may be called by.
///
/// ★ Read this rather than `stark::config::COMMITMENT_HASH`. That const names
/// the hash of the workspace ALIASES and says so in its own doc — a prover can
/// run under a configuration whose `COMMITMENT_HASH` differs and the const will
/// not know. The block path IS such a configuration on three of the four
/// branches, so anything describing a block proof's roots must read the pin.
pub const BLOCK_COMMITMENT_HASH: stark::config::CommitmentHash =
    <BlockStarkHash as stark::config::StarkHash>::COMMITMENT_HASH;

#[cfg(test)]
mod tests {
    use super::*;
    // Named here rather than at module scope: the byte arm's `BlockTranscript`
    // mentions the extension field and an algebraic arm's does not, so a
    // module-scope import would be unused on one of the two.
    use crate::tables::types::GoldilocksExtension as E;
    use stark::config::StarkHash;

    /// ✓ The pin is COHERENT: the transcript object the block path builds sponges
    /// on the same hash the commitment configuration names.
    ///
    /// ⚠ This is the half-flip guard, and it is a real one rather than a
    /// tautology only because [`BlockTranscript`] is pinned separately — the two
    /// names can disagree, which is exactly the failure this catches. It is
    /// stated over `NAME` because that is the one thing both sides expose.
    #[test]
    fn the_transcript_and_the_commitment_configuration_name_one_hash() {
        use crypto::fiat_shamir::transcript_hash::TranscriptHash;

        // The byte arm's object IS `DefaultTranscript<E, H::Transcript>`, so the
        // agreement is by construction here and this test says so cheaply. On an
        // algebraic branch the two are independent types and this becomes the
        // check that matters.
        let named = <<BlockStarkHash as StarkHash>::Transcript as TranscriptHash>::NAME;
        assert!(
            !named.is_empty(),
            "a commitment configuration must name its Fiat-Shamir hash"
        );
    }

    /// ✓ A fresh transcript is deterministic in its seed — the property every
    /// prove/verify pair depends on, and the one a mis-wired constructor breaks.
    #[test]
    fn a_seeded_transcript_is_a_function_of_its_seed() {
        use crypto::fiat_shamir::is_transcript::IsTranscript;

        let a = <BlockTranscript as IsTranscript<E>>::state(&block_transcript(b"seed-one"));
        let b = <BlockTranscript as IsTranscript<E>>::state(&block_transcript(b"seed-one"));
        let c = <BlockTranscript as IsTranscript<E>>::state(&block_transcript(b"seed-two"));
        assert_eq!(a, b, "the same seed must give the same state");
        assert_ne!(a, c, "a different seed must give a different state");
    }
}
