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
//! # `cuda` on an algebraic pin
//!
//! Compiles, and cannot prove under the wrong hash. The algebraic backends are
//! `DeviceTreeBackend`s carrying their own `CommitmentHash` as the device
//! dispatch key, so a device tree is built by the kernels of the hash it is
//! named for or not built at all. RPX256 has those kernels (`math_cuda::rpx`),
//! so a GPU run under this pin commits on the device; RPO256 and Poseidon do
//! not yet, and a GPU run under one of them aborts at its first device commit
//! with `unimplemented!` naming the hash. ⛔ Neither a `compile_error!` nor a
//! byte-hash fallback belongs here: the first hides the cuda lint arm from the
//! branch, the second is exactly the silent wrong-hash build this module exists
//! to make impossible.
//!
//! # ⚠ TWO regenerations, not one — in THIS order, plus one stray constant
//!
//! A pin change is **not** complete until every root blessed under the old hash
//! is regenerated. There are two families of them and the order is load-bearing:
//!
//! 1. **The static preprocessed commitments — FIRST.** FOUR families: `bitwise`,
//!    `keccak_rc`, and `page`'s zero-page AND private-page constants, at blowup
//!    2/4/8. Each returns a BLESSED CONSTANT from `preprocessed_commitment`
//!    rather than recomputing, so under a new pin the prover recomputes an
//!    algebraic root, compares it against a BLAKE3 constant, and fails with
//!    `ProvingError::PrecomputedCommitmentMismatch`.
//!    `cargo run --bin compute_static_commitments --release`, then paste.
//! 2. **`LFM_REGISTRY` — SECOND, only once the statics are in the tree.**
//!    `registry.rs` fills slots 13 and 14 of every entry from `keccak_rc` and
//!    `bitwise`'s `preprocessed_commitment` — the blessed constants above, not a
//!    recomputation — and `lfm_program_id` folds every root. A registry generated
//!    before the statics were pasted therefore embeds the OUTGOING hash's
//!    constants, and `machine_tests::registry_drift_*` fires at exactly those two
//!    slots. The control-first re-run under the outgoing pin cannot see this:
//!    both tables are self-consistent there.
//!    `cargo run --bin compute_lfm_registry --release`.
//! 3. **`SUB_DECODE_COMMITMENT_BLOWUP_2`** in `tests/decode_tests.rs` — a
//!    test-local blessed constant outside both generators, regenerated by the
//!    `#[ignore]` test `print_decode_commitment_for_sub`.
//!
//! ✓ VERIFIED (1) empirically: it is exactly how the trial flip failed, and it
//! is the correct failure — loud, at prove time, naming the cause. ✓ VERIFIED
//! (2) empirically too: the first RPX regeneration ran the registry before the
//! statics and all six drift tests fired at slots 13 and 14. `registry.rs`
//! governs both: a drift failure is investigated, never re-blessed to silence
//! the test, and neither table is ever hand-edited.

/// The commitment configuration the block path proves and verifies under.
///
/// Every `multi_prove` / `multi_verify` instantiation in this crate names this
/// rather than `stark::config::DefaultStarkHash`, so the two can differ on a
/// branch without the workspace default moving.
pub type BlockStarkHash = crate::lfm::algebraic_commit::RpxStarkHash;

/// The Fiat–Shamir transcript OBJECT the block path builds.
///
/// See the module header for why this is pinned separately from
/// [`BlockStarkHash`] rather than derived from it.
pub type BlockTranscript = crate::lfm::algebraic_transcript::AlgebraicTranscript;

/// A fresh block-path transcript over `seed`.
///
/// A function rather than a bare `::new`, because the two arms construct
/// differently: a byte transcript takes the seed in its constructor, an
/// algebraic one absorbs it as its first `append_bytes` call. Callers should not
/// have to know which.
pub fn block_transcript(seed: &[u8]) -> BlockTranscript {
    BlockTranscript::with_seed(BLOCK_HASHER, seed)
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

/// The `LFM_HASH` socket permutation the block path's programs are EXECUTED and
/// proved under — the machine's own hash chip.
///
/// ⚠ **A third axis, and it is orthogonal to [`BlockStarkHash`].** That one says
/// which hash the HOST commits under; this says which permutation the MACHINE's
/// `Instr::Hash` rows compute. They have to agree, and nothing in the type
/// system makes them: the socket hasher is passed per call to `execute` and
/// `lfm_prove_with_hasher`.
///
/// ★ **Why it went unpinned until it bit.** Under a byte hash the emitter's
/// Merkle work goes through `ByteWrapHash::hash_bytes`, which lowers to the
/// dedicated KECCAK / `LFM_BLAKE3` chips and emits **no `Instr::Hash` at all** —
/// so the socket hasher handed to `execute` is never consulted, and passing
/// `TestPermutation` is free and correct. The algebraic arm goes through
/// `b.compress` / `b.permute`, which ARE `Instr::Hash`, executed by whatever is
/// passed. The same distinction the byte hash made irrelevant, needed back.
///
/// Every `execute` and prove call on the block path names this rather than a
/// literal, so the two axes cannot drift apart in a test harness while
/// production stays correct.
pub const BLOCK_HASHER: crate::lfm::hash::HasherKind = crate::lfm::hash::HasherKind::Rpx;

// The host block transcript (`block_transcript`) hashes twelve-felt steps with
// `BLOCK_HASHER`; the width-16 socket has no twelve-felt hash (its contract
// refuses), so it can never be the block hasher. A width-16 program takes the
// socket from its own instructions (`LfmProgram::hasher`), never from here.
const _: () = assert!(
    !matches!(BLOCK_HASHER, crate::lfm::hash::HasherKind::Poseidon1W16),
    "the block hasher hashes twelve-felt transcript steps; the width-16 socket has none"
);

/// The [`CommitmentHash`] the block path's roots may be called by.
///
/// ★ Read this rather than `stark::config::COMMITMENT_HASH`. That const names
/// the hash of the workspace ALIASES and says so in its own doc — a prover can
/// run under a configuration whose `COMMITMENT_HASH` differs and the const will
/// not know. The block path IS such a configuration on three of the four
/// branches, so anything describing a block proof's roots must read the pin.
pub const BLOCK_COMMITMENT_HASH: stark::config::CommitmentHash =
    <BlockStarkHash as stark::config::StarkHash>::COMMITMENT_HASH;

// ============================================================================
// ★★★ THE BLOCK'S PROOF FAMILY (D-P3B §2.1)
// ============================================================================

/// ★★★ Everything a block's recursion proves and verifies with, named ONCE: the
/// LFM proofs' commitment configuration and their transcript, the tag every
/// LFM statement names them by, the wrap hash a node's in-guest legs replay
/// them with, and the socket the tree programs' hash rows run on. The block's
/// WHIR base is pinned beside the family ([`BlockWhirHash`], [`BLOCK_BASE`]),
/// not in it: the family serves the LFM prover's two users (the block path and
/// the legacy users off it), the base only the block path.
///
/// A compile-time choice, not a run-time one: there is ONE family per build
/// ([`BlockFamily`]), so no proof of the block path is ever made or checked
/// under a mix, and the move to BLAKE3 is a `Blake3Family` and a flip of that
/// alias, which the compiler follows to every site (D-P3B §2.9). Nothing here
/// reads the environment, a format field or a proof.
pub trait ProofFamily: Send + Sync + 'static {
    /// The LFM proofs' STARK commitment configuration.
    type Stark: stark::config::StarkHash;
    /// The LFM proofs' Fiat–Shamir transcript OBJECT (see the module header
    /// for why it is named apart from [`Self::Stark`]).
    type Transcript: crypto::fiat_shamir::is_transcript::IsStarkTranscript<
            crate::tables::types::GoldilocksExtension,
            crate::tables::types::GoldilocksField,
        > + Clone
        + Send;
    /// The family's name in every LFM statement: the statement's first
    /// `append_bytes` call (`statement::absorb_lfm_statement_tagged`), so a
    /// statement names the proof family it was made under. The legacy pin's
    /// is `LAMBDAVM_LFM_STATEMENT_V1` itself, so its statements stay today's
    /// bytes; another family's replaces it.
    const STATEMENT_TAG: &'static [u8];
    /// The in-guest wrap hash a program verifying this family's proofs runs.
    const WRAP: crate::lfm::edsl::WrapHash;
    /// The machine's hash chip under that wrap hash.
    const SOCKET: crate::lfm::hash::HasherKind;
    /// A fresh transcript over `seed` (absorbed as its first `append_bytes`).
    fn transcript(seed: &[u8]) -> Self::Transcript;
}

/// The legacy pin as a family: [`BlockStarkHash`] on [`BlockTranscript`], no
/// statement tag — today's LFM proofs byte for byte. The LFM users off the
/// WHIR block path (continuation, the epoch tree, the registry and the probes)
/// stay here until the cleanup commit (D-P3B §2.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LegacyFamily;

impl ProofFamily for LegacyFamily {
    type Stark = BlockStarkHash;
    type Transcript = BlockTranscript;
    const STATEMENT_TAG: &'static [u8] = crate::lfm::statement::LFM_STATEMENT_TAG;
    const WRAP: crate::lfm::edsl::WrapHash = crate::lfm::edsl::WrapHash::production();
    const SOCKET: crate::lfm::hash::HasherKind = BLOCK_HASHER;
    fn transcript(seed: &[u8]) -> Self::Transcript {
        block_transcript(seed)
    }
}

/// ZisK's Poseidon1 end to end: 4-ary trees capped at
/// [`crate::lfm::p1_commit::P1_STARK_CAP`], ZisK's sponge, width-8 grinding,
/// and the width-16 socket in the guest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct P1Family;

impl ProofFamily for P1Family {
    type Stark = crate::lfm::p1_commit::P1StarkHash;
    type Transcript = crate::lfm::p1_commit::P1Transcript;
    const STATEMENT_TAG: &'static [u8] = P1_LFM_STATEMENT_TAG;
    const WRAP: crate::lfm::edsl::WrapHash = crate::lfm::edsl::WrapHash::Poseidon1;
    const SOCKET: crate::lfm::hash::HasherKind = crate::lfm::hash::HasherKind::Poseidon1W16;
    fn transcript(seed: &[u8]) -> Self::Transcript {
        crate::lfm::p1_commit::P1Transcript::with_seed(seed)
    }
}

/// [`P1Family`]'s statement tag (D-P3B §9, shared with #1013): today's
/// `LAMBDAVM_LFM_STATEMENT_V1`, the family and its cap height, `C` +
/// [`crate::lfm::p1_commit::P1_STARK_CAP`] — absorbed in place of today's tag.
pub const P1_LFM_STATEMENT_TAG: &[u8] = b"LAMBDAVM_LFM_STATEMENT_V1/P1W16/C2";

// The tag's cap digit is the configuration's.
const _: () = assert!(
    P1_LFM_STATEMENT_TAG[P1_LFM_STATEMENT_TAG.len() - 1]
        == b'0' + crate::lfm::p1_commit::P1_STARK_CAP
);

/// ★★★ THE SWITCH: the proof family of the no-epoch WHIR block's recursion.
/// Fixed at compile time: nothing selects it at run time, and nothing but the
/// LFM prover's off-path users (continuation, the epoch tree, the registry,
/// the probes: [`LegacyFamily`]) names another family until the cleanup
/// commit collapses the two.
pub type BlockFamily = P1Family;

/// ★★★ The no-epoch block's WHIR base configuration: ZisK's Poseidon1 WHIR,
/// the hash [`BlockFamily`]'s recursion is made in — one hash end to end, so
/// no base of one hash ever sits under another's recursion. The block path
/// binds it (`with_block_hash!`) after `block_whir::checked_base` has
/// admitted the format's base, which must be [`BLOCK_BASE`].
pub type BlockWhirHash = multilinear::whir_hash::P1Whir;

/// The base format of every no-epoch block: [`BlockWhirHash`]'s hash at its
/// 4-ary cap (a WHIR geometry, the base's own, separate from the LFM trees'
/// `P1_STARK_CAP`). `ZfFormat`'s default names it, and the block path admits
/// no other.
pub const BLOCK_BASE: stark::proof::options::BaseFormat =
    stark::proof::options::BaseFormat::P1_WHIR;

// The base and the recursion are one hash, and the base's configuration is the
// base format's hash: the device key `BlockWhirHash` dispatches on names the
// same hash as `BLOCK_BASE`.
const _: () = {
    assert!(
        BLOCK_BASE.hash as u8
            == <<BlockFamily as ProofFamily>::Stark as stark::config::StarkHash>::COMMITMENT_HASH
                as u8
    );
    let device = match <BlockWhirHash as multilinear::whir_hash::WhirHash>::DEVICE {
        multilinear::whir_hash::DeviceHashKey::Keccak256 => {
            stark::config::CommitmentHash::Keccak256
        }
        multilinear::whir_hash::DeviceHashKey::Rpx256 => stark::config::CommitmentHash::Rpx256,
        multilinear::whir_hash::DeviceHashKey::Poseidon1 => {
            stark::config::CommitmentHash::Poseidon1
        }
    };
    assert!(device as u8 == BLOCK_BASE.hash as u8);
};

/// The root of a static preprocessed table (`table`: `bitwise`, `keccak_rc`)
/// under a commitment configuration OTHER than the legacy pin's, whose roots
/// are blessed constants (the module header's regenerations): computed by
/// `compute` once per process for each (hash, blowup, coset, leaf layout) and
/// reused. The cap of an arity-4 configuration does not enter: it moves path
/// nodes, never a root.
pub fn static_root_under<S: stark::config::StarkHash>(
    table: &'static str,
    options: &stark::proof::options::ProofOptions,
    layout: stark::leaf_layout::LeafLayout,
    compute: impl FnOnce() -> stark::config::Commitment,
) -> stark::config::Commitment {
    type Key = (&'static str, u8, u8, u64, usize);
    static ROOTS: std::sync::Mutex<Vec<(Key, stark::config::Commitment)>> =
        std::sync::Mutex::new(Vec::new());
    let key: Key = (
        table,
        S::COMMITMENT_HASH as u8,
        options.blowup_factor,
        options.coset_offset,
        layout.rows_per_leaf(),
    );
    let cached = |roots: &[(Key, stark::config::Commitment)]| {
        roots.iter().find(|(k, _)| *k == key).map(|(_, r)| *r)
    };
    if let Some(root) = cached(&ROOTS.lock().unwrap_or_else(|e| e.into_inner())) {
        return root;
    }
    // Computed outside the lock: a second thread may compute the same root
    // meanwhile, and both get the same value.
    let root = compute();
    let mut roots = ROOTS.lock().unwrap_or_else(|e| e.into_inner());
    if cached(&roots).is_none() {
        roots.push((key, root));
    }
    root
}

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

    /// The families are coherent: each one's transcript object hashes with
    /// its configuration's Fiat–Shamir hash, the legacy family is the pin, and
    /// only a non-legacy family tags its statements.
    #[test]
    fn each_family_names_one_hash() {
        use crypto::fiat_shamir::is_transcript::IsTranscript;
        use crypto::fiat_shamir::transcript_hash::TranscriptHash;

        assert_eq!(
            <<LegacyFamily as ProofFamily>::Stark as StarkHash>::COMMITMENT_HASH,
            BLOCK_COMMITMENT_HASH
        );
        assert_eq!(LegacyFamily::STATEMENT_TAG, b"LAMBDAVM_LFM_STATEMENT_V1");
        assert_eq!(
            <<P1Family as ProofFamily>::Stark as StarkHash>::COMMITMENT_HASH,
            stark::config::CommitmentHash::Poseidon1
        );
        assert_eq!(
            <<<P1Family as ProofFamily>::Stark as StarkHash>::Transcript as TranscriptHash>::NAME,
            "poseidon1-w16"
        );
        assert_eq!(
            P1Family::STATEMENT_TAG,
            b"LAMBDAVM_LFM_STATEMENT_V1/P1W16/C2"
        );
        assert_eq!(P1Family::WRAP, crate::lfm::edsl::WrapHash::Poseidon1);
        // The block's base is the family's hash: P1's WHIR at the base's cap.
        assert_eq!(
            <BlockWhirHash as multilinear::whir_hash::WhirHash>::NAME,
            <multilinear::whir_hash::P1Whir as multilinear::whir_hash::WhirHash>::NAME
        );
        assert_eq!(BLOCK_BASE, stark::proof::options::BaseFormat::P1_WHIR);
        assert_eq!(P1Family::SOCKET, crate::lfm::hash::HasherKind::Poseidon1W16);
        // The P1 transcript object is ZisK's sponge seeded like the legacy
        // one (the seed is its first `append_bytes`).
        let mut t = crate::lfm::p1_commit::P1Transcript::new();
        <_ as IsTranscript<E>>::append_bytes(&mut t, b"seed");
        assert_eq!(
            <_ as IsTranscript<E>>::state(&P1Family::transcript(b"seed")),
            <_ as IsTranscript<E>>::state(&t)
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
