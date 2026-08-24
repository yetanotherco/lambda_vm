use crypto::fiat_shamir::transcript_hash::{
    Blake3TranscriptHash, KeccakTranscriptHash, TranscriptHash,
};
use crypto::merkle_tree::backends::types::{BatchBlake3Backend, PairBlake3Backend};
use crypto::merkle_tree::{
    backends::types::{BatchKeccak256Backend, PairKeccak256Backend},
    merkle::MerkleTree,
    traits::{IsMerkleTreeBackend, IsStreamingLeafBackend},
};
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsField;
use math::traits::AsBytes;

// Merkle Trees configuration

// Security of both hashes should match

// If using hashes with 256-bit security, commitment size should be 32
// If using hashes with 512-bit security, commitment size should be 64
// TODO: Commitment type should be obtained from MerkleTrees
pub const COMMITMENT_SIZE: usize = 32;
pub type Commitment = [u8; COMMITMENT_SIZE];

/// The default commitment configuration — what every unparameterized `Prover`,
/// `Verifier` and Merkle alias in this crate resolves to — on every build,
/// `cuda` included.
///
/// A cuda build honours it because `gpu_lde`'s tree entry points dispatch on
/// the configuration: every backend they accept is a [`DeviceTreeBackend`],
/// whose `COMMITMENT_HASH` constant selects the device kernel family
/// (keccak or the Track G BLAKE3 kernels) at each leaf, level and FRI-layer
/// launch. The transcript and grinding follow the same configuration through
/// [`StarkHash::Transcript`], and the pairing asserts at the bottom of this
/// file keep a configuration's commitment and Fiat-Shamir hashes from ever
/// being edited apart.
pub type DefaultStarkHash = Blake3StarkHash;

// Spelled as the concrete backends rather than as `<DefaultStarkHash as
// StarkHash>::Batched<F>`: the associated types carry `F: 'static`, which the
// alias would propagate into every caller that is merely generic over a field.
// The assertion at the bottom of this file is what keeps the two spellings the
// same type.
pub type BatchedMerkleTreeBackend<F> = BatchBlake3Backend<F>;
pub type BatchedMerkleTree<F> = MerkleTree<BatchedMerkleTreeBackend<F>>;

/// The keccak tree families, named rather than reached through the aliases.
///
/// The CUDA keccak kernels compute keccak whatever the host default is, so
/// their parity tests need a CPU reference that says keccak rather than one
/// that says "whatever is default". While keccak WAS the default the aliases
/// served both purposes; the flip separated them, and a parity test left on an
/// alias would report a hash change as a kernel bug.
pub type KeccakBatchedMerkleTreeBackend<F> = BatchKeccak256Backend<F>;
pub type KeccakFriLayerMerkleTreeBackend<F> = PairKeccak256Backend<F>;
pub type KeccakFriLayerMerkleTree<F> = MerkleTree<KeccakFriLayerMerkleTreeBackend<F>>;

// FRI layer uses fixed-size pairs for efficiency (avoids Vec allocation per pair)
pub type FriLayerMerkleTreeBackend<F> = PairBlake3Backend<F>;
pub type FriLayerMerkleTree<F> = MerkleTree<FriLayerMerkleTreeBackend<F>>;

/// A Merkle backend the GPU tree entry points can honour, naming its hash.
///
/// `IsMerkleTreeBackend<Node = Commitment>` is too weak a bound wherever the
/// backend does not actually do the hashing. The GPU tree entry points in
/// `gpu_lde` are exactly that case — they hash on the device and use `B` only
/// as the label on the host `MerkleTree` the roots are wrapped in, so any
/// 32-byte-node backend would compile there and hand back trees wearing a
/// name whose hash the kernels never computed.
///
/// This trait closes that hole from both ends: `COMMITMENT_HASH` is the
/// dispatch key `gpu_lde` hands to `math-cuda` (selecting the keccak or the
/// BLAKE3 kernel family at every leaf, level and FRI-layer launch), and
/// implementing the trait is the reviewable claim that device kernels
/// producing exactly this backend's hash exist. A backend over some other
/// hash has no true constant to supply, so writing the impl is a deliberate
/// false statement rather than an omission nobody had to make.
pub trait DeviceTreeBackend: IsMerkleTreeBackend<Node = Commitment> {
    /// The hash the device kernels must compute for trees labelled `Self`.
    const COMMITMENT_HASH: CommitmentHash;
}

impl<F> DeviceTreeBackend for BatchKeccak256Backend<F>
where
    Self: IsMerkleTreeBackend<Node = Commitment>,
{
    const COMMITMENT_HASH: CommitmentHash = CommitmentHash::Keccak256;
}
impl<F> DeviceTreeBackend for PairKeccak256Backend<F>
where
    Self: IsMerkleTreeBackend<Node = Commitment>,
{
    const COMMITMENT_HASH: CommitmentHash = CommitmentHash::Keccak256;
}
impl<F> DeviceTreeBackend for BatchBlake3Backend<F>
where
    Self: IsMerkleTreeBackend<Node = Commitment>,
{
    const COMMITMENT_HASH: CommitmentHash = CommitmentHash::Blake3;
}
impl<F> DeviceTreeBackend for PairBlake3Backend<F>
where
    Self: IsMerkleTreeBackend<Node = Commitment>,
{
    const COMMITMENT_HASH: CommitmentHash = CommitmentHash::Blake3;
}

/// The hash every commitment this crate produces is built with.
///
/// It is the machine-readable form of what the aliases above say in types, and
/// it exists so that code reasoning about *which hash is inside a root* can
/// match on it exhaustively rather than assert it in prose — see
/// `build_artifacts_with_hasher` in `prover/src/lfm/registry.rs`, whose
/// artifacts name a hash. Every such match is a place that has to be revisited
/// before this crate commits under a second hash; adding [`Self::Blake3`] broke
/// them, which is what that mechanism is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitmentHash {
    /// Keccak-256 at both the leaf and the parent layer.
    Keccak256,
    /// `Blake3Chain` at both the leaf and the parent layer — the single-chunk
    /// BLAKE3 chain of PA-PLAN §1.7, at the round count `crypto`'s
    /// `blake3-6round` feature selects.
    ///
    /// The round count is deliberately **not** a second variant. It is a
    /// crate-global compile-time constant precisely so one build cannot produce
    /// two hashes, so within a build there is nothing here to distinguish: a
    /// proof's roots are named by this variant plus the build's feature set,
    /// exactly as the `LFM_BLAKE3` chip's round count is.
    Blake3,
}

/// The hash behind [`Commitment`], [`BatchedMerkleTree`] and
/// [`FriLayerMerkleTree`]. Pinned to the aliases by the assertion below.
///
/// ⚠ **This describes the DEFAULT configuration — the aliases — and nothing
/// else.** A prover can run under a configuration whose
/// [`StarkHash::COMMITMENT_HASH`] differs from this const and this const will
/// not know: it is a global, the configuration is per-type. Code that names the
/// hash inside a *particular* proof's roots must read `H::COMMITMENT_HASH` at
/// the call site; only code that names the hash of the aliases may read this.
/// `prover::lfm::registry` reads this one because its commit helpers are
/// hard-wired to the aliases — when they become generic over `H`, that read
/// moves with them (PA-PLAN §4.2).
///
/// It is defined as [`DefaultStarkHash`]'s own constant rather than restated, so
/// the `cuda` fork cannot be taken here and forgotten there.
pub const COMMITMENT_HASH: CommitmentHash = <DefaultStarkHash as StarkHash>::COMMITMENT_HASH;

/// One STARK commitment configuration: the Merkle backend families the
/// prover and verifier build trees with, named together so they cannot be
/// mixed, plus the [`CommitmentHash`] they all are.
///
/// The three are separate families because they hash different leaf shapes, not
/// The two are separate families because they hash different leaf shapes, not
/// because they are different hashes: [`Self::Batched`] takes a whole row group,
/// [`Self::Pair`] a fixed FRI-layer pair. An implementation is expected to build
/// both on one hash — that is what [`Self::COMMITMENT_HASH`] asserts, and what
/// makes a proof's roots describable by a single name.
///
/// Every member is generic over the field because the prover commits over both
/// the base field (main trace) and the extension (aux, composition, FRI) within
/// one proof, so the configuration cannot be pinned to one field.
///
/// `Node` is deliberately **not** an associated type: it is [`Commitment`] for
/// every implementation. Keeping 32 bytes on the wire is what lets a
/// configuration change leave `StarkProof`'s fields and their rkyv derives
/// byte-identical — no format bump, no disturbance to the in-place verify path.
///
/// # Invariant: the two families must agree on a two-element leaf
///
/// `<Batched<F>>::hash_data(&vec![a, b])` must equal `<Pair<F>>::hash_data(&[a, b])`.
///
/// This is load-bearing, not decorative. The prover builds FRI-layer trees with
/// [`Self::Pair`] (`fri/mod.rs`) and the verifier authenticates those same
/// openings with [`Self::Batched`] (`verify_fri_layer_openings`, which builds a
/// two-element `Vec`). Under keccak the two coincide — both stream the same
/// element bytes into one digest — which is why the split went unremarked while
/// there was only one configuration. A configuration whose families encode a
/// pair differently would reject every honest proof at its first FRI query.
///
/// An implementation that cannot honour this must make the prover and verifier
/// agree on one family instead of implementing this trait and hoping.
pub trait StarkHash: Send + Sync + 'static {
    /// The batched leaf backend: one leaf per row group, streamed.
    ///
    /// Under `cuda` this additionally has to be [`DeviceTreeBackend`]:
    /// `gpu_lde`'s tree entries hash on the device with the kernel family the
    /// backend's `COMMITMENT_HASH` names, and only *label* the result with
    /// this type — the bound is what guarantees the label and the kernels
    /// agree, at compile time.
    #[cfg(feature = "cuda")]
    type Batched<F>: IsStreamingLeafBackend<F, Node = Commitment, Data = Vec<FieldElement<F>>>
        + DeviceTreeBackend
        + 'static
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;

    /// The batched leaf backend: one leaf per row group, streamed.
    #[cfg(not(feature = "cuda"))]
    type Batched<F>: IsStreamingLeafBackend<F, Node = Commitment, Data = Vec<FieldElement<F>>>
        + 'static
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;

    /// The FRI-layer backend: one leaf per fixed pair, no `Vec` per leaf.
    ///
    /// Under `cuda` this carries the same [`DeviceTreeBackend`] obligation
    /// [`Self::Batched`] does, and for the same reason: `gpu_lde`'s FRI commit
    /// drives the whole commit phase on device, hashing every layer tree with
    /// the kernel family the backend's `COMMITMENT_HASH` names.
    #[cfg(feature = "cuda")]
    type Pair<F>: IsMerkleTreeBackend<Node = Commitment, Data = [FieldElement<F>; 2]>
        + DeviceTreeBackend
        + 'static
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;

    /// The FRI-layer backend: one leaf per fixed pair, no `Vec` per leaf.
    #[cfg(not(feature = "cuda"))]
    type Pair<F>: IsMerkleTreeBackend<Node = Commitment, Data = [FieldElement<F>; 2]> + 'static
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;

    /// The Fiat-Shamir configuration this commitment configuration is paired
    /// with — the hash the transcript sponges on, and the one grinding's
    /// proof-of-work computes over.
    ///
    /// Naming it here is what keeps a proof describable by one configuration.
    /// The transcript object is still built by the caller and handed to
    /// `multi_prove` / `multi_verify`, so this does not *force* the caller's
    /// transcript to match; what it forces is that everything the prover and
    /// verifier derive internally from the configuration — grinding — follows
    /// this hash instead of a hard-wired one.
    type Transcript: TranscriptHash;

    /// What both Merkle families hash with. The name a proof's roots may be
    /// called by.
    const COMMITMENT_HASH: CommitmentHash;
}

/// The digest a configuration grinds over: its transcript's hash, because the
/// grinding seed is `transcript.state()`.
pub type GrindingDigest<H> = <<H as StarkHash>::Transcript as TranscriptHash>::Digest;

/// ★ The Fiat-Shamir transcript of the default configuration — the one the
/// production prove and verify entry points must build.
///
/// `DefaultTranscript<F>`'s own type default is [`KeccakTranscriptHash`], and
/// that default is NOT the configuration's. The two coincided while keccak was
/// the only configuration, and a caller writing `DefaultTranscript::<E>::new(..)`
/// after the flip gets a keccak sponge over BLAKE3 commitments — self-consistent
/// between prover and verifier, and therefore silent, but a half-flip: the
/// Fiat-Shamir hash would not have moved with the commitment hash.
///
/// `multi_prove` / `multi_verify` take `impl IsStarkTranscript`, so the type
/// system cannot force this; naming the alias is what makes the production path
/// follow [`DefaultStarkHash`] instead of a `derive`'s default.
pub type DefaultStarkTranscript<F> = crypto::fiat_shamir::default_transcript::DefaultTranscript<
    F,
    <DefaultStarkHash as StarkHash>::Transcript,
>;

/// The keccak-256 configuration.
///
/// Reachable by naming it: the keccak-pinned LFM instruments and the GPU
/// keccak A/B arms commit under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeccakStarkHash;

impl StarkHash for KeccakStarkHash {
    type Batched<F>
        = BatchKeccak256Backend<F>
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;

    type Pair<F>
        = PairKeccak256Backend<F>
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;

    type Transcript = KeccakTranscriptHash;

    const COMMITMENT_HASH: CommitmentHash = CommitmentHash::Keccak256;
}

/// The BLAKE3 configuration — `Blake3Chain` at both the leaf and the parent
/// layer, over the *same* two generic backends the keccak instance uses.
///
/// Sharing those backends is what makes the two-element invariant above hold by
/// construction: `Batched::hash_data(&vec![a, b])` and `Pair::hash_data(&[a, b])`
/// serialize the same 16 bytes and hand them to the same digest, so there are
/// not two encodings to be shown equal.
/// `blake3_batched_and_pair_agree_on_a_two_element_leaf` pins it anyway, because
/// "holds by construction" is a claim about today's code and the invariant has
/// to survive tomorrow's.
///
/// Separately, and for the parent layer rather than the leaf: a parent's message
/// is the two 32-byte children, and at 64 bytes `Blake3Chain` is a single BLAKE3
/// compression in the framing the device kernels implement
/// (`crypto::hash::blake3::chain`, PA-PLAN §1.7 P2).
///
/// # What selects it
///
/// ★ **Everything, on every build.** This is [`DefaultStarkHash`]: every
/// `Prover` and `Verifier` alias resolves here, [`COMMITMENT_HASH`] names it,
/// and the transcript and grinding follow through [`StarkHash::Transcript`].
/// The RV64 guest reaches it through the same aliases and hashes with the
/// `blake3_compress_6round` precompile; a cuda build reaches it through
/// `gpu_lde`'s [`DeviceTreeBackend`] dispatch onto the Track G BLAKE3
/// kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blake3StarkHash;

impl StarkHash for Blake3StarkHash {
    type Batched<F>
        = BatchBlake3Backend<F>
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;

    type Pair<F>
        = PairBlake3Backend<F>
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;

    type Transcript = Blake3TranscriptHash;

    const COMMITMENT_HASH: CommitmentHash = CommitmentHash::Blake3;
}

/// [`Blake3StarkHash`]'s members are the BLAKE3 backends, not a mix.
///
/// The two families exist because they hash different leaf *shapes*, not because
/// they are different hashes — that is the whole content of
/// [`StarkHash::COMMITMENT_HASH`] being one constant. Asserting it here means a
/// configuration assembled from one hash's batched backend and another's pair
/// backend fails to compile, rather than producing proofs whose roots no single
/// name describes.
const _: fn() = || {
    fn assert_same<T>(_: core::marker::PhantomData<(T, T)>) {}

    assert_same::<BatchBlake3Backend<GoldilocksField>>(
        core::marker::PhantomData::<(
            BatchBlake3Backend<GoldilocksField>,
            <Blake3StarkHash as StarkHash>::Batched<GoldilocksField>,
        )>,
    );
    assert_same::<PairBlake3Backend<GoldilocksField>>(
        core::marker::PhantomData::<(
            PairBlake3Backend<GoldilocksField>,
            <Blake3StarkHash as StarkHash>::Pair<GoldilocksField>,
        )>,
    );
};

/// The device marker's tie-in: each configuration's members carry the
/// constant that names the configuration's own hash, which is what makes the
/// [`DeviceTreeBackend`] impls above true statements rather than decoration —
/// `gpu_lde` dispatches device kernels on that constant, so a mismatch here
/// would be a GPU run hashing under a name the roots do not deserve.
const _: fn() = || {
    fn assert_device_hash<B: DeviceTreeBackend>(expect: CommitmentHash) {
        assert!(matches!(
            (B::COMMITMENT_HASH, expect),
            (CommitmentHash::Keccak256, CommitmentHash::Keccak256)
                | (CommitmentHash::Blake3, CommitmentHash::Blake3)
        ));
    }
    fn assert_same<T>(_: core::marker::PhantomData<(T, T)>) {}

    assert_device_hash::<<KeccakStarkHash as StarkHash>::Batched<GoldilocksField>>(
        CommitmentHash::Keccak256,
    );
    assert_device_hash::<<KeccakStarkHash as StarkHash>::Pair<GoldilocksField>>(
        CommitmentHash::Keccak256,
    );
    assert_device_hash::<<Blake3StarkHash as StarkHash>::Batched<GoldilocksField>>(
        CommitmentHash::Blake3,
    );
    assert_device_hash::<<Blake3StarkHash as StarkHash>::Pair<GoldilocksField>>(
        CommitmentHash::Blake3,
    );

    // The aliases ARE [`DefaultStarkHash`]'s members, not a second opinion.
    // They are spelled concretely for the lifetime reason noted at their
    // definition, so this is the tie that makes the two spellings one type —
    // and it is what fails if only one side of the `cuda` fork is edited.
    assert_same::<BatchedMerkleTreeBackend<GoldilocksField>>(
        core::marker::PhantomData::<(
            BatchedMerkleTreeBackend<GoldilocksField>,
            <DefaultStarkHash as StarkHash>::Batched<GoldilocksField>,
        )>,
    );
    assert_same::<FriLayerMerkleTreeBackend<GoldilocksField>>(
        core::marker::PhantomData::<(
            FriLayerMerkleTreeBackend<GoldilocksField>,
            <DefaultStarkHash as StarkHash>::Pair<GoldilocksField>,
        )>,
    );
};

/// Stated positively: the default commits BLAKE3, on every build.
///
/// The alias definitions make [`COMMITMENT_HASH`] follow [`DefaultStarkHash`]
/// automatically, so nothing above can *disagree* — what this catches is the
/// default being re-pointed without the blessed artifacts moving with it.
/// Every pinned root in this workspace (`LFM_REGISTRY`,
/// `static_zero_page_commitment`, the preprocessed table commitments) was
/// generated under this configuration.
const _: () = assert!(matches!(COMMITMENT_HASH, CommitmentHash::Blake3));

/// ★ The transcript follows the commitment hash — per configuration, by
/// assertion.
///
/// `multi_prove` / `multi_verify` take `impl IsStarkTranscript`, so the type
/// system cannot force a caller's transcript to match its commitment
/// configuration; what CAN be forced is that each named configuration pairs
/// its commitment hash with its own family's Fiat-Shamir hash, so following
/// the configuration (as `DefaultStarkTranscript` does) can never produce the
/// half-flip — one family's sponge over the other family's roots, which is
/// self-consistent between prover and verifier and therefore silent.
const _: fn() = || {
    fn assert_same<T>(_: core::marker::PhantomData<(T, T)>) {}

    assert_same::<Blake3TranscriptHash>(
        core::marker::PhantomData::<(
            Blake3TranscriptHash,
            <Blake3StarkHash as StarkHash>::Transcript,
        )>,
    );
    assert_same::<KeccakTranscriptHash>(
        core::marker::PhantomData::<(
            KeccakTranscriptHash,
            <KeccakStarkHash as StarkHash>::Transcript,
        )>,
    );
};
const _: () = assert!(matches!(
    <Blake3StarkHash as StarkHash>::COMMITMENT_HASH,
    CommitmentHash::Blake3
));
const _: () = assert!(matches!(
    <KeccakStarkHash as StarkHash>::COMMITMENT_HASH,
    CommitmentHash::Keccak256
));

/// ★ The round-count lockstep, and the reason the default build must carry
/// `blake3-6round`.
///
/// `BLAKE3_ROUNDS` is a crate-global compile-time constant, so one build cannot
/// produce two hashes — but two *builds* can, and every blessed constant in this
/// workspace was generated at six rounds. Without this assertion a build that
/// merely lost the feature would commit seven-round roots and fail later, at
/// drift-test time, with a mismatch that names no cause. Here it is a compile
/// error that names one.
///
const _: () = assert!(
    crypto::hash::blake3::BLAKE3_ROUNDS == crypto::hash::blake3::BLAKE3_SIX_ROUNDS,
    "the default commitment configuration is BLAKE3, so this build must enable \
     `crypto/blake3-6round` (via `lambda-vm-prover/blake3-6round`, or the \
     default feature set): every blessed root was generated at six rounds"
);
