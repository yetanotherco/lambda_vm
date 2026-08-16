use crypto::fiat_shamir::transcript_hash::{KeccakTranscriptHash, TranscriptHash};
#[cfg(not(feature = "cuda"))]
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
/// `Verifier` and Merkle alias in this crate resolves to.
///
/// ★ **It forks on `cuda`, and that fork is a proof-format fork.** A non-cuda
/// build commits with [`Blake3StarkHash`]; a cuda build commits with
/// [`KeccakStarkHash`]. Proofs do not cross: a GPU-produced proof is not
/// verifiable by a CPU build of the same commit, and vice versa.
///
/// This is deliberate and it is the *conservative* arm of PA-PLAN §6.2 row 6,
/// taken because `gpu_lde` has no BLAKE3 dispatch (see the note above
/// [`Blake3StarkHash`] for exactly what is and is not built). The alternative —
/// flipping the aliases under `cuda` too — would not produce BLAKE3 GPU proofs,
/// it would produce keccak trees *labelled* BLAKE3, which is the failure the
/// [`KeccakTreeBackend`] marker exists to make impossible.
///
/// **Retirement condition**, both halves required: `gpu_lde`'s tree entry points
/// dispatch to `math_cuda::blake3::*` instead of `math_cuda::merkle::*`, and the
/// BLAKE3 device parity tests pass on real hardware. Until then GPU proving is
/// keccak-only and this fork stays.
#[cfg(not(feature = "cuda"))]
pub type DefaultStarkHash = Blake3StarkHash;
/// The default commitment configuration. See the non-cuda definition for the
/// fork and its retirement condition.
#[cfg(feature = "cuda")]
pub type DefaultStarkHash = KeccakStarkHash;

// Spelled as the concrete backends rather than as `<DefaultStarkHash as
// StarkHash>::Batched<F>`: the associated types carry `F: 'static`, which the
// alias would propagate into every caller that is merely generic over a field.
// The assertion at the bottom of this file is what keeps the two spellings the
// same type.
#[cfg(not(feature = "cuda"))]
pub type BatchedMerkleTreeBackend<F> = BatchBlake3Backend<F>;
#[cfg(feature = "cuda")]
pub type BatchedMerkleTreeBackend<F> = BatchKeccak256Backend<F>;
pub type BatchedMerkleTree<F> = MerkleTree<BatchedMerkleTreeBackend<F>>;

// FRI layer uses fixed-size pairs for efficiency (avoids Vec allocation per pair)
#[cfg(not(feature = "cuda"))]
pub type FriLayerMerkleTreeBackend<F> = PairBlake3Backend<F>;
#[cfg(feature = "cuda")]
pub type FriLayerMerkleTreeBackend<F> = PairKeccak256Backend<F>;
pub type FriLayerMerkleTree<F> = MerkleTree<FriLayerMerkleTreeBackend<F>>;

/// A Merkle backend whose leaves and parents are Keccak-256, byte for byte.
///
/// A marker: no methods, nothing to implement wrongly. It exists because
/// `IsMerkleTreeBackend<Node = Commitment>` is too weak a bound wherever the
/// backend does not actually do the hashing. The GPU tree entry points in
/// `gpu_lde` are exactly that case — they take a backend parameter and then
/// launch the `math-cuda` keccak kernels unconditionally, so `B` is a label on
/// bytes `B` never touched. Any 32-byte-node backend satisfies the weak bound,
/// so a backend over some other hash would compile there and hand back keccak
/// trees wearing its name, with nothing failing.
///
/// Requiring this marker instead makes that a compile error at the call site,
/// and makes implementing it for a non-keccak backend a deliberate, reviewable
/// false statement rather than an omission nobody had to make.
pub trait KeccakTreeBackend: IsMerkleTreeBackend<Node = Commitment> {}

impl<F> KeccakTreeBackend for BatchKeccak256Backend<F> where
    Self: IsMerkleTreeBackend<Node = Commitment>
{
}
impl<F> KeccakTreeBackend for PairKeccak256Backend<F> where
    Self: IsMerkleTreeBackend<Node = Commitment>
{
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
    /// Under `cuda` this additionally has to be [`KeccakTreeBackend`]. That is
    /// not a preference: `gpu_lde`'s tree entries hash on the device with the
    /// keccak kernels and only *label* the result with this type, so a cuda
    /// build has no way to honour any other configuration. The bound says so at
    /// compile time instead of letting the label be wrong. It comes off when
    /// the device kernels stop being keccak-only.
    #[cfg(feature = "cuda")]
    type Batched<F>: IsStreamingLeafBackend<F, Node = Commitment, Data = Vec<FieldElement<F>>>
        + KeccakTreeBackend
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
    /// Under `cuda` this carries the same [`KeccakTreeBackend`] obligation
    /// [`Self::Batched`] does, and for the same reason: `gpu_lde`'s FRI commit
    /// drives the whole commit phase on device, hashing every layer tree with
    /// the keccak kernels and only *labelling* the result with this type. A cuda
    /// build cannot honour any other configuration for FRI layers either, so the
    /// bound says so at compile time rather than letting the label be wrong.
    #[cfg(feature = "cuda")]
    type Pair<F>: IsMerkleTreeBackend<Node = Commitment, Data = [FieldElement<F>; 2]>
        + KeccakTreeBackend
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

/// The keccak-256 configuration.
///
/// Since the flip it is the default only under `cuda` (see [`DefaultStarkHash`]);
/// on every other build it is reachable by naming it, and is what the
/// keccak-pinned LFM instruments and the GPU path commit under.
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
/// ★ **Everything, on a non-`cuda` build.** This is [`DefaultStarkHash`]: every
/// `Prover` and `Verifier` alias resolves here, [`COMMITMENT_HASH`] names it,
/// and the transcript and grinding follow through [`StarkHash::Transcript`]. The
/// RV64 guest reaches it through the same aliases and hashes with the
/// `blake3_compress_6round` precompile.
#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blake3StarkHash;

// Under `cuda` there is deliberately no BLAKE3 configuration to name.
//
// [`StarkHash::Batched`] additionally requires [`KeccakTreeBackend`] there,
// because `gpu_lde`'s tree entry points hash on the device with the keccak
// kernels and only *label* the result with the backend type — so a cuda build
// has no way to honour any other configuration, and the bound says so at compile
// time instead of letting the label be wrong. Implementing `KeccakTreeBackend`
// for a BLAKE3 backend to get past it would be precisely the deliberate false
// statement that marker exists to require, so the configuration does not exist
// under `cuda` at all.
//
// ★ What is missing is NOT the kernels. Track G landed the whole device side:
// `math-cuda/kernels/blake3.cu` carries all nine leaf/level kernels and a
// device `Blake3Chain` transcribed from this host construction, and
// `math_cuda::blake3` carries a wrapper for every `math_cuda::merkle` entry the
// tree path uses. The gap is one layer up — `gpu_lde` calls
// `math_cuda::merkle::*` unconditionally and has no BLAKE3 dispatch — and none
// of the device side has been exercised on real hardware yet.
//
// So this comes off when `gpu_lde` dispatches on the configuration AND the
// BLAKE3 parity tests have passed on a GPU. Both halves: a dispatch without the
// hardware run would ship an unvalidated hash, and a hardware run without the
// dispatch changes nothing about what the prover commits.
#[cfg(not(feature = "cuda"))]
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

    type Transcript = crypto::fiat_shamir::transcript_hash::Blake3TranscriptHash;

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
#[cfg(not(feature = "cuda"))]
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

/// The H3 marker's tie-in, and — since the flip — the guard on the `cuda` fork.
///
/// [`KeccakStarkHash`]'s own members satisfy the marker, which is what makes the
/// impls above true statements rather than decoration. The load-bearing half is
/// the second block: under `cuda`, [`DefaultStarkHash`] must be a configuration
/// the device kernels can actually honour, because `gpu_lde` hashes with the
/// keccak kernels and only *labels* the result with the alias. Point the aliases
/// at BLAKE3 under `cuda` and this is where you find out — before a GPU run
/// hands back keccak trees wearing a BLAKE3 name.
const _: fn() = || {
    fn assert_keccak_backend<B: KeccakTreeBackend>() {}
    fn assert_same<T>(_: core::marker::PhantomData<(T, T)>) {}

    assert_keccak_backend::<<KeccakStarkHash as StarkHash>::Batched<GoldilocksField>>();
    assert_keccak_backend::<<KeccakStarkHash as StarkHash>::Pair<GoldilocksField>>();

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

/// Under `cuda` the aliases must stay keccak — see [`DefaultStarkHash`].
#[cfg(feature = "cuda")]
const _: fn() = || {
    fn assert_keccak_backend<B: KeccakTreeBackend>() {}

    assert_keccak_backend::<BatchedMerkleTreeBackend<GoldilocksField>>();
    assert_keccak_backend::<FriLayerMerkleTreeBackend<GoldilocksField>>();
};

/// The flip, stated positively: the shipping (non-`cuda`) default commits BLAKE3.
///
/// The alias definitions make [`COMMITMENT_HASH`] follow [`DefaultStarkHash`]
/// automatically, so nothing above can *disagree* — what this catches is the
/// whole fork being reverted or re-pointed without the blessed artifacts moving
/// with it. Every pinned root in this workspace (`LFM_REGISTRY`,
/// `static_zero_page_commitment`, the preprocessed table commitments) was
/// generated under this arm.
#[cfg(not(feature = "cuda"))]
const _: () = assert!(matches!(COMMITMENT_HASH, CommitmentHash::Blake3));

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
/// It is scoped to the arm that actually commits BLAKE3: a `cuda` build commits
/// keccak, so its round count is free.
#[cfg(not(feature = "cuda"))]
const _: () = assert!(
    crypto::hash::blake3::BLAKE3_ROUNDS == crypto::hash::blake3::BLAKE3_SIX_ROUNDS,
    "the default commitment configuration is BLAKE3, so this build must enable \
     `crypto/blake3-6round` (via `lambda-vm-prover/blake3-6round`, or the \
     default feature set): every blessed root was generated at six rounds"
);
