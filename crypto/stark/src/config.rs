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

pub type BatchedMerkleTreeBackend<F> = BatchKeccak256Backend<F>;
pub type BatchedMerkleTree<F> = MerkleTree<BatchedMerkleTreeBackend<F>>;

// FRI layer uses fixed-size pairs for efficiency (avoids Vec allocation per pair)
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
/// One variant, deliberately. It is the machine-readable form of what the three
/// aliases above already say in types, and it exists so that code reasoning
/// about *which hash is inside a root* can match on it exhaustively rather than
/// assert it in prose — see `build_artifacts_with_hasher` in
/// `prover/src/lfm/registry.rs`, whose artifacts name a hash. Adding a second
/// variant here breaks every such match, which is the point: it is the list of
/// places that have to be revisited before this crate can commit under two
/// hashes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitmentHash {
    /// Keccak-256 at both the leaf and the parent layer.
    Keccak256,
}

/// The hash behind [`Commitment`], [`BatchedMerkleTree`] and
/// [`FriLayerMerkleTree`]. Pinned to the aliases by the assertion below.
pub const COMMITMENT_HASH: CommitmentHash = CommitmentHash::Keccak256;

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
    type Pair<F>: IsMerkleTreeBackend<Node = Commitment, Data = [FieldElement<F>; 2]> + 'static
    where
        F: IsField + 'static,
        FieldElement<F>: AsBytes + Sync + Send;

    /// What both hash with. The name a proof's roots may be called by.
    const COMMITMENT_HASH: CommitmentHash;
}

/// The keccak-256 configuration — the only one, and the one every `Prover` and
/// `Verifier` alias resolves to.
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

    const COMMITMENT_HASH: CommitmentHash = CommitmentHash::Keccak256;
}

/// Ties the aliases, [`COMMITMENT_HASH`] and [`KeccakStarkHash`]'s members
/// to each other, so they cannot drift apart silently.
///
/// The `KeccakTreeBackend` assertions are the H3 marker's tie-in: it is not a
/// parallel ladder to [`StarkHash`] but a consequence of this instance, since
/// the GPU kernels are keccak-only regardless of which configuration the host
/// prover runs. When a second configuration exists, `gpu_lde` still demands
/// keccak and this is where you find out.
const _: fn() = || {
    fn assert_keccak_backend<B: KeccakTreeBackend>() {}
    fn assert_same<T>(_: core::marker::PhantomData<(T, T)>) {}

    assert_keccak_backend::<BatchedMerkleTreeBackend<GoldilocksField>>();
    assert_keccak_backend::<FriLayerMerkleTreeBackend<GoldilocksField>>();

    // The aliases ARE the keccak instance's members, not a second opinion.
    assert_same::<BatchedMerkleTreeBackend<GoldilocksField>>(
        core::marker::PhantomData::<(
            BatchedMerkleTreeBackend<GoldilocksField>,
            <KeccakStarkHash as StarkHash>::Batched<GoldilocksField>,
        )>,
    );
    assert_same::<FriLayerMerkleTreeBackend<GoldilocksField>>(
        core::marker::PhantomData::<(
            FriLayerMerkleTreeBackend<GoldilocksField>,
            <KeccakStarkHash as StarkHash>::Pair<GoldilocksField>,
        )>,
    );
};

const _: () = assert!(matches!(
    <KeccakStarkHash as StarkHash>::COMMITMENT_HASH,
    COMMITMENT_HASH
));
