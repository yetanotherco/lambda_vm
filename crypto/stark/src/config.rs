use crypto::merkle_tree::{
    backends::types::{BatchKeccak256Backend, Keccak256Backend, PairKeccak256Backend},
    merkle::MerkleTree,
    traits::IsMerkleTreeBackend,
};
use math::field::goldilocks::GoldilocksField;

// Merkle Trees configuration

// Security of both hashes should match

pub type FriMerkleTreeBackend<F> = Keccak256Backend<F>;
pub type FriMerkleTree<F> = MerkleTree<FriMerkleTreeBackend<F>>;

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

impl<F> KeccakTreeBackend for Keccak256Backend<F> where Self: IsMerkleTreeBackend<Node = Commitment> {}
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

/// The hash behind [`Commitment`], [`BatchedMerkleTree`], [`FriMerkleTree`] and
/// [`FriLayerMerkleTree`]. Pinned to the aliases by the assertion below.
pub const COMMITMENT_HASH: CommitmentHash = CommitmentHash::Keccak256;

/// Ties [`COMMITMENT_HASH`] to the aliases it describes. Repointing any of the
/// three at a backend not marked [`KeccakTreeBackend`] fails here — in the file
/// that makes the claim — instead of silently downstream where the claim is
/// consumed.
const _: fn() = || {
    fn assert_keccak_backend<B: KeccakTreeBackend>() {}
    assert_keccak_backend::<FriMerkleTreeBackend<GoldilocksField>>();
    assert_keccak_backend::<BatchedMerkleTreeBackend<GoldilocksField>>();
    assert_keccak_backend::<FriLayerMerkleTreeBackend<GoldilocksField>>();
};
