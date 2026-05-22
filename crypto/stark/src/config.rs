use crypto::merkle_tree::{
    backends::types::{BatchKeccak256Backend, Keccak256Backend, QuadKeccak256Backend},
    merkle::MerkleTree,
};

// Merkle Trees configuration

// Security of both hashes should match

pub type FriMerkleTreeBackend<F> = Keccak256Backend<F>;
pub type FriMerkleTree<F> = MerkleTree<FriMerkleTreeBackend<F>>;

// If using hashes with 256-bit security, commitment size should be 32
// If using hashes with 512-bit security, commitment size should be 64
// TODO: Commitment type should be obtained from MerkleTrees
pub const COMMITMENT_SIZE: usize = 32;
pub type Commitment = [u8; COMMITMENT_SIZE];

/// Height of the Merkle commitment cap. The commitment is the `2^MERKLE_CAP_HEIGHT`
/// nodes at this tree depth instead of a single root, so every opening path is
/// `MERKLE_CAP_HEIGHT` hashes shorter. Clamped to the tree depth for small trees.
pub const MERKLE_CAP_HEIGHT: usize = 4;

/// A Merkle commitment represented as a cap: the `2^MERKLE_CAP_HEIGHT` nodes at
/// tree depth `MERKLE_CAP_HEIGHT`, ordered left to right.
pub type MerkleCap = Vec<Commitment>;

pub type BatchedMerkleTreeBackend<F> = BatchKeccak256Backend<F>;
pub type BatchedMerkleTree<F> = MerkleTree<BatchedMerkleTreeBackend<F>>;

// FRI layer uses fixed-size quad leaves: one leaf per arity-4 fold orbit, so a
// single Keccak covers the four conjugate evaluations a query opens together.
pub type FriLayerMerkleTreeBackend<F> = QuadKeccak256Backend<F>;
pub type FriLayerMerkleTree<F> = MerkleTree<FriLayerMerkleTreeBackend<F>>;
