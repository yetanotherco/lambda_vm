use crypto::merkle_tree::{
    backends::types::{BatchKeccak256Backend, Keccak256Backend, PairKeccak256Backend, QuadKeccak256Backend},
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

pub type BatchedMerkleTreeBackend<F> = BatchKeccak256Backend<F>;
pub type BatchedMerkleTree<F> = MerkleTree<BatchedMerkleTreeBackend<F>>;

// FRI layer uses fixed-size pairs for efficiency (avoids Vec allocation per pair)
pub type FriLayerMerkleTreeBackend<F> = PairKeccak256Backend<F>;
pub type FriLayerMerkleTree<F> = MerkleTree<FriLayerMerkleTreeBackend<F>>;

// Arity-4 FRI layer: each leaf commits to 4 consecutive evaluations
pub type FriLayerQuadMerkleTreeBackend<F> = QuadKeccak256Backend<F>;
pub type FriLayerQuadMerkleTree<F> = MerkleTree<FriLayerQuadMerkleTreeBackend<F>>;
