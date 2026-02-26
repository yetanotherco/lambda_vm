use crypto::merkle_tree::{
    backends::types::{
        Keccak256Backend, QuaternaryBatchKeccak256Backend, QuaternaryPairKeccak256Backend,
    },
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

// Quaternary (arity-4) Merkle trees: ~3x fewer hash calls for tree construction
pub type BatchedMerkleTreeBackend<F> = QuaternaryBatchKeccak256Backend<F>;
pub type BatchedMerkleTree<F> = MerkleTree<BatchedMerkleTreeBackend<F>>;

// FRI layer uses fixed-size pairs with quaternary arity
pub type FriLayerMerkleTreeBackend<F> = QuaternaryPairKeccak256Backend<F>;
pub type FriLayerMerkleTree<F> = MerkleTree<FriLayerMerkleTreeBackend<F>>;
