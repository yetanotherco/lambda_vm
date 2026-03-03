#[cfg(feature = "quaternary-merkle")]
use crypto::merkle_tree::backends::types::{BatchKeccak256Backend, Keccak256Backend};
#[cfg(not(feature = "quaternary-merkle"))]
use crypto::merkle_tree::backends::types::{BinaryBatchKeccak256Backend, BinaryKeccak256Backend};
use crypto::merkle_tree::merkle::MerkleTree;

// Merkle Trees configuration

// Security of both hashes should match

// If using hashes with 256-bit security, commitment size should be 32
// If using hashes with 512-bit security, commitment size should be 64
// TODO: Commitment type should be obtained from MerkleTrees
pub const COMMITMENT_SIZE: usize = 32;
pub type Commitment = [u8; COMMITMENT_SIZE];

// When quaternary-merkle is enabled, use 4-ary Merkle trees (Lambda VM / Goldilocks)
#[cfg(feature = "quaternary-merkle")]
pub type BatchedMerkleTreeBackend<F> = BatchKeccak256Backend<F>;
#[cfg(feature = "quaternary-merkle")]
pub type BatchedMerkleTree<F> = MerkleTree<BatchedMerkleTreeBackend<F>>;
#[cfg(feature = "quaternary-merkle")]
pub type FriMerkleTreeBackend<F> = Keccak256Backend<F>;
#[cfg(feature = "quaternary-merkle")]
pub type FriMerkleTree<F> = MerkleTree<FriMerkleTreeBackend<F>>;
#[cfg(feature = "quaternary-merkle")]
pub type FriLayerMerkleTreeBackend<F> = BatchKeccak256Backend<F>;
#[cfg(feature = "quaternary-merkle")]
pub type FriLayerMerkleTree<F> = MerkleTree<FriLayerMerkleTreeBackend<F>>;

// Without the feature, use binary Merkle trees (Stone-compatible / Stark252)
#[cfg(not(feature = "quaternary-merkle"))]
pub type BatchedMerkleTreeBackend<F> = BinaryBatchKeccak256Backend<F>;
#[cfg(not(feature = "quaternary-merkle"))]
pub type BatchedMerkleTree<F> = MerkleTree<BatchedMerkleTreeBackend<F>>;
#[cfg(not(feature = "quaternary-merkle"))]
pub type FriMerkleTreeBackend<F> = BinaryKeccak256Backend<F>;
#[cfg(not(feature = "quaternary-merkle"))]
pub type FriMerkleTree<F> = MerkleTree<FriMerkleTreeBackend<F>>;
#[cfg(not(feature = "quaternary-merkle"))]
pub type FriLayerMerkleTreeBackend<F> = BinaryBatchKeccak256Backend<F>;
#[cfg(not(feature = "quaternary-merkle"))]
pub type FriLayerMerkleTree<F> = MerkleTree<FriLayerMerkleTreeBackend<F>>;
