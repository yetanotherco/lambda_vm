#[cfg(feature = "quaternary-merkle")]
use crypto::merkle_tree::backends::types::BatchKeccak256Backend;
#[cfg(not(feature = "quaternary-merkle"))]
use crypto::merkle_tree::backends::types::BinaryBatchKeccak256Backend;
use crypto::merkle_tree::merkle::MerkleTree;

// Merkle Trees configuration

pub const COMMITMENT_SIZE: usize = 32;
pub type Commitment = [u8; COMMITMENT_SIZE];

// Trace commitment trees: arity matches feature flag (4-ary or binary)
#[cfg(feature = "quaternary-merkle")]
pub type BatchedMerkleTreeBackend<F> = BatchKeccak256Backend<F>;
#[cfg(not(feature = "quaternary-merkle"))]
pub type BatchedMerkleTreeBackend<F> = BinaryBatchKeccak256Backend<F>;
pub type BatchedMerkleTree<F> = MerkleTree<BatchedMerkleTreeBackend<F>>;

// FRI layer trees: Vec leaves to support variable arity (2, 4, 8, ...)
#[cfg(feature = "quaternary-merkle")]
pub type FriLayerMerkleTreeBackend<F> = BatchKeccak256Backend<F>;
#[cfg(not(feature = "quaternary-merkle"))]
pub type FriLayerMerkleTreeBackend<F> = BinaryBatchKeccak256Backend<F>;
pub type FriLayerMerkleTree<F> = MerkleTree<FriLayerMerkleTreeBackend<F>>;
