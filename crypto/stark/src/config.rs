#[cfg(feature = "quaternary-merkle")]
use crypto::merkle_tree::backends::types::{BatchKeccak256Backend, Keccak256Backend};
#[cfg(not(feature = "quaternary-merkle"))]
use crypto::merkle_tree::backends::types::{BinaryBatchKeccak256Backend, BinaryKeccak256Backend};
use crypto::merkle_tree::backends::types::PairKeccak256Backend;
use crypto::merkle_tree::merkle::MerkleTree;

// Merkle Trees configuration

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

// Without the feature, use binary Merkle trees (Stone-compatible / Stark252)
#[cfg(not(feature = "quaternary-merkle"))]
pub type BatchedMerkleTreeBackend<F> = BinaryBatchKeccak256Backend<F>;
#[cfg(not(feature = "quaternary-merkle"))]
pub type BatchedMerkleTree<F> = MerkleTree<BatchedMerkleTreeBackend<F>>;
#[cfg(not(feature = "quaternary-merkle"))]
pub type FriMerkleTreeBackend<F> = BinaryKeccak256Backend<F>;
#[cfg(not(feature = "quaternary-merkle"))]
pub type FriMerkleTree<F> = MerkleTree<FriMerkleTreeBackend<F>>;

// FRI layer trees are always binary (pair-based), since FRI folding produces pairs
pub type FriLayerMerkleTreeBackend<F> = PairKeccak256Backend<F>;
pub type FriLayerMerkleTree<F> = MerkleTree<FriLayerMerkleTreeBackend<F>>;

// FRI layer verification backend: always binary (arity=2), vector-based data type
// This matches the hashing of PairKeccak256Backend but accepts Vec<FieldElement<F>> as Data
pub type FriLayerVerifyBackend<F> =
    crypto::merkle_tree::backends::field_element_vector::FieldElementVectorBackend<
        F,
        sha3::Keccak256,
        32,
        2,
    >;
