use crypto::merkle_tree::{
    backends::types::{BatchKeccak256Backend, Keccak256Backend, PairKeccak256Backend},
    merkle::MerkleTree,
    proof::verify_merkle_path_fe_slice,
};
use math::field::{element::FieldElement, traits::IsField};
use math::traits::ByteConversion;
use sha3::Keccak256;

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

/// Verify a Merkle inclusion proof over [`BatchedMerkleTreeBackend`] reading the
/// leaf value straight from a borrowed slice (no `Vec` materialization), producing
/// the identical root to [`BatchedMerkleTree::verify`]. Used by the verifier hot
/// path to hash trace/composition openings without per-opening allocation.
pub fn verify_batched_merkle_path_slice<F>(
    merkle_path: &[Commitment],
    root_hash: &Commitment,
    index: usize,
    value: &[FieldElement<F>],
) -> bool
where
    F: IsField,
    FieldElement<F>: ByteConversion,
{
    verify_merkle_path_fe_slice::<F, Keccak256, COMMITMENT_SIZE>(
        merkle_path,
        root_hash,
        index,
        value,
    )
}
