use crypto::merkle_tree::{
    backends::types::{BatchKeccak256Backend, Keccak256Backend, PairKeccak256Backend},
    merkle::MerkleTree,
};
use math::field::{element::FieldElement, traits::IsField};
use math::traits::ByteConversion;

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
///
/// Hashes via the specialized single-block Keccak256 sponge
/// ([`verify_merkle_path_keccak256`]), which runs each permutation as one
/// `keccak::f1600` (the `KeccakPermute` precompile on the guest) and skips the
/// generic `sha3` block-buffer wrapper. The Keccak256 output is identical, so this
/// is transparent — same roots, same proofs.
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
    // ARITY must match `BatchedMerkleTreeBackend`'s tree arity (the trees this
    // verifies against were committed with that backend). Asserted below so a
    // future arity change to the backend trips this rather than silently
    // mismatching the commitment.
    const ARITY: usize = 4;
    const _: () = assert!(
        ARITY
            == <BatchedMerkleTreeBackend<math::field::goldilocks::GoldilocksField> as crypto::merkle_tree::traits::IsMerkleTreeBackend>::ARITY
    );
    crypto::merkle_tree::proof::verify_merkle_path_keccak256::<F, ARITY>(
        merkle_path,
        root_hash,
        index,
        value,
    )
}

/// Like [`verify_batched_merkle_path_slice`] but for the FRI-layer commitment,
/// which uses the **binary** [`FriLayerMerkleTreeBackend`] (a `PairKeccak256`
/// tree). The FRI trees stay binary; only the trace/composition trees are
/// quaternary, so this opening must walk an arity-2 path.
pub fn verify_fri_merkle_path_slice<F>(
    merkle_path: &[Commitment],
    root_hash: &Commitment,
    index: usize,
    value: &[FieldElement<F>],
) -> bool
where
    F: IsField,
    FieldElement<F>: ByteConversion,
{
    const ARITY: usize = 2;
    const _: () = assert!(
        ARITY
            == <FriLayerMerkleTreeBackend<math::field::goldilocks::GoldilocksField> as crypto::merkle_tree::traits::IsMerkleTreeBackend>::ARITY
    );
    crypto::merkle_tree::proof::verify_merkle_path_keccak256::<F, ARITY>(
        merkle_path,
        root_hash,
        index,
        value,
    )
}
