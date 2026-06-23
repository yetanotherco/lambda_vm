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

/// Like [`verify_batched_merkle_path_slice`] but takes a caller-owned
/// `leaf_scratch` byte buffer reused across calls to eliminate the per-call
/// `Vec<u8>` allocation inside leaf serialization.
pub fn verify_batched_merkle_path_slice_with_scratch<F>(
    merkle_path: &[Commitment],
    root_hash: &Commitment,
    index: usize,
    value: &[FieldElement<F>],
    leaf_scratch: &mut alloc::vec::Vec<u8>,
) -> bool
where
    F: IsField,
    FieldElement<F>: ByteConversion,
{
    const ARITY: usize = 4;
    crypto::merkle_tree::proof::verify_merkle_path_keccak256_with_scratch::<F, ARITY>(
        merkle_path,
        root_hash,
        index,
        value,
        leaf_scratch,
    )
}

/// Verify TWO trace openings at `(iota*2, iota*2+1)` against the same root in a
/// single pass. For ARITY=4 trees both leaf indices are always in the same
/// level-0 quaternary group, so the level-0 parent and all ancestor hashes are
/// shared — this saves one full ancestor-path traversal per (iota, iota_sym) pair.
///
/// See [`crypto::merkle_tree::proof::verify_paired_keccak256_openings`] for details.
pub fn verify_paired_batched_openings<F>(
    merkle_path: &[Commitment],
    root_hash: &Commitment,
    index: usize,
    value_a: &[FieldElement<F>],
    value_b: &[FieldElement<F>],
    leaf_scratch: &mut alloc::vec::Vec<u8>,
) -> bool
where
    F: IsField,
    FieldElement<F>: ByteConversion,
{
    const ARITY: usize = 4;
    const _: () = assert!(
        ARITY
            == <BatchedMerkleTreeBackend<math::field::goldilocks::GoldilocksField> as crypto::merkle_tree::traits::IsMerkleTreeBackend>::ARITY
    );
    crypto::merkle_tree::proof::verify_paired_keccak256_openings::<F, ARITY>(
        merkle_path,
        root_hash,
        index,
        value_a,
        value_b,
        leaf_scratch,
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
    let mut scratch = alloc::vec::Vec::new();
    verify_fri_merkle_path_slice_with_scratch(merkle_path, root_hash, index, value, &mut scratch)
}

/// Like [`verify_fri_merkle_path_slice`] but takes a caller-owned `leaf_scratch`
/// byte buffer reused across calls to avoid per-call allocation.
pub fn verify_fri_merkle_path_slice_with_scratch<F>(
    merkle_path: &[Commitment],
    root_hash: &Commitment,
    index: usize,
    value: &[FieldElement<F>],
    leaf_scratch: &mut alloc::vec::Vec<u8>,
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
    crypto::merkle_tree::proof::verify_merkle_path_keccak256_with_scratch::<F, ARITY>(
        merkle_path,
        root_hash,
        index,
        value,
        leaf_scratch,
    )
}
