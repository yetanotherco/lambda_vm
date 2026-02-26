use core::marker::PhantomData;

use crate::hash::poseidon::Poseidon;
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use alloc::vec::Vec;
use digest::{Digest, Output};
use math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

/// A backend for Merkle trees that uses fixed-size pairs of field elements.
/// This is more efficient than `FieldElementVectorBackend` when the batch size is always 2,
/// as it avoids Vec allocation overhead.
#[derive(Clone)]
pub struct FieldElementPairBackend<F, D: Digest, const NUM_BYTES: usize> {
    phantom1: PhantomData<F>,
    phantom2: PhantomData<D>,
}

impl<F, D: Digest, const NUM_BYTES: usize> Default for FieldElementPairBackend<F, D, NUM_BYTES> {
    fn default() -> Self {
        Self {
            phantom1: PhantomData,
            phantom2: PhantomData,
        }
    }
}

impl<F, D: Digest, const NUM_BYTES: usize> IsMerkleTreeBackend
    for FieldElementPairBackend<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
{
    type Node = [u8; NUM_BYTES];
    type Data = [FieldElement<F>; 2];

    fn hash_data(input: &[FieldElement<F>; 2]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(input[0].as_bytes());
        hasher.update(input[1].as_bytes());
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }

    fn hash_new_parent(left: &[u8; NUM_BYTES], right: &[u8; NUM_BYTES]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(left);
        hasher.update(right);
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }
}

/// Quaternary (arity-4) backend for fixed-size pairs of field elements.
#[derive(Clone)]
pub struct QuaternaryFieldElementPairBackend<F, D: Digest, const NUM_BYTES: usize> {
    phantom1: PhantomData<F>,
    phantom2: PhantomData<D>,
}

impl<F, D: Digest, const NUM_BYTES: usize> Default
    for QuaternaryFieldElementPairBackend<F, D, NUM_BYTES>
{
    fn default() -> Self {
        Self {
            phantom1: PhantomData,
            phantom2: PhantomData,
        }
    }
}

impl<F, D: Digest, const NUM_BYTES: usize> IsMerkleTreeBackend
    for QuaternaryFieldElementPairBackend<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
{
    type Node = [u8; NUM_BYTES];
    type Data = [FieldElement<F>; 2];

    const ARITY: usize = 4;

    fn hash_data(input: &[FieldElement<F>; 2]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(input[0].as_bytes());
        hasher.update(input[1].as_bytes());
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }

    fn hash_new_parent(left: &[u8; NUM_BYTES], right: &[u8; NUM_BYTES]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(left);
        hasher.update(right);
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }

    fn hash_children(children: &[Self::Node]) -> Self::Node {
        debug_assert_eq!(children.len(), 4);
        let mut hasher = D::new();
        for child in children {
            hasher.update(child);
        }
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }
}

#[derive(Clone)]
pub struct FieldElementVectorBackend<F, D: Digest, const NUM_BYTES: usize> {
    phantom1: PhantomData<F>,
    phantom2: PhantomData<D>,
}

impl<F, D: Digest, const NUM_BYTES: usize> Default for FieldElementVectorBackend<F, D, NUM_BYTES> {
    fn default() -> Self {
        Self {
            phantom1: PhantomData,
            phantom2: PhantomData,
        }
    }
}

impl<F, D: Digest, const NUM_BYTES: usize> IsMerkleTreeBackend
    for FieldElementVectorBackend<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
    Vec<FieldElement<F>>: Sync + Send,
{
    type Node = [u8; NUM_BYTES];
    type Data = Vec<FieldElement<F>>;

    fn hash_data(input: &Vec<FieldElement<F>>) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        for element in input.iter() {
            hasher.update(element.as_bytes());
        }
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }

    fn hash_new_parent(left: &[u8; NUM_BYTES], right: &[u8; NUM_BYTES]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(left);
        hasher.update(right);
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }
}

#[derive(Clone, Default)]
pub struct BatchPoseidonTree<P: Poseidon + Default> {
    _poseidon: PhantomData<P>,
}

impl<P> IsMerkleTreeBackend for BatchPoseidonTree<P>
where
    P: Poseidon + Default,
    Vec<FieldElement<P::F>>: Sync + Send,
    FieldElement<P::F>: Sync + Send,
{
    type Node = FieldElement<P::F>;
    type Data = Vec<FieldElement<P::F>>;

    fn hash_data(input: &Vec<FieldElement<P::F>>) -> FieldElement<P::F> {
        P::hash_many(input)
    }

    fn hash_new_parent(
        left: &FieldElement<P::F>,
        right: &FieldElement<P::F>,
    ) -> FieldElement<P::F> {
        P::hash(left, right)
    }
}

/// A quaternary (arity-4) backend for Merkle trees using vectors of field elements.
/// Tree construction requires ~3x fewer hash calls than the binary equivalent,
/// at the cost of 1.5x larger proofs.
#[derive(Clone)]
pub struct QuaternaryFieldElementVectorBackend<F, D: Digest, const NUM_BYTES: usize> {
    phantom1: PhantomData<F>,
    phantom2: PhantomData<D>,
}

impl<F, D: Digest, const NUM_BYTES: usize> Default
    for QuaternaryFieldElementVectorBackend<F, D, NUM_BYTES>
{
    fn default() -> Self {
        Self {
            phantom1: PhantomData,
            phantom2: PhantomData,
        }
    }
}

impl<F, D: Digest, const NUM_BYTES: usize> IsMerkleTreeBackend
    for QuaternaryFieldElementVectorBackend<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<Output<D>>,
    Vec<FieldElement<F>>: Sync + Send,
{
    type Node = [u8; NUM_BYTES];
    type Data = Vec<FieldElement<F>>;

    const ARITY: usize = 4;

    fn hash_data(input: &Vec<FieldElement<F>>) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        for element in input.iter() {
            hasher.update(element.as_bytes());
        }
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }

    fn hash_new_parent(left: &[u8; NUM_BYTES], right: &[u8; NUM_BYTES]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(left);
        hasher.update(right);
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }

    fn hash_children(children: &[Self::Node]) -> Self::Node {
        debug_assert_eq!(children.len(), 4);
        let mut hasher = D::new();
        for child in children {
            hasher.update(child);
        }
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }
}

#[cfg(test)]
mod tests {
    use math::field::{
        element::FieldElement, fields::fft_friendly::u64_goldilocks::GoldilocksField,
    };
    use sha2::Sha512;
    use sha3::{Keccak256, Keccak512, Sha3_256, Sha3_512};

    use crate::merkle_tree::{
        backends::field_element_vector::FieldElementVectorBackend, merkle::MerkleTree,
    };

    type F = GoldilocksField;
    type FE = FieldElement<F>;

    #[test]
    fn hash_data_field_element_backend_works_with_sha3_256() {
        let values = [
            vec![FE::from(2u64), FE::from(11u64)],
            vec![FE::from(3u64), FE::from(14u64)],
            vec![FE::from(4u64), FE::from(7u64)],
            vec![FE::from(5u64), FE::from(3u64)],
            vec![FE::from(6u64), FE::from(5u64)],
            vec![FE::from(7u64), FE::from(16u64)],
            vec![FE::from(8u64), FE::from(19u64)],
            vec![FE::from(9u64), FE::from(21u64)],
        ];
        let merkle_tree =
            MerkleTree::<FieldElementVectorBackend<F, Sha3_256, 32>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementVectorBackend<F, Sha3_256, 32>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_keccak256() {
        let values = [
            vec![FE::from(2u64), FE::from(11u64)],
            vec![FE::from(3u64), FE::from(14u64)],
            vec![FE::from(4u64), FE::from(7u64)],
            vec![FE::from(5u64), FE::from(3u64)],
            vec![FE::from(6u64), FE::from(5u64)],
            vec![FE::from(7u64), FE::from(16u64)],
            vec![FE::from(8u64), FE::from(19u64)],
            vec![FE::from(9u64), FE::from(21u64)],
        ];
        let merkle_tree =
            MerkleTree::<FieldElementVectorBackend<F, Keccak256, 32>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementVectorBackend<F, Keccak256, 32>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_sha3_512() {
        let values = [
            vec![FE::from(2u64), FE::from(11u64)],
            vec![FE::from(3u64), FE::from(14u64)],
            vec![FE::from(4u64), FE::from(7u64)],
            vec![FE::from(5u64), FE::from(3u64)],
            vec![FE::from(6u64), FE::from(5u64)],
            vec![FE::from(7u64), FE::from(16u64)],
            vec![FE::from(8u64), FE::from(19u64)],
            vec![FE::from(9u64), FE::from(21u64)],
        ];
        let merkle_tree =
            MerkleTree::<FieldElementVectorBackend<F, Sha3_512, 64>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementVectorBackend<F, Sha3_512, 64>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_keccak512() {
        let values = [
            vec![FE::from(2u64), FE::from(11u64)],
            vec![FE::from(3u64), FE::from(14u64)],
            vec![FE::from(4u64), FE::from(7u64)],
            vec![FE::from(5u64), FE::from(3u64)],
            vec![FE::from(6u64), FE::from(5u64)],
            vec![FE::from(7u64), FE::from(16u64)],
            vec![FE::from(8u64), FE::from(19u64)],
            vec![FE::from(9u64), FE::from(21u64)],
        ];
        let merkle_tree =
            MerkleTree::<FieldElementVectorBackend<F, Keccak512, 64>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementVectorBackend<F, Keccak512, 64>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_sha2_512() {
        let values = [
            vec![FE::from(2u64), FE::from(11u64)],
            vec![FE::from(3u64), FE::from(14u64)],
            vec![FE::from(4u64), FE::from(7u64)],
            vec![FE::from(5u64), FE::from(3u64)],
            vec![FE::from(6u64), FE::from(5u64)],
            vec![FE::from(7u64), FE::from(16u64)],
            vec![FE::from(8u64), FE::from(19u64)],
            vec![FE::from(9u64), FE::from(21u64)],
        ];
        let merkle_tree =
            MerkleTree::<FieldElementVectorBackend<F, Sha512, 64>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementVectorBackend<F, Sha512, 64>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    // --- Quaternary backend tests ---

    use super::QuaternaryFieldElementVectorBackend;

    type Q4Keccak = QuaternaryFieldElementVectorBackend<F, Keccak256, 32>;
    type BinaryKeccak = FieldElementVectorBackend<F, Keccak256, 32>;

    fn make_leaves(n: usize) -> Vec<Vec<FE>> {
        (1..=n)
            .map(|i| vec![FE::from(i as u64), FE::from((i * 7) as u64)])
            .collect()
    }

    #[test]
    fn quaternary_build_16_leaves() {
        let values = make_leaves(16);
        let tree = MerkleTree::<Q4Keccak>::build(&values).unwrap();
        // 16 leaves in arity-4 tree: total = (4*16 - 1) / 3 = 21 nodes
        assert_eq!(tree.root, tree.root); // root exists and is well-formed
    }

    #[test]
    fn quaternary_build_4_leaves_minimal() {
        let values = make_leaves(4);
        let tree = MerkleTree::<Q4Keccak>::build(&values).unwrap();
        // 4 leaves in arity-4 tree: total = (4*4 - 1) / 3 = 5 nodes
        assert_eq!(tree.root, tree.root);
    }

    #[test]
    fn quaternary_single_proof_all_positions() {
        let values = make_leaves(16);
        let tree = MerkleTree::<Q4Keccak>::build(&values).unwrap();
        for pos in 0..16 {
            let proof = tree.get_proof_by_pos(pos).unwrap();
            assert!(
                proof.verify::<Q4Keccak>(&tree.root, pos, &values[pos]),
                "Single proof failed at position {pos}"
            );
        }
    }

    #[test]
    fn quaternary_single_proof_4_leaves() {
        let values = make_leaves(4);
        let tree = MerkleTree::<Q4Keccak>::build(&values).unwrap();
        for pos in 0..4 {
            let proof = tree.get_proof_by_pos(pos).unwrap();
            assert!(
                proof.verify::<Q4Keccak>(&tree.root, pos, &values[pos]),
                "Single proof failed at position {pos} for 4-leaf tree"
            );
        }
    }

    #[test]
    fn quaternary_single_proof_wrong_value_fails() {
        let values = make_leaves(16);
        let tree = MerkleTree::<Q4Keccak>::build(&values).unwrap();
        let proof = tree.get_proof_by_pos(3).unwrap();
        let wrong_value = vec![FE::from(999u64), FE::from(888u64)];
        assert!(!proof.verify::<Q4Keccak>(&tree.root, 3, &wrong_value));
    }

    #[test]
    fn quaternary_single_proof_wrong_root_fails() {
        let values = make_leaves(16);
        let tree = MerkleTree::<Q4Keccak>::build(&values).unwrap();
        let proof = tree.get_proof_by_pos(0).unwrap();
        let wrong_root = [0u8; 32];
        assert!(!proof.verify::<Q4Keccak>(&wrong_root, 0, &values[0]));
    }

    #[test]
    fn quaternary_batch_proof_adjacent_leaves() {
        let values = make_leaves(16);
        let tree = MerkleTree::<Q4Keccak>::build(&values).unwrap();
        // 4 siblings (same parent group)
        let pos_list = vec![0, 1, 2, 3];
        let batch_proof = tree.get_batch_proof(&pos_list).unwrap();
        let leaves: Vec<_> = pos_list.iter().map(|&i| values[i].clone()).collect();
        assert!(
            batch_proof.verify::<Q4Keccak>(&tree.root, &pos_list, &leaves, 16),
            "Batch proof for adjacent 4 siblings failed"
        );
    }

    #[test]
    fn quaternary_batch_proof_sparse_leaves() {
        let values = make_leaves(16);
        let tree = MerkleTree::<Q4Keccak>::build(&values).unwrap();
        // Spread across different subtrees
        let pos_list = vec![0, 5, 10, 15];
        let batch_proof = tree.get_batch_proof(&pos_list).unwrap();
        let leaves: Vec<_> = pos_list.iter().map(|&i| values[i].clone()).collect();
        assert!(
            batch_proof.verify::<Q4Keccak>(&tree.root, &pos_list, &leaves, 16),
            "Batch proof for sparse leaves failed"
        );
    }

    #[test]
    fn quaternary_batch_proof_single_leaf() {
        let values = make_leaves(16);
        let tree = MerkleTree::<Q4Keccak>::build(&values).unwrap();
        let pos_list = vec![7];
        let batch_proof = tree.get_batch_proof(&pos_list).unwrap();
        let leaves: Vec<_> = pos_list.iter().map(|&i| values[i].clone()).collect();
        assert!(
            batch_proof.verify::<Q4Keccak>(&tree.root, &pos_list, &leaves, 16),
            "Batch proof for single leaf failed"
        );
    }

    #[test]
    fn quaternary_batch_proof_many_leaves() {
        let values = make_leaves(16);
        let tree = MerkleTree::<Q4Keccak>::build(&values).unwrap();
        let pos_list: Vec<usize> = (0..16).collect();
        let batch_proof = tree.get_batch_proof(&pos_list).unwrap();
        let leaves: Vec<_> = pos_list.iter().map(|&i| values[i].clone()).collect();
        assert!(
            batch_proof.verify::<Q4Keccak>(&tree.root, &pos_list, &leaves, 16),
            "Batch proof for all 16 leaves failed"
        );
    }

    #[test]
    fn quaternary_batch_proof_wrong_values_fails() {
        let values = make_leaves(16);
        let tree = MerkleTree::<Q4Keccak>::build(&values).unwrap();
        let pos_list = vec![0, 5];
        let batch_proof = tree.get_batch_proof(&pos_list).unwrap();
        let wrong_leaves = vec![
            vec![FE::from(999u64), FE::from(888u64)],
            vec![FE::from(777u64), FE::from(666u64)],
        ];
        assert!(
            !batch_proof.verify::<Q4Keccak>(&tree.root, &pos_list, &wrong_leaves, 16),
            "Batch proof should fail with wrong values"
        );
    }

    #[test]
    fn quaternary_vs_binary_different_roots() {
        let values = make_leaves(16);
        let binary_tree = MerkleTree::<BinaryKeccak>::build(&values).unwrap();
        let quaternary_tree = MerkleTree::<Q4Keccak>::build(&values).unwrap();
        assert_ne!(
            binary_tree.root, quaternary_tree.root,
            "Binary and quaternary trees with same leaves should produce different roots"
        );
    }

    #[test]
    fn quaternary_padding_non_power_of_4() {
        // 5 leaves → padded to 16 (next power of 4)
        let values = make_leaves(5);
        let tree = MerkleTree::<Q4Keccak>::build(&values).unwrap();
        let proof = tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<Q4Keccak>(&tree.root, 0, &values[0]));
    }

    #[test]
    fn quaternary_64_leaves_single_and_batch() {
        let values = make_leaves(64);
        let tree = MerkleTree::<Q4Keccak>::build(&values).unwrap();

        // Single proofs at various positions
        for &pos in &[0, 15, 31, 47, 63] {
            let proof = tree.get_proof_by_pos(pos).unwrap();
            assert!(
                proof.verify::<Q4Keccak>(&tree.root, pos, &values[pos]),
                "Single proof failed for 64-leaf tree at position {pos}"
            );
        }

        // Batch proof
        let pos_list = vec![0, 15, 31, 47, 63];
        let batch_proof = tree.get_batch_proof(&pos_list).unwrap();
        let leaves: Vec<_> = pos_list.iter().map(|&i| values[i].clone()).collect();
        assert!(
            batch_proof.verify::<Q4Keccak>(&tree.root, &pos_list, &leaves, 64),
            "Batch proof failed for 64-leaf tree"
        );
    }
}
