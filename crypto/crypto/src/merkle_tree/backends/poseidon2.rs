//! Poseidon2 hash-based Merkle tree backend for Goldilocks field
//!
//! This backend uses the Poseidon2 hash function which is optimized for
//! zero-knowledge proof systems. It operates natively on Goldilocks field
//! elements, avoiding the overhead of byte serialization.
//!
//! Benefits over Keccak256/SHA3:
//! - Field-native operations (no byte conversions)
//! - ZK-friendly (low constraint count)
//! - GPU-acceleratable (Metal implementation available)

use crate::hash::poseidon2::{Fp, Poseidon2};
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use alloc::vec::Vec;
use core::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

/// Poseidon2 Merkle tree backend for single Goldilocks field elements
///
/// Node type: Goldilocks field element (64-bit)
/// Data type: Goldilocks field element (64-bit)
#[derive(Clone, Default)]
pub struct Poseidon2Backend;

impl IsMerkleTreeBackend for Poseidon2Backend {
    type Node = Fp;
    type Data = Fp;

    fn hash_data(leaf: &Self::Data) -> Self::Node {
        Poseidon2::hash_single(leaf)
    }

    fn hash_new_parent(left: &Self::Node, right: &Self::Node) -> Self::Node {
        Poseidon2::compress(left, right)
    }
}

/// Poseidon2 Merkle tree backend for vectors of Goldilocks field elements
///
/// Node type: Goldilocks field element (64-bit)
/// Data type: Vec<Goldilocks field element>
///
/// This is useful for committing to rows of field elements (e.g., trace columns)
#[derive(Clone, Default)]
pub struct BatchPoseidon2Backend;

impl IsMerkleTreeBackend for BatchPoseidon2Backend {
    type Node = Fp;
    type Data = Vec<Fp>;

    fn hash_data(leaf: &Self::Data) -> Self::Node {
        Poseidon2::hash_vec(leaf)
    }

    fn hash_leaves(unhashed_leaves: &[Self::Data]) -> Vec<Self::Node> {
        #[cfg(feature = "parallel")]
        let iter = unhashed_leaves.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = unhashed_leaves.iter();

        iter.map(Self::hash_data).collect()
    }

    fn hash_new_parent(left: &Self::Node, right: &Self::Node) -> Self::Node {
        Poseidon2::compress(left, right)
    }
}

/// Generic Poseidon2 backend that can work with any field that can be converted to Goldilocks
///
/// This allows using Poseidon2 with other fields by converting them to Goldilocks
/// for hashing purposes.
#[derive(Clone, Default)]
pub struct GenericPoseidon2Backend<F> {
    _phantom: PhantomData<F>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle_tree::merkle::MerkleTree;

    #[test]
    fn test_poseidon2_backend_single_element() {
        let values: Vec<Fp> = (1..6).map(|i| Fp::from(i as u64)).collect();
        let merkle_tree = MerkleTree::<Poseidon2Backend>::build(&values).unwrap();

        // Verify proof for first element
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<Poseidon2Backend>(&merkle_tree.root, 0, &values[0]));

        // Verify proof for last element
        let proof = merkle_tree.get_proof_by_pos(4).unwrap();
        assert!(proof.verify::<Poseidon2Backend>(&merkle_tree.root, 4, &values[4]));
    }

    #[test]
    fn test_poseidon2_backend_deterministic() {
        let values: Vec<Fp> = (1..10).map(|i| Fp::from(i as u64)).collect();

        let tree1 = MerkleTree::<Poseidon2Backend>::build(&values).unwrap();
        let tree2 = MerkleTree::<Poseidon2Backend>::build(&values).unwrap();

        assert_eq!(tree1.root, tree2.root);
    }

    #[test]
    fn test_poseidon2_backend_different_roots() {
        let values1: Vec<Fp> = (1..6).map(|i| Fp::from(i as u64)).collect();
        let values2: Vec<Fp> = (2..7).map(|i| Fp::from(i as u64)).collect();

        let tree1 = MerkleTree::<Poseidon2Backend>::build(&values1).unwrap();
        let tree2 = MerkleTree::<Poseidon2Backend>::build(&values2).unwrap();

        assert_ne!(tree1.root, tree2.root);
    }

    #[test]
    fn test_batch_poseidon2_backend() {
        let values: Vec<Vec<Fp>> = (0..4)
            .map(|i| (0..3).map(|j| Fp::from((i * 3 + j) as u64)).collect())
            .collect();

        let merkle_tree = MerkleTree::<BatchPoseidon2Backend>::build(&values).unwrap();

        // Verify proof for first element
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<BatchPoseidon2Backend>(&merkle_tree.root, 0, &values[0]));
    }

    #[test]
    fn test_poseidon2_backend_power_of_two() {
        // Test with exact power of 2
        let values: Vec<Fp> = (1..=8).map(|i| Fp::from(i as u64)).collect();
        let merkle_tree = MerkleTree::<Poseidon2Backend>::build(&values).unwrap();

        for i in 0..8 {
            let proof = merkle_tree.get_proof_by_pos(i).unwrap();
            assert!(proof.verify::<Poseidon2Backend>(&merkle_tree.root, i, &values[i]));
        }
    }

    #[test]
    fn test_poseidon2_backend_single_leaf() {
        let values = vec![Fp::from(42u64)];
        let merkle_tree = MerkleTree::<Poseidon2Backend>::build(&values).unwrap();

        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<Poseidon2Backend>(&merkle_tree.root, 0, &values[0]));
    }

    #[test]
    fn test_poseidon2_backend_large_tree() {
        // Test with larger tree (1024 leaves)
        let values: Vec<Fp> = (0..1024).map(|i| Fp::from(i as u64)).collect();
        let merkle_tree = MerkleTree::<Poseidon2Backend>::build(&values).unwrap();

        // Spot check a few proofs
        for i in [0, 100, 500, 1023] {
            let proof = merkle_tree.get_proof_by_pos(i).unwrap();
            assert!(
                proof.verify::<Poseidon2Backend>(&merkle_tree.root, i, &values[i]),
                "Failed for index {}",
                i
            );
        }
    }
}
