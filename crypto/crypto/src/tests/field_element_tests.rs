//! Tests for the field-element Merkle tree backend.

use math::field::{element::FieldElement, goldilocks::GoldilocksField};
use sha3::{Keccak256, Keccak512, Sha3_256, Sha3_512};

use crate::merkle_tree::{backends::field_element::FieldElementBackend, merkle::MerkleTree};

type F = GoldilocksField;
type FE = FieldElement<F>;

#[test]
fn hash_data_field_element_backend_works_with_keccak_256() {
    let values: Vec<FE> = (1..6).map(FE::from).collect();
    let merkle_tree = MerkleTree::<FieldElementBackend<F, Keccak256, 32>>::build(&values).unwrap();
    let proof = merkle_tree.get_proof_by_pos(0).unwrap();
    assert!(proof.verify::<FieldElementBackend<F, Keccak256, 32>>(
        &merkle_tree.root,
        0,
        &values[0]
    ));
}

#[test]
fn hash_data_field_element_backend_works_with_sha3_256() {
    let values: Vec<FE> = (1..6).map(FE::from).collect();
    let merkle_tree = MerkleTree::<FieldElementBackend<F, Sha3_256, 32>>::build(&values).unwrap();
    let proof = merkle_tree.get_proof_by_pos(0).unwrap();
    assert!(proof.verify::<FieldElementBackend<F, Sha3_256, 32>>(&merkle_tree.root, 0, &values[0]));
}

#[test]
fn hash_data_field_element_backend_works_with_keccak_512() {
    let values: Vec<FE> = (1..6).map(FE::from).collect();
    let merkle_tree = MerkleTree::<FieldElementBackend<F, Keccak512, 64>>::build(&values).unwrap();
    let proof = merkle_tree.get_proof_by_pos(0).unwrap();
    assert!(proof.verify::<FieldElementBackend<F, Keccak512, 64>>(
        &merkle_tree.root,
        0,
        &values[0]
    ));
}

#[test]
fn hash_data_field_element_backend_works_with_sha3_512() {
    let values: Vec<FE> = (1..6).map(FE::from).collect();
    let merkle_tree = MerkleTree::<FieldElementBackend<F, Sha3_512, 64>>::build(&values).unwrap();
    let proof = merkle_tree.get_proof_by_pos(0).unwrap();
    assert!(proof.verify::<FieldElementBackend<F, Sha3_512, 64>>(&merkle_tree.root, 0, &values[0]));
}

/// Tests batch proof with a real cryptographic hash (Keccak256).
/// This verifies that proof ordering works correctly with non-commutative hashes.
#[test]
fn batch_proof_with_keccak256_verifies_sparse_leaves() {
    let values: Vec<FE> = (1..=16).map(FE::from).collect();
    let merkle_tree = MerkleTree::<FieldElementBackend<F, Keccak256, 32>>::build(&values).unwrap();

    // Test with sparse leaves across different subtrees
    let pos_list: Vec<usize> = vec![1, 8, 9, 15];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();
    let leaves_to_verify: Vec<FE> = pos_list.iter().map(|&i| values[i]).collect();

    assert!(
        batch_proof.verify::<FieldElementBackend<F, Keccak256, 32>>(
            &merkle_tree.root,
            &pos_list,
            &leaves_to_verify,
            16
        ),
        "Batch proof verification failed with Keccak256"
    );
}

/// Tests batch proof with adjacent leaves using real hash.
#[test]
fn batch_proof_with_keccak256_verifies_adjacent_leaves() {
    let values: Vec<FE> = (1..=8).map(FE::from).collect();
    let merkle_tree = MerkleTree::<FieldElementBackend<F, Keccak256, 32>>::build(&values).unwrap();

    // Adjacent leaves (siblings)
    let pos_list: Vec<usize> = vec![0, 1];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();
    let leaves_to_verify: Vec<FE> = pos_list.iter().map(|&i| values[i]).collect();

    assert!(
        batch_proof.verify::<FieldElementBackend<F, Keccak256, 32>>(
            &merkle_tree.root,
            &pos_list,
            &leaves_to_verify,
            8
        ),
        "Batch proof verification failed for adjacent leaves"
    );
}

/// Tests that batch proof fails with wrong values using real hash.
#[test]
fn batch_proof_with_keccak256_fails_with_wrong_values() {
    let values: Vec<FE> = (1..=8).map(FE::from).collect();
    let merkle_tree = MerkleTree::<FieldElementBackend<F, Keccak256, 32>>::build(&values).unwrap();

    let pos_list: Vec<usize> = vec![0, 1];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();

    // Use wrong values
    let wrong_values: Vec<FE> = vec![FE::from(999), FE::from(998)];

    assert!(
        !batch_proof.verify::<FieldElementBackend<F, Keccak256, 32>>(
            &merkle_tree.root,
            &pos_list,
            &wrong_values,
            8
        ),
        "Batch proof should fail with wrong values"
    );
}
