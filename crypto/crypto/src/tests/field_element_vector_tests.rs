//! Tests for the field-element-vector Merkle tree backend.

use math::field::{element::FieldElement, goldilocks::GoldilocksField};
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
