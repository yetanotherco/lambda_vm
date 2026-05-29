//! Tests for the field-element-vector Merkle tree backend.

use math::field::{element::FieldElement, goldilocks::GoldilocksField};
use sha2::Sha512;
use sha3::{Keccak256, Keccak512, Sha3_256, Sha3_512};

use crate::merkle_tree::{
    backends::field_element_vector::FieldElementVectorBackend, merkle::MerkleTree,
    traits::IsMerkleTreeBackend,
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

#[test]
fn hash_elements_matches_hash_data_byte_for_byte() {
    type Backend = FieldElementVectorBackend<F, Keccak256, 32>;

    // Pseudo-random Vec generated from a simple LCG so the test is deterministic
    // yet exercises a non-trivial sequence of field elements.
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let v: Vec<FE> = (0..37)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            FE::from(state)
        })
        .collect();

    let via_hash_data = Backend::hash_data(&v);
    let via_hash_elements = Backend::hash_elements(v.iter());

    assert_eq!(
        via_hash_data, via_hash_elements,
        "hash_elements must be byte-identical to hash_data over the same sequence"
    );

    // Empty sequence must also agree.
    let empty: Vec<FE> = Vec::new();
    assert_eq!(
        Backend::hash_data(&empty),
        Backend::hash_elements(empty.iter())
    );
}
