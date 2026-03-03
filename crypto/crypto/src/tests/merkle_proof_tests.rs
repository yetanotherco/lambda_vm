use alloc::vec::Vec;
use math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

use crate::merkle_tree::merkle::MerkleTree;
use crate::tests::merkle_tests::TestBackend;

/// Small field useful for starks, sometimes called min i goldilocks
/// Used in miden and winterfell
pub type Ecgfp5 = U64PrimeField<0xFFFF_FFFF_0000_0001_u64>;
pub type Ecgfp5FE = FieldElement<Ecgfp5>;
pub type TestMerkleTreeEcgfp = MerkleTree<TestBackend<Ecgfp5>>;

const MODULUS: u64 = 13;
type U64PF = U64PrimeField<MODULUS>;
type FE = FieldElement<U64PF>;

#[test]
fn create_a_proof_over_value_that_belongs_to_a_given_merkle_tree_when_given_the_leaf_position() {
    // 5 values get padded to 16 leaves in a 4-ary tree
    let values: Vec<FE> = (1..6).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let proof = merkle_tree.get_proof_by_pos(1).unwrap();
    assert!(proof.verify::<TestBackend<U64PF>>(&merkle_tree.root, 1, &FE::new(2)));
}

#[test]
fn create_a_merkle_tree_with_10000_elements_and_verify_that_an_element_is_part_of_it() {
    let values: Vec<Ecgfp5FE> = (1..10000).map(Ecgfp5FE::new).collect();
    let merkle_tree = TestMerkleTreeEcgfp::build(&values).unwrap();
    let proof = merkle_tree.get_proof_by_pos(9349).unwrap();
    assert!(proof.verify::<TestBackend<Ecgfp5>>(&merkle_tree.root, 9349, &Ecgfp5FE::new(9350)));
}

#[test]
fn verify_merkle_proof_for_single_value() {
    let values: Vec<FE> = vec![FE::new(1)];
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    // hash_data(1) = 2, single leaf (1 is a power of 4), root = 2
    let expected_root = FE::new(2);
    assert_eq!(
        merkle_tree.root, expected_root,
        "The root of the Merkle tree does not match the expected value."
    );

    let proof = merkle_tree.get_proof_by_pos(0).unwrap();
    assert!(
        proof.verify::<TestBackend<U64PF>>(&merkle_tree.root, 0, &values[0]),
        "The proof verification failed for the element at position 0."
    );
}

#[test]
fn verify_merkle_proof_for_four_values() {
    // 4 values = power of 4, no padding needed
    let values: Vec<FE> = (1..5).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    // hash_data: [2, 4, 6, 8], root = 2+4+6+8 = 20 mod 13 = 7
    assert_eq!(merkle_tree.root, FE::new(7));

    // Verify each position
    for i in 0..4 {
        let proof = merkle_tree.get_proof_by_pos(i).unwrap();
        assert!(
            proof.verify::<TestBackend<U64PF>>(&merkle_tree.root, i, &values[i]),
            "Proof verification failed for position {i}."
        );
    }
}

#[test]
fn verify_proof_fails_with_wrong_root() {
    let values: Vec<FE> = (1..5).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let proof = merkle_tree.get_proof_by_pos(0).unwrap();
    let wrong_root = FE::new(999);
    assert!(!proof.verify::<TestBackend<U64PF>>(&wrong_root, 0, &values[0]));
}

#[test]
fn verify_proof_fails_with_wrong_value() {
    let values: Vec<FE> = (1..5).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let proof = merkle_tree.get_proof_by_pos(0).unwrap();
    let wrong_value = FE::new(999);
    assert!(!proof.verify::<TestBackend<U64PF>>(&merkle_tree.root, 0, &wrong_value));
}

#[test]
fn batch_proof_verify_adjacent_leaves() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    // Prove adjacent leaves (0 and 1 share the same parent in 4-ary)
    let pos_list = vec![0, 1];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();
    let leaves_to_verify: Vec<FE> = pos_list.iter().map(|&i| values[i]).collect();

    // num_leaves=8, verify internally pads to 16
    assert!(batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        &pos_list,
        &leaves_to_verify,
        8
    ));
}

#[test]
fn batch_proof_verify_non_adjacent_leaves() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    // Prove non-adjacent leaves (0 and 5)
    let pos_list = vec![0, 5];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();
    let leaves_to_verify: Vec<FE> = pos_list.iter().map(|&i| values[i]).collect();

    assert!(batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        &pos_list,
        &leaves_to_verify,
        8
    ));
}

#[test]
fn batch_proof_verify_single_leaf() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    // Prove single leaf
    let pos_list = vec![3];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();
    let leaves_to_verify: Vec<FE> = pos_list.iter().map(|&i| values[i]).collect();

    assert!(batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        &pos_list,
        &leaves_to_verify,
        8
    ));
}

#[test]
fn batch_proof_verify_many_leaves() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    // Prove many leaves
    let pos_list: Vec<usize> = (0..=9).collect();
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();
    let leaves_to_verify: Vec<FE> = pos_list.iter().map(|&i| values[i]).collect();

    assert!(batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        &pos_list,
        &leaves_to_verify,
        16
    ));
}

#[test]
fn batch_proof_verify_fails_with_wrong_root() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let pos_list = vec![0, 1];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();
    let leaves_to_verify: Vec<FE> = pos_list.iter().map(|&i| values[i]).collect();

    let wrong_root = FE::new(999);
    assert!(!batch_proof.verify::<TestBackend<U64PF>>(
        &wrong_root,
        &pos_list,
        &leaves_to_verify,
        8
    ));
}

#[test]
fn batch_proof_verify_fails_with_wrong_leaves_values() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let pos_list = vec![0, 1];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();
    // Use wrong values
    let wrong_leaves = vec![FE::new(999), FE::new(998)];

    assert!(!batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        &pos_list,
        &wrong_leaves,
        8
    ));
}

#[test]
fn batch_proof_duplicate_positions_are_deduplicated() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    // Position 0 appears twice - should be deduplicated in proof generation
    let pos_list_with_duplicates = vec![0, 0, 1];
    let batch_proof = merkle_tree
        .get_batch_proof(&pos_list_with_duplicates)
        .unwrap();

    // Compare with clean version (no duplicates)
    let pos_list_clean = vec![0, 1];
    let batch_proof_clean = merkle_tree.get_batch_proof(&pos_list_clean).unwrap();

    // Both proofs should have the same path since duplicates are deduplicated
    assert_eq!(batch_proof.path, batch_proof_clean.path);
}

#[test]
fn batch_proof_duplicate_positions_with_conflicting_values_fails() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let pos_list = vec![0, 1];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();

    // Duplicate position with different values should fail
    let pos_list_duplicated = vec![0, 0, 1];
    let values_conflicting = vec![FE::new(999), values[0], values[1]];

    let result = batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        &pos_list_duplicated,
        &values_conflicting,
        8,
    );
    assert!(!result, "Conflicting values for same position should fail");
}

#[test]
fn batch_proof_duplicate_positions_with_same_values_passes() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let pos_list = vec![0, 1];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();

    // Duplicate position with same value should pass (deduplicated)
    let pos_list_duplicated = vec![0, 0, 1];
    let values_same = vec![values[0], values[0], values[1]];

    let result = batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        &pos_list_duplicated,
        &values_same,
        8,
    );
    assert!(result, "Same values for duplicate position should pass");
}

#[test]
fn batch_proof_all_leaves_has_empty_path() {
    const MODULUS: u64 = 512;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    // Prove all leaves - no extra nodes needed since all parents are computable
    let pos_list: Vec<usize> = (0..16).collect();
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();

    // When proving all leaves, proof path should be empty
    // because every sibling is already known (it's another leaf we're proving)
    assert_eq!(
        batch_proof.path.len(),
        0,
        "Proving all leaves should require no additional proof nodes"
    );

    // Verification should still work
    let leaves_to_verify: Vec<FE> = pos_list.iter().map(|&i| values[i]).collect();
    assert!(batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        &pos_list,
        &leaves_to_verify,
        16
    ));
}

#[test]
fn batch_proof_verify_sparse_leaves_across_tree() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let pos_list = &[1, 8, 9, 15];
    let batch_proof = merkle_tree.get_batch_proof(pos_list).unwrap();

    // Verify the proof validates correctly
    let leaves_to_verify: Vec<FE> = pos_list.iter().map(|&i| values[i]).collect();
    assert!(batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        pos_list,
        &leaves_to_verify,
        16
    ));
}
