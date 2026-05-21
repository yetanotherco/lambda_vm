use alloc::vec::Vec;
use math::field::{element::FieldElement, test_fields::u64_test_field::U64Field};

use crate::merkle_tree::merkle::MerkleTree;
use crate::merkle_tree::proof::BatchProof;
use crate::tests::merkle_tests::TestBackend;

/// Small field useful for starks, sometimes called min i goldilocks
/// Used in miden and winterfell
pub type Ecgfp5 = U64Field<0xFFFF_FFFF_0000_0001_u64>;
pub type Ecgfp5FE = FieldElement<Ecgfp5>;
pub type TestMerkleTreeEcgfp = MerkleTree<TestBackend<Ecgfp5>>;

const MODULUS: u64 = 13;
type U64PF = U64Field<MODULUS>;
type FE = FieldElement<U64PF>;

#[test]
// expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
fn create_a_proof_over_value_that_belongs_to_a_given_merkle_tree_when_given_the_leaf_position() {
    // 8 leaves (power of two) — the same set the old non-power-of-two padding produced.
    let values: Vec<FE> = [1, 2, 3, 4, 5, 5, 5, 5].into_iter().map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let proof = &merkle_tree.get_proof_by_pos(1).unwrap();
    assert_merkle_path(&proof.merkle_path, &[FE::new(2), FE::new(1), FE::new(1)]);
    assert!(proof.verify::<TestBackend<U64PF>>(&merkle_tree.root, 1, &FE::new(2)));
}

#[test]
fn create_a_merkle_tree_with_16384_elements_and_verify_that_an_element_is_part_of_it() {
    let values: Vec<Ecgfp5FE> = (1..=16384).map(Ecgfp5FE::new).collect();
    let merkle_tree = TestMerkleTreeEcgfp::build(&values).unwrap();
    let proof = merkle_tree.get_proof_by_pos(9349).unwrap();
    assert!(proof.verify::<TestBackend<Ecgfp5>>(&merkle_tree.root, 9349, &Ecgfp5FE::new(9350)));
}

fn assert_merkle_path(values: &[FE], expected_values: &[FE]) {
    for (node, expected_node) in values.iter().zip(expected_values) {
        assert_eq!(node, expected_node);
    }
}

#[test]
fn verify_merkle_proof_for_single_value() {
    const MODULUS: u64 = 13;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = vec![FE::new(1)]; // Single element
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    // Update the expected root value based on the actual logic of TestBackend
    // For example, in this case hashing a single `1` results in `2`
    let expected_root = FE::new(2); // Assuming hashing a `1`s results in `2`
    assert_eq!(
        merkle_tree.root, expected_root,
        "The root of the Merkle tree does not match the expected value."
    );

    // Verify the proof for the single element
    let proof = merkle_tree.get_proof_by_pos(0).unwrap();
    assert!(
        proof.verify::<TestBackend<U64PF>>(&merkle_tree.root, 0, &values[0]),
        "The proof verification failed for the element at position 0."
    );
}

#[test]
fn batch_proof_verify_adjacent_leaves() {
    const MODULUS: u64 = 70;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    // Prove adjacent leaves (0 and 1 are siblings)
    let pos_list = vec![0, 1];
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
fn batch_proof_verify_non_adjacent_leaves() {
    const MODULUS: u64 = 70;
    type U64PF = U64Field<MODULUS>;
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
    type U64PF = U64Field<MODULUS>;
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
    type U64PF = U64Field<MODULUS>;
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
    type U64PF = U64Field<MODULUS>;
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
    type U64PF = U64Field<MODULUS>;
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
    type U64PF = U64Field<MODULUS>;
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
    type U64PF = U64Field<MODULUS>;
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
    type U64PF = U64Field<MODULUS>;
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
    type U64PF = U64Field<MODULUS>;
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
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let batch_proof = merkle_tree.get_batch_proof(&[1, 8, 9, 15]).unwrap();

    // Verify the proof structure is as expected
    let expected_batch_proof = BatchProof {
        path: vec![
            FE::new(30), // index 29 - leaf level sibling
            FE::new(2),  // index 15 - leaf level sibling
            FE::new(54), // index 13 - internal node
            FE::new(46), // index 12 - internal node
            FE::new(14), // index 8  - internal node
            FE::new(52), // index 4  - internal node (closest to root)
        ],
    };
    assert_eq!(batch_proof.path, expected_batch_proof.path);

    // Verify the proof validates correctly
    let pos_list = &[1, 8, 9, 15];
    let leaves_to_verify: Vec<FE> = pos_list.iter().map(|&i| values[i]).collect();
    assert!(batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        pos_list,
        &leaves_to_verify,
        16
    ));
}
