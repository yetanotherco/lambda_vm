use alloc::vec::Vec;
use math::field::{element::FieldElement, test_fields::u64_test_field::U64Field};

use crate::merkle_tree::merkle::MerkleTree;
use crate::tests::merkle_tests::TestBackend;

pub type Ecgfp5 = U64Field<0xFFFF_FFFF_0000_0001_u64>;
pub type Ecgfp5FE = FieldElement<Ecgfp5>;
pub type TestMerkleTreeEcgfp = MerkleTree<TestBackend<Ecgfp5>>;

const MODULUS: u64 = 13;
type U64PF = U64Field<MODULUS>;
type FE = FieldElement<U64PF>;

#[test]
fn create_a_proof_over_value_that_belongs_to_a_given_merkle_tree() {
    // With 4-ary, 4 leaves form a single-level tree
    let values: Vec<FE> = (1..5).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let proof = &merkle_tree.get_proof_by_pos(1).unwrap();
    // merkle_path has 1 level with 3 siblings
    assert_eq!(proof.merkle_path.len(), 1);
    assert_eq!(proof.merkle_path[0].len(), 3);
    assert!(proof.verify::<TestBackend<U64PF>>(&merkle_tree.root, 1, &FE::new(2)));
}

#[test]
fn create_a_merkle_tree_with_many_elements_and_verify() {
    // Use power-of-4 count for clean tree
    let values: Vec<Ecgfp5FE> = (1..=256).map(Ecgfp5FE::new).collect();
    let merkle_tree = TestMerkleTreeEcgfp::build(&values).unwrap();
    let proof = merkle_tree.get_proof_by_pos(100).unwrap();
    assert!(proof.verify::<TestBackend<Ecgfp5>>(&merkle_tree.root, 100, &Ecgfp5FE::new(101)));
}

#[test]
fn verify_merkle_proof_for_single_value() {
    let values: Vec<FE> = vec![FE::new(1)];
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let expected_root = FE::new(2);
    assert_eq!(merkle_tree.root, expected_root);

    let proof = merkle_tree.get_proof_by_pos(0).unwrap();
    assert!(proof.verify::<TestBackend<U64PF>>(&merkle_tree.root, 0, &values[0]));
}

#[test]
fn batch_proof_verify_adjacent_leaves() {
    const MODULUS: u64 = 1000;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let pos_list = vec![0, 1];
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
fn batch_proof_verify_non_adjacent_leaves() {
    const MODULUS: u64 = 1000;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let pos_list = vec![0, 5];
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
fn batch_proof_verify_single_leaf() {
    const MODULUS: u64 = 1000;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let pos_list = vec![3];
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
fn batch_proof_verify_many_leaves() {
    const MODULUS: u64 = 1000;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

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
    const MODULUS: u64 = 1000;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let pos_list = vec![0, 1];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();
    let leaves_to_verify: Vec<FE> = pos_list.iter().map(|&i| values[i]).collect();

    let wrong_root = FE::new(999);
    assert!(!batch_proof.verify::<TestBackend<U64PF>>(
        &wrong_root,
        &pos_list,
        &leaves_to_verify,
        16
    ));
}

#[test]
fn batch_proof_verify_fails_with_wrong_leaves_values() {
    const MODULUS: u64 = 1000;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let pos_list = vec![0, 1];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();
    let wrong_leaves = vec![FE::new(999), FE::new(998)];

    assert!(!batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        &pos_list,
        &wrong_leaves,
        16
    ));
}

#[test]
fn batch_proof_duplicate_positions_are_deduplicated() {
    const MODULUS: u64 = 1000;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let pos_list_with_duplicates = vec![0, 0, 1];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list_with_duplicates).unwrap();

    let pos_list_clean = vec![0, 1];
    let batch_proof_clean = merkle_tree.get_batch_proof(&pos_list_clean).unwrap();

    assert_eq!(batch_proof.path, batch_proof_clean.path);
}

#[test]
fn batch_proof_duplicate_positions_with_conflicting_values_fails() {
    const MODULUS: u64 = 1000;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let pos_list = vec![0, 1];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();

    let pos_list_duplicated = vec![0, 0, 1];
    let values_conflicting = vec![FE::new(999), values[0], values[1]];

    let result = batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        &pos_list_duplicated,
        &values_conflicting,
        16,
    );
    assert!(!result);
}

#[test]
fn batch_proof_duplicate_positions_with_same_values_passes() {
    const MODULUS: u64 = 1000;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let pos_list = vec![0, 1];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();

    let pos_list_duplicated = vec![0, 0, 1];
    let values_same = vec![values[0], values[0], values[1]];

    let result = batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        &pos_list_duplicated,
        &values_same,
        16,
    );
    assert!(result);
}

#[test]
fn batch_proof_all_leaves_has_empty_path() {
    const MODULUS: u64 = 1000;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let pos_list: Vec<usize> = (0..16).collect();
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();

    assert_eq!(batch_proof.path.len(), 0);

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
    const MODULUS: u64 = 1000;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let pos_list = &[1, 8, 9, 15];
    let batch_proof = merkle_tree.get_batch_proof(pos_list).unwrap();

    let leaves_to_verify: Vec<FE> = pos_list.iter().map(|&i| values[i]).collect();
    assert!(batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        pos_list,
        &leaves_to_verify,
        16
    ));
}
