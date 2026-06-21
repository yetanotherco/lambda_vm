use alloc::vec::Vec;
use math::field::{element::FieldElement, test_fields::u64_test_field::U64Field};

use crate::merkle_tree::{
    traits::IsMerkleTreeBackend,
    utils::{build, complete_until_power_of_arity},
};
use crate::tests::merkle_tests::TestBackend;

const MODULUS: u64 = 13;
type U64PF = U64Field<MODULUS>;
type FE = FieldElement<U64PF>;

#[test]
fn build_merkle_tree_one_element_must_succeed() {
    let mut nodes = [FE::zero()];

    build::<TestBackend<U64PF>>(&mut nodes, 1);
}

#[test]
// expected |2|4|6|8|
fn hash_leaves_from_a_list_of_field_elemnts() {
    let values: Vec<FE> = (1..5).map(FE::new).collect();
    let hashed_leaves = TestBackend::hash_leaves(&values);
    let list_of_nodes = &[FE::new(2), FE::new(4), FE::new(6), FE::new(8)];
    for (leaf, expected_leaf) in hashed_leaves.iter().zip(list_of_nodes) {
        assert_eq!(leaf, expected_leaf);
    }
}

#[test]
// expected |1|2|3|4|5|5|5|5|
fn complete_the_length_of_a_list_of_fields_elements_to_be_a_power_of_two() {
    let values: Vec<FE> = (1..6).map(FE::new).collect();
    let hashed_leaves = complete_until_power_of_arity(values, 2);

    let mut expected_leaves = (1..6).map(FE::new).collect::<Vec<FE>>();
    expected_leaves.extend([FE::new(5); 3]);

    for (leaf, expected_leaves) in hashed_leaves.iter().zip(expected_leaves) {
        assert_eq!(*leaf, expected_leaves);
    }
}

#[test]
// expected |2|2|
fn complete_the_length_of_one_field_element_to_be_a_power_of_two() {
    let values: Vec<FE> = vec![FE::new(2)];
    let hashed_leaves = complete_until_power_of_arity(values, 2);

    let mut expected_leaves = vec![FE::new(2)];
    expected_leaves.extend([FE::new(2)]);
    assert_eq!(hashed_leaves.len(), 1);
    assert_eq!(hashed_leaves[0], expected_leaves[0]);
}

const ROOT: usize = 0;

#[test]
// expected |10|10|13|3|7|11|2|1|2|3|4|5|6|7|8|
fn complete_a_merkle_tree_from_a_set_of_leaves() {
    let leaves: Vec<FE> = (1..9).map(FE::new).collect();
    let leaves_len = leaves.len();

    let mut nodes = vec![FE::zero(); leaves.len() - 1];
    nodes.extend(leaves);

    build::<TestBackend<U64PF>>(&mut nodes, leaves_len);
    assert_eq!(nodes[ROOT], FE::new(10));
}
