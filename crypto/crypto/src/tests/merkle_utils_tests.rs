use alloc::vec::Vec;
use math::field::{element::FieldElement, test_fields::u64_test_field::U64Field};

use crate::merkle_tree::{traits::IsMerkleTreeBackend, utils::build};
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
