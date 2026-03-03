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
fn hash_leaves_from_a_list_of_field_elemnts() {
    let values: Vec<FE> = (1..5).map(FE::new).collect();
    let hashed_leaves = TestBackend::hash_leaves(&values);
    let list_of_nodes = &[FE::new(2), FE::new(4), FE::new(6), FE::new(8)];
    for (leaf, expected_leaf) in hashed_leaves.iter().zip(list_of_nodes) {
        assert_eq!(leaf, expected_leaf);
    }
}

#[test]
fn complete_the_length_of_a_list_of_fields_elements_to_be_a_power_of_four() {
    let values: Vec<FE> = (1..6).map(FE::new).collect();
    let padded = complete_until_power_of_arity(values, 4);
    // 5 elements -> next power of 4 is 16
    assert_eq!(padded.len(), 16);
    // First 5 elements unchanged
    for (i, val) in padded.iter().enumerate().take(5) {
        assert_eq!(*val, FE::new((i + 1) as u64));
    }
    // Remaining padded with last element (5)
    for val in &padded[5..16] {
        assert_eq!(*val, FE::new(5));
    }
}

#[test]
fn complete_the_length_of_one_field_element_stays_one() {
    let values: Vec<FE> = vec![FE::new(2)];
    let padded = complete_until_power_of_arity(values, 4);
    assert_eq!(padded.len(), 1);
    assert_eq!(padded[0], FE::new(2));
}

const ROOT: usize = 0;

#[test]
fn complete_a_merkle_tree_from_16_leaves() {
    const MODULUS: u64 = 1000;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let leaves: Vec<FE> = (1..=16).map(FE::new).collect();
    let leaves_len = leaves.len();

    // internal nodes = (16-1)/3 = 5
    let mut nodes = vec![FE::zero(); 5];
    nodes.extend(leaves);

    build::<TestBackend<U64PF>>(&mut nodes, leaves_len);
    // Just verify the root was computed
    assert_ne!(nodes[ROOT], FE::zero());
}
