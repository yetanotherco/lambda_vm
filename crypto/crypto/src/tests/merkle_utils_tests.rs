use alloc::vec::Vec;
use math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

use crate::merkle_tree::{
    traits::IsMerkleTreeBackend,
    utils::{build, complete_until_power_of_arity},
};
use crate::tests::merkle_tests::TestBackend;

const MODULUS: u64 = 13;
type U64PF = U64PrimeField<MODULUS>;
type FE = FieldElement<U64PF>;

#[test]
fn build_merkle_tree_one_element_must_succeed() {
    // 4-ary tree with 4 leaves needs 1 internal + 4 leaf = 5 nodes.
    // Leaf value is FE::new(1); pad remaining 3 leaves with same value.
    let leaf = FE::new(1);
    let mut nodes = vec![FE::zero(); 1]; // 1 internal node
    nodes.extend([leaf, leaf, leaf, leaf]); // 4 leaves

    build::<TestBackend<U64PF>>(&mut nodes, 4);
    // root = 1 + 1 + 1 + 1 = 4
    assert_eq!(nodes[0], FE::new(4));
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
// 5 elements padded to power of 4 = 16 (next power of 4 after 5)
// expected |1|2|3|4|5|5|5|5|5|5|5|5|5|5|5|5|
fn complete_the_length_of_a_list_of_fields_elements_to_be_a_power_of_four() {
    let values: Vec<FE> = (1..6).map(FE::new).collect();
    let padded = complete_until_power_of_arity(values, 4);

    let mut expected = (1..6).map(FE::new).collect::<Vec<FE>>();
    expected.extend([FE::new(5); 11]); // 5 + 11 = 16

    assert_eq!(padded.len(), 16);
    for (leaf, expected_leaf) in padded.iter().zip(expected) {
        assert_eq!(*leaf, expected_leaf);
    }
}

#[test]
// 1 is already a power of 4 (4^0 = 1), so no padding
fn complete_the_length_of_one_field_element_to_be_a_power_of_four() {
    let values: Vec<FE> = vec![FE::new(2)];
    let padded = complete_until_power_of_arity(values, 4);

    assert_eq!(padded.len(), 1);
    assert_eq!(padded[0], FE::new(2));
}

const ROOT: usize = 0;

#[test]
// 4-ary tree with 4 leaves [1, 2, 3, 4]
// Internal nodes = (4 - 1) / 3 = 1 (just root)
// root = 1 + 2 + 3 + 4 = 10
fn complete_a_merkle_tree_from_a_set_of_four_leaves() {
    let leaves: Vec<FE> = (1..5).map(FE::new).collect();
    let leaves_len = leaves.len(); // 4, which is a power of 4

    let internal_nodes = (leaves_len - 1) / 3; // 1
    let mut nodes = vec![FE::zero(); internal_nodes];
    nodes.extend(leaves);

    build::<TestBackend<U64PF>>(&mut nodes, leaves_len);
    assert_eq!(nodes[ROOT], FE::new(10));
}

#[test]
// 4-ary tree with 16 leaves [1..16] (mod 13)
// Leaf level: [1,2,3,4,5,6,7,8,9,10,11,12,0,1,2,3]
// Level 1:   [1+2+3+4=10, 5+6+7+8=26→0, 9+10+11+12=42→3, 0+1+2+3=6]
// Root:      10+0+3+6=19→6
fn complete_a_merkle_tree_from_a_set_of_sixteen_leaves() {
    let leaves: Vec<FE> = (1..17).map(FE::new).collect();
    let leaves_len = leaves.len(); // 16, which is a power of 4

    let internal_nodes = (leaves_len - 1) / 3; // 5
    let mut nodes = vec![FE::zero(); internal_nodes];
    nodes.extend(leaves);

    build::<TestBackend<U64PF>>(&mut nodes, leaves_len);
    assert_eq!(nodes[ROOT], FE::new(6));
}
