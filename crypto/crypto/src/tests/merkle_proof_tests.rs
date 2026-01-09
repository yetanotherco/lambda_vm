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
// expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
fn create_a_proof_over_value_that_belongs_to_a_given_merkle_tree_when_given_the_leaf_position() {
    let values: Vec<FE> = (1..6).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let proof = &merkle_tree.get_proof_by_pos(1).unwrap();
    assert_merkle_path(&proof.merkle_path, &[FE::new(2), FE::new(1), FE::new(1)]);
    assert!(proof.verify::<TestBackend<U64PF>>(&merkle_tree.root, 1, &FE::new(2)));
}

#[test]
fn create_a_merkle_tree_with_10000_elements_and_verify_that_an_element_is_part_of_it() {
    let values: Vec<Ecgfp5FE> = (1..10000).map(Ecgfp5FE::new).collect();
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
    type U64PF = U64PrimeField<MODULUS>;
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
