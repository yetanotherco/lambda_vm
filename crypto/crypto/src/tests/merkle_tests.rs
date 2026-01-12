use core::marker::PhantomData;

use math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField, traits::IsField};

use crate::merkle_tree::{merkle::MerkleTree, traits::IsMerkleTreeBackend};

pub type TestMerkleTree<F> = MerkleTree<FieldElement<F>>;

#[derive(Debug, Clone)]
/// This hasher is for testing purposes
/// It adds the fields
/// Under no circumstance it can be used in production
pub struct TestBackend<F> {
    phantom: PhantomData<F>,
}

impl<F: IsField> Default for TestBackend<F> {
    fn default() -> Self {
        Self {
            phantom: Default::default(),
        }
    }
}

impl<F: IsField> IsMerkleTreeBackend for TestBackend<F>
where
    FieldElement<F>: Sync + Send,
{
    type Node = FieldElement<F>;
    type Data = FieldElement<F>;

    fn hash_data(input: &Self::Data) -> Self::Node {
        input + input
    }

    fn hash_new_parent(left: &Self::Node, right: &Self::Node) -> Self::Node {
        left + right
    }
}

const MODULUS: u64 = 13;
type U64PF = U64PrimeField<MODULUS>;
type FE = FieldElement<U64PF>;

#[test]
fn build_merkle_tree_from_a_power_of_two_list_of_values() {
    let values: Vec<FE> = (1..5).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    assert_eq!(merkle_tree.root, FE::new(7)); // Adjusted expected value
}

#[test]
// expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
fn build_merkle_tree_from_an_odd_set_of_leaves() {
    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..6).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    assert_eq!(merkle_tree.root, FE::new(8)); // Adjusted expected value
}

#[test]
fn build_merkle_tree_from_a_single_value() {
    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = vec![FE::new(1)]; // Single element
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    assert_eq!(merkle_tree.root, FE::new(2)); // Adjusted expected value
}

#[test]
fn build_empty_tree_should_not_panic() {
    assert!(MerkleTree::<TestBackend<U64PF>>::build(&[]).is_none());
}
#[test]
fn batch_proof_len_is_expected() {
    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    assert_eq!(merkle_tree.get_batch_proof(&[0, 5]).len(), 4);
}
#[test]
fn batch_proof_is_expected() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let batch_proof = merkle_tree.get_batch_proof(&[0, 1]);
    assert_eq!(batch_proof, vec![FE::new(14), FE::new(52)]);
}

#[test]
fn batch_proof_len_is_expected_for_long_pos_list() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let pos_list = (0..=9).collect::<Vec<usize>>();
    let batch_proof = merkle_tree.get_batch_proof(&pos_list);
    assert_eq!(batch_proof.len(), 2);
}
