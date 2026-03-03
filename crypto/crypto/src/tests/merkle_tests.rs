use core::marker::PhantomData;

use math::field::{element::FieldElement, test_fields::u64_test_field::U64Field, traits::IsField};

use crate::merkle_tree::{merkle::MerkleTree, traits::IsMerkleTreeBackend};

pub type TestMerkleTree<F> = MerkleTree<FieldElement<F>>;

#[derive(Debug, Clone)]
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

    const ARITY: usize = 4;

    fn hash_data(input: &Self::Data) -> Self::Node {
        input + input
    }

    fn hash_new_parent(children: &[Self::Node]) -> Self::Node {
        children
            .iter()
            .fold(FieldElement::<F>::zero(), |acc, x| acc + x)
    }
}

const MODULUS: u64 = 13;
type U64PF = U64Field<MODULUS>;
type FE = FieldElement<U64PF>;

#[test]
fn build_merkle_tree_from_a_power_of_four_list_of_values() {
    let values: Vec<FE> = (1..5).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    // 4 leaves -> hash_data doubles each: 2,4,6,8 -> hash_new_parent(2,4,6,8)=20 mod 13=7
    assert_eq!(merkle_tree.root, FE::new(7));
}

#[test]
fn build_merkle_tree_from_an_odd_set_of_leaves() {
    // 5 leaves -> padded to 16 for 4-ary
    let values: Vec<FE> = (1..6).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    // Just verify it builds and root is deterministic
    let merkle_tree2 = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    assert_eq!(merkle_tree.root, merkle_tree2.root);
}

#[test]
fn build_merkle_tree_from_a_single_value() {
    let values: Vec<FE> = vec![FE::new(1)];
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    assert_eq!(merkle_tree.root, FE::new(2)); // hash_data(1) = 2
}

#[test]
fn build_empty_tree_should_not_panic() {
    assert!(MerkleTree::<TestBackend<U64PF>>::build(&[]).is_none());
}

#[test]
fn batch_proof_for_four_ary_tree() {
    const MODULUS: u64 = 1000;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let pos_list: Vec<usize> = vec![0, 5, 10, 15];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();
    let leaves_to_verify: Vec<FE> = pos_list.iter().map(|&i| values[i]).collect();

    assert!(batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        &pos_list,
        &leaves_to_verify,
        16
    ));
}
