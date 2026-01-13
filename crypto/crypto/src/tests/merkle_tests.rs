use core::marker::PhantomData;

use math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField, traits::IsField};

use crate::merkle_tree::{merkle::MerkleTree, proof::BatchProof, traits::IsMerkleTreeBackend};

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
    assert_eq!(merkle_tree.get_batch_proof(&[0, 5]).path.len(), 4);
}
#[test]
fn batch_proof_is_expected() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let batch_proof = merkle_tree.get_batch_proof(&[0, 1]);
    let expected_batch_proof = BatchProof {
        path: vec![FE::new(14), FE::new(52)],
    };
    assert_eq!(batch_proof.path, expected_batch_proof.path);
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
    assert_eq!(batch_proof.path.len(), 2);
}

#[test]
fn batch_proof_verify_adjacent_leaves() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    // Prove adjacent leaves (0 and 1 are siblings)
    let pos_list = vec![0, 1];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list);
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
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    // Prove non-adjacent leaves (0 and 5)
    let pos_list = vec![0, 5];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list);
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
    let batch_proof = merkle_tree.get_batch_proof(&pos_list);
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
    let batch_proof = merkle_tree.get_batch_proof(&pos_list);
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
    let batch_proof = merkle_tree.get_batch_proof(&pos_list);
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
fn batch_proof_verify_fails_with_wrong_value() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

    let pos_list = vec![0, 1];
    let batch_proof = merkle_tree.get_batch_proof(&pos_list);
    // Use wrong values
    let wrong_leaves = vec![FE::new(999), FE::new(998)];

    assert!(!batch_proof.verify::<TestBackend<U64PF>>(
        &merkle_tree.root,
        &pos_list,
        &wrong_leaves,
        8
    ));
}
