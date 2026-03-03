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

    const ARITY: usize = 4;

    fn hash_data(input: &Self::Data) -> Self::Node {
        input + input
    }

    fn hash_new_parent(children: &[Self::Node]) -> Self::Node {
        children.iter().fold(FieldElement::<F>::zero(), |acc, x| acc + x)
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
// 5 values [1..5] -> hashed [2,4,6,8,10] -> padded to 16 leaves (power of 4)
// Level 1: [2+4+6+8=20→7, 10*4=40→1, 10*4=40→1, 10*4=40→1]
// Root: 7+1+1+1=10
fn build_merkle_tree_from_an_odd_set_of_leaves() {
    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..6).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    assert_eq!(merkle_tree.root, FE::new(10));
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
    // 8 values -> padded to 16 leaves in 4-ary tree
    // positions [0, 5] -> tree indices [5, 10]
    // Auth path: 3 siblings for node 5 + 3 siblings for node 10 + 2 new siblings at level 1 = 8
    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    assert_eq!(merkle_tree.get_batch_proof(&[0, 5]).unwrap().path.len(), 8);
}
#[test]
fn batch_proof_is_expected() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    // 8 values -> padded to 16 leaves. positions [0,1] -> tree indices [5,6]
    // Auth: siblings [7,8] at leaf level, siblings [2,3,4] at level 1
    // Proof in ascending order within each level, bottom to top:
    //   leaf level: [7, 8], level 1: [2, 3, 4]
    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let batch_proof = merkle_tree.get_batch_proof(&[0, 1]).unwrap();
    let expected_batch_proof = BatchProof {
        path: vec![
            FE::new(6),  // nodes[7] = leaf[2] = hash(3) = 6
            FE::new(8),  // nodes[8] = leaf[3] = hash(4) = 8
            FE::new(52), // nodes[2] = sum of leaves[4..7] = 10+12+14+16 = 52
            FE::new(64), // nodes[3] = sum of leaves[8..11] = 16*4 = 64
            FE::new(64), // nodes[4] = sum of leaves[12..15] = 16*4 = 64
        ],
    };
    assert_eq!(batch_proof.path, expected_batch_proof.path);
}

#[test]
fn batch_proof_larger_path_contains_expected_elements() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let batch_proof = merkle_tree.get_batch_proof(&[1, 8, 9, 15]).unwrap();

    // 4-ary tree with 16 leaves (hashed: [2,4,...,32])
    // positions [1,8,9,15] -> tree indices [6,13,14,20]
    // Auth path in ascending order within each level, bottom to top:
    let expected_batch_proof = BatchProof {
        path: vec![
            FE::new(2),  // nodes[5]  = leaf[0]  = hash(1)  = 2
            FE::new(6),  // nodes[7]  = leaf[2]  = hash(3)  = 6
            FE::new(8),  // nodes[8]  = leaf[3]  = hash(4)  = 8
            FE::new(22), // nodes[15] = leaf[10] = hash(11) = 22
            FE::new(24), // nodes[16] = leaf[11] = hash(12) = 24
            FE::new(26), // nodes[17] = leaf[12] = hash(13) = 26
            FE::new(28), // nodes[18] = leaf[13] = hash(14) = 28
            FE::new(30), // nodes[19] = leaf[14] = hash(15) = 30
            FE::new(52), // nodes[2]  = level 1 = 10+12+14+16 = 52
        ],
    };
    assert_eq!(batch_proof.path, expected_batch_proof.path);
}

#[test]
fn batch_proof_len_is_expected_for_long_pos_list() {
    const MODULUS: u64 = 70;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    // 16 leaves, positions [0..=9] -> 10 of 16 leaves known
    // Auth: siblings [15,16] at leaf level + sibling [4] at level 1 = 3 nodes
    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let pos_list = (0..=9).collect::<Vec<usize>>();
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();
    assert_eq!(batch_proof.path.len(), 3);
}
