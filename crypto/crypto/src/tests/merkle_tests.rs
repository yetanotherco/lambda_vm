use core::marker::PhantomData;

use math::field::{element::FieldElement, test_fields::u64_test_field::U64Field, traits::IsField};

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
type U64PF = U64Field<MODULUS>;
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
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..6).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    assert_eq!(merkle_tree.root, FE::new(8)); // Adjusted expected value
}

#[test]
fn build_merkle_tree_from_a_single_value() {
    const MODULUS: u64 = 13;
    type U64PF = U64Field<MODULUS>;
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
    assert_eq!(merkle_tree.get_batch_proof(&[0, 5]).unwrap().path.len(), 4);
}
#[test]
fn batch_proof_is_expected() {
    const MODULUS: u64 = 70;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=8).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let batch_proof = merkle_tree.get_batch_proof(&[0, 1]).unwrap();
    let expected_batch_proof = BatchProof {
        path: vec![FE::new(14), FE::new(52)],
    };
    assert_eq!(batch_proof.path, expected_batch_proof.path);
}

#[test]
fn batch_proof_larger_path_contains_expected_elements() {
    const MODULUS: u64 = 70;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let batch_proof = merkle_tree.get_batch_proof(&[1, 8, 9, 15]).unwrap();

    // The proof stores nodes in descending order by tree index
    // - Higher indices (closer to leaves) come first
    // - Lower indices (closer to root) come last
    // - In same level, nodes are ordered form right (larger indices) to left (smaller indices).
    let expected_batch_proof = BatchProof {
        path: vec![
            FE::new(30), // index 29 - leaf level sibling
            FE::new(2),  // index 15 - leaf level sibling
            FE::new(54), // index 13 - internal node
            FE::new(46), // index 12 - internal node
            FE::new(14), // index 8  - internal node
            FE::new(52), // index 4  - internal node (closest to root)
        ],
    };
    assert_eq!(batch_proof.path, expected_batch_proof.path);
}

#[test]
fn batch_proof_len_is_expected_for_long_pos_list() {
    const MODULUS: u64 = 70;
    type U64PF = U64Field<MODULUS>;
    type FE = FieldElement<U64PF>;

    let values: Vec<FE> = (1..=16).map(FE::new).collect();
    let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
    let pos_list = (0..=9).collect::<Vec<usize>>();
    let batch_proof = merkle_tree.get_batch_proof(&pos_list).unwrap();
    assert_eq!(batch_proof.path.len(), 2);
}

#[cfg(all(feature = "serde", feature = "disk-spill"))]
mod disk_spill_serde_tests {
    use crate::merkle_tree::backends::field_element::FieldElementBackend;
    use crate::merkle_tree::merkle::MerkleTree;
    use math::field::{element::FieldElement, goldilocks::GoldilocksField};
    use sha3::Keccak256;

    type F = GoldilocksField;
    type FE = FieldElement<F>;
    type Backend = FieldElementBackend<F, Keccak256, 32>;

    /// Serializing a spilled MerkleTree must produce identical bytes to
    /// serializing the same tree before spilling, and round-trip back to an
    /// equal tree.
    #[test]
    fn test_serialize_spilled_merkle_tree_matches_unspilled() {
        let values: Vec<FE> = (1..17).map(FE::from).collect();
        let unspilled = MerkleTree::<Backend>::build(&values).expect("build merkle tree");
        let unspilled_bytes = bincode::serialize(&unspilled).expect("serialize unspilled");

        let mut spilled = MerkleTree::<Backend>::build(&values).expect("build merkle tree");
        spilled.spill_nodes_to_disk().expect("spill_nodes_to_disk");
        let spilled_bytes = bincode::serialize(&spilled).expect("serialize spilled");

        assert_eq!(
            spilled_bytes, unspilled_bytes,
            "spilled and unspilled trees must serialize to identical bytes"
        );

        let restored: MerkleTree<Backend> =
            bincode::deserialize(&spilled_bytes).expect("deserialize spilled bytes");
        assert!(!restored.has_mmap_backing());
        assert_eq!(restored.root, unspilled.root);
    }
}
