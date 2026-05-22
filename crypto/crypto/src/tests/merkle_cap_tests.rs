//! Tests for Merkle caps: committing the top `2^cap_height` nodes instead of a
//! single root, so authentication paths stop `cap_height` levels early.

use alloc::vec::Vec;
use math::field::{element::FieldElement, goldilocks::GoldilocksField};

use crate::merkle_tree::backends::types::Keccak256Backend;
use crate::merkle_tree::merkle::MerkleTree;

type F = GoldilocksField;
type FE = FieldElement<F>;
type Backend = Keccak256Backend<F>;

// A 16-leaf tree has depth 4, so cap heights 0..=4 are all valid.
const DEPTH: usize = 4;

fn leaves() -> Vec<FE> {
    (1..=16u64).map(FE::from).collect()
}

fn tree() -> MerkleTree<Backend> {
    MerkleTree::<Backend>::build(&leaves()).unwrap()
}

#[test]
fn cap_height_zero_is_the_root() {
    let t = tree();
    let cap = t.cap(0);
    assert_eq!(cap.len(), 1);
    assert_eq!(cap[0], t.root);
}

#[test]
fn cap_has_two_pow_height_nodes() {
    let t = tree();
    for c in 0..=DEPTH {
        assert_eq!(t.cap(c).len(), 1 << c, "cap height {c}");
    }
}

#[test]
fn cap_height_is_clamped_to_tree_depth() {
    let t = tree();
    // A cap taller than the tree clamps to the leaf level.
    assert_eq!(t.cap(99).len(), 16);
}

#[test]
fn capped_path_is_shorter_by_cap_height() {
    let t = tree();
    for c in 0..=DEPTH {
        let proof = t.get_proof_by_pos_capped(0, c).unwrap();
        assert_eq!(proof.merkle_path.len(), DEPTH - c, "cap height {c}");
    }
}

#[test]
fn capped_proof_roundtrips_for_every_leaf_and_cap_height() {
    let values = leaves();
    let t = MerkleTree::<Backend>::build(&values).unwrap();
    for c in 0..=DEPTH {
        let cap = t.cap(c);
        for (pos, value) in values.iter().enumerate() {
            let proof = t.get_proof_by_pos_capped(pos, c).unwrap();
            assert!(
                proof.verify_capped::<Backend>(&cap, pos, value),
                "leaf {pos}, cap height {c}",
            );
        }
    }
}

#[test]
fn cap_height_zero_matches_uncapped() {
    let values = leaves();
    let t = MerkleTree::<Backend>::build(&values).unwrap();
    for (pos, value) in values.iter().enumerate() {
        let capped = t.get_proof_by_pos_capped(pos, 0).unwrap();
        let plain = t.get_proof_by_pos(pos).unwrap();
        assert_eq!(capped.merkle_path, plain.merkle_path);
        assert!(capped.verify_capped::<Backend>(&t.cap(0), pos, value));
        assert!(plain.verify::<Backend>(&t.root, pos, value));
    }
}

#[test]
fn capped_proof_rejects_wrong_value() {
    let t = tree();
    let cap = t.cap(2);
    let proof = t.get_proof_by_pos_capped(5, 2).unwrap();
    assert!(!proof.verify_capped::<Backend>(&cap, 5, &FE::from(999u64)));
}

#[test]
fn capped_proof_rejects_wrong_position() {
    let values = leaves();
    let t = MerkleTree::<Backend>::build(&values).unwrap();
    let cap = t.cap(2);
    let proof = t.get_proof_by_pos_capped(5, 2).unwrap();
    assert!(!proof.verify_capped::<Backend>(&cap, 6, &values[5]));
}

#[test]
fn capped_proof_rejects_tampered_cap() {
    let values = leaves();
    let t = MerkleTree::<Backend>::build(&values).unwrap();
    let mut cap = t.cap(2);
    let proof = t.get_proof_by_pos_capped(5, 2).unwrap();
    // The proof for leaf 5 resolves to cap entry `5 >> (DEPTH - 2)`.
    let cap_index = 5 >> (DEPTH - 2);
    cap[cap_index] = [0u8; 32];
    assert!(!proof.verify_capped::<Backend>(&cap, 5, &values[5]));
}

#[test]
fn batch_capped_roundtrips() {
    let values = leaves();
    let t = MerkleTree::<Backend>::build(&values).unwrap();
    let positions = [1usize, 4, 5, 11, 15];
    let leaf_values: Vec<FE> = positions.iter().map(|&p| values[p]).collect();
    for c in 0..=DEPTH {
        let cap = t.cap(c);
        let batch = t.get_batch_proof_capped(&positions, c).unwrap();
        assert!(
            batch.verify_capped::<Backend>(&cap, &positions, &leaf_values, 16),
            "cap height {c}",
        );
    }
}

#[test]
fn batch_capped_zero_matches_uncapped() {
    let values = leaves();
    let t = MerkleTree::<Backend>::build(&values).unwrap();
    let positions = [2usize, 7, 13];
    let leaf_values: Vec<FE> = positions.iter().map(|&p| values[p]).collect();
    let capped = t.get_batch_proof_capped(&positions, 0).unwrap();
    let plain = t.get_batch_proof(&positions).unwrap();
    assert_eq!(capped.path, plain.path);
    assert!(capped.verify_capped::<Backend>(&t.cap(0), &positions, &leaf_values, 16));
    assert!(plain.verify::<Backend>(&t.root, &positions, &leaf_values, 16));
}

#[test]
fn batch_capped_rejects_tampered_value() {
    let values = leaves();
    let t = MerkleTree::<Backend>::build(&values).unwrap();
    let positions = [1usize, 4, 9];
    let mut leaf_values: Vec<FE> = positions.iter().map(|&p| values[p]).collect();
    leaf_values[1] = FE::from(12345u64);
    let batch = t.get_batch_proof_capped(&positions, 2).unwrap();
    assert!(!batch.verify_capped::<Backend>(&t.cap(2), &positions, &leaf_values, 16));
}
