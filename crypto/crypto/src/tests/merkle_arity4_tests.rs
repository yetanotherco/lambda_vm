//! `MerkleTree` at arity 4 against the Poseidon1 host oracle
//! ([`crate::hash::poseidon1_stark::Merkle4`], whose roots and paths ZisK's own
//! code reproduces), and the binary layout's invariants left alone.

use alloc::vec::Vec;

use crate::hash::poseidon1_stark::{self as p1, Merkle4};
use crate::hash::poseidon1_w16::{Digest, Fp};
use crate::merkle_tree::merkle::MerkleTree;
use crate::merkle_tree::proof::verify_merkle_path_from_leaf_hash;
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use crate::merkle_tree::utils::{leaves_len4, level_sizes4};

/// ZisK's leaf hash and 4-ary node over canonical digests.
struct P1Backend;

fn canon(d: &Digest) -> [u64; 4] {
    core::array::from_fn(|i| d[i].canonical())
}

fn digest(n: &[u64; 4]) -> Digest {
    core::array::from_fn(|i| Fp::from(n[i]))
}

impl IsMerkleTreeBackend for P1Backend {
    type Node = [u64; 4];
    type Data = Vec<Fp>;
    const ARITY: usize = 4;

    fn hash_data(leaf: &Vec<Fp>) -> [u64; 4] {
        canon(&p1::linear_hash(leaf))
    }

    fn hash_new_parent(_: &[u64; 4], _: &[u64; 4]) -> [u64; 4] {
        unreachable!("an arity-4 backend has no binary parent")
    }

    fn hash_four(children: &[[u64; 4]; 4]) -> [u64; 4] {
        canon(&crate::hash::poseidon1_w16::compress4(
            &children.map(|c| digest(&c)),
        ))
    }

    fn padding_node() -> [u64; 4] {
        [0; 4]
    }
}

fn rows(n: usize) -> Vec<Vec<Fp>> {
    (0..n)
        .map(|r| {
            (0..5)
                .map(|c| Fp::from((r * 7 + c) as u64 * 0x9E37_79B9))
                .collect()
        })
        .collect()
}

#[test]
fn arity_4_trees_match_the_oracle_root_paths_and_caps() {
    for k in 0..=9 {
        let n = 1usize << k;
        let data = rows(n);
        let tree = MerkleTree::<P1Backend>::build(&data).expect("non-empty");
        let leaves: Vec<Digest> = data.iter().map(|r| p1::linear_hash(r)).collect();
        let oracle = Merkle4::new(&leaves).expect("non-empty");
        assert_eq!(tree.root, canon(&oracle.root()), "{n} leaves");
        assert_eq!(tree.depth(), Some(oracle.depth()), "{n} leaves");
        for pos in [0, n / 3, n - 1] {
            let proof = tree.get_proof_by_pos(pos).expect("in range");
            let want: Vec<[u64; 4]> = oracle
                .path(pos)
                .expect("in range")
                .iter()
                .flat_map(|level| {
                    level
                        .chunks_exact(4)
                        .map(|d| canon(&[d[0], d[1], d[2], d[3]]))
                        .collect::<Vec<_>>()
                })
                .collect();
            assert_eq!(proof.merkle_path, want, "{n} leaves, pos {pos}");
            assert!(proof.verify::<P1Backend>(&tree.root, pos, &data[pos]));
        }
        assert!(tree.get_proof_by_pos(n).is_none());
        // The cap one level down is the oracle's level below the root.
        if let Some(depth) = tree.depth().filter(|&d| d >= 1) {
            let cap = tree.cap(1).expect("depth ≥ 1");
            assert!(
                cap.len() == 4 || (cap.len() == 2 && k % 2 == 1),
                "{n} leaves: cap {}",
                cap.len()
            );
            assert_eq!(tree.cap(0), Some(alloc::vec![tree.root]));
            assert_eq!(tree.cap(depth).map(|c| c.len()), Some(n));
            assert!(tree.cap(depth + 1).is_none());
        }
    }
}

#[test]
fn an_arity_4_tree_round_trips_through_its_node_vector() {
    for k in 0..=8 {
        let tree = MerkleTree::<P1Backend>::build(&rows(1 << k)).expect("non-empty");
        let nodes = tree.nodes().to_vec();
        assert_eq!(nodes.len(), level_sizes4(1 << k).iter().sum::<usize>());
        assert_eq!(leaves_len4(nodes.len()), Some(1 << k));
        let back = MerkleTree::<P1Backend>::from_precomputed_nodes(nodes).expect("a layout");
        assert_eq!(back.root, tree.root);
        assert_eq!(
            back.get_proof_by_pos(0).map(|p| p.merkle_path),
            tree.get_proof_by_pos(0).map(|p| p.merkle_path)
        );
    }
    // A node count no power-of-two tree has.
    assert!(MerkleTree::<P1Backend>::from_precomputed_nodes(alloc::vec![[0; 4]; 4]).is_none());
    assert!(MerkleTree::<P1Backend>::from_precomputed_nodes(alloc::vec![[0; 4]; 7]).is_none());
}

#[test]
fn a_tampered_arity_4_path_is_rejected() {
    let data = rows(32);
    let tree = MerkleTree::<P1Backend>::build(&data).expect("non-empty");
    for pos in [0usize, 13, 31] {
        let path = tree.get_proof_by_pos(pos).expect("in range").merkle_path;
        let leaf = P1Backend::hash_data(&data[pos]);
        assert!(verify_merkle_path_from_leaf_hash::<P1Backend>(
            &path, &tree.root, pos, leaf
        ));
        for i in 0..path.len() {
            let mut bad = path.clone();
            bad[i][i % 4] ^= 1;
            assert!(
                !verify_merkle_path_from_leaf_hash::<P1Backend>(&bad, &tree.root, pos, leaf),
                "sibling {i}"
            );
        }
        let mut leaf_bad = leaf;
        leaf_bad[0] ^= 1;
        assert!(!verify_merkle_path_from_leaf_hash::<P1Backend>(
            &path, &tree.root, pos, leaf_bad
        ));
        assert!(!verify_merkle_path_from_leaf_hash::<P1Backend>(
            &path,
            &tree.root,
            pos ^ 1,
            leaf
        ));
        // A path that is not whole levels, and one a level short.
        assert!(!verify_merkle_path_from_leaf_hash::<P1Backend>(
            &path[..path.len() - 1],
            &tree.root,
            pos,
            leaf
        ));
        assert!(!verify_merkle_path_from_leaf_hash::<P1Backend>(
            &path[..path.len() - 3],
            &tree.root,
            pos,
            leaf
        ));
    }
    assert!(tree.get_batch_proof(&[0, 1]).is_err());
}
