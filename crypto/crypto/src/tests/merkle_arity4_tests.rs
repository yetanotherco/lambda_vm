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

#[test]
fn arity_4_cap_lengths_are_pinned() {
    use crate::merkle_tree::cap::{MAX_CAP_HEIGHT, cap_len, tree_levels};
    // Odd depth: the top group is two real nodes and two paddings.
    assert_eq!(tree_levels(3, 4), 2);
    assert_eq!(
        (0..=2).map(|c| cap_len(3, c, 4)).collect::<Vec<_>>(),
        [Some(1), Some(2), Some(8)]
    );
    assert_eq!(cap_len(3, 3, 4), None);
    // Even depth: powers of four.
    assert_eq!(tree_levels(4, 4), 2);
    assert_eq!(
        (0..=2).map(|c| cap_len(4, c, 4)).collect::<Vec<_>>(),
        [Some(1), Some(4), Some(16)]
    );
    assert_eq!(cap_len(22, 4, 4), Some(256));
    assert_eq!(cap_len(21, 4, 4), Some(128));
    // No cap holds more than 2^MAX_CAP_HEIGHT nodes; the binary rule is unchanged.
    assert_eq!(
        cap_len(40, MAX_CAP_HEIGHT / 2, 4),
        Some(1 << MAX_CAP_HEIGHT)
    );
    assert_eq!(cap_len(40, MAX_CAP_HEIGHT / 2 + 1, 4), None);
    assert_eq!(cap_len(22, 3, 2), Some(8));
    assert_eq!(cap_len(2, 3, 2), None);
}

/// Every opening of every leaf position class, at every cap height of trees
/// over `2^k` leaves, `k` odd and even: the cap is the tree's level, hashes to
/// the root, rides on the owner path and every opening verifies against it.
#[test]
fn arity_4_caps_round_trip_at_every_height() {
    use crate::merkle_tree::cap::{
        CappedRoot, cap_len, cap_root, embed_cap_arity, split_owner_path_arity, tree_levels,
    };
    for k in 0..=9usize {
        let n = 1usize << k;
        let data = rows(n);
        let tree = MerkleTree::<P1Backend>::build(&data).expect("non-empty");
        let h = tree_levels(k, 4);
        for c in 0..=h {
            let cap = tree.cap(c).expect("c ≤ h");
            assert_eq!(Some(cap.len()), cap_len(k, c, 4), "k {k} c {c}");
            assert_eq!(cap_root::<P1Backend>(&cap), Some(tree.root), "k {k} c {c}");
            let positions: Vec<usize> = [0, n / 3, n / 2, n - 1].into_iter().collect();
            let mut paths: Vec<Vec<[u64; 4]>> = positions
                .iter()
                .map(|&p| tree.get_proof_by_pos(p).expect("in range").merkle_path)
                .collect();
            let mut refs: Vec<&mut Vec<[u64; 4]>> = paths.iter_mut().collect();
            embed_cap_arity(&mut refs, k, &cap, 4).expect("full paths");
            let keep = 3 * (h - c);
            let owner_len = if c == 0 { keep } else { keep + cap.len() };
            assert_eq!(paths[0].len(), owner_len, "k {k} c {c}");
            let (split_siblings, split_cap) =
                split_owner_path_arity(&paths[0], k, c, 4).expect("owner length");
            assert_eq!(split_siblings.len(), keep);
            if c > 0 {
                assert_eq!(split_cap, &cap[..]);
            }
            let (check, owner) =
                CappedRoot::from_owner::<P1Backend>(&tree.root, &paths[0], k, c).expect("honest");
            let root_cap = [tree.root];
            assert_eq!(check.cap(), if c == 0 { &root_cap[..] } else { &cap[..] });
            for (q, &pos) in positions.iter().enumerate() {
                let siblings = if q == 0 { owner } else { &paths[q][..] };
                assert_eq!(siblings.len(), keep, "k {k} c {c} q {q}");
                let leaf = P1Backend::hash_data(&data[pos]);
                assert!(
                    check.verify::<P1Backend>(siblings, pos, leaf),
                    "k {k} c {c} pos {pos}"
                );
            }
        }
        assert!(tree.cap(h + 1).is_none());
    }
}

/// A tree, its rows, the owner path of one opening (cap embedded), the binary
/// depth, the cap height and the opened position.
type CappedFixture = (
    MerkleTree<P1Backend>,
    Vec<Vec<Fp>>,
    Vec<[u64; 4]>,
    usize,
    usize,
    usize,
);

/// The whole capped check of one owner opening: `(root, owner path, depth, c,
/// index, leaf hash)`.
type AcceptOwner = fn(&[u64; 4], &[[u64; 4]], usize, usize, usize, [u64; 4]) -> bool;

/// [`crate::merkle_tree::cap::verify_merkle_path_to_cap_from_leaf_hash`]'s shape.
type VerifyToCap = fn(&[[u64; 4]], &[[u64; 4]], usize, usize, [u64; 4]) -> bool;

/// One capped tree (`k = 9`, `h = 5`, `c = 3`: 32 cap nodes, 6 kept siblings)
/// and one honest opening, for the tampers below.
fn capped_fixture() -> CappedFixture {
    use crate::merkle_tree::cap::embed_cap_arity;
    let (k, c, pos) = (9usize, 3usize, 300usize);
    let data = rows(1 << k);
    let tree = MerkleTree::<P1Backend>::build(&data).expect("non-empty");
    let mut owner = tree.get_proof_by_pos(pos).expect("in range").merkle_path;
    embed_cap_arity(&mut [&mut owner], k, &tree.cap(c).expect("c ≤ h"), 4).expect("full");
    (tree, data, owner, k, c, pos)
}

#[test]
fn a_tampered_capped_arity_4_opening_is_rejected() {
    use crate::merkle_tree::cap::CappedRoot;
    let (tree, data, owner, k, c, pos) = capped_fixture();
    let leaf = P1Backend::hash_data(&data[pos]);
    let (check, siblings) =
        CappedRoot::from_owner::<P1Backend>(&tree.root, &owner, k, c).expect("honest");
    assert!(check.verify::<P1Backend>(siblings, pos, leaf));
    let keep = siblings.len();
    // Every cap node: the cap no longer hashes to the root.
    for i in keep..owner.len() {
        let mut bad = owner.clone();
        bad[i][i % 4] ^= 1;
        assert!(
            CappedRoot::from_owner::<P1Backend>(&tree.root, &bad, k, c).is_none(),
            "cap node {}",
            i - keep
        );
    }
    // Every kept sibling.
    for i in 0..keep {
        let mut bad = siblings.to_vec();
        bad[i][0] ^= 1;
        assert!(!check.verify::<P1Backend>(&bad, pos, leaf), "sibling {i}");
    }
    // The owner path one node long or short, and at another height.
    let mut long = owner.clone();
    long.push([0; 4]);
    assert!(CappedRoot::from_owner::<P1Backend>(&tree.root, &long, k, c).is_none());
    assert!(
        CappedRoot::from_owner::<P1Backend>(&tree.root, &owner[..owner.len() - 1], k, c).is_none()
    );
    assert!(CappedRoot::from_owner::<P1Backend>(&tree.root, &owner, k, c - 1).is_none());
    assert!(CappedRoot::from_owner::<P1Backend>(&tree.root, &owner, k, c + 1).is_none());
    // An opening a node short or long, another index, an index past the tree,
    // another leaf.
    assert!(!check.verify::<P1Backend>(&siblings[..keep - 1], pos, leaf));
    assert!(!check.verify::<P1Backend>(&[siblings, &[[0; 4]]].concat(), pos, leaf));
    assert!(!check.verify::<P1Backend>(siblings, pos ^ 1, leaf));
    assert!(!check.verify::<P1Backend>(siblings, pos + (1 << k), leaf));
    assert!(!check.verify::<P1Backend>(siblings, pos, P1Backend::hash_data(&data[pos + 1])));
}

/// The owner path of a FORGED tree's opening, carried next to the honest
/// root: its cap is consistent with its own siblings, so only the cap-to-root
/// check stands between it and acceptance. `accept` is the whole capped check
/// of one owner opening: `(root, owner path, depth, c, index, leaf hash)`.
fn admits_forged_cap(accept: AcceptOwner) -> bool {
    use crate::merkle_tree::cap::embed_cap_arity;
    let (honest, _, _, k, c, pos) = capped_fixture();
    let forged_data: Vec<Vec<Fp>> = rows((1 << k) + 1)[1..].to_vec();
    let forged = MerkleTree::<P1Backend>::build(&forged_data).expect("non-empty");
    assert_ne!(forged.root, honest.root);
    let mut owner = forged.get_proof_by_pos(pos).expect("in range").merkle_path;
    embed_cap_arity(&mut [&mut owner], k, &forged.cap(c).expect("c ≤ h"), 4).expect("full");
    accept(
        &honest.root,
        &owner,
        k,
        c,
        pos,
        P1Backend::hash_data(&forged_data[pos]),
    )
}

#[test]
fn the_cap_to_root_check_rejects_a_forged_arity_4_cap() {
    assert!(!admits_forged_cap(|root, owner, d, c, pos, leaf| {
        crate::merkle_tree::cap::CappedRoot::from_owner::<P1Backend>(root, owner, d, c)
            .is_some_and(|(check, siblings)| check.verify::<P1Backend>(siblings, pos, leaf))
    }));
}

#[test]
fn mutation_without_the_cap_to_root_check_admits_a_forged_arity_4_cap() {
    assert!(admits_forged_cap(|_root, owner, d, c, pos, leaf| {
        use crate::merkle_tree::cap::{
            split_owner_path_arity, verify_merkle_path_to_cap_from_leaf_hash,
        };
        // Mutation: the split and the walk to the cap, no cap-to-root check.
        split_owner_path_arity(owner, d, c, 4).is_some_and(|(siblings, cap)| {
            verify_merkle_path_to_cap_from_leaf_hash::<P1Backend>(siblings, cap, d, pos, leaf)
        })
    }));
}

/// The real node one 4-ary level above leaf 0 presented as the leaf hash, with
/// the path from it upward: three siblings short. Index 0 keeps every slot the
/// fold reads at 0, so only the exact-length check rejects it.
fn admits_internal_node(verify: VerifyToCap) -> bool {
    let (tree, data, owner, k, c, _) = capped_fixture();
    let cap = &owner[owner.len() - tree.cap(c).expect("c ≤ h").len()..];
    let full = tree.get_proof_by_pos(0).expect("in range").merkle_path;
    let parent = P1Backend::hash_four(&[P1Backend::hash_data(&data[0]), full[0], full[1], full[2]]);
    let keep = 3 * (crate::merkle_tree::cap::tree_levels(k, 4) - c);
    verify(&full[3..keep], cap, k, 0, parent)
}

#[test]
fn the_exact_length_check_rejects_an_internal_node_at_arity_4() {
    assert!(!admits_internal_node(
        crate::merkle_tree::cap::verify_merkle_path_to_cap_from_leaf_hash::<P1Backend>
    ));
}

#[test]
fn mutation_without_the_length_check_admits_an_internal_node_at_arity_4() {
    fn mutant(s: &[[u64; 4]], cap: &[[u64; 4]], d: usize, i: usize, l: [u64; 4]) -> bool {
        // Mutation: no `siblings.len() == 3(h − c)` check; the cap node read
        // as the honest verifier reads it.
        let h = crate::merkle_tree::cap::tree_levels(d, 4);
        let c = (0..=h)
            .find(|&c| crate::merkle_tree::cap::cap_len(d, c, 4) == Some(cap.len()))
            .expect("a cap length");
        let walked = h - c;
        verify_merkle_path_from_leaf_hash::<P1Backend>(s, &cap[i >> (2 * walked)], i, l)
    }
    assert!(admits_internal_node(mutant));
}

#[test]
fn an_embed_or_split_of_the_wrong_shape_is_refused() {
    use crate::merkle_tree::cap::{embed_cap_arity, split_owner_path_arity};
    let data = rows(16);
    let tree = MerkleTree::<P1Backend>::build(&data).expect("non-empty");
    let full = tree.get_proof_by_pos(5).expect("in range").merkle_path;
    // A path of another depth, a cap of no height (two nodes at even depth),
    // no owner.
    assert!(embed_cap_arity(&mut [&mut full.clone()], 6, &tree.cap(1).unwrap(), 4).is_err());
    assert!(embed_cap_arity(&mut [&mut full.clone()], 4, &[tree.root, tree.root], 4).is_err());
    assert!(embed_cap_arity::<[u64; 4]>(&mut [], 4, &tree.cap(1).unwrap(), 4).is_err());
    // A full path is not an owner path of height 1.
    assert!(split_owner_path_arity(&full, 4, 1, 4).is_none());
    assert!(split_owner_path_arity(&full, 4, 0, 4).is_some());
    assert!(split_owner_path_arity(&full, 4, 3, 4).is_none());
}
