//! A P1 root binds its shape (REV-P1-JUDGE §3, F3): under ZisK's untagged leaf
//! (`zisk_linear_hash`) a family of leaf shapes, a narrow tree and a wide tree
//! whose leaves carry the narrow tree's nodes, reach one root; the STARK's leaf
//! (`linear_hash`, ZisK's chain with the width tag `[len, LEAF_DOMAIN, 0, 0]` as
//! its first capacity) parts every member of the family, and the 12-felt-leaf /
//! zero-fourth-child and in-block zero-padding identities with it.

use alloc::vec::Vec;

use crate::hash::poseidon1_stark::{linear_hash, zisk_linear_hash};
use crate::hash::poseidon1_w16::{Digest, Fp, compress4};
use crate::merkle_tree::cap::{
    tree_levels, verify_cap_shaped, verify_merkle_path_to_cap_from_leaf_hash,
};
use crate::merkle_tree::merkle::MerkleTree;
use crate::merkle_tree::traits::IsMerkleTreeBackend;

const P: u64 = 0xFFFF_FFFF_0000_0001;

fn canon(d: &Digest) -> [u64; 4] {
    core::array::from_fn(|i| d[i].canonical())
}
fn digest(n: &[u64; 4]) -> Digest {
    core::array::from_fn(|i| Fp::from(n[i]))
}
fn lcg(seed: u64, n: usize) -> Vec<Fp> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            Fp::from(x % P)
        })
        .collect()
}

/// ZisK's untagged leaf (`zisk_linear_hash`) over 4-ary nodes: the reference
/// the family is built against.
struct Today;
impl IsMerkleTreeBackend for Today {
    type Node = [u64; 4];
    type Data = Vec<Fp>;
    const ARITY: usize = 4;
    fn hash_data(leaf: &Vec<Fp>) -> [u64; 4] {
        canon(&zisk_linear_hash(leaf))
    }
    fn hash_new_parent(l: &[u64; 4], r: &[u64; 4]) -> [u64; 4] {
        Self::hash_four(&[*l, *r, [0; 4], [0; 4]])
    }
    fn hash_four(c: &[[u64; 4]; 4]) -> [u64; 4] {
        canon(&compress4(&c.map(|x| digest(&x))))
    }
    fn padding_node() -> Option<[u64; 4]> {
        Some([0; 4])
    }
}

/// The STARK's leaf (`linear_hash`, width-tagged) over the same 4-ary nodes.
struct Tagged;
impl IsMerkleTreeBackend for Tagged {
    type Node = [u64; 4];
    type Data = Vec<Fp>;
    const ARITY: usize = 4;
    fn hash_data(leaf: &Vec<Fp>) -> [u64; 4] {
        canon(&linear_hash(leaf))
    }
    fn hash_new_parent(l: &[u64; 4], r: &[u64; 4]) -> [u64; 4] {
        Self::hash_four(&[*l, *r, [0; 4], [0; 4]])
    }
    fn hash_four(c: &[[u64; 4]; 4]) -> [u64; 4] {
        Today::hash_four(c)
    }
    fn padding_node() -> Option<[u64; 4]> {
        Some([0; 4])
    }
}

/// Wide leaves over a narrow tree of `w`-felt rows, `m` 4-ary levels up:
/// the wide leaf for narrow block `i` (4^m narrow rows) is the last narrow
/// row, zero-extended to whole 12-felt blocks, followed by three digests per
/// level, bottom level first. Returns (narrow rows, wide leaves).
fn dual_family<B: IsMerkleTreeBackend<Node = [u64; 4], Data = Vec<Fp>>>(
    w: usize,
    m: u32,
    n_wide: usize,
) -> (Vec<Vec<Fp>>, Vec<Vec<Fp>>) {
    let span = 4usize.pow(m);
    let narrow: Vec<Vec<Fp>> = (0..n_wide * span)
        .map(|k| lcg(0xD0A1 + k as u64, w))
        .collect();
    // Level digests of the narrow tree, bottom (leaf digests) first.
    let mut levels: Vec<Vec<[u64; 4]>> = vec![narrow.iter().map(B::hash_data).collect()];
    for _ in 1..m {
        let up = levels
            .last()
            .unwrap()
            .chunks_exact(4)
            .map(|c| B::hash_four(&[c[0], c[1], c[2], c[3]]))
            .collect();
        levels.push(up);
    }
    let wide = (0..n_wide)
        .map(|i| {
            let mut leaf = narrow[i * span + span - 1].clone();
            leaf.resize(w.div_ceil(12) * 12, Fp::zero());
            for (lvl, ds) in levels.iter().enumerate() {
                // the last group of 4 at this level under wide leaf i
                let per = span / 4usize.pow(lvl as u32); // nodes at this level under i
                let base = i * per + per - 4;
                for k in 0..3 {
                    leaf.extend_from_slice(&digest(&ds[base + k]));
                }
            }
            leaf
        })
        .collect();
    (narrow, wide)
}

fn opens_all<B: IsMerkleTreeBackend<Node = [u64; 4], Data = Vec<Fp>>>(
    data: &[Vec<Fp>],
    depth: usize,
    tree: &MerkleTree<B>,
    c: usize,
) -> bool {
    let levels = tree_levels(depth, 4);
    let cap = if c == 0 {
        vec![tree.root]
    } else {
        tree.cap(c).expect("cap")
    };
    verify_cap_shaped::<B>(&cap, &tree.root, depth, c)
        && data.iter().enumerate().all(|(pos, row)| {
            let full = tree.get_proof_by_pos(pos).expect("in range").merkle_path;
            verify_merkle_path_to_cap_from_leaf_hash::<B>(
                &full[..3 * (levels - c)],
                &cap,
                depth,
                pos,
                B::hash_data(row),
            )
        })
}

/// For every width w ≥ 12 with w ≡ 0 (mod 6), a tree of w-felt leaves and a tree
/// of 2w-felt leaves built from its nodes reach one root under ZisK's untagged
/// leaf, and both open at every cap height. The STARK's width-tagged leaf parts
/// every one of them: a P1 root binds its leaf width. w = 12, 18, 24 and 30.
#[test]
fn the_width_tag_parts_the_shape_dual_family() {
    // (w, m): 2w = 12·(⌈w/12⌉ + m) felts.
    for (w, m) in [(12usize, 1u32), (18, 1), (24, 2), (30, 2)] {
        assert_eq!(2 * w, 12 * (w.div_ceil(12) + m as usize), "w {w}");
        let n_wide = 16usize;
        let (narrow, wide) = dual_family::<Today>(w, m, n_wide);
        assert!(wide.iter().all(|l| l.len() == 2 * w));
        let tn = MerkleTree::<Today>::build(&narrow).unwrap();
        let tw = MerkleTree::<Today>::build(&wide).unwrap();
        assert_eq!(tn.root, tw.root, "w {w}: one root, two shapes");
        let dn = (narrow.len()).trailing_zeros() as usize;
        let dw = (wide.len()).trailing_zeros() as usize;
        assert_eq!(dn - dw, 2 * m as usize);
        for c in 0..=tree_levels(dw, 4) {
            assert!(opens_all(&narrow, dn, &tn, c), "w {w} narrow c {c}");
            assert!(opens_all(&wide, dw, &tw, c), "w {w} wide c {c}");
            if c > 0 {
                assert_eq!(tn.cap(c), tw.cap(c), "w {w} c {c}: one cap too");
            }
        }

        // The width tag: the same construction lands on two roots.
        let (narrow, wide) = dual_family::<Tagged>(w, m, n_wide);
        let tn = MerkleTree::<Tagged>::build(&narrow).unwrap();
        let tw = MerkleTree::<Tagged>::build(&wide).unwrap();
        assert_ne!(tn.root, tw.root, "w {w}: the width tag parts the shapes");
    }
    // Widths that are not ≡ 0 (mod 6), or below 12, have no dual: the wide
    // leaf's last block would need zero digest lanes.
    for w in [6usize, 10, 13, 16, 17, 19] {
        assert!(w < 12 || w % 6 != 0);
    }
}

/// The width tag also removes the 12-felt-leaf / zero-fourth-child identity
/// (node(a, b, c, 0) = leaf(a ‖ b ‖ c)).
#[test]
fn the_width_tag_parts_a_short_leaf_from_a_padded_node() {
    let row = lcg(77, 12);
    let d = |k: usize| -> Digest { core::array::from_fn(|j| row[4 * k + j]) };
    let node = compress4(&[d(0), d(1), d(2), [Fp::zero(); 4]]);
    assert_eq!(node, zisk_linear_hash(&row), "ZisK's untagged leaf");
    assert_ne!(node, linear_hash(&row), "the STARK's leaf");
    // ... and the in-block zero-padding identity (REV-P1-B M1's leaf half).
    let x = lcg(78, 5);
    let mut x0 = x.clone();
    x0.push(Fp::zero());
    assert_eq!(zisk_linear_hash(&x), zisk_linear_hash(&x0));
    assert_ne!(linear_hash(&x), linear_hash(&x0));
}
