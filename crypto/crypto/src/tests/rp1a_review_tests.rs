//! The adversarial review REV-P1-A's crypto tests (the subset the judge,
//! REV-P1-JUDGE §7, keeps): the node compression's collisions under the inverse
//! permutation (the documented assumption: a tree binds through its anchored
//! leaves), and the uncapped odd-depth top's padding, which the host now binds.

use alloc::vec::Vec;

use crate::hash::poseidon1_stark::linear_hash;
use crate::hash::poseidon1_w16::{
    self as w16, Digest, Fp, STATE_FELTS, compress4, constants::MDS_CIRC_ROW,
    constants::ROUND_CONSTANTS, is_full_round,
};
use crate::merkle_tree::cap::CappedRoot;
use crate::merkle_tree::traits::IsMerkleTreeBackend;

const P: u64 = 0xFFFF_FFFF_0000_0001;

/// The P1 leaf hash and 4-ary node over canonical digests (the same test
/// backend `merkle_arity4_tests` uses; `prover`'s `P1BatchBackend` hashes the
/// same felts).
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
        canon(&linear_hash(leaf))
    }

    fn hash_new_parent(left: &[u64; 4], right: &[u64; 4]) -> [u64; 4] {
        Self::hash_four(&[*left, *right, [0; 4], [0; 4]])
    }

    fn hash_four(children: &[[u64; 4]; 4]) -> [u64; 4] {
        canon(&compress4(&children.map(|c| digest(&c))))
    }

    fn padding_node() -> Option<[u64; 4]> {
        Some([0; 4])
    }
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

// ------------------------------------------------------------------ inverse

/// `7^{-1} mod (p − 1)`: the inverse S-box exponent.
fn inv7() -> u64 {
    let m = (P - 1) as i128;
    let (mut a, mut b, mut x0, mut x1) = (7i128, m, 1i128, 0i128);
    while b != 0 {
        let q = a / b;
        (a, b) = (b, a - q * b);
        (x0, x1) = (x1, x0 - q * x1);
    }
    assert_eq!(a, 1);
    (((x0 % m) + m) % m) as u64
}

/// The circulant MDS as a dense matrix (`poseidon1_w16`'s `MDS_MATRIX`).
fn mds_matrix() -> [[Fp; STATE_FELTS]; STATE_FELTS] {
    core::array::from_fn(|i| {
        core::array::from_fn(|j| Fp::from(MDS_CIRC_ROW[(j + STATE_FELTS - i) % STATE_FELTS]))
    })
}

/// Solve `M · x = y` by Gauss–Jordan elimination.
fn mds_solve(y: &[Fp; STATE_FELTS]) -> [Fp; STATE_FELTS] {
    let mut a: Vec<Vec<Fp>> = mds_matrix()
        .iter()
        .zip(y)
        .map(|(row, v)| {
            let mut r = row.to_vec();
            r.push(*v);
            r
        })
        .collect();
    let n = STATE_FELTS;
    for col in 0..n {
        let piv = (col..n)
            .find(|&r| a[r][col] != Fp::zero())
            .expect("the MDS is invertible");
        a.swap(col, piv);
        let inv = a[col][col].inv().expect("nonzero pivot");
        for v in a[col][col..].iter_mut() {
            *v *= inv;
        }
        let pivot_row = a[col].clone();
        for (r, row) in a.iter_mut().enumerate() {
            if r != col && row[col] != Fp::zero() {
                let f = row[col];
                for (v, p) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                    *v = *v - *p * f;
                }
            }
        }
    }
    core::array::from_fn(|i| a[i][n])
}

/// `poseidon1_w16::permute`'s inverse: per round, MDS⁻¹, S-box⁻¹, minus the
/// round constants.
fn permute_inv(state: [Fp; STATE_FELTS]) -> [Fp; STATE_FELTS] {
    let d = inv7();
    let mut s = state;
    for r in (0..w16::NUM_ROUNDS).rev() {
        s = mds_solve(&s);
        if is_full_round(r) {
            for v in s.iter_mut() {
                *v = v.pow(d);
            }
        } else {
            s[0] = s[0].pow(d);
        }
        for (v, c) in s.iter_mut().zip(ROUND_CONSTANTS[r].iter()) {
            *v = *v - Fp::from(*c);
        }
    }
    s
}

fn split4(x: &[Fp; STATE_FELTS]) -> [Digest; 4] {
    core::array::from_fn(|c| core::array::from_fn(|j| x[4 * c + j]))
}

/// The 4-ary node is a TRUNCATED permutation of sixteen free lanes, so with
/// the inverse permutation (public, cheap) it has collisions for free: any two
/// outputs agreeing on lanes 0..4 pull back to two node inputs with one
/// digest. What still binds a tree is the leaf: its first block's capacity is
/// the width tag, so a pulled-back input is a leaf only if its four capacity
/// lanes are that tag.
#[test]
fn rp1a_the_node_compression_collides_under_the_inverse_permutation() {
    let mut fwd = [Fp::zero(); STATE_FELTS];
    for (i, v) in fwd.iter_mut().enumerate() {
        *v = Fp::from(i as u64 * 1_000_003 + 17);
    }
    assert_eq!(permute_inv(w16::permute(fwd)), fwd, "the inverse is exact");

    let target: Digest = core::array::from_fn(|i| Fp::from(0xC0FF_EE00 + i as u64));
    let pull = |tail_seed: u64| {
        let tail = lcg(tail_seed, 12);
        let mut out = [Fp::zero(); STATE_FELTS];
        out[..4].copy_from_slice(&target);
        out[4..].copy_from_slice(&tail);
        permute_inv(out)
    };
    let (x, y) = (pull(1), pull(2));
    assert_ne!(x, y);
    let (cx, cy) = (split4(&x), split4(&y));
    assert_eq!(compress4(&cx), target);
    assert_eq!(compress4(&cy), target);
    assert_ne!(
        cx, cy,
        "two different child quadruples, one node: a collision"
    );

    // The anchor: a pulled-back input is a one-block leaf only if its lanes
    // 12..16 (the leaf's zero capacity) vanish, which they do not.
    assert!(x[12..].iter().any(|v| *v != Fp::zero()));
}

// ----------------------------------------------- the uncapped odd-depth top

/// At an UNCAPPED odd-depth tree the walk reaches the top group, two real
/// children and two padding nodes. The host checker requires the path's two
/// padding siblings to BE the padding (REV-P1-JUDGE F4), as the in-guest walk
/// (`p1w16_emit::walk4`) supplies them as constants: a root over non-zero
/// padding opens on no index, the honest root on every one. At c ≥ 1 the
/// host's `cap_root` pads with the backend's padding node itself.
#[test]
fn rp1a_uncapped_odd_top_padding_is_bound_on_the_host() {
    let depth = 3; // 8 leaves: one full 4-ary level, then a two-node top
    let rows: Vec<Vec<Fp>> = (0..8).map(|k| lcg(0xABC + k, 5)).collect();
    let leaves: Vec<[u64; 4]> = rows.iter().map(P1Backend::hash_data).collect();
    let n0 = P1Backend::hash_four(&[leaves[0], leaves[1], leaves[2], leaves[3]]);
    let n1 = P1Backend::hash_four(&[leaves[4], leaves[5], leaves[6], leaves[7]]);
    let junk = [1u64, 2, 3, 4];
    let forged_root = P1Backend::hash_four(&[n0, n1, junk, junk]);
    let honest_root = P1Backend::hash_four(&[n0, n1, [0; 4], [0; 4]]);
    assert_ne!(forged_root, honest_root);

    for pos in 0..8 {
        let base = pos / 4 * 4;
        let mut path: Vec<[u64; 4]> = (base..base + 4)
            .filter(|&c| c != pos)
            .map(|c| leaves[c])
            .collect();
        // Top level: the partner, then the two padding slots, in child order.
        let partner = if pos < 4 { n1 } else { n0 };
        let mut forged = path.clone();
        forged.extend([partner, junk, junk]);
        assert!(
            !CappedRoot::uncapped(&forged_root, depth).verify::<P1Backend>(
                &forged,
                pos,
                leaves[pos]
            ),
            "the host refuses non-zero padding siblings at pos {pos}"
        );
        path.extend([partner, [0; 4], [0; 4]]);
        assert!(
            CappedRoot::uncapped(&honest_root, depth).verify::<P1Backend>(&path, pos, leaves[pos]),
            "the honest opening at pos {pos}"
        );
    }

    // At cap height 1 the cap is [n0, n1] and its root pads with zeros
    // itself: the forged root is not that cap's root.
    assert!(!crate::merkle_tree::cap::verify_cap_shaped::<P1Backend>(
        &[n0, n1],
        &forged_root,
        depth,
        1
    ));
    assert!(crate::merkle_tree::cap::verify_cap_shaped::<P1Backend>(
        &[n0, n1],
        &honest_root,
        depth,
        1
    ));
}
