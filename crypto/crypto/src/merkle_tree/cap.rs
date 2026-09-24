//! Merkle caps: authentication paths that stop `c` levels below the root.
//!
//! A tree of depth `D` (so `2^D` padded leaves) has, at height `c`, the `2^c`
//! nodes that sit `c` levels below its root — its **cap**. A proof can carry a
//! tree's cap once and cut every authentication path of that tree to its first
//! `D − c` siblings: the verifier folds a path up to the cap node
//! `cap[index >> (D − c)]` instead of the root, and checks once per tree that
//! the cap hashes up to the committed root (`2^c − 1` compressions).
//!
//! **The root stays the commitment.** Nothing about the transcript changes:
//! the root is what is absorbed, and a second cap with the same root is a
//! compression collision. Any capped acceptance extends to a full-path
//! acceptance (append the cap-to-root computation above the cap node), so the
//! query-phase bound is the one the full paths had.
//!
//! **Four checks are load-bearing** — dropping any one is a soundness break:
//! 1. `cap.len() == 2^c` and `cap_root(cap) == root`, once per tree
//!    ([`verify_cap`], run by [`CappedRoot::from_owner`]);
//! 2. every path is exactly `D − c` siblings long, and the owner path exactly
//!    `D − c + 2^c` ([`verify_merkle_path_to_cap_from_leaf_hash`],
//!    [`split_owner_path`]);
//! 3. the cap node is `cap[index >> (D − c)]` with `index < 2^D`, the index
//!    being the transcript's;
//! 4. `c` itself is a verifier constant ([`CapPolicy::height`] of public shape
//!    data), never read from the proof.
//!
//! At `c = 0` the cap is `[root]` and the capped check is exactly
//! [`verify_merkle_path_from_leaf_hash`] plus the two exact-length checks.
//!
//! **Wire encoding (the owner path).** A tree's cap rides at the end of the
//! authentication path of that tree's first opening in proof order
//! ([`embed_cap`] / [`split_owner_path`]); every other opening of the tree
//! carries exactly `D − c` siblings. At `c = 0` nothing moves, so the default
//! proof bytes are today's by construction.

use alloc::vec::Vec;
use core::fmt;
use core::str::FromStr;

use super::proof::{Proof, verify_merkle_path_from_leaf_hash};
use super::traits::IsMerkleTreeBackend;

/// The tallest cap any policy may ask for. A proof-size guard: a cap costs
/// `2^c` digests per tree. The `Auto` policy never exceeds 3.
pub const MAX_CAP_HEIGHT: usize = 16;

/// Why a cap operation refused its input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapError {
    /// The cap height exceeds the tree depth, or [`MAX_CAP_HEIGHT`].
    CapTooTall { cap_height: usize, depth: usize },
    /// A path did not have the exact length the shape requires.
    PathLength { expected: usize, got: usize },
    /// A cap whose length is not a power of two.
    CapLength(usize),
    /// A cap with no opening to carry it.
    NoOwner,
}

impl fmt::Display for CapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapTooTall { cap_height, depth } => write!(
                f,
                "cap height {cap_height} exceeds the tree depth {depth} or the maximum {MAX_CAP_HEIGHT}"
            ),
            Self::PathLength { expected, got } => {
                write!(
                    f,
                    "authentication path has {got} nodes, expected {expected}"
                )
            }
            Self::CapLength(len) => write!(f, "cap of {len} nodes is not a power of two"),
            Self::NoOwner => write!(f, "a cap needs at least one opening to carry it"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CapError {}

/// `log2(cap.len())` when the cap is a non-empty power of two no taller than
/// [`MAX_CAP_HEIGHT`].
fn cap_height_of<N>(cap: &[N]) -> Option<usize> {
    let len = cap.len();
    if !len.is_power_of_two() {
        return None;
    }
    let c = len.ilog2() as usize;
    (c <= MAX_CAP_HEIGHT).then_some(c)
}

/// `c ≤ depth`, `c ≤ MAX_CAP_HEIGHT`, and `depth` small enough that `2^depth`
/// is a `usize`.
fn shape_ok(depth: usize, cap_height: usize) -> bool {
    cap_height <= depth && cap_height <= MAX_CAP_HEIGHT && depth < usize::BITS as usize
}

impl<T: PartialEq + Eq> Proof<T> {
    /// Keep the first `depth − cap_height` siblings of a full path.
    ///
    /// Refuses unless the path is exactly `depth` long and the cap fits the
    /// tree, so a path that was already cut, or one of another tree, is not
    /// silently cut again.
    pub fn truncate_to_cap(&mut self, depth: usize, cap_height: usize) -> Result<(), CapError> {
        if !shape_ok(depth, cap_height) {
            return Err(CapError::CapTooTall { cap_height, depth });
        }
        if self.merkle_path.len() != depth {
            return Err(CapError::PathLength {
                expected: depth,
                got: self.merkle_path.len(),
            });
        }
        self.merkle_path.truncate(depth - cap_height);
        Ok(())
    }
}

/// The root of a cap: the standard bottom-up build over the cap as leaves,
/// `2^c − 1` compressions (none at `c = 0`). `None` unless `cap.len()` is a
/// power of two `≥ 1` no taller than [`MAX_CAP_HEIGHT`].
pub fn cap_root<B: IsMerkleTreeBackend>(cap: &[B::Node]) -> Option<B::Node> {
    cap_height_of(cap)?;
    let mut level: Vec<B::Node> = cap.to_vec();
    while level.len() > 1 {
        level = level
            .chunks_exact(2)
            .map(|pair| B::hash_new_parent(&pair[0], &pair[1]))
            .collect();
    }
    level.pop()
}

/// `cap.len() == 2^cap_height` and the cap hashes up to `root`.
pub fn verify_cap<B: IsMerkleTreeBackend>(
    cap: &[B::Node],
    root: &B::Node,
    cap_height: usize,
) -> bool {
    if cap_height > MAX_CAP_HEIGHT || cap.len() != 1usize << cap_height {
        return false;
    }
    cap_root::<B>(cap).is_some_and(|r| &r == root)
}

/// The capped inclusion check for one opening.
///
/// `c = log2(cap.len())`. Accepts iff
/// `siblings.len() == depth − c`, `index < 2^depth`, `cap.len() == 2^c ≤ 2^depth`,
/// and the existing fold ([`verify_merkle_path_from_leaf_hash`], unchanged)
/// of `leaf_hash` along `siblings` lands on `cap[index >> (depth − c)]`.
///
/// It does NOT check the cap against a root — that is [`verify_cap`], once per
/// tree. [`CappedRoot`] ties the two together.
pub fn verify_merkle_path_to_cap_from_leaf_hash<B: IsMerkleTreeBackend>(
    siblings: &[B::Node],
    cap: &[B::Node],
    depth: usize,
    index: usize,
    leaf_hash: B::Node,
) -> bool {
    let Some(c) = cap_height_of(cap) else {
        return false;
    };
    if !shape_ok(depth, c) || siblings.len() != depth - c || index >> depth != 0 {
        return false;
    }
    verify_merkle_path_from_leaf_hash::<B>(siblings, &cap[index >> (depth - c)], index, leaf_hash)
}

/// Split an owner path (the wire encoding) into `(siblings, cap)`.
///
/// At `c = 0` the whole path is siblings (its length must be `depth`) and the
/// cap is empty — the caller uses the root. At `c ≥ 1` the length must be
/// exactly `depth − c + 2^c`. `None` on any other length or shape.
pub fn split_owner_path<N>(path: &[N], depth: usize, cap_height: usize) -> Option<(&[N], &[N])> {
    if !shape_ok(depth, cap_height) {
        return None;
    }
    let siblings = depth - cap_height;
    let expected = if cap_height == 0 {
        depth
    } else {
        siblings + (1usize << cap_height)
    };
    (path.len() == expected).then(|| path.split_at(siblings))
}

/// Prover side of the owner-path encoding: cut every path of one tree to
/// `depth − c` siblings and append the cap to `paths[0]`, the tree's first
/// opening in proof order. `c = log2(cap.len())`.
///
/// Every path must be a full `depth`-long path (checked). At `c = 0` (a cap of
/// one node, the root) it changes nothing.
pub fn embed_cap<N: Clone>(
    paths: &mut [&mut Vec<N>],
    depth: usize,
    cap: &[N],
) -> Result<(), CapError> {
    let c = cap_height_of(cap).ok_or(CapError::CapLength(cap.len()))?;
    if !shape_ok(depth, c) {
        return Err(CapError::CapTooTall {
            cap_height: c,
            depth,
        });
    }
    for path in paths.iter() {
        if path.len() != depth {
            return Err(CapError::PathLength {
                expected: depth,
                got: path.len(),
            });
        }
    }
    if c == 0 {
        return Ok(());
    }
    let Some((owner, rest)) = paths.split_first_mut() else {
        return Err(CapError::NoOwner);
    };
    owner.truncate(depth - c);
    owner.extend_from_slice(cap);
    for path in rest {
        path.truncate(depth - c);
    }
    Ok(())
}

/// One tree's authenticated cap: built once per tree, then used for every
/// opening of that tree.
///
/// The only constructors are [`CappedRoot::uncapped`] (`c = 0`, the cap is the
/// root itself) and [`CappedRoot::from_owner`], which runs [`verify_cap`]
/// against the root before it hands the cap out — so a `CappedRoot` never
/// holds an unauthenticated cap.
#[derive(Debug, Clone, Copy)]
pub struct CappedRoot<'a, N> {
    depth: usize,
    cap_height: usize,
    cap: &'a [N],
}

impl<'a, N: PartialEq + Eq + Clone> CappedRoot<'a, N> {
    /// `c = 0`: every path must be exactly `depth` long and fold to `root`.
    pub fn uncapped(root: &'a N, depth: usize) -> Self {
        Self {
            depth,
            cap_height: 0,
            cap: core::slice::from_ref(root),
        }
    }

    /// Split the owner path, authenticate its cap against `root` once, and
    /// return the owner's own siblings.
    ///
    /// Only the cap is checked here. The owner's opening is still an opening:
    /// the caller must run [`verify`](Self::verify) on the returned siblings
    /// like on any other path. `None` on a wrong length or a cap that does not
    /// hash to `root`.
    pub fn from_owner<B: IsMerkleTreeBackend<Node = N>>(
        root: &'a N,
        owner_path: &'a [N],
        depth: usize,
        cap_height: usize,
    ) -> Option<(Self, &'a [N])> {
        let (siblings, cap) = split_owner_path(owner_path, depth, cap_height)?;
        if cap_height == 0 {
            return Some((Self::uncapped(root, depth), siblings));
        }
        if !verify_cap::<B>(cap, root, cap_height) {
            return None;
        }
        Some((
            Self {
                depth,
                cap_height,
                cap,
            },
            siblings,
        ))
    }

    /// Check one opening: exactly `depth − c` siblings folding `leaf_hash` at
    /// `index` onto its cap node.
    pub fn verify<B: IsMerkleTreeBackend<Node = N>>(
        &self,
        siblings: &[N],
        index: usize,
        leaf_hash: N,
    ) -> bool {
        verify_merkle_path_to_cap_from_leaf_hash::<B>(
            siblings, self.cap, self.depth, index, leaf_hash,
        )
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn cap_height(&self) -> usize {
        self.cap_height
    }

    /// The authenticated cap (`[root]` at `c = 0`).
    pub fn cap(&self) -> &'a [N] {
        self.cap
    }
}

// ===========================================================================
// The cap-height policy
// ===========================================================================

/// How tall a cap each tree gets. A proof-format parameter: the prover and
/// every verifier (host and in-guest) derive the same height from public
/// shape data through [`CapPolicy::height`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CapPolicy {
    /// No cap: every path runs to the root. Today's format.
    #[default]
    Off,
    /// The height that minimises the in-guest verifier's cost-law price
    /// ([`AUTO_WEIGHTS`]); 3 for a tree opened ≥ 20 times, 2 for 4–19, 0
    /// below, clamped to the tree depth.
    Auto,
    /// This height for every opened tree, clamped to its depth and to
    /// [`MAX_CAP_HEIGHT`]. `Fixed(0)` is `Off`.
    Fixed(u8),
}

/// Per-row prices (ns) of the in-guest verifier operations a cap trades, from
/// the node cost law (421 ns/instruction + 5.63 ns/cell) and the committed
/// widths of the chips that execute them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapWeights {
    /// One two-to-one compression (an `LFM_HASH` row).
    pub compress: u64,
    /// One two-way `Select`.
    pub select: u64,
    /// One `Unpack` (a digest compared as lanes).
    pub unpack: u64,
    /// One hinted word.
    pub hint: u64,
    /// One digest-equals-root comparison.
    pub compare: u64,
}

/// The weights [`CapPolicy::Auto`] optimises. ⚠ A FORMAT CONSTANT: changing
/// any of them changes the cap heights, and so the proofs, of every tree under
/// `Auto`. Pinned by the policy tests.
pub const AUTO_WEIGHTS: CapWeights = CapWeights {
    compress: 2251,
    select: 567,
    unpack: 528,
    hint: 460,
    compare: 3789,
};

/// The in-guest saving (ns, cost-law units) of a height-`c` cap on a tree
/// opened `openings` times. Integer arithmetic only, so every verifier
/// reproduces it exactly.
///
/// Signed and bounded: the gain is negative for few openings, so it is an
/// `i128`, and every term fits it for any `usize` opening count because
/// `cap_height` is refused past [`MAX_CAP_HEIGHT`] (the result is then
/// `i128::MIN`, a height no argmax picks) — no shift or product can overflow on
/// any target, 32-bit `wasm` included. `gain(o, 0) = 0`; for `c ≥ 1`:
///
/// ```text
/// o·( c·(compress + select) − (2^c − 1)·select − unpack )
///   − ( (2^c − 1)·compress + 2^c·hint + compare )
/// ```
///
/// Per opening the walk loses `c` levels (a `Select` and a compression each),
/// the cap mux adds `2^c − 1` selects and the variable-cell compare one
/// `Unpack`; per tree the cap costs `2^c − 1` compressions to its root, `2^c`
/// hints and one root compare.
pub fn cap_gain(weights: &CapWeights, openings: usize, cap_height: usize) -> i128 {
    if cap_height == 0 {
        return 0;
    }
    if cap_height > MAX_CAP_HEIGHT {
        return i128::MIN;
    }
    let o = openings as i128;
    let c = cap_height as i128;
    let nodes = 1i128 << cap_height;
    let w = |x: u64| x as i128;
    let per_opening = c * (w(weights.compress) + w(weights.select))
        - (nodes - 1) * w(weights.select)
        - w(weights.unpack);
    let per_tree = (nodes - 1) * w(weights.compress) + nodes * w(weights.hint) + w(weights.compare);
    o * per_opening - per_tree
}

impl CapPolicy {
    /// True when this policy caps nothing (`Off` or `Fixed(0)`).
    pub const fn is_off(self) -> bool {
        matches!(self, Self::Off | Self::Fixed(0))
    }

    /// The cap height of a tree of `depth` levels opened `openings` times.
    /// Always `≤ depth` and `≤ MAX_CAP_HEIGHT`, and 0 for an unopened tree.
    ///
    /// `Auto` is RULINGS 1's table, stated directly — 3 for a tree opened at
    /// least [`AUTO_CAP3_MIN_OPENINGS`] times, 2 from
    /// [`AUTO_CAP2_MIN_OPENINGS`], 0 below — then clamped to the depth. No
    /// arithmetic runs at all, so no verifier can disagree on an overflow.
    /// The table is the argmax of [`cap_gain`] under [`AUTO_WEIGHTS`] for
    /// every opening count (pinned by a test over all counts up to 10^6 and
    /// at the `usize` extremes).
    pub fn height(self, openings: usize, depth: usize) -> usize {
        if openings == 0 {
            return 0;
        }
        let limit = depth.min(MAX_CAP_HEIGHT);
        match self {
            Self::Off => 0,
            Self::Fixed(c) => (c as usize).min(limit),
            Self::Auto => {
                let c = if openings >= AUTO_CAP3_MIN_OPENINGS {
                    3
                } else if openings >= AUTO_CAP2_MIN_OPENINGS {
                    2
                } else {
                    0
                };
                c.min(limit)
            }
        }
    }
}

/// `Auto` gives a height-3 cap to a tree opened at least this many times
/// (RULINGS 1). ⚠ A FORMAT CONSTANT, like [`AUTO_WEIGHTS`].
pub const AUTO_CAP3_MIN_OPENINGS: usize = 20;

/// `Auto` gives a height-2 cap to a tree opened at least this many times and
/// fewer than [`AUTO_CAP3_MIN_OPENINGS`] (RULINGS 1). ⚠ A FORMAT CONSTANT.
pub const AUTO_CAP2_MIN_OPENINGS: usize = 4;

impl fmt::Display for CapPolicy {
    /// `off`, `auto`, or the fixed height (`Fixed(0)` prints `off`) — the
    /// spelling the `LAMBDA_VM_ZF_*CAP` knobs accept.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Off | Self::Fixed(0) => f.write_str("off"),
            Self::Auto => f.write_str("auto"),
            Self::Fixed(c) => write!(f, "{c}"),
        }
    }
}

/// A cap-policy spelling that is none of `off`, `auto`, `0..=16`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseCapPolicyError;

impl fmt::Display for ParseCapPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "expected `off`, `auto`, or a cap height 0..={MAX_CAP_HEIGHT}"
        )
    }
}

impl FromStr for CapPolicy {
    type Err = ParseCapPolicyError;

    /// `off` | `auto` | an integer `0..=MAX_CAP_HEIGHT` (`0` is `off`).
    /// Exact spellings only: no case folding, no whitespace.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "off" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            _ => {
                // `u8::from_str` accepts a leading `+`; the knob does not.
                if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(ParseCapPolicyError);
                }
                match s.parse::<u8>() {
                    Ok(0) => Ok(Self::Off),
                    Ok(c) if c as usize <= MAX_CAP_HEIGHT => Ok(Self::Fixed(c)),
                    _ => Err(ParseCapPolicyError),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle_tree::backends::types::BatchKeccak256Backend;
    use crate::merkle_tree::merkle::MerkleTree;
    use alloc::string::ToString;
    use math::field::{element::FieldElement, goldilocks::GoldilocksField};

    type F = GoldilocksField;
    type Fe = FieldElement<F>;
    type K = BatchKeccak256Backend<F>;
    type Node = [u8; 32];

    fn leaves(n: usize, salt: u64) -> Vec<Vec<Fe>> {
        (0..n as u64)
            .map(|i| vec![Fe::from(i * 7 + salt), Fe::from(i ^ 0x55 ^ salt)])
            .collect()
    }

    fn tree(n: usize, salt: u64) -> MerkleTree<K> {
        MerkleTree::<K>::build(&leaves(n, salt)).expect("non-empty")
    }

    /// The leaf data at position `p` of the padded tree (padding repeats the
    /// last leaf).
    fn leaf_at(data: &[Vec<Fe>], p: usize) -> &Vec<Fe> {
        &data[p.min(data.len() - 1)]
    }

    const LEAF_COUNTS: &[usize] = &[1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 512, 1024];

    // ---------------------------------------------------------------- primitive

    #[test]
    fn cap_is_the_heap_slice_and_hashes_to_the_root() {
        for &n in LEAF_COUNTS {
            let t = tree(n, 1);
            let d = t.depth().unwrap();
            assert_eq!(1usize << d, n.next_power_of_two(), "n={n}");
            for c in 0..=d {
                let cap = t.cap(c).unwrap();
                assert_eq!(cap.len(), 1 << c);
                assert_eq!(
                    &cap[..],
                    &t.nodes()[(1 << c) - 1..(2 << c) - 1],
                    "n={n} c={c}"
                );
                assert_eq!(cap_root::<K>(&cap), Some(t.root), "n={n} c={c}");
                assert!(verify_cap::<K>(&cap, &t.root, c), "n={n} c={c}");
            }
            assert_eq!(t.cap(0).unwrap(), vec![t.root]);
            assert!(t.cap(d + 1).is_none(), "c > depth must be None (n={n})");
        }
    }

    #[test]
    fn root_only_tree_has_no_depth_and_no_cap() {
        let t = MerkleTree::<K>::from_root([7u8; 32]);
        assert_eq!(t.depth(), None);
        assert!(t.cap(0).is_none());
    }

    /// Every leaf of every tree verifies against its cap node at every height,
    /// and at `c = 0` the capped check agrees with the full-path check.
    fn every_leaf_verifies(verify: impl Fn(&[Node], &[Node], usize, usize, Node) -> bool) -> bool {
        for &n in LEAF_COUNTS {
            let data = leaves(n, 2);
            let t = MerkleTree::<K>::build(&data).unwrap();
            let d = t.depth().unwrap();
            for c in 0..=d {
                let cap = t.cap(c).unwrap();
                for p in 0..(1usize << d) {
                    let mut proof = t.get_proof_by_pos(p).unwrap();
                    let full = proof.merkle_path.clone();
                    proof.truncate_to_cap(d, c).unwrap();
                    assert_eq!(proof.merkle_path.len(), d - c);
                    let leaf = K::hash_data(leaf_at(&data, p));
                    if !verify(&proof.merkle_path, &cap, d, p, leaf) {
                        return false;
                    }
                    if c == 0 {
                        assert!(verify_merkle_path_from_leaf_hash::<K>(
                            &full, &t.root, p, leaf
                        ));
                    }
                }
            }
        }
        true
    }

    fn real_verify(s: &[Node], cap: &[Node], d: usize, i: usize, l: Node) -> bool {
        verify_merkle_path_to_cap_from_leaf_hash::<K>(s, cap, d, i, l)
    }

    #[test]
    fn every_leaf_verifies_against_its_cap_node() {
        assert!(every_leaf_verifies(real_verify));
    }

    #[test]
    fn cap_taller_than_the_tree_is_refused_everywhere() {
        let t = tree(8, 3);
        let d = 3;
        let full = t.get_proof_by_pos(0).unwrap();
        let mut p = full.clone();
        assert_eq!(
            p.truncate_to_cap(d, d + 1),
            Err(CapError::CapTooTall {
                cap_height: 4,
                depth: 3
            })
        );
        let big_cap = vec![[0u8; 32]; 16];
        assert!(!verify_merkle_path_to_cap_from_leaf_hash::<K>(
            &[],
            &big_cap,
            d,
            0,
            [0u8; 32]
        ));
        assert!(split_owner_path(&big_cap, d, d + 1).is_none());
        let mut a = full.merkle_path.clone();
        assert!(embed_cap(&mut [&mut a], d, &big_cap).is_err());
        assert!(!verify_cap::<K>(&big_cap, &t.root, MAX_CAP_HEIGHT + 1));
    }

    #[test]
    fn truncate_refuses_a_path_that_is_not_full_length() {
        let t = tree(16, 4);
        let mut p = t.get_proof_by_pos(5).unwrap();
        p.truncate_to_cap(4, 2).unwrap();
        // Already cut: a second cut must not silently shorten it further.
        assert_eq!(
            p.truncate_to_cap(4, 2),
            Err(CapError::PathLength {
                expected: 4,
                got: 2
            })
        );
    }

    #[test]
    fn cap_root_needs_a_power_of_two() {
        assert!(cap_root::<K>(&[]).is_none());
        assert!(cap_root::<K>(&[[1u8; 32]; 3]).is_none());
        assert_eq!(cap_root::<K>(&[[1u8; 32]]), Some([1u8; 32]));
    }

    // ------------------------------------------------------ owner-path encoding

    #[test]
    fn split_owner_path_takes_exact_lengths_only() {
        let d: usize = 6;
        for c in 0..=d {
            let want = if c == 0 { d } else { d - c + (1 << c) };
            for len in want.saturating_sub(1)..=want + 1 {
                let path = vec![0u8; len];
                let got = split_owner_path(&path, d, c);
                if len == want {
                    let (s, cap) = got.unwrap();
                    assert_eq!(s.len(), d - c);
                    assert_eq!(cap.len(), if c == 0 { 0 } else { 1 << c });
                } else {
                    assert!(got.is_none(), "c={c} len={len}");
                }
            }
        }
    }

    #[test]
    fn embed_then_split_round_trips_and_every_opening_verifies() {
        let data = leaves(64, 5);
        let t = MerkleTree::<K>::build(&data).unwrap();
        let d = 6;
        let positions = [17usize, 3, 63, 0, 17];
        for c in 0..=d {
            let cap = t.cap(c).unwrap();
            let mut paths: Vec<Vec<Node>> = positions
                .iter()
                .map(|&p| t.get_proof_by_pos(p).unwrap().merkle_path)
                .collect();
            let full0 = paths[0].clone();
            {
                let mut refs: Vec<&mut Vec<Node>> = paths.iter_mut().collect();
                embed_cap(&mut refs, d, &cap).unwrap();
            }
            if c == 0 {
                assert_eq!(paths[0], full0, "c = 0 must be a no-op");
            } else {
                assert_eq!(paths[0].len(), d - c + (1 << c));
                assert_eq!(&paths[0][d - c..], &cap[..]);
            }
            for p in &paths[1..] {
                assert_eq!(p.len(), d - c);
            }
            let (check, owner) = CappedRoot::from_owner::<K>(&t.root, &paths[0], d, c).unwrap();
            assert_eq!(check.cap_height(), c);
            assert_eq!(owner.len(), d - c);
            assert!(check.verify::<K>(owner, positions[0], K::hash_data(&data[positions[0]])));
            for (path, &pos) in paths[1..].iter().zip(&positions[1..]) {
                assert!(check.verify::<K>(path, pos, K::hash_data(&data[pos])));
            }
        }
    }

    #[test]
    fn embed_refuses_short_paths_and_an_empty_owner_list() {
        let t = tree(16, 6);
        let cap = t.cap(2).unwrap();
        let mut short = vec![[0u8; 32]; 3];
        assert_eq!(
            embed_cap(&mut [&mut short], 4, &cap),
            Err(CapError::PathLength {
                expected: 4,
                got: 3
            })
        );
        assert_eq!(embed_cap::<Node>(&mut [], 4, &cap), Err(CapError::NoOwner));
        assert_eq!(
            embed_cap(&mut [&mut vec![[0u8; 32]; 4]], 4, &cap[..3]),
            Err(CapError::CapLength(3))
        );
    }

    // ------------------------------------------------------------- tamper tests

    struct Fixture {
        data: Vec<Vec<Fe>>,
        t: MerkleTree<K>,
        d: usize,
        c: usize,
    }

    fn fixture() -> Fixture {
        let data = leaves(256, 9);
        let t = MerkleTree::<K>::build(&data).unwrap();
        Fixture {
            data,
            t,
            d: 8,
            c: 3,
        }
    }

    impl Fixture {
        fn owner_path(&self, pos: usize) -> Vec<Node> {
            let mut p = self.t.get_proof_by_pos(pos).unwrap().merkle_path;
            embed_cap(&mut [&mut p], self.d, &self.t.cap(self.c).unwrap()).unwrap();
            p
        }
        fn path(&self, pos: usize) -> Vec<Node> {
            let mut p = self.t.get_proof_by_pos(pos).unwrap();
            p.truncate_to_cap(self.d, self.c).unwrap();
            p.merkle_path
        }
        fn leaf(&self, pos: usize) -> Node {
            K::hash_data(&self.data[pos])
        }
    }

    #[test]
    fn a_flipped_cap_byte_is_rejected() {
        let f = fixture();
        let honest = f.owner_path(10);
        assert!(CappedRoot::from_owner::<K>(&f.t.root, &honest, f.d, f.c).is_some());
        for k in 0..(1 << f.c) {
            let mut owner = honest.clone();
            owner[f.d - f.c + k][0] ^= 1;
            assert!(
                CappedRoot::from_owner::<K>(&f.t.root, &owner, f.d, f.c).is_none(),
                "k={k}"
            );
            assert!(!verify_cap::<K>(&owner[f.d - f.c..], &f.t.root, f.c));
        }
    }

    #[test]
    fn a_flipped_path_node_is_rejected() {
        let f = fixture();
        let owner = f.owner_path(10);
        let (check, _) = CappedRoot::from_owner::<K>(&f.t.root, &owner, f.d, f.c).unwrap();
        let honest = f.path(77);
        assert!(check.verify::<K>(&honest, 77, f.leaf(77)));
        for k in 0..honest.len() {
            let mut p = honest.clone();
            p[k][31] ^= 0x80;
            assert!(!check.verify::<K>(&p, 77, f.leaf(77)), "k={k}");
        }
    }

    #[test]
    fn swapped_cap_nodes_are_rejected() {
        let f = fixture();
        let mut owner = f.owner_path(10);
        let base = f.d - f.c;
        owner.swap(base, base + 5);
        assert!(CappedRoot::from_owner::<K>(&f.t.root, &owner, f.d, f.c).is_none());
    }

    #[test]
    fn a_cap_from_another_tree_is_rejected() {
        let f = fixture();
        let other = tree(256, 1234);
        let mut owner = f.path(10);
        owner.extend(other.cap(f.c).unwrap());
        assert!(CappedRoot::from_owner::<K>(&f.t.root, &owner, f.d, f.c).is_none());
        // And another tree's cap cannot vouch for this tree's openings even
        // when paired with that tree's own root.
        let (check, _) = CappedRoot::from_owner::<K>(&other.root, &owner, f.d, f.c).unwrap();
        assert!(!check.verify::<K>(&f.path(77), 77, f.leaf(77)));
    }

    #[test]
    fn an_index_with_a_flipped_top_bit_is_rejected() {
        let f = fixture();
        let owner = f.owner_path(10);
        let (check, _) = CappedRoot::from_owner::<K>(&f.t.root, &owner, f.d, f.c).unwrap();
        let pos = 77usize;
        let path = f.path(pos);
        assert!(check.verify::<K>(&path, pos, f.leaf(pos)));
        for bit in 0..f.d {
            let wrong = pos ^ (1 << bit);
            assert!(!check.verify::<K>(&path, wrong, f.leaf(pos)), "bit={bit}");
        }
        // Past the tree: an index ≥ 2^D is refused, not wrapped.
        assert!(!check.verify::<K>(&path, pos + (1 << f.d), f.leaf(pos)));
    }

    #[test]
    fn a_path_one_node_too_long_or_short_is_rejected() {
        let f = fixture();
        let owner = f.owner_path(10);
        let (check, _) = CappedRoot::from_owner::<K>(&f.t.root, &owner, f.d, f.c).unwrap();
        let path = f.path(77);
        assert!(!check.verify::<K>(&path[..path.len() - 1], 77, f.leaf(77)));
        let mut long = path.clone();
        long.push(path[0]);
        assert!(!check.verify::<K>(&long, 77, f.leaf(77)));
        // The full, uncut path is also refused under a cap.
        let full = f.t.get_proof_by_pos(77).unwrap().merkle_path;
        assert!(!check.verify::<K>(&full, 77, f.leaf(77)));
        // And the owner path one node short or long.
        assert!(CappedRoot::from_owner::<K>(&f.t.root, &owner[1..], f.d, f.c).is_none());
        let mut owner_long = owner.clone();
        owner_long.push(owner[0]);
        assert!(CappedRoot::from_owner::<K>(&f.t.root, &owner_long, f.d, f.c).is_none());
    }

    #[test]
    fn a_cap_moved_to_the_second_opening_is_rejected() {
        let f = fixture();
        let cap = f.t.cap(f.c).unwrap();
        // Query 0 carries a plain path, query 1 the cap: the wrong owner.
        let q0 = f.path(10);
        let mut q1 = f.path(77);
        q1.extend(cap.iter().copied());
        assert!(CappedRoot::from_owner::<K>(&f.t.root, &q0, f.d, f.c).is_none());
        // Even with a correctly authenticated cap in hand, query 1's path
        // (siblings + cap) is the wrong length for a non-owner.
        let owner = f.owner_path(10);
        let (check, _) = CappedRoot::from_owner::<K>(&f.t.root, &owner, f.d, f.c).unwrap();
        assert!(!check.verify::<K>(&q1, 77, f.leaf(77)));
    }

    #[test]
    fn a_proof_capped_at_one_height_fails_at_another() {
        let f = fixture();
        let owner = f.owner_path(10); // c = 3
        for other in [0, 1, 2, 4] {
            assert!(
                CappedRoot::from_owner::<K>(&f.t.root, &owner, f.d, other).is_none(),
                "c=3 proof accepted at c={other}"
            );
        }
    }

    #[test]
    fn uncapped_is_the_full_path_check_plus_exact_length() {
        let data = leaves(32, 11);
        let t = MerkleTree::<K>::build(&data).unwrap();
        let check = CappedRoot::uncapped(&t.root, 5);
        for (p, value) in data.iter().enumerate() {
            let path = t.get_proof_by_pos(p).unwrap().merkle_path;
            let leaf = K::hash_data(value);
            assert!(check.verify::<K>(&path, p, leaf));
            assert!(!check.verify::<K>(&path[..4], p, leaf));
            let mut long = path.clone();
            long.push(path[0]);
            assert!(!check.verify::<K>(&long, p, leaf));
        }
    }

    // ----------------------------------------------------------- mutation tests
    //
    // Each load-bearing check has a property function that the named test runs
    // against the real primitive, and a mutation test that runs the SAME
    // function against a copy with that one check removed and asserts it
    // fails. So removing the check from the real code makes the named test
    // fail: the check is shown to carry the property, not just to be present.

    /// A toy backend whose leaf hash is the identity, so an internal node can
    /// be presented as a leaf — the forgery a missing length check admits.
    struct IdentityLeaf;
    impl IsMerkleTreeBackend for IdentityLeaf {
        type Node = u64;
        type Data = u64;
        fn hash_data(leaf: &u64) -> u64 {
            *leaf
        }
        fn hash_new_parent(a: &u64, b: &u64) -> u64 {
            a.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17) ^ b.wrapping_add(0x0123_4567_89AB)
        }
    }

    type U64Verify = fn(&[u64], &[u64], usize, usize, u64) -> bool;

    /// A path one level short, whose "leaf" is really the internal node over
    /// leaves 0 and 1, must not verify at index 0.
    fn rejects_short_path_forgery(verify: U64Verify) -> bool {
        let d = 6;
        let c = 2;
        let data: Vec<u64> = (0..64u64).map(|i| i * 1_000_003 + 17).collect();
        let t = MerkleTree::<IdentityLeaf>::build(&data).unwrap();
        let cap = t.cap(c).unwrap();
        let full = t.get_proof_by_pos(0).unwrap().merkle_path;
        // Honest opening of leaf 0 verifies.
        assert!(verify(&full[..d - c], &cap, d, 0, data[0]));
        // The forgery: claim the parent of leaves 0 and 1 as the value at
        // index 0, with the path from that parent up to the cap.
        let internal = IdentityLeaf::hash_new_parent(&data[0], &data[1]);
        assert_ne!(
            internal, data[0],
            "the forged value must not be the real leaf"
        );
        !verify(&full[1..d - c], &cap, d, 0, internal)
    }

    #[test]
    fn exact_length_check_rejects_a_short_path_forgery() {
        assert!(rejects_short_path_forgery(
            verify_merkle_path_to_cap_from_leaf_hash::<IdentityLeaf>
        ));
    }

    #[test]
    fn mutation_without_the_length_check_admits_the_forgery() {
        fn mutant(s: &[u64], cap: &[u64], d: usize, i: usize, l: u64) -> bool {
            let c = cap.len().ilog2() as usize;
            // Mutation: no `siblings.len() == depth − c` check.
            verify_merkle_path_from_leaf_hash::<IdentityLeaf>(s, &cap[i >> (d - c)], i, l)
        }
        assert!(!rejects_short_path_forgery(mutant));
    }

    #[test]
    fn mutation_with_the_wrong_cap_index_fails_honest_openings() {
        fn mutant(s: &[Node], cap: &[Node], d: usize, i: usize, l: Node) -> bool {
            let c = cap.len().ilog2() as usize;
            if s.len() != d - c || c == d {
                // `d − c − 1` would underflow; keep the mutant defined there.
                return verify_merkle_path_to_cap_from_leaf_hash::<K>(s, cap, d, i, l);
            }
            // Mutation: `index >> (D − c − 1)` instead of `index >> (D − c)`.
            cap.get(i >> (d - c - 1))
                .is_some_and(|node| verify_merkle_path_from_leaf_hash::<K>(s, node, i, l))
        }
        assert!(!every_leaf_verifies(mutant));
    }

    /// A cap made from another tree, carried by the owner path next to the
    /// honest root, must not let a leaf of that other tree verify.
    /// `CappedRoot::from_owner`'s shape, so a mutant can stand in for it.
    type FromOwner = for<'a> fn(
        &'a Node,
        &'a [Node],
        usize,
        usize,
    ) -> Option<(CappedRoot<'a, Node>, &'a [Node])>;

    fn rejects_forged_cap(from_owner: FromOwner) -> bool {
        let d = 8;
        let c = 3;
        let honest = tree(256, 21);
        let forged_data = leaves(256, 99);
        let forged = MerkleTree::<K>::build(&forged_data).unwrap();
        let mut owner = forged.get_proof_by_pos(40).unwrap().merkle_path;
        embed_cap(&mut [&mut owner], d, &forged.cap(c).unwrap()).unwrap();
        match from_owner(&honest.root, &owner, d, c) {
            None => true,
            Some((check, siblings)) => {
                !check.verify::<K>(siblings, 40, K::hash_data(&forged_data[40]))
            }
        }
    }

    #[test]
    fn cap_to_root_check_rejects_a_forged_cap() {
        assert!(rejects_forged_cap(
            |r, p, d, c| CappedRoot::from_owner::<K>(r, p, d, c)
        ));
    }

    #[test]
    fn mutation_without_the_cap_to_root_check_admits_a_forged_cap() {
        fn mutant<'a>(
            _root: &'a Node,
            path: &'a [Node],
            d: usize,
            c: usize,
        ) -> Option<(CappedRoot<'a, Node>, &'a [Node])> {
            let (siblings, cap) = split_owner_path(path, d, c)?;
            // Mutation: no `verify_cap(cap, root)`.
            Some((
                CappedRoot {
                    depth: d,
                    cap_height: c,
                    cap,
                },
                siblings,
            ))
        }
        assert!(!rejects_forged_cap(mutant));
    }

    // ------------------------------------------- the only-rejecting-check fixtures
    //
    // REVIEW-CAP M1: a tamper that some OTHER check also rejects cannot show a
    // check is load-bearing — removing it leaves the test green. These two
    // fixtures are built so that exactly one check rejects them, on the real
    // keccak backend (no toy hash): delete that check and the test fails.

    /// Heap index of the ancestor at `height` levels above leaf `pos` in a tree
    /// of depth `d` (`height = 0` is the leaf itself).
    fn ancestor(d: usize, pos: usize, height: usize) -> usize {
        (1usize << (d - height)) - 1 + (pos >> height)
    }

    /// M1(a). The real internal node one level above leaf `pos`, presented as a
    /// "leaf hash" with the path from that node upward — one sibling short.
    /// At `pos = 0` and `pos = 2^D − 1` the index bits the fold consumes stay
    /// consistent after the shift (all 0 / all 1), so the length-agnostic fold
    /// ACCEPTS: only `siblings.len() == D − c` rejects it. Hash-agnostic — the
    /// node is read out of the tree, not forged — and at `c = 0` it is exactly
    /// the C1b case.
    #[test]
    fn an_internal_node_as_leaf_hash_is_rejected_only_by_the_length_check() {
        let t = tree(64, 5);
        let d = t.depth().unwrap();
        assert_eq!(d, 6);
        for c in 0..d {
            let cap = t.cap(c).unwrap();
            for pos in [0usize, (1 << d) - 1] {
                let full = t.get_proof_by_pos(pos).unwrap().merkle_path;
                let node = t.nodes()[ancestor(d, pos, 1)];
                let forged = &full[1..d - c];
                // The length-agnostic fold accepts the forgery: no other check
                // stands between it and acceptance.
                assert!(
                    verify_merkle_path_from_leaf_hash::<K>(forged, &cap[pos >> (d - c)], pos, node),
                    "c={c} pos={pos}: fixture precondition, the fold alone accepts"
                );
                // The real check refuses it.
                assert!(
                    !verify_merkle_path_to_cap_from_leaf_hash::<K>(forged, &cap, d, pos, node),
                    "c={c} pos={pos}: an internal node passed for a leaf"
                );
                if c == 0 {
                    assert!(!CappedRoot::uncapped(&t.root, d).verify::<K>(forged, pos, node));
                }
            }
        }
    }

    /// M1(b). Few openings under a tall cap: with 3 queries and `c = 3`, at
    /// least 5 of the 8 cap nodes are reached by no query. Flipping one of those
    /// leaves every per-query check green, so only the cap-to-root check
    /// (`verify_cap`, run by `from_owner`) rejects it.
    #[test]
    fn an_unreached_cap_node_is_rejected_only_by_the_cap_to_root_check() {
        let f = fixture();
        let queries = [10usize, 20, 30]; // all under cap node 0 (pos >> 5 == 0)
        let reached: Vec<usize> = queries.iter().map(|q| q >> (f.d - f.c)).collect();
        let unreached = (0..1usize << f.c)
            .find(|k| !reached.contains(k))
            .expect("some cap node is unreached");
        let mut owner = f.owner_path(queries[0]);
        owner[f.d - f.c + unreached][0] ^= 1;
        let (siblings0, tampered_cap) = split_owner_path(&owner, f.d, f.c).unwrap();
        // Every query still verifies against the tampered cap: no per-query
        // check sees the unreached node.
        assert!(verify_merkle_path_to_cap_from_leaf_hash::<K>(
            siblings0,
            tampered_cap,
            f.d,
            queries[0],
            f.leaf(queries[0])
        ));
        for &q in &queries[1..] {
            assert!(
                verify_merkle_path_to_cap_from_leaf_hash::<K>(
                    &f.path(q),
                    tampered_cap,
                    f.d,
                    q,
                    f.leaf(q)
                ),
                "q={q}: fixture precondition, per-query checks pass"
            );
        }
        // Only the cap-to-root check rejects it.
        assert!(!verify_cap::<K>(tampered_cap, &f.t.root, f.c));
        assert!(CappedRoot::from_owner::<K>(&f.t.root, &owner, f.d, f.c).is_none());
    }

    // ------------------------------------------------------------ policy pins

    #[test]
    fn auto_heights_are_pinned() {
        let deep = 30;
        for (openings, want) in [
            (0, 0),
            (1, 0),
            (3, 0),
            (4, 2),
            (19, 2),
            (20, 3),
            (110, 3),
            (112, 3),
            (224, 3),
            (10_000, 3),
        ] {
            assert_eq!(CapPolicy::Auto.height(openings, deep), want, "o={openings}");
        }
    }

    #[test]
    fn auto_never_goes_past_three_under_the_pinned_weights() {
        for o in 0..5_000 {
            assert!(CapPolicy::Auto.height(o, 40) <= 3, "o={o}");
        }
        // c = 4 loses to c = 3 on both the per-opening and the per-tree term.
        assert!(cap_gain(&AUTO_WEIGHTS, 1_000_000, 4) < cap_gain(&AUTO_WEIGHTS, 1_000_000, 3));
    }

    /// The argmax of the cost law over every height `0..=MAX_CAP_HEIGHT`, ties
    /// to the smaller height (`gain(·, 0) = 0`).
    fn cost_law_argmax(openings: usize, limit: usize) -> usize {
        let mut best = (0usize, 0i128);
        for c in 1..=limit {
            let g = cap_gain(&AUTO_WEIGHTS, openings, c);
            if g > best.1 {
                best = (c, g);
            }
        }
        best.0
    }

    /// REVIEW-CAP S5: `Auto` is RULINGS 1's table; this pins that the table is
    /// the cost-law argmax for every opening count, so the table and the
    /// weights cannot drift apart.
    #[test]
    fn the_auto_table_is_the_cost_law_argmax_at_every_opening_count() {
        for o in 0..=1_000_000usize {
            assert_eq!(
                CapPolicy::Auto.height(o, MAX_CAP_HEIGHT),
                cost_law_argmax(o, MAX_CAP_HEIGHT),
                "o={o}"
            );
        }
        for o in [usize::MAX, usize::MAX / 2, 1 << 40, u32::MAX as usize] {
            assert_eq!(CapPolicy::Auto.height(o, 64), 3, "o={o}");
            assert_eq!(cost_law_argmax(o, MAX_CAP_HEIGHT), 3, "o={o}");
        }
    }

    /// Clamping the table to the depth (RULINGS 1) is not the same function as
    /// an argmax bounded by the depth, at exactly one point: 4 openings of a
    /// depth-1 tree, where the table says 1 and the bounded argmax 0 (a c = 1
    /// cap loses 68 ns there). The table is the rule; this pins the one
    /// difference so any other one is a failure.
    #[test]
    fn the_depth_clamped_table_differs_from_a_bounded_argmax_at_one_point() {
        let mut diffs = Vec::new();
        for d in 0..=6usize {
            for o in 0..5_000usize {
                if CapPolicy::Auto.height(o, d) != cost_law_argmax(o, d.min(MAX_CAP_HEIGHT)) {
                    diffs.push((o, d));
                }
            }
        }
        assert_eq!(diffs, vec![(4, 1)]);
    }

    #[test]
    fn the_cost_law_is_bounded_for_every_input() {
        // No overflow at the `usize` extremes and the tallest height.
        let top = cap_gain(&AUTO_WEIGHTS, usize::MAX, MAX_CAP_HEIGHT);
        assert!(top < 0, "a height-16 cap loses at any opening count");
        assert!(cap_gain(&AUTO_WEIGHTS, usize::MAX, 3) > 0);
        // A height past the maximum is refused, never shifted.
        for c in [MAX_CAP_HEIGHT + 1, 127, 128, usize::MAX] {
            assert_eq!(cap_gain(&AUTO_WEIGHTS, 1_000, c), i128::MIN, "c={c}");
        }
    }

    #[test]
    fn heights_clamp_to_the_depth() {
        for d in 0..6 {
            assert_eq!(CapPolicy::Auto.height(110, d), d.min(3), "d={d}");
        }
        assert_eq!(CapPolicy::Fixed(5).height(110, 2), 2);
        assert_eq!(CapPolicy::Fixed(5).height(110, 9), 5);
        assert_eq!(CapPolicy::Fixed(16).height(1, 40), 16);
        assert_eq!(CapPolicy::Fixed(200).height(1, 40), MAX_CAP_HEIGHT);
        assert_eq!(
            CapPolicy::Fixed(5).height(0, 9),
            0,
            "an unopened tree has no cap"
        );
    }

    #[test]
    fn off_and_fixed_zero_are_zero_everywhere() {
        for o in 0..300 {
            for d in 0..24 {
                assert_eq!(CapPolicy::Off.height(o, d), 0);
                assert_eq!(CapPolicy::Fixed(0).height(o, d), 0);
            }
        }
        assert!(CapPolicy::Off.is_off());
        assert!(CapPolicy::Fixed(0).is_off());
        assert!(!CapPolicy::Auto.is_off());
        assert!(!CapPolicy::Fixed(1).is_off());
        assert_eq!(CapPolicy::default(), CapPolicy::Off);
    }

    #[test]
    fn auto_weights_are_pinned() {
        assert_eq!(
            AUTO_WEIGHTS,
            CapWeights {
                compress: 2251,
                select: 567,
                unpack: 528,
                hint: 460,
                compare: 3789,
            }
        );
        // The gains the pinned heights rest on (CAP.md §2).
        assert_eq!(cap_gain(&AUTO_WEIGHTS, 20, 2), 55_758);
        assert_eq!(cap_gain(&AUTO_WEIGHTS, 20, 3), 55_914);
        assert_eq!(cap_gain(&AUTO_WEIGHTS, 19, 2), 52_351);
        assert_eq!(cap_gain(&AUTO_WEIGHTS, 19, 3), 51_957);
        assert_eq!(cap_gain(&AUTO_WEIGHTS, 4, 1), -68);
        assert_eq!(cap_gain(&AUTO_WEIGHTS, 4, 2), 1_246);
    }

    #[test]
    fn policy_spellings_parse_and_print() {
        for (s, want) in [
            ("off", CapPolicy::Off),
            ("auto", CapPolicy::Auto),
            ("0", CapPolicy::Off),
            ("1", CapPolicy::Fixed(1)),
            ("16", CapPolicy::Fixed(16)),
        ] {
            assert_eq!(s.parse::<CapPolicy>(), Ok(want), "{s}");
        }
        for bad in [
            "", "17", "256", "-1", "+3", " 3", "3 ", "Auto", "OFF", "on", "3.0", "x",
        ] {
            assert!(bad.parse::<CapPolicy>().is_err(), "{bad:?} must be refused");
        }
        assert_eq!(CapPolicy::Off.to_string(), "off");
        assert_eq!(CapPolicy::Fixed(0).to_string(), "off");
        assert_eq!(CapPolicy::Auto.to_string(), "auto");
        assert_eq!(CapPolicy::Fixed(7).to_string(), "7");
        for p in [
            CapPolicy::Off,
            CapPolicy::Auto,
            CapPolicy::Fixed(1),
            CapPolicy::Fixed(16),
        ] {
            assert_eq!(p.to_string().parse::<CapPolicy>(), Ok(p));
        }
    }
}
