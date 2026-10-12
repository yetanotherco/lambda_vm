//! ★ One tree's authenticated Merkle cap, in-guest — the one gadget the WHIR
//! chain verifier (W1) and the STARK sub-proof verifier (S1) share.
//!
//! A tree of depth `D` committed with a height-`c` cap is authenticated in two
//! places, and the in-guest verifier makes the dangerous state of each
//! unconstructible rather than checked:
//!
//! - **Once per tree**, [`CapCells::authenticate`] hashes the `2^c` hinted cap
//!   digests up to their root and asserts it equals the tree's root lanes. It
//!   is the ONLY constructor, so every [`CapCells`] value is a cap that hashes
//!   to its root — and a tree has exactly one: the cells checked against the
//!   root and the cells the mux reads are the same cells.
//! - **Per opening**, [`CapCells::verify_path`] is the ONE entry point. It takes
//!   the opened leaf, the tree's WHOLE leaf index (low bit first, one bit per
//!   level) and the path to the cap, walks the low `D − c` bits, picks
//!   `cap[index >> (D − c)]` with the top `c` bits and asserts the two digests
//!   equal. The split point is computed here from the index's own length and
//!   the cap's height; the mux is private, so no caller can feed it a constant,
//!   a hinted bit or a sub-slice of its own choosing.
//!
//! The mux is a balanced tree of `2^c − 1` `Select`s per digest cell: the LFM
//! has no load at a computed address, which is why the cap height is priced by
//! the cost law and stays at most 3.
//!
//! **Arity 4** (a [`edsl::WrapHash::Poseidon1`] builder, the host's 4-ary
//! trees). `D` stays the binary depth, the tree has `h = ⌈D/2⌉` levels and `c`
//! counts them; the cap is the level `c` below the root, real nodes only:
//! [`cap_nodes`] = `4^c`, or `2·4^(c−1)` at odd `D` (the top group is two real
//! nodes and two zero digests). Both are powers of two, so the same mux picks
//! the node with the top [`mux_bits`] = `log2(cap_nodes)` bits; the walk takes
//! the low `D − mux_bits = 2(h − c)` bits, whole 4-ary levels of three hints
//! each, and the cap's root is the 4-ary build with `Merkle4`'s zero padding
//! (`p1w16_emit::tree_root4`, the host's `cap_root`).
//!
//! ⚠ What a caller still owes: `index_bits` must be the tree's own leaf index
//! as the TRANSCRIPT produced it — the query's bits, or a suffix of them for a
//! tree whose leaves cover several positions (a FRI layer). Those bits reach
//! every caller as cells of the one `sample_u64_pow2` decomposition; nothing
//! here can tell a transcript bit from a hinted one.

use super::builder::{Bit, Felt, LfmBuilder};
use super::edsl::{self, WrapDigest};
use super::instr::ArenaId;

/// Nodes in the height-`c` cap of a tree of `arity` children per node over
/// `2^depth` leaves: `2^c` at arity 2, the real nodes of the level `c` below
/// the root at arity 4 (`crypto`'s `cap_len`, the host's own count); `1` (the
/// root) at `c = 0`.
pub fn cap_nodes(depth: usize, c: usize, arity: usize) -> usize {
    crypto::merkle_tree::cap::cap_len(depth, c, arity)
        .unwrap_or_else(|| panic!("a height-{c} cap over depth {depth} at arity {arity}"))
}

/// Index bits the cap's mux reads: `log2(cap_nodes)` (`c` at arity 2, none
/// uncapped).
pub fn mux_bits(depth: usize, c: usize, arity: usize) -> usize {
    cap_nodes(depth, c, arity).trailing_zeros() as usize
}

/// Sibling digests one opening's path to the cap carries: one per walked
/// level at arity 2 (`D − c`), three per walked 4-ary level at arity 4 and one
/// at an uncapped odd-depth top (`p1w16_emit::path_hints` over the walked
/// bits).
pub fn path_hints(depth: usize, c: usize, arity: usize) -> usize {
    let walked = depth - mux_bits(depth, c, arity);
    if arity == 4 {
        super::p1w16_emit::path_hints(walked)
    } else {
        walked
    }
}

/// Node hashes one opening's walk to the cap costs: one per walked level
/// (`⌈walked bits / 2⌉` at arity 4).
pub fn path_permutations(depth: usize, c: usize, arity: usize) -> usize {
    let walked = depth - mux_bits(depth, c, arity);
    if arity == 4 {
        walked.div_ceil(2)
    } else {
        walked
    }
}

/// Node hashes a cap of `nodes` costs up to its root: `nodes − 1` at arity 2,
/// `⌈n/4⌉` per level at arity 4 (a short group padded). None for `nodes = 1`.
pub fn cap_root_permutations(nodes: usize, arity: usize) -> usize {
    if arity != 4 {
        return nodes - 1;
    }
    let (mut n, mut perms) = (nodes, 0);
    while n > 1 {
        n = n.div_ceil(4);
        perms += n;
    }
    perms
}

/// One tree's authenticated Merkle cap.
pub struct CapCells {
    cap: Vec<WrapDigest>,
    /// The cap height in the tree's own levels (4-ary under Poseidon1).
    height: usize,
    /// Index bits the mux reads, `log2(cap.len())`.
    mux_bits: usize,
}

impl CapCells {
    /// Authenticate a hinted binary cap against a tree's root lanes, once per
    /// tree.
    ///
    /// `cap` must be `2^c` digests, `c ≥ 1`: a tree at `c = 0` has no cap and
    /// is checked against its root. `root_lanes` holds one entry per digest
    /// cell (one for an algebraic root, two for a byte digest), as
    /// [`edsl::assert_digest_eq_lanes`] takes them. An arity-4 cap's height
    /// depends on its tree's depth: [`Self::authenticate_at`].
    pub fn authenticate(b: &mut LfmBuilder, cap: &[WrapDigest], root_lanes: &[[Felt; 4]]) -> Self {
        assert_eq!(
            b.wrap_hash().arity(),
            2,
            "an arity-4 cap names its height (authenticate_at)"
        );
        let height = cap.len().trailing_zeros() as usize;
        Self::authenticate_at(b, cap, height, root_lanes)
    }

    /// [`Self::authenticate`] for a cap of `height` levels of the builder's
    /// arity: `2^height` digests at arity 2, `4^height` or `2·4^(height−1)`
    /// at arity 4 ([`cap_nodes`]). The root is the arity's own build.
    pub fn authenticate_at(
        b: &mut LfmBuilder,
        cap: &[WrapDigest],
        height: usize,
        root_lanes: &[[Felt; 4]],
    ) -> Self {
        assert!(
            height >= 1 && cap.len() >= 2 && cap.len().is_power_of_two(),
            "a cap is 2^k digests at height c >= 1, got {} at height {height}",
            cap.len()
        );
        let mux_bits = cap.len().trailing_zeros() as usize;
        let fits = match b.wrap_hash().arity() {
            4 => mux_bits == 2 * height || mux_bits + 1 == 2 * height,
            _ => mux_bits == height,
        };
        assert!(
            fits,
            "{} cap nodes are not a height-{height} cap at arity {}",
            cap.len(),
            b.wrap_hash().arity()
        );
        let root = edsl::wrap_merkle_tree_root(b, cap);
        edsl::assert_digest_eq_lanes(b, root, root_lanes);
        Self {
            cap: cap.to_vec(),
            height,
            mux_bits,
        }
    }

    /// The cap height `c`, in the tree's own levels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// ★ Authenticate one opening against this cap, as a REFUSAL: `leaf` is the
    /// opened leaf's digest, `index_bits` the tree's WHOLE leaf index (low
    /// first, `D` bits) and `siblings` the path to the cap ([`path_hints`]
    /// digests, leaf level first). The low `D − mux_bits` bits are walked, the
    /// top `mux_bits` pick the cap node, and the walked digest must equal it.
    pub fn verify_path(
        &self,
        b: &mut LfmBuilder,
        leaf: WrapDigest,
        index_bits: &[Bit],
        siblings: &[WrapDigest],
    ) {
        assert!(
            index_bits.len() >= self.mux_bits,
            "the index covers the cap: {} bits under a {}-bit mux",
            index_bits.len(),
            self.mux_bits
        );
        let walked = index_bits.len() - self.mux_bits;
        let hash = b.wrap_hash();
        assert!(
            hash.arity() != 4 || walked.is_multiple_of(2),
            "an arity-4 cap sits on a 4-ary level: {walked} walked bits"
        );
        assert_eq!(
            siblings.len(),
            hash.path_hints(walked),
            "a path to the cap: its hints for every level below it"
        );
        let (walk_bits, top_bits) = index_bits.split_at(walked);
        let walked = edsl::wrap_merkle_walk(b, leaf, walk_bits, siblings);
        let node = self.select(b, top_bits);
        for (x, y) in walked.iter().zip(node.iter()) {
            edsl::assert_word_eq(b, *x, *y);
        }
    }

    /// `cap[index >> (depth − mux_bits)]` from the index's top bits, LOW
    /// first: a balanced mux, `cap.len() − 1` `Select` rows a digest cell.
    /// Pairs are `(2t, 2t + 1)` because the bits arrive low first (the slot
    /// mux's reason, `whir_chain::emit_slot_mux`).
    fn select(&self, b: &mut LfmBuilder, top_bits: &[Bit]) -> WrapDigest {
        assert_eq!(top_bits.len(), self.mux_bits, "one mux level per index bit");
        let mut level: Vec<WrapDigest> = self.cap.clone();
        for bit in top_bits {
            level = level
                .chunks_exact(2)
                .map(|pair| {
                    let cells: Vec<_> = pair[0]
                        .iter()
                        .zip(pair[1].iter())
                        .map(|(l, r)| b.select(*bit, *l, *r).0)
                        .collect();
                    WrapDigest::from_cells(&cells)
                })
                .collect();
        }
        level[0]
    }
}

/// Hint the height-`c` cap of a tree over `2^depth` leaves — [`cap_nodes`]
/// digests at the builder's arity, [`edsl::digest_words`] words each — out of
/// `arena` from word `base`, and authenticate it against `root_lanes`.
/// Returns the cells and the next free word.
pub fn hint_and_authenticate(
    b: &mut LfmBuilder,
    arena: ArenaId,
    base: u32,
    c: usize,
    depth: usize,
    root_lanes: &[[Felt; 4]],
) -> (CapCells, u32) {
    let dw = edsl::digest_words(b);
    let nodes = cap_nodes(depth, c, b.wrap_hash().arity());
    let mut cursor = base;
    let cap: Vec<WrapDigest> = (0..nodes)
        .map(|_| {
            let d = edsl::hint_digest(b, arena, cursor);
            cursor += dw;
            d
        })
        .collect();
    (CapCells::authenticate_at(b, &cap, c, root_lanes), cursor)
}
