//! ★ One tree's authenticated Merkle cap, in-guest — the one gadget the WHIR
//! chain verifier (W1) and the STARK sub-proof verifier (S1) share
//! (design/CAP.md §6.1, §6.2, §9.2; REVIEW-CAP S1).
//!
//! A tree of depth `D` committed with a height-`c` cap is authenticated in two
//! places, and the in-guest verifier makes the dangerous state of each
//! unconstructible rather than checked:
//!
//! - **Once per tree**, [`CapCells::authenticate`] hashes the `2^c` hinted cap
//!   digests up to their root and asserts it equals the tree's root lanes. It
//!   is the ONLY constructor, so every [`CapCells`] value is a cap that hashes
//!   to its root — and a tree has exactly one: the cells checked against the
//!   root and the cells the mux reads are the same cells (REVIEW-CAP (e)).
//! - **Per opening**, [`CapCells::verify_path`] is the ONE entry point. It takes
//!   the opened leaf, the tree's WHOLE leaf index (low bit first, one bit per
//!   level) and the path to the cap, walks the low `D − c` bits, picks
//!   `cap[index >> (D − c)]` with the top `c` bits and asserts the two digests
//!   equal. The split point is computed here from the index's own length and
//!   the cap's height; the mux is private, so no caller can feed it a constant,
//!   a hinted bit or a sub-slice of its own choosing (REVIEW-CAP (d)).
//!
//! The mux is a balanced tree of `2^c − 1` `Select`s per digest cell: the LFM
//! has no load at a computed address, which is why the cap height is priced by
//! the cost law and stays at most 3 (RULINGS 1).
//!
//! ⚠ What a caller still owes: `index_bits` must be the tree's own leaf index
//! as the TRANSCRIPT produced it — the query's bits, or a suffix of them for a
//! tree whose leaves cover several positions (a FRI layer). Those bits reach
//! every caller as cells of the one `sample_u64_pow2` decomposition; nothing
//! here can tell a transcript bit from a hinted one.

use super::builder::{Bit, Felt, LfmBuilder};
use super::edsl::{self, WrapDigest};
use super::instr::ArenaId;

/// One tree's authenticated Merkle cap.
pub struct CapCells {
    cap: Vec<WrapDigest>,
    height: usize,
}

impl CapCells {
    /// Authenticate a hinted cap against a tree's root lanes, once per tree.
    ///
    /// `cap` must be `2^c` digests, `c ≥ 1`: a tree at `c = 0` has no cap and
    /// is checked against its root. `root_lanes` holds one entry per digest
    /// cell (one for an algebraic root, two for a byte digest), as
    /// [`edsl::assert_digest_eq_lanes`] takes them.
    pub fn authenticate(b: &mut LfmBuilder, cap: &[WrapDigest], root_lanes: &[[Felt; 4]]) -> Self {
        assert!(
            cap.len() >= 2 && cap.len().is_power_of_two(),
            "a cap is 2^c digests with c >= 1, got {}",
            cap.len()
        );
        let root = edsl::wrap_merkle_tree_root(b, cap);
        edsl::assert_digest_eq_lanes(b, root, root_lanes);
        Self {
            cap: cap.to_vec(),
            height: cap.len().trailing_zeros() as usize,
        }
    }

    /// The cap height `c`.
    pub fn height(&self) -> usize {
        self.height
    }

    /// ★ Authenticate one opening against this cap, as a REFUSAL: `leaf` is the
    /// opened leaf's digest, `index_bits` the tree's WHOLE leaf index (low
    /// first, `D` bits) and `siblings` the path to the cap (`D − c` digests,
    /// leaf level first). The low `D − c` bits are walked, the top `c` pick the
    /// cap node, and the walked digest must equal it.
    pub fn verify_path(
        &self,
        b: &mut LfmBuilder,
        leaf: WrapDigest,
        index_bits: &[Bit],
        siblings: &[WrapDigest],
    ) {
        assert_eq!(
            siblings.len() + self.height,
            index_bits.len(),
            "a path to the cap: one sibling per level below it"
        );
        let (walk_bits, top_bits) = index_bits.split_at(siblings.len());
        let walked = edsl::wrap_merkle_walk(b, leaf, walk_bits, siblings);
        let node = self.select(b, top_bits);
        for (x, y) in walked.iter().zip(node.iter()) {
            edsl::assert_word_eq(b, *x, *y);
        }
    }

    /// `cap[index >> (depth − c)]` from the index's top `c` bits, LOW first:
    /// a balanced mux, `2^c − 1` `Select` rows a digest cell. Pairs are
    /// `(2t, 2t + 1)` because the bits arrive low first (the slot mux's
    /// reason, `whir_chain::emit_slot_mux`).
    fn select(&self, b: &mut LfmBuilder, top_bits: &[Bit]) -> WrapDigest {
        assert_eq!(top_bits.len(), self.height, "one mux level per cap level");
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

/// Hint a height-`c` cap — `2^c` digests at [`edsl::digest_words`] words each —
/// out of `arena` from word `base`, and authenticate it against `root_lanes`.
/// Returns the cells and the next free word.
pub fn hint_and_authenticate(
    b: &mut LfmBuilder,
    arena: ArenaId,
    base: u32,
    c: usize,
    root_lanes: &[[Felt; 4]],
) -> (CapCells, u32) {
    let dw = edsl::digest_words(b);
    let mut cursor = base;
    let cap: Vec<WrapDigest> = (0..1usize << c)
        .map(|_| {
            let d = edsl::hint_digest(b, arena, cursor);
            cursor += dw;
            d
        })
        .collect();
    (CapCells::authenticate(b, &cap, root_lanes), cursor)
}

/// Permutations one tree's cap check costs: the cap hashed up to its root,
/// `2^c − 1` parents. Nothing at `c = 0`.
pub const fn cap_root_permutations(c: usize) -> usize {
    (1usize << c) - 1
}
