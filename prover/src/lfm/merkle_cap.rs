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

use super::builder::{Bit, Cell, Felt, LfmBuilder};
use super::edsl::{self, WrapDigest};
use super::instr::ArenaId;
use stark::config::Commitment;

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

/// The knob that chooses how a BLAKE3 opening is checked against its cap: over
/// PACKED digests ([`pack_digest`], the default), where the mux picks one word
/// a node instead of two (half its `Select` rows), or over both digest words
/// (`plain`, the rollback). Read by the in-guest verifier alone (the proof
/// does not move; the leaf programs do): see [`cap_mux_setting`].
pub const CAP_MUX_ENV: &str = "LAMBDA_VM_LFM_CAP_MUX";

/// [`CAP_MUX_ENV`]'s value: unset or `packed` muxes the packed word (the
/// default); `plain` muxes both digest words. Anything else is refused.
pub fn cap_mux_setting(value: Option<&str>) -> Result<bool, String> {
    match value.map(str::trim) {
        None | Some("packed") => Ok(true),
        Some("plain") => Ok(false),
        Some(other) => Err(format!("{CAP_MUX_ENV}={other:?}: expected plain or packed")),
    }
}

/// [`CAP_MUX_ENV`] in this process, read once; a refused value aborts the
/// process at its first read (a configuration error, before any program is
/// emitted), as the process format does.
pub fn cap_mux_packed() -> bool {
    static PACKED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PACKED.get_or_init(|| {
        let value = match std::env::var(CAP_MUX_ENV) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(v)) => {
                eprintln!("{CAP_MUX_ENV}={v:?}: expected plain or packed");
                std::process::abort()
            }
        };
        cap_mux_setting(value.as_deref()).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::abort()
        })
    })
}

#[cfg(test)]
thread_local! {
    /// Test only: on this thread, the cap mux is packed (`Some(true)`) or
    /// plain (`Some(false)`) whatever the knob says.
    pub(crate) static PACKED_MUX: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether a cap authenticated on `b` muxes packed digests: under BLAKE3
/// alone (two `u32`-lane words a digest), as [`cap_mux_packed`] says.
fn packs(b: &LfmBuilder) -> bool {
    if b.wrap_hash() != edsl::WrapHash::Blake3 {
        return false;
    }
    #[cfg(test)]
    if let Some(packed) = PACKED_MUX.with(std::cell::Cell::get) {
        return packed;
    }
    cap_mux_packed()
}

/// ★ A BLAKE3 digest's two words as ONE word: lanes `(l0..l7)` (each a `u32`,
/// as the hash chip writes an output and range-checks an input) become the
/// four felts `l_{2i} + 2^32 · l_{2i+1}`. Only the cap's compare reads it:
/// the walk's hashes and the index bits stay unpacked.
///
/// ⚠ Not injective at the top of the range: a felt is `lo + 2^32·hi mod p`,
/// so `(lo = a, hi = 0)` and `(lo = a + 1, hi = 2^32 − 1)` give the same felt
/// (their integers differ by `p`), and no other pair of `u32` pairs does. A
/// cap node therefore admits at most `2^4 = 16` packed-equal digests, and a
/// walk passes against a node it is not only by its hash's output equalling
/// one of those 16 fixed 256-bit values.
pub fn pack_digest(b: &mut LfmBuilder, d: WrapDigest) -> Cell {
    assert_eq!(d.len(), 2, "a BLAKE3 digest is two words");
    let two32 = b.felt_const(crate::tables::types::FE::from(1u64 << 32));
    let w0 = b.unpack(d[0]);
    let w1 = b.unpack(d[1]);
    let lanes = [w0[0], w0[1], w0[2], w0[3], w1[0], w1[1], w1[2], w1[3]];
    let felts: [Felt; 4] =
        core::array::from_fn(|i| b.mul_add(lanes[2 * i + 1], two32, lanes[2 * i]));
    b.pack_word(felts)
}

/// One tree's authenticated Merkle cap.
pub struct CapCells {
    cap: Vec<WrapDigest>,
    /// The cap height in the tree's own levels (4-ary under Poseidon1).
    height: usize,
    /// Index bits the mux reads, `log2(cap.len())`.
    mux_bits: usize,
    /// The cap's nodes packed ([`pack_digest`]) when this cap's mux is packed
    /// ([`packs`]): one word a node, the same nodes the root check read.
    packed: Option<Vec<Cell>>,
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
        let packed = packs(b).then(|| cap.iter().map(|node| pack_digest(b, *node)).collect());
        Self {
            cap: cap.to_vec(),
            height,
            mux_bits,
            packed,
        }
    }

    /// The cap height `c`, in the tree's own levels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Index bits the cap node's mux reads: an opening walks the others.
    pub fn mux_bits(&self) -> usize {
        self.mux_bits
    }

    /// ★ Authenticate EVERY leaf of a tree that lies wholly in this cap (no
    /// level walked: the cap is the leaf level), as a REFUSAL: leaf `j`'s
    /// digest must equal cap node `j`, which is what [`Self::verify_path`]
    /// muxes for index `j` with no walked bits. The positions are program
    /// constants, so no node is muxed.
    pub fn verify_every_leaf(&self, b: &mut LfmBuilder, leaves: &[WrapDigest]) {
        assert_eq!(
            leaves.len(),
            self.cap.len(),
            "one digest a leaf of a tree lying wholly in its cap"
        );
        for (leaf, node) in leaves.iter().zip(&self.cap) {
            for (x, y) in leaf.iter().zip(node.iter()) {
                edsl::assert_word_eq(b, *x, *y);
            }
        }
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
        self.verify_path_by(b, leaf, index_bits, siblings, false);
    }

    /// [`Self::verify_path`] with an arity-4 path's siblings in CHILD order, as
    /// the proof carries them ([`super::p1w16_emit::walk4_child`]); at arity 2
    /// it is [`Self::verify_path`].
    pub fn verify_path_child(
        &self,
        b: &mut LfmBuilder,
        leaf: WrapDigest,
        index_bits: &[Bit],
        siblings: &[WrapDigest],
    ) {
        self.verify_path_by(b, leaf, index_bits, siblings, true);
    }

    fn verify_path_by(
        &self,
        b: &mut LfmBuilder,
        leaf: WrapDigest,
        index_bits: &[Bit],
        siblings: &[WrapDigest],
        child_order: bool,
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
        let walked = if child_order && hash.arity() == 4 {
            super::p1w16_emit::walk4_child(b, leaf, walk_bits, siblings)
        } else {
            edsl::wrap_merkle_walk(b, leaf, walk_bits, siblings)
        };
        if let Some(packed) = &self.packed {
            // The walked digest against the packed node: one word muxed.
            let node = Self::select_words(b, packed.clone(), top_bits);
            let walked = pack_digest(b, walked);
            edsl::assert_word_eq(b, walked, node);
            return;
        }
        let node = self.select(b, top_bits);
        for (x, y) in walked.iter().zip(node.iter()) {
            edsl::assert_word_eq(b, *x, *y);
        }
    }

    /// The packed mux: `words[index]` from the top bits, low first, one
    /// `Select` row a pair a level (`words.len() − 1` in all).
    fn select_words(b: &mut LfmBuilder, words: Vec<Cell>, top_bits: &[Bit]) -> Cell {
        let mut level = words;
        for bit in top_bits {
            level = level
                .chunks_exact(2)
                .map(|pair| b.select(*bit, pair[0], pair[1]).0)
                .collect();
        }
        level[0]
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

/// One opening's path, cut to the cap, in the arena's order, and the cap
/// itself when this is the tree's owner path (its first opening in proof
/// order, of a capped tree).
///
/// At arity 2 the path to the cap is the first `D − c` siblings as the proof
/// carries them. At arity 4 the proof carries three siblings per 4-ary level in
/// child order (`3(⌈D/2⌉ − c)` of them, the owner path then the cap's nodes);
/// they are laid out in the walk's hint order for the opening at `index`
/// ([`super::p1w16_emit::hint_order`], which drops an uncapped odd-depth top's
/// two padding siblings), so an arity-4 path needs its leaf index.
pub fn path_to_cap(
    path: &[Commitment],
    depth: usize,
    c: usize,
    arity: usize,
    owner: bool,
    index: Option<usize>,
) -> Result<(Vec<Commitment>, Option<Vec<Commitment>>), String> {
    let check = |got: usize, want: usize, what: &str| {
        if got == want {
            Ok(())
        } else {
            Err(format!("{what}: {got} nodes, the shape takes {want}"))
        }
    };
    let (siblings, cap) = if c == 0 || !owner {
        let kept = if arity == 4 {
            3 * (crypto::merkle_tree::cap::tree_levels(depth, 4) - c)
        } else {
            depth - c
        };
        check(path.len(), kept, "a path to the cap")?;
        (path, None)
    } else {
        let (siblings, cap) =
            crypto::merkle_tree::cap::split_owner_path_arity(path, depth, c, arity)
                .ok_or("the owner path is the path to the cap, then the cap")?;
        (siblings, Some(cap.to_vec()))
    };
    if arity != 4 {
        return Ok((siblings.to_vec(), cap));
    }
    let index = index.ok_or("an arity-4 path is laid out at its leaf index")?;
    let walked = depth - mux_bits(depth, c, 4);
    check(
        siblings.len(),
        3 * walked.div_ceil(2),
        "three siblings per walked 4-ary level",
    )?;
    let triples: Vec<[Commitment; 3]> = siblings
        .chunks_exact(3)
        .map(|t| [t[0], t[1], t[2]])
        .collect();
    Ok((super::p1w16_emit::hint_order(index, walked, &triples), cap))
}

/// [`path_to_cap`] with an arity-4 path left in CHILD order (the proof's own,
/// [`super::p1w16_emit::walk4_child`]'s hints), so it needs no leaf index. An
/// uncapped odd-depth tree's top triple keeps only its first sibling, the
/// partner: the other two are the padding the walk supplies as constants.
pub fn child_path_to_cap(
    path: &[Commitment],
    depth: usize,
    c: usize,
    arity: usize,
    owner: bool,
) -> Result<(Vec<Commitment>, Option<Vec<Commitment>>), String> {
    if arity != 4 {
        return path_to_cap(path, depth, c, arity, owner, None);
    }
    let levels = crypto::merkle_tree::cap::tree_levels(depth, 4);
    let (siblings, cap) = if c == 0 || !owner {
        if path.len() != 3 * (levels - c) {
            return Err(format!(
                "a path to the cap: {} nodes, the shape takes {}",
                path.len(),
                3 * (levels - c)
            ));
        }
        (path, None)
    } else {
        let (siblings, cap) = crypto::merkle_tree::cap::split_owner_path_arity(path, depth, c, 4)
            .ok_or("the owner path is the path to the cap, then the cap")?;
        (siblings, Some(cap.to_vec()))
    };
    let walked = depth - mux_bits(depth, c, 4);
    let mut out = siblings.to_vec();
    if walked % 2 == 1 {
        // The uncapped odd top: drop its two padding siblings.
        out.truncate(out.len() - 2);
    }
    debug_assert_eq!(out.len(), super::p1w16_emit::path_hints(walked));
    Ok((out, cap))
}
