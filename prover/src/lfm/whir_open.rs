//! A WHIR query opening as a machine leg — the hashing half of the chain's
//! dominant term.
//!
//! `whir_commit::verify_opening` (`crypto/multilinear/src/whir_commit.rs:416`)
//! is one line: `opening.proof.verify::<Backend<F, H>>(root, index, &values)`.
//! Under the RPX pin that unfolds into exactly two primitives this machine
//! already has —
//!
//! - the LEAF is `sponge_leaf` over the block's felts
//!   (`merkle_tree/backends/rpx.rs:80-87`), which is
//!   [`WrapHash::Algebraic`](super::edsl::WrapHash)'s leaf hash; and
//! - a PARENT is `compress`, one permutation of `[left ‖ right ‖ 0⁴]`
//!   (`:116-126`), which is `b.compress`
//!
//! — walked by `verify_merkle_path_from_leaf_hash`
//! (`crypto/crypto/src/merkle_tree/proof.rs:30-50`): level by level, the
//! current node is the LEFT child when the index's low bit is zero, then the
//! index shifts right. That is [`edsl::wrap_merkle_walk`]'s contract verbatim,
//! so the walk is a reuse and not a port.
//!
//! # ⚠ The felt decomposition, and why it is not a choice
//!
//! The leaf hashes `element_felts` of each value in order (`rpx/mod.rs:470`),
//! which streams the element's big-endian bytes: ONE felt for a base element,
//! THREE for a cubic extension element in coordinate order. Coordinate order is
//! the order `unpack` returns lanes 0–2 in, which is why an extension value
//! costs one `Unpack` here and no byte work at all.
//!
//! ★ **The capacity is keyed on the FELT count, not the value count.** A block
//! of sixteen extension values is forty-eight felts, and `leaf_capacity` reads
//! `len % 8` off that (`rpx/mod.rs:349-354`). The two counts agree modulo eight
//! for a sixteen-wide block and DISAGREE for the two-wide tail block the last
//! round opens (six felts against two values), which is the shape that catches
//! the mistake.
//!
//! # ★ Why a round-0 base value may be lifted into the fold, and what pins it
//!
//! Round 0's current codeword is base-field (`whir_chain.rs:983` rejects any
//! other arrangement), so its opened values arrive as hinted felts and the fold
//! then treats them as extension elements through [`Felt::as_ext`]. That lift
//! is only sound if lanes 1–3 of the hinted word are zero, and a hint
//! constrains nothing: `hint_felt` is `hint_word` retyped
//! (`builder.rs:632-634`) and the arena chip sends a full `word_token`
//! (`chips.rs:105`).
//!
//! What pins them is the leaf hash itself, and it is pinned TWICE by two
//! independent mechanisms. `algebraic_leaf_hash` feeds every value to
//! `pack_word`; in the AIR, `Pack` RECEIVES each lane as a `base_token`,
//! `(addr, v, 0, 0, 0)` with three tuple-constant zeros (`chips.rs:64-66`,
//! `:1934-1937`), and the memory bus is a multiset, so the arena's send of
//! `(addr, l0, l1, l2, l3)` can only balance against that receive when
//! `l1 = l2 = l3 = 0` — that is the one that carries the soundness. In the
//! executor, the same `Pack` reads each lane through `read_base`, which refuses
//! a word with a nonzero upper lane outright (`executor.rs:871` → `:398-401`,
//! `LfmExecError::NotBaseWord`) — that is the one a test can watch, and
//! `whir_open_tests::the_leaf_pins_a_hinted_base_value_to_its_low_lane` names
//! the variant rather than settling for `is_err`. The block is leaf-hashed in
//! full by construction here, so every base value the fold reads has been
//! pinned before it reads it.
//!
//! ⚠ The obligation that follows, written down because it is exactly the kind a
//! later caller can forget: **a base value that is NOT part of a leaf this leg
//! hashes is not pinned, and must not be lifted.** The bad state is unreachable
//! only while the values folded and the values hashed are the same slice.

use super::builder::{Bit, Ext, Felt, LfmBuilder};
use super::edsl::{self, WrapDigest};

/// Base felts a cubic extension element decomposes to.
const FELTS_PER_EXT: usize = 3;

/// The sponge's rate, in felts — the block size `leaf_capacity` is keyed on.
///
/// Shared with [`super::whir_transcript`], whose sponge is the same one: a
/// second spelling of the rate is a second thing to keep in step.
pub(super) const RATE_FELTS: usize = 8;

/// Felts a `Pack` assembles into one sponge word.
const FELTS_PER_WORD: usize = 4;

/// A queried block's values, in the field the round holds them in.
///
/// Two variants rather than one slice of `Ext`, because the FIELD is what the
/// leaf hash reads: a base block is one felt per value and an extension block
/// is three, and hashing sixteen base values as though they were extension
/// elements produces a digest no committer ever computed.
#[derive(Clone, Copy)]
pub enum BlockValues<'a> {
    /// Round 0's current codeword, before any extension challenge has touched
    /// it — `RoundOpenings::Base` on the host.
    Base(&'a [Felt]),
    /// Every successor block, and every current block after round 0.
    Ext(&'a [Ext]),
}

impl BlockValues<'_> {
    /// Values in the block: `2^k` for a fold of `k` variables.
    pub fn len(&self) -> usize {
        match self {
            BlockValues::Base(v) => v.len(),
            BlockValues::Ext(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Base felts the leaf hashes — the number `leaf_capacity` is keyed on.
    pub fn felts(&self) -> usize {
        match self {
            BlockValues::Base(v) => v.len(),
            BlockValues::Ext(v) => FELTS_PER_EXT * v.len(),
        }
    }
}

/// INSTRUCTIONS [`emit_block_leaf`] emits over a block of `felts` felts reached
/// through `unpacks` `Unpack` rows.
///
/// Every term by its shape: one `Unpack` per EXTENSION value and none for a
/// base one (a base value is already a lane); `ceil(felts/4)` `Pack` rows, one
/// per sponge word, the tail zero-padded; and `ceil(words/2)` permutations,
/// because a rate block is two words. The two ceilings compose to
/// `ceil(felts/8)` at every width.
pub const fn block_leaf_rows(felts: usize, unpacks: usize) -> usize {
    if felts == 0 {
        // `sponge_leaf` returns the zero digest without permuting, and so does
        // the emitter. Unreachable from a chain — a block is `2^k` values with
        // `k >= 1` — and stated rather than left to underflow.
        return unpacks;
    }
    unpacks + felts.div_ceil(FELTS_PER_WORD) + felts.div_ceil(RATE_FELTS)
}

/// INSTRUCTIONS [`emit_verify_opening`] emits for one query.
///
/// The leaf, then the walk at two rows a level — one `Select` and one
/// `compress`, because an algebraic digest is ONE cell and the whole digest
/// swaps on the same bit (`edsl.rs:756-771`) — then the root comparison, which
/// is one `Unpack` of the walked cell and four lowered asserts at two rows each
/// (`edsl.rs:846-851`, `builder.rs:283-287`). The root's own lanes are unpacked
/// once by the caller and shared across every query against that root, so they
/// are not charged here.
pub const fn verify_opening_rows(felts: usize, unpacks: usize, depth: usize) -> usize {
    verify_opening_rows_capped(felts, unpacks, depth, 0)
}

/// [`verify_opening_rows`] against a tree capped at height `cap`
/// ([`CapCells`]): the walk stops `cap` levels short (`2·cap` rows fewer),
/// the cap mux picks the node with `2^cap − 1` `Select` rows, and the
/// comparison is of two VARIABLE cells, so it unpacks both (one `Unpack` more
/// than against the hoisted root lanes). At `cap = 0` it is the root form.
pub const fn verify_opening_rows_capped(
    felts: usize,
    unpacks: usize,
    depth: usize,
    cap: usize,
) -> usize {
    let leaf = block_leaf_rows(felts, unpacks);
    let walk = 2 * (depth - cap);
    let compare = 1 + 2 * FELTS_PER_WORD;
    if cap == 0 {
        leaf + walk + compare
    } else {
        leaf + walk + ((1usize << cap) - 1) + 1 + compare
    }
}

/// PERMUTATIONS one query's opening costs: the leaf's blocks plus one parent a
/// level. This is the term the chain's cost is dominated by, and it is a
/// function of the block's felts and the tree's depth alone — no row
/// bookkeeping enters it.
pub const fn verify_opening_perms(felts: usize, depth: usize) -> usize {
    verify_opening_perms_capped(felts, depth, 0)
}

/// [`verify_opening_perms`] against a tree capped at height `cap`: `cap`
/// parents fewer. The cap's own `2^cap − 1` parents are paid once per TREE,
/// by [`cap_check_perms`].
pub const fn verify_opening_perms_capped(felts: usize, depth: usize, cap: usize) -> usize {
    felts.div_ceil(RATE_FELTS) + depth - cap
}

/// PERMUTATIONS one tree's cap check costs: the cap hashed up to its root,
/// `2^cap − 1` parents. Nothing at `cap = 0`.
pub const fn cap_check_perms(cap: usize) -> usize {
    (1usize << cap) - 1
}

/// INSTRUCTIONS one tree's cap check costs beyond its hinted words: the
/// `2^cap − 1` parents (one `compress` each) and the root comparison (one
/// `Unpack` and four lowered asserts). Nothing at `cap = 0`: an uncapped tree
/// is compared against its root lanes query by query.
pub const fn cap_check_rows(cap: usize) -> usize {
    if cap == 0 {
        0
    } else {
        cap_check_perms(cap) + 1 + 2 * FELTS_PER_WORD
    }
}

/// ★ The block's Merkle leaf: `sponge_leaf` over its felts.
///
/// Goes through [`edsl::wrap_leaf_hash`] rather than restating the duplex, so
/// the leaf this authenticates against and the leaf the commitment builder
/// computes have one definition between them.
pub fn emit_block_leaf(b: &mut LfmBuilder, values: BlockValues<'_>) -> WrapDigest {
    match values {
        BlockValues::Base(felts) => edsl::wrap_leaf_hash(b, felts),
        BlockValues::Ext(cells) => {
            let mut felts = Vec::with_capacity(FELTS_PER_EXT * cells.len());
            for value in cells {
                let lanes = b.unpack(value.as_cell());
                felts.extend_from_slice(&lanes[..FELTS_PER_EXT]);
            }
            edsl::wrap_leaf_hash(b, &felts)
        }
    }
}

/// ★ `whir_commit::verify_opening`, emitted — as a REFUSAL rather than a bool.
///
/// The host returns `false` and its caller turns that into
/// `Error::OpeningRejected`. Here the comparison IS the rejection:
/// `assert_digest_eq_lanes` lowers to `diff / 0`, which has no satisfying
/// assignment unless the two agree (`builder.rs:283-292`), so an opening that
/// does not authenticate has no execution rather than a `false` somebody could
/// forget to branch on.
///
/// `index_bits` are the leaf index low-to-high with one bit per level, which is
/// both what `sample_u64_pow2` hands back and what the walk consumes; `depth`
/// is therefore `index_bits.len()` and is not passed separately. `root_lanes`
/// is the root unpacked ONCE by the caller: a round opens `Q` blocks against
/// the same root, and unpacking it per query would charge `Q − 1` rows for a
/// value that never changes.
pub fn emit_verify_opening(
    b: &mut LfmBuilder,
    values: BlockValues<'_>,
    index_bits: &[Bit],
    siblings: &[WrapDigest],
    root_lanes: &[Felt; 4],
) {
    assert_eq!(
        index_bits.len(),
        siblings.len(),
        "one sibling and one index bit per level of the tree"
    );
    let leaf = emit_block_leaf(b, values);
    let walked = edsl::wrap_merkle_walk(b, leaf, index_bits, siblings);
    edsl::assert_digest_eq_lanes(b, walked, std::slice::from_ref(root_lanes));
}

/// ★ One tree's authenticated Merkle cap — the gadget the STARK verifier
/// shares ([`super::merkle_cap`]): its only constructor checks the cap against
/// the tree's root, and its one entry point walks, muxes and compares.
pub use super::merkle_cap::CapCells;

/// How one tree's openings are authenticated in-guest: against its root
/// lanes (no cap — today's emission, instruction for instruction), or
/// against its authenticated [`CapCells`].
pub enum TreeAuth {
    Root([Felt; 4]),
    Cap(CapCells),
}

impl TreeAuth {
    /// The cap height the openings are cut to (0 for a root).
    pub fn cap_height(&self) -> usize {
        match self {
            TreeAuth::Root(_) => 0,
            TreeAuth::Cap(cap) => cap.height(),
        }
    }

    /// ★ `whir_commit::verify_opening_capped`, emitted as a refusal.
    ///
    /// `index_bits` is the WHOLE leaf index, low first, one bit per tree
    /// level; `siblings` is the path to the cap, `index_bits.len() − c` long.
    /// The low bits are walked and the top `c` pick the cap node. With a
    /// [`TreeAuth::Root`] this is [`emit_verify_opening`] exactly.
    pub fn verify_opening(
        &self,
        b: &mut LfmBuilder,
        values: BlockValues<'_>,
        index_bits: &[Bit],
        siblings: &[WrapDigest],
    ) {
        match self {
            TreeAuth::Root(lanes) => emit_verify_opening(b, values, index_bits, siblings, lanes),
            TreeAuth::Cap(cap) => {
                assert_eq!(
                    siblings.len() + cap.height(),
                    index_bits.len(),
                    "a path to the cap: one sibling per level below it"
                );
                let leaf = emit_block_leaf(b, values);
                cap.verify_path(b, leaf, index_bits, siblings);
            }
        }
    }
}
