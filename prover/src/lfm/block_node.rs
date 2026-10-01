//! A no-epoch BLOCK NODE — verifies block leaves (or other block nodes) and
//! binds them into one statement about one block; at the top of the tree it
//! closes the block's bus.
//!
//! # The checks (D-NOEPOCH §12.3)
//!
//! Each child is verified exactly as an epoch node verifies its children
//! ([`super::per_table_aggregator::emit_leg`]: the child's LFM statement over its
//! claimed words, its `program_id` absorbed as an emit-time constant, every
//! sub-proof verified). Then, on the children's published words:
//!
//! 1. **One challenge set** — the transcript state digest equal across children,
//!    so every leaf forked every instance from the same statement and the same
//!    roots (and drew the same `z, α`).
//! 2. **The same block** — the attestation id and the public output halves equal
//!    across children.
//! 3. **The bus** — the children's sums added; a non-top node publishes the
//!    total, the TOP node asserts it is zero: Σ over every instance of its bus
//!    contribution equals the COMMIT-bus target, which exactly one leaf
//!    subtracted (the plan's carrier, `block_plan::BlockTreePlan::carrier`).
//!    This is the monolithic verifier's bus check (`verifier.rs`, "Σ
//!    table_contribution = expected_bus_balance"), moved to where every
//!    contribution is in scope.
//!
//! Coverage is not a check here: the children's ids are constants of this
//! program, a leaf's id commits to its instance list, and the lists were
//! validated when the leaves were emitted (`block_leaf::BlockPartition`).

use crate::tables::types::{FE, FEE};

use super::builder::{Ext, LfmBuilder};
use super::per_table_aggregator::{
    ChildShape, HintedPublicWord, LegArenas, LegCells, assert_words_equal, declare_leg_arenas,
    emit_leg,
};

/// Where each field sits in a block leaf's or block node's published words.
///
/// `[id₀, id₁, state…, out halves…, sum]` — a leaf and a non-top node publish
/// the same layout, so a node reads either kind of child alike. The top node
/// publishes the block's claim without the sum it has just closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockLayout {
    /// Words the transcript state digest occupies: one on an algebraic hash, two
    /// on a byte hash (`edsl::digest_words`).
    pub state_words: usize,
    pub out_halves: usize,
    /// Whether the last word is the bus sum.
    pub has_sum: bool,
}

impl BlockLayout {
    /// What a leaf and a non-top node publish.
    pub fn child(state_words: usize, out_halves: usize) -> Self {
        Self {
            state_words,
            out_halves,
            has_sum: true,
        }
    }

    /// What the top node publishes.
    pub fn top(state_words: usize, out_halves: usize) -> Self {
        Self {
            state_words,
            out_halves,
            has_sum: false,
        }
    }

    pub fn id(&self, half: usize) -> usize {
        half
    }
    pub fn state(&self, w: usize) -> usize {
        2 + w
    }
    pub fn out_half(&self, i: usize) -> usize {
        2 + self.state_words + i
    }
    pub fn sum(&self) -> usize {
        assert!(self.has_sum, "the top node publishes no sum");
        2 + self.state_words + self.out_halves
    }
    pub fn total(&self) -> usize {
        2 + self.state_words + self.out_halves + usize::from(self.has_sum)
    }
}

/// Which of the cross-child checks a node emits. Production emits all of them
/// ([`BlockBindings::ALL`]) and no production signature takes another set; the
/// weakened sets exist so each negative test can show its check is the one doing
/// the refusing ([`emit_block_node_with`], [`bind_and_publish_with`], test-only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockBindings {
    id: bool,
    state: bool,
    out: bool,
    /// The top node's zero assert on the bus sum.
    bus: bool,
}

impl BlockBindings {
    pub(crate) const ALL: Self = Self {
        id: true,
        state: true,
        out: true,
        bus: true,
    };

    /// [`Self::ALL`] minus the named check — a mutation, test-only.
    #[cfg(test)]
    pub(crate) fn without(check: &str) -> Self {
        let mut b = Self::ALL;
        match check {
            "id" => b.id = false,
            "state" => b.state = false,
            "out" => b.out = false,
            "bus" => b.bus = false,
            other => panic!("no block binding named `{other}`"),
        }
        b
    }
}

/// The sum a child published, as an extension cell. Lane 3 of an extension
/// publish is zero by construction; asserted so a word that is not one cannot be
/// summed as if it were.
fn hinted_ext(b: &mut LfmBuilder, w: &HintedPublicWord) -> Ext {
    let zero = b.felt_const(FE::from(0u64));
    b.assert_eq(w.lanes[3], zero);
    b.pack_ext(w.lanes[0], w.lanes[1], w.lanes[2])
}

/// The cross-child checks on verified children's published words, and the sum of
/// their bus shares.
fn emit_block_bindings(
    b: &mut LfmBuilder,
    legs: &[LegCells],
    layout: &BlockLayout,
    checks: BlockBindings,
) -> Ext {
    assert!(!legs.is_empty(), "a node has children");
    assert!(layout.has_sum, "children publish their bus share");
    let first = &legs[0];
    for leg in &legs[1..] {
        if checks.id {
            for half in 0..2 {
                assert_words_equal(
                    b,
                    &first.publics[layout.id(half)],
                    &leg.publics[layout.id(half)],
                );
            }
        }
        if checks.state {
            for w in 0..layout.state_words {
                assert_words_equal(
                    b,
                    &first.publics[layout.state(w)],
                    &leg.publics[layout.state(w)],
                );
            }
        }
        if checks.out {
            for i in 0..layout.out_halves {
                assert_words_equal(
                    b,
                    &first.publics[layout.out_half(i)],
                    &leg.publics[layout.out_half(i)],
                );
            }
        }
    }
    let shares: Vec<Ext> = legs
        .iter()
        .map(|leg| hinted_ext(b, &leg.publics[layout.sum()]))
        .collect();
    let (head, rest) = shares.split_first().expect("nonempty");
    rest.iter().fold(*head, |acc, s| b.eadd(acc, *s))
}

/// Everything a block node's program is a function of.
pub struct BlockNodeInputs<'a> {
    pub children: &'a [ChildShape<'a>],
    /// The children's layout — one, since leaves and non-top nodes share it.
    pub layout: BlockLayout,
    /// Whether this node closes the bus and publishes the block's claim.
    pub top: bool,
}

/// Declare every child's arenas (in child order), verify every child, bind them,
/// publish.
pub fn emit_block_node(b: &mut LfmBuilder, inputs: &BlockNodeInputs<'_>) {
    emit_node(b, inputs, BlockBindings::ALL);
}

/// [`emit_block_node`] under a weakened binding set — a mutation, test-only.
#[cfg(test)]
pub(crate) fn emit_block_node_with(
    b: &mut LfmBuilder,
    inputs: &BlockNodeInputs<'_>,
    checks: BlockBindings,
) {
    emit_node(b, inputs, checks);
}

fn emit_node(b: &mut LfmBuilder, inputs: &BlockNodeInputs<'_>, checks: BlockBindings) {
    let BlockNodeInputs {
        children,
        layout,
        top,
    } = *inputs;
    assert!(!children.is_empty(), "a node verifies at least one child");
    for child in children {
        assert_eq!(
            child.num_public_words,
            layout.total(),
            "every child publishes the block layout"
        );
    }
    let arenas: Vec<LegArenas> = children
        .iter()
        .map(|child| declare_leg_arenas(b, child))
        .collect();
    let legs: Vec<LegCells> = children
        .iter()
        .zip(&arenas)
        .map(|(child, a)| emit_leg(b, child, a))
        .collect();
    bind(b, &legs, &layout, top, checks);
}

/// Everything a node does after verifying its children: the cross-child checks,
/// the top node's bus close, and the publishes.
pub fn bind_and_publish(b: &mut LfmBuilder, legs: &[LegCells], layout: &BlockLayout, top: bool) {
    bind(b, legs, layout, top, BlockBindings::ALL);
}

/// [`bind_and_publish`] under a weakened binding set — a mutation, test-only.
#[cfg(test)]
pub(crate) fn bind_and_publish_with(
    b: &mut LfmBuilder,
    legs: &[LegCells],
    layout: &BlockLayout,
    top: bool,
    checks: BlockBindings,
) {
    bind(b, legs, layout, top, checks);
}

fn bind(
    b: &mut LfmBuilder,
    legs: &[LegCells],
    layout: &BlockLayout,
    top: bool,
    checks: BlockBindings,
) {
    let sum = emit_block_bindings(b, legs, layout, checks);
    if top && checks.bus {
        let zero = b.ext_const(&FEE::zero());
        b.assert_eq_ext(sum, zero);
    }
    emit_block_publishes(b, &legs[0], layout, (!top).then_some(sum));
}

/// Republish the first child's id, state and output halves in the form a child
/// published them, then the node's sum unless this is the top.
fn emit_block_publishes(
    b: &mut LfmBuilder,
    first: &LegCells,
    layout: &BlockLayout,
    sum: Option<Ext>,
) {
    for half in 0..2 {
        let lanes = &first.publics[layout.id(half)].lanes;
        let word = b.pack_word([lanes[0], lanes[1], lanes[2], lanes[3]]);
        b.public(word);
    }
    for w in 0..layout.state_words {
        let lanes = &first.publics[layout.state(w)].lanes;
        let word = b.pack_word([lanes[0], lanes[1], lanes[2], lanes[3]]);
        b.public(word);
    }
    for i in 0..layout.out_halves {
        b.public(first.publics[layout.out_half(i)].lanes[0].as_cell());
    }
    if let Some(s) = sum {
        b.public(s.as_cell());
    }
}
