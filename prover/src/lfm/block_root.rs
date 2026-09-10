//! The BLOCK-ARTIFACT ROOT — the one program whose output is the block artifact.
//!
//! # Why this is its own file
//!
//! Every other program in the tree is an aggregation node, and one emitter
//! serves them all. The root is not one of them: it takes `fan_in + 1` children
//! (the global wrap is the extra), performs the L2G compare that a split tree
//! defers to the children's common ancestor, and answers the campaign's finish
//! line — *what does this artifact claim about block N?* That question should be
//! one file to read. Folded into [`super::per_table_aggregator::emit_node`] it
//! would be answered by tracing branches, and the thing most likely to be
//! quietly wrong at the end of a campaign is the claim, not the code.
//!
//! # Where it sits
//!
//! The root REPLACES the top interior level rather than sitting above it. Run
//! the interior until `<= fan_in` nodes remain; the root takes those plus the
//! global wrap. At 19 epochs and fan-in 2 the interior is levels 1..4 (10 + 5 +
//! 3 + 2 = 20 nodes) and the root is level 5, with 2 + 1 = 3 children. The total
//! is unchanged at 21 nodes — what changes is what the top node IS.
//!
//! # ⛔ What this does NOT close
//!
//! `super::programs`' own doc: the attestation is deliberately **not
//! self-enforcing**. The guest uses supplied roots verbatim without binding them
//! to the inner ELF; the binding happens outside, when a consumer recomputes the
//! id from an ELF it trusts and compares (`recursion::check_attestation`, a
//! native FFT + Merkle pass done once at top level, never in-VM). "One proof for
//! block N" therefore terminates in a host-side recompute against a trusted ELF.
//! That is deliberate, predates this campaign, and the root does not change it.

use super::builder::LfmBuilder;
use super::edsl::WrapDigest;
use super::per_table_aggregator::{
    ChildShape, LegCells, SchemaLayout, declare_leg_arenas, digest_from_lanes, emit_leg, fold_l2g,
};

/// How the interior grouped the epochs, level by level — the shape the root's
/// L2G fold must replicate exactly.
///
/// ⛔ **THE FOLD IS TREE-SHAPED, NOT FLAT, AND THIS IS THE WHOLE REASON THIS
/// TYPE EXISTS.** A level-1 node publishes `H(r0, r1)` over its two wraps; a
/// level-2 node folds its two CHILDREN'S published digests, i.e.
/// `H(H(r0,r1), H(r2,r3))`. `hash_pair` is a two-to-one compression and is not
/// associative, so that is **not** the flat left fold `H(H(H(r0,r1),r2),r3)`.
///
/// ⇒ The global wrap publishes the FLAT list of per-epoch L2G roots, so the root
/// cannot fold them left-to-right and compare. It must group them exactly as the
/// interior did, level by level, and fold each group. Getting this wrong fails an
/// HONEST prover — a completeness bug that reads like soundness working, which is
/// the worst kind to diagnose from a failing prove.
pub struct FoldShape {
    /// Per level, the children each node consumed — `tree_shape`'s arities for
    /// the interior levels the root sits above.
    pub levels: Vec<Vec<usize>>,
}

impl FoldShape {
    /// The interior's shape for `epochs` epochs at `fan_in`, excluding the level
    /// the ROOT replaces.
    ///
    /// `tree_shape` describes every interior level including the one that closes
    /// two nodes into one; the root takes those two directly, so its own level is
    /// dropped here.
    pub fn interior(epochs: usize, fan_in: usize) -> Self {
        let mut levels: Vec<Vec<usize>> = super::per_table_aggregator::tree_shape(epochs, fan_in)
            .into_iter()
            .map(|l| l.arities)
            .collect();
        levels.pop();
        Self { levels }
    }

    /// Replicate the interior's fold over a FLAT list of per-epoch digests.
    ///
    /// One group at a time, level by level, exactly as the nodes did it. The
    /// result is what the top interior children published, and therefore what the
    /// root compares against.
    pub fn refold(&self, b: &mut LfmBuilder, epochs: &[WrapDigest]) -> Vec<WrapDigest> {
        let mut level_in: Vec<WrapDigest> = epochs.to_vec();
        for arities in &self.levels {
            let want: usize = arities.iter().sum();
            assert_eq!(
                level_in.len(),
                want,
                "the fold shape expects {want} digests at this level but has {}; \
                 the interior's grouping and the root's have diverged, and an \
                 honest prover would fail the compare",
                level_in.len()
            );
            let mut out = Vec::with_capacity(arities.len());
            let mut cursor = 0usize;
            for a in arities {
                out.push(fold_l2g(b, &level_in[cursor..cursor + a]));
                cursor += a;
            }
            level_in = out;
        }
        level_in
    }
}

/// The GLOBAL wrap's published layout, which no `SchemaLayout` constructor
/// describes because it is unlike a wrap's and unlike a node's.
///
/// `global_verifier_program` publishes `z`, `alpha`, then *each epoch's L2G
/// re-commit root, the very cells Phase A absorbed* — `num_epochs` roots at
/// `lanes_per_root()` words apiece. No attestation id, no register run, no label
/// pair, no bus tail.
///
/// ⚠ **This schema IS variable in the EPOCH count, and that is correct.** It is
/// an INTERIOR INTERFACE, not the artifact. The campaign's rule — *schema may
/// depend on the BLOCK (pages, state access), never on the PROVING STRATEGY
/// (epoch count, fan-in, depth)* — governs what the ROOT publishes onward. Seeing
/// `num_epochs` here is not licence to publish per-epoch at the root.
pub struct GlobalLayout {
    pub num_epochs: usize,
    pub lanes_per_root: usize,
}

impl GlobalLayout {
    /// Words the global wrap publishes: `z`, `alpha`, then the roots.
    pub fn total(&self) -> usize {
        2 + self.num_epochs * self.lanes_per_root
    }

    /// Index of lane `w` of epoch `k`'s L2G root.
    pub fn l2g_word(&self, k: usize, w: usize) -> usize {
        2 + k * self.lanes_per_root + w
    }

    /// Every epoch's root, as digests, in epoch order.
    ///
    /// ⚠ ORDER IS AN OBLIGATION, NOT AN OBSERVATION. These are read in the
    /// global proof's SUB-PROOF order, and the interior folded in EPOCH order
    /// (its label pins fix that). The two must coincide. If they ever diverge an
    /// honest prove fails the compare, so the correspondence is asserted where it
    /// can be — and a reordered-fold tamper arm exists precisely because the
    /// thing it tests should be impossible.
    pub fn epoch_digests(&self, b: &mut LfmBuilder, leg: &LegCells) -> Vec<WrapDigest> {
        (0..self.num_epochs)
            .map(|k| {
                let lanes: Vec<_> = (0..self.lanes_per_root)
                    .map(|w| leg.publics[self.l2g_word(k, w)].lanes[0])
                    .collect();
                digest_from_lanes(b, &lanes)
            })
            .collect()
    }
}

/// Emit the root's L2G COMPARE: the interior children's published folds against
/// the same folds recomputed over the global wrap's per-epoch roots.
///
/// This is the compare a single-program aggregator did locally and a tree must
/// defer to the common ancestor of the epoch wraps and the global wrap.
pub fn emit_l2g_compare(
    b: &mut LfmBuilder,
    interior: &[LegCells],
    layouts: &[SchemaLayout],
    global: &LegCells,
    global_layout: &GlobalLayout,
    shape: &FoldShape,
) {
    assert_eq!(
        interior.len(),
        layouts.len(),
        "one layout per interior child"
    );
    let epochs = global_layout.epoch_digests(b, global);
    let recomputed = shape.refold(b, &epochs);
    assert_eq!(
        recomputed.len(),
        interior.len(),
        "the interior's top level has {} nodes but {} were refolded; the root's \
         fold shape does not match the tree it sits on",
        interior.len(),
        recomputed.len()
    );
    for (k, (leg, layout)) in interior.iter().zip(layouts).enumerate() {
        let published: Vec<_> = (0..layout.l2g_words)
            .map(|w| leg.publics[layout.l2g_word(w)].lanes[0])
            .collect();
        let claimed = digest_from_lanes(b, &published);
        // Lane by lane, not cell by cell: `assert_eq` takes felts, and the lane
        // view is the one every other equality in the tree is written against.
        let (cl, re) = (claimed.cells().to_vec(), recomputed[k].cells().to_vec());
        assert_eq!(
            cl.len(),
            re.len(),
            "interior child {k}: a {}-cell digest cannot equal a {}-cell one",
            cl.len(),
            re.len()
        );
        for (c, r) in cl.iter().zip(&re) {
            let (lc, lr) = (b.unpack(*c), b.unpack(*r));
            for (x, y) in lc.iter().zip(&lr) {
                b.assert_eq(*x, *y);
            }
        }
    }
}

/// Verify one child and hand back its leg — the same machinery every level uses,
/// named here so the root's three heterogeneous children read alike.
pub fn emit_child_leg(b: &mut LfmBuilder, child: &ChildShape<'_>) -> LegCells {
    let arenas = declare_leg_arenas(b, child);
    emit_leg(b, child, &arenas)
}

/// Bind the GLOBAL child, which `emit_chain_bindings` cannot take.
///
/// ⛔ The interior binding pass asserts an attestation id, a register run and a
/// label pair on EVERY child. The global wrap has none of them: it publishes
/// `z`, `alpha` and the L2G roots and nothing else. Handing it to the shared pass
/// would index past its published words. So the global child is bound by the L2G
/// compare alone — which is the entirety of what it is FOR — and this function
/// exists to say that explicitly rather than leave it as an omission.
pub fn assert_global_child_is_bound_only_by_l2g(global_layout: &GlobalLayout, published: usize) {
    assert_eq!(
        global_layout.total(),
        published,
        "the global wrap published {published} words but the layout describes {} \
         (z, alpha, then {} roots x {} lanes) — a mismatch here would silently \
         shift every L2G index the compare reads",
        global_layout.total(),
        global_layout.num_epochs,
        global_layout.lanes_per_root,
    );
}

// ⓘ Not yet called: the root's assembly is the next commit. Marked narrowly and
// with a reason rather than silencing the module, so an item that becomes dead
// for a REAL reason still shows up.
#[cfg(test)]
mod tests {
    use super::super::per_table_aggregator::tree_shape;
    use super::*;

    /// ★ The root's fold shape is the interior MINUS the level it replaces.
    ///
    /// The root takes the top interior level's nodes directly as children, so
    /// the level that would have closed them into one does not exist. Getting
    /// this off by one level means the root compares a fold of the wrong depth
    /// against its children — and it fails an HONEST prover, which is the worst
    /// way to find out.
    #[test]
    fn the_fold_shape_is_the_interior_minus_the_root_level() {
        for epochs in [2usize, 3, 4, 5, 10, 19, 36] {
            for fan_in in [2usize, 3] {
                let full = tree_shape(epochs, fan_in);
                let shape = FoldShape::interior(epochs, fan_in);
                assert_eq!(
                    shape.levels.len(),
                    full.len() - 1,
                    "{epochs}@{fan_in}: the root replaces exactly one level"
                );
                for (a, b) in shape.levels.iter().zip(&full) {
                    assert_eq!(a, &b.arities, "{epochs}@{fan_in}: same grouping");
                }
                // The top interior level must leave at most `fan_in` nodes, or
                // the root cannot take them plus the global wrap.
                let top = shape.levels.last().map(|l| l.len()).unwrap_or(epochs);
                assert!(
                    top <= fan_in,
                    "{epochs}@{fan_in}: the root would need {top} interior \
                     children plus the global wrap"
                );
            }
        }
    }

    /// ★ `refold` consumes every epoch digest and yields exactly the root's
    /// interior children — the invariant the L2G compare rests on.
    #[test]
    fn refold_yields_one_digest_per_root_child() {
        use super::super::builder::LfmBuilder;
        use super::super::edsl::WrapHash;
        use crate::tables::types::FE;

        for epochs in [2usize, 4, 5, 10, 19] {
            let fan_in = 2usize;
            let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::production());
            let lanes_per_root = super::super::proof_arena::lanes_per_root();
            let digests: Vec<_> = (0..epochs)
                .map(|k| {
                    let lanes: Vec<_> = (0..lanes_per_root)
                        .map(|w| b.felt_const(FE::from((17 * k + w) as u64)))
                        .collect();
                    digest_from_lanes(&mut b, &lanes)
                })
                .collect();
            let shape = FoldShape::interior(epochs, fan_in);
            let out = shape.refold(&mut b, &digests);
            let expect = shape.levels.last().map(|l| l.len()).unwrap_or(epochs);
            assert_eq!(
                out.len(),
                expect,
                "{epochs} epochs at fan-in {fan_in}: the root takes {expect} \
                 interior children"
            );
        }
    }

    /// ★★ THE ARTIFACT'S WIDTH DOES NOT DEPEND ON HOW WE PROVED IT.
    ///
    /// The campaign's rule: the artifact's schema may depend on the BLOCK
    /// (pages, public output), never on the PROVING STRATEGY (epoch count,
    /// fan-in, tree depth). `root_schema_words` takes neither an epoch count nor
    /// an arity, so today the rule is enforced by the SIGNATURE — and this test
    /// is what makes adding one a failure rather than a quiet regression.
    ///
    /// It also pins the two arms against each other: `WithFold` is exactly the
    /// NODE schema, and `AssertOnly` is that minus the one strategy-dependent
    /// item — which is the whole content of the open ruling, as arithmetic.
    #[test]
    fn the_artifact_width_is_independent_of_the_proving_strategy() {
        let num_reg = crate::tables::register::NUM_REGISTER_ADDRESSES;
        let lanes = super::super::proof_arena::lanes_per_root();
        for out_halves in [0usize, 1, 7, 64] {
            let assert_only = root_schema_words(num_reg, out_halves, RootPublishSet::AssertOnly);
            let with_fold = root_schema_words(num_reg, out_halves, RootPublishSet::WithFold);
            assert_eq!(
                with_fold - assert_only,
                lanes,
                "the two arms differ by exactly the folded L2G digest and \
                 nothing else — that difference IS the open ruling"
            );
            assert_eq!(
                with_fold,
                SchemaLayout::node(out_halves).total(),
                "the WithFold root publishes exactly the NODE schema; if these \
                 drift, a parent could no longer read a root as it reads a node"
            );
            // Block-dependent, as the rule allows: the width moves with the
            // block's public output and with nothing else in this call.
            assert_eq!(assert_only, 2 + 2 * num_reg + 4 + out_halves);
        }
    }

    /// ⛔ A grouping divergence must be LOUD. `refold` is handed the flat epoch
    /// list, and if the interior grouped differently the counts stop matching —
    /// that has to abort at emit time, not produce a digest that quietly differs.
    #[test]
    #[should_panic(expected = "the fold shape expects")]
    fn refold_refuses_a_digest_count_the_shape_does_not_expect() {
        use super::super::builder::LfmBuilder;
        use super::super::edsl::WrapHash;
        use crate::tables::types::FE;

        let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::production());
        let lanes_per_root = super::super::proof_arena::lanes_per_root();
        let digests: Vec<_> = (0..3)
            .map(|k| {
                let lanes: Vec<_> = (0..lanes_per_root)
                    .map(|w| b.felt_const(FE::from((k + w) as u64)))
                    .collect();
                digest_from_lanes(&mut b, &lanes)
            })
            .collect();
        // A shape built for TEN epochs, handed three.
        FoldShape::interior(10, 2).refold(&mut b, &digests);
    }
}

// ======================== the root's assembly ========================

/// What the root publishes — the campaign's finish line, as an enum so the
/// decision is a measurement rather than an argument.
///
/// ⛔ **THE OPEN RULING.** The interior's L2G digest is TREE-SHAPED (see
/// [`FoldShape`]), so its VALUE depends on fan-in and depth — the proving
/// strategy. Publishing it satisfies the letter of *schema may not depend on the
/// proving strategy* (it is fixed-size) while breaking its purpose: two provers
/// at fan-in 2 and 3 would emit **different artifact bytes for the same block**,
/// and the levers moving 19 epochs to 10 would change the thing being claimed.
///
/// Both arms are built so the choice can be made on a census and a published
/// word count rather than on a preference.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RootPublishSet {
    /// The block-level claim only. The L2G agreement is ASSERTED in-machine and
    /// nothing L2G-shaped is published, so no strategy-dependent value enters the
    /// artifact. ★ **RULED 2026-09-10, and the default.**
    AssertOnly,
    /// The block-level claim plus the folded L2G digest, as a node publishes it.
    /// ⚠ Carries a strategy-dependent value into the artifact; see above.
    WithFold,
}

impl Default for RootPublishSet {
    /// ★ RULED 2026-09-10. The whole decision is one field — [`WithFold`] is
    /// exactly the node schema and [`AssertOnly`] is that minus
    /// `lanes_per_root()` — and that field carries a value depending on fan-in
    /// and tree depth. Publishing it would mean two honest provers at different
    /// postures emit different artifact bytes for the same block, which is the
    /// property the campaign's rule protects. Nothing external can consume the
    /// digest anyway: the global roots live inside the same proof, so the compare
    /// binds in-machine and the published digest would have no reader.
    ///
    /// [`WithFold`]: RootPublishSet::WithFold
    /// [`AssertOnly`]: RootPublishSet::AssertOnly
    fn default() -> Self {
        Self::AssertOnly
    }
}

/// ★ WHY NOTHING PAGE-SHAPED IS PUBLISHED EITHER — ✓ VERIFIED, not assumed.
///
/// The archive's aggregator published each folded page's base, the private-input
/// page count and the touched-page list. None of it belongs here, and the reason
/// is that a consumer can already derive or already holds every piece:
///
/// - **The identity pages are ELF-DERIVED.** `recursion::check_attestation` calls
///   `expected_program_id(trusted_elf, opts)` → `precomputed_commitments(elf,
///   opts)`, which builds the DECODE commitment and the page commitments from
///   `Traces::page_configs_from_elf` — **the ELF bytes and the proof options
///   alone**. No block data, no private input, no bundle. So a consumer holding a
///   trusted ELF recomputes the whole id unaided, and republishing the pages
///   would restate what the id already commits to.
/// - **The runtime touched-page list is a different set, and is already bound.**
///   It is verifier input for rebuilding the GLOBAL_MEMORY AIR set, and
///   `continuation.rs:585-590` records that it is bus-enforced: a wrong set
///   imbalances the GlobalMemory bus or mismatches the AIR count, and it is bound
///   into the global Fiat-Shamir statement. A published copy would add nothing a
///   forger could not already not-do.
///
/// ⇒ Publishing page material would be redundant twice over, by two different
/// mechanisms. That is why this is an absence with a reason rather than a gap.
/// Everything the root verifies and binds.
///
/// The interior children are the top interior level's nodes — `<= fan_in` of
/// them — and `global` is the extra child that makes `fan_in + 1`.
pub struct RootInputs<'a> {
    pub interior: &'a [ChildShape<'a>],
    pub interior_layouts: &'a [SchemaLayout],
    /// Per interior child, the epoch labels it must carry (the two ends of its
    /// subtree).
    pub labels: &'a [&'a [u64]],
    /// The first and last epoch label of the whole BLOCK.
    pub label_range: (u64, u64),
    pub global: &'a ChildShape<'a>,
    pub global_layout: &'a GlobalLayout,
    pub fold_shape: &'a FoldShape,
    pub publishes: RootPublishSet,
}

/// Emit the block-artifact root: verify every child, bind them, compare the L2G,
/// publish the claim.
///
/// # The one thing that is not like a node
///
/// A node's children are homogeneous and go through one binding pass. The root's
/// are not: `fan_in` interior children carry an attestation id, a register run
/// and a label pair, and the global wrap carries **none of them** — it publishes
/// `z`, `alpha` and the per-epoch L2G roots and nothing else. Handing it to
/// [`super::per_table_aggregator::emit_chain_bindings`] would index past its
/// published words. So the interior children are bound by that pass unchanged and
/// the global child is bound by the L2G compare, which is the entirety of what it
/// is for.
pub fn emit_block_root(b: &mut LfmBuilder, inputs: &RootInputs<'_>) {
    let RootInputs {
        interior,
        interior_layouts,
        labels,
        label_range,
        global,
        global_layout,
        fold_shape,
        publishes,
    } = *inputs;
    assert!(
        !interior.is_empty(),
        "the root aggregates at least one interior child"
    );
    assert_eq!(
        interior.len(),
        interior_layouts.len(),
        "one layout per interior child"
    );
    assert_eq!(
        interior.len(),
        labels.len(),
        "one label run per interior child"
    );
    for (child, layout) in interior.iter().zip(interior_layouts) {
        layout.assert_covers(child.num_public_words);
    }
    assert_global_child_is_bound_only_by_l2g(global_layout, global.num_public_words);

    // ⚠ DECLARATION ORDER IS ABSORB ORDER, and the global child goes LAST.
    // Every child's arenas are declared before any leg is emitted, exactly as a
    // node does it; putting the global wrap last keeps the interior children's
    // arena indices identical to what they would be under `emit_node`, so a
    // reader comparing the two programs is comparing like with like.
    let interior_legs: Vec<LegCells> = interior
        .iter()
        .map(|child| emit_child_leg(b, child))
        .collect();
    let global_leg = emit_child_leg(b, global);

    super::per_table_aggregator::emit_chain_bindings(b, &interior_legs, interior_layouts, labels);
    emit_l2g_compare(
        b,
        &interior_legs,
        interior_layouts,
        &global_leg,
        global_layout,
        fold_shape,
    );
    emit_root_publishes(b, &interior_legs, interior_layouts, label_range, publishes);
}

/// The block artifact's claim.
///
/// The block-level fields are exactly what a node republishes, minus the L2G
/// item under [`RootPublishSet::AssertOnly`]: the attestation id every wrap
/// agreed on, the block's opening register vector, its closing register vector,
/// the first and last epoch labels as constants of THIS program, and the block's
/// public output.
///
/// ⓘ OPEN, and deliberately not invented here: the archive's `aggregator_program`
/// also published each folded page's base, the private-input page count and the
/// touched-page list. Those are BLOCK-dependent (allowed to vary) but they need
/// page material at the root that nothing currently hands it. Adding them is a
/// separate, named piece rather than a guess made inside this function.
fn emit_root_publishes(
    b: &mut LfmBuilder,
    legs: &[LegCells],
    layouts: &[SchemaLayout],
    label_range: (u64, u64),
    publishes: RootPublishSet,
) {
    use crate::tables::types::FE;

    let first = &legs[0];
    let last = legs.last().expect("nonempty");
    let l_first = &layouts[0];
    let l_last = layouts.last().expect("nonempty");

    for half in 0..2 {
        let lanes = &first.publics[l_first.id(half)].lanes;
        let word = b.pack_word([lanes[0], lanes[1], lanes[2], lanes[3]]);
        b.public(word);
    }
    for r in 0..l_first.num_reg {
        b.public(first.publics[l_first.reg_init(r)].lanes[0].as_cell());
    }
    for r in 0..l_last.num_reg {
        b.public(last.publics[l_last.reg_fini(r)].lanes[0].as_cell());
    }
    for label in [label_range.0, label_range.1] {
        let lo = b.felt_const(FE::from(label & 0xFFFF_FFFF));
        b.public(lo.as_cell());
        let hi = b.felt_const(FE::from(label >> 32));
        b.public(hi.as_cell());
    }
    for i in 0..l_last.out_halves {
        b.public(last.publics[l_last.out_half(i)].lanes[0].as_cell());
    }
    if publishes == RootPublishSet::WithFold {
        let digests: Vec<WrapDigest> = legs
            .iter()
            .zip(layouts)
            .map(|(leg, layout)| {
                let lanes: Vec<_> = (0..layout.l2g_words)
                    .map(|w| leg.publics[layout.l2g_word(w)].lanes[0])
                    .collect();
                digest_from_lanes(b, &lanes)
            })
            .collect();
        let folded = fold_l2g(b, &digests);
        for cell in folded.cells() {
            for lane in b.unpack(*cell) {
                b.public(lane.as_cell());
            }
        }
    }
}

/// Words the root publishes under `publishes` — the artifact's width, which a
/// consumer must know before it reads a single one.
///
/// ★ Under [`RootPublishSet::AssertOnly`] this depends on `num_reg` (a machine
/// constant) and `out_halves` (the BLOCK's public output) and on nothing else.
/// It does NOT depend on the epoch count, the fan-in or the tree's depth. That is
/// the rule the artifact has to satisfy, stated as arithmetic so a test can hold
/// it rather than a comment asking to be believed.
pub fn root_schema_words(num_reg: usize, out_halves: usize, publishes: RootPublishSet) -> usize {
    let base = 2 + 2 * num_reg + 4 + out_halves;
    match publishes {
        RootPublishSet::AssertOnly => base,
        RootPublishSet::WithFold => base + super::proof_arena::lanes_per_root(),
    }
}
