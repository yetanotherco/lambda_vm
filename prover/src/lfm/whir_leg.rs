//! ★ The W-leg: a node's in-guest verifier of ONE W-LFM child (D-WHIR §3).
//!
//! A W-leg is [`super::whir_epoch::whir_epoch_program`]'s body with three
//! things swapped, and nothing new underneath:
//!
//! | step | W-leg | the wrap's epoch verify |
//! |---|---|---|
//! | statement | the W-LFM statement over the child's HINTED public words | the epoch statement, all program text |
//! | derived roots | the child's prepared-stack roots, interned | DECODE's prepared root, interned |
//! | plans | every table `settled = count, route: None` | DECODE settled, BITWISE closed form, … |
//! | closure | Σ p/q == the `LfmPublic` balance of the hinted words | Σ p/q == the COMMIT offset |
//! | prepared opening | one stack, every table's prefix at its own point | DECODE's five columns at one point |
//!
//! The roots block, the alpha ladder, the table walk, the group walk and the
//! stacked opening are the wrap's own emitters, gated there against the host.
//!
//! # ⛔ The count comes from the AIR here too
//!
//! [`emit_whir_leg`] builds its plan through [`WhirLfmPlan::build`], which
//! refuses — at EMIT time — any prepared plan that does not settle exactly
//! `0..air.num_precomputed_columns()` of every table. A program emitted for a
//! child whose preprocessed columns nothing checks is a build failure, never a
//! program that runs.
//!
//! # The public words are hinted as FELTS, and that is sound
//!
//! Four hinted felts a word, no halves, no canonicity guard: a machine felt is
//! canonical by construction. A hinted word is unconstrained by the `HINT`
//! chip, but every lane is absorbed into the statement, and the sponge's `Pack`
//! rows receive each lane as a BASE token (`(addr, v, 0, 0, 0)`), so a hinted
//! "felt" whose upper lanes are not zero has no satisfying assignment. The same
//! lanes then feed the balance (`emul_base`, another base read) and the
//! node's bindings and publishes, which read exactly what was absorbed.

use multilinear::stacking::StackedLayout;
use multilinear::whir::Domain;
use stark::config::Commitment;
use stark::multilinear_logup::InteractionShape;
use stark::proof::options::ProofOptions;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::airs::{ChipSet, DynLfmAir};
use super::builder::{Cell, Ext, Felt, LfmBuilder};
use super::hash::HasherKind;
use super::instr::ArenaId;
use super::per_table_aggregator::{
    HintedPublicWord, LegCells, NodePublishSet, NodePublishes, SchemaLayout, emit_chain_bindings,
    emit_node_publishes, emit_public_balance,
};
use super::whir_bus::Cost;
use super::whir_chain::{ChainRoundWires, ChainShape, RoundStorage};
use super::whir_epoch::{
    GroupWires, PreprocessedPlan, PreprocessedRoute, TableWalk, TableWires, emit_group_walk,
    emit_roots_block, emit_table_walk, hint_group_chains, hint_table_wires, push_table_words,
};
use super::whir_proof::{
    PrepPolicy, WhirLfmArtifacts, WhirLfmPlan, WhirLfmProof, airs_for, statement_runs,
};
use super::whir_stacked::{StackedPolyWires, emit_stacked_verify};
use super::whir_table::{TableProofWires, TableShape};
use super::whir_transcript::{CANDIDATES_PER_SQUEEZE, SpongeEntry, WhirTranscript};
use super::word::{LfmWord, WORD_LANES, base_word};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// One W-LFM child, as a node's emitter and arena writer read it.
///
/// ⚠ The EMITTER reads the proof for its SHAPE only — how many roots, rounds
/// and evaluations it carries, all of which the child's program and config fix
/// — and never a value; the ARENA WRITER reads the values. A program emitted
/// from one honest proof therefore verifies every honest proof of the same
/// child program.
pub struct WhirChild<'a> {
    /// The child program's artifacts: its identity, heights, config and
    /// prepared stack — program text of the node.
    pub artifacts: &'a WhirLfmArtifacts,
    /// The child's proof and published words.
    pub proof: &'a WhirLfmProof,
    /// The options the child's AIR set is built under (inert on this path).
    pub options: &'a ProofOptions,
}

impl WhirChild<'_> {
    /// Words the child publishes.
    pub fn num_public_words(&self) -> usize {
        self.proof.public_words.len()
    }
}

/// The arenas one W-leg reads, in declaration order: the published words, then
/// the proof.
pub struct WhirLegArenas {
    publics: ArenaId,
    proof: ArenaId,
    proof_words: u32,
}

/// Everything the emitter derives about one child, derived ONCE per leg from
/// the child's artifacts and its AIR set — the plan the host verifier builds,
/// plus the in-guest shapes over it.
struct LegPlan<'a> {
    plan: WhirLfmPlan<'a>,
    buses: Vec<Vec<InteractionShape<E>>>,
}

impl<'a> LegPlan<'a> {
    fn build(artifacts: &WhirLfmArtifacts, airs: &[DynLfmAir<'a>]) -> Self {
        // ⛔ The emit-time refusal: a plan that does not settle every table's
        // prefix exactly is not a program anyone may run.
        let plan = WhirLfmPlan::build(artifacts, airs)
            .unwrap_or_else(|e| panic!("a W-leg cannot be emitted for this child: {e:?}"));
        let buses =
            plan.layouts
                .iter()
                .zip(airs)
                .map(|(layout, air)| {
                    let slots = layout.slot_of().to_vec();
                    stark::multilinear_logup::interaction_shapes(
                        air.bus_interactions(),
                        slots.len(),
                        |column| {
                            slots.get(column).copied().ok_or(
                                multilinear::Error::UnknownPolynomial {
                                    index: column,
                                    len: slots.len(),
                                },
                            )
                        },
                    )
                    .expect("the bus probes")
                })
                .collect();
        Self { plan, buses }
    }

    fn table_shapes(&self) -> Vec<TableShape<'_>> {
        self.plan
            .layouts
            .iter()
            .zip(&self.buses)
            .map(|(layout, bus)| TableShape {
                ir: layout.shape(),
                bus,
                kinds: layout.kinds(),
                num_columns: layout.num_columns(),
                num_vars: layout.num_vars(),
            })
            .collect()
    }

    fn slots(&self) -> Vec<&[usize]> {
        self.plan
            .layouts
            .iter()
            .map(|layout| layout.slot_of())
            .collect()
    }

    /// ★ Every table's plan: its whole prefix settled by the prepared opening,
    /// nothing left for a route. `settled` IS the AIR's count — the number the
    /// plan's refusal just checked the prepared stack covers.
    fn preprocessed_plans(&self) -> Vec<PreprocessedPlan<'static>> {
        self.plan
            .counts
            .iter()
            .map(|&count| PreprocessedPlan {
                settled: count,
                route: PreprocessedRoute::None,
            })
            .collect()
    }

    fn group_shape(&self) -> ChainShape {
        ChainShape::new(&self.plan.config, self.plan.group_layouts[0].n_stack())
    }

    fn prepared_shape(&self, artifacts: &WhirLfmArtifacts) -> ChainShape {
        ChainShape::new(&self.plan.config, artifacts.prepared_layout.n_stack())
    }

    /// The prepared stack's runs, `(table, count)` in stack order.
    fn prepared_runs(&self) -> Vec<(usize, usize)> {
        self.plan
            .counts
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count > 0)
            .map(|(table, &count)| (table, count))
            .collect()
    }
}

/// The child's proof words, in the order [`emit_whir_leg`] hints them: the
/// carried roots, every table's words, the main group's chains, the prepared
/// stack's chains.
fn proof_words(child: &WhirChild<'_>, plan: &LegPlan<'_>) -> Vec<LfmWord> {
    let proof = &child.proof.proof;
    let mut words: Vec<LfmWord> = proof
        .roots
        .iter()
        .map(super::algebraic_commit::commitment_to_digest)
        .collect();
    for table in &proof.tables {
        push_table_words(&mut words, table);
    }
    let group_shape = plan.group_shape();
    for opening in &proof.columns {
        for chain in &opening.polys {
            words.push(super::word::ext_word(&chain.final_value));
            super::whir_chain::push_round_words(&mut words, &group_shape, chain);
        }
    }
    let prepared = proof
        .preprocessed
        .as_ref()
        .expect("a W-LFM proof carries its prepared opening");
    let prepared_shape = plan.prepared_shape(child.artifacts);
    for chain in &prepared.polys {
        words.push(super::word::ext_word(&chain.final_value));
        super::whir_chain::push_round_words(&mut words, &prepared_shape, chain);
    }
    words
}

/// ★ The child's arenas, in [`declare_whir_leg_arenas`]' order: the published
/// words as four base felts each, then the proof.
pub fn whir_leg_arena_words(child: &WhirChild<'_>) -> Vec<Vec<LfmWord>> {
    let airs = airs_for(child.artifacts, child.options);
    let refs = airs.air_refs();
    let plan = LegPlan::build(child.artifacts, &refs);
    let publics: Vec<LfmWord> = child
        .proof
        .public_words
        .iter()
        .flat_map(|(_, word)| word.iter().map(|lane| base_word(*lane)))
        .collect();
    vec![publics, proof_words(child, &plan)]
}

/// Declare one W-leg's arenas. The proof arena's length is the child's word
/// count, which its program and config fix.
pub fn declare_whir_leg_arenas(b: &mut LfmBuilder, child: &WhirChild<'_>) -> WhirLegArenas {
    let airs = airs_for(child.artifacts, child.options);
    let refs = airs.air_refs();
    let plan = LegPlan::build(child.artifacts, &refs);
    let proof_words = proof_words(child, &plan).len() as u32;
    let publics = b.declare_arena((WORD_LANES * child.num_public_words()) as u32);
    let proof = b.declare_arena(proof_words);
    WhirLegArenas {
        publics,
        proof,
        proof_words,
    }
}

/// Hint a child's published words: four felts each, emit-time indices.
fn hint_public_felts(b: &mut LfmBuilder, arena: ArenaId, count: usize) -> Vec<HintedPublicWord> {
    (0..count)
        .map(|index| {
            let lanes: Vec<Felt> = (0..WORD_LANES)
                .map(|lane| b.hint_felt(arena, (WORD_LANES * index + lane) as u32))
                .collect();
            HintedPublicWord {
                index: index as u32,
                halves: Vec::new(),
                lanes,
            }
        })
        .collect()
}

/// ★ The W-LFM statement, emitted: the constant head, the hinted lanes as
/// felts, the constant tail — `whir_proof::absorb_whir_lfm_statement`'s three
/// runs, from the same [`statement_runs`].
fn emit_whir_lfm_statement(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    artifacts: &WhirLfmArtifacts,
    config: &multilinear::whir_chain::ChainConfig,
    words: &[HintedPublicWord],
) {
    let runs = statement_runs(
        &artifacts.program_id,
        words.len(),
        &artifacts.table_num_vars,
        config,
    );
    transcript.absorb_const_bytes(&runs.head);
    let felts: Vec<Felt> = words
        .iter()
        .flat_map(|word| word.lanes.iter().copied())
        .collect();
    if !felts.is_empty() {
        transcript.absorb_felts(b, &felts);
    }
    transcript.absorb_const_bytes(&runs.tail);
}

/// The walk the main group opens under policy B: every table's columns past
/// its preprocessed prefix — `None` under policy A, where the main stack holds
/// every column and the walk is used as it is.
fn main_stack_walk(walk: &TableWalk, plan: &WhirLfmPlan<'_>) -> Option<TableWalk> {
    if !plan.policy.excludes_prefix() {
        return None;
    }
    let mut values = Vec::with_capacity(walk.values.len());
    let mut start = 0usize;
    for (&width, &count) in walk.widths.iter().zip(&plan.counts) {
        values.extend_from_slice(&walk.values[start + count..start + width]);
        start += width;
    }
    Some(TableWalk {
        outputs: walk.outputs.clone(),
        points: walk.points.clone(),
        widths: walk
            .widths
            .iter()
            .zip(&plan.counts)
            .map(|(&width, &count)| width - count)
            .collect(),
        values,
    })
}

/// The leg's closure: `Σ_t p_t / q_t == Σ_i 1/(z − fingerprint_i)`, the
/// `LfmPublic` balance of the hinted words. A `Div` refuses a vanished `q`,
/// which is the host's `BusImbalance` on `contribution`'s `None`.
fn emit_whir_closure(
    b: &mut LfmBuilder,
    walk: &TableWalk,
    words: &[HintedPublicWord],
    z: Ext,
    alpha: Ext,
) {
    let target = emit_public_balance(b, words, z, alpha);
    let mut balance: Option<Ext> = None;
    for (p, q) in &walk.outputs {
        let share = b.ediv(*p, *q);
        balance = Some(match balance {
            None => share,
            Some(running) => b.eadd(running, share),
        });
    }
    let balance = balance.expect("a W-LFM proof argues at least one table");
    b.assert_eq_ext(balance, target);
}

/// ★★ ONE W-LFM CHILD'S VERIFY — `whir_proof::verify_whir_checked`'s body as
/// machine code, in `multi_verify`'s order: the statement, the roots block
/// (carried, then the prepared roots, then `z, α, β`), the table walk, the
/// closure, the group walk, and the prepared opening last.
pub fn emit_whir_leg(b: &mut LfmBuilder, child: &WhirChild<'_>, a: &WhirLegArenas) -> LegCells {
    let artifacts = child.artifacts;
    let proof = &child.proof.proof;
    let airs = airs_for(artifacts, child.options);
    let refs = airs.air_refs();
    let plan = LegPlan::build(artifacts, &refs);

    // 1. The published words, then the statement over them.
    let publics = hint_public_felts(b, a.publics, child.num_public_words());
    let mut transcript = WhirTranscript::new();
    emit_whir_lfm_statement(b, &mut transcript, artifacts, &plan.plan.config, &publics);

    // 2. The roots block: the carried group roots hinted, the prepared stack's
    //    roots INTERNED — program text of this node, derived from the child's
    //    program, never read from the arena.
    let mut at = 0u32;
    let carried: Vec<Cell> = proof
        .roots
        .iter()
        .map(|_| {
            let cell = b.hint_word(a.proof, at);
            at += 1;
            cell
        })
        .collect();
    let derived: Vec<LfmWord> = artifacts
        .prepared_roots
        .iter()
        .map(super::algebraic_commit::commitment_to_digest)
        .collect();
    let (z, alpha, beta) = emit_roots_block(b, &mut transcript, &carried, &derived);

    // 3. The shared alpha ladder, then the table walk with every table's
    //    prefix settled by the prepared opening.
    let shapes = plan.table_shapes();
    let ladder_len = shapes
        .iter()
        .map(|shape| super::whir_bus::alpha_powers_read(shape.bus))
        .max()
        .unwrap_or(1);
    let ladder = super::whir_poly::emit_challenge_powers(b, alpha, ladder_len);
    let mut store = Vec::with_capacity(proof.tables.len());
    for table in &proof.tables {
        store.push(hint_table_wires(b, a.proof, &mut at, table));
    }
    let wires: Vec<TableProofWires<'_>> = store.iter().map(TableWires::borrow).collect();
    let plans = plan.preprocessed_plans();
    let slots = plan.slots();
    let walk = emit_table_walk(
        b,
        &mut transcript,
        &wires,
        &shapes,
        &plans,
        &slots,
        z,
        &ladder,
        beta,
    );

    // 4. The closure against the claimed words' `LfmPublic` balance.
    emit_whir_closure(b, &walk, &publics, z, alpha);

    // 5. The main group — one, holding every table.
    let group_shape = plan.group_shape();
    let mut chains = Vec::with_capacity(proof.columns.len());
    for opening in &proof.columns {
        chains.push(hint_group_chains(
            b,
            a.proof,
            &mut at,
            opening,
            &group_shape,
        ));
    }
    let group_openings: Vec<Vec<_>> = chains
        .iter()
        .map(|held| {
            held.storage
                .iter()
                .map(RoundStorage::openings)
                .collect::<Vec<_>>()
        })
        .collect();
    let group_rounds: Vec<Vec<Vec<ChainRoundWires<'_>>>> = chains
        .iter()
        .zip(&group_openings)
        .map(|(held, openings)| {
            held.storage
                .iter()
                .zip(openings)
                .map(|(chain, (current, next))| chain.wires(current, next))
                .collect()
        })
        .collect();
    let group_wires: Vec<Vec<StackedPolyWires<'_>>> = {
        let mut root_at = 0usize;
        let mut out = Vec::with_capacity(chains.len());
        for (group, held) in chains.iter().enumerate() {
            out.push(
                (0..held.finals.len())
                    .map(|poly| StackedPolyWires {
                        rounds: &group_rounds[group][poly],
                        root: carried[root_at + poly],
                        final_value: held.finals[poly],
                    })
                    .collect(),
            );
            root_at += plan.plan.group_layouts[group].num_polys();
        }
        out
    };
    let group_shapes = vec![group_shape; chains.len()];
    let groups: Vec<GroupWires<'_>> = (0..chains.len())
        .map(|group| GroupWires {
            layout: &plan.plan.group_layouts[group],
            polys: &group_wires[group],
            shape: &group_shapes[group],
            domain: &plan.plan.group_domains[group],
        })
        .collect();
    // ★ Under policy B the main stack holds each table's columns PAST its
    // prefix, so the group walk sees those claims only; the prefix's are the
    // prepared opening's below, against the full walk.
    let main_walk = main_stack_walk(&walk, &plan.plan);
    emit_group_walk(
        b,
        &mut transcript,
        &groups,
        &plan.plan.sizes(),
        main_walk.as_ref().unwrap_or(&walk),
    );

    // 6. ★★ The prepared opening, last: every table's prefix at that table's
    //    own reduced point, against the INTERNED roots the roots block absorbed
    //    (`word_const` hands back the same cell, so no second `Const`). The
    //    values are the tables' own settled column values, gathered — so the
    //    opening proves the pinned stack takes exactly what each table's
    //    argument settled on, and no separate equality is needed.
    let prepared = proof
        .preprocessed
        .as_ref()
        .expect("a W-LFM proof carries its prepared opening");
    let prepared_shape = plan.prepared_shape(artifacts);
    let held = hint_group_chains(b, a.proof, &mut at, prepared, &prepared_shape);
    let prepared_roots: Vec<Cell> = derived
        .iter()
        .map(|word| b.digest_const(*word).as_cell())
        .collect();
    assert_eq!(
        prepared_roots.len(),
        artifacts.prepared_layout.num_polys(),
        "the prepared stack committed {} polynomials and the child carries {} roots",
        artifacts.prepared_layout.num_polys(),
        prepared_roots.len()
    );
    let openings: Vec<_> = held.storage.iter().map(RoundStorage::openings).collect();
    let rounds: Vec<Vec<ChainRoundWires<'_>>> = held
        .storage
        .iter()
        .zip(&openings)
        .map(|(chain, (current, next))| chain.wires(current, next))
        .collect();
    let prepared_wires: Vec<StackedPolyWires<'_>> = (0..held.finals.len())
        .map(|poly| StackedPolyWires {
            rounds: &rounds[poly],
            root: prepared_roots[poly],
            final_value: held.finals[poly],
        })
        .collect();
    let column_points = walk.column_points();
    let mut points: Vec<&[Ext]> = Vec::new();
    let mut values: Vec<Ext> = Vec::new();
    for (table, count) in plan.prepared_runs() {
        let start = walk.column_at(table);
        points.extend_from_slice(&column_points[start..start + count]);
        values.extend_from_slice(&walk.values[start..start + count]);
    }
    assert_eq!(
        points.len(),
        artifacts.prepared_layout.placements().len(),
        "the prepared stack has {} columns and the runs gather {} claims",
        artifacts.prepared_layout.placements().len(),
        points.len()
    );
    emit_stacked_verify(
        b,
        &mut transcript,
        &artifacts.prepared_layout,
        &prepared_wires,
        &points,
        &values,
        &prepared_shape,
        &artifacts.prepared_domain,
    );

    assert_eq!(
        at, a.proof_words,
        "the leg must hint exactly the words the proof arena holds"
    );
    LegCells {
        publics,
        z_alpha: (z, alpha),
    }
}

// =============================================================================
// The node over W-LFM children
// =============================================================================

/// A node over W-LFM children: [`super::per_table_aggregator::NodeInputs`] with
/// W-LFM legs.
pub struct WhirNodeInputs<'a> {
    pub children: &'a [WhirChild<'a>],
    pub layouts: &'a [SchemaLayout],
    pub labels: &'a [&'a [u64]],
    pub label_range: (u64, u64),
    pub publishes: NodePublishSet,
}

/// ★ One node over W-LFM children, end to end: verify every child with a W-leg,
/// bind them, publish the node's schema — `emit_node` with the leg swapped.
///
/// A node never mixes child kinds, so the dispatch is per program: the
/// bindings and the publishes read only the legs' published lanes, which a
/// W-leg and a STARK leg hand over in the same shape.
pub fn emit_whir_node(b: &mut LfmBuilder, inputs: &WhirNodeInputs<'_>) {
    let WhirNodeInputs {
        children,
        layouts,
        labels,
        label_range,
        publishes,
    } = *inputs;
    assert!(!children.is_empty(), "a node verifies at least one child");
    assert_eq!(children.len(), layouts.len(), "one layout per child");
    assert_eq!(children.len(), labels.len(), "one label list per child");
    for (child, layout) in children.iter().zip(layouts) {
        layout.assert_covers(child.num_public_words());
    }
    // Declaration order IS absorb order: every child's arenas before any leg.
    let arenas: Vec<WhirLegArenas> = children
        .iter()
        .map(|child| declare_whir_leg_arenas(b, child))
        .collect();
    let legs: Vec<LegCells> = children
        .iter()
        .zip(&arenas)
        .map(|(child, a)| emit_whir_leg(b, child, a))
        .collect();
    emit_chain_bindings(b, &legs, layouts, labels);
    emit_node_publishes(
        b,
        &NodePublishes {
            legs: &legs,
            layouts,
            label_range,
        },
    );
    if publishes == NodePublishSet::Diagnostic {
        for leg in &legs {
            b.public(leg.z_alpha.0.as_cell());
            b.public(leg.z_alpha.1.as_cell());
        }
    }
}

/// A node's arenas over W-LFM children, in [`emit_whir_node`]'s order.
pub fn whir_node_arena_words(children: &[WhirChild<'_>]) -> Vec<Vec<LfmWord>> {
    children.iter().flat_map(whir_leg_arena_words).collect()
}

// =============================================================================
// The block-artifact root over W-LFM children
// =============================================================================

/// [`super::block_root::RootInputs`] over W-LFM children: the top interior
/// level's nodes and the global child, all W-LFM proofs.
pub struct WhirRootInputs<'a> {
    pub interior: &'a [WhirChild<'a>],
    pub interior_layouts: &'a [SchemaLayout],
    /// Per interior child, the epoch labels it must carry.
    pub labels: &'a [&'a [u64]],
    /// The first and last epoch label of the whole block.
    pub label_range: (u64, u64),
    pub global: &'a WhirChild<'a>,
    /// What the global child published.
    pub global_child_layout: &'a super::block_root::GlobalLayout,
    pub fold_shape: &'a super::block_root::FoldShape,
    pub publishes: super::block_root::RootPublishSet,
}

/// ★ The block-artifact root over W-LFM children —
/// [`super::block_root::emit_block_root`] with its legs swapped for W-legs.
///
/// Everything the root does besides verifying its children — the length pins,
/// the cross-child bindings, the L2G compare against the global child's roots,
/// the published claim — is `emit_root_checks_and_publishes`, the STARK root's
/// own code over the legs' published lanes, which a W-leg hands over in the same
/// shape. Declaration order is arena order and the global child goes LAST, as
/// in the STARK root, so the arenas are [`whir_leg_arena_words`] of the interior
/// children and then of the global child.
pub fn emit_whir_block_root(b: &mut LfmBuilder, inputs: &WhirRootInputs<'_>) {
    let WhirRootInputs {
        interior,
        interior_layouts,
        labels,
        label_range,
        global,
        global_child_layout,
        fold_shape,
        publishes,
    } = *inputs;
    let mut leg_of = |child: &WhirChild<'_>| {
        let arenas = declare_whir_leg_arenas(b, child);
        emit_whir_leg(b, child, &arenas)
    };
    let interior_legs: Vec<LegCells> = interior.iter().map(&mut leg_of).collect();
    let global_leg = leg_of(global);
    super::block_root::emit_root_checks_and_publishes(
        b,
        &super::block_root::RootLegs {
            interior: &interior_legs,
            interior_layouts,
            labels,
            label_range,
            global: &global_leg,
            global_child_layout,
            fold_shape,
            publishes,
        },
    );
}

// =============================================================================
// The cost form (F1)
// =============================================================================

/// What one W-leg costs, kept as the terms it is made of.
///
/// ⚠ The convention is `whir_global::GlobalCost`'s: [`Self::operations`] is
/// CONST-FREE, HINT-FREE and PUBLISH-FREE; the constants are the leg's pool BY
/// VALUE, unioned, so a caller assembling several legs unions their pools
/// rather than adding counts.
#[derive(Debug, Clone, Default)]
pub struct WhirLegCost {
    /// The roots block and the alpha ladder.
    pub spine: usize,
    /// The tables' legs, their sponge rows included (no preprocessed route).
    pub tables: usize,
    /// The closure: the `LfmPublic` balance, Σ p/q and the assert.
    pub closure: usize,
    /// The main group's wrapper and chains.
    pub groups: usize,
    /// The prepared stack's wrapper and chains.
    pub prepared: usize,
    /// `LFM_HINT` rows: the published felts plus the proof words.
    pub hints: usize,
    /// Permutations, whole.
    pub perms: usize,
    /// The pool, by value.
    pub constants: Vec<LfmWord>,
}

impl WhirLegCost {
    /// INSTRUCTIONS excluding every `LFM_CONST`, hint and publish.
    pub fn operations(&self) -> usize {
        self.spine + self.tables + self.closure + self.groups + self.prepared
    }

    /// Every instruction a program holding this leg alone should hold.
    pub fn instructions(&self) -> usize {
        self.operations() + self.constants.len() + self.hints
    }
}

/// Everything a W-leg's cost depends on: the child PROGRAM's shape and its
/// public word count — no proof. The identity and the prepared roots enter only
/// the constant pool, by value.
pub struct WhirLegShape<'a> {
    pub artifacts: &'a WhirLfmArtifacts,
    pub num_public_words: usize,
    pub options: &'a ProofOptions,
}

/// The words one table's proof occupies in the arena, from its shape alone —
/// [`push_table_words`]' count: the bus output's two, each GKR layer's rounds
/// (three evaluations a round) and its four claims, the main sumcheck's rounds
/// at its degree, the committed factors' values, the reduction's rounds (two
/// evaluations a round) and the column values.
pub fn table_proof_words(shape: &TableShape<'_>) -> usize {
    let gkr: usize = (0..shape.gkr_layers())
        .map(|layer| super::whir_gkr::GKR_SUMCHECK_DEGREE * layer + 4)
        .sum();
    let factors = shape.sources().len();
    2 + gkr
        + shape.num_vars * shape.sumcheck_degree()
        + factors
        + shape.num_vars * super::whir_reduce::REDUCE_DEGREE
        + shape.num_columns
}

/// ★ [`emit_whir_leg`]'s cost, leg by leg, threading the sponge — the F1.
///
/// Every term is one of the landed forms the wrap's own F1s are gated on,
/// evaluated at the child's shapes in the leg's own order.
pub fn whir_leg_cost(shape: &WhirLegShape<'_>) -> WhirLegCost {
    let artifacts = shape.artifacts;
    let airs = airs_for(artifacts, shape.options);
    let refs = airs.air_refs();
    let plan = LegPlan::build(artifacts, &refs);
    let config = &plan.plan.config;
    let shapes = plan.table_shapes();
    let plans = plan.preprocessed_plans();
    let n = shape.num_public_words;

    let mut cost = WhirLegCost::default();
    let mut pool = Cost::default();

    // The statement: no operation, only its two constant runs — whose felts
    // are interned when the next runtime absorb flushes them.
    let runs = statement_runs(&artifacts.program_id, n, &artifacts.table_num_vars, config);
    for run in [&runs.head, &runs.tail] {
        for group in run.chunks(8) {
            let mut whole = [0u8; 8];
            whole[..group.len()].copy_from_slice(group);
            pool.constant_word(base_word(FE::from(u64::from_be_bytes(whole))));
        }
    }
    let statement_felts = (runs.head.len() + runs.tail.len()) / 8 + WORD_LANES * n;

    // The roots block.
    let carried = plan.plan.group_layouts[0].num_polys();
    let derived: Vec<LfmWord> = artifacts
        .prepared_roots
        .iter()
        .map(super::algebraic_commit::commitment_to_digest)
        .collect();
    let (roots_ops, schedule) = super::whir_epoch::roots_block_cost(
        carried,
        derived.len(),
        SpongeEntry {
            buffered_felts: statement_felts,
            out_pos: CANDIDATES_PER_SQUEEZE,
            p1: None,
        },
    );
    cost.spine += roots_ops + schedule.rows();
    cost.perms += schedule.perms();
    for word in super::whir_epoch::roots_block_constants(&derived, &schedule) {
        pool.constant_word(word);
    }

    // The alpha ladder.
    let ladder_len = shapes
        .iter()
        .map(|shape| super::whir_bus::alpha_powers_read(shape.bus))
        .max()
        .unwrap_or(1);
    cost.spine += super::whir_poly::challenge_powers_rows(ladder_len);
    pool.constant(FEE::one());

    // The table walk.
    let walk = super::whir_epoch::table_walk_cost(&shapes, &plans, schedule.entry());
    cost.tables += walk.operations();
    cost.perms += walk.perms;
    pool.merge(&walk.leg);

    // The closure: `emit_public_balance` (α²..α⁵, then per word the index term,
    // four lane terms, the fingerprint and its inverse, and the running sum),
    // Σ p/q (a `Div` per table, an `Add` for all but the first) and the assert.
    cost.closure += public_balance_rows(n) + 2 * shapes.len() - 1 + ASSERT_ROWS;
    pool.constant(FEE::from(crate::tables::types::BusId::LfmPublic as u64));
    pool.constant(FEE::one());
    // The assert's zero — and the balance's own, when there are no words.
    pool.constant(FEE::zero());
    for index in 0..n {
        pool.constant_word(base_word(FE::from(index as u64)));
    }

    // The Newton pool: the union over every sumcheck round is the set for the
    // largest degree among them (`whir_global::global_cost`'s reasoning).
    let newton_degree = shapes.iter().map(TableShape::sumcheck_degree).fold(
        super::whir_gkr::GKR_SUMCHECK_DEGREE
            .max(super::whir_reduce::REDUCE_DEGREE)
            .max(super::whir_chain::SUMCHECK_DEGREE),
        usize::max,
    );
    for word in super::whir_poly::sumcheck_round_constants(newton_degree) {
        pool.constant_word(word);
    }

    // The main group, threaded from where the tables left the sponge.
    let groups = super::whir_epoch::epoch_group_costs(
        &plan.plan.main_shapes,
        &plan.plan.sizes(),
        &plan.plan.group_layouts,
        config,
        walk.entry,
    );
    let group_shape = plan.group_shape();
    for (group, domain) in groups.iter().zip(&plan.plan.group_domains) {
        cost.groups += group.operations();
        cost.perms += group.perms();
        chain_constants(&mut pool, group, &group_shape, domain);
    }

    // The prepared opening, threaded from where the main group left it. Its
    // groups are the TABLES: each table's run is claimed at that table's point.
    let prepared_shape = plan.prepared_shape(artifacts);
    let group_of: Vec<usize> = plan
        .prepared_runs()
        .iter()
        .enumerate()
        .flat_map(|(run, &(_, count))| std::iter::repeat_n(run, count))
        .collect();
    let entry = groups
        .last()
        .map_or(walk.entry, super::whir_stacked::StackedCost::entry);
    let prepared = super::whir_stacked::stacked_verify_cost(
        &artifacts.prepared_layout,
        &group_of,
        &prepared_shape,
        entry,
    );
    cost.prepared += prepared.operations();
    cost.perms += prepared.perms();
    chain_constants(
        &mut pool,
        &prepared,
        &prepared_shape,
        &artifacts.prepared_domain,
    );

    // The hints: four felts a published word, the carried roots, every
    // table's words, and one final value plus the rounds per chain.
    cost.hints += WORD_LANES * n + carried;
    cost.hints += shapes.iter().map(table_proof_words).sum::<usize>();
    cost.hints += carried * (1 + RoundStorage::words(&group_shape) as usize);
    cost.hints +=
        artifacts.prepared_layout.num_polys() * (1 + RoundStorage::words(&prepared_shape) as usize);

    cost.constants = pool.constant_values().to_vec();
    cost
}

/// A chain wrapper's own words, its fold constants and its grind constants
/// (`whir_global::global_cost`'s three chain terms).
fn chain_constants(
    pool: &mut Cost,
    cost: &super::whir_stacked::StackedCost,
    shape: &ChainShape,
    domain: &Domain<F>,
) {
    for word in cost.own_constants() {
        pool.constant_word(word);
    }
    for word in super::whir_chain::chain_fold_constants(shape, domain) {
        pool.constant_word(word);
    }
    let (folding, ood, query) = shape.grind;
    for bits in [folding, ood, query] {
        for word in super::epoch::grinding_check_constants(bits as u8) {
            pool.constant_word(word);
        }
    }
}

/// Rows [`emit_public_balance`] emits over `n` words, constants excluded:
/// four powers (α²..α⁵), then per word an `emul_base` and an `eadd` for the
/// index, an `emul_base` and an `eadd` per lane, the `esub` and the `ediv` —
/// twelve — and the running sum's `eadd` for every word but the first. Zero
/// words emit nothing but the interned zero.
pub const fn public_balance_rows(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    4 + 12 * n + (n - 1)
}

/// Rows an `assert_eq_ext` lowers to: the difference and the division by zero.
const ASSERT_ROWS: usize = 2;

/// The shapes the design's sizing used, re-exported for the pins: log2 padded
/// rows per chip in the frozen order CONST BALU XALU SELECT BITDEC HASH LANES
/// HINT PUBLIC RANGE (D-WHIR §3.2, the wt714 panels).
pub const SIZING_PROFILES: &[(&str, [u8; 10])] = &[
    ("wrap 0", [8, 17, 19, 18, 12, 17, 18, 18, 8, 16]),
    ("global wrap", [8, 19, 20, 19, 14, 19, 20, 20, 6, 16]),
    ("L1N0", [10, 19, 21, 19, 13, 19, 19, 20, 8, 16]),
    ("L2N0", [10, 19, 21, 20, 13, 19, 19, 20, 8, 16]),
    ("node -1", [10, 18, 20, 18, 12, 18, 18, 19, 8, 16]),
];

/// The chip set every W-LFM recursion program carries.
pub const RECURSION_CHIP_SET: ChipSet = ChipSet {
    keccak: false,
    blake3: false,
    bitwise: false,
};

/// The hasher every recursion program runs.
pub const RECURSION_HASHER: HasherKind = HasherKind::Rpx;

/// A program's W-LFM artifacts at a SHAPE, with no commitment behind them: the
/// prepared stack's layout and domain are the ones its shape gives, its roots
/// are zero digests of the right count and the identity is zero.
///
/// ⚠ For cost forms ONLY — every quantity a W-leg's operations, permutations
/// and hints depend on is a function of the shape, and only the constant pool
/// reads the roots' and the identity's VALUES. A verifier handed these would
/// refuse every proof, which is the right failure for an object that describes
/// no commitment.
pub fn shape_only_artifacts(
    table_num_vars: &[u8],
    options: &ProofOptions,
    policy: PrepPolicy,
) -> WhirLfmArtifacts {
    let airs = super::whir_proof::whir_lfm_airs(RECURSION_HASHER, RECURSION_CHIP_SET, 1, options);
    let refs = airs.air_refs();
    assert_eq!(
        refs.len(),
        table_num_vars.len(),
        "one height per table of the recursion chip set"
    );
    let shapes = super::whir_proof::table_shapes(&refs, table_num_vars);
    let config = super::whir_proof::whir_lfm_config(&shapes);
    let counts: Vec<usize> = refs
        .iter()
        .map(|air| air.num_precomputed_columns())
        .collect();
    let prep_shapes: Vec<(usize, usize)> = counts
        .iter()
        .zip(table_num_vars)
        .filter(|&(&count, _)| count > 0)
        .map(|(&count, &num_vars)| (count, num_vars as usize))
        .collect();
    let prepared_layout: StackedLayout =
        stark::multilinear_table::global_layout(&prep_shapes, config.format.stack)
            .expect("the prepared stack lays out");
    let prepared_domain = Domain::<F>::new(prepared_layout.n_stack() + config.log_blowup)
        .expect("the prepared domain exists");
    let prepared_at = counts
        .iter()
        .enumerate()
        .flat_map(|(table, &count)| stark::multilinear_table::leading_columns(table, count))
        .collect();
    let zero: Commitment = [0u8; 32];
    WhirLfmArtifacts {
        hasher: RECURSION_HASHER,
        chip_set: RECURSION_CHIP_SET,
        hash_chunks: 1,
        table_num_vars: table_num_vars.to_vec(),
        config,
        policy,
        prepared_roots: vec![zero; prepared_layout.num_polys()],
        prepared_at,
        prepared_layout,
        prepared_domain,
        program_id: zero,
    }
}
