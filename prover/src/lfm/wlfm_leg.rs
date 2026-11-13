//! ★ The W-leg: an LFM program's in-guest verifier of one W-LFM proof
//! ([`super::wlfm`]).
//!
//! A W-LFM proof is a block proof of an LFM program's tables, so its verifier
//! is the block leaf's walk ([`super::whir_block`]'s `emit_leaf_over`) over
//! every group of the child, with two things swapped:
//!
//! 1. **The statement** is the child's ([`super::wlfm::absorb_wlfm_statement`]):
//!    the tag, the child's program id — a PROGRAM CONSTANT of the parent, which
//!    pins the program and its format — the machine version with the word
//!    count, then each hinted public word as one 32-byte call of its four lanes.
//! 2. **The closure** is the `LfmPublic` balance of those words
//!    ([`emit_public_balance`]): `Σ p·(1/q)` over the child's tables must equal
//!    it, which ties "the proof verifies" to "it published these words". A zero
//!    `q` has no satisfying assignment ([`emit_share`]'s inverse form).
//!
//! The rest is the leaf's, call for call: the roots (one a round-0 tree) and the
//! commit-time answers (one a stacked polynomial) hinted in the front and
//! absorbed after the statement and the derived prepared roots, `(z, α, β)`,
//! then per group its fork, its batched argue, its opening against its own root
//! cells carrying its answers at its own commit-time point, and its prepared
//! stack against the derived roots, held as constants.
//!
//! # ⛔ Every preprocessed prefix settled
//!
//! An in-guest table shape carries no preprocessed count, and the table walk's
//! `PreprocessedRoute::None` checks nothing. So every LFM table's prefix is
//! bound here only by its prepared stack's opening, and the leg refuses — at
//! emit time — a child whose stacks do not settle each table's AIR count
//! exactly ([`check_coverage`]). A child under [`WlfmPrep::Evaluated`] has no
//! stacks at all and is refused: only the host can evaluate its columns.
//!
//! # ⛔ Today's chains only
//!
//! The in-guest chain verifier has no final-polynomial round and folds at the
//! pinned width, so a child proved with a final polynomial or another fold is
//! refused at emit time.

use multilinear::stacking::StackedLayout;
use multilinear::whir_chain::{ChainConfig, FinalPoly};

use crate::multilinear_prove::stacks;
use crate::tables::types::{FEE, GoldilocksField};

use super::builder::{Cell, Ext, LfmBuilder};
use super::compiler::LfmProgram;
use super::instr::ArenaId;
use super::per_table_aggregator::{HintedPublicWord, LegCells, emit_public_balance};
use super::whir_block::{
    Shapes, batched_cap, emit_commit_ood_point, emit_share, finish_program, push_chains, run_trees,
};
use super::whir_chain::{ChainShape, LeafQuery};
use super::whir_epoch::{
    GroupChainWires, PreprocessedPlan, PreprocessedRoute, emit_absorb_roots,
    emit_group_walk_chains, hint_group_chains_shaped,
};
use super::whir_stacked::{OodWires, StackedChainWires};
use super::whir_transcript::WhirTranscript;
use super::wlfm::{WlfmArtifacts, WlfmPrep, WlfmProof, check_coverage, statement_head};
use super::word::{LfmWord, ext_word};
use crate::hash_pin::{BlockFamily, ProofFamily};

/// A W-LFM child as its parent's emitter reads it: its program's artifacts —
/// every shape, root and the id are program constants of the parent — and how
/// many words it publishes.
#[derive(Clone, Debug)]
pub struct WlfmChild {
    pub artifacts: WlfmArtifacts,
    pub num_public_words: usize,
}

/// The chain shape a W-leg walks for a stack of `n_stack` variables: the
/// child's config at the block family's tree arity, under the lean query check
/// (the block leaf's default).
pub fn wlfm_chain_shape(config: &ChainConfig, n_stack: usize) -> ChainShape {
    ChainShape {
        query: LeafQuery::Lean,
        ..ChainShape::new_at(config, n_stack, BlockFamily::WRAP.arity())
    }
}

/// A stack's round-0 trees under the child's config, each a range of its
/// polynomials: one a polynomial, or one a joint run.
fn trees(layout: &StackedLayout, config: &ChainConfig) -> Vec<std::ops::Range<usize>> {
    multilinear::stacked_eval::round0_trees(layout.num_polys(), config)
}

/// What a W-leg derives from its child before it hints anything: the groups'
/// stacked layouts and domains. Refused where the child is not one an in-guest
/// verifier can check: its stacks do not settle every preprocessed prefix
/// exactly, it binds its columns by evaluation, or its chains stop at a final
/// polynomial or fold otherwise than the pinned chain.
fn child_frame(
    a: &WlfmArtifacts,
    shapes: &[(usize, usize)],
) -> Result<
    (
        Vec<StackedLayout>,
        Vec<multilinear::whir::Domain<GoldilocksField>>,
    ),
    String,
> {
    if a.prep != WlfmPrep::Stacked {
        return Err("a W-leg verifies a child whose columns are stacked, not evaluated".into());
    }
    if a.final_poly != FinalPoly::Off || a.fold.is_some() {
        return Err(format!(
            "a W-leg verifies today's chains, not final {:?} fold {:?}",
            a.final_poly, a.fold
        ));
    }
    let plan: Vec<(usize, &[(usize, usize)])> = a
        .stacks
        .iter()
        .map(|s| (s.group, s.tables.as_slice()))
        .collect();
    check_coverage(&a.counts, &a.sizes, &plan)?;
    stacks(shapes, &a.sizes, &a.config).map_err(|e| format!("{e:?}"))
}

/// ★ Emits ONE W-LFM child's complete verification over `arena`, which it
/// fills in [`wlfm_leg_arena`]'s order and sizes: the child's public words, its
/// roots, its commit-time answers, then each group's batched argue, opening and
/// prepared opening.
///
/// ⚠ ONE TRANSCRIPT PER CHILD, as the STARK leg: each child was proved against
/// a transcript seeded by its own statement.
pub fn emit_wlfm_leg(
    b: &mut LfmBuilder,
    child: &WlfmChild,
    arena: ArenaId,
) -> Result<LegCells, String> {
    if b.wrap_hash() != BlockFamily::WRAP {
        return Err(
            "a W-leg replays the block family's transcript; the builder hashes otherwise".into(),
        );
    }
    let a = &child.artifacts;
    let airs = a.airs();
    let refs = airs.air_refs();
    let shapes = a.shapes(&airs);
    let (layouts, domains) = child_frame(a, &shapes)?;
    let cap = batched_cap(&a.config)?.ok_or("a W-LFM child argues batched")?;
    let mut at = 0u32;

    // ---- 1. the front: the public words, the roots (one a round-0 tree), the
    // answers (one a stacked polynomial).
    let publics: Vec<HintedPublicWord> = (0..child.num_public_words)
        .map(|index| {
            let cell = b.hint_word(arena, at);
            at += 1;
            let lanes = b.unpack(cell);
            HintedPublicWord {
                index: index as u32,
                halves: Vec::new(),
                lanes: lanes.to_vec(),
            }
        })
        .collect();
    let num_roots: usize = layouts.iter().map(|l| trees(l, &a.config).len()).sum();
    let num_answers: usize = layouts.iter().map(StackedLayout::num_polys).sum();
    let carried: Vec<Cell> = (0..num_roots)
        .map(|_| {
            let cell = b.hint_word(arena, at);
            at += 1;
            cell
        })
        .collect();
    let answers: Vec<Ext> = (0..num_answers)
        .map(|_| {
            let cell = b.hint_word(arena, at);
            at += 1;
            cell.as_ext()
        })
        .collect();

    // ---- 2. the statement (the id a constant, the words their lanes), the
    // roots block, the answers, then (z, α, β).
    let mut transcript = WhirTranscript::for_builder(b);
    for call in statement_head(&a.program_id, child.num_public_words) {
        transcript.absorb_const_bytes(&call);
    }
    for word in &publics {
        let lanes: [_; 4] = [word.lanes[0], word.lanes[1], word.lanes[2], word.lanes[3]];
        transcript.absorb_root_lanes(b, &lanes);
    }
    let derived: Vec<LfmWord> = a
        .derived_roots()
        .iter()
        .map(super::algebraic_commit::commitment_to_digest)
        .collect();
    emit_absorb_roots(b, &mut transcript, &carried, &derived);
    #[cfg(test)]
    let late = faults::get() == faults::Fault::AnswersAfterDraws;
    #[cfg(not(test))]
    let late = false;
    #[cfg(test)]
    let skipped = match faults::get() {
        faults::Fault::SkipFirstGroupsAnswers => {
            layouts.first().map_or(0, StackedLayout::num_polys)
        }
        _ => 0,
    };
    #[cfg(not(test))]
    let skipped = 0;
    if !late {
        for answer in &answers[skipped..] {
            transcript.absorb_ext(b, *answer);
        }
    }
    let z = transcript.sample_ext(b);
    let alpha = transcript.sample_ext(b);
    let beta = transcript.sample_ext(b);
    if late {
        for answer in &answers {
            transcript.absorb_ext(b, *answer);
        }
    }

    // ---- 3. every group, on its fork.
    let mut outputs: Vec<(Ext, Ext)> = Vec::new();
    let (mut root_at, mut answer_at) = (0usize, 0usize);
    for (g, tables) in a.groups().iter().enumerate() {
        let start = a.group_start(g);
        let owned = Shapes::build(&refs, &shapes, tables)?;
        let table_shapes = owned.table_shapes();
        // ⛔ Each table's prefix: settled by its group's stack, exactly its
        // AIR's count (refused above otherwise), and checked nowhere else.
        let plans: Vec<PreprocessedPlan<'_>> = tables
            .iter()
            .map(|&t| {
                let settled = a
                    .stacks
                    .iter()
                    .filter(|s| s.group == g)
                    .flat_map(|s| s.tables.iter())
                    .find(|&&(table, _)| table == t)
                    .map_or(0, |&(_, n)| n);
                if settled != a.counts[t] {
                    return Err(format!(
                        "table {t} has {} preprocessed columns and its stack settles {settled}",
                        a.counts[t]
                    ));
                }
                Ok(PreprocessedPlan {
                    settled,
                    route: PreprocessedRoute::None,
                })
            })
            .collect::<Result<_, String>>()?;
        let slots: Vec<&[usize]> = owned.layouts.iter().map(|l| l.slot_of()).collect();
        let ladder_len = table_shapes
            .iter()
            .map(|shape| super::whir_bus::alpha_powers_read(shape.bus))
            .max()
            .unwrap_or(1);
        let ladder = super::whir_poly::emit_challenge_powers(b, alpha, ladder_len);
        let mut fork = transcript.clone();
        fork.absorb_const_bytes(&(g as u64).to_le_bytes());
        let argue_plan = owned.argue_plan(cap);
        let store =
            super::whir_batch::hint_batched_shaped(b, arena, &mut at, &table_shapes, &argue_plan);
        let walk = super::whir_batch::emit_batched_walk(
            b,
            &mut fork,
            &store,
            &table_shapes,
            &argue_plan,
            &plans,
            &slots,
            z,
            &ladder,
            beta,
        );

        // The group's opening, against its own root cells, its answers carried
        // at its own commit-time point — the front's cells, never a second copy.
        let layout = &layouts[g];
        let polys = layout.num_polys();
        let group_trees = trees(layout, &a.config);
        let group_roots = &carried[root_at..root_at + group_trees.len()];
        let point = emit_commit_ood_point(
            b,
            &stark::multilinear_block::commit_ood_prefix(g, layout, &a.config),
            group_roots,
            layout.n_stack(),
            layout.n_stack() + a.config.log_blowup,
        );
        #[cfg(test)]
        let fresh: Vec<Ext> = match faults::get() {
            faults::Fault::FreshAnswerHint => (0..polys)
                .map(|_| {
                    let cell = b.hint_word(arena, at);
                    at += 1;
                    cell.as_ext()
                })
                .collect(),
            _ => Vec::new(),
        };
        #[cfg(test)]
        let group_answers: &[Ext] = if fresh.is_empty() {
            &answers[answer_at..answer_at + polys]
        } else {
            &fresh
        };
        #[cfg(not(test))]
        let group_answers: &[Ext] = &answers[answer_at..answer_at + polys];
        let chain = wlfm_chain_shape(&a.config, layout.n_stack());
        let held = hint_group_chains_shaped(b, arena, &mut at, polys, &chain);
        let openings: Vec<_> = held
            .storage
            .iter()
            .map(super::whir_chain::RoundStorage::openings)
            .collect();
        let rounds: Vec<Vec<super::whir_chain::ChainRoundWires<'_>>> = held
            .storage
            .iter()
            .zip(&openings)
            .map(|(chain, (current, next))| chain.wires(current, next))
            .collect();
        // Each chain against its run's slice of the group's root cells: tree
        // `p` of a run is checked against the root at the run's first tree plus
        // `p` — one root for a joint run.
        let chains: Vec<StackedChainWires<'_>> = held
            .runs
            .iter()
            .enumerate()
            .map(|(i, run)| StackedChainWires {
                rounds: &rounds[i],
                roots: &group_roots[run_trees(&group_trees, run)],
                final_value: held.finals[i],
            })
            .collect();
        emit_group_walk_chains(
            b,
            &mut fork,
            &[GroupChainWires {
                layout,
                chains: &chains,
                shape: &chain,
                domain: &domains[g],
                ood: Some(OodWires {
                    point: &point,
                    answers: group_answers,
                }),
            }],
            &[tables.len()],
            &walk,
        );

        // Its prepared stack, against the derived roots held as constants:
        // each table's prefix at that table's point. Prepared roots carry no
        // answers: the verifier derives them, so each is a codeword.
        for stack in a.stacks.iter().filter(|s| s.group == g) {
            let shape = wlfm_chain_shape(&a.config, stack.layout.n_stack());
            let held =
                hint_group_chains_shaped(b, arena, &mut at, stack.layout.num_polys(), &shape);
            let roots: Vec<Cell> = stack
                .roots
                .iter()
                .map(|root| {
                    b.digest_const(super::algebraic_commit::commitment_to_digest(root))
                        .as_cell()
                })
                .collect();
            let openings: Vec<_> = held
                .storage
                .iter()
                .map(super::whir_chain::RoundStorage::openings)
                .collect();
            let rounds: Vec<Vec<super::whir_chain::ChainRoundWires<'_>>> = held
                .storage
                .iter()
                .zip(&openings)
                .map(|(chain, (current, next))| chain.wires(current, next))
                .collect();
            let stack_trees = trees(&stack.layout, &a.config);
            let chains: Vec<StackedChainWires<'_>> = held
                .runs
                .iter()
                .enumerate()
                .map(|(i, run)| StackedChainWires {
                    rounds: &rounds[i],
                    roots: &roots[run_trees(&stack_trees, run)],
                    final_value: held.finals[i],
                })
                .collect();
            let mut points: Vec<&[Ext]> = Vec::new();
            let mut values: Vec<Ext> = Vec::new();
            for &(table, n) in &stack.tables {
                let slot = table - start;
                let column_at = walk.column_at(slot);
                points.extend(std::iter::repeat_n(walk.points[slot].as_slice(), n));
                values.extend_from_slice(&walk.values[column_at..column_at + n]);
            }
            super::whir_stacked::emit_stacked_verify_chains(
                b,
                &mut fork,
                &stack.layout,
                &chains,
                &points,
                &values,
                None,
                &shape,
                &stack.domain,
            );
        }
        outputs.extend(walk.outputs.iter().copied());
        root_at += group_trees.len();
        answer_at += polys;
    }

    // ---- 4. the closure, against the PUBLIC balance of the hinted words.
    let mut sum = b.ext_const(&FEE::zero());
    for (p, q) in &outputs {
        let share = emit_share(b, *p, *q, true);
        sum = b.eadd(sum, share);
    }
    let target = emit_public_balance(b, &publics, z, alpha);
    #[cfg(test)]
    let closed = faults::get() != faults::Fault::SkipClosure;
    #[cfg(not(test))]
    let closed = true;
    if closed {
        b.assert_eq_ext(sum, target);
    }
    b.set_arena_len(arena, at);
    Ok(LegCells {
        publics,
        z_alpha: (z, alpha),
    })
}

/// ★ A W-leg's arena from the child's proof — the witness — in
/// [`emit_wlfm_leg`]'s order. Refuses a proof whose parts do not have the
/// shapes the child's artifacts derive (a count the leg would misread).
pub fn wlfm_leg_arena(child: &WlfmChild, proof: &WlfmProof) -> Result<Vec<LfmWord>, String> {
    let a = &child.artifacts;
    let airs = a.airs();
    let refs = airs.air_refs();
    let shapes = a.shapes(&airs);
    let (layouts, _) = child_frame(a, &shapes)?;
    let cap = batched_cap(&a.config)?.ok_or("a W-LFM child argues batched")?;
    if proof.public_words.len() != child.num_public_words {
        return Err(format!(
            "the proof publishes {} words and the child {}",
            proof.public_words.len(),
            child.num_public_words
        ));
    }
    let num_roots: usize = layouts.iter().map(|l| trees(l, &a.config).len()).sum();
    let num_answers: usize = layouts.iter().map(StackedLayout::num_polys).sum();
    if proof.proof.roots.len() != num_roots || proof.commit_ood.len() != num_answers {
        return Err(format!(
            "the proof carries {} roots and {} answers for {num_roots} trees and {num_answers} \
             polynomials",
            proof.proof.roots.len(),
            proof.commit_ood.len()
        ));
    }
    if proof.argues.len() != a.sizes.len()
        || proof.proof.columns.len() != a.sizes.len()
        || proof.prepared.len() != a.stacks.len()
    {
        return Err("the proof's groups or prepared openings are not the child's".into());
    }
    let mut words: Vec<LfmWord> = proof.public_words.iter().map(|(_, w)| *w).collect();
    words.extend(
        proof
            .proof
            .roots
            .iter()
            .map(super::algebraic_commit::commitment_to_digest),
    );
    words.extend(proof.commit_ood.iter().map(ext_word));
    for (g, tables) in a.groups().iter().enumerate() {
        let owned = Shapes::build(&refs, &shapes, tables)?;
        let table_shapes = owned.table_shapes();
        let before = words.len();
        super::whir_batch::push_batched_words(&mut words, &proof.argues[g]);
        let expected = super::whir_batch::batched_words(&table_shapes, &owned.argue_plan(cap));
        if words.len() - before != expected {
            return Err(format!(
                "group {g}'s batched argue carries {} words, its shapes {expected}",
                words.len() - before
            ));
        }
        let layout = &layouts[g];
        let chain = wlfm_chain_shape(&a.config, layout.n_stack());
        push_chains(
            &mut words,
            &proof.proof.columns[g],
            layout.num_polys(),
            &chain,
        )?;
        for (k, stack) in a.stacks.iter().enumerate().filter(|(_, s)| s.group == g) {
            let shape = wlfm_chain_shape(&a.config, stack.layout.n_stack());
            push_chains(
                &mut words,
                &proof.prepared[k],
                stack.layout.num_polys(),
                &shape,
            )?;
        }
    }
    Ok(words)
}

/// A program that runs one W-leg and republishes the child's words in order,
/// each as a whole word: the second compression step's body
/// ([`super::compress`]), and what a test executes a child's arena against.
pub fn wlfm_leg_program(child: &WlfmChild) -> Result<LfmProgram, String> {
    let mut b = LfmBuilder::new().with_wrap_hash(BlockFamily::WRAP);
    let arena = b.declare_arena(0);
    let leg = emit_wlfm_leg(&mut b, child, arena)?;
    for word in &leg.publics {
        let packed = b.pack_word([word.lanes[0], word.lanes[1], word.lanes[2], word.lanes[3]]);
        b.public(packed);
    }
    finish_program(b)
}

/// Test only: the leg's mutations of its commit-time answers, one at a time,
/// each of which a test must catch. Thread-local, so a test that sets one moves
/// no other test's program.
#[cfg(test)]
pub(crate) mod faults {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(crate) enum Fault {
        #[default]
        None,
        /// The front absorbs the answers after `(z, α, β)`.
        AnswersAfterDraws,
        /// The front leaves the first group's answers out of the transcript.
        SkipFirstGroupsAnswers,
        /// Each group's opening reads its answers from a fresh hint, not the
        /// front's cells.
        FreshAnswerHint,
        /// The closure's assert (the tables' shares against the words'
        /// `LfmPublic` balance) left out.
        SkipClosure,
    }

    std::thread_local! {
        static FAULT: std::cell::Cell<Fault> = const { std::cell::Cell::new(Fault::None) };
    }

    pub(crate) fn get() -> Fault {
        FAULT.with(std::cell::Cell::get)
    }

    /// Runs `f` under `fault`, restoring the previous one after.
    pub(crate) fn with<R>(fault: Fault, f: impl FnOnce() -> R) -> R {
        struct Restore(Fault);
        impl Drop for Restore {
            fn drop(&mut self) {
                FAULT.with(|c| c.set(self.0));
            }
        }
        let _restore = Restore(FAULT.with(|c| c.replace(fault)));
        f()
    }
}
