//! Laptop gates for the W-leg ([`super::wlfm_leg`]): the in-guest verifier of
//! one W-LFM proof executes an honest child's arena at each rate a child is
//! proved at, republishes its words, and has no execution for a child whose
//! proof was tampered anywhere the leg reads; each commit-time-answer check is
//! load-bearing; a child the leg cannot check is refused at emit time.
//!
//! The fixture is [`super::wlfm_tests`]'s program, at a small query count so
//! the leg stays a laptop-sized program.

use crate::tables::types::{FE, FEE};

use super::compiler::LfmProgram;
use super::executor::execute;
use super::wlfm::{WlfmBuild, WlfmLevers, WlfmPrep, WlfmProof, WlfmRate, test_format};
use super::wlfm_leg::{WlfmChild, faults, wlfm_leg_arena, wlfm_leg_program};
use super::wlfm_tests::{HASHER, build, honest, prove, under};
use super::word::LfmWord;

fn child_of(build: &WlfmBuild, words: usize) -> WlfmChild {
    WlfmChild {
        artifacts: build.artifacts.clone(),
        num_public_words: words,
    }
}

fn runs(program: &LfmProgram, arena: Vec<LfmWord>) -> Result<Vec<(u32, LfmWord)>, String> {
    execute(program, &[arena], &HASHER)
        .map(|e| e.public_words)
        .map_err(|e| format!("{e:?}"))
}

/// The honest child at `rate`: its build, its proof, the leg and its arena.
fn honest_child(rate: WlfmRate) -> (WlfmChild, WlfmProof, LfmProgram, Vec<LfmWord>) {
    let program = honest();
    let build = build(&program, rate);
    let proof = prove(&program, &build);
    let child = child_of(&build, proof.public_words.len());
    let leg = wlfm_leg_program(&child).unwrap_or_else(|e| panic!("the leg at {rate:?}: {e}"));
    let arena =
        wlfm_leg_arena(&child, &proof).unwrap_or_else(|e| panic!("the arena at {rate:?}: {e}"));
    (child, proof, leg, arena)
}

/// ★ The W-leg executes an honest child's arena and republishes its words, at
/// blowup 2 and at blowup 16 (the first compression step's rate, which C2
/// verifies), and a tampered public word, root, commit-time answer, argue word
/// or prepared-opening word has no execution.
#[test]
fn the_leg_executes_an_honest_child_and_refuses_tampered_ones() {
    let _small = test_format::set(Some(12));
    for rate in [WlfmRate::Two, WlfmRate::Sixteen] {
        let (_, proof, leg, arena) = honest_child(rate);
        assert_eq!(leg.arena_schema.lens, vec![arena.len() as u32]);
        let published =
            runs(&leg, arena.clone()).unwrap_or_else(|e| panic!("the honest leg at {rate:?}: {e}"));
        assert_eq!(published, proof.public_words);

        let words = proof.public_words.len();
        let roots = proof.proof.roots.len();
        let answers = proof.commit_ood.len();
        for (at, what) in [
            (0, "a public word"),
            (words, "a root"),
            (words + roots, "a commit-time answer"),
            (words + roots + answers, "the first argue word"),
            (arena.len() - 1, "the last prepared-opening word"),
        ] {
            let mut tampered = arena.clone();
            tampered[at][0] += FE::one();
            assert!(
                runs(&leg, tampered).is_err(),
                "{rate:?}: {what} tampered has no execution"
            );
        }
    }
}

/// A tampered proof, re-read into an arena (the arena checks shapes only):
/// the main opening's and the prepared opening's final values each move, and
/// the leg has no execution for either.
#[test]
fn the_leg_refuses_a_tampered_opening() {
    let _small = test_format::set(Some(12));
    let (child, proof, leg, _) = honest_child(WlfmRate::Sixteen);
    let mut main = proof.clone();
    main.proof.columns[0].polys[0].final_value += FEE::one();
    let arena = wlfm_leg_arena(&child, &main).expect("the shapes hold");
    assert!(runs(&leg, arena).is_err(), "a tampered main opening");
    let mut prepared = proof.clone();
    prepared.prepared[0].polys[0].final_value += FEE::one();
    let arena = wlfm_leg_arena(&child, &prepared).expect("the shapes hold");
    assert!(runs(&leg, arena).is_err(), "a tampered prepared opening");
}

/// The commit-time answers in the leg: absorbed after the draws, or the first
/// group's left out, make a program the honest arena does not execute — the
/// host draws `(z, α, β)` after every answer — and a walk reading its answers
/// from a fresh hint hints words the proof's arena does not hold.
#[test]
fn the_leg_absorbs_every_answer_before_the_draws_and_reads_the_fronts_cells() {
    use faults::Fault;
    let _small = test_format::set(Some(12));
    let (child, _, _, arena) = honest_child(WlfmRate::Sixteen);
    for fault in [Fault::AnswersAfterDraws, Fault::SkipFirstGroupsAnswers] {
        let mutated = faults::with(fault, || wlfm_leg_program(&child)).expect("the mutated leg");
        assert!(runs(&mutated, arena.clone()).is_err(), "{fault:?}");
    }
    let fresh =
        faults::with(Fault::FreshAnswerHint, || wlfm_leg_program(&child)).expect("the mutated leg");
    assert_ne!(fresh.arena_schema.lens, vec![arena.len() as u32]);
}

/// ⛔ The in-guest count trap: a child whose stacks settle fewer of a table's
/// preprocessed columns than its AIR declares, or none, is refused at emit
/// time — the leg's table walk checks no preprocessed column itself.
#[test]
fn the_leg_refuses_a_child_whose_stacks_leave_a_prefix_unbound() {
    let _small = test_format::set(Some(12));
    let program = honest();
    let build = build(&program, WlfmRate::Two);
    let mut short = child_of(&build, 4);
    let stack = short
        .artifacts
        .stacks
        .iter_mut()
        .find(|s| !s.tables.is_empty())
        .expect("a stack");
    stack.tables[0].1 -= 1;
    assert!(wlfm_leg_program(&short).is_err());
    let mut bare = child_of(&build, 4);
    bare.artifacts.stacks.clear();
    assert!(wlfm_leg_program(&bare).is_err());
}

/// ⛔ A child only the host can check is refused at emit time: one bound by
/// evaluation (no stack to open in-guest), one whose chains stop at a final
/// polynomial, one on the byte-minimising schedule.
#[test]
fn the_leg_refuses_a_child_only_the_host_can_check() {
    let _small = test_format::set(Some(12));
    let program = honest();
    for levers in [
        WlfmLevers::with_prep(WlfmPrep::Evaluated),
        WlfmLevers {
            prep: WlfmPrep::Stacked,
            final_poly: multilinear::whir_chain::FinalPoly::Vars(6),
            fold: None,
            grind: None,
        },
        super::compress::folds_levers(WlfmPrep::Stacked),
    ] {
        let build = under(&program, WlfmRate::Sixteen, levers);
        let child = child_of(&build, 4);
        assert!(wlfm_leg_program(&child).is_err(), "{levers:?}");
    }
}
