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
use super::wlfm::{
    WlfmBuild, WlfmLevers, WlfmPrep, WlfmProof, WlfmRate, execute_and_fill_unmerged,
    prove_wlfm_traces, test_format, verify_wlfm,
};
use super::wlfm_leg::{WlfmChild, faults, wlfm_leg_arena, wlfm_leg_program};
use super::wlfm_tests::{HASHER, arenas, build, honest, prove, under};
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

/// ★ The second step in small: the leg over a blowup-16 child, proved by the
/// W-LFM prover at blowup 64 under C2's levers (evaluated columns, the
/// byte-minimising schedule, a 22-bit grind), verifies on the host and
/// publishes the child's words; a tampered child proof has no execution, so
/// no proof.
#[test]
fn the_leg_proved_at_blowup_64_under_the_second_steps_levers_verifies() {
    use super::wlfm::{build_wlfm_under, prove_wlfm, verify_wlfm};
    use crate::hash_pin::{BlockFamily, ProofFamily};
    let _small = test_format::set(Some(12));
    let (_, proof, leg, arena) = honest_child(WlfmRate::Sixteen);
    let hasher = leg.hasher(<BlockFamily as ProofFamily>::SOCKET);
    let c2 = build_wlfm_under(
        &leg,
        hasher,
        WlfmRate::SixtyFour,
        super::compress::compress2_levers(22),
        false,
    )
    .unwrap_or_else(|e| panic!("the second step builds: {e}"));
    assert_eq!(c2.artifacts.config.log_blowup, 6);
    assert_eq!(c2.artifacts.prep, WlfmPrep::Evaluated);
    let c2_proof = prove_wlfm(&leg, &c2, std::slice::from_ref(&arena))
        .unwrap_or_else(|e| panic!("the second step proves: {e}"));
    assert_eq!(c2_proof.public_words, proof.public_words);
    verify_wlfm(&c2.artifacts, &c2_proof, &c2_proof.public_words)
        .unwrap_or_else(|e| panic!("the second step verifies: {e}"));
    let mut tampered = arena.clone();
    tampered[proof.public_words.len()][0] += FE::one();
    assert!(
        prove_wlfm(&leg, &c2, &[tampered]).is_err(),
        "a tampered child has no execution"
    );
}

/// ★ The closure is load-bearing: a proof of the honest trace under a FALSE
/// claimed word (the forger's statement absorbs the false word, so the
/// transcript is consistent with it) is refused by the host verifier (its
/// `LfmPublic` balance) and has no execution under the leg; with the leg's
/// closure assert left out, the same arena executes and republishes the false
/// word.
#[test]
fn the_legs_public_balance_refuses_a_false_word_the_transcript_carries() {
    let _small = test_format::set(Some(12));
    let program = honest();
    let build = build(&program, WlfmRate::Sixteen);
    let mut filled =
        execute_and_fill_unmerged(&program, &arenas(), HASHER).expect("the program executes");
    filled.public_words_mut()[0].1[0] += FE::one();
    let forged = prove_wlfm_traces(&build, filled, true)
        .unwrap_or_else(|e| panic!("the forger's prover proves a false claim: {e}"));
    assert!(
        verify_wlfm(&build.artifacts, &forged, &forged.public_words).is_err(),
        "the host refuses the false word"
    );
    let child = child_of(&build, forged.public_words.len());
    let arena = wlfm_leg_arena(&child, &forged).expect("the shapes hold");
    let leg = wlfm_leg_program(&child).expect("the leg");
    assert!(
        runs(&leg, arena.clone()).is_err(),
        "the leg refuses the false word"
    );
    let open = faults::with(faults::Fault::SkipClosure, || wlfm_leg_program(&child))
        .expect("the leg without its closure");
    let published = runs(&open, arena).unwrap_or_else(|e| {
        panic!("without its closure the leg accepts the false word (else the test isolates nothing): {e}")
    });
    assert_eq!(published, forged.public_words);
}
