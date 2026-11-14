//! Laptop gates for the W-leg's BLAKE3 arm ([`super::wlfm_leg`] under
//! [`WlfmFamily::B3`]): the in-guest verifier of one W-LFM proof made under
//! 7-round BLAKE3 (`Blake3Whir` commitments, the `Blake3Std` transcript)
//! executes an honest child's arena and republishes its words, hashing with
//! the lean tables' rows alone, and has no execution for a child whose proof
//! was tampered anywhere the leg reads, a root's second word included. A
//! BLAKE3 child is walked under BLAKE3 only, and only on the lean rows.
//!
//! The fixture is [`super::wlfm_tests`]'s program at a small query count, as
//! the Poseidon1 leg's gates take it.

use crate::tables::types::{FE, FEE};

use super::compiler::LfmProgram;
use super::executor::execute;
use super::instr::Instr;
use super::wlfm::{
    WlfmBuild, WlfmFamily, WlfmLevers, WlfmProof, WlfmRate, test_format, verify_wlfm,
};
use super::wlfm_leg::{
    WlfmChild, emit_wlfm_leg, wlfm_leg_arena, wlfm_leg_builder, wlfm_leg_program,
};
use super::wlfm_tests::{HASHER, honest, prove, under};
use super::word::LfmWord;

fn b3_build(program: &LfmProgram, rate: WlfmRate) -> WlfmBuild {
    under(program, rate, WlfmLevers::with_family(WlfmFamily::B3))
}

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

/// The honest BLAKE3 child at `rate`: its proof (verified on the host), the
/// leg and its arena.
fn b3_child(rate: WlfmRate) -> (WlfmChild, WlfmProof, LfmProgram, Vec<LfmWord>) {
    let program = honest();
    let build = b3_build(&program, rate);
    assert_eq!(build.artifacts.family, WlfmFamily::B3);
    let proof = prove(&program, &build);
    verify_wlfm(&build.artifacts, &proof, &proof.public_words)
        .unwrap_or_else(|e| panic!("the host verifies the BLAKE3 child at {rate:?}: {e}"));
    let child = child_of(&build, proof.public_words.len());
    let leg = wlfm_leg_program(&child).unwrap_or_else(|e| panic!("the leg at {rate:?}: {e}"));
    let arena =
        wlfm_leg_arena(&child, &proof).unwrap_or_else(|e| panic!("the arena at {rate:?}: {e}"));
    (child, proof, leg, arena)
}

/// The leg's hash rows: `(lean BLAKE3, any other hash row)`.
fn hash_rows(program: &LfmProgram) -> (usize, usize) {
    let lean = program
        .instrs
        .iter()
        .filter(|i| matches!(i, Instr::B3l(_)))
        .count();
    let other = program
        .instrs
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instr::Hash { .. } | Instr::Hash16(_) | Instr::Blake3(_) | Instr::KeccakF(_)
            )
        })
        .count();
    (lean, other)
}

/// ★ THE GATE. The BLAKE3 leg executes an honest BLAKE3 child's arena and
/// republishes its words, at blowup 2 and at blowup 16, hashing with lean
/// rows only; and a tampered public word, root (its first word, and its
/// second), commit-time answer, argue word or prepared-opening word has no
/// execution.
#[test]
fn the_blake3_leg_executes_an_honest_child_and_refuses_tampered_ones() {
    let _small = test_format::set(Some(12));
    for rate in [WlfmRate::Two, WlfmRate::Sixteen] {
        let (_, proof, leg, arena) = b3_child(rate);
        let (lean, other) = hash_rows(&leg);
        assert!(lean > 0, "{rate:?}: the leg hashes with the lean rows");
        assert_eq!(other, 0, "{rate:?}: and with nothing else");
        assert_eq!(leg.arena_schema.lens, vec![arena.len() as u32]);
        let published =
            runs(&leg, arena.clone()).unwrap_or_else(|e| panic!("the honest leg at {rate:?}: {e}"));
        assert_eq!(published, proof.public_words);
        println!(
            "BLAKE3 W-LEG {rate:?}: {} instructions · {lean} compressions · arena {} words",
            leg.instrs.len(),
            arena.len()
        );

        let words = proof.public_words.len();
        let roots = 2 * proof.proof.roots.len();
        let answers = proof.commit_ood.len();
        for (at, lane, what) in [
            (0, 0, "a public word"),
            (words, 0, "a root's first word"),
            (words + 1, 3, "a root's second word"),
            (words + roots, 0, "a commit-time answer"),
            (words + roots + answers, 0, "the first argue word"),
            (arena.len() - 1, 0, "the last prepared-opening word"),
        ] {
            let mut tampered = arena.clone();
            tampered[at][lane] += FE::one();
            assert!(
                runs(&leg, tampered).is_err(),
                "{rate:?}: {what} tampered has no execution"
            );
        }
    }
}

/// A tampered BLAKE3 proof re-read into an arena: the main opening's and the
/// prepared opening's final values, and a root's last byte, each move, and the
/// leg has no execution for any of them.
#[test]
fn the_blake3_leg_refuses_a_tampered_opening() {
    let _small = test_format::set(Some(12));
    let (child, proof, leg, _) = b3_child(WlfmRate::Sixteen);
    let mut main = proof.clone();
    main.proof.columns[0].polys[0].final_value += FEE::one();
    let arena = wlfm_leg_arena(&child, &main).expect("the shapes hold");
    assert!(runs(&leg, arena).is_err(), "a tampered main opening");
    let mut prepared = proof.clone();
    prepared.prepared[0].polys[0].final_value += FEE::one();
    let arena = wlfm_leg_arena(&child, &prepared).expect("the shapes hold");
    assert!(runs(&leg, arena).is_err(), "a tampered prepared opening");
    let mut root = proof.clone();
    root.proof.roots[0][31] ^= 0x80;
    let arena = wlfm_leg_arena(&child, &root).expect("the shapes hold");
    assert!(runs(&leg, arena).is_err(), "a root's last byte");
}

/// ⛔ A BLAKE3 child is walked under BLAKE3 only, and only on the lean rows:
/// a Poseidon1 builder refuses it, a BLAKE3 builder without the lean rows
/// refuses it, and a Poseidon1 child refuses a BLAKE3 builder. A Poseidon1
/// child's proof has no execution in the BLAKE3 child's leg.
#[test]
fn a_blake3_child_is_walked_under_blake3_alone() {
    use super::builder::LfmBuilder;
    use super::edsl::WrapHash;
    let _small = test_format::set(Some(12));
    let program = honest();
    let b3 = b3_build(&program, WlfmRate::Two);
    let p1 = under(&program, WlfmRate::Two, WlfmLevers::DEFAULT);
    let b3_proof = prove(&program, &b3);
    let p1_proof = prove(&program, &p1);
    let b3_child = child_of(&b3, b3_proof.public_words.len());
    let p1_child = child_of(&p1, p1_proof.public_words.len());
    assert_ne!(b3.artifacts.program_id, p1.artifacts.program_id);

    let mut b = wlfm_leg_builder(WlfmFamily::P1);
    let arena = b.declare_arena(0);
    assert!(
        emit_wlfm_leg(&mut b, &b3_child, arena).is_err(),
        "a Poseidon1 builder"
    );
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let arena = b.declare_arena(0);
    let Err(err) = emit_wlfm_leg(&mut b, &b3_child, arena) else {
        panic!("a BLAKE3 builder without the lean rows walked a BLAKE3 child");
    };
    assert!(err.contains("lean"), "{err}");
    let mut b = wlfm_leg_builder(WlfmFamily::B3);
    let arena = b.declare_arena(0);
    assert!(
        emit_wlfm_leg(&mut b, &p1_child, arena).is_err(),
        "a BLAKE3 builder over P1"
    );

    let leg = wlfm_leg_program(&b3_child).expect("the BLAKE3 leg");
    runs(
        &leg,
        wlfm_leg_arena(&b3_child, &b3_proof).expect("its arena"),
    )
    .expect("its own child");
    match wlfm_leg_arena(&b3_child, &p1_proof) {
        Err(_) => {}
        Ok(arena) => assert!(
            runs(&leg, arena).is_err(),
            "a Poseidon1 child's proof in the BLAKE3 leg"
        ),
    }
}
