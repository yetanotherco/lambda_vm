//! The one-output select ([`Instr::Mux`]) and its lean table `LEAN_MUX`: in a
//! lean program a `Select` of which one output nothing reads is compiled to a
//! mux row, proved by the W-LFM prover's lean tables, and refused when its
//! chosen word is not the bit's arm.

use crate::tables::types::FE;

use super::b3l_tables::{MUX_NAME, test_mux};
use super::builder::{B3Block, B3Cv, Cell, LfmBuilder};
use super::chips::mux::MuxConstraints;
use super::compiler::{LfmProgram, compile};
use super::edsl::WrapHash;
use super::executor::execute;
use super::hash::HasherKind;
use super::instr::Instr;
use super::wlfm::{
    WlfmBuild, WlfmRate, build_wlfm, prove_wlfm, prove_wlfm_traces, test_format, verify_wlfm,
};

const HASHER: HasherKind = HasherKind::Poseidon1W16;

fn word(b: &mut LfmBuilder, v: u64) -> Cell {
    b.digest_const([FE::from(v); 4]).as_cell()
}

/// One compression (so the program has lean tables), three bits of `0b101`,
/// and three selects of the words 10 and 20: one read on its left output, one
/// on its right, one on both; every read output published, in that order.
fn program(builder: LfmBuilder) -> LfmProgram {
    let mut b = builder;
    let block: [Cell; 4] = core::array::from_fn(|i| word(&mut b, i as u64 + 1));
    b.b3_compress(B3Cv::Iv, B3Block::Lanes(block), 64, 11);
    let x = b.felt_const(FE::from(0b101u64));
    let bits = b.bit_dec(x, 3);
    let (l, r) = (word(&mut b, 10), word(&mut b, 20));
    let (a, _) = b.select(bits[0], l, r);
    let (_, c) = b.select(bits[1], l, r);
    let (d, e) = b.select(bits[2], l, r);
    for cell in [a, c, d, e] {
        b.public(cell);
    }
    compile(b.finish())
}

fn lean() -> LfmProgram {
    program(
        LfmBuilder::new()
            .with_wrap_hash(WrapHash::Blake3)
            .with_b3_lean(true),
    )
}

fn count(program: &LfmProgram, f: impl Fn(&Instr) -> bool) -> usize {
    program.instrs.iter().filter(|i| f(i)).count()
}

fn build(program: &LfmProgram) -> WlfmBuild {
    build_wlfm(program, HASHER, WlfmRate::Two).unwrap_or_else(|e| panic!("the W-LFM build: {e}"))
}

/// Only a lean program's one-output selects become muxes, and the words they
/// publish are the selects': `bit = 1` picks the right arm, a right output
/// under `bit = 0` is the right arm, both outputs under `bit = 1` swap.
#[test]
fn one_output_selects_become_muxes_in_a_lean_program_only() {
    let lean = lean();
    let plain = program(LfmBuilder::new());
    assert_eq!(count(&lean, |i| matches!(i, Instr::Mux { .. })), 2);
    assert_eq!(count(&lean, |i| matches!(i, Instr::Select { .. })), 1);
    assert_eq!(count(&plain, |i| matches!(i, Instr::Mux { .. })), 0);
    assert_eq!(count(&plain, |i| matches!(i, Instr::Select { .. })), 3);
    assert_eq!(lean.groups.mux.real_rows, 2);
    super::validator::validate(&lean).expect("admissible");
    let published = |p: &LfmProgram| -> Vec<u64> {
        execute(p, &[], &HASHER)
            .expect("executes")
            .public_words
            .iter()
            .map(|(_, w)| w[0].canonical())
            .collect()
    };
    assert_eq!(published(&lean), vec![20, 20, 20, 10]);
    assert_eq!(published(&lean), published(&plain));
}

/// ★ A lean program's muxes prove and verify through W-LFM: the mux chunk is
/// the last lean table, and the shape folds into the program's identity.
#[test]
fn muxes_round_trip_through_wlfm() {
    let _fast = test_format::set(Some(2));
    let program = lean();
    let build = build(&program);
    let a = &build.artifacts;
    assert_eq!(a.b3l.muxes.len(), 1, "one mux chunk");
    let airs = a.airs();
    assert_eq!(airs.air_refs().last().map(|r| r.name()), Some(MUX_NAME));
    let proof = prove_wlfm(&program, &build, &[]).unwrap_or_else(|e| panic!("proves: {e}"));
    verify_wlfm(a, &proof, &proof.public_words).unwrap_or_else(|e| panic!("verifies: {e}"));
}

/// The first mux row (the left-read select, `bit = 1`) forged to choose the
/// left arm, with the word it publishes forged to match: the bus balances,
/// and only the mux's choice constraint stands between the forger and a
/// false word.
fn forged_proof(build: &WlfmBuild, program: &LfmProgram) -> super::wlfm::WlfmProof {
    let mut exec = execute(program, &[], &HASHER).expect("executes");
    let left = exec.records.mux[0].in_l;
    assert_ne!(
        exec.records.mux[0].out, left,
        "the honest choice is the right arm"
    );
    exec.records.mux[0].out = left;
    exec.records.public[0] = left;
    exec.public_words[0].1 = left;
    let filled = super::proof::lfm_fill_executed_under(program, exec, HASHER, 0.0, &Vec::new());
    prove_wlfm_traces(build, filled, true).expect("the prover does not check the choice")
}

/// ★ MUTATION: the choice constraint is load-bearing. The forged mux row is
/// refused; with the choice dropped on both sides it is accepted.
#[test]
fn the_mux_choice_constraint_is_load_bearing() {
    let _fast = test_format::set(Some(2));
    let program = lean();
    let run = |drop: bool| {
        let _mutant = test_mux::set(MuxConstraints { drop_choice: drop });
        let build = build(&program);
        let proof = forged_proof(&build, &program);
        verify_wlfm(&build.artifacts, &proof, &proof.public_words)
    };
    assert!(
        run(false).is_err(),
        "the choice constraint refuses the forged arm"
    );
    run(true).unwrap_or_else(|e| panic!("with the choice dropped the forgery is accepted: {e}"));
}

/// A mux row's chosen word changed alone (nothing forged to match): the bus
/// refuses it.
#[test]
fn a_tampered_mux_output_is_refused() {
    let _fast = test_format::set(Some(2));
    let program = lean();
    let build = build(&program);
    let mut exec = execute(&program, &[], &HASHER).expect("executes");
    exec.records.mux[1].out[0] += FE::one();
    let filled = super::proof::lfm_fill_executed_under(&program, exec, HASHER, 0.0, &Vec::new());
    let proof = prove_wlfm_traces(&build, filled, true).expect("the prover does not check");
    assert!(verify_wlfm(&build.artifacts, &proof, &proof.public_words).is_err());
}

/// The BLAKE3 W-leg over a lean child takes its one-output selects as mux
/// rows: every select whose second output nothing reads (the cap's and the
/// slots' muxes) is one, and the leg still executes over the child's proof.
#[test]
fn the_blake3_w_leg_takes_its_muxes_as_mux_rows() {
    use super::wlfm::{WlfmFamily, WlfmLevers, build_wlfm_under};
    let _fast = test_format::set(Some(2));
    let program = lean();
    let child_build = build_wlfm_under(
        &program,
        HASHER,
        WlfmRate::Two,
        WlfmLevers::with_family(WlfmFamily::B3),
        true,
    )
    .expect("the child builds");
    let proof = prove_wlfm(&program, &child_build, &[]).expect("the child proves");
    let child = super::wlfm_leg::WlfmChild {
        artifacts: child_build.artifacts.clone(),
        num_public_words: proof.public_words.len(),
        cap_mux: None,
    };
    let leg = super::wlfm_leg::wlfm_leg_program(&child).expect("the BLAKE3 leg emits");
    let muxes = count(&leg, |i| matches!(i, Instr::Mux { .. }));
    let selects = count(&leg, |i| matches!(i, Instr::Select { .. }));
    let unread_selects = leg
        .instrs
        .iter()
        .filter(
            |i| matches!(i, Instr::Select { mult_l, mult_r, .. } if *mult_l == 0 || *mult_r == 0),
        )
        .count();
    println!(
        "LEAN MUX W-LEG: {muxes} mux rows · {selects} select rows ({unread_selects} with an unread \
         output) · select-type cells {} (were {})",
        muxes * (super::layout::mux::PREP_WIDTH + 13)
            + selects * (super::layout::select::PREP_WIDTH + 17),
        (muxes + selects) * (super::layout::select::PREP_WIDTH + 17)
    );
    assert!(muxes > 0, "the leg's muxes are mux rows");
    assert_eq!(unread_selects, 0, "no select keeps an unread output");
    let arena = super::wlfm_leg::wlfm_leg_arena(&child, &proof).expect("the leg's arena");
    execute(&leg, &[arena], &leg.hasher(HASHER)).expect("the leg executes over the child");
}

/// ★ The BLAKE3 W-leg over the lean child, pinned by its program's digest.
/// A W-leg plans its child's batched argue on the one-layer GKR ladder only
/// (`child_planning`), so its program is the one it was before the recursion
/// leaf learned the two-layer ladder (the digest is the one the same test
/// gives at the head before it, built there and here, equal); a change to the
/// leg's batched walk, its hints or its planning moves it. Not blind: with the
/// leg's closure assert left out, the digest moves.
#[test]
fn the_blake3_w_leg_program_is_pinned() {
    use super::wlfm::{WlfmFamily, WlfmLevers, build_wlfm_under};
    const DIGEST: &str = "983a387d28f3aa482e7c4514850a621f6dade5edb91e51867d859854762050e7";
    let _fast = test_format::set(Some(2));
    let program = lean();
    let child_build = build_wlfm_under(
        &program,
        HASHER,
        WlfmRate::Two,
        WlfmLevers::with_family(WlfmFamily::B3),
        true,
    )
    .expect("the child builds");
    let proof = prove_wlfm(&program, &child_build, &[]).expect("the child proves");
    let child = super::wlfm_leg::WlfmChild {
        artifacts: child_build.artifacts.clone(),
        num_public_words: proof.public_words.len(),
        cap_mux: None,
    };
    let leg = super::wlfm_leg::wlfm_leg_program(&child).expect("the BLAKE3 leg emits");
    let digest = super::whir_chain_tests::program_digest(&leg);
    println!(
        "W-LEG PIN: {} instructions · digest {digest}",
        leg.instrs.len()
    );
    assert_eq!(digest, DIGEST, "the BLAKE3 W-leg's program");
    // The LFM_HASH split is inert on a BLAKE3 leg: it has no hash row a split
    // could cut, so `LAMBDA_VM_LFM_HASH_SPLIT` moves no BLAKE3 program.
    let rows = leg.groups.hash.real_rows;
    assert_eq!(
        super::chunking::HashChunking::for_rows(rows),
        super::chunking::HashChunking::unbounded(),
        "{rows} hash rows a split would cut"
    );
    // The cap check named by the deriver: the default's own (the same
    // program), and another (another program).
    let named = |mode| {
        let child = super::wlfm_leg::WlfmChild {
            cap_mux: Some(mode),
            ..child.clone()
        };
        super::whir_chain_tests::program_digest(
            &super::wlfm_leg::wlfm_leg_program(&child).expect("the named leg emits"),
        )
    };
    assert_eq!(
        named(super::merkle_cap::CapMux::Packed),
        DIGEST,
        "the packed check named: the default's program"
    );
    assert_ne!(
        named(super::merkle_cap::CapMux::Plain),
        DIGEST,
        "the plain check named: another program"
    );
    use super::wlfm_leg::faults::{Fault, with};
    let moved = with(Fault::SkipClosure, || {
        super::wlfm_leg::wlfm_leg_program(&child).expect("the faulted leg emits")
    });
    assert_ne!(
        super::whir_chain_tests::program_digest(&moved),
        DIGEST,
        "the pin sees the leg"
    );
}
