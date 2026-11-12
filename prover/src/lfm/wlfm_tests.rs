//! Laptop gates for the W-LFM proof ([`super::wlfm`]): round trips at each
//! rate, and one named refusal per way a W-LFM proof can fail to bind its
//! program or its words.
//!
//! ⚠ The fixture is a small `Hash16` program and a FORGED twin of it: the same
//! program with one constant changed, so every table keeps its height and the
//! execution completes. The twin's proof is a valid proof of a DIFFERENT
//! program — exactly what the preprocessed columns must refuse, and nothing else
//! about it can.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::element::FieldElement;
use multilinear::whir_hash::WhirHash;
use stark::multilinear_table::TableStatement;
use stark::multilinear_table::batched::VerifierChecks;

use crate::tables::types::{FE, FEE, GoldilocksExtension as E, GoldilocksField as F};

use super::builder::{Cell, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::hash::HasherKind;
use super::wlfm::{
    WLFM_B4_QUERIES, WLFM_B16_STACK, WlfmBuild, WlfmHash, WlfmProof, WlfmRate,
    absorb_wlfm_statement, build_wlfm, check_coverage, execute_and_fill_unmerged, prove_wlfm,
    prove_wlfm_traces, test_format, verify_wlfm, verify_wlfm_with, wlfm_config,
};
use super::word::LfmWord;

const HASHER: HasherKind = HasherKind::Poseidon1W16;

fn word(seed: u64) -> [FE; 4] {
    core::array::from_fn(|i| FE::from(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i as u64))
}

/// The fixture's arena: four arbitrary words.
fn arenas() -> Vec<Vec<LfmWord>> {
    vec![(0..4u64).map(|i| word(1_000 + i)).collect()]
}

/// A program touching every WHIR recursion chip: the base ALU, the extension
/// ALU, a bit decomposition driving two selects, two chained `Hash16` rows
/// over hinted words, and four published words. `y` is the constant the
/// forged twin changes (`q = (x + y)·x / (x + y) = x` for any `y`).
fn program(y: u64) -> LfmProgram {
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(4);
    let h: Vec<Cell> = (0..4).map(|i| b.hint_word(arena, i)).collect();
    let x = b.felt_const(FE::from(7u64));
    let yc = b.felt_const(FE::from(y));
    let s = b.add(x, yc);
    let m = b.mul(s, x);
    let q = b.div(m, s);
    b.assert_eq(q, x);
    let e1 = b.ext_const(&FEE::new([FE::from(1u64), FE::from(2u64), FE::from(3u64)]));
    let e2 = b.ext_const(&FEE::new([FE::from(4u64), FE::from(5u64), FE::from(6u64)]));
    let p = b.emul(e1, e2);
    let pm = b.emul_add(p, e1, e2);
    let pb = b.emul_base(pm, q);
    let bits = b.bit_dec(m, 8);
    let (l, _) = b.select(bits[4], h[0], h[1]);
    let (l2, _) = b.select(bits[0], l, h[2]);
    let first = b.hash16([h[0], l2, h[2], h[3]]);
    let second = b.hash16([first[0], h[1], first[1], h[3]]);
    b.public(second[0]);
    b.public(second[3]);
    b.public(pb.as_cell());
    b.public(m.as_cell());
    compile(b.finish())
}

fn honest() -> LfmProgram {
    program(9)
}

fn forged() -> LfmProgram {
    program(10)
}

fn build(program: &LfmProgram, rate: WlfmRate) -> WlfmBuild {
    build_wlfm(program, HASHER, rate).unwrap_or_else(|e| panic!("the W-LFM build: {e}"))
}

fn prove(program: &LfmProgram, build: &WlfmBuild) -> WlfmProof {
    prove_wlfm(program, build, &arenas()).unwrap_or_else(|e| panic!("the W-LFM prove: {e}"))
}

/// `program`'s execution proved against `build`, WITHOUT the prover's prefix
/// check: the forger's prover.
fn forge(program: &LfmProgram, build: &WlfmBuild) -> WlfmProof {
    let filled =
        execute_and_fill_unmerged(program, &arenas(), HASHER).expect("the program executes");
    prove_wlfm_traces(build, filled, false).unwrap_or_else(|e| panic!("the forger's prover: {e}"))
}

// =============================================================================
// Round trips and the format
// =============================================================================

/// ★ THE ANCHOR: an honest W-LFM proof at the production config — Poseidon1,
/// the base's chain format, grinding included — verifies, and carries one
/// commit-time answer per root and one opening per prepared stack.
#[test]
fn the_fixture_round_trips_at_the_production_config() {
    let program = honest();
    assert_eq!(program.hasher(HasherKind::Rpx), HASHER, "a Hash16 program");
    let build = build(&program, WlfmRate::Two);
    let a = &build.artifacts;
    assert_eq!(
        a.config.grind.query,
        crate::zf_format::PRODUCTION_WHIR_GRIND_BITS
    );
    assert_eq!(a.config.log_blowup, 1);
    assert!(!a.stacks.is_empty(), "the program's prefixes are stacked");
    let proof = prove(&program, &build);
    assert_eq!(proof.commit_ood.len(), proof.proof.roots.len());
    assert_eq!(proof.prepared.len(), a.stacks.len());
    assert_eq!(proof.argues.len(), a.sizes.len());
    verify_wlfm(a, &proof, &proof.public_words)
        .unwrap_or_else(|e| panic!("the honest W-LFM proof must verify: {e}"));
}

/// The blowup-4 arm round-trips too, at [`WLFM_B4_QUERIES`], and is another
/// program identity: a proof at one rate is refused by the other's artifacts.
#[test]
fn the_blowup_4_arm_round_trips_and_is_another_identity() {
    let _unground = test_format::set(None);
    let program = honest();
    let two = build(&program, WlfmRate::Two);
    let four = build(&program, WlfmRate::Four);
    assert_eq!(four.artifacts.config.log_blowup, 2);
    assert_eq!(four.artifacts.config.num_queries, WLFM_B4_QUERIES);
    assert_ne!(two.artifacts.program_id, four.artifacts.program_id);
    let proof = prove(&program, &four);
    verify_wlfm(&four.artifacts, &proof, &proof.public_words)
        .unwrap_or_else(|e| panic!("the blowup-4 proof must verify: {e}"));
    assert!(
        verify_wlfm(&two.artifacts, &proof, &proof.public_words).is_err(),
        "a blowup-4 proof is refused at blowup 2"
    );
}

/// The blowup-16 arm (the compression step's rate) round-trips at the
/// production query rule under the 2^26 stack, and is another identity than
/// blowup 2: a proof at one rate is refused by the other's artifacts.
#[test]
fn the_blowup_16_arm_round_trips_and_is_another_identity() {
    let _unground = test_format::set(None);
    let program = honest();
    let two = build(&program, WlfmRate::Two);
    let sixteen = build(&program, WlfmRate::Sixteen);
    let c = &sixteen.artifacts.config;
    assert_eq!(c.log_blowup, 4);
    assert_eq!(c.format.stack.get(), WLFM_B16_STACK);
    assert!(
        c.num_queries < two.artifacts.config.num_queries,
        "blowup 16 runs fewer queries: {} vs {}",
        c.num_queries,
        two.artifacts.config.num_queries
    );
    assert_ne!(two.artifacts.program_id, sixteen.artifacts.program_id);
    let proof = prove(&program, &sixteen);
    verify_wlfm(&sixteen.artifacts, &proof, &proof.public_words)
        .unwrap_or_else(|e| panic!("the blowup-16 proof must verify: {e}"));
    assert!(
        verify_wlfm(&two.artifacts, &proof, &proof.public_words).is_err(),
        "a blowup-16 proof is refused at blowup 2"
    );
}

/// [`WLFM_B4_QUERIES`] is the fewest queries that keep the unique-decoding
/// figure at blowup 4 at or above the base's 113.04 bits.
#[test]
fn the_blowup_4_query_count_keeps_the_proven_bits() {
    let per_query = (8.0f64 / 5.0).log2();
    let grind = f64::from(crate::zf_format::PRODUCTION_WHIR_GRIND_BITS);
    let bits = |q: usize| q as f64 * per_query + grind;
    assert!(bits(WLFM_B4_QUERIES) >= 113.04, "{}", bits(WLFM_B4_QUERIES));
    assert!(bits(WLFM_B4_QUERIES - 1) < 113.04);
}

/// The format is the code's constant: neither module reads the environment or
/// the process's format, so a verifier's environment cannot choose the config
/// it checks a proof under.
#[test]
fn the_wlfm_format_reads_no_environment() {
    for (name, src) in [("wlfm.rs", include_str!("wlfm.rs"))] {
        for forbidden in ["std::env", "ZfFormat::global", "env::var"] {
            assert!(!src.contains(forbidden), "{name} reads `{forbidden}`");
        }
    }
    let shapes = [(329usize, 18usize), (23, 21), (25, 20)];
    let config = wlfm_config(WlfmRate::Two, &shapes);
    let base =
        crate::multilinear_prove::chain_config_under(&crate::zf_format::ZfFormat::DEFAULT, &shapes);
    assert_eq!(config.num_queries, base.num_queries);
    assert_eq!(config.log_blowup, base.log_blowup);
    assert_eq!(
        config.format.argue,
        multilinear::whir_chain::ArgueFormat::BATCHED
    );
}

// =============================================================================
// Refusals
// =============================================================================

/// A tampered public word is refused: the statement absorbs it, so every
/// challenge moves and the `LfmPublic` balance is another.
#[test]
fn a_tampered_public_word_is_refused() {
    let _unground = test_format::set(None);
    let program = honest();
    let build = build(&program, WlfmRate::Two);
    let proof = prove(&program, &build);
    let mut claimed = proof.public_words.clone();
    claimed[0].1[0] += FE::one();
    assert!(verify_wlfm(&build.artifacts, &proof, &claimed).is_err());
    let mut reindexed = proof.public_words.clone();
    reindexed[1].0 = 7;
    assert!(verify_wlfm(&build.artifacts, &proof, &reindexed).is_err());
}

/// The commit-time answers on a W-LFM proof: a false one, a missing one and an
/// extra one are each refused.
#[test]
fn a_false_missing_or_extra_commit_time_answer_is_refused() {
    let _unground = test_format::set(None);
    let program = honest();
    let build = build(&program, WlfmRate::Two);
    let proof = prove(&program, &build);
    let mut altered = proof.clone();
    altered.commit_ood[0] += FieldElement::<E>::one();
    assert!(verify_wlfm(&build.artifacts, &altered, &altered.public_words).is_err());
    let mut missing = proof.clone();
    missing.commit_ood.pop();
    assert!(verify_wlfm(&build.artifacts, &missing, &missing.public_words).is_err());
    let mut extra = proof.clone();
    extra.commit_ood.push(FieldElement::<E>::one());
    assert!(verify_wlfm(&build.artifacts, &extra, &extra.public_words).is_err());
    let mut extra_ok = proof;
    extra_ok.commit_ood.truncate(extra_ok.proof.roots.len());
    verify_wlfm(&build.artifacts, &extra_ok, &extra_ok.public_words)
        .unwrap_or_else(|e| panic!("the honest answers verify: {e}"));
}

/// ★ A forged instruction column — the twin's program proved against the
/// honest program's prepared stacks — is refused by the prepared opening, and
/// only by it: with the openings skipped, the same proof verifies.
#[test]
fn a_forged_instruction_column_is_refused_by_the_prepared_opening() {
    let _unground = test_format::set(None);
    let build = build(&honest(), WlfmRate::Two);
    let proof = forge(&forged(), &build);
    assert!(
        verify_wlfm(&build.artifacts, &proof, &proof.public_words).is_err(),
        "the forged program's proof is refused"
    );
    verify_wlfm_with(
        &build.artifacts,
        &proof,
        &proof.public_words,
        true,
        VerifierChecks::ALL,
    )
    .unwrap_or_else(|e| panic!("with the prepared openings skipped the forgery verifies: {e}"));
}

/// The prepared openings deleted from an honest proof are refused (the
/// omission, not a reordering).
#[test]
fn a_deleted_prepared_opening_is_refused() {
    let _unground = test_format::set(None);
    let program = honest();
    let build = build(&program, WlfmRate::Two);
    let mut proof = prove(&program, &build);
    proof.prepared.clear();
    assert!(verify_wlfm(&build.artifacts, &proof, &proof.public_words).is_err());
}

/// A restated height — artifacts claiming a table one variable taller — is
/// refused before any proof is read: the config, the groups and the layouts are
/// the artifacts' own derivation.
#[test]
fn a_restated_table_height_is_refused() {
    let _unground = test_format::set(None);
    let program = honest();
    let build = build(&program, WlfmRate::Two);
    let proof = prove(&program, &build);
    let mut restated = build.artifacts.clone();
    restated.table_num_vars[1] += 1;
    assert!(verify_wlfm(&restated, &proof, &proof.public_words).is_err());
}

/// What a verifier that counted the AIRs' `precomputed_columns()` (empty for
/// an LFM AIR) would run: every statement at count zero, no prepared check.
fn count_zero_verify(build: &WlfmBuild, proof: &WlfmProof) -> Result<(), String> {
    let a = &build.artifacts;
    let airs = a.airs();
    let refs = airs.air_refs();
    let shapes = a.shapes(&airs);
    let layouts: Vec<_> = refs
        .iter()
        .zip(&shapes)
        .map(|(air, &(w, n))| crate::multilinear_prove::layout_of(*air, w, n).expect("a layout"))
        .collect();
    let statements: Vec<TableStatement<'_, F, E>> = layouts.iter().map(|l| l.statement()).collect();
    let (stack_layouts, domains) =
        crate::multilinear_prove::stacks(&shapes, &a.sizes, &a.config).expect("the stacks");
    let mut transcript = <WlfmHash as WhirHash>::sponge();
    absorb_wlfm_statement(&mut transcript, &a.program_id, &proof.public_words);
    let mut probe = transcript.clone();
    stark::multilinear_block::absorb_roots_and_answers::<E, _>(
        &mut probe,
        &proof.proof.roots,
        &[],
        &proof.commit_ood,
    );
    let z: FieldElement<E> = probe.sample_field_element();
    let alpha: FieldElement<E> = probe.sample_field_element();
    let expected =
        super::proof::expected_public_balance(&proof.public_words, &z, &alpha).expect("a balance");
    stark::multilinear_block::block_verify_with::<F, E, _, WlfmHash>(
        &proof.proof,
        &proof.commit_ood,
        &proof.argues,
        &[],
        &[],
        &statements,
        &stack_layouts,
        &domains,
        &a.sizes,
        &expected,
        &a.config,
        &mut transcript,
        false,
        VerifierChecks::ALL,
    )
    .map_err(|e| format!("{e:?}"))
}

/// ★ THE COUNT TRAP, paired. A forged program proved with no prepared stacks
/// VERIFIES under a count-zero verifier — nothing binds its columns — and is
/// REFUSED under the AIRs' counts: by the honest artifacts (which stack every
/// prefix) and by stack-less artifacts (which the coverage rule refuses before
/// reading the proof).
#[test]
fn the_count_trap_binds_nothing_at_count_zero_and_is_refused_at_the_airs_count() {
    let _unground = test_format::set(None);
    let honest_build = build(&honest(), WlfmRate::Two);
    let bare = build(&honest(), WlfmRate::Two).without_stacks();
    let proof = forge(&forged(), &bare);
    assert!(proof.prepared.is_empty());
    count_zero_verify(&bare, &proof)
        .unwrap_or_else(|e| panic!("a count-zero verifier accepts the forged program: {e}"));
    assert!(verify_wlfm(&honest_build.artifacts, &proof, &proof.public_words).is_err());
    let refusal = verify_wlfm(&bare.artifacts, &proof, &proof.public_words)
        .expect_err("stack-less artifacts are refused");
    assert!(refusal.contains("no stack settles"), "{refusal}");
}

/// The coverage rule names each way a plan can leave a prefix unbound.
#[test]
fn the_coverage_rule_refuses_every_unbound_plan() {
    let counts = [2usize, 0, 3];
    let sizes = [2usize, 1];
    assert!(check_coverage(&counts, &sizes, &[(0, &[(0, 2)]), (1, &[(2, 3)])]).is_ok());
    for (stacks, why) in [
        (vec![(0usize, vec![(0usize, 2usize)])], "no stack settles"),
        (vec![(0, vec![(0, 1)]), (1, vec![(2, 3)])], "settles 1"),
        (
            vec![(0, vec![(0, 2), (1, 1)]), (1, vec![(2, 3)])],
            "no preprocessed",
        ),
        (vec![(0, vec![(0, 2), (2, 3)])], "is in group 1"),
        (
            vec![(0, vec![(0, 2)]), (1, vec![(2, 3)]), (1, vec![(2, 3)])],
            "twice",
        ),
    ] {
        let plan: Vec<(usize, &[(usize, usize)])> =
            stacks.iter().map(|(g, t)| (*g, t.as_slice())).collect();
        let e = check_coverage(&counts, &sizes, &plan).expect_err(why);
        assert!(e.contains(why), "{why}: {e}");
    }
}
