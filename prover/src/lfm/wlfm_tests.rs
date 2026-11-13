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
use multilinear::mle::Mle;
use multilinear::whir_hash::WhirHash;
use stark::multilinear_table::TableStatement;
use stark::multilinear_table::batched::VerifierChecks;

use crate::tables::types::{FE, FEE, GoldilocksExtension as E, GoldilocksField as F};

use super::builder::{Cell, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::hash::HasherKind;
use super::wlfm::{
    WLFM_B4_QUERIES, WLFM_B16_STACK, WLFM_B64_STACK, WlfmBuild, WlfmHash, WlfmLevers, WlfmPrep,
    WlfmProof, WlfmRate, absorb_wlfm_statement, build_wlfm, build_wlfm_on_host, build_wlfm_under,
    check_coverage, execute_and_fill_unmerged, prove_wlfm, prove_wlfm_traces, test_format,
    verify_wlfm, verify_wlfm_with, wlfm_config, wlfm_config_under,
};
use super::word::LfmWord;

pub(super) const HASHER: HasherKind = HasherKind::Poseidon1W16;

fn word(seed: u64) -> [FE; 4] {
    core::array::from_fn(|i| FE::from(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i as u64))
}

/// The fixture's arena: four arbitrary words.
pub(super) fn arenas() -> Vec<Vec<LfmWord>> {
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

pub(super) fn honest() -> LfmProgram {
    program(9)
}

fn forged() -> LfmProgram {
    program(10)
}

pub(super) fn build(program: &LfmProgram, rate: WlfmRate) -> WlfmBuild {
    build_wlfm(program, HASHER, rate).unwrap_or_else(|e| panic!("the W-LFM build: {e}"))
}

fn evaluated(program: &LfmProgram, rate: WlfmRate) -> WlfmBuild {
    build_wlfm_under(
        program,
        HASHER,
        rate,
        WlfmLevers::with_prep(WlfmPrep::Evaluated),
        false,
    )
    .unwrap_or_else(|e| panic!("the evaluated W-LFM build: {e}"))
}

fn proof_bytes(proof: &WlfmProof) -> usize {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("a W-LFM proof serializes")
        .len()
}

pub(super) fn prove(program: &LfmProgram, build: &WlfmBuild) -> WlfmProof {
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

/// The blowup-64 arm (the second compression step's rate) under the 2^24
/// stack, and its query-grind lever: `None` is the format's grind and today's
/// config at every rate; a set grind moves the query count by the production
/// rule. At a 2^23 stack (six rounds under first6 + 4s, charged log2 6):
/// grind 18 runs Q 39, 22 runs 37, 26 runs 36, each ≥ 128 + log2 6 bits at
/// the Johnson bound.
#[test]
fn the_blowup_64_arm_and_its_grind_follow_the_production_rule() {
    use multilinear::whir_chain::FinalPoly;
    let shapes = [(1usize, 23usize)];
    let at = |grind| wlfm_config_under(WlfmRate::SixtyFour, FinalPoly::Off, None, grind, &shapes);
    let today = at(None);
    assert_eq!(today.log_blowup, 6);
    assert_eq!(today.format.stack.get(), WLFM_B64_STACK);
    assert_eq!(
        today.grind.query,
        crate::zf_format::PRODUCTION_WHIR_GRIND_BITS
    );
    assert_eq!(
        today,
        at(Some(crate::zf_format::PRODUCTION_WHIR_GRIND_BITS))
    );
    let johnson = -(0.125f64 + 1.0 / 300.0).log2();
    for (grind, q) in [(18u8, 39usize), (22, 37), (26, 36)] {
        let c = at(Some(grind));
        assert_eq!(c.grind.query, grind);
        assert_eq!(c.num_queries, q, "grind {grind}");
        let bits = q as f64 * johnson + f64::from(grind);
        assert!(bits >= 128.0 + 6f64.log2(), "grind {grind}: {bits}");
        assert!((q - 1) as f64 * johnson + f64::from(grind) < 128.0 + 6f64.log2());
    }
    for rate in [
        WlfmRate::Two,
        WlfmRate::Four,
        WlfmRate::Sixteen,
        WlfmRate::SixtyFour,
    ] {
        assert_eq!(
            wlfm_config(rate, &shapes),
            wlfm_config_under(rate, FinalPoly::Off, None, None, &shapes)
        );
    }
}

/// The blowup-64 arm round-trips under the second step's levers (evaluated
/// columns, the byte-minimising schedule) with its grind set, and the grind
/// is part of the identity: a proof at one grind is refused by the other's
/// artifacts.
#[test]
fn the_blowup_64_arm_round_trips_and_its_grind_is_another_identity() {
    let _unground = test_format::set(None);
    let program = honest();
    let levers = |grind| WlfmLevers {
        grind,
        ..super::compress::folds_levers(WlfmPrep::Evaluated)
    };
    let g22 = under(&program, WlfmRate::SixtyFour, levers(Some(22)));
    let g26 = under(&program, WlfmRate::SixtyFour, levers(Some(26)));
    assert_eq!(g22.artifacts.config.log_blowup, 6);
    assert_ne!(g22.artifacts.program_id, g26.artifacts.program_id);
    let proof = prove(&program, &g22);
    verify_wlfm(&g22.artifacts, &proof, &proof.public_words)
        .unwrap_or_else(|e| panic!("the blowup-64 proof must verify: {e}"));
    assert!(verify_wlfm(&g26.artifacts, &proof, &proof.public_words).is_err());
    let mut restated = g22.artifacts.clone();
    restated.grind = None;
    assert!(
        verify_wlfm(&restated, &proof, &proof.public_words).is_err(),
        "artifacts restating the config without its grind"
    );
}

/// The verifier's host-committed build is the prover's: the same prepared
/// roots and identity, and the prover's proof verifies against it. On a card
/// build this compares the card's prepared commit with the host's.
#[test]
fn the_host_build_is_the_provers_build() {
    let _unground = test_format::set(None);
    let program = honest();
    for rate in [WlfmRate::Two, WlfmRate::Sixteen] {
        let prover = build(&program, rate);
        let host = build_wlfm_on_host(&program, HASHER, rate)
            .unwrap_or_else(|e| panic!("the host build: {e}"));
        assert_eq!(
            prover.artifacts.derived_roots(),
            host.artifacts.derived_roots()
        );
        assert_eq!(prover.artifacts.program_id, host.artifacts.program_id);
        let proof = prove(&program, &prover);
        verify_wlfm(&host.artifacts, &proof, &proof.public_words)
            .unwrap_or_else(|e| panic!("the proof verifies against the host build: {e}"));
    }
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

// =============================================================================
// The evaluated binding (the verifier evaluates the instruction columns)
// =============================================================================

/// ★ THE EVALUATED ANCHOR: at blowup 2 and 16 the evaluated build carries every
/// table's prefix at its AIR's count and no prepared stack; its proof carries
/// no prepared opening, is smaller, and verifies; it is another identity than
/// the stacked build, and each binding's proof is refused by the other's
/// artifacts, as is the evaluated proof with a stacked opening added.
#[test]
fn the_evaluated_binding_round_trips_with_no_prepared_chain_and_is_another_identity() {
    let _unground = test_format::set(None);
    let program = honest();
    for rate in [WlfmRate::Two, WlfmRate::Sixteen] {
        let stacked = build(&program, rate);
        let evaluated = evaluated(&program, rate);
        let a = &evaluated.artifacts;
        assert_eq!(a.prep, WlfmPrep::Evaluated);
        assert_eq!(stacked.artifacts.prep, WlfmPrep::Stacked);
        assert!(a.stacks.is_empty(), "no prepared stack");
        assert!(stacked.artifacts.evaluated.is_empty());
        assert_eq!(
            a.evaluated.iter().map(Vec::len).collect::<Vec<_>>(),
            a.counts,
            "every table's prefix at its AIR's count"
        );
        assert!(a.counts.iter().any(|&c| c > 0), "the fixture has prefixes");
        assert_ne!(a.program_id, stacked.artifacts.program_id);
        let proof = prove(&program, &evaluated);
        assert!(proof.prepared.is_empty(), "no prepared opening");
        verify_wlfm(a, &proof, &proof.public_words)
            .unwrap_or_else(|e| panic!("the evaluated proof must verify at {rate:?}: {e}"));
        let stacked_proof = prove(&program, &stacked);
        assert!(
            proof_bytes(&proof) < proof_bytes(&stacked_proof),
            "the evaluated proof is the smaller: {} vs {}",
            proof_bytes(&proof),
            proof_bytes(&stacked_proof)
        );
        assert!(
            verify_wlfm(&stacked.artifacts, &proof, &proof.public_words).is_err(),
            "an evaluated proof under stacked artifacts"
        );
        assert!(
            verify_wlfm(a, &stacked_proof, &stacked_proof.public_words).is_err(),
            "a stacked proof under evaluated artifacts"
        );
        let mut added = proof.clone();
        added.prepared.push(stacked_proof.prepared[0].clone());
        assert!(
            verify_wlfm(a, &added, &added.public_words).is_err(),
            "an evaluated proof with a prepared opening added"
        );
    }
}

/// An evaluated proof at blowup 16 refuses a false commit-time answer, a
/// missing one, and a tampered public word.
#[test]
fn an_evaluated_proof_refuses_a_tampered_answer_or_word() {
    let _unground = test_format::set(None);
    let program = honest();
    let build = evaluated(&program, WlfmRate::Sixteen);
    let proof = prove(&program, &build);
    let a = &build.artifacts;
    let mut altered = proof.clone();
    altered.commit_ood[0] += FieldElement::<E>::one();
    assert!(verify_wlfm(a, &altered, &altered.public_words).is_err());
    let mut missing = proof.clone();
    missing.commit_ood.pop();
    assert!(verify_wlfm(a, &missing, &missing.public_words).is_err());
    let mut claimed = proof.public_words.clone();
    claimed[0].1[0] += FE::one();
    assert!(verify_wlfm(a, &proof, &claimed).is_err());
}

/// ★ A forged instruction column — the twin's program proved against the
/// honest program's evaluated artifacts — is refused by the verifier's own
/// evaluation of the columns, and only by it: with the preprocessed check off
/// (a mutation), the same proof verifies. The honest prover refuses to make it.
#[test]
fn a_forged_instruction_column_is_refused_by_the_evaluated_columns() {
    let _unground = test_format::set(None);
    let build = evaluated(&honest(), WlfmRate::Two);
    assert!(
        prove_wlfm(&forged(), &build, &arenas()).is_err(),
        "the prover's prefix check refuses the forged program"
    );
    let proof = forge(&forged(), &build);
    assert!(proof.prepared.is_empty());
    assert!(
        verify_wlfm(&build.artifacts, &proof, &proof.public_words).is_err(),
        "the forged program's proof is refused"
    );
    verify_wlfm_with(
        &build.artifacts,
        &proof,
        &proof.public_words,
        false,
        VerifierChecks {
            preprocessed: false,
            ..VerifierChecks::ALL
        },
    )
    .unwrap_or_else(|e| panic!("with the column check off the forgery verifies: {e}"));
}

/// ⛔ THE EVALUATED RULE: artifacts that evaluate a prefix short, none of it,
/// over another height, with a changed value, or that mix the two bindings
/// are refused — the first four before or by the column check, the mixes
/// before any proof is read.
#[test]
fn the_evaluated_rule_refuses_a_short_resized_changed_or_mixed_binding() {
    let _unground = test_format::set(None);
    let program = honest();
    let build = evaluated(&program, WlfmRate::Two);
    let proof = prove(&program, &build);
    let a = &build.artifacts;
    let t = a
        .counts
        .iter()
        .position(|&c| c > 0)
        .expect("a table with a prefix");
    let refused = |artifacts: &super::wlfm::WlfmArtifacts, why: &str| {
        let e = verify_wlfm(artifacts, &proof, &proof.public_words).expect_err(why);
        assert!(e.contains(why), "{why}: {e}");
    };
    let mut short = a.clone();
    short.evaluated[t].pop();
    refused(&short, "and the verifier evaluates");
    let mut none = a.clone();
    none.evaluated.iter_mut().for_each(Vec::clear);
    refused(&none, "and the verifier evaluates");
    let mut resized = a.clone();
    let rows = resized.evaluated[t][0].evals().len();
    resized.evaluated[t][0] = Mle::new(vec![FE::zero(); 2 * rows]).expect("a column");
    refused(&resized, "rows and a preprocessed column");
    let mut changed = a.clone();
    let mut values = changed.evaluated[t][0].evals().to_vec();
    values[0] += FE::one();
    changed.evaluated[t][0] = Mle::new(values).expect("a column");
    assert!(
        verify_wlfm(&changed, &proof, &proof.public_words).is_err(),
        "a changed instruction value"
    );
    let stacked = build_wlfm(&program, HASHER, WlfmRate::Two).expect("the stacked build");
    let mut mixed = a.clone();
    mixed.stacks = stacked.artifacts.stacks.clone();
    refused(&mixed, "evaluated artifacts carry prepared stacks");
    let mut mixed = stacked.artifacts.clone();
    mixed.evaluated = a.evaluated.clone();
    refused(&mixed, "stacked artifacts carry evaluated columns");
}

// =============================================================================
// The final polynomial (the chains stop committing once few variables remain)
// =============================================================================

pub(super) fn under(program: &LfmProgram, rate: WlfmRate, levers: WlfmLevers) -> WlfmBuild {
    build_wlfm_under(program, HASHER, rate, levers, false)
        .unwrap_or_else(|e| panic!("the W-LFM build under {levers:?}: {e}"))
}

fn final_levers(prep: WlfmPrep) -> WlfmLevers {
    WlfmLevers {
        prep,
        final_poly: multilinear::whir_chain::FinalPoly::Vars(6),
        fold: None,
        grind: None,
    }
}

/// Every chain round that sends a final polynomial, as `(group, chain,
/// round)`: no successor root, one successor block with no path, and rounds
/// after it.
fn final_rounds(proof: &WlfmProof) -> Vec<(usize, usize, usize)> {
    use multilinear::whir_chain::RoundOpenings;
    let mut out = Vec::new();
    for (g, opening) in proof.proof.columns.iter().enumerate() {
        for (p, chain) in opening.polys.iter().enumerate() {
            for (r, round) in chain.rounds.iter().enumerate() {
                if let RoundOpenings::Extension(o) = &round.openings
                    && round.next_root.is_none()
                    && r + 1 < chain.rounds.len()
                    && o.next.len() == 1
                    && o.next[0].proof.merkle_path.is_empty()
                {
                    out.push((g, p, r));
                }
            }
        }
    }
    out
}

/// ★ THE FINAL-POLYNOMIAL ANCHOR: at blowup 2 and 16, under both bindings,
/// the proof verifies, at least one chain stops at a final polynomial, the
/// proof is smaller than the same binding without it, the identity moves, and
/// each format's proof is refused by the other's artifacts.
#[test]
fn the_final_polynomial_round_trips_smaller_and_is_another_identity() {
    let _unground = test_format::set(None);
    let program = honest();
    for rate in [WlfmRate::Two, WlfmRate::Sixteen] {
        for prep in [WlfmPrep::Stacked, WlfmPrep::Evaluated] {
            let off = under(&program, rate, WlfmLevers::with_prep(prep));
            let on = under(&program, rate, final_levers(prep));
            assert_eq!(on.artifacts.final_poly, final_levers(prep).final_poly);
            assert_ne!(on.artifacts.program_id, off.artifacts.program_id);
            let proof = prove(&program, &on);
            verify_wlfm(&on.artifacts, &proof, &proof.public_words)
                .unwrap_or_else(|e| panic!("{rate:?} {prep:?}: {e}"));
            assert!(
                !final_rounds(&proof).is_empty(),
                "{rate:?} {prep:?}: a chain stops at a final polynomial"
            );
            let today = prove(&program, &off);
            assert!(
                proof_bytes(&proof) < proof_bytes(&today),
                "{rate:?} {prep:?}: {} vs {}",
                proof_bytes(&proof),
                proof_bytes(&today)
            );
            assert!(final_rounds(&today).is_empty(), "today's chains send none");
            assert!(verify_wlfm(&off.artifacts, &proof, &proof.public_words).is_err());
            assert!(verify_wlfm(&on.artifacts, &today, &today.public_words).is_err());
        }
    }
}

/// A final-polynomial W-LFM proof refuses a changed polynomial value, a
/// changed final value, a tampered commit-time answer and a tampered word.
#[test]
fn a_final_polynomial_proof_refuses_a_tampered_polynomial() {
    use multilinear::whir_chain::RoundOpenings;
    let _unground = test_format::set(None);
    let program = honest();
    let build = under(&program, WlfmRate::Sixteen, final_levers(WlfmPrep::Stacked));
    let proof = prove(&program, &build);
    let a = &build.artifacts;
    verify_wlfm(a, &proof, &proof.public_words).expect("honest");
    let (g, p, r) = final_rounds(&proof)[0];
    let mut poly = proof.clone();
    match &mut poly.proof.columns[g].polys[p].rounds[r].openings {
        RoundOpenings::Extension(o) => o.next[0].values[0] += FieldElement::<E>::one(),
        _ => panic!("the final round is an extension round"),
    }
    assert!(verify_wlfm(a, &poly, &poly.public_words).is_err(), "poly");
    let mut value = proof.clone();
    value.proof.columns[g].polys[p].final_value += FieldElement::<E>::one();
    assert!(
        verify_wlfm(a, &value, &value.public_words).is_err(),
        "final value"
    );
    let mut answer = proof.clone();
    answer.commit_ood[0] += FieldElement::<E>::one();
    assert!(
        verify_wlfm(a, &answer, &answer.public_words).is_err(),
        "answer"
    );
    let mut claimed = proof.public_words.clone();
    claimed[0].1[0] += FE::one();
    assert!(verify_wlfm(a, &proof, &claimed).is_err(), "word");
}

/// The final polynomial is the compression step's alone: the base block's
/// chain, the default ZF format and a W-LFM proof at the default levers (the
/// formats a parent verifies in-guest) all keep `Off`.
#[test]
fn only_an_opted_in_wlfm_proof_stops_at_a_final_polynomial() {
    use multilinear::whir_chain::FinalPoly;
    let shapes = [(329usize, 18usize), (23, 21), (25, 20)];
    assert_eq!(
        crate::block_whir::BlockFormat::pinned()
            .chain_config(&shapes)
            .format
            .final_poly,
        FinalPoly::Off
    );
    assert_eq!(
        crate::zf_format::ZfFormat::DEFAULT
            .chain_format()
            .final_poly,
        FinalPoly::Off
    );
    for rate in [WlfmRate::Two, WlfmRate::Four, WlfmRate::Sixteen] {
        assert_eq!(wlfm_config(rate, &shapes).format.final_poly, FinalPoly::Off);
    }
    assert_eq!(WlfmLevers::DEFAULT.final_poly, FinalPoly::Off);
}

// =============================================================================
// The compact wire encoding
// =============================================================================

/// ★ Every W-LFM proof the levers make — blowup 2 and 16, both bindings, with
/// and without the final polynomial — decodes from its compact bytes to the
/// same proof (the same canonical bytes), which verifies,
/// and the compact bytes are fewer than rkyv's.
#[test]
fn the_compact_encoding_round_trips_every_wlfm_proof_and_is_smaller() {
    use multilinear::whir_chain::FinalPoly;
    let _unground = test_format::set(None);
    let program = honest();
    for rate in [WlfmRate::Two, WlfmRate::Sixteen] {
        for prep in [WlfmPrep::Stacked, WlfmPrep::Evaluated] {
            for final_poly in [FinalPoly::Off, FinalPoly::Vars(6)] {
                let build = under(
                    &program,
                    rate,
                    WlfmLevers {
                        prep,
                        final_poly,
                        fold: None,
                        grind: None,
                    },
                );
                let proof = prove(&program, &build);
                let bytes = super::wire::encode(&proof);
                let back = super::wire::decode(&bytes)
                    .unwrap_or_else(|e| panic!("{rate:?} {prep:?} {final_poly:?}: {e}"));
                // The same values: the compact bytes are canonical (rkyv's are
                // not: a field element may sit unreduced in memory).
                assert_eq!(super::wire::encode(&back), bytes);
                verify_wlfm(&build.artifacts, &back, &back.public_words)
                    .unwrap_or_else(|e| panic!("the decoded proof verifies: {e}"));
                println!(
                    "WIRE {rate:?} {prep:?} {final_poly:?}: compact {} B vs rkyv {} B",
                    bytes.len(),
                    proof_bytes(&proof)
                );
                assert!(
                    bytes.len() < proof_bytes(&proof),
                    "{} compact vs {} rkyv",
                    bytes.len(),
                    proof_bytes(&proof)
                );
            }
        }
    }
}

/// The decoder takes exactly one proof's bytes: a trailing byte, any
/// truncation, and a changed value that decodes are refused or verify false.
#[test]
fn the_compact_decoder_refuses_trailing_truncated_and_tampered_bytes() {
    let _unground = test_format::set(None);
    let program = honest();
    let build = build(&program, WlfmRate::Sixteen);
    let proof = prove(&program, &build);
    let bytes = super::wire::encode(&proof);
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(super::wire::decode(&trailing).is_err());
    for cut in [0, 1, bytes.len() / 3, bytes.len() / 2, bytes.len() - 1] {
        assert!(super::wire::decode(&bytes[..cut]).is_err(), "cut at {cut}");
    }
    // One byte flipped in the last field element (a public word's lane):
    // either refused by the decoder or refused by the verifier.
    let mut flipped = bytes.clone();
    let at = bytes.len() - 3;
    flipped[at] ^= 1;
    match super::wire::decode(&flipped) {
        Err(_) => {}
        Ok(p) => assert!(verify_wlfm(&build.artifacts, &p, &p.public_words).is_err()),
    }
}

// =============================================================================
// The byte-minimising schedule (fold 5 after round 0, a final polynomial of 10)
// =============================================================================

/// ★ The fixture on the byte-minimising schedule, at blowup 2 and 16 under
/// both bindings: verifies, folds 5 after round 0, sends a final polynomial,
/// is smaller than today's chains, is another identity, and each format's
/// proof is refused by the other's artifacts.
#[test]
fn the_byte_minimising_schedule_round_trips_smaller_and_is_another_identity() {
    let _unground = test_format::set(None);
    let program = honest();
    for rate in [WlfmRate::Two, WlfmRate::Sixteen] {
        for prep in [WlfmPrep::Stacked, WlfmPrep::Evaluated] {
            let off = under(&program, rate, WlfmLevers::with_prep(prep));
            let on = under(&program, rate, super::compress::folds_levers(prep));
            assert_eq!(on.artifacts.config.log_folding, 5);
            assert_eq!(
                on.artifacts.config.num_queries,
                off.artifacts.config.num_queries
            );
            assert_ne!(on.artifacts.program_id, off.artifacts.program_id);
            assert_ne!(
                on.artifacts.config.fold_word(),
                off.artifacts.config.fold_word()
            );
            let proof = prove(&program, &on);
            verify_wlfm(&on.artifacts, &proof, &proof.public_words)
                .unwrap_or_else(|e| panic!("{rate:?} {prep:?}: {e}"));
            let today = prove(&program, &off);
            println!(
                "FOLDS {rate:?} {prep:?}: {} B vs today {} B, {} final rounds",
                proof_bytes(&proof),
                proof_bytes(&today),
                final_rounds(&proof).len()
            );
            assert!(proof_bytes(&proof) < proof_bytes(&today));
            assert!(verify_wlfm(&off.artifacts, &proof, &proof.public_words).is_err());
            assert!(verify_wlfm(&on.artifacts, &today, &today.public_words).is_err());
            // The verifier re-derives the config from the levers: artifacts
            // that claim the schedule without its fold are refused.
            let mut restated = on.artifacts.clone();
            restated.fold = None;
            assert!(verify_wlfm(&restated, &proof, &proof.public_words).is_err());
        }
    }
}

/// ★ The dedup encoding (repeated Merkle siblings sent once) of every W-LFM
/// proof the levers make — two rates, both bindings, today's chains, the
/// final polynomial and the byte-minimising schedule — decodes to the same
/// canonical bytes, verifies, and is no larger than the compact encoding.
#[test]
fn the_dedup_encoding_round_trips_every_wlfm_proof() {
    use multilinear::whir_chain::FinalPoly;
    let _unground = test_format::set(None);
    let program = honest();
    for rate in [WlfmRate::Two, WlfmRate::Sixteen] {
        for prep in [WlfmPrep::Stacked, WlfmPrep::Evaluated] {
            for levers in [
                WlfmLevers::with_prep(prep),
                WlfmLevers {
                    prep,
                    final_poly: FinalPoly::Vars(6),
                    fold: None,
                    grind: None,
                },
                super::compress::folds_levers(prep),
            ] {
                let build = under(&program, rate, levers);
                let proof = prove(&program, &build);
                let compact = super::wire::encode(&proof);
                let dedup = super::wire::encode_dedup(&proof);
                let back = super::wire::decode_dedup(&dedup)
                    .unwrap_or_else(|e| panic!("{rate:?} {levers:?}: {e}"));
                assert_eq!(super::wire::encode(&back), compact);
                assert_eq!(super::wire::encode_dedup(&back), dedup);
                verify_wlfm(&build.artifacts, &back, &back.public_words)
                    .unwrap_or_else(|e| panic!("the decoded proof verifies: {e}"));
                println!(
                    "DEDUP {rate:?} {levers:?}: dedup {} B · compact {} B · rkyv {} B",
                    dedup.len(),
                    compact.len(),
                    proof_bytes(&proof)
                );
                assert!(dedup.len() <= compact.len());
                // The two decoders do not take each other's bytes.
                assert!(super::wire::decode(&dedup).is_err() || dedup == compact);
            }
        }
    }
}

// =============================================================================
// Probe: the verifier's host prepared commit, by step
// =============================================================================

/// ⛔ A MEASUREMENT (ignored; laptop or box): the host prepared commit a C2
/// verifier pays for C1 — `commit_groups_on_host` at blowup 16 over C1's
/// prepared columns at C1's table heights (median and large, from the box's
/// `C1 ROWS` lines), synthetic values (the cost does not read them) — whole,
/// then split on one stacked polynomial of the same height into the lift
/// (Möbius + bit-reversed gather), the encode (host NTT + bit-reverse) and the
/// tree (leaf gather + Poseidon1 + nodes). Prints `PREP COMMIT` lines.
#[test]
#[ignore = "a measurement of the verifier's host prepared commit"]
fn probe_the_host_prepared_commit_by_step() {
    use crate::block_whir::{PreparedColumns, commit_groups_on_host, group_stacks};
    use multilinear::whir::{Domain, encode, lift_coefficients};
    use multilinear::whir_chain::LeafOrder;
    use multilinear::whir_commit::CodewordCommitment;
    use std::time::Instant;
    let program = honest();
    let fixture = build(&program, WlfmRate::Sixteen);
    let counts = fixture.artifacts.counts.clone();
    let config = fixture.artifacts.config;
    assert_eq!(counts.len(), 10, "the WHIR recursion chip set's ten tables");
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        FE::from(seed >> 1)
    };
    println!(
        "PREP COMMIT: rayon threads {} · counts {counts:?} · blowup {} · first fold {:?} · leaves {:?}",
        rayon::current_num_threads(),
        1usize << config.log_blowup,
        config.schedule(24).first(),
        config.format.leaves
    );
    for (name, heights) in [
        ("median", [11u32, 17, 19, 18, 11, 16, 17, 19, 2, 16]),
        ("large", [11u32, 17, 20, 18, 11, 16, 17, 19, 2, 16]),
    ] {
        let prepared: Vec<PreparedColumns> = counts
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 0)
            .map(|(t, &c)| {
                let rows = 1usize << heights[t];
                (
                    t,
                    (0..c)
                        .map(|_| (0..rows).map(|_| next()).collect())
                        .collect(),
                )
            })
            .collect();
        let cells: usize = prepared
            .iter()
            .map(|(_, cols)| cols.iter().map(Vec::len).sum::<usize>())
            .sum();
        let groups = vec![(0..10u32).collect::<Vec<u32>>()];
        let t = Instant::now();
        let stacks = group_stacks(prepared, &groups).expect("the stacks");
        let stacked = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let committed = commit_groups_on_host::<WlfmHash>(stacks, &config).expect("the commit");
        let whole = t.elapsed().as_secs_f64();
        let layout = committed[0].commitment.layout().clone();
        let (n, polys) = (layout.n_stack(), layout.num_polys());
        drop(committed);
        println!(
            "PREP COMMIT {name}: prepared cells {cells} (2^{:.3}) · stacks {polys}x2^{n} · group_stacks {stacked:.3}s · commit_groups_on_host {whole:.3}s",
            (cells as f64).log2()
        );
        // The split, on one stacked polynomial of the same height.
        let values: Vec<FieldElement<F>> = (0..1usize << n).map(|_| next()).collect();
        let mle = Mle::new(values).expect("an MLE");
        let t = Instant::now();
        let lifted = lift_coefficients(&mle);
        let lift = t.elapsed().as_secs_f64();
        drop(mle);
        let domain = Domain::<F>::new(n + config.log_blowup).expect("the domain");
        let t = Instant::now();
        let codeword = encode::<F, F>(&lifted, &domain).expect("the encode");
        let enc = t.elapsed().as_secs_f64();
        drop(lifted);
        let first = config.schedule(n).first().copied().unwrap_or(0);
        let t = Instant::now();
        let tree = CodewordCommitment::<F, WlfmHash>::from_codeword_on_host_in(
            codeword,
            first,
            LeafOrder::Coset,
        )
        .expect("the tree");
        let hashed = t.elapsed().as_secs_f64();
        drop(tree);
        println!(
            "PREP COMMIT {name} SPLIT (one 2^{n} polynomial, {polys} in the stack): lift {lift:.3}s · encode {enc:.3}s · tree {hashed:.3}s · sum {:.3}s",
            lift + enc + hashed
        );
    }
}
