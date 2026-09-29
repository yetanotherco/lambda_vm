//! Host gates for the W-LFM proof (`whir_proof`): round trips, and one named
//! refusal per way a W-LFM proof can fail to bind its program (D-WHIR §6.1).
//!
//! ⚠ The fixture is `TrivialV0` and a FORGED twin of it: the same program with
//! one constant changed. The twin has every table at the same height and
//! executes to completion (the constant is the `y` of `q = (x + y)·x / (x + y)`,
//! so the program's own assert holds for any `y`), which makes it a valid proof
//! of a DIFFERENT program — exactly what the preprocessed columns must refuse,
//! and nothing else about it can.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::element::FieldElement;
use stark::multilinear_table;
use stark::proof::options::ProofOptions;

use crate::tables::types::{FE, GoldilocksExtension as E};

use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::hash::HasherKind;
use super::instr::Instr;
use super::programs::{trivial_program, trivial_program_source};
use super::registry::{LfmProgramKind, REGISTRY_HASHER};
use super::trace::build_traces_with_hasher;
use super::whir_proof::test_grind;
use super::whir_proof::{
    PrepPolicy, WhirLfmBuild, WhirLfmError, WhirLfmPlan, WhirLfmTranscript,
    absorb_whir_lfm_statement, airs_for, build_whir_artifacts, build_whir_artifacts_under,
    check_chip_set, lfm_prove_whir, lfm_verify_whir, prove_traces_whir_opening,
    verify_whir_checked,
};
use super::word::LfmWord;

fn options() -> ProofOptions {
    ProofOptions::default_test_options()
}

/// `TrivialV0`'s arena: four arbitrary words, which every hasher but BLAKE3
/// compresses.
fn arenas() -> Vec<Vec<LfmWord>> {
    vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ]
}

/// `TrivialV0` with its `y = 9` constant replaced by `10`: the same shape, a
/// different CONST instruction group, and an execution that still completes.
fn forged_trivial_program() -> LfmProgram {
    let mut source = trivial_program_source();
    let nine = [FE::from(9u64), FE::zero(), FE::zero(), FE::zero()];
    let mut changed = 0usize;
    for instr in &mut source.instrs {
        if let Instr::Const { value, .. } = instr
            && *value == nine
        {
            *value = [FE::from(10u64), FE::zero(), FE::zero(), FE::zero()];
            changed += 1;
        }
    }
    assert_eq!(changed, 1, "TrivialV0 interns its `y = 9` exactly once");
    compile(source)
}

fn build(program: &LfmProgram, hasher: HasherKind) -> WhirLfmBuild {
    build_whir_artifacts(program, &options(), hasher)
        .unwrap_or_else(|e| panic!("the W-LFM artifacts build: {e:?}"))
}

/// [`build`] under policy A, where a proof WITHOUT its prepared opening is still
/// well-formed (the prefix is in the main stack too) — the only policy under
/// which the §2.4 trap can be shown. Under policy B, the default, the prover
/// refuses to leave a prefix out that no opening settles.
fn build_a(program: &LfmProgram, hasher: HasherKind) -> WhirLfmBuild {
    build_whir_artifacts_under(program, &options(), hasher, PrepPolicy::Both)
        .unwrap_or_else(|e| panic!("the W-LFM artifacts build under policy A: {e:?}"))
}

/// Proves `program`'s execution against `build`, WITHOUT the prover's prefix
/// check — the forger's prover.
fn forge(
    program: &LfmProgram,
    build: &WhirLfmBuild,
    open_prepared: bool,
) -> (
    multilinear_table::MultiProof<crate::tables::types::GoldilocksField, E>,
    Vec<(u32, LfmWord)>,
) {
    let hasher = build.artifacts.hasher;
    let exec = execute(program, &arenas(), &hasher).expect("the program executes");
    let mut traces = build_traces_with_hasher(program, &exec.records, hasher);
    let proof = prove_traces_whir_opening(
        build,
        &mut traces,
        &exec.public_words,
        &options(),
        false,
        open_prepared,
    )
    .unwrap_or_else(|e| panic!("the forger's prover runs: {e:?}"));
    (proof, exec.public_words)
}

// =============================================================================
// Round trips
// =============================================================================

/// ★ THE ANCHOR: an honest W-LFM proof of `TrivialV0` under RPX — the recursion
/// programs' hasher — at the PRODUCTION config, grinding included, verifies.
/// Every other test here runs its chains ungrinded (`test_grind`); this one is
/// what says the production config itself round-trips.
#[test]
fn the_trivial_program_round_trips_at_the_production_config() {
    let program = trivial_program();
    let build = build(&program, HasherKind::Rpx);
    // The production grind is P2's: 20 bits before each round's queries only.
    assert_eq!(
        build.artifacts.config.grind,
        multilinear::whir_chain::GrindBits::query_only(20),
        "the anchor must run the production grind"
    );
    let proved = lfm_prove_whir(&program, &build, &arenas(), &options())
        .unwrap_or_else(|e| panic!("the W-LFM prove runs: {e:?}"));
    assert!(
        proved.proof.preprocessed.is_some(),
        "a W-LFM proof carries its prepared opening"
    );
    verify_whir_checked(
        &build.artifacts,
        &proved.proof,
        &proved.public_words,
        &options(),
    )
    .unwrap_or_else(|e| panic!("the honest W-LFM proof must verify: {e:?}"));
}

/// ★ Policy B round-trips: the prefix out of the main stack, settled by the
/// prepared opening alone — and a policy-B proof read under policy A is
/// refused, the main stack being another stack there.
#[test]
fn policy_b_round_trips_and_is_not_read_as_policy_a() {
    let _ungrinded = test_grind::off();
    let program = trivial_program();
    let b = build_whir_artifacts_under(
        &program,
        &options(),
        REGISTRY_HASHER,
        PrepPolicy::PreparedOnly,
    )
    .expect("builds under policy B");
    assert_eq!(
        build(&program, REGISTRY_HASHER).artifacts.policy,
        PrepPolicy::PreparedOnly,
        "the default is policy B"
    );
    let a = build_a(&program, REGISTRY_HASHER);
    assert_eq!(
        a.artifacts.prepared_roots, b.artifacts.prepared_roots,
        "the prepared stack does not depend on the policy"
    );
    assert_ne!(
        a.artifacts.program_id, b.artifacts.program_id,
        "the policy is part of the identity"
    );
    let proved = lfm_prove_whir(&program, &b, &arenas(), &options()).expect("proves under B");
    verify_whir_checked(
        &b.artifacts,
        &proved.proof,
        &proved.public_words,
        &options(),
    )
    .unwrap_or_else(|e| panic!("the honest policy-B proof must verify: {e:?}"));
    let mut as_a = b.artifacts.clone();
    as_a.policy = PrepPolicy::Both;
    assert!(
        !lfm_verify_whir(&as_a, &proved.proof, &proved.public_words, &options()),
        "a policy-B proof must not verify as policy A"
    );
    // And a policy-A proof must not verify as policy B.
    let proved_a = lfm_prove_whir(&program, &a, &arenas(), &options()).expect("proves under A");
    let mut as_b = a.artifacts.clone();
    as_b.policy = PrepPolicy::PreparedOnly;
    assert!(
        !lfm_verify_whir(&as_b, &proved_a.proof, &proved_a.public_words, &options()),
        "a policy-A proof must not verify as policy B"
    );
}

/// Under policy B the prepared opening is the prefix's ONLY binding: the
/// forged program's proof is refused.
#[test]
fn policy_b_refuses_a_forged_instruction_column() {
    let _ungrinded = test_grind::off();
    let honest = build_whir_artifacts_under(
        &trivial_program(),
        &options(),
        REGISTRY_HASHER,
        PrepPolicy::PreparedOnly,
    )
    .expect("builds under policy B");
    let (proof, words) = forge(&forged_trivial_program(), &honest, true);
    assert!(
        !lfm_verify_whir(&honest.artifacts, &proof, &words, &options()),
        "under policy B the forged prefix must not open the program's stack"
    );
}

/// The same under the registry's hasher, ungrinded.
#[test]
fn the_trivial_program_round_trips_under_the_registry_hasher() {
    let _ungrinded = test_grind::off();
    let program = trivial_program();
    let build = build(&program, REGISTRY_HASHER);
    let proved = lfm_prove_whir(&program, &build, &arenas(), &options()).expect("proves");
    verify_whir_checked(
        &build.artifacts,
        &proved.proof,
        &proved.public_words,
        &options(),
    )
    .unwrap_or_else(|e| panic!("the honest W-LFM proof must verify: {e:?}"));
}

/// The registry programs with an arena fixture here round-trip under W-LFM:
/// `TrivialV0` and the FRI verifier `FriToyV0`, both the WHIR recursion chip
/// set. The keccak programs are REFUSED at the build by the chip-set rule,
/// never proved with columns nothing binds; the two replay programs are
/// recorded either way (their arenas are other suites' fixtures).
#[test]
fn the_registry_programs_round_trip_or_are_refused_by_chip_set() {
    let _ungrinded = test_grind::off();
    let fri = super::fixture::fixture_prove();
    let proved_kinds = [
        (LfmProgramKind::TrivialV0, arenas()),
        (
            LfmProgramKind::FriToyV0,
            vec![fri.commitments.clone(), fri.openings.clone()],
        ),
    ];
    for (kind, arena) in proved_kinds {
        let program = kind.program();
        let build = build(&program, REGISTRY_HASHER);
        let proof = lfm_prove_whir(&program, &build, &arena, &options())
            .unwrap_or_else(|e| panic!("{kind:?}: the W-LFM prove runs: {e:?}"));
        verify_whir_checked(
            &build.artifacts,
            &proof.proof,
            &proof.public_words,
            &options(),
        )
        .unwrap_or_else(|e| panic!("{kind:?}: the honest W-LFM proof must verify: {e:?}"));
    }
    for kind in [
        LfmProgramKind::KeccakChainV0,
        LfmProgramKind::KeccakSpongeV0,
    ] {
        match build_whir_artifacts(&kind.program(), &options(), REGISTRY_HASHER) {
            Err(WhirLfmError::Shape(why)) => assert!(
                why.contains("chip set"),
                "{kind:?} was refused for {why:?}, which is not the chip-set refusal"
            ),
            other => panic!(
                "{kind:?} instantiates the keccak family and must be refused, got {:?}",
                other.map(|b| b.artifacts.program_id)
            ),
        }
    }
    for kind in [
        LfmProgramKind::TranscriptReplayV0,
        LfmProgramKind::StatementReplayV0,
    ] {
        let outcome = match build_whir_artifacts(&kind.program(), &options(), REGISTRY_HASHER) {
            Ok(_) => "builds".to_string(),
            Err(WhirLfmError::Shape(why)) => {
                assert!(why.contains("chip set"), "{kind:?}: {why}");
                "refused by chip set".to_string()
            }
            Err(e) => panic!("{kind:?}: the build failed for a reason other than shape: {e:?}"),
        };
        println!("W-LFM REGISTRY: {kind:?} {outcome}");
    }
}

// =============================================================================
// The preprocessed columns: what binds the program (D-WHIR §2.4)
// =============================================================================

/// ⛔ THE TRAP, SHOWN ON THE LFM'S OWN TABLES: the forged program, proved with no
/// prepared opening, VERIFIES against the honest program's identity when every
/// statement counts zero preprocessed columns — which is what
/// `statement_with_preprocessed(&air.precomputed_columns())` gives an LFM AIR,
/// whose count lives only in `with_preprocessed`.
///
/// This is the control for the refusals below: the forgery is valid in every
/// respect but the program, so they refuse it for the program's sake.
#[test]
fn a_count_zero_statement_accepts_a_forged_lfm_program() {
    let _ungrinded = test_grind::off();
    let honest = build_a(&trivial_program(), REGISTRY_HASHER);
    let (proof, words) = forge(&forged_trivial_program(), &honest, false);

    let airs = airs_for(&honest.artifacts, &options());
    let refs = airs.air_refs();
    let plan = WhirLfmPlan::build(&honest.artifacts, &refs).expect("the plan builds");
    // The statements an LFM AIR's column LIST gives: no copies, count zero.
    let count_zero: Vec<_> = refs
        .iter()
        .zip(&plan.layouts)
        .map(|(air, layout)| {
            assert!(
                air.precomputed_columns().is_empty(),
                "an LFM AIR carries no column builder — the premise of the trap"
            );
            layout.statement()
        })
        .collect();
    let mut transcript = WhirLfmTranscript::new(&[]);
    absorb_whir_lfm_statement(
        &mut transcript,
        &honest.artifacts.program_id,
        &words,
        &honest.artifacts.table_num_vars,
        &plan.config,
    );
    let mut probe = transcript.clone();
    multilinear_table::absorb_roots::<E, _>(&mut probe, &proof.roots, &[]);
    let z: FieldElement<E> = probe.sample_field_element();
    let alpha: FieldElement<E> = probe.sample_field_element();
    let expected =
        super::proof::expected_public_balance(&words, &z, &alpha).expect("no fingerprint collides");
    multilinear_table::multi_verify::<_, _, _, super::whir_proof::WhirLfmHash>(
        &proof,
        &count_zero,
        &plan.group_layouts,
        &plan.group_domains,
        &plan.sizes(),
        &expected,
        &plan.config,
        &mut transcript,
        None,
    )
    .expect("the forged program verifies under count-zero statements — the trap D-WHIR §2.4 names");
}

/// ★ The trap closed, first direction: the same forged proof, carrying no
/// prepared opening, is REFUSED by the W-LFM verifier — the AIR's count with
/// nothing settling it.
#[test]
fn a_forged_program_without_a_prepared_opening_is_refused() {
    let _ungrinded = test_grind::off();
    let honest = build_a(&trivial_program(), REGISTRY_HASHER);
    let (proof, words) = forge(&forged_trivial_program(), &honest, false);
    assert!(
        !lfm_verify_whir(&honest.artifacts, &proof, &words, &options()),
        "a proof whose preprocessed columns nothing settles must be refused"
    );
}

/// ★ A forged instruction column is refused by the prepared opening: the forger
/// opens the HONEST program's stack at its forged tables' points, and the values
/// its tables settled on are not what the pinned stack takes there.
#[test]
fn a_forged_instruction_column_is_refused_by_the_prepared_opening() {
    let _ungrinded = test_grind::off();
    let honest = build(&trivial_program(), REGISTRY_HASHER);
    let (proof, words) = forge(&forged_trivial_program(), &honest, true);
    assert!(
        proof.preprocessed.is_some(),
        "the forger carried an opening, so the refusal is the opening's"
    );
    assert!(
        !lfm_verify_whir(&honest.artifacts, &proof, &words, &options()),
        "the forged CONST column must not open the program's prepared stack"
    );
    // And the honest prover refuses to produce it at all.
    let forged = forged_trivial_program();
    let exec = execute(&forged, &arenas(), &REGISTRY_HASHER).expect("executes");
    let mut traces = build_traces_with_hasher(&forged, &exec.records, REGISTRY_HASHER);
    assert!(
        matches!(
            super::whir_proof::prove_traces_whir(
                &honest,
                &mut traces,
                &exec.public_words,
                &options(),
                true
            ),
            Err(WhirLfmError::Shape(_))
        ),
        "the honest prover must refuse a trace whose prefix is not the program's"
    );
}

/// ★ The OMISSION mutation: the honest proof with its prepared opening deleted
/// is refused — not only a reordered or corrupted opening, an absent one.
#[test]
fn a_deleted_prepared_opening_is_refused() {
    let _ungrinded = test_grind::off();
    let program = trivial_program();
    let build = build(&program, REGISTRY_HASHER);
    let mut proved = lfm_prove_whir(&program, &build, &arenas(), &options()).expect("proves");
    assert!(lfm_verify_whir(
        &build.artifacts,
        &proved.proof,
        &proved.public_words,
        &options()
    ));
    proved.proof.preprocessed = None;
    assert!(
        !lfm_verify_whir(
            &build.artifacts,
            &proved.proof,
            &proved.public_words,
            &options()
        ),
        "a W-LFM proof without its prepared opening must be refused"
    );
}

/// A plan that leaves one table's prefix unsettled is refused at the plan,
/// before any argument runs — the host twin of the in-guest emit-time refusal.
#[test]
fn a_plan_that_misses_a_table_is_refused() {
    let _ungrinded = test_grind::off();
    let program = trivial_program();
    let build = build(&program, REGISTRY_HASHER);
    let proved = lfm_prove_whir(&program, &build, &arenas(), &options()).expect("proves");
    let mut artifacts = build.artifacts.clone();
    let last = artifacts
        .prepared_at
        .last()
        .expect("the stack settles something")
        .table;
    artifacts.prepared_at.retain(|c| c.table != last);
    match verify_whir_checked(&artifacts, &proved.proof, &proved.public_words, &options()) {
        Err(WhirLfmError::Shape(why)) => assert!(
            why.contains("does not settle every table"),
            "refused for {why:?}"
        ),
        other => panic!("a plan missing table {last} must be refused at the plan, got {other:?}"),
    }
    // An empty plan — "an LFM table with preprocessed columns and no plan".
    let mut artifacts = build.artifacts.clone();
    artifacts.prepared_at.clear();
    assert!(
        !lfm_verify_whir(&artifacts, &proved.proof, &proved.public_words, &options()),
        "an empty prepared plan must be refused"
    );
}

// =============================================================================
// The statement
// =============================================================================

/// A restated table height is refused. The height is program shape, absorbed by
/// the statement and read by the layout; the config is re-derived at the
/// restated shape so the refusal is not merely the artifacts' config check.
#[test]
fn a_restated_table_height_is_refused() {
    let _ungrinded = test_grind::off();
    let program = trivial_program();
    let build = build(&program, REGISTRY_HASHER);
    let proved = lfm_prove_whir(&program, &build, &arenas(), &options()).expect("proves");
    for table in [0usize, 5] {
        let mut artifacts = build.artifacts.clone();
        artifacts.table_num_vars[table] += 1;
        let airs = airs_for(&artifacts, &options());
        let shapes = super::whir_proof::table_shapes(&airs.air_refs(), &artifacts.table_num_vars);
        artifacts.config = super::whir_proof::whir_lfm_config(&shapes);
        assert!(
            !lfm_verify_whir(&artifacts, &proved.proof, &proved.public_words, &options()),
            "table {table}'s height restated must be refused"
        );
    }
}

/// A tampered public word is refused: the statement absorbs it and the bus
/// closes against it.
#[test]
fn a_tampered_public_word_is_refused() {
    let _ungrinded = test_grind::off();
    let program = trivial_program();
    let build = build(&program, REGISTRY_HASHER);
    let proved = lfm_prove_whir(&program, &build, &arenas(), &options()).expect("proves");
    let mut words = proved.public_words.clone();
    words[2].1[0] += FE::one();
    assert!(
        !lfm_verify_whir(&build.artifacts, &proved.proof, &words, &options()),
        "a tampered public word must be refused"
    );
    // The indices are bound by position: a claim at another index is refused
    // before any argument.
    let mut words = proved.public_words.clone();
    words.swap(0, 1);
    assert!(
        !lfm_verify_whir(&build.artifacts, &proved.proof, &words, &options()),
        "public words out of position must be refused"
    );
}

/// ★ KAT on the absorb sequence: the prepared roots are absorbed AFTER the
/// carried roots and BEFORE `z` — the one placement no round trip can see,
/// because prover and verifier share the roots block.
///
/// Checked against an INDEPENDENTLY built transcript, through the one quantity
/// the proof fixes at its own challenges: the tables' bus outputs sum to the
/// claimed words' `LfmPublic` balance at the `(z, α)` the statement, the carried
/// roots and then the prepared roots produce — and NOT at the pair drawn before
/// the prepared roots.
#[test]
fn the_prepared_roots_are_absorbed_before_the_first_challenge() {
    let _ungrinded = test_grind::off();
    let program = trivial_program();
    let build = build(&program, REGISTRY_HASHER);
    let proved = lfm_prove_whir(&program, &build, &arenas(), &options()).expect("proves");
    let a = &build.artifacts;

    let balance_at = |absorb_prepared: bool| -> (FieldElement<E>, FieldElement<E>) {
        let mut t = WhirLfmTranscript::new(&[]);
        absorb_whir_lfm_statement(
            &mut t,
            &a.program_id,
            &proved.public_words,
            &a.table_num_vars,
            &a.config,
        );
        for root in &proved.proof.roots {
            t.append_bytes(root);
        }
        if absorb_prepared {
            for root in &a.prepared_roots {
                t.append_bytes(root);
            }
        }
        let z: FieldElement<E> = t.sample_field_element();
        let alpha: FieldElement<E> = t.sample_field_element();
        let owed = super::proof::expected_public_balance(&proved.public_words, &z, &alpha)
            .expect("no fingerprint collides");
        let sum = proved
            .proof
            .tables
            .iter()
            .map(|table| multilinear_table::contribution(&table.bus_output).expect("q != 0"))
            .fold(FieldElement::<E>::zero(), |acc, share| acc + share);
        (sum, owed)
    };
    let (sum, owed) = balance_at(true);
    assert_eq!(
        sum, owed,
        "the bus closes at the challenges drawn after the prepared roots"
    );
    let (sum, owed) = balance_at(false);
    assert_ne!(
        sum, owed,
        "drawing before the prepared roots closes the bus too — this KAT cannot see the placement"
    );
}

/// The identity moves with the prepared stack: the forged program has another
/// `program_id_w`, so a parent interning the honest one never accepts the
/// forger's statement.
#[test]
fn the_identity_folds_the_prepared_roots() {
    let _ungrinded = test_grind::off();
    let honest = build(&trivial_program(), REGISTRY_HASHER);
    let forged = build(&forged_trivial_program(), REGISTRY_HASHER);
    assert_eq!(
        honest.artifacts.table_num_vars, forged.artifacts.table_num_vars,
        "the twin has the honest shape"
    );
    assert_ne!(
        honest.artifacts.prepared_roots,
        forged.artifacts.prepared_roots
    );
    assert_ne!(honest.artifacts.program_id, forged.artifacts.program_id);
    // And the same program built twice is the same identity.
    let again = build(&trivial_program(), REGISTRY_HASHER);
    assert_eq!(honest.artifacts.program_id, again.artifacts.program_id);
    assert_eq!(
        honest.artifacts.prepared_roots,
        again.artifacts.prepared_roots
    );
}

/// The chip-set refusal names the mask it refused.
#[test]
fn a_family_chip_set_is_refused() {
    use super::airs::ChipSet;
    check_chip_set(ChipSet {
        keccak: false,
        blake3: false,
        bitwise: false,
    })
    .expect("the WHIR recursion chip set");
    for chip_set in [
        ChipSet {
            keccak: true,
            blake3: false,
            bitwise: false,
        },
        ChipSet {
            keccak: false,
            blake3: true,
            bitwise: false,
        },
        ChipSet {
            keccak: false,
            blake3: false,
            bitwise: true,
        },
    ] {
        assert!(
            check_chip_set(chip_set).is_err(),
            "{chip_set:?} must be refused"
        );
    }
}
