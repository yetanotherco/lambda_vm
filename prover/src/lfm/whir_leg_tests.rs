//! In-guest gates for the W-leg (`whir_leg`): the emitted leg EXECUTES an honest
//! W-LFM proof and refuses each mutated one, refuses a forged program's proof,
//! is refused at emit time for a plan that leaves a prefix unsettled, and costs
//! exactly its form (D-WHIR §6.1, items 3–5).
//!
//! The child is `TrivialV0` proved as a W-LFM proof (`whir_proof_tests` holds
//! its host gates). The leg program is executed under the block hasher, RPX:
//! its sponge is the algebraic one, and every hash it emits is an `LFM_HASH`
//! row. The child's own LFM hasher does not matter to the leg — the leg
//! verifies the child's multilinear proof, whose transcript and trees are RPX by
//! type whatever the child program runs.

use stark::proof::options::ProofOptions;

use crate::tables::types::FE;

use super::compiler::{LfmProgram, compile};
use super::executor::{LfmExecError, execute};
use super::instr::Instr;
use super::programs::{trivial_program, trivial_program_source};
use super::registry::REGISTRY_HASHER;
use super::whir_leg::{
    SIZING_PROFILES, WhirChild, WhirLegShape, declare_whir_leg_arenas, emit_whir_leg,
    shape_only_artifacts, table_proof_words, whir_leg_arena_words, whir_leg_cost,
};
use super::whir_proof::{
    PrepPolicy, WhirLfmBuild, WhirLfmProof, build_whir_artifacts, build_whir_artifacts_under,
    lfm_prove_whir, prove_traces_whir_opening, test_grind,
};
use super::word::LfmWord;

fn options() -> ProofOptions {
    ProofOptions::default_test_options()
}

fn arenas() -> Vec<Vec<LfmWord>> {
    vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ]
}

/// The child: `TrivialV0`, built and proved as a W-LFM proof.
fn child() -> (WhirLfmBuild, WhirLfmProof) {
    child_under(PrepPolicy::Both)
}

/// [`child`] under an explicit prefix policy.
fn child_under(policy: PrepPolicy) -> (WhirLfmBuild, WhirLfmProof) {
    let program = trivial_program();
    let build = build_whir_artifacts_under(&program, &options(), REGISTRY_HASHER, policy)
        .unwrap_or_else(|e| panic!("the child's artifacts build: {e:?}"));
    let proof = lfm_prove_whir(&program, &build, &arenas(), &options())
        .unwrap_or_else(|e| panic!("the child proves: {e:?}"));
    assert!(
        super::whir_proof::lfm_verify_whir(
            &build.artifacts,
            &proof.proof,
            &proof.public_words,
            &options()
        ),
        "the gates only read children the host verifier accepts"
    );
    (build, proof)
}

/// A program that is ONE W-leg and nothing else, publishing the leg's `(z, α)`
/// so it has an output and the challenges are observable.
fn leg_program(child: &WhirChild<'_>) -> LfmProgram {
    let mut b =
        super::builder::LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arenas = declare_whir_leg_arenas(&mut b, child);
    let leg = emit_whir_leg(&mut b, child, &arenas);
    b.public(leg.z_alpha.0.as_cell());
    b.public(leg.z_alpha.1.as_cell());
    let program = compile(b.finish());
    super::validator::validate(&program).expect("a W-leg program must be admissible");
    program
}

fn run(program: &LfmProgram, arena: &[Vec<LfmWord>]) -> Result<(), LfmExecError> {
    execute(program, arena, &crate::hash_pin::BLOCK_HASHER).map(|_| ())
}

fn run_or_locate(program: &LfmProgram, arena: &[Vec<LfmWord>]) {
    match execute(program, arena, &crate::hash_pin::BLOCK_HASHER) {
        Ok(_) => {}
        Err(LfmExecError::DivByZero { addr }) => panic!(
            "the W-leg REFUSED an honest child — a failing equality assert.\n{}",
            super::executor::locate_addr(program, addr)
        ),
        Err(why) => panic!("the W-leg must execute an honest child: {why:?}"),
    }
}

/// ★★ THE EXECUTION GATE: the emitted leg runs to completion on the honest
/// child's arena. Every challenge is derived in the machine from its own
/// transcript, so a statement absorbed wrong, a roots block with the prepared
/// roots in the wrong place, a layout that differs from the host's, or a leg at
/// the wrong point in the stream all give a `z` the child was never argued at —
/// and the argument stops satisfying its own refusals.
///
/// At the PRODUCTION config, grinding included: the one in-guest arm that
/// checks the chains' nonces.
#[test]
fn the_w_leg_executes_an_honest_child_at_the_production_config() {
    let (build, proof) = child();
    let opts = options();
    let child = WhirChild {
        artifacts: &build.artifacts,
        proof: &proof,
        options: &opts,
    };
    let program = leg_program(&child);
    let arena = whir_leg_arena_words(&child);
    run_or_locate(&program, &arena);
    println!(
        "W-LEG (production config): {} instrs, arenas {:?}",
        program.instrs.len(),
        arena.iter().map(Vec::len).collect::<Vec<_>>()
    );
}

/// The leg published its `(z, α)`: they are the host's, drawn after the
/// statement, the carried roots and then the prepared roots — so the leg's
/// roots block is the host's, value for value.
#[test]
fn the_w_leg_draws_the_hosts_challenges() {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    let _ungrinded = test_grind::off();
    let (build, proof) = child();
    let opts = options();
    let child = WhirChild {
        artifacts: &build.artifacts,
        proof: &proof,
        options: &opts,
    };
    let program = leg_program(&child);
    let exec = execute(
        &program,
        &whir_leg_arena_words(&child),
        &crate::hash_pin::BLOCK_HASHER,
    )
    .expect("the honest child executes");
    let a = &build.artifacts;
    let mut t = super::whir_proof::WhirLfmTranscript::new(&[]);
    super::whir_proof::absorb_whir_lfm_statement(
        &mut t,
        &a.program_id,
        &proof.public_words,
        &a.table_num_vars,
        &a.config,
    );
    stark::multilinear_table::absorb_roots::<crate::tables::types::GoldilocksExtension, _>(
        &mut t,
        &proof.proof.roots,
        &a.prepared_roots,
    );
    let z: crate::tables::types::FEE = t.sample_field_element();
    let alpha: crate::tables::types::FEE = t.sample_field_element();
    assert_eq!(exec.public_words.len(), 2);
    assert_eq!(exec.public_words[0].1, super::word::ext_word(&z), "z");
    assert_eq!(
        exec.public_words[1].1,
        super::word::ext_word(&alpha),
        "alpha"
    );
}

/// ★ Each mutation of the arena is refused: a published felt, a carried root, a
/// table's bus output, a sumcheck evaluation, a column value, the main chain's
/// final value, the prepared chain's final value. (Nonce words are left alone:
/// with grinding off no nonce is spent, and an unspent nonce is read by
/// nothing — see F-P2's `NonceLayout`.)
#[test]
fn every_mutated_arena_is_refused() {
    let _ungrinded = test_grind::off();
    let (build, proof) = child();
    let opts = options();
    let child = WhirChild {
        artifacts: &build.artifacts,
        proof: &proof,
        options: &opts,
    };
    let program = leg_program(&child);
    let honest = whir_leg_arena_words(&child);
    run_or_locate(&program, &honest);

    // Positions in the proof arena, from its own order.
    let roots = proof.proof.roots.len();
    let mut table_words = Vec::new();
    super::whir_epoch::push_table_words(&mut table_words, &proof.proof.tables[0]);
    let first_table = roots;
    let tables_end = roots
        + proof
            .proof
            .tables
            .iter()
            .map(|table| {
                let mut w = Vec::new();
                super::whir_epoch::push_table_words(&mut w, table);
                w.len()
            })
            .sum::<usize>();
    let group_final = tables_end;
    let prepared_final = {
        let group_shape = super::whir_chain::ChainShape::new(
            &build.artifacts.config,
            super::whir_proof::WhirLfmPlan::build(
                &build.artifacts,
                &super::whir_proof::airs_for(&build.artifacts, &opts).air_refs(),
            )
            .expect("plan")
            .group_layouts[0]
                .n_stack(),
        );
        group_final
            + proof.proof.columns[0].polys.len()
                * (1 + super::whir_chain::RoundStorage::words(&group_shape) as usize)
    };
    let cases: Vec<(&str, usize, usize)> = vec![
        ("a published felt", 0, 1),
        ("a carried root", 1, 0),
        ("a table's bus output", 1, first_table),
        ("a GKR or sumcheck word", 1, first_table + 3),
        ("a column value", 1, first_table + table_words.len() - 1),
        ("the main chain's final value", 1, group_final),
        ("the prepared chain's final value", 1, prepared_final),
    ];
    for (what, arena, at) in cases {
        let mut mutated = honest.clone();
        mutated[arena][at][0] += FE::one();
        assert!(
            run(&program, &mutated).is_err(),
            "{what} (arena {arena}, word {at}) mutated must be refused"
        );
    }
}

/// `TrivialV0` with `y = 9` replaced by `10` — see `whir_proof_tests`.
fn forged_trivial_program() -> LfmProgram {
    let mut source = trivial_program_source();
    let nine = [FE::from(9u64), FE::zero(), FE::zero(), FE::zero()];
    for instr in &mut source.instrs {
        if let Instr::Const { value, .. } = instr
            && *value == nine
        {
            *value = [FE::from(10u64), FE::zero(), FE::zero(), FE::zero()];
        }
    }
    compile(source)
}

/// ★ The in-guest twin of the host's forged-column refusal: the leg emitted for
/// the HONEST program refuses the forger's proof — the forged tables opened
/// against the honest program's prepared stack.
#[test]
fn the_w_leg_refuses_a_forged_program() {
    refuses_a_forged_program(PrepPolicy::Both);
}

/// ★ Under policy B the leg executes an honest child, and refuses the forged
/// program — the prepared opening being the prefix's only binding there.
#[test]
fn the_w_leg_executes_and_refuses_a_forgery_under_policy_b() {
    let _ungrinded = test_grind::off();
    let (build, honest) = child_under(PrepPolicy::PreparedOnly);
    let opts = options();
    let child = WhirChild {
        artifacts: &build.artifacts,
        proof: &honest,
        options: &opts,
    };
    run_or_locate(&leg_program(&child), &whir_leg_arena_words(&child));
    refuses_a_forged_program(PrepPolicy::PreparedOnly);
}

fn refuses_a_forged_program(policy: PrepPolicy) {
    let _ungrinded = test_grind::off();
    let (build, honest) = child_under(policy);
    let forged_program = forged_trivial_program();
    let exec = execute(&forged_program, &arenas(), &REGISTRY_HASHER).expect("executes");
    let mut traces =
        super::trace::build_traces_with_hasher(&forged_program, &exec.records, REGISTRY_HASHER);
    let forged = WhirLfmProof {
        proof: prove_traces_whir_opening(
            &build,
            &mut traces,
            &exec.public_words,
            &options(),
            false,
            true,
        )
        .expect("the forger's prover runs"),
        public_words: exec.public_words,
    };
    let opts = options();
    let honest_child = WhirChild {
        artifacts: &build.artifacts,
        proof: &honest,
        options: &opts,
    };
    let program = leg_program(&honest_child);
    let forged_child = WhirChild {
        artifacts: &build.artifacts,
        proof: &forged,
        options: &opts,
    };
    let arena = whir_leg_arena_words(&forged_child);
    assert_eq!(
        arena.iter().map(Vec::len).collect::<Vec<_>>(),
        whir_leg_arena_words(&honest_child)
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>(),
        "the forgery has the honest shape, so the refusal is about values"
    );
    assert!(
        run(&program, &arena).is_err(),
        "the leg must refuse a forged program's proof"
    );
}

/// ⛔ The emit-time refusal: a child whose prepared plan leaves one table's
/// prefix unsettled cannot have a leg emitted for it at all.
#[test]
#[should_panic(expected = "a W-leg cannot be emitted for this child")]
fn a_leg_for_an_unsettled_prefix_is_refused_at_emit_time() {
    let _ungrinded = test_grind::off();
    let (build, proof) = child();
    let mut artifacts = build.artifacts.clone();
    let last = artifacts
        .prepared_at
        .last()
        .expect("settles something")
        .table;
    artifacts.prepared_at.retain(|c| c.table != last);
    let opts = options();
    let child = WhirChild {
        artifacts: &artifacts,
        proof: &proof,
        options: &opts,
    };
    let _ = leg_program(&child);
}

/// ★★ F1 — the emitted leg costs EXACTLY its form, by kind: operations, the
/// constant pool (both directions), hints and permutations; and the form's
/// shape-derived table word count is each table's real arena length.
#[test]
fn the_w_legs_instruction_count_is_its_form() {
    f1_holds(PrepPolicy::Both);
}

/// The same F1 under policy B, where the main group opens each table's
/// columns past its prefix and the prepared opening is the prefix's only one.
#[test]
fn the_w_legs_instruction_count_is_its_form_under_policy_b() {
    f1_holds(PrepPolicy::PreparedOnly);
}

fn f1_holds(policy: PrepPolicy) {
    let _ungrinded = test_grind::off();
    let (build, proof) = child_under(policy);
    let opts = options();
    let child = WhirChild {
        artifacts: &build.artifacts,
        proof: &proof,
        options: &opts,
    };
    // A program that is the leg alone, publishing nothing.
    let mut b =
        super::builder::LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arenas = declare_whir_leg_arenas(&mut b, &child);
    let _ = emit_whir_leg(&mut b, &child, &arenas);
    let program = compile(b.finish());

    let count = |pred: fn(&Instr) -> bool| program.instrs.iter().filter(|i| pred(i)).count();
    let consts = count(|i| matches!(i, Instr::Const { .. }));
    let hints = count(|i| matches!(i, Instr::Hint { .. }));
    let publics = count(|i| matches!(i, Instr::Public { .. }));
    let perms = count(|i| matches!(i, Instr::Hash { .. }));
    let ops = program.instrs.len() - consts - hints - publics;

    let cost = whir_leg_cost(&WhirLegShape {
        artifacts: &build.artifacts,
        num_public_words: proof.public_words.len(),
        options: &opts,
    });
    println!(
        "W-LEG F1 [{policy:?}]: instrs {} = ops {ops} + consts {consts} + hints {hints}; perms {perms}; \
         form ops {} (spine {} tables {} closure {} groups {} prepared {}), consts {}, hints {}, \
         perms {}",
        program.instrs.len(),
        cost.operations(),
        cost.spine,
        cost.tables,
        cost.closure,
        cost.groups,
        cost.prepared,
        cost.constants.len(),
        cost.hints,
        cost.perms,
    );

    // The shape-derived table words against the real arena, table by table.
    let airs = super::whir_proof::airs_for(&build.artifacts, &opts);
    let refs = airs.air_refs();
    let plan = super::whir_proof::WhirLfmPlan::build(&build.artifacts, &refs).expect("plan");
    for ((table, layout), air) in proof.proof.tables.iter().zip(&plan.layouts).zip(&refs) {
        let slots = layout.slot_of().to_vec();
        let bus = stark::multilinear_logup::interaction_shapes(
            air.bus_interactions(),
            slots.len(),
            |c| {
                slots
                    .get(c)
                    .copied()
                    .ok_or(multilinear::Error::UnknownPolynomial {
                        index: c,
                        len: slots.len(),
                    })
            },
        )
        .expect("bus");
        let shape = super::whir_table::TableShape {
            ir: layout.shape(),
            bus: &bus,
            kinds: layout.kinds(),
            num_columns: layout.num_columns(),
            num_vars: layout.num_vars(),
        };
        let mut words = Vec::new();
        super::whir_epoch::push_table_words(&mut words, table);
        assert_eq!(
            table_proof_words(&shape),
            words.len(),
            "{}: the shape form counts {} words, the proof writes {}",
            air.name(),
            table_proof_words(&shape),
            words.len()
        );
    }

    let interned: Vec<LfmWord> = program
        .instrs
        .iter()
        .filter_map(|i| match i {
            Instr::Const { value, .. } => Some(*value),
            _ => None,
        })
        .collect();
    let unnamed: Vec<&LfmWord> = interned
        .iter()
        .filter(|w| !cost.constants.contains(w))
        .collect();
    let unemitted: Vec<&LfmWord> = cost
        .constants
        .iter()
        .filter(|w| !interned.contains(w))
        .collect();
    for w in unnamed.iter().take(20) {
        println!("  UNNAMED   {w:?}");
    }
    for w in unemitted.iter().take(20) {
        println!("  UNEMITTED {w:?}");
    }
    assert_eq!(publics, 0, "the leg publishes nothing of its own");
    assert_eq!(ops, cost.operations(), "operations");
    assert_eq!(hints, cost.hints, "hints");
    assert_eq!(
        hints,
        whir_leg_arena_words(&child)
            .iter()
            .map(Vec::len)
            .sum::<usize>(),
        "the leg hints every arena word exactly once"
    );
    assert_eq!(perms, cost.perms, "permutations");
    assert!(
        unnamed.is_empty() && unemitted.is_empty(),
        "the pool: {} interned and not named, {} named and not interned",
        unnamed.len(),
        unemitted.len()
    );
    assert_eq!(consts, cost.constants.len(), "constants");
}

/// The leg program itself — a verifier of a W-LFM proof — laid out as a W-LFM
/// program: its heights, cells and stacks, printed. The shape of the smallest
/// pure-WHIR recursion step, and the input to the box-scale round trip below.
#[test]
fn the_leg_program_builds_as_a_w_lfm_program() {
    let _ungrinded = test_grind::off();
    let (build, proof) = child();
    let opts = options();
    let child = WhirChild {
        artifacts: &build.artifacts,
        proof: &proof,
        options: &opts,
    };
    let program = leg_program(&child);
    let leg = build_whir_artifacts(&program, &opts, crate::hash_pin::BLOCK_HASHER)
        .unwrap_or_else(|e| panic!("the leg program builds as a W-LFM program: {e:?}"));
    let airs = super::whir_proof::airs_for(&leg.artifacts, &opts);
    let refs = airs.air_refs();
    let plan = super::whir_proof::WhirLfmPlan::build(&leg.artifacts, &refs).expect("plan");
    let cells: usize = plan.shapes.iter().map(|&(w, n)| w << n).sum();
    println!(
        "W-LEG AS W-LFM: heights {:?} · cells {cells} · main n{} x{} · prepared n{} x{} · Q {}",
        leg.artifacts.table_num_vars,
        plan.group_layouts[0].n_stack(),
        plan.group_layouts[0].num_polys(),
        leg.artifacts.prepared_layout.n_stack(),
        leg.artifacts.prepared_layout.num_polys(),
        leg.artifacts.config.num_queries,
    );
}

/// ★ A SMALL EMITTED NODE, round-tripped: the leg program — a machine verifier
/// of a W-LFM proof — proved AS a W-LFM proof and verified on the host. Pure
/// WHIR recursion, one step, on a program with a real RPX sponge in it.
///
/// Box-scale by the laptop rule (a ~2^25-cell stack): it runs in the W1 box
/// script, `thoughts/zf/box/I-WHIR-W1.sh`.
#[test]
#[ignore = "box-scale: a ~2^25-cell W-LFM prove; the W1 box script runs it"]
fn the_leg_program_round_trips_as_a_w_lfm_proof() {
    let (build, proof) = child();
    let opts = options();
    let child = WhirChild {
        artifacts: &build.artifacts,
        proof: &proof,
        options: &opts,
    };
    let program = leg_program(&child);
    let leg = build_whir_artifacts(&program, &opts, crate::hash_pin::BLOCK_HASHER)
        .unwrap_or_else(|e| panic!("the leg program builds as a W-LFM program: {e:?}"));
    let t = std::time::Instant::now();
    let proved = lfm_prove_whir(&program, &leg, &whir_leg_arena_words(&child), &opts)
        .unwrap_or_else(|e| panic!("the leg program proves as a W-LFM proof: {e:?}"));
    let prove_secs = t.elapsed().as_secs_f64();
    super::whir_proof::verify_whir_checked(
        &leg.artifacts,
        &proved.proof,
        &proved.public_words,
        &opts,
    )
    .unwrap_or_else(|e| panic!("the leg program's W-LFM proof verifies: {e:?}"));
    println!(
        "W-LEG NODE ROUND TRIP: prove {prove_secs:.2} s · roots {} · publics {}",
        proved.proof.roots.len(),
        proved.public_words.len()
    );
}

/// ★ THE PINS at D-WHIR §3.2's shapes (card-free, no proof): the leg's
/// permutations, operations and hints for a child of each measured shape, the
/// form the F1 above holds exact. Printed beside the design's instrument
/// numbers, which approximated the statement and the closure.
///
/// The child publishes a NODE schema (`SchemaLayout::node(0)`), the instrument's
/// assumption, so the two are compared like for like.
#[test]
fn the_w_leg_costs_at_the_sizing_shapes() {
    let opts = options();
    let publics = super::per_table_aggregator::SchemaLayout::node(0).total();
    let mut rows = Vec::new();
    for (label, logs) in SIZING_PROFILES {
        let artifacts = shape_only_artifacts(logs, &opts, PrepPolicy::Both);
        let cost = whir_leg_cost(&WhirLegShape {
            artifacts: &artifacts,
            num_public_words: publics,
            options: &opts,
        });
        let airs = super::whir_proof::airs_for(&artifacts, &opts);
        let plan =
            super::whir_proof::WhirLfmPlan::build(&artifacts, &airs.air_refs()).expect("plan");
        println!(
            "W-LEG SIZING {label}: perms {} · ops {} · hints {} · main n{} x{} · prepared n{} x{} \
             · Q {} · publics {publics}",
            cost.perms,
            cost.operations(),
            cost.hints,
            plan.group_layouts[0].n_stack(),
            plan.group_layouts[0].num_polys(),
            artifacts.prepared_layout.n_stack(),
            artifacts.prepared_layout.num_polys(),
            artifacts.config.num_queries,
        );
        rows.push((*label, cost.perms, cost.operations(), cost.hints));
    }
    // Policy B at the same shapes: the main stack holds the value columns only.
    let mut rows_b = Vec::new();
    for (label, logs) in SIZING_PROFILES {
        let artifacts = shape_only_artifacts(logs, &opts, PrepPolicy::PreparedOnly);
        let cost = whir_leg_cost(&WhirLegShape {
            artifacts: &artifacts,
            num_public_words: publics,
            options: &opts,
        });
        let airs = super::whir_proof::airs_for(&artifacts, &opts);
        let plan =
            super::whir_proof::WhirLfmPlan::build(&artifacts, &airs.air_refs()).expect("plan");
        println!(
            "W-LEG SIZING B {label}: perms {} · ops {} · hints {} · main n{} x{} · prepared n{} x{}",
            cost.perms,
            cost.operations(),
            cost.hints,
            plan.group_layouts[0].n_stack(),
            plan.group_layouts[0].num_polys(),
            artifacts.prepared_layout.n_stack(),
            artifacts.prepared_layout.num_polys(),
        );
        rows_b.push((*label, cost.perms, cost.operations(), cost.hints));
    }
    // ★ The policy-B pins, same format.
    //
    // Re-blessed under P2 (the chains grind before their queries only, with
    // one spent nonce a round): each chain round drops its folding and OOD
    // grinds and their two nonce words, e.g. wrap 0 −66 perms, −1,110 ops,
    // −24 hints over its 12 rounds. Before P2: wrap 0 38,440 / 384,825 /
    // 73,961 · global 62,576 / 591,284 / 117,796 · L1N0 62,621 / 592,012 /
    // 117,853 · L2N0 62,673 / 592,698 / 117,928 · node −1 41,191 / 405,242 /
    // 77,677.
    let pins_b: [(&str, usize, usize, usize); 5] = [
        ("wrap 0", 38_374, 383_715, 73_937),
        ("global wrap", 62_465, 589_417, 117_756),
        ("L1N0", 62_510, 590_145, 117_813),
        ("L2N0", 62_562, 590_831, 117_888),
        ("node -1", 41_119, 404_031, 77_651),
    ];
    for (label, perms, ops, hints) in pins_b {
        let row = rows_b
            .iter()
            .find(|(l, ..)| *l == label)
            .expect("the profile is in the table");
        assert_eq!(
            (row.1, row.2, row.3),
            (perms, ops, hints),
            "{label} (B): (perms, ops, hints) moved off the pin"
        );
    }
    // The design's policy-B instrument: wrap 0 38,441 permutations, L2N0
    // 62,673 — the same 0.5 % band as policy A's.
    for (label, want) in [("wrap 0", 38_441usize), ("L2N0", 62_673)] {
        let (_, perms, _, _) = rows_b
            .iter()
            .find(|(l, ..)| *l == label)
            .expect("the profile is in the table");
        let off = (*perms as f64 - want as f64).abs() / want as f64;
        assert!(
            off < 0.005,
            "{label} (B): the form's {perms} permutations are {:.2} % from the design's {want}",
            100.0 * off
        );
    }

    // ★ THE PINS, at the default format (stack 27, first6, cap auto, grind 20
    // before the queries only — P2): `(label, permutations, operations,
    // hints)`, the exact form's values. Re-blessed under P2 as the B pins
    // above; before it: wrap 0 39,795 / 398,742 / 75,432 · global 62,651 /
    // 594,462 / 117,796 · L1N0 62,696 / 595,307 / 117,853 · L2N0 82,155 /
    // 767,664 / 154,805 · node −1 41,266 / 408,727 / 77,677.
    let pins: [(&str, usize, usize, usize); 5] = [
        ("wrap 0", 39_723, 397_531, 75_406),
        ("global wrap", 62_540, 592_595, 117_756),
        ("L1N0", 62_585, 593_440, 117_813),
        ("L2N0", 82_005, 765_141, 154_751),
        ("node -1", 41_194, 407_516, 77_651),
    ];
    for (label, perms, ops, hints) in pins {
        let row = rows
            .iter()
            .find(|(l, ..)| *l == label)
            .expect("the profile is in the table");
        assert_eq!(
            (row.1, row.2, row.3),
            (perms, ops, hints),
            "{label}: (perms, ops, hints) moved off the pin"
        );
    }
    // The design's instrument (D-WHIR §3.2, policy A): wrap 0 39,796 perms,
    // the node shape 41,267, the global wrap 62,651, L1N0 62,696. The exact
    // form differs from it only in the statement's felts and the closure,
    // neither of which moves a permutation by more than the statement's own
    // sponge blocks — so the permutations must agree within 0.5 %.
    for (label, want) in [
        ("wrap 0", 39_796usize),
        ("node -1", 41_267),
        ("global wrap", 62_651),
        ("L1N0", 62_696),
    ] {
        let (_, perms, _, _) = rows
            .iter()
            .find(|(l, ..)| *l == label)
            .expect("the profile is in the table");
        let off = (*perms as f64 - want as f64).abs() / want as f64;
        assert!(
            off < 0.005,
            "{label}: the form's {perms} permutations are {:.2} % from the design's {want}",
            100.0 * off
        );
    }
}
