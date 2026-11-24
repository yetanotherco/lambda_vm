//! Level 1 over a 7-round BLAKE3 base proved under the WHIR round form: the
//! base's chains open no successor, and level 1's in-guest verifier checks
//! each round's fold values as claims on the successor instead.
//!
//! Over a small block, on the host: the closed level-1 program executes over
//! the round-form base proof's witness and its words claim the block; one
//! group's program publishes the host's bus sum; a base proof with one value
//! changed (a later round's opened value, a round's out-of-domain answer, a
//! round-0 opened value) is refused in execution; and a plan under one form
//! takes no proof of the other. Each run prints the level's instructions and
//! lean compressions against today's form over the same block.

use super::executor::execute;
use super::levels::level_plan;
use super::whir_block::{
    WhirBlockPlan, block_leaf_arena, first_groups_arena, first_groups_bus_sum, top_claims,
};
use super::wlfm::WlfmFamily;
use crate::block_whir::{B3Rounds, BlockFormat, BlockWhirProof};
use crate::hash_pin::{BlockFamily, ProofFamily};
use crate::tables::types::{FE, GoldilocksExtension};
use math::field::element::FieldElement;
use multilinear::whir_chain::{
    ArgueFormat, FinalPoly, GkrLadder, GroupFold, RoundOpenings, WhirFolds,
};
use stark::proof::options::{BaseFormat, ProofOptions};

type E = GoldilocksExtension;

/// The small blocks' BLAKE3 format with the production's group opening
/// (indexed runs, one tree a run, coset leaves), batched argue and lean query
/// check, under `rounds` and `folds`.
fn format(rounds: B3Rounds, folds: WhirFolds) -> BlockFormat {
    let mut f = crate::tests::block_whir_p1_tests::many_groups(BaseFormat::B3_WHIR);
    f.group = GroupFold::Indexed;
    f.argue = ArgueFormat::BATCHED;
    f.leaf_query = super::whir_chain::LeafQuery::Lean;
    f.zf.whir_folds = folds;
    f.b3_rounds = rounds;
    f
}

/// The small block `name` proved under `format` inside the levels' admission,
/// and accepted by the host verifier.
fn base(name: &str, format: &BlockFormat) -> (Vec<u8>, BlockWhirProof) {
    base_with(name, format, crate::tests::block_whir_p1_tests::options(3))
}

/// [`base`] under the block options `options`.
fn base_with(
    name: &str,
    format: &BlockFormat,
    options: crate::block_whir::BlockOptions,
) -> (Vec<u8>, BlockWhirProof) {
    let elf = crate::test_utils::asm_elf_bytes(name);
    let opts = ProofOptions::default_test_options();
    let proof = crate::base_only::with_blake3_levels(|| {
        crate::block_whir::prove_block_whir_with(
            &elf,
            &[],
            &opts,
            format,
            &options,
            &crate::block_whir::Deviations::default(),
        )
    })
    .expect("the BLAKE3 base proves")
    .0;
    assert!(
        crate::base_only::with_permit(|| crate::block_whir::verify_block_whir(
            &proof, &elf, &opts, format
        )
        .expect("verify runs")),
        "the host verifier accepts the BLAKE3 base"
    );
    (elf, proof)
}

fn plan(elf: &[u8], proof: &BlockWhirProof, format: &BlockFormat) -> WhirBlockPlan {
    level_plan(
        elf,
        &ProofOptions::default_test_options(),
        format,
        proof.statement(),
        WlfmFamily::B3,
        None,
    )
    .expect("the BLAKE3 plan derives")
}

fn hasher(program: &super::compiler::LfmProgram) -> super::hash::HasherKind {
    program.hasher(<BlockFamily as ProofFamily>::SOCKET)
}

/// What one form's level 1 costs over a block.
struct Census {
    groups: usize,
    instructions: usize,
    compressions: usize,
    first_instructions: usize,
    first_compressions: usize,
}

/// Level 1's closed program over `proof`'s block executed, its words claiming
/// the block, and group 0's own program publishing the host's bus sum.
fn level_1_executes(plan: &WhirBlockPlan, proof: &BlockWhirProof) -> Census {
    let program = plan.closed_block_program().expect("level 1 emits");
    let arenas = block_leaf_arena(plan, proof, 0).expect("the arena");
    let exec = execute(&program, &arenas, &hasher(&program)).expect("level 1 accepts the base");
    assert!(
        top_claims(plan, &exec.public_words),
        "level 1's words claim the block"
    );
    let (first, _) = plan.first_groups_program_marked(1).expect("emits");
    let arenas = first_groups_arena(plan, proof, 1).expect("the arena");
    let exec = execute(&first, &arenas, &hasher(&first)).expect("group 0's program accepts");
    let sum = first_groups_bus_sum(plan, proof, 1).expect("the host's sum");
    assert_eq!(
        exec.public_words.last().expect("a sum").1,
        sum,
        "the bus sum"
    );
    Census {
        groups: plan.num_groups(),
        instructions: program.instrs.len(),
        compressions: program.groups.b3l.real_rows,
        first_instructions: first.instrs.len(),
        first_compressions: first.groups.b3l.real_rows,
    }
}

/// Group 0's first chain with a round after round 0 that opens its tree:
/// (chain, round).
fn later_round<L>(proof: &BlockWhirProof<L>) -> (usize, usize) {
    proof.proof.columns[0]
        .polys
        .iter()
        .enumerate()
        .find_map(|(c, chain)| {
            chain
                .rounds
                .iter()
                .enumerate()
                .skip(1)
                .find(|(_, round)| match &round.openings {
                    RoundOpenings::Extension(p) => !p.current.is_empty(),
                    _ => false,
                })
                .map(|(r, _)| (c, r))
        })
        .expect("a later round of group 0 opens its tree")
}

/// A one-value edit of a base proof (of either kind, `L`).
type Tamper<L = ()> = Box<dyn Fn(&mut BlockWhirProof<L>)>;

/// Three one-value tampers in group 0's chain `c`: round `r`'s first opened
/// value, round 0's out-of-domain answer and round 0's first opened value.
fn tampers<L: 'static>(c: usize, r: usize) -> Vec<(&'static str, Tamper<L>)> {
    vec![
        (
            "a later round's opened value",
            Box::new(move |p: &mut BlockWhirProof<L>| {
                match &mut p.proof.columns[0].polys[c].rounds[r].openings {
                    RoundOpenings::Extension(o) => {
                        o.current[0].values[0] += FieldElement::<E>::one()
                    }
                    other => panic!("an extension round: {other:?}"),
                }
            }),
        ),
        (
            "round 0's out-of-domain answer",
            Box::new(move |p: &mut BlockWhirProof<L>| {
                let round = &mut p.proof.columns[0].polys[c].rounds[0];
                let value = round.ood_value.as_mut().expect("round 0 has a successor");
                *value += FieldElement::<E>::one();
            }),
        ),
        (
            "a round-0 opened value",
            Box::new(move |p: &mut BlockWhirProof<L>| {
                match &mut p.proof.columns[0].polys[c].rounds[0].openings {
                    RoundOpenings::Base(o) => o.current[0].values[0] += FE::one(),
                    RoundOpenings::BaseGrouped(o) => o.current[0][0].values[0] += FE::one(),
                    RoundOpenings::BaseJoint(o) => o.current[0].values[0] += FE::one(),
                    RoundOpenings::Extension(o) => {
                        o.current[0].values[0] += FieldElement::<E>::one()
                    }
                }
            }),
        ),
    ]
}

/// ★ Level 1 over a small BLAKE3 base under the round form (shrink 3, first
/// fold 6 and 4): it executes, its words claim the block, and group 0's
/// program publishes the host's bus sum; three one-value tampers of the base
/// proof are refused in execution, the arena building as before. Printed:
/// instructions and lean compressions, the whole level and group 0's, against
/// today's form over the same block and first fold.
#[test]
fn level_1_executes_over_a_round_form_blake3_base_and_refuses_its_tampers() {
    let name = "test_commit_4";
    for folds in [
        crate::zf_format::ZfFormat::DEFAULT.whir_folds,
        WhirFolds::Uniform,
    ] {
        let fri = format(B3Rounds::Fri, folds);
        let w3 = format(B3Rounds::WHIR3, folds);
        let (elf, today) = base(name, &fri);
        let (_, honest) = base(name, &w3);
        let today_plan = plan(&elf, &today, &fri);
        let w3_plan = plan(&elf, &honest, &w3);
        let c_fri = level_1_executes(&today_plan, &today);
        let c_w3 = level_1_executes(&w3_plan, &honest);
        assert_eq!(c_fri.groups, c_w3.groups, "the same groups");
        println!(
            "LEVEL 1 ROUND FORM {name} {folds:?}: {} groups · instructions {} (today {}) · lean \
             compressions {} (today {}) · group 0: instructions {} (today {}) · compressions {} \
             (today {})",
            c_w3.groups,
            c_w3.instructions,
            c_fri.instructions,
            c_w3.compressions,
            c_fri.compressions,
            c_w3.first_instructions,
            c_fri.first_instructions,
            c_w3.first_compressions,
            c_fri.first_compressions,
        );
        assert!(
            c_w3.compressions < c_fri.compressions,
            "the round form opens no successor"
        );

        // A plan under one form takes no proof of the other: the successor
        // blocks are where the round form has none, or missing where today's
        // needs them.
        assert!(
            block_leaf_arena(&w3_plan, &today, 0).is_err(),
            "a round-form plan over today's proof"
        );
        assert!(
            block_leaf_arena(&today_plan, &honest, 0).is_err(),
            "today's plan over a round-form proof"
        );

        let program = w3_plan.closed_block_program().expect("emits");
        let hasher = hasher(&program);
        let (c, r) = later_round(&honest);
        for (what, edit) in tampers(c, r) {
            let mut proof = honest.clone();
            edit(&mut proof);
            let arenas = block_leaf_arena(&w3_plan, &proof, 0)
                .unwrap_or_else(|e| panic!("{what}: the arena builds: {e}"));
            let why = execute(&program, &arenas, &hasher);
            println!(
                "LEVEL 1 ROUND FORM TAMPER {folds:?} {what}: {:?}",
                why.as_ref().err()
            );
            assert!(why.is_err(), "{what}: level 1 accepted it");
        }
    }
}

/// ★ Level 1 over a small BLAKE3 base whose GKR ladders step two tree layers
/// (the round form, today's first fold): the base proves and verifies; level
/// 1's closed program — the real leaf, its batched walk hinted by its own
/// plan's shapes and its transcript BLAKE3, drawing `c₂` — executes over the
/// base's witness and claims the block, and group 0's program publishes the
/// host's bus sum; four one-value tampers of the two-layer ladder (a value of
/// each of a step's two halves, a degree-5 round's value, step 0's first
/// value) are refused in execution. The ordinary lib run's twin of the
/// box-tier words test below.
#[test]
fn level_1_executes_over_a_two_layer_blake3_base_and_refuses_ladder_tampers() {
    let name = "test_commit_4";
    let mut double = format(
        B3Rounds::WHIR3,
        crate::zf_format::ZfFormat::DEFAULT.whir_folds,
    );
    double.argue = double.argue.with_ladder(GkrLadder::Double);
    let (elf, proof) = base(name, &double);
    let plan = plan(&elf, &proof, &double);
    let census = level_1_executes(&plan, &proof);
    println!(
        "LEVEL 1 TWO-LAYER {name}: {} groups · instructions {} · lean compressions {}",
        census.groups, census.instructions, census.compressions
    );
    let program = plan.closed_block_program().expect("emits");
    let hasher = hasher(&program);
    let tampers = ladder_tampers(&proof);
    for (what, edit) in [0usize, 7, 8, 9].map(|k| &tampers[k]) {
        let mut forged = proof.clone();
        edit(&mut forged);
        let arenas = block_leaf_arena(&plan, &forged, 0)
            .unwrap_or_else(|e| panic!("{what}: the arena builds: {e}"));
        let why = execute(&program, &arenas, &hasher);
        println!("LEVEL 1 TWO-LAYER TAMPER {what}: {:?}", why.as_ref().err());
        assert!(why.is_err(), "{what}: level 1 accepted it");
    }
}

/// A streamed BLAKE3 base (`window_log2`, the production options' and the
/// levels driver's) proved, and its host verifier's refusal if any.
fn streamed(elf: &[u8], format: &BlockFormat) -> Result<Option<String>, String> {
    let opts = ProofOptions::default_test_options();
    let mut options = crate::tests::block_whir_p1_tests::options(3);
    options.window_log2 = Some(4);
    let proof = crate::base_only::with_blake3_levels(|| {
        crate::block_whir::prove_block_whir_with(
            elf,
            &[],
            &opts,
            format,
            &options,
            &crate::block_whir::Deviations::default(),
        )
    })
    .map_err(|e| format!("{e}"))?
    .0;
    let why = crate::base_only::with_permit(|| {
        crate::block_whir::verify_block_whir_traced(
            &proof,
            elf,
            &opts,
            format,
            false,
            stark::multilinear_table::batched::VerifierChecks::ALL,
        )
    });
    Ok(match why {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(format!("{e:?}")),
        Err(e) => Some(format!("{e:?}")),
    })
}

/// ★ A streamed BLAKE3 base under the round form proves and verifies, as
/// today's does. Phase A commits before the block's shapes are known, and
/// every commit-time out-of-domain point absorbs the fold word, which binds
/// the round form: its union bound must read no shape. The small format's
/// stack cap (10) is under the block's tallest stack, the case a shape-read
/// bound got wrong.
#[test]
fn a_streamed_round_form_base_proves_and_verifies() {
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    for rounds in [B3Rounds::Fri, B3Rounds::WHIR3] {
        let format = format(rounds, WhirFolds::Uniform);
        assert_eq!(
            streamed(&elf, &format),
            Ok(None),
            "{rounds:?}: the streamed base proves and verifies"
        );
    }
}

/// ★ Box only: level 1 over a REAL block's BLAKE3 base, executed (not
/// proved), under the process's format (`LAMBDA_VM_B3_BASE_ROUNDS`,
/// `LAMBDA_VM_ZF_WHIR_FOLDS` and the rest, as the levels harness installs
/// them). The base proved and verified on the host; level 1's closed program
/// emitted and executed over the base proof's witness, its words claiming the
/// block; group 0's own program publishing the host's bus sum; three
/// one-value tampers in group 0 refused by group 0's program. Prints `RF
/// CONFIG`, `RF BASE`, `RF L1 WHOLE`, `RF L1 GROUP0`, one `RF L1 TAMPER` line
/// a tamper and `RF DONE`.
#[test]
#[ignore = "a real block: box only"]
fn rf_level_1_over_a_real_block() {
    use crate::zf_format::{LeafOrder, ZfFormat};
    use std::time::Instant;
    let _permit = super::device_permit::serial_for_test();
    let elf = std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF"))
        .expect("read the ELF");
    let input = std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT"))
        .expect("read the input");
    // `prove-block-base --base b3`'s format, installed before anything reads
    // it (the levels harness's own).
    let zf = ZfFormat {
        whir_blowup: 2,
        whir_leaves: LeafOrder::Coset,
        whir_trees: multilinear::whir_chain::Round0Trees::One,
        base: BaseFormat::B3_WHIR,
        ..ZfFormat::from_env().expect("the environment's format")
    };
    zf.check_leaves().expect("leaves");
    zf.check_trees().expect("trees");
    ZfFormat::install(zf).expect("install the format");
    let mut cfg =
        super::whir_block_tree::WhirTreeConfig::from_env().unwrap_or_else(|e| panic!("{e}"));
    cfg.format.group = GroupFold::Indexed;
    let format = &cfg.format;
    let opts = super::proof::block_base_options();
    println!(
        "RF CONFIG: tag {} · rounds {:?} · folds {:?} · group polys {} · card drop {:?}",
        String::from_utf8_lossy(&crate::block_whir::block_format_tag(format)),
        format.b3_rounds,
        format.zf.whir_folds,
        format.group_polys,
        cfg.options.drop_levels,
    );

    let roots: std::sync::Mutex<Option<Vec<crate::block_whir::PreparedRoots>>> =
        std::sync::Mutex::new(None);
    let observe = |_: crate::block_whir::BlockStatement<'_>,
                   prepared: &[crate::block_whir::PreparedRoots]| {
        *roots.lock().unwrap_or_else(|e| e.into_inner()) = Some(prepared.to_vec());
    };
    let on_group = |_: stark::multilinear_block::GroupOpened<'_, _, _>| {};
    let t = Instant::now();
    // The BLAKE3 base argues KECCAK_RND's byte lookups per group unless
    // `LAMBDA_VM_B3_BASE_LOCAL=off`: either kind of proof, held as one type.
    let proof: crate::block_whir::AnyBlockWhirProof = crate::base_only::with_blake3_levels(|| {
        if format.local_lookups() {
            crate::block_whir::prove_block_whir_local_observed_led(
                &elf,
                &input,
                &opts,
                format,
                &cfg.options,
                &observe,
                &on_group,
                None,
            )
            .map(|(mut proof, _)| {
                let local = std::mem::take(&mut proof.local);
                proof.with_local(Some(local))
            })
        } else {
            crate::block_whir::prove_block_whir_observed_groups(
                &elf,
                &input,
                &opts,
                format,
                &cfg.options,
                &observe,
                &on_group,
            )
            .map(|(proof, _)| proof.with_local(None))
        }
    })
    .unwrap_or_else(|e| panic!("the base proves: {e}"));
    let prove = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let verified = crate::base_only::with_permit(|| {
        crate::block_whir::verify_block_whir(&proof, &elf, &opts, format)
    })
    .expect("verify runs");
    let verify = t.elapsed().as_secs_f64();
    let roots = roots
        .into_inner()
        .unwrap_or_else(|e| e.into_inner())
        .expect("the base stated");
    let t = Instant::now();
    let plan = level_plan(
        &elf,
        &opts,
        format,
        proof.statement(),
        WlfmFamily::B3,
        Some(&roots),
    )
    .unwrap_or_else(|e| panic!("the plan derives: {e}"));
    let planned = t.elapsed().as_secs_f64();
    let groups = plan.num_groups();
    println!(
        "RF BASE: prove {prove:.3}s · verified {verified} · verify {verify:.3}s · groups {groups} \
         · plan {planned:.3}s"
    );
    assert!(verified, "the base verifies");

    // The whole level, executed.
    let t = Instant::now();
    let program = plan.closed_block_program().expect("level 1 emits");
    let emit = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let arenas = block_leaf_arena(&plan, &proof, 0).expect("the arena");
    let arena = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let exec = execute(&program, &arenas, &hasher(&program));
    let execute_secs = t.elapsed().as_secs_f64();
    let claims = exec
        .as_ref()
        .is_ok_and(|e| top_claims(&plan, &e.public_words));
    let (instructions, compressions) = (program.instrs.len(), program.groups.b3l.real_rows);
    println!(
        "RF L1 WHOLE: groups {groups} · instructions {instructions} · compressions {compressions} \
         · compressions a group {:.1} · instructions a compression {:.2} · emit {emit:.3}s · \
         arena {arena:.3}s · execute {execute_secs:.3}s · accepted {} · claims the block {claims}",
        compressions as f64 / groups as f64,
        instructions as f64 / compressions.max(1) as f64,
        exec.is_ok(),
    );
    if let Err(e) = &exec {
        println!("RF L1 WHOLE REFUSED: {e:?}");
    }
    assert!(claims, "level 1 executes and claims the block");
    drop(exec);
    drop(arenas);
    drop(program);

    // Group 0's own program, executed honest and tampered.
    let (first, _) = plan.first_groups_program_marked(1).expect("emits");
    let hasher = hasher(&first);
    let arenas = first_groups_arena(&plan, &proof, 1).expect("the arena");
    let exec = execute(&first, &arenas, &hasher);
    let sum = first_groups_bus_sum(&plan, &proof, 1).expect("the host's sum");
    let summed = exec
        .as_ref()
        .is_ok_and(|e| e.public_words.last().is_some_and(|(_, w)| *w == sum));
    println!(
        "RF L1 GROUP0: instructions {} · compressions {} · accepted {} · bus sum {summed}",
        first.instrs.len(),
        first.groups.b3l.real_rows,
        exec.is_ok(),
    );
    assert!(summed, "group 0's program publishes the host's bus sum");
    let (c, r) = later_round(&proof);
    let mut refused_all = true;
    for (what, edit) in tampers(c, r) {
        let mut forged = proof.clone();
        edit(&mut forged);
        let why = match first_groups_arena(&plan, &forged, 1) {
            Err(e) => format!("arena refused: {e}"),
            Ok(arenas) => match execute(&first, &arenas, &hasher) {
                Err(e) => format!("execution refused: {e:?}"),
                Ok(_) => "ACCEPTED".to_string(),
            },
        };
        refused_all &= why.starts_with("execution refused");
        println!("RF L1 TAMPER chain {c} round {r} · {what}: {why}");
    }
    assert!(refused_all, "every tamper is refused in execution");
    println!("RF DONE");
}

/// The levels' chains follow the base's round form: `None` over today's
/// base, the base's shrink over a round-form one; every level's config is
/// then the round form's (and today's is unchanged where the base is today's),
/// and the verifier's artifact check re-derives it. Level 2's program emits
/// over a round-form level 1. Printed: each level's instructions and lean
/// compressions under each form (emitted, not proved).
#[test]
fn the_levels_follow_the_bases_round_form() {
    use super::levels::{derive_levels, level_rounds};
    use multilinear::whir_chain::RoundForm;
    let name = "test_commit_4";
    let folds = WhirFolds::Uniform;
    let mut counts = Vec::new();
    for rounds in [B3Rounds::Fri, B3Rounds::WHIR3] {
        let format = format(rounds, folds);
        let (elf, proof) = base(name, &format);
        let plan = plan(&elf, &proof, &format);
        let want = match rounds {
            B3Rounds::Fri => None,
            B3Rounds::Whir { shrink, .. } => Some(shrink),
        };
        assert_eq!(
            level_rounds(&plan),
            want,
            "{rounds:?}: the levels' round form"
        );
        let levels = derive_levels(&plan, WlfmFamily::B3, true).expect("the levels derive");
        assert_eq!(levels.len(), 2);
        for (k, level) in levels.iter().enumerate() {
            let a = &level.build.artifacts;
            assert_eq!(a.rounds, want, "level {}", k + 1);
            // Every chain the level opens passes the host's round-form checks.
            let (layouts, _) =
                crate::multilinear_prove::stacks(&a.shapes(&a.airs()), &a.sizes, &a.config)
                    .expect("the level's layouts");
            super::wlfm::check_wlfm_round_form(a, &layouts)
                .unwrap_or_else(|e| panic!("level {}: {e}", k + 1));
            multilinear::whir_chain::check_round_form(&a.config)
                .unwrap_or_else(|e| panic!("level {}: {e:?}", k + 1));
            match (want, a.config.format.rounds) {
                (None, RoundForm::Fri) if a.levers() == super::levels::level_levers(a.family) => {
                    let today = super::wlfm::wlfm_config_under(
                        a.rate,
                        a.final_poly,
                        a.fold,
                        a.grind,
                        &a.shapes(&a.airs()),
                    );
                    assert_eq!(a.config, today, "level {}: today's config", k + 1);
                }
                (None, RoundForm::Fri) => {
                    // The last level's own levers (the pinned last level):
                    // the config its levers give without a round form.
                    assert_eq!(k + 1, levels.len(), "only the last level has its own");
                    let own = super::wlfm::wlfm_config_levers(
                        a.rate,
                        &a.levers(),
                        None,
                        &a.shapes(&a.airs()),
                    )
                    .expect("the last level's config");
                    assert_eq!(a.config, own, "level {}: its levers' config", k + 1);
                }
                (Some(s), RoundForm::Whir(w)) => {
                    assert_eq!(w.shrink(), usize::from(s), "level {}", k + 1);
                    assert!(w.caps_by_queries(), "level {}: caps by Q", k + 1);
                }
                (want, got) => panic!("level {}: {want:?} levels under {got:?}", k + 1),
            }
            counts.push((
                rounds,
                k + 1,
                level.program.instrs.len(),
                level.program.groups.b3l.real_rows,
                a.sizes.len(),
            ));
        }
    }
    for (rounds, k, instructions, compressions, groups) in counts {
        println!(
            "LEVELS ROUND FORM {rounds:?} level {k}: {groups} groups · {instructions} \
             instructions · {compressions} lean compressions"
        );
    }
}

/// A level's artifacts under one round form are not another's: the
/// verifier's re-derivation from the held lever refuses a config the lever
/// does not give, and the round form is refused off blowup 2.
#[test]
fn a_level_config_is_its_round_forms_alone() {
    use super::wlfm::{WlfmRate, wlfm_config_rounds};
    let shapes = [(20usize, 18usize), (8, 20), (64, 12)];
    let today = wlfm_config_rounds(WlfmRate::Two, FinalPoly::Off, None, None, None, &shapes)
        .expect("today's");
    assert_eq!(
        today,
        super::wlfm::wlfm_config_under(WlfmRate::Two, FinalPoly::Off, None, None, &shapes),
        "no round form is today's config"
    );
    let w3 = wlfm_config_rounds(WlfmRate::Two, FinalPoly::Off, None, None, Some(3), &shapes)
        .expect("the round form");
    let w2 = wlfm_config_rounds(WlfmRate::Two, FinalPoly::Off, None, None, Some(2), &shapes)
        .expect("the round form");
    assert_ne!(today, w3);
    assert_ne!(w3, w2);
    assert_ne!(
        today.fold_word(),
        w3.fold_word(),
        "the fold word binds the form"
    );
    assert!(
        wlfm_config_rounds(WlfmRate::Four, FinalPoly::Off, None, None, Some(3), &shapes).is_err(),
        "the round form at blowup 4"
    );
}

/// ★ Box tier: a small BLAKE3 block's base under the round form and its two
/// levels, each under the round form, proved; each verifies against its own
/// artifacts; the block's verifier, deriving every level itself, accepts the
/// last; a changed word is refused; level 2 refuses (in execution) a level-1
/// proof with one opened value changed; and the verifier of today's form
/// refuses the round form's last level.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels; box tier"]
fn the_round_form_levels_prove_a_small_block_to_its_last_level() {
    use super::levels::{derive_levels, prove_levels, verify_levels};
    use super::wlfm::verify_wlfm;
    use super::wlfm_leg::{wlfm_leg_arena, wlfm_leg_program};
    let format = format(B3Rounds::WHIR3, WhirFolds::Uniform);
    let (elf, proof) = base("test_commit_4", &format);
    // Today's verifier derives today's levels over the same statement.
    let today_plan = plan(
        &elf,
        &proof,
        &BlockFormat {
            b3_rounds: B3Rounds::Fri,
            ..format
        },
    );
    let plan = plan(&elf, &proof, &format);
    let levels = derive_levels(&plan, WlfmFamily::B3, true).expect("the levels derive");
    let (proofs, secs) = prove_levels(&plan, &levels, &proof).expect("every level proves");
    for (k, (level, p)) in levels.iter().zip(&proofs).enumerate() {
        verify_wlfm(&level.build.artifacts, p, &p.public_words)
            .unwrap_or_else(|e| panic!("level {} verifies: {e}", k + 1));
        println!(
            "LEVEL ROUND FORM {}: {} groups · {} instructions · {} lean compressions · arena \
             {:.3}s · prove {:.3}s",
            k + 1,
            level.build.artifacts.sizes.len(),
            level.program.instrs.len(),
            level.program.groups.b3l.real_rows,
            secs[k].arena,
            secs[k].prove
        );
    }
    let last = proofs.last().expect("a last level");
    verify_levels(&plan, WlfmFamily::B3, last).expect("the block's verifier accepts");
    let mut forged = last.clone();
    forged.public_words[0].1[0] += FE::one();
    assert!(verify_levels(&plan, WlfmFamily::B3, &forged).is_err());

    // Level 2 over a level-1 proof with one value of a later round changed.
    let child = super::levels::child_of(&plan, &levels[0]);
    let program = wlfm_leg_program(&child).expect("level 2 emits");
    let hasher = levels[1].build.artifacts.hasher;
    let mut tampered = proofs[0].clone();
    let (g, c, r) = tampered
        .proof
        .columns
        .iter()
        .enumerate()
        .find_map(|(g, column)| {
            column.polys.iter().enumerate().find_map(|(c, chain)| {
                chain
                    .rounds
                    .iter()
                    .enumerate()
                    .skip(1)
                    .find(|(_, round)| match &round.openings {
                        RoundOpenings::Extension(p) => !p.current.is_empty(),
                        _ => false,
                    })
                    .map(|(r, _)| (g, c, r))
            })
        })
        .expect("a later round of level 1 opens its tree");
    match &mut tampered.proof.columns[g].polys[c].rounds[r].openings {
        RoundOpenings::Extension(o) => o.current[0].values[0] += FieldElement::<E>::one(),
        other => panic!("an extension round: {other:?}"),
    }
    let arena = wlfm_leg_arena(&child, &tampered).expect("the arena builds");
    let why = execute(&program, &[arena], &hasher);
    println!(
        "LEVEL ROUND FORM TAMPER level 1 round {r}: {:?}",
        why.as_ref().err()
    );
    assert!(why.is_err(), "level 2 accepted a tampered level 1");

    // The round form's last level is not today's levels'.
    assert!(verify_levels(&today_plan, WlfmFamily::B3, last).is_err());
}

/// A lean BLAKE3 W-LFM proof's identity and bytes under each family,
/// printed: the two-build check for a change to the lean tables' buses or
/// layout (the lean fixture program of `b3l_wlfm_tests`, the tests' fast
/// format). Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1`.
#[test]
#[ignore = "two-build pin: run with LAMBDA_VM_DETERMINISTIC_GRIND=1 in both builds"]
fn b3_lean_pin_print() {
    use super::wlfm::{WlfmLevers, WlfmRate, build_wlfm_under, prove_wlfm, test_format};
    use crypto::hash::blake3::Blake3Std;
    use digest::Digest;
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the pin needs the deterministic grind"
    );
    let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
    let _fast = test_format::set(Some(2));
    let program = super::b3l_wlfm_tests::honest();
    for family in [WlfmFamily::P1, WlfmFamily::B3] {
        let build = build_wlfm_under(
            &program,
            super::hash::HasherKind::Poseidon1W16,
            WlfmRate::Two,
            WlfmLevers::with_family(family),
            false,
        )
        .expect("the lean fixture builds");
        let proof = prove_wlfm(&program, &build, &[]).expect("the lean fixture proves");
        super::wlfm::verify_wlfm(&build.artifacts, &proof, &proof.public_words)
            .expect("and verifies");
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("serializes");
        let digest: [u8; 32] = <Blake3Std as Digest>::digest(&bytes).into();
        println!(
            "B3L PIN {family:?}: id {} · proof {} bytes · proof digest {}",
            hex(&build.artifacts.program_id),
            bytes.len(),
            hex(&digest)
        );
    }
}

/// ★ Level 1 over a small BLAKE3 base under the WORD memory format: under
/// today's form and the round form (first fold 4), the base proves and
/// verifies, level 1's closed program executes over its witness and its words
/// claim the block, and group 0's program publishes the host's bus sum; under
/// the round form three one-value tampers are refused in execution. Printed:
/// the level's instructions and lean compressions under each form.
///
/// Ignored for its cost (two small bases proved); a dev build runs it too,
/// its word-format debug assertions on.
#[test]
#[ignore = "proves two small BLAKE3 words bases; run by name"]
fn level_1_executes_over_a_words_blake3_base_and_refuses_its_tampers() {
    let name = "test_commit_4";
    let streamed = words_options();
    let (fri, w3) = (words_format(B3Rounds::Fri), words_format(B3Rounds::WHIR3));
    let (elf, today) = base_with(name, &fri, streamed.clone());
    let (_, honest) = base_with(name, &w3, streamed);
    let today_plan = plan(&elf, &today, &fri);
    let w3_plan = plan(&elf, &honest, &w3);
    let c_fri = level_1_executes(&today_plan, &today);
    let c_w3 = level_1_executes(&w3_plan, &honest);
    println!(
        "LEVEL 1 WORDS {name}: {} groups · instructions {} (today {}) · lean compressions {} \
         (today {})",
        c_w3.groups, c_w3.instructions, c_fri.instructions, c_w3.compressions, c_fri.compressions,
    );
    let program = w3_plan.closed_block_program().expect("emits");
    let hasher = hasher(&program);
    let (c, r) = later_round(&honest);
    for (what, edit) in tampers(c, r) {
        let mut proof = honest.clone();
        edit(&mut proof);
        let arenas = block_leaf_arena(&w3_plan, &proof, 0)
            .unwrap_or_else(|e| panic!("{what}: the arena builds: {e}"));
        let why = execute(&program, &arenas, &hasher);
        println!("LEVEL 1 WORDS TAMPER {what}: {:?}", why.as_ref().err());
        assert!(why.is_err(), "{what}: level 1 accepted it");
    }
}

/// The small BLAKE3 format under the WORD memory format: the in-row
/// registers it needs, and with them the production format's limb-ordered
/// aligned loads and stores and lean keccak rounds; first fold 4.
fn words_format(rounds: B3Rounds) -> BlockFormat {
    BlockFormat {
        memory: crate::tables::fused_memory::MemoryFormat::Words,
        registers: crate::tables::register::RegisterFormat::InRow,
        memw_aligned: crate::tables::memw_aligned::MemwAlignedOrder::Limbs,
        keccak_rnd: crate::tables::keccak_rnd::KeccakRndForm::Lean,
        ..format(rounds, WhirFolds::Uniform)
    }
}

/// The small blocks' options, streamed at the production window: the word
/// format's tables are built by the windowed build alone.
fn words_options() -> crate::block_whir::BlockOptions {
    let mut options = crate::tests::block_whir_p1_tests::options(3);
    options.window_log2 = Some(crate::block_whir::BLOCK_WINDOW_LOG2);
    options
}

/// ★ Box tier: a small BLAKE3 block under the WORD memory format and the
/// round form, and its two round-form levels, proved; each level verifies
/// against its own artifacts; the block's verifier, deriving every level
/// itself, accepts the last; and a changed word is refused.
#[test]
#[ignore = "proves a small BLAKE3 words block and its levels; box tier"]
fn the_round_form_levels_prove_a_small_words_block_to_its_last_level() {
    use super::levels::{derive_levels, prove_levels, verify_levels};
    use super::wlfm::verify_wlfm;
    let format = words_format(B3Rounds::WHIR3);
    let (elf, proof) = base_with("test_commit_4", &format, words_options());
    let plan = plan(&elf, &proof, &format);
    let levels = derive_levels(&plan, WlfmFamily::B3, true).expect("the levels derive");
    let (proofs, secs) = prove_levels(&plan, &levels, &proof).expect("every level proves");
    for (k, (level, p)) in levels.iter().zip(&proofs).enumerate() {
        verify_wlfm(&level.build.artifacts, p, &p.public_words)
            .unwrap_or_else(|e| panic!("level {} verifies: {e}", k + 1));
        println!(
            "LEVEL WORDS ROUND FORM {}: {} groups · {} instructions · {} lean compressions · \
             rounds {:?} · prove {:.3}s",
            k + 1,
            level.build.artifacts.sizes.len(),
            level.program.instrs.len(),
            level.program.groups.b3l.real_rows,
            level.build.artifacts.rounds,
            secs[k].prove
        );
    }
    let last = proofs.last().expect("a last level");
    verify_levels(&plan, WlfmFamily::B3, last).expect("the block's verifier accepts");
    let mut forged = last.clone();
    forged.public_words[0].1[0] += FE::one();
    assert!(verify_levels(&plan, WlfmFamily::B3, &forged).is_err());
}

/// ★ A level's round form passes the host's checks or is never built,
/// verified or emitted: the config refuses a schedule stamped after its count
/// was sized (a hand-set fold, a final polynomial); and a W-leg over a child
/// whose round form charges a union bound short of its chains' rounds (or a
/// round-0 count the bound does not give) refuses to emit, where the honest
/// child emits. Printed: whether stamping a fold after the count, as the
/// config once allowed, gives a config the host's `check_round_form` refuses.
#[test]
fn a_level_round_form_the_host_refuses_is_never_built_or_emitted() {
    use super::levels::derive_levels;
    use super::wlfm::{WlfmRate, wlfm_config_rounds, wlfm_config_under};
    use super::wlfm_leg::{WlfmChild, wlfm_leg_program};
    use multilinear::whir_chain::{check_round_form, check_round_form_union};
    let shapes = [(20usize, 18usize), (8, 20), (64, 12), (300, 22)];
    let refused = wlfm_config_rounds(
        WlfmRate::Two,
        FinalPoly::Off,
        Some(5u8),
        None,
        Some(3),
        &shapes,
    )
    .expect_err("a stamped schedule under the round form");
    assert!(refused.contains("stamped fold"), "{refused}");
    // A final polynomial moves no round count (a chain's rounds are its
    // schedule's with or without it), so the round form takes it, at today's
    // count, and the host's count check holds (the last level's lever,
    // `last_level_tests`).
    let today = wlfm_config_rounds(WlfmRate::Two, FinalPoly::Off, None, None, Some(3), &shapes)
        .expect("today's round form");
    let with_final = wlfm_config_rounds(
        WlfmRate::Two,
        FinalPoly::Vars(6),
        None,
        None,
        Some(3),
        &shapes,
    )
    .expect("a final polynomial under the round form");
    check_round_form(&with_final).expect("the count is the bound's");
    assert_eq!(with_final.num_queries, today.num_queries);
    assert_eq!(with_final.format.rounds, today.format.rounds);
    assert_ne!(with_final.fold_word(), today.fold_word());
    // The stamping order the config refuses: today's count, then a fold, then
    // the union bound over the folded schedule, at the config's own tallest
    // stack. The pinned fold stamps nothing and passes; a fold that moves the
    // union's round count leaves a count the host refuses.
    let tallest = {
        let today = wlfm_config_under(WlfmRate::Two, FinalPoly::Off, None, None, &shapes);
        let zf = crate::zf_format::ZfFormat {
            whir_stack: today.format.stack,
            ..crate::zf_format::ZfFormat::DEFAULT
        };
        crate::multilinear_prove::tallest_stack(&zf, &shapes)
    };
    let mut refused_any = false;
    for k in [2u8, 3, 4, 5, 6] {
        let stamped = wlfm_config_under(WlfmRate::Two, FinalPoly::Off, Some(k), None, &shapes)
            .with_whir_rounds(3, crate::multilinear_prove::CHAIN_SECURITY_BITS, tallest)
            .expect("a round form");
        let count = check_round_form(&stamped).err();
        let union = check_round_form_union(&stamped, tallest).err();
        println!("STAMPED fold {k} at {tallest} variables: count {count:?} · union {union:?}");
        assert!(
            union.is_none(),
            "the union is taken over the stamped schedule"
        );
        if usize::from(k) == crate::zf_format::PRODUCTION_WHIR_LOG_FOLDING {
            assert!(count.is_none(), "the pinned fold stamps nothing");
        }
        refused_any |= count.is_some();
    }
    assert!(
        refused_any,
        "some stamped fold leaves a count the host refuses"
    );

    let format = format(B3Rounds::WHIR3, WhirFolds::Uniform);
    let (elf, proof) = base("test_commit_4", &format);
    let plan = plan(&elf, &proof, &format);
    let levels = derive_levels(&plan, WlfmFamily::B3, true).expect("the levels derive");
    let honest = super::levels::child_of(&plan, &levels[0]);
    wlfm_leg_program(&honest).expect("the honest child emits");
    let a = &honest.artifacts;
    let (layouts, _) = crate::multilinear_prove::stacks(&a.shapes(&a.airs()), &a.sizes, &a.config)
        .expect("the layouts");
    let tallest = layouts
        .iter()
        .map(|l| l.n_stack())
        .chain(a.stacks.iter().map(|s| s.layout.n_stack()))
        .max()
        .expect("a stack");
    // A union bound over chains a few variables short of the tallest.
    let short = a
        .config
        .with_whir_rounds(
            3,
            crate::multilinear_prove::CHAIN_SECURITY_BITS,
            tallest.saturating_sub(8),
        )
        .and_then(multilinear::whir_chain::ChainConfig::with_caps_by_queries)
        .expect("a round form");
    let child = WlfmChild {
        artifacts: super::wlfm::WlfmArtifacts {
            config: short,
            ..a.clone()
        },
        num_public_words: honest.num_public_words,
        cap_mux: None,
    };
    let why = wlfm_leg_program(&child).expect_err("a child the host refuses");
    println!("SHORT UNION CHILD: {why}");
    assert!(why.contains("RoundFormTarget"), "{why}");
    // The same short bound with round 0's count re-sized for it: the count
    // check holds, and only the union check refuses the taller chains.
    let mut resized = short;
    let union = match resized.format.rounds {
        multilinear::whir_chain::RoundForm::Whir(w) => w.union_rounds(),
        multilinear::whir_chain::RoundForm::Fri => unreachable!("a round form"),
    };
    resized.num_queries = multilinear::query_count::num_queries(
        resized.log_blowup,
        union,
        crate::multilinear_prove::CHAIN_SECURITY_BITS,
        resized.grind.query,
    );
    check_round_form(&resized).expect("the count is the bound's");
    let child = WlfmChild {
        artifacts: super::wlfm::WlfmArtifacts {
            config: resized,
            ..a.clone()
        },
        num_public_words: honest.num_public_words,
        cap_mux: None,
    };
    let why = wlfm_leg_program(&child).expect_err("a chain past the union bound");
    println!("SHORT UNION, COUNT RESIZED: {why}");
    assert!(why.contains("RoundFormUnion"), "{why}");
}

/// Where a level's prepared and group cells go under each form: level 1 and
/// level 2 over a small BLAKE3 block under today's form (first fold 6), the
/// round form at first fold 4 and at first fold 6. Printed per level: every
/// table's real and padded rows, width and preprocessed count; the sums of
/// real and padded instruction rows, prepared cells (padded rows times
/// preprocessed columns) and main cells (padded rows times width), and the
/// share of each the lean BLAKE3 tables take.
#[test]
#[ignore = "probe: proves three small BLAKE3 bases; run by name"]
fn level_cells_by_table_under_each_form() {
    use super::levels::derive_levels;
    let first6 = WhirFolds::First(multilinear::whir_chain::FirstFold::new(6).expect("a fold"));
    for (arm, rounds, folds) in [
        ("C", B3Rounds::Fri, first6),
        ("W", B3Rounds::WHIR3, WhirFolds::Uniform),
        ("W6", B3Rounds::WHIR3, first6),
    ] {
        let format = format(rounds, folds);
        let (elf, proof) = base("test_commit_4", &format);
        let plan = plan(&elf, &proof, &format);
        let levels = derive_levels(&plan, WlfmFamily::B3, true).expect("the levels derive");
        for (k, level) in levels.iter().enumerate() {
            let a = &level.build.artifacts;
            let airs = a.airs();
            let refs = airs.air_refs();
            let shapes = a.shapes(&airs);
            let mut real: Vec<usize> = Vec::new();
            for g in super::wlfm::program_prep_groups(&level.program) {
                let chunks = super::wlfm::chip_chunk_rows(g.real_rows, g.padded_rows, 22);
                let mut left = g.real_rows;
                for rows in chunks {
                    real.push(left.min(rows));
                    left = left.saturating_sub(rows);
                }
            }
            real.extend(
                super::b3l_tables::b3l_prep_groups(&level.program)
                    .iter()
                    .map(|g| g.real_rows),
            );
            let (mut real_sum, mut padded_sum, mut prep, mut main) =
                (0usize, 0usize, 0usize, 0usize);
            let (mut lean_prep, mut lean_main) = (0usize, 0usize);
            for (t, (air, &(width, nv))) in refs.iter().zip(&shapes).enumerate() {
                let padded = 1usize << nv;
                let count = a.counts[t];
                let r = real.get(t).copied().unwrap_or(0);
                real_sum += r;
                padded_sum += padded;
                prep += padded * count;
                main += padded * width;
                let lean = air.name().starts_with("B3l") || air.name().contains("b3l");
                if lean {
                    lean_prep += padded * count;
                    lean_main += padded * width;
                }
                println!(
                    "CELLS {arm} L{}: table {t} {} · real {r} · padded 2^{nv} · fill {:.2} · width {width} \
                     · prepared {count}",
                    k + 1,
                    air.name(),
                    r as f64 / padded as f64
                );
            }
            println!(
                "CELLS {arm} L{} SUM: tables {} · instructions {} · lean compressions {} · real rows \
                 {real_sum} · padded rows {padded_sum} · fill {:.3} · prepared cells {prep} (lean {:.3}) \
                 · main cells {main} (lean {:.3}) · groups {}",
                k + 1,
                refs.len(),
                level.program.instrs.len(),
                level.program.groups.b3l.real_rows,
                real_sum as f64 / padded_sum as f64,
                lean_prep as f64 / prep.max(1) as f64,
                lean_main as f64 / main.max(1) as f64,
                a.sizes.len()
            );
        }
    }
}

/// ★ Route A's card path under the round form: a small BLAKE3 block under
/// the WORD memory format and the round form (first fold 4, then 6), driven
/// with every level's prepared stacks committed on the card and let go
/// (revived at each group's opening, each group opened alone), proves each
/// level's bytes as [`prove_levels`] does over the same base with the stacks
/// held; every level is the round form's, and the block's verifier accepts
/// the last. Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an untimed identity
/// check.
///
/// [`prove_levels`]: super::levels::prove_levels
#[test]
#[ignore = "proves two small BLAKE3 words blocks and their levels twice; box tier, deterministic grind"]
fn the_round_form_driver_with_retired_stacks_proves_the_held_levels_bytes() {
    use super::levels::{
        LevelStop, derive_levels, prove_block_levels, prove_levels, set_stacks_for_test,
        verify_levels,
    };
    use super::wlfm::StackPlace;
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    let first6 = WhirFolds::First(multilinear::whir_chain::FirstFold::new(6).expect("a fold"));
    for folds in [WhirFolds::Uniform, first6] {
        let format = BlockFormat {
            zf: crate::zf_format::ZfFormat {
                whir_folds: folds,
                ..words_format(B3Rounds::WHIR3).zf
            },
            ..words_format(B3Rounds::WHIR3)
        };
        let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
        let opts = ProofOptions::default_test_options();
        let run = {
            let _retired = set_stacks_for_test(Some(StackPlace::CardRetired));
            prove_block_levels(
                &elf,
                &[],
                &opts,
                &format,
                &words_options(),
                WlfmFamily::B3,
                LevelStop::DEFAULT,
                true,
                true,
            )
            .expect("the round-form block proves to its last level")
        };
        assert!(
            run.times
                .builds
                .iter()
                .all(|b| b.place == StackPlace::CardRetired || b.place == StackPlace::Host),
            "{folds:?}: every level's stacks retired (or on the host without a card): {:?}",
            run.times.builds
        );
        let held = {
            let _held = set_stacks_for_test(Some(StackPlace::Card));
            derive_levels(&run.plan, WlfmFamily::B3, false).expect("the levels derive")
        };
        for (k, level) in held.iter().enumerate() {
            assert_eq!(
                level.build.artifacts.rounds,
                Some(3),
                "{folds:?} level {}: the round form",
                k + 1
            );
        }
        let (proofs, _) =
            prove_levels(&run.plan, &held, &run.base_proof).expect("the levels prove");
        assert_eq!(proofs.len(), run.proofs.len());
        for (k, (a, b)) in run.proofs.iter().zip(&proofs).enumerate() {
            let a = rkyv::to_bytes::<rkyv::rancor::Error>(a).expect("serializes");
            let b = rkyv::to_bytes::<rkyv::rancor::Error>(b).expect("serializes");
            assert!(
                a[..] == b[..],
                "{folds:?} level {}: the held stacks' proof bytes",
                k + 1
            );
            println!(
                "ROUND FORM RETIRED IDENTITY {folds:?} level {}: {} B equal",
                k + 1,
                a.len()
            );
        }
        let last = run.proofs.last().expect("a last level");
        verify_levels(&run.plan, WlfmFamily::B3, last).expect("the block's verifier accepts");
    }
}

// ─── Level 1 over a base whose GKR ladders step two tree layers ─────────────

/// [`words_format`] under the round form with its batched argue's GKR ladder
/// `ladder`.
fn words_ladder(ladder: GkrLadder) -> BlockFormat {
    let mut f = words_format(B3Rounds::WHIR3);
    f.argue = f.argue.with_ladder(ladder);
    f
}

/// One-value tampers of the first bin's two-layer ladder that has a degree-5
/// step with rounds (groups in order, then bins): each of the eight values of
/// that step's first two halves, its first round's last value, and the bin's
/// step 0's first value.
fn ladder_tampers(proof: &BlockWhirProof) -> Vec<(String, Tamper)> {
    let (g, bin, step) = proof
        .argues
        .iter()
        .enumerate()
        .flat_map(|(g, argue)| {
            argue
                .gkr
                .iter()
                .enumerate()
                .filter_map(move |(bin, ladder)| {
                    ladder
                        .layers
                        .iter()
                        .position(|l| {
                            l.sumcheck
                                .rounds
                                .first()
                                .is_some_and(|r| r.evaluations.len() == 5)
                        })
                        .map(|step| (g, bin, step))
                })
        })
        .next()
        .expect("a degree-5 step with rounds in some bin");
    let mut tampers: Vec<(String, Tamper)> = Vec::new();
    for half in 0..2 {
        for value in 0..4 {
            tampers.push((
                format!("group {g} bin {bin} step {step} half {half} value {value}"),
                Box::new(move |p: &mut BlockWhirProof| {
                    let h = &mut p.argues[g].gkr[bin].layers[step].halves[half];
                    *[&mut h.p_lo, &mut h.p_hi, &mut h.q_lo, &mut h.q_hi][value] +=
                        FieldElement::<E>::one();
                }),
            ));
        }
    }
    tampers.push((
        format!("group {g} bin {bin} step {step}'s first round's last value"),
        Box::new(move |p: &mut BlockWhirProof| {
            p.argues[g].gkr[bin].layers[step].sumcheck.rounds[0].evaluations[4] +=
                FieldElement::<E>::one();
        }),
    ));
    tampers.push((
        format!("group {g} bin {bin} step 0's first value"),
        Box::new(move |p: &mut BlockWhirProof| {
            p.argues[g].gkr[bin].layers[0].halves[0].p_lo += FieldElement::<E>::one();
        }),
    ));
    tampers
}

/// ★ Level 1 over a small BLAKE3 base whose GKR ladders step two tree layers
/// at a time (the word memory format, the round form): the base proves and
/// verifies on the host; level 1's closed program executes over its witness,
/// its words claim the block, and group 0's program publishes the host's bus
/// sum; a plan under one ladder takes no proof of the other; and one-value
/// tampers of the two-layer ladder are refused in execution. Printed: the
/// level's instructions and lean compressions against the one-layer ladder's
/// over the same block.
#[test]
#[ignore = "proves two small BLAKE3 words bases; box tier"]
fn level_1_executes_over_a_two_layer_blake3_base_and_refuses_its_tampers() {
    let name = "test_commit_4";
    let (single, double) = (
        words_ladder(GkrLadder::Single),
        words_ladder(GkrLadder::Double),
    );
    let (elf, one) = base_with(name, &single, words_options());
    let (_, two) = base_with(name, &double, words_options());
    let one_plan = plan(&elf, &one, &single);
    let two_plan = plan(&elf, &two, &double);
    let c_one = level_1_executes(&one_plan, &one);
    let c_two = level_1_executes(&two_plan, &two);
    assert_eq!(c_one.groups, c_two.groups, "the same groups");
    println!(
        "LEVEL 1 TWO-LAYER {name}: {} groups · instructions {} (one layer {}) · lean compressions \
         {} (one layer {}) · group 0: instructions {} (one layer {}) · compressions {} (one layer \
         {})",
        c_two.groups,
        c_two.instructions,
        c_one.instructions,
        c_two.compressions,
        c_one.compressions,
        c_two.first_instructions,
        c_one.first_instructions,
        c_two.first_compressions,
        c_one.first_compressions,
    );

    // A plan under one ladder takes no proof of the other. The arena builder
    // (the prover's side) refuses another word count; in-guest the leaf reads
    // every word by its own plan's shape, so another ladder's words are
    // values its constraints refuse, whatever arena a prover writes.
    for (what, plan, proof) in [
        ("a two-layer plan over a one-layer proof", &two_plan, &one),
        ("a one-layer plan over a two-layer proof", &one_plan, &two),
    ] {
        let program = plan.closed_block_program().expect("emits");
        let refused = block_leaf_arena(plan, proof, 0).map_or(true, |arenas| {
            execute(&program, &arenas, &hasher(&program)).is_err()
        });
        assert!(refused, "{what}");
    }

    let program = two_plan.closed_block_program().expect("emits");
    let hasher = hasher(&program);
    for (what, edit) in ladder_tampers(&two) {
        let mut proof = two.clone();
        edit(&mut proof);
        let arenas = block_leaf_arena(&two_plan, &proof, 0)
            .unwrap_or_else(|e| panic!("{what}: the arena builds: {e}"));
        let why = execute(&program, &arenas, &hasher);
        println!("LEVEL 1 TWO-LAYER TAMPER {what}: {:?}", why.as_ref().err());
        assert!(why.is_err(), "{what}: level 1 accepted it");
    }
}

/// ★ Box tier: a small BLAKE3 words block under the round form whose GKR
/// ladders step two tree layers, and its levels, proved; each level verifies
/// against its own artifacts; the block's verifier, deriving every level
/// itself, accepts the last; and a changed word is refused.
#[test]
#[ignore = "proves a small BLAKE3 words block and its levels; box tier"]
fn the_two_layer_ladder_levels_prove_a_small_words_block_to_its_last_level() {
    use super::levels::{derive_levels, prove_levels, verify_levels};
    use super::wlfm::verify_wlfm;
    let format = words_ladder(GkrLadder::Double);
    let (elf, proof) = base_with("test_commit_4", &format, words_options());
    let plan = plan(&elf, &proof, &format);
    let levels = derive_levels(&plan, WlfmFamily::B3, true).expect("the levels derive");
    let (proofs, secs) = prove_levels(&plan, &levels, &proof).expect("every level proves");
    for (k, (level, p)) in levels.iter().zip(&proofs).enumerate() {
        verify_wlfm(&level.build.artifacts, p, &p.public_words)
            .unwrap_or_else(|e| panic!("level {} verifies: {e}", k + 1));
        println!(
            "LEVEL WORDS TWO-LAYER {}: {} groups · {} instructions · {} lean compressions · \
             prove {:.3}s",
            k + 1,
            level.build.artifacts.sizes.len(),
            level.program.instrs.len(),
            level.program.groups.b3l.real_rows,
            secs[k].prove
        );
    }
    let last = proofs.last().expect("a last level");
    verify_levels(&plan, WlfmFamily::B3, last).expect("the block's verifier accepts");
    let mut forged = last.clone();
    forged.public_words[0].1[0] += FE::one();
    assert!(verify_levels(&plan, WlfmFamily::B3, &forged).is_err());
}

/// ★ The last level's own format over a small BLAKE3 block's levels, derived
/// on the host: under every lever of it, level 1 is today's (the same
/// identity and config) and level 2 alone takes the levers; the process's
/// last-level format reaches the verifier's derivation; a chain stopped by
/// its group count gives the levers to whichever level is its last; and the
/// pinned verifier's process check refuses a non-default last level.
#[test]
fn the_last_levels_levers_reach_the_last_level_alone() {
    use super::last_level::{LastLevel, set_last_level_for_test};
    use super::levels::{LevelStop, derive_levels_last, derive_levels_to};
    use super::wlfm::WlfmPrep;
    let all = LastLevel {
        evaluated: true,
        first_fold: Some(4),
        final_poly: FinalPoly::Vars(9),
    };
    let format = format(B3Rounds::WHIR3, WhirFolds::Uniform);
    let (elf, proof) = base("test_commit_4", &format);
    let plan = plan(&elf, &proof, &format);
    let stop = LevelStop::Levels(2);
    let today = derive_levels_last(&plan, WlfmFamily::B3, stop, LastLevel::OFF, true)
        .expect("today's levels");
    let on = derive_levels_last(&plan, WlfmFamily::B3, stop, all, true).expect("the levels");
    assert_eq!(on.len(), 2);
    let (t1, o1) = (&today[0].build.artifacts, &on[0].build.artifacts);
    assert_eq!(o1.program_id, t1.program_id, "level 1's identity");
    assert_eq!(o1.config, t1.config, "level 1's config");
    assert_eq!(o1.prep, WlfmPrep::Stacked);
    let (t2, o2) = (&today[1].build.artifacts, &on[1].build.artifacts);
    assert_eq!(t2.prep, WlfmPrep::Stacked);
    assert_eq!(o2.prep, WlfmPrep::Evaluated);
    assert!(o2.stacks.is_empty() && !o2.evaluated.is_empty());
    assert_eq!(
        (o2.first_fold, o2.final_poly),
        (Some(4), FinalPoly::Vars(9))
    );
    assert_eq!(o2.rounds, t2.rounds, "the base's round form");
    assert_ne!(o2.program_id, t2.program_id);
    assert_eq!(
        on[1].program.instrs.len(),
        today[1].program.instrs.len(),
        "level 2's program is level 1's verifier either way"
    );
    // The process's format reaches the derivation (the verifier's): every
    // lever by default (the pinned BLAKE3 last level), every lever off when
    // the process turns them off, which the pinned verifier refuses.
    let derived = derive_levels_to(&plan, WlfmFamily::B3, stop, true).expect("derives");
    assert_eq!(
        derived[1].build.artifacts.program_id, o2.program_id,
        "the pinned last level by default"
    );
    assert_eq!(derived[0].build.artifacts.program_id, t1.program_id);
    super::levels::check_process_format_for_tests(WlfmFamily::B3)
        .expect("the default process is the pinned format");
    {
        let _last = set_last_level_for_test(LastLevel::OFF);
        let derived = derive_levels_to(&plan, WlfmFamily::B3, stop, true).expect("derives");
        assert_eq!(derived[1].build.artifacts.program_id, t2.program_id);
        assert_eq!(derived[0].build.artifacts.program_id, t1.program_id);
        let e = super::levels::check_process_format_for_tests(WlfmFamily::B3)
            .expect_err("the pinned verifier's process check");
        assert!(e.contains("last level"), "{e}");
    }
    // Stopped by group count: the levers go to whichever level is the last.
    let groups = LevelStop::Groups(1);
    let by_groups =
        derive_levels_last(&plan, WlfmFamily::B3, groups, all, true).expect("the levels");
    let (last, rest) = by_groups.split_last().expect("a level");
    assert_eq!(last.build.artifacts.prep, WlfmPrep::Evaluated);
    assert_eq!(last.build.artifacts.first_fold, Some(4));
    for level in rest {
        assert_eq!(level.build.artifacts.prep, WlfmPrep::Stacked);
        assert_eq!(level.build.artifacts.first_fold, None);
    }
    println!(
        "LAST LEVEL SMALL BLOCK: level 2 today {} groups, prepared {} stacks · by groups {} levels",
        t2.sizes.len(),
        t2.stacks.len(),
        by_groups.len()
    );
}

/// ★ Box tier: a small BLAKE3 block's two levels proved with the last level
/// under all its levers; each level verifies against its own artifacts; the
/// block's verifier under the same process format accepts the last and
/// refuses a changed word; today's verifier refuses it; level 1's proof is
/// today's byte for byte (compact bytes, with
/// `LAMBDA_VM_DETERMINISTIC_GRIND=1`); the last proof's census printed.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels; box tier"]
fn the_last_levels_levers_prove_a_small_block() {
    use super::last_level::{LastLevel, set_last_level_for_test};
    use super::levels::{LevelStop, derive_levels_last, prove_levels, verify_levels};
    use super::wlfm::verify_wlfm;
    let all = LastLevel {
        evaluated: true,
        first_fold: Some(4),
        final_poly: FinalPoly::Vars(9),
    };
    let format = format(B3Rounds::WHIR3, WhirFolds::Uniform);
    let (elf, proof) = base("test_commit_4", &format);
    let plan = plan(&elf, &proof, &format);
    let stop = LevelStop::Levels(2);
    let today = derive_levels_last(&plan, WlfmFamily::B3, stop, LastLevel::OFF, true)
        .expect("today's levels");
    let (today_proofs, _) = prove_levels(&plan, &today, &proof).expect("today proves");
    let on = derive_levels_last(&plan, WlfmFamily::B3, stop, all, true).expect("the levels");
    let (proofs, _) = prove_levels(&plan, &on, &proof).expect("every level proves");
    for (k, (level, p)) in on.iter().zip(&proofs).enumerate() {
        verify_wlfm(&level.build.artifacts, p, &p.public_words)
            .unwrap_or_else(|e| panic!("level {} verifies: {e}", k + 1));
    }
    let last = proofs.last().expect("a last level");
    {
        let _last = set_last_level_for_test(all);
        verify_levels(&plan, WlfmFamily::B3, last).expect("the block's verifier accepts");
        let mut forged = last.clone();
        forged.public_words[0].1[0] += FE::one();
        assert!(verify_levels(&plan, WlfmFamily::B3, &forged).is_err());
    }
    let today_last = today_proofs.last().expect("today's last");
    // The default process is the pinned last level (every lever on): it
    // accepts the last level's own and refuses every lever off.
    verify_levels(&plan, WlfmFamily::B3, last).expect("the default verifier accepts");
    assert!(
        verify_levels(&plan, WlfmFamily::B3, today_last).is_err(),
        "the default verifier refuses every lever off"
    );
    {
        let _off = set_last_level_for_test(LastLevel::OFF);
        assert!(
            verify_levels(&plan, WlfmFamily::B3, last).is_err(),
            "the verifier for every lever off refuses the last level's own format"
        );
        verify_levels(&plan, WlfmFamily::B3, today_last)
            .expect("the verifier for every lever off, its proof");
    }
    if std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some() {
        assert_eq!(
            super::wire::encode(&proofs[0]),
            super::wire::encode(&today_proofs[0]),
            "level 1's proof is today's"
        );
    }
    for (name, p) in [("today", today_last), ("last level", last)] {
        let c = super::wire::census(p);
        println!(
            "SMALL BLOCK LEVEL 2 CENSUS {name}: rkyv {} · compact {} · dedup {} · {}",
            c.rkyv,
            c.compact,
            c.dedup,
            c.line()
        );
    }
}

/// One level's pin: its fold word, round-0 count, round counts at its
/// tallest stack and program identity.
fn level_pin(level: &super::levels::Level) -> String {
    let a = &level.build.artifacts;
    let (layouts, _) = crate::multilinear_prove::stacks(&a.shapes(&a.airs()), &a.sizes, &a.config)
        .expect("the layouts");
    let tallest = layouts.iter().map(|l| l.n_stack()).max().expect("a stack");
    format!(
        "fold word {:#x} · Q {} · rounds at {tallest} {:?} · id {}",
        a.config.fold_word(),
        a.config.num_queries,
        a.config.round_queries(tallest),
        a.program_id
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    )
}

/// ★ LITERAL PINS: a small BLAKE3 block's two levels under the round form at
/// shrink 3 (the benchmark's), derived on the host. Level 1, and level 2 with
/// every last-level lever off, are the values printed at the head before the
/// last level had a format of its own (both heads printed them equal); level
/// 2 at the pinned last level (every lever on, the default) is its own
/// value, pinned when the default was flipped.
#[test]
#[ignore = "derives a small BLAKE3 block's levels three ways; box tier, literal pins"]
fn round_form_level_pins() {
    use super::last_level::LastLevel;
    use super::levels::{LevelStop, derive_levels, derive_levels_last};
    const LEVEL_1: &str = "fold word 0x8041eb0180070006 · Q 229 · rounds at 27 [229, 57, 46, 39, \
                           33, 29, 26] · id \
                           b71ac280daebd66bd994de83902324d40502641e9035651a7123ca4214337469";
    const LEVEL_2_OFF: &str = "fold word 0x8041eb0180070006 · Q 229 · rounds at 27 [229, 57, 46, \
                               39, 33, 29, 26] · id \
                               4e902bb869c8bbbf7d0e18a4d97eb0603540907ab1d366f77bb5ea50d6f3231e";
    const LEVEL_2_PINNED: &str = "UNPINNED";
    let format = format(B3Rounds::WHIR3, WhirFolds::Uniform);
    let (elf, proof) = base("test_commit_4", &format);
    let plan = plan(&elf, &proof, &format);
    let pinned = derive_levels(&plan, WlfmFamily::B3, true).expect("the levels derive");
    let off = derive_levels_last(
        &plan,
        WlfmFamily::B3,
        LevelStop::DEFAULT,
        LastLevel::OFF,
        true,
    )
    .expect("the levels derive with every lever off");
    assert_eq!((pinned.len(), off.len()), (2, 2));
    let got = [
        ("level 1", level_pin(&pinned[0]), LEVEL_1),
        ("level 1, every lever off", level_pin(&off[0]), LEVEL_1),
        ("level 2, every lever off", level_pin(&off[1]), LEVEL_2_OFF),
        ("level 2, pinned", level_pin(&pinned[1]), LEVEL_2_PINNED),
    ];
    for (name, pin, _) in &got {
        println!("LEVEL PIN {name}: {pin}");
    }
    for (name, pin, want) in got {
        assert_eq!(pin, want, "{name}");
    }
}
