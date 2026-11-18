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
use multilinear::whir_chain::{ArgueFormat, FinalPoly, GroupFold, RoundOpenings, WhirFolds};
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
fn later_round(proof: &BlockWhirProof) -> (usize, usize) {
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

/// A one-value edit of a base proof.
type Tamper = Box<dyn Fn(&mut BlockWhirProof)>;

/// Three one-value tampers in group 0's chain `c`: round `r`'s first opened
/// value, round 0's out-of-domain answer and round 0's first opened value.
fn tampers(c: usize, r: usize) -> Vec<(&'static str, Tamper)> {
    vec![
        (
            "a later round's opened value",
            Box::new(move |p: &mut BlockWhirProof| {
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
            Box::new(move |p: &mut BlockWhirProof| {
                let round = &mut p.proof.columns[0].polys[c].rounds[0];
                let value = round.ood_value.as_mut().expect("round 0 has a successor");
                *value += FieldElement::<E>::one();
            }),
        ),
        (
            "a round-0 opened value",
            Box::new(move |p: &mut BlockWhirProof| {
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
    let (proof, _) = crate::base_only::with_blake3_levels(|| {
        crate::block_whir::prove_block_whir_observed_groups(
            &elf,
            &input,
            &opts,
            format,
            &cfg.options,
            &observe,
            &on_group,
        )
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
                (None, RoundForm::Fri) => {
                    let today = super::wlfm::wlfm_config_under(
                        a.rate,
                        a.final_poly,
                        a.fold,
                        a.grind,
                        &a.shapes(&a.airs()),
                    );
                    assert_eq!(a.config, today, "level {}: today's config", k + 1);
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
    for (fold, final_poly) in [(Some(5u8), FinalPoly::Off), (None, FinalPoly::Vars(6))] {
        let refused = wlfm_config_rounds(WlfmRate::Two, final_poly, fold, None, Some(3), &shapes)
            .expect_err("a stamped schedule under the round form");
        assert!(refused.contains("today's schedule"), "{refused}");
    }
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
    };
    let why = wlfm_leg_program(&child).expect_err("a chain past the union bound");
    println!("SHORT UNION, COUNT RESIZED: {why}");
    assert!(why.contains("RoundFormUnion"), "{why}");
}
