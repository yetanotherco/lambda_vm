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
use multilinear::whir_chain::{ArgueFormat, GroupFold, RoundOpenings, WhirFolds};
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
    let elf = crate::test_utils::asm_elf_bytes(name);
    let opts = ProofOptions::default_test_options();
    let proof = crate::base_only::with_blake3_levels(|| {
        crate::block_whir::prove_block_whir_with(
            &elf,
            &[],
            &opts,
            format,
            &crate::tests::block_whir_p1_tests::options(3),
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
