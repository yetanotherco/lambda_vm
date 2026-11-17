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
    assert_eq!(exec.public_words.last().expect("a sum").1, sum, "the bus sum");
    Census {
        groups: plan.num_groups(),
        instructions: program.instrs.len(),
        compressions: program.groups.b3l.real_rows,
        first_instructions: first.instrs.len(),
        first_compressions: first.groups.b3l.real_rows,
    }
}

/// The first chain of the proof with a round after round 0 that opens its
/// tree: (group, chain, round).
fn later_round(proof: &BlockWhirProof) -> (usize, usize, usize) {
    proof
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
        .expect("a later round opens its tree")
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
    for folds in [crate::zf_format::ZfFormat::DEFAULT.whir_folds, WhirFolds::Uniform] {
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
        let refused = |what: &str, edit: &dyn Fn(&mut BlockWhirProof)| {
            let mut proof = honest.clone();
            edit(&mut proof);
            let arenas = block_leaf_arena(&w3_plan, &proof, 0)
                .unwrap_or_else(|e| panic!("{what}: the arena builds: {e}"));
            let why = execute(&program, &arenas, &hasher);
            println!("LEVEL 1 ROUND FORM TAMPER {folds:?} {what}: {:?}", why.as_ref().err());
            assert!(why.is_err(), "{what}: level 1 accepted it");
        };
        let (g, c, r) = later_round(&honest);
        refused("a later round's opened value", &|p| {
            match &mut p.proof.columns[g].polys[c].rounds[r].openings {
                RoundOpenings::Extension(o) => o.current[0].values[0] += FieldElement::<E>::one(),
                other => panic!("an extension round: {other:?}"),
            }
        });
        refused("round 0's out-of-domain answer", &|p| {
            let round = &mut p.proof.columns[g].polys[c].rounds[0];
            let value = round.ood_value.as_mut().expect("round 0 has a successor");
            *value += FieldElement::<E>::one();
        });
        refused("a round-0 opened value", &|p| {
            match &mut p.proof.columns[g].polys[c].rounds[0].openings {
                RoundOpenings::Base(o) => o.current[0].values[0] += FE::one(),
                RoundOpenings::BaseGrouped(o) => o.current[0][0].values[0] += FE::one(),
                RoundOpenings::BaseJoint(o) => o.current[0].values[0] += FE::one(),
                RoundOpenings::Extension(o) => o.current[0].values[0] += FieldElement::<E>::one(),
            }
        });
    }
}
