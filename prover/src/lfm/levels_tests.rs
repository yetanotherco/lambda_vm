//! The block's recursion as levels ([`super::levels`]) over small blocks: the
//! closed level-1 program, the chain to its last level, and the refusals that
//! make each piece load-bearing. Every test proves a block's base: box tier.

use super::executor::execute;
use super::levels::{
    LevelStop, derive_levels, derive_levels_to, prove_levels, verify_levels, verify_levels_to,
};
use super::whir_block::{LeafChecks, block_leaf_arena, top_claims};
use super::whir_block_tests::{at_blowup_4, plan_of, small_block, small_format};
use super::wlfm::{WlfmFamily, verify_wlfm};

/// ★ Level 1 over a small block: one program over every group executes over
/// the base proof's witness, closes the bus and publishes the top's words,
/// which claim the block.
#[test]
#[ignore = "proves a small block's base; box tier"]
fn the_closed_block_program_verifies_a_small_block() {
    let format = at_blowup_4(small_format());
    for name in ["test_commit_4", "all_instructions_64"] {
        let (elf, proof) = small_block(name, &format);
        let plan = plan_of(&elf, &proof, &format, Some(1));
        let program = plan
            .closed_block_program()
            .expect("the closed program emits");
        let arenas = block_leaf_arena(&plan, &proof, 0).expect("the arena");
        let hasher = program.hasher(super::hash::HasherKind::Poseidon1W16);
        let exec = execute(&program, &arenas, &hasher).expect("the closed program accepts");
        assert_eq!(exec.public_words.len(), plan.top_layout().total());
        assert!(
            top_claims(&plan, &exec.public_words),
            "{name}: the words claim the block"
        );
        println!(
            "LEVEL 1 {name}: {} groups, {} instructions, {} hash rows, {} lean compressions",
            plan.num_groups(),
            program.instrs.len(),
            program.groups.hash.real_rows,
            program.groups.b3l.real_rows
        );
    }
}

/// ★ MUTATION: the closed program's bus assert is load-bearing. With the
/// COMMIT-bus target left out of the sum, the bus does not close: the closed
/// program refuses to execute, and the same program without the assert
/// executes and publishes a nonzero sum.
#[test]
#[ignore = "proves a small block's base; box tier"]
fn the_closed_bus_assert_refuses_an_open_bus() {
    let format = at_blowup_4(small_format());
    let (elf, proof) = small_block("test_commit_4", &format);
    assert!(!proof.public_output.is_empty(), "the block commits words");
    let plan = plan_of(&elf, &proof, &format, Some(1));
    let arenas = block_leaf_arena(&plan, &proof, 0).expect("the arena");
    let no_target = LeafChecks {
        prepared: true,
        target: false,
    };
    let hasher = super::hash::HasherKind::Poseidon1W16;
    let closed = plan
        .closed_block_program_with(no_target, true)
        .expect("emits");
    assert!(
        execute(&closed, &arenas, &hasher).is_err(),
        "an open bus is refused by the assert"
    );
    let open = plan
        .closed_block_program_with(no_target, false)
        .expect("emits");
    let exec = execute(&open, &arenas, &hasher).expect("without the assert it executes");
    let sum = exec.public_words.last().expect("the sum").1;
    assert!(
        sum.iter().any(|v| *v != crate::tables::types::FE::zero()),
        "and its sum is not zero"
    );
}

/// ★ THE CHAIN: every level of a small block proves; each verifies against
/// its own artifacts; the host verifier, deriving every level itself, accepts
/// the last; and a last level with a word changed is refused.
#[test]
#[ignore = "proves a small block and its levels; box tier"]
fn the_levels_prove_a_small_block_to_its_last_level() {
    let format = at_blowup_4(small_format());
    let (elf, proof) = small_block("test_commit_4", &format);
    let plan = plan_of(&elf, &proof, &format, Some(1));
    let levels = derive_levels(&plan, WlfmFamily::P1, false).expect("the levels derive");
    assert_eq!(levels.len(), 2, "the default chain is two levels");
    let (proofs, secs) = prove_levels(&plan, &levels, &proof).expect("every level proves");
    for (k, (level, p)) in levels.iter().zip(&proofs).enumerate() {
        verify_wlfm(&level.build.artifacts, p, &p.public_words)
            .unwrap_or_else(|e| panic!("level {} verifies: {e}", k + 1));
        println!(
            "LEVEL {}: {} groups · {} instructions · arena {:.3}s · prove {:.3}s",
            k + 1,
            level.build.artifacts.sizes.len(),
            level.program.instrs.len(),
            secs[k].arena,
            secs[k].prove
        );
    }
    let last = proofs.last().expect("a last level");
    verify_levels(&plan, WlfmFamily::P1, last).expect("the block's verifier accepts");
    let mut forged = last.clone();
    forged.public_words[0].1[0] += crate::tables::types::FE::one();
    assert!(verify_levels(&plan, WlfmFamily::P1, &forged).is_err());
    // A level proved over another block's chain is refused: the chain's
    // programs are this plan's.
    let (elf2, proof2) = small_block("all_instructions_64", &format);
    let plan2 = plan_of(&elf2, &proof2, &format, Some(1));
    assert!(verify_levels(&plan2, WlfmFamily::P1, last).is_err());
}

/// The levels refuse a family whose in-guest arm the plan does not replay.
#[test]
#[ignore = "proves a small block's base; box tier"]
fn the_levels_refuse_a_family_with_no_arm() {
    let format = at_blowup_4(small_format());
    let (elf, proof) = small_block("test_commit_4", &format);
    let plan = plan_of(&elf, &proof, &format, Some(1));
    assert!(derive_levels(&plan, WlfmFamily::B3, true).is_err());
}

/// A chain stopped at one group instead (the compression modes' rule) proves
/// and verifies under its own constant, and the default verifier refuses its
/// last proof when the chains differ in length.
#[test]
#[ignore = "proves a small block and its levels; box tier"]
fn a_chain_stopped_at_one_group_is_its_own_verifiers() {
    let format = at_blowup_4(small_format());
    let (elf, proof) = small_block("test_commit_4", &format);
    let plan = plan_of(&elf, &proof, &format, Some(1));
    let stop = LevelStop::Groups(1);
    let levels = derive_levels_to(&plan, WlfmFamily::P1, stop, false).expect("derives");
    assert!(levels.last().expect("a level").build.artifacts.sizes.len() <= 1);
    let (proofs, _) = prove_levels(&plan, &levels, &proof).expect("proves");
    let last = proofs.last().expect("a last level");
    verify_levels_to(&plan, WlfmFamily::P1, stop, last).expect("its own verifier accepts");
    if levels.len() != 2 {
        assert!(verify_levels(&plan, WlfmFamily::P1, last).is_err());
    }
}
