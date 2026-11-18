//! The block's recursion as levels ([`super::levels`]) over small blocks: the
//! closed level-1 program, the chain to its last level, and the refusals that
//! make each piece load-bearing, under Poseidon1 and under 7-round BLAKE3.
//! Every test that proves a block's base is box tier; the hash and base
//! pairings are checked on the laptop.

use super::edsl::WrapHash;
use super::executor::execute;
use super::levels::{
    LevelStop, check_base, check_hash, derive_levels, derive_levels_to, level_plan, prove_levels,
    verify_levels, verify_levels_to,
};
use super::whir_block::{LeafChecks, WhirBlockPlan, block_leaf_arena, top_claims};
use super::whir_block_tests::{
    at_blowup_4, plan_of, small_block, small_block_options, small_format,
};
use super::wlfm::{StackPlace, WlfmFamily, verify_wlfm};
use crate::block_whir::{BlockFormat, BlockWhirProof};
use stark::proof::options::{BaseFormat, ProofOptions};

/// The small blocks' format over the 7-round BLAKE3 base.
fn b3_format() -> BlockFormat {
    crate::tests::block_whir_p1_tests::many_groups(BaseFormat::B3_WHIR)
}

/// A small block's BLAKE3 base, proved inside the levels' admission (the
/// driver's own) and accepted by the host verifier (under the test permit).
fn b3_small_block(name: &str, format: &BlockFormat) -> (Vec<u8>, BlockWhirProof) {
    let elf = crate::test_utils::asm_elf_bytes(name);
    let opts = ProofOptions::default_test_options();
    let proof = crate::base_only::with_blake3_levels(|| {
        crate::block_whir::prove_block_whir_with(
            &elf,
            &[],
            &opts,
            format,
            &small_block_options(),
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

/// The small blocks' options, streamed (`window_log2`): what the levels
/// driver needs, since only a streamed base tells it as each group opens.
fn streamed_options() -> crate::block_whir::BlockOptions {
    let mut options = small_block_options();
    options.window_log2 = Some(4);
    options
}

/// Level 1's BLAKE3 plan, as a verifier derives it (prepared roots on the
/// host).
fn b3_plan(elf: &[u8], proof: &BlockWhirProof, format: &BlockFormat) -> WhirBlockPlan {
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

/// One hash end to end: a family's levels take a plan of its own hash and no
/// other.
#[test]
fn the_levels_take_a_plan_of_their_own_hash_only() {
    let hashes = [
        WrapHash::Keccak,
        WrapHash::Blake3,
        WrapHash::Algebraic,
        WrapHash::Poseidon1,
    ];
    for family in [WlfmFamily::P1, WlfmFamily::B3] {
        let own = match family {
            WlfmFamily::P1 => WrapHash::Poseidon1,
            WlfmFamily::B3 => WrapHash::Blake3,
        };
        for wrap in hashes {
            assert_eq!(
                check_hash(wrap, family).is_ok(),
                wrap == own,
                "{family:?} levels over a {wrap:?} plan"
            );
        }
    }
}

/// A family's levels verify their own hash's base and no other, and the driver
/// refuses a mismatched base before it proves anything.
#[test]
fn the_levels_take_a_base_of_their_own_hash_only() {
    let p1 = crate::hash_pin::BLOCK_BASE;
    let b3 = BaseFormat::B3_WHIR;
    assert!(check_base(&p1, WlfmFamily::P1).is_ok());
    assert!(check_base(&b3, WlfmFamily::B3).is_ok());
    assert!(check_base(&b3, WlfmFamily::P1).is_err());
    assert!(check_base(&p1, WlfmFamily::B3).is_err());
    let opts = ProofOptions::default_test_options();
    for (format, family) in [
        (small_format(), WlfmFamily::B3),
        (b3_format(), WlfmFamily::P1),
    ] {
        let Err(err) = super::levels::prove_block_levels(
            &[],
            &[],
            &opts,
            &format,
            &small_block_options(),
            family,
            LevelStop::DEFAULT,
            true,
            true,
        ) else {
            panic!("{family:?} levels proved over another hash's base");
        };
        assert!(err.contains("own hash's base"), "{err}");
    }
}

/// The driver refuses a whole-run (unstreamed) base before proving anything:
/// only a streamed base tells it as each group opens, and without that level
/// 1 would never get its groups.
#[test]
fn the_levels_driver_refuses_an_unstreamed_base() {
    let opts = ProofOptions::default_test_options();
    let mut unstreamed = small_block_options();
    unstreamed.window_log2 = None;
    let Err(err) = super::levels::prove_block_levels(
        &[],
        &[],
        &opts,
        &small_format(),
        &unstreamed,
        WlfmFamily::P1,
        LevelStop::DEFAULT,
        true,
        true,
    ) else {
        panic!("the driver took an unstreamed base");
    };
    assert!(err.contains("must stream"), "{err}");
}

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

/// The levels refuse a family whose hash the plan's programs do not replay:
/// BLAKE3 levels over a Poseidon1 plan.
#[test]
#[ignore = "proves a small block's base; box tier"]
fn the_levels_refuse_another_hashs_family() {
    let format = at_blowup_4(small_format());
    let (elf, proof) = small_block("test_commit_4", &format);
    let plan = plan_of(&elf, &proof, &format, Some(1));
    assert!(derive_levels(&plan, WlfmFamily::B3, true).is_err());
}

/// ★ THE BLAKE3 CHAIN: a small block's 7-round BLAKE3 base, level 1 its
/// in-guest verifier under BLAKE3 (the lean tables' compressions, no felt
/// sponge), level 2 the BLAKE3 W-leg over level 1, both proved under the
/// BLAKE3 family; each verifies against its own artifacts; the host verifier,
/// deriving every level itself, accepts the last; a changed word, another
/// block's plan and the Poseidon1 family are refused.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels; box tier"]
fn the_blake3_levels_prove_a_small_block_to_its_last_level() {
    let format = b3_format();
    let (elf, proof) = b3_small_block("test_commit_4", &format);
    let plan = b3_plan(&elf, &proof, &format);
    assert_eq!(plan.digest_words(), 2, "a BLAKE3 root is two words");
    // The prepared stacks on the host: a level's are gigabytes the card cannot
    // hold beside a group's commit.
    let levels = derive_levels(&plan, WlfmFamily::B3, true).expect("the levels derive");
    assert_eq!(levels.len(), 2, "the default chain is two levels");
    for (k, level) in levels.iter().enumerate() {
        let rows = &level.program.groups;
        assert!(
            rows.b3l.real_rows > 0 && rows.hash.real_rows == 0,
            "level {}: lean compressions only ({} lean, {} felt-sponge rows)",
            k + 1,
            rows.b3l.real_rows,
            rows.hash.real_rows
        );
        assert_eq!(level.build.artifacts.family, WlfmFamily::B3);
    }
    let (proofs, secs) = prove_levels(&plan, &levels, &proof).expect("every level proves");
    for (k, (level, p)) in levels.iter().zip(&proofs).enumerate() {
        verify_wlfm(&level.build.artifacts, p, &p.public_words)
            .unwrap_or_else(|e| panic!("level {} verifies: {e}", k + 1));
        println!(
            "LEVEL B3 {}: {} groups · {} instructions · {} lean compressions · arena {:.3}s · \
             prove {:.3}s",
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
    forged.public_words[0].1[0] += crate::tables::types::FE::one();
    assert!(verify_levels(&plan, WlfmFamily::B3, &forged).is_err());
    assert!(
        verify_levels(&plan, WlfmFamily::P1, last).is_err(),
        "the Poseidon1 family is not this chain's"
    );
    let (elf2, proof2) = b3_small_block("all_instructions_64", &format);
    let plan2 = b3_plan(&elf2, &proof2, &format);
    assert!(verify_levels(&plan2, WlfmFamily::B3, last).is_err());
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

/// ★ THE DRIVER: a small block proved to its last level with level 1's
/// program emitted the moment the statement is final, executed as phase B
/// finishes its groups, and every level built beside phase B with its prepared
/// stacks committed on the host. The base verifies, every level verifies
/// against its own artifacts, and the block's verifier — its plan derived on
/// its own from the ELF and the statement — accepts the last level.
#[test]
#[ignore = "proves a small block and its levels; box tier"]
fn the_block_levels_driver_proves_a_small_block() {
    use super::levels::prove_block_levels;
    let format = at_blowup_4(small_format());
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = stark::proof::options::ProofOptions::default_test_options();
    let options = streamed_options();
    let run = prove_block_levels(
        &elf,
        &[],
        &opts,
        &format,
        &options,
        WlfmFamily::P1,
        LevelStop::DEFAULT,
        true,
        true,
    )
    .expect("the block proves to its last level");
    assert!(
        crate::block_whir::verify_block_whir(&run.base_proof, &elf, &opts, &format)
            .expect("verify runs"),
        "the base verifies"
    );
    assert_eq!(run.proofs.len(), 2);
    for (k, (level, p)) in run.levels.iter().zip(&run.proofs).enumerate() {
        verify_wlfm(&level.build.artifacts, p, &p.public_words)
            .unwrap_or_else(|e| panic!("level {} verifies: {e}", k + 1));
    }
    let plan = plan_of(&elf, &run.base_proof, &format, Some(1));
    verify_levels(
        &plan,
        WlfmFamily::P1,
        run.proofs.last().expect("a last level"),
    )
    .expect("the block's own verifier accepts the last level");
    let t = &run.times;
    println!(
        "LEVELS DRIVER: base {:.3}s · planned {:.3}s · emitted {:.3}s · pieces {} · waves before \
         the base ended {} ({:.3}s) · execute tail {:.3}s · built {:.3}s · levels {:?}",
        t.base,
        t.planned,
        t.emitted,
        t.pieces,
        t.waves_before_base,
        t.wave_secs_before_base,
        t.execute_tail,
        t.built,
        t.levels
    );
}

/// ★ THE BLAKE3 DRIVER: a small block's 7-round BLAKE3 base proved with level
/// 1 executed and its prepared stacks' columns built beside its phase B (the
/// stacks committed on the card once the base ends), then both levels proved
/// under the BLAKE3 family.
/// The base verifies, every level verifies against its own artifacts, and the
/// block's verifier — its BLAKE3 plan derived on its own — accepts the last.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels; box tier"]
fn the_blake3_levels_driver_proves_a_small_block() {
    use super::levels::prove_block_levels;
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    let run = prove_block_levels(
        &elf,
        &[],
        &opts,
        &format,
        &streamed_options(),
        WlfmFamily::B3,
        LevelStop::DEFAULT,
        true,
        true,
    )
    .expect("the BLAKE3 block proves to its last level");
    assert!(
        crate::base_only::with_permit(|| crate::block_whir::verify_block_whir(
            &run.base_proof,
            &elf,
            &opts,
            &format
        )
        .expect("verify runs")),
        "the base verifies"
    );
    assert_eq!(run.proofs.len(), 2);
    for (k, (level, p)) in run.levels.iter().zip(&run.proofs).enumerate() {
        verify_wlfm(&level.build.artifacts, p, &p.public_words)
            .unwrap_or_else(|e| panic!("level {} verifies: {e}", k + 1));
    }
    let plan = b3_plan(&elf, &run.base_proof, &format);
    verify_levels(
        &plan,
        WlfmFamily::B3,
        run.proofs.last().expect("a last level"),
    )
    .expect("the block's own verifier accepts the last level");
    let t = &run.times;
    println!(
        "LEVELS DRIVER B3: base {:.3}s · planned {:.3}s · emitted {:.3}s · pieces {} · waves \
         before the base ended {} ({:.3}s) · execute tail {:.3}s · built {:.3}s · levels {:?}",
        t.base,
        t.planned,
        t.emitted,
        t.pieces,
        t.waves_before_base,
        t.wave_secs_before_base,
        t.execute_tail,
        t.built,
        t.levels
    );
}

/// ★ The driver's stacks are the prover's choice, never the proof's: a small
/// BLAKE3 block driven with every level's prepared stacks committed on the
/// card and let go (revived at each group's opening) proves each level's
/// bytes as [`prove_levels`] does over the same base with the stacks held.
/// Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1` (the grind may take any valid
/// nonce otherwise); an untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind"]
fn the_blake3_driver_with_retired_stacks_proves_the_held_levels_bytes() {
    use super::levels::{prove_block_levels, set_stacks_for_test};
    use super::wlfm::StackPlace;
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    let run = {
        let _retired = set_stacks_for_test(Some(StackPlace::CardRetired));
        prove_block_levels(
            &elf,
            &[],
            &opts,
            &format,
            &streamed_options(),
            WlfmFamily::B3,
            LevelStop::DEFAULT,
            true,
            true,
        )
        .expect("the BLAKE3 block proves to its last level")
    };
    assert!(
        run.times
            .builds
            .iter()
            .all(|b| b.place == StackPlace::CardRetired || b.place == StackPlace::Host),
        "every level's stacks retired (or on the host without a card): {:?}",
        run.times.builds
    );
    let held = {
        let _held = set_stacks_for_test(Some(StackPlace::Card));
        derive_levels(&run.plan, WlfmFamily::B3, false).expect("the levels derive")
    };
    let (proofs, _) = prove_levels(&run.plan, &held, &run.base_proof).expect("the levels prove");
    assert_eq!(proofs.len(), run.proofs.len());
    for (k, (a, b)) in run.proofs.iter().zip(&proofs).enumerate() {
        let a = rkyv::to_bytes::<rkyv::rancor::Error>(a).expect("serializes");
        let b = rkyv::to_bytes::<rkyv::rancor::Error>(b).expect("serializes");
        assert!(
            a[..] == b[..],
            "level {}: the held stacks' proof bytes",
            k + 1
        );
        println!(
            "LEVELS RETIRED IDENTITY level {}: {} B equal",
            k + 1,
            a.len()
        );
    }
}

/// ★ Level 1's program compiled in parallel and validated beside its
/// execution (`LAMBDA_VM_LEVELS_PROGRAM_FAST`) is the same program and the
/// same proofs: a small BLAKE3 block driven with
/// the knob on proves each level's bytes as [`prove_levels`] does over the
/// same base, and the driver reports the validator's verdict. Needs
/// `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind"]
fn the_blake3_driver_with_its_program_fast_proves_the_same_level_bytes() {
    use super::levels::{prove_block_levels, set_program_fast_for_test};
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    let run = {
        let _beside = set_program_fast_for_test(true);
        prove_block_levels(
            &elf,
            &[],
            &opts,
            &format,
            &streamed_options(),
            WlfmFamily::B3,
            LevelStop::DEFAULT,
            true,
            true,
        )
        .expect("the BLAKE3 block proves to its last level")
    };
    let (secs, _) = run
        .times
        .validated_beside
        .expect("level 1 was validated beside its execution");
    assert!(secs > 0.0);
    let held = derive_levels(&run.plan, WlfmFamily::B3, false).expect("the levels derive");
    let (proofs, _) = prove_levels(&run.plan, &held, &run.base_proof).expect("the levels prove");
    assert_eq!(proofs.len(), run.proofs.len());
    for (k, (a, b)) in run.proofs.iter().zip(&proofs).enumerate() {
        let a = rkyv::to_bytes::<rkyv::rancor::Error>(a).expect("serializes");
        let b = rkyv::to_bytes::<rkyv::rancor::Error>(b).expect("serializes");
        assert!(a[..] == b[..], "level {}: the same proof bytes", k + 1);
        println!(
            "LEVELS VALIDATE BESIDE IDENTITY level {}: {} B equal",
            k + 1,
            a.len()
        );
    }
}

/// The BLAKE3 driver with level 1's lean round tables filled beside its
/// execution (`LAMBDA_VM_WLFM_LEAN_BESIDE`, the round chunks held as their
/// records): level 1's fill took the prefill, which was fed every one of its
/// compressions, and the base, every level and the block's verifier accept.
/// (The tables and the level proof's bytes against the fill after the
/// execution: `lean_beside_tests`.)
#[test]
#[ignore = "proves a small BLAKE3 block and its levels; box tier"]
fn the_blake3_levels_driver_fills_level_1_lean_tables_beside() {
    use super::b3l_tables::{B3lHold, test_hold};
    use super::levels::prove_block_levels;
    use super::wlfm::test_format;
    let _records = test_hold::set(B3lHold::Records);
    let _beside = test_format::set_lean_beside(true);
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    let run = prove_block_levels(
        &elf,
        &[],
        &opts,
        &format,
        &streamed_options(),
        WlfmFamily::B3,
        LevelStop::DEFAULT,
        true,
        true,
    )
    .expect("the BLAKE3 block proves to its last level");
    let t = &run.times;
    let (stats, wait) = t.lean_beside.expect("level 1's lean fill ran beside");
    let compressions = run.levels[0].program.groups.b3l.real_rows;
    assert!(compressions > 0, "level 1 has compressions");
    assert_eq!(stats.compressions, compressions, "every compression fed");
    assert!(stats.waves >= 1, "fed by the waves");
    assert!(t.lean_prefilled, "level 1's fill took the prefill");
    assert!(
        crate::base_only::with_permit(|| crate::block_whir::verify_block_whir(
            &run.base_proof,
            &elf,
            &opts,
            &format
        )
        .expect("verify runs")),
        "the base verifies"
    );
    for (k, (level, p)) in run.levels.iter().zip(&run.proofs).enumerate() {
        verify_wlfm(&level.build.artifacts, p, &p.public_words)
            .unwrap_or_else(|e| panic!("level {} verifies: {e}", k + 1));
    }
    let plan = b3_plan(&elf, &run.base_proof, &format);
    verify_levels(
        &plan,
        WlfmFamily::B3,
        run.proofs.last().expect("a last level"),
    )
    .expect("the block's own verifier accepts the last level");
    println!(
        "LEVELS DRIVER B3 LEAN BESIDE: waves {} · compressions {} · fed {:.3}s · wait {:.3}s · \
         fill chips {:.3}s · fill lean {:.3}s",
        stats.waves, stats.compressions, stats.secs, wait, t.fill_split.0, t.fill_split.1
    );
}

/// Level 1's pieces: the front, then the groups in order, one piece a group up
/// to 63, consecutive groups merged beyond; the pieces tile the arena.
#[test]
fn level_1_lands_in_at_most_64_pieces_that_tile_its_arena() {
    use super::levels::level1_pieces;
    for groups in [1usize, 2, 41, 63, 64, 82, 200] {
        let marks: Vec<u32> = (0..=groups as u32).map(|i| 10 + 7 * i).collect();
        let pieces = level1_pieces(&marks);
        assert!(
            pieces.len() <= 64,
            "{groups} groups: {} pieces",
            pieces.len()
        );
        assert_eq!(pieces[0].0, 0..10, "the front first");
        let mut at = 0u32;
        let mut g = 0usize;
        for (range, gs) in &pieces {
            assert_eq!(range.start, at);
            at = range.end;
            if !gs.is_empty() {
                assert_eq!(gs.start, g);
                g = gs.end;
            }
        }
        assert_eq!((at, g), (*marks.last().expect("marks"), groups));
        if groups <= 63 {
            assert_eq!(pieces.len(), groups + 1, "one piece a group");
        }
    }
}

/// ★ A real block through the levels driver (the box's timing harness):
/// `BLOCK_WHIR_ELF`, `BLOCK_WHIR_INPUT`, the tree harness's format and options
/// words (`WhirTreeConfig::from_env`), `RTA_FAMILY` (`p1`, the default, or
/// `b3`: the 7-round BLAKE3 base at `prove-block-base --base b3`'s format —
/// blowup 2, coset leaves, one tree a run, indexed groups — and BLAKE3
/// levels), `RTA_BESIDE` (1, the default: level 1 executes beside phase B; 0
/// the control), `RTA_BUILD_BESIDE` (1: level 1's stacks' columns prepared on
/// the host beside phase B; 0, the default: after the base; their commits are
/// on the card either way, `LAMBDA_VM_LEVELS_STACKS`), `RTA_LEVELS` (the
/// chain's length; 2), `RTA_RUNS` (runs in this process, the first a warm-up;
/// 1). Prints `RTA …` lines, each run's tagged `run k`; everything is verified
/// off the clock.
///
/// `RTA_L1_GROUPS=k1,k2,…` measures level 1's prove rate at a chosen size
/// instead ([`rta_first_groups`]): the base proved once, then one level-1 run
/// an entry over the base's first `k` groups (the first entry a warm-up when
/// there are several); `RTA_L1_STACKS` (`card`, `retired`, `host`, or unset:
/// as the levels' prover places them) places its prepared stacks. `RTA_BESIDE`,
/// `RTA_BUILD_BESIDE`, `RTA_LEVELS` and `RTA_RUNS` do not apply to it.
#[test]
#[ignore = "a real block and its levels: box only"]
fn rta_a_real_block_through_its_levels() {
    use super::whir_block_tree::WhirTreeConfig;
    let _permit = super::device_permit::serial_for_test();
    let elf = std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF"))
        .expect("read the ELF");
    let input = std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT"))
        .expect("read the input");
    let family = match std::env::var("RTA_FAMILY").as_deref().map(str::trim) {
        Err(_) | Ok("p1") => WlfmFamily::P1,
        Ok("b3") => WlfmFamily::B3,
        Ok(other) => panic!("RTA_FAMILY is p1 or b3, not {other}"),
    };
    if family == WlfmFamily::B3 {
        use crate::zf_format::{LeafOrder, ZfFormat};
        // `prove-block-base --base b3`'s format, installed before anything
        // reads it.
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
    }
    let mut cfg = WhirTreeConfig::from_env().unwrap_or_else(|e| panic!("{e}"));
    if family == WlfmFamily::B3 {
        cfg.format.group = multilinear::whir_chain::GroupFold::Indexed;
        assert_eq!(cfg.format.zf.base, BaseFormat::B3_WHIR);
    }
    let beside = std::env::var("RTA_BESIDE").map_or(true, |v| v.trim() != "0");
    let build_beside = std::env::var("RTA_BUILD_BESIDE").is_ok_and(|v| v.trim() == "1");
    let levels: usize =
        std::env::var("RTA_LEVELS").map_or(2, |v| v.trim().parse().expect("RTA_LEVELS"));
    let runs: usize = std::env::var("RTA_RUNS").map_or(1, |v| v.trim().parse().expect("RTA_RUNS"));
    let opts = super::proof::block_base_options();
    println!(
        "RTA CONFIG: base {:?} · tag {} · group polys {} · family {family:?} · levels {levels} · \
         beside {beside} · build beside {build_beside} · runs {runs}",
        cfg.format.zf.base.hash,
        String::from_utf8_lossy(&crate::block_whir::block_format_tag(&cfg.format)),
        cfg.format.group_polys
    );
    if let Ok(list) = std::env::var("RTA_L1_GROUPS") {
        let ks: Vec<usize> = list
            .split(',')
            .map(|k| match k.trim().parse::<usize>() {
                Ok(k) if k > 0 => k,
                _ => panic!("RTA_L1_GROUPS is a list of positive group counts, not {list:?}"),
            })
            .collect();
        let stacks = match std::env::var("RTA_L1_STACKS").as_deref().map(str::trim) {
            Err(_) | Ok("") | Ok("auto") => None,
            Ok("card") => Some(StackPlace::Card),
            Ok("retired") => Some(StackPlace::CardRetired),
            Ok("host") => Some(StackPlace::Host),
            Ok(other) => panic!("RTA_L1_STACKS is card, retired, host or auto, not {other}"),
        };
        rta_first_groups(&elf, &input, &opts, &cfg, family, &ks, stacks);
        println!("RTA DONE");
        return;
    }
    for k in 0..runs.max(1) {
        println!(
            "RTA RUN {k}{}",
            if k == 0 && runs > 1 { " (warm-up)" } else { "" }
        );
        rta_one_run(
            &elf,
            &input,
            &opts,
            &cfg,
            family,
            levels,
            beside,
            build_beside,
        );
    }
    println!("RTA DONE");
}

/// One run of [`rta_a_real_block_through_its_levels`]: proved, printed,
/// verified.
#[allow(clippy::too_many_arguments)]
fn rta_one_run(
    elf: &[u8],
    input: &[u8],
    opts: &ProofOptions,
    cfg: &super::whir_block_tree::WhirTreeConfig,
    family: WlfmFamily,
    levels: usize,
    beside: bool,
    build_beside: bool,
) {
    use super::levels::prove_block_levels;
    let (elf, input) = (elf.to_vec(), input.to_vec());
    let opts = opts.clone();
    // A panic anywhere under the driver (a card that ran out where a path
    // has no fallback, say) is a refused block on its own line too, never a
    // run that just stops.
    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prove_block_levels(
            &elf,
            &input,
            &opts,
            &cfg.format,
            &cfg.options,
            family,
            LevelStop::Levels(levels),
            beside,
            build_beside,
        )
    }))
    .unwrap_or_else(|panic| {
        let what = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        Err(format!("the prover panicked: {what}"))
    })
    .unwrap_or_else(|e| {
        // The driver refused the block: said on its own line, then the test
        // fails with it.
        println!("RTA REFUSED: {e}");
        panic!("{e}")
    });
    let t = &run.times;
    let s = &run.base_stamps;
    println!(
        "RTA BASE: end {:.3}s · execute {:.3}s · build {:.3}s · phase A {:.3}s · phase B {:.3}s",
        t.base, s.execute, s.build, s.phase_a, s.phase_b
    );
    let l1_prove_from = t.built + t.levels.first().map_or(0.0, |l| l.fill);
    println!(
        "RTA LEVEL1 STREAM: planned {:.3}s · emitted {:.3}s · pieces {} · waves before the base \
         ended {} ({:.3}s) · execute tail {:.3}s · built {:.3}s · card idle after the base {:.3}s",
        t.planned,
        t.emitted,
        t.pieces,
        t.waves_before_base,
        t.wave_secs_before_base,
        t.execute_tail,
        t.built,
        l1_prove_from - t.base
    );
    println!(
        "RTA LEVEL1 EMIT: program fast {} · walk {:.3}s · compile {:.3}s · validate before the \
         execution {:.3}s · beside nice {}",
        t.validated_beside.is_some(),
        t.emit_split.0,
        t.emit_split.1,
        t.emit_split.2,
        super::levels::beside_nice()
    );
    match t.validated_beside {
        Some((secs, at)) => println!(
            "RTA LEVEL1 VALIDATE: beside on · {secs:.3}s · done by {at:.3}s · the execution ended at \
             {:.3}s",
            t.levels.first().map_or(0.0, |l| l.executed)
        ),
        None => println!("RTA LEVEL1 VALIDATE: beside off"),
    }
    let landing = &t.landing;
    let list = |xs: Vec<String>| xs.join(" ");
    println!(
        "RTA LEVEL1 LANDING: setup {:.3}s · front at {:.3}s (wave {:.3}s) · base end {:.3}s · arrivals {} · \
         landed {} · bucketed waves {} · public columns {} · host tree lean {} · setup fast {}",
        landing.setup,
        landing.front.0,
        landing.front.1,
        t.base,
        list(
            landing
                .arrivals
                .iter()
                .map(|(g, at)| format!("{g}@{at:.3}"))
                .collect()
        ),
        list(
            landing
                .landed
                .iter()
                .map(|(i, at, secs)| format!("{i}@{at:.3}+{secs:.3}"))
                .collect()
        ),
        super::levels::bucket_waves(),
        multilinear::logup::public_columns(),
        multilinear::logup::host_tree_lean(),
        super::levels::setup_fast(),
    );
    if let Some(p) = &landing.placement {
        let usage = |u: Option<super::placement::Usage>| match u {
            Some((minor, major, vol, invol)) => {
                format!(
                    "faults minor {minor} major {major} · switches voluntary {vol} involuntary {invol}"
                )
            }
            None => "unread".to_string(),
        };
        let mut totals = [0u64; super::placement::MAX_L3];
        for (_, jobs) in &p.waves {
            for (t, j) in totals.iter_mut().zip(jobs) {
                *t += j;
            }
        }
        let used = totals.iter().rposition(|&j| j > 0).map_or(1, |i| i + 1);
        let per_l3 = |jobs: &[u64]| {
            jobs[..used]
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join("/")
        };
        println!(
            "RTA LEVEL1 PLACEMENT: driver cpu@l3 per wave {} · hash jobs per L3 per wave {} · hash jobs per L3 in all {} \
             · driver thread {} · process {}",
            list(
                p.waves
                    .iter()
                    .map(|(cpu, _)| match cpu {
                        Some((cpu, l3)) => format!("{cpu}@{l3}"),
                        None => "?".to_string(),
                    })
                    .collect()
            ),
            list(p.waves.iter().map(|(_, jobs)| per_l3(jobs)).collect()),
            per_l3(&totals),
            usage(p.thread),
            usage(p.process),
        );
    }
    if let Some(x) = &t.level1_exec {
        println!(
            "RTA LEVEL1 EXEC: schedule {:.3}s · need pass {:.3}s · buckets {:.3}s · first wave {:.3}s · setup {:.3}s · \
             levels {} (on the pool {}, {} hashes) · hashing {:.3}s (on the driver's thread {} hashes in {:.3}s) · apply \
             {:.3}s · the rest {:.3}s",
            x.depth_pass,
            x.need_pass,
            x.bucket_build,
            x.first_wave,
            x.setup,
            x.levels,
            x.parallel_levels,
            x.parallel_hashes,
            x.hash_phase,
            x.serial_hashes,
            x.serial_hash_secs,
            x.apply,
            x.residue
        );
    }
    let (lean, lean_wait) = t.lean_beside.unwrap_or_default();
    // Level 1's thread ends once its execution and the lean fill beside it
    // have: what of that is after the base is in series.
    let thread_tail = t
        .levels
        .first()
        .map_or(0.0, |l| (l.executed + lean_wait - t.base).max(0.0));
    println!(
        "RTA LEAN BESIDE: {} · waves {} · compressions {} · fed {:.3}s · wait after the execution \
         {:.3}s · thread tail after the base {:.3}s · prefilled {} · fill chips {:.3}s · fill lean \
         {:.3}s (tables {:.3}s · mux and cap {:.3}s)",
        if t.lean_beside.is_some() { "on" } else { "off" },
        lean.waves,
        lean.compressions,
        lean.secs,
        lean_wait,
        thread_tail,
        t.lean_prefilled,
        t.fill_split.0,
        t.fill_split.1,
        t.lean_parts.0,
        t.lean_parts.1
    );
    for (k, (l, (level, p))) in t
        .levels
        .iter()
        .zip(run.levels.iter().zip(&run.proofs))
        .enumerate()
    {
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(p).map_or(0, |b| b.len());
        let (instructions, compressions) = t.sizes.get(k).copied().unwrap_or_default();
        println!(
            "RTA LEVEL {}: groups {} · instructions {instructions} · compressions {compressions} · \
             execute {:.3}s · fill {:.3}s · prove {:.3}s · end {:.3}s · proof {bytes} B",
            k + 1,
            level.build.artifacts.sizes.len(),
            l.executed - l.execute_from,
            l.fill,
            l.prove,
            l.end
        );
        if let Some(b) = t.builds.get(k) {
            println!(
                "RTA LEVEL {} BUILD: stacks {} · prepared cells {} · prepare {:.3}s · commit {:.3}s \
                 · prefix pack {:.3}s · committed at {:.3}s",
                k + 1,
                place_word(b.place),
                b.cells,
                b.prepare,
                b.commit,
                b.prefix,
                b.committed
            );
        }
        if let Some(st) = t.stamps.get(k) {
            print_level_stamps(&format!("RTA LEVEL {}", k + 1), st);
        }
    }
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let purge = |p: &Option<crate::alloc_purge::Purge>| match p {
        Some(p) => format!(
            "{:.3}s ({:.2} → {:.2} GiB)",
            p.secs,
            p.resident_before as f64 / (1u64 << 30) as f64,
            p.resident_after as f64 / (1u64 << 30) as f64
        ),
        None => "none".to_string(),
    };
    let points: Vec<String> = t
        .resident
        .iter()
        .map(|(at, b)| format!("{at} {:.2} GiB", gib(*b)))
        .collect();
    let (rss, hwm) = rss_gib();
    println!(
        "RTA MEMORY: peak reset {} · VmRSS {} · purge at the base's end {} · purge after the \
         levels {} · purge after level 1's fill {} · purge at the start {} · now VmRSS {rss:.2} GiB · \
         VmHWM {hwm:.2} GiB",
        t.peak_reset,
        points.join(" · "),
        purge(&t.purges[0]),
        purge(&t.purges[1]),
        purge(&t.purges[2]),
        purge(&t.purges[3]),
    );
    let cards: Vec<String> = t
        .card
        .iter()
        .map(|(at, c)| {
            format!(
                "{at}: pool {:.2} GiB (peak before {:.2}) · free {:.2} of {:.2} GiB",
                gib(c.pool_used),
                gib(c.pool_peak),
                gib(c.free),
                gib(c.total)
            )
        })
        .collect();
    println!(
        "RTA CARD: levels pad {:.2} GiB · {}",
        gib(t.card_pad),
        if cards.is_empty() {
            "no card".to_string()
        } else {
            cards.join(" · ")
        }
    );
    let end = t.levels.last().map_or(t.base, |l| l.end);
    println!(
        "RTA RECURSION: {:.3}s after the base (end {end:.3}s)",
        end - t.base
    );
    // Off the clock: the base (a BLAKE3 one under the test permit), every
    // level, and the block's verifier.
    let ok_base = crate::base_only::with_permit(|| {
        crate::block_whir::verify_block_whir(&run.base_proof, &elf, &opts, &cfg.format)
    })
    .expect("verify runs");
    let mut ok_levels = true;
    for (level, p) in run.levels.iter().zip(&run.proofs) {
        ok_levels &= verify_wlfm(&level.build.artifacts, p, &p.public_words).is_ok();
    }
    let ok_last = super::levels::verify_levels_to(
        &run.plan,
        family,
        LevelStop::Levels(levels),
        run.proofs.last().expect("a last level"),
    )
    .is_ok();
    println!("RTA VERIFIED: base {ok_base} · levels {ok_levels} · block {ok_last}");
    // A tampered last proof (one published word changed) is refused against
    // the last level's artifacts.
    let mut tampered = run.proofs.last().expect("a last level").clone();
    if let Some((_, w)) = tampered.public_words.first_mut() {
        w[0] += crate::tables::types::FE::one();
    }
    let top = &run.levels.last().expect("a last level").build.artifacts;
    let refused = verify_wlfm(top, &tampered, &tampered.public_words).is_err();
    println!("RTA TAMPER: a changed word of the last level refused {refused}");
    assert!(ok_base && ok_levels && ok_last, "everything verifies");
    assert!(refused, "a tampered last level is refused");
}

/// One run of level 1 over a block's first `k` groups
/// ([`WhirBlockPlan::first_groups_program_marked`]): its level, its proof, where
/// its prepared stacks went, and its seconds by part.
struct FirstGroupsRun {
    level: super::levels::Level,
    proof: super::wlfm::WlfmProof,
    place: StackPlace,
    prepared_cells: u64,
    emit: f64,
    arena: f64,
    build: f64,
    /// Under the chip-records knob, the build's packing of the chip tables'
    /// prefixes (part of `build`).
    prefix_pack: f64,
    execute: f64,
    fill: f64,
    /// The fill's split: the fixed chips' traces, the lean tables'.
    fill_split: (f64, f64),
    prove: f64,
    /// Narrow tables (and their cells) the prove widened on the host.
    host_widens: (u64, u64),
    /// Tables filled from their records during the prove: on the card, on
    /// the host.
    records_fills: (u64, u64),
    stamps: super::wlfm::WlfmStamps,
}

/// ★ Level 1 over the first `k` groups of `plan`'s block, as the levels'
/// prover runs a level: emitted, its prepared stacks built where `stacks`
/// says (as the levels' prover places them when `None`), executed over the
/// base proof's witness, filled, and proved streamed. Not verified here.
fn prove_first_groups(
    plan: &WhirBlockPlan,
    proof: &BlockWhirProof,
    family: WlfmFamily,
    k: usize,
    stacks: Option<StackPlace>,
) -> Result<FirstGroupsRun, String> {
    use super::levels::{prefix_check, prepare_level, stack_place};
    use super::wlfm::{WlfmProving, prove_wlfm_traces_as, take_wlfm_stamps};
    use std::time::Instant;
    let t = Instant::now();
    let (program, _) = plan.first_groups_program_marked(k)?;
    let emit = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let arenas = super::whir_block::first_groups_arena(plan, proof, k)?;
    let arena = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let pre = prepare_level(&program, family, super::levels::level_rounds(plan))?;
    let cells = pre.prepared_cells();
    let place = stack_place(cells, 1, stacks);
    let build = pre.commit(place)?;
    let level = super::levels::Level { program, build };
    let build = t.elapsed().as_secs_f64();
    let prefix_pack = level.build.chip_prefix_secs();
    let hasher = level.build.artifacts.hasher;
    let t = Instant::now();
    let execution = execute(&level.program, &arenas, &hasher)
        .map_err(|e| format!("the first {k} groups' level executes: {e:?}"))?;
    let execute_secs = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let filled = super::proof::lfm_fill_executed_holding(
        &level.program,
        execution,
        hasher,
        execute_secs,
        &Vec::new(),
        super::wlfm::chip_records(),
    );
    let fill = t.elapsed().as_secs_f64();
    let fill_split = super::proof::take_fill_split().unwrap_or_default();
    let _ = take_wlfm_stamps();
    let widens = multilinear::narrow::host_widens();
    let fills = multilinear::records::fills();
    let t = Instant::now();
    let proof = prove_wlfm_traces_as(&level.build, filled, prefix_check(), WlfmProving::Streamed)
        .map_err(|e| format!("the first {k} groups' level proves: {e}"))?;
    let prove = t.elapsed().as_secs_f64();
    let after = multilinear::narrow::host_widens();
    let host_widens = (after.0 - widens.0, after.1 - widens.1);
    let after = multilinear::records::fills();
    let records_fills = (after.0 - fills.0, after.1 - fills.1);
    let stamps = take_wlfm_stamps().unwrap_or_default();
    Ok(FirstGroupsRun {
        level,
        proof,
        place,
        prepared_cells: cells,
        emit,
        arena,
        build,
        prefix_pack,
        execute: execute_secs,
        fill,
        fill_split,
        prove,
        host_widens,
        records_fills,
        stamps,
    })
}

/// A level prove's stamps as `<tag> PROVE` and one `<tag> GROUP g` line a
/// group: the card prove by phase, the ledger, and each group's turn.
fn print_level_stamps(tag: &str, st: &super::wlfm::WlfmStamps) {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let peak = st
        .groups
        .iter()
        .flat_map(|g| {
            [
                g.commit_reserved,
                g.argue_reserved,
                g.open_reserved,
                g.transition_reserved,
            ]
        })
        .chain([st.card_reserved_at_start])
        .max()
        .unwrap_or(0);
    let cells: usize = st.groups.iter().map(|g| g.cells).sum();
    println!(
        "{tag} PROVE: prep {:.3}s · phase A {:.3}s · phase B {:.3}s · group cells {cells} · card \
         budget {:.2} GiB · reserved at start {:.2} GiB · ledger peak {:.2} GiB · narrow {} tables \
         · records {} tables · prep parts: merge {:.3}s airs {:.3}s pairs {:.3}s check {:.3}s \
         layouts {:.3}s tables {:.3}s",
        st.prep,
        st.phase_a,
        st.phase_b,
        gib(st.card_budget),
        gib(st.card_reserved_at_start),
        gib(peak),
        st.narrow_tables,
        st.records_tables,
        st.prep_parts[0],
        st.prep_parts[1],
        st.prep_parts[2],
        st.prep_parts[3],
        st.prep_parts[4],
        st.prep_parts[5],
    );
    for (g, s) in st.groups.iter().enumerate() {
        println!(
            "{tag} GROUP {g}: tables {} · polys {} · cells {} · wait A {:.3}s · up A {:.3}s (paid \
             {:.3}) · commit {:.3}s · retire {:.3}s · up B {:.3}s · argue {:.3}s · encode {:.3}s · \
             open {:.3}s · prepared revive {:.3}s · open paid {:.3}s · overlapped {} · columns gone \
             before revive {} · turn {:.3}-{:.3}s · GiB promised: commit {:.2}/{:.2} · argue {:.2}/{:.2} · open {:.2}{}",
            s.tables,
            s.polys,
            s.cells,
            s.wait_a,
            s.upload_a,
            s.upload_paid,
            s.commit,
            s.retire,
            s.upload_b,
            s.argue,
            s.encode,
            s.open,
            s.prepared_revive,
            s.open_paid,
            s.overlapped,
            s.columns_gone_before_revive,
            s.start_b,
            s.end_b,
            gib(s.commit_base),
            gib(s.commit_reserved),
            gib(s.argue_base),
            gib(s.argue_reserved),
            gib(s.open_reserved),
            match s.argue_card {
                Some(c) => format!(
                    " · card at the argue: real peak {:.2} of {:.2} GiB (pool reserved peak {:.2}, used \
                     peak {:.2}; free at its end {:.2}) · margin {:.2} GiB",
                    gib(c.real_peak()),
                    gib(c.total),
                    gib(c.reserved_peak),
                    gib(c.used_peak),
                    gib(c.free_now),
                    gib(c.total.saturating_sub(c.real_peak())),
                ),
                None => String::new(),
            },
        );
    }
}

/// A stack place as the harness's lines print it.
fn place_word(place: StackPlace) -> &'static str {
    match place {
        StackPlace::Host => "host",
        StackPlace::Card => "card",
        StackPlace::CardRetired => "card retired",
    }
}

/// This process's resident and peak resident memory, in GiB (Linux; `NaN`
/// elsewhere).
fn rss_gib() -> (f64, f64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let read = |key: &str| {
        status
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .and_then(|v| v.split_whitespace().next()?.parse::<f64>().ok())
            .map_or(f64::NAN, |kb| kb / (1u64 << 20) as f64)
    };
    (read("VmRSS:"), read("VmHWM:"))
}

/// [`rta_a_real_block_through_its_levels`] under `RTA_L1_GROUPS`: the base
/// proved once (its prepared roots handed to the plan, as the driver does)
/// and verified, then one run of level 1 over the base's first `k` groups a
/// list entry ([`prove_first_groups`]). Each run prints `RTA L1 RESULT` (its
/// sizes, its seconds, the card prove — host preparation, phase A and phase
/// B — and its rate a committed main cell and a compression), `RTA L1 SPLIT`
/// and one `RTA L1 GROUP` line a group of the level, then, off the clock,
/// `RTA L1 VERIFIED`: the level's proof verifies and its last word is the
/// host's sum of `p/q` over those groups' bus outputs in the base proof.
#[allow(clippy::too_many_arguments)]
fn rta_first_groups(
    elf: &[u8],
    input: &[u8],
    opts: &ProofOptions,
    cfg: &super::whir_block_tree::WhirTreeConfig,
    family: WlfmFamily,
    ks: &[usize],
    stacks: Option<StackPlace>,
) {
    use std::time::Instant;
    let roots: std::sync::Mutex<Option<Vec<crate::block_whir::PreparedRoots>>> =
        std::sync::Mutex::new(None);
    let observe = |_: crate::block_whir::BlockStatement<'_>,
                   prepared: &[crate::block_whir::PreparedRoots]| {
        *roots.lock().unwrap_or_else(|e| e.into_inner()) = Some(prepared.to_vec());
    };
    let on_group = |_: stark::multilinear_block::GroupOpened<'_, _, _>| {};
    let t = Instant::now();
    let prove_base = || {
        crate::block_whir::prove_block_whir_observed_groups(
            elf,
            input,
            opts,
            &cfg.format,
            &cfg.options,
            &observe,
            &on_group,
        )
    };
    let (proof, s) = match family {
        WlfmFamily::P1 => prove_base(),
        WlfmFamily::B3 => crate::base_only::with_blake3_levels(prove_base),
    }
    .unwrap_or_else(|e| panic!("the base proves: {e}"));
    let base = t.elapsed().as_secs_f64();
    println!(
        "RTA L1 BASE: prove {base:.3}s · execute {:.3}s · build {:.3}s · phase A {:.3}s · phase B \
         {:.3}s",
        s.execute, s.build, s.phase_a, s.phase_b
    );
    let ok_base = crate::base_only::with_permit(|| {
        crate::block_whir::verify_block_whir(&proof, elf, opts, &cfg.format)
    })
    .expect("verify runs");
    let roots = roots
        .into_inner()
        .unwrap_or_else(|e| e.into_inner())
        .expect("the base stated");
    let t = Instant::now();
    let plan = level_plan(
        elf,
        opts,
        &cfg.format,
        proof.statement(),
        family,
        Some(&roots),
    )
    .unwrap_or_else(|e| panic!("the plan derives: {e}"));
    println!(
        "RTA L1 PLAN: base verified {ok_base} · base groups {} · planned {:.3}s",
        plan.num_groups(),
        t.elapsed().as_secs_f64()
    );
    assert!(ok_base, "the base verifies");
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    for (run, &k) in ks.iter().enumerate() {
        let warm = if run == 0 && ks.len() > 1 {
            " (warm-up)"
        } else {
            ""
        };
        let r = prove_first_groups(&plan, &proof, family, k, stacks)
            .unwrap_or_else(|e| panic!("run {run}: {e}"));
        let a = &r.level.build.artifacts;
        let cells: usize = a.shapes(&a.airs()).iter().map(|&(w, v)| w << v).sum();
        let group_cells: usize = r.stamps.groups.iter().map(|g| g.cells).sum();
        let compressions = r.level.program.groups.b3l.real_rows;
        let instructions = r.level.program.instrs.len();
        let st = &r.stamps;
        let card = st.prep + st.phase_a + st.phase_b;
        let peak = st
            .groups
            .iter()
            .flat_map(|g| {
                [
                    g.commit_reserved,
                    g.argue_reserved,
                    g.open_reserved,
                    g.transition_reserved,
                ]
            })
            .chain([st.card_reserved_at_start])
            .max()
            .unwrap_or(0);
        let (rss, hwm) = rss_gib();
        println!(
            "RTA L1 RESULT k {k} · run {run}{warm} · base groups {k} of {} · level groups {} · \
             instructions {instructions} · compressions {compressions} · committed cells {cells} \
             · group cells {group_cells} · prepared cells {} · stacks {} · emit {:.3}s · arena \
             {:.3}s · build {:.3}s · execute {:.3}s · fill {:.3}s · prove {:.3}s · card prove \
             {card:.3}s (prep {:.3}s · phase A {:.3}s · phase B {:.3}s) · ns a cell {:.4} · us a \
             compression {:.4} · execute ns an instruction {:.2} · card budget {:.2} GiB · ledger \
             peak {:.2} GiB · VmRSS {rss:.2} GiB · VmHWM {hwm:.2} GiB",
            plan.num_groups(),
            a.sizes.len(),
            r.prepared_cells,
            place_word(r.place),
            r.emit,
            r.arena,
            r.build,
            r.execute,
            r.fill,
            r.prove,
            st.prep,
            st.phase_a,
            st.phase_b,
            1e9 * card / cells as f64,
            1e6 * card / compressions.max(1) as f64,
            1e9 * r.execute / instructions.max(1) as f64,
            gib(st.card_budget),
            gib(peak),
        );
        println!(
            "RTA L1 SPLIT k {k} · run {run} · fill chips {:.3}s · fill lean {:.3}s · prep parts: \
             merge {:.3}s airs {:.3}s pairs {:.3}s check {:.3}s layouts {:.3}s tables {:.3}s",
            r.fill_split.0,
            r.fill_split.1,
            st.prep_parts[0],
            st.prep_parts[1],
            st.prep_parts[2],
            st.prep_parts[3],
            st.prep_parts[4],
            st.prep_parts[5],
        );
        println!(
            "RTA L1 NARROW k {k} · run {run} · chips narrow {} · pack {:.3}s · tables {} · cells {} \
             · bytes {} · host widens {} tables {} cells",
            if super::wlfm::narrow_chips() {
                "on"
            } else {
                "off"
            },
            st.narrow_pack,
            st.narrow_tables,
            st.narrow_cells,
            st.narrow_bytes,
            r.host_widens.0,
            r.host_widens.1,
        );
        println!(
            "RTA L1 RECORDS k {k} · run {run} · chip records {} · prefix pack (build) {:.3}s · \
             prefix pack (prove) {:.3}s · tables {} · cells {} · bytes {} · fills card {} host {}",
            if super::wlfm::chip_records() {
                "on"
            } else {
                "off"
            },
            r.prefix_pack,
            st.records_prefix_pack,
            st.records_tables,
            st.records_cells,
            st.records_bytes,
            r.records_fills.0,
            r.records_fills.1,
        );
        for (g, s) in st.groups.iter().enumerate() {
            println!(
                "RTA L1 STAMPS k {k} · run {run} · group {g} · wait A {:.3}s · retire {:.3}s · ood \
                 {:.3}s · pack {:.3}s · pack busy {:.3}s · packed tables {} · packed cells {} · \
                 packed bytes {} · open paid {:.3}s · overlapped {}",
                s.wait_a,
                s.retire,
                s.ood,
                s.pack,
                s.pack_busy,
                s.packed_tables,
                s.packed_cells,
                s.packed_bytes,
                s.open_paid,
                s.overlapped,
            );
        }
        for (g, s) in st.groups.iter().enumerate() {
            println!(
                "RTA L1 GROUP k {k} · run {run} · group {g} · tables {} · polys {} · cells {} · \
                 up A {:.3}s (paid {:.3}) · commit {:.3}s · up B {:.3}s · argue {:.3}s · encode \
                 {:.3}s · open {:.3}s · turn {:.3}-{:.3}s · GiB promised: commit {:.2}/{:.2} · \
                 argue {:.2}/{:.2} · open {:.2}",
                s.tables,
                s.polys,
                s.cells,
                s.upload_a,
                s.upload_paid,
                s.commit,
                s.upload_b,
                s.argue,
                s.encode,
                s.open,
                s.start_b,
                s.end_b,
                gib(s.commit_base),
                gib(s.commit_reserved),
                gib(s.argue_base),
                gib(s.argue_reserved),
                gib(s.open_reserved),
            );
        }
        // Off the clock: the level verifies, and its words carry the base
        // groups' bus sum.
        let words = &r.proof.public_words;
        let ok_proof = verify_wlfm(a, &r.proof, words).is_ok();
        let sum = super::whir_block::first_groups_bus_sum(&plan, &proof, k)
            .unwrap_or_else(|e| panic!("the host's sum: {e}"));
        let ok_sum = words.len() == plan.child_layout().total()
            && words.last().is_some_and(|(_, w)| *w == sum);
        println!("RTA L1 VERIFIED k {k} · run {run} · proof {ok_proof} · sum {ok_sum}");
        assert!(ok_proof && ok_sum, "the first {k} groups' level verifies");
    }
}

/// ★ Level 1 over a small BLAKE3 block's first groups (the timing harness's
/// `RTA_L1_GROUPS` program): over one group, its prepared stacks on the host,
/// and over two, where the card holds them, it proves under the BLAKE3 family,
/// verifies, and publishes as its last word the host's sum of `p/q` over those
/// groups' bus outputs; a changed sum word is refused. Over every group its
/// sum is the closed program's own with the COMMIT-bus target left out; no
/// groups, or more than the block has, are refused at emission.
#[test]
#[ignore = "proves a small BLAKE3 block and a level over it; box tier"]
fn the_first_groups_level_proves_and_publishes_their_bus_sum() {
    use super::whir_block::{first_groups_arena, first_groups_bus_sum};
    let format = b3_format();
    let (elf, proof) = b3_small_block("test_commit_4", &format);
    let plan = b3_plan(&elf, &proof, &format);
    let n = plan.num_groups();
    assert!(n >= 2, "the small block has {n} groups");
    assert!(plan.first_groups_program_marked(0).is_err());
    assert!(plan.first_groups_program_marked(n + 1).is_err());
    for (k, stacks) in [
        (1, Some(StackPlace::Host)),
        (2, None),
        (2, Some(StackPlace::CardRetired)),
    ] {
        let run = prove_first_groups(&plan, &proof, WlfmFamily::B3, k, stacks)
            .unwrap_or_else(|e| panic!("{e}"));
        let words = &run.proof.public_words;
        let artifacts = &run.level.build.artifacts;
        verify_wlfm(artifacts, &run.proof, words)
            .unwrap_or_else(|e| panic!("the first {k} groups' level verifies: {e}"));
        assert_eq!(words.len(), plan.child_layout().total());
        let sum = first_groups_bus_sum(&plan, &proof, k).expect("the host's sum");
        assert_eq!(words.last().expect("a sum").1, sum, "k = {k}: the bus sum");
        let mut forged = run.proof.clone();
        forged.public_words.last_mut().expect("a sum").1[0] += crate::tables::types::FE::one();
        assert!(
            verify_wlfm(artifacts, &forged, &forged.public_words).is_err(),
            "k = {k}: a changed sum word is refused"
        );
        println!(
            "LEVEL B3 FIRST {k}: {} level groups · {} instructions · {} lean compressions · \
             stacks on the {} · prove {:.3}s",
            artifacts.sizes.len(),
            run.level.program.instrs.len(),
            run.level.program.groups.b3l.real_rows,
            place_word(run.place),
            run.prove
        );
    }
    // Over every group, executed only: the same sum as the closed program's
    // with the target left out and the bus open.
    let (every, _) = plan.first_groups_program_marked(n).expect("emits");
    let hasher =
        every.hasher(<crate::hash_pin::BlockFamily as crate::hash_pin::ProofFamily>::SOCKET);
    let arenas = first_groups_arena(&plan, &proof, n).expect("the arena");
    let exec = execute(&every, &arenas, &hasher).expect("executes");
    let sum = first_groups_bus_sum(&plan, &proof, n).expect("the host's sum");
    assert_eq!(exec.public_words.last().expect("a sum").1, sum);
    let open = plan
        .closed_block_program_with(
            LeafChecks {
                prepared: true,
                target: false,
            },
            false,
        )
        .expect("emits");
    let closed_arenas = block_leaf_arena(&plan, &proof, 0).expect("the arena");
    let exec = execute(&open, &closed_arenas, &hasher).expect("executes");
    assert_eq!(exec.public_words.last().expect("a sum").1, sum);
}
