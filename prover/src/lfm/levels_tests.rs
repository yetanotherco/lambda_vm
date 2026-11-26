//! The block's recursion as levels ([`super::levels`]) over small blocks: the
//! closed level-1 program, the chain to its last level, and the refusals that
//! make each piece load-bearing, under Poseidon1 and under 7-round BLAKE3.
//! CI proves a compact two-level chain at the TEST posture. Whole-block
//! level fixtures run on the box.

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

/// [`streamed_options`] with the chunks laid out off the packer's thread and
/// the rest streamed through the sliding finish, as the block lays them out:
/// where the base can state its statement early and the finish can start its
/// late families first. The rest streams only with KECCAK_RND's tables at
/// 2^5 rows or more ([`crate::block_whir::BlockOptions::rest_streams`]); at
/// the small blocks' 2^3 the finish builds the rest whole and states nothing.
fn pipelined_options() -> crate::block_whir::BlockOptions {
    let mut options = streamed_options();
    options.layout_workers = 2;
    options.keccak_rnd_rows_log2 = 5;
    assert!(options.rest_streams(), "the rest streams");
    options
}

/// Level 1's BLAKE3 plan, as a verifier derives it (prepared roots on the
/// host).
fn b3_plan<L>(elf: &[u8], proof: &BlockWhirProof<L>, format: &BlockFormat) -> WhirBlockPlan {
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
    verify_levels_to(
        &plan,
        WlfmFamily::P1,
        stop,
        super::last_level::LastLevel::OFF,
        last,
    )
    .expect("its own verifier accepts");
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
    let last = run.proofs.last().expect("a last level");
    let plan = b3_plan(&elf, &run.base_proof, &format);
    verify_levels(&plan, WlfmFamily::B3, last)
        .expect("the block's own verifier accepts the last level");
    // The verifier's derivation (prepared roots committed on the host, the
    // inverse share form), at this fixture's format and options, accepts the
    // same proof.
    super::levels::verify_block_levels_under(
        &elf,
        &opts,
        &format,
        run.base_proof.statement(),
        WlfmFamily::B3,
        LevelStop::DEFAULT,
        last,
    )
    .expect("the verifier's own plan accepts the last level");
    // A plan given one flipped prepared root (as a prover or a file could
    // hand it): the plan-taking call checks the proof against the programs
    // that plan defines, so its verdict follows the caller's roots — this
    // proof is refused there only because it was proved for the true ones.
    // The verifier's entries take no plan, so no caller's roots reach them.
    let mut roots: Vec<(usize, Vec<multilinear::whir_commit::Commitment>)> = plan
        .prepared()
        .iter()
        .map(|p| (p.group, p.roots.clone()))
        .collect();
    assert!(!roots.is_empty(), "the block has prepared stacks");
    roots[0].1[0][0] ^= 1;
    let flipped = level_plan(
        &elf,
        &opts,
        &format,
        run.base_proof.statement(),
        WlfmFamily::B3,
        Some(&roots),
    )
    .expect("a plan takes the caller's roots, checking only their groups and count");
    assert_eq!(flipped.prepared()[0].roots, roots[0].1, "the caller's root");
    assert!(
        verify_levels(&flipped, WlfmFamily::B3, last).is_err(),
        "the flipped plan defines other programs"
    );
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

/// ★ The levels driver over a BLAKE3 base under local lookups (KECCAK_RND's
/// byte lookups argued per group): the base is proved in its own kind, each
/// group's local argue reaches level 1 as phase B opens the group, every
/// level verifies, and the block's own verifier — its plan derived under the
/// same format — accepts the last. What the whole-pipeline A/B runs on a
/// real block.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels; box tier"]
fn the_blake3_levels_driver_proves_a_local_base_to_its_last_level() {
    use super::levels::prove_block_levels;
    let format = BlockFormat {
        argue: multilinear::whir_chain::ArgueFormat::BATCHED,
        keccak_rnd: crate::tables::keccak_rnd::KeccakRndForm::LeanLocal,
        ..b3_format()
    };
    assert!(format.local_lookups());
    let elf = crate::test_utils::asm_elf_bytes("test_keccak");
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
    .expect("the local BLAKE3 block proves to its last level");
    let locals = run
        .base_proof
        .local
        .as_ref()
        .expect("a base of the local kind")
        .iter()
        .filter(|a| !a.is_empty())
        .count();
    assert!(locals > 0, "the base argues local lookups");
    assert!(
        crate::base_only::with_permit(|| crate::block_whir::verify_block_whir(
            &run.base_proof,
            &elf,
            &opts,
            &format
        )
        .expect("verify runs")),
        "the local base verifies"
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
    // Today's plan over the same statement is refused (its frame has no
    // LOCAL_BYTES): the last level claims a block only its own format derives.
    let todays = BlockFormat {
        keccak_rnd: crate::tables::keccak_rnd::KeccakRndForm::Lean,
        ..format
    };
    assert!(
        super::levels::level_plan(
            &elf,
            &opts,
            &todays,
            run.base_proof.statement(),
            WlfmFamily::B3,
            None
        )
        .is_err(),
        "today's plan over a local statement"
    );
    println!(
        "LEVELS DRIVER B3 LOCAL: local groups {locals} · base {:.3}s · levels {:?}",
        run.times.base, run.times.levels
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

/// Compare the full serialized base and every recursion proof at each pool size.
#[test]
#[ignore = "proves a small BLAKE3 block seven times; run alone with deterministic grind and fixed trace hash"]
fn the_blake3_driver_on_each_program_pool_proves_the_same_bytes() {
    use super::levels::{
        PROGRAM_POOL_ENV, PROGRAM_POOL_FOR_TEST, prove_block_levels, set_program_fast_for_test,
    };
    assert!(crypto::grinding::deterministic());
    assert_eq!(
        std::env::var("LAMBDA_VM_FIXED_TRACE_HASH").as_deref(),
        Ok("1")
    );
    assert!(std::env::var_os(PROGRAM_POOL_ENV).is_none());
    let _serial = super::device_permit::serial_for_test();
    let _options = super::wlfm::test_format::set_driver_options();
    let _fast = set_program_fast_for_test(true);
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    let options = streamed_options();
    let proof_bytes = |threads| {
        PROGRAM_POOL_FOR_TEST.set(threads);
        let run = prove_block_levels(
            &elf,
            &[],
            &opts,
            &format,
            &options,
            WlfmFamily::B3,
            LevelStop::DEFAULT,
            true,
            true,
        );
        PROGRAM_POOL_FOR_TEST.set(None);
        let run = run.expect("the small block proves through level 2");
        assert_eq!(run.proofs.len(), 2);
        assert!(run.times.validated_beside.is_some());
        assert!(run.times.builder_end.is_some());
        let base = rkyv::to_bytes::<rkyv::rancor::Error>(&run.base_proof).expect("serializes");
        let levels: Vec<_> = run
            .proofs
            .iter()
            .map(|p| rkyv::to_bytes::<rkyv::rancor::Error>(p).expect("serializes"))
            .collect();
        (base, levels)
    };
    let (base, levels) = proof_bytes(Some(0));
    for threads in [None, Some(1), Some(2), Some(4), Some(6), Some(8)] {
        let (other_base, other_levels) = proof_bytes(threads);
        let pool = threads.map_or_else(|| "default".into(), |n| format!("{n} threads"));
        assert!(base[..] == other_base[..], "pool {pool}: base");
        for (i, (a, b)) in levels.iter().zip(&other_levels).enumerate() {
            assert!(a[..] == b[..], "pool {pool}: level {}", i + 1);
        }
        println!(
            "LEVELS PROGRAM POOL IDENTITY: {pool} · base {} B · levels {} and {} B equal",
            base.len(),
            levels[0].len(),
            levels[1].len(),
        );
    }
}

/// ★ Level 1 filled on its own thread beside the stacks' commit
/// (`LAMBDA_VM_LEVELS_FILL_BESIDE`) is the same fill and the same proofs: a
/// small BLAKE3 block driven with the knob on proves each level's bytes as
/// [`prove_levels`] does over the same base, and the driver reports the fill
/// as done beside. Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an untimed identity
/// check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind"]
fn the_blake3_driver_with_level_1_filled_beside_proves_the_same_level_bytes() {
    use super::levels::{prove_block_levels, set_fill_beside_for_test};
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_fill_beside_for_test(None);
        }
    }
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    let run = {
        let _restore = Restore;
        set_fill_beside_for_test(Some(true));
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
    let (fill, done) = run
        .times
        .fill_beside
        .expect("level 1 was filled beside the stacks' commit");
    assert!(fill > 0.0 && done > 0.0);
    let held = derive_levels(&run.plan, WlfmFamily::B3, false).expect("the levels derive");
    let (proofs, _) = prove_levels(&run.plan, &held, &run.base_proof).expect("the levels prove");
    assert_eq!(proofs.len(), run.proofs.len());
    for (k, (a, b)) in run.proofs.iter().zip(&proofs).enumerate() {
        let a = rkyv::to_bytes::<rkyv::rancor::Error>(a).expect("serializes");
        let b = rkyv::to_bytes::<rkyv::rancor::Error>(b).expect("serializes");
        assert!(a[..] == b[..], "level {}: the same proof bytes", k + 1);
        println!(
            "LEVELS FILL BESIDE IDENTITY level {}: {} B equal",
            k + 1,
            a.len()
        );
    }
}

/// The BLAKE3 driver with level 2 executed beside level 1's prove
/// (`LAMBDA_VM_LEVELS_STREAM_NEXT`, its arena landing piece by piece as level 1
/// opens its groups) proves the same level bytes as [`prove_levels`] does over
/// the same base, and the driver reports every piece landed. Needs
/// `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind"]
fn the_blake3_driver_with_level_2_streamed_proves_the_same_level_bytes() {
    use super::levels::{prove_block_levels, set_stream_next_for_test};
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_stream_next_for_test(None);
        }
    }
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    let run = {
        let _restore = Restore;
        set_stream_next_for_test(Some(true));
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
    assert!(run.proofs.len() >= 2, "a level 2 to stream");
    let stream = run
        .times
        .next_stream
        .expect("level 2 was executed beside level 1's prove");
    assert!(stream.pieces >= 2 && stream.first <= stream.last && stream.last <= stream.executed);
    let held = derive_levels(&run.plan, WlfmFamily::B3, false).expect("the levels derive");
    let (proofs, _) = prove_levels(&run.plan, &held, &run.base_proof).expect("the levels prove");
    assert_eq!(proofs.len(), run.proofs.len());
    for (k, (a, b)) in run.proofs.iter().zip(&proofs).enumerate() {
        let a = rkyv::to_bytes::<rkyv::rancor::Error>(a).expect("serializes");
        let b = rkyv::to_bytes::<rkyv::rancor::Error>(b).expect("serializes");
        assert!(a[..] == b[..], "level {}: the same proof bytes", k + 1);
        println!(
            "LEVELS STREAM NEXT IDENTITY level {}: {} B equal",
            k + 1,
            a.len()
        );
    }
}

/// The levels' chain steps a driver test forces. Unselected steps are
/// disabled to isolate the step under test.
#[derive(Clone, Copy, Debug, Default)]
struct ChainSteps {
    /// `LAMBDA_VM_LEVELS_FILL_PARTS`.
    fill_parts: bool,
    /// `LAMBDA_VM_LEVELS_COMMIT_NEXT_EARLY`.
    commit_early: bool,
    /// `LAMBDA_VM_LEVELS_COLUMNS_TILED`.
    tiled: bool,
    /// `LAMBDA_VM_LEVELS_PACKS_BESIDE`.
    packs: bool,
    /// `LAMBDA_VM_LEVELS_NEXT_BESIDE`.
    next_beside: bool,
    /// `LAMBDA_VM_LEVELS_WAVES_IN_ORDER`.
    waves_in_order: bool,
    /// `LAMBDA_VM_LEVELS_PLAN_LEAN`.
    plan_lean: bool,
    /// `LAMBDA_VM_LEVELS_STATEMENT_EARLY`.
    statement_early: bool,
}

/// The levels' chain steps forced as given (the first two on this thread,
/// the prepare's two for the process), the small BLAKE3 block driven under
/// them, and each level's proof bytes against [`prove_levels`] over the same
/// base with every step off (today's derivation). The run, for its stamps.
fn driven_under_chain_steps(steps: ChainSteps) -> super::levels::LevelsRun {
    use super::levels::{
        prove_block_levels, set_commit_next_early_for_test, set_fill_parts_for_test,
        set_next_beside_for_test, set_packs_beside_for_test, set_plan_lean_for_test,
        set_statement_early_for_test, set_waves_in_order_for_test,
    };
    use super::wlfm::set_columns_tiled_for_test;
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_fill_parts_for_test(None);
            set_commit_next_early_for_test(None);
            set_columns_tiled_for_test(None);
            set_packs_beside_for_test(None);
            set_next_beside_for_test(None);
            set_waves_in_order_for_test(None);
            set_plan_lean_for_test(None);
            set_statement_early_for_test(None);
        }
    }
    let ChainSteps {
        fill_parts,
        commit_early,
        tiled,
        packs,
        next_beside,
        waves_in_order,
        plan_lean,
        statement_early,
    } = steps;

    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    // The plan's measuring probes, counted over the driver's run alone (the
    // reference plan below measures). Run alone: the count is the process's.
    let probes = super::whir_block::plan_probes_for_test();
    let run = {
        let _restore = Restore;
        set_columns_tiled_for_test(Some(tiled));
        set_packs_beside_for_test(Some(packs));
        set_fill_parts_for_test(Some(fill_parts));
        set_commit_next_early_for_test(Some(commit_early));
        set_next_beside_for_test(Some(next_beside));
        set_waves_in_order_for_test(Some(waves_in_order));
        set_plan_lean_for_test(Some(plan_lean));
        set_statement_early_for_test(Some(statement_early));
        prove_block_levels(
            &elf,
            &[],
            &opts,
            &format,
            &if statement_early {
                pipelined_options()
            } else {
                streamed_options()
            },
            WlfmFamily::B3,
            LevelStop::DEFAULT,
            true,
            true,
        )
        .expect("the BLAKE3 block proves to its last level")
    };
    let probes = super::whir_block::plan_probes_for_test() - probes;
    // Check both speculative paths against a fresh plan from the final base
    // statement, with its prepared roots derived on the host.
    let measured = (plan_lean || statement_early).then(|| {
        super::levels::level_plan_measured(
            &elf,
            &opts,
            &format,
            run.base_proof.statement(),
            WlfmFamily::B3,
            None,
            true,
        )
        .expect("the measured plan derives")
    });
    if plan_lean {
        assert!(
            run.plan.chip_loads().0.is_empty(),
            "the driver's plan measured nothing"
        );
        assert_eq!(probes, 0, "no probe ran in the driver's plan");
    } else {
        assert!(probes > 0, "the driver's plan measured");
    }
    match (statement_early, run.times.early) {
        (false, early) => assert!(early.is_none(), "off: no early statement came"),
        (true, Some(note)) => {
            println!(
                "LEVELS EARLY STATEMENT NOTE: {note:?} · level 1 emitted at {:.3}",
                run.times.emitted
            );
            // Nothing of level 1 published before the final statement came.
            assert!(
                run.times.emitted >= note.confirmed_at,
                "level 1's program published before the statement was confirmed: {note:?}"
            );
        }
        (true, None) => panic!("on: no early statement came"),
    }
    let reference = measured.as_ref().unwrap_or(&run.plan);
    let held = derive_levels(reference, WlfmFamily::B3, false).expect("the levels derive");
    let (proofs, _) = prove_levels(reference, &held, &run.base_proof).expect("the levels prove");
    assert_eq!(proofs.len(), run.proofs.len());
    for (k, (a, b)) in run.proofs.iter().zip(&proofs).enumerate() {
        let a = rkyv::to_bytes::<rkyv::rancor::Error>(a).expect("serializes");
        let b = rkyv::to_bytes::<rkyv::rancor::Error>(b).expect("serializes");
        assert!(a[..] == b[..], "level {}: the same proof bytes", k + 1);
        println!(
            "LEVELS CHAIN STEPS ({steps:?}) IDENTITY level {}: {} B equal",
            k + 1,
            a.len()
        );
    }
    run
}

/// ★ Each level's fill with its three parts at once
/// (`LAMBDA_VM_LEVELS_FILL_PARTS`) is the same fill: a small BLAKE3 block
/// driven with the knob on proves each level's bytes as [`prove_levels`] does
/// over the same base. Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an untimed
/// identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind"]
fn the_blake3_driver_with_its_fill_parts_at_once_proves_the_same_level_bytes() {
    let run = driven_under_chain_steps(ChainSteps {
        fill_parts: true,
        ..ChainSteps::default()
    });
    assert!(run.proofs.len() >= 2, "level 2 filled too");
}

/// ★ Level 2's stacks committed as level 1's prove returns
/// (`LAMBDA_VM_LEVELS_COMMIT_NEXT_EARLY`) are the same commit: a small BLAKE3
/// block driven with the knob on proves each level's bytes as [`prove_levels`]
/// does, and the driver started level 2's commit before it joined level 2's
/// streamed execution; with the knob off, after. Needs
/// `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels three times; box tier, deterministic grind"]
fn the_blake3_driver_committing_level_2_early_proves_the_same_level_bytes() {
    let run = driven_under_chain_steps(ChainSteps {
        commit_early: true,
        ..ChainSteps::default()
    });
    assert!(run.proofs.len() >= 2, "a level 2 to commit");
    assert!(run.times.next_stream.is_some(), "level 2 streamed");
    let early = run.times.next_commit.expect("level 2 committed");
    let joined = early.joined.expect("level 2's stream joined");
    assert!(early.early, "committed early: {early:?}");
    assert!(early.from <= joined, "committed before the join: {early:?}");
    let off = driven_under_chain_steps(ChainSteps::default());
    let late = off.times.next_commit.expect("level 2 committed");
    let joined = late.joined.expect("level 2's stream joined");
    assert!(
        !late.early && late.from >= joined,
        "off: after the join: {late:?}"
    );
}

/// ★ Every level's columns made by tiles, its chips' chunks read in place
/// (`LAMBDA_VM_LEVELS_COLUMNS_TILED`), are the same stacks: a small BLAKE3
/// block driven with the knob on proves each level's bytes as
/// [`prove_levels`] does over the same base. Needs
/// `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind"]
fn the_blake3_driver_with_its_columns_tiled_proves_the_same_level_bytes() {
    let run = driven_under_chain_steps(ChainSteps {
        tiled: true,
        ..ChainSteps::default()
    });
    assert!(run.proofs.len() >= 2, "level 2 prepared too");
}

/// ★ Every level's chip prefixes and narrow packs made at once
/// (`LAMBDA_VM_LEVELS_PACKS_BESIDE`) are the same packs: a small BLAKE3 block
/// driven with the knob on proves each level's bytes as [`prove_levels`]
/// does over the same base. Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an
/// untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind"]
fn the_blake3_driver_with_its_packs_at_once_proves_the_same_level_bytes() {
    let run = driven_under_chain_steps(ChainSteps {
        packs: true,
        ..ChainSteps::default()
    });
    assert!(run.proofs.len() >= 2, "level 2 prepared too");
}

/// ★ Level 1's validator with its checks at once
/// (`LAMBDA_VM_LEVELS_VALIDATE_AT_ONCE`) and its program let go beside the
/// prove (`LAMBDA_VM_LEVELS_FREE_BESIDE`): a small BLAKE3 block driven with
/// both on proves each level's bytes as [`prove_levels`] does over the same
/// base. Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind"]
fn the_blake3_driver_validating_at_once_and_freeing_beside_proves_the_same_level_bytes() {
    use super::levels::{set_free_beside_for_test, set_validate_at_once_for_test};
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_validate_at_once_for_test(None);
            set_free_beside_for_test(None);
        }
    }
    let _restore = Restore;
    set_validate_at_once_for_test(Some(true));
    set_free_beside_for_test(Some(true));
    let run = driven_under_chain_steps(ChainSteps::default());
    assert!(run.times.release.1, "level 1's program freed beside");
    assert!(run.proofs.len() >= 2, "level 2 too");
}

/// ★ Level 1's validator joined after its fill
/// (`LAMBDA_VM_LEVELS_VALIDATE_AFTER_FILL`) and its builder on the whole pool
/// once the base returned (`LAMBDA_VM_LEVELS_BUILDER_WHOLE_POOL`): a small
/// BLAKE3 block driven with both on proves each level's bytes as
/// [`prove_levels`] does over the same base, the join after the fill and the
/// builder's watch both stamped. Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an
/// untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind"]
fn the_blake3_driver_validating_after_the_fill_with_its_builder_on_the_whole_pool_proves_the_same_level_bytes()
 {
    use super::levels::set_validate_after_fill_for_test;
    use super::wlfm::set_builder_whole_pool_for_test;
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_validate_after_fill_for_test(None);
            set_builder_whole_pool_for_test(None);
        }
    }
    let _restore = Restore;
    set_validate_after_fill_for_test(Some(true));
    set_builder_whole_pool_for_test(Some(true));
    let run = driven_under_chain_steps(ChainSteps::default());
    assert!(
        run.times.validated_after_fill.is_some(),
        "level 1's validator joined after its fill"
    );
    let note = run
        .times
        .builder_whole_pool
        .expect("the builder watched the base");
    assert!(note.tables > 0 && note.tables_before <= note.tables);
    println!("LEVELS BUILDER POOL (identity run): {note:?}");
    assert!(run.proofs.len() >= 2, "level 2 too");
}

/// ★ Level 1's program made inadmissible (a constant's count raised): the
/// driver refuses it, with its validator joined before the fill (today's) and
/// after it (`LAMBDA_VM_LEVELS_VALIDATE_AFTER_FILL`): an error naming the
/// admission, never a level proof, never a panic.
#[test]
#[ignore = "proves a small BLAKE3 block twice up to level 1's admission; box tier"]
fn an_inadmissible_level_1_is_refused_with_its_validator_before_or_after_the_fill() {
    use super::levels::{
        prove_block_levels, set_level1_program_fault_for_test, set_validate_after_fill_for_test,
    };
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_level1_program_fault_for_test(false);
            set_validate_after_fill_for_test(None);
        }
    }
    let _restore = Restore;
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    set_level1_program_fault_for_test(true);
    for after in [false, true] {
        set_validate_after_fill_for_test(Some(after));
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
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
        }));
        match run {
            Ok(Err(e)) => {
                assert!(e.contains("not admissible"), "after the fill {after}: {e}");
                println!("LEVELS INADMISSIBLE (after the fill {after}): refused · {e}");
            }
            Ok(Ok(_)) => panic!("after the fill {after}: an inadmissible level 1 proved"),
            Err(_) => panic!("after the fill {after}: the driver panicked"),
        }
    }
}

/// The chain steps used to overlap level preparation: the same level bytes.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind"]
fn the_blake3_driver_under_every_chain_step_proves_the_same_level_bytes() {
    let run = driven_under_chain_steps(ChainSteps {
        fill_parts: true,
        commit_early: true,
        tiled: true,
        packs: true,
        next_beside: true,
        waves_in_order: true,
        plan_lean: false,
        statement_early: false,
    });
    assert!(run.times.next_commit.is_some_and(|c| c.early));
    assert!(run.times.next_stream.is_some_and(|s| s.prebuilt.is_some()));
}

/// ★ Level 1's plan without its measuring probes
/// (`LAMBDA_VM_LEVELS_PLAN_LEAN`) makes the same program: a small BLAKE3
/// block driven with the knob on proves each level's bytes as
/// [`prove_levels`] does over a plan that measured, its plan's chip rows
/// empty and its probe-call count zero. Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1`;
/// an untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind"]
fn the_blake3_driver_with_its_plan_lean_proves_the_same_level_bytes() {
    let _options = super::wlfm::test_format::set_driver_options();
    let run = driven_under_chain_steps(ChainSteps {
        plan_lean: true,
        ..ChainSteps::default()
    });
    assert!(run.proofs.len() >= 2, "level 2 over the lean level 1");
}

/// ★ Level 1's program made from the base's early statement
/// (`LAMBDA_VM_LEVELS_STATEMENT_EARLY`), confirmed against the final one, is
/// the same program: a small BLAKE3 block driven with the knob on proves each
/// level's bytes as [`prove_levels`] does, the early program was used, and
/// nothing of level 1 was published before the final statement came. With
/// the plan lean too, the same. Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an
/// untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels four times; box tier, deterministic grind"]
fn the_blake3_driver_with_its_statement_early_proves_the_same_level_bytes() {
    for plan_lean in [false, true] {
        let before = super::levels::early_mismatches();
        let run = driven_under_chain_steps(ChainSteps {
            statement_early: true,
            plan_lean,
            ..ChainSteps::default()
        });
        let note = run.times.early.expect("an early statement came");
        assert!(
            note.used,
            "plan lean {plan_lean}: the early program was used"
        );
        assert_eq!(super::levels::early_mismatches(), before, "no mismatch");
        assert!(run.proofs.len() >= 2, "level 2 over level 1");
    }
}

/// An early fault falls back to the proof bytes derived from the final
/// statement and its independently committed roots.
fn assert_early_fault_fallback(fault: crate::block_whir::EarlyFault) {
    use crate::block_whir::{EarlyFault, set_early_fault_for_test};
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_early_fault_for_test(EarlyFault::None);
        }
    }
    let _restore = Restore;
    let _options = super::wlfm::test_format::set_driver_options();
    set_early_fault_for_test(fault);
    let before_level = super::levels::early_mismatches();
    let before_base = crate::block_whir::early_statement_mismatches();
    let run = driven_under_chain_steps(ChainSteps {
        statement_early: true,
        ..ChainSteps::default()
    });
    let note = run.times.early.expect("an early statement came");
    assert!(!note.used, "{fault:?}: the early program was not used");
    assert_eq!(super::levels::early_mismatches(), before_level + 1);
    assert_eq!(
        crate::block_whir::early_statement_mismatches(),
        before_base + 1
    );
}

#[test]
#[ignore = "box tier, run alone with deterministic grind"]
fn early_fault_doubled_rows_falls_back_to_final_bytes() {
    assert_early_fault_fallback(crate::block_whir::EarlyFault::DoubledRows);
}

#[test]
#[ignore = "box tier, run alone with deterministic grind"]
fn early_fault_short_replay_falls_back_to_final_bytes() {
    assert_early_fault_fallback(crate::block_whir::EarlyFault::ShortReplay);
}

#[test]
#[ignore = "box tier, run alone with deterministic grind"]
fn early_fault_root_word_falls_back_to_final_bytes() {
    assert_early_fault_fallback(crate::block_whir::EarlyFault::RootWord);
}

#[test]
#[ignore = "box tier, run alone with deterministic grind"]
fn early_fault_public_output_falls_back_to_final_bytes() {
    assert_early_fault_fallback(crate::block_whir::EarlyFault::PublicOutput);
}

/// ★ The base's early statement is its final one: small blocks' streamed
/// BLAKE3 bases, with and without local lookups, proved with an early
/// observer through the entry the levels driver takes for the format hand it
/// the statement and the prepared roots the final observer gets, field for
/// field, before the final one comes; every table's rows were told as stated
/// (no miss) and the block counted no mismatch.
#[test]
#[ignore = "proves three small BLAKE3 blocks' bases in two formats; box tier"]
fn the_early_statement_is_the_final_one() {
    type Seen =
        std::sync::Mutex<Vec<(&'static str, String, Vec<crate::block_whir::PreparedRoots>)>>;
    let local = BlockFormat {
        argue: multilinear::whir_chain::ArgueFormat::BATCHED,
        keccak_rnd: crate::tables::keccak_rnd::KeccakRndForm::LeanLocal,
        ..b3_format()
    };
    assert!(local.local_lookups() && !b3_format().local_lookups());
    let opts = ProofOptions::default_test_options();
    let misses = crate::tables::trace_builder::rows_misses();
    let mismatches = crate::block_whir::early_statement_mismatches();
    for (format, name) in [b3_format(), local].into_iter().flat_map(|format| {
        ["test_commit_4", "all_instructions_64", "test_keccak_multi"]
            .map(move |name| (format, name))
    }) {
        let kind = if format.local_lookups() {
            "local"
        } else {
            "global"
        };
        let elf = crate::test_utils::asm_elf_bytes(name);
        let seen: Seen = std::sync::Mutex::new(Vec::new());
        let note = |kind: &'static str| {
            let seen = &seen;
            move |s: crate::block_whir::BlockStatement<'_>,
                  roots: &[crate::block_whir::PreparedRoots]| {
                seen.lock().unwrap_or_else(|e| e.into_inner()).push((
                    kind,
                    format!("{:?}", s.to_owned()),
                    roots.to_vec(),
                ));
            }
        };
        let (fin, early) = (note("final"), note("early"));
        // The entry the levels driver takes for the format: a format that
        // argues local lookups proves a proof of its own kind.
        crate::base_only::with_blake3_levels(|| {
            if format.local_lookups() {
                crate::block_whir::prove_block_whir_local_observed_early(
                    &elf,
                    &[],
                    &opts,
                    &format,
                    &pipelined_options(),
                    &fin,
                    Some(&early),
                    &|_| {},
                    None,
                )
                .map(|_| ())
            } else {
                crate::block_whir::prove_block_whir_observed_early(
                    &elf,
                    &[],
                    &opts,
                    &format,
                    &pipelined_options(),
                    &fin,
                    Some(&early),
                    &|_| {},
                )
                .map(|_| ())
            }
        })
        .expect("the BLAKE3 base proves");
        let seen = seen.into_inner().unwrap_or_else(|e| e.into_inner());
        let kinds: Vec<&str> = seen.iter().map(|s| s.0).collect();
        assert_eq!(
            kinds,
            ["early", "final"],
            "{name} ({kind}): early first, then final"
        );
        assert_eq!(
            seen[0].1, seen[1].1,
            "{name} ({kind}): the statement, field for field"
        );
        assert_eq!(seen[0].2, seen[1].2, "{name} ({kind}): the prepared roots");
        println!("EARLY STATEMENT {name} ({kind}): confirmed");
    }
    assert_eq!(
        crate::tables::trace_builder::rows_misses(),
        misses,
        "a table built at other rows than stated"
    );
    assert_eq!(
        crate::block_whir::early_statement_mismatches(),
        mismatches,
        "the block counted a mismatch"
    );
}

/// ★ The finish starting its late families' tables first
/// (`LAMBDA_VM_FINISH_LATE_FIRST`) proves the same base: small blocks proved
/// streamed and pipelined with the knob on and off, each proof's bytes equal.
/// Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an untimed identity check.
#[test]
#[ignore = "proves two small blocks' bases twice; box tier, deterministic grind"]
fn the_finish_starting_its_late_tables_first_proves_the_same_base_bytes() {
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    let format = at_blowup_4(small_format());
    let opts = ProofOptions::default_test_options();
    for name in ["test_commit_4", "all_instructions_64"] {
        let elf = crate::test_utils::asm_elf_bytes(name);
        let prove = |late: bool| {
            let deviations = crate::block_whir::Deviations {
                finish_late_first: Some(late),
                ..Default::default()
            };
            let (proof, _) = crate::block_whir::prove_block_whir_observed_with(
                &elf,
                &[],
                &opts,
                &format,
                &pipelined_options(),
                &deviations,
                &|_, _| {},
            )
            .expect("the base proves");
            rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
                .expect("serializes")
                .to_vec()
        };
        let (off, on) = (prove(false), prove(true));
        assert!(
            off == on,
            "{name}: the late families started first moved a byte"
        );
        println!("FINISH LATE FIRST IDENTITY {name}: {} B equal", on.len());
    }
}

/// ★ A level plan derived without measuring emits the measured plan's
/// program: the same instructions, addresses, arena schema, column groups and
/// leaf marks, the same costs and partition, over two small BLAKE3 blocks;
/// only its chip rows are empty. A prover's plan with prepared roots is lean
/// by default. A verifier's plan always measures.
#[test]
#[ignore = "proves two small BLAKE3 blocks' bases; box tier"]
fn a_lean_level_plan_emits_the_measured_plans_program() {
    use super::levels::{
        level_plan_measured, set_plan_lean_for_test, verifier_level_plan_for_test,
    };
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_plan_lean_for_test(None);
        }
    }
    let _restore = Restore;
    let digest = |text: String| -> String {
        use digest::Digest;
        crypto::hash::platform_keccak::PlatformKeccak256::digest(text.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    };
    let format = b3_format();
    let opts = ProofOptions::default_test_options();
    for name in ["test_commit_4", "all_instructions_64"] {
        let (elf, proof) = b3_small_block(name, &format);
        let plan = |measure: bool| {
            level_plan_measured(
                &elf,
                &opts,
                &format,
                proof.statement(),
                WlfmFamily::B3,
                None,
                measure,
            )
            .expect("the plan derives")
        };
        let (lean, measured) = (plan(false), plan(true));
        assert!(
            lean.chip_loads().0.is_empty(),
            "{name}: a lean plan measures nothing"
        );
        assert_eq!(
            measured.chip_loads().0.len(),
            proof.statement().groups.len(),
            "{name}: a measured plan, every group"
        );
        assert_eq!(lean.costs(), measured.costs(), "{name}: the costs");
        assert_eq!(
            format!("{:?}", lean.partition()),
            format!("{:?}", measured.partition()),
            "{name}: the partition"
        );
        let (a, a_marks) = lean
            .closed_block_program_marked()
            .expect("the lean program");
        let (b, b_marks) = measured
            .closed_block_program_marked()
            .expect("the measured program");
        assert_eq!(a.instrs.len(), b.instrs.len(), "{name}: the instructions");
        assert_eq!(a.num_addrs, b.num_addrs, "{name}: the addresses");
        assert_eq!(a_marks, b_marks, "{name}: the leaf marks");
        assert_eq!(
            digest(format!("{:?}", a.arena_schema)),
            digest(format!("{:?}", b.arena_schema)),
            "{name}: the arena schema"
        );
        assert_eq!(
            digest(format!("{a:?}")),
            digest(format!("{b:?}")),
            "{name}: the program, instruction for instruction"
        );
        // The knob on: a prover's plan (its roots in) is lean; a plan without
        // roots and a verifier's measure.
        set_plan_lean_for_test(Some(true));
        let verifier = verifier_level_plan_for_test(&elf, &opts, &format, proof.statement())
            .expect("the verifier's plan");
        assert_eq!(
            verifier.chip_loads().0.len(),
            proof.statement().groups.len(),
            "{name}: a verifier measures with the knob on"
        );
        let rootless = level_plan(
            &elf,
            &opts,
            &format,
            proof.statement(),
            WlfmFamily::B3,
            None,
        )
        .expect("the plan derives");
        assert!(
            !rootless.chip_loads().0.is_empty(),
            "{name}: a plan without the prover's roots measures"
        );
        set_plan_lean_for_test(None);
    }
}

/// ★ Level 1's and level 2's waves run in program order on their own thread
/// (`LAMBDA_VM_LEVELS_WAVES_IN_ORDER`) are the same executions: a small
/// BLAKE3 block driven with the knob on proves each level's bytes as
/// [`prove_levels`] does over the same base. Needs
/// `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind"]
fn the_blake3_driver_with_its_waves_in_order_proves_the_same_level_bytes() {
    let run = driven_under_chain_steps(ChainSteps {
        waves_in_order: true,
        ..ChainSteps::default()
    });
    assert!(run.proofs.len() >= 2, "level 2 streamed in order too");
}

/// The small BLAKE3 block driven with the levels' two-card split at `setting`
/// (`LAMBDA_VM_TWO_CARD_LEVELS`; the base's split off, so the base proves on
/// one card), each level's proof bytes against [`prove_levels`] over the same
/// base (one card, its stacks held, never split), and the host paths the run
/// took. The level groups the second driver proved.
#[cfg(feature = "cuda")]
fn driven_under_levels_split(setting: math_cuda::device::TwoCardSplit) -> u64 {
    driven_under_levels_split_stacks(setting, false)
}

/// [`driven_under_levels_split`], with `by_card` every level's stacks held
/// and each committed on its group's card by default
/// (`LAMBDA_VM_LEVELS_STACKS_BY_CARD`). The retired control turns the knob off.
/// The second card's ledger is empty again at the end.
#[cfg(feature = "cuda")]
fn driven_under_levels_split_stacks(
    setting: math_cuda::device::TwoCardSplit,
    by_card: bool,
) -> u64 {
    use super::levels::{prove_block_levels, set_stacks_by_card_for_test};
    use math_cuda::device::{
        TwoCardSplit, set_two_card_levels_for_test, set_two_card_split_for_test,
    };
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_two_card_levels_for_test(None);
            set_two_card_split_for_test(None);
            set_stacks_by_card_for_test(None);
        }
    }
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    let _restore = Restore;
    // With the test query counts, a production stack can fit the whole
    // level in one group. Use a smaller cap for the smoke test's two drivers.
    #[cfg(feature = "test-security")]
    let _stack = super::wlfm::test_format::use_small_card_stack();
    let guest = if cfg!(feature = "test-security") {
        "basic_program"
    } else {
        "test_commit_4"
    };
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes(guest);
    let opts = ProofOptions::default_test_options();
    set_two_card_split_for_test(Some(TwoCardSplit::Off));
    set_two_card_levels_for_test(Some(setting));
    assert!(
        multilinear::gpu::two_card_levels_on(),
        "the levels' split is on"
    );
    assert!(
        !multilinear::gpu::two_card_split_on(),
        "the base's split is off"
    );
    assert!(
        multilinear::gpu::two_card_split_ready(),
        "the levels' second driver is ready"
    );
    let declines = multilinear::gpu::cross_card_declines();
    let before = stark::multilinear_block::split_second_groups();
    // Every level's stacks retired: a held stack pins its group to card 0, and
    // a small block's levels would hold theirs (the same bytes either way).
    // By card: held, each on its group's card.
    set_stacks_by_card_for_test(if by_card { None } else { Some(false) });
    assert_eq!(super::levels::stacks_by_card(), by_card);
    let run = {
        let _retired = super::levels::set_stacks_for_test(Some(if by_card {
            super::wlfm::StackPlace::Card
        } else {
            super::wlfm::StackPlace::CardRetired
        }));
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
        .expect("the BLAKE3 block proves to its last level with the levels split")
    };
    let second = stark::multilinear_block::split_second_groups() - before;
    let declined = multilinear::gpu::cross_card_declines() - declines;
    if by_card {
        let ledger = multilinear::gpu::in_second_role(multilinear::gpu::ledger_reserved);
        println!(
            "LEVELS STACKS BY CARD: the second card's ledger after the levels {:.2} GiB",
            ledger as f64 / (1u64 << 30) as f64
        );
        assert_eq!(ledger, 0, "the second card's stacks and promises let go");
    }
    set_stacks_by_card_for_test(Some(false));
    set_two_card_levels_for_test(Some(TwoCardSplit::Off));
    let by = if by_card { ", stacks by card" } else { "" };
    let held = {
        let _held = super::levels::set_stacks_for_test(Some(super::wlfm::StackPlace::Card));
        derive_levels(&run.plan, WlfmFamily::B3, false).expect("the levels derive")
    };
    let (proofs, _) = prove_levels(&run.plan, &held, &run.base_proof).expect("the levels prove");
    assert_eq!(run.proofs.len(), 2, "the base feeds both recursion levels");
    assert_eq!(proofs.len(), run.proofs.len());
    verify_levels(
        &run.plan,
        WlfmFamily::B3,
        run.proofs.last().expect("the last level"),
    )
    .expect("the split proof verifies and claims the block");
    for (k, (a, b)) in run.proofs.iter().zip(&proofs).enumerate() {
        let a = rkyv::to_bytes::<rkyv::rancor::Error>(a).expect("serializes");
        let b = rkyv::to_bytes::<rkyv::rancor::Error>(b).expect("serializes");
        assert!(a[..] == b[..], "level {}: the same proof bytes", k + 1);
        println!(
            "LEVELS TWO-CARD ({setting:?}{by}) IDENTITY level {}: {} B equal",
            k + 1,
            a.len()
        );
    }
    println!(
        "LEVELS TWO-CARD ({setting:?}{by}): {second} level group(s) proved by the second driver · \
         cross-card declines {declined}"
    );
    assert_eq!(declined, 0, "a store was read from the other card");
    second
}

/// ★ Card, run alone, deterministic grind. The levels' phase B split over two
/// drivers forced on one card (a second backend of card 0 drives the second
/// share: its own streams, staging, ledger and pools) proves each level's
/// bytes as one card does, and the second driver proved level groups.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: proves a small BLAKE3 block and its levels twice; run alone, deterministic grind"]
fn the_blake3_driver_with_its_levels_split_forced_on_one_card_proves_the_same_level_bytes() {
    let second = driven_under_levels_split(math_cuda::device::TwoCardSplit::Forced);
    assert!(second > 0, "no level group proved by the second driver");
}

/// ★ Two cards, run alone, deterministic grind. With two cards seen and the
/// levels' knob at `auto`, the levels' second driver runs on card 1 and each
/// level proves the bytes of one card.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "two cards: proves a small BLAKE3 block and its levels twice; run alone, deterministic grind"]
fn the_blake3_driver_with_its_levels_split_on_card_1_proves_the_same_level_bytes() {
    use math_cuda::device::{TwoCardSplit, card_count, split_placement};
    assert!(card_count() >= 2, "needs two cards, saw {}", card_count());
    let second = {
        math_cuda::device::set_two_card_levels_for_test(Some(TwoCardSplit::Auto));
        assert_eq!(split_placement().ordinal, 1, "the second driver on card 1");
        driven_under_levels_split(TwoCardSplit::Auto)
    };
    assert!(second > 0, "no level group proved on card 1");
}

/// ★ Card, run alone, deterministic grind. Every level's stacks held and
/// each committed on its group's card (`LAMBDA_VM_LEVELS_STACKS_BY_CARD`),
/// the levels' split forced on one card (a second backend of card 0): each
/// level proves the bytes of one card with its stacks held on card 0, the
/// second driver proved level groups, no store was read across, and the
/// second backend's ledger is empty after the levels.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: proves a small BLAKE3 block and its levels twice; run alone, deterministic grind"]
fn the_blake3_driver_with_its_stacks_by_card_forced_on_one_card_proves_the_same_level_bytes() {
    let second = driven_under_levels_split_stacks(math_cuda::device::TwoCardSplit::Forced, true);
    assert!(second > 0, "no level group proved by the second driver");
}

/// ★ Two cards, run alone, deterministic grind. The same with the levels'
/// second driver on card 1: held stacks committed there, their groups proved
/// there, each level the bytes of one card.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "two cards: proves a small BLAKE3 block and its levels twice; run alone, deterministic grind"]
fn the_blake3_driver_with_its_stacks_by_card_on_card_1_proves_the_same_level_bytes() {
    use math_cuda::device::{TwoCardSplit, card_count, split_placement};
    assert!(card_count() >= 2, "needs two cards, saw {}", card_count());
    math_cuda::device::set_two_card_levels_for_test(Some(TwoCardSplit::Auto));
    assert_eq!(split_placement().ordinal, 1, "the second driver on card 1");
    let second = driven_under_levels_split_stacks(TwoCardSplit::Auto, true);
    assert!(second > 0, "no level group proved on card 1");
}

/// The three level defaults preserve the base and both recursion proofs
/// when compared with the opt-outs.
/// Forced split exercises both CUDA backends on a single physical card.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: run alone with deterministic grind and fixed trace hash"]
fn the_blake3_driver_with_all_three_early_knobs_preserves_the_chain() {
    use super::levels::{
        prove_block_levels, set_plan_lean_for_test, set_stacks_by_card_for_test,
        set_stacks_for_test, set_statement_early_for_test, verify_block_levels_under,
    };
    use math_cuda::device::{
        TwoCardSplit, set_two_card_levels_for_test, set_two_card_split_for_test,
    };

    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_statement_early_for_test(None);
            set_plan_lean_for_test(None);
            set_stacks_by_card_for_test(None);
            set_two_card_levels_for_test(None);
            set_two_card_split_for_test(None);
        }
    }

    let _restore = Restore;
    let _options = super::wlfm::test_format::set_driver_options();
    let _stacks = set_stacks_for_test(Some(StackPlace::Card));
    assert!(crypto::grinding::deterministic());
    assert_eq!(
        std::env::var("LAMBDA_VM_FIXED_TRACE_HASH").as_deref(),
        Ok("1")
    );
    set_two_card_split_for_test(Some(TwoCardSplit::Off));
    set_two_card_levels_for_test(Some(TwoCardSplit::Forced));
    assert!(multilinear::gpu::two_card_split_ready());
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    let options = pipelined_options();
    let prove = |setting: Option<bool>| {
        set_statement_early_for_test(setting);
        set_plan_lean_for_test(setting);
        set_stacks_by_card_for_test(setting);
        let on = setting.is_none();
        let before = super::whir_block::plan_probes_for_test();
        let run = prove_block_levels(
            &elf,
            &[],
            &opts,
            &format,
            &options,
            WlfmFamily::B3,
            LevelStop::DEFAULT,
            true,
            true,
        )
        .expect("the small block proves through level 2");
        assert_eq!(run.proofs.len(), 2);
        assert_eq!(run.times.early.map(|note| note.used), on.then_some(true));
        assert_eq!(super::whir_block::plan_probes_for_test() == before, on);
        let second_holds_a_stack = run.levels[0]
            .build
            .artifacts
            .stacks
            .iter()
            .any(|stack| run.times.stamps[0].groups[stack.group].card == 1);
        assert_eq!(
            second_holds_a_stack, on,
            "level 1 prepared stacks on card 1"
        );
        run
    };
    let default = prove(None);
    let off = prove(Some(false));
    let base_bytes = |run: &super::levels::LevelsRun| {
        rkyv::to_bytes::<rkyv::rancor::Error>(&run.base_proof).expect("serializes")
    };
    let (a, b) = (base_bytes(&off), base_bytes(&default));
    assert!(a[..] == b[..], "the same base proof bytes");
    println!("ALL THREE KNOBS base: {} B equal", a.len());
    for (level, (a, b)) in off.proofs.iter().zip(&default.proofs).enumerate() {
        let a = rkyv::to_bytes::<rkyv::rancor::Error>(a).expect("serializes");
        let b = rkyv::to_bytes::<rkyv::rancor::Error>(b).expect("serializes");
        assert!(a[..] == b[..], "level {} proof bytes", level + 1);
        println!("ALL THREE KNOBS level {}: {} B equal", level + 1, a.len());
    }
    verify_block_levels_under(
        &elf,
        &opts,
        &format,
        default.base_proof.statement(),
        WlfmFamily::B3,
        LevelStop::DEFAULT,
        default.proofs.last().expect("the last level"),
    )
    .expect("the independently derived verifier accepts the final proof");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: run alone with default commit lanes and program pool, deterministic grind and fixed trace hash"]
fn the_blake3_driver_with_stacks_and_commits_by_card_preserves_the_chain() {
    use super::levels::{
        COMMITS_BY_CARD_ENV, PROGRAM_POOL_ENV, PROGRAM_POOL_FOR_TEST, prove_block_levels,
        set_commits_by_card_for_test, set_stacks_by_card_for_test, set_stacks_for_test,
        verify_block_levels_under,
    };
    use math_cuda::device::{
        TwoCardSplit, backend, second_backend_started, set_two_card_levels_for_test,
        set_two_card_split_for_test,
    };
    use stark::multilinear_block::PhaseALanes;

    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            PROGRAM_POOL_FOR_TEST.set(None);
            set_commits_by_card_for_test(None);
            set_stacks_by_card_for_test(None);
            set_two_card_levels_for_test(None);
            set_two_card_split_for_test(None);
        }
    }
    let _restore = Restore;
    let _options = super::wlfm::test_format::set_driver_options();
    let _stacks = set_stacks_for_test(Some(StackPlace::Card));
    assert!(crypto::grinding::deterministic());
    assert_eq!(
        std::env::var("LAMBDA_VM_FIXED_TRACE_HASH").as_deref(),
        Ok("1")
    );
    assert!(std::env::var_os(COMMITS_BY_CARD_ENV).is_none());
    assert!(std::env::var_os(PROGRAM_POOL_ENV).is_none());
    set_two_card_split_for_test(Some(TwoCardSplit::Off));
    set_two_card_levels_for_test(Some(TwoCardSplit::Forced));
    assert!(multilinear::gpu::two_card_split_ready());
    let main = backend().expect("the main CUDA backend");
    let second = second_backend_started().expect("the second CUDA backend");
    let ledger = || (main.reserved_bytes(), second.reserved_bytes());
    let baseline = ledger();
    let declines = multilinear::gpu::cross_card_declines();
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    set_stacks_by_card_for_test(Some(true));
    let prove = |on: bool| {
        PROGRAM_POOL_FOR_TEST.set((!on).then_some(0));
        set_commits_by_card_for_test((!on).then(PhaseALanes::default));
        let run = prove_block_levels(
            &elf,
            &[],
            &opts,
            &format,
            &pipelined_options(),
            WlfmFamily::B3,
            LevelStop::DEFAULT,
            true,
            true,
        )
        .expect("the block proves through level 2");
        assert_eq!(run.proofs.len(), 2);
        assert_eq!(run.times.early.map(|note| note.used), Some(true));
        assert_eq!(
            ledger(),
            baseline,
            "both ledgers released their reservations"
        );
        run
    };
    let off = prove(false);
    let on = prove(true);
    assert_eq!(multilinear::gpu::cross_card_declines(), declines);
    for level in &on.times.stamps {
        assert!(level.groups.iter().all(|g| g.commit_card == g.card));
    }
    assert!(
        on.levels
            .iter()
            .zip(&on.times.stamps)
            .any(|(level, stamps)| {
                level
                    .build
                    .artifacts
                    .stacks
                    .iter()
                    .any(|s| stamps.groups[s.group].commit_card == 1)
            })
    );
    assert!(
        off.times
            .stamps
            .iter()
            .flat_map(|s| &s.groups)
            .all(|g| g.commit_card == 0)
    );

    let base = |run: &super::levels::LevelsRun| {
        rkyv::to_bytes::<rkyv::rancor::Error>(&run.base_proof).expect("serializes")
    };
    let level = |proof: &super::wlfm::WlfmProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(proof).expect("serializes")
    };
    for (part, a, b) in [
        ("base", base(&off), base(&on)),
        ("L1", level(&off.proofs[0]), level(&on.proofs[0])),
        ("L2", level(&off.proofs[1]), level(&on.proofs[1])),
    ] {
        assert!(a[..] == b[..], "{part}: the same proof bytes");
        println!("STACKS AND COMMITS BY CARD {part}: {} B equal", a.len());
    }
    verify_block_levels_under(
        &elf,
        &opts,
        &format,
        on.base_proof.statement(),
        WlfmFamily::B3,
        LevelStop::DEFAULT,
        &on.proofs[1],
    )
    .expect("the independently derived verifier accepts level 2");
}

/// The small BLAKE3 block driven with the levels' split at `setting` and each
/// level's phase A on two commit lanes (`LAMBDA_VM_LEVELS_COMMITS_BY_CARD`, the
/// second lane on the second card), against the same driver with one lane:
/// its stacks retired (no upload beside a commit) and held (the next group
/// uploaded beside each commit, on a committer thread), each lane taking the
/// next group when free and alternating groups. Each level's proof bytes are
/// one lane's. Check the planned groups, fixed lane assignment, second
/// backend's ordinal and zero cross-card store declines. Print differences
/// in the commit paths and return the second lane's group count.
#[cfg(feature = "cuda")]
fn driven_with_commits_by_card(setting: math_cuda::device::TwoCardSplit) -> usize {
    use super::levels::{
        prove_block_levels, set_commits_by_card_for_test, set_stacks_by_card_for_test,
    };
    use math_cuda::device::{
        TwoCardSplit, backend, card_count, set_two_card_levels_for_test,
        set_two_card_split_for_test, split_placement_for,
    };
    use stark::multilinear_block::{GroupStamps, LanePolicy, PhaseALanes, TwoCardKnob};
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_commits_by_card_for_test(None);
            set_stacks_by_card_for_test(None);
            set_two_card_levels_for_test(None);
            set_two_card_split_for_test(None);
        }
    }
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    let _restore = Restore;
    // Use the same compact, multi-group chain in both build postures.
    let _options = super::wlfm::test_format::set_driver_options();
    set_stacks_by_card_for_test(Some(false));
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    set_two_card_split_for_test(Some(TwoCardSplit::Off));
    set_two_card_levels_for_test(Some(setting));
    assert!(
        multilinear::gpu::two_card_levels_on(),
        "the levels' split is on"
    );
    assert!(
        multilinear::gpu::two_card_split_ready(),
        "the levels' second driver is ready"
    );
    let expected = split_placement_for(setting, card_count());
    let ordinal = multilinear::gpu::in_second_role(|| {
        backend().expect("the second driver's backend").ordinal()
    });
    assert_eq!(ordinal, expected.ordinal, "the second driver's card");
    let drive = |place: StackPlace, lanes: Option<PhaseALanes>| {
        set_commits_by_card_for_test(Some(lanes.unwrap_or_default()));
        let _place = super::levels::set_stacks_for_test(Some(place));
        let declines = multilinear::gpu::cross_card_declines();
        let errors = multilinear::gpu::commit_errors();
        let run = prove_block_levels(
            &elf,
            &[],
            &opts,
            &format,
            &pipelined_options(),
            WlfmFamily::B3,
            LevelStop::DEFAULT,
            true,
            true,
        )
        .expect("the BLAKE3 block proves to its last level");
        let declined = multilinear::gpu::cross_card_declines() - declines;
        let refused = multilinear::gpu::commit_errors() - errors;
        assert_eq!(run.proofs.len(), run.levels.len(), "one proof per level");
        assert_eq!(
            run.times.stamps.len(),
            run.levels.len(),
            "one stamp per level"
        );
        for (k, (stamp, level)) in run.times.stamps.iter().zip(&run.levels).enumerate() {
            let planned_groups = &level.build.artifacts.sizes;
            assert!(
                stamp
                    .groups
                    .iter()
                    .map(|g| g.tables)
                    .eq(planned_groups.iter().copied()),
                "level {}: the planned groups in order",
                k + 1
            );
        }
        let proofs: Vec<Vec<u8>> = run
            .proofs
            .iter()
            .map(|p| {
                rkyv::to_bytes::<rkyv::rancor::Error>(p)
                    .expect("serializes")
                    .to_vec()
            })
            .collect();
        let stamps: Vec<Vec<GroupStamps>> =
            run.times.stamps.iter().map(|s| s.groups.clone()).collect();
        (proofs, stamps, declined, refused)
    };
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    // Each lane's ledger budget, for the lines of a group whose path differs.
    let budgets = [
        multilinear::gpu::reserve_budget(),
        multilinear::gpu::in_second_role(multilinear::gpu::reserve_budget),
    ];
    let ledger = |s: &GroupStamps| {
        format!(
            "lane {} ledger {:.2} → peak {:.2} of {:.2} GiB, priced {:.2} GiB, host commits {}, \
             streamed runs {}",
            s.commit_card,
            gib(s.commit_base),
            gib(s.commit_reserved),
            gib(budgets[usize::from(s.commit_card.min(1))]),
            gib(s.commit_priced),
            s.commit_host_fallbacks,
            s.joint_runs[2]
        )
    };
    let mut on_second = 0usize;
    for place in [StackPlace::CardRetired, StackPlace::Card] {
        let (one, one_stamps, one_declined, one_refused) = drive(place, None);
        assert_eq!(
            one_declined, 0,
            "{place:?}: one lane read a store across cards"
        );
        assert!(one_stamps.iter().flatten().all(|s| s.commit_card == 0));
        assert!(
            one_stamps.iter().any(|groups| groups.len() > 1),
            "{place:?}: needs a level with at least two groups"
        );
        let alternating: Vec<u8> = (0..64).map(|g| (g % 2) as u8).collect();
        for policy in [LanePolicy::NextFree, LanePolicy::Fixed(alternating)] {
            let what = format!("{setting:?} {place:?} {policy:?}");
            let lanes = PhaseALanes {
                knob: Some(TwoCardKnob::Levels),
                policy: policy.clone(),
                ..PhaseALanes::default()
            };
            let (two, two_stamps, declined, refused) = drive(place, Some(lanes));
            assert_eq!(declined, 0, "{what}: a store was read from the other card");
            assert_eq!(two.len(), one.len(), "{what}: the levels");
            for (k, (a, b)) in two.iter().zip(&one).enumerate() {
                assert!(a == b, "{what}: level {} moved a byte", k + 1);
                println!(
                    "LEVELS COMMITS BY CARD ({what}) IDENTITY level {}: {} B equal",
                    k + 1,
                    a.len()
                );
            }
            // Host commits one run made beyond the other, over every level.
            let (mut host_beyond_one, mut host_beyond_two) = (0u64, 0u64);
            for (k, (two, one)) in two_stamps.iter().zip(&one_stamps).enumerate() {
                assert_eq!(two.len(), one.len(), "{what}: level {}'s groups", k + 1);
                for (g, (a, b)) in two.iter().zip(one).enumerate() {
                    let at = format!("{what}: level {} group {g} (lane {})", k + 1, a.commit_card);
                    assert!(a.commit_card <= 1, "{at}: a lane past the second");
                    if let LanePolicy::Fixed(cards) = &policy {
                        assert_eq!(
                            a.commit_card,
                            cards.get(g).copied().unwrap_or(0),
                            "{at}: planned commit lane"
                        );
                    }
                    // What the group's layout fixes: the group at its index,
                    // whichever lane committed it and whatever its card's
                    // room.
                    assert_eq!(
                        (a.tables, a.polys, a.cells, a.commit_priced),
                        (b.tables, b.polys, b.cells, b.commit_priced),
                        "{at}: tables, polynomials, cells and priced bytes"
                    );
                    // The joint runs a group commits, and those committed to
                    // be retired, are its layout's: one lane's on either
                    // lane, counted on the lane's own card.
                    assert_eq!(
                        a.joint_runs[..2],
                        b.joint_runs[..2],
                        "{at}: joint runs committed and to be retired"
                    );
                    // The rest is the committing card's call. Whether a run
                    // streams into its leaves: each codeword asks the lane's
                    // ledger for its room, and a refusal commits the run
                    // batched (the same tree). Within the group the two
                    // counters must agree, on either run.
                    for (s, run) in [(a, "two lanes"), (b, "one lane")] {
                        assert!(
                            s.joint_runs[2] <= s.joint_runs[1],
                            "{at}: {run}: streamed runs {:?}",
                            s.joint_runs
                        );
                        assert_eq!(
                            s.joint_runs[2] > 0,
                            s.commit_streamed_not_held > 0,
                            "{at}: {run}: streamed runs {:?} against codeword bytes not held {}",
                            s.joint_runs,
                            s.commit_streamed_not_held
                        );
                    }
                    // Fallback and room counters are for diagnosis.
                    host_beyond_one += a
                        .commit_host_fallbacks
                        .saturating_sub(b.commit_host_fallbacks);
                    host_beyond_two += b
                        .commit_host_fallbacks
                        .saturating_sub(a.commit_host_fallbacks);
                    if a.joint_runs[2] != b.joint_runs[2]
                        || a.commit_host_fallbacks != b.commit_host_fallbacks
                    {
                        println!(
                            "LEVELS COMMITS BY CARD ({at}): path differs · two lanes: {} · one \
                             lane: {}",
                            ledger(a),
                            ledger(b)
                        );
                    }
                    // Tables packed narrow after the commit: packed from the
                    // group's store on its card, so the card's room decides
                    // (a store refused, or a pack the card declines, leaves a
                    // table wide); never more than the group's tables.
                    assert!(
                        a.packed_tables <= a.tables,
                        "{at}: packed {} of {} tables",
                        a.packed_tables,
                        a.tables
                    );
                    if a.packed_tables != b.packed_tables {
                        println!(
                            "LEVELS COMMITS BY CARD ({at}): tables packed {} against one lane's {}",
                            a.packed_tables, b.packed_tables
                        );
                    }
                    if a.commit_card == 1 {
                        on_second += 1;
                    }
                }
                let cards: Vec<u8> = two.iter().map(|s| s.commit_card).collect();
                let packed: usize = two.iter().map(|s| s.packed_tables).sum();
                let packed_one: usize = one.iter().map(|s| s.packed_tables).sum();
                println!(
                    "LEVELS COMMITS BY CARD ({what}) level {}: groups by lane {cards:?} · tables \
                     packed {packed} (one lane {packed_one})",
                    k + 1
                );
            }
            println!(
                "LEVELS COMMITS BY CARD ({what}): host commits beyond the other run: two lanes \
                 {host_beyond_one} ({refused} device refusals), one lane {host_beyond_two} \
                 ({one_refused} device refusals)"
            );
        }
    }
    println!(
        "LEVELS COMMITS BY CARD ({setting:?}): {on_second} level group(s) committed by the second lane"
    );
    on_second
}

/// ★ Card, run alone, deterministic grind. Each level's phase A on two commit
/// lanes forced on one card (the second lane on a second backend of card 0:
/// its own streams, staging, ledger and pools) proves each level's bytes as
/// one lane does, with one lane's layout counters group by group and each
/// room-dependent path its own card's call, and the second lane committed
/// level groups.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: proves a small BLAKE3 block and its levels six times; run alone, deterministic grind"]
fn the_blake3_driver_with_its_level_commits_forced_on_one_card_proves_the_same_level_bytes() {
    let second = driven_with_commits_by_card(math_cuda::device::TwoCardSplit::Forced);
    assert!(second > 0, "no level group committed by the second lane");
}

/// ★ Two cards, run alone, deterministic grind. With two cards seen and the
/// levels' knob at `auto`, the second commit lane runs on card 1 and each
/// level proves the bytes of one lane.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "two cards: proves a small BLAKE3 block and its levels six times; run alone, deterministic grind"]
fn the_blake3_driver_with_its_level_commits_on_card_1_proves_the_same_level_bytes() {
    use math_cuda::device::{TwoCardSplit, card_count, split_placement};
    assert!(card_count() >= 2, "needs two cards, saw {}", card_count());
    math_cuda::device::set_two_card_levels_for_test(Some(TwoCardSplit::Auto));
    assert_eq!(split_placement().ordinal, 1, "the second driver on card 1");
    let second = driven_with_commits_by_card(TwoCardSplit::Auto);
    assert!(second > 0, "no level group committed on card 1");
}

/// ★ Level 2 executed beside its own preparation
/// (`LAMBDA_VM_LEVELS_NEXT_BESIDE`), alone and with its stacks committed
/// early, is the same level: a small BLAKE3 block driven with the knob on
/// proves each level's bytes as [`prove_levels`] does over the same base, and
/// the driver stamps the prebuild made beside the stream. Needs
/// `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels three times; box tier, deterministic grind"]
fn the_blake3_driver_with_level_2_beside_its_prepare_proves_the_same_level_bytes() {
    for commit_early in [false, true] {
        let run = driven_under_chain_steps(ChainSteps {
            next_beside: true,
            commit_early,
            ..ChainSteps::default()
        });
        let stream = run.times.next_stream.expect("level 2 streamed");
        assert!(stream.prebuilt.is_some(), "the prebuild made beside");
        assert_eq!(
            run.times.next_commit.map(|c| c.early),
            Some(commit_early),
            "committed early as asked"
        );
    }
}

/// The level-2 streaming knob's name; it is on by default, and only `0`
/// turns it off.
#[test]
fn the_stream_next_knob_is_named_and_on_by_default() {
    use super::levels::{STREAM_NEXT_ENV, stream_next};
    use super::wlfm_knobs::knob_word_off;
    assert_eq!(STREAM_NEXT_ENV, "LAMBDA_VM_LEVELS_STREAM_NEXT");
    if std::env::var_os(STREAM_NEXT_ENV).is_none() {
        assert!(stream_next());
    }
    assert!(knob_word_off(Some("0")) && !knob_word_off(Some("1")) && !knob_word_off(None));
}

/// The BLAKE3 driver with every level's prepared stacks retired and each group
/// opened beside the next group's argue where the plan fits
/// (`LAMBDA_VM_LEVELS_OVERLAP_PLANNED`) proves the same level bytes as
/// [`prove_levels`] does over the same base with held stacks; it reports how
/// many groups opened beside and how many the plan kept alone. Needs
/// `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind"]
fn the_blake3_driver_with_the_planned_overlap_proves_the_same_level_bytes() {
    use super::levels::{prove_block_levels, set_stacks_for_test};
    use super::wlfm::{StackPlace, set_overlap_planned_for_test};
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_overlap_planned_for_test(None);
        }
    }
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    let run = {
        let _restore = Restore;
        let _retired = set_stacks_for_test(Some(StackPlace::CardRetired));
        set_overlap_planned_for_test(Some(true));
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
    let count = |f: &dyn Fn(&stark::multilinear_block::GroupStamps) -> bool| -> usize {
        run.times
            .stamps
            .iter()
            .map(|st| st.groups.iter().filter(|g| f(g)).count())
            .sum()
    };
    let beside = count(&|g| g.overlapped);
    let planned_out = count(&|g| g.overlap_planned_out);
    println!("LEVELS OVERLAP PLANNED: {beside} groups opened beside · {planned_out} planned out");
    let held = {
        let _held = set_stacks_for_test(Some(StackPlace::Card));
        derive_levels(&run.plan, WlfmFamily::B3, false).expect("the levels derive")
    };
    let (proofs, _) = prove_levels(&run.plan, &held, &run.base_proof).expect("the levels prove");
    assert_eq!(proofs.len(), run.proofs.len());
    for (k, (a, b)) in run.proofs.iter().zip(&proofs).enumerate() {
        let a = rkyv::to_bytes::<rkyv::rancor::Error>(a).expect("serializes");
        let b = rkyv::to_bytes::<rkyv::rancor::Error>(b).expect("serializes");
        assert!(a[..] == b[..], "level {}: the same proof bytes", k + 1);
        println!(
            "LEVELS OVERLAP PLANNED IDENTITY level {}: {} B equal",
            k + 1,
            a.len()
        );
    }
}

/// The BLAKE3 driver with every level's prepared stacks retired and each
/// retired stack's columns put on the card once (`LAMBDA_VM_LEVELS_UPLOAD_ONCE`)
/// proves the same level bytes as [`prove_levels`] does over the same base with
/// held stacks; on a card some group used its store and none was refused.
/// Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind"]
fn the_blake3_driver_uploading_once_proves_the_same_level_bytes() {
    use super::levels::{prove_block_levels, set_stacks_for_test};
    use super::wlfm::{StackPlace, set_upload_once_for_test};
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_upload_once_for_test(None);
        }
    }
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    let run = {
        let _restore = Restore;
        let _retired = set_stacks_for_test(Some(StackPlace::CardRetired));
        set_upload_once_for_test(Some(true));
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
    let stamps: Vec<_> = run
        .times
        .stamps
        .iter()
        .flat_map(|st| st.groups.iter().map(|g| g.upload_once))
        .collect();
    let used = stamps.iter().filter(|u| u.used).count();
    let refused = stamps.iter().filter(|u| u.refused).count();
    println!(
        "LEVELS UPLOAD ONCE: {used} stores used · {refused} refused · {} planned out",
        stamps.iter().filter(|u| u.planned_out).count()
    );
    assert_eq!(refused, 0, "no store refused by the card");
    if multilinear::gpu::reserve_budget() > 0 {
        assert!(
            used > 0,
            "on a card, some retired stack's columns went up once"
        );
    }
    let held = {
        let _held = set_stacks_for_test(Some(StackPlace::Card));
        derive_levels(&run.plan, WlfmFamily::B3, false).expect("the levels derive")
    };
    let (proofs, _) = prove_levels(&run.plan, &held, &run.base_proof).expect("the levels prove");
    assert_eq!(proofs.len(), run.proofs.len());
    for (k, (a, b)) in run.proofs.iter().zip(&proofs).enumerate() {
        let a = rkyv::to_bytes::<rkyv::rancor::Error>(a).expect("serializes");
        let b = rkyv::to_bytes::<rkyv::rancor::Error>(b).expect("serializes");
        assert!(a[..] == b[..], "level {}: the same proof bytes", k + 1);
        println!(
            "LEVELS UPLOAD ONCE IDENTITY level {}: {} B equal",
            k + 1,
            a.len()
        );
    }
}

/// [`the_blake3_driver_uploading_once_proves_the_same_level_bytes`] with each
/// store sent narrow (`LAMBDA_VM_LEVELS_NARROW_STORE=1` in the process's
/// environment, which the test refuses to run without): the same level bytes
/// as held stacks; on a card some store went up narrow and none was refused.
/// Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind, narrow store"]
fn the_blake3_driver_with_a_narrow_store_proves_the_same_level_bytes() {
    use super::levels::{prove_block_levels, set_stacks_for_test};
    use super::wlfm::{StackPlace, narrow_store, set_upload_once_for_test};
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_upload_once_for_test(None);
        }
    }
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    assert!(narrow_store(), "run with LAMBDA_VM_LEVELS_NARROW_STORE=1");
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    let run = {
        let _restore = Restore;
        let _retired = set_stacks_for_test(Some(StackPlace::CardRetired));
        set_upload_once_for_test(Some(true));
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
    let stamps: Vec<_> = run
        .times
        .stamps
        .iter()
        .flat_map(|st| st.groups.iter().map(|g| g.upload_once))
        .collect();
    let used = stamps.iter().filter(|u| u.used).count();
    let narrow = stamps.iter().filter(|u| u.used && u.narrow).count();
    let refused = stamps.iter().filter(|u| u.refused).count();
    println!(
        "LEVELS NARROW STORE: {used} stores used · {narrow} narrow · {refused} refused · {} planned out \
         · words {} B · crossed {} B",
        stamps.iter().filter(|u| u.planned_out).count(),
        stamps
            .iter()
            .filter(|u| u.used)
            .map(|u| u.bytes)
            .sum::<u64>(),
        stamps
            .iter()
            .filter(|u| u.used)
            .map(|u| if u.narrow { u.packed_bytes } else { u.bytes })
            .sum::<u64>(),
    );
    assert_eq!(refused, 0, "no store refused by the card");
    if multilinear::gpu::reserve_budget() > 0 {
        assert!(
            used > 0,
            "on a card, some retired stack's columns went up once"
        );
        assert_eq!(narrow, used, "every store used went up narrow");
    }
    let held = {
        let _held = set_stacks_for_test(Some(StackPlace::Card));
        derive_levels(&run.plan, WlfmFamily::B3, false).expect("the levels derive")
    };
    let (proofs, _) = prove_levels(&run.plan, &held, &run.base_proof).expect("the levels prove");
    assert_eq!(proofs.len(), run.proofs.len());
    for (k, (a, b)) in run.proofs.iter().zip(&proofs).enumerate() {
        let a = rkyv::to_bytes::<rkyv::rancor::Error>(a).expect("serializes");
        let b = rkyv::to_bytes::<rkyv::rancor::Error>(b).expect("serializes");
        assert!(a[..] == b[..], "level {}: the same proof bytes", k + 1);
        println!(
            "LEVELS NARROW STORE IDENTITY level {}: {} B equal",
            k + 1,
            a.len()
        );
    }
}

/// The fill-beside knob's name; it is on by default, and only `0` turns it
/// off.
#[test]
fn the_fill_beside_knob_is_named_and_on_by_default() {
    use super::levels::{FILL_BESIDE_ENV, fill_beside};
    use super::wlfm_knobs::knob_word_off;
    assert_eq!(FILL_BESIDE_ENV, "LAMBDA_VM_LEVELS_FILL_BESIDE");
    if std::env::var_os(FILL_BESIDE_ENV).is_none() {
        assert!(fill_beside());
    }
    assert!(knob_word_off(Some("0")) && !knob_word_off(Some("1")) && !knob_word_off(None));
}

/// The validator-at-once and free-beside knobs' names; both off by default
/// (and for any word but `1`), a test's setting read back.
#[test]
fn the_validate_at_once_and_free_beside_knobs_are_named_and_off_by_default() {
    use super::levels::{
        FREE_BESIDE_ENV, VALIDATE_AT_ONCE_ENV, free_beside, set_free_beside_for_test,
        set_validate_at_once_for_test, validate_at_once,
    };
    assert_eq!(VALIDATE_AT_ONCE_ENV, "LAMBDA_VM_LEVELS_VALIDATE_AT_ONCE");
    assert_eq!(FREE_BESIDE_ENV, "LAMBDA_VM_LEVELS_FREE_BESIDE");
    if std::env::var_os(VALIDATE_AT_ONCE_ENV).is_none() {
        assert!(!validate_at_once(), "validate at once off by default");
    }
    if std::env::var_os(FREE_BESIDE_ENV).is_none() {
        assert!(!free_beside(), "free beside off by default");
    }
    set_validate_at_once_for_test(Some(true));
    set_free_beside_for_test(Some(true));
    assert!(validate_at_once() && free_beside());
    set_validate_at_once_for_test(None);
    set_free_beside_for_test(None);
}

/// The validator-after-the-fill and builder-whole-pool knobs' names; both off
/// by default (and for any word but `1`), a test's setting read back.
#[test]
fn the_validate_after_fill_and_builder_whole_pool_knobs_are_named_and_off_by_default() {
    use super::levels::{
        VALIDATE_AFTER_FILL_ENV, set_validate_after_fill_for_test, validate_after_fill,
    };
    use super::wlfm::{
        BUILDER_WHOLE_POOL_ENV, builder_whole_pool, set_builder_whole_pool_for_test,
    };
    assert_eq!(
        VALIDATE_AFTER_FILL_ENV,
        "LAMBDA_VM_LEVELS_VALIDATE_AFTER_FILL"
    );
    assert_eq!(
        BUILDER_WHOLE_POOL_ENV,
        "LAMBDA_VM_LEVELS_BUILDER_WHOLE_POOL"
    );
    if std::env::var_os(VALIDATE_AFTER_FILL_ENV).is_none() {
        assert!(
            !validate_after_fill(),
            "validate after the fill off by default"
        );
    }
    if std::env::var_os(BUILDER_WHOLE_POOL_ENV).is_none() {
        assert!(!builder_whole_pool(), "builder whole pool off by default");
    }
    set_validate_after_fill_for_test(Some(true));
    set_builder_whole_pool_for_test(Some(true));
    assert!(validate_after_fill() && builder_whole_pool());
    set_validate_after_fill_for_test(None);
    set_builder_whole_pool_for_test(None);
}

/// A program taken ([`super::levels::take_program`], the free-beside path) is
/// left as a released one, with the same counts noted, and what was taken
/// holds every instruction and column.
#[test]
fn a_taken_program_is_left_as_a_released_one() {
    let mut taken = super::programs::trivial_program();
    let mut released = super::programs::trivial_program();
    let instrs = taken.instrs.len();
    assert!(instrs > 0);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let held = super::levels::take_program(&mut taken, &mut a);
    super::levels::release_program_for_test(&mut released, &mut b);
    assert_eq!(a, b, "the same counts noted");
    assert_eq!(a, vec![(instrs, released.groups.b3l.real_rows)]);
    assert!(taken.instrs.is_empty() && released.instrs.is_empty());
    assert_eq!(
        taken.groups.const_.data.len(),
        released.groups.const_.data.len()
    );
    assert_eq!(taken.groups.const_.data.len(), 0);
    assert_eq!(held.instrs_len(), instrs, "every instruction taken");
}

/// The settle-late knob's name; it is off by default, and only `1` turns it
/// on.
#[test]
fn the_settle_late_knob_is_named_and_off_by_default() {
    use super::levels::{SETTLE_LATE_ENV, settle_late, settle_late_word};
    assert_eq!(SETTLE_LATE_ENV, "LAMBDA_VM_LEVELS_SETTLE_LATE");
    if std::env::var_os(SETTLE_LATE_ENV).is_none() {
        assert!(!settle_late());
    }
    for (word, on) in [
        (None, false),
        (Some(""), false),
        (Some("0"), false),
        (Some("on"), false),
        (Some("1"), true),
        (Some(" 1 "), true),
    ] {
        assert_eq!(settle_late_word(word), on, "{word:?}");
    }
}

/// ★ Level 1's stacks committed at the base's end on the pool's pages, the
/// pool trimmed only after (`LAMBDA_VM_LEVELS_SETTLE_LATE`), are the same
/// proofs: a small BLAKE3 block driven with the knob on proves each level's
/// bytes as [`prove_levels`] does over the same base, and the driver read the
/// card at the base's end without a trim (no card: no reading). Needs
/// `LAMBDA_VM_DETERMINISTIC_GRIND=1`; an untimed identity check.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice; box tier, deterministic grind"]
fn the_blake3_driver_settling_late_proves_the_same_level_bytes() {
    use super::levels::{prove_block_levels, set_settle_late_for_test};
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_settle_late_for_test(None);
        }
    }
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    let run = {
        let _restore = Restore;
        set_settle_late_for_test(Some(true));
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
    let (secs, trimmed) = run
        .times
        .settle_base_end
        .expect("the driver read the card at the base's end");
    assert!(
        secs >= 0.0 && !trimmed,
        "the base-end reading skipped the trim"
    );
    let held = derive_levels(&run.plan, WlfmFamily::B3, false).expect("the levels derive");
    let (proofs, _) = prove_levels(&run.plan, &held, &run.base_proof).expect("the levels prove");
    assert_eq!(proofs.len(), run.proofs.len());
    for (k, (a, b)) in run.proofs.iter().zip(&proofs).enumerate() {
        let a = rkyv::to_bytes::<rkyv::rancor::Error>(a).expect("serializes");
        let b = rkyv::to_bytes::<rkyv::rancor::Error>(b).expect("serializes");
        assert!(a[..] == b[..], "level {}: the same proof bytes", k + 1);
        println!(
            "LEVELS SETTLE LATE IDENTITY level {}: {} B equal",
            k + 1,
            a.len()
        );
    }
}

/// The levels' tail knob's name; it is on by default, and only `0` turns it
/// off.
#[test]
fn the_levels_tail_knob_is_named_and_on_by_default() {
    use super::wlfm_knobs::{TAIL_ENV, tail, tail_word};
    assert_eq!(TAIL_ENV, "LAMBDA_VM_LEVELS_TAIL");
    if std::env::var_os(TAIL_ENV).is_none() {
        assert!(tail());
    }
    for (word, on) in [
        (None, true),
        (Some(""), true),
        (Some("0"), false),
        (Some(" 0 "), false),
        (Some("on"), true),
        (Some("1"), true),
    ] {
        assert_eq!(tail_word(word), on, "{word:?}");
    }
}

/// The untrimmed-readings knob's name; it is off by default, and only `1`
/// turns it on.
#[test]
fn the_levels_untrimmed_knob_is_named_and_off_by_default() {
    use super::wlfm_knobs::{TAIL_ENV, UNTRIMMED_ENV, knob_word_on, levels_untrimmed};
    assert_eq!(UNTRIMMED_ENV, "LAMBDA_VM_LEVELS_UNTRIMMED");
    if std::env::var_os(UNTRIMMED_ENV).is_none() && std::env::var_os(TAIL_ENV).is_none() {
        // The tail, on by default, takes its readings untrimmed.
        assert!(levels_untrimmed());
    }
    assert!(knob_word_on(Some("1")) && !knob_word_on(Some("0")) && !knob_word_on(None));
}

/// ★ The levels' tail taken off the path, in parts and whole
/// (`LAMBDA_VM_LEVELS_UNTRIMMED`: every card reading inside the levels
/// untrimmed; `LAMBDA_VM_WLFM_LEAN_BESIDE`: level 1's lean tables filled
/// beside its execution; `LAMBDA_VM_LEVELS_TAIL`: both) proves the same level
/// bytes: a small BLAKE3 block driven under each proves each level's bytes as
/// [`prove_levels`] does over the same base. Every knob off is today's driver:
/// no reading skipped its trim, the base-end reading trimmed, no lean
/// prefill. Untrimmed (alone or in the tail): the base-end reading and every
/// reading inside the levels skipped theirs. The tail: level 1's fill took the
/// lean prefill. Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1`; run alone (the
/// overrides are process wide).
#[test]
#[ignore = "proves a small BLAKE3 block and its levels four times; box tier, deterministic grind"]
fn the_blake3_driver_with_the_tail_off_the_path_proves_the_same_level_bytes() {
    use super::b3l_tables::{B3lHold, test_hold};
    use super::levels::{prove_block_levels, untrimmed_readings};
    use super::wlfm_knobs::{set_tail_for_test, set_untrimmed_for_test};
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_tail_for_test(None);
            set_untrimmed_for_test(None);
        }
    }
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    let _records = test_hold::set(B3lHold::Records);
    let format = b3_format();
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
    let opts = ProofOptions::default_test_options();
    let drive = |tail: bool, untrimmed: bool| {
        set_tail_for_test(Some(tail));
        set_untrimmed_for_test(Some(untrimmed));
        let before = untrimmed_readings();
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
        (run, untrimmed_readings() - before)
    };
    let _restore = Restore;
    // Every knob off: today's driver.
    let (off, none) = drive(false, false);
    assert_eq!(none, 0, "the knobs off skip no trim");
    let (_, trimmed) = off.times.settle_base_end.expect("the base-end reading");
    assert!(trimmed, "the knobs off: the base-end reading trimmed");
    assert!(!off.times.lean_prefilled, "the knobs off: no lean prefill");
    let held = derive_levels(&off.plan, WlfmFamily::B3, false).expect("the levels derive");
    let (want, _) = prove_levels(&off.plan, &held, &off.base_proof).expect("the levels prove");
    let bytes = |p: &super::wlfm::WlfmProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serializes")
            .to_vec()
    };
    for (k, (a, b)) in off.proofs.iter().zip(&want).enumerate() {
        assert!(
            bytes(a) == bytes(b),
            "off, level {}: the levels prover's bytes",
            k + 1
        );
    }
    for (tail, untrimmed, what) in [(false, true, "untrimmed"), (true, false, "the tail")] {
        let (run, skipped) = drive(tail, untrimmed);
        let (_, trimmed) = run.times.settle_base_end.expect("the base-end reading");
        assert!(!trimmed, "{what}: the base-end reading skipped its trim");
        assert!(
            skipped >= 3,
            "{what}: the readings inside the levels skipped theirs ({skipped})"
        );
        assert_eq!(
            run.times.lean_prefilled, tail,
            "{what}: the lean prefill only under the tail"
        );
        assert_eq!(run.proofs.len(), want.len(), "{what}: the levels");
        for (k, (a, b)) in run.proofs.iter().zip(&want).enumerate() {
            assert!(
                bytes(a) == bytes(b),
                "{what}, level {}: the same proof bytes",
                k + 1
            );
            println!(
                "LEVELS TAIL IDENTITY {what} level {}: {} B equal",
                k + 1,
                bytes(a).len()
            );
        }
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
/// off the clock, as `RTA_VERIFY` says: unset or `all` (the default) every
/// run; `last` the process's last run alone; `off` no run (`RTA VERIFIED:
/// skipped` a run skipped: a timing process whose arm the job's identity step
/// verifies).
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
    // The build's security posture (`crate::security_posture`): production in
    // a timing binary, test in a `test-security` one. `RTA_POSTURE` names the
    // one the caller expects, and any other refuses before anything proves.
    let posture = crate::security_posture::name();
    println!("RTA POSTURE: {posture}");
    rta_posture_check(std::env::var("RTA_POSTURE").ok().as_deref(), posture)
        .unwrap_or_else(|e| panic!("{e}"));
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
    // A misspelt RTA_VERIFY refuses here, before any prove.
    let verify = RtaVerify::from_word(std::env::var("RTA_VERIFY").ok().as_deref());
    // A pricing probe: keep the process's memory between blocks (see
    // `KEEP_MEMORY_FOR_TEST`); off unless `RTA_KEEP_MEMORY=1`.
    let keep = std::env::var("RTA_KEEP_MEMORY").is_ok_and(|v| v.trim() == "1");
    super::levels::KEEP_MEMORY_FOR_TEST.store(keep, std::sync::atomic::Ordering::Relaxed);
    let opts = super::proof::block_base_options();
    match verify {
        RtaVerify::All => {}
        RtaVerify::Last => {
            println!("RTA VERIFY: last (RTA_VERIFY=last): the process's last run alone verified")
        }
        RtaVerify::Off => {
            println!("RTA VERIFY: off (RTA_VERIFY=off): no run verified in this process")
        }
    }
    println!(
        "RTA CONFIG: base {:?} · tag {} · group polys {} · family {family:?} · levels {levels} · \
         beside {beside} · build beside {build_beside} · runs {runs}",
        cfg.format.zf.base.hash,
        String::from_utf8_lossy(&crate::block_whir::block_format_tag(&cfg.format)),
        cfg.format.group_polys
    );
    println!(
        "RTA LAST LEVEL: {:?} · file {:?}",
        super::last_level::last_level(family).unwrap_or_else(|e| panic!("{e}")),
        super::last_level::last_wire().unwrap_or_else(|e| panic!("{e}"))
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
            verify.skips(k, runs.max(1)),
        );
    }
    println!("RTA DONE");
}

/// `RTA_POSTURE` against the build's posture: unset or empty takes the
/// build's; `production` or `test` must be the build's; any other word refuses.
fn rta_posture_check(want: Option<&str>, build: &str) -> Result<(), String> {
    match want.map(str::trim) {
        None | Some("") => Ok(()),
        Some(w @ ("production" | "test")) if w == build => Ok(()),
        Some(w @ ("production" | "test")) => Err(format!(
            "RTA_POSTURE={w}, but this binary proves under the {build} posture \
             (a timing binary is built without `test-security`, a gate binary with it)"
        )),
        Some(w) => Err(format!("RTA_POSTURE is production or test, not {w:?}")),
    }
}

/// `RTA_POSTURE` takes the build's posture or nothing, and refuses the other
/// and any misspelling before the harness proves anything.
#[test]
fn the_rta_posture_word_must_be_the_builds() {
    let build = crate::security_posture::name();
    let other = if build == "test" {
        "production"
    } else {
        "test"
    };
    assert_eq!(rta_posture_check(None, build), Ok(()));
    assert_eq!(rta_posture_check(Some(" "), build), Ok(()));
    assert_eq!(rta_posture_check(Some(build), build), Ok(()));
    assert!(rta_posture_check(Some(other), build).is_err());
    assert!(rta_posture_check(Some("prod"), build).is_err());
    assert!(rta_posture_check(Some("Test"), build).is_err());
}

/// Level 1's and level 2's host chains, on the driver's clock: what each
/// waited for before its prove could start, so a lever on the chain reads as
/// a moved stamp (the levels' chain steps, `LAMBDA_VM_LEVELS_FILL_PARTS`,
/// `LAMBDA_VM_LEVELS_COMMIT_NEXT_EARLY`).
fn print_level_chains(t: &super::levels::LevelsTimes, l1_prove_from: f64) {
    let p = t.plan_split;
    println!(
        "RTA LEVEL1 PLAN: frame {:.3}s · local {:.3}s · prepared {:.3}s · costs {:.3}s · receivers {:.3}s · \
         probes {:.3}s · plan lean {} · stacks by card {} · statement early {}",
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        p[5],
        super::levels::plan_lean(),
        super::levels::stacks_by_card(),
        super::levels::statement_early()
    );
    match t.early {
        Some(e) => println!(
            "RTA LEVEL1 EARLY: statement at {:.3}s · planned and emitted at {:.3}s · confirmed at {:.3}s · \
             waited {:.3}s · {} · mismatches {}",
            e.stated_at,
            e.made_at,
            e.confirmed_at,
            e.waited,
            if e.used { "used" } else { "NOT USED" },
            super::levels::early_mismatches()
        ),
        None => println!(
            "RTA LEVEL1 EARLY: {}",
            if super::levels::statement_early() {
                "none came"
            } else {
                "off"
            }
        ),
    }
    let c = t.compile_split;
    println!(
        "RTA LEVEL1 COMPILE: multiplicities {:.3}s · rows {:.3}s · join {:.3}s",
        c[0], c[1], c[2]
    );
    let landing = &t.landing;
    println!(
        "RTA LEVEL1 WORDS: groups {} · {:.3}s",
        landing.words.0, landing.words.1
    );
    let front = landing.front.0 + landing.front.1;
    let last = landing
        .landed
        .iter()
        .map(|&(_, at, secs)| at + secs)
        .fold(front, f64::max);
    // Every share at hand at the front: the pieces' words and waves back to
    // back after it.
    let busy: f64 = landing.landed.iter().map(|&(_, _, secs)| secs).sum::<f64>() + landing.words.1;
    let could = front + busy;
    let l1 = t.levels.first().copied().unwrap_or_default();
    let commit = t.builds.first().map_or(0.0, |b| b.commit);
    let fmt = |x: Option<f64>| x.map_or("-".to_string(), |x| format!("{x:.3}s"));
    let floor = t.builder_end.map(|b| (could + l1.fill).max(b + commit));
    println!(
        "RTA LEVEL1 CHAIN: planned {:.3}s · emitted {:.3}s · front {front:.3}s · last piece {last:.3}s · \
         last piece could land at {could:.3}s · fill {:.3}s · builder ended at {} · commit {commit:.3}s · \
         base end {:.3}s · slack {} · chain floor {} · prove from {l1_prove_from:.3}s · fill parts {} · \
         columns tiled {} · packs beside {}",
        t.planned,
        t.emitted,
        l1.fill,
        fmt(t.builder_end),
        t.base,
        fmt(t.builder_end.map(|b| t.base - b)),
        fmt(floor),
        super::levels::fill_parts(),
        super::wlfm::columns_tiled(),
        super::levels::packs_beside(),
    );
    let l2 = t.levels.get(1).copied().unwrap_or_default();
    if let (Some(ns), Some(nc), Some(b)) = (t.next_stream, t.next_commit, t.builds.get(1)) {
        let prove_from = l2.end - l2.prove;
        println!(
            "RTA LEVEL2 CHAIN: prepared {:.3}s · first piece {:.3}s · last piece {:.3}s · executed {:.3}s · \
             fill {:.3}s · filled {:.3}s · commit from {:.3}s · early {} · joined {} · committed {:.3}s · \
             level 1 proved {:.3}s · prove from {prove_from:.3}s · after level 1 {:.3}s · prebuilt {} · \
             next beside {}",
            ns.from,
            ns.first,
            ns.last,
            ns.executed,
            ns.fill,
            ns.executed + ns.fill,
            nc.from,
            nc.early,
            fmt(nc.joined),
            b.committed,
            l1.end,
            prove_from - l1.end,
            fmt(ns.prebuilt),
            super::levels::next_beside(),
        );
    } else {
        println!("RTA LEVEL2 CHAIN: not streamed");
    }
}

/// This process's minor and major faults and its user and system seconds,
/// from `/proc/self/stat` (Linux; `None` elsewhere or on a parse miss).
fn self_usage() -> Option<(u64, u64, f64, f64)> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The fields after the command's closing parenthesis, from field 3 on.
    let rest = &stat[stat.rfind(')')? + 2..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    let num = |k: usize| f.get(k - 3)?.parse::<u64>().ok();
    // `sysconf(_SC_CLK_TCK)` is 100 on the boxes' Linux.
    let tick = 100.0;
    Some((
        num(10)?,
        num(12)?,
        num(14)? as f64 / tick,
        num(15)? as f64 / tick,
    ))
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
    skip_checks: Option<&str>,
) {
    use super::levels::prove_block_levels;
    let (elf, input) = (elf.to_vec(), input.to_vec());
    let opts = opts.clone();
    let usage_before = self_usage();
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
    if let (Some(a), Some(b)) = (usage_before, self_usage()) {
        println!(
            "RTA USAGE: minor faults {} · major faults {} · user {:.2}s · system {:.2}s (the driver's call, \
             its purge and trim before the clock included)",
            b.0.saturating_sub(a.0),
            b.1.saturating_sub(a.1),
            b.2 - a.2,
            b.3 - a.3
        );
    }
    let t = &run.times;
    let s = &run.base_stamps;
    println!(
        "RTA BASE: end {:.3}s · execute {:.3}s · build {:.3}s · phase A {:.3}s · phase B {:.3}s",
        t.base, s.execute, s.build, s.phase_a, s.phase_b
    );
    // The base's own stamp report (BLOCK SHAPE, a BLOCK GROUP line a group, BLOCK PHASE-B SPAN,
    // BLOCK PHASES, …), as the CLI prints it: printed after the run, outside every timed span.
    eprint!("{}", s.report());
    // The base's groups, in the order phase B proves them: each its index,
    // table count and first table's name, so a late group can be named.
    let names = run.plan.table_names();
    println!(
        "RTA BASE GROUPS: {}",
        run.base_proof
            .statement()
            .groups
            .iter()
            .enumerate()
            .map(|(g, tables)| {
                let first = tables
                    .first()
                    .and_then(|&t| names.get(t as usize))
                    .map_or("-", String::as_str);
                format!("{g}:{}:{first}", tables.len())
            })
            .collect::<Vec<_>>()
            .join(" ")
    );
    // Level 1's prove starts once its stacks are committed and its traces
    // filled: after one another, or, filled beside the commit
    // (`LAMBDA_VM_LEVELS_FILL_BESIDE`), at the later of the two.
    let l1_prove_from = match t.fill_beside {
        Some((_, done)) => t.built.max(done),
        None => t.built + t.levels.first().map_or(0.0, |l| l.fill),
    };
    println!(
        "RTA LEVEL1 FILL: beside {} · fill {:.3}s · executed at {:.3}s · filled at {} · built {:.3}s · prove from {:.3}s",
        t.fill_beside.is_some(),
        t.levels.first().map_or(0.0, |l| l.fill),
        t.levels.first().map_or(0.0, |l| l.executed),
        match t.fill_beside {
            Some((_, done)) => format!("{done:.3}s"),
            None => "after the build".to_string(),
        },
        t.built,
        l1_prove_from
    );
    if let Some((secs, trimmed)) = t.settle_base_end {
        println!(
            "RTA LEVEL1 SETTLE: at the base end {secs:.3}s · trimmed {trimmed} · tail {} · readings untrimmed {}",
            super::wlfm_knobs::tail(),
            super::levels::untrimmed_readings()
        );
    }
    // Level 2 streamed beside level 1's prove (`LAMBDA_VM_LEVELS_STREAM_NEXT`):
    // its pieces' landing against level 1's proved time, and level 1's own
    // prove seconds (the stream must not slow it).
    let l1 = t.levels.first().copied().unwrap_or_default();
    let l2 = t.levels.get(1).copied().unwrap_or_default();
    match t.next_stream {
        Some(ns) => println!(
            "RTA LEVEL2 STREAM: on true · pieces {} · from {:.3}s · first piece at {:.3}s · last piece at {:.3}s · \
             executed at {:.3}s · fill {:.3}s · level 1 prove {:.3}s · level 1 proved at {:.3}s · \
             executed after level 1 {:.3}s · level 2 prove {:.3}s",
            ns.pieces,
            ns.from,
            ns.first,
            ns.last,
            ns.executed,
            ns.fill,
            l1.prove,
            l1.end,
            ns.executed + ns.fill - l1.end,
            l2.prove
        ),
        None => println!(
            "RTA LEVEL2 STREAM: on false · level 1 prove {:.3}s · level 1 proved at {:.3}s · \
             level 2 executed at {:.3}s · executed after level 1 {:.3}s · level 2 prove {:.3}s",
            l1.prove,
            l1.end,
            l2.executed,
            l2.executed - l1.end,
            l2.prove
        ),
    }
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
             {:.3}s · at once {}",
            t.levels.first().map_or(0.0, |l| l.executed),
            super::levels::validate_at_once()
        ),
        None => println!("RTA LEVEL1 VALIDATE: beside off"),
    }
    println!(
        "RTA LEVEL1 RELEASE: {:.3}s on the driver's thread · beside {}",
        t.release.0, t.release.1
    );
    match t.validated_after_fill {
        Some(wait) => println!("RTA LEVEL1 VALIDATE AFTER FILL: on · the join waited {wait:.3}s"),
        None => println!("RTA LEVEL1 VALIDATE AFTER FILL: off"),
    }
    match t.builder_whole_pool {
        Some(note) => println!(
            "RTA LEVEL1 BUILDER POOL: watched · {} of {} tables' columns on one thread · the base seen at {} \
             into the columns · packs on the whole pool {}",
            note.tables_before,
            note.tables,
            note.seen_at
                .map_or_else(|| "never".to_string(), |s| format!("{s:.3}s")),
            note.packs_whole
        ),
        None => println!("RTA LEVEL1 BUILDER POOL: off"),
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
             {:.3}s · the rest {:.3}s · waves in order {}",
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
            x.residue,
            super::levels::waves_in_order(),
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
        // The level's compact bytes (canonical, unlike the archive's) as one
        // digest: two heads' runs under `LAMBDA_VM_DETERMINISTIC_GRIND=1`
        // compare it, off the clock.
        {
            use digest::Digest;
            let d: [u8; 32] =
                <crypto::hash::blake3::Blake3Std as Digest>::digest(super::wire::encode(p)).into();
            println!(
                "RTA LEVEL {} COMPACT HASH: {}",
                k + 1,
                d.iter().map(|b| format!("{b:02x}")).collect::<String>()
            );
        }
        // The proof's bytes by component (printed after the run, off the
        // clock): the compact encoding's parts, which sum to its total.
        print_level_census(k + 1, level, p);
        if let Some(b) = t.builds.get(k) {
            println!(
                "RTA LEVEL {} BUILD: stacks {} · prepared cells {} · prepare {:.3}s · commit {:.3}s \
                 · prefix pack {:.3}s · committed at {:.3}s · prepare groups {:.3}s · columns {:.3}s · \
                 stacks {:.3}s · columns on the pool {}",
                k + 1,
                place_word(b.place),
                b.cells,
                b.prepare,
                b.commit,
                b.prefix,
                b.committed,
                b.prepare_split[0],
                b.prepare_split[1],
                b.prepare_split[2],
                super::wlfm::prepare_par(),
            );
        }
        let narrow = level.build.narrow_commit_stamps();
        if !narrow.is_empty() {
            let gb = |f: &dyn Fn(&super::wlfm::NarrowCommitStamp) -> u64| {
                narrow.iter().map(f).sum::<u64>() as f64 / 1e9
            };
            println!(
                "RTA LEVEL {} NARROW COMMIT: stacks {} · sent {} · packed ahead {} · words {:.3} GB · crossed \
                 {:.3} GB · pack {:.4}s · up {:.4}s",
                k + 1,
                narrow.len(),
                narrow.iter().filter(|n| n.sent).count(),
                narrow.iter().filter(|n| n.packed_ahead).count(),
                gb(&|n| n.words),
                gb(&|n| if n.sent { n.packed_bytes } else { 0 }),
                narrow.iter().map(|n| n.pack_secs).sum::<f64>(),
                narrow.iter().map(|n| n.upload_secs).sum::<f64>(),
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
    print_level_chains(t, l1_prove_from);
    // The base's and each level's proof bytes, digested: equal across arms
    // that only move when host work runs (under
    // `LAMBDA_VM_DETERMINISTIC_GRIND=1`).
    let base_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&run.base_proof).expect("serializes");
    println!(
        "RTA PROOF DIGEST: deterministic grind {} · base {} B {} · {}",
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        base_bytes.len(),
        &blake3::hash(&base_bytes).to_hex()[..16],
        run.proofs
            .iter()
            .enumerate()
            .map(|(k, p)| {
                let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(p).expect("serializes");
                format!(
                    "level {} {} B {}",
                    k + 1,
                    bytes.len(),
                    &blake3::hash(&bytes).to_hex()[..16]
                )
            })
            .collect::<Vec<_>>()
            .join(" · ")
    );
    // Off the clock: the base (a BLAKE3 one under the test permit), every
    // level, and the block's verifier, each step's seconds on one line —
    // unless `RTA_VERIFY` skips this run (`skip_checks`: its word).
    if let Some(word) = skip_checks {
        println!("RTA VERIFIED: skipped (RTA_VERIFY={word})");
    } else {
        let t = std::time::Instant::now();
        let ok_base = crate::base_only::with_permit(|| {
            crate::block_whir::verify_block_whir(&run.base_proof, &elf, &opts, &cfg.format)
        })
        .expect("verify runs");
        let base_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let mut ok_levels = true;
        for (level, p) in run.levels.iter().zip(&run.proofs) {
            ok_levels &= verify_wlfm(&level.build.artifacts, p, &p.public_words).is_ok();
        }
        let levels_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let ok_last = super::levels::verify_block_levels_under(
            &elf,
            &opts,
            &cfg.format,
            run.base_proof.statement(),
            family,
            LevelStop::Levels(levels),
            run.proofs.last().expect("a last level"),
        )
        .is_ok();
        let block_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let ok_file = rta_last_file(&elf, &opts, cfg, &run, family, levels);
        let file_s = t.elapsed().as_secs_f64();
        println!("RTA VERIFIED: base {ok_base} · levels {ok_levels} · block {ok_last}");
        // A tampered last proof (one published word changed) is refused against
        // the last level's artifacts.
        let t = std::time::Instant::now();
        let mut tampered = run.proofs.last().expect("a last level").clone();
        if let Some((_, w)) = tampered.public_words.first_mut() {
            w[0] += crate::tables::types::FE::one();
        }
        let top = &run.levels.last().expect("a last level").build.artifacts;
        let refused = verify_wlfm(top, &tampered, &tampered.public_words).is_err();
        let tamper_s = t.elapsed().as_secs_f64();
        println!("RTA TAMPER: a changed word of the last level refused {refused}");
        println!(
            "RTA VERIFY SECONDS: base {base_s:.3}s · levels {levels_s:.3}s · block {block_s:.3}s · \
             last file {file_s:.3}s · tamper {tamper_s:.3}s"
        );
        assert!(
            ok_base && ok_levels && ok_last && ok_file,
            "everything verifies"
        );
        assert!(refused, "a tampered last level is refused");
    }
    // Off the clock, a box row (`RTA_IDENTITY=host`, with
    // `LAMBDA_VM_DETERMINISTIC_GRIND=1`): every level proved again over the
    // same base with its prepared stacks on the host must give the same bytes
    // as the run's, whatever placement, upload once and overlap pricing the
    // run took (the large's retired path among them).
    if std::env::var("RTA_IDENTITY").is_ok_and(|v| v.trim() == "host") {
        assert!(
            std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
            "the identity needs the deterministic grind"
        );
        let host = {
            let _host = super::levels::set_stacks_for_test(Some(StackPlace::Host));
            super::levels::derive_levels_to(&run.plan, family, LevelStop::Levels(levels), false)
                .expect("the levels derive on the host")
        };
        let (proofs, _) = super::levels::prove_levels(&run.plan, &host, &run.base_proof)
            .expect("the levels prove");
        assert_eq!(proofs.len(), run.proofs.len(), "as many levels");
        for (k, (a, b)) in run.proofs.iter().zip(&proofs).enumerate() {
            let a = rkyv::to_bytes::<rkyv::rancor::Error>(a).expect("serializes");
            let b = rkyv::to_bytes::<rkyv::rancor::Error>(b).expect("serializes");
            println!(
                "RTA IDENTITY level {}: {} B · equal {}",
                k + 1,
                a.len(),
                a[..] == b[..]
            );
            assert!(
                a[..] == b[..],
                "level {}: the host-placed stacks' bytes",
                k + 1
            );
        }
    }
}

/// `RTA LEVEL k CENSUS` and `RTA LEVEL k QUERIES` lines: level `k`'s proof
/// by component ([`super::wire::census`]) and its chains' query counts by
/// round, with its schedule and final round.
fn print_level_census(k: usize, level: &super::levels::Level, p: &super::wlfm::WlfmProof) {
    let c = super::wire::census(p);
    let sum: usize = c.parts.iter().map(|(_, v)| v).sum();
    println!(
        "RTA LEVEL {k} CENSUS: rkyv {} B · compact {} B · dedup {} B · parts sum {sum} B · {}",
        c.rkyv,
        c.compact,
        c.dedup,
        c.line()
    );
    let a = &level.build.artifacts;
    let (layouts, _) = crate::multilinear_prove::stacks(&a.shapes(&a.airs()), &a.sizes, &a.config)
        .unwrap_or_default();
    let tallest = layouts.iter().map(|l| l.n_stack()).max().unwrap_or(0);
    println!(
        "RTA LEVEL {k} QUERIES: prep {:?} · first fold {:?} · final {:?} · rounds {:?} · schedule at \
         {tallest} {:?} · final round {:?} · Q a round {:?} · opened a chain {:?}",
        a.prep,
        a.first_fold,
        a.final_poly,
        a.rounds,
        a.config.schedule(tallest),
        a.config.final_round(tallest),
        a.config.round_queries(tallest),
        c.queries
    );
}

/// The last level's proof as a file in [`super::last_level::last_wire`]'s
/// encoding: its bytes printed, read back for the process's last-level
/// format and verified by the block's verifier (the decoded proof must be the
/// one the prover made). `true` when it verifies.
fn rta_last_file(
    elf: &[u8],
    opts: &ProofOptions,
    cfg: &super::whir_block_tree::WhirTreeConfig,
    run: &super::levels::LevelsRun,
    family: WlfmFamily,
    levels: usize,
) -> bool {
    use super::last_level::{last_level, last_level_file, last_wire, read_last_level_file};
    let wire = last_wire().unwrap_or_else(|e| panic!("{e}"));
    let format = last_level(family).unwrap_or_else(|e| panic!("{e}"));
    let last = run.proofs.last().expect("a last level");
    let bytes = last_level_file(last, format, wire).unwrap_or_else(|e| panic!("{e}"));
    let ok = match read_last_level_file(&bytes, format) {
        Ok(back) => super::levels::verify_block_levels_under(
            elf,
            opts,
            &cfg.format,
            run.base_proof.statement(),
            family,
            LevelStop::Levels(levels),
            &back,
        )
        .is_ok(),
        Err(e) => {
            println!("RTA LAST FILE REFUSED: {e}");
            false
        }
    };
    println!(
        "RTA LAST FILE: wire {wire:?} · {} B · read back and verified {ok}",
        bytes.len()
    );
    ok
}

/// Which runs of [`rta_a_real_block_through_its_levels`] are verified off the
/// clock (the base, every level, the shipped last-level file, a tampered last level
/// refused): `RTA_VERIFY` unset, empty or `all` (the default) every run;
/// `last` the process's last run alone (no run of the process follows a
/// verify); `off` none. Any other word refuses the process before its first
/// prove.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RtaVerify {
    All,
    Last,
    Off,
}

impl RtaVerify {
    fn from_word(word: Option<&str>) -> Self {
        match word.map(str::trim) {
            None | Some("") | Some("all") => Self::All,
            Some("last") => Self::Last,
            Some("off") => Self::Off,
            Some(other) => panic!("RTA_VERIFY is all, last or off, not {other:?}"),
        }
    }

    /// `None` when run `k` of `runs` is verified; else the word that skips it.
    fn skips(self, k: usize, runs: usize) -> Option<&'static str> {
        match self {
            Self::All => None,
            Self::Last if k + 1 == runs => None,
            Self::Last => Some("last"),
            Self::Off => Some("off"),
        }
    }
}

/// The harness verifies every run unless told `last` or `off`, `last` verifies
/// the last run alone, and a word it does not take refuses (a misspelt `off`
/// must not leave a run verified by surprise, nor a misspelt `all` skip it).
#[test]
fn the_rta_verify_word_verifies_by_default_and_refuses_other_words() {
    for on in [None, Some(""), Some("all"), Some(" all ")] {
        let v = RtaVerify::from_word(on);
        assert_eq!(v, RtaVerify::All, "{on:?}");
        assert!((0..3).all(|k| v.skips(k, 3).is_none()), "{on:?}");
    }
    let last = RtaVerify::from_word(Some("last"));
    assert_eq!(
        (0..3).map(|k| last.skips(k, 3)).collect::<Vec<_>>(),
        [Some("last"), Some("last"), None]
    );
    assert_eq!(last.skips(0, 1), None, "one run: it is the last");
    for off in [Some("off"), Some(" off\n")] {
        let v = RtaVerify::from_word(off);
        assert!((0..3).all(|k| v.skips(k, 3) == Some("off")), "{off:?}");
    }
    for bad in ["0", "1", "none", "Off", "skip", "all,off", "first"] {
        assert!(
            std::panic::catch_unwind(|| RtaVerify::from_word(Some(bad))).is_err(),
            "{bad:?} taken"
        );
    }
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
fn prove_first_groups<L: crate::block_whir::LocalPart>(
    plan: &WhirBlockPlan,
    proof: &BlockWhirProof<L>,
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
             before revive {} · overlap planned out {} at {:.2} of {:.2} GiB · upload once used {} ({:.2} GB, \
             ledger {:.2} GiB, planned out {} at {:.2} of {:.2} GiB, refused {}) · turn {:.3}-{:.3}s · GiB \
             promised: commit {:.2}/{:.2} · argue {:.2}/{:.2} · open {:.2}{}",
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
            s.overlap_planned_out,
            gib(s.overlap_plan_need),
            gib(s.overlap_plan_free),
            s.upload_once.used,
            s.upload_once.bytes as f64 / 1e9,
            gib(s.upload_once.after),
            s.upload_once.planned_out,
            gib(s.upload_once.need),
            gib(s.upload_once.free),
            s.upload_once.refused,
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
        if s.store_promised || s.store_promise_refused {
            println!(
                "{tag} PROMISE {g}: taken at admission {} · refused {} · spent by the store {}",
                s.store_promised, s.store_promise_refused, s.upload_once.promised
            );
        }
        if s.upload_once.used {
            let u = s.upload_once;
            println!(
                "{tag} STORE {g}: narrow {} · words {:.3} GB · crossed {:.3} GB · pack {:.4}s · up {:.4}s · \
                 packed at the commit {}",
                u.narrow,
                u.bytes as f64 / 1e9,
                if u.narrow { u.packed_bytes } else { u.bytes } as f64 / 1e9,
                u.pack_secs,
                u.upload_secs,
                u.prepacked
            );
        }
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
    // Either kind of base (local lookups are the BLAKE3 base's default),
    // held as one type.
    let prove_base = || -> Result<
        (crate::block_whir::AnyBlockWhirProof, crate::block_whir::BlockStamps),
        crate::Error,
    > {
        if cfg.format.local_lookups() {
            crate::block_whir::prove_block_whir_local_observed_led(
                elf,
                input,
                opts,
                &cfg.format,
                &cfg.options,
                &observe,
                &on_group,
                None,
            )
            .map(|(mut proof, stamps)| {
                let local = std::mem::take(&mut proof.local);
                (proof.with_local(Some(local)), stamps)
            })
        } else {
            crate::block_whir::prove_block_whir_observed_groups(
                elf,
                input,
                opts,
                &cfg.format,
                &cfg.options,
                &observe,
                &on_group,
            )
            .map(|(proof, stamps)| (proof.with_local(None), stamps))
        }
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

/// ★ No production code builds a weakened [`VerifierChecks`]: outside test
/// files, the struct is spelled with braces only where it is defined (its
/// `ALL`), so every verifier a non-test caller reaches runs every check, the
/// commit-time answers' batching included. Scans the prover's, the STARK
/// crate's and the multilinear crate's sources.
///
/// [`VerifierChecks`]: stark::multilinear_table::batched::VerifierChecks
#[test]
fn no_production_code_builds_weakened_verifier_checks() {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("a source dir") {
            let path = entry.expect("an entry").path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n != "tests") {
                    walk(&path, out);
                }
            } else if path.extension().is_some_and(|e| e == "rs")
                && !path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with("_tests.rs") || n == "tests.rs")
            {
                out.push(path);
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for dir in ["src", "../crypto/stark/src", "../crypto/multilinear/src"] {
        walk(&root.join(dir), &mut files);
    }
    assert!(
        files.len() > 100,
        "the scan found the sources: {}",
        files.len()
    );
    let mut built = Vec::new();
    for file in &files {
        let src = std::fs::read_to_string(file).expect("a source file");
        for (at, _) in src.match_indices("VerifierChecks {") {
            let line = src[..at].lines().last().unwrap_or("");
            if !line.trim_start().starts_with("pub struct")
                && !line.trim_start().starts_with("impl")
            {
                built.push(format!("{}:{}", file.display(), src[..at].lines().count()));
            }
        }
    }
    assert!(
        built.is_empty(),
        "production code builds a VerifierChecks other than ALL: {built:?}"
    );
}

/// ★ The levels' public verifier derives its own plan, in a test GitHub CI
/// runs (the one test that executes an entry is box tier). Read from the
/// source, every function on the entry's path down to the plan and the
/// level children, whitespace collapsed:
/// - both public entries go through `verify_block_levels_pinned`, and neither
///   calls `level_plan(`, reads a knob or names a root;
/// - each link of the path: the pinned body refuses the process's format knobs
///   (`check_process_format`) and calls `verifier_level_plan`, which calls
///   `verifier_level_plan_with`, which calls `level_plan_with(.., None, true,
///   Some(cap_mux), true)`: no prepared roots, the inverse share form, the
///   named cap check and the chip rows measured, before anything is emitted;
/// - `level_plan_with` derives through the share-taking derivations only (not
///   `derive_blake3(`, which reads the share knob) and passes the cap check on;
/// - every later level's child carries the plan's cap check;
/// - no function on the path reads `share_inverse()`, `cap_mux()`,
///   `plan_lean()` or a BLAKE3 base knob (`b3_base_…`), and `pin_family`
///   reads no environment;
/// - the plan-taking calls stay crate-private: `verify_levels_to` is
///   `pub(crate)`, `verify_levels` and `verify_block_levels_under` test-only.
#[test]
fn the_levels_verifier_derives_its_plan_and_takes_none() {
    let src = include_str!("levels.rs");
    let body = |name: &str| -> String {
        let at = src
            .find(&format!("fn {name}("))
            .unwrap_or_else(|| panic!("levels.rs defines {name}"));
        let rest = &src[at..];
        let end = rest.find("\n}\n").expect("a body's end");
        rest[..end].split_whitespace().collect::<Vec<_>>().join(" ")
    };
    // `last_level(` and `last_wire(` are the process's last-level knobs.
    let knobs = [
        "share_inverse()",
        "cap_mux()",
        "b3_base_",
        "std::env",
        "last_level(",
        "last_wire(",
        "plan_lean()",
    ];
    for public in ["verify_block_levels", "verify_block_levels_at"] {
        let b = body(public);
        assert!(
            src.contains(&format!("\npub fn {public}(")),
            "{public} is public"
        );
        assert!(
            b.contains("verify_block_levels_pinned("),
            "{public} verifies through the pinned body"
        );
        for forbidden in knobs.iter().chain(&["level_plan(", "Some("]) {
            assert!(!b.contains(forbidden), "{public} reads {forbidden}");
        }
    }
    let links = [
        (
            "verify_block_levels_pinned",
            "check_process_format(family, last_level)?;",
        ),
        ("verify_block_levels_pinned", "verifier_level_plan("),
        (
            "verify_block_levels_pinned",
            "verify_levels_to(&plan, family, stop, last_level, last)",
        ),
        (
            "verify_levels_to",
            "derive_levels_last(plan, family, stop, last_level, true)",
        ),
        ("verifier_level_plan", "verifier_level_plan_with("),
        ("verifier_level_plan_with", "level_plan_with("),
        ("verifier_level_plan_with", "None, true, Some(cap_mux)"),
        // A verifier's plan measures its chip rows whatever the prover's
        // lean-plan knob says.
        ("verifier_level_plan_with", "Some(cap_mux), true, )"),
        ("level_plan_with", "measure, share_inverse, cap_mux, )"),
        ("level_plan_with", "derive_full("),
        ("level_plan_with", "derive_blake3_with_share("),
        ("level_plan_with", "share_inverse, cap_mux, )"),
        ("child_of_build", "cap_mux: plan.cap_mux()"),
        ("check_process_format", "check_format_knobs("),
    ];
    // The argument-less entry's constants: its pinned choices.
    for constant in [
        "BlockFormat::pinned_with(GroupFold::Indexed)",
        "B3Rounds::Fri",
        "LevelStop::DEFAULT",
        "LastLevel::pinned(family)",
        "CapMux::Packed",
    ] {
        assert!(
            body("verify_block_levels").contains(constant),
            "verify_block_levels passes {constant}"
        );
    }
    // The file entry: read for the family's pinned format, then the pinned
    // entry; no knob read.
    assert!(src.contains("\npub fn verify_block_levels_file("));
    let file = body("verify_block_levels_file");
    for call in [
        "read_last_level_file(file, LastLevel::pinned(family))?",
        "verify_block_levels(elf, statement, family, &last)",
    ] {
        assert!(
            file.contains(call),
            "verify_block_levels_file must contain `{call}`"
        );
    }
    for knob in knobs {
        assert!(
            !file.contains(knob),
            "verify_block_levels_file reads {knob}"
        );
    }
    // The derivations that read the process's last-level format are test
    // only.
    for derive in ["derive_levels", "derive_levels_to"] {
        assert!(
            src.contains(&format!("#[cfg(test)]\npub(crate) fn {derive}(")),
            "{derive} is test only"
        );
    }
    for (function, call) in links {
        assert!(
            body(function).contains(call),
            "{function} must contain `{call}`"
        );
    }
    assert!(
        !body("level_plan_with").contains("derive_blake3("),
        "level_plan_with derives through the share-reading derive_blake3"
    );
    // A knob is read by a bare call; `plan.cap_mux()` reads the plan's, and
    // `is_last_level(` is another function: a match counts only where no `.`
    // and no identifier character comes before it.
    let reads = |b: &str, knob: &str| {
        b.match_indices(knob).any(|(at, _)| {
            !b[..at].ends_with(|c: char| c == '.' || c == '_' || c.is_ascii_alphanumeric())
        })
    };
    assert!(
        reads("x = last_level(family)", "last_level(")
            && reads("super::last_level(family)", "last_level(")
            && !reads("is_last_level(stop)", "last_level(")
            && !reads("plan.cap_mux()", "cap_mux()"),
        "the scan tells a knob's call from a method and from a longer name"
    );
    for function in [
        "verify_block_levels",
        "verify_block_levels_at",
        "verify_block_levels_pinned",
        "verifier_level_plan",
        "verifier_level_plan_with",
        "level_plan_with",
        "verify_levels_to",
        "derive_levels_last",
        "build_level",
        "prepare_as_level",
        "child_of_build",
        "pin_family",
    ] {
        let b = body(function);
        for knob in knobs {
            assert!(!reads(&b, knob), "{function} reads {knob}");
        }
    }
    assert!(src.contains("\npub(crate) fn verify_levels_to("));
    assert!(src.contains("#[cfg(test)]\npub(crate) fn verify_levels("));
    assert!(src.contains("#[cfg(test)]\npub(crate) fn verify_block_levels_under("));
    assert!(!src.contains("\npub fn verify_levels"));
}

/// One binary, one format: the public entries' refusal of the process's
/// format knobs, on given values. The BLAKE3 base's cap height and final
/// polynomial must be the pinned ones, and its local lookups on (unset or
/// `keccak`; `off` refused by name); under the Poseidon1 base the `LFM_HASH`
/// split must be off and local lookups not asked for; under BLAKE3 the split
/// is inert and taken.
#[test]
fn the_levels_verifier_refuses_format_knobs_off_their_pinned_values() {
    use super::levels::check_format_knobs;
    use multilinear::whir_chain::FinalPoly;
    let cap = crate::block_whir::B3_BASE_CAP;
    let on = FinalPoly::Vars(crate::block_whir::B3_BASE_FINAL_VARS);
    assert!(check_format_knobs(WlfmFamily::B3, cap, on, false, None).is_ok());
    assert!(check_format_knobs(WlfmFamily::B3, cap, on, true, None).is_ok());
    assert!(check_format_knobs(WlfmFamily::B3, cap, on, false, Some(true)).is_ok());
    assert!(check_format_knobs(WlfmFamily::P1, cap, on, false, None).is_ok());
    assert!(check_format_knobs(WlfmFamily::P1, cap, on, false, Some(false)).is_ok());
    for (what, refused, knob) in [
        (
            "cap 5",
            check_format_knobs(WlfmFamily::B3, 5, on, false, None),
            "LAMBDA_VM_B3_BASE_CAP",
        ),
        (
            "cap 3",
            check_format_knobs(WlfmFamily::B3, 3, on, false, None),
            "LAMBDA_VM_B3_BASE_CAP",
        ),
        (
            "no final polynomial",
            check_format_knobs(WlfmFamily::B3, cap, FinalPoly::Off, false, None),
            "LAMBDA_VM_B3_BASE_FINAL",
        ),
        (
            "local lookups off",
            check_format_knobs(WlfmFamily::B3, cap, on, false, Some(false)),
            "LAMBDA_VM_B3_BASE_LOCAL",
        ),
        (
            "the hash split",
            check_format_knobs(WlfmFamily::P1, cap, on, true, None),
            "LAMBDA_VM_LFM_HASH_SPLIT",
        ),
        (
            "local lookups on the Poseidon1 base",
            check_format_knobs(WlfmFamily::P1, cap, on, false, Some(true)),
            "LAMBDA_VM_B3_BASE_LOCAL",
        ),
    ] {
        let refused = refused.expect_err(what);
        assert!(refused.contains(knob), "{what}: {refused}");
    }
}

/// The scenarios [`the_levels_verifier_reads_no_knob_in_a_process_off_the_defaults`]
/// runs in processes of their own: an environment, and its child test's name.
const KNOB_CHILDREN: [(&str, &[(&str, &str)]); 8] = [
    (
        "cap",
        &[
            ("LAMBDA_VM_B3_BASE_CAP", "5"),
            ("LAMBDA_VM_B3_BASE_GROUP", "3"),
            ("LAMBDA_VM_B3_BASE_GKR", "single"),
            ("LAMBDA_VM_LFM_HASH_SPLIT", "1"),
        ],
    ),
    ("final", &[("LAMBDA_VM_B3_BASE_FINAL", "off")]),
    ("local-off", &[("LAMBDA_VM_B3_BASE_LOCAL", "off")]),
    ("local-word", &[("LAMBDA_VM_B3_BASE_LOCAL", "sometimes")]),
    (
        "malformed",
        &[
            ("LAMBDA_VM_B3_BASE_CAP", "banana"),
            ("LAMBDA_VM_B3_BASE_FINAL", "sometimes"),
            ("LAMBDA_VM_LFM_HASH_SPLIT", "maybe"),
        ],
    ),
    (
        "last-level",
        &[
            ("LAMBDA_VM_LAST_LEVEL_EVALUATED", "0"),
            ("LAMBDA_VM_LAST_LEVEL_FIRST_FOLD", "off"),
            ("LAMBDA_VM_LAST_LEVEL_FINAL", "off"),
        ],
    ),
    ("last-level-on", &[("LAMBDA_VM_LAST_LEVEL_EVALUATED", "1")]),
    ("last-level-word", &[("LAMBDA_VM_LAST_LEVEL_FINAL", "5")]),
];

/// ★ In processes whose knobs are off the defaults, the public entries'
/// pins hold: the family pin keeps the BLAKE3 base's own group size and GKR
/// ladder, and the format check refuses the cap, the final polynomial, local
/// lookups off and (under Poseidon1) the hash split by name, and a malformed
/// word of any of them is an `Err` under either family, never an abort or a
/// panic.
/// Each scenario runs
/// [`the_levels_pins_hold_under_knobs_off_the_default`] in a child process.
#[test]
fn the_levels_verifier_reads_no_knob_in_a_process_off_the_defaults() {
    let name = "lfm::levels_tests::the_levels_pins_hold_under_knobs_off_the_default";
    for (scenario, env) in KNOB_CHILDREN {
        let out = std::process::Command::new(std::env::current_exe().expect("the test binary"))
            .args([
                "--exact",
                name,
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .envs(env.iter().copied())
            .env("LEVELS_KNOB_SCENARIO", scenario)
            .output()
            .expect("the test binary runs");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success() && stdout.contains("1 passed"),
            "the child under the {scenario} knobs:\n{stdout}\n{stderr}"
        );
    }
}

/// Run by [`the_levels_verifier_reads_no_knob_in_a_process_off_the_defaults`]
/// in a process whose environment is one of [`KNOB_CHILDREN`]; refuses to run
/// anywhere else.
#[test]
#[ignore = "run in a process of its own by the_levels_verifier_reads_no_knob_in_a_process_off_the_defaults"]
fn the_levels_pins_hold_under_knobs_off_the_default() {
    use crate::block_whir::{B3_BASE_GKR_DEFAULT, B3_BASE_GROUP_POLYS, B3Rounds, BlockFormat};
    use multilinear::whir_chain::GroupFold;
    let scenario = std::env::var("LEVELS_KNOB_SCENARIO").expect("run by its parent");
    let pinned = BlockFormat::pinned_with(GroupFold::Indexed);
    let b3 = super::levels::pin_family(pinned, WlfmFamily::B3, B3Rounds::Fri).expect("the pin");
    match scenario.as_str() {
        "cap" => {
            assert_eq!(crate::block_whir::b3_base_group_polys(), 3, "the knob took");
            assert_eq!(b3.group_polys, B3_BASE_GROUP_POLYS, "the pinned group size");
            assert_eq!(b3.argue, pinned.argue.with_ladder(B3_BASE_GKR_DEFAULT));
            let refused = super::levels::check_process_format_for_tests(WlfmFamily::B3)
                .expect_err("a cap off the pinned one");
            assert!(refused.contains("LAMBDA_VM_B3_BASE_CAP"), "{refused}");
            let refused = super::levels::check_process_format_for_tests(WlfmFamily::P1)
                .expect_err("the hash split under Poseidon1");
            assert!(refused.contains("LAMBDA_VM_LFM_HASH_SPLIT"), "{refused}");
        }
        "final" => {
            let refused = super::levels::check_process_format_for_tests(WlfmFamily::B3)
                .expect_err("no final polynomial");
            assert!(refused.contains("LAMBDA_VM_B3_BASE_FINAL"), "{refused}");
            super::levels::check_process_format_for_tests(WlfmFamily::P1)
                .expect("the Poseidon1 base reads no BLAKE3 knob");
        }
        "local-off" => {
            assert_eq!(
                b3.keccak_rnd,
                crate::block_whir::B3_BASE_KECCAK_RND,
                "the pinned form"
            );
            let refused = super::levels::check_process_format_for_tests(WlfmFamily::B3)
                .expect_err("local lookups off");
            assert!(refused.contains("LAMBDA_VM_B3_BASE_LOCAL"), "{refused}");
            super::levels::check_process_format_for_tests(WlfmFamily::P1)
                .expect("the Poseidon1 base takes local lookups off");
        }
        "local-word" => {
            for family in [WlfmFamily::B3, WlfmFamily::P1] {
                let refused = super::levels::check_process_format_for_tests(family)
                    .expect_err("a malformed local-lookups word");
                assert!(
                    refused.contains("LAMBDA_VM_B3_BASE_LOCAL"),
                    "{family:?}: {refused}"
                );
            }
        }
        "malformed" => {
            for family in [WlfmFamily::B3, WlfmFamily::P1] {
                let refused = super::levels::check_process_format_for_tests(family)
                    .expect_err("a malformed knob word");
                assert!(
                    refused.contains("LAMBDA_VM_B3_BASE_CAP"),
                    "{family:?}: {refused}"
                );
            }
        }
        "last-level" => {
            // Every lever turned off: the knobs took under both families; the
            // BLAKE3 verifier, naming its pinned last level (every lever on),
            // refuses the process; the Poseidon1 verifier's pinned last level
            // is every lever off, so it passes; one naming the knobs' format
            // passes the last-level check.
            use super::last_level::{LastLevel, last_level};
            for family in [WlfmFamily::B3, WlfmFamily::P1] {
                let named = last_level(family).expect("the knobs read");
                assert_eq!(named, LastLevel::OFF, "{family:?}");
                super::levels::check_last_level_knobs(named, named).expect("the named format");
            }
            let refused = super::levels::check_process_format_for_tests(WlfmFamily::B3)
                .expect_err("a last level off the pinned BLAKE3 one");
            assert!(
                refused.contains("the process asks for the last level"),
                "{refused}"
            );
            super::levels::check_process_format_for_tests(WlfmFamily::P1)
                .expect("the Poseidon1 levels' pinned last level is every lever off");
        }
        "last-level-on" => {
            // One lever asked for: under BLAKE3 it is the pinned value already
            // (the process is the pinned format, accepted); under Poseidon1
            // it is a lever on, and the verifier, pinned at every lever off,
            // refuses the process.
            use super::last_level::{LastLevel, last_level};
            assert_eq!(last_level(WlfmFamily::B3), Ok(LastLevel::ON));
            super::levels::check_process_format_for_tests(WlfmFamily::B3)
                .expect("the pinned BLAKE3 format");
            let refused = super::levels::check_process_format_for_tests(WlfmFamily::P1)
                .expect_err("a lever on under Poseidon1");
            assert!(
                refused.contains("the process asks for the last level"),
                "{refused}"
            );
        }
        "last-level-word" => {
            for family in [WlfmFamily::B3, WlfmFamily::P1] {
                let refused = super::levels::check_process_format_for_tests(family)
                    .expect_err("a malformed last-level word");
                assert!(
                    refused.contains("LAMBDA_VM_LAST_LEVEL_FINAL"),
                    "{family:?}: {refused}"
                );
            }
        }
        other => panic!("no scenario {other}"),
    }
}

/// The pinned format a family's verifier checks against: the BLAKE3 base's
/// own constants (group size, GKR ladder, KECCAK_RND's local form) and the
/// named round form on `B3`; a round form on `P1` refused, and `P1` keeps the
/// preset's lean form.
#[test]
fn a_levels_verifier_pins_its_familys_base() {
    use crate::block_whir::{B3_BASE_GROUP_POLYS, B3Rounds, BlockFormat};
    use multilinear::whir_chain::GroupFold;
    let pinned = BlockFormat::pinned_with(GroupFold::Indexed);
    let b3 =
        super::levels::pin_family(pinned, WlfmFamily::B3, B3Rounds::WHIR3).expect("the BLAKE3 pin");
    assert_eq!(b3.zf.base, BaseFormat::B3_WHIR);
    assert_eq!(b3.group_polys, B3_BASE_GROUP_POLYS);
    assert_eq!(b3.b3_rounds, B3Rounds::WHIR3);
    assert_eq!(
        b3.argue,
        pinned
            .argue
            .with_ladder(crate::block_whir::B3_BASE_GKR_DEFAULT)
    );
    assert_eq!(b3.keccak_rnd, crate::block_whir::B3_BASE_KECCAK_RND);
    assert!(b3.local_lookups(), "the BLAKE3 pin argues local lookups");
    let p1 = super::levels::pin_family(pinned, WlfmFamily::P1, B3Rounds::Fri)
        .expect("the Poseidon1 pin");
    assert_eq!(p1.zf.base, pinned.zf.base);
    assert_eq!(p1.keccak_rnd, pinned.keccak_rnd);
    assert!(!p1.local_lookups(), "the Poseidon1 pin argues none");
    assert!(
        super::levels::pin_family(pinned, WlfmFamily::P1, B3Rounds::WHIR3).is_err(),
        "a round form on the Poseidon1 base"
    );
    // A round form the chains cannot take is refused, never a panic.
    for shrink in [0u8, 4, u8::MAX] {
        for caps_by_q in [false, true] {
            assert!(
                super::levels::pin_family(
                    pinned,
                    WlfmFamily::B3,
                    B3Rounds::Whir { shrink, caps_by_q }
                )
                .is_err(),
                "shrink {shrink}"
            );
        }
    }
    for shrink in 1..=3u8 {
        super::levels::pin_family(
            pinned,
            WlfmFamily::B3,
            B3Rounds::Whir {
                shrink,
                caps_by_q: true,
            },
        )
        .unwrap_or_else(|e| panic!("shrink {shrink}: {e}"));
    }
}

#[cfg(feature = "test-security")]
#[test]
fn two_compact_blake3_levels_verify_at_test_security() {
    use super::builder::{B3Block, B3Cv, LfmBuilder};
    use super::compiler::compile;
    use super::last_level::LastLevel;
    use super::levels::build_level;
    use super::wlfm::{WlfmProving, prove_wlfm_as};
    use super::wlfm_leg::{WlfmChild, wlfm_leg_arena, wlfm_leg_program};
    use crate::tables::types::FE;

    let family = WlfmFamily::B3;
    let mut builder = LfmBuilder::new()
        .with_wrap_hash(WrapHash::Blake3)
        .with_b3_lean(true);
    let zero = builder.digest_const([FE::zero(); 4]).as_cell();
    let digest = builder.b3_compress(
        B3Cv::Iv,
        B3Block::Lanes([zero; 4]),
        0,
        crypto::hash::blake3::chain::FLAGS_ONE_BLOCK as u8,
    );
    for word in digest {
        builder.public(word);
    }
    let first = build_level(
        compile(builder.finish()),
        family,
        None,
        false,
        (LevelStop::DEFAULT, 1),
        LastLevel::OFF,
    )
    .expect("build level 1");
    let first_proof = prove_wlfm_as(&first.program, &first.build, &[], WlfmProving::Streamed)
        .expect("prove level 1");
    verify_wlfm(
        &first.build.artifacts,
        &first_proof,
        &first_proof.public_words,
    )
    .expect("verify level 1");

    let child = WlfmChild {
        artifacts: first.build.artifacts,
        num_public_words: first_proof.public_words.len(),
        cap_mux: Some(super::merkle_cap::CapMux::Packed),
    };
    let program = wlfm_leg_program(&child).expect("emit the level 2 verifier");
    let arena = wlfm_leg_arena(&child, &first_proof).expect("encode the level 1 proof");
    let second = build_level(
        program,
        family,
        None,
        false,
        (LevelStop::DEFAULT, 2),
        LastLevel::OFF,
    )
    .expect("build level 2");
    let second_proof = prove_wlfm_as(
        &second.program,
        &second.build,
        &[arena],
        WlfmProving::Streamed,
    )
    .expect("prove level 2");
    assert_eq!(
        second_proof.public_words, first_proof.public_words,
        "level 2 republishes the verified level 1 words"
    );
    verify_wlfm(
        &second.build.artifacts,
        &second_proof,
        &first_proof.public_words,
    )
    .expect("verify level 2");
}

#[cfg(feature = "test-security")]
#[test]
fn the_pinned_blake3_base_and_both_levels_verify_at_test_security() {
    let family = WlfmFamily::B3;
    let format = super::levels::pin_family(
        BlockFormat::pinned_with(multilinear::whir_chain::GroupFold::Indexed),
        family,
        crate::block_whir::B3Rounds::Fri,
    )
    .expect("the pinned BLAKE3 format");
    let opts = super::proof::block_base_options_under(&format.zf);
    let elf = crate::test_utils::asm_elf_bytes("basic_program");
    let run = super::levels::prove_block_levels(
        &elf,
        &[],
        &opts,
        &format,
        &streamed_options(),
        family,
        LevelStop::DEFAULT,
        true,
        true,
    )
    .expect("the block proves through both levels");
    assert!(
        crate::base_only::with_permit(|| crate::block_whir::verify_block_whir(
            &run.base_proof,
            &elf,
            &opts,
            &format
        )
        .expect("base verification runs")),
        "the BLAKE3 base verifies"
    );
    assert_eq!(run.levels.len(), 2, "the driver builds both levels");
    assert_eq!(run.proofs.len(), 2, "the driver proves both levels");
    for (index, (level, proof)) in run.levels.iter().zip(&run.proofs).enumerate() {
        verify_wlfm(&level.build.artifacts, proof, &proof.public_words)
            .unwrap_or_else(|e| panic!("level {} verifies: {e}", index + 1));
    }
    super::levels::verify_block_levels(&elf, run.base_proof.statement(), family, &run.proofs[1])
        .expect("the public verifier accepts the last level");
}

/// ★ A public entry accepts: a small block with a keccak call (`test_keccak`,
/// so its statement holds a KECCAK_RND group closed by its LOCAL_BYTES table
/// and the pinned format argues that group's lookups locally) proved at the format
/// `verify_block_levels` pins (the tree's pinned format on the BLAKE3 base),
/// under the options it derives, through its two levels, and the last level
/// verified by `verify_block_levels` itself — so the entry's own constants
/// (the format, the round form, the cap check, the stop) are the ones an
/// honest proof is made at. Another program's last proof is refused at the
/// last level's verify, after the whole derivation. And no level program the
/// entry's plan derives has a hash row a split could cut.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels at the pinned format; box tier"]
fn the_public_levels_verifier_accepts_at_the_pinned_format() {
    use crate::block_whir::{B3Rounds, BlockFormat};
    use multilinear::whir_chain::GroupFold;
    let format = super::levels::pin_family(
        BlockFormat::pinned_with(GroupFold::Indexed),
        WlfmFamily::B3,
        B3Rounds::Fri,
    )
    .expect("the pin");
    let opts = super::proof::block_base_options_under(&format.zf);
    assert!(
        format.local_lookups(),
        "the pinned BLAKE3 format argues local lookups"
    );
    let elf = crate::test_utils::asm_elf_bytes("test_keccak");
    // The pinned format's KECCAK_RND lean byte checks need the windowed build.
    let run = super::levels::prove_block_levels(
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
    .expect("the block proves to its last level at the pinned format");
    let local_groups = run
        .base_proof
        .local
        .as_ref()
        .expect("a base of the local kind")
        .iter()
        .filter(|argue| !argue.is_empty())
        .count();
    println!(
        "PUBLIC LEVELS: {} groups, {local_groups} local",
        run.base_proof.groups.len()
    );
    assert!(
        local_groups >= 1,
        "the base argues at least one local group"
    );
    let last = run.proofs.last().expect("a last level");
    super::levels::verify_block_levels(&elf, run.base_proof.statement(), WlfmFamily::B3, last)
        .expect("the public entry accepts the honest last level");
    let plan = super::levels::verifier_level_plan(
        &elf,
        &format,
        run.base_proof.statement(),
        WlfmFamily::B3,
        super::merkle_cap::CapMux::Packed,
    )
    .expect("the verifier's plan derives");
    assert_eq!(plan.cap_mux(), Some(super::merkle_cap::CapMux::Packed));
    let levels = derive_levels_to(&plan, WlfmFamily::B3, LevelStop::DEFAULT, true)
        .expect("the levels derive");
    for (k, level) in levels.iter().enumerate() {
        assert_eq!(
            level.program.groups.hash.real_rows,
            0,
            "level {}: a BLAKE3 level program's hash rows",
            k + 1
        );
    }
    let foreign = super::wlfm::WlfmProof {
        proof: stark::multilinear_table::MultiProof {
            roots: Vec::new(),
            tables: Vec::new(),
            columns: Vec::new(),
            preprocessed: None,
        },
        argues: Vec::new(),
        prepared: Vec::new(),
        commit_ood: Vec::new(),
        public_words: Vec::new(),
    };
    let refused = super::levels::verify_block_levels(
        &elf,
        run.base_proof.statement(),
        WlfmFamily::B3,
        &foreign,
    )
    .expect_err("another program's proof");
    assert!(
        refused.starts_with("the last level does not verify"),
        "the derivation ran to the last level's verify: {refused}"
    );
}

/// [`super::levels::verify_block_levels_at`] with every named choice at the
/// pinned entry's value (the tree's pinned format, `Indexed`, on the BLAKE3
/// base, its FRI round form, the packed cap check, the default stop) and the
/// last level's format `last_level`.
fn verify_at_pinned(
    elf: &[u8],
    statement: crate::block_whir::BlockStatement<'_>,
    last_level: super::last_level::LastLevel,
    last: &super::wlfm::WlfmProof,
) -> Result<(), String> {
    use crate::block_whir::{B3Rounds, BlockFormat};
    use multilinear::whir_chain::{ArgueFormat, GroupFold};
    let f = BlockFormat::pinned_with(GroupFold::Indexed);
    let blowup = match f.zf.whir_blowup {
        2 => crate::zf_format::WhirBlowup::Two,
        4 => crate::zf_format::WhirBlowup::Four,
        other => return Err(format!("a pinned blowup of {other}")),
    };
    let ArgueFormat::Batched { pack, .. } = f.argue else {
        return Err("the pinned argue is batched".into());
    };
    super::levels::verify_block_levels_at(
        elf,
        statement,
        WlfmFamily::B3,
        LevelStop::DEFAULT,
        last_level,
        last,
        GroupFold::Indexed,
        blowup,
        f.zf.whir_leaves,
        f.zf.whir_trees,
        f.leaf_scale,
        f.leaf_query,
        pack,
        f.memory,
        f.page_run_log2,
        B3Rounds::Fri,
        super::merkle_cap::CapMux::Packed,
    )
}

/// ★ The BLAKE3 levels' public verifier names the pinned last level (every
/// lever on), and the format is a NAMED constant of the public entries: a
/// small block proved at the pinned format twice, with every knob unset (the
/// pinned last level) and with every lever off (the last level's format
/// before the flip). In the default process the argument-less entry, the
/// named entry at the pinned format and the file entry accept the pinned
/// proof and refuse the off one: the proof entries AT THE LAST LEVEL'S
/// VERIFY, naming the format they check, the file entry from the header,
/// naming both formats, before anything is decoded. A verifier built for the
/// off format (the named entry at it, in a process whose knobs name it)
/// accepts the off proof and refuses the pinned one; and a process whose
/// knobs differ from the named format is refused before anything is
/// derived.
#[test]
#[ignore = "proves a small BLAKE3 block and its levels twice at the pinned format; box tier"]
fn the_public_entries_name_the_last_levels_format() {
    use super::last_level::{LastLevel, last_level_file, set_last_level_for_test};
    use super::wire::WireFormat;
    use crate::block_whir::{B3Rounds, BlockFormat};
    use multilinear::whir_chain::GroupFold;
    let pinned = LastLevel::pinned(WlfmFamily::B3);
    assert_eq!(pinned, LastLevel::ON);
    let off = LastLevel::OFF;
    let format = super::levels::pin_family(
        BlockFormat::pinned_with(GroupFold::Indexed),
        WlfmFamily::B3,
        B3Rounds::Fri,
    )
    .expect("the pin");
    let opts = super::proof::block_base_options_under(&format.zf);
    let elf = crate::test_utils::asm_elf_bytes("test_keccak");
    let prove = || {
        super::levels::prove_block_levels(
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
    };
    // Every knob unset: the pinned format.
    let new = prove().unwrap_or_else(|e| panic!("the block proves at the pinned last level: {e}"));
    let old = {
        let _off = set_last_level_for_test(off);
        prove().unwrap_or_else(|e| panic!("the block proves with every lever off: {e}"))
    };
    let (n_statement, o_statement) = (new.base_proof.statement(), old.base_proof.statement());
    let (n_last, o_last) = (
        new.proofs.last().expect("the pinned last level"),
        old.proofs.last().expect("the off last level"),
    );
    assert_eq!(
        new.levels[1].build.artifacts.prep,
        super::wlfm::WlfmPrep::Evaluated
    );
    assert_eq!(
        old.levels[1].build.artifacts.prep,
        super::wlfm::WlfmPrep::Stacked
    );
    // The default process: the pinned entries accept the pinned proof and
    // refuse the off one at the last level's verify, naming their format.
    super::levels::verify_block_levels(&elf, n_statement, WlfmFamily::B3, n_last)
        .expect("the pinned entry accepts the pinned last level");
    verify_at_pinned(&elf, n_statement, pinned, n_last)
        .expect("the named entry at the pinned format accepts the pinned last level");
    for (name, e) in [
        (
            "the pinned entry",
            super::levels::verify_block_levels(&elf, o_statement, WlfmFamily::B3, o_last)
                .expect_err("the off last level under the pinned verifier"),
        ),
        (
            "the named entry at the pinned format",
            verify_at_pinned(&elf, o_statement, pinned, o_last)
                .expect_err("the off last level under the pinned verifier"),
        ),
    ] {
        println!("PINNED VERIFIER, OFF PROOF ({name}): {e}");
        assert!(
            e.starts_with("the last level does not verify") && e.contains(&pinned.to_string()),
            "{name}: {e}"
        );
    }
    // The file entry: one version, one format.
    for wire in [WireFormat::Dedup, WireFormat::Rkyv] {
        let n_file = last_level_file(n_last, pinned, wire).expect("the pinned file");
        let o_file = last_level_file(o_last, off, wire).expect("the off file");
        super::levels::verify_block_levels_file(&elf, n_statement, WlfmFamily::B3, &n_file)
            .unwrap_or_else(|e| panic!("{wire:?}: the file entry accepts the pinned file: {e}"));
        let e = super::levels::verify_block_levels_file(&elf, o_statement, WlfmFamily::B3, &o_file)
            .expect_err("the off file under the pinned verifier");
        println!("PINNED FILE VERIFIER, OFF FILE ({wire:?}): {e}");
        assert!(
            e.contains("format 0 (") && e.contains("format 7 ("),
            "{wire:?}: {e}"
        );
    }
    // A verifier built for the off format: it accepts the off proof and
    // refuses the pinned one at the last level's verify; the pinned entry
    // refuses its process before deriving anything.
    {
        let _off = set_last_level_for_test(off);
        verify_at_pinned(&elf, o_statement, off, o_last)
            .expect("the named entry at the off format accepts the off proof");
        let e = verify_at_pinned(&elf, n_statement, off, n_last)
            .expect_err("the pinned last level under the off verifier");
        println!("OFF VERIFIER, PINNED PROOF: {e}");
        assert!(e.starts_with("the last level does not verify"), "{e}");
        let e = super::levels::verify_block_levels(&elf, n_statement, WlfmFamily::B3, n_last)
            .expect_err("an off process under the pinned entry");
        assert!(e.contains("the process asks for the last level"), "{e}");
    }
    // The default process, the named off format: refused before any
    // derivation.
    let e = verify_at_pinned(&elf, o_statement, off, o_last)
        .expect_err("a process whose knobs are not the named format");
    assert!(e.contains("the process asks for the last level"), "{e}");
}
