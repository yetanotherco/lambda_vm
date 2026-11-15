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
use super::wlfm::{WlfmFamily, verify_wlfm};
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
    let levels = derive_levels(&plan, WlfmFamily::B3, false).expect("the levels derive");
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
    let options = small_block_options();
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
/// 1 executed and every level built (prepared stacks on the host) beside its
/// phase B, then both levels proved under the BLAKE3 family.
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
        &small_block_options(),
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
/// the control), `RTA_BUILD_BESIDE` (1: every level built on the host beside
/// phase B; 0, the default: on the card after the base), `RTA_LEVELS` (the
/// chain's length; 2). Prints `RTA …` lines; everything is verified off the
/// clock.
#[test]
#[ignore = "a real block and its levels: box only"]
fn rta_a_real_block_through_its_levels() {
    use super::levels::prove_block_levels;
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
    let opts = super::proof::block_base_options();
    println!(
        "RTA CONFIG: base {:?} · family {family:?} · levels {levels} · beside {beside} · build \
         beside {build_beside}",
        cfg.format.zf.base.hash
    );
    let run = prove_block_levels(
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
    .unwrap_or_else(|e| panic!("{e}"));
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
    for (k, (l, (level, p))) in t
        .levels
        .iter()
        .zip(run.levels.iter().zip(&run.proofs))
        .enumerate()
    {
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(p).map_or(0, |b| b.len());
        println!(
            "RTA LEVEL {}: groups {} · instructions {} · compressions {} · execute {:.3}s · fill \
             {:.3}s · prove {:.3}s · end {:.3}s · proof {bytes} B",
            k + 1,
            level.build.artifacts.sizes.len(),
            level.program.instrs.len(),
            level.program.groups.b3l.real_rows,
            l.executed - l.execute_from,
            l.fill,
            l.prove,
            l.end
        );
    }
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
    assert!(ok_base && ok_levels && ok_last, "everything verifies");
    println!("RTA DONE");
}
