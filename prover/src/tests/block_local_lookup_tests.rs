//! Local lookups argued per group on real blocks
//! ([`BlockFormat::local_lookups`], KECCAK_RND's [`KeccakRndForm::LeanLocal`]):
//! the knob; a block with KECCAK_RND proves and verifies, each group with a
//! KECCAK_RND table closing with its LOCAL_BYTES table; each format's
//! verifier refuses the other's proof; today's proof archives as it did
//! before the local field; and the frame refuses LOCAL_BYTES tables out of
//! place or at another height.
//!
//! Every test that proves a block is `#[ignore]`d (box tier): run them with
//! `--ignored --test-threads=1`, which shares one proof per format.

use crate::block_whir::{
    self, BlockFormat, BlockWhirProof, Deviations, LocalArgues, b3_base_local_setting,
    block_format_tag, local_lane_setting, prove_block_whir_local_with, prove_block_whir_with,
    verify_block_whir,
};
use crate::tables::keccak_rnd::KeccakRndForm;
use crate::tables::local_bytes;
use crate::test_utils::asm_elf_bytes;
use crate::tests::block_whir_p1_tests::{many_groups, options};
use multilinear::whir_chain::ArgueFormat;
use stark::proof::options::{BaseFormat, ProofOptions};

fn format(keccak_rnd: KeccakRndForm) -> BlockFormat {
    BlockFormat {
        argue: ArgueFormat::BATCHED,
        keccak_rnd,
        ..many_groups(BaseFormat::P1_WHIR)
    }
}

fn local() -> BlockFormat {
    format(KeccakRndForm::LeanLocal)
}

fn lean() -> BlockFormat {
    format(KeccakRndForm::Lean)
}

fn opts() -> ProofOptions {
    ProofOptions::default_test_options()
}

fn block_options() -> block_whir::BlockOptions {
    let mut o = options(0);
    o.window_log2 = Some(3);
    o
}

fn elf() -> Vec<u8> {
    asm_elf_bytes("test_keccak")
}

/// The block proved once under local lookups, shared by every test here.
fn prove_local(_elf: &[u8]) -> BlockWhirProof<LocalArgues> {
    static PROOF: std::sync::OnceLock<BlockWhirProof<LocalArgues>> = std::sync::OnceLock::new();
    PROOF
        .get_or_init(|| {
            let t = std::time::Instant::now();
            let proof = prove_block_whir_local_with(
                &elf(),
                &[],
                &opts(),
                &local(),
                &block_options(),
                &Deviations::default(),
            )
            .expect("prove")
            .0;
            eprintln!("local prove: {:.1}s", t.elapsed().as_secs_f64());
            proof
        })
        .clone()
}

/// The block proved once in today's lean format, shared.
fn prove_lean(_elf: &[u8]) -> BlockWhirProof {
    static PROOF: std::sync::OnceLock<BlockWhirProof> = std::sync::OnceLock::new();
    PROOF
        .get_or_init(|| {
            let t = std::time::Instant::now();
            let proof = prove_block_whir_with(
                &elf(),
                &[],
                &opts(),
                &lean(),
                &block_options(),
                &Deviations::default(),
            )
            .expect("prove")
            .0;
            eprintln!("lean prove: {:.1}s", t.elapsed().as_secs_f64());
            proof
        })
        .clone()
}

fn verify<L: block_whir::LocalPart>(
    proof: &BlockWhirProof<L>,
    elf: &[u8],
    f: &BlockFormat,
) -> bool {
    verify_block_whir(proof, elf, &opts(), f).unwrap_or(false)
}

/// `LAMBDA_VM_B3_BASE_LOCAL`: unset is the base's default (on for the
/// BLAKE3 base, `B3_BASE_LOCAL_DEFAULT`), `0` or `off` is off, `keccak` argues
/// KECCAK_RND's byte lookups per group; anything else is refused. The
/// BLAKE3 base's default form is the local one, and the pinned (Poseidon1)
/// preset's stays lean.
#[test]
fn the_local_knob_reads_off_or_keccak() {
    use crate::block_whir::{B3_BASE_KECCAK_RND, B3_BASE_LOCAL_DEFAULT};
    use crate::tables::keccak_rnd::KeccakRndForm;
    assert_eq!(b3_base_local_setting(None), Ok(None));
    assert_eq!(b3_base_local_setting(Some("0")), Ok(Some(false)));
    assert_eq!(b3_base_local_setting(Some(" off ")), Ok(Some(false)));
    assert_eq!(b3_base_local_setting(Some("keccak")), Ok(Some(true)));
    for bad in ["", "1", "on", "all", "lean"] {
        assert!(b3_base_local_setting(Some(bad)).is_err(), "{bad:?}");
    }
    const { assert!(B3_BASE_LOCAL_DEFAULT) };
    assert_eq!(B3_BASE_KECCAK_RND, KeccakRndForm::LeanLocal);
    assert!(!BlockFormat::pinned().local_lookups());
    assert!(local().local_lookups());
    assert!(!lean().local_lookups());
    // The test process sets no knob and keeps the Poseidon1 base: its
    // production format argues none (the BLAKE3 base's default is pinned by
    // `levels_tests::a_levels_verifier_pins_its_familys_base`).
    assert_eq!(
        crate::block_whir::b3_base_local(),
        None,
        "no test sets the knob"
    );
    assert!(!BlockFormat::production().local_lookups());
}

/// `LAMBDA_VM_LOCAL_LANE`: unset, empty or `1` is on (phase A's lane runs
/// the local argues); `0` is off (phase B runs them); anything else is
/// refused.
#[test]
fn the_local_lane_knob_reads_on_or_off() {
    assert_eq!(local_lane_setting(None), Ok(true));
    assert_eq!(local_lane_setting(Some("")), Ok(true));
    assert_eq!(local_lane_setting(Some(" 1 ")), Ok(true));
    assert_eq!(local_lane_setting(Some("0")), Ok(false));
    for bad in ["on", "off", "2", "true", "lane", "01"] {
        assert!(local_lane_setting(Some(bad)).is_err(), "{bad:?}");
    }
}

/// The statement tag names local lookups (`/LL1`), and today's does not.
#[test]
fn the_tag_names_local_lookups() {
    let with = block_format_tag(&local());
    let without = block_format_tag(&lean());
    assert!(with.ends_with(block_whir::LOCAL_LOOKUPS_TAG));
    assert!(!without.ends_with(block_whir::LOCAL_LOOKUPS_TAG));
    assert_eq!(&with[..without.len()], &without[..]);
}

/// A block with KECCAK_RND proves and verifies under local lookups: each
/// group with a KECCAK_RND table ends with its LOCAL_BYTES table, the `j`-th
/// such group with table `N + j`, at 2^16 rows, and carries a non-empty
/// local argue; every other group an empty one. BITWISE no longer counts
/// KECCAK_RND's rounds, or the global bus would not balance.
#[test]
#[ignore = "proves a real block: box tier, run with --ignored --test-threads=1"]
fn a_block_with_local_lookups_proves_and_verifies() {
    let elf = elf();
    let proof = prove_local(&elf);
    let verdict = block_whir::verify_block_whir_traced(
        &proof,
        &elf,
        &opts(),
        &local(),
        false,
        stark::multilinear_table::batched::VerifierChecks::ALL,
    );
    assert!(
        matches!(verdict, Ok(Ok(()))),
        "the local verifier: {verdict:?}"
    );
    let frame = block_whir::block_frame(proof.statement(), &elf, &opts(), &local()).expect("frame");
    let base = frame.local_base.expect("a local frame");
    let refs = frame.airs.air_refs();
    let mut j = 0;
    for (g, group) in proof.groups.iter().enumerate() {
        let keccak = group
            .iter()
            .any(|&i| refs[i as usize].name().starts_with("KECCAK_RND["));
        if keccak {
            assert_eq!(*group.last().unwrap() as usize, base + j, "group {g}");
            assert!(!proof.local[g].is_empty(), "group {g}");
            j += 1;
        } else {
            assert!(group.iter().all(|&i| (i as usize) < base), "group {g}");
            assert!(proof.local[g].is_empty(), "group {g}");
        }
    }
    assert!(j > 0, "the block holds KECCAK_RND");
    assert_eq!(proof.table_num_vars.len(), base + j);
    assert!(
        proof.table_num_vars[base..]
            .iter()
            .all(|&n| usize::from(n) == local_bytes::LOG_ROWS)
    );
    eprintln!(
        "local keccak block: {} tables, {} groups, {j} with local lookups",
        proof.table_num_vars.len(),
        proof.groups.len()
    );
}

/// Format confusion both ways: today's verifier refuses a local proof (as
/// today's type, and carrying local argues), and the local verifier refuses
/// today's proof.
#[test]
#[ignore = "proves a real block: box tier, run with --ignored --test-threads=1"]
fn each_formats_verifier_refuses_the_others_proof() {
    let elf = elf();
    let local_proof = prove_local(&elf);
    assert!(!verify(&local_proof, &elf, &lean()), "today's verifier");
    let todays = prove_lean(&elf);
    assert!(verify(&todays, &elf, &lean()), "today's proof verifies");
    assert!(!verify(&todays, &elf, &local()), "the local verifier");
    // Today's proof carrying empty local argues is still today's statement.
    let padded = todays
        .clone()
        .with_local(vec![Default::default(); todays.groups.len()]);
    assert!(!verify(&padded, &elf, &local()));
    assert!(!verify(&padded, &elf, &lean()));
    // A local format is proved by its own entry point only.
    assert!(
        prove_block_whir_with(
            &elf,
            &[],
            &opts(),
            &local(),
            &block_options(),
            &Deviations::default()
        )
        .is_err()
    );
}

/// The fields of today's proof before the local field, in order: what
/// today's proof archived as.
#[derive(rkyv::Archive, rkyv::Serialize)]
struct BeforeLocal {
    proof: stark::multilinear_table::MultiProof<crate::test_utils::F, crate::test_utils::E>,
    table_num_vars: Vec<u8>,
    runtime_page_ranges: Vec<crate::RuntimePageRange>,
    table_counts: crate::TableCounts,
    public_output: Vec<u8>,
    num_private_input_pages: usize,
    groups: Vec<Vec<u32>>,
    prepared:
        Vec<multilinear::stacked_eval::StackedProof<crate::test_utils::F, crate::test_utils::E>>,
    argues: Vec<stark::multilinear_table::BatchedArgue<crate::test_utils::E>>,
    commit_ood: Vec<math::field::element::FieldElement<crate::test_utils::E>>,
    fused_memory: [u64; 2],
    word_memory: [u64; 7],
}

/// Today's proof (`local: ()`) archives byte for byte as the proof before
/// the local field did.
#[test]
#[ignore = "proves a real block: box tier, run with --ignored --test-threads=1"]
fn todays_proof_archives_as_before_the_local_field() {
    let elf = elf();
    let todays = prove_lean(&elf);
    let before = BeforeLocal {
        proof: todays.proof.clone(),
        table_num_vars: todays.table_num_vars.clone(),
        runtime_page_ranges: todays.runtime_page_ranges.clone(),
        table_counts: todays.table_counts.clone(),
        public_output: todays.public_output.clone(),
        num_private_input_pages: todays.num_private_input_pages,
        groups: todays.groups.clone(),
        prepared: todays.prepared.clone(),
        argues: todays.argues.clone(),
        commit_ood: todays.commit_ood.clone(),
        fused_memory: todays.fused_memory,
        word_memory: todays.word_memory,
    };
    let now = rkyv::to_bytes::<rkyv::rancor::Error>(&todays).expect("serialize");
    let then = rkyv::to_bytes::<rkyv::rancor::Error>(&before).expect("serialize");
    assert_eq!(now.as_slice(), then.as_slice());
}

/// The frame refuses LOCAL_BYTES tables out of place: a group with a local
/// sender and no LOCAL_BYTES, one not last, two in one group, one in a group
/// with no local sender, one at another height, and a LOCAL_BYTES too many
/// or too few.
#[test]
#[ignore = "proves a real block: box tier, run with --ignored --test-threads=1"]
fn the_frame_refuses_local_bytes_out_of_place() {
    let elf = elf();
    let honest = prove_local(&elf);
    let frame =
        block_whir::block_frame(honest.statement(), &elf, &opts(), &local()).expect("frame");
    let base = frame.local_base.expect("a local frame");
    let k: Vec<usize> = (0..honest.groups.len())
        .filter(|&g| honest.groups[g].iter().any(|&i| i as usize >= base))
        .collect();
    let plain = (0..honest.groups.len())
        .find(|g| !k.contains(g))
        .expect("a group with no local sender");
    let plan_of = |p: &BlockWhirProof<LocalArgues>| {
        crate::lfm::whir_block::WhirBlockPlan::derive(
            &elf,
            &opts(),
            &local(),
            p.statement(),
            Some(p.groups.len()),
        )
    };
    assert!(
        plan_of(&honest).is_ok(),
        "the honest statement's plan derives"
    );
    let refused = |edit: &dyn Fn(&mut BlockWhirProof<LocalArgues>), what: &str| {
        let mut p = honest.clone();
        edit(&mut p);
        assert!(
            block_whir::block_frame(p.statement(), &elf, &opts(), &local()).is_err(),
            "{what}"
        );
        assert!(!verify(&p, &elf, &local()), "{what}");
        // The recursion's plan derives through the same frame: no leaf is
        // derived over a statement the host verifier refuses.
        let plan = plan_of(&p);
        assert!(
            plan.as_ref().is_err_and(|e| e.starts_with("statement:")),
            "{what}: the recursion's plan, by its frame: {:?}",
            plan.err()
        );
    };
    let first = k[0];
    refused(
        &|p| {
            let local = p.groups[first].pop().unwrap();
            p.groups[plain].push(local);
        },
        "LOCAL_BYTES moved to a group with no local sender",
    );
    refused(
        &|p| {
            let local = p.groups[first].pop().unwrap();
            p.groups[first].insert(0, local);
        },
        "LOCAL_BYTES not last",
    );
    refused(
        &|p| {
            let last = p.table_num_vars.len() - 1;
            p.table_num_vars[last] = (local_bytes::LOG_ROWS - 1) as u8;
        },
        "LOCAL_BYTES at 2^15 rows",
    );
    refused(
        &|p| {
            let last = p.table_num_vars.len() - 1;
            p.table_num_vars[last] = (local_bytes::LOG_ROWS + 1) as u8;
        },
        "LOCAL_BYTES at 2^17 rows",
    );
    refused(
        &|p| {
            let extra = p.table_num_vars.len() as u32;
            p.table_num_vars.push(local_bytes::LOG_ROWS as u8);
            p.groups[first].push(extra);
        },
        "two LOCAL_BYTES in one group",
    );
    refused(
        &|p| {
            let local = p.groups[first].pop().unwrap();
            assert_eq!(local as usize, p.table_num_vars.len() - 1 - (k.len() - 1));
            p.table_num_vars.remove(local as usize);
            for group in p.groups.iter_mut() {
                for i in group.iter_mut() {
                    if *i > local {
                        *i -= 1;
                    }
                }
            }
        },
        "a group with a local sender and no LOCAL_BYTES",
    );
    if k.len() > 1 {
        refused(
            &|p| {
                let (a, b) = (k[0], k[1]);
                let la = p.groups[a].pop().unwrap();
                let lb = p.groups[b].pop().unwrap();
                p.groups[a].push(lb);
                p.groups[b].push(la);
            },
            "two groups' LOCAL_BYTES traded",
        );
    }
}

/// The recursion's plan derives over a base under local lookups (its leaf
/// verifies each group's local argue, `lfm::whir_block_local_tests`).
#[test]
#[ignore = "proves a real block: box tier, run with --ignored --test-threads=1"]
fn the_recursion_derives_a_plan_over_a_base_under_local_lookups() {
    let elf = elf();
    let proof = prove_local(&elf);
    // One group a leaf: the rule's leaf heights hold this small format's
    // KECCAK_W group in no leaf, whatever the format.
    let plan = crate::lfm::whir_block::WhirBlockPlan::derive(
        &elf,
        &opts(),
        &local(),
        proof.statement(),
        Some(proof.groups.len()),
    );
    assert!(plan.is_ok(), "{:?}", plan.err());
}

/// Phase A's lane moves no byte on a real block: the block proved with the
/// lane on and with it off ([`Deviations::local_lane`]) is the same proof,
/// and it verifies. Phase A takes every table's digest from its own layout,
/// and the prove refuses digests that are not the verifier's from its AIRs,
/// so both proves pin them. On a card the lane runs local bins beside the
/// commits; on the host a group has no store on the card and every job is
/// left to phase B. Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1` (one grind
/// nonce for both proves).
#[test]
#[ignore = "box tier: proves a real block twice, run with --ignored --test-threads=1 and \
            LAMBDA_VM_DETERMINISTIC_GRIND=1"]
fn the_local_lane_moves_no_byte_on_a_real_block() {
    assert!(
        crypto::grinding::deterministic(),
        "export LAMBDA_VM_DETERMINISTIC_GRIND=1"
    );
    let elf = elf();
    let prove_with = |lane: bool| {
        let t = std::time::Instant::now();
        let proved = prove_block_whir_local_with(
            &elf,
            &[],
            &opts(),
            &local(),
            &block_options(),
            &Deviations {
                local_lane: Some(lane),
                ..Deviations::default()
            },
        )
        .expect("prove");
        eprintln!(
            "local prove, lane {lane}: {:.1}s\n{}",
            t.elapsed().as_secs_f64(),
            stark::multilinear_block::local_lane_lines(&proved.1.groups)
        );
        proved
    };
    let (on, on_stamps) = prove_with(true);
    let (off, off_stamps) = prove_with(false);
    let bytes = |proof: &BlockWhirProof<LocalArgues>| {
        rkyv::to_bytes::<rkyv::rancor::Error>(proof)
            .expect("serialize")
            .to_vec()
    };
    assert!(
        bytes(&on) == bytes(&off),
        "the lane moved the proof's bytes"
    );
    assert!(verify(&on, &elf, &local()), "the lane's proof verifies");
    let local_groups = on.local.iter().filter(|argue| !argue.is_empty()).count();
    assert!(local_groups > 0, "the block holds local groups");
    // Lane off: no job, every local group's bins in phase B.
    assert!(
        off_stamps
            .groups
            .iter()
            .all(|g| g.zc_lane.state == stark::multilinear_block::LaneState::Off)
    );
    assert_eq!(
        off_stamps
            .groups
            .iter()
            .filter(|g| !g.local_b.is_empty())
            .count(),
        local_groups
    );
    // Lane on: a job a local group; on a card, bins ran beside the commits.
    let jobs = on_stamps.groups.iter().filter(|g| g.zc_lane.local).count();
    assert_eq!(jobs, local_groups, "a lane job a local group");
    let ran: usize = on_stamps.groups.iter().map(|g| g.zc_lane.bins_ran).sum();
    if multilinear::gpu::reserve_budget() > 0 {
        assert!(ran > 0, "on a card the lane ran local bins");
    }
}

/// ★ The lane's room knobs move no byte, and a misprediction never fails a
/// prove. The block proved with the lane under each of its knobs — the next
/// commit's growth by group shape (`LAMBDA_VM_LANE_GROWTH=shape`), reserves of
/// 1 GiB (`LAMBDA_VM_LANE_RESERVE_MIB=1024`), and both — is the lane-off proof
/// byte for byte, and verifies. Then every prediction forced to 0 with no
/// reserves, under a budget below the lane-off run's ledger peak: the lane
/// holds what it would not, a commit's reservations are refused where they
/// ask and take the host path, and the prove still ends with the same bytes.
/// On a card the refusals are counted; on the host the budget is 0 and the
/// lane's room rule is off. Needs `LAMBDA_VM_DETERMINISTIC_GRIND=1`.
#[test]
#[ignore = "box tier: proves a real block five times, run with --ignored --test-threads=1 and \
            LAMBDA_VM_DETERMINISTIC_GRIND=1 (it moves the process's card budget)"]
fn the_lane_room_knobs_move_no_byte_and_a_misprediction_falls_back() {
    use stark::multilinear_block::{LaneGrowth, LaneLimits};
    assert!(
        crypto::grinding::deterministic(),
        "export LAMBDA_VM_DETERMINISTIC_GRIND=1"
    );
    let elf = elf();
    let prove_with = |lane: bool, limits: Option<LaneLimits>, what: &str| {
        let t = std::time::Instant::now();
        let proved = prove_block_whir_local_with(
            &elf,
            &[],
            &opts(),
            &local(),
            &block_options(),
            &Deviations {
                local_lane: Some(lane),
                lane_limits: limits,
                ..Deviations::default()
            },
        )
        .unwrap_or_else(|e| panic!("{what}: the prove failed: {e:?}"));
        eprintln!(
            "lane room {what}: {:.1}s\n{}",
            t.elapsed().as_secs_f64(),
            stark::multilinear_block::local_lane_lines(&proved.1.groups)
        );
        proved
    };
    let bytes = |proof: &BlockWhirProof<LocalArgues>| {
        rkyv::to_bytes::<rkyv::rancor::Error>(proof)
            .expect("serialize")
            .to_vec()
    };
    let (reference, off_stamps) = prove_with(false, None, "lane off");
    let reference = bytes(&reference);
    let gib = 1u64 << 30;
    for (what, limits) in [
        (
            "growth by shape",
            LaneLimits {
                growth: LaneGrowth::Shape,
                ..LaneLimits::default()
            },
        ),
        (
            "reserves 1 GiB",
            LaneLimits {
                margin: gib,
                unledgered: gib,
                ..LaneLimits::default()
            },
        ),
        (
            "both",
            LaneLimits {
                growth: LaneGrowth::Shape,
                margin: gib,
                unledgered: gib,
                ..LaneLimits::default()
            },
        ),
    ] {
        let (proof, stamps) = prove_with(true, Some(limits), what);
        assert!(bytes(&proof) == reference, "{what}: the bytes moved");
        assert!(verify(&proof, &elf, &local()), "{what}: the proof verifies");
        if limits.growth == LaneGrowth::Shape && multilinear::gpu::reserve_budget() > 0 {
            assert!(
                stamps
                    .groups
                    .iter()
                    .skip(1)
                    .any(|g| g.lane_predicted_growth > 0),
                "{what}: the lane predicted the commits' growth"
            );
        }
    }
    // Every prediction 0, no reserves, under a budget below the lane-off
    // run's peak: commits are refused where they ask, and the prove ends.
    let peak = off_stamps
        .groups
        .iter()
        .map(|g| g.commit_reserved)
        .max()
        .unwrap_or(0);
    let tight = peak / 4 * 3;
    let before = multilinear::gpu::set_reserve_budget(tight);
    let fallbacks_before = multilinear::gpu::host_fallbacks();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prove_with(
            true,
            Some(LaneLimits {
                growth: LaneGrowth::Shape,
                predict: Some(0),
                margin: 0,
                unledgered: 0,
                ..LaneLimits::default()
            }),
            "every prediction 0 under a tight budget",
        )
    }));
    let fallbacks = multilinear::gpu::host_fallbacks().saturating_sub(fallbacks_before);
    multilinear::gpu::set_reserve_budget(before);
    let (proof, stamps) = outcome.unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    let refused: u64 = stamps.groups.iter().map(|g| g.commit_host_fallbacks).sum();
    eprintln!(
        "lane room misprediction: budget {} MiB (the lane-off peak {} MiB) · commits with a host \
         fallback {refused} · host fallbacks {fallbacks}",
        tight >> 20,
        peak >> 20
    );
    assert!(
        bytes(&proof) == reference,
        "a misprediction moved the bytes"
    );
    assert!(
        verify(&proof, &elf, &local()),
        "the mispredicted proof verifies"
    );
    if before > 0 {
        assert!(
            refused > 0 || fallbacks > 0,
            "on a card the tight budget refused a commit's reservation"
        );
    }
}

/// ★ Where the LOCAL_BYTES tables sit ([`block_whir::check_local_groups`]),
/// on hand-written groups (no proof): tables 0–3 the run's (0 and 2 send a
/// local lookup), 4 and 5 LOCAL_BYTES. The honest packing passes; each
/// misplacement is refused by its own message — two sending groups'
/// LOCAL_BYTES traded (the `base + j` rule: the j-th sending group ends with
/// table `base + j`), one not last, two in one group, a sending group with
/// none, one in a group with no sender, and a LOCAL_BYTES too many.
#[test]
fn local_bytes_tables_sit_last_in_their_groups_in_group_order() {
    let senders = [true, false, true, false, false, false];
    let check = |groups: &[&[u32]], senders: &[bool]| {
        let groups: Vec<Vec<u32>> = groups.iter().map(|g| g.to_vec()).collect();
        block_whir::check_local_groups(&groups, senders, 4).map_err(|e| format!("{e:?}"))
    };
    assert_eq!(check(&[&[0, 1, 4], &[2, 3, 5]], &senders), Ok(()));
    for (groups, want) in [
        (
            &[&[0u32, 1, 5][..], &[2, 3, 4]][..],
            "group 0 ends with LOCAL_BYTES table 5, not 4",
        ),
        (
            &[&[0, 4, 1], &[2, 3, 5]],
            "group 0 holds a local sender: it ends with exactly one LOCAL_BYTES table",
        ),
        (
            &[&[0, 1, 4, 5], &[2, 3]],
            "group 0 holds a local sender: it ends with exactly one LOCAL_BYTES table",
        ),
        (
            &[&[0, 1], &[2, 3, 4], &[5]],
            "group 0 holds a local sender: it ends with exactly one LOCAL_BYTES table",
        ),
        (
            &[&[0, 4], &[1, 5], &[2, 3]],
            "group 1 holds a LOCAL_BYTES table and no local sender",
        ),
    ] {
        let refused = check(groups, &senders);
        assert!(
            refused.as_ref().is_err_and(|e| e.contains(want)),
            "{groups:?}: {refused:?}"
        );
    }
    // A LOCAL_BYTES table stated for no sending group.
    let seven = [true, false, true, false, false, false, false];
    let refused = check(&[&[0, 1, 4], &[2, 3, 5], &[6]], &seven);
    assert!(
        refused
            .as_ref()
            .is_err_and(|e| e.contains("group 2 holds a LOCAL_BYTES table and no local sender")),
        "{refused:?}"
    );
}

/// ★ Which tables may receive a local bus ([`block_whir::check_local_airs`]),
/// on hand-built AIRs (no proof): the run's tables, then LOCAL_BYTES at 2^16
/// rows, passes; a LOCAL_BYTES before the base, a run table after it, a
/// LOCAL_BYTES at 2^15 rows and a run table receiving a local bus are each
/// refused by their own message.
#[test]
fn only_local_bytes_tables_after_the_base_receive_local_buses() {
    use crate::tables::types::BusId;
    use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
    use stark::traits::AIR;
    let opts = opts();
    let plain = crate::test_utils::create_plain_air(&opts, 4, Vec::new(), "PLAIN");
    let receiver = crate::test_utils::create_plain_air(
        &opts,
        4,
        vec![BusInteraction::receiver(
            BusId::LocalAreBytes,
            Multiplicity::Column(2),
            vec![
                BusValue::Packed {
                    start_column: 0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: 1,
                    packing: Packing::Direct,
                },
            ],
        )],
        "RECEIVER",
    );
    let fixed = crate::test_utils::create_local_bytes_air(&opts).with_name("LOCAL_BYTES[0]");
    type Air<'a> = &'a dyn AIR<
        Field = crate::tables::types::GoldilocksField,
        FieldExtension = crate::tables::types::GoldilocksExtension,
        PublicInputs = (),
    >;
    let check = |airs: &[Air<'_>], heights: &[u8], base: usize| {
        block_whir::check_local_airs(airs, heights, base).map_err(|e| format!("{e:?}"))
    };
    let lb = local_bytes::LOG_ROWS as u8;
    assert_eq!(check(&[&plain, &plain, &fixed], &[3, 3, lb], 2), Ok(()));
    for (airs, heights, base, want) in [
        (
            vec![&fixed as Air<'_>, &plain],
            vec![lb, 3],
            2,
            "LOCAL_BYTES[0] (table 0) receives a local bus: only LOCAL_BYTES does",
        ),
        (
            vec![&plain as Air<'_>, &plain],
            vec![3, 3],
            1,
            "table 1 after the run's tables is PLAIN, not LOCAL_BYTES",
        ),
        (
            vec![&plain as Air<'_>, &fixed],
            vec![3, lb - 1],
            1,
            "LOCAL_BYTES (table 1) states 2^15 rows — it has 2^16",
        ),
        (
            vec![&receiver as Air<'_>, &fixed],
            vec![3, lb],
            1,
            "RECEIVER (table 0) receives a local bus: only LOCAL_BYTES does",
        ),
    ] {
        let refused = check(&airs, &heights, base);
        assert!(
            refused.as_ref().is_err_and(|e| e.contains(want)),
            "{want}: {refused:?}"
        );
    }
}

/// ★ The verifier's digests ([`block_whir::local_digests`]) of two tables
/// with no interaction, of one shape and different preprocessed counts, in a
/// group with a local sender: each is the digest of its own inputs, and they
/// differ. Every table with no interaction shares the one empty list, so a
/// cache keyed by the list's address alone would give both one digest.
#[test]
fn local_digests_of_two_tables_without_interactions_differ_by_their_fixed_count() {
    use crate::tables::types::BusId;
    use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
    use stark::multilinear_block::table_digest;
    use stark::traits::AIR;
    type T = <multilinear::whir_hash::P1Whir as multilinear::whir_hash::WhirHash>::Sponge;
    let opts = opts();
    let sender = crate::test_utils::create_plain_air(
        &opts,
        local_bytes::cols::NUM_COLUMNS,
        vec![BusInteraction::sender(
            BusId::LocalAreBytes,
            Multiplicity::One,
            vec![
                BusValue::Packed {
                    start_column: 0,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: 1,
                    packing: Packing::Direct,
                },
            ],
        )],
        "SENDER",
    );
    let bare = crate::test_utils::create_plain_air(
        &opts,
        local_bytes::cols::NUM_COLUMNS,
        Vec::new(),
        "BARE",
    );
    let fixed = crate::test_utils::create_local_bytes_air_with(&opts, Vec::new());
    assert!(bare.bus_interactions().is_empty() && fixed.bus_interactions().is_empty());
    assert_eq!(
        fixed.num_precomputed_columns(),
        local_bytes::NUM_PRECOMPUTED_COLS
    );
    type Air<'a> = &'a dyn AIR<
        Field = crate::tables::types::GoldilocksField,
        FieldExtension = crate::tables::types::GoldilocksExtension,
        PublicInputs = (),
    >;
    let airs: [Air<'_>; 3] = [&sender, &bare, &fixed];
    let width = local_bytes::cols::NUM_COLUMNS;
    let shapes = vec![(width, 4); 3];
    let digests =
        block_whir::local_digests::<T>(&[vec![0, 1, 2]], &shapes, |i| airs.get(i).copied())
            .expect("the digests");
    let own = |preprocessed: usize| {
        table_digest::<crate::tables::types::GoldilocksExtension, T>(4, width, preprocessed, &[])
    };
    assert_eq!(digests[1], own(0), "the bare table's own digest");
    assert_eq!(
        digests[2],
        own(local_bytes::NUM_PRECOMPUTED_COLS),
        "the fixed table's own digest"
    );
    assert_ne!(digests[1], digests[2]);
}
