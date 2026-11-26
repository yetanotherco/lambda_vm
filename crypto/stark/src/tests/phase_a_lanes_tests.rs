//! Phase A's commits on two lanes ([`BlockCommitted::commit_on_lanes`]) on
//! the host: the second lane runs without a second card
//! ([`PhaseALanes::on_host`], its counts kept as the second card's), so the
//! feed, the indexes, the merge by index, the hand-back of a failed second
//! lane and the per-lane counters are checked here; the second card's device
//! is the box's.
//!
//! The block is [`super::phase_b_overlap_tests`]'s: seven groups, two of them
//! with prepared stacks. Its proof in group order is pinned
//! ([`BEFORE_THE_OVERLAP`]), whichever lane committed which group.

use std::sync::Mutex;

use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as Fp,
};
use multilinear::{
    whir_chain::{ArgueFormat, ChainConfig},
    whir_commit::TreeDrop,
    whir_hash::WhirHash,
};

use super::phase_b_overlap_tests::{
    ARGUES, BEFORE_THE_OVERLAP, H, SIZES, airs, config, digest, in_order, prove_block_with, tables,
    verify,
};
use crate::multilinear_block::{
    BlockCommitted, BlockProved, GroupStamps, LanePolicy, Narrowing, OverlapFaults, PhaseALanes,
    PhaseBOverlap, TwoCardKnob, commit_ood_point,
};
use crate::multilinear_table::{CommittedTable, global_layout};

type Sponge = <H as WhirHash>::Sponge;

/// Lane blocks with the indexes of their groups, as the merge takes them.
type Parts<'a> = Vec<(BlockCommitted<'a, Fp, Ext>, Vec<usize>)>;

/// The second lane's counts are the second card's slot, shared by every test
/// here that runs two lanes: one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const NARROWINGS: [Narrowing; 2] = [Narrowing::Wide, Narrowing::Host];

/// The assignments: all on lane 0, all on lane 1, alternating, a seeded
/// random one, and each lane taking the next group when free.
fn policies() -> Vec<LanePolicy> {
    let n = SIZES.len();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let random = (0..n)
        .map(|_| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            u8::from(seed >> 63 == 1)
        })
        .collect();
    vec![
        LanePolicy::Fixed(vec![0; n]),
        LanePolicy::Fixed(vec![1; n]),
        LanePolicy::Fixed((0..n).map(|g| (g % 2) as u8).collect()),
        LanePolicy::Fixed(random),
        LanePolicy::NextFree,
    ]
}

fn alternating() -> LanePolicy {
    LanePolicy::Fixed((0..SIZES.len()).map(|g| (g % 2) as u8).collect())
}

fn on_host(policy: LanePolicy, fault: Option<usize>) -> PhaseALanes {
    PhaseALanes {
        knob: None,
        policy,
        on_host: true,
        fault,
    }
}

/// The block's phase A on `lanes`.
fn commit_lanes<'a>(
    tables: Vec<CommittedTable<'a, Fp, Ext>>,
    config: &ChainConfig,
    narrow: Narrowing,
    upload_ahead: bool,
    lanes: &PhaseALanes,
) -> BlockCommitted<'a, Fp, Ext> {
    BlockCommitted::commit_on_lanes::<H, Sponge>(
        tables,
        &SIZES,
        config,
        TreeDrop::uniform(1),
        narrow,
        upload_ahead,
        lanes,
    )
    .expect("phase A on the lanes")
}

/// The block proved in group order (the overlap off, the stacks held), its
/// phase A on `lanes`.
fn prove_on_lanes(
    argue: ArgueFormat,
    narrow: Narrowing,
    upload_ahead: bool,
    lanes: &PhaseALanes,
) -> BlockProved<Fp, Ext> {
    let (proved, _) = prove_block_with::<H>(
        &config(argue),
        PhaseBOverlap::Off,
        &in_order(),
        OverlapFaults::default(),
        &|_| {},
        None,
        false,
        Some(&|tables, config| commit_lanes(tables, config, narrow, upload_ahead, lanes)),
    );
    proved.unwrap_or_else(|e| panic!("{argue:?}: {e:?}"))
}

fn commit_cards(stamps: &[GroupStamps]) -> Vec<u8> {
    stamps.iter().map(|s| s.commit_card).collect()
}

/// ★ Two lanes commit the block of one: under either argue format, the next
/// group uploaded beside the commit or not, the tables held wide or packed,
/// and every assignment of groups to lanes, the merged block's sizes and
/// roots are one lane's, in group order, and the proof is the pinned one-lane
/// proof byte for byte and verifies; each group says the lane the assignment
/// gave it; each group's tables and polynomials are one lane's; and the
/// second lane's groups count the host commits an all-second-lane run counts
/// for them (its counters are its own card's, never the other lane's).
#[test]
fn two_lanes_commit_the_same_block() {
    let _serial = serial();
    let n = SIZES.len();
    for (argue, pinned) in ARGUES.into_iter().zip(BEFORE_THE_OVERLAP) {
        for narrow in NARROWINGS {
            for upload_ahead in [false, true] {
                let what = format!("{argue:?} {narrow:?} upload ahead {upload_ahead}");
                let one = prove_on_lanes(argue, narrow, upload_ahead, &PhaseALanes::default());
                assert_eq!(digest(&one), pinned, "{what}: one lane, the pinned bytes");
                assert!(
                    commit_cards(&one.3).iter().all(|&c| c == 0),
                    "{what}: one lane commits on card 0"
                );
                let all_second = prove_on_lanes(
                    argue,
                    narrow,
                    upload_ahead,
                    &on_host(LanePolicy::Fixed(vec![1; n]), None),
                );
                let airs = airs();
                let one_block = commit_lanes(
                    tables(&airs),
                    &config(argue),
                    narrow,
                    upload_ahead,
                    &PhaseALanes::default(),
                );
                for policy in policies() {
                    let what = format!("{what} {policy:?}");
                    let block = commit_lanes(
                        tables(&airs),
                        &config(argue),
                        narrow,
                        upload_ahead,
                        &on_host(policy.clone(), None),
                    );
                    assert_eq!(block.sizes(), one_block.sizes(), "{what}: the sizes");
                    assert_eq!(
                        block.roots(),
                        one_block.roots(),
                        "{what}: the roots in group order"
                    );
                    let two =
                        prove_on_lanes(argue, narrow, upload_ahead, &on_host(policy.clone(), None));
                    assert_eq!(digest(&two), pinned, "{what}: two lanes moved a byte");
                    verify(argue, &two).unwrap_or_else(|e| panic!("{what}: verifies: {e:?}"));
                    let cards = commit_cards(&two.3);
                    match &policy {
                        LanePolicy::Fixed(want) => {
                            assert_eq!(&cards, want, "{what}: each group on its lane");
                        }
                        LanePolicy::NextFree => {
                            assert!(cards.iter().all(|&c| c <= 1), "{what}: {cards:?}");
                        }
                    }
                    for (g, (a, b)) in two.3.iter().zip(&one.3).enumerate() {
                        assert_eq!(
                            (a.tables, a.polys, a.cells),
                            (b.tables, b.polys, b.cells),
                            "{what}: group {g}'s shape"
                        );
                    }
                    for (g, (a, b)) in two.3.iter().zip(&all_second.3).enumerate() {
                        if a.commit_card == 1 {
                            assert_eq!(
                                a.commit_host_fallbacks, b.commit_host_fallbacks,
                                "{what}: group {g}'s host commits, counted on its own lane"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Group `g`'s commit-time point from either lane is the one drawn for `g`
/// over its own layout and roots ([`commit_ood_point`]), never for the place
/// it had in its lane's order.
#[test]
fn a_group_keeps_its_index_on_either_lane() {
    let _serial = serial();
    let airs = airs();
    for argue in ARGUES {
        let config = config(argue);
        for upload_ahead in [false, true] {
            let block = commit_lanes(
                tables(&airs),
                &config,
                Narrowing::Wide,
                upload_ahead,
                &on_host(alternating(), None),
            );
            assert_eq!(block.sizes(), &SIZES, "{argue:?}: the sizes in group order");
            let points = block.commit_points();
            assert_eq!(points.len(), SIZES.len());
            let (mut first_table, mut first_root) = (0usize, 0usize);
            for (g, &size) in SIZES.iter().enumerate() {
                let shapes: Vec<(usize, usize)> = block.tables()[first_table..first_table + size]
                    .iter()
                    .map(|t| (t.num_committed_columns(), t.num_vars()))
                    .collect();
                let layout = global_layout(&shapes, config.format.stack).expect("a layout");
                let trees =
                    multilinear::stacked_eval::round0_trees(layout.num_polys(), &config).len();
                let roots = &block.roots()[first_root..first_root + trees];
                let want: FieldElement<Ext> =
                    commit_ood_point::<Ext, Sponge>(g, &layout, roots, &config).expect("the point");
                assert_eq!(points[g], want, "{argue:?}: group {g}'s commit-time point");
                assert_eq!(
                    block.stamps()[g].commit_card,
                    (g % 2) as u8,
                    "{argue:?}: group {g}'s lane"
                );
                first_table += size;
                first_root += trees;
            }
            assert_eq!(first_root, block.roots().len(), "every root in a group");
        }
    }
}

/// The merge puts groups by index and refuses, without a panic, an index
/// filled twice, one left empty, one past the block, and a part whose indexes
/// do not count its groups.
#[test]
fn merge_lanes_refuses_a_missing_or_doubled_group() {
    let airs = airs();
    let config = config(ArgueFormat::BATCHED);
    let n = SIZES.len();
    let part = || {
        BlockCommitted::commit::<H, Sponge>(
            tables(&airs),
            &SIZES,
            &config,
            TreeDrop::uniform(1),
            Narrowing::Wide,
            false,
        )
        .expect("phase A")
    };
    let all: Vec<usize> = (0..n).collect();
    let whole = BlockCommitted::merge_lanes(vec![(part(), all.clone())], n, &config, false)
        .expect("one whole part merges");
    assert_eq!(
        whole.roots(),
        part().roots(),
        "the same roots in group order"
    );
    assert_eq!(whole.sizes(), &SIZES);
    let mut doubled = all.clone();
    doubled[n - 1] = n - 2;
    let mut past = all.clone();
    past[n - 1] = n + 2;
    let cases: Vec<(&str, Parts<'_>, usize)> = vec![
        ("an index twice", vec![(part(), doubled)], n),
        ("an index left empty", vec![(part(), all.clone())], n + 1),
        ("an index past the block", vec![(part(), past)], n),
        (
            "every group twice",
            vec![(part(), all.clone()), (part(), all.clone())],
            n,
        ),
        (
            "indexes short of the groups",
            vec![(part(), all[..n - 1].to_vec())],
            n,
        ),
    ];
    for (what, parts, n) in cases {
        let merged = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            BlockCommitted::merge_lanes(parts, n, &config, false)
        }));
        match merged {
            Ok(Err(_)) => {}
            Ok(Ok(_)) => panic!("{what}: merged"),
            Err(_) => panic!("{what}: panicked"),
        }
    }
}

/// ★ The second lane failing before its first, second or last commit hands
/// every group it holds back (the group in hand, the one taken ahead, every
/// one it committed): card 0 commits them, every group says card 0, and the
/// proof is the pinned one-lane proof. With no fault the second lane keeps
/// its groups.
#[test]
fn the_second_lanes_fault_commits_its_groups_on_the_first() {
    let _serial = serial();
    let argue = ArgueFormat::BATCHED;
    let pinned = BEFORE_THE_OVERLAP[1];
    // Alternating: the second lane's groups are 1, 3 and 5.
    let second: Vec<usize> = (0..SIZES.len()).filter(|g| g % 2 == 1).collect();
    for narrow in NARROWINGS {
        for upload_ahead in [false, true] {
            for (policy, faults) in [
                (alternating(), vec![0, 1, second.len() - 1]),
                (LanePolicy::NextFree, vec![0]),
            ] {
                for fault in faults {
                    let what =
                        format!("{narrow:?} upload ahead {upload_ahead} {policy:?} fault {fault}");
                    let two = prove_on_lanes(
                        argue,
                        narrow,
                        upload_ahead,
                        &on_host(policy.clone(), Some(fault)),
                    );
                    assert_eq!(two.3.len(), SIZES.len(), "{what}: every group committed");
                    assert_eq!(digest(&two), pinned, "{what}: the bytes of one lane");
                    verify(argue, &two).unwrap_or_else(|e| panic!("{what}: verifies: {e:?}"));
                    assert!(
                        commit_cards(&two.3).iter().all(|&c| c == 0),
                        "{what}: every group on card 0, read {:?}",
                        commit_cards(&two.3)
                    );
                }
            }
            let clean = prove_on_lanes(
                argue,
                narrow,
                upload_ahead,
                &on_host(alternating(), Some(second.len())),
            );
            assert_eq!(digest(&clean), pinned);
            let on_second: Vec<usize> = (0..SIZES.len())
                .filter(|&g| clean.3[g].commit_card == 1)
                .collect();
            assert_eq!(
                on_second, second,
                "no fault: the second lane keeps its groups"
            );
        }
    }
}

/// One lane, today's commit, when nothing asks for two, when the knob's split
/// is off (no second card on the host), or with one group: every group on
/// card 0 and the pinned bytes. Two lanes only for a test's lanes on the host
/// with two groups or more.
#[test]
fn one_lane_when_the_knob_or_the_card_says_so() {
    let levels = PhaseALanes {
        knob: Some(TwoCardKnob::Levels),
        ..Default::default()
    };
    assert_eq!(
        PhaseALanes::default().one_lane_because(SIZES.len()),
        Some("not asked")
    );
    assert!(
        levels.one_lane_because(SIZES.len()).is_some(),
        "the levels' split is off on the host"
    );
    assert_eq!(
        on_host(LanePolicy::NextFree, None).one_lane_because(1),
        Some("one group")
    );
    assert_eq!(
        on_host(LanePolicy::NextFree, None).one_lane_because(2),
        None
    );
    for (argue, pinned) in ARGUES.into_iter().zip(BEFORE_THE_OVERLAP) {
        let one = prove_on_lanes(argue, Narrowing::Wide, true, &levels);
        assert_eq!(digest(&one), pinned, "{argue:?}: the knob's one lane");
        assert!(commit_cards(&one.3).iter().all(|&c| c == 0));
    }
}
