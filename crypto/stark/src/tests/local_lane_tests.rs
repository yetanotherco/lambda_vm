//! Local lookups from phase A on ([`LocalPhaseA`]): each group's digests and
//! local challenges taken right after its commit, and phase A's lane running
//! the local groups' bins beside the commits, the bins left at the group's
//! turn in phase B. On the small block of [`super::local_lookup_tests`], with
//! no card, so the lane runs on the host.
//!
//! Wherever the bins run — every one on the lane, none, stopped after each
//! bin index, refused for memory, or a unit failing partway — the proof is
//! the one phase B alone proves, byte for byte, and it verifies.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::time::{Duration, Instant};

use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as Fp,
};
use multilinear::{
    Error as MlError,
    whir_chain::{ChainConfig, GkrLadder, GkrPack},
    whir_commit::TreeDrop,
    whir_hash::WhirHash,
};

use super::local_lookup_tests::{
    Airs, BUSES, Block, FIXED, H, KINDS, Kind, SIZES, bytes, config, config_with, digests,
    false_tuple, faulted, honest, local, num_vars, production, prove, verify_local,
};
use crate::multilinear_air::Uniforms;
use crate::multilinear_block::{
    self, ArgueDeviation, BlockCommitted, BlockProved, CsProbe, GroupStamps, LaneDefer, LaneLimits,
    LaneState, LocalPhaseA, Narrowing, ZcLaneStamp, local_lane_lines,
};
use crate::multilinear_table::{
    CommittedTable, TableLayout,
    batched::{LocalBinding, ProverFaults},
};
use crate::traits::AIR;

/// Where the local groups' bins run.
#[derive(Clone, Copy, Debug)]
enum Placement {
    /// Phase A takes the digests and challenges; phase B runs every bin.
    PhaseB,
    /// The lane on the host, under these limits.
    Lane(LaneLimits),
    /// No group taken beside a commit: no lane, whatever is asked.
    NoUpload,
}

/// The block's tables, each layout carrying its AIR's preprocessed count
/// (the RECV's fixed columns when the verifier holds them).
fn tables<'a>(airs: &'a Airs, block: &Block) -> Vec<CommittedTable<'a, Fp, Ext>> {
    tables_counted(airs, block, true)
}

/// [`tables`], the RECV's preprocessed count set or (`counted` false) left
/// at 0.
fn tables_counted<'a>(
    airs: &'a Airs,
    block: &Block,
    counted: bool,
) -> Vec<CommittedTable<'a, Fp, Ext>> {
    block
        .columns
        .iter()
        .enumerate()
        .map(|(t, columns)| {
            let air = airs.of(t);
            let fixed = if KINDS[t] == Kind::Recv && block.fixed && counted {
                FIXED
            } else {
                0
            };
            let layout = TableLayout::new(
                air.constraint_program(),
                air.constraints_meta(),
                air.bus_interactions(),
                columns.len(),
                num_vars(columns),
                Uniforms::default(),
            )
            .expect("a layout")
            .with_preprocessed_count(fixed);
            CommittedTable::from_layout(layout, |col| columns[col as usize].clone())
                .expect("a table")
        })
        .collect()
}

/// Phase A over the block's groups under local lookups, placed as
/// `placement` says. On the lane, group `k` is handed over only once group
/// `k − 1`'s job was decided, so each job runs before the next commit.
fn commit<'a>(
    tables: Vec<CommittedTable<'a, Fp, Ext>>,
    config: &ChainConfig,
    placement: Placement,
) -> Result<BlockCommitted<'a, Fp, Ext>, MlError> {
    commit_bound(tables, config, placement, LocalBinding::ALL)
}

/// [`commit`] with phase A's local transcripts binding what `binding` says
/// (a mutant phase A unless it is [`LocalBinding::ALL`]).
fn commit_bound<'a>(
    tables: Vec<CommittedTable<'a, Fp, Ext>>,
    config: &ChainConfig,
    placement: Placement,
    binding: LocalBinding,
) -> Result<BlockCommitted<'a, Fp, Ext>, MlError> {
    let mut tables = tables.into_iter();
    let mut groups = SIZES
        .iter()
        .map(|&size| tables.by_ref().take(size).collect::<Vec<_>>())
        .collect::<Vec<_>>()
        .into_iter();
    let (lane, limits) = match placement {
        Placement::Lane(limits) => (true, limits),
        _ => (false, LaneLimits::default()),
    };
    let decided = lane.then(|| Arc::new(AtomicUsize::new(0)));
    let gate = decided.clone();
    let mut handed = 0usize;
    let gated = std::iter::from_fn(move || {
        if let Some(decided) = &gate
            && handed > 0
        {
            let since = Instant::now();
            while decided.load(SeqCst) < handed && since.elapsed() < Duration::from_secs(120) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        handed += 1;
        groups.next()
    });
    BlockCommitted::commit_groups_probed::<H, <H as WhirHash>::Sponge>(
        gated,
        config,
        TreeDrop::uniform(1),
        Narrowing::Wide,
        !matches!(placement, Placement::NoUpload),
        None,
        None,
        CsProbe {
            host: true,
            decided,
            limits,
            ..CsProbe::default()
        },
        Some(LocalPhaseA {
            buses: BUSES.to_vec(),
            lane,
            binding,
        }),
    )
}

/// The block proved with local lookups from phase A on, the bins placed as
/// `placement` says, group `deviation.group` arguing as `deviation` says.
fn prove_placed(
    airs: &Airs,
    block: &Block,
    config: &ChainConfig,
    placement: Placement,
    deviation: ArgueDeviation,
) -> Result<BlockProved<Fp, Ext>, MlError> {
    let committed = commit(tables(airs, block), config, placement)?;
    let order: Vec<usize> = (0..SIZES.len()).collect();
    let mut transcript = H::sponge();
    multilinear_block::block_prove_in_order::<Fp, Ext, _, H>(
        committed,
        config,
        &mut transcript,
        &[],
        &[],
        &|g| g,
        &deviation,
        &|_| {},
        &order,
        None,
    )
}

/// Each local group's lane stamp (groups 1 and 2; group 0 has no local
/// interaction).
fn lane_stamps(proved: &BlockProved<Fp, Ext>) -> Vec<(ZcLaneStamp, Vec<f64>)> {
    proved
        .3
        .iter()
        .map(|g| (g.zc_lane, g.local_b.clone()))
        .collect()
}

fn lane(limits: LaneLimits) -> Placement {
    Placement::Lane(limits)
}

/// The digests phase A takes from the tables' own layouts are the verifier's,
/// from its AIRs: zeros for the group with no local interaction, and each
/// local group's tables' digests. Without the RECV's preprocessed count in
/// its layout they are not (the count is what the digest reads of it).
#[test]
fn the_digests_taken_at_commit_time_are_the_verifiers() {
    let airs = Airs::new(None);
    let block = Block::honest();
    let config = production();
    let expected = digests(&airs, &block, true);
    let committed = commit(tables(&airs, &block), &config, Placement::PhaseB).expect("phase A");
    let taken = committed.local_digests().expect("local lookups").to_vec();
    assert_eq!(taken.len(), expected.len());
    for (t, digest) in taken.iter().enumerate().take(SIZES[0]) {
        assert_eq!(
            *digest, [0u8; 32],
            "table {t}: a group with no local interaction"
        );
    }
    assert_eq!(
        taken[SIZES[0]..],
        expected[SIZES[0]..],
        "the local groups' digests"
    );

    let uncounted = commit(
        tables_counted(&airs, &block, false),
        &config,
        Placement::PhaseB,
    )
    .expect("phase A");
    let taken = uncounted.local_digests().expect("local lookups");
    let recv = KINDS.iter().position(|&k| k == Kind::Recv).unwrap();
    assert_ne!(taken[recv], expected[recv], "the RECV's count is read");
}

/// ★ Wherever the local bins run, the bytes are the ones phase B alone proves
/// ([`BlockCommitted::with_local_lookups`]), and they verify: every bin on the
/// lane, every bin in phase B, the lane stopped after each bin index, every
/// job refused for memory, a unit failing partway (dropped whole: phase B
/// argues the group from its challenges again), and no lane at all. Under a
/// cap that puts each tree in its own bin and the production one, on the
/// single and the two-layer ladder.
#[test]
fn the_local_lane_moves_no_byte() {
    let airs = Airs::new(None);
    let block = Block::honest();
    for config in [
        production(),
        config(2, GkrPack::Off),
        config_with(2, GkrPack::Off, GkrLadder::Double),
    ] {
        let reference = bytes(
            &prove(&airs, &block, &config, Some(local(&airs, &block)), honest())
                .expect("phase B alone"),
        );
        let check = |placement: Placement| -> BlockProved<Fp, Ext> {
            let proved = prove_placed(&airs, &block, &config, placement, honest())
                .unwrap_or_else(|e| panic!("{placement:?}: {e:?}"));
            assert_eq!(bytes(&proved), reference, "{placement:?}: the bytes moved");
            verify_local(&airs, &block, &config, &proved)
                .unwrap_or_else(|e| panic!("{placement:?}: {e:?}"));
            proved
        };

        // Every bin on the lane: nothing left for phase B.
        let proved = check(lane(LaneLimits::default()));
        let stamps = lane_stamps(&proved);
        assert_eq!(stamps[0].0.state, LaneState::Off, "group 0: no local job");
        assert!(!stamps[0].0.local && stamps[0].1.is_empty());
        let mut most = 0;
        for (g, (job, left)) in stamps.iter().enumerate().skip(1) {
            assert!(job.local, "group {g}: a local job");
            assert_eq!(job.state, LaneState::Ran, "group {g}: {job:?}");
            assert!(
                job.bins > 0 && job.bins_ran == job.bins,
                "group {g}: {job:?}"
            );
            assert!(left.is_empty(), "group {g}: phase B ran {left:?}");
            most = most.max(job.bins);
        }
        let bins: usize = stamps.iter().map(|(job, _)| job.bins).sum();
        let lines = local_lane_lines(&proved.3);
        assert!(
            lines.contains(&format!(
                "BLOCK LOCAL LANE SUM: groups 2 · lane jobs 2 · bins {bins} of {bins} in phase A · \
                 ran 2 · "
            )),
            "{lines}"
        );

        // No lane: phase B runs every bin.
        for placement in [Placement::PhaseB, Placement::NoUpload] {
            let proved = check(placement);
            for (g, (job, left)) in lane_stamps(&proved).iter().enumerate().skip(1) {
                assert_eq!(job.state, LaneState::Off, "group {g}: {job:?}");
                assert!(!left.is_empty(), "group {g}: phase B ran its bins");
            }
        }

        // Stopped after each bin index: phase B runs the rest.
        for stop in 0..=most {
            let proved = check(lane(LaneLimits {
                bins: Some(stop),
                ..LaneLimits::default()
            }));
            for (g, (job, left)) in lane_stamps(&proved).iter().enumerate().skip(1) {
                let ran = stop.min(job.bins);
                assert_eq!(job.bins_ran, ran, "group {g}, stop {stop}: {job:?}");
                assert_eq!(left.len(), job.bins - ran, "group {g}, stop {stop}");
                let state = if ran == job.bins {
                    LaneState::Ran
                } else {
                    LaneState::Deferred(LaneDefer::Stopped)
                };
                assert_eq!(job.state, state, "group {g}, stop {stop}");
            }
        }

        // Refused for memory: every job deferred, phase B runs every bin.
        let proved = check(lane(LaneLimits {
            hold_bytes: Some(0),
            ..LaneLimits::default()
        }));
        for (g, (job, left)) in lane_stamps(&proved).iter().enumerate().skip(1) {
            assert_eq!(
                job.state,
                LaneState::Deferred(LaneDefer::Memory),
                "group {g}: {job:?}"
            );
            assert_eq!(job.bins_ran, 0);
            assert!(!left.is_empty());
        }

        // A unit failing after it ran, its transcript moved: the job is
        // dropped whole, and phase B argues the group anew.
        for fail in [0, 1] {
            let proved = check(lane(LaneLimits {
                fail_bin: Some(fail),
                ..LaneLimits::default()
            }));
            for (g, (job, left)) in lane_stamps(&proved).iter().enumerate().skip(1) {
                if fail < job.bins {
                    assert_eq!(job.state, LaneState::Failed, "group {g}: {job:?}");
                    assert_eq!(job.bins_ran, fail, "group {g}");
                    assert_eq!(left.len(), job.bins, "group {g}: every bin again");
                } else {
                    assert_eq!(job.state, LaneState::Ran, "group {g}: {job:?}");
                }
            }
        }
    }
}

/// A faulted argue, or a mutant binding (every group's), does not take the
/// argue the lane began: it starts over at the group's turn, as phase B alone
/// would, so the bytes are phase B's own faulted bytes and the verifier
/// refuses them.
#[test]
fn a_faulted_group_does_not_resume_the_lanes_argue() {
    let airs = Airs::new(None);
    let block = Block::honest();
    let config = config(2, GkrPack::Off);
    for faults in [
        ProverFaults {
            local_tamper_claim: Some(0),
            ..ProverFaults::default()
        },
        ProverFaults {
            local_swap_claims: Some((0, 1)),
            ..ProverFaults::default()
        },
        ProverFaults {
            local_binding: Some(LocalBinding {
                roots: false,
                ..LocalBinding::ALL
            }),
            ..ProverFaults::default()
        },
    ] {
        let alone = prove(
            &airs,
            &block,
            &config,
            Some(local(&airs, &block)),
            faulted(2, faults),
        )
        .expect("phase B alone");
        let laned = prove_placed(
            &airs,
            &block,
            &config,
            lane(LaneLimits::default()),
            faulted(2, faults),
        )
        .expect("the lane");
        assert_eq!(bytes(&laned), bytes(&alone), "{faults:?}");
        assert!(
            verify_local(&airs, &block, &config, &laned).is_err(),
            "{faults:?}"
        );
    }
    // A false lookup the lane argued is refused, as phase B's would be.
    let block = false_tuple();
    let proved = prove_placed(
        &airs,
        &block,
        &config,
        lane(LaneLimits::default()),
        honest(),
    )
    .expect("the lane");
    assert!(matches!(
        verify_local(&airs, &block, &config, &proved),
        Err(MlError::LocalImbalance { group: 1 })
    ));
}

/// ★ The challenges phase A draws come after the group's roots are in its
/// local transcript ([`multilinear_block::local_challenges`]): the lane's
/// bytes are the verifier's rebuild from the proof's roots. A phase A that
/// drew them before the roots — the binding without them, on the prover's
/// side alone, every bin then run from that start on the lane or in phase B
/// — sends other bytes, and the verifier refuses them.
#[test]
fn the_lanes_challenges_come_after_the_roots() {
    let airs = Airs::new(None);
    let block = Block::honest();
    let config = config(2, GkrPack::Off);
    let honest_proof = prove_placed(
        &airs,
        &block,
        &config,
        lane(LaneLimits::default()),
        honest(),
    )
    .expect("the lane");
    verify_local(&airs, &block, &config, &honest_proof).expect("verifies");
    let before_the_roots = LocalBinding {
        roots: false,
        ..LocalBinding::ALL
    };
    for placement in [lane(LaneLimits::default()), Placement::PhaseB] {
        let committed = commit_bound(tables(&airs, &block), &config, placement, before_the_roots)
            .expect("phase A");
        let order: Vec<usize> = (0..SIZES.len()).collect();
        let mut transcript = H::sponge();
        let mutant = multilinear_block::block_prove_in_order::<Fp, Ext, _, H>(
            committed,
            &config,
            &mut transcript,
            &[],
            &[],
            &|g| g,
            &honest(),
            &|_| {},
            &order,
            None,
        )
        .expect("the mutant proves");
        assert_ne!(bytes(&mutant), bytes(&honest_proof), "{placement:?}");
        assert!(
            verify_local(&airs, &block, &config, &mutant).is_err(),
            "{placement:?}: challenges drawn before the roots verified"
        );
    }
}

/// The readout: a line a local group (its lane job, or phase B's bins alone),
/// then the sum, on a fixed set of stamps; empty with no local group.
#[test]
fn the_local_lane_lines_add_up() {
    let job = |state, bins, bins_ran, ladders| ZcLaneStamp {
        state,
        bins,
        bins_ran,
        ladders,
        local: true,
        store_bytes: 3 << 20,
        commit_wait: 0.125,
        ended_at: 2.0,
        ..ZcLaneStamp::default()
    };
    let stamps = [
        GroupStamps::default(),
        GroupStamps {
            zc_lane: job(LaneState::Ran, 2, 2, 0.5),
            ..GroupStamps::default()
        },
        GroupStamps {
            zc_lane: job(LaneState::Deferred(LaneDefer::Tail), 3, 1, 0.25),
            local_b: vec![0.25, 0.5],
            ..GroupStamps::default()
        },
        GroupStamps {
            local_b: vec![1.0],
            ..GroupStamps::default()
        },
    ];
    let lines = local_lane_lines(&stamps);
    assert_eq!(
        lines,
        "BLOCK LOCAL LANE: group 1 · ran · bins 2 of 2 in phase A · lane 0.500s · store 3 MiB · \
         commit wait 0.125s · phase B 0.000s []\n\
         BLOCK LOCAL LANE: group 2 · tail · bins 1 of 3 in phase A · lane 0.250s · store 3 MiB · \
         commit wait 0.125s · phase B 0.750s [0.2500 0.5000]\n\
         BLOCK LOCAL LANE: group 3 · off · bins 0 of 1 in phase A · lane 0.000s · store 0 MiB · \
         commit wait 0.000s · phase B 1.000s [1.0000]\n\
         BLOCK LOCAL LANE SUM: groups 3 · lane jobs 2 · bins 3 of 6 in phase A · ran 1 · deferred \
         hold 0 · memory 0 · tail 1 · shed 0 · no store 0 · stopped 0 · declined 0 · failed 0 · \
         lane 0.750s · phase B 1.750s · commits waited 0.250s (most 0.125s) · last job ended at \
         2.000s · phase A's end waited 0.000s\n"
    );
    assert_eq!(local_lane_lines(&[GroupStamps::default()]), "");
}
