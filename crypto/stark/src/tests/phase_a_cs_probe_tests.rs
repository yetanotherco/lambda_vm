//! Phase A's constraint-sumcheck probe ([`CsProbe`]): a measurement that runs
//! each group's constraint sumcheck after its commit on a throwaway transcript
//! and moves no byte of the proof. On the small block of
//! [`super::phase_b_overlap_tests`], with no card, so the probe runs on the
//! host.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::time::{Duration, Instant};

use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as Fp,
};
use multilinear::{
    whir_chain::{ArgueFormat, ChainConfig},
    whir_commit::TreeDrop,
    whir_hash::WhirHash,
};

use super::phase_b_overlap_tests::{
    ARGUES, BEFORE_THE_OVERLAP, H, SIZES, config, digest, in_order, prove_block_with, verify,
};
use crate::multilinear_block::{
    BlockCommitted, CsProbe, CsProbeMode, CsProbeStamp, CsProbeState, GroupStamps, LaneDefer,
    LaneLimits, LaneState, LaneWork, MergeStamp, OverlapFaults, PhaseBOverlap, ZcLaneStamp,
    cs_probe_lines, merge_probe_lines, zc_lane_lines,
};
use crate::multilinear_table::CommittedTable;

/// Phase A over the block's groups, taken one at a time beside each commit,
/// under `probe`. With a `decided` counter, group `k` is handed over only once
/// group `k − 1`'s probe was decided (and the producer's end only once the
/// last group's was), so no group is waiting when a commit ends.
fn commit_probed<'a>(
    tables: Vec<CommittedTable<'a, Fp, Ext>>,
    config: &ChainConfig,
    probe: &CsProbe,
) -> BlockCommitted<'a, Fp, Ext> {
    let mut tables = tables.into_iter();
    let mut groups = SIZES
        .iter()
        .map(|&size| tables.by_ref().take(size).collect::<Vec<_>>())
        .collect::<Vec<_>>()
        .into_iter();
    let decided = probe.decided.clone();
    let mut handed = 0usize;
    let gated = std::iter::from_fn(move || {
        if let Some(decided) = &decided
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
        crate::multilinear_block::Narrowing::Wide,
        true,
        None,
        None,
        probe.clone(),
        None,
    )
    .expect("phase A")
}

/// The block proved in group order, phase A under `probe`.
fn prove_probed(
    argue: ArgueFormat,
    probe: CsProbe,
) -> crate::multilinear_block::BlockProved<Fp, Ext> {
    let (proved, _) = prove_block_with::<H>(
        &config(argue),
        PhaseBOverlap::Off,
        &in_order(),
        OverlapFaults::default(),
        &|_| {},
        None,
        false,
        Some(&|tables, config| commit_probed(tables, config, &probe)),
    );
    proved.unwrap_or_else(|e| panic!("{argue:?}: {e:?}"))
}

fn probes(stamps: &[GroupStamps]) -> Vec<CsProbeStamp> {
    stamps.iter().map(|g| g.cs_probe).collect()
}

/// ★ The probe moves no byte. Under the batched argue it runs after every
/// group's commit (no group is waiting), each over the group's tables, and the
/// proof is the one pinned before the probe existed, and it verifies. Off, or
/// under the per-table argue it does not apply to, no group is probed, and the
/// bytes are the same.
#[test]
fn the_probe_runs_after_each_commit_and_moves_no_byte() {
    let decided = Arc::new(AtomicUsize::new(0));
    let on = CsProbe {
        mode: CsProbeMode::AfterCommit,
        host: true,
        decided: Some(decided.clone()),
        ..Default::default()
    };
    let batched = ArgueFormat::BATCHED;
    let before = BEFORE_THE_OVERLAP[1];
    assert_eq!(ARGUES[1], batched, "the pin's format");

    let proved = prove_probed(batched, on.clone());
    assert_eq!(digest(&proved), before, "probed: the bytes moved");
    verify(batched, &proved).expect("probed: verifies");
    let stamps = &proved.3;
    assert_eq!(decided.load(SeqCst), SIZES.len(), "every group decided");
    for (g, (probe, &size)) in probes(stamps).iter().zip(&SIZES).enumerate() {
        assert_eq!(probe.state, CsProbeState::Ran, "group {g}: {probe:?}");
        assert_eq!(probe.tables, size, "group {g}: its tables");
        assert!(probe.on_card <= size, "group {g}: {probe:?}");
        assert!(
            probe.rounds > 0.0 && probe.secs >= probe.rounds,
            "group {g}: {probe:?}"
        );
        assert!(probe.held <= probe.secs + 1e-9, "group {g}: {probe:?}");
        assert!(
            probe.ended_at >= probe.committed_at && stamps[g].committed_at == probe.committed_at,
            "group {g}: the commit's end is the commit's: {probe:?}"
        );
    }
    let lines = cs_probe_lines(stamps);
    assert!(
        lines.starts_with(&format!(
            "BLOCK CS PROBE: groups {} · ran {} · skipped with a group waiting 0 · declined 0 · \
             failed 0 · tables {} (",
            SIZES.len(),
            SIZES.len(),
            SIZES.iter().sum::<usize>()
        )),
        "{lines}"
    );

    for (argue, before, what) in [
        (batched, BEFORE_THE_OVERLAP[1], "off"),
        (ArgueFormat::PerTable, BEFORE_THE_OVERLAP[0], "per table"),
    ] {
        let probe = if what == "off" {
            CsProbe::default()
        } else {
            on.clone()
        };
        let proved = prove_probed(argue, probe);
        assert_eq!(digest(&proved), before, "{what}: the bytes moved");
        verify(argue, &proved).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert!(
            probes(&proved.3)
                .iter()
                .all(|p| p.state == CsProbeState::Off),
            "{what}: no group probed"
        );
        assert_eq!(cs_probe_lines(&proved.3), "", "{what}: no line");
    }
}

/// Without a card a group has no store, and a probe that may not run on the
/// host declines it: nothing runs, and the bytes are the same.
#[test]
fn a_group_with_no_store_is_declined_off_the_host() {
    let probe = CsProbe {
        mode: CsProbeMode::AfterCommit,
        host: false,
        decided: Some(Arc::new(AtomicUsize::new(0))),
        ..Default::default()
    };
    let proved = prove_probed(ArgueFormat::BATCHED, probe);
    assert_eq!(digest(&proved), BEFORE_THE_OVERLAP[1]);
    let probes = probes(&proved.3);
    assert!(
        probes
            .iter()
            .all(|p| p.state == CsProbeState::Declined && p.rounds == 0.0),
        "{probes:?}"
    );
    assert!(
        cs_probe_lines(&proved.3).contains("· ran 0 · skipped with a group waiting 0 · declined 7"),
        "{}",
        cs_probe_lines(&proved.3)
    );
}

/// `LAMBDA_VM_PHASE_A_CS_PROBE`: unset, empty or `0` off, `1` after each
/// commit, `queue` the lane; `LAMBDA_VM_PHASE_B_MERGE_PROBE`: unset, empty or
/// `0` off, `1` on. Any other word is refused, never read as off.
#[test]
fn the_knobs_read_their_words_and_refuse_the_rest() {
    let read = |a, b| CsProbe::from_values(a, b).map(|p| (p.mode, p.merge));
    assert_eq!(read(None, None), Ok((CsProbeMode::Off, false)));
    assert_eq!(read(Some("0"), Some("0")), Ok((CsProbeMode::Off, false)));
    assert_eq!(read(Some(""), Some("")), Ok((CsProbeMode::Off, false)));
    assert_eq!(read(Some("1"), None), Ok((CsProbeMode::AfterCommit, false)));
    assert_eq!(
        read(Some(" 1 "), None),
        Ok((CsProbeMode::AfterCommit, false))
    );
    assert_eq!(
        read(Some("queue"), None),
        Ok((CsProbeMode::Lane(LaneWork::Zc), false))
    );
    assert_eq!(
        read(Some("queue"), Some("1")),
        Ok((CsProbeMode::Lane(LaneWork::Zc), true))
    );
    assert_eq!(read(None, Some("1")), Ok((CsProbeMode::Off, true)));
    assert_eq!(
        read(Some("ladders"), None),
        Ok((CsProbeMode::Lane(LaneWork::Ladders), false))
    );
    assert_eq!(
        read(Some("ladders+zc"), None),
        Ok((CsProbeMode::Lane(LaneWork::LaddersZc), false))
    );
    for bad in [
        "on",
        "2",
        "Queue",
        "lane",
        "true",
        "ladder",
        "zc+ladders",
        "ladders zc",
    ] {
        assert!(read(Some(bad), None).is_err(), "phase A {bad:?}");
        assert!(read(None, Some(bad)).is_err(), "merge {bad:?}");
    }
    assert!(!CsProbe::default().any());
    assert!(CsProbe::from_values(None, Some("1")).unwrap().any());
}

/// The readout's hidden seconds are the ran groups' seconds less what they
/// held back; a group skipped with a group waiting counts as skipped.
#[test]
fn the_readout_counts_hidden_seconds_and_skips() {
    let stamp = |state, secs, held| GroupStamps {
        commit_reserved: 3 << 20,
        cs_probe: CsProbeStamp {
            state,
            tables: 2,
            secs,
            setup: secs / 2.0,
            rounds: secs / 2.0,
            held,
            ended_at: 1.5,
            ..Default::default()
        },
        ..Default::default()
    };
    let stamps = [
        stamp(CsProbeState::Ran, 0.5, 0.0),
        stamp(CsProbeState::Waiting, 0.0, 0.0),
        stamp(CsProbeState::Ran, 0.25, 0.125),
    ];
    let lines = cs_probe_lines(&stamps);
    assert!(
        lines.starts_with(
            "BLOCK CS PROBE: groups 3 · ran 2 · skipped with a group waiting 1 · declined 0 · \
             failed 0 · tables 4 (0 on the card) · sumcheck 0.750s (setup 0.375, rounds 0.375, \
             host tail 0.000) · hidden 0.625s · held the next commit 0.125s · last ran ended at \
             1.500s · phase-A ledger peak 3 MiB\n"
        ),
        "{lines}"
    );
    assert!(
        lines.ends_with(
            "BLOCK CS PROBE GROUPS (group:state:s:held): 0:ran:0.5000:0.0000 \
             1:wait:0.0000:0.0000 2:ran:0.2500:0.1250\n"
        ),
        "{lines}"
    );
}

/// The lane's probe, gated so each group's job is decided before the next
/// group is handed over.
fn lane(merge: bool, host: bool, limits: LaneLimits) -> (CsProbe, Arc<AtomicUsize>) {
    lane_doing(LaneWork::Zc, merge, host, limits)
}

/// [`lane`] with each job running `work`.
fn lane_doing(
    work: LaneWork,
    merge: bool,
    host: bool,
    limits: LaneLimits,
) -> (CsProbe, Arc<AtomicUsize>) {
    let decided = Arc::new(AtomicUsize::new(0));
    (
        CsProbe {
            mode: CsProbeMode::Lane(work),
            host,
            decided: Some(decided.clone()),
            merge,
            limits,
        },
        decided,
    )
}

/// ★ The lane (`queue`) moves no byte. On the host, each group's job is
/// handed to the lane after its commit and runs there — the constraint
/// sumcheck alone, over the group's tables — and the proof is the one pinned
/// before any probe existed, and it verifies. Phase B records each group's
/// own constraint sumcheck, and the hidden credit is their sum over the groups
/// whose job ran.
#[test]
fn the_lane_runs_each_group_and_moves_no_byte() {
    let (probe, decided) = lane(false, true, LaneLimits::default());
    let proved = prove_probed(ArgueFormat::BATCHED, probe);
    assert_eq!(
        digest(&proved),
        BEFORE_THE_OVERLAP[1],
        "the lane: the bytes moved"
    );
    verify(ArgueFormat::BATCHED, &proved).expect("the lane: verifies");
    assert_eq!(decided.load(SeqCst), SIZES.len(), "every group decided");
    let stamps = &proved.3;
    for (g, (s, &size)) in stamps.iter().zip(&SIZES).enumerate() {
        let job = &s.zc_lane;
        assert_eq!(job.state, LaneState::Ran, "group {g}: {job:?}");
        assert_eq!(job.tables, size, "group {g}: its tables");
        assert!(job.rounds > 0.0, "group {g}: {job:?}");
        assert!(
            job.committed_at <= job.started_at && job.started_at <= job.ended_at,
            "group {g}: after its commit: {job:?}"
        );
        assert!(s.zc_b > 0.0, "group {g}: phase B's own sumcheck");
        assert_eq!(
            s.cs_probe.state,
            CsProbeState::Off,
            "group {g}: no probe after the commit"
        );
        assert_eq!(s.merge.state, LaneState::Off, "group {g}: no merge");
    }
    let hidden: f64 = stamps.iter().map(|s| s.zc_b).sum();
    let lines = zc_lane_lines(stamps);
    assert!(
        lines.contains(&format!(
            "BLOCK ZC LANE SUM: groups {n} · ran {n} · deferred 0 (hold 0 · memory 0 · tail 0 · no \
             store 0) · declined 0 · failed 0 · hidden credit {hidden:.3}s · ",
            n = SIZES.len()
        )),
        "{lines}"
    );
    assert_eq!(
        lines.matches("BLOCK ZC LANE: group ").count(),
        SIZES.len(),
        "{lines}"
    );
    assert_eq!(cs_probe_lines(stamps), "");
    assert_eq!(merge_probe_lines(stamps), "");
}

/// ★ The lane and phase B's merge probe together move no byte:
/// after each group's argue the merge runs over the group's tables on a
/// throwaway copy of its fork, and the proof is the pinned one and verifies.
/// The merge alone does too, and under the per-table argue it does not run.
#[test]
fn the_merge_probe_runs_after_each_argue_and_moves_no_byte() {
    let (probe, _) = lane(true, true, LaneLimits::default());
    let proved = prove_probed(ArgueFormat::BATCHED, probe);
    assert_eq!(
        digest(&proved),
        BEFORE_THE_OVERLAP[1],
        "lane + merge: the bytes moved"
    );
    verify(ArgueFormat::BATCHED, &proved).expect("lane + merge: verifies");
    for (g, (s, &size)) in proved.3.iter().zip(&SIZES).enumerate() {
        assert_eq!(s.zc_lane.state, LaneState::Ran, "group {g}");
        assert_eq!(s.merge.state, LaneState::Ran, "group {g}: {:?}", s.merge);
        assert_eq!(s.merge.tables, size, "group {g}: its tables");
        assert!(s.merge.rounds > 0.0, "group {g}: {:?}", s.merge);
    }
    let lines = merge_probe_lines(&proved.3);
    assert!(
        lines.starts_with(&format!(
            "BLOCK MERGE PROBE: groups {} · ran {} · declined 0 · failed 0 · tables {} (",
            SIZES.len(),
            SIZES.len(),
            SIZES.iter().sum::<usize>()
        )),
        "{lines}"
    );

    let merge_only = CsProbe {
        merge: true,
        host: true,
        ..Default::default()
    };
    let proved = prove_probed(ArgueFormat::BATCHED, merge_only.clone());
    assert_eq!(
        digest(&proved),
        BEFORE_THE_OVERLAP[1],
        "merge alone: the bytes moved"
    );
    verify(ArgueFormat::BATCHED, &proved).expect("merge alone: verifies");
    assert!(
        proved
            .3
            .iter()
            .all(|s| s.merge.state == LaneState::Ran && s.zc_lane.state == LaneState::Off)
    );
    assert_eq!(zc_lane_lines(&proved.3), "");

    let proved = prove_probed(ArgueFormat::PerTable, merge_only);
    assert_eq!(
        digest(&proved),
        BEFORE_THE_OVERLAP[0],
        "per table: the bytes moved"
    );
    verify(ArgueFormat::PerTable, &proved).expect("per table: verifies");
    assert!(
        proved
            .3
            .iter()
            .all(|s| s.merge.state == LaneState::Off && s.zc_b == 0.0)
    );
    assert_eq!(merge_probe_lines(&proved.3), "");
}

/// The lane holds a group only while it may: a full ledger (no bytes to hold),
/// no room for a job (hold 0), or no store and no host each defer every
/// group, by name; nothing runs, and the bytes are the same.
///
/// On CUDA, run with column uploads disabled so the no-store case can
/// exercise the same deferral as the host build.
#[test]
fn the_lane_defers_a_group_it_may_not_hold() {
    #[cfg(feature = "cuda")]
    if !multilinear::gpu::columns_disabled() {
        // A CUDA worker can upload this fixture's columns. Configure a
        // child process before the upload switch is read so the NoStore
        // arm runs without a store on both builds.
        let (_, module) = module_path!().split_once("::").expect("the test module");
        let test = format!("{module}::the_lane_defers_a_group_it_may_not_hold");
        let out = std::process::Command::new(std::env::current_exe().expect("the test binary"))
            .args(["--exact", &test, "--nocapture"])
            .env("LAMBDA_VM_NO_GPU_COLUMNS", "1")
            .output()
            .expect("run the lane test without a column store");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success() && stdout.contains("1 passed"),
            "{test} did not pass without a column store\n{stdout}\n{stderr}"
        );
        return;
    }
    for (limits, host, why) in [
        (
            LaneLimits {
                hold_bytes: Some(0),
                ..LaneLimits::default()
            },
            true,
            LaneDefer::Memory,
        ),
        (
            LaneLimits {
                hold: 0,
                ..LaneLimits::default()
            },
            true,
            LaneDefer::Hold,
        ),
        (LaneLimits::default(), false, LaneDefer::NoStore),
    ] {
        let (probe, decided) = lane(false, host, limits);
        let proved = prove_probed(ArgueFormat::BATCHED, probe);
        assert_eq!(
            digest(&proved),
            BEFORE_THE_OVERLAP[1],
            "{why:?}: the bytes moved"
        );
        assert_eq!(
            decided.load(SeqCst),
            SIZES.len(),
            "{why:?}: every group decided"
        );
        for (g, s) in proved.3.iter().enumerate() {
            assert_eq!(s.zc_lane.state, LaneState::Deferred(why), "group {g}");
            assert_eq!(s.zc_lane.rounds, 0.0, "group {g}");
        }
        let lines = zc_lane_lines(&proved.3);
        assert!(
            lines.contains(&format!("· ran 0 · deferred {} (", SIZES.len()))
                && lines.contains("· hidden credit 0.000s ·"),
            "{why:?}: {lines}"
        );
    }
}

/// The lane's readout: the hidden credit is phase B's own constraint sumcheck
/// summed over the groups whose job ran, and only those; the groups not run
/// are listed with theirs.
#[test]
fn the_lane_readout_credits_only_the_groups_that_ran() {
    let stamp = |state, zc_b| GroupStamps {
        commit_reserved: 5 << 20,
        zc_b,
        zc_lane: ZcLaneStamp {
            state,
            zc_ran: state == LaneState::Ran,
            tables: 2,
            setup: 0.125,
            rounds: 0.25,
            ended_at: 1.5,
            ..Default::default()
        },
        ..Default::default()
    };
    let stamps = [
        stamp(LaneState::Ran, 0.5),
        stamp(LaneState::Deferred(LaneDefer::Tail), 0.75),
        stamp(LaneState::Ran, 0.25),
        stamp(LaneState::Declined, 1.0),
    ];
    let lines = zc_lane_lines(&stamps);
    assert!(
        lines.contains(
            "BLOCK ZC LANE SUM: groups 4 · ran 2 · deferred 1 (hold 0 · memory 0 · tail 1 · no store 0) \
             · declined 1 · failed 0 · hidden credit 0.750s · lane card 0.750s (setup 0.250, rounds \
             0.500, host tail 0.000) · zc in phase B 2.500s (of groups not run 1.750s) · last job ended \
             at 1.500s · phase A's end waited 0.000s · phase-A ledger peak 5 MiB · deferred groups \
             [1:tail:0.750 3:declined:1.000] · ladders: bins 0 of 0 ·"
        ),
        "{lines}"
    );

    let merged = |state, secs: f64| GroupStamps {
        zc_b: 1.0,
        merge: MergeStamp {
            state,
            tables: 3,
            on_card: 3,
            setup: secs / 2.0,
            rounds: secs / 2.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let lines = merge_probe_lines(&[
        merged(LaneState::Ran, 0.5),
        merged(LaneState::Declined, 0.0),
    ]);
    assert_eq!(
        lines,
        "BLOCK MERGE PROBE: groups 2 · ran 1 · declined 1 · failed 0 · tables 3 (3 on the card) · \
         merge 0.500s (setup 0.250, rounds 0.250, host tail 0.000) · zc in phase B 2.000s\n\
         BLOCK MERGE PROBE GROUPS (group:state:s): 0:ran:0.5000 1:declined:0.0000\n"
    );
}

/// ★ The lane's ladders (`ladders`, and `ladders+zc`) move no byte. On the
/// host each group's job runs every bin of the batched argue's plan, one at a
/// time, on throwaway challenges (and, with `+zc`, the constraint sumcheck
/// after them); the proof is the pinned one and verifies. Phase B records each
/// bin's own seconds, and the ladders' hidden credit is their sum over the bins
/// the lane ran — here all of them.
#[test]
fn the_lane_runs_each_groups_ladders_and_moves_no_byte() {
    for work in [LaneWork::Ladders, LaneWork::LaddersZc] {
        let (probe, decided) = lane_doing(work, false, true, LaneLimits::default());
        let proved = prove_probed(ArgueFormat::BATCHED, probe);
        assert_eq!(
            digest(&proved),
            BEFORE_THE_OVERLAP[1],
            "{work:?}: the bytes moved"
        );
        verify(ArgueFormat::BATCHED, &proved).unwrap_or_else(|e| panic!("{work:?}: {e:?}"));
        assert_eq!(
            decided.load(SeqCst),
            SIZES.len(),
            "{work:?}: every group decided"
        );
        let stamps = &proved.3;
        for (g, s) in stamps.iter().enumerate() {
            let job = &s.zc_lane;
            assert_eq!(job.state, LaneState::Ran, "{work:?} group {g}: {job:?}");
            assert!(
                job.bins >= 1 && job.bins_ran == job.bins,
                "{work:?} group {g}: {job:?}"
            );
            assert!(job.ladders > 0.0, "{work:?} group {g}: {job:?}");
            assert_eq!(
                s.bins_b.len(),
                job.bins,
                "{work:?} group {g}: phase B's bins"
            );
            assert_eq!(
                job.zc_ran,
                work == LaneWork::LaddersZc,
                "{work:?} group {g}"
            );
            assert_eq!(
                job.rounds > 0.0,
                work == LaneWork::LaddersZc,
                "{work:?} group {g}"
            );
        }
        assert!(
            stamps
                .iter()
                .all(|s| s.zc_lane.commit_wait >= 0.0 && s.zc_lane.commit_wait.is_finite()),
            "{work:?}: every commit's wait for a unit stamped"
        );
        let credit: f64 = stamps.iter().flat_map(|s| s.bins_b.iter()).sum();
        let bins: usize = stamps.iter().map(|s| s.zc_lane.bins).sum();
        let hidden: f64 = if work == LaneWork::LaddersZc {
            stamps.iter().map(|s| s.zc_b).sum()
        } else {
            0.0
        };
        let lines = zc_lane_lines(stamps);
        assert!(
            lines.contains(&format!("· hidden credit {hidden:.3}s ·"))
                && lines.contains(&format!("· ladders: bins {bins} of {bins} · lane ladders "))
                && lines.contains(&format!("· ladders hidden credit {credit:.3}s ·")),
            "{work:?}: {lines}"
        );
    }
}

/// The ladders' credit counts the bins the lane ran and only those: the
/// first `bins_ran` of a group's phase-B bins.
#[test]
fn the_ladders_credit_counts_only_the_bins_run() {
    let stamp = |bins_ran, bins_b: Vec<f64>| GroupStamps {
        bins_b,
        zc_lane: ZcLaneStamp {
            state: LaneState::Deferred(LaneDefer::Tail),
            bins: 3,
            bins_ran,
            ladders: 0.5,
            started_at: 1.0,
            ended_at: 1.25,
            ..Default::default()
        },
        ..Default::default()
    };
    let lines = zc_lane_lines(&[
        stamp(2, vec![0.5, 0.25, 1.0]),
        stamp(0, vec![2.0, 2.0, 2.0]),
    ]);
    assert!(
        lines.contains(
            "· ladders: bins 2 of 6 · lane ladders 1.000s · ladders hidden credit 0.750s · ladders in \
             phase B 7.750s · lane busy 0.500s · shed 0 · commits waited 0.000s (most 0.000s)\n"
        ),
        "{lines}"
    );
}
