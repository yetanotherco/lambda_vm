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
    BlockCommitted, CsProbe, CsProbeStamp, CsProbeState, GroupStamps, OverlapFaults, PhaseBOverlap,
    cs_probe_lines,
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
        on: true,
        host: true,
        decided: Some(decided.clone()),
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
        on: true,
        host: false,
        decided: Some(Arc::new(AtomicUsize::new(0))),
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

/// `LAMBDA_VM_PHASE_A_CS_PROBE`: `1` is on; unset, `0` or another word is off.
#[test]
fn the_knob_reads_one_as_on() {
    assert!(CsProbe::from_value(Some("1")).on);
    assert!(CsProbe::from_value(Some(" 1 ")).on);
    for value in [None, Some("0"), Some("on"), Some("")] {
        assert!(!CsProbe::from_value(value).on, "{value:?}");
    }
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
