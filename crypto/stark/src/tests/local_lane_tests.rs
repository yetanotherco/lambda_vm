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
    /// Card 0's lane and the second lane, both on the host, under these
    /// limits; each group's copy bound for the second lane (a group with a
    /// table past it is not light, and stays on card 0's lane).
    Second(LaneLimits, [usize; 3]),
    /// The lane with the second card asked for as production asks
    /// (`LocalPhaseA::second_card`, no test hook): without the two-card
    /// split on and ready, no second lane.
    SecondAsked,
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
    commit_counted(tables, config, placement, binding).0
}

/// [`commit_bound`], with the count of groups decided (each group's job
/// run or refused, or the group passed by every lane) when phase A ended.
fn commit_counted<'a>(
    tables: Vec<CommittedTable<'a, Fp, Ext>>,
    config: &ChainConfig,
    placement: Placement,
    binding: LocalBinding,
) -> (Result<BlockCommitted<'a, Fp, Ext>, MlError>, usize) {
    let mut tables = tables.into_iter();
    let mut groups = SIZES
        .iter()
        .map(|&size| tables.by_ref().take(size).collect::<Vec<_>>())
        .collect::<Vec<_>>()
        .into_iter();
    let (lane, limits, second_card, second) = match placement {
        Placement::Lane(limits) => (true, limits, false, None),
        Placement::Second(limits, bounds) => (true, limits, true, Some(bounds.to_vec())),
        Placement::SecondAsked => (true, LaneLimits::default(), true, None),
        _ => (false, LaneLimits::default(), false, None),
    };
    let decided = lane.then(|| Arc::new(AtomicUsize::new(0)));
    let count = decided.clone();
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
    let committed = BlockCommitted::commit_groups_probed::<H, <H as WhirHash>::Sponge>(
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
            second_card,
            binding,
            second_on_host: second,
        }),
    );
    (committed, count.map_or(0, |count| count.load(SeqCst)))
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

/// Phase B split over two drivers on the host (the two-card split's schedule
/// without a card): a group whose local argue began on the lane in phase A is
/// host data, so it is not kept on card 0; a group of it finishes its argue on
/// the second driver, and the proof is the one-driver proof, byte for byte,
/// and verifies.
#[test]
fn a_lanes_begun_argue_finishes_on_the_second_driver() {
    let airs = Airs::new(None);
    let block = Block::honest();
    let config = production();
    let order: Vec<usize> = (0..SIZES.len()).collect();
    let prove_with = |split: bool| -> BlockProved<Fp, Ext> {
        let mut committed =
            commit(tables(&airs, &block), &config, lane(LaneLimits::default())).expect("phase A");
        if split {
            committed.set_two_card_split(Some(vec![false; SIZES.len()]), true);
        }
        let mut transcript = H::sponge();
        multilinear_block::block_prove_in_order::<Fp, Ext, _, H>(
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
        .expect("phase B")
    };
    let one = prove_with(false);
    let two = prove_with(true);
    let cards: Vec<u8> = two.3.iter().map(|s| s.card).collect();
    let began_on_card_1 = two
        .3
        .iter()
        .any(|s| s.card == 1 && s.zc_lane.local && s.zc_lane.state == LaneState::Ran);
    assert!(
        began_on_card_1,
        "no group whose argue began on the lane finished on the second driver: cards {cards:?}"
    );
    assert_eq!(bytes(&two), bytes(&one), "two drivers moved a byte");
    verify_local(&airs, &block, &config, &two).expect("two drivers verify");
}

/// Every group light: the production copy bound for each.
const LIGHT: [usize; 3] = [usize::MAX; 3];

/// Group 2 not light (no table may be copied): it stays on card 0's lane.
const MIXED: [usize; 3] = [usize::MAX, usize::MAX, 0];

fn second(limits: LaneLimits, bounds: [usize; 3]) -> Placement {
    Placement::Second(limits, bounds)
}

/// The cells of a test table's columns, `5 × 2^3`.
const CELLS: usize = 5 << 3;

/// Host fills of [`OwnRecords`]: a reader that asked for its columns on the
/// host. Its own counter, so no other test's fills move it.
static OWN_HOST_FILLS: AtomicUsize = AtomicUsize::new(0);

/// Records of a test table, `8 × 5`, whose host fill counts itself.
#[derive(Debug)]
struct OwnRecords(Vec<multilinear::records::KeccakRndOp>);

impl multilinear::records::TableRecords for OwnRecords {
    fn rows(&self) -> usize {
        8
    }
    fn cols(&self) -> usize {
        5
    }
    fn fill(&self) -> multilinear::records::RecordsFill<'_> {
        multilinear::records::RecordsFill::KeccakRnd {
            ops: &self.0,
            skip: 0,
        }
    }
    fn host_fill(&self) -> Option<multilinear::narrow::NarrowColumns> {
        OWN_HOST_FILLS.fetch_add(1, SeqCst);
        None
    }
}

/// L1. A second-card job's columns ([`CommittedTable::lane_columns`]): a
/// table held as records gives the same handle (whatever its size), and
/// never fills on the host; a small wide table an equal copy; a small packed
/// table its packed bytes (never widened); a table past the bound, wide or
/// packed, `None`. The shell ([`CommittedTable::lane_shell`]) keeps the
/// shape and no device copy.
#[test]
fn a_second_card_job_owns_its_columns() {
    use crate::examples::multi_table_lookup::new_cpu_air_with_lookup;
    use crate::multilinear_table::LaneColumns;
    use crate::proof::options::ProofOptions;
    use math::field::element::FieldElement;
    use multilinear::records::{KeccakRndOp, Records};

    let air = new_cpu_air_with_lookup(&ProofOptions::default_test_options());
    let layout = || {
        TableLayout::<Fp, Ext>::new(
            air.constraint_program(),
            air.constraints_meta(),
            air.bus_interactions(),
            5,
            3,
            Uniforms::default(),
        )
        .expect("a layout")
    };
    let wide = || {
        CommittedTable::<Fp, Ext>::new(
            air.constraint_program(),
            air.constraints_meta(),
            air.bus_interactions(),
            5,
            3,
            Uniforms::default(),
            |col| {
                (0..8u64)
                    .map(|r| FieldElement::from(r * 5 + u64::from(col)))
                    .collect()
            },
        )
        .expect("a table")
    };
    let records = Records(Arc::new(OwnRecords(vec![
        KeccakRndOp {
            timestamp: 3,
            input: [7; 25],
        };
        2
    ])));
    let held = CommittedTable::from_records(layout(), records.clone()).expect("a table");
    let fills = OWN_HOST_FILLS.load(SeqCst);
    for bound in [0, CELLS, usize::MAX] {
        match held.lane_columns(bound) {
            Some(LaneColumns::Records(taken)) => {
                assert!(Arc::ptr_eq(&taken.0, &records.0), "the same handle")
            }
            _ => panic!("bound {bound}: a records table gives its handle"),
        }
    }
    let shell = held.lane_shell(false);
    assert_eq!(
        OWN_HOST_FILLS.load(SeqCst),
        fills,
        "records filled on the host"
    );
    assert_eq!(
        (shell.num_committed_columns(), shell.num_vars()),
        (5, 3),
        "the shell's shape"
    );
    assert!(shell.trace().resident().is_none(), "no device copy");
    assert!(shell.trace().records().is_none() && shell.trace().narrow().is_none());

    let table = wide();
    match table.lane_columns(CELLS) {
        Some(LaneColumns::Wide(copy)) => assert_eq!(copy, table.columns().to_vec()),
        _ => panic!("a small wide table gives a copy"),
    }
    assert!(table.lane_columns(CELLS - 1).is_none(), "past the bound");

    let mut packed = wide();
    assert!(packed.pack_on_host(), "packs");
    let bytes = packed.trace().narrow().expect("packed").clone();
    match packed.lane_columns(CELLS) {
        Some(LaneColumns::Narrow(copy)) => assert_eq!(copy, bytes),
        _ => panic!("a small packed table gives its packed bytes, not widened"),
    }
    assert!(packed.lane_columns(CELLS - 1).is_none(), "past the bound");
}

/// Each local group's stamps: `(lane card, state, bins, bins run, bins left
/// to phase B)`.
fn second_stamps(proved: &BlockProved<Fp, Ext>) -> Vec<(u8, LaneState, usize, usize, usize)> {
    lane_stamps(proved)
        .iter()
        .map(|(job, left)| (job.card, job.state, job.bins, job.bins_ran, left.len()))
        .collect()
}

/// ★ L2, L4. The second lane moves no byte: with every local group on it,
/// with the mixed group on card 0's lane, stopped after each bin index,
/// refused for memory as its job starts, and a unit failing partway (the
/// group argued anew in phase B), the proof is phase B's alone, the one-lane
/// proof's, and it verifies. On the host the second lane runs its jobs
/// without a store; under the card's build with a card it uploads its own
/// (a wrong column offset is then a refused or different argue).
#[test]
fn the_second_lane_moves_no_byte() {
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
        check(lane(LaneLimits::default()));

        // Every local group on the second lane, every bin run there.
        let proved = check(second(LaneLimits::default(), LIGHT));
        let stamps = second_stamps(&proved);
        assert_eq!(stamps[0], (0, LaneState::Off, 0, 0, 0), "group 0: no job");
        let mut most = 0;
        for (g, &(card, state, bins, ran, left)) in stamps.iter().enumerate().skip(1) {
            assert_eq!((card, state), (1, LaneState::Ran), "group {g}: {stamps:?}");
            assert!(
                bins > 0 && ran == bins && left == 0,
                "group {g}: {stamps:?}"
            );
            most = most.max(bins);
        }
        let lines = local_lane_lines(&proved.3);
        assert!(
            lines.contains("· on the second card: groups 2, ran 2, bins "),
            "{lines}"
        );

        // The mixed group on card 0's lane, the light one on the second.
        let stamps = second_stamps(&check(second(LaneLimits::default(), MIXED)));
        assert_eq!(
            (stamps[1].0, stamps[1].1),
            (1, LaneState::Ran),
            "{stamps:?}"
        );
        assert_eq!(
            (stamps[2].0, stamps[2].1),
            (0, LaneState::Ran),
            "{stamps:?}"
        );

        // Stopped after each bin index: phase B runs the rest.
        for stop in 0..=most {
            let proved = check(second(
                LaneLimits {
                    bins: Some(stop),
                    ..LaneLimits::default()
                },
                LIGHT,
            ));
            for (g, &(card, state, bins, ran, left)) in
                second_stamps(&proved).iter().enumerate().skip(1)
            {
                let expected = stop.min(bins);
                assert_eq!(
                    (card, ran, left),
                    (1, expected, bins - expected),
                    "group {g}"
                );
                let want = if expected == bins {
                    LaneState::Ran
                } else {
                    LaneState::Deferred(LaneDefer::Stopped)
                };
                assert_eq!(state, want, "group {g}, stop {stop}");
            }
        }

        // No room as the job starts: phase B runs every bin.
        let proved = check(second(
            LaneLimits {
                hold_bytes: Some(0),
                ..LaneLimits::default()
            },
            LIGHT,
        ));
        for (g, &(card, state, _, ran, left)) in second_stamps(&proved).iter().enumerate().skip(1) {
            assert_eq!(
                (card, state, ran),
                (1, LaneState::Deferred(LaneDefer::Memory), 0),
                "group {g}"
            );
            assert!(left > 0, "group {g}");
        }

        // A unit failing after it ran: dropped whole, argued anew.
        for fail in [0, 1] {
            let proved = check(second(
                LaneLimits {
                    fail_bin: Some(fail),
                    ..LaneLimits::default()
                },
                LIGHT,
            ));
            for (g, &(card, state, bins, ran, left)) in
                second_stamps(&proved).iter().enumerate().skip(1)
            {
                assert_eq!(card, 1, "group {g}");
                if fail < bins {
                    assert_eq!(
                        (state, ran, left),
                        (LaneState::Failed, fail, bins),
                        "group {g}"
                    );
                } else {
                    assert_eq!(state, LaneState::Ran, "group {g}");
                }
            }
        }
    }
}

/// L3. Routing: each local group is one lane's job, decided once — a light
/// group the second lane's, a group that is not light card 0's, a group with
/// no local interaction neither's (decided at once). Each group of the block
/// is decided exactly once, so no group is handed to both lanes.
#[test]
fn each_local_group_is_one_lanes() {
    let airs = Airs::new(None);
    let block = Block::honest();
    let config = production();
    for (bounds, cards) in [(LIGHT, [1u8, 1]), (MIXED, [1, 0]), ([0; 3], [0, 0])] {
        let (committed, decided) = commit_counted(
            tables(&airs, &block),
            &config,
            second(LaneLimits::default(), bounds),
            LocalBinding::ALL,
        );
        let committed = committed.expect("phase A");
        assert_eq!(decided, SIZES.len(), "{bounds:?}: each group decided once");
        let stamps = committed.stamps();
        assert!(
            !stamps[0].zc_lane.local && stamps[0].zc_lane.state == LaneState::Off,
            "group 0: no local job"
        );
        for (g, &card) in cards.iter().enumerate() {
            let job = &stamps[g + 1].zc_lane;
            assert!(job.local, "{bounds:?}, group {}: a local job", g + 1);
            assert_eq!(
                (job.card, job.state),
                (card, LaneState::Ran),
                "{bounds:?}, group {}",
                g + 1
            );
        }
    }
}

/// L5. Phase B split over two drivers on the host: a group whose argue the
/// second lane began (run whole, or stopped after one bin) finishes on
/// either driver, and the proof is the one-driver proof, byte for byte.
#[test]
fn a_second_lanes_begun_argue_finishes_on_either_driver() {
    let airs = Airs::new(None);
    let block = Block::honest();
    let config = production();
    let order: Vec<usize> = (0..SIZES.len()).collect();
    for limits in [
        LaneLimits::default(),
        LaneLimits {
            bins: Some(1),
            ..LaneLimits::default()
        },
    ] {
        let prove_with = |split: bool| -> BlockProved<Fp, Ext> {
            let mut committed =
                commit(tables(&airs, &block), &config, second(limits, LIGHT)).expect("phase A");
            if split {
                committed.set_two_card_split(Some(vec![false; SIZES.len()]), true);
            }
            let mut transcript = H::sponge();
            multilinear_block::block_prove_in_order::<Fp, Ext, _, H>(
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
            .expect("phase B")
        };
        let one = prove_with(false);
        let two = prove_with(true);
        assert!(
            two.3
                .iter()
                .any(|s| s.card == 1 && s.zc_lane.local && s.zc_lane.card == 1),
            "{limits:?}: no second-lane group on the second driver"
        );
        assert_eq!(
            bytes(&two),
            bytes(&one),
            "{limits:?}: two drivers moved a byte"
        );
        verify_local(&airs, &block, &config, &two).expect("two drivers verify");
    }
}

/// L6. The second card asked for as production asks, without the two-card
/// split ready (no card here): no second lane, every local group on card
/// 0's lane, the lines as they were, the same bytes.
#[test]
fn no_second_lane_without_the_split() {
    let airs = Airs::new(None);
    let block = Block::honest();
    let config = production();
    let one = prove_placed(
        &airs,
        &block,
        &config,
        lane(LaneLimits::default()),
        honest(),
    )
    .expect("one lane");
    let asked =
        prove_placed(&airs, &block, &config, Placement::SecondAsked, honest()).expect("asked");
    assert_eq!(bytes(&asked), bytes(&one));
    for (g, (job, _)) in lane_stamps(&asked).iter().enumerate().skip(1) {
        assert_eq!((job.card, job.state), (0, LaneState::Ran), "group {g}");
        assert_eq!(job.upload, 0.0, "group {g}");
    }
    let lines = local_lane_lines(&asked.3);
    assert!(
        !lines.contains(" · card ") && !lines.contains("second card"),
        "{lines}"
    );
}

/// L7. The lines with a second lane: each local group's names its lane's
/// card and upload, and the sum what ran on the second card.
#[test]
fn the_second_lanes_lines_add_up() {
    let job = |card, state, bins, bins_ran, ladders, upload, ended_at| ZcLaneStamp {
        state,
        bins,
        bins_ran,
        ladders,
        local: true,
        store_bytes: 3 << 20,
        card,
        upload,
        ended_at,
        ..ZcLaneStamp::default()
    };
    let stamps = [
        GroupStamps::default(),
        GroupStamps {
            zc_lane: job(1, LaneState::Ran, 2, 2, 0.5, 0.0625, 2.0),
            ..GroupStamps::default()
        },
        GroupStamps {
            zc_lane: job(
                1,
                LaneState::Deferred(LaneDefer::Memory),
                0,
                0,
                0.0,
                0.0,
                2.5,
            ),
            local_b: vec![0.25, 0.5],
            ..GroupStamps::default()
        },
        GroupStamps {
            zc_lane: job(0, LaneState::Ran, 1, 1, 0.25, 0.0, 3.0),
            ..GroupStamps::default()
        },
    ];
    let lines = local_lane_lines(&stamps);
    assert_eq!(
        lines,
        "BLOCK LOCAL LANE: group 1 · ran · bins 2 of 2 in phase A · lane 0.500s · store 3 MiB · \
         commit wait 0.000s · phase B 0.000s [] · card 1 · upload 0.062s\n\
         BLOCK LOCAL LANE: group 2 · memory · bins 0 of 2 in phase A · lane 0.000s · store 3 MiB · \
         commit wait 0.000s · phase B 0.750s [0.2500 0.5000] · card 1 · upload 0.000s\n\
         BLOCK LOCAL LANE: group 3 · ran · bins 1 of 1 in phase A · lane 0.250s · store 3 MiB · \
         commit wait 0.000s · phase B 0.000s [] · card 0 · upload 0.000s\n\
         BLOCK LOCAL LANE SUM: groups 3 · lane jobs 3 · bins 3 of 5 in phase A · ran 2 · deferred \
         hold 0 · memory 1 · tail 0 · shed 0 · no store 0 · stopped 0 · declined 0 · failed 0 · \
         lane 0.750s · phase B 0.750s · commits waited 0.000s (most 0.000s) · last job ended at \
         3.000s · phase A's end waited 0.000s · on the second card: groups 2, ran 1, bins 2, lane \
         0.500s, uploads 0.062s, last job ended at 2.500s\n"
    );
}
