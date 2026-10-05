//! Live regeneration (R-REGEN step 3): phase A drops what the policy moves off
//! the host when it can be regenerated, and the live regenerator brings every
//! dropped trace back, in rank order, as the one dropped — or fails every
//! slot it does not bring back, on every exit, so phase B never waits forever.
//!
//! Phase B is stood in for by one taker that waits for each dropped trace in
//! rank order and takes it, as its fused task would; the laptop has no device,
//! so these runs drop traces whose fused task recommits on the host
//! ([`LiveRegen::for_test`]), which production never does (A1).

use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use executor::elf::Elf;
use stark::narrow::NarrowMain;
use stark::regen::{RegenError, RegenSlot};

use super::live::{DropJob, LiveFaults, LivePlan, LiveRegen, LiveReport, WindowGuard, run_live};
use super::{RegenMode, Table};
use crate::block::{
    LiveRun, SpillPolicy, StreamConfig, stream_config, stream_live_for_test, stream_spill_for_test,
};
use crate::tables::MaxRowsConfig;
use crate::tables::gpack::TraceForm;
use crate::tables::trace_builder::{RegenBuilder, StreamSkip, StreamTable, Traces};
use crate::tests::windowed_builder_tests::{same_traces, widen_all};

const GIB: u64 = 1 << 30;

/// How long a regenerator and its taker may take before the test calls it a
/// hang (they take well under a second here).
const DEADLINE: Duration = Duration::from_secs(60);

/// The cases, kept to a few for the laptop (R-REGEN S7): a program and phase
/// A's (committers, generators, finish in phase A) — generators ahead with
/// the finish in phase A, the committers generating with it in phase B, and
/// KECCAK chunked.
const CASES: [(&str, (usize, usize, bool)); 3] = [
    ("all_instructions_64", (2, 3, true)),
    ("all_instructions_64", (3, 0, false)),
    ("test_keccak_multi", (2, 3, true)),
];

fn max_rows() -> MaxRowsConfig {
    MaxRowsConfig {
        keccak_rnd: 48,
        ..MaxRowsConfig::small()
    }
}

fn program(name: &str) -> Elf {
    Elf::load(&crate::test_utils::asm_elf_bytes(name)).expect("the ELF loads")
}

fn list(traces: &Traces, table: StreamTable) -> &Vec<Table> {
    match table {
        StreamTable::Cpu => &traces.cpus,
        StreamTable::MemwRegister => &traces.memw_registers,
        StreamTable::MemwAligned => &traces.memw_aligneds,
        StreamTable::Memw => &traces.memws,
        StreamTable::Load => &traces.loads,
        StreamTable::Lt => &traces.lts,
        StreamTable::Shift => &traces.shifts,
        StreamTable::Store => &traces.stores,
    }
}

fn list_mut(traces: &mut Traces, table: StreamTable) -> &mut Vec<Table> {
    match table {
        StreamTable::Cpu => &mut traces.cpus,
        StreamTable::MemwRegister => &mut traces.memw_registers,
        StreamTable::MemwAligned => &mut traces.memw_aligneds,
        StreamTable::Memw => &mut traces.memws,
        StreamTable::Load => &mut traces.loads,
        StreamTable::Lt => &mut traces.lts,
        StreamTable::Shift => &mut traces.shifts,
        StreamTable::Store => &mut traces.stores,
    }
}

fn handed_out(streamed: &StreamSkip, table: StreamTable) -> usize {
    match table {
        StreamTable::Cpu => streamed.cpu,
        StreamTable::MemwRegister => streamed.memw_register,
        StreamTable::MemwAligned => streamed.memw_aligned,
        StreamTable::Memw => streamed.memw,
        StreamTable::Load => streamed.load,
        StreamTable::Lt => streamed.lt,
        StreamTable::Shift => streamed.shift,
        StreamTable::Store => streamed.store,
    }
}

/// Every streamed chunk the stream handed out, as (table, index).
fn streamed_chunks(streamed: &StreamSkip) -> Vec<(StreamTable, usize)> {
    StreamTable::ALL
        .iter()
        .flat_map(|&t| (0..handed_out(streamed, t)).map(move |i| (t, i)))
        .collect()
}

/// One of these tests (and the shadow's) at a time: each holds a few runs'
/// traces, and side by side on the laptop they reached 5.3 GiB (R-REGEN S7);
/// alone, 2.5 at most.
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(super) fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

/// Phase A under live regeneration `mode` with a window of `ahead` bytes.
fn phase_a(
    program: &Elf,
    config: StreamConfig,
    mode: RegenMode,
    ahead: u64,
    policy: Option<SpillPolicy>,
    fake_host_after: Option<u64>,
) -> LiveRun {
    stream_live_for_test(
        program,
        &crate::lfm::proof::block_base_options(),
        &max_rows(),
        config,
        LiveRegen::for_test(mode, ahead),
        policy,
        fake_host_after,
    )
    .expect("phase A")
}

/// The same run with neither regeneration nor a spill.
fn resident(program: &Elf, config: StreamConfig) -> Traces {
    stream_spill_for_test(
        program,
        &crate::lfm::proof::block_base_options(),
        &max_rows(),
        config,
        None,
    )
    .expect("phase A, resident")
    .0
}

/// The packed trace the resident run holds for `key`.
fn reference(traces: &Traces, (table, index): (StreamTable, usize)) -> &NarrowMain {
    let trace = &list(traces, table)[index];
    trace.narrow_main().unwrap_or_else(|| {
        panic!(
            "the resident run holds {table:?}[{index}] packed ({} of {}; spilled {}, {} rows)",
            index,
            list(traces, table).len(),
            trace.is_main_spilled(),
            trace.num_rows()
        )
    })
}

/// The regenerator over `plan` beside `taker` (phase B's stand-in), each on a
/// thread of its own, walking phase A's windows and writing packed (G-pack,
/// as phase A here). Returns the regenerator's
/// report and the taker's result, or `None` for either that did not end
/// within [`DEADLINE`] — the window is then closed, which must release both.
fn regenerate<R: Send>(
    program: &Elf,
    plan: LivePlan,
    generators: usize,
    faults: LiveFaults,
    taker: impl FnOnce() -> R + Send,
) -> (Option<LiveReport>, Option<R>) {
    let window = max_rows().cpu;
    regenerate_on(
        program,
        window,
        TraceForm::Narrow,
        plan,
        generators,
        faults,
        taker,
    )
}

/// [`regenerate`] on windows of `window` cycles, writing in `form`.
fn regenerate_on<R: Send>(
    program: &Elf,
    window: usize,
    form: TraceForm,
    plan: LivePlan,
    generators: usize,
    faults: LiveFaults,
    taker: impl FnOnce() -> R + Send,
) -> (Option<LiveReport>, Option<R>) {
    let slots = Arc::clone(&plan.window);
    let builder = RegenBuilder::new(program, &[], &max_rows()).expect("the regenerator");
    std::thread::scope(|s| {
        let (report_tx, report_rx) = mpsc::channel();
        let (taken_tx, taken_rx) = mpsc::channel();
        s.spawn(move || {
            let report = run_live(
                program,
                &[],
                builder,
                window,
                plan,
                generators,
                form,
                faults,
            );
            let _ = report_tx.send(report);
        });
        s.spawn(move || {
            let _ = taken_tx.send(taker());
        });
        let deadline = Instant::now() + DEADLINE;
        let taken = taken_rx
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .ok();
        let report = report_rx
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .ok();
        if taken.is_none() || report.is_none() {
            slots.close("the test's deadline");
        }
        (report, taken)
    })
}

/// Phase B's stand-in that never refuses: each dropped trace, in rank order,
/// waited for and taken back into its table on the host (a refusal panics).
fn widen_in_rank_order(traces: &mut Traces, keys: &[(StreamTable, usize)]) {
    for &(table, index) in keys {
        list_mut(traces, table)[index].widen_main_on_host();
    }
}

/// Phase B's stand-in that records refusals: each slot, in rank order, waited
/// for and taken.
fn take_in_rank_order(slots: &[RegenSlot]) -> Vec<Result<NarrowMain, RegenError>> {
    slots
        .iter()
        .map(|slot| {
            slot.wait();
            slot.take_for_test()
        })
        .collect()
}

fn keys_of(plan: &LivePlan) -> Vec<(StreamTable, usize)> {
    plan.dropped.iter().map(|d| d.key).collect()
}

fn slots_of(plan: &LivePlan) -> Vec<RegenSlot> {
    plan.dropped.iter().map(|d| d.slot.clone()).collect()
}

/// ★ `always`: phase A drops every streamed chunk (each a slot holding the
/// resident run's digest, no words), the plan lists them in rank order, and
/// the regenerator brings every one back while phase B takes them in rank
/// order — with the window at 4 GiB, and at one byte (only the frontier may
/// deposit, the reviewer's hang case), written packed (G-pack) or at 8 bytes
/// a cell then packed — so the traces are the resident run's, table for
/// table.
#[test]
fn always_drops_every_streamed_chunk_and_the_regenerator_brings_each_back() {
    let _one = one_at_a_time();
    let arms = [
        (4 * GIB, 3, TraceForm::Narrow),
        (1, 1, TraceForm::Wide),
        (1, 3, TraceForm::Narrow),
    ];
    for ((name, (committers, generators, finish_in_a)), arm) in CASES.into_iter().zip(arms) {
        let program = program(name);
        let config = stream_config(committers, generators, finish_in_a);
        {
            let (ahead, generators, form) = arm;
            {
                let mut reference_traces = resident(&program, config);
                let mut run = phase_a(&program, config, RegenMode::Always, ahead, None, None);
                let chunks = streamed_chunks(&run.streamed);
                assert!(!chunks.is_empty(), "{name}: nothing streamed");
                for &(table, index) in &chunks {
                    let trace = &list(&run.traces, table)[index];
                    let slot = trace
                        .regen_main()
                        .unwrap_or_else(|| panic!("{name}: {table:?}[{index}] was not dropped"));
                    assert!(!trace.is_main_narrow() && !trace.is_main_spilled());
                    assert_eq!(
                        slot.digest(),
                        reference(&reference_traces, (table, index)).digest(),
                        "{name}: {table:?}[{index}]"
                    );
                }
                let plan = run.plan.take().expect("something was dropped");
                assert_eq!(plan.dropped.len(), chunks.len(), "{name}: {}", run.line);
                assert!(
                    plan.dropped
                        .windows(2)
                        .all(|w| w[0].slot.order_key() < w[1].slot.order_key()),
                    "{name}: the plan is in rank order"
                );
                assert!(plan.dropped.iter().all(|d| !d.back));
                assert!(
                    run.line.starts_with(&format!(
                        "BLOCK REGEN dropped: Always · {} instances",
                        chunks.len()
                    )),
                    "{}",
                    run.line
                );
                let keys = keys_of(&plan);
                let n = keys.len();
                let traces = &mut run.traces;
                let (report, taken) = regenerate_on(
                    &program,
                    max_rows().cpu,
                    form,
                    plan,
                    generators,
                    LiveFaults::default(),
                    move || widen_in_rank_order(traces, &keys),
                );
                let report = report.expect("the regenerator ended");
                assert!(taken.is_some(), "{name}: phase B's taker hung");
                assert_eq!(report.deposited, n, "{name}: {}", report.line());
                assert_eq!(report.mismatches, 0, "{}", report.line());
                assert_eq!(report.skipped, 0, "{}", report.line());
                assert!(report.failures.is_empty(), "{}", report.line());
                assert!(report.error.is_none(), "{}", report.line());
                widen_all(&mut run.traces);
                widen_all(&mut reference_traces);
                same_traces(&reference_traces, &run.traces);
            }
        }
    }
}

/// ★ `auto` that never arms drops nothing, plans no regenerator (R8), and
/// builds the resident run's traces: a budget the block never reaches, and no
/// spill policy at all.
#[test]
fn auto_that_never_arms_drops_nothing() {
    let _one = one_at_a_time();
    let program = program("all_instructions_64");
    let config = stream_config(2, 3, true);
    let mut reference_traces = resident(&program, config);
    widen_all(&mut reference_traces);
    for policy in [Some(SpillPolicy::Budget(u64::MAX / 2)), None] {
        let mut run = phase_a(&program, config, RegenMode::Auto, 4 * GIB, policy, None);
        assert!(run.plan.is_none(), "{policy:?}: {}", run.line);
        assert!(run.armed.is_none(), "{policy:?}: {}", run.line);
        assert!(
            run.line.contains(" 0 instances 0.00 GiB") && run.line.contains("never armed"),
            "{}",
            run.line
        );
        if policy.is_some() {
            assert!(run.spill.contains("spilled 0 instances"), "{}", run.spill);
        }
        widen_all(&mut run.traces);
        same_traces(&reference_traces, &run.traces);
    }
}

/// ★ `auto` under pressure (the host read as the target once half the
/// streamed bytes were considered): regeneration arms once; the streamed
/// chunks kept before it are dropped back (at this scale the need, at least
/// the 2 GiB margin, exceeds them all) and every later one is dropped, never
/// spilled; only what cannot be regenerated spills; the kept bytes left on the
/// host are counted; and the regenerator brings every dropped trace back, so
/// the traces (the spilled ones read back) are the resident run's.
#[test]
fn auto_drops_the_regenerable_and_spills_only_the_rest() {
    let _one = one_at_a_time();
    for (name, (committers, generators, finish_in_a)) in CASES {
        let program = program(name);
        {
            let config = stream_config(committers, generators, finish_in_a);
            let mut reference_traces = resident(&program, config);
            let streamed_bytes: u64 = {
                let run = phase_a(&program, config, RegenMode::Always, 4 * GIB, None, None);
                streamed_chunks(&run.streamed)
                    .iter()
                    .map(|&k| reference(&reference_traces, k).data().len() as u64)
                    .sum()
            };
            let mut run = phase_a(
                &program,
                config,
                RegenMode::Auto,
                4 * GIB,
                Some(SpillPolicy::Auto),
                Some(streamed_bytes / 2),
            );
            let (need, planned) = run.armed.expect("regeneration armed");
            assert!(need >= 2 * GIB, "{name}: need {need}");
            let chunks = streamed_chunks(&run.streamed);
            for &(table, index) in &chunks {
                let trace = &list(&run.traces, table)[index];
                assert!(
                    trace.is_main_regenerable() && !trace.is_main_spilled(),
                    "{name}: {table:?}[{index}] not dropped: {}",
                    run.line
                );
            }
            let plan = run.plan.take().expect("something was dropped");
            assert_eq!(plan.dropped.len(), chunks.len(), "{}", run.line);
            let back: Vec<_> = plan.dropped.iter().filter(|d| d.back).collect();
            assert!(
                !back.is_empty() && back.len() < chunks.len(),
                "{name}: drop-back {} of {}: {}",
                back.len(),
                chunks.len(),
                run.line
            );
            assert_eq!(
                planned,
                back.iter().map(|d| d.bytes).sum::<u64>(),
                "{name}: drop-back dropped what it was given"
            );
            if !finish_in_a {
                assert_eq!(run.resident, 0, "{name}: kept bytes left: {}", run.spill);
            } else {
                let narrow: u64 = all_tables(&run.traces)
                    .filter_map(|t| t.narrow_main())
                    .map(|n| n.data().len() as u64)
                    .sum();
                assert!(
                    run.resident <= narrow,
                    "{name}: {} > {narrow}",
                    run.resident
                );
            }
            let keys = keys_of(&plan);
            let traces = &mut run.traces;
            let (report, taken) = regenerate(&program, plan, 2, LiveFaults::default(), move || {
                widen_in_rank_order(traces, &keys)
            });
            let report = report.expect("the regenerator ended");
            assert!(taken.is_some(), "{name}: phase B's taker hung");
            assert_eq!(report.deposited, chunks.len(), "{}", report.line());
            widen_all(&mut run.traces);
            widen_all(&mut reference_traces);
            same_traces(&reference_traces, &run.traces);
        }
    }
}

fn all_tables(t: &Traces) -> impl Iterator<Item = &Table> {
    [
        &t.cpus,
        &t.memw_registers,
        &t.memw_aligneds,
        &t.memws,
        &t.loads,
        &t.stores,
        &t.shifts,
        &t.cpu32s,
        &t.commits,
        &t.keccaks,
        &t.keccak_rnds,
        &t.ecsms,
        &t.ecdases,
        &t.hints,
        &t.pages,
        &t.lts,
        &t.muls,
        &t.dvrms,
        &t.branches,
        &t.eqs,
        &t.bytewises,
    ]
    .into_iter()
    .flatten()
}

/// ★ R6: drop-back stops on the bytes it is given, counted once at arming,
/// never on a host reading: the resident candidates go in rank order until
/// their bytes reach the need. It arms once. A candidate kept by a decision
/// taken before another committer armed is dropped back while drop-back is
/// short of its need, and kept once it is not; a need of 0 drops nothing back.
#[test]
fn drop_back_stops_on_the_bytes_it_was_given() {
    let key = |i: usize| (StreamTable::Cpu, i);
    let sent = |rx: &mpsc::Receiver<DropJob>| -> Vec<(u64, usize, bool)> {
        rx.try_iter().map(|j| (j.rank, j.index, j.back)).collect()
    };
    // (rank, bytes) of five resident candidates, kept out of rank order.
    let candidates = [(5u64, 10u64), (1, 20), (4, 30), (2, 40), (3, 50)];
    let armed = |need: u64| {
        let live = LiveRegen::for_test(RegenMode::Auto, 4 * GIB);
        let (tx, rx) = mpsc::channel();
        live.set_drop_tx(tx);
        for (i, &(rank, bytes)) in candidates.iter().enumerate() {
            live.kept(i, key(i), rank, bytes);
        }
        assert!(sent(&rx).is_empty(), "nothing drops before arming");
        assert!(live.arm(need));
        (live, rx)
    };

    let (live, rx) = armed(55);
    assert_eq!(sent(&rx), [(1, 1, true), (2, 3, true)]);
    assert_eq!(live.armed_for_test(), Some((55, 60)));
    assert!(!live.arm(1000), "it arms once");
    assert!(sent(&rx).is_empty());
    live.kept(9, key(9), 9, 7);
    assert!(sent(&rx).is_empty(), "the need is met");

    let (live, rx) = armed(1000);
    assert_eq!(
        sent(&rx),
        [
            (1, 1, true),
            (2, 3, true),
            (3, 4, true),
            (4, 2, true),
            (5, 0, true)
        ]
    );
    live.kept(7, key(7), 0, 8);
    assert_eq!(sent(&rx), [(0, 7, true)], "a late candidate while short");
    assert_eq!(live.armed_for_test(), Some((1000, 158)));
    live.close_drops();
    live.kept(8, key(8), 6, 8);
    assert!(sent(&rx).is_empty(), "no queue, no drop");
    assert_eq!(
        live.armed_for_test(),
        Some((1000, 158)),
        "and nothing counted"
    );

    let (live, rx) = armed(0);
    assert!(sent(&rx).is_empty());
    assert_eq!(live.armed_for_test(), Some((0, 0)));
    assert!(
        live.into_plan().is_none(),
        "nothing dropped, no regenerator"
    );
}

/// ★ No disk (I-REGEN §14 P1): with the spill policy `off`, `auto` still decides
/// with no store. Unarmed it drops nothing and builds the resident run's
/// traces. Under pressure (the host read as the target once half the streamed
/// bytes were considered) it arms once, drops back **every** streamed chunk
/// kept before arming and drops every later one, spills nothing (no store),
/// keeps what cannot be regenerated on the host (counted resident), and the
/// regenerator brings every dropped trace back.
#[test]
fn no_disk_auto_drops_every_regenerable_and_spills_nothing() {
    let _one = one_at_a_time();
    for (name, (committers, generators, finish_in_a)) in &CASES[..2] {
        let (name, finish_in_a) = (*name, *finish_in_a);
        let program = program(name);
        let config = stream_config(*committers, *generators, finish_in_a);
        let mut reference_traces = resident(&program, config);
        let mut unarmed = phase_a(
            &program,
            config,
            RegenMode::Auto,
            4 * GIB,
            Some(SpillPolicy::Off),
            None,
        );
        assert!(unarmed.plan.is_none(), "{name}: {}", unarmed.line);
        assert!(
            unarmed.line.contains("Auto (no disk) · 0 instances")
                && unarmed.line.contains("never armed"),
            "{}",
            unarmed.line
        );
        assert!(unarmed.spill.contains("no store"), "{}", unarmed.spill);
        widen_all(&mut unarmed.traces);
        let streamed_bytes: u64 = streamed_chunks(&unarmed.streamed)
            .iter()
            .map(|&k| reference(&reference_traces, k).data().len() as u64)
            .sum();
        let mut run = phase_a(
            &program,
            config,
            RegenMode::Auto,
            4 * GIB,
            Some(SpillPolicy::Off),
            Some(streamed_bytes / 2),
        );
        assert!(run.armed.is_some(), "{name}: {}", run.line);
        assert!(run.line.contains("Auto (no disk) · "), "{}", run.line);
        assert!(
            run.spill.contains("no store") && run.spill.contains("spilled 0 instances"),
            "{name}: {}",
            run.spill
        );
        let chunks = streamed_chunks(&run.streamed);
        for &(table, index) in &chunks {
            let trace = &list(&run.traces, table)[index];
            assert!(
                trace.is_main_regenerable(),
                "{name}: {table:?}[{index}] kept: {}",
                run.line
            );
        }
        assert!(
            all_tables(&run.traces).all(|t| !t.is_main_spilled()),
            "{name}: something spilled"
        );
        let plan = run.plan.take().expect("something was dropped");
        assert_eq!(plan.dropped.len(), chunks.len(), "{}", run.line);
        let back: u64 = plan
            .dropped
            .iter()
            .filter(|d| d.back)
            .map(|d| d.bytes)
            .sum();
        assert!(back > 0, "{name}: no drop-back: {}", run.line);
        assert_eq!(
            run.armed.map(|(_, given)| given),
            Some(back),
            "{name}: drop-back took it all"
        );
        let narrow: u64 = all_tables(&run.traces)
            .filter_map(|t| t.narrow_main())
            .map(|n| n.data().len() as u64)
            .sum();
        if finish_in_a {
            assert!(
                run.resident > 0 && run.resident <= narrow,
                "{name}: {} of {narrow}",
                run.resident
            );
        } else {
            assert_eq!(run.resident, 0, "{name}: {}", run.spill);
        }
        let keys = keys_of(&plan);
        let traces = &mut run.traces;
        let (report, taken) = regenerate(&program, plan, 2, LiveFaults::default(), move || {
            widen_in_rank_order(traces, &keys)
        });
        let report = report.expect("the regenerator ended");
        assert!(taken.is_some(), "{name}: phase B's taker hung");
        assert_eq!(report.deposited, chunks.len(), "{}", report.line());
        widen_all(&mut run.traces);
        widen_all(&mut reference_traces);
        same_traces(&reference_traces, &run.traces);
        same_traces(&reference_traces, &unarmed.traces);
    }
}

/// No disk's drop-back takes every candidate whatever the need, and every
/// candidate kept after arming is dropped back too.
#[test]
fn no_disk_drop_back_takes_every_candidate() {
    let live = LiveRegen::for_test(RegenMode::Auto, 4 * GIB);
    live.set_no_disk();
    let (tx, rx) = mpsc::channel();
    live.set_drop_tx(tx);
    for (i, &(rank, bytes)) in [(3u64, 10u64), (1, 20), (2, 30)].iter().enumerate() {
        live.kept(i, (StreamTable::Cpu, i), rank, bytes);
    }
    assert!(live.arm(1));
    let sent: Vec<u64> = rx.try_iter().map(|j| j.rank).collect();
    assert_eq!(sent, [1, 2, 3]);
    assert_eq!(live.armed_for_test(), Some((1, 60)));
    live.kept(9, (StreamTable::Cpu, 9), 9, 5);
    let late: Vec<(u64, bool)> = rx.try_iter().map(|j| (j.rank, j.back)).collect();
    assert_eq!(late, [(9, true)]);
    assert_eq!(live.armed_for_test(), Some((1, 65)));
    assert!(live.line().contains("Auto (no disk)"), "{}", live.line());
}

/// Phase A of `all_instructions_64` under `always` for the fault tests, its
/// window one byte (only the frontier deposits): the program and the plan.
fn dropped_run() -> (Elf, LivePlan) {
    let program = program("all_instructions_64");
    let mut run = phase_a(&program, fault_config(), RegenMode::Always, 1, None, None);
    let plan = run.plan.take().expect("something was dropped");
    (program, plan)
}

fn fault_config() -> StreamConfig {
    stream_config(2, 3, true)
}

/// The resident traces of [`dropped_run`]'s run.
fn dropped_reference() -> Traces {
    resident(&program("all_instructions_64"), fault_config())
}

/// Every slot of `taken` refused but those in `ok`, which are the resident
/// run's traces, and every slot of the plan settled.
fn check_taken(
    what: &str,
    slots: &[RegenSlot],
    keys: &[(StreamTable, usize)],
    taken: &[Result<NarrowMain, RegenError>],
    reference_traces: &Traces,
    ok: impl Fn(usize) -> bool,
) {
    assert_eq!(taken.len(), slots.len(), "{what}");
    for (i, out) in taken.iter().enumerate() {
        match out {
            Ok(narrow) => {
                assert!(ok(i), "{what}: slot {i} came back");
                assert_eq!(
                    narrow.data(),
                    reference(reference_traces, keys[i]).data(),
                    "{what}: slot {i}"
                );
            }
            Err(e) => assert!(!ok(i), "{what}: slot {i} refused: {e}"),
        }
    }
    assert!(
        slots.iter().all(RegenSlot::is_ready),
        "{what}: a slot unsettled"
    );
}

/// ★ R10 on the real regenerator: a dropped chunk the slicer does not cut is
/// failed before any higher rank is handed out — the frontier, a middle rank
/// — so a single in-order taker with only the frontier admitted never waits
/// on it; a skipped last rank fails when the run ends. Every other trace
/// comes back.
#[test]
fn a_chunk_the_slicer_skips_is_failed_before_a_higher_rank() {
    let _one = one_at_a_time();
    let reference_traces = dropped_reference();
    for case in 0..4 {
        let (program, plan) = dropped_run();
        let n = plan.dropped.len();
        assert!(n >= 3, "too few dropped: {n}");
        let (skip, generators) = [(0, 1), (0, 3), (n / 2, 1), (n - 1, 2)][case];
        let (slots, keys) = (slots_of(&plan), keys_of(&plan));
        let faults = LiveFaults {
            skip_rank: Some(plan.dropped[skip].slot.rank()),
            ..LiveFaults::default()
        };
        let taker_slots = slots.clone();
        let (report, taken) = regenerate(&program, plan, generators, faults, move || {
            take_in_rank_order(&taker_slots)
        });
        let what = format!("skip {skip} of {n}, {generators} generators");
        let report = report.unwrap_or_else(|| panic!("{what}: the regenerator hung"));
        let taken = taken.unwrap_or_else(|| panic!("{what}: phase B's taker hung"));
        check_taken(&what, &slots, &keys, &taken, &reference_traces, |i| {
            i != skip
        });
        let why = match &taken[skip] {
            Err(RegenError::Failed(why)) => why.clone(),
            other => panic!("{what}: {:?}", other.as_ref().map(|_| ())),
        };
        if skip == n - 1 {
            assert!(why.contains("the run ended"), "{what}: {why}");
        } else {
            assert!(why.contains("before a higher rank"), "{what}: {why}");
            assert!(report.error.is_none(), "{what}: {}", report.line());
        }
        assert_eq!(
            (report.skipped, report.deposited),
            (1, n - 1),
            "{what}: {}",
            report.line()
        );
    }
}

/// ★ R2/R3/R10: every way the regenerator stops leaves no slot waiting and no
/// taker hanging — the walk failing at its first window or midway, the
/// executor ending early, the only generator dying outside its catch (and one
/// of two), no generator at all. What came back is the dropped trace; the
/// rest is refused.
#[test]
fn every_exit_of_the_regenerator_settles_every_slot() {
    let _one = one_at_a_time();
    let reference_traces = dropped_reference();
    let (program, plan) = dropped_run();
    let n = plan.dropped.len();
    let (slots, keys) = (slots_of(&plan), keys_of(&plan));
    let taker_slots = slots.clone();
    let (report, taken) = regenerate(&program, plan, 2, LiveFaults::default(), move || {
        take_in_rank_order(&taker_slots)
    });
    let clean = report.expect("the clean run ended");
    let taken = taken.expect("taken");
    check_taken("clean", &slots, &keys, &taken, &reference_traces, |_| true);
    let windows = clean.windows;
    assert!(windows >= 2, "too few windows: {}", clean.line());

    type Case = (&'static str, fn(&mut LiveFaults, usize, u64), usize);
    let cases: [Case; 6] = [
        (
            "the walk fails at once",
            |f, _, _| f.walk_error_at = Some(0),
            2,
        ),
        (
            "the walk fails midway",
            |f, w, _| f.walk_error_at = Some(w / 2),
            2,
        ),
        (
            "the executor ends early",
            |f, w, _| f.exec_stops_after = Some(w / 2),
            2,
        ),
        (
            "the only generator dies",
            |f, _, r| f.generator_panics_at_rank = Some(r),
            1,
        ),
        (
            "one of two generators dies",
            |f, _, r| f.generator_panics_at_rank = Some(r),
            2,
        ),
        ("no generator starts", |f, _, _| f.no_generators = true, 2),
    ];
    for (what, set, generators) in cases {
        let (program, plan) = dropped_run();
        let (slots, keys) = (slots_of(&plan), keys_of(&plan));
        let mut faults = LiveFaults::default();
        set(&mut faults, windows, plan.dropped[1].slot.rank());
        let taker_slots = slots.clone();
        let (report, taken) = regenerate(&program, plan, generators, faults, move || {
            take_in_rank_order(&taker_slots)
        });
        let report = report.unwrap_or_else(|| panic!("{what}: the regenerator hung"));
        let taken = taken.unwrap_or_else(|| panic!("{what}: phase B's taker hung"));
        let came_back: Vec<bool> = taken.iter().map(Result::is_ok).collect();
        check_taken(what, &slots, &keys, &taken, &reference_traces, |i| {
            came_back[i]
        });
        let refused = came_back.iter().filter(|&&ok| !ok).count();
        assert!(refused > 0, "{what}: nothing refused: {}", report.line());
        assert_eq!(report.deposited, n - refused, "{what}: {}", report.line());
        match what {
            "one of two generators dies" => {
                assert_eq!(refused, 1, "{what}: {}", report.line());
                assert!(taken[1].is_err(), "{what}");
                assert!(
                    report
                        .failures
                        .iter()
                        .any(|f| f.contains("a generator panicked"))
                );
            }
            "the walk fails at once" | "no generator starts" => {
                assert_eq!(refused, n, "{what}: {}", report.line());
                assert!(report.error.is_some(), "{what}: {}", report.line());
            }
            _ => assert!(report.error.is_some(), "{what}: {}", report.line()),
        }
    }
}

/// ★ R4: the prove ending closes the window (the block's guard, dropped on
/// every exit, an unwind included), which releases a regenerator waiting to
/// deposit for takers that are gone: it returns, with no error, and every
/// slot is settled.
#[test]
fn the_proves_end_releases_a_regenerator_waiting_on_its_takers() {
    let _one = one_at_a_time();
    let (program, plan) = dropped_run();
    let slots = slots_of(&plan);
    let window = Arc::clone(&plan.window);
    let first = slots[0].clone();
    let (report, _) = regenerate(&program, plan, 2, LiveFaults::default(), move || {
        // Phase B takes nothing: once the frontier is in, the window (one
        // byte) admits no other deposit, and the prove "ends".
        first.wait();
        let ended = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = WindowGuard::new(&window);
            panic!("the prove unwinds (test)");
        }));
        assert!(ended.is_err());
    });
    let report = report.expect("the regenerator was released");
    assert!(report.error.is_none(), "{}", report.line());
    assert!(report.deposited >= 1, "{}", report.line());
    assert!(report.deposited < slots.len(), "{}", report.line());
    assert!(slots.iter().all(RegenSlot::is_ready));
    assert!(slots[1].take_for_test().is_err());
}

/// ★ R4: an AIR set and traces that disagree on a table's count are refused
/// with an error on the block path, never asserted.
#[test]
fn an_air_set_that_disagrees_with_the_traces_is_an_error() {
    let _one = one_at_a_time();
    let program = program("all_instructions_64");
    let mut traces = resident(&program, stream_config(2, 3, true));
    let airs = crate::VmAirs::new(
        &program,
        &crate::lfm::proof::block_base_options(),
        false,
        &traces.page_configs,
        &traces.table_counts(),
        None,
        true,
        None,
        None,
        None,
    );
    assert!(airs.try_air_trace_pairs(&mut traces).is_ok());
    traces.cpus.pop().expect("a CPU table");
    match airs.try_air_trace_pairs(&mut traces) {
        Err(crate::Error::Prover(why)) => assert!(why.contains("disagree"), "{why}"),
        other => panic!("not refused: {:?}", other.map(|p| p.len())),
    }
}

/// ★ Invariant (I-REGEN §10.2): the ranks are phase A's hand-out order, window
/// by window, so the regenerator walks phase A's windows. A plan whose phase A
/// walked other windows (1, 7 and 33 cycles against phase A's 32) is refused
/// whole — every slot failed at once, nothing run, never a wait (R-REGEN S3).
/// Past that check (the plan made to claim the regenerator's windows), R10
/// still refuses, never waits: cycle by cycle some chunk is cut after a
/// higher-ranked one.
#[test]
fn a_regenerator_on_other_windows_refuses_never_waits() {
    let _one = one_at_a_time();
    let reference_traces = dropped_reference();
    for (window, claimed) in [(1, false), (7, false), (33, false), (1, true), (7, true)] {
        let (program, mut plan) = dropped_run();
        assert_eq!(
            plan.phase_a_window,
            Some(max_rows().cpu),
            "phase A's windows"
        );
        if claimed {
            plan.phase_a_window = Some(window);
        }
        let n = plan.dropped.len();
        let (slots, keys) = (slots_of(&plan), keys_of(&plan));
        let taker_slots = slots.clone();
        let (report, taken) = regenerate_on(
            &program,
            window,
            TraceForm::Narrow,
            plan,
            2,
            LiveFaults::default(),
            move || take_in_rank_order(&taker_slots),
        );
        let what = format!("windows of {window}, claimed {claimed}");
        let report = report.unwrap_or_else(|| panic!("{what}: the regenerator hung"));
        let taken = taken.unwrap_or_else(|| panic!("{what}: phase B's taker hung"));
        let came_back: Vec<bool> = taken.iter().map(Result::is_ok).collect();
        check_taken(&what, &slots, &keys, &taken, &reference_traces, |i| {
            came_back[i]
        });
        let refused = came_back.iter().filter(|&&ok| !ok).count();
        assert_eq!(report.deposited + refused, n, "{what}: {}", report.line());
        let mark = if claimed {
            "before a higher rank"
        } else {
            "phase A walked 32"
        };
        for e in taken.iter().filter_map(|t| t.as_ref().err()) {
            assert!(e.to_string().contains(mark), "{what}: {e}");
        }
        if claimed {
            assert!(refused > 0, "{what}: {}", report.line());
        } else {
            assert_eq!(
                (refused, report.windows),
                (n, 0),
                "{what}: {}",
                report.line()
            );
            assert!(report.error.as_deref().is_some_and(|e| e.contains(mark)));
        }
    }
}
