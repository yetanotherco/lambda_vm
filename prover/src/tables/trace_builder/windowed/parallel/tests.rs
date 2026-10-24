//! A run walked window by window from the lean pass's entries and the
//! executor's snapshots, on one thread, against the whole-run build; and each
//! check refusing what it guards.

use std::sync::Arc;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::logs::Log;

use super::super::lean_pass::LeanPass;
use super::super::{ChunkJob, WalkedWindow, WindowedTraceBuilder};
use super::{WalkSettings, WindowJob, WindowWalker, walk_numbered, walk_windows};
use crate::Error;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;
use crate::tests::windowed_builder_tests::{run, same_traces, whole};

/// A mistake put into the windows on purpose, to show a check catches it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Tamper {
    None,
    /// Window `window` without its entry number `at`.
    DropEntry {
        window: u32,
        at: usize,
    },
    /// Window `window`'s entry number `at` one timestamp later.
    LaterEntry {
        window: u32,
        at: usize,
    },
    /// Window `window` with the next window's snapshot.
    LateStart(u32),
    /// Window `window` walked from one cycle later.
    LaterFirst(u32),
    /// Window `window` starting with x5 one larger.
    Register(u32),
}

/// The run of `program` built by windows of `window` cycles walked from their
/// entries (the last window, the finish's, walked on the state built from the
/// pass), with its streamed chunks put back. With `dropped`, the builder drops
/// the ops it hands out, as the block's prove does.
fn walked_from_entries(
    program: &Elf,
    max_rows: &MaxRowsConfig,
    window: usize,
    dropped: bool,
    tamper: Tamper,
) -> Result<Traces, Error> {
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)?;
    if dropped {
        builder = builder.drop_streamed_ops()?;
    }
    let settings = Arc::new(WalkSettings {
        decode: Arc::clone(&builder.decode),
        lookups: !builder.lean.lookups,
        batch: builder.walk_batch,
    });
    let mut pass = LeanPass::new(Arc::clone(&builder.decode), &builder.register_state)?;
    let mut executor =
        Executor::new(program, Vec::new()).map_err(|e| Error::Execution(format!("{e}")))?;
    let mut start = Arc::new(executor.memory().snapshot());
    let mut windows = Vec::new();
    while let Some(logs) = executor
        .resume_with_limit(window)
        .map_err(|e| Error::Execution(format!("{e}")))?
    {
        let logs = Arc::new(logs.to_vec());
        let next = Arc::new(executor.memory().snapshot());
        windows.push((logs, std::mem::replace(&mut start, next)));
    }
    let (last_logs, last_start) = windows.pop().expect("a run has a window");
    let (mut jobs, mut first) = (Vec::new(), 0usize);
    for (k, (logs, start)) in windows.into_iter().enumerate() {
        let k = u32::try_from(k).expect("few windows");
        let registers = pass.registers();
        pass.begin(k)?;
        pass.pass(&logs, first)?;
        let cycles = logs.len();
        jobs.push(WindowJob {
            window: k,
            first,
            logs,
            start,
            registers,
            entries: Arc::new(pass.take_entries()),
        });
        first += cycles;
    }
    let state = pass.walked_state(&last_start, &builder.image, builder.memory_state.lean);
    let registers = pass.registers();
    tamper_with(&mut jobs, &last_start, tamper);
    let mut walker = WindowWalker::new(settings);
    let mut chunks = Vec::new();
    {
        let (_, mut accumulator) = builder.split();
        for k in 0..jobs.len() {
            let walked = walker.walk(&jobs[k])?;
            match jobs.get(k + 1) {
                Some(next) => {
                    walker.check_exit(&jobs[k], &next.registers, &next.entries, &next.start)?
                }
                None => walker.check_exit_into(&jobs[k], &registers, &state)?,
            }
            chunks.extend(
                accumulator
                    .absorb(walked)
                    .into_iter()
                    .map(ChunkJob::generate),
            );
        }
    }
    builder.install_walked(state, registers, first, jobs.len(), 0.0);
    let mut traces = builder.finish(&last_logs)?;
    traces.insert_streamed(chunks)?;
    Ok(traces)
}

fn tamper_with(
    jobs: &mut [WindowJob],
    last_start: &Arc<executor::vm::memory::MemorySnapshot>,
    tamper: Tamper,
) {
    let at = |w: u32| w as usize;
    match tamper {
        Tamper::None => {}
        Tamper::DropEntry { window, at: i } => {
            Arc::make_mut(&mut jobs[at(window)].entries).remove(i);
        }
        Tamper::LaterEntry { window, at: i } => {
            Arc::make_mut(&mut jobs[at(window)].entries)[i].1 += 1
        }
        Tamper::LateStart(window) => {
            let next = jobs
                .get(at(window) + 1)
                .map_or_else(|| Arc::clone(last_start), |j| Arc::clone(&j.start));
            jobs[at(window)].start = next;
        }
        Tamper::LaterFirst(window) => jobs[at(window)].first += 1,
        Tamper::Register(window) => jobs[at(window)].registers.regs[5].0 += 1,
    }
}

const PROGRAMS: [&str; 12] = [
    "sub",
    "all_instructions_64",
    "lw_sw_offset_odd",
    "test_memw_split_ts",
    "test_keccak",
    "test_keccak_multi",
    "test_blake3",
    "test_blake3_absorb",
    "test_commit_split",
    "test_ecsm",
    "test_copy_ecalls",
    "test_copy_overlaps",
];

/// ★ Walked from their entries, at every window length (one cycle, less than a
/// chunk, exactly a chunk, not a divisor of it, longer than the run), with the
/// handed-out ops kept or dropped, the windows build the whole-run tables.
#[test]
fn windows_walked_from_their_entries_build_the_whole_run_tables() {
    let max_rows = MaxRowsConfig::small();
    for name in PROGRAMS {
        let (program, logs) = run(name);
        let reference = whole(&program, &logs, &max_rows);
        for window in [1, 7, 32, 33, 1000, logs.len()] {
            for dropped in [false, true] {
                let traces =
                    walked_from_entries(&program, &max_rows, window.max(1), dropped, Tamper::None)
                        .unwrap_or_else(|e| panic!("{name}, window {window}: {e:?}"));
                same_traces(&reference, &traces);
            }
        }
    }
}

/// The memory bytes whose old timestamp the window's MEMW and MEMW_A rows keep
/// (within each access's width), sorted: where a moved entry timestamp reaches
/// the tables.
fn kept_timestamps(window: &WalkedWindow) -> Vec<u64> {
    let mut out = Vec::new();
    for op in window.walk.memw.general.iter().filter(|op| !op.is_register) {
        out.extend((0..u64::from(op.width)).map(|i| op.base_address.wrapping_add(i)));
    }
    for row in window
        .walk
        .memw
        .aligned
        .iter()
        .filter(|row| !row.is_register())
    {
        out.extend((0..row.width() as u64).map(|i| row.base_address().wrapping_add(i)));
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// A walked window's entries, whether it wrote a byte it touches, and the
/// bytes whose old timestamp its memory rows keep ([`entries_of`]).
type Picks = (Vec<(u64, u64)>, bool, Vec<u64>);

/// What a tamper test picks from, per walked window: its entries; whether the
/// window wrote a byte it touches (so the next window's snapshot differs from
/// its own there); and the bytes whose old timestamp its memory rows keep.
fn entries_of(program: &Elf, window: usize) -> Vec<Picks> {
    let mut builder =
        WindowedTraceBuilder::new(program, &[], &MaxRowsConfig::small()).expect("builder");
    let mut pass =
        LeanPass::new(Arc::clone(&builder.decode), &builder.register_state).expect("pass");
    let mut executor = Executor::new(program, Vec::new()).expect("the executor");
    let mut starts = vec![executor.memory().snapshot()];
    let mut windows = Vec::new();
    while let Some(logs) = executor
        .resume_with_limit(window)
        .expect("the run")
        .map(<[Log]>::to_vec)
    {
        windows.push(logs);
        starts.push(executor.memory().snapshot());
    }
    windows.pop();
    let mut first = 0;
    let (mut walker, _) = builder.split();
    windows
        .iter()
        .enumerate()
        .map(|(k, logs)| {
            pass.begin(k as u32).expect("a window");
            pass.pass(logs, first).expect("the pass");
            first += logs.len();
            let entries = pass.take_entries();
            let wrote = entries
                .iter()
                .any(|&(a, _)| starts[k].load_byte(a) != starts[k + 1].load_byte(a));
            let kept = kept_timestamps(&walker.walk(logs).expect("the walk"));
            (entries, wrote, kept)
        })
        .collect()
}

/// Built with `tamper`: refused (an error or a panic in the walk), or a build
/// whose tables differ from the whole run's.
fn caught(program: &Elf, reference: &Traces, window: usize, tamper: Tamper) -> &'static str {
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        walked_from_entries(program, &MaxRowsConfig::small(), window, true, tamper)
    }));
    match built {
        Err(_) => "stopped",
        Ok(Err(_)) => "refused",
        Ok(Ok(traces)) => {
            let same = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                same_traces(reference, &traces)
            }));
            if same.is_err() { "differs" } else { "unseen" }
        }
    }
}

/// ★ Each check catches what it guards. Over several programs and windows:
/// a window missing any one of its entries is refused, with no panic first
/// (the walk reads a byte it does not hold, and ECSM computes no witness from
/// such a byte); a window walked from a cycle off is refused (its
/// registers end off); and an entry's timestamp moved (where a memory row
/// keeps it), the next window's snapshot (where the window wrote a byte it
/// touches), or a register off never builds the whole-run tables.
#[test]
fn each_check_catches_what_it_guards() {
    let window = 7;
    let mut seen = std::collections::BTreeMap::<(&str, &str), usize>::new();
    for name in [
        "all_instructions_64",
        "test_keccak_multi",
        "test_ecsm",
        "test_commit_split",
    ] {
        let (program, logs) = run(name);
        let reference = whole(&program, &logs, &MaxRowsConfig::small());
        let entries = entries_of(&program, window);
        let windows = entries.len() as u32;
        assert!(windows > 2, "{name}: {windows} windows");
        for w in [0, windows / 2, windows - 1] {
            let (entries, wrote, kept) = &entries[w as usize];
            let n = entries.len();
            for i in [0, n / 2, n.saturating_sub(1)]
                .into_iter()
                .filter(|&i| i < n)
            {
                let how = caught(
                    &program,
                    &reference,
                    window,
                    Tamper::DropEntry { window: w, at: i },
                );
                assert_eq!(how, "refused", "{name}: window {w} without entry {i}");
                *seen.entry(("drop", how)).or_default() += 1;
            }
            // A moved timestamp reaches the tables where a memory row keeps it.
            let reached: Vec<usize> = (0..n)
                .filter(|&i| kept.binary_search(&entries[i].0).is_ok())
                .collect();
            let picks = [
                reached.first(),
                reached.get(reached.len() / 2),
                reached.last(),
            ];
            for &i in picks.into_iter().flatten() {
                let how = caught(
                    &program,
                    &reference,
                    window,
                    Tamper::LaterEntry { window: w, at: i },
                );
                assert_ne!(
                    how, "unseen",
                    "{name}: window {w}, entry {i} a timestamp later"
                );
                *seen.entry(("later entry", how)).or_default() += 1;
            }
            let how = caught(&program, &reference, window, Tamper::LaterFirst(w));
            assert!(
                matches!(how, "refused" | "stopped"),
                "{name}: window {w} a cycle later: {how}"
            );
            *seen.entry(("later first", how)).or_default() += 1;
            // The next window's snapshot differs from the window's own only
            // where the window wrote.
            if *wrote {
                let how = caught(&program, &reference, window, Tamper::LateStart(w));
                assert_ne!(how, "unseen", "{name}: window {w} with the next snapshot");
                *seen.entry(("late start", how)).or_default() += 1;
            }
            let how = caught(&program, &reference, window, Tamper::Register(w));
            assert_ne!(how, "unseen", "{name}: window {w} with x5 off");
            *seen.entry(("register", how)).or_default() += 1;
        }
    }
    println!("caught: {seen:?}");
    assert!(seen.values().sum::<usize>() > 60, "{seen:?}");
}

/// The bytes each value-only reader reads in a walked window: HINT's input,
/// ECSM's operands, KECCAK's and BLAKE3's states, the absorb's control block
/// and message. (The bytes COPY moves, the committed output's among them, are
/// read and written through MEMW ops like any load's.)
fn value_reads(window: &WalkedWindow) -> Vec<(&'static str, u64, u64)> {
    let w = &window.walk;
    let mut out = Vec::new();
    out.extend(w.hint_ops.iter().map(|o| ("HINT input", o.in_addr, 32)));
    for o in &w.ecsm_ops {
        out.push(("ECSM xG", o.addr_xg, 32));
        out.push(("ECSM k", o.addr_k, 32));
    }
    out.extend(
        w.keccak_ops
            .iter()
            .map(|o| ("KECCAK lanes", o.state_addr, 200)),
    );
    out.extend(
        w.blake3_ops
            .iter()
            .map(|o| ("BLAKE3 state", o.state_addr, 112)),
    );
    for o in &w.blake3_absorb_ops {
        out.push(("absorb control", o.ctrl_addr, 32));
        out.push(("absorb message", o.msg_addr, 64 * o.blocks.len() as u64));
    }
    out
}

/// A program's ELF and logs: an assembly test program, or the Rust guest
/// `hint_min` built by `make compile-programs-rust`.
fn program_of(name: &str) -> (Elf, Vec<Log>) {
    if name != "hint_min" {
        return run(name);
    }
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the workspace root")
        .join("executor/program_artifacts/rust/hint_min.elf");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("{} — run `make compile-programs-rust`: {e}", path.display()));
    let program = Elf::load(&bytes).expect("the ELF loads");
    let logs = Executor::new(&program, Vec::new())
        .expect("the executor")
        .run()
        .expect("the program runs")
        .logs;
    (program, logs)
}

/// ★ Every value-only reader goes through the reads that mark a byte the
/// window's state does not hold: for each kind, an entry inside that reader's
/// bytes, dropped, is refused before anything is computed from it (no panic,
/// no tables).
#[test]
fn a_dropped_entry_a_value_only_reader_reads_is_refused() {
    let mut seen = std::collections::BTreeMap::<&str, usize>::new();
    for name in [
        "hint_min",
        "test_ecsm",
        "test_keccak_multi",
        "test_blake3",
        "test_blake3_absorb",
        "test_commit_split",
    ] {
        let (program, logs) = program_of(name);
        let reference = whole(&program, &logs, &MaxRowsConfig::small());
        let window = if logs.len() < 5_000 {
            7
        } else {
            (logs.len() / 40) | 1
        };
        let mut builder =
            WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small()).expect("builder");
        let mut pass =
            LeanPass::new(Arc::clone(&builder.decode), &builder.register_state).expect("pass");
        let mut executor = Executor::new(&program, Vec::new()).expect("the executor");
        let mut windows = Vec::new();
        while let Some(logs) = executor
            .resume_with_limit(window)
            .expect("the run")
            .map(<[Log]>::to_vec)
        {
            windows.push(logs);
        }
        windows.pop();
        let (mut walker, _) = builder.split();
        let mut first = 0;
        for (k, logs) in windows.iter().enumerate() {
            pass.begin(k as u32).expect("a window");
            pass.pass(logs, first).expect("the pass");
            first += logs.len();
            let entries = pass.take_entries();
            let walked = walker.walk(logs).expect("the walk");
            for (reader, from, len) in value_reads(&walked) {
                let Some(at) = entries
                    .iter()
                    .position(|&(a, _)| a.wrapping_sub(from) < len)
                else {
                    continue;
                };
                if seen.get(reader).copied().unwrap_or(0) >= 3 {
                    continue;
                }
                let tamper = Tamper::DropEntry {
                    window: k as u32,
                    at,
                };
                let how = caught(&program, &reference, window, tamper);
                assert_eq!(
                    how, "refused",
                    "{name}: window {k}, a {reader} byte dropped"
                );
                *seen.entry(reader).or_default() += 1;
            }
        }
    }
    println!("refused: {seen:?}");
    for reader in [
        "HINT input",
        "ECSM xG",
        "ECSM k",
        "KECCAK lanes",
        "BLAKE3 state",
        "absorb control",
        "absorb message",
    ] {
        assert!(
            seen.contains_key(reader),
            "no {reader} byte dropped: {seen:?}"
        );
    }
}

/// ★ The state built from the pass for the finish is the walk's at the last
/// window's start: the same pages, the same set cells, each with the walk's
/// value and timestamp.
#[test]
fn the_state_built_for_the_finish_is_the_walks() {
    for name in PROGRAMS {
        let (program, logs) = run(name);
        for window in [7, 33, logs.len()] {
            let mut builder =
                WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small()).expect("builder");
            let mut pass =
                LeanPass::new(Arc::clone(&builder.decode), &builder.register_state).expect("pass");
            let mut executor = Executor::new(&program, Vec::new()).expect("the executor");
            let mut start = executor.memory().snapshot();
            let mut windows = Vec::new();
            while let Some(logs) = executor
                .resume_with_limit(window)
                .expect("the run")
                .map(<[Log]>::to_vec)
            {
                let next = executor.memory().snapshot();
                windows.push((logs, std::mem::replace(&mut start, next)));
            }
            let (_, last_start) = windows.pop().expect("a window");
            {
                let (mut walker, _) = builder.split();
                let mut first = 0;
                for (k, (logs, _)) in windows.iter().enumerate() {
                    pass.begin(k as u32).expect("a window");
                    pass.pass(logs, first).expect("the pass");
                    walker.walk(logs).expect("the walk");
                    first += logs.len();
                }
            }
            let built = pass.walked_state(&last_start, &builder.image, builder.memory_state.lean);
            let walked = &builder.memory_state;
            assert!(
                built.cells.page_bases().eq(walked.cells.page_bases()),
                "{name}/{window}: pages"
            );
            assert!(
                built.cells.iter().eq(walked.cells.iter()),
                "{name}/{window}: set cells, values and timestamps"
            );
            let regs = pass.registers();
            let w = &builder.register_state;
            assert!(
                regs.regs == w.regs
                    && regs.index_register == w.index_register
                    && regs.pc_register == w.pc_register,
                "{name}/{window}: registers"
            );
        }
    }
}

/// The run of `program` built by windows of `window` cycles walked on
/// `walkers` threads ([`walk_windows`]), the executor on a thread of its own as
/// the block's prove runs it; with `cut`, the executor sends window `cut`
/// without its snapshot.
fn walked_on_threads(
    program: &Elf,
    max_rows: &MaxRowsConfig,
    window: usize,
    walkers: usize,
    cut: Option<usize>,
) -> Result<Traces, Error> {
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)?.drop_streamed_ops()?;
    let (tx, rx) = std::sync::mpsc::sync_channel(2);
    let mut chunks = Vec::new();
    let report = std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut executor = Executor::new(program, Vec::new()).expect("the executor");
            let mut start = Some(executor.memory().snapshot());
            let mut k = 0;
            while let Some(logs) = executor
                .resume_with_limit(window)
                .expect("the run")
                .map(<[Log]>::to_vec)
            {
                let next = Some(executor.memory().snapshot());
                let mut at = std::mem::replace(&mut start, next);
                if cut == Some(k) {
                    at = None;
                }
                if tx.send((logs, at)).is_err() {
                    break;
                }
                k += 1;
            }
        });
        walk_windows(&mut builder, rx, walkers, |accumulator, walked, _| {
            chunks.extend(
                accumulator
                    .absorb(walked)
                    .into_iter()
                    .map(ChunkJob::generate),
            );
            Ok(())
        })
    })?;
    let mut traces = builder.finish(&report.last)?;
    traces.insert_streamed(chunks)?;
    Ok(traces)
}

/// ★ Walked on 1, 2, 3 and 8 threads at window lengths 1, 7, 32 and 33, the
/// windows build the whole-run tables.
#[test]
fn windows_walked_on_threads_build_the_whole_run_tables() {
    let max_rows = MaxRowsConfig::small();
    for name in [
        "all_instructions_64",
        "test_keccak_multi",
        "test_ecsm",
        "test_commit_split",
        "test_copy_ecalls",
        "test_copy_overlaps",
    ] {
        let (program, logs) = run(name);
        let reference = whole(&program, &logs, &max_rows);
        for window in [1, 7, 32, 33] {
            for walkers in [1, 2, 3, 8] {
                let traces = walked_on_threads(&program, &max_rows, window, walkers, None)
                    .unwrap_or_else(|e| {
                        panic!("{name}, window {window}, {walkers} walkers: {e:?}")
                    });
                same_traces(&reference, &traces);
            }
        }
    }
}

/// A window the executor sends without its memory stops every thread and is
/// refused, wherever it falls; so is an error from the absorbing side.
#[test]
fn an_error_anywhere_stops_the_walk_and_is_returned() {
    let max_rows = MaxRowsConfig::small();
    let (program, _) = run("test_keccak_multi");
    for cut in [0, 1, 5] {
        for walkers in [1, 3] {
            let built = walked_on_threads(&program, &max_rows, 7, walkers, Some(cut));
            assert!(
                matches!(&built, Err(Error::Prover(m)) if m.contains("without its memory")),
                "window {cut} cut, {walkers} walkers: {:?}",
                built.err()
            );
        }
    }
    let mut builder = WindowedTraceBuilder::new(&program, &[], &max_rows).expect("the builder");
    let (tx, rx) = std::sync::mpsc::sync_channel(2);
    let walked = std::thread::scope(|scope| {
        scope.spawn(move || {
            let mut executor = Executor::new(&program, Vec::new()).expect("the executor");
            let mut start = Some(executor.memory().snapshot());
            while let Some(logs) = executor
                .resume_with_limit(7)
                .expect("the run")
                .map(<[Log]>::to_vec)
            {
                let next = Some(executor.memory().snapshot());
                if tx
                    .send((logs, std::mem::replace(&mut start, next)))
                    .is_err()
                {
                    break;
                }
            }
        });
        let mut absorbed = 0;
        walk_windows(&mut builder, rx, 3, |_, _, _| {
            absorbed += 1;
            if absorbed == 4 {
                return Err(Error::Prover("the absorbing side stopped".into()));
            }
            Ok(())
        })
    });
    assert!(
        matches!(&walked, Err(Error::Prover(m)) if m.contains("absorbing side stopped")),
        "{:?}",
        walked.err()
    );
}

/// The run of program `name` by windows of 7 cycles on `walkers` threads,
/// each job numbered `number(k)`, on a thread of its own: what it returns, or
/// `None` if it has not returned within 30 s.
fn numbered_on_a_thread(
    name: &'static str,
    walkers: usize,
    number: fn(u32) -> u32,
) -> Option<Result<(), Error>> {
    let (out_tx, out_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (program, _) = run(name);
        let max_rows = MaxRowsConfig::small();
        let walked = WindowedTraceBuilder::new(&program, &[], &max_rows).and_then(|mut builder| {
            let (tx, rx) = std::sync::mpsc::sync_channel(2);
            std::thread::scope(|scope| {
                let program = &program;
                scope.spawn(move || {
                    let mut executor = Executor::new(program, Vec::new()).expect("the executor");
                    let mut start = Some(executor.memory().snapshot());
                    while let Some(logs) = executor
                        .resume_with_limit(7)
                        .expect("the run")
                        .map(<[Log]>::to_vec)
                    {
                        let next = Some(executor.memory().snapshot());
                        if tx
                            .send((logs, std::mem::replace(&mut start, next)))
                            .is_err()
                        {
                            break;
                        }
                    }
                });
                walk_numbered(&mut builder, rx, walkers, number, |_, _, _| Ok(()))
            })
            .map(|_| ())
        });
        let _ = out_tx.send(walked);
    });
    out_rx.recv_timeout(std::time::Duration::from_secs(30)).ok()
}

/// A window number the pass never hands out is refused, not waited for
/// forever; one handed out twice is refused too.
#[test]
fn a_window_never_walked_or_walked_twice_is_refused() {
    let name = "test_keccak_multi";
    // Enough windows to fill every permit with window 1 missing, at 4 walkers.
    let (_, logs) = run(name);
    assert!(logs.len() > 7 * 12, "{} cycles", logs.len());
    for walkers in [1, 2, 4] {
        let skipped = numbered_on_a_thread(name, walkers, |k| k + (k > 0) as u32);
        assert!(
            matches!(&skipped, Some(Err(Error::Prover(m))) if m == "window 1 was never walked"),
            "{walkers} walkers, window 1 skipped: {skipped:?}"
        );
        let repeated = numbered_on_a_thread(name, walkers, |k| if k == 2 { 1 } else { k });
        assert!(
            matches!(&repeated, Some(Err(Error::Prover(m))) if m == "window 1 was walked twice"),
            "{walkers} walkers, window 1 repeated: {repeated:?}"
        );
    }
}
