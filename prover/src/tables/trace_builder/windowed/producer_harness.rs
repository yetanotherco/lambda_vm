//! The block producer alone: executor → walker → accumulator → chunk
//! generators → finish, as the streamed block provers run it, with nothing
//! downstream. Each chunk is generated and dropped (or dropped ungenerated,
//! [`Sink::Skip`]); the finish's tables are dropped once built. It prints
//! where each stage spends its time, so the producer can be measured apart from
//! the card.
//!
//! The threads are the provers' (`block_whir::prove_streamed`,
//! `block::build_streamed`): the executor runs a bounded two windows ahead of
//! the walk and copies each window out; the walk holds one window back (the
//! run's last is `finish`'s); the accumulator routes each walked window and
//! hands out its chunk jobs. What the provers do with a chunk — the WHIR layout
//! thread's transposition, the STARK committers' Round-1 commit — is not here,
//! and neither are KECCAK_RND's and ECDAS's splits after the finish.
//!
//! Thread seconds come from `/proc/thread-self/schedstat` (Linux; `None`
//! elsewhere), peak memory from `VmHWM`. The threads are named (`prod-exec`,
//! `prod-walk`, `prod-acc`, `prod-gen<i>`, `prod-rss`) so a profiler can tell
//! them apart. With `live`, each stage prints a `PRODUCER LIVE` line as it
//! happens (every window, each thread's end, the finish) and `prod-rss` prints
//! the resident set every second, so a run killed for memory still says how far
//! it got.

use std::sync::Mutex;
use std::sync::mpsc;
use std::time::Instant;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::logs::Log;

use super::{ChunkJob, StreamTable, WalkedWindow, WindowedTraceBuilder};
use crate::Error;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::build_stamps;
use crate::test_utils::asm_elf_bytes;

/// What a generator does with a chunk job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sink {
    /// Generate the chunk's table, then drop it.
    Generate,
    /// Drop the job without generating it: the serial chain without the
    /// generators' share of the machine.
    Skip,
}

#[derive(Clone, Debug)]
pub(crate) struct ProducerConfig {
    /// Cycles per window.
    pub window: usize,
    pub max_rows: MaxRowsConfig,
    /// Generator threads.
    pub generators: usize,
    /// [`WindowedTraceBuilder::drop_streamed_ops`].
    pub drop_ops: bool,
    pub sink: Sink,
    /// Print `PRODUCER LIVE` lines as the run goes.
    pub live: bool,
}

/// One walked window, as the accumulator saw it.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WindowRow {
    pub cycles: usize,
    pub walk: f64,
    pub absorb: f64,
    pub jobs: usize,
    /// Memory bytes the window touches, each once: its accesses whose old
    /// timestamp precedes the window's first cycle (the state a window takes
    /// from the windows before it).
    pub first_touch_bytes: usize,
}

/// A thread's span and its own time.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ThreadTimes {
    /// Seconds from the producer's start to this thread's last step.
    pub done_at: f64,
    /// CPU seconds of the thread, when the platform reports them.
    pub cpu: Option<f64>,
    /// Seconds blocked receiving from the stage before.
    pub recv_blocked: f64,
    /// Seconds blocked sending to the stage after.
    pub send_blocked: f64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ProducerReport {
    pub cycles: usize,
    pub exec_setup: f64,
    pub exec_resume: f64,
    pub exec_copy: f64,
    pub exec: ThreadTimes,
    /// When each window left the executor.
    pub executed_at: Vec<f64>,
    pub builder_setup: f64,
    pub walker: ThreadTimes,
    /// When each window's walk ended.
    pub walked_at: Vec<f64>,
    pub accumulator: ThreadTimes,
    pub windows: Vec<WindowRow>,
    /// The builder's own stamps: walk, route, hand-out (summed over windows).
    pub stamp_walk: f64,
    pub stamp_route: f64,
    pub stamp_handout: f64,
    /// Seconds spent counting `first_touch_bytes` (not in `absorb`).
    pub census: f64,
    pub windows_done_at: f64,
    /// Per generator thread.
    pub generators: Vec<ThreadTimes>,
    /// Per streamed table: chunks and the seconds generating them.
    pub generated: Vec<(StreamTable, usize, f64)>,
    /// When each chunk was done (generated or skipped), in completion order.
    pub chunk_done_at: Vec<f64>,
    pub finish: f64,
    pub finish_marks: Vec<(String, f64)>,
    pub drop_tables: f64,
    pub total: f64,
    /// `VmHWM` after the windows, after the finish, at the end (GiB).
    pub peak_windows: Option<f64>,
    pub peak_finish: Option<f64>,
    pub peak_end: Option<f64>,
}

/// This thread's CPU seconds so far (Linux).
fn thread_cpu() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/thread-self/schedstat").ok()?;
    let ns: u64 = stat.split_whitespace().next()?.parse().ok()?;
    Some(ns as f64 * 1e-9)
}

/// The process's `VmHWM` in GiB (Linux).
fn peak_gib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / (1024.0 * 1024.0))
}

/// The process's `VmRSS` in GiB (Linux).
fn rss_gib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
    let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / (1024.0 * 1024.0))
}

fn cpu_since(start: Option<f64>) -> Option<f64> {
    Some(thread_cpu()? - start?)
}

/// [`WindowRow::first_touch_bytes`].
fn first_touch_bytes(window: &WalkedWindow) -> usize {
    let Some(t0) = window.cpu_ops.first().map(|op| op.timestamp) else {
        return 0;
    };
    let memw = &window.walk.memw;
    memw.aligned
        .iter()
        .chain(&memw.general)
        .filter(|op| !op.is_register)
        .map(|op| {
            op.old_timestamp[..usize::from(op.width).min(8)]
                .iter()
                .filter(|&&ts| ts < t0)
                .count()
        })
        .sum()
}

fn spawn_named<'scope, 'env, T: Send + 'scope>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    name: &str,
    f: impl FnOnce() -> T + Send + 'scope,
) -> Result<std::thread::ScopedJoinHandle<'scope, T>, Error> {
    std::thread::Builder::new()
        .name(name.to_string())
        .spawn_scoped(scope, f)
        .map_err(|e| Error::Prover(format!("spawn {name}: {e}")))
}

fn join<T>(handle: std::thread::ScopedJoinHandle<'_, T>, what: &str) -> Result<T, Error> {
    handle
        .join()
        .map_err(|_| Error::Prover(format!("the producer's {what} panicked")))
}

/// What the executor thread hands back.
struct ExecPart {
    setup: f64,
    resume: f64,
    copy: f64,
    times: ThreadTimes,
    executed_at: Vec<f64>,
    cycles: usize,
}

/// Runs the producer over `elf_bytes` on `input` and reports its stages.
pub(crate) fn run_producer(
    elf_bytes: &[u8],
    input: &[u8],
    cfg: &ProducerConfig,
) -> Result<ProducerReport, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::Execution(format!("{e}")))?;
    let program = &program;
    let window = cfg.window.max(1);
    let start = Instant::now();
    let since = || start.elapsed().as_secs_f64();
    let live = |line: String| {
        if cfg.live {
            println!("PRODUCER LIVE {:.3} {line}", since());
        }
    };
    let stop_rss = std::sync::atomic::AtomicBool::new(false);

    let jobs_rx: Mutex<Option<mpsc::Receiver<ChunkJob>>> = Mutex::new(None);
    let generated: Mutex<Vec<(StreamTable, usize, f64)>> = Mutex::new(Vec::new());
    let chunk_done_at: Mutex<Vec<f64>> = Mutex::new(Vec::new());

    let (mut report, exec, generators) = std::thread::scope(|scope| {
        if cfg.live {
            let stop_rss = &stop_rss;
            spawn_named(scope, "prod-rss", move || {
                let stopped = || stop_rss.load(std::sync::atomic::Ordering::Relaxed);
                while !stopped() {
                    live(format!(
                        "rss {} GiB · hwm {} GiB",
                        opt(rss_gib()),
                        opt(peak_gib())
                    ));
                    for _ in 0..10 {
                        if stopped() {
                            return;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                }
            })?;
        }
        // The executor, a window at a time, a bounded two windows ahead.
        let (log_tx, log_rx) = mpsc::sync_channel::<Vec<Log>>(2);
        let exec = spawn_named(scope, "prod-exec", move || {
            let cpu0 = thread_cpu();
            let mut times = ThreadTimes::default();
            let (mut resume, mut copy) = (0.0, 0.0);
            let mut executed_at = Vec::new();
            let mut cycles = 0usize;
            let t_setup = Instant::now();
            let mut executor = Executor::new(program, input.to_vec())
                .map_err(|e| Error::Execution(format!("{e}")))?;
            let setup = t_setup.elapsed().as_secs_f64();
            loop {
                let t_resume = Instant::now();
                let Some(logs) = executor
                    .resume_with_limit(window)
                    .map_err(|e| Error::Execution(format!("{e}")))?
                else {
                    break;
                };
                resume += t_resume.elapsed().as_secs_f64();
                let t_copy = Instant::now();
                let logs = logs.to_vec();
                copy += t_copy.elapsed().as_secs_f64();
                cycles += logs.len();
                let t_send = Instant::now();
                let sent = log_tx.send(logs);
                times.send_blocked += t_send.elapsed().as_secs_f64();
                executed_at.push(since());
                if sent.is_err() {
                    break;
                }
            }
            times.done_at = since();
            times.cpu = cpu_since(cpu0);
            live(format!(
                "executor done · {cycles} cycles · resume {resume:.3} · copy {copy:.3} · cpu {} s",
                opt(times.cpu)
            ));
            Ok::<_, Error>(ExecPart {
                setup,
                resume,
                copy,
                times,
                executed_at,
                cycles,
            })
        })?;

        // The chunk generators.
        let (job_tx, job_rx) = mpsc::sync_channel::<ChunkJob>(64);
        *jobs_rx.lock().unwrap_or_else(|e| e.into_inner()) = Some(job_rx);
        let mut generators = Vec::new();
        for i in 0..cfg.generators.max(1) {
            let (jobs_rx, generated, chunk_done_at) = (&jobs_rx, &generated, &chunk_done_at);
            generators.push(spawn_named(scope, &format!("prod-gen{i}"), move || {
                let cpu0 = thread_cpu();
                let mut times = ThreadTimes::default();
                loop {
                    let t_recv = Instant::now();
                    let job = {
                        let rx = jobs_rx.lock().unwrap_or_else(|e| e.into_inner());
                        match rx.as_ref().map(mpsc::Receiver::recv) {
                            Some(Ok(job)) => job,
                            _ => break,
                        }
                    };
                    times.recv_blocked += t_recv.elapsed().as_secs_f64();
                    let table = job.table;
                    let t_gen = Instant::now();
                    match cfg.sink {
                        Sink::Generate => drop(job.generate()),
                        Sink::Skip => drop(job),
                    }
                    let secs = t_gen.elapsed().as_secs_f64();
                    let mut by_table = generated.lock().unwrap_or_else(|e| e.into_inner());
                    match by_table.iter_mut().find(|(k, _, _)| *k == table) {
                        Some(row) => {
                            row.1 += 1;
                            row.2 += secs;
                        }
                        None => by_table.push((table, 1, secs)),
                    }
                    drop(by_table);
                    chunk_done_at
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(since());
                }
                times.done_at = since();
                times.cpu = cpu_since(cpu0);
                times
            })?);
        }

        // The builder: the accumulator on this thread, the walker on its own,
        // then the finish. `job_tx` goes with it, so the generators stop when
        // it does.
        let builder = spawn_named(
            scope,
            "prod-acc",
            move || -> Result<ProducerReport, Error> {
                let mut report = ProducerReport::default();
                let t_setup = Instant::now();
                let mut builder = WindowedTraceBuilder::new(program, input, &cfg.max_rows)?;
                if cfg.drop_ops {
                    builder = builder.drop_streamed_ops()?;
                }
                report.builder_setup = t_setup.elapsed().as_secs_f64();
                let cpu0 = thread_cpu();
                let last = {
                    let (mut walker, mut accumulator) = builder.split();
                    std::thread::scope(|inner| -> Result<Vec<Log>, Error> {
                        let (walked_tx, walked_rx) = mpsc::sync_channel::<(WalkedWindow, f64)>(2);
                        let walking = spawn_named(inner, "prod-walk", move || {
                            let cpu0 = thread_cpu();
                            let mut times = ThreadTimes::default();
                            let mut walked_at = Vec::new();
                            let mut walk_total = 0.0;
                            // One window held back: the run's last is `finish`'s.
                            let mut held: Option<Vec<Log>> = None;
                            loop {
                                let t_recv = Instant::now();
                                let Ok(logs) = log_rx.recv() else {
                                    break;
                                };
                                times.recv_blocked += t_recv.elapsed().as_secs_f64();
                                if let Some(prev) = held.replace(logs) {
                                    let t_walk = Instant::now();
                                    let walked = walker.walk(&prev)?;
                                    let secs = t_walk.elapsed().as_secs_f64();
                                    walk_total += secs;
                                    walked_at.push(since());
                                    let t_send = Instant::now();
                                    let sent = walked_tx.send((walked, secs));
                                    times.send_blocked += t_send.elapsed().as_secs_f64();
                                    if sent.is_err() {
                                        break;
                                    }
                                }
                            }
                            times.done_at = since();
                            times.cpu = cpu_since(cpu0);
                            live(format!(
                                "walker done · {} windows · walk {walk_total:.3} · cpu {} s",
                                walked_at.len(),
                                opt(times.cpu)
                            ));
                            let last = held.ok_or_else(|| {
                                Error::Execution("the run executed no cycle".to_string())
                            })?;
                            Ok::<_, Error>((last, times, walked_at))
                        })?;
                        loop {
                            let t_recv = Instant::now();
                            let Ok((walked, walk)) = walked_rx.recv() else {
                                break;
                            };
                            report.accumulator.recv_blocked += t_recv.elapsed().as_secs_f64();
                            let t_census = Instant::now();
                            let row = WindowRow {
                                cycles: walked.cpu_ops.len(),
                                walk,
                                first_touch_bytes: first_touch_bytes(&walked),
                                ..WindowRow::default()
                            };
                            report.census += t_census.elapsed().as_secs_f64();
                            let t_absorb = Instant::now();
                            let jobs = accumulator.absorb(walked);
                            let absorb = t_absorb.elapsed().as_secs_f64();
                            let row = WindowRow {
                                absorb,
                                jobs: jobs.len(),
                                ..row
                            };
                            live(format!(
                                "window {} · cycles {} · walk {:.4} · absorb {:.4} · jobs {} · first-touch \
                                 bytes {} · rss {} GiB",
                                report.windows.len(),
                                row.cycles,
                                row.walk,
                                row.absorb,
                                row.jobs,
                                row.first_touch_bytes,
                                opt(rss_gib())
                            ));
                            report.windows.push(row);
                            let t_send = Instant::now();
                            for job in jobs {
                                if job_tx.send(job).is_err() {
                                    return Err(Error::Prover("the generators stopped".into()));
                                }
                            }
                            report.accumulator.send_blocked += t_send.elapsed().as_secs_f64();
                        }
                        let (last, times, walked_at) = join(walking, "walker")??;
                        report.walker = times;
                        report.walked_at = walked_at;
                        Ok(last)
                    })?
                };
                drop(job_tx);
                report.windows_done_at = since();
                report.accumulator.done_at = report.windows_done_at;
                report.accumulator.cpu = cpu_since(cpu0);
                let stamps = builder.stamps();
                report.stamp_walk = stamps.walk;
                report.stamp_route = stamps.route;
                report.stamp_handout = stamps.generate;
                report.peak_windows = peak_gib();
                live(format!(
                    "windows done · walk {:.3} · route {:.3} · hand-out {:.3} · accumulator cpu {} s · \
                     rss {} GiB · hwm {} GiB",
                    report.stamp_walk,
                    report.stamp_route,
                    report.stamp_handout,
                    opt(report.accumulator.cpu),
                    opt(rss_gib()),
                    opt(report.peak_windows)
                ));
                build_stamps::start();
                let t_finish = Instant::now();
                let traces = builder.finish(&last)?;
                report.finish = t_finish.elapsed().as_secs_f64();
                report.finish_marks = build_stamps::take();
                report.peak_finish = peak_gib();
                live(format!(
                    "finish done · {:.3} s · hwm {} GiB",
                    report.finish,
                    opt(report.peak_finish)
                ));
                let t_drop = Instant::now();
                drop(traces);
                report.drop_tables = t_drop.elapsed().as_secs_f64();
                Ok(report)
            },
        )?;

        let report = join(builder, "builder");
        stop_rss.store(true, std::sync::atomic::Ordering::Relaxed);
        let report = report?;
        let generators: Vec<ThreadTimes> = generators
            .into_iter()
            .map(|g| join(g, "generator"))
            .collect::<Result<_, _>>()?;
        let exec = join(exec, "executor")?;
        let exec = exec?;
        Ok::<_, Error>((report?, exec, generators))
    })?;
    report.exec_setup = exec.setup;
    report.exec_resume = exec.resume;
    report.exec_copy = exec.copy;
    report.exec = exec.times;
    report.executed_at = exec.executed_at;
    report.cycles = exec.cycles;
    report.generators = generators;
    report.total = since();
    report.generated = generated.into_inner().unwrap_or_else(|e| e.into_inner());
    report.chunk_done_at = chunk_done_at
        .into_inner()
        .unwrap_or_else(|e| e.into_inner());
    report.peak_end = peak_gib();
    Ok(report)
}

fn opt(x: Option<f64>) -> String {
    x.map_or_else(|| "-".to_string(), |v| format!("{v:.3}"))
}

impl ProducerReport {
    pub(crate) fn chunks(&self) -> usize {
        self.windows.iter().map(|w| w.jobs).sum()
    }

    /// The report as `PRODUCER …` lines; the `PRODUCER SUMMARY` line is
    /// `key=value` pairs for a script to read.
    pub(crate) fn lines(&self, cfg: &ProducerConfig, label: &str) -> Vec<String> {
        let mut out = Vec::new();
        let cycles = self.cycles.max(1) as f64;
        let ns = |secs: f64| secs * 1e9 / cycles;
        let sum = |f: fn(&ThreadTimes) -> Option<f64>| -> Option<f64> {
            self.generators.iter().map(f).sum::<Option<f64>>()
        };
        out.push(format!(
            "PRODUCER CONFIG {label} · cycles {} · windows {} (+ the last) · window {} · max_rows.cpu {} · \
             generators {} · drop_ops {} · sink {:?}",
            self.cycles,
            self.windows.len(),
            cfg.window,
            cfg.max_rows.cpu,
            cfg.generators,
            if cfg.drop_ops { "on" } else { "off" },
            cfg.sink,
        ));
        let e = &self.exec;
        out.push(format!(
            "PRODUCER STAGE executor: span {:.3} s · cpu {} s · setup {:.3} · resume {:.3} · copy {:.3} · \
             send-blocked {:.3} · {:.1} ns/cycle (resume) · {:.1} M cycles/s (resume)",
            e.done_at,
            opt(e.cpu),
            self.exec_setup,
            self.exec_resume,
            self.exec_copy,
            e.send_blocked,
            ns(self.exec_resume),
            cycles / self.exec_resume.max(1e-9) / 1e6,
        ));
        let w = &self.walker;
        out.push(format!(
            "PRODUCER STAGE walker: span {:.3} s · cpu {} s · walk {:.3} · recv-blocked {:.3} · \
             send-blocked {:.3} · {:.1} ns/cycle (walk) · {:.1} M cycles/s (walk)",
            w.done_at,
            opt(w.cpu),
            self.stamp_walk,
            w.recv_blocked,
            w.send_blocked,
            ns(self.stamp_walk),
            cycles / self.stamp_walk.max(1e-9) / 1e6,
        ));
        let a = &self.accumulator;
        let absorb: f64 = self.windows.iter().map(|w| w.absorb).sum();
        out.push(format!(
            "PRODUCER STAGE accumulator: cpu {} s · absorb {absorb:.3} (route+append {:.3} · hand-out {:.3}) · \
             census {:.3} · recv-blocked {:.3} · send-blocked {:.3} · {:.1} ns/cycle (absorb)",
            opt(a.cpu),
            self.stamp_route,
            self.stamp_handout,
            self.census,
            a.recv_blocked,
            a.send_blocked,
            ns(absorb),
        ));
        let gen_secs: f64 = self.generated.iter().map(|(_, _, s)| s).sum();
        out.push(format!(
            "PRODUCER STAGE generators: {} chunks · generate {gen_secs:.3} thread-s · cpu {} thread-s · \
             recv-blocked {:.3} thread-s · last done {:.3} s",
            self.chunk_done_at.len(),
            opt(sum(|t| t.cpu)),
            self.generators.iter().map(|t| t.recv_blocked).sum::<f64>(),
            self.chunk_done_at.iter().copied().fold(0.0, f64::max),
        ));
        for (table, n, secs) in &self.generated {
            out.push(format!(
                "PRODUCER GENERATE {table:?}: {n} chunks · {secs:.3} s · {:.3} s/chunk",
                secs / (*n).max(1) as f64
            ));
        }
        out.push(format!(
            "PRODUCER STAGE finish: {:.3} s · tables dropped in {:.3} s · windows done at {:.3} s · total {:.3} s",
            self.finish, self.drop_tables, self.windows_done_at, self.total,
        ));
        for (mark, at) in &self.finish_marks {
            out.push(format!("PRODUCER FINISH-MARK {mark}: {at:.3} s"));
        }
        let at = |v: &[f64], i: usize| {
            v.get(i)
                .map_or_else(|| "-".to_string(), |s| format!("{s:.3}"))
        };
        let mut done = self.chunk_done_at.clone();
        done.sort_by(f64::total_cmp);
        out.push(format!(
            "PRODUCER HEAD: executor setup {:.3} · builder setup {:.3} · window 0 executed {} · window 1 \
             executed {} · window 0 walked {} · 1st chunk done {} · 10th chunk done {} (s since start)",
            self.exec_setup,
            self.builder_setup,
            at(&self.executed_at, 0),
            at(&self.executed_at, 1),
            at(&self.walked_at, 0),
            at(&done, 0),
            at(&done, 9),
        ));
        let mut bytes: Vec<usize> = self.windows.iter().map(|w| w.first_touch_bytes).collect();
        bytes.sort_unstable();
        let (b_max, b_med) = (
            bytes.last().copied().unwrap_or(0),
            bytes.get(bytes.len() / 2).copied().unwrap_or(0),
        );
        let b_mean = bytes.iter().sum::<usize>() as f64 / bytes.len().max(1) as f64;
        out.push(format!(
            "PRODUCER BYTES first-touch bytes per window: max {b_max} · median {b_med} · mean {b_mean:.0}"
        ));
        out.push(format!(
            "PRODUCER RSS VmHWM GiB: after the windows {} · after the finish {} · at the end {}",
            opt(self.peak_windows),
            opt(self.peak_finish),
            opt(self.peak_end),
        ));

        out.push(format!(
            "PRODUCER SUMMARY label={label} cycles={} windows={} exec_cpu={} exec_resume={:.3} exec_span={:.3} \
             walker_walk={:.3} walker_cpu={} acc_route={:.3} acc_handout={:.3} acc_cpu={} gen_secs={gen_secs:.3} \
             chunks={} finish={:.3} windows_done={:.3} total={:.3} bytes_max={b_max} peak_gib={}",
            self.cycles,
            self.windows.len(),
            opt(e.cpu),
            self.exec_resume,
            e.done_at,
            self.stamp_walk,
            opt(w.cpu),
            self.stamp_route,
            self.stamp_handout,
            opt(a.cpu),
            self.chunks(),
            self.finish,
            self.windows_done_at,
            self.total,
            opt(self.peak_end),
        ));
        out
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

/// The block producer over a real block (D-EXEC X0). Box only.
///
/// ```text
/// PRODUCER_ELF=/root/fixtures/ethrex_8f826601.elf \
/// PRODUCER_INPUT=/root/fixtures/ethrex_mainnet_25368371_573004e6.bin \
/// LAMBDA_VM_PRODUCER_WINDOW_LOG2=20 LAMBDA_VM_PRODUCER_ROWS_LOG2=21 \
/// LAMBDA_VM_PRODUCER_GENERATORS=3 LAMBDA_VM_PRODUCER_DROP_OPS=1 LAMBDA_VM_PRODUCER_SINK=drop \
/// cargo test --release -p lambda-vm-prover --lib the_block_producer_alone -- --ignored --nocapture
/// ```
///
/// Window 2^20 is the WHIR prover's (`BLOCK_WINDOW_LOG2`), 2^21 the STARK
/// prover's (`max_rows.cpu`). `LAMBDA_VM_PRODUCER_SINK=skip` drops each chunk
/// job ungenerated.
#[test]
#[ignore = "box only: executes and walks a real block"]
fn the_block_producer_alone() {
    let elf = std::env::var("PRODUCER_ELF").expect("PRODUCER_ELF: the guest ELF's path");
    let input = std::env::var("PRODUCER_INPUT").expect("PRODUCER_INPUT: the block input's path");
    let elf_bytes = std::fs::read(&elf).unwrap_or_else(|e| panic!("read {elf}: {e}"));
    let input_bytes = std::fs::read(&input).unwrap_or_else(|e| panic!("read {input}: {e}"));
    let sink = match std::env::var("LAMBDA_VM_PRODUCER_SINK").as_deref() {
        Ok("skip") => Sink::Skip,
        Ok("drop") | Err(_) => Sink::Generate,
        Ok(other) => panic!("LAMBDA_VM_PRODUCER_SINK={other}: drop or skip"),
    };
    let cfg = ProducerConfig {
        window: 1 << env_usize("LAMBDA_VM_PRODUCER_WINDOW_LOG2", 20),
        max_rows: MaxRowsConfig::uniform(1 << env_usize("LAMBDA_VM_PRODUCER_ROWS_LOG2", 21)),
        generators: env_usize("LAMBDA_VM_PRODUCER_GENERATORS", 3),
        drop_ops: env_usize("LAMBDA_VM_PRODUCER_DROP_OPS", 1) != 0,
        sink,
        live: true,
    };
    let label = std::path::Path::new(&input)
        .file_stem()
        .map_or_else(|| input.clone(), |s| s.to_string_lossy().into_owned());
    let report = run_producer(&elf_bytes, &input_bytes, &cfg).expect("the producer runs");
    for line in report.lines(&cfg, &label) {
        println!("{line}");
    }
}

/// The chunk jobs a plain `push_jobs` build hands out over the same
/// windows: the harness must drive the builder to the same chunks.
fn reference_jobs(elf_bytes: &[u8], max_rows: &MaxRowsConfig, window: usize) -> (usize, usize) {
    let program = Elf::load(elf_bytes).expect("the ELF loads");
    let logs = Executor::new(&program, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs;
    let mut builder = WindowedTraceBuilder::new(&program, &[], max_rows).expect("the builder");
    let windows: Vec<&[Log]> = logs.chunks(window).collect();
    let (body, _last) = windows.split_at(windows.len() - 1);
    let jobs: usize = body
        .iter()
        .map(|w| builder.push_jobs(w).expect("a window").len())
        .sum();
    (logs.len(), jobs)
}

/// [`first_touch_bytes`] counts each byte a window touches once: on the
/// keccak programs it is the number of distinct bytes the window's memory
/// accesses cover, window by window.
#[test]
fn first_touch_bytes_are_the_distinct_bytes_a_window_touches() {
    for name in ["test_keccak", "test_keccak_multi"] {
        let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
        let logs = Executor::new(&program, Vec::new())
            .expect("the executor starts")
            .run()
            .expect("the program runs")
            .logs;
        let mut builder =
            WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small()).expect("the builder");
        let (mut walker, _) = builder.split();
        let mut touched_any = false;
        for (i, w) in logs[..logs.len() - 1].chunks(33).enumerate() {
            let walked = walker.walk(w).expect("a window");
            let memw = &walked.walk.memw;
            let distinct: std::collections::BTreeSet<u64> = memw
                .aligned
                .iter()
                .chain(&memw.general)
                .filter(|op| !op.is_register)
                .flat_map(|op| (0..u64::from(op.width)).map(|k| op.base_address.wrapping_add(k)))
                .collect();
            assert_eq!(
                first_touch_bytes(&walked),
                distinct.len(),
                "{name}, window {i}"
            );
            touched_any |= !distinct.is_empty();
        }
        assert!(touched_any, "{name}: no window touched memory");
    }
}

/// The harness runs the producer end to end on small programs: every cycle
/// executed, the same chunks a plain windowed build hands out, every job
/// reaching a generator under both sinks, and the windows' memory census
/// non-empty.
#[test]
fn the_producer_harness_drives_the_builder() {
    let max_rows = MaxRowsConfig::small();
    for name in ["all_instructions_64", "test_keccak", "test_keccak_multi"] {
        let elf = asm_elf_bytes(name);
        for window in [7, 33] {
            let (cycles, jobs) = reference_jobs(&elf, &max_rows, window);
            for (sink, drop_ops) in [(Sink::Generate, true), (Sink::Skip, false)] {
                let cfg = ProducerConfig {
                    window,
                    max_rows: max_rows.clone(),
                    generators: 2,
                    drop_ops,
                    sink,
                    live: false,
                };
                let report = run_producer(&elf, &[], &cfg).expect("the producer runs");
                let what = format!("{name}, window {window}, {sink:?}");
                assert_eq!(report.cycles, cycles, "{what}: cycles");
                assert_eq!(
                    report.windows.len(),
                    cycles.div_ceil(window) - 1,
                    "{what}: windows"
                );
                assert_eq!(report.chunks(), jobs, "{what}: chunk jobs");
                assert_eq!(report.chunk_done_at.len(), jobs, "{what}: jobs done");
                // `all_instructions_64` makes no memory access.
                assert_eq!(
                    report.windows.iter().any(|w| w.first_touch_bytes > 0),
                    name != "all_instructions_64",
                    "{what}: memory touched"
                );
                assert!(!report.lines(&cfg, name).is_empty());
            }
        }
    }
}
