//! Regeneration on the WHIR block (D-WHIR-NODISK, #1013's D-REGEN): the shadow
//! regenerator.
//!
//! `LAMBDA_VM_BLOCK_REGEN=shadow` runs the sequential regenerator beside phase
//! B and drops nothing. Phase A records, per streamed chunk, a [`Recipe`]:
//! which chunk it is, the windows it spans, its place in the hand-out (its
//! rank, which is also its place in the packer's order, so in the groups), and
//! the digest of its packed columns taken on the layout thread that generated
//! them. Phase B re-executes the run from the start on threads of its own,
//! walks it with the regeneration walk ([`RegenBuilder`]), generates every
//! chunk that has a recipe as phase A did, checks its digest, and throws it
//! away. It reports the mismatches, its CPU, and when each chunk would have
//! been ready against phase B's groups as they ran, in the order they ran and
//! with the groups that hold no streamed chunk first ([`Plan`]). The proof is
//! the one the knob's absence makes.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::Error;
use crate::tables::gpack::TraceForm;
use crate::tables::trace_builder::{ChunkJob, RegenBuilder, StreamTable};

const GIB: f64 = (1u64 << 30) as f64;

/// What phase B does with the streamed chunks (`LAMBDA_VM_BLOCK_REGEN`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegenMode {
    /// `off` (and unset): nothing is recorded or regenerated.
    Off,
    /// `shadow`: phase A records every streamed chunk's recipe, and phase B
    /// regenerates each beside the prove, checks it and throws it away.
    Shadow,
}

/// [`RegenMode`] from `LAMBDA_VM_BLOCK_REGEN`; any other value is refused (an
/// error, never a panic).
pub(crate) fn regen_mode() -> Result<RegenMode, Error> {
    parse_regen_mode(std::env::var("LAMBDA_VM_BLOCK_REGEN").ok().as_deref())
}

/// [`regen_mode`] from the knob's `value`.
pub(crate) fn parse_regen_mode(value: Option<&str>) -> Result<RegenMode, Error> {
    match value.map(str::trim) {
        None | Some("off") => Ok(RegenMode::Off),
        Some("shadow") => Ok(RegenMode::Shadow),
        Some(other) => Err(Error::Prover(format!(
            "LAMBDA_VM_BLOCK_REGEN must be `off` or `shadow`, got `{other}`"
        ))),
    }
}

/// Generator threads of the regenerator: `LAMBDA_VM_BLOCK_REGEN_GENERATORS`
/// (1..=16), unset 3 (#1013's).
pub(crate) fn regen_generators() -> usize {
    std::env::var("LAMBDA_VM_BLOCK_REGEN_GENERATORS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| (1..=16).contains(n))
        .unwrap_or(3)
}

/// Packed columns' shape and digest ([`stark::regen::digest_of`], the spill
/// store's digest).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Packed {
    pub(crate) digest: [u64; 2],
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) bytes: usize,
}

impl Packed {
    pub(crate) fn of(packed: &multilinear::narrow::NarrowColumns) -> Self {
        Self {
            digest: stark::regen::digest_of(packed),
            rows: packed.rows(),
            cols: packed.cols(),
            bytes: packed.data().len(),
        }
    }
}

/// A regenerated chunk that is not the one phase A committed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RegenMismatch {
    pub(crate) table: StreamTable,
    pub(crate) index: usize,
    pub(crate) want: Packed,
    pub(crate) got: Packed,
}

impl std::fmt::Display for RegenMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (w, g) = (&self.want, &self.got);
        write!(
            f,
            "{:?}[{}]: regenerated {} rows × {} columns, {} B, digest {:016x}{:016x}; phase A \
             committed {} × {}, {} B, digest {:016x}{:016x}",
            self.table,
            self.index,
            g.rows,
            g.cols,
            g.bytes,
            g.digest[0],
            g.digest[1],
            w.rows,
            w.cols,
            w.bytes,
            w.digest[0],
            w.digest[1],
        )
    }
}

/// The check every regenerated chunk passes before anything reads it: its
/// packed shape and digest are the ones phase A recorded.
pub(crate) fn check_regenerated(
    table: StreamTable,
    index: usize,
    want: &Packed,
    got: &Packed,
) -> Result<(), RegenMismatch> {
    if want == got {
        Ok(())
    } else {
        Err(RegenMismatch {
            table,
            index,
            want: *want,
            got: *got,
        })
    }
}

/// A streamed chunk as phase A made it (D-REGEN §2.1): which chunk it is,
/// where its ops lie in the run, and its packed columns' digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Recipe {
    pub(crate) table: StreamTable,
    pub(crate) index: usize,
    /// Its ops (all streamed chunks are full).
    pub(crate) ops: usize,
    /// The window its first op lies in and its offset in that window's
    /// list, when phase A's windows were counted.
    pub(crate) first: Option<(usize, usize)>,
    /// The window whose hand-out completed it.
    pub(crate) last_window: usize,
    /// Its place in phase A's hand-out: its rank, and its place in the
    /// packer's order.
    pub(crate) order: usize,
    /// Its packed columns, once the thread that generated them digested them.
    pub(crate) packed: Option<Packed>,
}

impl Recipe {
    /// Its main cells, once packed.
    pub(crate) fn cells(&self) -> Option<u64> {
        self.packed.map(|p| (p.rows * p.cols) as u64)
    }
}

/// A streamed table's place in the recorder's lists: [`StreamTable::ALL`]'s,
/// KECCAK_RND aside (a build that streams it is never recorded).
fn slot(table: StreamTable) -> Option<usize> {
    match table {
        StreamTable::Cpu => Some(0),
        StreamTable::MemwRegister => Some(1),
        StreamTable::MemwAligned => Some(2),
        StreamTable::Memw => Some(3),
        StreamTable::Load => Some(4),
        StreamTable::Lt => Some(5),
        StreamTable::Shift => Some(6),
        StreamTable::Store => Some(7),
        StreamTable::KeccakRnd => None,
    }
}

/// Phase A's recipes: fed by the accumulator (each window's list lengths,
/// then the chunks it handed out) and by the threads that generate the
/// streamed chunks (their digests).
#[derive(Default)]
pub(crate) struct Recorder {
    inner: Mutex<Recorded>,
}

#[derive(Default)]
struct Recorded {
    /// Per streamed table, its list's length at the end of each window.
    ends: [Vec<usize>; 8],
    windows: usize,
    recipes: BTreeMap<(usize, usize), Recipe>,
    /// Digests for a chunk with no recipe, and chunks of a table no recipe
    /// can hold.
    stray: usize,
}

impl Recorder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Recorded> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The next window, with the ops it added to each streamed table's list
    /// ([`StreamTable::ALL`] order, KECCAK_RND aside), and the chunks its
    /// hand-out completed, in hand-out order.
    pub(crate) fn window(&self, added: [usize; 8], jobs: &[ChunkJob]) {
        let mut r = self.lock();
        let w = r.windows;
        r.windows += 1;
        for (t, n) in added.into_iter().enumerate() {
            let end = r.ends[t].last().copied().unwrap_or(0) + n;
            r.ends[t].push(end);
        }
        for job in jobs {
            let Some(t) = slot(job.table) else {
                r.stray += 1;
                continue;
            };
            let ops = job.op_count();
            let start = job.index * ops;
            let first = first_window(&r.ends[t], start);
            let order = r.recipes.len();
            r.recipes.insert(
                (t, job.index),
                Recipe {
                    table: job.table,
                    index: job.index,
                    ops,
                    first,
                    last_window: w,
                    order,
                    packed: None,
                },
            );
        }
    }

    /// The packed columns of chunk `index` of `table`, as the thread that
    /// generated them holds them.
    pub(crate) fn packed(
        &self,
        table: StreamTable,
        index: usize,
        packed: &multilinear::narrow::NarrowColumns,
    ) {
        let packed = Packed::of(packed);
        let mut r = self.lock();
        match slot(table).and_then(|t| r.recipes.get_mut(&(t, index))) {
            Some(recipe) => recipe.packed = Some(packed),
            None => r.stray += 1,
        }
    }

    /// The recipes in hand-out order, and the digests that found no recipe.
    pub(crate) fn finish(self) -> (Vec<Recipe>, usize) {
        let r = self.inner.into_inner().unwrap_or_else(|e| e.into_inner());
        let mut recipes: Vec<Recipe> = r.recipes.into_values().collect();
        recipes.sort_by_key(|recipe| recipe.order);
        (recipes, r.stray)
    }
}

/// The window op `start` of a list lies in, and its offset in that window's
/// part, from the list's length at the end of each window (`ends`).
fn first_window(ends: &[usize], start: usize) -> Option<(usize, usize)> {
    let w = ends.iter().position(|&end| end > start)?;
    let before = if w == 0 { 0 } else { ends[w - 1] };
    Some((w, start - before))
}

/// Phase A's recorder under `mode`, when phase A can feed it: `streams` (it
/// streams its windows and generates each streamed chunk packed, G-pack, and
/// streams neither KECCAK_RND nor the MEMW-derived LT ops, whose chunks the
/// regenerator does not cut). None otherwise, saying why.
pub(crate) fn shadow_recorder(mode: RegenMode, streams: bool) -> Option<Recorder> {
    match (mode, streams) {
        (RegenMode::Off, _) => None,
        (RegenMode::Shadow, true) => Some(Recorder::new()),
        (RegenMode::Shadow, false) => {
            eprintln!(
                "BLOCK REGEN: shadow wanted, but phase A does not stream its chunks packed (windows, \
                 G-pack, narrow groups; KECCAK_RND and the MEMW-derived LT ops not streamed): off"
            );
            None
        }
    }
}

/// This thread's CPU time so far, in seconds (Linux,
/// `/proc/thread-self/schedstat`).
fn thread_cpu_secs() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/thread-self/schedstat").ok()?;
    let ns: f64 = s.split_whitespace().next()?.parse().ok()?;
    Some(ns / 1e9)
}

/// The CPU a thread used between two readings.
fn cpu_since(start: Option<f64>) -> Option<f64> {
    Some(thread_cpu_secs()? - start?)
}

/// A chunk the shadow regenerated and checked.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Regenerated {
    pub(crate) table: StreamTable,
    pub(crate) index: usize,
    /// Seconds from the regenerator's start to the chunk checked.
    pub(crate) ready: f64,
    pub(crate) matched: bool,
    pub(crate) bytes: usize,
}

/// What the shadow regenerator did.
#[derive(Debug)]
pub(crate) struct ShadowReport {
    /// Recipes it was given, and of those with no digest (not regenerated).
    pub(crate) recipes: usize,
    pub(crate) undigested: usize,
    pub(crate) regenerated: Vec<Regenerated>,
    pub(crate) mismatches: Vec<RegenMismatch>,
    /// Chunks whose generation failed (a panic caught, columns that would not
    /// pack).
    pub(crate) failures: Vec<String>,
    /// Chunks cut that phase A did not stream (the finish's), not generated.
    pub(crate) beyond: usize,
    pub(crate) windows: usize,
    pub(crate) cycles: usize,
    /// Seconds from its start to its last chunk checked, or to its stop.
    pub(crate) wall: f64,
    /// Thread CPU seconds: the executor, the walker (with the slicer and its
    /// start state), the generators summed (Linux only).
    pub(crate) cpu: [Option<f64>; 3],
    pub(crate) generators: usize,
    /// Why it stopped before the end, if it did.
    pub(crate) error: Option<String>,
    /// The walker's carried memory state, in bytes, at its end.
    pub(crate) state_bytes: usize,
}

impl ShadowReport {
    fn new(recipes: usize, undigested: usize, generators: usize) -> Self {
        Self {
            recipes,
            undigested,
            regenerated: Vec::new(),
            mismatches: Vec::new(),
            failures: Vec::new(),
            beyond: 0,
            windows: 0,
            cycles: 0,
            wall: 0.0,
            cpu: [None; 3],
            generators,
            error: None,
            state_bytes: 0,
        }
    }

    /// The regenerator's CPU seconds, its three parts summed (Linux only).
    pub(crate) fn cpu_total(&self) -> Option<f64> {
        Some(self.cpu[0]? + self.cpu[1]? + self.cpu[2]?)
    }

    /// Recipes with a digest the shadow neither checked nor failed.
    pub(crate) fn missing(&self) -> usize {
        (self.recipes - self.undigested)
            .saturating_sub(self.regenerated.len() + self.failures.len())
    }

    /// The `BLOCK REGEN shadow` line.
    pub(crate) fn line(&self) -> String {
        let n = self.regenerated.len();
        let bytes: usize = self.regenerated.iter().map(|r| r.bytes).sum();
        let secs = |s: Option<f64>| s.map_or("n/a".to_string(), |s| format!("{s:.1}"));
        let (first, last) = self
            .regenerated
            .iter()
            .fold((f64::INFINITY, 0.0f64), |(lo, hi), r| {
                (lo.min(r.ready), hi.max(r.ready))
            });
        format!(
            "BLOCK REGEN shadow: {n} of {} chunks regenerated · {} mismatches · {} failed · {} \
             missing · {} undigested · {} not streamed (skipped) · {} windows walked ({:.2} M \
             cycles) · {:.2} GiB packed · ready from {:.2} s to {:.2} s · wall {:.2} s · CPU exec {} \
             · walk {} · {} generators {} = {} s · walk state {:.2} GiB{}",
            self.recipes,
            self.mismatches.len(),
            self.failures.len(),
            self.missing(),
            self.undigested,
            self.beyond,
            self.windows,
            self.cycles as f64 / 1e6,
            bytes as f64 / GIB,
            if first.is_finite() { first } else { 0.0 },
            last,
            self.wall,
            secs(self.cpu[0]),
            secs(self.cpu[1]),
            self.generators,
            secs(self.cpu[2]),
            secs(self.cpu_total()),
            self.state_bytes as f64 / GIB,
            self.error
                .as_ref()
                .map_or(String::new(), |e| format!(" · stopped: {e}")),
        )
    }
}

/// The shadow regenerator (D-REGEN §2.3, sequential form): `builder` walks
/// the run of `program` on `private_input`, executed `window` cycles at a
/// time on a thread of its own, and every chunk it cuts that has a digested
/// recipe is generated in `form`, packed if it is not, and checked on one of
/// `generators` threads, then dropped. Every time is from `started`. Stops
/// when every recipe is checked, on an error (reported), or when `stop` is
/// set. Never panics: a panic in generation counts as that chunk's failure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_shadow(
    program: &Elf,
    private_input: &[u8],
    mut builder: RegenBuilder,
    window: usize,
    recipes: &[Recipe],
    generators: usize,
    form: TraceForm,
    started: Instant,
    stop: &AtomicBool,
) -> ShadowReport {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::mpsc;
    let wanted: BTreeMap<(usize, usize), Packed> = recipes
        .iter()
        .filter_map(|r| Some(((slot(r.table)?, r.index), r.packed?)))
        .collect();
    let mut report = ShadowReport::new(recipes.len(), recipes.len() - wanted.len(), generators);
    if wanted.is_empty() {
        return report;
    }
    let generators = generators.max(1);
    let left = AtomicUsize::new(wanted.len());
    let done = AtomicBool::new(false);
    let regenerated = Mutex::new(Vec::with_capacity(wanted.len()));
    let mismatches = Mutex::new(Vec::new());
    let failures = Mutex::new(Vec::new());
    let last_ready = Mutex::new(0.0f64);
    let quit = || stop.load(Ordering::Relaxed) || done.load(Ordering::Relaxed);
    let (job_tx, job_rx) = mpsc::sync_channel::<(ChunkJob, Packed)>(2 * generators);
    let job_rx = Mutex::new(job_rx);
    std::thread::scope(|s| {
        let (log_tx, log_rx) = mpsc::sync_channel::<Vec<executor::vm::logs::Log>>(2);
        let quit = &quit;
        let exec = std::thread::Builder::new()
            .name("regen-exec".to_string())
            .spawn_scoped(s, move || {
                let cpu0 = thread_cpu_secs();
                let run = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
                    let mut executor = Executor::new(program, private_input.to_vec())
                        .map_err(|e| format!("the executor: {e}"))?;
                    while let Some(logs) = executor
                        .resume_with_limit(window)
                        .map_err(|e| format!("the execution: {e}"))?
                    {
                        if quit() || log_tx.send(logs.to_vec()).is_err() {
                            break;
                        }
                    }
                    Ok(())
                }))
                .unwrap_or_else(|_| Err("the executor panicked".to_string()));
                (run, cpu_since(cpu0))
            });
        let exec = match exec {
            Ok(exec) => exec,
            Err(e) => {
                report.error = Some(format!("the executor thread did not start: {e}"));
                return;
            }
        };
        let mut gens = Vec::with_capacity(generators);
        for g in 0..generators {
            let (job_rx, regenerated, mismatches, failures, left, last_ready, done) = (
                &job_rx,
                &regenerated,
                &mismatches,
                &failures,
                &left,
                &last_ready,
                &done,
            );
            let spawned =
                std::thread::Builder::new()
                    .name(format!("regen-gen-{g}"))
                    .spawn_scoped(s, move || {
                        let cpu0 = thread_cpu_secs();
                        loop {
                            let next = job_rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                            let Ok((job, want)) = next else {
                                break;
                            };
                            if stop.load(Ordering::Relaxed) {
                                continue;
                            }
                            let (table, index) = (job.table, job.index);
                            let got = catch_unwind(AssertUnwindSafe(|| {
                                let mut trace = job.generate_as(form).trace;
                                trace
                                    .pack_main_narrow()
                                    .then(|| trace.narrow_main().map(Packed::of))
                            }));
                            let ready = started.elapsed().as_secs_f64();
                            let got = match got {
                                Ok(Some(Some(got))) => got,
                                Ok(_) => {
                                    failures.lock().unwrap_or_else(|e| e.into_inner()).push(
                                        format!("{table:?}[{index}]: the columns did not pack"),
                                    );
                                    continue;
                                }
                                Err(_) => {
                                    failures
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner())
                                        .push(format!("{table:?}[{index}]: generation panicked"));
                                    continue;
                                }
                            };
                            let matched = match check_regenerated(table, index, &want, &got) {
                                Ok(()) => true,
                                Err(mismatch) => {
                                    mismatches
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner())
                                        .push(mismatch);
                                    false
                                }
                            };
                            regenerated.lock().unwrap_or_else(|e| e.into_inner()).push(
                                Regenerated {
                                    table,
                                    index,
                                    ready,
                                    matched,
                                    bytes: got.bytes,
                                },
                            );
                            {
                                let mut last = last_ready.lock().unwrap_or_else(|e| e.into_inner());
                                *last = last.max(ready);
                            }
                            if left.fetch_sub(1, Ordering::Relaxed) == 1 {
                                done.store(true, Ordering::Relaxed);
                            }
                        }
                        cpu_since(cpu0)
                    });
            match spawned {
                Ok(handle) => gens.push(handle),
                Err(e) => report
                    .failures
                    .push(format!("generator {g} did not start: {e}")),
            }
        }
        if gens.is_empty() {
            report.error = Some("no generator thread started".to_string());
        }
        // This thread walks and cuts the chunks, and hands out those with a
        // recipe.
        let cpu0 = thread_cpu_secs();
        let mut dispatched = 0usize;
        for logs in log_rx.iter() {
            if quit() || gens.is_empty() {
                break;
            }
            report.windows += 1;
            let jobs = match catch_unwind(AssertUnwindSafe(|| builder.push(&logs))) {
                Ok(Ok(jobs)) => jobs,
                Ok(Err(e)) => {
                    report.error = Some(format!("the regeneration walk: {e}"));
                    break;
                }
                Err(_) => {
                    report.error = Some("the regeneration walk panicked".to_string());
                    break;
                }
            };
            for job in jobs {
                match slot(job.table).and_then(|t| wanted.get(&(t, job.index))) {
                    Some(&want) => {
                        if job_tx.send((job, want)).is_err() {
                            break;
                        }
                        dispatched += 1;
                    }
                    None => report.beyond += 1,
                }
            }
            if dispatched == wanted.len() {
                break;
            }
        }
        drop(log_rx);
        drop(job_tx);
        report.cycles = builder.cycles();
        report.state_bytes = builder.state_bytes();
        report.cpu[1] = cpu_since(cpu0);
        let mut gen_cpu = Some(0.0);
        for g in gens {
            let cpu = g.join().unwrap_or(None);
            gen_cpu = gen_cpu.zip(cpu).map(|(a, b)| a + b);
        }
        report.cpu[2] = gen_cpu;
        match exec.join() {
            Ok((run, cpu)) => {
                report.cpu[0] = cpu;
                if let Err(e) = run
                    && report.error.is_none()
                {
                    report.error = Some(e);
                }
            }
            Err(_) => report.error = Some("the executor thread panicked".to_string()),
        }
    });
    if stop.load(Ordering::Relaxed) && report.error.is_none() {
        report.error = Some("stopped by the prove".to_string());
    }
    report.regenerated = regenerated.into_inner().unwrap_or_else(|e| e.into_inner());
    report.mismatches = mismatches.into_inner().unwrap_or_else(|e| e.into_inner());
    report
        .failures
        .extend(failures.into_inner().unwrap_or_else(|e| e.into_inner()));
    report.wall = if report.regenerated.len() == wanted.len() {
        *last_ready.lock().unwrap_or_else(|e| e.into_inner())
    } else {
        started.elapsed().as_secs_f64()
    };
    report
}

/// The shadow regenerator on a thread of its own (`regen-walk`), for a prove
/// that goes on beside it. Dropped, it stops the regenerator and joins it, so
/// no regenerator outlives its prove on any exit.
pub(crate) struct ShadowRun {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<ShadowReport>>,
    started: Instant,
    recipes: usize,
    /// Why the thread did not start, if it did not.
    spawn_error: Option<String>,
}

impl ShadowRun {
    /// [`run_shadow`] over the run of `elf_bytes` on `private_input`, executed
    /// `window` cycles at a time (phase A's windows), its chunks cut at
    /// `max_rows` and generated in `form` (phase A's), on
    /// [`regen_generators`] generators; its times are from now. `drop_one_op`
    /// is a test's slicer bug ([`RegenBuilder`]'s), `None` outside tests.
    pub(crate) fn spawn(
        elf_bytes: Vec<u8>,
        private_input: Vec<u8>,
        max_rows: crate::tables::MaxRowsConfig,
        window: usize,
        recipes: Vec<Recipe>,
        form: TraceForm,
        drop_one_op: Option<StreamTable>,
    ) -> Self {
        let started = Instant::now();
        let stop = Arc::new(AtomicBool::new(false));
        let count = recipes.len();
        let stop_in = Arc::clone(&stop);
        let spawned = std::thread::Builder::new()
            .name("regen-walk".to_string())
            .spawn(move || {
                // The walker's CPU includes building its start state (the
                // image, the decode table).
                let cpu0 = thread_cpu_secs();
                let program = match Elf::load(&elf_bytes) {
                    Ok(program) => program,
                    Err(e) => {
                        let mut report = ShadowReport::new(recipes.len(), 0, 0);
                        report.error = Some(format!("the regenerator's ELF: {e}"));
                        return report;
                    }
                };
                match RegenBuilder::new(&program, &private_input, &max_rows) {
                    Ok(builder) => {
                        #[cfg(test)]
                        let builder = match drop_one_op {
                            Some(table) => builder.drop_one_op(table),
                            None => builder,
                        };
                        #[cfg(not(test))]
                        let _ = drop_one_op;
                        let setup = cpu_since(cpu0);
                        let mut report = run_shadow(
                            &program,
                            &private_input,
                            builder,
                            window,
                            &recipes,
                            regen_generators(),
                            form,
                            started,
                            &stop_in,
                        );
                        report.cpu[1] = report.cpu[1].zip(setup).map(|(walk, setup)| walk + setup);
                        report
                    }
                    Err(e) => {
                        let mut report = ShadowReport::new(recipes.len(), 0, 0);
                        report.error = Some(format!("the regenerator: {e}"));
                        report
                    }
                }
            });
        let (handle, spawn_error) = match spawned {
            Ok(handle) => (Some(handle), None),
            Err(e) => (None, Some(format!("the regenerator did not start: {e}"))),
        };
        Self {
            stop,
            handle,
            started,
            recipes: count,
            spawn_error,
        }
    }

    /// When it started: every time its report gives is from here.
    pub(crate) fn started(&self) -> Instant {
        self.started
    }

    /// Its report once done; a thread that did not start or that panicked is
    /// a report with the error.
    pub(crate) fn join(mut self) -> ShadowReport {
        let failed = |error: String| {
            let mut report = ShadowReport::new(self.recipes, 0, 0);
            report.error = Some(error);
            report
        };
        match self.handle.take() {
            Some(handle) => handle
                .join()
                .unwrap_or_else(|_| failed("the regenerator panicked".to_string())),
            None => failed(
                self.spawn_error
                    .take()
                    .unwrap_or_else(|| "the regenerator did not start".to_string()),
            ),
        }
    }
}

impl Drop for ShadowRun {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// When the regenerated chunks would have been needed by phase B's groups as
/// they ran: each group takes what it took (its measured span), in `order`,
/// and a group's chunks are needed when the group before it in that order
/// starts — phase B brings a group's columns back with the group before it,
/// whose argue uploads them. Lateness is a chunk's ready time (on phase B's
/// clock) less that need; with a serial consumer the stall the order would pay
/// is the largest lateness, at least 0. A chunk not regenerated, or
/// regenerated wrong, is missing, and with any missing there is no
/// would-wait.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Plan {
    pub(crate) would_wait: f64,
    /// The latest chunk, and how late it was (negative: early).
    pub(crate) worst: Option<(StreamTable, usize, f64)>,
    /// Chunks ready after they were needed.
    pub(crate) late: usize,
    /// Chunks in the plan, those never regenerated, and those whose group
    /// is not known.
    pub(crate) planned: usize,
    pub(crate) missing: usize,
    pub(crate) ungrouped: usize,
    /// Groups holding a streamed chunk, and the others.
    pub(crate) streamed_groups: usize,
    pub(crate) other_groups: usize,
    /// When the first streamed group's chunks are needed (seconds of phase B).
    pub(crate) head: f64,
}

/// The order of phase B's `groups` a [`Plan`] reads: as they ran, or with
/// every group that holds no streamed chunk first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlanOrder {
    GroupOrder,
    RestFirst,
}

/// [`Plan`] for `recipes` regenerated as `report` says, under `order`:
/// `group_of` gives a recipe's group, `groups` each group's phase-B span
/// (start, end) in seconds of phase B, and phase B began `prove_offset`
/// seconds after the regenerator.
pub(crate) fn plan(
    recipes: &[Recipe],
    group_of: &dyn Fn(&Recipe) -> Option<usize>,
    report: &ShadowReport,
    groups: &[(f64, f64)],
    prove_offset: f64,
    order: PlanOrder,
) -> Plan {
    // A mismatched chunk would be refused, so it is never ready.
    let ready: std::collections::HashMap<(StreamTable, usize), f64> = report
        .regenerated
        .iter()
        .filter(|r| r.matched)
        .map(|r| ((r.table, r.index), r.ready - prove_offset))
        .collect();
    let mut streamed = vec![false; groups.len()];
    for recipe in recipes {
        if let Some(g) = group_of(recipe).filter(|&g| g < groups.len()) {
            streamed[g] = true;
        }
    }
    let sequence: Vec<usize> = match order {
        PlanOrder::GroupOrder => (0..groups.len()).collect(),
        PlanOrder::RestFirst => (0..groups.len())
            .filter(|&g| !streamed[g])
            .chain((0..groups.len()).filter(|&g| streamed[g]))
            .collect(),
    };
    // Each group's need: the start of the group before it in `sequence`.
    let mut need = vec![0.0f64; groups.len()];
    let mut at = 0.0f64;
    let mut previous_start = 0.0f64;
    for (p, &g) in sequence.iter().enumerate() {
        need[g] = if p == 0 { 0.0 } else { previous_start };
        previous_start = at;
        let (start, end) = groups[g];
        at += (end - start).max(0.0);
    }
    let head = sequence
        .iter()
        .find(|&&g| streamed[g])
        .map_or(0.0, |&g| need[g]);
    let (mut worst, mut late, mut missing, mut ungrouped, mut planned) =
        (None::<(StreamTable, usize, f64)>, 0, 0, 0, 0);
    for recipe in recipes.iter().filter(|r| r.packed.is_some()) {
        planned += 1;
        let Some(g) = group_of(recipe).filter(|&g| g < groups.len()) else {
            ungrouped += 1;
            continue;
        };
        let Some(&ready) = ready.get(&(recipe.table, recipe.index)) else {
            missing += 1;
            continue;
        };
        let lateness = ready - need[g];
        if lateness > 0.0 {
            late += 1;
        }
        if worst.is_none_or(|(_, _, w)| lateness > w) {
            worst = Some((recipe.table, recipe.index, lateness));
        }
    }
    Plan {
        would_wait: worst.map_or(0.0, |(_, _, w)| w.max(0.0)),
        worst,
        late,
        planned,
        missing,
        ungrouped,
        streamed_groups: streamed.iter().filter(|&&s| s).count(),
        other_groups: streamed.iter().filter(|&&s| !s).count(),
        head,
    }
}

impl Plan {
    /// The `BLOCK REGEN plan` line, with the regenerator's CPU over phase B.
    pub(crate) fn line(
        &self,
        order: PlanOrder,
        report: &ShadowReport,
        prove: f64,
        prove_offset: f64,
    ) -> String {
        let cores = report
            .cpu_total()
            .filter(|_| prove > 0.0)
            .map_or("n/a".to_string(), |cpu| format!("{:.2}", cpu / prove));
        format!(
            "BLOCK REGEN plan ({}): {} streamed chunks in {} groups after {} other groups (first \
             needed at {:.2} s of phase B) · would-wait {}{} · late {} · missing {} · ungrouped {} \
             · regen CPU {} s = {cores} cores over phase B ({prove:.2} s, starting {prove_offset:.2} \
             s after the regenerator)",
            match order {
                PlanOrder::GroupOrder => "group order",
                PlanOrder::RestFirst => "rest first",
            },
            self.planned,
            self.streamed_groups,
            match order {
                PlanOrder::GroupOrder => "or between the".to_string(),
                PlanOrder::RestFirst => format!("{}", self.other_groups),
            },
            self.head,
            if self.missing > 0 || self.ungrouped > 0 {
                "n/a".to_string()
            } else {
                format!("{:.3} s", self.would_wait)
            },
            self.worst.map_or(String::new(), |(t, i, w)| format!(
                " (latest {t:?}[{i}] at {w:+.3} s)"
            )),
            self.late,
            self.missing,
            self.ungrouped,
            report
                .cpu_total()
                .map_or("n/a".to_string(), |c| format!("{c:.1}")),
        )
    }
}

/// The `BLOCK REGEN recorded` line: what phase A recorded against the
/// streamed class (`streamed` chunks of `streamed_cells` cells).
pub(crate) fn recorded_line(
    recipes: &[Recipe],
    stray: usize,
    streamed: usize,
    streamed_cells: u64,
) -> String {
    let digested = recipes.iter().filter(|r| r.packed.is_some()).count();
    let cells: u64 = recipes.iter().filter_map(Recipe::cells).sum();
    let spans: Vec<usize> = recipes
        .iter()
        .filter_map(|r| Some(r.last_window + 1 - r.first?.0))
        .collect();
    format!(
        "BLOCK REGEN recorded: {} streamed chunks of {streamed} · {digested} digested · {stray} \
         stray digests · {:.3} G cells (the streamed class {:.3} G) · windows spanned mean {:.2} \
         max {}",
        recipes.len(),
        cells as f64 / 1e9,
        streamed_cells as f64 / 1e9,
        spans.iter().sum::<usize>() as f64 / spans.len().max(1) as f64,
        spans.iter().max().copied().unwrap_or(0),
    )
}

/// The shadow's readout once phase B is done: the `BLOCK REGEN` lines (what
/// phase A recorded against the `streamed` chunks of `streamed_cells` cells,
/// what the shadow did and each mismatch and failure, the first 20; the plan
/// in group order and rest first over phase B's group `spans`; when the shadow
/// was joined) and the counts. `group_of` gives a streamed chunk's group;
/// phase B took `prove` seconds from `prove_offset` seconds after the
/// regenerator started, and its join waited `joined` seconds after phase B.
#[allow(clippy::too_many_arguments)]
pub(crate) fn shadow_readout(
    recipes: &[Recipe],
    stray: usize,
    report: &ShadowReport,
    group_of: &dyn Fn(StreamTable, usize) -> Option<usize>,
    streamed: usize,
    streamed_cells: u64,
    spans: &[(f64, f64)],
    prove: f64,
    prove_offset: f64,
    joined: f64,
) -> RegenStamps {
    let mut lines = vec![
        recorded_line(recipes, stray, streamed, streamed_cells),
        report.line(),
    ];
    lines.extend(
        report
            .mismatches
            .iter()
            .take(20)
            .map(|m| format!("BLOCK REGEN MISMATCH {m}")),
    );
    lines.extend(
        report
            .failures
            .iter()
            .take(20)
            .map(|f| format!("BLOCK REGEN FAILED {f}")),
    );
    let of = |r: &Recipe| group_of(r.table, r.index);
    let mut would_wait_rest_first = None;
    for order in [PlanOrder::GroupOrder, PlanOrder::RestFirst] {
        let p = plan(recipes, &of, report, spans, prove_offset, order);
        if order == PlanOrder::RestFirst && p.missing == 0 && p.ungrouped == 0 {
            would_wait_rest_first = Some(p.would_wait);
        }
        lines.push(p.line(order, report, prove, prove_offset));
    }
    lines.push(format!(
        "BLOCK REGEN shadow joined {joined:.2} s after phase B"
    ));
    RegenStamps {
        recipes: recipes.len(),
        digested: recipes.iter().filter(|r| r.packed.is_some()).count(),
        stray,
        regenerated: report.regenerated.len(),
        mismatches: report.mismatches.len(),
        failed: report.failures.len(),
        missing: report.missing(),
        error: report.error.clone(),
        would_wait_rest_first,
        lines,
    }
}

/// What a shadow run left for the readout ([`super::BlockStamps::regen`]).
#[derive(Clone, Debug, Default)]
pub struct RegenStamps {
    pub recipes: usize,
    pub digested: usize,
    pub stray: usize,
    pub regenerated: usize,
    pub mismatches: usize,
    pub failed: usize,
    pub missing: usize,
    pub error: Option<String>,
    /// Would-wait under the rest-first plan, when it is known.
    pub would_wait_rest_first: Option<f64>,
    /// The `BLOCK REGEN` lines.
    pub lines: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `LAMBDA_VM_BLOCK_REGEN`: `off` (and unset) and `shadow`; anything else
    /// is refused (an error, not a panic).
    #[test]
    fn the_regen_mode_reads_its_knob() {
        assert_eq!(parse_regen_mode(None).unwrap(), RegenMode::Off);
        assert_eq!(parse_regen_mode(Some(" off ")).unwrap(), RegenMode::Off);
        assert_eq!(parse_regen_mode(Some("shadow")).unwrap(), RegenMode::Shadow);
        for bad in ["", "on", "Shadow", "1", "auto", "always"] {
            assert!(parse_regen_mode(Some(bad)).is_err(), "{bad:?}");
        }
    }

    /// A chunk's first op in the windows: lists of 3, 3, 9 and 14 ops after
    /// windows 0..4 (window 1 added none), chunks of 4 ops.
    #[test]
    fn a_chunks_first_op_is_found_in_its_window() {
        let ends = [3, 3, 9, 14];
        assert_eq!(first_window(&ends, 0), Some((0, 0)));
        assert_eq!(first_window(&ends, 4), Some((2, 1)));
        assert_eq!(first_window(&ends, 8), Some((2, 5)));
        assert_eq!(first_window(&ends, 12), Some((3, 3)));
        assert_eq!(first_window(&ends, 14), None);
        assert_eq!(first_window(&[], 0), None);
    }

    fn recipe(table: StreamTable, index: usize, order: usize) -> Recipe {
        Recipe {
            table,
            index,
            ops: 4,
            first: Some((order, 0)),
            last_window: order,
            order,
            packed: Some(Packed {
                digest: [0, 0],
                rows: 10,
                cols: 10,
                bytes: 100,
            }),
        }
    }

    fn ready(table: StreamTable, index: usize, at: f64, matched: bool) -> Regenerated {
        Regenerated {
            table,
            index,
            ready: at,
            matched,
            bytes: 100,
        }
    }

    /// ★ The plan over five groups of phase B, each 2 s: groups 0 and 1 hold
    /// the streamed chunks (CPU[0], MEMW[0] in 0; CPU[1] in 1), groups 2–4 the
    /// rest. In group order group 0 is needed at once and group 1 when group 0
    /// starts (0 s); rest first, the groups run 2 3 4 0 1, so group 0 is needed
    /// when group 4 starts (4 s) and group 1 when group 0 starts (6 s).
    /// Lateness is ready − need on phase B's clock; the would-wait the largest,
    /// at least 0; a mismatched chunk is missing, and the line says n/a.
    #[test]
    fn the_plan_reads_phase_b_in_group_order_and_rest_first() {
        use StreamTable::{Cpu, Memw};
        let recipes = [recipe(Cpu, 0, 0), recipe(Memw, 0, 1), recipe(Cpu, 1, 2)];
        let group_of = |r: &Recipe| Some(usize::from(r.table == Cpu && r.index == 1));
        let groups: Vec<(f64, f64)> = (0..5)
            .map(|g| (2.0 * g as f64, 2.0 * g as f64 + 2.0))
            .collect();
        let mut report = ShadowReport::new(3, 0, 2);
        report.regenerated = vec![
            ready(Cpu, 0, 3.0, true),
            ready(Memw, 0, 5.0, true),
            ready(Cpu, 1, 8.0, true),
        ];
        let p = plan(
            &recipes,
            &group_of,
            &report,
            &groups,
            0.0,
            PlanOrder::GroupOrder,
        );
        assert_eq!((p.planned, p.missing, p.ungrouped, p.late), (3, 0, 0, 3));
        assert_eq!((p.streamed_groups, p.other_groups), (2, 3));
        assert_eq!(p.worst, Some((Cpu, 1, 8.0)));
        assert!((p.would_wait - 8.0).abs() < 1e-9);
        let p = plan(
            &recipes,
            &group_of,
            &report,
            &groups,
            0.0,
            PlanOrder::RestFirst,
        );
        assert!((p.head - 4.0).abs() < 1e-9, "{p:?}");
        // CPU[0] 3 − 4, MEMW[0] 5 − 4, CPU[1] 8 − 6.
        assert_eq!(p.late, 2);
        assert_eq!(p.worst, Some((Cpu, 1, 2.0)));
        assert!((p.would_wait - 2.0).abs() < 1e-9);
        // The regenerator 3 s ahead of phase B: everything early.
        let p = plan(
            &recipes,
            &group_of,
            &report,
            &groups,
            3.0,
            PlanOrder::RestFirst,
        );
        assert_eq!((p.late, p.would_wait), (0, 0.0));
        // A mismatched chunk is missing, and the line says n/a.
        report.regenerated[1].matched = false;
        let p = plan(
            &recipes,
            &group_of,
            &report,
            &groups,
            0.0,
            PlanOrder::RestFirst,
        );
        assert_eq!(p.missing, 1);
        assert!(
            p.line(PlanOrder::RestFirst, &report, 10.0, 0.0)
                .contains("would-wait n/a")
        );
        // A chunk whose group is unknown is not planned against anything.
        let p = plan(
            &recipes,
            &|_| None,
            &report,
            &groups,
            0.0,
            PlanOrder::RestFirst,
        );
        assert_eq!(p.ungrouped, 3);
    }
}
