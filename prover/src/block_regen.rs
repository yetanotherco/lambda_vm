//! Regeneration on the block path (D-REGEN): the measurements it is planned
//! from (R0) and the shadow regenerator (R1).
//!
//! `LAMBDA_VM_BLOCK_REGEN_PROBE=1` prints two `BLOCK REGEN PROBE` lines and
//! changes nothing else:
//! - **first-touch**: per walked window, the memory bytes whose carried state
//!   the window reads ([`WalkedWindow::first_touch_bytes`]), summed over the
//!   windows the walker walks (the run's last is the finish's): the size of a
//!   per-window entry state, D-REGEN §2.4;
//! - **traces by class**: the main traces phase B starts from, split by how
//!   they could be rebuilt (D-REGEN §2.2): the streamed chunks (rebuilt from a
//!   window replay), class 2 (KECCAK_RND, ECDAS, CPU32, EQ, BYTEWISE, BRANCH:
//!   ordered lists, once streamed) and whole-run (the rest).
//!
//! `LAMBDA_VM_BLOCK_REGEN=shadow` runs the sequential regenerator beside phase
//! B and drops nothing (D-REGEN §8 R1). Phase A records, per streamed
//! instance, a [`Recipe`]: which chunk it is, the windows it spans, and the
//! digest of its packed trace taken on the generator that packed it. Phase B
//! re-executes the run from the start on threads of its own, walks it with
//! the regeneration walk ([`RegenBuilder`]), generates and packs every chunk
//! that has a recipe, checks its digest, and throws it away. It reports the
//! mismatches, its CPU, and when each instance would have been ready against
//! a plan that proves the regenerable instances last, in the order phase A
//! completed them ([`Plan`]). The proof is the one the knob's absence makes.
//!
//! [`WalkedWindow::first_touch_bytes`]: crate::tables::trace_builder::WalkedWindow::first_touch_bytes

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use executor::elf::Elf;
use executor::vm::execution::Executor;

use crate::Error;
use crate::tables::trace_builder::{ChunkJob, RegenBuilder, StreamSkip, StreamTable, Traces};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};

pub(crate) mod live;
#[cfg(test)]
mod live_tests;

type Table = stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>;

const GIB: f64 = (1u64 << 30) as f64;

/// `LAMBDA_VM_BLOCK_REGEN_PROBE=1`: the R0 probe lines (module docs). Off by
/// default; read once.
pub(crate) fn probe() -> bool {
    static PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PROBE
        .get_or_init(|| std::env::var("LAMBDA_VM_BLOCK_REGEN_PROBE").is_ok_and(|v| v.trim() == "1"))
}

/// Per walked window: its cycles and its first-touch bytes.
#[derive(Debug, Default)]
pub(crate) struct TouchCensus {
    windows: Vec<(usize, usize)>,
}

impl TouchCensus {
    /// The next window's cycles and first-touch bytes.
    pub(crate) fn push(&mut self, cycles: usize, bytes: usize) {
        self.windows.push((cycles, bytes));
    }

    /// The `BLOCK REGEN PROBE first-touch` line: the windows, the cycles they
    /// hold, the bytes summed, per cycle, and the windows' spread (window 0,
    /// which first-touches the program's image, also apart).
    pub(crate) fn line(&self) -> String {
        let n = self.windows.len();
        let cycles: usize = self.windows.iter().map(|w| w.0).sum();
        let total: usize = self.windows.iter().map(|w| w.1).sum();
        let mut sorted: Vec<usize> = self.windows.iter().map(|w| w.1).collect();
        sorted.sort_unstable();
        let at = |q: f64| {
            sorted
                .get(((n as f64 - 1.0) * q).round() as usize)
                .copied()
                .unwrap_or(0)
        };
        let (max_at, max) = self
            .windows
            .iter()
            .enumerate()
            .max_by_key(|(_, w)| w.1)
            .map_or((0, 0), |(i, w)| (i, w.1));
        let first = self.windows.first().map_or(0, |w| w.1);
        let rest_cycles = cycles - self.windows.first().map_or(0, |w| w.0);
        let per = |bytes: usize, cycles: usize| {
            if cycles == 0 {
                0.0
            } else {
                bytes as f64 / cycles as f64
            }
        };
        let m = |b: usize| b as f64 / 1e6;
        format!(
            "BLOCK REGEN PROBE first-touch: {n} windows · {:.2} M cycles · {:.2} M bytes · \
             {:.4} per cycle · without window 0 {:.4} per cycle · per window max {:.3} M \
             (window {max_at}) · p50 {:.3} M · p90 {:.3} M · window 0 {:.3} M",
            m(cycles),
            m(total),
            per(total, cycles),
            per(total - first, rest_cycles),
            m(max),
            m(at(0.5)),
            m(at(0.9)),
            m(first),
        )
    }
}

/// How a main trace could be rebuilt for phase B (D-REGEN §2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TraceClass {
    /// A chunk phase A streamed: rebuilt by replaying the windows it spans.
    Streamed,
    /// KECCAK_RND, ECDAS, CPU32, EQ, BYTEWISE, BRANCH: lists in execution
    /// order, rebuildable once streamed.
    Ordered,
    /// The rest: built from whole-run data.
    WholeRun,
}

/// One class's traces: instances, main cells, and their bytes as phase B
/// starts (packed, spilled, at 8 bytes a cell).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ClassBytes {
    pub(crate) instances: usize,
    pub(crate) cells: u64,
    pub(crate) packed: u64,
    pub(crate) spilled: u64,
    pub(crate) wide: u64,
    /// Dropped for regeneration: not held anywhere until phase B rebuilds it.
    pub(crate) dropped: u64,
}

impl ClassBytes {
    fn add(&mut self, trace: &Table) {
        let cells = (trace.num_rows() * trace.num_main_columns) as u64;
        self.instances += 1;
        self.cells += cells;
        if let Some(narrow) = trace.narrow_main() {
            self.packed += narrow.data().len() as u64;
        } else if let Some(slot) = trace.spilled_main() {
            self.spilled += slot.len() as u64;
        } else if let Some(slot) = trace.regen_main() {
            self.dropped += slot.len() as u64;
        } else {
            self.wide += cells * std::mem::size_of::<u64>() as u64;
        }
    }

    /// Its bytes, wherever they are (dropped ones included).
    pub(crate) fn bytes(&self) -> u64 {
        self.packed + self.spilled + self.wide + self.dropped
    }
}

/// The main traces phase B starts from, by [`TraceClass`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TraceClasses {
    pub(crate) streamed: ClassBytes,
    pub(crate) ordered: ClassBytes,
    pub(crate) whole_run: ClassBytes,
}

impl TraceClasses {
    /// Every trace in `traces` with rows, by class: the first `streamed.<t>`
    /// chunks of each streamed table are [`TraceClass::Streamed`].
    pub(crate) fn of(traces: &Traces, streamed: &StreamSkip) -> Self {
        let mut classes = Self::default();
        for (class, trace) in classify(traces, streamed) {
            if trace.num_rows() == 0 || trace.num_main_columns == 0 {
                continue;
            }
            match class {
                TraceClass::Streamed => classes.streamed.add(trace),
                TraceClass::Ordered => classes.ordered.add(trace),
                TraceClass::WholeRun => classes.whole_run.add(trace),
            }
        }
        classes
    }

    /// All classes together.
    pub(crate) fn total(&self) -> ClassBytes {
        let [a, b, c] = [self.streamed, self.ordered, self.whole_run];
        ClassBytes {
            instances: a.instances + b.instances + c.instances,
            cells: a.cells + b.cells + c.cells,
            packed: a.packed + b.packed + c.packed,
            spilled: a.spilled + b.spilled + c.spilled,
            wide: a.wide + b.wide + c.wide,
            dropped: a.dropped + b.dropped + c.dropped,
        }
    }

    /// The `BLOCK REGEN PROBE traces by class` line: per class its instances,
    /// cells and bytes (packed + spilled + 8 B/cell), and the streamed share
    /// of the bytes and of the cells.
    pub(crate) fn line(&self) -> String {
        let total = self.total();
        let share = |part: u64, whole: u64| {
            if whole == 0 {
                0.0
            } else {
                part as f64 / whole as f64
            }
        };
        let class = |name: &str, c: &ClassBytes| {
            format!(
                "{name} {} instances {:.3} G cells {:.2} GiB ({:.2} packed + {:.2} spilled + {:.2} at 8 B/cell)",
                c.instances,
                c.cells as f64 / 1e9,
                c.bytes() as f64 / GIB,
                c.packed as f64 / GIB,
                c.spilled as f64 / GIB,
                c.wide as f64 / GIB,
            )
        };
        format!(
            "BLOCK REGEN PROBE traces by class: {} · {} · {} · total {} instances {:.2} GiB · \
             streamed share {:.4} of bytes, {:.4} of cells · with class 2 {:.4} of bytes",
            class("streamed", &self.streamed),
            class("class-2", &self.ordered),
            class("whole-run", &self.whole_run),
            total.instances,
            total.bytes() as f64 / GIB,
            share(self.streamed.bytes(), total.bytes()),
            share(self.streamed.cells, total.cells),
            share(self.streamed.bytes() + self.ordered.bytes(), total.bytes()),
        )
    }
}

/// Every trace in `traces`, each with its [`TraceClass`]. Every field of
/// [`Traces`] that holds a trace is listed here.
fn classify<'a>(
    traces: &'a Traces,
    streamed: &StreamSkip,
) -> impl Iterator<Item = (TraceClass, &'a Table)> {
    use TraceClass::{Ordered, Streamed, WholeRun};
    let split = |list: &'a [Table], n: usize| {
        list.iter().enumerate().map(
            move |(i, t)| {
                if i < n { (Streamed, t) } else { (WholeRun, t) }
            },
        )
    };
    let all = |class: TraceClass, list: &'a [Table]| list.iter().map(move |t| (class, t));
    let one = |class: TraceClass, t: &'a Table| std::iter::once((class, t));
    let Traces {
        cpus,
        bitwise,
        lts,
        shifts,
        memws,
        memw_aligneds,
        loads,
        decode,
        muls,
        dvrms,
        pages,
        page_configs: _,
        register,
        public_output_bytes: _,
        branches,
        halt,
        commits,
        keccaks,
        keccak_rnds,
        keccak_rc,
        blake3,
        num_blake3_ops,
        ecsms,
        ecdases,
        hints,
        memw_registers,
        local_to_global,
        touched_memory_cells: _,
        eqs,
        bytewises,
        stores,
        cpu32s,
    } = traces;
    split(cpus, streamed.cpu)
        .chain(split(memw_registers, streamed.memw_register))
        .chain(split(memw_aligneds, streamed.memw_aligned))
        .chain(split(memws, streamed.memw))
        .chain(split(loads, streamed.load))
        .chain(split(lts, streamed.lt))
        .chain(split(shifts, streamed.shift))
        .chain(split(stores, streamed.store))
        .chain(all(Ordered, keccak_rnds))
        .chain(all(Ordered, ecdases))
        .chain(all(Ordered, cpu32s))
        .chain(all(Ordered, eqs))
        .chain(all(Ordered, bytewises))
        .chain(all(Ordered, branches))
        .chain(all(WholeRun, muls))
        .chain(all(WholeRun, dvrms))
        .chain(all(WholeRun, pages))
        .chain(all(WholeRun, commits))
        .chain(all(WholeRun, keccaks))
        .chain(all(WholeRun, ecsms))
        .chain(all(WholeRun, hints))
        .chain(one(WholeRun, bitwise))
        .chain(one(WholeRun, decode))
        .chain(one(WholeRun, register))
        .chain(one(WholeRun, halt))
        .chain(one(WholeRun, keccak_rc))
        // An unused BLAKE3 table still pads to rows; the proof leaves it out.
        .chain((*num_blake3_ops > 0).then_some((WholeRun, blake3)))
        .chain(one(WholeRun, local_to_global))
}

/// What phase B does with the regenerable instances
/// (`LAMBDA_VM_BLOCK_REGEN`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RegenMode {
    /// Unset or `off`: nothing is recorded or regenerated.
    Off,
    /// `shadow`: phase A records every streamed instance's recipe, and phase
    /// B regenerates each beside the prove, checks it and throws it away.
    Shadow,
    /// `auto`: a tier of the spill policy's `auto`. Phase A records every
    /// streamed instance's recipe; once the policy would move an instance off
    /// the host, a regenerable one is dropped instead of spilled (and the
    /// resident regenerable ones are dropped back until the host is under
    /// the target), and phase B rebuilds the dropped ones ([`live`]). A
    /// block that fits drops nothing and runs no regenerator. With the spill
    /// policy `off` (`LAMBDA_VM_BLOCK_SPILL=off`) `auto` still decides, with
    /// no store (no disk, I-REGEN §14 P1): once armed every regenerable
    /// instance is dropped and the rest stays on the host.
    Auto,
    /// `always`: every regenerable instance is dropped and rebuilt — the
    /// byte-identity test mode, not a policy.
    Always,
}

/// [`RegenMode`] from `LAMBDA_VM_BLOCK_REGEN`; any other value is refused.
pub(crate) fn regen_mode() -> Result<RegenMode, Error> {
    parse_regen_mode(std::env::var("LAMBDA_VM_BLOCK_REGEN").ok().as_deref())
}

fn parse_regen_mode(value: Option<&str>) -> Result<RegenMode, Error> {
    match value.map(str::trim) {
        None | Some("off") => Ok(RegenMode::Off),
        Some("shadow") => Ok(RegenMode::Shadow),
        Some("auto") => Ok(RegenMode::Auto),
        Some("always") => Ok(RegenMode::Always),
        Some(other) => Err(Error::Prover(format!(
            "LAMBDA_VM_BLOCK_REGEN must be `off`, `shadow`, `auto` or `always`, got `{other}`"
        ))),
    }
}

/// Generator threads of the regenerator: `LAMBDA_VM_BLOCK_REGEN_GENERATORS`
/// (1..=16), unset 3. D-REGEN §4 puts the rebuild at ≈ 2.7 threads of BIG
/// against p90's phase B.
pub(crate) fn regen_generators() -> usize {
    std::env::var("LAMBDA_VM_BLOCK_REGEN_GENERATORS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| (1..=16).contains(n))
        .unwrap_or(3)
}

/// A packed main trace's shape and digest ([`stark::narrow::NarrowMain::digest`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Packed {
    pub(crate) digest: [u64; 2],
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) bytes: usize,
}

impl Packed {
    pub(crate) fn of(narrow: &stark::narrow::NarrowMain) -> Self {
        Self {
            digest: narrow.digest(),
            rows: narrow.rows(),
            cols: narrow.cols(),
            bytes: narrow.data().len(),
        }
    }
}

/// A regenerated instance that is not the one phase A committed.
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

/// The check every regenerated instance passes before anything reads it: its
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

/// A streamed instance as phase A made it (D-REGEN §2.1): which chunk it is,
/// where its ops lie in the run, and its packed trace's digest.
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
    /// Its place in phase A's hand-out.
    pub(crate) order: usize,
    /// Its packed trace, once the thread that packed it digested it.
    pub(crate) packed: Option<Packed>,
}

impl Recipe {
    /// Its main cells, once packed.
    pub(crate) fn cells(&self) -> Option<u64> {
        self.packed.map(|p| (p.rows * p.cols) as u64)
    }
}

/// [`StreamTable::ALL`]'s position of `table`.
fn slot(table: StreamTable) -> usize {
    match table {
        StreamTable::Cpu => 0,
        StreamTable::MemwRegister => 1,
        StreamTable::MemwAligned => 2,
        StreamTable::Memw => 3,
        StreamTable::Load => 4,
        StreamTable::Lt => 5,
        StreamTable::Shift => 6,
        StreamTable::Store => 7,
    }
}

/// Phase A's recipes (shadow mode): fed by the accumulator (each window's list
/// lengths, then the chunks it handed out) and by the threads that pack the
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
    /// Digests for a chunk with no recipe.
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
    /// ([`StreamTable::ALL`] order), and the chunks its hand-out completed.
    pub(crate) fn window(&self, added: [usize; 8], jobs: &[ChunkJob]) {
        let mut r = self.lock();
        let w = r.windows;
        r.windows += 1;
        for (t, n) in added.into_iter().enumerate() {
            let end = r.ends[t].last().copied().unwrap_or(0) + n;
            r.ends[t].push(end);
        }
        for job in jobs {
            let t = slot(job.table);
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

    /// The packed trace of chunk `index` of `table`, as the thread that
    /// packed it holds it.
    pub(crate) fn packed(
        &self,
        table: StreamTable,
        index: usize,
        narrow: &stark::narrow::NarrowMain,
    ) {
        let packed = Packed::of(narrow);
        let mut r = self.lock();
        match r.recipes.get_mut(&(slot(table), index)) {
            Some(recipe) => recipe.packed = Some(packed),
            None => r.stray += 1,
        }
    }

    /// The rank of chunk `index` of `table`: its place in the hand-out.
    pub(crate) fn rank_of(&self, table: StreamTable, index: usize) -> Option<u64> {
        self.lock()
            .recipes
            .get(&(slot(table), index))
            .map(|r| r.order as u64)
    }

    /// The recipes recorded so far.
    pub(crate) fn len(&self) -> usize {
        self.lock().recipes.len()
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

/// An instance the shadow regenerated and checked.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Regenerated {
    pub(crate) table: StreamTable,
    pub(crate) index: usize,
    /// Seconds from the regenerator's start to the instance checked.
    pub(crate) ready: f64,
    pub(crate) matched: bool,
    pub(crate) bytes: usize,
}

/// What the shadow regenerator did.
#[derive(Debug)]
pub(crate) struct ShadowReport {
    pub(crate) started: Instant,
    /// Recipes it was given, and of those with no digest (not regenerated).
    pub(crate) recipes: usize,
    pub(crate) undigested: usize,
    pub(crate) regenerated: Vec<Regenerated>,
    pub(crate) mismatches: Vec<RegenMismatch>,
    /// Instances whose generation failed (a panic caught, a trace that
    /// would not pack).
    pub(crate) failures: Vec<String>,
    /// Chunks cut that phase A did not stream (the finish's), not generated.
    pub(crate) beyond: usize,
    pub(crate) windows: usize,
    pub(crate) cycles: usize,
    /// Seconds from its start to its last instance checked, or to its stop.
    pub(crate) wall: f64,
    /// Thread CPU seconds: the executor, the walker (with the slicer), the
    /// generators summed (Linux only).
    pub(crate) cpu: [Option<f64>; 3],
    pub(crate) generators: usize,
    /// Why it stopped before the end, if it did.
    pub(crate) error: Option<String>,
    /// The walker's carried memory state, in bytes, at its end.
    pub(crate) state_bytes: usize,
}

impl ShadowReport {
    fn new(started: Instant, recipes: usize, undigested: usize, generators: usize) -> Self {
        Self {
            started,
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

    /// The `BLOCK REGEN shadow` line.
    pub(crate) fn line(&self) -> String {
        let n = self.regenerated.len();
        let missing = (self.recipes - self.undigested).saturating_sub(n + self.failures.len());
        let bytes: usize = self.regenerated.iter().map(|r| r.bytes).sum();
        let secs = |s: Option<f64>| s.map_or("n/a".to_string(), |s| format!("{s:.1}"));
        let (first, last) = self
            .regenerated
            .iter()
            .fold((f64::INFINITY, 0.0f64), |(lo, hi), r| {
                (lo.min(r.ready), hi.max(r.ready))
            });
        format!(
            "BLOCK REGEN shadow: {n} of {} instances regenerated · {} mismatches · {} failed · \
             {missing} missing · {} undigested · {} not streamed (skipped) · {} windows walked \
             ({:.2} M cycles) · {:.2} GiB packed · ready from {:.2} s to {:.2} s · wall {:.2} s · \
             CPU exec {} · walk {} · {} generators {} = {} s · walk state {:.2} GiB{}",
            self.recipes,
            self.mismatches.len(),
            self.failures.len(),
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
/// recipe is generated, packed and checked on one of `generators` threads,
/// then dropped. Stops when every recipe is checked, on an error (reported),
/// or when `stop` is set. Never panics: a panic in generation counts as that
/// instance's failure.
pub(crate) fn run_shadow(
    program: &Elf,
    private_input: &[u8],
    mut builder: RegenBuilder,
    window: usize,
    recipes: &[Recipe],
    generators: usize,
    stop: &AtomicBool,
) -> ShadowReport {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::mpsc;
    let started = Instant::now();
    let wanted: BTreeMap<(usize, usize), Packed> = recipes
        .iter()
        .filter_map(|r| Some(((slot(r.table), r.index), r.packed?)))
        .collect();
    let mut report = ShadowReport::new(
        started,
        recipes.len(),
        recipes.len() - wanted.len(),
        generators,
    );
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
                                let mut trace = job.generate().trace;
                                trace
                                    .pack_main_narrow()
                                    .then(|| trace.narrow_main().map(Packed::of))
                            }));
                            let ready = started.elapsed().as_secs_f64();
                            let got = match got {
                                Ok(Some(Some(got))) => got,
                                Ok(_) => {
                                    failures.lock().unwrap_or_else(|e| e.into_inner()).push(
                                        format!("{table:?}[{index}]: the trace did not pack"),
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
                match wanted.get(&(slot(job.table), job.index)) {
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

/// When the regenerated instances would have been needed under the plan
/// D-REGEN §2.3 gives phase B: every other instance first (resident and
/// spilled, today's order), then the regenerable ones in the order phase A
/// completed them (by last window, then hand-out order). Phase B is taken to
/// consume main cells at its measured average rate, so the instance after
/// `C` cells starts at `prove × C / total`. With a serial consumer the stall
/// the plan would pay is the largest lateness (ready − needed), at least 0.
/// An instance not regenerated, or regenerated wrong, is missing, and with
/// any missing there is no would-wait.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Plan {
    pub(crate) would_wait: f64,
    /// The latest instance, and how late it was (negative: early).
    pub(crate) worst: Option<(StreamTable, usize, f64)>,
    /// Instances ready after their slot.
    pub(crate) late: usize,
    /// Instances in the plan, and those never regenerated.
    pub(crate) planned: usize,
    pub(crate) missing: usize,
    /// The other instances' cells, and all cells.
    pub(crate) other_cells: u64,
    pub(crate) total_cells: u64,
}

/// [`Plan`] for `recipes` regenerated as `report` says, the prove starting
/// `prove_offset` seconds after the regenerator and lasting `prove` seconds,
/// over `total_cells` main cells.
pub(crate) fn plan(
    recipes: &[Recipe],
    report: &ShadowReport,
    prove_offset: f64,
    prove: f64,
    total_cells: u64,
) -> Plan {
    // A mismatched instance would be refused, so it is never ready.
    let ready: BTreeMap<(usize, usize), f64> = report
        .regenerated
        .iter()
        .filter(|r| r.matched)
        .map(|r| ((slot(r.table), r.index), r.ready - prove_offset))
        .collect();
    let mut order: Vec<&Recipe> = recipes.iter().filter(|r| r.packed.is_some()).collect();
    order.sort_by_key(|r| (r.last_window, r.order));
    let regen_cells: u64 = order.iter().filter_map(|r| r.cells()).sum();
    let other_cells = total_cells.saturating_sub(regen_cells);
    let total = total_cells.max(1) as f64;
    let mut before = other_cells;
    let (mut worst, mut late, mut missing) = (None::<(StreamTable, usize, f64)>, 0, 0);
    for recipe in &order {
        let needed = prove * before as f64 / total;
        before += recipe.cells().unwrap_or(0);
        let Some(&ready) = ready.get(&(slot(recipe.table), recipe.index)) else {
            missing += 1;
            continue;
        };
        let lateness = ready - needed;
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
        planned: order.len(),
        missing,
        other_cells,
        total_cells,
    }
}

impl Plan {
    /// The `BLOCK REGEN plan` line, with the regenerator's CPU over the prove.
    pub(crate) fn line(&self, report: &ShadowReport, prove: f64, prove_offset: f64) -> String {
        let cores = report
            .cpu_total()
            .filter(|_| prove > 0.0)
            .map_or("n/a".to_string(), |cpu| format!("{:.2}", cpu / prove));
        format!(
            "BLOCK REGEN plan: {} regenerable instances last, in completion order, after {:.3} G \
             of {:.3} G cells · would-wait {}{} · late {} · missing {} · regen CPU {} s = {cores} \
             cores over the prove ({prove:.2} s, starting {prove_offset:.2} s after the regenerator)",
            self.planned,
            self.other_cells as f64 / 1e9,
            self.total_cells as f64 / 1e9,
            if self.missing > 0 {
                "n/a".to_string()
            } else {
                format!("{:.3} s", self.would_wait)
            },
            self.worst.map_or(String::new(), |(t, i, w)| format!(
                " (latest {t:?}[{i}] at {w:+.3} s)"
            )),
            self.late,
            self.missing,
            report
                .cpu_total()
                .map_or("n/a".to_string(), |c| format!("{c:.1}")),
        )
    }
}

/// Phase A's recorder under `mode`: one when the shadow is wanted and the
/// block can feed it (`streamed`: phase A streams its windows on the walker's
/// own thread and packs each streamed instance), else none, saying why.
pub(crate) fn shadow_recorder(mode: RegenMode, streamed: bool) -> Option<Recorder> {
    match (mode, streamed) {
        (RegenMode::Off | RegenMode::Auto | RegenMode::Always, _) => None,
        (RegenMode::Shadow, true) => Some(Recorder::new()),
        (RegenMode::Shadow, false) => {
            eprintln!(
                "BLOCK REGEN: shadow wanted, but phase A does not stream and pack its instances \
                 (LAMBDA_VM_BLOCK_STREAM / LAMBDA_VM_BLOCK_NARROW): off"
            );
            None
        }
    }
}

/// Phase A's regeneration: the shadow's recorder (digests at pack), or live
/// regeneration (recipes, drops under the policy).
pub(crate) enum PhaseARegen {
    Shadow(Box<Recorder>),
    Live(Box<live::LiveRegen>),
}

impl PhaseARegen {
    /// The regeneration `mode` asks for, when phase A can feed it
    /// (`streamed`); none otherwise (saying why).
    pub(crate) fn new(mode: RegenMode, streamed: bool) -> Option<Self> {
        match mode {
            RegenMode::Off => None,
            RegenMode::Shadow => shadow_recorder(mode, streamed).map(|r| Self::Shadow(Box::new(r))),
            RegenMode::Auto | RegenMode::Always => {
                live::LiveRegen::new(mode, streamed).map(|live| Self::Live(Box::new(live)))
            }
        }
    }

    pub(crate) fn recorder(&self) -> &Recorder {
        match self {
            Self::Shadow(recorder) => recorder,
            Self::Live(live) => live.recorder(),
        }
    }

    /// Live regeneration, when it is.
    pub(crate) fn live(&self) -> Option<&live::LiveRegen> {
        match self {
            Self::Shadow(_) => None,
            Self::Live(live) => Some(live),
        }
    }

    /// Whether the thread that packs a streamed instance digests it (the
    /// shadow compares against it; live regeneration digests at drop).
    pub(crate) fn digests_at_pack(&self) -> bool {
        matches!(self, Self::Shadow(_))
    }
}

/// The shadow regenerator on a thread of `scope`.
pub(crate) struct ShadowHandle<'scope> {
    started: Instant,
    recipes: usize,
    handle: std::io::Result<std::thread::ScopedJoinHandle<'scope, ShadowReport>>,
}

impl ShadowHandle<'_> {
    /// Its report, once it is done; a thread that did not start or that
    /// panicked is a report with the error.
    pub(crate) fn join(self) -> ShadowReport {
        let failed = |error: String| {
            let mut report = ShadowReport::new(self.started, self.recipes, 0, 0);
            report.error = Some(error);
            report
        };
        match self.handle {
            Ok(handle) => handle
                .join()
                .unwrap_or_else(|_| failed("the regenerator panicked".to_string())),
            Err(e) => failed(format!("the regenerator did not start: {e}")),
        }
    }
}

/// [`run_shadow`] over the block's run on a thread of `scope` (`regen-walk`):
/// windows of `max_rows.cpu` cycles, as phase A's executor ran,
/// [`regen_generators`] generators. `stop` ends it early.
pub(crate) fn spawn_shadow<'scope>(
    scope: &'scope std::thread::Scope<'scope, '_>,
    program: &'scope Elf,
    private_input: &'scope [u8],
    max_rows: &'scope crate::tables::MaxRowsConfig,
    recipes: &'scope [Recipe],
    stop: &'scope AtomicBool,
) -> ShadowHandle<'scope> {
    let started = Instant::now();
    let handle = std::thread::Builder::new()
        .name("regen-walk".to_string())
        .spawn_scoped(scope, move || {
            // The walker's CPU includes building its start state (the image,
            // the decode table).
            let cpu0 = thread_cpu_secs();
            match RegenBuilder::new(program, private_input, max_rows) {
                Ok(builder) => {
                    let setup = cpu_since(cpu0);
                    let mut report = run_shadow(
                        program,
                        private_input,
                        builder,
                        max_rows.cpu,
                        recipes,
                        regen_generators(),
                        stop,
                    );
                    report.cpu[1] = report.cpu[1].zip(setup).map(|(walk, setup)| walk + setup);
                    report
                }
                Err(e) => {
                    let mut report = ShadowReport::new(started, recipes.len(), 0, 0);
                    report.error = Some(format!("the regenerator: {e}"));
                    report
                }
            }
        });
    ShadowHandle {
        started,
        recipes: recipes.len(),
        handle,
    }
}

/// The `BLOCK REGEN` lines after the prove: what phase A recorded, what the
/// shadow regenerated (each mismatch and failure, the first 20), and the plan
/// over `total_cells` main cells, the prove lasting `prove` seconds from
/// `prove_offset` seconds after the regenerator started.
pub(crate) fn report_shadow(
    report: &ShadowReport,
    recipes: &[Recipe],
    stray: usize,
    streamed_cells: u64,
    total_cells: u64,
    prove: f64,
    prove_offset: f64,
) {
    let digested = recipes.iter().filter(|r| r.packed.is_some()).count();
    let cells: u64 = recipes.iter().filter_map(Recipe::cells).sum();
    let spans: Vec<usize> = recipes
        .iter()
        .filter_map(|r| Some(r.last_window + 1 - r.first?.0))
        .collect();
    eprintln!(
        "BLOCK REGEN recorded: {} streamed instances · {digested} digested · {stray} stray digests · \
         {:.3} G cells (the streamed class {:.3} G) · windows spanned mean {:.2} max {}",
        recipes.len(),
        cells as f64 / 1e9,
        streamed_cells as f64 / 1e9,
        spans.iter().sum::<usize>() as f64 / spans.len().max(1) as f64,
        spans.iter().max().copied().unwrap_or(0),
    );
    eprintln!("{}", report.line());
    for mismatch in report.mismatches.iter().take(20) {
        eprintln!("BLOCK REGEN MISMATCH {mismatch}");
    }
    for failure in report.failures.iter().take(20) {
        eprintln!("BLOCK REGEN FAILED {failure}");
    }
    let plan = plan(recipes, report, prove_offset, prove, total_cells);
    eprintln!("{}", plan.line(report, prove, prove_offset));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::MaxRowsConfig;

    /// The classes of a streamed build: the streamed chunks are exactly the
    /// ones the stream handed out, class 2 is the six ordered tables, and the
    /// three cover every trace with rows.
    #[test]
    fn the_traces_split_into_their_classes() {
        let opts = crate::lfm::proof::block_base_options();
        let max_rows = MaxRowsConfig {
            keccak_rnd: 48,
            ..MaxRowsConfig::small()
        };
        for name in ["all_instructions_64", "test_keccak_multi"] {
            let program = executor::elf::Elf::load(&crate::test_utils::asm_elf_bytes(name))
                .expect("the ELF loads");
            let (traces, streamed) =
                crate::block::stream_skip_for_test(&program, &opts, &max_rows).expect("streamed");
            let classes = TraceClasses::of(&traces, &streamed);
            let handed_out = streamed.cpu
                + streamed.memw_register
                + streamed.memw_aligned
                + streamed.memw
                + streamed.load
                + streamed.lt
                + streamed.shift
                + streamed.store;
            assert!(handed_out > 0, "{name}: nothing was streamed");
            assert_eq!(classes.streamed.instances, handed_out, "{name}");
            let rows = |list: &[Table]| list.iter().filter(|t| t.num_rows() > 0).count();
            assert_eq!(
                classes.ordered.instances,
                rows(&traces.keccak_rnds)
                    + rows(&traces.ecdases)
                    + rows(&traces.cpu32s)
                    + rows(&traces.eqs)
                    + rows(&traces.bytewises)
                    + rows(&traces.branches),
                "{name}"
            );
            let total = classes.total();
            assert!(
                classes.whole_run.instances > 0 && total.bytes() > 0,
                "{name}"
            );
            assert_eq!(
                total.instances,
                classes.streamed.instances
                    + classes.ordered.instances
                    + classes.whole_run.instances
            );
            let line = classes.line();
            assert!(
                line.starts_with("BLOCK REGEN PROBE traces by class: streamed "),
                "{line}"
            );
            eprintln!("{line}");
        }
    }

    /// The census line: the sum, the rate per cycle with and without window
    /// 0, and the largest window.
    #[test]
    fn the_touch_census_sums_its_windows() {
        let mut census = TouchCensus::default();
        for bytes in [500_000, 100_000, 300_000, 200_000] {
            census.push(1_000_000, bytes);
        }
        let line = census.line();
        assert!(line.contains("4 windows · 4.00 M cycles"), "{line}");
        assert!(line.contains("· 1.10 M bytes"), "{line}");
        assert!(line.contains("0.2750 per cycle"), "{line}");
        assert!(line.contains("without window 0 0.2000 per cycle"), "{line}");
        assert!(line.contains("max 0.500 M (window 0)"), "{line}");
        assert!(line.contains("p50 0.300 M"), "{line}");
        assert!(TouchCensus::default().line().contains("0 windows"));
    }
}

#[cfg(test)]
mod shadow_tests {
    use super::*;
    use crate::tables::MaxRowsConfig;

    const PROGRAMS: [&str; 4] = [
        "all_instructions_64",
        "test_keccak_multi",
        "test_blake3",
        "test_ecsm",
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

    /// The trace phase B proves for `recipe`'s instance.
    fn instance<'a>(traces: &'a Traces, recipe: &Recipe) -> &'a Table {
        let list = match recipe.table {
            StreamTable::Cpu => &traces.cpus,
            StreamTable::MemwRegister => &traces.memw_registers,
            StreamTable::MemwAligned => &traces.memw_aligneds,
            StreamTable::Memw => &traces.memws,
            StreamTable::Load => &traces.loads,
            StreamTable::Lt => &traces.lts,
            StreamTable::Shift => &traces.shifts,
            StreamTable::Store => &traces.stores,
        };
        &list[recipe.index]
    }

    fn shadow(
        program: &Elf,
        builder: RegenBuilder,
        window: usize,
        recipes: &[Recipe],
    ) -> ShadowReport {
        run_shadow(
            program,
            &[],
            builder,
            window,
            recipes,
            2,
            &AtomicBool::new(false),
        )
    }

    /// `LAMBDA_VM_BLOCK_REGEN`: unset and `off` are off, `shadow` is the shadow,
    /// anything else is refused (an error, not a panic).
    #[test]
    fn the_regen_mode_reads_its_knob() {
        assert_eq!(parse_regen_mode(None).unwrap(), RegenMode::Off);
        assert_eq!(parse_regen_mode(Some(" off ")).unwrap(), RegenMode::Off);
        assert_eq!(parse_regen_mode(Some("shadow")).unwrap(), RegenMode::Shadow);
        assert_eq!(parse_regen_mode(Some("auto")).unwrap(), RegenMode::Auto);
        assert_eq!(parse_regen_mode(Some("always")).unwrap(), RegenMode::Always);
        for bad in ["", "on", "Always", "Shadow", "1"] {
            assert!(parse_regen_mode(Some(bad)).is_err(), "{bad:?}");
        }
    }

    /// ★ Phase A records a recipe for every streamed instance, digested where
    /// it was packed (generators ahead of the committers, or the committers
    /// themselves), whose digest is the trace phase B proves; and the shadow
    /// regenerates every one of them from a fresh execution, at windows of 1, 7
    /// and 33 cycles, with no mismatch.
    #[test]
    fn the_shadow_regenerates_every_streamed_instance() {
        let _one = super::live_tests::one_at_a_time();
        let opts = crate::lfm::proof::block_base_options();
        let max_rows = max_rows();
        let mut regenerated = 0;
        for name in PROGRAMS {
            let program = program(name);
            for stream in [
                crate::block::stream_config(2, 3, true),
                crate::block::stream_config(3, 0, false),
            ] {
                let (traces, streamed, recipes, stray) =
                    crate::block::stream_recorded_for_test(&program, &opts, &max_rows, stream)
                        .expect("phase A");
                let handed_out = streamed.cpu
                    + streamed.memw_register
                    + streamed.memw_aligned
                    + streamed.memw
                    + streamed.load
                    + streamed.lt
                    + streamed.shift
                    + streamed.store;
                assert_eq!(recipes.len(), handed_out, "{name}: a recipe per instance");
                assert_eq!(stray, 0, "{name}: digests with no recipe");
                for recipe in &recipes {
                    let packed = recipe.packed.expect("every recipe digested");
                    let trace = instance(&traces, recipe);
                    let narrow = trace.narrow_main().expect("phase B holds it packed");
                    assert_eq!(Packed::of(narrow), packed, "{name}: {recipe:?}");
                    let (first, _) = recipe.first.expect("the windows were counted");
                    assert!(first <= recipe.last_window, "{name}: {recipe:?}");
                }
                let classes = TraceClasses::of(&traces, &streamed);
                assert_eq!(
                    recipes.iter().filter_map(Recipe::cells).sum::<u64>(),
                    classes.streamed.cells,
                    "{name}"
                );
                for window in [1, 7, 33] {
                    let builder =
                        RegenBuilder::new(&program, &[], &max_rows).expect("the regenerator");
                    let report = shadow(&program, builder, window, &recipes);
                    assert!(report.error.is_none(), "{name}: {}", report.line());
                    assert!(report.mismatches.is_empty(), "{name}: {}", report.line());
                    assert!(report.failures.is_empty(), "{name}: {}", report.line());
                    assert_eq!(report.regenerated.len(), recipes.len(), "{}", report.line());
                    assert!(report.regenerated.iter().all(|r| r.matched));
                    let plan = plan(&recipes, &report, 0.0, 10.0, classes.total().cells);
                    assert_eq!((plan.planned, plan.missing), (recipes.len(), 0));
                    if window == 7 {
                        report_shadow(
                            &report,
                            &recipes,
                            stray,
                            classes.streamed.cells,
                            classes.total().cells,
                            10.0,
                            0.0,
                        );
                    }
                    regenerated += report.regenerated.len();
                }
            }
        }
        assert!(regenerated > 0, "nothing was regenerated");
    }

    /// ★ The check is load-bearing: a regenerator whose slicer drops one op, or
    /// a recipe whose digest is off by a bit, is reported as a mismatch (an
    /// entry in the report, never a panic), and the plan counts the instance
    /// as missing.
    #[test]
    fn a_wrong_regeneration_is_a_mismatch() {
        let _one = super::live_tests::one_at_a_time();
        let opts = crate::lfm::proof::block_base_options();
        let max_rows = max_rows();
        let program = program("all_instructions_64");
        let (traces, streamed, recipes, _) = crate::block::stream_recorded_for_test(
            &program,
            &opts,
            &max_rows,
            crate::block::stream_config(2, 3, true),
        )
        .expect("phase A");
        let total = TraceClasses::of(&traces, &streamed).total().cells;
        for table in [StreamTable::Cpu, StreamTable::MemwRegister] {
            assert!(recipes.iter().any(|r| r.table == table && r.index == 0));
            let builder = RegenBuilder::new(&program, &[], &max_rows)
                .expect("the regenerator")
                .drop_one_op(table);
            let report = shadow(&program, builder, 7, &recipes);
            assert!(
                report
                    .mismatches
                    .iter()
                    .any(|m| m.table == table && m.index == 0),
                "{table:?}: {}",
                report.line()
            );
            assert!(
                report
                    .line()
                    .contains(&format!("{} mismatches", report.mismatches.len()))
            );
            assert!(plan(&recipes, &report, 0.0, 10.0, total).missing > 0);
        }
        let mut bent = recipes.clone();
        if let Some(packed) = bent[0].packed.as_mut() {
            packed.digest[1] ^= 1;
        }
        let builder = RegenBuilder::new(&program, &[], &max_rows).expect("the regenerator");
        let report = shadow(&program, builder, 7, &bent);
        assert_eq!(report.mismatches.len(), 1, "{}", report.line());
        let mismatch = &report.mismatches[0];
        assert_eq!(
            (mismatch.table, mismatch.index),
            (bent[0].table, bent[0].index)
        );
        assert!(format!("{mismatch}").contains("digest"));
    }

    /// Stopped before it starts, the shadow regenerates nothing and says why.
    #[test]
    fn a_stopped_shadow_says_so() {
        let _one = super::live_tests::one_at_a_time();
        let opts = crate::lfm::proof::block_base_options();
        let max_rows = max_rows();
        let program = program("test_keccak_multi");
        let (_, _, recipes, _) = crate::block::stream_recorded_for_test(
            &program,
            &opts,
            &max_rows,
            crate::block::stream_config(2, 3, true),
        )
        .expect("phase A");
        let builder = RegenBuilder::new(&program, &[], &max_rows).expect("the regenerator");
        let report = run_shadow(
            &program,
            &[],
            builder,
            7,
            &recipes,
            2,
            &AtomicBool::new(true),
        );
        assert!(report.regenerated.is_empty());
        assert_eq!(report.error.as_deref(), Some("stopped by the prove"));
    }

    fn recipe(table: StreamTable, index: usize, last_window: usize, order: usize) -> Recipe {
        Recipe {
            table,
            index,
            ops: 4,
            first: Some((last_window, 0)),
            last_window,
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

    /// The plan: 1000 cells, 600 of them the other instances', a prove of
    /// 10 s. The regenerable instances (100 cells each) go last by completion
    /// window (then hand-out order), so their slots start at 6, 7, 8, 9 s;
    /// lateness is ready − slot on the prove's clock, the would-wait the
    /// largest of them, at least 0.
    #[test]
    fn the_plan_puts_the_regenerable_instances_last_in_completion_order() {
        use StreamTable::{Cpu, Memw};
        let recipes = [
            recipe(Cpu, 0, 1, 0),
            recipe(Memw, 0, 1, 1),
            recipe(Cpu, 1, 3, 2),
            recipe(Memw, 1, 2, 3),
        ];
        let mut report = ShadowReport::new(Instant::now(), 4, 0, 2);
        // Lateness at offset 0 (offset 1): CPU[0] slot 6 +0.5 (−0.5), MEMW[0]
        // slot 7 −4 (−5), MEMW[1] slot 8 +1 (0), CPU[1] slot 9 +2 (+1).
        report.regenerated = vec![
            ready(Cpu, 0, 6.5, true),
            ready(Memw, 0, 3.0, true),
            ready(Memw, 1, 9.0, true),
            ready(Cpu, 1, 11.0, true),
        ];
        let p = plan(&recipes, &report, 0.0, 10.0, 1000);
        assert_eq!((p.planned, p.missing, p.late), (4, 0, 3));
        assert_eq!(p.other_cells, 600);
        assert_eq!(p.worst, Some((Cpu, 1, 2.0)));
        assert!((p.would_wait - 2.0).abs() < 1e-9);
        let p = plan(&recipes, &report, 1.0, 10.0, 1000);
        assert_eq!(p.late, 1);
        assert!((p.would_wait - 1.0).abs() < 1e-9);
        // Everything early: no wait.
        let p = plan(&recipes, &report, 5.0, 10.0, 1000);
        assert_eq!((p.late, p.would_wait), (0, 0.0));
        // A mismatched instance is missing, and the line says n/a.
        report.regenerated[1].matched = false;
        let p = plan(&recipes, &report, 0.0, 10.0, 1000);
        assert_eq!(p.missing, 1);
        assert!(p.line(&report, 10.0, 0.0).contains("would-wait n/a"));
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
}
