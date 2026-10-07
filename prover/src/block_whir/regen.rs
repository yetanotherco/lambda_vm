//! Regeneration on the WHIR block (D-WHIR-NODISK, #1013's D-REGEN): the shadow
//! regenerator, and live regeneration.
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
//!
//! `auto` and `always` are live regeneration (#1013's R2, its no-disk P1):
//! phase A drops a streamed chunk instead of keeping or spilling it once the
//! spill's policy would move a table off the host (`always`: every one), and
//! phase B's regenerator ([`run_live`]) deposits each into its slot, paced by
//! the groups that take them ([`stark::multilinear_block::BlockRegen`]).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use stark::regen::{RegenError, RegenProducer, RegenSlot, RegenWindow};

use crate::Error;
use crate::tables::gpack::TraceForm;
use crate::tables::trace_builder::{ChunkJob, RegenBuilder, RegenFamily, RestRegen, StreamTable};

const GIB: f64 = (1u64 << 30) as f64;

/// What phase B does with the streamed chunks (`LAMBDA_VM_BLOCK_REGEN`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegenMode {
    /// `off` (and unset): nothing is recorded or regenerated.
    Off,
    /// `shadow`: phase A records every streamed chunk's recipe, and phase B
    /// regenerates each beside the prove, checks it and throws it away.
    Shadow,
    /// `auto`: a tier of the spill's policy. Once it would move a table off
    /// the host, a streamed chunk is dropped instead (and the parked ones are
    /// dropped back), and phase B rebuilds the dropped ones. With the spill
    /// policy `off` it still decides as `auto` does, with no store: no disk.
    /// A block that never arms drops nothing.
    Auto,
    /// `always`: every streamed chunk is dropped and rebuilt — the
    /// byte-identity test mode, not a policy.
    Always,
}

impl RegenMode {
    /// Whether phase A drops chunks for phase B to rebuild.
    pub fn live(self) -> bool {
        matches!(self, Self::Auto | Self::Always)
    }
}

/// [`RegenMode`] from `LAMBDA_VM_BLOCK_REGEN` alone (unset: `off`); any other
/// value is refused (an error, never a panic). The production default couples
/// it to the spill knob ([`production_regen`]).
pub(crate) fn regen_mode() -> Result<RegenMode, Error> {
    parse_regen_mode(std::env::var("LAMBDA_VM_BLOCK_REGEN").ok().as_deref())
}

/// [`crate::block_whir::BlockOptions::production`]'s regeneration
/// ([`production_regen_from`] of the two knobs).
pub(crate) fn production_regen() -> Option<RegenMode> {
    production_regen_from(
        std::env::var("LAMBDA_VM_BLOCK_REGEN").ok().as_deref(),
        std::env::var("LAMBDA_VM_BLOCK_SPILL").ok().as_deref(),
    )
}

/// The production default, the knobs coupled as #1013's 1a9853790: both unset,
/// regeneration is `auto` and the spill `off` ([`super::production_spill`]): no
/// disk, whenever `auto` arms the tables phase B can build again (the streamed
/// chunks, KECCAK_RND's and LT's) are dropped and rebuilt in phase B, and the
/// rest stays. A knob that is set keeps the meaning it had: a set `regen` is
/// read when the prove starts (`None`, so a value it does not know refuses the
/// prove), and an unset `regen` beside a set `spill` is `off`, as before.
pub(crate) fn production_regen_from(regen: Option<&str>, spill: Option<&str>) -> Option<RegenMode> {
    match (regen, spill) {
        (Some(_), _) => None,
        (None, None) => Some(RegenMode::Auto),
        (None, Some(_)) => Some(RegenMode::Off),
    }
}

/// The `BLOCK REGEN mode` line: the mode a prove runs, the two knobs as set
/// (the production default is `auto` with the spill `off` when both are
/// unset), and `no disk` when live regeneration runs with the spill off.
pub(crate) fn regen_mode_line(
    mode: RegenMode,
    no_disk: bool,
    regen: Option<&str>,
    spill: Option<&str>,
) -> String {
    let knob = |name: &str, value: Option<&str>| match value {
        Some(v) => format!("{name}={}", v.trim()),
        None => format!("{name} unset"),
    };
    format!(
        "BLOCK REGEN mode: {mode:?} · {} · {}{}",
        knob("LAMBDA_VM_BLOCK_REGEN", regen),
        knob("LAMBDA_VM_BLOCK_SPILL", spill),
        if no_disk { " · no disk" } else { "" }
    )
}

/// [`regen_mode`] from the knob's `value`.
pub(crate) fn parse_regen_mode(value: Option<&str>) -> Result<RegenMode, Error> {
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

/// The regenerator's window: the bytes it may hold deposited ahead of the
/// groups that take them, counted from the window's frontier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum AheadPolicy {
    /// `LAMBDA_VM_BLOCK_REGEN_AHEAD_GIB`: these bytes, the whole prove.
    Fixed(u64),
    /// Unset: [`adaptive_ahead`], read by a pacer beside the regenerator.
    Adaptive,
}

impl AheadPolicy {
    /// The window before the pacer's first reading.
    pub(crate) fn initial(self) -> u64 {
        match self {
            AheadPolicy::Fixed(bytes) => bytes,
            AheadPolicy::Adaptive => AHEAD_FLOOR,
        }
    }
}

/// The adaptive window's floor: the window it replaced (the spill read-back's).
pub(crate) const AHEAD_FLOOR: u64 = 4 << 30;
/// Its cap: the smallest window with which phase B's takers did not wait at
/// T200 (RYZEN 052: 0.0 s at 16 and 32 GiB, 3.02 s at 4).
pub(crate) const AHEAD_CAP: u64 = 16 << 30;
/// What it leaves below the run's peak.
pub(crate) const AHEAD_MARGIN: u64 = 4 << 30;

/// [`AheadPolicy`] from `LAMBDA_VM_BLOCK_REGEN_AHEAD_GIB`.
pub(crate) fn ahead_policy() -> AheadPolicy {
    ahead_policy_from(
        std::env::var("LAMBDA_VM_BLOCK_REGEN_AHEAD_GIB")
            .ok()
            .as_deref(),
    )
}

/// [`ahead_policy`] from the knob's `value`: a size in GiB fixes the window;
/// unset or unreadable, it is adaptive.
pub(crate) fn ahead_policy_from(value: Option<&str>) -> AheadPolicy {
    value
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|g| g.is_finite() && *g >= 0.0)
        .map_or(AheadPolicy::Adaptive, |g| {
            AheadPolicy::Fixed((g * GIB) as u64)
        })
}

/// The adaptive window: what the run's peak resident set `hwm` leaves above
/// its resident set now (`rss`, the window's own `parked` deposits left out),
/// less [`AHEAD_MARGIN`], clamped to [[`AHEAD_FLOOR`], [`AHEAD_CAP`]]. Above
/// the floor the window holds only memory the run has already reached, so it
/// never raises the run's peak; phase A sets that peak, and phase B holds far
/// less (RYZEN 049 at T200: 72.5 GiB against 15–21).
pub(crate) fn adaptive_ahead(hwm: u64, rss: u64, parked: u64) -> u64 {
    let base = rss.saturating_sub(parked);
    hwm.saturating_sub(base)
        .saturating_sub(AHEAD_MARGIN)
        .clamp(AHEAD_FLOOR, AHEAD_CAP)
}

/// The process's peak and current resident sets (`VmHWM`, `VmRSS`), where
/// `/proc/self/status` gives them.
fn resident() -> Option<(u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kib = |key: &str| {
        status
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            .map(|k| k << 10)
    };
    Some((kib("VmHWM:")?, kib("VmRSS:")?))
}

/// How often the pacer reads the host.
const PACE: std::time::Duration = std::time::Duration::from_millis(250);

/// The least the pacer moves the window by. A smaller move is the window's own
/// churn, not the host's: a chunk taken or deposited (≈ 0.1 GiB at p90), whose
/// pages the allocator keeps (RYZEN 057 at p90 with no step: 109 shrinks and 28
/// grows in 455 readings).
pub(crate) const PACE_STEP: u64 = 1 << 30;

/// The pacer's next window from the current one (`now`, whole steps) and the
/// reading (`reading`, [`adaptive_ahead`]): the reading rounded down to whole
/// steps, taken only once the reading is a step or more away from the window
/// either way. So the window stays under a step above the reading (the run's
/// peak less [`AHEAD_MARGIN`] is still never passed, a step to spare), and a
/// reading that churns inside a step of it moves nothing.
pub(crate) fn paced_ahead(now: u64, reading: u64) -> u64 {
    if now >= reading.saturating_add(PACE_STEP) || now.saturating_add(PACE_STEP) <= reading {
        reading / PACE_STEP * PACE_STEP
    } else {
        now
    }
}

/// What the pacer set the window to, and how often it moved it each way (an
/// oscillating window is a finding even when phase B never waits).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PacerReport {
    pub(crate) first: Option<u64>,
    pub(crate) most: u64,
    pub(crate) last: u64,
    pub(crate) readings: usize,
    pub(crate) grew: usize,
    pub(crate) shrank: usize,
}

/// The adaptive window's pacer: until `window` closes, every [`PACE`] it
/// moves the window to [`adaptive_ahead`] of the process's resident sets, in
/// [`paced_ahead`]'s steps.
fn pace(window: &RegenWindow) -> PacerReport {
    let mut report = PacerReport::default();
    while !window.is_closed() {
        if let Some((hwm, rss)) = resident() {
            let now = window.ahead();
            let ahead = paced_ahead(now, adaptive_ahead(hwm, rss, window.parked()));
            if ahead != now {
                if ahead > now {
                    report.grew += 1;
                } else {
                    report.shrank += 1;
                }
                window.set_ahead(ahead);
            }
            report.first.get_or_insert(ahead);
            report.most = report.most.max(ahead);
            report.last = ahead;
            report.readings += 1;
        }
        std::thread::sleep(PACE);
    }
    report
}

/// The `BLOCK REGEN window pacer` line.
pub(crate) fn pacer_line(policy: AheadPolicy, report: Option<&PacerReport>) -> String {
    match (policy, report) {
        (AheadPolicy::Fixed(bytes), _) => format!(
            "BLOCK REGEN window pacer: fixed {:.2} GiB (LAMBDA_VM_BLOCK_REGEN_AHEAD_GIB)",
            bytes as f64 / GIB
        ),
        (AheadPolicy::Adaptive, Some(r)) if r.readings > 0 => format!(
            "BLOCK REGEN window pacer: adaptive (VmHWM − (VmRSS − parked) − {:.0} GiB, in [{:.0}, \
             {:.0}] GiB, steps of {:.0} GiB) · first {:.2} GiB · most {:.2} GiB · last {:.2} GiB · {} readings · grew {} \
             times · shrank {} times",
            AHEAD_MARGIN as f64 / GIB,
            AHEAD_FLOOR as f64 / GIB,
            AHEAD_CAP as f64 / GIB,
            PACE_STEP as f64 / GIB,
            r.first.unwrap_or(0) as f64 / GIB,
            r.most as f64 / GIB,
            r.last as f64 / GIB,
            r.readings,
            r.grew,
            r.shrank
        ),
        (AheadPolicy::Adaptive, _) => format!(
            "BLOCK REGEN window pacer: adaptive, no readings (no /proc/self/status) · {:.2} GiB",
            AHEAD_FLOOR as f64 / GIB
        ),
    }
}

/// `LAMBDA_VM_BLOCK_REGEN_REST_AHEAD_GIB` (default 4): the bytes the rest's
/// regenerator (N3) may hold deposited ahead of the groups that take them. Under
/// phase B's interleaved order it supplies only its share of phase B's pace, so
/// a small window suffices (D-WHIR-NODISK § N3.2).
pub(crate) fn rest_ahead_bytes() -> u64 {
    gib_knob("LAMBDA_VM_BLOCK_REGEN_REST_AHEAD_GIB", 4.0)
}

/// `LAMBDA_VM_BLOCK_REGEN_RESERVE_GIB` (default 10, #1013's): the host bytes
/// `auto` reserves for the regenerator in phase B once regeneration is armed —
/// its walk state, windows, jobs and generators' outputs, and the columns
/// deposited ahead of their groups. Never added unarmed.
pub(crate) fn regen_reserve_bytes() -> u64 {
    gib_knob("LAMBDA_VM_BLOCK_REGEN_RESERVE_GIB", 10.0)
}

fn gib_knob(var: &str, default: f64) -> u64 {
    let gib = std::env::var(var)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|g| g.is_finite() && *g >= 0.0)
        .unwrap_or(default);
    (gib * GIB) as u64
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
    /// Whether the layout's threads digest each chunk as they generate it
    /// (the shadow; live regeneration digests what it drops, at the drop).
    digests: bool,
}

#[derive(Default)]
struct Recorded {
    /// Per streamed table, its list's length at the end of each window.
    ends: [Vec<usize>; 8],
    windows: usize,
    recipes: BTreeMap<(usize, usize), Recipe>,
    /// Each recipe's chunk, by its place in the hand-out.
    by_order: Vec<(StreamTable, usize)>,
    /// Digests for a chunk with no recipe, and chunks of a table no recipe
    /// can hold.
    stray: usize,
    /// The rest's tables phase B can build again (N3), by their place among
    /// the rest's laid-out tables, and the block index of the rest's first.
    rest: std::collections::HashMap<usize, (RegenFamily, usize)>,
    rest_base: Option<usize>,
}

impl Recorder {
    /// A recorder whose layout threads digest each chunk (`digests`, the
    /// shadow's) or not (live regeneration's).
    pub(crate) fn new(digests: bool) -> Self {
        Self {
            digests,
            ..Self::default()
        }
    }

    /// The `k`-th table of the rest laid out is `family`'s `j`-th
    /// (`KECCAK_RND[j]`, `LT[j]`): phase B can build it again.
    pub(crate) fn rest_tag(&self, k: usize, family: RegenFamily, j: usize) {
        self.lock().rest.insert(k, (family, j));
    }

    /// The rest's tables start at block index `base` (every streamed chunk
    /// placed before them).
    pub(crate) fn set_rest_base(&self, base: usize) {
        self.lock().rest_base = Some(base);
    }

    /// The rest table phase B can build again at block index `t`.
    pub(crate) fn rest_at(&self, t: usize) -> Option<(RegenFamily, usize)> {
        let recorded = self.lock();
        let k = t.checked_sub(recorded.rest_base?)?;
        recorded.rest.get(&k).copied()
    }

    /// Each rest table phase B can build again, by family and index: its
    /// block index.
    pub(crate) fn rest_blocks(&self) -> std::collections::HashMap<(RegenFamily, usize), usize> {
        let recorded = self.lock();
        let Some(base) = recorded.rest_base else {
            return std::collections::HashMap::new();
        };
        recorded
            .rest
            .iter()
            .map(|(&k, &key)| (key, base + k))
            .collect()
    }

    /// Whether the layout's threads digest each chunk ([`Self::packed`]).
    pub(crate) fn digests(&self) -> bool {
        self.digests
    }

    /// The chunk handed out `order`-th, when one was: phase A places the
    /// streamed chunks first, in hand-out order, so it is the table at block
    /// index `order`.
    pub(crate) fn chunk_at(&self, order: usize) -> Option<(StreamTable, usize)> {
        self.lock().by_order.get(order).copied()
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
            r.by_order.push((job.table, job.index));
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
    pub(crate) fn recipes(&self) -> (Vec<Recipe>, usize) {
        let r = self.lock();
        let mut recipes: Vec<Recipe> = r.recipes.values().cloned().collect();
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
pub(crate) fn recorder(mode: RegenMode, streams: bool) -> Option<Recorder> {
    match (mode, streams) {
        (RegenMode::Off, _) => None,
        (_, true) => Some(Recorder::new(mode == RegenMode::Shadow)),
        (_, false) => {
            eprintln!(
                "BLOCK REGEN: {mode:?} wanted, but phase A does not stream its chunks packed (windows, \
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

/// Which streamed chunk a table is.
pub(crate) type StreamKey = (StreamTable, usize);

/// A generator alive while held (unwinding included).
struct Alive<'a>(&'a AtomicUsize);

impl Drop for Alive<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A job and the slot it fills: dropped undeposited (a generator that
/// panicked outside its catch, a channel torn down), it fails its slot (R10).
struct JobGuard {
    job: Option<ChunkJob>,
    slot: RegenSlot,
    settled: bool,
}

impl Drop for JobGuard {
    fn drop(&mut self) {
        if !self.settled {
            self.slot.fail("its generator stopped before depositing it");
        }
    }
}

/// What the live regenerator did.
#[derive(Debug, Default)]
pub(crate) struct LiveReport {
    pub(crate) dropped: usize,
    pub(crate) deposited: usize,
    pub(crate) mismatches: usize,
    /// Deposits refused because the window had closed.
    pub(crate) closed: usize,
    pub(crate) failures: Vec<String>,
    /// Dropped chunks the slicer did not cut before a higher rank, failed at
    /// once (R10).
    pub(crate) skipped: usize,
    /// Chunks cut that were not dropped, not generated.
    pub(crate) beyond: usize,
    pub(crate) windows: usize,
    pub(crate) cycles: usize,
    pub(crate) bytes: u64,
    pub(crate) first_ready: Option<f64>,
    pub(crate) last_ready: Option<f64>,
    pub(crate) wall: f64,
    /// Thread CPU: the executor, the walker (with its start state), the
    /// generators summed (Linux only).
    pub(crate) cpu: [Option<f64>; 3],
    pub(crate) generators: usize,
    pub(crate) error: Option<String>,
    pub(crate) state_bytes: usize,
    /// The `BLOCK REGEN window pacer` line, once the run is joined.
    pub(crate) pacer: Option<String>,
}

impl LiveReport {
    pub(crate) fn cpu_total(&self) -> Option<f64> {
        Some(self.cpu[0]? + self.cpu[1]? + self.cpu[2]?)
    }

    /// The `BLOCK REGEN live` line.
    pub(crate) fn line(&self) -> String {
        let secs = |s: Option<f64>| s.map_or("n/a".to_string(), |s| format!("{s:.1}"));
        format!(
            "BLOCK REGEN live: {} of {} dropped chunks deposited · {} mismatches · {} failed · {} \
             skipped (not cut in rank order) · {} refused (window closed) · {} not dropped \
             (skipped) · {} windows walked ({:.2} M cycles) · {:.2} GiB packed · ready from {} s \
             to {} s · wall {:.2} s · CPU exec {} · walk {} · {} generators {} = {} s · walk state \
             {:.2} GiB{}",
            self.deposited,
            self.dropped,
            self.mismatches,
            self.failures.len(),
            self.skipped,
            self.closed,
            self.beyond,
            self.windows,
            self.cycles as f64 / 1e6,
            self.bytes as f64 / GIB,
            secs(self.first_ready),
            secs(self.last_ready),
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

/// A test's faults in the live regenerator's exits and deposits (R2/R10):
/// each makes one of them happen, so a test can check that no slot is left
/// waiting and nothing wrong is proved. All off outside tests.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct LiveFaults {
    /// The walk fails at this window.
    pub(crate) walk_error_at: Option<usize>,
    /// The generator that takes the job of this rank panics outside its
    /// catch.
    pub(crate) generator_panics_at_rank: Option<u64>,
    /// The slicer never hands out this rank's chunk.
    pub(crate) skip_rank: Option<u64>,
    /// The chunk of this rank is deposited with one packed bit flipped.
    pub(crate) bend_rank: Option<u64>,
    /// The window checks deposits on their shape alone (the digest off), so
    /// a bent deposit reaches the kept tree's check.
    pub(crate) digest_off: bool,
}

/// The live regenerator (D-REGEN §2.3, sequential form): `builder` walks the
/// run of `program` on `private_input`, executed `window_cycles` cycles at a
/// time on a thread of its own; every chunk it cuts that was dropped is
/// generated in `form`, packed if it is not, and deposited into its slot on
/// one of `generators` threads, in rank order of dispatch. Never panics; every
/// exit leaves no slot waiting.
///
/// The ranks are phase A's hand-out order, window by window, so the windows
/// must be phase A's: a rank the slicer cuts after a higher one is failed — its
/// table refused, never waited on (R10).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_live(
    program: &Elf,
    private_input: &[u8],
    builder: RegenBuilder,
    window_cycles: usize,
    window: Arc<RegenWindow>,
    producer: RegenProducer,
    dropped: Vec<(StreamKey, RegenSlot)>,
    generators: usize,
    form: TraceForm,
    started: Instant,
    faults: LiveFaults,
) -> LiveReport {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::mpsc;
    let mut builder = builder;
    let mut report = LiveReport {
        dropped: dropped.len(),
        generators,
        ..LiveReport::default()
    };
    #[cfg(test)]
    if faults.digest_off {
        window.set_verify(false);
    }
    // Position of each dropped chunk in rank order.
    let position: BTreeMap<(usize, usize), usize> = dropped
        .iter()
        .enumerate()
        .filter_map(|(i, ((table, index), _))| Some(((slot(*table)?, *index), i)))
        .collect();
    let mut settled = vec![false; dropped.len()];
    let fail_from = |settled: &mut [bool], from: usize, upto: usize, why: &str| -> usize {
        let mut n = 0;
        for i in from..upto {
            if !settled[i] {
                dropped[i].1.fail(why);
                settled[i] = true;
                n += 1;
            }
        }
        n
    };
    let deposited = AtomicUsize::new(0);
    let mismatches = AtomicUsize::new(0);
    let closed = AtomicUsize::new(0);
    let failures = Mutex::new(Vec::new());
    let ready = Mutex::new((None::<f64>, None::<f64>, 0u64));
    let generators = generators.max(1);
    let (job_tx, job_rx) = mpsc::sync_channel::<JobGuard>(2 * generators);
    let job_rx = Mutex::new(job_rx);
    // Generators alive: when none is, the walker stops handing out (a full
    // queue would otherwise hold it forever).
    let alive = AtomicUsize::new(0);
    std::thread::scope(|s| {
        let (log_tx, log_rx) = mpsc::sync_channel::<Vec<executor::vm::logs::Log>>(2);
        let window_ref = &window;
        let exec = std::thread::Builder::new()
            .name("regen-exec".to_string())
            .spawn_scoped(s, move || {
                let cpu0 = thread_cpu_secs();
                let run = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
                    let mut executor = Executor::new(program, private_input.to_vec())
                        .map_err(|e| format!("the executor: {e}"))?;
                    while let Some(logs) = executor
                        .resume_with_limit(window_cycles)
                        .map_err(|e| format!("the execution: {e}"))?
                    {
                        if window_ref.is_closed() || log_tx.send(logs.to_vec()).is_err() {
                            break;
                        }
                    }
                    Ok(())
                }))
                .unwrap_or_else(|_| Err("the executor panicked".to_string()));
                (run, cpu_since(cpu0))
            });
        let exec = match exec {
            Ok(exec) => Some(exec),
            Err(e) => {
                report.error = Some(format!("the executor thread did not start: {e}"));
                None
            }
        };
        let mut gens = Vec::with_capacity(generators);
        if exec.is_some() {
            for g in 0..generators {
                let Some(token) = window.producer() else {
                    break;
                };
                let (job_rx, deposited, mismatches, closed, failures, ready, alive) = (
                    &job_rx,
                    &deposited,
                    &mismatches,
                    &closed,
                    &failures,
                    &ready,
                    &alive,
                );
                alive.fetch_add(1, Ordering::SeqCst);
                let spawned = std::thread::Builder::new()
                    .name(format!("regen-gen-{g}"))
                    .spawn_scoped(s, move || {
                        let _token = token;
                        let _alive = Alive(alive);
                        let cpu0 = thread_cpu_secs();
                        loop {
                            let next = job_rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                            let Ok(mut guard) = next else {
                                break;
                            };
                            let Some(job) = guard.job.take() else {
                                continue;
                            };
                            if faults.generator_panics_at_rank == Some(guard.slot.rank()) {
                                panic!("a generator dies outside its catch (a test's fault)");
                            }
                            let (table, index) = (job.table, job.index);
                            let built = catch_unwind(AssertUnwindSafe(|| {
                                let mut trace = job.generate_as(form).trace;
                                if trace.narrow_main().is_none() {
                                    trace.pack_main_narrow();
                                }
                                // Moved out, not copied.
                                trace.take_narrow_main()
                            }));
                            let packed = match built {
                                Ok(Some(packed)) => packed,
                                Ok(None) => {
                                    let why =
                                        format!("{table:?}[{index}]: the columns did not pack");
                                    guard.slot.fail(&why);
                                    guard.settled = true;
                                    failures.lock().unwrap_or_else(|e| e.into_inner()).push(why);
                                    continue;
                                }
                                Err(_) => {
                                    let why = format!("{table:?}[{index}]: generation panicked");
                                    guard.slot.fail(&why);
                                    guard.settled = true;
                                    failures.lock().unwrap_or_else(|e| e.into_inner()).push(why);
                                    continue;
                                }
                            };
                            let packed = if faults.bend_rank == Some(guard.slot.rank()) {
                                bend(packed)
                            } else {
                                packed
                            };
                            let bytes = packed.data().len() as u64;
                            let out = guard.slot.deposit(packed);
                            // Deposited, refused or closed: the slot is settled
                            // either way (a closed window settles every waiter).
                            guard.settled = true;
                            match out {
                                Ok(()) => {
                                    deposited.fetch_add(1, Ordering::Relaxed);
                                    let at = started.elapsed().as_secs_f64();
                                    let mut r = ready.lock().unwrap_or_else(|e| e.into_inner());
                                    r.0 = Some(r.0.map_or(at, |f: f64| f.min(at)));
                                    r.1 = Some(r.1.map_or(at, |l: f64| l.max(at)));
                                    r.2 += bytes;
                                }
                                Err(RegenError::Mismatch) => {
                                    mismatches.fetch_add(1, Ordering::Relaxed);
                                }
                                // Keep draining (R3): the walker may be waiting
                                // on a full queue.
                                Err(RegenError::Closed(_)) => {
                                    closed.fetch_add(1, Ordering::Relaxed);
                                }
                                Err(RegenError::Failed(why)) => {
                                    failures
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner())
                                        .push(format!("{table:?}[{index}]: {why}"));
                                }
                            }
                        }
                        cpu_since(cpu0)
                    });
                match spawned {
                    Ok(handle) => gens.push(handle),
                    Err(e) => {
                        alive.fetch_sub(1, Ordering::SeqCst);
                        report
                            .failures
                            .push(format!("generator {g} did not start: {e}"));
                    }
                }
            }
        }
        // This thread walks and cuts, and hands out the dropped chunks in rank
        // order; a dropped chunk not cut before a higher rank is failed at once
        // (R10).
        let cpu0 = thread_cpu_secs();
        let mut next = 0usize;
        let mut stop: Option<String> = None;
        if gens.is_empty() {
            stop = Some(
                report
                    .error
                    .clone()
                    .unwrap_or_else(|| "no generator started".to_string()),
            );
        } else {
            for logs in log_rx.iter() {
                if window.is_closed() {
                    stop = Some("the window closed".to_string());
                    break;
                }
                if faults.walk_error_at == Some(report.windows) {
                    stop = Some("the regeneration walk failed (a test's fault)".to_string());
                    break;
                }
                report.windows += 1;
                let jobs = match catch_unwind(AssertUnwindSafe(|| builder.push(&logs))) {
                    Ok(Ok(jobs)) => jobs,
                    Ok(Err(e)) => {
                        stop = Some(format!("the regeneration walk: {e}"));
                        break;
                    }
                    Err(_) => {
                        stop = Some("the regeneration walk panicked".to_string());
                        break;
                    }
                };
                // The window's dropped chunks in rank order: within a window
                // the order is the slicer's own, so only the windows must be
                // phase A's.
                let mut cut: Vec<(usize, ChunkJob)> = Vec::with_capacity(jobs.len());
                for job in jobs {
                    match slot(job.table).and_then(|t| position.get(&(t, job.index))) {
                        Some(&pos) => cut.push((pos, job)),
                        None => report.beyond += 1,
                    }
                }
                cut.sort_by_key(|&(pos, _)| pos);
                for (pos, job) in cut {
                    if faults.skip_rank == Some(dropped[pos].1.rank()) {
                        continue;
                    }
                    if pos < next || settled[pos] {
                        continue;
                    }
                    report.skipped += fail_from(
                        &mut settled,
                        next,
                        pos,
                        "the slicer did not cut it before a higher rank",
                    );
                    settled[pos] = true;
                    next = pos + 1;
                    let mut guard = JobGuard {
                        job: Some(job),
                        slot: dropped[pos].1.clone(),
                        settled: false,
                    };
                    // A job that cannot be handed out is dropped, and its
                    // guard fails its slot.
                    loop {
                        match job_tx.try_send(guard) {
                            Ok(()) => break,
                            Err(mpsc::TrySendError::Full(back)) => {
                                if alive.load(Ordering::SeqCst) == 0 {
                                    drop(back);
                                    stop = Some("every generator stopped".to_string());
                                    break;
                                }
                                guard = back;
                                std::thread::sleep(std::time::Duration::from_millis(2));
                            }
                            Err(mpsc::TrySendError::Disconnected(back)) => {
                                drop(back);
                                stop = Some("the generators stopped".to_string());
                                break;
                            }
                        }
                    }
                    if stop.is_some() {
                        break;
                    }
                }
                if stop.is_some() || next == dropped.len() {
                    break;
                }
            }
        }
        // Every exit: what was not handed out fails, with why (R2).
        let why = stop
            .clone()
            .unwrap_or_else(|| "the run ended before its chunk was cut".to_string());
        let unsent = fail_from(&mut settled, next, dropped.len(), &why);
        report.skipped += unsent;
        match stop {
            Some(stop) if stop != "the window closed" => {
                report.error.get_or_insert(stop);
            }
            None if unsent > 0 => {
                report.error.get_or_insert(why);
            }
            _ => {}
        }
        drop(log_rx);
        drop(job_tx);
        drop(producer);
        report.cycles = builder.cycles();
        report.state_bytes = builder.state_bytes();
        report.cpu[1] = cpu_since(cpu0);
        let mut gen_cpu = Some(0.0);
        for g in gens {
            let cpu = g.join().unwrap_or_else(|_| {
                failures
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push("a generator panicked".to_string());
                None
            });
            gen_cpu = gen_cpu.zip(cpu).map(|(a, b)| a + b);
        }
        // Jobs no generator took (every one stopped): their guards fail their
        // slots now, not when this function returns.
        while let Ok(guard) = job_rx.lock().unwrap_or_else(|e| e.into_inner()).try_recv() {
            drop(guard);
        }
        report.cpu[2] = gen_cpu;
        if let Some(exec) = exec {
            match exec.join() {
                Ok((run, cpu)) => {
                    report.cpu[0] = cpu;
                    if let Err(e) = run {
                        report.error.get_or_insert(e);
                    }
                }
                Err(_) => {
                    report
                        .error
                        .get_or_insert("the executor thread panicked".to_string());
                }
            }
        }
    });
    report.deposited = deposited.into_inner();
    report.mismatches = mismatches.into_inner();
    report.closed = closed.into_inner();
    report
        .failures
        .extend(failures.into_inner().unwrap_or_else(|e| e.into_inner()));
    let (first, last, bytes) = ready.into_inner().unwrap_or_else(|e| e.into_inner());
    report.first_ready = first;
    report.last_ready = last;
    report.bytes = bytes;
    report.wall = started.elapsed().as_secs_f64();
    report
}

/// `packed` with the low bit of its first byte flipped: a wrong word of the
/// same shape (a test's fault).
fn bend(packed: multilinear::narrow::NarrowColumns) -> multilinear::narrow::NarrowColumns {
    let (rows, widths, mut data) = packed.into_parts();
    if let Some(b) = data.first_mut() {
        *b ^= 1;
    }
    multilinear::narrow::NarrowColumns::from_parts(rows, widths, data)
        .expect("one bit flipped keeps the shape")
}

/// The live regenerator on a thread of its own (`regen-walk`), for the phase
/// B beside it. Dropped — the prove returned, refused or unwound — it closes
/// the window, so a regenerator waiting to deposit for takers that are gone
/// returns, and joins it (R4): no regenerator outlives its prove.
pub(crate) struct LiveRun {
    window: Arc<RegenWindow>,
    handle: Option<std::thread::JoinHandle<LiveReport>>,
    /// The adaptive window's pacer ([`AheadPolicy::Adaptive`]).
    pacer: Option<std::thread::JoinHandle<PacerReport>>,
    policy: AheadPolicy,
    started: Instant,
    dropped: usize,
    spawn_error: Option<String>,
}

impl LiveRun {
    /// [`run_live`] over the run of `elf_bytes` on `private_input`, executed
    /// `window_cycles` at a time (phase A's windows), its chunks cut at
    /// `max_rows` and generated in `form` (phase A's), on [`regen_generators`]
    /// generators, depositing `dropped` into `window`, which `policy` sizes;
    /// its times are from now. A regenerator that cannot start drops
    /// `producer`, so every slot fails at once.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn spawn(
        elf_bytes: Vec<u8>,
        private_input: Vec<u8>,
        max_rows: crate::tables::MaxRowsConfig,
        window_cycles: usize,
        window: Arc<RegenWindow>,
        producer: RegenProducer,
        dropped: Vec<(StreamKey, RegenSlot)>,
        form: TraceForm,
        faults: LiveFaults,
        policy: AheadPolicy,
    ) -> Self {
        let started = Instant::now();
        window.set_ahead(policy.initial());
        let pacer = (policy == AheadPolicy::Adaptive)
            .then(|| {
                let window = Arc::clone(&window);
                std::thread::Builder::new()
                    .name("regen-pacer".to_string())
                    .spawn(move || pace(&window))
                    .ok()
            })
            .flatten();
        let count = dropped.len();
        let window_in = Arc::clone(&window);
        let spawned = std::thread::Builder::new()
            .name("regen-walk".to_string())
            .spawn(move || {
                let cpu0 = thread_cpu_secs();
                let failed = |why: String, dropped: &[(StreamKey, RegenSlot)]| {
                    for (_, slot) in dropped {
                        slot.fail(&why);
                    }
                    LiveReport {
                        dropped: dropped.len(),
                        skipped: dropped.len(),
                        error: Some(why),
                        ..LiveReport::default()
                    }
                };
                let program = match Elf::load(&elf_bytes) {
                    Ok(program) => program,
                    Err(e) => return failed(format!("the regenerator's ELF: {e}"), &dropped),
                };
                match RegenBuilder::new(&program, &private_input, &max_rows) {
                    Ok(builder) => {
                        let setup = cpu_since(cpu0);
                        let mut report = run_live(
                            &program,
                            &private_input,
                            builder,
                            window_cycles,
                            window_in,
                            producer,
                            dropped,
                            regen_generators(),
                            form,
                            started,
                            faults,
                        );
                        report.cpu[1] = report.cpu[1].zip(setup).map(|(walk, setup)| walk + setup);
                        report
                    }
                    Err(e) => failed(format!("the regenerator: {e}"), &dropped),
                }
            });
        let (handle, spawn_error) = match spawned {
            Ok(handle) => (Some(handle), None),
            // The closure went with the failed spawn, its producer with it:
            // every slot has failed.
            Err(e) => (None, Some(format!("the regenerator did not start: {e}"))),
        };
        Self {
            window,
            handle,
            pacer,
            policy,
            started,
            dropped: count,
            spawn_error,
        }
    }

    /// When it started: every time its report gives is from here.
    pub(crate) fn started(&self) -> Instant {
        self.started
    }

    /// Its report once the prove is done (the window closed first, so it
    /// returns); a thread that did not start or that panicked is a report
    /// with the error.
    pub(crate) fn join(mut self) -> LiveReport {
        self.window.close("the block's prove ended");
        let failed = |error: String| LiveReport {
            dropped: self.dropped,
            error: Some(error),
            ..LiveReport::default()
        };
        let mut report = match self.handle.take() {
            Some(handle) => handle
                .join()
                .unwrap_or_else(|_| failed("the regenerator panicked".to_string())),
            None => failed(
                self.spawn_error
                    .take()
                    .unwrap_or_else(|| "the regenerator did not start".to_string()),
            ),
        };
        let paced = self.pacer.take().and_then(|pacer| pacer.join().ok());
        report.pacer = Some(pacer_line(self.policy, paced.as_ref()));
        report
    }
}

impl Drop for LiveRun {
    fn drop(&mut self) {
        self.window.close("the block's prove ended");
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        if let Some(pacer) = self.pacer.take() {
            let _ = pacer.join();
        }
    }
}

/// What the rest's regenerator did ([`RestRun`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct RestReport {
    pub(crate) dropped: usize,
    pub(crate) deposited: usize,
    pub(crate) mismatches: usize,
    /// Deposits refused because the window had closed.
    pub(crate) closed: usize,
    pub(crate) failures: Vec<String>,
    /// Tables built again, by family (KECCAK_RND, LT).
    pub(crate) built: [usize; 2],
    pub(crate) bytes: u64,
    pub(crate) first_ready: Option<f64>,
    pub(crate) last_ready: Option<f64>,
    pub(crate) wall: f64,
    /// The generators' thread CPU summed (Linux only).
    pub(crate) cpu: Option<f64>,
    pub(crate) generators: usize,
    pub(crate) error: Option<String>,
}

impl RestReport {
    /// The `BLOCK REGEN rest` line.
    pub(crate) fn line(&self) -> String {
        let secs = |s: Option<f64>| s.map_or("n/a".to_string(), |s| format!("{s:.1}"));
        format!(
            "BLOCK REGEN rest: {} of {} dropped tables deposited · {} mismatches · {} failed · {} \
             refused (window closed) · KECCAK_RND {} · LT {} built again · {:.2} GiB packed · \
             ready from {} s to {} s · wall {:.2} s · {} generators CPU {} s{}",
            self.deposited,
            self.dropped,
            self.mismatches,
            self.failures.len(),
            self.closed,
            self.built[0],
            self.built[1],
            self.bytes as f64 / GIB,
            secs(self.first_ready),
            secs(self.last_ready),
            self.wall,
            self.generators,
            secs(self.cpu),
            self.error
                .as_ref()
                .map_or(String::new(), |e| format!(" · stopped: {e}")),
        )
    }
}

/// One table the rest's regenerator builds: its place among the rest's
/// dropped tables, its rank (block index), its slot and its job.
type RestWork = (u64, u64, RegenSlot, crate::tables::trace_builder::RegenJob);

/// The rest's regenerator (D-WHIR-NODISK N3) on threads of its own, for the
/// phase B beside it: KECCAK_RND's and LT's dropped tables built again from the
/// lists the finish kept ([`RestRegen`]), on `generators` threads taking them in
/// rank order, each deposited into its slot (digest-checked: another table
/// fails its slot). A family's list goes once its jobs have run. Dropped — the
/// prove returned, refused or unwound — it closes the window and joins.
pub(crate) struct RestRun {
    window: Arc<RegenWindow>,
    handle: Option<std::thread::JoinHandle<RestReport>>,
    dropped: usize,
    spawn_error: Option<String>,
}

impl RestRun {
    /// The regenerator over `regen`, depositing `dropped` (each a rank, the
    /// block index `blocks` gives a family's table, and its slot) into
    /// `window`; `producer` holds the window's slots open until it ends.
    /// `faults` name a table by its place among the dropped ones, in rank
    /// order.
    pub(crate) fn spawn(
        mut regen: RestRegen,
        blocks: std::collections::HashMap<(RegenFamily, usize), usize>,
        window: Arc<RegenWindow>,
        producer: RegenProducer,
        dropped: Vec<(u64, RegenSlot)>,
        generators: usize,
        faults: LiveFaults,
    ) -> Self {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let count = dropped.len();
        let started = Instant::now();
        let spawned = std::thread::Builder::new()
            .name("regen-rest".to_string())
            .spawn(move || {
                let mut slots: std::collections::HashMap<u64, RegenSlot> =
                    dropped.into_iter().collect();
                let mut report = RestReport {
                    dropped: count,
                    generators,
                    ..RestReport::default()
                };
                // The work, in rank order: each family in AIR order, each table
                // in the family's order (the rest's block indices ascend so).
                let mut work: Vec<RestWork> = Vec::new();
                let mut built = [0usize; 2];
                for (f, family) in RegenFamily::ALL.into_iter().enumerate() {
                    let jobs = match regen.take(family) {
                        Ok(jobs) => jobs,
                        Err(e) => {
                            report.error = Some(format!("{e:?}"));
                            break;
                        }
                    };
                    for (j, job) in jobs.into_iter().enumerate() {
                        let Some(job) = job else { continue };
                        let Some(slot) = blocks
                            .get(&(family, j))
                            .and_then(|&t| slots.remove(&(t as u64)).map(|slot| (t as u64, slot)))
                        else {
                            continue;
                        };
                        built[f] += 1;
                        work.push((0, slot.0, slot.1, job));
                    }
                }
                work.sort_by_key(|(_, _, slot, _)| slot.order_key());
                for (k, item) in work.iter_mut().enumerate() {
                    item.0 = k as u64;
                }
                // A dropped table no job builds is failed now, as is every one
                // after a stop.
                for (rank, slot) in slots.drain() {
                    let why = format!("rest table {rank} has no job to build it again");
                    slot.fail(&why);
                    report.failures.push(why);
                }
                if let Some(why) = report.error.clone() {
                    for (_, _, slot, _) in &work {
                        slot.fail(&why);
                    }
                    drop(producer);
                    report.wall = started.elapsed().as_secs_f64();
                    return report;
                }
                report.built = built;
                let queue = Mutex::new(std::collections::VecDeque::from(work));
                let deposited = AtomicUsize::new(0);
                let mismatches = AtomicUsize::new(0);
                let closed = AtomicUsize::new(0);
                let failures: Mutex<Vec<String>> = Mutex::new(Vec::new());
                let ready: Mutex<(Option<f64>, Option<f64>, u64)> = Mutex::new((None, None, 0));
                let cpu: Mutex<Option<f64>> = Mutex::new(Some(0.0));
                std::thread::scope(|scope| {
                    for _ in 0..generators.max(1) {
                        scope.spawn(|| {
                            let cpu0 = thread_cpu_secs();
                            loop {
                                // FIFO: ranks go out in order, so the window's
                                // frontier is always a table some thread builds.
                                let next =
                                    queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
                                let Some((k, rank, slot, job)) = next else {
                                    break;
                                };
                                let fail = |why: String| {
                                    slot.fail(&why);
                                    failures.lock().unwrap_or_else(|e| e.into_inner()).push(why);
                                };
                                if faults.skip_rank == Some(k) {
                                    fail(format!("rest table {rank}: skipped (a test's fault)"));
                                    continue;
                                }
                                let built = catch_unwind(AssertUnwindSafe(|| {
                                    if faults.generator_panics_at_rank == Some(k) {
                                        panic!(
                                            "a test's fault: the rest's generator panics at {rank}"
                                        );
                                    }
                                    job()
                                }));
                                let mut table = match built {
                                    Ok(Ok(table)) => table,
                                    Ok(Err(e)) => {
                                        fail(format!("rest table {rank}: {e:?}"));
                                        continue;
                                    }
                                    Err(_) => {
                                        fail(format!("rest table {rank}: its generator panicked"));
                                        continue;
                                    }
                                };
                                let Some(packed) = table.take_narrow_main() else {
                                    fail(format!(
                                        "rest table {rank}: built without packed columns"
                                    ));
                                    continue;
                                };
                                let packed = if faults.bend_rank == Some(k) {
                                    bend(packed)
                                } else {
                                    packed
                                };
                                let bytes = packed.data().len() as u64;
                                match slot.deposit(packed) {
                                    Ok(()) => {
                                        deposited.fetch_add(1, Ordering::Relaxed);
                                        let at = started.elapsed().as_secs_f64();
                                        let mut r = ready.lock().unwrap_or_else(|e| e.into_inner());
                                        r.0 = Some(r.0.map_or(at, |a| a.min(at)));
                                        r.1 = Some(r.1.map_or(at, |a| a.max(at)));
                                        r.2 += bytes;
                                    }
                                    Err(RegenError::Mismatch) => {
                                        mismatches.fetch_add(1, Ordering::Relaxed);
                                    }
                                    Err(RegenError::Closed(_)) => {
                                        closed.fetch_add(1, Ordering::Relaxed);
                                    }
                                    Err(e) => {
                                        failures
                                            .lock()
                                            .unwrap_or_else(|e| e.into_inner())
                                            .push(format!("rest table {rank}: {e}"));
                                    }
                                }
                            }
                            let spent = cpu_since(cpu0);
                            let mut total = cpu.lock().unwrap_or_else(|e| e.into_inner());
                            *total = total.zip(spent).map(|(a, b)| a + b);
                        });
                    }
                });
                drop(producer);
                report.deposited = deposited.into_inner();
                report.mismatches = mismatches.into_inner();
                report.closed = closed.into_inner();
                report
                    .failures
                    .extend(failures.into_inner().unwrap_or_else(|e| e.into_inner()));
                let (first, last, bytes) = ready.into_inner().unwrap_or_else(|e| e.into_inner());
                report.first_ready = first;
                report.last_ready = last;
                report.bytes = bytes;
                report.cpu = cpu.into_inner().unwrap_or_else(|e| e.into_inner());
                report.wall = started.elapsed().as_secs_f64();
                report
            });
        let (handle, spawn_error) = match spawned {
            Ok(handle) => (Some(handle), None),
            // The closure went with the failed spawn, its producer with it:
            // every slot has failed.
            Err(e) => (
                None,
                Some(format!("the rest's regenerator did not start: {e}")),
            ),
        };
        Self {
            window,
            handle,
            dropped: count,
            spawn_error,
        }
    }

    /// Its report once the prove is done (the window closed first, so it
    /// returns).
    pub(crate) fn join(mut self) -> RestReport {
        self.window.close("the block's prove ended");
        match self.handle.take() {
            Some(handle) => handle.join().unwrap_or_else(|_| RestReport {
                dropped: self.dropped,
                error: Some("the rest's regenerator panicked".to_string()),
                ..RestReport::default()
            }),
            None => RestReport {
                dropped: self.dropped,
                error: Some(
                    self.spawn_error
                        .take()
                        .unwrap_or_else(|| "the rest's regenerator did not start".to_string()),
                ),
                ..RestReport::default()
            },
        }
    }
}

impl Drop for RestRun {
    fn drop(&mut self) {
        self.window.close("the block's prove ended");
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// The `BLOCK REGEN dropped` line: what phase A dropped, of `recipes` streamed
/// chunks.
pub(crate) fn dropped_line(
    mode: RegenMode,
    no_disk: bool,
    report: &stark::multilinear_block::DropReport,
    recipes: usize,
) -> String {
    let armed = match report.armed {
        Some(at) => format!("armed at {at:.1} s"),
        None if mode == RegenMode::Always => "always".to_string(),
        None => "never armed".to_string(),
    };
    format!(
        "BLOCK REGEN dropped: {mode:?}{} · {} chunks {:.2} GiB (drop-back {} chunks {:.2} GiB) · \
         {armed} · {recipes} recipes · {} refused (no longer on the host) · {} held (their group \
         dropped another class)",
        if no_disk { " (no disk)" } else { "" },
        report.tables,
        report.bytes as f64 / GIB,
        report.back_tables,
        report.back_bytes as f64 / GIB,
        report.refused_late,
        report.held,
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
        ..RegenStamps::default()
    }
}

/// What a shadow run left for the readout ([`super::BlockStamps::regen`]).
#[derive(Clone, Debug, Default)]
pub struct RegenStamps {
    pub recipes: usize,
    /// Live regeneration: the chunks phase A dropped, and their packed bytes.
    pub dropped: usize,
    pub dropped_bytes: u64,
    pub digested: usize,
    pub stray: usize,
    pub regenerated: usize,
    pub mismatches: usize,
    pub failed: usize,
    pub missing: usize,
    pub error: Option<String>,
    /// Would-wait under the rest-first plan, when it is known.
    pub would_wait_rest_first: Option<f64>,
    /// Of `dropped` and `regenerated`, the rest's tables (N3: KECCAK_RND's and
    /// LT's, built again from the kept lists).
    pub rest_dropped: usize,
    pub rest_regenerated: usize,
    /// Tables phase B could rebuild kept on the host because their group
    /// dropped the other class (N3's one class a group).
    pub held: usize,
    /// The groups in the order phase B took them.
    pub phase_b_order: Vec<usize>,
    /// The `BLOCK REGEN` lines.
    pub lines: Vec<String>,
}

/// The `BLOCK REGEN phase B order taken` line (N3): the order phase B took,
/// run-length by each group's rebuilt class (`P` nothing rebuilt, `S` the
/// streamed chunks, `R` the rest), and the rest's share of the rebuilt groups'
/// `cells` started by each tenth of phase B (`spans`: each group's start and
/// end, seconds), against its share of them all, the plan's.
pub(crate) fn order_taken_line(
    order: &[usize],
    classes: &[Option<usize>],
    cells: &[u64],
    spans: &[(f64, f64)],
) -> String {
    let class = |g: usize| classes.get(g).copied().flatten();
    let mut runs: Vec<(char, usize)> = Vec::new();
    for &g in order {
        let c = match class(g) {
            None => 'P',
            Some(0) => 'S',
            Some(_) => 'R',
        };
        match runs.last_mut() {
            Some((last, n)) if *last == c => *n += 1,
            _ => runs.push((c, 1)),
        }
    }
    let cells_of = |g: usize| cells.get(g).copied().unwrap_or(0) as f64;
    let (mut rest_all, mut rebuilt_all) = (0.0, 0.0);
    for &g in order {
        if let Some(c) = class(g) {
            rebuilt_all += cells_of(g);
            if c > 0 {
                rest_all += cells_of(g);
            }
        }
    }
    let share = |rest: f64, all: f64| {
        if all > 0.0 {
            format!("{:.2}", rest / all)
        } else {
            "-".to_string()
        }
    };
    let begin = spans.iter().map(|s| s.0).fold(f64::INFINITY, f64::min);
    let end = spans.iter().map(|s| s.1).fold(f64::NEG_INFINITY, f64::max);
    let tenths: Vec<String> = (1..=10)
        .map(|k| {
            let by = begin + (end - begin) * k as f64 / 10.0;
            let (mut rest, mut all) = (0.0, 0.0);
            for &g in order {
                if class(g).is_some() && spans.get(g).is_some_and(|s| s.0 <= by) {
                    all += cells_of(g);
                    if class(g) > Some(0) {
                        rest += cells_of(g);
                    }
                }
            }
            share(rest, all)
        })
        .collect();
    format!(
        "BLOCK REGEN phase B order taken: rest share of the rebuilt cells started by each tenth \
         of phase B [{}] against {} overall · {}",
        tenths.join(", "),
        share(rest_all, rebuilt_all),
        runs.iter()
            .map(|(c, n)| format!("{c}{n}"))
            .collect::<Vec<_>>()
            .join(" ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ★ The production default with the knobs coupled (N4a): both unset →
    /// `auto` (the spill tier stays `auto`); a set regen is read when the
    /// prove starts, whatever the spill; an unset regen beside a set spill is
    /// `off`, as before.
    #[test]
    fn the_production_regen_is_auto_unless_a_knob_is_set() {
        assert_eq!(production_regen_from(None, None), Some(RegenMode::Auto));
        for spill in ["auto", "off", "always", "2.5"] {
            assert_eq!(
                production_regen_from(None, Some(spill)),
                Some(RegenMode::Off),
                "{spill}"
            );
        }
        for regen in ["off", "shadow", "auto", "always", "bogus"] {
            assert_eq!(production_regen_from(Some(regen), None), None, "{regen}");
            assert_eq!(
                production_regen_from(Some(regen), Some("auto")),
                None,
                "{regen}"
            );
        }
        // A set regen parses as it did; unset, alone, it is off.
        assert_eq!(parse_regen_mode(None).unwrap(), RegenMode::Off);
        assert!(parse_regen_mode(Some("bogus")).is_err());
        assert_eq!(
            regen_mode_line(RegenMode::Auto, true, None, None),
            "BLOCK REGEN mode: Auto · LAMBDA_VM_BLOCK_REGEN unset · LAMBDA_VM_BLOCK_SPILL unset \
             · no disk"
        );
        assert_eq!(
            regen_mode_line(RegenMode::Off, false, Some(" off "), Some("auto")),
            "BLOCK REGEN mode: Off · LAMBDA_VM_BLOCK_REGEN=off · LAMBDA_VM_BLOCK_SPILL=auto"
        );
    }

    /// The order line (N3): the classes in the order taken, run-length, and
    /// the rest's share of the rebuilt cells started by each tenth of phase B.
    #[test]
    fn the_order_line_reads_the_rest_share_over_phase_b() {
        // Groups 0 and 1 streamed, 2 and 3 rest, 4 plain; phase B takes the
        // plain one, then S R S R, each a second long.
        let classes = [Some(0), Some(0), Some(1), Some(1), None];
        let cells = [10, 10, 30, 30, 5];
        let order = [4, 0, 2, 1, 3];
        let mut spans = [(0.0, 0.0); 5];
        for (p, &g) in order.iter().enumerate() {
            spans[g] = (p as f64, p as f64 + 1.0);
        }
        assert_eq!(
            order_taken_line(&order, &classes, &cells, &spans),
            "BLOCK REGEN phase B order taken: rest share of the rebuilt cells started by each tenth \
             of phase B [-, 0.00, 0.00, 0.75, 0.75, 0.60, 0.60, 0.75, 0.75, 0.75] against 0.75 overall \
             · P1 S1 R1 S1 R1"
        );
        assert!(
            order_taken_line(&[0, 1], &[None, None], &[1, 1], &[(0.0, 1.0), (1.0, 2.0)])
                .ends_with("[-, -, -, -, -, -, -, -, -, -] against - overall · P2")
        );
    }

    /// `LAMBDA_VM_BLOCK_REGEN`: `off` (and unset), `shadow`, `auto`, `always`;
    /// anything else is refused (an error, not a panic).
    #[test]
    fn the_regen_mode_reads_its_knob() {
        assert_eq!(parse_regen_mode(None).unwrap(), RegenMode::Off);
        assert_eq!(parse_regen_mode(Some(" off ")).unwrap(), RegenMode::Off);
        assert_eq!(parse_regen_mode(Some("shadow")).unwrap(), RegenMode::Shadow);
        assert_eq!(parse_regen_mode(Some("auto")).unwrap(), RegenMode::Auto);
        assert_eq!(parse_regen_mode(Some("always")).unwrap(), RegenMode::Always);
        for bad in ["", "on", "Shadow", "1", "Auto", "ALWAYS"] {
            assert!(parse_regen_mode(Some(bad)).is_err(), "{bad:?}");
        }
        assert!(!RegenMode::Off.live() && !RegenMode::Shadow.live());
        assert!(RegenMode::Auto.live() && RegenMode::Always.live());
    }

    /// ★ The adaptive window: what the peak leaves above the resident set (the
    /// window's own deposits left out), less 4 GiB, in [4, 16] GiB.
    #[test]
    fn the_adaptive_window_is_clamped_to_its_bounds() {
        const G: u64 = 1 << 30;
        // Phase B at T200 (RYZEN 049): a 72.5 GiB peak over ≈ 15 GiB: the cap.
        assert_eq!(adaptive_ahead(72 * G + G / 2, 15 * G, 0), AHEAD_CAP);
        // p90 with no disk (RYZEN 044): a 66 GiB peak over ≈ 56: in between.
        assert_eq!(adaptive_ahead(66 * G, 56 * G, 0), 6 * G);
        // The window's own deposits are left out: the same window.
        assert_eq!(adaptive_ahead(66 * G, 59 * G, 3 * G), 6 * G);
        // No room above the resident set, or above the peak: the floor.
        assert_eq!(adaptive_ahead(60 * G, 58 * G, 0), AHEAD_FLOOR);
        assert_eq!(adaptive_ahead(60 * G, 61 * G, 0), AHEAD_FLOOR);
        assert_eq!(adaptive_ahead(0, 0, 0), AHEAD_FLOOR);
        // Exactly at either bound.
        assert_eq!(adaptive_ahead(30 * G, 22 * G, 0), AHEAD_FLOOR);
        assert_eq!(adaptive_ahead(30 * G, 10 * G, 0), AHEAD_CAP);
        assert_eq!(adaptive_ahead(u64::MAX, 0, u64::MAX), AHEAD_CAP);
    }

    /// Above the floor, the window and the resident set without it never
    /// pass the peak less the margin: the window holds only memory the run has
    /// already reached.
    #[test]
    fn the_adaptive_window_never_passes_the_peak() {
        const G: u64 = 1 << 30;
        for hwm in (0..=120).step_by(3).map(|g| g * G) {
            for rss in (0..=120).step_by(5).map(|g| g * G) {
                for parked in [0, G, 3 * G, 16 * G] {
                    let ahead = adaptive_ahead(hwm, rss, parked);
                    assert!((AHEAD_FLOOR..=AHEAD_CAP).contains(&ahead));
                    if ahead > AHEAD_FLOOR {
                        let base = rss.saturating_sub(parked);
                        assert!(base + ahead + AHEAD_MARGIN <= hwm, "{hwm} {rss} {parked}");
                    }
                }
            }
        }
    }

    /// The pacer moves the window in whole steps, and only once the reading is
    /// a step away: a reading that churns inside a step (a chunk taken or
    /// deposited) moves nothing, even across a whole GiB; a drift moves it a
    /// step at a time; the window never stands a step or more above the reading.
    #[test]
    fn the_pacer_steps_past_the_window_churn() {
        const G: u64 = 1 << 30;
        assert_eq!(paced_ahead(4 * G, 16 * G), 16 * G);
        assert_eq!(paced_ahead(16 * G, 4 * G), 4 * G);
        assert_eq!(paced_ahead(6 * G, 6 * G + G / 2), 6 * G, "inside a step up");
        assert_eq!(
            paced_ahead(6 * G, 6 * G - G / 2),
            6 * G,
            "inside a step down"
        );
        assert_eq!(paced_ahead(6 * G, 5 * G), 5 * G, "a step down");
        assert_eq!(paced_ahead(6 * G, 7 * G + G / 3), 7 * G, "a step up, whole");
        // Churn of ± 0.3 GiB around 6.5 and around 6.0 GiB (across a whole
        // GiB): one or two moves up from 4 GiB, then none.
        for centre in [6 * G + G / 2, 6 * G] {
            let mut now = 4 * G;
            let mut moves = 0;
            for k in 0..400u64 {
                let reading = centre + (k % 7) * G / 10 - 3 * G / 10;
                let next = paced_ahead(now, reading);
                moves += usize::from(next != now);
                now = next;
                assert!(now < reading + PACE_STEP, "{now} over {reading}");
            }
            assert!((1..=2).contains(&moves), "{moves} moves around {centre}");
        }
        // A drift from 16 to 4 GiB in steps of 0.1 GiB: twelve moves down.
        let (mut now, mut shrank) = (16 * G, 0);
        for k in 0..=120u64 {
            let reading = (16 * G).saturating_sub(k * G / 10).max(4 * G);
            let next = paced_ahead(now, reading);
            shrank += usize::from(next < now);
            assert!(next <= now);
            now = next;
            assert!(now < reading + PACE_STEP);
        }
        assert_eq!((now, shrank), (4 * G, 12));
    }

    /// `LAMBDA_VM_BLOCK_REGEN_AHEAD_GIB` fixes the window (RYZEN 052's arms);
    /// unset or unreadable, it is adaptive, from the floor.
    #[test]
    fn the_window_knob_overrides_the_adaptive_window() {
        const G: u64 = 1 << 30;
        assert_eq!(ahead_policy_from(Some("04")), AheadPolicy::Fixed(4 * G));
        assert_eq!(ahead_policy_from(Some(" 32 ")), AheadPolicy::Fixed(32 * G));
        assert_eq!(ahead_policy_from(Some("0.5")), AheadPolicy::Fixed(G / 2));
        assert_eq!(ahead_policy_from(None), AheadPolicy::Adaptive);
        for bad in ["", "x", "-1", "nan", "inf"] {
            assert_eq!(
                ahead_policy_from(Some(bad)),
                AheadPolicy::Adaptive,
                "{bad:?}"
            );
        }
        assert_eq!(AheadPolicy::Fixed(16 * G).initial(), 16 * G);
        assert_eq!(AheadPolicy::Adaptive.initial(), AHEAD_FLOOR);
        assert!(pacer_line(AheadPolicy::Fixed(16 * G), None).contains("fixed 16.00 GiB"));
        let r = PacerReport {
            first: Some(4 * G),
            most: 16 * G,
            last: 16 * G,
            readings: 9,
            grew: 3,
            shrank: 1,
        };
        let line = pacer_line(AheadPolicy::Adaptive, Some(&r));
        assert!(
            line.contains(
                "first 4.00 GiB · most 16.00 GiB · last 16.00 GiB · 9 readings · grew 3 times · \
                 shrank 1 times"
            ),
            "{line}"
        );
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
