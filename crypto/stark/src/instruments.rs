use std::cell::RefCell;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// Wall clock span timeline: the trustworthy per step latency breakdown.
//
// Spans open and close on the main thread at phase boundaries. They do not
// overlap and sum to their parent, so the tree is a true latency breakdown
// (unlike the accum_* thread local sub timers below, which sum per worker CPU
// time across rayon threads and can exceed 100%). A parallel region is one span
// around the blocking call; its internal split is reported separately as CPU
// time, never mixed into the wall tree.
//
//     let _s = instruments::span("trace_build");   // RAII, stops on drop
//
// Instant::now() is about 20 ns, fine at phase granularity, not in per op loops.

#[derive(Clone, Debug)]
pub struct SpanRecord {
    pub label: &'static str,
    pub depth: u16,
    pub wall: Duration,
    /// Open-order, so the tree reconstructs in start-order (records push on close).
    pub order: u32,
    /// Wall clock epoch (ns) when the span opened, for aligning with external
    /// samplers (e.g. nvidia-smi GPU util) to attribute device busy time per step.
    pub start_ns: u128,
}

static TIMELINE: Mutex<Vec<SpanRecord>> = Mutex::new(Vec::new());
static SPAN_ORDER: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static SPAN_DEPTH: std::cell::Cell<u16> = const { std::cell::Cell::new(0) };
}

#[must_use]
pub struct SpanGuard {
    label: &'static str,
    depth: u16,
    order: u32,
    start: Instant,
    start_ns: u128,
}

/// Open a wall-clock span; records elapsed time when the guard drops.
pub fn span(label: &'static str) -> SpanGuard {
    let depth = SPAN_DEPTH.with(|d| {
        let v = d.get();
        d.set(v + 1);
        v
    });
    let order = SPAN_ORDER.fetch_add(1, Ordering::Relaxed) as u32;
    let start_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    SpanGuard {
        label,
        depth,
        order,
        start: Instant::now(),
        start_ns,
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        let wall = self.start.elapsed();
        SPAN_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        if let Ok(mut t) = TIMELINE.lock() {
            t.push(SpanRecord {
                label: self.label,
                depth: self.depth,
                wall,
                order: self.order,
                start_ns: self.start_ns,
            });
        }
    }
}

/// Clear recorded spans. Call at the start of a measured prove.
pub fn reset_timeline() {
    SPAN_ORDER.store(0, Ordering::Relaxed);
    SPAN_DEPTH.with(|d| d.set(0));
    if let Ok(mut t) = TIMELINE.lock() {
        t.clear();
    }
}

/// Drain recorded spans, sorted in start-order (ready for the tree).
pub fn take_timeline() -> Vec<SpanRecord> {
    let mut spans = TIMELINE
        .lock()
        .map(|mut t| std::mem::take(&mut *t))
        .unwrap_or_default();
    spans.sort_by_key(|s| s.order);
    spans
}

/// Indented wall-clock tree with % of the root span.
pub fn format_timeline(spans: &[SpanRecord]) -> String {
    use std::fmt::Write;
    if spans.is_empty() {
        return String::new();
    }
    let total_s = spans
        .first()
        .map(|s| s.wall.as_secs_f64())
        .unwrap_or(1e-9)
        .max(1e-9);
    let mut out = String::from("=== TIMELINE (wall-clock) ===\n");
    for s in spans {
        let indent = "  ".repeat(s.depth as usize);
        let pct = 100.0 * s.wall.as_secs_f64() / total_s;
        let _ = writeln!(
            out,
            "{:<42} {:>10.3?} {:>6.1}%",
            format!("{indent}{}", s.label),
            s.wall,
            pct
        );
    }
    out
}

/// JSON array of `{label, depth, wall_ns, order}` for diffing / plotting.
pub fn timeline_json(spans: &[SpanRecord]) -> String {
    let mut out = String::from("[");
    for (i, s) in spans.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        // Escape the label so a quote or backslash cannot break the JSON.
        let label = s.label.replace('\\', "\\\\").replace('"', "\\\"");
        out.push_str(&format!(
            "{{\"label\":\"{}\",\"depth\":{},\"wall_ns\":{},\"order\":{},\"start_ns\":{}}}",
            label,
            s.depth,
            s.wall.as_nanos(),
            s.order,
            s.start_ns
        ));
    }
    out.push(']');
    out
}

static HEAP_READER: OnceLock<fn() -> Option<usize>> = OnceLock::new();

pub fn set_heap_reader(f: fn() -> Option<usize>) {
    let _ = HEAP_READER.set(f);
}

pub fn heap_bytes() -> Option<usize> {
    HEAP_READER.get().and_then(|f| f())
}

pub type HeapSnapshot = (&'static str, usize);

pub fn snap(label: &'static str) -> Option<HeapSnapshot> {
    heap_bytes().map(|b| (label, b))
}

pub struct ProveHeapProfile {
    pub before: Option<usize>,
    pub after_execute: Option<usize>,
    pub after_trace_build: Option<usize>,
    pub after_air: Option<usize>,
}

/// Sub-operation timing breakdown for a single table in Rounds 2-4.
#[derive(Clone, Debug, Default)]
pub struct TableSubOps {
    /// evaluator.evaluate()
    pub constraints: Duration,
    /// decompose_and_extend_d2
    pub comp_decompose: Duration,
    /// commit_bit_reversed (composition-polynomial commit step)
    pub comp_commit: Duration,
    /// Round 3: barycentric OOD evaluation
    pub ood: Duration,
    /// Round 4: compute_deep_composition_poly_evaluations
    pub deep_comp: Duration,
    /// Round 4: serial CPU bit-reverse between GPU DEEP and GPU FRI.
    pub deep_extend: Duration,
    /// fri::commit_phase_from_evaluations
    pub fri_commit: Duration,
    /// Round 4: proof-of-work nonce search.
    pub grinding: Duration,
    /// Round 4: query-index sampling and FRI-layer decommitment.
    pub fri_query: Duration,
    /// Round 4: trace/composition Merkle openings.
    pub openings: Duration,
}

/// Sub-operation breakdown for Round 1 aux commit pass.
#[derive(Clone, Debug, Default)]
pub struct Round1SubOps {
    /// Main trace: expand_columns_to_lde (LDE/FFT)
    pub main_lde: Duration,
    /// Main trace: commit_bit_reversed (Merkle)
    pub main_merkle: Duration,
    /// Aux trace: expand_columns_to_lde (LDE/FFT)
    pub aux_lde: Duration,
    /// Aux trace: commit_bit_reversed (Merkle)
    pub aux_merkle: Duration,
    /// Aux build: LogUp fingerprint computation (CPU).
    pub aux_fingerprint: Duration,
    /// Aux build: fingerprint batch inverse (CPU).
    pub aux_invert: Duration,
    /// Aux build: term combine (CPU).
    pub aux_term: Duration,
    /// Aux build: accumulated-column running sum (CPU).
    pub aux_accumulate: Duration,
}

/// Timing data collected inside `multi_prove`.
pub struct MultiProveTiming {
    pub prepass: Duration,
    pub main_commits: Duration,
    pub aux_build: Duration,
    pub aux_commit: Duration,
    pub rounds_2_4: Duration,
    /// Sub-op breakdown for Round 1 (main + aux LDE vs Merkle).
    pub round1_sub: Round1SubOps,
    /// (name, rows, duration, sub_ops) per table for rounds 2-4.
    pub table_timings: Vec<(String, usize, Duration, TableSubOps)>,
    pub heap_snapshots: Vec<HeapSnapshot>,
}

/// Round 1 sub-timings: atomics so parallel rayon workers can accumulate safely.
static R1_MAIN_LDE_US: AtomicU64 = AtomicU64::new(0);
static R1_MAIN_MERKLE_US: AtomicU64 = AtomicU64::new(0);
static R1_AUX_LDE_US: AtomicU64 = AtomicU64::new(0);
static R1_AUX_MERKLE_US: AtomicU64 = AtomicU64::new(0);
// Aux build (LogUp) sub-phases, CPU time accumulated across tables/chunks.
static AUX_FINGERPRINT_US: AtomicU64 = AtomicU64::new(0);
static AUX_INVERT_US: AtomicU64 = AtomicU64::new(0);
static AUX_TERM_US: AtomicU64 = AtomicU64::new(0);
static AUX_ACCUM_US: AtomicU64 = AtomicU64::new(0);

type R4SubDurations = (Duration, Duration, Duration, Duration, Duration, Duration);

thread_local! {
    static TIMING_DATA: RefCell<Option<MultiProveTiming>> = const { RefCell::new(None) };
    /// Round 2 sub-timings: (constraints, fft, merkle)
    static R2_SUB: RefCell<Option<(Duration, Duration, Duration)>> = const { RefCell::new(None) };
    /// Round 4 sub-timings: (bit_reverse, fri_commit, deep_comp, grinding,
    /// fri_query, openings).
    static R4_SUB: RefCell<Option<R4SubDurations>> = const { RefCell::new(None) };
    /// Assembled sub-ops from prove_rounds_2_to_4 (without reconstruct_round1 LDE time).
    static ROUND_SUB_OPS: RefCell<Option<TableSubOps>> = const { RefCell::new(None) };
}

pub fn store(data: MultiProveTiming) {
    TIMING_DATA.with(|cell| {
        *cell.borrow_mut() = Some(data);
    });
}

pub fn take() -> Option<MultiProveTiming> {
    TIMING_DATA.with(|cell| cell.borrow_mut().take())
}

pub fn accum_r1_main(lde: Duration, merkle: Duration) {
    R1_MAIN_LDE_US.fetch_add(lde.as_micros() as u64, Ordering::Relaxed);
    R1_MAIN_MERKLE_US.fetch_add(merkle.as_micros() as u64, Ordering::Relaxed);
}

pub fn accum_r1_aux(lde: Duration, merkle: Duration) {
    R1_AUX_LDE_US.fetch_add(lde.as_micros() as u64, Ordering::Relaxed);
    R1_AUX_MERKLE_US.fetch_add(merkle.as_micros() as u64, Ordering::Relaxed);
}

/// Aux build (LogUp term column) sub-phase CPU times, summed across chunks.
pub fn accum_aux_term(fingerprint: Duration, invert: Duration, term: Duration) {
    AUX_FINGERPRINT_US.fetch_add(fingerprint.as_micros() as u64, Ordering::Relaxed);
    AUX_INVERT_US.fetch_add(invert.as_micros() as u64, Ordering::Relaxed);
    AUX_TERM_US.fetch_add(term.as_micros() as u64, Ordering::Relaxed);
}

/// Aux build accumulated-column (running sum) CPU time.
pub fn accum_aux_accumulate(d: Duration) {
    AUX_ACCUM_US.fetch_add(d.as_micros() as u64, Ordering::Relaxed);
}

pub fn take_r1_sub() -> Round1SubOps {
    Round1SubOps {
        main_lde: Duration::from_micros(R1_MAIN_LDE_US.swap(0, Ordering::Relaxed)),
        main_merkle: Duration::from_micros(R1_MAIN_MERKLE_US.swap(0, Ordering::Relaxed)),
        aux_lde: Duration::from_micros(R1_AUX_LDE_US.swap(0, Ordering::Relaxed)),
        aux_merkle: Duration::from_micros(R1_AUX_MERKLE_US.swap(0, Ordering::Relaxed)),
        aux_fingerprint: Duration::from_micros(AUX_FINGERPRINT_US.swap(0, Ordering::Relaxed)),
        aux_invert: Duration::from_micros(AUX_INVERT_US.swap(0, Ordering::Relaxed)),
        aux_term: Duration::from_micros(AUX_TERM_US.swap(0, Ordering::Relaxed)),
        aux_accumulate: Duration::from_micros(AUX_ACCUM_US.swap(0, Ordering::Relaxed)),
    }
}

/// Reset all instrument state. Call at the start of `multi_prove` to avoid
/// stale data from a previous run in the same process.
///
/// Note: thread local stores (R2_SUB, R4_SUB, ROUND_SUB_OPS) are only cleared
/// for the calling thread. Rayon worker threads are not reset, so stale data is
/// possible if a previous run panicked without consuming stored values.
/// In practice this is safe because store/take pairs always execute within the
/// same rayon task closure.
pub fn reset_all() {
    R1_MAIN_LDE_US.store(0, Ordering::Relaxed);
    R1_MAIN_MERKLE_US.store(0, Ordering::Relaxed);
    R1_AUX_LDE_US.store(0, Ordering::Relaxed);
    R1_AUX_MERKLE_US.store(0, Ordering::Relaxed);
    AUX_FINGERPRINT_US.store(0, Ordering::Relaxed);
    AUX_INVERT_US.store(0, Ordering::Relaxed);
    AUX_TERM_US.store(0, Ordering::Relaxed);
    AUX_ACCUM_US.store(0, Ordering::Relaxed);
    TIMING_DATA.with(|cell| {
        cell.borrow_mut().take();
    });
    R2_SUB.with(|cell| {
        cell.borrow_mut().take();
    });
    R4_SUB.with(|cell| {
        cell.borrow_mut().take();
    });
    ROUND_SUB_OPS.with(|cell| {
        cell.borrow_mut().take();
    });
}

pub fn store_r2_sub(constraints: Duration, fft: Duration, merkle: Duration) {
    R2_SUB.with(|cell| *cell.borrow_mut() = Some((constraints, fft, merkle)));
}

pub fn take_r2_sub() -> Option<(Duration, Duration, Duration)> {
    R2_SUB.with(|cell| cell.borrow_mut().take())
}

pub fn store_r4_sub(
    bit_reverse: Duration,
    fri_commit: Duration,
    deep_comp: Duration,
    grinding: Duration,
    fri_query: Duration,
    openings: Duration,
) {
    R4_SUB.with(|cell| {
        *cell.borrow_mut() = Some((
            bit_reverse,
            fri_commit,
            deep_comp,
            grinding,
            fri_query,
            openings,
        ));
    });
}

pub fn take_r4_sub() -> Option<(Duration, Duration, Duration, Duration, Duration, Duration)> {
    R4_SUB.with(|cell| cell.borrow_mut().take())
}

pub fn store_round_sub_ops(data: TableSubOps) {
    ROUND_SUB_OPS.with(|cell| {
        *cell.borrow_mut() = Some(data);
    });
}

pub fn take_round_sub_ops() -> Option<TableSubOps> {
    ROUND_SUB_OPS.with(|cell| cell.borrow_mut().take())
}

// ── Concurrency-aware interval recorder (rounds_2to4 attribution) ────────────
//
// The `span()` timeline above is main-thread, nested, non-overlapping — a true
// latency tree. The `store_r2_sub`/`store_r4_sub` sub-timers, by contrast, are
// summed across rayon workers ("work sum") and can exceed wall, so they cannot
// tell how much wall an op actually occupies.
//
// This recorder closes that gap. Rounds 2-4 run one table per rayon worker in
// parallel; each sub-op wraps its region with `iv_now()` / `iv_push(op, start)`.
// Timestamps are ns from a single monotonic `Instant` epoch, comparable across
// threads. From the resulting intervals we report, per op:
//   * work_sum   — Σ(end-start): total worker time (can exceed wall)
//   * union_wall — length of the temporal union of intervals: the wall the op
//                  really occupies (overlapping tables merged) ← decision metric
//   * concurrency— max simultaneous intervals (how parallel the op ran)
// A Chrome-Trace/Perfetto JSON (one lane per worker) is emitted when the caller
// asks for it (GPU_INTERVAL_TRACE=1), for visual inspection.

#[derive(Clone, Copy)]
pub struct IntervalRecord {
    pub op: &'static str,
    pub start_ns: u64,
    pub end_ns: u64,
    /// Rayon worker slot — a stable proxy for the table processed in parallel.
    pub lane: u64,
}

static INTERVAL_EPOCH: Mutex<Option<Instant>> = Mutex::new(None);
static INTERVALS: Mutex<Vec<IntervalRecord>> = Mutex::new(Vec::new());
static NEXT_LANE: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static LANE: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

fn lane() -> u64 {
    LANE.with(|c| {
        c.get().unwrap_or_else(|| {
            let v = NEXT_LANE.fetch_add(1, Ordering::Relaxed);
            c.set(Some(v));
            v
        })
    })
}

/// Start a fresh interval epoch and drop any prior records. Call once per prove.
pub fn reset_intervals() {
    if let Ok(mut e) = INTERVAL_EPOCH.lock() {
        *e = Some(Instant::now());
    }
    if let Ok(mut v) = INTERVALS.lock() {
        v.clear();
    }
}

/// Nanoseconds since the interval epoch (0 if none set — recorder disabled).
fn iv_now() -> u64 {
    INTERVAL_EPOCH
        .lock()
        .ok()
        .and_then(|e| *e)
        .map(|e| e.elapsed().as_nanos() as u64)
        .unwrap_or(0)
}

/// Record an interval ending ~now with the measured `dur`. Pairs with the
/// existing `let x_dur = t_sub.elapsed();` sub-timers (one line per op, anchored
/// on the unique duration variable — no need to touch the ambiguous start line).
pub fn iv_push_dur(op: &'static str, dur: Duration) {
    let end_ns = iv_now();
    let start_ns = end_ns.saturating_sub(dur.as_nanos() as u64);
    let lane = lane();
    if let Ok(mut v) = INTERVALS.lock() {
        v.push(IntervalRecord {
            op,
            start_ns,
            end_ns,
            lane,
        });
    }
}

/// Drain the recorded intervals.
pub fn take_intervals() -> Vec<IntervalRecord> {
    INTERVALS
        .lock()
        .map(|mut v| std::mem::take(&mut *v))
        .unwrap_or_default()
}

/// Union length (s) and max concurrency of a set of `[start,end)` intervals,
/// via a sweep line. At equal timestamps, opens are processed before closes so
/// touching intervals merge (union) and count as overlapping (concurrency).
fn union_and_concurrency(intervals: &[(u64, u64)]) -> (f64, usize) {
    let mut evts: Vec<(u64, i32)> = Vec::with_capacity(intervals.len() * 2);
    for &(s, e) in intervals {
        if e > s {
            evts.push((s, 1));
            evts.push((e, -1));
        }
    }
    evts.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    let (mut depth, mut max_c, mut union, mut cur_start) = (0i32, 0i32, 0u64, 0u64);
    for (t, d) in evts {
        if depth == 0 && d == 1 {
            cur_start = t;
        }
        depth += d;
        max_c = max_c.max(depth);
        if depth == 0 {
            union += t.saturating_sub(cur_start);
        }
    }
    (union as f64 / 1e9, max_c as usize)
}

/// Per-op attribution table with three wall metrics that bracket the true
/// recoverable wall (ops overlap across tables, so no single number is exact):
///   * union_wall  — wall where ≥1 interval of the op is active (UPPER bound)
///   * weighted    — each instant's wall split evenly among all active
///                   intervals; Σ over ops = span. This is an ACTIVITY-share
///                   heuristic, NOT literal recoverable wall: removing an op need
///                   not recover its share (freed capacity may be refilled, or
///                   the op may never be the critical path).
///   * exclusive   — wall where ONLY this op is active anywhere (LOWER bound)
/// Sorted by `weighted` desc. Also reports peak/avg global concurrency. Note the
/// concurrency here counts instrumented intervals (GPU waits included), so it is
/// NOT a measure of CPU-core utilization.
pub fn format_intervals(records: &[IntervalRecord]) -> String {
    use std::collections::HashMap;
    use std::fmt::Write;
    if records.is_empty() {
        return String::new();
    }

    // Stable op indexing (first-seen order).
    let mut op_idx: HashMap<&'static str, usize> = HashMap::new();
    let mut ops: Vec<&'static str> = Vec::new();
    for r in records {
        if !op_idx.contains_key(r.op) {
            op_idx.insert(r.op, ops.len());
            ops.push(r.op);
        }
    }
    let n_ops = ops.len();

    // Per-op interval lists (for union + work_sum + count).
    let mut per_op: Vec<Vec<(u64, u64)>> = vec![Vec::new(); n_ops];
    for r in records {
        if r.end_ns > r.start_ns {
            per_op[op_idx[r.op]].push((r.start_ns, r.end_ns));
        }
    }

    // Global sweep for weighted, exclusive, and concurrency.
    let mut evts: Vec<(u64, i32, usize)> = Vec::with_capacity(records.len() * 2);
    for r in records {
        if r.end_ns > r.start_ns {
            let oi = op_idx[r.op];
            evts.push((r.start_ns, 1, oi));
            evts.push((r.end_ns, -1, oi));
        }
    }
    evts.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));

    let mut weighted = vec![0f64; n_ops];
    let mut exclusive = vec![0f64; n_ops];
    let mut active = vec![0i32; n_ops];
    let mut total = 0i32;
    let mut peak = 0i32;
    let mut conc_area = 0f64; // Σ dt*total  → avg concurrency = conc_area/span
    let mut prev_t = evts.first().map(|e| e.0).unwrap_or(0);
    for (t, d, oi) in evts {
        if t > prev_t && total > 0 {
            let dt = (t - prev_t) as f64;
            for (op, &a) in active.iter().enumerate() {
                if a > 0 {
                    weighted[op] += dt * a as f64 / total as f64;
                }
            }
            if total == 1 {
                if let Some(op) = active.iter().position(|&a| a > 0) {
                    exclusive[op] += dt;
                }
            }
            conc_area += dt * total as f64;
        }
        active[oi] += d;
        total += d;
        peak = peak.max(total);
        prev_t = t;
    }

    let span_lo = records.iter().map(|r| r.start_ns).min().unwrap_or(0);
    let span_hi = records.iter().map(|r| r.end_ns).max().unwrap_or(0);
    let span_s = span_hi.saturating_sub(span_lo) as f64 / 1e9;

    let mut rows: Vec<(&'static str, f64, f64, f64, f64, usize, usize)> = (0..n_ops)
        .map(|i| {
            let work_sum = per_op[i]
                .iter()
                .map(|&(s, e)| e.saturating_sub(s))
                .sum::<u64>() as f64
                / 1e9;
            let (union, conc) = union_and_concurrency(&per_op[i]);
            (
                ops[i],
                work_sum,
                union,
                weighted[i] / 1e9,
                exclusive[i] / 1e9,
                conc,
                per_op[i].len(),
            )
        })
        .collect();
    rows.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

    let avg_conc = if span_s > 0.0 {
        conc_area / 1e9 / span_s
    } else {
        0.0
    };
    let mut out = String::from("=== INTERVAL ATTRIBUTION (rounds_2to4, concurrency-aware) ===\n");
    let _ = writeln!(
        out,
        "  {:<24} {:>9} {:>10} {:>10} {:>10} {:>5} {:>4}",
        "op", "work_sum", "union", "weighted", "exclusive", "conc", "n"
    );
    let (mut sum_union, mut sum_weighted) = (0.0, 0.0);
    for (op, ws, uw, ww, ex, conc, n) in &rows {
        sum_union += uw;
        sum_weighted += ww;
        let _ = writeln!(
            out,
            "  {op:<24} {ws:>8.3}s {uw:>9.3}s {ww:>9.3}s {ex:>9.3}s {conc:>5} {n:>4}"
        );
    }
    let _ = writeln!(
        out,
        "  span {span_s:.3}s (≈ rounds_2to4) · Σweighted {sum_weighted:.3}s (=span, true decomposition) \
         · Σunion {sum_union:.3}s · concurrency peak {peak} avg {avg_conc:.1}"
    );
    let _ = writeln!(
        out,
        "  → weighted = activity-share heuristic (Σ=span), NOT literal recoverable wall; \
         union=upper bound, exclusive=lower bound. concurrency = instrumented intervals \
         (GPU waits included), not CPU-core utilization."
    );
    out
}

/// Chrome-Trace / Perfetto JSON: one complete ("X") event per interval, µs
/// units, `tid` = worker lane. Load in chrome://tracing or ui.perfetto.dev.
pub fn intervals_perfetto_json(records: &[IntervalRecord]) -> String {
    let mut out = String::from("[");
    for (i, r) in records.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let name = r.op.replace('\\', "\\\\").replace('"', "\\\"");
        let ts = r.start_ns as f64 / 1000.0;
        let dur = r.end_ns.saturating_sub(r.start_ns) as f64 / 1000.0;
        out.push_str(&format!(
            "{{\"name\":\"{name}\",\"ph\":\"X\",\"ts\":{ts:.3},\"dur\":{dur:.3},\"pid\":1,\"tid\":{},\"cat\":\"r2to4\"}}",
            r.lane
        ));
    }
    out.push(']');
    out
}
