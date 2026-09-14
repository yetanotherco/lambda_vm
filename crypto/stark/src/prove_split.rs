//! One `PROVE SPLIT:` line per `multi_prove`, behind `LFM_PROVE_SPLIT=1`.
//!
//! # Why this exists next to `instruments`
//!
//! [`crate::instruments`] already carries a finer breakdown of the same prove —
//! and it is compiled out unless `--features instruments` is named. That is the
//! wrong shape for this question. The number we need to attribute is the block
//! record, and the record is produced by a binary that does NOT enable the
//! feature; a figure taken from a differently-compiled binary answers a
//! different run. ⇒ this module is **always compiled and runtime-gated**, so the
//! split can be read off the record build itself, with the knob unset for the
//! record and set for the profile.
//!
//! It does not replace `instruments`: the spans there nest and reconstruct a
//! tree, which this does not attempt. This is one flat line per prove.
//!
//! # What the numbers mean
//!
//! Two kinds of field, and mixing them is the way to misread the line:
//!
//! - **Phase walls** (`prepass`, `main_commit`, `absorb`, `fused`) are measured
//!   on the calling thread at phase boundaries. They do not overlap and they sum
//!   to the prove.
//! - **Per-table sums** (everything inside `fused`) are accumulated from up to
//!   `table_parallelism(num_airs)` driver threads. They are **worker-seconds,
//!   not wall**, and their total runs up to `k` times over `fused`.
//!
//! The line says which is which by printing the per-table group under a
//! `tables[Σ]` prefix.
//!
//! # Cost when disabled
//!
//! [`enabled`] is a `OnceLock<bool>` load and a predictable branch; [`mark`]
//! returns `None` and no `Instant` is read. Every call site is at phase
//! granularity — tens of calls per table per prove, never inside a per-row loop
//! — so even enabled the instrumentation is far below the noise of the phases it
//! measures.
//!
//! # ⚠ One prove at a time
//!
//! The accumulators are process-global (the fused region's drivers are rayon
//! workers, so a thread-local cannot see them). Two `multi_prove` calls in
//! flight would therefore mix their numbers. On the LFM tree that cannot happen
//! — `lfm::device_permit` holds the card across `multi_prove` — and in the base
//! there is a single prover thread. Rather than assume it, [`begin`] counts
//! concurrent proves and the printed line carries a loud `OVERLAPPED` marker if
//! one was ever seen, so a mixed reading says so instead of looking clean.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// `LFM_PROVE_SPLIT=1` (any non-empty value other than `0`) turns the line on.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("LFM_PROVE_SPLIT") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    })
}

/// Unix epoch seconds, for aligning a phase with an external sampler
/// (`nvidia-smi --query-gpu=... --format=csv` carries wall-clock timestamps).
///
/// Printed rather than a process-relative offset on purpose: a relative stamp
/// needs the reader to know the process start, and the two logs being aligned
/// are written by different processes.
pub fn epoch_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_default()
}

/// A nanosecond accumulator. Public so call sites name their slot as a
/// constant rather than passing an index.
#[derive(Debug)]
pub struct Slot(AtomicU64);

impl Slot {
    const fn new() -> Self {
        Self(AtomicU64::new(0))
    }
    fn take(&self) -> f64 {
        self.0.swap(0, Ordering::Relaxed) as f64 / 1e9
    }
}

/// Start a timed region — `None`, and no clock read, when the knob is off.
#[inline]
pub fn mark() -> Option<Instant> {
    enabled().then(Instant::now)
}

/// Close a region opened by [`mark`] into `slot`.
#[inline]
pub fn add(slot: &Slot, start: Option<Instant>) {
    if let Some(t) = start {
        slot.0
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

// ── phase walls, one per prove, measured on the calling thread ──────────────
/// Domains, twiddles, walk orders and the VRAM gate.
pub static PREPASS: Slot = Slot::new();
/// Round 1 main commits: the admitted device region, every table.
pub static MAIN_COMMIT: Slot = Slot::new();
/// Host-only: the sequential absorption of the main roots into the transcript,
/// plus the LogUp challenge sampling that follows it. The one Fiat-Shamir
/// barrier between the main commits and the fused region.
pub static MAIN_ABSORB: Slot = Slot::new();
/// The fused per-table region's wall: aux build, aux commit, rounds 2-4.
pub static FUSED: Slot = Slot::new();

// ── per-table sums inside `fused`: worker-seconds across `k` drivers ────────
/// LogUp auxiliary trace construction.
pub static AUX_BUILD: Slot = Slot::new();
/// Auxiliary trace LDE + Merkle commit.
pub static AUX_COMMIT: Slot = Slot::new();
/// `build_round1` and the bus-contribution absorption.
pub static R1_ASSEMBLE: Slot = Slot::new();
/// Round 2: constraint evaluation over the LDE.
pub static R2_CONSTRAINTS: Slot = Slot::new();
/// Round 2: composition-polynomial decompose/extend.
pub static R2_DECOMPOSE: Slot = Slot::new();
/// Round 2: composition-polynomial Merkle commit.
pub static R2_COMMIT: Slot = Slot::new();
/// Round 3: out-of-domain evaluation.
pub static R3_OOD: Slot = Slot::new();
/// Host-only: absorbing the OOD values into the table's transcript fork.
pub static R3_ABSORB: Slot = Slot::new();
/// Round 4: DEEP composition + the FRI commit phase.
pub static R4_DEEP_FRI: Slot = Slot::new();
/// Round 4: the proof-of-work grind.
pub static R4_GRIND: Slot = Slot::new();
/// Round 4: query sampling, the FRI query phase and the DEEP openings.
pub static R4_QUERIES: Slot = Slot::new();

/// Proves that have started; the sequence number the line carries.
static SEQ: AtomicUsize = AtomicUsize::new(0);
/// Proves inside `multi_prove` right now — the overlap falsifier.
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
/// Set once if two proves were ever concurrent. Never cleared: one mixed
/// reading taints every later line, because a slot it polluted is only zeroed
/// by the print that reports it.
static OVERLAPPED: AtomicUsize = AtomicUsize::new(0);

/// What [`begin`] hands back, so [`report`] can price the whole prove without
/// the caller threading a second timer.
///
/// ⛔ It releases the concurrency count on DROP, not in [`report`]. `multi_prove`
/// has `?` early-returns, and a release that only ran on the success path would
/// leave the count stuck at one forever — every later line would then be stamped
/// `OVERLAPPED` by a prove that failed rather than by two that overlapped. A
/// false alarm on the falsifier is worse than no falsifier, because it reads as
/// evidence.
#[derive(Debug)]
pub struct ProveMark {
    seq: usize,
    start: Instant,
    start_epoch: f64,
}

impl Drop for ProveMark {
    fn drop(&mut self) {
        IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Open a prove. Cheap and inert when the knob is off.
pub fn begin() -> Option<ProveMark> {
    if !enabled() {
        return None;
    }
    if IN_FLIGHT.fetch_add(1, Ordering::SeqCst) + 1 > 1 {
        OVERLAPPED.store(1, Ordering::Relaxed);
    }
    Some(ProveMark {
        seq: SEQ.fetch_add(1, Ordering::Relaxed),
        start: Instant::now(),
        start_epoch: epoch_secs(),
    })
}

/// Close a prove and return its line, or `None` when the knob is off.
///
/// Reading a slot CLEARS it, so a caller that drops the line silently would
/// hand this prove's time to the next one. There is exactly one caller and it
/// prints.
pub fn report(m: Option<ProveMark>, num_airs: usize, total_rows: usize) -> Option<String> {
    let m = m?;
    let wall = m.start.elapsed().as_secs_f64();
    let end_epoch = epoch_secs();

    let prepass = PREPASS.take();
    let main_commit = MAIN_COMMIT.take();
    let absorb = MAIN_ABSORB.take();
    let fused = FUSED.take();

    let aux_build = AUX_BUILD.take();
    let aux_commit = AUX_COMMIT.take();
    let r1_assemble = R1_ASSEMBLE.take();
    let r2_constraints = R2_CONSTRAINTS.take();
    let r2_decompose = R2_DECOMPOSE.take();
    let r2_commit = R2_COMMIT.take();
    let r3_ood = R3_OOD.take();
    let r3_absorb = R3_ABSORB.take();
    let r4_deep_fri = R4_DEEP_FRI.take();
    let r4_grind = R4_GRIND.take();
    let r4_queries = R4_QUERIES.take();

    let table_sum = aux_build
        + aux_commit
        + r1_assemble
        + r2_constraints
        + r2_decompose
        + r2_commit
        + r3_ood
        + r3_absorb
        + r4_deep_fri
        + r4_grind
        + r4_queries;
    // What the four phase walls leave over: the sequential prep between them
    // (transcript forking, the per-table slot vectors, the shape and weight
    // walks that are not inside `prepass`). Named rather than left implicit —
    // an unattributed remainder is how a phase hides.
    let other = (wall - prepass - main_commit - absorb - fused).max(0.0);

    Some(format!(
        "PROVE SPLIT #{seq}{tainted}: airs {num_airs} · rows {total_rows} · \
         wall {wall:.2}s · t=[{t0:.3},{t1:.3}] · \
         prepass {prepass:.2} · main_commit {main_commit:.2} · absorb {absorb:.3} · \
         fused {fused:.2} · other {other:.2} || \
         tables[Σ] aux_build {aux_build:.2} · aux_commit {aux_commit:.2} · \
         r1_assemble {r1_assemble:.2} · r2_constraints {r2_constraints:.2} · \
         r2_decompose {r2_decompose:.2} · r2_commit {r2_commit:.2} · \
         r3_ood {r3_ood:.2} · r3_absorb {r3_absorb:.3} · r4_deep_fri {r4_deep_fri:.2} · \
         r4_grind {r4_grind:.2} · r4_queries {r4_queries:.2} · Σ {table_sum:.2}",
        seq = m.seq,
        tainted = if OVERLAPPED.load(Ordering::Relaxed) == 0 {
            ""
        } else {
            " ⛔OVERLAPPED"
        },
        t0 = m.start_epoch,
        t1 = end_epoch,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⛔ The disabled path must read no clock. Asserted through the only
    /// observable it has: `mark` returns `None`, so `add` cannot move a slot.
    ///
    /// The knob is a process-wide `OnceLock` and the test binary does not set
    /// it, so this is the state every other test in the crate runs under too.
    #[test]
    fn disabled_is_inert() {
        assert!(!enabled(), "the test binary must not set LFM_PROVE_SPLIT");
        static S: Slot = Slot::new();
        let t = mark();
        assert!(t.is_none(), "no clock is read when the knob is off");
        add(&S, t);
        assert_eq!(S.0.load(Ordering::Relaxed), 0, "and no slot moved");
        assert!(begin().is_none(), "and no prove is opened");
        assert!(report(None, 3, 1024).is_none(), "and no line is produced");
    }

    /// A slot accumulates across threads and `take` CLEARS it — the property
    /// that makes one line describe one prove rather than every prove so far.
    #[test]
    fn a_slot_sums_across_threads_and_clears_on_take() {
        static S: Slot = Slot::new();
        // Written directly rather than through `mark`, which is off in this
        // binary by design (see `disabled_is_inert`).
        std::thread::scope(|sc| {
            for _ in 0..4 {
                sc.spawn(|| S.0.fetch_add(250_000_000, Ordering::Relaxed));
            }
        });
        assert!((S.take() - 1.0).abs() < 1e-9, "four workers, 0.25s each");
        assert_eq!(S.take(), 0.0, "and the second read finds it cleared");
    }
}
