//! ⛔ ROUND-3 ARGUE DISCRIMINATOR (diagnostic; on NO production decision path).
//!
//! The occupancy sweep put the RPX permutation at its arithmetic floor, so round
//! 3's only remaining stage is `argue` (the WHIR field-arithmetic argument:
//! sumcheck + gkr + columns, ~16.6s, per-table dominated by KECCAK_RND). STEP 1
//! (the O5 device-fallback counter, read from the round-2 logs) established argue
//! does NOT host-fall-back at the tip — it is device-side and stable. This
//! instrument answers STEP 2: is argue's device work MEMORY-bound (a layout/reuse
//! lever exists) or COMPUTE-bound (near a floor, like the permutation)?
//!
//! ncu is unavailable on the box (ERR_NVGPUCTRPERM), so the memory roofline is
//! read by ARITHMETIC instead of a hardware counter: this records, per argue
//! surface, the DEVICE-path bytes RESERVED and the number of device operations,
//! at the same five `reserve` sites the fallback counter already guards
//! (sumcheck.rs, gkr.rs, columns.rs). The harness prints the totals; divided by
//! the known argue wall time they give an achieved HBM bandwidth:
//!   near the card roofline (~1.7 TB/s on the 5090) ⇒ MEMORY-bound ⇒ the lever is
//!     MLE layout/reuse (keep folds resident, fuse fold+eval, stop
//!     re-materialising factors);
//!   far below it ⇒ the resident data is reused from cache, argue is
//!     COMPUTE-bound on the extension-field arithmetic ⇒ near a floor.
//!
//! Reserved bytes is the DEVICE working set per operation, so summed over the run
//! it is the HBM traffic to first order (a resident buffer re-read within an op
//! is L2/L1, not HBM). It is a robust, additive quantity — unlike a wall-timer at
//! these sites, which would double-count when one surface's reserved scope nests
//! another's (gkr's layer sumcheck inside a reserve). A finer per-surface TIME /
//! device-vs-host split is STEP 2b, added only if the byte roofline is borderline.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// The three argue device surfaces, matching the crates the reserve sites live
/// in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    Sumcheck,
    Gkr,
    Columns,
}

struct Counters {
    /// Device-path bytes reserved (the working set promised to the card).
    bytes: AtomicU64,
    /// Device operations that got their reservation (i.e. ran on the device).
    calls: AtomicU64,
}

impl Counters {
    const fn zero() -> Self {
        Self {
            bytes: AtomicU64::new(0),
            calls: AtomicU64::new(0),
        }
    }
}

static SUMCHECK: Counters = Counters::zero();
static GKR: Counters = Counters::zero();
static COLUMNS: Counters = Counters::zero();

fn counters(surface: Surface) -> &'static Counters {
    match surface {
        Surface::Sumcheck => &SUMCHECK,
        Surface::Gkr => &GKR,
        Surface::Columns => &COLUMNS,
    }
}

/// Record one device-path reservation on `surface`: the bytes it promised and
/// one operation. Called at each `reserve`-SUCCESS site, beside the
/// `note_device_fallback()` the else-branch already calls.
pub fn note_device(surface: Surface, bytes: u64) {
    let c = counters(surface);
    c.bytes.fetch_add(bytes, Ordering::Relaxed);
    c.calls.fetch_add(1, Ordering::Relaxed);
}

/// `(bytes, calls)` for one surface, whole-run scope.
pub fn surface_totals(surface: Surface) -> (u64, u64) {
    let c = counters(surface);
    (
        c.bytes.load(Ordering::Relaxed),
        c.calls.load(Ordering::Relaxed),
    )
}

// ── device-busy sizing (round-3 argue idle-fraction discriminator) ──────────
//
// The concurrency lever is soundness-dead (the whole argue is one sequential
// Fiat-Shamir chain — the transcript order IS the proof), so argue's ~2% util is
// mostly INHERENT per-round host-sync latency. This measures how much of the
// argue wall the round kernels are actually BUSY: the sizing timer records CUDA
// events around the sumcheck round kernels and reads the elapsed after the
// EXISTING per-round synchronize(), accumulating device-busy nanoseconds. idle =
// argue_wall − device_busy sizes the total gap; the recoverable-WITHOUT-a-rewrite
// fraction is only what transcript-independent prefetch can fill (the
// orchestration read sizes that), so a large idle here is NOT a large recoverable
// win by itself.

static DEVICE_BUSY_NS: AtomicU64 = AtomicU64::new(0);

/// Enabled only when `LAMBDA_VM_ARGUE_BUSY_PROBE` is set — read ONCE and cached,
/// so a production round() pays a single relaxed bool load and nothing else. The
/// event recording (and its one-time timing-event creation) happens only when on.
pub fn busy_probe_enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var_os("LAMBDA_VM_ARGUE_BUSY_PROBE").is_some())
}

/// Add device-busy nanoseconds measured across one round's kernels.
pub fn add_device_busy_ns(ns: u64) {
    DEVICE_BUSY_NS.fetch_add(ns, Ordering::Relaxed);
}

/// Device-busy nanoseconds accumulated over the run (round kernels only; fold and
/// setup are excluded, so idle = wall − this is a slight OVER-estimate).
pub fn device_busy_ns() -> u64 {
    DEVICE_BUSY_NS.load(Ordering::Relaxed)
}

/// Zero every surface — call before a run whose totals are to be read, exactly
/// as the fallback counter is reset.
pub fn reset() {
    for c in [&SUMCHECK, &GKR, &COLUMNS] {
        c.bytes.store(0, Ordering::Relaxed);
        c.calls.store(0, Ordering::Relaxed);
    }
    DEVICE_BUSY_NS.store(0, Ordering::Relaxed);
}
