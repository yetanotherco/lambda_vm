//! At most one proof inside its device phases.
//!
//! # Why a gate above the prover's own
//!
//! There are TWO places an LFM proof reaches the card, and neither keeps an
//! account the other can see.
//!
//! ✓ `commit_group_device_or_host` calls `stark::gpu_lde::try_commit_row_major`
//! directly, inside `build_artifacts` — outside `multi_prove` and so outside
//! every `VramGate`. Its only check is `gpu_lde::admit` → `device_set::
//! admit_bytes`, a `const fn` comparing ONE dispatch's bytes against the WHOLE
//! card budget. No state, no running total.
//!
//! ✓ `multi_prove` builds a fresh `VramGate::new(vram_budget)` per call, with
//! the full budget. It keeps a running total — over the tables of its OWN prove
//! and nothing else.
//!
//! ⇒ **Two proofs in flight can ask the card for twice its budget from two
//! directions at once.** That is the two-VramGates condition that caused the
//! production base aborts, one level up; the answer there was to serialise the
//! callers, and nothing since has given either mechanism a cross-proof total.
//!
//! ⛔ **So this is MUTUAL EXCLUSION, and it cannot be a byte budget.** A
//! byte-budget permit would admit a second holder "because its bytes fit"
//! against a budget the first holder is already spending — there is no
//! cross-proof total for it to compose with.
//!
//! # Why it is acquired TWICE per proof, not held across both phases
//!
//! A node's phases run `host → build_artifacts → host → multi_prove → host`.
//! The two device phases are NOT adjacent: the LFM executor and the trace fill
//! sit between them, and on a measured arity-2 level-1 node that is 2.53 s of a
//! 7.66 s node. Held across both, the permit covers 6.33 s and a level's floor
//! is 83% of its serial wall; released between them it covers 3.80 s and the
//! floor is half. Overlapping the executor IS the lever.
//!
//! Mutual exclusion is unaffected: at every instant at most one proof is inside
//! a device phase. What changes is only that a proof stops occupying the card
//! while it walks the host.
//!
//! ✓ Releasing there is safe because the artifact commit keeps nothing:
//! `try_commit_row_major` binds `(tree, _handle, _lde)` and returns the root,
//! and `_handle` — the `GpuLdeBase` holding the device tree and buffers —
//! drops at function exit. At the release point the proof holds no device
//! allocation of its own.
//!
//! # Inert until armed
//!
//! Unarmed, and at one worker, [`hold`] takes no lock and touches no counter on
//! the contended path: it reads one relaxed `usize` plus the round-3 tree
//! probe's cached `OnceLock<bool>` and returns. Nothing that ships arms it, and
//! nothing that ships sets `LAMBDA_VM_TREE_BUSY_PROBE`.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// How many sibling proofs the driver intends to run at once. 1 = the serial
/// driver, and the permit is inert.
static WORKERS: AtomicUsize = AtomicUsize::new(1);

/// The card. One holder.
static CARD: Mutex<()> = Mutex::new(());

/// Holders right now. ★ The evidence for the falsifier, not a statistic: the
/// acquire asserts this is 1, so "two holders at once" fails loudly at the
/// moment it happens rather than being inferred afterwards from a VRAM abort.
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
/// The largest value `IN_FLIGHT` has ever taken. Printed, so a green run says
/// so rather than merely not failing.
static PEAK_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
static ACQUISITIONS: AtomicUsize = AtomicUsize::new(0);
static HELD_NANOS: AtomicU64 = AtomicU64::new(0);
/// Holds since the process started, armed or not — the number the trace line
/// carries. Separate from `ACQUISITIONS`, which a level clears.
static TRACE_SEQ: AtomicUsize = AtomicUsize::new(0);

/// `LFM_CARD_TRACE=1` prints one line per hold: the phase, what it waited, what
/// it held, and the two wall-clock stamps that bracket the window.
///
/// ★ The stamps are what make an external sampler attributable. A 10 Hz
/// `nvidia-smi` log says how busy the card was; it cannot say whose window that
/// was. Slicing the sample by these two numbers answers "how idle is the card
/// INSIDE the held windows", which is the question a mutual-exclusion gate
/// raises and no aggregate can answer.
pub fn trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("LFM_CARD_TRACE") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    })
}

thread_local! {
    /// ★ HOW LONG THIS THREAD HAS BLOCKED ON THE CARD, cumulative and never
    /// cleared.
    ///
    /// A waiting worker's time lands inside whatever phase was running when it
    /// asked — so at two siblings a node's `prove` field silently absorbs the
    /// wait for `multi_prove` and stops being comparable to the serial arm's.
    /// Measured at level 2: `prove` read 5.0 and 7.1 concurrently against 4.8
    /// and 4.1 serial, entirely from this.
    ///
    /// ⇒ MONOTONE AND READ-ONLY, not a take-and-clear. A counter that clears
    /// couples every reader to every other one: whoever samples first silently
    /// steals the wait from whoever samples next. Callers bracket the span they
    /// care about and subtract, which composes.
    static WAITED_NANOS: Cell<u64> = const { Cell::new(0) };
    /// ⛔ Re-entry detection. `Mutex` is not reentrant, so a second `hold()` on
    /// a thread that already holds one parks it forever — and a silent park is
    /// exactly how the census memo cost twelve minutes of a test binary sitting
    /// at 0% CPU with nothing printed. A panic names it instead.
    static HELD_HERE: Cell<bool> = const { Cell::new(false) };
    /// The latch this thread's `multi_prove` holds wait behind — see
    /// [`defer_multi_prove_until`].
    static DEFER_PROVE: RefCell<Option<Arc<CardLatch>>> = const { RefCell::new(None) };
    /// The proof this thread's holds belong to, for the trace line — see
    /// [`name_holder`].
    static HOLDER: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Name the proof this thread's card holds belong to until the guard drops,
/// so each `CARD HOLD` trace line ends `· who=<name>`. A level's tasks run one
/// to a thread, so the name is what tells the global child's prove from a wide
/// node's in a log where both are just holds. Trace-only: nothing else reads it.
pub fn name_holder(name: impl Into<String>) -> HolderGuard {
    let previous = HOLDER.with(|h| h.borrow_mut().replace(name.into()));
    HolderGuard {
        previous,
        _this_thread: std::marker::PhantomData,
    }
}

/// Puts the thread's previous holder name back when it drops.
pub struct HolderGuard {
    previous: Option<String>,
    _this_thread: std::marker::PhantomData<*const ()>,
}

impl Drop for HolderGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        HOLDER.with(|h| *h.borrow_mut() = previous);
    }
}

/// ` · who=<name>` for a trace line, or nothing when no name is set.
fn holder_suffix() -> String {
    HOLDER.with(|h| {
        h.borrow()
            .as_ref()
            .map(|name| format!(" · who={name}"))
            .unwrap_or_default()
    })
}

/// A one-way gate a `multi_prove` can be told to wait behind: shut until
/// [`open`](Self::open), then open for good.
///
/// ★ What it orders is level 1's card. There, the WHIR global child's prove and
/// the last wide node's artifact build ask for the card within a few hundred
/// milliseconds of each other. When the global asks first, the last node's
/// artifacts wait out its whole prove, and the last node's host prep then runs
/// while the card idles; asked the other way round, the global's prove covers
/// that prep. The driver's `LFM_TREE_GLOBAL_AFTER_LAST` puts the global behind
/// this latch and has the last node open it.
pub struct CardLatch {
    open: Mutex<bool>,
    opened: Condvar,
}

impl CardLatch {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            open: Mutex::new(false),
            opened: Condvar::new(),
        })
    }

    /// Open it, for every waiter now and later.
    pub fn open(&self) {
        *self.open.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.opened.notify_all();
    }

    pub fn is_open(&self) -> bool {
        *self.open.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Wait until it opens or `bound` passes: the time waited, and whether the
    /// bound ended the wait.
    fn wait(&self, bound: Duration) -> (Duration, bool) {
        let since = Instant::now();
        let open = self.open.lock().unwrap_or_else(|e| e.into_inner());
        let (open, _) = self
            .opened
            .wait_timeout_while(open, bound, |open| !*open)
            .unwrap_or_else(|e| e.into_inner());
        (since.elapsed(), !*open)
    }
}

/// Opens its latch when it drops: where it is dropped on purpose, and as a
/// panic unwinds, so a waiter is never left behind a task that died.
pub struct OpenOnDrop(Arc<CardLatch>);

impl OpenOnDrop {
    pub fn new(latch: Arc<CardLatch>) -> Self {
        Self(latch)
    }
}

impl Drop for OpenOnDrop {
    fn drop(&mut self) {
        self.0.open();
    }
}

/// The longest a deferred `multi_prove` waits for its latch before it queues
/// for the card anyway, and says so. The latch orders the card; it must never
/// be able to stop a proof. On the block the wait is at most about a second.
const DEFER_BOUND: Duration = Duration::from_secs(30);

/// Deferred `multi_prove` holds, and those the bound released.
static DEFERRED: AtomicUsize = AtomicUsize::new(0);
static DEFER_TIMEOUTS: AtomicUsize = AtomicUsize::new(0);

/// (deferred `multi_prove` holds, those the bound released) this process.
pub fn defer_stats() -> (usize, usize) {
    (
        DEFERRED.load(Ordering::Relaxed),
        DEFER_TIMEOUTS.load(Ordering::Relaxed),
    )
}

/// While the guard lives, this thread's `multi_prove` holds wait for `latch`
/// to open before they queue for the card — for at most [`DEFER_BOUND`]. Other
/// phases do not wait, and neither does anything while the permit is unarmed:
/// the serial driver takes no card at all, so there is nothing to order.
pub fn defer_multi_prove_until(latch: Arc<CardLatch>) -> DeferGuard {
    DEFER_PROVE.with(|d| *d.borrow_mut() = Some(latch));
    DeferGuard {
        _this_thread: std::marker::PhantomData,
    }
}

/// Clears this thread's deferral when it drops. Not `Send`: the deferral
/// belongs to the thread that installed it.
pub struct DeferGuard {
    _this_thread: std::marker::PhantomData<*const ()>,
}

impl Drop for DeferGuard {
    fn drop(&mut self) {
        DEFER_PROVE.with(|d| *d.borrow_mut() = None);
    }
}

/// Wait out this thread's deferral before a `multi_prove` queues for the card,
/// if it has one, and print what it cost.
fn wait_deferral(phase: &'static str) {
    if phase != "multi_prove" {
        return;
    }
    let Some(latch) = DEFER_PROVE.with(|d| d.borrow().clone()) else {
        return;
    };
    let t0 = stark::prove_split::epoch_secs();
    let (waited, timed_out) = latch.wait(DEFER_BOUND);
    DEFERRED.fetch_add(1, Ordering::Relaxed);
    if timed_out {
        DEFER_TIMEOUTS.fetch_add(1, Ordering::Relaxed);
    }
    println!(
        "CARD DEFER #{} {phase}: waited {:.3}s for its latch{} · t=[{t0:.3},{:.3}]{}",
        DEFERRED.load(Ordering::Relaxed),
        waited.as_secs_f64(),
        if timed_out {
            " — THE BOUND RAN OUT, queued anyway"
        } else {
            ""
        },
        stark::prove_split::epoch_secs(),
        holder_suffix(),
    );
}

/// Arm the permit for `workers` concurrent proofs. `workers <= 1` leaves it
/// inert, which is the control arm on the same binary.
pub fn arm(workers: usize) {
    WORKERS.store(workers.max(1), Ordering::Relaxed);
}

/// How many sibling proofs the driver is running at once.
pub fn workers() -> usize {
    WORKERS.load(Ordering::Relaxed)
}

/// Seconds THIS THREAD has spent blocked on the card, cumulative since the
/// process started. Bracket a span and subtract to price its wait.
pub fn waited_secs() -> f64 {
    WAITED_NANOS.with(|w| w.get()) as f64 / 1e9
}

/// What the permit did, for the line a level prints.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PermitStats {
    pub acquisitions: usize,
    pub peak_holders: usize,
    pub held_nanos: u64,
}

/// Read and CLEAR the counters, so a level reports its own.
pub fn take_stats() -> PermitStats {
    PermitStats {
        acquisitions: ACQUISITIONS.swap(0, Ordering::Relaxed),
        peak_holders: PEAK_IN_FLIGHT.swap(0, Ordering::Relaxed),
        held_nanos: HELD_NANOS.swap(0, Ordering::Relaxed),
    }
}

impl PermitStats {
    /// The line a level prints beside its wall. `held` against the level's own
    /// wall is the reading that says which resource bound the level: near 100%
    /// and the card is the wall, well under and the host is.
    pub fn describe(&self, level_wall_secs: f64) -> String {
        let held = self.held_nanos as f64 / 1e9;
        format!(
            "card permit: {} acquisitions · max holders {} · held {held:.1}s of {level_wall_secs:.1}s ({:.0}%)",
            self.acquisitions,
            self.peak_holders,
            if level_wall_secs > 0.0 {
                100.0 * held / level_wall_secs
            } else {
                0.0
            },
        )
    }
}

/// Exclusive use of the card, for as long as it is held.
pub struct CardPermit {
    guard: Option<std::sync::MutexGuard<'static, ()>>,
    since: Instant,
    /// ⛔ ROUND-3 TREE PROBE ONLY, `None` unless `LAMBDA_VM_TREE_BUSY_PROBE` is
    /// set: which device phase this hold is, and the nanoseconds its holder
    /// queued for the card.
    ///
    /// ⚠ Carried on EVERY permit the probe sees, not only the armed ones. An
    /// unarmed stage (`arm(1)` — the WHIR global child, the block-artifact
    /// root) returns before a guard exists and accumulates nothing into
    /// `HELD_NANOS`, so without this its device-phase time appears in no
    /// summary at all — only on the per-hold `LFM_CARD_TRACE` lines.
    probe: Option<(&'static str, u64)>,
    /// Trace-only, `None` unless `LFM_CARD_TRACE` is set: which device phase
    /// this hold is, the seconds it queued, its sequence number, and the epoch
    /// second it was acquired.
    trace: Option<(&'static str, f64, usize, f64)>,
    /// Trace-only: the holder's name suffix, read when the hold began.
    who: String,
}

impl Drop for CardPermit {
    fn drop(&mut self) {
        if self.guard.is_some() {
            HELD_NANOS.fetch_add(self.since.elapsed().as_nanos() as u64, Ordering::Relaxed);
            IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
            HELD_HERE.with(|h| h.set(false));
        }
        // ⛔ DELIBERATELY OUTSIDE the `guard` branch above. `HELD_NANOS` is the
        // ARMED accounting a level reports through `take_stats`; this is a
        // SECOND, separate accounting that also sees the `K = 1` stages. It
        // never touches `HELD_NANOS`, `ACQUISITIONS` or `PEAK_IN_FLIGHT`, so
        // every line the drivers already print keeps its exact meaning and the
        // probe cannot move an existing number.
        if let Some((phase, waited_nanos)) = self.probe {
            super::tree_probe::note_hold(
                phase,
                self.since.elapsed().as_nanos() as u64,
                waited_nanos,
            );
        }
        if let Some((phase, waited, seq, t0)) = self.trace {
            println!(
                "CARD HOLD #{seq} {phase}: waited {waited:.3}s · held {:.3}s · t=[{t0:.3},{:.3}]{}",
                self.since.elapsed().as_secs_f64(),
                stark::prove_split::epoch_secs(),
                self.who,
            );
        }
    }
}

/// Take the card. Blocks until the current holder releases it.
///
/// Inert — no lock, no counter, two relaxed reads (the worker count and the
/// round-3 tree probe's `OnceLock`) — while the driver is serial, so every call
/// site can take it unconditionally.
///
/// # Panics
///
/// If this thread already holds it (a self-deadlock, named rather than parked),
/// or if two holders are ever observed (the falsifier, caught at the instant it
/// happens rather than inferred later from a VRAM abort).
pub fn hold() -> CardPermit {
    hold_labeled("card")
}

/// [`hold`] with the device phase named, for the trace line. The name is the
/// only thing that tells an artifact commit's window from a `multi_prove`'s in
/// a log where both are just holds.
pub fn hold_labeled(phase: &'static str) -> CardPermit {
    let traced = trace_enabled();
    // One `OnceLock` read per hold — tens of them in a whole block run, never in
    // a loop. Off, this is the only thing the probe costs anywhere.
    let probed = super::tree_probe::enabled();
    if workers() <= 1 && !traced && !probed {
        return CardPermit {
            guard: None,
            since: Instant::now(),
            probe: None,
            trace: None,
            who: String::new(),
        };
    }
    // ★ Unarmed BUT traced or probed: there is no card to take (the serial
    // driver holds it by construction), and the window is still exactly the
    // device phase — which is the window the sampler has to be sliced by in the
    // K=1 control too, or the two arms are compared on different definitions.
    if workers() <= 1 {
        let seq = TRACE_SEQ.fetch_add(1, Ordering::Relaxed);
        return CardPermit {
            guard: None,
            since: Instant::now(),
            probe: probed.then_some((phase, 0)),
            trace: traced.then(|| (phase, 0.0, seq, stark::prove_split::epoch_secs())),
            who: if traced {
                holder_suffix()
            } else {
                String::new()
            },
        };
    }
    assert!(
        !HELD_HERE.with(|h| h.get()),
        "the card permit is not reentrant and this thread already holds it; \
         taking it twice without releasing parks the thread forever"
    );
    // ★ Before the queue, not in it: a deferred prove must not hold a place in
    // the line while it waits for its latch, or it would still go first.
    wait_deferral(phase);
    // A poisoned card is a panic already being reported: take it through the
    // poison rather than turning one failure into two.
    let blocked_from = Instant::now();
    let guard = CARD.lock().unwrap_or_else(|e| e.into_inner());
    // Read ONCE. The trace line and `WAITED_NANOS` must carry the same wait, or
    // a level's accounting and its per-hold log disagree by the bookkeeping
    // below them.
    let waited = blocked_from.elapsed();
    WAITED_NANOS.with(|w| w.set(w.get() + waited.as_nanos() as u64));
    HELD_HERE.with(|h| h.set(true));
    let now = IN_FLIGHT.fetch_add(1, Ordering::SeqCst) + 1;
    PEAK_IN_FLIGHT.fetch_max(now, Ordering::Relaxed);
    ACQUISITIONS.fetch_add(1, Ordering::Relaxed);
    assert_eq!(
        now, 1,
        "★ TWO HOLDERS ON THE CARD. The permit is not a single gate, and two \
         proofs in their device phases can ask for twice the budget: the \
         artifact commit's admission is a per-dispatch bound with no running \
         total, and each `multi_prove` builds its own full-budget VramGate"
    );
    CardPermit {
        guard: Some(guard),
        since: Instant::now(),
        // ⚠ The SAME `waited` the trace line and `WAITED_NANOS` carry, not a
        // second reading of the clock — three accountings of one wait that
        // disagreed would be worse than two that do not exist.
        probe: probed.then_some((phase, waited.as_nanos() as u64)),
        trace: traced.then(|| {
            (
                phase,
                waited.as_secs_f64(),
                TRACE_SEQ.fetch_add(1, Ordering::Relaxed),
                stark::prove_split::epoch_secs(),
            )
        }),
        who: if traced {
            holder_suffix()
        } else {
            String::new()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `arm` is process-global, so the tests that change it run one at a time.
    static ARM: Mutex<()> = Mutex::new(());

    fn disarm(_g: &std::sync::MutexGuard<'_, ()>) {
        arm(1);
        let _ = take_stats();
    }

    /// ★ THE CONTROL ARM IS FREE. Unarmed, `hold` takes no lock — so the K=1
    /// control on the same binary is the serial driver, not the serial driver
    /// plus a mutex per phase.
    #[test]
    fn unarmed_the_permit_takes_no_lock_and_counts_nothing() {
        let g = ARM.lock().expect("the arm guard is never poisoned");
        disarm(&g);
        {
            let _a = hold();
            // ⛔ The check that makes this a test rather than a smoke run: a
            // second hold on this thread would DEADLOCK if the permit were
            // live, so reaching the line after it proves it is not.
            let _b = hold();
        }
        // ⓘ Read while nothing else is armed: `disarm` ran above and the `ARM`
        // guard is held, so `workers()` is 1 for every thread and no build
        // anywhere in the process can reach a counter.
        let stats = take_stats();
        assert_eq!(stats.acquisitions, 0, "an inert permit counts nothing");
        assert_eq!(stats.peak_holders, 0);
        assert_eq!(stats.held_nanos, 0);
        disarm(&g);
    }

    /// ★★★ THE FALSIFIER, ASSERTED — with the overlap FORCED.
    ///
    /// ⛔ An earlier version of this test spun two workers through twenty fast
    /// acquisitions each and asserted `peak_holders == 1`. It passed with the
    /// exclusion REMOVED: the critical section was nanoseconds, so two threads
    /// that were free to overlap simply never did. A check that cannot fail is
    /// worse than no check, because it reads as evidence.
    ///
    /// ⇒ The overlap is now forced rather than hoped for. The first worker
    /// takes the card and KEEPS it; the second is released only once the first
    /// is inside, and then times its own acquire. With exclusion the second
    /// waits out the hold; without it, it walks straight in — and both the
    /// wait and `peak_holders` say so.
    #[test]
    fn armed_a_second_worker_waits_for_the_first() {
        const HOLD: std::time::Duration = std::time::Duration::from_millis(200);
        let g = ARM.lock().expect("the arm guard is never poisoned");
        disarm(&g);
        arm(2);

        let first_is_in = std::sync::Barrier::new(2);
        let waited = Mutex::new(std::time::Duration::ZERO);
        // ★ The counter the TIMING lines subtract with, read on the waiting
        // thread itself. Without it a queued worker's wait stays inside
        // whatever phase was running and the per-node numbers stop being
        // comparable between arms.
        let self_reported = Mutex::new(0.0f64);
        std::thread::scope(|s| {
            s.spawn(|| {
                let _card = hold();
                first_is_in.wait();
                std::thread::sleep(HOLD);
            });
            s.spawn(|| {
                first_is_in.wait();
                let before = waited_secs();
                let t = Instant::now();
                let _card = hold();
                *waited.lock().expect("the timing lock") = t.elapsed();
                *self_reported.lock().expect("the timing lock") = waited_secs() - before;
            });
        });

        let self_reported = *self_reported.lock().expect("the timing lock");
        let waited = *waited.lock().expect("the timing lock");
        assert!(
            self_reported >= HOLD.as_secs_f64() / 2.0,
            "the thread must report its OWN wait, got {self_reported}s"
        );
        assert!(
            (self_reported - waited.as_secs_f64()).abs() < 0.05,
            "the reported wait must be the observed one: {self_reported} vs {waited:?}"
        );
        assert!(
            waited >= HOLD / 2,
            "the second worker took the card after {waited:?}, so it did not \
             wait for the first: the permit is not excluding"
        );
        let stats = take_stats();
        assert_eq!(
            stats.peak_holders, 1,
            "★ THE FALSIFIER: two holders on the card at once"
        );
        // ⚠ NOT an exact count, and the reason is worth keeping. `arm` is
        // process-global by design — a driver arms it for a level, not for a
        // call — so while this test holds `ARM`, any OTHER test thread building
        // artifacts takes the card too and lands in these counters. An exact 2
        // failed for exactly that reason. The three assertions that remain are
        // each true whatever else the process is doing, which is what makes
        // them the right ones: exclusion bounds the peak at 1 no matter how
        // many threads want the card.
        assert!(
            stats.acquisitions >= 2,
            "both workers must have taken it: {stats:?}"
        );
        assert!(stats.held_nanos > 0, "and the held time was accumulated");
        disarm(&g);
    }

    /// ⛔ A self-deadlock is NAMED, not parked. Without the re-entry check this
    /// test would hang rather than fail, which is the whole point of it.
    #[test]
    fn armed_a_second_hold_on_one_thread_panics_rather_than_parking() {
        let g = ARM.lock().expect("the arm guard is never poisoned");
        disarm(&g);
        arm(2);
        let panicked = std::thread::spawn(|| {
            let _a = hold();
            let _b = hold();
        })
        .join();
        assert!(panicked.is_err(), "a reentrant hold must panic");
        disarm(&g);
    }

    /// ★ A DEFERRED `multi_prove` WAITS FOR ITS LATCH, AND NOTHING ELSE DOES.
    ///
    /// The worker's artifact hold goes straight to the card; its prove hold
    /// takes the card only after the latch opens, which the main thread does
    /// after a pause the prove cannot have waited out by accident. A permit
    /// that ignored the deferral takes the prove's card within milliseconds,
    /// before the latch opens, and fails the order assertion.
    #[test]
    fn armed_a_deferred_prove_waits_for_its_latch_and_an_artifact_build_does_not() {
        const PAUSE: Duration = Duration::from_millis(200);
        let g = ARM.lock().expect("the arm guard is never poisoned");
        disarm(&g);
        arm(2);
        let latch = CardLatch::new();
        let (deferred_before, _) = defer_stats();
        let artifacts_at = Mutex::new(None);
        let prove_at = Mutex::new(None);
        let opened_at = Mutex::new(None);
        let t0 = Instant::now();
        std::thread::scope(|s| {
            s.spawn(|| {
                let _deferred = defer_multi_prove_until(latch.clone());
                {
                    let _card = hold_labeled("build_artifacts");
                }
                *artifacts_at.lock().expect("the timing lock") = Some(t0.elapsed());
                let _card = hold_labeled("multi_prove");
                *prove_at.lock().expect("the timing lock") = Some(t0.elapsed());
            });
            std::thread::sleep(PAUSE);
            *opened_at.lock().expect("the timing lock") = Some(t0.elapsed());
            latch.open();
        });
        let artifacts = artifacts_at
            .lock()
            .expect("the timing lock")
            .expect("built");
        let prove = prove_at.lock().expect("the timing lock").expect("proved");
        let opened = opened_at.lock().expect("the timing lock").expect("opened");
        assert!(
            artifacts < PAUSE / 2,
            "an artifact build must not wait for the latch: it took the card at {artifacts:?}"
        );
        assert!(
            prove >= opened,
            "the deferred prove took the card at {prove:?}, before its latch opened at {opened:?}"
        );
        assert!(
            defer_stats().0 > deferred_before,
            "the deferral must be counted"
        );
        disarm(&g);
    }

    /// An open latch costs nothing, and a dropped guard leaves no deferral
    /// behind.
    #[test]
    fn armed_an_open_latch_or_a_dropped_guard_does_not_delay_a_prove() {
        const QUICK: Duration = Duration::from_millis(100);
        let g = ARM.lock().expect("the arm guard is never poisoned");
        disarm(&g);
        arm(2);
        let open = CardLatch::new();
        open.open();
        {
            let _deferred = defer_multi_prove_until(open);
            let t = Instant::now();
            let _card = hold_labeled("multi_prove");
            assert!(
                t.elapsed() < QUICK,
                "an open latch must not delay the prove"
            );
        }
        let shut = CardLatch::new();
        drop(defer_multi_prove_until(shut));
        let t = Instant::now();
        {
            let _card = hold_labeled("multi_prove");
        }
        assert!(
            t.elapsed() < QUICK,
            "a dropped guard must clear the deferral, or the prove waits for a shut latch"
        );
        disarm(&g);
    }

    /// Unarmed, the permit takes no card, so there is nothing to order: a
    /// deferral is ignored rather than parking the serial driver behind a
    /// latch only a later task would open.
    #[test]
    fn unarmed_a_deferral_is_ignored() {
        let g = ARM.lock().expect("the arm guard is never poisoned");
        disarm(&g);
        let shut = CardLatch::new();
        let _deferred = defer_multi_prove_until(shut);
        let t = Instant::now();
        {
            let _card = hold_labeled("multi_prove");
        }
        assert!(
            t.elapsed() < Duration::from_millis(100),
            "the serial driver must never wait for a latch"
        );
        disarm(&g);
    }

    /// The opener opens as a panic unwinds, so a waiter never outlives the
    /// task that was to open its latch; and the wait itself is bounded.
    #[test]
    fn an_opener_opens_as_its_task_unwinds_and_the_wait_is_bounded() {
        let latch = CardLatch::new();
        let (waited, bound_ran_out) = latch.wait(Duration::from_millis(50));
        assert!(
            bound_ran_out && waited >= Duration::from_millis(40),
            "a shut latch must hold a waiter until the bound: {waited:?}"
        );
        let opener = OpenOnDrop::new(latch.clone());
        let died = std::thread::spawn(move || {
            let _opener = opener;
            panic!("the task that was to open the latch died");
        })
        .join();
        assert!(died.is_err(), "the task must have panicked");
        assert!(latch.is_open(), "its unwind must have opened the latch");
        let (waited, bound_ran_out) = latch.wait(Duration::from_secs(5));
        assert!(
            !bound_ran_out && waited < Duration::from_millis(100),
            "an open latch must not hold a waiter: {waited:?}"
        );
    }

    /// With no global child in the level's pool the driver makes no latch, so
    /// nothing defers: a node's opener would open a latch nobody waits on, and
    /// a prove on a thread with no deferral takes the card at once. Both are
    /// asserted, so the no-global tree cannot depend on a latch being made.
    #[test]
    fn armed_with_no_deferral_nothing_waits_and_an_unwaited_opener_is_harmless() {
        let g = ARM.lock().expect("the arm guard is never poisoned");
        disarm(&g);
        arm(2);
        let unwaited = CardLatch::new();
        drop(OpenOnDrop::new(unwaited.clone()));
        assert!(unwaited.is_open(), "the opener opened its latch");
        let (deferred_before, _) = defer_stats();
        let t = Instant::now();
        {
            let _card = hold_labeled("multi_prove");
        }
        assert!(
            t.elapsed() < Duration::from_millis(100),
            "a prove with no deferral must take the card at once"
        );
        assert_eq!(
            defer_stats().0,
            deferred_before,
            "no deferral may be counted on a thread that installed none"
        );
        disarm(&g);
    }

    /// The holder name is the thread's, restored when its guard drops.
    #[test]
    fn a_holder_name_tags_this_threads_holds_and_is_restored() {
        assert_eq!(holder_suffix(), "");
        {
            let _outer = name_holder("global");
            assert_eq!(holder_suffix(), " · who=global");
            {
                let _inner = name_holder("L1N2");
                assert_eq!(holder_suffix(), " · who=L1N2");
            }
            assert_eq!(
                holder_suffix(),
                " · who=global",
                "the inner name must be undone"
            );
            let elsewhere = std::thread::spawn(holder_suffix)
                .join()
                .expect("the other thread");
            assert_eq!(elsewhere, "", "the name is this thread's alone");
        }
        assert_eq!(holder_suffix(), "");
    }

    /// The level line says which resource bound the level, so a green run
    /// reports the permit's occupancy rather than only its safety.
    #[test]
    fn the_permit_line_reports_occupancy_against_the_level_wall() {
        let stats = PermitStats {
            acquisitions: 20,
            peak_holders: 1,
            held_nanos: 38_000_000_000,
        };
        let line = stats.describe(43.2);
        assert!(line.contains("max holders 1"), "{line}");
        assert!(line.contains("held 38.0s of 43.2s (88%)"), "{line}");
    }
}
