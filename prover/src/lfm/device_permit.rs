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
//!
//! # The order it hands the card on (`LFM_CARD_PERMIT_ORDER`)
//!
//! A proof asks for the card twice: a SHORT commit (`build_artifacts`,
//! 0.1–1.1 s) and, after its host execute and fill, a LONG prove
//! (`multi_prove`, 0.6–1.7 s). Unset or `fifo`, the permit is one `Mutex` and
//! its waiters are served in the order they asked (measured: every hold of job
//! 222's level 1). A late node's commit then waits out every prove queued
//! before it, and its own host execute starts only after that — with nothing
//! left on the card. Level 1 of job 222 idled 0.26 s that way.
//!
//! `commits-first` serves a waiting commit before a waiting prove, each class
//! in the order it asked, so a commit's host work overlaps the proves instead
//! of following them. No prove is overtaken more than [`MAX_OVERTAKES`] times:
//! past that the gate serves strictly in order until it has been served.
//!
//! Only the ORDER changes. Mutual exclusion is the same assertion on both
//! paths, and a proof's bytes are a function of its program and arenas alone.
//! The order is latched by [`arm`], so it cannot change under a waiting holder.

use std::cell::Cell;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

/// How many sibling proofs the driver intends to run at once. 1 = the serial
/// driver, and the permit is inert.
static WORKERS: AtomicUsize = AtomicUsize::new(1);

/// The card. One holder.
static CARD: Mutex<()> = Mutex::new(());

/// The order the ARMED permit serves in, latched by [`arm`]: [`Order::Fifo`]
/// takes `CARD`, [`Order::CommitsFirst`] takes `GATE`.
static MODE: AtomicU8 = AtomicU8::new(Order::Fifo as u8);

/// The card under `commits-first`. One holder, like `CARD`.
static GATE: Gate = Gate::new();

/// The most commits that may be served ahead of one waiting prove. A level
/// has one commit per task, so at six siblings a prove meets at most five: the
/// bound never binds on today's trees and guarantees progress on any other.
pub const MAX_OVERTAKES: u32 = 8;

/// Commits served ahead of an older waiting prove, and the most times one prove
/// was overtaken, since the last [`take_stats`] — the level line's evidence
/// that the order moved.
static REORDERS: AtomicUsize = AtomicUsize::new(0);
static MAX_OVERTAKEN: AtomicUsize = AtomicUsize::new(0);

/// The environment variable that picks the order.
pub const ORDER_ENV: &str = "LFM_CARD_PERMIT_ORDER";

/// How the armed permit hands the card on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Order {
    /// One `Mutex`: waiters in the order they asked. The default.
    Fifo = 0,
    /// A waiting commit before a waiting prove, each class in order, no prove
    /// overtaken more than [`MAX_OVERTAKES`] times.
    CommitsFirst = 1,
}

impl Order {
    /// Unset, empty or `fifo` is [`Order::Fifo`]; `commits-first` is
    /// [`Order::CommitsFirst`].
    ///
    /// # Panics
    ///
    /// On any other value: a misspelt arm must not run as the control.
    pub fn parse(value: Option<&str>) -> Order {
        match value {
            None | Some("") | Some("fifo") => Order::Fifo,
            Some("commits-first") => Order::CommitsFirst,
            Some(other) => {
                panic!("{ORDER_ENV} must be `fifo` or `commits-first`, got `{other}`")
            }
        }
    }

    /// The name the hold lines and the banner print.
    pub fn name(self) -> &'static str {
        match self {
            Order::Fifo => "fifo",
            Order::CommitsFirst => "commits-first",
        }
    }

    fn from_u8(v: u8) -> Order {
        if v == Order::CommitsFirst as u8 {
            Order::CommitsFirst
        } else {
            Order::Fifo
        }
    }
}

/// The order the process asks for: the environment, read once — or, in a test,
/// the order it set with [`set_order_for_tests`].
fn configured_order() -> Order {
    #[cfg(test)]
    {
        let v = ORDER_FOR_TESTS.load(Ordering::Relaxed);
        if v != NO_TEST_ORDER {
            return Order::from_u8(v);
        }
    }
    static ORDER: std::sync::OnceLock<Order> = std::sync::OnceLock::new();
    *ORDER.get_or_init(|| Order::parse(std::env::var(ORDER_ENV).ok().as_deref()))
}

/// The order the permit is armed with now.
pub fn order() -> Order {
    Order::from_u8(MODE.load(Ordering::Relaxed))
}

#[cfg(test)]
const NO_TEST_ORDER: u8 = u8::MAX;
#[cfg(test)]
static ORDER_FOR_TESTS: AtomicU8 = AtomicU8::new(NO_TEST_ORDER);

/// Tests only: the order the next [`arm`] latches, in place of the
/// environment's; `None` gives the environment back. Hold [`TEST_ARM`] around
/// it, as around any `arm`.
#[cfg(test)]
pub(crate) fn set_order_for_tests(order: Option<Order>) {
    ORDER_FOR_TESTS.store(order.map_or(NO_TEST_ORDER, |o| o as u8), Ordering::Relaxed);
}

/// `arm` and the order are process-global, so every test that arms the permit,
/// in any module, holds this.
#[cfg(test)]
pub(crate) static TEST_ARM: Mutex<()> = Mutex::new(());

/// Tests only: the commits and proves waiting at the process's gate, so a test
/// can force a queue before it lets the card go.
#[cfg(test)]
pub(crate) fn gate_queue_for_tests() -> (usize, usize) {
    let (commits, proves, _) = GATE.snapshot();
    (commits, proves)
}

/// The commits-first card: who holds it, and who waits in which class.
struct Gate {
    state: Mutex<GateState>,
    freed: Condvar,
}

impl Gate {
    const fn new() -> Self {
        Gate {
            state: Mutex::new(GateState::new()),
            freed: Condvar::new(),
        }
    }

    /// Wait for the card as a commit or as a prove. Returns how many commits
    /// were served ahead of this holder while it waited (a prove's) and how
    /// many older proves it was served ahead of (a commit's).
    fn acquire(&self, commit: bool) -> (u32, u32) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let ticket = st.enqueue(commit);
        loop {
            if !st.held && st.next(MAX_OVERTAKES) == Some(ticket) {
                let (served, passed) = st.grant(ticket);
                return (served.overtaken, passed);
            }
            st = self.freed.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn release(&self) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.held = false;
        drop(st);
        // Every waiter re-reads `next`; the one it names takes the card.
        self.freed.notify_all();
    }

    /// Tests only: waiting commits and proves, and whether the card is held.
    #[cfg(test)]
    fn snapshot(&self) -> (usize, usize, bool) {
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let commits = st.waiting.iter().filter(|w| w.commit).count();
        (commits, st.waiting.len() - commits, st.held)
    }
}

/// The gate's queue. Pure: [`GateState::next`] and [`GateState::grant`] are the
/// whole policy, and the tests drive them directly.
#[derive(Debug)]
struct GateState {
    held: bool,
    next_ticket: u64,
    waiting: Vec<Waiter>,
}

#[derive(Clone, Copy, Debug)]
struct Waiter {
    ticket: u64,
    commit: bool,
    /// Commits served ahead of this prove while it waited.
    overtaken: u32,
}

impl GateState {
    const fn new() -> Self {
        GateState {
            held: false,
            next_ticket: 0,
            waiting: Vec::new(),
        }
    }

    /// Join the queue; the ticket is the arrival order.
    fn enqueue(&mut self, commit: bool) -> u64 {
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        self.waiting.push(Waiter {
            ticket,
            commit,
            overtaken: 0,
        });
        ticket
    }

    /// Whom the card goes to next: the oldest waiting commit — unless a waiting
    /// prove has been overtaken `max_overtakes` times, and then the oldest
    /// waiter of either class, so that prove is served before any commit that
    /// came after it.
    fn next(&self, max_overtakes: u32) -> Option<u64> {
        let starved = self
            .waiting
            .iter()
            .any(|w| !w.commit && w.overtaken >= max_overtakes);
        let commit = self
            .waiting
            .iter()
            .filter(|w| w.commit)
            .map(|w| w.ticket)
            .min();
        match commit {
            Some(t) if !starved => Some(t),
            _ => self.waiting.iter().map(|w| w.ticket).min(),
        }
    }

    /// Hand the card to `ticket`. A commit counts one overtake on every older
    /// waiting prove; returns the served waiter and how many proves it passed.
    fn grant(&mut self, ticket: u64) -> (Waiter, u32) {
        let at = self
            .waiting
            .iter()
            .position(|w| w.ticket == ticket)
            .expect("only a waiting ticket is granted");
        let served = self.waiting.remove(at);
        let mut passed = 0;
        if served.commit {
            for w in self
                .waiting
                .iter_mut()
                .filter(|w| !w.commit && w.ticket < served.ticket)
            {
                w.overtaken += 1;
                passed += 1;
            }
        }
        self.held = true;
        (served, passed)
    }
}

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
}

/// Arm the permit for `workers` concurrent proofs. `workers <= 1` leaves it
/// inert, which is the control arm on the same binary.
///
/// ⛔ It also latches the ORDER ([`ORDER_ENV`]), and the drivers call it only
/// between levels, with no holder: a holder of one order and a waiter of the
/// other would take different locks. The falsifier below would catch it.
pub fn arm(workers: usize) {
    let order = configured_order();
    MODE.store(order as u8, Ordering::Relaxed);
    WORKERS.store(workers.max(1), Ordering::Relaxed);
    if workers > 1 {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            println!(
                "   ★ CARD PERMIT ORDER: {} ({ORDER_ENV}; unset = fifo, the default; \
                 commits-first serves a waiting build_artifacts before a waiting \
                 multi_prove, no prove overtaken more than {MAX_OVERTAKES} times)",
                order.name()
            );
        });
    }
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
    /// Under `commits-first`: commits served ahead of an older waiting prove,
    /// and the most times one prove was overtaken. Zero under `fifo`.
    pub reorders: usize,
    pub max_overtaken: usize,
    pub commits_first: bool,
}

/// Read and CLEAR the counters, so a level reports its own.
pub fn take_stats() -> PermitStats {
    PermitStats {
        acquisitions: ACQUISITIONS.swap(0, Ordering::Relaxed),
        peak_holders: PEAK_IN_FLIGHT.swap(0, Ordering::Relaxed),
        held_nanos: HELD_NANOS.swap(0, Ordering::Relaxed),
        reorders: REORDERS.swap(0, Ordering::Relaxed),
        max_overtaken: MAX_OVERTAKEN.swap(0, Ordering::Relaxed),
        commits_first: order() == Order::CommitsFirst,
    }
}

impl PermitStats {
    /// The line a level prints beside its wall. `held` against the level's own
    /// wall is the reading that says which resource bound the level: near 100%
    /// and the card is the wall, well under and the host is.
    pub fn describe(&self, level_wall_secs: f64) -> String {
        let held = self.held_nanos as f64 / 1e9;
        let line = format!(
            "card permit: {} acquisitions · max holders {} · held {held:.1}s of {level_wall_secs:.1}s ({:.0}%)",
            self.acquisitions,
            self.peak_holders,
            if level_wall_secs > 0.0 {
                100.0 * held / level_wall_secs
            } else {
                0.0
            },
        );
        // ⓘ The fifo line is the line every earlier log carries, unchanged.
        if self.commits_first {
            format!(
                "{line} · commits-first: {} commit(s) served ahead of an older prove, a prove \
                 overtaken at most {} time(s)",
                self.reorders, self.max_overtaken
            )
        } else {
            line
        }
    }
}

/// Exclusive use of the card, for as long as it is held.
pub struct CardPermit {
    guard: Option<Held>,
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
    /// this hold is, the seconds it queued, its sequence number, the epoch
    /// second it was acquired, and — armed — the order that served it.
    trace: Option<(&'static str, f64, usize, f64, String)>,
}

/// What an armed permit holds: the fifo `Mutex`'s guard, or the gate's card.
enum Held {
    Fifo(std::sync::MutexGuard<'static, ()>),
    Gate(GateHold),
}

/// The gate's card while it lives. Dropping it hands the card on — on an
/// unwind too, as a `MutexGuard` does, so a panic between the acquire and the
/// permit (the falsifier's own assert) cannot park every other waiter.
struct GateHold;

impl Drop for GateHold {
    fn drop(&mut self) {
        GATE.release();
    }
}

/// Whether a hold of `phase` waits as a commit (the short device phase).
fn is_commit(phase: &str) -> bool {
    phase == "build_artifacts"
}

/// The armed hold line's suffix: the order, and under `commits-first` how
/// many commits overtook this prove, or how many older proves this commit
/// passed.
fn order_note(order: Order, commit: bool, overtaken: u32, passed: u32) -> String {
    match (order, commit) {
        (Order::Fifo, _) => " · order fifo".to_string(),
        (Order::CommitsFirst, true) => format!(" · order commits-first · passed {passed}"),
        (Order::CommitsFirst, false) => {
            format!(" · order commits-first · overtaken {overtaken}")
        }
    }
}

impl Drop for CardPermit {
    fn drop(&mut self) {
        let held = self.guard.take();
        if held.is_some() {
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
        if let Some((phase, waited, seq, t0, note)) = &self.trace {
            println!(
                "CARD HOLD #{seq} {phase}: waited {waited:.3}s · held {:.3}s · t=[{t0:.3},{:.3}]{note}",
                self.since.elapsed().as_secs_f64(),
                stark::prove_split::epoch_secs(),
            );
        }
        // ⓘ The card is handed on LAST, after this hold's own lines, as the
        // `Mutex` guard always was when it dropped with the permit's fields.
        match held {
            Some(Held::Fifo(guard)) => drop(guard),
            Some(Held::Gate(card)) => drop(card),
            None => {}
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
            trace: traced.then(|| {
                (
                    phase,
                    0.0,
                    seq,
                    stark::prove_split::epoch_secs(),
                    String::new(),
                )
            }),
        };
    }
    assert!(
        !HELD_HERE.with(|h| h.get()),
        "the card permit is not reentrant and this thread already holds it; \
         taking it twice without releasing parks the thread forever"
    );
    // A commit is the short device phase; anything else (`multi_prove`, a bare
    // `hold`) waits as a prove.
    let commit = is_commit(phase);
    let order = order();
    let blocked_from = Instant::now();
    // A poisoned card is a panic already being reported: take it through the
    // poison rather than turning one failure into two.
    let (guard, overtaken, passed) = match order {
        Order::Fifo => (
            Held::Fifo(CARD.lock().unwrap_or_else(|e| e.into_inner())),
            0,
            0,
        ),
        Order::CommitsFirst => {
            let (overtaken, passed) = GATE.acquire(commit);
            if passed > 0 {
                REORDERS.fetch_add(1, Ordering::Relaxed);
            }
            MAX_OVERTAKEN.fetch_max(overtaken as usize, Ordering::Relaxed);
            (Held::Gate(GateHold), overtaken, passed)
        }
    };
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
                order_note(order, commit, overtaken, passed),
            )
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `arm` is process-global, so the tests that change it — in this module or
    /// any other — run one at a time. A failed test must not fail the next one
    /// too, so the lock is taken through its poison.
    fn lock_arm() -> std::sync::MutexGuard<'static, ()> {
        TEST_ARM.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn disarm(_g: &std::sync::MutexGuard<'_, ()>) {
        arm(1);
        let _ = take_stats();
    }

    /// Arms `workers` under `order` for as long as it lives; dropping it gives the
    /// environment's order back and disarms, on a failed assert too.
    struct Armed;

    impl Armed {
        fn new(order: Order, workers: usize) -> Self {
            set_order_for_tests(Some(order));
            arm(workers);
            assert_eq!(super::order(), order, "arm latches the order the test set");
            Armed
        }
    }

    impl Drop for Armed {
        fn drop(&mut self) {
            set_order_for_tests(None);
            arm(1);
            let _ = take_stats();
        }
    }

    /// ★ THE CONTROL ARM IS FREE. Unarmed, `hold` takes no lock — so the K=1
    /// control on the same binary is the serial driver, not the serial driver
    /// plus a mutex per phase.
    #[test]
    fn unarmed_the_permit_takes_no_lock_and_counts_nothing() {
        let g = lock_arm();
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
    /// wait and `peak_holders` say so. Asserted under BOTH orders: the gate is a
    /// second lock, and it must exclude as the `Mutex` does.
    #[test]
    fn armed_a_second_worker_waits_for_the_first() {
        let _g = lock_arm();
        let _armed = Armed::new(Order::Fifo, 2);
        a_second_worker_waits_for_the_first();
    }

    #[test]
    fn under_commits_first_a_second_worker_waits_for_the_first() {
        let _g = lock_arm();
        let _armed = Armed::new(Order::CommitsFirst, 2);
        a_second_worker_waits_for_the_first();
    }

    fn a_second_worker_waits_for_the_first() {
        const HOLD: std::time::Duration = std::time::Duration::from_millis(200);
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
    }

    /// ⛔ A self-deadlock is NAMED, not parked. Without the re-entry check this
    /// test would hang rather than fail, which is the whole point of it. Under
    /// commits-first the unwind must also hand the gate's card on: the second
    /// half takes it again on another thread, which parks forever if it did not.
    #[test]
    fn armed_a_second_hold_on_one_thread_panics_rather_than_parking() {
        let _g = lock_arm();
        for order in [Order::Fifo, Order::CommitsFirst] {
            let _armed = Armed::new(order, 2);
            let panicked = std::thread::spawn(|| {
                let _a = hold();
                let _b = hold();
            })
            .join();
            assert!(panicked.is_err(), "a reentrant hold must panic ({order:?})");
            let retaken = std::thread::spawn(|| drop(hold_labeled("multi_prove"))).join();
            assert!(
                retaken.is_ok(),
                "the card is free again after the panic ({order:?})"
            );
        }
    }

    /// The level line says which resource bound the level, so a green run
    /// reports the permit's occupancy rather than only its safety.
    #[test]
    fn the_permit_line_reports_occupancy_against_the_level_wall() {
        let stats = PermitStats {
            acquisitions: 20,
            peak_holders: 1,
            held_nanos: 38_000_000_000,
            ..PermitStats::default()
        };
        let line = stats.describe(43.2);
        assert!(line.contains("max holders 1"), "{line}");
        assert!(line.contains("held 38.0s of 43.2s (88%)"), "{line}");
        // ⓘ The fifo line is every earlier log's line, byte for byte.
        assert_eq!(
            line,
            "card permit: 20 acquisitions · max holders 1 · held 38.0s of 43.2s (88%)"
        );
        let reordered = PermitStats {
            reorders: 3,
            max_overtaken: 2,
            commits_first: true,
            ..stats
        };
        assert_eq!(
            reordered.describe(43.2),
            format!(
                "{line} · commits-first: 3 commit(s) served ahead of an older prove, a prove \
                 overtaken at most 2 time(s)"
            )
        );
    }

    /// ★ Each hold line names the order that served it, so an arm's log says
    /// which arm it is.
    #[test]
    fn the_hold_line_names_the_order() {
        assert_eq!(order_note(Order::Fifo, true, 0, 0), " · order fifo");
        assert_eq!(order_note(Order::Fifo, false, 0, 0), " · order fifo");
        assert_eq!(
            order_note(Order::CommitsFirst, true, 0, 2),
            " · order commits-first · passed 2"
        );
        assert_eq!(
            order_note(Order::CommitsFirst, false, 3, 0),
            " · order commits-first · overtaken 3"
        );
    }

    #[test]
    fn the_order_parses_and_a_misspelt_arm_is_refused() {
        assert_eq!(Order::parse(None), Order::Fifo);
        assert_eq!(Order::parse(Some("")), Order::Fifo);
        assert_eq!(Order::parse(Some("fifo")), Order::Fifo);
        assert_eq!(Order::parse(Some("commits-first")), Order::CommitsFirst);
        for bad in ["commit-first", "COMMITS-FIRST", "1", "lifo"] {
            assert!(
                std::panic::catch_unwind(|| Order::parse(Some(bad))).is_err(),
                "`{bad}` must be refused, not read as the control"
            );
        }
    }

    /// The pure policy, one step at a time: a commit that arrives after two
    /// proves is served first and counts one overtake on each; the proves then
    /// go in their own order.
    #[test]
    fn the_gate_serves_a_waiting_commit_before_older_proves() {
        let mut g = GateState::new();
        let p0 = g.enqueue(false);
        let p1 = g.enqueue(false);
        let c2 = g.enqueue(true);
        assert_eq!(g.next(MAX_OVERTAKES), Some(c2), "the commit goes first");
        let (served, passed) = g.grant(c2);
        assert!(served.commit && passed == 2, "it passed both proves");
        g.held = false;
        assert_eq!(
            g.next(MAX_OVERTAKES),
            Some(p0),
            "then the proves, oldest first"
        );
        let (served, _) = g.grant(p0);
        assert_eq!(served.overtaken, 1, "p0 was overtaken once");
        g.held = false;
        let (served, _) = g.grant(g.next(MAX_OVERTAKES).expect("p1 waits"));
        assert_eq!((served.ticket, served.overtaken), (p1, 1));
        assert_eq!(g.next(MAX_OVERTAKES), None, "nobody left");
    }

    /// Within a class the gate is FIFO, and a commit older than every prove
    /// passes nobody.
    #[test]
    fn the_gate_serves_each_class_in_arrival_order() {
        let mut g = GateState::new();
        let c0 = g.enqueue(true);
        let p1 = g.enqueue(false);
        let c2 = g.enqueue(true);
        let p3 = g.enqueue(false);
        let mut order = Vec::new();
        while let Some(t) = g.next(MAX_OVERTAKES) {
            let (served, passed) = g.grant(t);
            order.push((served.ticket, passed));
            g.held = false;
        }
        assert_eq!(order, vec![(c0, 0), (c2, 1), (p1, 0), (p3, 0)]);
    }

    /// ★ NO PROVE STARVES. A prove waits while commits keep arriving: the gate
    /// serves `max` of them ahead of it, then the prove, then the rest — and over
    /// every interleaving tried, no prove is ever overtaken more than `max`
    /// times and none is passed by a commit younger than it once it has been.
    #[test]
    fn no_prove_is_overtaken_more_than_the_bound() {
        for max in [1u32, 2, 3, MAX_OVERTAKES] {
            let mut g = GateState::new();
            let p = g.enqueue(false);
            let commits: Vec<u64> = (0..max + 3).map(|_| g.enqueue(true)).collect();
            let mut served = Vec::new();
            while let Some(t) = g.next(max) {
                let (w, _) = g.grant(t);
                served.push(w.ticket);
                g.held = false;
            }
            let at = served
                .iter()
                .position(|&t| t == p)
                .expect("the prove is served");
            assert_eq!(
                at, max as usize,
                "max {max}: the prove comes right after {max} commits"
            );
            assert_eq!(served.len(), commits.len() + 1);
        }

        // A deterministic walk over mixed arrivals and grants (an LCG, so the
        // run is the same everywhere): the bound holds at every step.
        let mut seed: u64 = 0x2545_f491_4f6c_dd1d;
        let mut step = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (seed >> 33) as u32
        };
        for max in [1u32, 2, 4] {
            let mut g = GateState::new();
            for _ in 0..4000 {
                if step() % 3 != 0 || g.waiting.is_empty() {
                    g.enqueue(step() % 2 == 0);
                } else if let Some(t) = g.next(max) {
                    let (w, _) = g.grant(t);
                    g.held = false;
                    assert!(
                        w.overtaken <= max,
                        "a prove was overtaken {} > {max} times",
                        w.overtaken
                    );
                }
                for w in &g.waiting {
                    assert!(
                        w.overtaken <= max,
                        "a waiting prove at {} > {max}",
                        w.overtaken
                    );
                }
            }
        }
    }

    /// Waits until `gate` holds exactly `commits` waiting commits and `proves`
    /// waiting proves, so a threaded test releases the card only once its queue
    /// is the one it means.
    fn wait_for_queue(gate: &Gate, commits: usize, proves: usize) {
        let t = Instant::now();
        loop {
            let (c, p, _) = gate.snapshot();
            if (c, p) == (commits, proves) {
                return;
            }
            assert!(
                t.elapsed() < std::time::Duration::from_secs(10),
                "the gate never queued {commits} commit(s) and {proves} prove(s): it has {c} and {p}"
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// ★ A gate, threaded, with its queue FORCED: a prove waits behind a
    /// holder, then ten commits arrive. On release the gate serves eight
    /// commits, then the prove, then the last two — whichever threads they are.
    ///
    /// ⓘ On a gate of its own, not the process's: any other test that builds
    /// artifacts while the permit is armed joins the process gate's queue, and
    /// this test needs the queue it forced.
    #[test]
    fn a_waiting_prove_is_served_within_the_bound() {
        let gate = Gate::new();
        let served: Mutex<Vec<bool>> = Mutex::new(Vec::new());
        std::thread::scope(|s| {
            gate.acquire(false);
            s.spawn(|| {
                gate.acquire(false);
                served.lock().expect("the order lock").push(false);
                gate.release();
            });
            wait_for_queue(&gate, 0, 1);
            for _ in 0..(MAX_OVERTAKES + 2) {
                s.spawn(|| {
                    gate.acquire(true);
                    served.lock().expect("the order lock").push(true);
                    gate.release();
                });
            }
            wait_for_queue(&gate, MAX_OVERTAKES as usize + 2, 1);
            gate.release();
        });
        let served = served.into_inner().expect("the order lock");
        let mut want = vec![true; MAX_OVERTAKES as usize];
        want.push(false);
        want.extend([true, true]);
        assert_eq!(
            served, want,
            "commits first, the prove after {MAX_OVERTAKES}, then the rest"
        );
    }

    /// ★ ONLY THE ORDER CHANGES. The same two holders behind the same forced
    /// queue — one prove waiting, then one commit — under the `Mutex` and under
    /// the gate: every holder computes the same thing, and under the gate the
    /// late commit goes first and passes one prove.
    #[test]
    fn only_the_order_changes_between_the_two_orders() {
        let work = |i: u64| -> u64 {
            // A deterministic stand-in for a device phase's output.
            (0..10_000u64).fold(i, |h, k| {
                h.rotate_left(5) ^ k.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            })
        };
        let mut outputs = Vec::new();
        for order in [Order::Fifo, Order::CommitsFirst] {
            let card: Mutex<()> = Mutex::new(());
            let gate = Gate::new();
            let results: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
            let firsts: Mutex<Vec<(&'static str, u32)>> = Mutex::new(Vec::new());
            // One holder's device phase, under this order's lock.
            let phase = |commit: bool, i: u64| match order {
                Order::Fifo => {
                    let _card = card.lock().expect("the card");
                    firsts
                        .lock()
                        .expect("lock")
                        .push((if commit { "commit" } else { "prove" }, 0));
                    results.lock().expect("lock").push((i, work(i)));
                }
                Order::CommitsFirst => {
                    let (_, passed) = gate.acquire(commit);
                    firsts
                        .lock()
                        .expect("lock")
                        .push((if commit { "commit" } else { "prove" }, passed));
                    results.lock().expect("lock").push((i, work(i)));
                    gate.release();
                }
            };
            std::thread::scope(|s| {
                let fifo_holder = (order == Order::Fifo).then(|| card.lock().expect("the card"));
                if order == Order::CommitsFirst {
                    gate.acquire(false);
                }
                s.spawn(|| phase(false, 1));
                if order == Order::CommitsFirst {
                    wait_for_queue(&gate, 0, 1);
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                s.spawn(|| phase(true, 2));
                if order == Order::CommitsFirst {
                    wait_for_queue(&gate, 1, 1);
                    gate.release();
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                drop(fifo_holder);
            });
            let firsts = firsts.into_inner().expect("lock");
            if order == Order::CommitsFirst {
                assert_eq!(
                    firsts,
                    [("commit", 1), ("prove", 0)],
                    "the late commit went first, past the one waiting prove"
                );
            }
            let mut results = results.into_inner().expect("lock");
            results.sort_unstable();
            outputs.push(results);
        }
        assert_eq!(
            outputs[0], outputs[1],
            "every holder computed the same thing"
        );
        assert_eq!(outputs[0].len(), 2);
    }

    /// The phase names the drivers pass decide the class: `build_artifacts`
    /// waits as a commit, everything else as a prove.
    #[test]
    fn only_build_artifacts_waits_as_a_commit() {
        assert!(is_commit("build_artifacts"));
        for phase in ["multi_prove", "card", "build"] {
            assert!(!is_commit(phase), "{phase}");
        }
    }
}
