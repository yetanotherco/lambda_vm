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
//! # Under one shared gate, no exclusion
//!
//! With `LAMBDA_VM_SHARED_VRAM_GATE=1` (`stark::prover::shared_vram_gate_on`)
//! both device paths above take their bytes from ONE process-wide
//! `VramGate`: every `multi_prove` admits its tables through it, and the
//! artifact commit admits its device set through it
//! (`commit_group_device_or_host_with`). That is the cross-proof running total
//! this permit stands in for, so [`hold`] then takes no card: sibling proofs
//! share the card by bytes, and one proof's host stages (uploads, transcript,
//! queries) no longer leave the card idle while the others wait.
//!
//! Two bounds keep that within the card (FAST 473, 474: six or seven proves at
//! once overran it with the budget alone):
//! - at most [`shared_proves`] proofs are inside `multi_prove` at once, so the
//!   bytes each prove holds beyond its tables' estimates multiply by a small
//!   number;
//! - [`arm`] calibrates the shared gate to the card's free memory less
//!   [`shared_margin_bytes`] (`stark::prover::arm_shared_vram_gate`), so what
//!   no gate counts is outside the budget rather than on top of it.
//!
//! # Inert until armed
//!
//! Unarmed, and at one worker, [`hold`] takes no lock and touches no counter on
//! the contended path: it reads one relaxed `usize` plus the round-3 tree
//! probe's cached `OnceLock<bool>` and returns. Nothing that ships arms it, and
//! nothing that ships sets `LAMBDA_VM_TREE_BUSY_PROBE`.

use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

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
}

/// Arm the permit for `workers` concurrent proofs. `workers <= 1` leaves it
/// inert, which is the control arm on the same binary. Under the shared gate's
/// knob it also arms (and calibrates) or disarms that gate.
pub fn arm(workers: usize) {
    WORKERS.store(workers.max(1), Ordering::Relaxed);
    stark::prover::arm_shared_vram_gate(workers > 1, shared_margin_bytes());
}

/// `LAMBDA_VM_SHARED_GATE_PROVES=n` (n ≥ 1): the proofs that may be inside
/// `multi_prove` at once under the shared gate. Unset: 3.
pub fn shared_proves() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("LAMBDA_VM_SHARED_GATE_PROVES")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(3)
    })
}

/// The shared gate's margin below the card's free memory: 2 GiB, plus 0.5 GiB
/// per proof allowed inside `multi_prove` at once (what each holds beyond its
/// tables' estimates, FAST 474).
pub fn shared_margin_bytes() -> u64 {
    (2u64 << 30) + (shared_proves() as u64) * (1u64 << 29)
}

/// Proofs inside `multi_prove` under the shared gate right now, and the wait
/// for a place.
static PROVES_IN: Mutex<usize> = Mutex::new(0);
static PROVE_ROOM: std::sync::Condvar = std::sync::Condvar::new();

/// A place among [`shared_proves`], released on drop.
struct ProveSlot;

impl ProveSlot {
    fn take() -> Self {
        let cap = shared_proves();
        let mut n = PROVES_IN.lock().unwrap_or_else(|e| e.into_inner());
        while *n >= cap {
            n = PROVE_ROOM.wait(n).unwrap_or_else(|e| e.into_inner());
        }
        *n += 1;
        ProveSlot
    }
}

impl Drop for ProveSlot {
    fn drop(&mut self) {
        let mut n = PROVES_IN.lock().unwrap_or_else(|e| e.into_inner());
        *n = n.saturating_sub(1);
        drop(n);
        PROVE_ROOM.notify_one();
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

/// Exclusive use of the card, for as long as it is held (under the shared
/// gate: one of its proof places, for a `multi_prove`).
pub struct CardPermit {
    guard: Option<std::sync::MutexGuard<'static, ()>>,
    /// Under the shared gate, a `multi_prove`'s place among [`shared_proves`].
    _slot: Option<ProveSlot>,
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
                "CARD HOLD #{seq} {phase}: waited {waited:.3}s · held {:.3}s · t=[{t0:.3},{:.3}]",
                self.since.elapsed().as_secs_f64(),
                stark::prove_split::epoch_secs(),
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
    // Under the shared gate the bytes are the exclusion (see the module doc):
    // no card to take, a `multi_prove` takes a proof place, and the window is
    // still traced.
    let shared = stark::prover::shared_vram_gate_on();
    if shared {
        let blocked_from = Instant::now();
        let slot = (phase == "multi_prove").then(ProveSlot::take);
        let waited = blocked_from.elapsed();
        WAITED_NANOS.with(|w| w.set(w.get() + waited.as_nanos() as u64));
        return CardPermit {
            guard: None,
            _slot: slot,
            since: Instant::now(),
            probe: probed.then_some((phase, waited.as_nanos() as u64)),
            trace: traced.then(|| {
                (
                    phase,
                    waited.as_secs_f64(),
                    TRACE_SEQ.fetch_add(1, Ordering::Relaxed),
                    stark::prove_split::epoch_secs(),
                )
            }),
        };
    }
    if workers() <= 1 && !traced && !probed {
        return CardPermit {
            guard: None,
            _slot: None,
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
            _slot: None,
            since: Instant::now(),
            probe: probed.then_some((phase, 0)),
            trace: traced.then(|| (phase, 0.0, seq, stark::prove_split::epoch_secs())),
        };
    }
    assert!(
        !HELD_HERE.with(|h| h.get()),
        "the card permit is not reentrant and this thread already holds it; \
         taking it twice without releasing parks the thread forever"
    );
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
        _slot: None,
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ★ UNDER THE SHARED GATE AT MOST `shared_proves` PROOFS ARE INSIDE
    /// `multi_prove`. With the places full, one more `multi_prove` waits until a
    /// place is released, while an artifact build takes no place.
    #[test]
    fn under_the_shared_gate_a_prove_waits_for_a_place() {
        let g = ARM.lock().expect("the arm guard is never poisoned");
        disarm(&g);
        stark::prover::pin_shared_vram_gate(Some(true));
        let cap = shared_proves();
        // `cap` holders, each on its own thread (a permit is not `Send`),
        // holding until told to release.
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = std::sync::Arc::new(Mutex::new(release_rx));
        let (taken_tx, taken_rx) = std::sync::mpsc::channel::<()>();
        let holders: Vec<_> = (0..cap)
            .map(|_| {
                let (rx, taken) = (release_rx.clone(), taken_tx.clone());
                std::thread::spawn(move || {
                    let _p = hold_labeled("multi_prove");
                    taken.send(()).expect("the test is listening");
                    let _ = rx.lock().expect("never poisoned").recv();
                })
            })
            .collect();
        for _ in 0..cap {
            taken_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("a place is free");
        }
        let _artifacts = hold_labeled("build_artifacts");
        let (tx, rx) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let _p = hold_labeled("multi_prove");
            tx.send(()).expect("the test is listening");
        });
        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "a prove beyond the places must wait"
        );
        release_tx.send(()).expect("a holder is listening");
        rx.recv_timeout(std::time::Duration::from_secs(5))
            .expect("a released place lets it in");
        waiter.join().expect("the waiter finishes");
        drop(release_tx);
        for h in holders {
            h.join().expect("a holder finishes");
        }
        stark::prover::pin_shared_vram_gate(None);
        disarm(&g);
    }

    /// ★ UNDER THE SHARED GATE THE PERMIT TAKES NO CARD. Armed for siblings, a
    /// second hold on this thread would park it if the card were taken, and a
    /// hold on another thread would trip the two-holders assert; under the
    /// shared gate neither happens, because the bytes are the exclusion.
    #[test]
    fn under_the_shared_gate_the_permit_takes_no_card() {
        let g = ARM.lock().expect("the arm guard is never poisoned");
        disarm(&g);
        stark::prover::pin_shared_vram_gate(Some(true));
        arm(4);
        {
            let _a = hold_labeled("multi_prove");
            let _b = hold_labeled("build_artifacts");
            std::thread::scope(|s| {
                s.spawn(|| {
                    let _c = hold_labeled("multi_prove");
                });
            });
        }
        assert_eq!(take_stats().acquisitions, 0, "no card was taken");
        stark::prover::pin_shared_vram_gate(None);
        disarm(&g);
    }

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
