//! ⛔ ROUND-3 TREE/WRAP DISCRIMINATOR (diagnostic; on NO production decision
//! path, OFF by default, byte-identical when off).
//!
//! # The question it answers
//!
//! Rounds 1 and 2 landed on the WHIR BASE and round 3 measured the base at two
//! floors (the RPX permutation's arithmetic floor and argue's Fiat-Shamir
//! sequencing floor). What was never examined is the ~55 s ABOVE the base: the
//! recursion tree — level 0's epoch wraps, the cross-epoch GLOBAL child, the
//! interior fold levels, and the block-artifact root. Either that phase has
//! CONCURRENCY SLACK (a lever: overlap stages that are independent) or it is
//! already saturated and bound by the same permutation (a floor).
//!
//! # ⛔ Why this measures the PERMIT and not SM occupancy
//!
//! The tree phase's serialising resource is not the card's utilisation, it is
//! [`super::device_permit`] — MUTUAL EXCLUSION over the two device phases of a
//! proof, because two proofs in flight can ask the card for twice its budget
//! from two directions (`build_artifacts`' per-dispatch admission has no running
//! total; each `multi_prove` builds its own full-budget `VramGate`). So even a
//! card that idles INSIDE a device phase cannot be filled by another sibling:
//! the permit excludes it, and the acquire panics rather than allowing it.
//!
//! ⇒ the ceiling on every concurrency lever in this phase is `Σ held`, the time
//! spent inside a device phase. `held / stage wall` is therefore the number that
//! discriminates:
//!
//!   near 1  ⇒ the card is the wall; more siblings buy nothing and the only
//!             lever left is making the device phases themselves cheaper —
//!             which is the permutation, already at its arithmetic floor;
//!   well under 1 ⇒ the phase idles the card while workers walk the host, and
//!             overlapping independent work recovers it.
//!
//! # What is missing from the shipped instruments, and why this exists
//!
//! ✓ `PermitStats::describe` already prints exactly this reading — but only
//! where a level calls `take_stats()`, and there are three such sites: the
//! interior's BARRIER loop and the two drivers' level 0. The production harness
//! runs with `LFM_TREE_LEVEL_POOL=1` and `LFM_TREE_LEVEL_POOL_FROM` at its
//! default of 1, so `barrier_levels` is 0, the barrier loop runs ZERO levels and
//! the whole interior goes through the POOLED span — which prints no permit
//! line. And the GLOBAL child and the ROOT run with the permit armed at 1, where
//! `hold` returns early with no guard and `Drop` accumulates nothing, so their
//! held time exists only on the per-hold `LFM_CARD_TRACE` lines and in no
//! summary at all.
//!
//! ⇒ this probe accumulates on EVERY hold, armed or not, into counters of its
//! own — it never touches `HELD_NANOS` or `take_stats()`, so every line the
//! drivers already print keeps its exact current meaning.
//!
//! # ★ The second reading: how much of the `build_artifacts` hold is DEVICE
//!
//! ✓ The permit is held across the WHOLE of `build_artifacts_with_hasher`,
//! while the only device work inside it is one `try_commit_row_major` per column
//! group that clears the admission floor (`super::commit::commit_group_device_or_host`,
//! and ✓ its only two production call sites are both inside that build —
//! `registry.rs`'s prep-group and blake3-chunk walks). If the dispatch total is
//! much smaller than the hold, the permit is held over HOST work and narrowing
//! its scope would free the card — a second, independent lever with a mechanism.
//! Measured with a host stopwatch around a host-blocking call: ⛔ NO CUDA event
//! and NO added synchronize, so the device path cannot be perturbed by the
//! measurement even when it is on.
//!
//! ⚠⚠ THE DISPATCH TOTAL IS WORKER-SECONDS, NOT A WALL, and mixing the two is
//! the way to misread this line. ✓ Both walks go through `map_maybe_parallel`,
//! which runs on rayon when `LFM_ARTIFACT_PARALLEL` is not `0` (default ON) over
//! windows of `LFM_ARTIFACT_GROUPS_IN_FLIGHT` groups (default 4). So up to four
//! dispatches overlap and the total can EXCEED the hold it is quoted against:
//! above 100% is concurrency, not a bug. What the reading still says is the
//! thing it is for — a ratio WELL UNDER 100% means the card was idle inside an
//! exclusive hold even allowing for that concurrency.
//!
//! # Cost when off
//!
//! [`enabled`] is a `OnceLock<bool>` load. Every call site reads it once per
//! HOLD or per device dispatch — tens of times in a whole block run, never in a
//! loop — and does nothing else. Off, no counter moves and no line prints.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// `LAMBDA_VM_TREE_BUSY_PROBE=1` (any non-empty value other than `0`) turns it
/// on, read ONCE and cached.
///
/// ⚠ Unset, empty and `0` all read as OFF — `trace_enabled`'s spelling, not the
/// argue probe's `is_some()`. `FOO=` is the shell clearing a variable and
/// `FOO=0` is somebody switching a knob off; a probe that turned ON for either
/// would put a diagnostic line into a run nobody asked for one in.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("LAMBDA_VM_TREE_BUSY_PROBE") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    })
}

/// Held nanoseconds inside `build_artifacts` device phases, and how many.
static BUILD_NS: AtomicU64 = AtomicU64::new(0);
static BUILD_HOLDS: AtomicU64 = AtomicU64::new(0);
/// Held nanoseconds inside `multi_prove` device phases, and how many.
static PROVE_NS: AtomicU64 = AtomicU64::new(0);
static PROVE_HOLDS: AtomicU64 = AtomicU64::new(0);
/// Held nanoseconds under any other label, so a new hold site cannot be lost.
static OTHER_NS: AtomicU64 = AtomicU64::new(0);
static OTHER_HOLDS: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds workers spent QUEUED on the card — nonzero only where the permit
/// is armed above one.
static WAITED_NS: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds inside the device commit dispatch, and how many groups reached it.
static DEVICE_COMMIT_NS: AtomicU64 = AtomicU64::new(0);
static DEVICE_COMMIT_CALLS: AtomicU64 = AtomicU64::new(0);

/// Record one released card permit: which phase it was, how long it was held,
/// and how long its holder queued for it.
///
/// Called from `device_permit::CardPermit::drop` for EVERY hold — including the
/// unarmed ones a `K = 1` stage takes, which is the whole reason this exists.
pub fn note_hold(phase: &'static str, held_nanos: u64, waited_nanos: u64) {
    let (ns, holds) = match phase {
        "build_artifacts" => (&BUILD_NS, &BUILD_HOLDS),
        "multi_prove" => (&PROVE_NS, &PROVE_HOLDS),
        _ => (&OTHER_NS, &OTHER_HOLDS),
    };
    ns.fetch_add(held_nanos, Ordering::Relaxed);
    holds.fetch_add(1, Ordering::Relaxed);
    WAITED_NS.fetch_add(waited_nanos, Ordering::Relaxed);
}

/// Record one device commit dispatch — the admitted `try_commit_row_major`
/// inside `build_artifacts`.
pub fn note_device_commit(nanos: u64) {
    DEVICE_COMMIT_NS.fetch_add(nanos, Ordering::Relaxed);
    DEVICE_COMMIT_CALLS.fetch_add(1, Ordering::Relaxed);
}

/// One stage's counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub build_holds: u64,
    pub build_nanos: u64,
    pub prove_holds: u64,
    pub prove_nanos: u64,
    pub other_holds: u64,
    pub other_nanos: u64,
    pub waited_nanos: u64,
    pub device_commit_calls: u64,
    pub device_commit_nanos: u64,
}

impl Stats {
    /// Seconds this stage spent inside a device phase — the serialised total
    /// every concurrency lever is bounded by.
    pub fn held_secs(&self) -> f64 {
        (self.build_nanos + self.prove_nanos + self.other_nanos) as f64 / 1e9
    }

    /// How many holds the stage took.
    pub fn holds(&self) -> u64 {
        self.build_holds + self.prove_holds + self.other_holds
    }

    /// ★ THE LINE THE DISCRIMINATOR IS READ OFF.
    ///
    /// ⛔ NO DEVICE-MEMORY FIELD, and its removal is a finding rather than a
    /// simplification. Splitting `math_cuda::device::reserved_high_water` per
    /// stage means CUTTING a process-wide counter at every boundary, which
    /// costs the harness's own whole-run `reserved high-water` line its
    /// meaning — and the free read of wt27/wt28 showed the number was not worth
    /// that: the whole-run figure is 25,003 MiB against a 32,607 MiB card, the
    /// 10 Hz `nvidia-smi` trace the harness already keeps splits device memory
    /// by phase, and no tree stage is anywhere near the budget. A measurement
    /// that degrades an existing one has to buy more than that.
    pub fn describe(&self, stage: &str, wall_secs: f64) -> String {
        let held = self.held_secs();
        let pct = |x: f64| {
            if wall_secs > 0.0 {
                100.0 * x / wall_secs
            } else {
                0.0
            }
        };
        format!(
            "TREE PROBE {stage}: wall {wall_secs:.1}s · card-held {held:.1}s ({:.0}%) over {} hold(s) \
             [build_artifacts {:.1}s/{} · multi_prove {:.1}s/{} · other {:.1}s/{}] · \
             queued {:.1}s · device commit dispatch {:.1}s WORKER-SEC over {} group(s) \
             ({:.0}% of the build_artifacts hold — worker-seconds over a wall, so \
             above 100% is concurrency)",
            pct(held),
            self.holds(),
            self.build_nanos as f64 / 1e9,
            self.build_holds,
            self.prove_nanos as f64 / 1e9,
            self.prove_holds,
            self.other_nanos as f64 / 1e9,
            self.other_holds,
            self.waited_nanos as f64 / 1e9,
            self.device_commit_nanos as f64 / 1e9,
            self.device_commit_calls,
            if self.build_nanos > 0 {
                100.0 * self.device_commit_nanos as f64 / self.build_nanos as f64
            } else {
                0.0
            },
        )
    }
}

/// Read and CLEAR every counter, so a stage reports its own.
pub fn take() -> Stats {
    Stats {
        build_holds: BUILD_HOLDS.swap(0, Ordering::Relaxed),
        build_nanos: BUILD_NS.swap(0, Ordering::Relaxed),
        prove_holds: PROVE_HOLDS.swap(0, Ordering::Relaxed),
        prove_nanos: PROVE_NS.swap(0, Ordering::Relaxed),
        other_holds: OTHER_HOLDS.swap(0, Ordering::Relaxed),
        other_nanos: OTHER_NS.swap(0, Ordering::Relaxed),
        waited_nanos: WAITED_NS.swap(0, Ordering::Relaxed),
        device_commit_calls: DEVICE_COMMIT_CALLS.swap(0, Ordering::Relaxed),
        device_commit_nanos: DEVICE_COMMIT_NS.swap(0, Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⛔ The counters are process-global, so the two tests that WRITE them run
    /// one at a time. Without this they interleave and each reads the other's
    /// holds — a flake that would read as the probe losing time.
    ///
    /// ⓘ No other test in this binary can reach them: `note_hold` is called
    /// only from `CardPermit::drop` and only when the permit carries a `probe`
    /// tag, which needs `LAMBDA_VM_TREE_BUSY_PROBE` in the environment, and a
    /// test binary does not set it.
    static COUNTERS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// `take` must CLEAR, so two stages cannot report each other's holds — the
    /// failure that makes a late stage look card-bound because an early one was.
    #[test]
    fn take_clears_so_a_stage_reports_its_own() {
        let _g = COUNTERS.lock().unwrap_or_else(|e| e.into_inner());
        let _ = take();
        note_hold("build_artifacts", 3_000_000_000, 1_000_000_000);
        note_hold("multi_prove", 7_000_000_000, 0);
        note_device_commit(2_000_000_000);
        let first = take();
        assert_eq!(first.build_holds, 1);
        assert_eq!(first.prove_holds, 1);
        assert_eq!(first.device_commit_calls, 1);
        assert!((first.held_secs() - 10.0).abs() < 1e-9);
        let second = take();
        assert_eq!(second, Stats::default(), "the second read must be empty");
    }

    /// An unrecognised phase label lands in `other` rather than being dropped —
    /// a third hold site added later must show up as time somewhere, or the
    /// held fraction silently understates and reads as slack that is not there.
    #[test]
    fn an_unknown_phase_is_counted_not_dropped() {
        let _g = COUNTERS.lock().unwrap_or_else(|e| e.into_inner());
        let _ = take();
        note_hold("a phase nobody has written yet", 5_000_000_000, 0);
        let s = take();
        assert_eq!(s.other_holds, 1);
        assert_eq!(s.build_holds + s.prove_holds, 0);
        assert!((s.held_secs() - 5.0).abs() < 1e-9);
    }

    /// The line carries the fraction the verdict is read off, and the
    /// build-hold's device share beside it.
    #[test]
    fn the_line_reports_the_held_fraction_and_the_dispatch_share() {
        let s = Stats {
            build_holds: 2,
            build_nanos: 4_000_000_000,
            prove_holds: 2,
            prove_nanos: 6_000_000_000,
            device_commit_calls: 9,
            device_commit_nanos: 1_000_000_000,
            ..Stats::default()
        };
        let line = s.describe("level 0", 20.0);
        assert!(line.contains("card-held 10.0s (50%)"), "{line}");
        assert!(line.contains("over 4 hold(s)"), "{line}");
        assert!(
            line.contains("dispatch 1.0s WORKER-SEC over 9 group(s) (25% of the"),
            "{line}"
        );
    }

    /// ⛔ A stage with no holds must not divide by zero into a fake percentage.
    #[test]
    fn an_empty_stage_reports_zero_rather_than_nan() {
        let line = Stats::default().describe("nothing", 0.0);
        assert!(line.contains("card-held 0.0s (0%)"), "{line}");
        assert!(line.contains("over 0 group(s) (0% of the"), "{line}");
    }
}
