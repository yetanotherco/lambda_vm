//! Returning the allocator's freed pages to the OS at a phase boundary
//! (`LAMBDA_VM_ALLOC_PURGE`; by default only where memory is short).
//!
//! The posture's jemalloc never purges (`dirty_decay_ms:-1`), and a phase
//! rarely reuses the pages the one before it freed: they sit in other threads'
//! arenas or in size classes the next phase does not ask for. So a block's RSS
//! ratchets phase by phase. At the spill-on median (BIG 584) the live peaks are
//! flat, ≈ 36 GiB in the base, the tree and the verifier (decay 0), while
//! VmRSS climbs 52.75 → 56.56 → 60.28 GiB. One
//! `arena.<MALLCTL_ARENAS_ALL>.purge` hands every arena's dirty pages back, so
//! the next phase starts near its live set. Decay 0 does the same continuously
//! at +54 s of base.
//!
//! [`purge_point`] is the call a pipeline makes at a named boundary. It purges
//! only in the lib's own test builds, whose global allocator is jemalloc
//! (`lib.rs`); anywhere else the binary owns the allocator, and the call does
//! nothing.
//!
//! By default (`auto`) a purge runs only once the block's memory is short: at
//! phase A's end and the base's end, after the spill's queue budgets armed
//! ([`note_memory_pressure`], `block.rs`). At the p90 block 25481021 those two
//! purges took the base's phase-B peak 119.79 → 116.60 GiB and level 0's start
//! ≈ 115 → 55.6 GiB, where without them level 0 was OOM-killed (BIG 120, 1215);
//! a block that fits never arms, so it pays nothing.

/// What one purge did.
#[derive(Clone, Copy, Debug)]
pub struct Purge {
    /// The purge's own wall time.
    pub secs: f64,
    /// jemalloc's resident bytes before and after (live, dirty and metadata).
    pub resident_before: usize,
    pub resident_after: usize,
}

/// `LAMBDA_VM_ALLOC_PURGE`: `auto` (and unset or empty) purges at
/// [`AUTO_POINTS`] once memory is short ([`note_memory_pressure`]); `off`
/// purges nowhere; `all`, or a list of the boundary names [`purge_point`] is
/// called with, separated by commas or dots (`base.tree`, for job specs that
/// split their knobs on commas), purges there regardless. The block's points:
/// `phase-a` (phase A done, `block.rs`), `base` (the base proved) and `tree`
/// (the top proved, before the block verifier), the last two in the
/// whole-block harness.
pub const ALLOC_PURGE_ENV: &str = "LAMBDA_VM_ALLOC_PURGE";

/// The points `auto` purges at under memory pressure. `tree` is not one: it
/// trims the block verifier's start, not the proof's peak.
const AUTO_POINTS: [&str; 2] = ["phase-a", "base"];

/// Whether the block in progress found its memory short.
static PRESSURE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The block's memory is short (its spill's queue budgets armed): `auto`
/// purges at its next points.
pub fn note_memory_pressure() {
    PRESSURE.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// A new block starts: no pressure seen yet.
pub fn clear_memory_pressure() {
    PRESSURE.store(false, std::sync::atomic::Ordering::Relaxed);
}

/// What the knob says at a point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decision {
    Purge,
    /// `auto` at one of its points, with no memory pressure.
    Skip,
    /// Not a point the knob purges at.
    No,
}

fn decide(setting: Option<&str>, point: &str, pressure: bool) -> Decision {
    match setting.map(str::trim) {
        None | Some("") | Some("auto") if AUTO_POINTS.contains(&point) => {
            if pressure {
                Decision::Purge
            } else {
                Decision::Skip
            }
        }
        None | Some("") | Some("auto") | Some("off") => Decision::No,
        Some(setting) if names_point(setting, point) => Decision::Purge,
        Some(_) => Decision::No,
    }
}

/// Whether `point` is one of the boundaries `setting` names.
fn names_point(setting: &str, point: &str) -> bool {
    setting
        .split([',', '.'])
        .map(str::trim)
        .any(|p| p == "all" || p == point)
}

/// At the boundary `point`, purge every jemalloc arena's dirty pages when
/// [`ALLOC_PURGE_ENV`] says so ([`decide`]), and print one `ALLOC PURGE` line
/// (the wall time and jemalloc's resident bytes before and after, or that
/// `auto` skipped it). Returns what it did, or `None` when it did not purge or
/// the build cannot.
pub fn purge_point(point: &str) -> Option<Purge> {
    let setting = std::env::var(ALLOC_PURGE_ENV).ok();
    let pressure = PRESSURE.load(std::sync::atomic::Ordering::Relaxed);
    match decide(setting.as_deref(), point, pressure) {
        Decision::No => return None,
        Decision::Skip => {
            eprintln!("ALLOC PURGE {point}: skipped (auto, no memory pressure)");
            return None;
        }
        Decision::Purge => {}
    }
    let purge = purge_all_arenas()?;
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    eprintln!(
        "ALLOC PURGE {point}: {:.3}s · jemalloc resident {:.2} → {:.2} GiB",
        purge.secs,
        gib(purge.resident_before),
        gib(purge.resident_after)
    );
    Some(purge)
}

/// `arena.<MALLCTL_ARENAS_ALL>.purge` (4096, jemalloc's "every arena" index): a
/// control that neither reads nor writes, so every pointer is null.
#[cfg(test)]
fn purge_all_arenas() -> Option<Purge> {
    use tikv_jemalloc_ctl::{epoch, stats};
    let resident = || -> Option<usize> {
        epoch::advance().ok()?;
        stats::resident::read().ok()
    };
    let resident_before = resident()?;
    let t = std::time::Instant::now();
    // SAFETY: the name is NUL-terminated, and a control that takes no value is
    // called with null old and new pointers and zero lengths, as jemalloc's
    // `NEITHER_READ_NOR_WRITE` requires.
    let rc = unsafe {
        tikv_jemalloc_sys::mallctl(
            c"arena.4096.purge".as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        )
    };
    let secs = t.elapsed().as_secs_f64();
    if rc != 0 {
        eprintln!("ALLOC PURGE: arena.4096.purge returned {rc}");
        return None;
    }
    Some(Purge {
        secs,
        resident_before,
        resident_after: resident()?,
    })
}

/// Outside the lib's tests the binary owns the allocator: nothing to purge.
#[cfg(not(test))]
fn purge_all_arenas() -> Option<Purge> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The knob names its boundaries: `all`, or a list split on commas or dots,
    /// spaces allowed.
    #[test]
    fn the_purge_knob_names_its_boundaries() {
        assert!(names_point("all", "base"));
        assert!(names_point("base,tree", "tree"));
        assert!(names_point("base.tree", "base"));
        assert!(names_point("phase-a.base", "phase-a"));
        assert!(names_point(" base , tree ", "base"));
        assert!(!names_point("base,tree", "phase-a"));
        assert!(!names_point("", "base"));
    }

    /// `auto` (the default) purges at phase A's end and the base's end only
    /// under memory pressure, and says it skipped otherwise; never at `tree`;
    /// `off` purges nowhere; `all` and a list purge regardless of pressure.
    #[test]
    fn auto_purges_only_under_memory_pressure() {
        for setting in [None, Some(""), Some("auto"), Some(" auto ")] {
            for point in ["phase-a", "base"] {
                assert_eq!(
                    decide(setting, point, true),
                    Decision::Purge,
                    "{setting:?} {point}"
                );
                assert_eq!(
                    decide(setting, point, false),
                    Decision::Skip,
                    "{setting:?} {point}"
                );
            }
            assert_eq!(decide(setting, "tree", true), Decision::No);
        }
        for point in ["phase-a", "base", "tree"] {
            assert_eq!(decide(Some("off"), point, true), Decision::No);
            assert_eq!(decide(Some("all"), point, false), Decision::Purge);
        }
        assert_eq!(decide(Some("base.tree"), "tree", false), Decision::Purge);
        assert_eq!(decide(Some("base.tree"), "phase-a", true), Decision::No);
    }

    /// A purge hands back the pages freed buffers left: 512 buffers of 1 MiB,
    /// touched and freed on this thread. Under the posture's never-purge
    /// allocator they stay resident until the purge, which returns at least
    /// half of them (laptop: 525 → 4 MiB in 4 ms). Under jemalloc's default decay
    /// they may be gone already, so the test asks for the posture.
    #[test]
    #[ignore = "run under the posture: _RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1"]
    fn a_purge_returns_freed_buffers_pages() {
        let resident = || {
            tikv_jemalloc_ctl::epoch::advance().expect("the epoch turns");
            tikv_jemalloc_ctl::stats::resident::read().expect("stats.resident reads")
        };
        let before = resident();
        let buffers: Vec<Vec<u8>> = (0..512).map(|_| vec![1u8; 1 << 20]).collect();
        drop(buffers);
        let retained = resident();
        assert!(
            retained >= before + (256 << 20),
            "the freed buffers were not retained ({} → {} MiB): run under the posture's allocator",
            before >> 20,
            retained >> 20
        );
        let purge = purge_all_arenas().expect("the test build purges");
        assert!(
            purge.resident_before >= purge.resident_after + (256 << 20),
            "resident {} → {} MiB: the purge returned less than half of 512 MiB freed",
            purge.resident_before >> 20,
            purge.resident_after >> 20
        );
        println!(
            "PURGE UNIT: {:.3}s · resident {} → {} MiB",
            purge.secs,
            purge.resident_before >> 20,
            purge.resident_after >> 20
        );
    }
}
