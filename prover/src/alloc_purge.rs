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
//! [`purge_point`] is the call a pipeline makes at a named boundary. The lib
//! does not own the allocator, the binary does: it purges through the
//! [`AllocatorHooks`] the binary [`install`]s (the CLI installs jemalloc's), or,
//! in the lib's own test builds, through the jemalloc their `lib.rs` installs.
//! Anywhere else the call does nothing. The same hooks give the allocator's
//! statistics ([`stats`]) to the readers that need them: the block's memory
//! ledger and the tree's late leaf emission.
//!
//! By default (`auto`) a purge runs only once the block's memory is short: at
//! phase A's end and the base's end, after the spill's queue budgets armed
//! ([`note_memory_pressure`], `block.rs`). At the p90 block 25481021 those two
//! purges took the base's phase-B peak 119.79 → 116.60 GiB and level 0's start
//! ≈ 115 → 55.6 GiB, where without them level 0 was OOM-killed (BIG 120, 1215);
//! a block that fits never arms, so it pays nothing.

/// The allocator's statistics, in bytes (jemalloc's `stats.*`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocStats {
    /// Live: what the program asked for and has not freed.
    pub allocated: usize,
    /// In active pages.
    pub active: usize,
    /// Held in pages: live, freed-and-dirty, and metadata (an upper bound: a
    /// fresh extent counts before it is touched).
    pub resident: usize,
    pub mapped: usize,
    pub retained: usize,
}

/// The binary's allocator, as the lib may ask it: its statistics, and a purge
/// of every arena's dirty pages (`true` when it ran).
#[derive(Clone, Copy, Debug)]
pub struct AllocatorHooks {
    pub stats: fn() -> Option<AllocStats>,
    pub purge_all: fn() -> bool,
}

static HOOKS: std::sync::OnceLock<AllocatorHooks> = std::sync::OnceLock::new();

/// The binary installs its allocator's hooks, once, at startup. Returns `false`
/// when hooks were already installed (the first stay).
pub fn install(hooks: AllocatorHooks) -> bool {
    HOOKS.set(hooks).is_ok()
}

/// The installed hooks, else the lib's own test build's jemalloc.
fn hooks() -> Option<AllocatorHooks> {
    HOOKS.get().copied().or(BUILT_IN)
}

/// The lib's test build installs jemalloc as its global allocator (`lib.rs`).
#[cfg(test)]
const BUILT_IN: Option<AllocatorHooks> = Some(AllocatorHooks {
    stats: jemalloc_stats,
    purge_all: jemalloc_purge_all,
});

/// Outside the lib's tests the binary owns the allocator: nothing built in.
#[cfg(not(test))]
const BUILT_IN: Option<AllocatorHooks> = None;

/// The allocator's statistics, when the binary's hooks (or the lib's test
/// build) can read them.
pub fn stats() -> Option<AllocStats> {
    hooks().and_then(|h| (h.stats)())
}

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

/// `LAMBDA_VM_TREE_LEVEL_PURGE`: a purge after each level of the block tree.
/// `auto` (unset or empty, the default) purges after a level only when the
/// block's memory is short ([`note_memory_pressure`]) and the process's
/// resident set has reached [`LEVEL_PURGE_SHARE`] of the host target, so a
/// host with room never purges; `off` never purges there; `always` purges after
/// every level (a measurement arm). A tree level frees what it allocated —
/// its programs, its proofs' working sets — into pages the next level rarely
/// reuses: at the p90 block on a 74 GiB emulated host the recursion's VmRSS
/// peaked 12.9–30 GiB over its live heap (BIG 662, decay 0 against the
/// posture, I-MEMFIT §1.7b).
pub const LEVEL_PURGE_ENV: &str = "LAMBDA_VM_TREE_LEVEL_PURGE";

/// `auto` purges after a level once VmRSS reaches this share of the host
/// target. The p90 block's recursion reaches ≈ 94 GiB against a 110.7 GiB
/// target on a 128 GiB host (85 %), so the purges stay off there.
pub const LEVEL_PURGE_SHARE: f64 = 0.9;

/// What [`LEVEL_PURGE_ENV`] says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelPurge {
    Auto,
    Off,
    Always,
}

/// A value of [`LEVEL_PURGE_ENV`]; nonsense is an error.
pub fn parse_level_purge(value: Option<&str>) -> Result<LevelPurge, String> {
    match value.map(str::trim) {
        None | Some("" | "auto") => Ok(LevelPurge::Auto),
        Some("off") => Ok(LevelPurge::Off),
        Some("always") => Ok(LevelPurge::Always),
        Some(v) => Err(format!(
            "{LEVEL_PURGE_ENV} must be auto, off or always, got `{v}`"
        )),
    }
}

/// Whether to purge after a level: the setting, the block's memory pressure,
/// and the resident set against the target (no reading: no purge under
/// `auto`). `Err` says why `auto` skipped it.
fn level_decision(
    setting: LevelPurge,
    pressure: bool,
    resident: Option<u64>,
    target: u64,
) -> Result<(), &'static str> {
    match setting {
        LevelPurge::Off => Err("off"),
        LevelPurge::Always => Ok(()),
        LevelPurge::Auto if !pressure => Err("auto, no memory pressure"),
        LevelPurge::Auto => match resident {
            Some(r) if r as f64 >= LEVEL_PURGE_SHARE * target as f64 => Ok(()),
            Some(_) => Err("auto, the host has room"),
            None => Err("auto, no resident reading"),
        },
    }
}

/// After the block tree's level `level` (0 = the leaves): purge every arena
/// when [`LEVEL_PURGE_ENV`] says so against the host `target`, and print one
/// `ALLOC PURGE level-N` line through `say` (what it did, or why `auto`
/// skipped it; nothing for `off`). Returns the purge when it ran.
pub fn purge_after_level(level: usize, target: u64, say: &dyn Fn(&str)) -> Option<Purge> {
    let setting = match parse_level_purge(std::env::var(LEVEL_PURGE_ENV).ok().as_deref()) {
        Ok(s) => s,
        Err(why) => {
            say(&format!("ALLOC PURGE level-{level}: {why}; not purged"));
            return None;
        }
    };
    let pressure = PRESSURE.load(std::sync::atomic::Ordering::Relaxed);
    let resident = crate::lfm::program_budget::resident_bytes();
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    if let Err(why) = level_decision(setting, pressure, resident, target) {
        if setting != LevelPurge::Off {
            say(&format!(
                "ALLOC PURGE level-{level}: skipped ({why}; VmRSS {} of the target {:.2} GiB)",
                resident.map_or("unread".to_string(), |r| format!("{:.2}", gib(r))),
                gib(target)
            ));
        }
        return None;
    }
    let purge = purge_all_arenas()?;
    say(&format!(
        "ALLOC PURGE level-{level}: {:.3}s · jemalloc resident {:.2} → {:.2} GiB",
        purge.secs,
        gib(purge.resident_before as u64),
        gib(purge.resident_after as u64)
    ));
    Some(purge)
}

/// `LAMBDA_VM_ALLOC_WATERMARK`: a monitor beside the whole block that purges
/// every arena when the block's memory is short ([`note_memory_pressure`]),
/// the process's resident set has reached [`WATERMARK_SHARE`] of the host
/// target, and the allocator holds at least [`WATERMARK_MIN_FREED`] of freed
/// pages (resident − allocated), at most once per [`WATERMARK_SPACING`].
/// `auto` (unset or empty, the default) runs it; `off` does not. It reaches the
/// pages a phase frees and does not reuse *inside* the phase, which no
/// boundary point can: at the p90 block on a 48 GiB emulated host the binding
/// peaks were phase A's end and level 1's end, both retention made after the
/// last purge point (BIG 683, I-MEMFIT §5.1–§6). A host with room never
/// reaches the share, so it never fires there. Each fire prints one
/// `ALLOC WATERMARK fire` line; [`Watermark::finish`] prints the count.
pub const WATERMARK_ENV: &str = "LAMBDA_VM_ALLOC_WATERMARK";

/// The watermark: this share of the host target (the level purges' share,
/// [`LEVEL_PURGE_SHARE`]).
pub const WATERMARK_SHARE: f64 = LEVEL_PURGE_SHARE;

/// The least freed-and-held bytes worth a purge.
pub const WATERMARK_MIN_FREED: u64 = 4 << 30;

/// How often the monitor reads the host.
const WATERMARK_TICK: std::time::Duration = std::time::Duration::from_millis(500);

/// The least time between two fires.
pub const WATERMARK_SPACING: std::time::Duration = std::time::Duration::from_secs(5);

/// The highest-VmRSS reading the watermark monitor took since the window was
/// last read ([`peak_window`]): VmRSS, jemalloc's allocated and resident
/// bytes, and the seconds since the monitor started.
#[derive(Clone, Copy, Debug, Default)]
struct PeakSample {
    rss: u64,
    allocated: u64,
    resident: u64,
    at: f64,
}

static PEAK: std::sync::Mutex<Option<PeakSample>> = std::sync::Mutex::new(None);

/// The window's highest-VmRSS reading in words, and a new window; `None`
/// without a running monitor (or before its first reading). A phase's
/// `ALLOC PEAK` line: its live heap (allocated) against what the allocator holds
/// (resident) at its highest VmRSS, read every [`WATERMARK_TICK`].
pub fn peak_window() -> Option<String> {
    let sample = PEAK.lock().unwrap_or_else(|e| e.into_inner()).take()?;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    Some(format!(
        "VmRSS {:.2} GiB at {:.1}s · jemalloc allocated {:.2} · resident {:.2} (resident − allocated {:.2})",
        gib(sample.rss),
        sample.at,
        gib(sample.allocated),
        gib(sample.resident),
        gib(sample.resident.saturating_sub(sample.allocated))
    ))
}

/// What [`WATERMARK_ENV`] says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WatermarkSetting {
    Auto,
    Off,
}

/// A value of [`WATERMARK_ENV`]; nonsense is an error.
pub fn parse_watermark(value: Option<&str>) -> Result<WatermarkSetting, String> {
    match value.map(str::trim) {
        None | Some("" | "auto") => Ok(WatermarkSetting::Auto),
        Some("off") => Ok(WatermarkSetting::Off),
        Some(v) => Err(format!("{WATERMARK_ENV} must be auto or off, got `{v}`")),
    }
}

/// Whether the watermark fires now: memory pressure, the resident set at or
/// over the share of `target`, enough freed pages held, and the spacing since
/// the last fire. No reading: no fire.
fn watermark_fires(
    pressure: bool,
    resident: Option<u64>,
    freed: Option<u64>,
    target: u64,
    since_last: Option<std::time::Duration>,
) -> bool {
    pressure
        && resident.is_some_and(|r| r as f64 >= WATERMARK_SHARE * target as f64)
        && freed.is_some_and(|f| f >= WATERMARK_MIN_FREED)
        && since_last.is_none_or(|s| s >= WATERMARK_SPACING)
}

/// The running watermark monitor ([`start_watermark`]); stopped and joined by
/// [`Watermark::finish`] or when dropped.
pub struct Watermark {
    stop: std::sync::Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    handle: Option<std::thread::JoinHandle<(usize, f64)>>,
    target: u64,
    purges: bool,
}

/// Starts the watermark monitor against the host `target` (bytes), unless the
/// build cannot read the allocator. Under [`WATERMARK_ENV`] `off` (or
/// nonsense, which it prints) it only samples, for the `ALLOC PEAK` lines
/// ([`peak_window`]), and never purges.
pub fn start_watermark(target: u64) -> Option<Watermark> {
    let purges = match parse_watermark(std::env::var(WATERMARK_ENV).ok().as_deref()) {
        Ok(WatermarkSetting::Auto) => true,
        Ok(WatermarkSetting::Off) => false,
        Err(why) => {
            eprintln!("ALLOC WATERMARK: {why}; sampling only");
            false
        }
    };
    let hooks = hooks()?;
    *PEAK.lock().unwrap_or_else(|e| e.into_inner()) = None;
    let stop = std::sync::Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let flag = stop.clone();
    let handle = std::thread::Builder::new()
        .name("alloc-watermark".to_string())
        .spawn(move || watch(hooks, target, purges, &flag))
        .ok()?;
    Some(Watermark {
        stop,
        handle: Some(handle),
        target,
        purges,
    })
}

/// The monitor's loop: a reading every [`WATERMARK_TICK`] until stopped.
/// Returns the fires and their summed seconds.
fn watch(
    hooks: AllocatorHooks,
    target: u64,
    purges: bool,
    stop: &(std::sync::Mutex<bool>, std::sync::Condvar),
) -> (usize, f64) {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let t0 = std::time::Instant::now();
    let (mut fires, mut secs, mut last) = (0usize, 0.0f64, None::<std::time::Instant>);
    let mut stopped = stop.0.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        stopped = stop
            .1
            .wait_timeout(stopped, WATERMARK_TICK)
            .unwrap_or_else(|e| e.into_inner())
            .0;
        if *stopped {
            return (fires, secs);
        }
        let resident = crate::lfm::program_budget::resident_bytes();
        let stats = (hooks.stats)();
        if let (Some(rss), Some(st)) = (resident, stats) {
            let mut peak = PEAK.lock().unwrap_or_else(|e| e.into_inner());
            if peak.is_none_or(|p| rss > p.rss) {
                *peak = Some(PeakSample {
                    rss,
                    allocated: st.allocated as u64,
                    resident: st.resident as u64,
                    at: t0.elapsed().as_secs_f64(),
                });
            }
        }
        let pressure = PRESSURE.load(std::sync::atomic::Ordering::Relaxed);
        if !purges || !pressure {
            continue;
        }
        let freed = stats.map(|s| s.resident.saturating_sub(s.allocated) as u64);
        if !watermark_fires(pressure, resident, freed, target, last.map(|l| l.elapsed())) {
            continue;
        }
        let Some(purge) = purge_with(hooks) else {
            continue;
        };
        let after = crate::lfm::program_budget::resident_bytes();
        fires += 1;
        secs += purge.secs;
        last = Some(std::time::Instant::now());
        eprintln!(
            "ALLOC WATERMARK fire #{fires} at {:.1}s: VmRSS {:.2} → {} GiB · jemalloc resident {:.2} \
             → {:.2} GiB · {:.3}s",
            t0.elapsed().as_secs_f64(),
            gib(resident.unwrap_or(0)),
            after.map_or("unread".to_string(), |a| format!("{:.2}", gib(a))),
            gib(purge.resident_before as u64),
            gib(purge.resident_after as u64),
            purge.secs
        );
    }
}

impl Watermark {
    fn stop(&mut self) -> Option<(usize, f64)> {
        *self.stop.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.stop.1.notify_all();
        self.handle.take().and_then(|h| h.join().ok())
    }

    /// Stops the monitor and says what it did: the `ALLOC WATERMARK end` line.
    pub fn finish(mut self) -> String {
        let (fires, secs) = self.stop().unwrap_or((0, 0.0));
        if !self.purges {
            return "ALLOC WATERMARK end: off (sampling only), 0 fire(s)".to_string();
        }
        format!(
            "ALLOC WATERMARK end: {fires} fire(s), Σ {secs:.2}s (auto: memory pressure, VmRSS ≥ \
             {WATERMARK_SHARE:.2} × {:.2} GiB, ≥ {} GiB freed, ≥ {}s apart)",
            self.target as f64 / (1u64 << 30) as f64,
            WATERMARK_MIN_FREED >> 30,
            WATERMARK_SPACING.as_secs()
        )
    }
}

impl Drop for Watermark {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Purge every arena through the hooks, timed, with the resident bytes
/// before and after.
fn purge_all_arenas() -> Option<Purge> {
    purge_with(hooks()?)
}

fn purge_with(hooks: AllocatorHooks) -> Option<Purge> {
    let resident = || (hooks.stats)().map(|s| s.resident);
    let resident_before = resident()?;
    let t = std::time::Instant::now();
    if !(hooks.purge_all)() {
        return None;
    }
    let secs = t.elapsed().as_secs_f64();
    Some(Purge {
        secs,
        resident_before,
        resident_after: resident()?,
    })
}

/// jemalloc's statistics, the epoch turned first (they are cached until it
/// turns).
#[cfg(test)]
fn jemalloc_stats() -> Option<AllocStats> {
    use tikv_jemalloc_ctl::{epoch, stats};
    epoch::advance().ok()?;
    Some(AllocStats {
        allocated: stats::allocated::read().ok()?,
        active: stats::active::read().ok()?,
        resident: stats::resident::read().ok()?,
        mapped: stats::mapped::read().ok()?,
        retained: stats::retained::read().ok()?,
    })
}

/// `arena.<MALLCTL_ARENAS_ALL>.purge` (4096, jemalloc's "every arena" index): a
/// control that neither reads nor writes, so every pointer is null.
#[cfg(test)]
fn jemalloc_purge_all() -> bool {
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
    if rc != 0 {
        eprintln!("ALLOC PURGE: arena.4096.purge returned {rc}");
    }
    rc == 0
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
    fn the_level_purge_knob_reads_auto_off_or_always() {
        assert_eq!(parse_level_purge(None), Ok(LevelPurge::Auto));
        assert_eq!(parse_level_purge(Some("")), Ok(LevelPurge::Auto));
        assert_eq!(parse_level_purge(Some("off")), Ok(LevelPurge::Off));
        assert_eq!(parse_level_purge(Some(" always ")), Ok(LevelPurge::Always));
        assert!(parse_level_purge(Some("yes")).is_err());
    }

    /// `auto` purges after a level only under memory pressure and with the
    /// host near its target; a host with room never purges.
    #[test]
    fn a_level_purge_needs_pressure_and_a_full_host() {
        const G: u64 = 1 << 30;
        let target = 100 * G;
        use LevelPurge::*;
        assert_eq!(level_decision(Auto, true, Some(95 * G), target), Ok(()));
        assert_eq!(level_decision(Auto, true, Some(90 * G), target), Ok(()));
        assert_eq!(
            level_decision(Auto, true, Some(89 * G), target),
            Err("auto, the host has room")
        );
        assert_eq!(
            level_decision(Auto, false, Some(99 * G), target),
            Err("auto, no memory pressure")
        );
        assert_eq!(
            level_decision(Auto, true, None, target),
            Err("auto, no resident reading")
        );
        assert_eq!(level_decision(Off, true, Some(99 * G), target), Err("off"));
        assert_eq!(level_decision(Always, false, Some(G), target), Ok(()));
    }

    #[test]
    fn the_watermark_knob_reads_auto_or_off() {
        assert_eq!(parse_watermark(None), Ok(WatermarkSetting::Auto));
        assert_eq!(parse_watermark(Some(" auto ")), Ok(WatermarkSetting::Auto));
        assert_eq!(parse_watermark(Some("off")), Ok(WatermarkSetting::Off));
        assert!(parse_watermark(Some("on")).is_err());
    }

    /// The watermark fires only under memory pressure, with the host at the
    /// share of its target, enough freed pages held and the spacing kept.
    #[test]
    fn the_watermark_fires_only_under_pressure_on_a_full_host() {
        const G: u64 = 1 << 30;
        let target = 100 * G;
        let s = std::time::Duration::from_secs;
        assert!(watermark_fires(
            true,
            Some(90 * G),
            Some(4 * G),
            target,
            None
        ));
        assert!(watermark_fires(
            true,
            Some(99 * G),
            Some(9 * G),
            target,
            Some(s(5))
        ));
        assert!(
            !watermark_fires(false, Some(99 * G), Some(9 * G), target, None),
            "no pressure"
        );
        assert!(
            !watermark_fires(true, Some(89 * G), Some(9 * G), target, None),
            "the host has room"
        );
        assert!(
            !watermark_fires(true, Some(99 * G), Some(3 * G), target, None),
            "too little freed"
        );
        assert!(
            !watermark_fires(true, Some(99 * G), Some(9 * G), target, Some(s(4))),
            "too soon"
        );
        assert!(
            !watermark_fires(true, None, Some(9 * G), target, None),
            "no reading"
        );
        assert!(
            !watermark_fires(true, Some(99 * G), None, target, None),
            "no stats"
        );
    }

    /// A started monitor samples (a peak window to read), stops, and reports
    /// its fires.
    #[test]
    fn the_watermark_monitor_starts_and_stops() {
        if std::env::var_os(WATERMARK_ENV).is_some() {
            return;
        }
        let w = start_watermark(u64::MAX).expect("the test build reads its jemalloc");
        std::thread::sleep(WATERMARK_TICK * 3);
        // The window needs a VmRSS reading (Linux's /proc); elsewhere it stays empty.
        if crate::lfm::program_budget::resident_bytes().is_some() {
            let peak = peak_window().expect("a reading after three ticks");
            assert!(peak.contains("jemalloc allocated"), "{peak}");
        }
        let line = w.finish();
        assert!(line.starts_with("ALLOC WATERMARK end: 0 fire(s)"), "{line}");
    }

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

    /// A purge goes through the hooks it is given: it reads the resident bytes
    /// before and after, and a purge that did not run is no purge.
    #[test]
    fn a_purge_runs_through_its_hooks() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static RESIDENT: AtomicUsize = AtomicUsize::new(900);
        fn stats() -> Option<AllocStats> {
            Some(AllocStats {
                resident: RESIDENT.load(Ordering::SeqCst),
                ..AllocStats::default()
            })
        }
        fn purged() -> bool {
            RESIDENT.store(100, Ordering::SeqCst);
            true
        }
        fn refused() -> bool {
            false
        }
        fn blind() -> Option<AllocStats> {
            None
        }
        let purge = purge_with(AllocatorHooks {
            stats,
            purge_all: purged,
        })
        .expect("the hooks purge");
        assert_eq!((purge.resident_before, purge.resident_after), (900, 100));
        let refused = AllocatorHooks {
            stats,
            purge_all: refused,
        };
        assert!(purge_with(refused).is_none());
        let blind = AllocatorHooks {
            stats: blind,
            purge_all: purged,
        };
        assert!(purge_with(blind).is_none(), "no reading, no purge line");
    }

    /// The lib's test build reads its own jemalloc with no hook installed.
    #[test]
    fn the_test_build_reads_its_jemalloc() {
        assert!(stats().is_some_and(|s| s.allocated > 0 && s.resident >= s.allocated));
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
