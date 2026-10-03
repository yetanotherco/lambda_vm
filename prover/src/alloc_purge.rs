//! Returning the allocator's freed pages to the OS at a phase boundary
//! (`LAMBDA_VM_ALLOC_PURGE`, off by default).
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

/// What one purge did.
#[derive(Clone, Copy, Debug)]
pub struct Purge {
    /// The purge's own wall time.
    pub secs: f64,
    /// jemalloc's resident bytes before and after (live, dirty and metadata).
    pub resident_before: usize,
    pub resident_after: usize,
}

/// `LAMBDA_VM_ALLOC_PURGE`: `all`, or a list of the boundary names
/// [`purge_point`] is called with, separated by commas or dots (`base.tree`,
/// for job specs that split their knobs on commas). Unset or empty purges
/// nowhere. The block's points: `phase-a` (phase A done, `block.rs`), `base`
/// (the base proved) and `tree` (the top proved, before the block verifier),
/// the last two in the whole-block harness.
pub const ALLOC_PURGE_ENV: &str = "LAMBDA_VM_ALLOC_PURGE";

/// Whether `point` is one of the boundaries `setting` names.
fn names_point(setting: &str, point: &str) -> bool {
    setting
        .split([',', '.'])
        .map(str::trim)
        .any(|p| p == "all" || p == point)
}

/// At the boundary `point`, purge every jemalloc arena's dirty pages when
/// [`ALLOC_PURGE_ENV`] names it, and print one `ALLOC PURGE` line (the wall
/// time and jemalloc's resident bytes before and after). Returns what it did,
/// or `None` when the point is not named or the build cannot purge.
pub fn purge_point(point: &str) -> Option<Purge> {
    let setting = std::env::var(ALLOC_PURGE_ENV).ok()?;
    if !names_point(&setting, point) {
        return None;
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

/// `LAMBDA_VM_ALLOC_DIRTY_LOG=<secs>`: while the returned guard lives, a thread
/// prints one `ALLOC DIRTY` line every `secs` seconds. Each line carries:
/// - jemalloc's dirty bytes (freed pages it keeps) in the shared oversize arena
///   and in the per-thread arenas;
/// - the thread arenas' cumulative large allocations, in bytes by allocation
///   size: < 1 MiB, 1–8 MiB, ≥ 8 MiB.
///
/// Freed extents coalesce, so a dirty extent's size says nothing about the
/// allocations behind it; the allocation counters do. Diffed across a phase,
/// they say whether a lower `oversize_threshold` (for 1–8 MiB allocations)
/// could route that phase's churn to the shared arena, where freed pages are
/// reused across threads, or only fewer arenas could. BIG 588: the tree phase
/// makes ≈ 19 GiB of retention for itself. Unset, or outside the lib's test
/// builds, nothing runs.
pub fn dirty_log_from_env() -> Option<DirtyLog> {
    let secs: f64 = std::env::var("LAMBDA_VM_ALLOC_DIRTY_LOG")
        .ok()
        .filter(|v| !v.is_empty())?
        .parse()
        .ok()
        .filter(|s: &f64| s.is_finite() && *s > 0.0)?;
    DirtyLog::start(secs)
}

/// The running `ALLOC DIRTY` logger; stops when dropped.
pub struct DirtyLog {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl DirtyLog {
    #[cfg(test)]
    fn start(secs: f64) -> Option<Self> {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = stop.clone();
        let t0 = std::time::Instant::now();
        let thread = std::thread::Builder::new()
            .name("alloc-dirty-log".into())
            .spawn(move || {
                let period = std::time::Duration::from_secs_f64(secs);
                while !flag.load(std::sync::atomic::Ordering::SeqCst) {
                    if let Some(r) = arena_report() {
                        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
                        eprintln!(
                            "ALLOC DIRTY t={:.1} · dirty shared {:.2} · thread {:.2} GiB over {} arenas · \
                             thread large allocations (cumulative) < 1 MiB {:.2} · 1–8 MiB {:.2} · ≥ 8 MiB \
                             {:.2} GiB",
                            t0.elapsed().as_secs_f64(),
                            gib(r.shared_dirty),
                            gib(r.thread_dirty),
                            r.thread_arenas,
                            gib(r.thread_large[0]),
                            gib(r.thread_large[1]),
                            gib(r.thread_large[2]),
                        );
                    }
                    std::thread::sleep(period);
                }
            })
            .ok()?;
        Some(Self {
            stop,
            thread: Some(thread),
        })
    }

    #[cfg(not(test))]
    fn start(_secs: f64) -> Option<Self> {
        None
    }
}

impl Drop for DirtyLog {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// One reading of where jemalloc's pages sit (see [`dirty_log_from_env`]).
#[cfg_attr(not(test), allow(dead_code))]
struct ArenaReport {
    shared_dirty: u64,
    thread_dirty: u64,
    /// Thread arenas that hold dirty pages or ever made a large allocation.
    thread_arenas: usize,
    /// The thread arenas' cumulative large allocations, bytes by size.
    thread_large: [u64; 3],
}

/// Reads every arena's dirty pages (`stats.arenas.<i>.pdirty` × `arenas.page`)
/// and large allocations (`stats.arenas.<i>.lextents.<j>.nmalloc` ×
/// `arenas.lextent.<j>.size`). The shared oversize arena is the one after the
/// automatic arenas (`opt.narenas`: jemalloc 5.3's `arena_init_huge` takes the
/// index `narenas_total` right after setting it to `narenas_auto`).
#[cfg(test)]
fn arena_report() -> Option<ArenaReport> {
    use tikv_jemalloc_ctl::{epoch, raw};
    epoch::advance().ok()?;
    let auto: u32 = unsafe { raw::read(b"opt.narenas\0") }.ok()?;
    let total: u32 = unsafe { raw::read(b"arenas.narenas\0") }.ok()?;
    let page: usize = unsafe { raw::read(b"arenas.page\0") }.ok()?;
    let nlextents: u32 = unsafe { raw::read(b"arenas.nlextents\0") }.ok()?;
    let mut size_mib = [0usize; 4];
    raw::name_to_mib(b"arenas.lextent.0.size\0", &mut size_mib).ok()?;
    let sizes: Vec<usize> = (0..nlextents as usize)
        .map(|j| {
            size_mib[2] = j;
            unsafe { raw::read_mib::<usize>(&size_mib) }.unwrap_or(0)
        })
        .collect();
    let mut pdirty_mib = [0usize; 4];
    raw::name_to_mib(b"stats.arenas.0.pdirty\0", &mut pdirty_mib).ok()?;
    let mut nmalloc_mib = [0usize; 6];
    raw::name_to_mib(b"stats.arenas.0.lextents.0.nmalloc\0", &mut nmalloc_mib).ok()?;
    let mut r = ArenaReport {
        shared_dirty: 0,
        thread_dirty: 0,
        thread_arenas: 0,
        thread_large: [0; 3],
    };
    for i in 0..total as usize {
        pdirty_mib[2] = i;
        // An arena index that was never initialised has no stats.
        let Ok(pdirty) = (unsafe { raw::read_mib::<usize>(&pdirty_mib) }) else {
            continue;
        };
        let dirty = (pdirty * page) as u64;
        if i == auto as usize {
            r.shared_dirty += dirty;
            continue;
        }
        r.thread_dirty += dirty;
        nmalloc_mib[2] = i;
        let mut large = 0u64;
        for (j, &size) in sizes.iter().enumerate() {
            nmalloc_mib[4] = j;
            let n: u64 = unsafe { raw::read_mib(&nmalloc_mib) }.unwrap_or(0);
            if n == 0 {
                continue;
            }
            let bytes = n * size as u64;
            large += bytes;
            let bucket = if size < 1 << 20 {
                0
            } else if size < 8 << 20 {
                1
            } else {
                2
            };
            r.thread_large[bucket] += bytes;
        }
        if dirty > 0 || large > 0 {
            r.thread_arenas += 1;
        }
    }
    Some(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The arena reader counts this thread's 512 allocations of 2 MiB as 1–8
    /// MiB large allocations in a thread arena, and, under the posture, their
    /// pages as the thread arenas' dirty bytes once freed.
    #[test]
    #[ignore = "run under the posture: _RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1"]
    fn the_arena_reader_sees_freed_buffers() {
        let before = arena_report().expect("the stats read");
        let buffers: Vec<Vec<u8>> = (0..512).map(|_| vec![1u8; 2 << 20]).collect();
        drop(buffers);
        let after = arena_report().expect("the stats read");
        let allocated = after.thread_large[1].saturating_sub(before.thread_large[1]);
        let dirty = after.thread_dirty.saturating_sub(before.thread_dirty);
        println!(
            "ARENA UNIT: 1–8 MiB large allocations +{} MiB · thread dirty +{} MiB · shared dirty {} MiB · {} arenas",
            allocated >> 20,
            dirty >> 20,
            after.shared_dirty >> 20,
            after.thread_arenas
        );
        assert!(
            allocated >= 1 << 30,
            "512 allocations of 2 MiB not counted as 1–8 MiB"
        );
        assert!(
            dirty >= 512 << 20,
            "their freed pages not counted as thread-arena dirty bytes"
        );
    }

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
