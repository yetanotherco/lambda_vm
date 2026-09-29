//! How a level's proofs share the card and the host: three knobs, all OFF by
//! default, none of which changes a byte any proof commits to.
//!
//! # The term they attack
//!
//! At level 0 of the STARK tree six wraps prove at once under ONE card permit
//! ([`super::device_permit`]), and the permit is held 90% of the level. Inside
//! those holds the card is idle 42% of the time — the `build_artifacts` holds
//! 64%, the `multi_prove` holds 30% — against 10% and 13% for the same two
//! holds in the interior, whose programs are LARGER. What the level-0 holder
//! waits for is the host: its own host work inside the hold, and the CPU the
//! other five workers' host phases (the epoch reconstruct above all) take from
//! it.
//!
//! # The knobs
//!
//! - `LAMBDA_VM_GAP_PREP_SCOPE=1` — [`device_scope`]: an artifact build holds
//!   the card only around its DEVICE commits. The groups below the device
//!   floor are committed on the host first, outside the hold.
//! - `LAMBDA_VM_GAP_PREP_NICE=<1..19>` — [`host_nice`]: the host-only phases of
//!   a recursion proof (execute, fill, the build's host pass, and the tree
//!   driver's reconstruct, emit and harvest) run at that nice value on a pool
//!   of the CALLING thread's own ([`host_phase`]), so the proof holding the card
//!   keeps the CPU and the global rayon pool.
//! - `LAMBDA_VM_GAP_PREP_AHEAD=1` — [`ahead`]: a level-0 wrap builds its
//!   artifacts on a helper thread while it executes and fills, instead of
//!   before.
//!
//! # What none of them changes
//!
//! ⛔ The permit stays MUTUAL EXCLUSION: every knob moves work off the card or
//! off the holder's CPU, none lets two proofs onto the card at once. A build's
//! device commits run in the windows they always ran in, so its device working
//! set is unchanged. Roots, traces and program ids are pure functions of the
//! program and options; a pool, a priority or a thread only moves where they
//! are computed.

use std::sync::OnceLock;
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize, Ordering};

/// `LAMBDA_VM_GAP_PREP_SCOPE=1`: the artifact build holds the card only around
/// its device commits. Unset, empty or `0` = the hold spans the whole build.
pub fn device_scope() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| flag("LAMBDA_VM_GAP_PREP_SCOPE"))
}

/// `LAMBDA_VM_GAP_PREP_AHEAD=1`: a level-0 wrap builds its artifacts beside its
/// execute and fill. Unset, empty or `0` = build, then execute and fill.
pub fn ahead() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| flag("LAMBDA_VM_GAP_PREP_AHEAD"))
}

/// `LAMBDA_VM_GAP_PREP_NICE=<1..19>`: the nice value the host-phase pool runs
/// at. `None` (unset, empty or `0`) = no pool, host phases run where they are
/// called.
pub fn host_nice() -> Option<i32> {
    static NICE: OnceLock<Option<i32>> = OnceLock::new();
    *NICE.get_or_init(|| parse_nice(std::env::var("LAMBDA_VM_GAP_PREP_NICE").ok().as_deref()))
}

/// An on/off knob: unset, empty or `0` is off, `1` is on, anything else stops
/// the run rather than guessing.
fn flag(name: &str) -> bool {
    match std::env::var(name).ok().as_deref() {
        None | Some("") | Some("0") => false,
        Some("1") => true,
        Some(other) => panic!("{name} must be `0` or `1`, got `{other}`"),
    }
}

fn parse_nice(value: Option<&str>) -> Option<i32> {
    match value {
        None | Some("") | Some("0") => None,
        Some(v) => Some(
            v.parse::<i32>()
                .ok()
                .filter(|n| (1..=19).contains(n))
                .unwrap_or_else(|| {
                    panic!("LAMBDA_VM_GAP_PREP_NICE must be an integer in 1..=19, got `{v}`")
                }),
        ),
    }
}

/// Run a host-only phase: on the CALLING thread's own host-phase pool when
/// [`host_nice`] is set, inline otherwise.
///
/// # ⛔ One pool per calling thread, never one shared by the workers
///
/// A rayon thread blocked in a join runs whatever `find_work` hands it —
/// INJECTED jobs included — and returns to its own work only when that job
/// returns (rayon-core 1.13, `registry.rs` `wait_until_cold`). In one pool
/// shared by six workers every whole phase was an injected job, so a thread
/// waiting inside one wrap's reconstruct ran another wrap's whole phase nested
/// on its stack: in job 202's NICE arms, reconstructs of 2–3 s took up to 22 s,
/// the ones that started first finishing last, and level 0 lost 9.6 s. A
/// worker runs one phase at a time, so its own pool only ever holds that
/// phase's jobs and nothing foreign can nest inside it.
///
/// A caller that is already a rayon worker, of any pool, runs the phase inline:
/// it is inside a phase or a parallel section already.
///
/// ⚠ Only for phases that never reach the card. A device phase run here would
/// drive the card from a low-priority thread, which is the opposite of the
/// knob's point, and nothing in the pool would stop it.
pub fn host_phase<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    host_phase_at(host_nice(), f)
}

/// [`host_phase`] at an explicit nice value, so a test can run one without the
/// process-wide knob.
///
/// ⚠ A thread's pool is built at its first call and kept for the thread's life;
/// a later call at another value reuses it.
pub(crate) fn host_phase_at<R: Send>(nice: Option<i32>, f: impl FnOnce() -> R + Send) -> R {
    #[cfg(feature = "parallel")]
    if let Some(nice) = nice
        && rayon::current_thread_index().is_none()
    {
        return CALLER_POOL.with(|pool| pool.get_or_init(|| caller_pool(nice)).install(f));
    }
    #[cfg(not(feature = "parallel"))]
    let _ = nice;
    f()
}

#[cfg(feature = "parallel")]
thread_local! {
    /// The calling thread's host-phase pool: built at its first host phase,
    /// dropped with the thread (a dropped `ThreadPool` lets its threads exit).
    static CALLER_POOL: std::cell::OnceCell<rayon::ThreadPool> =
        const { std::cell::OnceCell::new() };
}

/// Host-phase pools built so far in this process; the index names a pool's threads.
#[cfg(feature = "parallel")]
static POOLS: AtomicUsize = AtomicUsize::new(0);

/// The pool for the calling thread, announced once with how many of its threads
/// took the nice value — the line an arm's readout checks the knob by.
#[cfg(feature = "parallel")]
fn caller_pool(nice: i32) -> rayon::ThreadPool {
    let k = POOLS.fetch_add(1, Ordering::Relaxed);
    let (pool, lowered) = new_host_pool(k, nice);
    let caller = std::thread::current();
    let width = pool.current_num_threads();
    println!(
        "CARD SCHEDULE HOST POOL #{k} for thread {}: {width} threads at nice {nice}, \
         {lowered} lowered, {} not",
        caller
            .name()
            .map_or_else(|| format!("{:?}", caller.id()), str::to_string),
        width - lowered,
    );
    pool
}

/// [`host_phase`] on an explicit pool, so a test can run one without the
/// process-wide knob.
#[cfg(all(test, feature = "parallel"))]
pub(crate) fn host_phase_in<R: Send>(
    pool: Option<&rayon::ThreadPool>,
    f: impl FnOnce() -> R + Send,
) -> R {
    match pool {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

/// A host-phase pool at `nice`, for a test that wants one of its own.
#[cfg(all(test, feature = "parallel"))]
pub(crate) fn build_host_pool(nice: i32) -> rayon::ThreadPool {
    new_host_pool(POOLS.fetch_add(1, Ordering::Relaxed), nice).0
}

/// A pool as wide as the global one whose threads run at `nice`, returned once
/// every thread has started, with how many of them took the value.
///
/// ★ As wide, not narrower: the host phases lose nothing while the card
/// holder is idle, because a nice value is a WEIGHT, not a cap. Under the
/// kernel's fair scheduler a nice-10 thread gets about a tenth of a core it
/// shares with a nice-0 one, and all of a core nobody else wants.
#[cfg(feature = "parallel")]
fn new_host_pool(k: usize, nice: i32) -> (rayon::ThreadPool, usize) {
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    let width = rayon::current_num_threads();
    let started = Arc::new(AtomicUsize::new(0));
    let lowered = Arc::new(AtomicUsize::new(0));
    let (on_start, on_lower) = (Arc::clone(&started), Arc::clone(&lowered));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(width)
        .thread_name(move |i| format!("lfm-host{k}-{i}"))
        .start_handler(move |_| {
            if lower_this_thread(nice) {
                on_lower.fetch_add(1, Ordering::Relaxed);
            }
            on_start.fetch_add(1, Ordering::Release);
        })
        .build()
        .expect("the host-phase pool must build");
    // Every thread runs the handler as it starts, so the count is exact once all
    // have; the deadline only keeps a thread that never starts from hanging this.
    let deadline = Instant::now() + Duration::from_secs(5);
    while started.load(Ordering::Acquire) < width && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    (pool, lowered.load(Ordering::Relaxed))
}

/// Set the CALLING thread's nice value; `true` when it took. Raising one's own
/// nice value needs no privilege; a refusal is reported, not fatal.
#[cfg(all(feature = "parallel", target_os = "linux"))]
fn lower_this_thread(nice: i32) -> bool {
    // SAFETY: two syscalls on the calling thread's own id; neither touches
    // memory the caller owns.
    unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        libc::setpriority(libc::PRIO_PROCESS, tid, nice) == 0
    }
}

/// Per-thread nice values are a Linux interface; elsewhere the pool still
/// separates the host phases from the global pool, at the default priority.
#[cfg(all(feature = "parallel", not(target_os = "linux")))]
fn lower_this_thread(_nice: i32) -> bool {
    false
}

/// The line a driver prints once, so an arm's log names its schedule. `None`
/// when every knob is off, so a default run prints exactly what it always did.
pub fn banner() -> Option<String> {
    if !device_scope() && !ahead() && host_nice().is_none() {
        return None;
    }
    let nice = match host_nice() {
        Some(n) if cfg!(feature = "parallel") => format!(
            "host phases at nice {n} on one pool per worker thread (each pool prints a HOST POOL line)"
        ),
        Some(n) => format!("nice {n} set but no pool in this build"),
        None => "host phases where they are called".to_string(),
    };
    Some(format!(
        "CARD SCHEDULE (F-PREP): artifact hold {} · {} · level-0 artifacts {} \
         (LAMBDA_VM_GAP_PREP_SCOPE / _NICE / _AHEAD; all unset = the record schedule)",
        if device_scope() {
            "around the device commits only"
        } else {
            "around the whole build"
        },
        nice,
        if ahead() {
            "built beside execute + fill"
        } else {
            "built before execute + fill"
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_nice_knob_reads_off_for_unset_empty_and_zero() {
        assert_eq!(parse_nice(None), None);
        assert_eq!(parse_nice(Some("")), None);
        assert_eq!(parse_nice(Some("0")), None);
        assert_eq!(parse_nice(Some("10")), Some(10));
        assert_eq!(parse_nice(Some("19")), Some(19));
    }

    #[test]
    #[should_panic(expected = "must be an integer in 1..=19")]
    fn a_nice_value_outside_the_range_stops_the_run() {
        let _ = parse_nice(Some("20"));
    }

    #[test]
    #[should_panic(expected = "must be an integer in 1..=19")]
    fn a_negative_nice_value_stops_the_run() {
        // A negative value would ask for MORE priority than the card holder,
        // which is the opposite of the knob, and needs a privilege besides.
        let _ = parse_nice(Some("-5"));
    }

    /// ★ The pool runs the phase on ITS thread and hands back the value. Without
    /// the thread check a `host_phase_in` that silently ran inline would pass.
    #[cfg(feature = "parallel")]
    #[test]
    fn a_host_phase_runs_on_the_pool_and_returns_its_value() {
        let pool = build_host_pool(10);
        let (value, name) = host_phase_in(Some(&pool), || {
            (
                (1..=100u64).sum::<u64>(),
                std::thread::current().name().map(str::to_string),
            )
        });
        assert_eq!(value, 5050);
        assert!(
            name.as_deref().is_some_and(|n| n.starts_with("lfm-host")),
            "the phase ran on {name:?}, not on the host-phase pool"
        );
        let inline = host_phase_in(None, || std::thread::current().id());
        assert_eq!(
            inline,
            std::thread::current().id(),
            "no pool = run inline, on the caller"
        );
    }

    /// ★ A parallel iterator inside the phase uses the POOL, not the global
    /// one — the property that frees the global pool for the card holder.
    #[cfg(feature = "parallel")]
    #[test]
    fn a_parallel_iterator_inside_a_host_phase_runs_on_the_pool() {
        use rayon::prelude::*;
        let pool = build_host_pool(10);
        let names: Vec<Option<String>> = host_phase_in(Some(&pool), || {
            (0..64)
                .into_par_iter()
                .map(|_| std::thread::current().name().map(str::to_string))
                .collect()
        });
        assert!(
            names
                .iter()
                .all(|n| n.as_deref().is_some_and(|n| n.starts_with("lfm-host"))),
            "an item ran off the host-phase pool: {names:?}"
        );
    }

    /// Every thread of a pool rayon owns, read by running on each of them.
    #[cfg(feature = "parallel")]
    fn pool_threads() -> std::collections::HashSet<std::thread::ThreadId> {
        rayon::broadcast(|_| std::thread::current().id())
            .into_iter()
            .collect()
    }

    /// ★★★ THE REGRESSION JOB 202 FOUND, ASSERTED: two callers' host phases
    /// never share a thread. A rayon thread waiting in a join runs the jobs of
    /// whoever else is in its pool, so sharing one is how a wrap's reconstruct
    /// ran nested inside another's.
    ///
    /// ⛔ The control is the design it replaces — one pool for both callers —
    /// and it must put both phases on the SAME threads, or this test could not
    /// have caught it.
    #[cfg(feature = "parallel")]
    #[test]
    fn two_callers_never_share_a_host_pool_thread() {
        let (a, b) = std::thread::scope(|s| {
            let a = s.spawn(|| host_phase_at(Some(10), pool_threads));
            let b = s.spawn(|| host_phase_at(Some(10), pool_threads));
            (a.join().expect("caller a"), b.join().expect("caller b"))
        });
        assert_eq!(
            a.len(),
            rayon::current_num_threads(),
            "a pool as wide as the global one"
        );
        assert!(
            a.is_disjoint(&b),
            "two callers' host phases ran on {} shared thread(s)",
            a.intersection(&b).count()
        );
        let shared = build_host_pool(10);
        let (c, d) = std::thread::scope(|s| {
            let c = s.spawn(|| host_phase_in(Some(&shared), pool_threads));
            let d = s.spawn(|| host_phase_in(Some(&shared), pool_threads));
            (c.join().expect("caller c"), d.join().expect("caller d"))
        });
        assert_eq!(
            c, d,
            "the control: one shared pool runs both callers on the same threads"
        );
    }

    /// A thread keeps its pool: its second host phase runs where its first did,
    /// and only its own phases do.
    #[cfg(feature = "parallel")]
    #[test]
    fn a_calling_thread_reuses_its_own_pool() {
        let (first, second, other) = std::thread::scope(|s| {
            let mine = s.spawn(|| {
                (
                    host_phase_at(Some(10), pool_threads),
                    host_phase_at(Some(10), pool_threads),
                )
            });
            let theirs = s.spawn(|| host_phase_at(Some(10), pool_threads));
            let (first, second) = mine.join().expect("the first caller");
            (first, second, theirs.join().expect("the second caller"))
        });
        assert_eq!(
            first, second,
            "a caller's second phase moved to another pool"
        );
        assert!(first.is_disjoint(&other), "two callers shared a pool");
    }

    /// A rayon worker runs a host phase INLINE: it is inside a parallel section
    /// already, and a pool of its own would nest one pool under another.
    #[cfg(feature = "parallel")]
    #[test]
    fn a_rayon_worker_runs_a_host_phase_inline() {
        let pool = build_host_pool(10);
        let (outer, inner) = pool.install(|| {
            (
                std::thread::current().id(),
                host_phase_at(Some(10), || std::thread::current().id()),
            )
        });
        assert_eq!(outer, inner, "a worker's host phase left its thread");
    }

    /// Unset, a host phase runs on the caller, exactly as the code before the knob.
    #[test]
    fn with_the_knob_unset_a_host_phase_runs_on_the_caller() {
        let here = std::thread::current().id();
        assert_eq!(host_phase_at(None, || std::thread::current().id()), here);
    }

    /// ★ The pool's threads carry the nice value — read back from the kernel,
    /// not assumed from the call.
    #[cfg(all(feature = "parallel", target_os = "linux"))]
    #[test]
    fn the_pool_threads_run_at_the_requested_nice_value() {
        let pool = build_host_pool(7);
        let nice = host_phase_in(Some(&pool), || {
            // SAFETY: syscalls on the calling thread's own id. `getpriority`
            // can legitimately return -1, so errno is cleared first.
            unsafe {
                *libc::__errno_location() = 0;
                let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
                libc::getpriority(libc::PRIO_PROCESS, tid)
            }
        });
        assert_eq!(nice, 7, "the pool thread did not take the nice value");
        let here = unsafe {
            let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
            libc::getpriority(libc::PRIO_PROCESS, tid)
        };
        assert_ne!(here, 7, "the caller's own priority must be untouched");
    }

    /// The count a HOST POOL line reports is taken after every thread started:
    /// all of them on Linux, none elsewhere.
    #[cfg(feature = "parallel")]
    #[test]
    fn a_new_pool_reports_every_thread() {
        let (pool, lowered) = new_host_pool(POOLS.fetch_add(1, Ordering::Relaxed), 10);
        let want = if cfg!(target_os = "linux") {
            pool.current_num_threads()
        } else {
            0
        };
        assert_eq!(
            lowered, want,
            "the count was read before every thread started"
        );
    }
}
