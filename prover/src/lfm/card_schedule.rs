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
//!   driver's reconstruct, emit and harvest) run on a pool of their own at that
//!   nice value, so the proof holding the card keeps the CPU and the global
//!   rayon pool.
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

/// Run a host-only phase: on the host-phase pool when [`host_nice`] is set,
/// inline otherwise.
///
/// ⚠ Only for phases that never reach the card. A device phase run here would
/// drive the card from a low-priority thread, which is the opposite of the
/// knob's point, and nothing in the pool would stop it.
pub fn host_phase<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    #[cfg(feature = "parallel")]
    {
        host_phase_in(host_pool(), f)
    }
    #[cfg(not(feature = "parallel"))]
    {
        f()
    }
}

/// [`host_phase`] on an explicit pool, so a test can run one without the
/// process-wide knob.
#[cfg(feature = "parallel")]
pub(crate) fn host_phase_in<R: Send>(
    pool: Option<&rayon::ThreadPool>,
    f: impl FnOnce() -> R + Send,
) -> R {
    match pool {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

/// The host-phase pool, built once on first use when [`host_nice`] is set.
#[cfg(feature = "parallel")]
fn host_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| host_nice().map(build_host_pool))
        .as_ref()
}

/// A pool as wide as the global one whose threads run at `nice`.
///
/// ★ As wide, not narrower: the host phases lose nothing while the card
/// holder is idle, because a nice value is a WEIGHT, not a cap. Under the
/// kernel's fair scheduler a nice-10 thread gets about a tenth of a core it
/// shares with a nice-0 one, and all of a core nobody else wants.
#[cfg(feature = "parallel")]
pub(crate) fn build_host_pool(nice: i32) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(rayon::current_num_threads())
        .thread_name(|i| format!("lfm-host-{i}"))
        .start_handler(move |_| lower_this_thread(nice))
        .build()
        .expect("the host-phase pool must build")
}

/// Pool threads that took the nice value, and those that could not.
static LOWERED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static NOT_LOWERED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Set the CALLING thread's nice value. Raising one's own nice value needs no
/// privilege; a refusal is counted, not fatal, and the banner reports it.
#[cfg(target_os = "linux")]
fn lower_this_thread(nice: i32) {
    // SAFETY: two syscalls on the calling thread's own id; neither touches
    // memory the caller owns.
    let lowered = unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        libc::setpriority(libc::PRIO_PROCESS, tid, nice) == 0
    };
    let counter = if lowered { &LOWERED } else { &NOT_LOWERED };
    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Per-thread nice values are a Linux interface; elsewhere the pool still
/// separates the host phases from the global pool, at the default priority.
#[cfg(not(target_os = "linux"))]
fn lower_this_thread(_nice: i32) {
    NOT_LOWERED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The line a driver prints once, so an arm's log names its schedule. `None`
/// when every knob is off, so a default run prints exactly what it always did.
pub fn banner() -> Option<String> {
    if !device_scope() && !ahead() && host_nice().is_none() {
        return None;
    }
    #[cfg(feature = "parallel")]
    let pool = host_pool().map(|p| p.current_num_threads());
    #[cfg(not(feature = "parallel"))]
    let pool: Option<usize> = None;
    let nice = match (host_nice(), pool) {
        (Some(n), Some(threads)) => format!(
            "host phases on a {threads}-thread pool at nice {n} ({} lowered, {} not)",
            LOWERED.load(std::sync::atomic::Ordering::Relaxed),
            NOT_LOWERED.load(std::sync::atomic::Ordering::Relaxed),
        ),
        (Some(n), None) => format!("nice {n} set but no pool in this build"),
        (None, _) => "host phases where they are called".to_string(),
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
            name.as_deref().is_some_and(|n| n.starts_with("lfm-host-")),
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
                .all(|n| n.as_deref().is_some_and(|n| n.starts_with("lfm-host-"))),
            "an item ran off the host-phase pool: {names:?}"
        );
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
}
