//! Where level 1's streamed execution ran: the driver thread's CPU and its
//! last-level cache (the L3 a CPU shares with its neighbours) at each wave,
//! how the waves' hash jobs spread over the L3s, and the driver thread's and
//! the process's page faults and context switches over the execution
//! (`LAMBDA_VM_LEVELS_PLACEMENT=1`). A report: nothing it reads changes what
//! runs. Off by default (and for any word but `1`); read once a process.
//! Linux only; elsewhere every reading is empty.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The knob.
pub const PLACEMENT_ENV: &str = "LAMBDA_VM_LEVELS_PLACEMENT";

/// The most L3s counted; a CPU in a later one is counted in the last.
pub const MAX_L3: usize = 8;

/// Whether [`PLACEMENT_ENV`] is on.
pub fn placement() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        super::wlfm_knobs::knob_word_on(std::env::var(PLACEMENT_ENV).ok().as_deref())
    })
}

/// Whether the executor's hash jobs are being counted now.
static COUNTING: AtomicBool = AtomicBool::new(false);

/// Hash jobs started on each L3 since the last [`take_jobs`].
static JOBS: [AtomicU64; MAX_L3] = [const { AtomicU64::new(0) }; MAX_L3];

/// Turns the executor's job counting on or off.
pub fn count_jobs(on: bool) {
    COUNTING.store(on, Ordering::Relaxed);
}

/// Whether the executor counts its jobs ([`count_jobs`]).
pub fn counting() -> bool {
    COUNTING.load(Ordering::Relaxed)
}

/// One hash job started on this thread: counted against its L3.
pub fn note_job() {
    if let Some(l3) = this_l3() {
        JOBS[l3.min(MAX_L3 - 1)].fetch_add(1, Ordering::Relaxed);
    }
}

/// The jobs counted per L3 since the last call, and zeroes them.
pub fn take_jobs() -> [u64; MAX_L3] {
    std::array::from_fn(|i| JOBS[i].swap(0, Ordering::Relaxed))
}

/// The CPU this thread runs on now, and its L3's index.
pub fn this_cpu() -> Option<(usize, usize)> {
    let cpu = current_cpu()?;
    Some((cpu, l3_of(cpu)?))
}

fn this_l3() -> Option<usize> {
    this_cpu().map(|(_, l3)| l3)
}

#[cfg(target_os = "linux")]
fn current_cpu() -> Option<usize> {
    // SAFETY: `sched_getcpu` takes no arguments and reads no memory of ours.
    let cpu = unsafe { libc::sched_getcpu() };
    usize::try_from(cpu).ok()
}

#[cfg(not(target_os = "linux"))]
fn current_cpu() -> Option<usize> {
    None
}

/// Each CPU's L3, as the index of its L3 among the distinct ids the
/// machine lists (`/sys/devices/system/cpu/cpuN/cache/index3/id`), read once.
fn l3_of(cpu: usize) -> Option<usize> {
    static MAP: std::sync::OnceLock<Vec<Option<usize>>> = std::sync::OnceLock::new();
    MAP.get_or_init(|| {
        let mut ids: Vec<Option<u64>> = Vec::new();
        for n in 0..1024usize {
            let base = format!("/sys/devices/system/cpu/cpu{n}");
            if !std::path::Path::new(&base).exists() {
                break;
            }
            let id = std::fs::read_to_string(format!("{base}/cache/index3/id"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok());
            ids.push(id);
        }
        let mut distinct: Vec<u64> = ids.iter().flatten().copied().collect();
        distinct.sort_unstable();
        distinct.dedup();
        ids.iter()
            .map(|id| id.and_then(|id| distinct.iter().position(|&d| d == id)))
            .collect()
    })
    .get(cpu)
    .copied()
    .flatten()
}

/// Page faults and context switches: `(minor, major, voluntary, involuntary)`.
pub type Usage = (u64, u64, u64, u64);

/// This thread's [`Usage`] so far.
pub fn thread_usage() -> Option<Usage> {
    usage(Who::Thread)
}

/// The process's [`Usage`] so far.
pub fn process_usage() -> Option<Usage> {
    usage(Who::Process)
}

enum Who {
    Thread,
    Process,
}

#[cfg(target_os = "linux")]
fn usage(who: Who) -> Option<Usage> {
    let mut ru = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    let which = match who {
        Who::Thread => libc::RUSAGE_THREAD,
        Who::Process => libc::RUSAGE_SELF,
    };
    // SAFETY: `ru` is a valid, zeroed `rusage` the call fills.
    if unsafe { libc::getrusage(which, ru.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: filled by the successful call above (and zeroed before it).
    let ru = unsafe { ru.assume_init() };
    let n = |v: libc::c_long| u64::try_from(v).unwrap_or(0);
    Some((
        n(ru.ru_minflt),
        n(ru.ru_majflt),
        n(ru.ru_nvcsw),
        n(ru.ru_nivcsw),
    ))
}

#[cfg(not(target_os = "linux"))]
fn usage(_who: Who) -> Option<Usage> {
    None
}

/// `b − a`, field by field.
pub fn usage_delta(a: Usage, b: Usage) -> Usage {
    (
        b.0.saturating_sub(a.0),
        b.1.saturating_sub(a.1),
        b.2.saturating_sub(a.2),
        b.3.saturating_sub(a.3),
    )
}

/// One wave's reading: the driver's CPU and L3 as it began, and its hash
/// jobs per L3.
pub type WaveReading = (Option<(usize, usize)>, [u64; MAX_L3]);

/// What level 1's execution read under the knob.
#[derive(Clone, Debug, Default)]
pub struct Placement {
    /// Per landed piece, in landing order: the driver's CPU and L3 as the
    /// wave began, and the wave's hash jobs per L3.
    pub waves: Vec<WaveReading>,
    /// The driver thread's usage over the execution, and the process's.
    pub thread: Option<Usage>,
    pub process: Option<Usage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The knob is named and opt-in; a delta never underflows.
    #[test]
    fn the_placement_knob_is_named_and_off_by_default() {
        assert_eq!(PLACEMENT_ENV, "LAMBDA_VM_LEVELS_PLACEMENT");
        assert!(!super::super::wlfm_knobs::knob_word_on(None));
        assert!(!super::super::wlfm_knobs::knob_word_on(Some("0")));
        assert!(super::super::wlfm_knobs::knob_word_on(Some("1")));
        assert_eq!(usage_delta((5, 1, 9, 2), (7, 1, 12, 1)), (2, 0, 3, 0));
        // Counting reads back what it counted, and zeroes.
        let _ = take_jobs();
        assert_eq!(take_jobs(), [0; MAX_L3]);
    }
}
