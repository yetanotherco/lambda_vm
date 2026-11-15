//! Optional device timers for the argue's kernels (measurement only).
//!
//! With `LAMBDA_VM_ARGUE_KERNEL_TIMES=1` a launch site brackets its launch with
//! [`start`] and [`end`]: two timing events on the launch's own stream. The
//! elapsed device time, the launch count, the bytes the site says the launch
//! moves and its unit of work are summed per `(kind, depth)` into a process-wide
//! table, read with [`totals`]. Events are settled lazily, as they complete;
//! [`totals`] waits for the rest. Unset (the default), [`start`] makes no event
//! and [`end`] does nothing: no launch, no ordering and no value changes. The
//! events sit on the stream the launch is on, so they order nothing the
//! launches did not already order.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaEvent, CudaStream};

/// A test's override: 0 the environment's word, 1 on, 2 off.
static FORCED: AtomicU8 = AtomicU8::new(0);

/// Whether the timers are on (`LAMBDA_VM_ARGUE_KERNEL_TIMES=1`). Read once.
pub fn enabled() -> bool {
    match FORCED.load(Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LAMBDA_VM_ARGUE_KERNEL_TIMES").is_ok_and(|v| v == "1" || v == "true")
    })
}

/// Forces the timers on or off; `None` back to the environment's word.
#[doc(hidden)]
pub fn force(on: Option<bool>) {
    FORCED.store(
        match on {
            None => 0,
            Some(true) => 1,
            Some(false) => 2,
        },
        Ordering::Relaxed,
    );
}

/// A table row's key: the launch site's kind and its depth.
pub type Key = (&'static str, u32);

/// One `(kind, depth)`'s sums.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Entry {
    pub seconds: f64,
    pub launches: u64,
    /// The bytes the launch sites say they move (their reads and writes of
    /// device memory, counted once — what a cache might save is not taken off).
    pub bytes: u64,
    /// The sites' unit of work: cells, cube indices, or program step-rows.
    pub work: u64,
}

struct Pending {
    key: (&'static str, u32),
    start: CudaEvent,
    end: CudaEvent,
    bytes: u64,
    work: u64,
}

#[derive(Default)]
struct State {
    pending: VecDeque<Pending>,
    pool: Vec<CudaEvent>,
    table: BTreeMap<(&'static str, u32), Entry>,
    lost: u64,
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::default()))
}

/// A launch's start event, when the timers are on.
pub struct Mark(Option<CudaEvent>);

fn event(state: &mut State, stream: &Arc<CudaStream>) -> Option<CudaEvent> {
    let event = match state.pool.pop() {
        Some(event) => event,
        None => stream
            .context()
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .ok()?,
    };
    event.record(stream).ok()?;
    Some(event)
}

fn settle(state: &mut State, pending: Pending) {
    match pending.start.elapsed_ms(&pending.end) {
        Ok(ms) => {
            let entry = state.table.entry(pending.key).or_default();
            entry.seconds += f64::from(ms) * 1e-3;
            entry.launches += 1;
            entry.bytes += pending.bytes;
            entry.work += pending.work;
        }
        Err(_) => state.lost += 1,
    }
    state.pool.push(pending.start);
    state.pool.push(pending.end);
}

/// The next launch's start on `stream`; nothing when the timers are off.
pub fn start(stream: &Arc<CudaStream>) -> Mark {
    if !enabled() {
        return Mark(None);
    }
    let Ok(mut state) = state().lock() else {
        return Mark(None);
    };
    Mark(event(&mut state, stream))
}

/// The launch since `mark` done: its end event on `stream`, summed under
/// `(kind, depth)` once it completes, with the `bytes` and `work` given.
pub fn end(
    stream: &Arc<CudaStream>,
    mark: Mark,
    kind: &'static str,
    depth: u32,
    bytes: u64,
    work: u64,
) {
    let Some(start) = mark.0 else {
        return;
    };
    let Ok(mut state) = state().lock() else {
        return;
    };
    match event(&mut state, stream) {
        Some(end) => state.pending.push_back(Pending {
            key: (kind, depth),
            start,
            end,
            bytes,
            work,
        }),
        None => state.lost += 1,
    }
    // Settle what has completed, oldest first, without waiting.
    while state
        .pending
        .front()
        .is_some_and(|pending| pending.end.is_complete())
    {
        let pending = state.pending.pop_front().expect("a front entry");
        settle(&mut state, pending);
    }
}

/// The process's sums so far, every pending pair waited for: `(kind, depth)`
/// ascending, and the launches whose time could not be read.
pub fn totals() -> (Vec<(Key, Entry)>, u64) {
    let Ok(mut state) = state().lock() else {
        return (Vec::new(), 0);
    };
    while let Some(pending) = state.pending.pop_front() {
        settle(&mut state, pending);
    }
    let rows = state.table.iter().map(|(k, v)| (*k, *v)).collect();
    (rows, state.lost)
}

/// `⌈log2 n⌉`, the depth a program of `n` steps is filed under (0 for none).
pub fn log2_bucket(n: u64) -> u32 {
    if n <= 1 {
        0
    } else {
        64 - (n - 1).leading_zeros()
    }
}

/// A kernel's resources and occupancy at `block` threads and `shared` dynamic
/// bytes: registers a thread, local (spill) bytes a thread, blocks an SM.
pub fn attributes(
    function: &cudarc::driver::CudaFunction,
    block: u32,
    shared: usize,
) -> (i32, i32, u32) {
    (
        function.num_regs().unwrap_or(-1),
        function.local_size_bytes().unwrap_or(-1),
        function
            .occupancy_max_active_blocks_per_multiprocessor(block, shared, None)
            .unwrap_or(0),
    )
}

/// A host wait timed while the timers are on: the wall time from [`HostWait::new`]
/// to its drop, summed under `(kind, depth)` with the drop counted as a launch —
/// a read-back's round trip, as the host sees it.
pub struct HostWait {
    started: Option<(std::time::Instant, &'static str, u32)>,
}

impl HostWait {
    pub fn new(kind: &'static str, depth: u32) -> Self {
        Self {
            started: enabled().then(|| (std::time::Instant::now(), kind, depth)),
        }
    }
}

impl Drop for HostWait {
    fn drop(&mut self) {
        if let Some((started, kind, depth)) = self.started.take()
            && let Ok(mut state) = state().lock()
        {
            let entry = state.table.entry((kind, depth)).or_default();
            entry.seconds += started.elapsed().as_secs_f64();
            entry.launches += 1;
        }
    }
}
