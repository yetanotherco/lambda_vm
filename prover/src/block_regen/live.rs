//! Live regeneration (D-REGEN §8 R2): a tier of the spill policy's `auto`.
//!
//! Phase A records every streamed instance's recipe ([`super::Recorder`]).
//! Once the policy would move an instance off the host, regeneration arms: a
//! regenerable instance is dropped instead of spilled — its packed bytes freed
//! after their digest, a [`RegenSlot`] left in their place
//! (`stark::regen`) — and the resident regenerable instances are dropped back,
//! in rank order, until the host is under the target (R-REGEN R6: counted in
//! dropped bytes, since the host's reading cannot fall before the purge).
//! With no disk (the spill policy `off`, I-REGEN §14 P1) the policy decides
//! with no store: once armed, drop-back takes every resident regenerable
//! instance, every later one is dropped, and the rest stays on the host.
//! The drops are made on a thread of their own, never under the committers'
//! lock (R7). Phase B's regenerator ([`run_live`]) rebuilds the dropped
//! instances in rank order and deposits each into its slot, which paces it
//! against the fused tasks that take them.
//!
//! Only a streamed chunk whose fused task recommits on the device is dropped
//! (R-REGEN A1): the kept-top check stands behind the digest there. A block
//! that never arms drops nothing, and phase B starts no regenerator (R8).
//!
//! Nothing hangs: the window ([`RegenWindow`]) is born with its first
//! producer, every thread that may deposit holds one, every job carries a
//! guard that fails its slot when dropped undeposited, a slot the slicer did
//! not cut is failed before any higher rank is dispatched (R10), every exit
//! fails what it did not deliver (R2), the generators keep draining after the
//! window closes (R3), and the block closes the window on every exit of its
//! prove, unwinding included ([`WindowGuard`], R4).

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use stark::regen::{RegenError, RegenProducer, RegenSlot, RegenWindow};

use super::{
    GIB, Recorder, RegenMode, cpu_since, regen_generators, slot as table_slot, thread_cpu_secs,
};
use crate::tables::gpack::TraceForm;
use crate::tables::trace_builder::{ChunkJob, RegenBuilder, StreamTable};

/// Which streamed chunk an instance is.
pub(crate) type StreamKey = (StreamTable, usize);

/// `LAMBDA_VM_BLOCK_REGEN_AHEAD_GIB` (default 4, as the spill read-back's
/// window): the bytes the regenerator may hold deposited ahead of the fused
/// tasks, counted from the window's frontier.
pub(crate) fn regen_ahead_bytes() -> u64 {
    gib_knob("LAMBDA_VM_BLOCK_REGEN_AHEAD_GIB", 4.0)
}

/// `LAMBDA_VM_BLOCK_REGEN_RESERVE_GIB` (default 10): the host bytes auto
/// reserves for the regenerator in phase B once regeneration is armed — its
/// walk state, windows, jobs and generators' outputs, and the traces
/// deposited ahead of their fused tasks (R-REGEN A3: BIG 636's phase-B peak
/// VmRSS with the shadow less without it, +5.90 GiB at p90, plus the 4 GiB
/// window). Never added unarmed.
pub(crate) fn regen_reserve_bytes() -> u64 {
    gib_knob("LAMBDA_VM_BLOCK_REGEN_RESERVE_GIB", 10.0)
}

fn gib_knob(var: &str, default: f64) -> u64 {
    let gib = std::env::var(var)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|g| g.is_finite() && *g >= 0.0)
        .unwrap_or(default);
    (gib * GIB) as u64
}

/// A dropped instance: which chunk, its slot, its packed bytes, and whether
/// drop-back dropped it (it was resident when regeneration armed).
pub(crate) struct Dropped {
    pub(crate) key: StreamKey,
    pub(crate) slot: RegenSlot,
    pub(crate) bytes: u64,
    pub(crate) back: bool,
}

/// One drop for the drop thread: the committed entry it is in, its chunk and
/// rank, and whether it is a drop-back.
pub(crate) struct DropJob {
    pub(crate) index: usize,
    pub(crate) key: StreamKey,
    pub(crate) rank: u64,
    pub(crate) back: bool,
}

/// A resident instance that may be dropped back when regeneration arms.
struct Candidate {
    rank: u64,
    index: usize,
    key: StreamKey,
    bytes: u64,
}

#[derive(Default)]
struct LiveState {
    /// When regeneration armed (seconds after phase A began), the bytes it
    /// had to free, and the bytes drop-back was given to drop.
    armed: Option<(f64, u64, u64)>,
    candidates: Vec<Candidate>,
    drop_tx: Option<mpsc::Sender<DropJob>>,
    dropped: Vec<Dropped>,
    /// Streamed instances not dropped because their fused task would not
    /// recommit on the device (A1), and drops the drop thread found no
    /// longer droppable.
    refused_host: usize,
    refused_late: usize,
    errors: Vec<String>,
    /// The cycles phase A's executor ran at a time: the windows the ranks
    /// were handed out in.
    phase_a_window: Option<usize>,
}

/// Phase A's side of live regeneration (`auto` or `always`): the recipes,
/// the window the dropped traces go into, and what was dropped.
pub(crate) struct LiveRegen {
    mode: RegenMode,
    recorder: Recorder,
    window: Arc<RegenWindow>,
    /// The window's first producer, for the regenerator (or dropped with
    /// nothing to regenerate).
    producer: Mutex<Option<RegenProducer>>,
    reserve: u64,
    started: Instant,
    state: Mutex<LiveState>,
    /// No disk (the spill policy `off`, I-REGEN §14 P1): once armed, every
    /// resident regenerable instance is dropped back and every later one is
    /// dropped; what cannot be regenerated stays on the host.
    no_disk: std::sync::atomic::AtomicBool,
    /// Test only: drop traces whose fused task recommits on the host (the
    /// laptop has no device), which production never does (A1).
    #[cfg(test)]
    pub(crate) host_recommit_ok: std::sync::atomic::AtomicBool,
}

impl LiveRegen {
    /// Live regeneration under `mode` (`auto` or `always`) when phase A can
    /// feed it (`streamed`: it streams its windows on the walker's thread and
    /// packs every streamed instance); none otherwise, saying why.
    pub(crate) fn new(mode: RegenMode, streamed: bool) -> Option<Self> {
        if !matches!(mode, RegenMode::Auto | RegenMode::Always) {
            return None;
        }
        if !streamed {
            eprintln!(
                "BLOCK REGEN: {mode:?} wanted, but phase A does not stream and pack its instances \
                 (LAMBDA_VM_BLOCK_STREAM / LAMBDA_VM_BLOCK_NARROW): off"
            );
            return None;
        }
        Some(Self::with(mode, regen_ahead_bytes(), regen_reserve_bytes()))
    }

    fn with(mode: RegenMode, ahead: u64, reserve: u64) -> Self {
        let (window, producer) = RegenWindow::new(ahead);
        Self {
            mode,
            recorder: Recorder::new(),
            window,
            producer: Mutex::new(Some(producer)),
            reserve,
            started: Instant::now(),
            state: Mutex::new(LiveState::default()),
            no_disk: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            host_recommit_ok: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Test only: live regeneration under `mode` with a window of `ahead`
    /// bytes, dropping traces whose fused task recommits on the host (the
    /// laptop has no device).
    #[cfg(test)]
    pub(crate) fn for_test(mode: RegenMode, ahead: u64) -> Self {
        let live = Self::with(mode, ahead, regen_reserve_bytes());
        live.host_recommit_ok.store(true, Ordering::Relaxed);
        live
    }

    /// Test only: drop-back's need and the bytes it was given, once armed.
    #[cfg(test)]
    pub(crate) fn armed_for_test(&self) -> Option<(u64, u64)> {
        self.lock().armed.map(|(_, need, planned)| (need, planned))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LiveState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn recorder(&self) -> &Recorder {
        &self.recorder
    }

    pub(crate) fn window(&self) -> &Arc<RegenWindow> {
        &self.window
    }

    /// Whether every regenerable instance is dropped (the test mode).
    pub(crate) fn always(&self) -> bool {
        self.mode == RegenMode::Always
    }

    /// Whether this is `auto` (the policy's tier).
    pub(crate) fn auto(&self) -> bool {
        self.mode == RegenMode::Auto
    }

    /// No disk: the spill policy is `off`, so the policy decides with no
    /// store (I-REGEN §14 P1). Set before phase A starts.
    pub(crate) fn set_no_disk(&self) {
        self.no_disk.store(true, Ordering::Relaxed);
    }

    pub(crate) fn no_disk(&self) -> bool {
        self.no_disk.load(Ordering::Relaxed)
    }

    pub(crate) fn is_armed(&self) -> bool {
        self.lock().armed.is_some()
    }

    /// The reserve regeneration adds once armed (the spill policy carries it
    /// from then on; unarmed, its decisions, queue budgets and purge are
    /// today's).
    pub(crate) fn reserve(&self) -> u64 {
        self.reserve
    }

    /// The rank of an instance that may be dropped: a streamed chunk with a
    /// recipe whose fused task recommits on the device (A1). `None`
    /// otherwise (a streamed one refused for A1 is counted).
    pub(crate) fn droppable(
        &self,
        key: Option<StreamKey>,
        recommits_on_device: bool,
    ) -> Option<u64> {
        let (table, index) = key?;
        let rank = self.recorder.rank_of(table, index)?;
        #[cfg(test)]
        let recommits_on_device =
            recommits_on_device || self.host_recommit_ok.load(Ordering::Relaxed);
        if !recommits_on_device {
            self.lock().refused_host += 1;
            return None;
        }
        Some(rank)
    }

    /// The cycles phase A's executor runs at a time (its windows, which the
    /// ranks follow): the regenerator must walk the same (R-REGEN S3).
    pub(crate) fn set_phase_a_window(&self, cycles: usize) {
        self.lock().phase_a_window = Some(cycles);
    }

    /// The drop thread's queue.
    pub(crate) fn set_drop_tx(&self, tx: mpsc::Sender<DropJob>) {
        self.lock().drop_tx = Some(tx);
    }

    /// No more drops: the drop thread finishes what is queued and ends.
    pub(crate) fn close_drops(&self) {
        self.lock().drop_tx = None;
    }

    /// Arm on the first decision to move an instance off the host: from now
    /// on the reserve counts, and the resident regenerable instances are
    /// dropped back in rank order until their bytes reach `need` (R6) — all
    /// of them with no disk (§14 P1). Returns whether this call armed.
    pub(crate) fn arm(&self, need: u64) -> bool {
        let mut state = self.lock();
        if state.armed.is_some() {
            return false;
        }
        let all = self.no_disk();
        let mut candidates = std::mem::take(&mut state.candidates);
        candidates.sort_by_key(|c| c.rank);
        let mut planned = 0u64;
        for c in candidates {
            if planned >= need && !all {
                break;
            }
            // Counted only once queued (R-REGEN S4).
            if let Some(tx) = &state.drop_tx
                && tx
                    .send(DropJob {
                        index: c.index,
                        key: c.key,
                        rank: c.rank,
                        back: true,
                    })
                    .is_ok()
            {
                planned += c.bytes;
            }
        }
        state.armed = Some((self.started.elapsed().as_secs_f64(), need, planned));
        true
    }

    /// A droppable instance kept resident: a drop-back candidate, or — kept
    /// by a decision made before another committer armed, while drop-back is
    /// short of its need (always, with no disk) — dropped back now.
    pub(crate) fn kept(&self, index: usize, key: StreamKey, rank: u64, bytes: u64) {
        let all = self.no_disk();
        let mut state = self.lock();
        let state = &mut *state;
        match &mut state.armed {
            None => state.candidates.push(Candidate {
                rank,
                index,
                key,
                bytes,
            }),
            Some((_, need, planned)) if all || *planned < *need => {
                if let Some(tx) = &state.drop_tx
                    && tx
                        .send(DropJob {
                            index,
                            key,
                            rank,
                            back: true,
                        })
                        .is_ok()
                {
                    *planned += bytes;
                }
            }
            Some(_) => {}
        }
    }

    /// Drop an instance now (on the drop thread).
    pub(crate) fn drop_later(&self, index: usize, key: StreamKey, rank: u64) {
        let state = self.lock();
        if let Some(tx) = &state.drop_tx {
            let _ = tx.send(DropJob {
                index,
                key,
                rank,
                back: false,
            });
        }
    }

    pub(crate) fn dropped_one(&self, dropped: Dropped) {
        self.lock().dropped.push(dropped);
    }

    /// A drop the drop thread found no longer droppable (spilled, shared or
    /// already dropped): it stays as it is.
    pub(crate) fn refused_late(&self) {
        self.lock().refused_late += 1;
    }

    pub(crate) fn drop_error(&self, why: String) {
        self.lock().errors.push(why);
    }

    pub(crate) fn errors(&self) -> Vec<String> {
        self.lock().errors.clone()
    }

    /// The `BLOCK REGEN dropped` line.
    pub(crate) fn line(&self) -> String {
        let state = self.lock();
        let (n, bytes) = (
            state.dropped.len(),
            state.dropped.iter().map(|d| d.bytes).sum::<u64>(),
        );
        let back: Vec<&Dropped> = state.dropped.iter().filter(|d| d.back).collect();
        let armed = match state.armed {
            Some((at, need, planned)) => format!(
                "armed at {at:.1} s (need {:.2} GiB, drop-back given {:.2} GiB)",
                need as f64 / GIB,
                planned as f64 / GIB
            ),
            None => "never armed".to_string(),
        };
        format!(
            "BLOCK REGEN dropped: {:?}{} · {n} instances {:.2} GiB (drop-back {} instances {:.2} \
             GiB) · {armed} · {} recipes · refused {} (host recommit) {} (no longer droppable) · \
             reserve {:.1} GiB once armed · {} errors",
            self.mode,
            if self.no_disk() { " (no disk)" } else { "" },
            bytes as f64 / GIB,
            back.len(),
            back.iter().map(|d| d.bytes).sum::<u64>() as f64 / GIB,
            self.recorder.len(),
            state.refused_host,
            state.refused_late,
            self.reserve as f64 / GIB,
            state.errors.len(),
        )
    }

    /// What phase B needs to regenerate, when something was dropped; `None`
    /// otherwise, and the window's producer goes with it (R8: no regenerator,
    /// no builder, no thread for a block that dropped nothing).
    pub(crate) fn into_plan(self) -> Option<LivePlan> {
        let producer = self
            .producer
            .into_inner()
            .unwrap_or_else(|e| e.into_inner());
        let state = self.state.into_inner().unwrap_or_else(|e| e.into_inner());
        let mut dropped = state.dropped;
        if dropped.is_empty() {
            return None;
        }
        let producer = producer?;
        dropped.sort_by_key(|d| d.slot.order_key());
        Some(LivePlan {
            window: self.window,
            producer,
            dropped,
            phase_a_window: state.phase_a_window,
        })
    }
}

/// Phase B's regeneration: the window, its first producer, the dropped
/// instances in rank order, and the windows phase A handed them out in.
pub(crate) struct LivePlan {
    pub(crate) window: Arc<RegenWindow>,
    pub(crate) producer: RegenProducer,
    pub(crate) dropped: Vec<Dropped>,
    pub(crate) phase_a_window: Option<usize>,
}

/// Closes the window when dropped, and on demand (R4): the block's prove
/// ended — returned, refused or unwound — so a regenerator waiting to deposit
/// for takers that are gone returns, before the scope joins it.
pub(crate) struct WindowGuard(Arc<RegenWindow>);

impl WindowGuard {
    pub(crate) fn new(window: &Arc<RegenWindow>) -> Self {
        Self(Arc::clone(window))
    }

    pub(crate) fn close(&self, why: &str) {
        self.0.close(why);
    }
}

impl Drop for WindowGuard {
    fn drop(&mut self) {
        self.0.close("the block's prove ended");
    }
}

/// A generator alive while held (unwinding included).
struct Alive<'a>(&'a std::sync::atomic::AtomicUsize);

impl Drop for Alive<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A job and the slot it fills: dropped undeposited (a generator that
/// panicked outside its catch, a channel torn down), it fails its slot (R10).
struct JobGuard {
    job: Option<ChunkJob>,
    slot: RegenSlot,
    settled: bool,
}

impl Drop for JobGuard {
    fn drop(&mut self) {
        if !self.settled {
            self.slot.fail("its generator stopped before depositing it");
        }
    }
}

/// What the live regenerator did.
#[derive(Debug, Default)]
pub(crate) struct LiveReport {
    pub(crate) dropped: usize,
    pub(crate) deposited: usize,
    pub(crate) mismatches: usize,
    /// Deposits refused because the window had closed.
    pub(crate) closed: usize,
    pub(crate) failures: Vec<String>,
    /// Dropped instances the slicer did not cut before a higher rank, failed
    /// at once (R10).
    pub(crate) skipped: usize,
    /// Chunks cut that were not dropped (resident or spilled), not generated.
    pub(crate) beyond: usize,
    pub(crate) windows: usize,
    pub(crate) cycles: usize,
    pub(crate) bytes: u64,
    pub(crate) first_ready: Option<f64>,
    pub(crate) last_ready: Option<f64>,
    pub(crate) wall: f64,
    /// Thread CPU: the executor, the walker (with its start state), the
    /// generators summed (Linux only).
    pub(crate) cpu: [Option<f64>; 3],
    pub(crate) generators: usize,
    pub(crate) error: Option<String>,
    pub(crate) state_bytes: usize,
}

impl LiveReport {
    pub(crate) fn cpu_total(&self) -> Option<f64> {
        Some(self.cpu[0]? + self.cpu[1]? + self.cpu[2]?)
    }

    /// The `BLOCK REGEN live` line.
    pub(crate) fn line(&self) -> String {
        let secs = |s: Option<f64>| s.map_or("n/a".to_string(), |s| format!("{s:.1}"));
        format!(
            "BLOCK REGEN live: {} of {} dropped instances deposited · {} mismatches · {} failed · {} \
             skipped (not cut in rank order) · {} refused (window closed) · {} not dropped (skipped) · \
             {} windows walked ({:.2} M cycles) · {:.2} GiB packed · ready from {} s to {} s · wall \
             {:.2} s · CPU exec {} · walk {} · {} generators {} = {} s · walk state {:.2} GiB{}",
            self.deposited,
            self.dropped,
            self.mismatches,
            self.failures.len(),
            self.skipped,
            self.closed,
            self.beyond,
            self.windows,
            self.cycles as f64 / 1e6,
            self.bytes as f64 / GIB,
            secs(self.first_ready),
            secs(self.last_ready),
            self.wall,
            secs(self.cpu[0]),
            secs(self.cpu[1]),
            self.generators,
            secs(self.cpu[2]),
            secs(self.cpu_total()),
            self.state_bytes as f64 / GIB,
            self.error
                .as_ref()
                .map_or(String::new(), |e| format!(" · stopped: {e}")),
        )
    }
}

/// Test hooks into the live regenerator's exits (R2/R10): each makes one of
/// them happen, so a test can check that no slot is left waiting.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LiveFaults {
    /// The walk fails at this window.
    pub(crate) walk_error_at: Option<usize>,
    /// The executor ends (cleanly) after this many windows.
    pub(crate) exec_stops_after: Option<usize>,
    /// The generator that takes the job of this rank panics outside its
    /// catch.
    pub(crate) generator_panics_at_rank: Option<u64>,
    /// The slicer never hands out this rank's chunk.
    pub(crate) skip_rank: Option<u64>,
    /// No generator starts.
    pub(crate) no_generators: bool,
}

/// The live regenerator (D-REGEN §2.3, sequential form): `builder` walks the
/// run of `program` on `private_input`, executed `window_cycles` cycles at a
/// time on a thread of its own; every chunk it cuts that was dropped is
/// generated in `form` (packed as it is written under G-pack, else 64-bit),
/// packed if it is not, and deposited into its slot on one of `generators`
/// threads, in rank order of dispatch. Never panics; every exit leaves no
/// slot waiting.
///
/// The ranks are phase A's hand-out order, window by window, so the windows
/// must be phase A's (`max_rows.cpu` cycles): a plan whose phase A walked
/// other windows than `window_cycles` (or did not say) is refused whole, every
/// slot failed before anything runs (R-REGEN S3). Within the same windows, a
/// rank the slicer cuts after a higher one is failed — its table refused,
/// never waited on (R10).
#[cfg_attr(test, allow(clippy::too_many_arguments))]
pub(crate) fn run_live(
    program: &Elf,
    private_input: &[u8],
    builder: RegenBuilder,
    window_cycles: usize,
    plan: LivePlan,
    generators: usize,
    form: TraceForm,
    #[cfg(test)] faults: LiveFaults,
) -> LiveReport {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let started = Instant::now();
    let LivePlan {
        window,
        producer,
        dropped,
        phase_a_window,
    } = plan;
    if phase_a_window != Some(window_cycles) {
        let why = format!(
            "the regenerator walks windows of {window_cycles} cycles, phase A walked {}: its \
             ranks do not hold",
            phase_a_window.map_or("unknown windows".to_string(), |w| format!("{w}"))
        );
        for d in &dropped {
            d.slot.fail(&why);
        }
        drop(producer);
        return LiveReport {
            dropped: dropped.len(),
            skipped: dropped.len(),
            generators,
            error: Some(why),
            wall: started.elapsed().as_secs_f64(),
            ..LiveReport::default()
        };
    }
    let mut builder = builder;
    let mut report = LiveReport {
        dropped: dropped.len(),
        generators,
        ..LiveReport::default()
    };
    // Position of each dropped chunk in rank order.
    let position: BTreeMap<(usize, usize), usize> = dropped
        .iter()
        .enumerate()
        .map(|(i, d)| ((table_slot(d.key.0), d.key.1), i))
        .collect();
    let mut settled = vec![false; dropped.len()];
    let fail_from = |settled: &mut [bool], from: usize, upto: usize, why: &str| -> usize {
        let mut n = 0;
        for i in from..upto {
            if !settled[i] {
                dropped[i].slot.fail(why);
                settled[i] = true;
                n += 1;
            }
        }
        n
    };
    let deposited = std::sync::atomic::AtomicUsize::new(0);
    let mismatches = std::sync::atomic::AtomicUsize::new(0);
    let closed = std::sync::atomic::AtomicUsize::new(0);
    let failures = Mutex::new(Vec::new());
    let ready = Mutex::new((None::<f64>, None::<f64>, 0u64));
    let generators = generators.max(1);
    let (job_tx, job_rx) = mpsc::sync_channel::<JobGuard>(2 * generators);
    let job_rx = Mutex::new(job_rx);
    // Generators alive: when none is, the walker stops handing out (a full
    // queue would otherwise hold it forever).
    let alive = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        let (log_tx, log_rx) = mpsc::sync_channel::<Vec<executor::vm::logs::Log>>(2);
        let window_ref = &window;
        let exec = std::thread::Builder::new()
            .name("regen-exec".to_string())
            .spawn_scoped(s, move || {
                let cpu0 = thread_cpu_secs();
                let run = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
                    let mut executor = Executor::new(program, private_input.to_vec())
                        .map_err(|e| format!("the executor: {e}"))?;
                    #[cfg(test)]
                    let mut sent = 0usize;
                    while let Some(logs) = executor
                        .resume_with_limit(window_cycles)
                        .map_err(|e| format!("the execution: {e}"))?
                    {
                        #[cfg(test)]
                        {
                            if faults.exec_stops_after.is_some_and(|n| sent >= n) {
                                break;
                            }
                            sent += 1;
                        }
                        if window_ref.is_closed() || log_tx.send(logs.to_vec()).is_err() {
                            break;
                        }
                    }
                    Ok(())
                }))
                .unwrap_or_else(|_| Err("the executor panicked".to_string()));
                (run, cpu_since(cpu0))
            });
        let exec = match exec {
            Ok(exec) => Some(exec),
            Err(e) => {
                report.error = Some(format!("the executor thread did not start: {e}"));
                None
            }
        };
        let mut gens = Vec::with_capacity(generators);
        #[cfg(test)]
        let generators = if faults.no_generators { 0 } else { generators };
        if exec.is_some() {
            for g in 0..generators {
                let Some(token) = window.producer() else {
                    break;
                };
                let (job_rx, deposited, mismatches, closed, failures, ready, alive) = (
                    &job_rx,
                    &deposited,
                    &mismatches,
                    &closed,
                    &failures,
                    &ready,
                    &alive,
                );
                alive.fetch_add(1, Ordering::SeqCst);
                let spawned = std::thread::Builder::new()
                    .name(format!("regen-gen-{g}"))
                    .spawn_scoped(s, move || {
                        let _token = token;
                        let _alive = Alive(alive);
                        let cpu0 = thread_cpu_secs();
                        loop {
                            let next = job_rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                            let Ok(mut guard) = next else {
                                break;
                            };
                            let Some(job) = guard.job.take() else {
                                continue;
                            };
                            #[cfg(test)]
                            if faults.generator_panics_at_rank == Some(guard.slot.rank()) {
                                panic!("a generator dies outside its catch (test)");
                            }
                            let (table, index) = (job.table, job.index);
                            let built = catch_unwind(AssertUnwindSafe(|| {
                                let mut trace = job.generate_as(form).trace;
                                if !trace.is_main_narrow() {
                                    trace.pack_main_narrow();
                                }
                                // Moved out, not copied.
                                trace.take_main_for_regen()
                            }));
                            let narrow = match built {
                                Ok(Some(narrow)) => narrow,
                                Ok(_) => {
                                    let why = format!("{table:?}[{index}]: the trace did not pack");
                                    guard.slot.fail(&why);
                                    guard.settled = true;
                                    failures.lock().unwrap_or_else(|e| e.into_inner()).push(why);
                                    continue;
                                }
                                Err(_) => {
                                    let why = format!("{table:?}[{index}]: generation panicked");
                                    guard.slot.fail(&why);
                                    guard.settled = true;
                                    failures.lock().unwrap_or_else(|e| e.into_inner()).push(why);
                                    continue;
                                }
                            };
                            let bytes = narrow.data().len() as u64;
                            let out = guard.slot.deposit(narrow);
                            // Deposited, refused or closed: the slot is settled
                            // either way (a closed window settles every waiter).
                            guard.settled = true;
                            match out {
                                Ok(()) => {
                                    deposited.fetch_add(1, Ordering::Relaxed);
                                    let at = started.elapsed().as_secs_f64();
                                    let mut r = ready.lock().unwrap_or_else(|e| e.into_inner());
                                    r.0 = Some(r.0.map_or(at, |f: f64| f.min(at)));
                                    r.1 = Some(r.1.map_or(at, |l: f64| l.max(at)));
                                    r.2 += bytes;
                                }
                                Err(RegenError::Mismatch) => {
                                    mismatches.fetch_add(1, Ordering::Relaxed);
                                }
                                // Keep draining (R3): the walker may be waiting
                                // on a full queue.
                                Err(RegenError::Closed(_)) => {
                                    closed.fetch_add(1, Ordering::Relaxed);
                                }
                                Err(RegenError::Failed(why)) => {
                                    failures
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner())
                                        .push(format!("{table:?}[{index}]: {why}"));
                                }
                            }
                        }
                        cpu_since(cpu0)
                    });
                match spawned {
                    Ok(handle) => gens.push(handle),
                    Err(e) => {
                        alive.fetch_sub(1, Ordering::SeqCst);
                        report
                            .failures
                            .push(format!("generator {g} did not start: {e}"));
                    }
                }
            }
        }
        // This thread walks and cuts, and hands out the dropped chunks in
        // rank order; a dropped chunk not cut before a higher rank is failed
        // at once (R10).
        let cpu0 = thread_cpu_secs();
        let mut next = 0usize;
        let mut stop: Option<String> = None;
        if gens.is_empty() {
            stop = Some(
                report
                    .error
                    .clone()
                    .unwrap_or_else(|| "no generator started".to_string()),
            );
        } else {
            for logs in log_rx.iter() {
                if window.is_closed() {
                    stop = Some("the window closed".to_string());
                    break;
                }
                #[cfg(test)]
                if faults.walk_error_at == Some(report.windows) {
                    stop = Some("the regeneration walk failed (test)".to_string());
                    break;
                }
                report.windows += 1;
                let jobs = match catch_unwind(AssertUnwindSafe(|| builder.push(&logs))) {
                    Ok(Ok(jobs)) => jobs,
                    Ok(Err(e)) => {
                        stop = Some(format!("the regeneration walk: {e}"));
                        break;
                    }
                    Err(_) => {
                        stop = Some("the regeneration walk panicked".to_string());
                        break;
                    }
                };
                // The window's dropped chunks in rank order: within a window
                // the order is the slicer's own, so only the windows must be
                // phase A's.
                let mut cut: Vec<(usize, ChunkJob)> = Vec::with_capacity(jobs.len());
                for job in jobs {
                    match position.get(&(table_slot(job.table), job.index)) {
                        Some(&pos) => cut.push((pos, job)),
                        None => report.beyond += 1,
                    }
                }
                cut.sort_by_key(|&(pos, _)| pos);
                for (pos, job) in cut {
                    #[cfg(test)]
                    if faults.skip_rank == Some(dropped[pos].slot.rank()) {
                        continue;
                    }
                    if pos < next || settled[pos] {
                        continue;
                    }
                    report.skipped += fail_from(
                        &mut settled,
                        next,
                        pos,
                        "the slicer did not cut it before a higher rank",
                    );
                    settled[pos] = true;
                    next = pos + 1;
                    let mut guard = JobGuard {
                        job: Some(job),
                        slot: dropped[pos].slot.clone(),
                        settled: false,
                    };
                    // A job that cannot be handed out is dropped, and its
                    // guard fails its slot.
                    loop {
                        match job_tx.try_send(guard) {
                            Ok(()) => break,
                            Err(mpsc::TrySendError::Full(back)) => {
                                if alive.load(Ordering::SeqCst) == 0 {
                                    drop(back);
                                    stop = Some("every generator stopped".to_string());
                                    break;
                                }
                                guard = back;
                                std::thread::sleep(std::time::Duration::from_millis(2));
                            }
                            Err(mpsc::TrySendError::Disconnected(back)) => {
                                drop(back);
                                stop = Some("the generators stopped".to_string());
                                break;
                            }
                        }
                    }
                    if stop.is_some() {
                        break;
                    }
                }
                if stop.is_some() || next == dropped.len() {
                    break;
                }
            }
        }
        // Every exit: what was not handed out fails, with why (R2).
        let why = stop
            .clone()
            .unwrap_or_else(|| "the run ended before its chunk was cut".to_string());
        let unsent = fail_from(&mut settled, next, dropped.len(), &why);
        report.skipped += unsent;
        match stop {
            Some(stop) if stop != "the window closed" => {
                report.error.get_or_insert(stop);
            }
            None if unsent > 0 => {
                report.error.get_or_insert(why);
            }
            _ => {}
        }
        drop(log_rx);
        drop(job_tx);
        drop(producer);
        report.cycles = builder.cycles();
        report.state_bytes = builder.state_bytes();
        report.cpu[1] = cpu_since(cpu0);
        let mut gen_cpu = Some(0.0);
        for g in gens {
            let cpu = g.join().unwrap_or_else(|_| {
                failures
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push("a generator panicked".to_string());
                None
            });
            gen_cpu = gen_cpu.zip(cpu).map(|(a, b)| a + b);
        }
        // Jobs no generator took (every one stopped): their guards fail
        // their slots now, not when this function returns.
        while let Ok(guard) = job_rx.lock().unwrap_or_else(|e| e.into_inner()).try_recv() {
            drop(guard);
        }
        report.cpu[2] = gen_cpu;
        if let Some(exec) = exec {
            match exec.join() {
                Ok((run, cpu)) => {
                    report.cpu[0] = cpu;
                    if let Err(e) = run {
                        report.error.get_or_insert(e);
                    }
                }
                Err(_) => {
                    report
                        .error
                        .get_or_insert("the executor thread panicked".to_string());
                }
            }
        }
    });
    report.deposited = deposited.into_inner();
    report.mismatches = mismatches.into_inner();
    report.closed = closed.into_inner();
    report
        .failures
        .extend(failures.into_inner().unwrap_or_else(|e| e.into_inner()));
    let (first, last, bytes) = ready.into_inner().unwrap_or_else(|e| e.into_inner());
    report.first_ready = first;
    report.last_ready = last;
    report.bytes = bytes;
    report.wall = started.elapsed().as_secs_f64();
    report
}

/// The live regenerator on a thread of `scope` (`regen-walk`).
pub(crate) struct LiveHandle<'scope> {
    handle: std::io::Result<std::thread::ScopedJoinHandle<'scope, LiveReport>>,
    dropped: usize,
}

impl LiveHandle<'_> {
    /// Its report once done; a thread that did not start or that panicked is
    /// a report with the error (its slots failed when its producer went).
    pub(crate) fn join(self) -> LiveReport {
        let failed = |error: String| LiveReport {
            dropped: self.dropped,
            error: Some(error),
            ..LiveReport::default()
        };
        match self.handle {
            Ok(handle) => handle
                .join()
                .unwrap_or_else(|_| failed("the regenerator panicked".to_string())),
            Err(e) => failed(format!("the regenerator did not start: {e}")),
        }
    }
}

/// [`run_live`] over the block's run on a thread of `scope`: windows of
/// `max_rows.cpu` cycles, phase A's (which the ranks need), [`regen_generators`]
/// generators writing each chunk in `form` (phase A's). A regenerator that
/// cannot start drops `plan` — its producer — so every slot fails at once.
pub(crate) fn spawn_live<'scope>(
    scope: &'scope std::thread::Scope<'scope, '_>,
    program: &'scope Elf,
    private_input: &'scope [u8],
    max_rows: &'scope crate::tables::MaxRowsConfig,
    plan: LivePlan,
    form: TraceForm,
) -> LiveHandle<'scope> {
    let dropped = plan.dropped.len();
    let handle = std::thread::Builder::new()
        .name("regen-walk".to_string())
        .spawn_scoped(scope, move || {
            let cpu0 = thread_cpu_secs();
            match RegenBuilder::new(program, private_input, max_rows) {
                Ok(builder) => {
                    let setup = cpu_since(cpu0);
                    let mut report = run_live(
                        program,
                        private_input,
                        builder,
                        max_rows.cpu,
                        plan,
                        regen_generators(),
                        form,
                        #[cfg(test)]
                        LiveFaults::default(),
                    );
                    report.cpu[1] = report.cpu[1].zip(setup).map(|(walk, setup)| walk + setup);
                    report
                }
                Err(e) => {
                    let why = format!("the regenerator: {e}");
                    for d in &plan.dropped {
                        d.slot.fail(&why);
                    }
                    LiveReport {
                        dropped,
                        error: Some(why),
                        ..LiveReport::default()
                    }
                }
            }
        });
    LiveHandle { handle, dropped }
}
