//! ⛔ MEASUREMENTS, never part of a proof: phase A's constraint-sumcheck lane
//! ([`CsProbeMode::Lane`]), phase B's merge probe ([`CsProbe::merge`]), and
//! their readout lines.
//!
//! The lane is a thread beside phase A. After each group's commit the
//! committer hands it a job: the group's tables as views over the group's
//! columns on the card (the store stays alive as long as the job holds it).
//! The lane runs its jobs oldest first on background-class streams, so the
//! commits' kernels are dispatched first; it never starts a job while a commit
//! is in flight; and when phase A's loop ends it finishes the job under way and
//! drops the rest. A job runs the group's constraint sumcheck alone (the bus's
//! rules weighted 0) on a throwaway transcript, as a format that ran it in
//! phase A would, and drops what it proved. Nothing it draws or sends reaches
//! the block's transcript: the proof is the same byte for byte.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use multilinear::whir_chain::GkrPack;

use super::*;

/// The label that opens a lane job's transcript: no proof's transcript opens
/// with it.
const LANE_LABEL: &[u8; 32] = b"LAMBDAVM_PHASE_A_ZC_LANE_ONLY_01";

/// The label a merge probe appends to its copy of the group's fork: the copy
/// is thrown away, and the label keeps its draws apart from the fork's own.
const MERGE_LABEL: &[u8; 32] = b"LAMBDAVM_PHASE_B_MERGE_PROBE_01_";

/// What the lane may hold ([`CsProbeMode::Lane`]).
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct LaneLimits {
    /// Groups held at once, the job under way included.
    pub hold: usize,
    /// When set, the held stores' bytes may not pass it (a test's full
    /// ledger); else the card's ledger decides.
    pub hold_bytes: Option<u64>,
    /// Room kept free beside the held stores, the next commit's growth and the
    /// ledger's promise: a job's own session on the card.
    pub margin: u64,
}

impl Default for LaneLimits {
    fn default() -> Self {
        Self {
            hold: 3,
            hold_bytes: None,
            margin: 5 << 29,
        }
    }
}

/// Why a group's job did not run in phase A.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneDefer {
    /// The lane held [`LaneLimits::hold`] groups already.
    Hold,
    /// Its store would not fit beside the next commit ([`LaneLimits`]).
    Memory,
    /// Still queued when phase A's loop ended.
    Tail,
    /// No store on the card, and the host may not take it.
    NoStore,
}

/// What became of a group's lane job, or of its merge probe.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LaneState {
    /// Off, or it does not apply.
    #[default]
    Off,
    /// It ran to its end.
    Ran,
    /// It did not run in phase A.
    Deferred(LaneDefer),
    /// The card declined a table: stopped before its rounds.
    Declined,
    /// It failed otherwise; the prove goes on.
    Failed,
}

impl LaneState {
    fn word(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Ran => "ran",
            Self::Deferred(LaneDefer::Hold) => "hold",
            Self::Deferred(LaneDefer::Memory) => "memory",
            Self::Deferred(LaneDefer::Tail) => "tail",
            Self::Deferred(LaneDefer::NoStore) => "nostore",
            Self::Declined => "declined",
            Self::Failed => "failed",
        }
    }
}

/// A group's job on the lane ([`CsProbeMode::Lane`]), in seconds since phase
/// A started where a moment.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ZcLaneStamp {
    pub state: LaneState,
    pub tables: usize,
    pub on_card: usize,
    /// The job's set-up, its rounds and, of those, the host's later rounds.
    pub setup: f64,
    pub rounds: f64,
    pub host_tail: f64,
    pub committed_at: f64,
    pub started_at: f64,
    pub ended_at: f64,
    /// The group's store, held from its commit to its job's end.
    pub store_bytes: u64,
    /// Groups held when it was handed over (queued or under way).
    pub ahead: usize,
    /// On the last group only: what phase A's end waited for the lane (the
    /// job under way finishing).
    pub close_wait: f64,
    /// Its constraint sumcheck ran to its end.
    pub zc_ran: bool,
    /// Its bins in the argue's plan, those whose ladders ran (the first ones,
    /// in plan order), and their seconds on the lane.
    pub bins: usize,
    pub bins_ran: usize,
    pub ladders: f64,
}

/// A group's merge probe in phase B ([`CsProbe::merge`]).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MergeStamp {
    pub state: LaneState,
    pub tables: usize,
    pub on_card: usize,
    pub setup: f64,
    pub rounds: f64,
    pub host_tail: f64,
}

impl MergeStamp {
    pub fn secs(&self) -> f64 {
        self.setup + self.rounds
    }
}

/// A job: a group's tables as views over its columns (the store held by
/// them), its roots and its stamp so far.
struct LaneJob<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    group: usize,
    tables: Vec<CommittedTable<'a, F, E>>,
    roots: Vec<Commitment>,
    on_card: bool,
    stamp: ZcLaneStamp,
}

struct LaneQueue<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    jobs: VecDeque<LaneJob<'a, F, E>>,
    /// The store bytes of the job under way.
    running: Option<u64>,
    closing: bool,
    stamps: Vec<(usize, ZcLaneStamp)>,
}

/// The lane ([`CsProbeMode::Lane`]): shared by phase A's committer, which
/// hands it jobs, and the lane's own thread, which runs them.
pub(super) struct ZcLane<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    started: Instant,
    host: bool,
    limits: LaneLimits,
    decided: Option<Arc<AtomicUsize>>,
    bin_log_cells: u8,
    pack: GkrPack,
    work: LaneWork,
    /// A commit is in flight: no job starts.
    committing: AtomicBool,
    /// The most a commit has grown the ledger so far (bytes).
    max_growth: AtomicU64,
    queue: Mutex<LaneQueue<'a, F, E>>,
    wake: Condvar,
}

impl<'a, F, E> ZcLane<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    /// The lane for `probe` under `config`; `None` unless the argue is
    /// batched.
    pub(super) fn new(probe: &CsProbe, config: &ChainConfig) -> Option<Self> {
        let ArgueFormat::Batched {
            bin_log_cells,
            pack,
        } = config.format.argue
        else {
            return None;
        };
        Some(Self {
            started: Instant::now(),
            host: probe.host,
            limits: probe.limits,
            decided: probe.decided.clone(),
            bin_log_cells,
            pack,
            work: match probe.mode {
                CsProbeMode::Lane(work) => work,
                _ => LaneWork::Zc,
            },
            committing: AtomicBool::new(false),
            max_growth: AtomicU64::new(0),
            queue: Mutex::new(LaneQueue {
                jobs: VecDeque::new(),
                running: None,
                closing: false,
                stamps: Vec::new(),
            }),
            wake: Condvar::new(),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LaneQueue<'a, F, E>> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn secs(&self, at: Instant) -> f64 {
        at.saturating_duration_since(self.started).as_secs_f64()
    }

    fn decide(&self) {
        if let Some(decided) = &self.decided {
            decided.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// A commit begins: no job starts until it ends.
    pub(super) fn commit_begins(&self) {
        self.committing
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// The commit ended.
    pub(super) fn commit_ends(&self) {
        self.committing
            .store(false, std::sync::atomic::Ordering::SeqCst);
        self.wake.notify_all();
    }

    /// A commit grew the ledger by `bytes` at its peak.
    pub(super) fn note_growth(&self, bytes: u64) {
        self.max_growth
            .fetch_max(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    /// Group `group`, just committed: its job, held while the lane may hold it
    /// ([`LaneLimits`]), else stamped deferred. `on_card`: its tables point at
    /// the group's store on the card.
    pub(super) fn submit(
        &self,
        group: usize,
        tables: &[CommittedTable<'a, F, E>],
        roots: &[Commitment],
        on_card: bool,
        committed: Instant,
    ) {
        let store_bytes: u64 = tables.iter().map(|t| wide_bytes(t) as u64).sum();
        let mut stamp = ZcLaneStamp {
            tables: tables.len(),
            committed_at: self.secs(committed),
            store_bytes,
            ..Default::default()
        };
        let mut queue = self.lock();
        let held = queue.jobs.len() + usize::from(queue.running.is_some());
        let held_bytes: u64 = queue.jobs.iter().map(|j| j.stamp.store_bytes).sum::<u64>()
            + queue.running.unwrap_or(0);
        stamp.ahead = held;
        let budget = multilinear::gpu::reserve_budget();
        let defer = if !on_card && !self.host {
            Some(LaneDefer::NoStore)
        } else if held >= self.limits.hold {
            Some(LaneDefer::Hold)
        } else if self
            .limits
            .hold_bytes
            .is_some_and(|cap| held_bytes + store_bytes > cap)
            || (budget > 0
                && multilinear::gpu::ledger_reserved()
                    + self.max_growth.load(std::sync::atomic::Ordering::Relaxed)
                    + self.limits.margin
                    > budget)
        {
            Some(LaneDefer::Memory)
        } else {
            None
        };
        if let Some(why) = defer {
            stamp.state = LaneState::Deferred(why);
            queue.stamps.push((group, stamp));
            drop(queue);
            self.decide();
            return;
        }
        // Views over the group's columns: the store lives as long as they do.
        let views = tables.iter().map(|t| t.probe_view(self.host)).collect();
        queue.jobs.push_back(LaneJob {
            group,
            tables: views,
            roots: roots.to_vec(),
            on_card,
            stamp,
        });
        drop(queue);
        self.wake.notify_all();
    }

    /// Phase A's loop ended: the job under way finishes, the rest are dropped.
    pub(super) fn close(&self) {
        self.lock().closing = true;
        self.wake.notify_all();
    }

    /// The lane's thread: runs the jobs until [`Self::close`].
    pub(super) fn run<T>(&self)
    where
        T: crypto::fiat_shamir::is_transcript::IsTranscript<E> + Default,
    {
        // Behind the commits: every stream this thread draws is a background one.
        let _background = multilinear::gpu::background_streams();
        loop {
            let job = {
                let mut queue = self.lock();
                loop {
                    if queue.closing {
                        break None;
                    }
                    if !queue.jobs.is_empty()
                        && !self.committing.load(std::sync::atomic::Ordering::SeqCst)
                    {
                        let job = queue.jobs.pop_front();
                        queue.running = job.as_ref().map(|j| j.stamp.store_bytes);
                        break job;
                    }
                    queue = self
                        .wake
                        .wait_timeout(queue, Duration::from_millis(1))
                        .map(|(guard, _)| guard)
                        .unwrap_or_else(|e| e.into_inner().0);
                }
            };
            let Some(job) = job else { break };
            let (group, stamp) = self.run_job::<T>(job);
            let mut queue = self.lock();
            queue.running = None;
            queue.stamps.push((group, stamp));
            drop(queue);
            self.decide();
        }
        // What was still queued at phase A's end: dropped, its stores with it.
        let left: Vec<LaneJob<'a, F, E>> = self.lock().jobs.drain(..).collect();
        for job in left {
            let mut stamp = job.stamp;
            stamp.state = LaneState::Deferred(LaneDefer::Tail);
            self.lock().stamps.push((job.group, stamp));
            drop(job.tables);
            self.decide();
        }
    }

    /// Waits until no commit is in flight; `false` once phase A's loop has
    /// ended (the job stops there).
    fn wait_turn(&self) -> bool {
        let mut queue = self.lock();
        loop {
            if queue.closing {
                return false;
            }
            if !self.committing.load(std::sync::atomic::Ordering::SeqCst) {
                return true;
            }
            queue = self
                .wake
                .wait_timeout(queue, Duration::from_millis(1))
                .map(|(guard, _)| guard)
                .unwrap_or_else(|e| e.into_inner().0);
        }
    }

    /// A throwaway transcript for group `group`'s part `part` of a job.
    fn transcript<T>(group: usize, part: u64, roots: &[Commitment]) -> T
    where
        T: crypto::fiat_shamir::is_transcript::IsTranscript<E> + Default,
    {
        let mut transcript = T::default();
        transcript.append_bytes(LANE_LABEL);
        transcript.append_bytes(&(group as u64).to_le_bytes());
        transcript.append_bytes(&part.to_le_bytes());
        for root in roots {
            transcript.append_bytes(root);
        }
        transcript
    }

    fn run_job<T>(&self, job: LaneJob<'a, F, E>) -> (usize, ZcLaneStamp)
    where
        T: crypto::fiat_shamir::is_transcript::IsTranscript<E> + Default,
    {
        let LaneJob {
            group,
            tables,
            roots,
            on_card,
            mut stamp,
        } = job;
        stamp.started_at = self.secs(Instant::now());
        let at = if on_card { Where::Device } else { Where::Host };
        let mut state = LaneState::Ran;
        if self.work.ladders() {
            // One bin of the argue's plan at a time, on throwaway challenges;
            // no bin starts while a commit is in flight, none after phase A.
            stamp.bins = batched::argue_bins(&tables, self.bin_log_cells, self.pack);
            for bin in 0..stamp.bins {
                if bin > 0 && !self.wait_turn() {
                    state = LaneState::Deferred(LaneDefer::Tail);
                    break;
                }
                let mut transcript = Self::transcript::<T>(group, 1 + bin as u64, &roots);
                let z: FieldElement<E> = transcript.sample_field_element();
                let alpha: FieldElement<E> = transcript.sample_field_element();
                let beta: FieldElement<E> = transcript.sample_field_element();
                let tick = Instant::now();
                let ran = batched::prove_argue(
                    &tables,
                    &z,
                    &alpha,
                    &beta,
                    self.bin_log_cells,
                    self.pack,
                    &mut transcript,
                    ProverFaults {
                        ladders_only: true,
                        only_bin: Some(bin),
                        ..ProverFaults::default()
                    },
                    at,
                );
                match ran {
                    Err(MlError::DeviceFailed {
                        stage: batched::LADDERS_ONLY_DONE,
                    }) => {
                        stamp.bins_ran += 1;
                        stamp.ladders += tick.elapsed().as_secs_f64();
                    }
                    other => {
                        for table in &tables {
                            table.trace().release_device();
                        }
                        static SAID: std::sync::Once = std::sync::Once::new();
                        SAID.call_once(|| {
                            eprintln!(
                                "BLOCK ZC LANE: group {group} bin {bin}: ladders not run: {:?}",
                                other.err()
                            )
                        });
                        // Off the host, a tree the card would not build is the
                        // format's to run in phase B: declined, not failed.
                        state = if self.host {
                            LaneState::Failed
                        } else {
                            LaneState::Declined
                        };
                        break;
                    }
                }
            }
        }
        if self.work.zc() && state == LaneState::Ran {
            if self.work.ladders() && !self.wait_turn() {
                state = LaneState::Deferred(LaneDefer::Tail);
            } else {
                let mut transcript = Self::transcript::<T>(group, 0, &roots);
                let probed = batched::probe_constraint_sumcheck(
                    &tables,
                    self.bin_log_cells,
                    self.pack,
                    at,
                    self.host,
                    true,
                    &mut transcript,
                );
                state = match probed {
                    Ok(probe) => {
                        stamp.on_card = probe.on_card;
                        stamp.setup = probe.setup;
                        stamp.rounds = probe.rounds;
                        stamp.host_tail = probe.host_tail;
                        stamp.zc_ran = true;
                        LaneState::Ran
                    }
                    Err(MlError::DeviceFailed {
                        stage: batched::PROBE_DECLINED,
                    }) => LaneState::Declined,
                    Err(error) => {
                        static SAID: std::sync::Once = std::sync::Once::new();
                        SAID.call_once(|| {
                            eprintln!("BLOCK ZC LANE: group {group} failed: {error:?}")
                        });
                        LaneState::Failed
                    }
                };
            }
        }
        drop(tables);
        stamp.ended_at = self.secs(Instant::now());
        stamp.state = state;
        (group, stamp)
    }

    /// Every group's stamp, once the lane's thread has ended.
    pub(super) fn into_stamps(self) -> Vec<(usize, ZcLaneStamp)> {
        self.queue
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
            .stamps
    }
}

/// Phase B's merge probe over a group's tables after its argue
/// ([`CsProbe::merge`]), on `fork` (a throwaway copy of the group's fork).
#[allow(clippy::too_many_arguments)]
pub(super) fn run_merge_probe<F, E, T>(
    tables: &[CommittedTable<'_, F, E>],
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
    bin_log_cells: u8,
    pack: GkrPack,
    at: Where,
    host: bool,
    mut fork: T,
) -> MergeStamp
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>,
{
    fork.append_bytes(MERGE_LABEL);
    match batched::probe_merge_sumcheck(tables, z, alpha, bin_log_cells, pack, at, host, &mut fork)
    {
        Ok(probe) => MergeStamp {
            state: LaneState::Ran,
            tables: probe.tables,
            on_card: probe.on_card,
            setup: probe.setup,
            rounds: probe.rounds,
            host_tail: probe.host_tail,
        },
        Err(MlError::DeviceFailed {
            stage: batched::PROBE_DECLINED,
        }) => MergeStamp {
            state: LaneState::Declined,
            tables: tables.len(),
            ..Default::default()
        },
        Err(error) => {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| eprintln!("BLOCK MERGE PROBE: a group failed: {error:?}"));
            MergeStamp {
                state: LaneState::Failed,
                tables: tables.len(),
                ..Default::default()
            }
        }
    }
}

/// The lane's lines for the block's readout ([`CsProbeMode::Lane`]): one a
/// group, then the sum. Empty when the lane was off.
///
/// The hidden credit is the sum, over the groups whose constraint sumcheck
/// ran to its end in phase A, of that group's own constraint sumcheck in
/// phase B ([`GroupStamps::zc_b`]): what a format that ran it in phase A would
/// no longer run in phase B. The ladders' hidden credit is the same for the
/// ladders: over every bin whose ladder ran on the lane, that bin's own trees
/// and ladder in phase B ([`GroupStamps::bins_b`]).
#[doc(hidden)]
pub fn zc_lane_lines(stamps: &[GroupStamps]) -> String {
    if stamps.iter().all(|g| g.zc_lane.state == LaneState::Off) {
        return String::new();
    }
    let mib = |b: u64| b >> 20;
    let mut out = String::new();
    for (g, s) in stamps.iter().enumerate() {
        let l = &s.zc_lane;
        out.push_str(&format!(
            "BLOCK ZC LANE: group {g} · committed at {:.3}s · job {:.3}s to {:.3}s · card {:.3}s \
             (setup {:.3}, rounds {:.3}, host tail {:.3}) · {} · tables {} ({} on the card) · \
             store {} MiB · held ahead {} · zc in phase B {:.3}s · bins {}/{} · ladders {:.3}s (phase B \
             {:.3}s, of the bins run here {:.3}s)\n",
            l.committed_at,
            l.started_at,
            l.ended_at,
            l.setup + l.rounds,
            l.setup,
            l.rounds,
            l.host_tail,
            l.state.word(),
            l.tables,
            l.on_card,
            mib(l.store_bytes),
            l.ahead,
            s.zc_b,
            l.bins_ran,
            l.bins,
            l.ladders,
            s.bins_b.iter().fold(0.0, |a, b| a + b),
            ladders_credit(s),
        ));
    }
    let count = |state: LaneState| stamps.iter().filter(|g| g.zc_lane.state == state).count();
    let ran: Vec<&GroupStamps> = stamps
        .iter()
        .filter(|g| g.zc_lane.state == LaneState::Ran)
        .collect();
    let sum = |f: &dyn Fn(&GroupStamps) -> f64, of: &[&GroupStamps]| {
        of.iter().map(|g| f(g)).fold(0.0, |a, b| a + b)
    };
    let all: Vec<&GroupStamps> = stamps.iter().collect();
    let not_ran: Vec<&GroupStamps> = stamps
        .iter()
        .filter(|g| g.zc_lane.state != LaneState::Ran)
        .collect();
    let deferred = stamps
        .iter()
        .filter(|g| matches!(g.zc_lane.state, LaneState::Deferred(_)))
        .count();
    let hidden = stamps
        .iter()
        .filter(|g| g.zc_lane.zc_ran)
        .map(|g| g.zc_b)
        .fold(0.0, |a, b| a + b);
    let total = |f: &dyn Fn(&GroupStamps) -> f64| stamps.iter().map(f).fold(0.0, |a, b| a + b);
    let peak = stamps.iter().map(|g| g.commit_reserved).max().unwrap_or(0);
    out.push_str(&format!(
        "BLOCK ZC LANE SUM: groups {} · ran {} · deferred {} (hold {} · memory {} · tail {} · no store {}) · \
         declined {} · failed {} · hidden credit {:.3}s · lane card {:.3}s (setup {:.3}, rounds {:.3}, host tail \
         {:.3}) · zc in phase B {:.3}s (of groups not run {:.3}s) · last job ended at {:.3}s · phase A's end \
         waited {:.3}s · phase-A ledger peak {} MiB · deferred groups [{}] · ladders: bins {} of {} · lane \
         ladders {:.3}s · ladders hidden credit {:.3}s · ladders in phase B {:.3}s · lane busy {:.3}s\n",
        stamps.len(),
        ran.len(),
        deferred,
        count(LaneState::Deferred(LaneDefer::Hold)),
        count(LaneState::Deferred(LaneDefer::Memory)),
        count(LaneState::Deferred(LaneDefer::Tail)),
        count(LaneState::Deferred(LaneDefer::NoStore)),
        count(LaneState::Declined),
        count(LaneState::Failed),
        hidden,
        sum(&|g| g.zc_lane.setup + g.zc_lane.rounds, &ran),
        sum(&|g| g.zc_lane.setup, &ran),
        sum(&|g| g.zc_lane.rounds, &ran),
        sum(&|g| g.zc_lane.host_tail, &ran),
        sum(&|g| g.zc_b, &all),
        sum(&|g| g.zc_b, &not_ran),
        ran.iter().map(|g| g.zc_lane.ended_at).fold(0.0, f64::max),
        stamps.iter().map(|g| g.zc_lane.close_wait).fold(0.0, f64::max),
        mib(peak),
        stamps
            .iter()
            .enumerate()
            .filter(|(_, g)| g.zc_lane.state != LaneState::Ran)
            .map(|(i, g)| format!("{i}:{}:{:.3}", g.zc_lane.state.word(), g.zc_b))
            .collect::<Vec<_>>()
            .join(" "),
        stamps.iter().map(|g| g.zc_lane.bins_ran).sum::<usize>(),
        stamps.iter().map(|g| g.zc_lane.bins).sum::<usize>(),
        total(&|g| g.zc_lane.ladders),
        total(&ladders_credit),
        total(&|g| g.bins_b.iter().fold(0.0, |a, b| a + b)),
        total(&|g| (g.zc_lane.ended_at - g.zc_lane.started_at).max(0.0)),
    ));
    out
}

/// A group's ladders' hidden credit: its phase-B bins' own seconds over the
/// bins whose ladders ran on the lane (the first ones, in plan order).
fn ladders_credit(stamps: &GroupStamps) -> f64 {
    stamps
        .bins_b
        .iter()
        .take(stamps.zc_lane.bins_ran)
        .fold(0.0, |a, b| a + b)
}

/// The merge probe's lines for the block's readout ([`CsProbe::merge`]): the
/// sum, then each group's `index:state:seconds`. Empty when it was off.
#[doc(hidden)]
pub fn merge_probe_lines(stamps: &[GroupStamps]) -> String {
    if stamps.iter().all(|g| g.merge.state == LaneState::Off) {
        return String::new();
    }
    let ran: Vec<&MergeStamp> = stamps
        .iter()
        .map(|g| &g.merge)
        .filter(|m| m.state == LaneState::Ran)
        .collect();
    let count = |state: LaneState| stamps.iter().filter(|g| g.merge.state == state).count();
    let sum = |f: &dyn Fn(&MergeStamp) -> f64| ran.iter().map(|m| f(m)).fold(0.0, |a, b| a + b);
    format!(
        "BLOCK MERGE PROBE: groups {} · ran {} · declined {} · failed {} · tables {} ({} on the card) · \
         merge {:.3}s (setup {:.3}, rounds {:.3}, host tail {:.3}) · zc in phase B {:.3}s\n\
         BLOCK MERGE PROBE GROUPS (group:state:s): {}\n",
        stamps.len(),
        ran.len(),
        count(LaneState::Declined),
        count(LaneState::Failed),
        ran.iter().map(|m| m.tables).sum::<usize>(),
        ran.iter().map(|m| m.on_card).sum::<usize>(),
        sum(&|m| m.secs()),
        sum(&|m| m.setup),
        sum(&|m| m.rounds),
        sum(&|m| m.host_tail),
        stamps.iter().map(|g| g.zc_b).fold(0.0, |a, b| a + b),
        stamps
            .iter()
            .enumerate()
            .map(|(g, s)| format!("{g}:{}:{:.4}", s.merge.state.word(), s.merge.secs()))
            .collect::<Vec<_>>()
            .join(" "),
    )
}
