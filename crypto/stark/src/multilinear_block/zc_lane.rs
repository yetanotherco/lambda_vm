//! Phase A's lane: a thread beside the commits that runs, under local
//! lookups, each local group's local argue one bin at a time
//! ([`batched::LocalProver`], on the group's own transcript: the proof's
//! bytes, wherever the bins run); and ⛔ MEASUREMENTS, never part of a proof:
//! the constraint-sumcheck lane ([`CsProbeMode::Lane`]), phase B's merge
//! probe ([`CsProbe::merge`]), and their readout lines.
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
//!
//! A local job is the same kind of job with the group's local argue in it:
//! the challenges and the transcript the committer drew right after the
//! group's commit ([`super::LocalStarted`]). Each unit is one bin. A job that
//! stops (phase A's end, a commit's shed, a hold or memory refusal) keeps its
//! prover and transcript, and phase B runs the bins left at the group's turn;
//! a job whose unit fails is dropped whole, and phase B argues the group
//! from its challenges again. Either way the bytes are the ones phase B alone
//! would send.

use std::any::Any;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use multilinear::whir_chain::GkrPack;

use super::*;

/// A local argue the lane hands on ([`super::LocalStarted`], type-erased:
/// the lane is not generic over the transcript).
type LocalHandOff = Box<dyn Any + Send>;

/// Every group's stamp and every local argue the lane let go of, by group.
type LaneParts = (Vec<(usize, ZcLaneStamp)>, Vec<(usize, LocalHandOff)>);

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
    /// Room kept below the ledger's budget for what the card holds outside
    /// the ledger: a commit's predicted peak (the ledger now, and the largest
    /// growth a commit has made) with the held stores stays under
    /// `budget − unledgered`, or the lane lets stores go before the commit.
    pub unledgered: u64,
    /// A test's stop: each local job stops after this many bins
    /// ([`LaneDefer::Stopped`]); `None` in production.
    pub bins: Option<usize>,
    /// ⛔ A test's fault: each local job's unit at this bin fails after it
    /// ran, its transcript moved past what the bin sent — a failure that
    /// leaves the job partway. `None` in production.
    pub fail_bin: Option<usize>,
    /// How the next commit's growth is predicted ([`LaneGrowth`]).
    pub growth: LaneGrowth,
    /// ⛔ A test's fault: every prediction of a commit's growth is this many
    /// bytes, whatever the commits grew (a misprediction on purpose). `None`
    /// in production.
    pub predict: Option<u64>,
}

impl Default for LaneLimits {
    fn default() -> Self {
        Self {
            hold: 3,
            hold_bytes: None,
            margin: LANE_RESERVE_DEFAULT,
            unledgered: LANE_RESERVE_DEFAULT,
            bins: None,
            fail_bin: None,
            growth: LaneGrowth::Max,
            predict: None,
        }
    }
}

/// How the lane predicts a commit's growth of the card's ledger, the room it
/// keeps for the commit beside the stores it holds ([`LaneLimits::growth`]).
/// A prediction moves only which groups' local bins run in phase A and which
/// at their turn in phase B: the proof is the same byte for byte. A commit
/// that grows past its prediction is refused by the ledger where it asks
/// ([`multilinear::gpu::reserve_budget`]) and takes its host path.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LaneGrowth {
    /// The most any commit of the run has grown the ledger so far.
    #[default]
    Max,
    /// The growth of the last commit of the same group shape (its tables'
    /// widths and heights), with a slack ([`shape_slack`]), capped by
    /// [`LaneGrowth::Max`]; a shape not committed yet takes the maximum. At a
    /// group's own commit the prediction is its shape's; for the commit after
    /// a group's, the group's shape when it repeats the shape before it, else
    /// the maximum.
    Shape,
}

/// The knob that picks [`LaneGrowth`] (`max`, the default, or `shape`).
pub const LANE_GROWTH_ENV: &str = "LAMBDA_VM_LANE_GROWTH";

/// The knob that sets [`LaneLimits::margin`] and [`LaneLimits::unledgered`],
/// each, in MiB ([`LANE_RESERVE_DEFAULT`] unset).
pub const LANE_RESERVE_ENV: &str = "LAMBDA_VM_LANE_RESERVE_MIB";

/// Each of the lane's two reserves by default: 2.5 GiB.
pub const LANE_RESERVE_DEFAULT: u64 = 5 << 29;

/// The fewest and the most MiB [`LANE_RESERVE_ENV`] takes.
const LANE_RESERVE_MIB: std::ops::RangeInclusive<u64> = 256..=8192;

/// The slack a [`LaneGrowth::Shape`] prediction adds to the shape's last
/// growth: a quarter of it, and at least 256 MiB.
pub fn shape_slack(growth: u64) -> u64 {
    (growth / 4).max(1 << 28)
}

/// [`LaneLimits`] as [`LANE_GROWTH_ENV`] and [`LANE_RESERVE_ENV`] read
/// (`growth`, `reserve`): unset or empty, today's limits
/// ([`LaneLimits::default`]); `max` or `shape`; a reserve of 256 to 8192 MiB.
/// Anything else is refused.
pub fn lane_limits_setting(
    growth: Option<&str>,
    reserve: Option<&str>,
) -> Result<LaneLimits, String> {
    let growth = match growth.map(str::trim) {
        None | Some("" | "max") => LaneGrowth::Max,
        Some("shape") => LaneGrowth::Shape,
        Some(v) => {
            return Err(format!(
                "{LANE_GROWTH_ENV}={v}: max (the run's largest commit growth, the default) or \
                 shape (the last commit of the same group shape)"
            ));
        }
    };
    let reserve = match reserve.map(str::trim) {
        None | Some("") => LANE_RESERVE_DEFAULT,
        Some(v) => {
            let mib = v
                .parse::<u64>()
                .ok()
                .filter(|mib| LANE_RESERVE_MIB.contains(mib))
                .ok_or_else(|| {
                    format!(
                        "{LANE_RESERVE_ENV}={v}: each of the lane's two reserves in MiB, {} to {} \
                         (unset: {})",
                        LANE_RESERVE_MIB.start(),
                        LANE_RESERVE_MIB.end(),
                        LANE_RESERVE_DEFAULT >> 20
                    )
                })?;
            mib << 20
        }
    };
    Ok(LaneLimits {
        growth,
        margin: reserve,
        unledgered: reserve,
        ..LaneLimits::default()
    })
}

/// [`lane_limits_setting`] of this process's environment.
pub fn lane_limits_from_env() -> Result<LaneLimits, String> {
    lane_limits_setting(
        std::env::var(LANE_GROWTH_ENV).ok().as_deref(),
        std::env::var(LANE_RESERVE_ENV).ok().as_deref(),
    )
}

/// A group's shape for [`LaneGrowth::Shape`]: its tables' committed widths
/// and heights, in order.
pub type GroupShape = [(usize, usize)];

/// The growths the lane has seen, by group shape, and the last commit's
/// shape ([`LaneGrowth::Shape`]).
#[derive(Default)]
pub(crate) struct GrowthBook {
    by_shape: Vec<(Vec<(usize, usize)>, u64)>,
    last_shape: Option<Vec<(usize, usize)>>,
}

impl GrowthBook {
    /// A commit of `shape` grew the ledger by `bytes`.
    pub(crate) fn note(&mut self, shape: &GroupShape, bytes: u64) {
        match self
            .by_shape
            .iter_mut()
            .find(|(s, _)| s.as_slice() == shape)
        {
            Some((_, last)) => *last = bytes,
            None => self.by_shape.push((shape.to_vec(), bytes)),
        }
        self.last_shape = Some(shape.to_vec());
    }

    /// The predicted growth of a commit of `shape` (`None`: not known), with
    /// `max` the run's largest growth so far.
    pub(crate) fn predict(&self, mode: LaneGrowth, max: u64, shape: Option<&GroupShape>) -> u64 {
        match (mode, shape) {
            (LaneGrowth::Max, _) | (LaneGrowth::Shape, None) => max,
            (LaneGrowth::Shape, Some(shape)) => self
                .by_shape
                .iter()
                .find(|(s, _)| s.as_slice() == shape)
                .map_or(max, |&(_, last)| {
                    last.saturating_add(shape_slack(last)).min(max)
                }),
        }
    }

    /// The predicted growth of the commit after one of `shape`: `shape`'s,
    /// when the commit before was of the same shape (a run), else `max`.
    pub(crate) fn predict_next(&self, mode: LaneGrowth, max: u64, shape: &GroupShape) -> u64 {
        let repeats = self.last_shape.as_deref().is_some_and(|last| last == shape);
        self.predict(mode, max, repeats.then_some(shape))
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
    /// Let go before a commit whose predicted peak left no room for it.
    Shed,
    /// No store on the card, and the host may not take it.
    NoStore,
    /// A local job stopped by a test after [`LaneLimits::bins`] bins.
    Stopped,
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
            Self::Deferred(LaneDefer::Shed) => "shed",
            Self::Deferred(LaneDefer::NoStore) => "nostore",
            Self::Deferred(LaneDefer::Stopped) => "stopped",
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
    /// What this group's commit waited for the lane's unit under way.
    pub commit_wait: f64,
    /// A local job (local lookups): `bins`, `bins_ran` and `ladders` are its
    /// local argue's.
    pub local: bool,
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
    /// A local job's argue begun at commit time ([`super::LocalStarted`]).
    local: Option<Box<dyn Any + Send>>,
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
    /// A unit of the job under way (a bin's ladder, or the constraint
    /// sumcheck) is on the card: no commit starts until it ends.
    unit: bool,
    /// A commit asked the job under way to let its store go at its next unit
    /// boundary.
    shed: bool,
    closing: bool,
    stamps: Vec<(usize, ZcLaneStamp)>,
    /// What each group's commit waited for a unit, by group.
    commit_waits: Vec<(usize, f64)>,
    /// Every local argue the lane let go of, by group, for phase B to
    /// finish.
    local: Vec<(usize, Box<dyn Any + Send>)>,
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
    /// The argue's GKR ladder (one tree layer a step, or two).
    ladder: multilinear::gkr_lockstep::GkrLadder,
    /// The measurement a group with no local job runs, if any.
    work: Option<LaneWork>,
    /// The format's local buses, when the lane runs local jobs.
    local_buses: Option<Vec<u64>>,
    /// A commit is in flight: no job starts.
    committing: AtomicBool,
    /// The most a commit has grown the ledger so far (bytes).
    max_growth: AtomicU64,
    /// The growths seen by group shape ([`LaneGrowth::Shape`]).
    book: Mutex<GrowthBook>,
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
    /// The lane for `probe` under `config`, running local jobs over
    /// `local_buses` when they are given; `None` unless the argue is batched
    /// and the lane has work (a lane probe or local jobs).
    pub(super) fn new(
        probe: &CsProbe,
        config: &ChainConfig,
        local_buses: Option<Vec<u64>>,
    ) -> Option<Self> {
        let ArgueFormat::Batched {
            bin_log_cells,
            pack,
            ladder,
        } = config.format.argue
        else {
            return None;
        };
        let work = match probe.mode {
            CsProbeMode::Lane(work) => Some(work),
            _ => None,
        };
        if work.is_none() && local_buses.is_none() {
            return None;
        }
        Some(Self {
            started: Instant::now(),
            host: probe.host,
            limits: probe.limits,
            decided: probe.decided.clone(),
            bin_log_cells,
            pack,
            ladder,
            work,
            local_buses,
            committing: AtomicBool::new(false),
            max_growth: AtomicU64::new(0),
            book: Mutex::new(GrowthBook::default()),
            queue: Mutex::new(LaneQueue {
                jobs: VecDeque::new(),
                running: None,
                unit: false,
                shed: false,
                closing: false,
                stamps: Vec::new(),
                commit_waits: Vec::new(),
                local: Vec::new(),
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

    /// Whether a commit starting now that grows the ledger by `growth`, with
    /// the ledger at `reserved`, would pass `budget − unledgered` at its
    /// peak.
    fn over(&self, reserved: u64, growth: u64) -> bool {
        let budget = multilinear::gpu::reserve_budget();
        budget > 0 && reserved + growth > budget.saturating_sub(self.limits.unledgered)
    }

    fn book(&self) -> std::sync::MutexGuard<'_, GrowthBook> {
        self.book.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn max_growth(&self) -> u64 {
        self.max_growth.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The predicted growth of a commit of `shape` ([`LaneLimits::growth`];
    /// a test's [`LaneLimits::predict`] first).
    fn predicted(&self, shape: &GroupShape) -> u64 {
        self.limits.predict.unwrap_or_else(|| {
            self.book()
                .predict(self.limits.growth, self.max_growth(), Some(shape))
        })
    }

    /// The predicted growth of the commit after one of `shape`.
    fn predicted_next(&self, shape: &GroupShape) -> u64 {
        self.limits.predict.unwrap_or_else(|| {
            self.book()
                .predict_next(self.limits.growth, self.max_growth(), shape)
        })
    }

    /// Group `group`'s commit begins, its tables of `shape`: no unit starts
    /// until it ends. It waits for the unit under way, and before it lets
    /// queued stores go, oldest first, while its predicted peak would pass
    /// `budget − unledgered`; if that is not enough, the job under way lets
    /// its store go at its unit boundary, and the commit waits for that too.
    /// Returns the commit's predicted growth.
    ///
    /// It never waits for room: the ledger refuses a reservation at once
    /// rather than block (`math_cuda::device::reserve`), so a commit that
    /// grows past its prediction takes its host path where it is refused,
    /// and the only waits here are for the unit under way and the job's next
    /// unit boundary, which the lane reaches without the commit.
    pub(super) fn commit_begins(&self, group: usize, shape: &GroupShape) -> u64 {
        let growth = self.predicted(shape);
        let waited = Instant::now();
        let mut queue = self.lock();
        self.committing
            .store(true, std::sync::atomic::Ordering::SeqCst);
        while queue.unit {
            queue = self
                .wake
                .wait_timeout(queue, Duration::from_millis(1))
                .map(|(guard, _)| guard)
                .unwrap_or_else(|e| e.into_inner().0);
        }
        while self.over(multilinear::gpu::ledger_reserved(), growth) {
            let Some(job) = queue.jobs.pop_front() else {
                break;
            };
            let mut stamp = job.stamp;
            stamp.state = LaneState::Deferred(LaneDefer::Shed);
            queue.stamps.push((job.group, stamp));
            if let Some(local) = job.local {
                queue.local.push((job.group, local));
            }
            // Its store goes now, before the ledger is read again.
            drop(job.tables);
            self.decide();
        }
        if queue.running.is_some() && self.over(multilinear::gpu::ledger_reserved(), growth) {
            queue.shed = true;
            self.wake.notify_all();
            while queue.running.is_some() {
                queue = self
                    .wake
                    .wait_timeout(queue, Duration::from_millis(1))
                    .map(|(guard, _)| guard)
                    .unwrap_or_else(|e| e.into_inner().0);
            }
        }
        queue.shed = false;
        queue
            .commit_waits
            .push((group, waited.elapsed().as_secs_f64()));
        growth
    }

    /// Whether the lane runs local jobs.
    pub(super) fn runs_local(&self) -> bool {
        self.local_buses.is_some()
    }

    /// The commit ended.
    pub(super) fn commit_ends(&self) {
        self.committing
            .store(false, std::sync::atomic::Ordering::SeqCst);
        self.wake.notify_all();
    }

    /// A commit of `shape` grew the ledger by `bytes` at its peak.
    pub(super) fn note_growth(&self, shape: &GroupShape, bytes: u64) {
        self.max_growth
            .fetch_max(bytes, std::sync::atomic::Ordering::Relaxed);
        self.book().note(shape, bytes);
    }

    /// Group `group`, just committed: its job, held while the lane may hold it
    /// ([`LaneLimits`]), else stamped deferred. `on_card`: its tables point at
    /// the group's store on the card. `local`: its local argue begun at
    /// commit time — a local job; a group with none runs the lane's probe, if
    /// one is on, and nothing otherwise.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn submit(
        &self,
        group: usize,
        shape: &GroupShape,
        tables: &[CommittedTable<'a, F, E>],
        roots: &[Commitment],
        on_card: bool,
        committed: Instant,
        local: Option<Box<dyn Any + Send>>,
    ) {
        if local.is_none() && self.work.is_none() {
            // Nothing for the lane: decided at once.
            self.decide();
            return;
        }
        let store_bytes: u64 = tables.iter().map(|t| wide_bytes(t) as u64).sum();
        let mut stamp = ZcLaneStamp {
            tables: tables.len(),
            committed_at: self.secs(committed),
            store_bytes,
            local: local.is_some(),
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
                    + self.predicted_next(shape)
                    + self.limits.margin
                    > budget.saturating_sub(self.limits.unledgered))
        {
            Some(LaneDefer::Memory)
        } else {
            None
        };
        if let Some(why) = defer {
            stamp.state = LaneState::Deferred(why);
            queue.stamps.push((group, stamp));
            if let Some(local) = local {
                queue.local.push((group, local));
            }
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
            local,
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
        T: crypto::fiat_shamir::is_transcript::IsTranscript<E> + Default + Send + 'static,
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
            let (group, stamp, local) = self.run_job::<T>(job);
            let mut queue = self.lock();
            queue.running = None;
            queue.stamps.push((group, stamp));
            if let Some(local) = local {
                queue.local.push((group, local));
            }
            drop(queue);
            self.wake.notify_all();
            self.decide();
        }
        // What was still queued at phase A's end: dropped, its stores with it.
        let left: Vec<LaneJob<'a, F, E>> = self.lock().jobs.drain(..).collect();
        for job in left {
            let mut stamp = job.stamp;
            stamp.state = LaneState::Deferred(LaneDefer::Tail);
            let mut queue = self.lock();
            queue.stamps.push((job.group, stamp));
            if let Some(local) = job.local {
                queue.local.push((job.group, local));
            }
            drop(queue);
            drop(job.tables);
            self.decide();
        }
    }

    /// Starts a unit of the job under way once no commit is in flight:
    /// `Ok(())`, or why the job stops here — phase A's loop ended
    /// ([`LaneDefer::Tail`]), or a commit asked for the job's store
    /// ([`LaneDefer::Shed`]).
    fn begin_unit(&self) -> Result<(), LaneDefer> {
        let mut queue = self.lock();
        loop {
            if queue.closing {
                return Err(LaneDefer::Tail);
            }
            if queue.shed {
                return Err(LaneDefer::Shed);
            }
            if !self.committing.load(std::sync::atomic::Ordering::SeqCst) {
                queue.unit = true;
                return Ok(());
            }
            queue = self
                .wake
                .wait_timeout(queue, Duration::from_millis(1))
                .map(|(guard, _)| guard)
                .unwrap_or_else(|e| e.into_inner().0);
        }
    }

    /// The unit under way ended: a waiting commit may start.
    fn end_unit(&self) {
        self.lock().unit = false;
        self.wake.notify_all();
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

    /// Runs `job`: its local argue's bins when it is a local job, else the
    /// lane's probe. Returns the group, its stamp and the local argue to
    /// hand on (none when a unit failed: phase B argues the group anew).
    fn run_job<T>(
        &self,
        job: LaneJob<'a, F, E>,
    ) -> (usize, ZcLaneStamp, Option<Box<dyn Any + Send>>)
    where
        T: crypto::fiat_shamir::is_transcript::IsTranscript<E> + Default + Send + 'static,
    {
        let LaneJob {
            group,
            tables,
            roots,
            on_card,
            mut stamp,
            local,
        } = job;
        stamp.started_at = self.secs(Instant::now());
        let at = if on_card { Where::Device } else { Where::Host };
        if let Some(local) = local {
            let local = self.run_local::<T>(group, &tables, local, at, &mut stamp);
            drop(tables);
            stamp.ended_at = self.secs(Instant::now());
            return (group, stamp, local);
        }
        let Some(work) = self.work else {
            drop(tables);
            return (group, stamp, None);
        };
        let mut state = LaneState::Ran;
        if work.ladders() {
            // One bin of the argue's plan at a time, on throwaway challenges;
            // no bin starts while a commit is in flight, none after phase A.
            stamp.bins = batched::argue_bins(&tables, self.bin_log_cells, self.pack);
            for bin in 0..stamp.bins {
                if let Err(why) = self.begin_unit() {
                    state = LaneState::Deferred(why);
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
                    self.ladder,
                    &mut transcript,
                    ProverFaults {
                        ladders_only: true,
                        only_bin: Some(bin),
                        ..ProverFaults::default()
                    },
                    at,
                );
                self.end_unit();
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
        if work.zc() && state == LaneState::Ran {
            if let Err(why) = self.begin_unit() {
                state = LaneState::Deferred(why);
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
                self.end_unit();
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
        (group, stamp, None)
    }

    /// A local job: its argue's bins, one a unit, from the first not yet
    /// run, until every bin ran or the job must stop. Returns the argue to
    /// hand on, or none when it is not this lane's (a transcript of another
    /// kind) or a unit failed: a failed bin leaves its transcript partway.
    fn run_local<T>(
        &self,
        group: usize,
        tables: &[CommittedTable<'a, F, E>],
        mut local: Box<dyn Any + Send>,
        at: Where,
        stamp: &mut ZcLaneStamp,
    ) -> Option<Box<dyn Any + Send>>
    where
        T: crypto::fiat_shamir::is_transcript::IsTranscript<E> + Default + Send + 'static,
    {
        let (Some(started), Some(buses)) = (
            local.downcast_mut::<super::LocalStarted<E, T>>(),
            self.local_buses.as_deref(),
        ) else {
            stamp.state = LaneState::Failed;
            return None;
        };
        let super::LocalStarted {
            transcript,
            z,
            alpha,
            prover,
        } = started;
        if prover.is_none() {
            match batched::LocalProver::new(
                tables,
                buses,
                z,
                alpha,
                self.bin_log_cells,
                self.pack,
                self.ladder,
                ProverFaults::default(),
            ) {
                Ok(made) => *prover = Some(made),
                Err(error) => {
                    note_local_failure(group, &error);
                    stamp.state = LaneState::Failed;
                    return None;
                }
            }
        }
        let Some(prover) = prover.as_mut() else {
            stamp.state = LaneState::Failed;
            return None;
        };
        stamp.bins = prover.bins();
        let mut state = LaneState::Ran;
        while !prover.done() {
            if self.limits.bins.is_some_and(|stop| stamp.bins_ran >= stop) {
                state = LaneState::Deferred(LaneDefer::Stopped);
                break;
            }
            if let Err(why) = self.begin_unit() {
                state = LaneState::Deferred(why);
                break;
            }
            let tick = Instant::now();
            let bin = prover.bins_run();
            let ran = prover.step(tables, transcript, at).and_then(|()| {
                if self.limits.fail_bin == Some(bin) {
                    transcript.append_bytes(b"a test's failed unit");
                    return Err(MlError::DeviceFailed {
                        stage: "a test's failed local unit",
                    });
                }
                Ok(())
            });
            self.end_unit();
            match ran {
                Ok(()) => {
                    stamp.bins_ran += 1;
                    stamp.ladders += tick.elapsed().as_secs_f64();
                }
                Err(error) => {
                    for table in tables {
                        table.trace().release_device();
                    }
                    note_local_failure(group, &error);
                    // Off the host, a tree the card would not build is phase
                    // B's to argue: declined, not failed.
                    stamp.state = if self.host {
                        LaneState::Failed
                    } else {
                        LaneState::Declined
                    };
                    return None;
                }
            }
        }
        stamp.state = state;
        Some(local)
    }

    /// Every group's stamp and every local argue the lane let go of, once
    /// the lane's thread has ended.
    pub(super) fn into_parts(self) -> LaneParts {
        let queue = self.queue.into_inner().unwrap_or_else(|e| e.into_inner());
        let local = queue.local;
        let mut stamps = queue.stamps;
        for (group, wait) in queue.commit_waits {
            match stamps.iter_mut().find(|(g, _)| *g == group) {
                Some((_, stamp)) => stamp.commit_wait = wait,
                None => stamps.push((
                    group,
                    ZcLaneStamp {
                        commit_wait: wait,
                        ..Default::default()
                    },
                )),
            }
        }
        (stamps, local)
    }
}

/// The first local job that fails, said once: the group's argue goes to
/// phase B whole.
fn note_local_failure(group: usize, error: &MlError) {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| eprintln!("BLOCK LOCAL LANE: group {group}: left to phase B: {error:?}"));
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
    // The lane's local jobs have lines of their own (`local_lane_lines`).
    if stamps
        .iter()
        .all(|g| g.zc_lane.state == LaneState::Off || g.zc_lane.local)
    {
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
         ladders {:.3}s · ladders hidden credit {:.3}s · ladders in phase B {:.3}s · lane busy {:.3}s · shed {} · commits \
         waited {:.3}s (most {:.3}s)\n",
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
        count(LaneState::Deferred(LaneDefer::Shed)),
        total(&|g| g.zc_lane.commit_wait),
        stamps.iter().map(|g| g.zc_lane.commit_wait).fold(0.0, f64::max),
    ));
    out
}

/// The local argues' lines for the block's readout (local lookups): one a
/// local group, then the sum; empty when no group argued local lookups.
///
/// A group's line: what became of its lane job, its local bins run in phase
/// A of its plan's, the lane's seconds over them, and each local bin's
/// seconds at the group's turn in phase B (the bins the lane left, in plan
/// order). With the lane off every bin runs in phase B, so a lane-off run's
/// per-bin seconds price the bins a lane-on run of the same block ran.
#[doc(hidden)]
pub fn local_lane_lines(stamps: &[GroupStamps]) -> String {
    let local: Vec<(usize, &GroupStamps)> = stamps
        .iter()
        .enumerate()
        .filter(|(_, s)| s.zc_lane.local || !s.local_b.is_empty())
        .collect();
    if local.is_empty() {
        return String::new();
    }
    let secs = |v: &[f64]| v.iter().fold(0.0, |a, b| a + b);
    let mut out = String::new();
    for (g, s) in &local {
        let l = &s.zc_lane;
        let bins = local_bins(s);
        out.push_str(&format!(
            "BLOCK LOCAL LANE: group {g} · {} · bins {} of {bins} in phase A · lane {:.3}s · \
             store {} MiB · commit wait {:.3}s · phase B {:.3}s [{}]\n",
            l.state.word(),
            l.bins_ran,
            l.ladders,
            l.store_bytes >> 20,
            l.commit_wait,
            secs(&s.local_b),
            s.local_b
                .iter()
                .map(|b| format!("{b:.4}"))
                .collect::<Vec<_>>()
                .join(" "),
        ));
    }
    let count = |state: LaneState| {
        local
            .iter()
            .filter(|(_, s)| s.zc_lane.local && s.zc_lane.state == state)
            .count()
    };
    let total =
        |f: &dyn Fn(&GroupStamps) -> f64| local.iter().map(|(_, s)| f(s)).fold(0.0, |a, b| a + b);
    out.push_str(&format!(
        "BLOCK LOCAL LANE SUM: groups {} · lane jobs {} · bins {} of {} in phase A · ran {} · \
         deferred hold {} · memory {} · tail {} · shed {} · no store {} · stopped {} · declined {} · \
         failed {} · lane {:.3}s · phase B {:.3}s · commits waited {:.3}s (most {:.3}s) · last job \
         ended at {:.3}s · phase A's end waited {:.3}s\n",
        local.len(),
        local.iter().filter(|(_, s)| s.zc_lane.local).count(),
        local.iter().map(|(_, s)| s.zc_lane.bins_ran).sum::<usize>(),
        local
            .iter()
            .map(|(_, s)| local_bins(s))
            .sum::<usize>(),
        count(LaneState::Ran),
        count(LaneState::Deferred(LaneDefer::Hold)),
        count(LaneState::Deferred(LaneDefer::Memory)),
        count(LaneState::Deferred(LaneDefer::Tail)),
        count(LaneState::Deferred(LaneDefer::Shed)),
        count(LaneState::Deferred(LaneDefer::NoStore)),
        count(LaneState::Deferred(LaneDefer::Stopped)),
        count(LaneState::Declined),
        count(LaneState::Failed),
        total(&|s| s.zc_lane.ladders),
        total(&|s| secs(&s.local_b)),
        stamps.iter().map(|s| s.zc_lane.commit_wait).fold(0.0, |a, b| a + b),
        stamps.iter().map(|s| s.zc_lane.commit_wait).fold(0.0, f64::max),
        local
            .iter()
            .filter(|(_, s)| s.zc_lane.local)
            .map(|(_, s)| s.zc_lane.ended_at)
            .fold(0.0, f64::max),
        stamps.iter().map(|s| s.zc_lane.close_wait).fold(0.0, f64::max),
    ));
    out.push_str(&lane_growth_line(stamps));
    out
}

/// The lane's predictions of each commit's growth against what the commit
/// grew ([`LaneGrowth`]): per group `predicted/actual` MiB, the largest
/// under-prediction, and the commits that fell back to the host.
#[doc(hidden)]
pub fn lane_growth_line(stamps: &[GroupStamps]) -> String {
    if stamps.iter().all(|s| s.lane_predicted_growth == 0) {
        return String::new();
    }
    let actual = |s: &GroupStamps| s.commit_reserved.saturating_sub(s.commit_base);
    // The first commit is predicted with nothing seen (0 in every mode).
    let under = stamps
        .iter()
        .enumerate()
        .skip(1)
        .map(|(g, s)| (actual(s).saturating_sub(s.lane_predicted_growth), g))
        .max()
        .unwrap_or((0, 0));
    format!(
        "BLOCK LOCAL LANE GROWTH: per group predicted/actual [{}] MiB · largest under-prediction \
         after the first commit {} MiB (group {}) · commits under-predicted {} · commits with a \
         host fallback {}\n",
        stamps
            .iter()
            .map(|s| format!("{}/{}", s.lane_predicted_growth >> 20, actual(s) >> 20))
            .collect::<Vec<_>>()
            .join(", "),
        under.0 >> 20,
        under.1,
        stamps
            .iter()
            .skip(1)
            .filter(|s| actual(s) > s.lane_predicted_growth)
            .count(),
        stamps
            .iter()
            .filter(|s| s.commit_host_fallbacks > 0)
            .count(),
    )
}

/// A local group's bins: those run on the lane and those run in phase B (a
/// job refused before its prover was made knows no plan of its own).
fn local_bins(stamps: &GroupStamps) -> usize {
    stamps
        .zc_lane
        .bins
        .max(stamps.zc_lane.bins_ran + stamps.local_b.len())
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

#[cfg(test)]
mod room_tests {
    use super::*;

    /// The knobs: unset is today's limits; `max` / `shape`; a reserve of
    /// 256 to 8192 MiB sets both reserves; anything else is refused.
    #[test]
    fn the_lane_room_knobs_read_their_words() {
        let today = LaneLimits::default();
        let unset = lane_limits_setting(None, None).expect("unset");
        assert_eq!(
            (unset.growth, unset.margin, unset.unledgered, unset.hold),
            (today.growth, today.margin, today.unledgered, today.hold)
        );
        assert_eq!(today.growth, LaneGrowth::Max);
        assert_eq!(today.margin, 5 << 29);
        assert_eq!(today.unledgered, 5 << 29);
        assert_eq!(today.predict, None);
        let words = lane_limits_setting(Some(" shape "), Some("1024")).expect("words");
        assert_eq!(words.growth, LaneGrowth::Shape);
        assert_eq!((words.margin, words.unledgered), (1 << 30, 1 << 30));
        assert_eq!(
            lane_limits_setting(Some("max"), Some("")).map(|l| l.growth),
            Ok(LaneGrowth::Max)
        );
        for bad in ["shapes", "1", "min"] {
            assert!(lane_limits_setting(Some(bad), None).is_err(), "{bad:?}");
        }
        for bad in ["255", "8193", "-1", "1g", "lots"] {
            assert!(lane_limits_setting(None, Some(bad)).is_err(), "{bad:?}");
        }
        assert!(lane_limits_setting(None, Some("256")).is_ok());
        assert!(lane_limits_setting(None, Some("8192")).is_ok());
    }

    /// The predictions: `max` is the run's largest growth whatever the shape;
    /// `shape` is the shape's last growth with its slack, capped by the
    /// largest, and the largest for a shape not seen; for the commit after
    /// one, the shape's only when the commit before repeated it.
    #[test]
    fn the_growth_book_predicts_by_shape_and_falls_back_to_the_largest() {
        let mib = 1u64 << 20;
        let cpu: &GroupShape = &[(57, 21), (57, 21)];
        let keccak: &GroupShape = &[(400, 16), (7, 16)];
        let mut book = GrowthBook::default();
        assert_eq!(
            book.predict(LaneGrowth::Shape, 0, Some(keccak)),
            0,
            "nothing seen"
        );
        book.note(cpu, 10_000 * mib);
        let max = 10_000 * mib;
        assert_eq!(book.predict(LaneGrowth::Max, max, Some(cpu)), max);
        assert_eq!(
            book.predict(LaneGrowth::Shape, max, Some(keccak)),
            max,
            "unseen shape"
        );
        assert_eq!(book.predict(LaneGrowth::Shape, max, None), max, "no shape");
        book.note(keccak, 50 * mib);
        assert_eq!(
            book.predict(LaneGrowth::Shape, max, Some(keccak)),
            50 * mib + shape_slack(50 * mib),
            "the shape's last growth and its slack"
        );
        assert_eq!(shape_slack(50 * mib), 256 * mib, "the slack's floor");
        assert_eq!(shape_slack(4_000 * mib), 1_000 * mib, "a quarter");
        assert_eq!(
            book.predict(LaneGrowth::Shape, max, Some(cpu)),
            max,
            "capped by the largest"
        );
        assert_eq!(book.predict(LaneGrowth::Max, max, Some(keccak)), max);
        // The commit after a KECCAK_RND group: the last commit was one too.
        assert_eq!(
            book.predict_next(LaneGrowth::Shape, max, keccak),
            50 * mib + shape_slack(50 * mib)
        );
        // After a CPU group, or under `max`: the largest.
        assert_eq!(book.predict_next(LaneGrowth::Shape, max, cpu), max);
        assert_eq!(book.predict_next(LaneGrowth::Max, max, keccak), max);
        // A shape that grows later is predicted from its last growth.
        book.note(keccak, 3_000 * mib);
        assert_eq!(
            book.predict(LaneGrowth::Shape, max, Some(keccak)),
            3_000 * mib + 750 * mib
        );
    }
}
