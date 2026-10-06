//! The block tree's programs and artifacts handed to the provers through
//! slots, filled ahead or as a pipeline into level 0 (the whole-block harness's
//! `NOEPOCH_TREE_AHEAD`).
//!
//! - `1`: every program and its artifacts derived beside the base, on the host
//!   ([`super::block_plan::BlockTreePlan::derive_tree`]); [`Pipe::filled`].
//! - `pipe`, the default (`0` derives each program inline, before it proves):
//!   the leaf programs emitted beside the base (host, no artifacts) — all of
//!   them, or with an emission window W only the first W, the rest emitted by
//!   the builder at most W ahead of its builds (`NOEPOCH_TREE_EMIT_WINDOW`);
//!   once the base is done, [`Pipe::run_builder`] builds each leaf's artifacts on
//!   the card, then emits each node program as soon as its children's ARTIFACTS
//!   exist (a node reads its children's shapes, not their proofs) and builds its
//!   artifacts, level by level, while the leaves prove.
//!
//! A prover takes its program from its slot, waiting for it if the builder is
//! behind; a builder failure fails every waiter. The whole-block driver
//! ([`super::block_tree`]) runs it, in the harness and in the CLI alike.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use super::LfmArtifacts;
use super::block_plan::{BlockTreePlan, TreePrograms};
use super::compiler::LfmProgram;
use super::per_table_aggregator::{DerivedChild, Level};
use super::program_budget::{Permit, ProgramBudget};

/// A program, its artifacts, and its admission under the tree's program
/// budget (none without one): what a slot holds and a prover takes.
pub(super) type Taken = (LfmProgram, LfmArtifacts, Option<Permit>);

/// A program as it is emitted: with its admission, under a budget.
pub(super) type Emitted = (LfmProgram, Option<Permit>);

/// One program and its artifacts, once its builder has put them.
#[derive(Default)]
struct Slot {
    value: Mutex<Option<Taken>>,
    ready: Condvar,
}

/// Threads of the streaming emitter that emits, under a program budget, the
/// leaf programs not emitted beside the base. Level 0 takes ≈ 1.2 leaves a
/// second at p90 and a leaf program takes ≈ 1.8–2.5 s to emit; the window's
/// chunks of 4 kept level 0 waiting (BIG 662: +40 s), four streaming threads
/// keep ahead [I].
const DEFERRED_LEAF_EMITTERS: usize = 4;

/// Fails the budget when dropped armed: an error or a panic in the builder
/// wakes every emitter parked in an admission before the builder's scope
/// joins them.
struct FailOnDrop<'a>(Option<&'a ProgramBudget>);

impl Drop for FailOnDrop<'_> {
    fn drop(&mut self) {
        if let Some(budget) = self.0 {
            budget.fail();
        }
    }
}

/// Every slot of a block tree: the leaves, then each node level.
pub(super) struct Pipe {
    leaves: Vec<Slot>,
    levels: Vec<Vec<Slot>>,
    failed: AtomicBool,
}

/// The pool the builder emits programs on: its own host-only threads, or none
/// (the global pool).
#[cfg(feature = "parallel")]
type EmitPool = rayon::ThreadPool;
#[cfg(not(feature = "parallel"))]
type EmitPool = ();

/// Emits `items` in order, `window` at a time on `pool`, and sends each result
/// down `tx` as soon as its chunk is done. The receiver holds at most `window`
/// sent and not taken, and one chunk is in emission, so at most `2 × window`
/// programs exist ahead of the consumer. Stops after sending an error, or when
/// the receiver is gone.
fn emit_in_order<T: Send>(
    items: std::ops::Range<usize>,
    window: usize,
    pool: Option<&EmitPool>,
    emit: &(dyn Fn(usize) -> Result<T, String> + Sync),
    tx: &std::sync::mpsc::SyncSender<Result<T, String>>,
) {
    let window = window.max(1);
    let mut next = items.start;
    while next < items.end {
        let chunk = next..(next + window).min(items.end);
        next = chunk.end;
        #[cfg(feature = "parallel")]
        let emitted: Vec<Result<T, String>> = {
            use rayon::prelude::*;
            let run = || chunk.clone().into_par_iter().map(emit).collect::<Vec<_>>();
            match pool {
                Some(pool) => pool.install(run),
                None => run(),
            }
        };
        #[cfg(not(feature = "parallel"))]
        let emitted: Vec<Result<T, String>> = {
            let _ = pool;
            chunk.map(emit).collect()
        };
        for result in emitted {
            let failed = result.is_err();
            if tx.send(result).is_err() || failed {
                return;
            }
        }
    }
}

/// When the builder finished each stage, in seconds since it started.
pub(super) struct BuilderTimes {
    /// The last leaf's artifacts built.
    pub leaves: f64,
    /// Each node level's last artifacts built.
    pub levels: Vec<f64>,
    /// Summed over the builder's work: node emission, and artifact builds
    /// (leaves and nodes, each holding the card permit).
    pub emit: f64,
    pub build: f64,
    /// The program budget's summary, with one.
    pub budget: Option<String>,
}

/// The tree's program budget and every leaf program's estimated bytes (what
/// a leaf is admitted with before it is emitted).
pub(super) struct BudgetPlan {
    pub budget: Arc<ProgramBudget>,
    pub leaf_estimates: Vec<u64>,
}

impl Pipe {
    /// Empty slots for a tree of `leaves` leaves laid out as `levels`.
    pub(super) fn new(leaves: usize, levels: &[Level]) -> Self {
        Self {
            leaves: (0..leaves).map(|_| Slot::default()).collect(),
            levels: levels
                .iter()
                .map(|l| (0..l.arities.len()).map(|_| Slot::default()).collect())
                .collect(),
            failed: AtomicBool::new(false),
        }
    }

    /// Every slot filled from a tree derived ahead.
    pub(super) fn filled(tree: TreePrograms) -> Self {
        let slots = |level: Vec<(LfmProgram, LfmArtifacts)>| -> Vec<Slot> {
            level
                .into_iter()
                .map(|(program, artifacts)| Slot {
                    value: Mutex::new(Some((program, artifacts, None))),
                    ready: Condvar::new(),
                })
                .collect()
        };
        let mut levels: Vec<Vec<Slot>> = tree.into_iter().map(slots).collect();
        let leaves = if levels.is_empty() {
            Vec::new()
        } else {
            levels.remove(0)
        };
        Self {
            leaves,
            levels,
            failed: AtomicBool::new(false),
        }
    }

    /// Leaf `k`'s program and artifacts, waiting for them; its permit is
    /// claimed (the prover drops it with the program).
    pub(super) fn take_leaf(&self, k: usize, label: &str) -> Result<Taken, String> {
        let slot = self
            .leaves
            .get(k)
            .ok_or_else(|| format!("{label}: no leaf slot {k}"))?;
        self.take(slot, label)
    }

    /// Node `j` of node level `lv` (0 = the first above the leaves), waiting.
    pub(super) fn take_node(&self, lv: usize, j: usize, label: &str) -> Result<Taken, String> {
        let slot = self
            .levels
            .get(lv)
            .and_then(|level| level.get(j))
            .ok_or_else(|| format!("{label}: no node slot {lv}/{j}"))?;
        self.take(slot, label)
    }

    /// A slot's value, waiting for it; an error once the builder has failed.
    fn take(&self, slot: &Slot, label: &str) -> Result<Taken, String> {
        let mut value = slot.value.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(v) = value.take() {
                if let Some(permit) = &v.2 {
                    permit.claim();
                }
                return Ok(v);
            }
            if self.failed.load(Ordering::SeqCst) {
                return Err(format!("{label}: the tree's builder failed"));
            }
            value = slot
                .ready
                .wait_timeout(value, Duration::from_millis(200))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    fn put(
        &self,
        slot: &Slot,
        program: LfmProgram,
        artifacts: LfmArtifacts,
        permit: Option<Permit>,
    ) {
        *slot.value.lock().unwrap_or_else(|e| e.into_inner()) = Some((program, artifacts, permit));
        slot.ready.notify_all();
    }

    /// The pipeline's builder, on the card: each leaf's artifacts in leaf
    /// order, then each node level's programs (emitted in parallel on the host
    /// from the children's derived shapes) and their artifacts, each put in its
    /// slot as soon as it exists. A failure marks the pipe failed, so no prover
    /// waits forever. With `emit_threads` > 0 the node programs are emitted on a
    /// pool of that many host-only threads of the builder's own, not on the
    /// global pool the provers' host phases use.
    ///
    /// `leaf_programs` are the first leaves' programs. With a nonzero
    /// `emit_window` W they may be fewer than the leaves, and the builder emits
    /// the rest itself, in leaf order, on the same pool, at most `2 × W` ahead
    /// of its builds (`NOEPOCH_TREE_EMIT_WINDOW`): the same programs, emitted
    /// later, so they do not all sit in memory beside the base.
    ///
    /// With `node_emit_early` (`NOEPOCH_TREE_NODE_EMIT`, early by default) each
    /// node's program is emitted as soon as its children's artifacts exist,
    /// during the level below, instead of a level's programs together once the
    /// whole level below is built (`=level`). The build order is the same; the programs are the same (a
    /// node's program is a function of its children's derived shapes); only the
    /// moment each is emitted moves. At the median block the per-level emission
    /// leaves the card idle 8.9 s between level 0's last hold and level 1's
    /// first (BIG 481).
    ///
    /// Under a program budget (`LAMBDA_VM_TREE_PROGRAM_BUDGET`,
    /// [`super::program_budget`]) every program is admitted before it is
    /// emitted, in prove order: `leaf_programs` are the leaves admitted beside
    /// the base (each with its permit), the builder emits the rest on
    /// [`DEFERRED_LEAF_EMITTERS`] streaming threads as the budget admits them,
    /// and each node program is admitted before its emission. A permit travels
    /// in the slot and is dropped with its program by the prover.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_builder(
        &self,
        plan: &BlockTreePlan,
        leaf_programs: Vec<Emitted>,
        wrap_opts: &crate::ProofOptions,
        emit_threads: usize,
        emit_window: usize,
        node_emit_early: bool,
        budget: Option<BudgetPlan>,
    ) -> Result<BuilderTimes, String> {
        let run = std::panic::AssertUnwindSafe(|| {
            self.build(
                plan,
                leaf_programs,
                wrap_opts,
                emit_threads,
                emit_window,
                node_emit_early,
                budget,
            )
        });
        let outcome = std::panic::catch_unwind(run);
        if !matches!(outcome, Ok(Ok(_))) {
            self.failed.store(true, Ordering::SeqCst);
            for slot in self.leaves.iter().chain(self.levels.iter().flatten()) {
                slot.ready.notify_all();
            }
        }
        outcome.unwrap_or_else(|payload| std::panic::resume_unwind(payload))
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        plan: &BlockTreePlan,
        leaf_programs: Vec<Emitted>,
        wrap_opts: &crate::ProofOptions,
        emit_threads: usize,
        emit_window: usize,
        node_emit_early: bool,
        budget: Option<BudgetPlan>,
    ) -> Result<BuilderTimes, String> {
        let start = Instant::now();
        // ⚠ On the global pool, a prover that joins inside its `multi_prove`
        // can steal one of these ≈ 1 s emissions and hold the card idle until
        // it finishes (FAST 454: one leaf hold of 1.2 s at a third busy, in half
        // the runs, at the instant the level-1 emission starts).
        #[cfg(feature = "parallel")]
        let emit_pool = if emit_threads > 0 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(emit_threads)
                    .thread_name(|i| format!("tree-emit-{i}"))
                    .start_handler(|_| super::commit::mark_thread_host_only())
                    .build()
                    .map_err(|e| format!("the emission pool: {e}"))?,
            )
        } else {
            None
        };
        #[cfg(not(feature = "parallel"))]
        let emit_pool: Option<EmitPool> = None;
        let words = plan.child_layout().total();
        let mut build_secs = 0.0;
        let mut built = |program: &LfmProgram| -> Result<(LfmArtifacts, DerivedChild), String> {
            let t = Instant::now();
            let artifacts = super::program_census::build_artifacts_counted(
                program,
                wrap_opts,
                program.hasher(crate::hash_pin::BLOCK_HASHER),
            );
            let derived = DerivedChild::from_artifacts(&artifacts, wrap_opts, words)?;
            build_secs += t.elapsed().as_secs_f64();
            Ok((artifacts, derived))
        };
        let n = self.leaves.len();
        let given = leaf_programs.len();
        if given > n || (given < n && emit_window == 0 && budget.is_none()) {
            return Err(format!(
                "{given} leaf programs for {n} leaves with an emission window of {emit_window} \
                 and {} program budget",
                if budget.is_some() { "a" } else { "no" }
            ));
        }
        let levels = plan.levels();
        let pool = emit_pool.as_ref();
        // A program's place in prove order: the leaves, then each node level.
        let mut node_orders = Vec::with_capacity(levels.len());
        let mut next_order = n;
        for level in &levels {
            node_orders.push(next_order);
            next_order += level.arities.len();
        }
        let node_order = |lv: usize, j: usize| node_orders[lv] + j;
        let held_bytes = |p: &LfmProgram| super::block_tree::ProgramBytes::of(p).total() as u64;
        // A node program's estimate before it is emitted: the largest node seen,
        // or twice the largest leaf before the first (5.6–5.8 M instructions
        // against a leaf's 2.6 M at p90, BIG 662).
        let node_seen = AtomicU64::new(0);
        let node_estimate = || {
            let leaf = budget
                .as_ref()
                .and_then(|b| b.leaf_estimates.iter().max().copied())
                .unwrap_or(0);
            node_seen.load(Ordering::SeqCst).max(2 * leaf)
        };
        let emit_leaf = |k: usize| plan.leaf_program(k).map(|p| (p, None));
        let emit_leaf_on_pool = |k: usize| -> Result<LfmProgram, String> {
            #[cfg(feature = "parallel")]
            if let Some(pool) = pool {
                return pool.install(|| plan.leaf_program(k));
            }
            plan.leaf_program(k)
        };
        let emit_node = |kids: &[DerivedChild], top: bool| -> Result<LfmProgram, String> {
            #[cfg(feature = "parallel")]
            if let Some(pool) = pool {
                return pool.install(|| plan.node_program(kids, top));
            }
            plan.node_program(kids, top)
        };
        // A node's program, admitted first under a budget.
        let admitted_node =
            |order: usize, kids: &[DerivedChild], top: bool| -> Result<Emitted, String> {
                match &budget {
                    Some(b) => {
                        let mut permit = b.budget.acquire(order, node_estimate())?;
                        let program = emit_node(kids, top)?;
                        let bytes = held_bytes(&program);
                        node_seen.fetch_max(bytes, Ordering::SeqCst);
                        permit.emitted(bytes);
                        Ok((program, Some(permit)))
                    }
                    None => Ok((emit_node(kids, top)?, None)),
                }
            };
        let leaf_channel = if budget.is_some() { n } else { emit_window };
        let (leaf_tx, leaf_rx) =
            std::sync::mpsc::sync_channel::<Result<Emitted, String>>(leaf_channel.max(1));
        let (job_tx, job_rx) = std::sync::mpsc::channel::<(usize, usize, Vec<DerivedChild>)>();
        type Done = (usize, usize, Result<Emitted, String>, f64);
        let (done_tx, done_rx) = std::sync::mpsc::channel::<Done>();
        let job_rx = Mutex::new(job_rx);
        let (leaves_at, level_at, emit) = std::thread::scope(|scope| {
            // An error or a panic below fails the budget before this scope
            // joins its threads, so no emitter stays parked in an admission;
            // disarmed once every slot is put.
            let mut fail_budget = FailOnDrop(budget.as_ref().map(|b| &*b.budget));
            if given < n {
                match &budget {
                    Some(b) => {
                        let (b, emit_leaf_on_pool, held_bytes) =
                            (b, &emit_leaf_on_pool, &held_bytes);
                        scope.spawn(move || {
                            let emit = |k: usize| -> Result<Emitted, String> {
                                let estimate = b.leaf_estimates.get(k).copied().unwrap_or(0);
                                let mut permit = b.budget.acquire(k, estimate)?;
                                let program = emit_leaf_on_pool(k)?;
                                permit.emitted(held_bytes(&program));
                                Ok((program, Some(permit)))
                            };
                            super::program_budget::emit_ordered(
                                given..n,
                                DEFERRED_LEAF_EMITTERS,
                                &emit,
                                &mut |_, program| leaf_tx.send(program).is_ok(),
                            );
                        });
                    }
                    None => {
                        let emit_leaf = &emit_leaf;
                        scope.spawn(move || {
                            emit_in_order(given..n, emit_window, pool, emit_leaf, &leaf_tx)
                        });
                    }
                }
            } else {
                drop(leaf_tx);
            }
            // The early node emitters: each takes the next completed group and
            // emits its node on the builder's pool.
            if node_emit_early {
                for _ in 0..emit_threads.max(1) {
                    let (job_rx, done_tx, admitted_node, levels, node_order) = (
                        &job_rx,
                        done_tx.clone(),
                        &admitted_node,
                        &levels,
                        &node_order,
                    );
                    scope.spawn(move || {
                        loop {
                            let job = job_rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                            let Ok((lv, j, kids)) = job else {
                                return;
                            };
                            let t = Instant::now();
                            let program =
                                admitted_node(node_order(lv, j), &kids, lv + 1 == levels.len());
                            let secs = t.elapsed().as_secs_f64();
                            if done_tx.send((lv, j, program, secs)).is_err() {
                                return;
                            }
                        }
                    });
                }
            }
            drop(done_tx);
            let mut emit = 0.0;
            // Early mode: each node level's children in arrival order, cut into
            // their groups as they complete; the emitted programs as they come
            // back.
            let mut groupers: Vec<Grouper> = levels
                .iter()
                .map(|l| Grouper::new(l.arities.clone()))
                .collect();
            let mut ready: Vec<Vec<Option<Result<Emitted, String>>>> = levels
                .iter()
                .map(|l| (0..l.arities.len()).map(|_| None).collect())
                .collect();

            let mut given = leaf_programs.into_iter();
            let mut children = Vec::with_capacity(n);
            for k in 0..n {
                let (program, permit) = match given.next() {
                    Some(program) => program,
                    None => leaf_rx
                        .recv()
                        .map_err(|_| format!("leaf {k}: the leaf emitter stopped early"))??,
                };
                let (artifacts, derived) = built(&program)?;
                self.put(&self.leaves[k], program, artifacts, permit);
                if node_emit_early {
                    submit(&mut groupers, &job_tx, 0, derived)?;
                } else {
                    children.push(derived);
                }
            }
            let leaves_at = start.elapsed().as_secs_f64();
            let mut level_at = Vec::new();
            for (lv, arities) in levels.iter().enumerate() {
                let top = lv + 1 == levels.len();
                let nodes = arities.arities.len();
                let mut programs: Vec<Option<Emitted>> = (0..nodes).map(|_| None).collect();
                if node_emit_early {
                    if !groupers[lv].is_complete() {
                        return Err(format!(
                            "level {}: arities do not cover the children",
                            lv + 1
                        ));
                    }
                } else {
                    let mut rest = std::mem::take(&mut children).into_iter();
                    let groups: Vec<Vec<DerivedChild>> = arities
                        .arities
                        .iter()
                        .map(|&a| rest.by_ref().take(a).collect())
                        .collect();
                    if groups
                        .iter()
                        .zip(&arities.arities)
                        .any(|(g, &a)| g.len() != a)
                        || rest.next().is_some()
                    {
                        return Err(format!(
                            "level {}: arities do not cover the children",
                            lv + 1
                        ));
                    }
                    // Under a budget, the level's nodes are admitted in order
                    // here, on the builder's own thread, before they are
                    // emitted together (never parked inside the pool).
                    let mut permits: Vec<Option<Permit>> = match &budget {
                        Some(b) => (0..nodes)
                            .map(|j| {
                                b.budget
                                    .acquire(node_order(lv, j), node_estimate())
                                    .map(Some)
                            })
                            .collect::<Result<_, String>>()?,
                        None => (0..nodes).map(|_| None).collect(),
                    };
                    let t = Instant::now();
                    #[cfg(feature = "parallel")]
                    let emitted: Vec<LfmProgram> = {
                        use rayon::prelude::*;
                        let emit = || {
                            groups
                                .par_iter()
                                .map(|kids| plan.node_program(kids, top))
                                .collect::<Result<_, String>>()
                        };
                        match pool {
                            Some(pool) => pool.install(emit),
                            None => emit(),
                        }?
                    };
                    #[cfg(not(feature = "parallel"))]
                    let emitted: Vec<LfmProgram> = groups
                        .iter()
                        .map(|kids| plan.node_program(kids, top))
                        .collect::<Result<_, String>>()?;
                    emit += t.elapsed().as_secs_f64();
                    programs = emitted
                        .into_iter()
                        .zip(permits.iter_mut())
                        .map(|(program, permit)| {
                            let mut permit = permit.take();
                            if let Some(p) = permit.as_mut() {
                                let bytes = held_bytes(&program);
                                node_seen.fetch_max(bytes, Ordering::SeqCst);
                                p.emitted(bytes);
                            }
                            Some((program, permit))
                        })
                        .collect();
                }
                for j in 0..nodes {
                    let program = match programs[j].take() {
                        Some(program) => program,
                        None => {
                            while ready[lv][j].is_none() {
                                let (l, jj, program, secs) = done_rx.recv().map_err(|_| {
                                    format!("level {} node {j}: the node emitters stopped", lv + 1)
                                })?;
                                emit += secs;
                                ready[l][jj] = Some(program);
                            }
                            ready[lv][j].take().ok_or_else(|| {
                                format!("level {} node {j}: its program never arrived", lv + 1)
                            })??
                        }
                    };
                    let (program, permit) = program;
                    let (artifacts, derived) = built(&program)?;
                    self.put(&self.levels[lv][j], program, artifacts, permit);
                    if node_emit_early && !top {
                        submit(&mut groupers, &job_tx, lv + 1, derived)?;
                    } else {
                        children.push(derived);
                    }
                }
                level_at.push(start.elapsed().as_secs_f64());
            }
            drop(job_tx);
            fail_budget.0 = None;
            Ok::<_, String>((leaves_at, level_at, emit))
        })?;
        Ok(BuilderTimes {
            leaves: leaves_at,
            levels: level_at,
            emit,
            build: build_secs,
            budget: budget.map(|b| b.budget.summary()),
        })
    }
}

/// Hands `child` to node level `lv`'s grouper and, once it completes a group,
/// that group to the early node emitters.
fn submit(
    groupers: &mut [Grouper],
    jobs: &std::sync::mpsc::Sender<(usize, usize, Vec<DerivedChild>)>,
    lv: usize,
    child: DerivedChild,
) -> Result<(), String> {
    if let Some((j, kids)) = groupers[lv].push(child) {
        jobs.send((lv, j, kids))
            .map_err(|_| format!("level {}: the node emitters stopped", lv + 1))?;
    }
    Ok(())
}

/// One node level's children in arrival order, cut into the level's groups as
/// each group completes: what the early node emission submits.
struct Grouper<T = DerivedChild> {
    arities: Vec<usize>,
    next: usize,
    pending: Vec<T>,
}

impl<T> Grouper<T> {
    fn new(arities: Vec<usize>) -> Self {
        Self {
            arities,
            next: 0,
            pending: Vec::new(),
        }
    }

    /// Adds the next child; returns a group (its index and children) once the
    /// child completes it.
    fn push(&mut self, child: T) -> Option<(usize, Vec<T>)> {
        self.pending.push(child);
        let want = *self.arities.get(self.next)?;
        (self.pending.len() == want).then(|| {
            self.next += 1;
            (self.next - 1, std::mem::take(&mut self.pending))
        })
    }

    /// Every group submitted, no child left over.
    fn is_complete(&self) -> bool {
        self.next == self.arities.len() && self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// The deferred leaf emitter hands its items over in order and never runs
    /// more than two windows ahead of a slow consumer.
    #[test]
    fn the_leaf_emitter_keeps_its_order_and_its_window() {
        let window = 3;
        let emitted = AtomicUsize::new(0);
        let emit = |k: usize| -> Result<usize, String> {
            emitted.fetch_add(1, Ordering::SeqCst);
            Ok(k * 10)
        };
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<usize, String>>(window);
        let mut worst = 0;
        std::thread::scope(|scope| {
            scope.spawn(|| emit_in_order(5..25, window, None, &emit, &tx));
            for (received, k) in (5..25).enumerate() {
                std::thread::sleep(Duration::from_millis(5));
                worst = worst.max(emitted.load(Ordering::SeqCst) - received);
                assert_eq!(rx.recv().expect("the emitter sends every item"), Ok(k * 10));
            }
        });
        assert_eq!(
            emitted.load(Ordering::SeqCst),
            20,
            "every item emitted once"
        );
        assert!(
            worst <= 2 * window,
            "{worst} items ahead of the consumer, window {window}"
        );
    }

    /// An emission error is the last thing the emitter sends, after the items
    /// before it in order; the consumer then sees the channel close.
    #[test]
    fn the_leaf_emitter_stops_at_an_error() {
        let emit = |k: usize| -> Result<usize, String> {
            if k == 7 {
                Err(format!("leaf {k} does not emit"))
            } else {
                Ok(k)
            }
        };
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<usize, String>>(2);
        std::thread::scope(|scope| {
            scope.spawn(move || emit_in_order(5..20, 2, None, &emit, &tx));
            assert_eq!(rx.recv().unwrap(), Ok(5));
            assert_eq!(rx.recv().unwrap(), Ok(6));
            assert_eq!(rx.recv().unwrap(), Err("leaf 7 does not emit".to_string()));
            assert!(rx.recv().is_err(), "nothing after the error");
        });
    }

    /// The early emission's grouper cuts a level's children into its groups in
    /// order, a group as soon as its last child arrives, and is complete only
    /// when every group is out and no child is left over.
    #[test]
    fn the_grouper_cuts_complete_groups_in_order() {
        let mut g = Grouper::<usize>::new(vec![4, 4, 1]);
        let mut cut = Vec::new();
        for k in 0..9 {
            assert!(!g.is_complete(), "child {k}: not complete yet");
            if let Some((j, kids)) = g.push(k) {
                cut.push((j, kids));
            }
        }
        assert!(g.is_complete());
        assert_eq!(
            cut,
            vec![(0, vec![0, 1, 2, 3]), (1, vec![4, 5, 6, 7]), (2, vec![8])]
        );
        let mut over = Grouper::<usize>::new(vec![2]);
        assert_eq!(over.push(0), None);
        assert_eq!(over.push(1), Some((0, vec![0, 1])));
        assert_eq!(over.push(2), None, "a child past the last group");
        assert!(!over.is_complete(), "a leftover child is not complete");
    }
}
