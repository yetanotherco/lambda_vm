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
//! behind; a builder failure fails every waiter.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use super::LfmArtifacts;
use super::block_plan::{BlockTreePlan, TreePrograms};
use super::compiler::LfmProgram;
use super::per_table_aggregator::{DerivedChild, Level};

/// One program and its artifacts, once its builder has put them.
#[derive(Default)]
struct Slot {
    value: Mutex<Option<(LfmProgram, LfmArtifacts)>>,
    ready: Condvar,
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
}

/// `NOEPOCH_TREE_BUILDERS=N`: artifact builds in flight at once in the
/// pipeline's builder. Unset or 1 (the default): the serial builder. N ≥ 2: N
/// workers, each on a rayon pool of its own ([`tree_builder_threads`]), handed
/// the same targets in the same order; the node groupers still see the children
/// in index order ([`build_in_parallel`]). Under the shared VRAM gate a build
/// takes bytes, not the card, so builds overlap the leaf proofs and each other;
/// with the gate off they queue on the card permit as before.
fn tree_builders() -> usize {
    std::env::var("NOEPOCH_TREE_BUILDERS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(1)
        .max(1)
}

/// `NOEPOCH_TREE_BUILDER_THREADS` (default 8): each parallel builder worker's
/// pool size. One caller per pool: a pool shared by several long installing
/// callers runs their jobs nested and inverts their completion.
fn tree_builder_threads() -> usize {
    std::env::var("NOEPOCH_TREE_BUILDER_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&t| t > 0)
        .unwrap_or(8)
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
                .map(|p| Slot {
                    value: Mutex::new(Some(p)),
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

    /// Leaf `k`'s program and artifacts, waiting for them.
    pub(super) fn take_leaf(&self, k: usize, label: &str) -> (LfmProgram, LfmArtifacts) {
        self.take(&self.leaves[k], label)
    }

    /// Node `j` of node level `lv` (0 = the first above the leaves), waiting.
    pub(super) fn take_node(&self, lv: usize, j: usize, label: &str) -> (LfmProgram, LfmArtifacts) {
        self.take(&self.levels[lv][j], label)
    }

    fn take(&self, slot: &Slot, label: &str) -> (LfmProgram, LfmArtifacts) {
        let mut value = slot.value.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(v) = value.take() {
                return v;
            }
            assert!(
                !self.failed.load(Ordering::SeqCst),
                "{label}: the tree's builder failed"
            );
            value = slot
                .ready
                .wait_timeout(value, Duration::from_millis(200))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    fn put(&self, slot: &Slot, program: LfmProgram, artifacts: LfmArtifacts) {
        *slot.value.lock().unwrap_or_else(|e| e.into_inner()) = Some((program, artifacts));
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
    pub(super) fn run_builder(
        &self,
        plan: &BlockTreePlan,
        leaf_programs: Vec<LfmProgram>,
        wrap_opts: &crate::ProofOptions,
        emit_threads: usize,
        emit_window: usize,
        node_emit_early: bool,
    ) -> Result<BuilderTimes, String> {
        let run = std::panic::AssertUnwindSafe(|| {
            self.build(
                plan,
                leaf_programs,
                wrap_opts,
                emit_threads,
                emit_window,
                node_emit_early,
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

    fn build(
        &self,
        plan: &BlockTreePlan,
        leaf_programs: Vec<LfmProgram>,
        wrap_opts: &crate::ProofOptions,
        emit_threads: usize,
        emit_window: usize,
        node_emit_early: bool,
    ) -> Result<BuilderTimes, String> {
        let builders = tree_builders();
        if builders > 1 {
            return self.build_parallel(
                plan,
                leaf_programs,
                wrap_opts,
                emit_threads,
                emit_window,
                node_emit_early,
                builders,
            );
        }
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
                crate::hash_pin::BLOCK_HASHER,
            );
            let derived = DerivedChild::from_artifacts(&artifacts, wrap_opts, words)?;
            build_secs += t.elapsed().as_secs_f64();
            Ok((artifacts, derived))
        };
        let n = self.leaves.len();
        let given = leaf_programs.len();
        if given > n || (given < n && emit_window == 0) {
            return Err(format!(
                "{given} leaf programs for {n} leaves with an emission window of {emit_window}"
            ));
        }
        let levels = plan.levels();
        let pool = emit_pool.as_ref();
        let emit_leaf = |k: usize| plan.leaf_program(k);
        let emit_node = |kids: &[DerivedChild], top: bool| -> Result<LfmProgram, String> {
            #[cfg(feature = "parallel")]
            if let Some(pool) = pool {
                return pool.install(|| plan.node_program(kids, top));
            }
            plan.node_program(kids, top)
        };
        let (leaf_tx, leaf_rx) =
            std::sync::mpsc::sync_channel::<Result<LfmProgram, String>>(emit_window.max(1));
        let (job_tx, job_rx) = std::sync::mpsc::channel::<(usize, usize, Vec<DerivedChild>)>();
        let (done_tx, done_rx) =
            std::sync::mpsc::channel::<(usize, usize, Result<LfmProgram, String>, f64)>();
        let job_rx = Mutex::new(job_rx);
        let (leaves_at, level_at, emit) = std::thread::scope(|scope| {
            if given < n {
                let emit_leaf = &emit_leaf;
                scope
                    .spawn(move || emit_in_order(given..n, emit_window, pool, emit_leaf, &leaf_tx));
            } else {
                drop(leaf_tx);
            }
            // The early node emitters: each takes the next completed group and
            // emits its node on the builder's pool.
            if node_emit_early {
                for _ in 0..emit_threads.max(1) {
                    let (job_rx, done_tx, emit_node, levels) =
                        (&job_rx, done_tx.clone(), &emit_node, &levels);
                    scope.spawn(move || {
                        loop {
                            let job = job_rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                            let Ok((lv, j, kids)) = job else {
                                return;
                            };
                            let t = Instant::now();
                            let program = emit_node(&kids, lv + 1 == levels.len());
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
            let mut ready: Vec<Vec<Option<Result<LfmProgram, String>>>> = levels
                .iter()
                .map(|l| (0..l.arities.len()).map(|_| None).collect())
                .collect();

            let mut given = leaf_programs.into_iter();
            let mut children = Vec::with_capacity(n);
            for k in 0..n {
                let program = match given.next() {
                    Some(program) => program,
                    None => leaf_rx
                        .recv()
                        .map_err(|_| format!("leaf {k}: the leaf emitter stopped early"))??,
                };
                let (artifacts, derived) = built(&program)?;
                self.put(&self.leaves[k], program, artifacts);
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
                let mut programs: Vec<Option<LfmProgram>> = (0..nodes).map(|_| None).collect();
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
                    programs = emitted.into_iter().map(Some).collect();
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
                            ready[lv][j].take().expect("filled above")?
                        }
                    };
                    let (artifacts, derived) = built(&program)?;
                    self.put(&self.levels[lv][j], program, artifacts);
                    if node_emit_early && !top {
                        submit(&mut groupers, &job_tx, lv + 1, derived)?;
                    } else {
                        children.push(derived);
                    }
                }
                level_at.push(start.elapsed().as_secs_f64());
            }
            drop(job_tx);
            Ok::<_, String>((leaves_at, level_at, emit))
        })?;
        Ok(BuilderTimes {
            leaves: leaves_at,
            levels: level_at,
            emit,
            build: build_secs,
        })
    }
}

impl Pipe {
    /// [`Pipe::build`] with `builders` artifact builds in flight
    /// (`NOEPOCH_TREE_BUILDERS`): the same targets in the same order, each
    /// worker building under a rayon pool of its own, each slot put as soon as
    /// its artifacts exist, and the node programs emitted from the children in
    /// index order ([`build_in_parallel`]).
    #[allow(clippy::too_many_arguments)]
    fn build_parallel(
        &self,
        plan: &BlockTreePlan,
        leaf_programs: Vec<LfmProgram>,
        wrap_opts: &crate::ProofOptions,
        emit_threads: usize,
        emit_window: usize,
        node_emit_early: bool,
        builders: usize,
    ) -> Result<BuilderTimes, String> {
        let start = Instant::now();
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
        // One pool per worker, device commits allowed: no build runs on the
        // provers' global pool, and no pool has more than one caller.
        #[cfg(feature = "parallel")]
        let build_pools: Vec<rayon::ThreadPool> = (0..builders)
            .map(|w| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(tree_builder_threads())
                    .thread_name(move |i| format!("tree-build-{w}-{i}"))
                    .build()
                    .map_err(|e| format!("builder {w}'s pool: {e}"))
            })
            .collect::<Result<_, String>>()?;
        let words = plan.child_layout().total();
        let n = self.leaves.len();
        let given = leaf_programs.len();
        if given > n || (given < n && emit_window == 0) {
            return Err(format!(
                "{given} leaf programs for {n} leaves with an emission window of {emit_window}"
            ));
        }
        let levels = plan.levels();
        let arities: Vec<Vec<usize>> = levels.iter().map(|l| l.arities.clone()).collect();
        let pool = emit_pool.as_ref();
        let emit_leaf = |k: usize| plan.leaf_program(k);
        let (leaf_tx, leaf_rx) =
            std::sync::mpsc::sync_channel::<Result<LfmProgram, String>>(emit_window.max(1));
        // The leaves' programs in order: the given ones, then the deferred
        // emitter's. Taken only under the scheduler's job lock, so the k-th take
        // is leaf k. Dropping the receiver releases an emitter blocked on a full
        // window when the build stops early.
        let leaf_source = Mutex::new((leaf_programs.into_iter(), Some(leaf_rx)));
        let next_leaf = |k: usize, stop: &AtomicBool| -> Result<LfmProgram, String> {
            let mut src = leaf_source.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(program) = src.0.next() {
                return Ok(program);
            }
            loop {
                if stop.load(Ordering::SeqCst) {
                    src.1 = None;
                    return Err(format!("leaf {k}: the build stopped"));
                }
                let rx = src
                    .1
                    .as_ref()
                    .ok_or_else(|| format!("leaf {k}: the leaf emitter is gone"))?;
                match rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(program) => return program,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        return Err(format!("leaf {k}: the leaf emitter stopped early"));
                    }
                }
            }
        };
        let build =
            |w: usize, program: &LfmProgram| -> Result<(LfmArtifacts, DerivedChild, f64), String> {
                let run = || {
                    let t = Instant::now();
                    let artifacts = super::program_census::build_artifacts_counted(
                        program,
                        wrap_opts,
                        crate::hash_pin::BLOCK_HASHER,
                    );
                    let derived = DerivedChild::from_artifacts(&artifacts, wrap_opts, words)?;
                    Ok((artifacts, derived, t.elapsed().as_secs_f64()))
                };
                #[cfg(feature = "parallel")]
                return build_pools[w].install(run);
                #[cfg(not(feature = "parallel"))]
                {
                    let _ = w;
                    run()
                }
            };
        let emit_node = |kids: &[DerivedChild], top: bool| -> Result<LfmProgram, String> {
            #[cfg(feature = "parallel")]
            if let Some(pool) = pool {
                return pool.install(|| plan.node_program(kids, top));
            }
            plan.node_program(kids, top)
        };
        let outcome = std::thread::scope(|scope| {
            if given < n {
                let emit_leaf = &emit_leaf;
                scope
                    .spawn(move || emit_in_order(given..n, emit_window, pool, emit_leaf, &leaf_tx));
            } else {
                drop(leaf_tx);
            }
            let outcome = build_in_parallel(
                n,
                &arities,
                node_emit_early,
                builders,
                emit_threads.max(1),
                &next_leaf,
                &build,
                &emit_node,
                &mut |target, program, artifacts| match target {
                    Target::Leaf(k) => self.put(&self.leaves[k], program, artifacts),
                    Target::Node(lv, j) => self.put(&self.levels[lv][j], program, artifacts),
                },
                start,
            );
            // An emitter blocked on a full window is released before the scope
            // joins it.
            leaf_source.lock().unwrap_or_else(|e| e.into_inner()).1 = None;
            outcome
        })?;
        Ok(outcome)
    }
}

/// A build target of the parallel builder, ordered as the serial builder builds
/// them: every leaf in index order, then each node level by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Target {
    Leaf(usize),
    Node(usize, usize),
}

/// What a parallel builder worker or node emitter hands the coordinator.
enum Done<P, A, D> {
    Built(Target, P, A, D, f64),
    Failed(String),
}

/// The jobs the parallel builder hands out: leaves by index from the leaf
/// source, then node programs as their emitters produce them, lowest target
/// first.
struct Jobs<P> {
    next_leaf: usize,
    nodes: std::collections::BTreeMap<(usize, usize), P>,
    handed: usize,
}

/// A parallel builder worker's build: worker index and program to artifacts,
/// derived shape and seconds.
type BuildFn<'a, P, A, D> = dyn Fn(usize, &P) -> Result<(A, D, f64), String> + Sync + 'a;

/// A node's program from its children's derived shapes (and whether it is the
/// top).
type EmitFn<'a, P, D> = dyn Fn(&[D], bool) -> Result<P, String> + Sync + 'a;

/// The parallel builder's scheduler, generic over what a program, its
/// artifacts and its derived shape are (so it is testable without a card).
///
/// - `workers` threads take targets in the serial builder's order: leaf k is
///   the k-th call of `next_leaf` (made under the job lock), then node targets
///   as `emit_node` produces their programs, lowest (level, index) first.
/// - Each built target is `put` at once, from this (the caller's) thread.
/// - Each level's derived children pass through a reorder buffer, so node
///   level `lv`'s groups are cut from its children in index order whatever
///   order they were built in: with `early`, as each group completes (emitted
///   on `emitters` threads); otherwise once the whole level below is in. A
///   node's program is a function of its children in order, so every program
///   is the serial builder's.
///
/// Returns when the last leaf was built, when each node level's last node was,
/// and the summed emission and build seconds.
#[allow(clippy::too_many_arguments)]
fn build_in_parallel<P: Send, A: Send, D: Send>(
    n: usize,
    arities: &[Vec<usize>],
    early: bool,
    workers: usize,
    emitters: usize,
    next_leaf: &(dyn Fn(usize, &AtomicBool) -> Result<P, String> + Sync),
    build: &BuildFn<'_, P, A, D>,
    emit_node: &EmitFn<'_, P, D>,
    put: &mut dyn FnMut(Target, P, A),
    start: Instant,
) -> Result<BuilderTimes, String> {
    let levels = arities.len();
    let total = n + arities.iter().map(Vec::len).sum::<usize>();
    let jobs = Mutex::new(Jobs::<P> {
        next_leaf: 0,
        nodes: std::collections::BTreeMap::new(),
        handed: 0,
    });
    let more = Condvar::new();
    let stop = AtomicBool::new(false);
    let emit_nanos = std::sync::atomic::AtomicU64::new(0);
    let (done_tx, done_rx) = std::sync::mpsc::channel::<Done<P, A, D>>();
    let (group_tx, group_rx) = std::sync::mpsc::channel::<(usize, usize, Vec<D>)>();
    let group_rx = Mutex::new(group_rx);
    let add_node = |lv: usize, j: usize, program: P| {
        jobs.lock()
            .unwrap_or_else(|e| e.into_inner())
            .nodes
            .insert((lv, j), program);
        more.notify_all();
    };
    std::thread::scope(|scope| {
        for w in 0..workers {
            let (jobs, more, stop, done_tx) = (&jobs, &more, &stop, done_tx.clone());
            scope.spawn(move || {
                loop {
                    let job = {
                        let mut q = jobs.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if stop.load(Ordering::SeqCst) {
                                break None;
                            }
                            if q.next_leaf < n {
                                let k = q.next_leaf;
                                q.next_leaf += 1;
                                q.handed += 1;
                                break Some((Target::Leaf(k), next_leaf(k, stop)));
                            }
                            if let Some(key) = q.nodes.keys().next().copied() {
                                let program = q.nodes.remove(&key).expect("the key was just read");
                                q.handed += 1;
                                break Some((Target::Node(key.0, key.1), Ok(program)));
                            }
                            if q.handed == total {
                                break None;
                            }
                            q = more
                                .wait_timeout(q, Duration::from_millis(50))
                                .unwrap_or_else(|e| e.into_inner())
                                .0;
                        }
                    };
                    let Some((target, program)) = job else {
                        return;
                    };
                    let done = match program.and_then(|p| build(w, &p).map(|b| (p, b))) {
                        Ok((p, (artifacts, derived, secs))) => {
                            Done::Built(target, p, artifacts, derived, secs)
                        }
                        Err(e) => Done::Failed(e),
                    };
                    if done_tx.send(done).is_err() {
                        return;
                    }
                }
            });
        }
        if early {
            for _ in 0..emitters {
                let (group_rx, done_tx, add_node, emit_nanos) =
                    (&group_rx, done_tx.clone(), &add_node, &emit_nanos);
                scope.spawn(move || {
                    loop {
                        let group = group_rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                        let Ok((lv, j, kids)) = group else {
                            return;
                        };
                        let t = Instant::now();
                        let program = emit_node(&kids, lv + 1 == levels);
                        emit_nanos.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        match program {
                            Ok(p) => add_node(lv, j, p),
                            Err(e) => {
                                let _ = done_tx.send(Done::Failed(e));
                                return;
                            }
                        }
                    }
                });
            }
        }
        drop(done_tx);
        let coordinate = || -> Result<BuilderTimes, String> {
            let group_tx = group_tx;
            let mut build_secs = 0.0;
            // Node level `lv`'s children: built ones waiting for their turn, and
            // the next index the level takes.
            let mut waiting: Vec<std::collections::BTreeMap<usize, D>> =
                (0..levels).map(|_| Default::default()).collect();
            let mut next = vec![0usize; levels];
            let mut collected: Vec<Vec<D>> = (0..levels).map(|_| Vec::new()).collect();
            let mut groupers: Vec<Grouper<D>> =
                arities.iter().map(|a| Grouper::new(a.clone())).collect();
            let children: Vec<usize> = arities.iter().map(|a| a.iter().sum()).collect();
            let mut leaves_left = n;
            let mut nodes_left: Vec<usize> = arities.iter().map(Vec::len).collect();
            let mut leaves_at = if n == 0 { 0.0 } else { f64::NAN };
            let mut level_at = vec![f64::NAN; levels];
            let mut received = 0;
            while received < total {
                let done = done_rx
                    .recv()
                    .map_err(|_| "the tree's builders stopped".to_string())?;
                let (target, program, artifacts, derived, secs) = match done {
                    Done::Built(t, p, a, d, s) => (t, p, a, d, s),
                    Done::Failed(e) => return Err(e),
                };
                received += 1;
                build_secs += secs;
                put(target, program, artifacts);
                // The level the derived child feeds: leaves feed node level 0,
                // node level lv feeds lv + 1, the top feeds nothing.
                let feeds = match target {
                    Target::Leaf(_) => {
                        leaves_left -= 1;
                        if leaves_left == 0 {
                            leaves_at = start.elapsed().as_secs_f64();
                        }
                        Some(0)
                    }
                    Target::Node(lv, _) => {
                        nodes_left[lv] -= 1;
                        if nodes_left[lv] == 0 {
                            level_at[lv] = start.elapsed().as_secs_f64();
                        }
                        (lv + 1 < levels).then_some(lv + 1)
                    }
                };
                let Some(lv) = feeds else { continue };
                if lv >= levels {
                    continue;
                }
                let index = match target {
                    Target::Leaf(k) => k,
                    Target::Node(_, j) => j,
                };
                waiting[lv].insert(index, derived);
                while let Some(child) = waiting[lv].remove(&next[lv]) {
                    next[lv] += 1;
                    if early {
                        if let Some((j, kids)) = groupers[lv].push(child) {
                            group_tx.send((lv, j, kids)).map_err(|_| {
                                format!("level {}: the node emitters stopped", lv + 1)
                            })?;
                        }
                    } else {
                        collected[lv].push(child);
                        if collected[lv].len() == children[lv] {
                            let mut rest = std::mem::take(&mut collected[lv]).into_iter();
                            let t = Instant::now();
                            for (j, &a) in arities[lv].iter().enumerate() {
                                let kids: Vec<D> = rest.by_ref().take(a).collect();
                                add_node(lv, j, emit_node(&kids, lv + 1 == levels)?);
                            }
                            emit_nanos.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }
                    }
                }
            }
            if early && groupers.iter().any(|g| !g.is_complete()) {
                return Err("the arities do not cover the children".to_string());
            }
            Ok(BuilderTimes {
                leaves: leaves_at,
                levels: level_at,
                emit: emit_nanos.load(Ordering::Relaxed) as f64 * 1e-9,
                build: build_secs,
            })
        };
        let outcome = coordinate();
        // Every worker and emitter ends: on success there is nothing left to
        // hand out; on a failure they stop at their next look.
        stop.store(true, Ordering::SeqCst);
        more.notify_all();
        outcome
    })
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

    /// The parallel scheduler's tree for `n` leaves under `arities`, with
    /// builds that finish out of order (a target's build sleeps longer the
    /// lower its index): what each target was put with, and the outcome.
    #[allow(clippy::type_complexity)]
    fn parallel_tree(
        n: usize,
        arities: &[Vec<usize>],
        early: bool,
        workers: usize,
        fail_at: Option<usize>,
    ) -> (Vec<(Target, String)>, Result<BuilderTimes, String>) {
        let next_leaf =
            |k: usize, _: &AtomicBool| -> Result<String, String> { Ok(format!("L{k}")) };
        let build = |_: usize, p: &String| -> Result<(String, String, f64), String> {
            let index: usize = p
                .trim_start_matches('L')
                .split(|c: char| !c.is_ascii_digit())
                .next()
                .and_then(|d| d.parse().ok())
                .unwrap_or(0);
            if fail_at.is_some_and(|f| p == &format!("L{f}")) {
                return Err(format!("{p} does not build"));
            }
            std::thread::sleep(Duration::from_millis(
                (8usize.saturating_sub(index % 8) * 2) as u64,
            ));
            Ok((format!("a:{p}"), p.clone(), 0.001))
        };
        let emit_node = |kids: &[String], top: bool| -> Result<String, String> {
            Ok(format!(
                "{}({})",
                if top { "T" } else { "N" },
                kids.join(",")
            ))
        };
        let puts = Mutex::new(Vec::new());
        let outcome = build_in_parallel(
            n,
            arities,
            early,
            workers,
            2,
            &next_leaf,
            &build,
            &emit_node,
            &mut |t, p: String, a: String| {
                assert_eq!(
                    a,
                    format!("a:{p}"),
                    "{t:?}: the artifacts of its own program"
                );
                puts.lock().unwrap().push((t, p));
            },
            Instant::now(),
        );
        (puts.into_inner().unwrap(), outcome)
    }

    /// ★ Under builds that complete out of order, the parallel builder puts every
    /// target once, with the program the serial builder would emit: each node's
    /// children in index order, in both emission modes and for 1–4 workers.
    #[test]
    fn the_parallel_builder_emits_the_serial_builders_programs() {
        let arities = vec![vec![4, 4, 4, 1], vec![2, 2], vec![2]];
        for early in [true, false] {
            for workers in 1..=4 {
                let (puts, outcome) = parallel_tree(13, &arities, early, workers, None);
                let times = outcome.unwrap();
                assert!(times.leaves.is_finite() && times.levels.iter().all(|t| t.is_finite()));
                let mut puts = puts;
                puts.sort();
                let leaf = |k: usize| format!("L{k}");
                let l1: Vec<String> = [(0, 4), (4, 4), (8, 4), (12, 1)]
                    .iter()
                    .map(|&(s, a)| {
                        format!("N({})", (s..s + a).map(leaf).collect::<Vec<_>>().join(","))
                    })
                    .collect();
                let l2 = [
                    format!("N({},{})", l1[0], l1[1]),
                    format!("N({},{})", l1[2], l1[3]),
                ];
                let top = format!("T({},{})", l2[0], l2[1]);
                let mut want: Vec<(Target, String)> =
                    (0..13).map(|k| (Target::Leaf(k), leaf(k))).collect();
                want.extend(
                    l1.iter()
                        .enumerate()
                        .map(|(j, p)| (Target::Node(0, j), p.clone())),
                );
                want.extend(
                    l2.iter()
                        .enumerate()
                        .map(|(j, p)| (Target::Node(1, j), p.clone())),
                );
                want.push((Target::Node(2, 0), top));
                assert_eq!(puts, want, "early {early}, {workers} workers");
            }
        }
    }

    /// A build that fails fails the whole tree, promptly and without a hang.
    #[test]
    fn a_failing_build_stops_the_parallel_builder() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let (_, outcome) = parallel_tree(13, &[vec![4, 4, 4, 1], vec![4]], true, 3, Some(5));
            let _ = tx.send(outcome.err());
        });
        let err = rx
            .recv_timeout(Duration::from_secs(20))
            .expect("the parallel builder hung after a failed build");
        assert_eq!(err.as_deref(), Some("L5 does not build"));
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
