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
    /// `leaf_programs` are the first leaves' programs. With an `emit_window` W
    /// > 0 they may be fewer than the leaves, and the builder emits the rest
    /// itself, in leaf order, on the same pool, at most `2 × W` ahead of its
    /// builds (`NOEPOCH_TREE_EMIT_WINDOW`): the same programs, emitted later, so
    /// they do not all sit in memory beside the base.
    pub(super) fn run_builder(
        &self,
        plan: &BlockTreePlan,
        leaf_programs: Vec<LfmProgram>,
        wrap_opts: &crate::ProofOptions,
        emit_threads: usize,
        emit_window: usize,
    ) -> Result<BuilderTimes, String> {
        let run = std::panic::AssertUnwindSafe(|| {
            self.build(plan, leaf_programs, wrap_opts, emit_threads, emit_window)
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
        let emit_pool: Option<EmitPool> = {
            let _ = emit_threads;
            None
        };
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
        let (tx, rx) =
            std::sync::mpsc::sync_channel::<Result<LfmProgram, String>>(emit_window.max(1));
        let emit_leaf = |k: usize| plan.leaf_program(k);
        let mut children = std::thread::scope(|scope| -> Result<Vec<DerivedChild>, String> {
            if given < n {
                let pool = emit_pool.as_ref();
                let emit_leaf = &emit_leaf;
                scope.spawn(move || emit_in_order(given..n, emit_window, pool, emit_leaf, &tx));
            } else {
                drop(tx);
            }
            let mut given = leaf_programs.into_iter();
            let mut children = Vec::with_capacity(n);
            for k in 0..n {
                let program = match given.next() {
                    Some(program) => program,
                    None => rx
                        .recv()
                        .map_err(|_| format!("leaf {k}: the leaf emitter stopped early"))??,
                };
                let (artifacts, derived) = built(&program)?;
                children.push(derived);
                self.put(&self.leaves[k], program, artifacts);
            }
            Ok(children)
        })?;
        let leaves_at = start.elapsed().as_secs_f64();
        let mut level_at = Vec::new();
        let mut emit = 0.0;
        let levels = plan.levels();
        for (lv, arities) in levels.iter().enumerate() {
            let top = lv + 1 == levels.len();
            let mut rest = children.into_iter();
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
            let programs: Vec<LfmProgram> = {
                use rayon::prelude::*;
                let emit = || {
                    groups
                        .par_iter()
                        .map(|kids| plan.node_program(kids, top))
                        .collect::<Result<_, String>>()
                };
                match &emit_pool {
                    Some(pool) => pool.install(emit),
                    None => emit(),
                }?
            };
            #[cfg(not(feature = "parallel"))]
            let programs: Vec<LfmProgram> = groups
                .iter()
                .map(|kids| plan.node_program(kids, top))
                .collect::<Result<_, String>>()?;
            emit += t.elapsed().as_secs_f64();
            children = Vec::with_capacity(programs.len());
            for (j, program) in programs.into_iter().enumerate() {
                let (artifacts, derived) = built(&program)?;
                children.push(derived);
                self.put(&self.levels[lv][j], program, artifacts);
            }
            level_at.push(start.elapsed().as_secs_f64());
        }
        Ok(BuilderTimes {
            leaves: leaves_at,
            levels: level_at,
            emit,
            build: build_secs,
        })
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
}
