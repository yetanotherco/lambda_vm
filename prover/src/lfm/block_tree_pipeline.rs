//! The block tree's programs and artifacts handed to the provers through
//! slots, filled ahead or as a pipeline into level 0 (the whole-block harness's
//! `NOEPOCH_TREE_AHEAD`).
//!
//! - `1`: every program and its artifacts derived beside the base, on the host
//!   ([`super::block_plan::BlockTreePlan::derive_tree`]); [`Pipe::filled`].
//! - `pipe`, the default (`0` derives each program inline, before it proves):
//!   the leaf programs emitted beside the base (host, no artifacts);
//!   once the base is done, [`Pipe::run_builder`] builds each leaf's artifacts on
//!   the card, then emits each node program as soon as its children's ARTIFACTS
//!   exist (a node reads its children's shapes, not their proofs) and builds its
//!   artifacts, level by level, while the leaves prove.
//!
//! A prover takes its program from its slot, waiting for it if the builder is
//! behind; a builder failure fails every waiter.
//!
//! Eager (`NOEPOCH_TREE_EAGER=1`, [`Pipe::run_builder`]'s `eager`): every program
//! is put in its slot before its artifacts, so a prover executes and fills while
//! they are built and waits for them only at `multi_prove`; and each node is
//! emitted as soon as ITS children's artifacts exist, not after its whole level.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use super::LfmArtifacts;
use super::block_plan::{BlockTreePlan, TreePrograms};
use super::compiler::LfmProgram;
use super::per_table_aggregator::{DerivedChild, Level};

/// One program and its artifacts, each once its builder has put it.
#[derive(Default)]
struct Slot {
    value: Mutex<SlotValue>,
    ready: Condvar,
}

#[derive(Default)]
struct SlotValue {
    program: Option<Arc<LfmProgram>>,
    artifacts: Option<LfmArtifacts>,
}

/// Every slot of a block tree: the leaves, then each node level.
pub(super) struct Pipe {
    leaves: Vec<Slot>,
    levels: Vec<Vec<Slot>>,
    failed: AtomicBool,
}

/// When the builder finished each stage, in seconds since it started.
pub(super) struct BuilderTimes {
    /// The last leaf's artifacts built.
    pub leaves: f64,
    /// Each node level's last artifacts built.
    pub levels: Vec<f64>,
    /// Each node level's last program emitted (eager only; empty otherwise).
    pub emitted: Vec<f64>,
    /// Summed over the builder's work: node emission (per level in turn; per
    /// node when eager), and artifact builds (leaves and nodes, each holding
    /// the card permit).
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
                .map(|(program, artifacts)| Slot {
                    value: Mutex::new(SlotValue {
                        program: Some(Arc::new(program)),
                        artifacts: Some(artifacts),
                    }),
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

    /// Leaf `k`'s program and artifacts, waiting for both.
    pub(super) fn take_leaf(&self, k: usize, label: &str) -> (Arc<LfmProgram>, LfmArtifacts) {
        let program = self.take_program(&self.leaves[k], label);
        (program, self.take_artifacts(&self.leaves[k], label))
    }

    /// Node `j` of node level `lv` (0 = the first above the leaves), waiting
    /// for both.
    pub(super) fn take_node(
        &self,
        lv: usize,
        j: usize,
        label: &str,
    ) -> (Arc<LfmProgram>, LfmArtifacts) {
        let program = self.take_program(&self.levels[lv][j], label);
        (program, self.take_artifacts(&self.levels[lv][j], label))
    }

    /// Leaf `k`'s program alone, waiting for it; its artifacts stay in the slot.
    pub(super) fn take_leaf_program(&self, k: usize, label: &str) -> Arc<LfmProgram> {
        self.take_program(&self.leaves[k], label)
    }

    /// Leaf `k`'s artifacts, waiting for them.
    pub(super) fn take_leaf_artifacts(&self, k: usize, label: &str) -> LfmArtifacts {
        self.take_artifacts(&self.leaves[k], label)
    }

    /// Node `j` of node level `lv`'s program alone, waiting for it.
    pub(super) fn take_node_program(&self, lv: usize, j: usize, label: &str) -> Arc<LfmProgram> {
        self.take_program(&self.levels[lv][j], label)
    }

    /// Node `j` of node level `lv`'s artifacts, waiting for them.
    pub(super) fn take_node_artifacts(&self, lv: usize, j: usize, label: &str) -> LfmArtifacts {
        self.take_artifacts(&self.levels[lv][j], label)
    }

    fn take_program(&self, slot: &Slot, label: &str) -> Arc<LfmProgram> {
        self.wait(slot, label, |v| v.program.take())
    }

    fn take_artifacts(&self, slot: &Slot, label: &str) -> LfmArtifacts {
        self.wait(slot, label, |v| v.artifacts.take())
    }

    fn wait<T>(
        &self,
        slot: &Slot,
        label: &str,
        mut take: impl FnMut(&mut SlotValue) -> Option<T>,
    ) -> T {
        let mut value = slot.value.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(v) = take(&mut value) {
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

    fn put(&self, slot: &Slot, program: Option<Arc<LfmProgram>>, artifacts: Option<LfmArtifacts>) {
        {
            let mut value = slot.value.lock().unwrap_or_else(|e| e.into_inner());
            if program.is_some() {
                value.program = program;
            }
            if artifacts.is_some() {
                value.artifacts = artifacts;
            }
        }
        slot.ready.notify_all();
    }

    /// The pipeline's builder, on the card: each leaf's artifacts in leaf
    /// order, then each node level's programs (emitted in parallel on the host
    /// from the children's derived shapes) and their artifacts, each put in its
    /// slot as soon as it exists. A failure marks the pipe failed, so no prover
    /// waits forever. With `emit_threads` > 0 the node programs are emitted on a
    /// pool of that many host-only threads of the builder's own, not on the
    /// global pool the provers' host phases use. `eager`: [`Self::build_eager`].
    pub(super) fn run_builder(
        &self,
        plan: &BlockTreePlan,
        leaf_programs: Vec<LfmProgram>,
        wrap_opts: &crate::ProofOptions,
        emit_threads: usize,
        eager: bool,
    ) -> Result<BuilderTimes, String> {
        let run = std::panic::AssertUnwindSafe(|| {
            if eager {
                self.build_eager(plan, leaf_programs, wrap_opts, emit_threads)
            } else {
                self.build(plan, leaf_programs, wrap_opts, emit_threads)
            }
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
    ) -> Result<BuilderTimes, String> {
        let start = Instant::now();
        let emit_pool = emit_pool(emit_threads)?;
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
        let mut children = Vec::with_capacity(leaf_programs.len());
        for (k, program) in leaf_programs.into_iter().enumerate() {
            let (artifacts, derived) = built(&program)?;
            children.push(derived);
            self.put(&self.leaves[k], Some(Arc::new(program)), Some(artifacts));
        }
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
            let programs: Vec<LfmProgram> = {
                let _ = &emit_pool;
                groups
                    .iter()
                    .map(|kids| plan.node_program(kids, top))
                    .collect::<Result<_, String>>()?
            };
            emit += t.elapsed().as_secs_f64();
            children = Vec::with_capacity(programs.len());
            for (j, program) in programs.into_iter().enumerate() {
                let (artifacts, derived) = built(&program)?;
                children.push(derived);
                self.put(
                    &self.levels[lv][j],
                    Some(Arc::new(program)),
                    Some(artifacts),
                );
            }
            level_at.push(start.elapsed().as_secs_f64());
        }
        Ok(BuilderTimes {
            leaves: leaves_at,
            levels: level_at,
            emitted: Vec::new(),
            emit,
            build: build_secs,
        })
    }

    /// The eager builder. Every leaf program goes in its slot first, so the
    /// leaves execute and fill from level 0's start. Then the card builds each
    /// leaf's artifacts in leaf order, and a node's program is emitted (on a
    /// thread of its own, on the emission pool) as soon as ITS children's
    /// artifacts exist. Each emitted program goes in its slot at once, so its
    /// node can execute as soon as its level starts. Its artifacts follow when
    /// the card has built them: leaves first, then the lowest level emitted.
    fn build_eager(
        &self,
        plan: &BlockTreePlan,
        leaf_programs: Vec<LfmProgram>,
        wrap_opts: &crate::ProofOptions,
        emit_threads: usize,
    ) -> Result<BuilderTimes, String> {
        let start = Instant::now();
        let emit_pool = emit_pool(emit_threads)?;
        let words = plan.child_layout().total();
        let levels = plan.levels();
        // Stage 0 = the leaves, stage s = node level s - 1. `parent[s][i]` is the
        // node of stage s + 1 that child i of stage s feeds.
        let mut sizes = vec![leaf_programs.len()];
        sizes.extend(levels.iter().map(|l| l.arities.len()));
        let mut parent = Vec::with_capacity(levels.len());
        let mut groups = Vec::with_capacity(levels.len());
        for (lv, l) in levels.iter().enumerate() {
            let mut of = Vec::with_capacity(sizes[lv]);
            let mut ranges = Vec::with_capacity(l.arities.len());
            for (j, &a) in l.arities.iter().enumerate() {
                ranges.push(of.len()..of.len() + a);
                of.extend(std::iter::repeat_n(j, a));
            }
            if of.len() != sizes[lv] {
                return Err(format!(
                    "level {}: arities do not cover the children",
                    lv + 1
                ));
            }
            parent.push(of);
            groups.push(ranges);
        }
        let mut derived: Vec<Vec<Option<DerivedChild>>> = sizes
            .iter()
            .map(|&n| (0..n).map(|_| None).collect())
            .collect();
        let leaves: Vec<Arc<LfmProgram>> = leaf_programs.into_iter().map(Arc::new).collect();
        for (k, program) in leaves.iter().enumerate() {
            self.put(&self.leaves[k], Some(program.clone()), None);
        }
        let total_nodes: usize = sizes[1..].iter().sum();
        let mut level_built = vec![0usize; levels.len()];
        let mut level_emitted = vec![0usize; levels.len()];
        let mut level_at = vec![0.0; levels.len()];
        let mut emitted_at = vec![0.0; levels.len()];
        let (mut emit, mut build_secs, mut leaves_at) = (0.0, 0.0, 0.0);
        let (tx, rx) =
            std::sync::mpsc::channel::<(usize, usize, Result<LfmProgram, String>, f64)>();
        std::thread::scope(|scope| -> Result<(), String> {
            let pool = &emit_pool;
            let spawn_emit = |lv: usize, j: usize, kids: Vec<DerivedChild>| {
                let tx = tx.clone();
                let top = lv + 1 == levels.len();
                scope.spawn(move || {
                    let t = Instant::now();
                    let program = emit_one(pool, plan, &kids, top);
                    let _ = tx.send((lv, j, program, t.elapsed().as_secs_f64()));
                });
            };
            let mut ready: std::collections::BTreeMap<(usize, usize), Arc<LfmProgram>> =
                std::collections::BTreeMap::new();
            let (mut next_leaf, mut nodes_built, mut in_flight) = (0usize, 0usize, 0usize);
            let mut take_emitted =
                |(lv, j, program, secs): (usize, usize, Result<LfmProgram, String>, f64),
                 ready: &mut std::collections::BTreeMap<(usize, usize), Arc<LfmProgram>>|
                 -> Result<(), String> {
                    let program = Arc::new(program?);
                    emit += secs;
                    level_emitted[lv] += 1;
                    if level_emitted[lv] == sizes[lv + 1] {
                        emitted_at[lv] = start.elapsed().as_secs_f64();
                    }
                    self.put(&self.levels[lv][j], Some(program.clone()), None);
                    ready.insert((lv, j), program);
                    Ok(())
                };
            while nodes_built < total_nodes || next_leaf < leaves.len() {
                while let Ok(msg) = rx.try_recv() {
                    in_flight -= 1;
                    take_emitted(msg, &mut ready)?;
                }
                let (stage, i, program) = if next_leaf < leaves.len() {
                    next_leaf += 1;
                    (0, next_leaf - 1, leaves[next_leaf - 1].clone())
                } else if let Some(((lv, j), program)) = ready.pop_first() {
                    (lv + 1, j, program)
                } else if in_flight > 0 {
                    let msg = rx
                        .recv()
                        .map_err(|_| "an emission thread stopped".to_string())?;
                    in_flight -= 1;
                    take_emitted(msg, &mut ready)?;
                    continue;
                } else {
                    return Err("the eager builder has nothing to build or wait for".into());
                };
                let t = Instant::now();
                let artifacts = super::program_census::build_artifacts_counted(
                    &program,
                    wrap_opts,
                    crate::hash_pin::BLOCK_HASHER,
                );
                derived[stage][i] =
                    Some(DerivedChild::from_artifacts(&artifacts, wrap_opts, words)?);
                build_secs += t.elapsed().as_secs_f64();
                let slot = if stage == 0 {
                    &self.leaves[i]
                } else {
                    &self.levels[stage - 1][i]
                };
                self.put(slot, None, Some(artifacts));
                if stage == 0 {
                    if next_leaf == leaves.len() {
                        leaves_at = start.elapsed().as_secs_f64();
                    }
                } else {
                    nodes_built += 1;
                    level_built[stage - 1] += 1;
                    if level_built[stage - 1] == sizes[stage] {
                        level_at[stage - 1] = start.elapsed().as_secs_f64();
                    }
                }
                // This child's parent, once every child of it is built.
                if stage < levels.len() {
                    let j = parent[stage][i];
                    let range = groups[stage][j].clone();
                    if derived[stage][range.clone()].iter().all(Option::is_some) {
                        let kids = derived[stage][range]
                            .iter_mut()
                            .map(|d| d.take().expect("checked above"))
                            .collect();
                        spawn_emit(stage, j, kids);
                        in_flight += 1;
                    }
                }
            }
            Ok(())
        })?;
        Ok(BuilderTimes {
            leaves: leaves_at,
            levels: level_at,
            emitted: emitted_at,
            emit,
            build: build_secs,
        })
    }
}

/// The builder's own emission pool (`emit_threads` > 0), or none: the global
/// pool. ⚠ On the global pool, a prover that joins inside its `multi_prove` can
/// steal one of the ≈ 1 s emissions and hold the card idle until it finishes
/// (FAST 454: one leaf hold of 1.2 s at a third busy, in half the runs).
#[cfg(feature = "parallel")]
type EmitPool = Option<rayon::ThreadPool>;
#[cfg(not(feature = "parallel"))]
type EmitPool = ();

#[cfg(feature = "parallel")]
fn emit_pool(emit_threads: usize) -> Result<EmitPool, String> {
    if emit_threads == 0 {
        return Ok(None);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(emit_threads)
        .thread_name(|i| format!("tree-emit-{i}"))
        .start_handler(|_| super::commit::mark_thread_host_only())
        .build()
        .map(Some)
        .map_err(|e| format!("the emission pool: {e}"))
}

#[cfg(not(feature = "parallel"))]
fn emit_pool(_emit_threads: usize) -> Result<EmitPool, String> {
    Ok(())
}

/// One node program, emitted on `pool` when there is one.
fn emit_one(
    pool: &EmitPool,
    plan: &BlockTreePlan,
    kids: &[DerivedChild],
    top: bool,
) -> Result<LfmProgram, String> {
    #[cfg(feature = "parallel")]
    if let Some(pool) = pool {
        return pool.install(|| plan.node_program(kids, top));
    }
    #[cfg(not(feature = "parallel"))]
    let _ = pool;
    plan.node_program(kids, top)
}
