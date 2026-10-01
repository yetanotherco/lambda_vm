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
    /// Of `build`, the seconds the builder waited for the card.
    pub waited: f64,
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
    /// waits forever. With `first`, the builder takes the card permit ahead of
    /// every other waiter ([`super::device_permit::go_first_here`]).
    pub(super) fn run_builder(
        &self,
        plan: &BlockTreePlan,
        leaf_programs: Vec<LfmProgram>,
        wrap_opts: &crate::ProofOptions,
        first: bool,
    ) -> Result<BuilderTimes, String> {
        let _first = first.then(super::device_permit::go_first_here);
        let run = std::panic::AssertUnwindSafe(|| self.build(plan, leaf_programs, wrap_opts));
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
    ) -> Result<BuilderTimes, String> {
        let start = Instant::now();
        let waited_from = super::device_permit::waited_secs();
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
            self.put(&self.leaves[k], program, artifacts);
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
                groups
                    .par_iter()
                    .map(|kids| plan.node_program(kids, top))
                    .collect::<Result<_, String>>()?
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
            waited: super::device_permit::waited_secs() - waited_from,
        })
    }
}
