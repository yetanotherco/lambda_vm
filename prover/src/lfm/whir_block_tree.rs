//! ★ The no-epoch WHIR block, base to top node (W3): the one driver the
//! whole-block harness (`whir_block_tests::the_whir_block_tree_on_a_real_block`)
//! and the CLI's `prove-block` both run, so their numbers cannot diverge.
//!
//! [`prove_whir_block_tree`] proves the block in one proof at the production
//! format ([`crate::block_whir::prove_block_whir_observed_groups`]); the moment
//! its statement is final it derives the plan and emits the leaves' programs
//! beside phase B, and the first leaf whose groups phase B finished executes
//! and fills right then; then it proves the tree ([`prove_tree_pipelined`]):
//! the leaves a bounded number at once, the nodes' programs and artifacts built
//! beside them, each node proved as soon as its own children are. Every line
//! goes through a [`WhirTreeSink`]: the harness prints to stdout, the CLI to
//! stderr. What the harness checks off the clock (every tree proof against its
//! artifacts, the base on the host, the top through the block verifier) stays
//! in the harness.
//!
//! Every knob is read once, by [`WhirTreeConfig::from_env`], under the names
//! and defaults the harness always read. [`POSTURE`] is the production posture
//! the measurements run under; the CLI sets it for knobs the environment
//! leaves unset, and the driver's first line says how the running process
//! compares with it.

use multilinear::whir_chain::ArgueFormat;

use crate::block_whir::{self, BlockFormat, BlockOptions, BlockWhirProof};

use super::compiler::LfmProgram;
use super::harvest::{HarvestedChild, harvest_child, try_child_arena_words};
use super::per_table_aggregator::DerivedChild;
use super::proof::{
    LfmFilled, LfmProof, aggregation_wrap_options, decide_lfm_residency, lfm_execute_and_fill,
    take_prove_split,
};
use super::whir_block::{
    BLOCK_FAN_IN, WhirBlockPlan, artifacts_of, block_leaf_arena, group_arena_words, leaf_arena,
};
use super::word::LfmWord;

// ================================ the sink ================================

/// Where the driver's lines go.
pub trait WhirTreeSink: Send + Sync {
    /// Whole lines, each ending in `\n`, written at once.
    fn write(&self, text: &str);

    /// One line.
    fn line(&self, line: &str) {
        self.write(&format!("{line}\n"));
    }
}

/// Lines to stdout (the harness's).
pub struct StdoutSink;

impl WhirTreeSink for StdoutSink {
    fn write(&self, text: &str) {
        print!("{text}");
    }
}

/// Lines to stderr (the CLI's: its stdout carries the results).
pub struct StderrSink;

impl WhirTreeSink for StderrSink {
    fn write(&self, text: &str) {
        eprint!("{text}");
    }
}

// =============================== the posture ==============================

/// The production posture the block's measurements run under: the knobs the
/// suite env sets that change behaviour or performance, at its values (ULTRA's
/// and BIG's W3 env). The CLI sets each one the environment leaves unset, and
/// the harness never does: its arms rely on unset meaning the library default.
/// `LAMBDA_VM_WHIR_HASH` is a format knob, so the CLI's `verify-block` sets it
/// too, and a verifier configured otherwise refuses (completeness, never
/// soundness). The measurement-only `LAMBDA_VM_BASE_SPLIT` is not posture, and
/// the allocator's never-purge posture is compiled into the binary.
pub const POSTURE: &[(&str, &str)] = &[
    ("TABLE_PARALLELISM", "4"),
    ("LAMBDA_VM_MAX_ROWS_LOG2", "21"),
    ("LFM_WHIR_RETENTION", "1"),
    ("LAMBDA_VM_WHIR_HASH", "rpx"),
    ("LFM_PRECOMPUTED_TREE_CACHE_CAP", "64"),
    ("LFM_EXEC_PARALLEL", "1"),
    ("LAMBDA_VM_GRIND_SCAN_FACTOR", "8"),
    ("LAMBDA_VM_GRIND_GRID", "1024"),
];

/// The `BLOCK POSTURE:` line: each posture knob as the process has it — equal
/// to the posture, unset, or another value.
pub fn posture_line() -> String {
    let words: Vec<String> = POSTURE
        .iter()
        .map(|(name, want)| match std::env::var(name) {
            Ok(v) if v == *want => format!("{name}={v}"),
            Ok(v) => format!("{name}={v} (≠ posture {want})"),
            Err(_) => format!("{name} unset (≠ posture {want})"),
        })
        .collect();
    format!("BLOCK POSTURE: {}", words.join(" · "))
}

// ================================ the knobs ===============================

/// Every knob the W3 block reads, once.
#[derive(Clone, Debug)]
pub struct WhirTreeConfig {
    /// `W3_LEAVES`: the leaf count; unset is the plan's rule. A forced count
    /// is another tree than the production verifier derives.
    pub leaves: Option<usize>,
    /// `W3_SIBLINGS` (3 by default): programs of the tree proved at once.
    pub siblings: usize,
    /// `W3_FAN_IN` ([`BLOCK_FAN_IN`] by default): another fan-in is another
    /// tree than the production verifier derives.
    pub fan_in: usize,
    /// `W3_EXEC_BESIDE_ARTIFACTS=0|1` (default 1): the leaves execute and fill
    /// while their artifacts are built, or (0, the control) after all of them.
    pub beside: bool,
    /// `W3_LEAF_DURING_PHASE_B=0|1` (default 1): the first leaf whose groups
    /// phase B finished while another group was still to come executes and
    /// fills right then, or (0, the control) with the others after the base.
    pub early_on: bool,
    /// `BLOCK_WHIR_ARGUE=batched|per-table` (production: batched, at the
    /// format's bin cap; `BLOCK_WHIR_ARGUE_CAP=k` for 2^k). Another argue is
    /// another format than the production verifier's.
    pub argue: ArgueFormat,
    /// The block format: production, with `argue`.
    pub format: BlockFormat,
    /// The base's options: production, with the `BLOCK_WHIR_*` knobs.
    pub options: BlockOptions,
    /// `W3_NODE_PIPE=0|1` (default 1): the nodes' programs and artifacts built
    /// on a pool of `W3_EMIT_THREADS` threads of their own (default 4), each
    /// level proving as its own nodes arrive; or (0, the control) one after
    /// another on one thread, level 1 waiting for the whole tree's builds.
    pub node_pipe: Option<usize>,
    /// `W3_DATAFLOW=0|1` (default 1): each node proves as soon as its own
    /// children are proved, or (0, the control) once its whole level below is.
    pub dataflow: bool,
    /// `W3_STREAM_TOP=0|1` (default 1): when a child is still unproved as the
    /// top node can execute, the top executes each child's share as that child
    /// is proved; otherwise, and at 0 (the control), all of it after the last.
    pub stream_top: bool,
    /// `W3_EXEC_EARLY=0|1` (default 1): a node executes and fills from its
    /// program as soon as the builder emits it, and only its prove waits for
    /// its artifacts; or (0, the control) it executes once its artifacts are
    /// built too.
    pub exec_early: bool,
    /// `W3_NODE_TIMES=1`: the per-program stamps readout after the tree.
    pub node_times: bool,
}

/// A count knob: unset is `None`; anything but a count is an error.
fn count_knob(name: &str) -> Result<Option<usize>, String> {
    match std::env::var(name) {
        Err(_) => Ok(None),
        Ok(v) => v
            .trim()
            .parse()
            .map(Some)
            .map_err(|_| format!("{name} is a count, got `{v}`")),
    }
}

impl WhirTreeConfig {
    /// Every knob from the environment, under the names and defaults the
    /// harness always read. A value a knob cannot take is an error (the
    /// `BLOCK_WHIR_*` readers of [`crate::block_whir`] keep their own refusals).
    pub fn from_env() -> Result<Self, String> {
        let leaves = count_knob("W3_LEAVES")?;
        let siblings = count_knob("W3_SIBLINGS")?.unwrap_or(3);
        let fan_in = count_knob("W3_FAN_IN")?.unwrap_or(BLOCK_FAN_IN);
        let beside = count_knob("W3_EXEC_BESIDE_ARTIFACTS")?.is_none_or(|v| v != 0);
        let early_on = count_knob("W3_LEAF_DURING_PHASE_B")?.is_none_or(|v| v != 0);
        let argue = match std::env::var("BLOCK_WHIR_ARGUE").as_deref().map(str::trim) {
            Ok("batched") => match count_knob("BLOCK_WHIR_ARGUE_CAP")? {
                Some(cap) => ArgueFormat::Batched {
                    bin_log_cells: u8::try_from(cap)
                        .map_err(|_| format!("BLOCK_WHIR_ARGUE_CAP={cap}: a cap below 2^256"))?,
                },
                None => ArgueFormat::BATCHED,
            },
            Ok("per-table") => ArgueFormat::PerTable,
            Err(_) => BlockFormat::production().argue,
            Ok(other) => return Err(format!("BLOCK_WHIR_ARGUE={other}: per-table or batched")),
        };
        let format = BlockFormat {
            argue,
            ..BlockFormat::production()
        };
        let mut options = BlockOptions::production();
        // `BLOCK_WHIR_STREAM_KECCAK_RND=1`: KECCAK_RND's chunks streamed (off by
        // default: NO EFFECT on this block, FAST 418).
        options.stream_keccak_rnd =
            std::env::var("BLOCK_WHIR_STREAM_KECCAK_RND").is_ok_and(|v| v.trim() == "1");
        // `BLOCK_WHIR_STREAM_MEMW_LT=1`: the MEMW-derived LT ops streamed per window.
        options.stream_memw_lt =
            std::env::var("BLOCK_WHIR_STREAM_MEMW_LT").is_ok_and(|v| v.trim() == "1");
        // `BLOCK_WHIR_LAYOUT_WORKERS=n`: the streamed chunks laid out on n threads
        // (production 3; 0 is the inline layout).
        if let Some(n) = count_knob("BLOCK_WHIR_LAYOUT_WORKERS")? {
            options.layout_workers = n;
        }
        // `BLOCK_WHIR_LAYOUT_AHEAD=k` bounds the chunks unpacked at k + 1;
        // `none` lifts the bound (the reverted version, BIG 390).
        match std::env::var("BLOCK_WHIR_LAYOUT_AHEAD")
            .as_deref()
            .map(str::trim)
        {
            Ok("none") => options.layout_ahead = None,
            Ok(k) => {
                if let Ok(k) = k.parse() {
                    options.layout_ahead = Some(k);
                }
            }
            Err(_) => {}
        }
        // `BLOCK_WHIR_PACK_REST=0|1` (production 0): the rest packed as it is laid
        // out.
        match std::env::var("BLOCK_WHIR_PACK_REST")
            .as_deref()
            .map(str::trim)
        {
            Ok("0") => options.pack_rest_as_laid_out = false,
            Ok("1") => options.pack_rest_as_laid_out = true,
            Ok(other) => return Err(format!("BLOCK_WHIR_PACK_REST={other}: 0 or 1")),
            Err(_) => {}
        }
        // `BLOCK_WHIR_DROP_OPS=0`: the builder keeps the streamed chunks' ops
        // (production drops them).
        match std::env::var("BLOCK_WHIR_DROP_OPS")
            .as_deref()
            .map(str::trim)
        {
            Ok("0") => options.drop_streamed_ops = false,
            Ok("1") => options.drop_streamed_ops = true,
            _ => {}
        }
        // `BLOCK_WHIR_NARROW=wide|card|host` (production card): how phase A holds
        // each group's columns once committed.
        if let Some(narrow) = block_whir::narrow_from_env() {
            options.narrow = narrow;
        }
        // `BLOCK_WHIR_KECCAK_LOG2=k`, `BLOCK_WHIR_ECSM_LOG2=k`: force a KECCAK or
        // ECSM split (production 2^18 / 2^17).
        block_whir::chunk_cuts_from_env(&mut options);
        // `BLOCK_WHIR_UPLOAD_AHEAD=0|1` (production 1): phase A puts each group's
        // columns on the card beside the previous group's commit.
        if let Some(ahead) = block_whir::upload_ahead_from_env() {
            options.upload_ahead = ahead;
        }
        // `BLOCK_WHIR_REST_LAYOUT=all|<MiB>` (production 2048),
        // `BLOCK_WHIR_KR_FINISH_CHUNKS=0|1` and `BLOCK_WHIR_PACK_FINISHED=0|1`
        // (production 1): the rest's layout in waves, KECCAK_RND built as its
        // tables, the finish's tables packed as they are built.
        block_whir::rest_layout_from_env(&mut options);
        let node_pipe = count_knob("W3_NODE_PIPE")?
            .is_none_or(|v| v != 0)
            .then(|| count_knob("W3_EMIT_THREADS").map(|t| t.unwrap_or(4).max(1)))
            .transpose()?;
        Ok(Self {
            leaves,
            siblings,
            fan_in,
            beside,
            early_on,
            argue,
            format,
            options,
            node_pipe,
            dataflow: count_knob("W3_DATAFLOW")?.is_none_or(|v| v != 0),
            stream_top: count_knob("W3_STREAM_TOP")?.is_none_or(|v| v != 0),
            exec_early: count_knob("W3_EXEC_EARLY")?.is_none_or(|v| v != 0),
            node_times: count_knob("W3_NODE_TIMES")? == Some(1),
        })
    }

    /// Whether the tree is the production verifier's: the plan's leaf count,
    /// [`BLOCK_FAN_IN`] and the production argue.
    pub fn at_presets(&self) -> bool {
        self.leaves.is_none()
            && self.fan_in == BLOCK_FAN_IN
            && self.argue == BlockFormat::production().argue
    }
}

// ============================ the tree's machinery ========================

/// A node's program as the builder emits it, before its artifacts: all its
/// execute and fill need, and when it was emitted (seconds since the tree
/// started).
pub(crate) struct NodeProgram {
    pub(crate) program: std::sync::Arc<LfmProgram>,
    pub(crate) at: f64,
}

/// A node built ahead of the proofs below it: its artifacts (what its prove
/// needs; its program was published before them, [`NodeProgram`]).
pub(crate) struct TreeNode {
    pub(crate) artifacts: super::registry::LfmArtifacts,
    pub(crate) derived: DerivedChild,
    /// Seconds emitting it and building its artifacts.
    pub(crate) built: f64,
    /// When its builder finished it, in seconds since the tree started.
    pub(crate) built_at: f64,
}

/// One level's readout: its wall, and per program (artifacts, prove) seconds
/// and its [`ProgramTimes`].
pub(crate) struct LevelTiming {
    pub(crate) wall: f64,
    pub(crate) programs: Vec<(f64, f64)>,
    pub(crate) times: Vec<ProgramTimes>,
}

/// When one program of the tree ran, in seconds since the tree started, and
/// its prove's split ([`take_prove_split`]: execute, fill, the prove net of
/// the card wait, the card wait) — the `W3 TIMES` readout.
#[derive(Clone, Copy, Default)]
pub(crate) struct ProgramTimes {
    /// Its program emitted (nodes; the leaves' exist before the tree).
    pub(crate) program_at: f64,
    /// Its program (nodes) and artifacts built.
    pub(crate) built_at: f64,
    /// Its worker took it.
    pub(crate) start: f64,
    /// Its proof done.
    pub(crate) end: f64,
    pub(crate) split: Option<(f64, f64, f64, f64)>,
}

/// The split of the prove that just ran on this thread.
pub(crate) fn prove_split_now() -> Option<(f64, f64, f64, f64)> {
    take_prove_split().map(|s| (s.execute, s.fill, s.multi_prove, s.permit_wait))
}

/// One group's share of the proof, owned, as phase B hands it over
/// ([`block_whir::GroupObserver`]).
pub(crate) struct GroupMsg {
    pub(crate) group: usize,
    pub(crate) roots: Vec<multilinear::whir_commit::Commitment>,
    pub(crate) tables:
        Vec<stark::multilinear_table::TableProof<crate::tables::types::GoldilocksExtension>>,
    pub(crate) argue:
        Option<stark::multilinear_table::BatchedArgue<crate::tables::types::GoldilocksExtension>>,
    pub(crate) opening: multilinear::stacked_eval::StackedProof<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >,
    pub(crate) prepared: Option<
        multilinear::stacked_eval::StackedProof<
            crate::tables::types::GoldilocksField,
            crate::tables::types::GoldilocksExtension,
        >,
    >,
    pub(crate) at: f64,
}

/// What the early leaf's thread hands back: the filled leaf, its arena (for
/// the off-the-clock check against the finished proof's), and when its
/// execute + fill ran.
pub(crate) type EarlyOut = (LfmFilled, Vec<Vec<LfmWord>>, f64, f64);

/// The first leaf whose groups phase B finished while another group was
/// still to come, executing and filling on a thread of its own.
pub(crate) struct EarlyLeaf {
    pub(crate) leaf: usize,
    /// When its last group's opening ended (seconds since the prove started).
    pub(crate) complete_at: f64,
    pub(crate) handle: std::thread::JoinHandle<Result<EarlyOut, String>>,
}

/// A value one thread publishes once and others wait for. A publisher that
/// unwinds before publishing leaves an error behind ([`PublishGuard`]), so no
/// waiter blocks on a value that will never come.
pub(crate) struct Published<T>(pub(crate) std::sync::OnceLock<Result<T, String>>);

impl<T> Published<T> {
    pub(crate) fn new() -> Self {
        Self(std::sync::OnceLock::new())
    }

    pub(crate) fn wait(&self) -> Result<&T, String> {
        self.0.wait().as_ref().map_err(Clone::clone)
    }

    pub(crate) fn take(self) -> Result<T, String> {
        self.0
            .into_inner()
            .unwrap_or_else(|| Err("never published".to_string()))
    }
}

/// Publishes an error on drop unless [`PublishGuard::publish`] ran first.
pub(crate) struct PublishGuard<'a, T>(pub(crate) &'a Published<T>);

impl<T> PublishGuard<'_, T> {
    pub(crate) fn publish(self, value: Result<T, String>) {
        let _ = self.0.0.set(value);
    }
}

impl<T> Drop for PublishGuard<'_, T> {
    fn drop(&mut self) {
        let _ = self.0.0.set(Err("the publisher unwound".to_string()));
    }
}

/// The leaves' artifacts as their builder publishes them: per leaf, its
/// artifacts, its derived shape and the seconds building them.
pub(crate) type LeafBuilt = Vec<(super::registry::LfmArtifacts, DerivedChild, f64)>;

/// Publishes an error into every slot still empty when it drops: a node
/// builder that stops early (an error, or a panic) leaves no level waiting
/// for a node that will never come.
pub(crate) struct FailUnpublished<'a, T>(pub(crate) &'a [Vec<Published<T>>]);

impl<T> Drop for FailUnpublished<'_, T> {
    fn drop(&mut self) {
        for slot in self.0.iter().flatten() {
            let _ = slot.0.set(Err("the node builder stopped".to_string()));
        }
    }
}

/// A tree's nodes built level by level from the leaves' shapes `leaves`, each
/// published to its slot (level, node) as soon as it is built: `emit` makes a
/// node's program from its level, index, children's shapes (in order) and
/// whether it is the top, as the part published to `programs` at once (what a
/// node executes) and the part `finish` builds the rest of the node from (its
/// artifacts: it may take the card); `shape_of` gives a built node's shape for
/// the level above. A node depends on its children's shapes, never on their
/// proofs, so the whole tree can be built while the leaves prove.
///
/// With `pool`, a level's programs are emitted together on the pool's threads,
/// and this thread finishes each as its emission ends and publishes it into
/// ITS OWN slot, whatever order they finish in; without, emitted and finished
/// one after another here. `finish` always runs on this thread, never on a
/// rayon worker: a worker that waits inside rayon while it holds the card runs
/// queued jobs meanwhile, and a sibling that takes the card is a second hold on
/// one thread (BIG 569). A node's program is published as its emission ends
/// (on the pool, with `pool`), before its finish starts and whatever the other
/// nodes' finishes are waiting for. A build that fails leaves its error in its slots, and every slot
/// still empty when this returns or unwinds gets one ([`FailUnpublished`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_levels<C: Sync, E: Send + Sync, P: Send, N: Send + Sync>(
    shape: &[super::per_table_aggregator::Level],
    leaves: &[&C],
    programs: &[Vec<Published<E>>],
    slots: &[Vec<Published<N>>],
    shape_of: impl Fn(&N) -> &C + Sync,
    emit: impl Fn(usize, usize, &[&C], bool) -> Result<(E, P), String> + Sync,
    finish: impl Fn(usize, usize, P) -> Result<N, String>,
    pool: Option<&rayon::ThreadPool>,
) {
    let _fail_programs = FailUnpublished(programs);
    let _fail = FailUnpublished(slots);
    // Publishes an emitted node's program, and hands on what its finish takes.
    let publish = |lv: usize, j: usize, emitted: Result<(E, P), String>| match emitted {
        Ok((program, rest)) => {
            let _ = programs[lv][j].0.set(Ok(program));
            Ok(rest)
        }
        Err(e) => {
            let _ = programs[lv][j].0.set(Err(e.clone()));
            Err(e)
        }
    };
    for (lv, arities) in shape.iter().enumerate() {
        let top = lv + 1 == shape.len();
        let below: Vec<&C> = match lv {
            0 => leaves.to_vec(),
            _ => match slots[lv - 1]
                .iter()
                .map(|s| s.wait().map(&shape_of))
                .collect::<Result<Vec<_>, String>>()
            {
                Ok(below) => below,
                Err(_) => return,
            },
        };
        let mut at = 0usize;
        let groups: Vec<std::ops::Range<usize>> = arities
            .arities
            .iter()
            .map(|&a| {
                at += a;
                at - a..at
            })
            .collect();
        if groups.last().map_or(0, |g| g.end) != below.len() {
            let _ = slots[lv][0].0.set(Err(format!(
                "level {}: arities cover {at} of {} children",
                lv + 1,
                below.len()
            )));
            return;
        }
        match pool {
            Some(pool) => {
                let (tx, rx) = std::sync::mpsc::channel::<(usize, Result<P, String>)>();
                pool.in_place_scope(|scope| {
                    for (j, kids) in groups.iter().enumerate() {
                        let (tx, emit, publish) = (tx.clone(), &emit, &publish);
                        let kids = &below[kids.clone()];
                        // Each program is published here, as its emission
                        // ends: never behind another node's finish, which
                        // may wait for the card (BIG 622: a program queued
                        // 2.1 s behind a sibling's artifacts).
                        scope.spawn(move |_| {
                            let _ = tx.send((j, publish(lv, j, emit(lv, j, kids, top))));
                        });
                    }
                    drop(tx);
                    // This thread, not a pool worker, finishes each node as
                    // its program arrives.
                    for (j, rest) in rx {
                        let _ = slots[lv][j].0.set(rest.and_then(|p| finish(lv, j, p)));
                    }
                });
            }
            None => {
                for (j, kids) in groups.iter().enumerate() {
                    let rest = publish(lv, j, emit(lv, j, &below[kids.clone()], top));
                    let _ = slots[lv][j].0.set(rest.and_then(|p| finish(lv, j, p)));
                }
            }
        }
    }
}

/// The tree's node programs and artifacts ([`build_levels`] over the leaves'
/// derived shapes): each node's program published to `programs` as soon as it
/// is emitted (what its execute and fill need), and the node with its
/// artifacts to `slots` once they are built (what its prove needs).
///
/// With `pool` (`W3_NODE_PIPE`, on by default) a level's programs are emitted
/// together on the pool's threads — the builder's own, not the global pool, on
/// which a prover's join could steal an emission and leave the card idle
/// (#1013's FAST 454) — and this thread builds each one's artifacts (holding
/// the card) as it arrives; each level proves as its own nodes arrive. Without
/// it, one after another on this thread (the control: the builder before,
/// whose whole tree level 1 then waits for — at the median 32 s of serial
/// builds holding level 0 open 10.7 s past its last proof, BIG 565 / 568).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_nodes(
    plan: &WhirBlockPlan,
    shape: &[super::per_table_aggregator::Level],
    leaves: &Published<LeafBuilt>,
    programs: &[Vec<Published<NodeProgram>>],
    slots: &[Vec<Published<TreeNode>>],
    wrap: &crate::ProofOptions,
    words: usize,
    t_tree: std::time::Instant,
    pool: Option<&rayon::ThreadPool>,
) {
    let Ok(leaf_built) = leaves.wait() else {
        let _fail_programs = FailUnpublished(programs);
        let _fail = FailUnpublished(slots);
        return;
    };
    let leaf_shapes: Vec<&DerivedChild> = leaf_built.iter().map(|(_, d, _)| d).collect();
    type Emitted = (NodeProgram, (std::sync::Arc<LfmProgram>, f64));
    build_levels(
        shape,
        &leaf_shapes,
        programs,
        slots,
        |node: &TreeNode| &node.derived,
        |_, _, kids: &[&DerivedChild], top| -> Result<Emitted, String> {
            let t = std::time::Instant::now();
            let program = std::sync::Arc::new(plan.node_program(kids, top)?);
            let at = t_tree.elapsed().as_secs_f64();
            Ok((
                NodeProgram {
                    program: std::sync::Arc::clone(&program),
                    at,
                },
                (program, t.elapsed().as_secs_f64()),
            ))
        },
        |_, _, (program, emitted): (std::sync::Arc<LfmProgram>, f64)| -> Result<TreeNode, String> {
            let t = std::time::Instant::now();
            let artifacts = artifacts_of(&program, wrap);
            let derived = DerivedChild::from_artifacts(&artifacts, wrap, words)?;
            Ok(TreeNode {
                artifacts,
                derived,
                built: emitted + t.elapsed().as_secs_f64(),
                built_at: t_tree.elapsed().as_secs_f64(),
            })
        },
        pool,
    );
}

/// A proved program of the tree, as [`prove_dataflow`] publishes it: its
/// proof, its harvest for the level above, its prove's seconds (artifacts
/// wait, execute, fill and prove), its build seconds (nodes) and its stamps.
pub(crate) struct ProvedProgram {
    pub(crate) lfm: LfmProof,
    pub(crate) child: HarvestedChild,
    pub(crate) prove: f64,
    pub(crate) built: f64,
    pub(crate) times: ProgramTimes,
}

/// One program of the tree: its level (0 = the leaves) and its index there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeAt {
    pub(crate) lv: usize,
    pub(crate) j: usize,
}

/// Every program of a tree of `leaves` leaves laid out as `shape`, in
/// topological order: the leaves, then each node level in turn. Workers that
/// take programs in this order reach a node only after every program before it
/// — its children among them — has been taken, so the earliest unfinished
/// program waits on nothing untaken: no deadlock, at any number of workers.
pub(crate) fn tree_order(
    leaves: usize,
    shape: &[super::per_table_aggregator::Level],
) -> Vec<TreeAt> {
    let mut order: Vec<TreeAt> = (0..leaves).map(|j| TreeAt { lv: 0, j }).collect();
    for (l, level) in shape.iter().enumerate() {
        order.extend((0..level.arities.len()).map(|j| TreeAt { lv: l + 1, j }));
    }
    order
}

/// Node `at`'s children, as indices into level `at.lv − 1`.
pub(crate) fn tree_children(
    shape: &[super::per_table_aggregator::Level],
    at: TreeAt,
) -> std::ops::Range<usize> {
    let arities = &shape[at.lv - 1].arities;
    let start: usize = arities[..at.j].iter().sum();
    start..start + arities[at.j]
}

/// Proves every program of a tree on `workers` threads, taking them in `order`
/// ([`tree_order`]) and publishing each result into its slot (`results[lv][j]`,
/// level 0 the leaves): `prove` gets the program and its children's slots in
/// order (none for a leaf), to wait for as it needs them ([`wait_all`]), so a
/// node whose children are done proves while the rest of the level below still
/// does — and one that streams can start on its first child. With
/// `barrier` (the control, `W3_DATAFLOW=0`), a node first waits for the whole
/// level below, as level-by-level proving did. A failed or panicking prove
/// leaves an error in its slot and so in every slot above it; every slot still
/// empty when this returns or unwinds gets one.
pub(crate) fn prove_dataflow<R: Send + Sync>(
    shape: &[super::per_table_aggregator::Level],
    order: &[TreeAt],
    results: &[Vec<Published<R>>],
    workers: usize,
    barrier: bool,
    prove: impl Fn(TreeAt, &[&Published<R>]) -> Result<R, String> + Sync,
) {
    use super::tree_run::in_index_order;
    let _fail = FailUnpublished(results);
    in_index_order(order.len(), workers, |i| {
        let at = order[i];
        let kids = match at.lv {
            0 => Ok(Vec::new()),
            _ => {
                let below = &results[at.lv - 1];
                let all = match barrier {
                    true => below.iter().try_for_each(|s| s.wait().map(drop)),
                    false => Ok(()),
                };
                all.map(|()| {
                    tree_children(shape, at)
                        .map(|c| &below[c])
                        .collect::<Vec<_>>()
                })
            }
        };
        let result = kids.and_then(|kids| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prove(at, &kids)))
                .unwrap_or_else(|payload| {
                    let why = payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_default();
                    Err(format!("level {} program {}: panicked: {why}", at.lv, at.j))
                })
        });
        let _ = results[at.lv][at.j].0.set(result);
    });
}

/// The top node streamed: its execution starts on the first of its children to
/// be proved and runs each child's share as that child lands
/// ([`super::executor::StreamedExecution`]: the witness is the unstreamed
/// one's, word for word), so only the last child's share and the fill remain
/// after the last child. A node's arenas are its children's, in order and the
/// same number each, so child `c` is group `c`. The execute the split reports
/// is the part after the last child landed. Returns the filled traces, for a
/// prove under artifacts built for `hasher`.
pub(crate) fn execute_streamed(
    program: &LfmProgram,
    kids: &[&Published<ProvedProgram>],
    hasher: crate::lfm::hash::HasherKind,
) -> Result<LfmFilled, String> {
    use super::executor::StreamedExecution;
    let k = kids.len();
    let arenas = program.arena_schema.lens.len();
    if k == 0 || !arenas.is_multiple_of(k) {
        return Err(format!("{arenas} arenas over {k} children"));
    }
    let per = arenas / k;
    let held: Vec<std::cell::OnceCell<Vec<Vec<LfmWord>>>> =
        (0..k).map(|_| std::cell::OnceCell::new()).collect();
    let mut ex = StreamedExecution::new(program, (0..arenas).map(|a| a / per).collect(), &hasher)
        .map_err(|e| format!("{e:?}"))?;
    let mut last = std::time::Instant::now();
    in_arrival_order(kids, |c, kid| {
        let words = match held[c].get() {
            Some(words) => words,
            None => {
                let words = try_child_arena_words(&kid.child)?;
                held[c].get_or_init(|| words)
            }
        };
        last = std::time::Instant::now();
        ex.land(c, words)
            .map_err(|e| format!("child {c} lands: {e:?}"))
    })?;
    let execution = ex.finish().map_err(|e| format!("{e:?}"))?;
    Ok(super::proof::lfm_fill_executed(
        program,
        execution,
        hasher,
        last.elapsed().as_secs_f64(),
    ))
}

/// One node of the tree: its execute and fill (`execute`) from its program as
/// soon as the builder has emitted it, and its prove (`prove`) only once its
/// OWN artifacts are in its slot — the program in `programs[lv − 1][j]`, the
/// artifacts in `slots[lv − 1][j]`. With `early` off (the control,
/// `W3_EXEC_EARLY=0`), the execute also waits for the artifacts, as when one
/// slot held both. The artifacts are a card hold that queues behind the
/// proves below (BIG 617: the top's came 0.12-0.15 s after its last child
/// although its program was emitted 0.03-2.1 s before it).
pub(crate) fn node_flow<E, N, F, R>(
    programs: &[Vec<Published<E>>],
    slots: &[Vec<Published<N>>],
    at: TreeAt,
    early: bool,
    execute: impl FnOnce(&E) -> Result<F, String>,
    prove: impl FnOnce(F, &N) -> Result<R, String>,
) -> Result<R, String> {
    let (program, node) = (&programs[at.lv - 1][at.j], &slots[at.lv - 1][at.j]);
    let program = program.wait()?;
    if !early {
        node.wait()?;
    }
    let filled = execute(program)?;
    prove(filled, node.wait()?)
}

/// Whether the top streams: only if a child is still unproved when the top can
/// execute (its program emitted; with `W3_EXEC_EARLY=0`, its artifacts built
/// too). With every child already proved there is nothing to stream behind,
/// and the streamed path (its forward pass, then a wave per child) costs more
/// than one whole execution: at 1× the top's artifacts come after its last
/// child, and streaming it anyway cost 0.19 s (BIG 617 / 618). A failed child
/// counts as done: the whole path reports its error.
pub(crate) fn top_streams<R>(kids: &[&Published<R>]) -> bool {
    kids.iter().any(|k| k.0.get().is_none())
}

/// Calls `land` on each child's result as it is published, in the order they
/// arrive, each exactly once; a child that failed fails this with its error.
pub(crate) fn in_arrival_order<R>(
    kids: &[&Published<R>],
    mut land: impl FnMut(usize, &R) -> Result<(), String>,
) -> Result<(), String> {
    let mut landed = vec![false; kids.len()];
    while landed.contains(&false) {
        let mut moved = false;
        for (c, kid) in kids.iter().enumerate() {
            if landed[c] {
                continue;
            }
            match kid.0.get() {
                None => {}
                Some(Err(e)) => return Err(e.clone()),
                Some(Ok(r)) => {
                    land(c, r)?;
                    landed[c] = true;
                    moved = true;
                }
            }
        }
        if !moved {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    Ok(())
}

/// Every child's result, each waited for in order.
pub(crate) fn wait_all<'r, R>(kids: &[&'r Published<R>]) -> Result<Vec<&'r R>, String> {
    kids.iter().map(|k| k.wait()).collect()
}

/// The shared VRAM gate armed for the tree's sibling proofs while this lives
/// (`device_permit::arm_shared_gate`; inert with the knob off or at one
/// sibling), disarmed on drop, the error and unwind paths included.
struct SharedGateArmed(bool);

impl SharedGateArmed {
    fn arm(on: bool) -> Self {
        Self(on && super::device_permit::arm_shared_gate(true).is_some())
    }
}

impl Drop for SharedGateArmed {
    fn drop(&mut self) {
        if self.0 {
            super::device_permit::arm_shared_gate(false);
        }
    }
}

/// The tree proved the way a prover would run it, with the leaves' programs
/// emitted beforehand (`leaves`, while phase B ran):
/// 1. the leaves' artifacts, one after another on a thread of their own (each
///    holds the card);
/// 2. the nodes' programs and artifacts — functions of the leaves' artifacts,
///    not of any proof — on a thread of their own ([`build_nodes`]), each
///    published to its slot as it is built, beside
/// 3. every program proved `siblings` at a time, leaves first, each harvested
///    without its verify ([`prove_dataflow`]): with `beside`, each leaf
///    executes and fills its traces while the artifacts are built (neither
///    reads them) and waits for them only to prove; without, it waits for every
///    artifact first (the order before, `W3_EXEC_BESIDE_ARTIFACTS=0`); each
///    node as soon as its own children are proved and its program is in its
///    slot — with `dataflow` off (the control), once its whole level below is
///    proved; without `node_pipe` (that control), once the builder has built
///    the whole tree.
///
/// `node_pipe` is the builder's own pool ([`build_nodes`]); `early`, when
/// given, is a leaf that already executed and filled while phase B ran
/// ([`EarlyLeaf`]): its worker joins it instead.
///
/// Returns each level's timing and every proof with its artifacts, level by
/// level, the top last — for the harness to verify off the clock — and the
/// early leaf's arena, if any.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn prove_tree_pipelined(
    plan: &WhirBlockPlan,
    proof: &BlockWhirProof,
    leaves: Vec<std::sync::Arc<LfmProgram>>,
    siblings: usize,
    beside: bool,
    early: Option<EarlyLeaf>,
    node_pipe: Option<&rayon::ThreadPool>,
    dataflow: bool,
    stream_top: Option<&std::sync::OnceLock<bool>>,
    exec_early: bool,
) -> Result<
    (
        Vec<LevelTiming>,
        Vec<(super::registry::LfmArtifacts, LfmProof)>,
        Option<(usize, Vec<Vec<LfmWord>>, f64, f64, f64)>,
    ),
    String,
> {
    // The shared VRAM gate (`LAMBDA_VM_SHARED_VRAM_GATE=1`) is armed here, for
    // the tree's sibling proofs only: after the base, which keeps its retained
    // pool and never shares the card with the tree, so the arming calibrates
    // against a card the base has left; and disarmed when the tree returns or
    // unwinds, before anything (an off-the-clock verify) takes holds on rayon
    // workers again.
    let _shared_gate = SharedGateArmed::arm(siblings > 1);
    let early = std::sync::Mutex::new(early);
    let early_out: std::sync::Mutex<Option<(usize, Vec<Vec<LfmWord>>, f64, f64, f64)>> =
        std::sync::Mutex::new(None);
    let wrap = aggregation_wrap_options();
    let words = plan.child_layout().total();
    let t_tree = std::time::Instant::now();
    let shape = plan.levels();
    let mut timings = Vec::with_capacity(shape.len() + 1);
    let mut proofs = Vec::new();
    let built: Published<LeafBuilt> = Published::new();
    let leaf_built_at: Vec<std::sync::Mutex<f64>> =
        leaves.iter().map(|_| std::sync::Mutex::new(0.0)).collect();
    let slots: Vec<Vec<Published<TreeNode>>> = shape
        .iter()
        .map(|l| l.arities.iter().map(|_| Published::new()).collect())
        .collect();
    let programs: Vec<Vec<Published<NodeProgram>>> = shape
        .iter()
        .map(|l| l.arities.iter().map(|_| Published::new()).collect())
        .collect();

    let order = tree_order(leaves.len(), &shape);
    let results: Vec<Vec<Published<ProvedProgram>>> = std::iter::once(leaves.len())
        .chain(shape.iter().map(|l| l.arities.len()))
        .map(|n| (0..n).map(|_| Published::new()).collect())
        .collect();

    let n_leaves = std::thread::scope(|outer| -> Result<usize, String> {
        // 2. the nodes' programs and artifacts, into their slots.
        let builder = outer.spawn(|| {
            build_nodes(
                plan, &shape, &built, &programs, &slots, &wrap, words, t_tree, node_pipe,
            )
        });
        std::thread::scope(|scope| {
            // 1. the leaves' artifacts, one after another on this thread: each
            // build holds the card throughout (armed), so building them as
            // rayon jobs bought no overlap, and a holder on a rayon worker can
            // run a sibling build that takes the card again (BIG 569).
            scope.spawn(|| {
                let guard = PublishGuard(&built);
                guard.publish(
                    leaves
                        .iter()
                        .zip(&leaf_built_at)
                        .map(|(program, at)| -> Result<_, String> {
                            let t = std::time::Instant::now();
                            let artifacts = artifacts_of(program, &wrap);
                            let derived = DerivedChild::from_artifacts(&artifacts, &wrap, words)?;
                            *at.lock().map_err(|_| "a built-at stamp is poisoned")? =
                                t_tree.elapsed().as_secs_f64();
                            Ok((artifacts, derived, t.elapsed().as_secs_f64()))
                        })
                        .collect::<Result<_, String>>(),
                );
            });
            // 3. every program of the tree, `siblings` at a time, leaves first;
            // each node as soon as its children are proved and its program is
            // in its slot ([`prove_dataflow`]).
            prove_dataflow(
                &shape,
                &order,
                &results,
                siblings,
                !dataflow,
                |at, kids: &[&Published<ProvedProgram>]| -> Result<ProvedProgram, String> {
                    let start = t_tree.elapsed().as_secs_f64();
                    if at.lv == 0 {
                        let k = at.j;
                        let ahead = {
                            let mut slot =
                                early.lock().map_err(|_| "the early slot is poisoned")?;
                            if slot.as_ref().is_some_and(|e| e.leaf == k) {
                                slot.take()
                            } else {
                                None
                            }
                        };
                        let t = std::time::Instant::now();
                        let filled = match ahead {
                            Some(e) => {
                                let (filled, arena, started, secs) =
                                    e.handle.join().map_err(|_| {
                                        format!("leaf {k}: the early executor panicked")
                                    })??;
                                *early_out
                                    .lock()
                                    .map_err(|_| "the early readout is poisoned")? =
                                    Some((k, arena, e.complete_at, started, secs));
                                filled
                            }
                            None => {
                                let arenas = block_leaf_arena(plan, proof, k)?;
                                if !beside {
                                    built.wait()?;
                                }
                                lfm_execute_and_fill(
                                    &leaves[k],
                                    &arenas,
                                    crate::hash_pin::BLOCK_HASHER,
                                )
                                .map_err(|e| format!("leaf {k}: {e:?}"))?
                            }
                        };
                        let artifacts = &built.wait()?[k].0;
                        let lfm = filled
                            .prove(artifacts, &wrap, decide_lfm_residency())
                            .map_err(|e| format!("leaf {k}: {e:?}"))?;
                        let prove = t.elapsed().as_secs_f64();
                        let times = ProgramTimes {
                            program_at: 0.0,
                            built_at: 0.0,
                            start,
                            end: t_tree.elapsed().as_secs_f64(),
                            split: prove_split_now(),
                        };
                        let child = harvest_child(artifacts.clone(), wrap.clone(), &lfm)?;
                        return Ok(ProvedProgram {
                            lfm,
                            child,
                            prove,
                            built: 0.0,
                            times,
                        });
                    }
                    // The control of the node pipeline: no node proves before
                    // the whole tree is built, as when the builder joined level 0.
                    if node_pipe.is_none() {
                        for slot in slots.iter().flatten() {
                            slot.wait()?;
                        }
                    }
                    // Execute and fill from the program as soon as it is
                    // emitted, prove once the node's own artifacts are built
                    // ([`node_flow`]); traces filled under the block hasher,
                    // which the node's artifacts are built for (the prove
                    // asserts it).
                    let hasher = crate::hash_pin::BLOCK_HASHER;
                    node_flow(
                        &programs,
                        &slots,
                        at,
                        exec_early,
                        |program: &NodeProgram| -> Result<_, String> {
                            let t = std::time::Instant::now();
                            // The top streams only if a child is still unproved
                            // now that it can execute; the choice is kept for
                            // the readout.
                            let streamed = match stream_top {
                                Some(chose) if at.lv == shape.len() => {
                                    let streams = top_streams(kids);
                                    let _ = chose.set(streams);
                                    streams
                                }
                                _ => false,
                            };
                            let filled = if streamed {
                                execute_streamed(&program.program, kids, hasher)
                                    .map_err(|e| format!("the top, streamed: {e}"))?
                            } else {
                                let kids = wait_all(kids)?;
                                let mut arenas: Vec<Vec<LfmWord>> = Vec::new();
                                for k in &kids {
                                    arenas.extend(try_child_arena_words(&k.child)?);
                                }
                                lfm_execute_and_fill(&program.program, &arenas, hasher)
                                    .map_err(|e| format!("level {} node {}: {e:?}", at.lv, at.j))?
                            };
                            Ok((filled, t, program.at))
                        },
                        |(filled, t, program_at), node: &TreeNode| {
                            let lfm = filled
                                .prove(&node.artifacts, &wrap, decide_lfm_residency())
                                .map_err(|e| format!("level {} node {}: {e:?}", at.lv, at.j))?;
                            let prove = t.elapsed().as_secs_f64();
                            let times = ProgramTimes {
                                program_at,
                                built_at: node.built_at,
                                start,
                                end: t_tree.elapsed().as_secs_f64(),
                                split: prove_split_now(),
                            };
                            let child = harvest_child(node.artifacts.clone(), wrap.clone(), &lfm)?;
                            Ok(ProvedProgram {
                                lfm,
                                child,
                                prove,
                                built: node.built,
                                times,
                            })
                        },
                    )
                },
            );
        });
        let leaf_built = built.wait()?;
        builder
            .join()
            .map_err(|_| "the node builder panicked".to_string())?;
        Ok(leaf_built.len())
    })?;
    // Each level's readout: a level's wall is from the level below's last
    // proof to its own (level 0: from the tree's start), now that a node may
    // prove before its level below is done.
    let leaf_secs: Vec<f64> = built.wait()?.iter().map(|(_, _, s)| *s).collect();
    let mut level_proofs: Vec<Vec<LfmProof>> = Vec::with_capacity(results.len());
    let mut below_end = 0.0f64;
    for (lv, level) in results.into_iter().enumerate() {
        let proved: Vec<ProvedProgram> = level
            .into_iter()
            .map(Published::take)
            .collect::<Result<_, String>>()?;
        let mut times = Vec::with_capacity(proved.len());
        let mut programs = Vec::with_capacity(proved.len());
        for (j, p) in proved.iter().enumerate() {
            let built = match lv {
                0 => leaf_secs[j],
                _ => p.built,
            };
            let built_at = match lv {
                0 => *leaf_built_at[j]
                    .lock()
                    .map_err(|_| "a built-at stamp is poisoned")?,
                _ => p.times.built_at,
            };
            times.push(ProgramTimes {
                built_at,
                ..p.times
            });
            programs.push((built, p.prove));
        }
        let end = proved.iter().map(|p| p.times.end).fold(0.0, f64::max);
        timings.push(LevelTiming {
            wall: end - below_end,
            programs,
            times,
        });
        below_end = end;
        level_proofs.push(proved.into_iter().map(|p| p.lfm).collect());
    }
    if level_proofs.last().map(Vec::len) != Some(1) {
        return Err(format!(
            "the tree closes to {:?} nodes",
            level_proofs.last().map(Vec::len)
        ));
    }
    // Off the clock: every proof with its artifacts, level by level.
    let mut level_proofs = level_proofs.into_iter();
    let leaf_lfms = level_proofs.next().unwrap_or_default();
    debug_assert_eq!(leaf_lfms.len(), n_leaves);
    for ((artifacts, _, _), lfm) in built.take()?.into_iter().zip(leaf_lfms) {
        proofs.push((artifacts, lfm));
    }
    for (level, lfms) in slots.into_iter().zip(level_proofs) {
        for (slot, lfm) in level.into_iter().zip(lfms) {
            proofs.push((slot.take()?.artifacts, lfm));
        }
    }
    let early_out = early_out
        .into_inner()
        .map_err(|_| "the early readout is poisoned")?;
    Ok((timings, proofs, early_out))
}

// ================================ the driver ==============================

/// The whole block's walls, in seconds.
#[derive(Clone, Debug, Default)]
pub struct WhirTreeTimes {
    pub base: f64,
    /// The tree, from its start to the top's proof.
    pub tree: f64,
    /// From before the base to the top's proof, the readouts inside.
    pub whole: f64,
    /// The readouts printed between the base and the tree (on the clock).
    pub readouts: f64,
}

/// The early leaf's readout: its index, its arena (for the harness's check
/// against the finished proof's), when its groups were complete, when its
/// execute + fill started and how long it took (seconds).
#[cfg(test)]
pub(crate) type EarlyReadout = (usize, Vec<Vec<LfmWord>>, f64, f64, f64);

/// What a whole W3 block run hands back.
pub struct WhirTreeRun {
    /// The block's statement, as a consumer receives it with the top proof.
    pub statement: block_whir::OwnedBlockStatement,
    /// Every proof of the tree with its artifacts, level by level, the top
    /// last.
    pub proofs: Vec<(super::registry::LfmArtifacts, LfmProof)>,
    pub times: WhirTreeTimes,
    /// The base proof (the harness verifies it, and the early leaf's arena
    /// against it, off the clock).
    #[cfg(test)]
    pub(crate) base_proof: BlockWhirProof,
    #[cfg(test)]
    pub(crate) plan: WhirBlockPlan,
    #[cfg(test)]
    pub(crate) early: Option<EarlyReadout>,
}

impl WhirTreeRun {
    /// The top node's proof.
    pub fn top(&self) -> Option<&LfmProof> {
        self.proofs.last().map(|(_, p)| p)
    }
}

/// What the planner thread hands back: the plan, the leaves' programs, when
/// the statement came and when the programs were ready, and the early leaf.
type Planned = (
    WhirBlockPlan,
    Vec<std::sync::Arc<LfmProgram>>,
    f64,
    f64,
    Option<EarlyLeaf>,
);

/// ★ W3 on a real block, run the way a prover runs it:
/// - the block proved in one proof at the production format;
/// - the moment its statement is final (after phase A), the plan derived and
///   the leaves' programs emitted on threads of their own, beside phase B;
/// - then the tree ([`prove_tree_pipelined`]): the leaves `siblings` at a time,
///   the nodes' programs built beside them.
///
/// Prints the harness's `W3 …` lines, up to `W3 RECURSION`. A failure anywhere
/// is an error.
pub fn prove_whir_block_tree(
    elf: &[u8],
    input: &[u8],
    cfg: &WhirTreeConfig,
    sink: &dyn WhirTreeSink,
) -> Result<WhirTreeRun, String> {
    let (leaves, siblings, fan_in, beside, early_on) = (
        cfg.leaves,
        cfg.siblings,
        cfg.fan_in,
        cfg.beside,
        cfg.early_on,
    );
    let (argue, format, options) = (cfg.argue, cfg.format, &cfg.options);
    sink.line(&posture_line());
    sink.line(&format!("W3 ARGUE: {argue:?}"));
    sink.line(&format!("BLOCK UPLOAD AHEAD: {}", options.upload_ahead));
    sink.line(&format!(
        "BLOCK REST LAYOUT CONFIG: waves of {} · KECCAK_RND built as its tables {} · finish packed {}",
        options
            .rest_layout_bytes
            .map_or("all at once".to_string(), |b| format!("{} MiB", b >> 20)),
        options.finish_keccak_rnd_chunks,
        options.pack_finished,
    ));
    let opts = super::proof::block_base_options();
    super::device_permit::arm(siblings);
    sink.line(&format!(
        "W3 CONFIG: leaves {leaves:?} · siblings {siblings} · fan-in {fan_in} · KECCAK_RND streamed {} · MEMW LT streamed {} · layout workers {} (ahead {:?}) · rest packed as laid out {} · streamed ops dropped {}",
        options.stream_keccak_rnd,
        options.stream_memw_lt,
        options.layout_workers,
        options.layout_ahead,
        options.pack_rest_as_laid_out,
        options.drop_streamed_ops
    ));

    let t0 = std::time::Instant::now();
    // The wall clock at the prove's start, to place a box's memory samples.
    sink.line(&format!(
        "W3 PROVE START: unix {:.3}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64())
    ));
    let sender = std::sync::Mutex::new(None::<std::sync::mpsc::Sender<_>>);
    let (tx, rx) = std::sync::mpsc::channel::<(
        block_whir::OwnedBlockStatement,
        Vec<block_whir::PreparedRoots>,
    )>();
    *sender.lock().unwrap_or_else(|e| e.into_inner()) = Some(tx);
    let observe = |statement: block_whir::BlockStatement<'_>,
                   roots: &[block_whir::PreparedRoots]| {
        if let Some(tx) = sender.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            let _ = tx.send((statement.to_owned(), roots.to_vec()));
        }
    };
    // Each group's share of the proof as its opening ends, for the early leaf.
    let group_sender = std::sync::Mutex::new(None::<std::sync::mpsc::Sender<GroupMsg>>);
    let (gtx, grx) = std::sync::mpsc::channel::<GroupMsg>();
    // Without the early leaf no sender is kept, so the planner's loop ends
    // at once.
    *group_sender.lock().unwrap_or_else(|e| e.into_inner()) = early_on.then_some(gtx);
    let on_group = |opened: stark::multilinear_block::GroupOpened<'_, _, _>| {
        if let Some(tx) = group_sender
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            let _ = tx.send(GroupMsg {
                group: opened.group,
                roots: opened.roots.to_vec(),
                tables: opened.tables.to_vec(),
                argue: opened.argue.cloned(),
                opening: opened.opening.clone(),
                prepared: opened.prepared.cloned(),
                at: t0.elapsed().as_secs_f64(),
            });
        }
    };
    let (elf_ref, opts_ref, format_ref) = (elf, &opts, &format);
    let (proved, pre) = std::thread::scope(|scope| {
        let pre = scope.spawn(move || -> Result<Planned, String> {
            let (statement, roots) = rx
                .recv()
                .map_err(|_| "the prover never stated".to_string())?;
            let at = t0.elapsed().as_secs_f64();
            let plan = WhirBlockPlan::derive_with(
                elf_ref,
                opts_ref,
                format_ref,
                statement.view(),
                leaves,
                fan_in,
                Some(&roots),
            )?;
            // One plain thread a leaf, off the rayon pool phase B is using.
            let programs: Vec<LfmProgram> = std::thread::scope(|inner| {
                let handles: Vec<_> = (0..plan.partition().num_leaves())
                    .map(|k| {
                        let plan = &plan;
                        inner.spawn(move || plan.leaf_program(k))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join()
                            .map_err(|_| "a leaf emitter panicked".to_string())?
                    })
                    .collect::<Result<_, String>>()
            })?;
            let programs: Vec<std::sync::Arc<LfmProgram>> =
                programs.into_iter().map(std::sync::Arc::new).collect();
            let ready = t0.elapsed().as_secs_f64();
            // The groups as phase B finishes them (none unless `early_on`):
            // the first leaf complete while a group is still to come runs
            // its execute + fill now, on a thread of its own.
            let mut early = None;
            let mut words: Vec<Option<Vec<LfmWord>>> =
                (0..plan.num_groups()).map(|_| None).collect();
            let mut block_roots = Vec::new();
            let mut done = 0usize;
            while let Ok(msg) = grx.recv() {
                done += 1;
                let g = msg.group;
                let slot = words.get_mut(g).ok_or("a group the plan does not hold")?;
                *slot = Some(group_arena_words(
                    &plan,
                    g,
                    &msg.tables,
                    msg.argue.as_ref(),
                    &msg.opening,
                    msg.prepared.as_ref(),
                )?);
                if block_roots.is_empty() {
                    block_roots = msg.roots;
                }
                if early.is_some() || done >= plan.num_groups() {
                    continue;
                }
                let partition = plan.partition();
                let complete = (0..partition.num_leaves())
                    .find(|&k| partition.leaf(k).iter().all(|&g| words[g].is_some()));
                if let Some(k) = complete {
                    let groups: Vec<Vec<LfmWord>> = partition
                        .leaf(k)
                        .iter()
                        .map(|&g| words[g].clone().unwrap_or_default())
                        .collect();
                    let arena = leaf_arena(&block_roots, groups);
                    let program = std::sync::Arc::clone(&programs[k]);
                    let handle = std::thread::spawn(move || -> Result<EarlyOut, String> {
                        let started = t0.elapsed().as_secs_f64();
                        let t = std::time::Instant::now();
                        let filled =
                            lfm_execute_and_fill(&program, &arena, crate::hash_pin::BLOCK_HASHER)
                                .map_err(|e| format!("early leaf {k}: {e:?}"))?;
                        Ok((filled, arena, started, t.elapsed().as_secs_f64()))
                    });
                    early = Some(EarlyLeaf {
                        leaf: k,
                        complete_at: msg.at,
                        handle,
                    });
                }
            }
            Ok((plan, programs, at, ready, early))
        });
        let proved = block_whir::prove_block_whir_observed_groups(
            elf, input, &opts, &format, options, &observe, &on_group,
        );
        // A prover that failed before stating must not leave the planner waiting.
        *sender.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *group_sender.lock().unwrap_or_else(|e| e.into_inner()) = None;
        (proved, pre.join())
    });
    let (proof, stamps) = proved.map_err(|e| format!("the block proves: {e}"))?;
    let base = t0.elapsed().as_secs_f64();
    let (plan, programs, stated_at, ready_at, early) = pre
        .map_err(|_| "the planner panicked".to_string())?
        .map_err(|e| format!("the plan and the leaves derive: {e}"))?;
    sink.write(&stamps.report());
    // LT's chunk heights (what its per-chunk deduplication leaves): equal across
    // runs of one setting when the chunking is deterministic.
    {
        let frame = block_whir::block_frame(proof.statement(), elf, &opts, &format)
            .map_err(|e| format!("the statement's frame: {e}"))?;
        let refs = frame.airs.air_refs();
        let lt: Vec<u8> = refs
            .iter()
            .zip(&proof.table_num_vars)
            .filter(|(air, _)| air.name().starts_with("LT["))
            .map(|(_, &n)| n)
            .collect();
        sink.line(&format!("W3 LT HEIGHTS: {lt:?}"));
        sink.line(&format!(
            "W3 CHUNKED: {} (cuts KECCAK 2^{} · ECSM 2^{})",
            block_whir::chunked_census(&proof.table_counts, &proof.table_num_vars),
            options.keccak_rows_log2,
            options.ecsm_rows_log2,
        ));
        // Each group's tables by AIR, in the group's order: which tables the
        // card waited for, and where the layout's chunks went.
        for (g, group) in proof.groups.iter().enumerate() {
            let mut kinds: Vec<(String, usize)> = Vec::new();
            for &index in group {
                let name = refs[index as usize].name();
                let kind = name.split('[').next().unwrap_or(name).to_string();
                match kinds.last_mut() {
                    Some((last, n)) if *last == kind => *n += 1,
                    _ => kinds.push((kind, 1)),
                }
            }
            let kinds: Vec<String> = kinds
                .into_iter()
                .map(|(kind, n)| if n == 1 { kind } else { format!("{kind}×{n}") })
                .collect();
            sink.line(&format!("W3 GROUP TABLES {g}: {}", kinds.join(" ")));
        }
    }
    sink.line(&format!(
        "W3 BASE: {base:.2}s · statement at {stated_at:.2}s · plan + {} leaves emitted by {ready_at:.2}s ({})",
        programs.len(),
        if ready_at <= base {
            "inside the base"
        } else {
            "after the base"
        }
    ));
    sink.line(&format!(
        "W3 PLAN: {} groups · costs {:?} (Σ {}) · {} leaves {:?} · {} prepared",
        plan.num_groups(),
        plan.costs(),
        plan.costs().iter().sum::<usize>(),
        plan.partition().num_leaves(),
        plan.partition().leaves(),
        plan.prepared().len()
    ));
    // Each leaf's chips, real / padded rows: how far each sits from its next
    // doubling, and which one doubled when the in-guest work moves.
    for (k, program) in programs.iter().enumerate() {
        let chips: Vec<String> =
            super::airs::lfm_chip_census_with_hasher(program, crate::hash_pin::BLOCK_HASHER)
                .iter()
                .filter(|c| c.real_rows > 0)
                .map(|c| format!("{}={}/{}", c.name, c.real_rows, c.rows))
                .collect();
        sink.line(&format!("W3 LEAF CENSUS {k}: {}", chips.join(" ")));
    }

    // The readouts above run on the clock, between the base and the tree: a
    // prover prints none of them, so the whole block is also given without.
    let readouts = t0.elapsed().as_secs_f64() - base;
    let node_pipe = match cfg.node_pipe {
        Some(threads) => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("w3-emit-{i}"))
                .build()
                .map_err(|e| format!("the node builder's pool: {e}"))?,
        ),
        None => None,
    };
    let (dataflow, stream_top, exec_early) = (cfg.dataflow, cfg.stream_top, cfg.exec_early);
    let top_streamed = std::sync::OnceLock::new();
    let t = std::time::Instant::now();
    let tree_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    let (timings, proofs, early_out) = prove_tree_pipelined(
        &plan,
        &proof,
        programs,
        siblings,
        beside,
        early,
        node_pipe.as_ref(),
        dataflow,
        stream_top.then_some(&top_streamed),
        exec_early,
    )
    .map_err(|e| format!("the tree proves: {e}"))?;
    let tree = t.elapsed().as_secs_f64();
    let whole = t0.elapsed().as_secs_f64();
    for (lv, level) in timings.iter().enumerate() {
        let programs: Vec<String> = level
            .programs
            .iter()
            .map(|(built, prove)| format!("{built:.2}+{prove:.2}"))
            .collect();
        sink.line(&format!(
            "W3 LEVEL {lv}: {:.2}s wall · per program build+prove {}",
            level.wall,
            programs.join(" · ")
        ));
    }
    // `W3_NODE_TIMES=1`: when each program of the tree was built, taken and
    // proved (seconds since the tree started), its prove's split, and for each
    // node level when the level below was done and its nodes were built. Off
    // the clock; the stamps are taken either way.
    if cfg.node_times {
        sink.line(&format!(
            "W3 TIMES TREE START: unix {tree_unix:.3} (the card trace's clock)"
        ));
        let shape = plan.levels();
        let split = |t: &ProgramTimes| match t.split {
            Some((x, f, p, w)) => format!("exec {x:.2} fill {f:.2} prove {p:.2} wait {w:.2}"),
            None => "no split".to_string(),
        };
        for (lv, level) in timings.iter().enumerate() {
            let mut at = 0usize;
            let rows: Vec<String> = level
                .times
                .iter()
                .enumerate()
                .map(|(j, t)| {
                    let landed = match lv {
                        0 => String::new(),
                        _ => {
                            let a = shape[lv - 1].arities[j];
                            let kids: Vec<String> = timings[lv - 1].times[at..at + a]
                                .iter()
                                .map(|c| format!("{:.2}", c.end))
                                .collect();
                            at += a;
                            format!(" kids done [{}]", kids.join(" "))
                        }
                    };
                    let program = match lv {
                        0 => String::new(),
                        _ => format!(" program@{:.2}", t.program_at),
                    };
                    format!(
                        "{j}: built@{:.2}{landed} took@{:.2} done@{:.2} ({}){program}",
                        t.built_at,
                        t.start,
                        t.end,
                        split(t)
                    )
                })
                .collect();
            sink.line(&format!("W3 TIMES L{lv}: {}", rows.join(" · ")));
            if lv > 0 {
                let last_kid = timings[lv - 1]
                    .times
                    .iter()
                    .map(|t| t.end)
                    .fold(0.0, f64::max);
                let built = level.times.iter().map(|t| t.built_at).fold(0.0, f64::max);
                let first = level
                    .times
                    .iter()
                    .map(|t| t.start)
                    .fold(f64::INFINITY, f64::min);
                sink.line(&format!(
                    "W3 TIMES START L{lv}: level {} done@{last_kid:.2} · its nodes built by {built:.2} · first node took@{first:.2} ({:+.2} after the later)",
                    lv - 1,
                    first - last_kid.max(built)
                ));
            }
        }
    }
    sink.line(&format!(
        "W3 STREAM TOP: {}",
        if stream_top {
            "on (while a child is unproved as its program arrives, the top executes each child's share as that child is proved)"
        } else {
            "off (the top executes after its last child)"
        }
    ));
    sink.line(&format!(
        "W3 EXEC EARLY: {}",
        if exec_early {
            "on (a node executes and fills from its program as it is emitted; only its prove waits for its artifacts)"
        } else {
            "off (a node executes once its program and its artifacts are both built)"
        }
    ));
    sink.line(&format!(
        "W3 TOP EXECUTE: {}",
        match (stream_top, top_streamed.get()) {
            (false, _) => "whole (streaming off)",
            (true, Some(true)) =>
                "streamed (a child was still unproved when the top could execute)",
            (true, Some(false)) => "whole (every child was proved before the top could execute)",
            (true, None) => "unknown (the top never chose)",
        }
    ));
    sink.line(&format!(
        "W3 DATAFLOW: {}",
        if dataflow {
            "on (each node proves as soon as its own children are proved)"
        } else {
            "off (each node waits for its whole level below)"
        }
    ));
    sink.line(&format!(
        "W3 NODE PIPE: {}",
        match &node_pipe {
            Some(pool) => format!(
                "on ({} builder threads; each level proves as its own nodes arrive)",
                pool.current_num_threads()
            ),
            None => "off (one builder thread; level 1 waits for the whole tree)".to_string(),
        }
    ));
    sink.line(&format!(
        "W3 RECURSION: {:.2}s after the base (tree {tree:.2}s) · whole block {whole:.2}s · whole excl. harness readouts {:.2}s (readouts {readouts:.2}s) · leaves execute beside their artifacts {beside}",
        whole - base,
        whole - readouts
    ));
    if proofs.is_empty() {
        return Err("the tree has no top".to_string());
    }
    // The early leaf's arena is checked against the proof's by the harness.
    #[cfg(not(test))]
    drop(early_out);
    Ok(WhirTreeRun {
        statement: proof.statement().to_owned(),
        proofs,
        times: WhirTreeTimes {
            base,
            tree,
            whole,
            readouts,
        },
        #[cfg(test)]
        base_proof: proof,
        #[cfg(test)]
        plan,
        #[cfg(test)]
        early: early_out,
    })
}

// ============================== the proof file ============================

/// The proof file's first bytes (the STARK block tree's file shares them).
const PROOF_MAGIC: [u8; 8] = *b"LVMBLKTR";

/// The proof file's layout version.
const PROOF_VERSION: u32 = 1;

/// Which tree a proof file holds: the STARK block tree (0) or this one. A
/// verifier refuses the other's file by its tag, before reading it as its own.
pub const PIPELINE_WHIR: u8 = 1;

/// A whole W3 block's proof as a consumer receives it: the block's statement
/// and the top node's proof. The block verifier ([`verify_whir_block_tree_proof`])
/// takes the statement as a claim — the plan and the top program are derived
/// from the trusted ELF and the statement, under the block presets — so
/// nothing here is a format parameter: the file only carries what
/// [`super::whir_block::verify_block_tree`] already takes from its caller.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct WhirBlockTreeProof {
    magic: [u8; 8],
    version: u32,
    pipeline: u8,
    pub table_num_vars: Vec<u8>,
    pub runtime_page_ranges: Vec<crate::RuntimePageRange>,
    pub table_counts: crate::TableCounts,
    pub public_output: Vec<u8>,
    pub num_private_input_pages: usize,
    pub groups: Vec<Vec<u32>>,
    pub top_proof: stark::proof::stark::MultiProof<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
        (),
    >,
    pub top_public_words: Vec<(u32, LfmWord)>,
}

impl WhirBlockTreeProof {
    /// The file for a W3 block's statement and its top proof.
    pub fn new(statement: block_whir::OwnedBlockStatement, top: LfmProof) -> Self {
        Self {
            magic: PROOF_MAGIC,
            version: PROOF_VERSION,
            pipeline: PIPELINE_WHIR,
            table_num_vars: statement.table_num_vars,
            runtime_page_ranges: statement.runtime_page_ranges,
            table_counts: statement.table_counts,
            public_output: statement.public_output,
            num_private_input_pages: statement.num_private_input_pages,
            groups: statement.groups,
            top_proof: top.proof,
            top_public_words: top.public_words,
        }
    }

    /// The claimed statement.
    pub fn statement(&self) -> block_whir::BlockStatement<'_> {
        block_whir::BlockStatement {
            table_num_vars: &self.table_num_vars,
            runtime_page_ranges: &self.runtime_page_ranges,
            table_counts: &self.table_counts,
            public_output: &self.public_output,
            num_private_input_pages: self.num_private_input_pages,
            groups: &self.groups,
        }
    }

    /// The file's bytes.
    pub fn to_bytes(&self) -> Result<rkyv::util::AlignedVec, String> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self).map_err(|e| format!("serialize: {e}"))
    }

    /// A file read back: refused unless it is this layout's W3 block tree.
    /// `bytes` must be aligned for rkyv (an `AlignedVec`).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let proof = rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes)
            .map_err(|e| format!("not a W3 block proof file: {e}"))?;
        if proof.magic != PROOF_MAGIC {
            return Err("not a block proof file (magic)".to_string());
        }
        if proof.version != PROOF_VERSION {
            return Err(format!(
                "block proof file version {}, this build reads {PROOF_VERSION}",
                proof.version
            ));
        }
        if proof.pipeline != PIPELINE_WHIR {
            return Err(format!(
                "the file holds pipeline {} ({}), not the WHIR block tree",
                proof.pipeline,
                if proof.pipeline == 0 {
                    "the STARK block tree"
                } else {
                    "unknown"
                }
            ));
        }
        Ok(proof)
    }
}

/// ★ A W3 block proof file verified: [`super::whir_block::verify_block_tree`]
/// over its claimed statement and its top proof, under the block presets,
/// against the trusted `elf_bytes`.
pub fn verify_whir_block_tree_proof(
    elf_bytes: &[u8],
    proof: &WhirBlockTreeProof,
) -> Result<(), String> {
    let top = LfmProof {
        proof: proof.top_proof.clone(),
        public_words: proof.top_public_words.clone(),
    };
    super::whir_block::verify_block_tree(elf_bytes, proof.statement(), &top)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The posture line names every posture knob, in the table's order, and
    /// the WHIR hash (a format knob both sides must share) is one of them.
    #[test]
    fn the_posture_line_names_every_knob() {
        let line = posture_line();
        assert!(line.starts_with("BLOCK POSTURE: "));
        let mut at = 0;
        for (name, _) in POSTURE {
            let found = line[at..]
                .find(name)
                .unwrap_or_else(|| panic!("{name} missing or out of order in `{line}`"));
            at += found + name.len();
        }
        assert!(POSTURE.contains(&("LAMBDA_VM_WHIR_HASH", "rpx")));
    }

    /// A synthetic statement and an empty top proof.
    fn synthetic_proof() -> WhirBlockTreeProof {
        let counts = crate::TableCounts {
            cpu: 3,
            lt: 1,
            memw: 2,
            memw_aligned: 1,
            load: 1,
            mul: 1,
            dvrm: 1,
            shift: 1,
            branch: 1,
            memw_register: 1,
            eq: 1,
            bytewise: 1,
            store: 1,
            cpu32: 1,
            keccak: 2,
            keccak_rnd: 4,
            ecsm: 1,
            ecdas: 1,
            hint: 0,
            commit: 1,
            blake3: 0,
        };
        let statement = block_whir::OwnedBlockStatement {
            table_num_vars: vec![21, 16, 6],
            runtime_page_ranges: vec![crate::RuntimePageRange {
                base: 0x7000_0000,
                count: 3,
            }],
            table_counts: counts,
            public_output: vec![1, 2, 3, 4, 5],
            num_private_input_pages: 2,
            groups: vec![vec![0, 2], vec![1]],
        };
        let words: Vec<(u32, LfmWord)> = (0..3u32)
            .map(|i| (i, [crate::tables::types::FE::from(u64::from(i) + 7); 4]))
            .collect();
        let top = LfmProof {
            proof: stark::proof::stark::MultiProof { proofs: vec![] },
            public_words: words,
        };
        WhirBlockTreeProof::new(statement, top)
    }

    /// The proof file reads back what was written; another tag, version or
    /// magic, or bytes that are not a file, are refused.
    #[test]
    fn the_proof_file_round_trips_and_refuses_another_file() {
        let proof = synthetic_proof();
        let bytes = proof.to_bytes().expect("serializes");
        let back = WhirBlockTreeProof::from_bytes(&bytes).expect("reads back");
        assert_eq!(
            format!("{:?}", back.statement()),
            format!("{:?}", proof.statement())
        );
        assert_eq!(back.top_public_words, proof.top_public_words);
        assert_eq!(back.top_proof.proofs.len(), 0);

        for (what, tamper) in [
            (
                "pipeline",
                (|p: &mut WhirBlockTreeProof| p.pipeline = 0) as fn(&mut WhirBlockTreeProof),
            ),
            ("version", |p| p.version = PROOF_VERSION + 1),
            ("magic", |p| p.magic[0] ^= 1),
        ] {
            let mut other = synthetic_proof();
            tamper(&mut other);
            let bytes = other.to_bytes().expect("serializes");
            assert!(
                WhirBlockTreeProof::from_bytes(&bytes).is_err(),
                "a file with another {what} is refused"
            );
        }
        let mut junk = rkyv::util::AlignedVec::<16>::new();
        junk.extend_from_slice(&[0u8; 7]);
        assert!(
            WhirBlockTreeProof::from_bytes(&junk).is_err(),
            "seven zero bytes"
        );
    }

    /// Only the production tree is at the presets: the plan's leaf count, the
    /// block fan-in and the production argue.
    #[test]
    fn only_the_production_tree_is_at_the_presets() {
        let production = WhirTreeConfig {
            leaves: None,
            siblings: 3,
            fan_in: BLOCK_FAN_IN,
            beside: true,
            early_on: true,
            argue: BlockFormat::production().argue,
            format: BlockFormat::production(),
            options: BlockOptions::production(),
            node_pipe: Some(4),
            dataflow: true,
            stream_top: true,
            exec_early: true,
            node_times: false,
        };
        assert!(production.at_presets());
        let off = |f: fn(&mut WhirTreeConfig)| {
            let mut c = production.clone();
            f(&mut c);
            c.at_presets()
        };
        assert!(!off(|c| c.leaves = Some(2)), "a forced leaf count");
        assert!(!off(|c| c.fan_in = BLOCK_FAN_IN + 1), "another fan-in");
        let other = match BlockFormat::production().argue {
            ArgueFormat::PerTable => ArgueFormat::BATCHED,
            _ => ArgueFormat::PerTable,
        };
        let mut c = production.clone();
        c.argue = other;
        assert!(!c.at_presets(), "another argue");
        assert!(
            off(|c| c.siblings = 6),
            "the sibling count is scheduling only"
        );
    }
}
