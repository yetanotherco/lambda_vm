//! ★ The no-epoch block, base to top node: the one driver the whole-block
//! harness (`block_tree_tests::the_block_tree_composes_to_a_top_node`) and the
//! CLI's `prove-block` both run, so their numbers cannot diverge.
//!
//! [`prove_block_tree`] proves the block ([`crate::block::prove_block_observed`]),
//! harvests it against the plan ([`harvest_block_over`]), proves the leaves
//! (level 0), the interior and the top, and checks what the harness always
//! checked on the clock: the shape handed out before the prove is the proof's,
//! every leaf publishes the host's words, the base and every child verify (on
//! helper threads, joined before anything is reported), the top claims the
//! block and verifies against its emitted program. Every line goes through a
//! [`BlockTreeSink`]: the harness prints to stdout, the CLI to stderr.
//!
//! Every knob is read once, by [`BlockTreeConfig::from_env`], under the names
//! and defaults the harness always read. [`POSTURE`] is the production posture
//! the measurements run under; the CLI sets it for knobs the environment
//! leaves unset, and the driver's first line says how the running process
//! compares with it.

use std::sync::Arc;
use std::time::Instant;

use stark::config::Commitment;
use stark::proof::view::MultiProofView;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::LfmArtifacts;
use super::block_leaf::BlockPartition;
use super::block_plan::{BlockShape, BlockTreePlan, ElfConstants, le_halves, top_claims};
use super::block_tree_pipeline::Pipe;
use super::compiler::LfmProgram;
use super::harvest::{HarvestedChild, HostTable, TableLegs};
use super::per_table_aggregator::DerivedChild;
use super::proof::LfmProof;
use super::tree_run::{HostSampler, cgroup_limit_gib, in_index_order};
use super::word::{LfmWord, base_word, ext_word};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;

// ================================ the sink ================================

/// Where the driver's lines go, and the two hooks a caller may want.
pub trait BlockTreeSink: Send + Sync {
    /// Whole lines, each ending in `\n`, written at once.
    fn write(&self, text: &str);

    /// One line.
    fn line(&self, line: &str) {
        self.write(&format!("{line}\n"));
    }

    /// The base proof, right after it proves (inside the whole run's clock).
    fn base_proved(&self, _proof: &crate::VmProof) {}

    /// A note appended to the `BLOCK PARTITION SOURCE` line, about the plan's
    /// partition of the instances `names`.
    fn partition_note(&self, _names: &[&str], _partition: &BlockPartition) -> Option<String> {
        None
    }
}

/// Lines to stdout (the harness's).
pub struct StdoutSink;

impl BlockTreeSink for StdoutSink {
    fn write(&self, text: &str) {
        print!("{text}");
    }
}

/// Lines to stderr (the CLI's: its stdout carries the results).
pub struct StderrSink;

impl BlockTreeSink for StderrSink {
    fn write(&self, text: &str) {
        eprint!("{text}");
    }
}

// =============================== the posture ==============================

/// The production posture the block's measurements run under: the knobs that
/// change behaviour or performance, at the values the suite env sets (ULTRA's
/// and BIG's `STARK_ENV` + `STARK_TREE_ENV`). The CLI sets each one the
/// environment leaves unset — `LAMBDA_VM_VRAM_BUDGET_MB` only on a card of at
/// least [`POSTURE_VRAM_MIN_GIB`] — and the harness never does: its arms rely
/// on unset meaning the library default. Measurement-only knobs
/// (`LFM_PROVE_SPLIT`, `LAMBDA_VM_BASE_SPLIT`) are not posture. The allocator's
/// never-purge posture is compiled into the binary.
pub const POSTURE: &[(&str, &str)] = &[
    ("TABLE_PARALLELISM", "8"),
    (POSTURE_VRAM_KNOB, "24000"),
    ("LAMBDA_VM_GATE_PACKING", "1"),
    ("LAMBDA_VM_MAX_ROWS_LOG2", "21"),
    ("LFM_PRECOMPUTED_TREE_CACHE_CAP", "64"),
    ("LFM_EXEC_PARALLEL", "1"),
    ("LFM_TREE_SIBLINGS_L0", "8"),
    ("LFM_TREE_SIBLINGS", "4"),
    (crate::alloc_purge::ALLOC_PURGE_ENV, "auto"),
];

/// The posture's VRAM budget: 24000 MB is the 32 GiB card's.
pub const POSTURE_VRAM_KNOB: &str = "LAMBDA_VM_VRAM_BUDGET_MB";

/// The smallest card [`POSTURE`]'s VRAM budget is set on, in GiB (an RTX 5090
/// reports 31.84).
pub const POSTURE_VRAM_MIN_GIB: f64 = 31.0;

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

/// Threads for the ELF constants beside the base by default: FAST 393 measured
/// the harvest −1.87 s and the whole −1.80 s at four, the base +0.02 s.
const ELF_BESIDE_THREADS: usize = 4;

/// How the tree's programs come ahead (`NOEPOCH_TREE_AHEAD`). Unset or `pipe`,
/// the default (FAST 451: recursion −2.56 s): the leaf programs are emitted
/// beside the base on the host and the artifacts built on the card during level
/// 0, each node program emitted as its children's artifacts exist
/// ([`super::block_tree_pipeline`]). `1` derives every program and its
/// artifacts beside the base on the host; `0` derives each inline, before it
/// proves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AheadMode {
    Host,
    Pipe,
}

/// Threads of the pipeline builder's own pool for the node emission, by
/// default: one per level-1 node of the block's 8 leaves at fan-in 2, where it
/// was measured (FAST 455: level 0 −0.40 s, recursion −0.29 s; on the global
/// pool a leaf proof's join could steal an emission and hold the card idle). At
/// fan-in 4 level 1 has two nodes to emit.
const EMIT_POOL_THREADS: usize = 4;

/// How the pipeline emits the leaf programs beside the base
/// (`NOEPOCH_TREE_EMIT_LATE`).
///
/// Late: leaf 0 at the shape, the rest once the heap's live bytes have fallen
/// `margin` × their estimated bytes below the shape's (phase B frees each trace
/// as its table proves), or have stopped falling, or the base has returned
/// ([`late_trigger`]). Freed trace buffers sit in jemalloc's shared oversize
/// arena and 99.5 % of a program's bytes are allocations that size, so the
/// programs take the freed pages instead of raising the base's high-water. At
/// the median, spill off, that is base-phase VmRSS −10.37 GiB, recursion −0.31
/// s, whole −2.15 s, 98 % of the programs' bytes on reused pages (BIG 585, 2 +
/// 2); with spill on it is inert (−1.28 GiB, +0.55 s), the room phase A leaves
/// being elsewhere. The live bytes are the allocator's
/// ([`crate::alloc_purge::stats`]); with no reading the programs are emitted
/// at the shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LateMode {
    /// `0` or `off`: every leaf program at the shape.
    Off,
    /// Unset, empty or `auto` (the default): late at [`LATE_MARGIN`] when the
    /// programs' estimate reaches [`LATE_MIN_ESTIMATE`], at the shape otherwise.
    Auto,
    /// `<margin>`: late at that margin, whatever the estimate.
    Margin(f64),
}

/// Every knob the whole block reads, once.
#[derive(Clone, Debug)]
pub struct BlockTreeConfig {
    /// `NOEPOCH_ELF_BESIDE=<threads>`: threads for the ELF constants beside the
    /// base, unset meaning [`ELF_BESIDE_THREADS`]; `0` (`None`) computes them
    /// inline in the harvest.
    pub elf_beside: Option<usize>,
    /// `NOEPOCH_TREE_AHEAD`: `0` (`None`), `1` or `pipe` (the default). Needs
    /// the ELF constants beside the base.
    pub tree_ahead: Option<AheadMode>,
    /// `NOEPOCH_EMIT_POOL=<threads>`: the pipeline's builder emits the node
    /// programs on a host-only pool of its own with that many threads, unset
    /// meaning [`EMIT_POOL_THREADS`]; `0` emits them on the global pool the
    /// provers use.
    pub emit_threads: usize,
    /// `NOEPOCH_TREE_EMIT_WINDOW=<W>`: in the pipeline mode, only the first W
    /// leaf programs are emitted beside the base; the builder emits the rest in
    /// leaf order, at most `2 × W` ahead of its artifact builds, on its own
    /// pool. Unset (the default) emits every leaf program beside the base,
    /// which holds them all through the base's prove (≈ 338 MiB a leaf at the
    /// median block, BIG 480).
    pub emit_window: Option<usize>,
    /// `NOEPOCH_TREE_NODE_EMIT`: in the pipeline mode, `early` or unset (the
    /// default) emits each node's program as soon as its children's artifacts
    /// exist, during the level below; `level` emits a level's programs
    /// together once the whole level below is built. Per level, the median
    /// block's card sat idle 8.9 s between level 0's last hold and level 1's
    /// first (BIG 481); early takes the median's recursion −7.32 s (BIG 483,
    /// 2 + 2) and 1× −0.15 s (FAST 668, 4 + 4), the programs and the top
    /// unchanged.
    pub node_emit_early: bool,
    /// `NOEPOCH_TREE_EMIT_LATE`: [`LateMode`].
    pub emit_late: LateMode,
    /// `NOEPOCH_HARVEST_VERIFY=inline` verifies the base before level 0; by
    /// default the verify runs on a helper thread beside level 0 and is joined
    /// before anything is reported: a refused block still fails the run, it
    /// only stops delaying the leaves.
    pub harvest_verify_inline: bool,
    /// The tree's child proofs verified beside the timed path, by default (FAST
    /// 398: −0.62 s on level 0 + interior, the join waiting 0.00 s);
    /// `NOEPOCH_CHILD_VERIFY=inline` verifies each before its parent.
    pub child_verify_beside: bool,
    /// `NOEPOCH_LEAVES`: the leaf count, forced; unset is the plan's rule
    /// (`⌈Σ cost / 279 000⌉`). A forced count is another tree than the plan's,
    /// which the block verifier refuses: a harness arm only.
    pub forced_leaves: Option<usize>,
    /// `LFM_TREE_SIBLINGS_L0` (or `LFM_TREE_K_L0`): leaf proofs at once.
    pub siblings_l0: usize,
    /// `LFM_TREE_SIBLINGS` (or `LFM_TREE_K`): node proofs at once.
    pub siblings: usize,
}

impl BlockTreeConfig {
    /// Every knob from the environment, under the names and defaults the
    /// harness always read. A value a knob cannot take is an error.
    pub fn from_env() -> Result<Self, String> {
        let var = |name: &str| std::env::var(name).ok();
        Ok(Self {
            elf_beside: parse_elf_beside(var("NOEPOCH_ELF_BESIDE").as_deref())?,
            tree_ahead: parse_tree_ahead(var("NOEPOCH_TREE_AHEAD").as_deref())?,
            emit_threads: parse_emit_pool(var("NOEPOCH_EMIT_POOL").as_deref())?,
            emit_window: parse_emit_window(var("NOEPOCH_TREE_EMIT_WINDOW").as_deref())?,
            node_emit_early: parse_node_emit(var("NOEPOCH_TREE_NODE_EMIT").as_deref())?,
            emit_late: parse_emit_late(var("NOEPOCH_TREE_EMIT_LATE").as_deref())?,
            harvest_verify_inline: var("NOEPOCH_HARVEST_VERIFY").is_some_and(|v| v == "inline"),
            child_verify_beside: var("NOEPOCH_CHILD_VERIFY").is_none_or(|v| v != "inline"),
            forced_leaves: parse_leaves(var("NOEPOCH_LEAVES").as_deref())?,
            siblings_l0: super::tree_run::tree_siblings_l0()?,
            siblings: super::tree_run::tree_siblings()?,
        })
    }
}

fn parse_elf_beside(v: Option<&str>) -> Result<Option<usize>, String> {
    let threads = match v {
        Some(v) if !v.is_empty() => v
            .parse::<usize>()
            .map_err(|_| format!("NOEPOCH_ELF_BESIDE must be a thread count, got `{v}`"))?,
        _ => ELF_BESIDE_THREADS,
    };
    Ok(Some(threads).filter(|&t| t > 0))
}

fn parse_tree_ahead(v: Option<&str>) -> Result<Option<AheadMode>, String> {
    match v {
        Some("0") => Ok(None),
        Some("1") => Ok(Some(AheadMode::Host)),
        None | Some("" | "pipe") => Ok(Some(AheadMode::Pipe)),
        Some(v) => Err(format!(
            "NOEPOCH_TREE_AHEAD must be 0, 1 or pipe, got `{v}`"
        )),
    }
}

fn parse_emit_pool(v: Option<&str>) -> Result<usize, String> {
    match v {
        Some(v) if !v.is_empty() => v
            .parse::<usize>()
            .map_err(|_| format!("NOEPOCH_EMIT_POOL must be a thread count, got `{v}`")),
        _ => Ok(EMIT_POOL_THREADS),
    }
}

fn parse_node_emit(v: Option<&str>) -> Result<bool, String> {
    match v {
        None | Some("" | "early") => Ok(true),
        Some("level") => Ok(false),
        Some(v) => Err(format!(
            "NOEPOCH_TREE_NODE_EMIT must be early or level, got `{v}`"
        )),
    }
}

fn parse_emit_window(v: Option<&str>) -> Result<Option<usize>, String> {
    match v.filter(|v| !v.is_empty()) {
        None => Ok(None),
        Some(v) => v
            .parse()
            .ok()
            .filter(|w: &usize| *w >= 1)
            .map(Some)
            .ok_or_else(|| {
                format!("NOEPOCH_TREE_EMIT_WINDOW must be a positive integer, got `{v}`")
            }),
    }
}

fn parse_emit_late(v: Option<&str>) -> Result<LateMode, String> {
    match v {
        None | Some("" | "auto") => Ok(LateMode::Auto),
        Some("0" | "off") => Ok(LateMode::Off),
        Some(v) => v
            .parse()
            .ok()
            .filter(|m: &f64| m.is_finite() && *m > 0.0)
            .map(LateMode::Margin)
            .ok_or_else(|| {
                format!("NOEPOCH_TREE_EMIT_LATE must be auto, off or a positive margin, got `{v}`")
            }),
    }
}

fn parse_leaves(v: Option<&str>) -> Result<Option<usize>, String> {
    match v.filter(|v| !v.is_empty()) {
        None => Ok(None),
        Some(v) => v
            .parse()
            .ok()
            .filter(|k: &usize| *k >= 1)
            .map(Some)
            .ok_or_else(|| format!("NOEPOCH_LEAVES must be a positive integer, got `{v}`")),
    }
}

// ============================== the late emission =========================

/// The default late emission's margin over the programs' estimated bytes: the
/// pages they need plus phase B's own transients (BIG 585).
const LATE_MARGIN: f64 = 1.25;

/// The default late emission's floor: below this estimate the programs are
/// emitted at the shape. A small tree's programs barely move the peak, and its
/// short phase B may not free `margin` × their bytes before the base returns,
/// which would leave the harvest waiting on them (the record block's 7 more
/// leaves are ≈ 3.8 GiB [I]; the median's 55 are 30.6 GiB).
const LATE_MIN_ESTIMATE: usize = 8 << 30;

/// The margin a late emission waits at, or `None` to emit at the shape.
fn late_margin(mode: LateMode, estimate: usize) -> Option<f64> {
    match mode {
        LateMode::Off => None,
        LateMode::Margin(m) => Some(m),
        LateMode::Auto => (estimate >= LATE_MIN_ESTIMATE).then_some(LATE_MARGIN),
    }
}

/// The other `beside - 1` leaves' programs' bytes, from leaf 0's (`first`)
/// bytes a permutation of in-guest verification.
fn late_estimate(plan: &BlockTreePlan, first: &LfmProgram, beside: usize) -> usize {
    let costs = plan.costs();
    let perms =
        |k: usize| -> usize { plan.partition().leaves()[k].iter().map(|&i| costs[i]).sum() };
    let per_perm = ProgramBytes::of(first).total() as f64 / perms(0).max(1) as f64;
    (per_perm * (1..beside).map(perms).sum::<usize>() as f64) as usize
}

/// [`late_trigger`]'s settle rule: the heap's live bytes have made no new low
/// by [`LATE_LOW_STEP`] for this long (spill on: phase B holds only its
/// read-back window, so the live bytes stop falling long before the programs'
/// bytes are freed).
const LATE_SETTLE_SECS: f64 = 30.0;

/// The step a new low of the heap's live bytes must beat the last one by.
const LATE_LOW_STEP: usize = 1 << 30;

/// When the late emission (`NOEPOCH_TREE_EMIT_LATE`) starts: once the heap's
/// live bytes are `need` below their value at the shape (`fell`), or have made
/// no new low for [`LATE_SETTLE_SECS`] (`settled`), or the base has returned
/// (`base returned`); `None` keeps waiting.
fn late_trigger(
    live_at_shape: usize,
    live: usize,
    need: usize,
    since_low_secs: f64,
    base_done: bool,
) -> Option<&'static str> {
    if live_at_shape.saturating_sub(live) >= need {
        Some("fell")
    } else if base_done {
        Some("base returned")
    } else if since_low_secs >= LATE_SETTLE_SECS {
        Some("settled")
    } else {
        None
    }
}

/// The late emission's wait and what the heap did around it.
struct LateWait {
    margin: f64,
    trigger: &'static str,
    waited: f64,
    /// Seconds since the base started, when the wait ended.
    at: f64,
    estimate: usize,
    need: usize,
    live_at_shape: usize,
    live_at_start: usize,
    resident_at_start: usize,
}

/// The heap's live bytes, from the allocator's statistics (0 without them;
/// the late emission is armed only with a reading at the shape).
fn heap_allocated() -> usize {
    crate::alloc_purge::stats().map_or(0, |s| s.allocated)
}

/// The pages the allocator holds: live, freed-and-dirty, and its metadata.
fn heap_resident() -> usize {
    crate::alloc_purge::stats().map_or(0, |s| s.resident)
}

/// Waits for [`late_trigger`], polling the heap every 200 ms, for a fall of
/// `margin` × `estimate` ([`late_estimate`]).
fn wait_for_late_emission(
    estimate: usize,
    margin: f64,
    live_at_shape: usize,
    base_done: &std::sync::atomic::AtomicBool,
    t_base0: Instant,
) -> LateWait {
    let need = (margin * estimate as f64) as usize;
    let t = Instant::now();
    let (mut low, mut low_at) = (live_at_shape, Instant::now());
    let trigger = loop {
        let live = heap_allocated();
        if live + LATE_LOW_STEP <= low {
            (low, low_at) = (live, Instant::now());
        }
        let done = base_done.load(std::sync::atomic::Ordering::SeqCst);
        if let Some(why) = late_trigger(
            live_at_shape,
            live,
            need,
            low_at.elapsed().as_secs_f64(),
            done,
        ) {
            break why;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    };
    LateWait {
        margin,
        trigger,
        waited: t.elapsed().as_secs_f64(),
        at: t_base0.elapsed().as_secs_f64(),
        estimate,
        need,
        live_at_shape,
        live_at_start: heap_allocated(),
        resident_at_start: heap_resident(),
    }
}

// ======================= the leaf programs' host bytes ====================

/// jemalloc's `opt.oversize_threshold` (5.3's default; the instruments read it
/// back): an allocation of at least this many bytes comes from one arena every
/// thread shares, a smaller one from its thread's own arena. The pages a freed
/// buffer leaves behind serve only its arena's later requests.
pub(crate) const JEMALLOC_OVERSIZE: usize = 8 << 20;

/// A program's host bytes as held (capacities, not lengths), by part, and the
/// share in allocations at or over [`JEMALLOC_OVERSIZE`]: the part that can
/// take the pages a freed trace left in the shared arena.
#[derive(Default, Clone, Copy)]
pub(crate) struct ProgramBytes {
    /// The instruction vector itself.
    pub(crate) instrs: usize,
    /// `Instr::BitDec`'s bit lists.
    pub(crate) bitdec_heap: usize,
    /// `KeccakF`'s and `Blake3`'s boxed operands.
    pub(crate) boxed: usize,
    /// The column groups as held: padded rows, at capacity.
    pub(crate) groups: usize,
    /// The column groups' padded rows (the committed matrices).
    pub(crate) groups_padded: usize,
    /// The column groups' real rows alone.
    pub(crate) groups_real: usize,
    /// The arena schema.
    pub(crate) other: usize,
    /// Of the total, the bytes in allocations at or over [`JEMALLOC_OVERSIZE`].
    pub(crate) large: usize,
    pub(crate) allocs: usize,
}

impl ProgramBytes {
    pub(crate) fn of(p: &LfmProgram) -> Self {
        use super::instr::{Addr, Blake3Operands, Instr, KeccakOperands};
        let mut b = Self::default();
        b.instrs = b.note(p.instrs.capacity() * size_of::<Instr>());
        for instr in &p.instrs {
            match instr {
                Instr::BitDec { bits, .. } => {
                    b.bitdec_heap += b.note(bits.capacity() * size_of::<(Addr, u64)>());
                }
                Instr::KeccakF(_) => b.boxed += b.note(size_of::<KeccakOperands>()),
                Instr::Blake3(_) => b.boxed += b.note(size_of::<Blake3Operands>()),
                _ => {}
            }
        }
        for g in program_groups(p) {
            b.groups += b.note(g.data.capacity() * size_of::<FE>());
            b.groups_padded += g.padded_rows * g.width * size_of::<FE>();
            b.groups_real += g.real_rows * g.width * size_of::<FE>();
        }
        b.other = b.note(p.arena_schema.lens.capacity() * size_of::<u32>());
        b
    }

    /// Counts one allocation of `bytes` and returns them.
    fn note(&mut self, bytes: usize) -> usize {
        if bytes > 0 {
            self.allocs += 1;
            if bytes >= JEMALLOC_OVERSIZE {
                self.large += bytes;
            }
        }
        bytes
    }

    pub(crate) fn total(&self) -> usize {
        self.instrs + self.bitdec_heap + self.boxed + self.groups + self.other
    }

    pub(crate) fn add(&mut self, o: &Self) {
        self.instrs += o.instrs;
        self.bitdec_heap += o.bitdec_heap;
        self.boxed += o.boxed;
        self.groups += o.groups;
        self.groups_padded += o.groups_padded;
        self.groups_real += o.groups_real;
        self.other += o.other;
        self.large += o.large;
        self.allocs += o.allocs;
    }
}

/// A program's column groups with their chip names, in the frozen chip order.
pub(crate) fn program_groups(p: &LfmProgram) -> [&super::compiler::ColumnGroup; 11] {
    let g = &p.groups;
    [
        &g.const_, &g.balu, &g.xalu, &g.select, &g.bitdec, &g.hash, &g.keccak, &g.blake3, &g.lanes,
        &g.hint, &g.public,
    ]
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

// =============================== the harvest ==============================

/// The host transcript every block prover and verifier starts from:
/// `StatementKind::Monolithic` over the block's statement fields.
pub(crate) fn block_seed(
    elf_digest: &[u8; 32],
    public_output: &[u8],
    table_counts: &crate::TableCounts,
    num_private_input_pages: usize,
    runtime_page_ranges: &[crate::RuntimePageRange],
    fri_final_poly_log_degree: u8,
) -> crate::hash_pin::BlockTranscript {
    let mut t = crate::hash_pin::block_transcript(&[]);
    crate::statement::absorb_statement_with_digest(
        &mut t,
        crate::statement::StatementKind::Monolithic,
        elf_digest,
        public_output,
        table_counts,
        num_private_input_pages,
        runtime_page_ranges,
        fri_final_poly_log_degree,
    );
    t
}

/// One block proof, production-accepted, read for leaf emission.
///
/// `plan` is the verifier's: every shape and constant a leaf is emitted from,
/// derived from the ELF, the options and the shape the proof claims — never
/// from its data. The other fields are the proof's DATA, which the leaves'
/// arenas carry: every instance's main root, the shared state, and per instance
/// the fork's data ([`HostTable`]) and the legs' openings ([`TableLegs`]), each
/// checked against the AIR's shape as it is read.
pub(crate) struct BlockWitness {
    pub(crate) plan: BlockTreePlan,
    pub(crate) public_output: Vec<u8>,
    pub(crate) main_roots: Vec<Commitment>,
    pub(crate) tables: Vec<HostTable>,
    pub(crate) legs: Vec<TableLegs>,
    /// The host transcript's state after `z, α`.
    pub(crate) state: LfmWord,
    /// The COMMIT-bus target production computed.
    pub(crate) expected_bus_balance: FEE,
}

impl BlockWitness {
    pub(crate) fn num_instances(&self) -> usize {
        self.plan.num_instances()
    }

    /// AIR names, in AIR order.
    pub(crate) fn names(&self) -> Vec<&str> {
        self.plan
            .instances()
            .iter()
            .map(|i| i.name.as_str())
            .collect()
    }

    /// The rule over `num_leaves` leaves: a TEST partition, another tree than
    /// the plan's unless the count is the plan's own.
    #[cfg(test)]
    pub(crate) fn try_partition_over(&self, num_leaves: usize) -> Result<BlockPartition, String> {
        super::block_leaf::partition_by_rule(&self.names(), &self.plan.costs(), num_leaves)
            .map_err(|e| format!("the rule fills every leaf: {e}"))
    }

    /// [`Self::try_partition_over`], panicking: the suites' form.
    #[cfg(test)]
    pub(crate) fn partition_over(&self, num_leaves: usize) -> BlockPartition {
        self.try_partition_over(num_leaves)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// What leaf `k` of `partition` must publish, from the host's own values:
    /// the id, the state, the output halves, and its share of the bus — less
    /// the target when it `carries` it.
    pub(crate) fn expected_leaf_publics(
        &self,
        partition: &BlockPartition,
        k: usize,
        carries: bool,
    ) -> Vec<LfmWord> {
        let mut words =
            super::programs::program_id_words(&self.plan.attested().program_id()).to_vec();
        words.push(self.state);
        words.extend(le_halves(&self.public_output).into_iter().map(base_word));
        let mut share = partition
            .leaf(k)
            .iter()
            .filter_map(|&i| self.tables[i].contribution)
            .fold(FEE::zero(), |acc, l| acc + l);
        if carries {
            share = share - self.expected_bus_balance;
        }
        words.push(ext_word(&share));
        words
    }

    /// The top's published words the host expects: the id, the state, the
    /// public output.
    pub(crate) fn expected_top_publics(&self) -> Vec<LfmWord> {
        let mut want =
            super::programs::program_id_words(&self.plan.attested().program_id()).to_vec();
        want.push(self.state);
        want.extend(le_halves(&self.public_output).into_iter().map(base_word));
        want
    }
}

/// Harvest a block proof that production's block verifier accepts, over ELF
/// constants computed ahead (beside the base), or computed here when `None`.
///
/// The plan comes first ([`BlockTreePlan::derive`], from the shape the proof
/// claims), and the harvest verifies (when `verify`) and reads the proof over
/// the plan's own AIR set — `verify_proof_parts`' set. Returns the harvest and
/// the seconds of its two halves — the plan and the host verify, which only
/// refuses, and the replay a driver needs — and prints the split of its first
/// half: the plan, the COMMIT-bus target, the host verify.
pub(crate) fn harvest_block_over(
    opts: &crate::ProofOptions,
    elf_bytes: &[u8],
    proof: &crate::VmProof,
    verify: bool,
    consts: Option<&ElfConstants>,
    sink: &dyn BlockTreeSink,
) -> Result<(BlockWitness, f64, f64), String> {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use rayon::prelude::*;
    use stark::verifier::IsStarkVerifier;

    let t_verify = Instant::now();
    let shape = BlockShape::of_proof(proof);
    let plan = match consts {
        Some(c) => BlockTreePlan::derive_with(elf_bytes, opts, &shape, c)?,
        None => BlockTreePlan::derive(elf_bytes, opts, &shape)?,
    };
    let plan_secs = t_verify.elapsed().as_secs_f64();
    let view = MultiProofView::Owned(&proof.proof);
    let refs = plan.airs().air_refs();
    let seed = || {
        block_seed(
            plan.elf_digest(),
            &proof.public_output,
            &proof.table_counts,
            proof.num_private_input_pages,
            &proof.runtime_page_ranges,
            opts.fri_final_poly_log_degree,
        )
    };
    let expected = crate::compute_expected_commit_bus_balance_view(
        &refs,
        view,
        &proof.public_output,
        0,
        &mut seed(),
    )
    .ok_or("the COMMIT bus target must compute")?;
    let target_secs = t_verify.elapsed().as_secs_f64() - plan_secs;
    if verify
        && !crate::hash_pin::BlockVerifier::<Gl, Ext3, ()>::multi_verify_views(
            &refs,
            view,
            &mut seed(),
            &expected,
        )
    {
        return Err("production's verifier rejects the block".to_string());
    }
    let verify_secs = t_verify.elapsed().as_secs_f64();
    sink.line(&format!(
        "   harvest split ({}): plan {plan_secs:.2}s ({}) · bus target {target_secs:.2}s · \
         host verify {:.2}s",
        if verify { "verifying" } else { "reading" },
        if consts.is_some() {
            "ELF constants ahead"
        } else {
            "ELF constants inline"
        },
        verify_secs - plan_secs - target_secs
    ));

    // ---- Phase A, as `multi_verify_views` absorbs it: the plan's preprocessed
    // roots (the AIR's, never the proof's), then the proof's main roots.
    let t_replay = Instant::now();
    let n = refs.len();
    let mut transcript = seed();
    let mut main_roots = Vec::with_capacity(n);
    for idx in 0..n {
        let v = view.get(idx);
        if let Some(p) = &plan.instance(idx).precomputed_root {
            transcript.append_bytes(p);
        }
        transcript.append_bytes(v.lde_trace_main_merkle_root());
        main_roots.push(*v.lde_trace_main_merkle_root());
    }
    let lookup: Vec<FEE> = (0..stark::lookup::LOGUP_NUM_CHALLENGES)
        .map(|_| transcript.sample_field_element())
        .collect();
    let state = transcript.state_word();

    // ---- one fork per instance, and the legs' reading of the same sub-proof.
    let per_instance: Vec<(HostTable, TableLegs)> = (0..n)
        .into_par_iter()
        .map(|idx| -> Result<(HostTable, TableLegs), String> {
            let air = refs[idx];
            let v = view.get(idx);
            let mut fork = transcript.clone();
            if n > 1 {
                fork.append_bytes(&(idx as u64).to_le_bytes());
            }
            if let Some(root) = v.lde_trace_aux_merkle_root() {
                fork.append_bytes(root);
            }
            if let Some(c) = v.bus_table_contribution() {
                fork.append_field_element(&c);
            }
            let table = super::harvest::host_table_forked(air, v, idx, n, &mut fork, &lookup)?;
            let legs = super::harvest::build_table_legs(air, v, &lookup)?;
            Ok((table, legs))
        })
        .collect::<Result<_, String>>()?;
    let (tables, legs): (Vec<_>, Vec<_>) = per_instance.into_iter().unzip();
    // The proof-reading path and the plan derive one shape per instance.
    for (i, (t, l)) in tables.iter().zip(&legs).enumerate() {
        let planned = plan.instance(i);
        if t.precomputed_root != planned.precomputed_root {
            return Err(format!(
                "instance {i}: the fork's and the plan's preprocessed roots are one value"
            ));
        }
        if format!("{:?}", t.shape) != format!("{:?}", planned.challenge) {
            return Err(format!(
                "instance {i}: the plan's challenge shape is the one the proof is read against"
            ));
        }
        if format!("{:?}", l.verify) != format!("{:?}", planned.verify) {
            return Err(format!(
                "instance {i}: the plan's legs shape is the one the proof is read against"
            ));
        }
    }
    let replay_secs = t_replay.elapsed().as_secs_f64();

    Ok((
        BlockWitness {
            plan,
            public_output: proof.public_output.clone(),
            main_roots,
            tables,
            legs,
            state,
            expected_bus_balance: expected,
        },
        verify_secs,
        replay_secs,
    ))
}

// ============================ the programs and arenas =====================

/// Leaf `k` of the plan — the program the verifier derives.
pub(crate) fn block_leaf_program(rb: &BlockWitness, k: usize) -> Result<LfmProgram, String> {
    rb.plan
        .leaf_program(k)
        .map_err(|e| format!("leaf {k} must emit: {e}"))
}

/// Leaf `k`'s arenas, in `emit_block_leaf`'s declaration order.
pub(crate) fn block_leaf_arenas(
    rb: &BlockWitness,
    partition: &BlockPartition,
    k: usize,
) -> Result<Vec<Vec<LfmWord>>, String> {
    let mut arenas: Vec<Vec<LfmWord>> = vec![
        le_halves(&rb.public_output)
            .into_iter()
            .map(base_word)
            .collect(),
        super::proof_arena::commitments_to_arena(&rb.main_roots),
    ];
    for &i in partition.leaf(k) {
        let (h, leg) = (&rb.tables[i], &rb.legs[i]);
        if let Some(root) = &h.aux_root {
            arenas.push(super::proof_arena::commitments_to_arena(&[*root]));
        }
        if let Some(l) = &h.contribution {
            arenas.push(vec![ext_word(l)]);
        }
        arenas.push(super::proof_arena::commitments_to_arena(&[
            h.composition_root
        ]));
        arenas.push(h.ood_current.iter().map(ext_word).collect());
        arenas.push(h.ood_next.iter().map(ext_word).collect());
        arenas.push(h.parts.iter().map(ext_word).collect());
        arenas.push(super::proof_arena::commitments_to_arena(&h.fri_roots));
        arenas.push(h.fri_coeffs.iter().map(ext_word).collect());
        if let Some(nonce) = h.nonce {
            arenas.push(vec![base_word(FE::from(nonce))]);
        }
        arenas.push(leg.try_opening_arena()?);
        arenas.push(leg.try_fri_arena()?);
        arenas.extend(leg.try_caps_arena()?);
    }
    Ok(arenas)
}

/// The plan's node over `children`, from the shapes their ARTIFACTS give — no
/// child proof — checked against the heights their proofs carry.
pub(crate) fn block_node_program(
    plan: &BlockTreePlan,
    children: &[&HarvestedChild],
    top: bool,
) -> Result<LfmProgram, String> {
    let words = plan.child_layout().total();
    let derived = children
        .iter()
        .map(|c| -> Result<DerivedChild, String> {
            let d = DerivedChild::from_artifacts(&c.artifacts, &c.opts, words)
                .map_err(|e| format!("a child's shapes derive from its artifacts: {e}"))?;
            let proved: Vec<u32> = c.tables.iter().map(|t| t.shape.log2_trace_length).collect();
            if d.log2_trace_lengths() != proved {
                return Err(format!(
                    "the artifacts' heights are the child proof's trace lengths: {:?} != {proved:?}",
                    d.log2_trace_lengths()
                ));
            }
            Ok(d)
        })
        .collect::<Result<Vec<_>, String>>()?;
    plan.node_program(&derived, top)
        .map_err(|e| format!("a block node must emit: {e}"))
}

/// A block node's arenas: its children's, in child order.
pub(crate) fn block_node_arenas(children: &[&HarvestedChild]) -> Result<Vec<Vec<LfmWord>>, String> {
    let mut arenas = Vec::new();
    for c in children {
        arenas.extend(super::harvest::try_child_arena_words(c)?);
    }
    Ok(arenas)
}

// ================================ the proves ==============================

/// Prove `program` over `arenas` — over the artifacts built ahead, when given
/// — and read the proof back as a child. With `verify_inline` the proof is
/// verified first (production must accept it); without, it is read back
/// unverified and the caller verifies it elsewhere ([`BesideVerifies`]) before
/// anything is reported. Prints the prove's split and one timing line.
pub(crate) fn prove_program_with(
    label: &str,
    program: &LfmProgram,
    built: Option<LfmArtifacts>,
    arenas: &[Vec<LfmWord>],
    opts: &crate::ProofOptions,
    verify_inline: bool,
    sink: &dyn BlockTreeSink,
) -> Result<(HarvestedChild, LfmProof), String> {
    let t = Instant::now();
    let artifacts = built.unwrap_or_else(|| {
        super::program_census::build_artifacts_counted(program, opts, crate::hash_pin::BLOCK_HASHER)
    });
    let t_artifacts = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let proved = super::proof::lfm_prove(program, &artifacts, arenas, opts)
        .map_err(|e| format!("{label} must prove: {e:?}"))?;
    let t_prove = t.elapsed().as_secs_f64();
    if let Some(split) = super::tree_run::prove_split_text(label) {
        sink.write(&split);
    }
    let t = Instant::now();
    let (child, verified) = if verify_inline {
        let (child, t_verify) =
            super::harvest::harvest_child_verified(artifacts, opts.clone(), &proved)
                .map_err(|e| format!("{label}: {e}"))?;
        (child, format!("verify {t_verify:.2}"))
    } else {
        let child = super::harvest::harvest_child(artifacts, opts.clone(), &proved)
            .map_err(|e| format!("{label}: {e}"))?;
        (child, "verified beside".to_string())
    };
    sink.line(&format!(
        "   {label} TIMING: {} instructions · build_artifacts {t_artifacts:.2}s · prove \
         {t_prove:.2}s · harvest {:.2}s ({verified})",
        program.instrs.len(),
        t.elapsed().as_secs_f64()
    ));
    Ok((child, proved))
}

/// A child verify on a helper thread: its label and whether production
/// accepted the proof.
type BesideVerify = (String, std::thread::JoinHandle<bool>);

/// Child proofs verified on helper threads beside the timed path (the default;
/// `NOEPOCH_CHILD_VERIFY=inline` is the old order): production's verify of
/// each, every one joined, a refusal failing the run, before anything is
/// reported.
#[derive(Default)]
pub(crate) struct BesideVerifies(std::sync::Mutex<Vec<BesideVerify>>);

impl BesideVerifies {
    pub(crate) fn spawn(
        &self,
        label: &str,
        artifacts: LfmArtifacts,
        proof: LfmProof,
        opts: crate::ProofOptions,
    ) {
        let handle = std::thread::spawn(move || {
            super::proof::verify_against_artifacts(
                &artifacts,
                &proof.proof,
                &proof.public_words,
                &opts,
            )
        });
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((label.to_string(), handle));
    }

    /// Joins every verify; a refusal (or a verify that panicked) is an error.
    /// Returns how many and the seconds the join waited.
    pub(crate) fn join_all(&self) -> Result<(usize, f64), String> {
        let t = Instant::now();
        let jobs = std::mem::take(&mut *self.0.lock().unwrap_or_else(|e| e.into_inner()));
        let n = jobs.len();
        let mut refused = None;
        for (label, handle) in jobs {
            match handle.join() {
                Ok(true) => {}
                Ok(false) => {
                    refused.get_or_insert_with(|| {
                        format!("{label}: production refuses the child (verified beside)")
                    });
                }
                Err(_) => {
                    refused.get_or_insert_with(|| format!("{label}: the verify beside panicked"));
                }
            }
        }
        match refused {
            Some(e) => Err(e),
            None => Ok((n, t.elapsed().as_secs_f64())),
        }
    }
}

/// Levels above the leaves, as the plan lays them out
/// ([`BlockTreePlan::levels`], fan-in [`super::block_plan::block_fan_in`]), the
/// last level's single node the top. A one-leaf block still gets a top node:
/// the bus closes there. Each node is proved from the program and artifacts
/// derived ahead when `ahead` holds the node levels, and each node below the top
/// is verified on `beside` when given (the top is verified inline).
/// Returns the top child, its proof and each level's wall.
pub(super) fn compose_block_tree_with(
    plan: &BlockTreePlan,
    leaves: Vec<HarvestedChild>,
    opts: &crate::ProofOptions,
    siblings: usize,
    ahead: Option<&Pipe>,
    beside: Option<&BesideVerifies>,
    sink: &dyn BlockTreeSink,
) -> Result<(HarvestedChild, LfmProof, Vec<f64>), String> {
    let fan_in = super::block_plan::block_fan_in();
    let shape = plan.levels();
    let mut level = leaves;
    let mut top_proof = None;
    let mut walls = Vec::with_capacity(shape.len());
    for (lv, arities) in shape.iter().enumerate() {
        let top = lv + 1 == shape.len();
        let t = Instant::now();
        let mut starts = Vec::with_capacity(arities.arities.len());
        let mut at = 0usize;
        for a in &arities.arities {
            starts.push(at..at + a);
            at += a;
        }
        if at != level.len() {
            return Err(format!(
                "the level's arities cover its children: {at} != {}",
                level.len()
            ));
        }
        let failed = std::sync::atomic::AtomicBool::new(false);
        let next = in_index_order(starts.len(), siblings, |j| {
            if failed.load(std::sync::atomic::Ordering::SeqCst) {
                return Err("skipped: a sibling failed".to_string());
            }
            let out = (|| -> Result<(HarvestedChild, Option<LfmProof>), String> {
                let kids: Vec<&HarvestedChild> = level[starts[j].clone()].iter().collect();
                let label = format!(
                    "BLOCK L{}N{j} ({} child{}){}",
                    lv + 1,
                    kids.len(),
                    if kids.len() == 1 { "" } else { "ren" },
                    if top { " TOP" } else { "" }
                );
                let te = Instant::now();
                let (program, built) = match ahead {
                    Some(p) => {
                        let (program, artifacts) = p.take_node(lv, j, &label)?;
                        (program, Some(artifacts))
                    }
                    None => (block_node_program(plan, &kids, top)?, None),
                };
                sink.line(&format!(
                    "   {label}: {} in {:.2}s",
                    if built.is_some() {
                        "program ahead"
                    } else {
                        "emitted"
                    },
                    te.elapsed().as_secs_f64()
                ));
                sink.write(&super::tree_run::census_panel(&program, &label, fan_in).2);
                let inline = top || beside.is_none();
                let (child, proof) = prove_program_with(
                    &label,
                    &program,
                    built,
                    &block_node_arenas(&kids)?,
                    opts,
                    inline,
                    sink,
                )?;
                if let (false, Some(b)) = (inline, beside) {
                    b.spawn(&label, child.artifacts.clone(), proof, opts.clone());
                    return Ok((child, None));
                }
                Ok((child, top.then_some(proof)))
            })();
            if out.is_err() {
                failed.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            out
        });
        let next = next.into_iter().collect::<Result<Vec<_>, String>>()?;
        let (next, mut proofs): (Vec<HarvestedChild>, Vec<Option<LfmProof>>) =
            next.into_iter().unzip();
        if top {
            top_proof = proofs.pop().flatten();
        }
        let wall = t.elapsed().as_secs_f64();
        sink.line(&format!(
            "   BLOCK LEVEL {}: {} node(s) in {wall:.2}s{}",
            lv + 1,
            next.len(),
            if top { " (the top)" } else { "" }
        ));
        walls.push(wall);
        level = next;
    }
    if level.len() != 1 {
        return Err(format!("the tree closes to one node, not {}", level.len()));
    }
    let top = level.pop().ok_or("the tree closes to one node")?;
    let top_proof = top_proof.ok_or("the top level keeps its proof")?;
    Ok((top, top_proof, walls))
}

// ================================ the driver ==============================

/// The whole block's walls, in seconds.
#[derive(Clone, Debug, Default)]
pub struct BlockTreeTimes {
    pub base: f64,
    pub harvest: f64,
    pub level0: f64,
    pub interior: f64,
    /// Each interior level's wall, the top's last.
    pub levels: Vec<f64>,
    /// From before the base to the last proof, the in-run verifies inside.
    pub whole: f64,
    /// The host's peak VmRSS over the whole run (GiB) and when (UNIX seconds).
    pub host_peak: (f64, f64),
}

/// What a whole block run hands back.
pub struct BlockTreeRun {
    /// The shape a consumer receives with the top proof.
    pub shape: BlockShape,
    /// The block's public output.
    pub public_output: Vec<u8>,
    /// The top node's proof.
    pub top_proof: LfmProof,
    /// The top program's id.
    pub top_program_id: Commitment,
    pub times: BlockTreeTimes,
    /// The top node as read back: its artifacts verify the top proof (the
    /// harness's post-run checks).
    #[cfg(test)]
    pub(crate) top: HarvestedChild,
    /// The harvest the leaves were filled from, with the plan.
    #[cfg(test)]
    pub(crate) witness: BlockWitness,
    /// The ELF constants computed beside the base, when they were.
    #[cfg(test)]
    pub(crate) consts: Option<Arc<ElfConstants>>,
    /// Whether `NOEPOCH_LEAVES` forced the partition (another tree).
    #[cfg(test)]
    pub(crate) forced: Option<usize>,
}

/// A program's id as hex.
pub fn hex_id(id: &[u8]) -> String {
    id.iter().map(|b| format!("{b:02x}")).collect()
}

/// ★★★ THE WHOLE NO-EPOCH BLOCK, base to top node: `block::prove_block`, the
/// harvest, the leaves (level 0), the interior and the top — the block's
/// recursion, and the number the epoch/no-epoch decision is taken on.
///
/// Its `★★★ WHOLE RUN` line is measured as the epoch tree's: from before the
/// base to the last proof, the in-run verifies inside. The top node is the
/// last proof, and it closes the block's bus. A failure anywhere is an error;
/// a helper thread still running then is left to finish or to block on a
/// channel nobody sends on.
pub fn prove_block_tree(
    elf_bytes: &[u8],
    input: &[u8],
    cfg: &BlockTreeConfig,
    sink: Arc<dyn BlockTreeSink>,
) -> Result<BlockTreeRun, String> {
    if !cfg!(feature = "cuda") {
        return Err("the production block tree requires `--features cuda`".to_string());
    }
    // A forced partition is another tree than the plan's, which the block
    // verifier refuses: only the harness (the lib's test build) may run one.
    #[cfg(not(test))]
    if let Some(k) = cfg.forced_leaves {
        return Err(format!(
            "NOEPOCH_LEAVES={k} forces another tree than the plan's, which the block verifier \
             refuses: a harness arm, not run outside the lib's tests"
        ));
    }
    let elf_bytes: Arc<[u8]> = Arc::from(elf_bytes);
    let inner = super::proof::block_base_options();
    let wrap_opts = super::proof::aggregation_wrap_options();
    let ceiling = cgroup_limit_gib();
    let pct = |g: f64| match &ceiling {
        Ok(c) => format!(" ({:.1}% of {c:.2})", 100.0 * g / c),
        Err(_) => String::new(),
    };
    sink.line(&posture_line());
    sink.line(&format!(
        "★★★ NO-EPOCH BLOCK TREE (base + leaves + interior + top)\n   \
         {} input bytes · inner blowup {} / {} q · wrap blowup {} / {} q · cgroup {}",
        input.len(),
        inner.blowup_factor,
        inner.fri_number_of_queries,
        wrap_opts.blowup_factor,
        wrap_opts.fri_number_of_queries,
        match &ceiling {
            Ok(g) => format!("{g:.2} GiB"),
            Err(why) => format!("UNKNOWN — {why}"),
        },
    ));

    let whole = HostSampler::start();
    let t_all = Instant::now();

    // ---- the ELF constants beside the base (`NOEPOCH_ELF_BESIDE=<threads>`, four
    // by default): the plan's ELF-only input (DECODE's root, recomputed on the
    // host) computed on a pool of its own while the base proves, joined by the
    // harvest. `NOEPOCH_ELF_BESIDE=0`: the harvest computes it inline.
    let elf_beside = cfg.elf_beside;
    // `NOEPOCH_TREE_AHEAD` (the pipeline by default): the same pool then emits the
    // leaf programs (with `1`, every tree program and its artifacts, on the host)
    // from the shape the base hands out before its prove, and the levels prove
    // from them.
    let tree_ahead = elf_beside.and(cfg.tree_ahead);
    let emit_threads = cfg.emit_threads;
    let emit_window = cfg.emit_window;
    let node_emit_early = cfg.node_emit_early;
    let emit_late = cfg.emit_late;
    let base_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (shape_tx, shape_rx) = std::sync::mpsc::channel::<BlockShape>();
    // The thread hands its results back on `ready` and, in the pipeline mode,
    // stays on as the tree's builder once `go` says the base is done.
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<BesideResult>();
    let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
    let t_base0 = Instant::now();
    let consts_beside = elf_beside.map(|threads| {
        let (elf, opts, wrap) = (elf_bytes.clone(), inner.clone(), wrap_opts.clone());
        let base_done = base_done.clone();
        std::thread::spawn(move || -> Option<Result<super::block_tree_pipeline::BuilderTimes, String>> {
            let t = Instant::now();
            let pool = match rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("elf-beside-{i}"))
                .start_handler(|_| super::commit::mark_thread_host_only())
                .build()
            {
                Ok(pool) => pool,
                Err(e) => {
                    let _ = ready_tx.send((
                        Err(format!("the ELF constants pool builds: {e}")),
                        0.0,
                        None,
                        None,
                    ));
                    return None;
                }
            };
            let consts = pool.install(|| ElfConstants::compute(&elf, &opts));
            let secs = t.elapsed().as_secs_f64();
            let mut job = None;
            let mut late_line = None;
            let ahead = match (&consts, tree_ahead) {
                (Ok(c), Some(mode)) => shape_rx.recv().ok().map(|shape| {
                    let live_at_shape = if emit_late != LateMode::Off {
                        crate::alloc_purge::stats().map(|s| s.allocated)
                    } else {
                        None
                    };
                    let t = Instant::now();
                    let derived = pool.install(|| -> Result<_, String> {
                        let plan = BlockTreePlan::derive_with(&elf, &opts, &shape, c)?;
                        Ok(match mode {
                            AheadMode::Host => {
                                let (tree, phases) = plan.derive_tree(&wrap, &|program| {
                                    super::registry::build_artifacts_with_hasher(
                                        program,
                                        &wrap,
                                        crate::hash_pin::BLOCK_HASHER,
                                    )
                                })?;
                                (Arc::new(Pipe::filled(tree)), phases)
                            }
                            AheadMode::Pipe => {
                                use rayon::prelude::*;
                                let n = plan.partition().num_leaves();
                                let beside = emit_window.map_or(n, |w| w.min(n));
                                // `NOEPOCH_TREE_EMIT_LATE`: leaf 0 now, sizing the
                                // wait; the rest once phase B has freed their room.
                                let mut leaves = Vec::with_capacity(beside);
                                let late = match live_at_shape {
                                    Some(live) if beside > 1 => {
                                        leaves.push(plan.leaf_program(0)?);
                                        let estimate = late_estimate(&plan, &leaves[0], beside);
                                        match late_margin(emit_late, estimate) {
                                            Some(margin) => Some(wait_for_late_emission(
                                                estimate, margin, live, &base_done, t_base0,
                                            )),
                                            None => {
                                                late_line = Some(format!(
                                                    "   TREE LATE: auto, the other {} leaves' programs \
                                                     estimated {:.2} GiB, under the {:.0} GiB floor: \
                                                     emitted at the shape",
                                                    beside - 1,
                                                    gib(estimate),
                                                    gib(LATE_MIN_ESTIMATE)
                                                ));
                                                None
                                            }
                                        }
                                    }
                                    None if emit_late != LateMode::Off && beside > 1 => {
                                        late_line = Some(
                                            "   TREE LATE: no allocator reading: emitted at the shape"
                                                .to_string(),
                                        );
                                        None
                                    }
                                    _ => None,
                                };
                                let te = Instant::now();
                                let first = leaves.len();
                                leaves.extend(
                                    (first..beside)
                                        .into_par_iter()
                                        .map(|k| plan.leaf_program(k))
                                        .collect::<Result<Vec<_>, String>>()?,
                                );
                                if let Some(w) = late {
                                    let mut held = ProgramBytes::default();
                                    for p in &leaves {
                                        held.add(&ProgramBytes::of(p));
                                    }
                                    let resident = heap_resident();
                                    late_line = Some(format!(
                                        "   TREE LATE: leaf 0 at the shape, {} more after {:.2}s ({}) \
                                         at {:.2}s of the base, emitted in {:.2}s, done at {:.2}s · heap \
                                         live {:.2} GiB at the shape → {:.2} at the start (need a fall of \
                                         {:.2} = {} × {:.2} estimated) → {:.2} at the end · resident \
                                         {:.2} → {:.2} GiB over the emission (+{:.2}) · programs held \
                                         {:.2} GiB, ≥ oversize {:.1} %",
                                        beside - 1,
                                        w.waited,
                                        w.trigger,
                                        w.at,
                                        te.elapsed().as_secs_f64(),
                                        t_base0.elapsed().as_secs_f64(),
                                        gib(w.live_at_shape),
                                        gib(w.live_at_start),
                                        gib(w.need),
                                        w.margin,
                                        gib(w.estimate),
                                        gib(heap_allocated()),
                                        gib(w.resident_at_start),
                                        gib(resident),
                                        gib(resident.saturating_sub(w.resident_at_start)),
                                        gib(held.total()),
                                        100.0 * held.large as f64 / held.total().max(1) as f64,
                                    ));
                                }
                                let pipe = Arc::new(Pipe::new(n, &plan.levels()));
                                let emitted = super::block_plan::PhaseTimes {
                                    programs: leaves.len(),
                                    wall: te.elapsed().as_secs_f64(),
                                    emit: te.elapsed().as_secs_f64(),
                                    build: 0.0,
                                };
                                job = Some((pipe.clone(), plan, leaves));
                                (pipe, vec![emitted])
                            }
                        })
                    });
                    (
                        shape,
                        derived,
                        t.elapsed().as_secs_f64(),
                        t_base0.elapsed().as_secs_f64(),
                    )
                }),
                _ => None,
            };
            let _ = ready_tx.send((consts, secs, ahead, late_line));
            let (pipe, plan, leaves) = job?;
            go_rx.recv().ok()?;
            Some(pipe.run_builder(
                &plan,
                leaves,
                &wrap,
                emit_threads,
                emit_window.unwrap_or(0),
                node_emit_early,
            ))
        })
    });

    // ---- the base.
    let base_sampler = HostSampler::start();
    let t = Instant::now();
    let mut shape_at = None;
    let (proof, times) = crate::block::prove_block_observed(&elf_bytes, input, &inner, &mut |s| {
        shape_at = Some(t.elapsed().as_secs_f64());
        let _ = shape_tx.send(s.clone());
    })
    .map_err(|e| format!("the block must prove: {e}"))?;
    base_done.store(true, std::sync::atomic::Ordering::SeqCst);
    drop(shape_tx);
    sink.base_proved(&proof);
    let base = t.elapsed().as_secs_f64();
    let (base_peak, _) = base_sampler.stop();
    sink.line(&format!(
        "   base: {} sub-proofs in {base:.2}s (execute {:.2} · build {:.2} · setup {:.2} · \
         prove {:.2}) · host peak {base_peak:.3} GiB{}",
        proof.proof.proofs.len(),
        times.execute,
        times.build,
        times.setup,
        times.prove,
        pct(base_peak)
    ));

    // The base's freed pages back to the OS before the tree allocates: under
    // memory pressure by default, or as `LAMBDA_VM_ALLOC_PURGE` names it
    // (counted in the whole run, in no phase's wall).
    crate::alloc_purge::purge_point("base");

    // ---- the harvest. Production's verify of the base runs, by default, on a
    // helper thread beside level 0 and is joined before anything is reported: a
    // refused block still fails the run, it only stops delaying the leaves (the
    // epoch tree's per-epoch verifies likewise run beside its other wraps).
    // `NOEPOCH_HARVEST_VERIFY=inline` verifies first.
    let inline_verify = cfg.harvest_verify_inline;
    let t = Instant::now();
    let mut pipe = None;
    let consts = match consts_beside.as_ref() {
        None => None,
        Some(_) => {
            let tj = Instant::now();
            let (consts, secs, ahead, late_line) = ready_rx
                .recv()
                .map_err(|_| "the ELF constants thread stopped".to_string())?;
            sink.line(&format!(
                "   ELF constants beside the base: {secs:.2}s on {} thread(s), joined after the base, \
                 waited {:.2}s (counted in the harvest)",
                elf_beside.unwrap_or(0),
                tj.elapsed().as_secs_f64()
            ));
            if let Some((shape, derived, secs, done_at)) = ahead {
                let (filled, phases) =
                    derived.map_err(|e| format!("the tree derives ahead: {e}"))?;
                if format!("{shape:?}") != format!("{:?}", BlockShape::of_proof(&proof)) {
                    return Err("the shape handed out before the prove is the proof's".to_string());
                }
                let split: Vec<String> = phases
                    .iter()
                    .map(|p| {
                        format!(
                            "{} in {:.2} (emit Σ {:.2}, build Σ {:.2})",
                            p.programs, p.wall, p.emit, p.build
                        )
                    })
                    .collect();
                sink.line(&format!(
                    "   TREE AHEAD: {} programs derived beside the base in {secs:.2}s on the host (shape at \
                     {:.2}s, done at {done_at:.2}s of a {base:.2}s base) · {}",
                    phases.iter().map(|p| p.programs).sum::<usize>(),
                    shape_at.unwrap_or(f64::NAN),
                    split.join(" · ")
                ));
                pipe = Some(filled);
            }
            if let Some(line) = late_line {
                sink.line(&line);
            }
            Some(Arc::new(
                consts.map_err(|e| format!("the ELF constants compute: {e}"))?,
            ))
        }
    };
    if tree_ahead.is_some() && pipe.is_none() {
        return Err("NOEPOCH_TREE_AHEAD derived no tree".to_string());
    }
    let shape = BlockShape::of_proof(&proof);
    let (rb, verify, replay, beside) = if inline_verify {
        let (rb, verify, replay) =
            harvest_block_over(&inner, &elf_bytes, &proof, true, consts.as_deref(), &*sink)
                .map_err(|e| format!("harvest: {e}"))?;
        drop(proof);
        (rb, Some(verify), replay, None)
    } else {
        let proof = Arc::new(proof);
        let beside = {
            let (proof, opts, elf) = (proof.clone(), inner.clone(), elf_bytes.clone());
            let (consts, sink) = (consts.clone(), sink.clone());
            std::thread::spawn(move || {
                harvest_block_over(&opts, &elf, &proof, true, consts.as_deref(), &*sink)
                    .map(|(rb, verify, _)| (rb.state, verify))
            })
        };
        let (rb, _, replay) =
            harvest_block_over(&inner, &elf_bytes, &proof, false, consts.as_deref(), &*sink)
                .map_err(|e| format!("harvest: {e}"))?;
        drop(proof);
        (rb, None, replay, Some(beside))
    };
    let mut harvest = t.elapsed().as_secs_f64();
    sink.line(&format!(
        "   harvest: {harvest:.2}s (production verify {}, harness-only · replay {replay:.2}s) · \
         {} instances · public output {} bytes",
        match verify {
            Some(v) => format!("{v:.2}s inline"),
            None => "beside level 0".to_string(),
        },
        rb.num_instances(),
        rb.public_output.len()
    ));

    // ---- the partition: the plan's — the rule over the closed-form costs, a
    // pure function of the block's shape. `NOEPOCH_LEAVES` forces another leaf
    // count: another tree, which the final check then derives over.
    let forced = cfg.forced_leaves;
    #[cfg(test)]
    let rb = match forced {
        Some(k) => {
            let p = rb.try_partition_over(k)?;
            BlockWitness {
                plan: rb.plan.with_partition(p),
                ..rb
            }
        }
        None => rb,
    };
    let partition = rb.plan.partition().clone();
    let names = rb.names();
    let note = sink.partition_note(&names, &partition);
    sink.line(&format!(
        "   BLOCK PARTITION SOURCE: {}{}",
        if forced.is_some() {
            "NOEPOCH_LEAVES (forced: NOT the plan's tree)"
        } else {
            "the plan (the rule)"
        },
        note.map_or(String::new(), |n| format!(
            " · D-NOEPOCH §12.2's lists (legmodel.py costs): {n}"
        ))
    ));
    let costs = rb.plan.costs();
    let k = partition.num_leaves();
    for (j, leaf) in partition.leaves().iter().enumerate() {
        let perms: usize = leaf.iter().map(|&i| costs[i]).sum();
        sink.line(&format!(
            "   BLOCK LEAF {j}: {} instances · {perms} perms (closed form) · idx {leaf:?}",
            leaf.len()
        ));
    }
    sink.line(&format!(
        "   BLOCK PARTITION: {k} leaves, Σ {} perms, heaviest {} (cap {}, cost model v{})",
        costs.iter().sum::<usize>(),
        partition
            .leaves()
            .iter()
            .map(|l| l.iter().map(|&i| costs[i]).sum::<usize>())
            .max()
            .unwrap_or(0),
        super::block_plan::LEAF_PERMS_CAP,
        rb.plan.cost_model()
    ));
    // Which leaves verify the chunked accelerators' instances (the rule seeds
    // each table's first instance and fills its later chunks by load).
    let leaves_of = |kind: &str| -> Vec<usize> {
        let prefix = format!("{kind}[");
        (0..k)
            .filter(|&j| {
                partition
                    .leaf(j)
                    .iter()
                    .any(|&i| names[i].starts_with(&prefix))
            })
            .collect()
    };
    sink.line(&format!(
        "   BLOCK PARTITION CHUNKED: KECCAK on leaves {:?} · ECSM {:?} · ECDAS {:?} · KECCAK_RND {:?}",
        leaves_of("KECCAK"),
        leaves_of("ECSM"),
        leaves_of("ECDAS"),
        leaves_of("KECCAK_RND")
    ));
    drop(names);

    // ---- level 0: the leaves.
    let beside_verifies = cfg.child_verify_beside.then(BesideVerifies::default);
    let l0 = cfg.siblings_l0.min(k);
    sink.line(&format!(
        "   ★ LEVEL-0 CONCURRENCY: {l0} leaf proof(s) at once (LFM_TREE_SIBLINGS_L0)"
    ));
    super::device_permit::arm(l0);
    // The pipeline's builder starts on the card now that the base is done.
    let _ = go_tx.send(());
    let l0_sampler = HostSampler::start();
    let t = Instant::now();
    let failed = std::sync::atomic::AtomicBool::new(false);
    let leaves = in_index_order(k, l0, |j| {
        if failed.load(std::sync::atomic::Ordering::SeqCst) {
            return Err("skipped: another leaf failed".to_string());
        }
        let out = (|| -> Result<HarvestedChild, String> {
            let label = format!("BLOCK L0 leaf {j}");
            let te = Instant::now();
            let (program, built) = match &pipe {
                Some(p) => {
                    let (program, artifacts) = p.take_leaf(j, &label)?;
                    (program, Some(artifacts))
                }
                None => (block_leaf_program(&rb, j)?, None),
            };
            let arenas = block_leaf_arenas(&rb, &partition, j)?;
            sink.line(&format!(
                "   {label}: {} + arenas in {:.2}s",
                if built.is_some() {
                    "program ahead"
                } else {
                    "emitted"
                },
                te.elapsed().as_secs_f64()
            ));
            sink.write(
                &super::tree_run::census_panel(&program, &label, super::block_plan::block_fan_in())
                    .2,
            );
            let (child, proof) = prove_program_with(
                &label,
                &program,
                built,
                &arenas,
                &wrap_opts,
                beside_verifies.is_none(),
                &*sink,
            )?;
            if let Some(b) = &beside_verifies {
                b.spawn(&label, child.artifacts.clone(), proof, wrap_opts.clone());
            }
            let words: Vec<LfmWord> = child.public_words.iter().map(|(_, w)| *w).collect();
            if words != rb.expected_leaf_publics(&partition, j, j == rb.plan.carrier()) {
                return Err(format!(
                    "{label} publishes the host's id, state, output and share"
                ));
            }
            Ok(child)
        })();
        if out.is_err() {
            failed.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        out
    });
    let leaves = leaves.into_iter().collect::<Result<Vec<_>, String>>()?;
    let level0 = t.elapsed().as_secs_f64();
    let (l0_peak, _) = l0_sampler.stop();
    sink.line(&format!(
        "   BLOCK LEVEL 0: {k} leaves in {level0:.2}s · host peak {l0_peak:.3} GiB{}",
        pct(l0_peak)
    ));

    // ---- the interior and the top.
    let siblings = cfg.siblings;
    sink.line(&format!(
        "   ★ SIBLING CONCURRENCY: {siblings} node proof(s) at once (LFM_TREE_SIBLINGS)"
    ));
    super::device_permit::arm(siblings);
    let t = Instant::now();
    let (top, top_proof, walls) = compose_block_tree_with(
        &rb.plan,
        leaves,
        &wrap_opts,
        siblings,
        pipe.as_deref(),
        beside_verifies.as_ref(),
        &*sink,
    )?;
    // The pipeline's builder is done by now (every node took its program).
    if let Some(handle) = consts_beside {
        let built = handle
            .join()
            .map_err(|_| "the ELF constants thread panicked".to_string())?;
        if let Some(times) = built {
            let times = times.map_err(|e| format!("the tree's builder: {e}"))?;
            sink.line(&format!(
                "   TREE PIPE: leaf artifacts built by {:.2}s of level 0, node levels by {:?} s · node \
                 emission Σ {:.2}s ({}) · artifact builds Σ {:.2}s (holding the card permit) · leaf \
                 programs {}",
                times.leaves,
                times.levels,
                times.emit,
                if emit_threads > 0 {
                    format!("own pool of {emit_threads}")
                } else {
                    "global pool".to_string()
                },
                times.build,
                match emit_window {
                    Some(w) =>
                        format!("emitted in a window of {w} (the first {w} beside the base)"),
                    None => "all emitted beside the base".to_string(),
                }
            ));
            sink.line(&format!(
                "   TREE PIPE node emission: {}",
                if node_emit_early {
                    "early (each node once its children's artifacts exist)"
                } else {
                    "per level (each level once the level below is built)"
                }
            ));
        }
    }
    super::device_permit::arm(1);
    // Every child verified beside is joined here, inside the interior's time: a
    // refusal fails the run before anything is reported.
    if let Some(b) = &beside_verifies {
        let (n, waited) = b.join_all()?;
        sink.line(&format!(
            "   CHILD VERIFIES beside the timed path: {n} accepted (production's verify), the join \
             waited {waited:.2}s (counted in the interior)"
        ));
    }
    let interior = t.elapsed().as_secs_f64();
    // The verify beside level 0 must have accepted the block, over the same
    // reconstruction the leaves read (one state word), before anything counts.
    // What the join waits is on the critical path, so it is the harvest's.
    if let Some(beside) = beside {
        let t = Instant::now();
        let (state, verify) = beside
            .join()
            .map_err(|_| "the harvest verify beside level 0 panicked".to_string())?
            .map_err(|e| format!("harvest (verified beside level 0): {e}"))?;
        if state != rb.state {
            return Err(
                "the verified harvest and the leaves' harvest read one transcript".to_string(),
            );
        }
        let waited = t.elapsed().as_secs_f64();
        harvest += waited;
        sink.line(&format!(
            "   harvest verify beside level 0: production verify {verify:.2}s, harness-only · \
             joined after the top, waited {waited:.2}s (counted in the harvest)"
        ));
    }
    let words: Vec<LfmWord> = top.public_words.iter().map(|(_, w)| *w).collect();
    if words != rb.expected_top_publics() {
        return Err(
            "the top node publishes the block's claim: the attestation id, the state and \
             the public output"
                .to_string(),
        );
    }
    if !top_claims(&rb.plan, &top.public_words, &rb.public_output) {
        return Err("the block verifier's claim check accepts the honest top".to_string());
    }
    let t = Instant::now();
    // ★ The block tree's FINAL check over the plan the run holds: the top proof
    // against the artifacts of the top program the plan derives — never against
    // a program named by whoever produced the proof. The partition is bound
    // only through program identity, so this is where a tree over any other
    // partition is refused. Production's whole verifier, which derives the plan
    // itself, is [`super::block_plan::verify_block_tree`].
    if !super::proof::verify_against_artifacts(
        &top.artifacts,
        &top_proof.proof,
        &top_proof.public_words,
        &wrap_opts,
    ) {
        return Err(
            "the top proof must verify against the top program the run emitted".to_string(),
        );
    }
    sink.line(&format!(
        "   BLOCK FINAL CHECK (harness): the top verifies against its emitted program \
         ({:.2}s, program id {})",
        t.elapsed().as_secs_f64(),
        hex_id(&top.artifacts.program_id)
    ));

    let total = t_all.elapsed().as_secs_f64();
    let (peak, at) = whole.stop();
    sink.line(&format!(
        "★★★ NO-EPOCH BLOCK: base {base:.2}s · harvest {harvest:.2}s · level 0 {level0:.2}s \
         ({k} leaves) · interior {interior:.2}s (levels {}) · recursion {:.2}s · whole {total:.2}s",
        walls
            .iter()
            .map(|w| format!("{w:.2}"))
            .collect::<Vec<_>>()
            .join(" + "),
        harvest + level0 + interior
    ));
    sink.line(&format!(
        "★★★ WHOLE RUN: host peak {peak:.3} GiB at t={at:.1}, {total:.1}s total"
    ));

    Ok(BlockTreeRun {
        shape,
        public_output: rb.public_output.clone(),
        top_program_id: top.artifacts.program_id,
        top_proof,
        times: BlockTreeTimes {
            base,
            harvest,
            level0,
            interior,
            levels: walls,
            whole: total,
            host_peak: (peak, at),
        },
        #[cfg(test)]
        top,
        #[cfg(test)]
        witness: rb,
        #[cfg(test)]
        consts,
        #[cfg(test)]
        forced,
    })
}

/// What the ELF-constants thread hands back on `ready`: the constants and
/// their seconds; the tree derived ahead (the shape it was derived from, the
/// pipe or the filled tree with its phases, its seconds, and when it was done
/// in the base's clock); the late emission's line.
type BesideResult = (
    Result<ElfConstants, String>,
    f64,
    Option<(
        BlockShape,
        Result<(Arc<Pipe>, Vec<super::block_plan::PhaseTimes>), String>,
        f64,
        f64,
    )>,
    Option<String>,
);

#[cfg(test)]
mod tests {
    use super::*;

    /// The default emits late only above its floor, a margin forces it, `off`
    /// never.
    #[test]
    fn the_default_late_emission_needs_its_floor() {
        const G: usize = 1 << 30;
        assert_eq!(late_margin(LateMode::Auto, 30 * G), Some(LATE_MARGIN));
        assert_eq!(
            late_margin(LateMode::Auto, LATE_MIN_ESTIMATE),
            Some(LATE_MARGIN)
        );
        assert_eq!(late_margin(LateMode::Auto, 4 * G), None);
        assert_eq!(late_margin(LateMode::Margin(2.0), G), Some(2.0));
        assert_eq!(late_margin(LateMode::Off, 30 * G), None);
    }

    /// `NOEPOCH_TREE_EMIT_LATE`'s trigger: a fall of `need` below the shape's
    /// live bytes starts the emission; otherwise the base's return or a heap
    /// that has settled does; nothing else does.
    #[test]
    fn the_late_emission_waits_for_its_room() {
        const G: usize = 1 << 30;
        // The median, spill off: 68 GiB live at the shape, a fall of 30 GiB needed.
        assert_eq!(late_trigger(68 * G, 60 * G, 30 * G, 0.5, false), None);
        assert_eq!(
            late_trigger(68 * G, 38 * G, 30 * G, 0.5, false),
            Some("fell")
        );
        // Phase B's transients lifting the live bytes over the shape's are no fall.
        assert_eq!(late_trigger(68 * G, 70 * G, 30 * G, 0.5, false), None);
        let settled = LATE_SETTLE_SECS;
        assert_eq!(
            late_trigger(68 * G, 60 * G, 30 * G, settled - 0.1, false),
            None
        );
        assert_eq!(
            late_trigger(68 * G, 60 * G, 30 * G, settled, false),
            Some("settled")
        );
        assert_eq!(
            late_trigger(68 * G, 60 * G, 30 * G, 0.5, true),
            Some("base returned")
        );
        // The fall is reported over the other two.
        assert_eq!(
            late_trigger(68 * G, 38 * G, 30 * G, settled, true),
            Some("fell")
        );
    }

    /// Every knob reads the harness's defaults when unset, and a value it
    /// cannot take is an error, not a panic.
    #[test]
    fn the_knobs_read_the_harness_defaults_and_refuse_nonsense() {
        assert_eq!(parse_elf_beside(None), Ok(Some(ELF_BESIDE_THREADS)));
        assert_eq!(parse_elf_beside(Some("")), Ok(Some(ELF_BESIDE_THREADS)));
        assert_eq!(parse_elf_beside(Some("0")), Ok(None));
        assert_eq!(parse_elf_beside(Some("2")), Ok(Some(2)));
        assert!(parse_elf_beside(Some("four")).is_err());

        assert_eq!(parse_tree_ahead(None), Ok(Some(AheadMode::Pipe)));
        assert_eq!(parse_tree_ahead(Some("pipe")), Ok(Some(AheadMode::Pipe)));
        assert_eq!(parse_tree_ahead(Some("1")), Ok(Some(AheadMode::Host)));
        assert_eq!(parse_tree_ahead(Some("0")), Ok(None));
        assert!(parse_tree_ahead(Some("2")).is_err());

        assert_eq!(parse_emit_pool(None), Ok(EMIT_POOL_THREADS));
        assert_eq!(parse_emit_pool(Some("0")), Ok(0));
        assert!(parse_emit_pool(Some("-1")).is_err());

        assert_eq!(parse_node_emit(None), Ok(true));
        assert_eq!(parse_node_emit(Some("level")), Ok(false));
        assert!(parse_node_emit(Some("late")).is_err());

        assert_eq!(parse_emit_window(None), Ok(None));
        assert_eq!(parse_emit_window(Some("3")), Ok(Some(3)));
        assert!(parse_emit_window(Some("0")).is_err());

        assert_eq!(parse_emit_late(None), Ok(LateMode::Auto));
        assert_eq!(parse_emit_late(Some("off")), Ok(LateMode::Off));
        assert_eq!(parse_emit_late(Some("1.5")), Ok(LateMode::Margin(1.5)));
        assert!(parse_emit_late(Some("-1")).is_err());

        assert_eq!(parse_leaves(None), Ok(None));
        assert_eq!(parse_leaves(Some("8")), Ok(Some(8)));
        assert!(parse_leaves(Some("0")).is_err());
    }

    /// The posture line names every posture knob, in the table's order.
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
        assert!(
            POSTURE.iter().any(|(n, _)| *n == POSTURE_VRAM_KNOB),
            "the VRAM budget is a posture knob"
        );
    }
}
