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

use multilinear::whir_chain::{ArgueFormat, GkrPack, GroupFold, LeafOrder, Round0Trees};

use crate::block_whir::{self, BlockFormat, BlockOptions, BlockWhirProof};
use crate::tables::keccak_rnd::KeccakRndForm;
use crate::tables::memw_aligned::MemwAlignedOrder;
use crate::zf_format::WhirBlowup;

use super::compiler::LfmProgram;
use super::harvest::{HarvestedChild, harvest_child_under, try_child_arena_words};
use super::per_table_aggregator::DerivedChild;
use super::proof::{
    LfmFilled, LfmProof, aggregation_wrap_options, decide_lfm_residency, lfm_execute_and_fill,
    take_prove_split,
};
use super::whir_block::{
    BLOCK_FAN_IN, LeafScale, WhirBlockPlan, artifacts_of, block_leaf_arena, group_arena_words,
    leaf_arena, leaf_scale_from_env,
};
use super::whir_chain::LeafQuery;
use super::word::LfmWord;
use crate::hash_pin::{BlockFamily, ProofFamily};

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
/// The base hash is not posture: it is the build's (`hash_pin::BLOCK_BASE`, the
/// only base `block_whir::checked_base` admits), and `LAMBDA_VM_WHIR_HASH` is
/// not read on the block path. The measurement-only `LAMBDA_VM_BASE_SPLIT` is not posture, and
/// the allocator's never-purge posture is compiled into the binary. Neither is
/// `LAMBDA_VM_MAX_ROWS_LOG2`: the block cuts every chunked table at 2^21 rows
/// ([`BlockOptions::production`]), a height its verifier bounds, and the word
/// is read only by the epoch pipeline and the monolithic prove.
/// `LFM_PRECOMPUTED_TREE_CACHE_CAP`'s posture follows the host target
/// ([`posture_tree_cache_cap`]): the table's 64 from a 64 GiB target up, 16
/// below it.
pub const POSTURE: &[(&str, &str)] = &[
    ("TABLE_PARALLELISM", "4"),
    ("LFM_WHIR_RETENTION", "1"),
    ("LFM_PRECOMPUTED_TREE_CACHE_CAP", "64"),
    ("LFM_EXEC_PARALLEL", "1"),
    ("LAMBDA_VM_GRIND_SCAN_FACTOR", "8"),
    ("LAMBDA_VM_GRIND_GRID", "1024"),
];

/// The precomputed-tree cache's knob, whose posture follows the host target.
pub const POSTURE_TREE_CACHE_KNOB: &str = "LFM_PRECOMPUTED_TREE_CACHE_CAP";

/// The host target (GiB, [`block_whir::memory_target_bytes`]) from which the
/// posture keeps [`POSTURE`]'s 64 precomputed-tree cache entries.
pub const POSTURE_TREE_CACHE_FULL_TARGET_GIB: u64 = 64;

/// `LFM_PRECOMPUTED_TREE_CACHE_CAP`'s posture for a host `target` in bytes:
/// 64 entries (≈ 100 MiB each, ≈ 6.2 GiB once full) from a
/// [`POSTURE_TREE_CACHE_FULL_TARGET_GIB`] target up, 16 below it, where the
/// ≈ 4.7 GiB they free count for more than the hits they lose. A miss only
/// rebuilds the tree: the cache's key is the root, so nothing a proof commits
/// to moves (#1013's rule, 49c4bb512).
pub fn posture_tree_cache_cap(target: u64) -> &'static str {
    if target >= POSTURE_TREE_CACHE_FULL_TARGET_GIB << 30 {
        "64"
    } else {
        "16"
    }
}

/// The `BLOCK POSTURE:` line: each posture knob as the process has it — equal
/// to the posture, unset, or another value.
pub fn posture_line() -> String {
    let words: Vec<String> = POSTURE
        .iter()
        .map(|&(name, want)| {
            if name == POSTURE_TREE_CACHE_KNOB {
                (
                    name,
                    posture_tree_cache_cap(block_whir::memory_target_bytes()),
                )
            } else {
                (name, want)
            }
        })
        .map(|(name, want)| match std::env::var(name) {
            Ok(v) if v == *want => format!("{name}={v}"),
            Ok(v) => format!("{name}={v} (≠ posture {want})"),
            Err(_) => format!("{name} unset (≠ posture {want})"),
        })
        .collect();
    format!("BLOCK POSTURE: {}", words.join(" · "))
}

/// The `BLOCK BASE:` line: the base hash and 4-ary cap the format names, the
/// block path's only source for its hash (`LAMBDA_VM_WHIR_HASH` is not read
/// there, D-WHIR-P1 S7).
pub fn base_line(base: &stark::proof::options::BaseFormat) -> String {
    format!(
        "BLOCK BASE: {:?} · 4-ary cap {:?} · the format's ({} is not read on the block path)",
        base.hash,
        base.arity4_cap,
        crate::whir_hash_knob::ENV
    )
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
    /// `W3_LEAF_ARTIFACTS_EACH=0|1` (default 0): the leaves' artifacts are
    /// published all at once after the last is built while every leaf's
    /// program is there, and each as it is built from the first leaf whose
    /// program is not ([`publish_each`]); or (1, a measurement arm) each as it
    /// is built throughout.
    pub leaves_each: bool,
    /// `LAMBDA_VM_TREE_PROGRAM_BUDGET` (`auto` by default): how many bytes of
    /// tree programs may exist ahead of their provers
    /// ([`super::program_budget`]), against the memory target.
    pub program_budget: super::program_budget::BudgetSetting,
    /// `W3_LEAF_DURING_PHASE_B=0|1` (default 1): the first leaf whose groups
    /// phase B finished while another group was still to come executes and
    /// fills right then, or (0, the control) with the others after the base.
    pub early_on: bool,
    /// `W3_EARLY_LEAVES=<n>` (default [`EARLY_LEAVES_DEFAULT`], every leaf
    /// that can): with the early leaf on and the leaves offered to phase B
    /// (`W3_LEAF_FIRST`), phase B leads with the groups of up to `n` leaves, one
    /// leaf after another ([`block_whir::PhaseBLead::leading`]), and each of
    /// them that completes while a group is still to come, and whose execute +
    /// fill is predicted to end before the base ([`early_leaf_fits`]), executes
    /// and fills right then, on a thread of its own, host only. The tree starts
    /// its own leaves' execute first ([`early_leaves_order`]). 1 is the single
    /// early leaf this replaced (the control). Prover-only: the bytes are the
    /// same.
    pub early_leaves: usize,
    /// `W3_EARLY_POOL=<threads>` (default [`EARLY_POOL_DEFAULT`]): the early
    /// leaves after the first execute and fill on a pool of this many threads
    /// of their own, below the prover's priority on Linux
    /// ([`EARLY_POOL_NICE`]), so phase B's host work, which they run beside,
    /// keeps its cores. 0: on the prover's own pool, beside phase B's work.
    pub early_pool: usize,
    /// `W3_LEAF_EMITTERS` ([`LEAF_EMITTERS`] by default): how many of the
    /// leaves' programs are emitted at once beside phase B, in leaf order,
    /// each published as it is ready while the planner takes phase B's groups;
    /// or (0, the control: `None`) one thread a leaf, all at once, the first
    /// group taken after the last is emitted.
    pub leaf_emitters: Option<usize>,
    /// `BLOCK_WHIR_ARGUE=batched|per-table` (production: batched, at the
    /// format's bin cap; `BLOCK_WHIR_ARGUE_CAP=k` for 2^k). Another argue is
    /// another format than the production verifier's.
    pub argue: ArgueFormat,
    /// The block format: production, with `argue`, its groups' stacked
    /// polynomials opened per `LAMBDA_VM_WHIR_GROUP=per-poly|indexed`
    /// ([`group_setting`]; production: per-poly).
    pub format: BlockFormat,
    /// The base's options: production, with the `BLOCK_WHIR_*` knobs.
    pub options: BlockOptions,
    /// `W3_EMIT_THREADS` (4 by default): the nodes' programs and artifacts
    /// are built on a pool of this many threads of their own, each level
    /// proving as its own nodes arrive.
    pub emit_threads: usize,
    /// `W3_NODE_TIMES=0|1` (default 0): the per-program stamps readout after
    /// the tree.
    pub node_times: bool,
    /// `W3_LEAF_FIRST=0|1` (default 1): with the early leaf on, the planner
    /// offers the leaves to phase B as soon as it has derived the plan, and
    /// phase B takes the groups of the leaf it can finish soonest next
    /// ([`block_whir::PhaseBLead`]); the tree's workers take that early leaf
    /// first. Or (0, the control) phase B keeps its own order and the tree
    /// takes the leaves in index order. Prover-only: the bytes are the same.
    pub leaf_first: bool,
    /// `LAMBDA_VM_BLOCK_COMPRESS` (the four compression fields are
    /// [`super::compress::compress_plan`]'s; default
    /// [`super::compress::DEFAULT_PLAN`], `=0` the rollback): after the top,
    /// one compression proof of the top's verification (C1) at this rate,
    /// which then ships instead of the top.
    pub compress: Option<super::wlfm::WlfmRate>,
    /// `LAMBDA_VM_BLOCK_COMPRESS_EVAL`, `_FINAL` and `_FOLDS`: how C1 binds
    /// its instruction columns and where its chains stop committing.
    pub compress_levers: super::wlfm::WlfmLevers,
    /// `LAMBDA_VM_BLOCK_COMPRESS_WIRE`: the shipped file's encoding.
    pub compress_wire: super::wire::WireFormat,
    /// `LAMBDA_VM_BLOCK_COMPRESS2`: after C1, a second compression proof C2
    /// that verifies C1 in-guest, at this query grind; what ships then.
    pub compress2: Option<u8>,
}

/// `W3_EARLY_LEAVES` unset: every leaf that can lead and fits goes early
/// (measured on a 9950X with a 5090: whole block −0.40 s on a median block,
/// −0.76 s on a large one, against a single early leaf).
pub const EARLY_LEAVES_DEFAULT: usize = usize::MAX;

/// `W3_EARLY_LEAVES`: unset is [`EARLY_LEAVES_DEFAULT`]; 1 is the single early
/// leaf (the control); 0 is refused, since `W3_LEAF_DURING_PHASE_B=0` is the
/// knob that turns early leaves off.
fn early_leaves_setting(knob: Option<usize>) -> Result<usize, String> {
    match knob {
        None => Ok(EARLY_LEAVES_DEFAULT),
        Some(0) => Err(
            "W3_EARLY_LEAVES=0: a count of at least 1 (W3_LEAF_DURING_PHASE_B=0 turns the early \
             leaf off)"
                .to_string(),
        ),
        Some(n) => Ok(n),
    }
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

/// An on/off knob read as a count ([`count_knob`]'s value): unset or `0` is
/// off, `1` on; any other count is an error naming the knob, as a word that is
/// no count is.
fn flag_setting(name: &str, knob: Option<usize>) -> Result<bool, String> {
    match knob {
        None | Some(0) => Ok(false),
        Some(1) => Ok(true),
        Some(n) => Err(format!("{name}={n}: 0 or 1")),
    }
}

/// [`flag_setting`] of the knob `name` from the environment.
fn flag_knob(name: &str) -> Result<bool, String> {
    flag_setting(name, count_knob(name)?)
}

/// `LAMBDA_VM_WHIR_GROUP`'s reading: unset or `indexed` is one chain per run
/// of up to four stacked polynomials ([`GroupFold::Indexed`], production);
/// `per-poly` one chain per polynomial ([`GroupFold::PerPoly`]); anything else
/// an error. A format choice: the block's verifier must be given the same one.
pub fn group_setting(value: Option<&str>) -> Result<GroupFold, String> {
    match value.map(str::trim) {
        None | Some("indexed") => Ok(GroupFold::Indexed),
        Some("per-poly") => Ok(GroupFold::PerPoly),
        Some(other) => Err(format!("LAMBDA_VM_WHIR_GROUP={other}: per-poly or indexed")),
    }
}

/// The CLI's `--blowup` reading: unset or `2` is the production WHIR base code
/// ([`WhirBlowup::Two`]), `4` is the legacy blowup ([`WhirBlowup::Four`]);
/// anything else an error. A format choice: the block's verifier must be given
/// the same one, and the proof file does not carry it.
pub fn blowup_setting(value: Option<&str>) -> Result<WhirBlowup, String> {
    match value.map(str::trim) {
        None | Some("2") => Ok(WhirBlowup::Two),
        Some("4") => Ok(WhirBlowup::Four),
        Some(word) => Err(format!("--blowup {word}: 4 or 2")),
    }
}

/// The CLI's `--leaves` reading: unset or `coset` is the production round-0
/// leaf order ([`LeafOrder::Coset`], Poseidon1 trees only), `natural` the
/// codeword's own order ([`LeafOrder::Natural`]); anything else an error. A
/// format choice: the block's verifier must be given the same one, and the
/// proof file does not carry it.
pub fn leaves_setting(value: Option<&str>) -> Result<LeafOrder, String> {
    match value.map(str::trim) {
        None | Some("coset") => Ok(LeafOrder::Coset),
        Some("natural") => Ok(LeafOrder::Natural),
        Some(word) => Err(format!("--leaves {word}: natural or coset")),
    }
}

/// The CLI's `--trees` reading: how many round-0 trees a block group's chain
/// commits, `each` (one a stacked polynomial) or `one` (one a run); unset is
/// the format's default ([`crate::zf_format::ZfFormat::DEFAULT`]); anything
/// else an error. A format choice: the block's verifier must be given the same
/// one, and the proof file does not carry it.
pub fn trees_setting(value: Option<&str>) -> Result<Round0Trees, String> {
    match value.map(str::trim) {
        None => Ok(crate::zf_format::ZfFormat::DEFAULT.whir_trees),
        Some("each") => Ok(Round0Trees::Each),
        Some("one") => Ok(Round0Trees::One),
        Some(word) => Err(format!("--trees {word}: each or one")),
    }
}

/// `W3 LEAF SCALE`: the format's leaf scale, the leaf count at the single
/// heights and whether the partition doubled them; a fixed leaf count
/// (`W3_LEAVES`) bounds no chip, so no scale applies.
pub fn leaf_scale_line(scale: LeafScale, plan: &WhirBlockPlan) -> String {
    let leaves = plan.partition().num_leaves();
    match plan.leaf_bound() {
        None => {
            format!("W3 LEAF SCALE: {scale} · {leaves} leaves forced (W3_LEAVES): no scale applies")
        }
        Some(bound) if bound.doubled => format!(
            "W3 LEAF SCALE: {scale} · {} leaves at the single heights · doubled to {leaves} leaves",
            bound.single_leaves
        ),
        Some(bound) => format!(
            "W3 LEAF SCALE: {scale} · {} leaves at the single heights · not doubled",
            bound.single_leaves
        ),
    }
}

/// How many leaf programs are emitted at once beside phase B by default
/// (`W3_LEAF_EMITTERS`). One thread a leaf, all started at the statement, took
/// the host from phase B's first groups (measured on a 16-core host with one
/// RTX 5090: at the median 0.61 s a group while 13 emitters ran against
/// 0.45 s after them; on a large block 0.95 s with 26), and
/// nothing needs a leaf's program before the base ends but the early leaf,
/// which is among the first emitted.
pub const LEAF_EMITTERS: usize = 3;

/// `BLOCK_WHIR_ARGUE_PACK`'s reading: unset is the production packing
/// ([`GkrPack::PRODUCTION`]); `off` is one GKR tree a table ([`GkrPack::Off`]);
/// a count `τ` in `0..=40` cuts a table into equal chunks where that saves
/// input cells, each extra tree priced at `2^τ` ([`GkrPack::Chunks`]). A format
/// value: the prover and the verifier of one proof take the same one.
pub fn pack_setting(knob: Option<&str>) -> Result<GkrPack, String> {
    match knob.map(str::trim) {
        None => Ok(GkrPack::PRODUCTION),
        Some("off") => Ok(GkrPack::Off),
        Some(v) => v
            .parse::<u8>()
            .ok()
            .filter(|tau| *tau <= 40)
            .map(|tree_log_cells| GkrPack::Chunks { tree_log_cells })
            .ok_or_else(|| format!("BLOCK_WHIR_ARGUE_PACK={v}: off or a count in 0..=40")),
    }
}

/// `BLOCK_WHIR_KECCAK_RND`'s reading: unset is the pinned preset's form
/// ([`BlockFormat::pinned`], [`KeccakRndForm::Lean`]), so the harness proves
/// what `verify-block` verifies; `lean` names it and `full` names the variant
/// with every byte check ([`KeccakRndForm::Full`]). A format value: the prover
/// and the verifier of one proof take the same one.
pub fn keccak_rnd_setting(knob: Option<&str>) -> Result<KeccakRndForm, String> {
    match knob.map(str::trim) {
        None => Ok(BlockFormat::pinned().keccak_rnd),
        Some("full") => Ok(KeccakRndForm::Full),
        Some("lean") => Ok(KeccakRndForm::Lean),
        Some(v) => Err(format!("BLOCK_WHIR_KECCAK_RND={v}: full or lean")),
    }
}

/// `BLOCK_WHIR_MEMW_A`'s reading: unset is the pinned preset's form
/// ([`BlockFormat::pinned`]), so the harness proves what `verify-block`
/// verifies; `limbs` names the step's two 16-bit limbs in the row
/// ([`MemwAlignedOrder::Limbs`]) and `lt` one LT lookup a row
/// ([`MemwAlignedOrder::Lt`]). A format value: the prover and the verifier of
/// one proof take the same one.
pub fn memw_aligned_setting(knob: Option<&str>) -> Result<MemwAlignedOrder, String> {
    match knob.map(str::trim) {
        None => Ok(BlockFormat::pinned().memw_aligned),
        Some("limbs") => Ok(MemwAlignedOrder::Limbs),
        Some("lt") => Ok(MemwAlignedOrder::Lt),
        Some(v) => Err(format!("BLOCK_WHIR_MEMW_A={v}: limbs or lt")),
    }
}

/// `argue` with its trees laid out per `pack`: only the batched argue has
/// trees to pack.
fn with_pack(argue: ArgueFormat, pack: GkrPack) -> Result<ArgueFormat, String> {
    match (argue, pack) {
        (ArgueFormat::PerTable, GkrPack::Off) => Ok(argue),
        (ArgueFormat::Batched { bin_log_cells, .. }, pack) => Ok(ArgueFormat::Batched {
            bin_log_cells,
            pack,
        }),
        (ArgueFormat::PerTable, _) => {
            Err("BLOCK_WHIR_ARGUE_PACK packs the batched argue's trees: not per-table".to_string())
        }
    }
}

/// `W3_LEAF_EMITTERS`'s reading: unset is [`LEAF_EMITTERS`]; 0 is the control,
/// one thread a leaf.
fn leaf_emitters_setting(knob: Option<usize>) -> Option<usize> {
    match knob {
        Some(0) => None,
        other => Some(other.unwrap_or(LEAF_EMITTERS)),
    }
}

impl WhirTreeConfig {
    /// Every knob from the environment, under the names and defaults the
    /// harness always read. A value a knob cannot take is an error (the
    /// `BLOCK_WHIR_*` readers of [`crate::block_whir`] keep their own refusals).
    pub fn from_env() -> Result<Self, String> {
        // The leaves' scale first: the production format reads the same knob
        // and refuses what it cannot take, which this returns as an error.
        let leaf_scale = leaf_scale_from_env()?;
        let leaf_query = super::whir_block::leaf_query_from_env()?;
        let leaves = count_knob("W3_LEAVES")?;
        let siblings = count_knob("W3_SIBLINGS")?.unwrap_or(3);
        let fan_in = count_knob("W3_FAN_IN")?.unwrap_or(BLOCK_FAN_IN);
        let leaves_each = flag_knob("W3_LEAF_ARTIFACTS_EACH")?;
        let early_on = count_knob("W3_LEAF_DURING_PHASE_B")?.is_none_or(|v| v != 0);
        let early_leaves = early_leaves_setting(count_knob("W3_EARLY_LEAVES")?)?;
        let early_pool = count_knob("W3_EARLY_POOL")?.unwrap_or(EARLY_POOL_DEFAULT);
        let leaf_emitters = leaf_emitters_setting(count_knob("W3_LEAF_EMITTERS")?);
        let argue = match std::env::var("BLOCK_WHIR_ARGUE").as_deref().map(str::trim) {
            Ok("batched") => match count_knob("BLOCK_WHIR_ARGUE_CAP")? {
                Some(cap) => ArgueFormat::batched(
                    u8::try_from(cap)
                        .map_err(|_| format!("BLOCK_WHIR_ARGUE_CAP={cap}: a cap below 2^256"))?,
                ),
                None => ArgueFormat::BATCHED,
            },
            Ok("per-table") => ArgueFormat::PerTable,
            Err(_) => BlockFormat::production().argue,
            Ok(other) => return Err(format!("BLOCK_WHIR_ARGUE={other}: per-table or batched")),
        };
        // Unset, the argue keeps its own packing (the production one when
        // batched); set, the knob's packing replaces it.
        let argue = match std::env::var("BLOCK_WHIR_ARGUE_PACK") {
            Ok(word) => with_pack(argue, pack_setting(Some(&word))?)?,
            Err(_) => argue,
        };
        let format = BlockFormat {
            argue,
            leaf_scale,
            leaf_query,
            keccak_rnd: keccak_rnd_setting(std::env::var("BLOCK_WHIR_KECCAK_RND").ok().as_deref())?,
            memw_aligned: memw_aligned_setting(std::env::var("BLOCK_WHIR_MEMW_A").ok().as_deref())?,
            ..BlockFormat::production_with(group_setting(
                std::env::var("LAMBDA_VM_WHIR_GROUP").ok().as_deref(),
            )?)
        };
        eprintln!(
            "★ BLOCK KECCAK_RND: {} (BLOCK_WHIR_KECCAK_RND; lean drops the 20 byte checks its D \
             lookups imply)",
            match format.keccak_rnd {
                KeccakRndForm::Full => "full",
                KeccakRndForm::Lean => "lean",
            }
        );
        eprintln!(
            "★ BLOCK MEMW_A: {} (BLOCK_WHIR_MEMW_A; limbs proves each timestamp step by two 16-bit \
             limbs in the row, lt by one LT lookup a row)",
            match format.memw_aligned {
                MemwAlignedOrder::Limbs => "limbs",
                MemwAlignedOrder::Lt => "lt",
            }
        );
        let mut options = BlockOptions::production();
        // `BLOCK_WHIR_LAYOUT_WORKERS=n`: the streamed chunks laid out on n threads
        // (production 6; 0 is the inline layout).
        if let Some(n) = count_knob("BLOCK_WHIR_LAYOUT_WORKERS")? {
            options.layout_workers = n;
        }
        // `BLOCK_WHIR_LAYOUT_AHEAD=k` bounds the chunks unpacked at k + 1;
        // `none` lifts the bound (the reverted version); anything else
        // refuses the prove.
        match std::env::var("BLOCK_WHIR_LAYOUT_AHEAD")
            .as_deref()
            .map(str::trim)
        {
            Ok("none") => options.layout_ahead = None,
            Ok(k) => {
                options.layout_ahead = Some(k.parse().map_err(|_| {
                    format!("BLOCK_WHIR_LAYOUT_AHEAD is a count or `none`, got `{k}`")
                })?);
            }
            Err(_) => {}
        }
        // `BLOCK_WHIR_NARROW=wide|card|host` (production card): how phase A holds
        // each group's columns once committed.
        if let Some(narrow) = block_whir::narrow_from_env() {
            options.narrow = narrow;
        }
        // `BLOCK_WHIR_KECCAK_LOG2=k`, `BLOCK_WHIR_ECSM_LOG2=k`: force a KECCAK or
        // ECSM split (production 2^18 / 2^17).
        block_whir::chunk_cuts_from_env(&mut options);
        // `BLOCK_WHIR_REST_LAYOUT=all|<MiB>` (production 4096) and
        // `BLOCK_WHIR_REST_BUILD=<MiB>` (production 2048): the rest's gate (its
        // waves where it is laid out in waves) and the bytes it builds at once.
        block_whir::rest_layout_from_env(&mut options);
        let super::compress::CompressPlan {
            rate: compress,
            levers: compress_levers,
            wire: compress_wire,
            c2: compress2,
        } = super::compress::compress_plan_from_env()?;
        let emit_threads = count_knob("W3_EMIT_THREADS")?.unwrap_or(4).max(1);
        Ok(Self {
            leaves,
            siblings,
            fan_in,
            leaves_each,
            program_budget: super::program_budget::setting_from_env()?,
            early_on,
            early_leaves,
            early_pool,
            leaf_emitters,
            argue,
            format,
            options,
            emit_threads,
            node_times: flag_knob("W3_NODE_TIMES")?,
            leaf_first: count_knob("W3_LEAF_FIRST")?.is_none_or(|v| v != 0),
            compress,
            compress_levers,
            compress_wire,
            compress2,
        })
    }

    /// Whether the tree is the production verifier's: the plan's leaf count,
    /// [`BLOCK_FAN_IN`] and the production argue — with its GKR trees laid out
    /// as the format's packing says, a named constant the block verifier is
    /// given too ([`BlockFormat::with_pack`]) — and the verifier's preset
    /// KECCAK_RND and MEMW_A forms, which the CLI has no flag for.
    pub fn at_presets(&self) -> bool {
        let pack = match self.argue {
            ArgueFormat::Batched { pack, .. } => pack,
            ArgueFormat::PerTable => GkrPack::Off,
        };
        self.leaves.is_none()
            && self.fan_in == BLOCK_FAN_IN
            && self.argue == BlockFormat::production().with_pack(pack).argue
            && self.format.argue == self.argue
            && self.format.keccak_rnd == BlockFormat::pinned().keccak_rnd
            && self.format.memw_aligned == BlockFormat::pinned().memw_aligned
    }
}

// ============================ the tree's machinery ========================

/// A tree program as its emitter made it, with its admission under the tree's
/// program budget (none without one). The program is declared first, so it is
/// dropped before the permit that accounts for it.
pub(crate) struct Admitted<P = LfmProgram> {
    pub(crate) program: P,
    permit: Option<super::program_budget::Permit>,
}

/// One program's place from its emission to its proof: shared by its
/// artifacts' build and its execute, and let go once its proof is done, which
/// frees the program (with the last of its users) and gives its bytes back to
/// the budget.
pub(crate) struct ProgramSlot<P = LfmProgram>(
    std::sync::Mutex<Option<std::sync::Arc<Admitted<P>>>>,
);

impl<P> ProgramSlot<P> {
    pub(crate) fn new(program: P, permit: Option<super::program_budget::Permit>) -> Self {
        Self(std::sync::Mutex::new(Some(std::sync::Arc::new(Admitted {
            program,
            permit,
        }))))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<std::sync::Arc<Admitted<P>>>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The program, until its proof lets it go.
    pub(crate) fn get(&self) -> Result<std::sync::Arc<Admitted<P>>, String> {
        self.lock()
            .clone()
            .ok_or_else(|| "the program was let go after its proof".to_string())
    }

    /// A prover took the program ([`super::program_budget::Permit::claim`]).
    pub(crate) fn claim(&self) {
        if let Some(permit) = self.lock().as_ref().and_then(|a| a.permit.as_ref()) {
            permit.claim();
        }
    }

    /// Its proof is done: the slot lets go of the program.
    pub(crate) fn release(&self) {
        let program = self.lock().take();
        drop(program);
    }
}

/// A node's program as the builder emits it, before its artifacts: all its
/// execute and fill need, and when it was emitted (seconds since the tree
/// started).
pub(crate) struct NodeProgram {
    pub(crate) program: ProgramSlot,
    pub(crate) at: f64,
    /// Seconds its emission took, after its admission: the builder's walk,
    /// the compile and the admission validator.
    pub(crate) emit: f64,
    /// How many of the program's arenas each child supplies, in child order
    /// ([`super::per_table_aggregator::leg_arena_count`] over the shapes the
    /// program was emitted over): where a streamed execution splits them
    /// ([`execute_streamed`]).
    pub(crate) child_arenas: Vec<usize>,
}

/// A leaf program's bytes before it is emitted, a group of it: 0.157 GiB at
/// T200-R (130 leaves, 81.25 GiB of programs; I-T200 §11b), 0.55–0.63 GiB a
/// leaf of 3–4 groups at p90 (I-WFULL §1.3). The budget learns the real bytes
/// when the program is emitted.
pub(crate) const LEAF_BYTES_PER_GROUP: u64 = 161 << 20;

/// The streaming threads that emit, under a bounded budget, the leaf programs
/// not admitted beside phase B (#1013's `DEFERRED_LEAF_EMITTERS`).
const DEFERRED_LEAF_EMITTERS: usize = 4;

/// A program's host bytes as held (capacities): its instructions with what
/// they box, its column groups and its arena schema (#1013's
/// `ProgramBytes::total`).
pub(crate) fn program_bytes(p: &LfmProgram) -> u64 {
    use super::instr::{Addr, Blake3Operands, Instr, KeccakOperands};
    let mut bytes = p.instrs.capacity() * size_of::<Instr>();
    for instr in &p.instrs {
        bytes += match instr {
            Instr::BitDec { bits, .. } => bits.capacity() * size_of::<(Addr, u64)>(),
            Instr::KeccakF(_) => size_of::<KeccakOperands>(),
            Instr::Blake3(_) => size_of::<Blake3Operands>(),
            _ => 0,
        };
    }
    let g = &p.groups;
    for group in [
        &g.const_, &g.balu, &g.xalu, &g.select, &g.bitdec, &g.hash, &g.keccak, &g.blake3, &g.lanes,
        &g.hint, &g.public,
    ] {
        bytes += group.data.capacity() * size_of::<crate::tables::types::FE>();
    }
    (bytes + p.arena_schema.lens.capacity() * size_of::<u32>()) as u64
}

/// The tree's program budget as its emitters use it: each program admitted in
/// prove order ([`tree_order`]'s numbering: the leaves, then each node level)
/// with its estimated bytes before it is emitted.
pub(crate) struct TreeBudget {
    pub(crate) budget: std::sync::Arc<super::program_budget::ProgramBudget>,
    /// Each leaf's estimate: [`LEAF_BYTES_PER_GROUP`] a group.
    leaf_estimates: Vec<u64>,
    /// Where each node level starts in prove order.
    level_starts: Vec<usize>,
    /// The largest node program emitted so far.
    node_seen: std::sync::atomic::AtomicU64,
}

impl TreeBudget {
    /// The budget of a tree of `shape` over leaves of `leaf_groups` groups each.
    pub(crate) fn new(
        budget: std::sync::Arc<super::program_budget::ProgramBudget>,
        leaf_groups: &[usize],
        shape: &[super::per_table_aggregator::Level],
    ) -> Self {
        let mut level_starts = Vec::with_capacity(shape.len());
        let mut next = leaf_groups.len();
        for level in shape {
            level_starts.push(next);
            next += level.arities.len();
        }
        Self {
            budget,
            leaf_estimates: leaf_groups
                .iter()
                .map(|&g| g as u64 * LEAF_BYTES_PER_GROUP)
                .collect(),
            level_starts,
            node_seen: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub(crate) fn leaf_estimate(&self, k: usize) -> u64 {
        self.leaf_estimates.get(k).copied().unwrap_or(0)
    }

    /// Node `j` of node level `lv`'s place in prove order (0 = the level above
    /// the leaves).
    pub(crate) fn node_order(&self, lv: usize, j: usize) -> usize {
        self.level_starts[lv] + j
    }

    /// A node program's estimate before it is emitted: the largest node seen,
    /// or twice the largest leaf before the first (#1013's rule).
    pub(crate) fn node_estimate(&self) -> u64 {
        let leaf = self.leaf_estimates.iter().max().copied().unwrap_or(0);
        self.node_seen
            .load(std::sync::atomic::Ordering::SeqCst)
            .max(2 * leaf)
    }

    pub(crate) fn node_emitted(&self, bytes: u64) {
        self.node_seen
            .fetch_max(bytes, std::sync::atomic::Ordering::SeqCst);
    }
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
    /// Seconds its emission took after its admission (nodes).
    pub(crate) program_emit: f64,
    /// Its program (nodes) and artifacts built.
    pub(crate) built_at: f64,
    /// Its worker took it.
    pub(crate) start: f64,
    /// Its proof done.
    pub(crate) end: f64,
    pub(crate) split: Option<(f64, f64, f64, f64)>,
}

/// One `ALLOC PEAK <phase>` line, when the watermark monitor runs: the phase's
/// highest VmRSS reading with jemalloc's allocated (live) and resident bytes
/// at it ([`crate::alloc_purge::peak_window`]), and the precomputed-tree
/// cache's live bytes now. Under the pipelined tree a level's window runs from
/// the line before to its last proof, so it overlaps the level above.
fn alloc_peak_line(say: &dyn Fn(&str), phase: &str) {
    if let Some(peak) = crate::alloc_purge::peak_window() {
        say(&format!(
            "ALLOC PEAK {phase}: {peak} · tree cache live {:.2} GiB ({} entries)",
            stark::prover::precomputed_tree_cache_live_bytes() as f64 / (1u64 << 30) as f64,
            stark::prover::precomputed_tree_cache_entries()
        ));
    }
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
    /// Every carried polynomial's commit-time answer, in root order.
    pub(crate) commit_ood: Vec<crate::tables::types::FEE>,
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
/// unwinds before publishing leaves an error behind ([`FailUnpublished`]), so
/// no waiter blocks on a value that will never come.
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

/// The leaves' artifacts as their builder publishes them: per leaf, its
/// artifacts, its derived shape and the seconds building them.
pub(crate) type LeafBuilt = (super::registry::LfmArtifacts, DerivedChild, f64);

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

/// Publishes `why` into every slot of a row still empty when it drops: a
/// publisher that stops early (an error, or a panic) leaves no waiter hanging
/// on a value that will never come.
pub(crate) struct FailEmpty<'a, T>(pub(crate) &'a [Published<T>], pub(crate) &'static str);

impl<T> Drop for FailEmpty<'_, T> {
    fn drop(&mut self) {
        for slot in self.0 {
            let _ = slot.0.set(Err(self.1.to_string()));
        }
    }
}

/// The leaves admitted beside phase B (one permit each, in leaf order; `None`
/// without a program budget): emitted on `threads` plain threads, off the rayon
/// pool phase B is using, and each published to its slot strictly in leaf
/// order, as soon as it and every leaf before it are emitted
/// ([`super::program_budget::emit_ordered`]). `published(k)` is called on this
/// thread after leaf k's slot is set. Returns once every admitted leaf's slot
/// holds a program or an error: after an emitter fails, its error is in its
/// slot and the slots after it say the emitter stopped, so no waiter blocks on
/// a program that will never come. The first error is the result.
pub(crate) fn emit_admitted_leaves<P: Send + Sync>(
    slots: &[Published<ProgramSlot<P>>],
    permits: Vec<Option<super::program_budget::Permit>>,
    threads: usize,
    emit: &(dyn Fn(usize) -> Result<P, String> + Sync),
    bytes: &(dyn Fn(&P) -> u64 + Sync),
    published: &mut dyn FnMut(usize),
) -> Result<(), String> {
    let upto = permits.len();
    let permits: Vec<std::sync::Mutex<Option<super::program_budget::Permit>>> =
        permits.into_iter().map(std::sync::Mutex::new).collect();
    let _fail = FailEmpty(&slots[..upto], "the leaf emitter stopped");
    let admitted = |k: usize| -> Result<ProgramSlot<P>, String> {
        let program = emit(k)?;
        let mut permit = permits[k].lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(permit) = permit.as_mut() {
            permit.emitted(bytes(&program));
        }
        Ok(ProgramSlot::new(program, permit))
    };
    let mut failed = None;
    super::program_budget::emit_ordered(0..upto, threads, &admitted, &mut |k, slot| {
        let emitted = match &slot {
            Ok(_) => true,
            Err(e) => {
                failed = Some(e.clone());
                false
            }
        };
        let _ = slots[k].0.set(slot);
        published(k);
        emitted
    });
    failed.map_or(Ok(()), Err)
}

/// Each item's value built one after another on this thread and published to
/// its slot. While `ready(k)` says item k's input is already there, the values
/// are held and published all at once after the last (ULTRA 101: published one
/// by one, each build queued behind a sibling's prove on the card, and the 1×
/// tree's top waited 0.8 s for its last leaf). Before a build whose input is
/// not there yet, everything held is published, and each value from then on
/// as it is built (`each`: from the start), so nothing waits for an input
/// while holding what others need: under a program budget a missing input is
/// a program the budget admits only once the provers take the ones before it.
/// Returns the item where per-item publication began, if it did. A build that
/// fails is published at once and ends the run; every slot still empty when
/// this returns or unwinds gets an error ([`FailEmpty`]).
pub(crate) fn publish_each<I, T>(
    items: &[I],
    slots: &[Published<T>],
    each: bool,
    ready: impl Fn(usize) -> bool,
    build: impl Fn(usize, &I) -> Result<T, String>,
) -> Option<usize> {
    let _fail = FailEmpty(slots, "the publisher stopped");
    let mut held = Vec::new();
    let mut from = each.then_some(0);
    for ((k, item), slot) in items.iter().enumerate().zip(slots) {
        if from.is_none() && !ready(k) {
            from = Some(k);
            for (one, slot) in held.drain(..).zip(slots) {
                let _ = slot.0.set(one);
            }
        }
        let one = build(k, item);
        let failed = one.is_err();
        if from.is_none() && !failed {
            held.push(one);
        } else {
            let _ = slot.0.set(one);
        }
        if failed {
            break;
        }
    }
    for (one, slot) in held.into_iter().zip(slots) {
        let _ = slot.0.set(one);
    }
    from
}

/// A tree's nodes built level by level from the leaves' shapes — leaf `i`'s
/// from `leaf(i)`, of `n_leaves`, which may wait for it to be published: a node
/// over leaves waits for its own children only — each node published to its
/// slot (level, node) as soon as it is built: `admit` lets a node in (its
/// level and index; under a program budget it may wait), then `emit` makes its
/// program from its level, index, children's shapes (in order), whether it is
/// the top and its admission, as the part published to `programs` at once (what
/// a node executes) and the part `finish` builds the rest of the node from (its
/// artifacts: it may take the card); `shape_of` gives a built node's shape for
/// the level above. A node depends on its children's shapes, never on their
/// proofs, so the whole tree can be built while the leaves prove.
///
/// With `pool`, as many plain threads as the pool has take a level's nodes in
/// order, each admitting its node there and emitting it on the pool, and this
/// thread finishes each as its emission ends and publishes it into ITS OWN
/// slot, whatever order they finish in; without, admitted, emitted and finished
/// one after another here. An admission never waits on a rayon worker, nor
/// ahead of a finish another node's prover needs. `finish` always runs on this
/// thread, never on a rayon worker: a worker that waits inside rayon while it
/// holds the card runs queued jobs meanwhile, and a sibling that takes the card
/// is a second hold on one thread (BIG 569). A node's program is published as
/// its emission ends, before its finish starts and whatever the other nodes'
/// finishes are waiting for. A build that fails leaves its error in its slots,
/// and every slot still empty when this returns or unwinds gets one
/// ([`FailUnpublished`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_levels<'l, C: Sync, A: Send, E: Send + Sync, P: Send, N: Send + Sync>(
    shape: &[super::per_table_aggregator::Level],
    leaf: &(dyn Fn(usize) -> Result<&'l C, String> + Sync),
    n_leaves: usize,
    programs: &[Vec<Published<E>>],
    slots: &[Vec<Published<N>>],
    shape_of: impl Fn(&N) -> &C + Sync,
    admit: impl Fn(usize, usize) -> Result<A, String> + Sync,
    emit: impl Fn(usize, usize, &[&C], bool, A) -> Result<(E, P), String> + Sync,
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
        // The level below's shapes. Over the leaves each node waits for its
        // own children as it is emitted (`leaf`); a node level is whole by the
        // time the next one starts.
        let below: Option<Vec<&C>> = match lv {
            0 => None,
            _ => match slots[lv - 1]
                .iter()
                .map(|s| s.wait().map(&shape_of))
                .collect::<Result<Vec<_>, String>>()
            {
                Ok(below) => Some(below),
                Err(_) => return,
            },
        };
        let kids_of = |range: std::ops::Range<usize>| -> Result<Vec<&C>, String> {
            match &below {
                Some(below) => Ok(below[range].to_vec()),
                None => range.map(leaf).collect(),
            }
        };
        let n_below = below.as_ref().map_or(n_leaves, Vec::len);
        let mut at = 0usize;
        let groups: Vec<std::ops::Range<usize>> = arities
            .arities
            .iter()
            .map(|&a| {
                at += a;
                at - a..at
            })
            .collect();
        if groups.last().map_or(0, |g| g.end) != n_below {
            let _ = slots[lv][0].0.set(Err(format!(
                "level {}: arities cover {at} of {n_below} children",
                lv + 1
            )));
            return;
        }
        // A node's kids (over the leaves, each waited for), its admission,
        // then its emission (on the pool, when there is one).
        let made = |j: usize, on: Option<&rayon::ThreadPool>| -> Result<(E, P), String> {
            let kids = kids_of(groups[j].clone())?;
            let admitted = admit(lv, j)?;
            match on {
                Some(pool) => pool.install(|| emit(lv, j, &kids, top, admitted)),
                None => emit(lv, j, &kids, top, admitted),
            }
        };
        match pool {
            Some(pool) => {
                let (tx, rx) = std::sync::mpsc::channel::<(usize, Result<P, String>)>();
                let next = std::sync::atomic::AtomicUsize::new(0);
                std::thread::scope(|scope| {
                    for _ in 0..pool.current_num_threads().max(1) {
                        let (tx, made, publish, next, nodes) =
                            (tx.clone(), &made, &publish, &next, groups.len());
                        // Each program is published here, as its emission
                        // ends: never behind another node's finish, which may
                        // wait for the card (BIG 622: a program queued 2.1 s
                        // behind a sibling's artifacts).
                        scope.spawn(move || {
                            loop {
                                let j = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                if j >= nodes {
                                    return;
                                }
                                if tx.send((j, publish(lv, j, made(j, Some(pool))))).is_err() {
                                    return;
                                }
                            }
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
                for (j, slot) in slots[lv].iter().enumerate() {
                    let rest = publish(lv, j, made(j, None));
                    let _ = slot.0.set(rest.and_then(|p| finish(lv, j, p)));
                }
            }
        }
    }
}

/// The tree's node programs and artifacts ([`build_levels`] over the leaves'
/// derived shapes): each node's program published to `programs` as soon as it
/// is emitted (what its execute and fill need), and the node with its
/// artifacts to `slots` once they are built (what its prove needs). Under a
/// program `budget` each node is admitted in prove order before it is emitted
/// ([`TreeBudget`]), its permit travelling with its program.
///
/// A level's programs are emitted together on `pool`'s threads — the
/// builder's own, not the global pool, on which a prover's join could steal an
/// emission and leave the card idle (#1013's FAST 454) — and this thread
/// builds each one's artifacts (holding the card) as it arrives; each level
/// proves as its own nodes arrive. (Built one after another on one thread
/// instead, level 1 waited for the whole tree: at the median 32 s of serial
/// builds held level 0 open 10.7 s past its last proof, BIG 565 / 568.)
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_nodes(
    plan: &WhirBlockPlan,
    shape: &[super::per_table_aggregator::Level],
    leaves: &[Published<LeafBuilt>],
    programs: &[Vec<Published<NodeProgram>>],
    slots: &[Vec<Published<TreeNode>>],
    wrap: &crate::ProofOptions,
    words: usize,
    t_tree: std::time::Instant,
    pool: &rayon::ThreadPool,
    budget: Option<&TreeBudget>,
) {
    type Emitted = (NodeProgram, (std::sync::Arc<Admitted>, f64));
    let leaf = |i: usize| -> Result<&DerivedChild, String> {
        leaves
            .get(i)
            .ok_or_else(|| format!("no leaf {i}"))?
            .wait()
            .map(|(_, derived, _)| derived)
    };
    build_levels(
        shape,
        &leaf,
        leaves.len(),
        programs,
        slots,
        |node: &TreeNode| &node.derived,
        // Under a program budget each node is admitted in prove order before
        // it is emitted: on an emitter's plain thread, never the pool's.
        |lv, j| match budget {
            Some(b) => b
                .budget
                .acquire(b.node_order(lv, j), b.node_estimate())
                .map(Some),
            None => Ok(None),
        },
        |_, _, kids: &[&DerivedChild], top, permit| -> Result<Emitted, String> {
            let t = std::time::Instant::now();
            let program = plan.node_program(kids, top)?;
            let child_arenas = kids
                .iter()
                .map(|kid| {
                    super::per_table_aggregator::leg_arena_count(&kid.shape(), BlockFamily::WRAP)
                })
                .collect();
            let mut permit: Option<super::program_budget::Permit> = permit;
            if let (Some(b), Some(permit)) = (budget, permit.as_mut()) {
                let bytes = program_bytes(&program);
                b.node_emitted(bytes);
                permit.emitted(bytes);
            }
            let slot = ProgramSlot::new(program, permit);
            let program = slot.get()?;
            let at = t_tree.elapsed().as_secs_f64();
            let emit = t.elapsed().as_secs_f64();
            Ok((
                NodeProgram {
                    program: slot,
                    at,
                    emit,
                    child_arenas,
                },
                (program, emit),
            ))
        },
        |_, _, (program, emitted): (std::sync::Arc<Admitted>, f64)| -> Result<TreeNode, String> {
            let t = std::time::Instant::now();
            let artifacts = artifacts_of(&program.program, wrap);
            let derived = super::whir_block::derived_of(&artifacts, wrap, words)?;
            Ok(TreeNode {
                artifacts,
                derived,
                built: emitted + t.elapsed().as_secs_f64(),
                built_at: t_tree.elapsed().as_secs_f64(),
            })
        },
        Some(pool),
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

/// `order` ([`tree_order`]) with leaf `first`, when it is one, taken before
/// the other leaves: a permutation of the leaves alone, so every node still
/// comes after every leaf and the order stays topological.
pub(crate) fn early_leaf_first(mut order: Vec<TreeAt>, first: Option<usize>) -> Vec<TreeAt> {
    let at = first.and_then(|j| order.iter().position(|&t| t == TreeAt { lv: 0, j }));
    if let Some(at) = at {
        order[..=at].rotate_right(1);
    }
    order
}

/// The early leaves' own pool, threads (`W3_EARLY_POOL`). Half of a 16-core
/// host's cores: the later early leaves' execute + fill ran 1.2–1.4 s on the
/// prover's whole pool with 3–12 s to spare before the base's end, and beside
/// phase B that whole pool cost phase B 0.18–0.36 s of its host work.
pub const EARLY_POOL_DEFAULT: usize = 8;

/// The nice value the early pool's threads run at on Linux: below the
/// prover's threads, so phase B's host work is scheduled first.
pub const EARLY_POOL_NICE: i32 = 10;

/// How much slower a later early leaf's execute + fill is taken to run on the
/// early pool than the first early leaf's on the prover's pool, until one on
/// the early pool has been measured ([`early_leaf_fits`]'s input).
pub const EARLY_POOL_SLOWDOWN: f64 = 2.0;

/// The early leaves' own pool ([`EARLY_POOL_DEFAULT`]): its threads lowered
/// to [`EARLY_POOL_NICE`] on Linux. `None` for 0 threads.
fn early_pool(threads: usize) -> Result<Option<std::sync::Arc<rayon::ThreadPool>>, String> {
    if threads == 0 {
        return Ok(None);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("w3-early-{i}"))
        .start_handler(|_| lower_this_threads_priority(EARLY_POOL_NICE))
        .build()
        .map(|p| Some(std::sync::Arc::new(p)))
        .map_err(|e| format!("the early leaves' pool: {e}"))
}

/// This thread at `nice` (Linux; elsewhere nothing). A refusal leaves it as
/// it was: the order of work changes, never its result.
fn lower_this_threads_priority(nice: i32) {
    #[cfg(target_os = "linux")]
    // SAFETY: setpriority on this thread's own id; no memory is touched.
    unsafe {
        let _ = libc::setpriority(libc::PRIO_PROCESS, libc::gettid() as libc::id_t, nice);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = nice;
}

/// How much longer a later early leaf's execute + fill is taken to run than the
/// first early leaf's, scaled by their costs: it runs beside phase B's end and
/// the tree's start, with more on the host than the first had.
pub(crate) const EARLY_FIT_FACTOR: f64 = 1.3;

/// A later early leaf's fit ([`early_leaf_fits`]): when its execute + fill is
/// predicted to end, when the base is, and whether the first is before the
/// second.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EarlyFit {
    pub(crate) fill_end: f64,
    pub(crate) base_end: f64,
    pub(crate) fits: bool,
}

/// Whether a leaf after the first goes early (`W3_EARLY_LEAVES` > 1): only when
/// its execute + fill, predicted from the first early leaf's measured one scaled
/// by the leaves' plan costs and by [`EARLY_FIT_FACTOR`], ends before the base
/// is predicted to end. The base's end is now plus phase B's groups still to
/// come, at the pace phase B has kept since its first group. Seconds are since
/// the prove started; costs are the plan's. `None` (stay for the tree) while
/// either pace is unknown: the first early leaf's fill has not ended, or no
/// group has followed phase B's first.
pub(crate) fn early_leaf_fits(
    now: f64,
    leaf_cost: u64,
    first_fill: Option<(f64, u64)>,
    first_group_at: Option<f64>,
    cost_since_first_group: u64,
    cost_to_come: u64,
) -> Option<EarlyFit> {
    let (first_secs, first_cost) = first_fill?;
    let t1 = first_group_at?;
    if first_cost == 0 || cost_since_first_group == 0 || now <= t1 {
        return None;
    }
    let pace = (now - t1) / cost_since_first_group as f64;
    let base_end = now + pace * cost_to_come as f64;
    let fill_end = now + EARLY_FIT_FACTOR * first_secs * leaf_cost as f64 / first_cost as f64;
    Some(EarlyFit {
        fill_end,
        base_end,
        fits: fill_end <= base_end,
    })
}

/// How many early leaves the planner may start: `early_leaves`
/// (`W3_EARLY_LEAVES`) when phase B led at least two leaves, else one, the
/// single early leaf's path exactly (with fewer led, the other leaves complete
/// with phase B's last groups).
pub(crate) fn early_room(early_leaves: usize, led: usize) -> usize {
    if early_leaves > 1 && led >= 2 {
        early_leaves
    } else {
        1
    }
}

/// `order` with the leaves `first` taken before the others, in `first`'s
/// order ([`early_leaf_first`] for each, the last first); with one leaf it is
/// [`early_leaf_first`].
pub(crate) fn early_leaves_first(order: Vec<TreeAt>, first: &[usize]) -> Vec<TreeAt> {
    first
        .iter()
        .rev()
        .fold(order, |order, &k| early_leaf_first(order, Some(k)))
}

/// The tree's order with several early leaves: up to `in_tree_ahead` of the
/// leaves the tree still executes (not in `early`), in their order, then the
/// early leaves in `early`'s order, then the other leaves. A leaf the tree
/// executes needs a worker for its whole execute + fill before its prove, while
/// an early leaf is ready and needs only the card; taking the early ones first
/// (on the large block, three early leaves on three workers) left the last leaf's
/// execute waiting 1.8 s for a worker. With `in_tree_ahead` 0 it is
/// [`early_leaves_first`]. The nodes keep their places after every leaf.
pub(crate) fn early_leaves_order(
    order: Vec<TreeAt>,
    early: &[usize],
    in_tree_ahead: usize,
) -> Vec<TreeAt> {
    if in_tree_ahead == 0 {
        return early_leaves_first(order, early);
    }
    let leaves: Vec<TreeAt> = order.iter().copied().filter(|t| t.lv == 0).collect();
    let rest: Vec<TreeAt> = order.iter().copied().filter(|t| t.lv != 0).collect();
    let in_tree: Vec<TreeAt> = leaves
        .iter()
        .copied()
        .filter(|t| !early.contains(&t.j))
        .collect();
    let ahead = in_tree.len().min(in_tree_ahead);
    let early_at: Vec<TreeAt> = early
        .iter()
        .filter_map(|&j| leaves.iter().copied().find(|t| t.j == j))
        .collect();
    in_tree[..ahead]
        .iter()
        .copied()
        .chain(early_at)
        .chain(in_tree[ahead..].iter().copied())
        .chain(rest)
        .collect()
}

/// `W3 LEAF FIRST`: whether the leaves were offered to phase B and when, what
/// phase B did with them (the turn it read the offer, the leaf it led with and
/// how many groups it moved) and which leaf was early.
fn leaf_first_line(
    on: bool,
    offered_at: Option<f64>,
    stated_at: f64,
    taken: Option<block_whir::LeadTaken>,
    early: &[usize],
) -> String {
    if !on {
        return "W3 LEAF FIRST: off (phase B in its own order; the tree takes the leaves in index \
                order)"
            .to_string();
    }
    let offered = offered_at.map_or("never".to_string(), |t| {
        format!("{t:.2}s ({:.2}s after the statement)", t - stated_at)
    });
    let phase_b = match taken {
        None => "phase B ended before it read them".to_string(),
        Some(t) => match t.leaf {
            Some(leaf) if t.leaves.len() <= 1 => format!(
                "phase B read them at its turn {} ({:.2}s in) and moved leaf {leaf}'s {} groups next",
                t.turn,
                t.after,
                t.moved.len()
            ),
            Some(_) => format!(
                "phase B read them at its turn {} ({:.2}s in) and moved leaves {}' {} groups next, \
                 leaf by leaf",
                t.turn,
                t.after,
                t.leaves
                    .iter()
                    .map(|k| k.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                t.moved.len()
            ),
            None => format!(
                "phase B read them at its turn {} ({:.2}s in); no leaf could lead (each holds a \
                 rebuilt group, or none was left)",
                t.turn, t.after
            ),
        },
    };
    let early = match early {
        [] => "early leaf none".to_string(),
        [k] => format!("early leaf {k}"),
        ks => format!(
            "early leaves {}",
            ks.iter()
                .map(|k| k.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    format!(
        "W3 LEAF FIRST: on · leaves offered at {offered} · {phase_b} · {early} (taken first by the \
         tree)"
    )
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
/// `barrier` (a test's control), a node first waits for the whole level
/// below, as level-by-level proving did. A failed or panicking prove
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
/// after the last child. A node's arenas are its children's, in child order:
/// child `c` supplies `child_arenas[c]` of them ([`NodeProgram::child_arenas`])
/// and they are group `c`. The counts differ between children whose AIR sets
/// differ (under the equal-height merge each child's merged tables follow its
/// own heights), so they are never an equal split. The execute the split
/// reports is the part after the last child landed. Returns the filled traces,
/// for a prove under artifacts built for `hasher`.
pub(crate) fn execute_streamed(
    program: &LfmProgram,
    child_arenas: &[usize],
    kids: &[&Published<ProvedProgram>],
    hasher: crate::lfm::hash::HasherKind,
) -> Result<LfmFilled, String> {
    let (execution, after) = execute_streamed_with(program, child_arenas, kids, &hasher, |kid| {
        try_child_arena_words(&kid.child)
    })?;
    Ok(super::proof::lfm_fill_executed(
        program, execution, hasher, after,
    ))
}

/// [`execute_streamed`]'s execution, over children of any kind: `arenas_of` gives
/// a landed child's arenas. Returns the execution and the seconds it took
/// after the last child landed. Refuses counts that do not cover the
/// program's arenas, one count a child.
pub(crate) fn execute_streamed_with<R: Send + Sync>(
    program: &LfmProgram,
    child_arenas: &[usize],
    kids: &[&Published<R>],
    hasher: &(impl super::hash::LfmHasher + Sync),
    arenas_of: impl Fn(&R) -> Result<Vec<Vec<LfmWord>>, String>,
) -> Result<(super::executor::LfmExecution, f64), String> {
    use super::executor::StreamedExecution;
    let k = kids.len();
    let arenas = program.arena_schema.lens.len();
    if k == 0 || child_arenas.len() != k || child_arenas.iter().sum::<usize>() != arenas {
        return Err(format!(
            "{arenas} arenas over {k} children of {child_arenas:?} arenas"
        ));
    }
    let group_of: Vec<usize> = child_arenas
        .iter()
        .enumerate()
        .flat_map(|(c, &n)| std::iter::repeat_n(c, n))
        .collect();
    let held: Vec<std::cell::OnceCell<Vec<Vec<LfmWord>>>> =
        (0..k).map(|_| std::cell::OnceCell::new()).collect();
    let mut ex = StreamedExecution::new(program, group_of, hasher).map_err(|e| format!("{e:?}"))?;
    let mut last = std::time::Instant::now();
    in_arrival_order(kids, |c, kid| {
        let words = match held[c].get() {
            Some(words) => words,
            None => {
                let words = arenas_of(kid)?;
                held[c].get_or_init(|| words)
            }
        };
        last = std::time::Instant::now();
        ex.land(c, words)
            .map_err(|e| format!("child {c} lands: {e:?}"))
    })?;
    let execution = ex.finish().map_err(|e| format!("{e:?}"))?;
    Ok((execution, last.elapsed().as_secs_f64()))
}

/// One node of the tree: its execute and fill (`execute`) from its program as
/// soon as the builder has emitted it, and its prove (`prove`) only once its
/// OWN artifacts are in its slot — the program in `programs[lv − 1][j]`, the
/// artifacts in `slots[lv − 1][j]`. With `early` off (a test's control), the
/// execute also waits for the artifacts, as when one slot held both. The artifacts are a card hold that queues behind the
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
/// execute (its program emitted). With every child already proved there is nothing to stream behind,
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

/// The tree proved the way a prover would run it, with the leaves' programs in
/// `leaves`: those emitted beforehand (while phase B ran) already published,
/// the rest emitted here as the program `budget` admits them, on
/// [`DEFERRED_LEAF_EMITTERS`] streaming threads in leaf order:
/// 1. the leaves' artifacts, one after another on a thread of their own (each
///    holds the card), all published at once after the last while every
///    leaf's program is there, and each as it is built from the first whose
///    program is not ([`publish_each`]; `leaves_each`: each throughout);
/// 2. the nodes' programs and artifacts — functions of the leaves' artifacts,
///    not of any proof — on a thread of their own ([`build_nodes`]), each
///    admitted under the budget, then published to its slot as it is built, a
///    node over leaves once its own leaves' artifacts are, beside
/// 3. every program proved `siblings` at a time, leaves first, each harvested
///    without its verify ([`prove_dataflow`]): each leaf executes and fills
///    its traces while the artifacts are built (neither reads them) and waits
///    for its own only to prove; each node as soon as its own children are
///    proved and its program is in its slot, executing and filling from its
///    program as it is emitted ([`node_flow`]); the top streams its children
///    when one is still unproved as it can execute ([`top_streams`]), its
///    choice kept in `top_streamed`.
///
/// Each program's prover claims it as it takes it and lets it go once it is
/// done with it — a leaf once proved, a node once filled (its artifacts' build
/// holds its own reference) — so a program lives from its emission to its
/// proof, and its admission with it. After the last proof of a level below the
/// top, `LAMBDA_VM_TREE_LEVEL_PURGE` may purge the allocator
/// ([`crate::alloc_purge::purge_after_level`], its line through `say`).
///
/// `node_pipe` is the builder's own pool ([`build_nodes`]); `early`, when
/// given, is a leaf that already executed and filled while phase B ran
/// ([`EarlyLeaf`]): its worker joins it instead.
///
/// Returns each level's timing and every proof with its artifacts, level by
/// level, the top last — for the harness to verify off the clock — the early
/// leaf's arena, if any, and the leaf where per-leaf publication began, if it
/// did.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn prove_tree_pipelined(
    plan: &WhirBlockPlan,
    proof: &BlockWhirProof,
    leaves: Vec<Published<ProgramSlot>>,
    budget: Option<&TreeBudget>,
    siblings: usize,
    leaves_each: bool,
    early: Vec<EarlyLeaf>,
    early_first: bool,
    node_pipe: &rayon::ThreadPool,
    top_streamed: &std::sync::OnceLock<bool>,
    top_handoff: Option<&TopHandoff>,
    base_end: f64,
    in_tree_ahead: usize,
    say: &(dyn Fn(&str) + Sync),
) -> Result<
    (
        Vec<LevelTiming>,
        Vec<(super::registry::LfmArtifacts, LfmProof)>,
        Option<(usize, Vec<Vec<LfmWord>>, f64, f64, f64)>,
        Option<usize>,
    ),
    String,
> {
    // With `early_first` (`W3_LEAF_FIRST`), the early leaves are taken first,
    // in the order they completed.
    let first: Vec<usize> = if early_first {
        early.iter().map(|e| e.leaf).collect()
    } else {
        Vec::new()
    };
    let early_n = early.len();
    let early = std::sync::Mutex::new(early);
    // Every early leaf's stamps, for the `W3 EARLY LEAVES` line.
    let early_stamps: std::sync::Mutex<Vec<(usize, f64, f64, f64)>> =
        std::sync::Mutex::new(Vec::new());
    let early_out: std::sync::Mutex<Option<(usize, Vec<Vec<LfmWord>>, f64, f64, f64)>> =
        std::sync::Mutex::new(None);
    let wrap = aggregation_wrap_options();
    let words = plan.child_layout().total();
    let t_tree = std::time::Instant::now();
    let shape = plan.levels();
    let mut timings = Vec::with_capacity(shape.len() + 1);
    let mut proofs = Vec::new();
    // Each leaf's artifacts ([`publish_each`]).
    let built: Vec<Published<LeafBuilt>> = leaves.iter().map(|_| Published::new()).collect();
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

    // The tree's own leaves go ahead only when there are several early ones;
    // with one, the order is the single early leaf's.
    let in_tree_ahead = if early_n > 1 { in_tree_ahead } else { 0 };
    let order = early_leaves_order(tree_order(leaves.len(), &shape), &first, in_tree_ahead);
    let results: Vec<Vec<Published<ProvedProgram>>> = std::iter::once(leaves.len())
        .chain(shape.iter().map(|l| l.arities.len()))
        .map(|n| (0..n).map(|_| Published::new()).collect())
        .collect();

    // The programs a level still has to prove, for the purge after its last.
    let remaining: Vec<std::sync::atomic::AtomicUsize> = results
        .iter()
        .map(|level| std::sync::atomic::AtomicUsize::new(level.len()))
        .collect();

    let (n_leaves, each_from) = std::thread::scope(|outer| -> Result<_, String> {
        // 0. the leaves the budget left to the tree, emitted in leaf order as
        // it admits them, on plain threads (an admission may wait).
        let deferred = leaves.iter().position(|slot| slot.0.get().is_none());
        if let Some(from) = deferred {
            let (leaves, emit_leaf) = (&leaves, move |k: usize| -> Result<ProgramSlot, String> {
                let b = budget.ok_or("a leaf left to the tree without a program budget")?;
                let mut permit = b.budget.acquire(k, b.leaf_estimate(k))?;
                let program = plan.leaf_program(k)?;
                permit.emitted(program_bytes(&program));
                Ok(ProgramSlot::new(program, Some(permit)))
            });
            outer.spawn(move || {
                let _fail = FailEmpty(&leaves[from..], "the leaf emitter stopped");
                super::program_budget::emit_ordered(
                    from..leaves.len(),
                    DEFERRED_LEAF_EMITTERS,
                    &emit_leaf,
                    &mut |k, slot| {
                        let emitted = slot.is_ok();
                        let _ = leaves[k].0.set(slot);
                        emitted
                    },
                );
            });
        }
        // 2. the nodes' programs and artifacts, into their slots.
        let builder = outer.spawn(|| {
            build_nodes(
                plan, &shape, &built, &programs, &slots, &wrap, words, t_tree, node_pipe, budget,
            )
        });
        // The top's artifacts to the compression step as soon as they are
        // built. The builder fails every slot it leaves empty, so this waiter
        // always returns.
        if let Some(handoff) = top_handoff {
            let top = slots.last().and_then(|level| level.first());
            outer.spawn(move || {
                let got = match top {
                    Some(slot) => slot.wait().map(|node| node.artifacts.clone()),
                    None => Err("the tree has no top slot".to_string()),
                };
                let _ = handoff.artifacts.0.set(got);
            });
        }
        let each_from = std::thread::scope(|scope| {
            // 1. the leaves' artifacts, one after another on this thread: each
            // build holds the card throughout (armed), so building them as
            // rayon jobs bought no overlap, and a holder on a rayon worker can
            // run a sibling build that takes the card again (BIG 569).
            let artifacts = scope.spawn(|| {
                publish_each(
                    &leaves,
                    &built,
                    leaves_each,
                    |k| leaves[k].0.get().is_some(),
                    |k, slot| {
                        let program = slot.wait()?.get()?;
                        let t = std::time::Instant::now();
                        let artifacts = artifacts_of(&program.program, &wrap);
                        drop(program);
                        let derived = super::whir_block::derived_of(&artifacts, &wrap, words)?;
                        *leaf_built_at[k]
                            .lock()
                            .map_err(|_| "a built-at stamp is poisoned")? =
                            t_tree.elapsed().as_secs_f64();
                        Ok((artifacts, derived, t.elapsed().as_secs_f64()))
                    },
                )
            });
            // 3. every program of the tree, `siblings` at a time, leaves first;
            // each node as soon as its children are proved and its program is
            // in its slot ([`prove_dataflow`]).
            let prove_one = |at: TreeAt,
                             kids: &[&Published<ProvedProgram>]|
             -> Result<ProvedProgram, String> {
                let start = t_tree.elapsed().as_secs_f64();
                if at.lv == 0 {
                    let k = at.j;
                    // The leaf's program, once emitted: this prover takes it.
                    let program = leaves[k].wait()?;
                    program.claim();
                    let ahead = {
                        let mut slot = early.lock().map_err(|_| "the early slot is poisoned")?;
                        slot.iter()
                            .position(|e| e.leaf == k)
                            .map(|at| slot.remove(at))
                    };
                    let t = std::time::Instant::now();
                    let filled = match ahead {
                        Some(e) => {
                            let (filled, arena, started, secs) = e
                                .handle
                                .join()
                                .map_err(|_| format!("leaf {k}: the early executor panicked"))??;
                            early_stamps
                                .lock()
                                .map_err(|_| "the early stamps are poisoned")?
                                .push((k, e.complete_at, started, secs));
                            // The harness checks the first early leaf's arena.
                            let mut out = early_out
                                .lock()
                                .map_err(|_| "the early readout is poisoned")?;
                            if out.is_none() {
                                *out = Some((k, arena, e.complete_at, started, secs));
                            }
                            filled
                        }
                        None => {
                            let arenas = block_leaf_arena(plan, proof, k)?;
                            let program = program.get()?;
                            lfm_execute_and_fill(
                                &program.program,
                                &arenas,
                                program.program.hasher(<BlockFamily as ProofFamily>::SOCKET),
                            )
                            .map_err(|e| format!("leaf {k}: {e:?}"))?
                        }
                    };
                    let artifacts = &built[k].wait()?.0;
                    let lfm =
                        filled.prove_under::<BlockFamily>(artifacts, &wrap, decide_lfm_residency());
                    // Proved, so its artifacts are built: nothing reads
                    // the program again.
                    program.release();
                    let lfm = lfm.map_err(|e| format!("leaf {k}: {e:?}"))?;
                    let prove = t.elapsed().as_secs_f64();
                    let times = ProgramTimes {
                        program_at: 0.0,
                        program_emit: 0.0,
                        built_at: 0.0,
                        start,
                        end: t_tree.elapsed().as_secs_f64(),
                        split: prove_split_now(),
                    };
                    let child =
                        harvest_child_under::<BlockFamily>(artifacts.clone(), wrap.clone(), &lfm)?;
                    return Ok(ProvedProgram {
                        lfm,
                        child,
                        prove,
                        built: 0.0,
                        times,
                    });
                }
                // Execute and fill from the program as soon as it is
                // emitted, prove once the node's own artifacts are built
                // ([`node_flow`]); traces filled under the socket its hash
                // rows run on, which the node's artifacts are built for (the
                // prove asserts it).
                node_flow(
                    &programs,
                    &slots,
                    at,
                    true,
                    |emitted: &NodeProgram| -> Result<_, String> {
                        let t = std::time::Instant::now();
                        // This prover takes the node's program; once filled,
                        // nothing but its artifacts' build (which holds its
                        // own reference) reads it again.
                        emitted.program.claim();
                        let program = emitted.program.get()?;
                        let hasher = program.program.hasher(<BlockFamily as ProofFamily>::SOCKET);
                        // The top streams only if a child is still unproved
                        // now that it can execute; the choice is kept for
                        // the readout.
                        let streamed = at.lv == shape.len() && {
                            let streams = top_streams(kids);
                            let _ = top_streamed.set(streams);
                            streams
                        };
                        let filled = if streamed {
                            execute_streamed(&program.program, &emitted.child_arenas, kids, hasher)
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
                        drop(program);
                        emitted.program.release();
                        Ok((filled, t, (emitted.at, emitted.emit)))
                    },
                    |(filled, t, (program_at, program_emit)), node: &TreeNode| {
                        let lfm = filled
                            .prove_under::<BlockFamily>(
                                &node.artifacts,
                                &wrap,
                                decide_lfm_residency(),
                            )
                            .map_err(|e| format!("level {} node {}: {e:?}", at.lv, at.j))?;
                        let prove = t.elapsed().as_secs_f64();
                        let times = ProgramTimes {
                            program_at,
                            program_emit,
                            built_at: node.built_at,
                            start,
                            end: t_tree.elapsed().as_secs_f64(),
                            split: prove_split_now(),
                        };
                        let child = harvest_child_under::<BlockFamily>(
                            node.artifacts.clone(),
                            wrap.clone(),
                            &lfm,
                        )?;
                        Ok(ProvedProgram {
                            lfm,
                            child,
                            prove,
                            built: node.built,
                            times,
                        })
                    },
                )
            };
            let target = block_whir::memory_target_bytes();
            prove_dataflow(&shape, &order, &results, siblings, false, |at, kids| {
                let proved = prove_one(at, kids);
                // The level's last program: its freed pages back to the OS
                // before the levels above go on, where the host is short.
                let left = remaining[at.lv].fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                if left == 1 {
                    alloc_peak_line(say, &format!("level-{}", at.lv));
                }
                if left == 1 && at.lv + 1 < remaining.len() {
                    crate::alloc_purge::purge_after_level(at.lv, target, say);
                }
                proved
            });
            // Every program is proved or failed: an emitter still waiting for
            // an admission (only after a failure) returns an error instead.
            if let Some(b) = budget {
                b.budget.fail();
            }
            artifacts
                .join()
                .map_err(|_| "the leaves' artifacts builder panicked".to_string())
        })?;
        for slot in &built {
            slot.wait()?;
        }
        builder
            .join()
            .map_err(|_| "the node builder panicked".to_string())?;
        Ok((built.len(), each_from))
    })?;
    // Each level's readout: a level's wall is from the level below's last
    // proof to its own (level 0: from the tree's start), now that a node may
    // prove before its level below is done.
    let leaf_secs: Vec<f64> = built
        .iter()
        .map(|slot| slot.wait().map(|(_, _, s)| *s))
        .collect::<Result<_, String>>()?;
    let mut level_proofs: Vec<Vec<LfmProof>> = Vec::with_capacity(results.len());
    let mut below_end = 0.0f64;
    let n_levels = results.len();
    for (lv, level) in results.into_iter().enumerate() {
        let proved: Vec<ProvedProgram> = level
            .into_iter()
            .map(Published::take)
            .collect::<Result<_, String>>()?;
        // The top's words as a parent's arena reads them, for C1.
        if let (Some(handoff), true, [top]) = (top_handoff, lv + 1 == n_levels, proved.as_slice()) {
            let arena = try_child_arena_words(&top.child)?;
            *handoff
                .arena
                .lock()
                .map_err(|_| "the top's arena slot is poisoned")? = Some(arena);
        }
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
    for (slot, lfm) in built.into_iter().zip(leaf_lfms) {
        proofs.push((slot.take()?.0, lfm));
    }
    for (level, lfms) in slots.into_iter().zip(level_proofs) {
        for (slot, lfm) in level.into_iter().zip(lfms) {
            proofs.push((slot.take()?.artifacts, lfm));
        }
    }
    if early_n > 1 {
        let mut stamps = early_stamps
            .into_inner()
            .map_err(|_| "the early stamps are poisoned")?;
        stamps.sort_by(|a, b| a.1.total_cmp(&b.1));
        say(&format!(
            "W3 EARLY LEAVES: {} · {}",
            stamps.len(),
            stamps
                .iter()
                .map(|(k, at, started, secs)| format!(
                    "leaf {k} complete at {at:.2}s, execute + fill from {started:.2}s for {secs:.2}s, \
                     margin {:+.2}s to the base's end",
                    base_end - (started + secs)
                ))
                .collect::<Vec<_>>()
                .join(" · ")
        ));
    }
    let early_out = early_out
        .into_inner()
        .map_err(|_| "the early readout is poisoned")?;
    Ok((timings, proofs, early_out, each_from))
}

/// What the tree hands the compression step (C1) when the driver asks for it:
/// the top's artifacts as soon as its builder publishes them (before the top
/// is proved, so C1's program and build go on beside the top's prove), and
/// the top's arena words once the top is proved.
pub(crate) struct TopHandoff {
    pub(crate) artifacts: Published<super::registry::LfmArtifacts>,
    pub(crate) arena: std::sync::Mutex<Option<Vec<Vec<LfmWord>>>>,
}

impl TopHandoff {
    pub(crate) fn new() -> Self {
        Self {
            artifacts: Published::new(),
            arena: std::sync::Mutex::new(None),
        }
    }
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

/// The compression step's proof (C1), the rate it was proved at and how it
/// binds its instruction columns.
pub struct CompressedTop {
    pub rate: super::wlfm::WlfmRate,
    pub levers: super::wlfm::WlfmLevers,
    /// The encoding the file ships in.
    pub wire: super::wire::WireFormat,
    pub proof: super::wlfm::WlfmProof,
    /// C2 over C1 (`LAMBDA_VM_BLOCK_COMPRESS2`): what ships then.
    pub c2: Option<Compressed2Top>,
}

/// C2's program and build on its own thread: the build, its seconds, and
/// when it was done (seconds since the tree started).
type C2BuildHandle =
    std::thread::JoinHandle<Result<(super::compress::CompressBuild, f64, f64), String>>;

/// The second compression step's proof (C2) and the query grind it was
/// proved at; its rate and levers are [`super::compress::COMPRESS2_RATE`] and
/// [`super::compress::compress2_levers`].
pub struct Compressed2Top {
    pub grind: u8,
    pub proof: super::wlfm::WlfmProof,
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
    /// C1 over the top, under `LAMBDA_VM_BLOCK_COMPRESS`: what ships then.
    pub compressed: Option<CompressedTop>,
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
    Vec<Published<ProgramSlot>>,
    usize,
    Option<TreeBudget>,
    f64,
    Option<f64>,
    f64,
    (Vec<EarlyLeaf>, Vec<String>),
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
    let (leaves, siblings, fan_in, early_on, leaf_emitters) = (
        cfg.leaves,
        cfg.siblings,
        cfg.fan_in,
        cfg.early_on,
        cfg.leaf_emitters,
    );
    let (argue, format, options) = (cfg.argue, cfg.format, &cfg.options);
    // The recursion's base is its family's, and a base-only process runs no
    // recursion (`crate::base_only`): both refused before anything is proved.
    crate::base_only::refuse_recursion()?;
    block_whir::checked_base(&format.zf.base).map_err(|e| format!("{e:?}"))?;
    sink.line(&posture_line());
    sink.line(&base_line(&format.zf.base));
    sink.line(&format!("W3 ARGUE: {argue:?}"));
    sink.line(&format!("W3 GROUP OPENING: {:?}", format.group));
    sink.line(&format!("BLOCK UPLOAD AHEAD: {}", options.upload_ahead));
    sink.line(&format!(
        "BLOCK REST LAYOUT CONFIG: {} · KECCAK_RND built as its tables {} · finish packed {} · building at most {}",
        block_whir::rest_layout_label(options, true),
        options.finish_keccak_rnd_chunks,
        options.pack_finished,
        options
            .rest_build_bytes
            .map_or("the gate".to_string(), |b| format!("{} MiB", b >> 20)),
    ));
    let opts = super::proof::block_base_options();
    // Armed for the siblings while the tree proves, and put back as it returns.
    let _armed = super::device_permit::arm_scoped(siblings);
    sink.line(&format!(
        "W3 CONFIG: leaves {leaves:?} · siblings {siblings} · fan-in {fan_in} · layout workers {} (ahead {:?}) · streamed ops dropped {}",
        options.layout_workers,
        options.layout_ahead,
        options.drop_streamed_ops
    ));

    // `LAMBDA_VM_ALLOC_WATERMARK`: freed pages back to the OS inside a phase,
    // only where the host is short (`alloc_purge`).
    let watermark = crate::alloc_purge::start_watermark(block_whir::memory_target_bytes());
    // `LAMBDA_VM_BACKEND_BEFORE_PROVE=1`: the card set up before the prove's
    // clock, as a long-lived prover has it, instead of beside the prove's
    // start. `LAMBDA_VM_PINNED_POOL_GIB`: the page-locked pool locked here,
    // after the card and before the clock, once (every driver call waits
    // while it runs).
    if std::env::var("LAMBDA_VM_BACKEND_BEFORE_PROVE").is_ok_and(|v| v.trim() == "1") {
        let t = std::time::Instant::now();
        multilinear::gpu::set_up_device();
        sink.line(&format!(
            "W3 BACKEND BEFORE PROVE: up in {:.3}s",
            t.elapsed().as_secs_f64()
        ));
    }
    if let Some(pool) = multilinear::pinned::lock_pool() {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        sink.line(&format!(
            "W3 PINNED POOL: {:.2} GiB, {:.2} locked in {:.3}s ({:.3} s/GiB) · {} pieces, {} refused",
            gib(pool.bytes),
            gib(pool.locked),
            pool.seconds,
            pool.seconds / gib(pool.bytes).max(1e-9),
            pool.pieces,
            pool.refused,
        ));
    }
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
                commit_ood: opened.commit_ood.to_vec(),
                tables: opened.tables.to_vec(),
                argue: opened.argue.cloned(),
                opening: opened.opening.clone(),
                prepared: opened.prepared.cloned(),
                at: t0.elapsed().as_secs_f64(),
            });
        }
    };
    // The tree programs' budget (`LAMBDA_VM_TREE_PROGRAM_BUDGET`): against the
    // memory target, which `LAMBDA_VM_BLOCK_MEMORY_TARGET_GIB` moves too, `auto`'s
    // room line at `LAMBDA_VM_TREE_PROGRAM_BUDGET_SHARE` of it, the host read
    // less what a purge could return; `off` emits every leaf beside phase B,
    // as before.
    let budget = match cfg.program_budget {
        super::program_budget::BudgetSetting::Off => None,
        setting => Some(super::program_budget::ProgramBudget::new(
            super::program_budget::Room::of(
                setting,
                block_whir::memory_target_bytes(),
                super::program_budget::share_from_env()?,
            ),
            super::program_budget::READING_DEFAULT,
        )),
    };
    // The leaves offered to phase B (`W3_LEAF_FIRST`), and when.
    let lead = block_whir::PhaseBLead::leading(cfg.early_leaves);
    let early_leaves = cfg.early_leaves;
    // The later early leaves' pool, only when several can go early.
    let early_threads = if cfg.early_leaves > 1 {
        cfg.early_pool
    } else {
        0
    };
    let early_pool = early_pool(early_threads)?;
    let offered_at = std::sync::Mutex::new(None::<f64>);
    let leaf_first = cfg.leaf_first && early_on;
    let (lead_ref, offered_ref) = (&lead, &offered_at);
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
            // Under a program budget, the leaves admitted now, in leaf order
            // while it has room (`try_acquire` never waits: the next AHEAD
            // always come in); the tree emits the rest as it admits them.
            let n = plan.partition().num_leaves();
            let tree_budget = budget.map(|b| {
                let groups: Vec<usize> = (0..n).map(|k| plan.partition().leaf(k).len()).collect();
                TreeBudget::new(b, &groups, &plan.levels())
            });
            let permits: Vec<Option<super::program_budget::Permit>> = match &tree_budget {
                Some(b) => (0..n)
                    .map_while(|k| b.budget.try_acquire(k, b.leaf_estimate(k)))
                    .map(Some)
                    .collect(),
                None => (0..n).map(|_| None).collect(),
            };
            let upto = permits.len();
            // The admitted leaves' groups to phase B, before their programs are
            // emitted: it takes the soonest leaf's groups next, and that leaf
            // is the early one below.
            if leaf_first {
                let leaves = (0..upto)
                    .map(|k| plan.partition().leaf(k).to_vec())
                    .collect();
                if lead_ref.offer(leaves) {
                    *offered_ref.lock().unwrap_or_else(|e| e.into_inner()) =
                        Some(t0.elapsed().as_secs_f64());
                }
            }
            let programs: Vec<Published<ProgramSlot>> = (0..n).map(|_| Published::new()).collect();
            // The admitted leaves' programs, on plain threads off the rayon
            // pool phase B is using; the stamps are leaf 0's and the last's.
            let emit = |k: usize| plan.leaf_program(k);
            let emit_leaves = |permits: Vec<Option<super::program_budget::Permit>>,
                               threads: usize|
             -> Result<(Option<f64>, f64), String> {
                let mut first = None;
                emit_admitted_leaves(
                    &programs,
                    permits,
                    threads,
                    &emit,
                    &program_bytes,
                    &mut |k| {
                        if k == 0 {
                            first = Some(t0.elapsed().as_secs_f64());
                        }
                    },
                )?;
                Ok((first, t0.elapsed().as_secs_f64()))
            };
            // The control (`W3_LEAF_EMITTERS=0`): one thread a leaf, all at
            // once, and the first group taken after the last is emitted.
            // Otherwise `leaf_emitters` at a time beside the groups below.
            let (control, beside) = match leaf_emitters {
                None => (Some(emit_leaves(permits, upto)?), None),
                Some(threads) => (None, Some((permits, threads))),
            };
            let (looped, emitted) = std::thread::scope(|inner| {
                let emitter = beside
                    .map(|(permits, threads)| inner.spawn(move || emit_leaves(permits, threads)));
                let looped = (|| -> Result<(Vec<EarlyLeaf>, Vec<String>), String> {
                    // The groups as phase B finishes them (none unless `early_on`):
                    // each leaf complete while a group is still to come, up to
                    // `W3_EARLY_LEAVES` of them, runs its execute + fill now, on a
                    // thread of its own; the first also warms the first prove.
                    let mut early: Vec<EarlyLeaf> = Vec::new();
                    // A later leaf goes early only if it fits before the base ends
                    // ([`early_leaf_fits`]): the paces it is predicted from, and
                    // the leaves it kept for the tree.
                    let costs: Vec<u64> = plan.costs().iter().map(|&c| c as u64).collect();
                    let leaf_cost = |k: usize| -> u64 {
                        plan.partition().leaf(k).iter().map(|&g| costs.get(g).copied().unwrap_or(0)).sum()
                    };
                    let mut cost_to_come: u64 = costs.iter().sum();
                    let mut first_group_at: Option<f64> = None;
                    let mut cost_since_first_group: u64 = 0;
                    let first_fill: std::sync::Arc<std::sync::Mutex<Option<f64>>> =
                        std::sync::Arc::new(std::sync::Mutex::new(None));
                    // A later early leaf measured on the early pool: (secs, cost).
                    let pool_fill: std::sync::Arc<std::sync::Mutex<Option<(f64, u64)>>> =
                        std::sync::Arc::new(std::sync::Mutex::new(None));
                    let mut first_cost: u64 = 0;
                    let mut kept: Vec<usize> = Vec::new();
                    let mut fit_notes: Vec<String> = Vec::new();
                    let mut words: Vec<Option<Vec<LfmWord>>> =
                        (0..plan.num_groups()).map(|_| None).collect();
                    let mut block_roots = Vec::new();
                    let mut block_answers = Vec::new();
                    let mut done = 0usize;
                    while let Ok(msg) = grx.recv() {
                        done += 1;
                        let g = msg.group;
                        let c = costs.get(g).copied().unwrap_or(0);
                        cost_to_come = cost_to_come.saturating_sub(c);
                        match first_group_at {
                            None => first_group_at = Some(msg.at),
                            Some(_) => cost_since_first_group += c,
                        }
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
                            block_answers = msg.commit_ood;
                        }
                        // More than one early leaf only when phase B led several
                        // leaves: with fewer (no valid order) the other leaves
                        // complete with phase B's last groups, so the single
                        // early leaf's path is kept exactly.
                        let room = early_room(
                            early_leaves,
                            lead_ref.taken().map_or(0, |t| t.leaves.len()),
                        );
                        if early.len() >= room || done >= plan.num_groups() {
                            continue;
                        }
                        let partition = plan.partition();
                        let complete = (0..partition.num_leaves())
                            .filter(|&k| early.iter().all(|e| e.leaf != k) && !kept.contains(&k))
                            .find(|&k| partition.leaf(k).iter().all(|&g| words[g].is_some()))
                            .filter(|&k| k < upto);
                        // After the first: only a leaf whose execute + fill fits
                        // before the base ends; one that does not stays for the
                        // tree, and so does one whose fit cannot be told yet.
                        let complete = match complete {
                            Some(k) if !early.is_empty() => {
                                // The pace a later early leaf runs at: one measured
                                // on the early pool, else the first's, slowed by
                                // the pool, else unknown.
                                let on_pool = *pool_fill.lock().unwrap_or_else(|e| e.into_inner());
                                let first = first_fill
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .map(|s| {
                                        let slow = if early_pool.is_some() { EARLY_POOL_SLOWDOWN } else { 1.0 };
                                        (s * slow, first_cost)
                                    });
                                match early_leaf_fits(
                                    msg.at,
                                    leaf_cost(k),
                                    on_pool.or(first),
                                    first_group_at,
                                    cost_since_first_group,
                                    cost_to_come,
                                ) {
                                    Some(fit) if fit.fits => Some(k),
                                    Some(fit) => {
                                        kept.push(k);
                                        fit_notes.push(format!(
                                            "leaf {k} complete at {:.2}s kept for the tree: execute + fill predicted to end {:.2}s, the base {:.2}s",
                                            msg.at, fit.fill_end, fit.base_end
                                        ));
                                        None
                                    }
                                    None => None,
                                }
                            }
                            other => other,
                        };
                        if let Some(k) = complete {
                            let first = early.is_empty();
                            if first {
                                first_cost = leaf_cost(k);
                            }
                            let first_fill_slot = first_fill.clone();
                            let pool_fill_slot = pool_fill.clone();
                            let pool = early_pool.clone();
                            let cost = leaf_cost(k);
                            let groups: Vec<Vec<LfmWord>> = partition
                                .leaf(k)
                                .iter()
                                .map(|&g| words[g].clone().unwrap_or_default())
                                .collect();
                            let arena = leaf_arena(&block_roots, &block_answers, groups);
                            // This executor takes the leaf's program.
                            let slot = programs[k].wait()?;
                            slot.claim();
                            let program = slot.get()?;
                            let handle = std::thread::spawn(move || -> Result<EarlyOut, String> {
                                let started = t0.elapsed().as_secs_f64();
                                if !first {
                                    // A later early leaf: its execute + fill only;
                                    // the first one warmed what a prove builds once.
                                    // On the early pool, below phase B's priority.
                                    let t = std::time::Instant::now();
                                    let hasher = program
                                        .program
                                        .hasher(<BlockFamily as ProofFamily>::SOCKET);
                                    let run = || lfm_execute_and_fill(&program.program, &arena, hasher);
                                    let filled = match &pool {
                                        Some(pool) => pool.install(run),
                                        None => run(),
                                    }
                                    .map_err(|e| format!("early leaf {k}: {e:?}"))?;
                                    let secs = t.elapsed().as_secs_f64();
                                    if pool.is_some() {
                                        *pool_fill_slot.lock().unwrap_or_else(|e| e.into_inner()) =
                                            Some((secs, cost));
                                    }
                                    return Ok((filled, arena, started, secs));
                                }
                                // The staging pairs the tree's first card holds
                                // would allocate, made now, beside the base.
                                let staging =
                                    std::thread::spawn(super::first_prove_warm::prewarm_staging);
                                let t = std::time::Instant::now();
                                let hasher =
                                    program.program.hasher(<BlockFamily as ProofFamily>::SOCKET);
                                let mut filled =
                                    lfm_execute_and_fill(&program.program, &arena, hasher)
                                        .map_err(|e| format!("early leaf {k}: {e:?}"))?;
                                let secs = t.elapsed().as_secs_f64();
                                *first_fill_slot.lock().unwrap_or_else(|e| e.into_inner()) =
                                    Some(secs);
                                let wrap = super::proof::aggregation_wrap_options();
                                // Its tables' domains and twiddles, warm before
                                // the tree's first card hold ([`super::domain_warm`]).
                                super::domain_warm::warm(filled.traces(), &wrap);
                                super::domain_warm::note_early(t0.elapsed().as_secs_f64());
                                // And the rest of what its prove builds on first
                                // use: the per-domain prove constants and the
                                // DEEP's LDE cosets. A warm-up that fails leaves
                                // the prove to build what it needs; the line says
                                // so.
                                match super::first_prove_warm::warm(
                                    &program.program,
                                    filled.traces_mut(),
                                    &wrap,
                                    hasher,
                                ) {
                                    Ok(_) => super::first_prove_warm::note_early(
                                        t0.elapsed().as_secs_f64(),
                                    ),
                                    Err(e) => eprintln!(
                                        "W3 FIRST PROVE WARM: early leaf {k}'s warm-up refused: {e}"
                                    ),
                                }
                                if staging.join().is_err() {
                                    eprintln!("W3 FIRST PROVE WARM: the staging warm-up panicked");
                                }
                                Ok((filled, arena, started, secs))
                            });
                            early.push(EarlyLeaf {
                                leaf: k,
                                complete_at: msg.at,
                                handle,
                            });
                        }
                    }
                    Ok((early, fit_notes))
                })();
                // Every admitted leaf published before the tree may look: an
                // empty slot is, to it, a leaf the budget left to the tree.
                let emitted = emitter.map(|h| {
                    h.join()
                        .map_err(|_| "a leaf emitter panicked".to_string())
                        .and_then(|stamps| stamps)
                });
                (looped, emitted)
            });
            let early = looped?;
            let (first_ready, ready) = match control {
                Some(stamps) => stamps,
                None => emitted.ok_or("the leaf emitters never ran")??,
            };
            Ok((
                plan,
                programs,
                upto,
                tree_budget,
                at,
                first_ready,
                ready,
                early,
            ))
        });
        let proved = block_whir::prove_block_whir_observed_led(
            elf,
            input,
            &opts,
            &format,
            options,
            &observe,
            &on_group,
            leaf_first.then_some(&lead),
        );
        // A prover that failed before stating must not leave the planner waiting.
        *sender.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *group_sender.lock().unwrap_or_else(|e| e.into_inner()) = None;
        (proved, pre.join())
    });
    let (proof, stamps) = proved.map_err(|e| format!("the block proves: {e}"))?;
    let base = t0.elapsed().as_secs_f64();
    alloc_peak_line(&|l: &str| sink.line(l), "base");
    // The base's freed pages back to the OS before the tree allocates: when
    // the block found its memory short, or as `LAMBDA_VM_ALLOC_PURGE` names it
    // (`alloc_purge`).
    crate::alloc_purge::purge_point("base");
    let (
        plan,
        programs,
        upto,
        tree_budget,
        stated_at,
        first_ready_at,
        ready_at,
        (early, fit_notes),
    ) = pre
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
            "W3 CHUNKED: {} (cuts KECCAK 2^{} · ECSM 2^{} · COPY 2^{})",
            block_whir::chunked_census(&proof.table_counts, &proof.table_num_vars),
            options.keccak_rows_log2,
            options.ecsm_rows_log2,
            options.copy_rows_log2,
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
        // The groups' round-0 trees (one a polynomial, or one a joint run)
        // and the round-0 leaf permutations the card's commits paid.
        let polys: usize = frame
            .stack_layouts
            .iter()
            .map(multilinear::stacking::StackedLayout::num_polys)
            .sum();
        sink.line(&format!(
            "W3 WHIR TREES: {} · round-0 trees {} · polys {polys} · card leaf permutations {}",
            match format.zf.whir_trees {
                multilinear::whir_chain::Round0Trees::Each => "each",
                multilinear::whir_chain::Round0Trees::One => "one",
            },
            proof.proof.roots.len(),
            multilinear::gpu::commit_leaf_perms(),
        ));
    }
    sink.line(&format!(
        "W3 BASE: {base:.2}s · statement at {stated_at:.2}s · plan + {} leaves emitted by {ready_at:.2}s ({})",
        if upto == programs.len() {
            format!("{upto}")
        } else {
            format!("{upto} of {}", programs.len())
        },
        if ready_at <= base {
            "inside the base"
        } else {
            "after the base"
        }
    ));
    // The card's upload traffic at the base's end, read against the same line at
    // the tree's end: what the tree's proves uploaded.
    #[cfg(feature = "cuda")]
    sink.line(&format!(
        "W3 STAGING AFTER BASE: {} · main traces built on the card {}",
        stark::gpu_lde::staging_report(),
        stark::trace::main_dev_producer_calls()
    ));
    sink.line(&format!(
        "W3 LEAF EMITTERS: {} · leaf 0's program at {} · the last at {ready_at:.2}s",
        match leaf_emitters {
            Some(threads) => format!(
                "{threads} at a time in leaf order, beside phase B's groups (W3_LEAF_EMITTERS)"
            ),
            None => "one thread a leaf, all at once, before the first group is taken \
                     (W3_LEAF_EMITTERS=0, the control)"
                .to_string(),
        },
        first_ready_at.map_or("-".to_string(), |t| format!("{t:.2}s")),
    ));
    sink.line(&leaf_first_line(
        leaf_first,
        *offered_at.lock().unwrap_or_else(|e| e.into_inner()),
        stated_at,
        lead.taken(),
        &early.iter().map(|e| e.leaf).collect::<Vec<_>>(),
    ));
    for note in &fit_notes {
        sink.line(&format!("W3 EARLY LEAF KEPT: {note}"));
    }
    if early_threads > 0 {
        sink.line(&format!(
            "W3 EARLY POOL: the early leaves after the first on {early_threads} threads of their own at nice \
             {EARLY_POOL_NICE} (Linux; elsewhere at the prover's priority)"
        ));
    }
    // Several leaves asked for: which led, and the rebuilt groups that decide it
    // (a leaf leads only when its rebuilt groups are next in their class).
    if cfg.early_leaves > 1
        && let Some(t) = lead.taken()
    {
        let partition = plan.partition();
        let owner = |g: usize| {
            (0..partition.num_leaves())
                .find(|&k| partition.leaf(k).contains(&g))
                .map_or("-".to_string(), |k| k.to_string())
        };
        let rebuilt = if t.rebuilt.is_empty() {
            "none".to_string()
        } else {
            t.rebuilt
                .iter()
                .map(|&(g, c)| format!("{g} (class {c}, leaf {})", owner(g)))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let n = partition.num_leaves();
        sink.line(&format!(
            "W3 EARLY ORDER: phase B led {} of {n} leaves [{}] · rebuilt groups to come: {rebuilt} · classes fitting their window [{}] · {}",
            t.leaves.len(),
            t.leaves
                .iter()
                .zip(&t.rules)
                .map(|(k, r)| format!("{k} ({})", r.word()))
                .collect::<Vec<_>>()
                .join(", "),
            t.class_fits
                .iter()
                .enumerate()
                .map(|(c, f)| format!("class {c} {}", if *f { "fits" } else { "does not" }))
                .collect::<Vec<_>>()
                .join(", "),
            if t.leaves.len() + 1 >= n {
                "every leaf but the last led".to_string()
            } else {
                format!(
                    "NO VALID ORDER for the other {} leaves: their rebuilt groups are not next in their class",
                    n - t.leaves.len()
                )
            }
        ));
    }
    sink.line(&format!(
        "W3 TREE PROGRAM BUDGET: {}",
        match &tree_budget {
            Some(b) => format!(
                "{} · {upto} of {} leaf programs beside the base, the rest emitted in the tree as \
                 the budget admits them",
                b.budget.describe(),
                programs.len()
            ),
            None => "off (every leaf program beside the base)".to_string(),
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
    sink.line(&leaf_scale_line(format.leaf_scale, &plan));
    sink.line(&format!("W3 LEAF QUERY: {:?}", format.leaf_query));
    // Each leaf's chips, real / padded rows: how far each sits from its next
    // doubling, and which one doubled when the in-guest work moves.
    // (The leaves emitted beside the base: the others do not exist yet.)
    for (k, slot) in programs.iter().take(upto).enumerate() {
        let program = slot.wait()?.get()?;
        let chips: Vec<String> = super::airs::lfm_chip_census_with_hasher(
            &program.program,
            program.program.hasher(<BlockFamily as ProofFamily>::SOCKET),
        )
        .iter()
        .filter(|c| c.real_rows > 0)
        .map(|c| format!("{}={}/{}", c.name, c.real_rows, c.rows))
        .collect();
        sink.line(&format!("W3 LEAF CENSUS {k}: {}", chips.join(" ")));
    }

    // The readouts above run on the clock, between the base and the tree: a
    // prover prints none of them, so the whole block is also given without.
    let readouts = t0.elapsed().as_secs_f64() - base;
    let node_pipe = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.emit_threads)
        .thread_name(|i| format!("w3-emit-{i}"))
        .build()
        .map_err(|e| format!("the node builder's pool: {e}"))?;
    let top_streamed = std::sync::OnceLock::new();
    let t = std::time::Instant::now();
    let tree_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    // `LAMBDA_VM_BLOCK_COMPRESS`: C1's program and W-LFM build on a thread of
    // their own from the moment the top's artifacts are built, beside the
    // top's prove; C1 itself proves after the top.
    let handoff = cfg.compress.map(|_| TopHandoff::new());
    let c1_built: Published<(super::compress::CompressBuild, f64, f64)> = Published::new();
    // `LAMBDA_VM_BLOCK_COMPRESS2`: C2's program and build on a thread of their
    // own, started the moment C1's build is done (C2's program holds C1's
    // artifacts) and outliving the tree's scope, so it runs beside the top's
    // prove and C1's; C2 proves after C1.
    let c2_build: std::sync::Mutex<Option<C2BuildHandle>> = std::sync::Mutex::new(None);
    let (timings, proofs, early_out, each_from) = std::thread::scope(|scope| {
        if let (Some(rate), Some(handoff)) = (cfg.compress, handoff.as_ref()) {
            let (plan, c1_built, c2_build) = (&plan, &c1_built, &c2_build);
            scope.spawn(move || {
                let _fail = FailEmpty(std::slice::from_ref(c1_built), "the C1 builder stopped");
                let built = (|| -> Result<_, String> {
                    let top = handoff.artifacts.wait()?;
                    let tb = std::time::Instant::now();
                    let wrap = aggregation_wrap_options();
                    let child = super::compress::top_child(plan, top, &wrap)?;
                    let c = super::compress::build_compress(
                        &child,
                        plan.top_layout(),
                        rate,
                        cfg.compress_levers,
                    )?;
                    Ok((c, tb.elapsed().as_secs_f64(), t.elapsed().as_secs_f64()))
                })();
                let c1_artifacts = built
                    .as_ref()
                    .ok()
                    .map(|(c, _, _)| c.build.artifacts.clone());
                let _ = c1_built.0.set(built);
                if let (Some(grind), Some(c1)) = (cfg.compress2, c1_artifacts) {
                    let layout = plan.top_layout();
                    let handle = std::thread::spawn(move || {
                        let tb = std::time::Instant::now();
                        super::compress::build_compress2(&c1, layout, grind)
                            .map(|c| (c, tb.elapsed().as_secs_f64(), t.elapsed().as_secs_f64()))
                    });
                    if let Ok(mut slot) = c2_build.lock() {
                        *slot = Some(handle);
                    }
                }
            });
        }
        prove_tree_pipelined(
            &plan,
            &proof,
            programs,
            tree_budget.as_ref(),
            siblings,
            cfg.leaves_each,
            early,
            leaf_first,
            &node_pipe,
            &top_streamed,
            handoff.as_ref(),
            base,
            // Several early leaves: up to `siblings − 1` leaves the tree still has
            // to execute are taken before the early ones, so their execute starts
            // with the tree while the remaining worker keeps the card on the ready
            // early leaves ([`early_leaves_order`]). One early leaf: today's order.
            if cfg.early_leaves > 1 {
                siblings.saturating_sub(1)
            } else {
                0
            },
            &|line: &str| sink.line(line),
        )
    })
    .map_err(|e| format!("the tree proves: {e}"))?;
    if let Some(b) = &tree_budget {
        sink.line(&format!(
            "W3 TREE PROGRAM BUDGET end: {}",
            b.budget.summary()
        ));
    }
    let tree = t.elapsed().as_secs_f64();
    // C1 after the top, on the clock: its build (begun beside the tree) waited
    // for, then executed over the top's words and proved.
    let compressed = match (cfg.compress, handoff) {
        (Some(rate), Some(handoff)) => {
            let built = c1_built
                .take()
                .map_err(|e| format!("C1's program and build: {e}"))?;
            let (c, build_secs, built_at) = built;
            let arena = handoff
                .arena
                .into_inner()
                .map_err(|_| "the top's arena slot is poisoned")?
                .ok_or("the tree handed over no top arena")?;
            let tc = std::time::Instant::now();
            let c1 = super::compress::prove_compress(&c, &arena)
                .map_err(|e| format!("C1 proves: {e}"))?;
            let c1_secs = tc.elapsed().as_secs_f64();
            let split = super::proof::take_prove_split().map_or("no split".to_string(), |p| {
                format!(
                    "execute {:.2} fill {:.2} prove {:.2}",
                    p.execute, p.fill, p.multi_prove
                )
            });
            let a = &c.build.artifacts;
            sink.line(&format!(
                "W3 COMPRESS: C1 at blowup {} · Q {} · binding {:?} · final {:?} · fold {} · groups {:?} · prepared stacks {} · tables 2^{:?} · built in {build_secs:.2}s (done@{built_at:.2}, top proved@{tree:.2}, waited {:.2}s) · C1 {c1_secs:.2}s after the top ({split})",
                rate.factor(),
                a.config.num_queries,
                a.prep,
                a.final_poly,
                a.config.log_folding,
                a.sizes,
                a.stacks.len(),
                a.table_num_vars,
                (built_at - tree).max(0.0),
            ));
            // C2 after C1, on the clock: its build (begun on C1's builder
            // thread) waited for, then executed over C1's proof and proved.
            let c2 = match cfg.compress2 {
                Some(grind) => {
                    let c1_done = t.elapsed().as_secs_f64();
                    let (c2b, build2_secs, built2_at) = c2_build
                        .into_inner()
                        .map_err(|_| "the C2 build's slot is poisoned")?
                        .ok_or("C1's builder started no C2 build")?
                        .join()
                        .map_err(|_| "the C2 builder panicked")?
                        .map_err(|e| format!("C2's program and build: {e}"))?;
                    let tc = std::time::Instant::now();
                    let c2 = super::compress::prove_compress2(
                        &c2b,
                        &c.build.artifacts,
                        plan.top_layout(),
                        &c1,
                    )
                    .map_err(|e| format!("C2 proves: {e}"))?;
                    let c2_secs = tc.elapsed().as_secs_f64();
                    let split2 =
                        super::proof::take_prove_split().map_or("no split".to_string(), |p| {
                            format!(
                                "execute {:.2} fill {:.2} prove {:.2}",
                                p.execute, p.fill, p.multi_prove
                            )
                        });
                    let a2 = &c2b.build.artifacts;
                    sink.line(&format!(
                        "W3 COMPRESS2: C2 at blowup {} · Q {} · grind {} · binding {:?} · final {:?} · fold {} · groups {:?} · tables 2^{:?} · built in {build2_secs:.2}s (done@{built2_at:.2}, C1 proved@{c1_done:.2}, waited {:.2}s) · C2 {c2_secs:.2}s after C1 ({split2})",
                        a2.rate.factor(),
                        a2.config.num_queries,
                        grind,
                        a2.prep,
                        a2.final_poly,
                        a2.config.log_folding,
                        a2.sizes,
                        a2.table_num_vars,
                        (built2_at - c1_done).max(0.0),
                    ));
                    Some(Compressed2Top { grind, proof: c2 })
                }
                None => None,
            };
            Some(CompressedTop {
                rate,
                levers: cfg.compress_levers,
                wire: cfg.compress_wire,
                proof: c1,
                c2,
            })
        }
        _ => None,
    };
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
                        _ => format!(" program@{:.2} emit {:.2}", t.program_at, t.program_emit),
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
        "W3 TOP EXECUTE: {}",
        match top_streamed.get() {
            Some(true) => "streamed (a child was still unproved when the top could execute)",
            Some(false) => "whole (every child was proved before the top could execute)",
            None => "unknown (the top never chose)",
        }
    ));
    sink.line(&format!(
        "W3 NODE PIPE: {} builder threads; each level proves as its own nodes arrive",
        node_pipe.current_num_threads()
    ));
    // How the leaves' artifacts were published ([`publish_each`]), when the
    // first leaf was proved and the first node over the leaves had its
    // program, against when the last leaf's artifacts were built.
    let (first_leaf, last_leaf) = timings.first().map_or((f64::NAN, 0.0), |level| {
        let first = level
            .times
            .iter()
            .map(|t| t.end)
            .fold(f64::INFINITY, f64::min);
        let last = level.times.iter().map(|t| t.built_at).fold(0.0, f64::max);
        (first, last)
    });
    let first_program = timings.get(1).map_or(f64::NAN, |level| {
        level
            .times
            .iter()
            .map(|t| t.program_at)
            .fold(f64::INFINITY, f64::min)
    });
    sink.line(&format!(
        "W3 LEAF ARTIFACTS: {} · first leaf proved@{first_leaf:.2} · first level-1 program@{first_program:.2} · last leaf built@{last_leaf:.2}",
        match each_from {
            None => "together (every leaf's published after the last is built)".to_string(),
            Some(_) if cfg.leaves_each => "each published as it is built (W3_LEAF_ARTIFACTS_EACH=1)".to_string(),
            Some(k) => format!(
                "together until leaf {k}'s program was not there, then each published as it is built"
            ),
        }
    ));
    #[cfg(feature = "cuda")]
    sink.line(&format!(
        "W3 STAGING AT END: {} · main traces built on the card {}",
        stark::gpu_lde::staging_report(),
        stark::trace::main_dev_producer_calls()
    ));
    sink.line(&format!(
        "W3 RECURSION: {:.2}s after the base (tree {tree:.2}s) · whole block {whole:.2}s · whole excl. harness readouts {:.2}s (readouts {readouts:.2}s)",
        whole - base,
        whole - readouts
    ));
    if let Some(w) = watermark {
        sink.line(&w.finish());
    }
    if proofs.is_empty() {
        return Err("the tree has no top".to_string());
    }
    // The early leaf's arena is checked against the proof's by the harness.
    #[cfg(not(test))]
    drop(early_out);
    Ok(WhirTreeRun {
        statement: proof.statement().to_owned(),
        proofs,
        compressed,
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

/// The proof file's layout version. 2: the archived `TableCounts` gained
/// `is_wh`, so a version-1 file is a different layout. 3: the top proof
/// carries its commitment-time answers.
const PROOF_VERSION: u32 = 3;

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
    pub top_proof: super::proof::LfmStarkProof,
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

/// Which tree a proof file holds: the WHIR block tree compressed by C1
/// ([`super::compress`]). Another tag than [`PIPELINE_WHIR`], so neither
/// verifier reads the other's file as its own.
pub const PIPELINE_WHIR_COMPRESSED: u8 = 2;

/// Which tree a proof file holds: the WHIR block tree compressed by C1 with
/// C1's instruction columns evaluated by the verifier
/// (`LAMBDA_VM_BLOCK_COMPRESS_EVAL`). A file names the verifier it is for: each
/// compressed verifier refuses the other's tag before it reads anything else.
pub const PIPELINE_WHIR_COMPRESSED_EVALUATED: u8 = 3;

/// The compressed file's tag for C1 bound per `prep`.
pub fn compressed_pipeline(prep: super::wlfm::WlfmPrep) -> u8 {
    match prep {
        super::wlfm::WlfmPrep::Stacked => PIPELINE_WHIR_COMPRESSED,
        super::wlfm::WlfmPrep::Evaluated => PIPELINE_WHIR_COMPRESSED_EVALUATED,
    }
}

/// The compressed tree with C1's chains stopping at a final polynomial of
/// [`super::compress::COMPRESS_FINAL_VARS`] variables
/// (`LAMBDA_VM_BLOCK_COMPRESS_FINAL`), C1 stacked.
pub const PIPELINE_WHIR_COMPRESSED_FINAL: u8 = 4;

/// The same with C1's instruction columns evaluated.
pub const PIPELINE_WHIR_COMPRESSED_EVALUATED_FINAL: u8 = 5;

/// The compressed file's tag for C1 under `levers`; refused for a lever
/// setting no verifier takes.
pub fn compressed_pipeline_of(levers: super::wlfm::WlfmLevers) -> Result<u8, String> {
    use super::wlfm::WlfmPrep;
    use multilinear::whir_chain::FinalPoly;
    if levers == super::compress::folds_levers(levers.prep) {
        return Ok(match levers.prep {
            WlfmPrep::Stacked => PIPELINE_WHIR_COMPRESSED_FOLDS,
            WlfmPrep::Evaluated => PIPELINE_WHIR_COMPRESSED_FOLDS_EVALUATED,
        });
    }
    match (levers.prep, levers.final_poly, levers.fold) {
        (prep, FinalPoly::Off, None) => Ok(compressed_pipeline(prep)),
        (WlfmPrep::Stacked, FinalPoly::Vars(super::compress::COMPRESS_FINAL_VARS), None) => {
            Ok(PIPELINE_WHIR_COMPRESSED_FINAL)
        }
        (WlfmPrep::Evaluated, FinalPoly::Vars(super::compress::COMPRESS_FINAL_VARS), None) => {
            Ok(PIPELINE_WHIR_COMPRESSED_EVALUATED_FINAL)
        }
        (_, other, fold) => Err(format!(
            "no compressed verifier takes C1's chains at {other:?}, fold {fold:?}"
        )),
    }
}

/// The compressed tree with C1's chains on the byte-minimising schedule
/// (`LAMBDA_VM_BLOCK_COMPRESS_FOLDS`: fold 5 after round 0, final polynomial of
/// 10 variables), C1 stacked, rkyv.
pub const PIPELINE_WHIR_COMPRESSED_FOLDS: u8 = 10;
/// The same with C1's instruction columns evaluated.
pub const PIPELINE_WHIR_COMPRESSED_FOLDS_EVALUATED: u8 = 11;
/// The compact twin of [`PIPELINE_WHIR_COMPRESSED_FOLDS`].
pub const PIPELINE_WHIR_COMPRESSED_COMPACT_FOLDS: u8 = 12;
/// The compact twin of [`PIPELINE_WHIR_COMPRESSED_FOLDS_EVALUATED`].
pub const PIPELINE_WHIR_COMPRESSED_COMPACT_FOLDS_EVALUATED: u8 = 13;

/// Whether `tag` is an rkyv compressed file's.
fn is_rkyv_compressed(tag: u8) -> bool {
    (PIPELINE_WHIR_COMPRESSED..=PIPELINE_WHIR_COMPRESSED_EVALUATED_FINAL).contains(&tag)
        || (PIPELINE_WHIR_COMPRESSED_FOLDS..=PIPELINE_WHIR_COMPRESSED_FOLDS_EVALUATED)
            .contains(&tag)
        || tag == PIPELINE_WHIR_COMPRESSED2_G22
        || tag == PIPELINE_WHIR_COMPRESSED2_G26
}

/// The twice-compressed tree (`LAMBDA_VM_BLOCK_COMPRESS2`): the file carries
/// C2's proof over C1 ([`super::compress::build_compress2`]) at a 22-bit query
/// grind as rkyv (20), compact (21) or dedup (22); 23–25 are the same at 26
/// bits. Each is its own verifier's ([`Compressed2Format`]).
pub const PIPELINE_WHIR_COMPRESSED2_G22: u8 = 20;
pub const PIPELINE_WHIR_COMPRESSED2_G22_COMPACT: u8 = 21;
pub const PIPELINE_WHIR_COMPRESSED2_G22_DEDUP: u8 = 22;
pub const PIPELINE_WHIR_COMPRESSED2_G26: u8 = 23;
pub const PIPELINE_WHIR_COMPRESSED2_G26_COMPACT: u8 = 24;
pub const PIPELINE_WHIR_COMPRESSED2_G26_DEDUP: u8 = 25;

/// A twice-compressed file's format as a verifier names it: C2's query grind
/// and the file's encoding. A constant of the caller, never the file's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Compressed2Format {
    pub grind: u8,
    pub wire: super::wire::WireFormat,
}

impl Compressed2Format {
    /// The tag a file of this format carries; refused for a grind C2 is not
    /// built at.
    pub fn tag(&self) -> Result<u8, String> {
        use super::wire::WireFormat;
        let rkyv = match self.grind {
            22 => PIPELINE_WHIR_COMPRESSED2_G22,
            26 => PIPELINE_WHIR_COMPRESSED2_G26,
            other => return Err(format!("no twice-compressed verifier takes grind {other}")),
        };
        Ok(rkyv
            + match self.wire {
                WireFormat::Rkyv => 0,
                WireFormat::Compact => 1,
                WireFormat::Dedup => 2,
            })
    }
}

/// The six twice-compressed formats (tags 20–25).
pub const COMPRESSED2_G22: Compressed2Format = Compressed2Format {
    grind: 22,
    wire: super::wire::WireFormat::Rkyv,
};
pub const COMPRESSED2_G22_COMPACT: Compressed2Format = Compressed2Format {
    grind: 22,
    wire: super::wire::WireFormat::Compact,
};
pub const COMPRESSED2_G22_DEDUP: Compressed2Format = Compressed2Format {
    grind: 22,
    wire: super::wire::WireFormat::Dedup,
};
pub const COMPRESSED2_G26: Compressed2Format = Compressed2Format {
    grind: 26,
    wire: super::wire::WireFormat::Rkyv,
};
pub const COMPRESSED2_G26_COMPACT: Compressed2Format = Compressed2Format {
    grind: 26,
    wire: super::wire::WireFormat::Compact,
};
pub const COMPRESSED2_G26_DEDUP: Compressed2Format = Compressed2Format {
    grind: 26,
    wire: super::wire::WireFormat::Dedup,
};

/// Which compressed file a run ships, as a verifier names it: C1's (its
/// levers and encoding) or C2's (its grind and encoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShippedFormat {
    C1(CompressedFormat),
    C2(Compressed2Format),
}

impl ShippedFormat {
    /// The file a run under `plan` ships; `None` without compression (the
    /// top's file).
    pub fn of(plan: super::compress::CompressPlan) -> Option<Self> {
        plan.rate?;
        Some(match plan.c2 {
            Some(grind) => Self::C2(Compressed2Format {
                grind,
                wire: plan.wire,
            }),
            None => Self::C1(CompressedFormat {
                levers: plan.levers,
                wire: plan.wire,
            }),
        })
    }

    /// The file's tag.
    pub fn tag(&self) -> Result<u8, String> {
        match self {
            Self::C1(format) => format.tag(),
            Self::C2(format) => format.tag(),
        }
    }

    /// The file's encoding.
    pub fn wire(&self) -> super::wire::WireFormat {
        match self {
            Self::C1(format) => format.wire,
            Self::C2(format) => format.wire,
        }
    }
}

/// ★ The file a run ships by default ([`super::compress::DEFAULT_PLAN`]): what
/// `cli verify-block` verifies unless told the top's file.
pub const DEFAULT_SHIPPED: ShippedFormat = match super::compress::DEFAULT_COMPRESS2 {
    Some(grind) => ShippedFormat::C2(Compressed2Format {
        grind,
        wire: super::compress::DEFAULT_WIRE,
    }),
    None => ShippedFormat::C1(CompressedFormat {
        levers: super::compress::DEFAULT_C1_LEVERS,
        wire: super::compress::DEFAULT_WIRE,
    }),
};

/// A shipped file's bytes read back in `format`'s encoding (the CALLER'S
/// constant): the rkyv reader for an rkyv format, the compact reader otherwise.
/// The verifier checks the tag after.
pub fn read_shipped(
    bytes: &[u8],
    format: ShippedFormat,
) -> Result<WhirBlockCompressedProof, String> {
    match format.wire() {
        super::wire::WireFormat::Rkyv => WhirBlockCompressedProof::from_bytes(bytes),
        _ => WhirBlockCompressedProof::from_compact_bytes(bytes),
    }
}

/// ★ A shipped W3 block file verified as `format` — the CALLER'S constant,
/// never the file's: C1's verifier ([`verify_whir_block_compressed_as`]) or
/// C2's ([`verify_whir_block_compressed2_as`]), each checking the file's tag
/// first.
pub fn verify_whir_block_shipped_as(
    elf_bytes: &[u8],
    proof: &WhirBlockCompressedProof,
    format: ShippedFormat,
) -> Result<(), String> {
    match format {
        ShippedFormat::C1(format) => verify_whir_block_compressed_as(elf_bytes, proof, format),
        ShippedFormat::C2(format) => verify_whir_block_compressed2_as(elf_bytes, proof, format),
    }
}

impl CompressedTop {
    /// The plan this run proved under.
    pub fn plan(&self) -> super::compress::CompressPlan {
        super::compress::CompressPlan {
            rate: Some(self.rate),
            levers: self.levers,
            wire: self.wire,
            c2: self.c2.as_ref().map(|c2| c2.grind),
        }
    }

    /// The file this run ships for `statement`, and its bytes: C2's when C2
    /// ran, C1's otherwise, in the run's encoding.
    pub fn shipped(
        &self,
        statement: block_whir::OwnedBlockStatement,
    ) -> Result<(ShippedFormat, rkyv::util::AlignedVec), String> {
        let format = ShippedFormat::of(self.plan()).ok_or("a compressed run ships a file")?;
        let file = match &self.c2 {
            Some(c2) => WhirBlockCompressedProof::for_c2(statement, c2.proof.clone(), c2.grind)?,
            None => {
                WhirBlockCompressedProof::for_levers(statement, self.proof.clone(), self.levers)?
            }
        };
        Ok((format, file.to_wire_bytes(self.wire)?))
    }
}

/// ★ A twice-compressed W3 block file verified as `format` — the CALLER'S
/// constant, never the file's: the file's tag must be `format`'s before
/// anything else is read; then the plan is derived from the trusted ELF and
/// the claimed statement under the pinned presets, C1 over the derived top at
/// [`super::compress::COMPRESS_RATE`] with today's levers, C2 over C1 at
/// [`super::compress::COMPRESS2_RATE`] and `format`'s grind, and the file's C2
/// proof checked against it and to claim the block.
pub fn verify_whir_block_compressed2_as(
    elf_bytes: &[u8],
    proof: &WhirBlockCompressedProof,
    format: Compressed2Format,
) -> Result<(), String> {
    require_compressed_tag(proof, format.tag()?)?;
    let presets = super::whir_block::pinned_verifier_presets(BlockFormat::pinned().group);
    let plan = super::whir_block::verifier_plan(elf_bytes, proof.statement(), &presets)?;
    super::compress::verify_compressed2(
        &plan,
        &presets.wrap_options,
        format.grind,
        &proof.compressed,
    )
}

/// The dedup twins (`LAMBDA_VM_BLOCK_COMPRESS_WIRE=dedup`): tags 14–17 are
/// 2–5's, 18–19 are 10–11's, C1's proof compact with every repeated Merkle
/// sibling sent once ([`super::wire::encode_dedup`]).
pub const PIPELINE_DEDUP_OFFSET: u8 = 12;
pub const PIPELINE_WHIR_COMPRESSED_DEDUP: u8 = 14;
pub const PIPELINE_WHIR_COMPRESSED_DEDUP_EVALUATED: u8 = 15;
pub const PIPELINE_WHIR_COMPRESSED_DEDUP_FINAL: u8 = 16;
pub const PIPELINE_WHIR_COMPRESSED_DEDUP_EVALUATED_FINAL: u8 = 17;
pub const PIPELINE_WHIR_COMPRESSED_DEDUP_FOLDS: u8 = 18;
pub const PIPELINE_WHIR_COMPRESSED_DEDUP_FOLDS_EVALUATED: u8 = 19;

/// Whether `tag` is a dedup compressed file's.
fn is_dedup_compressed(tag: u8) -> bool {
    (PIPELINE_WHIR_COMPRESSED_DEDUP..=PIPELINE_WHIR_COMPRESSED_DEDUP_FOLDS_EVALUATED).contains(&tag)
        || tag == PIPELINE_WHIR_COMPRESSED2_G22_DEDUP
        || tag == PIPELINE_WHIR_COMPRESSED2_G26_DEDUP
}

/// The dedup twin of an rkyv compressed file's tag.
fn dedup_twin(tag: u8) -> Option<u8> {
    if (PIPELINE_WHIR_COMPRESSED..=PIPELINE_WHIR_COMPRESSED_EVALUATED_FINAL).contains(&tag) {
        Some(tag + PIPELINE_DEDUP_OFFSET)
    } else if (PIPELINE_WHIR_COMPRESSED_FOLDS..=PIPELINE_WHIR_COMPRESSED_FOLDS_EVALUATED)
        .contains(&tag)
    {
        Some(tag + (PIPELINE_WHIR_COMPRESSED_DEDUP_FOLDS - PIPELINE_WHIR_COMPRESSED_FOLDS))
    } else if tag == PIPELINE_WHIR_COMPRESSED2_G22 || tag == PIPELINE_WHIR_COMPRESSED2_G26 {
        Some(tag + 2)
    } else {
        None
    }
}

/// Whether `tag` is a compact compressed file's.
fn is_compact_compressed(tag: u8) -> bool {
    (PIPELINE_WHIR_COMPRESSED_COMPACT..=PIPELINE_WHIR_COMPRESSED_COMPACT_EVALUATED_FINAL)
        .contains(&tag)
        || (PIPELINE_WHIR_COMPRESSED_COMPACT_FOLDS
            ..=PIPELINE_WHIR_COMPRESSED_COMPACT_FOLDS_EVALUATED)
            .contains(&tag)
        || tag == PIPELINE_WHIR_COMPRESSED2_G22_COMPACT
        || tag == PIPELINE_WHIR_COMPRESSED2_G26_COMPACT
}

/// The compact twin of an rkyv compressed file's tag.
fn compact_twin(tag: u8) -> Option<u8> {
    if (PIPELINE_WHIR_COMPRESSED..=PIPELINE_WHIR_COMPRESSED_EVALUATED_FINAL).contains(&tag) {
        Some(tag + PIPELINE_COMPACT_OFFSET)
    } else if (PIPELINE_WHIR_COMPRESSED_FOLDS..=PIPELINE_WHIR_COMPRESSED_FOLDS_EVALUATED)
        .contains(&tag)
    {
        Some(tag + (PIPELINE_WHIR_COMPRESSED_COMPACT_FOLDS - PIPELINE_WHIR_COMPRESSED_FOLDS))
    } else if tag == PIPELINE_WHIR_COMPRESSED2_G22 || tag == PIPELINE_WHIR_COMPRESSED2_G26 {
        Some(tag + 1)
    } else {
        None
    }
}

/// A compressed file's format as a verifier names it: C1's levers and whether
/// the file is compact. A constant of the caller, never the file's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompressedFormat {
    pub levers: super::wlfm::WlfmLevers,
    pub wire: super::wire::WireFormat,
}

impl CompressedFormat {
    /// The tag a file of this format carries; refused for a setting no
    /// verifier takes.
    pub fn tag(&self) -> Result<u8, String> {
        use super::wire::WireFormat;
        let rkyv = compressed_pipeline_of(self.levers)?;
        match self.wire {
            WireFormat::Rkyv => Ok(rkyv),
            WireFormat::Compact => {
                compact_twin(rkyv).ok_or_else(|| format!("pipeline {rkyv} has no compact twin"))
            }
            WireFormat::Dedup => {
                dedup_twin(rkyv).ok_or_else(|| format!("pipeline {rkyv} has no dedup twin"))
            }
        }
    }
}

/// The byte-minimising schedule's four formats (tags 10–13).
pub const COMPRESSED_FOLDS: CompressedFormat = CompressedFormat {
    levers: super::compress::folds_levers(super::wlfm::WlfmPrep::Stacked),
    wire: super::wire::WireFormat::Rkyv,
};
pub const COMPRESSED_FOLDS_EVALUATED: CompressedFormat = CompressedFormat {
    levers: super::compress::folds_levers(super::wlfm::WlfmPrep::Evaluated),
    wire: super::wire::WireFormat::Rkyv,
};
pub const COMPRESSED_COMPACT_FOLDS: CompressedFormat = CompressedFormat {
    levers: super::compress::folds_levers(super::wlfm::WlfmPrep::Stacked),
    wire: super::wire::WireFormat::Compact,
};
pub const COMPRESSED_COMPACT_FOLDS_EVALUATED: CompressedFormat = CompressedFormat {
    levers: super::compress::folds_levers(super::wlfm::WlfmPrep::Evaluated),
    wire: super::wire::WireFormat::Compact,
};

/// The dedup file of C1 alone (tag 14) and of every lever (tag 19).
pub const COMPRESSED_DEDUP: CompressedFormat = CompressedFormat {
    levers: super::wlfm::WlfmLevers::DEFAULT,
    wire: super::wire::WireFormat::Dedup,
};
pub const COMPRESSED_DEDUP_FOLDS: CompressedFormat = CompressedFormat {
    levers: super::compress::folds_levers(super::wlfm::WlfmPrep::Stacked),
    wire: super::wire::WireFormat::Dedup,
};
pub const COMPRESSED_DEDUP_FOLDS_EVALUATED: CompressedFormat = CompressedFormat {
    levers: super::compress::folds_levers(super::wlfm::WlfmPrep::Evaluated),
    wire: super::wire::WireFormat::Dedup,
};

/// ★ A compressed W3 block file verified as `format` — the CALLER'S constant
/// (one of the `COMPRESSED_*` formats), never the file's: the file's tag must
/// be `format`'s before anything else is read, then C1 is built under
/// `format`'s levers at [`super::compress::COMPRESS_RATE`].
pub fn verify_whir_block_compressed_as(
    elf_bytes: &[u8],
    proof: &WhirBlockCompressedProof,
    format: CompressedFormat,
) -> Result<(), String> {
    require_compressed_tag(proof, format.tag()?)?;
    let presets = super::whir_block::pinned_verifier_presets(BlockFormat::pinned().group);
    let plan = super::whir_block::verifier_plan(elf_bytes, proof.statement(), &presets)?;
    super::compress::verify_compressed(
        &plan,
        &presets.wrap_options,
        super::compress::COMPRESS_RATE,
        format.levers,
        &proof.compressed,
    )
}

/// A whole W3 block's proof compressed (`LAMBDA_VM_BLOCK_COMPRESS`): the
/// block's statement and C1's proof over the top. As for
/// [`WhirBlockTreeProof`], nothing here is a format parameter: the rate C1 is
/// checked at is the verifier's constant ([`super::compress::COMPRESS_RATE`]).
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct WhirBlockCompressedProof {
    magic: [u8; 8],
    version: u32,
    pipeline: u8,
    pub table_num_vars: Vec<u8>,
    pub runtime_page_ranges: Vec<crate::RuntimePageRange>,
    pub table_counts: crate::TableCounts,
    pub public_output: Vec<u8>,
    pub num_private_input_pages: usize,
    pub groups: Vec<Vec<u32>>,
    pub compressed: super::wlfm::WlfmProof,
}

impl WhirBlockCompressedProof {
    /// The file for a W3 block's statement and C1's (stacked) proof.
    pub fn new(statement: block_whir::OwnedBlockStatement, c1: super::wlfm::WlfmProof) -> Self {
        Self::for_binding(statement, c1, super::wlfm::WlfmPrep::Stacked)
    }

    /// The file for a W3 block's statement and C1's proof bound per `prep`,
    /// tagged for that binding's verifier ([`compressed_pipeline`]).
    pub fn for_binding(
        statement: block_whir::OwnedBlockStatement,
        c1: super::wlfm::WlfmProof,
        prep: super::wlfm::WlfmPrep,
    ) -> Self {
        Self {
            magic: PROOF_MAGIC,
            version: PROOF_VERSION,
            pipeline: compressed_pipeline(prep),
            table_num_vars: statement.table_num_vars,
            runtime_page_ranges: statement.runtime_page_ranges,
            table_counts: statement.table_counts,
            public_output: statement.public_output,
            num_private_input_pages: statement.num_private_input_pages,
            groups: statement.groups,
            compressed: c1,
        }
    }

    /// The file for C2's proof (over C1) at `grind`, tagged for that grind's
    /// rkyv verifier ([`Compressed2Format`]).
    pub fn for_c2(
        statement: block_whir::OwnedBlockStatement,
        c2: super::wlfm::WlfmProof,
        grind: u8,
    ) -> Result<Self, String> {
        Ok(Self {
            magic: PROOF_MAGIC,
            version: PROOF_VERSION,
            pipeline: Compressed2Format {
                grind,
                wire: super::wire::WireFormat::Rkyv,
            }
            .tag()?,
            table_num_vars: statement.table_num_vars,
            runtime_page_ranges: statement.runtime_page_ranges,
            table_counts: statement.table_counts,
            public_output: statement.public_output,
            num_private_input_pages: statement.num_private_input_pages,
            groups: statement.groups,
            compressed: c2,
        })
    }

    /// The file for C1's proof under `levers`, tagged for that setting's
    /// verifier ([`compressed_pipeline_of`]).
    pub fn for_levers(
        statement: block_whir::OwnedBlockStatement,
        c1: super::wlfm::WlfmProof,
        levers: super::wlfm::WlfmLevers,
    ) -> Result<Self, String> {
        Ok(Self {
            magic: PROOF_MAGIC,
            version: PROOF_VERSION,
            pipeline: compressed_pipeline_of(levers)?,
            table_num_vars: statement.table_num_vars,
            runtime_page_ranges: statement.runtime_page_ranges,
            table_counts: statement.table_counts,
            public_output: statement.public_output,
            num_private_input_pages: statement.num_private_input_pages,
            groups: statement.groups,
            compressed: c1,
        })
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

    /// A file read back: refused unless it is this layout's compressed W3
    /// block. `bytes` must be aligned for rkyv (an `AlignedVec`).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let proof = rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes)
            .map_err(|e| format!("not a compressed W3 block proof file: {e}"))?;
        if proof.magic != PROOF_MAGIC {
            return Err("not a block proof file (magic)".to_string());
        }
        if proof.version != PROOF_VERSION {
            return Err(format!(
                "block proof file version {}, this build reads {PROOF_VERSION}",
                proof.version
            ));
        }
        if !is_rkyv_compressed(proof.pipeline) {
            return Err(format!(
                "the file holds pipeline {}, not the compressed WHIR block tree",
                proof.pipeline
            ));
        }
        Ok(proof)
    }

    /// The file's tag: which compressed verifier it is for.
    pub fn pipeline(&self) -> u8 {
        self.pipeline
    }

    /// The file's compact bytes (`LAMBDA_VM_BLOCK_COMPRESS_WIRE`): the
    /// statement as rkyv writes it, C1's proof in [`super::wire`]'s encoding,
    /// under the compact twin of this file's tag
    /// ([`PIPELINE_COMPACT_OFFSET`] above it).
    pub fn to_compact_bytes(&self) -> Result<rkyv::util::AlignedVec, String> {
        self.to_wire_bytes(super::wire::WireFormat::Compact)
    }

    /// The file's bytes in `wire`'s encoding, under that encoding's twin of
    /// this file's tag.
    pub fn to_wire_bytes(
        &self,
        wire: super::wire::WireFormat,
    ) -> Result<rkyv::util::AlignedVec, String> {
        use super::wire::WireFormat;
        let twin = match wire {
            WireFormat::Rkyv => return self.to_bytes(),
            WireFormat::Compact => compact_twin(self.pipeline),
            WireFormat::Dedup => dedup_twin(self.pipeline),
        };
        let Some(twin) = twin else {
            return Err(format!("pipeline {} has no {wire:?} twin", self.pipeline));
        };
        let file = WhirBlockCompactFile {
            magic: PROOF_MAGIC,
            version: PROOF_VERSION,
            pipeline: twin,
            table_num_vars: self.table_num_vars.clone(),
            runtime_page_ranges: self.runtime_page_ranges.clone(),
            table_counts: self.table_counts.clone(),
            public_output: self.public_output.clone(),
            num_private_input_pages: self.num_private_input_pages,
            groups: self.groups.clone(),
            compact: super::wire::encode_as(&self.compressed, wire).unwrap_or_default(),
        };
        rkyv::to_bytes::<rkyv::rancor::Error>(&file).map_err(|e| format!("serialize: {e}"))
    }

    /// A compact file read back as the proof it carries, tagged with the
    /// compact tag (so only a compact verifier takes it). Refused unless it is
    /// this layout's compact compressed W3 block and C1's bytes are exactly one
    /// proof's encoding. `bytes` must be aligned for rkyv.
    pub fn from_compact_bytes(bytes: &[u8]) -> Result<Self, String> {
        let file = rkyv::from_bytes::<WhirBlockCompactFile, rkyv::rancor::Error>(bytes)
            .map_err(|e| format!("not a compact compressed W3 block proof file: {e}"))?;
        if file.magic != PROOF_MAGIC {
            return Err("not a block proof file (magic)".to_string());
        }
        if file.version != PROOF_VERSION {
            return Err(format!(
                "block proof file version {}, this build reads {PROOF_VERSION}",
                file.version
            ));
        }
        if !is_compact_compressed(file.pipeline) && !is_dedup_compressed(file.pipeline) {
            return Err(format!(
                "the file holds pipeline {}, not a compact compressed WHIR block tree",
                file.pipeline
            ));
        }
        Ok(Self {
            magic: file.magic,
            version: file.version,
            pipeline: file.pipeline,
            table_num_vars: file.table_num_vars,
            runtime_page_ranges: file.runtime_page_ranges,
            table_counts: file.table_counts,
            public_output: file.public_output,
            num_private_input_pages: file.num_private_input_pages,
            groups: file.groups,
            compressed: if is_dedup_compressed(file.pipeline) {
                super::wire::decode_dedup(&file.compact)?
            } else {
                super::wire::decode(&file.compact)?
            },
        })
    }
}

/// How far a compact file's tag sits above its rkyv twin's: tags 6–9 are 2–5
/// with C1's proof in [`super::wire`]'s encoding (`LAMBDA_VM_BLOCK_COMPRESS_WIRE`).
pub const PIPELINE_COMPACT_OFFSET: u8 = 4;
/// The compact twin of [`PIPELINE_WHIR_COMPRESSED`].
pub const PIPELINE_WHIR_COMPRESSED_COMPACT: u8 = 6;
/// The compact twin of [`PIPELINE_WHIR_COMPRESSED_EVALUATED`].
pub const PIPELINE_WHIR_COMPRESSED_COMPACT_EVALUATED: u8 = 7;
/// The compact twin of [`PIPELINE_WHIR_COMPRESSED_FINAL`].
pub const PIPELINE_WHIR_COMPRESSED_COMPACT_FINAL: u8 = 8;
/// The compact twin of [`PIPELINE_WHIR_COMPRESSED_EVALUATED_FINAL`].
pub const PIPELINE_WHIR_COMPRESSED_COMPACT_EVALUATED_FINAL: u8 = 9;

/// The compact file: [`WhirBlockCompressedProof`]'s statement fields, rkyv as
/// there, and C1's proof as [`super::wire`] bytes.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct WhirBlockCompactFile {
    magic: [u8; 8],
    version: u32,
    pipeline: u8,
    table_num_vars: Vec<u8>,
    runtime_page_ranges: Vec<crate::RuntimePageRange>,
    table_counts: crate::TableCounts,
    public_output: Vec<u8>,
    num_private_input_pages: usize,
    groups: Vec<Vec<u32>>,
    compact: Vec<u8>,
}

/// A compact compressed file verified under `levers` once it carries `tag`:
/// the compact verifiers' one body, each naming its own constants.
fn verify_compact_file(
    elf_bytes: &[u8],
    proof: &WhirBlockCompressedProof,
    tag: u8,
    levers: super::wlfm::WlfmLevers,
) -> Result<(), String> {
    require_compressed_tag(proof, tag)?;
    let presets = super::whir_block::pinned_verifier_presets(BlockFormat::pinned().group);
    let plan = super::whir_block::verifier_plan(elf_bytes, proof.statement(), &presets)?;
    super::compress::verify_compressed(
        &plan,
        &presets.wrap_options,
        super::compress::COMPRESS_RATE,
        levers,
        &proof.compressed,
    )
}

/// ★ The compact compressed W3 block verifier (tag 6): the pinned verifier's
/// C1 read from [`WhirBlockCompressedProof::from_compact_bytes`].
pub fn verify_whir_block_compressed_compact_proof(
    elf_bytes: &[u8],
    proof: &WhirBlockCompressedProof,
) -> Result<(), String> {
    verify_compact_file(
        elf_bytes,
        proof,
        PIPELINE_WHIR_COMPRESSED_COMPACT,
        super::wlfm::WlfmLevers {
            prep: super::compress::COMPRESS_PREP,
            final_poly: super::compress::COMPRESS_FINAL,
            fold: None,
            grind: None,
        },
    )
}

/// ★ Compact, C1's instruction columns evaluated (tag 7).
pub fn verify_whir_block_compressed_compact_evaluated_proof(
    elf_bytes: &[u8],
    proof: &WhirBlockCompressedProof,
) -> Result<(), String> {
    verify_compact_file(
        elf_bytes,
        proof,
        PIPELINE_WHIR_COMPRESSED_COMPACT_EVALUATED,
        super::wlfm::WlfmLevers {
            prep: super::wlfm::WlfmPrep::Evaluated,
            final_poly: super::compress::COMPRESS_FINAL,
            fold: None,
            grind: None,
        },
    )
}

/// ★ Compact, C1's chains stopping at the final polynomial (tag 8).
pub fn verify_whir_block_compressed_compact_final_proof(
    elf_bytes: &[u8],
    proof: &WhirBlockCompressedProof,
) -> Result<(), String> {
    verify_compact_file(
        elf_bytes,
        proof,
        PIPELINE_WHIR_COMPRESSED_COMPACT_FINAL,
        super::wlfm::WlfmLevers {
            prep: super::compress::COMPRESS_PREP,
            final_poly: multilinear::whir_chain::FinalPoly::Vars(
                super::compress::COMPRESS_FINAL_VARS,
            ),
            fold: None,
            grind: None,
        },
    )
}

/// ★ Compact, both: evaluated columns and the final polynomial (tag 9).
pub fn verify_whir_block_compressed_compact_evaluated_final_proof(
    elf_bytes: &[u8],
    proof: &WhirBlockCompressedProof,
) -> Result<(), String> {
    verify_compact_file(
        elf_bytes,
        proof,
        PIPELINE_WHIR_COMPRESSED_COMPACT_EVALUATED_FINAL,
        super::wlfm::WlfmLevers {
            prep: super::wlfm::WlfmPrep::Evaluated,
            final_poly: multilinear::whir_chain::FinalPoly::Vars(
                super::compress::COMPRESS_FINAL_VARS,
            ),
            fold: None,
            grind: None,
        },
    )
}

/// A compressed file refused unless it carries `tag`, the calling verifier's
/// own: run before the verifier reads anything else.
fn require_compressed_tag(proof: &WhirBlockCompressedProof, tag: u8) -> Result<(), String> {
    if proof.pipeline != tag {
        return Err(format!(
            "the file holds pipeline {}, and this verifier reads pipeline {tag}",
            proof.pipeline
        ));
    }
    Ok(())
}

/// ★ A compressed W3 block proof file verified: the plan derived from the
/// trusted ELF and the claimed statement under the pinned presets (as
/// [`verify_whir_block_tree_proof`]), C1 built over the derived top at
/// [`super::compress::COMPRESS_RATE`] and [`super::compress::COMPRESS_PREP`],
/// and the file's C1 proof checked against it and to claim the block. No format
/// choice is the caller's or the file's.
pub fn verify_whir_block_compressed_proof(
    elf_bytes: &[u8],
    proof: &WhirBlockCompressedProof,
) -> Result<(), String> {
    require_compressed_tag(proof, PIPELINE_WHIR_COMPRESSED)?;
    let presets = super::whir_block::pinned_verifier_presets(BlockFormat::pinned().group);
    let plan = super::whir_block::verifier_plan(elf_bytes, proof.statement(), &presets)?;
    super::compress::verify_compressed(
        &plan,
        &presets.wrap_options,
        super::compress::COMPRESS_RATE,
        super::wlfm::WlfmLevers {
            prep: super::compress::COMPRESS_PREP,
            final_poly: super::compress::COMPRESS_FINAL,
            fold: None,
            grind: None,
        },
        &proof.compressed,
    )
}

/// ★ The evaluated arm's compressed W3 block verifier
/// (`LAMBDA_VM_BLOCK_COMPRESS_EVAL`): [`verify_whir_block_compressed_proof`]
/// with C1's instruction columns evaluated by this verifier
/// ([`super::wlfm::WlfmPrep::Evaluated`]), a constant of this function. A file
/// of one binding is refused by the other's verifier up front, by its tag
/// ([`PIPELINE_WHIR_COMPRESSED_EVALUATED`]); behind that, the binding is folded
/// into C1's identity, which the statement absorbs before any root.
pub fn verify_whir_block_compressed_evaluated_proof(
    elf_bytes: &[u8],
    proof: &WhirBlockCompressedProof,
) -> Result<(), String> {
    require_compressed_tag(proof, PIPELINE_WHIR_COMPRESSED_EVALUATED)?;
    let presets = super::whir_block::pinned_verifier_presets(BlockFormat::pinned().group);
    let plan = super::whir_block::verifier_plan(elf_bytes, proof.statement(), &presets)?;
    super::compress::verify_compressed(
        &plan,
        &presets.wrap_options,
        super::compress::COMPRESS_RATE,
        super::wlfm::WlfmLevers {
            prep: super::wlfm::WlfmPrep::Evaluated,
            final_poly: super::compress::COMPRESS_FINAL,
            fold: None,
            grind: None,
        },
        &proof.compressed,
    )
}

/// ★ The final-polynomial arm's compressed W3 block verifier
/// (`LAMBDA_VM_BLOCK_COMPRESS_FINAL`): [`verify_whir_block_compressed_proof`]
/// with C1's chains stopping at a final polynomial of
/// [`super::compress::COMPRESS_FINAL_VARS`] variables, a constant of this
/// function. Its tag ([`PIPELINE_WHIR_COMPRESSED_FINAL`]) is checked first.
pub fn verify_whir_block_compressed_final_proof(
    elf_bytes: &[u8],
    proof: &WhirBlockCompressedProof,
) -> Result<(), String> {
    require_compressed_tag(proof, PIPELINE_WHIR_COMPRESSED_FINAL)?;
    let presets = super::whir_block::pinned_verifier_presets(BlockFormat::pinned().group);
    let plan = super::whir_block::verifier_plan(elf_bytes, proof.statement(), &presets)?;
    super::compress::verify_compressed(
        &plan,
        &presets.wrap_options,
        super::compress::COMPRESS_RATE,
        super::wlfm::WlfmLevers {
            prep: super::compress::COMPRESS_PREP,
            final_poly: multilinear::whir_chain::FinalPoly::Vars(
                super::compress::COMPRESS_FINAL_VARS,
            ),
            fold: None,
            grind: None,
        },
        &proof.compressed,
    )
}

/// ★ Both levers: C1's instruction columns evaluated and its chains stopping
/// at the final polynomial; tag [`PIPELINE_WHIR_COMPRESSED_EVALUATED_FINAL`].
pub fn verify_whir_block_compressed_evaluated_final_proof(
    elf_bytes: &[u8],
    proof: &WhirBlockCompressedProof,
) -> Result<(), String> {
    require_compressed_tag(proof, PIPELINE_WHIR_COMPRESSED_EVALUATED_FINAL)?;
    let presets = super::whir_block::pinned_verifier_presets(BlockFormat::pinned().group);
    let plan = super::whir_block::verifier_plan(elf_bytes, proof.statement(), &presets)?;
    super::compress::verify_compressed(
        &plan,
        &presets.wrap_options,
        super::compress::COMPRESS_RATE,
        super::wlfm::WlfmLevers {
            prep: super::wlfm::WlfmPrep::Evaluated,
            final_poly: multilinear::whir_chain::FinalPoly::Vars(
                super::compress::COMPRESS_FINAL_VARS,
            ),
            fold: None,
            grind: None,
        },
        &proof.compressed,
    )
}

/// ★ A W3 block proof file verified: [`super::whir_block::verify_block_tree`]
/// over its claimed statement and its top proof, under the pinned block
/// presets (constants, never the environment's), against the trusted
/// `elf_bytes`.
pub fn verify_whir_block_tree_proof(
    elf_bytes: &[u8],
    proof: &WhirBlockTreeProof,
) -> Result<(), String> {
    verify_whir_block_tree_proof_grouped(elf_bytes, proof, BlockFormat::pinned().group)
}

/// [`verify_whir_block_tree_proof`] for a block whose groups were opened per
/// `group` — the verifier's constant, the caller's to give
/// ([`super::whir_block::verify_block_tree_grouped`]); the file does not
/// carry it.
pub fn verify_whir_block_tree_proof_grouped(
    elf_bytes: &[u8],
    proof: &WhirBlockTreeProof,
    group: GroupFold,
) -> Result<(), String> {
    let top = LfmProof {
        proof: proof.top_proof.clone(),
        public_words: proof.top_public_words.clone(),
    };
    super::whir_block::verify_block_tree_grouped(elf_bytes, proof.statement(), &top, group)
}

/// [`verify_whir_block_tree_proof`] at the pinned format with the caller's
/// named choices, its group opening, its WHIR blowup, its round-0 leaf order,
/// its round-0 trees, its leaf scale, its leaves' query check and its GKR
/// packing
/// ([`super::whir_block::verify_block_tree_at`]); the file carries none of
/// them, and the environment sets none of the rest. No public entry point
/// takes a format of the caller's own.
///
/// ```no_run
/// use lambda_vm_prover::lfm::whir_block::LeafScale;
/// use lambda_vm_prover::lfm::whir_block_tree::{
///     WhirBlockTreeProof, verify_whir_block_tree_proof_at,
/// };
/// use lambda_vm_prover::lfm::whir_chain::LeafQuery;
/// use lambda_vm_prover::zf_format::WhirBlowup;
/// use multilinear::whir_chain::{GkrPack, GroupFold, LeafOrder, Round0Trees};
///
/// fn check(elf: &[u8], proof: &WhirBlockTreeProof) -> Result<(), String> {
///     verify_whir_block_tree_proof_at(
///         elf,
///         proof,
///         GroupFold::Indexed,
///         WhirBlowup::Two,
///         LeafOrder::Coset,
///         Round0Trees::Each,
///         LeafScale::DoubleFrom(12),
///         LeafQuery::Lean,
///         GkrPack::Off,
///     )
/// }
/// ```
///
/// The any-format verifier is crate-internal, so outside the crate this does
/// not compile:
///
/// ```compile_fail
/// use lambda_vm_prover::lfm::whir_block::verify_block_tree_as;
/// ```
#[allow(clippy::too_many_arguments)]
pub fn verify_whir_block_tree_proof_at(
    elf_bytes: &[u8],
    proof: &WhirBlockTreeProof,
    group: GroupFold,
    blowup: WhirBlowup,
    leaves: LeafOrder,
    trees: Round0Trees,
    leaf_scale: LeafScale,
    leaf_query: LeafQuery,
    pack: GkrPack,
) -> Result<(), String> {
    let top = LfmProof {
        proof: proof.top_proof.clone(),
        public_words: proof.top_public_words.clone(),
    };
    super::whir_block::verify_block_tree_at(
        elf_bytes,
        proof.statement(),
        &top,
        group,
        blowup,
        leaves,
        trees,
        leaf_scale,
        leaf_query,
        pack,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With the early leaf first (`W3_LEAF_FIRST`), the tree's order takes
    /// that leaf before every other and is otherwise [`tree_order`]'s: a
    /// permutation of the leaves alone, every node after every leaf. With no
    /// early leaf, leaf 0, or a leaf out of range it is [`tree_order`]
    /// unchanged. Proved over it, with one worker (inline) or two, every
    /// program proves, and one worker takes the early leaf first.
    #[test]
    fn the_early_leaves_are_taken_first_in_the_order_they_completed() {
        let shape = vec![super::super::per_table_aggregator::Level {
            arities: vec![3, 2],
        }];
        let base = tree_order(5, &shape);
        let leaf = |j| TreeAt { lv: 0, j };
        let led = early_leaves_first(base.clone(), &[3, 1]);
        assert_eq!(&led[..5], &[leaf(3), leaf(1), leaf(0), leaf(2), leaf(4)]);
        assert_eq!(
            &led[5..],
            &base[5..],
            "the nodes keep their order, after every leaf"
        );
        assert_eq!(early_leaves_first(base.clone(), &[]), base);
        assert_eq!(
            early_leaves_first(base.clone(), &[2]),
            early_leaf_first(base.clone(), Some(2)),
            "one early leaf is the single case"
        );
        assert_eq!(early_leaves_setting(None), Ok(EARLY_LEAVES_DEFAULT));
        assert_eq!(early_leaves_setting(Some(1)), Ok(1), "the control");
        assert_eq!(early_leaves_setting(Some(3)), Ok(3));
        assert!(early_leaves_setting(Some(0)).is_err());
    }

    /// The on/off knobs (`W3_LEAF_ARTIFACTS_EACH`, `W3_NODE_TIMES`): unset and
    /// `0` off, `1` on, any other count refused by the knob's name instead of
    /// read as off.
    #[test]
    fn an_on_off_knob_takes_0_or_1_and_refuses_other_counts() {
        for name in ["W3_LEAF_ARTIFACTS_EACH", "W3_NODE_TIMES"] {
            assert_eq!(flag_setting(name, None), Ok(false), "{name} unset");
            assert_eq!(flag_setting(name, Some(0)), Ok(false), "{name}=0");
            assert_eq!(flag_setting(name, Some(1)), Ok(true), "{name}=1");
            for n in [2, 3, 10] {
                let err = flag_setting(name, Some(n)).expect_err("not 0 or 1");
                assert!(err.contains(name) && err.contains("0 or 1"), "{err}");
            }
        }
    }

    /// A later leaf goes early only when its predicted execute + fill ends
    /// before the base's predicted end; with either pace unknown it stays.
    #[test]
    fn the_early_leaves_fit_only_before_the_base_ends() {
        // Phase B: first group at 10 s, 100 cost units since in 5 s (0.05 s a
        // unit), 60 units to come: the base ends at 15 + 3 = 18 s.
        let fit =
            early_leaf_fits(15.0, 40, Some((1.0, 40)), Some(10.0), 100, 60).expect("both paces");
        assert_eq!(fit.base_end, 18.0);
        assert!((fit.fill_end - (15.0 + EARLY_FIT_FACTOR)).abs() < 1e-9);
        assert!(fit.fits);
        // Twice the first's cost and a slower first fill: past the base's end.
        let late =
            early_leaf_fits(15.0, 80, Some((1.5, 40)), Some(10.0), 100, 60).expect("both paces");
        assert!(!late.fits && late.fill_end > late.base_end);
        // Little left of phase B: no room.
        let tight =
            early_leaf_fits(17.9, 40, Some((1.0, 40)), Some(10.0), 158, 2).expect("both paces");
        assert!(!tight.fits);
        // Unknown paces: stay for the tree.
        assert_eq!(early_leaf_fits(15.0, 40, None, Some(10.0), 100, 60), None);
        assert_eq!(
            early_leaf_fits(15.0, 40, Some((1.0, 40)), None, 100, 60),
            None
        );
        assert_eq!(
            early_leaf_fits(15.0, 40, Some((1.0, 40)), Some(10.0), 0, 60),
            None
        );
        assert_eq!(
            early_leaf_fits(15.0, 40, Some((1.0, 0)), Some(10.0), 100, 60),
            None
        );
    }

    /// With several early leaves, up to `in_tree_ahead` leaves the tree still
    /// executes go first, then the early ones in completion order, then the
    /// rest; 0 is the early-first order; the nodes stay after every leaf.
    #[test]
    fn the_leaves_left_to_the_tree_start_their_execute_first() {
        use super::super::per_table_aggregator::Level;
        let shape = vec![
            Level {
                arities: vec![3, 1],
            },
            Level { arities: vec![2] },
        ];
        let base = tree_order(4, &shape);
        let leaf = |j| TreeAt { lv: 0, j };
        // The large block's case: early 0, 3, 1; leaf 2 left to the tree; 3 workers.
        let o = early_leaves_order(base.clone(), &[0, 3, 1], 2);
        assert_eq!(&o[..4], &[leaf(2), leaf(0), leaf(3), leaf(1)]);
        assert_eq!(&o[4..], &base[4..]);
        // More leaves left to the tree than room ahead: the rest follow the early ones.
        let shape8 = vec![
            Level {
                arities: vec![4, 4],
            },
            Level { arities: vec![2] },
        ];
        let base8 = tree_order(8, &shape8);
        let o = early_leaves_order(base8.clone(), &[5, 1], 2);
        let js: Vec<usize> = o[..8].iter().map(|t| t.j).collect();
        assert_eq!(js, vec![0, 2, 5, 1, 3, 4, 6, 7]);
        assert_eq!(
            early_leaves_order(base.clone(), &[0, 3, 1], 0),
            early_leaves_first(base.clone(), &[0, 3, 1])
        );
        // Every leaf early: their order.
        let o = early_leaves_order(base.clone(), &[3, 2, 1, 0], 2);
        assert_eq!(&o[..4], &[leaf(3), leaf(2), leaf(1), leaf(0)]);
    }

    /// Several early leaves only when phase B led several leaves; otherwise
    /// (no valid order, or one leaf led) the single early leaf.
    #[test]
    fn several_early_leaves_only_when_phase_b_led_several() {
        assert_eq!(early_room(EARLY_LEAVES_DEFAULT, 4), EARLY_LEAVES_DEFAULT);
        assert_eq!(early_room(EARLY_LEAVES_DEFAULT, 2), EARLY_LEAVES_DEFAULT);
        assert_eq!(early_room(EARLY_LEAVES_DEFAULT, 1), 1);
        assert_eq!(
            early_room(EARLY_LEAVES_DEFAULT, 0),
            1,
            "no valid order: the single early leaf"
        );
        assert_eq!(early_room(1, 4), 1, "the control");
        assert_eq!(early_room(3, 3), 3);
    }

    /// The early leaves' pool: none for 0 threads; otherwise that many
    /// threads of its own, and work installed on it runs there.
    #[test]
    fn the_early_pool_runs_its_work_on_its_own_threads() {
        assert!(early_pool(0).expect("builds").is_none());
        let pool = early_pool(2).expect("builds").expect("a pool");
        assert_eq!(pool.current_num_threads(), 2);
        let names: Vec<String> = pool.install(|| {
            use rayon::prelude::*;
            (0..64)
                .into_par_iter()
                .map(|_| std::thread::current().name().unwrap_or("").to_string())
                .collect()
        });
        assert!(
            names.iter().all(|n| n.starts_with("w3-early-")),
            "{names:?}"
        );
    }

    #[test]
    fn the_early_leaf_is_taken_first() {
        use super::super::per_table_aggregator::Level;
        let shape = vec![
            Level {
                arities: vec![3, 2],
            },
            Level { arities: vec![2] },
        ];
        let base = tree_order(5, &shape);
        for unchanged in [None, Some(0), Some(9)] {
            assert_eq!(
                early_leaf_first(base.clone(), unchanged),
                base,
                "{unchanged:?}"
            );
        }
        let led = early_leaf_first(base.clone(), Some(3));
        let leaves: Vec<TreeAt> = (0..5).map(|j| TreeAt { lv: 0, j }).collect();
        assert_eq!(
            led[..5],
            [leaves[3], leaves[0], leaves[1], leaves[2], leaves[4]]
        );
        assert_eq!(led[5..], base[5..]);
        for workers in [1, 2] {
            let results: Vec<Vec<Published<usize>>> = [5usize, 2, 1]
                .iter()
                .map(|&n| (0..n).map(|_| Published::new()).collect())
                .collect();
            let taken = std::sync::Mutex::new(Vec::new());
            prove_dataflow(&shape, &led, &results, workers, false, |at, kids| {
                taken.lock().expect("taken").push(at);
                wait_all(kids)?;
                Ok(at.lv)
            });
            for (lv, level) in results.iter().enumerate() {
                for slot in level {
                    assert_eq!(slot.wait(), Ok(&lv), "{workers} workers");
                }
            }
            let taken = taken.into_inner().expect("taken");
            assert_eq!(taken.len(), led.len(), "{workers} workers");
            if workers == 1 {
                assert_eq!(taken, led);
            }
        }
    }

    /// ★ No public block-tree verifier takes a format of the caller's own (a
    /// caller who could would choose the programs the top is checked against):
    /// every top-level `pub fn verify…` of this file and of `whir_block.rs` is
    /// read off the source and its parameters must not name `BlockFormat`, and
    /// the one any-format verifier, `whir_block::verify_block_tree_as`, is
    /// declared crate-internal. From outside the crate the same is checked by
    /// [`verify_whir_block_tree_proof_at`]'s `compile_fail` doc test. The
    /// verifiers of named choices (`…_at`) take each choice typed: the group
    /// opening, the WHIR blowup, the round-0 leaf order, the leaf scale and
    /// the leaves' query check.
    #[test]
    fn no_public_tree_verifier_takes_a_block_format() {
        let block = include_str!("whir_block.rs");
        for (file, src) in [
            ("whir_block.rs", block),
            ("whir_block_tree.rs", include_str!("whir_block_tree.rs")),
        ] {
            let mut public = Vec::new();
            for (at, _) in src.match_indices("\npub fn verify") {
                let sig = &src[at + 1..];
                let params = &sig[..sig.find(')').expect("a parameter list")];
                let name = &sig[7..sig.find('(').expect("a name")];
                assert!(
                    !params.contains("BlockFormat"),
                    "{file}: the public verifier {name} takes a BlockFormat"
                );
                if name.ends_with("_at") {
                    for choice in [
                        "GroupFold",
                        "WhirBlowup",
                        "LeafOrder",
                        "LeafScale",
                        "LeafQuery",
                    ] {
                        assert!(
                            params.contains(choice),
                            "{file}: {name} takes its {choice} as a named choice"
                        );
                    }
                }
                public.push(name.to_string());
            }
            assert!(
                public.len() >= 2,
                "{file}: the scan found its public verifiers: {public:?}"
            );
            assert!(
                public.iter().any(|name| name.ends_with("_at")),
                "{file}: a verifier of named choices: {public:?}"
            );
        }
        assert!(block.contains("\npub(crate) fn verify_block_tree_as("));
        assert!(!block.contains("\npub fn verify_block_tree_as("));
    }

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
        assert!(
            POSTURE
                .iter()
                .all(|(name, _)| *name != crate::whir_hash_knob::ENV),
            "the base hash is the format's, not posture"
        );
    }

    /// `LAMBDA_VM_WHIR_GROUP` reads two words, unset is production's one chain
    /// per run, and anything else is refused rather than defaulted.
    #[test]
    fn the_group_opening_reads_two_words() {
        assert_eq!(group_setting(None), Ok(GroupFold::Indexed));
        assert_eq!(BlockFormat::production().group, GroupFold::Indexed);
        assert_eq!(group_setting(Some("per-poly")), Ok(GroupFold::PerPoly));
        assert_eq!(group_setting(Some(" indexed ")), Ok(GroupFold::Indexed));
        for word in ["index", "Indexed", "1", ""] {
            assert!(group_setting(Some(word)).is_err(), "`{word}` is refused");
        }
        assert_eq!(
            BlockFormat::production_with(GroupFold::PerPoly).group,
            GroupFold::PerPoly
        );
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
            copy: 1,
            blake3: 0,
            is_wh: 0,
            ecrec: 0,
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
            proof: super::super::proof::LfmStarkProof {
                multi: stark::proof::stark::MultiProof { proofs: vec![] },
                commit_ood: vec![],
            },
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

    /// The synthetic statement with an empty C1 proof, as a compressed file.
    fn synthetic_compressed() -> WhirBlockCompressedProof {
        let tree = synthetic_proof();
        let statement = tree.statement().to_owned();
        let c1 = super::super::wlfm::WlfmProof {
            proof: stark::multilinear_table::MultiProof {
                roots: vec![[3u8; 32]],
                tables: vec![],
                columns: vec![],
                preprocessed: None,
            },
            argues: vec![],
            prepared: vec![],
            commit_ood: vec![],
            public_words: tree.top_public_words.clone(),
        };
        WhirBlockCompressedProof::new(statement, c1)
    }

    /// The compressed file reads back what was written; another tag, version
    /// or magic is refused; and neither file is read as the other.
    #[test]
    fn the_compressed_file_round_trips_and_the_two_files_refuse_each_other() {
        let proof = synthetic_compressed();
        let bytes = proof.to_bytes().expect("serializes");
        let back = WhirBlockCompressedProof::from_bytes(&bytes).expect("reads back");
        assert_eq!(
            format!("{:?}", back.statement()),
            format!("{:?}", proof.statement())
        );
        assert_eq!(back.compressed.public_words, proof.compressed.public_words);
        assert_eq!(back.compressed.proof.roots, proof.compressed.proof.roots);
        for (what, tamper) in [
            (
                "pipeline",
                (|p: &mut WhirBlockCompressedProof| p.pipeline = PIPELINE_WHIR)
                    as fn(&mut WhirBlockCompressedProof),
            ),
            ("version", |p| p.version = PROOF_VERSION + 1),
            ("magic", |p| p.magic[0] ^= 1),
        ] {
            let mut other = synthetic_compressed();
            tamper(&mut other);
            let bytes = other.to_bytes().expect("serializes");
            assert!(
                WhirBlockCompressedProof::from_bytes(&bytes).is_err(),
                "a compressed file with another {what} is refused"
            );
        }
        assert!(
            WhirBlockTreeProof::from_bytes(&bytes).is_err(),
            "the tree verifier refuses a compressed file"
        );
        let tree = synthetic_proof().to_bytes().expect("serializes");
        assert!(
            WhirBlockCompressedProof::from_bytes(&tree).is_err(),
            "the compressed verifier refuses the tree's file"
        );
        assert_ne!(PIPELINE_WHIR, PIPELINE_WHIR_COMPRESSED);
        // The evaluated file reads back with its own tag; the tree's reader
        // refuses it too.
        let evaluated = WhirBlockCompressedProof::for_binding(
            proof.statement().to_owned(),
            proof.compressed.clone(),
            super::super::wlfm::WlfmPrep::Evaluated,
        );
        let bytes = evaluated.to_bytes().expect("serializes");
        let back = WhirBlockCompressedProof::from_bytes(&bytes).expect("reads back");
        assert_eq!(back.pipeline(), PIPELINE_WHIR_COMPRESSED_EVALUATED);
        assert!(WhirBlockTreeProof::from_bytes(&bytes).is_err());
        assert_eq!(proof.pipeline(), PIPELINE_WHIR_COMPRESSED);
        assert_ne!(PIPELINE_WHIR_COMPRESSED, PIPELINE_WHIR_COMPRESSED_EVALUATED);
        assert_ne!(PIPELINE_WHIR, PIPELINE_WHIR_COMPRESSED_EVALUATED);
    }

    /// Each compressed verifier refuses every other compressed tag before it
    /// reads the ELF or the statement (with no ELF at all, the refusal is the
    /// tag's), and passes its own tag's gate. A lever setting no verifier
    /// takes has no tag.
    #[test]
    fn each_compressed_verifier_refuses_the_others_tag_up_front() {
        use super::super::wlfm::{WlfmLevers, WlfmPrep};
        use multilinear::whir_chain::FinalPoly;
        let stacked = synthetic_compressed();
        let six = FinalPoly::Vars(super::super::compress::COMPRESS_FINAL_VARS);
        type Verifier = fn(&[u8], &WhirBlockCompressedProof) -> Result<(), String>;
        let arms: [(WlfmLevers, u8, Verifier); 4] = [
            (
                WlfmLevers::DEFAULT,
                PIPELINE_WHIR_COMPRESSED,
                verify_whir_block_compressed_proof,
            ),
            (
                WlfmLevers::with_prep(WlfmPrep::Evaluated),
                PIPELINE_WHIR_COMPRESSED_EVALUATED,
                verify_whir_block_compressed_evaluated_proof,
            ),
            (
                WlfmLevers {
                    prep: WlfmPrep::Stacked,
                    final_poly: six,
                    fold: None,
                    grind: None,
                },
                PIPELINE_WHIR_COMPRESSED_FINAL,
                verify_whir_block_compressed_final_proof,
            ),
            (
                WlfmLevers {
                    prep: WlfmPrep::Evaluated,
                    final_poly: six,
                    fold: None,
                    grind: None,
                },
                PIPELINE_WHIR_COMPRESSED_EVALUATED_FINAL,
                verify_whir_block_compressed_evaluated_final_proof,
            ),
        ];
        let file = |levers| {
            WhirBlockCompressedProof::for_levers(
                stacked.statement().to_owned(),
                stacked.compressed.clone(),
                levers,
            )
            .expect("a tagged setting")
        };
        for (levers, tag, _) in &arms {
            let f = file(*levers);
            assert_eq!(f.pipeline(), *tag);
            let back = WhirBlockCompressedProof::from_bytes(&f.to_bytes().expect("serializes"))
                .expect("reads back");
            assert_eq!(back.pipeline(), *tag);
        }
        assert_eq!(
            WhirBlockCompressedProof::new(
                stacked.statement().to_owned(),
                stacked.compressed.clone()
            )
            .pipeline(),
            PIPELINE_WHIR_COMPRESSED
        );
        // The compact twins: tags 6–9, read back by the compact reader only,
        // each refused by every verifier but its own.
        let compact = |levers| {
            let bytes = file(levers).to_compact_bytes().expect("a compact file");
            assert!(WhirBlockCompressedProof::from_bytes(&bytes).is_err());
            WhirBlockCompressedProof::from_compact_bytes(&bytes).expect("reads back")
        };
        let plain = file(WlfmLevers::DEFAULT).to_bytes().expect("serializes");
        assert!(WhirBlockCompressedProof::from_compact_bytes(&plain).is_err());
        let compact_arms: [(WlfmLevers, u8, Verifier); 4] = [
            (
                WlfmLevers::DEFAULT,
                PIPELINE_WHIR_COMPRESSED_COMPACT,
                verify_whir_block_compressed_compact_proof,
            ),
            (
                WlfmLevers::with_prep(WlfmPrep::Evaluated),
                PIPELINE_WHIR_COMPRESSED_COMPACT_EVALUATED,
                verify_whir_block_compressed_compact_evaluated_proof,
            ),
            (
                WlfmLevers {
                    prep: WlfmPrep::Stacked,
                    final_poly: six,
                    fold: None,
                    grind: None,
                },
                PIPELINE_WHIR_COMPRESSED_COMPACT_FINAL,
                verify_whir_block_compressed_compact_final_proof,
            ),
            (
                WlfmLevers {
                    prep: WlfmPrep::Evaluated,
                    final_poly: six,
                    fold: None,
                    grind: None,
                },
                PIPELINE_WHIR_COMPRESSED_COMPACT_EVALUATED_FINAL,
                verify_whir_block_compressed_compact_evaluated_final_proof,
            ),
        ];
        let mut all: Vec<(WhirBlockCompressedProof, u8, Verifier)> = Vec::new();
        for (levers, tag, verify) in &arms {
            all.push((file(*levers), *tag, *verify));
        }
        for (levers, tag, verify) in &compact_arms {
            let f = compact(*levers);
            assert_eq!(f.pipeline(), *tag);
            assert_eq!(
                *tag,
                compressed_pipeline_of(*levers).unwrap() + PIPELINE_COMPACT_OFFSET
            );
            all.push((f, *tag, *verify));
        }
        let mut tags: std::collections::HashSet<u8> = all.iter().map(|(_, t, _)| *t).collect();
        // The byte-minimising schedule's formats, through the format-taking
        // verifier: each refuses every other file up front, and every named
        // verifier refuses theirs.
        use super::super::wire::WireFormat;
        let dedup = |levers| CompressedFormat {
            levers,
            wire: WireFormat::Dedup,
        };
        let formats = [
            (COMPRESSED_FOLDS, PIPELINE_WHIR_COMPRESSED_FOLDS),
            (
                COMPRESSED_FOLDS_EVALUATED,
                PIPELINE_WHIR_COMPRESSED_FOLDS_EVALUATED,
            ),
            (
                COMPRESSED_COMPACT_FOLDS,
                PIPELINE_WHIR_COMPRESSED_COMPACT_FOLDS,
            ),
            (
                COMPRESSED_COMPACT_FOLDS_EVALUATED,
                PIPELINE_WHIR_COMPRESSED_COMPACT_FOLDS_EVALUATED,
            ),
            (dedup(WlfmLevers::DEFAULT), PIPELINE_WHIR_COMPRESSED_DEDUP),
            (
                dedup(WlfmLevers::with_prep(WlfmPrep::Evaluated)),
                PIPELINE_WHIR_COMPRESSED_DEDUP_EVALUATED,
            ),
            (
                dedup(WlfmLevers {
                    prep: WlfmPrep::Stacked,
                    final_poly: six,
                    fold: None,
                    grind: None,
                }),
                PIPELINE_WHIR_COMPRESSED_DEDUP_FINAL,
            ),
            (
                dedup(WlfmLevers {
                    prep: WlfmPrep::Evaluated,
                    final_poly: six,
                    fold: None,
                    grind: None,
                }),
                PIPELINE_WHIR_COMPRESSED_DEDUP_EVALUATED_FINAL,
            ),
            (COMPRESSED_DEDUP_FOLDS, PIPELINE_WHIR_COMPRESSED_DEDUP_FOLDS),
            (
                COMPRESSED_DEDUP_FOLDS_EVALUATED,
                PIPELINE_WHIR_COMPRESSED_DEDUP_FOLDS_EVALUATED,
            ),
        ];
        let mut format_files = Vec::new();
        for (format, tag) in formats {
            assert_eq!(format.tag(), Ok(tag));
            let f = file(format.levers);
            let f = if format.wire == WireFormat::Rkyv {
                let bytes = f.to_bytes().expect("serializes");
                assert!(WhirBlockCompressedProof::from_compact_bytes(&bytes).is_err());
                WhirBlockCompressedProof::from_bytes(&bytes).expect("reads back")
            } else {
                let bytes = f.to_wire_bytes(format.wire).expect("a compact file");
                assert!(WhirBlockCompressedProof::from_bytes(&bytes).is_err());
                WhirBlockCompressedProof::from_compact_bytes(&bytes).expect("reads back")
            };
            assert_eq!(f.pipeline(), tag);
            assert!(tags.insert(tag), "tag {tag} repeats");
            format_files.push((format, f));
        }
        assert_eq!(tags.len(), 18, "every lever setting has its own tag");
        for (format, _) in &format_files {
            let ours = format.tag().unwrap();
            for (f, theirs, _) in &all {
                let e = verify_whir_block_compressed_as(&[], f, *format).expect_err("no ELF");
                assert!(
                    e.contains(&format!(
                        "the file holds pipeline {theirs}, and this verifier reads pipeline {ours}"
                    )),
                    "{e}"
                );
            }
            for (other, f) in &format_files {
                let e = verify_whir_block_compressed_as(&[], f, *format).expect_err("no ELF");
                assert_eq!(
                    e.contains("and this verifier reads pipeline"),
                    other != format,
                    "{e}"
                );
            }
        }
        for (_, ours, verify) in &all {
            for (_, f) in &format_files {
                let e = verify(&[], f).expect_err("no ELF");
                assert!(
                    e.contains(&format!("this verifier reads pipeline {ours}")),
                    "{e}"
                );
            }
        }
        for (_, ours, verify) in &all {
            for (f, theirs, _) in &all {
                let e = verify(&[], f).expect_err("no ELF");
                let tag_refusal = format!(
                    "the file holds pipeline {theirs}, and this verifier reads pipeline {ours}"
                );
                if theirs == ours {
                    assert!(!e.contains("and this verifier reads pipeline"), "{e}");
                } else {
                    assert!(e.contains(&tag_refusal), "{e}");
                }
            }
        }
        for (_, ours, verify) in &arms {
            for (levers, theirs, _) in &arms {
                let e = verify(&[], &file(*levers)).expect_err("no ELF");
                let tag_refusal = format!(
                    "the file holds pipeline {theirs}, and this verifier reads pipeline {ours}"
                );
                if theirs == ours {
                    assert!(!e.contains("and this verifier reads pipeline"), "{e}");
                } else {
                    assert!(e.contains(&tag_refusal), "{e}");
                }
            }
        }
        assert!(
            compressed_pipeline_of(WlfmLevers {
                prep: WlfmPrep::Stacked,
                final_poly: FinalPoly::Vars(4),
                fold: None,
                grind: None,
            })
            .is_err()
        );
    }

    /// The twice-compressed files (tags 20–25): each format's tag is its own
    /// and none of C1's; each reads back by its encoding's reader only; each
    /// C2 verifier refuses every other twice-compressed file and C1's files up
    /// front (no ELF needed); C1's verifiers refuse C2's files; a grind C2 is
    /// not built at has no tag.
    #[test]
    fn each_twice_compressed_verifier_refuses_every_other_tag_up_front() {
        use super::super::wire::WireFormat;
        use super::super::wlfm::{WlfmLevers, WlfmPrep};
        let c1 = synthetic_compressed();
        let formats = [
            (COMPRESSED2_G22, PIPELINE_WHIR_COMPRESSED2_G22),
            (
                COMPRESSED2_G22_COMPACT,
                PIPELINE_WHIR_COMPRESSED2_G22_COMPACT,
            ),
            (COMPRESSED2_G22_DEDUP, PIPELINE_WHIR_COMPRESSED2_G22_DEDUP),
            (COMPRESSED2_G26, PIPELINE_WHIR_COMPRESSED2_G26),
            (
                COMPRESSED2_G26_COMPACT,
                PIPELINE_WHIR_COMPRESSED2_G26_COMPACT,
            ),
            (COMPRESSED2_G26_DEDUP, PIPELINE_WHIR_COMPRESSED2_G26_DEDUP),
        ];
        let mut files = Vec::new();
        for (format, tag) in formats {
            assert_eq!(format.tag(), Ok(tag));
            assert!((20..=25).contains(&tag));
            let f = WhirBlockCompressedProof::for_c2(
                c1.statement().to_owned(),
                c1.compressed.clone(),
                format.grind,
            )
            .expect("a built grind");
            let f = if format.wire == WireFormat::Rkyv {
                let bytes = f.to_bytes().expect("serializes");
                assert!(WhirBlockCompressedProof::from_compact_bytes(&bytes).is_err());
                WhirBlockCompressedProof::from_bytes(&bytes).expect("reads back")
            } else {
                let bytes = f.to_wire_bytes(format.wire).expect("a compact file");
                assert!(WhirBlockCompressedProof::from_bytes(&bytes).is_err());
                WhirBlockCompressedProof::from_compact_bytes(&bytes).expect("reads back")
            };
            assert_eq!(f.pipeline(), tag);
            files.push((format, f));
        }
        // C1's files, the same proof under C1's tags.
        let c1_files: Vec<WhirBlockCompressedProof> = [
            WlfmLevers::DEFAULT,
            WlfmLevers::with_prep(WlfmPrep::Evaluated),
            super::super::compress::folds_levers(WlfmPrep::Evaluated),
        ]
        .into_iter()
        .map(|levers| {
            WhirBlockCompressedProof::for_levers(
                c1.statement().to_owned(),
                c1.compressed.clone(),
                levers,
            )
            .expect("a tagged setting")
        })
        .collect();
        for (format, _) in &files {
            let ours = format.tag().unwrap();
            for (other, f) in &files {
                let e = verify_whir_block_compressed2_as(&[], f, *format).expect_err("no ELF");
                assert_eq!(
                    e.contains("and this verifier reads pipeline"),
                    other != format,
                    "{e}"
                );
            }
            for f in &c1_files {
                let e = verify_whir_block_compressed2_as(&[], f, *format).expect_err("no ELF");
                assert!(
                    e.contains(&format!(
                        "the file holds pipeline {}, and this verifier reads pipeline {ours}",
                        f.pipeline()
                    )),
                    "{e}"
                );
            }
        }
        for (_, f) in &files {
            for verify in [
                verify_whir_block_compressed_proof as fn(&[u8], &WhirBlockCompressedProof) -> _,
                verify_whir_block_compressed_evaluated_proof,
            ] {
                let e = verify(&[], f).expect_err("no ELF");
                assert!(e.contains("and this verifier reads pipeline"), "{e}");
            }
            let e = verify_whir_block_compressed_as(&[], f, COMPRESSED_DEDUP_FOLDS_EVALUATED)
                .expect_err("no ELF");
            assert!(e.contains("and this verifier reads pipeline"), "{e}");
        }
        for grind in [18u8, 20, 24] {
            assert!(
                Compressed2Format {
                    grind,
                    wire: WireFormat::Rkyv
                }
                .tag()
                .is_err()
            );
            assert!(
                WhirBlockCompressedProof::for_c2(
                    c1.statement().to_owned(),
                    c1.compressed.clone(),
                    grind
                )
                .is_err()
            );
        }
    }

    /// The default shipped format is the default plan's file, its tag one a
    /// verifier reads, and the rollback plan ships no compressed file.
    #[test]
    fn the_default_shipped_file_is_the_default_plans() {
        use super::super::compress::{DEFAULT_PLAN, UNCOMPRESSED_PLAN};
        assert_eq!(ShippedFormat::of(DEFAULT_PLAN), Some(DEFAULT_SHIPPED));
        assert_eq!(ShippedFormat::of(UNCOMPRESSED_PLAN), None);
        let tag = DEFAULT_SHIPPED.tag().expect("a verifier reads it");
        assert!((2..=25).contains(&tag));
        let c1 = synthetic_compressed();
        let file = match DEFAULT_SHIPPED {
            ShippedFormat::C2(f) => WhirBlockCompressedProof::for_c2(
                c1.statement().to_owned(),
                c1.compressed.clone(),
                f.grind,
            ),
            ShippedFormat::C1(f) => WhirBlockCompressedProof::for_levers(
                c1.statement().to_owned(),
                c1.compressed.clone(),
                f.levers,
            ),
        }
        .expect("a tagged file");
        let bytes = file
            .to_wire_bytes(DEFAULT_SHIPPED.wire())
            .expect("the shipped bytes");
        let back = read_shipped(&bytes, DEFAULT_SHIPPED).expect("reads back");
        assert_eq!(back.pipeline(), tag);
        let e = verify_whir_block_shipped_as(&[], &back, DEFAULT_SHIPPED).expect_err("no ELF");
        assert!(!e.contains("and this verifier reads pipeline"), "{e}");
        assert!(WhirBlockTreeProof::from_bytes(&bytes).is_err());
    }

    /// The twice-compressed verifier takes C2's rate, levers and C1's from
    /// the code: its body checks the caller's tag first and reads neither the
    /// environment nor the file for a format; `verify_compressed2` derives C1
    /// at today's levers and C2 at its rate.
    #[test]
    fn the_twice_compressed_verifier_pins_its_format() {
        let src = include_str!("whir_block_tree.rs");
        let at = src
            .find("\npub fn verify_whir_block_compressed2_as(")
            .expect("the twice-compressed verifier");
        let body = &src[at..at + src[at..].find("\n}\n").expect("its end")];
        assert!(
            body.contains("require_compressed_tag(proof, format.tag()?)?"),
            "{body}"
        );
        for forbidden in [
            "env::var",
            "compress2_from_env",
            "ZfFormat::global",
            "proof.pipeline",
            ".levers",
        ] {
            assert!(
                !body.contains(forbidden),
                "the verifier reads `{forbidden}`"
            );
        }
        let src = include_str!("compress.rs");
        for (name, needles) in [
            (
                "\npub fn compressed2_artifacts(",
                &[
                    "COMPRESS_RATE, WlfmLevers::DEFAULT",
                    "build_compress2_with(&c1, plan.top_layout(), grind, true)",
                ][..],
            ),
            (
                "\nfn build_compress2_with(",
                &[
                    "COMPRESS2_RATE",
                    "compress2_levers(grind)",
                    "COMPRESS2_GRINDS",
                ][..],
            ),
        ] {
            let at = src.find(name).expect(name);
            let body = &src[at..at + src[at..].find("\n}\n").expect("its end")];
            for needle in needles {
                assert!(body.contains(needle), "{name}: {body}");
            }
            assert!(!body.contains("env::var"), "{name} reads the environment");
        }
    }

    /// The compressed block's verifier takes its rate from the code: its body
    /// names [`super::super::compress::COMPRESS_RATE`] and reads neither the
    /// environment nor the file for a format.
    #[test]
    fn the_compressed_verifier_pins_its_rate() {
        let src = include_str!("whir_block_tree.rs");
        let at = src
            .find("\npub fn verify_whir_block_compressed_proof(")
            .expect("the compressed verifier");
        let body = &src[at..at + src[at..].find("\n}\n").expect("its end")];
        assert!(body.contains("super::compress::COMPRESS_RATE"), "{body}");
        assert!(body.contains("super::compress::COMPRESS_PREP"), "{body}");
        assert!(
            body.contains("require_compressed_tag(proof, PIPELINE_WHIR_COMPRESSED)?"),
            "{body}"
        );
        let forbidden_in_verifiers = [
            "env::var",
            "compress_from_env",
            "compress_levers_from_env",
            "ZfFormat::global",
            ".rate",
            ".prep",
            ".levers",
            ".final_poly",
        ];
        for forbidden in forbidden_in_verifiers {
            assert!(
                !body.contains(forbidden),
                "the verifier reads `{forbidden}`"
            );
        }
        // The evaluated arm's verifier: the same constants but its binding,
        // named in its body.
        let at = src
            .find("\npub fn verify_whir_block_compressed_evaluated_proof(")
            .expect("the evaluated compressed verifier");
        let body = &src[at..at + src[at..].find("\n}\n").expect("its end")];
        assert!(body.contains("super::compress::COMPRESS_RATE"), "{body}");
        assert!(body.contains("WlfmPrep::Evaluated"), "{body}");
        assert!(
            body.contains("require_compressed_tag(proof, PIPELINE_WHIR_COMPRESSED_EVALUATED)?"),
            "{body}"
        );
        for forbidden in forbidden_in_verifiers {
            assert!(
                !body.contains(forbidden),
                "the evaluated verifier reads `{forbidden}`"
            );
        }
        // The final-polynomial arms: their tag first, the width a constant.
        for (name, tag, prep) in [
            (
                "verify_whir_block_compressed_final_proof",
                "PIPELINE_WHIR_COMPRESSED_FINAL",
                "super::compress::COMPRESS_PREP",
            ),
            (
                "verify_whir_block_compressed_evaluated_final_proof",
                "PIPELINE_WHIR_COMPRESSED_EVALUATED_FINAL",
                "WlfmPrep::Evaluated",
            ),
        ] {
            let at = src
                .find(&format!("\npub fn {name}("))
                .unwrap_or_else(|| panic!("{name}"));
            let body = &src[at..at + src[at..].find("\n}\n").expect("its end")];
            assert!(
                body.contains(&format!("require_compressed_tag(proof, {tag})?")),
                "{body}"
            );
            for needed in [
                "super::compress::COMPRESS_RATE",
                "super::compress::COMPRESS_FINAL_VARS",
                prep,
            ] {
                assert!(body.contains(needed), "{name} names {needed}");
            }
            for forbidden in forbidden_in_verifiers {
                assert!(!body.contains(forbidden), "{name} reads `{forbidden}`");
            }
        }
        // The compact verifiers: one shared body, each naming its tag and levers.
        for (name, tag, prep, final_poly) in [
            (
                "verify_whir_block_compressed_compact_proof",
                "PIPELINE_WHIR_COMPRESSED_COMPACT,",
                "super::compress::COMPRESS_PREP",
                "super::compress::COMPRESS_FINAL,",
            ),
            (
                "verify_whir_block_compressed_compact_evaluated_proof",
                "PIPELINE_WHIR_COMPRESSED_COMPACT_EVALUATED,",
                "WlfmPrep::Evaluated",
                "super::compress::COMPRESS_FINAL,",
            ),
            (
                "verify_whir_block_compressed_compact_final_proof",
                "PIPELINE_WHIR_COMPRESSED_COMPACT_FINAL,",
                "super::compress::COMPRESS_PREP",
                "super::compress::COMPRESS_FINAL_VARS",
            ),
            (
                "verify_whir_block_compressed_compact_evaluated_final_proof",
                "PIPELINE_WHIR_COMPRESSED_COMPACT_EVALUATED_FINAL,",
                "WlfmPrep::Evaluated",
                "super::compress::COMPRESS_FINAL_VARS",
            ),
        ] {
            let at = src
                .find(&format!("\npub fn {name}("))
                .unwrap_or_else(|| panic!("{name}"));
            let body = &src[at..at + src[at..].find("\n}\n").expect("its end")];
            for needed in ["verify_compact_file(", tag, prep, final_poly] {
                assert!(body.contains(needed), "{name} names {needed}");
            }
            for forbidden in forbidden_in_verifiers {
                assert!(!body.contains(forbidden), "{name} reads `{forbidden}`");
            }
        }
        let at = src
            .find("\nfn verify_compact_file(")
            .expect("the compact verifiers' body");
        let body = &src[at..at + src[at..].find("\n}\n").expect("its end")];
        assert!(
            body.contains("require_compressed_tag(proof, tag)?"),
            "{body}"
        );
        assert!(body.contains("super::compress::COMPRESS_RATE"), "{body}");
        for forbidden in [
            "env::var",
            "compress_from_env",
            "ZfFormat::global",
            "proof.pipeline",
        ] {
            assert!(
                !body.contains(forbidden),
                "the compact body reads `{forbidden}`"
            );
        }
        let compress = include_str!("compress.rs");
        let verify = &compress[compress
            .find("\npub fn verify_compressed(")
            .expect("verify_compressed")..];
        for forbidden in [
            "env::var",
            "compress_from_env",
            "compress_levers_from_env",
            "ZfFormat::global",
        ] {
            assert!(
                !verify.contains(forbidden),
                "compress.rs's verifiers read `{forbidden}`"
            );
        }
    }

    /// Only the production tree is at the presets: the plan's leaf count, the
    /// block fan-in and the production argue.
    #[test]
    fn only_the_production_tree_is_at_the_presets() {
        let production = WhirTreeConfig {
            leaves: None,
            siblings: 3,
            fan_in: BLOCK_FAN_IN,
            leaves_each: false,
            program_budget: super::super::program_budget::BudgetSetting::Auto,
            early_on: true,
            early_leaves: EARLY_LEAVES_DEFAULT,
            early_pool: EARLY_POOL_DEFAULT,
            leaf_emitters: Some(LEAF_EMITTERS),
            argue: BlockFormat::production().argue,
            format: BlockFormat::production(),
            options: BlockOptions::production(),
            emit_threads: 4,
            node_times: false,
            leaf_first: true,
            compress: super::super::compress::DEFAULT_PLAN.rate,
            compress_levers: super::super::compress::DEFAULT_PLAN.levers,
            compress_wire: super::super::compress::DEFAULT_PLAN.wire,
            compress2: super::super::compress::DEFAULT_PLAN.c2,
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
        // A packing is a named constant the block verifier is given too: a
        // tree argued under another (here `Off`), its format saying so, is at
        // the presets; a config whose argue and format disagree is not.
        let pack = GkrPack::Off;
        let mut c = production.clone();
        c.format = c.format.with_pack(pack);
        c.argue = c.format.argue;
        assert!(c.at_presets(), "an unpacked argue, named on both");
        let mut c = production.clone();
        c.argue = BlockFormat::production().with_pack(pack).argue;
        assert!(!c.at_presets(), "an argue its format does not carry");
        let mut c = production.clone();
        c.argue = ArgueFormat::batched(20);
        c.format.argue = c.argue;
        assert!(!c.at_presets(), "another bin cap");
        // KECCAK_RND's form has no CLI flag: a tree under the form the
        // verifier's preset does not name is not at the presets.
        let other = match BlockFormat::pinned().keccak_rnd {
            KeccakRndForm::Full => KeccakRndForm::Lean,
            KeccakRndForm::Lean => KeccakRndForm::Full,
        };
        let mut c = production.clone();
        c.format = c.format.with_keccak_rnd(other);
        assert!(!c.at_presets(), "the other KECCAK_RND form");
        // Nor has MEMW_A's.
        let other = match BlockFormat::pinned().memw_aligned {
            MemwAlignedOrder::Lt => MemwAlignedOrder::Limbs,
            MemwAlignedOrder::Limbs => MemwAlignedOrder::Lt,
        };
        let mut c = production.clone();
        c.format = c.format.with_memw_aligned(other);
        assert!(!c.at_presets(), "the other MEMW_A form");
    }

    /// `W3_LEAF_EMITTERS`: unset is the default count, 0 is the control (one
    /// thread a leaf), any other count is itself.
    #[test]
    fn the_leaf_emitters_knob_reads_a_count_and_zero_as_the_control() {
        assert_eq!(leaf_emitters_setting(None), Some(LEAF_EMITTERS));
        assert_eq!(leaf_emitters_setting(Some(0)), None);
        assert_eq!(leaf_emitters_setting(Some(1)), Some(1));
        assert_eq!(leaf_emitters_setting(Some(26)), Some(26));
        assert_eq!(LEAF_EMITTERS, 3);
    }

    /// The admitted leaves' programs are emitted at most `threads` at a time
    /// (and more than one at a time), and each is published strictly in leaf
    /// order although leaf 0 is the slowest; the leaves the budget did not
    /// admit keep an empty slot, which is how the tree knows them.
    #[test]
    fn the_admitted_leaves_are_emitted_a_few_at_a_time_and_published_in_leaf_order() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (n, upto, threads) = (9usize, 7usize, 3usize);
        let slots: Vec<Published<ProgramSlot<usize>>> = (0..n).map(|_| Published::new()).collect();
        let (active, peak) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let emit = |k: usize| -> Result<usize, String> {
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            if k == 0 {
                // Leaf 0 waits for a second emitter (a bounded wait), then is
                // the last of its batch to finish.
                let t = std::time::Instant::now();
                while active.load(Ordering::SeqCst) < 2 && t.elapsed().as_secs() < 5 {
                    std::thread::yield_now();
                }
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
            active.fetch_sub(1, Ordering::SeqCst);
            Ok(100 + k)
        };
        let mut order = Vec::new();
        let mut set_when_published = true;
        let emitted = emit_admitted_leaves(
            &slots,
            (0..upto).map(|_| None).collect(),
            threads,
            &emit,
            &|_: &usize| 0,
            &mut |k| {
                set_when_published &= slots[k].0.get().is_some();
                order.push(k);
            },
        );
        assert_eq!(emitted, Ok(()));
        assert_eq!(
            order,
            (0..upto).collect::<Vec<_>>(),
            "strictly in leaf order"
        );
        assert!(
            set_when_published,
            "a leaf's slot is set before it is announced"
        );
        let peak = peak.load(Ordering::SeqCst);
        assert!(
            peak <= threads,
            "{peak} emitters at once, the cap is {threads}"
        );
        assert!(peak >= 2, "the emitters run beside each other");
        for (k, slot) in slots.iter().enumerate().take(upto) {
            let program = slot
                .0
                .get()
                .map(|s| s.as_ref().map(|s| s.get().map(|a| a.program)));
            assert_eq!(program, Some(Ok(Ok(100 + k))));
        }
        for slot in &slots[upto..] {
            assert!(slot.0.get().is_none(), "a leaf left to the tree");
        }

        // The control: one thread a leaf.
        let slots: Vec<Published<ProgramSlot<usize>>> = (0..n).map(|_| Published::new()).collect();
        let emitted = emit_admitted_leaves(
            &slots,
            (0..n).map(|_| None).collect(),
            n,
            &|k| Ok(k),
            &|_: &usize| 0,
            &mut |_| {},
        );
        assert_eq!(emitted, Ok(()));
        assert!(slots.iter().all(|slot| slot.0.get().is_some()));
    }

    /// An emitter that fails leaves its error in its own slot and no admitted
    /// slot empty after it, so nothing waits for a program that will not come;
    /// the leaves before it keep their programs, and the error is the result.
    #[test]
    fn a_failed_leaf_emitter_leaves_no_admitted_slot_empty() {
        let (n, upto) = (8usize, 6usize);
        let slots: Vec<Published<ProgramSlot<usize>>> = (0..n).map(|_| Published::new()).collect();
        let emit = |k: usize| -> Result<usize, String> {
            if k == 2 {
                Err("leaf 2 does not emit".to_string())
            } else {
                Ok(k)
            }
        };
        let emitted = emit_admitted_leaves(
            &slots,
            (0..upto).map(|_| None).collect(),
            2,
            &emit,
            &|_: &usize| 0,
            &mut |_| {},
        );
        assert_eq!(emitted, Err("leaf 2 does not emit".to_string()));
        // Read without waiting: a slot left empty fails here, it does not hang.
        let state = |k: usize| {
            slots[k]
                .0
                .get()
                .map(|r| r.as_ref().map(|_| ()).map_err(Clone::clone))
        };
        assert_eq!(state(0), Some(Ok(())));
        assert_eq!(state(1), Some(Ok(())));
        assert_eq!(state(2), Some(Err("leaf 2 does not emit".to_string())));
        for k in 3..upto {
            assert_eq!(state(k), Some(Err("the leaf emitter stopped".to_string())));
        }
        for slot in &slots[upto..] {
            assert!(slot.0.get().is_none());
        }
    }
}
