//! The block proved with the multilinear argument in ONE proof, no epochs:
//! prove-and-retire on WHIR.
//!
//! Same executor, traces and AIRs as [`crate::multilinear_prove`], at a table
//! height the card proves today (2^21 by default); every chunk of every table
//! type is a table of the proof — including KECCAK_RND, split at 2^16 rows so
//! no argument is taller than today's — and memory is the monolithic PAGE
//! argument (no local-to-global bookend, no cross-epoch proof).
//!
//! The proof is [`stark::multilinear_block`]'s: the tables split into groups a
//! card holds ([`block_groups`], a function of the shapes and of
//! [`BlockFormat`]); phase A commits each group and lets its codewords go;
//! every root goes into the transcript and `(z, α, β)` are drawn once; phase B
//! argues and opens each group on its own fork of that transcript. The bus
//! balance is checked once, over every table of the block.
//!
//! The columns an in-guest verifier cannot evaluate — DECODE's, and the dense
//! genesis pages' — are settled by PREPARED openings ([`prepared_tables`]): one
//! stack a group, which both sides derive from the program and the partition,
//! absorbed after the groups' roots and opened on the group's fork, as an epoch
//! settles DECODE.
//!
//! ★ #1010's epoch pipeline is untouched: this is a separate entry point with a
//! statement tag of its own ([`statement::MULTILINEAR_BLOCK_TAG`]).

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use executor::elf::Elf;
use executor::vm::execution::Executor;
use math::field::element::FieldElement;
use multilinear::mle::Mle;
use multilinear::whir_chain::{ArgueFormat, ChainConfig};
use rayon::prelude::*;
use stark::multilinear_block::{self, BlockCommitted, GroupStamps, block_groups};
use stark::multilinear_table::{
    BatchedArgue, CommittedTable, MultiProof, TableLayout, TableStatement, batched::VerifierChecks,
};
use stark::table::Table;
use stark::trace::TraceTable;
use std::time::Instant;

use crate::multilinear_prove::{
    absorb_tagged, chain_config_under, layout_of, preprocessed_mles, shapes_of, stacks,
};
use crate::statement::{self, MULTILINEAR_BLOCK_TAG};
use crate::tables::gpack::{self, TraceForm};
use crate::tables::trace_builder::{
    ChunkJob, StreamTable, Traces, WindowStamps, WindowedTraceBuilder,
};
use crate::test_utils::{E, F};
use crate::zf_format::ZfFormat;
use crate::{
    Error, FIXED_TABLE_COUNT, MaxRowsConfig, ProofOptions, RuntimePageRange, TableCounts, VmAirs,
};

mod memlog;
pub mod regen;

/// How many stacked polynomials a group may take. A format constant: both sides
/// derive the groups from it ([`block_groups`]). Three is today's heaviest
/// epoch (three 2^27 polynomials, epochs 3–6 and 13 of block 25368371), so no
/// group asks the card for more than an epoch does.
pub const BLOCK_GROUP_POLYS: usize = 3;

/// The most KECCAK_RND tables a block statement may declare: a bound on what
/// the verifier builds before anything is checked, far above any block.
pub const BLOCK_MAX_KECCAK_RND: usize = 1 << 12;

/// The rows each KECCAK_RND table holds at most — a prover's choice, under
/// the format's cap [`BLOCK_KECCAK_RND_MAX_VARS`]: 2^16 is today's tallest, so
/// its argument's tree (the largest a block has) is today's.
pub const BLOCK_KECCAK_RND_ROWS_LOG2: usize = 16;

/// The most ECDAS tables a block statement may declare: a bound on what the
/// verifier builds before anything is checked, far above any block (a p99
/// mainnet block makes ≈ 15 at 2^17 rows).
pub const BLOCK_MAX_ECDAS: usize = 1 << 12;

/// The rows each ECDAS table holds at most — a prover's choice, under the
/// format's cap [`BLOCK_ECDAS_MAX_VARS`]. ECDAS is one row per double/add step
/// (≈ 382 per ECSM call), so a median mainnet block makes ≈ 420 k rows: one
/// table of 2^19 would argue on a 12 GiB tree and a p90 block's 2^20 would
/// stack into five polynomials. A call's steps may straddle two tables: they
/// chain only through the Ecdas bus, keyed by the call's timestamp and the
/// step's `(round, op)`.
pub const BLOCK_ECDAS_ROWS_LOG2: usize = 17;

/// The most KECCAK tables a block statement may declare: a bound on what the
/// verifier builds before anything is checked, far above any block (a
/// keccak-heavy block at the gas limit makes ≈ 8 at 2^18 rows).
pub const BLOCK_MAX_KECCAK: usize = 1 << 12;

/// The rows each KECCAK table holds at most — a prover's choice, under the
/// format's cap [`BLOCK_KECCAK_MAX_VARS`]. KECCAK is one row per permutation
/// call, so a cut falls between calls. A keccak-heavy block at the gas limit
/// makes ≈ 2^21 calls, whose one table would stack into eight polynomials of a
/// 2^27 stack; at 2^18 rows its 511 columns fit one.
pub const BLOCK_KECCAK_ROWS_LOG2: usize = 18;

/// The most ECSM tables a block statement may declare: a bound on what the
/// verifier builds before anything is checked, far above any block.
pub const BLOCK_MAX_ECSM: usize = 1 << 12;

/// The rows each ECSM table holds at most — a prover's choice, under the
/// format's cap [`BLOCK_ECSM_MAX_VARS`]. ECSM is one row per scalar
/// multiplication, so a cut falls between calls (a call's double/add steps are
/// ECDAS rows, reached through the Ecdas bus keyed by the call's timestamp). At
/// 2^17 rows its 667 columns fit one polynomial of a 2^27 stack.
pub const BLOCK_ECSM_ROWS_LOG2: usize = 17;

/// The tallest KECCAK_RND table a block statement may state, in variables: a
/// verifier constant ([`check_chunked_heights`]), so no statement makes the
/// argument's tree taller than a 2^16-row table's.
pub const BLOCK_KECCAK_RND_MAX_VARS: usize = 16;

/// The tallest ECDAS table a block statement may state, in variables: a
/// verifier constant ([`check_chunked_heights`]). At 2^17 rows ECDAS's argument
/// tree is ≈ 3 GiB and its columns fit one polynomial of a 2^27 stack.
pub const BLOCK_ECDAS_MAX_VARS: usize = 17;

/// The tallest KECCAK table a block statement may state, in variables: a
/// verifier constant ([`check_chunked_heights`]), so no statement makes one
/// KECCAK table stack into more than one polynomial of a 2^27 stack.
pub const BLOCK_KECCAK_MAX_VARS: usize = 18;

/// The tallest ECSM table a block statement may state, in variables: a
/// verifier constant ([`check_chunked_heights`]), so no statement makes one
/// ECSM table stack into more than one polynomial of a 2^27 stack.
pub const BLOCK_ECSM_MAX_VARS: usize = 17;

/// The tallest table a block statement may state, in variables: a verifier
/// constant ([`check_table_heights`]). A group stacks at `want.min(cap).max(tallest)`
/// variables, so one taller table would make its chain taller than the 2^27
/// stack and take its fold phase below the chains' 130.393 bits (127.39 at a
/// 2^30 table). Honest tables are at most 2^21 rows.
pub const BLOCK_MAX_TABLE_VARS: usize = 27;

const _: () = assert!(BLOCK_KECCAK_RND_ROWS_LOG2 <= BLOCK_KECCAK_RND_MAX_VARS);
const _: () = assert!(BLOCK_ECDAS_ROWS_LOG2 <= BLOCK_ECDAS_MAX_VARS);
const _: () = assert!(BLOCK_KECCAK_ROWS_LOG2 <= BLOCK_KECCAK_MAX_VARS);
const _: () = assert!(BLOCK_ECSM_ROWS_LOG2 <= BLOCK_ECSM_MAX_VARS);

/// Tree levels a group keeps OFF the host between its commit and its opening,
/// for a codeword on the host: a query re-hashes `2^4` of its codeword's cosets
/// there to rebuild them. A prover's choice; the proof does not depend on it.
pub const BLOCK_TREE_DROP_LEVELS: usize = 4;

/// The same for a codeword the card holds, whose queried blocks are re-hashed
/// on the card (`DeviceCodeword::block_subtrees`): `2^8` cosets a query, which
/// costs the card ≈ 1 ms a round and takes the kept tops to 1/16 of
/// [`BLOCK_TREE_DROP_LEVELS`]'s — 0.0059 GiB a group, 1.1 GiB at p90 (I-WFULL
/// §5). Should the card's re-hash fail, the host re-hashes them instead, ≈ 16×
/// its cost at 4, and the block reports it as a WARN line.
pub const BLOCK_TREE_DROP_LEVELS_DEVICE: usize = 8;

/// `LAMBDA_VM_BLOCK_TREE_DROP_LEVELS`: `<device>` or `<device>,<host>`, the
/// levels a retired tree drops ([`multilinear::whir_commit::TreeDrop`]);
/// unset or unreadable, [`BLOCK_TREE_DROP_LEVELS_DEVICE`] and
/// [`BLOCK_TREE_DROP_LEVELS`]. A knob a RAM-adaptive policy can set; the proof
/// does not depend on it.
pub fn tree_drop_from_env() -> multilinear::whir_commit::TreeDrop {
    parse_tree_drop(
        std::env::var("LAMBDA_VM_BLOCK_TREE_DROP_LEVELS")
            .ok()
            .as_deref(),
    )
}

fn parse_tree_drop(value: Option<&str>) -> multilinear::whir_commit::TreeDrop {
    let default = multilinear::whir_commit::TreeDrop {
        device: BLOCK_TREE_DROP_LEVELS_DEVICE,
        host: BLOCK_TREE_DROP_LEVELS,
    };
    let Some(value) = value.map(str::trim).filter(|v| !v.is_empty()) else {
        return default;
    };
    let mut parts = value.split(',').map(|p| p.trim().parse::<usize>().ok());
    match (parts.next().flatten(), parts.next(), parts.next()) {
        (Some(device), None, None) => multilinear::whir_commit::TreeDrop {
            device,
            host: default.host,
        },
        (Some(device), Some(Some(host)), None) => {
            multilinear::whir_commit::TreeDrop { device, host }
        }
        _ => default,
    }
}

/// The verifier-side constants of a block proof. [`BlockFormat::production`]
/// is what the block runs at; the tests shrink the stack so a small program
/// spans several groups.
#[derive(Clone, Copy, Debug)]
pub struct BlockFormat {
    pub zf: ZfFormat,
    /// The most stacked polynomials a group may take: the prover packs to it,
    /// and the verifier refuses a group whose stack needs more.
    pub group_polys: usize,
    /// The most groups a block statement may declare (the verifier refuses
    /// more, and the prover refuses as its groups close). The proven bits are
    /// quoted at this count (I-NOEPOCH-W.md §8).
    pub max_groups: usize,
    /// Whether DECODE's and the dense genesis pages' columns are settled by
    /// prepared openings ([`prepared_tables`]) — the production format, which
    /// the recursion needs — or evaluated by the host verifier. Off exists to
    /// measure what the openings cost the base.
    pub prepared: bool,
    /// How each group's tables argue: one argument per table
    /// ([`ArgueFormat::PerTable`], production), or one per group — a lockstep
    /// GKR per bin of its trees and one constraint sumcheck over its tables
    /// ([`ArgueFormat::Batched`], D-BATCH's format on the block). The proof
    /// carries the other format's messages in [`BlockWhirProof::argues`].
    pub argue: ArgueFormat,
}

/// The most groups a block statement may declare. The block has 9; the bound
/// leaves room for larger blocks and caps what a statement can make the
/// verifier build.
pub const BLOCK_MAX_GROUPS: usize = 64;

/// Refuses `groups` over `max_groups` ([`BlockFormat::max_groups`]). The
/// verifier refuses a statement with more ([`block_frame`]), and the prover
/// refuses as its groups close, so a block the verifier would refuse is not
/// proved.
fn check_group_count(groups: usize, max_groups: usize) -> Result<(), Error> {
    if groups > max_groups {
        return Err(Error::InvalidTableCounts(format!(
            "{groups} groups — a block takes at most {max_groups}"
        )));
    }
    Ok(())
}

impl BlockFormat {
    /// The process's WHIR format (as [`crate::multilinear_prove::chain_config`]
    /// reads it), [`BLOCK_GROUP_POLYS`], [`BLOCK_MAX_GROUPS`], and each group's
    /// tables argued together ([`ArgueFormat::BATCHED`]; Mauro 10-02: "batch the
    /// constraints"). `ArgueFormat::PerTable` stays a format to measure against.
    pub fn production() -> Self {
        Self {
            zf: *ZfFormat::global(),
            group_polys: BLOCK_GROUP_POLYS,
            max_groups: BLOCK_MAX_GROUPS,
            prepared: true,
            argue: ArgueFormat::BATCHED,
        }
    }

    /// The chain's config over `shapes`: the WHIR format's, with this format's
    /// argue. Every side of a block builds its config here.
    pub fn chain_config(&self, shapes: &[(usize, usize)]) -> ChainConfig {
        let mut config = chain_config_under(&self.zf, shapes);
        config.format.argue = self.argue;
        config
    }
}

/// The prover's own choices: none of them is in the proof.
#[derive(Clone, Debug)]
pub struct BlockOptions {
    pub max_rows: MaxRowsConfig,
    pub keccak_rnd_rows_log2: usize,
    /// ECDAS's tables are cut at 2^`ecdas_rows_log2` rows
    /// ([`BLOCK_ECDAS_ROWS_LOG2`] in production).
    pub ecdas_rows_log2: usize,
    /// KECCAK's tables are cut at 2^`keccak_rows_log2` rows
    /// ([`BLOCK_KECCAK_ROWS_LOG2`] in production).
    pub keccak_rows_log2: usize,
    /// ECSM's tables are cut at 2^`ecsm_rows_log2` rows
    /// ([`BLOCK_ECSM_ROWS_LOG2`] in production).
    pub ecsm_rows_log2: usize,
    /// The bottom levels each committed tree drops until its opening, on the
    /// card and on the host. Production: [`tree_drop_from_env`].
    pub drop_levels: multilinear::whir_commit::TreeDrop,
    /// `Some(k)`: build the traces in windows of 2^k cycles and commit each
    /// table as its chunk completes ([`WindowedTraceBuilder`]); `None`: build
    /// the whole run, then commit.
    pub window_log2: Option<usize>,
    /// With windows: stream KECCAK_RND's 2^`keccak_rnd_rows_log2`-row chunks as
    /// their ops arrive, instead of building them at the run's end. Off by
    /// default: on block 25368371 the keccak work sits at the run's end, and
    /// streaming it read NO EFFECT (FAST 418).
    pub stream_keccak_rnd: bool,
    /// With windows: derive each window's MEMW-derived LT ops as it arrives, so
    /// LT's full chunks stream with them
    /// ([`WindowedTraceBuilder::stream_memw_lt`]: other chunks than the
    /// whole-run build's, the same LT multiplicities). Off by default: the
    /// extra chunks bind the layout thread, a REGRESSION (FAST 420).
    pub stream_memw_lt: bool,
    /// With windows: the builder drops each streamed chunk's ops as the chunk
    /// leaves ([`WindowedTraceBuilder::drop_streamed_ops`]), so the build does
    /// not hold the run's op lists to its end; the traces are the same. On in
    /// production (D-MEMORY M1, FAST 501: 1× base −0.74 s, peak 47.4 → 36.6
    /// GiB, the 1.20× block proves).
    pub drop_streamed_ops: bool,
    /// With windows: `0` lays each streamed chunk out on the layout thread as
    /// it arrives; `n > 0` lays them out on `n` threads, packed in arrival
    /// order all the same (the same groups and proof bytes), while the layout
    /// thread lays out the rest of the run as soon as it is built. Production:
    /// 3. Once phase A uploads ahead, the inline layout closes the middle groups
    /// after the card is free; three workers close them in time (FAST 835:
    /// phase A −0.93 s, whole block −1.00 s, peak −0.68 GiB at 1×). Unbounded
    /// they held more host memory as the block grew (BIG 390); `layout_ahead`
    /// bounds them (BIG 392: flat).
    pub layout_workers: usize,
    /// With `layout_workers > 0`: `Some(k)` lets at most `k + 1` streamed
    /// chunks be laid out (or in the making) and not yet packed, where the
    /// inline layout holds one; `None` leaves only the channels to bound them,
    /// which BIG 390 saw grow with the block. Production `Some(2)` (D-EXEC E3).
    pub layout_ahead: Option<usize>,
    /// With `layout_workers > 0`: pack the rest of the run as it is laid out,
    /// in AIR order, so a group closes once its own tables are ready instead
    /// of after all of them; the packing order is the same. Production: off.
    /// With [`Self::rest_layout_bytes`]'s waves the groups do close earlier,
    /// but at 1× the committer is still busy when they do: on top of the waves
    /// it read +0.01 s, and alone +0.21 s (FAST 852 / 853; +0.21 s at FAST 422).
    pub pack_rest_as_laid_out: bool,
    /// How phase A holds each group's columns once committed
    /// ([`multilinear_block::Narrowing`]): production packs them narrow on the
    /// card, which leaves the proof's bytes as they are.
    pub narrow: multilinear_block::Narrowing,
    /// Phase A takes the next group and puts its columns on the card while the
    /// current group commits, when the ledger takes them
    /// ([`BlockCommitted::commit_groups`]); the commits and the proof's bytes
    /// are the same. Production: on.
    pub upload_ahead: bool,
    /// With windows: print where the host memory is, term by term, every half
    /// second and at each mark ([`memlog`]'s `BLOCK MEM` lines). A
    /// measurement; it moves no proof byte. Production reads
    /// `LAMBDA_VM_BLOCK_MEMLOG=1` (off by default).
    pub memlog: bool,
    /// With windows and KECCAK_RND not streamed: the finish builds KECCAK_RND
    /// as its 2^`keccak_rnd_rows_log2`-row tables
    /// ([`WindowedTraceBuilder::keccak_rnd_chunks_at_finish`]) instead of one
    /// table that [`split_keccak_rnd`] then copies apart; the tables are the
    /// same. Production: on (BIG 561: the split's copy is the finish's last
    /// ≈ 6 GiB at 4.13×). A cut under 32 rows (tests) takes the split.
    pub finish_keccak_rnd_chunks: bool,
    /// With windows: the finish builds KECCAK, ECSM and ECDAS as their
    /// 2^`keccak_rows_log2` / 2^`ecsm_rows_log2` / 2^`ecdas_rows_log2`-row
    /// tables ([`WindowedTraceBuilder::cuts_at_finish`]), each packed as it is
    /// built with [`Self::pack_finished`], instead of one wide table each that
    /// [`split_keccak`], [`split_ecsm`] and [`split_ecdas`] then copy apart;
    /// the tables are the same. Production: on unless
    /// `LAMBDA_VM_BLOCK_FINISH_PLAN=0` (the A arm: built whole and wide, then
    /// split). At p90 the three are the rest's only wide tables (5.09 GiB) and
    /// the rest's layout transposes them into fresh pages (+4.7 GiB VmRSS, BIG
    /// 660).
    pub finish_cuts: bool,
    /// With windows, the pipelined layout and the finish's cuts: the finish
    /// streams the rest ([`WindowedTraceBuilder::finish_streamed`]): its tables
    /// built in waves in AIR order under a byte gate of
    /// [`Self::rest_layout_bytes`] (generated and not yet placed), each family's
    /// ops freed with its last table, the layout taking each table as it comes
    /// and the packer placing it, so the rest's first groups commit while the
    /// finish still builds the later ones. The tables, their order and the
    /// groups are the same. Production: on unless `LAMBDA_VM_BLOCK_FINISH_STREAM=0`
    /// (the A arm: every table built at once, then laid out).
    pub finish_stream: bool,
    /// With windows: `Some(bytes)` lays the rest of the run out in AIR order,
    /// in waves of at most `bytes` of rows (a larger table is a wave of its
    /// own), each wave in parallel; `None` lays every table out at once. A
    /// table's column copy exists before its rows go, so laying out all of
    /// them at once holds most of the rest twice (BIG 561: ≈ 20 GiB at 4.13×,
    /// the block's high-water). The tables, their order and the groups are the
    /// same. Production: [`BLOCK_REST_LAYOUT_BYTES`].
    pub rest_layout_bytes: Option<usize>,
    /// With windows: the finish packs each table it builds at the bytes its
    /// columns need as the table is generated
    /// ([`WindowedTraceBuilder::pack_finished_tables`]), and the table is laid
    /// out narrow from that, committed from the card and never held wide on
    /// the host. KECCAK, ECSM, ECDAS and an unchunked KECCAK_RND stay wide.
    /// The proof's bytes are the same. Production: on.
    pub pack_finished: bool,
    /// G-pack (`tables::gpack`): the generators write each table packed as
    /// they generate it, with no 64-bit copy — every streamed chunk (laid out
    /// narrow from that, when the groups are held narrow) and, with
    /// [`Self::pack_finished`], every table the finish packs. The proof's bytes
    /// are the same. Production: on unless `LAMBDA_VM_BLOCK_GPACK=0` (the A
    /// arm: built at 8 bytes a cell, then laid out or packed as before).
    pub gpack: bool,
    /// Whether phase A hands the committed groups' packed tables to a spill
    /// store, which phase B reads back in group order
    /// ([`multilinear_block::BlockSpill`]). The proof's bytes are the same.
    /// Production: [`spill_from_env`] (`LAMBDA_VM_BLOCK_SPILL`), `auto` unless
    /// set.
    pub spill: BlockSpillPolicy,
    /// What phase B does with the streamed chunks ([`regen::RegenMode`]):
    /// `None` reads `LAMBDA_VM_BLOCK_REGEN` when the prove starts (unset:
    /// `off`; a value it does not know refuses the prove), `Some` is chosen.
    /// Production: [`regen::production_regen`], `auto` when that knob and
    /// `LAMBDA_VM_BLOCK_SPILL` are both unset. The proof's bytes are the same
    /// under every mode.
    pub regen: Option<regen::RegenMode>,
}

/// When a block spills its held tables ([`BlockOptions::spill`]): the policy
/// and the rule of #1013's block pipeline (`prover/src/block.rs`,
/// `SpillPolicy` … `spill_decision`, @ edddc6873), with #1014's reserve
/// ([`spill_reserve_bytes`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlockSpillPolicy {
    /// Every table stays in memory.
    Off,
    /// Every committed group past the first two spills, as far as the store's
    /// writer queue has room (a measurement arm).
    Always,
    /// Keep at most this many bytes of committed packed tables; spill the
    /// rest.
    Budget(u64),
    /// Spill once the host would pass the target ([`spill_target_bytes`]):
    /// see [`spill_wanted`]. The default: a block that fits spills nothing.
    #[default]
    Auto,
}

/// `LAMBDA_VM_BLOCK_SPILL`: `auto` (and unset) | `off` | `always` | `<GiB>` (a
/// resident budget for committed packed tables). Anything else is `off`. As
/// #1013's `parse_spill_policy`.
pub fn spill_from_env() -> BlockSpillPolicy {
    parse_spill_policy(std::env::var("LAMBDA_VM_BLOCK_SPILL").ok().as_deref())
}

fn parse_spill_policy(value: Option<&str>) -> BlockSpillPolicy {
    match value.map(str::trim) {
        Some("always") => BlockSpillPolicy::Always,
        Some("auto") => BlockSpillPolicy::Auto,
        Some(gib) => gib
            .parse::<f64>()
            .ok()
            .filter(|g| g.is_finite() && *g >= 0.0)
            .map_or(BlockSpillPolicy::Off, |g| {
                BlockSpillPolicy::Budget((g * (1u64 << 30) as f64) as u64)
            }),
        None => BlockSpillPolicy::Auto,
    }
}

/// `auto`'s target for the host: `LAMBDA_VM_BLOCK_SPILL_TARGET_GIB`, else the
/// smaller of the cgroup's memory limit (v2 or v1, [`cgroup_memory`]) and
/// `MemTotal`, less 10 GiB ([`spill_target_from`]) (#1013's
/// `spill_target_bytes` @ 035aef5d6).
pub fn spill_target_bytes() -> u64 {
    if let Some(gib) = std::env::var("LAMBDA_VM_BLOCK_SPILL_TARGET_GIB")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|g| g.is_finite() && *g > 0.0)
    {
        return (gib * (1u64 << 30) as f64) as u64;
    }
    let limit = cgroup_memory(
        &std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default(),
        std::path::Path::new(CGROUP_ROOT),
        CgroupValue::File("memory.max"),
        CgroupValue::File("memory.limit_in_bytes"),
    );
    let mem_total = std::fs::read_to_string("/proc/meminfo").ok().and_then(|m| {
        m.lines()
            .find_map(|l| l.strip_prefix("MemTotal:"))
            .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            .map(|kib| kib << 10)
    });
    spill_target_from(limit, mem_total)
}

/// The smaller of a cgroup limit and `MemTotal`, less 10 GiB; no spill
/// (`u64::MAX`) when neither is known. A v1 cgroup without a limit reads a
/// sentinel far above `MemTotal`, which the minimum discards (#1013's
/// `spill_target_from`).
fn spill_target_from(limit: Option<u64>, mem_total: Option<u64>) -> u64 {
    const MARGIN: u64 = 10 << 30;
    match (limit, mem_total) {
        (Some(limit), Some(total)) => limit.min(total),
        (Some(bytes), None) | (None, Some(bytes)) => bytes,
        (None, None) => return u64::MAX,
    }
    .saturating_sub(MARGIN)
}

/// Where the cgroup filesystem is mounted.
const CGROUP_ROOT: &str = "/sys/fs/cgroup";

/// One number of a cgroup's memory files: a file that holds just the number,
/// or one `key value` line of `memory.stat` (#1013's `CgroupValue`).
#[derive(Clone, Copy)]
enum CgroupValue<'a> {
    File(&'a str),
    Stat(&'a str),
}

impl CgroupValue<'_> {
    fn read(self, dir: &std::path::Path) -> Option<u64> {
        let (file, key) = match self {
            CgroupValue::File(file) => (file, None),
            CgroupValue::Stat(key) => ("memory.stat", Some(key)),
        };
        let text = std::fs::read_to_string(dir.join(file)).ok()?;
        match key {
            None => text.trim().parse::<u64>().ok(),
            Some(key) => text.lines().find_map(|l| {
                let (k, v) = l.split_once(' ')?;
                (k == key).then(|| v.trim().parse::<u64>().ok()).flatten()
            }),
        }
    }
}

/// A number from this process's cgroup memory files: `v2` under the unified
/// hierarchy (`memory.max`, `memory.current`, `memory.stat`'s `inactive_file`),
/// else `v1` under the memory controller (`memory.limit_in_bytes`,
/// `memory.usage_in_bytes`, `memory.stat`'s `total_inactive_file`).
/// `proc_cgroup` is `/proc/self/cgroup`, `root` the cgroup mount. Each is read
/// at the process's cgroup path, then at its hierarchy's root, which a
/// container without a cgroup namespace sees as its own cgroup. `None` where
/// neither reads as a number (v2's `max` included) (#1013's
/// `cgroup_memory`).
fn cgroup_memory(
    proc_cgroup: &str,
    root: &std::path::Path,
    v2: CgroupValue,
    v1: CgroupValue,
) -> Option<u64> {
    let at = |base: &std::path::Path, path: &str, value: CgroupValue| {
        value
            .read(&base.join(path.trim().trim_start_matches('/')))
            .or_else(|| value.read(base))
    };
    let (mut unified, mut memory) = (None, None);
    for line in proc_cgroup.lines() {
        let mut fields = line.splitn(3, ':');
        let (Some(id), Some(controllers), Some(path)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        if id == "0" && controllers.is_empty() {
            unified = Some(path);
        } else if controllers.split(',').any(|c| c == "memory") {
            memory = Some(path);
        }
    }
    unified
        .and_then(|path| at(root, path, v2))
        .or_else(|| memory.and_then(|path| at(&root.join("memory"), path, v1)))
}

/// What `auto` reads of the host ([`HostReading::bytes`]) (#1013's
/// `HostReading` @ 035aef5d6, from cf253d235).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct HostReading {
    /// The process's peak resident set (`VmHWM`).
    hwm: u64,
    /// The cgroup's charge (`memory.current`, or v1's `memory.usage_in_bytes`),
    /// page cache included.
    charged: u64,
    /// Its inactive file pages (`memory.stat`'s `inactive_file`, or v1's
    /// `total_inactive_file`): the page cache the kernel reclaims first.
    inactive_file: u64,
}

impl HostReading {
    fn now() -> Self {
        let hwm = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find_map(|l| l.strip_prefix("VmHWM:"))
                    .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
                    .map(|kib| kib << 10)
            });
        let proc_cgroup = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
        let root = std::path::Path::new(CGROUP_ROOT);
        let charged = cgroup_memory(
            &proc_cgroup,
            root,
            CgroupValue::File("memory.current"),
            CgroupValue::File("memory.usage_in_bytes"),
        );
        let inactive_file = cgroup_memory(
            &proc_cgroup,
            root,
            CgroupValue::Stat("inactive_file"),
            CgroupValue::Stat("total_inactive_file"),
        );
        Self {
            hwm: hwm.unwrap_or(0),
            charged: charged.unwrap_or(0),
            inactive_file: inactive_file.unwrap_or(0),
        }
    }

    /// The host's bytes: the larger of the peak resident set (the resident set
    /// itself is not monotone under the allocator's posture, which drops and
    /// re-faults recycled extents) and the cgroup's working set, its charge
    /// less its inactive file pages (the kubelet's measure). Page cache the
    /// kernel would give back first is not counted; active file pages are,
    /// and the target's 10 GiB margin covers them.
    fn bytes(&self) -> u64 {
        self.hwm
            .max(self.charged.saturating_sub(self.inactive_file))
    }
}

/// `auto`'s reserve beside the host's bytes: what the block still needs once
/// the tables committed so far (`cells` main cells) are on the host. #1014's
/// own, inferred from the median block (BIG 565, 33.5 G cells): the finish's
/// p5 transient (25.5 GiB at its peak) and the tree's leaf programs, which
/// live through phase B (16.3 GiB at its end), ≈ 1.25 GiB per total G cells,
/// or ≈ 1.65 per G cells committed so far (the streamed share ≈ 0.76); plus
/// 6 GiB for phase B's bump and the read-back window.
fn spill_reserve_bytes(cells: u64) -> u64 {
    const PER_G: f64 = 1.65 * (1u64 << 30) as f64;
    (cells as f64 / 1e9 * PER_G) as u64 + (6 << 30)
}

/// The share of `auto`'s target ([`spill_target_bytes`]) past which the block's
/// memory is short for the allocator's purge ([`crate::alloc_purge`]): 85 %, a
/// share so it follows the host's RAM.
const PRESSURE_SHARE: (u64, u64) = (17, 20);

/// Notes memory pressure for [`crate::alloc_purge`] once the host's bytes
/// ([`HostReading::bytes`]) pass [`PRESSURE_SHARE`] of `target`, and says
/// whether they did.
///
/// ★ THE ONE WAY #1014'S PURGE DIFFERS FROM #1013'S. The module, its knob
/// (`LAMBDA_VM_ALLOC_PURGE`, `auto` by default) and its points are #1013's;
/// #1013 notes the pressure when its spill's queue budgets arm, and #1014's
/// spill has no budgets, so it reads the host where the spill decides (each
/// committed table) and once more at phase A's end. A block whose host stays
/// under the share (the median: at most 86.6 GiB against 94.1 on BIG) never
/// purges.
fn note_pressure_past(host: u64, target: u64) -> bool {
    let short = memory_short(host, target);
    if short {
        crate::alloc_purge::note_memory_pressure();
    }
    short
}

/// Whether `host` bytes are past [`PRESSURE_SHARE`] of `target`.
fn memory_short(host: u64, target: u64) -> bool {
    let (num, den) = PRESSURE_SHARE;
    host > target / den * num
}

/// The finish's growth of the resident set, from the walk's end to phase A's
/// peak, per walked cycle: BIG 660's p90 (D, at defaults) went 85.19 → 115.27
/// GiB over 612.3 M cycles (I-WFULL §4.1). The median (653) grew 12.41 GiB over
/// 298.4 M and full gas (660 D) 18.06 over 602.9 M, both under it.
const FINISH_GROWTH_PER_CYCLE: f64 = 30.08 * (1u64 << 30) as f64 / 612.3e6;

/// What a block hands back to its spill at the walk's end
/// ([`multilinear_block::BlockSpill::hand_off`]), decided once there: the
/// forecast of phase A's peak, `hwm` (the resident high-water so far) plus
/// [`FINISH_GROWTH_PER_CYCLE`] for each of the `cycles` walked, past
/// [`PRESSURE_SHARE`] of `target`; 0 when the forecast fits under it. The
/// finish is still ahead, so the parked tables leave before its hump builds,
/// and the hump reuses their pages.
fn hand_off_need(hwm: u64, cycles: u64, target: u64) -> u64 {
    let (num, den) = PRESSURE_SHARE;
    let forecast = hwm.saturating_add((FINISH_GROWTH_PER_CYCLE * cycles as f64) as u64);
    forecast.saturating_sub(target / den * num)
}

/// `LAMBDA_VM_BLOCK_HAND_OFF`: `auto` (and unset) decides the hand-off at the
/// walk's end ([`hand_off_need`]); `off` never hands off (the parked tables
/// stay on the host, as `auto` kept them). Anything else is `auto`.
fn hand_off_from_env() -> bool {
    parse_hand_off(std::env::var("LAMBDA_VM_BLOCK_HAND_OFF").ok().as_deref())
}

fn parse_hand_off(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("off")
}

/// The policy's choice for one committed table of `bytes` packed bytes:
/// `kept` the committed packed bytes kept so far, `cells` the main cells
/// committed so far (this table's included), `host` the host's bytes
/// ([`HostReading::bytes`]), read only by `auto` (#1013's `spill_decision`, with
/// #1014's reserve).
fn spill_wanted(
    policy: BlockSpillPolicy,
    target: u64,
    kept: u64,
    cells: u64,
    bytes: u64,
    host: impl FnOnce() -> u64,
) -> bool {
    match policy {
        BlockSpillPolicy::Off => false,
        BlockSpillPolicy::Always => true,
        BlockSpillPolicy::Budget(budget) => kept + bytes > budget,
        BlockSpillPolicy::Auto => {
            host()
                .saturating_add(spill_reserve_bytes(cells))
                .saturating_add(bytes)
                > target
        }
    }
}

impl BlockOptions {
    /// Every chunked table at 2^21 rows, KECCAK_RND at 2^16, ECDAS at 2^17,
    /// KECCAK at 2^18, ECSM at 2^17, windows of 2^20 cycles.
    pub fn production() -> Self {
        Self {
            max_rows: MaxRowsConfig::uniform(1 << 21),
            keccak_rnd_rows_log2: BLOCK_KECCAK_RND_ROWS_LOG2,
            ecdas_rows_log2: BLOCK_ECDAS_ROWS_LOG2,
            keccak_rows_log2: BLOCK_KECCAK_ROWS_LOG2,
            ecsm_rows_log2: BLOCK_ECSM_ROWS_LOG2,
            drop_levels: tree_drop_from_env(),
            window_log2: Some(BLOCK_WINDOW_LOG2),
            stream_keccak_rnd: false,
            stream_memw_lt: false,
            drop_streamed_ops: true,
            layout_workers: 3,
            layout_ahead: Some(2),
            pack_rest_as_laid_out: false,
            narrow: multilinear_block::Narrowing::CARD,
            upload_ahead: true,
            memlog: memlog::from_env(),
            finish_keccak_rnd_chunks: true,
            finish_cuts: finish_plan_from_env(),
            finish_stream: finish_stream_from_env(),
            rest_layout_bytes: Some(BLOCK_REST_LAYOUT_BYTES),
            pack_finished: true,
            gpack: gpack_from_env(),
            spill: spill_from_env(),
            regen: regen::production_regen(),
        }
    }
}

/// `LAMBDA_VM_BLOCK_GPACK`: [`BlockOptions::gpack`]'s production value, on
/// unless `0`.
pub(crate) fn gpack_from_env() -> bool {
    !std::env::var("LAMBDA_VM_BLOCK_GPACK").is_ok_and(|v| v.trim() == "0")
}

/// `LAMBDA_VM_BLOCK_FINISH_PLAN`: [`BlockOptions::finish_cuts`]'s production
/// value, on unless `0`.
pub(crate) fn finish_plan_from_env() -> bool {
    !std::env::var("LAMBDA_VM_BLOCK_FINISH_PLAN").is_ok_and(|v| v.trim() == "0")
}

/// `LAMBDA_VM_BLOCK_FINISH_STREAM`: [`BlockOptions::finish_stream`]'s
/// production value, on unless `0`.
pub(crate) fn finish_stream_from_env() -> bool {
    !std::env::var("LAMBDA_VM_BLOCK_FINISH_STREAM").is_ok_and(|v| v.trim() == "0")
}

/// `LAMBDA_VM_BLOCK_COMPACT_LT=0`: the LT ops derived from the MEMW ops the
/// builder drops are held at 24 bytes each until the finish
/// ([`WindowedTraceBuilder::raw_memw_lt`]), the A arm of keeping them compact;
/// unset or anything else keeps them compact. The tables are the same (#1013's
/// knob).
fn compact_lt() -> bool {
    !std::env::var("LAMBDA_VM_BLOCK_COMPACT_LT").is_ok_and(|v| v.trim() == "0")
}

/// `LAMBDA_VM_BLOCK_LT_CONCAT=1`: the finish concatenates LT's ops into one
/// list before chunking them ([`WindowedTraceBuilder::concat_lt`]), the A arm
/// of keeping them as segments; off by default. With
/// `LAMBDA_VM_BLOCK_COMPACT_LT=0` it is the finish as it was before both. The
/// tables are the same (#1013's knob).
fn lt_concat() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_LT_CONCAT").is_ok_and(|v| v.trim() == "1")
}

/// The rows the rest's layout transposes at once ([`BlockOptions::rest_layout_bytes`]):
/// each transposition is parallel within its table, so a wave of a few
/// tables keeps the cores busy while bounding the copies in flight.
pub const BLOCK_REST_LAYOUT_BYTES: usize = 2 << 30;

/// `BLOCK_WHIR_REST_LAYOUT=all|<MiB>`, `BLOCK_WHIR_KR_FINISH_CHUNKS=0|1` and
/// `BLOCK_WHIR_PACK_FINISHED=0|1`: the real-block tests' choice of
/// [`BlockOptions::rest_layout_bytes`], [`BlockOptions::finish_keccak_rnd_chunks`]
/// and [`BlockOptions::pack_finished`]; unset leaves the production ones.
pub(crate) fn rest_layout_from_env(options: &mut BlockOptions) {
    match std::env::var("BLOCK_WHIR_REST_LAYOUT")
        .as_deref()
        .map(str::trim)
    {
        Ok("all") => options.rest_layout_bytes = None,
        Ok(mib) => match mib.parse::<usize>() {
            Ok(mib) if mib > 0 => options.rest_layout_bytes = Some(mib << 20),
            _ => panic!("BLOCK_WHIR_REST_LAYOUT={mib}: all or a positive MiB count"),
        },
        Err(_) => {}
    }
    match std::env::var("BLOCK_WHIR_KR_FINISH_CHUNKS")
        .as_deref()
        .map(str::trim)
    {
        Ok("0") => options.finish_keccak_rnd_chunks = false,
        Ok("1") => options.finish_keccak_rnd_chunks = true,
        Ok(other) => panic!("BLOCK_WHIR_KR_FINISH_CHUNKS={other}: 0 or 1"),
        Err(_) => {}
    }
    match std::env::var("BLOCK_WHIR_PACK_FINISHED")
        .as_deref()
        .map(str::trim)
    {
        Ok("0") => options.pack_finished = false,
        Ok("1") => options.pack_finished = true,
        Ok(other) => panic!("BLOCK_WHIR_PACK_FINISHED={other}: 0 or 1"),
        Err(_) => {}
    }
}

/// `BLOCK_WHIR_NARROW=wide|card|host`: the real-block tests' choice of
/// [`BlockOptions::narrow`]; `None` leaves the production one.
pub(crate) fn narrow_from_env() -> Option<multilinear_block::Narrowing> {
    use multilinear_block::Narrowing;
    match std::env::var("BLOCK_WHIR_NARROW").as_deref().map(str::trim) {
        Ok("wide") => Some(Narrowing::Wide),
        Ok("card") => Some(Narrowing::CARD),
        Ok("host") => Some(Narrowing::Host),
        Ok(other) => panic!("BLOCK_WHIR_NARROW={other}: wide, card or host"),
        Err(_) => None,
    }
}

/// `BLOCK_WHIR_UPLOAD_AHEAD=0|1`: the real-block tests' choice of
/// [`BlockOptions::upload_ahead`]; `None` leaves the production one.
pub(crate) fn upload_ahead_from_env() -> Option<bool> {
    match std::env::var("BLOCK_WHIR_UPLOAD_AHEAD")
        .as_deref()
        .map(str::trim)
    {
        Ok("0") => Some(false),
        Ok("1") => Some(true),
        Ok(other) => panic!("BLOCK_WHIR_UPLOAD_AHEAD={other}: 0 or 1"),
        Err(_) => None,
    }
}

/// `BLOCK_WHIR_KECCAK_LOG2=k` (0..=[`BLOCK_KECCAK_MAX_VARS`]) and
/// `BLOCK_WHIR_ECSM_LOG2=k` (0..=[`BLOCK_ECSM_MAX_VARS`]): the real-block
/// tests' KECCAK and ECSM cuts, to force a split on a block whose tables fit
/// one; unset leaves the production ones.
pub(crate) fn chunk_cuts_from_env(options: &mut BlockOptions) {
    for (name, cap, rows_log2) in [
        (
            "BLOCK_WHIR_KECCAK_LOG2",
            BLOCK_KECCAK_MAX_VARS,
            &mut options.keccak_rows_log2,
        ),
        (
            "BLOCK_WHIR_ECSM_LOG2",
            BLOCK_ECSM_MAX_VARS,
            &mut options.ecsm_rows_log2,
        ),
    ] {
        if let Ok(v) = std::env::var(name) {
            match v.trim().parse() {
                Ok(k) if k <= cap => *rows_log2 = k,
                _ => panic!("{name}={v}: 0..={cap}"),
            }
        }
    }
}

/// The chunked tables' heights a statement states, for the real-block
/// readouts: `KECCAK 1 × [14] · KECCAK_RND 4 × [16, 16, 16, 16] · …`.
pub(crate) fn chunked_census(counts: &TableCounts, table_num_vars: &[u8]) -> String {
    chunked_table_ranges(counts)
        .into_iter()
        .map(|(name, range, _)| {
            let heights = &table_num_vars[range];
            format!("{name} {} × {heights:?}", heights.len())
        })
        .collect::<Vec<_>>()
        .join(" · ")
}

/// Cycles a streamed prove collects at a time: half a CPU instance at 2^21, so
/// every chunk is handed to phase A within a window of completing.
pub const BLOCK_WINDOW_LOG2: usize = 20;

/// A block proved in one proof.
///
/// The statement is [`crate::multilinear_prove::MultilinearVmProof`]'s; the
/// proof is read under the block's forked schedule, and only by
/// [`verify_block_whir`].
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BlockWhirProof {
    pub proof: MultiProof<F, E>,
    /// Each table's height in variables, in [`VmAirs::air_refs`] order.
    pub table_num_vars: Vec<u8>,
    pub runtime_page_ranges: Vec<RuntimePageRange>,
    pub table_counts: TableCounts,
    pub public_output: Vec<u8>,
    pub num_private_input_pages: usize,
    /// The groups, each its tables' indices in [`VmAirs::air_refs`] order, in
    /// the order the group stacks them. The prover's choice — a streamed prove
    /// packs tables as they arrive — bound in the statement before any root
    /// ([`absorb_block`]); the verifier checks it is a partition and rebuilds
    /// each group's stack from it under its own stack cap. The proof's tables
    /// are in this order.
    pub groups: Vec<Vec<u32>>,
    /// The prepared openings ([`group_stacks`]), one per group that holds a
    /// prepared table, in group order.
    pub prepared: Vec<multilinear::stacked_eval::StackedProof<F, E>>,
    /// Under [`ArgueFormat::Batched`], each group's batched argue, in group
    /// order, and `proof.tables` is empty; under [`ArgueFormat::PerTable`],
    /// empty. Which one a verifier reads is its format's, never the proof's.
    pub argues: Vec<BatchedArgue<E>>,
}

/// Where a block's prove spent its time, for the readout. Seconds.
#[derive(Clone, Debug, Default)]
pub struct BlockStamps {
    pub execute: f64,
    pub build: f64,
    /// KECCAK_RND split, AIRs, transposition and layouts.
    pub prep: f64,
    pub phase_a: f64,
    pub phase_b: f64,
    pub tables: usize,
    pub cells: usize,
    pub groups: Vec<GroupStamps>,
    /// The trace build's phase ends and each phase-5 generator's finish,
    /// seconds into the build.
    pub build_marks: Vec<(String, f64)>,
    /// Phase B's first-round trees: `open_many` calls served from the kept
    /// tops and the leaves re-hashed for them. A revived commitment never
    /// builds a device tree, so nothing re-commits.
    pub top_paths: (u64, u64),
    /// Of those, for codewords on the card: `(calls, seconds)` re-hashed on
    /// the card, and on the host (the fallback).
    pub top_routes: ((u64, f64), (u64, f64)),
    /// The levels the retired trees dropped (on the card · on the host).
    pub drop_levels: Option<multilinear::whir_commit::TreeDrop>,
    /// Phase B's openings, by host stage (`LAMBDA_VM_BASE_SPLIT=1`, else
    /// empty): the WHIR chains' six stages, their query split, and the kept-top
    /// paths' gather and re-hash, seconds on the prover's thread.
    pub open_split: Vec<(&'static str, f64)>,
    /// A streamed build: when its windows were all collected (seconds since
    /// the build started) and how many chunks they handed out.
    pub streamed: (f64, usize),
    /// A streamed build's windows: the walk, the routing and the streamed
    /// chunks' generation, summed.
    pub windows: WindowStamps,
    /// The prepared commitments: how many, and the seconds deriving them.
    pub prepared: (usize, f64),
    /// A streamed build's layout: its threads, its marks (seconds since the
    /// prove started), when each group closed, the seconds the packer waited
    /// on phase A and the seconds laying out chunks, summed over threads.
    pub layout: LayoutStamps,
    /// How phase A held the committed columns, and the narrow tables phase B
    /// widened on the host, `(tables, cells)`: a device path widens on the
    /// card.
    pub narrow: multilinear_block::Narrowing,
    pub host_widens: (u64, u64),
    /// With [`BlockOptions::memlog`]: every memory term when the proof is
    /// done, bytes; empty without.
    pub mem_terms: Vec<(&'static str, usize)>,
    /// With [`BlockOptions::spill`]: the store's counters and the read-back's
    /// report, or why no store opened.
    pub spill: Option<String>,
    /// With [`BlockOptions::spill`]: the store's counters after phase B.
    pub spill_stats: Option<stark::spill::SpillStats>,
    /// With [`BlockOptions::regen`] set to regenerate: what the regenerator
    /// did, and its `BLOCK REGEN` lines (printed as phase B ends, not in
    /// [`Self::report`]).
    pub regen: Option<regen::RegenStamps>,
}

/// A streamed build's layout, for the readout ([`BlockStamps::layout`]).
#[derive(Clone, Debug, Default)]
pub struct LayoutStamps {
    pub workers: usize,
    pub marks: Vec<(&'static str, f64)>,
    pub closed_at: Vec<f64>,
    pub blocked: f64,
    pub chunks: f64,
    /// The rest's five slowest tables to lay out and its first in AIR order.
    pub slowest: Vec<(String, f64)>,
    /// The most streamed chunks laid out (or in the making) and not yet
    /// packed at once, under [`BlockOptions::layout_ahead`].
    pub ahead_most: usize,
    /// The rest's tables laid out narrow from the finish's packed columns
    /// ([`BlockOptions::pack_finished`]): how many, and their packed bytes.
    pub packed_rest: (usize, usize),
    /// Each table of the rest in AIR order: its name, when its layout ended
    /// (seconds since the prove started) and its group.
    pub rest_tables: Vec<(String, f64, usize)>,
    /// G-pack ([`BlockOptions::gpack`]): whether it was on, and its counts
    /// (`tables::gpack::counts`) at phase A's end.
    pub gpack: (bool, [usize; 4]),
}

impl BlockStamps {
    /// One line a group and a total, `BLOCK …` prefixed, for a box log.
    pub fn report(&self) -> String {
        let sum = |f: fn(&GroupStamps) -> f64| self.groups.iter().map(f).sum::<f64>();
        let polys: usize = self.groups.iter().map(|g| g.polys).sum();
        let tree: usize = self.groups.iter().map(|g| g.tree_bytes).sum();
        let mut out = format!(
            "BLOCK SHAPE: tables {} · groups {} · chains {} · cells {:.3} G · tree tops {:.2} GiB\n",
            self.tables,
            self.groups.len(),
            polys,
            self.cells as f64 / 1e9,
            tree as f64 / (1u64 << 30) as f64,
        );
        for (g, s) in self.groups.iter().enumerate() {
            out.push_str(&format!(
                "BLOCK GROUP {g}: tables {} · polys {} · cells {:.1} M || A wait {:.3} upload {:.3} paid {:.3} commit {:.3} retire {:.3} from@{:.2} done@{:.2} || B upload {:.3} argue {:.3} encode {:.3} open {:.3}\n",
                s.tables,
                s.polys,
                s.cells as f64 / 1e6,
                s.wait_a,
                s.upload_a,
                s.upload_paid,
                s.commit,
                s.retire,
                s.committed_at - s.commit - s.retire,
                s.committed_at,
                s.upload_b,
                s.argue,
                s.encode,
                s.open,
            ));
        }
        if !self.build_marks.is_empty() {
            let marks: Vec<String> = self
                .build_marks
                .iter()
                .map(|(label, at)| format!("{label} {at:.2}"))
                .collect();
            out.push_str(&format!("BLOCK BUILD MARKS: {}\n", marks.join(" · ")));
        }
        if self.streamed.1 > 0 || self.streamed.0 > 0.0 {
            out.push_str(&format!(
                "BLOCK STREAM: windows collected at {:.2}s · run built at {:.2}s · {} chunks streamed · layout busy {:.2}s · phase A ended at {:.2}s (all since the streamed prove started, execution included)\n",
                self.streamed.0, self.build, self.streamed.1, self.prep, self.phase_a,
            ));
            out.push_str(&format!(
                "BLOCK WINDOWS: {} windows · walk {:.2}s (walker thread) · route+append {:.2}s · chunk handout {:.2}s (accumulator thread) · finish {:.2}s\n",
                self.windows.windows,
                self.windows.walk,
                self.windows.route,
                self.windows.generate,
                self.build - self.streamed.0,
            ));
            let l = &self.layout;
            let marks: Vec<String> = l
                .marks
                .iter()
                .map(|(label, at)| format!("{label} {at:.2}"))
                .collect();
            let closed: Vec<String> = l.closed_at.iter().map(|at| format!("{at:.2}")).collect();
            out.push_str(&format!(
                "BLOCK LAYOUT: {} worker(s), at most {} chunk(s) unpacked · {} · groups closed at [{}] · packer waited on phase A {:.2}s · chunks laid out {:.2}s (summed)\n",
                l.workers,
                if l.ahead_most == 0 && l.workers > 0 {
                    "unbounded".to_string()
                } else {
                    l.ahead_most.to_string()
                },
                marks.join(" · "),
                closed.join(", "),
                l.blocked,
                l.chunks,
            ));
            let slowest: Vec<String> = l
                .slowest
                .iter()
                .map(|(name, secs)| format!("{name} {secs:.3}"))
                .collect();
            out.push_str(&format!(
                "BLOCK REST LAYOUT: slowest tables (s) {}\n",
                slowest.join(" · ")
            ));
            out.push_str(&format!(
                "BLOCK REST PACKED: {} tables laid out narrow from the finish, {:.2} GiB\n",
                l.packed_rest.0,
                l.packed_rest.1 as f64 / (1u64 << 30) as f64,
            ));
            let (on, [direct, narrowed, again, wide]) = l.gpack;
            out.push_str(&format!(
                "BLOCK GPACK: G-pack {}: {direct} written packed, {narrowed} narrowed after, {again} \
                 written again, {wide} built wide then packed\n",
                if on { "on" } else { "off" },
            ));
            let tables: Vec<String> = l
                .rest_tables
                .iter()
                .map(|(name, at, group)| format!("{name}@{at:.2}·g{group}"))
                .collect();
            out.push_str(&format!(
                "BLOCK REST TABLES (laid out at · group): {}\n",
                tables.join(" ")
            ));
        }
        let ((card_calls, card_secs), (host_calls, host_secs)) = self.top_routes;
        out.push_str(&format!(
            "BLOCK RECOMMIT: 0 (a revived commitment builds no tree) · first-round paths from kept tops: {} calls · {} leaves re-hashed · on the card {card_calls} calls in {card_secs:.3} s · on the host (fallback) {host_calls} calls in {host_secs:.3} s · dropped {} · phase B on one thread + one upload helper\n",
            self.top_paths.0,
            self.top_paths.1,
            self.drop_levels
                .map_or("-".to_string(), |d| d.to_string()),
        ));
        if host_calls > 0 {
            out.push_str(&format!(
                "BLOCK WARN: {host_calls} first-round path calls of codewords on the card were re-hashed on the host (the fallback; LAMBDA_VM_TOP_PATHS_HOST or a failed card re-hash): {host_secs:.2} s\n"
            ));
        }
        let reserved: Vec<String> = self
            .groups
            .iter()
            .map(|g| format!("{}", g.argue_reserved >> 20))
            .collect();
        out.push_str(&format!(
            "BLOCK ARGUE HW: max {} MiB · per group [{}] (the ledger's peak through each argue, the next group's upload included)\n",
            self.groups
                .iter()
                .map(|g| g.argue_reserved >> 20)
                .max()
                .unwrap_or(0),
            reserved.join(", "),
        ));
        // Phase B's ledger, for an overlap of group g's openings with group
        // g + 1's argue: what the two would hold at once, against the budget.
        let mib = |b: u64| b >> 20;
        let per_group: Vec<String> = self
            .groups
            .iter()
            .map(|g| {
                format!(
                    "{}/{}/{}/{}",
                    mib(g.argue_base),
                    mib(g.argue_reserved),
                    mib(g.open_room),
                    mib(g.open_reserved)
                )
            })
            .collect();
        let overlap: Vec<String> = self
            .groups
            .windows(2)
            .map(|w| {
                let both = w[0].open_reserved + w[1].argue_reserved.saturating_sub(w[1].argue_base);
                format!("{}", mib(both))
            })
            .collect();
        out.push_str(&format!(
            "BLOCK PHASE-B LEDGER: budget {} MiB · per group base/argue/room/open [{}] MiB · open(g) beside argue(g+1) [{}] MiB · revive room refusals {} · turn refusals {} · GKR tree refusals {} · openings on the host {}\n",
            mib(multilinear::gpu::reserve_budget()),
            per_group.join(", "),
            overlap.join(", "),
            multilinear::gpu::room_revive_refusals(),
            multilinear::gpu::room_turn_refusals(),
            multilinear::gpu::gkr_tree_refusals(),
            multilinear::gpu::open_host_fallbacks(),
        ));
        // Phase A's ledger and its upload: what each group's upload took, what
        // the committer paid of it, and what the commit promised beside it.
        let hidden = sum(|g| g.upload_a - g.upload_paid);
        let uploaded = sum(|g| g.upload_a);
        let per_group: Vec<String> = self
            .groups
            .iter()
            .map(|g| format!("{}/{}", mib(g.commit_base), mib(g.commit_reserved)))
            .collect();
        out.push_str(&format!(
            "BLOCK PHASE-A LEDGER: budget {} MiB · per group base/peak [{}] MiB · upload {:.3}s, paid {:.3}s, hidden {:.3}s ({:.0} %) · stores refused beside a commit {} · commit room refusals {}\n",
            mib(multilinear::gpu::reserve_budget()),
            per_group.join(", "),
            uploaded,
            sum(|g| g.upload_paid),
            hidden,
            if uploaded > 0.0 { 100.0 * hidden / uploaded } else { 0.0 },
            self.groups.iter().filter(|g| g.ahead_refused).count(),
            multilinear::gpu::room_commit_refusals(),
        ));
        if !self.open_split.is_empty() {
            let parts: Vec<String> = self
                .open_split
                .iter()
                .map(|(name, v)| format!("{name} {v:.3}"))
                .collect();
            out.push_str(&format!(
                "BLOCK OPEN SPLIT (host s on the prover's thread): {}\n",
                parts.join(" · ")
            ));
        }
        let argue = sum(|g| g.argue);
        let open = sum(|g| g.open);
        let tax = sum(|g| g.upload_b + g.encode);
        out.push_str(&format!(
            "BLOCK PREPARED: tables {} · derived in {:.3}\n",
            self.prepared.0, self.prepared.1
        ));
        let packed_cells: usize = self.groups.iter().map(|g| g.packed_cells).sum();
        let packed_bytes: usize = self.groups.iter().map(|g| g.packed_bytes).sum();
        let gib = |bytes: usize| bytes as f64 / (1u64 << 30) as f64;
        out.push_str(&format!(
            "BLOCK NARROW: {:?} · tables packed {} of {} · cells {:.3} of {:.3} G · {:.2} → {:.2} GiB ({:.2} B/cell) · pack waited {:.3}s, busy {:.3}s (phase A) · host widens {} tables, {:.3} G cells\n",
            self.narrow,
            self.groups.iter().map(|g| g.packed_tables).sum::<usize>(),
            self.tables,
            packed_cells as f64 / 1e9,
            self.cells as f64 / 1e9,
            gib(packed_cells * 8),
            gib(packed_bytes),
            if packed_cells == 0 {
                0.0
            } else {
                packed_bytes as f64 / packed_cells as f64
            },
            sum(|g| g.pack),
            sum(|g| g.pack_busy),
            self.host_widens.0,
            self.host_widens.1 as f64 / 1e9,
        ));
        out.push_str(&format!(
            "BLOCK PHASES: execute {:.2} · build {:.2} · prep {:.2} · A {:.2} (wait {:.2} upload {:.2} paid {:.2} commit {:.2} retire {:.2}) · B {:.2} (argue {:.2} open {:.2} tax {:.2} = upload {:.2} + encode {:.2})\n",
            self.execute,
            self.build,
            self.prep,
            self.phase_a,
            sum(|g| g.wait_a),
            sum(|g| g.upload_a),
            sum(|g| g.upload_paid),
            sum(|g| g.commit),
            sum(|g| g.retire),
            self.phase_b,
            argue,
            open,
            tax,
            sum(|g| g.upload_b),
            sum(|g| g.encode),
        ));
        if let Some(spill) = &self.spill {
            out.push_str(&format!("BLOCK SPILL: {spill}\n"));
        }
        out
    }
}

/// The counts a block statement may declare: [`TableCounts::validate`], with
/// KECCAK, KECCAK_RND, ECSM and ECDAS chunked (at most [`BLOCK_MAX_KECCAK`],
/// [`BLOCK_MAX_KECCAK_RND`], [`BLOCK_MAX_ECSM`] and [`BLOCK_MAX_ECDAS`]
/// tables).
pub fn validate_block_counts(counts: &TableCounts) -> Result<(), Error> {
    for (name, count, max) in [
        ("keccak", counts.keccak, BLOCK_MAX_KECCAK),
        ("keccak_rnd", counts.keccak_rnd, BLOCK_MAX_KECCAK_RND),
        ("ecsm", counts.ecsm, BLOCK_MAX_ECSM),
        ("ecdas", counts.ecdas, BLOCK_MAX_ECDAS),
    ] {
        if count > max {
            return Err(Error::InvalidTableCounts(format!(
                "{name} count is {count} — a block takes at most {max}"
            )));
        }
    }
    let mut rest = counts.clone();
    rest.keccak = rest.keccak.min(1);
    rest.keccak_rnd = rest.keccak_rnd.min(1);
    rest.ecsm = rest.ecsm.min(1);
    rest.ecdas = rest.ecdas.min(1);
    rest.validate()
}

/// The proof-order index range of each chunked table, with its height cap in
/// variables. The order is [`VmAirs::air_refs`]'s: the [`FIXED_TABLE_COUNT`]
/// fixed tables, then COMMIT, KECCAK, KECCAK_RND, ECSM, ECDAS, … (pinned
/// against the AIR names by `chunked_table_ranges_name_the_chunked_airs`).
/// Counts whose total passed [`TableCounts::total`] cannot overflow these sums.
pub(crate) fn chunked_table_ranges(
    counts: &TableCounts,
) -> [(&'static str, std::ops::Range<usize>, usize); 4] {
    let keccak = FIXED_TABLE_COUNT + counts.commit;
    let keccak_rnd = keccak + counts.keccak;
    let ecsm = keccak_rnd + counts.keccak_rnd;
    let ecdas = ecsm + counts.ecsm;
    [
        ("KECCAK", keccak..keccak_rnd, BLOCK_KECCAK_MAX_VARS),
        ("KECCAK_RND", keccak_rnd..ecsm, BLOCK_KECCAK_RND_MAX_VARS),
        ("ECSM", ecsm..ecdas, BLOCK_ECSM_MAX_VARS),
        ("ECDAS", ecdas..ecdas + counts.ecdas, BLOCK_ECDAS_MAX_VARS),
    ]
}

/// Every table at most [`BLOCK_MAX_TABLE_VARS`] variables, so no group's chain
/// is taller than the larger of the format's stack and 2^27.
pub(crate) fn check_table_heights(table_num_vars: &[u8]) -> Result<(), Error> {
    if let Some((idx, &num_vars)) = table_num_vars
        .iter()
        .enumerate()
        .find(|&(_, &n)| usize::from(n) > BLOCK_MAX_TABLE_VARS)
    {
        return Err(Error::InvalidTableCounts(format!(
            "table {idx} states 2^{num_vars} rows — a block table takes at most \
             2^{BLOCK_MAX_TABLE_VARS}"
        )));
    }
    Ok(())
}

/// Every chunked table under its cap, against the statement's heights: KECCAK
/// at most [`BLOCK_KECCAK_MAX_VARS`] variables, KECCAK_RND
/// [`BLOCK_KECCAK_RND_MAX_VARS`], ECSM [`BLOCK_ECSM_MAX_VARS`] and ECDAS
/// [`BLOCK_ECDAS_MAX_VARS`]. Run once the heights are known to be one per
/// table.
pub(crate) fn check_chunked_heights(
    counts: &TableCounts,
    table_num_vars: &[u8],
) -> Result<(), Error> {
    for (name, range, max_vars) in chunked_table_ranges(counts) {
        for (k, idx) in range.enumerate() {
            let num_vars = usize::from(table_num_vars[idx]);
            if num_vars > max_vars {
                return Err(Error::InvalidTableCounts(format!(
                    "{name}[{k}] (table {idx}) states 2^{num_vars} rows — a block takes at most \
                     2^{max_vars}"
                )));
            }
        }
    }
    Ok(())
}

/// The block statement: the monolithic multilinear one under
/// [`MULTILINEAR_BLOCK_TAG`], then the group partition — the count, then per
/// group its size and its tables' indices, every value a little-endian `u64`
/// (so the roots that follow stay on a field-element boundary).
#[allow(clippy::too_many_arguments)]
pub(crate) fn absorb_block(
    t: &mut impl IsTranscript<E>,
    elf_bytes: &[u8],
    public_output: &[u8],
    table_counts: &TableCounts,
    num_private_input_pages: usize,
    runtime_page_ranges: &[RuntimePageRange],
    table_num_vars: &[u8],
    config: &multilinear::whir_chain::ChainConfig,
    groups: &[Vec<u32>],
) {
    absorb_tagged(
        t,
        MULTILINEAR_BLOCK_TAG,
        "block",
        &statement::elf_digest(elf_bytes),
        public_output,
        table_counts,
        num_private_input_pages,
        runtime_page_ranges,
        table_num_vars,
        config,
    );
    t.append_bytes(&(groups.len() as u64).to_le_bytes());
    for group in groups {
        t.append_bytes(&(group.len() as u64).to_le_bytes());
        for &index in group {
            t.append_bytes(&u64::from(index).to_le_bytes());
        }
    }
}

/// A partition of `tables` tables into non-empty groups: every index below
/// `tables`, each exactly once. Returns the indices in proof order.
pub fn validate_groups(groups: &[Vec<u32>], tables: usize) -> Result<Vec<usize>, Error> {
    let mut seen = vec![false; tables];
    let mut order = Vec::with_capacity(tables);
    for group in groups {
        if group.is_empty() {
            return Err(Error::InvalidTableCounts("an empty group".to_string()));
        }
        for &index in group {
            let index = index as usize;
            match seen.get_mut(index) {
                Some(slot) if !*slot => *slot = true,
                Some(_) => {
                    return Err(Error::InvalidTableCounts(format!(
                        "table {index} is in two groups"
                    )));
                }
                None => {
                    return Err(Error::InvalidTableCounts(format!(
                        "group names table {index} of {tables}"
                    )));
                }
            }
            order.push(index);
        }
    }
    if order.len() != tables {
        return Err(Error::InvalidTableCounts(format!(
            "the groups cover {} of {tables} tables",
            order.len()
        )));
    }
    Ok(order)
}

/// A table cut into tables of `rows` rows each, in row order. Every VM table's
/// constraints read one row (no shifted reads), so each piece is a table of
/// the same AIR, and the bus sums over the pieces are the sum over the whole.
pub(crate) fn split_rows(table: TraceTable<F, E>, rows: usize) -> Vec<TraceTable<F, E>> {
    let height = table.main_table.height;
    if height <= rows {
        return vec![table];
    }
    let step = table.step_size;
    let width = table.main_table.width;
    // Row-major in, row-major out: each piece is a run of whole rows, copied
    // as they lie (a transposition of KECCAK_RND's 1,480 columns is what this
    // used to cost).
    (0..height / rows)
        .into_par_iter()
        .map(|k| {
            let mut data = Vec::with_capacity(rows * width);
            for row in k * rows..(k + 1) * rows {
                data.extend_from_slice(table.main_table.get_row(row));
            }
            TraceTable::new_main(data, width, step)
        })
        .collect()
}

/// One table of the proof: its layout and its columns, taken out of the trace
/// (whose row-major copy goes when `release_rows`), checked against the
/// columns the program implies.
fn table_of<'a>(
    air: &'a dyn stark::traits::AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
    trace: &mut TraceTable<F, E>,
    (width, num_vars): (usize, usize),
    release_rows: bool,
) -> Result<CommittedTable<'a, F, E>, Error> {
    let layout = layout_of(air, width, num_vars)
        .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))?;
    let mut columns = trace.main_table.columns_blocked();
    // The row-major copy goes as soon as the columns exist: a block's traces
    // are tens of GiB, and two copies of them at once is the peak this entry
    // point exists to avoid.
    if release_rows {
        trace.main_table = Table::new(Vec::new(), 0);
    }
    for (col, expected) in air.precomputed_columns().iter().enumerate() {
        if columns.get(col) != Some(expected) {
            return Err(Error::Prover(format!(
                "{}: preprocessed column {col} is not what the program implies",
                air.name(),
            )));
        }
    }
    CommittedTable::from_layout(layout, |col| core::mem::take(&mut columns[col as usize]))
        .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))
}

/// [`table_of`] for a trace the finish packed as it built it
/// ([`BlockOptions::pack_finished`]): its packed columns become the table's,
/// held narrow from the start, with no column of field elements made; the
/// preprocessed columns are checked word for word against the program's.
pub(crate) fn table_of_narrow<'a>(
    air: &'a dyn stark::traits::AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
    trace: &mut TraceTable<F, E>,
    (width, num_vars): (usize, usize),
) -> Result<CommittedTable<'a, F, E>, Error> {
    let layout = layout_of(air, width, num_vars)
        .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))?;
    let packed = trace
        .take_narrow_main()
        .ok_or_else(|| Error::Prover(format!("{}: no packed columns", air.name())))?;
    trace.main_table = Table::new(Vec::new(), 0);
    if packed.cols() != width || packed.rows() != 1usize << num_vars {
        return Err(Error::Prover(format!(
            "{}: packed {} × {}, the table is {width} × 2^{num_vars}",
            air.name(),
            packed.cols(),
            packed.rows()
        )));
    }
    for (col, expected) in air.precomputed_columns().iter().enumerate() {
        let same = col < packed.cols()
            && packed
                .column(col)
                .iter()
                .eq(expected.iter().map(|value| value.value()));
        if !same {
            return Err(Error::Prover(format!(
                "{}: preprocessed column {col} is not what the program implies",
                air.name(),
            )));
        }
    }
    CommittedTable::from_narrow(layout, packed)
        .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))
}

/// A prepared table's index in [`VmAirs::air_refs`] order and the columns its
/// opening settles.
pub(crate) type PreparedColumns = (usize, Vec<Vec<FieldElement<F>>>);

/// The prepared tables' columns' bytes, at their capacities.
fn prepared_bytes(prepared: &[PreparedColumns]) -> usize {
    prepared
        .iter()
        .flat_map(|(_, columns)| columns)
        .map(crate::tables::trace_builder::vec_heap_bytes)
        .sum()
}

/// One group's PREPARED commitment ([`multilinear_block::BlockPrepared`]): the
/// leading preprocessed columns of the group's prepared tables, stacked in the
/// group's order and committed under the block's parameters. Both sides derive
/// it from the program and the partition — the prover opens it, the verifier
/// absorbs its roots and checks the opening — and no proof carries it.
pub(crate) struct GroupPrepared<H: multilinear::whir_hash::WhirHash> {
    pub(crate) group: usize,
    /// `(position in the proof's table order, prefix length)`, in stack order.
    pub(crate) tables: Vec<(usize, usize)>,
    pub(crate) columns: Vec<Mle<F>>,
    pub(crate) roots: Vec<multilinear::whir_commit::Commitment>,
    pub(crate) commitment: multilinear::stacked_eval::StackedCommitment<F, H>,
}

/// One group's prepared stack before its commitment: which tables, and their
/// prefixes' columns concatenated in the group's order.
pub(crate) struct GroupStack {
    pub(crate) group: usize,
    /// `(position in the proof's table order, AIR index, prefix length)`.
    pub(crate) tables: Vec<(usize, usize, usize)>,
    pub(crate) columns: Vec<Vec<FieldElement<F>>>,
}

impl GroupStack {
    /// The stack's shapes, one `(columns, variables)` per table: what its layout
    /// is built from.
    pub(crate) fn shapes(&self) -> Vec<(usize, usize)> {
        let mut at = 0usize;
        self.tables
            .iter()
            .map(|&(_, _, n)| {
                let vars = self.columns[at].len().trailing_zeros() as usize;
                at += n;
                (n, vars)
            })
            .collect()
    }
}

/// The groups' prepared stacks, in group order: each group's prepared tables
/// ([`prepared_tables`]) in the group's table order. A pure function of the
/// prepared set and the partition — what both sides commit.
pub(crate) fn group_stacks(
    prepared: Vec<PreparedColumns>,
    groups: &[Vec<u32>],
) -> Result<Vec<GroupStack>, Error> {
    let mut by_table: Vec<Option<Vec<Vec<FieldElement<F>>>>> = Vec::new();
    for (table, columns) in prepared {
        if by_table.len() <= table {
            by_table.resize_with(table + 1, || None);
        }
        by_table[table] = Some(columns);
    }
    let mut stacks = Vec::new();
    let mut position = 0usize;
    for (g, group) in groups.iter().enumerate() {
        let mut stack = GroupStack {
            group: g,
            tables: Vec::new(),
            columns: Vec::new(),
        };
        for &table in group {
            let table = table as usize;
            if let Some(columns) = by_table.get_mut(table).and_then(Option::take) {
                stack.tables.push((position, table, columns.len()));
                stack.columns.extend(columns);
            }
            position += 1;
        }
        if !stack.tables.is_empty() {
            stacks.push(stack);
        }
    }
    if let Some(table) = by_table.iter().position(Option::is_some) {
        return Err(Error::Prover(format!(
            "prepared table {table} is in no group"
        )));
    }
    Ok(stacks)
}

/// The tables a block opens out of band, in [`VmAirs::air_refs`] order, with
/// the columns each opening settles:
/// - DECODE: its ELF-derived columns, all of them;
/// - every genesis page the cross-epoch rule stacks
///   ([`crate::continuation::genesis_stack_plan`]): both its columns.
///
/// These are the columns an in-guest verifier cannot evaluate: five columns of
/// 2^20, and a dense page whose sparse form is millions of rows. So the block
/// settles them with an opening, as an epoch settles DECODE. The result is a
/// function of the ELF and the page configs alone. The rule reads no private
/// page's bytes, so the prover's and the verifier's views of the configs give
/// the same set.
pub(crate) fn prepared_tables(
    airs: &VmAirs,
    page_configs: &[crate::tables::page::PageConfig],
    format: &BlockFormat,
) -> Result<Vec<PreparedColumns>, Error> {
    if !format.prepared {
        return Ok(Vec::new());
    }
    let refs = airs.air_refs();
    if page_configs.len() != airs.pages.len() {
        return Err(Error::Prover(format!(
            "{} page configs for {} page tables",
            page_configs.len(),
            airs.pages.len()
        )));
    }
    let first_page = first_page_index(airs)?;
    let decode = crate::multilinear_continuation::decode_table_index(&refs)?;
    let mut tables = vec![(decode, refs[decode].precomputed_columns())];
    let plan = crate::continuation::genesis_stack_plan(
        page_configs,
        first_page,
        crate::continuation::PAGE_NUM_VARS,
    );
    for route in plan.routes.iter().filter(|route| route.dense) {
        tables.push((
            route.table,
            crate::tables::page::preprocessed_columns(&page_configs[route.table - first_page]),
        ));
    }
    tables.sort_by_key(|&(table, _)| table);
    Ok(tables)
}

/// Where the page tables start in [`VmAirs::air_refs`] order (its length when
/// there are none): page `i` of the configs is table `first + i`.
pub(crate) fn first_page_index(airs: &VmAirs) -> Result<usize, Error> {
    let refs = airs.air_refs();
    match airs.pages.first() {
        Some(page) => refs
            .iter()
            .position(|air| {
                std::ptr::eq(
                    *air as *const _ as *const (),
                    page.as_ref() as *const _ as *const (),
                )
            })
            .ok_or_else(|| Error::Prover("the page tables are not in the AIR set".into())),
        None => Ok(refs.len()),
    }
}

/// Commits one group's prepared stack under `config`.
pub(crate) fn commit_group<H: multilinear::whir_hash::WhirHash>(
    stack: GroupStack,
    config: &multilinear::whir_chain::ChainConfig,
) -> Result<GroupPrepared<H>, Error> {
    let shapes = stack.shapes();
    let group = stack.group;
    let columns: Vec<Mle<F>> = stack
        .columns
        .into_iter()
        .map(|values| Mle::new(values).map_err(|e| Error::Prover(format!("group {group}: {e:?}"))))
        .collect::<Result<_, _>>()?;
    let layout = stark::multilinear_table::global_layout(&shapes, config.format.stack)
        .map_err(|e| Error::Prover(format!("{e:?}")))?;
    let commitment = multilinear::stacked_eval::StackedCommitment::<F, H>::commit(
        layout,
        &multilinear::stacking::borrow(&columns),
        None,
        config,
    )
    .map_err(|e| Error::Prover(format!("group {group}: {e:?}")))?;
    let roots = commitment.roots();
    Ok(GroupPrepared {
        group,
        tables: stack.tables.iter().map(|&(p, _, n)| (p, n)).collect(),
        columns,
        roots,
        commitment,
    })
}

/// Every group's prepared commitment, in group order.
pub(crate) fn commit_groups<H: multilinear::whir_hash::WhirHash>(
    stacks: Vec<GroupStack>,
    config: &multilinear::whir_chain::ChainConfig,
) -> Result<Vec<GroupPrepared<H>>, Error> {
    stacks
        .into_iter()
        .map(|stack| commit_group::<H>(stack, config))
        .collect()
}

/// The prepared commitments' roots by group, in group order.
fn prepared_roots<H: multilinear::whir_hash::WhirHash>(
    prepared: &[GroupPrepared<H>],
) -> Vec<PreparedRoots> {
    prepared
        .iter()
        .map(|p| (p.group, p.roots.clone()))
        .collect()
}

/// The prover's prepared commitments over `columns` ([`prepared_tables`]) for
/// the partition `groups`, the deviations applied, the deviations in the block
/// prover's terms, and the seconds they took.
#[allow(clippy::type_complexity)]
fn prover_prepared<H: multilinear::whir_hash::WhirHash>(
    columns: Vec<PreparedColumns>,
    groups: &[Vec<u32>],
    config: &multilinear::whir_chain::ChainConfig,
    deviations: &Deviations,
) -> Result<
    (
        Vec<GroupPrepared<H>>,
        Vec<multilinear_block::PreparedDeviation>,
        f64,
    ),
    Error,
> {
    let t = Instant::now();
    let originals: Vec<PreparedColumns> = if deviations.prepared_stack.is_empty() {
        Vec::new()
    } else {
        columns.clone()
    };
    let mut stacks = group_stacks(columns, groups)?;
    let mut touched: Vec<usize> = Vec::new();
    // Where table `air`'s block sits: its stack, and its columns' range there.
    let block_of = |stacks: &[GroupStack], air: usize| -> Option<(usize, std::ops::Range<usize>)> {
        stacks.iter().enumerate().find_map(|(k, stack)| {
            let mut at = 0usize;
            for &(_, table, n) in &stack.tables {
                if table == air {
                    return Some((k, at..at + n));
                }
                at += n;
            }
            None
        })
    };
    if deviations.other_prepared
        && let Some(stack) = stacks.first_mut()
    {
        stack.columns[0][0] += FieldElement::<F>::one();
        touched.push(stack.group);
    }
    for tamper in &deviations.prepared_stack {
        match *tamper {
            StackTamper::Swap(a, b) => {
                let (ka, ra) = block_of(&stacks, a)
                    .ok_or_else(|| Error::Prover(format!("{a} is not prepared")))?;
                let (kb, rb) = block_of(&stacks, b)
                    .ok_or_else(|| Error::Prover(format!("{b} is not prepared")))?;
                if ka != kb || ra.len() != rb.len() {
                    return Err(Error::Prover(
                        "a swap within one stack, of equal blocks".into(),
                    ));
                }
                for (i, j) in ra.zip(rb) {
                    stacks[ka].columns.swap(i, j);
                }
                touched.push(stacks[ka].group);
            }
            StackTamper::ColumnsOf { table, from } => {
                let (k, range) = block_of(&stacks, table)
                    .ok_or_else(|| Error::Prover(format!("{table} is not prepared")))?;
                let source = &originals
                    .iter()
                    .find(|(t, _)| *t == from)
                    .ok_or_else(|| Error::Prover(format!("{from} is not prepared")))?
                    .1;
                if source.len() != range.len() {
                    return Err(Error::Prover("blocks of different widths".into()));
                }
                for (i, column) in range.zip(source) {
                    stacks[k].columns[i] = column.clone();
                }
                touched.push(stacks[k].group);
            }
        }
    }
    let position_of = |air: usize| {
        groups
            .iter()
            .flatten()
            .position(|&t| t as usize == air)
            .ok_or_else(|| Error::Prover(format!("table {air} is in no group")))
    };
    let mut points: Vec<(usize, (usize, usize))> = Vec::new();
    for &(table, other) in &deviations.prepared_points {
        let (k, _) = block_of(&stacks, table)
            .ok_or_else(|| Error::Prover(format!("{table} is not prepared")))?;
        points.push((stacks[k].group, (position_of(table)?, position_of(other)?)));
        touched.push(stacks[k].group);
    }
    let prepared = commit_groups::<H>(stacks, config)?;
    touched.sort_unstable();
    touched.dedup();
    let tampered = touched
        .into_iter()
        .map(|group| {
            let open = prepared
                .iter()
                .position(|p| p.group == group)
                .expect("a touched group holds a stack");
            multilinear_block::PreparedDeviation {
                open,
                at_point_of: points
                    .iter()
                    .filter(|(g, _)| *g == group)
                    .map(|&(_, pair)| pair)
                    .collect(),
                consistent: true,
            }
        })
        .collect();
    Ok((prepared, tampered, t.elapsed().as_secs_f64()))
}

/// Proves a block in one proof.
pub fn prove_block_whir(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
) -> Result<(BlockWhirProof, BlockStamps), Error> {
    prove_block_whir_with(
        elf_bytes,
        private_inputs,
        proof_options,
        format,
        options,
        &Deviations::default(),
    )
}

/// What a test makes the prover do wrong, for the verifier to refuse. The
/// default is an honest prover.
#[derive(Default)]
pub(crate) struct Deviations {
    /// Leave the first KECCAK_RND table out of the proof — its rows, its
    /// argument and its count — as a prover hiding an instance would. The
    /// first, because it holds real rounds: a table of padding rows alone
    /// carries nothing on the bus, and leaving one out is an honest proof.
    pub omit_first_keccak_rnd: bool,
    /// Prove group `g` on the fork of this index instead of `g`.
    pub fork_of: Option<fn(usize) -> usize>,
    /// Commit the first prepared stack over columns that differ from the
    /// program's in one entry (DECODE's, the first prepared table), as a prover
    /// of another program would, and open it consistently.
    pub other_prepared: bool,
    /// Commit the prepared stacks wrongly ([`StackTamper`]), opening each
    /// consistently.
    pub prepared_stack: Vec<StackTamper>,
    /// `(table, other)`, AIR indices: open `table`'s prepared block at `other`'s
    /// point (a table of its group and its height), consistently.
    pub prepared_points: Vec<(usize, usize)>,
    /// Under the batched argue: a group argued with faults, or every group
    /// argued on the host ([`multilinear_block::ArgueDeviation`]).
    pub argue: multilinear_block::ArgueDeviation,
    /// Between the phases, the first narrow table's width map made wrong, so
    /// phase B widens other words than were committed
    /// ([`BlockCommitted::fault_narrow_width_map`]).
    pub narrow_width_map: bool,
    /// Between the phases, one byte of the first spilled table flipped on
    /// disk ([`BlockCommitted::fault_spilled_byte`]), so phase B reads back
    /// other bytes than were written.
    pub spilled_byte: bool,
    /// Between the phases, the first spilled table's slot lost
    /// ([`BlockCommitted::fault_lose_spilled_slot`]), so its columns never
    /// come back.
    pub spilled_slot_lost: bool,
    /// At the walk's end, every parked table handed to the store, whatever the
    /// forecast ([`hand_off_need`]).
    pub hand_off_all: bool,
    /// The shadow regenerator's slicer drops the first op of this table's
    /// list ([`regen::ShadowRun::spawn`]): a regeneration bug, which the shadow
    /// must report and the proof must not see.
    pub regen_drop_one_op: Option<StreamTable>,
    /// Phase B takes the groups in this order, given their count, instead of
    /// the prove's own ([`multilinear_block::block_prove_in_order`]): every
    /// order proves the same bytes.
    pub phase_b_order: Option<fn(usize) -> Vec<usize>>,
    /// Live regeneration's faults ([`regen::LiveFaults`]): a regenerator that
    /// dies or deposits wrong columns, which phase B must refuse, never prove
    /// over.
    pub regen_faults: regen::LiveFaults,
    /// `auto`'s target for the host, in bytes, instead of the machine's
    /// ([`spill_target_bytes`]): 0 makes the policy want every table out.
    pub spill_target: Option<u64>,
}

/// A prepared stack committed wrongly, tables by AIR index.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum StackTamper {
    /// Two tables' blocks exchanged in their (common) group's stack.
    Swap(usize, usize),
    /// `table`'s block holds `from`'s prepared columns instead of its own.
    ColumnsOf { table: usize, from: usize },
}

pub(crate) fn prove_block_whir_with(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    deviations: &Deviations,
) -> Result<(BlockWhirProof, BlockStamps), Error> {
    prove_block_whir_inner(
        elf_bytes,
        private_inputs,
        proof_options,
        format,
        options,
        deviations,
        &|_, _| {},
        &|_| {},
    )
}

/// What a caller is handed when the statement is final — after phase A,
/// before phase B: everything the recursion's programs are a function of, and
/// the prover's own prepared roots by group, in group order.
pub type StatementObserver<'a> = &'a (dyn Fn(BlockStatement<'_>, &[PreparedRoots]) + Sync);

/// A group with a prepared stack, and its stack's roots.
pub type PreparedRoots = (usize, Vec<multilinear::whir_commit::Commitment>);

/// What a caller is handed as each group's opening ends in phase B: that
/// group's share of the proof ([`multilinear_block::GroupOpened`]), so the
/// recursion's leaves over finished groups can start before phase B ends.
pub type GroupObserver<'a> = &'a (dyn Fn(multilinear_block::GroupOpened<'_, F, E>) + Sync);

/// [`prove_block_whir`], calling `on_statement` as soon as the statement is
/// final, so a caller can derive the recursion's programs while phase B runs.
pub fn prove_block_whir_observed(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    on_statement: StatementObserver<'_>,
) -> Result<(BlockWhirProof, BlockStamps), Error> {
    prove_block_whir_observed_groups(
        elf_bytes,
        private_inputs,
        proof_options,
        format,
        options,
        on_statement,
        &|_| {},
    )
}

/// [`prove_block_whir_observed`], also calling `on_group` as each group's
/// opening ends ([`GroupObserver`]). The observers read; the proof is the same.
pub fn prove_block_whir_observed_groups(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    on_statement: StatementObserver<'_>,
    on_group: GroupObserver<'_>,
) -> Result<(BlockWhirProof, BlockStamps), Error> {
    prove_block_whir_inner(
        elf_bytes,
        private_inputs,
        proof_options,
        format,
        options,
        &Deviations::default(),
        on_statement,
        on_group,
    )
}

/// [`prove_block_whir_observed`] with a test's deviations.
#[cfg(test)]
pub(crate) fn prove_block_whir_observed_with(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    deviations: &Deviations,
    on_statement: StatementObserver<'_>,
) -> Result<(BlockWhirProof, BlockStamps), Error> {
    prove_block_whir_inner(
        elf_bytes,
        private_inputs,
        proof_options,
        format,
        options,
        deviations,
        on_statement,
        &|_| {},
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_block_whir_inner(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    deviations: &Deviations,
    on_statement: StatementObserver<'_>,
    on_group: GroupObserver<'_>,
) -> Result<(BlockWhirProof, BlockStamps), Error> {
    let mut stamps = BlockStamps::default();
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;

    if let Some(window_log2) = options.window_log2 {
        let proof = prove_streamed(
            &program,
            elf_bytes,
            private_inputs,
            1usize << window_log2,
            proof_options,
            format,
            options,
            deviations,
            on_statement,
            on_group,
            &mut stamps,
        )?;
        return Ok((proof, stamps));
    }

    let t = Instant::now();
    let result = Executor::new(&program, private_inputs.to_vec())
        .map_err(|e| Error::Execution(format!("{e}")))?
        .run()
        .map_err(|e| Error::Execution(format!("{e}")))?;
    stamps.execute = t.elapsed().as_secs_f64();

    crate::tables::trace_builder::build_stamps::start();
    let t = Instant::now();
    let mut traces = Traces::from_elf_and_logs(
        &program,
        &result.logs,
        &options.max_rows,
        private_inputs,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )?;
    drop(result);
    stamps.build = t.elapsed().as_secs_f64();
    stamps.build_marks = crate::tables::trace_builder::build_stamps::take();

    let t = Instant::now();
    split_keccak_rnd(&mut traces, options.keccak_rnd_rows_log2);
    split_ecdas(&mut traces, options.ecdas_rows_log2);
    split_keccak(&mut traces, options.keccak_rows_log2);
    split_ecsm(&mut traces, options.ecsm_rows_log2);
    if deviations.omit_first_keccak_rnd && !traces.keccak_rnds.is_empty() {
        traces.keccak_rnds.remove(0);
    }
    stamps.prep = t.elapsed().as_secs_f64();
    let proof = prove_traces(
        &program,
        elf_bytes,
        &mut traces,
        proof_options,
        format,
        options,
        deviations,
        true,
        on_statement,
        &mut stamps,
    )?;
    Ok((proof, stamps))
}

/// Cuts every KECCAK_RND table into tables of at most `2^rows_log2` rows.
pub(crate) fn split_keccak_rnd(traces: &mut Traces, rows_log2: usize) {
    let rows = 1usize << rows_log2;
    traces.keccak_rnds = std::mem::take(&mut traces.keccak_rnds)
        .into_iter()
        .flat_map(|table| split_rows(table, rows))
        .collect();
}

/// Cuts every ECDAS table into tables of at most `2^rows_log2` rows: a cut
/// may fall inside one scalar multiplication, whose steps chain through the
/// Ecdas bus alone ([`BLOCK_ECDAS_ROWS_LOG2`]).
pub(crate) fn split_ecdas(traces: &mut Traces, rows_log2: usize) {
    let rows = 1usize << rows_log2;
    traces.ecdases = std::mem::take(&mut traces.ecdases)
        .into_iter()
        .flat_map(|table| split_rows(table, rows))
        .collect();
}

/// Cuts every KECCAK table into tables of at most `2^rows_log2` rows: a row is
/// one whole permutation call, so a cut falls between calls
/// ([`BLOCK_KECCAK_ROWS_LOG2`]).
pub(crate) fn split_keccak(traces: &mut Traces, rows_log2: usize) {
    let rows = 1usize << rows_log2;
    traces.keccaks = std::mem::take(&mut traces.keccaks)
        .into_iter()
        .flat_map(|table| split_rows(table, rows))
        .collect();
}

/// Cuts every ECSM table into tables of at most `2^rows_log2` rows: a row is
/// one whole scalar multiplication, so a cut falls between calls
/// ([`BLOCK_ECSM_ROWS_LOG2`]).
pub(crate) fn split_ecsm(traces: &mut Traces, rows_log2: usize) {
    let rows = 1usize << rows_log2;
    traces.ecsms = std::mem::take(&mut traces.ecsms)
        .into_iter()
        .flat_map(|table| split_rows(table, rows))
        .collect();
}

/// The block's proof over traces already built (and the chunked tables already
/// split). `release_rows` lets each table's row-major copy go as its columns
/// are taken — what a prove does; a test proving the same traces twice keeps
/// them. Adds `prep`, `phase_a`, `phase_b` and the groups to `stamps`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_traces(
    program: &Elf,
    elf_bytes: &[u8],
    traces: &mut Traces,
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    deviations: &Deviations,
    release_rows: bool,
    on_statement: StatementObserver<'_>,
    stamps: &mut BlockStamps,
) -> Result<BlockWhirProof, Error> {
    let t = Instant::now();
    let table_counts = traces.table_counts();
    validate_block_counts(&table_counts)?;
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();
    let public_output = traces.public_output_bytes.clone();

    let airs = VmAirs::new(
        program,
        proof_options,
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );

    let prepared_columns = prepared_tables(&airs, &traces.page_configs, format)?;
    let pairs = airs.air_trace_pairs(traces);
    let shapes = shapes_of(&pairs)?;
    let table_num_vars: Vec<u8> = shapes.iter().map(|&(_, n)| n as u8).collect();
    let config = format.chain_config(&shapes);
    let sizes = block_groups(&shapes, config.format.stack, format.group_polys)
        .map_err(|e| Error::Prover(format!("{e:?}")))?;
    check_group_count(sizes.len(), format.max_groups)?;

    stamps.tables = pairs.len();
    stamps.cells = shapes.iter().map(|&(w, n)| w << n).sum();
    stamps.prep += t.elapsed().as_secs_f64();

    // The pairs, cut into the groups phase A commits: a producer lays each
    // group's tables out (its tables in parallel) while the card commits the
    // group before it.
    let mut groups_of_pairs: Vec<Vec<(crate::AirTracePair<'_>, (usize, usize))>> =
        Vec::with_capacity(sizes.len());
    {
        let mut pairs = pairs.into_iter().zip(shapes.iter().copied());
        for &size in &sizes {
            groups_of_pairs.push(pairs.by_ref().take(size).collect());
        }
    }
    // The groups are contiguous runs in table order here.
    let groups: Vec<Vec<u32>> = {
        let mut at = 0u32;
        sizes
            .iter()
            .map(|&size| {
                let group = (at..at + size as u32).collect();
                at += size as u32;
                group
            })
            .collect()
    };
    let proof = crate::with_whir_hash!(|H| {
        let mut transcript =
            DefaultTranscript::<E, <H as multilinear::whir_hash::WhirHash>::Transcript>::new(&[]);
        absorb_block(
            &mut transcript,
            elf_bytes,
            &public_output,
            &table_counts,
            num_private_input_pages,
            &runtime_page_ranges,
            &table_num_vars,
            &config,
            &groups,
        );
        let t = Instant::now();
        let (block, produced) = std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            let producer = scope.spawn(move || -> Result<f64, Error> {
                let mut busy = 0.0;
                for group in groups_of_pairs {
                    let t = Instant::now();
                    let tables = group
                        .into_par_iter()
                        .map(|((air, trace, _), shape)| table_of(air, trace, shape, release_rows))
                        .collect::<Result<Vec<_>, Error>>()?;
                    busy += t.elapsed().as_secs_f64();
                    if tx.send(tables).is_err() {
                        break;
                    }
                }
                Ok(busy)
            });
            let block = BlockCommitted::commit_streamed::<H>(
                rx.iter(),
                &sizes,
                &config,
                options.drop_levels,
                options.narrow,
                options.upload_ahead,
            );
            (block, producer.join())
        });
        // The producer's error names the cause; a commit that ran out of
        // groups only says that it did.
        let produced =
            produced.map_err(|_| Error::Prover("the block's table producer panicked".into()))??;
        let mut block = block.map_err(|e| Error::Prover(format!("{e:?}")))?;
        if deviations.narrow_width_map && !block.fault_narrow_width_map() {
            return Err(Error::Prover("no narrow table to break".into()));
        }
        stamps.prep += produced;
        stamps.phase_a = t.elapsed().as_secs_f64();
        let (prepared, tampered, derive) =
            prover_prepared::<H>(prepared_columns, &groups, &config, deviations)?;
        stamps.prepared = (prepared.len(), derive);
        on_statement(
            BlockStatement {
                table_num_vars: &table_num_vars,
                runtime_page_ranges: &runtime_page_ranges,
                table_counts: &table_counts,
                public_output: &public_output,
                num_private_input_pages,
                groups: &groups,
            },
            &prepared_roots(&prepared),
        );
        let borrowed: Vec<Vec<&Mle<F>>> = prepared
            .iter()
            .map(|p| multilinear::stacking::borrow(&p.columns))
            .collect();
        let openings: Vec<multilinear_block::BlockPrepared<'_, F, H>> = prepared
            .iter()
            .zip(&borrowed)
            .map(|(p, columns)| multilinear_block::BlockPrepared {
                group: p.group,
                tables: p.tables.clone(),
                commitment: &p.commitment,
                columns,
            })
            .collect();
        let t = Instant::now();
        let paths_before = multilinear::whir_commit::top_path_counts();
        let routes_before = multilinear::whir_commit::top_path_routes();
        let widens_before = multilinear::narrow::host_widens();
        let identity = |g: usize| g;
        let fork_of: &dyn Fn(usize) -> usize = match &deviations.fork_of {
            Some(f) => f,
            None => &identity,
        };
        let (proof, argues, prepared_openings, groups) =
            multilinear_block::block_prove_on_forks::<_, _, _, H>(
                block,
                &config,
                &mut transcript,
                &openings,
                &tampered,
                fork_of,
                &deviations.argue,
            )
            .map_err(|e| Error::Prover(format!("{e:?}")))?;
        stamps.phase_b = t.elapsed().as_secs_f64();
        let paths_after = multilinear::whir_commit::top_path_counts();
        stamps.top_paths = (
            paths_after.0 - paths_before.0,
            paths_after.1 - paths_before.1,
        );
        let routes_after = multilinear::whir_commit::top_path_routes();
        stamps.top_routes = (
            (
                routes_after.0.0 - routes_before.0.0,
                routes_after.0.1 - routes_before.0.1,
            ),
            (
                routes_after.1.0 - routes_before.1.0,
                routes_after.1.1 - routes_before.1.1,
            ),
        );
        stamps.drop_levels = Some(options.drop_levels);
        let widens_after = multilinear::narrow::host_widens();
        stamps.narrow = options.narrow;
        stamps.host_widens = (
            widens_after.0 - widens_before.0,
            widens_after.1 - widens_before.1,
        );
        stamps.groups = groups;
        (proof, argues, prepared_openings)
    });

    Ok(BlockWhirProof {
        proof: proof.0,
        table_num_vars,
        runtime_page_ranges,
        table_counts,
        public_output,
        num_private_input_pages,
        groups,
        prepared: proof.2,
        argues: proof.1,
    })
}

/// One AIR per streamed table type, alive for the whole prove: a streamed
/// chunk is laid out (and its layout borrows its AIR) before the run's AIR set
/// exists. Every chunk of a type has the same AIR as the run's `TYPE[i]` — the
/// set builds them all with the same constructor — and the verifier checks the
/// result against its own set.
struct StreamAirs {
    /// The form the streamed chunks are written in (G-pack, `tables::gpack`).
    form: TraceForm,
    cpu: crate::VmAir,
    memw_register: crate::VmAir,
    memw_aligned: crate::VmAir,
    memw: crate::VmAir,
    load: crate::VmAir,
    lt: crate::VmAir,
    shift: crate::VmAir,
    store: crate::VmAir,
    keccak_rnd: crate::VmAir,
}

type DynAir = dyn stark::traits::AIR<Field = F, FieldExtension = E, PublicInputs = ()>;

impl StreamAirs {
    fn new(opts: &ProofOptions, form: TraceForm) -> Self {
        use crate::test_utils::*;
        Self {
            form,
            cpu: Box::new(create_cpu_air(opts)),
            memw_register: Box::new(create_memw_register_air(opts)),
            memw_aligned: Box::new(create_memw_aligned_air(opts)),
            memw: Box::new(create_memw_air(opts)),
            load: Box::new(create_load_air(opts)),
            lt: Box::new(create_lt_air(opts)),
            shift: Box::new(create_shift_air(opts)),
            store: Box::new(create_store_air(opts)),
            keccak_rnd: Box::new(create_keccak_rnd_air(opts)),
        }
    }

    fn of(&self, table: StreamTable) -> &DynAir {
        match table {
            StreamTable::Cpu => self.cpu.as_ref(),
            StreamTable::MemwRegister => self.memw_register.as_ref(),
            StreamTable::MemwAligned => self.memw_aligned.as_ref(),
            StreamTable::Memw => self.memw.as_ref(),
            StreamTable::Load => self.load.as_ref(),
            StreamTable::Lt => self.lt.as_ref(),
            StreamTable::Shift => self.shift.as_ref(),
            StreamTable::Store => self.store.as_ref(),
            StreamTable::KeccakRnd => self.keccak_rnd.as_ref(),
        }
    }
}

/// The name the run's AIR set gives chunk `index` of a streamed table.
fn stream_name(table: StreamTable, index: usize) -> String {
    let base = match table {
        StreamTable::Cpu => "CPU",
        StreamTable::MemwRegister => "MEMW_R",
        StreamTable::MemwAligned => "MEMW_A",
        StreamTable::Memw => "MEMW",
        StreamTable::Load => "LOAD",
        StreamTable::Lt => "LT",
        StreamTable::Shift => "SHIFT",
        StreamTable::Store => "STORE",
        StreamTable::KeccakRnd => "KECCAK_RND",
    };
    format!("{base}[{index}]")
}

/// What the builder thread reports when the run is built: when its windows
/// were all collected and when the run was built (seconds since the prove
/// started), how many chunks it handed out, the windows' stamps and `finish`'s
/// phase marks.
type BuilderReport = (f64, f64, usize, WindowStamps, Vec<(String, f64)>);

/// What the builder thread hands the layout thread.
enum Built {
    Job(Box<ChunkJob>),
    Rest(Box<Traces>),
    /// The rest streamed ([`BlockOptions::finish_stream`]).
    Stream(Box<RestStream>),
}

/// The rest as the finish streams it: the header (what the statement reads,
/// and the run's AIRs) and, in AIR order, each table or streamed chunk's slot.
struct RestStream {
    header: crate::tables::trace_builder::RestHeader,
    tables: std::sync::mpsc::Receiver<crate::tables::trace_builder::Emitted>,
}

/// The rest as the layout thread receives it.
enum RestIn {
    Whole(Box<Traces>),
    Stream(Box<RestStream>),
}

/// The rest's tables as they stream in, checked against the run's AIRs: each
/// the next AIR in order, a streamed chunk's slot or a table of that AIR's
/// width, and every AIR reached. A table's height is checked where it is laid
/// out ([`layout_of`]).
struct RestOrder {
    widths: Vec<usize>,
    next: usize,
}

impl RestOrder {
    fn admit(&mut self, position: usize, width: Option<usize>) -> Result<(), Error> {
        if position != self.next {
            return Err(Error::Prover(format!(
                "the rest's table {position} came where table {} was due",
                self.next
            )));
        }
        let want = *self.widths.get(position).ok_or_else(|| {
            Error::Prover(format!(
                "the rest's table {position} is past the run's {} tables",
                self.widths.len()
            ))
        })?;
        if let Some(width) = width
            && width != want
        {
            return Err(Error::Prover(format!(
                "the rest's table {position}: {width} columns, its AIR declares {want}"
            )));
        }
        self.next += 1;
        Ok(())
    }

    fn finish(&self) -> Result<(), Error> {
        if self.next != self.widths.len() {
            return Err(Error::Prover(format!(
                "table {} of the rest was never laid out",
                self.next
            )));
        }
        Ok(())
    }
}

/// A table before the run's AIR order exists: a streamed chunk, or a table of
/// the final build by its AIR index.
#[derive(Clone, Copy)]
enum Key {
    Streamed(StreamTable, usize),
    Air(usize),
}

/// A table of the final build, laid out: its AIR index, shape and table.
type Placed<'a> = (usize, (usize, usize), CommittedTable<'a, F, E>);

/// Packs tables, in the order they come, into groups of at most `max_polys`
/// stacked polynomials, and sends each group to phase A as it closes.
struct Packer<'a> {
    open: Vec<CommittedTable<'a, F, E>>,
    open_shapes: Vec<(usize, usize)>,
    open_keys: Vec<Key>,
    group_keys: Vec<Vec<Key>>,
    out: std::sync::mpsc::SyncSender<Vec<CommittedTable<'a, F, E>>>,
    cap: multilinear::whir_chain::StackVars,
    max_polys: usize,
    /// The most groups the statement may declare: the group past it is
    /// refused as it closes.
    max_groups: usize,
    /// The prove's clock, when each group closed, and the seconds spent
    /// waiting for phase A to take one.
    start: Instant,
    closed_at: Vec<f64>,
    blocked: f64,
    /// A memory log, and the open group's bytes in it.
    ledger: Option<&'a memlog::Ledger>,
    open_bytes: usize,
}

impl<'a> Packer<'a> {
    /// A group closes, and goes to the card, when the next table would take
    /// it past its polynomial budget.
    fn place(
        &mut self,
        key: Key,
        shape: (usize, usize),
        table: CommittedTable<'a, F, E>,
    ) -> Result<(), Error> {
        self.open_shapes.push(shape);
        let polys = stark::multilinear_table::global_layout(&self.open_shapes, self.cap)
            .map_err(|e| Error::Prover(format!("{e:?}")))?
            .num_polys();
        if polys > self.max_polys && !self.open.is_empty() {
            self.open_shapes.clear();
            self.open_shapes.push(shape);
            self.close()?;
        }
        if let Some(ledger) = self.ledger {
            let bytes = memlog::table_bytes(&table);
            // A table of the run's AIR order comes from the rest's layout.
            if let Key::Air(_) = key {
                ledger
                    .rest_laid
                    .fetch_sub(bytes, std::sync::atomic::Ordering::Relaxed);
            }
            ledger
                .open
                .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
            self.open_bytes += bytes;
        }
        self.open.push(table);
        self.open_keys.push(key);
        Ok(())
    }

    fn close(&mut self) -> Result<(), Error> {
        check_group_count(self.group_keys.len() + 1, self.max_groups)?;
        if let Some(ledger) = self.ledger {
            use std::sync::atomic::Ordering::Relaxed;
            let bytes = std::mem::take(&mut self.open_bytes);
            ledger.open.fetch_sub(bytes, Relaxed);
            ledger.sent.fetch_add(bytes, Relaxed);
        }
        self.group_keys.push(std::mem::take(&mut self.open_keys));
        self.closed_at.push(self.start.elapsed().as_secs_f64());
        let t = Instant::now();
        let sent = self
            .out
            .send(std::mem::take(&mut self.open))
            .map_err(|_| Error::Prover("phase A stopped".into()));
        self.blocked += t.elapsed().as_secs_f64();
        sent
    }

    /// Sends the last group; hands back every group's tables, when each
    /// closed and the seconds spent waiting on phase A.
    fn finish(mut self) -> Result<Packed, Error> {
        if !self.open.is_empty() {
            self.close()?;
        }
        Ok((self.group_keys, self.closed_at, self.blocked))
    }
}

/// Every group's tables, when each group closed, and the seconds the packer
/// waited on phase A ([`Packer::finish`]).
type Packed = (Vec<Vec<Key>>, Vec<f64>, f64);

/// A streamed chunk laid out: its table, its index, its shape and the table.
type LaidChunk<'a> = (StreamTable, usize, (usize, usize), CommittedTable<'a, F, E>);

/// One streamed chunk generated and laid out; a regeneration recorder takes
/// the digest of its packed columns, as committed.
fn lay_out_chunk<'a>(
    airs: &'a StreamAirs,
    job: ChunkJob,
    ledger: Option<&memlog::Ledger>,
    recorder: Option<&regen::Recorder>,
) -> Result<LaidChunk<'a>, Error> {
    use std::sync::atomic::Ordering::Relaxed;
    // A memory log: the job leaves the queue as its ops, then its trace.
    let ops = ledger.map_or(0, |ledger| {
        let ops = job.op_bytes();
        ledger.jobs.fetch_sub(ops, Relaxed);
        ledger.laying.fetch_add(ops, Relaxed);
        ops
    });
    let mut chunk = job.generate_as(airs.form);
    if let (Some(recorder), Some(packed)) =
        (recorder.filter(|r| r.digests()), chunk.trace.narrow_main())
    {
        recorder.packed(chunk.table, chunk.index, packed);
    }
    let rows = ledger.map_or(0, |ledger| {
        let rows = memlog::rows_bytes(&chunk.trace);
        ledger.laying.fetch_add(rows, Relaxed);
        ledger.laying.fetch_sub(ops, Relaxed);
        rows
    });
    let height = chunk.trace.main_table.height;
    let shape = (
        chunk.trace.main_table.width,
        height.trailing_zeros() as usize,
    );
    // Written packed (G-pack): laid out narrow from its packed columns, with
    // no transposition.
    let table = if chunk.trace.narrow_main().is_some() {
        table_of_narrow(airs.of(chunk.table), &mut chunk.trace, shape)
    } else {
        table_of(airs.of(chunk.table), &mut chunk.trace, shape, true)
    };
    if let Some(ledger) = ledger {
        ledger.laying.fetch_sub(rows, Relaxed);
    }
    Ok((chunk.table, chunk.index, shape, table?))
}

/// The streamed chunks, packed in arrival order: the packer, each chunk's
/// shape, the seconds laying them out (summed over threads) and when the last
/// was placed; and the rest's tables it packed as they were laid out, by AIR
/// index ([`BlockOptions::pack_rest_as_laid_out`]).
struct StreamLaid<'a> {
    packer: Packer<'a>,
    shapes: Vec<(StreamTable, usize, (usize, usize))>,
    chunks: f64,
    placed_at: f64,
    rest: Vec<(usize, (usize, usize))>,
    /// The most chunks laid out (or in the making) and not yet packed at once;
    /// 0 when nothing bounded them.
    ahead_most: usize,
}

/// A table of the rest laid out, sent to the packer: its position in AIR order
/// among the rest, the seconds laying it out, and the table.
type RestDone<'a> = (usize, f64, Result<Placed<'a>, Error>);

/// Each chunk laid out and packed on this thread as it arrives; hands back
/// the run once it is built.
fn stream_inline<'a>(
    brx: std::sync::mpsc::Receiver<Built>,
    airs: &'a StreamAirs,
    mut packer: Packer<'a>,
    recorder: Option<&regen::Recorder>,
) -> Result<(StreamLaid<'a>, Box<Traces>), Error> {
    let mut shapes = Vec::new();
    let mut chunks = 0.0;
    let ledger = packer.ledger;
    for item in brx {
        match item {
            Built::Job(job) => {
                let t = Instant::now();
                let (table, index, shape, laid) = lay_out_chunk(airs, *job, ledger, recorder)?;
                chunks += t.elapsed().as_secs_f64();
                shapes.push((table, index, shape));
                packer.place(Key::Streamed(table, index), shape, laid)?;
            }
            Built::Stream(_) => {
                return Err(Error::Prover(
                    "a streamed rest needs the pipelined layout".into(),
                ));
            }
            Built::Rest(traces) => {
                let placed_at = packer.start.elapsed().as_secs_f64();
                let ahead_most = usize::from(!shapes.is_empty());
                let streamed = StreamLaid {
                    packer,
                    shapes,
                    chunks,
                    placed_at,
                    rest: Vec::new(),
                    ahead_most,
                };
                return Ok((streamed, traces));
            }
        }
    }
    Err(Error::Prover(
        "the builder stopped before the run was built".into(),
    ))
}

/// Streamed chunks laid out (or in the making) and not yet packed, at most so
/// many ([`BlockOptions::layout_ahead`]): a worker takes a permit before it
/// takes a job, so every job taken holds one and the next chunk to pack is
/// never starved; the packer gives it back once the chunk is packed.
struct Permits {
    state: std::sync::Mutex<PermitState>,
    freed: std::sync::Condvar,
}

#[derive(Default)]
struct PermitState {
    free: usize,
    out: usize,
    most: usize,
    closed: bool,
}

impl Permits {
    fn new(n: usize) -> Self {
        Self {
            state: std::sync::Mutex::new(PermitState {
                free: n,
                ..PermitState::default()
            }),
            freed: std::sync::Condvar::new(),
        }
    }

    /// Waits for a permit; `false` once the packer has stopped.
    fn take(&self) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        loop {
            if state.closed {
                return false;
            }
            if state.free > 0 {
                state.free -= 1;
                state.out += 1;
                state.most = state.most.max(state.out);
                return true;
            }
            state = match self.freed.wait(state) {
                Ok(state) => state,
                Err(_) => return false,
            };
        }
    }

    fn give(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.free += 1;
            state.out = state.out.saturating_sub(1);
        }
        self.freed.notify_one();
    }

    /// Wakes every waiting worker for good.
    fn close(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.closed = true;
        }
        self.freed.notify_all();
    }

    fn most(&self) -> usize {
        self.state.lock().map_or(0, |state| state.most)
    }
}

/// Closes the permits when the packer stops, however it stops.
struct ClosePermits<'p>(Option<&'p Permits>);

impl Drop for ClosePermits<'_> {
    fn drop(&mut self) {
        if let Some(permits) = self.0 {
            permits.close();
        }
    }
}

/// The chunks laid out on `workers` threads and packed on one more, in
/// arrival order all the same, so a waiting phase A holds back only the
/// packing; this thread hands them out and, once the run is built, lays out
/// the rest of it (`rest`) while the last chunks are packed. With
/// `pack_rest`, or a streamed rest, `rest` sends the rest's tables to the
/// packer as they are laid out, and it packs them after the chunks, in AIR
/// order, giving each table's bytes back to `gate` as it places it (a
/// streamed rest's byte gate, stopped when the packer stops). Their channel is
/// unbounded: the rest is already in memory, or held under the gate, and a
/// full channel would park the rayon threads phase A's commits need.
#[allow(clippy::too_many_arguments)]
fn stream_pipelined<'a, R>(
    brx: std::sync::mpsc::Receiver<Built>,
    airs: &'a StreamAirs,
    mut packer: Packer<'a>,
    workers: usize,
    ahead: Option<usize>,
    pack_rest: bool,
    gate: Option<&'a crate::tables::trace_builder::gate::ByteGate>,
    recorder: Option<&regen::Recorder>,
    rest: impl FnOnce(RestIn, Option<std::sync::mpsc::Sender<RestDone<'a>>>) -> Result<R, Error>,
) -> Result<(StreamLaid<'a>, R), Error> {
    type Done<'a> = (usize, f64, Result<LaidChunk<'a>, Error>);
    let permits = ahead.map(|k| Permits::new(k + 1));
    let permits = permits.as_ref();
    let ledger = packer.ledger;
    std::thread::scope(|scope| {
        let (jtx, jrx) = std::sync::mpsc::sync_channel::<(usize, Box<ChunkJob>)>(workers);
        let jrx = std::sync::Arc::new(std::sync::Mutex::new(jrx));
        let (dtx, drx) = std::sync::mpsc::sync_channel::<Done<'a>>(workers);
        for _ in 0..workers {
            let jrx = std::sync::Arc::clone(&jrx);
            let dtx = dtx.clone();
            scope.spawn(move || {
                loop {
                    if let Some(permits) = permits
                        && !permits.take()
                    {
                        return;
                    }
                    // The lock is held for the receive alone. A poisoned lock
                    // means a sibling panicked, which the scope reports.
                    let next = match jrx.lock() {
                        Ok(jobs) => jobs.recv(),
                        Err(_) => Err(std::sync::mpsc::RecvError),
                    };
                    let Ok((seq, job)) = next else {
                        if let Some(permits) = permits {
                            permits.give();
                        }
                        return;
                    };
                    let t = Instant::now();
                    let laid = lay_out_chunk(airs, *job, ledger, recorder);
                    if dtx.send((seq, t.elapsed().as_secs_f64(), laid)).is_err() {
                        return;
                    }
                }
            });
        }
        // The workers hold the only receivers and senders: once they stop,
        // handing out fails and the packer's loop ends.
        drop(jrx);
        drop(dtx);
        let (rtx, rrx) = std::sync::mpsc::channel::<RestDone<'a>>();
        let placer = scope.spawn(move || -> Result<StreamLaid<'a>, Error> {
            let _close = ClosePermits(permits);
            let _stop = gate.map(|gate| {
                crate::tables::trace_builder::gate::StopGate(gate, "the packer stopped")
            });
            let mut pending = std::collections::BTreeMap::new();
            let mut next = 0usize;
            let mut shapes = Vec::new();
            let mut chunks = 0.0;
            for (seq, secs, laid) in drx {
                chunks += secs;
                pending.insert(seq, laid);
                while let Some(laid) = pending.remove(&next) {
                    let (table, index, shape, laid) = laid?;
                    shapes.push((table, index, shape));
                    packer.place(Key::Streamed(table, index), shape, laid)?;
                    if let Some(permits) = permits {
                        permits.give();
                    }
                    next += 1;
                }
            }
            if !pending.is_empty() {
                return Err(Error::Prover(format!(
                    "streamed chunk {next} was never laid out"
                )));
            }
            let placed_at = packer.start.elapsed().as_secs_f64();
            // The rest's tables, after every chunk, in AIR order.
            let mut placed = Vec::new();
            let mut waiting = std::collections::BTreeMap::new();
            let mut next = 0usize;
            for (k, _, laid) in rrx {
                waiting.insert(k, laid);
                while let Some(laid) = waiting.remove(&next) {
                    let (i, shape, table) = laid?;
                    placed.push((i, shape));
                    packer.place(Key::Air(i), shape, table)?;
                    if let Some(gate) = gate {
                        gate.release(next);
                    }
                    next += 1;
                }
            }
            if !waiting.is_empty() {
                return Err(Error::Prover(format!(
                    "table {next} of the rest was never laid out"
                )));
            }
            Ok(StreamLaid {
                packer,
                shapes,
                chunks,
                placed_at,
                rest: placed,
                ahead_most: 0,
            })
        });
        let mut handed = 0usize;
        let mut traces = None;
        for item in brx {
            match item {
                Built::Job(job) => {
                    if jtx.send((handed, job)).is_err() {
                        break;
                    }
                    handed += 1;
                }
                Built::Rest(built) => {
                    traces = Some(RestIn::Whole(built));
                    break;
                }
                Built::Stream(stream) => {
                    traces = Some(RestIn::Stream(stream));
                    break;
                }
            }
        }
        drop(jtx);
        let rest = match traces {
            // A streamed rest is always placed as it is laid out.
            Some(traces @ RestIn::Stream(_)) => Some(rest(traces, Some(rtx))),
            Some(traces) => Some(rest(traces, pack_rest.then_some(rtx))),
            None => {
                drop(rtx);
                None
            }
        };
        let mut streamed = placer
            .join()
            .map_err(|_| Error::Prover("the block's packer panicked".into()))??;
        streamed.ahead_most = permits.map_or(0, Permits::most);
        if streamed.shapes.len() != handed {
            return Err(Error::Prover(format!(
                "{} of {handed} streamed chunks packed",
                streamed.shapes.len()
            )));
        }
        let rest = rest.ok_or_else(|| {
            Error::Prover("the builder stopped before the run was built".into())
        })??;
        Ok((streamed, rest))
    })
}

/// The run's tables the windows did not stream, laid out in parallel, and what
/// the statement reads off the run; `marks` are seconds since the prove
/// started. `prepared` is `None` when it is left for after the packing.
struct RestLaid<'a> {
    table_counts: TableCounts,
    runtime_page_ranges: Vec<RuntimePageRange>,
    num_private_input_pages: usize,
    public_output: Vec<u8>,
    airs: &'a VmAirs,
    page_configs: Vec<crate::tables::page::PageConfig>,
    refs: Vec<&'a dyn stark::traits::AIR<Field = F, FieldExtension = E, PublicInputs = ()>>,
    prepared: Option<Vec<PreparedColumns>>,
    tables: Vec<Placed<'a>>,
    marks: Vec<(&'static str, f64)>,
    /// The five slowest tables to lay out and the first in AIR order, seconds.
    slowest: Vec<(String, f64)>,
    busy: f64,
    /// [`LayoutStamps::packed_rest`].
    packed_rest: (usize, usize),
    /// Each table's AIR index and when its layout ended (seconds since the
    /// prove started), in AIR order.
    laid_at: Vec<(usize, f64)>,
}

/// `prepared_now`: derive the prepared columns here, before the tables are
/// laid out; otherwise the caller derives them once the groups are sent (phase
/// A reads none of them). `sink`: send each table to the packer as it is laid
/// out, the first in AIR order first, and hand back none.
#[allow(clippy::too_many_arguments)]
fn lay_out_rest<'a>(
    mut traces: Box<Traces>,
    program: &Elf,
    opts: &ProofOptions,
    format: &BlockFormat,
    run_airs: &'a std::sync::OnceLock<VmAirs>,
    start: Instant,
    prepared_now: bool,
    sink: Option<std::sync::mpsc::Sender<RestDone<'a>>>,
    budget: Option<usize>,
    ledger: Option<&memlog::Ledger>,
) -> Result<RestLaid<'a>, Error> {
    let t = Instant::now();
    let at = || start.elapsed().as_secs_f64();
    let mut marks = vec![("rest received", at())];
    let table_counts = traces.table_counts();
    validate_block_counts(&table_counts)?;
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();
    let public_output = traces.public_output_bytes.clone();
    let airs = run_airs.get_or_init(|| {
        VmAirs::new(
            program,
            opts,
            false,
            &traces.page_configs,
            &table_counts,
            None,
            true,
            None,
            None,
            None,
        )
    });
    marks.push(("airs", at()));
    let prepared = if prepared_now {
        let prepared = prepared_tables(airs, &traces.page_configs, format)?;
        marks.push(("prepared", at()));
        if let Some(ledger) = ledger {
            ledger.prepared.store(
                prepared_bytes(&prepared),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        Some(prepared)
    } else {
        None
    };
    let page_configs = traces.page_configs.clone();
    let refs = airs.air_refs();
    let pairs = airs.air_trace_pairs(&mut traces);
    let built: Vec<_> = pairs
        .into_iter()
        .zip(refs.iter().copied())
        .enumerate()
        .filter(|(_, ((_, trace, _), _))| trace.main_table.width != 0)
        .collect();
    // Each table's seconds and when its layout ended, for the readout.
    let timed = std::sync::Mutex::new(Vec::with_capacity(built.len()));
    let laid_at = std::sync::Mutex::new(Vec::with_capacity(built.len()));
    let narrow_tables = std::sync::atomic::AtomicUsize::new(0);
    let narrow_bytes = std::sync::atomic::AtomicUsize::new(0);
    let lay_out = |(i, ((_, trace, _), air)): (usize, ((_, &mut TraceTable<F, E>, _), _))| {
        let t = Instant::now();
        let shape = (
            trace.main_table.width,
            trace.main_table.height.trailing_zeros() as usize,
        );
        let rows = ledger.map_or(0, |_| memlog::rows_bytes(trace));
        let laid = if let Some(packed) = trace.narrow_main() {
            use std::sync::atomic::Ordering::Relaxed;
            narrow_tables.fetch_add(1, Relaxed);
            narrow_bytes.fetch_add(packed.data().len(), Relaxed);
            table_of_narrow(air, trace, shape)
        } else {
            table_of(air, trace, shape, true)
        }
        .map(|table| (i, shape, table));
        if let Ok(mut laid_at) = laid_at.lock() {
            laid_at.push((i, start.elapsed().as_secs_f64()));
        }
        if let (Some(ledger), Ok((_, _, table))) = (ledger, &laid) {
            use std::sync::atomic::Ordering::Relaxed;
            ledger.rest.fetch_sub(rows, Relaxed);
            ledger
                .rest_laid
                .fetch_add(memlog::table_bytes(table), Relaxed);
        }
        if let Ok(mut timed) = timed.lock() {
            timed.push((i, t.elapsed().as_secs_f64()));
        }
        laid
    };
    let rows_of = |(_, ((_, trace, _), _)): &(usize, ((_, &mut TraceTable<F, E>, _), _))| {
        memlog::rows_bytes(trace)
    };
    let tables: Vec<Placed<'a>> = match (sink, budget) {
        (None, None) => built
            .into_par_iter()
            .map(&lay_out)
            .collect::<Result<_, Error>>()?,
        (None, Some(budget)) => {
            let mut tables = Vec::with_capacity(built.len());
            for wave in waves(built, budget, rows_of) {
                tables.extend(
                    wave.into_par_iter()
                        .map(&lay_out)
                        .collect::<Result<Vec<_>, Error>>()?,
                );
            }
            tables
        }
        (Some(sink), Some(budget)) => {
            // Each wave's tables to the packer in AIR order once it is laid
            // out.
            let mut k = 0usize;
            for wave in waves(built, budget, rows_of) {
                let laid: Vec<(f64, Result<Placed<'a>, Error>)> = wave
                    .into_par_iter()
                    .map(|table| {
                        let t = Instant::now();
                        let laid = lay_out(table);
                        (t.elapsed().as_secs_f64(), laid)
                    })
                    .collect();
                for (secs, laid) in laid {
                    // A packer that stopped has its own error to report.
                    let _ = sink.send((k, secs, laid));
                    k += 1;
                }
            }
            Vec::new()
        }
        (Some(sink), None) => {
            // FIFO: the tables that close the next group are laid out first.
            rayon::scope_fifo(|scope| {
                for (k, table) in built.into_iter().enumerate() {
                    let sink = sink.clone();
                    let lay_out = &lay_out;
                    scope.spawn_fifo(move |_| {
                        let t = Instant::now();
                        let laid = lay_out(table);
                        // A packer that stopped has its own error to report.
                        let _ = sink.send((k, t.elapsed().as_secs_f64(), laid));
                    });
                }
            });
            Vec::new()
        }
    };
    marks.push(("rest laid out", at()));
    if let Some(ledger) = ledger {
        // What is left of the run's tables (those no AIR proves: a padded
        // table of an unused chip) goes with `traces` as this returns.
        ledger.rest.store(0, std::sync::atomic::Ordering::Relaxed);
        ledger.line("rest laid out");
    }
    let mut timed = timed.into_inner().unwrap_or_default();
    let first = timed.iter().map(|&(i, _)| i).min();
    timed.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut slowest: Vec<(String, f64)> = timed
        .iter()
        .take(5)
        .map(|&(i, secs)| (refs[i].name().to_string(), secs))
        .collect();
    if let Some(&(i, secs)) = first.and_then(|f| timed.iter().find(|&&(i, _)| i == f)) {
        slowest.push((format!("first in AIR order: {}", refs[i].name()), secs));
    }
    Ok(RestLaid {
        table_counts,
        runtime_page_ranges,
        num_private_input_pages,
        public_output,
        airs,
        page_configs,
        refs,
        prepared,
        tables,
        marks,
        slowest,
        busy: t.elapsed().as_secs_f64(),
        packed_rest: (narrow_tables.into_inner(), narrow_bytes.into_inner()),
        laid_at: {
            let mut laid_at = laid_at.into_inner().unwrap_or_default();
            laid_at.sort_unstable_by_key(|&(i, _)| i);
            laid_at
        },
    })
}

/// The rest as the finish streams it ([`RestStream`]): each table laid out
/// narrow as it comes (wide ones through [`table_of`]) and sent to the packer
/// in AIR order (`sink`), every table checked against the run's AIRs built
/// from the header ([`RestOrder`]). The prepared columns are left for after
/// the packing, as on the pipelined path.
#[allow(clippy::too_many_arguments)]
fn lay_out_rest_streamed<'a>(
    stream: Box<RestStream>,
    program: &Elf,
    opts: &ProofOptions,
    run_airs: &'a std::sync::OnceLock<VmAirs>,
    start: Instant,
    sink: Option<std::sync::mpsc::Sender<RestDone<'a>>>,
    ledger: Option<&memlog::Ledger>,
) -> Result<RestLaid<'a>, Error> {
    use std::sync::atomic::Ordering::Relaxed;
    let sink =
        sink.ok_or_else(|| Error::Prover("a streamed rest needs the packer's sink".into()))?;
    let at = || start.elapsed().as_secs_f64();
    let mut marks = vec![("rest received", at())];
    let RestStream { header, tables } = *stream;
    validate_block_counts(&header.table_counts)?;
    let runtime_page_ranges = header.runtime_page_ranges();
    let num_private_input_pages = header
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();
    let airs = run_airs.get_or_init(|| {
        VmAirs::new(
            program,
            opts,
            false,
            &header.page_configs,
            &header.table_counts,
            None,
            true,
            None,
            None,
            None,
        )
    });
    marks.push(("airs", at()));
    let refs = airs.air_refs();
    let mut order = RestOrder {
        widths: refs.iter().map(|air| air.trace_layout().0).collect(),
        next: 0,
    };
    let mut timed: Vec<(usize, f64)> = Vec::new();
    let mut laid_at: Vec<(usize, f64)> = Vec::new();
    let (mut narrow_tables, mut narrow_bytes) = (0usize, 0usize);
    let mut k = 0usize;
    for crate::tables::trace_builder::Emitted { position, table } in tables {
        order.admit(position, table.as_ref().map(|t| t.main_table.width))?;
        let Some(mut trace) = table else {
            continue;
        };
        if k == 0 {
            marks.push(("rest first received", at()));
        }
        let t = Instant::now();
        let shape = (
            trace.main_table.width,
            trace.main_table.height.trailing_zeros() as usize,
        );
        let rows = memlog::rows_bytes(&trace);
        let laid = if let Some(packed) = trace.narrow_main() {
            narrow_tables += 1;
            narrow_bytes += packed.data().len();
            table_of_narrow(refs[position], &mut trace, shape)
        } else {
            table_of(refs[position], &mut trace, shape, true)
        };
        let secs = t.elapsed().as_secs_f64();
        timed.push((position, secs));
        laid_at.push((position, at()));
        if let (Some(ledger), Ok(table)) = (ledger, &laid) {
            ledger.rest.fetch_sub(rows, Relaxed);
            ledger
                .rest_laid
                .fetch_add(memlog::table_bytes(table), Relaxed);
        }
        // A packer that stopped has its own error to report.
        let _ = sink.send((k, secs, laid.map(|table| (position, shape, table))));
        k += 1;
    }
    order.finish()?;
    marks.push(("rest laid out", at()));
    if let Some(ledger) = ledger {
        ledger.line("rest laid out");
    }
    let busy = timed.iter().map(|&(_, secs)| secs).sum();
    let first = timed.first().map(|&(i, _)| i);
    let mut slowest_of = timed.clone();
    slowest_of.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut slowest: Vec<(String, f64)> = slowest_of
        .iter()
        .take(5)
        .map(|&(i, secs)| (refs[i].name().to_string(), secs))
        .collect();
    if let Some(&(i, secs)) = first.and_then(|f| timed.iter().find(|&&(i, _)| i == f)) {
        slowest.push((format!("first in AIR order: {}", refs[i].name()), secs));
    }
    let crate::tables::trace_builder::RestHeader {
        table_counts,
        page_configs,
        public_output_bytes,
        ..
    } = header;
    Ok(RestLaid {
        table_counts,
        runtime_page_ranges,
        num_private_input_pages,
        public_output: public_output_bytes,
        airs,
        page_configs,
        refs,
        prepared: None,
        tables: Vec::new(),
        marks,
        slowest,
        busy,
        packed_rest: (narrow_tables, narrow_bytes),
        laid_at,
    })
}

/// `items` in order, cut into consecutive waves of at most `budget` bytes
/// (`bytes` of an item); a wave holds at least one item.
pub(crate) fn waves<T>(items: Vec<T>, budget: usize, bytes: impl Fn(&T) -> usize) -> Vec<Vec<T>> {
    let mut waves: Vec<Vec<T>> = Vec::new();
    let mut held = 0usize;
    for item in items {
        let b = bytes(&item);
        match waves.last_mut() {
            Some(wave) if !wave.is_empty() && held + b <= budget => {
                held += b;
                wave.push(item);
            }
            _ => {
                held = b;
                waves.push(vec![item]);
            }
        }
    }
    waves
}

/// What the layout thread knows once the run is built: the statement, the AIR
/// order's shapes, and each group's tables as AIR indices.
struct Laid {
    table_counts: TableCounts,
    runtime_page_ranges: Vec<RuntimePageRange>,
    num_private_input_pages: usize,
    public_output: Vec<u8>,
    shapes: Vec<(usize, usize)>,
    groups: Vec<Vec<u32>>,
    /// The prepared tables' columns ([`prepared_tables`]).
    prepared: Vec<PreparedColumns>,
    busy: f64,
    layout: LayoutStamps,
    /// Each streamed chunk's table and index, and its AIR index.
    streamed_air: Vec<(StreamTable, usize, usize)>,
}

/// The block's proof with the build streamed into phase A: a builder thread
/// collects the run in windows of `window` cycles ([`WindowedTraceBuilder`])
/// and hands each full chunk over as it completes; a layout thread lays each
/// one out and packs the tables, in arrival order, into groups of at most
/// `format.group_polys` stacked polynomials; this thread commits each group as
/// it closes. When the run is built the rest of its tables join the packing.
/// The groups are the statement's ([`BlockWhirProof::groups`]).
#[allow(clippy::too_many_arguments)]
fn prove_streamed(
    program: &Elf,
    elf_bytes: &[u8],
    private_inputs: &[u8],
    window: usize,
    opts: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    deviations: &Deviations,
    on_statement: StatementObserver<'_>,
    on_group: GroupObserver<'_>,
    stamps: &mut BlockStamps,
) -> Result<BlockWhirProof, Error> {
    // G-pack writes the streamed chunks packed when the groups are held
    // narrow; they are then laid out narrow, as the finish's packed tables.
    let stream_form = if options.gpack && options.narrow != multilinear_block::Narrowing::Wide {
        TraceForm::Narrow
    } else {
        TraceForm::Wide
    };
    gpack::reset_counts();
    // A new block: no memory pressure seen yet (`alloc_purge`).
    crate::alloc_purge::clear_memory_pressure();
    let stream_airs = StreamAirs::new(opts, stream_form);
    // Regeneration (`BlockOptions::regen`): its recorder, when phase A can feed
    // it — its chunks generated packed, and none of KECCAK_RND's or of the
    // MEMW-derived LT ops streamed (chunks no regenerator cuts).
    let regen_mode = match options.regen {
        Some(mode) => mode,
        None => regen::regen_mode()?,
    };
    eprintln!(
        "{}",
        regen::regen_mode_line(
            regen_mode,
            std::env::var("LAMBDA_VM_BLOCK_REGEN").ok().as_deref(),
            std::env::var("LAMBDA_VM_BLOCK_SPILL").ok().as_deref(),
        )
    );
    let regen_recorder = regen::recorder(
        regen_mode,
        stream_form == TraceForm::Narrow && !options.stream_keccak_rnd && !options.stream_memw_lt,
    )
    .map(std::sync::Arc::new);
    let recorder = regen_recorder.as_deref();
    // Live regeneration: phase A drops the streamed chunks the policy moves off
    // the host (`always`: every one). The streamed chunks are placed first, in
    // hand-out order, so the table at block index `t` is the `t`-th chunk
    // handed out when there was one.
    let live_regen = regen_recorder
        .as_ref()
        .filter(|_| regen_mode.live())
        .map(|recorder| {
            let recorder = std::sync::Arc::clone(recorder);
            multilinear_block::BlockRegen::new(
                std::sync::Arc::new(move |t| recorder.chunk_at(t).map(|_| t as u64)),
                regen::ahead_policy().initial(),
                regen_mode == regen::RegenMode::Always,
            )
        });
    let run_airs: std::sync::OnceLock<VmAirs> = std::sync::OnceLock::new();
    // Phase A's commits read the blowup, the fold schedule and the format of
    // the config and nothing else; the full config (its query count reads every
    // shape) is built when the shapes are known, and must agree on those.
    let commit_config = format.chain_config(&[]);
    let cap = commit_config.format.stack;
    let start = Instant::now();
    // A memory log (`BlockOptions::memlog`): its sampler runs until the
    // function returns.
    let logged = options.memlog.then(|| memlog::Ledger::new(start));
    let _sampler = logged.clone().map(memlog::Sampler::start);
    let ledger = logged.as_deref();
    if let Some(ledger) = ledger {
        ledger.thread("prove");
        ledger.pool();
        ledger.line("start");
    }
    // The spill (`BlockOptions::spill`): a store for this prove, unless the
    // policy is off. A store that does not open leaves every table in memory.
    let policy = options.spill;
    let target = deviations.spill_target.unwrap_or_else(spill_target_bytes);
    let policy_name = match policy {
        BlockSpillPolicy::Off => "off".to_string(),
        BlockSpillPolicy::Always => "always".to_string(),
        BlockSpillPolicy::Budget(b) => format!("budget {:.1} GiB", b as f64 / (1u64 << 30) as f64),
        BlockSpillPolicy::Auto => format!(
            "auto (target {:.1} GiB)",
            target as f64 / (1u64 << 30) as f64
        ),
    };
    // The largest host reading `auto` decided on, for the BLOCK SPILL line
    // (#1013's `Spill::host_most`).
    let host_most: std::sync::Arc<std::sync::Mutex<Option<HostReading>>> =
        std::sync::Arc::new(std::sync::Mutex::new(None));
    // Live regeneration with the spill off is no disk: the policy decides as
    // `auto` does, with no store, and a table it would move off the host is
    // dropped if phase B can rebuild it, else stays.
    let no_disk = live_regen.is_some() && policy == BlockSpillPolicy::Off;
    let decide = if no_disk {
        BlockSpillPolicy::Auto
    } else {
        policy
    };
    // The host is read at each decision, whatever the policy: `auto` decides
    // on it and the purge's pressure is noted from it (`note_pressure_past`).
    // Once live regeneration has armed, its phase-B reserve counts.
    let wanted: std::sync::Arc<multilinear_block::SpillWanted> = {
        let host_most = std::sync::Arc::clone(&host_most);
        let armed = live_regen.clone();
        let reserve = regen::regen_reserve_bytes();
        std::sync::Arc::new(move |kept, cells, bytes| {
            let reading = HostReading::now();
            {
                let mut most = host_most.lock().unwrap_or_else(|e| e.into_inner());
                if most.is_none_or(|m| reading.bytes() > m.bytes()) {
                    *most = Some(reading);
                }
            }
            note_pressure_past(reading.bytes(), target);
            let extra = if armed.as_ref().is_some_and(|r| r.is_armed()) {
                reserve
            } else {
                0
            };
            spill_wanted(decide, target, kept, cells, bytes, || {
                reading.bytes().saturating_add(extra)
            })
        })
    };
    let spill = match policy {
        BlockSpillPolicy::Off if live_regen.is_none() => None,
        BlockSpillPolicy::Off => Some(multilinear_block::BlockSpill::new(None, 0, wanted)),
        _ => {
            let store_options = stark::spill::SpillOptions::default();
            let queue = store_options.queue_bytes;
            match stark::spill::SpillStore::open(store_options) {
                Ok(store) => Some(multilinear_block::BlockSpill::new(
                    Some(store),
                    queue,
                    wanted,
                )),
                Err(e) if live_regen.is_some() => {
                    stamps.spill = Some(format!(
                        "{policy_name} · no store ({e}); live regeneration decides with none"
                    ));
                    Some(multilinear_block::BlockSpill::new(None, 0, wanted))
                }
                Err(e) => {
                    stamps.spill =
                        Some(format!("{policy_name} · no store ({e}); every table held"));
                    None
                }
            }
        }
    }
    .map(|spill| match &live_regen {
        Some(regen) => spill.with_regen(regen.clone()),
        None => spill,
    });

    // The streamed finish's byte gate: the rest's tables generated and not yet
    // placed ([`BlockOptions::finish_stream`]).
    let rest_gate = crate::tables::trace_builder::gate::ByteGate::new(
        options.rest_layout_bytes.unwrap_or(usize::MAX),
    );
    let rest_gate = &rest_gate;
    crate::with_whir_hash!(|H| {
        let (block, built, laid, executed) = std::thread::scope(|scope| {
            use std::sync::atomic::Ordering::Relaxed;
            // The executor, a window at a time, two windows ahead of the walk.
            let (ltx, lrx) = std::sync::mpsc::sync_channel::<Vec<executor::vm::logs::Log>>(2);
            let executor = scope.spawn(move || -> Result<f64, Error> {
                if let Some(ledger) = ledger {
                    ledger.thread("executor");
                }
                let mut executor = Executor::new(program, private_inputs.to_vec())
                    .map_err(|e| Error::Execution(format!("{e}")))?;
                while let Some(logs) = executor
                    .resume_with_limit(window)
                    .map_err(|e| Error::Execution(format!("{e}")))?
                {
                    let logs = logs.to_vec();
                    if let Some(ledger) = ledger {
                        // The executor keeps a window of logs of its own.
                        let bytes = memlog::logs_bytes(&logs);
                        ledger.logs.fetch_add(bytes, Relaxed);
                        ledger
                            .exec
                            .store(executor.memory().heap_bytes() + bytes, Relaxed);
                    }
                    if ltx.send(logs).is_err() {
                        break;
                    }
                }
                let executed = start.elapsed().as_secs_f64();
                drop(executor);
                if let Some(ledger) = ledger {
                    ledger.exec.store(0, Relaxed);
                    ledger.line("executor done");
                }
                Ok(executed)
            });
            let (btx, brx) = std::sync::mpsc::sync_channel::<Built>(64);
            let finish_ledger = logged.clone();
            let walk_spill = spill.clone();
            let builder = scope.spawn(move || -> Result<BuilderReport, Error> {
                if let Some(ledger) = ledger {
                    ledger.thread("builder");
                }
                let mut builder =
                    WindowedTraceBuilder::new(program, private_inputs, &options.max_rows)?;
                if options.stream_keccak_rnd {
                    builder = builder.keccak_rnd_chunks(1usize << options.keccak_rnd_rows_log2)?;
                } else if options.finish_keccak_rnd_chunks && options.keccak_rnd_rows_log2 >= 5 {
                    builder = builder
                        .keccak_rnd_chunks_at_finish(1usize << options.keccak_rnd_rows_log2)?;
                }
                if options.finish_cuts {
                    builder = builder.cuts_at_finish(
                        1usize << options.keccak_rows_log2,
                        1usize << options.ecsm_rows_log2,
                        1usize << options.ecdas_rows_log2,
                    )?;
                }
                if options.pack_finished {
                    builder = builder.pack_finished_tables();
                    if options.gpack {
                        builder = builder.generate_packed();
                    }
                }
                if options.stream_memw_lt {
                    builder = builder.stream_memw_lt();
                }
                if options.drop_streamed_ops {
                    builder = builder.drop_streamed_ops()?;
                }
                if !compact_lt() {
                    builder = builder.raw_memw_lt()?;
                }
                if lt_concat() {
                    builder = builder.concat_lt();
                }
                if let Some(ledger) = ledger {
                    ledger.image.store(builder.image_bytes(), Relaxed);
                    ledger.line("builder created");
                }
                let mut streamed = 0usize;
                // The walk on its own thread, doing nothing but walk; this
                // thread appends each walked window, routes it and hands its
                // chunks out as jobs (the layout thread generates them).
                let last = {
                    let (mut walker, mut accumulator) = builder.split();
                    std::thread::scope(|inner| -> Result<Vec<executor::vm::logs::Log>, Error> {
                        let (wtx, wrx) = std::sync::mpsc::sync_channel(2);
                        let walking = inner.spawn(move || -> Result<_, Error> {
                            if let Some(ledger) = ledger {
                                ledger.thread("walker");
                            }
                            // One window held back: only the run's last window
                            // is `finish`'s, and it is the last only once the
                            // executor stops.
                            let mut held: Option<Vec<executor::vm::logs::Log>> = None;
                            for logs in lrx {
                                if let Some(w) = held.replace(logs) {
                                    let walked = walker.walk(&w)?;
                                    if let Some(ledger) = ledger {
                                        ledger.walk.store(walker.state_bytes(), Relaxed);
                                        ledger.logs.fetch_sub(memlog::logs_bytes(&w), Relaxed);
                                        ledger.walked.fetch_add(walked.heap_bytes(), Relaxed);
                                    }
                                    drop(w);
                                    if wtx.send(walked).is_err() {
                                        break;
                                    }
                                }
                            }
                            held.ok_or_else(|| {
                                Error::Execution("the run executed no cycle".to_string())
                            })
                        });
                        for walked in wrx {
                            if let Some(ledger) = ledger {
                                ledger.walked.fetch_sub(walked.heap_bytes(), Relaxed);
                            }
                            // A recorder notes each chunk before it goes, so the
                            // layout's digest finds its recipe.
                            let jobs = match recorder {
                                Some(recorder) => {
                                    let (jobs, added) = accumulator.absorb_counted(walked);
                                    recorder.window(added, &jobs);
                                    jobs
                                }
                                None => accumulator.absorb(walked),
                            };
                            for job in jobs {
                                streamed += 1;
                                if let Some(ledger) = ledger {
                                    ledger.jobs.fetch_add(job.op_bytes(), Relaxed);
                                }
                                if btx.send(Built::Job(Box::new(job))).is_err() {
                                    return Err(Error::Prover("the layout thread stopped".into()));
                                }
                            }
                            if let Some(ledger) = ledger {
                                ledger.builder.store(accumulator.held_bytes(), Relaxed);
                            }
                        }
                        walking
                            .join()
                            .map_err(|_| Error::Prover("the block's walker panicked".into()))?
                    })?
                };
                let windows_done = start.elapsed().as_secs_f64();
                let window_stamps = builder.stamps();
                if let Some(ledger) = ledger {
                    ledger.line("windows walked");
                    ledger.parts("builder", builder.heap_parts());
                    ledger.largest("builder", builder.largest_parts());
                }
                // The walk is done and the block's size known: when the
                // finish's forecast passes the pressure share, the parked
                // tables go to the store now, before its hump builds.
                if let Some(spill) = walk_spill.as_ref().filter(|_| {
                    deviations.hand_off_all || hand_off_from_env()
                }) {
                    let hwm = HostReading::now().hwm;
                    let cycles = (window_stamps.windows as u64).saturating_mul(window as u64);
                    let need = if deviations.hand_off_all {
                        u64::MAX
                    } else {
                        hand_off_need(hwm, cycles, target)
                    };
                    if need > 0 {
                        crate::alloc_purge::note_memory_pressure();
                        let started = spill.hand_off(need, ledger.map(|l| l.block.clone()));
                        let g = |b: u64| b as f64 / (1u64 << 30) as f64;
                        eprintln!(
                            "BLOCK HAND-OFF: at the walk's end ({windows_done:.2} s), VmHWM {:.2} GiB + \
                             {:.2} for {:.1} M cycles walked is past {:.2} GiB ({}/{} of the target) by {} · \
                             parked tables to the store, oldest first: {}",
                            g(hwm),
                            FINISH_GROWTH_PER_CYCLE * cycles as f64 / (1u64 << 30) as f64,
                            cycles as f64 / 1e6,
                            g(target / PRESSURE_SHARE.1 * PRESSURE_SHARE.0),
                            PRESSURE_SHARE.0,
                            PRESSURE_SHARE.1,
                            if need == u64::MAX {
                                "everything (a test's deviation)".to_string()
                            } else {
                                format!("{:.2} GiB", g(need))
                            },
                            if started { "started" } else { "not started" },
                        );
                    }
                }
                // The table phase's marks, for `finish` alone.
                crate::tables::trace_builder::build_stamps::start();
                if let Some(ledger) = finish_ledger {
                    crate::tables::trace_builder::build_stamps::set_hook(Some(
                        std::sync::Arc::new(move |label: &str| {
                            ledger.line(&format!("finish {label}"))
                        }),
                    ));
                }
                // The rest streamed: the header to the layout thread first,
                // then each table as its wave is built, under the byte gate.
                let streams = options.finish_stream
                    && options.layout_workers > 0
                    && options.finish_cuts
                    && (options.stream_keccak_rnd
                        || (options.finish_keccak_rnd_chunks && options.keccak_rnd_rows_log2 >= 5))
                    && !deviations.omit_first_keccak_rnd;
                if streams {
                    let emitted = builder.finish_streamed(&last, rest_gate, |header| {
                        let (tx, rx) = std::sync::mpsc::channel();
                        btx.send(Built::Stream(Box::new(RestStream {
                            header,
                            tables: rx,
                        })))
                        .map_err(|_| Error::Prover("the layout thread stopped".into()))?;
                        Ok(Box::new(
                            move |emitted: crate::tables::trace_builder::Emitted| {
                                if let (Some(ledger), Some(table)) = (ledger, &emitted.table) {
                                    ledger.rest.fetch_add(memlog::rows_bytes(table), Relaxed);
                                }
                                tx.send(emitted).map_err(|_| {
                                    Error::Prover(
                                        "the layout thread stopped taking the rest".into(),
                                    )
                                })
                            },
                        ))
                    });
                    if let Some(ledger) = ledger {
                        crate::tables::trace_builder::build_stamps::set_hook(None);
                        ledger.builder.store(0, Relaxed);
                        ledger.walk.store(0, Relaxed);
                        ledger.image.store(0, Relaxed);
                    }
                    emitted?;
                    let g = |b: usize| b as f64 / (1u64 << 30) as f64;
                    eprintln!(
                        "BLOCK REST STREAM: the rest built in waves under a {:.2} GiB gate, held at most \
                         {:.2} GiB (built and not yet placed)",
                        g(rest_gate.budget()),
                        g(rest_gate.most()),
                    );
                    let finish_marks = crate::tables::trace_builder::build_stamps::take();
                    let finished = start.elapsed().as_secs_f64();
                    if let Some(ledger) = ledger {
                        ledger.line("finish done");
                        ledger.logs.fetch_sub(memlog::logs_bytes(&last), Relaxed);
                    }
                    return Ok((
                        windows_done,
                        finished,
                        streamed,
                        window_stamps,
                        finish_marks,
                    ));
                }
                let built_rest = builder.finish(&last);
                if let Some(ledger) = ledger {
                    crate::tables::trace_builder::build_stamps::set_hook(None);
                    ledger.builder.store(0, Relaxed);
                    ledger.walk.store(0, Relaxed);
                    ledger.image.store(0, Relaxed);
                }
                let mut rest = built_rest?;
                let finish_marks = crate::tables::trace_builder::build_stamps::take();
                split_keccak_rnd(&mut rest, options.keccak_rnd_rows_log2);
                // Cut at the finish already, and held packed: a packed table
                // has no rows for `split_rows` to copy.
                if !options.finish_cuts {
                    split_ecdas(&mut rest, options.ecdas_rows_log2);
                    split_keccak(&mut rest, options.keccak_rows_log2);
                    split_ecsm(&mut rest, options.ecsm_rows_log2);
                }
                if deviations.omit_first_keccak_rnd && options.stream_keccak_rnd {
                    return Err(Error::Prover(
                        "omit_first_keccak_rnd is a non-streamed deviation: a streamed chunk \
                         lands by its index"
                            .into(),
                    ));
                }
                if deviations.omit_first_keccak_rnd && !rest.keccak_rnds.is_empty() {
                    rest.keccak_rnds.remove(0);
                }
                let finished = start.elapsed().as_secs_f64();
                if let Some(ledger) = ledger {
                    ledger.rest.store(rest.main_bytes(), Relaxed);
                    ledger.line("finish done");
                }
                let _ = btx.send(Built::Rest(Box::new(rest)));
                if let Some(ledger) = ledger {
                    // The run's last window goes as this thread ends.
                    ledger.logs.fetch_sub(memlog::logs_bytes(&last), Relaxed);
                }
                Ok((
                    windows_done,
                    finished,
                    streamed,
                    window_stamps,
                    finish_marks,
                ))
            });

            let (gtx, grx) = std::sync::mpsc::sync_channel::<Vec<CommittedTable<'_, F, E>>>(1);
            let stream_airs = &stream_airs;
            let run_airs = &run_airs;
            let layout = scope.spawn(move || -> Result<Laid, Error> {
                if let Some(ledger) = ledger {
                    ledger.thread("layout");
                }
                let packer = Packer {
                    open: Vec::new(),
                    open_shapes: Vec::new(),
                    open_keys: Vec::new(),
                    group_keys: Vec::new(),
                    out: gtx,
                    cap,
                    max_polys: format.group_polys,
                    max_groups: format.max_groups,
                    start,
                    closed_at: Vec::new(),
                    blocked: 0.0,
                    ledger,
                    open_bytes: 0,
                };
                // Off the inline path, the prepared columns wait until the
                // groups are sent.
                let inline = options.layout_workers == 0;
                let rest_of = |rest: RestIn, sink| match rest {
                    RestIn::Whole(traces) => lay_out_rest(
                        traces,
                        program,
                        opts,
                        format,
                        run_airs,
                        start,
                        inline,
                        sink,
                        options.rest_layout_bytes,
                        ledger,
                    ),
                    RestIn::Stream(stream) => {
                        lay_out_rest_streamed(stream, program, opts, run_airs, start, sink, ledger)
                    }
                };
                let (streamed, rest) = if inline {
                    let (streamed, traces) = stream_inline(brx, stream_airs, packer, recorder)?;
                    (streamed, rest_of(RestIn::Whole(traces), None)?)
                } else {
                    stream_pipelined(
                        brx,
                        stream_airs,
                        packer,
                        options.layout_workers,
                        options.layout_ahead,
                        options.pack_rest_as_laid_out,
                        Some(rest_gate),
                        recorder,
                        rest_of,
                    )?
                };
                let t = Instant::now();
                let StreamLaid {
                    mut packer,
                    shapes: streamed_shapes,
                    chunks,
                    placed_at,
                    rest: rest_packed,
                    ahead_most,
                } = streamed;
                let RestLaid {
                    table_counts,
                    runtime_page_ranges,
                    num_private_input_pages,
                    public_output,
                    airs,
                    page_configs,
                    refs,
                    prepared,
                    tables: rest_tables,
                    marks: rest_marks,
                    slowest,
                    busy: rest_busy,
                    packed_rest,
                    laid_at,
                } = rest;
                let names: std::collections::HashMap<String, usize> = refs
                    .iter()
                    .enumerate()
                    .map(|(i, air)| (air.name().to_string(), i))
                    .collect();
                let mut shapes: Vec<Option<(usize, usize)>> = vec![None; refs.len()];
                let mut streamed_air = Vec::with_capacity(streamed_shapes.len());
                for &(table, index, shape) in &streamed_shapes {
                    let at = *names.get(&stream_name(table, index)).ok_or_else(|| {
                        Error::Prover(format!("no AIR for {}", stream_name(table, index)))
                    })?;
                    shapes[at] = Some(shape);
                    streamed_air.push((table, index, at));
                }
                // The tables the windows did not stream, packed in AIR order
                // after every streamed chunk (or already, as they were laid
                // out).
                for &(i, shape) in &rest_packed {
                    if shapes[i].replace(shape).is_some() {
                        return Err(Error::Prover(format!("table {i} built twice")));
                    }
                }
                for (i, shape, table) in rest_tables {
                    if shapes[i].replace(shape).is_some() {
                        return Err(Error::Prover(format!("table {i} built twice")));
                    }
                    packer.place(Key::Air(i), shape, table)?;
                }
                let (group_keys, closed_at, blocked) = packer.finish()?;
                let mut marks = vec![("streamed placed", placed_at)];
                marks.extend(rest_marks);
                marks.push(("groups closed", start.elapsed().as_secs_f64()));
                let prepared = match prepared {
                    Some(prepared) => prepared,
                    None => {
                        let prepared = prepared_tables(airs, &page_configs, format)?;
                        marks.push(("prepared", start.elapsed().as_secs_f64()));
                        if let Some(ledger) = ledger {
                            ledger.prepared.store(prepared_bytes(&prepared), Relaxed);
                        }
                        prepared
                    }
                };
                let shapes: Vec<(usize, usize)> = shapes
                    .into_iter()
                    .enumerate()
                    .map(|(i, s)| s.ok_or_else(|| Error::Prover(format!("table {i} never built"))))
                    .collect::<Result<_, _>>()?;
                for (air, &(width, _)) in refs.iter().zip(&shapes) {
                    if air.trace_layout().0 != width {
                        return Err(Error::Prover(format!(
                            "{}: {width} columns, the AIR declares {}",
                            air.name(),
                            air.trace_layout().0
                        )));
                    }
                }
                let groups = group_keys
                    .into_iter()
                    .map(|keys| {
                        keys.into_iter()
                            .map(|key| match key {
                                Key::Air(i) => Ok(i as u32),
                                Key::Streamed(table, index) => names
                                    .get(&stream_name(table, index))
                                    .map(|&i| i as u32)
                                    .ok_or_else(|| {
                                        Error::Prover(format!(
                                            "no AIR for {}",
                                            stream_name(table, index)
                                        ))
                                    }),
                            })
                            .collect::<Result<Vec<u32>, Error>>()
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let busy = chunks + rest_busy + t.elapsed().as_secs_f64();
                let group_of = |table: usize| {
                    groups
                        .iter()
                        .position(|g| g.contains(&(table as u32)))
                        .unwrap_or(usize::MAX)
                };
                let rest_tables = laid_at
                    .iter()
                    .map(|&(i, at)| (refs[i].name().to_string(), at, group_of(i)))
                    .collect();
                Ok(Laid {
                    table_counts,
                    runtime_page_ranges,
                    num_private_input_pages,
                    public_output,
                    shapes,
                    groups,
                    prepared,
                    busy,
                    layout: LayoutStamps {
                        workers: options.layout_workers,
                        marks,
                        closed_at,
                        blocked,
                        chunks,
                        slowest,
                        ahead_most,
                        packed_rest,
                        rest_tables,
                        gpack: (false, [0; 4]),
                    },
                    streamed_air,
                })
            });

            // A memory log: a group leaves the channel as the committer takes
            // it.
            let block = BlockCommitted::commit_groups_logged::<H>(
                grx.iter().inspect(|group| {
                    if let Some(ledger) = ledger {
                        ledger.sent.fetch_sub(memlog::group_bytes(group), Relaxed);
                    }
                }),
                &commit_config,
                options.drop_levels,
                options.narrow,
                options.upload_ahead,
                ledger.map(|ledger| ledger.block.clone()),
                spill.clone(),
            );
            (block, builder.join(), layout.join(), executor.join())
        });
        stamps.execute =
            executed.map_err(|_| Error::Prover("the block's executor panicked".into()))??;
        // A thread's error names the cause; a commit that ran out of groups only
        // says that it did.
        let (windows_done, finished, streamed, window_stamps, finish_marks) =
            built.map_err(|_| Error::Prover("the block's builder panicked".into()))??;
        stamps.windows = window_stamps;
        stamps.build_marks = finish_marks;
        let laid =
            laid.map_err(|_| Error::Prover("the block's layout thread panicked".into()))??;
        let mut block = block.map_err(|e| Error::Prover(format!("{e:?}")))?;
        if deviations.narrow_width_map && !block.fault_narrow_width_map() {
            return Err(Error::Prover("no narrow table to break".into()));
        }
        if deviations.spilled_byte && !block.fault_spilled_byte() {
            return Err(Error::Prover("no spilled table to break".into()));
        }
        if deviations.spilled_slot_lost && !block.fault_lose_spilled_slot() {
            return Err(Error::Prover("no spilled table to lose".into()));
        }
        stamps.build = finished;
        stamps.streamed = (windows_done, streamed);
        stamps.prep = laid.busy;
        stamps.layout = laid.layout.clone();
        stamps.layout.gpack = (options.gpack, gpack::counts());
        stamps.phase_a = start.elapsed().as_secs_f64();
        if let Some(ledger) = ledger {
            ledger.line("phase A end");
        }
        // Phase A's freed pages back to the OS before phase B allocates: when
        // the block's memory is short (`note_pressure_past`, read once more
        // here), or as `LAMBDA_VM_ALLOC_PURGE` names it.
        note_pressure_past(HostReading::now().bytes(), target);
        crate::alloc_purge::purge_point("phase-a");
        // The shadow regenerator starts here, beside the rest of the prove; it
        // drops nothing, and any exit of the prove stops and joins it.
        let shadow_recipes = regen_recorder
            .as_ref()
            .filter(|r| r.digests())
            .map(|r| r.recipes());
        let shadow = shadow_recipes.as_ref().map(|(recipes, _)| {
            regen::ShadowRun::spawn(
                elf_bytes.to_vec(),
                private_inputs.to_vec(),
                options.max_rows.clone(),
                window,
                recipes.clone(),
                stream_form,
                deviations.regen_drop_one_op,
            )
        });
        // Live regeneration: what phase A dropped, rebuilt beside phase B in
        // rank order. A dropped table is the chunk its rank names (the chunks
        // are placed first, in hand-out order), or the prove stops here.
        let live = match (block.regen().cloned(), &regen_recorder) {
            (Some(live_regen), Some(recorder)) => {
                let drops = live_regen.report();
                let window_of = std::sync::Arc::clone(live_regen.window());
                let run = match live_regen.into_plan() {
                    Some((dropped, producer)) => {
                        let mut keyed = Vec::with_capacity(dropped.len());
                        for (rank, slot) in dropped {
                            let key = recorder
                                .chunk_at(rank as usize)
                                .filter(|key| {
                                    laid.streamed_air
                                        .get(rank as usize)
                                        .is_some_and(|&(t, i, _)| (t, i) == *key)
                                })
                                .ok_or_else(|| {
                                    Error::Prover(format!(
                                        "dropped table {rank} is not the chunk its rank names"
                                    ))
                                })?;
                            keyed.push((key, slot));
                        }
                        Some(regen::LiveRun::spawn(
                            elf_bytes.to_vec(),
                            private_inputs.to_vec(),
                            options.max_rows.clone(),
                            window,
                            std::sync::Arc::clone(&window_of),
                            producer,
                            keyed,
                            stream_form,
                            deviations.regen_faults,
                            regen::ahead_policy(),
                        ))
                    }
                    None => None,
                };
                Some((drops, window_of, run))
            }
            _ => None,
        };
        // Phase B's order: the groups phase B rebuilds last (rest first), so
        // the regenerator has the others' phase B as a head start; the group
        // order when nothing was dropped.
        let rebuilt = block.rebuilt_groups();
        stamps.tables = laid.shapes.len();
        stamps.cells = laid.shapes.iter().map(|&(w, n)| w << n).sum();

        let config = format.chain_config(&laid.shapes);
        if config.log_blowup != commit_config.log_blowup
            || config.log_folding != commit_config.log_folding
            || config.format != commit_config.format
        {
            return Err(Error::Prover(
                "phase A committed under a config the block's shapes do not give".into(),
            ));
        }
        let table_num_vars: Vec<u8> = laid.shapes.iter().map(|&(_, n)| n as u8).collect();
        let mut transcript =
            DefaultTranscript::<E, <H as multilinear::whir_hash::WhirHash>::Transcript>::new(&[]);
        absorb_block(
            &mut transcript,
            elf_bytes,
            &laid.public_output,
            &laid.table_counts,
            laid.num_private_input_pages,
            &laid.runtime_page_ranges,
            &table_num_vars,
            &config,
            &laid.groups,
        );
        let (prepared, tampered, derive) =
            prover_prepared::<H>(laid.prepared, &laid.groups, &config, deviations)?;
        stamps.prepared = (prepared.len(), derive);
        if let Some(ledger) = ledger {
            ledger.line("prepared committed");
        }
        on_statement(
            BlockStatement {
                table_num_vars: &table_num_vars,
                runtime_page_ranges: &laid.runtime_page_ranges,
                table_counts: &laid.table_counts,
                public_output: &laid.public_output,
                num_private_input_pages: laid.num_private_input_pages,
                groups: &laid.groups,
            },
            &prepared_roots(&prepared),
        );
        let borrowed: Vec<Vec<&Mle<F>>> = prepared
            .iter()
            .map(|p| multilinear::stacking::borrow(&p.columns))
            .collect();
        let openings: Vec<multilinear_block::BlockPrepared<'_, F, H>> = prepared
            .iter()
            .zip(&borrowed)
            .map(|(p, columns)| multilinear_block::BlockPrepared {
                group: p.group,
                tables: p.tables.clone(),
                commitment: &p.commitment,
                columns,
            })
            .collect();
        let _ = multilinear::whir_split::take_chain();
        let top_secs_before = multilinear::whir_commit::top_path_secs();
        let t = Instant::now();
        let paths_before = multilinear::whir_commit::top_path_counts();
        let routes_before = multilinear::whir_commit::top_path_routes();
        let widens_before = multilinear::narrow::host_widens();
        let identity = |g: usize| g;
        let fork_of: &dyn Fn(usize) -> usize = match &deviations.fork_of {
            Some(f) => f,
            None => &identity,
        };
        let order: Vec<usize> = match deviations.phase_b_order {
            Some(order) => order(laid.groups.len()),
            None => multilinear_block::rebuilt_last(&rebuilt),
        };
        // The regenerator deposits in rank order, the rebuilt groups' order: an
        // order that takes them otherwise could wait on a deposit its window
        // does not admit yet, so it is refused rather than run.
        let taken: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&g| rebuilt.get(g).copied().unwrap_or(false))
            .collect();
        if taken.windows(2).any(|w| w[0] > w[1]) {
            return Err(Error::Prover(
                "phase B must take the rebuilt groups in group order".into(),
            ));
        }
        let phase_b_start = Instant::now();
        let (proof, argues, prepared_openings, groups) =
            multilinear_block::block_prove_in_order::<_, _, _, H>(
                block,
                &config,
                &mut transcript,
                &openings,
                &tampered,
                fork_of,
                &deviations.argue,
                on_group,
                &order,
            )
            .map_err(|e| Error::Prover(format!("{e:?}")))?;
        stamps.phase_b = t.elapsed().as_secs_f64();
        if let Some((drops, window_of, run)) = live {
            let recipes = recorder.map_or(0, |r| r.recipes().0.len());
            let mut lines = vec![regen::dropped_line(regen_mode, no_disk, &drops, recipes)];
            let offset = run.as_ref().map(|run| {
                phase_b_start
                    .saturating_duration_since(run.started())
                    .as_secs_f64()
            });
            let report = run.map(regen::LiveRun::join);
            if let Some(report) = &report {
                lines.push(report.line());
                lines.extend(report.pacer.clone());
                for failure in report.failures.iter().take(20) {
                    lines.push(format!("BLOCK REGEN FAILED {failure}"));
                }
            }
            lines.push(format!("BLOCK REGEN window: {}", window_of.report()));
            lines.push(format!(
                "BLOCK REGEN phase B order: {} of {} groups rebuilt, taken last · phase B began {} \
                 after the regenerator",
                rebuilt.iter().filter(|&&r| r).count(),
                rebuilt.len(),
                offset.map_or("n/a".to_string(), |s| format!("{s:.2} s")),
            ));
            for line in &lines {
                eprintln!("{line}");
            }
            stamps.regen = Some(regen::RegenStamps {
                recipes,
                dropped: drops.tables,
                dropped_bytes: drops.bytes,
                regenerated: report.as_ref().map_or(0, |r| r.deposited),
                mismatches: report.as_ref().map_or(0, |r| r.mismatches),
                failed: report.as_ref().map_or(0, |r| r.failures.len() + r.skipped),
                error: report.as_ref().and_then(|r| r.error.clone()),
                lines,
                ..regen::RegenStamps::default()
            });
        }
        if let (Some(shadow), Some((recipes, stray))) = (shadow, &shadow_recipes) {
            let prove_offset = phase_b_start
                .saturating_duration_since(shadow.started())
                .as_secs_f64();
            let waited = Instant::now();
            let report = shadow.join();
            let joined = waited.elapsed().as_secs_f64();
            let group_of_air: std::collections::HashMap<u32, usize> = laid
                .groups
                .iter()
                .enumerate()
                .flat_map(|(g, tables)| tables.iter().map(move |&t| (t, g)))
                .collect();
            let air_of: std::collections::HashMap<(StreamTable, usize), usize> = laid
                .streamed_air
                .iter()
                .map(|&(table, index, air)| ((table, index), air))
                .collect();
            let streamed_cells: u64 = laid
                .streamed_air
                .iter()
                .map(|&(_, _, air)| {
                    let (w, n) = laid.shapes[air];
                    (w as u64) << n
                })
                .sum();
            let spans: Vec<(f64, f64)> = groups.iter().map(|g| (g.start_b, g.end_b)).collect();
            let readout = regen::shadow_readout(
                recipes,
                *stray,
                &report,
                &|table, index| {
                    air_of
                        .get(&(table, index))
                        .and_then(|&air| group_of_air.get(&(air as u32)).copied())
                },
                laid.streamed_air.len(),
                streamed_cells,
                &spans,
                stamps.phase_b,
                prove_offset,
                joined,
            );
            for line in &readout.lines {
                eprintln!("{line}");
            }
            stamps.regen = Some(readout);
        }
        if let Some((spill, store)) = spill
            .as_ref()
            .and_then(|spill| Some((spill, spill.store.as_ref()?)))
        {
            let read_back = spill
                .prefetch
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
                .unwrap_or_else(|| "nothing to read back".to_string());
            let stats = store.stats();
            let g = |b: u64| b as f64 / (1u64 << 30) as f64;
            let host = match *host_most.lock().unwrap_or_else(|e| e.into_inner()) {
                Some(h) => format!(
                    " · host at most {:.2} GiB (VmHWM {:.2} · charge {:.2} − inactive file {:.2})",
                    g(h.bytes()),
                    g(h.hwm),
                    g(h.charged),
                    g(h.inactive_file),
                ),
                None => String::new(),
            };
            let hand_off = match spill.hand_off_report() {
                Some(r) => format!(
                    " · hand-off at the walk's end: {} tables {:.2} GiB of {} in {:.2} s",
                    r.tables,
                    g(r.handed),
                    if r.asked == u64::MAX {
                        "every parked one".to_string()
                    } else {
                        format!("{:.2} GiB asked", g(r.asked))
                    },
                    r.secs
                ),
                None => String::new(),
            };
            stamps.spill = Some(format!(
                "{policy_name} · {stats} · read-back {read_back}{host}{hand_off}"
            ));
            stamps.spill_stats = Some(stats);
        } else if spill.is_some() && stamps.spill.is_none() {
            let g = |b: u64| b as f64 / (1u64 << 30) as f64;
            let host = match *host_most.lock().unwrap_or_else(|e| e.into_inner()) {
                Some(h) => format!(" · host at most {:.2} GiB", g(h.bytes())),
                None => String::new(),
            };
            stamps.spill = Some(format!(
                "{policy_name} · no store: live regeneration decides as `auto`, nothing \
                 written{host}"
            ));
        }
        if multilinear::whir_split::enabled() {
            let chain = multilinear::whir_split::take_chain();
            let top_secs = multilinear::whir_commit::top_path_secs();
            let names = [
                "grind",
                "sumcheck",
                "fold",
                "commit_folded",
                "ood",
                "queries",
            ];
            stamps.open_split = names.into_iter().zip(chain.six).collect();
            stamps.open_split.push(("round_wall", chain.round_wall));
            let q = [
                "query_sample",
                "tree_rebuild",
                "coset_gather",
                "open_assemble",
            ];
            stamps.open_split.extend(q.into_iter().zip(chain.queries));
            stamps
                .open_split
                .push(("top_gather", top_secs.0 - top_secs_before.0));
            stamps
                .open_split
                .push(("top_rehash", top_secs.1 - top_secs_before.1));
            stamps.open_split.push(("chains", chain.chain_count as f64));
            stamps.open_split.push(("rounds", chain.round_count as f64));
        }
        let paths_after = multilinear::whir_commit::top_path_counts();
        stamps.top_paths = (
            paths_after.0 - paths_before.0,
            paths_after.1 - paths_before.1,
        );
        let routes_after = multilinear::whir_commit::top_path_routes();
        stamps.top_routes = (
            (
                routes_after.0.0 - routes_before.0.0,
                routes_after.0.1 - routes_before.0.1,
            ),
            (
                routes_after.1.0 - routes_before.1.0,
                routes_after.1.1 - routes_before.1.1,
            ),
        );
        stamps.drop_levels = Some(options.drop_levels);
        let widens_after = multilinear::narrow::host_widens();
        stamps.narrow = options.narrow;
        stamps.host_widens = (
            widens_after.0 - widens_before.0,
            widens_after.1 - widens_before.1,
        );
        stamps.groups = groups;
        if let Some(ledger) = ledger {
            ledger.line("proof done");
            ledger.report();
            stamps.mem_terms = ledger.terms();
        }
        Ok(BlockWhirProof {
            proof,
            table_num_vars,
            runtime_page_ranges: laid.runtime_page_ranges,
            table_counts: laid.table_counts,
            public_output: laid.public_output,
            num_private_input_pages: laid.num_private_input_pages,
            groups: laid.groups,
            prepared: prepared_openings,
            argues,
        })
    })
}

/// The bytes [`absorb_block`] appends, in its order and with its pad: what an
/// in-guest verifier absorbs as one run of program constants. One stream, so a
/// test can hold the two to the same transcript.
pub(crate) fn block_statement_bytes(
    statement: BlockStatement<'_>,
    elf_digest: &[u8; 32],
    config: &multilinear::whir_chain::ChainConfig,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MULTILINEAR_BLOCK_TAG);
    bytes.extend_from_slice(elf_digest);
    bytes.extend_from_slice(&(statement.public_output.len() as u64).to_le_bytes());
    bytes.extend_from_slice(statement.public_output);
    for count in statement::table_count_values(statement.table_counts) {
        bytes.extend_from_slice(&count.to_le_bytes());
    }
    bytes.extend_from_slice(&(statement.num_private_input_pages as u64).to_le_bytes());
    bytes.extend_from_slice(&(statement.runtime_page_ranges.len() as u64).to_le_bytes());
    for range in statement.runtime_page_ranges {
        bytes.extend_from_slice(&range.base.to_le_bytes());
        bytes.extend_from_slice(&range.count.to_le_bytes());
    }
    bytes.extend_from_slice(&(statement.table_num_vars.len() as u64).to_le_bytes());
    bytes.extend_from_slice(statement.table_num_vars);
    for value in [
        config.log_blowup as u64,
        config.fold_word(),
        config.num_queries as u64,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&[config.grind.folding, config.grind.ood, config.grind.query]);
    bytes.resize(bytes.len() + statement::statement_padding(bytes.len()), 0);
    bytes.extend_from_slice(&(statement.groups.len() as u64).to_le_bytes());
    for group in statement.groups {
        bytes.extend_from_slice(&(group.len() as u64).to_le_bytes());
        for &index in group {
            bytes.extend_from_slice(&u64::from(index).to_le_bytes());
        }
    }
    bytes
}

/// The statement's fields: everything of a [`BlockWhirProof`] the verifier
/// reads before any root, which is everything but its argument.
#[derive(Clone, Copy, Debug)]
pub struct BlockStatement<'a> {
    pub table_num_vars: &'a [u8],
    pub runtime_page_ranges: &'a [RuntimePageRange],
    pub table_counts: &'a TableCounts,
    pub public_output: &'a [u8],
    pub num_private_input_pages: usize,
    pub groups: &'a [Vec<u32>],
}

/// A [`BlockStatement`] owned: what a statement observer keeps
/// ([`prove_block_whir_observed`]).
#[derive(Clone, Debug)]
pub struct OwnedBlockStatement {
    pub table_num_vars: Vec<u8>,
    pub runtime_page_ranges: Vec<RuntimePageRange>,
    pub table_counts: TableCounts,
    pub public_output: Vec<u8>,
    pub num_private_input_pages: usize,
    pub groups: Vec<Vec<u32>>,
}

impl OwnedBlockStatement {
    pub fn view(&self) -> BlockStatement<'_> {
        BlockStatement {
            table_num_vars: &self.table_num_vars,
            runtime_page_ranges: &self.runtime_page_ranges,
            table_counts: &self.table_counts,
            public_output: &self.public_output,
            num_private_input_pages: self.num_private_input_pages,
            groups: &self.groups,
        }
    }
}

impl BlockStatement<'_> {
    pub fn to_owned(&self) -> OwnedBlockStatement {
        OwnedBlockStatement {
            table_num_vars: self.table_num_vars.to_vec(),
            runtime_page_ranges: self.runtime_page_ranges.to_vec(),
            table_counts: self.table_counts.clone(),
            public_output: self.public_output.to_vec(),
            num_private_input_pages: self.num_private_input_pages,
            groups: self.groups.to_vec(),
        }
    }
}

impl BlockWhirProof {
    /// The proof's statement.
    pub fn statement(&self) -> BlockStatement<'_> {
        BlockStatement {
            table_num_vars: &self.table_num_vars,
            runtime_page_ranges: &self.runtime_page_ranges,
            table_counts: &self.table_counts,
            public_output: &self.public_output,
            num_private_input_pages: self.num_private_input_pages,
            groups: &self.groups,
        }
    }
}

/// What a verifier derives from the program and a statement before any root,
/// once the statement's checks pass. The host verifier and the recursion's tree
/// plan both start from it, so they check one statement the same way.
pub(crate) struct BlockFrame {
    pub(crate) page_configs: Vec<crate::tables::page::PageConfig>,
    pub(crate) airs: VmAirs,
    /// Per table in [`VmAirs::air_refs`] order: the AIR's width and the stated
    /// height in variables.
    pub(crate) shapes: Vec<(usize, usize)>,
    pub(crate) config: multilinear::whir_chain::ChainConfig,
    /// The proof's table order: AIR indices, the groups concatenated.
    pub(crate) order: Vec<usize>,
    /// Tables per group.
    pub(crate) sizes: Vec<usize>,
    /// Each group's stack and domain, from the shapes and the verifier's stack
    /// cap.
    pub(crate) stack_layouts: Vec<multilinear::stacking::StackedLayout>,
    pub(crate) domains: Vec<multilinear::whir::Domain<F>>,
}

/// The statement's checks, and what they leave: the counts within bounds, the
/// page layout the ELF and the ranges imply, one height per table the counts
/// imply, each height under its cap, the AIR set, the groups an exact partition within the group maximum
/// and each within the stack budget.
pub(crate) fn block_frame(
    statement: BlockStatement<'_>,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
) -> Result<BlockFrame, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;

    validate_block_counts(statement.table_counts)?;
    let max_pages = crate::tables::page::max_private_input_pages();
    if statement.num_private_input_pages > max_pages {
        return Err(Error::InvalidTableCounts(format!(
            "num_private_input_pages ({}) exceeds max ({max_pages})",
            statement.num_private_input_pages,
        )));
    }
    let num_tables = statement.table_num_vars.len();
    let page_configs = Traces::page_configs_from_elf_and_runtime(
        &program,
        statement.runtime_page_ranges,
        statement.num_private_input_pages,
        num_tables,
    )?;
    let Some(expected) = statement
        .table_counts
        .total()
        .and_then(|t| t.checked_add(FIXED_TABLE_COUNT))
        .and_then(|t| t.checked_add(page_configs.len()))
    else {
        return Err(Error::InvalidTableCounts(
            "the table counts do not sum to a usize".to_string(),
        ));
    };
    if expected != num_tables {
        return Err(Error::InvalidTableCounts(format!(
            "the statement implies {expected} tables and states {num_tables} heights",
        )));
    }
    check_table_heights(statement.table_num_vars)?;
    check_chunked_heights(statement.table_counts, statement.table_num_vars)?;

    let airs = VmAirs::new(
        &program,
        proof_options,
        false,
        &page_configs,
        statement.table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let air_refs = airs.air_refs();
    if air_refs.len() != num_tables {
        return Err(Error::InvalidTableCounts(format!(
            "the layout has {} tables, the statement {num_tables}",
            air_refs.len(),
        )));
    }
    // The width is the AIR's, never the proof's; only the height is stated.
    let shapes: Vec<(usize, usize)> = air_refs
        .iter()
        .zip(statement.table_num_vars)
        .map(|(air, &num_vars)| (air.trace_layout().0, num_vars as usize))
        .collect();
    let config = format.chain_config(&shapes);
    drop(air_refs);

    // The groups are the statement's; their stacks are built here, from the
    // shapes and the verifier's stack cap.
    check_group_count(statement.groups.len(), format.max_groups)?;
    let order = validate_groups(statement.groups, shapes.len())?;
    let sizes: Vec<usize> = statement.groups.iter().map(Vec::len).collect();
    let group_shapes: Vec<(usize, usize)> = order.iter().map(|&i| shapes[i]).collect();
    let (stack_layouts, domains) = stacks(&group_shapes, &sizes, &config)?;
    // Every group within the stack budget — the verifier's constant, not the
    // prover's packing — except a table that alone needs more: it cannot be
    // split, so it is a group of its own (the packing's rule too).
    if let Some((g, layout)) = stack_layouts
        .iter()
        .enumerate()
        .find(|(g, layout)| layout.num_polys() > format.group_polys && sizes[*g] > 1)
    {
        return Err(Error::InvalidTableCounts(format!(
            "group {g} stacks into {} polynomials — a group takes at most {}",
            layout.num_polys(),
            format.group_polys
        )));
    }
    Ok(BlockFrame {
        page_configs,
        airs,
        shapes,
        config,
        order,
        sizes,
        stack_layouts,
        domains,
    })
}

/// Verifies a proof from [`prove_block_whir`] under `format` — the verifier's
/// own constants, never the proof's.
pub fn verify_block_whir(
    proof: &BlockWhirProof,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
) -> Result<bool, Error> {
    verify_block_whir_with(
        proof,
        elf_bytes,
        proof_options,
        format,
        false,
        VerifierChecks::ALL,
    )
}

/// [`verify_block_whir`] with the prepared openings left unchecked when
/// `skip_prepared`, and the batched argue's checks and the bus balance switched
/// by `argue_checks` — mutations, for the tests that show each check is what refuses
/// its forgery.
pub(crate) fn verify_block_whir_with(
    proof: &BlockWhirProof,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    skip_prepared: bool,
    argue_checks: VerifierChecks,
) -> Result<bool, Error> {
    // One argument per table, or one per group, by the verifier's format; the
    // other format's messages absent.
    let (tables, argues) = match format.argue {
        ArgueFormat::PerTable => (proof.table_num_vars.len(), 0),
        ArgueFormat::Batched { .. } => (0, proof.groups.len()),
    };
    if proof.proof.tables.len() != tables || proof.argues.len() != argues {
        return Err(Error::InvalidTableCounts(format!(
            "the proof carries {} table arguments and {} batched argues; its format \
             ({:?}) takes {tables} and {argues}",
            proof.proof.tables.len(),
            proof.argues.len(),
            format.argue,
        )));
    }
    let frame = block_frame(proof.statement(), elf_bytes, proof_options, format)?;
    let BlockFrame {
        airs,
        page_configs,
        shapes,
        config,
        order,
        sizes,
        stack_layouts,
        domains,
        ..
    } = &frame;
    let air_refs = airs.air_refs();
    let layouts: Vec<TableLayout<'_, F, E>> = air_refs
        .iter()
        .zip(shapes)
        .map(|(air, &(width, num_vars))| {
            layout_of(*air, width, num_vars)
                .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))
        })
        .collect::<Result<_, _>>()?;
    let preprocessed: Vec<Vec<Mle<F>>> = air_refs
        .iter()
        .map(|air| preprocessed_mles(*air))
        .collect::<Result<_, _>>()?;
    let statements: Vec<TableStatement<'_, F, E>> = layouts
        .iter()
        .zip(&preprocessed)
        .map(|(layout, cols)| layout.statement_with_preprocessed(cols))
        .collect();
    // The proof's tables are in group order, so the statements are taken in
    // that order too.
    let statements: Vec<TableStatement<'_, F, E>> = order.iter().map(|&i| statements[i]).collect();
    let prepared_columns = prepared_tables(airs, page_configs, format)?;

    Ok(crate::with_whir_hash!(|H| {
        let mut transcript =
            DefaultTranscript::<E, <H as multilinear::whir_hash::WhirHash>::Transcript>::new(&[]);
        absorb_block(
            &mut transcript,
            elf_bytes,
            &proof.public_output,
            &proof.table_counts,
            proof.num_private_input_pages,
            &proof.runtime_page_ranges,
            &proof.table_num_vars,
            config,
            &proof.groups,
        );
        // The prepared commitments, derived here from the program: their roots
        // are the verifier's, never the proof's.
        let prepared: Vec<GroupPrepared<H>> = match group_stacks(prepared_columns, &proof.groups)
            .and_then(|stacks| commit_groups::<H>(stacks, config))
        {
            Ok(prepared) => prepared,
            Err(_) => return Ok(false),
        };
        let checks: Vec<multilinear_block::BlockPreparedCheck<'_, F>> = prepared
            .iter()
            .map(|p| multilinear_block::BlockPreparedCheck {
                group: p.group,
                tables: p.tables.clone(),
                roots: &p.roots,
                layout: p.commitment.layout(),
                domain: p.commitment.domain(),
            })
            .collect();
        let derived: Vec<multilinear::whir_commit::Commitment> = checks
            .iter()
            .flat_map(|c| c.roots.iter().copied())
            .collect();
        // What the tables owe: the COMMIT bus's counterparty, at the block's
        // challenges — replayed on a fork through the roots block itself.
        let mut probe = transcript.clone();
        stark::multilinear_table::absorb_roots::<E, _>(&mut probe, &proof.proof.roots, &derived);
        let z: FieldElement<E> = probe.sample_field_element();
        let alpha: FieldElement<E> = probe.sample_field_element();
        let Some(owed) = crate::compute_commit_bus_offset(&proof.public_output, 0, &z, &alpha)
        else {
            return Ok(false);
        };
        multilinear_block::block_verify_with::<_, _, _, H>(
            &proof.proof,
            &proof.argues,
            &proof.prepared,
            &checks,
            &statements,
            stack_layouts,
            domains,
            sizes,
            &owed,
            config,
            &mut transcript,
            skip_prepared,
            argue_checks,
        )
        .is_ok()
    }))
}

#[cfg(test)]
mod spill_policy_tests {
    use super::{
        BlockSpillPolicy, CgroupValue, HostReading, cgroup_memory, hand_off_need, memory_short,
        parse_hand_off, parse_spill_policy, parse_tree_drop, spill_reserve_bytes,
        spill_target_from, spill_wanted,
    };

    const GIB: u64 = 1 << 30;

    /// `LAMBDA_VM_BLOCK_SPILL`'s values, as #1013 reads them: `auto` (and
    /// unset), `off`, `always`, a budget in GiB; anything else is off.
    #[test]
    fn the_spill_knob_reads_as_on_1013() {
        assert_eq!(parse_spill_policy(None), BlockSpillPolicy::Auto);
        assert_eq!(parse_spill_policy(Some("auto")), BlockSpillPolicy::Auto);
        assert_eq!(
            parse_spill_policy(Some(" always ")),
            BlockSpillPolicy::Always
        );
        assert_eq!(parse_spill_policy(Some("off")), BlockSpillPolicy::Off);
        assert_eq!(
            parse_spill_policy(Some("2.5")),
            BlockSpillPolicy::Budget(5 * GIB / 2)
        );
        assert_eq!(parse_spill_policy(Some("-1")), BlockSpillPolicy::Off);
        assert_eq!(parse_spill_policy(Some("nan")), BlockSpillPolicy::Off);
        assert_eq!(BlockSpillPolicy::default(), BlockSpillPolicy::Auto);
    }

    /// The cgroup memory files, v2 and v1, from fake `/proc/self/cgroup` texts
    /// and cgroup trees: v2 at the process's path, v2's `max` falling through
    /// to v1, v1 at its path or (a container without a cgroup namespace) at
    /// the controller's root, a v1 controller sharing its hierarchy, and none
    /// (#1013's test @ 035aef5d6, in a directory of its own).
    #[test]
    fn the_cgroup_memory_files_read_v2_then_v1() {
        let root = std::env::temp_dir().join(format!("im4b-whir-cgroup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let put = |rel: &str, file: &str, value: &str| {
            let dir = root.join(rel);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(file), value).unwrap();
        };
        let read = |proc_cgroup: &str, v2, v1| cgroup_memory(proc_cgroup, &root, v2, v1);
        let limit = |proc_cgroup: &str| {
            read(
                proc_cgroup,
                CgroupValue::File("memory.max"),
                CgroupValue::File("memory.limit_in_bytes"),
            )
        };
        let charge = |proc_cgroup: &str| {
            read(
                proc_cgroup,
                CgroupValue::File("memory.current"),
                CgroupValue::File("memory.usage_in_bytes"),
            )
        };
        let inactive = |proc_cgroup: &str| {
            read(
                proc_cgroup,
                CgroupValue::Stat("inactive_file"),
                CgroupValue::Stat("total_inactive_file"),
            )
        };
        // v2 at the process's path, its charge and its inactive file pages,
        // read by their exact key (a decoy ending in the same name comes
        // first).
        put("a/b", "memory.max", "129584070656\n");
        put("a/b", "memory.current", "4096\n");
        put(
            "a/b",
            "memory.stat",
            "anon 100\nfile 9000\nx_inactive_file 1\nactive_file 10\ninactive_file 8192\n",
        );
        assert_eq!(limit("0::/a/b\n"), Some(129_584_070_656));
        assert_eq!(charge("0::/a/b\n"), Some(4096));
        assert_eq!(inactive("0::/a/b\n"), Some(8192));
        // v2 unlimited and no v1: no limit.
        put("c", "memory.max", "max\n");
        assert_eq!(limit("0::/c\n"), None);
        // A hybrid host, as FAST: the unified line names a path with no
        // memory files, and the memory controller's own cgroup is its root.
        // v1's inactive file pages are its hierarchical key, not the local
        // one that ends in the same name.
        put("memory", "memory.limit_in_bytes", "61774757888\n");
        put("memory", "memory.usage_in_bytes", "17855025152\n");
        put(
            "memory",
            "memory.stat",
            "cache 16000\ninactive_file 5\ntotal_cache 16000\ntotal_inactive_file 15180000000\n",
        );
        let fast = "12:memory:/docker/d005\n9:cpu,cpuacct:/docker/d005\n0::/docker/d005\n";
        assert_eq!(limit(fast), Some(61_774_757_888));
        assert_eq!(charge(fast), Some(17_855_025_152));
        assert_eq!(inactive(fast), Some(15_180_000_000));
        // v1 at the process's path wins over the controller's root, and a
        // controller sharing its hierarchy is found.
        put("memory/docker/d005", "memory.limit_in_bytes", "777\n");
        assert_eq!(limit(fast), Some(777));
        put("memory/p", "memory.limit_in_bytes", "888\n");
        assert_eq!(limit("4:cpu,memory:/p\n"), Some(888));
        // v2's `max` falls through to v1.
        assert_eq!(limit("4:memory:/p\n0::/c\n"), Some(888));
        // Nothing readable.
        assert_eq!(limit(""), None);
        assert_eq!(limit("5:pids:/x\n"), None);
        assert_eq!(inactive("0::/c\n"), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `auto`'s host bytes: the larger of the peak resident set and the
    /// cgroup's working set (its charge less its inactive file pages). FAST at
    /// 17:28Z: 40.58 GB charged with 15.18 GB inactive file reads 25.40 GB, not
    /// 40.58; inactive pages past the charge read zero; the peak resident set
    /// wins when it is larger (#1013's test @ 035aef5d6).
    #[test]
    fn auto_reads_the_hosts_working_set() {
        let reading = |hwm, charged, inactive_file| HostReading {
            hwm,
            charged,
            inactive_file,
        };
        assert_eq!(
            reading(24_680_000_000, 40_580_000_000, 15_180_000_000).bytes(),
            25_400_000_000
        );
        assert_eq!(reading(0, 4 * GIB, 6 * GIB).bytes(), 0);
        assert_eq!(reading(30 * GIB, 40 * GIB, 15 * GIB).bytes(), 30 * GIB);
        assert_eq!(reading(0, 120 * GIB, 0).bytes(), 120 * GIB);
    }

    /// The target is the smaller of the cgroup limit and `MemTotal`, less
    /// 10 GiB: v1's unlimited sentinel gives `MemTotal`'s, a limit above
    /// `MemTotal` is capped, and with neither nothing spills.
    #[test]
    fn the_spill_target_is_the_smaller_limit_less_ten_gib() {
        let total = 62_840_956u64 << 10;
        assert_eq!(
            spill_target_from(Some(61_774_757_888), Some(total)),
            61_774_757_888 - 10 * GIB
        );
        assert_eq!(
            spill_target_from(Some(9_223_372_036_854_771_712), Some(total)),
            total - 10 * GIB
        );
        assert_eq!(
            spill_target_from(Some(200 * GIB), Some(125 * GIB)),
            115 * GIB
        );
        assert_eq!(spill_target_from(None, Some(64 * GIB)), 54 * GIB);
        assert_eq!(
            spill_target_from(Some(129_584_070_656), None),
            129_584_070_656 - 10 * GIB
        );
        assert_eq!(spill_target_from(Some(8 * GIB), Some(16 * GIB)), 0);
        assert_eq!(spill_target_from(None, None), u64::MAX);
    }

    /// `auto` spills a table only once the host, its reserve and the table
    /// would pass the target; it reads the host only then. `off` never
    /// spills, `always` always, a budget once the kept bytes would pass it.
    #[test]
    fn auto_spills_only_past_the_target() {
        let target = 110 * GIB;
        let cells = 10_000_000_000; // 10 G cells: a reserve of 6 + 16.5 GiB
        assert_eq!(spill_reserve_bytes(cells), (22.5 * GIB as f64) as u64);
        let unread = || -> u64 { panic!("only auto reads the host") };
        assert!(!spill_wanted(
            BlockSpillPolicy::Off,
            target,
            0,
            cells,
            GIB,
            unread
        ));
        assert!(spill_wanted(
            BlockSpillPolicy::Always,
            target,
            0,
            cells,
            GIB,
            unread
        ));
        let budget = BlockSpillPolicy::Budget(4 * GIB);
        assert!(!spill_wanted(budget, target, 3 * GIB, cells, GIB, unread));
        assert!(spill_wanted(
            budget,
            target,
            3 * GIB + 1,
            cells,
            GIB,
            unread
        ));
        let auto = BlockSpillPolicy::Auto;
        assert!(!spill_wanted(auto, target, 0, cells, GIB, || 86 * GIB));
        assert!(spill_wanted(auto, target, 0, cells, GIB, || 87 * GIB));
        assert!(!spill_wanted(auto, u64::MAX, 0, cells, GIB, || u64::MAX));
    }

    /// The purge's memory pressure on #1014: the host past 85 % of `auto`'s
    /// target, a share of whatever host it runs on. On BIG (target 110.69 GiB,
    /// share 94.09) the median's highest reading (BIG 653, 86.64 GiB) is not
    /// short and p90's and full gas's (BIG 660, 120.05 and 105.55) are; on a
    /// 64 GiB host (target 54) the share is 45.9; with no limit known, never.
    #[test]
    fn memory_is_short_past_85_percent_of_the_target() {
        let big = 129_584_070_656 - 10 * GIB;
        let share = big / 20 * 17;
        assert!(!memory_short(share, big));
        assert!(memory_short(share + 1, big));
        let gib = |g: f64| (g * GIB as f64) as u64;
        assert!(!memory_short(gib(86.64), big), "the median never purges");
        assert!(memory_short(gib(120.05), big), "p90 purges");
        assert!(memory_short(gib(105.55), big), "full gas purges");
        let small = spill_target_from(None, Some(64 * GIB));
        assert!(!memory_short(gib(45.8), small));
        assert!(memory_short(gib(46.0), small));
        assert!(!memory_short(1024 * GIB, u64::MAX));
    }

    /// The hand-off at the walk's end, against the rows its coefficient comes
    /// from (I-WFULL §4.1; BIG, target 110.69 GiB, line 94.09): p90 (660 D,
    /// VmHWM 85.19 at the walk's end, 584 windows of 2^20 cycles) hands back
    /// ≈ 21.2 GiB, which its forecast puts at the line; full gas (86.81, 575)
    /// ≈ 22.3; the median (653: 66.74, 285), p90 with nothing kept (660 S:
    /// 48.57) and 1× hand back nothing; with no limit known, never. On a
    /// 64 GiB host (target 54, line 45.9) a 40 GiB median walk does.
    #[test]
    fn the_hand_off_forecast_reads_660s_rows() {
        let big = 129_584_070_656 - 10 * GIB;
        let gib = |g: f64| (g * GIB as f64) as u64;
        let as_gib = |b: u64| b as f64 / GIB as f64;
        let windows = |n: u64| n << 20;
        let p90 = as_gib(hand_off_need(gib(85.19), windows(584), big));
        assert!((20.7..21.7).contains(&p90), "p90 hands back {p90:.2} GiB");
        let full = as_gib(hand_off_need(gib(86.81), windows(575), big));
        assert!(
            (21.8..22.8).contains(&full),
            "full gas hands back {full:.2} GiB"
        );
        assert_eq!(
            hand_off_need(gib(66.74), windows(285), big),
            0,
            "the median"
        );
        assert_eq!(
            hand_off_need(gib(48.57), windows(584), big),
            0,
            "p90, nothing kept"
        );
        assert_eq!(hand_off_need(gib(30.0), windows(30), big), 0, "1×");
        assert_eq!(hand_off_need(gib(85.19), windows(584), u64::MAX), 0);
        let small = spill_target_from(None, Some(64 * GIB));
        assert!(hand_off_need(gib(40.0), windows(285), small) > 0);
    }

    /// `LAMBDA_VM_BLOCK_TREE_DROP_LEVELS`: unset is 8 on the card and 4 on the
    /// host; `<card>` or `<card>,<host>` sets them; anything else is the
    /// default.
    #[test]
    fn the_tree_drop_knob_reads_card_then_host() {
        use multilinear::whir_commit::TreeDrop;
        let default = TreeDrop { device: 8, host: 4 };
        assert_eq!(parse_tree_drop(None), default);
        assert_eq!(parse_tree_drop(Some("")), default);
        assert_eq!(parse_tree_drop(Some("6")), TreeDrop { device: 6, host: 4 });
        assert_eq!(
            parse_tree_drop(Some(" 6 , 5 ")),
            TreeDrop { device: 6, host: 5 }
        );
        assert_eq!(parse_tree_drop(Some("4,4")), TreeDrop::uniform(4));
        assert_eq!(parse_tree_drop(Some("x")), default);
        assert_eq!(parse_tree_drop(Some("6,5,1")), default);
    }

    /// `LAMBDA_VM_BLOCK_HAND_OFF`: unset, `auto` or anything else decides at
    /// the walk's end; `off` never hands off.
    #[test]
    fn the_hand_off_knob_reads_off_only() {
        assert!(parse_hand_off(None));
        assert!(parse_hand_off(Some("auto")));
        assert!(parse_hand_off(Some(" on ")));
        assert!(!parse_hand_off(Some("off")));
        assert!(!parse_hand_off(Some(" off ")));
    }
}

#[cfg(test)]
mod rest_order_tests {
    use super::RestOrder;

    fn order() -> RestOrder {
        RestOrder {
            widths: vec![21, 6, 38, 38, 17],
            next: 0,
        }
    }

    /// Every AIR in order, streamed slots among them: accepted.
    #[test]
    fn the_rest_in_air_order_is_taken() {
        let mut o = order();
        for (position, width) in [
            (0, Some(21)),
            (1, Some(6)),
            (2, None),
            (3, Some(38)),
            (4, Some(17)),
        ] {
            o.admit(position, width).expect("in order");
        }
        o.finish().expect("every AIR reached");
    }

    /// A table out of order, of another AIR's width, past the AIRs, or a
    /// stream that stops short: refused, as an error.
    #[test]
    fn a_table_out_of_order_wrong_or_missing_is_refused() {
        let mut o = order();
        o.admit(0, Some(21)).expect("0");
        let err = o.admit(2, Some(38)).expect_err("1 is due");
        assert!(
            format!("{err:?}").contains("came where table 1 was due"),
            "{err:?}"
        );
        let mut o = order();
        let err = o.admit(0, Some(6)).expect_err("BITWISE is 21 wide");
        assert!(
            format!("{err:?}").contains("6 columns, its AIR declares 21"),
            "{err:?}"
        );
        let mut o = order();
        for (position, width) in [(0, Some(21)), (1, Some(6)), (2, None)] {
            o.admit(position, width).expect("in order");
        }
        let err = o.finish().expect_err("two AIRs never came");
        assert!(
            format!("{err:?}").contains("table 3 of the rest was never laid out"),
            "{err:?}"
        );
        let mut o = RestOrder {
            widths: vec![21],
            next: 0,
        };
        o.admit(0, Some(21)).expect("0");
        let err = o.admit(1, Some(6)).expect_err("past the AIRs");
        assert!(
            format!("{err:?}").contains("past the run's 1 tables"),
            "{err:?}"
        );
    }
}
