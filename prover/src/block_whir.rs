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

use crypto::fiat_shamir::is_transcript::IsTranscript;
use executor::elf::Elf;
use executor::vm::engine::{Engine, EngineKind};
use executor::vm::execution::Executor;
use executor::vm::shadow::{ShadowCheck, WindowStart};
use math::field::element::FieldElement;
use multilinear::mle::Mle;
use multilinear::whir_chain::{ArgueFormat, ChainConfig, FinalPoly, GkrLadder, GroupFold};
use rayon::prelude::*;
use stark::multilinear_block::{self, BlockCommitted, GroupStamps, block_groups};
use stark::multilinear_table::{
    BatchedArgue, CommittedTable, MultiProof, TableLayout, TableStatement,
    batched::{LocalArgue, VerifierChecks},
};
use stark::table::Table;
use stark::trace::TraceTable;
use std::time::Instant;

use crate::multilinear_prove::{
    absorb_tagged, chain_config_under, layout_of, preprocessed_mles, shapes_of, stacks,
};
use crate::statement::{self, MULTILINEAR_BLOCK_TAG};
use crate::tables::gpack::{self, TraceForm};
use crate::tables::local_bytes::{self, LocalCounts};
use crate::tables::trace_builder::{
    ChunkJob, CutPadding, RowFamilies, StreamTable, Traces, WindowStamps, WindowedTraceBuilder,
    head_log, parallel,
};
use crate::test_utils::{E, F};
use crate::zf_format::ZfFormat;
use crate::{
    BlockAirInputs, Error, FIXED_TABLE_COUNT, MaxRowsConfig, ProofOptions, RuntimePageRange,
    TableCounts, VmAirs, block_airs,
};
use stark::config::CommitmentHash;
use stark::proof::options::{BaseFormat, CapPolicy};

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

/// The most COPY tables a block statement may declare: a bound on what the
/// verifier builds before anything is checked, far above any block (a p90
/// mainnet block makes 17 at 2^20 rows).
pub const BLOCK_MAX_COPY: usize = 1 << 12;

/// The rows each COPY table holds at most — a prover's choice, under the
/// format's cap [`BLOCK_COPY_MAX_VARS`]. COPY is one row per eight bytes a
/// copy, memset or `write` moves (a row per byte for the last seven), so a
/// median mainnet block makes millions, which one table would pad to the next
/// power of two. A call's rows may straddle two tables: they chain only
/// through the CopyNext bus, keyed by the call's timestamp and its running
/// `(src, dst, count)`. At 2^20 rows a table is 41 columns of 2^20.
pub const BLOCK_COPY_ROWS_LOG2: usize = 20;

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
/// keccak-heavy block at the gas limit makes ≈ 16 at 2^17 rows).
pub const BLOCK_MAX_KECCAK: usize = 1 << 12;

/// The rows each KECCAK table holds at most — a prover's choice, under the
/// format's cap [`BLOCK_KECCAK_MAX_VARS`]. KECCAK is one row per permutation,
/// so a cut falls between permutations; the rows of one Keccak-256 call may
/// straddle two tables, chained only through the KeccakSponge bus keyed by the
/// call's timestamp and the block index. A keccak-heavy block at the gas limit
/// makes ≈ 2^21 permutations, whose one table would stack into many
/// polynomials of a 2^27 stack; at 2^17 rows its 907 columns fit one.
pub const BLOCK_KECCAK_ROWS_LOG2: usize = 17;

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
pub const BLOCK_KECCAK_MAX_VARS: usize = 17;

/// The tallest ECSM table a block statement may state, in variables: a
/// verifier constant ([`check_chunked_heights`]), so no statement makes one
/// ECSM table stack into more than one polynomial of a 2^27 stack.
pub const BLOCK_ECSM_MAX_VARS: usize = 17;

/// The tallest COPY table a block statement may state, in variables: a
/// verifier constant ([`check_chunked_heights`]), the block's chunk height
/// (`BlockOptions::production`'s 2^21).
pub const BLOCK_COPY_MAX_VARS: usize = 21;

/// The tallest CPU table a block statement may state, in variables: a
/// verifier constant ([`block_cpu_heights`]), the block's chunk height
/// (`BlockOptions::production`'s 2^21). The CPU tables' total stays under the
/// clock's bound (`tables::cpu::validate_heights`).
pub const BLOCK_CPU_MAX_VARS: usize = 21;

/// The tallest table a block statement may state, in variables: a verifier
/// constant ([`check_table_heights`]). A group stacks at `want.min(cap).max(tallest)`
/// variables, so one taller table would make its chain taller than the 2^27
/// stack and take its fold phase below the chains' conjectured (Johnson)
/// 130.393 bits — 128.393 under [`GroupFold::Indexed`] — (127.39 at a 2^30
/// table). Honest tables are at most 2^21 rows.
pub const BLOCK_MAX_TABLE_VARS: usize = 27;

const _: () = assert!(BLOCK_KECCAK_RND_ROWS_LOG2 <= BLOCK_KECCAK_RND_MAX_VARS);
const _: () = assert!(BLOCK_ECDAS_ROWS_LOG2 <= BLOCK_ECDAS_MAX_VARS);
const _: () = assert!(BLOCK_KECCAK_ROWS_LOG2 <= BLOCK_KECCAK_MAX_VARS);
// The KECCAK rows a block statement may declare stay under the sponge mode's
// counting bound (`tables::keccak::SPONGE_MAX_ROWS`): raising either cap past it
// fails to compile instead of weakening the argument.
const _: () = assert!(
    (BLOCK_MAX_KECCAK as u128) << BLOCK_KECCAK_MAX_VARS <= crate::tables::keccak::SPONGE_MAX_ROWS
);
const _: () = assert!(BLOCK_ECSM_ROWS_LOG2 <= BLOCK_ECSM_MAX_VARS);
const _: () = assert!(BLOCK_COPY_ROWS_LOG2 <= BLOCK_COPY_MAX_VARS);

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
/// unset or empty, [`BLOCK_TREE_DROP_LEVELS_DEVICE`] and
/// [`BLOCK_TREE_DROP_LEVELS`]. A knob a RAM-adaptive policy can set; the proof
/// does not depend on it. A value it cannot read is the default here and
/// refuses the prove ([`tree_drop_refusal`]), so a mistyped value never runs
/// as the default unnoticed.
pub fn tree_drop_from_env() -> multilinear::whir_commit::TreeDrop {
    let value = std::env::var("LAMBDA_VM_BLOCK_TREE_DROP_LEVELS").ok();
    match value.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
        None => tree_drop_for(ZfFormat::global()),
        Some(v) => parse_tree_drop(Some(v)).unwrap_or(TREE_DROP_DEFAULT),
    }
}

/// Levels below a round-0 coset that the card's coset openings fold from a
/// group's stacks at most (`math_cuda::whir_coset::CosetPlan`: a 256-value
/// tile, then one warp's lanes, 8 + 5). A coset spans `2^(k0 + dropped)`
/// values, so a `2^n` stack needs `k0 + dropped ≥ n − 13`; past that the plan
/// declines and the opening re-encodes every codeword of the run on the card.
pub const CARD_COSET_FOLD_LEVELS: usize = 13;

/// The levels a retired tree drops under `zf` when the knob is unset: the
/// card drops [`BLOCK_TREE_DROP_LEVELS_DEVICE`], or more where the format's
/// first fold is under 6 — enough that a `2^whir_stack` stack's round-0
/// cosets stay within [`CARD_COSET_FOLD_LEVELS`] (first fold 4 at a `2^27`
/// stack: 10). The host drops [`BLOCK_TREE_DROP_LEVELS`]. A prover's choice:
/// the proof does not depend on it, and at the default format (first fold 6)
/// it is today's.
pub fn tree_drop_for(zf: &ZfFormat) -> multilinear::whir_commit::TreeDrop {
    let first_fold = match zf.whir_folds {
        multilinear::whir_chain::WhirFolds::Uniform => {
            crate::zf_format::PRODUCTION_WHIR_LOG_FOLDING
        }
        multilinear::whir_chain::WhirFolds::First(k0) => k0.get(),
    };
    let needed = zf
        .whir_stack
        .get()
        .saturating_sub(CARD_COSET_FOLD_LEVELS + first_fold);
    multilinear::whir_commit::TreeDrop {
        device: BLOCK_TREE_DROP_LEVELS_DEVICE.max(needed),
        host: BLOCK_TREE_DROP_LEVELS,
    }
}

/// The prove's refusal of a `LAMBDA_VM_BLOCK_TREE_DROP_LEVELS` value
/// [`tree_drop_from_env`] cannot read.
fn tree_drop_refusal() -> Result<(), Error> {
    parse_tree_drop(
        std::env::var("LAMBDA_VM_BLOCK_TREE_DROP_LEVELS")
            .ok()
            .as_deref(),
    )
    .map(|_| ())
    .map_err(Error::Prover)
}

/// [`tree_drop_from_env`] unset.
const TREE_DROP_DEFAULT: multilinear::whir_commit::TreeDrop = multilinear::whir_commit::TreeDrop {
    device: BLOCK_TREE_DROP_LEVELS_DEVICE,
    host: BLOCK_TREE_DROP_LEVELS,
};

fn parse_tree_drop(value: Option<&str>) -> Result<multilinear::whir_commit::TreeDrop, String> {
    let Some(value) = value.map(str::trim).filter(|v| !v.is_empty()) else {
        return Ok(TREE_DROP_DEFAULT);
    };
    let mut parts = value.split(',').map(|p| p.trim().parse::<usize>().ok());
    match (parts.next().flatten(), parts.next(), parts.next()) {
        (Some(device), None, None) => Ok(multilinear::whir_commit::TreeDrop {
            device,
            host: TREE_DROP_DEFAULT.host,
        }),
        (Some(device), Some(Some(host)), None) => {
            Ok(multilinear::whir_commit::TreeDrop { device, host })
        }
        _ => Err(format!(
            "LAMBDA_VM_BLOCK_TREE_DROP_LEVELS={value}: <card levels> or <card levels>,<host levels>"
        )),
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
    /// How the BLAKE3 base's chains check their successors ([`B3Rounds`]):
    /// read by [`Self::chain_config`] and [`block_format_tag`] under the
    /// BLAKE3 base alone. `Fri` in every other format.
    pub b3_rounds: B3Rounds,
    /// The most groups a block statement may declare (the verifier refuses
    /// more, and the prover refuses as its groups close). The proven bits are
    /// quoted at this count (I-NOEPOCH-W.md §8).
    pub max_groups: usize,
    /// Whether DECODE's and the dense genesis pages' columns are settled by
    /// prepared openings ([`prepared_tables`]) — the production format, which
    /// the recursion needs — or evaluated by the host verifier. Off exists to
    /// measure what the openings cost the base.
    pub prepared: bool,
    /// How big the recursion's leaves are
    /// ([`crate::lfm::whir_block::LeafScale`]): the recursion's alone — the
    /// block proof does not read it — and, like the rest, a constant the
    /// prover and the verifier each hold.
    pub leaf_scale: crate::lfm::whir_block::LeafScale,
    /// How the recursion's leaf checks each chain's queries
    /// ([`crate::lfm::whir_chain::LeafQuery`]): the recursion's alone, like
    /// [`Self::leaf_scale`].
    pub leaf_query: crate::lfm::whir_chain::LeafQuery,
    /// How each group's tables argue: one per group — a lockstep GKR per bin
    /// of its trees and one constraint sumcheck over its tables
    /// ([`ArgueFormat::Batched`], production: [`ArgueFormat::BATCHED`]) — or
    /// one argument per table ([`ArgueFormat::PerTable`], a format to measure
    /// against). Under the batched format the proof carries each group's argue
    /// in [`BlockWhirProof::argues`].
    pub argue: ArgueFormat,
    /// How each group's stacked polynomials are opened: one chain per run of
    /// at most four ([`GroupFold::Indexed`], production), or
    /// one chain each ([`GroupFold::PerPoly`]). A caller constant like the rest
    /// of the format, bound into the statement through the fold word.
    pub group: GroupFold,
    /// Which byte checks KECCAK_RND sends
    /// ([`KeccakRndForm`](crate::tables::keccak_rnd::KeccakRndForm)): without
    /// the 20 checks its D lookups imply in every preset
    /// ([`Lean`](crate::tables::keccak_rnd::KeccakRndForm::Lean)), or every
    /// check ([`Full`](crate::tables::keccak_rnd::KeccakRndForm::Full), the
    /// variant a caller names). Its interactions, so its bus and its GKR trees,
    /// follow it.
    pub keccak_rnd: crate::tables::keccak_rnd::KeccakRndForm,
    /// How MEMW_A proves each access's old timestamp below its timestamp
    /// ([`MemwAlignedOrder`](crate::tables::memw_aligned::MemwAlignedOrder)):
    /// the step's two 16-bit limbs in the row
    /// ([`Limbs`](crate::tables::memw_aligned::MemwAlignedOrder::Limbs), the
    /// production constant), or one LT lookup a row
    /// ([`Lt`](crate::tables::memw_aligned::MemwAlignedOrder::Lt), the format
    /// before it, a variant a caller names). Its interactions, so its bus, the
    /// LT tables the block holds and the recursion's programs, follow it.
    pub memw_aligned: crate::tables::memw_aligned::MemwAlignedOrder,
    /// How register accesses are checked
    /// ([`RegisterFormat`](crate::tables::register::RegisterFormat)): in the
    /// CPU row on one 64-bit token an access, the residual register table
    /// taking the rest
    /// ([`InRow`](crate::tables::register::RegisterFormat::InRow), the
    /// production constant), or as 32-bit word tokens through MEMW_R
    /// ([`Tokens32`](crate::tables::register::RegisterFormat::Tokens32), the
    /// format before it, a variant a caller names). The CPU, MEMW_R's slot,
    /// REGISTER and HALT, so their buses, the block's tables and the
    /// recursion's programs, follow it.
    pub registers: crate::tables::register::RegisterFormat,
    /// How the CPU's loads and stores are checked
    /// ([`MemoryFormat`](crate::tables::fused_memory::MemoryFormat)): an
    /// access MEMW_A would take as one MEM_LOAD or MEM_STORE row holding its
    /// own memory tokens
    /// ([`Fused`](crate::tables::fused_memory::MemoryFormat::Fused), the
    /// production format's, which the base proves), or each a LOAD or STORE
    /// row and a MEMW message to MEMW_A or MEMW
    /// ([`Split`](crate::tables::fused_memory::MemoryFormat::Split), the
    /// pinned preset's, the recursion's tree's default, and the base's
    /// rollback; the tree also takes the other two, each named to its
    /// verifier). The block's tables, so its buses and the recursion's
    /// programs, follow it.
    pub memory: crate::tables::fused_memory::MemoryFormat,
    /// Which relations ECDAS proves its steps with
    /// ([`EcdasForm`](crate::tables::ecdas::EcdasForm)): at 16-bit limb
    /// positions ([`Halves`](crate::tables::ecdas::EcdasForm::Halves), the
    /// production and pinned constant), or at byte positions
    /// ([`Bytes`](crate::tables::ecdas::EcdasForm::Bytes), the form before
    /// it, a variant a caller names). Its columns, constraints and interactions, so its GKR
    /// trees and the recursion's programs, follow it.
    pub ecdas: crate::tables::ecdas::EcdasForm,
    /// How many pages a page table holds under the word memory format
    /// ([`page_tables`](crate::tables::page_w::page_tables)): `k` in
    /// `1..=`[`MAX_PAGE_RUN_LOG2`](crate::tables::page_w::MAX_PAGE_RUN_LOG2),
    /// each run of contiguous zero pages, or of private-input pages, cut into
    /// blocks of up to `2^k` pages aligned to their span, a table each, a page
    /// of program data still a table of its own ([`BLOCK_PAGE_RUN_LOG2`], runs
    /// of 16 pages, every preset's); `0`, one PAGE_W table a page (the format
    /// before it, a variant a caller names). A constant the prover and the
    /// verifier of one block both take, never read from a proof: the page
    /// tables, so the table count, their heights and the recursion's
    /// programs, follow it. It cuts the word memory format's pages alone: under
    /// the split and the fused formats every page is a PAGE table of its own
    /// whatever it says ([`Self::page_runs`]).
    pub page_run_log2: u8,
}

/// The page runs of every block preset ([`BlockFormat::page_run_log2`]): runs
/// of zero and private-input pages cut into blocks of up to `2^4 = 16` pages
/// under the word memory format.
pub const BLOCK_PAGE_RUN_LOG2: u8 = 4;

/// The most groups a block statement may declare: a verifier constant, never in
/// the transcript. The 1× block has 9 and the median under 100; a 200 Mgas
/// block makes some 600 to 925. The bound caps what a statement can make the
/// verifier build, and the block-wide LogUp support: its worst case at 1024
/// groups is pinned by `block_wide_logup_at_the_group_maximum_is_pinned`.
pub const BLOCK_MAX_GROUPS: usize = 1024;

/// The most tables a block statement may state: a verifier constant, never in
/// the transcript. The large block states under 700 tables at runs off, and a
/// 200 Mgas block, by the estimate behind [`BLOCK_MAX_GROUPS`], some 13
/// thousand. The frame
/// builds an AIR for every stated table before any root is read, so the bound
/// caps what a statement can make the verifier build: [`block_frame`] refuses
/// a statement with more before it loads the program, and the prover refuses
/// an AIR set with more, so a block the verifier would refuse is not proved.
pub const BLOCK_MAX_TABLES: usize = 1 << 16;

/// Refuses `tables` over [`BLOCK_MAX_TABLES`].
pub(crate) fn check_table_count(tables: usize) -> Result<(), Error> {
    if tables > BLOCK_MAX_TABLES {
        return Err(Error::InvalidTableCounts(format!(
            "{tables} tables — a block states at most {BLOCK_MAX_TABLES}"
        )));
    }
    Ok(())
}

/// Refuses `groups` over `max_groups` ([`BlockFormat::max_groups`]). The
/// verifier refuses a statement with more ([`block_frame`]), and the prover
/// refuses as its groups close, so a block the verifier would refuse is not
/// proved.
pub(crate) fn check_group_count(groups: usize, max_groups: usize) -> Result<(), Error> {
    if groups > max_groups {
        return Err(Error::InvalidTableCounts(format!(
            "{groups} groups — a block takes at most {max_groups}"
        )));
    }
    Ok(())
}

/// The tallest 4-ary cap a Poseidon1 base may name: the trees clamp a taller
/// one (`multilinear::whir_commit::tree_cap_height`), so [`checked_base`]
/// refuses it rather than let the tag name a height no tree has.
pub const MAX_P1_CAP_HEIGHT: usize = crypto::merkle_tree::cap::MAX_CAP_HEIGHT / 2;

/// The block's base, or a typed refusal: exactly [`crate::hash_pin::BLOCK_BASE`],
/// the block proof family's, whose configuration [`crate::with_block_hash!`]
/// binds. The block proves and verifies one family end to end (D-P3B §2.1),
/// so any other base — RPX's included — is refused, never dispatched: no base
/// of one hash under another's recursion.
pub fn checked_base(base: &BaseFormat) -> Result<(), Error> {
    let family = crate::hash_pin::BLOCK_BASE;
    if *base != family {
        return Err(Error::Prover(format!(
            "a block's base is its proof family's ({:?}, 4-ary cap {}), not {:?} (cap {})",
            family.hash, family.arity4_cap, base.hash, base.arity4_cap
        )));
    }
    Ok(())
}

/// The base of a block proved with NO recursion over it
/// ([`crate::base_only`]), or a typed refusal: [`checked_base`]'s base, or —
/// once the process declared base-only — [`BaseFormat::B3_WHIR`], the
/// opt-in 7-round BLAKE3 base. Read by the base's own prove and verify
/// sites alone ([`crate::with_base_hash!`]); the recursion admits through
/// [`checked_base`], which refuses BLAKE3 whatever this process declared.
pub fn checked_base_only(base: &BaseFormat) -> Result<(), Error> {
    if *base == BaseFormat::B3_WHIR {
        if crate::base_only::admitted() {
            return Ok(());
        }
        return Err(Error::Prover(
            "a BLAKE3 base is proved only with no recursion over it: this process did not              declare base-only (prove-block-base)"
                .to_string(),
        ));
    }
    checked_base(base)
}

// The block's base is Poseidon1 at a cap its trees run at.
const _: () = {
    assert!(matches!(
        crate::hash_pin::BLOCK_BASE.hash,
        CommitmentHash::Poseidon1
    ));
    if let CapPolicy::Fixed(c) = crate::hash_pin::BLOCK_BASE.arity4_cap {
        assert!(c as usize <= MAX_P1_CAP_HEIGHT);
    }
};

/// The block statement's leading tag under `base`'s geometry:
/// `LAMBDAVM_MULTILINEAR_BLOCK_STATEMENT_V2/P1W16/C<h>`, the effective cap:
/// `C0` uncapped (`Off` and `Fixed(0)`), `C<h>` at a fixed height and `Cauto`
/// under `Auto`, so no two geometries share a tag (REV-P1-JUDGE F2). The hash
/// part is the block family's (`hash_pin::BLOCK_BASE`, Poseidon1 at width
/// 16): every site that absorbs a statement admits the base through
/// [`checked_base`] first, which refuses any other. Domain separation only.
pub fn block_statement_tag(base: &BaseFormat) -> Vec<u8> {
    #[cfg(test)]
    if P1_TAG_OMITTED.with(std::cell::Cell::get) {
        return MULTILINEAR_BLOCK_TAG.to_vec();
    }
    // The 7-round BLAKE3 base's binary trees: their cap is the base's own,
    // [`b3_base_cap`], a verifier-side constant, and the tag names it so no
    // two BLAKE3 geometries share a tag either; then its final polynomial's
    // width ([`b3_base_final`], `F0` without one).
    if *base == BaseFormat::B3_WHIR {
        let mut tag = MULTILINEAR_BLOCK_TAG.to_vec();
        tag.extend_from_slice(B3_BASE_TAG_SUFFIX);
        tag.extend_from_slice(format!("/C{}", b3_base_cap()).as_bytes());
        let final_vars = match b3_base_final() {
            FinalPoly::Off => 0,
            FinalPoly::Vars(v) => v,
        };
        tag.extend_from_slice(format!("/F{final_vars}").as_bytes());
        return tag;
    }
    let height = match base.arity4_cap {
        CapPolicy::Off | CapPolicy::Fixed(0) => "0".to_string(),
        CapPolicy::Fixed(c) => c.to_string(),
        CapPolicy::Auto => "auto".to_string(),
    };
    let mut tag = MULTILINEAR_BLOCK_TAG.to_vec();
    tag.extend_from_slice(format!("/P1W16/C{height}").as_bytes());
    tag
}

/// The 7-round BLAKE3 base's statement tag after
/// `LAMBDAVM_MULTILINEAR_BLOCK_STATEMENT_V2` (see [`block_statement_tag`]),
/// before its cap height.
pub const B3_BASE_TAG_SUFFIX: &[u8] = b"/B3R7";

/// The 7-round BLAKE3 base's binary cap height: every tree of its chains is
/// cut to a cap of `2^8` nodes (or to the tree itself, when it is shallower),
/// whatever the process's binary cap knob says. A verifier-side constant,
/// named in the statement tag ([`block_statement_tag`]).
///
/// ★ Why 8 and not the binary default's height (3 at production): a tree a
/// query phase opens `Q = 229` times (round 0) or `2Q` (a successor) pays one
/// parent a walked level an opening, against `2^c − 1` parents for its cap
/// once; past about 8 the cap costs more than the levels it saves. The base's
/// proof carries the cap's nodes (≈ 8 KB a tree).
pub const B3_BASE_CAP: u8 = 8;

/// The knob that proves and verifies the BLAKE3 base at another binary cap
/// height than [`B3_BASE_CAP`] (a measurement: the cap's in-guest mux is
/// `2^c − 1` `Select` rows a digest word an opening, so the recursion's cells
/// and its compressions move in opposite directions with `c`). Both sides of
/// one process read it, and the tag names it: see [`b3_base_cap_setting`].
pub const B3_BASE_CAP_ENV: &str = "LAMBDA_VM_B3_BASE_CAP";

/// [`B3_BASE_CAP_ENV`]'s value: unset is [`B3_BASE_CAP`] (the base's format);
/// a height `3` to `8` is that height. Anything else is refused.
pub fn b3_base_cap_setting(value: Option<&str>) -> Result<u8, String> {
    match value.map(str::trim) {
        None => Ok(B3_BASE_CAP),
        Some(v) => match v.parse::<u8>() {
            Ok(c @ 3..=8) => Ok(c),
            _ => Err(format!(
                "{B3_BASE_CAP_ENV}={v:?}: expected a height from 3 to 8"
            )),
        },
    }
}

/// The BLAKE3 base's binary cap height in this process
/// ([`b3_base_cap_setting`] of [`B3_BASE_CAP_ENV`]), read once; a refused
/// value aborts the process at its first read, as [`b3_base_final`] does.
pub fn b3_base_cap() -> u8 {
    static CAP: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let value = match std::env::var(B3_BASE_CAP_ENV) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(v)) => {
                eprintln!("{B3_BASE_CAP_ENV}={v:?}: expected a height from 3 to 8");
                std::process::abort()
            }
        };
        b3_base_cap_setting(value.as_deref()).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::abort()
        })
    })
}

/// The 7-round BLAKE3 base's final polynomial: each chain stops committing at
/// the first round past round 0, short of the last, after which at most `9`
/// variables remain (`ChainConfig::final_round`; round 3 of a `2^27` stack's
/// `[6, 4, 4, 4, 4, 4, 1]`). That round sends the folded polynomial's `2^9`
/// values, checks its queries against their encoding, and the rounds after it
/// are sumchecks alone. A verifier-side constant (the base's
/// [`b3_base_final`] unless [`B3_BASE_FINAL_ENV`] says otherwise), named in the
/// statement tag ([`block_statement_tag`]).
///
/// ★ Why: the recursion's leaf opens no successor at the final round and no
/// tree after it, for `2^9` values absorbed once and their encoding a query
/// (`lfm::whir_chain::emit_final_poly_phase`).
pub const B3_BASE_FINAL_VARS: u8 = 9;

/// The knob that proves and verifies the BLAKE3 base's chains with or without
/// their final polynomial (a measurement of what it costs the base; both sides
/// of one process read it, and the tag names it): see [`b3_base_final_setting`].
pub const B3_BASE_FINAL_ENV: &str = "LAMBDA_VM_B3_BASE_FINAL";

/// [`B3_BASE_FINAL_ENV`]'s value: unset, `on` or `9` is [`FinalPoly::Vars`] at
/// [`B3_BASE_FINAL_VARS`] (the base's format); `0` or `off` is
/// [`FinalPoly::Off`], every round committed. Anything else is refused.
pub fn b3_base_final_setting(value: Option<&str>) -> Result<FinalPoly, String> {
    let on = FinalPoly::Vars(B3_BASE_FINAL_VARS);
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        None => Ok(on),
        Some(v) => match v.as_str() {
            "on" | "9" => Ok(on),
            "0" | "off" => Ok(FinalPoly::Off),
            other => Err(format!(
                "{B3_BASE_FINAL_ENV}={other:?}: expected on, 9, off or 0"
            )),
        },
    }
}

/// The BLAKE3 base's final polynomial in this process ([`b3_base_final_setting`]
/// of [`B3_BASE_FINAL_ENV`]), read once. A value it refuses aborts the process
/// at its first read, as the process format does (`ZfFormat::global`): a
/// configuration error, before anything is proved or verified.
pub fn b3_base_final() -> FinalPoly {
    static FINAL: std::sync::OnceLock<FinalPoly> = std::sync::OnceLock::new();
    *FINAL.get_or_init(|| {
        let value = match std::env::var(B3_BASE_FINAL_ENV) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(v)) => {
                eprintln!("{B3_BASE_FINAL_ENV}={v:?}: expected on, 9, off or 0");
                std::process::abort()
            }
        };
        b3_base_final_setting(value.as_deref()).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::abort()
        })
    })
}

/// How this process's WHIR final rounds read their message so far,
/// `(host, one copy off a card, per point)`
/// (`multilinear::whir_chain::final_message_reads`): what a base-only run
/// prints, so a final polynomial's arm shows the path its reads took.
pub fn final_message_reads() -> (usize, usize, usize) {
    multilinear::whir_chain::final_message_reads()
}

/// The 7-round BLAKE3 base's group size: at most `4` stacked polynomials a
/// group (Poseidon1's is [`BLOCK_GROUP_POLYS`], three). A quarter fewer
/// groups for the recursion to verify, each a run of four polynomials in one
/// chain: about 10.7 k compressions a polynomial against 13.7 k in a run of
/// three at the binary default cap (by closed form). The production format's
/// ([`BlockFormat::production`] under the BLAKE3 base, the base's
/// [`b3_base_group_polys`] unless [`B3_BASE_GROUP_ENV`] says otherwise); the
/// statement tag names a BLAKE3 format's group size ([`block_format_tag`]).
pub const B3_BASE_GROUP_POLYS: usize = 4;

/// The knob that proves and verifies the BLAKE3 base at three polynomials a
/// group instead of [`B3_BASE_GROUP_POLYS`] (a measurement of what four cost
/// the base; both sides of one process read it, and the tag names it): see
/// [`b3_base_group_setting`].
pub const B3_BASE_GROUP_ENV: &str = "LAMBDA_VM_B3_BASE_GROUP";

/// [`B3_BASE_GROUP_ENV`]'s value: unset or `4` is [`B3_BASE_GROUP_POLYS`]
/// (the base's format); `3` is Poseidon1's group size. Anything else is
/// refused.
pub fn b3_base_group_setting(value: Option<&str>) -> Result<usize, String> {
    match value.map(str::trim) {
        None | Some("4") => Ok(B3_BASE_GROUP_POLYS),
        Some("3") => Ok(3),
        Some(other) => Err(format!("{B3_BASE_GROUP_ENV}={other:?}: expected 4 or 3")),
    }
}

/// The BLAKE3 base's group size in this process ([`b3_base_group_setting`]
/// of [`B3_BASE_GROUP_ENV`]), read once; a refused value aborts the process
/// at its first read, as [`b3_base_final`] does.
pub fn b3_base_group_polys() -> usize {
    static GROUP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GROUP.get_or_init(|| {
        let value = match std::env::var(B3_BASE_GROUP_ENV) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(v)) => {
                eprintln!("{B3_BASE_GROUP_ENV}={v:?}: expected 4 or 3");
                std::process::abort()
            }
        };
        b3_base_group_setting(value.as_deref()).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::abort()
        })
    })
}

/// How the 7-round BLAKE3 base's chains check the successors they commit
/// ([`BlockFormat::b3_rounds`]).
///
/// [`Fri`](Self::Fri) is today's: every query opens the successor too
/// (`multilinear::whir_chain::RoundForm::Fri`). [`Whir`](Self::Whir) is the
/// WHIR paper's round (`RoundForm::Whir`): the successor is the bound message
/// encoded on a domain that shrinks by `shrink` bits a round (the rate rises
/// by `k − shrink` bits a round), it is never opened, and each query's fold
/// value is a claim on it at the folded domain's point; each round queries the
/// count for its own rate under the base's target. Its trees' cap rule is part
/// of it: caps by Q (each tree's cap held to what its own openings pay for,
/// under the base's cap as the ceiling — production) or the base's cap at
/// every tree.
///
/// ★ A FORMAT VALUE. The prover and the verifier of one block take it from
/// their [`BlockFormat`] (production: [`b3_base_rounds`], the base's knob;
/// the pinned verifier: `Fri`); it reaches the proof through the chains'
/// config ([`BlockFormat::chain_config`], bound by the fold word) and the
/// statement tag (`/W<shrink>` and the cap rule, `/Cq` for caps by Q, both
/// from [`B3Rounds::tag_word`], [`block_format_tag`]). Never read from a
/// proof. The BLAKE3 base's alone: a Poseidon1 format's chains are `Fri`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum B3Rounds {
    /// Every query opens the successor. Today's.
    #[default]
    Fri,
    /// The WHIR round form at this shrink (1 to 3); each tree's cap by its
    /// own query count when `caps_by_q`, else the base's cap at every tree.
    Whir { shrink: u8, caps_by_q: bool },
}

impl B3Rounds {
    /// The round form the design picked: shrink 3 under four variables a
    /// round, so the rate doubles every round after round 0.
    /// Its trees' caps by Q.
    pub const WHIR3: Self = Self::Whir {
        shrink: 3,
        caps_by_q: true,
    };

    /// The statement tag's word for it, the round form and its cap rule in one
    /// place: nothing for `Fri`; `/W<shrink>/Cq` for the round form with caps
    /// by Q, `/W<shrink>` with the base's cap at every tree (whose height the
    /// tag already names, `/C<h>`).
    pub fn tag_word(self) -> String {
        match self {
            Self::Fri => String::new(),
            Self::Whir {
                shrink,
                caps_by_q: true,
            } => format!("/W{shrink}/Cq"),
            Self::Whir {
                shrink,
                caps_by_q: false,
            } => format!("/W{shrink}"),
        }
    }
}

/// The stack height the BLAKE3 base's round form charges its union bound
/// for ([`BlockFormat::chain_config`]): no chain of a block is taller — a
/// group stacks at its cap ([`multilinear::whir_chain::StackVars::WIDEST`]
/// at most) or at its tallest table ([`BLOCK_MAX_TABLE_VARS`] at most, a
/// verifier constant), whichever is taller — and a chain that were is refused
/// by the chains themselves (`check_round_form_union`).
pub const B3_ROUND_FORM_STACK_BOUND: usize = 27;
const _: () = assert!(B3_ROUND_FORM_STACK_BOUND >= BLOCK_MAX_TABLE_VARS);
const _: () = assert!(B3_ROUND_FORM_STACK_BOUND >= multilinear::whir_chain::StackVars::WIDEST);

/// The knob that proves and verifies the BLAKE3 base under the WHIR round
/// form ([`B3Rounds`]); both sides of one process read it, and the tag names
/// it: see [`b3_base_rounds_setting`]. Read here, never by the format module
/// (`zf_format`), whose knobs a pinned verifier refuses.
pub const B3_BASE_ROUNDS_ENV: &str = "LAMBDA_VM_B3_BASE_ROUNDS";

/// [`B3_BASE_ROUNDS_ENV`]'s value: unset, `0` or `fri` is [`B3Rounds::Fri`]
/// (today's); `1` or `w3` is [`B3Rounds::WHIR3`] (shrink 3, caps by Q). Only
/// these two words turn it on: anything else is refused.
pub fn b3_base_rounds_setting(value: Option<&str>) -> Result<B3Rounds, String> {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        None => Ok(B3Rounds::Fri),
        Some(v) => match v.as_str() {
            "0" | "fri" => Ok(B3Rounds::Fri),
            "1" | "w3" => Ok(B3Rounds::WHIR3),
            other => Err(format!(
                "{B3_BASE_ROUNDS_ENV}={other:?}: expected 0, fri, 1 or w3"
            )),
        },
    }
}

/// The BLAKE3 base's round form in this process ([`b3_base_rounds_setting`]
/// of [`B3_BASE_ROUNDS_ENV`]), read once; a refused value aborts the process
/// at its first read, as [`b3_base_final`] does.
pub fn b3_base_rounds() -> B3Rounds {
    static ROUNDS: std::sync::OnceLock<B3Rounds> = std::sync::OnceLock::new();
    *ROUNDS.get_or_init(|| {
        let value = match std::env::var(B3_BASE_ROUNDS_ENV) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(v)) => {
                eprintln!("{B3_BASE_ROUNDS_ENV}={v:?}: expected 0, fri, 1 or w3");
                std::process::abort()
            }
        };
        b3_base_rounds_setting(value.as_deref()).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::abort()
        })
    })
}

/// The knob that chooses the GKR ladder of the BLAKE3 base's batched argues:
/// one tree layer a step ([`GkrLadder::Single`]) or two
/// ([`GkrLadder::Double`]: each step proves a tree layer against its
/// grandchildren). Both sides of one process read it, and the tag names a
/// two-layer ladder: see [`b3_base_gkr_setting`]. Read here, never by the
/// format module (`zf_format`), whose knobs a pinned verifier refuses.
pub const B3_BASE_GKR_ENV: &str = "LAMBDA_VM_B3_BASE_GKR";

/// The BLAKE3 base's GKR ladder when [`B3_BASE_GKR_ENV`] is unset: two tree
/// layers a step. Any other base argues one layer a step.
pub const B3_BASE_GKR_DEFAULT: GkrLadder = GkrLadder::Double;

/// [`B3_BASE_GKR_ENV`]'s value: unset is `None` (the base's default,
/// [`B3_BASE_GKR_DEFAULT`] under BLAKE3); `0` or `single` is
/// [`GkrLadder::Single`]; `1` or `double` is [`GkrLadder::Double`]. Only these
/// words are read: anything else is refused.
pub fn b3_base_gkr_setting(value: Option<&str>) -> Result<Option<GkrLadder>, String> {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        None => Ok(None),
        Some(v) => match v.as_str() {
            "0" | "single" => Ok(Some(GkrLadder::Single)),
            "1" | "double" => Ok(Some(GkrLadder::Double)),
            other => Err(format!(
                "{B3_BASE_GKR_ENV}={other:?}: expected 0, single, 1 or double"
            )),
        },
    }
}

/// The GKR ladder [`B3_BASE_GKR_ENV`] names in this process
/// ([`b3_base_gkr_setting`]), `None` when unset; read once; a refused value
/// aborts the process at its first read, as [`b3_base_rounds`] does.
pub fn b3_base_gkr() -> Option<GkrLadder> {
    static LADDER: std::sync::OnceLock<Option<GkrLadder>> = std::sync::OnceLock::new();
    *LADDER.get_or_init(|| {
        let value = match std::env::var(B3_BASE_GKR_ENV) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(v)) => {
                eprintln!("{B3_BASE_GKR_ENV}={v:?}: expected 0, single, 1 or double");
                std::process::abort()
            }
        };
        b3_base_gkr_setting(value.as_deref()).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::abort()
        })
    })
}

/// The statement tag's word for a batched argue's GKR ladder: nothing for
/// today's one-layer steps, `/L2` for two layers a step.
pub fn gkr_ladder_tag_word(ladder: GkrLadder) -> &'static str {
    match ladder {
        GkrLadder::Single => "",
        GkrLadder::Double => "/L2",
    }
}

/// The knob over the BLAKE3 base's KECCAK_RND byte lookups: argued per group
/// ([`KeccakRndForm::LeanLocal`], the default, [`B3_BASE_LOCAL_DEFAULT`]) or
/// in the global bus (`off`, [`KeccakRndForm::Lean`]); both sides of one
/// process read it, and the tag names local lookups (`/LL1`): see
/// [`b3_base_local_setting`]. The BLAKE3 base's alone: asking for local
/// lookups on another base is a configuration error.
///
/// [`KeccakRndForm::LeanLocal`]: crate::tables::keccak_rnd::KeccakRndForm::LeanLocal
/// [`KeccakRndForm::Lean`]: crate::tables::keccak_rnd::KeccakRndForm::Lean
pub const B3_BASE_LOCAL_ENV: &str = "LAMBDA_VM_B3_BASE_LOCAL";

/// Whether the BLAKE3 base argues KECCAK_RND's byte lookups per group when
/// [`B3_BASE_LOCAL_ENV`] is unset: yes. The levels verifier pins the same
/// value ([`B3_BASE_KECCAK_RND`]).
pub const B3_BASE_LOCAL_DEFAULT: bool = true;

/// KECCAK_RND's form on the BLAKE3 base at the default
/// ([`B3_BASE_LOCAL_DEFAULT`]): the form the levels verifier pins.
pub const B3_BASE_KECCAK_RND: crate::tables::keccak_rnd::KeccakRndForm = if B3_BASE_LOCAL_DEFAULT {
    crate::tables::keccak_rnd::KeccakRndForm::LeanLocal
} else {
    crate::tables::keccak_rnd::KeccakRndForm::Lean
};

/// The knob that, under local lookups ([`B3_BASE_LOCAL_ENV`]), lets phase
/// A's lane run each local group's local argue beside the commits, one bin
/// at a time, the bins left running at the group's turn in phase B
/// (`multilinear_block::LocalPhaseA::lane`). The proof is the same byte for
/// byte either way: see [`local_lane_setting`].
pub const LOCAL_LANE_ENV: &str = "LAMBDA_VM_LOCAL_LANE";

/// [`LOCAL_LANE_ENV`]'s value: unset, empty or `1` is on (the lane runs the
/// local argues); `0` is off (every local argue at its group's turn in phase
/// B). Anything else is refused.
pub fn local_lane_setting(value: Option<&str>) -> Result<bool, String> {
    match value.map(str::trim) {
        None | Some("" | "1") => Ok(true),
        Some("0") => Ok(false),
        Some(other) => Err(format!(
            "{LOCAL_LANE_ENV}={other:?}: expected 1 (the lane runs the local argues, the \
             default) or 0 (phase B runs them)"
        )),
    }
}

/// [`B3_BASE_LOCAL_ENV`]'s value: unset is `None` (the base's default,
/// [`B3_BASE_LOCAL_DEFAULT`] on the BLAKE3 base, off on any other); `0` or
/// `off` is off (KECCAK_RND's byte lookups in the global bus); `keccak`
/// argues them per group. Anything else is refused.
pub fn b3_base_local_setting(value: Option<&str>) -> Result<Option<bool>, String> {
    match value.map(|v| v.trim().to_ascii_lowercase()) {
        None => Ok(None),
        Some(v) => match v.as_str() {
            "0" | "off" => Ok(Some(false)),
            "keccak" => Ok(Some(true)),
            other => Err(format!(
                "{B3_BASE_LOCAL_ENV}={other:?}: expected 0, off or keccak"
            )),
        },
    }
}

/// The BLAKE3 base's local lookups in this process ([`b3_base_local_setting`]
/// of [`B3_BASE_LOCAL_ENV`]; `None` unset), read once; a refused value aborts
/// the process at its first read, as [`b3_base_rounds`] does.
pub fn b3_base_local() -> Option<bool> {
    static LOCAL: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *LOCAL.get_or_init(|| {
        let value = match std::env::var(B3_BASE_LOCAL_ENV) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(v)) => {
                eprintln!("{B3_BASE_LOCAL_ENV}={v:?}: expected 0, off or keccak");
                std::process::abort()
            }
        };
        b3_base_local_setting(value.as_deref()).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::abort()
        })
    })
}

/// The statement tag's word for local lookups: `/LL1` when the format argues
/// KECCAK_RND's byte lookups per group, nothing otherwise. Domain separation
/// only: the format is each side's own.
pub const LOCAL_LOOKUPS_TAG: &[u8] = b"/LL1";

/// The statement tag of a block proved under `format`: [`block_statement_tag`]
/// of its base, and under the 7-round BLAKE3 base its group size too
/// (`/G<n>`, the format's [`BlockFormat::group_polys`]), so no two BLAKE3
/// group sizes share a tag, then its round form and cap rule (`/W<shrink>/Cq`
/// under the WHIR round form with caps by Q, nothing under today's:
/// [`B3Rounds::tag_word`]); then, under any base, its argue's GKR ladder
/// (`/L2` for two layers a step, nothing under today's:
/// [`gkr_ladder_tag_word`]). Every block statement's prover and verifier
/// (and the recursion's plan) take their tag here.
pub fn block_format_tag(format: &BlockFormat) -> Vec<u8> {
    let mut tag = block_statement_tag(&format.zf.base);
    if format.zf.base == BaseFormat::B3_WHIR {
        #[cfg(test)]
        if P1_TAG_OMITTED.with(std::cell::Cell::get) {
            return tag;
        }
        tag.extend_from_slice(format!("/G{}", format.group_polys).as_bytes());
        tag.extend_from_slice(format.b3_rounds.tag_word().as_bytes());
    }
    #[cfg(test)]
    if LADDER_TAG_OMITTED.with(std::cell::Cell::get) {
        return tag;
    }
    tag.extend_from_slice(gkr_ladder_tag_word(format.argue.ladder()).as_bytes());
    if format.local_lookups() {
        tag.extend_from_slice(LOCAL_LOOKUPS_TAG);
    }
    tag
}

#[cfg(test)]
thread_local! {
    /// Test only: on this thread, a statement's tag leaves its GKR ladder's
    /// word out (the tag's mutation). Thread-local, so a test that sets it
    /// moves no other test's statement.
    pub(crate) static LADDER_TAG_OMITTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
thread_local! {
    /// Test only: on this thread, a statement takes the untagged
    /// `LAMBDAVM_MULTILINEAR_BLOCK_STATEMENT_V2` (the tag's mutation).
    /// Thread-local, so a test that sets it moves no other test's statement.
    pub(crate) static P1_TAG_OMITTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl BlockFormat {
    /// The process's WHIR format (as [`crate::multilinear_prove::chain_config`]
    /// reads it), [`BLOCK_GROUP_POLYS`], [`BLOCK_MAX_GROUPS`], each group's
    /// tables argued together ([`ArgueFormat::BATCHED`]; Mauro 10-02: "batch the
    /// constraints"), and one WHIR chain per run of a group's stacked
    /// polynomials ([`GroupFold::Indexed`]). `ArgueFormat::PerTable` and
    /// [`GroupFold::PerPoly`] stay formats to measure against. The leaves'
    /// scale is the environment's (`LAMBDA_VM_WHIR_LEAF_DOUBLE_FROM`; unset,
    /// [`LeafScale::DEFAULT`](crate::lfm::whir_block::LeafScale::DEFAULT),
    /// every leaf doubled and one top over at most three of them; a count `T`,
    /// doubled from `T` leaves; `off`, every leaf at the single heights). The
    /// leaves' query check is the environment's too
    /// (`LAMBDA_VM_WHIR_LEAF_QUERY`; unset,
    /// [`LeafQuery::Lean`](crate::lfm::whir_chain::LeafQuery::Lean); `v1`, the
    /// check before it).
    pub fn production() -> Self {
        let zf = *ZfFormat::global();
        Self {
            zf,
            // The BLAKE3 base's own group size ([`b3_base_group_polys`]).
            group_polys: if zf.base == BaseFormat::B3_WHIR {
                b3_base_group_polys()
            } else {
                BLOCK_GROUP_POLYS
            },
            // The BLAKE3 base's round form ([`b3_base_rounds`]); the knob on
            // any other base is a configuration error.
            b3_rounds: match (zf.base == BaseFormat::B3_WHIR, b3_base_rounds()) {
                (true, rounds) => rounds,
                (false, B3Rounds::Fri) => B3Rounds::Fri,
                (false, rounds) => {
                    eprintln!(
                        "{B3_BASE_ROUNDS_ENV} asks for {rounds:?}: the round form is the BLAKE3 \
                         base's alone"
                    );
                    std::process::abort()
                }
            },
            max_groups: BLOCK_MAX_GROUPS,
            prepared: true,
            leaf_scale: crate::lfm::whir_block::process_leaf_scale(),
            leaf_query: crate::lfm::whir_block::process_leaf_query(),
            // The BLAKE3 base's GKR ladder ([`b3_base_gkr`]; unset, two layers
            // a step, [`B3_BASE_GKR_DEFAULT`]); any other base argues one
            // layer a step, and the knob asking it for two is a configuration
            // error.
            argue: ArgueFormat::BATCHED.with_ladder(
                match (zf.base == BaseFormat::B3_WHIR, b3_base_gkr()) {
                    (true, ladder) => ladder.unwrap_or(B3_BASE_GKR_DEFAULT),
                    (false, None | Some(GkrLadder::Single)) => GkrLadder::Single,
                    (false, Some(ladder)) => {
                        eprintln!(
                            "{B3_BASE_GKR_ENV} asks for {ladder:?}: the two-layer GKR ladder is \
                             the BLAKE3 base's alone"
                        );
                        std::process::abort()
                    }
                },
            ),
            group: GroupFold::Indexed,
            // KECCAK_RND's byte lookups on the BLAKE3 base: per group unless
            // its knob ([`b3_base_local`]) says `off`
            // ([`B3_BASE_KECCAK_RND`] unset); on any other base in the
            // global bus, and the knob asking there for local lookups is a
            // configuration error.
            keccak_rnd: match (zf.base == BaseFormat::B3_WHIR, b3_base_local()) {
                (true, None) => B3_BASE_KECCAK_RND,
                (true, Some(true)) => crate::tables::keccak_rnd::KeccakRndForm::LeanLocal,
                (_, Some(false)) | (false, None) => crate::tables::keccak_rnd::KeccakRndForm::Lean,
                (false, Some(true)) => {
                    eprintln!(
                        "{B3_BASE_LOCAL_ENV} asks for local lookups: they are the BLAKE3 base's alone"
                    );
                    std::process::abort()
                }
            },
            memw_aligned: crate::tables::memw_aligned::MemwAlignedOrder::Limbs,
            registers: crate::tables::register::RegisterFormat::InRow,
            memory: crate::tables::fused_memory::MemoryFormat::Fused,
            ecdas: crate::tables::ecdas::EcdasForm::Halves,
            page_run_log2: BLOCK_PAGE_RUN_LOG2,
        }
    }

    /// Whether this format argues local lookups per group: KECCAK_RND's byte
    /// lookups under [`KeccakRndForm::LeanLocal`], each group that holds a
    /// KECCAK_RND table ending with its LOCAL_BYTES table
    /// ([`crate::tables::local_bytes`]).
    ///
    /// [`KeccakRndForm::LeanLocal`]: crate::tables::keccak_rnd::KeccakRndForm::LeanLocal
    pub fn local_lookups(&self) -> bool {
        self.keccak_rnd.byte_lookups_local()
    }

    /// [`production`](Self::production) with each group's stacked polynomials
    /// opened per `group`: the caller's constant, which the prover and the
    /// verifier of one block must share (it is bound into the statement
    /// through the fold word, and never read from a proof).
    pub fn production_with(group: GroupFold) -> Self {
        Self {
            group,
            ..Self::production()
        }
    }

    /// ★ The block VERIFIER's production format, every field a constant:
    /// [`ZfFormat::DEFAULT`] (blowup, folds, stack, grind placement and bits,
    /// the caps, the round-0 leaf order),
    /// [`LeafScale::DEFAULT`](crate::lfm::whir_block::LeafScale::DEFAULT),
    /// [`LeafQuery::Lean`](crate::lfm::whir_chain::LeafQuery::Lean), and
    /// [`production`](Self::production)'s group size, group cap, argue and
    /// group opening. Where [`production`](Self::production) is the process's
    /// format (its environment knobs, for the prover and its experiments),
    /// this reads nothing from the environment: a verifier's environment
    /// cannot choose the format, and with it the security, it checks a proof
    /// against. Its memory format is the split one, the recursion's tree's
    /// default (the production format's is the fused one).
    pub fn pinned() -> Self {
        Self {
            zf: ZfFormat::DEFAULT,
            group_polys: BLOCK_GROUP_POLYS,
            b3_rounds: B3Rounds::Fri,
            max_groups: BLOCK_MAX_GROUPS,
            prepared: true,
            leaf_scale: crate::lfm::whir_block::LeafScale::DEFAULT,
            leaf_query: crate::lfm::whir_chain::LeafQuery::Lean,
            argue: ArgueFormat::BATCHED,
            group: GroupFold::Indexed,
            keccak_rnd: crate::tables::keccak_rnd::KeccakRndForm::Lean,
            memw_aligned: crate::tables::memw_aligned::MemwAlignedOrder::Limbs,
            registers: crate::tables::register::RegisterFormat::InRow,
            memory: crate::tables::fused_memory::MemoryFormat::Split,
            ecdas: crate::tables::ecdas::EcdasForm::Halves,
            page_run_log2: BLOCK_PAGE_RUN_LOG2,
        }
    }

    /// [`pinned`](Self::pinned) with each group's stacked polynomials opened
    /// per `group`, the verifier's named choice
    /// ([`production_with`](Self::production_with)).
    pub fn pinned_with(group: GroupFold) -> Self {
        Self {
            group,
            ..Self::pinned()
        }
    }

    /// This format with KECCAK_RND's byte checks as `keccak_rnd` sends them:
    /// a named constant the prover and
    /// the verifier of one block both take, never read from a proof.
    pub fn with_keccak_rnd(self, keccak_rnd: crate::tables::keccak_rnd::KeccakRndForm) -> Self {
        Self { keccak_rnd, ..self }
    }

    /// This format with ECDAS's relations in `ecdas`' form: a named constant
    /// the prover and the verifier of one block both take, never read from a
    /// proof.
    pub fn with_ecdas(self, ecdas: crate::tables::ecdas::EcdasForm) -> Self {
        Self { ecdas, ..self }
    }

    /// This format with MEMW_A's timestamp order in `memw_aligned`'s form: a
    /// named constant the prover and the verifier of one block both take,
    /// never read from a proof.
    pub fn with_memw_aligned(
        self,
        memw_aligned: crate::tables::memw_aligned::MemwAlignedOrder,
    ) -> Self {
        Self {
            memw_aligned,
            ..self
        }
    }

    /// This format with the CPU's loads and stores checked as `memory` says: a
    /// named constant the prover and the verifier of one block both take,
    /// never read from a proof.
    pub fn with_memory(self, memory: crate::tables::fused_memory::MemoryFormat) -> Self {
        Self { memory, ..self }
    }

    /// This format with its page tables at runs of up to `2^page_run_log2`
    /// pages ([`Self::page_run_log2`]): a named constant the prover and the
    /// verifier of one block both take, never read from a proof.
    pub fn with_page_run(self, page_run_log2: u8) -> Self {
        Self {
            page_run_log2,
            ..self
        }
    }

    /// The page runs this format cuts its pages at: [`Self::page_run_log2`]
    /// under the word memory format, 0 (one table a page) under any other,
    /// whose pages are PAGE's.
    pub fn page_runs(&self) -> u8 {
        if self.memory == crate::tables::fused_memory::MemoryFormat::Words {
            self.page_run_log2
        } else {
            0
        }
    }

    /// This format with its batched argue's GKR trees laid out per `pack`:
    /// a named constant the prover and the
    /// verifier of one block both take, never read from a proof. Production
    /// is [`GkrPack::PRODUCTION`](multilinear::whir_chain::GkrPack::PRODUCTION);
    /// [`GkrPack::Off`](multilinear::whir_chain::GkrPack::Off) is one tree a
    /// table. A per-table argue has no trees to pack and is returned as it is.
    pub fn with_pack(self, pack: multilinear::whir_chain::GkrPack) -> Self {
        let argue = match self.argue {
            ArgueFormat::Batched {
                bin_log_cells,
                ladder,
                ..
            } => ArgueFormat::Batched {
                bin_log_cells,
                pack,
                ladder,
            },
            ArgueFormat::PerTable => ArgueFormat::PerTable,
        };
        Self { argue, ..self }
    }

    /// The chain's config over `shapes`: the WHIR format's, with this format's
    /// argue, group fold and round-0 leaf order (the block's alone:
    /// `ZfFormat::chain_format` leaves every other chain natural). Every side
    /// of a block builds its config here.
    pub fn chain_config(&self, shapes: &[(usize, usize)]) -> ChainConfig {
        let mut config = chain_config_under(&self.zf, shapes);
        // The BLAKE3 base's binary trees take its own cap ([`b3_base_cap`]),
        // and its chains its final polynomial ([`b3_base_final`]).
        if self.zf.base == BaseFormat::B3_WHIR {
            config.format.cap = CapPolicy::Fixed(b3_base_cap());
            config.format.final_poly = b3_base_final();
        }
        config.format.argue = self.argue;
        config.format.group = self.group;
        config.format.leaves = self.zf.whir_leaves;
        // One tree a run needs runs: one chain a polynomial has none, and its
        // round-0 trees are one a polynomial whatever the knob says.
        config.format.round0 = match self.group {
            GroupFold::Indexed => self.zf.whir_trees,
            GroupFold::PerPoly => multilinear::whir_chain::Round0Trees::Each,
        };
        // ★ The BLAKE3 base's round form ([`B3Rounds`]): every round's count,
        // round 0's among them, derived for the 128-bit target over ONE union
        // bound that reads no shape — the rounds of a chain over
        // [`B3_ROUND_FORM_STACK_BOUND`] variables, which no chain of a block
        // exceeds (the chains themselves refuse one that does,
        // `check_round_form_union`). So the config's format, and the fold word
        // every commit-time out-of-domain point absorbs, is the same whether
        // phase A builds it before the block's shapes are known or the
        // verifier builds it from the statement; and the chains'
        // `check_round_form` takes it on both sides. Round 0 queries what
        // today's form queries over a `2^27` stack (229 at the default), never
        // fewer. Then its cap rule.
        if let (true, B3Rounds::Whir { shrink, caps_by_q }) =
            (self.zf.base == BaseFormat::B3_WHIR, self.b3_rounds)
        {
            config = config
                .with_whir_rounds(
                    usize::from(shrink),
                    crate::multilinear_prove::CHAIN_SECURITY_BITS,
                    B3_ROUND_FORM_STACK_BOUND,
                )
                .expect("a shrink from 1 to 3 is a round form");
            if let multilinear::whir_chain::RoundForm::Whir(w) = config.format.rounds {
                config.num_queries = multilinear::query_count::num_queries(
                    config.log_blowup,
                    w.union_rounds(),
                    w.security_bits(),
                    config.grind.query,
                );
            }
            if caps_by_q {
                config = config
                    .with_caps_by_queries()
                    .expect("a round-form config takes caps by Q");
            }
        }
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
    /// COPY's tables are cut at 2^`copy_rows_log2` rows
    /// ([`BLOCK_COPY_ROWS_LOG2`] in production).
    pub copy_rows_log2: usize,
    /// The bottom levels each committed tree drops until its opening, on the
    /// card and on the host. Production: [`tree_drop_from_env`].
    pub drop_levels: multilinear::whir_commit::TreeDrop,
    /// `Some(k)`: build the traces in windows of 2^k cycles and commit each
    /// table as its chunk completes ([`WindowedTraceBuilder`]); `None`: build
    /// the whole run, then commit.
    pub window_log2: Option<usize>,
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
    /// [`BLOCK_LAYOUT_WORKERS`]. Once phase A uploads ahead, the inline layout
    /// closes the middle groups after the card is free; three workers close
    /// them in time (FAST 835: phase A −0.93 s, whole block −1.00 s, peak −0.68
    /// GiB at 1×). Unbounded they held more host memory as the block grew (BIG
    /// 390); `layout_ahead` bounds them (BIG 392: flat).
    pub layout_workers: usize,
    /// With `layout_workers > 0`: `Some(k)` lets at most `k + 1` streamed
    /// chunks be laid out (or in the making) and not yet packed, where the
    /// inline layout holds one; `None` leaves only the channels to bound them,
    /// which BIG 390 saw grow with the block. Production
    /// `Some(`[`BLOCK_LAYOUT_AHEAD`]`)`.
    pub layout_ahead: Option<usize>,
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
    /// With windows: the finish builds KECCAK_RND
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
    /// the tables are the same. Production: on (off, a test's A arm: built
    /// whole and wide, then split). At p90 the three are the rest's only wide tables (5.09 GiB) and
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
    /// groups are the same. Production: on (off, a test's A arm and a build
    /// without `parallel`'s path: every table built at once, then laid out).
    pub finish_stream: bool,
    /// With windows: `Some(bytes)` lays the rest of the run out in AIR order,
    /// in waves of at most `bytes` of rows (a larger table is a wave of its
    /// own), each wave in parallel; `None` lays every table out at once. A
    /// table's column copy exists before its rows go, so laying out all of
    /// them at once holds most of the rest twice (BIG 561: ≈ 20 GiB at 4.13×,
    /// the block's high-water). The tables, their order and the groups are the
    /// same. Production: [`BLOCK_REST_LAYOUT_BYTES`].
    ///
    /// With the streamed finish ([`Self::finish_stream`]) the same bytes are
    /// its byte gate's budget, and the sliding finish has no waves: it starts
    /// each table as the gate admits it.
    pub rest_layout_bytes: Option<usize>,
    /// With the streamed finish sliding: at most this many estimated bytes of
    /// the rest's tables build at once, apart from the gate's budget
    /// ([`Self::rest_layout_bytes`]), which bounds what is held (built and not
    /// yet placed, or building). `None`: the budget alone. Production:
    /// [`BLOCK_REST_BUILD_BYTES`]. The tables, their order and the groups are
    /// the same.
    pub rest_build_bytes: Option<usize>,
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
    /// are the same. Production: on (off, a test's A arm: built at 8 bytes a
    /// cell, then laid out or packed as before).
    pub gpack: bool,
    /// What phase B does with the streamed chunks ([`regen::RegenMode`]):
    /// `None` reads `LAMBDA_VM_BLOCK_REGEN` when the prove starts (unset:
    /// `off`; a value it does not know refuses the prove), `Some` is chosen.
    /// Production: [`regen::production_regen`], `auto` when that knob is
    /// unset. Without live regeneration every committed table is held. The
    /// proof's bytes are the same under every mode.
    pub regen: Option<regen::RegenMode>,
    /// Phase B opens each group on a thread of its own beside the next
    /// group's argue, and under `Pinned` the openings' thread reads back
    /// through pinned staging ([`multilinear_block::PhaseBOverlap`]); the
    /// proof's bytes are the same. Production: [`phase_b_overlap_from_env`],
    /// `Pinned` by default on a card build ([`PHASE_B_OVERLAP_DEFAULT`]).
    pub phase_b_overlap: multilinear_block::PhaseBOverlap,
}

/// Stacked-polynomial commits that fell back to the host this process
/// (`multilinear::gpu::host_fallbacks`): a round-0 tree the card declined
/// (a joint run's or a polynomial's) is counted here. On a build without a
/// card every commit is counted.
pub fn host_commit_fallbacks() -> u64 {
    multilinear::gpu::host_fallbacks()
}

/// The WHIR round form's successors encoded on the host, its shift-claim
/// batches the card added, and of those the row-tiled kernel's
/// (`multilinear::gpu`'s counters): what a base-only
/// run under [`B3Rounds::Whir`] prints, so a run whose successors left the
/// card says so.
pub fn round_form_counts() -> (u64, u64, u64) {
    (
        multilinear::gpu::round_form_host_encodes(),
        multilinear::gpu::shift_claim_batches(),
        multilinear::gpu::shift_claim_tiled_batches(),
    )
}

/// Round-0 openings of coset-ordered trees served from their stacks on the
/// card, and those that re-encoded their codewords instead (a plan the card's
/// kernels do not take, or a decline: `multilinear::gpu`'s counters). What a
/// base-only run prints, so a first fold whose cosets outgrew the card's plan
/// ([`CARD_COSET_FOLD_LEVELS`]) shows as re-encodes.
pub fn coset_counts() -> (u64, u64) {
    (
        multilinear::gpu::coset_openings(),
        multilinear::gpu::coset_fallbacks(),
    )
}

/// The host's memory target, against which live regeneration's `auto` rule
/// arms ([`drop_wanted`]): `LAMBDA_VM_BLOCK_MEMORY_TARGET_GIB`
/// ([`memory_target_knob`]; its old name refuses the prove,
/// [`regen::refuse_removed_spill_words`]), else the smaller of the cgroup's memory limit (v2 or v1, [`cgroup_memory`])
/// and `MemTotal`, less 10 GiB ([`memory_target_from`]) (#1013's
/// `spill_target_bytes` @ 035aef5d6).
pub fn memory_target_bytes() -> u64 {
    if let Some(bytes) = memory_target_knob(|name| std::env::var(name).ok()) {
        return bytes;
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
    memory_target_from(limit, mem_total)
}

/// `LAMBDA_VM_BLOCK_MEMORY_TARGET_GIB` in `env` as bytes ([`parse_target_gib`]).
/// Its old name is not read here: set, it refuses the prove.
fn memory_target_knob(env: impl Fn(&str) -> Option<String>) -> Option<u64> {
    parse_target_gib(env("LAMBDA_VM_BLOCK_MEMORY_TARGET_GIB"))
}

/// The memory target's knob in GiB as bytes; unset, or not a finite number
/// above 0, is `None` (the machine's target).
fn parse_target_gib(value: Option<String>) -> Option<u64> {
    value
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|g| g.is_finite() && *g > 0.0)
        .map(|gib| (gib * (1u64 << 30) as f64) as u64)
}

/// The smaller of a cgroup limit and `MemTotal`, less 10 GiB; no target
/// (`u64::MAX`) when neither is known. A v1 cgroup without a limit reads a
/// sentinel far above `MemTotal`, which the minimum discards (#1013's
/// `spill_target_from`).
fn memory_target_from(limit: Option<u64>, mem_total: Option<u64>) -> u64 {
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
    /// The peak and the charge as read, the page-locked pool's free part
    /// included: the memory the host really holds ([`Self::real_bytes`]).
    /// Without a pool, `hwm` and `charged` themselves.
    raw_hwm: u64,
    raw_charged: u64,
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
        // The page-locked pool's free part is resident from its lock on, and is
        // the room the block's later allocations take: every reading of the
        // host (the drop rule's, the walk's hand-off) counts it once, as it is
        // taken ([`multilinear::pinned::unused_pool_bytes`]).
        let unused = multilinear::pinned::unused_pool_bytes();
        let (raw_hwm, raw_charged) = (hwm.unwrap_or(0), charged.unwrap_or(0));
        Self {
            hwm: raw_hwm.saturating_sub(unused),
            charged: raw_charged.saturating_sub(unused),
            inactive_file: inactive_file.unwrap_or(0),
            raw_hwm,
            raw_charged,
        }
    }

    /// The host's bytes: the larger of the peak resident set (the resident set
    /// itself is not monotone under the allocator's posture, which drops and
    /// re-faults recycled extents) and the cgroup's working set, its charge
    /// less its inactive file pages (the kubelet's measure). Page cache the
    /// kernel would give back first is not counted; active file pages are,
    /// and the target's 10 GiB margin covers them. The page-locked pool's
    /// free part is not counted either (taken off as the reading is made).
    fn bytes(&self) -> u64 {
        self.hwm
            .max(self.charged.saturating_sub(self.inactive_file))
    }

    /// [`Self::bytes`] with the page-locked pool's free part counted: what the
    /// host really holds. The netted reading forecasts as if that part will
    /// take the block's later allocations, which only the pool's own
    /// allocations can; this one is what the cgroup's limit sees. The same as
    /// [`Self::bytes`] without a pool.
    fn real_bytes(&self) -> u64 {
        self.raw_hwm
            .max(self.raw_charged.saturating_sub(self.inactive_file))
    }

    /// What the host holds now: the cgroup's working set and the allocator's
    /// resident bytes (`allocator`), the larger. Unlike [`Self::bytes`] it
    /// falls when a purge hands freed pages back.
    fn now_bytes(&self, allocator: Option<u64>) -> u64 {
        self.charged
            .saturating_sub(self.inactive_file)
            .max(allocator.unwrap_or(0))
    }
}

/// What the host really holds now ([`HostReading::real_bytes`]), for a card
/// test that sets a target against it.
#[cfg(all(test, feature = "cuda"))]
pub(crate) fn host_real_bytes_now() -> u64 {
    HostReading::now().real_bytes()
}

/// The host's bytes a drop decision weighs: its peak ([`HostReading::bytes`]),
/// or, under partial drops once they have armed (`now`), what it holds now
/// ([`HostReading::now_bytes`]): a dropped table's pages, once purged, leave
/// the reading, and later tables may stay.
fn decision_bytes(reading: &HostReading, allocator: Option<u64>, now: bool) -> u64 {
    if now {
        reading.now_bytes(allocator)
    } else {
        reading.bytes()
    }
}

/// `LAMBDA_VM_BLOCK_KEEP_CELLS_G`: partial drops' keep budget at arming, in G
/// main cells committed ([`multilinear_block::BlockRegen::keep_cells`]). Once
/// live regeneration arms, the tables phase B can rebuild that were committed
/// past it are dropped back, in rank order, so what phase B holds no longer
/// follows how far phase A got when the policy first wanted a table out.
/// Unset, empty or `off` (the default): no budget. It is a constant read
/// against today's forecast (its reserve per G cells and its target); the
/// `BLOCK REGEN keep` line says each run's arming point. Anything else refuses
/// the prove. The proof is the same.
fn block_keep_cells() -> Result<Option<u64>, Error> {
    parse_keep_cells(std::env::var(KEEP_CELLS_ENV).ok().as_deref())
}

const KEEP_CELLS_ENV: &str = "LAMBDA_VM_BLOCK_KEEP_CELLS_G";

/// `LAMBDA_VM_BLOCK_KEEP_BACK_G`: partial drops' keep budget relative to the
/// arming point, in G main cells
/// ([`multilinear_block::BlockRegen::keep_below`]): once live regeneration
/// arms, the tables phase B can rebuild that were committed in the last that
/// many cells before it are dropped back, in rank order. The arming point
/// moves between runs (the host's reading when the forecast crosses), and the
/// budget moves with it. Unset or empty: [`BLOCK_KEEP_BELOW`] (the default); a
/// G-cell count above 0: that margin; `off`: none, arming drops back nothing
/// (the rollback). Anything else refuses the prove. The proof is the same.
fn block_keep_below() -> Result<Option<u64>, Error> {
    parse_keep_below(std::env::var(KEEP_BELOW_ENV).ok().as_deref())
}

const KEEP_BELOW_ENV: &str = "LAMBDA_VM_BLOCK_KEEP_BACK_G";

/// Partial drops' keep budget relative to the arming point by default, in main
/// cells below those committed at arming ([`block_keep_below`]): about what
/// the streamed layout eight chunks ahead commits before arming beyond what it
/// does six ahead (1.08 and 1.34 G cells in two runs of the large block), so a
/// phase A that commits sooner keeps no more through phase B than it would
/// have, wherever the run arms. A block that never arms (the median) is
/// untouched.
pub const BLOCK_KEEP_BELOW: u64 = 1_300_000_000;

fn parse_keep_below(value: Option<&str>) -> Result<Option<u64>, Error> {
    match value.map(str::trim) {
        None | Some("") => Ok(Some(BLOCK_KEEP_BELOW)),
        Some("off") => Ok(None),
        Some(v) => v
            .parse::<f64>()
            .ok()
            .filter(|g| g.is_finite() && *g > 0.0)
            .map(|g| Some((g * 1e9) as u64))
            .ok_or_else(|| Error::Prover(format!("{KEEP_BELOW_ENV}={v}: off, or G cells above 0"))),
    }
}

fn parse_keep_cells(value: Option<&str>) -> Result<Option<u64>, Error> {
    match value.map(str::trim) {
        None | Some("" | "off") => Ok(None),
        Some(v) => v
            .parse::<f64>()
            .ok()
            .filter(|g| g.is_finite() && *g > 0.0)
            .map(|g| Some((g * 1e9) as u64))
            .ok_or_else(|| Error::Prover(format!("{KEEP_CELLS_ENV}={v}: off, or G cells above 0"))),
    }
}

/// The dropped bytes after which partial drops purge ([`DropPurges`]).
const DROP_PURGE_BYTES: u64 = 4 << 30;
/// The least time between two of their purges.
const DROP_PURGE_SPACING: std::time::Duration = std::time::Duration::from_secs(5);

/// Partial drops' purges: once [`DROP_PURGE_BYTES`] have been dropped since the
/// last purge, and that one was [`DROP_PURGE_SPACING`] ago, a purge on a thread
/// of its own hands the allocator's freed pages back, so the host's reading
/// falls by the dropped tables. The committer that decides never waits on it.
struct DropPurges {
    purge: fn() -> Option<crate::alloc_purge::Purge>,
    pending: std::sync::atomic::AtomicU64,
    running: std::sync::Arc<std::sync::atomic::AtomicBool>,
    last: std::sync::Mutex<Option<Instant>>,
    count: std::sync::atomic::AtomicUsize,
    micros: std::sync::atomic::AtomicU64,
    returned: std::sync::atomic::AtomicU64,
}

impl DropPurges {
    fn new(purge: fn() -> Option<crate::alloc_purge::Purge>) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            purge,
            pending: Default::default(),
            running: Default::default(),
            last: std::sync::Mutex::new(None),
            count: Default::default(),
            micros: Default::default(),
            returned: Default::default(),
        })
    }

    /// A table of `bytes` the policy dropped. Returns the purge's thread when
    /// this drop started one.
    fn dropped(self: &std::sync::Arc<Self>, bytes: u64) -> Option<std::thread::JoinHandle<()>> {
        use std::sync::atomic::Ordering::{AcqRel, Relaxed, Release};
        let pending = self.pending.fetch_add(bytes, Relaxed).saturating_add(bytes);
        if pending < DROP_PURGE_BYTES {
            return None;
        }
        let spaced = self
            .last
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_none_or(|t| t.elapsed() >= DROP_PURGE_SPACING);
        if !spaced || self.running.swap(true, AcqRel) {
            return None;
        }
        self.pending.store(0, Relaxed);
        let me = std::sync::Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name("block-drop-purge".to_string())
            .spawn(move || {
                if let Some(purge) = (me.purge)() {
                    me.count.fetch_add(1, Relaxed);
                    me.micros.fetch_add((purge.secs * 1e6) as u64, Relaxed);
                    let returned = purge.resident_before.saturating_sub(purge.resident_after);
                    me.returned.fetch_add(returned as u64, Relaxed);
                }
                *me.last.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
                me.running.store(false, Release);
            });
        match spawned {
            Ok(handle) => Some(handle),
            Err(_) => {
                self.running.store(false, Release);
                None
            }
        }
    }

    /// The `BLOCK DROP` line.
    fn line(&self) -> String {
        use std::sync::atomic::Ordering::Relaxed;
        format!(
            "BLOCK DROP: partial · {} purges of the dropped tables' pages, {:.2} s, {:.2} GiB returned",
            self.count.load(Relaxed),
            self.micros.load(Relaxed) as f64 / 1e6,
            self.returned.load(Relaxed) as f64 / (1u64 << 30) as f64
        )
    }
}

/// The `auto` rule's reserve beside the host's bytes: what the block still
/// needs once the tables committed so far (`cells` main cells) are on the host. #1014's
/// own, inferred from the median block (BIG 565, 33.5 G cells): the finish's
/// p5 transient (25.5 GiB at its peak) and the tree's leaf programs, which
/// live through phase B (16.3 GiB at its end), ≈ 1.25 GiB per total G cells,
/// or ≈ 1.65 per G cells committed so far (the streamed share ≈ 0.76); plus
/// 6 GiB for phase B's bump and the read-back window. `per_g` is the GiB per
/// G cells committed ([`MEMORY_RESERVE_PER_G`] unless the knob retunes it,
/// [`memory_reserve_per_g`]).
fn memory_reserve_bytes(cells: u64, per_g: f64) -> u64 {
    (cells as f64 / 1e9 * per_g * (1u64 << 30) as f64) as u64 + (6 << 30)
}

/// [`memory_reserve_bytes`]'s GiB per G cells committed.
const MEMORY_RESERVE_PER_G: f64 = 1.65;

/// `LAMBDA_VM_BLOCK_MEMORY_RESERVE_PER_G`: the
/// `auto` rule's reserve per G cells committed, in GiB
/// ([`memory_reserve_bytes`]), a measurement knob for its retune (D-FINISH S4:
/// it still budgets the finish's transient S2b removed); unset, or not a finite
/// number ≥ 0, is [`MEMORY_RESERVE_PER_G`].
fn memory_reserve_per_g() -> f64 {
    parse_reserve_per_g(
        std::env::var("LAMBDA_VM_BLOCK_MEMORY_RESERVE_PER_G")
            .ok()
            .as_deref(),
    )
}

fn parse_reserve_per_g(value: Option<&str>) -> f64 {
    value
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|g| g.is_finite() && *g >= 0.0)
        .unwrap_or(MEMORY_RESERVE_PER_G)
}

/// The share of the memory target ([`memory_target_bytes`]) past which the block's
/// memory is short for the allocator's purge ([`crate::alloc_purge`]): 85 %, a
/// share so it follows the host's RAM.
const PRESSURE_SHARE: (u64, u64) = (17, 20);

/// Notes memory pressure for [`crate::alloc_purge`] once the host's bytes
/// ([`HostReading::bytes`]) pass [`PRESSURE_SHARE`] of `target`, and says
/// whether they did.
///
/// ★ THE ONE WAY #1014'S PURGE DIFFERS FROM #1013'S. The module, its knob
/// (`LAMBDA_VM_ALLOC_PURGE`, `auto` by default) and its points are #1013's;
/// #1013 notes the pressure when its spill's queue budgets arm, and #1014 has
/// no budgets, so it reads the host where live regeneration decides (each
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

/// The line past which the block's memory is short: [`PRESSURE_SHARE`] of
/// `target`.
fn pressure_line(target: u64) -> u64 {
    let (num, den) = PRESSURE_SHARE;
    target / den * num
}

/// Whether `host` bytes are past [`PRESSURE_SHARE`] of `target`.
fn memory_short(host: u64, target: u64) -> bool {
    host > pressure_line(target)
}

/// What the allocator's purges at phase A's end and the base's end weigh in a
/// block that proves with live regeneration ([`crate::alloc_purge::has_room`]):
/// the pressure line of `target`, and the regeneration window's ceiling as
/// what the block may still add. `None` for a block without it: it purges at
/// both whenever its memory is short, as ever.
///
/// The pressure such a block notes comes from [`hand_off_need`]'s forecast of
/// the finish's growth, which is the growth of tables that stay; a block with
/// live regeneration drops them, and on a host with room its resident set
/// never comes near the line the forecast crossed.
fn no_disk_room(no_disk: bool, target: u64) -> Option<crate::alloc_purge::NoDisk> {
    no_disk.then(|| crate::alloc_purge::NoDisk {
        line: pressure_line(target),
        headroom: regen::AHEAD_CAP,
    })
}

/// The finish's growth of the resident set, from the walk's end to phase A's
/// peak, per walked cycle: BIG 660's p90 (D, at defaults) went 85.19 → 115.27
/// GiB over 612.3 M cycles (I-WFULL §4.1). The median (653) grew 12.41 GiB over
/// 298.4 M and full gas (660 D) 18.06 over 602.9 M, both under it.
const FINISH_GROWTH_PER_CYCLE: f64 = 30.08 * (1u64 << 30) as f64 / 612.3e6;

/// What a block hands back at the walk's end
/// ([`multilinear_block::BlockSpill::hand_off`]), decided once there: the
/// forecast of phase A's peak, `hwm` (the resident high-water so far) plus
/// [`FINISH_GROWTH_PER_CYCLE`] for each of the `cycles` walked, past
/// [`PRESSURE_SHARE`] of `target`; 0 when the forecast fits under it. The
/// finish is still ahead, so the parked tables are dropped before its hump
/// builds, and the hump reuses their pages.
fn hand_off_need(hwm: u64, cycles: u64, target: u64) -> u64 {
    let forecast = hwm.saturating_add((FINISH_GROWTH_PER_CYCLE * cycles as f64) as u64);
    forecast.saturating_sub(pressure_line(target))
}

/// The `auto` rule for one committed table of `bytes` packed bytes, whether
/// it is to leave the host (which arms live regeneration): the host's bytes
/// (`host`, [`HostReading::bytes`]), the reserve for the `cells` main cells
/// committed so far (this table's included) at `per_g`
/// ([`memory_reserve_bytes`]) and the table past `target` (#1013's
/// `spill_decision`, with #1014's reserve).
fn drop_wanted(
    target: u64,
    per_g: f64,
    cells: u64,
    bytes: u64,
    host: impl FnOnce() -> u64,
) -> bool {
    host()
        .saturating_add(memory_reserve_bytes(cells, per_g))
        .saturating_add(bytes)
        > target
}

/// The `auto` rule's verdict on one committed table of `bytes`, and the
/// real-memory guard's, `(rule, guard)`: the rule forecasts from `host`, the
/// netted reading ([`decision_bytes`]), with `extra` once live regeneration
/// has armed; the guard wants the table out when what the host really holds
/// ([`real_decision_bytes`]), with the table and `extra`, is already past
/// `target`. The rule takes the page-locked pool's free part as room for the
/// block's later allocations, which only the pool's own allocations can use;
/// the guard keeps the cgroup's limit in sight. Without a pool the two
/// readings are one and the rule adds its reserve to it, so the guard never
/// wants a table the rule keeps.
#[allow(clippy::too_many_arguments)]
fn auto_wants(
    reading: &HostReading,
    host: u64,
    now: bool,
    extra: u64,
    target: u64,
    per_g: f64,
    cells: u64,
    bytes: u64,
) -> (bool, bool) {
    let rule = drop_wanted(target, per_g, cells, bytes, || host.saturating_add(extra));
    let guard = real_decision_bytes(reading, now)
        .saturating_add(bytes)
        .saturating_add(extra)
        > target;
    (rule, guard)
}

/// What the host really holds, the page-locked pool's free part counted, in
/// the decision's own terms ([`decision_bytes`]): the peak
/// ([`HostReading::real_bytes`]), or under armed partial drops the cgroup's
/// working set now. Never above [`decision_bytes`] of the same reading
/// without a pool.
fn real_decision_bytes(reading: &HostReading, now: bool) -> u64 {
    if now {
        reading.raw_charged.saturating_sub(reading.inactive_file)
    } else {
        reading.real_bytes()
    }
}

/// The walk's hand-off by the real-memory guard: what the host really holds
/// at the walk's end (the peak, the pool's free part counted) past the
/// pressure line, whatever the forecast; 0 under it. Without a pool it never
/// asks more than [`hand_off_need`] of the same peak.
fn real_hand_off_need(raw_hwm: u64, target: u64) -> u64 {
    raw_hwm.saturating_sub(pressure_line(target))
}

/// The real-memory guard's account over a prove: the most the host really
/// held at a decision, and the decisions it alone made (a table out, or the
/// walk's hand-off, that the netted reading would have kept).
#[derive(Default)]
struct RealGuard {
    most: std::sync::atomic::AtomicU64,
    alone: std::sync::atomic::AtomicU64,
}

impl RealGuard {
    fn note(&self, real: u64, alone: bool) {
        use std::sync::atomic::Ordering::Relaxed;
        self.most.fetch_max(real, Relaxed);
        if alone {
            self.alone.fetch_add(1, Relaxed);
        }
    }
}

impl BlockOptions {
    /// Whether the finish streams the rest under its byte gate
    /// ([`Self::finish_stream`]): it needs the layout thread, the finish's
    /// cuts and KECCAK_RND built as its tables.
    pub(crate) fn rest_streams(&self) -> bool {
        self.finish_stream
            && self.layout_workers > 0
            && self.finish_cuts
            && self.finish_keccak_rnd_chunks
            && self.keccak_rnd_rows_log2 >= 5
    }

    /// Every chunked table at 2^21 rows, KECCAK_RND at 2^16, ECDAS at 2^17,
    /// KECCAK at 2^18, ECSM at 2^17, windows of 2^20 cycles.
    pub fn production() -> Self {
        Self {
            max_rows: MaxRowsConfig::uniform(1 << 21),
            keccak_rnd_rows_log2: BLOCK_KECCAK_RND_ROWS_LOG2,
            ecdas_rows_log2: BLOCK_ECDAS_ROWS_LOG2,
            keccak_rows_log2: BLOCK_KECCAK_ROWS_LOG2,
            ecsm_rows_log2: BLOCK_ECSM_ROWS_LOG2,
            copy_rows_log2: BLOCK_COPY_ROWS_LOG2,
            drop_levels: tree_drop_from_env(),
            window_log2: Some(BLOCK_WINDOW_LOG2),
            drop_streamed_ops: true,
            layout_workers: BLOCK_LAYOUT_WORKERS,
            layout_ahead: Some(BLOCK_LAYOUT_AHEAD),
            narrow: multilinear_block::Narrowing::CARD,
            upload_ahead: true,
            memlog: memlog::from_env(),
            finish_keccak_rnd_chunks: true,
            finish_cuts: true,
            finish_stream: true,
            rest_layout_bytes: Some(BLOCK_REST_LAYOUT_BYTES),
            rest_build_bytes: Some(BLOCK_REST_BUILD_BYTES),
            pack_finished: true,
            gpack: true,
            regen: regen::production_regen(),
            phase_b_overlap: phase_b_overlap_from_env(),
        }
    }
}

/// Phase B's overlap when `LAMBDA_VM_PHASE_B_OVERLAP` is unset: on a card
/// build the openings beside the next group's argue, reading back through
/// pinned staging, with the argue's streams ahead of theirs (the card's
/// stream classes default to `priority`), which is the configuration that was
/// timed; a host-only build keeps the openings in line, untimed beside an
/// argue on the same cores.
pub(crate) const PHASE_B_OVERLAP_DEFAULT: multilinear_block::PhaseBOverlap =
    if cfg!(feature = "cuda") {
        multilinear_block::PhaseBOverlap::Pinned
    } else {
        multilinear_block::PhaseBOverlap::Off
    };

/// `LAMBDA_VM_PHASE_B_OVERLAP`: [`BlockOptions::phase_b_overlap`]'s
/// production value: unset [`PHASE_B_OVERLAP_DEFAULT`], `0` off, `1` on,
/// `pinned` on with pinned read-backs.
pub(crate) fn phase_b_overlap_from_env() -> multilinear_block::PhaseBOverlap {
    phase_b_overlap_from(std::env::var("LAMBDA_VM_PHASE_B_OVERLAP").ok().as_deref())
}

/// [`phase_b_overlap_from_env`]'s reading of its variable: unset is
/// [`PHASE_B_OVERLAP_DEFAULT`], `0` off, `1` on, `pinned` on with the
/// openings' read-backs through pinned staging; another word is read as off,
/// and said once.
pub(crate) fn phase_b_overlap_from(value: Option<&str>) -> multilinear_block::PhaseBOverlap {
    use multilinear_block::PhaseBOverlap;
    match value.map(str::trim) {
        None => PHASE_B_OVERLAP_DEFAULT,
        Some("0") => PhaseBOverlap::Off,
        Some("1") => PhaseBOverlap::On,
        Some("pinned") => PhaseBOverlap::Pinned,
        Some(other) => {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                eprintln!("LAMBDA_VM_PHASE_B_OVERLAP={other}: 0, 1 or pinned — read as off");
            });
            PhaseBOverlap::Off
        }
    }
}

/// Phase B's overlap as [`BlockOptions::phase_b_overlap`] asks for it.
fn phase_b_overlap(options: &BlockOptions) -> multilinear_block::PhaseBOverlap {
    options.phase_b_overlap
}

/// `LAMBDA_VM_BLOCK_SCHEDSTAT=1`: phase B reads its driver thread's CPU time
/// and run-queue delay at each group turn and the readout prints them by
/// tenth of phase B ([`BlockStamps::phase_b_sched`]). A report, off by default.
pub(crate) fn phase_b_sched_from_env() -> bool {
    phase_b_sched_from(std::env::var("LAMBDA_VM_BLOCK_SCHEDSTAT").ok().as_deref())
}

/// [`phase_b_sched_from_env`]'s reading: `1` on, anything else off.
pub(crate) fn phase_b_sched_from(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

/// The card set up (its context, its modules and its staging pools) on a
/// thread of its own as the streamed prove starts, so the first group's upload
/// does not wait for it; nothing without the card's kernels. The proof is the
/// same.
fn set_up_card() {
    if cfg!(feature = "cuda") {
        std::thread::spawn(|| {
            multilinear::gpu::set_up_device();
            head_log::card_ready();
        });
    }
}

/// BITWISE's lookups from LT's, BRANCH's and BYTEWISE's tables, counted as
/// each is built from the table's own rows and their multiplicities, with
/// BITWISE handed on after every other table, where the finish slides and
/// keeps only the streamed chunks' tails
/// ([`WindowedTraceBuilder::bitwise_from_tables`]), unless a deviation asks
/// for phase 4's count from the ops with BITWISE leading the rest
/// (`Some(false)`). The tables are the same either way; their order, and so
/// the groups and the proof's bytes, are not.
fn bitwise_from_tables_or_default(deviation: Option<bool>) -> bool {
    deviation.unwrap_or(true)
}

/// KECCAK_RND's tables handed on before every other table when the streamed
/// finish slides ([`WindowedTraceBuilder::keccak_rnd_first`]), so the card
/// commits them while the tables before them in AIR order are built, unless a
/// deviation asks for AIR order (`Some(false)`). The tables are the same; the
/// groups they are packed into, and so the proof's bytes, are not; the
/// verifier takes any partition the statement declares.
fn keccak_rnd_first_or_default(deviation: Option<bool>) -> bool {
    deviation.unwrap_or(true)
}

/// The knob of [`pages_first`].
const PAGES_FIRST_ENV: &str = "LAMBDA_VM_PAGES_FIRST";

/// Whether the streamed finish hands the page tables on right after
/// KECCAK_RND's ([`WindowedTraceBuilder::pages_first`]) when it slides: a
/// deviation's choice, else `LAMBDA_VM_PAGES_FIRST` (on by default). The
/// group the page tables fill (hundreds of them) is then committed and argued
/// early instead of near the end, so a level proved beside phase B receives
/// it while phase B still runs. The tables are the same; the groups, and so
/// the proof's bytes, are not; the verifier takes any partition the statement
/// declares.
fn pages_first(deviation: Option<bool>) -> Result<bool, Error> {
    match deviation {
        Some(on) => Ok(on),
        None => parse_pages_first(std::env::var(PAGES_FIRST_ENV).ok().as_deref()),
    }
}

/// `LAMBDA_VM_PAGES_FIRST`: unset, empty or `1` (the default) hands the page
/// tables on next; `0` keeps AIR order after KECCAK_RND (the rollback).
/// Anything else refuses the prove.
fn parse_pages_first(value: Option<&str>) -> Result<bool, Error> {
    match value.map(str::trim) {
        None | Some("" | "1") => Ok(true),
        Some("0") => Ok(false),
        Some(v) => Err(Error::Prover(format!(
            "{PAGES_FIRST_ENV}={v}: 1 (the default: the page tables next) or 0 (AIR order after \
             KECCAK_RND)"
        ))),
    }
}

/// The page tables' positions in the run's AIR order, each named for a page
/// (`PAGE…`): the positions the finish hands on after KECCAK_RND's under
/// [`pages_first`].
pub(crate) fn page_positions(airs: &VmAirs) -> Result<Vec<usize>, Error> {
    let first = first_page_index(airs)?;
    let refs = airs.air_refs();
    let range = first..first + airs.pages.len();
    for p in range.clone() {
        let name = refs.get(p).map(|air| air.name());
        if !name.is_some_and(|n| n.starts_with("PAGE")) {
            return Err(Error::Prover(format!(
                "the page tables go next, but the rest's table {p} is {name:?}"
            )));
        }
    }
    Ok(range.collect())
}

/// A measurement's knob ([`parse_keccak_rnd_arrivals`]).
const KECCAK_RND_ARRIVALS_ENV: &str = "LAMBDA_VM_KECCAK_RND_ARRIVALS";

/// `LAMBDA_VM_KECCAK_RND_ARRIVALS`: `1` logs when each KECCAK_RND table's
/// operations are all in as the windows are taken (`BLOCK KECCAK_RND
/// ARRIVALS`); unset, empty or `0` (the default) does not. It moves no byte of
/// the proof. Anything else refuses the prove.
fn parse_keccak_rnd_arrivals(value: Option<&str>) -> Result<bool, Error> {
    match value.map(str::trim) {
        None | Some("" | "0") => Ok(false),
        Some("1") => Ok(true),
        Some(v) => Err(Error::Prover(format!(
            "{KECCAK_RND_ARRIVALS_ENV}={v}: 0 (the default) or 1 (log KECCAK_RND's arrivals)"
        ))),
    }
}

/// The knob of [`stream_keccak_rnd_records`].
const STREAM_KECCAK_RND_ENV: &str = "LAMBDA_VM_STREAM_KECCAK_RND";

/// Whether KECCAK_RND's tables are handed out during the windows, each once
/// the keccak operations its rows come from are all in, held as those
/// operations for the card to fill (where the card fills KECCAK_RND,
/// [`CardFill::keccak_rnd`]), so their commits use phase A's idle card instead
/// of running back to back after the finish's header: a deviation's choice,
/// else `LAMBDA_VM_STREAM_KECCAK_RND` (on by default). `Some` when asked for
/// (`Some(true)` refuses a build that cannot stream KECCAK_RND), `None` for
/// the default, which streams where the build cuts KECCAK_RND into its tables
/// and the card fills them from their records. The tables and their
/// words are the same; their place in the hand-out, and so the groups (the
/// prover's, bound in the statement), are not. With the KECCAK_W row-range
/// fill (`LAMBDA_VM_FINISH_KECCAK_W_SPLIT`) the median and the large block's
/// phase A end 0.23 / 0.27 s sooner; alone it adds 0.27 s to the large's
/// prove, so the two are on together.
fn stream_keccak_rnd_records(deviation: Option<bool>) -> Result<Option<bool>, Error> {
    match deviation {
        Some(on) => Ok(Some(on)),
        None => parse_stream_keccak_rnd(std::env::var(STREAM_KECCAK_RND_ENV).ok().as_deref()),
    }
}

/// `LAMBDA_VM_STREAM_KECCAK_RND`: unset or empty is the default (handed out
/// during the windows where the card fills KECCAK_RND, `None`); `1` asks for it
/// (`Some(true)`), `0` builds KECCAK_RND at the finish (the rollback,
/// `Some(false)`). Anything else refuses the prove.
fn parse_stream_keccak_rnd(value: Option<&str>) -> Result<Option<bool>, Error> {
    match value.map(str::trim) {
        None | Some("") => Ok(None),
        Some("1") => Ok(Some(true)),
        Some("0") => Ok(Some(false)),
        Some(v) => Err(Error::Prover(format!(
            "{STREAM_KECCAK_RND_ENV}={v}: 1 (the default: KECCAK_RND handed out during the \
             windows) or 0 (built at the finish)"
        ))),
    }
}

/// The knob of [`own_keccak_rnd_groups`].
const KECCAK_RND_OWN_GROUPS_ENV: &str = "LAMBDA_VM_KECCAK_RND_OWN_GROUPS";

/// Whether KECCAK_RND's streamed tables are packed into groups of their own
/// ([`OwnGroups`]) instead of among the other streamed chunks in arrival order:
/// a deviation's choice, else `LAMBDA_VM_KECCAK_RND_OWN_GROUPS` (on by
/// default). Among the 2^21-row chunks the 2^16-row tables stack worse than in
/// groups of their own (four a group, as the finish's are packed), and each
/// stacked polynomial more costs phase B its opening: on by default, the
/// large block's phase B 0.21 s and its prove 0.26 s shorter (the median's
/// prove 0.09 s), three stacked polynomials fewer. Only with KECCAK_RND
/// streamed ([`stream_keccak_rnd_records`]). The tables and their words are the
/// same; the groups (the prover's, bound in the statement) are not.
fn own_keccak_rnd_groups(deviation: Option<bool>) -> Result<bool, Error> {
    match deviation {
        Some(on) => Ok(on),
        None => {
            parse_own_keccak_rnd_groups(std::env::var(KECCAK_RND_OWN_GROUPS_ENV).ok().as_deref())
        }
    }
}

/// `LAMBDA_VM_KECCAK_RND_OWN_GROUPS`: unset, empty or `1` (the default) packs
/// KECCAK_RND's streamed tables in groups of their own; `0` among the other
/// streamed chunks (the rollback). Anything else refuses the prove.
fn parse_own_keccak_rnd_groups(value: Option<&str>) -> Result<bool, Error> {
    match value.map(str::trim) {
        None | Some("" | "1") => Ok(true),
        Some("0") => Ok(false),
        Some(v) => Err(Error::Prover(format!(
            "{KECCAK_RND_OWN_GROUPS_ENV}={v}: 1 (the default: KECCAK_RND's streamed tables in \
             groups of their own) or 0 (among the streamed chunks)"
        ))),
    }
}

/// The knob of [`rest_airs_beside`].
const REST_AIRS_BESIDE_ENV: &str = "LAMBDA_VM_REST_AIRS_BESIDE";

/// Whether the streamed rest builds the run's AIR set on a thread of its own
/// while the KECCAK_RND tables handed on first are laid out with their own
/// AIRs ([`lay_out_rest_streamed`]), so they reach the packer as the finish
/// hands them on rather than once the whole set exists: a deviation's choice,
/// else `LAMBDA_VM_REST_AIRS_BESIDE` (on by default). The AIRs, the tables,
/// their order and the proof's bytes are the same either way. On the large
/// block it takes the AIR set's ≈ 0.4 s off the first KECCAK_RND table's way to
/// the card: phase A −0.26 s, the median's unchanged.
fn rest_airs_beside(deviation: Option<bool>) -> Result<bool, Error> {
    match deviation {
        Some(on) => Ok(on),
        None => parse_rest_airs_beside(std::env::var(REST_AIRS_BESIDE_ENV).ok().as_deref()),
    }
}

/// `LAMBDA_VM_REST_AIRS_BESIDE`: unset, empty or `1` (the default) builds the
/// AIR set beside the first KECCAK_RND tables; `0` builds it before the rest's
/// first table (the rollback). Anything else refuses the prove.
fn parse_rest_airs_beside(value: Option<&str>) -> Result<bool, Error> {
    match value.map(str::trim) {
        None | Some("" | "1") => Ok(true),
        Some("0") => Ok(false),
        Some(v) => Err(Error::Prover(format!(
            "{REST_AIRS_BESIDE_ENV}={v}: 1 (the default: built beside the first KECCAK_RND \
             tables) or 0 (the AIR set before the rest's first table)"
        ))),
    }
}

/// How the rest's layout bytes ([`BlockOptions::rest_layout_bytes`]) bound
/// the rest, for the harnesses' `BLOCK REST LAYOUT CONFIG` line: waves of
/// them where the rest is laid out in waves, and the streamed finish's gate
/// where it slides (`slide`: the streamed finish builds each of the rest's
/// tables as soon as the byte gate admits it,
/// [`WindowedTraceBuilder::slide_rest`]), which has no waves.
pub(crate) fn rest_layout_label(options: &BlockOptions, slide: bool) -> String {
    let mib = |b: usize| format!("{} MiB", b >> 20);
    match (options.rest_streams() && slide, options.rest_layout_bytes) {
        (true, Some(b)) => format!("a gate of {}, sliding (no waves)", mib(b)),
        (true, None) => "no gate, sliding (no waves)".to_string(),
        (false, Some(b)) => format!("waves of {}", mib(b)),
        (false, None) => "waves of all at once".to_string(),
    }
}

/// The families whose tables the card fills from the records their rows come
/// from ([`block_card_fill`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CardFill {
    /// Each KECCAK_RND table the finish cuts, from its keccak operations
    /// ([`WindowedTraceBuilder::card_fill_keccak_rnd`]).
    pub(crate) keccak_rnd: bool,
    /// Each full streamed CPU chunk, from the executor's logs of its cycles,
    /// and each full MEMW_R chunk, from its register rows, while the host has
    /// room ([`WindowedTraceBuilder::card_fill_streamed`]).
    pub(crate) cpu: bool,
    pub(crate) memw_r: bool,
    /// Each full streamed chunk of these families, from its rows (MEMW_A's
    /// aligned rows, LOAD's, STORE's, MEMW's and SHIFT's ops, and under the
    /// fused memory format MEM_LOAD's and MEM_STORE's rows), under the same
    /// room rule ([`WindowedTraceBuilder::card_fill_streamed_rows`]).
    pub(crate) rows: RowFamilies,
}

impl CardFill {
    /// The families the card fills.
    const NAMES: [&'static str; 10] = [
        "keccak_rnd",
        "cpu",
        "memw_r",
        "memw_a",
        "load",
        "store",
        "memw",
        "shift",
        "mem_load",
        "mem_store",
    ];

    /// Every family.
    const EVERY: Self = Self {
        keccak_rnd: true,
        cpu: true,
        memw_r: true,
        rows: RowFamilies::ALL,
    };

    /// Unset or empty: every family (KECCAK_RND, CPU, MEMW_R and the row
    /// families) in a build with the card's kernels (the `cuda` feature),
    /// nothing in one without, where every fill from records would be the
    /// host's.
    const DEFAULT: Self = Self {
        keccak_rnd: cfg!(feature = "cuda"),
        cpu: cfg!(feature = "cuda"),
        memw_r: cfg!(feature = "cuda"),
        rows: if cfg!(feature = "cuda") {
            RowFamilies::ALL
        } else {
            RowFamilies::NONE
        },
    };

    /// Whether any streamed family is asked of the card.
    fn streamed(self) -> bool {
        self.cpu || self.memw_r || self.rows.any()
    }

    /// Each family's flag, in [`Self::NAMES`] order.
    fn flags(&mut self) -> [&mut bool; 10] {
        let rows = &mut self.rows;
        [
            &mut self.keccak_rnd,
            &mut self.cpu,
            &mut self.memw_r,
            &mut rows.memw_a,
            &mut rows.load,
            &mut rows.store,
            &mut rows.memw,
            &mut rows.shift,
            &mut rows.mem_load,
            &mut rows.mem_store,
        ]
    }

    fn name(mut self) -> String {
        let on = self.flags().map(|on| *on);
        let names: Vec<&str> = Self::NAMES
            .iter()
            .zip(on)
            .filter_map(|(name, on)| on.then_some(*name))
            .collect();
        if names.is_empty() {
            "off".to_string()
        } else {
            names.join(",")
        }
    }
}

/// `LAMBDA_VM_BLOCK_CARD_FILL`: `on` has the card fill every family's tables
/// ([`CardFill`]: KECCAK_RND, CPU, MEMW_R and the row families) from the
/// records their rows come from as their groups upload them, `off` none (the
/// host builds every table). Unset or empty is [`CardFill::DEFAULT`]: every
/// family in a build with the card's kernels, nothing without. Where the card
/// declines the room, a host reader fills them on the host. The tables' words,
/// and so the proof's bytes, are the same. Anything else refuses the prove.
pub(crate) fn block_card_fill() -> Result<CardFill, Error> {
    parse_card_fill(std::env::var("LAMBDA_VM_BLOCK_CARD_FILL").ok().as_deref())
}

/// [`block_card_fill`]'s reading of its variable.
fn parse_card_fill(value: Option<&str>) -> Result<CardFill, Error> {
    match value.map(str::trim) {
        None | Some("") => Ok(CardFill::DEFAULT),
        Some("off") => Ok(CardFill::default()),
        Some("on") => Ok(CardFill::EVERY),
        Some(v) => Err(Error::Prover(format!(
            "LAMBDA_VM_BLOCK_CARD_FILL={v}: off (the host builds every table) or on (the card \
             fills every family it can)"
        ))),
    }
}

/// Whether the streamed CPU and MEMW_R chunks may be held as their records,
/// and why not when they may not. They need the card asked for at least one of
/// the two families (`asked`), and G-pack's narrow chunks of a build that drops
/// its streamed ops (`narrow_dropping`). And they need a way off the host:
/// parked records drop only under live regeneration, so without it (a store,
/// or regeneration off) every one admitted would stay resident to the prove's
/// end, which no room rule bounds on a block that arms. Records kept on purpose
/// (a test's [`Deviations::records_keep`], `keep_records`) are undroppable by
/// design: the A/B control.
fn records_admission(
    asked: bool,
    narrow_dropping: bool,
    keep_records: bool,
    live_regen: bool,
) -> Result<(), &'static str> {
    if !asked {
        Err("off")
    } else if !narrow_dropping {
        Err("off (not a narrow build that drops its streamed ops)")
    } else if !keep_records && !live_regen {
        Err("off (parked records drop only under live regeneration, and it is off)")
    } else {
        Ok(())
    }
}

/// The most window walkers `LAMBDA_VM_BLOCK_WALKERS` takes: each walker holds
/// a window's state and walked lists, and this is the most whose memory was
/// measured on the largest blocks.
const MAX_WALKERS: usize = 4;

/// `LAMBDA_VM_BLOCK_WALKERS`: unset or empty walks the run's windows on
/// [`MAX_WALKERS`] threads (the default), each window from the bytes it
/// touches as they stood before it ([`parallel::walk_windows`]); `1` to
/// [`MAX_WALKERS`] on that many; `0` on one thread, the walk before the
/// walkers (the rollback). The tables and their order are the same. Anything
/// else refuses the prove.
fn block_walkers() -> Result<usize, Error> {
    parse_walkers(std::env::var("LAMBDA_VM_BLOCK_WALKERS").ok().as_deref())
}

/// [`block_walkers`] of the knob's value.
fn parse_walkers(value: Option<&str>) -> Result<usize, Error> {
    match value.map(str::trim) {
        None | Some("") => Ok(MAX_WALKERS),
        Some("0") => Ok(0),
        Some(v) => v
            .parse::<usize>()
            .ok()
            .filter(|n| (1..=MAX_WALKERS).contains(n))
            .ok_or_else(|| {
                Error::Prover(format!(
                    "LAMBDA_VM_BLOCK_WALKERS={v}: 0 (one walker), or 1 to {MAX_WALKERS} walkers"
                ))
            }),
    }
}

/// The rest's layout bytes ([`BlockOptions::rest_layout_bytes`]). On the
/// streamed finish that slides (production) they are its byte gate's budget:
/// what the rest holds built and not yet placed, or building, so they bound
/// the host memory the finish adds. A family's first table is admitted on an
/// estimate of eight bytes a cell, so a smaller gate, or a build cap under it,
/// starts the rest's tables one after another as earlier ones end; at 8 GiB,
/// building up to the gate, they start as the packer needs them. Where the
/// rest is laid out in waves (the finish not streamed, or not sliding) they
/// bound the rows each wave transposes: each transposition is parallel within
/// its table, so a wave of a few tables keeps the cores busy while bounding
/// the copies in flight.
pub const BLOCK_REST_LAYOUT_BYTES: usize = 8 << 30;

/// The estimated bytes of the rest's tables the sliding finish builds at once
/// by default ([`BlockOptions::rest_build_bytes`]): the gate's
/// [`BLOCK_REST_LAYOUT_BYTES`], so the gate alone bounds what builds.
pub const BLOCK_REST_BUILD_BYTES: usize = 8 << 30;

/// `BLOCK_WHIR_REST_BUILD=<MiB>`: the real-block tests' choice of
/// [`BlockOptions::rest_build_bytes`]; unset leaves production's
/// ([`BLOCK_REST_BUILD_BYTES`]). Anything but a positive MiB count is an
/// error; the gate's own budget is no cap beyond the gate.
fn parse_rest_build(value: Option<&str>) -> Result<Option<usize>, String> {
    match value.map(str::trim) {
        None => Ok(None),
        Some(mib) => match mib.parse::<usize>() {
            Ok(mib) if mib > 0 => Ok(Some(mib << 20)),
            _ => Err(format!("BLOCK_WHIR_REST_BUILD={mib}: a positive MiB count")),
        },
    }
}

/// `BLOCK_WHIR_REST_LAYOUT=all|<MiB>` and `BLOCK_WHIR_REST_BUILD=<MiB>`: the
/// real-block tests' choice of [`BlockOptions::rest_layout_bytes`] and
/// [`BlockOptions::rest_build_bytes`]; unset leaves the production ones.
pub(crate) fn rest_layout_from_env(options: &mut BlockOptions) {
    match parse_rest_build(std::env::var("BLOCK_WHIR_REST_BUILD").ok().as_deref()) {
        Ok(Some(bytes)) => options.rest_build_bytes = Some(bytes),
        Ok(None) => {}
        Err(e) => panic!("{e}"),
    }
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

/// `BLOCK_WHIR_KECCAK_LOG2=k` (0..=[`BLOCK_KECCAK_MAX_VARS`]),
/// `BLOCK_WHIR_ECSM_LOG2=k` (0..=[`BLOCK_ECSM_MAX_VARS`]) and
/// `BLOCK_WHIR_COPY_LOG2=k` (0..=[`BLOCK_COPY_MAX_VARS`]): the real-block
/// tests' KECCAK, ECSM and COPY cuts, to force a split on a block whose tables
/// fit one; unset leaves the production ones.
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
        (
            "BLOCK_WHIR_COPY_LOG2",
            BLOCK_COPY_MAX_VARS,
            &mut options.copy_rows_log2,
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

/// Threads a streamed prove lays its chunks out on
/// ([`BlockOptions::layout_workers`]). Three workers with two chunks ahead
/// bound the median block's windows: the layout thread stayed blocked handing
/// chunks on, and the accumulator waited on it. Six with five ahead leave the
/// walker to set the pace (ZEN5 020, P1 median: phase A −2.32 s against
/// three, peak RSS no higher). The groups and the proof bytes are the same at
/// any count. `BLOCK_WHIR_LAYOUT_WORKERS` overrides it in the tree harness
/// and the CLI.
pub const BLOCK_LAYOUT_WORKERS: usize = 6;

/// Streamed chunks laid out (or in the making) and not yet packed, beyond the
/// one the inline layout holds ([`BlockOptions::layout_ahead`]): one per
/// worker of [`BLOCK_LAYOUT_WORKERS`] and three more, so the workers go on
/// while the in-order packer waits for a slow chunk (one per worker held the
/// walk back once the windows were walked on several threads). What the extra
/// chunks would add to phase B's holding, the keep budget takes back
/// ([`BLOCK_KEEP_BELOW`]). `BLOCK_WHIR_LAYOUT_AHEAD` overrides it.
pub const BLOCK_LAYOUT_AHEAD: usize = 8;

/// A block proved in one proof.
///
/// The statement is [`crate::multilinear_prove::MultilinearVmProof`]'s; the
/// proof is read under the block's forked schedule, and only by
/// [`verify_block_whir`].
///
/// `L` is what the proof carries for local lookups ([`LocalPart`]): nothing
/// (`()`, today's format and type, archived byte for byte as before it), or
/// every group's local argue ([`LocalArgues`], under
/// [`BlockFormat::local_lookups`]). A verifier reads it under its own format,
/// never the proof's: a proof of the other kind is refused.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BlockWhirProof<L = ()> {
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
    /// Every carried stacked polynomial's commit-time out-of-domain answer, in
    /// polynomial order: one per stacked polynomial of every group (one per
    /// root of `proof.roots` unless a group's run shares one round-0 tree,
    /// `ZfFormat::whir_trees`), absorbed after the roots and before the
    /// block's first challenge, and each carried into its chain
    /// (`stark::multilinear_block::commit_ood_point`). The verifier refuses
    /// any other count.
    pub commit_ood: Vec<FieldElement<E>>,
    /// The fused memory format's tables: how many MEM_LOAD and MEM_STORE
    /// tables the block holds, right after STORE's in [`VmAirs::air_refs`]
    /// order. Absorbed into the statement under
    /// [`MemoryFormat::Fused`](crate::tables::fused_memory::MemoryFormat::Fused)
    /// alone (the verifier's format, never the proof's); under the split
    /// format the verifier refuses any but `[0, 0]` ([`fused_tables`]).
    pub fused_memory: [u64; 2],
    /// The word memory format's tables: how many of each family
    /// ([`WORD_TABLES`]) the block holds, after the fused tables' place in
    /// [`VmAirs::air_refs`] order. Absorbed into the statement under
    /// [`MemoryFormat::Words`](crate::tables::fused_memory::MemoryFormat::Words)
    /// alone; under any other format the verifier refuses any but all zero
    /// ([`word_counts_under`]). The last field, so a split or fused proof
    /// archives as before it with these counts after.
    pub word_memory: [u64; 7],
    /// Local lookups' messages ([`LocalPart`]): `()` in today's format, which
    /// archives as nothing, so today's proof keeps its bytes.
    pub local: L,
}

/// Every group's local argue, in group order (empty for a group with no
/// local interaction): what a proof under [`BlockFormat::local_lookups`]
/// carries in [`BlockWhirProof::local`].
pub type LocalArgues = Vec<LocalArgue<E>>;

/// What a [`BlockWhirProof`] carries for local lookups: nothing, or every
/// group's local argue.
pub trait LocalPart {
    /// The local argues, or `None` for a proof of today's format.
    fn local_argues(&self) -> Option<&[LocalArgue<E>]>;
}

impl LocalPart for () {
    fn local_argues(&self) -> Option<&[LocalArgue<E>]> {
        None
    }
}

impl LocalPart for LocalArgues {
    fn local_argues(&self) -> Option<&[LocalArgue<E>]> {
        Some(self)
    }
}

/// A proof of either kind, as a pipeline that proves either format carries
/// it: `None` for today's, the local argues under local lookups.
impl LocalPart for Option<LocalArgues> {
    fn local_argues(&self) -> Option<&[LocalArgue<E>]> {
        self.as_deref()
    }
}

/// A block proof of either kind ([`LocalPart`] for `Option<LocalArgues>`):
/// what the recursion's pipelines hold, whichever format they proved.
pub type AnyBlockWhirProof = BlockWhirProof<Option<LocalArgues>>;

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
    /// A streamed build's producer: what each of its threads waited on.
    pub producer: ProducerStamps,
    /// A streamed build's executor thread: its own work apart from its waits.
    pub executor: ExecutorStamps,
    /// How phase A held the committed columns, and the narrow tables phase B
    /// widened on the host, `(tables, cells)`: a device path widens on the
    /// card.
    pub narrow: multilinear_block::Narrowing,
    pub host_widens: (u64, u64),
    /// Tables filled from their records ([`block_card_fill`]) during the
    /// prove: `(on the card, on the host)`.
    pub records_fills: (u64, u64),
    /// Under `LAMBDA_VM_CARD_UPLOAD_TIMES=1`, the group uploads' tables timed
    /// on the card during the prove (`multilinear::records::upload_times`):
    /// `((packed tables, copy and widen s), (records tables, copy and fill s))`.
    pub upload_times: ((u64, f64), (u64, f64)),
    /// With [`BlockOptions::memlog`]: every memory term when the proof is
    /// done, bytes; empty without.
    pub mem_terms: Vec<(&'static str, usize)>,
    /// With live regeneration: the rule it decided by, and the largest host
    /// reading it decided on (the `BLOCK DROPS` line).
    pub drops: Option<String>,
    /// With live regeneration: the decisions the real-memory guard alone made
    /// ([`auto_wants`], [`real_hand_off_need`]); 0 without a pool.
    pub real_guard_alone: u64,
    /// With [`BlockOptions::regen`] set to regenerate: what the regenerator
    /// did, and its `BLOCK REGEN` lines (printed as phase B ends, not in
    /// [`Self::report`]).
    pub regen: Option<regen::RegenStamps>,
    /// A streamed build's `BLOCK HEAD` lines ([`head_log`], printed as phase A
    /// ends, not in [`Self::report`]); empty unless the head was stamped.
    pub head: Vec<String>,
    /// A streamed prove's phase B in unix seconds, (start, end), and the order
    /// it took the groups in: what a sampler's samples and another thread's
    /// stamps are placed against. Empty without a streamed prove.
    pub phase_b_unix: Option<(f64, f64)>,
    pub phase_b_order: Vec<usize>,
    /// Phase B ran with its overlap on ([`BlockOptions::phase_b_overlap`]).
    pub phase_b_overlap: bool,
    /// Its openings' thread read back through pinned staging (`Pinned`).
    pub phase_b_pinned: bool,
    /// The openings' threads' read-backs during phase B, `(through pinned
    /// staging, the stream's own way)`, and the bytes the pinned buffers held
    /// at its end.
    pub background_readbacks: (u64, u64, u64),
    /// Phase B stamped its driver thread's schedstat a group turn
    /// ([`phase_b_sched_from_env`]; each group's [`GroupStamps::sched_b`]).
    pub phase_b_sched: bool,
    /// The page-locked pool at phase A's end, when it is asked for
    /// ([`multilinear::pinned`]).
    pub pinned: Option<multilinear::pinned::Stats>,
}

/// A streamed build's layout, for the readout ([`BlockStamps::layout`]).
#[derive(Clone, Debug, Default)]
pub struct LayoutStamps {
    pub workers: usize,
    pub marks: Vec<(&'static str, f64)>,
    pub closed_at: Vec<f64>,
    /// The packer closed its open group ahead of BITWISE
    /// ([`close_before_bitwise`]).
    pub closed_before_bitwise: bool,
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
    /// The rest's tables laid out as the records their rows come from, for
    /// the card to fill ([`block_card_fill`]): how many, and the records'
    /// bytes.
    pub records_rest: (usize, usize),
    /// Each table of the rest in AIR order: its name, when its layout ended
    /// (seconds since the prove started) and its group.
    pub rest_tables: Vec<(String, f64, usize)>,
    /// G-pack ([`BlockOptions::gpack`]): whether it was on, and its counts
    /// (`tables::gpack::counts`) at phase A's end.
    pub gpack: (bool, [usize; 4]),
}

/// What a streamed build's producer threads waited on, seconds summed over
/// the windows (the workers' summed over the workers), for the readout
/// ([`BlockStamps::producer`]): the executor sending a window to the walk, the
/// walker waiting for a window and sending it on, the accumulator waiting for
/// a walked window and sending a chunk to the layout, the layout handing a
/// chunk to its workers, and the workers waiting for a permit and for a chunk.
/// A measurement; it changes nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProducerStamps {
    pub executor_send: f64,
    pub walker_recv: f64,
    pub walker_send: f64,
    pub accumulator_recv: f64,
    pub accumulator_send: f64,
    pub layout_send: f64,
    pub worker_permit: f64,
    pub worker_recv: f64,
}

/// Switches the block prove no longer reads, each with what it now always
/// does. Set, any of them refuses the prove ([`refuse_retired_words`]), so a
/// script that still sets one learns it at once instead of running the default
/// believing it ran its arm.
#[rustfmt::skip]
const RETIRED_WORDS: &[(&str, &str)] = &[
    ("LAMBDA_VM_EXEC_MEMORY", "the executor keeps its memory in pages"),
    ("LAMBDA_VM_EXEC_FIXED_ACCESS", "the executor's fixed-width access path is always on"),
    ("LAMBDA_VM_EXEC_DMA_BULK", "the DMA ecalls always move their range in one call"),
    ("LAMBDA_VM_EXEC_PREFAULT", "the executor writes each window into fresh buffers"),
    ("LAMBDA_VM_EXEC_RECYCLE_LOGS", "the executor writes each window into a fresh log buffer"),
    ("LAMBDA_VM_ECREC_DEFER", "ECREC witnesses are always built on their own pool, off the walk"),
    ("LAMBDA_VM_WALK_LEAN", "the lean walk always runs every part"),
    ("LAMBDA_VM_ROUTE_FUSED", "the accumulator's fused route is always taken"),
    ("LAMBDA_VM_WALK_SIZED", "the walk's lists are always sized from the window"),
    ("LAMBDA_VM_WALK_BATCH", "the walk always collects 2048 cycles at a time"),
    ("LAMBDA_VM_WALK_NO_CPU_LIST", "the walkers never keep a CPU list where the card fills CPU"),
    ("LAMBDA_VM_WALK_COUNT", "the walkers always count where they make the rows"),
    ("LAMBDA_VM_HANDOUT_COPY", "the hand-out always hands out the window parts the chunks lie in"),
    ("LAMBDA_VM_HANDOUT_COUNT_THREAD", "the hand-out always counts on its own thread"),
    ("LAMBDA_VM_BUILDER_CONCAT", "the whole-run builder always concatenates its windows in parallel"),
    ("LAMBDA_VM_P4_SLICED", "BITWISE's per-op sources are always cut into slices"),
    ("LAMBDA_VM_KR_PACKED_WAVE", "KECCAK_RND's packed chunks are always built at once"),
    ("LAMBDA_VM_BLOCK_CUT_PADDING", "tables of padding rows alone are always cut after the last real row"),
    ("LAMBDA_VM_BLOCK_LT_CONCAT", "LT's ops are never concatenated before chunking"),
    ("LAMBDA_VM_BLOCK_LT_MERGE", "LT's memory rows are always built by merging equal neighbours"),
    ("LAMBDA_VM_BLOCK_P4_BESIDE", "BITWISE is always counted from the built tables"),
    ("LAMBDA_VM_BLOCK_REST_BITWISE_LAST", "BITWISE is always counted from the built tables and handed on last"),
    ("LAMBDA_VM_BLOCK_BITWISE_FROM_TABLES", "BITWISE's lookups are always counted from the tables' rows, BITWISE last"),
    ("LAMBDA_VM_BLOCK_KECCAK_RND_FIRST", "KECCAK_RND's tables are always handed on first"),
    ("LAMBDA_VM_BLOCK_GPACK", "streamed chunks are always packed as built"),
    ("LAMBDA_VM_BLOCK_FINISH_PLAN", "the finish always runs its plan"),
    ("LAMBDA_VM_BLOCK_FINISH_STREAM", "the finish always streams the rest"),
    ("LAMBDA_VM_BLOCK_RECORDS_DROP", "records chunks always park and drop under regeneration"),
    ("LAMBDA_VM_BLOCK_COMMIT_OFF_POOL", "group commits always run on the committer's own threads"),
    ("LAMBDA_VM_BLOCK_UNVERIFIABLE_CLOCK", "a run past the 32-bit clock is always refused"),
    ("LAMBDA_VM_BLOCK_GROUP_QUEUE", "one closed group always waits for the card"),
    ("LAMBDA_VM_BLOCK_SPILL_RESERVE_PER_G", "the reserve is LAMBDA_VM_BLOCK_MEMORY_RESERVE_PER_G"),
    ("LAMBDA_VM_BLOCK_REGEN_EARLY", "the regenerator always starts as soon as the chunks' fates are known"),
    ("LAMBDA_VM_BLOCK_DERIVE_HOLD", "the verifier always drops each program once its child's shape is derived"),
    ("BLOCK_WHIR_DROP_OPS", "the streamed chunks' ops are always dropped"),
    ("BLOCK_WHIR_UPLOAD_AHEAD", "phase A always uploads ahead"),
    ("BLOCK_WHIR_KR_FINISH_CHUNKS", "KECCAK_RND is always built as its finish chunks"),
    ("BLOCK_WHIR_PACK_FINISHED", "the finish always packs its tables"),
    ("BLOCK_WHIR_WINDOW_LOG2", "the streamed window is always 2^20 cycles"),
    ("BLOCK_WHIR_REHASH_SERIAL", "kept-top paths are always re-hashed in parallel"),
    ("LAMBDA_VM_ARGUE_FUSED", "the argue always runs its fused zerocheck rounds"),
    ("LAMBDA_VM_ARGUE_INT_NODES", "the argue always takes its integer nodes"),
    ("LAMBDA_VM_ARGUE_GKR_GRUEN", "the GKR layers always run Gruen's rounds"),
    ("LAMBDA_VM_ARGUE_GKR_INPUT", "the GKR input layer is always built from the base columns"),
    ("LAMBDA_VM_ARGUE_NO_LIFT", "the argue never lifts its factors"),
    ("LAMBDA_VM_ARGUE_DEVICE_TABLES", "the argue's tables are always built on the card"),
    ("LAMBDA_VM_ARGUE_DEVICE_COLUMNS", "the argue's columns are always evaluated on the card"),
    ("LAMBDA_VM_ARGUE_LEAN_PROGRAM", "the argue's programs are always built on demand"),
    ("LAMBDA_VM_ARGUE_FUSED_WIDTHS", "the fused rounds take every width"),
    ("LAMBDA_VM_ARGUE_FUSED_LOG", "the fused rounds' timings print only under the cross-check"),
    ("LAMBDA_VM_ARGUE_TREE_SYNC", "the tree split does not time its parts"),
    ("LAMBDA_VM_ARGUE_BUSY_PROBE", "the argue's busy probe is gone"),
    ("LAMBDA_VM_WHIR_COSET_OPENINGS", "a coset-ordered commitment is always revived over its stack"),
    ("LAMBDA_VM_WHIR_COSET_INPLACE", "a coset opening always reads the stack where it lies"),
    ("LAMBDA_VM_NO_WHIR_ROOM_PARK", "the WHIR room always parks"),
    ("LAMBDA_VM_NO_WHIR_ROOM_RESIZE", "the WHIR room is always sized to its codeword"),
    ("LAMBDA_VM_NO_WHIR_FUSED_FOLD", "the WHIR fold is always fused"),
    ("LAMBDA_VM_WHIR_ENCODE_SUCCESSOR", "the first successor is always encoded"),
    ("LFM_WHIR_KEEP_FUTILE", "retained layers are always kept through a futile miss"),
    ("LFM_WHIR_WHOLE_TREES", "whole trees are always kept"),
    ("LAMBDA_VM_LDE_LEGACY", "every LDE runs through the column-major engine"),
    ("LAMBDA_VM_DEEP_INV_LEGACY", "the DEEP inverses are always row-wise up to their bound"),
    ("LAMBDA_VM_LDE_TRANSPOSE_UNBATCHED", "run copies are always batched"),
    ("LAMBDA_VM_STAGING_SHARED_SLAB", "staging always goes through pairs"),
    ("LAMBDA_VM_TREE_DOWNLOAD", "trees always download through staging"),
    ("LAMBDA_VM_RPX_WARP_MERKLE", "RPX Merkle tops always run a half-warp per parent"),
    ("LAMBDA_VM_RPX_GRIND_QUEUE", "the RPX grind always searches through its queue"),
    ("LAMBDA_VM_RPX_LIMB_PERMUTE", "the RPX permutation always runs on limbs"),
    ("W3_DATAFLOW", "nodes always prove as their own children land"),
    ("W3_STREAM_TOP", "the top always streams while a child is unproved"),
    ("W3_EXEC_EARLY", "nodes always execute from their emitted program"),
    ("W3_NODE_PIPE", "node programs are always built on the W3_EMIT_THREADS pool"),
    ("W3_EXEC_BESIDE_ARTIFACTS", "leaves always execute beside their artifacts"),
    ("W3_LEAF_FIRST", "phase B always leads with a leaf's groups when the early leaf runs"),
    ("LFM_ARTIFACT_PARALLEL", "artifact columns are always built in parallel"),
    ("LFM_DEVICE_ARTIFACTS", "artifact groups are always committed on the card when admitted"),
    ("LFM_WARM_DOMAINS", "domains are always warmed before the card holds"),
    ("LFM_VALIDATE_DENSE_COUNTS", "the validator always recounts densely"),
    ("LAMBDA_VM_WARM_PROVE_CONSTANTS", "first-use constants are always warmed"),
    ("LAMBDA_VM_LFM_LEAN_UPLOAD", "the socket always uploads lean rows"),
    ("LAMBDA_VM_COMPOSITION_SUBCOSET", "composition always runs on a sub-coset"),
    ("LAMBDA_VM_TREE_BUSY_PROBE", "the tree's busy probe is gone"),
    ("LAMBDA_VM_SHAPE_PROFILE", "the shape profile is gone"),
    ("LAMBDA_VM_BASE_PREP_ON_PROVER", "the epoch pipeline always prepares ahead of the prover thread"),
    ("LAMBDA_VM_TREE_PROGRAM_BUDGET_READING", "the program budget always reads live"),
];

/// Values of live switches that are no longer settings, each with what the
/// prove now always does instead. Set to one of them, the switch refuses the
/// prove ([`refuse_retired_words`]); its other values are read as before.
const RETIRED_VALUES: &[(&str, &str, &str)] = &[
    (
        "LAMBDA_VM_ARGUE_ROUND_SLOT",
        "0",
        "the GKR rounds' sums are always read from the slot (`check` also compares)",
    ),
    (
        "LAMBDA_VM_ARGUE_INPUT_REAL",
        "0",
        "the input layers are always kept without their padding (`check` also compares)",
    ),
    (
        "LAMBDA_VM_STREAM_CLASSES",
        "equal",
        "the background pool always runs at the least priority (`off` turns it off)",
    ),
];

/// [`RETIRED_WORDS`]: the first one set in `env`, whatever its value, refuses
/// the prove, naming the word and what the prove now always does; then a
/// switch set to one of its [`RETIRED_VALUES`] refuses it the same way.
fn refuse_retired_words(env: impl Fn(&str) -> Option<String>) -> Result<(), Error> {
    for &(word, now) in RETIRED_WORDS {
        if let Some(v) = env(word) {
            return Err(Error::Prover(format!(
                "{word}={} is no longer read: {now}; unset it",
                v.trim()
            )));
        }
    }
    for &(word, value, now) in RETIRED_VALUES {
        if env(word).is_some_and(|v| v.trim() == value) {
            return Err(Error::Prover(format!(
                "{word}={value} is no longer a setting: {now}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod retired_words_tests {
    use super::{RETIRED_VALUES, RETIRED_WORDS, refuse_retired_words};
    use crate::Error;

    /// Each retired switch refuses the prove when set, whatever its value
    /// (`0` and empty included), naming itself; unset, or another word set,
    /// nothing.
    #[test]
    fn a_retired_switch_refuses_the_prove_naming_itself() {
        assert!(refuse_retired_words(|_| None).is_ok());
        assert!(
            refuse_retired_words(|n| (n == "LAMBDA_VM_BLOCK_WALKERS").then(|| "4".to_string()))
                .is_ok()
        );
        // Every retired word is read nowhere else: the table is the whole list.
        assert_eq!(RETIRED_WORDS.len(), 85);
        for &(word, _) in RETIRED_WORDS {
            for value in ["0", "1", "", " 0 "] {
                let Err(Error::Prover(why)) =
                    refuse_retired_words(|n| (n == word).then(|| value.to_string()))
                else {
                    panic!("{word}={value} must refuse the prove");
                };
                assert!(
                    why.starts_with(word) && why.contains("is no longer read"),
                    "{why}"
                );
            }
        }
    }

    /// A live switch set to a value that is no longer a setting refuses the
    /// prove naming the switch and the value; its other values do not.
    #[test]
    fn a_retired_value_refuses_the_prove_and_the_others_do_not() {
        for &(word, value, _) in RETIRED_VALUES {
            for set in [value.to_string(), format!(" {value} ")] {
                let Err(Error::Prover(why)) =
                    refuse_retired_words(|n| (n == word).then(|| set.clone()))
                else {
                    panic!("{word}={set} must refuse the prove");
                };
                assert!(why.starts_with(&format!("{word}={value}")), "{why}");
            }
        }
        for (word, value) in [
            ("LAMBDA_VM_ARGUE_ROUND_SLOT", "1"),
            ("LAMBDA_VM_ARGUE_ROUND_SLOT", "check"),
            ("LAMBDA_VM_ARGUE_INPUT_REAL", "check"),
            ("LAMBDA_VM_STREAM_CLASSES", "off"),
            ("LAMBDA_VM_STREAM_CLASSES", "priority"),
        ] {
            assert!(
                refuse_retired_words(|n| (n == word).then(|| value.to_string())).is_ok(),
                "{word}={value} must prove"
            );
        }
    }
}

/// The block prove's interpreter and whether it runs in record mode, from
/// `LAMBDA_VM_EXEC_ENGINE` and `LAMBDA_VM_EXEC_HISTORY`, the build having
/// `walkers` walkers. In record mode the fast engine keeps the walk's access
/// history and writes it beside each window's logs; the walkers read it
/// instead of a lean pass's entries, the executor's window-start snapshots and
/// window states ([`parallel::walk_windows_from_history`]). The tables are the
/// same either way.
///
/// - Both unset (or empty): record mode on the fast engine, the default.
/// - `LAMBDA_VM_EXEC_HISTORY=0`: no record mode, on the engine
///   `LAMBDA_VM_EXEC_ENGINE` names, the reference when it is unset (the walk
///   from the lean pass's entries).
/// - `LAMBDA_VM_EXEC_ENGINE=reference`, the history unset: no record mode.
/// - `LAMBDA_VM_EXEC_HISTORY=1`: record mode, refused on the reference engine
///   and without walkers.
/// - No walkers (`LAMBDA_VM_BLOCK_WALKERS=0`), the history unset: no record
///   mode, on the engine named (the reference when unset).
///
/// Any other value of either refuses the prove.
pub(crate) fn exec_mode(walkers: usize) -> Result<(EngineKind, bool), Error> {
    exec_mode_from(
        std::env::var(executor::vm::engine::ENGINE_KNOB)
            .ok()
            .as_deref(),
        std::env::var("LAMBDA_VM_EXEC_HISTORY").ok().as_deref(),
        walkers,
    )
}

/// `LAMBDA_VM_POOL_EXEC_LOGS=1`: the executor writes each window's logs into a
/// buffer from the page-locked pool (`LAMBDA_VM_PINNED_POOL_GIB`), handed to it
/// before the window ([`executor::vm::engine::Engine::recycle`]), so the CPU
/// table's logs go up to the card at both of their uploads (phase A's commit,
/// phase B's argue) from page-locked memory, with no staging copy. A buffer
/// the pool cannot give is pageable, as today's, and counted so
/// ([`ExecutorStamps`]). Off by default and for any word but `1`; a no-op
/// without a locked pool or on the reference engine; the same logs, so the
/// same bytes. Read once a process.
pub const POOL_EXEC_LOGS_ENV: &str = "LAMBDA_VM_POOL_EXEC_LOGS";

/// [`POOL_EXEC_LOGS_ENV`]'s word: on only for `1`.
pub fn pool_exec_logs_word(word: Option<&str>) -> bool {
    crate::lfm::wlfm_knobs::knob_word_on(word)
}

/// A test's [`pool_exec_logs`], process wide (0 the knob, 1 off, 2 on).
static POOL_EXEC_LOGS_FOR_TEST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Tests only: [`pool_exec_logs`] forced for the whole process (`None`
/// restores the knob). A test that sets it runs alone.
#[doc(hidden)]
pub fn set_pool_exec_logs_for_test(on: Option<bool>) {
    let v = match on {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    POOL_EXEC_LOGS_FOR_TEST.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`POOL_EXEC_LOGS_ENV`] is on (or a test's override).
pub fn pool_exec_logs() -> bool {
    match POOL_EXEC_LOGS_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| pool_exec_logs_word(std::env::var(POOL_EXEC_LOGS_ENV).ok().as_deref()))
}

/// An empty log buffer of `capacity` logs, from the page-locked pool when it
/// is locked and has the room, pageable otherwise: a buffer the executor
/// writes a window into. From the pool's arena for the executor's buffers
/// ([`multilinear::pinned::with_pinned_logs`]): each run's executor is a new
/// thread, which a shared arena would bind elsewhere than the last run's, so
/// the windows freed by one run would sit in an arena the next never asks and
/// the pool would run out by the second run.
pub(crate) fn pooled_log_buffer(capacity: usize) -> Vec<executor::vm::logs::Log> {
    multilinear::pinned::with_pinned_logs(
        capacity * std::mem::size_of::<executor::vm::logs::Log>(),
        || Vec::with_capacity(capacity),
    )
}

/// [`exec_mode`] of the two variables' values.
fn exec_mode_from(
    engine: Option<&str>,
    history: Option<&str>,
    walkers: usize,
) -> Result<(EngineKind, bool), Error> {
    let engine = match engine.map(str::trim) {
        None | Some("") => None,
        named => Some(EngineKind::from_value(named).map_err(Error::Execution)?),
    };
    match parse_exec_history(history)? {
        Some(false) => Ok((engine.unwrap_or(EngineKind::Reference), false)),
        Some(true) => {
            if engine == Some(EngineKind::Reference) {
                return Err(Error::Prover(
                    "LAMBDA_VM_EXEC_HISTORY=1 needs the fast engine (LAMBDA_VM_EXEC_ENGINE=fast \
                     or unset)"
                        .into(),
                ));
            }
            if walkers == 0 {
                return Err(Error::Prover(
                    "LAMBDA_VM_EXEC_HISTORY=1 needs walkers (LAMBDA_VM_BLOCK_WALKERS)".into(),
                ));
            }
            Ok((EngineKind::Fast, true))
        }
        None if engine == Some(EngineKind::Reference) || walkers == 0 => {
            Ok((engine.unwrap_or(EngineKind::Reference), false))
        }
        None => Ok((EngineKind::Fast, true)),
    }
}

/// `LAMBDA_VM_EXEC_HISTORY`'s value: `Some(true)` for `1`, `Some(false)` for
/// `0`, `None` when unset or empty ([`exec_mode`]'s default); anything else
/// is refused.
fn parse_exec_history(value: Option<&str>) -> Result<Option<bool>, Error> {
    match value.map(str::trim) {
        None | Some("") => Ok(None),
        Some("0") => Ok(Some(false)),
        Some("1") => Ok(Some(true)),
        Some(v) => Err(Error::Prover(format!(
            "LAMBDA_VM_EXEC_HISTORY={v}: 1 (the walkers read the executor's history), 0 (they \
             do not) or unset (the default: 1 on the fast engine with walkers)"
        ))),
    }
}

/// `LAMBDA_VM_WORD_STREAM`'s reading: unset, empty or `1` (the default), the
/// word memory format's word families are cut into chunks as the walk fills
/// them ([`WindowedTraceBuilder::stream_word_tables`]); `0`, the finish builds
/// them. The tables are the same. Anything else refuses the prove.
fn parse_word_stream(value: Option<&str>) -> Result<bool, Error> {
    match value.map(str::trim) {
        None | Some("" | "1") => Ok(true),
        Some("0") => Ok(false),
        Some(v) => Err(Error::Prover(format!(
            "LAMBDA_VM_WORD_STREAM={v}: 1 (the default: cut into chunks as the walk fills them) \
             or 0 (the finish builds the word families)"
        ))),
    }
}

/// `LAMBDA_VM_LIGHT_WALK`: the walkers' light walk
/// ([`WindowedTraceBuilder::light_walk`]). Unset, empty or `2` (the default):
/// record mode's cycles that are no ECALL walked without their cells, each
/// load or store from its access record, and each window routed by its
/// walker; `1`: the cells skipped, the accumulator routing; `0`: the walk
/// before it (the rollback). The tables are the same at every level. Anything
/// else refuses the prove.
fn parse_light_walk(value: Option<&str>) -> Result<u8, Error> {
    match value.map(str::trim) {
        None | Some("" | "2") => Ok(2),
        Some("0") => Ok(0),
        Some("1") => Ok(1),
        Some(v) => Err(Error::Prover(format!(
            "LAMBDA_VM_LIGHT_WALK={v}: 2 (the default: record mode's cycles walked without their \
             cells, each window routed by its walker), 1 (the cells skipped only) or 0 (the walk \
             before it)"
        ))),
    }
}

/// `LAMBDA_VM_EXEC_HISTORY_CHECK=N`: in record mode, a check thread runs
/// today's lean pass over every window (its own timestamps and registers,
/// from the logs) and re-walks every `N`-th window as today's walkers do, from
/// the pass's entries and the executor's memory at the window's start; the
/// history's starting registers, first cells and the walkers' lists must
/// agree with it. Unset, empty or `0`: no check thread. Anything else refuses
/// the prove.
pub(crate) fn exec_history_check() -> Result<usize, Error> {
    parse_exec_history_check(
        std::env::var("LAMBDA_VM_EXEC_HISTORY_CHECK")
            .ok()
            .as_deref(),
    )
}

/// [`exec_history_check`]'s reading of its variable.
fn parse_exec_history_check(value: Option<&str>) -> Result<usize, Error> {
    match value.map(str::trim) {
        None | Some("") => Ok(0),
        Some(v) => v.parse().map_err(|_| {
            Error::Prover(format!(
                "LAMBDA_VM_EXEC_HISTORY_CHECK={v}: a window count (0: no re-walk)"
            ))
        }),
    }
}

/// The streamed build's executor thread, its own work apart from its waits
/// ([`BlockStamps::executor`]); with `send` ([`ProducerStamps::executor_send`])
/// they add up to the thread's span. A measurement; it changes nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExecutorStamps {
    /// When the thread started, since the streamed prove started.
    pub started: f64,
    /// The executor's setup: the program and the private input loaded.
    pub setup: f64,
    /// Executing, the logs written as it goes.
    pub run: f64,
    /// Handing each window out: its logs moved out as they lie
    /// ([`Executor::resume_owned`]) and the memory snapshot at the next
    /// window's start.
    pub handoff: f64,
    /// The windows handed out.
    pub windows: usize,
    /// Whether the executor wrote into buffers from the page-locked pool
    /// ([`POOL_EXEC_LOGS_ENV`] on, the pool locked, the fast engine).
    pub logs_pooled: bool,
    /// The windows whose logs lay in the pool, and their buffers' bytes; and
    /// those that lay in pageable memory.
    pub pooled_windows: usize,
    pub pooled_bytes: u64,
    pub pageable_windows: usize,
    pub pageable_bytes: u64,
}

impl ExecutorStamps {
    /// Counts a handed-out window's buffer as pooled or pageable.
    fn note_logs(&mut self, logs: &[executor::vm::logs::Log], capacity: usize) {
        let bytes =
            (capacity.max(logs.len()) * std::mem::size_of::<executor::vm::logs::Log>()) as u64;
        if multilinear::pinned::is_pinned(logs.as_ptr().cast(), bytes as usize) {
            self.pooled_windows += 1;
            self.pooled_bytes += bytes;
        } else {
            self.pageable_windows += 1;
            self.pageable_bytes += bytes;
        }
    }

    /// The `BLOCK EXECUTOR LOG BUFFERS:` line.
    fn log_buffers_line(&self) -> String {
        format!(
            "BLOCK EXECUTOR LOG BUFFERS: from the pool {} ({}) · {} windows in the pool, {:.3} GB · {} pageable, {:.3} GB\n",
            if self.logs_pooled { "on" } else { "off" },
            POOL_EXEC_LOGS_ENV,
            self.pooled_windows,
            self.pooled_bytes as f64 / 1e9,
            self.pageable_windows,
            self.pageable_bytes as f64 / 1e9,
        )
    }

    /// The `BLOCK EXECUTOR:` line, `send` the thread's blocked sends.
    fn line(&self, send: f64) -> String {
        format!(
            "BLOCK EXECUTOR: started at {:.2}s · setup {:.2}s · execute {:.2}s · hand-off {:.2}s ({} windows, logs moved) · send-blocked {:.2}s · own {:.2}s\n",
            self.started,
            self.setup,
            self.run,
            self.handoff,
            self.windows,
            send,
            self.setup + self.run + self.handoff,
        )
    }
}

/// [`ProducerStamps`] as the threads add to them, in nanoseconds.
#[derive(Default)]
struct ProducerWaits {
    executor_send: std::sync::atomic::AtomicU64,
    walker_recv: std::sync::atomic::AtomicU64,
    walker_send: std::sync::atomic::AtomicU64,
    accumulator_recv: std::sync::atomic::AtomicU64,
    accumulator_send: std::sync::atomic::AtomicU64,
    layout_send: std::sync::atomic::AtomicU64,
    worker_permit: std::sync::atomic::AtomicU64,
    worker_recv: std::sync::atomic::AtomicU64,
}

impl ProducerWaits {
    /// Adds the time since `since` to `wait`.
    fn add(wait: &std::sync::atomic::AtomicU64, since: Instant) {
        let nanos = u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX);
        wait.fetch_add(nanos, std::sync::atomic::Ordering::Relaxed);
    }

    fn stamps(&self) -> ProducerStamps {
        let s = |wait: &std::sync::atomic::AtomicU64| {
            wait.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9
        };
        ProducerStamps {
            executor_send: s(&self.executor_send),
            walker_recv: s(&self.walker_recv),
            walker_send: s(&self.walker_send),
            accumulator_recv: s(&self.accumulator_recv),
            accumulator_send: s(&self.accumulator_send),
            layout_send: s(&self.layout_send),
            worker_permit: s(&self.worker_permit),
            worker_recv: s(&self.worker_recv),
        }
    }
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
                "BLOCK GROUP {g}: tables {} · polys {} · cells {:.1} M || A wait {:.3} upload {:.3} paid {:.3} commit {:.3} retire {:.3} from@{:.2} done@{:.2} || B upload {:.3} argue {:.3} encode {:.3} open {:.3}{}\n",
                s.tables,
                s.polys,
                s.cells as f64 / 1e6,
                s.wait_a,
                s.upload_a,
                s.upload_paid,
                s.commit,
                s.retire,
                s.committed_at - s.commit - s.retire - s.ood,
                s.committed_at,
                s.upload_b,
                s.argue,
                s.encode,
                s.open,
                // Under the overlap: what of the openings phase B waited for.
                if self.phase_b_overlap {
                    format!(" paid {:.3}", s.open_paid)
                } else {
                    String::new()
                },
            ));
        }
        // Phase A's commit-time out-of-domain answers: how many, where their
        // columns were evaluated, and what they cost the committer.
        let ood_max = self.groups.iter().map(|g| g.ood).fold(0.0, f64::max);
        out.push_str(&format!(
            "BLOCK COMMIT OOD: answers {} · columns on card {} · on host {} · seconds {:.3} (max group {:.3})\n",
            polys,
            self.groups.iter().map(|g| g.ood_counts.on_card).sum::<usize>(),
            self.groups.iter().map(|g| g.ood_counts.on_host).sum::<usize>(),
            sum(|g| g.ood),
            ood_max,
        ));
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
            let p = &self.producer;
            out.push_str(&format!(
                "BLOCK PRODUCER: executor send-blocked {:.2}s · walker recv-wait {:.2}s send-blocked {:.2}s · accumulator recv-wait {:.2}s send-blocked {:.2}s · layout hand-on blocked {:.2}s · workers permit-wait {:.2}s recv-wait {:.2}s (summed)\n",
                p.executor_send,
                p.walker_recv,
                p.walker_send,
                p.accumulator_recv,
                p.accumulator_send,
                p.layout_send,
                p.worker_permit,
                p.worker_recv,
            ));
            if self.executor.windows > 0 {
                out.push_str(&self.executor.line(p.executor_send));
                out.push_str(&self.executor.log_buffers_line());
            }
            let h = &self.windows.handout;
            let tables: Vec<String> = StreamTable::ALL
                .iter()
                .filter(|t| h.chunks[t.position()] > 0)
                .map(|&t| {
                    let i = t.position();
                    format!(
                        "{} {:.2} · {:.2} · {} · {:.1} M",
                        stream_base(t),
                        h.take[i],
                        h.count[i],
                        h.chunks[i],
                        h.copied[i] as f64 / 1e6,
                    )
                })
                .collect();
            out.push_str(&format!(
                "BLOCK HANDOUT: take {:.2}s · count {:.2}s · copied {:.1} M ops || per table (take s · count s · chunks · copied ops): {}\n",
                h.take.iter().sum::<f64>(),
                h.count.iter().sum::<f64>(),
                h.copied.iter().sum::<usize>() as f64 / 1e6,
                tables.join(" | "),
            ));
            out.push_str(&match &self.windows.count_thread {
                Some(c) => {
                    let tables: Vec<String> = StreamTable::ALL
                        .iter()
                        .filter(|t| c.chunks[t.position()] > 0)
                        .map(|&t| {
                            let i = t.position();
                            format!("{} {:.2} · {}", stream_base(t), c.busy[i], c.chunks[i])
                        })
                        .collect();
                    format!(
                        "BLOCK HANDOUT COUNT: on a thread of its own · busy {:.2}s · waited for chunks {:.2}s · the windows' end waited {:.2}s for its last chunks and {:.3}s to add its counts || per table (busy s · chunks): {}\n",
                        c.busy.iter().sum::<f64>(),
                        c.idle,
                        c.drain,
                        c.merge,
                        tables.join(" | "),
                    )
                }
                None => "BLOCK HANDOUT COUNT: on the accumulator's thread (the count above)\n"
                    .to_string(),
            });
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
            let c = &self.windows.card_fill;
            let ((widens, widen_s), (fills, fill_s)) = self.upload_times;
            out.push_str(&format!(
                "BLOCK CARD FILL: KECCAK_RND {} tables held as their records, {:.2} MiB · filled \
                 from records on the card {} · on the host {} · CPU chunks {} as records, {} \
                 filled on the host · MEMW_R chunks {} as records, {} filled on the host · \
                 streamed records {:.2} GiB · room ran out at {} · CPU logs out of step {} · \
                 card upload (timed) packed {widens} tables {widen_s:.3} s, records {fills} \
                 tables {fill_s:.3} s · MEMW_A chunks {} as records, {} filled on the host · \
                 LOAD chunks {} as records, {} filled on the host · STORE chunks {} as records, \
                 {} filled on the host · MEMW chunks {} as records, {} filled on the host · \
                 SHIFT chunks {} as records, {} filled on the host · MEM_LOAD chunks {} as \
                 records, {} filled on the host · MEM_STORE chunks {} as records, {} filled on \
                 the host\n",
                l.records_rest.0,
                l.records_rest.1 as f64 / (1u64 << 20) as f64,
                self.records_fills.0,
                self.records_fills.1,
                c.cpu[0],
                c.cpu[1],
                c.memw_r[0],
                c.memw_r[1],
                c.records_bytes as f64 / (1u64 << 30) as f64,
                c.room_out
                    .map_or("never".to_string(), |(table, index)| format!(
                        "{table:?}[{index}]"
                    )),
                c.out_of_step,
                c.memw_a[0],
                c.memw_a[1],
                c.load[0],
                c.load[1],
                c.store[0],
                c.store[1],
                c.memw[0],
                c.memw[1],
                c.shift[0],
                c.shift[1],
                c.mem_load[0],
                c.mem_load[1],
                c.mem_store[0],
                c.mem_store[1],
            ));
            let released = self.groups.iter().filter(|g| g.records_released > 0);
            let mut pages = multilinear::records::PagesReturned::default();
            for g in &self.groups {
                pages += g.records_pages;
            }
            out.push_str(&format!(
                "BLOCK CARD FILL released: records let go in phase B after their groups' \
                 openings {:.3} GiB · {} groups · pages handed back to the OS {:.3} GiB of {} \
                 buffers ({} shared, {} refused) in {:.2} s on a thread of their own\n",
                released.clone().map(|g| g.records_released).sum::<usize>() as f64
                    / (1u64 << 30) as f64,
                released.count(),
                pages.bytes as f64 / (1u64 << 30) as f64,
                pages.buffers,
                pages.shared,
                pages.failed,
                self.groups
                    .iter()
                    .map(|g| g.records_return_secs)
                    .sum::<f64>(),
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
            "BLOCK RECOMMIT: 0 (a revived commitment builds no tree) · first-round paths from kept tops: {} calls · {} leaves re-hashed · on the card {card_calls} calls in {card_secs:.3} s · on the host (fallback) {host_calls} calls in {host_secs:.3} s · dropped {} · phase B on one thread + one upload helper{}\n",
            self.top_paths.0,
            self.top_paths.1,
            self.drop_levels
                .map_or("-".to_string(), |d| d.to_string()),
            if self.phase_b_overlap {
                " + one opening helper"
            } else {
                ""
            },
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
        // Under the overlap the two ran at once: each transition's own window
        // (in group order of the argue), measured.
        let overlap: Vec<String> = if self.phase_b_overlap {
            self.groups
                .iter()
                .filter(|g| g.transition_reserved > 0)
                .map(|g| format!("{}", mib(g.transition_reserved)))
                .collect()
        } else {
            self.groups
                .windows(2)
                .map(|w| {
                    let both =
                        w[0].open_reserved + w[1].argue_reserved.saturating_sub(w[1].argue_base);
                    format!("{}", mib(both))
                })
                .collect()
        };
        out.push_str(&format!(
            "BLOCK PHASE-B LEDGER: budget {} MiB · per group base/argue/room/open [{}] MiB · open(g) beside argue(g+1) [{}] MiB{} · revive room refusals {} · turn refusals {} · GKR tree refusals {} · openings on the host {} · round-0 openings re-encoded on the host {} ({} stacks, {:.2} s)\n",
            mib(multilinear::gpu::reserve_budget()),
            per_group.join(", "),
            overlap.join(", "),
            if self.phase_b_overlap { " measured" } else { "" },
            multilinear::gpu::room_revive_refusals(),
            multilinear::gpu::room_turn_refusals(),
            multilinear::gpu::gkr_tree_refusals(),
            multilinear::gpu::open_host_fallbacks(),
            multilinear::gpu::open_host_reencodes().0,
            multilinear::gpu::open_host_reencodes().1,
            multilinear::gpu::open_host_reencodes().2,
        ));
        out.push_str(&self.overlap_line());
        if self.phase_b_sched {
            out.push_str(&self.sched_line());
        }
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
        // The real card beside phase A's ledger: what the memory pool held
        // before and after each group's commit (it keeps freed blocks, so the
        // values only rise; a commit's growth is what it, and what ran beside
        // it, took from the card beyond what the pool already held).
        let pool: Vec<String> = self
            .groups
            .iter()
            .map(|g| format!("{}/{}", mib(g.commit_pool_before), mib(g.commit_pool_after)))
            .collect();
        out.push_str(&format!(
            "BLOCK PHASE-A POOL: per group reserved before/after its commit [{}] MiB\n",
            pool.join(", "),
        ));
        out.push_str(&multilinear_block::cs_probe_lines(&self.groups));
        out.push_str(&multilinear_block::zc_lane_lines(&self.groups));
        out.push_str(&multilinear_block::local_lane_lines(&self.groups));
        out.push_str(&multilinear_block::merge_probe_lines(&self.groups));
        if let Some(p) = self.pinned {
            let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
            out.push_str(&format!(
                "BLOCK PINNED POOL: on · {:.2} GiB, {:.2} locked in {:.3}s before the prove · handed out {:.2} GiB, then {:.2} GiB pageable once spent · phase A's uploads {:.2} of {:.2} GiB from the pool ({:.0} %) · at most {:.2} GiB allocated and not yet uploaded · held at phase A's end {} · lists gave back {:.2} GiB of tails in {} shrinks ({} moved out) · asked {:.2} of {:.2} GiB given ({:.1} %)\n",
                gib(p.pool_bytes),
                gib(p.locked_bytes),
                p.lock_seconds,
                gib(p.carved_bytes),
                gib(p.spilled_bytes),
                gib(p.upload_pinned),
                gib(p.upload_bytes),
                if p.upload_bytes > 0 { 100.0 * p.upload_pinned as f64 / p.upload_bytes as f64 } else { 0.0 },
                gib(p.pending_most),
                p.held_bytes
                    .map_or("unread".to_string(), |b| format!("{:.2} GiB", gib(b))),
                gib(p.shrunk_bytes),
                p.shrinks,
                p.shrink_moved,
                gib(p.requested_bytes),
                gib(p.usable_bytes),
                if p.usable_bytes > 0 { 100.0 * p.requested_bytes as f64 / p.usable_bytes as f64 } else { 0.0 },
            ));
        } else {
            out.push_str("BLOCK PINNED POOL: off\n");
        }
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
        if let Some((start, end)) = self.phase_b_unix {
            let spans: Vec<String> = self
                .phase_b_order
                .iter()
                .filter_map(|&g| {
                    self.groups
                        .get(g)
                        .map(|s| format!("{g}@{:.3}-{:.3}", s.start_b, s.end_b))
                })
                .collect();
            out.push_str(&format!(
                "BLOCK PHASE-B SPAN: unix {start:.3} to {end:.3} · groups in phase B's order (group@start-end, seconds into phase B): {}\n",
                spans.join(" ")
            ));
        }
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
        if let Some(drops) = &self.drops {
            out.push_str(&format!("BLOCK DROPS: {drops}\n"));
        }
        out
    }

    /// Phase B's overlap ([`BlockOptions::phase_b_overlap`]), on or off: the
    /// transitions whose openings ran beside the next argue and those that ran
    /// alone after a refusal; the openings' seconds beside an argue, how many
    /// of them the driver still waited for (paid) and the rest (hidden), and
    /// the seconds of the openings run alone; the argues' seconds; the largest
    /// transition's ledger peak; the openings' stream draws by pool, beside an
    /// argue (whose foreground count holds the argue's draws too) and alone
    /// (theirs only); the card's drain-and-trims. Off, every opening runs
    /// alone.
    /// Phase B's driver thread over its group turns
    /// ([`GroupStamps::sched_b`]): its CPU time and run-queue delay in all and
    /// by tenth of phase B (a turn counted in the tenth it started in), and
    /// the first three tenths', when the regenerators are busiest; or why
    /// there is nothing to print. A delay that rises where the card idles
    /// says the thread was runnable and not running: CPU contention.
    fn sched_line(&self) -> String {
        let turns: Vec<(f64, u64, u64)> = self
            .groups
            .iter()
            .filter_map(|g| g.sched_b.map(|(cpu, wait)| (g.start_b, cpu, wait)))
            .collect();
        if turns.is_empty() {
            return "BLOCK PHASE-B SCHED: unavailable (/proc/thread-self/schedstat did not read)\n"
                .to_string();
        }
        if turns.iter().all(|&(_, cpu, wait)| cpu == 0 && wait == 0) {
            return "BLOCK PHASE-B SCHED: unavailable (schedstat reads zeros: the kernel keeps no \
                    scheduler statistics)\n"
                .to_string();
        }
        let ns = 1e-9;
        let mut cpu = [0.0f64; 10];
        let mut wait = [0.0f64; 10];
        for &(start, c, w) in &turns {
            let tenth = if self.phase_b > 0.0 {
                ((start / self.phase_b * 10.0) as usize).min(9)
            } else {
                0
            };
            cpu[tenth] += c as f64 * ns;
            wait[tenth] += w as f64 * ns;
        }
        let list = |v: &[f64]| {
            v.iter()
                .map(|x| format!("{x:.2}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "BLOCK PHASE-B SCHED: the driver thread over {} group turns · CPU {:.3} s · run-queue delay \
             {:.3} s · first three tenths of phase B: CPU {:.3} s, run-queue delay {:.3} s · by tenth \
             (a turn in the tenth it began) CPU [{}] · run-queue delay [{}]\n",
            turns.len(),
            cpu.iter().sum::<f64>(),
            wait.iter().sum::<f64>(),
            cpu[..3].iter().sum::<f64>(),
            wait[..3].iter().sum::<f64>(),
            list(&cpu),
            list(&wait),
        )
    }

    fn overlap_line(&self) -> String {
        let beside = || self.groups.iter().filter(|g| g.overlapped);
        let alone = || self.groups.iter().filter(|g| !g.overlapped);
        // Summed from +0.0: an empty f64 sum is -0.0, which prints a sign.
        let sum = |values: &mut dyn Iterator<Item = f64>| values.fold(0.0, |a, b| a + b);
        let opened = sum(&mut beside().map(|g| g.encode + g.open));
        let paid = sum(&mut beside().map(|g| g.open_paid));
        let draws = |groups: &mut dyn Iterator<Item = &GroupStamps>| {
            groups.fold((0u64, 0u64), |(f, b), g| {
                (f + g.open_draws.0, b + g.open_draws.1)
            })
        };
        let (beside_f, beside_b) = draws(&mut beside());
        let (alone_f, alone_b) = draws(&mut alone());
        let peak = self
            .groups
            .iter()
            .map(|g| g.transition_reserved)
            .max()
            .unwrap_or(0);
        format!(
            "BLOCK PHASE-B OVERLAP: {} · transitions overlapped {} of {} (serial after a refusal {}) · openings beside an argue {opened:.3} s, hidden {:.3} s, paid {paid:.3} s + alone {:.3} s · argue {:.3} s · transition ledger peak max {} MiB (budget {} MiB) · openings' stream draws beside an argue background {beside_b} / foreground {beside_f} (the argue's too), alone background {alone_b} / foreground {alone_f} · trims {}{}\n",
            match (self.phase_b_overlap, self.phase_b_pinned) {
                (_, true) => "pinned",
                (true, false) => "on",
                (false, false) => "off",
            },
            beside().count(),
            self.groups.len().saturating_sub(1),
            self.groups
                .iter()
                .filter(|g| g.serial_after_refusal)
                .count(),
            opened - paid,
            sum(&mut alone().map(|g| g.open_paid)),
            self.groups.iter().map(|g| g.argue).sum::<f64>(),
            peak >> 20,
            multilinear::gpu::reserve_budget() >> 20,
            multilinear::gpu::device_trims(),
            if self.phase_b_overlap {
                let (pinned, pageable, bytes) = self.background_readbacks;
                format!(
                    " · the openings' read-backs pinned {pinned} / pageable {pageable} ({:.1} MiB pinned)",
                    bytes as f64 / (1u64 << 20) as f64
                )
            } else {
                String::new()
            },
        )
    }
}

/// The CPU tables' heights a block statement states, at the CPU tables'
/// positions (`crate::cpu_proof_start`), each at most [`BLOCK_CPU_MAX_VARS`]
/// variables. [`BlockAirInputs::new`] then checks them against the clock.
pub(crate) fn block_cpu_heights(
    counts: &TableCounts,
    table_num_vars: &[u8],
) -> Result<Vec<usize>, Error> {
    let start = crate::cpu_proof_start(true, counts);
    let stated = start
        .checked_add(counts.cpu)
        .and_then(|end| table_num_vars.get(start..end))
        .ok_or_else(|| {
            Error::InvalidTableCounts("the statement states fewer heights than CPU tables".into())
        })?;
    stated
        .iter()
        .enumerate()
        .map(|(k, &vars)| {
            if usize::from(vars) > BLOCK_CPU_MAX_VARS {
                return Err(Error::InvalidTableCounts(format!(
                    "CPU[{k}] states 2^{vars} rows — a block takes at most 2^{BLOCK_CPU_MAX_VARS}"
                )));
            }
            Ok(1usize << vars)
        })
        .collect()
}

/// The counts a block statement may declare: [`TableCounts::validate`], with
/// COPY, KECCAK, KECCAK_RND, ECSM and ECDAS chunked (at most
/// [`BLOCK_MAX_COPY`], [`BLOCK_MAX_KECCAK`], [`BLOCK_MAX_KECCAK_RND`],
/// [`BLOCK_MAX_ECSM`] and [`BLOCK_MAX_ECDAS`] tables).
pub fn validate_block_counts(counts: &TableCounts) -> Result<(), Error> {
    for (name, count, max) in [
        ("copy", counts.copy, BLOCK_MAX_COPY),
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
    rest.copy = rest.copy.min(1);
    rest.keccak = rest.keccak.min(1);
    rest.keccak_rnd = rest.keccak_rnd.min(1);
    rest.ecsm = rest.ecsm.min(1);
    rest.ecdas = rest.ecdas.min(1);
    rest.validate()
}

/// The proof-order index range of each chunked table, with its height cap in
/// variables. The order is [`VmAirs::air_refs`]'s: the [`FIXED_TABLE_COUNT`]
/// fixed tables, then COPY, KECCAK, KECCAK_RND, ECSM, ECDAS, … (pinned
/// against the AIR names by `chunked_table_ranges_name_the_chunked_airs`).
/// Counts whose total passed [`TableCounts::total`] cannot overflow these sums.
pub(crate) fn chunked_table_ranges(
    counts: &TableCounts,
) -> [(&'static str, std::ops::Range<usize>, usize); 5] {
    let copy = FIXED_TABLE_COUNT;
    let keccak = copy + counts.copy;
    let keccak_rnd = keccak + counts.keccak;
    let ecsm = keccak_rnd + counts.keccak_rnd;
    let ecdas = ecsm + counts.ecsm;
    [
        ("COPY", copy..keccak, BLOCK_COPY_MAX_VARS),
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

/// BITWISE's position in a block's tables: the first fixed table
/// ([`crate::VmAirs::air_refs`]'s order).
pub(crate) const BITWISE_TABLE: usize = 0;

/// BITWISE's stated height, held to its fixed 2^20 rows
/// ([`NUM_VARS`](crate::tables::bitwise::NUM_VARS)), the only height its
/// preprocessed columns have. Refused by name here rather
/// than left to the preprocessed check's dimension mismatch further on: the
/// WHIR query count is derived from the statement's tallest stack, and
/// BITWISE's (21 columns of 2^20 rows, 25 variables) is what holds it at the
/// production count for every block, so the pin is a rule of its own.
pub(crate) fn check_bitwise_height(table_num_vars: &[u8]) -> Result<(), Error> {
    match table_num_vars.get(BITWISE_TABLE) {
        Some(&n) if usize::from(n) == crate::tables::bitwise::NUM_VARS => Ok(()),
        stated => Err(Error::InvalidTableCounts(format!(
            "BITWISE states {} — its table is 2^{} rows",
            stated.map_or("no height".to_string(), |n| format!("2^{n} rows")),
            crate::tables::bitwise::NUM_VARS
        ))),
    }
}

/// Every chunked table under its cap, against the statement's heights: COPY
/// at most [`BLOCK_COPY_MAX_VARS`] variables, KECCAK
/// [`BLOCK_KECCAK_MAX_VARS`], KECCAK_RND [`BLOCK_KECCAK_RND_MAX_VARS`], ECSM
/// [`BLOCK_ECSM_MAX_VARS`] and ECDAS [`BLOCK_ECDAS_MAX_VARS`]. Run once the heights are known to be one per
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

/// Under the in-row register format MEMW_R's slot holds the residual register tables, whose
/// rows the verifier caps ([`crate::tables::register_residual::MAX_ROWS_TOTAL`]): a
/// keep-alive has no requester, so without the cap a statement could state
/// enough of them to carry a register's clock past the field.
/// `residual_num_vars` are the residual tables' heights in variables, as the statement
/// states them.
pub(crate) fn check_residual_rows(
    residual_num_vars: impl IntoIterator<Item = u8>,
) -> Result<(), Error> {
    let rows: u128 = residual_num_vars
        .into_iter()
        .map(|num_vars| 1u128 << num_vars.min(127))
        .sum();
    if rows > u128::from(crate::tables::register_residual::MAX_ROWS_TOTAL) {
        return Err(Error::InvalidTableCounts(format!(
            "the residual register table states {rows} rows — a block takes at most {}",
            crate::tables::register_residual::MAX_ROWS_TOTAL
        )));
    }
    Ok(())
}

/// The word the statement of a block under the fused memory format absorbs
/// right after the table counts, before its two fused counts: the format named
/// in the transcript, so no statement of the split format (which absorbs
/// neither) shares its bytes.
pub(crate) const FUSED_MEMORY_WORD: u64 = u64::from_le_bytes(*b"MEMFUSED");

/// The word the statement of a block under the word memory format absorbs
/// right after the table counts: the format named in the transcript, as
/// [`FUSED_MEMORY_WORD`] names the fused one.
pub(crate) const WORD_MEMORY_WORD: u64 = u64::from_le_bytes(*b"MEMWORDS");

/// The word memory format's table families, in [`VmAirs::air_refs`] order
/// and [`BlockWhirProof::word_memory`]'s
/// ([`crate::tables::word_cpu::WORD_TABLES`], defined where a build without
/// the block prover sees it too).
pub use crate::tables::word_cpu::WORD_TABLES;

/// What a block's statement absorbs right after the table counts under
/// `memory`: nothing under the split format (the statement as it always
/// was); under the fused format [`FUSED_MEMORY_WORD`] and the MEM_LOAD and
/// MEM_STORE tables' counts (`fused`, [`BlockWhirProof::fused_memory`]);
/// under the word format [`WORD_MEMORY_WORD`] and each word family's count
/// (`words`, [`BlockWhirProof::word_memory`]).
pub(crate) fn fused_statement_words(
    memory: crate::tables::fused_memory::MemoryFormat,
    fused: [u64; 2],
    words: [u64; 7],
) -> Vec<u64> {
    match memory {
        crate::tables::fused_memory::MemoryFormat::Split => Vec::new(),
        crate::tables::fused_memory::MemoryFormat::Fused => {
            vec![FUSED_MEMORY_WORD, fused[0], fused[1]]
        }
        crate::tables::fused_memory::MemoryFormat::Words => {
            std::iter::once(WORD_MEMORY_WORD).chain(words).collect()
        }
    }
}

/// The word tables a block states under `memory`: under the word format the
/// counts, each family's; under any other format none, and any other count is
/// refused (it is absorbed nowhere, so no value of it may ride in a proof).
pub(crate) fn word_counts_under(
    memory: crate::tables::fused_memory::MemoryFormat,
    stated: [u64; 7],
) -> Result<[usize; 7], Error> {
    #[cfg(test)]
    if memory != crate::tables::fused_memory::MemoryFormat::Words
        && WORD_COUNTS_UNCHECKED.with(std::cell::Cell::get)
    {
        return Ok([0; 7]);
    }
    match memory {
        crate::tables::fused_memory::MemoryFormat::Words => {
            let mut counts = [0usize; 7];
            for (count, &n) in counts.iter_mut().zip(&stated) {
                *count = usize::try_from(n).map_err(|_| {
                    Error::InvalidTableCounts(format!("{n} word tables do not fit a usize"))
                })?;
            }
            Ok(counts)
        }
        _ if stated == [0; 7] => Ok([0; 7]),
        other => Err(Error::InvalidTableCounts(format!(
            "the {other:?} memory format states no word table, not {stated:?}"
        ))),
    }
}

/// The fused memory format's MEM_LOAD and MEM_STORE tables a block states,
/// checked against the verifier's format ([`fused_tables_under`]).
pub fn fused_tables(format: &BlockFormat, stated: [u64; 2]) -> Result<[usize; 2], Error> {
    fused_tables_under(format.memory, stated)
}

/// The fused memory tables a block states under `memory`: under the split
/// format none, and any other pair is refused (it is absorbed nowhere, so no
/// value of it may ride in a proof); under the fused format the pair as
/// counts.
pub(crate) fn fused_tables_under(
    memory: crate::tables::fused_memory::MemoryFormat,
    stated: [u64; 2],
) -> Result<[usize; 2], Error> {
    #[cfg(test)]
    if memory == crate::tables::fused_memory::MemoryFormat::Split
        && FUSED_COUNTS_UNCHECKED.with(std::cell::Cell::get)
    {
        return Ok([0, 0]);
    }
    match memory {
        crate::tables::fused_memory::MemoryFormat::Split if stated == [0, 0] => Ok([0, 0]),
        crate::tables::fused_memory::MemoryFormat::Split => Err(Error::InvalidTableCounts(
            format!("the split memory format states no fused table, not {stated:?}"),
        )),
        crate::tables::fused_memory::MemoryFormat::Fused => {
            let count = |n: u64| {
                usize::try_from(n).map_err(|_| {
                    Error::InvalidTableCounts(format!("{n} fused tables do not fit a usize"))
                })
            };
            Ok([count(stated[0])?, count(stated[1])?])
        }
        // The word memory format retires the fused tables: none, as under the
        // split format.
        crate::tables::fused_memory::MemoryFormat::Words => {
            if stated == [0, 0] {
                Ok([0, 0])
            } else {
                Err(Error::InvalidTableCounts(format!(
                    "the word memory format states no fused table, not {stated:?}"
                )))
            }
        }
    }
}

/// The tables the word memory format retires, by name: their counts in a
/// statement under it.
fn retired_counts(counts: &TableCounts, fused: [u64; 2]) -> [(&'static str, u64); 6] {
    [
        ("LOAD", counts.load as u64),
        ("STORE", counts.store as u64),
        ("MEMW", counts.memw as u64),
        ("MEMW_A", counts.memw_aligned as u64),
        ("MEM_LOAD", fused[0]),
        ("MEM_STORE", fused[1]),
    ]
}

/// Under the word memory format, refuses a statement that states any table
/// the format retires: LOAD, STORE, MEMW, MEMW_A (`counts`), MEM_LOAD or
/// MEM_STORE (`fused`). Their byte tokens would be a second memory beside the
/// words, which nothing ties to it. Under any other format, nothing to check.
pub(crate) fn word_tables_under(
    memory: crate::tables::fused_memory::MemoryFormat,
    counts: &TableCounts,
    fused: [u64; 2],
) -> Result<(), Error> {
    if memory != crate::tables::fused_memory::MemoryFormat::Words {
        return Ok(());
    }
    #[cfg(test)]
    if RETIRED_COUNTS_UNCHECKED.with(std::cell::Cell::get) {
        return Ok(());
    }
    match retired_counts(counts, fused)
        .into_iter()
        .find(|&(_, n)| n != 0)
    {
        Some((table, n)) => Err(Error::InvalidTableCounts(format!(
            "the word memory format states no {table} table, not {n}"
        ))),
        None => Ok(()),
    }
}

/// The tables of the chips the word format takes no call of yet, by name:
/// the accelerators' counts in a statement but COPY's (COPY_W's under the
/// word format), KECCAK's (KECCAK_W's: keccak256 alone) and its round
/// chip's, ECREC's (ECREC_W's) and ECDAS's (no memory: ECREC's chains),
/// whose word messages MEMW_W takes. IS_WH holds the permute mode's
/// pointers, which KECCAK_W has not.
fn chip_counts(counts: &TableCounts, _words: [u64; 7]) -> [(&'static str, u64); 4] {
    // Every count named, so a table added to the counts is placed here before
    // the verifier builds: refused below, or one of the format's.
    let TableCounts {
        // The CPU's, the ALU's and MEMW_R's: no memory messages of their own.
        cpu: _,
        lt: _,
        mul: _,
        dvrm: _,
        shift: _,
        branch: _,
        memw_register: _,
        eq: _,
        bytewise: _,
        cpu32: _,
        // Retired under the word format (`retired_counts`).
        memw: _,
        memw_aligned: _,
        load: _,
        store: _,
        // The chips the word format takes.
        copy: _,
        keccak: _,
        keccak_rnd: _,
        ecrec: _,
        ecdas: _,
        // The chips it takes no call of yet.
        is_wh,
        blake3,
        ecsm,
        hint,
    } = counts;
    [
        ("IS_WH", *is_wh as u64),
        ("BLAKE3", *blake3 as u64),
        ("ECSM", *ecsm as u64),
        ("HINT", *hint as u64),
    ]
}

/// Under the word memory format, refuses a statement that states a table of
/// a chip the format takes no call of yet: IS_WH (KECCAK's permute mode),
/// BLAKE3, ECSM, HINT. Those chips still read and write
/// memory by byte messages, which nothing receives under the word format: a
/// block that calls one is not provable under it, and the verifier says so
/// by its counts instead of leaving it to the bus. COPY's tables (COPY_W's),
/// KECCAK's (KECCAK_W's) and its round chip's, ECREC's (ECREC_W's) and
/// ECDAS's, and MEMW_W's, which takes their word messages, are the format's.
/// Under any other format, nothing to check.
pub(crate) fn chip_tables_under(
    memory: crate::tables::fused_memory::MemoryFormat,
    counts: &TableCounts,
    words: [u64; 7],
) -> Result<(), Error> {
    if memory != crate::tables::fused_memory::MemoryFormat::Words {
        return Ok(());
    }
    #[cfg(test)]
    if CHIP_COUNTS_UNCHECKED.with(std::cell::Cell::get) {
        return Ok(());
    }
    match chip_counts(counts, words)
        .into_iter()
        .find(|&(_, n)| n != 0)
    {
        Some((table, n)) => Err(Error::InvalidTableCounts(format!(
            "the word memory format takes no {table} call yet: no {table} table, not {n}"
        ))),
        None => Ok(()),
    }
}

/// Every PAGE_W table at its one height, [`crate::tables::page_w::NUM_VARS`]
/// variables, by name: a taller one would mint the next page's words a second
/// time. `tables` pairs each table's name with its stated height.
pub(crate) fn check_page_w_heights<'a>(
    tables: impl IntoIterator<Item = (&'a str, u8)>,
) -> Result<(), Error> {
    match tables.into_iter().find(|(name, num_vars)| {
        name.starts_with(crate::tables::page_w::NAME_PREFIX)
            && usize::from(*num_vars) != crate::tables::page_w::NUM_VARS
    }) {
        Some((name, num_vars)) => Err(Error::InvalidTableCounts(format!(
            "{name} states 2^{num_vars} rows — a PAGE_W table is 2^{}",
            crate::tables::page_w::NUM_VARS
        ))),
        None => Ok(()),
    }
}

/// Every page table at its place, under its name and at its height, each
/// derived by the verifier from its page tables `pages`
/// ([`block_page_tables`]): the tables `first..first + pages.len()` of `tables`
/// (each table's name and stated height, in AIR order) are the page tables in
/// their order, each named [`PageTable::name`](crate::tables::page_w::PageTable::name)
/// and stated at exactly its
/// [`num_vars`](crate::tables::page_w::PageTable::num_vars). A page table
/// stated a row taller would mint the next page's first word a second time; a
/// block's name is its class, base and pages, so a statement can present no
/// block the verifier did not cut.
pub(crate) fn check_page_table_heights<'a>(
    pages: &[crate::tables::page_w::PageTable],
    first: usize,
    tables: impl IntoIterator<Item = (&'a str, u8)>,
) -> Result<(), Error> {
    let stated: Vec<(&str, u8)> = tables.into_iter().skip(first).take(pages.len()).collect();
    if stated.len() != pages.len() {
        return Err(Error::InvalidTableCounts(format!(
            "{} page tables from table {first}, the statement states {}",
            pages.len(),
            stated.len()
        )));
    }
    for (page, (name, num_vars)) in pages.iter().zip(stated) {
        if name != page.name() {
            return Err(Error::InvalidTableCounts(format!(
                "page table {} is {name}",
                page.name()
            )));
        }
        if usize::from(num_vars) != page.num_vars() {
            return Err(Error::InvalidTableCounts(format!(
                "{name} states 2^{num_vars} rows — its {} pages are 2^{}",
                page.pages(),
                page.num_vars()
            )));
        }
    }
    Ok(())
}

/// Every word table (a [`WORD_TABLES`] family's) at most
/// [`BLOCK_WORD_MAX_VARS`] variables, by name. `tables` pairs each table's name
/// with its stated height.
pub(crate) fn check_word_heights<'a>(
    tables: impl IntoIterator<Item = (&'a str, u8)>,
) -> Result<(), Error> {
    match tables.into_iter().find(|(name, num_vars)| {
        name.split_once('[')
            .is_some_and(|(family, _)| WORD_TABLES.contains(&family))
            && usize::from(*num_vars) > BLOCK_WORD_MAX_VARS
    }) {
        Some((name, num_vars)) => Err(Error::InvalidTableCounts(format!(
            "{name} states 2^{num_vars} rows — a block takes at most 2^{BLOCK_WORD_MAX_VARS}"
        ))),
        None => Ok(()),
    }
}

#[cfg(test)]
thread_local! {
    /// Test only: on this thread, the split format's verifier reads no fused
    /// count ([`fused_tables_under`]'s refusal taken out, a control mutant).
    pub(crate) static FUSED_COUNTS_UNCHECKED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Test only: on this thread, a split or fused verifier reads no word
    /// count ([`word_counts_under`]'s refusal taken out, a control mutant).
    pub(crate) static WORD_COUNTS_UNCHECKED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Test only: on this thread, the word format takes any chip's table
    /// ([`chip_tables_under`]'s refusal taken out, a control mutant).
    pub(crate) static CHIP_COUNTS_UNCHECKED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Test only: on this thread, the word format takes the tables it retires
    /// ([`word_tables_under`]'s refusal taken out, a control mutant).
    pub(crate) static RETIRED_COUNTS_UNCHECKED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Test only: on this thread, every page with initial data is settled by
    /// the prepared opening ([`prepared_tables`]' dense arm), on the prover's
    /// side and the verifier's alike, so a test reaches that arm with a
    /// program far below the density threshold.
    pub(crate) static DENSE_PAGES_FORCED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// The most rows a word table may state, in variables: the block's chunk of
/// every streamed family, as a fused table's.
pub const BLOCK_WORD_MAX_VARS: usize = BLOCK_FUSED_MAX_VARS;

/// The most rows a fused memory table may state, in variables: the block's
/// chunk of every streamed family (`MaxRowsConfig::uniform(1 << 21)`).
pub const BLOCK_FUSED_MAX_VARS: usize = 21;

/// The block statement: the monolithic multilinear one under
/// [`MULTILINEAR_BLOCK_TAG`] (`more_counts` after its table counts:
/// [`fused_statement_words`]), then the group partition — the count, then per
/// group its size and its tables' indices, every value a little-endian `u64`
/// (so the roots that follow stay on a field-element boundary).
#[allow(clippy::too_many_arguments)]
pub(crate) fn absorb_block(
    t: &mut impl IsTranscript<E>,
    tag: &[u8],
    elf_bytes: &[u8],
    public_output: &[u8],
    table_counts: &TableCounts,
    more_counts: &[u64],
    num_private_input_pages: usize,
    runtime_page_ranges: &[RuntimePageRange],
    table_num_vars: &[u8],
    config: &multilinear::whir_chain::ChainConfig,
    groups: &[Vec<u32>],
) {
    absorb_tagged(
        t,
        tag,
        "block",
        &statement::elf_digest(elf_bytes),
        public_output,
        table_counts,
        more_counts,
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
    let pieces = table.main_table.height / rows.max(1);
    first_pieces(table, rows, pieces)
}

/// [`split_rows`] for a family the block cuts at a fixed number of rows
/// (KECCAK_RND, KECCAK, ECSM, ECDAS), keeping as many pieces as `padding`
/// counts ([`CutPadding::tables`]) of the table's real rows: its rows up to the
/// last one whose `mu` (the family's multiplicity column, zero on a padding
/// row) is set. Under [`CutPadding::LastTable`] the pieces of padding rows
/// alone go, as the finish leaves them unbuilt; under [`CutPadding::WholeRun`]
/// every piece stays.
fn split_cut(
    table: TraceTable<F, E>,
    rows: usize,
    mu: usize,
    padding: CutPadding,
) -> Vec<TraceTable<F, E>> {
    let height = table.main_table.height;
    match padding {
        CutPadding::LastTable if height > rows => {
            let real = (0..height)
                .rev()
                .find(|&row| *table.main_table.get(row, mu) != FieldElement::zero())
                .map_or(0, |row| row + 1);
            let pieces = padding.tables(real, rows).clamp(1, height / rows);
            first_pieces(table, rows, pieces)
        }
        _ => split_rows(table, rows),
    }
}

/// The first `pieces` of [`split_rows`]'s tables; `table` itself when it is no
/// taller than `rows`.
fn first_pieces(table: TraceTable<F, E>, rows: usize, pieces: usize) -> Vec<TraceTable<F, E>> {
    let height = table.main_table.height;
    if height <= rows {
        return vec![table];
    }
    let step = table.step_size;
    let width = table.main_table.width;
    // Row-major in, row-major out: each piece is a run of whole rows, copied
    // as they lie (a transposition of KECCAK_RND's 1,480 columns is what this
    // used to cost).
    (0..pieces.min(height / rows))
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
    shape: (usize, usize),
    release_rows: bool,
) -> Result<CommittedTable<'a, F, E>, Error> {
    table_of_checked(air, trace, shape, release_rows, true)
}

/// [`table_of`], its preprocessed-column check `check`ed or not (a test's
/// [`Deviations::unchecked_preprocessed`]).
fn table_of_checked<'a>(
    air: &'a dyn stark::traits::AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
    trace: &mut TraceTable<F, E>,
    (width, num_vars): (usize, usize),
    release_rows: bool,
    check: bool,
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
        if check && columns.get(col) != Some(expected) {
            return Err(Error::Prover(format!(
                "{}: preprocessed column {col} is not what the program implies",
                air.name(),
            )));
        }
    }
    CommittedTable::from_layout(layout, |col| core::mem::take(&mut columns[col as usize]))
        .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))
}

/// The CPU tables' heights of a rest the finish built after a streamed walk. A
/// CPU chunk the walk streamed is still a placeholder here, with no rows; its
/// height is the streamed chunk's, `cpu_rows` (a streamed CPU chunk is always
/// full, `MaxRowsConfig::cpu` rows), the height [`StreamAirs`] lays it out with.
/// The finish puts the placeholders first; one after a built table is refused.
fn rest_cpu_heights(traces: &Traces, cpu_rows: usize) -> Result<Vec<usize>, Error> {
    let streamed = traces.cpus.iter().take_while(|t| t.num_rows() == 0).count();
    if traces.cpus[streamed..].iter().any(|t| t.num_rows() == 0) {
        return Err(Error::Prover(
            "a streamed CPU chunk's placeholder after a built CPU table".to_string(),
        ));
    }
    Ok(traces
        .cpus
        .iter()
        .enumerate()
        .map(|(k, t)| if k < streamed { cpu_rows } else { t.num_rows() })
        .collect())
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

/// [`table_of_narrow`] for a trace held as the records its rows come from (a
/// finish's KECCAK_RND table, [`WindowedTraceBuilder::card_fill_keccak_rnd`];
/// a streamed CPU or MEMW_R chunk,
/// [`WindowedTraceBuilder::card_fill_streamed`]): its records become the
/// table's, and the card fills its columns from them as its group uploads it.
/// A table whose AIR's preprocessed columns are not the leading columns its
/// records fill from the row index alone is refused
/// (`multilinear::records::TableRecords::row_index_columns`).
pub(crate) fn table_of_records<'a>(
    air: &'a dyn stark::traits::AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
    trace: &mut TraceTable<F, E>,
    (width, num_vars): (usize, usize),
) -> Result<CommittedTable<'a, F, E>, Error> {
    let layout = layout_of(air, width, num_vars)
        .map_err(|e| Error::Prover(format!("{}: {e:?}", air.name())))?;
    let records = trace
        .take_records_main()
        .ok_or_else(|| Error::Prover(format!("{}: no records", air.name())))?;
    trace.main_table = Table::new(Vec::new(), 0);
    if records.cols() != width || records.rows() != 1usize << num_vars {
        return Err(Error::Prover(format!(
            "{}: records of {} × {}, the table is {width} × 2^{num_vars}",
            air.name(),
            records.cols(),
            records.rows()
        )));
    }
    // The AIR's preprocessed columns must be the ones the fill writes from the
    // row index alone: the check of them reads the columns, which only the
    // fill makes, so a count the records do not declare is refused.
    let preprocessed = if air.is_preprocessed() {
        air.num_precomputed_columns()
    } else {
        0
    };
    if preprocessed != records.row_index_columns() {
        return Err(Error::Prover(format!(
            "{}: {preprocessed} preprocessed columns, its records fill {} from the row index",
            air.name(),
            records.row_index_columns(),
        )));
    }
    CommittedTable::from_records(layout, records)
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
    /// Under [`commit_group_narrow`], the columns packed narrow for the commit
    /// (kept for a later store of the same columns) and what the commit's
    /// narrow store cost; `None` otherwise, or where it packed nothing.
    pub(crate) packed: Option<NarrowCommit>,
}

/// A prepared stack committed from a store sent narrow ([`commit_group_narrow`]):
/// the columns' packed bytes, and whether the commit read them from the card
/// (`sent`; `false` when the plan or the card kept them on the host), with the
/// upload's seconds.
pub(crate) struct NarrowCommit {
    pub(crate) store: multilinear::gpu::PackedStore,
    pub(crate) words: u64,
    pub(crate) sent: bool,
    pub(crate) upload_secs: f64,
    /// The bytes were packed ahead, in the level's host part
    /// ([`GroupStack::pack_narrow`]), not on the commit's path.
    pub(crate) packed_ahead: bool,
}

/// What a narrow commit keeps free on the card beside its store and its own
/// promises.
const NARROW_COMMIT_MARGIN_BYTES: u64 = 1 << 30;

/// One group's prepared stack before its commitment: which tables, and their
/// prefixes' columns concatenated in the group's order.
pub(crate) struct GroupStack {
    pub(crate) group: usize,
    /// `(position in the proof's table order, AIR index, prefix length)`.
    pub(crate) tables: Vec<(usize, usize, usize)>,
    pub(crate) columns: Vec<Vec<FieldElement<F>>>,
    /// The columns packed narrow ahead of the commit (in the level's host
    /// part, beside the base's phase B), for [`commit_group_narrow`] to send
    /// without packing them on its path; `None` packs there.
    pub(crate) packed: Option<multilinear::gpu::PackedStore>,
}

impl GroupStack {
    /// [`Self::packed`] made now: the stack's columns packed narrow, on
    /// whatever pool the caller runs this on.
    pub(crate) fn pack_narrow(&mut self) {
        let raw: Vec<&[FieldElement<F>]> = self.columns.iter().map(Vec::as_slice).collect();
        self.packed = multilinear::gpu::pack_store_fields(&raw);
    }

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
            packed: None,
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

/// Where DECODE sits among a block's tables, found by NAME.
///
/// ⚠ Not a constant and not a position. The opening binds the pinned columns to
/// ONE table's reduced point, and a prover able to aim them at a different
/// table's point would be settling them against challenges they were never
/// bound to. So the index is asserted against the AIR that carries the columns,
/// and "exactly one" is part of the assertion: a table set with two DECODEs, or
/// none, is a layout nobody meant to build.
pub(crate) fn decode_table_index(
    airs: &[&dyn stark::traits::AIR<Field = F, FieldExtension = E, PublicInputs = ()>],
) -> Result<usize, Error> {
    let found: Vec<usize> = airs
        .iter()
        .enumerate()
        .filter(|(_, air)| air.name() == "DECODE")
        .map(|(i, _)| i)
        .collect();
    match found.as_slice() {
        [only] => Ok(*only),
        _ => Err(Error::Prover(format!(
            "a block's table set must carry exactly one DECODE, it carries {} (at {found:?})",
            found.len(),
        ))),
    }
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
    // The page tables, cut as the AIR set's were: one a page but under the
    // word memory format's page runs.
    let page_tables = block_page_tables(page_configs, format)?;
    if page_tables.len() != airs.pages.len() {
        return Err(Error::Prover(format!(
            "{} page tables for {} page AIRs",
            page_tables.len(),
            airs.pages.len()
        )));
    }
    let first_page = first_page_index(airs)?;
    let decode = decode_table_index(&refs)?;
    let mut tables = vec![(decode, refs[decode].precomputed_columns())];
    // Under the word memory format the page tables are PAGE_W's: a word a
    // row, its preprocessed columns `OFFSET` and the initial word's halves.
    let words = format.memory == crate::tables::fused_memory::MemoryFormat::Words;
    let num_vars = if words {
        crate::tables::page_w::NUM_VARS
    } else {
        crate::continuation::PAGE_NUM_VARS
    };
    // The rule reads the pages, never the tables: the page count it is
    // charged at and each page's candidacy are the same at any page run.
    let plan = crate::continuation::genesis_stack_plan(page_configs, 0, num_vars);
    #[cfg(test)]
    let forced = DENSE_PAGES_FORCED.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let forced = false;
    for route in plan
        .routes
        .iter()
        .filter(|route| route.dense || (forced && route.has_init && route.nonzero > 0))
    {
        // A dense page has a nonzero byte: a page of data, a table of its own
        // at any page run.
        let page = route.table;
        let table = page_tables
            .iter()
            .position(|t| t.first == page && t.kind == crate::tables::page_w::PageTableKind::Page)
            .ok_or_else(|| {
                Error::Prover(format!(
                    "the dense page {:#x} is not a page table of its own",
                    route.page_base
                ))
            })?;
        let config = &page_configs[page];
        let columns = if words {
            crate::tables::page_w::preprocessed_columns(config)
        } else {
            crate::tables::page::preprocessed_columns(config)
        };
        tables.push((first_page + table, columns));
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

/// The pages [`block_airs`] builds a PAGE AIR for under `memory`: under the
/// word memory format none, its page tables
/// ([`VmAirs::with_word_memory_at`]) built straight from the page list in
/// their place, so a stated page table costs one AIR at any page run; under any
/// other format every listed page.
pub(crate) fn page_air_pages(
    memory: crate::tables::fused_memory::MemoryFormat,
    page_configs: &[crate::tables::page::PageConfig],
) -> &[crate::tables::page::PageConfig] {
    if memory == crate::tables::fused_memory::MemoryFormat::Words {
        &[]
    } else {
        page_configs
    }
}

/// The page traces a build made, `pages` of them, against the AIR set's page
/// tables: one each, or the build cut its pages otherwise than the format.
pub(crate) fn check_page_traces(airs: &VmAirs, pages: usize) -> Result<(), Error> {
    if pages != airs.pages.len() {
        return Err(Error::Prover(format!(
            "the build made {pages} page tables, the format's AIR set holds {}",
            airs.pages.len()
        )));
    }
    Ok(())
}

/// Commits one group's prepared stack under `config`.
pub(crate) fn commit_group<H: multilinear::whir_hash::WhirHash>(
    stack: GroupStack,
    config: &multilinear::whir_chain::ChainConfig,
) -> Result<GroupPrepared<H>, Error> {
    commit_group_where::<H>(stack, config, false, false, false)
}

/// [`commit_group`] with the columns sent to the card narrow: packed on the
/// host once (in the level's host part when [`GroupStack::packed`] holds them,
/// else here; `multilinear::gpu::pack_store`), widened on the card into a
/// store, and committed from that store, where today the commit copies the
/// wide columns in; under `staged` each packed run goes up through the card's
/// pinned staging. The same words at the same places, so the same roots.
/// Where there is no card, or the plan (the store, its staging, a card
/// commit's price and a margin, against what the ledger has free) or the card
/// says no, it commits as [`commit_group`] does; the packed bytes are kept
/// either way ([`GroupPrepared::packed`]) for a later store of the columns.
pub(crate) fn commit_group_narrow<H: multilinear::whir_hash::WhirHash>(
    stack: GroupStack,
    config: &multilinear::whir_chain::ChainConfig,
    staged: bool,
) -> Result<GroupPrepared<H>, Error> {
    commit_group_where::<H>(stack, config, false, true, staged)
}

/// Whether a narrow commit's store fits: its words, the largest run's packed
/// bytes at most, a card commit's price and a margin, within what is free.
fn narrow_commit_fits(words: u64, staging: u64, commit: u64, free: u64) -> bool {
    words
        .saturating_add(staging)
        .saturating_add(commit)
        .saturating_add(NARROW_COMMIT_MARGIN_BYTES)
        <= free
}

/// [`commit_group`] on the host, whatever a card would do
/// (`StackedCommitment::commit_on_host`): the same roots. What a verifier
/// derives the prepared roots with, so no card kernel sits on its path.
pub(crate) fn commit_group_on_host<H: multilinear::whir_hash::WhirHash>(
    stack: GroupStack,
    config: &multilinear::whir_chain::ChainConfig,
) -> Result<GroupPrepared<H>, Error> {
    commit_group_where::<H>(stack, config, true, false, false)
}

fn commit_group_where<H: multilinear::whir_hash::WhirHash>(
    stack: GroupStack,
    config: &multilinear::whir_chain::ChainConfig,
    on_host: bool,
    narrow: bool,
    staged: bool,
) -> Result<GroupPrepared<H>, Error> {
    let shapes = stack.shapes();
    let group = stack.group;
    let prepacked = stack.packed;
    let columns: Vec<Mle<F>> = stack
        .columns
        .into_iter()
        .map(|values| Mle::new(values).map_err(|e| Error::Prover(format!("group {group}: {e:?}"))))
        .collect::<Result<_, _>>()?;
    let layout = stark::multilinear_table::global_layout(&shapes, config.format.stack)
        .map_err(|e| Error::Prover(format!("{e:?}")))?;
    let borrowed = multilinear::stacking::borrow(&columns);
    let budget = multilinear::gpu::reserve_budget();
    // Packed ahead (the level's host part) when it fits these columns; else
    // packed here.
    let mut packed = (narrow && !on_host && budget > 0 && !multilinear::gpu::columns_disabled())
        .then(|| match prepacked.filter(|p| p.fits(&borrowed)) {
            Some(p) => Some((p, true)),
            None => multilinear::gpu::pack_store(&borrowed).map(|p| (p, false)),
        })
        .flatten()
        .map(|(store, ahead)| NarrowCommit {
            store,
            words: borrowed.iter().map(|c| c.len() as u64 * 8).sum(),
            sent: false,
            upload_secs: 0.0,
            packed_ahead: ahead,
        });
    let store = match packed.as_mut() {
        Some(p) => {
            let lens: Vec<usize> = borrowed.iter().map(|c| c.len()).collect();
            let staging = multilinear::narrow::store_staging_bound(
                &lens,
                multilinear::narrow::STORE_RUN_CELLS,
            );
            let price = multilinear::stacked_eval::commit_card_bytes(&layout, config);
            let free = budget.saturating_sub(multilinear::gpu::ledger_reserved());
            if narrow_commit_fits(p.words, staging, price, free) {
                let t = Instant::now();
                let store = if staged {
                    multilinear::gpu::upload_packed_store_staged(&p.store)
                } else {
                    multilinear::gpu::upload_packed_store(&p.store)
                }
                .map(|(s, _)| s);
                p.sent = store.is_some();
                p.upload_secs = t.elapsed().as_secs_f64();
                store
            } else {
                None
            }
        }
        None => None,
    };
    let commitment = if on_host {
        multilinear::stacked_eval::StackedCommitment::<F, H>::commit_on_host(
            layout, &borrowed, config,
        )
    } else {
        multilinear::stacked_eval::StackedCommitment::<F, H>::commit(
            layout,
            &borrowed,
            store.as_ref().map(|s| (s, 0)),
            config,
        )
    }
    .map_err(|e| Error::Prover(format!("group {group}: {e:?}")))?;
    // The store goes when this returns, before a retired stack is let go.
    let roots = commitment.roots();
    Ok(GroupPrepared {
        group,
        tables: stack.tables.iter().map(|&(p, _, n)| (p, n)).collect(),
        columns,
        roots,
        commitment,
        packed,
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

/// [`commit_groups`] on the host ([`commit_group_on_host`]): the verifier's.
pub(crate) fn commit_groups_on_host<H: multilinear::whir_hash::WhirHash>(
    stacks: Vec<GroupStack>,
    config: &multilinear::whir_chain::ChainConfig,
) -> Result<Vec<GroupPrepared<H>>, Error> {
    stacks
        .into_iter()
        .map(|stack| commit_group_on_host::<H>(stack, config))
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

/// A prover's wrong LOCAL_BYTES tables ([`Deviations::local_bytes`]), each
/// proved honestly over: what one check alone must refuse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum LocalBytesFault {
    #[default]
    None,
    /// `MU_ARE_BYTES` one more at the byte pair `(1, 2)`: a lookup received
    /// that no table sent — a false lookup's mirror. Only the group's local
    /// balance sees it.
    OneMoreReceived,
    /// `X ⊕ Y` one more at the first row no XOR lookup reads (`MU_XOR` zero
    /// there): the table is not the verifier's, its lookups still balance.
    /// Only the fixed-column check sees it.
    WrongUnusedXor,
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
    /// Under local lookups: phase A's lane runs the local argues
    /// (`Some(true)`) or leaves them to phase B (`Some(false)`), instead of
    /// the knob ([`LOCAL_LANE_ENV`]). The proof is the same.
    pub local_lane: Option<bool>,
    /// Under local lookups: every LOCAL_BYTES table made wrong
    /// ([`LocalBytesFault`]) and proved honestly over.
    pub local_bytes: LocalBytesFault,
    /// Faults in phase B's overlap ([`multilinear_block::OverlapFaults`]).
    pub overlap_faults: multilinear_block::OverlapFaults,
    /// Between the phases, the first narrow table's width map made wrong, so
    /// phase B widens other words than were committed
    /// ([`BlockCommitted::fault_narrow_width_map`]).
    pub narrow_width_map: bool,
    /// At the walk's end, the hand-off asked for everything, whatever the
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
    /// Phase B takes the group of the most tables first (`Some(true)`) or
    /// keeps its own order (`Some(false)`), instead of the knob
    /// ([`phase_b_many_first`]).
    pub phase_b_many_first: Option<bool>,
    /// Live regeneration's faults ([`regen::LiveFaults`]): a regenerator that
    /// dies or deposits wrong columns, which phase B must refuse, never prove
    /// over.
    pub regen_faults: regen::LiveFaults,
    /// The rest's regenerator's faults (N3, [`regen::RestRun`]): as
    /// [`Self::regen_faults`], a table named by its place among the rest's
    /// dropped tables.
    pub rest_faults: regen::LiveFaults,
    /// Both regeneration windows fixed, in bytes (the streamed chunks', the
    /// rest's), instead of the knobs: a window of one byte admits only its
    /// frontier, so a phase-B order that takes a class out of its rank order
    /// cannot complete.
    pub regen_windows: Option<(u64, u64)>,
    /// The streamed groups phase B takes ahead of the rest's first
    /// ([`led_order`]), instead of the knob ([`regen::regen_lead`]).
    pub regen_lead: Option<usize>,
    /// A pause before each drop of a streamed chunk, so the streamed
    /// chunks' regenerator started early meets chunks whose fate phase A has
    /// not decided yet.
    pub streamed_drop_delay: Option<std::time::Duration>,
    /// The builder fails right after the finish (and after any early start):
    /// phase A ends in an error with the streamed chunks' regenerator
    /// running.
    pub abort_after_finish: bool,
    /// The rest handed on with BITWISE last, beside phase 4's count from the
    /// ops ([`WindowedTraceBuilder::rest_bitwise_last`]): what BITWISE's
    /// lookups from the tables are compared with.
    pub rest_bitwise_last: bool,
    /// BITWISE's lookups from LT's, BRANCH's and BYTEWISE's tables, BITWISE
    /// last (`Some(true)`), or from their ops in phase 4, BITWISE first
    /// (`Some(false)`), instead of the default (`None`: from the tables;
    /// [`bitwise_from_tables_or_default`]).
    pub bitwise_from_tables: Option<bool>,
    /// KECCAK_RND's tables handed on first (`Some(true)`) or the rest in AIR
    /// order (`Some(false)`), instead of the default (`None`: first;
    /// [`keccak_rnd_first_or_default`]).
    pub keccak_rnd_first: Option<bool>,
    /// The page tables handed on right after KECCAK_RND's (`Some(true)`) or
    /// in AIR order (`Some(false)`), instead of the knob ([`pages_first`]).
    pub pages_first: Option<bool>,
    /// The run's AIR set built on a thread of its own while the first
    /// KECCAK_RND tables are laid out (`Some(true)`), or before the rest's
    /// first table (`Some(false)`), instead of the knob
    /// ([`rest_airs_beside`]).
    pub rest_airs_beside: Option<bool>,
    /// The packer closes its open group ahead of BITWISE when BITWISE would
    /// not fit it (`Some(true)`), or at BITWISE's arrival (`Some(false)`),
    /// instead of the knob ([`close_before_bitwise`]).
    pub close_before_bitwise: Option<bool>,
    /// KECCAK_RND's tables handed out during the windows (`Some(true)`) or
    /// built at the finish (`Some(false)`), instead of the knob
    /// ([`stream_keccak_rnd_records`]).
    pub stream_keccak_rnd: Option<bool>,
    /// KECCAK_RND's streamed tables in groups of their own (`Some(true)`) or
    /// among the other streamed chunks (`Some(false)`), instead of the knob
    /// ([`own_keccak_rnd_groups`]).
    pub keccak_rnd_own_groups: Option<bool>,
    /// With the AIR set built beside them, each early KECCAK_RND table laid
    /// out with the AIR named for the next table: an AIR the run's set does
    /// not have at that place, which the prove must refuse.
    #[cfg(test)]
    pub rest_airs_beside_misnamed: bool,
    /// The hand-out's counting on a thread of its own (`Some(true)`) or on
    /// the accumulator's (`Some(false)`), instead of the default (`None`: a
    /// thread of its own; [`WindowedTraceBuilder::count_handout_on_thread`]).
    pub handout_count_thread: Option<bool>,
    /// The host's memory target, in bytes, instead of the machine's
    /// ([`memory_target_bytes`]): 0 makes the `auto` rule want every table out.
    pub memory_target: Option<u64>,
    /// On the whole-run path, the traces edited after their cuts and before
    /// the proof, as a prover of other tables would.
    pub tamper_traces: Option<fn(&mut Traces)>,
    /// On the whole-run path, the tables committed without the prover's own
    /// check of their preprocessed columns against the program's, so a table
    /// whose program columns were edited reaches the verifier.
    pub unchecked_preprocessed: bool,
    /// Whether the card fills KECCAK_RND from its records, instead of the
    /// knob ([`block_card_fill`]).
    pub card_fill: Option<bool>,
    /// With the card fill, each KECCAK_RND table's records start this many
    /// rows late ([`WindowedTraceBuilder::card_fill_records_fault`]): one
    /// wrong table, the same at both uploads and to any host reader, so the
    /// prover's own checks cannot tell it from the committed one.
    #[cfg(test)]
    pub card_fill_records_fault: usize,
    /// With the streamed CPU chunks held as records, each log's first source
    /// and written values XORed with this, and with row families' chunks held as their rows,
    /// each row's timestamp this much later (a SHIFT op's input)
    /// ([`WindowedTraceBuilder::card_fill_streamed_fault`]): one wrong table,
    /// the same at both uploads and to any host reader.
    #[cfg(test)]
    pub card_fill_streamed_fault: u64,
    /// Whether the card fills the streamed CPU and MEMW_R chunks from their
    /// records (`[cpu, memw_r]`), instead of the knob.
    pub card_fill_streamed: Option<[bool; 2]>,
    /// Which row families' streamed chunks the card fills from their rows,
    /// instead of the knob; with both of the above set and this unset, none.
    pub card_fill_rows: Option<RowFamilies>,
    /// The room rule the streamed chunks held as records are admitted by,
    /// instead of the host's resident set against its pressure line.
    pub records_room: Option<crate::tables::trace_builder::RecordsRoom>,
    /// Whether streamed chunks held as records stay on the host instead of
    /// parking and dropping (`Some(true)`, the A/B control; `None` parks).
    pub records_keep: Option<bool>,
    /// Whether the word memory format's word families stream in chunks as
    /// the walk fills them, instead of the knob (`LAMBDA_VM_WORD_STREAM`).
    pub word_stream: Option<bool>,
    /// Whether phase A's head is stamped, instead of the knob
    /// ([`head_log::knob`]).
    pub head_log: Option<bool>,
    /// The trace alone: phase A drains each closed group as it comes instead
    /// of committing it, and the prove stops after the trace with
    /// [`HOST_ONLY`] ([`trace_block_host_only`]).
    pub host_only: bool,
}

/// What a host-only build ([`Deviations::host_only`]) drained: its groups,
/// tables and cells, and when the last group was drained (seconds since the
/// streamed prove started).
struct Drained {
    groups: usize,
    tables: usize,
    cells: u64,
    at: f64,
}

/// The error a host-only build stops with, once its trace is built.
pub(crate) const HOST_ONLY: &str = "host only";

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
        None,
    )
    .and_then(todays)
}

/// A proof of today's format, from the prove chain's: one under local
/// lookups is [`prove_block_whir_local`]'s, never handed out as today's.
fn todays(
    (proof, stamps): (BlockWhirProof<LocalArgues>, BlockStamps),
) -> Result<(BlockWhirProof, BlockStamps), Error> {
    if !proof.local.is_empty() {
        return Err(Error::Prover(
            "a block under local lookups is proved by prove_block_whir_local".to_string(),
        ));
    }
    Ok((proof.with_local(()), stamps))
}

/// [`prove_block_whir`] under a format that argues local lookups per group
/// ([`BlockFormat::local_lookups`]): the proof carries every group's local
/// argue. Refuses any other format.
pub fn prove_block_whir_local(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
) -> Result<(BlockWhirProof<LocalArgues>, BlockStamps), Error> {
    prove_block_whir_local_with(
        elf_bytes,
        private_inputs,
        proof_options,
        format,
        options,
        &Deviations::default(),
    )
}

/// [`prove_block_whir_local`] with a test's deviations.
pub(crate) fn prove_block_whir_local_with(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    deviations: &Deviations,
) -> Result<(BlockWhirProof<LocalArgues>, BlockStamps), Error> {
    if !format.local_lookups() {
        return Err(Error::Prover(
            "prove_block_whir_local proves a format with local lookups".to_string(),
        ));
    }
    prove_block_whir_inner(
        elf_bytes,
        private_inputs,
        proof_options,
        format,
        options,
        deviations,
        &|_, _| {},
        &|_| {},
        None,
    )
}

/// [`prove_block_whir_observed_led`] under a format that argues local lookups
/// per group ([`BlockFormat::local_lookups`]): the proof carries every
/// group's local argue. Refuses any other format.
#[allow(clippy::too_many_arguments)]
pub fn prove_block_whir_local_observed_led(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    on_statement: StatementObserver<'_>,
    on_group: GroupObserver<'_>,
    lead: Option<&PhaseBLead>,
) -> Result<(BlockWhirProof<LocalArgues>, BlockStamps), Error> {
    if !format.local_lookups() {
        return Err(Error::Prover(
            "prove_block_whir_local proves a format with local lookups".to_string(),
        ));
    }
    prove_block_whir_inner(
        elf_bytes,
        private_inputs,
        proof_options,
        format,
        options,
        &Deviations::default(),
        on_statement,
        on_group,
        lead,
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
    prove_block_whir_observed_led(
        elf_bytes,
        private_inputs,
        proof_options,
        format,
        options,
        on_statement,
        on_group,
        None,
    )
}

/// [`prove_block_whir_observed_groups`] with phase B led by `lead`: once the
/// caller offers the recursion's leaves there ([`PhaseBLead::offer`]), phase B
/// takes the groups of the leaf it can finish soonest next. The proof is the
/// same in any order.
#[allow(clippy::too_many_arguments)]
pub fn prove_block_whir_observed_led(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    on_statement: StatementObserver<'_>,
    on_group: GroupObserver<'_>,
    lead: Option<&PhaseBLead>,
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
        lead,
    )
    .and_then(todays)
}

/// The recursion's leaves, offered to phase B while it runs (the early leaf first),
/// and what phase B did with them. The tree's planner offers each admitted
/// leaf's groups as soon as it has derived the plan; at its next turn phase B
/// moves the groups of the leaf it can finish soonest ([`lead_order`]) ahead of
/// the others still to come, once, so that leaf's proof can start its execute
/// long before the base ends. Prover-only: every order proves the same bytes.
#[derive(Default)]
pub struct PhaseBLead {
    leaves: std::sync::OnceLock<Vec<Vec<usize>>>,
    taken: std::sync::Mutex<Option<LeadTaken>>,
    /// How many leaves lead, one after another ([`PhaseBLead::leading`]);
    /// 0 reads as 1, today's single lead.
    lead_leaves: usize,
}

/// What phase B did with the offered leaves ([`PhaseBLead`]).
#[derive(Clone, Debug, PartialEq)]
pub struct LeadTaken {
    /// Phase B's turn (0-based) at which it read the offer.
    pub turn: usize,
    /// Seconds into phase B at that turn.
    pub after: f64,
    /// The leaf whose groups went first, or `None`: no leaf could (each held
    /// a group rebuilt in phase B) or its groups were all taken already.
    pub leaf: Option<usize>,
    /// The groups moved ahead of the others, in their new order.
    pub moved: Vec<usize>,
    /// Every leaf that led, in the order their groups go (`leaf` is the
    /// first); one leaf unless the lead was made for more
    /// ([`PhaseBLead::leading`]).
    pub leaves: Vec<usize>,
    /// The groups still to come that phase B rebuilds, with their class, as
    /// the lead found them.
    pub rebuilt: Vec<(usize, usize)>,
    /// The rule that let each leaf in `leaves` lead.
    pub rules: Vec<LeadRule>,
    /// Per rebuilt class, whether it fit its window ([`class_fits`]).
    pub class_fits: Vec<bool>,
    /// Phase B's order as it was taken.
    pub order: Vec<usize>,
}

impl PhaseBLead {
    /// A lead for up to `n` leaves (`W3_EARLY_LEAVES`): at its turn phase B
    /// takes the groups of the leaf it can finish soonest, then of the next
    /// soonest among the others, and so on for `n` leaves ([`lead_order_n`]),
    /// so each of them completes as early as the ones before it allow. `n` 1
    /// is [`PhaseBLead::default`]'s single lead.
    pub fn leading(n: usize) -> Self {
        Self {
            lead_leaves: n,
            ..Self::default()
        }
    }

    /// Offers the leaves' groups (leaf `k`'s at `leaves[k]`); only the first
    /// offer counts. Returns whether this one did.
    pub fn offer(&self, leaves: Vec<Vec<usize>>) -> bool {
        self.leaves.set(leaves).is_ok()
    }

    /// What phase B did with the offer, once it read it.
    pub fn taken(&self) -> Option<LeadTaken> {
        self.taken.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Phase B's re-ranking at turn `p` of `order` ([`multilinear_block::PhaseBReorder`]):
    /// nothing until the leaves are offered, then once [`lead_order`] over the
    /// groups still to come, recorded.
    fn turn(
        &self,
        p: usize,
        order: &[usize],
        classes: &[Option<usize>],
        cells: &[u64],
        after: f64,
        class_fits: &[bool],
    ) -> Option<Vec<usize>> {
        let leaves = self.leaves.get()?;
        let mut taken = self.taken.lock().unwrap_or_else(|e| e.into_inner());
        if taken.is_some() {
            return None;
        }
        let rest = &order[p + 1..];
        let led = lead_order_n_rules(
            leaves,
            rest,
            classes,
            cells,
            self.lead_leaves.max(1),
            class_fits,
        );
        let (led_leaves, rules, moved, now) = match &led {
            Some((led_leaves, rules, moved, now)) => (
                led_leaves.clone(),
                rules.clone(),
                moved.clone(),
                now.clone(),
            ),
            None => (Vec::new(), Vec::new(), rest.to_vec(), rest.to_vec()),
        };
        *taken = Some(LeadTaken {
            turn: p,
            after,
            leaf: led_leaves.first().copied(),
            leaves: led_leaves,
            rules,
            class_fits: class_fits.to_vec(),
            rebuilt: rest
                .iter()
                .filter_map(|&g| classes.get(g).copied().flatten().map(|c| (g, c)))
                .collect(),
            moved,
            order: order[..=p]
                .iter()
                .copied()
                .chain(now.iter().copied())
                .collect(),
        });
        led.map(|(_, _, _, now)| now).filter(|now| now != rest)
    }
}

/// Phase B's groups still to come (`rest`, in its order) led by one leaf's:
/// among `leaves` (each a list of groups), those none of whose groups in
/// `rest` is rebuilt in phase B (`classes[g]` is `Some`: a class keeps its
/// own order, and its regenerator's pace), the one with the fewest cells still
/// to take (ties to the lower leaf) and with at least one group to take. Its
/// groups in `rest` go first, in their order; the others keep theirs. Returns
/// the leaf, the groups moved and the new `rest`, or `None` when no leaf can
/// lead. The result is a permutation of `rest`.
#[cfg(test)]
pub(crate) fn lead_order(
    leaves: &[Vec<usize>],
    rest: &[usize],
    classes: &[Option<usize>],
    cells: &[u64],
) -> Option<(usize, Vec<usize>, Vec<usize>)> {
    lead_order_with(leaves, rest, classes, cells, false)
}

/// [`lead_order`], and with `class_next` a leaf with rebuilt groups to come may
/// lead too, when they are, class by class, the next ones `rest` takes of their
/// class: every group of that class in `rest` with a lower index is the leaf's
/// own. Its groups then go first and each class still goes in group order (the
/// regenerators deposit in that order). Without `class_next` it is
/// [`lead_order`]'s rule: no rebuilt group to come at all.
#[cfg(test)]
pub(crate) fn lead_order_with(
    leaves: &[Vec<usize>],
    rest: &[usize],
    classes: &[Option<usize>],
    cells: &[u64],
    class_next: bool,
) -> Option<(usize, Vec<usize>, Vec<usize>)> {
    lead_pick(leaves, rest, classes, cells, class_next, &[])
        .map(|(k, _, moved, now)| (k, moved, now))
}

/// Which rule let a leaf lead ([`lead_pick`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeadRule {
    /// No rebuilt group of its was still to come.
    Plain,
    /// Its rebuilt groups were, class by class, the next ones of their class.
    ClassNext,
    /// A rebuilt group of a class whose dropped bytes all fit the class's
    /// regeneration window ([`class_fits`]) went out of group order.
    Window,
}

impl LeadRule {
    pub fn word(self) -> &'static str {
        match self {
            LeadRule::Plain => "plain",
            LeadRule::ClassNext => "class-next",
            LeadRule::Window => "window",
        }
    }
}

/// Per rebuilt class, whether every byte phase B rebuilds of it fits the
/// class's regeneration window at its smallest in phase B (`floors[c]`: the
/// adaptive window's floor, or the fixed size). Then every deposit of the
/// class is admitted as soon as it is made, whatever the order its groups are
/// taken in, so no taker can wait on a deposit the window holds back
/// (`stark::regen`'s pacing from the frontier). `dropped[c]` is the class's
/// (tables, bytes); a class with no floor never fits.
pub(crate) fn class_fits(dropped: &[(usize, u64)], floors: &[u64]) -> Vec<bool> {
    dropped
        .iter()
        .enumerate()
        .map(|(c, &(_, bytes))| floors.get(c).is_some_and(|&floor| bytes <= floor))
        .collect()
}

/// [`lead_order_with`] that also says which rule let the leaf lead, and with
/// `class_fits` (per class, [`class_fits`]): leading several (`class_next`), a
/// rebuilt group of a class that fits its window may go out of group order.
fn lead_pick(
    leaves: &[Vec<usize>],
    rest: &[usize],
    classes: &[Option<usize>],
    cells: &[u64],
    class_next: bool,
    class_fits: &[bool],
) -> Option<(usize, LeadRule, Vec<usize>, Vec<usize>)> {
    let to_come: std::collections::BTreeSet<usize> = rest.iter().copied().collect();
    let class = |g: usize| classes.get(g).copied().flatten();
    let cell = |g: usize| cells.get(g).copied().unwrap_or(0);
    let (leaf, rule) = leaves
        .iter()
        .enumerate()
        .filter_map(|(k, groups)| {
            let left: Vec<usize> = groups
                .iter()
                .copied()
                .filter(|g| to_come.contains(g))
                .collect();
            let rebuilt = left.iter().any(|&g| class(g).is_some());
            // With `class_next`: each of the leaf's rebuilt groups to come has
            // only the leaf's own groups of its class below it in `rest`, or
            // (the window rule) belongs to a class that fits its window.
            let next_of = |g: usize, c: usize| {
                rest.iter()
                    .all(|&h| h >= g || class(h) != Some(c) || groups.contains(&h))
            };
            let fits = |c: usize| class_fits.get(c).copied().unwrap_or(false);
            let rule = if !rebuilt {
                Some(LeadRule::Plain)
            } else if !class_next {
                None
            } else if left.iter().all(|&g| class(g).is_none_or(|c| next_of(g, c))) {
                Some(LeadRule::ClassNext)
            } else if left
                .iter()
                .all(|&g| class(g).is_none_or(|c| next_of(g, c) || fits(c)))
            {
                Some(LeadRule::Window)
            } else {
                None
            };
            rule.filter(|_| !left.is_empty())
                .map(|rule| (k, rule, left.iter().map(|&g| cell(g)).sum::<u64>()))
        })
        .min_by_key(|&(k, _, cost)| (cost, k))
        .map(|(k, rule, _)| (k, rule))?;
    let mine: std::collections::BTreeSet<usize> = leaves[leaf].iter().copied().collect();
    let moved: Vec<usize> = rest.iter().copied().filter(|g| mine.contains(g)).collect();
    let now: Vec<usize> = moved
        .iter()
        .copied()
        .chain(rest.iter().copied().filter(|g| !mine.contains(g)))
        .collect();
    Some((leaf, rule, moved, now))
}

/// [`lead_order`] for up to `n` leaves, one after another: the leaf that
/// leads `rest`, then the one that leads what `rest` holds after its groups,
/// and so on, each by the same rule (no rebuilt group to come, fewest cells
/// left, ties to the lower leaf). Returns the leaves in that order, every group
/// moved (theirs, leaf by leaf, each leaf's in `rest`'s order) and the new
/// `rest`, a permutation of it; `None` when no leaf can lead. With `n` 1 it is
/// [`lead_order`].
#[cfg(test)]
pub(crate) fn lead_order_n(
    leaves: &[Vec<usize>],
    rest: &[usize],
    classes: &[Option<usize>],
    cells: &[u64],
    n: usize,
) -> Option<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    lead_order_n_rules(leaves, rest, classes, cells, n, &[])
        .map(|(led, _, moved, now)| (led, moved, now))
}

/// A multi-leaf lead: the leaves in order, the rule each led by, every group
/// moved, and phase B's new order of the groups still to come.
pub(crate) type LeadOrder = (Vec<usize>, Vec<LeadRule>, Vec<usize>, Vec<usize>);

/// [`lead_order_n`] with each leaf's rule, and `class_fits` ([`class_fits`]):
/// leading several, a class that fits its window may go out of group order.
pub(crate) fn lead_order_n_rules(
    leaves: &[Vec<usize>],
    rest: &[usize],
    classes: &[Option<usize>],
    cells: &[u64],
    n: usize,
    class_fits: &[bool],
) -> Option<LeadOrder> {
    let mut led = Vec::new();
    let mut rules = Vec::new();
    let mut moved = Vec::new();
    let mut left = rest.to_vec();
    while led.len() < n {
        // A leaf already led has no group left in `left`, so it cannot lead again.
        // Leading several, a leaf whose rebuilt groups are next in their class may
        // lead too ([`lead_order_with`]); one is today's rule exactly.
        let Some((leaf, rule, mine, _)) =
            lead_pick(leaves, &left, classes, cells, n > 1, class_fits)
        else {
            break;
        };
        left.retain(|g| !mine.contains(g));
        moved.extend(mine);
        led.push(leaf);
        rules.push(rule);
    }
    if led.is_empty() {
        return None;
    }
    let now = moved.iter().copied().chain(left).collect();
    Some((led, rules, moved, now))
}

/// The block's trace as the streamed prove builds it, each closed group
/// drained instead of committed ([`Deviations::host_only`]): a box harness's
/// measure of the host's own pace, with no card to wait for. Returns the
/// stamps and the line naming what was drained.
#[cfg(test)]
pub(crate) fn trace_block_host_only(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
) -> Result<(BlockStamps, String), Error> {
    let window_log2 = options
        .window_log2
        .ok_or_else(|| Error::Prover("a host-only build is a streamed one".into()))?;
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let mut stamps = BlockStamps::default();
    let deviations = Deviations {
        host_only: true,
        ..Default::default()
    };
    match prove_streamed(
        &program,
        elf_bytes,
        private_inputs,
        1usize << window_log2,
        proof_options,
        format,
        options,
        &deviations,
        &|_, _| {},
        &|_| {},
        None,
        &mut stamps,
    ) {
        Err(Error::Prover(line)) if line.starts_with(HOST_ONLY) => Ok((stamps, line)),
        Err(e) => Err(e),
        Ok(_) => Err(Error::Prover(
            "a host-only build committed its groups".into(),
        )),
    }
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
        None,
    )
    .and_then(todays)
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
    lead: Option<&PhaseBLead>,
) -> Result<(BlockWhirProof<LocalArgues>, BlockStamps), Error> {
    // The page-locked pool (`LAMBDA_VM_PINNED_POOL_GIB`), locked once in the
    // process: a caller that locked it before its own clock (the block tree)
    // finds it locked; any other has it locked here, before the build.
    if let Some(pool) = multilinear::pinned::lock_pool() {
        eprintln!(
            "BLOCK PINNED POOL LOCKED: {} of {} bytes in {:.3}s at the prove's entry",
            pool.locked, pool.bytes, pool.seconds
        );
    }
    // `LAMBDA_VM_CARD_ROWS` (the card-built rows): the card keeps no logs from
    // another prove, keeps this one's only until it ends, and the line below
    // counts this one's uploads alone.
    multilinear::gpu::card_rows_begin();
    let proved = prove_block_whir_body(
        elf_bytes,
        private_inputs,
        proof_options,
        format,
        options,
        deviations,
        on_statement,
        on_group,
        lead,
    );
    if let Some(line) = multilinear::gpu::card_rows_report() {
        eprintln!("{line}");
    }
    if let Some(peak) = multilinear::gpu::card_reserved_peak() {
        eprintln!("BLOCK CARD LEDGER PEAK: {peak} B promised at most in this process");
    }
    multilinear::gpu::card_rows_end();
    // The base is proved: the pool's free bytes are no longer room for its
    // allocations, and later readings (the recursion's) count them again.
    multilinear::pinned::retire_netting();
    // What the pool served this process so far, whichever path the prove took
    // (only the streamed one stamps it for the report).
    if let Some(p) = multilinear::pinned::stats() {
        eprintln!(
            "BLOCK PINNED POOL USED: {} of {} upload bytes from the pool · {} bytes handed out · {} pageable once spent · held now {}",
            p.upload_pinned,
            p.upload_bytes,
            p.carved_bytes,
            p.spilled_bytes,
            p.held_bytes
                .map_or("unread".to_string(), |b| format!("{b} bytes"))
        );
    }
    // The same tally by where the bytes come from (process wide so far).
    let sources = multilinear::pinned::upload_sources();
    if sources.iter().any(|&(_, bytes, _)| bytes > 0) {
        let parts: Vec<String> = sources
            .iter()
            .filter(|&&(_, bytes, _)| bytes > 0)
            .map(|&(name, bytes, pinned)| format!("{name} {pinned} of {bytes}"))
            .collect();
        eprintln!(
            "BLOCK UPLOAD SOURCES: bytes from the pool of the bytes uploaded, by source · {}",
            parts.join(" · ")
        );
    }
    proved
}

/// Refuses a block format no prove makes a proof under: the word memory
/// format under any register format but the in-row one (the 32-bit token
/// format sends some register accesses to MEMW and MEMW_A, which the word
/// format retires). The prover's alone: a verifier reads any statement under
/// the format and refuses, by its counts, the tables it retires.
pub(crate) fn refuse_unprovable(format: &BlockFormat) -> Result<(), Error> {
    match (format.memory, format.registers) {
        (
            crate::tables::fused_memory::MemoryFormat::Words,
            crate::tables::register::RegisterFormat::Tokens32,
        ) => Err(Error::Prover(
            "the word memory format needs the in-row register format".to_string(),
        )),
        _ => Ok(()),
    }
}

/// Refuses a memory format the block cannot hold: the fused tables check an
/// access's step by MEMW_A's limbs, so
/// [`MemoryFormat::Fused`](crate::tables::fused_memory::MemoryFormat::Fused)
/// needs MEMW_A's limbs' form
/// ([`MemwAlignedOrder::Limbs`](crate::tables::memw_aligned::MemwAlignedOrder::Limbs));
/// and page runs ([`BlockFormat::page_run_log2`]) past their cap. Its prover
/// and its verifier refuse alike.
pub(crate) fn check_memory_format(format: &BlockFormat) -> Result<(), Error> {
    use crate::tables::page_w::MAX_PAGE_RUN_LOG2;
    if format.page_run_log2 > MAX_PAGE_RUN_LOG2 {
        return Err(Error::Prover(format!(
            "page runs of 2^{} pages: at most 2^{MAX_PAGE_RUN_LOG2}",
            format.page_run_log2
        )));
    }
    match (format.memory, format.memw_aligned) {
        (
            crate::tables::fused_memory::MemoryFormat::Fused,
            crate::tables::memw_aligned::MemwAlignedOrder::Lt,
        ) => Err(Error::Prover(
            "the fused memory format needs MEMW_A's limbs' form".to_string(),
        )),
        _ => Ok(()),
    }
}

/// The most pages a block statement can list under `format` when it can hold
/// `page_tables` page tables (its tables less every table its counts imply
/// besides the pages): a page table holds at most `2^page_run_log2` pages (one
/// with page runs off), so a page list longer than this can never be cut into
/// the statement's tables. It bounds the statement's runtime ranges before the
/// page list is built ([`Traces::page_configs_from_elf_and_runtime`]); the
/// exact table count is held once the list is cut ([`block_frame`]). A run
/// past its cap is refused before ([`check_memory_format`]); the shift is
/// clamped all the same.
pub(crate) fn max_block_pages(page_tables: usize, format: &BlockFormat) -> usize {
    let run = u32::from(
        format
            .page_runs()
            .min(crate::tables::page_w::MAX_PAGE_RUN_LOG2),
    );
    page_tables.saturating_mul(1usize << run)
}

/// The block's page tables under `format`
/// ([`page_tables`](crate::tables::page_w::page_tables) of `page_configs` at
/// the format's page runs, [`BlockFormat::page_runs`]: the word memory
/// format's alone, one table a page under any other). The prover's AIR set and
/// the verifier's both cut their pages here.
pub(crate) fn block_page_tables(
    page_configs: &[crate::tables::page::PageConfig],
    format: &BlockFormat,
) -> Result<Vec<crate::tables::page_w::PageTable>, Error> {
    check_memory_format(format)?;
    crate::tables::page_w::page_tables(page_configs, format.page_runs()).map_err(Error::Prover)
}

/// [`prove_block_whir_inner`] once the pool is locked.
#[allow(clippy::too_many_arguments)]
fn prove_block_whir_body(
    elf_bytes: &[u8],
    private_inputs: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    options: &BlockOptions,
    deviations: &Deviations,
    on_statement: StatementObserver<'_>,
    on_group: GroupObserver<'_>,
    lead: Option<&PhaseBLead>,
) -> Result<(BlockWhirProof<LocalArgues>, BlockStamps), Error> {
    let mut stamps = BlockStamps::default();
    check_memory_format(format)?;
    refuse_unprovable(format)?;
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
            lead,
            &mut stamps,
        )?;
        return Ok((proof, stamps));
    }
    // The whole-run build counts BITWISE for KECCAK_RND's Full form alone.
    if format.keccak_rnd != crate::tables::keccak_rnd::KeccakRndForm::Full {
        return Err(Error::Prover(format!(
            "KECCAK_RND's {:?} byte checks need the windowed build (`window_log2`)",
            format.keccak_rnd
        )));
    }
    // And lays MEMW_A out in the per-table STARK's form alone.
    if format.memw_aligned != crate::tables::memw_aligned::MemwAlignedOrder::Lt {
        return Err(Error::Prover(format!(
            "MEMW_A's {:?} timestamp order needs the windowed build (`window_log2`)",
            format.memw_aligned
        )));
    }
    // And ECDAS in today's form alone.
    if format.ecdas != crate::tables::ecdas::EcdasForm::Bytes {
        return Err(Error::Prover(format!(
            "ECDAS's {:?} relations need the windowed build (`window_log2`)",
            format.ecdas
        )));
    }
    // And the registers in today's format alone.
    if format.registers != crate::tables::register::RegisterFormat::Tokens32 {
        return Err(Error::Prover(format!(
            "the {:?} register format needs the windowed build (`window_log2`)",
            format.registers
        )));
    }
    // And the memory in today's format alone.
    if format.memory != crate::tables::fused_memory::MemoryFormat::Split {
        return Err(Error::Prover(format!(
            "the {:?} memory format needs the windowed build (`window_log2`)",
            format.memory
        )));
    }

    let t = Instant::now();
    let result = Executor::new(&program, private_inputs.to_vec())
        .map_err(|e| Error::Execution(format!("{e}")))?
        .run()
        .map_err(|e| Error::Execution(format!("{e}")))?;
    stamps.execute = t.elapsed().as_secs_f64();

    crate::tables::trace_builder::build_stamps::start();
    let t = Instant::now();
    let mut traces =
        Traces::from_elf_and_logs(&program, &result.logs, &options.max_rows, private_inputs)?;
    drop(result);
    stamps.build = t.elapsed().as_secs_f64();
    stamps.build_marks = crate::tables::trace_builder::build_stamps::take();

    let t = Instant::now();
    let padding = CutPadding::LastTable;
    split_keccak_rnd(
        &mut traces,
        options.keccak_rnd_rows_log2,
        padding,
        format.keccak_rnd,
    )?;
    split_ecdas(&mut traces, options.ecdas_rows_log2, padding, format.ecdas);
    split_keccak(
        &mut traces,
        options.keccak_rows_log2,
        padding,
        format.memory,
    );
    split_ecsm(&mut traces, options.ecsm_rows_log2, padding);
    split_copy(&mut traces, options.copy_rows_log2, padding, format.memory);
    if deviations.omit_first_keccak_rnd && !traces.keccak_rnds.is_empty() {
        traces.keccak_rnds.remove(0);
    }
    if let Some(tamper) = deviations.tamper_traces {
        tamper(&mut traces);
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
    Ok((proof.with_local(Vec::new()), stamps))
}

/// Cuts every KECCAK_RND table into tables of at most `2^rows_log2` rows, as
/// many as `padding` keeps ([`split_cut`]). A padding row sends
/// `ARE_BYTES[0, 0]` (`keccak_rnd::ARE_BYTES_PER_ROW` times), and BITWISE
/// counted those of the whole padded table, so the rows of the tables the cut
/// leaves out (padding rows alone) are taken back off its multiplicity.
pub(crate) fn split_keccak_rnd(
    traces: &mut Traces,
    rows_log2: usize,
    padding: CutPadding,
    form: crate::tables::keccak_rnd::KeccakRndForm,
) -> Result<(), Error> {
    let rows = 1usize << rows_log2;
    let mu = crate::tables::keccak_rnd::cols::MU;
    let before: usize = traces.keccak_rnds.iter().map(|t| t.num_rows()).sum();
    traces.keccak_rnds = std::mem::take(&mut traces.keccak_rnds)
        .into_iter()
        .flat_map(|table| split_cut(table, rows, mu, padding))
        .collect();
    let after: usize = traces.keccak_rnds.iter().map(|t| t.num_rows()).sum();
    let left_out = before - after;
    crate::tables::bitwise::take_back_multiplicity(
        &mut traces.bitwise,
        crate::tables::bitwise::BitwiseOperation::byte_op(
            crate::tables::bitwise::BitwiseOperationType::AreBytes,
            0,
            0,
        ),
        (left_out * form.bitwise_are_bytes_per_row()) as u64,
    )
}

/// Cuts every ECDAS table into tables of at most `2^rows_log2` rows, as many
/// as `padding` keeps ([`split_cut`]): a cut may fall inside one scalar
/// multiplication, whose steps chain through the Ecdas bus alone
/// ([`BLOCK_ECDAS_ROWS_LOG2`]).
pub(crate) fn split_ecdas(
    traces: &mut Traces,
    rows_log2: usize,
    padding: CutPadding,
    form: crate::tables::ecdas::EcdasForm,
) {
    let rows = 1usize << rows_log2;
    let mu = form.mu();
    traces.ecdases = std::mem::take(&mut traces.ecdases)
        .into_iter()
        .flat_map(|table| split_cut(table, rows, mu, padding))
        .collect();
}

/// Cuts every KECCAK table into tables of at most `2^rows_log2` rows, as many
/// as `padding` keeps ([`split_cut`]): a row is one whole permutation call, so
/// a cut falls between calls ([`BLOCK_KECCAK_ROWS_LOG2`]).
pub(crate) fn split_keccak(
    traces: &mut Traces,
    rows_log2: usize,
    padding: CutPadding,
    memory: crate::tables::fused_memory::MemoryFormat,
) {
    let rows = 1usize << rows_log2;
    // Under the word memory format KECCAK's tables are KECCAK_W's.
    let mu = if memory == crate::tables::fused_memory::MemoryFormat::Words {
        crate::tables::keccak_w::cols::MU
    } else {
        crate::tables::keccak::cols::MU
    };
    traces.keccaks = std::mem::take(&mut traces.keccaks)
        .into_iter()
        .flat_map(|table| split_cut(table, rows, mu, padding))
        .collect();
}

/// Cuts every COPY table into tables of at most `2^rows_log2` rows, as many as
/// `padding` keeps ([`split_cut`]): a cut may fall inside one call, whose rows
/// chain through the CopyNext bus alone ([`BLOCK_COPY_ROWS_LOG2`]).
pub(crate) fn split_copy(
    traces: &mut Traces,
    rows_log2: usize,
    padding: CutPadding,
    memory: crate::tables::fused_memory::MemoryFormat,
) {
    let rows = 1usize << rows_log2;
    // Under the word memory format COPY's tables are COPY_W's.
    let mu = if memory == crate::tables::fused_memory::MemoryFormat::Words {
        crate::tables::copy_w::cols::MU
    } else {
        crate::tables::copy::cols::MU
    };
    traces.copies = std::mem::take(&mut traces.copies)
        .into_iter()
        .flat_map(|table| split_cut(table, rows, mu, padding))
        .collect();
}

/// Cuts every ECSM table into tables of at most `2^rows_log2` rows, as many as
/// `padding` keeps ([`split_cut`]): a row is one whole scalar multiplication,
/// so a cut falls between calls ([`BLOCK_ECSM_ROWS_LOG2`]).
pub(crate) fn split_ecsm(traces: &mut Traces, rows_log2: usize, padding: CutPadding) {
    let rows = 1usize << rows_log2;
    let mu = crate::tables::ecsm::cols::MU;
    traces.ecsms = std::mem::take(&mut traces.ecsms)
        .into_iter()
        .flat_map(|table| split_cut(table, rows, mu, padding))
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
    let fused = traces.fused_counts();
    let fused_memory = fused.map(|n| n as u64);
    fused_tables(format, fused_memory)?;
    let words = traces.word_counts();
    let word_memory = words.map(|n| n as u64);
    word_counts_under(format.memory, word_memory)?;
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();
    let public_output = traces.public_output_bytes.clone();

    let inputs = BlockAirInputs::new(
        proof_options,
        program,
        crate::cpu_heights_from_traces(traces),
        &public_output,
    )?;
    let mut airs = block_airs(
        program,
        proof_options,
        page_air_pages(format.memory, &traces.page_configs),
        &table_counts,
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
        format.registers,
        format.ecdas,
    );
    airs.with_fused_memory(proof_options, fused);
    if format.memory == crate::tables::fused_memory::MemoryFormat::Words {
        airs.with_word_memory_at(
            proof_options,
            words,
            &traces.page_configs,
            format.page_runs(),
        )?;
    }
    check_page_traces(&airs, traces.pages.len())?;
    check_table_count(airs.air_refs().len())?;

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
    let tag = block_format_tag(format);
    let proof = crate::with_base_hash!(format.zf.base, |H| {
        let mut transcript = <H as multilinear::whir_hash::WhirHash>::sponge();
        absorb_block(
            &mut transcript,
            &tag,
            elf_bytes,
            &public_output,
            &table_counts,
            &fused_statement_words(format.memory, fused_memory, word_memory),
            num_private_input_pages,
            &runtime_page_ranges,
            &table_num_vars,
            &config,
            &groups,
        );
        let t = Instant::now();
        let check_preprocessed = !deviations.unchecked_preprocessed;
        let (block, produced) = std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            let producer = scope.spawn(move || -> Result<f64, Error> {
                let mut busy = 0.0;
                for group in groups_of_pairs {
                    let t = Instant::now();
                    let tables = group
                        .into_par_iter()
                        .map(|((air, trace, _), shape)| {
                            table_of_checked(air, trace, shape, release_rows, check_preprocessed)
                        })
                        .collect::<Result<Vec<_>, Error>>()?;
                    busy += t.elapsed().as_secs_f64();
                    if tx.send(tables).is_err() {
                        break;
                    }
                }
                Ok(busy)
            });
            let block = BlockCommitted::commit_streamed::<
                H,
                <H as multilinear::whir_hash::WhirHash>::Sponge,
            >(
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
                fused_memory,
                word_memory,
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
                commitment: (&p.commitment).into(),
                columns,
                packed: None,
            })
            .collect();
        let t = Instant::now();
        let paths_before = multilinear::whir_commit::top_path_counts();
        let routes_before = multilinear::whir_commit::top_path_routes();
        let widens_before = multilinear::narrow::host_widens();
        let fills_before = multilinear::records::fills();
        let times_before = multilinear::records::upload_times();
        let identity = |g: usize| g;
        let fork_of: &dyn Fn(usize) -> usize = match &deviations.fork_of {
            Some(f) => f,
            None => &identity,
        };
        if probe_first_groups()?.is_some() {
            return Err(Error::Prover(
                "W3_PROBE_FIRST_GROUPS orders a streamed prove's phase B only".into(),
            ));
        }
        stamps.phase_b_overlap = options.phase_b_overlap != multilinear_block::PhaseBOverlap::Off;
        stamps.phase_b_pinned = options.phase_b_overlap == multilinear_block::PhaseBOverlap::Pinned;
        let phase_b_sched = phase_b_sched_from_env();
        stamps.phase_b_sched = phase_b_sched;
        let readbacks_before = multilinear::gpu::background_readbacks();
        let (proof, argues, prepared_openings, groups, commit_ood, _) =
            multilinear_block::block_prove_on_forks::<_, _, _, H>(
                block
                    .with_phase_b_overlap(phase_b_overlap(options))
                    .with_overlap_faults(deviations.overlap_faults)
                    .with_sched_stamps(phase_b_sched),
                &config,
                &mut transcript,
                &openings,
                &tampered,
                fork_of,
                &deviations.argue,
            )
            .map_err(|e| Error::Prover(format!("{e:?}")))?;
        stamps.phase_b = t.elapsed().as_secs_f64();
        let readbacks_after = multilinear::gpu::background_readbacks();
        stamps.background_readbacks = (
            readbacks_after.0 - readbacks_before.0,
            readbacks_after.1 - readbacks_before.1,
            readbacks_after.2,
        );
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
        let fills_after = multilinear::records::fills();
        stamps.records_fills = (
            fills_after.0 - fills_before.0,
            fills_after.1 - fills_before.1,
        );
        let ((widens, widen_s), (fills, fill_s)) = multilinear::records::upload_times();
        stamps.upload_times = (
            (widens - times_before.0.0, widen_s - times_before.0.1),
            (fills - times_before.1.0, fill_s - times_before.1.1),
        );
        stamps.groups = groups;
        (proof, argues, prepared_openings, commit_ood)
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
        commit_ood: proof.3,
        fused_memory,
        word_memory,
        local: (),
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
    /// One CPU AIR a streamed CPU chunk, built on first use: chunk `k` is the
    /// block's CPU table `k`, every one before it full, so its timestamps start
    /// at `4·k·cpu_rows` (`tables::cpu::timestamp_bases`) and its preprocessed
    /// TIMESTAMP is the column of its height. A CPU AIR is per table since main's
    /// preprocessed timestamp, and the argue reads the AIR a chunk was laid out
    /// with. In pages of [`CPU_AIR_PAGE`], each allocated on first use, so a
    /// short chunk height does not allocate a slot for every table the row
    /// bound allows up front.
    cpus: Vec<std::sync::OnceLock<Box<[std::sync::OnceLock<crate::VmAir>]>>>,
    /// The CPU chunk height (`MaxRowsConfig::cpu`).
    cpu_rows: usize,
    /// The run's register format: under
    /// [`RegisterFormat::InRow`](crate::tables::register::RegisterFormat::InRow)
    /// a CPU chunk is laid out with the in-row CPU AIR and MEMW_R's chunks
    /// with the residual register table's, as [`crate::block_airs`] builds
    /// them for the verifier.
    registers: crate::tables::register::RegisterFormat,
    opts: ProofOptions,
    memw_register: crate::VmAir,
    memw_aligned: crate::VmAir,
    memw: crate::VmAir,
    load: crate::VmAir,
    lt: crate::VmAir,
    shift: crate::VmAir,
    store: crate::VmAir,
    keccak_rnd: crate::VmAir,
    /// The fused memory format's families (streamed only under it).
    mem_load: crate::VmAir,
    mem_store: crate::VmAir,
    /// The word families' AIRs ([`StreamTable::WORDS`]' order), for their
    /// streamed chunks.
    words: [crate::VmAir; 7],
    /// KECCAK_RND's form, and the LOCAL_BYTES AIR its local lookups' groups
    /// end with ([`BlockFormat::local_lookups`]).
    keccak_rnd_form: crate::tables::keccak_rnd::KeccakRndForm,
    local_bytes: crate::VmAir,
}

type DynAir = dyn stark::traits::AIR<Field = F, FieldExtension = E, PublicInputs = ()>;

/// CPU AIR slots per page of [`StreamAirs`]'s CPU tables.
const CPU_AIR_PAGE: usize = 256;

impl StreamAirs {
    fn new(
        opts: &ProofOptions,
        form: TraceForm,
        cpu_rows: usize,
        keccak_rnd: crate::tables::keccak_rnd::KeccakRndForm,
        memw_aligned: crate::tables::memw_aligned::MemwAlignedOrder,
        registers: crate::tables::register::RegisterFormat,
    ) -> Self {
        use crate::tables::register::RegisterFormat;
        use crate::test_utils::*;
        // As many CPU tables as a run's rows allow (`tables::cpu::MAX_CPU_ROWS`).
        let most = (crate::tables::cpu::MAX_CPU_ROWS as usize).div_ceil(cpu_rows.max(1));
        Self {
            form,
            cpus: (0..most.div_ceil(CPU_AIR_PAGE))
                .map(|_| std::sync::OnceLock::new())
                .collect(),
            cpu_rows,
            registers,
            opts: opts.clone(),
            memw_register: match registers {
                RegisterFormat::Tokens32 => Box::new(create_memw_register_air(opts)),
                RegisterFormat::InRow => Box::new(create_register_residual_air(opts)),
            },
            memw_aligned: Box::new(create_memw_aligned_air_for(opts, memw_aligned)),
            memw: Box::new(create_memw_air(opts)),
            load: Box::new(create_load_air(opts)),
            lt: Box::new(create_lt_air(opts)),
            shift: Box::new(create_shift_air(opts)),
            store: Box::new(create_store_air(opts)),
            keccak_rnd: Box::new(create_keccak_rnd_air_for(opts, keccak_rnd)),
            mem_load: Box::new(create_mem_load_air(opts)),
            mem_store: Box::new(create_mem_store_air(opts)),
            words: [
                Box::new(crate::test_utils::create_mem_load8_air(opts)),
                Box::new(crate::test_utils::create_mem_store8_air(opts)),
                Box::new(crate::test_utils::create_mem_loadw_air(opts)),
                Box::new(crate::test_utils::create_mem_storew_air(opts)),
                Box::new(crate::test_utils::create_mem_load2w_air(opts)),
                Box::new(crate::test_utils::create_mem_store2w_air(opts)),
                Box::new(crate::test_utils::create_memw_w_air(opts)),
            ],
            keccak_rnd_form: keccak_rnd,
            local_bytes: Box::new(create_local_bytes_air(opts)),
        }
    }

    /// A laid-out table's local sends counted ([`local_bytes::local_counts_of`]),
    /// on the thread that laid it out; `None` outside local lookups or for a
    /// table with none.
    fn local_counts(&self, table: &CommittedTable<'_, F, E>) -> Result<Option<LocalCounts>, Error> {
        local_counts_under(self.keccak_rnd_form, table)
    }

    /// The AIR streamed chunk `index` of `table`, `height` rows, is laid out
    /// with: the family's, or CPU table `index`'s own.
    fn of_chunk(&self, table: StreamTable, index: usize, height: usize) -> Result<&DynAir, Error> {
        if let Some(air) = self.of(table) {
            return Ok(air);
        }
        let page = self.cpus.get(index / CPU_AIR_PAGE).ok_or_else(|| {
            Error::Prover(format!("CPU chunk {index} is past the run's row bound"))
        })?;
        let page = page.get_or_init(|| {
            (0..CPU_AIR_PAGE)
                .map(|_| std::sync::OnceLock::new())
                .collect()
        });
        Ok(page[index % CPU_AIR_PAGE]
            .get_or_init(|| {
                let ts_base = crate::tables::cpu::TIMESTAMP_STRIDE * (index * self.cpu_rows) as u64;
                let opts = &self.opts;
                let name = format!("CPU[{index}]");
                let commitment = crate::tables::cpu::lazy_timestamp_commitment(opts, height);
                let columns: std::sync::Arc<
                    dyn Fn() -> Vec<Vec<crate::tables::types::FE>> + Send + Sync,
                > = std::sync::Arc::new(move || vec![crate::tables::cpu::timestamp_column(height)]);
                let cols = crate::tables::cpu::NUM_PRECOMPUTED_COLS;
                match self.registers {
                    crate::tables::register::RegisterFormat::Tokens32 => Box::new(
                        crate::test_utils::create_cpu_air(opts, ts_base)
                            .with_name(&name)
                            .with_lazy_preprocessed_columns(commitment, cols, columns),
                    )
                        as crate::VmAir,
                    crate::tables::register::RegisterFormat::InRow => Box::new(
                        crate::test_utils::create_cpu_air_inrow(opts, ts_base)
                            .with_name(&name)
                            .with_lazy_preprocessed_columns(commitment, cols, columns),
                    ),
                }
            })
            .as_ref())
    }

    /// The family's AIR, for every streamed family but CPU, whose AIR is a
    /// table's own ([`Self::of_chunk`]).
    fn of(&self, table: StreamTable) -> Option<&DynAir> {
        Some(match table {
            StreamTable::Cpu => return None,
            StreamTable::MemwRegister => self.memw_register.as_ref(),
            StreamTable::MemwAligned => self.memw_aligned.as_ref(),
            StreamTable::Memw => self.memw.as_ref(),
            StreamTable::Load => self.load.as_ref(),
            StreamTable::Lt => self.lt.as_ref(),
            StreamTable::Shift => self.shift.as_ref(),
            StreamTable::Store => self.store.as_ref(),
            StreamTable::KeccakRnd => self.keccak_rnd.as_ref(),
            StreamTable::MemLoad => self.mem_load.as_ref(),
            StreamTable::MemStore => self.mem_store.as_ref(),
            word => self.words[word.word_family()?].as_ref(),
        })
    }
}

/// A laid-out table's local sends counted under `form`
/// ([`local_bytes::local_counts_of`]); `None` when `form` sends no local
/// lookup or the table sends none.
fn local_counts_under(
    form: crate::tables::keccak_rnd::KeccakRndForm,
    table: &CommittedTable<'_, F, E>,
) -> Result<Option<LocalCounts>, Error> {
    if !form.byte_lookups_local() {
        return Ok(None);
    }
    local_bytes::local_counts_of(table, form).map_err(Error::Prover)
}

/// Each table's digest ([`multilinear_block::table_digest`]) in the block's
/// table order (`groups` concatenated), for the groups that hold a local
/// sender; every other table's is never read and stays zero. `air_of(i)` is
/// table `i`'s AIR, `shapes[i]` its `(width, height in variables)`. One
/// function for both sides: the preprocessed count is the AIR's, which the
/// prover and the verifier share.
pub(crate) fn local_digests<'r, T>(
    groups: &[Vec<u32>],
    shapes: &[(usize, usize)],
    air_of: impl Fn(usize) -> Option<AirRef<'r>>,
) -> Result<Vec<[u8; 32]>, Error>
where
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E> + Default,
{
    let mut digests = Vec::new();
    // Tables of one definition share a digest: hashed once. The key is the
    // digest's every input — the interaction list (its address and its
    // length: every table with no interaction shares the one empty list),
    // the height, the width and the preprocessed count — so two tables
    // share a digest only when their digests are the same.
    type Key = (usize, usize, usize, usize, usize);
    let mut seen: Vec<(Key, [u8; 32])> = Vec::new();
    for group in groups {
        let airs: Vec<AirRef<'r>> = group
            .iter()
            .map(|&i| {
                air_of(i as usize).ok_or_else(|| Error::Prover(format!("table {i} has no AIR")))
            })
            .collect::<Result<_, _>>()?;
        let local = airs
            .iter()
            .any(|air| local_bytes::sends_local(air.bus_interactions()));
        for (&i, air) in group.iter().zip(&airs) {
            if !local {
                digests.push([0u8; 32]);
                continue;
            }
            let &(width, num_vars) = shapes
                .get(i as usize)
                .ok_or_else(|| Error::Prover(format!("table {i} has no shape")))?;
            let preprocessed = if air.is_preprocessed() {
                air.num_precomputed_columns()
            } else {
                0
            };
            let key = (
                air.bus_interactions().as_ptr() as usize,
                air.bus_interactions().len(),
                num_vars,
                width,
                preprocessed,
            );
            let digest = match seen.iter().find(|(k, _)| *k == key) {
                Some((_, digest)) => *digest,
                None => {
                    let digest = multilinear_block::table_digest::<E, T>(
                        num_vars,
                        width,
                        preprocessed,
                        air.bus_interactions(),
                    );
                    seen.push((key, digest));
                    digest
                }
            };
            digests.push(digest);
        }
    }
    Ok(digests)
}

/// The name the run's AIR set gives chunk `index` of a streamed table.
fn stream_name(table: StreamTable, index: usize) -> String {
    format!("{}[{index}]", stream_base(table))
}

/// A streamed table's AIR name, without the chunk's index.
fn stream_base(table: StreamTable) -> &'static str {
    match table {
        StreamTable::Cpu => "CPU",
        StreamTable::MemwRegister => "MEMW_R",
        StreamTable::MemwAligned => "MEMW_A",
        StreamTable::Memw => "MEMW",
        StreamTable::Load => "LOAD",
        StreamTable::Lt => "LT",
        StreamTable::Shift => "SHIFT",
        StreamTable::Store => "STORE",
        StreamTable::KeccakRnd => "KECCAK_RND",
        StreamTable::MemLoad => "MEM_LOAD",
        StreamTable::MemStore => "MEM_STORE",
        StreamTable::MemLoad8 => "MEM_LOAD8",
        StreamTable::MemStore8 => "MEM_STORE8",
        StreamTable::MemLoadW => "MEM_LOADW",
        StreamTable::MemStoreW => "MEM_STOREW",
        StreamTable::MemLoad2W => "MEM_LOAD2W",
        StreamTable::MemStore2W => "MEM_STORE2W",
        StreamTable::MemwW => "MEMW_W",
    }
}

/// What the builder thread reports when the run is built: when its windows
/// were all collected and when the run was built (seconds since the prove
/// started), how many chunks it handed out, the windows' stamps and `finish`'s
/// phase marks.
type BuilderReport = (
    f64,
    f64,
    usize,
    WindowStamps,
    Vec<(String, f64)>,
    Option<crate::tables::trace_builder::RestRegen>,
);

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
/// the next AIR in the order the finish hands them on (AIR order, or BITWISE
/// last: [`crate::tables::trace_builder::RestHeader::bitwise_last`], with
/// KECCAK_RND's first:
/// [`crate::tables::trace_builder::RestHeader::keccak_rnd_first`]), a
/// streamed chunk's slot or a table of that AIR's width, and every AIR
/// reached. A table's height is checked where it is laid
/// out ([`layout_of`]).
struct RestOrder {
    widths: Vec<usize>,
    /// The positions in the order the finish hands them on
    /// ([`crate::tables::trace_builder::hand_on_order_first`]).
    due: Vec<usize>,
    next: usize,
}

impl RestOrder {
    /// The rest's tables of `widths` columns, due in AIR order, or with the
    /// first (BITWISE) after every other when `bitwise_last`.
    #[cfg(test)]
    fn new(widths: Vec<usize>, bitwise_last: bool) -> Self {
        Self::with_first(widths, &[], bitwise_last)
    }

    /// [`Self::new`], with the positions `first` (ascending) due before every
    /// other ([`crate::tables::trace_builder::hand_on_order_first`]).
    fn with_first(widths: Vec<usize>, first: &[usize], bitwise_last: bool) -> Self {
        let due = crate::tables::trace_builder::hand_on_order_first(
            widths.len(),
            first,
            bitwise_last.then_some(0),
        );
        Self {
            widths,
            due,
            next: 0,
        }
    }

    fn admit(&mut self, position: usize, width: Option<usize>) -> Result<(), Error> {
        let want = *self.widths.get(position).ok_or_else(|| {
            Error::Prover(format!(
                "the rest's table {position} is past the run's {} tables",
                self.widths.len()
            ))
        })?;
        let due = self.due.get(self.next).copied();
        if Some(position) != due {
            return Err(Error::Prover(format!(
                "the rest's table {position} came where table {} was due",
                due.map_or("none".to_string(), |d| d.to_string())
            )));
        }
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

    /// Whether the only table still due is the one at `position`.
    fn only_due(&self, position: usize) -> bool {
        self.due.get(self.next..) == Some(&[position][..])
    }
}

/// The packer told, ahead of BITWISE (handed last), that only BITWISE is left
/// and what shape it has ([`close_before_bitwise`]): the rest's message after
/// which BITWISE is the only table due, and its shape; and whether the packer
/// closed its open group then.
#[derive(Default)]
pub(crate) struct CloseAhead {
    last: std::sync::OnceLock<(usize, (usize, usize))>,
    closed: std::sync::atomic::AtomicBool,
}

/// The knob of [`close_before_bitwise`].
const CLOSE_BEFORE_BITWISE_ENV: &str = "LAMBDA_VM_CLOSE_BEFORE_BITWISE";

/// Whether the packer closes its open group as soon as BITWISE (handed last,
/// its shape known) is the only table left and would not fit it, rather than
/// when BITWISE arrives: a deviation's choice, else
/// `LAMBDA_VM_CLOSE_BEFORE_BITWISE` (off by default). The group would close at
/// BITWISE's arrival all the same, so the groups and the proof's bytes are the
/// same; the group's commit runs beside BITWISE's build instead of after it.
fn close_before_bitwise(deviation: Option<bool>) -> Result<bool, Error> {
    match deviation {
        Some(on) => Ok(on),
        None => parse_close_before_bitwise(std::env::var(CLOSE_BEFORE_BITWISE_ENV).ok().as_deref()),
    }
}

/// `LAMBDA_VM_CLOSE_BEFORE_BITWISE`: unset, empty or `0` (the default) closes
/// at BITWISE's arrival; `1` closes ahead of it. Anything else refuses the
/// prove.
fn parse_close_before_bitwise(value: Option<&str>) -> Result<bool, Error> {
    match value.map(str::trim) {
        None | Some("" | "0") => Ok(false),
        Some("1") => Ok(true),
        Some(v) => Err(Error::Prover(format!(
            "{CLOSE_BEFORE_BITWISE_ENV}={v}: 0 (the default: at BITWISE's arrival) or 1 (ahead of it)"
        ))),
    }
}

/// Whether `open` and then `next` would take a group past `max_polys` under
/// `cap`: the rule [`Packer::place`] closes a group by.
fn past_budget(
    open: &[(usize, usize)],
    next: (usize, usize),
    cap: multilinear::whir_chain::StackVars,
    max_polys: usize,
) -> Result<bool, Error> {
    let mut shapes = open.to_vec();
    shapes.push(next);
    Ok(stark::multilinear_table::global_layout(&shapes, cap)
        .map_err(|e| Error::Prover(format!("{e:?}")))?
        .num_polys()
        > max_polys)
}

/// The rest's positions of KECCAK_RND's tables, the order the finish hands them
/// on in when it hands them first: their range among the chunked tables
/// ([`chunked_table_ranges`]), each checked to name a KECCAK_RND AIR.
fn keccak_rnd_positions(counts: &TableCounts, names: &[&str]) -> Result<Vec<usize>, Error> {
    let range = chunked_table_ranges(counts)
        .into_iter()
        .find(|(name, ..)| *name == "KECCAK_RND")
        .map(|(_, range, _)| range)
        .ok_or_else(|| Error::Prover("no KECCAK_RND range among the chunked tables".into()))?;
    for p in range.clone() {
        let name = names.get(p);
        if !name.is_some_and(|n| n.starts_with("KECCAK_RND")) {
            return Err(Error::Prover(format!(
                "KECCAK_RND goes first, but the rest's table {p} is {name:?}"
            )));
        }
    }
    Ok(range.collect())
}

/// A table before the run's AIR order exists: a streamed chunk, or a table of
/// the final build by its AIR index.
#[derive(Clone, Copy)]
enum Key {
    Streamed(StreamTable, usize),
    Air(usize),
    /// The `j`-th LOCAL_BYTES table, which closes its group: index `N + j`,
    /// `N` the run's other tables.
    Local(usize),
}

/// A table of the final build, laid out: its AIR index, shape and table, and
/// its local sends' counts ([`local_counts_under`]).
type Placed<'a> = (
    usize,
    (usize, usize),
    CommittedTable<'a, F, E>,
    Option<LocalCounts>,
);

/// Packs tables, in the order they come, into groups of at most `max_polys`
/// stacked polynomials, and sends each group to phase A as it closes. With
/// [`Self::own`], KECCAK_RND's streamed chunks are packed apart, in groups of
/// their own ([`OwnGroups`]).
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
    /// KECCAK_RND's streamed chunks' own open group, when they are packed
    /// apart (`None`: packed with the others).
    own: Option<OwnGroups<'a>>,
    /// Told ahead of BITWISE ([`close_before_bitwise`]).
    close_ahead: Option<std::sync::Arc<CloseAhead>>,
    /// Under local lookups: each group with a local sender closes with its
    /// LOCAL_BYTES table, over the group's counts.
    local: Option<LocalPack<'a>>,
}

/// The packer's local lookups ([`BlockFormat::local_lookups`]): the open
/// groups' counts and the LOCAL_BYTES tables made so far.
struct LocalPack<'a> {
    air: AirRef<'a>,
    /// A test's wrong tables ([`Deviations::local_bytes`]).
    fault: LocalBytesFault,
    /// The main open group's counts, and KECCAK_RND's own open group's.
    open: Option<LocalCounts>,
    own: Option<LocalCounts>,
    /// LOCAL_BYTES tables made: the next one's `j`.
    made: usize,
    /// Where each LOCAL_BYTES table sits in phase A's order of tables (the
    /// groups' tables one after another, as they are sent), for live
    /// regeneration ([`LocalPlaces`]).
    places: LocalPlaces,
}

/// Where the LOCAL_BYTES tables sit in phase A's order of tables: the groups'
/// tables one after another, as the packer sends them. A LOCAL_BYTES table's
/// block index is `N + j`, after every other table, while in phase A it is the
/// last table of its group; so a table's place in phase A is its block index
/// plus the LOCAL_BYTES tables sent before it. Live regeneration names a table
/// it drops by its block index ([`block_index_of`]).
#[derive(Clone, Default)]
pub(crate) struct LocalPlaces(std::sync::Arc<std::sync::Mutex<Vec<usize>>>);

impl LocalPlaces {
    /// A LOCAL_BYTES table sent at `place` in phase A's order.
    pub(crate) fn push(&self, place: usize) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).push(place);
    }

    /// The block index of the table at `place` in phase A's order, or `None`
    /// for a LOCAL_BYTES table (no regenerator builds one again).
    pub(crate) fn block_index_of(&self, place: usize) -> Option<usize> {
        let places = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if places.contains(&place) {
            return None;
        }
        Some(place - places.iter().filter(|&&q| q < place).count())
    }
}

impl LocalPack<'_> {
    /// `counts` added to an open group's (`own`: KECCAK_RND's own).
    fn add(&mut self, own: bool, counts: LocalCounts) -> Result<(), Error> {
        let slot = if own { &mut self.own } else { &mut self.open };
        match slot {
            Some(sum) => sum.add(&counts).map_err(Error::Prover),
            None => {
                *slot = Some(counts);
                Ok(())
            }
        }
    }
}

/// KECCAK_RND's streamed chunks packed apart from the other streamed chunks
/// ([`own_keccak_rnd_groups`]): its open group, the hand-out place of each
/// table of either open group, and the regeneration recorder told each closed
/// group's places, since the groups then close out of hand-out order.
struct OwnGroups<'a> {
    open: Vec<CommittedTable<'a, F, E>>,
    shapes: Vec<(usize, usize)>,
    keys: Vec<Key>,
    places: Vec<Option<usize>>,
    bytes: usize,
    /// The other open group's tables' hand-out places.
    main_places: Vec<Option<usize>>,
    /// The streamed chunks placed so far: the next one's hand-out place.
    placed: usize,
    recorder: Option<&'a regen::Recorder>,
}

impl<'a> Packer<'a> {
    /// A group closes, and goes to the card, when the next table would take
    /// it past its polynomial budget. Under [`Self::own`] a KECCAK_RND chunk
    /// joins its own open group, which closes by the same rule, and is closed
    /// as it stands before the rest's first table.
    fn place(
        &mut self,
        key: Key,
        shape: (usize, usize),
        table: CommittedTable<'a, F, E>,
        counts: Option<LocalCounts>,
    ) -> Result<(), Error> {
        if counts.is_some() && self.local.is_none() {
            return Err(Error::Prover(
                "a local sender outside the format's local lookups".into(),
            ));
        }
        let place = self.own.as_mut().and_then(|own| match key {
            Key::Streamed(..) => {
                own.placed += 1;
                Some(own.placed - 1)
            }
            Key::Air(_) | Key::Local(_) => None,
        });
        if let Some(own) = self.own.as_mut() {
            if let Key::Streamed(StreamTable::KeccakRnd, _) = key {
                own.shapes.push(shape);
                let polys = stark::multilinear_table::global_layout(&own.shapes, self.cap)
                    .map_err(|e| Error::Prover(format!("{e:?}")))?
                    .num_polys();
                if polys > self.max_polys && !own.open.is_empty() {
                    self.close_own()?;
                }
                let own = self.own.as_mut().expect("packed apart");
                if own.shapes.is_empty() {
                    own.shapes.push(shape);
                }
                let bytes = self.ledger.map_or(0, |ledger| {
                    let bytes = memlog::table_bytes(&table);
                    ledger
                        .open
                        .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
                    bytes
                });
                own.bytes += bytes;
                // Its counts join the group it joins (after any close above).
                if let (Some(counts), Some(local)) = (counts, self.local.as_mut()) {
                    local.add(true, counts)?;
                }
                let own = self.own.as_mut().expect("packed apart");
                own.open.push(table);
                own.keys.push(key);
                own.places.push(place);
                return Ok(());
            }
            // The rest begins: KECCAK_RND's open group goes first, so every
            // streamed chunk is committed before the rest's first table.
            if matches!(key, Key::Air(_)) && !own.open.is_empty() {
                self.close_own()?;
            }
        }
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
        // Its counts join the group it joins (after any close above).
        if let (Some(counts), Some(local)) = (counts, self.local.as_mut()) {
            local.add(false, counts)?;
        }
        self.open.push(table);
        self.open_keys.push(key);
        if let Some(own) = self.own.as_mut() {
            own.main_places.push(place);
        }
        Ok(())
    }

    /// After the rest's message `k`: when the packer was told that only
    /// BITWISE follows it and BITWISE's shape would take the open group past
    /// its budget, the group closes now, as it would at BITWISE's arrival.
    fn close_ahead_of_bitwise(&mut self, k: usize) -> Result<(), Error> {
        let Some(ahead) = self.close_ahead.clone() else {
            return Ok(());
        };
        let Some(&(after, shape)) = ahead.last.get() else {
            return Ok(());
        };
        if after != k || self.open.is_empty() {
            return Ok(());
        }
        if past_budget(&self.open_shapes, shape, self.cap, self.max_polys)? {
            eprintln!(
                "BLOCK PACKER: the open group ({} tables) closed before BITWISE, which would not fit it",
                self.open.len()
            );
            self.open_shapes.clear();
            self.close()?;
            ahead
                .closed
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    fn close(&mut self) -> Result<(), Error> {
        let mut tables = std::mem::take(&mut self.open);
        let mut keys = std::mem::take(&mut self.open_keys);
        let mut bytes = std::mem::take(&mut self.open_bytes);
        let mut places = self
            .own
            .as_mut()
            .map(|own| std::mem::take(&mut own.main_places));
        let counts = self.local.as_mut().and_then(|local| local.open.take());
        self.close_local(counts, &mut tables, &mut keys, &mut bytes, places.as_mut())?;
        self.send(tables, keys, bytes, places)
    }

    /// A group with a local sender closed with its LOCAL_BYTES table over
    /// `counts`, its last table (key [`Key::Local`]); nothing without counts.
    fn close_local(
        &mut self,
        counts: Option<LocalCounts>,
        tables: &mut Vec<CommittedTable<'a, F, E>>,
        keys: &mut Vec<Key>,
        bytes: &mut usize,
        places: Option<&mut Vec<Option<usize>>>,
    ) -> Result<(), Error> {
        let (Some(mut counts), Some(local)) = (counts, self.local.as_mut()) else {
            return Ok(());
        };
        let mut trace = match local.fault {
            LocalBytesFault::None => counts.trace(),
            LocalBytesFault::OneMoreReceived => {
                counts.are_bytes[local_bytes::row_of(1, 2)] += 1;
                counts.trace()
            }
            LocalBytesFault::WrongUnusedXor => {
                let row = counts.xor.iter().position(|&m| m == 0).ok_or_else(|| {
                    Error::Prover("every LOCAL_BYTES row is an XOR lookup's".to_string())
                })?;
                let mut columns = local_bytes::preprocessed_columns();
                columns[local_bytes::cols::XOR][row] += FieldElement::<F>::one();
                for column in [&counts.are_bytes, &counts.and, &counts.xor] {
                    columns.push(
                        column
                            .iter()
                            .map(|&m| FieldElement::<F>::from(u64::from(m)))
                            .collect(),
                    );
                }
                stark::trace::TraceTable::from_columns_main(columns, 1)
            }
        };
        // A test's wrong fixed column is committed as it stands (the
        // verifier's check is what refuses it).
        let table = table_of_checked(
            local.air,
            &mut trace,
            (local_bytes::cols::NUM_COLUMNS, local_bytes::LOG_ROWS),
            true,
            local.fault != LocalBytesFault::WrongUnusedXor,
        )?;
        if let Some(ledger) = self.ledger {
            let b = memlog::table_bytes(&table);
            ledger
                .open
                .fetch_add(b, std::sync::atomic::Ordering::Relaxed);
            *bytes += b;
        }
        tables.push(table);
        keys.push(Key::Local(local.made));
        local.made += 1;
        if let Some(places) = places {
            places.push(None);
        }
        Ok(())
    }

    /// KECCAK_RND's own open group, closed and sent.
    fn close_own(&mut self) -> Result<(), Error> {
        let Some(own) = self.own.as_mut() else {
            return Ok(());
        };
        let mut tables = std::mem::take(&mut own.open);
        let mut keys = std::mem::take(&mut own.keys);
        let mut bytes = std::mem::take(&mut own.bytes);
        let mut places = std::mem::take(&mut own.places);
        own.shapes.clear();
        let counts = self.local.as_mut().and_then(|local| local.own.take());
        self.close_local(
            counts,
            &mut tables,
            &mut keys,
            &mut bytes,
            Some(&mut places),
        )?;
        self.send(tables, keys, bytes, Some(places))
    }

    /// One group to phase A: its tables, keys and ledger bytes, and under
    /// [`Self::own`] each table's hand-out place, told to the recorder before
    /// phase A can take the group.
    fn send(
        &mut self,
        tables: Vec<CommittedTable<'a, F, E>>,
        keys: Vec<Key>,
        bytes: usize,
        places: Option<Vec<Option<usize>>>,
    ) -> Result<(), Error> {
        // As its groups close the prover knows only how far the block got, not
        // its total: the refusal says so, so the index is not read as a count.
        let reached = self.group_keys.len() + 1;
        if reached > self.max_groups {
            return Err(Error::InvalidTableCounts(format!(
                "a block takes at most {} groups; this one reached {reached} as its groups close",
                self.max_groups
            )));
        }
        if let Some(ledger) = self.ledger {
            use std::sync::atomic::Ordering::Relaxed;
            ledger.open.fetch_sub(bytes, Relaxed);
            ledger.sent.fetch_add(bytes, Relaxed);
        }
        if let (Some(places), Some(recorder)) =
            (places, self.own.as_ref().and_then(|own| own.recorder))
        {
            recorder.committed(&places);
        }
        if let Some(local) = &self.local {
            let first: usize = self.group_keys.iter().map(Vec::len).sum();
            for (k, key) in keys.iter().enumerate() {
                if matches!(key, Key::Local(_)) {
                    local.places.push(first + k);
                }
            }
        }
        self.group_keys.push(keys);
        self.closed_at.push(self.start.elapsed().as_secs_f64());
        let t = Instant::now();
        let sent = self
            .out
            .send(tables)
            .map_err(|_| Error::Prover("phase A stopped".into()));
        self.blocked += t.elapsed().as_secs_f64();
        sent
    }

    /// Sends the last groups (KECCAK_RND's own first, if any is open); hands
    /// back every group's tables, when each closed and the seconds spent
    /// waiting on phase A.
    fn finish(mut self) -> Result<Packed, Error> {
        if self.own.as_ref().is_some_and(|own| !own.open.is_empty()) {
            self.close_own()?;
        }
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
type LaidChunk<'a> = (
    StreamTable,
    usize,
    (usize, usize),
    CommittedTable<'a, F, E>,
    Option<LocalCounts>,
);

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
    // A chunk held as its records is never dropped: it has no recipe to
    // digest for.
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
    // Held as its records: the card fills its columns as its group uploads
    // it. Written packed (G-pack): laid out narrow from its packed columns,
    // with no transposition.
    let air = airs.of_chunk(chunk.table, chunk.index, height)?;
    let table = if chunk.trace.records_main().is_some() {
        table_of_records(air, &mut chunk.trace, shape)
    } else if chunk.trace.narrow_main().is_some() {
        table_of_narrow(air, &mut chunk.trace, shape)
    } else {
        table_of(air, &mut chunk.trace, shape, true)
    };
    if let Some(ledger) = ledger {
        ledger.laying.fetch_sub(rows, Relaxed);
    }
    // Its local sends counted here, on the layout worker.
    let table = table?;
    let counts = airs.local_counts(&table)?;
    Ok((chunk.table, chunk.index, shape, table, counts))
}

/// The streamed chunks, packed in arrival order: the packer, each chunk's
/// shape, the seconds laying them out (summed over threads) and when the last
/// was placed; and the rest's tables it packed as they were laid out, by AIR
/// index (a streamed rest's).
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
                let (seq, kind) = (shapes.len(), job.table);
                let t = Instant::now();
                let (table, index, shape, laid, counts) =
                    lay_out_chunk(airs, *job, ledger, recorder)?;
                head_log::chunk_laid(seq, stream_base(kind), t);
                chunks += t.elapsed().as_secs_f64();
                shapes.push((table, index, shape));
                packer.place(Key::Streamed(table, index), shape, laid, counts)?;
                head_log::chunk_placed(seq);
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
/// the rest of it (`rest`) while the last chunks are packed. With a streamed
/// rest, `rest` sends the rest's tables to the packer as they are laid out,
/// and it packs them after the chunks, in AIR
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
    gate: Option<&'a crate::tables::trace_builder::gate::ByteGate>,
    recorder: Option<&regen::Recorder>,
    waits: Option<&ProducerWaits>,
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
                    let t = Instant::now();
                    if let Some(permits) = permits
                        && !permits.take()
                    {
                        return;
                    }
                    if let Some(waits) = waits {
                        ProducerWaits::add(&waits.worker_permit, t);
                    }
                    // The lock is held for the receive alone. A poisoned lock
                    // means a sibling panicked, which the scope reports.
                    let t = Instant::now();
                    let next = match jrx.lock() {
                        Ok(jobs) => jobs.recv(),
                        Err(_) => Err(std::sync::mpsc::RecvError),
                    };
                    if let Some(waits) = waits {
                        ProducerWaits::add(&waits.worker_recv, t);
                    }
                    let Ok((seq, job)) = next else {
                        if let Some(permits) = permits {
                            permits.give();
                        }
                        return;
                    };
                    let (kind, t) = (job.table, Instant::now());
                    let laid = lay_out_chunk(airs, *job, ledger, recorder);
                    head_log::chunk_laid(seq, stream_base(kind), t);
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
                    let (table, index, shape, laid, counts) = laid?;
                    shapes.push((table, index, shape));
                    packer.place(Key::Streamed(table, index), shape, laid, counts)?;
                    head_log::chunk_placed(next);
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
            // The rest's tables, after every chunk, in AIR order: the first at
            // the block index after the last chunk's.
            if let Some(recorder) = recorder {
                recorder.set_rest_base(shapes.len());
            }
            let mut placed = Vec::new();
            let mut waiting = std::collections::BTreeMap::new();
            let mut next = 0usize;
            for (k, _, laid) in rrx {
                waiting.insert(k, laid);
                while let Some(laid) = waiting.remove(&next) {
                    let (i, shape, table, counts) = laid?;
                    placed.push((i, shape));
                    if let Some(ahead) = &packer.close_ahead
                        && let Some(&(after, told)) = ahead.last.get()
                        && after + 1 == next
                        && told != shape
                    {
                        // Told wrong, the groups may differ from the
                        // packer's own rule's (any partition proves; the
                        // bytes are then not the knob-off ones).
                        eprintln!(
                            "BLOCK PACKER: BITWISE's shape {shape:?} is not the {told:?} it was told"
                        );
                    }
                    packer.place(Key::Air(i), shape, table, counts)?;
                    packer.close_ahead_of_bitwise(next)?;
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
                    let t = Instant::now();
                    if jtx.send((handed, job)).is_err() {
                        break;
                    }
                    if let Some(waits) = waits {
                        ProducerWaits::add(&waits.layout_send, t);
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
            Some(traces) => {
                drop(rtx);
                Some(rest(traces, None))
            }
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
    /// [`BlockWhirProof::fused_memory`].
    fused_memory: [u64; 2],
    /// [`BlockWhirProof::word_memory`].
    word_memory: [u64; 7],
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
    /// [`LayoutStamps::records_rest`].
    records_rest: (usize, usize),
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
    cpu_rows: usize,
) -> Result<RestLaid<'a>, Error> {
    let t = Instant::now();
    let at = || start.elapsed().as_secs_f64();
    let mut marks = vec![("rest received", at())];
    let table_counts = traces.table_counts();
    validate_block_counts(&table_counts)?;
    let fused = traces.fused_counts();
    let fused_memory = fused.map(|n| n as u64);
    fused_tables(format, fused_memory)?;
    let words = traces.word_counts();
    let word_memory = words.map(|n| n as u64);
    word_counts_under(format.memory, word_memory)?;
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();
    let public_output = traces.public_output_bytes.clone();
    let inputs = BlockAirInputs::new(
        opts,
        program,
        rest_cpu_heights(&traces, cpu_rows)?,
        &public_output,
    )?;
    let mut built = Ok(());
    let airs = run_airs.get_or_init(|| {
        let mut airs = block_airs(
            program,
            opts,
            page_air_pages(format.memory, &traces.page_configs),
            &table_counts,
            &inputs,
            format.keccak_rnd,
            format.memw_aligned,
            format.registers,
            format.ecdas,
        );
        airs.with_fused_memory(opts, fused);
        if format.memory == crate::tables::fused_memory::MemoryFormat::Words {
            built = airs.with_word_memory_at(opts, words, &traces.page_configs, format.page_runs());
        }
        airs
    });
    built?;
    check_page_traces(airs, traces.pages.len())?;
    check_table_count(airs.air_refs().len())?;
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
    let records_tables = std::sync::atomic::AtomicUsize::new(0);
    let records_bytes = std::sync::atomic::AtomicUsize::new(0);
    let lay_out = |(i, ((_, trace, _), air)): (usize, ((_, &mut TraceTable<F, E>, _), _))| {
        let t = Instant::now();
        let shape = (
            trace.main_table.width,
            trace.main_table.height.trailing_zeros() as usize,
        );
        let rows = ledger.map_or(0, |_| memlog::rows_bytes(trace));
        let laid = if let Some(records) = trace.records_main() {
            use std::sync::atomic::Ordering::Relaxed;
            records_tables.fetch_add(1, Relaxed);
            records_bytes.fetch_add(records.bytes(), Relaxed);
            table_of_records(air, trace, shape)
        } else if let Some(packed) = trace.narrow_main() {
            use std::sync::atomic::Ordering::Relaxed;
            narrow_tables.fetch_add(1, Relaxed);
            narrow_bytes.fetch_add(packed.data().len(), Relaxed);
            table_of_narrow(air, trace, shape)
        } else {
            table_of(air, trace, shape, true)
        }
        .and_then(|table| {
            let counts = local_counts_under(format.keccak_rnd, &table)?;
            Ok((i, shape, table, counts))
        });
        if let Ok(mut laid_at) = laid_at.lock() {
            laid_at.push((i, start.elapsed().as_secs_f64()));
        }
        if let (Some(ledger), Ok((_, _, table, _))) = (ledger, &laid) {
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
        fused_memory,
        word_memory,
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
        records_rest: (records_tables.into_inner(), records_bytes.into_inner()),
        laid_at: {
            let mut laid_at = laid_at.into_inner().unwrap_or_default();
            laid_at.sort_unstable_by_key(|&(i, _)| i);
            laid_at
        },
    })
}

/// The KECCAK_RND tables' own AIRs, for the streamed rest that lays them out
/// while the run's set is built beside them ([`rest_airs_beside`]): built on
/// first use as [`crate::block_airs`] builds that family, and checked against
/// the run's set once it exists.
#[derive(Clone, Copy)]
struct EarlyAirs<'a> {
    airs: &'a std::sync::OnceLock<Vec<crate::VmAir>>,
    /// A test's fault: each AIR named for the next table.
    misnamed: bool,
}

/// An AIR borrowed for `'a` (the run's set's, or a table's own).
type AirRef<'a> = &'a dyn stark::traits::AIR<Field = F, FieldExtension = E, PublicInputs = ()>;

/// KECCAK_RND's places in AIR order: its range among the chunked tables.
fn keccak_rnd_range(counts: &TableCounts) -> Result<std::ops::Range<usize>, Error> {
    chunked_table_ranges(counts)
        .into_iter()
        .find(|(name, ..)| *name == "KECCAK_RND")
        .map(|(_, range, _)| range)
        .ok_or_else(|| Error::Prover("no KECCAK_RND range among the chunked tables".into()))
}

/// Refuses an AIR a table was laid out with that is not the run's AIR at its
/// place: its name, shape, constraint count and interactions.
fn same_air(laid: AirRef<'_>, run: AirRef<'_>) -> Result<(), Error> {
    if laid.name() != run.name()
        || laid.trace_layout() != run.trace_layout()
        || laid.num_transition_constraints() != run.num_transition_constraints()
        || laid.bus_interactions() != run.bus_interactions()
    {
        return Err(Error::Prover(format!(
            "a table laid out with {}, the run's AIR at its place is {}",
            laid.name(),
            run.name()
        )));
    }
    Ok(())
}

/// The rest as the finish streams it ([`RestStream`]): each table laid out
/// narrow as it comes (wide ones through [`table_of`]) and sent to the packer
/// in AIR order (`sink`), every table checked against the run's AIRs built
/// from the header ([`RestOrder`]). The prepared columns are left for after
/// the packing, as on the pipelined path.
///
/// With `early` and KECCAK_RND handed on first, the run's AIR set is built on
/// a thread of its own while the KECCAK_RND tables are laid out with their
/// own AIRs ([`EarlyAirs`]) and sent on; the first table after them waits for
/// the set, which then admits the early tables in their order and checks
/// their AIRs against its own ([`same_air`]).
#[allow(clippy::too_many_arguments)]
fn lay_out_rest_streamed<'a>(
    stream: Box<RestStream>,
    program: &Elf,
    opts: &ProofOptions,
    run_airs: &'a std::sync::OnceLock<VmAirs>,
    early: Option<EarlyAirs<'a>>,
    start: Instant,
    sink: Option<std::sync::mpsc::Sender<RestDone<'a>>>,
    ledger: Option<&memlog::Ledger>,
    recorder: Option<&regen::Recorder>,
    keccak_rnd: crate::tables::keccak_rnd::KeccakRndForm,
    memw_aligned: crate::tables::memw_aligned::MemwAlignedOrder,
    registers: crate::tables::register::RegisterFormat,
    memory: crate::tables::fused_memory::MemoryFormat,
    ecdas: crate::tables::ecdas::EcdasForm,
    page_run_log2: u8,
    close_ahead: Option<std::sync::Arc<CloseAhead>>,
) -> Result<RestLaid<'a>, Error> {
    use std::sync::atomic::Ordering::Relaxed;
    let sink =
        sink.ok_or_else(|| Error::Prover("a streamed rest needs the packer's sink".into()))?;
    let at = || start.elapsed().as_secs_f64();
    let mut marks = vec![("rest received", at())];
    let RestStream { header, tables } = *stream;
    validate_block_counts(&header.table_counts)?;
    // The finish cut its page tables at the format's page runs.
    if header.page_run_log2 != page_run_log2 {
        return Err(Error::Prover(format!(
            "the finish cut its pages at runs of 2^{}, the format's are 2^{page_run_log2}",
            header.page_run_log2
        )));
    }
    let fused = header.fused_counts;
    let fused_memory = fused.map(|n| n as u64);
    fused_tables_under(memory, fused_memory)?;
    let words = header.word_counts;
    let word_memory = words.map(|n| n as u64);
    word_counts_under(memory, word_memory)?;
    let runtime_page_ranges = header.runtime_page_ranges();
    let num_private_input_pages = header
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();
    let build_airs = || -> Result<&'a VmAirs, Error> {
        let inputs = BlockAirInputs::new(
            opts,
            program,
            header.cpu_heights.clone(),
            &header.public_output_bytes,
        )?;
        let mut built = Ok(());
        let airs = run_airs.get_or_init(|| {
            let mut airs = block_airs(
                program,
                opts,
                page_air_pages(memory, &header.page_configs),
                &header.table_counts,
                &inputs,
                keccak_rnd,
                memw_aligned,
                registers,
                ecdas,
            );
            airs.with_fused_memory(opts, fused);
            if memory == crate::tables::fused_memory::MemoryFormat::Words {
                built = airs.with_word_memory_at(opts, words, &header.page_configs, page_run_log2);
            }
            airs
        });
        built?;
        check_table_count(airs.air_refs().len())?;
        Ok(airs)
    };
    let mut timed: Vec<(usize, f64)> = Vec::new();
    let mut laid_at: Vec<(usize, f64)> = Vec::new();
    let (mut narrow_tables, mut narrow_bytes) = (0usize, 0usize);
    let (mut records_tables, mut records_bytes) = (0usize, 0usize);
    let mut k = 0usize;
    // One table laid out with `air` and sent to the packer.
    let mut lay_one = |air: AirRef<'a>,
                       position: usize,
                       mut trace: TraceTable<F, E>,
                       marks: &mut Vec<(&'static str, f64)>| {
        if k == 0 {
            marks.push(("rest first received", at()));
        }
        let t = Instant::now();
        let shape = (
            trace.main_table.width,
            trace.main_table.height.trailing_zeros() as usize,
        );
        let rows = memlog::rows_bytes(&trace);
        let laid = if let Some(records) = trace.records_main() {
            // Never dropped, so never one phase B builds again.
            records_tables += 1;
            records_bytes += records.bytes();
            table_of_records(air, &mut trace, shape)
        } else if let Some(packed) = trace.narrow_main() {
            narrow_tables += 1;
            narrow_bytes += packed.data().len();
            // A table phase B can build again from the kept lists (N3): its
            // place among the rest's, before the packer places it.
            if let (Some(recorder), Some((family, j))) = (
                recorder,
                crate::tables::trace_builder::RegenFamily::of_air(air.name()),
            ) {
                recorder.rest_tag(k, family, j);
            }
            table_of_narrow(air, &mut trace, shape)
        } else {
            table_of(air, &mut trace, shape, true)
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
        let laid = laid.and_then(|table| {
            let counts = local_counts_under(keccak_rnd, &table)?;
            Ok((position, shape, table, counts))
        });
        let _ = sink.send((k, secs, laid));
        k += 1;
    };
    let mut tables = tables.into_iter();
    // The tables laid out before the AIR set was there, by position and
    // width (`None`: not built here), with the AIR each was laid out with;
    // the first table after them, held for the set.
    let mut before: Vec<(usize, Option<usize>, Option<AirRef<'a>>)> = Vec::new();
    let mut held = None;
    let airs = match early.filter(|_| header.keccak_rnd_first) {
        Some(early) => {
            let range = keccak_rnd_range(&header.table_counts)?;
            let own = early.airs.get_or_init(|| {
                (0..range.len())
                    .map(|i| {
                        let i = if early.misnamed { i + 1 } else { i };
                        Box::new(
                            crate::test_utils::create_keccak_rnd_air_for(opts, keccak_rnd)
                                .with_name(&format!("KECCAK_RND[{i}]")),
                        ) as crate::VmAir
                    })
                    .collect()
            });
            marks.push(("rest airs beside", at()));
            let (airs, ready) = std::thread::scope(|scope| -> Result<_, Error> {
                let build = std::thread::Builder::new()
                    .name("rest-airs".into())
                    .spawn_scoped(scope, build_airs)
                    .map_err(|e| Error::Prover(format!("the AIR set's thread: {e}")))?;
                // The early tables laid out while the set was still building.
                let mut ready = 0usize;
                while before.len() < range.len() {
                    let Some(emitted) = tables.next() else {
                        break;
                    };
                    let crate::tables::trace_builder::Emitted { position, table } = emitted;
                    if !range.contains(&position) {
                        held = Some(crate::tables::trace_builder::Emitted { position, table });
                        break;
                    }
                    let Some(trace) = table else {
                        before.push((position, None, None));
                        continue;
                    };
                    let air: AirRef<'a> = own[position - range.start].as_ref();
                    before.push((position, Some(trace.main_table.width), Some(air)));
                    if !build.is_finished() {
                        ready += 1;
                    }
                    lay_one(air, position, trace, &mut marks);
                }
                let airs = build
                    .join()
                    .map_err(|_| Error::Prover("the AIR set's thread panicked".into()))??;
                Ok((airs, ready))
            })?;
            eprintln!(
                "BLOCK REST AIRS: beside · {} KECCAK_RND table(s) laid out before the rest's \
                 next table, {ready} of them while the AIR set was building",
                before.len()
            );
            airs
        }
        None => build_airs()?,
    };
    marks.push(("airs", at()));
    let refs = airs.air_refs();
    let mut first = if header.keccak_rnd_first {
        let names: Vec<&str> = refs.iter().map(|air| air.name()).collect();
        keccak_rnd_positions(&header.table_counts, &names)?
    } else {
        Vec::new()
    };
    let keccak_rnd_due = first.len();
    if header.pages_first {
        first.extend(page_positions(airs)?);
    }
    let mut order = RestOrder::with_first(
        refs.iter().map(|air| air.trace_layout().0).collect(),
        &first,
        header.bitwise_last,
    );
    if header.bitwise_last && refs.first().map(|air| air.name()) != Some("BITWISE") {
        return Err(Error::Prover(
            "the rest's BITWISE goes last, but BITWISE is not the rest's first table".into(),
        ));
    }
    if header.keccak_rnd_first {
        eprintln!(
            "BLOCK REST ORDER: KECCAK_RND first · {} table(s) at [{}]",
            keccak_rnd_due,
            first[..keccak_rnd_due]
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    if header.pages_first {
        let pages = &first[keccak_rnd_due..];
        eprintln!(
            "BLOCK REST ORDER: the page tables next · {} table(s) at {}..{}",
            pages.len(),
            pages.first().copied().unwrap_or(0),
            pages.last().map_or(0, |p| p + 1)
        );
    }
    // The tables laid out before the set: admitted in their order, each AIR
    // the run's own.
    for &(position, width, air) in &before {
        order.admit(position, width)?;
        if let Some(air) = air {
            let run = refs
                .get(position)
                .ok_or_else(|| Error::Prover(format!("table {position} is past the AIR set")))?;
            same_air(air, *run)?;
        }
    }
    // The rest's messages sent so far (each table laid out is one).
    let mut sent = before
        .iter()
        .filter(|(_, width, _)| width.is_some())
        .count();
    // BITWISE's shape, a fixed table's: its AIR's columns, 2^20 rows.
    let bitwise_shape = (
        refs.first().map_or(0, |air| air.trace_layout().0),
        crate::tables::bitwise::NUM_ROWS.trailing_zeros() as usize,
    );
    for crate::tables::trace_builder::Emitted { position, table } in held.into_iter().chain(tables)
    {
        order.admit(position, table.as_ref().map(|t| t.main_table.width))?;
        let Some(trace) = table else {
            continue;
        };
        // BITWISE handed last and the only table left once this one is laid
        // out: the packer is told before this table reaches it.
        if let Some(ahead) = &close_ahead
            && header.bitwise_last
            && position != 0
            && order.only_due(0)
        {
            let _ = ahead.last.set((sent, bitwise_shape));
        }
        lay_one(refs[position], position, trace, &mut marks);
        sent += 1;
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
        fused_memory,
        word_memory,
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
        records_rest: (records_tables, records_bytes),
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
    /// [`BlockWhirProof::fused_memory`].
    fused_memory: [u64; 2],
    /// [`BlockWhirProof::word_memory`].
    word_memory: [u64; 7],
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
    lead: Option<&PhaseBLead>,
    stamps: &mut BlockStamps,
) -> Result<BlockWhirProof<LocalArgues>, Error> {
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
    // The tables filled from their records over the whole prove: phase A's
    // commits and phase B's arguments alike.
    let fills_before = multilinear::records::fills();
    let times_before = multilinear::records::upload_times();
    let stream_airs = StreamAirs::new(
        opts,
        stream_form,
        options.max_rows.cpu,
        format.keccak_rnd,
        format.memw_aligned,
        format.registers,
    );
    // Regeneration (`BlockOptions::regen`): its recorder, when phase A can feed
    // it — its chunks generated packed. The disk store's knob, which used to
    // turn it off, refuses the prove when set.
    regen::refuse_removed_spill_words(|name| std::env::var(name).ok())?;
    // A switch that is gone refuses the prove when set, rather than being ignored.
    refuse_retired_words(|name| std::env::var(name).ok())?;
    // The constraint-sumcheck probes: measurements, off unless asked (the
    // proof is the same); a word they do not take refuses the prove.
    let cs_probe = multilinear_block::CsProbe::from_env().map_err(Error::Prover)?;
    // Under local lookups, whether phase A's lane runs the local argues: the
    // proof is the same either way; a word it does not take refuses the prove.
    let local_lane = match deviations.local_lane {
        Some(lane) => lane,
        None => local_lane_setting(std::env::var(LOCAL_LANE_ENV).ok().as_deref())
            .map_err(Error::Prover)?,
    };
    // Where the in-row side records are walked into: a word it does not take
    // refuses the prove.
    let sides_pinned = crate::tables::trace_builder::inrow_sides_pinned_setting(
        std::env::var("LAMBDA_VM_INROW_SIDES_PINNED")
            .ok()
            .as_deref(),
    )
    .map_err(Error::Prover)?;
    if format.registers == crate::tables::register::RegisterFormat::InRow {
        eprintln!(
            "★ BLOCK IN-ROW SIDES: {} (LAMBDA_VM_INROW_SIDES_PINNED)",
            if sides_pinned {
                "page-locked (the default)"
            } else {
                "heap (the rollback)"
            }
        );
    }
    // Under the word memory format, the word families streamed in chunks as
    // the walk fills them, or built at the finish: a word it does not take
    // refuses the prove. The tables are the same either way.
    let word_stream = format.memory == crate::tables::fused_memory::MemoryFormat::Words
        && match deviations.word_stream {
            Some(on) => on,
            None => parse_word_stream(std::env::var("LAMBDA_VM_WORD_STREAM").ok().as_deref())?,
        };
    if format.memory == crate::tables::fused_memory::MemoryFormat::Words {
        eprintln!(
            "★ BLOCK WORD STREAM: {} (LAMBDA_VM_WORD_STREAM)",
            if word_stream {
                "the word families cut into chunks as the walk fills them, the default"
            } else {
                "off: the finish builds the word families"
            }
        );
    }
    // The walkers' light walk: a word it does not take refuses the prove.
    let light_walk = parse_light_walk(std::env::var("LAMBDA_VM_LIGHT_WALK").ok().as_deref())?;
    eprintln!(
        "★ BLOCK LIGHT WALK: level {light_walk} ({}) (LAMBDA_VM_LIGHT_WALK)",
        match light_walk {
            0 => "the walk before it, the rollback",
            1 => "record mode's cycles walked without their cells",
            _ =>
                "record mode's cycles walked without their cells, each window routed by its walker, the default",
        }
    );
    // A card-rows word the uploads do not take refuses the prove, rather than
    // running as another mode.
    multilinear::gpu::card_rows_refusal().map_err(Error::Prover)?;
    let regen_mode = match options.regen {
        Some(mode) => mode,
        None => regen::regen_mode()?,
    };
    // A streamed chunk held as its records parks and drops like any chunk,
    // unless records are kept: then it has no recipe and no rank.
    // Live regeneration parks and drops a streamed chunk held as its records
    // like any chunk, phase B rebuilding its packed columns checked by its
    // records' digest; a test may keep such chunks on the host instead.
    let keep_records = deviations.records_keep.unwrap_or(false);
    let regen_recorder = regen::recorder(regen_mode, stream_form == TraceForm::Narrow)
        .map(|recorder| {
            if keep_records {
                recorder.keep_records()
            } else {
                recorder
            }
        })
        .map(std::sync::Arc::new);
    let recorder = regen_recorder.as_deref();
    // Live regeneration's recorder: the rest's tables it can build again are
    // tagged as they are laid out (N3).
    let live_recorder = recorder.filter(|_| regen_mode.live());
    // Under local lookups: where phase A's LOCAL_BYTES tables sit, for live
    // regeneration to name each table it drops by its block index.
    let local_places = LocalPlaces::default();
    // Live regeneration: phase A drops the streamed chunks the policy moves off
    // the host (`always`: every one). The streamed chunks are placed first, in
    // hand-out order, so the table at block index `t` is the `t`-th chunk
    // handed out when there was one.
    let keep_cells = block_keep_cells()?;
    let keep_below = block_keep_below()?;
    let live_regen = regen_recorder
        .as_ref()
        .filter(|_| regen_mode.live())
        .map(|recorder| {
            let recorder = std::sync::Arc::clone(recorder);
            let local_places = local_places.clone();
            // Class 0: the streamed chunks (the replay); class 1: KECCAK_RND's
            // and LT's tables of the rest (the kept lists, N3).
            multilinear_block::BlockRegen::new(
                // Phase A names a table by its place in its own order of
                // tables (`p`); under local lookups each LOCAL_BYTES table sent
                // before it moves that place one past the table's block index
                // `t`, and a LOCAL_BYTES table itself is never dropped
                // ([`LocalPlaces`]). A streamed chunk's rank is its hand-out
                // place, which is its block index unless phase A commits out of
                // hand-out order (KECCAK_RND's own groups:
                // [`Recorder::handout_place_of`], which reads phase A's places).
                std::sync::Arc::new(move |p| {
                    let t = local_places.block_index_of(p)?;
                    recorder
                        .handout_place_of(p, t)
                        .and_then(|r| recorder.chunk_at(r).map(|_| (0, r as u64)))
                        .or_else(|| recorder.rest_at(t).map(|_| (1, t as u64)))
                }),
                &match deviations.regen_windows {
                    Some((streamed, rest)) => [streamed, rest],
                    None => [regen::ahead_policy().initial(), regen::rest_ahead_bytes()],
                },
                regen_mode == regen::RegenMode::Always,
            )
        })
        // Once armed, it drops each table phase B can rebuild only while the
        // policy wants it out, on what the host holds now, and purges the
        // dropped tables' pages so the reading falls
        // ([`multilinear_block::BlockRegen::partial`], [`DropPurges`]).
        .map(|regen| {
            regen
                .partial()
                .keep_cells(keep_cells)
                .keep_below(keep_below)
        });
    if let (Some(regen), Some(pause)) = (&live_regen, deviations.streamed_drop_delay) {
        regen.delay_drops(0, pause);
    }
    // The streamed chunks' regenerator started before phase B, when it is
    // (N4c); phase B takes it, and an error before then closes its window and
    // joins.
    let regen_lead = deviations.regen_lead.unwrap_or_else(regen::regen_lead);
    let early_owned: std::sync::Mutex<Option<regen::EarlyStart>> = std::sync::Mutex::new(None);
    let early_slot = &early_owned;
    let (early_feed, early_recorder) = (live_regen.clone(), regen_recorder.clone());
    // Opened when phase B starts: until then an early start's windows hold
    // only what the host's room allows (N4c).
    let early_gate = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let early_gate_in = std::sync::Arc::clone(&early_gate);
    // Set at the finish's end: until then an early start's window holds only
    // its frontier, so the regenerator does not rebuild beside the finish.
    let early_paced = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let run_airs: std::sync::OnceLock<VmAirs> = std::sync::OnceLock::new();
    // The KECCAK_RND tables' AIRs the streamed rest lays them out with while
    // the run's set is built beside them ([`rest_airs_beside`]).
    let rest_keccak_airs: std::sync::OnceLock<Vec<crate::VmAir>> = std::sync::OnceLock::new();
    // Phase A's commits read the blowup, the fold schedule and the format of
    // the config and nothing else; the full config (its query count reads every
    // shape) is built when the shapes are known, and must agree on those.
    let commit_config = format.chain_config(&[]);
    let cap = commit_config.format.stack;
    let start = Instant::now();
    // A measurement ([`KECCAK_RND_ARRIVALS_ENV`]): when each KECCAK_RND table's
    // operations are all in, as the windows are taken.
    let arrivals =
        parse_keccak_rnd_arrivals(std::env::var(KECCAK_RND_ARRIVALS_ENV).ok().as_deref())?;
    if arrivals {
        crate::tables::trace_builder::keccak_rnd_arrivals::arm(1 << options.keccak_rnd_rows_log2);
    }
    head_log::begin(start, deviations.head_log.map_or_else(head_log::knob, Ok)?);
    set_up_card();
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
    // Live regeneration decides at each committed table whether it leaves the
    // host: the `auto` rule against the host's memory target, and a table it
    // would move off the host is dropped if phase B can rebuild it, else
    // stays. Without live regeneration every table is held.
    eprintln!(
        "{}",
        regen::regen_mode_line(
            regen_mode,
            live_regen.is_some(),
            std::env::var("LAMBDA_VM_BLOCK_REGEN").ok().as_deref(),
        )
    );
    let target = deviations.memory_target.unwrap_or_else(memory_target_bytes);
    // The largest host reading the rule decided on, for the BLOCK DROPS line
    // (#1013's `Spill::host_most`).
    let host_most: std::sync::Arc<std::sync::Mutex<Option<HostReading>>> =
        std::sync::Arc::new(std::sync::Mutex::new(None));
    // And what the host really held, with the decisions the real-memory guard
    // alone made ([`auto_wants`]).
    let real_guard = std::sync::Arc::new(RealGuard::default());
    let hand_off_guard = std::sync::Arc::clone(&real_guard);
    // With live regeneration the allocator's two boundary purges wait for the
    // resident set to come within the regeneration window of the pressure line.
    let room = no_disk_room(live_regen.is_some(), target);
    crate::alloc_purge::note_no_disk(room);
    // A tree-drop value the options could not read refuses the prove.
    tree_drop_refusal()?;
    // The host is read at each decision: the `auto` rule decides on it and the
    // purge's pressure is noted from it (`note_pressure_past`).
    // Once live regeneration has armed, its phase-B reserve counts. Partial
    // drops, once armed, weigh what the host holds now, and purge what they
    // dropped.
    let drop_purges = live_regen
        .is_some()
        .then(|| DropPurges::new(crate::alloc_purge::purge_now));
    let wanted: std::sync::Arc<multilinear_block::SpillWanted> = {
        let host_most = std::sync::Arc::clone(&host_most);
        let real_guard = std::sync::Arc::clone(&real_guard);
        let armed = live_regen.clone();
        let purges = drop_purges.clone();
        let reserve = regen::regen_reserve_bytes();
        let per_g = memory_reserve_per_g();
        std::sync::Arc::new(move |_kept, cells, bytes| {
            let reading = HostReading::now();
            {
                let mut most = host_most.lock().unwrap_or_else(|e| e.into_inner());
                if most.is_none_or(|m| reading.bytes() > m.bytes()) {
                    *most = Some(reading);
                }
            }
            note_pressure_past(reading.real_bytes(), target);
            let is_armed = armed.as_ref().is_some_and(|r| r.is_armed());
            let extra = if is_armed { reserve } else { 0 };
            let now = purges.is_some() && is_armed;
            let allocator = now
                .then(crate::alloc_purge::stats)
                .flatten()
                .map(|s| s.resident as u64);
            let host = decision_bytes(&reading, allocator, now);
            let (rule, guard) = auto_wants(&reading, host, now, extra, target, per_g, cells, bytes);
            real_guard.note(reading.real_bytes(), guard && !rule);
            let out = rule || guard;
            if out && let Some(purges) = &purges {
                purges.dropped(bytes);
            }
            out
        })
    };
    let spill = live_regen
        .as_ref()
        .map(|regen| multilinear_block::BlockSpill::new(wanted).with_regen(regen.clone()));

    // The streamed finish's byte gate: the rest's tables generated and not yet
    // placed ([`BlockOptions::finish_stream`]).
    let rest_gate = crate::tables::trace_builder::gate::ByteGate::new(
        options.rest_layout_bytes.unwrap_or(usize::MAX),
    )
    .with_build(options.rest_build_bytes);
    let rest_gate = &rest_gate;
    let waits = ProducerWaits::default();
    let waits = &waits;
    let tag = block_format_tag(format);
    crate::with_base_hash!(format.zf.base, |H| {
        let walkers = block_walkers()?;
        let bitwise_last = deviations.rest_bitwise_last;
        let keccak_rnd_first = keccak_rnd_first_or_default(deviations.keccak_rnd_first);
        let pages_first = pages_first(deviations.pages_first)?;
        eprintln!(
            "★ BLOCK COMMIT ENCODES: {} ({})",
            if multilinear::gpu::serial_encodes() {
                "a group's round-0 encodes one at a time (the default)"
            } else {
                "a group's round-0 encodes side by side"
            },
            multilinear::gpu::SERIAL_ENCODES_ENV
        );
        eprintln!(
            "★ BLOCK COMMIT LEAVES: {} ({}; what phase A streamed: BLOCK COMMIT LEAVES STREAMED)",
            if multilinear::gpu::streamed_joint_leaves() {
                "a joint run's codewords streamed into its leaves one at a time (the default)"
            } else {
                "a joint run's codewords all encoded, then its leaves"
            },
            multilinear::gpu::STREAMED_LEAVES_ENV
        );
        eprintln!(
            "★ BLOCK PAGES FIRST: {} ({PAGES_FIRST_ENV})",
            if pages_first {
                "the page tables handed on right after KECCAK_RND's (the default)"
            } else {
                "AIR order after KECCAK_RND"
            }
        );
        let airs_beside = rest_airs_beside(deviations.rest_airs_beside)?;
        let close_ahead = close_before_bitwise(deviations.close_before_bitwise)?
            .then(|| std::sync::Arc::new(CloseAhead::default()));
        eprintln!(
            "★ BLOCK CLOSE BEFORE BITWISE: {} ({CLOSE_BEFORE_BITWISE_ENV})",
            if close_ahead.is_some() {
                "on (the open group closes as soon as BITWISE alone is left and would not fit it)"
            } else {
                "off (the default; at BITWISE's arrival)"
            }
        );
        #[cfg(test)]
        let misnamed = deviations.rest_airs_beside_misnamed;
        #[cfg(not(test))]
        let misnamed = false;
        eprintln!(
            "★ BLOCK REST AIRS: {} ({REST_AIRS_BESIDE_ENV})",
            if airs_beside {
                "built beside the first KECCAK_RND tables (the default)"
            } else {
                "built before the rest's first table"
            }
        );
        let from_tables = bitwise_from_tables_or_default(deviations.bitwise_from_tables);
        let mut card_fill = match (
            deviations.card_fill,
            deviations.card_fill_streamed,
            deviations.card_fill_rows,
        ) {
            (Some(keccak_rnd), Some([cpu, memw_r]), rows) => CardFill {
                keccak_rnd,
                cpu,
                memw_r,
                rows: rows.unwrap_or(RowFamilies::NONE),
            },
            (keccak_rnd, streamed, rows) => {
                let knob = block_card_fill()?;
                let [cpu, memw_r] = streamed.unwrap_or([knob.cpu, knob.memw_r]);
                CardFill {
                    keccak_rnd: keccak_rnd.unwrap_or(knob.keccak_rnd),
                    cpu,
                    memw_r,
                    rows: rows.unwrap_or(knob.rows),
                }
            }
        };
        // KECCAK_RND handed out during the windows: by default where the build
        // cuts it into its tables and the card fills it from its records (as
        // the finish holds them: packed), so the streamed tables take no host
        // layout; built at the finish otherwise; or as asked.
        let can_stream = options.finish_keccak_rnd_chunks
            && options.keccak_rnd_rows_log2 >= 5
            && !deviations.omit_first_keccak_rnd;
        let stream_keccak_rnd = match stream_keccak_rnd_records(deviations.stream_keccak_rnd)? {
            None => can_stream && card_fill.keccak_rnd && options.pack_finished,
            Some(false) => false,
            Some(true) if can_stream => true,
            Some(true) => {
                return Err(Error::Prover(format!(
                    "{STREAM_KECCAK_RND_ENV}=1 needs KECCAK_RND cut into its tables \
                     (finish_keccak_rnd_chunks, at least 2^5 rows) and no first table \
                     omitted (a streamed table lands by its index)"
                )));
            }
        };
        eprintln!(
            "★ BLOCK KECCAK_RND STREAM: {} ({STREAM_KECCAK_RND_ENV})",
            if stream_keccak_rnd {
                "each table handed out during the windows once its operations are in"
            } else {
                "built at the finish"
            }
        );
        let own_groups =
            stream_keccak_rnd && own_keccak_rnd_groups(deviations.keccak_rnd_own_groups)?;
        if stream_keccak_rnd {
            eprintln!(
                "★ BLOCK KECCAK_RND GROUPS: {} ({KECCAK_RND_OWN_GROUPS_ENV})",
                if own_groups {
                    "the streamed tables packed in groups of their own (the default)"
                } else {
                    "the streamed tables packed among the other chunks"
                }
            );
        }
        if let (true, Some(recorder)) = (own_groups, recorder) {
            recorder.commits_out_of_order();
        }
        // Under the in-row register format MEMW_R's slot holds the residual
        // register table, which the host builds (MEMW_R's kernel fills MEMW_R's
        // columns, never the residual table's): MEMW_R's records, its words and
        // what the card derives from them are off. The CPU chunks go up with
        // their side records as the card fills them.
        if format.registers == crate::tables::register::RegisterFormat::InRow {
            card_fill.memw_r = false;
        }
        // The streamed chunks held as records ([`records_admission`]):
        // G-pack's narrow chunks only, from a build that drops its streamed
        // ops, with live regeneration to drop them (or kept on purpose),
        // admitted while the host has room for them by the purge rule's own
        // test (its resident set, the chunk and the regeneration window's
        // ceiling under the pressure line) and live regeneration has not
        // armed. Once a chunk is refused, every later one is filled on the
        // host: what is never dropped is a prefix admitted with room.
        let admission = records_admission(
            card_fill.streamed(),
            options.drop_streamed_ops && stream_form == TraceForm::Narrow,
            keep_records,
            live_regen.is_some(),
        );
        let records_room = admission.is_ok().then(|| match &deviations.records_room {
            Some(room) => std::sync::Arc::clone(room),
            None => {
                let line = no_disk_room(true, target);
                let armed = live_regen.clone();
                std::sync::Arc::new(move |bytes: usize| {
                    !armed.as_ref().is_some_and(|regen| regen.is_armed())
                        && crate::alloc_purge::has_room(
                            line,
                            crate::lfm::program_budget::resident_bytes()
                                .map(|resident| resident.saturating_add(bytes as u64)),
                        )
                }) as crate::tables::trace_builder::RecordsRoom
            }
        });
        eprintln!(
            "BLOCK CARD FILL: families {} · streamed chunks as records: {}",
            card_fill.name(),
            admission.err().unwrap_or("while the host has room")
        );
        // Record mode: the fast engine writes the walk's history, the walkers
        // read it.
        let (kind, history) = exec_mode(walkers)?;
        let history_check = exec_history_check()?;
        let (block, built, laid, executed) = std::thread::scope(|scope| {
            use std::sync::atomic::Ordering::Relaxed;
            // The executor, a window at a time, two windows ahead of the walk;
            // with walkers, each window with the memory at its start.
            // The logs shared between the walk and the card fill.
            let (ltx, lrx) = std::sync::mpsc::sync_channel::<parallel::ExecWindow>(2);
            let executor = scope.spawn(move || -> Result<(f64, ExecutorStamps), Error> {
                if let Some(ledger) = ledger {
                    ledger.thread("executor");
                }
                let mut own = ExecutorStamps {
                    started: start.elapsed().as_secs_f64(),
                    ..ExecutorStamps::default()
                };
                let t = Instant::now();
                let failed =
                    |e: executor::vm::execution::ExecutorError| Error::Execution(format!("{e}"));
                // The interpreter ([`exec_mode`]), and on the fast one the
                // sampled re-execution by the reference
                // (`LAMBDA_VM_EXEC_SHADOW`).
                let mut executor =
                    Engine::new(kind, program, private_inputs.to_vec()).map_err(failed)?;
                let shadow_every = match kind {
                    EngineKind::Fast => ShadowCheck::every_from_env().map_err(Error::Execution)?,
                    EngineKind::Reference => None,
                };
                let mut shadow = shadow_every
                    .map(|every| ShadowCheck::new(program, every))
                    .transpose()
                    .map_err(failed)?;
                // Under the word memory format the history keeps a timestamp
                // an aligned word: what the word walk reads.
                let word_history =
                    format.memory == crate::tables::fused_memory::MemoryFormat::Words;
                if history && word_history {
                    executor.enable_word_history().map_err(failed)?;
                } else if history {
                    executor.enable_history().map_err(failed)?;
                }
                own.setup = t.elapsed().as_secs_f64();
                eprintln!(
                    "BLOCK EXECUTOR ENGINE: {} · shadow check {} · log buffers fresh · history {}",
                    kind.name(),
                    shadow_every.map_or("off".to_string(), |n| format!("every {n} windows")),
                    if history {
                        format!("written (re-walk every {history_check} windows, 0 none)")
                    } else {
                        "off".to_string()
                    },
                );
                // Window-start snapshots for the walkers; none in record mode.
                let snapshots = walkers > 0 && !history;
                let mut at_start = snapshots.then(|| executor.snapshot());
                let mut cycles = 0u64;
                // The window's buffer capacity, and whether each next one
                // comes from the page-locked pool (`LAMBDA_VM_POOL_EXEC_LOGS`).
                let log_capacity = executor::vm::fast::log_window_capacity(window);
                own.logs_pooled = pool_exec_logs()
                    && kind == EngineKind::Fast
                    && multilinear::pinned::ready();
                loop {
                    let t = Instant::now();
                    if own.logs_pooled {
                        // Taken when this window is handed out, as the next
                        // window's buffer; a buffer of another capacity is
                        // never taken (and freed).
                        executor.recycle(pooled_log_buffer(log_capacity));
                    }
                    // A sampled window's start, for the shadow check.
                    let sampled = match &shadow {
                        Some(check) if check.samples(own.windows) => Some(WindowStart {
                            window: own.windows,
                            first_cycle: cycles,
                            pc: executor.pc(),
                            registers: executor.registers(),
                            memory: match &at_start {
                                Some(snapshot) => snapshot.clone(),
                                None => executor.snapshot(),
                            },
                            public_output: executor.public_output().to_vec(),
                        }),
                        _ => None,
                    };
                    // Record mode's check: a re-walked window's start memory.
                    let check_start = (history
                        && history_check > 0
                        && own.windows.is_multiple_of(history_check))
                    .then(|| executor.snapshot());
                    // `ran`: the window executed.
                    let (logs, ran) = if history {
                        let window = if word_history {
                            executor.resume_word_history(window).map_err(failed)?
                        } else {
                            executor.resume_history(window).map_err(failed)?
                        };
                        (window.map(|(logs, h)| (logs, Some(h))), Instant::now())
                    } else {
                        let logs = executor.resume_owned(window).map_err(failed)?;
                        (logs.map(|logs| (logs, None)), Instant::now())
                    };
                    own.run += (ran - t).as_secs_f64();
                    let Some((logs, window_history)) = logs else {
                        break;
                    };
                    cycles += logs.len() as u64;
                    own.note_logs(&logs, logs.capacity());
                    if let Some(check) = &mut shadow {
                        if let Some(start) = sampled {
                            check.submit(start, &logs, executor.pc(), executor.registers());
                        }
                        if let Some(mismatch) = check.failure() {
                            return Err(Error::Execution(mismatch));
                        }
                    }
                    let next = snapshots.then(|| executor.snapshot());
                    let start = std::mem::replace(&mut at_start, next);
                    // Record mode's last window carries the run's end.
                    let end = (history && executor.pc() == 0)
                        .then(|| {
                            if word_history {
                                executor
                                    .take_word_history_final()
                                    .map(parallel::RunEnd::boxed_words)
                            } else {
                                executor.take_history_final().map(parallel::RunEnd::boxed)
                            }
                        })
                        .flatten();
                    let logs = std::sync::Arc::new(logs);
                    own.windows += 1;
                    own.handoff += ran.elapsed().as_secs_f64();
                    if let Some(ledger) = ledger {
                        // The executor keeps a window of logs of its own.
                        let bytes = memlog::logs_bytes(&logs);
                        ledger.logs.fetch_add(bytes, Relaxed);
                        ledger
                            .exec
                            .store(executor.heap_bytes() + bytes, Relaxed);
                    }
                    let t = Instant::now();
                    let sent_window = parallel::ExecWindow {
                        logs,
                        start: if history { check_start } else { start },
                        history: window_history.map(Box::new),
                        last: end,
                    };
                    if ltx.send(sent_window).is_err() {
                        break;
                    }
                    ProducerWaits::add(&waits.executor_send, t);
                }
                let executed = start.elapsed().as_secs_f64();
                drop(executor);
                if let Some(check) = shadow {
                    let counts = check.finish().map_err(Error::Execution)?;
                    eprintln!(
                        "BLOCK EXECUTOR SHADOW: {} windows re-executed and equal · {} skipped (the check busy)",
                        counts.checked, counts.skipped
                    );
                }
                if let Some(ledger) = ledger {
                    ledger.exec.store(0, Relaxed);
                    ledger.line("executor done");
                }
                Ok((executed, own))
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
                if stream_keccak_rnd {
                    // Checked above: KECCAK_RND is cut into its tables.
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
                        1usize << options.copy_rows_log2,
                    )?;
                }
                if options.pack_finished {
                    builder = builder.pack_finished_tables();
                    if options.gpack {
                        builder = builder.generate_packed();
                    }
                }
                if let Some(on) = deviations.handout_count_thread {
                    builder = builder.count_handout_on_thread(on);
                }
                if options.drop_streamed_ops {
                    builder = builder.drop_streamed_ops()?;
                }
                // LT's memory comparisons merged with their neighbours instead
                // of the map's rows ([`WindowedTraceBuilder::lt_merge`]).
                builder = builder.lt_merge();
                builder = builder.p4_rows_on_pool().p4_lookups_direct();
                if bitwise_last {
                    builder = builder.rest_bitwise_last();
                }
                if from_tables {
                    builder = builder.bitwise_from_tables();
                }
                if keccak_rnd_first {
                    builder = builder.keccak_rnd_first();
                }
                if pages_first {
                    builder = builder.pages_first();
                }
                builder = builder.slide_rest();
                builder = builder.keccak_rnd_form(format.keccak_rnd);
                builder = builder.ecdas_form(format.ecdas);
                builder = builder.memw_aligned_order(format.memw_aligned)?;
                builder = builder.register_format(format.registers)?;
                builder = builder.memory_format(format.memory)?;
                builder = builder.page_run_log2(format.page_runs())?;
                if word_stream {
                    builder = builder.stream_word_tables()?;
                }
                if card_fill.keccak_rnd {
                    builder = builder.card_fill_keccak_rnd();
                    // The finish holds its KECCAK_RND tables as records only
                    // when it packs them; the streamed ones follow it.
                    if stream_keccak_rnd && options.pack_finished {
                        builder = builder.stream_keccak_rnd_records()?;
                    }
                    #[cfg(test)]
                    if deviations.card_fill_records_fault > 0 {
                        builder =
                            builder.card_fill_records_fault(deviations.card_fill_records_fault);
                    }
                }
                if let Some(room) = records_room.clone() {
                    builder = builder.card_fill_streamed(room, card_fill.cpu, card_fill.memw_r)?;
                    if card_fill.rows.any() {
                        builder = builder.card_fill_streamed_rows(card_fill.rows)?;
                    }
                    // `LAMBDA_VM_CARD_ROWS=words`, the default: the walkers write
                    // MEMW_R's rows as words the card builds them from. Needs record
                    // mode (the ECALL cycles from the history) and the CPU logs kept;
                    // without them MEMW_R goes up as rows (LOAD and STORE are still
                    // derived).
                    if multilinear::gpu::card_rows_builds_words() {
                        let words = card_fill.memw_r && history && walkers > 0 && builder.keeps_logs();
                        if words {
                            builder = builder.card_writes_words()?;
                        }
                        eprintln!(
                            "BLOCK CARD ROWS WORDS: {}",
                            if words {
                                "the walkers write MEMW_R's rows as words"
                            } else {
                                "off (no record mode, MEMW_R not held as records, or no CPU logs kept): MEMW_R goes up as rows"
                            }
                        );
                    }
                    #[cfg(test)]
                    if deviations.card_fill_streamed_fault > 0 {
                        builder =
                            builder.card_fill_streamed_fault(deviations.card_fill_streamed_fault);
                    }
                }
                builder = builder.light_walk(light_walk)?;
                if let Some(ledger) = ledger {
                    ledger.image.store(builder.image_bytes(), Relaxed);
                    ledger.line("builder created");
                }
                let setup = builder.setup_stamps();
                eprintln!(
                    "BLOCK BUILDER: ready at {:.2}s · image and memory state {:.2}s · decode table {:.2}s · DECODE's trace generated on a thread of its own",
                    start.elapsed().as_secs_f64(),
                    setup.image,
                    setup.decode,
                );
                let mut streamed = 0usize;
                // The walk on its own thread, doing nothing but walk; this
                // thread appends each walked window, routes it and hands its
                // chunks out as jobs (the layout thread generates them).
                let last = if walkers > 0 {
                    // The windows walked from their entries on `walkers`
                    // threads, absorbed here in order as below.
                    // This thread's wait for the next window in order: from
                    // the walk's start to the first window, then between
                    // windows (the `accumulator recv-wait` term).
                    let mut idle_from = Instant::now();
                    let report = parallel::walk_windows_with(
                        &mut builder,
                        lrx,
                        walkers,
                        history.then_some(history_check),
                        |accumulator, walked, states| {
                            ProducerWaits::add(&waits.accumulator_recv, idle_from);
                            // The walked windows wait in the walk's reorder
                            // buffer, not on a channel: no `walked` term.
                            if let Some(ledger) = ledger {
                                ledger.walk.store(states, Relaxed);
                                ledger.logs.fetch_sub(
                                    walked.cycles() * size_of::<executor::vm::logs::Log>(),
                                    Relaxed,
                                );
                            }
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
                                head_log::chunk_handed(
                                    streamed - 1,
                                    stream_base(job.table),
                                    job.index,
                                    job.op_count(),
                                );
                                let t = Instant::now();
                                if btx.send(Built::Job(Box::new(job))).is_err() {
                                    return Err(Error::Prover("the layout thread stopped".into()));
                                }
                                ProducerWaits::add(&waits.accumulator_send, t);
                            }
                            if let Some(ledger) = ledger {
                                ledger.builder.store(accumulator.held_bytes(), Relaxed);
                            }
                            idle_from = Instant::now();
                            Ok(())
                        },
                    )?;
                    if let Some(ledger) = ledger {
                        // The window states are gone; the walk's state is the
                        // finish's, as on one walker.
                        ledger.walk.store(builder.state_bytes(), Relaxed);
                    }
                    eprintln!("{}", report.line());
                    report.last
                } else {
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
                            let mut held: Option<std::sync::Arc<Vec<executor::vm::logs::Log>>> =
                                None;
                            loop {
                                let t = Instant::now();
                                let Ok(parallel::ExecWindow { logs, .. }) = lrx.recv() else {
                                    break;
                                };
                                ProducerWaits::add(&waits.walker_recv, t);
                                if let Some(w) = held.replace(logs) {
                                    let bytes = memlog::logs_bytes(&w);
                                    // The logs go with the window when the
                                    // card fills its CPU chunks from them.
                                    let walked = walker.walk_owned(w)?;
                                    if let Some(ledger) = ledger {
                                        ledger.walk.store(walker.state_bytes(), Relaxed);
                                        ledger.logs.fetch_sub(bytes, Relaxed);
                                        ledger.walked.fetch_add(walked.heap_bytes(), Relaxed);
                                    }
                                    let t = Instant::now();
                                    if wtx.send(walked).is_err() {
                                        break;
                                    }
                                    ProducerWaits::add(&waits.walker_send, t);
                                }
                            }
                            held.map(|last| {
                                std::sync::Arc::try_unwrap(last).unwrap_or_else(|last| (*last).clone())
                            })
                            .ok_or_else(|| {
                                Error::Execution("the run executed no cycle".to_string())
                            })
                        });
                        loop {
                            let t = Instant::now();
                            let Ok(walked) = wrx.recv() else {
                                break;
                            };
                            ProducerWaits::add(&waits.accumulator_recv, t);
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
                                head_log::chunk_handed(
                                    streamed - 1,
                                    stream_base(job.table),
                                    job.index,
                                    job.op_count(),
                                );
                                let t = Instant::now();
                                if btx.send(Built::Job(Box::new(job))).is_err() {
                                    return Err(Error::Prover("the layout thread stopped".into()));
                                }
                                ProducerWaits::add(&waits.accumulator_send, t);
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
                // The hand-out's counting thread, if it ran, done counting and
                // its counts the builder's: the windows end with it, as they
                // end with the count on the accumulator's thread.
                builder.join_handout_count()?;
                let windows_done = start.elapsed().as_secs_f64();
                let window_stamps = builder.stamps();
                if let Some(ledger) = ledger {
                    ledger.line("windows walked");
                    ledger.parts("builder", builder.heap_parts());
                    ledger.largest("builder", builder.largest_parts());
                }
                // The walk is done and the block's size known: when the
                // finish's forecast passes the pressure share, live
                // regeneration arms now, before its hump builds.
                if let Some(spill) = walk_spill.as_ref() {
                    let reading = HostReading::now();
                    let hwm = reading.hwm;
                    let cycles = (window_stamps.windows as u64).saturating_mul(window as u64);
                    let forecast = hand_off_need(hwm, cycles, target);
                    let real = real_hand_off_need(reading.raw_hwm, target);
                    hand_off_guard.note(reading.real_bytes(), real > 0 && forecast == 0);
                    let need = if deviations.hand_off_all {
                        u64::MAX
                    } else {
                        forecast.max(real)
                    };
                    if need > 0 {
                        crate::alloc_purge::note_memory_pressure();
                        let armed = spill.hand_off(need);
                        let g = |b: u64| b as f64 / (1u64 << 30) as f64;
                        let really = if reading.raw_hwm > hwm {
                            format!(" (really {:.2} GiB, the page-locked pool counted)", g(reading.raw_hwm))
                        } else {
                            String::new()
                        };
                        eprintln!(
                            "BLOCK HAND-OFF: at the walk's end ({windows_done:.2} s), VmHWM {:.2} GiB{really} + \
                             {:.2} for {:.1} M cycles walked is past {:.2} GiB ({}/{} of the target) by {} · \
                             live regeneration armed: {}",
                            g(hwm),
                            FINISH_GROWTH_PER_CYCLE * cycles as f64 / (1u64 << 30) as f64,
                            cycles as f64 / 1e6,
                            g(pressure_line(target)),
                            PRESSURE_SHARE.0,
                            PRESSURE_SHARE.1,
                            if need == u64::MAX {
                                "everything (a test's deviation)".to_string()
                            } else {
                                format!("{:.2} GiB", g(need))
                            },
                            if armed { "yes" } else { "no" },
                        );
                    }
                }
                // Partial drops: the last drop-back is past, so every table's
                // fate is known as phase A commits it from here, and the early
                // regenerator may start.
                if let Some(regen) = walk_spill.as_ref().and_then(|spill| spill.regen.as_ref()) {
                    regen.settle();
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
                let streams = options.rest_streams() && !deviations.omit_first_keccak_rnd;
                if streams {
                    let keep = live_recorder.is_some();
                    // The streamed chunks' regenerator: its thread waits
                    // for every streamed chunk's fate and starts it then; the
                    // rest's starts with phase B, from its final drops.
                    let start_early = || {
                        if let (Some(feed), Some(recorder)) = (&early_feed, &early_recorder) {
                            let early = regen::EarlyStart::spawn(
                                feed.clone(),
                                std::sync::Arc::clone(recorder),
                                regen::LiveArgs {
                                    elf_bytes: elf_bytes.to_vec(),
                                    private_input: private_inputs.to_vec(),
                                    max_rows: options.max_rows.clone(),
                                    window_cycles: window,
                                    form: stream_form,
                                    order: format.memw_aligned,
                                    registers: format.registers,
                                    memory: format.memory,
                                    faults: deviations.regen_faults,
                                    policy: deviations.regen_windows.map_or_else(
                                        regen::ahead_policy,
                                        |(streamed, _)| regen::AheadPolicy::Fixed(streamed),
                                    ),
                                },
                                std::sync::Arc::clone(&early_gate_in),
                                std::sync::Arc::clone(&early_paced),
                            );
                            *early_slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(early);
                        }
                    };
                    // Every streamed chunk is handed out and the finish begins:
                    // the thread goes up now, whether or not regeneration has
                    // armed, and takes each chunk's fate as phase A decides it.
                    // Its thread waits for the fates
                    // ([`stark::multilinear_block::BlockRegen::await_rank`]),
                    // which the arm and the streamed groups' commits decide
                    // while the finish still runs; until the finish's end its
                    // window holds only its frontier.
                    start_early();
                    let emitted = builder.finish_streamed_keeping(&last, rest_gate, keep, |header| {
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
                    let rest_regen = emitted?;
                    // The finish's end: the regenerator's window may now take
                    // the host's room.
                    early_paced.store(true, std::sync::atomic::Ordering::Release);
                    if deviations.abort_after_finish {
                        return Err(Error::Prover(
                            "a test's abort right after the finish".into(),
                        ));
                    }
                    let g = |b: usize| b as f64 / (1u64 << 30) as f64;
                    let (waited, waits) = rest_gate.waited();
                    eprintln!(
                        "BLOCK REST STREAM: the rest built as the gate admits each table under a \
                         {:.2} GiB gate, held at most {:.2} GiB (built and not yet placed) · {waits} \
                         acquisitions waited {waited:.2}s for room (summed) · building at most {} at \
                         once, {:.2} GiB the most",
                        g(rest_gate.budget()),
                        g(rest_gate.most()),
                        if rest_gate.build() == rest_gate.budget() {
                            "the gate".to_string()
                        } else {
                            format!("{:.2} GiB", g(rest_gate.build()))
                        },
                        g(rest_gate.building_most()),
                    );
                    if let Some(line) = crate::tables::trace_builder::take_row_scan_line() {
                        eprintln!("{line}");
                    }
                    if let Some(line) = crate::tables::trace_builder::take_bitwise_line() {
                        eprintln!("{line}");
                    }
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
                        rest_regen,
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
                // The builder's own rule.
                let padding = CutPadding::LastTable;
                split_keccak_rnd(&mut rest, options.keccak_rnd_rows_log2, padding, format.keccak_rnd)?;
                // Cut at the finish already, and held packed: a packed table
                // has no rows for `split_rows` to copy.
                if !options.finish_cuts {
                    split_ecdas(&mut rest, options.ecdas_rows_log2, padding, format.ecdas);
                    split_keccak(&mut rest, options.keccak_rows_log2, padding, format.memory);
                    split_ecsm(&mut rest, options.ecsm_rows_log2, padding);
                    split_copy(&mut rest, options.copy_rows_log2, padding, format.memory);
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
                    None,
                ))
            });

            // One closed group waits for phase A; the packer holds the next.
            let (gtx, grx) = std::sync::mpsc::sync_channel::<Vec<CommittedTable<'_, F, E>>>(1);
            let stream_airs = &stream_airs;
            let run_airs = &run_airs;
            let rest_keccak_airs = &rest_keccak_airs;
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
                    own: own_groups.then(|| OwnGroups {
                        open: Vec::new(),
                        shapes: Vec::new(),
                        keys: Vec::new(),
                        places: Vec::new(),
                        bytes: 0,
                        main_places: Vec::new(),
                        placed: 0,
                        recorder,
                    }),
                    close_ahead: close_ahead.clone(),
                    local: format.local_lookups().then(|| LocalPack {
                        air: stream_airs.local_bytes.as_ref(),
                        fault: deviations.local_bytes,
                        open: None,
                        own: None,
                        made: 0,
                        places: local_places.clone(),
                    }),
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
                        options.max_rows.cpu,
                    ),
                    RestIn::Stream(stream) => lay_out_rest_streamed(
                        stream,
                        program,
                        opts,
                        run_airs,
                        airs_beside.then_some(EarlyAirs {
                            airs: rest_keccak_airs,
                            misnamed,
                        }),
                        start,
                        sink,
                        ledger,
                        live_recorder,
                        format.keccak_rnd,
                        format.memw_aligned,
                        format.registers,
                        format.memory,
                        format.ecdas,
                        format.page_runs(),
                        close_ahead.clone(),
                    ),
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
                        Some(rest_gate),
                        recorder,
                        Some(waits),
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
                    fused_memory,
                    word_memory,
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
                    records_rest,
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
                for (i, shape, table, counts) in rest_tables {
                    if shapes[i].replace(shape).is_some() {
                        return Err(Error::Prover(format!("table {i} built twice")));
                    }
                    packer.place(Key::Air(i), shape, table, counts)?;
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
                // Each LOCAL_BYTES after every other table, `j`-th at `N + j`,
                // 2^16 rows.
                let locals = group_keys
                    .iter()
                    .flatten()
                    .filter(|key| matches!(key, Key::Local(_)))
                    .count();
                let base = shapes.len();
                let mut shapes = shapes;
                shapes.extend(std::iter::repeat_n(
                    (local_bytes::cols::NUM_COLUMNS, local_bytes::LOG_ROWS),
                    locals,
                ));
                let groups = group_keys
                    .into_iter()
                    .map(|keys| {
                        keys.into_iter()
                            .map(|key| match key {
                                Key::Local(j) => Ok((base + j) as u32),
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
                    fused_memory,
                    word_memory,
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
                        closed_before_bitwise: close_ahead.as_ref().is_some_and(|ahead| {
                            ahead.closed.load(std::sync::atomic::Ordering::Relaxed)
                        }),
                        blocked,
                        chunks,
                        slowest,
                        ahead_most,
                        packed_rest,
                        records_rest,
                        rest_tables,
                        gpack: (false, [0; 4]),
                    },
                    streamed_air,
                })
            });

            // A memory log: a group leaves the channel as the committer takes
            // it.
            let groups = grx.iter();
            let block = if deviations.host_only {
                // The trace alone: each closed group counted and dropped as
                // it comes, nothing committed.
                let mut drained = Drained {
                    groups: 0,
                    tables: 0,
                    cells: 0,
                    at: 0.0,
                };
                for group in groups {
                    drained.groups += 1;
                    drained.tables += group.len();
                    drained.cells += group
                        .iter()
                        .map(|table| (table.num_committed_columns() as u64) << table.num_vars())
                        .sum::<u64>();
                }
                drained.at = start.elapsed().as_secs_f64();
                Err(drained)
            } else {
                Ok(BlockCommitted::commit_groups_probed::<
                    H,
                    <H as multilinear::whir_hash::WhirHash>::Sponge,
                >(
                    groups.inspect(|group| {
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
                    // A measurement, off unless asked: the proof is the same.
                    cs_probe.clone(),
                    // Under local lookups: each group's digests and local
                    // challenges right after its commit, and the lane's local
                    // argues when it is on.
                    format
                        .local_lookups()
                        .then(|| multilinear_block::LocalPhaseA {
                            buses: local_bytes::LOCAL_BUSES.to_vec(),
                            lane: local_lane,
                            binding: stark::multilinear_table::batched::LocalBinding::ALL,
                        }),
                ))
            };
            (block, builder.join(), layout.join(), executor.join())
        });
        stamps.head = head_log::take_lines();
        for line in &stamps.head {
            eprintln!("{line}");
        }
        if arrivals {
            let (ops, at) = crate::tables::trace_builder::keccak_rnd_arrivals::take();
            eprintln!(
                "BLOCK KECCAK_RND ARRIVALS: tables of {} rows (24 an operation) · {ops} operations in the \
                 windows · {} table(s) complete during the windows, at [{}] s since the prove began",
                1usize << options.keccak_rnd_rows_log2,
                at.len(),
                at.iter()
                    .map(|t| format!("{:.2}", t.duration_since(start).as_secs_f64()))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        (stamps.execute, stamps.executor) =
            executed.map_err(|_| Error::Prover("the block's executor panicked".into()))??;
        // A thread's error names the cause; a commit that ran out of groups only
        // says that it did. A builder that stopped because the layout thread
        // stopped taking its tables, or because its packer stopped the finish's
        // byte gate as it went down with the layout, names the layout's error,
        // when it has one: which of the two the builder meets first is a race.
        let built = built.map_err(|_| Error::Prover("the block's builder panicked".into()))?;
        if let (Err(Error::Prover(message)), Ok(Err(cause))) = (&built, &laid)
            && (message.starts_with("the layout thread stopped")
                || message.starts_with("the finish's byte gate stopped"))
        {
            return Err(Error::Prover(format!("{message}: {cause}")));
        }
        let (windows_done, finished, streamed, window_stamps, finish_marks, rest_regen) = built?;
        stamps.windows = window_stamps;
        stamps.build_marks = finish_marks;
        stamps.producer = waits.stamps();
        let laid =
            laid.map_err(|_| Error::Prover("the block's layout thread panicked".into()))??;
        if stream_keccak_rnd {
            // Where the streamed KECCAK_RND tables went: each group's count.
            let mut groups: Vec<(usize, usize)> = Vec::new();
            for &(table, _, air) in &laid.streamed_air {
                if table != StreamTable::KeccakRnd {
                    continue;
                }
                let g = laid
                    .groups
                    .iter()
                    .position(|g| g.contains(&(air as u32)))
                    .unwrap_or(usize::MAX);
                match groups.last_mut() {
                    Some((last, n)) if *last == g => *n += 1,
                    _ => groups.push((g, 1)),
                }
            }
            eprintln!(
                "BLOCK KECCAK_RND STREAMED: {} table(s) handed out during the windows of {} \
                 · records {} · group:tables [{}]",
                groups.iter().map(|&(_, n)| n).sum::<usize>(),
                laid.table_counts.keccak_rnd,
                if card_fill.keccak_rnd && options.pack_finished {
                    "on (the card fills them)"
                } else {
                    "off (laid out on the host)"
                },
                groups
                    .iter()
                    .map(|(g, n)| format!("{g}:{n}"))
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
        let block = match block {
            Ok(block) => block,
            Err(drained) => {
                stamps.build = finished;
                stamps.streamed = (windows_done, streamed);
                stamps.prep = laid.busy;
                stamps.layout = laid.layout.clone();
                stamps.phase_a = start.elapsed().as_secs_f64();
                return Err(Error::Prover(format!(
                    "{HOST_ONLY}: the trace built and drained at {:.2}s · {} groups · {} tables · {} \
                     cells · nothing committed",
                    drained.at, drained.groups, drained.tables, drained.cells
                )));
            }
        };
        let mut block = block.map_err(|e| Error::Prover(format!("{e:?}")))?;
        if deviations.narrow_width_map && !block.fault_narrow_width_map() {
            return Err(Error::Prover("no narrow table to break".into()));
        }
        stamps.build = finished;
        stamps.streamed = (windows_done, streamed);
        stamps.prep = laid.busy;
        stamps.layout = laid.layout.clone();
        stamps.layout.gpack = (options.gpack, gpack::counts());
        stamps.phase_a = start.elapsed().as_secs_f64();
        stamps.pinned = multilinear::pinned::stats();
        if let Some(ledger) = ledger {
            ledger.line("phase A end");
        }
        // Phase A's freed pages back to the OS before phase B allocates: when
        // the block's memory is short (`note_pressure_past`, read once more
        // here), or as `LAMBDA_VM_ALLOC_PURGE` names it.
        note_pressure_past(HostReading::now().real_bytes(), target);
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
                format.memw_aligned,
                format.registers,
                format.memory,
                deviations.regen_drop_one_op,
            )
        });
        // Live regeneration: what phase A dropped, rebuilt beside phase B in
        // rank order. A dropped table is the chunk its rank names (the chunks
        // are placed first, in hand-out order), or the prove stops here.
        // The streamed chunks' regenerator started at the finish's end, when it
        // did (N4c).
        let early = early_owned.lock().unwrap_or_else(|e| e.into_inner()).take();
        let early_spawned = early.as_ref().map(regen::EarlyStart::started);
        let (early_runs, early_pacer) = match early.map(regen::EarlyStart::join) {
            Some((runs, pacer)) => (Some(runs), pacer),
            None => (None, None),
        };
        // Its head start is its regenerator's, started once the fates were
        // known; a thread that started none (nothing dropped) gave none.
        let early_started = early_runs.as_ref().and_then(|e| e.live_started);
        let early_streamed_waited = early_runs.as_ref().map(|e| e.streamed_waited);
        let mut rest_regen = rest_regen;
        let live = match (block.regen().cloned(), &regen_recorder) {
            (Some(live_regen), Some(recorder)) => {
                let drops = live_regen.report();
                let windows = live_regen.windows().to_vec();
                let window_of = std::sync::Arc::clone(&windows[0]);
                let mut plans = live_regen.into_plan().into_iter();
                let streamed_plan = plans.next();
                let rest_plan = plans.next();
                // The rest's dropped tables (N3), built again from the lists the
                // finish kept; with none dropped, the lists go now.
                let rest_run = match rest_plan {
                    Some(plan) if !plan.dropped.is_empty() => {
                        let regen = rest_regen.take().ok_or_else(|| {
                            Error::Prover(
                                "rest tables were dropped but the finish kept no lists".into(),
                            )
                        })?;
                        let producer = plan.producer.ok_or_else(|| {
                            Error::Prover("the rest's window has no producer".into())
                        })?;
                        Some(regen::RestRun::spawn(
                            regen,
                            recorder.rest_blocks(),
                            std::sync::Arc::clone(&plan.window),
                            producer,
                            plan.dropped,
                            regen::rest_generators(),
                            deviations.rest_faults,
                        ))
                    }
                    _ => None,
                };
                drop(rest_regen.take());
                let run = match early_runs {
                    Some(early) => {
                        if let Some(why) = early.error {
                            return Err(Error::Prover(format!("early regeneration: {why}")));
                        }
                        // It rebuilds the streamed chunks phase A dropped, each
                        // the chunk its rank names.
                        let planned: Vec<u64> =
                            streamed_plan.as_ref().map_or_else(Vec::new, |plan| {
                                plan.dropped.iter().map(|(r, _)| *r).collect()
                            });
                        let started: Vec<u64> = early.keys.iter().map(|(r, _)| *r).collect();
                        if planned != started {
                            return Err(Error::Prover(format!(
                                "the early regenerator took {} streamed chunks, phase A dropped {}",
                                started.len(),
                                planned.len()
                            )));
                        }
                        for (rank, key) in &early.keys {
                            if !laid
                                .streamed_air
                                .get(*rank as usize)
                                .is_some_and(|&(t, i, _)| (t, i) == *key)
                            {
                                return Err(Error::Prover(format!(
                                    "dropped table {rank} is not the chunk its rank names"
                                )));
                            }
                        }
                        early.live
                    }
                    None => match streamed_plan
                        .filter(|plan| !plan.dropped.is_empty())
                        .and_then(|plan| plan.producer.map(|producer| (plan.dropped, producer)))
                    {
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
                                format.memw_aligned,
                                format.registers,
                                format.memory,
                                deviations.regen_faults,
                                deviations
                                    .regen_windows
                                    .map_or_else(regen::ahead_policy, |(streamed, _)| {
                                        regen::AheadPolicy::Fixed(streamed)
                                    }),
                                None,
                            ))
                        }
                        None => None,
                    },
                };
                Some((drops, window_of, run, windows.get(1).cloned(), rest_run))
            }
            _ => None,
        };
        drop(rest_regen);
        // Each group's rebuilt class, which phase B's order follows (below);
        // all `None`, and the group order, when nothing was dropped.
        let classes = block.rebuilt_classes().map_err(|g| {
            Error::Prover(format!(
                "group {g} dropped tables of two classes for phase B to rebuild"
            ))
        })?;
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
        let mut transcript = <H as multilinear::whir_hash::WhirHash>::sponge();
        absorb_block(
            &mut transcript,
            &tag,
            elf_bytes,
            &laid.public_output,
            &laid.table_counts,
            &fused_statement_words(format.memory, laid.fused_memory, laid.word_memory),
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
                fused_memory: laid.fused_memory,
                word_memory: laid.word_memory,
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
                commitment: (&p.commitment).into(),
                columns,
                packed: None,
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
        // Phase B's order (N3): the groups with nothing dropped first, then the
        // rebuilt streamed and rest groups interleaved by their cells, so each
        // regenerator keeps up with only its share of phase B's pace.
        let group_cells = group_cells(&laid.shapes, &laid.groups);
        let order: Vec<usize> = match deviations.phase_b_order {
            Some(order) => order(laid.groups.len()),
            None => led_order(&classes, &group_cells, regen_lead),
        };
        // The group of the most tables first, under the knob, where phase B
        // rebuilds nothing (each rebuilt class goes in group order) and no
        // test's order is set; a measurement's list below still wins.
        let many_first = phase_b_many_first(deviations.phase_b_many_first)?;
        let order = match group_of_most_tables(&laid.groups) {
            Some(g)
                if many_first
                    && deviations.phase_b_order.is_none()
                    && classes.iter().all(Option::is_none) =>
            {
                eprintln!(
                    "BLOCK PHASE-B MANY FIRST: on ({PHASE_B_MANY_FIRST_ENV}) · group {g} of {} ({} tables) \
                     first",
                    laid.groups.len(),
                    laid.groups[g].len()
                );
                put_first(&[g], order).map_err(Error::Prover)?
            }
            _ => {
                eprintln!(
                    "BLOCK PHASE-B MANY FIRST: {} ({PHASE_B_MANY_FIRST_ENV})",
                    if !many_first {
                        "off (the default)"
                    } else if deviations.phase_b_order.is_some() {
                        "on, but a test's order is set: not applied"
                    } else {
                        "on, but phase B rebuilds groups: not applied"
                    }
                );
                order
            }
        };
        // `W3_PROBE_FIRST_GROUPS`, a measurement's: these groups first.
        let probe_first = probe_first_groups()?;
        // A measurement's or a test's order wins over the recursion's lead.
        let lead = lead.filter(|_| deviations.phase_b_order.is_none() && probe_first.is_none());
        let order = match probe_first {
            None => order,
            Some(_) if classes.iter().any(Option::is_some) => {
                return Err(Error::Prover(
                    "W3_PROBE_FIRST_GROUPS: a group is rebuilt in phase B, whose classes go in \
                     group order"
                        .into(),
                ));
            }
            Some(first) => put_first(&first, order).map_err(Error::Prover)?,
        };
        // Each class's regenerator deposits in its rank order, its groups'
        // order: an order that takes a class's groups otherwise could wait on a
        // deposit its window does not admit yet, so it is refused rather than
        // run.
        for class in 0..2usize {
            let taken: Vec<usize> = order
                .iter()
                .copied()
                .filter(|&g| classes.get(g).copied().flatten() == Some(class))
                .collect();
            if taken.windows(2).any(|w| w[0] > w[1]) {
                return Err(Error::Prover(
                    "phase B must take each class's rebuilt groups in group order".into(),
                ));
            }
        }
        // Phase B starts: an early start's windows go to their phase-B sizes.
        early_gate.store(true, std::sync::atomic::Ordering::Release);
        // The card's `words` mode (the default): the card-built rows' counters as
        // phase B starts (the block's end line holds both phases).
        if multilinear::gpu::card_rows_builds_words()
            && let Some(line) = multilinear::gpu::card_rows_report()
        {
            eprintln!(
                "BLOCK CARD ROWS AT PHASE B START: {}",
                line.trim_start_matches("BLOCK CARD ROWS: ")
            );
        }
        let phase_b_start = Instant::now();
        let phase_b_unix = stark::prove_split::epoch_secs();
        stamps.phase_b_order = order.clone();
        stamps.phase_b_overlap = options.phase_b_overlap != multilinear_block::PhaseBOverlap::Off;
        stamps.phase_b_pinned = options.phase_b_overlap == multilinear_block::PhaseBOverlap::Pinned;
        let phase_b_sched = phase_b_sched_from_env();
        stamps.phase_b_sched = phase_b_sched;
        let readbacks_before = multilinear::gpu::background_readbacks();
        // The recursion's lead (the early leaf first): once the tree offers its
        // leaves, the groups of the one phase B can finish soonest go next.
        // Per rebuilt class, whether all its dropped bytes fit its window at
        // the window's smallest in phase B (the adaptive window's floor, or
        // the fixed size): the windows the regeneration was opened with.
        let class_floors = match deviations.regen_windows {
            Some((streamed, rest)) => [streamed, rest],
            None => [regen::ahead_policy().initial(), regen::rest_ahead_bytes()],
        };
        let class_fit = live.as_ref().map_or_else(Vec::new, |(drops, ..)| {
            class_fits(&drops.by_class, &class_floors)
        });
        let lead_turn = |p: usize, order: &[usize]| {
            lead.and_then(|lead| {
                lead.turn(
                    p,
                    order,
                    &classes,
                    &group_cells,
                    phase_b_start.elapsed().as_secs_f64(),
                    &class_fit,
                )
            })
        };
        let reorder: Option<&multilinear_block::PhaseBReorder<'_>> =
            lead.is_some().then_some(&lead_turn);
        // Under local lookups: the digests phase A absorbed, each from its
        // table's own layout, must be the ones the verifier takes from its
        // AIRs ([`local_digests`]); a difference is the prover's error here,
        // not a proof the verifier refuses.
        if format.local_lookups() {
            let refs = run_airs
                .get()
                .ok_or_else(|| Error::Prover("the run's AIR set was never built".into()))?
                .air_refs();
            let local_air: AirRef<'_> = stream_airs.local_bytes.as_ref();
            let digests = local_digests::<<H as multilinear::whir_hash::WhirHash>::Sponge>(
                &laid.groups,
                &laid.shapes,
                |i| {
                    refs.get(i)
                        .copied()
                        .or_else(|| (i < laid.shapes.len()).then_some(local_air))
                },
            )?;
            if block.local_digests() != Some(&digests[..]) {
                return Err(Error::Prover(
                    "local lookups: the digests taken at commit time are not the AIRs'".into(),
                ));
            }
        }
        let (proof, argues, prepared_openings, groups, commit_ood, local_argues) =
            multilinear_block::block_prove_in_order::<_, _, _, H>(
                block
                    .with_phase_b_overlap(phase_b_overlap(options))
                    .with_overlap_faults(deviations.overlap_faults)
                    .with_sched_stamps(phase_b_sched)
                    // Under `LAMBDA_VM_TWO_CARD_SPLIT` the groups may split over
                    // two cards; a group phase B rebuilds stays on card 0, so
                    // each class keeps its order on one driver.
                    .with_two_card_split(classes.iter().map(Option::is_some).collect()),
                &config,
                &mut transcript,
                &openings,
                &tampered,
                fork_of,
                &deviations.argue,
                on_group,
                &order,
                reorder,
            )
            .map_err(|e| Error::Prover(format!("{e:?}")))?;
        stamps.phase_b = t.elapsed().as_secs_f64();
        // The process's counts of the paths a card refusal (or a store read
        // from another card) takes, as the base's phase B ends: a two-card arm
        // must show the one-card arm's counts, and no cross-card decline.
        eprintln!(
            "BLOCK HOST PATHS: host fallbacks {} · openings staged from the host {} · host widens {} tables \
             · host fills {} tables · cross-card declines {}",
            multilinear::gpu::host_fallbacks(),
            multilinear::gpu::openings_staged_from_host(),
            multilinear::narrow::host_widens().0,
            multilinear::records::fills().1,
            multilinear::gpu::cross_card_declines(),
        );
        // The order as phase B took it, when the lead moved groups.
        let order = match lead.and_then(PhaseBLead::taken) {
            Some(taken) if taken.order.len() == order.len() => taken.order,
            _ => order,
        };
        stamps.phase_b_order = order.clone();
        stamps.phase_b_unix = Some((phase_b_unix, stark::prove_split::epoch_secs()));
        let readbacks_after = multilinear::gpu::background_readbacks();
        stamps.background_readbacks = (
            readbacks_after.0 - readbacks_before.0,
            readbacks_after.1 - readbacks_before.1,
            readbacks_after.2,
        );
        if let Some((drops, window_of, run, rest_window, rest_run)) = live {
            let recipes = recorder.map_or(0, |r| r.recipes().0.len());
            let mut lines = vec![regen::dropped_line(regen_mode, &drops, recipes)];
            if let Some(purges) = &drop_purges {
                lines.push(regen::keep_line(&drops));
                lines.push(purges.line());
            }
            let rest_report = rest_run.map(regen::RestRun::join);
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
            let head_start =
                early_started.map(|t| phase_b_start.saturating_duration_since(t).as_secs_f64());
            let early_pace = early_pacer.and_then(|pacer| pacer.join().ok());
            lines.push(regen::head_start_line(
                head_start,
                early_spawned.map(|t| phase_b_start.saturating_duration_since(t).as_secs_f64()),
                early_streamed_waited,
                early_pace,
            ));
            lines.push(format!("BLOCK REGEN window: {}", window_of.report()));
            if let Some(report) = &rest_report {
                lines.push(report.line());
                for failure in report.failures.iter().take(20) {
                    lines.push(format!("BLOCK REGEN REST FAILED {failure}"));
                }
            }
            if let (Some(window), Some((tables, bytes))) = (&rest_window, drops.by_class.get(1))
                && *tables > 0
            {
                lines.push(format!(
                    "BLOCK REGEN rest window: {} · {tables} tables {:.2} GiB dropped",
                    window.report(),
                    *bytes as f64 / (1u64 << 30) as f64
                ));
            }
            let in_class = |c: usize| classes.iter().filter(|&&k| k == Some(c)).count();
            lines.push(format!(
                "BLOCK REGEN phase B order: {} of {} groups rebuilt ({} streamed, {} rest), \
                 interleaved after the others · phase B began {} after the regenerator · {} \
                 streamed groups ahead of the rest's first",
                classes.iter().filter(|c| c.is_some()).count(),
                classes.len(),
                in_class(0),
                in_class(1),
                offset.map_or("n/a".to_string(), |s| format!("{s:.2} s")),
                order
                    .iter()
                    .filter(|&&g| classes.get(g).copied().flatten().is_some())
                    .take_while(|&&g| classes[g] == Some(0))
                    .count(),
            ));
            let spans: Vec<(f64, f64)> = groups.iter().map(|g| (g.start_b, g.end_b)).collect();
            lines.push(regen::order_taken_line(
                &order,
                &classes,
                &group_cells,
                &spans,
            ));
            for line in &lines {
                eprintln!("{line}");
            }
            stamps.regen = Some(regen::RegenStamps {
                recipes,
                dropped: drops.tables,
                dropped_bytes: drops.bytes,
                armed_kept: drops.armed_kept.0,
                regenerated: report.as_ref().map_or(0, |r| r.deposited)
                    + rest_report.as_ref().map_or(0, |r| r.deposited),
                mismatches: report.as_ref().map_or(0, |r| r.mismatches)
                    + rest_report.as_ref().map_or(0, |r| r.mismatches),
                failed: report.as_ref().map_or(0, |r| r.failures.len() + r.skipped)
                    + rest_report.as_ref().map_or(0, |r| r.failures.len()),
                error: report
                    .as_ref()
                    .and_then(|r| r.error.clone())
                    .or_else(|| rest_report.as_ref().and_then(|r| r.error.clone())),
                rest_dropped: drops.by_class.get(1).map_or(0, |c| c.0),
                rest_regenerated: rest_report.as_ref().map_or(0, |r| r.deposited),
                held: drops.held,
                phase_b_order: order.clone(),
                head_start,
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
        if spill.is_some() {
            let g = |b: u64| b as f64 / (1u64 << 30) as f64;
            let host = match *host_most.lock().unwrap_or_else(|e| e.into_inner()) {
                Some(h) => format!(" · host at most {:.2} GiB", g(h.bytes())),
                None => String::new(),
            };
            // With the page-locked pool, what the host really held beside it
            // and what the real-memory guard alone decided (said only when the
            // pool hid some of the host).
            use std::sync::atomic::Ordering::Relaxed;
            let real_most = real_guard.most.load(Relaxed);
            stamps.real_guard_alone = real_guard.alone.load(Relaxed);
            let netted_most = host_most
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .map_or(0, |h| h.bytes());
            let host = if real_most > netted_most {
                format!(
                    "{host} · really {:.2} GiB, the page-locked pool counted · the real-memory guard alone decided {}",
                    g(real_most),
                    stamps.real_guard_alone
                )
            } else {
                host
            };
            stamps.drops = Some(format!(
                "live regeneration decides as `auto` (target {:.1} GiB){host}",
                g(target)
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
        let fills_after = multilinear::records::fills();
        stamps.records_fills = (
            fills_after.0 - fills_before.0,
            fills_after.1 - fills_before.1,
        );
        let ((widens, widen_s), (fills, fill_s)) = multilinear::records::upload_times();
        stamps.upload_times = (
            (widens - times_before.0.0, widen_s - times_before.0.1),
            (fills - times_before.1.0, fill_s - times_before.1.1),
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
            commit_ood,
            fused_memory: laid.fused_memory,
            word_memory: laid.word_memory,
            local: local_argues,
        })
    })
}

/// The bytes [`absorb_block`] appends, in its order and with its pad: what an
/// in-guest verifier absorbs as one run of program constants. One stream, so a
/// test can hold the two to the same transcript.
pub(crate) fn block_statement_bytes(
    statement: BlockStatement<'_>,
    tag: &[u8],
    elf_digest: &[u8; 32],
    config: &multilinear::whir_chain::ChainConfig,
    memory: crate::tables::fused_memory::MemoryFormat,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(tag);
    bytes.extend_from_slice(elf_digest);
    bytes.extend_from_slice(&(statement.public_output.len() as u64).to_le_bytes());
    bytes.extend_from_slice(statement.public_output);
    for count in statement::table_count_values(statement.table_counts) {
        bytes.extend_from_slice(&count.to_le_bytes());
    }
    for word in fused_statement_words(memory, statement.fused_memory, statement.word_memory) {
        bytes.extend_from_slice(&word.to_le_bytes());
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

/// The `append_bytes` calls [`absorb_block`] makes, in its order: what an
/// in-guest verifier of a Poseidon1 base absorbs call by call, because that
/// sponge binds each call's length (`P1Transcript::append_bytes`). Recorded
/// from [`absorb_block`] itself, so the two cannot drift; their concatenation
/// is [`block_statement_bytes`]. Refuses a statement whose absorb is not only
/// byte strings, which the replay could not follow.
pub(crate) fn block_statement_calls(
    statement: BlockStatement<'_>,
    tag: &[u8],
    elf_bytes: &[u8],
    config: &multilinear::whir_chain::ChainConfig,
    memory: crate::tables::fused_memory::MemoryFormat,
) -> Result<Vec<Vec<u8>>, String> {
    let mut calls = StatementCalls::default();
    absorb_block(
        &mut calls,
        tag,
        elf_bytes,
        statement.public_output,
        statement.table_counts,
        &fused_statement_words(memory, statement.fused_memory, statement.word_memory),
        statement.num_private_input_pages,
        statement.runtime_page_ranges,
        statement.table_num_vars,
        config,
        statement.groups,
    );
    if calls.foreign {
        return Err("the block statement absorbs more than byte strings".to_string());
    }
    Ok(calls.bytes)
}

/// A transcript that records `append_bytes` calls and nothing else
/// ([`block_statement_calls`]); any other call marks the record unusable.
#[derive(Default)]
struct StatementCalls {
    bytes: Vec<Vec<u8>>,
    foreign: bool,
}

impl IsTranscript<E> for StatementCalls {
    fn append_field_element(&mut self, _element: &FieldElement<E>) {
        self.foreign = true;
    }

    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.bytes.push(new_bytes.to_vec());
    }

    fn state(&self) -> [u8; 32] {
        [0; 32]
    }

    fn sample_field_element(&mut self) -> FieldElement<E> {
        self.foreign = true;
        FieldElement::zero()
    }

    fn sample_u64(&mut self, _upper_bound: u64) -> u64 {
        self.foreign = true;
        0
    }
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
    /// [`BlockWhirProof::fused_memory`].
    pub fused_memory: [u64; 2],
    /// [`BlockWhirProof::word_memory`].
    pub word_memory: [u64; 7],
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
    pub fused_memory: [u64; 2],
    pub word_memory: [u64; 7],
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
            fused_memory: self.fused_memory,
            word_memory: self.word_memory,
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
            fused_memory: self.fused_memory,
            word_memory: self.word_memory,
        }
    }
}

impl<L> BlockWhirProof<L> {
    /// The same proof carrying `local` for its local lookups.
    pub fn with_local<M>(self, local: M) -> BlockWhirProof<M> {
        BlockWhirProof {
            proof: self.proof,
            table_num_vars: self.table_num_vars,
            runtime_page_ranges: self.runtime_page_ranges,
            table_counts: self.table_counts,
            public_output: self.public_output,
            num_private_input_pages: self.num_private_input_pages,
            groups: self.groups,
            prepared: self.prepared,
            argues: self.argues,
            commit_ood: self.commit_ood,
            fused_memory: self.fused_memory,
            word_memory: self.word_memory,
            local,
        }
    }

    /// The proof's statement.
    pub fn statement(&self) -> BlockStatement<'_> {
        BlockStatement {
            table_num_vars: &self.table_num_vars,
            runtime_page_ranges: &self.runtime_page_ranges,
            table_counts: &self.table_counts,
            public_output: &self.public_output,
            num_private_input_pages: self.num_private_input_pages,
            groups: &self.groups,
            fused_memory: self.fused_memory,
            word_memory: self.word_memory,
        }
    }
}

/// What a verifier derives from the program and a statement before any root,
/// once the statement's checks pass. The host verifier and the recursion's tree
/// plan both start from it, so they check one statement the same way.
pub(crate) struct BlockFrame {
    pub(crate) page_configs: Vec<crate::tables::page::PageConfig>,
    /// The page tables cut from [`Self::page_configs`] under the format's page
    /// runs ([`block_page_tables`]), in AIR order.
    pub(crate) page_tables: Vec<crate::tables::page_w::PageTable>,
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
    /// Under local lookups ([`BlockFormat::local_lookups`]): the index of the
    /// first LOCAL_BYTES table, after every other ([`check_local_groups`]):
    /// where the in-guest plan routes their fixed columns.
    pub(crate) local_base: Option<usize>,
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
    check_memory_format(format)?;
    // At most the tables a block states, before anything is built from them.
    check_table_count(statement.table_num_vars.len())?;
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;

    validate_block_counts(statement.table_counts)?;
    let max_pages = crate::tables::page::max_private_input_pages();
    if statement.num_private_input_pages > max_pages {
        return Err(Error::InvalidTableCounts(format!(
            "num_private_input_pages ({}) exceeds max ({max_pages})",
            statement.num_private_input_pages,
        )));
    }
    // The fused memory tables, under the verifier's memory format: none under
    // the split format, where any stated pair is refused.
    let fused = fused_tables(format, statement.fused_memory)?;
    // Under the word memory format, none of the tables it retires, and its
    // own tables' counts; under any other format no word table.
    word_tables_under(
        format.memory,
        statement.table_counts,
        statement.fused_memory,
    )?;
    let words = word_counts_under(format.memory, statement.word_memory)?;
    // Under the word memory format no table of a chip it takes no call of
    // (IS_WH, BLAKE3, ECSM, HINT).
    chip_tables_under(format.memory, statement.table_counts, statement.word_memory)?;
    let num_tables = statement.table_num_vars.len();
    // Every table but the page tables, from the statement's counts.
    let Some(other_tables) = statement
        .table_counts
        .total()
        .and_then(|t| t.checked_add(FIXED_TABLE_COUNT))
        .and_then(|t| t.checked_add(fused[0]))
        .and_then(|t| t.checked_add(fused[1]))
        .and_then(|t| words.iter().try_fold(t, |t, &n| t.checked_add(n)))
    else {
        return Err(Error::InvalidTableCounts(
            "the table counts do not sum to a usize".to_string(),
        ));
    };
    // The page tables the statement can hold: its tables less every other one.
    let Some(page_room) = num_tables.checked_sub(other_tables) else {
        return Err(Error::InvalidTableCounts(format!(
            "the statement's counts imply {other_tables} tables besides its pages and it states \
             {num_tables} heights"
        )));
    };
    let page_configs = Traces::page_configs_from_elf_and_runtime(
        &program,
        statement.runtime_page_ranges,
        statement.num_private_input_pages,
        max_block_pages(page_room, format),
    )?;
    // The page tables: cut by the verifier from its own page list, at its own
    // page runs.
    let page_tables = block_page_tables(&page_configs, format)?;
    let Some(expected) = other_tables.checked_add(page_tables.len()) else {
        return Err(Error::InvalidTableCounts(
            "the table counts do not sum to a usize".to_string(),
        ));
    };
    // Under local lookups the LOCAL_BYTES tables come after every other: how
    // many is the statement's, held below to one a group with a local
    // sender.
    let local_count = if format.local_lookups() {
        num_tables.checked_sub(expected).ok_or_else(|| {
            Error::InvalidTableCounts(format!(
                "the statement implies {expected} tables and states {num_tables} heights",
            ))
        })?
    } else if expected != num_tables {
        return Err(Error::InvalidTableCounts(format!(
            "the statement implies {expected} tables and states {num_tables} heights",
        )));
    } else {
        0
    };
    check_table_heights(statement.table_num_vars)?;
    check_bitwise_height(statement.table_num_vars)?;
    check_chunked_heights(statement.table_counts, statement.table_num_vars)?;

    let cpu_heights = block_cpu_heights(statement.table_counts, statement.table_num_vars)?;
    let inputs = BlockAirInputs::new(
        proof_options,
        &program,
        cpu_heights,
        statement.public_output,
    )?;
    // Under the word memory format no PAGE AIR a listed page: the page tables
    // are built straight from the list.
    let mut airs = block_airs(
        &program,
        proof_options,
        page_air_pages(format.memory, &page_configs),
        statement.table_counts,
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
        format.registers,
        format.ecdas,
    );
    airs.with_fused_memory(proof_options, fused);
    if format.memory == crate::tables::fused_memory::MemoryFormat::Words {
        airs.with_word_memory_at(proof_options, words, &page_configs, format.page_runs())?;
    }
    airs.with_local_bytes(proof_options, local_count);
    // The CPU heights were read at `cpu_proof_start`: the tables must be there.
    airs.check_cpu_position(statement.table_counts)?;
    let air_refs = airs.air_refs();
    if air_refs.len() != num_tables {
        return Err(Error::InvalidTableCounts(format!(
            "the layout has {} tables, the statement {num_tables}",
            air_refs.len(),
        )));
    }
    // Every fused memory table at most the block's chunk.
    if let Some((air, &num_vars)) = air_refs
        .iter()
        .zip(statement.table_num_vars)
        .filter(|(air, _)| {
            air.name().starts_with("MEM_LOAD[") || air.name().starts_with("MEM_STORE[")
        })
        .find(|(_, num_vars)| usize::from(**num_vars) > BLOCK_FUSED_MAX_VARS)
    {
        return Err(Error::InvalidTableCounts(format!(
            "{} states 2^{num_vars} rows — a block takes at most 2^{BLOCK_FUSED_MAX_VARS}",
            air.name()
        )));
    }
    // Every word table at most the block's chunk.
    check_word_heights(
        air_refs
            .iter()
            .zip(statement.table_num_vars)
            .map(|(air, &num_vars)| (air.name(), num_vars)),
    )?;
    // Every PAGE_W table at its one height (none outside the word memory format).
    check_page_w_heights(
        air_refs
            .iter()
            .zip(statement.table_num_vars)
            .map(|(air, &num_vars)| (air.name(), num_vars)),
    )?;
    // Under the word memory format every page table at its place, name and
    // height, as the verifier cut them.
    if format.memory == crate::tables::fused_memory::MemoryFormat::Words {
        check_page_table_heights(
            &page_tables,
            first_page_index(&airs)?,
            air_refs
                .iter()
                .zip(statement.table_num_vars)
                .map(|(air, &num_vars)| (air.name(), num_vars)),
        )?;
    }
    // Under the in-row register format MEMW_R's slot holds the residual register tables,
    // their rows capped.
    if format.registers == crate::tables::register::RegisterFormat::InRow {
        check_residual_rows(
            air_refs
                .iter()
                .zip(statement.table_num_vars)
                .filter(|(air, _)| air.name().starts_with("MEMW_R["))
                .map(|(_, &num_vars)| num_vars),
        )?;
    }
    // Under local lookups: every receiver of a local bus is a LOCAL_BYTES
    // table, and every LOCAL_BYTES table is 2^16 rows.
    let local_base = format.local_lookups().then_some(expected);
    if let Some(base) = local_base {
        check_local_airs(&air_refs, statement.table_num_vars, base)?;
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
    if let Some(base) = local_base {
        let senders: Vec<bool> = airs
            .air_refs()
            .iter()
            .map(|air| local_bytes::sends_local(air.bus_interactions()))
            .collect();
        check_local_groups(statement.groups, &senders, base)?;
    }
    let sizes: Vec<usize> = statement.groups.iter().map(Vec::len).collect();
    let group_shapes: Vec<(usize, usize)> = order.iter().map(|&i| shapes[i]).collect();
    let (stack_layouts, domains) = stacks(&group_shapes, &sizes, &config)?;
    // Every group within the stack budget — the verifier's constant, not the
    // prover's packing — except a table that alone needs more: it cannot be
    // split, so it is a group of its own (the packing's rule too). A group's
    // LOCAL_BYTES table, which closes it, is outside the budget.
    for (g, (group, layout)) in statement.groups.iter().zip(&stack_layouts).enumerate() {
        let own: Vec<(usize, usize)> = group
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| local_base.is_none_or(|base| i < base))
            .map(|i| shapes[i])
            .collect();
        let polys = if own.len() == group.len() {
            layout.num_polys()
        } else {
            stark::multilinear_table::global_layout(&own, config.format.stack)
                .map_err(|e| Error::Prover(format!("{e:?}")))?
                .num_polys()
        };
        if polys > format.group_polys && own.len() > 1 {
            return Err(Error::InvalidTableCounts(format!(
                "group {g} stacks into {polys} polynomials — a group takes at most {}",
                format.group_polys
            )));
        }
    }
    Ok(BlockFrame {
        page_configs,
        page_tables,
        airs,
        shapes,
        config,
        order,
        sizes,
        stack_layouts,
        domains,
        local_base,
    })
}

/// The frame's static check of local lookups, over the verifier's own AIR
/// set: no table before `base` receives on a local bus, and every table from
/// `base` on is a LOCAL_BYTES table of 2^16 rows. With the multilinear
/// verifier's own check that every local receiver reads fixed columns only,
/// this is what makes each group's local argue a lookup into a table the
/// verifier fixes.
pub(crate) fn check_local_airs(
    air_refs: &[AirRef<'_>],
    table_num_vars: &[u8],
    base: usize,
) -> Result<(), Error> {
    for (i, air) in air_refs.iter().enumerate() {
        let local_receiver = air
            .bus_interactions()
            .iter()
            .any(|b| !b.is_sender && local_bytes::LOCAL_BUSES.contains(&b.bus_id));
        let is_local_bytes = air.name().starts_with("LOCAL_BYTES[");
        if i < base && (local_receiver || is_local_bytes) {
            return Err(Error::InvalidTableCounts(format!(
                "{} (table {i}) receives a local bus: only LOCAL_BYTES does",
                air.name()
            )));
        }
        if i >= base {
            if !is_local_bytes {
                return Err(Error::InvalidTableCounts(format!(
                    "table {i} after the run's tables is {}, not LOCAL_BYTES",
                    air.name()
                )));
            }
            if usize::from(table_num_vars[i]) != local_bytes::LOG_ROWS {
                return Err(Error::InvalidTableCounts(format!(
                    "LOCAL_BYTES (table {i}) states 2^{} rows — it has 2^{}",
                    table_num_vars[i],
                    local_bytes::LOG_ROWS
                )));
            }
        }
    }
    Ok(())
}

/// Where the LOCAL_BYTES tables sit: every group holding a local sender
/// (`senders[i]`, table `i` sends a local lookup) ends with exactly one, the
/// `j`-th such group with table `base + j`; no other group holds one; and
/// there is one a group with a local sender, no more. So each group's local
/// lookups balance against its own fixed table.
pub(crate) fn check_local_groups(
    groups: &[Vec<u32>],
    senders: &[bool],
    base: usize,
) -> Result<(), Error> {
    let mut j = 0usize;
    for (g, group) in groups.iter().enumerate() {
        let locals: Vec<usize> = group
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| i >= base)
            .collect();
        let sends = group
            .iter()
            .any(|&i| senders.get(i as usize).copied().unwrap_or(false));
        match (sends, locals.as_slice()) {
            (false, []) => {}
            (true, &[local]) if group.last().map(|&i| i as usize) == Some(local) => {
                if local != base + j {
                    return Err(Error::InvalidTableCounts(format!(
                        "group {g} ends with LOCAL_BYTES table {local}, not {}",
                        base + j
                    )));
                }
                j += 1;
            }
            (true, _) => {
                return Err(Error::InvalidTableCounts(format!(
                    "group {g} holds a local sender: it ends with exactly one LOCAL_BYTES table"
                )));
            }
            (false, _) => {
                return Err(Error::InvalidTableCounts(format!(
                    "group {g} holds a LOCAL_BYTES table and no local sender"
                )));
            }
        }
    }
    let stated = senders.len().saturating_sub(base);
    if j != stated {
        return Err(Error::InvalidTableCounts(format!(
            "{stated} LOCAL_BYTES tables for {j} groups with a local sender"
        )));
    }
    Ok(())
}

/// Verifies a proof from [`prove_block_whir`] under `format` — the verifier's
/// own constants, never the proof's.
pub fn verify_block_whir<L: LocalPart>(
    proof: &BlockWhirProof<L>,
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
pub(crate) fn verify_block_whir_with<L: LocalPart>(
    proof: &BlockWhirProof<L>,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    skip_prepared: bool,
    argue_checks: VerifierChecks,
) -> Result<bool, Error> {
    verify_block_whir_traced(
        proof,
        elf_bytes,
        proof_options,
        format,
        skip_prepared,
        argue_checks,
    )
    .map(|verdict| verdict.is_ok())
}

/// Why [`verify_block_whir_traced`] refused a proof whose statement frames: the
/// derived prepared stacks did not commit, the COMMIT bus's target does not
/// exist, or the multilinear proof's own check refused (with its error).
///
/// The error is read by the host-build tests only (a card build compiles none).
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum BlockRefusal {
    Prepared,
    Owed,
    Proof(multilinear::Error),
}

/// [`verify_block_whir_with`], a refusal carrying the check that refused: what
/// a negative test asserts, so it cannot pass on another error.
pub(crate) fn verify_block_whir_traced<L: LocalPart>(
    proof: &BlockWhirProof<L>,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    skip_prepared: bool,
    argue_checks: VerifierChecks,
) -> Result<Result<(), BlockRefusal>, Error> {
    // Local lookups by the verifier's format: a local argue a group under
    // it, none in today's — the other kind of proof is refused.
    let local_argues = match (format.local_lookups(), proof.local.local_argues()) {
        (true, Some(argues)) if argues.len() == proof.groups.len() => Some(argues),
        (false, None) => None,
        (local, argues) => {
            return Err(Error::InvalidTableCounts(format!(
                "the proof carries {} local argues for {} groups; its format {} local lookups",
                argues.map_or("no".to_string(), |a| a.len().to_string()),
                proof.groups.len(),
                if local { "argues" } else { "has no" },
            )));
        }
    };
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

    let tag = block_format_tag(format);
    Ok(crate::with_base_hash!(format.zf.base, |H| {
        let mut transcript = <H as multilinear::whir_hash::WhirHash>::sponge();
        absorb_block(
            &mut transcript,
            &tag,
            elf_bytes,
            &proof.public_output,
            &proof.table_counts,
            &fused_statement_words(format.memory, proof.fused_memory, proof.word_memory),
            proof.num_private_input_pages,
            &proof.runtime_page_ranges,
            &proof.table_num_vars,
            config,
            &proof.groups,
        );
        // The prepared commitments, derived here from the program: their roots
        // are the verifier's, never the proof's, and committed on the host, so
        // no card kernel decides what this verifier accepts.
        let prepared: Vec<GroupPrepared<H>> = match group_stacks(prepared_columns, &proof.groups)
            .and_then(|stacks| commit_groups_on_host::<H>(stacks, config))
        {
            Ok(prepared) => prepared,
            Err(_) => return Ok(Err(BlockRefusal::Prepared)),
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
        // challenges — replayed on a fork through the roots block itself,
        // answers and all.
        let mut probe = transcript.clone();
        multilinear_block::absorb_roots_and_answers::<E, _>(
            &mut probe,
            &proof.proof.roots,
            &derived,
            &proof.commit_ood,
        );
        let z: FieldElement<E> = probe.sample_field_element();
        let alpha: FieldElement<E> = probe.sample_field_element();
        let Some(owed) = crate::compute_commit_bus_offset(&proof.public_output, 0, &z, &alpha)
        else {
            return Ok(Err(BlockRefusal::Owed));
        };
        // Each table's digest, the verifier's own, from its AIR set.
        let digests = match local_argues {
            Some(_) => local_digests::<<H as multilinear::whir_hash::WhirHash>::Sponge>(
                &proof.groups,
                shapes,
                |i| air_refs.get(i).copied(),
            )?,
            None => Vec::new(),
        };
        let local = local_argues.map(|argues| multilinear_block::BlockLocalCheck {
            buses: &local_bytes::LOCAL_BUSES,
            digests: &digests,
            argues,
        });
        multilinear_block::block_verify_with::<_, _, _, H>(
            &proof.proof,
            &proof.commit_ood,
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
            local,
        )
        .map_err(BlockRefusal::Proof)
    }))
}

/// A stale `LAMBDA_VM_CARD_ROWS` word (`derive`, `1`, `registers`, or any
/// word the uploads do not take) refuses the block prove with an error that
/// names the values it takes; `off`, `order`, `words` and unset prove.
#[cfg(all(test, feature = "cuda"))]
mod card_rows_refusal_tests {
    use crate::Error;

    #[test]
    fn a_stale_card_rows_word_refuses_the_prove_and_names_the_values() {
        let refusal = |word| multilinear::gpu::card_rows_refusal_of(word).map_err(Error::Prover);
        for word in [
            "derive",
            " derive ",
            "1",
            "registers",
            "equal",
            "yes",
            "Words",
        ] {
            let Err(Error::Prover(why)) = refusal(Some(word)) else {
                panic!("LAMBDA_VM_CARD_ROWS={word} must refuse the prove");
            };
            assert!(why.contains("LAMBDA_VM_CARD_ROWS"), "{why}");
            assert!(why.contains("off, order or words"), "{why}");
        }
        for word in [
            None,
            Some(""),
            Some("off"),
            Some("0"),
            Some("order"),
            Some(" words "),
        ] {
            assert!(refusal(word).is_ok(), "{word:?} must prove");
        }
    }
}

#[cfg(test)]
mod memory_target_tests {
    use super::{
        CgroupValue, HostReading, MEMORY_RESERVE_PER_G, auto_wants, cgroup_memory, decision_bytes,
        drop_wanted, hand_off_need, memory_reserve_bytes, memory_short, memory_target_from,
        memory_target_knob, no_disk_room, parse_reserve_per_g, parse_target_gib, parse_tree_drop,
        pressure_line, real_decision_bytes, real_hand_off_need,
    };
    use crate::alloc_purge::{NoDisk, has_room};

    const GIB: u64 = 1 << 30;

    /// The memory target's knob is read under its name only (its old name
    /// refuses the prove, [`super::regen::refuse_removed_spill_words`]); its
    /// value is GiB above 0, else the machine's.
    #[test]
    fn the_memory_target_knob_reads_its_name_only() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                pairs
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, v)| v.to_string())
            }
        };
        assert_eq!(memory_target_knob(env(&[])), None);
        assert_eq!(
            memory_target_knob(env(&[("LAMBDA_VM_BLOCK_MEMORY_TARGET_GIB", "54")])),
            Some(54 * GIB)
        );
        assert_eq!(
            memory_target_knob(env(&[("LAMBDA_VM_BLOCK_SPILL_TARGET_GIB", "108.28")])),
            None,
            "the old name sets nothing"
        );
        assert_eq!(
            parse_target_gib(Some(" 2.5 ".to_string())),
            Some(5 * GIB / 2)
        );
        for bad in [
            None,
            Some("0"),
            Some("-1"),
            Some("nan"),
            Some("inf"),
            Some("x"),
        ] {
            assert_eq!(parse_target_gib(bad.map(str::to_string)), None, "{bad:?}");
        }
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
            ..HostReading::default()
        };
        assert_eq!(
            reading(24_680_000_000, 40_580_000_000, 15_180_000_000).bytes(),
            25_400_000_000
        );
        assert_eq!(reading(0, 4 * GIB, 6 * GIB).bytes(), 0);
        assert_eq!(reading(30 * GIB, 40 * GIB, 15 * GIB).bytes(), 30 * GIB);
        assert_eq!(reading(0, 120 * GIB, 0).bytes(), 120 * GIB);
    }

    /// A reading the page-locked pool hid `hidden` bytes of: `hwm` and
    /// `charged` netted, the raw figures with them.
    fn pooled(hwm: u64, charged: u64, inactive_file: u64, hidden: u64) -> HostReading {
        HostReading {
            hwm,
            charged,
            inactive_file,
            raw_hwm: hwm + hidden,
            raw_charged: charged + hidden,
        }
    }

    /// ★ Without a pool the real-memory guard decides nothing: the rule reads
    /// the same bytes and adds its reserve, so the guard never wants a table
    /// out that the rule keeps, peak or partial reading, armed or not; the
    /// walk's hand-off never asks more; and the real reading is the rule's.
    #[test]
    fn the_real_memory_guard_decides_nothing_without_a_pool() {
        let gib = |g: f64| (g * GIB as f64) as u64;
        let mut cases = 0;
        for hwm in [0, gib(20.0), gib(54.2), gib(92.0), gib(108.5), gib(140.0)] {
            for charged in [0, gib(30.0), gib(61.0), gib(109.0), gib(150.0)] {
                for inactive_file in [0, gib(2.0), gib(40.0)] {
                    let reading = pooled(hwm, charged, inactive_file, 0);
                    assert_eq!(reading.real_bytes(), reading.bytes());
                    for target in [0, gib(54.0), gib(108.5), u64::MAX] {
                        for cycles in [0u64, 300_000_000, 600_000_000] {
                            assert!(
                                real_hand_off_need(reading.raw_hwm, target)
                                    <= hand_off_need(reading.hwm, cycles, target)
                            );
                        }
                        for (now, allocator) in [
                            (false, None),
                            (true, None),
                            (true, Some(gib(70.0))),
                            (true, Some(0)),
                        ] {
                            let host = decision_bytes(&reading, allocator, now);
                            assert!(real_decision_bytes(&reading, now) <= host);
                            for extra in [0, gib(10.0)] {
                                for cells in [0, 10_000_000_000, 30_000_000_000] {
                                    for bytes in [0, 64 << 20, gib(2.0)] {
                                        let (rule, guard) = auto_wants(
                                            &reading,
                                            host,
                                            now,
                                            extra,
                                            target,
                                            MEMORY_RESERVE_PER_G,
                                            cells,
                                            bytes,
                                        );
                                        assert!(
                                            rule || !guard,
                                            "hwm {hwm} charged {charged} inactive {inactive_file} \
                                             target {target} now {now} {allocator:?} extra {extra} \
                                             cells {cells} bytes {bytes}"
                                        );
                                        cases += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(cases > 10_000, "{cases}");
    }

    /// ★ With the pool hiding its free part, the guard wants a table out the
    /// rule keeps once what the host really holds passes the target: the rule
    /// reads 80 GiB and forecasts 91.9 against 108.5, while the host holds
    /// 105 GiB and the table and the armed reserve take it to 109.25. The
    /// walk's hand-off likewise, from a real peak past the pressure line with a
    /// netted one under it. The large block's own readings (61 GiB netted,
    /// 88 GiB really, armed) stay under both, so the guard keeps out of it.
    #[test]
    fn the_real_memory_guard_arms_where_the_pool_hides_the_host() {
        let gib = |g: f64| (g * GIB as f64) as u64;
        let target = gib(108.5);
        let reading = pooled(gib(80.0), gib(80.0), 0, gib(25.0));
        for now in [false, true] {
            let host = decision_bytes(&reading, None, now);
            let (rule, guard) = auto_wants(
                &reading,
                host,
                now,
                gib(4.0),
                target,
                MEMORY_RESERVE_PER_G,
                1_000_000_000,
                gib(0.25),
            );
            assert!(!rule && guard, "now {now}: rule {rule} guard {guard}");
        }
        // Under the target the guard keeps it too.
        let host = decision_bytes(&reading, None, false);
        let (rule, guard) = auto_wants(
            &reading,
            host,
            false,
            0,
            target,
            MEMORY_RESERVE_PER_G,
            1_000_000_000,
            gib(0.25),
        );
        assert!(!rule && !guard);
        // The walk's end: 70 GiB netted forecasts under the line, 95 really
        // held is past it.
        assert_eq!(hand_off_need(gib(70.0), 100_000_000, target), 0);
        assert!(real_hand_off_need(gib(95.0), target) > 0);
        // The large block (25481021) with a 56 GiB pool, armed: neither.
        let large = pooled(gib(59.1), gib(61.0), 0, gib(27.0));
        let host = decision_bytes(&large, None, false);
        let (rule, guard) = auto_wants(
            &large,
            host,
            false,
            gib(10.0),
            target,
            MEMORY_RESERVE_PER_G,
            29_000_000_000,
            64 << 20,
        );
        assert!(!guard, "the guard on the large: rule {rule}");
        assert_eq!(real_hand_off_need(gib(86.7), target), 0);
    }

    /// The target is the smaller of the cgroup limit and `MemTotal`, less
    /// 10 GiB: v1's unlimited sentinel gives `MemTotal`'s, a limit above
    /// `MemTotal` is capped, and with neither there is no target.
    #[test]
    fn the_memory_target_is_the_smaller_limit_less_ten_gib() {
        let total = 62_840_956u64 << 10;
        assert_eq!(
            memory_target_from(Some(61_774_757_888), Some(total)),
            61_774_757_888 - 10 * GIB
        );
        assert_eq!(
            memory_target_from(Some(9_223_372_036_854_771_712), Some(total)),
            total - 10 * GIB
        );
        assert_eq!(
            memory_target_from(Some(200 * GIB), Some(125 * GIB)),
            115 * GIB
        );
        assert_eq!(memory_target_from(None, Some(64 * GIB)), 54 * GIB);
        assert_eq!(
            memory_target_from(Some(129_584_070_656), None),
            129_584_070_656 - 10 * GIB
        );
        assert_eq!(memory_target_from(Some(8 * GIB), Some(16 * GIB)), 0);
        assert_eq!(memory_target_from(None, None), u64::MAX);
    }

    /// The `auto` rule wants a table out only once the host, its reserve and
    /// the table would pass the target.
    #[test]
    fn the_auto_rule_wants_a_table_out_only_past_the_target() {
        let target = 110 * GIB;
        let cells = 10_000_000_000; // 10 G cells: a reserve of 6 + 16.5 GiB
        assert_eq!(
            memory_reserve_bytes(cells, MEMORY_RESERVE_PER_G),
            (22.5 * GIB as f64) as u64
        );
        // The knob's retune: 0.5 GiB per G cells is 6 + 5 GiB at 10 G cells;
        // unset or not a number ≥ 0 is today's 1.65.
        assert_eq!(memory_reserve_bytes(cells, 0.5), 11 * GIB);
        assert_eq!(parse_reserve_per_g(Some(" 0.5 ")), 0.5);
        assert_eq!(parse_reserve_per_g(Some("0")), 0.0);
        for bad in [None, Some("-1"), Some("nan"), Some("inf"), Some("x")] {
            assert_eq!(parse_reserve_per_g(bad), MEMORY_RESERVE_PER_G, "{bad:?}");
        }
        let wanted =
            |target: u64, per_g: f64, host: u64| drop_wanted(target, per_g, cells, GIB, || host);
        assert!(!wanted(target, MEMORY_RESERVE_PER_G, 86 * GIB));
        assert!(wanted(target, MEMORY_RESERVE_PER_G, 87 * GIB));
        assert!(!wanted(u64::MAX, MEMORY_RESERVE_PER_G, u64::MAX));
        // At 0.5 GiB per G cells the same table waits until 98 GiB is passed.
        assert!(!wanted(target, 0.5, 98 * GIB));
        assert!(wanted(target, 0.5, 99 * GIB));
    }

    /// The purge's memory pressure on #1014: the host past 85 % of the memory
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
        let small = memory_target_from(None, Some(64 * GIB));
        assert!(!memory_short(gib(45.8), small));
        assert!(memory_short(gib(46.0), small));
        assert!(!memory_short(1024 * GIB, u64::MAX));
    }

    /// What a block with live regeneration gives the allocator's two boundary
    /// purges to weigh:
    /// the pressure line of its target, 85 % of it and not the target, and the
    /// regeneration window's 16 GiB. On ZEN5 (target 108.28 GiB) the line is
    /// 92.03 and the p90 block has room at both boundaries (VmRSS 66.75 and
    /// 68.17 GiB, ZEN5 096); on a 64 GiB host (target 54) the line is 45.90 and
    /// neither p90 (36.08, 45.60) nor the median (33.13, 33.16) has room: under
    /// the target itself the median would have.
    #[test]
    fn a_no_disk_block_weighs_its_window_against_the_pressure_line() {
        let gib = |g: f64| (g * GIB as f64) as u64;
        let zen5 = 126_997_233_664 - 10 * GIB;
        let large = no_disk_room(true, zen5);
        assert_eq!(
            large,
            Some(NoDisk {
                line: pressure_line(zen5),
                headroom: 16 * GIB
            })
        );
        assert_eq!(pressure_line(zen5), zen5 / 20 * 17);
        assert!(memory_short(pressure_line(zen5) + 1, zen5));
        assert!(!memory_short(pressure_line(zen5), zen5));
        for resident in [66.75, 68.17] {
            assert!(has_room(large, Some(gib(resident))), "ZEN5 {resident}");
        }
        let small = no_disk_room(true, 54 * GIB);
        assert_eq!(small.map(|n| n.line), Some(54 * GIB / 20 * 17));
        assert!(pressure_line(54 * GIB).abs_diff(gib(45.9)) < 1 << 20);
        for resident in [36.08, 45.60, 33.13, 33.16] {
            assert!(!has_room(small, Some(gib(resident))), "64 GiB {resident}");
        }
    }

    /// A block without live regeneration gives the purges nothing to weigh:
    /// they run whenever its memory is short, whatever its resident set.
    #[test]
    fn a_block_without_live_regeneration_gives_the_purges_no_line() {
        assert_eq!(no_disk_room(false, 54 * GIB), None);
        assert!(!has_room(no_disk_room(false, 54 * GIB), Some(GIB)));
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
        let small = memory_target_from(None, Some(64 * GIB));
        assert!(hand_off_need(gib(40.0), windows(285), small) > 0);
    }

    /// `LAMBDA_VM_BLOCK_TREE_DROP_LEVELS`: unset or empty is 8 on the card and
    /// 4 on the host; `<card>` or `<card>,<host>` sets them; anything else is
    /// refused.
    #[test]
    fn the_tree_drop_knob_reads_card_then_host_and_refuses_the_rest() {
        use multilinear::whir_commit::TreeDrop;
        let default = TreeDrop { device: 8, host: 4 };
        assert_eq!(parse_tree_drop(None), Ok(default));
        assert_eq!(parse_tree_drop(Some("")), Ok(default));
        assert_eq!(
            parse_tree_drop(Some("6")),
            Ok(TreeDrop { device: 6, host: 4 })
        );
        assert_eq!(
            parse_tree_drop(Some(" 6 , 5 ")),
            Ok(TreeDrop { device: 6, host: 5 })
        );
        assert_eq!(parse_tree_drop(Some("4,4")), Ok(TreeDrop::uniform(4)));
        for bad in ["x", "6,5,1", "6,", ",4", "-1", "6;5"] {
            assert!(parse_tree_drop(Some(bad)).is_err(), "{bad} must be refused");
        }
    }
}

/// The knob of [`phase_b_many_first`].
const PHASE_B_MANY_FIRST_ENV: &str = "LAMBDA_VM_PHASE_B_MANY_FIRST";

/// Whether phase B takes the group of the most tables first: a deviation's
/// choice, else `LAMBDA_VM_PHASE_B_MANY_FIRST` (off by default). Every order
/// proves the same bytes ([`multilinear_block::block_prove_in_order`]); what
/// moves is when each group's parts reach a level proved beside phase B. The
/// group of the most tables (the page tables', hundreds of them) is the one a
/// level takes longest to land; taken first, it lands while phase B still
/// runs instead of after it ends.
fn phase_b_many_first(deviation: Option<bool>) -> Result<bool, Error> {
    match deviation {
        Some(on) => Ok(on),
        None => parse_phase_b_many_first(std::env::var(PHASE_B_MANY_FIRST_ENV).ok().as_deref()),
    }
}

/// `LAMBDA_VM_PHASE_B_MANY_FIRST`: unset, empty or `0` (the default) keeps
/// phase B's own order; `1` takes the group of the most tables first.
/// Anything else refuses the prove.
fn parse_phase_b_many_first(value: Option<&str>) -> Result<bool, Error> {
    match value.map(str::trim) {
        None | Some("" | "0") => Ok(false),
        Some("1") => Ok(true),
        Some(v) => Err(Error::Prover(format!(
            "{PHASE_B_MANY_FIRST_ENV}={v}: 0 (the default: phase B's own order) or 1 (the group of \
             the most tables first)"
        ))),
    }
}

/// The group of the most tables (the first such, by index), or `None` with
/// no group.
fn group_of_most_tables(groups: &[Vec<u32>]) -> Option<usize> {
    groups
        .iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.len().cmp(&b.len()).then(j.cmp(i)))
        .map(|(g, _)| g)
}

/// `W3_PROBE_FIRST_GROUPS=g,g,…`, a measurement knob unset in production:
/// phase B takes these groups first, in this order, then the others in its own.
/// Every order proves the same bytes ([`multilinear_block::block_prove_in_order`]).
/// A word that is not a list of group indices is an error.
fn probe_first_groups() -> Result<Option<Vec<usize>>, Error> {
    let Ok(value) = std::env::var("W3_PROBE_FIRST_GROUPS") else {
        return Ok(None);
    };
    value
        .split(',')
        .map(|g| g.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
        .map_err(|_| {
            Error::Prover(format!(
                "W3_PROBE_FIRST_GROUPS={value}: group indices, comma-separated"
            ))
        })
}

/// `first`, then the rest of `order` in its order. Refuses a group `order`
/// does not hold or one named twice, so the result is a permutation of `order`.
fn put_first(first: &[usize], order: Vec<usize>) -> Result<Vec<usize>, String> {
    let mut taken = vec![false; order.len()];
    for &g in first {
        match taken.get_mut(g) {
            None => {
                return Err(format!(
                    "W3_PROBE_FIRST_GROUPS: no group {g} of {}",
                    order.len()
                ));
            }
            Some(t) if *t => return Err(format!("W3_PROBE_FIRST_GROUPS: group {g} named twice")),
            Some(t) => *t = true,
        }
    }
    Ok(first
        .iter()
        .copied()
        .chain(order.into_iter().filter(|&g| !taken[g]))
        .collect())
}

/// Phase B's order when live regeneration rebuilds tables: the groups it
/// rebuilds nothing of first, then the first `lead` groups of the streamed
/// chunks' class (class 0), then the others of both classes interleaved by
/// their cells ([`multilinear_block::interleaved_order`]). Each class's groups
/// stay in group order. The rest's regenerator (class 1) starts with phase B
/// and its first group's tables take it about a second and a half, while the
/// streamed chunks' has started before; taken second, the rest's first group
/// waits for them ([`regen::regen_lead`]). With `lead` 0, or no group of the
/// rest's class, it is the interleave's order.
fn led_order(classes: &[Option<usize>], cells: &[u64], lead: usize) -> Vec<usize> {
    let rest = classes.contains(&Some(1));
    let led: Vec<usize> = (0..classes.len())
        .filter(|&g| classes[g] == Some(0))
        .take(if rest { lead } else { 0 })
        .collect();
    if led.is_empty() {
        return multilinear_block::interleaved_order(classes, cells);
    }
    // The interleave puts every group with no class first, in group order:
    // with the led groups' class masked, those are the groups phase B rebuilds
    // nothing of and the led ones. The first kind goes first.
    let mut masked = classes.to_vec();
    for &g in &led {
        masked[g] = None;
    }
    let order = multilinear_block::interleaved_order(&masked, cells);
    let front = masked
        .iter()
        .filter(|c| c.is_none())
        .count()
        .min(order.len());
    let (unclassed, interleaved) = order.split_at(front);
    unclassed
        .iter()
        .copied()
        .filter(|&g| classes[g].is_none())
        .chain(led.iter().copied())
        .chain(interleaved.iter().copied())
        .collect()
}

#[cfg(test)]
mod lead_order_tests {
    use super::{LeadRule, PhaseBLead, class_fits, lead_order, lead_order_n, lead_order_n_rules};

    /// Each class's groups in `order`, in the order taken.
    fn by_class(order: &[usize], classes: &[Option<usize>], class: usize) -> Vec<usize> {
        order
            .iter()
            .copied()
            .filter(|&g| classes[g] == Some(class))
            .collect()
    }

    /// The leaf with the fewest cells still to take leads (ties to the lower
    /// leaf); only the groups still to come count; its groups go first in
    /// their order, the others keep theirs; the result is a permutation.
    #[test]
    fn the_leaf_with_the_fewest_cells_left_leads() {
        let cells = [10, 10, 10, 10, 10, 10, 1, 1];
        let none = [None; 8];
        let leaves = vec![vec![0, 3, 6], vec![1, 4, 7], vec![2, 5]];
        // All to come: leaf 2 holds 20 cells, leaves 0 and 1 hold 21 each.
        let rest: Vec<usize> = (0..8).collect();
        assert_eq!(
            lead_order(&leaves, &rest, &none, &cells),
            Some((2, vec![2, 5], vec![2, 5, 0, 1, 3, 4, 6, 7]))
        );
        // Groups 0 and 3 taken: leaf 0 has one cell left.
        let rest = [1, 2, 4, 5, 6, 7];
        assert_eq!(
            lead_order(&leaves, &rest, &none, &cells),
            Some((0, vec![6], vec![6, 1, 2, 4, 5, 7]))
        );
        // A tie goes to the lower leaf; the moved groups keep `rest`'s order.
        let even = [5u64; 8];
        let rest = [7, 6, 5, 4, 3, 2, 1, 0];
        assert_eq!(
            lead_order(&leaves, &rest, &none, &even),
            Some((2, vec![5, 2], vec![5, 2, 7, 6, 4, 3, 1, 0]))
        );
        // Groups the plan holds but phase B has not (out of range) are ignored.
        let leaves_wide = vec![vec![1, 40, 41]];
        assert_eq!(
            lead_order(&leaves_wide, &[0, 1, 2], &none, &cells),
            Some((0, vec![1], vec![1, 0, 2]))
        );
    }

    /// A leaf with a group still to come that phase B rebuilds cannot lead
    /// (one whose rebuilt group is already taken can); every class keeps its
    /// own order; with no leaf able to lead, nothing moves.
    #[test]
    fn a_leaf_with_a_rebuilt_group_to_come_never_leads() {
        let cells = [1, 1, 1, 1, 9, 9, 9, 9];
        let classes = [None, None, None, None, Some(1), Some(0), Some(1), Some(0)];
        // Leaf 0 is the lightest but holds group 6, rebuilt and to come.
        let leaves = vec![vec![0, 6], vec![1, 2, 4], vec![3, 5, 7]];
        let rest: Vec<usize> = (0..8).collect();
        assert_eq!(lead_order(&leaves, &rest, &classes, &cells), None);
        let leaves = vec![vec![0, 6], vec![1, 2], vec![3, 5, 7]];
        let (leaf, moved, now) =
            lead_order(&leaves, &rest, &classes, &cells).expect("leaf 1 leads");
        assert_eq!((leaf, moved), (1, vec![1, 2]));
        for class in 0..2 {
            assert_eq!(
                by_class(&now, &classes, class),
                by_class(&rest, &classes, class),
                "class {class}"
            );
        }
        // Group 6 taken already: leaf 0 (one cell left) leads.
        let rest = [0, 1, 2, 3, 4, 5, 7];
        assert_eq!(
            lead_order(&leaves, &rest, &classes, &cells),
            Some((0, vec![0], vec![0, 1, 2, 3, 4, 5, 7]))
        );
        // A leaf whose groups are all taken cannot lead.
        let rest = [1, 2, 3, 4, 5, 7];
        assert_eq!(
            lead_order(&leaves, &rest, &classes, &cells),
            Some((1, vec![1, 2], vec![1, 2, 3, 4, 5, 7]))
        );
        // Every leaf's groups taken, or none offered: nothing leads.
        assert_eq!(lead_order(&leaves, &[], &classes, &cells), None);
        assert_eq!(lead_order(&[], &[0, 1], &classes, &cells), None);
    }

    /// Several leaves lead one after another, each by the single lead's rule
    /// over what the ones before it left; one leaf is exactly `lead_order`;
    /// a leaf with a rebuilt group to come is skipped, not waited for; the
    /// result is a permutation of `rest` and every class keeps its order.
    #[test]
    fn several_leaves_lead_one_after_another() {
        let cells = [10, 10, 10, 10, 10, 10, 1, 1];
        let none = [None; 8];
        let leaves = vec![vec![0, 3, 6], vec![1, 4, 7], vec![2, 5]];
        let rest: Vec<usize> = (0..8).collect();
        for r in [
            rest.clone(),
            vec![1, 2, 4, 5, 6, 7],
            vec![7, 6, 5, 4, 3, 2, 1, 0],
        ] {
            let one = lead_order(&leaves, &r, &none, &cells).map(|(k, m, n)| (vec![k], m, n));
            assert_eq!(
                lead_order_n(&leaves, &r, &none, &cells, 1),
                one,
                "n 1 is lead_order"
            );
        }
        // Leaf 2 (20 cells), then leaf 0 (21, the lower of the tie), then leaf 1.
        let (led, moved, now) = lead_order_n(&leaves, &rest, &none, &cells, 3).expect("leads");
        assert_eq!(led, vec![2, 0, 1]);
        assert_eq!(moved, vec![2, 5, 0, 3, 6, 1, 4, 7]);
        assert_eq!(now, moved, "every group belongs to a leaf");
        // Two of three: the third leaf's groups keep their order after.
        let (led, _, now) = lead_order_n(&leaves, &rest, &none, &cells, 2).expect("leads");
        assert_eq!((led, now), (vec![2, 0], vec![2, 5, 0, 3, 6, 1, 4, 7]));
        // More than there are leaves: all of them, once each.
        let (led, ..) = lead_order_n(&leaves, &rest, &none, &cells, 9).expect("leads");
        assert_eq!(led, vec![2, 0, 1]);
        // Rebuilt groups to come: a leaf leads only when its rebuilt groups are
        // next in their class, given the leaves before it; each class keeps
        // group order.
        let classes = [None, None, None, None, Some(1), Some(0), Some(1), Some(0)];
        let leaves = vec![vec![0, 6], vec![1, 2], vec![3, 5, 7], vec![4]];
        let (led, _, now) = lead_order_n(&leaves, &rest, &classes, &cells, 4).expect("leads");
        // Leaf 3 first (10 cells; its group 4 is class 1's first); then leaf 0
        // (11 cells; its group 6 is class 1's next once 4 is gone), leaf 1 (20)
        // and leaf 2 (21; 5 and 7 are all of class 0).
        assert_eq!(led, vec![3, 0, 1, 2]);
        let mut sorted = now.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, rest, "a permutation");
        for class in 0..2 {
            assert_eq!(
                by_class(&now, &classes, class),
                by_class(&rest, &classes, class)
            );
        }
        assert_eq!(lead_order_n(&leaves, &[], &classes, &cells, 3), None);
    }

    /// The single lead keeps today's rule: a rebuilt group to come bars a leaf
    /// even when it is next in its class; leading several, such a leaf may lead,
    /// one whose rebuilt group has a lower one of its class elsewhere may not;
    /// over many random cases every class stays in group order and the result
    /// is a permutation.
    #[test]
    fn a_leaf_may_lead_when_its_rebuilt_groups_are_next_in_their_class() {
        let cells = [1u64; 8];
        let classes = [None, None, None, None, Some(0), Some(0), Some(0), None];
        let rest: Vec<usize> = (0..8).collect();
        // Leaf 0 holds group 4, the class's first; leaf 1 holds 6, behind leaf 2's 5.
        let leaves = vec![vec![0, 4], vec![1, 6], vec![2, 5], vec![3, 7]];
        assert_eq!(
            lead_order(&leaves, &rest, &classes, &cells).map(|x| x.0),
            Some(3)
        );
        assert_eq!(
            lead_order_n(&leaves, &rest, &classes, &cells, 1).map(|x| x.0),
            Some(vec![3])
        );
        let (led, _, now) = lead_order_n(&leaves, &rest, &classes, &cells, 4).expect("leads");
        assert_eq!(
            led,
            vec![0, 2, 1, 3],
            "by cells, each when its rebuilt group is next"
        );
        assert_eq!(by_class(&now, &classes, 0), vec![4, 5, 6]);
        // Two rebuilt groups of one class held crosswise by two leaves: neither
        // can go first, so only the plain leaves lead.
        let classes = [None, None, Some(0), Some(0), Some(0), Some(0)];
        let leaves = vec![vec![0], vec![2, 5], vec![3, 4], vec![1]];
        let (led, _, now) =
            lead_order_n(&leaves, &(0..6).collect::<Vec<_>>(), &classes, &[1; 6], 4)
                .expect("leads");
        assert_eq!(
            led,
            vec![0, 3],
            "leaves 1 and 2 interlock: no class order puts either first"
        );
        assert_eq!(by_class(&now, &classes, 0), vec![2, 3, 4, 5]);
        // Random cases: a permutation, and every class in group order.
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..500 {
            let n = 4 + (next() % 12) as usize;
            let classes: Vec<Option<usize>> = (0..n)
                .map(|_| match next() % 4 {
                    0 => Some(0),
                    1 => Some(1),
                    _ => None,
                })
                .collect();
            let k = 1 + (next() % 4) as usize;
            let mut leaves = vec![Vec::new(); k];
            for g in 0..n {
                leaves[(next() % k as u64) as usize].push(g);
            }
            let cells: Vec<u64> = (0..n).map(|_| 1 + next() % 9).collect();
            let start = (next() % n as u64) as usize;
            let rest: Vec<usize> = (start..n).collect();
            for lead_n in 1..=k {
                if let Some((_, moved, now)) =
                    lead_order_n(&leaves, &rest, &classes, &cells, lead_n)
                {
                    let mut sorted = now.clone();
                    sorted.sort_unstable();
                    assert_eq!(sorted, rest, "a permutation");
                    assert!(now.starts_with(&moved));
                    for c in 0..2 {
                        let order = by_class(&now, &classes, c);
                        assert!(
                            order.windows(2).all(|w| w[0] < w[1]),
                            "class {c} in order: {now:?}"
                        );
                    }
                }
            }
        }
    }

    /// A class fits its window when all its dropped bytes do, against the
    /// window's floor; a class with no floor never fits.
    #[test]
    fn a_class_fits_its_window_when_all_its_dropped_bytes_do() {
        assert_eq!(
            class_fits(&[(2, 100), (5, 4096), (1, 4097)], &[100, 4096, 4096]),
            vec![true, true, false]
        );
        assert_eq!(class_fits(&[(1, 1)], &[]), vec![false], "no floor, no fit");
        assert_eq!(class_fits(&[], &[1]), Vec::<bool>::new());
    }

    /// Leading several, a leaf whose rebuilt groups are out of class order
    /// leads by the window rule only when its class fits; the rule each leaf
    /// led by is recorded; leading one, nothing changes.
    #[test]
    fn the_window_rule_lets_a_fitting_class_go_out_of_group_order() {
        let classes = [None, None, None, Some(1), Some(1), Some(1)];
        // Leaf 0 holds the class's last group and the fewest cells.
        let leaves = vec![vec![0, 5], vec![1, 4], vec![2, 3]];
        let cells = [1u64, 10, 10, 10, 10, 1];
        let rest: Vec<usize> = (0..6).collect();
        let no = [false, false];
        let (led, rules, ..) =
            lead_order_n_rules(&leaves, &rest, &classes, &cells, 3, &no).expect("leads");
        assert_eq!(led, vec![2, 1, 0]);
        assert_eq!(rules, vec![LeadRule::ClassNext; 3]);
        let yes = [false, true];
        let (led, rules, _, now) =
            lead_order_n_rules(&leaves, &rest, &classes, &cells, 3, &yes).expect("leads");
        assert_eq!(led, vec![0, 1, 2]);
        assert_eq!(
            rules,
            vec![LeadRule::Window, LeadRule::Window, LeadRule::ClassNext]
        );
        assert_eq!(
            by_class(&now, &classes, 1),
            vec![5, 4, 3],
            "out of group order"
        );
        // Leading one: today's rule, which no fit relaxes.
        assert_eq!(
            lead_order_n_rules(&leaves, &rest, &classes, &cells, 1, &yes).map(|x| x.0),
            None
        );
        // A plain leaf is labelled plain.
        let (_, rules, ..) =
            lead_order_n_rules(&[vec![0, 1]], &rest, &classes, &cells, 2, &yes).expect("leads");
        assert_eq!(rules, vec![LeadRule::Plain]);
    }

    /// The lead's take order against a real regeneration window: its
    /// depositor in rank (group) order, its taker in the lead's order, every
    /// wait bounded. A class larger than its window keeps group order and
    /// every take completes; a class that fits goes out of order and still
    /// completes. With the size check dropped (every class "fits"), the class
    /// larger than its window is taken out of order and the taker waits on a
    /// deposit the window holds back: this test fails within its bound.
    #[test]
    fn a_class_larger_than_its_window_keeps_group_order_and_never_deadlocks() {
        use multilinear::narrow::NarrowColumns;
        use stark::regen::RegenWindow;
        use std::sync::mpsc;
        use std::time::Duration;
        let cols = |seed: u64| {
            let words: Vec<u64> = (0..3000u64).map(|i| (i ^ seed) % 60_000).collect();
            NarrowColumns::pack_row_major(&words, 3).expect("packs")
        };
        let bytes = {
            let (w, _p) = RegenWindow::new(0);
            w.slot(&cols(0), 0).len() as u64
        };
        let classes = [None, None, None, Some(1), Some(1), Some(1)];
        let leaves = vec![vec![0, 5], vec![1, 4], vec![2, 3]];
        let cells = [1u64, 10, 10, 10, 10, 1];
        let rest: Vec<usize> = (0..6).collect();
        // The rebuilt groups' take order for a window of `floor` bytes, and
        // whether every take completed within the bound.
        let run = |floor: u64| -> (Vec<usize>, bool) {
            let fits = class_fits(&[(0, 0), (3, 3 * bytes)], &[u64::MAX, floor]);
            let (_, _, _, now) =
                lead_order_n_rules(&leaves, &rest, &classes, &cells, 3, &fits).expect("leads");
            let takes: Vec<usize> = now.into_iter().filter(|&g| classes[g].is_some()).collect();
            let (window, producer) = RegenWindow::new(floor);
            let slots: Vec<_> = (0..3u64).map(|r| window.slot(&cols(r), r)).collect();
            let deposits = slots.clone();
            let depositor = std::thread::spawn(move || {
                for (r, slot) in deposits.iter().enumerate() {
                    let _ = slot.deposit(cols(r as u64));
                }
                drop(producer);
            });
            let (tx, rx) = mpsc::channel();
            let order = takes.clone();
            let taker = std::thread::spawn(move || {
                let ok = order.iter().all(|&g| {
                    let slot = &slots[g - 3];
                    slot.wait();
                    slot.take().is_ok()
                });
                let _ = tx.send(ok);
            });
            let done = rx.recv_timeout(Duration::from_secs(10)).unwrap_or(false);
            // Closing the window wakes every waiter and depositor, so both
            // threads end whatever happened.
            window.close("the test's end");
            let _ = depositor.join();
            let _ = taker.join();
            (takes, done)
        };
        let (takes, done) = run(2 * bytes);
        assert_eq!(
            takes,
            vec![3, 4, 5],
            "a class larger than its window keeps group order"
        );
        assert!(done, "every take completes with the class in group order");
        let (takes, done) = run(4 * bytes);
        assert_eq!(
            takes,
            vec![5, 4, 3],
            "a class that fits goes out of group order"
        );
        assert!(done, "and every take still completes");
    }

    /// A lead made for several leaves records them all, the first as `leaf`.
    #[test]
    fn a_lead_for_several_leaves_records_them_in_order() {
        let cells = [3, 3, 3, 3, 1, 1];
        let none = [None; 6];
        let order: Vec<usize> = (0..6).collect();
        let lead = PhaseBLead::leading(2);
        assert!(lead.offer(vec![vec![0, 2], vec![1, 4, 5], vec![3]]));
        // Turn 1: leaf 1 has 2 cells left, leaf 0 three, leaf 2 three (tie to 0).
        assert_eq!(
            lead.turn(1, &order, &none, &cells, 0.5, &[]),
            Some(vec![4, 5, 2, 3])
        );
        let taken = lead.taken().expect("read");
        assert_eq!((taken.leaf, taken.leaves.clone()), (Some(1), vec![1, 0]));
        assert_eq!(taken.moved, vec![4, 5, 2]);
        assert_eq!(taken.order, vec![0, 1, 4, 5, 2, 3]);
    }

    /// The lead reads the offer once: nothing before it, the re-ranking at
    /// the first turn after it (recorded with the order taken), nothing after;
    /// a lead whose order is phase B's own is recorded but changes nothing.
    #[test]
    fn the_lead_reads_the_offer_once() {
        let cells = [3, 3, 3, 3, 1, 1];
        let none = [None; 6];
        let order: Vec<usize> = (0..6).collect();
        let lead = PhaseBLead::default();
        assert_eq!(lead.turn(0, &order, &none, &cells, 0.0, &[]), None);
        assert_eq!(lead.taken(), None);
        assert!(lead.offer(vec![vec![0, 2], vec![1, 4, 5]]));
        assert!(!lead.offer(vec![vec![3]]), "only the first offer counts");
        // At turn 1 (groups 0 and 1 taken): leaf 0 has 3 cells left, leaf 1 two.
        assert_eq!(
            lead.turn(1, &order, &none, &cells, 0.5, &[]),
            Some(vec![4, 5, 2, 3])
        );
        let taken = lead.taken().expect("read");
        assert_eq!(taken.turn, 1);
        assert_eq!(taken.leaf, Some(1));
        assert_eq!(taken.moved, vec![4, 5]);
        assert_eq!(taken.order, vec![0, 1, 4, 5, 2, 3]);
        assert_eq!(taken.after, 0.5);
        let now = [0, 1, 4, 5, 2, 3];
        assert_eq!(lead.turn(2, &now, &none, &cells, 0.6, &[]), None);
        assert_eq!(lead.taken(), Some(taken));
        // Phase B's own order already leads with the soonest leaf: no change.
        let cells = [1, 1, 1, 9, 9, 9];
        let lead = PhaseBLead::default();
        assert!(lead.offer(vec![vec![1, 2], vec![3, 4, 5]]));
        assert_eq!(lead.turn(0, &order, &none, &cells, 0.0, &[]), None);
        let taken = lead.taken().expect("read");
        assert_eq!((taken.leaf, taken.order), (Some(0), order.clone()));
    }
}

#[cfg(test)]
mod led_order_tests {
    use super::{led_order, multilinear_block, put_first};

    /// The probe's groups go first in the order named, the rest keep phase B's
    /// order; a group out of range or named twice is refused.
    #[test]
    fn named_groups_go_first_and_the_rest_keep_their_order() {
        assert_eq!(
            put_first(&[3, 1], vec![0, 1, 2, 3, 4]),
            Ok(vec![3, 1, 0, 2, 4])
        );
        assert_eq!(put_first(&[], vec![2, 0, 1]), Ok(vec![2, 0, 1]));
        assert_eq!(put_first(&[0, 2], vec![2, 1, 0]), Ok(vec![0, 2, 1]));
        assert!(put_first(&[5], vec![0, 1, 2]).is_err());
        assert!(put_first(&[1, 1], vec![0, 1, 2]).is_err());
    }

    const S: Option<usize> = Some(0);
    const R: Option<usize> = Some(1);

    /// The groups phase B rebuilds nothing of, then `lead` streamed groups,
    /// then the interleave of what is left: the rest's first group comes after
    /// `lead` + 1 streamed ones (the interleave's tie goes to the streamed
    /// class).
    #[test]
    fn the_rests_first_group_follows_the_led_streamed_groups() {
        let classes = [None, S, S, S, S, S, S, R, R, None];
        let ones = [1u64; 10];
        assert_eq!(
            led_order(&classes, &ones, 2),
            vec![0, 9, 1, 2, 3, 7, 4, 5, 8, 6]
        );
        // A lead of 0, and a block with no rest group, are the interleave's.
        assert_eq!(
            led_order(&classes, &ones, 0),
            multilinear_block::interleaved_order(&classes, &ones)
        );
        let no_rest = [None, S, S, S];
        assert_eq!(
            led_order(&no_rest, &[1; 4], 3),
            multilinear_block::interleaved_order(&no_rest, &[1; 4])
        );
        // A lead past the class takes all of it, then the rest's.
        assert_eq!(
            led_order(&[R, S, None, S, R], &[1; 5], 9),
            vec![2, 1, 3, 0, 4]
        );
        // Nothing rebuilt: the group order.
        assert_eq!(led_order(&[None; 3], &[], 4), vec![0, 1, 2]);
    }

    /// Whatever the lead and the weights: every group once, and each class's
    /// groups in group order (their regenerator deposits in that order).
    #[test]
    fn a_led_order_takes_every_group_once_and_each_class_in_order() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..200 {
            let n = 1 + (next() % 40) as usize;
            let classes: Vec<Option<usize>> = (0..n)
                .map(|_| match next() % 5 {
                    0 => None,
                    1..=3 => S,
                    _ => R,
                })
                .collect();
            let cells: Vec<u64> = (0..n).map(|_| 1 + next() % 1000).collect();
            let lead = (next() % 8) as usize;
            let order = led_order(&classes, &cells, lead);
            let mut seen = order.clone();
            seen.sort_unstable();
            assert_eq!(seen, (0..n).collect::<Vec<_>>(), "{classes:?} lead {lead}");
            for class in [S, R] {
                let taken: Vec<usize> = order
                    .iter()
                    .copied()
                    .filter(|&g| classes[g] == class)
                    .collect();
                assert!(
                    taken.windows(2).all(|w| w[0] < w[1]),
                    "{classes:?} lead {lead}"
                );
            }
            // The groups phase B rebuilds nothing of come first.
            let plain = classes.iter().filter(|c| c.is_none()).count();
            assert!(order[..plain].iter().all(|&g| classes[g].is_none()));
            // With a rest group, the streamed ones ahead of it are the lead's
            // (or all there are), and one more by the interleave's tie.
            if classes.contains(&R) {
                let ahead = order[plain..]
                    .iter()
                    .take_while(|&&g| classes[g] == S)
                    .count();
                let streamed = classes.iter().filter(|c| **c == S).count();
                assert!(
                    ahead >= lead.min(streamed),
                    "{classes:?} lead {lead}: {order:?}"
                );
            }
        }
    }
}

/// Each group's cells, from its own tables: `shapes` is in AIR order and a
/// group lists its tables' AIR indices, which phase A's packer does not keep
/// contiguous (a group of streamed CPU and MEMW_R chunks sits between rest
/// families in AIR order). Phase B's interleave and its readout weigh the
/// groups by these.
fn group_cells(shapes: &[(usize, usize)], groups: &[Vec<u32>]) -> Vec<u64> {
    groups
        .iter()
        .map(|group| {
            group
                .iter()
                .filter_map(|&i| shapes.get(i as usize))
                .map(|&(w, n)| (w as u64) << n)
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod group_cells_tests {
    use super::group_cells;

    /// A group weighs its own tables, wherever they sit in AIR order: here
    /// group 0 holds AIRs 3 and 1, group 1 AIRs 0, 4 and 2.
    #[test]
    fn each_group_weighs_its_own_tables_in_any_air_order() {
        let shapes = [(1, 4), (2, 10), (3, 0), (5, 20), (7, 1)];
        let groups = vec![vec![3, 1], vec![0, 4, 2]];
        assert_eq!(
            group_cells(&shapes, &groups),
            vec![(5 << 20) + (2 << 10), (1 << 4) + (7 << 1) + 3]
        );
    }

    /// Groups in AIR order weigh the same as before, and an index past the
    /// shapes adds nothing (no panic).
    #[test]
    fn contiguous_groups_and_a_stray_index() {
        let shapes = [(4, 2), (1, 3), (2, 1)];
        assert_eq!(group_cells(&shapes, &[vec![0, 1], vec![2]]), vec![24, 4]);
        assert_eq!(group_cells(&shapes, &[vec![2, 9]]), vec![4]);
    }
}

#[cfg(test)]
mod rest_order_tests {
    use super::RestOrder;

    fn order() -> RestOrder {
        RestOrder::new(vec![21, 6, 38, 38, 17], false)
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
        let mut o = RestOrder::new(vec![21], false);
        o.admit(0, Some(21)).expect("0");
        let err = o.admit(1, Some(6)).expect_err("past the AIRs");
        assert!(
            format!("{err:?}").contains("past the run's 1 tables"),
            "{err:?}"
        );
    }

    /// With BITWISE handed on last, the rest is due in AIR order but its first
    /// table (BITWISE), which comes after every other: that order is taken, AIR
    /// order is refused at BITWISE's turn, and without the flag BITWISE late is
    /// refused.
    #[test]
    fn with_bitwise_last_the_rest_is_due_in_air_order_but_bitwise() {
        let mut o = RestOrder::new(vec![21, 6, 38, 38, 17], true);
        for (position, width) in [
            (1, Some(6)),
            (2, None),
            (3, Some(38)),
            (4, Some(17)),
            (0, Some(21)),
        ] {
            o.admit(position, width).expect("BITWISE last");
        }
        o.finish().expect("every AIR reached");
        let mut o = RestOrder::new(vec![21, 6, 38, 38, 17], true);
        let err = o.admit(0, Some(21)).expect_err("BITWISE is due last");
        assert!(
            format!("{err:?}").contains("came where table 1 was due"),
            "{err:?}"
        );
        let mut o = order();
        let err = o.admit(1, Some(6)).expect_err("BITWISE is due first");
        assert!(
            format!("{err:?}").contains("came where table 0 was due"),
            "{err:?}"
        );
    }
}

#[cfg(test)]
mod keccak_rnd_first_tests {
    use super::{RestOrder, keccak_rnd_first_or_default, keccak_rnd_positions};

    fn counts(copy: usize, keccak: usize, keccak_rnd: usize) -> crate::TableCounts {
        crate::TableCounts {
            cpu: 1,
            lt: 0,
            memw: 0,
            memw_aligned: 0,
            load: 0,
            mul: 0,
            dvrm: 0,
            shift: 0,
            branch: 0,
            memw_register: 0,
            eq: 0,
            bytewise: 0,
            store: 0,
            cpu32: 0,
            keccak,
            keccak_rnd,
            ecsm: 0,
            ecdas: 0,
            hint: 0,
            copy,
            blake3: 1,
            is_wh: 1,
            ecrec: 0,
        }
    }

    /// The rest's AIR names for `counts`, as far as KECCAK_RND's tables and
    /// one table past them: the five fixed tables, COPY, KECCAK, KECCAK_RND.
    fn names(copy: usize, keccak: usize, keccak_rnd: usize) -> Vec<String> {
        ["BITWISE", "DECODE", "KECCAK_RC", "REGISTER", "HALT"]
            .iter()
            .map(|n| n.to_string())
            .chain((0..copy).map(|i| format!("COPY[{i}]")))
            .chain((0..keccak).map(|i| format!("KECCAK[{i}]")))
            .chain((0..keccak_rnd).map(|i| format!("KECCAK_RND[{i}]")))
            .chain(["ECSM[0]".to_string()])
            .collect()
    }

    /// KECCAK_RND's positions follow the five fixed tables, COPY's and
    /// KECCAK's; a name in that range that is not KECCAK_RND's, or a range
    /// past the AIRs, is refused.
    #[test]
    fn keccak_rnd_positions_are_its_range_and_name_its_airs() {
        let owned = names(4, 1, 3);
        let n: Vec<&str> = owned.iter().map(String::as_str).collect();
        assert_eq!(
            keccak_rnd_positions(&counts(4, 1, 3), &n).ok(),
            Some(vec![10, 11, 12])
        );
        assert_eq!(
            keccak_rnd_positions(&counts(4, 1, 0), &n).ok(),
            Some(Vec::new()),
            "no KECCAK_RND table"
        );
        // One COPY table fewer in the counts: the range starts at a KECCAK.
        let err = keccak_rnd_positions(&counts(3, 1, 3), &n).expect_err("off by one");
        assert!(format!("{err:?}").contains("table 9"), "{err:?}");
        let mut wrong = n.clone();
        wrong[11] = "ECSM[0]";
        let err = keccak_rnd_positions(&counts(4, 1, 3), &wrong).expect_err("a stray name");
        assert!(format!("{err:?}").contains("table 11"), "{err:?}");
        let err = keccak_rnd_positions(&counts(4, 1, 3), &n[..12]).expect_err("past the AIRs");
        assert!(format!("{err:?}").contains("table 12"), "{err:?}");
    }

    /// With KECCAK_RND's positions first and BITWISE last, the rest is due in
    /// that order: it is taken, AIR order is refused at the first KECCAK_RND
    /// table's turn, and without `first` a KECCAK_RND table early is refused.
    #[test]
    fn with_keccak_rnd_first_the_rest_is_due_with_keccak_rnd_first() {
        let widths = vec![21, 6, 38, 38, 17, 9];
        let mut o = RestOrder::with_first(widths.clone(), &[3, 4], true);
        for (position, width) in [
            (3, Some(38)),
            (4, None),
            (1, Some(6)),
            (2, Some(38)),
            (5, Some(9)),
            (0, Some(21)),
        ] {
            o.admit(position, width).expect("KECCAK_RND first");
        }
        o.finish().expect("every AIR reached");
        let mut o = RestOrder::with_first(widths.clone(), &[3, 4], true);
        let err = o.admit(1, Some(6)).expect_err("3 is due");
        assert!(
            format!("{err:?}").contains("came where table 3 was due"),
            "{err:?}"
        );
        let mut o = RestOrder::with_first(widths, &[], true);
        let err = o.admit(3, Some(38)).expect_err("1 is due");
        assert!(
            format!("{err:?}").contains("came where table 1 was due"),
            "{err:?}"
        );
    }

    /// KECCAK_RND first unless a deviation asks for AIR order.
    #[test]
    fn keccak_rnd_first_is_the_default_unless_a_deviation_says_otherwise() {
        assert!(keccak_rnd_first_or_default(None));
        assert!(keccak_rnd_first_or_default(Some(true)));
        assert!(!keccak_rnd_first_or_default(Some(false)));
    }
}

#[cfg(test)]
mod producer_report_tests {
    use super::{
        BLOCK_LAYOUT_AHEAD, BLOCK_LAYOUT_WORKERS, BLOCK_WINDOW_LOG2, BlockOptions, BlockStamps,
        ExecutorStamps, ProducerStamps, StreamTable,
    };
    use super::{
        exec_mode_from, parse_exec_history, parse_exec_history_check, parse_light_walk,
        parse_word_stream,
    };

    /// `LAMBDA_VM_WORD_STREAM`: on unless `0`; any other word refused.
    #[test]
    fn the_word_stream_is_on_unless_turned_off() {
        for (word, on) in [
            (None, true),
            (Some(""), true),
            (Some(" 1 "), true),
            (Some("0"), false),
            (Some(" 0 "), false),
        ] {
            assert_eq!(parse_word_stream(word).ok(), Some(on), "{word:?}");
        }
        for bad in ["2", "on", "true", "-1"] {
            assert!(parse_word_stream(Some(bad)).is_err(), "{bad}");
        }
    }

    /// `LAMBDA_VM_LIGHT_WALK`: the light walk with the walkers routing unless
    /// `1` or `0` (the rollback); any other word refused.
    #[test]
    fn the_light_walk_routes_where_walked_unless_turned_down() {
        for (word, level) in [
            (None, 2),
            (Some(""), 2),
            (Some("2"), 2),
            (Some(" 1 "), 1),
            (Some("0"), 0),
        ] {
            assert_eq!(parse_light_walk(word).ok(), Some(level), "{word:?}");
        }
        for bad in ["3", "on", "light", "-1"] {
            assert!(parse_light_walk(Some(bad)).is_err(), "{bad}");
        }
    }

    /// Record mode on the fast engine unless `LAMBDA_VM_EXEC_HISTORY=0`, the
    /// reference engine named, or no walkers; `1` with the reference engine
    /// or no walkers refused; any other value refused; the check's re-walk
    /// period a count.
    #[test]
    fn history_mode_is_on_unless_turned_off() {
        use executor::vm::engine::EngineKind::{Fast, Reference};
        for (engine, history, walkers, want) in [
            (None, None, 4, Some((Fast, true))),
            (Some(""), Some(" "), 4, Some((Fast, true))),
            (Some("fast"), None, 4, Some((Fast, true))),
            (None, Some("1"), 4, Some((Fast, true))),
            (Some("fast"), Some("1"), 1, Some((Fast, true))),
            (None, Some("0"), 4, Some((Reference, false))),
            (Some("fast"), Some("0"), 4, Some((Fast, false))),
            (Some("reference"), None, 4, Some((Reference, false))),
            (Some("reference"), Some("0"), 4, Some((Reference, false))),
            (None, None, 0, Some((Reference, false))),
            (Some("fast"), None, 0, Some((Fast, false))),
            (Some("reference"), Some("1"), 4, None),
            (None, Some("1"), 0, None),
            (None, Some("on"), 4, None),
            (Some("turbo"), None, 4, None),
        ] {
            let got = exec_mode_from(engine, history, walkers).ok();
            assert_eq!(got, want, "{engine:?} {history:?} {walkers} walkers");
        }
        assert_eq!(parse_exec_history(None).unwrap(), None);
        assert_eq!(parse_exec_history(Some("0")).unwrap(), Some(false));
        assert_eq!(parse_exec_history(Some("1")).unwrap(), Some(true));
        assert!(parse_exec_history(Some("on")).is_err());
        assert_eq!(parse_exec_history_check(None).unwrap(), 0);
        assert_eq!(parse_exec_history_check(Some("16")).unwrap(), 16);
        assert!(parse_exec_history_check(Some("often")).is_err());
    }

    /// Record mode, the default, holds as many cycles as the block holds CPU
    /// rows: the history's last cycle (counted from 0) is the last row's.
    #[test]
    fn record_mode_holds_the_cpu_row_bound() {
        assert_eq!(
            executor::vm::history::HISTORY_MAX_CYCLES + 1,
            crate::tables::cpu::MAX_CPU_ROWS
        );
    }

    /// Production lays the streamed chunks out on six workers with eight
    /// chunks ahead (nine unpacked at most): a permit per worker and three
    /// more, so a worker goes on while the packer waits for a slow chunk.
    #[test]
    fn production_lays_out_on_six_workers_eight_ahead() {
        assert_eq!(BLOCK_LAYOUT_WORKERS, 6);
        assert_eq!(BLOCK_LAYOUT_AHEAD, 8);
        let options = BlockOptions::production();
        assert_eq!(options.layout_workers, BLOCK_LAYOUT_WORKERS);
        assert_eq!(options.layout_ahead, Some(BLOCK_LAYOUT_AHEAD));
        assert_eq!(BLOCK_LAYOUT_AHEAD + 1, BLOCK_LAYOUT_WORKERS + 3);
    }

    /// The window is production's 2^20: another (#1013's one CPU chunk is
    /// 2^21 cycles) builds the same tables, but the chunks leave in another
    /// order across tables, so the groups and the proof bytes differ.
    #[test]
    fn the_window_is_2_20() {
        assert_eq!(BLOCK_WINDOW_LOG2, 20);
        assert_eq!(BlockOptions::production().window_log2, Some(20));
    }

    /// The executor's line: its own work apart from its blocked sends, and
    /// whether its logs were moved or copied; no line without a streamed run.
    #[test]
    fn the_report_says_what_the_executor_did_apart_from_waiting() {
        let mut stamps = BlockStamps {
            streamed: (8.62, 269),
            producer: ProducerStamps {
                executor_send: 3.5,
                ..ProducerStamps::default()
            },
            executor: ExecutorStamps {
                started: 0.02,
                setup: 0.01,
                run: 2.5,
                handoff: 0.04,
                windows: 134,
                ..ExecutorStamps::default()
            },
            ..BlockStamps::default()
        };
        let line = |stamps: &BlockStamps| {
            stamps
                .report()
                .lines()
                .find(|l| l.starts_with("BLOCK EXECUTOR:"))
                .map(str::to_string)
        };
        assert_eq!(
            line(&stamps).as_deref(),
            Some(
                "BLOCK EXECUTOR: started at 0.02s · setup 0.01s · execute 2.50s · hand-off 0.04s \
                 (134 windows, logs moved) · send-blocked 3.50s · own 2.55s"
            )
        );
        stamps.executor = ExecutorStamps::default();
        assert_eq!(line(&stamps), None);
    }

    /// The readout's two producer lines: what each thread waited on, and the
    /// hand-out table by table, only the tables that handed a chunk out (the
    /// box gates parse these lines).
    #[test]
    fn the_report_says_what_the_producer_waited_on() {
        let mut stamps = BlockStamps {
            streamed: (29.23, 558),
            producer: ProducerStamps {
                executor_send: 24.1,
                walker_recv: 0.4,
                walker_send: 11.9,
                accumulator_recv: 1.2,
                accumulator_send: 1.55,
                layout_send: 20.3,
                worker_permit: 40.2,
                worker_recv: 15.7,
            },
            ..BlockStamps::default()
        };
        let h = &mut stamps.windows.handout;
        let (cpu, store) = (StreamTable::Cpu.position(), StreamTable::Store.position());
        (h.take[cpu], h.count[cpu], h.chunks[cpu], h.copied[cpu]) = (6.5, 1.86, 142, 297_795_584);
        (h.take[store], h.count[store], h.chunks[store]) = (0.05, 0.35, 15);
        let report = stamps.report();
        let line = |prefix: &str| {
            report
                .lines()
                .find(|l| l.starts_with(prefix))
                .unwrap_or_else(|| panic!("no {prefix} line in\n{report}"))
                .to_string()
        };
        assert_eq!(
            line("BLOCK PRODUCER:"),
            "BLOCK PRODUCER: executor send-blocked 24.10s · walker recv-wait 0.40s send-blocked \
             11.90s · accumulator recv-wait 1.20s send-blocked 1.55s · layout hand-on blocked \
             20.30s · workers permit-wait 40.20s recv-wait 15.70s (summed)"
        );
        assert_eq!(
            line("BLOCK HANDOUT:"),
            "BLOCK HANDOUT: take 6.55s · count 2.21s · copied 297.8 M ops || per table (take s · \
             count s · chunks · copied ops): CPU 6.50 · 1.86 · 142 · 297.8 M | STORE 0.05 · 0.35 · \
             15 · 0.0 M"
        );
        assert_eq!(
            line("BLOCK HANDOUT COUNT:"),
            "BLOCK HANDOUT COUNT: on the accumulator's thread (the count above)"
        );
        let mut c = crate::tables::trace_builder::CountThreadStamps::default();
        let lt = StreamTable::Lt.position();
        (c.busy[cpu], c.chunks[cpu], c.busy[lt], c.chunks[lt]) = (1.25, 142, 0.5, 3);
        (c.idle, c.drain, c.merge) = (6.0, 0.04, 0.021);
        stamps.windows.count_thread = Some(c);
        let report = stamps.report();
        assert_eq!(
            report
                .lines()
                .find(|l| l.starts_with("BLOCK HANDOUT COUNT:"))
                .expect("the count's line"),
            "BLOCK HANDOUT COUNT: on a thread of its own · busy 1.75s · waited for chunks 6.00s · \
             the windows' end waited 0.04s for its last chunks and 0.021s to add its counts || \
             per table (busy s · chunks): CPU 1.25 · 142 | LT 0.50 · 3"
        );
    }
}

#[cfg(test)]
mod packer_tests {
    use std::time::Instant;

    use super::{Key, Packer, StreamAirs, table_of};
    use crate::tables::gpack::TraceForm;
    use crate::tables::lt::{self, LtOperation};
    use crate::tables::trace_builder::StreamTable;
    use multilinear::whir_chain::StackVars;
    use stark::proof::options::ProofOptions;

    /// ★ Every streamed chunk is laid out with the AIR the verifier builds for
    /// its table ([`crate::block_airs`]), under either register format: each
    /// CPU table (its own timestamp base, the last one shorter) and each
    /// streamed family the same width, constraints and interactions. The
    /// argue reads the AIR a chunk was laid out with, so a chunk laid out with
    /// another format's AIR proves a block its verifier refuses.
    #[test]
    fn streamed_chunks_are_laid_out_with_the_verifiers_airs() {
        use super::{BlockFormat, DynAir};
        use crate::tables::register::RegisterFormat;
        let opts = ProofOptions::default_test_options();
        let elf = crate::test_utils::asm_elf_bytes("keep_alive_regs");
        let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
        let pages = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
            &program,
            &[],
            0,
            1 << 20,
        )
        .expect("pages");
        let cpu_rows = 1 << 4;
        let heights = vec![cpu_rows, cpu_rows, cpu_rows, cpu_rows / 2];
        let counts = crate::TableCounts {
            cpu: heights.len(),
            lt: 2,
            memw: 2,
            memw_aligned: 2,
            load: 2,
            mul: 1,
            dvrm: 1,
            shift: 2,
            branch: 1,
            memw_register: 3,
            eq: 1,
            bytewise: 1,
            store: 2,
            cpu32: 1,
            keccak: 1,
            keccak_rnd: 2,
            ecsm: 1,
            ecdas: 1,
            hint: 1,
            copy: 1,
            blake3: 1,
            is_wh: 1,
            ecrec: 1,
        };
        let inputs = crate::BlockAirInputs::new(&opts, &program, heights.clone(), &[])
            .expect("the AIR inputs");
        // An AIR's interactions read on a row of distinct column values: each
        // one's bus, side, multiplicity and values.
        use crate::tables::types::FE;
        use stark::lookup::{LinearTerm, Multiplicity};
        let g = |c: usize| FE::from((c as u64 + 3) * 1_000_003);
        let weight = |m: &Multiplicity| match m {
            Multiplicity::One => FE::one(),
            Multiplicity::Column(c) => g(*c),
            Multiplicity::Sum(a, b) => g(*a) + g(*b),
            Multiplicity::Negated(c) => FE::one() - g(*c),
            Multiplicity::Diff(a, b) => g(*a) - g(*b),
            Multiplicity::Sum3(a, b, c) => g(*a) + g(*b) + g(*c),
            Multiplicity::Linear(ts) => ts.iter().fold(FE::zero(), |acc, t| match *t {
                LinearTerm::Column {
                    coefficient,
                    column,
                } => acc + g(column) * FE::from(coefficient),
                LinearTerm::ColumnUnsigned {
                    coefficient,
                    column,
                } => acc + g(column) * FE::from(coefficient),
                LinearTerm::Constant(v) => acc + FE::from(v),
            }),
        };
        let shape = |air: &DynAir| {
            let interactions: Vec<(u64, bool, u64, Vec<u64>)> = air
                .bus_interactions()
                .iter()
                .map(|bi| {
                    let values = bi
                        .values
                        .iter()
                        .flat_map(|v| v.combine_from(g))
                        .map(|x: FE| x.canonical_u64())
                        .collect();
                    (
                        bi.bus_id,
                        bi.is_sender,
                        weight(&bi.multiplicity).canonical_u64(),
                        values,
                    )
                })
                .collect();
            (
                air.trace_layout(),
                air.num_transition_constraints(),
                interactions,
            )
        };
        for registers in [RegisterFormat::Tokens32, RegisterFormat::InRow] {
            let format = BlockFormat {
                registers,
                ..BlockFormat::production()
            };
            let mut airs = crate::block_airs(
                &program,
                &opts,
                &pages,
                &counts,
                &inputs,
                format.keccak_rnd,
                format.memw_aligned,
                registers,
                format.ecdas,
            );
            // The fused memory format's families, as a fused block holds them.
            airs.with_fused_memory(&opts, [2, 2]);
            let stream = StreamAirs::new(
                &opts,
                TraceForm::Wide,
                cpu_rows,
                format.keccak_rnd,
                format.memw_aligned,
                registers,
            );
            for (i, &height) in heights.iter().enumerate() {
                let laid = stream
                    .of_chunk(StreamTable::Cpu, i, height)
                    .expect("a CPU chunk's AIR");
                assert_eq!(laid.name(), airs.cpus[i].name());
                assert_eq!(
                    shape(laid),
                    shape(airs.cpus[i].as_ref()),
                    "{registers:?} CPU[{i}]"
                );
            }
            for (table, verifier) in [
                (StreamTable::MemwRegister, &airs.memw_registers),
                (StreamTable::MemwAligned, &airs.memw_aligneds),
                (StreamTable::Memw, &airs.memws),
                (StreamTable::Load, &airs.loads),
                (StreamTable::Lt, &airs.lts),
                (StreamTable::Shift, &airs.shifts),
                (StreamTable::Store, &airs.stores),
                (StreamTable::KeccakRnd, &airs.keccak_rnds),
                (StreamTable::MemLoad, &airs.mem_loads),
                (StreamTable::MemStore, &airs.mem_stores),
            ] {
                let laid = stream.of(table).expect("the family's AIR");
                for (i, air) in verifier.iter().enumerate() {
                    assert_eq!(
                        shape(laid),
                        shape(air.as_ref()),
                        "{registers:?} {table:?}[{i}]"
                    );
                }
            }
        }
    }

    /// The packer closes every table into groups in order, and phase A, taking
    /// slowly through the channel of one, is handed those groups in that
    /// order.
    #[test]
    fn the_packer_hands_phase_a_its_groups_in_order() {
        let opts = ProofOptions::default_test_options();
        let airs = StreamAirs::new(
            &opts,
            TraceForm::Wide,
            1 << 21,
            crate::tables::keccak_rnd::KeccakRndForm::Full,
            crate::tables::memw_aligned::MemwAlignedOrder::Lt,
            crate::tables::register::RegisterFormat::Tokens32,
        );
        let air = airs.of(StreamTable::Lt).expect("LT's AIR is the family's");
        // Tables of 4, 8 and 16 rows, in an order that makes uneven groups.
        let heights: Vec<usize> = (0..60).map(|k| [3, 5, 12, 2][k * 7 % 4]).collect();
        let pack = |out: std::sync::mpsc::SyncSender<_>| -> Vec<Vec<usize>> {
            let mut packer = Packer {
                open: Vec::new(),
                open_shapes: Vec::new(),
                open_keys: Vec::new(),
                group_keys: Vec::new(),
                out,
                cap: StackVars::new(10).expect("a valid stack"),
                max_polys: 2,
                max_groups: 64,
                start: Instant::now(),
                closed_at: Vec::new(),
                blocked: 0.0,
                ledger: None,
                open_bytes: 0,
                own: None,
                close_ahead: None,
                local: None,
            };
            for (index, &ops) in heights.iter().enumerate() {
                let ops: Vec<LtOperation> = (0..ops as u64)
                    .map(|k| LtOperation::new(k, k + 1, false))
                    .collect();
                let mut trace = lt::generate_lt_trace_as(&ops, TraceForm::Wide);
                let shape = (
                    trace.main_table.width,
                    trace.main_table.height.trailing_zeros() as usize,
                );
                let table = table_of(air, &mut trace, shape, true).expect("a table");
                packer
                    .place(Key::Streamed(StreamTable::Lt, index), shape, table, None)
                    .expect("phase A takes it");
            }
            let (groups, _, _) = packer.finish().expect("phase A takes the last");
            groups
                .into_iter()
                .map(|keys| {
                    keys.into_iter()
                        .map(|key| match key {
                            Key::Streamed(_, index) => index,
                            Key::Air(index) | Key::Local(index) => index,
                        })
                        .collect()
                })
                .collect()
        };
        // What phase A is handed: each group's tables by their heights.
        let take = |groups: std::sync::mpsc::Receiver<Vec<_>>| -> Vec<Vec<usize>> {
            groups
                .iter()
                .map(
                    |group: Vec<stark::multilinear_table::CommittedTable<'_, _, _>>| {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                        group.iter().map(|table| table.num_vars()).collect()
                    },
                )
                .collect()
        };
        let (packed, taken) = std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            let taker = scope.spawn(move || take(rx));
            (pack(tx), taker.join().expect("the taker"))
        });
        assert!(packed.len() >= 4, "{} groups", packed.len());
        assert_eq!(packed.concat(), (0..heights.len()).collect::<Vec<_>>());
        assert_eq!(
            taken.iter().map(Vec::len).collect::<Vec<_>>(),
            packed.iter().map(Vec::len).collect::<Vec<_>>()
        );
    }
}

#[cfg(test)]
mod walkers_knob_tests {
    use super::{CardFill, Error, parse_card_fill, parse_walkers, records_admission};

    /// Streamed chunks are held as records only with live regeneration to drop
    /// them, or kept on purpose: never with a store or with regeneration off
    /// and parking, where none would ever leave the host. And never unless the
    /// card was asked and the build is narrow and drops its streamed ops.
    #[test]
    fn records_need_live_regeneration_unless_kept() {
        for (asked, narrow_dropping, keep, live, admitted) in [
            (true, true, false, true, true),
            (true, true, true, true, true),
            (true, true, true, false, true),
            (true, true, false, false, false),
            (true, false, false, true, false),
            (true, false, true, true, false),
            (false, true, false, true, false),
            (false, true, true, true, false),
        ] {
            assert_eq!(
                records_admission(asked, narrow_dropping, keep, live).is_ok(),
                admitted,
                "asked {asked}, narrow and dropping {narrow_dropping}, keep {keep}, live {live}"
            );
        }
        assert!(
            records_admission(true, true, false, false)
                .is_err_and(|why| why.contains("live regeneration")),
            "the line says why"
        );
    }

    /// `LAMBDA_VM_BLOCK_WALKERS`: unset or empty is four walkers (the
    /// default); 0 one walker (the rollback); 1 to 4 that many; any other
    /// word refuses the prove.
    #[test]
    fn the_walkers_knob_defaults_to_4_takes_0_to_4_and_refuses_anything_else() {
        for (value, walkers) in [
            (None, 4),
            (Some(""), 4),
            (Some(" "), 4),
            (Some("0"), 0),
            (Some("1"), 1),
            (Some(" 3 "), 3),
            (Some("4"), 4),
        ] {
            assert_eq!(parse_walkers(value).ok(), Some(walkers), "{value:?}");
        }
        for value in ["5", "8", "16", "17", "-1", "auto", "4x", "1.5"] {
            assert!(parse_walkers(Some(value)).is_err(), "{value}");
        }
    }

    /// `LAMBDA_VM_BLOCK_CARD_FILL`: `on` has the card fill every family, `off`
    /// none; unset or empty is every family with the card's kernels and
    /// nothing without; any other word refuses the prove.
    #[test]
    fn the_card_fill_knob_takes_off_and_on_and_refuses_anything_else() {
        let none = CardFill::default();
        let every = CardFill::EVERY;
        let card = cfg!(feature = "cuda");
        let default = if card { every } else { none };
        for (value, want) in [
            (None, default),
            (Some(""), default),
            (Some(" "), default),
            (Some("off"), none),
            (Some(" off "), none),
            (Some("on"), every),
            (Some(" on\n"), every),
        ] {
            assert_eq!(parse_card_fill(value).ok(), Some(want), "{value:?}");
        }
        // ★ Unset is every family (KECCAK_RND, CPU, MEMW_R and the row
        // families) with a card, nothing without.
        assert_eq!(CardFill::DEFAULT, default);
        assert_eq!(
            CardFill::DEFAULT.rows.any(),
            card,
            "the row families by default with the card's kernels"
        );
        assert_eq!(none.name(), "off");
        assert_eq!(
            every.name(),
            "keccak_rnd,cpu,memw_r,memw_a,load,store,memw,shift,mem_load,mem_store"
        );
        for value in [
            "1",
            "0",
            "ON",
            "Off",
            "yes",
            "true",
            "on,off",
            "keccak_rnd",
            "cpu,memw_r",
            "keccak_rnd,cpu,memw_r",
            "keccak_rnd,cpu,memw_r,memw_a,load,store,memw,shift",
            "MEMW_A",
        ] {
            let refused = parse_card_fill(Some(value));
            assert!(
                matches!(&refused, Err(Error::Prover(m)) if m.contains("LAMBDA_VM_BLOCK_CARD_FILL")),
                "{value}: {refused:?}"
            );
        }
    }
}

#[cfg(test)]
mod air_input_tests {
    use super::*;

    /// A block statement's CPU heights are read at the CPU tables' positions,
    /// each at most 2^21 rows.
    #[test]
    fn a_block_statement_states_cpu_tables_of_at_most_2_21_rows() {
        let mut counts = crate::TableCounts {
            cpu: 2,
            lt: 0,
            memw: 0,
            memw_aligned: 0,
            load: 0,
            mul: 0,
            dvrm: 0,
            shift: 0,
            branch: 0,
            memw_register: 0,
            eq: 0,
            bytewise: 0,
            store: 0,
            cpu32: 0,
            keccak: 0,
            keccak_rnd: 0,
            ecsm: 0,
            ecdas: 0,
            hint: 0,
            copy: 3,
            blake3: 1,
            is_wh: 1,
            ecrec: 0,
        };
        let start = crate::cpu_proof_start(true, &counts);
        let mut vars = vec![5u8; start + counts.cpu + 4];
        vars[start] = 21;
        vars[start + 1] = 2;
        assert_eq!(
            block_cpu_heights(&counts, &vars).expect("in bounds"),
            vec![1 << 21, 4]
        );
        vars[start + 1] = 22;
        assert!(matches!(
            block_cpu_heights(&counts, &vars),
            Err(Error::InvalidTableCounts(_))
        ));
        counts.cpu = vars.len();
        assert!(matches!(
            block_cpu_heights(&counts, &vars),
            Err(Error::InvalidTableCounts(_))
        ));
    }

    /// A block's AIR inputs refuse CPU heights the clock cannot take, and a
    /// public output whose byte count leaves x254's word, before either sizes
    /// anything; the block's own 2^21-row tables are taken.
    #[test]
    fn block_air_inputs_refuse_what_the_clock_cannot_take() {
        let opts = ProofOptions::default_test_options();
        let program = Elf::load(&crate::test_utils::asm_elf_bytes("all_instructions_64"))
            .expect("the ELF loads");
        let take = |heights: Vec<usize>| BlockAirInputs::new(&opts, &program, heights, &[]);
        assert!(take(vec![1 << 21, 1 << 21, 4]).is_ok());
        assert!(take(vec![crate::tables::cpu::MAX_TABLE_ROWS]).is_ok());
        let past_the_clock = vec![1 << 21; 1 << 9];
        for bad in [
            vec![3],
            vec![6],
            vec![1 << 21, 12],
            vec![crate::tables::cpu::MAX_TABLE_ROWS * 2],
            past_the_clock,
        ] {
            assert!(
                matches!(take(bad.clone()), Err(Error::InvalidTableCounts(_))),
                "heights {:?}",
                &bad[..bad.len().min(4)]
            );
        }
    }
}

#[cfg(test)]
mod p4_rows_knob_tests {
    use super::bitwise_from_tables_or_default;

    /// The finish's BITWISE default: its LT, BRANCH and BYTEWISE lookups from
    /// their tables, unless a deviation asks for phase 4's count from the ops.
    #[test]
    fn the_finish_takes_bitwise_from_its_tables_by_default() {
        assert!(bitwise_from_tables_or_default(None));
        assert!(bitwise_from_tables_or_default(Some(true)));
        assert!(!bitwise_from_tables_or_default(Some(false)));
    }
}

#[cfg(test)]
mod rest_slide_knob_tests {
    use super::{
        BLOCK_REST_BUILD_BYTES, BLOCK_REST_LAYOUT_BYTES, BlockOptions, parse_rest_build,
        rest_layout_label,
    };

    /// `BLOCK_WHIR_REST_BUILD`: unset is no cap beyond the gate (production);
    /// a positive MiB count is the cap in bytes; anything else is an error.
    #[test]
    fn the_build_cap_word_takes_a_positive_mib_count() {
        assert_eq!(parse_rest_build(None), Ok(None));
        for (value, bytes) in [("2048", 2usize << 30), (" 1 ", 1 << 20)] {
            assert_eq!(parse_rest_build(Some(value)), Ok(Some(bytes)), "{value}");
        }
        for value in ["", "0", "-1", "2 GiB", "all"] {
            assert!(parse_rest_build(Some(value)).is_err(), "{value}");
        }
    }

    /// The layout line says waves only where the rest is laid out in waves:
    /// the streamed finish that slides has a gate and no waves.
    #[test]
    fn the_layout_line_says_waves_only_where_the_rest_builds_in_waves() {
        let mut options = BlockOptions::production();
        options.finish_stream = true;
        options.finish_cuts = true;
        assert!(options.rest_streams());
        assert_eq!(
            rest_layout_label(&options, true),
            "a gate of 8192 MiB, sliding (no waves)"
        );
        assert_eq!(rest_layout_label(&options, false), "waves of 8192 MiB");
        options.rest_layout_bytes = None;
        assert_eq!(
            rest_layout_label(&options, true),
            "no gate, sliding (no waves)"
        );
        assert_eq!(rest_layout_label(&options, false), "waves of all at once");
        options.rest_layout_bytes = Some(2 << 30);
        options.finish_stream = false;
        assert!(!options.rest_streams());
        assert_eq!(rest_layout_label(&options, true), "waves of 2048 MiB");
        options.finish_stream = true;
        options.layout_workers = 0;
        assert_eq!(rest_layout_label(&options, true), "waves of 2048 MiB");
    }

    /// The rest holds up to 8 GiB, building up to the gate, by default;
    /// `BLOCK_WHIR_REST_LAYOUT` and `BLOCK_WHIR_REST_BUILD` set either in MiB
    /// (a cap equal to the gate's budget is the gate alone).
    #[test]
    fn the_rest_holds_8_gib_building_up_to_the_gate_by_default() {
        assert_eq!(BLOCK_REST_LAYOUT_BYTES, 8 << 30);
        assert_eq!(BLOCK_REST_BUILD_BYTES, 8 << 30);
        assert_eq!(BLOCK_REST_BUILD_BYTES, BLOCK_REST_LAYOUT_BYTES);
        assert_eq!(parse_rest_build(Some("4096")), Ok(Some(4 << 30)));
        let options = BlockOptions::production();
        assert_eq!(options.rest_layout_bytes, Some(BLOCK_REST_LAYOUT_BYTES));
        assert_eq!(options.rest_build_bytes, Some(BLOCK_REST_BUILD_BYTES));
        assert_eq!(parse_rest_build(Some("2048")), Ok(Some(2 << 30)));
        let earlier = crate::tables::trace_builder::gate::ByteGate::new(2 << 30);
        let rolled_back =
            crate::tables::trace_builder::gate::ByteGate::new(2 << 30).with_build(Some(2 << 30));
        assert_eq!(rolled_back.build(), earlier.build());
        assert_eq!(rolled_back.budget(), earlier.budget());
    }
}

#[cfg(test)]
mod drop_tests {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

    use super::{
        BLOCK_KEEP_BELOW, DROP_PURGE_BYTES, DropPurges, HostReading, decision_bytes,
        parse_keep_below, parse_keep_cells,
    };
    use crate::alloc_purge::Purge;

    const G: u64 = 1 << 30;

    /// `LAMBDA_VM_BLOCK_KEEP_CELLS_G`: unset, empty or `off` is no keep budget
    /// (the default); a G-cell count above 0 is the budget in main cells; any
    /// other word refuses.
    #[test]
    fn the_keep_budget_is_off_by_default_and_takes_g_cells_above_0() {
        for value in [None, Some(""), Some(" off ")] {
            assert_eq!(parse_keep_cells(value).ok(), Some(None), "{value:?}");
        }
        for (value, cells) in [
            ("20.9", 20_900_000_000u64),
            (" 21 ", 21_000_000_000),
            ("0.5", 500_000_000),
        ] {
            assert_eq!(
                parse_keep_cells(Some(value)).ok(),
                Some(Some(cells)),
                "{value}"
            );
        }
        for value in ["0", "-1", "on", "nan", "inf", "20.9G"] {
            assert!(parse_keep_cells(Some(value)).is_err(), "{value}");
        }
    }

    /// `LAMBDA_VM_BLOCK_KEEP_BACK_G`: unset or empty is the default margin,
    /// 1.3 G cells below the arming point; `off` is none (the rollback); a
    /// G-cell count above 0 is that margin in main cells; any other word
    /// refuses.
    #[test]
    fn the_relative_keep_budget_defaults_to_1_3_g_cells_and_off_rolls_back() {
        assert_eq!(BLOCK_KEEP_BELOW, 1_300_000_000);
        for value in [None, Some(""), Some("  ")] {
            assert_eq!(
                parse_keep_below(value).ok(),
                Some(Some(BLOCK_KEEP_BELOW)),
                "{value:?}"
            );
        }
        assert_eq!(parse_keep_below(Some(" off ")).ok(), Some(None));
        for (value, cells) in [("1.3", 1_300_000_000u64), (" 2 ", 2_000_000_000)] {
            assert_eq!(
                parse_keep_below(Some(value)).ok(),
                Some(Some(cells)),
                "{value}"
            );
        }
        for value in ["0", "-1", "on", "nan", "1.3G"] {
            assert!(parse_keep_below(Some(value)).is_err(), "{value}");
        }
    }

    /// Partial drops, once armed, weigh what the host holds now: after a purge
    /// hands the dropped tables' pages back, the reading falls (the working
    /// set and the allocator's resident bytes both did), while the peak the
    /// rule weighs otherwise stays where it was.
    #[test]
    fn after_a_purge_the_partial_reading_falls_and_the_peak_does_not() {
        let before = HostReading {
            hwm: 66 * G,
            charged: 65 * G,
            inactive_file: G,
            ..HostReading::default()
        };
        let after = HostReading {
            hwm: 66 * G,
            charged: 49 * G,
            inactive_file: G,
            ..HostReading::default()
        };
        assert_eq!(decision_bytes(&before, Some(60 * G), true), 64 * G);
        assert_eq!(decision_bytes(&after, Some(44 * G), true), 48 * G);
        // The allocator's resident bytes count when the cgroup reads lower.
        assert_eq!(decision_bytes(&after, Some(55 * G), true), 55 * G);
        // Not partial, or not armed: the peak, purge or not.
        for (reading, allocator) in [(&before, Some(60 * G)), (&after, Some(44 * G))] {
            assert_eq!(decision_bytes(reading, allocator, false), 66 * G);
        }
    }

    /// A jemalloc control of type `T` at `name`: written with `new` when
    /// given, the old value returned.
    fn mallctl<T: Copy + Default>(name: &std::ffi::CStr, new: Option<T>) -> Option<T> {
        let mut old = T::default();
        let mut old_len = std::mem::size_of::<T>();
        let mut new = new;
        let (new_ptr, new_len) = match new.as_mut() {
            Some(v) => (
                v as *mut T as *mut std::ffi::c_void,
                std::mem::size_of::<T>(),
            ),
            None => (std::ptr::null_mut(), 0),
        };
        // SAFETY: `name` is NUL-terminated; the old and new pointers point at a
        // `T` of the control's type, with their lengths.
        let rc = unsafe {
            tikv_jemalloc_sys::mallctl(
                name.as_ptr(),
                &mut old as *mut T as *mut std::ffi::c_void,
                &mut old_len,
                new_ptr,
                new_len,
            )
        };
        (rc == 0).then_some(old)
    }

    /// The allocator's resident bytes, the reading's other half, fall when a
    /// purge hands back a freed gigabyte. The lib's test build runs jemalloc;
    /// the calling thread's arena is set to keep its dirty pages until purged,
    /// as the block's posture keeps them (`dirty_decay_ms:-1`), for the length
    /// of the test.
    #[test]
    fn a_purge_hands_freed_pages_back_to_the_reading() {
        let resident = || {
            crate::alloc_purge::stats()
                .expect("jemalloc's stats")
                .resident as u64
        };
        let arena: u32 = mallctl(c"thread.arena", None).expect("the thread's arena");
        let decay = std::ffi::CString::new(format!("arena.{arena}.dirty_decay_ms"))
            .expect("a control name");
        let was: isize = mallctl(&decay, Some(-1isize)).expect("the arena's decay");
        let mut blocks: Vec<Vec<u8>> = (0..256).map(|_| vec![0u8; 4 << 20]).collect();
        for block in &mut blocks {
            for page in block.chunks_mut(4096) {
                page[0] = 1;
            }
        }
        std::hint::black_box(&blocks);
        let held = resident();
        drop(blocks);
        let freed = resident();
        let purge = crate::alloc_purge::purge_now().expect("the test build purges");
        let purged = resident();
        mallctl(&decay, Some(was));
        assert!(
            freed + G / 2 > held,
            "the freed pages left at once: {held} → {freed} bytes"
        );
        assert!(
            purged + G / 2 < freed,
            "resident {freed} → {purged} bytes ({purge:?})"
        );
    }

    static PURGES: AtomicUsize = AtomicUsize::new(0);

    fn counted() -> Option<Purge> {
        PURGES.fetch_add(1, Relaxed);
        Some(Purge {
            secs: 0.25,
            resident_before: (10 * G) as usize,
            resident_after: (6 * G) as usize,
        })
    }

    /// A purge starts once the dropped bytes reach the threshold, on a thread
    /// of its own, and no second one within the spacing; its line counts it.
    #[test]
    fn partial_drops_purge_past_their_bytes_and_their_spacing() {
        let purges = DropPurges::new(counted);
        let start = PURGES.load(Relaxed);
        assert!(purges.dropped(DROP_PURGE_BYTES / 2).is_none());
        assert!(purges.dropped(DROP_PURGE_BYTES / 2 - 1).is_none());
        purges
            .dropped(1)
            .expect("the threshold starts a purge")
            .join()
            .expect("the purge");
        assert_eq!(PURGES.load(Relaxed) - start, 1);
        // Past the bytes again, but within the spacing: none.
        assert!(purges.dropped(DROP_PURGE_BYTES).is_none());
        assert_eq!(PURGES.load(Relaxed) - start, 1);
        assert_eq!(
            purges.line(),
            "BLOCK DROP: partial · 1 purges of the dropped tables' pages, 0.25 s, 4.00 GiB returned"
        );
    }
}

#[cfg(test)]
mod records_layout_tests {
    use std::sync::Arc;

    use multilinear::narrow::NarrowColumns;
    use multilinear::records::{KeccakRndOp, Records, RecordsFill, TableRecords};
    use stark::proof::options::ProofOptions;

    use super::{E, F, TraceTable, table_of_records};

    /// A table's shape, held as records that fill `prefix` leading columns
    /// from the row index.
    #[derive(Debug)]
    struct Shaped {
        rows: usize,
        cols: usize,
        prefix: usize,
    }

    impl TableRecords for Shaped {
        fn rows(&self) -> usize {
            self.rows
        }

        fn cols(&self) -> usize {
            self.cols
        }

        fn fill(&self) -> RecordsFill<'_> {
            const NONE: &[KeccakRndOp] = &[];
            RecordsFill::KeccakRnd { ops: NONE, skip: 0 }
        }

        fn row_index_columns(&self) -> usize {
            self.prefix
        }

        fn host_fill(&self) -> Option<NarrowColumns> {
            None
        }
    }

    fn laid_out(
        air: &super::DynAir,
        (rows, cols): (usize, usize),
        prefix: usize,
    ) -> Result<(), crate::Error> {
        let records = Records(Arc::new(Shaped { rows, cols, prefix }));
        let mut trace =
            TraceTable::<F, E>::try_from_records_main(records, 1).expect("a records trace");
        table_of_records(air, &mut trace, (cols, rows.trailing_zeros() as usize)).map(|_| ())
    }

    /// ★ A table held as records is laid out only for an AIR whose
    /// preprocessed columns are the ones its records fill from the row index:
    /// CPU's TIMESTAMP, none for MEMW_R, all nine of KECCAK_RC's for a records
    /// table that declares them. Any other count is refused, either way.
    #[test]
    fn records_are_laid_out_only_for_their_row_index_columns() {
        use crate::tables::{cpu, keccak_rc, memw_register};
        use crate::test_utils::{create_cpu_air, create_keccak_rc_air, create_memw_register_air};
        let opts = ProofOptions::default_test_options();
        // A streamed CPU chunk's AIR, as the block builds it: its TIMESTAMP
        // column preprocessed.
        let height = 1usize << 4;
        let cpu = create_cpu_air(&opts, 0).with_lazy_preprocessed_columns(
            cpu::lazy_timestamp_commitment(&opts, height),
            cpu::NUM_PRECOMPUTED_COLS,
            Arc::new(move || vec![cpu::timestamp_column(height)]),
        );
        let memw_r = create_memw_register_air(&opts);
        let rc =
            create_keccak_rc_air(&opts).with_preprocessed([0; 32], keccak_rc::NUM_PRECOMPUTED_COLS);
        let shape = |cols| (height, cols);
        assert!(
            laid_out(
                &cpu,
                shape(cpu::cols::NUM_COLUMNS),
                cpu::NUM_PRECOMPUTED_COLS
            )
            .is_ok()
        );
        assert!(laid_out(&memw_r, shape(memw_register::cols::NUM_COLUMNS), 0).is_ok());
        for wrong in [0, 2] {
            let refused = laid_out(&cpu, shape(cpu::cols::NUM_COLUMNS), wrong);
            assert!(
                matches!(&refused, Err(crate::Error::Prover(m)) if m.contains("preprocessed")),
                "CPU, {wrong} declared: {refused:?}"
            );
        }
        let rc_shape = (keccak_rc::NUM_ROWS, keccak_rc::cols::NUM_COLUMNS);
        assert!(laid_out(&rc, rc_shape, keccak_rc::NUM_PRECOMPUTED_COLS).is_ok());
        for wrong in [0, keccak_rc::NUM_PRECOMPUTED_COLS - 1] {
            let refused = laid_out(&rc, rc_shape, wrong);
            assert!(
                matches!(&refused, Err(crate::Error::Prover(m)) if m.contains("preprocessed")),
                "KECCAK_RC, {wrong} declared: {refused:?}"
            );
        }
    }
}

#[cfg(test)]
mod rest_airs_beside_tests {
    use super::{AirRef, keccak_rnd_range, parse_rest_airs_beside, same_air};
    use crate::tables::keccak_rnd::KeccakRndForm;
    use crate::test_utils::create_keccak_rnd_air_for;
    use stark::proof::options::ProofOptions;

    /// The streamed KECCAK_RND knob: the default unset or empty (`None`);
    /// asked for at `1`; off at `0` (the rollback); any other word refuses.
    #[test]
    fn the_stream_keccak_rnd_knob_is_on_by_default_and_refuses_other_words() {
        for unset in [None, Some(""), Some(" ")] {
            assert_eq!(
                super::parse_stream_keccak_rnd(unset).expect("the default"),
                None
            );
        }
        assert_eq!(
            super::parse_stream_keccak_rnd(Some("1")).expect("on"),
            Some(true)
        );
        assert_eq!(
            super::parse_stream_keccak_rnd(Some(" 0\n")).expect("off"),
            Some(false)
        );
        for bad in ["yes", "2", "on", "off"] {
            assert!(super::parse_stream_keccak_rnd(Some(bad)).is_err(), "{bad}");
        }
    }

    /// KECCAK_RND's own groups knob: on unset, empty or `1` (the default);
    /// off at `0` (the rollback); any other word refuses.
    #[test]
    fn the_keccak_rnd_own_groups_knob_is_on_by_default_and_refuses_other_words() {
        for on in [None, Some(""), Some("1"), Some(" 1\n")] {
            assert!(super::parse_own_keccak_rnd_groups(on).expect("a word it takes"));
        }
        assert!(!super::parse_own_keccak_rnd_groups(Some("0")).expect("off"));
        for bad in ["yes", "2", "on"] {
            assert!(
                super::parse_own_keccak_rnd_groups(Some(bad)).is_err(),
                "{bad}"
            );
        }
    }

    /// The arrivals measurement's knob: off unset, empty or `0`; on at `1`;
    /// any other word refuses.
    #[test]
    fn the_arrivals_knob_is_off_by_default_and_refuses_other_words() {
        for off in [None, Some(""), Some("0")] {
            assert!(!super::parse_keccak_rnd_arrivals(off).expect("a word it takes"));
        }
        assert!(super::parse_keccak_rnd_arrivals(Some("1")).expect("on"));
        assert!(super::parse_keccak_rnd_arrivals(Some("yes")).is_err());
    }

    /// The knob: on unset, empty or `1`; off at `0` (the rollback); any other
    /// word refuses.
    #[test]
    fn the_rest_airs_knob_is_on_by_default_and_refuses_other_words() {
        for on in [None, Some(""), Some("1"), Some(" 1 ")] {
            assert!(
                parse_rest_airs_beside(on).expect("a word it takes"),
                "{on:?}"
            );
        }
        assert!(!parse_rest_airs_beside(Some("0")).expect("off"));
        assert!(!parse_rest_airs_beside(Some(" 0\n")).expect("off"));
        for bad in ["2", "on", "true", "yes"] {
            assert!(parse_rest_airs_beside(Some(bad)).is_err(), "{bad}");
        }
    }

    /// The page tables go next unless the knob says `0`.
    #[test]
    fn the_pages_first_knob_is_on_by_default_and_refuses_other_words() {
        for on in [None, Some(""), Some("1"), Some(" 1\n")] {
            assert!(super::parse_pages_first(on).expect("on"), "{on:?}");
        }
        assert!(!super::parse_pages_first(Some("0")).expect("off"));
        for bad in ["2", "on", "true", "yes"] {
            assert!(super::parse_pages_first(Some(bad)).is_err(), "{bad}");
        }
    }

    /// ★ The early KECCAK_RND AIRs, built as the streamed rest builds them,
    /// are the run's set's at KECCAK_RND's places under either form, and
    /// [`same_air`] refuses another name, another form, or another table's
    /// AIR.
    #[test]
    fn the_early_keccak_rnd_airs_are_the_run_sets() {
        let opts = ProofOptions::default_test_options();
        let elf = crate::test_utils::asm_elf_bytes("test_keccak_multi");
        let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
        let pages = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
            &program,
            &[],
            0,
            1 << 20,
        )
        .expect("pages");
        let counts = crate::TableCounts {
            cpu: 1,
            lt: 1,
            memw: 1,
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
            keccak: 1,
            keccak_rnd: 3,
            ecsm: 1,
            ecdas: 1,
            hint: 1,
            copy: 2,
            blake3: 1,
            is_wh: 1,
            ecrec: 1,
        };
        let inputs =
            crate::BlockAirInputs::new(&opts, &program, vec![1 << 4], &[]).expect("the AIR inputs");
        let range = keccak_rnd_range(&counts).expect("KECCAK_RND's range");
        assert_eq!(range.len(), counts.keccak_rnd);
        for form in [KeccakRndForm::Lean, KeccakRndForm::Full] {
            let airs = crate::block_airs(
                &program,
                &opts,
                &pages,
                &counts,
                &inputs,
                form,
                crate::tables::memw_aligned::MemwAlignedOrder::Limbs,
                crate::tables::register::RegisterFormat::InRow,
                crate::tables::ecdas::EcdasForm::Bytes,
            );
            let refs = airs.air_refs();
            let early = |i: usize, form: KeccakRndForm| {
                Box::new(
                    create_keccak_rnd_air_for(&opts, form).with_name(&format!("KECCAK_RND[{i}]")),
                ) as crate::VmAir
            };
            for (i, position) in range.clone().enumerate() {
                let own = early(i, form);
                same_air(own.as_ref(), refs[position])
                    .unwrap_or_else(|e| panic!("{form:?} KECCAK_RND[{i}]: {e:?}"));
                // Named for another table, or of the other form: refused.
                let misnamed = early(i + 1, form);
                assert!(same_air(misnamed.as_ref(), refs[position]).is_err());
                let other = match form {
                    KeccakRndForm::Lean | KeccakRndForm::LeanLocal => KeccakRndForm::Full,
                    KeccakRndForm::Full => KeccakRndForm::Lean,
                };
                assert!(
                    same_air(early(i, other).as_ref(), refs[position]).is_err(),
                    "{form:?}: the other form's AIR at {position}"
                );
            }
            // Another table's AIR at a KECCAK_RND place.
            let first: AirRef<'_> = refs[0];
            assert!(same_air(first, refs[range.start]).is_err());
        }
    }
}

#[cfg(test)]
mod rest_airs_probe {
    use super::StreamAirs;
    use crate::tables::gpack::TraceForm;
    use crate::tables::trace_builder::StreamTable;
    use std::time::Instant;

    /// An instrument, never a gate: the run's AIR set for a large block's
    /// table counts, built as the streamed rest builds it (the streamed
    /// chunks' AIRs first), timed step by step; then each family's
    /// constructors alone under options no AIR was built with yet.
    /// `PROBE_ELF` names the guest.
    #[test]
    #[ignore = "an instrument: PROBE_ELF names the guest"]
    fn rest_airs_build_seconds() {
        use crate::tables::fused_memory::MemoryFormat;
        use crate::tables::keccak_rnd::KeccakRndForm;
        use crate::tables::memw_aligned::MemwAlignedOrder;
        use crate::tables::register::RegisterFormat;
        let path = std::env::var("PROBE_ELF").expect("PROBE_ELF");
        let bytes = std::fs::read(path).expect("the guest");
        let program = executor::elf::Elf::load(&bytes).expect("the ELF loads");
        let opts = crate::lfm::proof::block_base_options();
        let cpu_rows = 1usize << 21;
        let mut heights = vec![cpu_rows; 116];
        heights.push(1 << 20);
        let mut pages = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
            &program,
            &[],
            0,
            1 << 20,
        )
        .expect("pages");
        let mut base = 0x4000_0000u64;
        while pages.len() < 351 {
            pages.push(crate::tables::page::PageConfig::zero_init(base));
            base += 1 << 18;
        }
        let counts = crate::TableCounts {
            cpu: heights.len(),
            lt: 7,
            memw: 0,
            memw_aligned: 0,
            load: 0,
            mul: 2,
            dvrm: 1,
            shift: 9,
            branch: 11,
            memw_register: 2,
            eq: 9,
            bytewise: 7,
            store: 0,
            cpu32: 1,
            keccak: 2,
            keccak_rnd: 66,
            ecsm: 0,
            hint: 0,
            ecdas: 4,
            copy: 8,
            blake3: 0,
            is_wh: 0,
            ecrec: 1,
        };
        let words = [23, 17, 7, 2, 3, 2, 9];
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
        // The streamed chunks' AIRs, as phase A builds them before the rest.
        let t = Instant::now();
        let stream = StreamAirs::new(
            &opts,
            TraceForm::Narrow,
            cpu_rows,
            KeccakRndForm::Lean,
            MemwAlignedOrder::Limbs,
            RegisterFormat::InRow,
        );
        for i in 0..116 {
            stream
                .of_chunk(StreamTable::Cpu, i, cpu_rows)
                .expect("a CPU chunk's AIR");
        }
        eprintln!("PROBE stream AIRs {:.1} ms", ms(t));
        // The rest's AIR set, step by step as `block_airs` and the streamed
        // rest build it.
        let t0 = Instant::now();
        let t = Instant::now();
        let inputs = crate::BlockAirInputs::new(&opts, &program, heights.clone(), &[])
            .expect("the AIR inputs");
        let inputs_ms = ms(t);
        let t = Instant::now();
        let mut airs = crate::block_airs(
            &program,
            &opts,
            &pages,
            &counts,
            &inputs,
            KeccakRndForm::Lean,
            MemwAlignedOrder::Limbs,
            RegisterFormat::InRow,
            crate::tables::ecdas::EcdasForm::Bytes,
        );
        let block_ms = ms(t);
        let t = Instant::now();
        airs.with_fused_memory(&opts, [0, 0]);
        airs.with_word_memory(&opts, words, &pages)
            .expect("the word tables");
        let words_ms = ms(t);
        let t = Instant::now();
        let refs = airs.air_refs();
        let refs_ms = ms(t);
        eprintln!(
            "PROBE rest AIR set {:.1} ms: inputs {inputs_ms:.1} · block_airs {block_ms:.1} · \
             words {words_ms:.1} · refs {refs_ms:.1} · {} AIRs",
            ms(t0),
            refs.len()
        );
        // Each family alone, under options whose cache keys are all new.
        let mut cold = opts.clone();
        cold.grinding_factor = cold.grinding_factor.wrapping_add(1);
        let o = &cold;
        let full = crate::tables::cpu::TIMESTAMP_STRIDE * cpu_rows as u64;
        let time = |label: &str, f: &mut dyn FnMut()| {
            let t = Instant::now();
            f();
            eprintln!("PROBE cold {label} {:.1} ms", ms(t));
        };
        use crate::test_utils::*;
        time("inputs (register files)", &mut || {
            let _ = crate::BlockAirInputs::new(o, &program, heights.clone(), &[]);
        });
        time("decode instructions", &mut || {
            let _ = crate::tables::decode::instructions_from_elf(&program);
        });
        time("zero page commitment", &mut || {
            let _ = crate::tables::page::zero_init_preprocessed_commitment(o);
        });
        time("117 CPU (split)", &mut || {
            for i in 0..117u64 {
                let _ = create_cpu_air(o, i * full);
            }
        });
        time("117 CPU (in-row)", &mut || {
            for i in 0..117u64 {
                let _ = create_cpu_air_inrow(o, i * full);
            }
        });
        time("117 CPU (in-row) again", &mut || {
            for i in 0..117u64 {
                let _ = create_cpu_air_inrow(o, i * full);
            }
        });
        time("66 KECCAK_RND full", &mut || {
            for _ in 0..66 {
                let _ = create_keccak_rnd_air_for(o, KeccakRndForm::Full);
            }
        });
        time("66 KECCAK_RND lean", &mut || {
            for _ in 0..66 {
                let _ = create_keccak_rnd_air_for(o, KeccakRndForm::Lean);
            }
        });
        time("351 PAGE", &mut || {
            for p in &pages {
                let _ = create_page_air(o, p.page_base);
            }
        });
        time("351 PAGE_W", &mut || {
            for p in &pages {
                let _ = create_page_w_air(o, p).expect("PAGE_W");
            }
        });
        time("351 PAGE_W again", &mut || {
            for p in &pages {
                let _ = create_page_w_air(o, p).expect("PAGE_W");
            }
        });
        time("8 COPY + 8 COPY_W", &mut || {
            for _ in 0..8 {
                let _ = create_copy_air(o);
                let _ = create_copy_w_air(o);
            }
        });
        time("2 KECCAK + 2 KECCAK_W", &mut || {
            for _ in 0..2 {
                let _ = create_keccak_air(o);
                let _ = create_keccak_w_air(o);
            }
        });
        time("4 ECDAS (limb form) + ECREC + ECREC_W", &mut || {
            for _ in 0..4 {
                let _ = create_ecdas_halves_air(o);
            }
            let _ = create_ecrec_air(o);
            let _ = create_ecrec_w_air(o);
        });
        time("the word families (64)", &mut || {
            for _ in 0..9 {
                let _ = create_mem_load8_air(o);
                let _ = create_mem_store8_air(o);
                let _ = create_mem_loadw_air(o);
                let _ = create_mem_storew_air(o);
                let _ = create_mem_load2w_air(o);
                let _ = create_mem_store2w_air(o);
                let _ = create_memw_w_air(o);
            }
        });
        let _ = MemoryFormat::Words;
    }
}

#[cfg(test)]
mod phase_b_many_first_tests {
    use super::{group_of_most_tables, parse_phase_b_many_first, put_first};

    /// The knob is off unless it says `1`; other words refuse the prove.
    #[test]
    fn the_many_first_knob_is_off_by_default_and_refuses_other_words() {
        for off in [None, Some(""), Some("0"), Some(" 0\n")] {
            assert!(!parse_phase_b_many_first(off).expect("off"), "{off:?}");
        }
        assert!(parse_phase_b_many_first(Some("1")).expect("on"));
        for bad in ["2", "on", "true", "yes"] {
            assert!(parse_phase_b_many_first(Some(bad)).is_err(), "{bad}");
        }
    }

    /// The group of the most tables, the first by index on a tie; none of no
    /// groups; and put first, the order stays a permutation.
    #[test]
    fn the_group_of_the_most_tables_goes_first() {
        let groups = vec![
            vec![0u32],
            vec![1, 2, 3],
            vec![4],
            vec![5, 6, 7],
            vec![8, 9],
        ];
        assert_eq!(group_of_most_tables(&groups), Some(1));
        assert_eq!(group_of_most_tables(&[]), None);
        let order = put_first(&[3], (0..5).collect()).expect("a permutation");
        assert_eq!(order, vec![3, 0, 1, 2, 4]);
    }
}

#[cfg(test)]
mod close_before_bitwise_tests {
    use super::{parse_close_before_bitwise, past_budget};

    /// The knob is off unless it says `1`; other words refuse the prove.
    #[test]
    fn the_close_before_bitwise_knob_is_off_by_default_and_refuses_other_words() {
        for off in [None, Some(""), Some("0"), Some(" 0\n")] {
            assert!(!parse_close_before_bitwise(off).expect("off"), "{off:?}");
        }
        assert!(parse_close_before_bitwise(Some("1")).expect("on"));
        for bad in ["2", "on", "true", "yes"] {
            assert!(parse_close_before_bitwise(Some(bad)).is_err(), "{bad}");
        }
    }

    /// The close-ahead rule is the packer's own: a group past its budget with
    /// the next shape added would close, one within it would not.
    #[test]
    fn the_close_ahead_rule_is_the_packers_budget() {
        let cap = multilinear::whir_chain::StackVars::new(10).expect("a stack");
        let small = (4usize, 10usize);
        let bitwise = (11usize, 20usize);
        assert!(!past_budget(&[small], small, cap, 8).expect("a layout"));
        assert!(past_budget(&[small], bitwise, cap, 8).expect("a layout"));
        assert!(past_budget(&[small, small], small, cap, 1).expect("a layout"));
    }
}

#[cfg(test)]
mod narrow_commit_tests {
    use super::{NARROW_COMMIT_MARGIN_BYTES, narrow_commit_fits};

    /// A narrow commit's store fits only with its words, its staging, a card
    /// commit's price and the margin all within what is free; one byte short
    /// refuses, and nothing overflows.
    #[test]
    fn a_narrow_commit_fits_with_its_store_staging_price_and_margin() {
        let (words, staging, price) = (2_480 << 20, 256 << 20, 6_000 << 20);
        let need = words + staging + price + NARROW_COMMIT_MARGIN_BYTES;
        assert!(narrow_commit_fits(words, staging, price, need));
        assert!(!narrow_commit_fits(words, staging, price, need - 1));
        assert!(!narrow_commit_fits(words, 0, price, words + price));
        assert!(!narrow_commit_fits(u64::MAX, staging, price, u64::MAX - 1));
    }
}
