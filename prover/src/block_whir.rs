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
use multilinear::whir_chain::{ArgueFormat, ChainConfig, GroupFold};
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
    parse_tree_drop(
        std::env::var("LAMBDA_VM_BLOCK_TREE_DROP_LEVELS")
            .ok()
            .as_deref(),
    )
    .unwrap_or(TREE_DROP_DEFAULT)
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
}

/// The most groups a block statement may declare: a verifier constant, never in
/// the transcript. The 1× block has 9 and the median under 100; a 200 Mgas
/// block makes some 600 to 925. The bound caps what a statement can make the
/// verifier build, and the block-wide LogUp support: its worst case at 1024
/// groups is pinned by `block_wide_logup_at_the_group_maximum_is_pinned`.
pub const BLOCK_MAX_GROUPS: usize = 1024;

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
    let height = match base.arity4_cap {
        CapPolicy::Off | CapPolicy::Fixed(0) => "0".to_string(),
        CapPolicy::Fixed(c) => c.to_string(),
        CapPolicy::Auto => "auto".to_string(),
    };
    let mut tag = MULTILINEAR_BLOCK_TAG.to_vec();
    tag.extend_from_slice(format!("/P1W16/C{height}").as_bytes());
    tag
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
        Self {
            zf: *ZfFormat::global(),
            group_polys: BLOCK_GROUP_POLYS,
            max_groups: BLOCK_MAX_GROUPS,
            prepared: true,
            leaf_scale: crate::lfm::whir_block::process_leaf_scale(),
            leaf_query: crate::lfm::whir_block::process_leaf_query(),
            argue: ArgueFormat::BATCHED,
            group: GroupFold::Indexed,
            keccak_rnd: crate::tables::keccak_rnd::KeccakRndForm::Lean,
            memw_aligned: crate::tables::memw_aligned::MemwAlignedOrder::Limbs,
        }
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
    /// against.
    pub fn pinned() -> Self {
        Self {
            zf: ZfFormat::DEFAULT,
            group_polys: BLOCK_GROUP_POLYS,
            max_groups: BLOCK_MAX_GROUPS,
            prepared: true,
            leaf_scale: crate::lfm::whir_block::LeafScale::DEFAULT,
            leaf_query: crate::lfm::whir_chain::LeafQuery::Lean,
            argue: ArgueFormat::BATCHED,
            group: GroupFold::Indexed,
            keccak_rnd: crate::tables::keccak_rnd::KeccakRndForm::Lean,
            memw_aligned: crate::tables::memw_aligned::MemwAlignedOrder::Limbs,
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

    /// This format with its batched argue's GKR trees laid out per `pack`:
    /// a named constant the prover and the
    /// verifier of one block both take, never read from a proof. Production
    /// is [`GkrPack::PRODUCTION`](multilinear::whir_chain::GkrPack::PRODUCTION);
    /// [`GkrPack::Off`](multilinear::whir_chain::GkrPack::Off) is one tree a
    /// table. A per-table argue has no trees to pack and is returned as it is.
    pub fn with_pack(self, pack: multilinear::whir_chain::GkrPack) -> Self {
        let argue = match self.argue {
            ArgueFormat::Batched { bin_log_cells, .. } => ArgueFormat::Batched {
                bin_log_cells,
                pack,
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
        config.format.argue = self.argue;
        config.format.group = self.group;
        config.format.leaves = self.zf.whir_leaves;
        // One tree a run needs runs: one chain a polynomial has none, and its
        // round-0 trees are one a polynomial whatever the knob says.
        config.format.round0 = match self.group {
            GroupFold::Indexed => self.zf.whir_trees,
            GroupFold::PerPoly => multilinear::whir_chain::Round0Trees::Each,
        };
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
    /// aligned rows, LOAD's, STORE's, MEMW's and SHIFT's ops), under the same
    /// room rule ([`WindowedTraceBuilder::card_fill_streamed_rows`]).
    pub(crate) rows: RowFamilies,
}

impl CardFill {
    /// The families the card fills.
    const NAMES: [&'static str; 8] = [
        "keccak_rnd",
        "cpu",
        "memw_r",
        "memw_a",
        "load",
        "store",
        "memw",
        "shift",
    ];

    /// Every family.
    const EVERY: Self = Self {
        keccak_rnd: true,
        cpu: true,
        memw_r: true,
        rows: RowFamilies::ALL,
    };

    /// Unset or empty: every family (KECCAK_RND, CPU, MEMW_R and the five row
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
    fn flags(&mut self) -> [&mut bool; 8] {
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
/// ([`CardFill`]: KECCAK_RND, CPU, MEMW_R and the five row families) from the
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
/// what the rest holds built and not yet placed, or building. At 2 GiB the
/// large block's gate filled and its builds waited for room (≈ 1.3 s); with
/// at most [`BLOCK_REST_BUILD_BYTES`] building at once, the first tables the
/// packer needs build as they did at 2 GiB. Where the rest is laid out in
/// waves (the finish not streamed, or not sliding) they bound the rows each
/// wave transposes: each transposition is parallel within its table, so a
/// wave of a few tables keeps the cores busy while bounding the copies in
/// flight.
pub const BLOCK_REST_LAYOUT_BYTES: usize = 4 << 30;

/// The estimated bytes of the rest's tables the sliding finish builds at once
/// by default ([`BlockOptions::rest_build_bytes`]), apart from the gate's
/// [`BLOCK_REST_LAYOUT_BYTES`]: more builds at once on the shared pool slow
/// the first tables the packer needs.
pub const BLOCK_REST_BUILD_BYTES: usize = 2 << 30;

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
    /// Every carried stacked polynomial's commit-time out-of-domain answer, in
    /// polynomial order: one per stacked polynomial of every group (one per
    /// root of `proof.roots` unless a group's run shares one round-0 tree,
    /// `ZfFormat::whir_trees`), absorbed after the roots and before the
    /// block's first challenge, and each carried into its chain
    /// (`stark::multilinear_block::commit_ood_point`). The verifier refuses
    /// any other count.
    pub commit_ood: Vec<FieldElement<E>>,
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

/// `LAMBDA_VM_EXEC_RECYCLE_LOGS=1`: the streamed prove's executor takes back
/// each window's log buffer once the walk and the card fill have let it go,
/// and writes a later window into it rather than into fresh memory (whose
/// first touch faults its pages in). Unset, empty or `0`: off. Anything else
/// refuses the prove. The logs are the same either way.
pub(crate) fn exec_recycle_logs() -> Result<bool, Error> {
    parse_exec_recycle_logs(std::env::var("LAMBDA_VM_EXEC_RECYCLE_LOGS").ok().as_deref())
}

/// [`exec_recycle_logs`]'s reading of its variable.
fn parse_exec_recycle_logs(value: Option<&str>) -> Result<bool, Error> {
    match value.map(str::trim) {
        None | Some("" | "0") => Ok(false),
        Some("1") => Ok(true),
        Some(v) => Err(Error::Prover(format!(
            "LAMBDA_VM_EXEC_RECYCLE_LOGS={v}: 1 (take window log buffers back) or 0"
        ))),
    }
}

/// `LAMBDA_VM_EXEC_PREFAULT=1`: a helper thread prepares the executor's next
/// window buffers (logs, and the history's extras in record mode) already
/// written, so the executor does not fault fresh pages in as it writes a
/// window; the card fill holds the buffers it sent long after (so they are
/// never free to reuse in time). Unset, empty or `0`: off. Anything else
/// refuses the prove. The logs are the same.
pub(crate) fn exec_prefault() -> Result<bool, Error> {
    parse_exec_prefault(std::env::var("LAMBDA_VM_EXEC_PREFAULT").ok().as_deref())
}

/// [`exec_prefault`]'s reading of its variable.
fn parse_exec_prefault(value: Option<&str>) -> Result<bool, Error> {
    match value.map(str::trim) {
        None | Some("" | "0") => Ok(false),
        Some("1") => Ok(true),
        Some(v) => Err(Error::Prover(format!(
            "LAMBDA_VM_EXEC_PREFAULT={v}: 1 (pre-faulted window buffers) or 0"
        ))),
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
}

impl ExecutorStamps {
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
                 SHIFT chunks {} as records, {} filled on the host\n",
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
            "BLOCK PHASE-B LEDGER: budget {} MiB · per group base/argue/room/open [{}] MiB · open(g) beside argue(g+1) [{}] MiB{} · revive room refusals {} · turn refusals {} · GKR tree refusals {} · openings on the host {}\n",
            mib(multilinear::gpu::reserve_budget()),
            per_group.join(", "),
            overlap.join(", "),
            if self.phase_b_overlap { " measured" } else { "" },
            multilinear::gpu::room_revive_refusals(),
            multilinear::gpu::room_turn_refusals(),
            multilinear::gpu::gkr_tree_refusals(),
            multilinear::gpu::open_host_fallbacks(),
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

/// The block statement: the monolithic multilinear one under
/// [`MULTILINEAR_BLOCK_TAG`], then the group partition — the count, then per
/// group its size and its tables' indices, every value a little-endian `u64`
/// (so the roots that follow stay on a field-element boundary).
#[allow(clippy::too_many_arguments)]
pub(crate) fn absorb_block(
    t: &mut impl IsTranscript<E>,
    tag: &[u8],
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
        tag,
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
    if page_configs.len() != airs.pages.len() {
        return Err(Error::Prover(format!(
            "{} page configs for {} page tables",
            page_configs.len(),
            airs.pages.len()
        )));
    }
    let first_page = first_page_index(airs)?;
    let decode = decode_table_index(&refs)?;
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
    commit_group_where::<H>(stack, config, false)
}

/// [`commit_group`] on the host, whatever a card would do
/// (`StackedCommitment::commit_on_host`): the same roots. What a verifier
/// derives the prepared roots with, so no card kernel sits on its path.
pub(crate) fn commit_group_on_host<H: multilinear::whir_hash::WhirHash>(
    stack: GroupStack,
    config: &multilinear::whir_chain::ChainConfig,
) -> Result<GroupPrepared<H>, Error> {
    commit_group_where::<H>(stack, config, true)
}

fn commit_group_where<H: multilinear::whir_hash::WhirHash>(
    stack: GroupStack,
    config: &multilinear::whir_chain::ChainConfig,
    on_host: bool,
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
    let borrowed = multilinear::stacking::borrow(&columns);
    let commitment = if on_host {
        multilinear::stacked_eval::StackedCommitment::<F, H>::commit_on_host(
            layout, &borrowed, config,
        )
    } else {
        multilinear::stacked_eval::StackedCommitment::<F, H>::commit(
            layout, &borrowed, None, config,
        )
    }
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
}

/// The recursion's leaves, offered to phase B while it runs (`W3_LEAF_FIRST`),
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
) -> Result<(BlockWhirProof, BlockStamps), Error> {
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
    proved
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
    split_ecdas(&mut traces, options.ecdas_rows_log2, padding);
    split_keccak(&mut traces, options.keccak_rows_log2, padding);
    split_ecsm(&mut traces, options.ecsm_rows_log2, padding);
    split_copy(&mut traces, options.copy_rows_log2, padding);
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
    Ok((proof, stamps))
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
        (left_out * form.are_bytes_per_row()) as u64,
    )
}

/// Cuts every ECDAS table into tables of at most `2^rows_log2` rows, as many
/// as `padding` keeps ([`split_cut`]): a cut may fall inside one scalar
/// multiplication, whose steps chain through the Ecdas bus alone
/// ([`BLOCK_ECDAS_ROWS_LOG2`]).
pub(crate) fn split_ecdas(traces: &mut Traces, rows_log2: usize, padding: CutPadding) {
    let rows = 1usize << rows_log2;
    let mu = crate::tables::ecdas::cols::MU;
    traces.ecdases = std::mem::take(&mut traces.ecdases)
        .into_iter()
        .flat_map(|table| split_cut(table, rows, mu, padding))
        .collect();
}

/// Cuts every KECCAK table into tables of at most `2^rows_log2` rows, as many
/// as `padding` keeps ([`split_cut`]): a row is one whole permutation call, so
/// a cut falls between calls ([`BLOCK_KECCAK_ROWS_LOG2`]).
pub(crate) fn split_keccak(traces: &mut Traces, rows_log2: usize, padding: CutPadding) {
    let rows = 1usize << rows_log2;
    let mu = crate::tables::keccak::cols::MU;
    traces.keccaks = std::mem::take(&mut traces.keccaks)
        .into_iter()
        .flat_map(|table| split_cut(table, rows, mu, padding))
        .collect();
}

/// Cuts every COPY table into tables of at most `2^rows_log2` rows, as many as
/// `padding` keeps ([`split_cut`]): a cut may fall inside one call, whose rows
/// chain through the CopyNext bus alone ([`BLOCK_COPY_ROWS_LOG2`]).
pub(crate) fn split_copy(traces: &mut Traces, rows_log2: usize, padding: CutPadding) {
    let rows = 1usize << rows_log2;
    let mu = crate::tables::copy::cols::MU;
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
    let airs = block_airs(
        program,
        proof_options,
        &traces.page_configs,
        &table_counts,
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
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
    let tag = block_statement_tag(&format.zf.base);
    let proof = crate::with_block_hash!(format.zf.base, |H| {
        let mut transcript = <H as multilinear::whir_hash::WhirHash>::sponge();
        absorb_block(
            &mut transcript,
            &tag,
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
        let (proof, argues, prepared_openings, groups, commit_ood) =
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
    opts: ProofOptions,
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

/// CPU AIR slots per page of [`StreamAirs`]'s CPU tables.
const CPU_AIR_PAGE: usize = 256;

impl StreamAirs {
    fn new(
        opts: &ProofOptions,
        form: TraceForm,
        cpu_rows: usize,
        keccak_rnd: crate::tables::keccak_rnd::KeccakRndForm,
        memw_aligned: crate::tables::memw_aligned::MemwAlignedOrder,
    ) -> Self {
        use crate::test_utils::*;
        // As many CPU tables as a run's rows allow (`tables::cpu::MAX_CPU_ROWS`).
        let most = (crate::tables::cpu::MAX_CPU_ROWS as usize).div_ceil(cpu_rows.max(1));
        Self {
            form,
            cpus: (0..most.div_ceil(CPU_AIR_PAGE))
                .map(|_| std::sync::OnceLock::new())
                .collect(),
            cpu_rows,
            opts: opts.clone(),
            memw_register: Box::new(create_memw_register_air(opts)),
            memw_aligned: Box::new(create_memw_aligned_air_for(opts, memw_aligned)),
            memw: Box::new(create_memw_air(opts)),
            load: Box::new(create_load_air(opts)),
            lt: Box::new(create_lt_air(opts)),
            shift: Box::new(create_shift_air(opts)),
            store: Box::new(create_store_air(opts)),
            keccak_rnd: Box::new(create_keccak_rnd_air_for(opts, keccak_rnd)),
        }
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
                Box::new(
                    crate::test_utils::create_cpu_air(&self.opts, ts_base)
                        .with_name(&format!("CPU[{index}]"))
                        .with_lazy_preprocessed_columns(
                            crate::tables::cpu::lazy_timestamp_commitment(&self.opts, height),
                            crate::tables::cpu::NUM_PRECOMPUTED_COLS,
                            std::sync::Arc::new(move || {
                                vec![crate::tables::cpu::timestamp_column(height)]
                            }),
                        ),
                )
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
        })
    }
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
    Ok((chunk.table, chunk.index, shape, table?))
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
                let (table, index, shape, laid) = lay_out_chunk(airs, *job, ledger, recorder)?;
                head_log::chunk_laid(seq, stream_base(kind), t);
                chunks += t.elapsed().as_secs_f64();
                shapes.push((table, index, shape));
                packer.place(Key::Streamed(table, index), shape, laid)?;
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
                    let (table, index, shape, laid) = laid?;
                    shapes.push((table, index, shape));
                    packer.place(Key::Streamed(table, index), shape, laid)?;
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
    let airs = run_airs.get_or_init(|| {
        block_airs(
            program,
            opts,
            &traces.page_configs,
            &table_counts,
            &inputs,
            format.keccak_rnd,
            format.memw_aligned,
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
        records_rest: (records_tables.into_inner(), records_bytes.into_inner()),
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
    recorder: Option<&regen::Recorder>,
    keccak_rnd: crate::tables::keccak_rnd::KeccakRndForm,
    memw_aligned: crate::tables::memw_aligned::MemwAlignedOrder,
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
    let inputs = BlockAirInputs::new(
        opts,
        program,
        header.cpu_heights.clone(),
        &header.public_output_bytes,
    )?;
    let airs = run_airs.get_or_init(|| {
        block_airs(
            program,
            opts,
            &header.page_configs,
            &header.table_counts,
            &inputs,
            keccak_rnd,
            memw_aligned,
        )
    });
    marks.push(("airs", at()));
    let refs = airs.air_refs();
    let first = if header.keccak_rnd_first {
        let names: Vec<&str> = refs.iter().map(|air| air.name()).collect();
        keccak_rnd_positions(&header.table_counts, &names)?
    } else {
        Vec::new()
    };
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
            first.len(),
            first
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    let mut timed: Vec<(usize, f64)> = Vec::new();
    let mut laid_at: Vec<(usize, f64)> = Vec::new();
    let (mut narrow_tables, mut narrow_bytes) = (0usize, 0usize);
    let (mut records_tables, mut records_bytes) = (0usize, 0usize);
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
        let laid = if let Some(records) = trace.records_main() {
            // Never dropped, so never one phase B builds again.
            records_tables += 1;
            records_bytes += records.bytes();
            table_of_records(refs[position], &mut trace, shape)
        } else if let Some(packed) = trace.narrow_main() {
            narrow_tables += 1;
            narrow_bytes += packed.data().len();
            // A table phase B can build again from the kept lists (N3): its
            // place among the rest's, before the packer places it.
            if let (Some(recorder), Some((family, j))) = (
                recorder,
                crate::tables::trace_builder::RegenFamily::of_air(refs[position].name()),
            ) {
                recorder.rest_tag(k, family, j);
            }
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
    );
    // Regeneration (`BlockOptions::regen`): its recorder, when phase A can feed
    // it — its chunks generated packed. The disk store's knob, which used to
    // turn it off, refuses the prove when set.
    regen::refuse_removed_spill_words(|name| std::env::var(name).ok())?;
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
            // Class 0: the streamed chunks (the replay); class 1: KECCAK_RND's
            // and LT's tables of the rest (the kept lists, N3).
            multilinear_block::BlockRegen::new(
                std::sync::Arc::new(move |t| {
                    recorder
                        .chunk_at(t)
                        .map(|_| (0, t as u64))
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
    // Phase A's commits read the blowup, the fold schedule and the format of
    // the config and nothing else; the full config (its query count reads every
    // shape) is built when the shapes are known, and must agree on those.
    let commit_config = format.chain_config(&[]);
    let cap = commit_config.format.stack;
    let start = Instant::now();
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
    let tag = block_statement_tag(&format.zf.base);
    crate::with_block_hash!(format.zf.base, |H| {
        let walkers = block_walkers()?;
        let bitwise_last = deviations.rest_bitwise_last;
        let keccak_rnd_first = keccak_rnd_first_or_default(deviations.keccak_rnd_first);
        let from_tables = bitwise_from_tables_or_default(deviations.bitwise_from_tables);
        let card_fill = match (
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
            // The logs shared, so that with `LAMBDA_VM_EXEC_RECYCLE_LOGS`
            // the executor can take a window's buffer back once every other
            // holder has let it go.
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
                if history {
                    executor.enable_history().map_err(failed)?;
                }
                // Pre-faulted window buffers from a helper, two ahead.
                let prefault = exec_prefault()?.then(|| {
                    let (tx, rx) = std::sync::mpsc::sync_channel(2);
                    let n = window.min(1 << 22);
                    std::thread::spawn(move || {
                        loop {
                            let logs = executor::vm::logs::prefaulted_logs(n);
                            let extras = history.then(|| {
                                executor::vm::logs::prefaulted(
                                    n,
                                    executor::vm::history::HistExtra::default(),
                                )
                            });
                            if tx.send((logs, extras)).is_err() {
                                break;
                            }
                        }
                    });
                    rx
                });
                let mut prefaulted = 0usize;
                // The windows sent, while someone downstream still holds them.
                let mut sent: Option<std::collections::VecDeque<std::sync::Arc<Vec<_>>>> =
                    exec_recycle_logs()?.then(Default::default);
                let mut recycled = 0usize;
                own.setup = t.elapsed().as_secs_f64();
                eprintln!(
                    "BLOCK EXECUTOR ENGINE: {} · shadow check {} · log buffers {} · history {}",
                    kind.name(),
                    shadow_every.map_or("off".to_string(), |n| format!("every {n} windows")),
                    if sent.is_some() { "recycled" } else { "fresh" },
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
                loop {
                    let t = Instant::now();
                    // The buffers nobody else holds any more: the oldest goes
                    // back to the executor, the rest are freed.
                    if let Some(sent) = &mut sent {
                        let mut back = None;
                        sent.retain_mut(|logs| {
                            if std::sync::Arc::strong_count(logs) > 1 {
                                return true;
                            }
                            // Only this list holds it: nobody can take another
                            // share now, so the unwrap below cannot fail.
                            let held = std::mem::take(logs);
                            if back.is_none() {
                                back = std::sync::Arc::try_unwrap(held).ok();
                            }
                            false
                        });
                        if let Some(buffer) = back {
                            recycled += 1;
                            executor.recycle(buffer);
                        }
                    }
                    // The next window's buffers, when the helper has them ready.
                    if let Some(ready) = &prefault
                        && let Ok((logs, extras)) = ready.try_recv()
                    {
                        prefaulted += 1;
                        executor.recycle(logs);
                        if let Some(extras) = extras {
                            executor.recycle_extras(extras);
                        }
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
                        let window = executor.resume_history(window).map_err(failed)?;
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
                        .then(|| executor.take_history_final())
                        .flatten();
                    let logs = std::sync::Arc::new(logs);
                    if let Some(sent) = &mut sent {
                        sent.push_back(std::sync::Arc::clone(&logs));
                    }
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
                        last: end.map(Box::new),
                    };
                    if ltx.send(sent_window).is_err() {
                        break;
                    }
                    ProducerWaits::add(&waits.executor_send, t);
                }
                let executed = start.elapsed().as_secs_f64();
                drop(executor);
                if prefault.is_some() {
                    eprintln!(
                        "BLOCK EXECUTOR PREFAULT: {prefaulted} of {} windows given pre-faulted buffers",
                        own.windows
                    );
                }
                drop(prefault);
                if sent.is_some() {
                    eprintln!(
                        "BLOCK EXECUTOR RECYCLE: {recycled} of {} windows written into a buffer taken back",
                        own.windows
                    );
                }
                drop(sent);
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
                if options.finish_keccak_rnd_chunks && options.keccak_rnd_rows_log2 >= 5 {
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
                builder = builder.slide_rest();
                builder = builder.keccak_rnd_form(format.keccak_rnd);
                builder = builder.memw_aligned_order(format.memw_aligned)?;
                if card_fill.keccak_rnd {
                    builder = builder.card_fill_keccak_rnd();
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
                    // The card's `registers` mode (a test's override, no
                    // setting): the card builds MEMW_R's
                    // rows from the CPU logs, and needs each chunk's ECALL
                    // cycles, which the walkers read from the executor's
                    // history. Without record mode MEMW_R goes up as records.
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
                    if multilinear::gpu::card_rows_builds_registers() {
                        let marks = card_fill.memw_r && history && walkers > 0;
                        if marks {
                            builder = builder.card_marks_ecalls()?;
                        }
                        eprintln!(
                            "BLOCK CARD ROWS REGISTERS: {}",
                            if marks {
                                "MEMW_R's chunks go out with their ECALL cycles"
                            } else {
                                "off (no record mode, or MEMW_R not held as records): MEMW_R goes up as records"
                            }
                        );
                    }
                    #[cfg(test)]
                    if deviations.card_fill_streamed_fault > 0 {
                        builder =
                            builder.card_fill_streamed_fault(deviations.card_fill_streamed_fault);
                    }
                }
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
                    split_ecdas(&mut rest, options.ecdas_rows_log2, padding);
                    split_keccak(&mut rest, options.keccak_rows_log2, padding);
                    split_ecsm(&mut rest, options.ecsm_rows_log2, padding);
                    split_copy(&mut rest, options.copy_rows_log2, padding);
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
                        options.max_rows.cpu,
                    ),
                    RestIn::Stream(stream) => lay_out_rest_streamed(
                        stream,
                        program,
                        opts,
                        run_airs,
                        start,
                        sink,
                        ledger,
                        live_recorder,
                        format.keccak_rnd,
                        format.memw_aligned,
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
                Ok(BlockCommitted::commit_groups_logged::<
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
                ))
            };
            (block, builder.join(), layout.join(), executor.join())
        });
        stamps.head = head_log::take_lines();
        for line in &stamps.head {
            eprintln!("{line}");
        }
        (stamps.execute, stamps.executor) =
            executed.map_err(|_| Error::Prover("the block's executor panicked".into()))??;
        // A thread's error names the cause; a commit that ran out of groups only
        // says that it did.
        let (windows_done, finished, streamed, window_stamps, finish_marks, rest_regen) =
            built.map_err(|_| Error::Prover("the block's builder panicked".into()))??;
        stamps.windows = window_stamps;
        stamps.build_marks = finish_marks;
        stamps.producer = waits.stamps();
        let laid =
            laid.map_err(|_| Error::Prover("the block's layout thread panicked".into()))??;
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
        // Phase B's order (N3): the groups with nothing dropped first, then the
        // rebuilt streamed and rest groups interleaved by their cells, so each
        // regenerator keeps up with only its share of phase B's pace.
        let group_cells = group_cells(&laid.shapes, &laid.groups);
        let order: Vec<usize> = match deviations.phase_b_order {
            Some(order) => order(laid.groups.len()),
            None => led_order(&classes, &group_cells, regen_lead),
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
        // The card's `words` (the default) or `registers` mode: the card-built rows' counters as
        // phase B starts (the block's end line holds both phases).
        if (multilinear::gpu::card_rows_builds_registers()
            || multilinear::gpu::card_rows_builds_words())
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
        // The recursion's lead (`W3_LEAF_FIRST`): once the tree offers its
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
        let (proof, argues, prepared_openings, groups, commit_ood) =
            multilinear_block::block_prove_in_order::<_, _, _, H>(
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
                on_group,
                &order,
                reorder,
            )
            .map_err(|e| Error::Prover(format!("{e:?}")))?;
        stamps.phase_b = t.elapsed().as_secs_f64();
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
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(tag);
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
) -> Result<Vec<Vec<u8>>, String> {
    let mut calls = StatementCalls::default();
    absorb_block(
        &mut calls,
        tag,
        elf_bytes,
        statement.public_output,
        statement.table_counts,
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
    check_bitwise_height(statement.table_num_vars)?;
    check_chunked_heights(statement.table_counts, statement.table_num_vars)?;

    let cpu_heights = block_cpu_heights(statement.table_counts, statement.table_num_vars)?;
    let inputs = BlockAirInputs::new(
        proof_options,
        &program,
        cpu_heights,
        statement.public_output,
    )?;
    let airs = block_airs(
        &program,
        proof_options,
        &page_configs,
        statement.table_counts,
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
    );
    // The CPU heights were read at `cpu_proof_start`: the tables must be there.
    airs.check_cpu_position(statement.table_counts)?;
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
pub(crate) fn verify_block_whir_traced(
    proof: &BlockWhirProof,
    elf_bytes: &[u8],
    proof_options: &ProofOptions,
    format: &BlockFormat,
    skip_prepared: bool,
    argue_checks: VerifierChecks,
) -> Result<Result<(), BlockRefusal>, Error> {
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

    let tag = block_statement_tag(&format.zf.base);
    Ok(crate::with_block_hash!(format.zf.base, |H| {
        let mut transcript = <H as multilinear::whir_hash::WhirHash>::sponge();
        absorb_block(
            &mut transcript,
            &tag,
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
        )
        .map_err(BlockRefusal::Proof)
    }))
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
        exec_mode_from, parse_exec_history, parse_exec_history_check, parse_exec_prefault,
        parse_exec_recycle_logs,
    };

    /// `LAMBDA_VM_EXEC_PREFAULT`: off unless `1`; anything else refused.
    #[test]
    fn prefault_is_off_unless_asked() {
        assert!(!parse_exec_prefault(None).unwrap());
        assert!(!parse_exec_prefault(Some("0")).unwrap());
        assert!(parse_exec_prefault(Some("1")).unwrap());
        assert!(parse_exec_prefault(Some("yes")).is_err());
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

    /// `LAMBDA_VM_EXEC_RECYCLE_LOGS`: off unless `1`; anything else refused.
    #[test]
    fn log_recycling_is_off_unless_asked() {
        assert!(!parse_exec_recycle_logs(None).unwrap());
        assert!(!parse_exec_recycle_logs(Some("")).unwrap());
        assert!(!parse_exec_recycle_logs(Some("0")).unwrap());
        assert!(parse_exec_recycle_logs(Some(" 1 ")).unwrap());
        assert!(parse_exec_recycle_logs(Some("yes")).is_err());
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
                    .place(Key::Streamed(StreamTable::Lt, index), shape, table)
                    .expect("phase A takes it");
            }
            let (groups, _, _) = packer.finish().expect("phase A takes the last");
            groups
                .into_iter()
                .map(|keys| {
                    keys.into_iter()
                        .map(|key| match key {
                            Key::Streamed(_, index) => index,
                            Key::Air(index) => index,
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
        // ★ Unset is every family (KECCAK_RND, CPU, MEMW_R and the five row
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
            "keccak_rnd,cpu,memw_r,memw_a,load,store,memw,shift"
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
            "a gate of 4096 MiB, sliding (no waves)"
        );
        assert_eq!(rest_layout_label(&options, false), "waves of 4096 MiB");
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

    /// The rest holds up to 4 GiB with at most 2 GiB building by default; the
    /// harness's words 2048 / 2048 are the earlier default (a cap equal to the
    /// gate's budget is the gate alone).
    #[test]
    fn the_rest_holds_4_gib_with_at_most_2_gib_building_by_default() {
        assert_eq!(BLOCK_REST_LAYOUT_BYTES, 4 << 30);
        assert_eq!(BLOCK_REST_BUILD_BYTES, 2 << 30);
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
