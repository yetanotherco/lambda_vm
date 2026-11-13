//! ★ The block tree's LFM programs proved by the block's own WHIR prover.
//!
//! An LFM program's chips go through [`stark::multilinear_block`] as a small
//! block: its tables in `LfmAirs::air_refs` order, cut into groups by
//! [`stark::multilinear_block::block_groups`] at [`WLFM_GROUP_POLYS`] stacked
//! polynomials a group, each group's preprocessed prefixes in a prepared stack,
//! the batched argue, one chain per run, Poseidon1 at the block's cap. So a
//! proof here answers WHIR's commitment-time out-of-domain point on every
//! carried polynomial before `(z, α, β)` exactly as the block does
//! (`BlockCommitted::commit`, `absorb_roots_answers_and_challenge`): the block
//! prover has no format without those answers.
//!
//! What this module adds is the glue around it:
//!
//! - the **artifacts** ([`WlfmArtifacts`]): heights, groups and every prepared
//!   stack's roots, derived from the program — the verifier's, never a proof's;
//! - the **statement** ([`absorb_wlfm_statement`]), bound before any root;
//! - the **identity** ([`wlfm_program_id`]), which folds every prepared root and
//!   the chain format, so a parent that holds it as a constant pins the program
//!   and the format it was proved under.
//!
//! A proof whose verifier is the host may bind the instruction columns by
//! EVALUATION instead ([`WlfmPrep::Evaluated`]): the verifier holds the columns
//! it derived and checks each prefix at its table's reduced point against the
//! main opening, so no prepared stack is committed or opened. The prefix count
//! is still the AIR's ([`check_evaluated`]).
//!
//! # ⛔ The preprocessed count comes from the AIR
//!
//! The LFM AIRs declare their preprocessed columns by COUNT only
//! (`with_preprocessed`), so `precomputed_columns()` is empty. A statement
//! built from that list would count none, and nothing would bind the program's
//! columns. Every statement here takes the AIR's count
//! ([`TableLayout::statement_with_prepared_prefix`]), and [`check_coverage`] —
//! run by the build, the verifier and the in-guest leg — refuses any plan in
//! which a table's prefix is not settled exactly, by one prepared stack of its
//! own group.
//!
//! # The format is a constant
//!
//! [`wlfm_config`] derives the chain config from [`ZfFormat::DEFAULT`] and a
//! [`WlfmRate`], never from the process's environment: both sides of a proof
//! name the rate, and the identity folds the config it gives.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::hash::platform_keccak::PlatformKeccak256 as Keccak256;
use digest::Digest;
use math::field::element::FieldElement;
use math::field::traits::IsPrimeField;
use multilinear::mle::Mle;
use multilinear::stacked_eval::StackedProof;
use multilinear::stacking::StackedLayout;
use multilinear::whir::Domain;
use multilinear::whir_chain::{ArgueFormat, ChainConfig};
use multilinear::whir_commit::Commitment;
use multilinear::whir_hash::WhirHash;
use stark::multilinear_block::{
    self, BlockCommitted, BlockPrepared, BlockPreparedCheck, Narrowing,
};
use stark::multilinear_table::batched::VerifierChecks;
use stark::multilinear_table::{BatchedArgue, CommittedTable, MultiProof, TableStatement};
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};

use crate::block_whir::{
    GroupPrepared, PreparedColumns, commit_groups, commit_groups_on_host, group_stacks,
};
use crate::multilinear_prove::{layout_of, stacks};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::zf_format::ZfFormat;

use super::airs::{ChipSet, LfmAirs};
use super::airs::{DynLfmAir, NUM_LFM_CHIPS};
use super::compiler::ColumnGroup;
use super::compiler::LfmProgram;
use super::hash::HasherKind;
use super::proof::LfmFilled;
use super::statement::LFM_MACHINE_VERSION;
use super::trace::range_group;
use super::word::LfmWord;
use std::borrow::Cow;

type F = GoldilocksField;
type E = GoldilocksExtension;

/// The hash every W-LFM proof commits, transcripts and grinds with: the block
/// family's ([`crate::hash_pin::BlockWhirHash`], Poseidon1 at width 16), so a
/// parent verifies a child with the socket its own hash rows run on.
pub type WlfmHash = crate::hash_pin::BlockWhirHash;

/// The transcript of a W-LFM proof: [`WlfmHash`]'s sponge.
pub type WlfmTranscript = <WlfmHash as WhirHash>::Sponge;

/// Stacked polynomials a W-LFM group may take: four, so a doubled leaf (four
/// polynomials of 2^27) is one group, one chain and one prepared stack. The
/// base's groups take three (`block_whir::BLOCK_GROUP_POLYS`).
pub const WLFM_GROUP_POLYS: usize = 4;

/// Queries a chain runs at [`WlfmRate::Four`]: the fewest that keep the unique-
/// decoding figure at the base's 113.04 bits, `141 · log2(8/5) + 18 = 113.61`
/// (140 would give 112.93). At blowup 2 the count is the base's own derivation.
pub const WLFM_B4_QUERIES: usize = 141;

/// The statement's domain tag.
pub const WLFM_STATEMENT_TAG: &[u8] = b"LAMBDAVM_LFM_WHIR_BLOCK_STATEMENT_V1/P1W16";

/// The identity's domain tag.
const WLFM_PROGRAM_TAG: &[u8] = b"LAMBDAVM_LFM_WHIR_BLOCK_PROGRAM_V1";

/// The code rate a W-LFM proof runs at: a verifier constant, named by both
/// sides and folded into the identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WlfmRate {
    /// Blowup 2, the base's format (`ZfFormat::DEFAULT`): Q 229 at production
    /// heights, 113.04 bits under unique decoding.
    Two,
    /// Blowup 4 at [`WLFM_B4_QUERIES`]: fewer queries for every parent, a
    /// twice-larger codeword for the prover.
    Four,
    /// Blowup 16 at the production query rule (128 conjectured bits, the
    /// format's query grind), under a 2^[`WLFM_B16_STACK`] stack: the
    /// compression step's rate, where a query buys the most bytes.
    Sixteen,
}

impl WlfmRate {
    /// The code's inverse rate.
    pub const fn factor(self) -> usize {
        match self {
            WlfmRate::Two => 2,
            WlfmRate::Four => 4,
            WlfmRate::Sixteen => 16,
        }
    }
}

/// How a W-LFM proof binds its program's instruction columns (the tables'
/// preprocessed prefixes): a verifier constant, named by both sides and folded
/// into the identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WlfmPrep {
    /// One prepared stack a group, committed from the program and opened at
    /// each table's reduced point: what a parent that verifies the proof
    /// in-guest needs, since it cannot hold the columns. Today's format.
    Stacked,
    /// The verifier holds the columns (it derived the program) and evaluates
    /// each prefix at its table's reduced point itself against the values the
    /// main opening gives: no prepared commit, no prepared opening. Only for a
    /// proof whose verifier is the host (the last proof of the pipeline),
    /// D-COMPRESS lever L1.
    Evaluated,
}

/// The stack cap at [`WlfmRate::Sixteen`]: a grouped chain lives on
/// `H_{n + m + lb}`, and Goldilocks' two-adicity (32) leaves `32 − 4 − 2 = 26`
/// stacked variables at blowup 16 and the widest index; the default 27 would
/// need 33.
pub const WLFM_B16_STACK: usize = 26;

/// ★ The chain config a W-LFM program proves under, over its table shapes:
/// the pinned block format's chain
/// ([`crate::block_whir::BlockFormat::chain_config`]: its argue, group fold,
/// round-0 leaf order and trees) at [`ZfFormat::DEFAULT`] with `rate`'s blowup
/// (and, at blowup 16, the [`WLFM_B16_STACK`] stack). The ONE call the build,
/// the prover and the verifier make.
pub fn wlfm_config(rate: WlfmRate, shapes: &[(usize, usize)]) -> ChainConfig {
    let mut zf = ZfFormat {
        whir_blowup: rate.factor() as u8,
        ..ZfFormat::DEFAULT
    };
    if rate == WlfmRate::Sixteen {
        zf.whir_stack = multilinear::whir_chain::StackVars::new(WLFM_B16_STACK)
            .expect("the blowup-16 stack is in range");
    }
    let format = crate::block_whir::BlockFormat {
        zf,
        ..crate::block_whir::BlockFormat::pinned()
    };
    let mut config = format.chain_config(shapes);
    if rate == WlfmRate::Four {
        config.num_queries = WLFM_B4_QUERIES;
    }
    #[cfg(test)]
    let config = test_format::apply(config);
    config
}

/// The AIR set's options: inert on the multilinear path (they reach only each
/// AIR's STARK context), fixed so no caller's options move a layout.
fn air_options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("blowup 2 is valid")
}

/// One group's prepared stack, as the build derived it.
#[derive(Clone, Debug)]
pub struct WlfmStack {
    pub group: usize,
    /// `(table, prefix length)` in stack order; a table is its position in
    /// `LfmAirs::air_refs` order, which is the proof's table order.
    pub tables: Vec<(usize, usize)>,
    /// Derived from the program, never read from a proof.
    pub roots: Vec<Commitment>,
    pub layout: StackedLayout,
    pub domain: Domain<F>,
}

/// ★ Everything a verifier of a W-LFM program needs, derived from the program.
#[derive(Clone, Debug)]
pub struct WlfmArtifacts {
    pub rate: WlfmRate,
    /// The `LFM_HASH` permutation the program runs.
    pub hasher: HasherKind,
    /// The WHIR recursion chip set ([`check_chip_set`]).
    pub chip_set: ChipSet,
    /// `LFM_HASH` instances.
    pub hash_chunks: usize,
    /// Each table's height in variables: the program's padded rows.
    pub table_num_vars: Vec<u8>,
    /// Each table's preprocessed count, from its AIR.
    pub counts: Vec<usize>,
    /// Tables per group ([`stark::multilinear_block::block_groups`]).
    pub sizes: Vec<usize>,
    pub config: ChainConfig,
    /// How the instruction columns are bound.
    pub prep: WlfmPrep,
    /// Under [`WlfmPrep::Stacked`]: the prepared stacks. Empty otherwise.
    pub stacks: Vec<WlfmStack>,
    /// Under [`WlfmPrep::Evaluated`]: each table's preprocessed prefix, as the
    /// verifier evaluates it (empty for a table with none). Empty otherwise.
    pub evaluated: Vec<Vec<Mle<F>>>,
    pub program_id: Commitment,
}

impl WlfmArtifacts {
    /// The program's AIR set, with no STARK roots ([`whir_lfm_airs`]).
    pub fn airs(&self) -> LfmAirs {
        whir_lfm_airs(self.hasher, self.chip_set, self.hash_chunks, &air_options())
    }

    /// `(main width, height)` per table.
    pub fn shapes(&self, airs: &LfmAirs) -> Vec<(usize, usize)> {
        table_shapes(&airs.air_refs(), &self.table_num_vars)
    }

    /// Each group's tables, as positions.
    pub fn groups(&self) -> Vec<Vec<usize>> {
        let mut at = 0usize;
        self.sizes
            .iter()
            .map(|&size| {
                let group = (at..at + size).collect();
                at += size;
                group
            })
            .collect()
    }

    /// Where group `g`'s first table sits.
    pub fn group_start(&self, g: usize) -> usize {
        self.sizes[..g].iter().sum()
    }

    /// The prepared roots in absorb order.
    pub fn derived_roots(&self) -> Vec<Commitment> {
        self.stacks
            .iter()
            .flat_map(|s| s.roots.iter().copied())
            .collect()
    }
}

/// A program's artifacts and the prover's half of its prepared stacks.
pub struct WlfmBuild {
    pub artifacts: WlfmArtifacts,
    stacks: Vec<GroupPrepared<WlfmHash>>,
}

#[cfg(test)]
impl WlfmBuild {
    /// The build with every prepared stack dropped, artifacts and prover alike:
    /// the shape a verifier that counted no preprocessed columns would accept.
    pub(crate) fn without_stacks(mut self) -> Self {
        self.artifacts.stacks.clear();
        self.stacks.clear();
        self
    }
}

/// A W-LFM proof and the words its execution published.
#[derive(Clone, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct WlfmProof {
    pub proof: MultiProof<F, E>,
    /// One batched argue per group.
    pub argues: Vec<BatchedArgue<E>>,
    /// One opening per prepared stack, in stack order.
    pub prepared: Vec<StackedProof<F, E>>,
    /// Every carried polynomial's commit-time answer, in root order.
    pub commit_ood: Vec<FieldElement<E>>,
    /// The public output, in emission order — indices `0..n`.
    pub public_words: Vec<(u32, LfmWord)>,
}

/// ⛔ THE COVERAGE RULE: every table whose AIR declares preprocessed columns
/// is settled by exactly one prepared stack — of its own group, over exactly
/// its `count` leading columns — and no other table appears in any stack.
///
/// `stacks` is `(group, (table, prefix) list)` per stack. One rule for the
/// build, the host verifier and the in-guest leg: a plan that fails it describes
/// a proof whose program columns something does not bind.
pub fn check_coverage(
    counts: &[usize],
    sizes: &[usize],
    stacks: &[(usize, &[(usize, usize)])],
) -> Result<(), String> {
    let mut group_of = Vec::with_capacity(counts.len());
    for (g, &size) in sizes.iter().enumerate() {
        group_of.extend(std::iter::repeat_n(g, size));
    }
    if group_of.len() != counts.len() {
        return Err(format!(
            "{} groups' tables for {} tables",
            group_of.len(),
            counts.len()
        ));
    }
    let mut settled: Vec<Option<usize>> = vec![None; counts.len()];
    for &(group, tables) in stacks {
        for &(table, n) in tables {
            let Some(&home) = group_of.get(table) else {
                return Err(format!(
                    "a prepared stack settles table {table}, which no group holds"
                ));
            };
            if home != group {
                return Err(format!(
                    "table {table} is in group {home} and a stack of group {group} settles it"
                ));
            }
            if settled[table].replace(n).is_some() {
                return Err(format!("table {table} is settled twice"));
            }
        }
    }
    for (table, (&count, settled)) in counts.iter().zip(&settled).enumerate() {
        match (count, settled) {
            (0, None) => {}
            (0, Some(n)) => {
                return Err(format!(
                    "table {table} has no preprocessed columns and a stack settles {n}"
                ));
            }
            (count, Some(n)) if *n == count => {}
            (count, Some(n)) => {
                return Err(format!(
                    "table {table} has {count} preprocessed columns and its stack settles {n}"
                ));
            }
            (count, None) => {
                return Err(format!(
                    "table {table} has {count} preprocessed columns and no stack settles them"
                ));
            }
        }
    }
    Ok(())
}

/// ★ Builds a program's artifacts and the prover's prepared stacks at `rate`,
/// the prepared commits on the card where there is one: the PROVER's build.
///
/// ONE derivation for both sides: the prover and every verifier of this program
/// take the groups, the stacks' roots, layouts and domains from here; a
/// verifier commits on the host ([`build_wlfm_on_host`]).
pub fn build_wlfm(
    program: &LfmProgram,
    hasher: HasherKind,
    rate: WlfmRate,
) -> Result<WlfmBuild, String> {
    build_wlfm_under(program, hasher, rate, WlfmPrep::Stacked, false)
}

/// [`build_wlfm`] with every prepared commit on the host: the VERIFIER's
/// build, so no card kernel is on its acceptance path (the block verifier's
/// rule since the joint round-0 trees, `block_whir::commit_groups_on_host`).
pub fn build_wlfm_on_host(
    program: &LfmProgram,
    hasher: HasherKind,
    rate: WlfmRate,
) -> Result<WlfmBuild, String> {
    build_wlfm_under(program, hasher, rate, WlfmPrep::Stacked, true)
}

/// [`build_wlfm`] at `rate` with the instruction columns bound per `prep`,
/// the prepared commits (under [`WlfmPrep::Stacked`]) on the host when
/// `on_host` (a verifier's build).
pub fn build_wlfm_under(
    program: &LfmProgram,
    hasher: HasherKind,
    rate: WlfmRate,
    prep: WlfmPrep,
    on_host: bool,
) -> Result<WlfmBuild, String> {
    // The mask with `BITWISE` dropped where nothing sends it: the default, as a
    // constant (`for_program_with_hasher` reads it from the environment).
    let chip_set = ChipSet::for_program_under(program, hasher, true);
    check_chip_set(chip_set)?;
    let hash_chunks = program.hash_chunk_count();
    let groups = program_prep_groups(program);
    let airs = whir_lfm_airs(hasher, chip_set, hash_chunks, &air_options());
    let refs = airs.air_refs();
    if refs.len() != groups.len() {
        return Err(format!(
            "the AIR set has {} tables and the program commits {} instruction groups",
            refs.len(),
            groups.len()
        ));
    }
    let mut table_num_vars = Vec::with_capacity(refs.len());
    let mut counts = Vec::with_capacity(refs.len());
    for (air, group) in refs.iter().zip(&groups) {
        let count = air.num_precomputed_columns();
        if group.width != count {
            return Err(format!(
                "{}: the program commits {} preprocessed columns and the AIR declares {count}",
                air.name(),
                group.width
            ));
        }
        if !group.padded_rows.is_power_of_two() {
            return Err(format!(
                "{}: {} rows, which the hypercube cannot hold",
                air.name(),
                group.padded_rows
            ));
        }
        table_num_vars.push(group.padded_rows.trailing_zeros() as u8);
        counts.push(count);
    }
    let shapes = table_shapes(&refs, &table_num_vars);
    let config = wlfm_config(rate, &shapes);
    let sizes = multilinear_block::block_groups(&shapes, config.format.stack, WLFM_GROUP_POLYS)
        .map_err(|e| format!("{e:?}"))?;
    let group_lists: Vec<Vec<u32>> = {
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
    let columns: Vec<PreparedColumns> = groups
        .iter()
        .zip(&counts)
        .enumerate()
        .filter(|&(_, (_, &count))| count > 0)
        .map(|(table, (group, _))| (table, group_columns(group)))
        .collect();
    if prep == WlfmPrep::Evaluated {
        let mut evaluated: Vec<Vec<Mle<F>>> = vec![Vec::new(); counts.len()];
        for (table, cols) in columns {
            evaluated[table] = cols
                .into_iter()
                .map(|c| Mle::new(c).map_err(|e| format!("a preprocessed column: {e:?}")))
                .collect::<Result<_, _>>()?;
        }
        check_evaluated(&counts, &table_num_vars, &evaluated)?;
        let mut artifacts = WlfmArtifacts {
            rate,
            hasher,
            chip_set,
            hash_chunks,
            table_num_vars,
            counts,
            sizes,
            config,
            prep,
            stacks: Vec::new(),
            evaluated,
            program_id: [0u8; 32],
        };
        artifacts.program_id = wlfm_program_id(&artifacts);
        return Ok(WlfmBuild {
            artifacts,
            stacks: Vec::new(),
        });
    }
    let stacks = group_stacks(columns, &group_lists).map_err(|e| format!("{e:?}"))?;
    // ⛔ THE CARD, as `program_census::build_artifacts_counted` holds it for a
    // STARK build: the prepared commits run on the device, so concurrent
    // siblings serialize them. Inert unless a driver armed the permit.
    let committed = if on_host {
        commit_groups_on_host::<WlfmHash>(stacks, &config).map_err(|e| format!("{e:?}"))?
    } else {
        let _card = super::device_permit::hold_labeled("build_artifacts");
        commit_groups::<WlfmHash>(stacks, &config).map_err(|e| format!("{e:?}"))?
    };
    let stacks: Vec<WlfmStack> = committed
        .iter()
        .map(|p| WlfmStack {
            group: p.group,
            tables: p.tables.clone(),
            roots: p.roots.clone(),
            layout: p.commitment.layout().clone(),
            domain: p.commitment.domain().clone(),
        })
        .collect();
    let plan: Vec<(usize, &[(usize, usize)])> = stacks
        .iter()
        .map(|s| (s.group, s.tables.as_slice()))
        .collect();
    check_coverage(&counts, &sizes, &plan)?;
    let mut artifacts = WlfmArtifacts {
        rate,
        hasher,
        chip_set,
        hash_chunks,
        table_num_vars,
        counts,
        sizes,
        config,
        prep,
        stacks,
        evaluated: Vec::new(),
        program_id: [0u8; 32],
    };
    artifacts.program_id = wlfm_program_id(&artifacts);
    Ok(WlfmBuild {
        artifacts,
        stacks: committed,
    })
}

/// ⛔ THE EVALUATED RULE ([`WlfmPrep::Evaluated`]): every table holds exactly
/// its AIR's count of preprocessed columns, each over the table's height. One
/// rule for the build and the verifier: a table whose prefix the verifier does
/// not evaluate in full is a table whose program columns nothing binds.
pub fn check_evaluated(
    counts: &[usize],
    table_num_vars: &[u8],
    evaluated: &[Vec<Mle<F>>],
) -> Result<(), String> {
    if evaluated.len() != counts.len() || table_num_vars.len() != counts.len() {
        return Err(format!(
            "{} evaluated prefixes and {} heights for {} tables",
            evaluated.len(),
            table_num_vars.len(),
            counts.len()
        ));
    }
    for (table, ((&count, &num_vars), cols)) in
        counts.iter().zip(table_num_vars).zip(evaluated).enumerate()
    {
        if cols.len() != count {
            return Err(format!(
                "table {table} has {count} preprocessed columns and the verifier evaluates {}",
                cols.len()
            ));
        }
        if let Some(c) = cols.iter().find(|c| c.num_vars() != num_vars as usize) {
            return Err(format!(
                "table {table} is 2^{num_vars} rows and a preprocessed column 2^{}",
                c.num_vars()
            ));
        }
    }
    Ok(())
}

/// A Merkle cap policy as two bytes: a variant tag and its height.
fn cap_bytes(cap: crypto::merkle_tree::cap::CapPolicy) -> [u8; 2] {
    use crypto::merkle_tree::cap::CapPolicy;
    match cap {
        CapPolicy::Off => [0, 0],
        CapPolicy::Auto => [1, 0],
        CapPolicy::Fixed(h) => [2, h],
    }
}

/// ★ The identity a parent holds for a W-LFM child: keccak over the tag, the
/// machine version, the hasher and the chip set, the WHIR hash's name, the rate,
/// every table's height and preprocessed count, the groups, every prepared
/// stack (its group, its tables and its roots, which bind the instruction
/// columns themselves), and the chain config's every word. Everything that
/// makes the program this program under this format; nothing a proof states.
pub fn wlfm_program_id(a: &WlfmArtifacts) -> Commitment {
    let mut h = Keccak256::new();
    h.update(WLFM_PROGRAM_TAG);
    h.update(LFM_MACHINE_VERSION.to_le_bytes());
    h.update([a.hasher.as_tag(), a.chip_set.as_tag()]);
    h.update(<WlfmHash as WhirHash>::NAME.as_bytes());
    h.update((a.rate.factor() as u64).to_le_bytes());
    h.update((a.hash_chunks as u64).to_le_bytes());
    h.update((a.table_num_vars.len() as u64).to_le_bytes());
    for (&num_vars, &count) in a.table_num_vars.iter().zip(&a.counts) {
        h.update([num_vars]);
        h.update((count as u64).to_le_bytes());
    }
    h.update((a.sizes.len() as u64).to_le_bytes());
    for &size in &a.sizes {
        h.update((size as u64).to_le_bytes());
    }
    h.update((a.stacks.len() as u64).to_le_bytes());
    for stack in &a.stacks {
        h.update((stack.group as u64).to_le_bytes());
        h.update((stack.tables.len() as u64).to_le_bytes());
        for &(table, n) in &stack.tables {
            h.update((table as u64).to_le_bytes());
            h.update((n as u64).to_le_bytes());
        }
        h.update((stack.roots.len() as u64).to_le_bytes());
        for root in &stack.roots {
            h.update(root);
        }
    }
    let c = &a.config;
    for word in [
        c.log_blowup as u64,
        c.log_folding as u64,
        c.num_queries as u64,
        c.fold_word(),
        c.format.stack.get() as u64,
    ] {
        h.update(word.to_le_bytes());
    }
    h.update([c.grind.folding, c.grind.ood, c.grind.query]);
    h.update(cap_bytes(c.format.cap));
    h.update(cap_bytes(c.format.arity4_cap));
    match c.format.argue {
        ArgueFormat::PerTable => h.update([0u8, 0]),
        ArgueFormat::Batched { bin_log_cells, .. } => h.update([1u8, bin_log_cells]),
    }
    // The rest of the format, whole: a lever added later moves the identity.
    h.update(format!("{:?}", c.format).as_bytes());
    // Under the evaluated binding there are no roots to fold: the columns
    // themselves, every value, one digest a column. (The stacked binding's
    // identity is unchanged.)
    if a.prep == WlfmPrep::Evaluated {
        h.update(b"EVALUATED");
        for cols in &a.evaluated {
            h.update((cols.len() as u64).to_le_bytes());
            for digest in column_digests(cols) {
                h.update(digest);
            }
        }
    }
    h.finalize().into()
}

/// Each column's Keccak-256 over its length and its values, eight
/// little-endian canonical bytes each: the evaluated identity's leaves.
fn column_digests(cols: &[Mle<F>]) -> Vec<Commitment> {
    let digest = |col: &Mle<F>| -> Commitment {
        let evals = col.evals();
        let mut bytes = Vec::with_capacity(8 * (evals.len() + 1));
        bytes.extend_from_slice(&(evals.len() as u64).to_le_bytes());
        for v in evals {
            bytes.extend_from_slice(&F::canonical(v.value()).to_le_bytes());
        }
        let mut h = Keccak256::new();
        h.update(&bytes);
        h.finalize().into()
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        cols.par_iter().map(digest).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        cols.iter().map(digest).collect()
    }
}

/// A word's four lanes, each its canonical value as eight big-endian bytes:
/// under Poseidon1 one 32-byte `append_bytes` is its length and these four
/// felts, which the in-guest leg absorbs as the word's unpacked lanes.
pub fn word_bytes(word: &LfmWord) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (lane, out) in word.iter().zip(bytes.chunks_mut(8)) {
        out.copy_from_slice(&F::canonical(lane.value()).to_be_bytes());
    }
    bytes
}

/// The statement's constant calls, before the words: the tag, the program id,
/// and the machine version with the word count. The in-guest leg absorbs these
/// as program constants, call for call.
pub fn statement_head(program_id: &Commitment, num_words: usize) -> [Vec<u8>; 3] {
    let mut counts = Vec::with_capacity(16);
    counts.extend_from_slice(&u64::from(LFM_MACHINE_VERSION).to_le_bytes());
    counts.extend_from_slice(&(num_words as u64).to_le_bytes());
    [WLFM_STATEMENT_TAG.to_vec(), program_id.to_vec(), counts]
}

/// ★ Binds a W-LFM statement before any root: [`statement_head`]'s calls, then
/// each public word as one 32-byte call ([`word_bytes`]). The word indices are
/// not absorbed: they are `0..n` by construction, and the verifier refuses any
/// other claim.
pub fn absorb_wlfm_statement(
    transcript: &mut impl IsTranscript<E>,
    program_id: &Commitment,
    public_words: &[(u32, LfmWord)],
) {
    for call in statement_head(program_id, public_words.len()) {
        transcript.append_bytes(&call);
    }
    for (_, word) in public_words {
        transcript.append_bytes(&word_bytes(word));
    }
}

/// ★ Proves a filled program against its build. The traces must have been
/// filled for the build's hasher.
pub(crate) fn prove_wlfm_filled(build: &WlfmBuild, filled: LfmFilled) -> Result<WlfmProof, String> {
    prove_wlfm_traces(build, filled, true)
}

/// [`prove_wlfm_filled`] with the prover's prefix check `check_prefix`ed or not:
/// only a test turns it off, to show the VERIFIER refuses a forged program.
pub(crate) fn prove_wlfm_traces(
    build: &WlfmBuild,
    filled: LfmFilled,
    check_prefix: bool,
) -> Result<WlfmProof, String> {
    let (mut traces, public_words, hasher, execute_secs, fill_secs, exec_split) =
        filled.into_parts();
    // Every chip its own table: a set filled under the process's merge is
    // split back (a copy; [`execute_and_fill_unmerged`] fills it so).
    traces.merge_to(&Vec::new())?;
    let artifacts = &build.artifacts;
    if hasher != artifacts.hasher {
        return Err(format!(
            "the traces were filled for {hasher:?} and the artifacts built for {:?}",
            artifacts.hasher
        ));
    }
    let t_prep = std::time::Instant::now();
    let airs = artifacts.airs();
    let shapes = artifacts.shapes(&airs);
    let pairs = airs.air_trace_pairs(&mut traces);
    if pairs.len() != shapes.len() {
        return Err(format!(
            "{} traces for {} tables",
            pairs.len(),
            shapes.len()
        ));
    }
    // Where each table's prefix sits among the prover's stacks' columns.
    let mut prefix_of: Vec<Option<(usize, usize, usize)>> = vec![None; shapes.len()];
    for (s, stack) in build.stacks.iter().enumerate() {
        let mut at = 0usize;
        for &(table, n) in &stack.tables {
            prefix_of[table] = Some((s, at, n));
            at += n;
        }
    }
    let mut tables: Vec<CommittedTable<'_, F, E>> = Vec::with_capacity(pairs.len());
    for (t, ((air, trace, _), &(width, num_vars))) in pairs.into_iter().zip(&shapes).enumerate() {
        if trace.main_table.width != width || trace.main_table.height != 1usize << num_vars {
            return Err(format!(
                "{}: the trace is {} columns of {} rows, the program's {width} of 2^{num_vars}",
                air.name(),
                trace.main_table.width,
                trace.main_table.height
            ));
        }
        let mut columns = trace.columns_main();
        if check_prefix && let Some((s, at, n)) = prefix_of[t] {
            for (column, pinned) in columns[..n]
                .iter()
                .zip(&build.stacks[s].columns[at..at + n])
            {
                if column.as_slice() != pinned.evals() {
                    return Err(format!(
                        "{}: a preprocessed column is not the program's",
                        air.name()
                    ));
                }
            }
        }
        if check_prefix && artifacts.prep == WlfmPrep::Evaluated {
            for (column, pinned) in columns.iter().zip(&artifacts.evaluated[t]) {
                if column.as_slice() != pinned.evals() {
                    return Err(format!(
                        "{}: a preprocessed column is not the program's",
                        air.name()
                    ));
                }
            }
        }
        let layout = layout_of(air, width, num_vars).map_err(|e| format!("{e:?}"))?;
        tables.push(
            CommittedTable::from_layout(layout, |col| core::mem::take(&mut columns[col as usize]))
                .map_err(|e| format!("{}: {e:?}", air.name()))?,
        );
    }
    let prep_secs = t_prep.elapsed().as_secs_f64();
    let waited_before = super::device_permit::waited_secs();
    let t_held = std::time::Instant::now();
    // ⛔ THE CARD, for the prove's device phases: the permit the STARK prover
    // holds around `multi_prove`. Taken after the host prep above.
    let _card = super::device_permit::hold_labeled("multi_prove");
    let config = &artifacts.config;
    let mut transcript = <WlfmHash as WhirHash>::sponge();
    absorb_wlfm_statement(&mut transcript, &artifacts.program_id, &public_words);
    let block = BlockCommitted::commit::<WlfmHash, WlfmTranscript>(
        tables,
        &artifacts.sizes,
        config,
        multilinear::whir_commit::TreeDrop::uniform(0),
        Narrowing::Wide,
        false,
    )
    .map_err(|e| format!("{e:?}"))?;
    let borrowed: Vec<Vec<&multilinear::mle::Mle<F>>> = build
        .stacks
        .iter()
        .map(|p| multilinear::stacking::borrow(&p.columns))
        .collect();
    let openings: Vec<BlockPrepared<'_, F, WlfmHash>> = build
        .stacks
        .iter()
        .zip(&borrowed)
        .map(|(p, columns)| BlockPrepared {
            group: p.group,
            tables: p.tables.clone(),
            commitment: &p.commitment,
            columns,
        })
        .collect();
    let (proof, argues, prepared, _stamps, commit_ood) =
        multilinear_block::block_prove_on_forks::<F, E, WlfmTranscript, WlfmHash>(
            block,
            config,
            &mut transcript,
            &openings,
            &[],
            &|g| g,
            &multilinear_block::ArgueDeviation::default(),
        )
        .map_err(|e| format!("{e:?}"))?;
    let permit_wait = (super::device_permit::waited_secs() - waited_before).max(0.0);
    let held = (t_held.elapsed().as_secs_f64() - permit_wait).max(0.0);
    super::proof::record_prove_split(super::proof::ProveSplit {
        execute: execute_secs,
        fill: fill_secs + prep_secs,
        multi_prove: held,
        permit_wait,
        exec: exec_split,
    });
    Ok(WlfmProof {
        proof,
        argues,
        prepared,
        commit_ood,
        public_words,
    })
}

/// `program` executed over `arenas` and its traces filled for `hasher` with
/// every chip in its own table: the W-LFM AIR set is unmerged.
pub(crate) fn execute_and_fill_unmerged(
    program: &LfmProgram,
    arenas: &[Vec<LfmWord>],
    hasher: HasherKind,
) -> Result<LfmFilled, String> {
    let t = std::time::Instant::now();
    let execution = super::executor::execute(program, arenas, &hasher)
        .map_err(|e| format!("the program does not execute: {e:?}"))?;
    Ok(super::proof::lfm_fill_executed_under(
        program,
        execution,
        hasher,
        t.elapsed().as_secs_f64(),
        &Vec::new(),
    ))
}

/// ★ Executes `program` over `arenas`, fills its traces and proves them.
pub fn prove_wlfm(
    program: &LfmProgram,
    build: &WlfmBuild,
    arenas: &[Vec<LfmWord>],
) -> Result<WlfmProof, String> {
    let filled = execute_and_fill_unmerged(program, arenas, build.artifacts.hasher)?;
    prove_wlfm_filled(build, filled)
}

/// The tables' shapes, and the groups' stacked layouts and domains.
type VerifierFrame = (Vec<(usize, usize)>, Vec<StackedLayout>, Vec<Domain<F>>);

/// Each table's statement at its AIR's preprocessed count, the stacks' checks'
/// inputs, and the groups' layouts and domains: what the verifier derives from
/// the artifacts. Refuses artifacts that do not describe their own program's
/// AIRs and format.
fn verifier_frame(artifacts: &WlfmArtifacts, airs: &LfmAirs) -> Result<VerifierFrame, String> {
    let refs = airs.air_refs();
    if refs.len() != artifacts.table_num_vars.len() || refs.len() != artifacts.counts.len() {
        return Err(format!(
            "the AIR set has {} tables and the artifacts state {} heights",
            refs.len(),
            artifacts.table_num_vars.len()
        ));
    }
    for (air, &count) in refs.iter().zip(&artifacts.counts) {
        if air.num_precomputed_columns() != count {
            return Err(format!(
                "{}: the artifacts count {count} preprocessed columns and the AIR {}",
                air.name(),
                air.num_precomputed_columns()
            ));
        }
    }
    let shapes = artifacts.shapes(airs);
    if wlfm_config(artifacts.rate, &shapes) != artifacts.config {
        return Err("the artifacts' config is not their rate's over their shapes".into());
    }
    let sizes =
        multilinear_block::block_groups(&shapes, artifacts.config.format.stack, WLFM_GROUP_POLYS)
            .map_err(|e| format!("{e:?}"))?;
    if sizes != artifacts.sizes {
        return Err("the artifacts' groups are not their shapes'".into());
    }
    let plan: Vec<(usize, &[(usize, usize)])> = artifacts
        .stacks
        .iter()
        .map(|s| (s.group, s.tables.as_slice()))
        .collect();
    match artifacts.prep {
        WlfmPrep::Stacked => {
            if !artifacts.evaluated.is_empty() {
                return Err("stacked artifacts carry evaluated columns".into());
            }
            check_coverage(&artifacts.counts, &artifacts.sizes, &plan)?;
        }
        WlfmPrep::Evaluated => {
            if !artifacts.stacks.is_empty() {
                return Err("evaluated artifacts carry prepared stacks".into());
            }
            check_evaluated(
                &artifacts.counts,
                &artifacts.table_num_vars,
                &artifacts.evaluated,
            )?;
        }
    }
    let (layouts, domains) =
        stacks(&shapes, &artifacts.sizes, &artifacts.config).map_err(|e| format!("{e:?}"))?;
    Ok((shapes, layouts, domains))
}

/// ★ Verifies a W-LFM proof against the program's artifacts and the CLAIMED
/// public words: `Ok` exactly when the proof is accepted.
///
/// The statement over the claimed words; the roots block replayed on a fork for
/// `(z, α)`; the `LfmPublic` balance the tables must reach, from the claimed
/// words; then `block_verify_with` with every statement at its AIR's
/// preprocessed count and the prepared checks built from the artifacts.
pub fn verify_wlfm(
    artifacts: &WlfmArtifacts,
    proof: &WlfmProof,
    claimed: &[(u32, LfmWord)],
) -> Result<(), String> {
    verify_wlfm_with(artifacts, proof, claimed, false, VerifierChecks::ALL)
}

/// [`verify_wlfm`] with the prepared openings skipped or the batched argue's
/// checks weakened — a mutation: a test shows each check is what refuses its
/// forgery. Production passes `false` and [`VerifierChecks::ALL`].
pub(crate) fn verify_wlfm_with(
    artifacts: &WlfmArtifacts,
    proof: &WlfmProof,
    claimed: &[(u32, LfmWord)],
    skip_prepared: bool,
    checks: VerifierChecks,
) -> Result<(), String> {
    for (position, (index, _)) in claimed.iter().enumerate() {
        if *index as usize != position {
            return Err(format!(
                "public word {position} claims index {index}; a W-LFM statement binds words by \
                 position"
            ));
        }
    }
    let airs = artifacts.airs();
    let (shapes, layouts, domains) = verifier_frame(artifacts, &airs)?;
    let refs = airs.air_refs();
    let table_layouts = refs
        .iter()
        .zip(&shapes)
        .map(|(air, &(width, num_vars))| layout_of(*air, width, num_vars))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{e:?}"))?;
    let statements: Vec<TableStatement<'_, F, E>> = table_layouts
        .iter()
        .zip(&artifacts.counts)
        .enumerate()
        .map(|(t, (layout, &count))| match artifacts.prep {
            WlfmPrep::Stacked => layout.statement_with_prepared_prefix(count),
            // The verifier's own columns: their count is the AIR's
            // (`check_evaluated`), and each is checked at the table's point.
            WlfmPrep::Evaluated => layout.statement_with_preprocessed(&artifacts.evaluated[t]),
        })
        .collect();
    let checks_of: Vec<BlockPreparedCheck<'_, F>> = artifacts
        .stacks
        .iter()
        .map(|s| BlockPreparedCheck {
            group: s.group,
            tables: s.tables.clone(),
            roots: &s.roots,
            layout: &s.layout,
            domain: &s.domain,
        })
        .collect();
    let derived = artifacts.derived_roots();
    let mut transcript = <WlfmHash as WhirHash>::sponge();
    absorb_wlfm_statement(&mut transcript, &artifacts.program_id, claimed);
    // What the tables owe: the `LfmPublic` balance of the claimed words, at the
    // challenges `block_verify_with` is about to draw — the roots block replayed
    // on a fork by the same function, answers and all.
    let mut probe = transcript.clone();
    multilinear_block::absorb_roots_and_answers::<E, _>(
        &mut probe,
        &proof.proof.roots,
        &derived,
        &proof.commit_ood,
    );
    let z: FieldElement<E> = probe.sample_field_element();
    let alpha: FieldElement<E> = probe.sample_field_element();
    let expected = super::proof::expected_public_balance(claimed, &z, &alpha)
        .ok_or("a public word's fingerprint is zero")?;
    multilinear_block::block_verify_with::<F, E, WlfmTranscript, WlfmHash>(
        &proof.proof,
        &proof.commit_ood,
        &proof.argues,
        &proof.prepared,
        &checks_of,
        &statements,
        &layouts,
        &domains,
        &artifacts.sizes,
        &expected,
        &artifacts.config,
        &mut transcript,
        skip_prepared,
        checks,
    )
    .map_err(|e| format!("{e:?}"))
}

// ============================ the AIR set helpers ===========================

/// ⛔ The chip sets a W-LFM program may carry: the WHIR recursion programs'
/// chips with no keccak family, no BLAKE3 family and no `BITWISE`.
///
/// The families carry FIXED preprocessed tables (`KECCAK_RC`, `BITWISE`) and a
/// chip with none at all (`KECCAK_RND`); none of that has a route through the
/// prepared stack here, so a program that instantiates them is refused rather
/// than proved with columns nothing binds.
pub fn check_chip_set(chip_set: ChipSet) -> Result<(), String> {
    if chip_set.keccak || chip_set.blake3 || chip_set.bitwise {
        return Err(format!(
            "a W-LFM program carries the WHIR recursion chip set (no keccak, no BLAKE3, no \
             BITWISE), and this one is {chip_set:?}"
        ));
    }
    Ok(())
}

/// The W-LFM AIR set: the program's chips, unmerged, with NO STARK
/// preprocessed roots.
///
/// ★ The roots are all zero, and that is sound because the multilinear path
/// never reads them: `TableLayout` takes the constraint program, the metadata
/// and the bus interactions, and the preprocessed prefix is bound by the
/// prepared stack instead. ⚠ So this set must never reach a STARK prover or
/// verifier, which is why it is built here and nowhere else. `options` is inert
/// on this path (it only reaches the STARK context of each AIR).
pub fn whir_lfm_airs(
    hasher: HasherKind,
    chip_set: ChipSet,
    hash_chunks: usize,
    options: &ProofOptions,
) -> LfmAirs {
    let zero = [[0u8; 32]; NUM_LFM_CHIPS];
    LfmAirs::new_chunked(&zero, &[], options, 0, hasher, chip_set).with_hash_tail(
        &vec![[0u8; 32]; hash_chunks.max(1)],
        options,
        hasher,
    )
}

/// Each table's preprocessed prefix as the program commits it, in
/// `LfmAirs::air_refs` order for the WHIR recursion chip set: the five ALU
/// groups, `LFM_HASH` (one group, or one per chunk), `LANES`, `HINT`,
/// `PUBLIC` and the fixed `RANGE` group.
///
/// These are the SAME groups the trace builder copies into each chip's leading
/// columns, so the prepared stack and the trace's prefix are one object; the
/// build checks each group's width against its AIR's count.
pub(crate) fn program_prep_groups(program: &LfmProgram) -> Vec<Cow<'_, ColumnGroup>> {
    let g = &program.groups;
    let mut out: Vec<Cow<'_, ColumnGroup>> = vec![
        Cow::Borrowed(&g.const_),
        Cow::Borrowed(&g.balu),
        Cow::Borrowed(&g.xalu),
        Cow::Borrowed(&g.select),
        Cow::Borrowed(&g.bitdec),
    ];
    if program.hash_chunk_count() > 1 {
        out.extend(
            (0..program.hash_chunk_count()).map(|c| Cow::Owned(program.hash_chunk_group(c))),
        );
    } else {
        out.push(Cow::Borrowed(&g.hash));
    }
    out.push(Cow::Borrowed(&g.lanes));
    out.push(Cow::Borrowed(&g.hint));
    out.push(Cow::Borrowed(&g.public));
    out.push(Cow::Owned(range_group()));
    out
}

/// A row-major group as its columns.
pub(crate) fn group_columns(group: &ColumnGroup) -> Vec<Vec<FieldElement<F>>> {
    (0..group.width)
        .map(|column| {
            (0..group.padded_rows)
                .map(|row| group.data[row * group.width + column])
                .collect()
        })
        .collect()
}

/// Every table's `(main width, height in variables)`, in table order: the width
/// is the AIR's, the height the artifacts'.
pub fn table_shapes(airs: &[DynLfmAir<'_>], table_num_vars: &[u8]) -> Vec<(usize, usize)> {
    airs.iter()
        .zip(table_num_vars)
        .map(|(air, &num_vars)| (air.trace_layout().0, num_vars as usize))
        .collect()
}

/// A TEST-ONLY override of every W-LFM config on this thread: grinding off
/// (a laptop CPU's grind is most of a small proof's time) and, where asked, a
/// smaller query count (so a parent over a child is small enough to prove on a
/// laptop). Applied inside [`wlfm_config`], so the build, the prover, the
/// verifier and the emitter of one test all see the same config, and no
/// production build can reach it.
#[cfg(test)]
pub(crate) mod test_format {
    use std::cell::Cell;

    use multilinear::whir_chain::{ChainConfig, GrindBits};

    thread_local! {
        static UNGROUND: Cell<bool> = const { Cell::new(false) };
        static QUERIES: Cell<Option<usize>> = const { Cell::new(None) };
    }

    /// The override, restored when dropped.
    pub(crate) struct Override {
        unground: bool,
        queries: Option<usize>,
    }

    impl Drop for Override {
        fn drop(&mut self) {
            UNGROUND.with(|c| c.set(self.unground));
            QUERIES.with(|c| c.set(self.queries));
        }
    }

    /// Every W-LFM config on this thread ungrinded, and at `queries` queries
    /// when given, until the guard drops.
    pub(crate) fn set(queries: Option<usize>) -> Override {
        Override {
            unground: UNGROUND.with(|c| c.replace(true)),
            queries: QUERIES.with(|c| c.replace(queries)),
        }
    }

    pub(super) fn apply(config: ChainConfig) -> ChainConfig {
        let mut config = config;
        if UNGROUND.with(Cell::get) {
            config.grind = GrindBits::uniform(0);
        }
        if let Some(q) = QUERIES.with(Cell::get) {
            config.num_queries = q;
        }
        config
    }
}
