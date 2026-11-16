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
use multilinear::whir_chain::{ArgueFormat, ChainConfig, FinalPoly};
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
use super::b3l_tables::{B3lAirs, B3lShape, B3lTraces, b3l_prep_groups};
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

/// The hash a [`WlfmFamily::B3`] proof commits, transcripts and grinds with:
/// standard 7-round BLAKE3 ([`multilinear::whir_hash::Blake3Whir`]), the
/// BLAKE3 base's, so a BLAKE3 base's recursion is one hash end to end.
pub type WlfmB3Hash = multilinear::whir_hash::Blake3Whir;

/// The hash family of a W-LFM proof: a verifier constant, named by both sides
/// and folded into the identity and the statement's tag, never read from a
/// proof.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WlfmFamily {
    /// [`WlfmHash`]: Poseidon1 at width 16, the block family's. Today's.
    #[default]
    P1,
    /// [`WlfmB3Hash`]: standard 7-round BLAKE3.
    B3,
}

impl WlfmFamily {
    /// The statement's domain tag under this family.
    pub const fn statement_tag(self) -> &'static [u8] {
        match self {
            WlfmFamily::P1 => WLFM_STATEMENT_TAG,
            WlfmFamily::B3 => WLFM_B3_STATEMENT_TAG,
        }
    }

    /// The commitment hash's name, as the identity folds it.
    pub fn hash_name(self) -> &'static str {
        match self {
            WlfmFamily::P1 => <WlfmHash as WhirHash>::NAME,
            WlfmFamily::B3 => <WlfmB3Hash as WhirHash>::NAME,
        }
    }
}

/// `$body` with `$h` the commitment hash of `$family`.
macro_rules! with_wlfm_hash {
    ($family:expr, |$h:ident| $body:expr) => {
        match $family {
            WlfmFamily::P1 => {
                #[allow(non_camel_case_types)]
                type $h = WlfmHash;
                $body
            }
            WlfmFamily::B3 => {
                #[allow(non_camel_case_types)]
                type $h = WlfmB3Hash;
                $body
            }
        }
    };
}

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

/// The statement's domain tag under [`WlfmFamily::B3`].
pub const WLFM_B3_STATEMENT_TAG: &[u8] = b"LAMBDAVM_LFM_WHIR_BLOCK_STATEMENT_V1/B3W";

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
    /// Blowup 64 at the production query rule, under a 2^[`WLFM_B64_STACK`]
    /// stack: the second compression step's rate (a proof whose verifier is
    /// the host), where a query buys ≈ 2.96 Johnson bits.
    SixtyFour,
}

impl WlfmRate {
    /// The code's inverse rate.
    pub const fn factor(self) -> usize {
        match self {
            WlfmRate::Two => 2,
            WlfmRate::Four => 4,
            WlfmRate::Sixteen => 16,
            WlfmRate::SixtyFour => 64,
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
    /// proof whose verifier is the host (the last proof of the pipeline).
    Evaluated,
}

/// The format levers of a W-LFM proof beyond its rate: verifier constants,
/// named by both sides and folded into the identity. [`WlfmLevers::DEFAULT`]
/// is today's format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WlfmLevers {
    /// How the instruction columns are bound.
    pub prep: WlfmPrep,
    /// Where the chains stop committing.
    pub final_poly: FinalPoly,
    /// The fold width of every chain round after the first; `None` is
    /// today's (`log_folding` of the pinned chain).
    pub fold: Option<u8>,
    /// The query grind's bits; `None` is the format's
    /// ([`crate::zf_format::PRODUCTION_WHIR_GRIND_BITS`]). The query count
    /// follows from it by the production rule, so more bits run fewer queries.
    pub grind: Option<u8>,
    /// The hash family.
    pub family: WlfmFamily,
}

impl WlfmLevers {
    /// Today's format: the stacked binding, no final polynomial.
    pub const DEFAULT: Self = Self {
        prep: WlfmPrep::Stacked,
        final_poly: FinalPoly::Off,
        fold: None,
        grind: None,
        family: WlfmFamily::P1,
    };

    /// Today's format with the instruction columns bound per `prep`.
    pub const fn with_prep(prep: WlfmPrep) -> Self {
        Self {
            prep,
            final_poly: FinalPoly::Off,
            fold: None,
            grind: None,
            family: WlfmFamily::P1,
        }
    }

    /// Today's format under `family`.
    pub const fn with_family(family: WlfmFamily) -> Self {
        Self {
            family,
            ..Self::DEFAULT
        }
    }
}

/// The rows an LFM chip's table may hold in a W-LFM proof: 2^22. A chip
/// whose padded height is above it (a recursion level's extension ALU, say)
/// is cut into tables of 2^22 rows and a last one padded to its own power of
/// two ([`chip_chunk_rows`]), each a table of the block, so no table outgrows a
/// group and the padding is at most one chunk's. A format constant: the
/// identity folds the cut, and no program today reaches it.
pub const WLFM_CHUNK_LOG: u32 = 22;

/// The padded rows of each table an LFM chip of `real_rows` rows (padded to
/// `padded_rows` as one table) takes under a `2^chunk_log` cap: the one table
/// as it is when it fits, else full chunks and the remainder's own power of
/// two. Every LFM chip's constraints read one row and its bus tokens are one
/// row's, so which table a row sits in is invisible to the proof's claims.
pub fn chip_chunk_rows(real_rows: usize, padded_rows: usize, chunk_log: u32) -> Vec<usize> {
    let cap = 1usize << chunk_log;
    if padded_rows <= cap {
        return vec![padded_rows];
    }
    let mut out = vec![cap; real_rows / cap];
    let rem = real_rows % cap;
    if rem > 0 {
        out.push(super::layout::padded_rows(rem));
    }
    out
}

/// The chunk cap a W-LFM build cuts its chips at: [`WLFM_CHUNK_LOG`], or a
/// test's override.
fn chunk_log() -> u32 {
    #[cfg(test)]
    if let Some(log) = test_format::chunk_log() {
        return log;
    }
    WLFM_CHUNK_LOG
}

/// `LAMBDA_VM_WLFM_NARROW_CHIPS=1`: the LFM chips' traces packed narrow
/// ([`multilinear::narrow::NarrowColumns`]) before the prove takes them, so
/// their columns go up at the bytes their words need and the card widens
/// them, as the lean tables' do. Off by default; a prover's choice, never the
/// proof's: the card widens the same words. Read once a process.
pub const NARROW_CHIPS_ENV: &str = "LAMBDA_VM_WLFM_NARROW_CHIPS";

/// Whether the prove packs the LFM chips narrow: [`NARROW_CHIPS_ENV`], or a
/// test's override.
pub fn narrow_chips() -> bool {
    #[cfg(test)]
    if let Some(on) = test_format::narrow_chips() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var(NARROW_CHIPS_ENV).is_ok_and(|v| v.trim() == "1"))
}

/// The bytes a word needs: 1, 2, 4 or 8 (`NarrowColumns`' rule).
fn narrow_width(max: u64) -> u8 {
    match max {
        0..=0xff => 1,
        0x100..=0xffff => 2,
        0x1_0000..=0xffff_ffff => 4,
        _ => 8,
    }
}

/// Writes `words` into `out` at `w` bytes each, little-endian.
fn put_narrow(out: &mut [u8], w: usize, words: impl Iterator<Item = u64>) {
    match w {
        1 => out.iter_mut().zip(words).for_each(|(o, v)| *o = v as u8),
        2 => out
            .chunks_exact_mut(2)
            .zip(words)
            .for_each(|(o, v)| o.copy_from_slice(&(v as u16).to_le_bytes())),
        4 => out
            .chunks_exact_mut(4)
            .zip(words)
            .for_each(|(o, v)| o.copy_from_slice(&(v as u32).to_le_bytes())),
        _ => out
            .chunks_exact_mut(8)
            .zip(words)
            .for_each(|(o, v)| o.copy_from_slice(&v.to_le_bytes())),
    }
}

/// A `width`-wide row-major trace's first `data.len() / width` rows as `rows`
/// rows of narrow columns (the rows past the data zero): each column at the
/// bytes its largest canonical word needs, row blocks in parallel, each block
/// reading its rows once and writing its share of every column.
pub(crate) fn pack_rows_narrow(
    data: &[FieldElement<F>],
    width: usize,
    rows: usize,
) -> Result<multilinear::narrow::NarrowColumns, String> {
    use rayon::prelude::*;
    const BLOCK: usize = 1 << 12;
    if width == 0 || !data.len().is_multiple_of(width) || data.len() / width > rows {
        return Err(format!(
            "{} cells are not whole rows of {width} within {rows}",
            data.len()
        ));
    }
    let word = |f: &FieldElement<F>| GoldilocksField::canonical(f.value());
    let max = data
        .par_chunks(BLOCK * width)
        .map(|block| {
            let mut m = vec![0u64; width];
            for row in block.chunks_exact(width) {
                for (m, f) in m.iter_mut().zip(row) {
                    *m = (*m).max(word(f));
                }
            }
            m
        })
        .reduce(
            || vec![0u64; width],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x = (*x).max(y));
                a
            },
        );
    let widths: Vec<u8> = max.into_iter().map(narrow_width).collect();
    let total: usize = widths.iter().map(|&w| rows * w as usize).sum();
    let mut bytes = multilinear::pinned::with_pinned(total, || vec![0u8; total]);
    let mut by_block: Vec<Vec<&mut [u8]>> = (0..rows.div_ceil(BLOCK))
        .map(|_| Vec::with_capacity(width))
        .collect();
    let mut rest = bytes.as_mut_slice();
    for &w in &widths {
        let (column, tail) = rest.split_at_mut(rows * w as usize);
        rest = tail;
        for (b, part) in column.chunks_mut(BLOCK * w as usize).enumerate() {
            by_block[b].push(part);
        }
    }
    let real = data.len() / width;
    by_block.into_par_iter().enumerate().for_each(|(b, parts)| {
        let first = (b * BLOCK).min(real);
        let block = &data[first * width..((b + 1) * BLOCK).min(real) * width];
        for (c, (out, &w)) in parts.into_iter().zip(&widths).enumerate() {
            put_narrow(
                out,
                w as usize,
                block.iter().skip(c).step_by(width).map(word),
            );
        }
    });
    multilinear::narrow::NarrowColumns::from_parts(rows, widths, bytes)
        .ok_or_else(|| "the narrow parts do not fit".to_string())
}

/// The stack cap at [`WlfmRate::Sixteen`]: a grouped chain lives on
/// `H_{n + m + lb}`, and Goldilocks' two-adicity (32) leaves `32 − 4 − 2 = 26`
/// stacked variables at blowup 16 and the widest index; the default 27 would
/// need 33.
pub const WLFM_B16_STACK: usize = 26;

/// The stack cap at [`WlfmRate::SixtyFour`]: `32 − 6 − 2 = 24` stacked
/// variables at blowup 64 and the widest index.
pub const WLFM_B64_STACK: usize = 24;

/// ★ The chain config a W-LFM program proves under, over its table shapes:
/// the pinned block format's chain
/// ([`crate::block_whir::BlockFormat::chain_config`]: its argue, group fold,
/// round-0 leaf order and trees) at [`ZfFormat::DEFAULT`] with `rate`'s blowup
/// (and, at blowup 16, the [`WLFM_B16_STACK`] stack). The ONE call the build,
/// the prover and the verifier make.
pub fn wlfm_config(rate: WlfmRate, shapes: &[(usize, usize)]) -> ChainConfig {
    wlfm_config_under(rate, FinalPoly::Off, None, None, shapes)
}

/// [`wlfm_config`] with the chains' [`FinalPoly`] and, when `fold` is set,
/// the fold width of every round after the first: a proof whose verifier is
/// the host may stop committing at a final polynomial and fold on a
/// byte-minimising schedule (the in-guest chain verifier has neither, so a
/// proof a parent verifies keeps both at today's). The query count stays the
/// one today's schedule derives: a wider fold has no more rounds, so the
/// count only over-covers its union bound. `grind`, when set, is the query
/// grind's bits, and the query count is the production rule's at those bits.
pub fn wlfm_config_under(
    rate: WlfmRate,
    final_poly: FinalPoly,
    fold: Option<u8>,
    grind: Option<u8>,
    shapes: &[(usize, usize)],
) -> ChainConfig {
    let mut zf = ZfFormat {
        whir_blowup: rate.factor() as u8,
        ..ZfFormat::DEFAULT
    };
    match rate {
        WlfmRate::Sixteen => {
            zf.whir_stack = multilinear::whir_chain::StackVars::new(WLFM_B16_STACK)
                .expect("the blowup-16 stack is in range");
        }
        WlfmRate::SixtyFour => {
            zf.whir_stack = multilinear::whir_chain::StackVars::new(WLFM_B64_STACK)
                .expect("the blowup-64 stack is in range");
        }
        WlfmRate::Two | WlfmRate::Four => {}
    }
    if let Some(bits) = grind {
        zf.whir_grind_bits = bits;
    }
    let format = crate::block_whir::BlockFormat {
        zf,
        ..crate::block_whir::BlockFormat::pinned()
    };
    let mut config = format.chain_config(shapes);
    if rate == WlfmRate::Four {
        config.num_queries = WLFM_B4_QUERIES;
    }
    config.format.final_poly = final_poly;
    if let Some(k) = fold {
        config.log_folding = usize::from(k);
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
    /// Where the chains stop committing and how wide they fold after round 0
    /// (folded into `config`; held here so the verifier re-derives the config
    /// from the levers, never the reverse).
    pub final_poly: FinalPoly,
    pub fold: Option<u8>,
    /// The query grind's bits when not the format's (folded into `config`).
    pub grind: Option<u8>,
    /// The hash family the proof commits, transcripts and grinds with.
    pub family: WlfmFamily,
    /// The padded rows of each table every LFM chip takes, in
    /// `LfmAirs::air_refs` order ([`chip_chunk_rows`]): one entry a chip
    /// unless it is cut.
    pub lfm_chunks: Vec<Vec<usize>>,
    /// The lean BLAKE3 tables' chunks ([`super::b3l_tables`]), after the LFM
    /// chips' tables: empty for a program with no `B3l` row, which then has
    /// none of them.
    pub b3l: B3lShape,
    /// Under [`WlfmPrep::Stacked`]: the prepared stacks. Empty otherwise.
    pub stacks: Vec<WlfmStack>,
    /// Under [`WlfmPrep::Evaluated`]: each table's preprocessed prefix, as the
    /// verifier evaluates it (empty for a table with none). Empty otherwise.
    pub evaluated: Vec<Vec<Mle<F>>>,
    pub program_id: Commitment,
}

impl WlfmArtifacts {
    /// The program's AIR set, with no STARK roots ([`whir_lfm_airs`]), and
    /// its lean BLAKE3 tables after them.
    pub fn airs(&self) -> WlfmAirs {
        WlfmAirs {
            lfm: whir_lfm_airs(self.hasher, self.chip_set, self.hash_chunks, &air_options()),
            lfm_chunks: self.lfm_chunks.clone(),
            b3l: B3lAirs::new(&self.b3l, &air_options()),
        }
    }

    /// `(main width, height)` per table.
    pub fn shapes(&self, airs: &WlfmAirs) -> Vec<(usize, usize)> {
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

/// A W-LFM program's AIR set: the LFM chips' ([`whir_lfm_airs`]), then the
/// lean BLAKE3 tables' ([`B3lAirs`]) — the proof's table order.
pub struct WlfmAirs {
    pub lfm: LfmAirs,
    /// Each LFM chip's tables' padded rows ([`WlfmArtifacts::lfm_chunks`]).
    pub lfm_chunks: Vec<Vec<usize>>,
    pub b3l: B3lAirs,
}

impl WlfmAirs {
    /// Every AIR in table order: an LFM chip's once a table it takes, then the
    /// lean tables'.
    pub fn air_refs(&self) -> Vec<DynLfmAir<'_>> {
        let mut refs: Vec<DynLfmAir<'_>> = Vec::new();
        for (i, air) in self.lfm.air_refs().into_iter().enumerate() {
            let n = self.lfm_chunks.get(i).map_or(1, Vec::len);
            refs.extend(std::iter::repeat_n(air, n));
        }
        refs.extend(self.b3l.refs());
        refs
    }

    /// Every AIR with its trace, in table order. `b3l` is the program's lean
    /// tables: present exactly when the AIR set has them.
    pub fn air_trace_pairs<'a>(
        &'a self,
        traces: &'a mut super::trace::LfmTraces,
        b3l: Option<&'a mut B3lTraces>,
    ) -> Result<Vec<(DynLfmAir<'a>, TableIn<'a>)>, String> {
        let mut pairs: Vec<(DynLfmAir<'a>, TableIn<'a>)> = Vec::new();
        let narrow = narrow_chips();
        for (i, (air, trace, _)) in self.lfm.air_trace_pairs(traces).into_iter().enumerate() {
            match self.lfm_chunks.get(i) {
                // Packed narrow by the prove, straight from the rows.
                Some(chunks) if chunks.len() > 1 && narrow => {
                    let trace: &'a stark::trace::TraceTable<F, E> = trace;
                    let (width, height) = (trace.main_table.width, trace.main_table.height);
                    let data = trace.main_table.row_major_data();
                    let mut start = 0usize;
                    for &rows in chunks {
                        let end = (start + rows).min(height).max(start.min(height));
                        let cells = &data[start.min(height) * width..end * width];
                        pairs.push((air, TableIn::Rows { cells, width, rows }));
                        start += rows;
                    }
                }
                Some(chunks) if chunks.len() > 1 => {
                    // A cut chip: each chunk its own table, transposed straight
                    // out of the row-major trace into its columns (one pass:
                    // no row-major copy to transpose again).
                    let width = trace.main_table.width;
                    let height = trace.main_table.height;
                    let data = trace.main_table.row_major_data();
                    let mut start = 0usize;
                    for &rows in chunks {
                        pairs.push((
                            air,
                            TableIn::Columns(chunk_columns(data, width, height, start, rows)),
                        ));
                        start += rows;
                    }
                }
                _ => pairs.push((air, TableIn::Wide(trace))),
            }
        }
        let lean = self.b3l.refs().len();
        match b3l {
            Some(t) => pairs.extend(
                self.b3l
                    .pairs(t)?
                    .into_iter()
                    .map(|(air, held)| (air, TableIn::Held(held))),
            ),
            None if lean == 0 => {}
            None => return Err("the AIR set has lean BLAKE3 tables and the fill none".into()),
        }
        Ok(pairs)
    }
}

/// Rows `[start, start + rows)` of a `width`-wide row-major trace of `height`
/// rows as columns, the rows past `height` zero: row blocks in parallel, each
/// reading its rows once and writing its share of every column.
fn chunk_columns(
    data: &[FieldElement<F>],
    width: usize,
    height: usize,
    start: usize,
    rows: usize,
) -> Vec<Vec<FieldElement<F>>> {
    use rayon::prelude::*;
    const BLOCK: usize = 1 << 12;
    let mut cols: Vec<Vec<FieldElement<F>>> = (0..width)
        .map(|_| crate::tables::types::zeroed_fe_vec(rows))
        .collect();
    let real = height.saturating_sub(start).min(rows);
    let mut by_block: Vec<Vec<&mut [FieldElement<F>]>> = (0..real.div_ceil(BLOCK))
        .map(|_| Vec::with_capacity(width))
        .collect();
    for col in cols.iter_mut() {
        for (b, segment) in col[..real].chunks_mut(BLOCK).enumerate() {
            by_block[b].push(segment);
        }
    }
    by_block
        .into_par_iter()
        .enumerate()
        .for_each(|(b, mut segments)| {
            let first = start + b * BLOCK;
            let len = segments.first().map_or(0, |s| s.len());
            for (i, row) in data[first * width..(first + len) * width]
                .chunks_exact(width)
                .enumerate()
            {
                for (segment, v) in segments.iter_mut().zip(row) {
                    segment[i] = *v;
                }
            }
        });
    cols
}

/// One table's columns as the W-LFM prover takes them: an LFM chip's trace, or
/// a lean table as its fill held it (wide or narrow).
pub enum TableIn<'a> {
    Wide(&'a mut stark::trace::TraceTable<F, E>),
    /// A chunk of a cut chip's trace, its columns.
    Columns(Vec<Vec<FieldElement<F>>>),
    Held(&'a mut super::b3l_tables::HeldTable),
    /// Under [`narrow_chips`]: a chunk of a cut chip, its first
    /// `cells.len() / width` rows of `rows` (the rest zero), packed narrow by
    /// the prove.
    Rows {
        cells: &'a [FieldElement<F>],
        width: usize,
        rows: usize,
    },
}

impl TableIn<'_> {
    /// Whether the table is held as its records (the card fills its columns).
    fn is_records(&self) -> bool {
        matches!(self, TableIn::Held(h) if h.is_records())
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            TableIn::Wide(t) => (t.main_table.width, t.main_table.height),
            TableIn::Columns(cols) => (cols.len(), cols.first().map_or(0, Vec::len)),
            TableIn::Held(h) => (h.cols(), h.rows()),
            TableIn::Rows { width, rows, .. } => (*width, *rows),
        }
    }

    /// The cell at `(r, c)`, read where it lies.
    fn get(&self, r: usize, c: usize) -> FieldElement<F> {
        match self {
            TableIn::Wide(t) => *t.main_table.get(r, c),
            TableIn::Columns(cols) => cols[c][r],
            TableIn::Held(h) => h.get(r, c),
            TableIn::Rows { cells, width, .. } => cells
                .get(r * width + c)
                .copied()
                .unwrap_or_else(FieldElement::zero),
        }
    }

    /// Whether the table's first `pinned.len()` columns are `pinned`, cell
    /// for cell: rows in parallel, each row's cells read where they lie (no
    /// column gathered, none widened).
    fn prefix_is(&self, pinned: &[&[FieldElement<F>]]) -> bool {
        use rayon::prelude::*;
        let rows = self.shape().1;
        if pinned.iter().any(|col| col.len() != rows) {
            return false;
        }
        (0..rows).into_par_iter().with_min_len(1 << 14).all(|r| {
            pinned
                .iter()
                .enumerate()
                .all(|(c, col)| self.get(r, c) == col[r])
        })
    }
}

/// How the W-LFM prover holds and streams a proof's tables: a prover's
/// choice, never the proof's (the same bytes either way).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WlfmProving {
    /// Every table up wide, every tree kept, the groups proved on their forks
    /// in order: today's, for the small programs C1 and C2 are.
    Held,
    /// The base's production streaming: the block options' tree drop, card
    /// narrowing and upload ahead, narrow lean tables as their fill held them,
    /// and phase B in group order under the pinned overlap. For a program of
    /// many groups (a recursion level).
    Streamed,
}

/// A program's artifacts and the prover's half of its prepared stacks.
pub struct WlfmBuild {
    pub artifacts: WlfmArtifacts,
    stacks: WlfmStacks,
}

/// The prover's prepared stacks, under the build's family.
enum WlfmStacks {
    P1(Vec<GroupPrepared<WlfmHash>>),
    B3(Vec<GroupPrepared<WlfmB3Hash>>),
}

impl WlfmStacks {
    fn empty(family: WlfmFamily) -> Self {
        match family {
            WlfmFamily::P1 => WlfmStacks::P1(Vec::new()),
            WlfmFamily::B3 => WlfmStacks::B3(Vec::new()),
        }
    }

    #[cfg(test)]
    fn clear(&mut self) {
        match self {
            WlfmStacks::P1(s) => s.clear(),
            WlfmStacks::B3(s) => s.clear(),
        }
    }
}

/// Every group's prepared commitment under `H`, on the host when `on_host`
/// (a verifier's build), else where the card is.
fn commit_under<H: WhirHash>(
    stacks: Vec<crate::block_whir::GroupStack>,
    config: &ChainConfig,
    on_host: bool,
) -> Result<Vec<GroupPrepared<H>>, String> {
    if on_host {
        commit_groups_on_host::<H>(stacks, config).map_err(|e| format!("{e:?}"))
    } else {
        // ⛔ THE CARD, as `program_census::build_artifacts_counted` holds it
        // for a STARK build: the prepared commits run on the device, so
        // concurrent siblings serialize them. Inert unless a driver armed it.
        let _card = super::device_permit::hold_labeled("build_artifacts");
        commit_groups::<H>(stacks, config).map_err(|e| format!("{e:?}"))
    }
}

/// What a build keeps of each prepared commitment, under any hash.
fn stack_of<H: WhirHash>(p: &GroupPrepared<H>) -> WlfmStack {
    WlfmStack {
        group: p.group,
        tables: p.tables.clone(),
        roots: p.roots.clone(),
        layout: p.commitment.layout().clone(),
        domain: p.commitment.domain().clone(),
    }
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
    build_wlfm_under(program, hasher, rate, WlfmLevers::DEFAULT, false)
}

/// [`build_wlfm`] with every prepared commit on the host: the VERIFIER's
/// build, so no card kernel is on its acceptance path (the block verifier's
/// rule since the joint round-0 trees, `block_whir::commit_groups_on_host`).
pub fn build_wlfm_on_host(
    program: &LfmProgram,
    hasher: HasherKind,
    rate: WlfmRate,
) -> Result<WlfmBuild, String> {
    build_wlfm_under(program, hasher, rate, WlfmLevers::DEFAULT, true)
}

/// [`build_wlfm`] at `rate` under `levers` (the instruction columns' binding
/// and the chains' final polynomial), the prepared commits (under
/// [`WlfmPrep::Stacked`]) on the host when `on_host` (a verifier's build).
pub fn build_wlfm_under(
    program: &LfmProgram,
    hasher: HasherKind,
    rate: WlfmRate,
    levers: WlfmLevers,
    on_host: bool,
) -> Result<WlfmBuild, String> {
    let WlfmLevers {
        prep,
        final_poly,
        fold,
        grind,
        family,
    } = levers;
    // The mask with `BITWISE` dropped where nothing sends it: the default, as a
    // constant (`for_program_with_hasher` reads it from the environment).
    let chip_set = ChipSet::for_program_under(program, hasher, true);
    check_chip_set(chip_set)?;
    let hash_chunks = program.hash_chunk_count();
    let b3l = B3lShape::of_program(program);
    let chip_groups = program_prep_groups(program);
    let lfm = whir_lfm_airs(hasher, chip_set, hash_chunks, &air_options());
    let chip_count = lfm.air_refs().len();
    if chip_count != chip_groups.len() {
        return Err(format!(
            "the AIR set has {chip_count} LFM tables and the program commits {} instruction \
             groups",
            chip_groups.len()
        ));
    }
    // Each chip cut at the chunk cap: its groups' rows sliced as its tables'.
    let cut = chunk_log();
    let mut lfm_chunks: Vec<Vec<usize>> = Vec::with_capacity(chip_groups.len());
    let mut groups: Vec<Cow<'_, ColumnGroup>> = Vec::with_capacity(chip_groups.len());
    for group in chip_groups {
        let chunks = chip_chunk_rows(group.real_rows, group.padded_rows, cut);
        if chunks.len() == 1 {
            groups.push(group);
        } else {
            let mut start = 0usize;
            for &rows in &chunks {
                let end = (start + rows).min(group.real_rows.max(start));
                let mut data = group.data[start * group.width..end * group.width].to_vec();
                data.resize(rows * group.width, FieldElement::zero());
                groups.push(Cow::Owned(ColumnGroup {
                    width: group.width,
                    real_rows: end - start,
                    padded_rows: rows,
                    data,
                }));
                start += rows;
            }
        }
        lfm_chunks.push(chunks);
    }
    groups.extend(b3l_prep_groups(program).into_iter().map(Cow::Owned));
    let airs = WlfmAirs {
        lfm,
        lfm_chunks: lfm_chunks.clone(),
        b3l: B3lAirs::new(&b3l, &air_options()),
    };
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
    let config = wlfm_config_under(rate, final_poly, fold, grind, &shapes);
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
            final_poly,
            fold,
            grind,
            family,
            lfm_chunks,
            b3l,
            stacks: Vec::new(),
            evaluated,
            program_id: [0u8; 32],
        };
        artifacts.program_id = wlfm_program_id(&artifacts);
        return Ok(WlfmBuild {
            artifacts,
            stacks: WlfmStacks::empty(family),
        });
    }
    let stacks = group_stacks(columns, &group_lists).map_err(|e| format!("{e:?}"))?;
    let committed = match family {
        WlfmFamily::P1 => WlfmStacks::P1(commit_under::<WlfmHash>(stacks, &config, on_host)?),
        WlfmFamily::B3 => WlfmStacks::B3(commit_under::<WlfmB3Hash>(stacks, &config, on_host)?),
    };
    let stacks: Vec<WlfmStack> = match &committed {
        WlfmStacks::P1(c) => c.iter().map(stack_of).collect(),
        WlfmStacks::B3(c) => c.iter().map(stack_of).collect(),
    };
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
        final_poly,
        fold,
        grind,
        family,
        lfm_chunks,
        b3l,
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
    h.update(a.family.hash_name().as_bytes());
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
    // The chips' cut, when any chip is cut: each chip's table count (their
    // heights are in the table list above). A program with no cut chip folds
    // nothing here.
    if a.lfm_chunks.iter().any(|c| c.len() > 1) {
        h.update(b"CHUNKS");
        for chunks in &a.lfm_chunks {
            h.update((chunks.len() as u64).to_le_bytes());
        }
    }
    // The lean BLAKE3 tables, when the program has any: their round count and
    // their chunks (their heights are in the table list above). A program
    // with none folds nothing here, so its identity is unchanged.
    if !a.b3l.is_empty() {
        h.update(b"B3L");
        h.update((super::b3l::ROUNDS as u64).to_le_bytes());
        h.update((a.b3l.rounds.len() as u64).to_le_bytes());
        h.update((a.b3l.heads.len() as u64).to_le_bytes());
        // The one-output selects' chunks, only when a program has them.
        if !a.b3l.muxes.is_empty() {
            h.update(b"MUX");
            h.update((a.b3l.muxes.len() as u64).to_le_bytes());
        }
        // The cap tables' and reads' chunks, only when a program has them.
        if !a.b3l.caps.is_empty() || !a.b3l.cap_reads.is_empty() {
            h.update(b"CAP");
            h.update((a.b3l.caps.len() as u64).to_le_bytes());
            h.update((a.b3l.cap_reads.len() as u64).to_le_bytes());
        }
    }
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
    statement_head_for(WlfmFamily::P1, program_id, num_words)
}

/// [`statement_head`] under `family`: its tag first.
pub fn statement_head_for(
    family: WlfmFamily,
    program_id: &Commitment,
    num_words: usize,
) -> [Vec<u8>; 3] {
    let mut counts = Vec::with_capacity(16);
    counts.extend_from_slice(&u64::from(LFM_MACHINE_VERSION).to_le_bytes());
    counts.extend_from_slice(&(num_words as u64).to_le_bytes());
    [family.statement_tag().to_vec(), program_id.to_vec(), counts]
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
    absorb_wlfm_statement_for(WlfmFamily::P1, transcript, program_id, public_words)
}

/// [`absorb_wlfm_statement`] under `family`'s tag.
pub fn absorb_wlfm_statement_for(
    family: WlfmFamily,
    transcript: &mut impl IsTranscript<E>,
    program_id: &Commitment,
    public_words: &[(u32, LfmWord)],
) {
    for call in statement_head_for(family, program_id, public_words.len()) {
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
    prove_wlfm_traces_as(build, filled, check_prefix, WlfmProving::Held)
}

/// [`prove_wlfm_traces`] held and streamed as `proving` says.
pub(crate) fn prove_wlfm_traces_as(
    build: &WlfmBuild,
    filled: LfmFilled,
    check_prefix: bool,
    proving: WlfmProving,
) -> Result<WlfmProof, String> {
    match &build.stacks {
        WlfmStacks::P1(stacks) => {
            prove_traces_under::<WlfmHash>(&build.artifacts, stacks, filled, check_prefix, proving)
        }
        WlfmStacks::B3(stacks) => prove_traces_under::<WlfmB3Hash>(
            &build.artifacts,
            stacks,
            filled,
            check_prefix,
            proving,
        ),
    }
}

/// [`prove_wlfm_traces_as`] under the build's hash `H`.
/// What one W-LFM prove spent, by phase and by group: the host's table
/// preparation, phase A (every group committed) and phase B (argued and
/// opened), each group's stamps, and the card ledger's budget and its promise
/// when the prove took the card.
#[derive(Clone, Debug, Default)]
pub struct WlfmStamps {
    pub prep: f64,
    /// `prep` by part: the merge split back, the AIR set and its shapes, the
    /// trace pairs, the prefix check, the layouts, the committed tables.
    pub prep_parts: [f64; 6],
    /// Under [`narrow_chips`], the chips packed narrow before the prove takes
    /// them (left out of `prep`): seconds, tables, cells, packed bytes.
    pub narrow_pack: f64,
    pub narrow_tables: usize,
    pub narrow_cells: usize,
    pub narrow_bytes: usize,
    pub phase_a: f64,
    pub phase_b: f64,
    pub groups: Vec<multilinear_block::GroupStamps>,
    pub card_budget: u64,
    pub card_reserved_at_start: u64,
}

thread_local! {
    static LAST_WLFM_STAMPS: std::cell::RefCell<Option<WlfmStamps>> =
        const { std::cell::RefCell::new(None) };
}

/// Take the most recent W-LFM prove's stamps ON THIS THREAD, clearing them;
/// `None` when nothing proved here since the last take.
pub fn take_wlfm_stamps() -> Option<WlfmStamps> {
    LAST_WLFM_STAMPS.with(|c| c.borrow_mut().take())
}

fn prove_traces_under<H: WhirHash>(
    artifacts: &WlfmArtifacts,
    stacks: &[GroupPrepared<H>],
    filled: LfmFilled,
    check_prefix: bool,
    proving: WlfmProving,
) -> Result<WlfmProof, String> {
    if artifacts.family.hash_name() != H::NAME {
        return Err(format!(
            "a {:?} build's stacks under {}",
            artifacts.family,
            H::NAME
        ));
    }
    let mut filled = filled;
    let t_prep = std::time::Instant::now();
    let mut parts = [0f64; 6];
    let mut b3l = filled.take_b3l();
    let (mut traces, public_words, hasher, execute_secs, fill_secs, exec_split) =
        filled.into_parts();
    // Every chip its own table: a set filled under the process's merge is
    // split back (a copy; [`execute_and_fill_unmerged`] fills it so).
    traces.merge_to(&Vec::new())?;
    parts[0] = t_prep.elapsed().as_secs_f64();
    if hasher != artifacts.hasher {
        return Err(format!(
            "the traces were filled for {hasher:?} and the artifacts built for {:?}",
            artifacts.hasher
        ));
    }
    let t = std::time::Instant::now();
    let airs = artifacts.airs();
    let shapes = artifacts.shapes(&airs);
    parts[1] = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let pairs = airs.air_trace_pairs(&mut traces, b3l.as_mut())?;
    parts[2] = t.elapsed().as_secs_f64();
    if pairs.len() != shapes.len() {
        return Err(format!(
            "{} traces for {} tables",
            pairs.len(),
            shapes.len()
        ));
    }
    // Where each table's prefix sits among the prover's stacks' columns.
    let mut prefix_of: Vec<Option<(usize, usize, usize)>> = vec![None; shapes.len()];
    for (s, stack) in stacks.iter().enumerate() {
        let mut at = 0usize;
        for &(table, n) in &stack.tables {
            prefix_of[table] = Some((s, at, n));
            at += n;
        }
    }
    let mut tables: Vec<CommittedTable<'_, F, E>> = Vec::with_capacity(pairs.len());
    // The chips packed narrow: seconds, tables, cells, packed bytes.
    let mut narrow = (0f64, 0usize, 0usize, 0usize);
    for (t, ((air, table), &(width, num_vars))) in pairs.into_iter().zip(&shapes).enumerate() {
        let (cols, rows) = table.shape();
        if cols != width || rows != 1usize << num_vars {
            return Err(format!(
                "{}: the trace is {cols} columns of {rows} rows, the program's {width} of \
                 2^{num_vars}",
                air.name()
            ));
        }
        // A table held as its records writes its prefix from them (a lean
        // round chunk's MU, TAG, RND, from each record's tag and the row): the
        // check would fill it on the host, which the records exist to avoid,
        // so the prepared opening is what binds it. (The host-against-card
        // fill test and the proofs' byte equality check the fill.)
        let t_check = std::time::Instant::now();
        if check_prefix
            && !table.is_records()
            && let Some((s, at, n)) = prefix_of[t]
        {
            let pinned: Vec<&[FieldElement<F>]> = stacks[s].columns[at..at + n]
                .iter()
                .map(|col| col.evals())
                .collect();
            if !table.prefix_is(&pinned) {
                return Err(format!(
                    "{}: a preprocessed column is not the program's",
                    air.name()
                ));
            }
        }
        if check_prefix && artifacts.prep == WlfmPrep::Evaluated {
            let pinned: Vec<&[FieldElement<F>]> = artifacts.evaluated[t]
                .iter()
                .map(|col| col.evals())
                .collect();
            if !table.prefix_is(&pinned) {
                return Err(format!(
                    "{}: a preprocessed column is not the program's",
                    air.name()
                ));
            }
        }
        parts[3] += t_check.elapsed().as_secs_f64();
        let t_layout = std::time::Instant::now();
        let layout = layout_of(air, width, num_vars).map_err(|e| format!("{e:?}"))?;
        parts[4] += t_layout.elapsed().as_secs_f64();
        let t_table = std::time::Instant::now();
        // Under `narrow_chips`, the chips packed narrow: stamped apart from
        // the preparation, as the fill's own work.
        let mut pack_narrow = |cells: &[FieldElement<F>], width: usize, rows: usize| {
            let t = std::time::Instant::now();
            let packed = pack_rows_narrow(cells, width, rows);
            narrow.0 += t.elapsed().as_secs_f64();
            if let Ok(p) = &packed {
                narrow.1 += 1;
                narrow.2 += p.rows() * p.cols();
                narrow.3 += p.data().len();
            }
            packed.map_err(|e| format!("{}: {e}", air.name()))
        };
        let committed = match table {
            TableIn::Wide(trace) if narrow_chips() => CommittedTable::from_narrow(
                layout,
                pack_narrow(
                    trace.main_table.row_major_data(),
                    trace.main_table.width,
                    trace.main_table.height,
                )?,
            ),
            TableIn::Rows { cells, width, rows } => {
                CommittedTable::from_narrow(layout, pack_narrow(cells, width, rows)?)
            }
            // The row-major traces transposed by tiles (`columns_blocked`):
            // each row's cache lines read once, not once a column.
            TableIn::Wide(trace) => {
                let mut columns = trace.main_table.columns_blocked();
                CommittedTable::from_layout(layout, |col| {
                    core::mem::take(&mut columns[col as usize])
                })
            }
            TableIn::Columns(mut columns) => CommittedTable::from_layout(layout, |col| {
                core::mem::take(&mut columns[col as usize])
            }),
            TableIn::Held(held) => match held.take() {
                super::b3l_tables::HeldTable::Wide(trace) => {
                    let mut columns = trace.main_table.columns_blocked();
                    CommittedTable::from_layout(layout, |col| {
                        core::mem::take(&mut columns[col as usize])
                    })
                }
                super::b3l_tables::HeldTable::Narrow(packed) => {
                    CommittedTable::from_narrow(layout, packed)
                }
                super::b3l_tables::HeldTable::Records(records) => {
                    CommittedTable::from_records(layout, records)
                }
            },
        };
        tables.push(committed.map_err(|e| format!("{}: {e:?}", air.name()))?);
        parts[5] += t_table.elapsed().as_secs_f64();
    }
    parts[5] -= narrow.0;
    let prep_secs = t_prep.elapsed().as_secs_f64() - narrow.0;
    let waited_before = super::device_permit::waited_secs();
    let t_held = std::time::Instant::now();
    // ⛔ THE CARD, for the prove's device phases: the permit the STARK prover
    // holds around `multi_prove`. Taken after the host prep above.
    let _card = super::device_permit::hold_labeled("multi_prove");
    let card_budget = multilinear::gpu::reserve_budget();
    let card_reserved_at_start = multilinear::gpu::ledger_reserved();
    let t_a = std::time::Instant::now();
    let config = &artifacts.config;
    let mut transcript = H::sponge();
    absorb_wlfm_statement_for(
        artifacts.family,
        &mut transcript,
        &artifacts.program_id,
        &public_words,
    );
    let block = match proving {
        WlfmProving::Held => BlockCommitted::commit::<H, H::Sponge>(
            tables,
            &artifacts.sizes,
            config,
            multilinear::whir_commit::TreeDrop::uniform(0),
            Narrowing::Wide,
            false,
        ),
        WlfmProving::Streamed => {
            let options = crate::block_whir::BlockOptions::production();
            BlockCommitted::commit::<H, H::Sponge>(
                tables,
                &artifacts.sizes,
                config,
                options.drop_levels,
                options.narrow,
                options.upload_ahead,
            )
            .map(|b| b.with_phase_b_overlap(multilinear_block::PhaseBOverlap::On))
        }
    }
    .map_err(|e| format!("{e:?}"))?;
    let phase_a = t_a.elapsed().as_secs_f64();
    let t_b = std::time::Instant::now();
    let borrowed: Vec<Vec<&multilinear::mle::Mle<F>>> = stacks
        .iter()
        .map(|p| multilinear::stacking::borrow(&p.columns))
        .collect();
    let openings: Vec<BlockPrepared<'_, F, H>> = stacks
        .iter()
        .zip(&borrowed)
        .map(|(p, columns)| BlockPrepared {
            group: p.group,
            tables: p.tables.clone(),
            commitment: &p.commitment,
            columns,
        })
        .collect();
    let order: Vec<usize> = (0..artifacts.sizes.len()).collect();
    let (proof, argues, prepared, group_stamps, commit_ood) = match proving {
        WlfmProving::Held => multilinear_block::block_prove_on_forks::<F, E, H::Sponge, H>(
            block,
            config,
            &mut transcript,
            &openings,
            &[],
            &|g| g,
            &multilinear_block::ArgueDeviation::default(),
        ),
        WlfmProving::Streamed => multilinear_block::block_prove_in_order::<F, E, H::Sponge, H>(
            block,
            config,
            &mut transcript,
            &openings,
            &[],
            &|g| g,
            &multilinear_block::ArgueDeviation::default(),
            &|_| {},
            &order,
            None,
        ),
    }
    .map_err(|e| format!("{e:?}"))?;
    let phase_b = t_b.elapsed().as_secs_f64();
    LAST_WLFM_STAMPS.with(|c| {
        *c.borrow_mut() = Some(WlfmStamps {
            prep: prep_secs,
            prep_parts: parts,
            narrow_pack: narrow.0,
            narrow_tables: narrow.1,
            narrow_cells: narrow.2,
            narrow_bytes: narrow.3,
            phase_a,
            phase_b,
            groups: group_stamps,
            card_budget,
            card_reserved_at_start,
        })
    });
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

/// [`prove_wlfm`] held and streamed as `proving` says.
pub fn prove_wlfm_as(
    program: &LfmProgram,
    build: &WlfmBuild,
    arenas: &[Vec<LfmWord>],
    proving: WlfmProving,
) -> Result<WlfmProof, String> {
    let filled = execute_and_fill_unmerged(program, arenas, build.artifacts.hasher)?;
    prove_wlfm_traces_as(build, filled, true, proving)
}

/// The tables' shapes, and the groups' stacked layouts and domains.
type VerifierFrame = (Vec<(usize, usize)>, Vec<StackedLayout>, Vec<Domain<F>>);

/// Each table's statement at its AIR's preprocessed count, the stacks' checks'
/// inputs, and the groups' layouts and domains: what the verifier derives from
/// the artifacts. Refuses artifacts that do not describe their own program's
/// AIRs and format.
fn verifier_frame(artifacts: &WlfmArtifacts, airs: &WlfmAirs) -> Result<VerifierFrame, String> {
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
    if wlfm_config_under(
        artifacts.rate,
        artifacts.final_poly,
        artifacts.fold,
        artifacts.grind,
        &shapes,
    ) != artifacts.config
    {
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
    with_wlfm_hash!(artifacts.family, |H| verify_block_under::<H>(
        artifacts,
        proof,
        claimed,
        &derived,
        &checks_of,
        &statements,
        &layouts,
        &domains,
        skip_prepared,
        checks,
    ))
}

/// [`verify_wlfm_with`]'s block check under the family's hash `H`: the
/// statement under its tag, the balance the claimed words owe, then the block
/// verifier.
#[allow(clippy::too_many_arguments)]
fn verify_block_under<H: WhirHash>(
    artifacts: &WlfmArtifacts,
    proof: &WlfmProof,
    claimed: &[(u32, LfmWord)],
    derived: &[Commitment],
    checks_of: &[BlockPreparedCheck<'_, F>],
    statements: &[TableStatement<'_, F, E>],
    layouts: &[StackedLayout],
    domains: &[Domain<F>],
    skip_prepared: bool,
    checks: VerifierChecks,
) -> Result<(), String> {
    let mut transcript = H::sponge();
    absorb_wlfm_statement_for(
        artifacts.family,
        &mut transcript,
        &artifacts.program_id,
        claimed,
    );
    // What the tables owe: the `LfmPublic` balance of the claimed words, at the
    // challenges `block_verify_with` is about to draw — the roots block replayed
    // on a fork by the same function, answers and all.
    let mut probe = transcript.clone();
    multilinear_block::absorb_roots_and_answers::<E, _>(
        &mut probe,
        &proof.proof.roots,
        derived,
        &proof.commit_ood,
    );
    let z: FieldElement<E> = probe.sample_field_element();
    let alpha: FieldElement<E> = probe.sample_field_element();
    let expected = super::proof::expected_public_balance(claimed, &z, &alpha)
        .ok_or("a public word's fingerprint is zero")?;
    multilinear_block::block_verify_with::<F, E, H::Sponge, H>(
        &proof.proof,
        &proof.commit_ood,
        &proof.argues,
        &proof.prepared,
        checks_of,
        statements,
        layouts,
        domains,
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
        static CHUNK_LOG: Cell<Option<u32>> = const { Cell::new(None) };
        static NARROW_CHIPS: Cell<Option<bool>> = const { Cell::new(None) };
    }

    /// Every W-LFM build on this thread cuts its chips at `2^log` rows until
    /// the guard drops.
    pub(crate) struct ChunkOverride(Option<u32>);

    impl Drop for ChunkOverride {
        fn drop(&mut self) {
            CHUNK_LOG.with(|c| c.set(self.0));
        }
    }

    pub(crate) fn set_chunk_log(log: u32) -> ChunkOverride {
        ChunkOverride(CHUNK_LOG.with(|c| c.replace(Some(log))))
    }

    pub(super) fn chunk_log() -> Option<u32> {
        CHUNK_LOG.with(Cell::get)
    }

    /// Every W-LFM prove on this thread packs its chips narrow (`Some(true)`)
    /// or not until the guard drops.
    pub(crate) struct NarrowOverride(Option<bool>);

    impl Drop for NarrowOverride {
        fn drop(&mut self) {
            NARROW_CHIPS.with(|c| c.set(self.0));
        }
    }

    pub(crate) fn set_narrow_chips(on: bool) -> NarrowOverride {
        NarrowOverride(NARROW_CHIPS.with(|c| c.replace(Some(on))))
    }

    pub(super) fn narrow_chips() -> Option<bool> {
        NARROW_CHIPS.with(Cell::get)
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
