//! ★ The W-LFM proof: an LFM program proved by the base's stacked-WHIR
//! multilinear prover instead of one STARK per table (D-WHIR §2).
//!
//! The LFM chips are `AirWithBuses` with a single-source constraint IR and bus
//! interactions — exactly what the multilinear table layer consumes. So a W-LFM
//! proof is `multilinear_table::multi_prove` over the program's tables, and
//! what this module adds is the glue around it:
//!
//! - the **prepared stack**: every table's preprocessed prefix (the program's
//!   instruction column groups) committed ONCE per program as one stacked WHIR
//!   commitment, opened at each table's reduced point — DECODE's mechanism,
//!   over several tables at once;
//! - the **statement**, bound before any challenge;
//! - the **identity** `program_id_w`, which folds the prepared roots.
//!
//! # ⛔ The preprocessed count comes from the AIR (D-WHIR §2.4)
//!
//! The LFM AIRs declare their preprocessed columns by COUNT only
//! (`with_preprocessed`), so their `precomputed_columns()` is empty. A statement
//! built from that list counts zero preprocessed columns: with no prepared
//! opening the program would be bound by nothing, and with one the honest proof
//! would be refused. This verifier builds every statement with
//! [`TableLayout::statement_with_prepared_prefix`] at `air.num_precomputed_columns()`
//! and refuses any plan that does not settle exactly `0..count` of every table.
//!
//! # The hash is a TYPE
//!
//! [`WhirLfmHash`] and [`WhirLfmTranscript`] are RPX by type, never through
//! `LAMBDA_VM_WHIR_HASH` (whose code default is keccak). The parent verifies a
//! W-LFM child in-guest with the algebraic sponge, so a keccak W-LFM proof would
//! be unverifiable by its parent; naming the hash as a type makes that proof
//! unconstructible rather than merely rejected.
//!
//! # Policy A
//!
//! One commitment group holds every table, preprocessed prefix included, so the
//! prefix is committed twice: in the main stack and in the prepared stack (as
//! DECODE's is in the base). Policy B (the prefix out of the main stack) is a
//! W1 measurement, not this module's default.

use std::borrow::Cow;
use std::time::Instant;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::hash::platform_keccak::PlatformKeccak256 as Keccak256;
use digest::Digest;
use math::field::element::FieldElement;
use math::field::traits::IsPrimeField;
use multilinear::mle::Mle;
use multilinear::stacked_eval::StackedCommitment;
use multilinear::stacking::StackedLayout;
use multilinear::whir::Domain;
use multilinear::whir_chain::ChainConfig;
use multilinear::whir_hash::{RpxWhir, WhirHash};
use stark::config::Commitment;
use stark::multilinear_air::Uniforms;
use stark::multilinear_table::{
    self, CommittedTable, CommittedTables, MultiProof, Prepared, PreparedCheck, PreparedColumn,
    TableLayout, TableStatement,
};
use stark::proof::options::ProofOptions;

use crate::tables::types::{GoldilocksExtension, GoldilocksField};

use super::airs::{ChipSet, DynLfmAir, LfmAirs, NUM_LFM_CHIPS};
use super::compiler::{ColumnGroup, LfmProgram};
use super::executor::{LfmExecError, LfmExecution, execute};
use super::hash::HasherKind;
use super::statement::LFM_MACHINE_VERSION;
use super::trace::{LfmTraces, build_traces_with_hasher, range_group};
use super::word::LfmWord;

type F = GoldilocksField;
type E = GoldilocksExtension;

/// The WHIR hash every W-LFM proof commits, transcripts and grinds with.
pub type WhirLfmHash = RpxWhir;

/// The Fiat–Shamir transcript of a W-LFM proof: [`WhirLfmHash`]'s own sponge,
/// which is the one the in-guest `WhirTranscript` replays.
pub type WhirLfmTranscript = DefaultTranscript<E, <WhirLfmHash as WhirHash>::Transcript>;

/// Commitment groups a W-LFM proof carries under policy A: one, holding every
/// table.
pub const WHIR_LFM_GROUPS: usize = 1;

/// The statement's domain tag. Padded to [`STATEMENT_TAG_BYTES`] so the
/// program id after it — and the public felts after that — start on a felt
/// boundary without a computed pad.
const LFM_WHIR_STATEMENT_TAG: &[u8] = b"LAMBDAVM_LFM_WHIR_STATEMENT_V1";

/// Bytes the padded statement tag occupies.
const STATEMENT_TAG_BYTES: usize = 32;

/// The identity's domain tag.
const LFM_WHIR_PROGRAM_TAG: &[u8] = b"LAMBDAVM_LFM_WHIR_PROGRAM_V1";

/// Bytes one felt occupies in the sponge's byte stream.
const FELT_BYTES: usize = 8;

/// Why a W-LFM build, prove or verify stopped.
#[derive(Debug)]
pub enum WhirLfmError {
    /// The program, the AIR set or the proof does not have the shape this path
    /// proves — a caller or a program the path does not cover, never a hostile
    /// proof's value.
    Shape(String),
    /// The LFM interpreter refused the program's execution.
    Exec(LfmExecError),
    /// The multilinear argument refused.
    Argument(multilinear::Error),
}

impl From<multilinear::Error> for WhirLfmError {
    fn from(e: multilinear::Error) -> Self {
        Self::Argument(e)
    }
}

fn shape_error(message: impl Into<String>) -> WhirLfmError {
    WhirLfmError::Shape(message.into())
}

/// ★ Where a W-LFM proof commits each table's preprocessed prefix (D-WHIR §2.4).
///
/// Either way the prefix's claims are settled by the prepared opening against
/// the program's own stack, which is what binds the program. The policies
/// differ in whether the main stack ALSO carries the prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrepPolicy {
    /// Policy A: in the main stack and in the prepared stack (DECODE's shape in
    /// the base). No layout change anywhere; the prefix is committed twice.
    Both,
    /// Policy B: in the prepared stack ONLY. The main stack holds each table's
    /// value columns, so it is smaller — a wrap's goes from a half-empty 2^27
    /// to a full 2^26 — and its claims on the prefix are the opening's alone.
    PreparedOnly,
}

impl PrepPolicy {
    /// The name the banner prints and the knob spells.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Both => "both",
            Self::PreparedOnly => "prepared",
        }
    }

    /// The byte `program_id_w` folds.
    const fn tag(self) -> u8 {
        match self {
            Self::Both => 0,
            Self::PreparedOnly => 1,
        }
    }

    /// Whether the main stack leaves the prefix out.
    pub const fn excludes_prefix(self) -> bool {
        matches!(self, Self::PreparedOnly)
    }
}

/// A W-LFM program's verifier-side artifacts: everything a verifier needs about
/// the program, derived from the program, and never read from a proof.
///
/// ⚠ Small on purpose. The prepared stack's codewords and columns are the
/// prover's ([`WhirLfmPrepared`]); a verifier holds the roots, the layout and
/// the domain they were committed under, cloned off the commitment rather than
/// rebuilt, so a verifier cannot settle the opening against a second derivation
/// of the stack the prover actually committed.
#[derive(Clone, Debug)]
pub struct WhirLfmArtifacts {
    /// The `LFM_HASH` permutation the program runs.
    pub hasher: HasherKind,
    /// The chip set: always the WHIR recursion one ([`check_chip_set`]).
    pub chip_set: ChipSet,
    /// `LFM_HASH` instances: one unless the program's hash split divided it.
    pub hash_chunks: usize,
    /// Each table's height in variables, in `LfmAirs::air_refs` order.
    ///
    /// ★ PROGRAM SHAPE: a chip's trace is exactly its instruction group's padded
    /// rows, so the heights are the program's and a proof cannot restate one.
    pub table_num_vars: Vec<u8>,
    /// The chain config the program proves under — its shapes and the process
    /// format, through `multilinear_prove::chain_config`.
    pub config: ChainConfig,
    /// Where the preprocessed prefix is committed: a FORMAT choice, folded into
    /// the identity.
    pub policy: PrepPolicy,
    /// The prepared stack's roots: derived from the program, never read from a
    /// proof.
    pub prepared_roots: Vec<Commitment>,
    /// Where each prepared column is settled: every table's `0..count`, in table
    /// order.
    pub prepared_at: Vec<PreparedColumn>,
    /// The prepared stack's layout, cloned off the commitment.
    pub prepared_layout: StackedLayout,
    /// The prepared stack's domain, cloned off the commitment.
    pub prepared_domain: Domain<F>,
    /// `program_id_w` — see [`lfm_program_id_w`].
    pub program_id: Commitment,
}

/// The prover's half of a build: the prepared stack's columns and commitment.
pub struct WhirLfmPrepared {
    /// Every table's preprocessed prefix, in stack order.
    pub columns: Vec<Mle<F>>,
    pub commitment: StackedCommitment<F, WhirLfmHash>,
}

/// One program's W-LFM build: the verifier's artifacts and the prover's stack.
pub struct WhirLfmBuild {
    pub artifacts: WhirLfmArtifacts,
    pub prepared: WhirLfmPrepared,
}

/// A W-LFM proof and the words its execution published.
#[derive(Clone, Debug)]
pub struct WhirLfmProof {
    pub proof: MultiProof<F, E>,
    /// The public output, in emission order — indices `0..n`.
    pub public_words: Vec<(u32, LfmWord)>,
}

/// ⛔ The chip sets a W-LFM program may carry: the WHIR recursion programs'
/// ten chips (no keccak family, no BLAKE3 family, no `BITWISE`).
///
/// The families carry FIXED preprocessed tables (`KECCAK_RC`, `BITWISE`) and a
/// chip with none at all (`KECCAK_RND`); none of that has a route through the
/// prepared stack here, so a program that instantiates them is refused rather
/// than proved with columns nothing binds.
pub fn check_chip_set(chip_set: ChipSet) -> Result<(), WhirLfmError> {
    if chip_set.keccak || chip_set.blake3 || chip_set.bitwise {
        return Err(shape_error(format!(
            "a W-LFM program carries the WHIR recursion chip set (no keccak, no BLAKE3, no \
             BITWISE), and this one is {chip_set:?}"
        )));
    }
    Ok(())
}

/// The W-LFM AIR set: the program's chips with NO STARK preprocessed roots.
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

/// [`whir_lfm_airs`] for a program's artifacts.
pub fn airs_for(artifacts: &WhirLfmArtifacts, options: &ProofOptions) -> LfmAirs {
    whir_lfm_airs(
        artifacts.hasher,
        artifacts.chip_set,
        artifacts.hash_chunks,
        options,
    )
}

/// Each table's preprocessed prefix as the program commits it, in
/// `LfmAirs::air_refs` order for the WHIR recursion chip set: the five ALU
/// groups, `LFM_HASH` (one group, or one per chunk), `LANES`, `HINT`,
/// `PUBLIC` and the fixed `RANGE` group.
///
/// These are the SAME groups the trace builder copies into each chip's leading
/// columns (`trace::build_traces_walked`), so the prepared stack and the
/// trace's prefix are one object.
fn program_prep_groups(program: &LfmProgram) -> Vec<Cow<'_, ColumnGroup>> {
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
fn group_columns(group: &ColumnGroup) -> Vec<Vec<FieldElement<F>>> {
    (0..group.width)
        .map(|column| {
            (0..group.padded_rows)
                .map(|row| group.data[row * group.width + column])
                .collect()
        })
        .collect()
}

/// ★ The chain config a W-LFM program proves under: the production
/// derivation (`multilinear_prove::chain_config`) over its table shapes. The
/// ONE call both sides make — the build, every plan, the emitter.
pub fn whir_lfm_config(shapes: &[(usize, usize)]) -> ChainConfig {
    let config = crate::multilinear_prove::chain_config(shapes);
    #[cfg(test)]
    let config = test_grind::apply(config);
    config
}

/// Every table's `(main width, height in variables)`, in table order: the width
/// is the AIR's, the height the artifacts'.
pub fn table_shapes(airs: &[DynLfmAir<'_>], table_num_vars: &[u8]) -> Vec<(usize, usize)> {
    airs.iter()
        .zip(table_num_vars)
        .map(|(air, &num_vars)| (air.trace_layout().0, num_vars as usize))
        .collect()
}

/// The prepared plan every table's preprocessed prefix must be settled by: each
/// table's `0..count`, in table order, a table with none absent.
fn expected_plan(counts: &[usize]) -> Vec<PreparedColumn> {
    counts
        .iter()
        .enumerate()
        .flat_map(|(table, &count)| multilinear_table::leading_columns(table, count))
        .collect()
}

/// ★ Builds a program's W-LFM artifacts and the prover's prepared stack, under
/// the process's prefix policy (`LAMBDA_VM_LFM_WHIR_PREP`).
///
/// ONE derivation for both sides: the prover and every verifier of this program
/// take the stack's roots, layout and domain from here, and the stack is a
/// function of the program's instruction groups and the process format alone.
pub fn build_whir_artifacts(
    program: &LfmProgram,
    options: &ProofOptions,
    hasher: HasherKind,
) -> Result<WhirLfmBuild, WhirLfmError> {
    build_whir_artifacts_under(
        program,
        options,
        hasher,
        crate::lfm_prover_knob::prep_policy(),
    )
}

/// [`build_whir_artifacts`] under an explicit prefix policy, so one process can
/// build the same program both ways (W1's A/B on one binary).
pub fn build_whir_artifacts_under(
    program: &LfmProgram,
    options: &ProofOptions,
    hasher: HasherKind,
    policy: PrepPolicy,
) -> Result<WhirLfmBuild, WhirLfmError> {
    let chip_set = ChipSet::for_program_with_hasher(program, hasher);
    check_chip_set(chip_set)?;
    let hash_chunks = program.hash_chunk_count();
    let groups = program_prep_groups(program);
    let airs = whir_lfm_airs(hasher, chip_set, hash_chunks, options);
    let refs = airs.air_refs();
    if refs.len() != groups.len() {
        return Err(shape_error(format!(
            "the AIR set has {} tables and the program commits {} instruction groups",
            refs.len(),
            groups.len()
        )));
    }

    let mut table_num_vars = Vec::with_capacity(refs.len());
    let mut counts = Vec::with_capacity(refs.len());
    for (air, group) in refs.iter().zip(&groups) {
        let count = air.num_precomputed_columns();
        if group.width != count {
            return Err(shape_error(format!(
                "{}: the program commits {} preprocessed columns and the AIR declares {count}",
                air.name(),
                group.width
            )));
        }
        if !group.padded_rows.is_power_of_two() {
            return Err(shape_error(format!(
                "{}: {} rows, which the hypercube cannot hold",
                air.name(),
                group.padded_rows
            )));
        }
        table_num_vars.push(group.padded_rows.trailing_zeros() as u8);
        counts.push(count);
    }
    let shapes = table_shapes(&refs, &table_num_vars);
    let config = whir_lfm_config(&shapes);

    // The prepared stack: every table's prefix, in table order.
    let mut columns: Vec<Mle<F>> = Vec::new();
    let mut prep_shapes: Vec<(usize, usize)> = Vec::new();
    for (group, &count) in groups.iter().zip(&counts) {
        if count == 0 {
            continue;
        }
        prep_shapes.push((count, group.padded_rows.trailing_zeros() as usize));
        for column in group_columns(group) {
            columns.push(Mle::new(column)?);
        }
    }
    let prepared_at = expected_plan(&counts);
    if prepared_at.len() != columns.len() {
        return Err(shape_error(format!(
            "the prepared stack has {} columns and {} destinations",
            columns.len(),
            prepared_at.len()
        )));
    }
    let layout = multilinear_table::global_layout(&prep_shapes, config.format.stack)?;
    // ⛔ THE CARD, FOR THE FIRST OF A W-LFM PROOF'S TWO DEVICE PHASES, as
    // `program_census::build_artifacts_counted` holds it for a STARK build: the
    // prepared commit runs on the device outside `multi_prove`, so concurrent
    // siblings must serialize it too. Inert unless a driver armed the permit.
    let commitment = {
        let _card = super::device_permit::hold_labeled("build_artifacts");
        StackedCommitment::<F, WhirLfmHash>::commit(
            layout,
            &multilinear::stacking::borrow(&columns),
            None,
            &config,
        )?
    };
    let prepared_roots = commitment.roots();
    if prepared_roots.is_empty() {
        return Err(shape_error("the prepared stack commits to nothing"));
    }
    let program_id = lfm_program_id_w(
        hasher,
        chip_set,
        &table_num_vars,
        &counts,
        &prepared_roots,
        &config,
        policy,
    );
    Ok(WhirLfmBuild {
        artifacts: WhirLfmArtifacts {
            hasher,
            chip_set,
            hash_chunks,
            table_num_vars,
            config,
            policy,
            prepared_roots,
            prepared_at,
            prepared_layout: commitment.layout().clone(),
            prepared_domain: commitment.domain().clone(),
            program_id,
        },
        prepared: WhirLfmPrepared {
            columns,
            commitment,
        },
    })
}

/// ★ `program_id_w`: the identity a parent interns for a W-LFM child.
///
/// Keccak over bytes, like [`super::statement::lfm_program_id`]: a host-side
/// identity, not a commitment. It folds everything that makes the program this
/// program under this prover — the hasher, the chip set, every table's height
/// and preprocessed count, the prepared roots (which bind the instruction
/// groups themselves), and the WHIR format the proof runs under (the absorbed
/// config words plus the stack cap and the Merkle cap policy, which the
/// statement does not absorb), the prefix policy and the WHIR hash's name.
pub fn lfm_program_id_w(
    hasher: HasherKind,
    chip_set: ChipSet,
    table_num_vars: &[u8],
    counts: &[usize],
    prepared_roots: &[Commitment],
    config: &ChainConfig,
    policy: PrepPolicy,
) -> Commitment {
    let mut h = Keccak256::new();
    h.update(LFM_WHIR_PROGRAM_TAG);
    h.update(LFM_MACHINE_VERSION.to_le_bytes());
    h.update([hasher.as_tag()]);
    h.update([chip_set.as_tag()]);
    h.update(<WhirLfmHash as WhirHash>::NAME.as_bytes());
    h.update((table_num_vars.len() as u64).to_le_bytes());
    for (&num_vars, &count) in table_num_vars.iter().zip(counts) {
        h.update([num_vars]);
        h.update((count as u64).to_le_bytes());
    }
    h.update((prepared_roots.len() as u64).to_le_bytes());
    for root in prepared_roots {
        h.update(root);
    }
    h.update(config_bytes(config));
    h.update([config.format.stack.get() as u8]);
    h.update(cap_policy_bytes(config.format.cap));
    h.update([policy.tag()]);
    h.finalize().into()
}

/// The Merkle cap policy as two bytes: a variant tag and its height.
fn cap_policy_bytes(cap: crypto::merkle_tree::cap::CapPolicy) -> [u8; 2] {
    use crypto::merkle_tree::cap::CapPolicy;
    match cap {
        CapPolicy::Off => [0, 0],
        CapPolicy::Auto => [1, 0],
        CapPolicy::Fixed(height) => [2, height],
    }
}

/// The chain config's absorbed words — `multilinear_prove::absorb`'s tail and
/// `whir_statement::push_config`'s: `log_blowup`, the fold word, the query
/// count, each as a little-endian `u64`, then the three grind bytes.
fn config_bytes(config: &ChainConfig) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(3 * 8 + 3);
    for value in [
        config.log_blowup as u64,
        config.fold_word(),
        config.num_queries as u64,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&[config.grind.folding, config.grind.ood, config.grind.query]);
    bytes
}

/// The statement's two CONSTANT runs, around the public felts.
///
/// ONE source for the layout: the host absorb writes these, and the in-guest
/// leg (`whir_leg`) absorbs the same two runs as program constants around the
/// hinted felts, so neither can drift from the other.
pub struct StatementRuns {
    /// The tag padded to 32 bytes, `program_id_w`, the machine version and the
    /// word count — 80 bytes, so the felts after it start aligned.
    pub head: Vec<u8>,
    /// The heights (a `u64` length and a byte each), the config words and the
    /// grind trailer, then the pad that brings the whole statement to a felt
    /// boundary so the roots after it start aligned.
    pub tail: Vec<u8>,
}

/// [`StatementRuns`] for a program's statement over `num_words` public words.
pub fn statement_runs(
    program_id: &Commitment,
    num_words: usize,
    table_num_vars: &[u8],
    config: &ChainConfig,
) -> StatementRuns {
    let mut head = Vec::with_capacity(STATEMENT_TAG_BYTES + 32 + 16);
    head.extend_from_slice(LFM_WHIR_STATEMENT_TAG);
    head.resize(STATEMENT_TAG_BYTES, 0);
    head.extend_from_slice(program_id);
    head.extend_from_slice(&u64::from(LFM_MACHINE_VERSION).to_le_bytes());
    head.extend_from_slice(&(num_words as u64).to_le_bytes());
    debug_assert_eq!(head.len() % FELT_BYTES, 0, "the felts start aligned");

    let mut tail = Vec::with_capacity(8 + table_num_vars.len() + 3 * 8 + 3 + FELT_BYTES);
    tail.extend_from_slice(&(table_num_vars.len() as u64).to_le_bytes());
    tail.extend_from_slice(table_num_vars);
    tail.extend_from_slice(&config_bytes(config));
    // The head and the felts are whole felts, so the tail alone decides the pad.
    let pad = crate::statement::statement_padding(tail.len());
    tail.resize(tail.len() + pad, 0);
    StatementRuns { head, tail }
}

/// A public lane as the felt the sponge reads: its canonical value, eight bytes
/// big-endian — `sponge_leaf_bytes`' own grouping, so the host bytes and the
/// machine's felt are the same element of the stream.
fn lane_bytes(lane: &FieldElement<F>) -> [u8; FELT_BYTES] {
    F::canonical(lane.value()).to_be_bytes()
}

/// ★ Binds a W-LFM statement: the tag, `program_id_w`, the machine version,
/// the word count, each public word as its four lane felts, the heights, the
/// config and the pad — before any challenge is drawn.
///
/// The word indices are not absorbed: they are `0..n` by construction
/// (`LfmBuilder::public` auto-increments), and the verifier REFUSES any other
/// claim rather than trusting it ([`verify_whir_checked`]).
pub fn absorb_whir_lfm_statement(
    transcript: &mut impl IsTranscript<E>,
    program_id: &Commitment,
    public_words: &[(u32, LfmWord)],
    table_num_vars: &[u8],
    config: &ChainConfig,
) {
    let runs = statement_runs(program_id, public_words.len(), table_num_vars, config);
    transcript.append_bytes(&runs.head);
    for (_, word) in public_words {
        for lane in word {
            transcript.append_bytes(&lane_bytes(lane));
        }
    }
    transcript.append_bytes(&runs.tail);
}

/// ★ Everything both sides derive from a program's artifacts and its AIR set:
/// ONE derivation for the prover, the host verifier, the in-guest emitter, its
/// arena writer and its cost form.
///
/// ⛔ `counts` ARE THE AIRS', and [`Self::build`] refuses a prepared plan that
/// does not settle exactly `0..count` of every table (D-WHIR §2.4).
pub struct WhirLfmPlan<'a> {
    pub config: ChainConfig,
    /// The artifacts' prefix policy.
    pub policy: PrepPolicy,
    /// `(main width, height in variables)` per table.
    pub shapes: Vec<(usize, usize)>,
    /// `(columns in the main stack, height)` per table: [`Self::shapes`] under
    /// policy A, the value columns alone under policy B.
    pub main_shapes: Vec<(usize, usize)>,
    pub layouts: Vec<TableLayout<'a, F, E>>,
    /// Each table's preprocessed count, from its AIR.
    pub counts: Vec<usize>,
    /// Each table's name, for messages.
    pub names: Vec<String>,
    /// The main group's stack and domain — one group, over `main_shapes`.
    pub group_layouts: Vec<StackedLayout>,
    pub group_domains: Vec<Domain<F>>,
}

impl<'a> WhirLfmPlan<'a> {
    pub fn build(
        artifacts: &WhirLfmArtifacts,
        airs: &[DynLfmAir<'a>],
    ) -> Result<Self, WhirLfmError> {
        check_chip_set(artifacts.chip_set)?;
        if airs.len() != artifacts.table_num_vars.len() {
            return Err(shape_error(format!(
                "the AIR set has {} tables and the artifacts state {} heights",
                airs.len(),
                artifacts.table_num_vars.len()
            )));
        }
        let shapes = table_shapes(airs, &artifacts.table_num_vars);
        let config = whir_lfm_config(&shapes);
        // ★ The artifacts' config is the one the prepared stack was committed
        // under; the proof argues under the one the shapes give now. They are
        // the same derivation, asserted rather than assumed — a stack committed
        // under another blowup, fold schedule or cap would be opened as nothing
        // the verifier can settle.
        if config != artifacts.config {
            return Err(shape_error(format!(
                "the artifacts were built under {:?} and this process argues under {config:?}",
                artifacts.config
            )));
        }
        let layouts: Vec<TableLayout<'a, F, E>> = airs
            .iter()
            .zip(&shapes)
            .map(|(air, &(width, num_vars))| {
                TableLayout::new(
                    air.constraint_program(),
                    air.constraints_meta(),
                    air.bus_interactions(),
                    width,
                    num_vars,
                    Uniforms::default(),
                )
                .map_err(WhirLfmError::from)
            })
            .collect::<Result<_, _>>()?;
        let counts: Vec<usize> = airs
            .iter()
            .map(|air| air.num_precomputed_columns())
            .collect();
        let names: Vec<String> = airs.iter().map(|air| air.name().to_string()).collect();

        // ⛔ THE REFUSAL. Every table's preprocessed prefix is settled by the
        // prepared opening, exactly `0..count`, or the plan describes a proof
        // whose program columns nothing binds.
        let want = expected_plan(&counts);
        if artifacts.prepared_at != want {
            let uncovered: Vec<&str> = names
                .iter()
                .zip(&counts)
                .enumerate()
                .filter(|&(table, (_, &count))| {
                    count > 0
                        && multilinear_table::leading_columns(table, count)
                            .iter()
                            .any(|c| !artifacts.prepared_at.contains(c))
                })
                .map(|(_, (name, _))| name.as_str())
                .collect();
            return Err(shape_error(format!(
                "the prepared plan does not settle every table's preprocessed prefix exactly: \
                 {} entries against {} wanted; uncovered: {uncovered:?}",
                artifacts.prepared_at.len(),
                want.len()
            )));
        }

        let main_shapes: Vec<(usize, usize)> = shapes
            .iter()
            .zip(&counts)
            .map(|(&(width, num_vars), &count)| {
                if artifacts.policy.excludes_prefix() {
                    (width - count, num_vars)
                } else {
                    (width, num_vars)
                }
            })
            .collect();
        let sizes = [airs.len()];
        let (group_layouts, group_domains) =
            crate::multilinear_prove::stacks(&main_shapes, &sizes, &config)
                .map_err(|e| shape_error(format!("{e:?}")))?;
        Ok(Self {
            config,
            policy: artifacts.policy,
            shapes,
            main_shapes,
            layouts,
            counts,
            names,
            group_layouts,
            group_domains,
        })
    }

    /// Tables per commitment group: every table in one.
    pub fn sizes(&self) -> Vec<usize> {
        vec![self.layouts.len()]
    }

    /// Per table, the leading columns the main stack leaves out: every
    /// prefix under policy B, none (an empty list) under policy A — the form
    /// `CommittedTables::commit_grouped_settled` takes.
    pub fn settled_out_of_main(&self) -> Vec<usize> {
        if self.policy.excludes_prefix() {
            self.counts.clone()
        } else {
            Vec::new()
        }
    }

    /// Each table's statement, at its AIR's preprocessed count.
    pub fn statements(&self) -> Vec<TableStatement<'_, F, E>> {
        self.layouts
            .iter()
            .zip(&self.counts)
            .map(|(layout, &count)| layout.statement_with_prepared_prefix(count))
            .collect()
    }
}

/// The prepared check a verifier settles the opening against — the artifacts'
/// derived roots, never a proof's.
pub fn prepared_check(artifacts: &WhirLfmArtifacts) -> PreparedCheck<'_, F> {
    PreparedCheck {
        roots: &artifacts.prepared_roots,
        layout: &artifacts.prepared_layout,
        domain: &artifacts.prepared_domain,
        at: &artifacts.prepared_at,
    }
}

/// ★ Proves `program` as a W-LFM proof: execute, fill the traces, and argue
/// every table in one multilinear proof with the prepared opening last.
///
/// The hasher is the build's, for the reason `proof::lfm_prove` takes it from
/// the artifacts: the identity binds it.
pub fn lfm_prove_whir(
    program: &LfmProgram,
    build: &WhirLfmBuild,
    arenas: &[Vec<LfmWord>],
    options: &ProofOptions,
) -> Result<WhirLfmProof, WhirLfmError> {
    let hasher = build.artifacts.hasher;
    let t = Instant::now();
    let LfmExecution {
        records,
        public_words,
        memory,
        split: exec_split,
    } = execute(program, arenas, &hasher).map_err(WhirLfmError::Exec)?;
    let execute_secs = t.elapsed().as_secs_f64();
    drop(memory);

    let t = Instant::now();
    let mut traces = build_traces_with_hasher(program, &records, hasher);
    let fill_secs = t.elapsed().as_secs_f64();
    drop(records);

    let t = Instant::now();
    let waited_before = super::device_permit::waited_secs();
    let proof = prove_traces_whir(build, &mut traces, &public_words, options, true)?;
    let permit_wait = (super::device_permit::waited_secs() - waited_before).max(0.0);
    let multi_prove_secs = (t.elapsed().as_secs_f64() - permit_wait).max(0.0);
    super::proof::record_prove_split(super::proof::ProveSplit {
        execute: execute_secs,
        fill: fill_secs,
        multi_prove: multi_prove_secs,
        permit_wait,
        exec: exec_split,
    });
    Ok(WhirLfmProof {
        proof,
        public_words,
    })
}

/// Proves an already-built trace set. `check_prefix` compares every table's
/// preprocessed prefix against the prepared stack before committing: an honest
/// prover must never hand out a proof nobody can verify. Only the tamper tests
/// turn it off, to show the VERIFIER refuses what the prover would.
pub(crate) fn prove_traces_whir(
    build: &WhirLfmBuild,
    traces: &mut LfmTraces,
    public_words: &[(u32, LfmWord)],
    options: &ProofOptions,
    check_prefix: bool,
) -> Result<MultiProof<F, E>, WhirLfmError> {
    prove_traces_whir_opening(build, traces, public_words, options, check_prefix, true)
}

/// [`prove_traces_whir`] with the prepared opening optional — `false` proves as
/// if the program had no prepared stack: no derived root absorbed, no opening.
/// Only the tamper tests take `false`, to build the proof a count-zero verifier
/// would accept and show this path's verifier refuses it.
pub(crate) fn prove_traces_whir_opening(
    build: &WhirLfmBuild,
    traces: &mut LfmTraces,
    public_words: &[(u32, LfmWord)],
    options: &ProofOptions,
    check_prefix: bool,
    open_prepared: bool,
) -> Result<MultiProof<F, E>, WhirLfmError> {
    // ⛔ THE CARD: the same exclusive permit the STARK prover holds around
    // `multi_prove`, so a tree driver serializes W-LFM proves exactly as it
    // serializes today's.
    let _card = super::device_permit::hold_labeled("multi_prove");
    // The prove's own split, recorded as a `WHIR PROVE SPLIT W-LFM` line under
    // `LAMBDA_VM_BASE_SPLIT=1` (the base's instrument, with the base's stages:
    // prep = the tables built, absorb = the statement, commit, prove).
    let t_wall = Instant::now();
    let artifacts = &build.artifacts;
    let airs = airs_for(artifacts, options);
    let refs = airs.air_refs();
    // The layouts the committed tables are built against are the PLAN's — the
    // same derivation the verifier makes — moved out rather than rebuilt.
    let plan = WhirLfmPlan::build(artifacts, &refs)?;
    // Under policy B the main stack leaves every prefix out; the prepared
    // opening, which `multi_prove` refuses to omit then, is its only binding.
    let settled = plan.settled_out_of_main();
    let WhirLfmPlan {
        config,
        shapes,
        layouts,
        counts,
        ..
    } = plan;
    let pairs = airs.air_trace_pairs(traces);
    if pairs.len() != layouts.len() {
        return Err(shape_error(format!(
            "{} traces for {} tables",
            pairs.len(),
            layouts.len()
        )));
    }

    // Where each table's prefix starts in the prepared stack.
    let mut prepared_at = 0usize;
    let mut tables = Vec::with_capacity(pairs.len());
    for ((((air, trace, _), layout), &(width, num_vars)), &count) in
        pairs.into_iter().zip(layouts).zip(&shapes).zip(&counts)
    {
        if trace.main_table.width != width || trace.main_table.height != 1usize << num_vars {
            return Err(shape_error(format!(
                "{}: the trace is {} columns of {} rows, the plan {width} of 2^{num_vars}",
                air.name(),
                trace.main_table.width,
                trace.main_table.height
            )));
        }
        let mut columns = trace.columns_main();
        if check_prefix {
            for (column, pinned) in columns[..count]
                .iter()
                .zip(&build.prepared.columns[prepared_at..prepared_at + count])
            {
                if column.as_slice() != pinned.evals() {
                    return Err(shape_error(format!(
                        "{}: a preprocessed column is not the program's",
                        air.name()
                    )));
                }
            }
        }
        prepared_at += count;
        tables.push(CommittedTable::from_layout(layout, |col| {
            core::mem::take(&mut columns[col as usize])
        })?);
    }

    let prep_secs = t_wall.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut transcript = WhirLfmTranscript::new(&[]);
    absorb_whir_lfm_statement(
        &mut transcript,
        &artifacts.program_id,
        public_words,
        &artifacts.table_num_vars,
        &config,
    );
    let absorb_secs = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let committed = CommittedTables::<F, E, WhirLfmHash>::commit_grouped_settled(
        tables,
        &[counts.len()],
        &config,
        &settled,
    )?;
    let commit_secs = t.elapsed().as_secs_f64();
    let borrowed = multilinear::stacking::borrow(&build.prepared.columns);
    let prepared = Prepared {
        commitment: &build.prepared.commitment,
        columns: &borrowed,
        at: &artifacts.prepared_at,
    };
    let t = Instant::now();
    let proof = multilinear_table::multi_prove(
        &committed,
        &config,
        &mut transcript,
        open_prepared.then_some(prepared),
    )?;
    if multilinear::whir_split::enabled() {
        multilinear::whir_split::push_prover(multilinear::whir_split::ProverSplit {
            index: multilinear::whir_split::LFM_INDEX,
            prep: prep_secs,
            absorb: absorb_secs,
            commit: commit_secs,
            prove: t.elapsed().as_secs_f64(),
            wall: t_wall.elapsed().as_secs_f64(),
            airs: counts.len(),
            ..Default::default()
        });
    }
    Ok(proof)
}

/// ★ Verifies a W-LFM proof against the program's artifacts and the CLAIMED
/// public words: `Ok` exactly when the proof is accepted.
///
/// The four steps of `proof::verify_against_chunked_with`, on this path: bind
/// the statement over the claimed words; replay the roots block on a fork to
/// recover `(z, α)`; compute the `LfmPublic` balance the tables must reach from
/// the claimed words; and run `multi_verify` with every statement at its AIR's
/// preprocessed count and the prepared check built from the artifacts.
pub fn verify_whir_checked(
    artifacts: &WhirLfmArtifacts,
    proof: &MultiProof<F, E>,
    claimed_public: &[(u32, LfmWord)],
    options: &ProofOptions,
) -> Result<(), WhirLfmError> {
    // The indices the statement does not absorb must be the ones it implies.
    for (position, (index, _)) in claimed_public.iter().enumerate() {
        if *index as usize != position {
            return Err(shape_error(format!(
                "public word {position} claims index {index}; a W-LFM statement binds words by \
                 position"
            )));
        }
    }
    let airs = airs_for(artifacts, options);
    let refs = airs.air_refs();
    let plan = WhirLfmPlan::build(artifacts, &refs)?;
    let statements = plan.statements();

    let mut transcript = WhirLfmTranscript::new(&[]);
    absorb_whir_lfm_statement(
        &mut transcript,
        &artifacts.program_id,
        claimed_public,
        &artifacts.table_num_vars,
        &plan.config,
    );
    // What the tables owe: the `LfmPublic` balance of the claimed words, at the
    // challenges `multi_verify` is about to draw — so the roots block is
    // replayed on a fork, by the SAME function, with the same derived roots.
    let mut probe = transcript.clone();
    multilinear_table::absorb_roots::<E, _>(&mut probe, &proof.roots, &artifacts.prepared_roots);
    let z: FieldElement<E> = probe.sample_field_element();
    let alpha: FieldElement<E> = probe.sample_field_element();
    let expected = super::proof::expected_public_balance(claimed_public, &z, &alpha)
        .ok_or(WhirLfmError::Argument(multilinear::Error::BusImbalance))?;

    multilinear_table::multi_verify_settled::<F, E, _, WhirLfmHash>(
        proof,
        &statements,
        &plan.group_layouts,
        &plan.group_domains,
        &plan.sizes(),
        &expected,
        &plan.config,
        &mut transcript,
        Some(prepared_check(artifacts)),
        plan.policy.excludes_prefix(),
    )?;
    Ok(())
}

/// [`verify_whir_checked`] as a verdict: `true` exactly when accepted.
pub fn lfm_verify_whir(
    artifacts: &WhirLfmArtifacts,
    proof: &MultiProof<F, E>,
    claimed_public: &[(u32, LfmWord)],
    options: &ProofOptions,
) -> bool {
    verify_whir_checked(artifacts, proof, claimed_public, options).is_ok()
}

/// A TEST-ONLY switch that proves W-LFM chains without grinding.
///
/// A W-LFM chain grinds 20 bits before each redrawable challenge, and on a
/// laptop's CPU that search is most of a small proof's time (≈ 24 grinds, ≈ 29 s
/// for `TrivialV0`). The refusals the tests check do not depend on it, so they
/// run with it off; the round trip that anchors the production config keeps it
/// on. Thread-local, and applied inside [`whir_lfm_config`], so the prover, the
/// verifier and the emitter of one test all see the same config and a
/// production build cannot reach it at all.
#[cfg(test)]
pub(crate) mod test_grind {
    use std::cell::Cell;

    use multilinear::whir_chain::{ChainConfig, GrindBits};

    thread_local! {
        static OFF: Cell<bool> = const { Cell::new(false) };
    }

    /// Grinding stays off on this thread while the guard lives.
    pub(crate) struct Ungrinded {
        before: bool,
    }

    impl Drop for Ungrinded {
        fn drop(&mut self) {
            OFF.with(|off| off.set(self.before));
        }
    }

    /// Every W-LFM config on this thread ungrinded until the guard drops.
    pub(crate) fn off() -> Ungrinded {
        Ungrinded {
            before: OFF.with(|off| off.replace(true)),
        }
    }

    pub(super) fn apply(config: ChainConfig) -> ChainConfig {
        if OFF.with(Cell::get) {
            ChainConfig {
                grind: GrindBits::uniform(0),
                ..config
            }
        } else {
            config
        }
    }
}
