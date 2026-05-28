use std::marker::PhantomData;
use std::sync::Arc;
#[cfg(feature = "instruments")]
use std::time::Instant;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::fft::bit_reversing::{in_place_bit_reverse_permute, reverse_index};
use math::fft::bowers_fft::LayerTwiddles;
use math::fft::errors::FFTError;

use log::info;
use math::field::traits::{IsField, IsSubFieldOf};
use math::spill_safe::SpillSafe;
use math::traits::{AsBytes, ByteConversion};
use math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};

#[cfg(feature = "parallel")]
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

#[cfg(feature = "debug-checks")]
use crate::debug::validate_trace;
use crate::fri;
use crate::lookup::LOGUP_NUM_CHALLENGES;
use crate::proof::stark::{
    CompositionTraceOpening, DeepPolynomialOpenings, MainTraceOpening, PolynomialOpenings,
};
#[cfg(feature = "disk-spill")]
use crate::storage_mode::StorageMode;
use crate::table::Table;
use crate::trace::LDETraceTable;
use crypto::merkle_tree::mmcs::{MatrixTag, Mmcs, MmcsError, StreamingMmcsBuilder};

use super::config::{BatchedMerkleTree, BatchedMerkleTreeBackend, Commitment};
use super::constraints::evaluator::ConstraintEvaluator;
use super::domain::{Domain, DomainConstants};
use super::fri::fri_decommit::FriDecommitment;
use super::grinding;
use super::lookup::BusPublicInputs;
use super::proof::stark::{DeepPolynomialOpening, MultiProof, StarkProof};
use super::trace::TraceTable;
use super::traits::AIR;

/// A triple of (AIR, TraceTable, PublicInputs) for proving.
type AirTracePair<'a, Field, FieldExtension, PI> = (
    &'a dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
    &'a mut TraceTable<Field, FieldExtension>,
    &'a PI,
);

/// A default STARK prover implementing `IsStarkProver`.
pub struct Prover<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
    PI,
> {
    p: PhantomData<(Field, FieldExtension, PI)>,
}

impl<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
    PI,
> IsStarkProver<Field, FieldExtension, PI> for Prover<Field, FieldExtension, PI>
where
    FieldElement<Field>: math::traits::ByteConversion,
    FieldElement<FieldExtension>: math::traits::ByteConversion,
{
}

#[derive(Debug)]
pub enum ProvingError {
    WrongParameter(String),
    EmptyCommitment,
    /// I/O failure while spilling prover state (traces, LDE, Merkle trees) to disk:
    /// out of disk space, fd exhaustion, or mmap failure.
    #[cfg(feature = "disk-spill")]
    DiskSpill(String),
}

/// Per-chunk main MMCS context. Shared across every non-preprocessed
/// table in a chunk: the chunk's MMCS Arc + Arc-cloned LDE columns for
/// chunk-mate non-preprocessed tables in MMCS-spec sort order. The
/// per-query open path uses this to rehash chunk-mate rows on demand
/// (the streaming MMCS dropped the per-chip leaf arrays at build time).
pub(crate) struct ChunkMainMmcsContext<F: IsField>
where
    FieldElement<F>: AsBytes,
{
    /// Chunk-scoped MMCS (built once per chunk in Phase A).
    pub(crate) mmcs: Arc<Mmcs<BatchedMerkleTreeBackend<F>>>,
    /// Arc-cloned LDE columns for the non-preprocessed chunk-mates,
    /// indexed in MMCS spec sort order (parallel to `mmcs.spec()`).
    /// Open path closures look up `lde_columns_in_spec_order[m_idx]` to
    /// rehash the row at the queried local position.
    pub(crate) lde_columns_in_spec_order: Vec<Arc<Vec<Vec<FieldElement<F>>>>>,
}

/// Per-table commitment artifacts for the main trace.
///
/// `Shared` tables borrow a per-chunk MMCS context (Arc) and remember
/// their chunk index so the verifier can look up the matching root +
/// spec in `MultiProof::main_mmcs_roots[chunk_idx]`.
pub(crate) enum MainCommit<F: IsField>
where
    FieldElement<F>: AsBytes,
{
    /// Non-preprocessed table: committed under the chunk's MMCS.
    Shared {
        chunk_ctx: Arc<ChunkMainMmcsContext<F>>,
        chunk_idx: usize,
        tag: MatrixTag,
        /// Padded height (== LDE row count); needed to translate a local
        /// FRI iota into a global MMCS index inside this chunk's MMCS.
        padded_height: usize,
    },
    /// Preprocessed table: two per-table Merkle trees, NOT in any MMCS.
    Preprocessed {
        multiplicities_tree: Arc<BatchedMerkleTree<F>>,
        multiplicities_root: Commitment,
        precomputed_tree: Arc<BatchedMerkleTree<F>>,
        precomputed_root: Commitment,
        num_precomputed_cols: usize,
    },
}

impl<F: IsField> MainCommit<F>
where
    FieldElement<F>: AsBytes,
{
    fn precomputed_root(&self) -> Option<Commitment> {
        match self {
            Self::Shared { .. } => None,
            Self::Preprocessed {
                precomputed_root, ..
            } => Some(*precomputed_root),
        }
    }

    fn main_tree_root(&self) -> Option<Commitment> {
        match self {
            Self::Shared { .. } => None,
            Self::Preprocessed {
                multiplicities_root,
                ..
            } => Some(*multiplicities_root),
        }
    }

    /// Cheap clone. Only bumps Arc refcounts.
    fn share(&self) -> Self {
        match self {
            Self::Shared {
                chunk_ctx,
                chunk_idx,
                tag,
                padded_height,
            } => Self::Shared {
                chunk_ctx: Arc::clone(chunk_ctx),
                chunk_idx: *chunk_idx,
                tag: *tag,
                padded_height: *padded_height,
            },
            Self::Preprocessed {
                multiplicities_tree,
                multiplicities_root,
                precomputed_tree,
                precomputed_root,
                num_precomputed_cols,
            } => Self::Preprocessed {
                multiplicities_tree: Arc::clone(multiplicities_tree),
                multiplicities_root: *multiplicities_root,
                precomputed_tree: Arc::clone(precomputed_tree),
                precomputed_root: *precomputed_root,
                num_precomputed_cols: *num_precomputed_cols,
            },
        }
    }
}

/// Per-table Phase-A output. Non-preprocessed tables contribute their
/// tagged leaf vector to the shared MMCS; preprocessed tables ship two
/// independent per-table Merkle trees that stay out of the MMCS.
enum MainPhaseAOutput<F: IsField>
where
    FieldElement<F>: AsBytes,
{
    Shared {
        tag: MatrixTag,
        leaves: Vec<Commitment>,
        padded_height: usize,
    },
    Preprocessed {
        multiplicities_tree: Arc<BatchedMerkleTree<F>>,
        multiplicities_root: Commitment,
        precomputed_tree: Arc<BatchedMerkleTree<F>>,
        precomputed_root: Commitment,
        num_precomputed_cols: usize,
    },
}

impl<F: IsField> MainPhaseAOutput<F>
where
    FieldElement<F>: AsBytes,
{
    fn precomputed_root(&self) -> Option<Commitment> {
        match self {
            Self::Shared { .. } => None,
            Self::Preprocessed {
                precomputed_root, ..
            } => Some(*precomputed_root),
        }
    }

    fn main_tree_root(&self) -> Option<Commitment> {
        match self {
            Self::Shared { .. } => None,
            Self::Preprocessed {
                multiplicities_root,
                ..
            } => Some(*multiplicities_root),
        }
    }
}

/// Per-chunk aux MMCS context. Sister of [`ChunkMainMmcsContext`] for
/// the aux trace.
pub(crate) struct ChunkAuxMmcsContext<E: IsField>
where
    FieldElement<E>: AsBytes,
{
    pub(crate) mmcs: Arc<Mmcs<BatchedMerkleTreeBackend<E>>>,
    /// Arc-cloned aux LDE columns for chunk-mates with aux, in MMCS
    /// spec sort order.
    pub(crate) lde_columns_in_spec_order: Vec<Arc<Vec<Vec<FieldElement<E>>>>>,
}

/// Per-table aux-trace commitment under a chunk's aux MMCS.
pub(crate) enum AuxCommit<E: IsField>
where
    FieldElement<E>: AsBytes,
{
    Shared {
        chunk_ctx: Arc<ChunkAuxMmcsContext<E>>,
        chunk_idx: usize,
        tag: MatrixTag,
        padded_height: usize,
    },
}

impl<E: IsField> AuxCommit<E>
where
    FieldElement<E>: AsBytes,
{
    fn share(&self) -> Self {
        match self {
            Self::Shared {
                chunk_ctx,
                chunk_idx,
                tag,
                padded_height,
            } => Self::Shared {
                chunk_ctx: Arc::clone(chunk_ctx),
                chunk_idx: *chunk_idx,
                tag: *tag,
                padded_height: *padded_height,
            },
        }
    }
}

/// Per-table aux Phase-C output collected BEFORE the shared aux MMCS is
/// built. `leaves` are aux-tagged Keccak digests over the committed aux-trace
/// LDE rows. Consumed by the single `MmcsBuilder::finalize` call once
/// every aux-bearing table has produced them.
struct AuxPhaseCOutput<E: IsField>
where
    FieldElement<E>: AsBytes,
{
    tag: MatrixTag,
    leaves: Vec<Commitment>,
    _marker: PhantomData<E>,
    padded_height: usize,
}

/// A container for the results of the first round of the STARK Prove protocol.
pub(crate) struct Round1<Field, FieldExtension>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField,
    FieldExtension: IsField,
    FieldElement<Field>: AsBytes,
    FieldElement<FieldExtension>: AsBytes,
{
    /// The table of evaluations over the LDE of the main and auxiliary trace tables.
    pub(crate) lde_trace: LDETraceTable<Field, FieldExtension>,
    /// Commitment to the main trace (shared MMCS handle + per-table tag).
    pub(crate) main: MainCommit<Field>,
    /// Commitment to the auxiliary (RAP) trace, if any.
    pub(crate) aux: Option<AuxCommit<FieldExtension>>,
    /// The challenges of the RAP round.
    pub(crate) rap_challenges: Vec<FieldElement<FieldExtension>>,
    /// Bus interaction public inputs (initial and final aux column values).
    pub(crate) bus_public_inputs: Option<BusPublicInputs<FieldExtension>>,
}

/// Round 1 commitment artifacts — Merkle trees, roots, challenges, and bus inputs.
/// Borrowed (not consumed) when building `Round1` in Phase D.
pub(crate) struct Round1Commitments<Field, FieldExtension>
where
    Field: IsFFTField + IsSubFieldOf<FieldExtension>,
    FieldExtension: IsField,
    FieldElement<Field>: AsBytes,
    FieldElement<FieldExtension>: AsBytes,
{
    main: MainCommit<Field>,
    aux: Option<AuxCommit<FieldExtension>>,
    rap_challenges: Vec<FieldElement<FieldExtension>>,
    bus_public_inputs: Option<BusPublicInputs<FieldExtension>>,
}

/// LDE columns for main (Phase A) and auxiliary (Phase C) traces.
/// Arc-wrapped so per-chunk MMCS contexts can hold cheap clones for the
/// open path while the originating table's `Round1.lde_trace` retains
/// the same data via Arc share (no duplication).
///
/// Memory trade-off: all N tables' LDE columns are live simultaneously
/// between Phase A/C and Phase D (O(N × cols × lde_size)).
struct Lde<Field: IsFFTField, FieldExtension: IsField> {
    main: Arc<Vec<Vec<FieldElement<Field>>>>,
    aux: Arc<Vec<Vec<FieldElement<FieldExtension>>>>,
}

impl<Field, FieldExtension> Round1Commitments<Field, FieldExtension>
where
    Field: IsFFTField + IsSubFieldOf<FieldExtension> + Send + Sync,
    FieldExtension: IsField + Send + Sync,
    FieldElement<Field>: AsBytes,
    FieldElement<FieldExtension>: AsBytes,
{
    /// Build a `Round1` by consuming a `Lde` and borrowing commitment data.
    /// The `share` calls are cheap — only bump Arc refcounts. The LDE
    /// columns are also Arc-shared (with this chunk's MMCS contexts) so
    /// the open path can rehash chunk-mate rows without copying.
    fn build_round1(
        &self,
        lde: Lde<Field, FieldExtension>,
        step_size: usize,
        blowup_factor: usize,
    ) -> Round1<Field, FieldExtension> {
        Round1 {
            lde_trace: LDETraceTable::from_columns_arc(
                lde.main,
                lde.aux,
                step_size,
                blowup_factor,
            ),
            main: self.main.share(),
            aux: self.aux.as_ref().map(AuxCommit::share),
            rap_challenges: self.rap_challenges.clone(),
            bus_public_inputs: self.bus_public_inputs.clone(),
        }
    }
}

/// Pre-computed twiddle factors and coset weights for a given domain size.
///
/// Shared across all columns of the same table, and across all phases (A, C, Rounds 2-4)
/// in sequential proving. This eliminates redundant root-of-unity generation:
/// many redundant `get_twiddles` calls reduced to one pair per distinct domain.
///
/// The `coset_weights` vector stores `[n_inv, n_inv*g, n_inv*g², ..., n_inv*g^{n-1}]`
/// where `g` is the coset offset and `n_inv = 1/n`. These are used in the iFFT+coset-shift
/// step of `expand_columns_to_lde`.
pub(crate) struct LdeTwiddles<F: IsFFTField> {
    inv: LayerTwiddles<F>,
    fwd: LayerTwiddles<F>,
    coset_weights: Vec<FieldElement<F>>,
}

impl<F: IsFFTField> LdeTwiddles<F> {
    /// Construct twiddles and coset weights for a domain of the given size and blowup factor.
    fn new(domain: &Domain<F>) -> Self {
        let domain_size = domain.interpolation_domain_size;
        let lde_size = domain_size * domain.blowup_factor;

        let domain_size_inv = FieldElement::<F>::from(domain_size as u64)
            .inv()
            .expect("domain_size is power of two");
        let offset = &domain.coset_offset;
        let coset_weights = {
            let mut w = Vec::with_capacity(domain_size);
            let mut offset_power = domain_size_inv;
            for _ in 0..domain_size {
                w.push(offset_power.clone());
                offset_power = offset * &offset_power;
            }
            w
        };

        Self {
            inv: LayerTwiddles::<F>::new_inverse(domain_size.trailing_zeros() as u64)
                .expect("valid inverse twiddles"),
            fwd: LayerTwiddles::<F>::new(lde_size.trailing_zeros() as u64)
                .expect("valid forward twiddles"),
            coset_weights,
        }
    }
}

/// Number of tables to process concurrently in `multi_prove`.
/// Default: num_cores / 3 (benchmarked optimal on both M3 Pro and EPYC 9454P).
/// Override with `TABLE_PARALLELISM` env var.
pub fn table_parallelism() -> usize {
    #[cfg(feature = "parallel")]
    {
        std::env::var("TABLE_PARALLELISM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                let cores = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
                (cores / 3).max(1)
            })
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// A container for the results of the second round of the STARK Prove protocol.
/// Per-chunk composition MMCS context.
pub(crate) struct ChunkCompMmcsContext<E: IsField>
where
    FieldElement<E>: AsBytes,
{
    pub(crate) mmcs: Arc<Mmcs<BatchedMerkleTreeBackend<E>>>,
    /// Arc-cloned composition LDE columns for chunk-mates, in MMCS spec
    /// sort order. Used by the per-query open path to rehash composition
    /// row-pair leaves on demand.
    pub(crate) lde_columns_in_spec_order: Vec<Arc<Vec<Vec<FieldElement<E>>>>>,
}

/// Per-table composition-trace commitment under the chunk's composition MMCS.
pub(crate) enum CompCommit<E: IsField>
where
    FieldElement<E>: AsBytes,
{
    Shared {
        chunk_ctx: Arc<ChunkCompMmcsContext<E>>,
        chunk_idx: usize,
        tag: MatrixTag,
        /// Padded height = lde_size / 2 (row-pair leaves).
        padded_height: usize,
    },
}

impl<E: IsField> CompCommit<E>
where
    FieldElement<E>: AsBytes,
{
    fn share(&self) -> Self {
        match self {
            Self::Shared {
                chunk_ctx,
                chunk_idx,
                tag,
                padded_height,
            } => Self::Shared {
                chunk_ctx: Arc::clone(chunk_ctx),
                chunk_idx: *chunk_idx,
                tag: *tag,
                padded_height: *padded_height,
            },
        }
    }
}

/// Per-table Round 2 partial — produced by `round_2a_build_composition_lde`
/// before the chunk composition MMCS is built.
pub(crate) struct R2aResult<E: IsField>
where
    FieldElement<E>: AsBytes,
{
    pub(crate) lde_composition_poly_evaluations: Arc<Vec<Vec<FieldElement<E>>>>,
    pub(crate) composition_leaves: Vec<Commitment>,
    pub(crate) padded_height: usize,
}

pub(crate) struct Round2<F>
where
    F: IsField,
    FieldElement<F>: AsBytes,
{
    /// Evaluations of the composition polynomial parts over the LDE
    /// domain (Arc-shared with the chunk composition MMCS context).
    pub(crate) lde_composition_poly_evaluations: Arc<Vec<Vec<FieldElement<F>>>>,
    /// This table's slot inside the chunk's composition MMCS.
    pub(crate) comp: CompCommit<F>,
}

/// A container for the results of the third round of the STARK Prove protocol.
pub(crate) struct Round3<F: IsField> {
    /// Evaluations of the trace polynomials, main ans auxiliary, at the out-of-domain challenge.
    trace_ood_evaluations: Table<F>,
    /// Evaluations of the composition polynomial parts at the out-of-domain challenge.
    composition_poly_parts_ood_evaluation: Vec<FieldElement<F>>,
}

/// A container for the results of the fourth round of the STARK Prove protocol.
pub(crate) struct Round4<F: IsSubFieldOf<E>, E: IsField> {
    /// The final value resulting from folding the Deep composition polynomial all the way down to a constant value.
    fri_last_value: FieldElement<E>,
    /// The commitments to the fold polynomials of the inner layers of FRI.
    fri_layers_merkle_roots: Vec<Commitment>,
    /// The values and proofs of validity of the evaluations of the trace polynomials and the composition polynomials
    /// parts at the domain values corresponding to the FRI query challenges and their symmetric counterparts.
    deep_poly_openings: DeepPolynomialOpenings<F, E>,
    /// The values and proofs of validity of the evaluations of the fold polynomials of the inner
    /// layers of FRI at the values corresponding to the symmetrics of the FRI query challenges.
    query_list: Vec<FriDecommitment<E>>,
    /// The proof of work nonce.
    nonce: Option<u64>,
}

/// Returns the evaluations of the polynomial `p` over the lde domain defined by the given
/// `blowup_factor`, `domain_size` and `offset`. The number of evaluations returned is `domain_size
/// * blowup_factor`. The domain generator used is the one given by the implementation of `F` as `IsFFTField`.
pub fn evaluate_polynomial_on_lde_domain<F, E>(
    p: &Polynomial<FieldElement<E>>,
    blowup_factor: usize,
    domain_size: usize,
    offset: &FieldElement<F>,
) -> Result<Vec<FieldElement<E>>, FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
{
    let evaluations = Polynomial::evaluate_offset_fft(p, blowup_factor, Some(domain_size), offset)?;
    let step = evaluations.len() / (domain_size * blowup_factor);
    match step {
        1 => Ok(evaluations),
        _ => Ok(evaluations.into_iter().step_by(step).collect()),
    }
}

/// Compute Keccak-256 leaf hashes for `commit_columns_bit_reversed`: one
/// leaf per row, where each row is read at `reverse_index(row_idx)` and the
/// columns are concatenated as big-endian bytes before hashing.
///
/// Returns `Vec<Commitment>` with the same length as `columns[0]`. Exposed
/// (instead of being a closure inside `commit_columns_bit_reversed`) so
/// parity tests in dependent crates can compare against the same code path
/// the prover uses.
pub fn keccak_leaves_bit_reversed<E>(columns: &[Vec<FieldElement<E>>]) -> Vec<Commitment>
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send + ByteConversion,
{
    if columns.is_empty() || columns[0].is_empty() {
        return Vec::new();
    }

    let num_rows = columns[0].len();
    let num_cols = columns.len();
    let byte_len = <FieldElement<E> as ByteConversion>::BYTE_LEN;

    debug_assert!(
        num_rows.is_power_of_two(),
        "num_rows must be a power of two for reverse_index"
    );

    let total_bytes = num_cols * byte_len;

    let hash_leaf = |buf: &mut [u8], row_idx: usize| -> Commitment {
        let br_idx = reverse_index(row_idx, num_rows as u64);
        for col_idx in 0..num_cols {
            columns[col_idx][br_idx]
                .write_bytes_be(&mut buf[col_idx * byte_len..(col_idx + 1) * byte_len]);
        }
        BatchedMerkleTreeBackend::<E>::hash_bytes(buf)
    };

    #[cfg(feature = "parallel")]
    let iter = (0..num_rows).into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = 0..num_rows;

    // Per-thread buffer reuse: map_init allocates one buffer per Rayon thread,
    // eliminating millions of small heap allocations under parallel contention.
    #[cfg(feature = "parallel")]
    let result: Vec<Commitment> = iter
        .map_init(|| vec![0u8; total_bytes], |buf, i| hash_leaf(buf, i))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let result: Vec<Commitment> = {
        let mut buf = vec![0u8; total_bytes];
        iter.map(|i| hash_leaf(&mut buf, i)).collect()
    };

    result
}

fn map_mmcs_err(e: MmcsError) -> ProvingError {
    ProvingError::WrongParameter(format!("MMCS: {e:?}"))
}

/// Rehash a single main-trace LDE row to its tagged leaf digest. Used by
/// the per-chunk open path: when `Mmcs::open_with_leaves` walks the chunk
/// MMCS spec to gather matrix_leaves at a queried position, this helper
/// recomputes each chunk-mate's leaf on demand from the chunk-shared LDE
/// columns. Mirrors what the verifier computes via `hash_tagged_row`.
pub fn rehash_main_chip_leaf<F>(
    tag: MatrixTag,
    columns: &Arc<Vec<Vec<FieldElement<F>>>>,
    local_idx: usize,
) -> Commitment
where
    F: IsField,
    FieldElement<F>: AsBytes + ByteConversion,
{
    let num_rows = columns
        .first()
        .map(|c| c.len())
        .expect("non-empty LDE columns");
    let br_idx = reverse_index(local_idx, num_rows as u64);
    let byte_len = <FieldElement<F> as ByteConversion>::BYTE_LEN;
    let mut buf = vec![0u8; columns.len() * byte_len];
    for (col_idx, col) in columns.iter().enumerate() {
        col[br_idx].write_bytes_be(&mut buf[col_idx * byte_len..(col_idx + 1) * byte_len]);
    }
    crate::mmcs_leaf::hash_tagged_row_bytes(tag, &buf)
}

/// Aux-trace counterpart of [`rehash_main_chip_leaf`] using the AUX
/// domain separator so aux/main leaves cannot collide.
pub fn rehash_aux_chip_leaf<E>(
    tag: MatrixTag,
    columns: &Arc<Vec<Vec<FieldElement<E>>>>,
    local_idx: usize,
) -> Commitment
where
    E: IsField,
    FieldElement<E>: AsBytes + ByteConversion,
{
    let num_rows = columns
        .first()
        .map(|c| c.len())
        .expect("non-empty aux LDE columns");
    let br_idx = reverse_index(local_idx, num_rows as u64);
    let byte_len = <FieldElement<E> as ByteConversion>::BYTE_LEN;
    let mut buf = vec![0u8; columns.len() * byte_len];
    for (col_idx, col) in columns.iter().enumerate() {
        col[br_idx].write_bytes_be(&mut buf[col_idx * byte_len..(col_idx + 1) * byte_len]);
    }
    crate::mmcs_leaf::hash_tagged_row_bytes_aux(tag, &buf)
}

/// Build a CHUNK-scoped main MMCS via [`StreamingMmcsBuilder`]. Consumes
/// the Shared phase-A outputs (drops their per-chip leaves once folded),
/// returns the chunk root + spec + an `Arc<ChunkMainMmcsContext>` that
/// every Shared table in the chunk borrows.
///
/// Returns `None` for the root/context when the chunk has no Shared
/// tables (entire chunk is preprocessed).
#[allow(clippy::type_complexity)]
fn build_chunk_main_mmcs<F>(
    shared_outputs: Vec<(MatrixTag, Vec<Commitment>, usize)>,
    chunk_lde_for_shared: Vec<(MatrixTag, Arc<Vec<Vec<FieldElement<F>>>>)>,
) -> Result<
    (
        Option<Commitment>,
        Vec<(MatrixTag, usize)>,
        Option<Arc<ChunkMainMmcsContext<F>>>,
    ),
    ProvingError,
>
where
    F: IsField + Send + Sync,
    FieldElement<F>: AsBytes + Send + Sync,
{
    if shared_outputs.is_empty() {
        return Ok((None, Vec::new(), None));
    }
    debug_assert_eq!(shared_outputs.len(), chunk_lde_for_shared.len());

    // Sort both vectors into MMCS spec order: height desc, tag asc.
    let mut shared_outputs = shared_outputs;
    shared_outputs.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
    let lde_by_tag: std::collections::BTreeMap<MatrixTag, Arc<Vec<Vec<FieldElement<F>>>>> =
        chunk_lde_for_shared.into_iter().collect();

    let mut builder: StreamingMmcsBuilder<BatchedMerkleTreeBackend<F>> =
        StreamingMmcsBuilder::new();
    let mut lde_columns_in_spec_order: Vec<Arc<Vec<Vec<FieldElement<F>>>>> =
        Vec::with_capacity(shared_outputs.len());
    for (tag, leaves, _padded_height) in shared_outputs {
        let lde = lde_by_tag
            .get(&tag)
            .ok_or_else(|| {
                ProvingError::WrongParameter(format!(
                    "missing chunk LDE for tag {tag:?} during chunk MMCS build"
                ))
            })?
            .clone();
        lde_columns_in_spec_order.push(lde);
        builder.add_matrix(tag, leaves).map_err(map_mmcs_err)?;
    }
    let mmcs = builder.finalize().map_err(map_mmcs_err)?;
    let root = *mmcs.root();
    let spec = mmcs.spec();
    let ctx = Arc::new(ChunkMainMmcsContext {
        mmcs: Arc::new(mmcs),
        lde_columns_in_spec_order,
    });
    Ok((Some(root), spec, Some(ctx)))
}

/// Tagged per-row leaf digest for the AUX-trace MMCS. Mirror of
/// [`compute_tagged_leaves_bit_reversed`] but uses the aux domain
/// separator so aux/main leaves cannot collide.
pub fn compute_tagged_leaves_bit_reversed_aux<E>(
    columns: &[Vec<FieldElement<E>>],
    tag: MatrixTag,
) -> Vec<Commitment>
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send + ByteConversion,
{
    if columns.is_empty() || columns[0].is_empty() {
        return Vec::new();
    }
    let num_rows = columns[0].len();
    let num_cols = columns.len();
    let byte_len = <FieldElement<E> as ByteConversion>::BYTE_LEN;
    debug_assert!(num_rows.is_power_of_two());
    let total_bytes = num_cols * byte_len;
    let hash_leaf =
        |buf: &mut [u8], row_idx: usize| -> Commitment {
            let br_idx = reverse_index(row_idx, num_rows as u64);
            for (col_idx, col) in columns.iter().enumerate() {
                col[br_idx]
                    .write_bytes_be(&mut buf[col_idx * byte_len..(col_idx + 1) * byte_len]);
            }
            crate::mmcs_leaf::hash_tagged_row_bytes_aux(tag, buf)
        };
    #[cfg(feature = "parallel")]
    {
        (0..num_rows)
            .into_par_iter()
            .map_init(|| vec![0u8; total_bytes], |buf, i| hash_leaf(buf, i))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut buf = vec![0u8; total_bytes];
        (0..num_rows).map(|i| hash_leaf(&mut buf, i)).collect()
    }
}

/// Build a CHUNK-scoped aux MMCS via [`StreamingMmcsBuilder`]. Sister of
/// [`build_chunk_main_mmcs`] for the aux trace. Returns `None` for root
/// and context when no chunk-mate has an aux trace.
#[allow(clippy::type_complexity)]
fn build_chunk_aux_mmcs<E>(
    aux_outputs: Vec<(MatrixTag, Vec<Commitment>, usize)>,
    chunk_aux_lde_for_shared: Vec<(MatrixTag, Arc<Vec<Vec<FieldElement<E>>>>)>,
) -> Result<
    (
        Option<Commitment>,
        Vec<(MatrixTag, usize)>,
        Option<Arc<ChunkAuxMmcsContext<E>>>,
    ),
    ProvingError,
>
where
    E: IsField + Send + Sync,
    FieldElement<E>: AsBytes + Send + Sync,
{
    if aux_outputs.is_empty() {
        return Ok((None, Vec::new(), None));
    }
    debug_assert_eq!(aux_outputs.len(), chunk_aux_lde_for_shared.len());

    let mut aux_outputs = aux_outputs;
    aux_outputs.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
    let lde_by_tag: std::collections::BTreeMap<MatrixTag, Arc<Vec<Vec<FieldElement<E>>>>> =
        chunk_aux_lde_for_shared.into_iter().collect();

    let mut builder: StreamingMmcsBuilder<BatchedMerkleTreeBackend<E>> =
        StreamingMmcsBuilder::new();
    let mut lde_columns_in_spec_order: Vec<Arc<Vec<Vec<FieldElement<E>>>>> =
        Vec::with_capacity(aux_outputs.len());
    for (tag, leaves, _padded_height) in aux_outputs {
        let lde = lde_by_tag
            .get(&tag)
            .ok_or_else(|| {
                ProvingError::WrongParameter(format!(
                    "missing chunk aux LDE for tag {tag:?} during chunk MMCS build"
                ))
            })?
            .clone();
        lde_columns_in_spec_order.push(lde);
        builder.add_matrix(tag, leaves).map_err(map_mmcs_err)?;
    }
    let mmcs = builder.finalize().map_err(map_mmcs_err)?;
    let root = *mmcs.root();
    let spec = mmcs.spec();
    let ctx = Arc::new(ChunkAuxMmcsContext {
        mmcs: Arc::new(mmcs),
        lde_columns_in_spec_order,
    });
    Ok((Some(root), spec, Some(ctx)))
}

/// Tagged per-row-PAIR leaf digest for the COMPOSITION-trace MMCS.
pub fn compute_tagged_leaves_row_pair_bit_reversed_composition<E>(
    parts: &[Vec<FieldElement<E>>],
    tag: MatrixTag,
) -> Vec<Commitment>
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send + ByteConversion,
{
    let num_parts = parts.len();
    if num_parts == 0 {
        return Vec::new();
    }
    let num_rows = parts[0].len();
    if num_rows == 0 {
        return Vec::new();
    }
    let num_leaves = num_rows / 2;
    debug_assert!(num_rows.is_power_of_two());
    let byte_len = <FieldElement<E> as ByteConversion>::BYTE_LEN;
    let total_bytes = 2 * num_parts * byte_len;
    let hash_leaf_pair = |buf: &mut [u8], leaf_idx: usize| -> Commitment {
        let br_0 = reverse_index(2 * leaf_idx, num_rows as u64);
        let br_1 = reverse_index(2 * leaf_idx + 1, num_rows as u64);
        let mut offset = 0;
        for part in parts.iter() {
            part[br_0].write_bytes_be(&mut buf[offset..offset + byte_len]);
            offset += byte_len;
        }
        for part in parts.iter() {
            part[br_1].write_bytes_be(&mut buf[offset..offset + byte_len]);
            offset += byte_len;
        }
        crate::mmcs_leaf::hash_tagged_row_pair_bytes_composition(tag, buf)
    };
    #[cfg(feature = "parallel")]
    {
        (0..num_leaves)
            .into_par_iter()
            .map_init(|| vec![0u8; total_bytes], |buf, i| hash_leaf_pair(buf, i))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut buf = vec![0u8; total_bytes];
        (0..num_leaves).map(|i| hash_leaf_pair(&mut buf, i)).collect()
    }
}

/// Build a CHUNK-scoped composition MMCS via StreamingMmcsBuilder.
#[allow(clippy::type_complexity)]
fn build_chunk_comp_mmcs<E>(
    comp_outputs: Vec<(MatrixTag, Vec<Commitment>, usize)>,
    chunk_comp_lde: Vec<(MatrixTag, Arc<Vec<Vec<FieldElement<E>>>>)>,
) -> Result<
    (
        Option<Commitment>,
        Vec<(MatrixTag, usize)>,
        Option<Arc<ChunkCompMmcsContext<E>>>,
    ),
    ProvingError,
>
where
    E: IsField + Send + Sync,
    FieldElement<E>: AsBytes + Send + Sync,
{
    if comp_outputs.is_empty() {
        return Ok((None, Vec::new(), None));
    }
    debug_assert_eq!(comp_outputs.len(), chunk_comp_lde.len());
    let mut comp_outputs = comp_outputs;
    comp_outputs.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
    let lde_by_tag: std::collections::BTreeMap<MatrixTag, Arc<Vec<Vec<FieldElement<E>>>>> =
        chunk_comp_lde.into_iter().collect();
    let mut builder: StreamingMmcsBuilder<BatchedMerkleTreeBackend<E>> =
        StreamingMmcsBuilder::new();
    let mut lde_columns_in_spec_order: Vec<Arc<Vec<Vec<FieldElement<E>>>>> =
        Vec::with_capacity(comp_outputs.len());
    for (tag, leaves, _padded_height) in comp_outputs {
        let lde = lde_by_tag
            .get(&tag)
            .ok_or_else(|| {
                ProvingError::WrongParameter(format!(
                    "missing chunk composition LDE for tag {tag:?}"
                ))
            })?
            .clone();
        lde_columns_in_spec_order.push(lde);
        builder.add_matrix(tag, leaves).map_err(map_mmcs_err)?;
    }
    let mmcs = builder.finalize().map_err(map_mmcs_err)?;
    let root = *mmcs.root();
    let spec = mmcs.spec();
    let ctx = Arc::new(ChunkCompMmcsContext {
        mmcs: Arc::new(mmcs),
        lde_columns_in_spec_order,
    });
    Ok((Some(root), spec, Some(ctx)))
}

/// Rehash a composition-trace row-PAIR leaf for the open path.
pub fn rehash_comp_chip_leaf<E>(
    tag: MatrixTag,
    parts: &Arc<Vec<Vec<FieldElement<E>>>>,
    local_idx: usize,
) -> Commitment
where
    E: IsField,
    FieldElement<E>: AsBytes + ByteConversion,
{
    let num_rows = parts
        .first()
        .map(|c| c.len())
        .expect("composition LDE columns non-empty by construction");
    let num_parts = parts.len();
    let byte_len = <FieldElement<E> as ByteConversion>::BYTE_LEN;
    let br_0 = reverse_index(2 * local_idx, num_rows as u64);
    let br_1 = reverse_index(2 * local_idx + 1, num_rows as u64);
    let mut buf = vec![0u8; 2 * num_parts * byte_len];
    let mut offset = 0;
    for part in parts.iter() {
        part[br_0].write_bytes_be(&mut buf[offset..offset + byte_len]);
        offset += byte_len;
    }
    for part in parts.iter() {
        part[br_1].write_bytes_be(&mut buf[offset..offset + byte_len]);
        offset += byte_len;
    }
    crate::mmcs_leaf::hash_tagged_row_pair_bytes_composition(tag, &buf)
}

/// Tagged per-row leaf digest for the main-trace MMCS.
pub fn compute_tagged_leaves_bit_reversed<E>(
    columns: &[Vec<FieldElement<E>>],
    tag: MatrixTag,
) -> Vec<Commitment>
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send + ByteConversion,
{
    if columns.is_empty() || columns[0].is_empty() {
        return Vec::new();
    }
    let num_rows = columns[0].len();
    let num_cols = columns.len();
    let byte_len = <FieldElement<E> as ByteConversion>::BYTE_LEN;
    debug_assert!(num_rows.is_power_of_two());
    let total_bytes = num_cols * byte_len;
    let hash_leaf =
        |buf: &mut [u8], row_idx: usize| -> Commitment {
            let br_idx = reverse_index(row_idx, num_rows as u64);
            for (col_idx, col) in columns.iter().enumerate() {
                col[br_idx]
                    .write_bytes_be(&mut buf[col_idx * byte_len..(col_idx + 1) * byte_len]);
            }
            crate::mmcs_leaf::hash_tagged_row_bytes(tag, buf)
        };
    #[cfg(feature = "parallel")]
    {
        (0..num_rows)
            .into_par_iter()
            .map_init(|| vec![0u8; total_bytes], |buf, i| hash_leaf(buf, i))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut buf = vec![0u8; total_bytes];
        (0..num_rows).map(|i| hash_leaf(&mut buf, i)).collect()
    }
}

/// Compute Keccak-256 leaf hashes for `commit_composition_polynomial`: one
/// leaf per row-pair, where leaf `i` hashes the BE concatenation of
/// `parts[..][br_0] ++ parts[..][br_1]` with
/// `br_k = reverse_index(2*i + k, num_rows)`.
///
/// Returns `Vec<Commitment>` of length `parts[0].len() / 2`.
pub fn keccak_leaves_row_pair_bit_reversed<E>(parts: &[Vec<FieldElement<E>>]) -> Vec<Commitment>
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send + ByteConversion,
{
    let num_parts = parts.len();
    if num_parts == 0 {
        return Vec::new();
    }
    let num_rows = parts[0].len();
    if num_rows == 0 {
        return Vec::new();
    }

    let num_leaves = num_rows / 2;
    debug_assert!(
        num_rows.is_power_of_two(),
        "num_rows must be a power of two for reverse_index"
    );

    let byte_len = <FieldElement<E> as ByteConversion>::BYTE_LEN;

    let total_bytes = 2 * num_parts * byte_len;

    let hash_leaf_pair = |buf: &mut [u8], leaf_idx: usize| -> Commitment {
        let br_0 = reverse_index(2 * leaf_idx, num_rows as u64);
        let br_1 = reverse_index(2 * leaf_idx + 1, num_rows as u64);
        let mut offset = 0;
        for part in parts.iter() {
            part[br_0].write_bytes_be(&mut buf[offset..offset + byte_len]);
            offset += byte_len;
        }
        for part in parts.iter() {
            part[br_1].write_bytes_be(&mut buf[offset..offset + byte_len]);
            offset += byte_len;
        }
        BatchedMerkleTreeBackend::<E>::hash_bytes(buf)
    };

    #[cfg(feature = "parallel")]
    let iter = (0..num_leaves).into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = 0..num_leaves;

    #[cfg(feature = "parallel")]
    let result: Vec<Commitment> = iter
        .map_init(|| vec![0u8; total_bytes], |buf, i| hash_leaf_pair(buf, i))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let result: Vec<Commitment> = {
        let mut buf = vec![0u8; total_bytes];
        iter.map(|i| hash_leaf_pair(&mut buf, i)).collect()
    };

    result
}

/// The functionality of a STARK prover providing methods to run the STARK Prove protocol
/// https://lambdaclass.github.io/lambdaworks/starks/protocol.html
/// The default implementation is complete and is compatible with Stone prover
/// https://github.com/starkware-libs/stone-prover
///
/// Note: many default-method signatures expose `pub(crate)` round-state types
/// (`Round1`, `Round2`, `Round3`, `Round4`, `LdeTwiddles`). These are internal
/// helpers — only `prove`, `multi_prove` are meant for callers. The
/// `private_interfaces` allow is removed once these helpers move off the trait.
#[allow(private_interfaces)]
pub trait IsStarkProver<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
    PI,
> where
    FieldElement<Field>: math::traits::ByteConversion,
    FieldElement<FieldExtension>: math::traits::ByteConversion,
{
    /// Builds a Merkle tree commitment from column-major LDE evaluations with
    /// bit-reverse permutation, without cloning the full evaluation matrix.
    ///
    /// For each row index `i`, we hash `col_0[br(i)] || col_1[br(i)] || ...`
    /// where `br(i)` is the bit-reversal of `i`. This produces the same Merkle
    /// tree as the old clone + bit-reverse + columns2rows + batch_commit flow,
    /// but avoids allocating the cloned and transposed matrices entirely.
    fn commit_columns_bit_reversed<E>(
        columns: &[Vec<FieldElement<E>>],
    ) -> Option<(BatchedMerkleTree<E>, Commitment)>
    where
        FieldElement<E>: AsBytes + Sync + Send + math::traits::ByteConversion,
        E: IsField,
    {
        if columns.is_empty() || columns[0].is_empty() {
            return None;
        }
        let hashed_leaves = keccak_leaves_bit_reversed(columns);
        let tree = BatchedMerkleTree::<E>::build_from_hashed_leaves(hashed_leaves)?;
        let root = tree.root;
        Some((tree, root))
    }

    /// Compute the LDE commitment for a subset of columns from a trace (for testing).
    ///
    /// This helper computes the same commitment the prover generates internally,
    /// useful for setting up soundness test scenarios. Only available under
    /// `cfg(test)` (in-crate) or with the `test-utils` Cargo feature
    /// (cross-crate tests).
    #[cfg(any(test, feature = "test-utils"))]
    fn compute_precomputed_commitment_for_testing(
        trace: &TraceTable<Field, FieldExtension>,
        air: &impl AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        num_precomputed_cols: usize,
    ) -> Option<Commitment>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let domain = Domain::new(air, trace.num_rows());
        let columns = trace.columns_main();
        let precomputed: Vec<_> = columns.into_iter().take(num_precomputed_cols).collect();
        let twiddles = LdeTwiddles::new(&domain);
        let evals =
            Self::compute_lde_from_columns_cached::<Field>(&precomputed, &domain, &twiddles);
        let (_, commitment) = Self::commit_columns_bit_reversed(&evals)?;
        Some(commitment)
    }

    /// Compute LDE evaluations with pre-computed twiddle factors and coset weights.
    ///
    /// Accepts shared [`LdeTwiddles`] to avoid redundant twiddle generation and weight
    /// computation across phases (A, C, Rounds 2-4).
    fn compute_lde_from_columns_cached<E>(
        columns: &[Vec<FieldElement<E>>],
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
    ) -> Vec<Vec<FieldElement<E>>>
    where
        E: IsSubFieldOf<FieldExtension> + Send + Sync,
        Field: IsSubFieldOf<E>,
        FieldElement<E>: Send + Sync,
    {
        if columns.is_empty() {
            return Vec::new();
        }

        #[cfg(not(feature = "parallel"))]
        let columns_iter = columns.iter();
        #[cfg(feature = "parallel")]
        let columns_iter = columns.par_iter();

        columns_iter
            .map(|col| {
                Polynomial::coset_lde_full::<Field>(
                    col,
                    domain.blowup_factor,
                    &twiddles.coset_weights,
                    &twiddles.inv,
                    &twiddles.fwd,
                )
            })
            .collect::<Result<Vec<Vec<FieldElement<E>>>, _>>()
            .expect("coset LDE computation")
    }

    /// Expand each column in-place from N evaluations to N×blowup LDE evaluations.
    ///
    /// Performs iFFT + coset shift + FFT in place. Coset weights are pre-cached in
    /// `LdeTwiddles` to avoid recomputation across phases.
    fn expand_columns_to_lde<E>(
        columns: &mut [Vec<FieldElement<E>>],
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
    ) where
        Field: IsSubFieldOf<E>,
        E: IsSubFieldOf<FieldExtension> + IsField + Send + Sync,
        FieldElement<E>: Send + Sync,
    {
        if columns.is_empty() {
            return;
        }

        #[cfg(feature = "parallel")]
        let iter = columns.par_iter_mut();
        #[cfg(not(feature = "parallel"))]
        let iter = columns.iter_mut();
        iter.for_each(|buf| {
            Polynomial::coset_lde_full_expand::<Field>(
                buf,
                domain.blowup_factor,
                &twiddles.coset_weights,
                &twiddles.inv,
                &twiddles.fwd,
            )
            .expect("coset LDE expansion");
        });
    }

    /// Compute the main-trace LDE and the per-table inputs needed by the
    /// shared MMCS build. Returns a `MainPhaseAOutput` (tagged leaves + the
    /// optional precomputed-columns Merkle tree) together with the owned
    /// LDE columns consumed later in Phase D.
    ///
    /// `tag`: the table's MatrixTag, fed into every leaf hash so the MMCS
    /// can authenticate (matrix, row) pairs uniquely.
    /// `precomputed`: if present, the leading `num_cols` columns are
    /// committed as a separate Merkle tree (the precomputed split) and the
    /// root is checked against the AIR-hardcoded commitment. The remaining
    /// columns feed the MMCS leaves. If absent, every column feeds the MMCS.
    #[allow(clippy::type_complexity)]
    fn commit_main_trace(
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
        tag: MatrixTag,
        precomputed: Option<(Commitment, usize)>,
        #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
    ) -> Result<(MainPhaseAOutput<Field>, Vec<Vec<FieldElement<Field>>>), ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let lde_size = domain.interpolation_domain_size * domain.blowup_factor;
        let mut columns = trace.extract_columns_main(lde_size);
        #[cfg(feature = "disk-spill")]
        if storage_mode == StorageMode::Disk {
            trace.main_table.advise_drop_cache();
        }
        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();
        Self::expand_columns_to_lde::<Field>(&mut columns, domain, twiddles);
        #[cfg(feature = "instruments")]
        let main_lde_dur = t_sub.elapsed();

        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();

        let output = match precomputed {
            None => {
                let leaves = compute_tagged_leaves_bit_reversed::<Field>(&columns, tag);
                if leaves.is_empty() {
                    return Err(ProvingError::EmptyCommitment);
                }
                let padded_height = leaves.len();
                MainPhaseAOutput::Shared {
                    tag,
                    leaves,
                    padded_height,
                }
            }
            Some((expected_precomputed_root, num_cols)) => {
                #[allow(unused_mut)]
                let (mut precomputed_tree, precomputed_root) =
                    Self::commit_columns_bit_reversed(&columns[..num_cols])
                        .ok_or(ProvingError::EmptyCommitment)?;
                debug_assert_eq!(
                    precomputed_root, expected_precomputed_root,
                    "Prover precomputed commitment must match the AIR-pinned value"
                );
                #[cfg(feature = "disk-spill")]
                if storage_mode == StorageMode::Disk {
                    precomputed_tree.spill_nodes_to_disk().map_err(|e| {
                        ProvingError::DiskSpill(format!("precomputed Merkle tree: {e}"))
                    })?;
                }
                #[allow(unused_mut)]
                let (mut multiplicities_tree, multiplicities_root) =
                    Self::commit_columns_bit_reversed(&columns[num_cols..])
                        .ok_or(ProvingError::EmptyCommitment)?;
                #[cfg(feature = "disk-spill")]
                if storage_mode == StorageMode::Disk {
                    multiplicities_tree.spill_nodes_to_disk().map_err(|e| {
                        ProvingError::DiskSpill(format!("multiplicities Merkle tree: {e}"))
                    })?;
                }
                MainPhaseAOutput::Preprocessed {
                    multiplicities_tree: Arc::new(multiplicities_tree),
                    multiplicities_root,
                    precomputed_tree: Arc::new(precomputed_tree),
                    precomputed_root,
                    num_precomputed_cols: num_cols,
                }
            }
        };

        #[cfg(feature = "instruments")]
        crate::instruments::accum_r1_main(main_lde_dur, t_sub.elapsed());

        Ok((output, columns))
    }

    /// Recompute Round1 from the trace, reusing the Merkle trees stored in commitments.
    ///
    /// Only used by `run_debug_checks` — Phase D consumes the cached LDE
    /// directly and does not go through this path.
    #[cfg(feature = "debug-checks")]
    fn reconstruct_round1(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        commitment: &Round1Commitments<Field, FieldExtension>,
        twiddles: &LdeTwiddles<Field>,
    ) -> Result<Round1<Field, FieldExtension>, ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let lde_size = domain.interpolation_domain_size * domain.blowup_factor;
        let mut main = trace.extract_columns_main(lde_size);
        Self::expand_columns_to_lde::<Field>(&mut main, domain, twiddles);

        let aux = if air.has_aux_trace() {
            let mut aux = trace.extract_columns_aux(lde_size);
            Self::expand_columns_to_lde::<FieldExtension>(&mut aux, domain, twiddles);
            aux
        } else {
            Vec::new()
        };

        Ok(commitment.build_round1(
            Lde {
                main: Arc::new(main),
                aux: Arc::new(aux),
            },
            air.step_size(),
            domain.blowup_factor,
        ))
    }

    /// Reconstruct Round1 for every table, print the bus balance report, and
    /// validate each trace. Called once after Phase C commits.
    #[cfg(feature = "debug-checks")]
    fn run_debug_checks(
        air_trace_pairs: &[AirTracePair<'_, Field, FieldExtension, PI>],
        commitments: &[Round1Commitments<Field, FieldExtension>],
        domains: &[Arc<Domain<Field>>],
        twiddle_caches: &[Arc<LdeTwiddles<Field>>],
    ) where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        PI: Send + Sync + Clone,
    {
        let mut temp_results: Vec<Round1<Field, FieldExtension>> =
            Vec::with_capacity(air_trace_pairs.len());
        for (((air, trace, _), commitment), (domain, twiddles)) in air_trace_pairs
            .iter()
            .zip(commitments.iter())
            .zip(domains.iter().zip(twiddle_caches.iter()))
        {
            let result = Self::reconstruct_round1(*air, *trace, domain, commitment, twiddles)
                .expect("reconstruct_round1 failed in debug-checks");
            temp_results.push(result);
        }

        let all_bus_public_inputs: Vec<Option<BusPublicInputs<FieldExtension>>> = temp_results
            .iter()
            .map(|r| r.bus_public_inputs.clone())
            .collect();
        print_bus_balance_report(&all_bus_public_inputs);

        for (((air, trace, pub_inputs), round_1_result), domain) in air_trace_pairs
            .iter()
            .zip(temp_results.iter())
            .zip(domains.iter())
        {
            validate_trace(
                *air,
                *pub_inputs,
                *trace,
                domain,
                &round_1_result.rap_challenges,
                round_1_result.bus_public_inputs.as_ref(),
            );
        }
    }

    /// Returns the Merkle tree and the commitment to the evaluations of the parts of the
    /// composition polynomial.
    fn commit_composition_polynomial(
        lde_composition_poly_parts_evaluations: &[Vec<FieldElement<FieldExtension>>],
    ) -> Option<(BatchedMerkleTree<FieldExtension>, Commitment)>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send + math::traits::ByteConversion,
    {
        let num_parts = lde_composition_poly_parts_evaluations.len();
        if num_parts == 0 {
            return None;
        }
        let num_rows = lde_composition_poly_parts_evaluations[0].len();
        if num_rows == 0 {
            return None;
        }
        let hashed_leaves =
            keccak_leaves_row_pair_bit_reversed(lde_composition_poly_parts_evaluations);
        let tree = BatchedMerkleTree::<FieldExtension>::build_from_hashed_leaves(hashed_leaves)?;
        let root = tree.root;
        Some((tree, root))
    }

    /// Algebraically decompose H(x) = H₀(x²) + x·H₁(x²) on the LDE coset, then
    /// extend each half to the full LDE domain. This replaces the expensive
    /// iFFT(2N) + break_in_parts + FFT(2N)×2 pipeline with:
    ///   O(N) pointwise ops + iFFT(N)×2 + FFT(2N)×2
    ///
    /// The identity used:
    ///   H₀(x²) = (H(x) + H(-x)) / 2
    ///   H₁(x²) = (H(x) - H(-x)) / (2x)
    ///
    /// On the LDE coset {g·ω^i | i=0..2N-1}, we have -g·ω^i = g·ω^{i+N}
    /// since ω^N = -1 for a 2N-th root of unity ω.
    fn decompose_and_extend_d2(
        constraint_evaluations: &[FieldElement<FieldExtension>],
        domain: &Domain<Field>,
    ) -> Vec<Vec<FieldElement<FieldExtension>>>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let two_n = constraint_evaluations.len();
        let n = two_n / 2;
        debug_assert_eq!(two_n, n * 2);

        // Step 1: Compute 1/(2·g·ω^i) for i=0..N-1 via batch inversion.
        // The LDE coset points are g·ω^i = domain.lde_roots_of_unity_coset[i].
        // Compute entirely in base field — mixed F×E multiplication when used with extension values.
        let two_base = FieldElement::<Field>::from(2u64);
        let mut inv_2x: Vec<FieldElement<Field>> = (0..n)
            .map(|i| &two_base * &domain.lde_roots_of_unity_coset[i])
            .collect();
        FieldElement::inplace_batch_inverse(&mut inv_2x).expect("Coset points are non-zero");

        // Step 2: Pointwise decomposition.
        // H₀((g·ω^i)²) = (evals[i] + evals[i+N]) / 2
        // H₁((g·ω^i)²) = (evals[i] - evals[i+N]) / (2·g·ω^i)
        let two_inv = two_base.inv().expect("2 is non-zero in the field");
        let (h0_evals, h1_evals) = crate::par::map_unzip(n, |i| {
            let sum = &constraint_evaluations[i] + &constraint_evaluations[i + n];
            let diff = &constraint_evaluations[i] - &constraint_evaluations[i + n];
            // F × E → E (base field scalar on left for mixed multiplication)
            (&two_inv * &sum, &inv_2x[i] * &diff)
        });

        // Step 3: Extend each part from N evals on g²-coset to 2N evals on g-coset.
        // The squared coset offset is g² (= coset_offset²).
        let coset_offset_squared = &domain.coset_offset * &domain.coset_offset;
        let (lde_h0, lde_h1) = crate::par::join(
            || Self::extend_half_to_lde(&h0_evals, &coset_offset_squared, domain),
            || Self::extend_half_to_lde(&h1_evals, &coset_offset_squared, domain),
        );
        vec![lde_h0, lde_h1]
    }

    /// Given N evaluations of a degree-<N polynomial on the g²-coset,
    /// extend to 2N evaluations on the g-coset (the full LDE domain).
    /// This is: iFFT(N, offset=g²) → coefficients → FFT(2N, offset=g).
    fn extend_half_to_lde(
        half_evals: &[FieldElement<FieldExtension>],
        squared_offset: &FieldElement<Field>,
        domain: &Domain<Field>,
    ) -> Vec<FieldElement<FieldExtension>>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        // iFFT on the N-point squared coset to get coefficients
        let poly = Polynomial::interpolate_offset_fft(half_evals, squared_offset)
            .expect("iFFT should succeed");
        // Evaluate on the full LDE domain (2N points on the g-coset)
        evaluate_polynomial_on_lde_domain(
            &poly,
            domain.blowup_factor,
            domain.interpolation_domain_size,
            &domain.coset_offset,
        )
        .expect("LDE evaluation should succeed")
    }

    /// Round 2 phase A: build the composition LDE parts + tagged leaves
    /// for the chunk MMCS, WITHOUT committing yet. The chunk MMCS is
    /// built externally once every chunk-mate has returned their
    /// [`R2aResult`]; only then does the resulting chunk root get
    /// absorbed back into each fork and R3 sampling proceeds.
    fn round_2a_build_composition_lde(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        pub_inputs: &PI,
        domain: &Domain<Field>,
        round_1_result: &Round1<Field, FieldExtension>,
        transition_coefficients: &[FieldElement<FieldExtension>],
        boundary_coefficients: &[FieldElement<FieldExtension>],
        tag: MatrixTag,
    ) -> Result<R2aResult<FieldExtension>, ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        // Compute the evaluations of the composition polynomial on the LDE domain.
        let trace_length = domain.interpolation_domain_size;
        let evaluator = ConstraintEvaluator::new(
            air,
            pub_inputs,
            &round_1_result.rap_challenges,
            round_1_result.bus_public_inputs.as_ref(),
            trace_length,
        );
        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();
        let constraint_evaluations = evaluator.evaluate(
            air,
            &round_1_result.lde_trace,
            domain,
            transition_coefficients,
            boundary_coefficients,
            &round_1_result.rap_challenges,
        );
        #[cfg(feature = "instruments")]
        let constraints_dur = t_sub.elapsed();

        let number_of_parts = air.composition_poly_degree_bound(trace_length) / trace_length;

        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();
        let lde_composition_poly_parts_evaluations = if number_of_parts == 2 {
            // Direct quotient decomposition: avoid full-size iFFT by algebraically
            // splitting H(x) = H₀(x²) + x·H₁(x²) using:
            //   H₀(x²) = (H(x) + H(-x)) / 2
            //   H₁(x²) = (H(x) - H(-x)) / (2x)
            // On the LDE coset {g·ω^i}, we have -g·ω^i = g·ω^{i+N} since ω^N = -1.
            Self::decompose_and_extend_d2(&constraint_evaluations, domain)
        } else if number_of_parts == 1 {
            // Degree bound equals trace length: constraint evals are the LDE directly.
            vec![constraint_evaluations]
        } else {
            // Fallback for any future AIR with d > 2.
            let composition_poly =
                Polynomial::interpolate_offset_fft(&constraint_evaluations, &domain.coset_offset)
                    .unwrap();
            let composition_poly_parts = composition_poly.break_in_parts(number_of_parts);
            composition_poly_parts
                .iter()
                .map(|part| {
                    evaluate_polynomial_on_lde_domain(
                        part,
                        domain.blowup_factor,
                        domain.interpolation_domain_size,
                        &domain.coset_offset,
                    )
                    .unwrap()
                })
                .collect()
        };
        #[cfg(feature = "instruments")]
        let fft_dur = t_sub.elapsed();

        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();
        let composition_leaves =
            compute_tagged_leaves_row_pair_bit_reversed_composition::<FieldExtension>(
                &lde_composition_poly_parts_evaluations,
                tag,
            );
        if composition_leaves.is_empty() {
            return Err(ProvingError::EmptyCommitment);
        }
        let padded_height = composition_leaves.len();
        #[cfg(feature = "instruments")]
        let merkle_dur = t_sub.elapsed();
        #[cfg(feature = "instruments")]
        crate::instruments::store_r2_sub(constraints_dur, fft_dur, merkle_dur);

        Ok(R2aResult {
            lde_composition_poly_evaluations: Arc::new(lde_composition_poly_parts_evaluations),
            composition_leaves,
            padded_height,
        })
    }

    /// Returns the result of the third round of the STARK Prove protocol.
    fn round_3_evaluate_polynomials_in_out_of_domain_element(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        domain: &Domain<Field>,
        round_1_result: &Round1<Field, FieldExtension>,
        round_2_result: &Round2<FieldExtension>,
        z: &FieldElement<FieldExtension>,
    ) -> Round3<FieldExtension>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let num_parts = round_2_result.lde_composition_poly_evaluations.len();
        let z_power = z.pow(num_parts);
        let domain_size = domain.interpolation_domain_size;
        let blowup_factor = domain.blowup_factor;

        // === Shared domain constants for barycentric evaluation ===
        let dc = DomainConstants::from_domain(domain);

        // === Composition poly parts: barycentric evaluation at z^num_parts ===
        let comp_z_pow_n = z_power.pow(domain_size);
        let comp_inv_denoms = math::polynomial::barycentric_inv_denoms(&z_power, &dc.points);

        let composition_poly_parts_ood_evaluation: Vec<_> = round_2_result
            .lde_composition_poly_evaluations
            .iter()
            .map(|lde_evals| {
                // Extract trace-size evaluations (stride = blowup_factor)
                let evals: Vec<FieldElement<FieldExtension>> = (0..domain_size)
                    .map(|i| lde_evals[i * blowup_factor].clone())
                    .collect();
                math::polynomial::interpolate_coset_eval_ext_with_g_n_inv(
                    &comp_z_pow_n,
                    &dc.offset_pow_n,
                    &dc.size_inv,
                    &dc.offset_pow_n_inv,
                    &dc.points,
                    &evals,
                    &comp_inv_denoms,
                )
            })
            .collect();

        // === Trace polynomials: barycentric evaluation via LDE ===
        let trace_ood_evaluations = crate::trace::get_trace_evaluations_from_lde(
            &round_1_result.lde_trace,
            domain,
            z,
            &air.context().transition_offsets,
            air.step_size(),
            &dc,
        );

        Round3 {
            trace_ood_evaluations,
            composition_poly_parts_ood_evaluation,
        }
    }

    /// Returns the result of the fourth round of the STARK Prove protocol.
    fn round_4_compute_and_run_fri_on_the_deep_composition_polynomial(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        domain: &Domain<Field>,
        round_1_result: &Round1<Field, FieldExtension>,
        round_2_result: &Round2<FieldExtension>,
        round_3_result: &Round3<FieldExtension>,
        z: &FieldElement<FieldExtension>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Round4<Field, FieldExtension>
    where
        FieldElement<FieldExtension>: AsBytes,
        FieldElement<Field>: AsBytes,
    {
        let coset_offset_u64 = air.context().proof_options.coset_offset;
        let coset_offset = FieldElement::<Field>::from(coset_offset_u64);

        let gamma = transcript.sample_field_element();

        let n_terms_composition_poly = round_2_result.lde_composition_poly_evaluations.len();
        let num_terms_trace =
            air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;

        // <<<< Receive challenges: 𝛾, 𝛾'
        let mut deep_composition_coefficients: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
                .take(n_terms_composition_poly + num_terms_trace)
                .collect();

        let trace_term_coeffs: Vec<_> = deep_composition_coefficients
            .drain(..num_terms_trace)
            .collect::<Vec<_>>()
            .chunks(air.context().transition_offsets.len() * air.step_size())
            .map(|chunk| chunk.to_vec())
            .collect();

        // <<<< Receive challenges: 𝛾ⱼ, 𝛾ⱼ'
        let gammas = deep_composition_coefficients;

        // Compute p₀ (deep composition polynomial) as N evaluations on trace-size coset
        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();
        let deep_evals = Self::compute_deep_composition_poly_evaluations(
            &round_1_result.lde_trace,
            round_2_result,
            round_3_result,
            z,
            domain,
            &domain.trace_primitive_root,
            &gammas,
            &trace_term_coeffs,
        );
        #[cfg(feature = "instruments")]
        let other_dur_1 = t_sub.elapsed();

        // DEEP evaluations are already at 2N LDE points — just bit-reverse for FRI.
        // No iFFT+FFT extension needed (Plonky3-style direct LDE computation).
        let domain_size = domain.lde_roots_of_unity_coset.len();
        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();
        let mut lde_evals = deep_evals;
        in_place_bit_reverse_permute(&mut lde_evals);
        #[cfg(feature = "instruments")]
        let r4_fft_dur = t_sub.elapsed();

        // FRI commit phase from pre-computed evaluations
        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();
        let (fri_last_value, fri_layers) =
            fri::commit_phase_from_evaluations::<Field, FieldExtension>(
                domain.root_order as usize,
                lde_evals,
                transcript,
                &coset_offset,
                domain_size,
            );
        #[cfg(feature = "instruments")]
        let r4_merkle_dur = t_sub.elapsed();

        // grinding: generate nonce and append it to the transcript
        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();
        let security_bits = air.context().proof_options.grinding_factor;
        let mut nonce = None;
        if security_bits > 0 {
            let nonce_value = grinding::generate_nonce(&transcript.state(), security_bits)
                .expect("nonce not found");
            transcript.append_bytes(&nonce_value.to_be_bytes());
            nonce = Some(nonce_value);
        }

        let number_of_queries = air.options().fri_number_of_queries;
        let iotas = Self::sample_query_indexes(number_of_queries, domain, transcript);

        let query_list = fri::query_phase(&fri_layers, &iotas);

        let fri_layers_merkle_roots: Vec<_> = fri_layers
            .iter()
            .map(|layer| layer.merkle_tree.root)
            .collect();

        let deep_poly_openings =
            Self::open_deep_composition_poly(domain, round_1_result, round_2_result, &iotas);

        #[cfg(feature = "instruments")]
        {
            let queries_dur = t_sub.elapsed();
            crate::instruments::store_r4_sub(r4_fft_dur, r4_merkle_dur, other_dur_1, queries_dur);
        }

        Round4 {
            fri_last_value,
            fri_layers_merkle_roots,
            deep_poly_openings,
            query_list,
            nonce,
        }
    }

    fn sample_query_indexes(
        number_of_queries: usize,
        domain: &Domain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Vec<usize> {
        let domain_size = domain.lde_roots_of_unity_coset.len() as u64;
        (0..number_of_queries)
            .map(|_| (transcript.sample_u64(domain_size >> 1)) as usize)
            .collect::<Vec<usize>>()
    }

    /// Computes the DEEP composition polynomial at all 2N LDE points (Plonky3-style).
    ///
    /// Evaluates directly on the full LDE domain, eliminating the iFFT(N)+FFT(2N)
    /// extension that was needed when computing at only N trace-coset points.
    /// The result is ready for FRI after bit-reversal — no FFT needed.
    ///
    /// The DEEP polynomial is:
    ///   deep(X) = Σ_j γ_j * (H_j(X) - H_j(z^K)) / (X - z^K)
    ///           + Σ_{j,k} γ'_{j,k} * (t_j(X) - t_j(z·w^k)) / (X - z·w^k)
    #[allow(clippy::too_many_arguments)]
    fn compute_deep_composition_poly_evaluations(
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        round_2_result: &Round2<FieldExtension>,
        round_3_result: &Round3<FieldExtension>,
        z: &FieldElement<FieldExtension>,
        domain: &Domain<Field>,
        primitive_root: &FieldElement<Field>,
        composition_poly_gammas: &[FieldElement<FieldExtension>],
        trace_terms_gammas: &[Vec<FieldElement<FieldExtension>>],
    ) -> Vec<FieldElement<FieldExtension>>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let num_parts = round_2_result.lde_composition_poly_evaluations.len();
        let z_power = z.pow(num_parts); // pole for H terms

        // Number of evaluation points per trace column (= transition_offsets.len() * step_size)
        let num_eval_points = if trace_terms_gammas.is_empty() {
            0
        } else {
            trace_terms_gammas[0].len()
        };

        // Trace poles: z_shifted[k] = primitive_root^k * z for k = 0..num_eval_points
        let mut z_shifted = Vec::with_capacity(num_eval_points);
        let mut current_z = z.clone();
        for _ in 0..num_eval_points {
            z_shifted.push(current_z.clone());
            current_z = primitive_root * &current_z;
        }

        // Number of main and aux columns in the LDE trace
        let num_main_cols = lde_trace.num_main_cols();
        let num_aux_cols = lde_trace.num_aux_cols();

        // Precompute all inverse denominators at ALL LDE points via batch inversion.
        let lde_size = domain.lde_roots_of_unity_coset.len();
        let num_denoms = lde_size * (1 + num_eval_points);
        let mut denoms: Vec<FieldElement<FieldExtension>> = Vec::with_capacity(num_denoms);

        // H-term denominators: x_i - z^K (all 2N LDE points)
        for i in 0..lde_size {
            let x_i = &domain.lde_roots_of_unity_coset[i];
            denoms.push(x_i - &z_power);
        }

        // Trace-term denominators: x_i - z_shifted[k] (all 2N LDE points)
        for z_k in z_shifted.iter().take(num_eval_points) {
            for i in 0..lde_size {
                let x_i = &domain.lde_roots_of_unity_coset[i];
                denoms.push(x_i - z_k);
            }
        }

        FieldElement::inplace_batch_inverse(&mut denoms)
            .expect("Denominators should be non-zero: coset points are base field, poles are extension field");

        let inv_h = &denoms[0..lde_size];

        // OOD evaluations
        let h_ood = &round_3_result.composition_poly_parts_ood_evaluation;
        let trace_ood_columns = round_3_result.trace_ood_evaluations.columns();
        let num_total_cols = num_main_cols + num_aux_cols;

        // OOD column compression (Plonky3-style): precompute one value per eval point,
        //   ood_compressed_k = Σ_j gamma[j][k] * ood[j][k].
        // The per-LDE-point trace column sums are NOT precomputed — they are fused
        // directly into the hot loop below. DEEP is evaluated at all 2N LDE points
        // (no stride), so every row is used.
        let mut ood_compressed: Vec<FieldElement<FieldExtension>> =
            vec![FieldElement::zero(); num_eval_points];
        for j in 0..num_total_cols {
            let ood_evals_j = &trace_ood_columns[j];
            let gammas_j = &trace_terms_gammas[j];
            for k in 0..num_eval_points {
                ood_compressed[k] += &gammas_j[k] * &ood_evals_j[k];
            }
        }

        // Fused single-pass: compute column compression AND DEEP polynomial inline.
        // Eliminates the intermediate `compressed` allocation (~400 MB for CPU table)
        // and reduces to a single rayon dispatch instead of num_eval_points + 1.
        // Each row i's column data is reused across all eval points k within a rayon
        // task, so the k=1 read hits L1 cache after k=0 just loaded it.

        // Pre-gather gamma references per eval point for cache-friendly access.
        let main_gammas_by_k: Vec<Vec<&FieldElement<FieldExtension>>> = (0..num_eval_points)
            .map(|k| {
                (0..num_main_cols)
                    .map(|j| &trace_terms_gammas[j][k])
                    .collect()
            })
            .collect();
        let aux_gammas_by_k: Vec<Vec<&FieldElement<FieldExtension>>> = (0..num_eval_points)
            .map(|k| {
                (0..num_aux_cols)
                    .map(|j| &trace_terms_gammas[num_main_cols + j][k])
                    .collect()
            })
            .collect();

        #[cfg(feature = "parallel")]
        let iter = (0..lde_size).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = 0..lde_size;

        iter.map(|i| {
            let mut result = FieldElement::<FieldExtension>::zero();

            // H terms
            for j in 0..num_parts {
                let h_j_val = &round_2_result.lde_composition_poly_evaluations[j][i];
                let h_j_ood = &h_ood[j];
                result += &composition_poly_gammas[j] * (h_j_val - h_j_ood) * &inv_h[i];
            }

            // Trace terms: for each eval point k, compute the column sum inline
            // and multiply by the denominator inverse in one pass.
            for k in 0..num_eval_points {
                let inv_t_k_i = &denoms[(1 + k) * lde_size + i];
                let mut col_sum = FieldElement::<FieldExtension>::zero();
                for (j, gamma) in main_gammas_by_k[k].iter().enumerate() {
                    col_sum += lde_trace.get_main(i, j) * *gamma;
                }
                for (j, gamma) in aux_gammas_by_k[k].iter().enumerate() {
                    col_sum += lde_trace.get_aux(i, j) * *gamma;
                }
                result += inv_t_k_i * (col_sum - &ood_compressed[k]);
            }

            result
        })
        .collect()
    }

    /// Compute the composition-poly opening for one query against the
    /// chunk composition MMCS. The opening's `mmcs_opening` carries
    /// matrix_leaves for every chunk-mate's composition matrix; the
    /// closure rehashes those row-pair leaves on demand from the
    /// chunk-shared LDE columns.
    fn open_composition_poly(
        comp: &CompCommit<FieldExtension>,
        lde_composition_poly_evaluations: &[Vec<FieldElement<FieldExtension>>],
        index: usize,
    ) -> CompositionTraceOpening<FieldExtension>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send + ByteConversion,
    {
        let CompCommit::Shared { chunk_ctx, .. } = comp;
        let mmcs = &chunk_ctx.mmcs;
        let lde_in_spec_order = &chunk_ctx.lde_columns_in_spec_order;

        // Composition row-pair leaves are indexed by row-pair, so the
        // opening's global_index equals the query index directly (no
        // shift). Per-table local index = global_index >> shift, which
        // is 0 when all chunk-mates share the max height.
        let local_idx = index;
        let mmcs_opening = mmcs
            .open_with_leaves(local_idx, |m_idx, local_idx_in_matrix| {
                rehash_comp_chip_leaf::<FieldExtension>(
                    mmcs.spec()[m_idx].0,
                    &lde_in_spec_order[m_idx],
                    local_idx_in_matrix,
                )
            })
            .expect("composition MMCS open_with_leaves: index in range");

        // Build the (evaluations, evaluations_sym) field arrays from this
        // table's composition LDE — same layout as the legacy opening.
        let lde_composition_poly_parts_evaluation: Vec<_> = lde_composition_poly_evaluations
            .iter()
            .flat_map(|part| {
                vec![
                    part[reverse_index(index * 2, part.len() as u64)].clone(),
                    part[reverse_index(index * 2 + 1, part.len() as u64)].clone(),
                ]
            })
            .collect();
        let evaluations = lde_composition_poly_parts_evaluation
            .clone()
            .into_iter()
            .step_by(2)
            .collect();
        let evaluations_sym = lde_composition_poly_parts_evaluation
            .into_iter()
            .skip(1)
            .step_by(2)
            .collect();

        CompositionTraceOpening::Mmcs {
            evaluations,
            evaluations_sym,
            mmcs_opening,
        }
    }

    /// Computes values and validity proofs of the evaluations of trace polynomials at
    /// the FRI query challenge `challenge` and its symmetric counterpart. The caller
    /// supplies a `gather` closure that pulls the row data from the column-major LDE
    /// storage (full main row, ranged main row, or aux row).
    fn open_polys_with<C, G>(
        domain: &Domain<Field>,
        tree: &BatchedMerkleTree<C>,
        challenge: usize,
        gather: G,
    ) -> PolynomialOpenings<C>
    where
        C: IsField,
        FieldElement<C>: AsBytes + Sync + Send,
        G: Fn(usize) -> Vec<FieldElement<C>>,
    {
        let domain_size = domain.lde_roots_of_unity_coset.len() as u64;
        let index = challenge * 2;
        let index_sym = challenge * 2 + 1;
        PolynomialOpenings {
            proof: tree.get_proof_by_pos(index).unwrap(),
            proof_sym: tree.get_proof_by_pos(index_sym).unwrap(),
            evaluations: gather(reverse_index(index, domain_size)),
            evaluations_sym: gather(reverse_index(index_sym, domain_size)),
        }
    }

    /// Open the deep composition polynomial on a list of indexes and their symmetric elements.
    fn open_deep_composition_poly(
        domain: &Domain<Field>,
        round_1_result: &Round1<Field, FieldExtension>,
        round_2_result: &Round2<FieldExtension>,
        indexes_to_open: &[usize],
    ) -> DeepPolynomialOpenings<Field, FieldExtension>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let mut openings = Vec::with_capacity(indexes_to_open.len());

        let lde_trace = &round_1_result.lde_trace;
        let main_commit = &round_1_result.main;
        let total_cols = lde_trace.num_main_cols();

        for index in indexes_to_open.iter() {
            let composition_openings = Self::open_composition_poly(
                &round_2_result.comp,
                &round_2_result.lde_composition_poly_evaluations,
                *index,
            );

            let aux_trace_polys = round_1_result.aux.as_ref().map(|aux| {
                let AuxCommit::Shared { chunk_ctx, padded_height, .. } = aux;
                let mmcs = &chunk_ctx.mmcs;
                let lde_in_spec_order = &chunk_ctx.lde_columns_in_spec_order;
                let max_height = mmcs
                    .spec()
                    .first()
                    .map(|(_, h)| *h)
                    .expect("aux MMCS spec is non-empty when aux commit exists");
                debug_assert!(padded_height.is_power_of_two() && max_height >= *padded_height);
                let shift = (max_height / *padded_height).trailing_zeros() as usize;
                let domain_size = domain.lde_roots_of_unity_coset.len() as u64;
                let primary = *index * 2;
                let sym = *index * 2 + 1;
                let evaluations = lde_trace.gather_aux_row(reverse_index(primary, domain_size));
                let evaluations_sym = lde_trace.gather_aux_row(reverse_index(sym, domain_size));
                let mmcs_opening = mmcs
                    .open_with_leaves(primary << shift, |m_idx, local_idx| {
                        rehash_aux_chip_leaf::<FieldExtension>(
                            mmcs.spec()[m_idx].0,
                            &lde_in_spec_order[m_idx],
                            local_idx,
                        )
                    })
                    .expect("aux MMCS open_with_leaves: primary index in range");
                let mmcs_opening_sym = mmcs
                    .open_with_leaves(sym << shift, |m_idx, local_idx| {
                        rehash_aux_chip_leaf::<FieldExtension>(
                            mmcs.spec()[m_idx].0,
                            &lde_in_spec_order[m_idx],
                            local_idx,
                        )
                    })
                    .expect("aux MMCS open_with_leaves: sym index in range");
                crate::proof::stark::AuxTraceOpening::Mmcs {
                    evaluations,
                    evaluations_sym,
                    mmcs_opening,
                    mmcs_opening_sym,
                }
            });

            let (main_trace_opening, precomputed_trace_opening) = match main_commit {
                MainCommit::Shared {
                    chunk_ctx,
                    padded_height,
                    ..
                } => {
                    let mmcs = &chunk_ctx.mmcs;
                    let lde_in_spec_order = &chunk_ctx.lde_columns_in_spec_order;
                    let max_height = mmcs
                        .spec()
                        .first()
                        .map(|(_, h)| *h)
                        .expect("MMCS spec is non-empty");
                    debug_assert!(
                        padded_height.is_power_of_two() && max_height >= *padded_height
                    );
                    let shift = (max_height / *padded_height).trailing_zeros() as usize;
                    let domain_size = domain.lde_roots_of_unity_coset.len() as u64;
                    let primary = *index * 2;
                    let sym = *index * 2 + 1;
                    let evaluations = lde_trace.gather_main_row(reverse_index(primary, domain_size));
                    let evaluations_sym = lde_trace.gather_main_row(reverse_index(sym, domain_size));
                    let mmcs_opening = mmcs
                        .open_with_leaves(primary << shift, |m_idx, local_idx| {
                            rehash_main_chip_leaf::<Field>(
                                mmcs.spec()[m_idx].0,
                                &lde_in_spec_order[m_idx],
                                local_idx,
                            )
                        })
                        .expect("main MMCS open_with_leaves: primary index in range");
                    let mmcs_opening_sym = mmcs
                        .open_with_leaves(sym << shift, |m_idx, local_idx| {
                            rehash_main_chip_leaf::<Field>(
                                mmcs.spec()[m_idx].0,
                                &lde_in_spec_order[m_idx],
                                local_idx,
                            )
                        })
                        .expect("main MMCS open_with_leaves: sym index in range");
                    let opening = MainTraceOpening::Mmcs {
                        evaluations,
                        evaluations_sym,
                        mmcs_opening,
                        mmcs_opening_sym,
                    };
                    (opening, None)
                }
                MainCommit::Preprocessed {
                    multiplicities_tree,
                    precomputed_tree,
                    num_precomputed_cols,
                    ..
                } => {
                    let num_precomputed_cols = *num_precomputed_cols;
                    let mult = Self::open_polys_with(
                        domain,
                        multiplicities_tree,
                        *index,
                        |row| {
                            lde_trace.gather_main_row_range(
                                row,
                                num_precomputed_cols,
                                total_cols,
                            )
                        },
                    );
                    let pre = Self::open_polys_with(
                        domain,
                        precomputed_tree,
                        *index,
                        |row| lde_trace.gather_main_row_range(row, 0, num_precomputed_cols),
                    );
                    (MainTraceOpening::Tree(mult), Some(pre))
                }
            };

            openings.push(DeepPolynomialOpening {
                composition_poly: composition_openings,
                main_trace_polys: main_trace_opening,
                precomputed_trace_polys: precomputed_trace_opening,
                aux_trace_polys,
            });
        }

        openings
    }

    // TODO: propagate errors instead of unwrap() in commit_columns, reconstruct_round1, and expand_columns_to_lde
    /// Generates STARK proofs for one or more AIRs with a shared transcript.
    ///
    /// # Multi-Table Proving with LogUp
    ///
    /// When proving multiple tables that communicate via LogUp (lookup arguments),
    /// all tables must use the **same** random challenges (z, α) for the LogUp bus
    /// to balance correctly. This function ensures challenge sharing by:
    ///
    /// 1. **Commit all main traces**: All main trace commitments go into the
    ///    transcript before any challenges are sampled.
    /// 2. **Sample shared LogUp challenges**: The challenges (z, α) are sampled
    ///    once from the transcript and shared by all AIRs.
    /// 3. **Build auxiliary traces**: Each AIR builds its LogUp running-sum
    ///    columns using the shared challenges.
    /// 4. **Rounds 2-4**: Standard STARK protocol rounds for each AIR.
    ///
    /// # Warning
    ///
    /// The transcript must be safely initialized before passing it to this method.
    fn multi_prove(
        mut air_trace_pairs: Vec<AirTracePair<'_, Field, FieldExtension, PI>>,
        main_tags: &[MatrixTag],
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone + Send),
        #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
    ) -> Result<MultiProof<Field, FieldExtension, PI>, ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        PI: Send + Sync + Clone,
        Field: Copy + 'static,
        FieldExtension: Copy + 'static,
        <Field as IsField>::BaseType: SpillSafe,
        <FieldExtension as IsField>::BaseType: SpillSafe,
    {
        info!("Started proof generation...");

        #[cfg(feature = "instruments")]
        crate::instruments::reset_all();
        #[cfg(feature = "instruments")]
        let mut heap_snaps: Vec<crate::instruments::HeapSnapshot> = Vec::new();

        let num_airs = air_trace_pairs.len();

        if main_tags.len() != num_airs {
            return Err(ProvingError::WrongParameter(format!(
                "main_tags len ({}) does not match number of AIRs ({})",
                main_tags.len(),
                num_airs
            )));
        }

        // Check if any AIR has an auxiliary trace
        let needs_lookup_challenges = air_trace_pairs
            .iter()
            .any(|(air, _, _)| air.has_aux_trace());

        // =====================================================================
        // Pre-pass: compute domains and twiddles
        // =====================================================================

        #[cfg(feature = "instruments")]
        let phase_start = Instant::now();

        // Deduplicate Domain + LdeTwiddles by (trace_length, blowup_factor, coset_offset).
        // Many tables share the same domain size (e.g., 7+ tables at 2^20).
        // Without dedup, each creates its own Domain (~24 MB) and LdeTwiddles (~32 MB).
        type DomainEntry<F> = (Arc<Domain<F>>, Arc<LdeTwiddles<F>>);
        let mut domain_cache: std::collections::HashMap<(usize, usize, u64), DomainEntry<Field>> =
            std::collections::HashMap::new();

        let mut domains = Vec::with_capacity(num_airs);
        let mut twiddle_caches: Vec<Arc<LdeTwiddles<Field>>> = Vec::with_capacity(num_airs);

        for (air, trace, _pub_inputs) in &*air_trace_pairs {
            let trace_length = trace.num_rows();
            let blowup = air.options().blowup_factor as usize;
            let coset_offset = air.options().coset_offset;
            let key = (trace_length, blowup, coset_offset);

            #[cfg(test)]
            let was_hit = domain_cache.contains_key(&key);

            let (domain, twiddles) = domain_cache
                .entry(key)
                .or_insert_with(|| {
                    let d = Domain::new(*air, trace_length);
                    let t = LdeTwiddles::new(&d);
                    (Arc::new(d), Arc::new(t))
                })
                .clone();

            #[cfg(test)]
            crate::tests::domain_cache_stats::record(was_hit);

            domains.push(domain);
            twiddle_caches.push(twiddles);
        }
        // Free the HashMap (which holds extra strong Arc references) before the
        // long proving rounds begin. `domains` and `twiddle_caches` already hold
        // the only surviving Arcs we care about.
        drop(domain_cache);

        let k = table_parallelism().min(num_airs).max(1);

        // Spill main traces to mmap before Round 1 LDE.
        #[cfg(feature = "disk-spill")]
        if storage_mode == StorageMode::Disk {
            #[cfg(feature = "parallel")]
            let spill_iter = air_trace_pairs.par_iter_mut();
            #[cfg(not(feature = "parallel"))]
            let mut spill_iter = air_trace_pairs.iter_mut();
            spill_iter.try_for_each(|(_, trace, _)| {
                trace
                    .main_table
                    .spill_to_disk()
                    .map_err(|e| ProvingError::DiskSpill(format!("early main: {e}")))
            })?;
        }

        #[cfg(feature = "instruments")]
        let prepass_elapsed = phase_start.elapsed();
        #[cfg(feature = "instruments")]
        if let Some(s) = crate::instruments::snap("After pool alloc") {
            heap_snaps.push(s);
        }

        // =====================================================================
        // Round 1, Phase A: Commit all main traces (parallel in chunks of K)
        // =====================================================================
        // All main trace commitments must be in the transcript before sampling
        // LogUp challenges.

        #[cfg(feature = "instruments")]
        let phase_start = Instant::now();

        // Per-chunk MMCS: each chunk of K tables builds its own streaming
        // MMCS, sharing chunk LDEs via Arc so per-query opens can rehash
        // chunk-mate rows on demand. Phase A absorb order: per table in
        // spec order, absorb preprocessed + main-tree roots (preprocessed
        // only); after each chunk, absorb the chunk's MMCS root (`Some`)
        // or skip when the chunk has no Shared tables (`None`).
        let mut main_commits: Vec<Option<MainCommit<Field>>> = (0..num_airs).map(|_| None).collect();
        let mut main_ldes: Vec<Option<Arc<Vec<Vec<FieldElement<Field>>>>>> =
            (0..num_airs).map(|_| None).collect();
        let mut main_mmcs_roots_per_chunk: Vec<Option<Commitment>> = Vec::new();
        let mut main_mmcs_specs_per_chunk: Vec<Vec<(MatrixTag, usize)>> = Vec::new();

        for chunk_start in (0..num_airs).step_by(k) {
            let chunk_end = (chunk_start + k).min(num_airs);
            let chunk_range = chunk_start..chunk_end;

            #[cfg(feature = "parallel")]
            let iter = chunk_range.clone().into_par_iter();
            #[cfg(not(feature = "parallel"))]
            let iter = chunk_range.clone();

            let chunk_results: Vec<Result<_, ProvingError>> = iter
                .map(|idx| {
                    let (air, trace, _) = &air_trace_pairs[idx];
                    let domain = &domains[idx];
                    let twiddles = &twiddle_caches[idx];
                    let tag = main_tags[idx];

                    let precomputed = air
                        .is_preprocessed()
                        .then(|| (air.precomputed_commitment(), air.num_precomputed_columns()));
                    Self::commit_main_trace(
                        *trace,
                        domain,
                        twiddles,
                        tag,
                        precomputed,
                        #[cfg(feature = "disk-spill")]
                        storage_mode,
                    )
                })
                .collect();

            // Sequential: absorb per-table preprocessed + main-tree roots
            // (preprocessed only) in order, then build this chunk's MMCS
            // from the chunk's Shared outputs and absorb its root.
            let mut chunk_shared_outputs: Vec<(MatrixTag, Vec<Commitment>, usize)> = Vec::new();
            let mut chunk_shared_ldes: Vec<(MatrixTag, Arc<Vec<Vec<FieldElement<Field>>>>)> =
                Vec::new();
            let chunk_idx = main_mmcs_roots_per_chunk.len();
            let chunk_outputs: Vec<_> = chunk_results.into_iter().collect::<Result<_, _>>()?;
            for (offset, (output, cached_main)) in chunk_outputs.into_iter().enumerate() {
                let idx = chunk_start + offset;
                if let Some(ref pre_root) = output.precomputed_root() {
                    transcript.append_bytes(pre_root);
                }
                if let Some(ref main_root) = output.main_tree_root() {
                    transcript.append_bytes(main_root);
                }
                let cached_main_arc = Arc::new(cached_main);
                main_ldes[idx] = Some(Arc::clone(&cached_main_arc));
                match output {
                    MainPhaseAOutput::Shared {
                        tag,
                        leaves,
                        padded_height,
                    } => {
                        chunk_shared_outputs.push((tag, leaves, padded_height));
                        chunk_shared_ldes.push((tag, cached_main_arc));
                        // MainCommit::Shared placeholder filled in after chunk MMCS build.
                        main_commits[idx] = None;
                    }
                    MainPhaseAOutput::Preprocessed {
                        multiplicities_tree,
                        multiplicities_root,
                        precomputed_tree,
                        precomputed_root,
                        num_precomputed_cols,
                    } => {
                        main_commits[idx] = Some(MainCommit::Preprocessed {
                            multiplicities_tree,
                            multiplicities_root,
                            precomputed_tree,
                            precomputed_root,
                            num_precomputed_cols,
                        });
                    }
                }
            }

            let (chunk_root, chunk_spec, chunk_ctx_opt) =
                build_chunk_main_mmcs::<Field>(chunk_shared_outputs, chunk_shared_ldes)?;
            if let Some(ref root) = chunk_root {
                transcript.append_bytes(root);
            }
            main_mmcs_roots_per_chunk.push(chunk_root);
            main_mmcs_specs_per_chunk.push(chunk_spec.clone());

            // Fill in MainCommit::Shared for this chunk's Shared tables.
            if let Some(chunk_ctx) = chunk_ctx_opt {
                // chunk_spec is in MMCS sort order (height desc, tag asc).
                // Use tag → padded_height lookup to populate Shared variants.
                let height_by_tag: std::collections::BTreeMap<MatrixTag, usize> =
                    chunk_spec.iter().copied().collect();
                for idx in chunk_range.clone() {
                    if main_commits[idx].is_none() {
                        let tag = main_tags[idx];
                        if let Some(&padded_height) = height_by_tag.get(&tag) {
                            main_commits[idx] = Some(MainCommit::Shared {
                                chunk_ctx: Arc::clone(&chunk_ctx),
                                chunk_idx,
                                tag,
                                padded_height,
                            });
                        }
                    }
                }
            }
        }

        let main_commits: Vec<MainCommit<Field>> = main_commits
            .into_iter()
            .map(|c| c.expect("main commit populated for every table"))
            .collect();
        let main_ldes: Vec<Arc<Vec<Vec<FieldElement<Field>>>>> = main_ldes
            .into_iter()
            .map(|l| l.expect("main LDE populated for every table"))
            .collect();

        #[cfg(feature = "instruments")]
        let main_commits_elapsed = phase_start.elapsed();
        #[cfg(feature = "instruments")]
        if let Some(s) = crate::instruments::snap("After main commits") {
            heap_snaps.push(s);
        }

        // =====================================================================
        // Round 1, Phase B: Sample shared LogUp challenges
        // =====================================================================

        let lookup_challenges: Vec<FieldElement<FieldExtension>> = if needs_lookup_challenges {
            (0..LOGUP_NUM_CHALLENGES)
                .map(|_| transcript.sample_field_element())
                .collect()
        } else {
            Vec::new()
        };

        // =====================================================================
        // Phase C + Rounds 2-4: Forked per table
        // =====================================================================
        // Each table gets an independent transcript fork (cloned from the shared
        // state after Phase B, domain-separated by table index). This matches
        // the verifier's forking and makes per-table proving independent.
        //
        // Split into two passes for parallelism:
        //   Pass 1 (parallel): Build all auxiliary traces (fingerprint + batch inversion)
        //   Pass 2 (parallel): Fork transcript → extract → LDE → commit

        // Pass 1: Build aux traces in parallel.
        // Each build_auxiliary_trace has internal parallelism (batch_inverse, par_chunks),
        // but outer parallelism over 12 tables also helps on high-core-count machines.
        #[cfg(feature = "instruments")]
        let phase_start = Instant::now();

        #[cfg(feature = "parallel")]
        let aux_iter = air_trace_pairs.par_iter_mut();
        #[cfg(not(feature = "parallel"))]
        let aux_iter = air_trace_pairs.iter_mut();
        let bus_inputs_vec: Vec<Option<BusPublicInputs<FieldExtension>>> = aux_iter
            .map(|(air, trace, _)| {
                if air.has_aux_trace() {
                    air.build_auxiliary_trace(*trace, &lookup_challenges)
                } else {
                    None
                }
            })
            .collect();

        // Spill all aux trace tables to mmap before any Round 1 aux LDE work.
        #[cfg(feature = "disk-spill")]
        if storage_mode == StorageMode::Disk {
            #[cfg(feature = "parallel")]
            let spill_iter = air_trace_pairs.par_iter_mut();
            #[cfg(not(feature = "parallel"))]
            let mut spill_iter = air_trace_pairs.iter_mut();
            spill_iter.try_for_each(|(air, trace, _)| {
                if air.has_aux_trace() {
                    trace
                        .spill_aux_to_disk()
                        .map_err(|e| ProvingError::DiskSpill(format!("aux trace: {e}")))?;
                }
                Ok(())
            })?;
        }

        #[cfg(feature = "instruments")]
        let aux_build_elapsed = phase_start.elapsed();
        #[cfg(feature = "instruments")]
        if let Some(s) = crate::instruments::snap("After aux build") {
            heap_snaps.push(s);
        }

        // Pass 2: parallel aux-LDE + tagged-leaf computation, then a single
        // shared aux MMCS build. The aux MMCS root is absorbed into the
        // SHARED transcript BEFORE per-table forking, so every table's
        // forked transcript sees the same aux MMCS commitment without
        // dragging per-table aux roots through Fiat-Shamir.
        #[cfg(feature = "instruments")]
        let phase_start = Instant::now();

        // Per-chunk aux MMCS: mirror of Phase A main, applied to the aux
        // trace. Each chunk's aux MMCS root is absorbed into the SHARED
        // transcript BEFORE per-table forking so every fork sees the
        // same per-chunk aux binding identically.
        let mut aux_commits: Vec<Option<AuxCommit<FieldExtension>>> =
            (0..num_airs).map(|_| None).collect();
        let mut aux_ldes_arc: Vec<Arc<Vec<Vec<FieldElement<FieldExtension>>>>> =
            Vec::with_capacity(num_airs);
        let mut aux_mmcs_roots_per_chunk: Vec<Option<Commitment>> = Vec::new();
        let mut aux_mmcs_specs_per_chunk: Vec<Vec<(MatrixTag, usize)>> = Vec::new();

        for chunk_start in (0..num_airs).step_by(k) {
            let chunk_end = (chunk_start + k).min(num_airs);
            let chunk_range = chunk_start..chunk_end;

            #[cfg(feature = "parallel")]
            let iter = chunk_range.clone().into_par_iter();
            #[cfg(not(feature = "parallel"))]
            let iter = chunk_range.clone();

            let chunk_aux: Vec<Result<_, ProvingError>> = iter
                .map(|idx| {
                    let (air, trace, _) = &air_trace_pairs[idx];
                    let domain = &domains[idx];
                    let twiddles = &twiddle_caches[idx];
                    let tag = main_tags[idx];

                    if air.has_aux_trace() {
                        let lde_size = domain.interpolation_domain_size * domain.blowup_factor;
                        let mut columns = trace.extract_columns_aux(lde_size);
                        #[cfg(feature = "disk-spill")]
                        if storage_mode == StorageMode::Disk {
                            trace.aux_table.advise_drop_cache();
                        }
                        #[cfg(feature = "instruments")]
                        let t_sub = Instant::now();
                        Self::expand_columns_to_lde::<FieldExtension>(
                            &mut columns,
                            domain,
                            twiddles,
                        );
                        #[cfg(feature = "instruments")]
                        let aux_lde_dur = t_sub.elapsed();
                        #[cfg(feature = "instruments")]
                        let t_sub = Instant::now();
                        let leaves =
                            compute_tagged_leaves_bit_reversed_aux::<FieldExtension>(&columns, tag);
                        if leaves.is_empty() {
                            return Err(ProvingError::EmptyCommitment);
                        }
                        let padded_height = leaves.len();
                        #[cfg(feature = "instruments")]
                        crate::instruments::accum_r1_aux(aux_lde_dur, t_sub.elapsed());
                        let output = AuxPhaseCOutput::<FieldExtension> {
                            tag,
                            leaves,
                            padded_height,
                            _marker: PhantomData,
                        };
                        Ok((Some(output), columns))
                    } else {
                        Ok((None, Vec::new()))
                    }
                })
                .collect();

            let chunk_idx = aux_mmcs_roots_per_chunk.len();
            let mut chunk_aux_outputs: Vec<(MatrixTag, Vec<Commitment>, usize)> = Vec::new();
            let mut chunk_aux_ldes: Vec<(MatrixTag, Arc<Vec<Vec<FieldElement<FieldExtension>>>>)> =
                Vec::new();
            let chunk_outputs: Vec<_> = chunk_aux.into_iter().collect::<Result<_, _>>()?;
            for (offset, (maybe_output, cached_aux)) in chunk_outputs.into_iter().enumerate() {
                let idx = chunk_start + offset;
                let cached_arc = Arc::new(cached_aux);
                aux_ldes_arc.push(Arc::clone(&cached_arc));
                if let Some(out) = maybe_output {
                    let AuxPhaseCOutput {
                        tag,
                        leaves,
                        padded_height,
                        ..
                    } = out;
                    chunk_aux_outputs.push((tag, leaves, padded_height));
                    chunk_aux_ldes.push((tag, cached_arc));
                    aux_commits[idx] = None; // filled in after MMCS build
                } else {
                    aux_commits[idx] = None;
                }
            }

            let (chunk_root, chunk_spec, chunk_ctx_opt) =
                build_chunk_aux_mmcs::<FieldExtension>(chunk_aux_outputs, chunk_aux_ldes)?;
            if let Some(ref root) = chunk_root {
                transcript.append_bytes(root);
            }
            aux_mmcs_roots_per_chunk.push(chunk_root);
            aux_mmcs_specs_per_chunk.push(chunk_spec.clone());

            if let Some(chunk_ctx) = chunk_ctx_opt {
                let height_by_tag: std::collections::BTreeMap<MatrixTag, usize> =
                    chunk_spec.iter().copied().collect();
                for idx in chunk_range.clone() {
                    let (air, _, _) = &air_trace_pairs[idx];
                    if air.has_aux_trace() {
                        let tag = main_tags[idx];
                        if let Some(&padded_height) = height_by_tag.get(&tag) {
                            aux_commits[idx] = Some(AuxCommit::Shared {
                                chunk_ctx: Arc::clone(&chunk_ctx),
                                chunk_idx,
                                tag,
                                padded_height,
                            });
                        }
                    }
                }
            }
        }

        // Pre-fork all transcripts (cheap, sequential — must match verifier ordering).
        // Happens AFTER all per-chunk aux MMCS roots have been absorbed.
        let mut table_transcripts: Vec<_> = (0..num_airs)
            .map(|idx| {
                let mut t = transcript.clone();
                if num_airs > 1 {
                    t.append_bytes(&(idx as u64).to_le_bytes());
                }
                t
            })
            .collect();

        #[allow(clippy::type_complexity)]
        let aux_results: Vec<(
            Option<AuxCommit<FieldExtension>>,
            Arc<Vec<Vec<FieldElement<FieldExtension>>>>,
        )> = aux_commits
            .into_iter()
            .zip(aux_ldes_arc)
            .collect();

        // Build commitments and cached LDEs as separate vecs:
        // commitments are borrowed in Phase D, LDEs are consumed by value.
        let mut commitments: Vec<Round1Commitments<Field, FieldExtension>> =
            Vec::with_capacity(num_airs);
        let mut cached_ldes: Vec<Lde<Field, FieldExtension>> = Vec::with_capacity(num_airs);
        for (((main_commit, main_lde), (aux_commit, cached_aux)), bus_public_inputs) in main_commits
            .into_iter()
            .zip(main_ldes)
            .zip(aux_results)
            .zip(bus_inputs_vec)
        {
            commitments.push(Round1Commitments {
                main: main_commit,
                aux: aux_commit,
                rap_challenges: lookup_challenges.clone(),
                bus_public_inputs,
            });
            cached_ldes.push(Lde {
                main: main_lde,
                aux: cached_aux,
            });
        }

        #[cfg(feature = "instruments")]
        let aux_commit_elapsed = phase_start.elapsed();
        #[cfg(feature = "instruments")]
        if let Some(s) = crate::instruments::snap("After aux commit") {
            heap_snaps.push(s);
        }

        #[cfg(feature = "debug-checks")]
        Self::run_debug_checks(&air_trace_pairs, &commitments, &domains, &twiddle_caches);

        // =====================================================================
        // Rounds 2-4: Parallel per-table proving in chunks of K
        // =====================================================================
        // Each chunk of K tables is processed in parallel. Cached LDE columns
        // from Phase A/C are consumed here (zero-copy move), eliminating the
        // expensive reconstruct_round1 recomputation.

        #[cfg(feature = "instruments")]
        let phase_start = Instant::now();
        #[cfg(feature = "instruments")]
        let mut table_timings: Vec<(
            String,
            usize,
            std::time::Duration,
            crate::instruments::TableSubOps,
        )> = Vec::with_capacity(num_airs);

        let mut proofs: Vec<Option<StarkProof<Field, FieldExtension, PI>>> =
            (0..num_airs).map(|_| None).collect();
        let mut comp_mmcs_roots_per_chunk: Vec<Option<Commitment>> = Vec::new();
        let mut comp_mmcs_specs_per_chunk: Vec<Vec<(MatrixTag, usize)>> = Vec::new();
        let mut lde_drain = cached_ldes.into_iter();
        for chunk_start in (0..num_airs).step_by(k) {
            let chunk_end = (chunk_start + k).min(num_airs);
            let chunk_size = chunk_end - chunk_start;
            let chunk_idx = comp_mmcs_roots_per_chunk.len();

            let chunk_ldes: Vec<Lde<Field, FieldExtension>> =
                lde_drain.by_ref().take(chunk_size).collect();
            let chunk_commitments = &commitments[chunk_start..chunk_end];
            // Build Round1 per-table sequentially (build_round1 only bumps
            // Arc refcounts), then run R2a in parallel.
            let chunk_round1: Vec<Round1<Field, FieldExtension>> = chunk_ldes
                .into_iter()
                .zip(chunk_commitments.iter())
                .enumerate()
                .map(|(j, (lde, commitment))| {
                    let idx = chunk_start + j;
                    let (air, _, _) = &air_trace_pairs[idx];
                    let domain = &domains[idx];
                    commitment.build_round1(lde, air.step_size(), domain.blowup_factor)
                })
                .collect();

            // Bind per-table table_contribution into forks before sampling beta.
            for (j, round_1_result) in chunk_round1.iter().enumerate() {
                let idx = chunk_start + j;
                if let Some(ref bpi) = round_1_result.bus_public_inputs {
                    table_transcripts[idx].append_field_element(&bpi.table_contribution);
                }
            }

            // Phase R2a (sequential within chunk): sample beta + build
            // composition LDE + tagged leaves per table. Internal
            // parallelism inside constraint eval / FFT keeps cores busy.
            // K is small (chunk size = table_parallelism()), so per-table
            // serialization here costs little.
            let chunk_transcripts = &mut table_transcripts[chunk_start..chunk_end];
            let r2a_iter = chunk_round1
                .iter()
                .zip(chunk_transcripts.iter_mut())
                .enumerate();

            #[allow(clippy::type_complexity)]
            let r2a_results: Vec<Result<
                (
                    usize,
                    Vec<FieldElement<FieldExtension>>,
                    Vec<FieldElement<FieldExtension>>,
                    R2aResult<FieldExtension>,
                ),
                ProvingError,
            >> = r2a_iter
                .map(|(j, (round_1_result, table_transcript))| {
                    let idx = chunk_start + j;
                    let (air, _, pub_inputs) = &air_trace_pairs[idx];
                    let domain = &domains[idx];
                    let tag = main_tags[idx];
                    let (tc, bc, r2a) = Self::prove_round_2a(
                        *air,
                        *pub_inputs,
                        round_1_result,
                        table_transcript,
                        domain,
                        tag,
                    )?;
                    Ok((j, tc, bc, r2a))
                })
                .collect();

            // Sequential: collect R2a outputs in chunk-local-index order;
            // build chunk composition MMCS over them.
            let mut chunk_r2a: Vec<Option<(
                Vec<FieldElement<FieldExtension>>,
                Vec<FieldElement<FieldExtension>>,
                R2aResult<FieldExtension>,
            )>> = (0..chunk_size).map(|_| None).collect();
            for r in r2a_results {
                let (j, tc, bc, r2a) = r?;
                chunk_r2a[j] = Some((tc, bc, r2a));
            }

            let mut chunk_comp_outputs: Vec<(MatrixTag, Vec<Commitment>, usize)> = Vec::new();
            let mut chunk_comp_ldes: Vec<(MatrixTag, Arc<Vec<Vec<FieldElement<FieldExtension>>>>)> =
                Vec::new();
            for (j, entry) in chunk_r2a.iter().enumerate() {
                let idx = chunk_start + j;
                let tag = main_tags[idx];
                let (_, _, r2a) = entry.as_ref().expect("R2a populated");
                chunk_comp_outputs.push((tag, r2a.composition_leaves.clone(), r2a.padded_height));
                chunk_comp_ldes.push((tag, Arc::clone(&r2a.lde_composition_poly_evaluations)));
            }

            let (chunk_comp_root, chunk_comp_spec, chunk_comp_ctx_opt) =
                build_chunk_comp_mmcs::<FieldExtension>(chunk_comp_outputs, chunk_comp_ldes)?;
            // Absorb chunk composition root into EACH chunk-mate's fork.
            if let Some(ref root) = chunk_comp_root {
                for idx in chunk_start..chunk_end {
                    table_transcripts[idx].append_bytes(root);
                }
            }
            comp_mmcs_roots_per_chunk.push(chunk_comp_root);
            comp_mmcs_specs_per_chunk.push(chunk_comp_spec.clone());

            let chunk_comp_ctx = chunk_comp_ctx_opt
                .expect("chunk has at least one composition matrix (every table has comp)");
            let height_by_tag: std::collections::BTreeMap<MatrixTag, usize> =
                chunk_comp_spec.iter().copied().collect();

            // Reassemble per-table Round2 from R2a + chunk MMCS context.
            let mut chunk_round2: Vec<Round2<FieldExtension>> = Vec::with_capacity(chunk_size);
            for j in 0..chunk_size {
                let idx = chunk_start + j;
                let tag = main_tags[idx];
                let (_, _, r2a) = chunk_r2a[j].take().unwrap();
                let padded_height = *height_by_tag.get(&tag).expect("spec contains tag");
                chunk_round2.push(Round2 {
                    lde_composition_poly_evaluations: r2a.lde_composition_poly_evaluations,
                    comp: CompCommit::Shared {
                        chunk_ctx: Arc::clone(&chunk_comp_ctx),
                        chunk_idx,
                        tag,
                        padded_height,
                    },
                });
            }

            // Phase R2b → R4 (sequential within chunk): each fork has
            // the chunk comp root absorbed; sample z, run R3 OOD + R4
            // FRI. Same rationale as R2a above.
            let chunk_transcripts = &mut table_transcripts[chunk_start..chunk_end];
            let r2b_iter = chunk_round1
                .iter()
                .zip(chunk_round2.iter())
                .zip(chunk_transcripts.iter_mut())
                .enumerate();

            let chunk_results: Vec<Result<_, ProvingError>> = r2b_iter
                .map(|(j, ((round_1_result, round_2_result), table_transcript))| {
                    let idx = chunk_start + j;
                    let (air, trace, pub_inputs) = &air_trace_pairs[idx];
                    let _ = trace; // used by instruments
                    let domain = &domains[idx];

                    #[cfg(feature = "instruments")]
                    let table_start = Instant::now();

                    let proof = Self::prove_rounds_2b_to_4(
                        *air,
                        *pub_inputs,
                        round_1_result,
                        round_2_result,
                        table_transcript,
                        domain,
                    )?;

                    #[cfg(feature = "instruments")]
                    let table_timing = {
                        let sub_ops = crate::instruments::take_round_sub_ops().unwrap_or_default();
                        (
                            air.name().to_string(),
                            trace.num_rows(),
                            table_start.elapsed(),
                            sub_ops,
                        )
                    };

                    #[cfg(feature = "instruments")]
                    return Ok((j, proof, table_timing));
                    #[cfg(not(feature = "instruments"))]
                    Ok((j, proof))
                })
                .collect();

            for result in chunk_results {
                #[cfg(feature = "instruments")]
                {
                    let (j, proof, timing) = result?;
                    let idx = chunk_start + j;
                    proofs[idx] = Some(proof);
                    table_timings.push(timing);
                }
                #[cfg(not(feature = "instruments"))]
                {
                    let (j, proof) = result?;
                    let idx = chunk_start + j;
                    proofs[idx] = Some(proof);
                }
            }
        }

        let proofs: Vec<StarkProof<Field, FieldExtension, PI>> = proofs
            .into_iter()
            .map(|p| p.expect("every table emits a proof"))
            .collect();

        #[cfg(feature = "instruments")]
        {
            // Store timing data for the top-level report in prove_with_options.
            // Uses a thread-local to avoid changing multi_prove's return type.
            crate::instruments::store(crate::instruments::MultiProveTiming {
                prepass: prepass_elapsed,
                main_commits: main_commits_elapsed,
                aux_build: aux_build_elapsed,
                aux_commit: aux_commit_elapsed,
                rounds_2_4: phase_start.elapsed(),
                round1_sub: crate::instruments::take_r1_sub(),
                table_timings,
                heap_snapshots: heap_snaps,
            });
        }

        Ok(MultiProof {
            proofs,
            main_mmcs_roots: main_mmcs_roots_per_chunk,
            main_mmcs_specs: main_mmcs_specs_per_chunk,
            aux_mmcs_roots: aux_mmcs_roots_per_chunk,
            aux_mmcs_specs: aux_mmcs_specs_per_chunk,
            comp_mmcs_roots: comp_mmcs_roots_per_chunk,
            comp_mmcs_specs: comp_mmcs_specs_per_chunk,
            chunk_size: k as u32,
        })
    }

    /// Generate a single-AIR STARK proof, returned as a one-element
    /// `MultiProof`. The MMCS root + spec live at the multi-proof level (see
    /// `MultiProof`), so single-table callers consume the wrapper directly.
    fn prove(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        trace: &mut TraceTable<Field, FieldExtension>,
        pub_inputs: &PI,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone + Send),
    ) -> Result<MultiProof<Field, FieldExtension, PI>, ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        PI: Send + Sync + Clone,
        Field: Copy + 'static,
        FieldExtension: Copy + 'static,
        <Field as IsField>::BaseType: SpillSafe,
        <FieldExtension as IsField>::BaseType: SpillSafe,
    {
        let air_trace_pairs = vec![(air, trace, pub_inputs)];
        // Single-AIR path: synthesize a default tag. Callers that need
        // distinct chip identities call `multi_prove` directly.
        let main_tags = [MatrixTag::new([0; 8])];
        Self::multi_prove(
            air_trace_pairs,
            &main_tags,
            transcript,
            #[cfg(feature = "disk-spill")]
            StorageMode::Ram,
        )
    }

    // TODO: propagate errors instead of unwrap() in open_deep_composition_poly and FRI operations
    /// Executes rounds 2-4 and generates a STARK proof for the trace `main_trace` with public inputs `pub_inputs`.
    /// Warning: the transcript must be safely initializated before passing it to this method.
    /// Part A of Round 2: sample beta + build the composition LDE parts
    /// + compute tagged row-pair leaves for the chunk composition MMCS.
    /// Returns the artefacts the chunk-level MMCS build consumes
    /// alongside this table's tag.
    fn prove_round_2a(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        pub_inputs: &PI,
        round_1_result: &Round1<Field, FieldExtension>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        domain: &Domain<Field>,
        tag: MatrixTag,
    ) -> Result<
        (
            Vec<FieldElement<FieldExtension>>,
            Vec<FieldElement<FieldExtension>>,
            R2aResult<FieldExtension>,
        ),
        ProvingError,
    >
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        PI: Send + Sync + Clone,
    {
        // <<<< Receive challenge: 𝛽
        let beta = transcript.sample_field_element();
        let trace_length = domain.interpolation_domain_size;
        let num_boundary_constraints = air
            .boundary_constraints(
                pub_inputs,
                &round_1_result.rap_challenges,
                round_1_result.bus_public_inputs.as_ref(),
                trace_length,
            )
            .constraints
            .len();
        let num_transition_constraints = air.context().num_transition_constraints;
        let mut coefficients: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &beta))
                .take(num_boundary_constraints + num_transition_constraints)
                .collect();
        let transition_coefficients: Vec<_> =
            coefficients.drain(..num_transition_constraints).collect();
        let boundary_coefficients = coefficients;
        let r2a = Self::round_2a_build_composition_lde(
            air,
            pub_inputs,
            domain,
            round_1_result,
            &transition_coefficients,
            &boundary_coefficients,
            tag,
        )?;
        Ok((transition_coefficients, boundary_coefficients, r2a))
    }

    /// Part B of Round 2 onward: assumes the chunk composition MMCS root
    /// has been absorbed into `transcript` already. Runs the absorb of
    /// the per-table H_i values, R3 OOD, and R4 FRI + opens, producing
    /// the final per-table StarkProof.
    #[allow(clippy::too_many_arguments)]
    fn prove_rounds_2b_to_4(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        pub_inputs: &PI,
        round_1_result: &Round1<Field, FieldExtension>,
        round_2_result: &Round2<FieldExtension>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        domain: &Domain<Field>,
    ) -> Result<StarkProof<Field, FieldExtension, PI>, ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        PI: Send + Sync + Clone,
    {
        info!("Started proof generation (post-R2 chunk join)...");

        // <<<< Receive challenge: z (transcript already saw chunk comp root)
        let z = transcript.sample_z_ood(
            &domain.lde_roots_of_unity_coset,
            &domain.trace_roots_of_unity,
        );

        #[cfg(feature = "instruments")]
        let t_r3 = Instant::now();
        let round_3_result = Self::round_3_evaluate_polynomials_in_out_of_domain_element(
            air,
            domain,
            round_1_result,
            round_2_result,
            &z,
        );
        #[cfg(feature = "instruments")]
        let round_3_dur = t_r3.elapsed();

        // >>>> Send values: tⱼ(zgᵏ)
        let trace_ood_evaluations_columns = round_3_result.trace_ood_evaluations.columns();
        for col in trace_ood_evaluations_columns.iter() {
            for elem in col.iter() {
                transcript.append_field_element(elem);
            }
        }

        // >>>> Send values: Hᵢ(z^N)
        for element in round_3_result.composition_poly_parts_ood_evaluation.iter() {
            transcript.append_field_element(element);
        }

        // ===================================
        // ==========|   Round 4   |==========
        // ===================================
        let round_4_result = Self::round_4_compute_and_run_fri_on_the_deep_composition_polynomial(
            air,
            domain,
            round_1_result,
            round_2_result,
            &round_3_result,
            &z,
            transcript,
        );

        #[cfg(feature = "instruments")]
        {
            let zero = std::time::Duration::ZERO;
            let (r2_constraints, r2_fft, r2_merkle) =
                crate::instruments::take_r2_sub().unwrap_or((zero, zero, zero));
            let (r4_fft, r4_merkle, r4_deep_comp, r4_queries) =
                crate::instruments::take_r4_sub().unwrap_or((zero, zero, zero, zero));
            crate::instruments::store_round_sub_ops(crate::instruments::TableSubOps {
                constraints: r2_constraints,
                comp_decompose: r2_fft,
                comp_commit: r2_merkle,
                ood: round_3_dur,
                deep_comp: r4_deep_comp,
                deep_extend: r4_fft,
                fri_commit: r4_merkle,
                queries: r4_queries,
            });
        }

        info!("End proof generation");

        Ok(StarkProof {
            lde_trace_main_merkle_root: round_1_result.main.main_tree_root(),
            lde_trace_precomputed_merkle_root: round_1_result.main.precomputed_root(),
            trace_ood_evaluations: round_3_result.trace_ood_evaluations,
            composition_poly_parts_ood_evaluation: round_3_result
                .composition_poly_parts_ood_evaluation,
            fri_layers_merkle_roots: round_4_result.fri_layers_merkle_roots,
            fri_last_value: round_4_result.fri_last_value,
            query_list: round_4_result.query_list,
            deep_poly_openings: round_4_result.deep_poly_openings,
            nonce: round_4_result.nonce,
            bus_public_inputs: round_1_result.bus_public_inputs.clone(),
            public_inputs: pub_inputs.clone(),
            trace_length: domain.interpolation_domain_size,
        })
    }
}

/// Print a global bus balance report aggregating per-bus sums across all tables.
///
/// Uses numeric bus IDs only (no VM-specific names) to keep the stark crate generic.
/// For bus ID → name mapping, see `BusId` in the prover crate.
#[cfg(feature = "debug-checks")]
fn print_bus_balance_report<FieldExtension>(
    all_bus_public_inputs: &[Option<BusPublicInputs<FieldExtension>>],
) where
    FieldExtension: IsField,
{
    use std::collections::HashMap;

    let has_logup = all_bus_public_inputs.iter().any(|r| r.is_some());
    if !has_logup {
        return;
    }

    let mut global_bus_sums: HashMap<u64, FieldElement<FieldExtension>> = HashMap::new();
    let mut bus_senders: HashMap<u64, Vec<(String, FieldElement<FieldExtension>)>> = HashMap::new();
    let mut bus_receivers: HashMap<u64, Vec<(String, FieldElement<FieldExtension>)>> =
        HashMap::new();
    let mut global_sender_sums: HashMap<u64, FieldElement<FieldExtension>> = HashMap::new();
    let mut global_receiver_sums: HashMap<u64, FieldElement<FieldExtension>> = HashMap::new();

    for bus_inputs in all_bus_public_inputs.iter().flatten() {
        for (&bus_id, sum) in &bus_inputs.per_bus_sums {
            *global_bus_sums
                .entry(bus_id)
                .or_insert(FieldElement::zero()) += sum.clone();
        }
        for (&bus_id, sum) in &bus_inputs.per_bus_sender_sums {
            *global_sender_sums
                .entry(bus_id)
                .or_insert(FieldElement::zero()) += sum.clone();
            bus_senders
                .entry(bus_id)
                .or_default()
                .push((bus_inputs.table_name.clone(), sum.clone()));
        }
        for (&bus_id, sum) in &bus_inputs.per_bus_receiver_sums {
            *global_receiver_sums
                .entry(bus_id)
                .or_insert(FieldElement::zero()) += sum.clone();
            bus_receivers
                .entry(bus_id)
                .or_default()
                .push((bus_inputs.table_name.clone(), sum.clone()));
        }
    }

    eprintln!("\n=== GLOBAL BUS BALANCE REPORT ===");
    let zero = FieldElement::<FieldExtension>::zero();
    let mut bus_ids: Vec<_> = global_bus_sums.keys().copied().collect();
    bus_ids.sort();
    for bus_id in bus_ids {
        let total = &global_bus_sums[&bus_id];
        if *total != zero {
            eprintln!("Bus {:2}: IMBALANCED", bus_id);

            if let Some(senders) = bus_senders.get(&bus_id) {
                eprintln!("  SENDERS:");
                for (table_name, sum) in senders {
                    eprintln!("    [{:12}]: {:?}", table_name, sum);
                }
                if let Some(total_sent) = global_sender_sums.get(&bus_id) {
                    eprintln!("    → Total sent: {:?}", total_sent);
                }
            }

            if let Some(receivers) = bus_receivers.get(&bus_id) {
                eprintln!("  RECEIVERS:");
                for (table_name, sum) in receivers {
                    eprintln!("    [{:12}]: {:?}", table_name, sum);
                }
                if let Some(total_recv) = global_receiver_sums.get(&bus_id) {
                    eprintln!("    → Total received: {:?}", total_recv);
                }
            }

            eprintln!("  IMBALANCE: {:?}\n", total);
        } else {
            eprintln!("Bus {:2}: BALANCED ✓", bus_id);
        }
    }
    eprintln!("=================================\n");

    {
        use crate::bus_debug::BUS_DEBUG_TRACKER;

        let tracker = BUS_DEBUG_TRACKER
            .lock()
            .expect("[BusDebugTracker] mutex poisoned — debug data may be inconsistent");
        if !tracker.is_empty() {
            if tracker.is_truncated() {
                eprintln!(
                    "[BusDebugTracker] WARNING: Log truncated at {} entries — results may be incomplete",
                    tracker.len()
                );
            }
            eprintln!(
                "[BusDebugTracker] Logged {} interactions, running analysis...",
                tracker.len()
            );
            let report = tracker.analyze_mismatches();
            report.print_summary();
        }
    }
}
