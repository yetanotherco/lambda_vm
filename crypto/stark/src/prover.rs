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
use crate::proof::stark::{DeepPolynomialOpenings, PolynomialOpenings};
#[cfg(feature = "disk-spill")]
use crate::storage_mode::StorageMode;
use crate::table::Table;
use crate::trace::LDETraceTable;

use super::config::{BatchedMerkleTree, BatchedMerkleTreeBackend, Commitment};
use super::constraints::evaluator::ConstraintEvaluator;
use super::domain::{Domain, DomainConstants};
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

/// Commitment artifacts for one trace table (main or auxiliary). Used for both
/// plain and preprocessed tables. Preprocessed tables additionally carry a
/// separate Merkle tree over their precomputed columns, hence the optional
/// `precomputed_tree`/`precomputed_root` pair and the `num_precomputed_cols`
/// index used when opening positions.
pub(crate) struct TableCommit<F: IsField>
where
    FieldElement<F>: AsBytes,
{
    /// Merkle tree over the trace columns (multiplicities only for preprocessed tables).
    pub(crate) tree: Arc<BatchedMerkleTree<F>>,
    /// Root of `tree`.
    pub(crate) root: Commitment,
    /// Preprocessed tables only: Merkle tree over precomputed columns.
    pub(crate) precomputed_tree: Option<Arc<BatchedMerkleTree<F>>>,
    /// Preprocessed tables only: root of `precomputed_tree`.
    pub(crate) precomputed_root: Option<Commitment>,
    /// Preprocessed tables only: number of precomputed columns. Zero otherwise.
    pub(crate) num_precomputed_cols: usize,
}

impl<F: IsField> TableCommit<F>
where
    FieldElement<F>: AsBytes,
{
    /// Build a `TableCommit` for a plain (non-preprocessed) table.
    ///
    /// In streaming mode (`leaf_drop = true`) the tree's leaf half is freed
    /// before wrapping in `Arc` (the only point we still have `&mut` access).
    /// The leaf siblings needed at open time are regenerated from the
    /// recomputed LDE — see [`IsStarkProver::open_polys_with`].
    fn plain(mut tree: BatchedMerkleTree<F>, root: Commitment, leaf_drop: bool) -> Self {
        if leaf_drop {
            tree.drop_leaves();
        }
        Self {
            tree: Arc::new(tree),
            root,
            precomputed_tree: None,
            precomputed_root: None,
            num_precomputed_cols: 0,
        }
    }

    /// Build a `TableCommit` for a preprocessed table.
    fn preprocessed(
        mut tree: BatchedMerkleTree<F>,
        root: Commitment,
        mut precomputed_tree: BatchedMerkleTree<F>,
        precomputed_root: Commitment,
        num_precomputed_cols: usize,
        leaf_drop: bool,
    ) -> Self {
        if leaf_drop {
            tree.drop_leaves();
            precomputed_tree.drop_leaves();
        }
        Self {
            tree: Arc::new(tree),
            root,
            precomputed_tree: Some(Arc::new(precomputed_tree)),
            precomputed_root: Some(precomputed_root),
            num_precomputed_cols,
        }
    }

    /// Cheap clone. Only bumps Arc refcounts, no tree data is copied.
    fn share(&self) -> Self {
        Self {
            tree: Arc::clone(&self.tree),
            root: self.root,
            precomputed_tree: self.precomputed_tree.as_ref().map(Arc::clone),
            precomputed_root: self.precomputed_root,
            num_precomputed_cols: self.num_precomputed_cols,
        }
    }

    fn is_preprocessed(&self) -> bool {
        self.precomputed_tree.is_some()
    }
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
    /// Commitment to the main trace.
    pub(crate) main: TableCommit<Field>,
    /// Commitment to the auxiliary (RAP) trace, if any.
    pub(crate) aux: Option<TableCommit<FieldExtension>>,
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
    main: TableCommit<Field>,
    aux: Option<TableCommit<FieldExtension>>,
    rap_challenges: Vec<FieldElement<FieldExtension>>,
    bus_public_inputs: Option<BusPublicInputs<FieldExtension>>,
}

/// LDE columns for main (Phase A) and auxiliary (Phase C) traces, consumed by value in Phase D.
///
/// Memory trade-off: all N tables' LDE columns are live simultaneously between Phase A/C
/// and Phase D (O(N × cols × lde_size)).
struct Lde<Field: IsFFTField, FieldExtension: IsField> {
    main: Vec<Vec<FieldElement<Field>>>,
    aux: Vec<Vec<FieldElement<FieldExtension>>>,
}

impl<Field, FieldExtension> Round1Commitments<Field, FieldExtension>
where
    Field: IsFFTField + IsSubFieldOf<FieldExtension> + Send + Sync,
    FieldExtension: IsField + Send + Sync,
    FieldElement<Field>: AsBytes,
    FieldElement<FieldExtension>: AsBytes,
{
    /// Build a `Round1` by consuming a `Lde` and borrowing commitment data.
    /// The `TableCommit::share` calls are cheap — only bump Arc refcounts.
    fn build_round1(
        &self,
        lde: Lde<Field, FieldExtension>,
        step_size: usize,
        blowup_factor: usize,
    ) -> Round1<Field, FieldExtension> {
        Round1 {
            lde_trace: LDETraceTable::from_columns(lde.main, lde.aux, step_size, blowup_factor),
            main: self.main.share(),
            aux: self.aux.as_ref().map(TableCommit::share),
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

/// Streaming "retire-LDE" mode (Approach 1, Milestone 1).
///
/// When `true`, the main/aux LDE columns are dropped immediately after committing
/// (Phase A / Phase C) instead of being cached, and each table's LDE is recomputed
/// on demand in Rounds 2-4 via [`reconstruct_round1`]. This trades extra FFT work
/// (one LDE expansion per table) for a large drop in peak working memory: the
/// `O(N × cols × lde_size)` cache of all tables' LDE columns is no longer held
/// simultaneously between Phase A/C and Rounds 2-4 (only the resident traces and
/// the Merkle trees remain). No re-execution of the VM is involved — the LDE is
/// rebuilt from the still-resident trace.
///
/// # Leaf-drop (T1)
///
/// In this same mode the main and aux Merkle trees are additionally
/// *leaf-dropped*: after committing, [`MerkleTree::drop_leaves`] frees the leaf
/// half of each tree (about half its memory), keeping only the inner nodes and
/// root. At open time the single leaf-level sibling each query needs is
/// regenerated by re-hashing the recomputed LDE row (see
/// [`IsStarkProver::open_polys_with`] / [`keccak_leaf_from_row`]); all higher
/// path nodes are read from the retained inner nodes. The resulting `Proof`s are
/// byte-identical to the full-tree ones, so the verifier is unchanged.
///
/// The per-table *composition* Merkle tree (built transiently in Round 2 and
/// dropped at the end of that table's Rounds 2-4) is **left full** — it does not
/// contribute to the cross-table peak, so leaf-dropping it would add complexity
/// (row-pair leaf regeneration) for no peak-memory win. Its
/// `open_composition_poly` path therefore keeps using the full-tree opener.
///
/// Opt-in via `LAMBDA_STREAM_LDE=1` (or `true`). Default: `false` (cache for speed).
pub fn streaming_retire_lde() -> bool {
    std::env::var("LAMBDA_STREAM_LDE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// On-demand trace source for streaming "retire-traces" mode (C.2b).
///
/// In the default (non-streaming) path every table's main+aux trace is resident
/// in `air_trace_pairs` for the whole of [`IsStarkProver::multi_prove`], and no
/// provider is used (`None`). In the streaming path (`LAMBDA_STREAM_LDE=1`) the
/// caller may instead pass a provider that builds the *log-derived* tables'
/// traces on demand from a compact routed intermediate: the prover asks the
/// provider for table `idx`'s freshly-built **main-only** trace at each point it
/// is needed (Phase A main commit, Phase C aux build, Rounds 2-4 reconstruct),
/// uses it transiently, and drops it. The auxiliary trace is rebuilt by the
/// prover on top of the freshly-built main via `AIR::build_auxiliary_trace`.
///
/// Because the underlying build is deterministic (C.2a), the trace produced for
/// `idx` is byte-identical across all three phases (and to the pre-built trace
/// the non-streaming path uses), so the resulting proof is byte-identical.
///
/// Tables for which [`TraceProvider::is_retired`] returns `false` (preprocessed
/// tables, PAGE, etc.) are *not* built on demand: the prover uses the resident
/// trace borrowed in `air_trace_pairs` exactly as in the non-streaming path.
pub trait TraceProvider<Field, FieldExtension>: Sync
where
    Field: IsSubFieldOf<FieldExtension> + IsField,
    FieldExtension: IsField,
{
    /// Whether table `idx` is retired (built on demand) rather than resident.
    fn is_retired(&self, idx: usize) -> bool;

    /// Number of rows of the main trace for table `idx`. Cheap; used in the
    /// pre-pass to size the LDE domain without materializing the trace.
    fn num_rows(&self, idx: usize) -> usize;

    /// Build the **main-only** trace (no auxiliary columns) for retired table
    /// `idx`. Must be deterministic: byte-identical across repeated calls.
    fn build_main(&self, idx: usize) -> TraceTable<Field, FieldExtension>;
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
pub(crate) struct Round2<F>
where
    F: IsField,
    FieldElement<F>: AsBytes,
{
    /// Evaluations of the composition polynomial parts over the LDE domain.
    pub(crate) lde_composition_poly_evaluations: Vec<Vec<FieldElement<F>>>,
    /// The Merkle tree built to compute the commitment to the composition polynomial parts.
    pub(crate) composition_poly_merkle_tree: BatchedMerkleTree<F>,
    /// The commitment to the composition polynomial parts.
    pub(crate) composition_poly_root: Commitment,
}

/// A container for the results of the third round of the STARK Prove protocol.
pub(crate) struct Round3<F: IsField> {
    /// Evaluations of the trace polynomials, main ans auxiliary, at the out-of-domain challenge.
    trace_ood_evaluations: Table<F>,
    /// Evaluations of the composition polynomial parts at the out-of-domain challenge.
    composition_poly_parts_ood_evaluation: Vec<FieldElement<F>>,
}

/// DEEP composition coefficients derived from the challenge 𝛾, sampled at the end
/// of Rounds 2-3 so Round 4 only builds the DEEP LDE and runs FRI.
pub(crate) struct DeepCoeffs<E: IsField> {
    /// Coefficients for the composition-polynomial-part terms.
    gammas: Vec<FieldElement<E>>,
    /// Per-trace-column coefficient chunks for the trace terms.
    trace_term_coeffs: Vec<Vec<FieldElement<E>>>,
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

/// Hash a single already-gathered LDE row into its leaf [`Commitment`], using
/// the exact byte layout and hasher of [`keccak_leaves_bit_reversed`]: BE
/// concatenation of the row's columns, then `hash_bytes`.
///
/// Used by the streaming leaf-drop opener to regenerate the one leaf-level
/// sibling needed per opened position (the rest of the path is read from the
/// retained inner nodes). Producing a different digest here than the build-time
/// `keccak_leaves_bit_reversed` would yield a wrong (non-byte-identical) proof.
pub fn keccak_leaf_from_row<E>(row: &[FieldElement<E>]) -> Commitment
where
    E: IsField,
    FieldElement<E>: AsBytes + ByteConversion,
{
    let byte_len = <FieldElement<E> as ByteConversion>::BYTE_LEN;
    let mut buf = vec![0u8; row.len() * byte_len];
    for (col_idx, value) in row.iter().enumerate() {
        value.write_bytes_be(&mut buf[col_idx * byte_len..(col_idx + 1) * byte_len]);
    }
    BatchedMerkleTreeBackend::<E>::hash_bytes(&buf)
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

    /// Compute the main-trace LDE and commit. Returns a `TableCommit` along
    /// with the owned LDE columns (consumed later in Phase D).
    ///
    /// `precomputed`: if present, the leading `num_cols` columns are committed
    /// as a separate Merkle tree (the precomputed split for preprocessed
    /// tables) and the root is checked against the AIR-hardcoded commitment.
    #[allow(clippy::type_complexity)]
    fn commit_main_trace(
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
        precomputed: Option<(Commitment, usize)>,
        leaf_drop: bool,
        #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
    ) -> Result<(TableCommit<Field>, Vec<Vec<FieldElement<Field>>>), ProvingError>
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

        let commit = match precomputed {
            None => {
                #[allow(unused_mut)]
                let (mut tree, root) = Self::commit_columns_bit_reversed(&columns)
                    .ok_or(ProvingError::EmptyCommitment)?;
                #[cfg(feature = "disk-spill")]
                if storage_mode == StorageMode::Disk {
                    tree.spill_nodes_to_disk()
                        .map_err(|e| ProvingError::DiskSpill(format!("main Merkle tree: {e}")))?;
                }
                TableCommit::plain(tree, root, leaf_drop)
            }
            Some((expected_precomputed_root, num_cols)) => {
                #[allow(unused_mut)]
                let (mut precomputed_tree, precomputed_root) =
                    Self::commit_columns_bit_reversed(&columns[..num_cols])
                        .ok_or(ProvingError::EmptyCommitment)?;
                #[allow(unused_mut)]
                let (mut mult_tree, mult_root) =
                    Self::commit_columns_bit_reversed(&columns[num_cols..])
                        .ok_or(ProvingError::EmptyCommitment)?;
                debug_assert_eq!(
                    precomputed_root, expected_precomputed_root,
                    "Prover's precomputed commitment doesn't match hardcoded AIR commitment"
                );
                #[cfg(feature = "disk-spill")]
                if storage_mode == StorageMode::Disk {
                    precomputed_tree.spill_nodes_to_disk().map_err(|e| {
                        ProvingError::DiskSpill(format!("precomputed Merkle tree: {e}"))
                    })?;
                    mult_tree
                        .spill_nodes_to_disk()
                        .map_err(|e| ProvingError::DiskSpill(format!("mult Merkle tree: {e}")))?;
                }
                TableCommit::preprocessed(
                    mult_tree,
                    mult_root,
                    precomputed_tree,
                    precomputed_root,
                    num_cols,
                    leaf_drop,
                )
            }
        };

        #[cfg(feature = "instruments")]
        crate::instruments::accum_r1_main(main_lde_dur, t_sub.elapsed());

        Ok((commit, columns))
    }

    /// Recompute Round1 from the trace, reusing the Merkle trees stored in commitments.
    ///
    /// Used by `run_debug_checks` and by streaming "retire-LDE" mode
    /// ([`streaming_retire_lde`]), where the cached LDE was dropped after commit
    /// and is rebuilt on demand from the still-resident trace.
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

        Ok(commitment.build_round1(Lde { main, aux }, air.step_size(), domain.blowup_factor))
    }

    /// Reconstruct Round1 for every table, print the bus balance report, and
    /// validate each trace. Called once after Phase C commits.
    #[cfg(feature = "debug-checks")]
    #[allow(clippy::too_many_arguments)]
    fn run_debug_checks(
        air_trace_pairs: &[AirTracePair<'_, Field, FieldExtension, PI>],
        commitments: &[Round1Commitments<Field, FieldExtension>],
        domains: &[Arc<Domain<Field>>],
        twiddle_caches: &[Arc<LdeTwiddles<Field>>],
        provider: Option<&dyn TraceProvider<Field, FieldExtension>>,
        stream_traces: bool,
        lookup_challenges: &[FieldElement<FieldExtension>],
    ) where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        PI: Send + Sync + Clone,
    {
        // For retired tables the resident trace is an empty placeholder; rebuild
        // each retired table's main+aux trace here so debug checks see the real
        // data (matching what the proving rounds reconstruct).
        let rebuilt: Vec<Option<TraceTable<Field, FieldExtension>>> = air_trace_pairs
            .iter()
            .enumerate()
            .map(|(idx, (air, _, _))| {
                if stream_traces && provider.is_some_and(|p| p.is_retired(idx)) {
                    let mut t = provider.unwrap().build_main(idx);
                    if air.has_aux_trace() {
                        air.build_auxiliary_trace(&mut t, lookup_challenges);
                    }
                    Some(t)
                } else {
                    None
                }
            })
            .collect();
        let trace_at = |idx: usize| -> &TraceTable<Field, FieldExtension> {
            match &rebuilt[idx] {
                Some(t) => t,
                None => air_trace_pairs[idx].1,
            }
        };

        let mut temp_results: Vec<Round1<Field, FieldExtension>> =
            Vec::with_capacity(air_trace_pairs.len());
        for (idx, (((air, _, _), commitment), (domain, twiddles))) in air_trace_pairs
            .iter()
            .zip(commitments.iter())
            .zip(domains.iter().zip(twiddle_caches.iter()))
            .enumerate()
        {
            let result =
                Self::reconstruct_round1(*air, trace_at(idx), domain, commitment, twiddles)
                    .expect("reconstruct_round1 failed in debug-checks");
            temp_results.push(result);
        }

        let all_bus_public_inputs: Vec<Option<BusPublicInputs<FieldExtension>>> = temp_results
            .iter()
            .map(|r| r.bus_public_inputs.clone())
            .collect();
        print_bus_balance_report(&all_bus_public_inputs);

        for (idx, (((air, _, pub_inputs), round_1_result), domain)) in air_trace_pairs
            .iter()
            .zip(temp_results.iter())
            .zip(domains.iter())
            .enumerate()
        {
            validate_trace(
                *air,
                *pub_inputs,
                trace_at(idx),
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

    /// Returns the result of the second round of the STARK Prove protocol.
    fn round_2_compute_composition_polynomial(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        pub_inputs: &PI,
        domain: &Domain<Field>,
        round_1_result: &Round1<Field, FieldExtension>,
        transition_coefficients: &[FieldElement<FieldExtension>],
        boundary_coefficients: &[FieldElement<FieldExtension>],
    ) -> Result<Round2<FieldExtension>, ProvingError>
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
        let Some((composition_poly_merkle_tree, composition_poly_root)) =
            Self::commit_composition_polynomial(&lde_composition_poly_parts_evaluations)
        else {
            return Err(ProvingError::EmptyCommitment);
        };
        #[cfg(feature = "instruments")]
        let merkle_dur = t_sub.elapsed();

        #[cfg(feature = "instruments")]
        crate::instruments::store_r2_sub(constraints_dur, fft_dur, merkle_dur);

        Ok(Round2 {
            lde_composition_poly_evaluations: lde_composition_poly_parts_evaluations,
            composition_poly_merkle_tree,
            composition_poly_root,
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
    /// Sample the DEEP composition challenge 𝛾 and expand it into the per-trace-term
    /// and per-composition-part coefficients. Done at the end of Rounds 2-3 (from the
    /// table's fork) so Round 4 only builds the DEEP LDE and runs FRI.
    fn sample_deep_coeffs(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        round_2_result: &Round2<FieldExtension>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> DeepCoeffs<FieldExtension>
    where
        FieldElement<FieldExtension>: AsBytes,
        FieldElement<Field>: AsBytes,
    {
        let gamma = transcript.sample_field_element();

        let n_terms_composition_poly = round_2_result.lde_composition_poly_evaluations.len();
        let num_terms_trace =
            air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;

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

        let gammas = deep_composition_coefficients;

        DeepCoeffs {
            gammas,
            trace_term_coeffs,
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

    /// Computes values and validity proofs of the evaluations of the composition polynomial parts
    /// at the domain value corresponding to the FRI query challenge `index` and its symmetric
    /// element.
    fn open_composition_poly(
        composition_poly_merkle_tree: &BatchedMerkleTree<FieldExtension>,
        lde_composition_poly_evaluations: &[Vec<FieldElement<FieldExtension>>],
        index: usize,
    ) -> PolynomialOpenings<FieldExtension>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let proof = composition_poly_merkle_tree
            .get_proof_by_pos(index)
            .unwrap();

        let lde_composition_poly_parts_evaluation: Vec<_> = lde_composition_poly_evaluations
            .iter()
            .flat_map(|part| {
                vec![
                    part[reverse_index(index * 2, part.len() as u64)].clone(),
                    part[reverse_index(index * 2 + 1, part.len() as u64)].clone(),
                ]
            })
            .collect();

        PolynomialOpenings {
            proof: proof.clone(),
            proof_sym: proof,
            evaluations: lde_composition_poly_parts_evaluation
                .clone()
                .into_iter()
                .step_by(2)
                .collect(),
            evaluations_sym: lde_composition_poly_parts_evaluation
                .into_iter()
                .skip(1)
                .step_by(2)
                .collect(),
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
        FieldElement<C>: AsBytes + Sync + Send + ByteConversion,
        G: Fn(usize) -> Vec<FieldElement<C>>,
    {
        let domain_size = domain.lde_roots_of_unity_coset.len() as u64;
        let index = challenge * 2;
        let index_sym = challenge * 2 + 1;
        // `proof_for` produces the same `Proof` whether or not the tree's leaves
        // were dropped: with a full tree we read the stored leaf sibling; with a
        // leaf-dropped tree (streaming mode) we regenerate that one sibling leaf
        // by re-hashing the LDE row at `sibling_leaf_position`, matching the
        // build-time `keccak_leaves_bit_reversed` byte layout exactly.
        let proof_for = |pos: usize| {
            if tree.leaves_dropped() {
                let sib_leaf_pos = tree.sibling_leaf_position(pos);
                let sib_row = gather(reverse_index(sib_leaf_pos, domain_size));
                let sibling = keccak_leaf_from_row(&sib_row);
                tree.get_proof_by_pos_with_leaf_sibling(pos, sibling)
                    .unwrap()
            } else {
                tree.get_proof_by_pos(pos).unwrap()
            }
        };
        PolynomialOpenings {
            proof: proof_for(index),
            proof_sym: proof_for(index_sym),
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
        let is_preprocessed = main_commit.is_preprocessed();
        let num_precomputed_cols = main_commit.num_precomputed_cols;
        let total_cols = lde_trace.num_main_cols();

        for index in indexes_to_open.iter() {
            // For preprocessed tables, open the main split (multiplicities only);
            // for normal tables, open all main columns.
            let main_trace_opening = if is_preprocessed {
                Self::open_polys_with(domain, &main_commit.tree, *index, |row| {
                    lde_trace.gather_main_row_range(row, num_precomputed_cols, total_cols)
                })
            } else {
                Self::open_polys_with(domain, &main_commit.tree, *index, |row| {
                    lde_trace.gather_main_row(row)
                })
            };

            // For preprocessed tables, also open the precomputed-columns tree.
            let precomputed_trace_opening = main_commit.precomputed_tree.as_ref().map(|tree| {
                Self::open_polys_with(domain, tree, *index, |row| {
                    lde_trace.gather_main_row_range(row, 0, num_precomputed_cols)
                })
            });

            let composition_openings = Self::open_composition_poly(
                &round_2_result.composition_poly_merkle_tree,
                &round_2_result.lde_composition_poly_evaluations,
                *index,
            );

            let aux_trace_polys = round_1_result.aux.as_ref().map(|aux| {
                Self::open_polys_with(domain, &aux.tree, *index, |row| {
                    lde_trace.gather_aux_row(row)
                })
            });

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
        air_trace_pairs: Vec<AirTracePair<'_, Field, FieldExtension, PI>>,
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
        Self::multi_prove_with_provider(
            air_trace_pairs,
            None,
            transcript,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        )
    }

    /// Like [`IsStarkProver::multi_prove`], but with an optional on-demand trace
    /// [`TraceProvider`] for streaming "retire-traces" mode (C.2b).
    ///
    /// When `provider` is `None` this is exactly [`IsStarkProver::multi_prove`].
    /// When `provider` is `Some` *and* [`streaming_retire_lde`] is enabled, the
    /// log-derived tables (those for which the provider reports `is_retired`) are
    /// built on demand from the provider at each phase (main commit, aux build,
    /// reconstruct) and dropped afterwards, so the resident traces in
    /// `air_trace_pairs` for those indices may be empty placeholders. Tables the
    /// provider does not retire (preprocessed, PAGE) use their resident traces.
    fn multi_prove_with_provider(
        mut air_trace_pairs: Vec<AirTracePair<'_, Field, FieldExtension, PI>>,
        provider: Option<&dyn TraceProvider<Field, FieldExtension>>,
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

        // Streaming "retire-traces" is active only when a provider is supplied
        // AND the streaming env flag is on. Off-path (`provider == None`) every
        // expression below collapses to today's resident-trace behaviour.
        let stream_traces = provider.is_some() && streaming_retire_lde();
        // Returns the main-trace row count for table `idx`, sourced from the
        // provider for retired tables (whose resident placeholder is empty) and
        // from the resident trace otherwise.
        let rows_of = |idx: usize| -> usize {
            match provider {
                Some(p) if stream_traces && p.is_retired(idx) => p.num_rows(idx),
                _ => air_trace_pairs[idx].1.num_rows(),
            }
        };

        for (idx, (air, _trace, _pub_inputs)) in air_trace_pairs.iter().enumerate() {
            let trace_length = rows_of(idx);
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
        let stream_lde = streaming_retire_lde();

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

        let mut main_commits: Vec<TableCommit<Field>> = Vec::with_capacity(num_airs);
        let mut main_ldes: Vec<Vec<Vec<FieldElement<Field>>>> = Vec::with_capacity(num_airs);

        for chunk_start in (0..num_airs).step_by(k) {
            let chunk_end = (chunk_start + k).min(num_airs);
            let chunk_range = chunk_start..chunk_end;

            #[cfg(feature = "parallel")]
            let iter = chunk_range.into_par_iter();
            #[cfg(not(feature = "parallel"))]
            let iter = chunk_range;

            let chunk_results: Vec<Result<_, ProvingError>> = iter
                .map(|idx| {
                    let (air, trace, _) = &air_trace_pairs[idx];
                    let domain = &domains[idx];
                    let twiddles = &twiddle_caches[idx];

                    // Retire-traces: build this table's main trace on demand and
                    // commit it, then drop it at the end of this closure. Other
                    // tables (preprocessed/PAGE, or non-streaming) use the
                    // resident trace borrowed above.
                    let retired_main;
                    let main_trace: &TraceTable<Field, FieldExtension> =
                        if stream_traces && provider.is_some_and(|p| p.is_retired(idx)) {
                            retired_main = provider.unwrap().build_main(idx);
                            &retired_main
                        } else {
                            trace
                        };

                    let precomputed = air
                        .is_preprocessed()
                        .then(|| (air.precomputed_commitment(), air.num_precomputed_columns()));
                    Self::commit_main_trace(
                        main_trace,
                        domain,
                        twiddles,
                        precomputed,
                        stream_lde,
                        #[cfg(feature = "disk-spill")]
                        storage_mode,
                    )
                })
                .collect();

            // Sequential: append roots to shared transcript (Fiat-Shamir ordering)
            for result in chunk_results {
                let (commit, cached_main) = result?;
                if let Some(ref pre_root) = commit.precomputed_root {
                    transcript.append_bytes(pre_root);
                }
                transcript.append_bytes(&commit.root);
                main_commits.push(commit);
                // Streaming retire-LDE: drop the main LDE now; Rounds 2-4 recompute
                // it on demand from the resident trace.
                main_ldes.push(if stream_lde { Vec::new() } else { cached_main });
            }
        }

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

        // For retired tables the aux trace is built on demand inside the Pass-2
        // loop (so it can be committed and dropped per table); their bus inputs
        // are filled there. Non-retired tables build aux into their resident
        // trace here as before.
        #[cfg(feature = "parallel")]
        let aux_iter = air_trace_pairs.par_iter_mut().enumerate();
        #[cfg(not(feature = "parallel"))]
        let aux_iter = air_trace_pairs.iter_mut().enumerate();
        let mut bus_inputs_vec: Vec<Option<BusPublicInputs<FieldExtension>>> = aux_iter
            .map(|(idx, (air, trace, _))| {
                let retired = stream_traces && provider.is_some_and(|p| p.is_retired(idx));
                if air.has_aux_trace() && !retired {
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

        // Pass 2: Parallel fork transcript → extract → LDE → commit in chunks of K.
        // Each table gets its own transcript fork.
        #[cfg(feature = "instruments")]
        let phase_start = Instant::now();

        // Capture the pre-fork shared transcript state. Phase D (batched FRI)
        // clones this per chunk and replays chunk-local data
        // (table_contributions, composition roots, all chunk-mate OOD
        // evaluations) canonically to derive each bucket's `delta_fri` and
        // query iotas. The verifier reconstructs an identical seed from proof
        // data only. This is the shared state after Phase B (LogUp challenges
        // sampled), before any per-table fork — it does NOT include per-table
        // aux roots (those live only in the per-table forks below).
        let pre_fork_transcript = transcript.clone();

        // Pre-fork all transcripts (cheap, sequential — must match verifier ordering)
        let mut table_transcripts: Vec<_> = (0..num_airs)
            .map(|idx| {
                let mut t = transcript.clone();
                if num_airs > 1 {
                    t.append_bytes(&(idx as u64).to_le_bytes());
                }
                t
            })
            .collect();

        // Parallel aux commit in chunks of K. Each entry holds the optional aux
        // `TableCommit` (`None` when the AIR has no aux trace) and the cached
        // aux LDE columns consumed in Phase D.
        #[allow(clippy::type_complexity)]
        let mut aux_results: Vec<(
            Option<TableCommit<FieldExtension>>,
            Vec<Vec<FieldElement<FieldExtension>>>,
        )> = Vec::with_capacity(num_airs);

        for chunk_start in (0..num_airs).step_by(k) {
            let chunk_end = (chunk_start + k).min(num_airs);
            let chunk_range = chunk_start..chunk_end;

            #[cfg(feature = "parallel")]
            let iter = chunk_range.into_par_iter();
            #[cfg(not(feature = "parallel"))]
            let iter = chunk_range;

            let chunk_aux: Vec<Result<_, ProvingError>> = iter
                .map(|idx| {
                    let (air, trace, _) = &air_trace_pairs[idx];
                    let domain = &domains[idx];
                    let twiddles = &twiddle_caches[idx];

                    // Retire-traces: rebuild this table's main trace, build its
                    // aux on top (the resident trace is an empty placeholder), and
                    // capture the bus inputs that Pass 1 skipped. The owned trace
                    // (main+aux) is dropped at the end of this closure. Other
                    // tables read aux columns from their resident trace.
                    let retired = stream_traces && provider.is_some_and(|p| p.is_retired(idx));
                    let mut retired_bus: Option<BusPublicInputs<FieldExtension>> = None;
                    let retired_trace;
                    let trace: &TraceTable<Field, FieldExtension> = if retired {
                        let mut t = provider.unwrap().build_main(idx);
                        if air.has_aux_trace() {
                            retired_bus = air.build_auxiliary_trace(&mut t, &lookup_challenges);
                        }
                        retired_trace = t;
                        &retired_trace
                    } else {
                        trace
                    };

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
                        #[allow(unused_mut)]
                        let (mut tree, root) = Self::commit_columns_bit_reversed(&columns)
                            .ok_or(ProvingError::EmptyCommitment)?;
                        #[cfg(feature = "instruments")]
                        crate::instruments::accum_r1_aux(aux_lde_dur, t_sub.elapsed());

                        #[cfg(feature = "disk-spill")]
                        if storage_mode == StorageMode::Disk {
                            tree.spill_nodes_to_disk().map_err(|e| {
                                ProvingError::DiskSpill(format!("aux Merkle tree: {e}"))
                            })?;
                        }
                        Ok((Some(TableCommit::plain(tree, root, stream_lde)), columns, retired_bus))
                    } else {
                        Ok((None, Vec::new(), retired_bus))
                    }
                })
                .collect();

            // Sequential: append aux roots to forked transcripts
            for (j, result) in chunk_aux.into_iter().enumerate() {
                let (aux_commit, cached_aux, retired_bus) = result?;
                if let Some(ref c) = aux_commit {
                    table_transcripts[chunk_start + j].append_bytes(&c.root);
                }
                // Retired tables compute their bus inputs here (Pass 1 skipped them).
                if retired_bus.is_some() {
                    bus_inputs_vec[chunk_start + j] = retired_bus;
                }
                aux_results.push((aux_commit, if stream_lde { Vec::new() } else { cached_aux }));
            }
        }

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
        Self::run_debug_checks(
            &air_trace_pairs,
            &commitments,
            &domains,
            &twiddle_caches,
            provider,
            stream_traces,
            &lookup_challenges,
        );

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

        let mut proofs = Vec::with_capacity(num_airs);
        // Per-(chunk, lde_size-bucket) batched FRI instances, outer index = chunk.
        let mut fri_chunk_buckets: Vec<Vec<crate::proof::stark::ChunkBucketFri<FieldExtension>>> =
            Vec::with_capacity(num_airs.div_ceil(k));
        let mut lde_drain = cached_ldes.into_iter();
        for chunk_start in (0..num_airs).step_by(k) {
            let chunk_end = (chunk_start + k).min(num_airs);
            let chunk_size = chunk_end - chunk_start;

            let chunk_ldes: Vec<Lde<Field, FieldExtension>> =
                lde_drain.by_ref().take(chunk_size).collect();
            let chunk_commitments = &commitments[chunk_start..chunk_end];

            // ---- Pass 1: per-table Rounds 2-3 ----------------------------------
            // Advance each table's forked transcript through the OOD evaluations.
            // No FRI yet: the intermediate results are collected so Round 4 can run
            // as a separate per-chunk phase (which a later step batches across the
            // chunk's tables into a single FRI per lde_size bucket).
            let pass1: Vec<Result<_, ProvingError>> = {
                let chunk_transcripts = &mut table_transcripts[chunk_start..chunk_end];

                #[cfg(feature = "parallel")]
                let iter = chunk_ldes
                    .into_par_iter()
                    .zip(chunk_commitments.par_iter())
                    .zip(chunk_transcripts.par_iter_mut())
                    .enumerate();
                #[cfg(not(feature = "parallel"))]
                let iter = chunk_ldes
                    .into_iter()
                    .zip(chunk_commitments.iter())
                    .zip(chunk_transcripts.iter_mut())
                    .enumerate();

                iter.map(|(j, ((lde, commitment), table_transcript))| {
                    let idx = chunk_start + j;
                    let (air, trace, pub_inputs) = &air_trace_pairs[idx];
                    let domain = &domains[idx];

                    #[cfg(feature = "instruments")]
                    let table_start = Instant::now();

                    // Build Round1 from the cached LDE (consumed by value), or, in
                    // streaming retire-LDE mode, recompute it from the trace. The
                    // trace is the resident one unless this table is retired, in
                    // which case it is rebuilt on demand (main + aux) and dropped
                    // after the reconstruct. Determinism (C.2a) makes this trace
                    // byte-identical to the Phase A/C builds, so the LDE matches.
                    let round_1_result = if stream_lde {
                        let _ = lde; // empty placeholder when streaming
                        let retired =
                            stream_traces && provider.is_some_and(|p| p.is_retired(idx));
                        let retired_trace;
                        let recon_trace: &TraceTable<Field, FieldExtension> = if retired {
                            let mut t = provider.unwrap().build_main(idx);
                            if air.has_aux_trace() {
                                air.build_auxiliary_trace(&mut t, &lookup_challenges);
                            }
                            retired_trace = t;
                            &retired_trace
                        } else {
                            trace
                        };
                        Self::reconstruct_round1(
                            *air,
                            recon_trace,
                            domain,
                            commitment,
                            &twiddle_caches[idx],
                        )?
                    } else {
                        commitment.build_round1(lde, air.step_size(), domain.blowup_factor)
                    };

                    if let Some(ref bpi) = round_1_result.bus_public_inputs {
                        table_transcript.append_field_element(&bpi.table_contribution);
                    }

                    let (round_2_result, round_3_result, z, deep_coeffs) = Self::prove_rounds_2_to_3(
                        *air,
                        *pub_inputs,
                        &round_1_result,
                        table_transcript,
                        domain,
                    )?;

                    #[cfg(feature = "instruments")]
                    let instr1 = {
                        let zero = std::time::Duration::ZERO;
                        (
                            air.name().to_string(),
                            trace.num_rows(),
                            table_start.elapsed(),
                            crate::instruments::take_r2_sub().unwrap_or((zero, zero, zero)),
                            crate::instruments::take_r3_ood().unwrap_or(zero),
                        )
                    };

                    #[cfg(feature = "instruments")]
                    return Ok((round_1_result, round_2_result, round_3_result, z, deep_coeffs, instr1));
                    #[cfg(not(feature = "instruments"))]
                    Ok((round_1_result, round_2_result, round_3_result, z, deep_coeffs))
                })
                .collect()
            };
            let intermediates = pass1.into_iter().collect::<Result<Vec<_>, ProvingError>>()?;

            // ---- Pass 2: per-(chunk, lde_size) batched FRI --------------------
            // Group the chunk's tables into lde_size buckets. For each bucket,
            // derive a single `delta_fri` from a shared `bucket_seed`, fold each
            // member's DEEP-composition LDE with successive powers of `delta_fri`
            // into one polynomial, run ONE FRI commit + grinding + query, and
            // produce per-table DEEP openings at the bucket-shared iotas.
            //
            // The per-table forks (`table_transcripts`) are NOT advanced past the
            // OOD evaluations here: FRI challenges come exclusively from the
            // chunk-shared `bucket_seed` below, so prover and verifier derive
            // byte-identical `delta_fri`/iotas.

            // Unpack `intermediates` (chunk-local order) into parallel vectors.
            #[cfg(feature = "instruments")]
            let mut chunk_instr1: Vec<_> = Vec::with_capacity(chunk_size);
            let mut chunk_round1: Vec<Round1<Field, FieldExtension>> =
                Vec::with_capacity(chunk_size);
            let mut chunk_round2: Vec<Round2<FieldExtension>> = Vec::with_capacity(chunk_size);
            let mut chunk_round3: Vec<Round3<FieldExtension>> = Vec::with_capacity(chunk_size);
            let mut chunk_z: Vec<FieldElement<FieldExtension>> = Vec::with_capacity(chunk_size);
            let mut chunk_deep_coeffs: Vec<DeepCoeffs<FieldExtension>> =
                Vec::with_capacity(chunk_size);
            for intermediate in intermediates {
                #[cfg(feature = "instruments")]
                let (round_1_result, round_2_result, round_3_result, z, deep_coeffs, instr1) =
                    intermediate;
                #[cfg(not(feature = "instruments"))]
                let (round_1_result, round_2_result, round_3_result, z, deep_coeffs) = intermediate;
                chunk_round1.push(round_1_result);
                chunk_round2.push(round_2_result);
                chunk_round3.push(round_3_result);
                chunk_z.push(z);
                chunk_deep_coeffs.push(deep_coeffs);
                #[cfg(feature = "instruments")]
                chunk_instr1.push(instr1);
            }

            // ---- Build the chunk-shared bucket seed ----------------------------
            // bucket_seed byte order (must match the verifier exactly):
            //   1. pre-fork shared transcript state (after Phase B)
            //   2. for each chunk-local j (ascending): table_contribution (if any)
            //   3. for each chunk-local j (ascending): composition_poly_root
            //   4. for each chunk-local j (ascending): all trace_ood_evaluations
            //      columns (column-major), then composition_poly_parts_ood
            let mut bucket_seed = pre_fork_transcript.clone();
            for r1 in chunk_round1.iter() {
                if let Some(ref bpi) = r1.bus_public_inputs {
                    bucket_seed.append_field_element(&bpi.table_contribution);
                }
            }
            for r2 in chunk_round2.iter() {
                bucket_seed.append_bytes(&r2.composition_poly_root);
            }
            for r3 in chunk_round3.iter() {
                for col in r3.trace_ood_evaluations.columns().iter() {
                    for elem in col.iter() {
                        bucket_seed.append_field_element(elem);
                    }
                }
                for elem in r3.composition_poly_parts_ood_evaluation.iter() {
                    bucket_seed.append_field_element(elem);
                }
            }

            // ---- Bucket by lde_size (first-encounter order) --------------------
            let mut bucket_members: Vec<Vec<usize>> = Vec::new();
            let mut bucket_lde_sizes: Vec<usize> = Vec::new();
            for j in 0..chunk_size {
                let lde_size = domains[chunk_start + j].lde_roots_of_unity_coset.len();
                match bucket_lde_sizes.iter().position(|&s| s == lde_size) {
                    Some(b) => bucket_members[b].push(j),
                    None => {
                        bucket_lde_sizes.push(lde_size);
                        bucket_members.push(vec![j]);
                    }
                }
            }

            let mut chunk_buckets: Vec<crate::proof::stark::ChunkBucketFri<FieldExtension>> =
                Vec::with_capacity(bucket_members.len());
            // Per chunk-local index: the bucket-shared iotas used for openings.
            let mut iotas_for: Vec<Option<Vec<usize>>> = (0..chunk_size).map(|_| None).collect();
            #[cfg(feature = "instruments")]
            let mut fri_sub_for: Vec<
                Option<(std::time::Duration, std::time::Duration, std::time::Duration)>,
            > = (0..chunk_size).map(|_| None).collect();

            for (members, &lde_size) in bucket_members.iter().zip(bucket_lde_sizes.iter()) {
                let mut bt = bucket_seed.clone();
                bt.append_bytes(&(lde_size as u64).to_le_bytes());
                let delta_fri: FieldElement<FieldExtension> = bt.sample_field_element();

                let leader_idx = chunk_start + members[0];
                let (leader_air, _, _) = &air_trace_pairs[leader_idx];
                let leader_domain = &domains[leader_idx];
                let coset_offset = FieldElement::<Field>::from(
                    leader_air.context().proof_options.coset_offset,
                );

                // Streaming bucket combine: build each member's DEEP LDE one at a
                // time, fold into the accumulator with delta_fri^i, then drop.
                // Peak DEEP memory inside this loop: 2 × |LDE|.
                #[cfg(feature = "instruments")]
                let mut deep_comp_dur = std::time::Duration::ZERO;
                #[cfg(feature = "instruments")]
                let mut deep_extend_dur = std::time::Duration::ZERO;
                let mut combined: Vec<FieldElement<FieldExtension>> = Vec::new();
                let mut delta_power = FieldElement::<FieldExtension>::one();
                for (i_local, &j) in members.iter().enumerate() {
                    let idx = chunk_start + j;
                    let domain_j = &domains[idx];

                    #[cfg(feature = "instruments")]
                    let t_sub = Instant::now();
                    let deep_evals = Self::compute_deep_composition_poly_evaluations(
                        &chunk_round1[j].lde_trace,
                        &chunk_round2[j],
                        &chunk_round3[j],
                        &chunk_z[j],
                        domain_j,
                        &domain_j.trace_primitive_root,
                        &chunk_deep_coeffs[j].gammas,
                        &chunk_deep_coeffs[j].trace_term_coeffs,
                    );
                    #[cfg(feature = "instruments")]
                    {
                        deep_comp_dur += t_sub.elapsed();
                    }

                    // DEEP evaluations are already at the LDE points; bit-reverse
                    // for FRI (no FFT extension needed).
                    #[cfg(feature = "instruments")]
                    let t_sub = Instant::now();
                    let mut deep_lde = deep_evals;
                    in_place_bit_reverse_permute(&mut deep_lde);
                    #[cfg(feature = "instruments")]
                    {
                        deep_extend_dur += t_sub.elapsed();
                    }

                    debug_assert_eq!(deep_lde.len(), lde_size);
                    if i_local == 0 {
                        // First member: assign directly to avoid mul-by-one.
                        combined = deep_lde;
                    } else {
                        for (acc, src) in combined.iter_mut().zip(deep_lde.iter()) {
                            *acc = &*acc + &delta_power * src;
                        }
                    }
                    delta_power = &delta_power * &delta_fri;
                }

                #[cfg(feature = "instruments")]
                let t_sub = Instant::now();
                let (last_value, fri_layers) =
                    fri::commit_phase_from_evaluations::<Field, FieldExtension>(
                        leader_domain.root_order as usize,
                        combined,
                        &mut bt,
                        &coset_offset,
                        lde_size,
                    );
                #[cfg(feature = "instruments")]
                let fri_commit_dur = t_sub.elapsed();

                // grinding: generate nonce and append it to the bucket transcript.
                let security_bits = leader_air.context().proof_options.grinding_factor;
                let nonce = if security_bits > 0 {
                    let nonce_value = grinding::generate_nonce(&bt.state(), security_bits)
                        .expect("bucket-FRI grinding nonce not found");
                    bt.append_bytes(&nonce_value.to_be_bytes());
                    Some(nonce_value)
                } else {
                    None
                };

                let number_of_queries = leader_air.options().fri_number_of_queries;
                #[cfg(feature = "instruments")]
                let t_sub = Instant::now();
                let iotas =
                    Self::sample_query_indexes(number_of_queries, leader_domain, &mut bt);
                let decommitments = fri::query_phase(&fri_layers, &iotas);
                #[cfg(feature = "instruments")]
                let queries_dur = t_sub.elapsed();
                let layer_roots: Vec<Commitment> = fri_layers
                    .iter()
                    .map(|layer| layer.merkle_tree.root)
                    .collect();

                chunk_buckets.push(crate::proof::stark::ChunkBucketFri {
                    lde_size: lde_size as u32,
                    members: members.clone(),
                    layer_roots,
                    last_value,
                    decommitments,
                    nonce,
                });

                for &j in members.iter() {
                    iotas_for[j] = Some(iotas.clone());
                }
                // Attribute the bucket's FRI sub-timings to the leader member.
                #[cfg(feature = "instruments")]
                {
                    fri_sub_for[members[0]] = Some((
                        deep_comp_dur,
                        deep_extend_dur,
                        fri_commit_dur + queries_dur,
                    ));
                    for &j in members.iter().skip(1) {
                        fri_sub_for[j] = Some((
                            std::time::Duration::ZERO,
                            std::time::Duration::ZERO,
                            std::time::Duration::ZERO,
                        ));
                    }
                }
            }
            fri_chunk_buckets.push(chunk_buckets);

            // ---- Per chunk-mate: open DEEP at bucket-shared iotas + assemble ---
            for j in 0..chunk_size {
                let idx = chunk_start + j;
                let (_air, _trace, pub_inputs) = &air_trace_pairs[idx];
                let domain = &domains[idx];
                let round_1_result = &chunk_round1[j];
                let round_2_result = &chunk_round2[j];
                let round_3_result = &chunk_round3[j];
                let iotas = iotas_for[j]
                    .as_ref()
                    .expect("every chunk-mate belongs to a bucket");

                let deep_poly_openings = Self::open_deep_composition_poly(
                    domain,
                    round_1_result,
                    round_2_result,
                    iotas,
                );

                let proof = StarkProof {
                    lde_trace_main_merkle_root: round_1_result.main.root,
                    lde_trace_aux_merkle_root: round_1_result.aux.as_ref().map(|x| x.root),
                    lde_trace_precomputed_merkle_root: round_1_result.main.precomputed_root,
                    trace_ood_evaluations: round_3_result.trace_ood_evaluations.clone(),
                    composition_poly_root: round_2_result.composition_poly_root,
                    composition_poly_parts_ood_evaluation: round_3_result
                        .composition_poly_parts_ood_evaluation
                        .clone(),
                    deep_poly_openings,
                    bus_public_inputs: round_1_result.bus_public_inputs.clone(),
                    public_inputs: (*pub_inputs).clone(),
                    trace_length: domain.interpolation_domain_size,
                };
                proofs.push(proof);

                #[cfg(feature = "instruments")]
                {
                    let (name, rows, pass1_dur, (r2_constraints, r2_fft, r2_merkle), r3_ood) =
                        chunk_instr1[j].clone();
                    let (deep_comp, deep_extend, fri_dur) = fri_sub_for[j]
                        .clone()
                        .unwrap_or((
                            std::time::Duration::ZERO,
                            std::time::Duration::ZERO,
                            std::time::Duration::ZERO,
                        ));
                    let sub_ops = crate::instruments::TableSubOps {
                        constraints: r2_constraints,
                        comp_decompose: r2_fft,
                        comp_commit: r2_merkle,
                        ood: r3_ood,
                        deep_comp,
                        deep_extend,
                        fri_commit: fri_dur,
                        queries: std::time::Duration::ZERO,
                    };
                    table_timings.push((name, rows, pass1_dur, sub_ops));
                }
            }
        }

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
            fri_chunk_buckets,
            chunk_size: k as u32,
        })
    }

    /// Generate a STARK proof for a single AIR/trace, returned as a one-element
    /// [`MultiProof`]. The batched-FRI bucket data lives at the multi-proof level
    /// (`MultiProof::fri_chunk_buckets`), so single-table callers consume the
    /// wrapper directly (chunk of size 1 = one bucket = one FRI instance).
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
        Self::multi_prove(
            air_trace_pairs,
            transcript,
            #[cfg(feature = "disk-spill")]
            StorageMode::Ram,
        )
    }

    // TODO: propagate errors instead of unwrap() in open_deep_composition_poly and FRI operations
    /// Rounds 2-3 (per-table): build the composition polynomial (Round 2) and
    /// evaluate it plus the trace at the out-of-domain point `z` (Round 3),
    /// appending the composition root and OOD evaluations to `transcript`.
    /// No FRI is run here — Round 4 (per-table today, per-chunk batched FRI in
    /// streaming) consumes the returned `(Round2, Round3, z)`.
    #[allow(clippy::type_complexity)]
    fn prove_rounds_2_to_3(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        pub_inputs: &PI,
        round_1_result: &Round1<Field, FieldExtension>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        domain: &Domain<Field>,
    ) -> Result<
        (
            Round2<FieldExtension>,
            Round3<FieldExtension>,
            FieldElement<FieldExtension>,
            DeepCoeffs<FieldExtension>,
        ),
        ProvingError,
    >
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        PI: Send + Sync + Clone,
    {
        // ===================================
        // ==========|   Round 2   |==========
        // ===================================

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

        let round_2_result = Self::round_2_compute_composition_polynomial(
            air,
            pub_inputs,
            domain,
            round_1_result,
            &transition_coefficients,
            &boundary_coefficients,
        )?;

        // >>>> Send commitments: [H₁], [H₂]
        transcript.append_bytes(&round_2_result.composition_poly_root);

        // ===================================
        // ==========|   Round 3   |==========
        // ===================================

        // <<<< Receive challenge: z
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
            &round_2_result,
            &z,
        );
        #[cfg(feature = "instruments")]
        crate::instruments::store_r3_ood(t_r3.elapsed());

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

        // <<<< Receive challenge: 𝛾 (DEEP) — sampled here so Round 4 only builds the
        //      DEEP LDE and runs FRI. Same transcript position as before the split.
        let deep_coeffs = Self::sample_deep_coeffs(air, &round_2_result, transcript);

        Ok((round_2_result, round_3_result, z, deep_coeffs))
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
