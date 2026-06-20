use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
#[cfg(feature = "instruments")]
use std::time::{Duration, Instant};

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
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + 'static,
    FieldExtension: Send + Sync + IsField + 'static,
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
    /// The prover's recomputed preprocessed Merkle root did not match the
    /// commitment the AIR was constructed with (e.g. a stale static constant
    /// in a table module, or a wrong caller-supplied entry such as
    /// `page_commitments` / `decode_commitment`). Continuing would yield a
    /// proof an honest verifier always rejects — fail fast on the prover side
    /// with a localized error instead.
    PrecomputedCommitmentMismatch,
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
    fn plain(tree: BatchedMerkleTree<F>, root: Commitment) -> Self {
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
        tree: BatchedMerkleTree<F>,
        root: Commitment,
        precomputed_tree: BatchedMerkleTree<F>,
        precomputed_root: Commitment,
        num_precomputed_cols: usize,
    ) -> Self {
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
    FieldElement<Field>: AsBytes + math::traits::ByteConversion,
    FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
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

/// Tuple returned by `commit_main_trace`: the commit, the cached LDE columns,
/// and (under cuda) the optional device LDE buffer kept alive for downstream
/// rounds when the R1 fused GPU pipeline ran.
#[cfg(feature = "cuda")]
type MainCommitTuple<F> = (
    TableCommit<F>,
    Vec<Vec<FieldElement<F>>>,
    Option<math_cuda::lde::GpuLdeBase>,
);
#[cfg(not(feature = "cuda"))]
type MainCommitTuple<F> = (TableCommit<F>, Vec<Vec<FieldElement<F>>>);

/// Round 1 commitment artifacts — Merkle trees, roots, challenges, and bus inputs.
/// Borrowed (not consumed) when building `Round1` in Phase D.
pub(crate) struct Round1Commitments<Field, FieldExtension>
where
    Field: IsFFTField + IsSubFieldOf<FieldExtension>,
    FieldExtension: IsField,
    FieldElement<Field>: AsBytes + math::traits::ByteConversion,
    FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
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
    /// Device-side main LDE buffer, populated only when the R1 GPU fused
    /// pipeline ran for this table. Kept so R2/R3/R4 GPU paths can read
    /// the LDE without re-H2D.
    #[cfg(feature = "cuda")]
    gpu_main: Option<math_cuda::lde::GpuLdeBase>,
    #[cfg(feature = "cuda")]
    gpu_aux: Option<math_cuda::lde::GpuLdeExt3>,
}

impl<Field, FieldExtension> Round1Commitments<Field, FieldExtension>
where
    Field: IsFFTField + IsSubFieldOf<FieldExtension> + Send + Sync,
    FieldExtension: IsField + Send + Sync,
    FieldElement<Field>: AsBytes + math::traits::ByteConversion,
    FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
{
    /// Build a `Round1` by consuming a `Lde` and borrowing commitment data.
    /// The `TableCommit::share` calls are cheap — only bump Arc refcounts.
    fn build_round1(
        &self,
        lde: Lde<Field, FieldExtension>,
        step_size: usize,
        blowup_factor: usize,
    ) -> Round1<Field, FieldExtension> {
        #[allow(unused_mut)]
        let mut lde_trace =
            LDETraceTable::from_columns(lde.main, lde.aux, step_size, blowup_factor);
        #[cfg(feature = "cuda")]
        {
            if let Some(h) = lde.gpu_main {
                lde_trace.set_gpu_main(h);
            }
            if let Some(h) = lde.gpu_aux {
                lde_trace.set_gpu_aux(h);
            }
        }
        Round1 {
            lde_trace,
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
    FieldElement<F>: AsBytes + math::traits::ByteConversion,
{
    /// Evaluations of the composition polynomial parts over the LDE domain.
    pub(crate) lde_composition_poly_evaluations: Vec<Vec<FieldElement<F>>>,
    /// The Merkle tree built to compute the commitment to the composition polynomial parts.
    pub(crate) composition_poly_merkle_tree: BatchedMerkleTree<F>,
    /// The commitment to the composition polynomial parts.
    pub(crate) composition_poly_root: Commitment,
    /// Device-resident de-interleaved LDE handle from the R2 fused GPU path
    /// (`try_evaluate_parts_on_lde_gpu_keep`). When present, R4 DEEP skips
    /// the `num_parts * 3 * lde_size * 8` byte H2D and reads parts on
    /// device. `None` when the GPU R2 path didn't run (number_of_parts <= 2,
    /// below threshold, or any CPU fallback).
    #[cfg(feature = "cuda")]
    pub(crate) gpu_composition_parts: Option<math_cuda::lde::GpuLdeExt3>,
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
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + 'static,
    FieldExtension: Send + Sync + IsField + 'static,
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
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
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
        E: IsSubFieldOf<FieldExtension> + IsField + Send + Sync + 'static,
        FieldElement<E>: Send + Sync,
    {
        if columns.is_empty() {
            return;
        }

        // GPU batched fast path: all columns at once in one pipeline on one
        // stream. Falls through to per-column rayon when the table is too
        // small, the element type isn't Goldilocks, or the `cuda` feature is
        // off.
        #[cfg(feature = "cuda")]
        if crate::gpu_lde::try_expand_columns_batched::<Field, E>(
            columns,
            domain.blowup_factor,
            &twiddles.coset_weights,
        )
        .is_some()
        {
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
    /// with the owned LDE columns (consumed later in Phase D) and (under
    /// cuda) the optional device LDE buffer kept alive for downstream rounds
    /// when the R1 fused GPU pipeline ran.
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
        #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
    ) -> Result<MainCommitTuple<Field>, ProvingError>
    where
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
    {
        let lde_size = domain.interpolation_domain_size * domain.blowup_factor;
        let mut columns = trace.extract_columns_main(lde_size);

        // Fused GPU path is only wired for non-preprocessed mains today. The
        // preprocessed split runs the CPU pipeline below.
        #[cfg(feature = "cuda")]
        if precomputed.is_none() {
            #[cfg(feature = "instruments")]
            let t_sub = Instant::now();
            if let Some((tree, handle)) =
                crate::gpu_lde::try_expand_leaf_and_tree_batched_keep::<
                    Field,
                    Field,
                    BatchedMerkleTreeBackend<Field>,
                >(&mut columns, domain.blowup_factor, &twiddles.coset_weights)
            {
                #[cfg(feature = "instruments")]
                let main_lde_dur = t_sub.elapsed();
                let root = tree.root;
                // Fused GPU path produces LDE + leaves + tree as one pipeline,
                // so the wall-clock total lands in `main_lde_dur`. Bill the
                // merkle bucket equal to LDE so the sum (lde + merkle) stays
                // comparable to the non-GPU path's combined LDE+commit total.
                #[cfg(feature = "instruments")]
                crate::instruments::accum_r1_main(main_lde_dur, main_lde_dur);
                return Ok((TableCommit::plain(tree, root), columns, Some(handle)));
            }
        }

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
                TableCommit::plain(tree, root)
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
                if precomputed_root != expected_precomputed_root {
                    return Err(ProvingError::PrecomputedCommitmentMismatch);
                }
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
                )
            }
        };

        #[cfg(feature = "instruments")]
        crate::instruments::accum_r1_main(main_lde_dur, t_sub.elapsed());

        #[cfg(feature = "cuda")]
        return Ok((commit, columns, None));
        #[cfg(not(feature = "cuda"))]
        Ok((commit, columns))
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
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
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
                main,
                aux,
                #[cfg(feature = "cuda")]
                gpu_main: None,
                #[cfg(feature = "cuda")]
                gpu_aux: None,
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
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
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
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
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
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    {
        let two_n = constraint_evaluations.len();
        let n = two_n / 2;
        debug_assert_eq!(two_n, n * 2);

        // Step 1: Compute 1/(2·g·ω^i) for i=0..N-1 via batch inversion.
        // The LDE coset points are g·ω^i = domain.lde_roots_of_unity_coset[i].
        // Compute entirely in base field — mixed F×E multiplication when used with extension values.
        let two_base = FieldElement::<Field>::from(2u64);
        let mut inv_2x: Vec<FieldElement<Field>> = (0..n)
            // 2·(g·ωⁱ) = (g·ωⁱ).double() — one add, vs a base mul+reduce per element.
            .map(|i| domain.lde_roots_of_unity_coset[i].double())
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

        // GPU fast path: batch both halves into one ext3 LDE call. Requires
        // `cuda` feature and a qualifying size. Falls through to CPU when not.
        #[cfg(feature = "cuda")]
        if let Some((lde_h0, lde_h1)) =
            crate::gpu_lde::try_extend_two_halves_gpu(&h0_evals, &h1_evals, domain)
        {
            return vec![lde_h0, lde_h1];
        }

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
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
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
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
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
        #[cfg(feature = "cuda")]
        let mut gpu_composition_parts: Option<math_cuda::lde::GpuLdeExt3> = None;
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

            let cpu_eval = || -> Vec<Vec<FieldElement<FieldExtension>>> {
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

            // GPU fast path: batched ext3 LDE for all parts in one call.
            // `_keep` variant retains the de-interleaved device buffer as a
            // `GpuLdeExt3` handle stored on Round2 so R4 DEEP can skip the
            // `num_parts * 3 * lde_size * 8` byte H2D.
            #[cfg(feature = "cuda")]
            {
                let parts_slices: Vec<&[FieldElement<FieldExtension>]> = composition_poly_parts
                    .iter()
                    .map(|p| p.coefficients.as_slice())
                    .collect();
                match crate::gpu_lde::try_evaluate_parts_on_lde_gpu_keep::<Field, FieldExtension>(
                    &parts_slices,
                    domain.blowup_factor,
                    domain.interpolation_domain_size,
                    &domain.coset_offset,
                ) {
                    Some((evals, handle)) => {
                        gpu_composition_parts = Some(handle);
                        evals
                    }
                    None => cpu_eval(),
                }
            }
            #[cfg(not(feature = "cuda"))]
            cpu_eval()
        };
        #[cfg(feature = "instruments")]
        let fft_dur = t_sub.elapsed();

        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();
        // GPU fast path for the comp-poly Merkle commit: row-pair Keccak
        // leaves + device-side inner tree, both wrapping the host eval Vecs.
        #[cfg(feature = "cuda")]
        let gpu_tree = crate::gpu_lde::try_build_comp_poly_tree_gpu::<
            FieldExtension,
            BatchedMerkleTreeBackend<FieldExtension>,
        >(&lde_composition_poly_parts_evaluations);
        #[cfg(not(feature = "cuda"))]
        let gpu_tree: Option<BatchedMerkleTree<FieldExtension>> = None;

        let (composition_poly_merkle_tree, composition_poly_root) = match gpu_tree {
            Some(tree) => {
                let root = tree.root;
                (tree, root)
            }
            None => Self::commit_composition_polynomial(&lde_composition_poly_parts_evaluations)
                .ok_or(ProvingError::EmptyCommitment)?,
        };
        #[cfg(feature = "instruments")]
        let merkle_dur = t_sub.elapsed();

        #[cfg(feature = "instruments")]
        crate::instruments::store_r2_sub(constraints_dur, fft_dur, merkle_dur);

        Ok(Round2 {
            lde_composition_poly_evaluations: lde_composition_poly_parts_evaluations,
            composition_poly_merkle_tree,
            composition_poly_root,
            #[cfg(feature = "cuda")]
            gpu_composition_parts,
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
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
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
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
    ) -> Round4<Field, FieldExtension>
    where
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
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
        let (fri_last_value, fri_layers) = fri::commit_phase_from_evaluations(
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
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
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
        let lde_size = domain.lde_roots_of_unity_coset.len();

        // OOD evaluations
        let h_ood = &round_3_result.composition_poly_parts_ood_evaluation;
        let trace_ood_columns = round_3_result.trace_ood_evaluations.columns();
        let num_total_cols = num_main_cols + num_aux_cols;

        // Fully device-resident GPU fast path: build inv_denoms on device
        // ([z^K, z_shifted[0..]] over the full LDE coset), then run R4
        // DEEP composition reading the same device buffer. Skips the
        // CPU `inplace_batch_inverse` on the happy path; on any GPU
        // failure we fall through and compute denoms on CPU below.
        #[cfg(feature = "cuda")]
        {
            let z_scalars: Vec<FieldElement<FieldExtension>> = core::iter::once(z_power.clone())
                .chain(z_shifted.iter().cloned())
                .collect();
            if let Some((inv_dev, stream)) =
                crate::gpu_lde::try_inv_denoms_dev_with_stream::<Field, FieldExtension>(
                    &domain.lde_roots_of_unity_coset,
                    &z_scalars,
                    math_cuda::inverse::DenomSign::XMinusZ,
                )
                && let Some(deep_evals) =
                    crate::gpu_lde::try_deep_composition_gpu::<Field, FieldExtension>(
                        lde_trace,
                        round_2_result.gpu_composition_parts.as_ref(),
                        &round_2_result.lde_composition_poly_evaluations,
                        h_ood,
                        &trace_ood_columns,
                        composition_poly_gammas,
                        trace_terms_gammas,
                        &[],
                        Some((&inv_dev, &stream)),
                        num_eval_points,
                    )
            {
                return deep_evals;
            }
        }

        // CPU denoms + batch inverse for the fallback paths below.
        // Single-source helper shared with the GPU parity test so any
        // sign/ordering/layout drift breaks the test instead of silently
        // diverging CUDA vs non-CUDA proofs.
        let denoms = crate::r4_denoms::build_r4_inv_denoms_cpu::<Field, FieldExtension>(
            &domain.lde_roots_of_unity_coset,
            &z_power,
            &z_shifted,
        )
        .expect("R4 inv denoms: coset points are base field, poles are extension field");

        let inv_h = &denoms[0..lde_size];

        // GPU mixed path: dev parts (when R2 keep handle exists) + host
        // inv_denoms. Used when the dev-inv-denoms path above didn't fire
        // (e.g., cudarc error in compute_denoms / scan).
        #[cfg(feature = "cuda")]
        {
            if let Some(deep_evals) =
                crate::gpu_lde::try_deep_composition_gpu::<Field, FieldExtension>(
                    lde_trace,
                    round_2_result.gpu_composition_parts.as_ref(),
                    &round_2_result.lde_composition_poly_evaluations,
                    h_ood,
                    &trace_ood_columns,
                    composition_poly_gammas,
                    trace_terms_gammas,
                    &denoms,
                    None,
                    num_eval_points,
                )
            {
                return deep_evals;
            }
        }

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
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
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
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
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
        mut air_trace_pairs: Vec<AirTracePair<'_, Field, FieldExtension, PI>>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone + Send),
        #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
    ) -> Result<MultiProof<Field, FieldExtension, PI>, ProvingError>
    where
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
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
        let mut domain_cache: hashbrown::HashMap<(usize, usize, u64), DomainEntry<Field>> =
            hashbrown::HashMap::new();

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

        let mut main_commits: Vec<TableCommit<Field>> = Vec::with_capacity(num_airs);
        let mut main_ldes: Vec<Vec<Vec<FieldElement<Field>>>> = Vec::with_capacity(num_airs);
        // Optional device-side LDE handle per table, populated only when the
        // R1 fused GPU pipeline produced one. Threaded through Phase D's zip
        // chain so each handle stays paired with its table by construction.
        #[cfg(feature = "cuda")]
        let mut main_gpu_handles: Vec<Option<math_cuda::lde::GpuLdeBase>> =
            Vec::with_capacity(num_airs);

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

                    let precomputed = air
                        .is_preprocessed()
                        .then(|| (air.precomputed_commitment(), air.num_precomputed_columns()));
                    Self::commit_main_trace(
                        *trace,
                        domain,
                        twiddles,
                        precomputed,
                        #[cfg(feature = "disk-spill")]
                        storage_mode,
                    )
                })
                .collect();

            // Sequential: append roots to shared transcript (Fiat-Shamir ordering)
            for result in chunk_results {
                #[cfg(feature = "cuda")]
                let (commit, cached_main, gpu_main) = result?;
                #[cfg(not(feature = "cuda"))]
                let (commit, cached_main) = result?;
                if let Some(ref pre_root) = commit.precomputed_root {
                    transcript.append_bytes(pre_root);
                }
                transcript.append_bytes(&commit.root);
                main_commits.push(commit);
                main_ldes.push(cached_main);
                #[cfg(feature = "cuda")]
                main_gpu_handles.push(gpu_main);
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

        // Pass 2: Parallel fork transcript → extract → LDE → commit in chunks of K.
        // Each table gets its own transcript fork.
        #[cfg(feature = "instruments")]
        let phase_start = Instant::now();

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

        // Parallel aux commit in chunks of K. The closure returns a cfg-gated
        // AuxResult. Under cuda it carries the optional ext3 GPU LDE handle as
        // a third element, so Phase D's zip chain keeps it paired with its
        // table without a separate handle vector.
        #[cfg(feature = "cuda")]
        type AuxResult<FE> = (
            Option<TableCommit<FE>>,
            Vec<Vec<FieldElement<FE>>>,
            Option<math_cuda::lde::GpuLdeExt3>,
        );
        #[cfg(not(feature = "cuda"))]
        type AuxResult<FE> = (Option<TableCommit<FE>>, Vec<Vec<FieldElement<FE>>>);
        #[allow(clippy::type_complexity)]
        let mut aux_results: Vec<AuxResult<FieldExtension>> = Vec::with_capacity(num_airs);

        for chunk_start in (0..num_airs).step_by(k) {
            let chunk_end = (chunk_start + k).min(num_airs);
            let chunk_range = chunk_start..chunk_end;

            #[cfg(feature = "parallel")]
            let iter = chunk_range.into_par_iter();
            #[cfg(not(feature = "parallel"))]
            let iter = chunk_range;

            #[allow(clippy::type_complexity)]
            let chunk_aux: Vec<Result<AuxResult<FieldExtension>, ProvingError>> = iter
                .map(|idx| {
                    let (air, trace, _) = &air_trace_pairs[idx];
                    let domain = &domains[idx];
                    let twiddles = &twiddle_caches[idx];

                    if air.has_aux_trace() {
                        let lde_size = domain.interpolation_domain_size * domain.blowup_factor;
                        let mut columns = trace.extract_columns_aux(lde_size);

                        // Fused GPU path: ext3 LDE + Keccak-256 leaf hashing + Merkle tree build
                        // in one on-device pipeline, also retaining the device LDE buffer and
                        // returning its handle for downstream GPU rounds.
                        #[cfg(feature = "cuda")]
                        {
                            #[cfg(feature = "instruments")]
                            let t_sub = Instant::now();
                            if let Some((tree, handle)) =
                                crate::gpu_lde::try_expand_leaf_and_tree_batched_ext3_keep::<
                                    Field,
                                    FieldExtension,
                                    BatchedMerkleTreeBackend<FieldExtension>,
                                >(
                                    &mut columns, domain.blowup_factor, &twiddles.coset_weights
                                )
                            {
                                #[cfg(feature = "instruments")]
                                let aux_lde_dur = t_sub.elapsed();
                                let root = tree.root;
                                // Fused GPU path: LDE + leaf hash + tree build run as one pipeline with
                                // no separate merkle timing, so bill the whole fused duration to the LDE
                                // bucket and zero to merkle. The (lde + merkle) sum then equals the fused
                                // time, comparable to the non-GPU path's combined R1 total.
                                #[cfg(feature = "instruments")]
                                crate::instruments::accum_r1_aux(aux_lde_dur, Duration::ZERO);
                                return Ok((
                                    Some(TableCommit::plain(tree, root)),
                                    columns,
                                    Some(handle),
                                ));
                            }
                        }

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
                        #[cfg(feature = "cuda")]
                        return Ok((Some(TableCommit::plain(tree, root)), columns, None));
                        #[cfg(not(feature = "cuda"))]
                        Ok((Some(TableCommit::plain(tree, root)), columns))
                    } else {
                        #[cfg(feature = "cuda")]
                        return Ok((None, Vec::new(), None));
                        #[cfg(not(feature = "cuda"))]
                        Ok((None, Vec::new()))
                    }
                })
                .collect();

            // Sequential: append aux roots to forked transcripts.
            for (j, result) in chunk_aux.into_iter().enumerate() {
                let aux_full = result?;
                // Tuple shape is cfg-gated; `.0` is the optional TableCommit
                // in both variants.
                if let Some(ref c) = aux_full.0 {
                    table_transcripts[chunk_start + j].append_bytes(&c.root);
                }
                aux_results.push(aux_full);
            }
        }

        // Build commitments and cached LDEs as separate vecs:
        // commitments are borrowed in Phase D, LDEs are consumed by value.
        let mut commitments: Vec<Round1Commitments<Field, FieldExtension>> =
            Vec::with_capacity(num_airs);
        let mut cached_ldes: Vec<Lde<Field, FieldExtension>> = Vec::with_capacity(num_airs);
        // Under cuda, fold main_gpu_handles into the zip chain so each handle
        // stays paired with its table by construction.
        #[cfg(feature = "cuda")]
        let main_iter = main_commits
            .into_iter()
            .zip(main_ldes)
            .zip(main_gpu_handles);
        #[cfg(not(feature = "cuda"))]
        let main_iter = main_commits.into_iter().zip(main_ldes);

        for ((main_pack, aux_full), bus_public_inputs) in
            main_iter.zip(aux_results).zip(bus_inputs_vec)
        {
            #[cfg(feature = "cuda")]
            let ((main_commit, main_lde), gpu_main) = main_pack;
            #[cfg(not(feature = "cuda"))]
            let (main_commit, main_lde) = main_pack;
            #[cfg(feature = "cuda")]
            let (aux_commit, cached_aux, gpu_aux) = aux_full;
            #[cfg(not(feature = "cuda"))]
            let (aux_commit, cached_aux) = aux_full;
            commitments.push(Round1Commitments {
                main: main_commit,
                aux: aux_commit,
                rap_challenges: lookup_challenges.clone(),
                bus_public_inputs,
            });
            #[cfg(feature = "cuda")]
            cached_ldes.push(Lde {
                main: main_lde,
                aux: cached_aux,
                gpu_main,
                gpu_aux,
            });
            #[cfg(not(feature = "cuda"))]
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
            Duration,
            crate::instruments::TableSubOps,
        )> = Vec::with_capacity(num_airs);

        let mut proofs = Vec::with_capacity(num_airs);
        let mut lde_drain = cached_ldes.into_iter();
        for chunk_start in (0..num_airs).step_by(k) {
            let chunk_end = (chunk_start + k).min(num_airs);
            let chunk_size = chunk_end - chunk_start;

            let chunk_ldes: Vec<Lde<Field, FieldExtension>> =
                lde_drain.by_ref().take(chunk_size).collect();
            let chunk_commitments = &commitments[chunk_start..chunk_end];
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

            let chunk_results: Vec<Result<_, ProvingError>> = iter
                .map(|(j, ((lde, commitment), table_transcript))| {
                    let idx = chunk_start + j;
                    let (air, trace, pub_inputs) = &air_trace_pairs[idx];
                    let _ = trace; // used by instruments
                    let domain = &domains[idx];

                    #[cfg(feature = "instruments")]
                    let table_start = Instant::now();

                    // Build Round1 from cached LDE (consumed by value, no recomputation).
                    let round_1_result =
                        commitment.build_round1(lde, air.step_size(), domain.blowup_factor);

                    if let Some(ref bpi) = round_1_result.bus_public_inputs {
                        table_transcript.append_field_element(&bpi.table_contribution);
                    }

                    let proof = Self::prove_rounds_2_to_4(
                        *air,
                        *pub_inputs,
                        &round_1_result,
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
                    return Ok((proof, table_timing));
                    #[cfg(not(feature = "instruments"))]
                    Ok(proof)
                })
                .collect();

            for result in chunk_results {
                #[cfg(feature = "instruments")]
                {
                    let (proof, timing) = result?;
                    proofs.push(proof);
                    table_timings.push(timing);
                }
                #[cfg(not(feature = "instruments"))]
                proofs.push(result?);
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

        Ok(MultiProof { proofs })
    }

    /// Generate a STARK proof for a single AIR/trace.
    /// This is equivalent to calling `multi_prove` with a single-element slice.
    fn prove(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        trace: &mut TraceTable<Field, FieldExtension>,
        pub_inputs: &PI,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone + Send),
    ) -> Result<StarkProof<Field, FieldExtension, PI>, ProvingError>
    where
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
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
        .map(|mut multi_proof| multi_proof.proofs.remove(0))
    }

    // TODO: propagate errors instead of unwrap() in open_deep_composition_poly and FRI operations
    /// Executes rounds 2-4 and generates a STARK proof for the trace `main_trace` with public inputs `pub_inputs`.
    /// Warning: the transcript must be safely initializated before passing it to this method.
    fn prove_rounds_2_to_4(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        pub_inputs: &PI,
        round_1_result: &Round1<Field, FieldExtension>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
        domain: &Domain<Field>,
    ) -> Result<StarkProof<Field, FieldExtension, PI>, ProvingError>
    where
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
        PI: Send + Sync + Clone,
    {
        info!("Started proof generation...");

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

        // Part of this round is running FRI, which is an interactive
        // protocol on its own. Therefore we pass it the transcript
        // to simulate the interactions with the verifier.
        let round_4_result = Self::round_4_compute_and_run_fri_on_the_deep_composition_polynomial(
            air,
            domain,
            round_1_result,
            &round_2_result,
            &round_3_result,
            &z,
            transcript,
        );

        #[cfg(feature = "instruments")]
        {
            let zero = Duration::ZERO;
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
            // [t]
            lde_trace_main_merkle_root: round_1_result.main.root,
            // [t]
            lde_trace_aux_merkle_root: round_1_result.aux.as_ref().map(|x| x.root),
            // For preprocessed tables: commitment to precomputed columns only
            lde_trace_precomputed_merkle_root: round_1_result.main.precomputed_root,
            // tⱼ(zgᵏ)
            trace_ood_evaluations: round_3_result.trace_ood_evaluations,
            // [H₁] and [H₂]
            composition_poly_root: round_2_result.composition_poly_root,
            // Hᵢ(z^N)
            composition_poly_parts_ood_evaluation: round_3_result
                .composition_poly_parts_ood_evaluation,
            // [pₖ]
            fri_layers_merkle_roots: round_4_result.fri_layers_merkle_roots,
            // pₙ
            fri_last_value: round_4_result.fri_last_value,
            // Open(p₀(D₀), 𝜐ₛ), Open(pₖ(Dₖ), −𝜐ₛ^(2ᵏ))
            query_list: round_4_result.query_list,
            // Open(H₁(D_LDE, 𝜐₀), Open(H₂(D_LDE, 𝜐₀), Open(tⱼ(D_LDE), 𝜐₀)
            // Open(H₁(D_LDE, -𝜐ᵢ), Open(H₂(D_LDE, -𝜐ᵢ), Open(tⱼ(D_LDE), -𝜐ᵢ)
            deep_poly_openings: round_4_result.deep_poly_openings,
            // nonce obtained from grinding
            nonce: round_4_result.nonce,
            // Bus interaction public inputs (for boundary constraints and bus balance check)
            bus_public_inputs: round_1_result.bus_public_inputs.clone(),
            // Public inputs for boundary constraints
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
