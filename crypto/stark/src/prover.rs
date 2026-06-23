use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
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
use crate::domain::new_domain;
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

#[cfg(test)]
pub(crate) mod domain_cache_stats {
    use std::cell::Cell;

    thread_local! {
        static COUNTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    }

    pub(crate) fn reset() {
        COUNTS.with(|c| c.set((0, 0)));
    }

    pub(crate) fn get() -> (usize, usize) {
        COUNTS.with(Cell::get)
    }

    pub(crate) fn record(was_hit: bool) {
        COUNTS.with(|c| {
            let (hits, misses) = c.get();
            c.set(if was_hit {
                (hits + 1, misses)
            } else {
                (hits, misses + 1)
            });
        });
    }
}

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

/// A container for the intermediate results of the commitments to a trace table, main or auxiliary in case of RAP,
/// in the first round of the STARK Prove protocol.
pub struct Round1CommitmentData<F>
where
    F: IsField,
    FieldElement<F>: AsBytes + math::traits::ByteConversion,
{
    /// The Merkle trees constructed to obtain the commitment of the entire trace table.
    /// For preprocessed tables, this contains only the multiplicity columns.
    /// Wrapped in Arc to share with Round1Commitments without deep-cloning (~64MB per table).
    pub(crate) lde_trace_merkle_tree: Arc<BatchedMerkleTree<F>>,
    /// The root of the Merkle tree in `lde_trace_merkle_tree`.
    pub(crate) lde_trace_merkle_root: Commitment,
    /// For preprocessed tables: Merkle tree over precomputed columns only.
    pub(crate) precomputed_merkle_tree: Option<Arc<BatchedMerkleTree<F>>>,
    /// For preprocessed tables: root of the precomputed Merkle tree.
    pub(crate) precomputed_merkle_root: Option<Commitment>,
    /// For preprocessed tables: number of precomputed columns (for splitting during opening).
    pub(crate) num_precomputed_cols: usize,
}

/// A container for the results of the first round of the STARK Prove protocol.
pub struct Round1<Field, FieldExtension>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField,
    FieldExtension: IsField,
    FieldElement<Field>: AsBytes + math::traits::ByteConversion,
    FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
{
    /// The table of evaluations over the LDE of the main and auxiliary trace tables.
    pub(crate) lde_trace: LDETraceTable<Field, FieldExtension>,
    /// The intermediate results of the commitment to the main trace table.
    pub(crate) main: Round1CommitmentData<Field>,
    /// The intermediate results of the commitment to the auxiliary trace table in case of RAP.
    pub(crate) aux: Option<Round1CommitmentData<FieldExtension>>,
    /// The challenges of the RAP round.
    pub(crate) rap_challenges: Vec<FieldElement<FieldExtension>>,
    /// Bus interaction public inputs (initial and final aux column values).
    pub(crate) bus_public_inputs: Option<BusPublicInputs<FieldExtension>>,
}

/// Intermediate results from committing a main trace in Phase A of sequential proving.
/// Holds the Merkle tree/root for the main trace and optionally for precomputed columns.
struct MainCommitData<Field: IsFFTField>
where
    FieldElement<Field>: AsBytes + math::traits::ByteConversion,
{
    main_tree: Arc<BatchedMerkleTree<Field>>,
    main_root: Commitment,
    precomputed_tree: Option<Arc<BatchedMerkleTree<Field>>>,
    precomputed_root: Option<Commitment>,
    num_precomputed_cols: usize,
}

/// Round 1 commitment artifacts — Merkle trees, roots, challenges, and bus inputs.
/// Borrowed (not consumed) when building `Round1` in Phase D.
pub struct Round1Commitments<Field, FieldExtension>
where
    Field: IsFFTField + IsSubFieldOf<FieldExtension>,
    FieldExtension: IsField,
    FieldElement<Field>: AsBytes + math::traits::ByteConversion,
    FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
{
    /// Merkle tree of the main trace (multiplicities for preprocessed tables).
    /// Wrapped in Arc to share with Round1CommitmentData without deep-cloning.
    main_merkle_tree: Arc<BatchedMerkleTree<Field>>,
    /// Root of the main trace Merkle tree.
    main_merkle_root: Commitment,
    /// For preprocessed tables: Merkle tree over precomputed columns.
    precomputed_merkle_tree: Option<Arc<BatchedMerkleTree<Field>>>,
    /// For preprocessed tables: root of the precomputed Merkle tree.
    precomputed_merkle_root: Option<Commitment>,
    /// For preprocessed tables: number of precomputed columns.
    num_precomputed_cols: usize,
    /// Merkle tree of the auxiliary trace (None if no aux trace).
    aux_merkle_tree: Option<Arc<BatchedMerkleTree<FieldExtension>>>,
    /// Root of the auxiliary trace Merkle tree (None if no aux trace).
    aux_merkle_root: Option<Commitment>,
    /// The RAP challenges used for auxiliary trace construction.
    rap_challenges: Vec<FieldElement<FieldExtension>>,
    /// Bus interaction public inputs (initial and final aux column values).
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
    FieldElement<Field>: AsBytes + math::traits::ByteConversion,
    FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
{
    /// Build a `Round1` by consuming a `Lde` and borrowing commitment data.
    fn build_round1(
        &self,
        lde: Lde<Field, FieldExtension>,
        step_size: usize,
        blowup_factor: usize,
        has_aux_trace: bool,
    ) -> Round1<Field, FieldExtension> {
        let lde_trace = LDETraceTable::from_columns(lde.main, lde.aux, step_size, blowup_factor);

        let main = Round1CommitmentData::<Field> {
            lde_trace_merkle_tree: Arc::clone(&self.main_merkle_tree),
            lde_trace_merkle_root: self.main_merkle_root,
            precomputed_merkle_tree: self.precomputed_merkle_tree.as_ref().map(Arc::clone),
            precomputed_merkle_root: self.precomputed_merkle_root,
            num_precomputed_cols: self.num_precomputed_cols,
        };

        let aux = if has_aux_trace {
            Some(Round1CommitmentData::<FieldExtension> {
                lde_trace_merkle_tree: Arc::clone(
                    self.aux_merkle_tree
                        .as_ref()
                        .expect("aux tree must exist when has_aux_trace"),
                ),
                lde_trace_merkle_root: self
                    .aux_merkle_root
                    .expect("aux root must exist when has_aux_trace"),
                precomputed_merkle_tree: None,
                precomputed_merkle_root: None,
                num_precomputed_cols: 0,
            })
        } else {
            None
        };

        Round1 {
            lde_trace,
            main,
            aux,
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
pub struct LdeTwiddles<F: IsFFTField> {
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
pub struct Round2<F>
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
}

/// A container for the results of the third round of the STARK Prove protocol.
pub struct Round3<F: IsField> {
    /// Evaluations of the trace polynomials, main ans auxiliary, at the out-of-domain challenge.
    trace_ood_evaluations: Table<F>,
    /// Evaluations of the composition polynomial parts at the out-of-domain challenge.
    composition_poly_parts_ood_evaluation: Vec<FieldElement<F>>,
}

/// A container for the results of the fourth round of the STARK Prove protocol.
pub struct Round4<F: IsSubFieldOf<E>, E: IsField> {
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

    #[cfg(feature = "parallel")]
    let iter = (0..num_rows).into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = 0..num_rows;

    iter.map(|row_idx| {
        let br_idx = reverse_index(row_idx, num_rows as u64);
        let total_bytes = num_cols * byte_len;
        let mut buf = vec![0u8; total_bytes];
        for col_idx in 0..num_cols {
            columns[col_idx][br_idx]
                .write_bytes_be(&mut buf[col_idx * byte_len..(col_idx + 1) * byte_len]);
        }
        BatchedMerkleTreeBackend::<E>::hash_bytes(&buf)
    })
    .collect()
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

    #[cfg(feature = "parallel")]
    let iter = (0..num_leaves).into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = 0..num_leaves;

    iter.map(|leaf_idx| {
        let br_0 = reverse_index(2 * leaf_idx, num_rows as u64);
        let br_1 = reverse_index(2 * leaf_idx + 1, num_rows as u64);
        let total_bytes = 2 * num_parts * byte_len;
        let mut buf = vec![0u8; total_bytes];
        let mut offset = 0;
        for part in parts.iter() {
            part[br_0].write_bytes_be(&mut buf[offset..offset + byte_len]);
            offset += byte_len;
        }
        for part in parts.iter() {
            part[br_1].write_bytes_be(&mut buf[offset..offset + byte_len]);
            offset += byte_len;
        }
        BatchedMerkleTreeBackend::<E>::hash_bytes(&buf)
    })
    .collect()
}

/// The functionality of a STARK prover providing methods to run the STARK Prove protocol
/// https://lambdaclass.github.io/lambdaworks/starks/protocol.html
/// The default implementation is complete and is compatible with Stone prover
/// https://github.com/starkware-libs/stone-prover
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
    /// useful for setting up soundness test scenarios.
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

    /// Compute main LDE, commit, and return the Merkle tree/root along with the
    /// owned LDE columns (consumed later in Phase D).
    #[allow(clippy::type_complexity)]
    fn commit_main_trace(
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
        #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
    ) -> Result<
        (
            BatchedMerkleTree<Field>,
            Commitment,
            Option<BatchedMerkleTree<Field>>,
            Option<Commitment>,
            usize,
            Vec<Vec<FieldElement<Field>>>,
        ),
        ProvingError,
    >
    where
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
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
        #[allow(unused_mut)]
        let (mut tree, root) =
            Self::commit_columns_bit_reversed(&columns).ok_or(ProvingError::EmptyCommitment)?;
        #[cfg(feature = "instruments")]
        crate::instruments::accum_r1_main(main_lde_dur, t_sub.elapsed());

        #[cfg(feature = "disk-spill")]
        if storage_mode == StorageMode::Disk {
            tree.spill_nodes_to_disk()
                .map_err(|e| ProvingError::DiskSpill(format!("main Merkle tree: {e}")))?;
        }

        Ok((tree, root, None, None, 0, columns))
    }

    /// Commit preprocessed trace: precomputed and multiplicity columns get separate trees.
    #[allow(clippy::type_complexity)]
    fn commit_preprocessed_trace(
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        precomputed_commitment: Commitment,
        num_precomputed_cols: usize,
        twiddles: &LdeTwiddles<Field>,
        #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
    ) -> Result<
        (
            BatchedMerkleTree<Field>,
            Commitment,
            Option<BatchedMerkleTree<Field>>,
            Option<Commitment>,
            usize,
            Vec<Vec<FieldElement<Field>>>,
        ),
        ProvingError,
    >
    where
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
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
        #[allow(unused_mut)]
        let (mut precomputed_tree, precomputed_root) =
            Self::commit_columns_bit_reversed(&columns[..num_precomputed_cols])
                .ok_or(ProvingError::EmptyCommitment)?;

        #[allow(unused_mut)]
        let (mut mult_tree, mult_root) =
            Self::commit_columns_bit_reversed(&columns[num_precomputed_cols..])
                .ok_or(ProvingError::EmptyCommitment)?;
        #[cfg(feature = "instruments")]
        crate::instruments::accum_r1_main(main_lde_dur, t_sub.elapsed());

        debug_assert_eq!(
            precomputed_root, precomputed_commitment,
            "Prover's precomputed commitment doesn't match hardcoded AIR commitment"
        );

        #[cfg(feature = "disk-spill")]
        if storage_mode == StorageMode::Disk {
            precomputed_tree
                .spill_nodes_to_disk()
                .map_err(|e| ProvingError::DiskSpill(format!("precomputed Merkle tree: {e}")))?;
            mult_tree
                .spill_nodes_to_disk()
                .map_err(|e| ProvingError::DiskSpill(format!("mult Merkle tree: {e}")))?;
        }

        Ok((
            mult_tree,
            mult_root,
            Some(precomputed_tree),
            Some(precomputed_root),
            num_precomputed_cols,
            columns,
        ))
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
            Lde { main, aux },
            air.step_size(),
            domain.blowup_factor,
            air.has_aux_trace(),
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
            .map(|i| &two_base * &domain.lde_roots_of_unity_coset[i])
            .collect();
        FieldElement::inplace_batch_inverse(&mut inv_2x).expect("Coset points are non-zero");

        // Step 2: Pointwise decomposition.
        // H₀((g·ω^i)²) = (evals[i] + evals[i+N]) / 2
        // H₁((g·ω^i)²) = (evals[i] - evals[i+N]) / (2·g·ω^i)
        let two_inv = two_base.inv().expect("2 is non-zero in the field");
        let (h0_evals, h1_evals) = {
            #[cfg(feature = "parallel")]
            {
                let (h0, h1): (Vec<_>, Vec<_>) = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let sum = &constraint_evaluations[i] + &constraint_evaluations[i + n];
                        let diff = &constraint_evaluations[i] - &constraint_evaluations[i + n];
                        // F × E → E (base field scalar on left for mixed multiplication)
                        (&two_inv * &sum, &inv_2x[i] * &diff)
                    })
                    .unzip();
                (h0, h1)
            }
            #[cfg(not(feature = "parallel"))]
            {
                let mut h0 = Vec::with_capacity(n);
                let mut h1 = Vec::with_capacity(n);
                for i in 0..n {
                    let sum = &constraint_evaluations[i] + &constraint_evaluations[i + n];
                    let diff = &constraint_evaluations[i] - &constraint_evaluations[i + n];
                    h0.push(&two_inv * &sum);
                    h1.push(&inv_2x[i] * &diff);
                }
                (h0, h1)
            }
        };

        // Step 3: Extend each part from N evals on g²-coset to 2N evals on g-coset.
        // The squared coset offset is g² (= coset_offset²).
        let coset_offset_squared = &domain.coset_offset * &domain.coset_offset;

        #[cfg(feature = "parallel")]
        let (lde_h0, lde_h1) = rayon::join(
            || Self::extend_half_to_lde(&h0_evals, &coset_offset_squared, domain),
            || Self::extend_half_to_lde(&h1_evals, &coset_offset_squared, domain),
        );

        #[cfg(not(feature = "parallel"))]
        let (lde_h0, lde_h1) = (
            Self::extend_half_to_lde(&h0_evals, &coset_offset_squared, domain),
            Self::extend_half_to_lde(&h1_evals, &coset_offset_squared, domain),
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
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
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

        // === Phase 1: Column compression (Plonky3-style) ===
        // Instead of iterating all ~95 columns per row in the hot loop, we precompute:
        //   compressed_k[i] = Σ_j gamma[j][k] * lde_trace.get_main(i, j)   for i in 0..lde_size
        //   ood_compressed_k = Σ_j gamma[j][k] * ood[j][k]
        // This moves the column sum outside the hot loop. Since the new path evaluates
        // DEEP directly at all 2N LDE points, no stride is needed — every row is used.

        // Precompute OOD compressed values (one per eval point)
        let mut ood_compressed: Vec<FieldElement<FieldExtension>> =
            vec![FieldElement::zero(); num_eval_points];
        for j in 0..num_total_cols {
            let ood_evals_j = &trace_ood_columns[j];
            let gammas_j = &trace_terms_gammas[j];
            for k in 0..num_eval_points {
                ood_compressed[k] += &gammas_j[k] * &ood_evals_j[k];
            }
        }

        // Compressed traces at ALL 2N LDE points (Plonky3-style).
        // Eliminates the iFFT(N)+FFT(2N) extension by computing directly at LDE size.
        let compressed: Vec<Vec<FieldElement<FieldExtension>>> = (0..num_eval_points)
            .map(|k| {
                let main_gammas: Vec<&FieldElement<FieldExtension>> = (0..num_main_cols)
                    .map(|j| &trace_terms_gammas[j][k])
                    .collect();
                let aux_gammas: Vec<&FieldElement<FieldExtension>> = (0..num_aux_cols)
                    .map(|j| &trace_terms_gammas[num_main_cols + j][k])
                    .collect();

                #[cfg(feature = "parallel")]
                let iter = (0..lde_size).into_par_iter();
                #[cfg(not(feature = "parallel"))]
                let iter = 0..lde_size;

                iter.map(|i| {
                    let mut sum = FieldElement::<FieldExtension>::zero();
                    for (j, gamma) in main_gammas.iter().enumerate() {
                        sum += lde_trace.get_main(i, j) * *gamma;
                    }
                    for (j, gamma) in aux_gammas.iter().enumerate() {
                        sum += lde_trace.get_aux(i, j) * *gamma;
                    }
                    sum
                })
                .collect()
            })
            .collect();

        // Hot loop at all 2N LDE points — no FFT extension needed.
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

            // Trace terms (compressed)
            for k in 0..num_eval_points {
                let inv_t_k_i = &denoms[(1 + k) * lde_size + i];
                result += inv_t_k_i * (&compressed[k][i] - &ood_compressed[k]);
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

    /// Computes values and validity proofs of the evaluations of the trace polynomials
    /// at the domain value corresponding to the FRI query challenge `index` and its symmetric
    /// element. Gathers row data from column-major LDE storage.
    fn open_trace_polys_main(
        domain: &Domain<Field>,
        tree: &BatchedMerkleTree<Field>,
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        challenge: usize,
    ) -> PolynomialOpenings<Field>
    where
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    {
        let domain_size = domain.lde_roots_of_unity_coset.len();

        let index = challenge * 2;
        let index_sym = challenge * 2 + 1;
        PolynomialOpenings {
            proof: tree.get_proof_by_pos(index).unwrap(),
            proof_sym: tree.get_proof_by_pos(index_sym).unwrap(),
            evaluations: lde_trace.gather_main_row(reverse_index(index, domain_size as u64)),
            evaluations_sym: lde_trace
                .gather_main_row(reverse_index(index_sym, domain_size as u64)),
        }
    }

    /// Variant that opens only a range of main columns (for preprocessed tables).
    fn open_trace_polys_main_range(
        domain: &Domain<Field>,
        tree: &BatchedMerkleTree<Field>,
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        challenge: usize,
        col_start: usize,
        col_end: usize,
    ) -> PolynomialOpenings<Field>
    where
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    {
        let domain_size = domain.lde_roots_of_unity_coset.len();

        let index = challenge * 2;
        let index_sym = challenge * 2 + 1;
        PolynomialOpenings {
            proof: tree.get_proof_by_pos(index).unwrap(),
            proof_sym: tree.get_proof_by_pos(index_sym).unwrap(),
            evaluations: lde_trace.gather_main_row_range(
                reverse_index(index, domain_size as u64),
                col_start,
                col_end,
            ),
            evaluations_sym: lde_trace.gather_main_row_range(
                reverse_index(index_sym, domain_size as u64),
                col_start,
                col_end,
            ),
        }
    }

    /// Opens auxiliary trace polynomials at the given challenge index.
    fn open_trace_polys_aux(
        domain: &Domain<Field>,
        tree: &BatchedMerkleTree<FieldExtension>,
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        challenge: usize,
    ) -> PolynomialOpenings<FieldExtension>
    where
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    {
        let domain_size = domain.lde_roots_of_unity_coset.len();

        let index = challenge * 2;
        let index_sym = challenge * 2 + 1;
        PolynomialOpenings {
            proof: tree.get_proof_by_pos(index).unwrap(),
            proof_sym: tree.get_proof_by_pos(index_sym).unwrap(),
            evaluations: lde_trace.gather_aux_row(reverse_index(index, domain_size as u64)),
            evaluations_sym: lde_trace.gather_aux_row(reverse_index(index_sym, domain_size as u64)),
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

        // Check if this is a preprocessed table (has separate precomputed tree)
        let is_preprocessed = round_1_result.main.precomputed_merkle_tree.is_some();
        let num_precomputed_cols = round_1_result.main.num_precomputed_cols;
        let total_cols = round_1_result.lde_trace.num_main_cols();

        for index in indexes_to_open.iter() {
            // For preprocessed tables, open main (multiplicities) with column range
            // For normal tables, open all columns
            let main_trace_opening = if is_preprocessed {
                Self::open_trace_polys_main_range(
                    domain,
                    &round_1_result.main.lde_trace_merkle_tree,
                    &round_1_result.lde_trace,
                    *index,
                    num_precomputed_cols,
                    total_cols,
                )
            } else {
                Self::open_trace_polys_main(
                    domain,
                    &round_1_result.main.lde_trace_merkle_tree,
                    &round_1_result.lde_trace,
                    *index,
                )
            };

            // For preprocessed tables, also open precomputed tree
            let precomputed_trace_opening = round_1_result
                .main
                .precomputed_merkle_tree
                .as_ref()
                .map(|tree| {
                    Self::open_trace_polys_main_range(
                        domain,
                        tree,
                        &round_1_result.lde_trace,
                        *index,
                        0,
                        num_precomputed_cols,
                    )
                });

            let composition_openings = Self::open_composition_poly(
                &round_2_result.composition_poly_merkle_tree,
                &round_2_result.lde_composition_poly_evaluations,
                *index,
            );

            let aux_trace_polys = round_1_result.aux.as_ref().map(|aux| {
                Self::open_trace_polys_aux(
                    domain,
                    &aux.lde_trace_merkle_tree,
                    &round_1_result.lde_trace,
                    *index,
                )
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
                    let d = new_domain(*air, trace_length);
                    let t = LdeTwiddles::new(&d);
                    (Arc::new(d), Arc::new(t))
                })
                .clone();

            #[cfg(test)]
            domain_cache_stats::record(was_hit);

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

        let mut main_commits: Vec<MainCommitData<Field>> = Vec::with_capacity(num_airs);
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

                    if air.is_preprocessed() {
                        Self::commit_preprocessed_trace(
                            *trace,
                            domain,
                            air.precomputed_commitment(),
                            air.num_precomputed_columns(),
                            twiddles,
                            #[cfg(feature = "disk-spill")]
                            storage_mode,
                        )
                    } else {
                        Self::commit_main_trace(
                            *trace,
                            domain,
                            twiddles,
                            #[cfg(feature = "disk-spill")]
                            storage_mode,
                        )
                    }
                })
                .collect();

            // Sequential: append roots to shared transcript (Fiat-Shamir ordering)
            for result in chunk_results {
                let (tree, root, pre_tree, pre_root, n_pre, cached_main) = result?;
                if let Some(ref pre_r) = pre_root {
                    transcript.append_bytes(pre_r);
                }
                transcript.append_bytes(&root);
                main_commits.push(MainCommitData {
                    main_tree: Arc::new(tree),
                    main_root: root,
                    precomputed_tree: pre_tree.map(Arc::new),
                    precomputed_root: pre_root,
                    num_precomputed_cols: n_pre,
                });
                main_ldes.push(cached_main);
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

        // Parallel aux commit in chunks of K
        #[allow(clippy::type_complexity)]
        let mut aux_results: Vec<(
            Option<Arc<BatchedMerkleTree<FieldExtension>>>,
            Option<Commitment>,
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

                        Ok((Some(Arc::new(tree)), Some(root), columns))
                    } else {
                        Ok((None, None, Vec::new()))
                    }
                })
                .collect();

            // Sequential: append aux roots to forked transcripts
            for (j, result) in chunk_aux.into_iter().enumerate() {
                let (aux_tree, aux_root, cached_aux) = result?;
                if let Some(ref root) = aux_root {
                    table_transcripts[chunk_start + j].append_bytes(root);
                }
                aux_results.push((aux_tree, aux_root, cached_aux));
            }
        }

        // Build commitments and cached LDEs as separate vecs:
        // commitments are borrowed in Phase D, LDEs are consumed by value.
        let mut commitments: Vec<Round1Commitments<Field, FieldExtension>> =
            Vec::with_capacity(num_airs);
        let mut cached_ldes: Vec<Lde<Field, FieldExtension>> = Vec::with_capacity(num_airs);
        for (((main_commit, main_lde), (aux_tree, aux_root, cached_aux)), bus_public_inputs) in
            main_commits
                .into_iter()
                .zip(main_ldes)
                .zip(aux_results)
                .zip(bus_inputs_vec)
        {
            commitments.push(Round1Commitments {
                main_merkle_tree: main_commit.main_tree,
                main_merkle_root: main_commit.main_root,
                precomputed_merkle_tree: main_commit.precomputed_tree,
                precomputed_merkle_root: main_commit.precomputed_root,
                num_precomputed_cols: main_commit.num_precomputed_cols,
                aux_merkle_tree: aux_tree,
                aux_merkle_root: aux_root,
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
                    let round_1_result = commitment.build_round1(
                        lde,
                        air.step_size(),
                        domain.blowup_factor,
                        air.has_aux_trace(),
                    );

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
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
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
            // [t]
            lde_trace_main_merkle_root: round_1_result.main.lde_trace_merkle_root,
            // [t]
            lde_trace_aux_merkle_root: round_1_result.aux.as_ref().map(|x| x.lde_trace_merkle_root),
            // For preprocessed tables: commitment to precomputed columns only
            lde_trace_precomputed_merkle_root: round_1_result.main.precomputed_merkle_root,
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
