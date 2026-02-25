use std::marker::PhantomData;
use std::rc::Rc;
#[cfg(feature = "instruments")]
use std::time::Instant;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use math::fft::cpu::bit_reversing::{in_place_bit_reverse_permute, reverse_index};
use math::fft::cpu::bowers_fft::LayerTwiddles;
use math::fft::errors::FFTError;

use log::info;
use math::field::traits::{IsField, IsSubFieldOf};
use math::traits::AsBytes;
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
use crate::table::Table;
use crate::trace::LDETraceTable;

use super::config::{BatchedMerkleTree, BatchedMerkleTreeBackend, Commitment};
use super::constraints::evaluator::ConstraintEvaluator;
use super::domain::Domain;
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
{
}

#[derive(Debug)]
pub enum ProvingError {
    WrongParameter(String),
    EmptyCommitment,
}

/// A container for the intermediate results of the commitments to a trace table, main or auxiliary in case of RAP,
/// in the first round of the STARK Prove protocol.
pub struct Round1CommitmentData<F>
where
    F: IsField,
    FieldElement<F>: AsBytes,
{
    /// The Merkle trees constructed to obtain the commitment of the entire trace table.
    /// For preprocessed tables, this contains only the multiplicity columns.
    /// Wrapped in Rc to share with Round1Metadata without deep-cloning (~64MB per table).
    pub(crate) lde_trace_merkle_tree: Rc<BatchedMerkleTree<F>>,
    /// The root of the Merkle tree in `lde_trace_merkle_tree`.
    pub(crate) lde_trace_merkle_root: Commitment,
    /// For preprocessed tables: Merkle tree over precomputed columns only.
    pub(crate) precomputed_merkle_tree: Option<Rc<BatchedMerkleTree<F>>>,
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
    FieldElement<Field>: AsBytes,
    FieldElement<FieldExtension>: AsBytes,
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
    FieldElement<Field>: AsBytes,
{
    main_tree: Rc<BatchedMerkleTree<Field>>,
    main_root: Commitment,
    precomputed_tree: Option<Rc<BatchedMerkleTree<Field>>>,
    precomputed_root: Option<Commitment>,
    num_precomputed_cols: usize,
}

/// Metadata from Round 1 commitments — stores Merkle trees and roots, but not LDE evaluations.
/// LDE evaluations are recomputed from the trace in Rounds 2-4; Merkle trees are retained
/// to avoid expensive re-hashing (~500ms Keccak per proof).
pub struct Round1Metadata<Field, FieldExtension>
where
    Field: IsFFTField + IsSubFieldOf<FieldExtension>,
    FieldExtension: IsField,
    FieldElement<Field>: AsBytes,
    FieldElement<FieldExtension>: AsBytes,
{
    /// Merkle tree of the main trace (multiplicities for preprocessed tables).
    /// Wrapped in Rc to share with Round1CommitmentData without deep-cloning.
    main_merkle_tree: Rc<BatchedMerkleTree<Field>>,
    /// Root of the main trace Merkle tree.
    main_merkle_root: Commitment,
    /// For preprocessed tables: Merkle tree over precomputed columns.
    precomputed_merkle_tree: Option<Rc<BatchedMerkleTree<Field>>>,
    /// For preprocessed tables: root of the precomputed Merkle tree.
    precomputed_merkle_root: Option<Commitment>,
    /// For preprocessed tables: number of precomputed columns.
    num_precomputed_cols: usize,
    /// Merkle tree of the auxiliary trace (None if no aux trace).
    aux_merkle_tree: Option<Rc<BatchedMerkleTree<FieldExtension>>>,
    /// Root of the auxiliary trace Merkle tree (None if no aux trace).
    aux_merkle_root: Option<Commitment>,
    /// The RAP challenges used for auxiliary trace construction.
    rap_challenges: Vec<FieldElement<FieldExtension>>,
    /// Bus interaction public inputs (initial and final aux column values).
    bus_public_inputs: Option<BusPublicInputs<FieldExtension>>,
}

/// Pre-computed twiddle factors and coset weights for a given domain size.
///
/// Shared across all columns of the same table, and across all phases (A, C, Rounds 2-4)
/// in sequential proving. This eliminates redundant root-of-unity generation:
/// ~700 `get_twiddles` calls per proof reduced to one pair per distinct domain.
///
/// The `coset_weights` vector stores `[n_inv, n_inv*g, n_inv*g², ..., n_inv*g^{n-1}]`
/// where `g` is the coset offset and `n_inv = 1/n`. These are used in the iFFT+coset-shift
/// step of `expand_pool_to_lde`.
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

/// A container for the results of the second round of the STARK Prove protocol.
pub struct Round2<F>
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
pub struct Round3<F: IsField> {
    /// Evaluations of the trace polynomials, main ans auxiliary, at the out-of-domain challenge.
    trace_ood_evaluations: Table<F>,
    /// Evaluations of the composition polynomial parts at the out-of-domain challenge.
    composition_poly_parts_ood_evaluation: Vec<FieldElement<F>>,
}

/// A container for the results of the fourth round of the STARK Prove protocol.
pub struct Round4<F: IsSubFieldOf<E>, E: IsField> {
    /// Coefficients of the final FRI polynomial (single constant for default config).
    fri_final_poly: Vec<FieldElement<E>>,
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

/// The functionality of a STARK prover providing methods to run the STARK Prove protocol
/// https://lambdaclass.github.io/lambdaworks/starks/protocol.html
/// The default implementation is complete and is compatible with Stone prover
/// https://github.com/starkware-libs/stone-prover
pub trait IsStarkProver<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
    PI,
>
{
    /// Returns the Merkle tree and the commitment to the vectors `vectors`.
    fn batch_commit_main(
        vectors: &[Vec<FieldElement<Field>>],
    ) -> Option<(BatchedMerkleTree<Field>, Commitment)>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
    {
        let tree = BatchedMerkleTree::build(vectors)?;

        let commitment = tree.root;
        Some((tree, commitment))
    }

    /// Returns the Merkle tree and the commitment to the vectors `vectors`.
    fn batch_commit_extension(
        vectors: &[Vec<FieldElement<FieldExtension>>],
    ) -> Option<(BatchedMerkleTree<FieldExtension>, Commitment)>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let tree = BatchedMerkleTree::build(vectors)?;

        let commitment = tree.root;
        Some((tree, commitment))
    }

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
        FieldElement<E>: AsBytes + Sync + Send,
        E: IsField,
    {
        if columns.is_empty() || columns[0].is_empty() {
            return None;
        }

        let num_rows = columns[0].len();
        let num_cols = columns.len();

        #[cfg(feature = "parallel")]
        let iter = (0..num_rows).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = 0..num_rows;

        let hashed_leaves: Vec<Commitment> = iter
            .map(|row_idx| {
                let br_idx = reverse_index(row_idx, num_rows as u64);
                let row: Vec<FieldElement<E>> = (0..num_cols)
                    .map(|col_idx| columns[col_idx][br_idx].clone())
                    .collect();
                BatchedMerkleTreeBackend::<E>::hash_data(&row)
            })
            .collect();

        let tree = BatchedMerkleTree::<E>::build_from_hashed_leaves(hashed_leaves)?;
        let root = tree.root;
        Some((tree, root))
    }

    /// Commit columns that are already in **bit-reversed order** (as produced by
    /// `coset_lde_full_expand_bitrev` / `coset_lde_full_into_bitrev`).
    ///
    /// Since the columns are already bit-reversed, we iterate rows sequentially
    /// instead of using `reverse_index`. Produces the same Merkle tree as
    /// `commit_columns_bit_reversed` when given the corresponding natural-order data.
    fn commit_columns<E>(
        columns: &[Vec<FieldElement<E>>],
    ) -> Option<(BatchedMerkleTree<E>, Commitment)>
    where
        FieldElement<E>: AsBytes + Sync + Send,
        E: IsField,
    {
        if columns.is_empty() || columns[0].is_empty() {
            return None;
        }

        let num_rows = columns[0].len();
        let num_cols = columns.len();

        #[cfg(feature = "parallel")]
        let iter = (0..num_rows).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = 0..num_rows;

        let hashed_leaves: Vec<Commitment> = iter
            .map(|row_idx| {
                let row: Vec<FieldElement<E>> = (0..num_cols)
                    .map(|col_idx| columns[col_idx][row_idx].clone())
                    .collect();
                BatchedMerkleTreeBackend::<E>::hash_data(&row)
            })
            .collect();

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
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        // Create domain for LDE
        let domain = Domain::new(air, trace.num_rows());

        // Interpolate columns to polynomials
        let trace_polys = trace.compute_trace_polys_main::<Field>();

        // Keep only precomputed columns
        let precomputed_polys: Vec<_> =
            trace_polys.into_iter().take(num_precomputed_cols).collect();

        // Evaluate on LDE domain
        let evaluations = Self::compute_lde_trace_evaluations::<Field>(&precomputed_polys, &domain);

        let (_, commitment) = Self::commit_columns_bit_reversed(&evaluations)?;

        Some(commitment)
    }

    /// Given a `TraceTable`, this method interpolates its columns, computes the commitment to the
    /// table and appends it to the transcript.
    /// Output: a tuple of length 3 with the following:
    /// • The evaluations of the trace polynomials over the LDE domain.
    /// • The Merkle tree of evaluations over the LDE domain.
    /// • The root of the above Merkle tree.
    ///
    /// Note: trace polynomials (coefficients) are computed transiently for the LDE
    /// but are NOT stored. All downstream operations use evaluation-form data.
    #[allow(clippy::type_complexity)]
    fn interpolate_and_commit_main(
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Option<(
        Vec<Vec<FieldElement<Field>>>,
        BatchedMerkleTree<Field>,
        Commitment,
    )>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        Field: IsSubFieldOf<FieldExtension>,
    {
        // Compute LDE evaluations directly from columns using fused coset LDE
        // (no intermediate coefficient-form polynomials).
        let columns = trace.columns_main();
        let lde_trace_evaluations = Self::compute_lde_from_columns::<Field>(&columns, domain);

        // Hash rows on-the-fly from column-major data with bit-reversal,
        // avoiding the full clone + transpose that the old code did.
        let (lde_trace_merkle_tree, lde_trace_merkle_root) =
            Self::commit_columns_bit_reversed(&lde_trace_evaluations)?;

        // >>>> Send commitment.
        transcript.append_bytes(&lde_trace_merkle_root);

        Some((
            lde_trace_evaluations,
            lde_trace_merkle_tree,
            lde_trace_merkle_root,
        ))
    }

    /// Interpolates columns of a preprocessed trace table and computes LDE evaluations.
    ///
    /// Does NOT build a Merkle tree (the caller builds separate trees for
    /// precomputed and multiplicity columns).
    ///
    /// Trace polynomials (coefficients) are transient and not stored.
    fn interpolate_and_compute_lde(
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
    ) -> Option<Vec<Vec<FieldElement<Field>>>>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        Field: IsSubFieldOf<FieldExtension>,
    {
        // Compute LDE evaluations directly from columns using fused coset LDE.
        let columns = trace.columns_main();
        let lde_trace_evaluations = Self::compute_lde_from_columns::<Field>(&columns, domain);

        Some(lde_trace_evaluations)
    }

    /// Given a `TraceTable`, this method interpolates its auxiliary columns, computes the
    /// commitment to the table and appends it to the transcript.
    #[allow(clippy::type_complexity)]
    fn interpolate_and_commit_aux(
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Option<(
        Vec<Vec<FieldElement<FieldExtension>>>,
        BatchedMerkleTree<FieldExtension>,
        Commitment,
    )>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        Field: IsSubFieldOf<FieldExtension> + IsFFTField,
    {
        // Compute LDE evaluations directly from aux columns using fused coset LDE.
        let columns = trace.columns_aux();
        let lde_trace_evaluations =
            Self::compute_lde_from_columns::<FieldExtension>(&columns, domain);

        // Hash rows on-the-fly from column-major data with bit-reversal.
        let (lde_trace_merkle_tree, lde_trace_merkle_root) =
            Self::commit_columns_bit_reversed(&lde_trace_evaluations)?;

        // >>>> Send commitment.
        transcript.append_bytes(&lde_trace_merkle_root);

        Some((
            lde_trace_evaluations,
            lde_trace_merkle_tree,
            lde_trace_merkle_root,
        ))
    }

    /// Evaluate polynomials `trace_polys` over the domain `domain`.
    /// The i-th entry of the returned vector contains the evaluations of the i-th polynomial in `trace_polys`.
    fn compute_lde_trace_evaluations<E>(
        trace_polys: &[Polynomial<FieldElement<E>>],
        domain: &Domain<Field>,
    ) -> Vec<Vec<FieldElement<E>>>
    where
        E: IsSubFieldOf<FieldExtension> + Send + Sync,
        Field: IsSubFieldOf<E>,
    {
        #[cfg(not(feature = "parallel"))]
        let trace_polys_iter = trace_polys.iter();
        #[cfg(feature = "parallel")]
        let trace_polys_iter = trace_polys.par_iter();

        trace_polys_iter
            .map(|poly| {
                evaluate_polynomial_on_lde_domain(
                    poly,
                    domain.blowup_factor,
                    domain.interpolation_domain_size,
                    &domain.coset_offset,
                )
            })
            .collect::<Result<Vec<Vec<FieldElement<E>>>, FFTError>>()
            .unwrap()
    }

    /// Compute LDE evaluations directly from column data using the fused coset LDE.
    ///
    /// This replaces the `compute_trace_polys_*` + `compute_lde_trace_evaluations` pipeline
    /// with a single `coset_lde` call per column, saving 2 intermediate allocations per column.
    ///
    /// Twiddle factors and coset weights are precomputed once and shared across all columns,
    /// eliminating redundant root-of-unity generation.
    fn compute_lde_from_columns<E>(
        columns: &[Vec<FieldElement<E>>],
        domain: &Domain<Field>,
    ) -> Vec<Vec<FieldElement<E>>>
    where
        E: IsSubFieldOf<FieldExtension> + Send + Sync,
        Field: IsSubFieldOf<E>,
        FieldElement<E>: Send + Sync,
    {
        if columns.is_empty() {
            return Vec::new();
        }

        let twiddles = LdeTwiddles::new(domain);
        Self::compute_lde_from_columns_cached(columns, domain, &twiddles)
    }

    /// Compute LDE evaluations with pre-computed twiddle factors and coset weights.
    ///
    /// Same as [`compute_lde_from_columns`] but accepts shared [`LdeTwiddles`] to avoid
    /// redundant twiddle generation and weight computation across phases (A, C, Rounds 2-4).
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

    /// Compute LDE evaluations into pre-allocated column buffers.
    ///
    /// Same as [`compute_lde_from_columns_cached`] but writes into `output` buffers instead of
    /// allocating new Vecs. When `output[i].capacity() >= lde_size`, no heap allocation occurs.
    /// Used for LDE buffer reuse across tables in sequential proving.
    fn compute_lde_from_columns_into<E>(
        columns: &[Vec<FieldElement<E>>],
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
        output: &mut [Vec<FieldElement<E>>],
    ) where
        E: IsSubFieldOf<FieldExtension> + Send + Sync,
        Field: IsSubFieldOf<E>,
        FieldElement<E>: Send + Sync,
    {
        if columns.is_empty() {
            return;
        }

        debug_assert!(
            output.len() >= columns.len(),
            "output pool has {} buffers but need {}",
            output.len(),
            columns.len()
        );

        #[cfg(not(feature = "parallel"))]
        {
            for (col, buf) in columns.iter().zip(output.iter_mut()) {
                Polynomial::coset_lde_full_into::<Field>(
                    col,
                    domain.blowup_factor,
                    &twiddles.coset_weights,
                    &twiddles.inv,
                    &twiddles.fwd,
                    buf,
                )
                .expect("coset LDE into");
            }
        }
        #[cfg(feature = "parallel")]
        {
            columns
                .par_iter()
                .zip(output.par_iter_mut())
                .for_each(|(col, buf)| {
                    Polynomial::coset_lde_full_into::<Field>(
                        col,
                        domain.blowup_factor,
                        &twiddles.coset_weights,
                        &twiddles.inv,
                        &twiddles.fwd,
                        buf,
                    )
                    .expect("coset LDE into");
                });
        }
    }

    /// Expand pool buffers in-place from N column evaluations to N×blowup LDE evaluations.
    ///
    /// The pool buffers already contain column data extracted via `extract_columns_*_into`.
    /// This performs iFFT + coset shift + FFT in-place, eliminating the T1 transpose copy.
    /// Coset weights are pre-cached in `LdeTwiddles` to avoid recomputation across phases.
    fn expand_pool_to_lde<E>(
        pool: &mut [Vec<FieldElement<E>>],
        num_cols: usize,
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
    ) where
        Field: IsSubFieldOf<E>,
        E: IsSubFieldOf<FieldExtension> + IsField + Send + Sync,
        FieldElement<E>: Send + Sync,
    {
        if num_cols == 0 {
            return;
        }

        #[cfg(feature = "parallel")]
        pool[..num_cols].par_iter_mut().for_each(|buf| {
            Polynomial::coset_lde_full_expand::<Field>(
                buf,
                domain.blowup_factor,
                &twiddles.coset_weights,
                &twiddles.inv,
                &twiddles.fwd,
            )
            .expect("coset LDE expansion");
        });
        #[cfg(not(feature = "parallel"))]
        for buf in pool[..num_cols].iter_mut() {
            Polynomial::coset_lde_full_expand::<Field>(
                buf,
                domain.blowup_factor,
                &twiddles.coset_weights,
                &twiddles.inv,
                &twiddles.fwd,
            )
            .expect("coset LDE expansion");
        }
    }

    /// Same as [`expand_pool_to_lde`], but returns evaluations in **bit-reversed order**.
    /// Use for commit-only paths where data won't be used for constraint evaluation.
    fn expand_pool_to_lde_bitrev<E>(
        pool: &mut [Vec<FieldElement<E>>],
        num_cols: usize,
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
    ) where
        Field: IsSubFieldOf<E>,
        E: IsSubFieldOf<FieldExtension> + IsField + Send + Sync,
        FieldElement<E>: Send + Sync,
    {
        if num_cols == 0 {
            return;
        }

        #[cfg(feature = "parallel")]
        pool[..num_cols].par_iter_mut().for_each(|buf| {
            Polynomial::coset_lde_full_expand_bitrev::<Field>(
                buf,
                domain.blowup_factor,
                &twiddles.coset_weights,
                &twiddles.inv,
                &twiddles.fwd,
            )
            .expect("coset LDE expansion");
        });
        #[cfg(not(feature = "parallel"))]
        for buf in pool[..num_cols].iter_mut() {
            Polynomial::coset_lde_full_expand_bitrev::<Field>(
                buf,
                domain.blowup_factor,
                &twiddles.coset_weights,
                &twiddles.inv,
                &twiddles.fwd,
            )
            .expect("coset LDE expansion");
        }
    }

    /// Phase 1a of Round 1: Commit only the main trace to the transcript.
    /// Returns the main trace commitment data and LDE evaluations.
    /// Does NOT sample RAP challenges or build auxiliary trace.
    #[allow(clippy::type_complexity)]
    fn round_1_commit_main_trace(
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Result<(Round1CommitmentData<Field>, Vec<Vec<FieldElement<Field>>>), ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let Some((evaluations, main_merkle_tree, main_merkle_root)) =
            Self::interpolate_and_commit_main(trace, domain, transcript)
        else {
            return Err(ProvingError::EmptyCommitment);
        };

        let main = Round1CommitmentData::<Field> {
            lde_trace_merkle_tree: Rc::new(main_merkle_tree),
            lde_trace_merkle_root: main_merkle_root,
            precomputed_merkle_tree: None,
            precomputed_merkle_root: None,
            num_precomputed_cols: 0,
        };

        Ok((main, evaluations))
    }

    /// Phase 1a variant for preprocessed tables: commits precomputed and multiplicities separately.
    ///
    /// For preprocessed tables (e.g., bitwise lookup):
    /// - Precomputed columns (0..num_precomputed_cols): separate tree, root must match hardcoded
    /// - Multiplicity columns (num_precomputed_cols..): separate tree, root in proof
    ///
    /// Both commitments are added to the transcript for Fiat-Shamir binding.
    #[allow(clippy::type_complexity)]
    fn round_1_commit_preprocessed_trace(
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        precomputed_commitment: Commitment,
        num_precomputed_cols: usize,
    ) -> Result<(Round1CommitmentData<Field>, Vec<Vec<FieldElement<Field>>>), ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        // Interpolate all columns and compute LDE evaluations (no tree building here).
        let Some(evaluations) = Self::interpolate_and_compute_lde(trace, domain) else {
            return Err(ProvingError::EmptyCommitment);
        };

        // --- Build PRECOMPUTED tree (cols 0..num_precomputed) ---
        let (precomputed_tree, precomputed_root) =
            Self::commit_columns_bit_reversed(&evaluations[..num_precomputed_cols])
                .ok_or(ProvingError::EmptyCommitment)?;

        // --- Build MULTIPLICITIES tree (cols num_precomputed..) ---
        let (mult_tree, mult_root) =
            Self::commit_columns_bit_reversed(&evaluations[num_precomputed_cols..])
                .ok_or(ProvingError::EmptyCommitment)?;

        // Verify that our computed precomputed root matches the hardcoded commitment.
        // This is a sanity check - if they don't match, something is wrong with the trace.
        debug_assert_eq!(
            precomputed_root, precomputed_commitment,
            "Prover's precomputed commitment doesn't match hardcoded AIR commitment"
        );

        // Add BOTH commitments to transcript for Fiat-Shamir binding.
        // The precomputed commitment binds challenges to the correct precomputed values.
        // The multiplicities commitment binds challenges to the actual lookups made.
        transcript.append_bytes(&precomputed_commitment);
        transcript.append_bytes(&mult_root);

        // Store multiplicities tree as main (for FRI openings), precomputed tree separately
        let main = Round1CommitmentData::<Field> {
            lde_trace_merkle_tree: Rc::new(mult_tree),
            lde_trace_merkle_root: mult_root,
            precomputed_merkle_tree: Some(Rc::new(precomputed_tree)),
            precomputed_merkle_root: Some(precomputed_root),
            num_precomputed_cols,
        };

        // Return full evaluations (all columns) for constraint evaluation
        Ok((main, evaluations))
    }

    /// Phase 1c of Round 1: Build and commit auxiliary trace using pre-sampled challenges.
    /// This is called after all main traces are committed and shared challenges are sampled.
    #[allow(clippy::type_complexity)]
    fn round_1_build_auxiliary_trace(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        trace: &mut TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        main: Round1CommitmentData<Field>,
        main_evaluations: Vec<Vec<FieldElement<Field>>>,
        rap_challenges: Vec<FieldElement<FieldExtension>>,
    ) -> Result<Round1<Field, FieldExtension>, ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let (aux, aux_evaluations, bus_public_inputs) = if air.has_trace_interaction() {
            let bus_public_inputs = air.build_auxiliary_trace(trace, &rap_challenges);
            let Some((aux_trace_polys_evaluations, aux_merkle_tree, aux_merkle_root)) =
                Self::interpolate_and_commit_aux(trace, domain, transcript)
            else {
                return Err(ProvingError::EmptyCommitment);
            };
            let aux = Some(Round1CommitmentData::<FieldExtension> {
                lde_trace_merkle_tree: Rc::new(aux_merkle_tree),
                lde_trace_merkle_root: aux_merkle_root,
                precomputed_merkle_tree: None,
                precomputed_merkle_root: None,
                num_precomputed_cols: 0,
            });
            (aux, aux_trace_polys_evaluations, bus_public_inputs)
        } else {
            (None, Vec::new(), None)
        };

        let lde_trace = LDETraceTable::from_columns(
            main_evaluations,
            aux_evaluations,
            air.step_size(),
            domain.blowup_factor,
        );

        Ok(Round1 {
            lde_trace,
            main,
            aux,
            rap_challenges,
            bus_public_inputs,
        })
    }

    /// Phase A helper for sequential proving: compute main LDE, commit, return tree and root.
    /// Uses the provided pool buffers to avoid allocation; the pool retains capacity for reuse.
    fn commit_main_trace_lightweight(
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        twiddles: &LdeTwiddles<Field>,
        main_pool: &mut [Vec<FieldElement<Field>>],
    ) -> Result<
        (
            BatchedMerkleTree<Field>,
            Commitment,
            Option<BatchedMerkleTree<Field>>,
            Option<Commitment>,
            usize,
        ),
        ProvingError,
    >
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let num_cols = trace.num_main_columns;
        trace.extract_columns_main_into(main_pool);
        Self::expand_pool_to_lde_bitrev::<Field>(main_pool, num_cols, domain, twiddles);

        let (tree, root) = Self::commit_columns(&main_pool[..num_cols])
            .ok_or(ProvingError::EmptyCommitment)?;

        transcript.append_bytes(&root);

        // Pool buffers retain capacity; tree is returned to caller
        Ok((tree, root, None, None, 0))
    }

    /// Phase A helper for sequential proving with preprocessed tables: commit separately,
    /// return trees and roots. Uses pool buffers to avoid allocation.
    fn commit_preprocessed_trace_lightweight(
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        precomputed_commitment: Commitment,
        num_precomputed_cols: usize,
        twiddles: &LdeTwiddles<Field>,
        main_pool: &mut [Vec<FieldElement<Field>>],
    ) -> Result<
        (
            BatchedMerkleTree<Field>,
            Commitment,
            Option<BatchedMerkleTree<Field>>,
            Option<Commitment>,
            usize,
        ),
        ProvingError,
    >
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let num_cols = trace.num_main_columns;
        trace.extract_columns_main_into(main_pool);
        Self::expand_pool_to_lde_bitrev::<Field>(main_pool, num_cols, domain, twiddles);

        let (precomputed_tree, precomputed_root) =
            Self::commit_columns(&main_pool[..num_precomputed_cols])
                .ok_or(ProvingError::EmptyCommitment)?;

        let (mult_tree, mult_root) =
            Self::commit_columns(&main_pool[num_precomputed_cols..num_cols])
                .ok_or(ProvingError::EmptyCommitment)?;

        debug_assert_eq!(
            precomputed_root, precomputed_commitment,
            "Prover's precomputed commitment doesn't match hardcoded AIR commitment"
        );

        transcript.append_bytes(&precomputed_commitment);
        transcript.append_bytes(&mult_root);

        // Pool buffers retain capacity; trees are returned to caller
        Ok((
            mult_tree,
            mult_root,
            Some(precomputed_tree),
            Some(precomputed_root),
            num_precomputed_cols,
        ))
    }

    /// Phase C helper for sequential proving: build aux trace, commit, return tree, root,
    /// and bus_public_inputs. Uses pool buffers to avoid allocation.
    fn commit_aux_trace_lightweight(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        trace: &mut TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        rap_challenges: &[FieldElement<FieldExtension>],
        twiddles: &LdeTwiddles<Field>,
        aux_pool: &mut [Vec<FieldElement<FieldExtension>>],
    ) -> Result<
        (
            Option<BatchedMerkleTree<FieldExtension>>,
            Option<Commitment>,
            Option<BusPublicInputs<FieldExtension>>,
        ),
        ProvingError,
    >
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        if !air.has_trace_interaction() {
            return Ok((None, None, None));
        }

        let bus_public_inputs = air.build_auxiliary_trace(trace, rap_challenges);

        let num_aux_cols = trace.num_aux_columns;
        trace.extract_columns_aux_into(aux_pool);
        Self::expand_pool_to_lde_bitrev::<FieldExtension>(aux_pool, num_aux_cols, domain, twiddles);

        let (aux_tree, aux_merkle_root) =
            Self::commit_columns(&aux_pool[..num_aux_cols])
                .ok_or(ProvingError::EmptyCommitment)?;

        transcript.append_bytes(&aux_merkle_root);

        // Pool buffers retain capacity; tree is returned to caller
        Ok((Some(aux_tree), Some(aux_merkle_root), bus_public_inputs))
    }

    /// Reconstruct a full Round1 struct by recomputing LDE evaluations and using
    /// the stored Merkle trees from metadata. Uses pool buffers to avoid allocation.
    ///
    /// The Merkle trees were already built during Phase A/C and are reused here,
    /// eliminating ~500ms of Keccak hashing per proof.
    fn reconstruct_round1(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        metadata: &Round1Metadata<Field, FieldExtension>,
        twiddles: &LdeTwiddles<Field>,
        main_pool: &mut [Vec<FieldElement<Field>>],
        aux_pool: &mut [Vec<FieldElement<FieldExtension>>],
    ) -> Result<Round1<Field, FieldExtension>, ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        // Recompute main LDE into pool buffers (extract columns directly, no T1 transpose)
        let num_main_cols = trace.num_main_columns;
        trace.extract_columns_main_into(main_pool);
        Self::expand_pool_to_lde::<Field>(main_pool, num_main_cols, domain, twiddles);

        // Use stored Merkle trees from Phase A/C via Rc (pointer copy, no deep clone)
        let main = Round1CommitmentData::<Field> {
            lde_trace_merkle_tree: Rc::clone(&metadata.main_merkle_tree),
            lde_trace_merkle_root: metadata.main_merkle_root,
            precomputed_merkle_tree: metadata.precomputed_merkle_tree.as_ref().map(Rc::clone),
            precomputed_merkle_root: metadata.precomputed_merkle_root,
            num_precomputed_cols: metadata.num_precomputed_cols,
        };

        // Recompute aux LDE into pool buffers, use stored aux Merkle tree
        let (aux, num_aux_cols) = if air.has_trace_interaction() {
            let n_aux = trace.num_aux_columns;
            trace.extract_columns_aux_into(aux_pool);
            Self::expand_pool_to_lde::<FieldExtension>(aux_pool, n_aux, domain, twiddles);
            // Safe: has_trace_interaction() is true only when Phase C stored aux tree/root
            let aux_commitment = Round1CommitmentData::<FieldExtension> {
                lde_trace_merkle_tree: Rc::clone(
                    metadata.aux_merkle_tree.as_ref()
                        .expect("aux tree must exist when has_trace_interaction"),
                ),
                lde_trace_merkle_root: metadata
                    .aux_merkle_root
                    .expect("aux root must exist when has_trace_interaction"),
                precomputed_merkle_tree: None,
                precomputed_merkle_root: None,
                num_precomputed_cols: 0,
            };
            (Some(aux_commitment), n_aux)
        } else {
            (None, 0)
        };

        // Take column Vecs from pool (zero-copy move) instead of cloning.
        // After prove_rounds_2_to_4, columns are returned to the pool via into_columns.
        let main_cols: Vec<_> = main_pool[..num_main_cols]
            .iter_mut()
            .map(std::mem::take)
            .collect();
        let aux_cols: Vec<_> = aux_pool[..num_aux_cols]
            .iter_mut()
            .map(std::mem::take)
            .collect();
        let lde_trace = LDETraceTable::from_columns(
            main_cols,
            aux_cols,
            air.step_size(),
            domain.blowup_factor,
        );

        Ok(Round1 {
            lde_trace,
            main,
            aux,
            rap_challenges: metadata.rap_challenges.clone(),
            bus_public_inputs: metadata.bus_public_inputs.clone(),
        })
    }

    /// Returns the Merkle tree and the commitment to the evaluations of the parts of the
    /// composition polynomial.
    fn commit_composition_polynomial(
        lde_composition_poly_parts_evaluations: &[Vec<FieldElement<FieldExtension>>],
    ) -> Option<(BatchedMerkleTree<FieldExtension>, Commitment)>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let num_parts = lde_composition_poly_parts_evaluations.len();
        let num_rows = lde_composition_poly_parts_evaluations[0].len();

        // Transpose columns → rows with pre-allocated capacity.
        let mut rows: Vec<Vec<FieldElement<FieldExtension>>> = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let mut row = Vec::with_capacity(num_parts);
            for part in lde_composition_poly_parts_evaluations.iter() {
                row.push(part[i].clone());
            }
            rows.push(row);
        }

        in_place_bit_reverse_permute(&mut rows);

        // Merge consecutive pairs: [row0, row1] → row0 ++ row1.
        // Drain pairs from the original vec to avoid cloning.
        let mut merged: Vec<Vec<FieldElement<FieldExtension>>> = Vec::with_capacity(num_rows / 2);
        let mut iter = rows.into_iter();
        while let (Some(mut first), Some(second)) = (iter.next(), iter.next()) {
            first.extend(second);
            merged.push(first);
        }

        Self::batch_commit_extension(&merged)
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
    fn decompose_and_extend_d2<E>(
        constraint_evaluations: &[FieldElement<FieldExtension>],
        domain: &Domain<Field>,
    ) -> Vec<Vec<FieldElement<FieldExtension>>>
    where
        E: IsField,
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let two_n = constraint_evaluations.len();
        let n = two_n / 2;
        debug_assert_eq!(two_n, n * 2);

        // Step 1: Compute 1/(2·g·ω^i) for i=0..N-1 via batch inversion.
        // The LDE coset points are g·ω^i = domain.lde_roots_of_unity_coset[i].
        // Compute entirely in base field — mixed F×E multiplication when used with extension values.
        let mut inv_2x: Vec<FieldElement<Field>> = (0..n)
            .map(|i| domain.lde_roots_of_unity_coset[i].double())
            .collect();
        FieldElement::inplace_batch_inverse(&mut inv_2x)
            .expect("Coset points are non-zero");

        // Step 2: Pointwise decomposition.
        // H₀((g·ω^i)²) = (evals[i] + evals[i+N]) / 2
        // H₁((g·ω^i)²) = (evals[i] - evals[i+N]) / (2·g·ω^i)
        let two_inv = FieldElement::<Field>::from(2u64)
            .inv()
            .expect("2 is non-zero in the field");
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
            || {
                Self::extend_half_to_lde(
                    &h0_evals,
                    &coset_offset_squared,
                    domain,
                )
            },
            || {
                Self::extend_half_to_lde(
                    &h1_evals,
                    &coset_offset_squared,
                    domain,
                )
            },
        );

        #[cfg(not(feature = "parallel"))]
        let (lde_h0, lde_h1) = (
            Self::extend_half_to_lde(
                &h0_evals,
                &coset_offset_squared,
                domain,
            ),
            Self::extend_half_to_lde(
                &h1_evals,
                &coset_offset_squared,
                domain,
            ),
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
        let constraint_evaluations = evaluator.evaluate(
            air,
            &round_1_result.lde_trace,
            domain,
            transition_coefficients,
            boundary_coefficients,
            &round_1_result.rap_challenges,
        );

        let number_of_parts = air.composition_poly_degree_bound(trace_length) / trace_length;

        let lde_composition_poly_parts_evaluations = if number_of_parts == 2 {
            // Direct quotient decomposition: avoid full-size iFFT by algebraically
            // splitting H(x) = H₀(x²) + x·H₁(x²) using:
            //   H₀(x²) = (H(x) + H(-x)) / 2
            //   H₁(x²) = (H(x) - H(-x)) / (2x)
            // On the LDE coset {g·ω^i}, we have -g·ω^i = g·ω^{i+N} since ω^N = -1.
            Self::decompose_and_extend_d2::<FieldExtension>(
                &constraint_evaluations,
                domain,
            )
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

        let Some((composition_poly_merkle_tree, composition_poly_root)) =
            Self::commit_composition_polynomial(&lde_composition_poly_parts_evaluations)
        else {
            return Err(ProvingError::EmptyCommitment);
        };

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

        // === Composition poly parts: barycentric evaluation at z^num_parts ===
        // Extract trace-size coset points from the LDE coset (stride = blowup_factor)
        // Keep coset points in base field — mixed F×E arithmetic is cheaper than E×E.
        let coset_points: Vec<FieldElement<Field>> = (0..domain_size)
            .map(|i| domain.lde_roots_of_unity_coset[i * blowup_factor].clone())
            .collect();
        // Keep coset_offset_pow_n and g_n_inv in base field F — the barycentric
        // functions use F×E→E mixed arithmetic, avoiding field conversions.
        let coset_offset_pow_n: FieldElement<Field> = domain.coset_offset.pow(domain_size);
        let domain_size_inv: FieldElement<Field> = FieldElement::<Field>::from(domain_size as u64)
            .inv()
            .expect("domain_size is a power of two, hence non-zero in the field");
        let g_n_inv: FieldElement<Field> = coset_offset_pow_n
            .inv()
            .expect("coset_offset_pow_n is non-zero");

        // Precompute inv_denoms for z^num_parts (shared across all composition poly parts)
        let comp_z_pow_n = z_power.pow(domain_size);
        let comp_inv_denoms =
            math::polynomial::barycentric_inv_denoms(&z_power, &coset_points);

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
                    &coset_offset_pow_n,
                    &domain_size_inv,
                    &g_n_inv,
                    &coset_points,
                    &evals,
                    &comp_inv_denoms,
                )
            })
            .collect();

        // === Trace polynomials: barycentric evaluation via LDE ===
        // Uses get_trace_evaluations_from_lde which performs barycentric interpolation
        // on the LDE trace data, avoiding the need for coefficient-form trace_polys.
        let trace_ood_evaluations = crate::trace::get_trace_evaluations_from_lde(
            &round_1_result.lde_trace,
            domain,
            z,
            &air.context().transition_offsets,
            air.step_size(),
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

        // Extend N trace-coset evaluations to 2N LDE-coset evaluations via standard LDE.
        // deep_evals[i] = h(offset·ω_N^i) = f(ω_N^i) where f(x) = h(offset·x).
        // Standard iFFT+FFT recovers f and evaluates on the 2N-th roots: f(Ω^j) = h(offset·Ω^j).
        // Use bitrev variant: FRI needs bit-reversed order, and FFT naturally produces it.
        // This skips two redundant bit-reverse permutations (FFT→BRP→natural→BRP→bitrev).
        let domain_size = domain.lde_roots_of_unity_coset.len();
        let deep_poly = Polynomial::interpolate_fft::<Field>(&deep_evals)
            .expect("iFFT should succeed");
        let lde_evals = Polynomial::evaluate_fft_bitrev::<Field>(
            &deep_poly, 1, Some(domain_size),
        ).expect("FFT should succeed");

        // FRI commit phase from pre-computed evaluations (no initial FFT)
        let (fri_final_poly, fri_layers) = fri::commit_phase_from_evaluations::<Field, FieldExtension>(
            domain.root_order as usize,
            lde_evals,
            transcript,
            &coset_offset,
            domain_size,
        );

        // grinding: generate nonce and append it to the transcript
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

        Round4 {
            fri_final_poly,
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

    /// Computes the DEEP composition polynomial as evaluations on the trace-size coset.
    ///
    /// Evaluates `deep(x_i)` at N points (every bf-th point of the LDE coset).
    /// The caller extends to the full 2N-point LDE domain before feeding to FRI.
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
        let domain_size = domain.interpolation_domain_size;
        let blowup_factor = domain.blowup_factor;
        let num_parts = round_2_result.lde_composition_poly_evaluations.len();
        let z_power = z.pow(num_parts); // pole for H terms

        // Number of evaluation points per trace column (= transition_offsets.len() * step_size)
        let num_eval_points = if trace_terms_gammas.is_empty() {
            0
        } else {
            trace_terms_gammas[0].len()
        };

        // Trace poles: z_shifted[k] = primitive_root^k * z for k = 0..num_eval_points
        let z_shifted: Vec<FieldElement<FieldExtension>> = (0..num_eval_points)
            .map(|k| primitive_root.pow(k) * z)
            .collect();

        // Number of main and aux columns in the LDE trace
        let num_main_cols = lde_trace.num_main_cols();
        let num_aux_cols = lde_trace.num_aux_cols();

        // Precompute all inverse denominators via batch inversion.
        let num_denoms = domain_size * (1 + num_eval_points);
        let mut denoms: Vec<FieldElement<FieldExtension>> = Vec::with_capacity(num_denoms);

        // H-term denominators: x_i - z^K
        for i in 0..domain_size {
            let x_i = &domain.lde_roots_of_unity_coset[i * blowup_factor];
            denoms.push(x_i - &z_power);
        }

        // Trace-term denominators: x_i - z_shifted[k]
        for k in 0..num_eval_points {
            for i in 0..domain_size {
                let x_i = &domain.lde_roots_of_unity_coset[i * blowup_factor];
                denoms.push(x_i - &z_shifted[k]);
            }
        }

        FieldElement::inplace_batch_inverse(&mut denoms)
            .expect("Denominators should be non-zero: coset points are base field, poles are extension field");

        let inv_h = &denoms[0..domain_size];

        // OOD evaluations
        let h_ood = &round_3_result.composition_poly_parts_ood_evaluation;
        let trace_ood_columns = round_3_result.trace_ood_evaluations.columns();

        // Compute deep(x_i) for each trace-size coset point
        #[cfg(feature = "parallel")]
        let iter = (0..domain_size).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = 0..domain_size;

        iter.map(|i| {
            let row_idx = i * blowup_factor; // LDE row index

            // H terms: Σ_j γ_j * (H_j(x_i) - H_j(z^K)) * inv_h[i]
            let mut result = FieldElement::<FieldExtension>::zero();
            for j in 0..num_parts {
                let h_j_val = &round_2_result.lde_composition_poly_evaluations[j][row_idx];
                let h_j_ood = &h_ood[j];
                let numerator = h_j_val - h_j_ood;
                result = result + &composition_poly_gammas[j] * numerator * &inv_h[i];
            }

            // Trace terms: Σ_{j,k} γ'_{j,k} * (t_j(x_i) - t_j(z·w^k)) * inv_t_k[i]
            let num_total_cols = num_main_cols + num_aux_cols;
            for j in 0..num_total_cols {
                let gammas_j = &trace_terms_gammas[j];
                let ood_evals_j = &trace_ood_columns[j];

                for k in 0..num_eval_points {
                    let inv_t_k_i = &denoms[(1 + k) * domain_size + i];

                    let t_j_ood = &ood_evals_j[k];
                    let numerator: FieldElement<FieldExtension> = if j < num_main_cols {
                        lde_trace.get_main(row_idx, j) - t_j_ood
                    } else {
                        lde_trace.get_aux(row_idx, j - num_main_cols) - t_j_ood
                    };
                    result = result + &gammas_j[k] * numerator * inv_t_k_i;
                }
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
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
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
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
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
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let domain_size = domain.lde_roots_of_unity_coset.len();

        let index = challenge * 2;
        let index_sym = challenge * 2 + 1;
        PolynomialOpenings {
            proof: tree.get_proof_by_pos(index).unwrap(),
            proof_sym: tree.get_proof_by_pos(index_sym).unwrap(),
            evaluations: lde_trace.gather_aux_row(reverse_index(index, domain_size as u64)),
            evaluations_sym: lde_trace
                .gather_aux_row(reverse_index(index_sym, domain_size as u64)),
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

    // TODO: propagate errors instead of unwrap() in commit_columns, reconstruct_round1, and expand_pool_to_lde
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
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Result<MultiProof<Field, FieldExtension, PI>, ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        PI: Send + Sync + Clone,
    {
        info!("Started proof generation...");

        let num_airs = air_trace_pairs.len();

        // Check if any AIR uses LogUp (has auxiliary trace for running sums)
        let needs_logup_challenges = air_trace_pairs
            .iter()
            .any(|(air, _, _)| air.has_trace_interaction());

        // =====================================================================
        // Pre-pass: compute domains, twiddles, and max dimensions for pool allocation
        // =====================================================================

        let mut domains = Vec::with_capacity(num_airs);
        let mut twiddle_caches: Vec<LdeTwiddles<Field>> = Vec::with_capacity(num_airs);
        let mut max_main_cols = 0usize;
        let mut max_aux_cols = 0usize;
        let mut max_lde_size = 0usize;

        for (air, trace, _pub_inputs) in &*air_trace_pairs {
            let trace_length = trace.num_rows();
            let domain = new_domain(*air, trace_length);

            let lde_size = domain.interpolation_domain_size * domain.blowup_factor;
            let twiddles = LdeTwiddles::new(&domain);

            max_main_cols = max_main_cols.max(trace.num_main_columns);
            max_aux_cols = max_aux_cols.max(air.num_auxiliary_rap_columns());
            max_lde_size = max_lde_size.max(lde_size);

            domains.push(domain);
            twiddle_caches.push(twiddles);
        }

        // Allocate LDE column buffer pools — reused across all tables and phases.
        let mut main_pool: Vec<Vec<FieldElement<Field>>> = (0..max_main_cols)
            .map(|_| Vec::with_capacity(max_lde_size))
            .collect();
        let mut aux_pool: Vec<Vec<FieldElement<FieldExtension>>> = (0..max_aux_cols)
            .map(|_| Vec::with_capacity(max_lde_size))
            .collect();

        // =====================================================================
        // Round 1, Phase A: Commit all main traces (lightweight)
        // =====================================================================
        // All main trace commitments must be in the transcript before sampling
        // LogUp challenges. Pool buffers are reused across tables.

        let mut main_commits: Vec<MainCommitData<Field>> = Vec::with_capacity(num_airs);

        for ((air, trace, _pub_inputs), twiddles) in
            air_trace_pairs.iter().zip(twiddle_caches.iter())
        {
            let domain = &domains[main_commits.len()];

            let (tree, root, pre_tree, pre_root, n_pre) = if air.is_preprocessed() {
                Self::commit_preprocessed_trace_lightweight(
                    *trace,
                    domain,
                    transcript,
                    air.precomputed_commitment(),
                    air.num_precomputed_columns(),
                    twiddles,
                    &mut main_pool,
                )?
            } else {
                Self::commit_main_trace_lightweight(
                    *trace,
                    domain,
                    transcript,
                    twiddles,
                    &mut main_pool,
                )?
            };

            main_commits.push(MainCommitData {
                main_tree: Rc::new(tree),
                main_root: root,
                precomputed_tree: pre_tree.map(Rc::new),
                precomputed_root: pre_root,
                num_precomputed_cols: n_pre,
            });
        }

        // =====================================================================
        // Round 1, Phase B: Sample shared LogUp challenges
        // =====================================================================

        let logup_challenges: Vec<FieldElement<FieldExtension>> = if needs_logup_challenges {
            (0..LOGUP_NUM_CHALLENGES)
                .map(|_| transcript.sample_field_element())
                .collect()
        } else {
            Vec::new()
        };

        // =====================================================================
        // Round 1, Phase C: Build and commit auxiliary traces
        // =====================================================================
        // Split into two passes for parallelism:
        //   Pass 1 (parallel): Build all auxiliary traces (fingerprint + batch inversion)
        //   Pass 2 (sequential): Extract → LDE → commit → transcript append (shared pool)

        // Pass 1: Build aux traces in parallel.
        // Each build_auxiliary_trace has internal parallelism (batch_inverse, par_chunks),
        // but outer parallelism over 12 tables also helps on high-core-count machines.
        #[cfg(feature = "parallel")]
        let bus_inputs_vec: Vec<Option<BusPublicInputs<FieldExtension>>> = air_trace_pairs
            .par_iter_mut()
            .map(|(air, trace, _)| {
                if air.has_trace_interaction() {
                    air.build_auxiliary_trace(*trace, &logup_challenges)
                } else {
                    None
                }
            })
            .collect();
        #[cfg(not(feature = "parallel"))]
        let bus_inputs_vec: Vec<Option<BusPublicInputs<FieldExtension>>> = air_trace_pairs
            .iter_mut()
            .map(|(air, trace, _)| {
                if air.has_trace_interaction() {
                    air.build_auxiliary_trace(*trace, &logup_challenges)
                } else {
                    None
                }
            })
            .collect();

        // Pass 2: Sequential extract → LDE → commit → transcript append.
        // Uses shared aux_pool. Transcript ordering is preserved.
        let mut metadatas: Vec<Round1Metadata<Field, FieldExtension>> = Vec::with_capacity(num_airs);
        for ((((air, trace, _pub_inputs), bus_public_inputs), main_commit), (domain, twiddles))
            in air_trace_pairs
                .iter()
                .zip(bus_inputs_vec.into_iter())
                .zip(main_commits.into_iter())
                .zip(domains.iter().zip(twiddle_caches.iter()))
        {
            let (aux_tree, aux_root) = if air.has_trace_interaction() {
                let num_aux_cols = trace.num_aux_columns;
                trace.extract_columns_aux_into(&mut aux_pool);
                Self::expand_pool_to_lde_bitrev::<FieldExtension>(
                    &mut aux_pool,
                    num_aux_cols,
                    domain,
                    twiddles,
                );

                let (tree, root) =
                    Self::commit_columns(&aux_pool[..num_aux_cols])
                        .ok_or(ProvingError::EmptyCommitment)?;

                transcript.append_bytes(&root);
                (Some(Rc::new(tree)), Some(root))
            } else {
                (None, None)
            };

            metadatas.push(Round1Metadata {
                main_merkle_tree: Rc::clone(&main_commit.main_tree),
                main_merkle_root: main_commit.main_root,
                precomputed_merkle_tree: main_commit.precomputed_tree.as_ref().map(Rc::clone),
                precomputed_merkle_root: main_commit.precomputed_root,
                num_precomputed_cols: main_commit.num_precomputed_cols,
                aux_merkle_tree: aux_tree,
                aux_merkle_root: aux_root,
                rap_challenges: logup_challenges.clone(),
                bus_public_inputs,
            });
        }

        #[cfg(feature = "debug-checks")]
        {
            // For debug checks, we need to reconstruct Round1 temporarily.
            let mut temp_results: Vec<Round1<Field, FieldExtension>> = Vec::with_capacity(num_airs);
            for (((air, trace, _), metadata), (domain, twiddles)) in air_trace_pairs
                .iter()
                .zip(metadatas.iter())
                .zip(domains.iter().zip(twiddle_caches.iter()))
            {
                temp_results.push(
                    Self::reconstruct_round1(
                        *air, *trace, domain, metadata, twiddles,
                        &mut main_pool, &mut aux_pool,
                    ).unwrap()
                );
            }

            print_bus_balance_report(&temp_results);

            for ((((air, trace, pub_inputs), round_1_result), domain), metadata) in air_trace_pairs
                .iter()
                .zip(temp_results.iter())
                .zip(domains.iter())
                .zip(metadatas.iter())
            {
                validate_trace(
                    *air,
                    *pub_inputs,
                    *trace,
                    domain,
                    &round_1_result.rap_challenges,
                    metadata.bus_public_inputs.as_ref(),
                );
            }
        }

        // =====================================================================
        // Rounds 2-4: Sequential per-table proving
        // =====================================================================
        // For each table, recompute LDE into pool buffers, reuse stored Merkle trees,
        // run rounds 2-4, then drop table data. Pool buffers retain capacity.

        let mut proofs = Vec::with_capacity(num_airs);
        for ((((air, trace, pub_inputs), metadata), domain), twiddles) in air_trace_pairs
            .iter()
            .zip(metadatas.iter())
            .zip(domains.iter())
            .zip(twiddle_caches.iter())
        {
            // Recompute LDE evaluations into pool, reuse stored Merkle trees
            let round_1_result = Self::reconstruct_round1(
                *air, *trace, domain, metadata, twiddles,
                &mut main_pool, &mut aux_pool,
            )?;

            let proof =
                Self::prove_rounds_2_to_4(*air, *pub_inputs, &round_1_result, transcript, domain)?;
            proofs.push(proof);

            // Return column Vecs to pool (zero-copy move back). Pool slots that were
            // `take`n in reconstruct_round1 get their buffers back with capacity intact.
            let (main_cols, aux_cols) = round_1_result.lde_trace.into_columns();
            for (slot, col) in main_pool.iter_mut().zip(main_cols) {
                *slot = col;
            }
            for (slot, col) in aux_pool.iter_mut().zip(aux_cols) {
                *slot = col;
            }
        }

        Ok(MultiProof::new(proofs))
    }

    /// Generate a STARK proof for a single AIR/trace.
    /// This is equivalent to calling `multi_prove` with a single-element slice.
    fn prove(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        trace: &mut TraceTable<Field, FieldExtension>,
        pub_inputs: &PI,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Result<StarkProof<Field, FieldExtension, PI>, ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        PI: Send + Sync + Clone,
    {
        let air_trace_pairs = vec![(air, trace, pub_inputs)];
        Self::multi_prove(air_trace_pairs, transcript)
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
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        PI: Send + Sync + Clone,
    {
        info!("Started proof generation...");

        // ===================================
        // ==========|   Round 2   |==========
        // ===================================

        #[cfg(feature = "instruments")]
        println!("- Started round 2: Compute composition polynomial");
        #[cfg(feature = "instruments")]
        let timer2 = Instant::now();

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

        #[cfg(feature = "instruments")]
        let elapsed2 = timer2.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed2);

        // ===================================
        // ==========|   Round 3   |==========
        // ===================================

        #[cfg(feature = "instruments")]
        println!("- Started round 3: Evaluate polynomial in out of domain elements");
        #[cfg(feature = "instruments")]
        let timer3 = Instant::now();

        // <<<< Receive challenge: z
        let z = transcript.sample_z_ood(
            &domain.lde_roots_of_unity_coset,
            &domain.trace_roots_of_unity,
        );

        let round_3_result = Self::round_3_evaluate_polynomials_in_out_of_domain_element(
            air,
            domain,
            round_1_result,
            &round_2_result,
            &z,
        );

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

        #[cfg(feature = "instruments")]
        let elapsed3 = timer3.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed3);

        // ===================================
        // ==========|   Round 4   |==========
        // ===================================

        #[cfg(feature = "instruments")]
        println!("- Started round 4: FRI");
        #[cfg(feature = "instruments")]
        let timer4 = Instant::now();

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
        let elapsed4 = timer4.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed4);

        #[cfg(feature = "instruments")]
        {
            let total_time = elapsed2 + elapsed3 + elapsed4;
            println!(
                " Fraction of proving time per round: {:.4} {:.4} {:.4}",
                elapsed2.as_nanos() as f64 / total_time.as_nanos() as f64,
                elapsed3.as_nanos() as f64 / total_time.as_nanos() as f64,
                elapsed4.as_nanos() as f64 / total_time.as_nanos() as f64
            );
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
            // Final FRI polynomial coefficients
            fri_final_poly: round_4_result.fri_final_poly,
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
fn print_bus_balance_report<Field, FieldExtension>(
    round_1_results: &[Round1<Field, FieldExtension>],
) where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField,
    FieldExtension: IsField,
    FieldElement<Field>: AsBytes,
    FieldElement<FieldExtension>: AsBytes,
{
    use std::collections::HashMap;

    let has_logup = round_1_results
        .iter()
        .any(|r| r.bus_public_inputs.is_some());
    if !has_logup {
        return;
    }

    let mut global_bus_sums: HashMap<u64, FieldElement<FieldExtension>> = HashMap::new();
    let mut bus_senders: HashMap<u64, Vec<(String, FieldElement<FieldExtension>)>> = HashMap::new();
    let mut bus_receivers: HashMap<u64, Vec<(String, FieldElement<FieldExtension>)>> =
        HashMap::new();
    let mut global_sender_sums: HashMap<u64, FieldElement<FieldExtension>> = HashMap::new();
    let mut global_receiver_sums: HashMap<u64, FieldElement<FieldExtension>> = HashMap::new();

    for round_1_result in round_1_results {
        if let Some(bus_inputs) = &round_1_result.bus_public_inputs {
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

#[cfg(test)]
mod tests {
    use crate::domain::VerifierDomain;
    use std::num::ParseIntError;

    fn decode_hex(s: &str) -> Result<Vec<u8>, ParseIntError> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
            .collect()
    }

    use crate::{
        Felt252,
        examples::{
            fibonacci_2_cols_shifted::{self, Fibonacci2ColsShifted},
            simple_fibonacci::{self},
        },
        proof::options::ProofOptions,
        transcript::StoneProverTranscript,
        verifier::{Challenges, IsStarkVerifier, Verifier},
    };

    use super::*;
    use math::{
        field::{
            element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
            traits::IsFFTField,
        },
        polynomial::Polynomial,
    };

    #[test]
    fn test_domain_constructor() {
        let trace = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 8);
        let trace_length = trace.num_rows();
        let coset_offset = 3;
        let blowup_factor: usize = 2;
        let grinding_factor = 20;

        let proof_options = ProofOptions {
            blowup_factor: blowup_factor as u8,
            fri_number_of_queries: 1,
            coset_offset,
            grinding_factor,
            fri_log_arity: 1,
            fri_log_final_poly_len: 0,
        };

        let domain = Domain::new(
            &simple_fibonacci::FibonacciAIR::new(&proof_options),
            trace_length,
        );
        assert_eq!(domain.blowup_factor, 2);
        assert_eq!(domain.interpolation_domain_size, trace_length);
        assert_eq!(domain.root_order, trace_length.trailing_zeros());
        assert_eq!(domain.coset_offset, FieldElement::from(coset_offset));

        let primitive_root = Stark252PrimeField::get_primitive_root_of_unity(
            (trace_length * blowup_factor).trailing_zeros() as u64,
        )
        .unwrap();

        assert_eq!(
            domain.trace_primitive_root,
            primitive_root.pow(blowup_factor)
        );
        for i in 0..(trace_length * blowup_factor) {
            assert_eq!(
                domain.lde_roots_of_unity_coset[i],
                primitive_root.pow(i) * FieldElement::from(coset_offset)
            );
        }
    }

    #[test]
    fn test_evaluate_polynomial_on_lde_domain_on_trace_polys() {
        let trace = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 8);

        let trace_length = trace.num_rows();

        let trace_polys = trace.compute_trace_polys_main::<Stark252PrimeField>();
        let coset_offset = Felt252::from(3);
        let blowup_factor: usize = 2;
        let domain_size = 8;

        let primitive_root = Stark252PrimeField::get_primitive_root_of_unity(
            (trace_length * blowup_factor).trailing_zeros() as u64,
        )
        .unwrap();

        for poly in trace_polys.iter() {
            let lde_evaluation =
                evaluate_polynomial_on_lde_domain(poly, blowup_factor, domain_size, &coset_offset)
                    .unwrap();
            assert_eq!(lde_evaluation.len(), trace_length * blowup_factor);
            for (i, evaluation) in lde_evaluation.iter().enumerate() {
                assert_eq!(
                    *evaluation,
                    poly.evaluate(&(coset_offset * primitive_root.pow(i)))
                );
            }
        }
    }

    #[test]
    fn test_evaluate_polynomial_on_lde_domain_edge_case() {
        let poly = Polynomial::new_monomial(Felt252::one(), 8);
        let blowup_factor: usize = 4;
        let domain_size: usize = 8;
        let offset = Felt252::from(3);
        let evaluations =
            evaluate_polynomial_on_lde_domain(&poly, blowup_factor, domain_size, &offset).unwrap();
        assert_eq!(evaluations.len(), domain_size * blowup_factor);

        let primitive_root: Felt252 = Stark252PrimeField::get_primitive_root_of_unity(
            (domain_size * blowup_factor).trailing_zeros() as u64,
        )
        .unwrap();
        for (i, eval) in evaluations.iter().enumerate() {
            assert_eq!(*eval, poly.evaluate(&(offset * primitive_root.pow(i))));
        }
    }

    #[allow(clippy::type_complexity)]
    fn proof_parts_stone_compatibility_case_1() -> (
        StarkProof<
            Stark252PrimeField,
            Stark252PrimeField,
            fibonacci_2_cols_shifted::PublicInputs<Stark252PrimeField>,
        >,
        Fibonacci2ColsShifted<Stark252PrimeField>,
        ProofOptions,
        [u8; 4],
        usize,
    ) {
        let mut trace = fibonacci_2_cols_shifted::compute_trace(FieldElement::one(), 4);

        let claimed_index = 3;
        let col = 0;
        let claimed_value = *trace.get_main(claimed_index, col);
        let mut proof_options = ProofOptions::default_test_options();
        proof_options.blowup_factor = 4;
        proof_options.coset_offset = 3;
        proof_options.grinding_factor = 0;
        proof_options.fri_number_of_queries = 1;

        let pub_inputs = fibonacci_2_cols_shifted::PublicInputs {
            claimed_value,
            claimed_index,
        };

        let transcript_init_seed = [0xca, 0xfe, 0xca, 0xfe];

        let air = Fibonacci2ColsShifted::<Stark252PrimeField>::new(&proof_options);
        let trace_length = trace.num_rows();

        let proof = Prover::prove(
            &air,
            &mut trace,
            &pub_inputs,
            &mut StoneProverTranscript::new(&transcript_init_seed),
        )
        .unwrap();
        (
            proof,
            air,
            proof_options,
            transcript_init_seed,
            trace_length,
        )
    }

    fn stone_compatibility_case_1_proof() -> StarkProof<
        Stark252PrimeField,
        Stark252PrimeField,
        fibonacci_2_cols_shifted::PublicInputs<Stark252PrimeField>,
    > {
        let (proof, _, _, _, _) = proof_parts_stone_compatibility_case_1();
        proof
    }

    fn stone_compatibility_case_1_challenges() -> Challenges<Stark252PrimeField> {
        let (proof, air, _, seed, trace_length) = proof_parts_stone_compatibility_case_1();

        let domain = VerifierDomain::new(&air, trace_length);
        Verifier::step_1_replay_rounds_and_recover_challenges(
            &air,
            &proof,
            &domain,
            &mut StoneProverTranscript::new(&seed),
        )
    }

    #[test]
    fn stone_compatibility_case_1_proof_is_valid() {
        let (proof, air, _options, seed, _) = proof_parts_stone_compatibility_case_1();
        assert!(Verifier::verify(
            &proof,
            &air,
            &mut StoneProverTranscript::new(&seed)
        ));
    }

    #[test]
    fn stone_compatibility_case_1_trace_commitment() {
        let proof = stone_compatibility_case_1_proof();

        assert_eq!(
            proof.lde_trace_main_merkle_root.to_vec(),
            decode_hex("0eb9dcc0fb1854572a01236753ce05139d392aa3aeafe72abff150fe21175594").unwrap()
        );
    }

    #[test]
    fn stone_compatibility_case_1_composition_poly_challenges() {
        let challenges = stone_compatibility_case_1_challenges();

        assert_eq!(challenges.transition_coeffs[0], FieldElement::one());
        let beta = challenges.transition_coeffs[1];
        assert_eq!(
            beta,
            FieldElement::from_hex_unchecked(
                "86105fff7b04ed4068ecccb8dbf1ed223bd45cd26c3532d6c80a818dbd4fa7"
            ),
        );

        assert_eq!(challenges.boundary_coeffs[0], beta.pow(2u64));
        assert_eq!(challenges.boundary_coeffs[1], beta.pow(3u64));
    }

    #[test]
    fn stone_compatibility_case_1_composition_poly_commitment() {
        let proof = stone_compatibility_case_1_proof();
        // Composition polynomial commitment
        assert_eq!(
            proof.composition_poly_root.to_vec(),
            decode_hex("7cdd8d5fe3bd62254a417e2e260e0fed4fccdb6c9005e828446f645879394f38").unwrap()
        );
    }

    #[test]
    fn stone_compatibility_case_1_out_of_domain_challenge() {
        let challenges = stone_compatibility_case_1_challenges();
        assert_eq!(
            challenges.z,
            FieldElement::from_hex_unchecked(
                "317629e783794b52cd27ac3a5e418c057fec9dd42f2b537cdb3f24c95b3e550"
            )
        );
    }

    #[test]
    fn stone_compatibility_case_1_out_of_domain_trace_evaluation() {
        let proof = stone_compatibility_case_1_proof();

        assert_eq!(
            proof.trace_ood_evaluations.get_row(0)[0],
            FieldElement::from_hex_unchecked(
                "70d8181785336cc7e0a0a1078a79ee6541ca0803ed3ff716de5a13c41684037",
            )
        );
        assert_eq!(
            proof.trace_ood_evaluations.get_row(1)[0],
            FieldElement::from_hex_unchecked(
                "29808fc8b7480a69295e4b61600480ae574ca55f8d118100940501b789c1630",
            )
        );
        assert_eq!(
            proof.trace_ood_evaluations.get_row(0)[1],
            FieldElement::from_hex_unchecked(
                "7d8110f21d1543324cc5e472ab82037eaad785707f8cae3d64c5b9034f0abd2",
            )
        );
        assert_eq!(
            proof.trace_ood_evaluations.get_row(1)[1],
            FieldElement::from_hex_unchecked(
                "1b58470130218c122f71399bf1e04cf75a6e8556c4751629d5ce8c02cc4e62d",
            )
        );
    }

    #[test]
    fn stone_compatibility_case_1_out_of_domain_composition_poly_evaluation() {
        let proof = stone_compatibility_case_1_proof();

        assert_eq!(
            proof.composition_poly_parts_ood_evaluation[0],
            FieldElement::from_hex_unchecked(
                "1c0b7c2275e36d62dfb48c791be122169dcc00c616c63f8efb2c2a504687e85",
            )
        );
    }

    #[test]
    fn stone_compatibility_case_1_deep_composition_poly_challenges() {
        let challenges = stone_compatibility_case_1_challenges();

        // Trace terms coefficients
        assert_eq!(challenges.trace_term_coeffs[0][0], FieldElement::one());
        let gamma = challenges.trace_term_coeffs[0][1];
        assert_eq!(
            &gamma,
            &FieldElement::from_hex_unchecked(
                "a0c79c1c77ded19520873d9c2440451974d23302e451d13e8124cf82fc15dd"
            )
        );
        assert_eq!(&challenges.trace_term_coeffs[1][0], &gamma.pow(2_u64));
        assert_eq!(&challenges.trace_term_coeffs[1][1], &gamma.pow(3_u64));

        // Composition polynomial parts terms coefficient
        assert_eq!(&challenges.gammas[0], &gamma.pow(4_u64));
    }

    #[test]
    fn stone_compatibility_case_1_fri_commit_phase_challenge_0() {
        let challenges = stone_compatibility_case_1_challenges();

        // Challenge to fold FRI polynomial
        assert_eq!(
            challenges.zetas[0],
            FieldElement::from_hex_unchecked(
                "5c6b5a66c9fda19f583f0b10edbaade98d0e458288e62c2fa40e3da2b293cef"
            )
        );
    }

    #[test]
    fn stone_compatibility_case_1_fri_commit_phase_layer_1_commitment() {
        let proof = stone_compatibility_case_1_proof();

        // Commitment of first layer of FRI
        assert_eq!(
            proof.fri_layers_merkle_roots[0].to_vec(),
            decode_hex("327d47da86f5961ee012b2b0e412de16023ffba97c82bfe85102f00daabd49fb").unwrap()
        );
    }

    #[test]
    fn stone_compatibility_case_1_fri_commit_phase_challenge_1() {
        let challenges = stone_compatibility_case_1_challenges();
        assert_eq!(
            challenges.zetas[1],
            FieldElement::from_hex_unchecked(
                "13c337c9dc727bea9eef1f82cab86739f17acdcef562f9e5151708f12891295"
            )
        );
    }

    #[test]
    fn stone_compatibility_case_1_fri_commit_phase_last_value() {
        let proof = stone_compatibility_case_1_proof();

        assert_eq!(
            proof.fri_final_poly[0],
            FieldElement::from_hex_unchecked(
                "43fedf9f9e3d1469309862065c7d7ca0e7e9ce451906e9c01553056f695aec9"
            )
        );
    }

    #[test]
    fn stone_compatibility_case_1_fri_query_iota_challenge() {
        let challenges = stone_compatibility_case_1_challenges();
        assert_eq!(challenges.iotas[0], 1);
    }

    #[test]
    fn stone_compatibility_case_1_fri_query_phase_trace_openings() {
        let proof = stone_compatibility_case_1_proof();

        // Trace Col 0
        assert_eq!(
            proof.deep_poly_openings[0].main_trace_polys.evaluations[0],
            FieldElement::from_hex_unchecked(
                "4de0d56f9cf97dff326c26592fbd4ae9ee756080b12c51cfe4864e9b8734f43"
            )
        );

        // Trace Col 1
        assert_eq!(
            proof.deep_poly_openings[0].main_trace_polys.evaluations[1],
            FieldElement::from_hex_unchecked(
                "1bc1aadf39f2faee64d84cb25f7a95d3dceac1016258a39fc90c9d370e69e8e"
            )
        );

        // Trace Col 0 symmetric
        assert_eq!(
            proof.deep_poly_openings[0].main_trace_polys.evaluations_sym[0],
            FieldElement::from_hex_unchecked(
                "321f2a9063068310cd93d9a6d042b516118a9f7f4ed3ae301b79b16478cb0c6"
            )
        );

        // Trace Col 1 symmetric
        assert_eq!(
            proof.deep_poly_openings[0].main_trace_polys.evaluations_sym[1],
            FieldElement::from_hex_unchecked(
                "643e5520c60d06219b27b34da0856a2c23153efe9da75c6036f362c8f196186"
            )
        );
    }

    #[test]
    fn stone_compatibility_case_1_fri_query_phase_trace_terms_authentication_path() {
        let proof = stone_compatibility_case_1_proof();

        // Trace poly auth path level 1
        assert_eq!(
            proof.deep_poly_openings[0]
                .main_trace_polys
                .proof
                .merkle_path[1]
                .to_vec(),
            decode_hex("91b0c0b24b9d00067b0efab50832b76cf97192091624d42b86740666c5d369e6").unwrap()
        );

        // Trace poly auth path level 2
        assert_eq!(
            proof.deep_poly_openings[0]
                .main_trace_polys
                .proof
                .merkle_path[2]
                .to_vec(),
            decode_hex("993b044db22444c0c0ebf1095b9a51faeb001c9b4dea36abe905f7162620dbbd").unwrap()
        );

        // Trace poly auth path level 3
        assert_eq!(
            proof.deep_poly_openings[0]
                .main_trace_polys
                .proof
                .merkle_path[3]
                .to_vec(),
            decode_hex("5017abeca33fa82576b5c5c2c61792693b48c9d4414a407eef66b6029dae07ea").unwrap()
        );
    }

    #[test]
    fn stone_compatibility_case_1_fri_query_phase_composition_poly_openings() {
        let proof = stone_compatibility_case_1_proof();

        // Composition poly
        assert_eq!(
            proof.deep_poly_openings[0].composition_poly.evaluations[0],
            FieldElement::from_hex_unchecked(
                "2b54852557db698e97253e9d110d60e9bf09f1d358b4c1a96f9f3cf9d2e8755"
            )
        );
        // Composition poly sym
        assert_eq!(
            proof.deep_poly_openings[0].composition_poly.evaluations_sym[0],
            FieldElement::from_hex_unchecked(
                "190f1b0acb7858bd3f5285b68befcf32b436a5f1e3a280e1f42565c1f35c2c3"
            )
        );
    }

    #[test]
    fn stone_compatibility_case_1_fri_query_phase_composition_poly_authentication_path() {
        let proof = stone_compatibility_case_1_proof();

        // Composition poly auth path level 0
        assert_eq!(
            proof.deep_poly_openings[0]
                .composition_poly
                .proof
                .merkle_path[0]
                .to_vec(),
            decode_hex("403b75a122eaf90a298e5d3db2cc7ca096db478078122379a6e3616e72da7546").unwrap()
        );

        // Composition poly auth path level 1
        assert_eq!(
            proof.deep_poly_openings[0]
                .composition_poly
                .proof
                .merkle_path[1]
                .to_vec(),
            decode_hex("07950888c0355c204a1e83ecbee77a0a6a89f93d41cc2be6b39ddd1e727cc965").unwrap()
        );

        // Composition poly auth path level 2
        assert_eq!(
            proof.deep_poly_openings[0]
                .composition_poly
                .proof
                .merkle_path[2]
                .to_vec(),
            decode_hex("58befe2c5de74cc5a002aa82ea219c5b242e761b45fd266eb95521e9f53f44eb").unwrap()
        );
    }

    #[test]
    fn stone_compatibility_case_1_fri_query_phase_query_lengths() {
        let proof = stone_compatibility_case_1_proof();

        assert_eq!(proof.query_list.len(), 1);

        assert_eq!(proof.query_list[0].layers_evaluations_sym.len(), 1);

        assert_eq!(
            proof.query_list[0].layers_auth_paths[0].merkle_path.len(),
            2
        );
    }

    #[test]
    fn stone_compatibility_case_1_fri_query_phase_layer_1_evaluation_symmetric() {
        let proof = stone_compatibility_case_1_proof();

        assert_eq!(
            proof.query_list[0].layers_evaluations_sym[0][0],
            FieldElement::from_hex_unchecked(
                "0684991e76e5c08db17f33ea7840596be876d92c143f863e77cad10548289fd0"
            )
        );
    }

    #[test]
    fn stone_compatibility_case_1_fri_query_phase_layer_1_authentication_path() {
        let proof = stone_compatibility_case_1_proof();

        // FRI layer 1 auth path level 0
        assert_eq!(
            proof.query_list[0].layers_auth_paths[0].merkle_path[0].to_vec(),
            decode_hex("0683622478e9e93cc2d18754872f043619f030b494d7ec8e003b1cbafe83b67b").unwrap()
        );

        // FRI layer 1 auth path level 1
        assert_eq!(
            proof.query_list[0].layers_auth_paths[0].merkle_path[1].to_vec(),
            decode_hex("7985d945abe659a7502698051ec739508ed6bab594984c7f25e095a0a57a2e55").unwrap()
        );
    }

    fn proof_parts_stone_compatibility_case_2() -> (
        StarkProof<
            Stark252PrimeField,
            Stark252PrimeField,
            fibonacci_2_cols_shifted::PublicInputs<Stark252PrimeField>,
        >,
        ProofOptions,
        [u8; 4],
    ) {
        let mut trace = fibonacci_2_cols_shifted::compute_trace(FieldElement::from(12345), 512);

        let claimed_index = 420;
        let col = 0;
        let claimed_value = *trace.get_main(claimed_index, col);
        let mut proof_options = ProofOptions::default_test_options();
        proof_options.blowup_factor = 1 << 6;
        proof_options.coset_offset = 3;
        proof_options.grinding_factor = 0;
        proof_options.fri_number_of_queries = 1;

        let pub_inputs = fibonacci_2_cols_shifted::PublicInputs {
            claimed_value,
            claimed_index,
        };

        let transcript_init_seed = [0xfa, 0xfa, 0xfa, 0xee];

        let air = Fibonacci2ColsShifted::<Stark252PrimeField>::new(&proof_options);

        let proof = Prover::prove(
            &air,
            &mut trace,
            &pub_inputs,
            &mut StoneProverTranscript::new(&transcript_init_seed),
        )
        .unwrap();
        (proof, proof_options, transcript_init_seed)
    }

    fn stone_compatibility_case_2_proof() -> StarkProof<
        Stark252PrimeField,
        Stark252PrimeField,
        fibonacci_2_cols_shifted::PublicInputs<Stark252PrimeField>,
    > {
        let (proof, _, _) = proof_parts_stone_compatibility_case_2();
        proof
    }

    fn stone_compatibility_case_2_challenges() -> Challenges<Stark252PrimeField> {
        let (proof, options, seed) = proof_parts_stone_compatibility_case_2();

        let air = Fibonacci2ColsShifted::new(&options);
        let domain = VerifierDomain::new(&air, proof.trace_length);
        Verifier::step_1_replay_rounds_and_recover_challenges(
            &air,
            &proof,
            &domain,
            &mut StoneProverTranscript::new(&seed),
        )
    }

    #[test]
    fn stone_compatibility_case_2_trace_commitment() {
        let proof = stone_compatibility_case_2_proof();

        assert_eq!(
            proof.lde_trace_main_merkle_root.to_vec(),
            decode_hex("6d31dd00038974bde5fe0c5e3a765f8ddc822a5df3254fca85a1950ae0208cbe").unwrap()
        );
    }

    #[test]
    fn stone_compatibility_case_2_fri_query_iota_challenge() {
        let challenges = stone_compatibility_case_2_challenges();
        assert_eq!(challenges.iotas[0], 4239);
    }

    #[test]
    fn stone_compatibility_case_2_fri_query_phase_layer_7_evaluation_symmetric() {
        let proof = stone_compatibility_case_2_proof();

        assert_eq!(
            proof.query_list[0].layers_evaluations_sym[7][0],
            FieldElement::from_hex_unchecked(
                "7aa40c5a4e30b44fee5bcc47c54072a435aa35c1a31b805cad8126118cc6860"
            )
        );
    }

    #[test]
    fn stone_compatibility_case_2_fri_query_phase_layer_8_authentication_path() {
        let proof = stone_compatibility_case_2_proof();

        // FRI layer 7 auth path level 5
        assert_eq!(
            proof.query_list[0].layers_auth_paths[7].merkle_path[5].to_vec(),
            decode_hex("f12f159b548ca2c571a270870d43e7ec2ead78b3e93b635738c31eb9bcda3dda").unwrap()
        );
    }

    /// Tests that `get_trace_evaluations_from_lde` (barycentric) produces identical
    /// results to `get_trace_evaluations` (Horner) for the Fibonacci trace.
    #[test]
    fn barycentric_trace_eval_matches_horner_trace_eval() {
        use crate::trace::{get_trace_evaluations, get_trace_evaluations_from_lde, LDETraceTable};

        let trace = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 8);
        let trace_length = trace.num_rows();
        let blowup_factor: usize = 4;
        let coset_offset = 3u64;

        let proof_options = ProofOptions {
            blowup_factor: blowup_factor as u8,
            fri_number_of_queries: 1,
            coset_offset,
            grinding_factor: 0,
            fri_log_arity: 1,
            fri_log_final_poly_len: 0,
        };

        let air = simple_fibonacci::FibonacciAIR::<Stark252PrimeField>::new(&proof_options);
        let domain = Domain::new(&air, trace_length);

        // Compute trace polys (Horner path)
        let trace_polys = trace.compute_trace_polys_main::<Stark252PrimeField>();

        // Compute LDE evaluations for each column
        let lde_evaluations: Vec<Vec<Felt252>> = trace_polys
            .iter()
            .map(|poly| {
                evaluate_polynomial_on_lde_domain(
                    poly,
                    domain.blowup_factor,
                    domain.interpolation_domain_size,
                    &domain.coset_offset,
                )
                .unwrap()
            })
            .collect();

        // Build LDE trace table
        let lde_trace = LDETraceTable::from_columns(
            lde_evaluations,
            Vec::<Vec<Felt252>>::new(),
            air.step_size(),
            domain.blowup_factor,
        );

        // Pick OOD point (just a deterministic value)
        let z = Felt252::from(12345u64);

        let frame_offsets = air.context().transition_offsets.clone();
        let step_size = air.step_size();

        // Horner-based evaluation (ground truth)
        let expected = get_trace_evaluations::<Stark252PrimeField, Stark252PrimeField>(
            &trace_polys,
            &[],
            &z,
            &frame_offsets,
            &domain.trace_primitive_root,
            step_size,
        );

        // Barycentric evaluation (new path)
        let result = get_trace_evaluations_from_lde(
            &lde_trace,
            &domain,
            &z,
            &frame_offsets,
            step_size,
        );

        assert_eq!(result.width, expected.width);
        assert_eq!(result.height, expected.height);
        assert_eq!(result.data, expected.data);
    }

    /// Test that direct quotient decomposition produces identical results to
    /// the original iFFT + break_in_parts + FFT pipeline.
    #[test]
    fn test_decompose_and_extend_d2_matches_original() {
        // Build a known polynomial H of degree < 2N, evaluate on LDE coset.
        let n = 16usize; // trace length
        let blowup_factor = 2usize;
        let two_n = n * blowup_factor;

        let proof_options = ProofOptions {
            blowup_factor: blowup_factor as u8,
            fri_number_of_queries: 1,
            coset_offset: 3,
            grinding_factor: 0,
            fri_log_arity: 1,
            fri_log_final_poly_len: 0,
        };

        // We need an AIR with composition_poly_degree_bound = 2 * trace_length.
        // Use QuadraticAIR for this.
        use crate::examples::quadratic_air::QuadraticAIR;
        let air = QuadraticAIR::<Stark252PrimeField>::new(&proof_options);
        let domain = Domain::new(&air, n);

        // Create a random-ish polynomial H(x) of degree < 2N (= 32 coefficients).
        let coeffs: Vec<Felt252> = (0..two_n)
            .map(|i| FieldElement::from((i * 37 + 13) as u64))
            .collect();
        let h_poly = Polynomial::new(&coeffs);

        // Evaluate H on the LDE coset (2N points: g·ω^i for i=0..2N-1)
        let constraint_evaluations: Vec<Felt252> = domain
            .lde_roots_of_unity_coset
            .iter()
            .map(|x| h_poly.evaluate(x))
            .collect();
        assert_eq!(constraint_evaluations.len(), two_n);

        // --- Original path: iFFT(2N) + break_in_parts(2) + FFT(2N) each ---
        let composition_poly =
            Polynomial::interpolate_offset_fft(&constraint_evaluations, &domain.coset_offset)
                .unwrap();
        let parts = composition_poly.break_in_parts(2);
        let original: Vec<Vec<Felt252>> = parts
            .iter()
            .map(|part| {
                evaluate_polynomial_on_lde_domain(part, blowup_factor, n, &domain.coset_offset)
                    .unwrap()
            })
            .collect();

        // --- New path: algebraic decomposition ---
        let new_result =
            Prover::<Stark252PrimeField, Stark252PrimeField, ()>::decompose_and_extend_d2::<
                Stark252PrimeField,
            >(&constraint_evaluations, &domain);

        assert_eq!(new_result.len(), 2);
        assert_eq!(new_result[0].len(), original[0].len());
        assert_eq!(new_result[1].len(), original[1].len());
        for i in 0..new_result[0].len() {
            assert_eq!(
                new_result[0][i], original[0][i],
                "H₀ mismatch at index {i}"
            );
            assert_eq!(
                new_result[1][i], original[1][i],
                "H₁ mismatch at index {i}"
            );
        }
    }
}
