use std::marker::PhantomData;
#[cfg(feature = "instruments")]
use std::time::Instant;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::fft::cpu::bit_reversing::{in_place_bit_reverse_permute, reverse_index};
use math::fft::errors::FFTError;

use log::info;
use math::field::traits::{IsField, IsSubFieldOf};
use math::traits::AsBytes;
use math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};

#[cfg(feature = "parallel")]
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

#[cfg(feature = "debug-checks")]
use crate::debug::validate_trace;
use crate::domain::new_domain;
use crate::fri;
use crate::lookup::LOGUP_NUM_CHALLENGES;
use crate::proof::stark::{DeepPolynomialOpenings, PolynomialOpenings};
use crate::table::Table;
use crate::trace::{LDETraceTable, columns2rows};

use super::config::{BatchedMerkleTree, Commitment};
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

type MainCommitment<Field> = (Round1CommitmentData<Field>, Vec<Vec<FieldElement<Field>>>);

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
    /// The result of the interpolation of the columns of the trace table.
    pub(crate) trace_polys: Vec<Polynomial<FieldElement<F>>>,
    /// The Merkle trees constructed to obtain the commitment of the entire trace table.
    /// For preprocessed tables, this contains only the multiplicity columns.
    pub(crate) lde_trace_merkle_tree: BatchedMerkleTree<F>,
    /// The root of the Merkle tree in `lde_trace_merkle_tree`.
    pub(crate) lde_trace_merkle_root: Commitment,
    /// For preprocessed tables: Merkle tree over precomputed columns only.
    pub(crate) precomputed_merkle_tree: Option<BatchedMerkleTree<F>>,
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

impl<Field, FieldExtension> Round1<Field, FieldExtension>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField,
    FieldExtension: IsField,
    FieldElement<Field>: AsBytes,
    FieldElement<FieldExtension>: AsBytes,
{
    /// Returns the full list of the polynomials interpolating the trace. It includes both
    /// main and auxiliary trace polynomials. The main trace polynomials are casted to
    /// polynomials with coefficients over `Self::FieldExtension`.
    fn all_trace_polys(&self) -> Vec<Polynomial<FieldElement<FieldExtension>>> {
        let mut trace_polys: Vec<_> = self
            .main
            .trace_polys
            .clone()
            .into_iter()
            .map(|poly| poly.to_extension())
            .collect();

        if let Some(aux) = &self.aux {
            trace_polys.extend_from_slice(&aux.trace_polys.to_owned())
        }
        trace_polys
    }
}

/// A container for the results of the second round of the STARK Prove protocol.
pub struct Round2<F>
where
    F: IsField,
    FieldElement<F>: AsBytes,
{
    /// The list of polynomials `H₀, ..., Hₙ` such that `H = ∑ᵢXⁱH(Xⁿ)`, where H is the composition polynomial.
    pub(crate) composition_poly_parts: Vec<Polynomial<FieldElement<F>>>,
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
    E: IsField,
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

    /// Compute the LDE commitment for a subset of columns from a trace (for testing).
    ///
    /// This helper computes the same commitment the prover generates internally,
    /// useful for setting up soundness test scenarios.
    ///
    /// The commitment is computed by:
    /// 1. Interpolating columns to polynomials
    /// 2. Evaluating on LDE domain (size = trace_size * blowup_factor)
    /// 3. Bit-reverse permuting
    /// 4. Building Merkle tree from rows
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

        // Bit-reverse permute
        let mut lde_permuted = evaluations;
        for col in lde_permuted.iter_mut() {
            in_place_bit_reverse_permute(col);
        }

        // Build commitment
        let rows = columns2rows(lde_permuted);
        let (_, commitment) = Self::batch_commit_main(&rows)?;

        Some(commitment)
    }

    /// Given a `TraceTable`, this method interpolates its columns, computes the commitment to the
    /// table and appends it to the transcript.
    /// Output: a touple of length 4 with the following:
    /// • The polynomials interpolating the columns of `trace`.
    /// • The evaluations of the above polynomials over the domain `domain`.
    /// • The Merkle tree of evaluations of the above polynomials over the domain `domain`.
    /// • The roots of the above Merkle trees.
    #[allow(clippy::type_complexity)]
    fn interpolate_and_commit_main(
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Option<(
        Vec<Polynomial<FieldElement<Field>>>,
        Vec<Vec<FieldElement<Field>>>,
        BatchedMerkleTree<Field>,
        Commitment,
    )>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        Field: IsSubFieldOf<FieldExtension>,
    {
        // Interpolate columns of `trace`.
        let trace_polys = trace.compute_trace_polys_main::<Field>();

        // Evaluate those polynomials t_j on the large domain D_LDE.
        let lde_trace_evaluations =
            Self::compute_lde_trace_evaluations::<Field>(&trace_polys, domain);

        let mut lde_trace_permuted = lde_trace_evaluations.clone();
        for col in lde_trace_permuted.iter_mut() {
            in_place_bit_reverse_permute(col);
        }

        // Compute commitment.
        let lde_trace_permuted_rows = columns2rows(lde_trace_permuted);

        let (lde_trace_merkle_tree, lde_trace_merkle_root) =
            Self::batch_commit_main(&lde_trace_permuted_rows)?;

        // >>>> Send commitment.
        transcript.append_bytes(&lde_trace_merkle_root);

        Some((
            trace_polys,
            lde_trace_evaluations,
            lde_trace_merkle_tree,
            lde_trace_merkle_root,
        ))
    }

    /// Variant of `interpolate_and_commit_main` for preprocessed tables.
    ///
    /// Does NOT append to transcript - the caller handles that with the hardcoded commitment.
    /// Returns the computed commitment for verification purposes.
    #[allow(clippy::type_complexity)]
    fn interpolate_and_commit_preprocessed(
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
    ) -> Option<(
        Vec<Polynomial<FieldElement<Field>>>,
        Vec<Vec<FieldElement<Field>>>,
        BatchedMerkleTree<Field>,
        Commitment,
    )>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        Field: IsSubFieldOf<FieldExtension>,
    {
        // Interpolate columns of `trace`.
        let trace_polys = trace.compute_trace_polys_main::<Field>();

        // Evaluate those polynomials t_j on the large domain D_LDE.
        let lde_trace_evaluations =
            Self::compute_lde_trace_evaluations::<Field>(&trace_polys, domain);

        let mut lde_trace_permuted = lde_trace_evaluations.clone();
        for col in lde_trace_permuted.iter_mut() {
            in_place_bit_reverse_permute(col);
        }

        // Compute commitment (but don't append to transcript - caller does that).
        let lde_trace_permuted_rows = columns2rows(lde_trace_permuted);

        let (lde_trace_merkle_tree, lde_trace_merkle_root) =
            Self::batch_commit_main(&lde_trace_permuted_rows)?;

        Some((
            trace_polys,
            lde_trace_evaluations,
            lde_trace_merkle_tree,
            lde_trace_merkle_root,
        ))
    }

    /// Given a `TraceTable`, this method interpolates its columns, computes the commitment to the
    /// table and appends it to the transcript.
    /// Output: a touple of length 4 with the following:
    /// • The polynomials interpolating the columns of `trace`.
    /// • The evaluations of the above polynomials over the domain `domain`.
    /// • The Merkle tree of evaluations of the above polynomials over the domain `domain`.
    /// • The roots of the above Merkle trees.
    #[allow(clippy::type_complexity)]
    fn interpolate_and_commit_aux(
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Option<(
        Vec<Polynomial<FieldElement<FieldExtension>>>,
        Vec<Vec<FieldElement<FieldExtension>>>,
        BatchedMerkleTree<FieldExtension>,
        Commitment,
    )>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        Field: IsSubFieldOf<FieldExtension> + IsFFTField,
    {
        // Interpolate columns of `trace`.
        let trace_polys = trace.compute_trace_polys_aux::<Field>();

        // Evaluate those polynomials t_j on the large domain D_LDE.
        let lde_trace_evaluations = Self::compute_lde_trace_evaluations(&trace_polys, domain);

        let mut lde_trace_permuted = lde_trace_evaluations.clone();
        for col in lde_trace_permuted.iter_mut() {
            in_place_bit_reverse_permute(col);
        }

        // Compute commitment.
        let lde_trace_permuted_rows = columns2rows(lde_trace_permuted);

        let (lde_trace_merkle_tree, lde_trace_merkle_root) =
            Self::batch_commit_extension(&lde_trace_permuted_rows)?;

        // >>>> Send commitment.
        transcript.append_bytes(&lde_trace_merkle_root);

        Some((
            trace_polys,
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
        E: IsSubFieldOf<FieldExtension>,
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
        let Some((trace_polys, evaluations, main_merkle_tree, main_merkle_root)) =
            Self::interpolate_and_commit_main(trace, domain, transcript)
        else {
            return Err(ProvingError::EmptyCommitment);
        };

        let main = Round1CommitmentData::<Field> {
            trace_polys,
            lde_trace_merkle_tree: main_merkle_tree,
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
        // Interpolate all columns (needed for constraint evaluation)
        let Some((trace_polys, evaluations, _full_tree, _full_root)) =
            Self::interpolate_and_commit_preprocessed(trace, domain)
        else {
            return Err(ProvingError::EmptyCommitment);
        };

        // --- Build PRECOMPUTED tree (cols 0..num_precomputed) ---
        let precomputed_evaluations: Vec<_> = evaluations[..num_precomputed_cols].to_vec();
        let mut precomputed_lde_permuted = precomputed_evaluations.clone();
        for col in precomputed_lde_permuted.iter_mut() {
            in_place_bit_reverse_permute(col);
        }
        let precomputed_rows = columns2rows(precomputed_lde_permuted);
        let (precomputed_tree, precomputed_root) =
            Self::batch_commit_main(&precomputed_rows).ok_or(ProvingError::EmptyCommitment)?;

        // --- Build MULTIPLICITIES tree (cols num_precomputed..) ---
        let multiplicity_evaluations: Vec<_> = evaluations[num_precomputed_cols..].to_vec();
        let mut mult_lde_permuted = multiplicity_evaluations.clone();
        for col in mult_lde_permuted.iter_mut() {
            in_place_bit_reverse_permute(col);
        }
        let mult_rows = columns2rows(mult_lde_permuted);
        let (mult_tree, mult_root) =
            Self::batch_commit_main(&mult_rows).ok_or(ProvingError::EmptyCommitment)?;

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
            trace_polys,
            lde_trace_merkle_tree: mult_tree,
            lde_trace_merkle_root: mult_root,
            precomputed_merkle_tree: Some(precomputed_tree),
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
            let Some((
                aux_trace_polys,
                aux_trace_polys_evaluations,
                aux_merkle_tree,
                aux_merkle_root,
            )) = Self::interpolate_and_commit_aux(trace, domain, transcript)
            else {
                return Err(ProvingError::EmptyCommitment);
            };
            let aux = Some(Round1CommitmentData::<FieldExtension> {
                trace_polys: aux_trace_polys,
                lde_trace_merkle_tree: aux_merkle_tree,
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

    /// Returns the Merkle tree and the commitment to the evaluations of the parts of the
    /// composition polynomial.
    fn commit_composition_polynomial(
        lde_composition_poly_parts_evaluations: &[Vec<FieldElement<FieldExtension>>],
    ) -> Option<(BatchedMerkleTree<FieldExtension>, Commitment)>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        // TODO: Remove clones
        let mut lde_composition_poly_evaluations = Vec::new();
        for i in 0..lde_composition_poly_parts_evaluations[0].len() {
            let mut row = Vec::new();
            for evaluation in lde_composition_poly_parts_evaluations.iter() {
                row.push(evaluation[i].clone());
            }
            lde_composition_poly_evaluations.push(row);
        }

        in_place_bit_reverse_permute(&mut lde_composition_poly_evaluations);

        let mut lde_composition_poly_evaluations_merged = Vec::new();
        for chunk in lde_composition_poly_evaluations.chunks(2) {
            let (mut chunk0, chunk1) = (chunk[0].clone(), &chunk[1]);
            chunk0.extend_from_slice(chunk1);
            lde_composition_poly_evaluations_merged.push(chunk0);
        }

        Self::batch_commit_extension(&lde_composition_poly_evaluations_merged)
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

        // Get coefficients of the composition poly H
        let composition_poly =
            Polynomial::interpolate_offset_fft(&constraint_evaluations, &domain.coset_offset)
                .unwrap();

        let trace_length = domain.interpolation_domain_size;
        let number_of_parts = air.composition_poly_degree_bound(trace_length) / trace_length;
        let composition_poly_parts = composition_poly.break_in_parts(number_of_parts);

        let lde_composition_poly_parts_evaluations: Vec<_> = composition_poly_parts
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
            .collect();

        let Some((composition_poly_merkle_tree, composition_poly_root)) =
            Self::commit_composition_polynomial(&lde_composition_poly_parts_evaluations)
        else {
            return Err(ProvingError::EmptyCommitment);
        };

        Ok(Round2 {
            lde_composition_poly_evaluations: lde_composition_poly_parts_evaluations,
            composition_poly_parts,
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
        let z_power = z.pow(round_2_result.composition_poly_parts.len());

        // Evaluate H_i in z^N for all i, where N is the number of parts the composition poly was
        // broken into.
        let composition_poly_parts_ood_evaluation: Vec<_> = round_2_result
            .composition_poly_parts
            .iter()
            .map(|part| part.evaluate(&z_power))
            .collect();

        // Returns the Out of Domain Frame for the given trace polynomials, out of domain evaluation point (called `z` in the literature),
        // frame offsets given by the AIR and primitive root used for interpolating the trace polynomials.
        // An out of domain frame is nothing more than the evaluation of the trace polynomials in the points required by the
        // verifier to check the consistency between the trace and the composition polynomial.
        //
        // In the fibonacci example, the ood frame is simply the evaluations `[t(z), t(z * g), t(z * g^2)]`, where `t` is the trace
        // polynomial and `g` is the primitive root of unity used when interpolating `t`.
        let trace_ood_evaluations = crate::trace::get_trace_evaluations::<Field, FieldExtension>(
            &round_1_result.main.trace_polys,
            round_1_result
                .aux
                .as_ref()
                .map(|aux| &aux.trace_polys)
                .unwrap_or(&vec![]),
            z,
            &air.context().transition_offsets,
            &domain.trace_primitive_root,
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

        // Compute p₀ (deep composition polynomial)
        let deep_composition_poly = Self::compute_deep_composition_poly(
            &round_1_result.all_trace_polys(),
            round_2_result,
            round_3_result,
            z,
            &domain.trace_primitive_root,
            &gammas,
            &trace_term_coeffs,
        );

        let domain_size = domain.lde_roots_of_unity_coset.len();

        // FRI commit and query phases
        let (fri_last_value, fri_layers) = fri::commit_phase::<Field, FieldExtension>(
            domain.root_order as usize,
            deep_composition_poly,
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

    /// Returns the DEEP composition polynomial that the prover then commits to using
    /// FRI. This polynomial is a linear combination of the trace polynomial and the
    /// composition polynomial, with coefficients sampled by the verifier (i.e. using Fiat-Shamir).
    #[allow(clippy::too_many_arguments)]
    fn compute_deep_composition_poly(
        trace_polys: &[Polynomial<FieldElement<FieldExtension>>],
        round_2_result: &Round2<FieldExtension>,
        round_3_result: &Round3<FieldExtension>,
        z: &FieldElement<FieldExtension>,
        primitive_root: &FieldElement<Field>,
        composition_poly_gammas: &[FieldElement<FieldExtension>],
        trace_terms_gammas: &[Vec<FieldElement<FieldExtension>>],
    ) -> Polynomial<FieldElement<FieldExtension>>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let z_power = z.pow(round_2_result.composition_poly_parts.len());

        // ∑ᵢ 𝛾ᵢ ( Hᵢ − Hᵢ(z^N) ) / ( X − z^N )
        let mut h_terms = Polynomial::zero();
        for (i, part) in round_2_result.composition_poly_parts.iter().enumerate() {
            // h_i_eval is the evaluation of the i-th part of the composition polynomial at z^N,
            // where N is the number of parts of the composition polynomial.
            let h_i_eval = &round_3_result.composition_poly_parts_ood_evaluation[i];
            let h_i_term = &composition_poly_gammas[i] * (part - h_i_eval);
            h_terms = h_terms + h_i_term;
        }
        assert_eq!(h_terms.evaluate(&z_power), FieldElement::zero());
        h_terms.ruffini_division_inplace(&z_power);

        // Get trace evaluations needed for the trace terms of the deep composition polynomial
        let trace_frame_evaluations = &round_3_result.trace_ood_evaluations;

        // Compute the sum of all the trace terms of the deep composition polynomial.
        // There is one term for every trace polynomial and for every row in the frame.
        // ∑ ⱼₖ [ 𝛾ₖ ( tⱼ − tⱼ(z) ) / ( X − zgᵏ )]

        let trace_evaluations_columns = &trace_frame_evaluations.columns();

        #[cfg(feature = "parallel")]
        let trace_terms = trace_polys
            .par_iter()
            .enumerate()
            .fold(Polynomial::zero, |trace_terms, (i, t_j)| {
                let gammas_i = &trace_terms_gammas[i];
                let trace_evaluations_i = &trace_evaluations_columns[i];
                Self::compute_trace_term(
                    &trace_terms,
                    t_j,
                    gammas_i,
                    trace_evaluations_i,
                    (z, primitive_root),
                )
            })
            .reduce(Polynomial::zero, |a, b| a + b);

        #[cfg(not(feature = "parallel"))]
        let trace_terms =
            trace_polys
                .iter()
                .enumerate()
                .fold(Polynomial::zero(), |trace_terms, (i, t_j)| {
                    let gammas_i = &trace_terms_gammas[i];
                    let trace_evaluations_i = &trace_evaluations_columns[i];
                    Self::compute_trace_term(
                        &trace_terms,
                        t_j,
                        gammas_i,
                        trace_evaluations_i,
                        (z, primitive_root),
                    )
                });

        h_terms + trace_terms
    }

    // FIXME: FIX THIS DOCS!
    /// Adds to `accumulator` the term corresponding to the trace polynomial `t_j` of the Deep
    /// composition polynomial. That is, returns `accumulator + \sum_i \gamma_i \frac{ t_j - t_j(zg^i) }{ X - zg^i }`,
    /// where `i` ranges from `T * j` to `T * j + T - 1`, where `T` is the number of offsets in every frame.
    fn compute_trace_term(
        accumulator: &Polynomial<FieldElement<FieldExtension>>,
        trace_term_poly: &Polynomial<FieldElement<FieldExtension>>,
        trace_terms_gammas: &[FieldElement<FieldExtension>],
        trace_frame_evaluations: &[FieldElement<FieldExtension>],
        (z, primitive_root): (&FieldElement<FieldExtension>, &FieldElement<Field>),
    ) -> Polynomial<FieldElement<FieldExtension>>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let trace_int = trace_frame_evaluations
            .iter()
            .enumerate()
            .zip(trace_terms_gammas)
            .fold(
                Polynomial::zero(),
                |trace_agg, ((offset, trace_term_poly_evaluation), trace_gamma)| {
                    // @@@ this can be pre-computed
                    let z_shifted = primitive_root.pow(offset) * z;
                    let mut poly = trace_term_poly - trace_term_poly_evaluation;
                    poly.ruffini_division_inplace(&z_shifted);
                    trace_agg + poly * trace_gamma
                },
            );

        accumulator + trace_int
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
    /// element.
    fn open_trace_polys<E>(
        domain: &Domain<Field>,
        tree: &BatchedMerkleTree<E>,
        lde_trace: &Table<E>,
        challenge: usize,
    ) -> PolynomialOpenings<E>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<E>: AsBytes + Sync + Send,
        Field: IsSubFieldOf<E>,
        E: IsField,
    {
        let domain_size = domain.lde_roots_of_unity_coset.len();

        let index = challenge * 2;
        let index_sym = challenge * 2 + 1;
        PolynomialOpenings {
            proof: tree.get_proof_by_pos(index).unwrap(),
            proof_sym: tree.get_proof_by_pos(index_sym).unwrap(),
            evaluations: lde_trace
                .get_row(reverse_index(index, domain_size as u64))
                .to_vec(),
            evaluations_sym: lde_trace
                .get_row(reverse_index(index_sym, domain_size as u64))
                .to_vec(),
        }
    }

    /// Variant of open_trace_polys that takes a column range for slicing.
    /// Used for preprocessed tables where we need to open only a subset of columns
    /// (either precomputed or multiplicities).
    fn open_trace_polys_with_columns<E>(
        domain: &Domain<Field>,
        tree: &BatchedMerkleTree<E>,
        lde_trace: &Table<E>,
        challenge: usize,
        col_start: usize,
        col_end: usize,
    ) -> PolynomialOpenings<E>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<E>: AsBytes + Sync + Send,
        Field: IsSubFieldOf<E>,
        E: IsField,
    {
        let domain_size = domain.lde_roots_of_unity_coset.len();

        let index = challenge * 2;
        let index_sym = challenge * 2 + 1;
        PolynomialOpenings {
            proof: tree.get_proof_by_pos(index).unwrap(),
            proof_sym: tree.get_proof_by_pos(index_sym).unwrap(),
            evaluations: lde_trace.get_row(reverse_index(index, domain_size as u64))
                [col_start..col_end]
                .to_vec(),
            evaluations_sym: lde_trace.get_row(reverse_index(index_sym, domain_size as u64))
                [col_start..col_end]
                .to_vec(),
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
        let total_cols = round_1_result.lde_trace.main_table.width;

        for index in indexes_to_open.iter() {
            // For preprocessed tables, open main (multiplicities) with column range
            // For normal tables, open all columns
            let main_trace_opening = if is_preprocessed {
                Self::open_trace_polys_with_columns::<Field>(
                    domain,
                    &round_1_result.main.lde_trace_merkle_tree,
                    &round_1_result.lde_trace.main_table,
                    *index,
                    num_precomputed_cols,
                    total_cols,
                )
            } else {
                Self::open_trace_polys::<Field>(
                    domain,
                    &round_1_result.main.lde_trace_merkle_tree,
                    &round_1_result.lde_trace.main_table,
                    *index,
                )
            };

            // For preprocessed tables, also open precomputed tree
            let precomputed_trace_opening = round_1_result
                .main
                .precomputed_merkle_tree
                .as_ref()
                .map(|tree| {
                    Self::open_trace_polys_with_columns::<Field>(
                        domain,
                        tree,
                        &round_1_result.lde_trace.main_table,
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
                Self::open_trace_polys::<FieldExtension>(
                    domain,
                    &aux.lde_trace_merkle_tree,
                    &round_1_result.lde_trace.aux_table,
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

    // FIXME remove unwrap() calls and return errors
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
        // Round 1, Phase A: Commit all main traces
        // =====================================================================
        // All main trace commitments must be in the transcript before sampling
        // LogUp challenges. This ensures the challenges depend on ALL tables.
        //
        // For preprocessed tables (e.g., bitwise lookup), we use a hardcoded
        // commitment instead of computing one. Both prover and verifier use the
        // same hardcoded value in the transcript.

        let mut domains = Vec::with_capacity(num_airs);
        let mut main_commitments: Vec<MainCommitment<Field>> = Vec::with_capacity(num_airs);

        for (air, trace, _pub_inputs) in &*air_trace_pairs {
            let trace_length = trace.num_rows();
            let domain = new_domain(*air, trace_length);

            let (main, evaluations) = if air.is_preprocessed() {
                // Preprocessed table: use hardcoded commitment for precomputed columns
                Self::round_1_commit_preprocessed_trace(
                    *trace,
                    &domain,
                    transcript,
                    air.precomputed_commitment(),
                    air.num_precomputed_columns(),
                )?
            } else {
                // Normal table: compute commitment as usual
                Self::round_1_commit_main_trace(*trace, &domain, transcript)?
            };

            main_commitments.push((main, evaluations));
            domains.push(domain);
        }

        // =====================================================================
        // Round 1, Phase B: Sample shared LogUp challenges
        // =====================================================================
        // For the LogUp bus to balance (sum of fingerprints = 0), all tables
        // must use identical (z, α) challenges. We sample them ONCE here.

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
        // Each AIR builds its LogUp running-sum columns using the shared challenges.

        let mut round_1_results: Vec<Round1<Field, FieldExtension>> = Vec::with_capacity(num_airs);
        for (((air, trace, _pub_inputs), (main, main_evaluations)), domain) in air_trace_pairs
            .iter_mut()
            .zip(main_commitments)
            .zip(domains.iter())
        {
            let round_1_result = Self::round_1_build_auxiliary_trace(
                *air,
                *trace,
                domain,
                transcript,
                main,
                main_evaluations,
                logup_challenges.clone(),
            )?;
            round_1_results.push(round_1_result);
        }

        #[cfg(feature = "debug-checks")]
        print_bus_balance_report(&round_1_results);

        // =====================================================================
        // Rounds 2-4: Standard STARK protocol for each AIR
        // =====================================================================

        let mut proofs = Vec::with_capacity(num_airs);
        for (((air, _, pub_inputs), round_1_result), domain) in
            air_trace_pairs.iter().zip(round_1_results).zip(domains)
        {
            let proof =
                Self::prove_rounds_2_to_4(*air, *pub_inputs, &round_1_result, transcript, &domain)?;
            proofs.push(proof);
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

    // FIXME remove unwrap() calls and return errors
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

        #[cfg(feature = "debug-checks")]
        validate_trace(
            air,
            pub_inputs,
            &round_1_result.main.trace_polys,
            round_1_result
                .aux
                .as_ref()
                .map(|a| &a.trace_polys)
                .unwrap_or(&vec![]),
            domain,
            &round_1_result.rap_challenges,
        );

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
    use crate::{
        examples::simple_fibonacci::{self},
        proof::options::ProofOptions,
    };

    use super::*;
    use math::{
        field::{
            element::FieldElement,
            goldilocks::GoldilocksField,
            traits::IsFFTField,
        },
        polynomial::Polynomial,
    };

    type Felt = FieldElement<GoldilocksField>;

    #[test]
    fn test_domain_constructor() {
        let trace = simple_fibonacci::fibonacci_trace([Felt::from(1), Felt::from(1)], 8);
        let trace_length = trace.num_rows();
        let coset_offset = 3;
        let blowup_factor: usize = 2;
        let grinding_factor = 20;

        let proof_options = ProofOptions {
            blowup_factor: blowup_factor as u8,
            fri_number_of_queries: 1,
            coset_offset,
            grinding_factor,
        };

        let domain = Domain::new(
            &simple_fibonacci::FibonacciAIR::new(&proof_options),
            trace_length,
        );
        assert_eq!(domain.blowup_factor, 2);
        assert_eq!(domain.interpolation_domain_size, trace_length);
        assert_eq!(domain.root_order, trace_length.trailing_zeros());
        assert_eq!(domain.coset_offset, FieldElement::from(coset_offset));

        let primitive_root = GoldilocksField::get_primitive_root_of_unity(
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
        let trace = simple_fibonacci::fibonacci_trace([Felt::from(1), Felt::from(1)], 8);

        let trace_length = trace.num_rows();

        let trace_polys = trace.compute_trace_polys_main::<GoldilocksField>();
        let coset_offset = Felt::from(3);
        let blowup_factor: usize = 2;
        let domain_size = 8;

        let primitive_root = GoldilocksField::get_primitive_root_of_unity(
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
        let poly = Polynomial::new_monomial(Felt::one(), 8);
        let blowup_factor: usize = 4;
        let domain_size: usize = 8;
        let offset = Felt::from(3);
        let evaluations =
            evaluate_polynomial_on_lde_domain(&poly, blowup_factor, domain_size, &offset).unwrap();
        assert_eq!(evaluations.len(), domain_size * blowup_factor);

        let primitive_root: Felt = GoldilocksField::get_primitive_root_of_unity(
            (domain_size * blowup_factor).trailing_zeros() as u64,
        )
        .unwrap();
        for (i, eval) in evaluations.iter().enumerate() {
            assert_eq!(*eval, poly.evaluate(&(offset * primitive_root.pow(i))));
        }
    }

}
