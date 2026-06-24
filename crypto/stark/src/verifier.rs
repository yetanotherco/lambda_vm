use super::{
    domain::VerifierDomain,
    grinding,
    proof::stark::StarkProof,
    traits::{AIR, TransitionEvaluationContext},
};
use crate::{
    config::Commitment,
    domain::new_verifier_domain,
    lookup::{LOGUP_CHALLENGE_ALPHA, LOGUP_NUM_CHALLENGES, PackingShifts, compute_alpha_powers},
    proof::stark::MultiProof,
    proof::zerocopy::{DeepPolynomialOpeningRef, FriDecommitmentRef, StarkProofRef},
};
use alloc::vec::Vec;
use core::marker::PhantomData;
use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
#[cfg(not(feature = "test_fiat_shamir"))]
use log::error;
#[cfg(feature = "debug-checks")]
use log::info;
use math::{
    fft::bit_reversing::reverse_index,
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
    traits::AsBytes,
};
#[cfg(feature = "instruments")]
use std::time::Instant;

/// A default STARK verifier implementing `IsStarkVerifier`.
pub struct Verifier<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
    PI,
> {
    phantom: PhantomData<(Field, FieldExtension, PI)>,
}

impl<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: IsField + Send + Sync,
    PI,
> IsStarkVerifier<Field, FieldExtension, PI> for Verifier<Field, FieldExtension, PI>
{
}

/// A container holding the complete list of challenges sent to the prover along with the seed used
/// to validate the proof-of-work nonce.
pub struct Challenges<FieldExtension>
where
    FieldExtension: Send + Sync + IsField,
{
    /// The out-of-domain challenge.
    pub z: FieldElement<FieldExtension>,
    /// The composition polynomial coefficients corresponding to the boundary constraints terms.
    pub boundary_coeffs: Vec<FieldElement<FieldExtension>>,
    /// The composition polynomial coefficients corresponding to the transition constraints terms.
    pub transition_coeffs: Vec<FieldElement<FieldExtension>>,
    /// The deep composition polynomial coefficients corresponding to the trace
    /// polynomial terms, stored **flat** in column-major order: the coefficient
    /// for trace column `col` and OOD row `row` is at index
    /// `col * trace_term_chunk_len + row`. Flattening the former
    /// `Vec<Vec<FieldElement>>` (one inner `Vec` per column) into a single buffer
    /// removes the per-column heap allocations that dominated the verifier's
    /// per-table allocation cost, and gives the deep-composition reconstruction
    /// a contiguous slice to index.
    pub trace_term_coeffs: Vec<FieldElement<FieldExtension>>,
    /// Stride (number of OOD rows) of each column's run in `trace_term_coeffs`.
    pub trace_term_chunk_len: usize,
    /// The deep composition polynomial coefficients corresponding to the composition polynomial parts terms.
    pub gammas: Vec<FieldElement<FieldExtension>>,
    /// The list of FRI commit phase folding challenges.
    pub zetas: Vec<FieldElement<FieldExtension>>,
    /// The list of FRI query phase index challenges.
    pub iotas: Vec<usize>,
    /// The challenges used to build the auxiliary trace.
    pub rap_challenges: Vec<FieldElement<FieldExtension>>,
    /// The seed used to verify the proof-of-work nonce.
    pub grinding_seed: [u8; 32],
}

pub type DeepPolynomialEvaluations<F> = (Vec<FieldElement<F>>, Vec<FieldElement<F>>);

/// Reusable scratch buffers threaded through the per-table verification loop so
/// the work each table does in `step_2_verify_claimed_composition_polynomial`
/// allocates once (on the first table) and reuses the same backing storage for
/// every subsequent table, rather than allocating a fresh `Vec` per table.
///
/// Public only because it appears in the signatures of the `pub` trait methods
/// `verify_rounds_2_to_4` / `step_2_verify_claimed_composition_polynomial`; it
/// is an internal implementation detail and not part of the stable API.
#[doc(hidden)]
pub struct VerifyScratch<FieldExtension>
where
    FieldExtension: Send + Sync + IsField,
{
    /// Transition-constraint evaluations at the OOD point (length =
    /// `num_transition_constraints`), filled by `compute_transition_into`.
    transition_evals: Vec<FieldElement<FieldExtension>>,
    /// Per-constraint zerofier denominators (same length as `transition_evals`).
    denominators: Vec<FieldElement<FieldExtension>>,
}

impl<FieldExtension> VerifyScratch<FieldExtension>
where
    FieldExtension: Send + Sync + IsField,
{
    fn new() -> Self {
        Self {
            transition_evals: Vec::new(),
            denominators: Vec::new(),
        }
    }
}

/// The functionality of a STARK verifier providing methods to run the STARK Verify protocol
/// https://lambdaclass.github.io/lambdaworks/starks/protocol.html
pub trait IsStarkVerifier<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
    PI,
>
{
    fn sample_query_indexes(
        number_of_queries: usize,
        domain: &VerifierDomain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Vec<usize> {
        let domain_size = domain.lde_length as u64;
        (0..number_of_queries)
            .map(|_| (transcript.sample_u64(domain_size >> 1)) as usize)
            .collect::<Vec<usize>>()
    }

    /// Returns the list of challenges sent to the prover.
    fn step_1_replay_rounds_and_recover_challenges<'p, P>(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &P,
        domain: &VerifierDomain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Challenges<FieldExtension>
    where
        P: crate::proof::zerocopy::StarkProofRef<'p, Field, FieldExtension, PI>,
        Field: 'p,
        FieldExtension: 'p,
        PI: 'p,
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
    {
        // ===================================
        // ==========|   Round 1   |==========
        // ===================================

        // <<<< Receive commitments:[tⱼ]
        transcript.append_bytes(proof.lde_trace_main_merkle_root());

        let rap_challenges = air.build_rap_challenges(transcript);

        if let Some(root) = proof.lde_trace_aux_merkle_root() {
            transcript.append_bytes(root);
        }

        // ===================================
        // ==========|   Round 2   |==========
        // ===================================

        // <<<< Receive challenge: 𝛽
        let beta = transcript.sample_field_element();
        let trace_length = proof.trace_length();
        let bus_public_inputs = proof
            .bus_table_contribution()
            .map(|c| crate::lookup::BusPublicInputs::from_contribution(c.clone()));
        let num_boundary_constraints = air
            .boundary_constraints(
                proof.public_inputs(),
                &rap_challenges,
                bus_public_inputs.as_ref(),
                trace_length,
            )
            .constraints
            .len();

        let num_transition_constraints = air.context().num_transition_constraints;

        let mut coefficients =
            compute_alpha_powers(&beta, num_boundary_constraints + num_transition_constraints);

        let transition_coeffs: Vec<_> = coefficients.drain(..num_transition_constraints).collect();
        let boundary_coeffs = coefficients;

        // <<<< Receive commitments: [H₁], [H₂]
        transcript.append_bytes(proof.composition_poly_root());

        // ===================================
        // ==========|   Round 3   |==========
        // ===================================

        // >>>> Send challenge: z
        let z = transcript.sample_z_ood_with_domain_params(
            domain.trace_length,
            domain.lde_length,
            &domain.coset_offset,
        );

        // <<<< Receive values: tⱼ(zgᵏ)
        // Column-major append (matches `Table::columns()` order) without
        // materializing the transposed columns.
        let ood = proof.trace_ood_evaluations();
        for col_idx in 0..ood.width() {
            for row_idx in 0..ood.height() {
                transcript.append_field_element(&ood.get_row(row_idx)[col_idx]);
            }
        }
        // <<<< Receive value: Hᵢ(z^N)
        let composition_poly_parts_ood = proof.composition_poly_parts_ood_evaluation();
        for element in composition_poly_parts_ood.iter() {
            transcript.append_field_element(element);
        }

        // ===================================
        // ==========|   Round 4   |==========
        // ===================================

        let num_terms_composition_poly = composition_poly_parts_ood.len();
        let num_terms_trace =
            air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;
        let gamma = transcript.sample_field_element();

        // <<<< Receive challenges: 𝛾, 𝛾'
        let mut deep_composition_coefficients: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
                .take(num_terms_composition_poly + num_terms_trace)
                .collect();

        // Split the contiguous coefficient buffer: the trace terms are the first
        // `num_terms_trace` (kept flat, column-major with stride `chunk_len`), the
        // composition-poly gammas are the rest. `split_off(num_terms_trace)` hands
        // the suffix to `gammas` and leaves the (already contiguous) trace prefix
        // as `trace_term_coeffs` — no per-column `Vec` allocation, no copy.
        let chunk_len = air.context().transition_offsets.len() * air.step_size();
        // <<<< Receive challenges: 𝛾ⱼ, 𝛾ⱼ'
        let gammas = deep_composition_coefficients.split_off(num_terms_trace);
        let trace_term_coeffs = deep_composition_coefficients;
        let trace_term_chunk_len = chunk_len;

        // FRI commit phase
        let merkle_roots = proof.fri_layers_merkle_roots();
        let mut zetas = merkle_roots
            .iter()
            .map(|root| {
                // >>>> Send challenge 𝜁ₖ
                let element = transcript.sample_field_element();
                // <<<< Receive commitment: [pₖ] (the first one is [p₀])
                transcript.append_bytes(root);
                element
            })
            .collect::<Vec<FieldElement<FieldExtension>>>();

        // >>>> Send challenge 𝜁ₙ₋₁
        zetas.push(transcript.sample_field_element());

        // <<<< Receive value: pₙ
        transcript.append_field_element(proof.fri_last_value());

        // Receive grinding value
        let security_bits = air.context().proof_options.grinding_factor;
        let mut grinding_seed = [0u8; 32];
        if security_bits > 0
            && let Some(nonce_value) = proof.nonce()
        {
            grinding_seed = transcript.state();
            transcript.append_bytes(&nonce_value.to_be_bytes());
        }

        // FRI query phase
        // <<<< Send challenges 𝜄ₛ (iota_s)
        let number_of_queries = air.options().fri_number_of_queries;
        let iotas = Self::sample_query_indexes(number_of_queries, domain, transcript);

        Challenges {
            z,
            boundary_coeffs,
            transition_coeffs,
            trace_term_coeffs,
            trace_term_chunk_len,
            gammas,
            zetas,
            iotas,
            rap_challenges,
            grinding_seed,
        }
    }

    /// Checks whether the purported evaluations of the composition polynomial parts and the trace
    /// polynomials at the out-of-domain challenge are consistent.
    /// See https://lambdaclass.github.io/lambdaworks/starks/protocol.html#step-2-verify-claimed-composition-polynomial
    fn step_2_verify_claimed_composition_polynomial<'p, P>(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &P,
        domain: &VerifierDomain<Field>,
        challenges: &Challenges<FieldExtension>,
        scratch: &mut VerifyScratch<FieldExtension>,
    ) -> bool
    where
        P: StarkProofRef<'p, Field, FieldExtension, PI>,
        Field: 'p,
        FieldExtension: 'p,
        PI: 'p,
    {
        let trace_length = proof.trace_length();
        let ood = proof.trace_ood_evaluations();
        // Reconstruct an owned BusPublicInputs (just the table contribution L —
        // one field element) from the borrowed view for the AIR boundary call.
        let bus_public_inputs = proof
            .bus_table_contribution()
            .map(|c| crate::lookup::BusPublicInputs::from_contribution(c.clone()));
        let boundary_constraints = air.boundary_constraints(
            proof.public_inputs(),
            &challenges.rap_challenges,
            bus_public_inputs.as_ref(),
            trace_length,
        );
        let number_of_b_constraints = boundary_constraints.constraints.len();

        let mut boundary_step_points: Vec<(usize, FieldElement<Field>)> = Vec::new();

        #[allow(clippy::type_complexity)]
        let (boundary_c_i_evaluations_num, mut boundary_c_i_evaluations_den): (
            Vec<FieldElement<FieldExtension>>,
            Vec<FieldElement<FieldExtension>>,
        ) = (0..number_of_b_constraints)
            .map(|index| {
                let step = boundary_constraints.constraints[index].step;
                let is_aux = boundary_constraints.constraints[index].is_aux;
                let point = match boundary_step_points.iter().find(|(s, _)| *s == step) {
                    Some((_, p)) => p.clone(),
                    None => {
                        let p = domain.trace_primitive_root.pow(step as u64);
                        boundary_step_points.push((step, p.clone()));
                        p
                    }
                };
                let column_idx = boundary_constraints.constraints[index].col;
                let trace_evaluation = if is_aux {
                    let column_idx = air.trace_layout().0 + column_idx;
                    &ood.get_row(0)[column_idx]
                } else {
                    &ood.get_row(0)[column_idx]
                };
                let boundary_zerofier_challenges_z_den = -point + &challenges.z;

                let boundary_quotient_ood_evaluation_num =
                    -&boundary_constraints.constraints[index].value + trace_evaluation;

                (
                    boundary_quotient_ood_evaluation_num,
                    boundary_zerofier_challenges_z_den,
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .unzip();

        FieldElement::inplace_batch_inverse(&mut boundary_c_i_evaluations_den).unwrap();

        let boundary_quotient_ood_evaluation: FieldElement<FieldExtension> = {
            let mut acc = FieldElement::<FieldExtension>::zero();
            for ((num, den), beta) in boundary_c_i_evaluations_num
                .iter()
                .zip(&boundary_c_i_evaluations_den)
                .zip(&challenges.boundary_coeffs)
            {
                acc.fma(&(num.clone() * den), beta);
            }
            acc
        };

        let periodic_values = air
            .get_periodic_column_polynomials(trace_length)
            .iter()
            .map(|poly| poly.evaluate(&challenges.z))
            .collect::<Vec<FieldElement<FieldExtension>>>();

        let num_main_trace_columns = ood.width() - air.num_auxiliary_rap_columns();

        let logup_alpha_powers: Vec<FieldElement<FieldExtension>> =
            if challenges.rap_challenges.len() > LOGUP_CHALLENGE_ALPHA {
                compute_alpha_powers(
                    &challenges.rap_challenges[LOGUP_CHALLENGE_ALPHA],
                    air.max_bus_elements(),
                )
            } else {
                Vec::new()
            };

        let logup_table_offset = match proof.bus_table_contribution() {
            Some(table_contribution) => {
                let n = FieldElement::<Field>::from(trace_length as u64);
                match n.inv() {
                    Ok(n_inv) => n_inv * table_contribution,
                    Err(_) => return false, // trace_length == 0 is invalid
                }
            }
            None => FieldElement::zero(),
        };

        let ood_frame = ood.into_frame(num_main_trace_columns, air.step_size());
        let packing_shifts = PackingShifts::<FieldExtension>::new();
        let transition_evaluation_context = TransitionEvaluationContext::new_verifier(
            &ood_frame,
            &periodic_values,
            &challenges.rap_challenges,
            &logup_alpha_powers,
            &logup_table_offset,
            &packing_shifts,
        );
        // Reuse the caller-owned scratch buffers across tables: size them to this
        // table's constraint count, then fill in place (`compute_transition_into`
        // zeroes the buffer itself, so a `resize` is enough to set the length).
        let num_transition_constraints = air.num_transition_constraints();
        scratch
            .transition_evals
            .resize(num_transition_constraints, FieldElement::zero());
        air.compute_transition_into(
            &transition_evaluation_context,
            &mut scratch.transition_evals,
        );

        // Reuse the caller-owned scratch buffer for zerofier denominators, and
        // memoize by constraint "shape". The zerofier value depends only on the
        // OOD point `z`, the trace primitive root, the trace length, and the
        // constraint's shape (period / offset / exemption parameters) — not on
        // its index. Many constraints in a table share the same shape (e.g. every
        // plain every-row constraint), so `evaluate_zerofier` otherwise recomputes
        // the same `(z^(n/period) - g^…)⁻¹ · P_exempt(z)` — an extension-field
        // `pow`, a field inversion, and an `end_exemptions_poly` allocation — once
        // per constraint. Memoize per distinct shape (a short linear scan; the
        // number of shapes is tiny) so the heavy work runs once per shape.
        scratch
            .denominators
            .resize(num_transition_constraints, FieldElement::zero());
        type ZerofierShape = (usize, usize, Option<usize>, Option<usize>, usize);
        let mut zerofier_cache: Vec<(ZerofierShape, FieldElement<FieldExtension>)> = Vec::new();
        air.transition_constraints().iter().for_each(|c| {
            let shape: ZerofierShape = (
                c.period(),
                c.offset(),
                c.exemptions_period(),
                c.periodic_exemptions_offset(),
                c.end_exemptions(),
            );
            let zerofier = match zerofier_cache.iter().find(|(s, _)| *s == shape) {
                Some((_, value)) => value.clone(),
                None => {
                    let value = c.evaluate_zerofier(
                        &challenges.z,
                        &domain.trace_primitive_root,
                        trace_length,
                    );
                    zerofier_cache.push((shape, value.clone()));
                    value
                }
            };
            scratch.denominators[c.constraint_idx()] = zerofier;
        });

        let transition_c_i_evaluations_sum = {
            let mut acc = FieldElement::zero();
            for (eval, beta, denominator) in itertools::izip!(
                &scratch.transition_evals,
                &challenges.transition_coeffs,
                &scratch.denominators
            ) {
                acc.fma(&(beta.clone() * eval), denominator);
            }
            acc
        };

        let composition_poly_ood_evaluation =
            &boundary_quotient_ood_evaluation + transition_c_i_evaluations_sum;

        let composition_poly_claimed_ood_evaluation = proof
            .composition_poly_parts_ood_evaluation()
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, coeff| {
                acc * &challenges.z + coeff
            });

        composition_poly_claimed_ood_evaluation == composition_poly_ood_evaluation
    }

    /// Reconstructs the Deep composition polynomial evaluations at the challenge indices values using the provided
    /// openings of the trace polynomials and the composition polynomial parts. It then uses these to verify that the
    /// FRI decommitments are valid and correspond to the Deep composition polynomial.
    fn step_3_verify_fri<'p, P>(
        proof: &P,
        domain: &VerifierDomain<Field>,
        challenges: &Challenges<FieldExtension>,
    ) -> bool
    where
        P: StarkProofRef<'p, Field, FieldExtension, PI>,
        Field: 'p,
        FieldExtension: 'p,
        PI: 'p,
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    {
        let (deep_poly_evaluations, deep_poly_evaluations_sym) =
            Self::reconstruct_deep_composition_poly_evaluations_for_all_queries(
                challenges, domain, proof,
            );

        // verify FRI
        let mut evaluation_point_inverse = challenges
            .iotas
            .iter()
            .map(|iota| Self::query_challenge_to_evaluation_point(*iota, domain))
            .collect::<Vec<FieldElement<Field>>>();
        FieldElement::inplace_batch_inverse(&mut evaluation_point_inverse).unwrap();

        let mut leaf_scratch: Vec<u8> = Vec::new();
        challenges
            .iotas
            .iter()
            .zip(evaluation_point_inverse)
            .enumerate()
            .fold(true, |mut result, (i, (iota_s, eval))| {
                let query = proof.query(i);
                result &= Self::verify_query_and_sym_openings(
                    proof,
                    &challenges.zetas,
                    *iota_s,
                    &query,
                    eval,
                    &deep_poly_evaluations[i],
                    &deep_poly_evaluations_sym[i],
                    &mut leaf_scratch,
                );
                result
            })
    }

    /// Returns the field element element of the domain `domain` corresponding to the given FRI query index challenge `iota`.
    fn query_challenge_to_evaluation_point(
        iota: usize,
        domain: &VerifierDomain<Field>,
    ) -> FieldElement<Field> {
        let index = reverse_index(iota * 2, domain.lde_length as u64);
        domain.lde_coset_element(index)
    }

    /// Returns the symmetric field element element of the domain `domain` corresponding to the given FRI query index challenge `iota`.
    fn query_challenge_to_evaluation_point_sym(
        iota: usize,
        domain: &VerifierDomain<Field>,
    ) -> FieldElement<Field> {
        let index = reverse_index(iota * 2 + 1, domain.lde_length as u64);
        domain.lde_coset_element(index)
    }

    /// Verify opening Open(tⱼ(D_LDE), 𝜐) and Open(tⱼ(D_LDE), -𝜐) for all trace polynomials tⱼ,
    /// where 𝜐 and -𝜐 are the elements corresponding to the index challenge `iota`.
    ///
    /// Uses the paired opening variant for the (index, index_sym) = (iota*2, iota*2+1) pairs:
    /// since both indices are always in the same quaternary (ARITY=4) level-0 group, the
    /// level-0 parent and all ancestors are shared, so each commitment root is verified with
    /// one ancestor-path walk instead of two independent ones.
    fn verify_trace_openings<'p, P>(
        proof: &P,
        deep_poly_openings: &DeepPolynomialOpeningRef<'_, Field, FieldExtension>,
        iota: usize,
        leaf_scratch: &mut Vec<u8>,
    ) -> bool
    where
        P: StarkProofRef<'p, Field, FieldExtension, PI>,
        Field: 'p,
        FieldExtension: 'p,
        PI: 'p,
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    {
        // index = iota*2, index_sym = iota*2+1 are always in the same ARITY=4
        // level-0 group — use the paired variant to walk the ancestor path once.
        let index = iota * 2;
        let mut result = true;

        let main_root = proof.lde_trace_main_merkle_root();

        // Main trace: both proof and proof_sym paths share the same level-0 group.
        // verify_paired_batched_openings hashes both leaves and walks ancestors once.
        result &= crate::config::verify_paired_batched_openings::<Field>(
            deep_poly_openings.main_trace_polys.proof,
            main_root,
            index,
            deep_poly_openings.main_trace_polys.evaluations,
            deep_poly_openings.main_trace_polys.evaluations_sym,
            leaf_scratch,
        );

        // Verify precomputed trace (for preprocessed tables only)
        match (
            proof.lde_trace_precomputed_merkle_root(),
            &deep_poly_openings.precomputed_trace_polys,
        ) {
            // Unreachable: multi_verify() already rejected proofs with None root for preprocessed AIRs,
            // and non-preprocessed AIRs never have openings. No valid execution path reaches here.
            (None, Some(_)) => result = false,
            (Some(_), None) => result = false,
            (Some(precomputed_root), Some(precomputed_opening)) => {
                result &= crate::config::verify_paired_batched_openings::<Field>(
                    precomputed_opening.proof,
                    precomputed_root,
                    index,
                    precomputed_opening.evaluations,
                    precomputed_opening.evaluations_sym,
                    leaf_scratch,
                );
            }
            _ => {}
        }

        // Verify auxiliary trace
        match (
            proof.lde_trace_aux_merkle_root(),
            &deep_poly_openings.aux_trace_polys,
        ) {
            (None, Some(_)) => result = false,
            (Some(_), None) => result = false,
            (Some(aux_root), Some(aux_trace_polys_opening)) => {
                result &= crate::config::verify_paired_batched_openings::<FieldExtension>(
                    aux_trace_polys_opening.proof,
                    aux_root,
                    index,
                    aux_trace_polys_opening.evaluations,
                    aux_trace_polys_opening.evaluations_sym,
                    leaf_scratch,
                );
            }
            _ => {}
        }

        result
    }

    /// Verify opening Open(Hᵢ(D_LDE), 𝜐) and Open(Hᵢ(D_LDE), -𝜐) for all parts Hᵢof the composition
    /// polynomial, where 𝜐 and -𝜐 are the elements corresponding to the index challenge `iota`.
    fn verify_composition_poly_opening(
        deep_poly_openings: &DeepPolynomialOpeningRef<'_, Field, FieldExtension>,
        composition_poly_merkle_root: &Commitment,
        iota: &usize,
        value: &mut Vec<FieldElement<FieldExtension>>,
        leaf_scratch: &mut Vec<u8>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    {
        // The composition-poly leaf is `evaluations` followed by `evaluations_sym`.
        // `value` is a caller-owned scratch buffer reused across queries: clear it
        // and refill from the two borrowed slices, hashing without a fresh `Vec`.
        value.clear();
        value.extend_from_slice(deep_poly_openings.composition_poly.evaluations);
        value.extend_from_slice(deep_poly_openings.composition_poly.evaluations_sym);

        crate::config::verify_batched_merkle_path_slice_with_scratch::<FieldExtension>(
            deep_poly_openings.composition_poly.proof,
            composition_poly_merkle_root,
            *iota,
            value,
            leaf_scratch,
        )
    }

    /// Verifies the validity of the purported values of the trace polynomials and the composition polynomial
    /// parts at the domain elements and their symmetric counterparts corresponding to all the FRI query
    /// index challenges.
    fn step_4_verify_trace_and_composition_openings<'p, P>(
        proof: &P,
        challenges: &Challenges<FieldExtension>,
    ) -> bool
    where
        P: StarkProofRef<'p, Field, FieldExtension, PI>,
        Field: 'p,
        FieldExtension: 'p,
        PI: 'p,
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    {
        let composition_poly_root = proof.composition_poly_root();
        // Scratch buffers reused across every query to avoid per-query allocation.
        let mut composition_leaf: Vec<FieldElement<FieldExtension>> = Vec::new();
        // `leaf_scratch` holds serialized field-element bytes for Merkle leaf hashing.
        let mut leaf_scratch: Vec<u8> = Vec::new();
        challenges
            .iotas
            .iter()
            .enumerate()
            .fold(true, |mut result, (i, iota_n)| {
                let deep_poly_opening = proof.deep_poly_opening(i);
                result &= Self::verify_composition_poly_opening(
                    &deep_poly_opening,
                    composition_poly_root,
                    iota_n,
                    &mut composition_leaf,
                    &mut leaf_scratch,
                );

                result &= Self::verify_trace_openings(
                    proof,
                    &deep_poly_opening,
                    *iota_n,
                    &mut leaf_scratch,
                );
                result
            })
    }

    /// Verifies the openings of a fold polynomial of an inner layer of FRI.
    fn verify_fri_layer_openings(
        merkle_root: &Commitment,
        auth_path_sym: &[Commitment],
        evaluation: &FieldElement<FieldExtension>,
        evaluation_sym: &FieldElement<FieldExtension>,
        iota: usize,
        leaf_scratch: &mut Vec<u8>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    {
        // Two-element leaf, ordered by parity of `iota`. Built on the stack as a
        // fixed-size array and hashed straight from the borrowed slice — no heap
        // allocation per FRI layer per query.
        let evaluations: [FieldElement<FieldExtension>; 2] = if iota % 2 == 1 {
            [evaluation_sym.clone(), evaluation.clone()]
        } else {
            [evaluation.clone(), evaluation_sym.clone()]
        };

        crate::config::verify_fri_merkle_path_slice_with_scratch::<FieldExtension>(
            auth_path_sym,
            merkle_root,
            iota >> 1,
            &evaluations,
            leaf_scratch,
        )
    }

    /// Verify a single FRI query
    /// `zetas`: the vector of all challenges sent by the verifier to the prover at the commit
    /// phase to fold polynomials.
    /// `iota`: the index challenge of this FRI query. This index uniquely determines two elements 𝜐 and -𝜐
    /// of the evaluation domain of FRI layer 0.
    /// `evaluation_point_inv`: precomputed value of 𝜐⁻¹.
    /// `deep_composition_evaluation`: precomputed value of p₀(𝜐), where p₀ is the deep composition polynomial.
    /// `deep_composition_evaluation_sym`: precomputed value of p₀(-𝜐), where p₀ is the deep composition polynomial.
    fn verify_query_and_sym_openings<'p, P>(
        proof: &P,
        zetas: &[FieldElement<FieldExtension>],
        iota: usize,
        fri_decommitment: &FriDecommitmentRef<'_, FieldExtension>,
        evaluation_point_inv: FieldElement<Field>,
        deep_composition_evaluation: &FieldElement<FieldExtension>,
        deep_composition_evaluation_sym: &FieldElement<FieldExtension>,
        leaf_scratch: &mut Vec<u8>,
    ) -> bool
    where
        P: StarkProofRef<'p, Field, FieldExtension, PI>,
        Field: 'p,
        FieldExtension: 'p,
        PI: 'p,
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    {
        let fri_layers_merkle_roots = proof.fri_layers_merkle_roots();
        let fri_last_value = proof.fri_last_value();

        let p0_eval = deep_composition_evaluation;
        let p0_eval_sym = deep_composition_evaluation_sym;

        // Reconstruct p₁(𝜐²)
        let mut v = (p0_eval + p0_eval_sym)
            + evaluation_point_inv.clone() * &zetas[0] * (p0_eval - p0_eval_sym);
        let mut index = iota;

        // Handle case with 0 FRI layers (trace_length <= 2)
        // In this case, the fold loop below doesn't iterate, so we need to verify
        // the final value directly here.
        if fri_layers_merkle_roots.is_empty() {
            return v == *fri_last_value;
        }

        let num_layer_evals = fri_decommitment.layers_evaluations_sym.len();

        // Lazy squaring iterator for the evaluation point powers — avoids
        // allocating a Vec per query by computing each power on demand.
        let evaluation_point_iter =
            core::iter::successors(Some(evaluation_point_inv.square()), |ep| {
                Some(ep.square())
            });

        // For each FRI layer, starting from the layer 1: use the proof to verify the validity of values pᵢ(−𝜐^(2ⁱ)) (given by the prover) and
        // pᵢ(𝜐^(2ⁱ)) (computed on the previous iteration by the verifier). Then use them to obtain pᵢ₊₁(𝜐^(2ⁱ⁺¹)).
        // Finally, check that the final value coincides with the given by the prover.
        fri_layers_merkle_roots
            .iter()
            .enumerate()
            .zip(fri_decommitment.layers_evaluations_sym)
            .zip(evaluation_point_iter)
            .fold(
                true,
                |result, (((i, merkle_root), evaluation_sym), evaluation_point_inv)| {
                    let auth_path_sym = fri_decommitment.layer_auth_path(i);
                    // Verify opening Open(pᵢ(Dₖ), −𝜐^(2ⁱ)) and Open(pᵢ(Dₖ), 𝜐^(2ⁱ)).
                    // `v` is pᵢ(𝜐^(2ⁱ)).
                    // `evaluation_sym` is pᵢ(−𝜐^(2ⁱ)).
                    let openings_ok = Self::verify_fri_layer_openings(
                        merkle_root,
                        auth_path_sym,
                        &v,
                        evaluation_sym,
                        index,
                        leaf_scratch,
                    );

                    // Update `v` with next value pᵢ₊₁(𝜐^(2ⁱ⁺¹)).
                    v = (&v + evaluation_sym)
                        + evaluation_point_inv * &zetas[i + 1] * (&v - evaluation_sym);

                    // Update index for next iteration. The index of the squares in the next layer
                    // is obtained by halving the current index. This is due to the bit-reverse
                    // ordering of the elements in the Merkle tree.
                    index >>= 1;

                    if i < num_layer_evals - 1 {
                        result & openings_ok
                    } else {
                        // Check that final value is the given by the prover
                        result & (v == *fri_last_value) & openings_ok
                    }
                },
            )
    }

    fn reconstruct_deep_composition_poly_evaluations_for_all_queries<'p, P>(
        challenges: &Challenges<FieldExtension>,
        domain: &VerifierDomain<Field>,
        proof: &P,
    ) -> DeepPolynomialEvaluations<FieldExtension>
    where
        P: StarkProofRef<'p, Field, FieldExtension, PI>,
        Field: 'p,
        FieldExtension: 'p,
        PI: 'p,
    {
        let num_queries = challenges.iotas.len();
        let mut deep_poly_evaluations = Vec::with_capacity(num_queries);
        let mut deep_poly_evaluations_sym = Vec::with_capacity(num_queries);
        // Scratch buffers reused across every query iteration.
        // Base-field columns (precomputed + main trace) stay as Field elements
        // so scalar_fma (Fp3ScalarFma ecall) handles acc += scalar × coeff,
        // avoiding both to_extension() copies and the Fp3 wrapper's extra 2-zero stores.
        // Extension-field columns (aux trace) use regular fma (Fp3Fma ecall).
        let mut evals_base: Vec<FieldElement<Field>> = Vec::new();
        let mut evals_base_sym: Vec<FieldElement<Field>> = Vec::new();
        let mut evals_ext: Vec<FieldElement<FieldExtension>> = Vec::new();
        let mut evals_ext_sym: Vec<FieldElement<FieldExtension>> = Vec::new();

        // Precompute the query-INVARIANT half of the deep-trace term, once for all
        // queries. The trace term is
        //   Σ_row denom_q[row] · Σ_col (lde_q[col] − ood[row][col])·coeff[col][row]
        // and only `lde_q` (the per-query opening) and `denom_q` (per-query point)
        // vary with the query. Splitting the column sum,
        //   Σ_col ood[row][col]·coeff[col][row]   =: b_terms[row]
        // depends only on the OOD table and the deep-composition coefficients —
        // both fixed across queries — so it is computed here once instead of being
        // recomputed inside every query (×num_queries, ×2 for the symmetric point).
        // On a realistic proof this function is ~56% of guest cycles and this term
        // was its dominant repeated work.
        let b_terms = Self::precompute_ood_coeff_terms(proof, challenges);
        // Hoist the primitive root computation out of the per-query loop — it is
        // the same value for every query (depends only on the domain order).
        let primitive_root =
            &Field::get_primitive_root_of_unity(domain.root_order as u64).unwrap();

        // Precompute `z^N_parts` once — both `challenges.z` and the number of
        // composition-poly parts are proof-global constants, so recomputing this
        // inside each of the 2×num_queries reconstruction calls wastes `num_parts`
        // field multiplications per call.
        let number_of_parts = proof.composition_poly_parts_ood_evaluation().len();
        let z_pow_n: FieldElement<FieldExtension> = challenges.z.pow(number_of_parts);

        // Batch-invert all 2×num_queries composition denominators in a single
        // `inplace_batch_inverse` call (1 inversion + 3×(2Q-1) muls) instead of
        // 2×num_queries independent `.inv()` calls inside the reconstruction loop.
        // Layout: [ep_0 − z^N, ep_sym_0 − z^N, ep_1 − z^N, ep_sym_1 − z^N, ...]
        let mut comp_denoms: Vec<FieldElement<FieldExtension>> =
            Vec::with_capacity(2 * num_queries);
        for iota in challenges.iotas.iter() {
            let ep = Self::query_challenge_to_evaluation_point(*iota, domain);
            let ep_sym = Self::query_challenge_to_evaluation_point_sym(*iota, domain);
            comp_denoms.push(ep.to_extension() - &z_pow_n);
            comp_denoms.push(ep_sym.to_extension() - &z_pow_n);
        }
        FieldElement::inplace_batch_inverse(&mut comp_denoms).unwrap();

        // Batch-invert all 2×num_queries×height trace denominators across all queries.
        // Currently each of the 146 calls to reconstruct_deep_composition_poly_evaluation
        // inverts its own 2-element denoms_trace (1 inversion per call = 146 total).
        // Collecting all 146×height values and inverting once reduces to 1 inversion.
        //
        // Layout: for each (iota, sym) pair interleaved, height rows:
        //   [ep_0−z, ep_0−z·g, ep_sym_0−z, ep_sym_0−z·g, ep_1−z, ep_1−z·g, ...]
        // Access: trace_denoms_inv[((2*i + sym_flag) * height) + row_idx]
        let ood_height = proof.trace_ood_evaluations().height();
        // OOD shift values: z·g^0, z·g^1, ..., z·g^(height-1), used as the
        // denominator bases for trace terms across all queries.
        let ood_z_shifts: Vec<FieldElement<FieldExtension>> = {
            let mut shifts = Vec::with_capacity(ood_height);
            let mut cur = challenges.z.clone();
            for _ in 0..ood_height {
                shifts.push(cur.clone());
                cur = primitive_root * &cur;
            }
            shifts
        };
        let mut trace_denoms_inv: Vec<FieldElement<FieldExtension>> =
            Vec::with_capacity(2 * num_queries * ood_height);
        for iota in challenges.iotas.iter() {
            let ep = Self::query_challenge_to_evaluation_point(*iota, domain).to_extension();
            let ep_sym =
                Self::query_challenge_to_evaluation_point_sym(*iota, domain).to_extension();
            for z_shift in ood_z_shifts.iter() {
                trace_denoms_inv.push(ep.clone() - z_shift);
            }
            for z_shift in ood_z_shifts.iter() {
                trace_denoms_inv.push(ep_sym.clone() - z_shift);
            }
        }
        FieldElement::inplace_batch_inverse(&mut trace_denoms_inv).unwrap();

        for (i, _iota) in challenges.iotas.iter().enumerate() {
            let opening = proof.deep_poly_opening(i);

            // Base-field columns (precomputed + main): kept as Field scalars for scalar_fma.
            evals_base.clear();
            if let Some(precomputed_polys) = &opening.precomputed_trace_polys {
                evals_base.extend_from_slice(precomputed_polys.evaluations);
            }
            evals_base.extend_from_slice(opening.main_trace_polys.evaluations);
            // Extension-field columns (aux trace): genuine Fp3 for regular fma.
            evals_ext.clear();
            if let Some(aux_trace_polys) = &opening.aux_trace_polys {
                evals_ext.extend_from_slice(aux_trace_polys.evaluations);
            }

            // trace_denoms_inv layout per query i: [ep_i row0..row(h-1), ep_sym_i row0..row(h-1)]
            let td_base = i * 2 * ood_height;
            deep_poly_evaluations.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                challenges,
                &evals_base,
                &evals_ext,
                opening.composition_poly.evaluations,
                &b_terms,
                &trace_denoms_inv[td_base..td_base + ood_height],
                comp_denoms[2 * i].clone(),
            ));

            // Symmetric point — same column split.
            evals_base_sym.clear();
            if let Some(precomputed_polys) = &opening.precomputed_trace_polys {
                evals_base_sym.extend_from_slice(precomputed_polys.evaluations_sym);
            }
            evals_base_sym.extend_from_slice(opening.main_trace_polys.evaluations_sym);
            evals_ext_sym.clear();
            if let Some(aux_trace_polys) = &opening.aux_trace_polys {
                evals_ext_sym.extend_from_slice(aux_trace_polys.evaluations_sym);
            }

            let td_sym_base = td_base + ood_height;
            deep_poly_evaluations_sym.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                challenges,
                &evals_base_sym,
                &evals_ext_sym,
                opening.composition_poly.evaluations_sym,
                &b_terms,
                &trace_denoms_inv[td_sym_base..td_sym_base + ood_height],
                comp_denoms[2 * i + 1].clone(),
            ));
        }
        (deep_poly_evaluations, deep_poly_evaluations_sym)
    }

    /// Precompute the query-invariant per-row term
    /// `b_terms[row] = Σ_col ood[row][col]·coeff[col][row]`, where `ood` is the
    /// committed trace OOD-evaluations table and `coeff` is the (flat,
    /// column-major) deep-composition trace coefficients. Neither depends on the
    /// FRI query, so this is computed once and reused for every query (and for the
    /// symmetric point) by [`reconstruct_deep_composition_poly_evaluation`].
    fn precompute_ood_coeff_terms<'p, P>(
        proof: &P,
        challenges: &Challenges<FieldExtension>,
    ) -> Vec<FieldElement<FieldExtension>>
    where
        P: StarkProofRef<'p, Field, FieldExtension, PI>,
        Field: 'p,
        FieldExtension: 'p,
        PI: 'p,
    {
        let ood = proof.trace_ood_evaluations();
        let height = ood.height();
        let width = ood.width();
        let trace_term_coeffs = &challenges.trace_term_coeffs;
        let chunk_len = challenges.trace_term_chunk_len;
        let mut b_terms = Vec::with_capacity(height);
        for row_idx in 0..height {
            let ood_row = ood.get_row(row_idx);
            let mut b = FieldElement::zero();
            for col_idx in 0..width {
                b.fma(&ood_row[col_idx], &trace_term_coeffs[col_idx * chunk_len + row_idx]);
            }
            b_terms.push(b);
        }
        b_terms
    }

    fn reconstruct_deep_composition_poly_evaluation<'p, P>(
        proof: &P,
        challenges: &Challenges<FieldExtension>,
        // Base-field (precomputed + main) trace evaluations as Field scalars.
        // Uses scalar_fma (Fp3ScalarFma ecall) — avoids to_extension() and Fp3 wrapper.
        lde_base_evaluations: &[FieldElement<Field>],
        // Extension-field (aux) trace evaluations as genuine Fp3 values.
        lde_ext_evaluations: &[FieldElement<FieldExtension>],
        lde_composition_poly_parts_evaluation: &[FieldElement<FieldExtension>],
        b_terms: &[FieldElement<FieldExtension>],
        // Pre-inverted trace denominators for this call's evaluation point, length = ood_height.
        // Batch-inverted by the caller across all queries (avoids 146 separate inversions).
        denoms_trace_inv: &[FieldElement<FieldExtension>],
        // Pre-inverted composition denominator: `(eval_point − z^N_parts)⁻¹`,
        // batch-computed by the caller across all queries (avoids 146 separate `.inv()` calls).
        denom_composition_inv: FieldElement<FieldExtension>,
    ) -> FieldElement<FieldExtension>
    where
        P: StarkProofRef<'p, Field, FieldExtension, PI>,
        Field: 'p,
        FieldExtension: 'p,
        PI: 'p,
    {
        let ood = proof.trace_ood_evaluations();
        let ood_evaluations_table_height = ood.height();
        let ood_evaluations_table_width = ood.width();
        let n_base_cols = lde_base_evaluations.len();
        let composition_poly_parts_ood = proof.composition_poly_parts_ood_evaluation();
        let trace_term_coeffs = &challenges.trace_term_coeffs;
        let trace_term_chunk_len = challenges.trace_term_chunk_len;
        debug_assert_eq!(
            ood_evaluations_table_height * ood_evaluations_table_width,
            trace_term_coeffs.len()
        );
        debug_assert_eq!(n_base_cols + lde_ext_evaluations.len(), ood_evaluations_table_width);
        // Each column's run has length `trace_term_chunk_len`, which equals the
        // number of OOD rows; the column-major index below relies on this.
        debug_assert_eq!(trace_term_chunk_len, ood_evaluations_table_height);
        debug_assert_eq!(b_terms.len(), ood_evaluations_table_height);
        debug_assert_eq!(denoms_trace_inv.len(), ood_evaluations_table_height);

        // Deep-trace term, with the query-invariant OOD·coeff half lifted out:
        //
        //   Σ_row denom[row] · Σ_col (lde[col] − ood[row][col])·coeff[col][row]
        // = Σ_row denom[row] · ( (Σ_col lde[col]·coeff[col][row]) − b_terms[row] )
        //
        // where `b_terms[row] = Σ_col ood[row][col]·coeff[col][row]` is precomputed
        // once across all queries (see `precompute_ood_coeff_terms`), and
        // `denom[row]` is pre-inverted by the caller via a single batch inversion
        // across all 2×num_queries×height trace denominators.
        // Fast path for the common OOD height=2 case: one pass through lde_trace_evaluations
        // serves both rows, halving the number of array loads vs two independent row loops.
        let mut trace_term = FieldElement::zero();
        if ood_evaluations_table_height == 2 {
            let (denom0, denom1) = (&denoms_trace_inv[0], &denoms_trace_inv[1]);
            let mut row_acc_0 = FieldElement::zero();
            let mut row_acc_1 = FieldElement::zero();
            // Base-field columns: scalar_fma (Fp3ScalarFma ecall) — 3 Goldilocks muls,
            // no Fp3 wrapper, no to_extension() copy.
            for col_idx in 0..n_base_cols {
                let base = col_idx * 2;
                let scalar = &lde_base_evaluations[col_idx];
                row_acc_0.scalar_fma::<Field>(scalar, &trace_term_coeffs[base]);
                row_acc_1.scalar_fma::<Field>(scalar, &trace_term_coeffs[base + 1]);
            }
            // Extension-field columns: Fp3 fma (Fp3Fma ecall).
            for (aux_idx, eval) in lde_ext_evaluations.iter().enumerate() {
                let col_idx = n_base_cols + aux_idx;
                let base = col_idx * 2;
                row_acc_0.fma(eval, &trace_term_coeffs[base]);
                row_acc_1.fma(eval, &trace_term_coeffs[base + 1]);
            }
            trace_term.fma(&(row_acc_0 - &b_terms[0]), denom0);
            trace_term.fma(&(row_acc_1 - &b_terms[1]), denom1);
        } else {
            for (row_idx, denom) in denoms_trace_inv.iter().enumerate() {
                let mut row_acc = FieldElement::zero();
                for col_idx in 0..n_base_cols {
                    row_acc.scalar_fma::<Field>(
                        &lde_base_evaluations[col_idx],
                        &trace_term_coeffs[col_idx * trace_term_chunk_len + row_idx],
                    );
                }
                for (aux_idx, eval) in lde_ext_evaluations.iter().enumerate() {
                    let col_idx = n_base_cols + aux_idx;
                    row_acc.fma(eval, &trace_term_coeffs[col_idx * trace_term_chunk_len + row_idx]);
                }
                trace_term.fma(&(row_acc - &b_terms[row_idx]), denom);
            }
        }

        let mut h_terms = FieldElement::zero();
        for (j, h_i_upsilon) in lde_composition_poly_parts_evaluation.iter().enumerate() {
            let h_i_zpower = &composition_poly_parts_ood[j];
            h_terms.fma(&(h_i_upsilon - h_i_zpower), &challenges.gammas[j]);
        }
        h_terms *= denom_composition_inv;

        trace_term + h_terms
    }

    /// Convenience wrapper over [`multi_verify`](Self::multi_verify) that takes an
    /// owned [`MultiProof`] (reads each sub-proof by reference). Equivalent to
    /// the generic form with `get_proof = |i| &multi_proof.proofs[i]`.
    fn multi_verify_owned(
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        multi_proof: &MultiProof<Field, FieldExtension, PI>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
        expected_bus_balance: &FieldElement<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    {
        Self::multi_verify(
            airs,
            multi_proof.proofs.len(),
            |i| &multi_proof.proofs[i],
            transcript,
            expected_bus_balance,
        )
    }

    /// Verifies one or more STARK proofs with their corresponding AIRs.
    ///
    /// # Multi-Table Verification with LogUp
    ///
    /// When verifying multiple tables that communicate via LogUp, the verifier
    /// must replay the transcript in the same order as the prover to derive
    /// identical challenges. This function ensures:
    ///
    /// 1. **Replay main trace commitments**: All commitments are appended to
    ///    the transcript in the same order as the prover.
    /// 2. **Sample shared LogUp challenges**: The same (z, α) challenges the
    ///    prover used are derived from the transcript.
    /// 3. **Replay auxiliary trace commitments**: Complete the Round 1 replay.
    /// 4. **Verify each proof**: Standard STARK verification for each AIR.
    ///
    /// # Warning
    ///
    /// The transcript must be safely initialized before passing it to this method.
    /// The AIRs must be in the same order as the proofs in the MultiProof.
    fn multi_verify<'p, P>(
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        num_proofs: usize,
        get_proof: impl Fn(usize) -> P,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
        expected_bus_balance: &FieldElement<FieldExtension>,
    ) -> bool
    where
        P: StarkProofRef<'p, Field, FieldExtension, PI>,
        Field: 'p,
        FieldExtension: 'p,
        PI: 'p,
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    {
        if airs.len() != num_proofs {
            error!(
                "AIR count ({}) does not match proof count ({})",
                airs.len(),
                num_proofs
            );
            return false;
        }

        // Check if any AIR has an auxiliary trace
        let needs_lookup_challenges = airs.iter().any(|air| air.has_aux_trace());

        // =====================================================================
        // Round 1, Phase A: Replay main trace commitments
        // =====================================================================
        // For preprocessed tables, use the hardcoded commitment (verifier cannot
        // trust the prover). For normal tables, use the commitment from the proof.

        for (idx, air) in airs.iter().enumerate() {
            let proof = get_proof(idx);
            if air.is_preprocessed() {
                // Preprocessed table: VERIFY precomputed commitment matches hardcoded.
                // This is the critical soundness check - ensures prover used correct precomputed values.
                let expected_precomputed = air.precomputed_commitment();
                match proof.lde_trace_precomputed_merkle_root() {
                    Some(actual) if *actual == expected_precomputed => {
                        // OK - commitment matches hardcoded
                    }
                    Some(actual) => {
                        error!(
                            "Preprocessed commitment MISMATCH for table {idx}: expected {:?}, got {:?}",
                            expected_precomputed, actual
                        );
                        return false;
                    }
                    None => {
                        error!("Preprocessed table {idx} proof missing precomputed commitment");
                        return false;
                    }
                }

                // Add BOTH commitments to transcript (Fiat-Shamir binding).
                // Precomputed commitment binds challenges to correct precomputed values.
                // Multiplicities commitment binds challenges to actual lookups made.
                transcript.append_bytes(&expected_precomputed);
                transcript.append_bytes(proof.lde_trace_main_merkle_root());
            } else {
                // Normal table: use commitment from proof
                transcript.append_bytes(proof.lde_trace_main_merkle_root());
            }
        }

        // =====================================================================
        // Round 1, Phase B: Sample shared LogUp challenges
        // =====================================================================
        // Must match exactly what the prover sampled.

        let lookup_challenges: Vec<FieldElement<FieldExtension>> = if needs_lookup_challenges {
            (0..LOGUP_NUM_CHALLENGES)
                .map(|_| transcript.sample_field_element())
                .collect()
        } else {
            Vec::new()
        };

        // =====================================================================
        // Validate bus_public_inputs presence against AIR layout
        // =====================================================================
        // A dishonest prover could omit bus_public_inputs entirely (None) to
        // bypass the bus balance check. With circular constraints, there are no
        // boundary constraints on LogUp columns, so the bus balance check is
        // the only cross-table validation.

        for (idx, air) in airs.iter().enumerate() {
            let proof = get_proof(idx);
            if air.has_trace_interaction() && !proof.has_bus_public_inputs() {
                error!(
                    "Table {idx}: AIR has LogUp interactions but proof is missing bus_public_inputs"
                );
                return false;
            }
            if !air.has_trace_interaction() && proof.has_bus_public_inputs() {
                error!(
                    "Table {idx}: AIR has no LogUp interactions but proof contains bus_public_inputs"
                );
                return false;
            }
        }

        // =====================================================================
        // Phase C + Rounds 2-4: Forked per table
        // =====================================================================
        // Each table gets an independent transcript fork (cloned from the shared
        // state after Phase B, domain-separated by table index). This matches
        // the prover's forking and makes per-table verification independent.

        // Scratch buffers reused across every table's step-2 evaluation. They are
        // resized (never shrunk) per table, so after the first table the backing
        // storage is reused with no further allocation.
        let mut verify_scratch = VerifyScratch::<FieldExtension>::new();

        for (idx, air) in airs.iter().enumerate() {
            let proof = get_proof(idx);
            // Must match prover: fork with domain separator for multi-table,
            // use original transcript directly for single-table.
            let num_tables = airs.len();
            let mut table_transcript = transcript.clone();
            if num_tables > 1 {
                table_transcript.append_bytes(&(idx as u64).to_le_bytes());
            }

            // Phase C: replay aux commitment
            if let Some(root) = proof.lde_trace_aux_merkle_root() {
                table_transcript.append_bytes(root);
            }

            // Bind table_contribution (L) to transcript, matching prover.
            if let Some(table_contribution) = proof.bus_table_contribution() {
                table_transcript.append_field_element(table_contribution);
            }

            // Rounds 2-4: verify
            if !Self::verify_rounds_2_to_4(
                *air,
                &proof,
                &mut table_transcript,
                lookup_challenges.clone(),
                &mut verify_scratch,
            ) {
                error!(
                    "Table {} failed verify_rounds_2_to_4 (num_constraints={}, trace_cols={})",
                    idx,
                    air.context().num_transition_constraints(),
                    air.context().trace_columns
                );
                return false;
            }
        }

        // =====================================================================
        // Bus Balance Check: Σ table_contribution = expected_bus_balance
        // =====================================================================
        // For LogUp with circular constraints, each table's total contribution L
        // (sum of all per-row terms) is exposed as a public input. The bus balances
        // when the sum of all table contributions equals the expected target.
        // When all bus participants are in-trace, the target is zero. When some
        // receiver contributions are computed externally (e.g. verifier-computed
        // COMMIT output bus), the target is the missing positive remainder.

        if needs_lookup_challenges {
            let mut total = FieldElement::<FieldExtension>::zero();
            for (idx, air) in airs.iter().enumerate() {
                let proof = get_proof(idx);
                if air.has_trace_interaction()
                    && let Some(table_contribution) = proof.bus_table_contribution()
                {
                    total = total + table_contribution;
                }
            }

            if total != *expected_bus_balance {
                #[cfg(not(feature = "test_fiat_shamir"))]
                error!(
                    "LogUp bus does not balance: sum of accumulated values does not match target. total={:?}, target={:?}",
                    total, expected_bus_balance
                );
                return false;
            }
            #[cfg(feature = "debug-checks")]
            info!("Bus balance check PASSED");
        }

        true
    }

    /// Verify a single STARK proof.
    /// This is equivalent to calling `multi_verify` with a single-element slice.
    fn verify(
        proof: &StarkProof<Field, FieldExtension, PI>,
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
    ) -> bool
    where
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
        PI: Clone,
    {
        Self::multi_verify(&[air], 1, |_| proof, transcript, &FieldElement::zero())
    }

    /// Replays rounds 2, 3 and 4 of the protocol for a given proof, assuming round 1 has
    /// already been replayed and the RAP challenges are known.
    fn replay_rounds_after_round_1<'p, P>(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &P,
        domain: &VerifierDomain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        rap_challenges: Vec<FieldElement<FieldExtension>>,
    ) -> Challenges<FieldExtension>
    where
        P: StarkProofRef<'p, Field, FieldExtension, PI>,
        Field: 'p,
        FieldExtension: 'p,
        PI: 'p,
        FieldElement<Field>: AsBytes + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion,
    {
        // ===================================
        // ==========|   Round 2   |==========
        // ===================================

        // <<<< Receive challenge: 𝛽
        let beta = transcript.sample_field_element();
        let trace_length = proof.trace_length();
        let bus_public_inputs = proof
            .bus_table_contribution()
            .map(|c| crate::lookup::BusPublicInputs::from_contribution(c.clone()));
        let num_boundary_constraints = air
            .boundary_constraints(
                proof.public_inputs(),
                &rap_challenges,
                bus_public_inputs.as_ref(),
                trace_length,
            )
            .constraints
            .len();

        let num_transition_constraints = air.context().num_transition_constraints;

        let mut coefficients =
            compute_alpha_powers(&beta, num_boundary_constraints + num_transition_constraints);

        let transition_coeffs: Vec<_> = coefficients.drain(..num_transition_constraints).collect();
        let boundary_coeffs = coefficients;

        // <<<< Receive commitments: [H₁], [H₂]
        transcript.append_bytes(proof.composition_poly_root());

        // ===================================
        // ==========|   Round 3   |==========
        // ===================================

        // >>>> Send challenge: z
        let z = transcript.sample_z_ood_with_domain_params(
            domain.trace_length,
            domain.lde_length,
            &domain.coset_offset,
        );

        // <<<< Receive values: tⱼ(zgᵏ)
        // Column-major append (matches `Table::columns()` order) without
        // materializing the transposed columns.
        let ood = proof.trace_ood_evaluations();
        for col_idx in 0..ood.width() {
            for row_idx in 0..ood.height() {
                transcript.append_field_element(&ood.get_row(row_idx)[col_idx]);
            }
        }
        // <<<< Receive value: Hᵢ(z^N)
        let composition_poly_parts_ood = proof.composition_poly_parts_ood_evaluation();
        for element in composition_poly_parts_ood.iter() {
            transcript.append_field_element(element);
        }

        // ===================================
        // ==========|   Round 4   |==========
        // ===================================

        let num_terms_composition_poly = composition_poly_parts_ood.len();
        let num_terms_trace =
            air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;
        let gamma = transcript.sample_field_element();

        // <<<< Receive challenges: 𝛾, 𝛾'
        let mut deep_composition_coefficients: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
                .take(num_terms_composition_poly + num_terms_trace)
                .collect();

        // Split the contiguous coefficient buffer: the trace terms are the first
        // `num_terms_trace` (kept flat, column-major with stride `chunk_len`), the
        // composition-poly gammas are the rest. `split_off(num_terms_trace)` hands
        // the suffix to `gammas` and leaves the (already contiguous) trace prefix
        // as `trace_term_coeffs` — no per-column `Vec` allocation, no copy.
        let chunk_len = air.context().transition_offsets.len() * air.step_size();
        // <<<< Receive challenges: 𝛾ⱼ, 𝛾ⱼ'
        let gammas = deep_composition_coefficients.split_off(num_terms_trace);
        let trace_term_coeffs = deep_composition_coefficients;
        let trace_term_chunk_len = chunk_len;

        // FRI commit phase
        let merkle_roots = proof.fri_layers_merkle_roots();
        let mut zetas = merkle_roots
            .iter()
            .map(|root| {
                // >>>> Send challenge 𝜁ₖ
                let element = transcript.sample_field_element();
                // <<<< Receive commitment: [pₖ] (the first one is [p₀])
                transcript.append_bytes(root);
                element
            })
            .collect::<Vec<FieldElement<FieldExtension>>>();

        // >>>> Send challenge 𝜁ₙ₋₁
        zetas.push(transcript.sample_field_element());

        // <<<< Receive value: pₙ
        transcript.append_field_element(proof.fri_last_value());

        // Receive grinding value
        let security_bits = air.context().proof_options.grinding_factor;
        let mut grinding_seed = [0u8; 32];
        if security_bits > 0
            && let Some(nonce_value) = proof.nonce()
        {
            grinding_seed = transcript.state();
            transcript.append_bytes(&nonce_value.to_be_bytes());
        }

        // FRI query phase
        // <<<< Send challenges 𝜄ₛ (iota_s)
        let number_of_queries = air.options().fri_number_of_queries;
        let iotas = Self::sample_query_indexes(number_of_queries, domain, transcript);

        Challenges {
            z,
            boundary_coeffs,
            transition_coeffs,
            trace_term_coeffs,
            trace_term_chunk_len,
            gammas,
            zetas,
            iotas,
            rap_challenges,
            grinding_seed,
        }
    }

    /// Verifies a single table after round 1 has been replayed.
    fn verify_rounds_2_to_4<'p, P>(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &P,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        rap_challenges: Vec<FieldElement<FieldExtension>>,
        scratch: &mut VerifyScratch<FieldExtension>,
    ) -> bool
    where
        P: StarkProofRef<'p, Field, FieldExtension, PI>,
        Field: 'p,
        FieldExtension: 'p,
        PI: 'p,
        FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    {
        let domain = new_verifier_domain(air, proof.trace_length());

        // Verify there are enough queries
        if proof.query_list_len() < air.options().fri_number_of_queries {
            return false;
        }

        #[cfg(feature = "instruments")]
        println!("- Started step 1: Recover challenges");
        #[cfg(feature = "instruments")]
        let timer1 = Instant::now();

        let challenges =
            Self::replay_rounds_after_round_1(air, proof, &domain, transcript, rap_challenges);

        // verify grinding
        let security_bits = air.context().proof_options.grinding_factor;
        if security_bits > 0 {
            let nonce_is_valid = proof.nonce().is_some_and(|nonce_value| {
                grinding::is_valid_nonce(&challenges.grinding_seed, nonce_value, security_bits)
            });

            if !nonce_is_valid {
                #[cfg(not(feature = "test_fiat_shamir"))]
                error!("Grinding factor not satisfied");
                return false;
            }
        }

        #[cfg(feature = "instruments")]
        let elapsed1 = timer1.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed1);

        #[cfg(feature = "instruments")]
        println!("- Started step 2: Verify claimed polynomial");
        #[cfg(feature = "instruments")]
        let timer2 = Instant::now();

        if !Self::step_2_verify_claimed_composition_polynomial(
            air,
            proof,
            &domain,
            &challenges,
            scratch,
        ) {
            #[cfg(not(feature = "test_fiat_shamir"))]
            error!("Composition Polynomial verification failed");
            return false;
        }

        #[cfg(feature = "instruments")]
        let elapsed2 = timer2.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed2);
        #[cfg(feature = "instruments")]
        println!("- Started step 3: Verify FRI");
        #[cfg(feature = "instruments")]
        let timer3 = Instant::now();

        if !Self::step_3_verify_fri(proof, &domain, &challenges) {
            #[cfg(not(feature = "test_fiat_shamir"))]
            error!("FRI verification failed");
            return false;
        }

        #[cfg(feature = "instruments")]
        let elapsed3 = timer3.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed3);

        #[cfg(feature = "instruments")]
        println!("- Started step 4: Verify deep composition polynomial");
        #[cfg(feature = "instruments")]
        let timer4 = Instant::now();

        #[allow(clippy::let_and_return)]
        if !Self::step_4_verify_trace_and_composition_openings(proof, &challenges) {
            #[cfg(not(feature = "test_fiat_shamir"))]
            error!("DEEP Composition Polynomial verification failed");
            return false;
        }

        #[cfg(feature = "instruments")]
        let elapsed4 = timer4.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed4);

        #[cfg(feature = "instruments")]
        {
            let total_time = elapsed1 + elapsed2 + elapsed3 + elapsed4;
            println!(
                " Fraction of verifying time per step: {:.4} {:.4} {:.4} {:.4}",
                elapsed1.as_nanos() as f64 / total_time.as_nanos() as f64,
                elapsed2.as_nanos() as f64 / total_time.as_nanos() as f64,
                elapsed3.as_nanos() as f64 / total_time.as_nanos() as f64,
                elapsed4.as_nanos() as f64 / total_time.as_nanos() as f64
            );
        }

        true
    }
}
