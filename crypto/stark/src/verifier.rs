use super::{
    config::BatchedMerkleTreeBackend,
    domain::VerifierDomain,
    fri::fri_decommit::FriDecommitment,
    grinding,
    proof::stark::StarkProof,
    traits::{AIR, TransitionEvaluationContext},
};
use crate::{
    config::Commitment,
    domain::new_verifier_domain,
    lookup::{LOGUP_CHALLENGE_ALPHA, LOGUP_NUM_CHALLENGES, PackingShifts, compute_alpha_powers},
    proof::stark::{DeepPolynomialOpening, MultiProof},
};
use crypto::{fiat_shamir::is_transcript::IsStarkTranscript, merkle_tree::proof::Proof};
#[cfg(not(feature = "test_fiat_shamir"))]
use log::error;
#[cfg(feature = "debug-checks")]
use log::info;
use math::{
    fft::cpu::bit_reversing::reverse_index,
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
    traits::AsBytes,
};
use std::marker::PhantomData;
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
    /// The deep composition polynomial coefficients corresponding to the trace polynomial terms.
    pub trace_term_coeffs: Vec<Vec<FieldElement<FieldExtension>>>,
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
    fn step_1_replay_rounds_and_recover_challenges(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &StarkProof<Field, FieldExtension, PI>,
        domain: &VerifierDomain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Challenges<FieldExtension>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        // ===================================
        // ==========|   Round 1   |==========
        // ===================================

        // <<<< Receive commitments:[tⱼ]
        transcript.append_bytes(&proof.lde_trace_main_merkle_root);

        let rap_challenges = air.build_rap_challenges(transcript);

        if let Some(root) = proof.lde_trace_aux_merkle_root {
            transcript.append_bytes(&root);
        }

        // ===================================
        // ==========|   Round 2   |==========
        // ===================================

        // <<<< Receive challenge: 𝛽
        let beta = transcript.sample_field_element();
        let trace_length = proof.trace_length;
        let num_boundary_constraints = air
            .boundary_constraints(
                &proof.public_inputs,
                &rap_challenges,
                proof.bus_public_inputs.as_ref(),
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
        transcript.append_bytes(&proof.composition_poly_root);

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
        let trace_ood_evaluations_columns = proof.trace_ood_evaluations.columns();
        for col in trace_ood_evaluations_columns.iter() {
            for elem in col.iter() {
                transcript.append_field_element(elem);
            }
        }
        // <<<< Receive value: Hᵢ(z^N)
        for element in proof.composition_poly_parts_ood_evaluation.iter() {
            transcript.append_field_element(element);
        }

        // ===================================
        // ==========|   Round 4   |==========
        // ===================================

        let num_terms_composition_poly = proof.composition_poly_parts_ood_evaluation.len();
        let num_terms_trace =
            air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;
        let gamma = transcript.sample_field_element();

        // <<<< Receive challenges: 𝛾, 𝛾'
        let mut deep_composition_coefficients: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
                .take(num_terms_composition_poly + num_terms_trace)
                .collect();

        let trace_term_coeffs: Vec<_> = deep_composition_coefficients
            .drain(..num_terms_trace)
            .collect::<Vec<_>>()
            .chunks(air.context().transition_offsets.len() * air.step_size())
            .map(|chunk| chunk.to_vec())
            .collect();

        // <<<< Receive challenges: 𝛾ⱼ, 𝛾ⱼ'
        let gammas = deep_composition_coefficients;

        // FRI commit phase
        let merkle_roots = &proof.fri_layers_merkle_roots;
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
        transcript.append_field_element(&proof.fri_last_value);

        // Receive grinding value
        let security_bits = air.context().proof_options.grinding_factor;
        let mut grinding_seed = [0u8; 32];
        if security_bits > 0
            && let Some(nonce_value) = proof.nonce
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
    fn step_2_verify_claimed_composition_polynomial(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &StarkProof<Field, FieldExtension, PI>,
        domain: &VerifierDomain<Field>,
        challenges: &Challenges<FieldExtension>,
    ) -> bool {
        let trace_length = proof.trace_length;
        let boundary_constraints = air.boundary_constraints(
            &proof.public_inputs,
            &challenges.rap_challenges,
            proof.bus_public_inputs.as_ref(),
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
                    &proof.trace_ood_evaluations.get_row(0)[column_idx]
                } else {
                    &proof.trace_ood_evaluations.get_row(0)[column_idx]
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

        let boundary_quotient_ood_evaluation: FieldElement<FieldExtension> =
            boundary_c_i_evaluations_num
                .iter()
                .zip(&boundary_c_i_evaluations_den)
                .zip(&challenges.boundary_coeffs)
                .map(|((num, den), beta)| num * den * beta)
                .fold(FieldElement::<FieldExtension>::zero(), |acc, x| acc + x);

        let periodic_values = air
            .get_periodic_column_polynomials(trace_length)
            .iter()
            .map(|poly| poly.evaluate(&challenges.z))
            .collect::<Vec<FieldElement<FieldExtension>>>();

        let num_main_trace_columns =
            proof.trace_ood_evaluations.width - air.num_auxiliary_rap_columns();

        let logup_alpha_powers: Vec<FieldElement<FieldExtension>> =
            if challenges.rap_challenges.len() > LOGUP_CHALLENGE_ALPHA {
                compute_alpha_powers(
                    &challenges.rap_challenges[LOGUP_CHALLENGE_ALPHA],
                    air.max_bus_elements(),
                )
            } else {
                Vec::new()
            };

        let logup_table_offset = match &proof.bus_public_inputs {
            Some(bpi) => {
                let n = FieldElement::<Field>::from(trace_length as u64);
                match n.inv() {
                    Ok(n_inv) => n_inv * &bpi.table_contribution,
                    Err(_) => return false, // trace_length == 0 is invalid
                }
            }
            None => FieldElement::zero(),
        };

        let ood_frame =
            (proof.trace_ood_evaluations).into_frame(num_main_trace_columns, air.step_size());
        let packing_shifts = PackingShifts::<FieldExtension>::new();
        let transition_evaluation_context = TransitionEvaluationContext::new_verifier(
            &ood_frame,
            &periodic_values,
            &challenges.rap_challenges,
            &logup_alpha_powers,
            &logup_table_offset,
            &packing_shifts,
        );
        let transition_ood_frame_evaluations =
            air.compute_transition(&transition_evaluation_context);

        let mut denominators =
            vec![FieldElement::<FieldExtension>::zero(); air.num_transition_constraints()];
        air.transition_constraints().iter().for_each(|c| {
            denominators[c.constraint_idx()] =
                c.evaluate_zerofier(&challenges.z, &domain.trace_primitive_root, trace_length);
        });

        let transition_c_i_evaluations_sum = itertools::izip!(
            transition_ood_frame_evaluations,
            &challenges.transition_coeffs,
            denominators
        )
        .fold(FieldElement::zero(), |acc, (eval, beta, denominator)| {
            acc + beta * eval * &denominator
        });

        let composition_poly_ood_evaluation =
            &boundary_quotient_ood_evaluation + transition_c_i_evaluations_sum;

        let composition_poly_claimed_ood_evaluation = proof
            .composition_poly_parts_ood_evaluation
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
    fn step_3_verify_fri(
        proof: &StarkProof<Field, FieldExtension, PI>,
        domain: &VerifierDomain<Field>,
        challenges: &Challenges<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
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

        proof
            .query_list
            .iter()
            .zip(&challenges.iotas)
            .zip(evaluation_point_inverse)
            .enumerate()
            .fold(true, |mut result, (i, ((proof_s, iota_s), eval))| {
                result &= Self::verify_query_and_sym_openings(
                    proof,
                    &challenges.zetas,
                    *iota_s,
                    proof_s,
                    eval,
                    &deep_poly_evaluations[i],
                    &deep_poly_evaluations_sym[i],
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

    /// Verifies the validity of the opening proof.
    fn verify_opening<E>(
        proof: &Proof<Commitment>,
        root: &Commitment,
        index: usize,
        value: &[FieldElement<E>],
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<E>: AsBytes + Sync + Send,
        E: IsField,
        Field: IsSubFieldOf<E>,
    {
        proof.verify::<BatchedMerkleTreeBackend<E>>(root, index, &value.to_owned())
    }

    /// Verify opening Open(tⱼ(D_LDE), 𝜐) and Open(tⱼ(D_LDE), -𝜐) for all trace polynomials tⱼ,
    /// where 𝜐 and -𝜐 are the elements corresponding to the index challenge `iota`.
    fn verify_trace_openings(
        proof: &StarkProof<Field, FieldExtension, PI>,
        deep_poly_openings: &DeepPolynomialOpening<Field, FieldExtension>,
        iota: usize,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let index = iota * 2;
        let index_sym = iota * 2 + 1;
        let mut result = true;

        // Verify main trace (multiplicities for preprocessed, full trace for normal)
        result &= Self::verify_opening::<Field>(
            &deep_poly_openings.main_trace_polys.proof,
            &proof.lde_trace_main_merkle_root,
            index,
            &deep_poly_openings.main_trace_polys.evaluations,
        );
        result &= Self::verify_opening::<Field>(
            &deep_poly_openings.main_trace_polys.proof_sym,
            &proof.lde_trace_main_merkle_root,
            index_sym,
            &deep_poly_openings.main_trace_polys.evaluations_sym,
        );

        // Verify precomputed trace (for preprocessed tables only)
        match (
            &proof.lde_trace_precomputed_merkle_root,
            &deep_poly_openings.precomputed_trace_polys,
        ) {
            // Unreachable: multi_verify() already rejected proofs with None root for preprocessed AIRs,
            // and non-preprocessed AIRs never have openings. No valid execution path reaches here.
            (None, Some(_)) => result = false,
            (Some(_), None) => result = false,
            (Some(precomputed_root), Some(precomputed_opening)) => {
                result &= Self::verify_opening::<Field>(
                    &precomputed_opening.proof,
                    precomputed_root,
                    index,
                    &precomputed_opening.evaluations,
                );
                result &= Self::verify_opening::<Field>(
                    &precomputed_opening.proof_sym,
                    precomputed_root,
                    index_sym,
                    &precomputed_opening.evaluations_sym,
                );
            }
            _ => {}
        }

        // Verify auxiliary trace
        match (
            proof.lde_trace_aux_merkle_root,
            &deep_poly_openings.aux_trace_polys,
        ) {
            (None, Some(_)) => result = false,
            (Some(_), None) => result = false,
            (Some(aux_root), Some(aux_trace_polys_opening)) => {
                result &= Self::verify_opening::<FieldExtension>(
                    &aux_trace_polys_opening.proof,
                    &aux_root,
                    index,
                    &aux_trace_polys_opening.evaluations,
                );
                result &= Self::verify_opening::<FieldExtension>(
                    &aux_trace_polys_opening.proof_sym,
                    &aux_root,
                    index_sym,
                    &aux_trace_polys_opening.evaluations_sym,
                );
            }
            _ => {}
        }

        result
    }

    /// Verify opening Open(Hᵢ(D_LDE), 𝜐) and Open(Hᵢ(D_LDE), -𝜐) for all parts Hᵢof the composition
    /// polynomial, where 𝜐 and -𝜐 are the elements corresponding to the index challenge `iota`.
    fn verify_composition_poly_opening(
        deep_poly_openings: &DeepPolynomialOpening<Field, FieldExtension>,
        composition_poly_merkle_root: &Commitment,
        iota: &usize,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let mut value = deep_poly_openings.composition_poly.evaluations.clone();
        value.extend_from_slice(&deep_poly_openings.composition_poly.evaluations_sym);

        deep_poly_openings
            .composition_poly
            .proof
            .verify::<BatchedMerkleTreeBackend<FieldExtension>>(
                composition_poly_merkle_root,
                *iota,
                &value,
            )
    }

    /// Verifies the validity of the purported values of the trace polynomials and the composition polynomial
    /// parts at the domain elements and their symmetric counterparts corresponding to all the FRI query
    /// index challenges.
    fn step_4_verify_trace_and_composition_openings(
        proof: &StarkProof<Field, FieldExtension, PI>,
        challenges: &Challenges<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        challenges.iotas.iter().zip(&proof.deep_poly_openings).fold(
            true,
            |mut result, (iota_n, deep_poly_opening)| {
                result &= Self::verify_composition_poly_opening(
                    deep_poly_opening,
                    &proof.composition_poly_root,
                    iota_n,
                );

                result &= Self::verify_trace_openings(proof, deep_poly_opening, *iota_n);
                result
            },
        )
    }

    /// Verifies the openings of a fold polynomial of an inner layer of FRI.
    fn verify_fri_layer_openings(
        merkle_root: &Commitment,
        auth_path_sym: &Proof<Commitment>,
        evaluation: &FieldElement<FieldExtension>,
        evaluation_sym: &FieldElement<FieldExtension>,
        iota: usize,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let evaluations = if iota % 2 == 1 {
            vec![evaluation_sym.clone(), evaluation.clone()]
        } else {
            vec![evaluation.clone(), evaluation_sym.clone()]
        };

        auth_path_sym.verify::<BatchedMerkleTreeBackend<FieldExtension>>(
            merkle_root,
            iota >> 1,
            &evaluations,
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
    fn verify_query_and_sym_openings(
        proof: &StarkProof<Field, FieldExtension, PI>,
        zetas: &[FieldElement<FieldExtension>],
        iota: usize,
        fri_decommitment: &FriDecommitment<FieldExtension>,
        evaluation_point_inv: FieldElement<Field>,
        deep_composition_evaluation: &FieldElement<FieldExtension>,
        deep_composition_evaluation_sym: &FieldElement<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let fri_layers_merkle_roots = &proof.fri_layers_merkle_roots;
        let evaluation_point_vec: Vec<FieldElement<Field>> =
            core::iter::successors(Some(evaluation_point_inv.square()), |evaluation_point| {
                Some(evaluation_point.square())
            })
            .take(fri_layers_merkle_roots.len())
            .collect();

        let p0_eval = deep_composition_evaluation;
        let p0_eval_sym = deep_composition_evaluation_sym;

        // Reconstruct p₁(𝜐²)
        let mut v =
            (p0_eval + p0_eval_sym) + evaluation_point_inv * &zetas[0] * (p0_eval - p0_eval_sym);
        let mut index = iota;

        // Handle case with 0 FRI layers (trace_length <= 2)
        // In this case, the fold loop below doesn't iterate, so we need to verify
        // the final value directly here.
        if fri_layers_merkle_roots.is_empty() {
            return v == proof.fri_last_value;
        }

        // For each FRI layer, starting from the layer 1: use the proof to verify the validity of values pᵢ(−𝜐^(2ⁱ)) (given by the prover) and
        // pᵢ(𝜐^(2ⁱ)) (computed on the previous iteration by the verifier). Then use them to obtain pᵢ₊₁(𝜐^(2ⁱ⁺¹)).
        // Finally, check that the final value coincides with the given by the prover.
        fri_layers_merkle_roots
            .iter()
            .enumerate()
            .zip(&fri_decommitment.layers_auth_paths)
            .zip(&fri_decommitment.layers_evaluations_sym)
            .zip(evaluation_point_vec)
            .fold(
                true,
                |result,
                 (
                    (((i, merkle_root), auth_path_sym), evaluation_sym),
                    evaluation_point_inv,
                )| {
                    // Verify opening Open(pᵢ(Dₖ), −𝜐^(2ⁱ)) and Open(pᵢ(Dₖ), 𝜐^(2ⁱ)).
                    // `v` is pᵢ(𝜐^(2ⁱ)).
                    // `evaluation_sym` is pᵢ(−𝜐^(2ⁱ)).
                    let openings_ok = Self::verify_fri_layer_openings(
                        merkle_root,
                        auth_path_sym,
                        &v,
                        evaluation_sym,
                        index,
                    );

                    // Update `v` with next value pᵢ₊₁(𝜐^(2ⁱ⁺¹)).
                    v = (&v + evaluation_sym) + evaluation_point_inv * &zetas[i + 1] * (&v - evaluation_sym);

                    // Update index for next iteration. The index of the squares in the next layer
                    // is obtained by halving the current index. This is due to the bit-reverse
                    // ordering of the elements in the Merkle tree.
                    index >>= 1;

                    if i < fri_decommitment.layers_evaluations_sym.len() - 1 {
                        result & openings_ok
                    } else {
                        // Check that final value is the given by the prover
                        result & (v == proof.fri_last_value) & openings_ok
                    }
                },
            )
    }

    fn reconstruct_deep_composition_poly_evaluations_for_all_queries(
        challenges: &Challenges<FieldExtension>,
        domain: &VerifierDomain<Field>,
        proof: &StarkProof<Field, FieldExtension, PI>,
    ) -> DeepPolynomialEvaluations<FieldExtension> {
        let num_queries = challenges.iotas.len();
        let mut deep_poly_evaluations = Vec::with_capacity(num_queries);
        let mut deep_poly_evaluations_sym = Vec::with_capacity(num_queries);
        for (i, iota) in challenges.iotas.iter().enumerate() {
            let primitive_root =
                &Field::get_primitive_root_of_unity(domain.root_order as u64).unwrap();

            // For preprocessed tables: precomputed columns come FIRST, then multiplicities
            let mut evaluations: Vec<FieldElement<FieldExtension>> = Vec::new();
            if let Some(precomputed_polys) = &proof.deep_poly_openings[i].precomputed_trace_polys {
                evaluations.extend(
                    precomputed_polys
                        .evaluations
                        .iter()
                        .cloned()
                        .map(|x| x.to_extension()),
                );
            }
            evaluations.extend(
                proof.deep_poly_openings[i]
                    .main_trace_polys
                    .evaluations
                    .iter()
                    .cloned()
                    .map(|x| x.to_extension()),
            );
            if let Some(aux_trace_polys) = &proof.deep_poly_openings[i].aux_trace_polys {
                evaluations.extend_from_slice(&aux_trace_polys.evaluations);
            }

            let evaluation_point = Self::query_challenge_to_evaluation_point(*iota, domain);
            deep_poly_evaluations.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                &evaluation_point,
                primitive_root,
                challenges,
                &evaluations,
                &proof.deep_poly_openings[i].composition_poly.evaluations,
            ));

            // For preprocessed tables: precomputed columns come FIRST, then multiplicities
            let mut evaluations_sym: Vec<FieldElement<FieldExtension>> = Vec::new();
            if let Some(precomputed_polys) = &proof.deep_poly_openings[i].precomputed_trace_polys {
                evaluations_sym.extend(
                    precomputed_polys
                        .evaluations_sym
                        .iter()
                        .cloned()
                        .map(|x| x.to_extension()),
                );
            }
            evaluations_sym.extend(
                proof.deep_poly_openings[i]
                    .main_trace_polys
                    .evaluations_sym
                    .iter()
                    .cloned()
                    .map(|x| x.to_extension()),
            );
            if let Some(aux_trace_polys) = &proof.deep_poly_openings[i].aux_trace_polys {
                evaluations_sym.extend_from_slice(&aux_trace_polys.evaluations_sym);
            }

            let evaluation_point = Self::query_challenge_to_evaluation_point_sym(*iota, domain);
            deep_poly_evaluations_sym.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                &evaluation_point,
                primitive_root,
                challenges,
                &evaluations_sym,
                &proof.deep_poly_openings[i].composition_poly.evaluations_sym,
            ));
        }
        (deep_poly_evaluations, deep_poly_evaluations_sym)
    }

    fn reconstruct_deep_composition_poly_evaluation(
        proof: &StarkProof<Field, FieldExtension, PI>,
        evaluation_point: &FieldElement<Field>,
        primitive_root: &FieldElement<Field>,
        challenges: &Challenges<FieldExtension>,
        lde_trace_evaluations: &[FieldElement<FieldExtension>],
        lde_composition_poly_parts_evaluation: &[FieldElement<FieldExtension>],
    ) -> FieldElement<FieldExtension> {
        let ood_evaluations_table_height = proof.trace_ood_evaluations.height;
        let ood_evaluations_table_width = proof.trace_ood_evaluations.width;
        let trace_term_coeffs = &challenges.trace_term_coeffs;
        debug_assert_eq!(
            ood_evaluations_table_height * ood_evaluations_table_width,
            trace_term_coeffs.len() * trace_term_coeffs[0].len()
        );

        let mut denoms_trace = Vec::with_capacity(ood_evaluations_table_height);
        let mut current_z = challenges.z.clone();
        for _ in 0..ood_evaluations_table_height {
            denoms_trace.push(evaluation_point - &current_z);
            current_z = primitive_root * &current_z;
        }
        FieldElement::inplace_batch_inverse(&mut denoms_trace).unwrap();

        let trace_term = (0..ood_evaluations_table_width)
            .zip(&challenges.trace_term_coeffs)
            .fold(FieldElement::zero(), |trace_terms, (col_idx, coeff_row)| {
                let trace_i = (0..ood_evaluations_table_height).zip(coeff_row).fold(
                    FieldElement::zero(),
                    |trace_t, (row_idx, coeff)| {
                        let poly_evaluation = (lde_trace_evaluations[col_idx].clone()
                            - proof.trace_ood_evaluations.get_row(row_idx)[col_idx].clone())
                            * &denoms_trace[row_idx];
                        trace_t + &poly_evaluation * coeff
                    },
                );
                trace_terms + trace_i
            });

        let number_of_parts = lde_composition_poly_parts_evaluation.len();
        let z_pow = &challenges.z.pow(number_of_parts);

        let denom_composition = (evaluation_point - z_pow).inv().unwrap();
        let mut h_terms = FieldElement::zero();
        for (j, h_i_upsilon) in lde_composition_poly_parts_evaluation.iter().enumerate() {
            let h_i_zpower = &proof.composition_poly_parts_ood_evaluation[j];
            let h_i_term = (h_i_upsilon - h_i_zpower) * &challenges.gammas[j];
            h_terms += h_i_term;
        }
        h_terms *= denom_composition;

        trace_term + h_terms
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
    fn multi_verify(
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        multi_proof: &MultiProof<Field, FieldExtension, PI>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
        expected_bus_balance: &FieldElement<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        // If unified_proof is present, use hybrid verification (requires PI: Default)
        // The caller should use multi_verify_hybrid directly when PI is not Default.

        if airs.len() != multi_proof.proofs.len() {
            error!(
                "AIR count ({}) does not match proof count ({})",
                airs.len(),
                multi_proof.proofs.len()
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

        for (idx, (air, proof)) in airs.iter().zip(&multi_proof.proofs).enumerate() {
            if air.is_preprocessed() {
                // Preprocessed table: VERIFY precomputed commitment matches hardcoded.
                // This is the critical soundness check - ensures prover used correct precomputed values.
                let expected_precomputed = air.precomputed_commitment();
                match &proof.lde_trace_precomputed_merkle_root {
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
                transcript.append_bytes(&proof.lde_trace_main_merkle_root);
            } else {
                // Normal table: use commitment from proof
                transcript.append_bytes(&proof.lde_trace_main_merkle_root);
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

        for (idx, (air, proof)) in airs.iter().zip(&multi_proof.proofs).enumerate() {
            if air.has_trace_interaction() && proof.bus_public_inputs.is_none() {
                error!(
                    "Table {idx}: AIR has LogUp interactions but proof is missing bus_public_inputs"
                );
                return false;
            }
            if !air.has_trace_interaction() && proof.bus_public_inputs.is_some() {
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

        for (idx, (air, proof)) in airs.iter().zip(&multi_proof.proofs).enumerate() {
            // Must match prover: fork with domain separator for multi-table,
            // use original transcript directly for single-table.
            let num_tables = airs.len();
            let mut table_transcript = transcript.clone();
            if num_tables > 1 {
                table_transcript.append_bytes(&(idx as u64).to_le_bytes());
            }

            // Phase C: replay aux commitment
            if let Some(root) = proof.lde_trace_aux_merkle_root {
                table_transcript.append_bytes(&root);
            }

            // Bind table_contribution (L) to transcript, matching prover.
            if let Some(ref bpi) = proof.bus_public_inputs {
                table_transcript.append_field_element(&bpi.table_contribution);
            }

            // Rounds 2-4: verify
            if !Self::verify_rounds_2_to_4(
                *air,
                proof,
                &mut table_transcript,
                lookup_challenges.clone(),
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
            for (air, proof) in airs.iter().zip(&multi_proof.proofs) {
                if air.has_trace_interaction()
                    && let Some(interaction) = &proof.bus_public_inputs
                {
                    total = total + &interaction.table_contribution;
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

    /// Verify a hybrid multi-proof (unified batched + individual per-table proofs).
    ///
    /// Replays the same transcript as `multi_prove_hybrid`:
    /// 1. Phase A: individual table roots + batched main root
    /// 2. Phase B: sample LogUp challenges
    /// 3. Fork: individual tables verified per-table, batched via unified path
    /// 4. Bus balance check across all tables
    fn multi_verify_hybrid(
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        multi_proof: &MultiProof<Field, FieldExtension, PI>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
        expected_bus_balance: &FieldElement<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
        PI: Default,
    {
        let unified_proof = match &multi_proof.unified_proof {
            Some(p) => p,
            None => return false,
        };
        let unified_indices = &multi_proof.unified_indices;

        let num_airs = airs.len();
        let needs_lookup = airs.iter().any(|a| a.has_aux_trace());

        // Build sets for classification
        let is_unified: Vec<bool> = (0..num_airs)
            .map(|i| unified_indices.contains(&i))
            .collect();

        // Verify proof counts match
        let expected_individual = is_unified.iter().filter(|&&u| !u).count();
        if multi_proof.proofs.len() != expected_individual {
            #[cfg(not(feature = "test_fiat_shamir"))]
            error!(
                "Individual proof count ({}) doesn't match expected ({})",
                multi_proof.proofs.len(),
                expected_individual
            );
            return false;
        }
        if unified_proof.table_data.len() != unified_indices.len() {
            return false;
        }

        // =================================================================
        // Phase A: Replay main trace commitments (same order as prover)
        // =================================================================
        // Individual tables first, then batched root

        for &idx in (0..num_airs).collect::<Vec<_>>().iter() {
            if is_unified[idx] {
                continue; // batched tables committed together below
            }
            let air = airs[idx];
            // Find this table's proof in the individual proofs list
            let individual_pos = (0..idx).filter(|&i| !is_unified[i]).count();
            let proof = &multi_proof.proofs[individual_pos];

            if air.is_preprocessed() {
                let expected = air.precomputed_commitment();
                match &proof.lde_trace_precomputed_merkle_root {
                    Some(actual) if *actual == expected => {}
                    _ => {
                        #[cfg(not(feature = "test_fiat_shamir"))]
                        error!("Table {idx}: preprocessed commitment mismatch");
                        return false;
                    }
                }
                transcript.append_bytes(&expected);
                transcript.append_bytes(&proof.lde_trace_main_merkle_root);
            } else {
                transcript.append_bytes(&proof.lde_trace_main_merkle_root);
            }
        }
        // Batched main root
        transcript.append_bytes(&unified_proof.main_trace_root);

        // =================================================================
        // Phase B: Sample shared LogUp challenges
        // =================================================================

        let lookup_challenges: Vec<FieldElement<FieldExtension>> = if needs_lookup {
            (0..LOGUP_NUM_CHALLENGES)
                .map(|_| transcript.sample_field_element())
                .collect()
        } else {
            Vec::new()
        };

        // =================================================================
        // Validate bus_public_inputs
        // =================================================================

        for (idx, air) in airs.iter().enumerate() {
            if is_unified[idx] {
                let pos = unified_indices.iter().position(|&i| i == idx).unwrap();
                if air.has_trace_interaction()
                    && unified_proof.table_data[pos].bus_public_inputs.is_none()
                {
                    return false;
                }
            } else {
                let individual_pos = (0..idx).filter(|&i| !is_unified[i]).count();
                let proof = &multi_proof.proofs[individual_pos];
                if air.has_trace_interaction() && proof.bus_public_inputs.is_none() {
                    return false;
                }
            }
        }

        // =================================================================
        // Individual tables: fork + verify per-table
        // =================================================================

        let mut individual_proof_idx = 0;
        for idx in 0..num_airs {
            if is_unified[idx] {
                continue;
            }
            let air = airs[idx];
            let proof = &multi_proof.proofs[individual_proof_idx];
            individual_proof_idx += 1;

            let mut table_transcript = transcript.clone();
            table_transcript.append_bytes(&(idx as u64).to_le_bytes());

            if let Some(root) = proof.lde_trace_aux_merkle_root {
                table_transcript.append_bytes(&root);
            }
            if let Some(ref bpi) = proof.bus_public_inputs {
                table_transcript.append_field_element(&bpi.table_contribution);
            }

            if !Self::verify_rounds_2_to_4(
                air,
                proof,
                &mut table_transcript,
                lookup_challenges.clone(),
            ) {
                #[cfg(not(feature = "test_fiat_shamir"))]
                error!("Table {idx} (individual) failed verification");
                return false;
            }
        }

        // =================================================================
        // Batched tables: unified verification on forked transcript
        // =================================================================

        let batched_airs: Vec<&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>> =
            unified_indices.iter().map(|&i| airs[i]).collect();

        let mut batched_transcript = transcript.clone();
        batched_transcript.append_bytes(&(num_airs as u64).to_le_bytes());

        // Compute partial bus balance for the batched group
        let batched_bus_balance: FieldElement<FieldExtension> = unified_proof
            .table_data
            .iter()
            .filter_map(|td| td.bus_public_inputs.as_ref())
            .map(|bpi| &bpi.table_contribution)
            .fold(FieldElement::zero(), |acc, c| acc + c);

        // Verify the unified proof for batched tables.
        // multi_verify_unified takes &[&PI]; construct from Default trait.
        // For the VM (PI=()), this is always valid.
        let default_pi = PI::default();
        let pi_refs: Vec<&PI> = vec![&default_pi; batched_airs.len()];

        if !Self::multi_verify_unified(
            &batched_airs,
            &pi_refs,
            unified_proof,
            &mut batched_transcript,
            &batched_bus_balance,
        ) {
            #[cfg(not(feature = "test_fiat_shamir"))]
            error!("Unified batched proof verification failed");
            return false;
        }

        // =================================================================
        // Global bus balance check
        // =================================================================

        if needs_lookup {
            let mut total = FieldElement::<FieldExtension>::zero();
            // Individual tables
            for proof in &multi_proof.proofs {
                if let Some(ref bpi) = proof.bus_public_inputs {
                    total = total + &bpi.table_contribution;
                }
            }
            // Batched tables
            for td in &unified_proof.table_data {
                if let Some(ref bpi) = td.bus_public_inputs {
                    total = total + &bpi.table_contribution;
                }
            }

            if total != *expected_bus_balance {
                #[cfg(not(feature = "test_fiat_shamir"))]
                error!("LogUp bus does not balance (hybrid verify)");
                return false;
            }
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
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
        PI: Clone,
    {
        let multi_proof = MultiProof {
            proofs: vec![proof.clone()],
            unified_proof: None,
            unified_indices: Vec::new(),
        };
        Self::multi_verify(&[air], &multi_proof, transcript, &FieldElement::zero())
    }

    /// Replays rounds 2, 3 and 4 of the protocol for a given proof, assuming round 1 has
    /// already been replayed and the RAP challenges are known.
    fn replay_rounds_after_round_1(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &StarkProof<Field, FieldExtension, PI>,
        domain: &VerifierDomain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        rap_challenges: Vec<FieldElement<FieldExtension>>,
    ) -> Challenges<FieldExtension>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        // ===================================
        // ==========|   Round 2   |==========
        // ===================================

        // <<<< Receive challenge: 𝛽
        let beta = transcript.sample_field_element();
        let trace_length = proof.trace_length;
        let num_boundary_constraints = air
            .boundary_constraints(
                &proof.public_inputs,
                &rap_challenges,
                proof.bus_public_inputs.as_ref(),
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
        transcript.append_bytes(&proof.composition_poly_root);

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
        let trace_ood_evaluations_columns = proof.trace_ood_evaluations.columns();
        for col in trace_ood_evaluations_columns.iter() {
            for elem in col.iter() {
                transcript.append_field_element(elem);
            }
        }
        // <<<< Receive value: Hᵢ(z^N)
        for element in proof.composition_poly_parts_ood_evaluation.iter() {
            transcript.append_field_element(element);
        }

        // ===================================
        // ==========|   Round 4   |==========
        // ===================================

        let num_terms_composition_poly = proof.composition_poly_parts_ood_evaluation.len();
        let num_terms_trace =
            air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;
        let gamma = transcript.sample_field_element();

        // <<<< Receive challenges: 𝛾, 𝛾'
        let mut deep_composition_coefficients: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
                .take(num_terms_composition_poly + num_terms_trace)
                .collect();

        let trace_term_coeffs: Vec<_> = deep_composition_coefficients
            .drain(..num_terms_trace)
            .collect::<Vec<_>>()
            .chunks(air.context().transition_offsets.len() * air.step_size())
            .map(|chunk| chunk.to_vec())
            .collect();

        // <<<< Receive challenges: 𝛾ⱼ, 𝛾ⱼ'
        let gammas = deep_composition_coefficients;

        // FRI commit phase
        let merkle_roots = &proof.fri_layers_merkle_roots;
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
        transcript.append_field_element(&proof.fri_last_value);

        // Receive grinding value
        let security_bits = air.context().proof_options.grinding_factor;
        let mut grinding_seed = [0u8; 32];
        if security_bits > 0
            && let Some(nonce_value) = proof.nonce
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
            gammas,
            zetas,
            iotas,
            rap_challenges,
            grinding_seed,
        }
    }

    /// Verifies a single table after round 1 has been replayed.
    fn verify_rounds_2_to_4(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &StarkProof<Field, FieldExtension, PI>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        rap_challenges: Vec<FieldElement<FieldExtension>>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let domain = new_verifier_domain(air, proof.trace_length);

        // Verify there are enough queries
        if proof.query_list.len() < air.options().fri_number_of_queries {
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
            let nonce_is_valid = proof.nonce.is_some_and(|nonce_value| {
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

        if !Self::step_2_verify_claimed_composition_polynomial(air, proof, &domain, &challenges) {
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

    /// Verifies a unified multi-table STARK proof with batched commitments and single FRI.
    ///
    /// The `airs` must be non-preprocessed, same-height tables, in the same order as
    /// the prover's `air_trace_pairs`.
    fn multi_verify_unified(
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        pub_inputs: &[&PI],
        proof: &crate::proof::unified::UnifiedMultiProof<Field, FieldExtension>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        expected_bus_balance: &FieldElement<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let num_airs = airs.len();
        if num_airs != proof.table_data.len() || num_airs != proof.column_layout.len() {
            #[cfg(not(feature = "test_fiat_shamir"))]
            error!(
                "AIR count ({}) doesn't match table_data ({}) or column_layout ({})",
                num_airs,
                proof.table_data.len(),
                proof.column_layout.len()
            );
            return false;
        }
        if num_airs == 0 {
            return true;
        }

        let needs_lookup = airs.iter().any(|a| a.has_aux_trace());
        let trace_length = proof.table_data[0].trace_length;
        let domain = new_verifier_domain(airs[0], trace_length);

        // =================================================================
        // Step 1: Replay transcript and recover challenges
        // =================================================================

        // Phase A: main trace root
        transcript.append_bytes(&proof.main_trace_root);

        // Phase B: LogUp challenges
        let rap_challenges: Vec<FieldElement<FieldExtension>> = if needs_lookup {
            (0..LOGUP_NUM_CHALLENGES)
                .map(|_| transcript.sample_field_element())
                .collect()
        } else {
            Vec::new()
        };

        // Phase C: aux root + bus_public_inputs
        if let Some(ref root) = proof.aux_trace_root {
            transcript.append_bytes(root);
        }
        for td in &proof.table_data {
            if let Some(ref bpi) = td.bus_public_inputs {
                transcript.append_field_element(&bpi.table_contribution);
            }
        }

        // Validate bus_public_inputs presence
        for (idx, (air, td)) in airs.iter().zip(&proof.table_data).enumerate() {
            if air.has_trace_interaction() && td.bus_public_inputs.is_none() {
                #[cfg(not(feature = "test_fiat_shamir"))]
                error!("Table {idx}: AIR has LogUp but proof missing bus_public_inputs");
                return false;
            }
        }

        // Round 2: sample beta, receive composition root
        let beta: FieldElement<FieldExtension> = transcript.sample_field_element();
        transcript.append_bytes(&proof.composition_poly_root);

        // Round 3: sample z, receive OOD data
        let z: FieldElement<FieldExtension> = transcript.sample_z_ood_with_domain_params(
            domain.trace_length,
            domain.lde_length,
            &domain.coset_offset,
        );

        for td in &proof.table_data {
            for col in td.trace_ood_evaluations.columns().iter() {
                for elem in col.iter() {
                    transcript.append_field_element(elem);
                }
            }
            for elem in &td.composition_poly_parts_ood_evaluation {
                transcript.append_field_element(elem);
            }
        }

        // Round 4: sample gamma
        let gamma: FieldElement<FieldExtension> = transcript.sample_field_element();

        // FRI commit replay
        let mut zetas: Vec<FieldElement<FieldExtension>> = proof
            .fri_layers_merkle_roots
            .iter()
            .map(|root| {
                let element = transcript.sample_field_element();
                transcript.append_bytes(root);
                element
            })
            .collect();
        zetas.push(transcript.sample_field_element());
        transcript.append_field_element(&proof.fri_last_value);

        // Grinding
        let security_bits = airs[0].context().proof_options.grinding_factor;
        if security_bits > 0 {
            let seed = transcript.state();
            match proof.nonce {
                Some(nonce_value) if grinding::is_valid_nonce(&seed, nonce_value, security_bits) => {
                    transcript.append_bytes(&nonce_value.to_be_bytes());
                }
                _ => {
                    #[cfg(not(feature = "test_fiat_shamir"))]
                    error!("Grinding factor not satisfied");
                    return false;
                }
            }
        }

        // FRI query indices
        let number_of_queries = airs[0].options().fri_number_of_queries;
        let iotas = Self::sample_query_indexes(number_of_queries, &domain, transcript);

        // =================================================================
        // Step 2: Verify composition polynomial per table
        // =================================================================

        for (idx, (air, td)) in airs.iter().zip(&proof.table_data).enumerate() {
            let pi = pub_inputs[idx];
            let num_boundary = air
                .boundary_constraints(
                    pi,
                    &rap_challenges,
                    td.bus_public_inputs.as_ref(),
                    trace_length,
                )
                .constraints
                .len();
            let num_transition = air.context().num_transition_constraints;

            let mut coefficients: Vec<FieldElement<FieldExtension>> =
                core::iter::successors(Some(FieldElement::one()), |x| Some(x * &beta))
                    .take(num_boundary + num_transition)
                    .collect();
            let transition_coefficients: Vec<_> =
                coefficients.drain(..num_transition).collect();
            let boundary_coefficients = coefficients;

            // Boundary quotient at z
            let boundary_constraints = air.boundary_constraints(
                pi,
                &rap_challenges,
                td.bus_public_inputs.as_ref(),
                trace_length,
            );

            let mut boundary_denoms: Vec<FieldElement<FieldExtension>> = Vec::new();
            let boundary_nums: Vec<FieldElement<FieldExtension>> = boundary_constraints
                .constraints
                .iter()
                .map(|bc| {
                    let point = domain.trace_primitive_root.pow(bc.step as u64);
                    boundary_denoms.push(-point + &z);
                    // For aux boundary constraints, adjust column index past main columns
                    let col_idx = if bc.is_aux {
                        td.num_main_cols + bc.col
                    } else {
                        bc.col
                    };
                    -&bc.value + &td.trace_ood_evaluations.get_row(0)[col_idx]
                })
                .collect();
            FieldElement::inplace_batch_inverse(&mut boundary_denoms).unwrap();

            let boundary_quotient: FieldElement<FieldExtension> = boundary_nums
                .iter()
                .zip(&boundary_denoms)
                .zip(&boundary_coefficients)
                .map(|((num, den), coeff)| num * den * coeff)
                .fold(FieldElement::zero(), |acc, x| acc + x);

            // Transition evaluation at z
            let periodic_values: Vec<FieldElement<FieldExtension>> = air
                .get_periodic_column_polynomials(trace_length)
                .iter()
                .map(|poly| poly.evaluate(&z))
                .collect();

            let logup_alpha_powers: Vec<FieldElement<FieldExtension>> =
                if rap_challenges.len() > LOGUP_CHALLENGE_ALPHA {
                    compute_alpha_powers(
                        &rap_challenges[LOGUP_CHALLENGE_ALPHA],
                        air.max_bus_elements(),
                    )
                } else {
                    Vec::new()
                };

            let logup_table_offset = match &td.bus_public_inputs {
                Some(bpi) => {
                    let n = FieldElement::<Field>::from(trace_length as u64);
                    match n.inv() {
                        Ok(n_inv) => n_inv * &bpi.table_contribution,
                        Err(_) => return false,
                    }
                }
                None => FieldElement::zero(),
            };

            let ood_frame =
                td.trace_ood_evaluations
                    .into_frame(td.num_main_cols, air.step_size());
            let packing_shifts = crate::lookup::PackingShifts::<FieldExtension>::new();
            let ctx = TransitionEvaluationContext::new_verifier(
                &ood_frame,
                &periodic_values,
                &rap_challenges,
                &logup_alpha_powers,
                &logup_table_offset,
                &packing_shifts,
            );
            let transition_vals = air.compute_transition(&ctx);

            let mut denominators = vec![
                FieldElement::<FieldExtension>::zero();
                air.num_transition_constraints()
            ];
            air.transition_constraints().iter().for_each(|c| {
                denominators[c.constraint_idx()] = c.evaluate_zerofier(
                    &z,
                    &domain.trace_primitive_root,
                    trace_length,
                );
            });

            let transition_sum = itertools::izip!(
                transition_vals,
                &transition_coefficients,
                denominators
            )
            .fold(FieldElement::zero(), |acc, (val, coeff, denom)| {
                acc + coeff * val * &denom
            });

            let computed = &boundary_quotient + transition_sum;
            let claimed: FieldElement<FieldExtension> = td
                .composition_poly_parts_ood_evaluation
                .iter()
                .rev()
                .fold(FieldElement::zero(), |acc, coeff| acc * &z + coeff);

            if computed != claimed {
                #[cfg(not(feature = "test_fiat_shamir"))]
                error!("Table {idx}: composition polynomial mismatch at z");
                return false;
            }
        }

        // =================================================================
        // Step 3: Verify FRI with unified DEEP reconstruction
        // =================================================================

        let num_eval_points = 2;
        let primitive_root =
            &Field::get_primitive_root_of_unity(domain.root_order as u64).unwrap();

        // Precompute gamma powers (must match prover's assignment)
        let mut total_terms = 0usize;
        for td in &proof.table_data {
            let num_cols = td.num_main_cols + td.num_aux_cols;
            total_terms +=
                num_cols * num_eval_points + td.composition_poly_parts_ood_evaluation.len();
        }
        let gamma_powers: Vec<FieldElement<FieldExtension>> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
                .take(total_terms)
                .collect();

        let mut eval_point_inv: Vec<FieldElement<Field>> = iotas
            .iter()
            .map(|iota| Self::query_challenge_to_evaluation_point(*iota, &domain))
            .collect();
        FieldElement::inplace_batch_inverse(&mut eval_point_inv).unwrap();

        for (q_idx, iota) in iotas.iter().enumerate() {
            let opening = &proof.query_openings[q_idx];
            let index = iota * 2;
            let index_sym = iota * 2 + 1;

            // Verify main tree openings
            if !opening.main_proof.verify::<BatchedMerkleTreeBackend<Field>>(
                &proof.main_trace_root,
                index,
                &opening.main_evaluations,
            ) || !opening.main_proof_sym.verify::<BatchedMerkleTreeBackend<Field>>(
                &proof.main_trace_root,
                index_sym,
                &opening.main_evaluations_sym,
            ) {
                #[cfg(not(feature = "test_fiat_shamir"))]
                error!("Query {q_idx}: main tree opening failed");
                return false;
            }

            // Verify aux tree openings
            if let Some(ref aux_root) = proof.aux_trace_root {
                if let (Some(p), Some(ps)) =
                    (&opening.aux_proof, &opening.aux_proof_sym)
                {
                    if !p.verify::<BatchedMerkleTreeBackend<FieldExtension>>(
                        aux_root,
                        index,
                        &opening.aux_evaluations,
                    ) || !ps.verify::<BatchedMerkleTreeBackend<FieldExtension>>(
                        aux_root,
                        index_sym,
                        &opening.aux_evaluations_sym,
                    ) {
                        #[cfg(not(feature = "test_fiat_shamir"))]
                        error!("Query {q_idx}: aux tree opening failed");
                        return false;
                    }
                }
            }

            // Verify composition tree opening (pair-merged)
            {
                let mut comp_leaf = opening.composition_evaluations.clone();
                comp_leaf.extend_from_slice(&opening.composition_evaluations_sym);
                if !opening
                    .composition_proof
                    .verify::<BatchedMerkleTreeBackend<FieldExtension>>(
                        &proof.composition_poly_root,
                        *iota,
                        &comp_leaf,
                    )
                {
                    #[cfg(not(feature = "test_fiat_shamir"))]
                    error!("Query {q_idx}: composition tree opening failed");
                    return false;
                }
            }

            // Reconstruct DEEP at query point and symmetric
            let ep = Self::query_challenge_to_evaluation_point(*iota, &domain);
            let ep_sym = Self::query_challenge_to_evaluation_point_sym(*iota, &domain);

            let deep = Self::reconstruct_unified_deep(
                &proof.table_data,
                &proof.column_layout,
                &z,
                &gamma_powers,
                &ep,
                primitive_root,
                &opening.main_evaluations,
                &opening.aux_evaluations,
                &opening.composition_evaluations,
            );
            let deep_sym = Self::reconstruct_unified_deep(
                &proof.table_data,
                &proof.column_layout,
                &z,
                &gamma_powers,
                &ep_sym,
                primitive_root,
                &opening.main_evaluations_sym,
                &opening.aux_evaluations_sym,
                &opening.composition_evaluations_sym,
            );

            // Verify FRI folding for this query
            let fri_roots = &proof.fri_layers_merkle_roots;
            let ep_inv_sq_chain: Vec<FieldElement<Field>> =
                core::iter::successors(Some(eval_point_inv[q_idx].square()), |ep| {
                    Some(ep.square())
                })
                .take(fri_roots.len())
                .collect();

            let mut v = (&deep + &deep_sym)
                + &eval_point_inv[q_idx] * &zetas[0] * (&deep - &deep_sym);
            let mut fri_index = *iota;

            if fri_roots.is_empty() {
                if v != proof.fri_last_value {
                    return false;
                }
                continue;
            }

            let fri_decommitment = &proof.fri_query_list[q_idx];
            let fri_ok = fri_roots
                .iter()
                .enumerate()
                .zip(&fri_decommitment.layers_auth_paths)
                .zip(&fri_decommitment.layers_evaluations_sym)
                .zip(ep_inv_sq_chain)
                .fold(
                    true,
                    |result,
                     ((((i, merkle_root), auth_path), eval_sym), ep_inv)| {
                        let layer_ok = Self::verify_fri_layer_openings(
                            merkle_root,
                            auth_path,
                            &v,
                            eval_sym,
                            fri_index,
                        );
                        v = (&v + eval_sym)
                            + ep_inv * &zetas[i + 1] * (&v - eval_sym);
                        fri_index >>= 1;

                        if i < fri_decommitment.layers_evaluations_sym.len() - 1 {
                            result & layer_ok
                        } else {
                            result & (v == proof.fri_last_value) & layer_ok
                        }
                    },
                );

            if !fri_ok {
                #[cfg(not(feature = "test_fiat_shamir"))]
                error!("Query {q_idx}: FRI verification failed");
                return false;
            }
        }

        // =================================================================
        // Step 4: Bus balance check
        // =================================================================

        if needs_lookup {
            let total: FieldElement<FieldExtension> = proof
                .table_data
                .iter()
                .filter_map(|td| td.bus_public_inputs.as_ref())
                .map(|bpi| &bpi.table_contribution)
                .fold(FieldElement::zero(), |acc, c| acc + c);

            if total != *expected_bus_balance {
                #[cfg(not(feature = "test_fiat_shamir"))]
                error!("LogUp bus does not balance");
                return false;
            }
        }

        true
    }

    /// Reconstruct the unified DEEP polynomial evaluation at a single point.
    #[allow(clippy::too_many_arguments)]
    fn reconstruct_unified_deep(
        table_data: &[crate::proof::unified::TableOodData<FieldExtension>],
        column_layout: &[crate::proof::unified::ColumnRange],
        z: &FieldElement<FieldExtension>,
        gamma_powers: &[FieldElement<FieldExtension>],
        x: &FieldElement<Field>,
        primitive_root: &FieldElement<Field>,
        main_evals: &[FieldElement<Field>],
        aux_evals: &[FieldElement<FieldExtension>],
        comp_evals: &[FieldElement<FieldExtension>],
    ) -> FieldElement<FieldExtension> {
        let z_shifted = [z.clone(), primitive_root * z];
        let mut trace_denoms: Vec<FieldElement<FieldExtension>> =
            z_shifted.iter().map(|zk| x - zk).collect();
        FieldElement::inplace_batch_inverse(&mut trace_denoms).unwrap();

        let mut result = FieldElement::<FieldExtension>::zero();
        let mut term_idx = 0usize;

        for (td, layout) in table_data.iter().zip(column_layout) {
            let ood_cols = td.trace_ood_evaluations.columns();

            // Main trace terms
            for j in 0..layout.main_count {
                let t_val: FieldElement<FieldExtension> =
                    main_evals[layout.main_start + j].clone().to_extension();
                for k in 0..2 {
                    let num = &t_val - &ood_cols[j][k];
                    result += &gamma_powers[term_idx] * &num * &trace_denoms[k];
                    term_idx += 1;
                }
            }

            // Aux trace terms
            for j in 0..layout.aux_count {
                let t_val = &aux_evals[layout.aux_start + j];
                let ood_idx = layout.main_count + j;
                for k in 0..2 {
                    let num = t_val - &ood_cols[ood_idx][k];
                    result += &gamma_powers[term_idx] * &num * &trace_denoms[k];
                    term_idx += 1;
                }
            }

            // Composition terms
            let num_parts = td.composition_poly_parts_ood_evaluation.len();
            let z_power_k = z.pow(num_parts);
            let comp_denom = (x - &z_power_k).inv().expect("comp denom non-zero");
            for p in 0..num_parts {
                let h_val = &comp_evals[layout.comp_start + p];
                let h_ood = &td.composition_poly_parts_ood_evaluation[p];
                let num = h_val - h_ood;
                result += &gamma_powers[term_idx] * &num * &comp_denom;
                term_idx += 1;
            }
        }

        result
    }
}
