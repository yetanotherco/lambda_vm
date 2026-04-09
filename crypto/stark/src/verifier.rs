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
    lookup::{
        LOGUP_CHALLENGE_ALPHA, LOGUP_NUM_CHALLENGES, PackingShifts, compute_alpha_powers,
        extend_rap_challenges_with_bridge, extract_column_indices,
    },
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
        // Phase B': Replay batch GKR proof (main transcript)
        // =====================================================================
        // Verify the batch GKR proof once, then distribute per-instance claims
        // to each table. This must match the prover's Phase B' exactly.

        let mut gkr_bridge_claims: Vec<Vec<(usize, FieldElement<FieldExtension>)>> =
            Vec::with_capacity(airs.len());
        let mut gkr_random_points: Vec<Vec<FieldElement<FieldExtension>>> =
            Vec::with_capacity(airs.len());

        // Collect which tables have interactions (and their n_layers)
        let mut gkr_table_indices: Vec<usize> = Vec::new();
        let mut n_layers_by_instance: Vec<usize> = Vec::new();
        for (idx, air) in airs.iter().enumerate() {
            if air.has_trace_interaction() {
                gkr_table_indices.push(idx);
                let trace_len = multi_proof.proofs[idx].trace_length;
                let n_vars = trace_len.trailing_zeros() as usize;
                n_layers_by_instance.push(n_vars);
            }
        }

        if !gkr_table_indices.is_empty() {
            let batch_proof = match &multi_proof.batch_gkr_proof {
                Some(p) => p,
                None => {
                    #[cfg(not(feature = "test_fiat_shamir"))]
                    error!("Tables have bus interactions but MultiProof missing batch_gkr_proof");
                    return false;
                }
            };

            // Verify batch GKR proof
            match crate::gkr::gkr_verify_batch(batch_proof, &n_layers_by_instance, transcript) {
                Ok((shared_random_point, per_instance_claims)) => {
                    // Distribute per-instance results to each table
                    // Initialize all tables with empty claims
                    for _ in 0..airs.len() {
                        gkr_bridge_claims.push(Vec::new());
                        gkr_random_points.push(Vec::new());
                    }

                    for (i, &table_idx) in gkr_table_indices.iter().enumerate() {
                        let air = airs[table_idx];
                        let proof = &multi_proof.proofs[table_idx];

                        let gkr_proof_data = match &proof.logup_gkr_proof {
                            Some(p) => p,
                            None => {
                                #[cfg(not(feature = "test_fiat_shamir"))]
                                error!(
                                    "Table {table_idx}: AIR has interactions but proof missing logup_gkr_proof"
                                );
                                return false;
                            }
                        };

                        let (n_claim, d_claim) = &per_instance_claims[i];

                        // Validate column_claims length matches AIR-expected column set
                        let expected_cols = extract_column_indices(air.bus_interactions());
                        if gkr_proof_data.column_claims.len() != expected_cols.len() {
                            #[cfg(not(feature = "test_fiat_shamir"))]
                            error!(
                                "Table {table_idx}: column_claims length mismatch: got {}, expected {}",
                                gkr_proof_data.column_claims.len(),
                                expected_cols.len()
                            );
                            return false;
                        }

                        // Verify that column_claims are consistent with GKR output
                        if !crate::lookup::reconstruct_and_verify_gkr_claims(
                            n_claim,
                            d_claim,
                            &gkr_proof_data.column_claims,
                            air.bus_interactions(),
                            &lookup_challenges,
                        ) {
                            #[cfg(not(feature = "test_fiat_shamir"))]
                            error!(
                                "Table {table_idx}: GKR column claims verification failed — \
                                 column_claims are inconsistent with GKR output (n_claim, d_claim)"
                            );
                            return false;
                        }

                        gkr_bridge_claims[table_idx] = gkr_proof_data.column_claims.clone();

                        // Compute the instance eval point from the shared random point
                        let n_vars = n_layers_by_instance[i];
                        let instance_point =
                            crate::gkr::instance_eval_point(&shared_random_point, n_vars);
                        gkr_random_points[table_idx] = instance_point;
                    }
                }
                Err(_e) => {
                    #[cfg(not(feature = "test_fiat_shamir"))]
                    error!("Batch GKR verification failed: {:?}", _e);
                    return false;
                }
            }
        } else {
            // No tables with interactions - fill with empty claims
            for _ in 0..airs.len() {
                gkr_bridge_claims.push(Vec::new());
                gkr_random_points.push(Vec::new());
            }
        }

        // =====================================================================
        // Phase B'': Sample γ for bridge batching (main transcript)
        // =====================================================================
        // Must match prover: γ sampled AFTER all GKR messages, BEFORE forking.

        let gamma: FieldElement<FieldExtension> = if needs_lookup_challenges {
            transcript.sample_field_element()
        } else {
            FieldElement::zero()
        };

        // =====================================================================
        // Phase C + Rounds 2-4: Forked per table
        // =====================================================================
        // Each table gets an independent transcript fork (cloned from the shared
        // state after Phase B'', domain-separated by table index). This matches
        // the prover's forking and makes per-table verification independent.

        for (idx, (air, proof)) in airs.iter().zip(&multi_proof.proofs).enumerate() {
            let num_tables = airs.len();
            let mut table_transcript = transcript.clone();
            if num_tables > 1 {
                table_transcript.append_bytes(&(idx as u64).to_le_bytes());
            }

            // Phase C: replay aux commitment
            if let Some(root) = proof.lde_trace_aux_merkle_root {
                table_transcript.append_bytes(&root);
            }

            // Build per-table rap_challenges with bridge params
            let mut table_rap_challenges = lookup_challenges.clone();
            if !gkr_bridge_claims[idx].is_empty() {
                extend_rap_challenges_with_bridge(
                    &mut table_rap_challenges,
                    &gkr_bridge_claims[idx],
                    &gamma,
                    proof.trace_length,
                    &gkr_random_points[idx],
                );
            }

            // Rounds 2-4: verify (bridge is now a transition constraint)
            if !Self::verify_rounds_2_to_4(
                *air,
                proof,
                &mut table_transcript,
                table_rap_challenges,
            ) {
                #[cfg(not(feature = "test_fiat_shamir"))]
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
        // With LogUp-GKR, the table contribution is the GKR claimed_sum.
        // The bus balances when the sum of all claimed_sums equals the expected target.
        // When all bus participants are in-trace, the target is zero. When some
        // receiver contributions are computed externally (e.g. verifier-computed
        // COMMIT output bus), the target is the missing positive remainder.

        if needs_lookup_challenges {
            let mut total = FieldElement::<FieldExtension>::zero();
            if let Some(ref batch_proof) = multi_proof.batch_gkr_proof {
                // Sum the table contributions from the batch GKR root claims.
                // Each root_claim is (numerator, denominator); the rational value n/d
                // is the table's contribution.
                for (root_n, root_d) in &batch_proof.root_claims {
                    let contrib = if *root_d == FieldElement::one() {
                        root_n.clone()
                    } else {
                        root_n * &root_d.inv().expect("GKR root denominator must be non-zero")
                    };
                    total = total + &contrib;
                }
            }

            if total != *expected_bus_balance {
                #[cfg(not(feature = "test_fiat_shamir"))]
                error!(
                    "LogUp bus does not balance: sum of GKR claimed_sums does not match target. total={:?}, target={:?}",
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
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
        PI: Clone,
    {
        let multi_proof = MultiProof {
            proofs: vec![proof.clone()],
            batch_gkr_proof: None,
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

        let challenges = Self::replay_rounds_after_round_1(
            air,
            proof,
            &domain,
            transcript,
            rap_challenges,
        );

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
}
