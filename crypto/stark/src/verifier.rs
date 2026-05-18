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
    domain::{QuotientDomain, new_verifier_domain},
    lookup::{LOGUP_CHALLENGE_ALPHA, LOGUP_NUM_CHALLENGES, PackingShifts, compute_alpha_powers},
    proof::stark::{
        DeepPolynomialOpening, DeepPolynomialOpeningChunks, MultiProof, StarkProofChunks,
    },
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
    polynomial::Polynomial,
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
        // pₙ: one element in the binary-fold path; the Vec carries the final
        // polynomial coefficients when early stopping is enabled.
        for coef in &proof.fri_last_value {
            transcript.append_field_element(coef);
        }

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

    /// Check the verifier's running FRI value `v` against the prover's
    /// `fri_last_value`. With early stopping `fri_last_value` carries the
    /// coefficients of the residual polynomial; with fold-to-constant it is
    /// a single field element (length 1, which Horner-evaluates to itself at
    /// any point).
    ///
    /// `k_folds` is the number of binary folds the verifier has performed
    /// (1 for the empty-FRI-layers branch, `fri_layers_merkle_roots.len() + 1`
    /// after the full fold loop).
    fn check_final_fri_value(
        fri_last_value: &[FieldElement<FieldExtension>],
        evaluation_point_inv: &FieldElement<Field>,
        k_folds: usize,
        v: &FieldElement<FieldExtension>,
    ) -> bool {
        if fri_last_value.is_empty() {
            return false;
        }
        // After k folds the evaluation point is `original_eval^(2^k)`. We
        // have `evaluation_point_inv = 1 / original_eval`, so the layer-k
        // forward point is `(evaluation_point_inv^(2^k)).inv()`.
        let mut inv_at_layer_k = evaluation_point_inv.clone();
        for _ in 0..k_folds {
            inv_at_layer_k = inv_at_layer_k.square();
        }
        let point_at_layer_k = match inv_at_layer_k.inv() {
            Ok(p) => p,
            Err(_) => return false,
        };
        // Lift to the extension field so the polynomial (with extension
        // coefficients) can be Horner-evaluated at it.
        let point_ext: FieldElement<FieldExtension> = point_at_layer_k.to_extension();
        let expected = Polynomial::new(fri_last_value).evaluate(&point_ext);
        v == &expected
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

        // Keep the layer-0 eval-point inverse so we can compute the final
        // point at layer K in `check_final_fri_value`.
        let initial_eval_inv = evaluation_point_inv.clone();

        // Reconstruct p₁(𝜐²)
        let mut v =
            (p0_eval + p0_eval_sym) + evaluation_point_inv * &zetas[0] * (p0_eval - p0_eval_sym);
        let mut index = iota;

        // Handle case with 0 FRI layers (trace_length <= 2)
        // In this case, the fold loop below doesn't iterate, so we need to verify
        // the final value directly here.
        if fri_layers_merkle_roots.is_empty() {
            // Only the initial fold has been performed (K = 1).
            return Self::check_final_fri_value(
                &proof.fri_last_value,
                &initial_eval_inv,
                1,
                &v,
            );
        }

        // For each FRI layer, starting from the layer 1: use the proof to verify the validity of values pᵢ(−𝜐^(2ⁱ)) (given by the prover) and
        // pᵢ(𝜐^(2ⁱ)) (computed on the previous iteration by the verifier). Then use them to obtain pᵢ₊₁(𝜐^(2ⁱ⁺¹)).
        // Finally, check that the final value coincides with the given by the prover.
        let total_folds = fri_layers_merkle_roots.len() + 1;
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
                    (((i, merkle_root), auth_path_sym), evaluation_sym_layer),
                    evaluation_point_inv,
                )| {
                    // Binary fold: each layer carries exactly one sibling.
                    let evaluation_sym = &evaluation_sym_layer[0];
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
                        // After total_folds binary folds, check v against the
                        // final polynomial shipped by the prover. For fold-to-
                        // constant, fri_last_value is length 1 and Horner
                        // collapses to the constant; for early-stop it carries
                        // the coefficients of the residual polynomial.
                        let last_ok = Self::check_final_fri_value(
                            &proof.fri_last_value,
                            &initial_eval_inv,
                            total_folds,
                            &v,
                        );
                        result & last_ok & openings_ok
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
        // pₙ: one element in the binary-fold path; the Vec carries the final
        // polynomial coefficients when early stopping is enabled.
        for coef in &proof.fri_last_value {
            transcript.append_field_element(coef);
        }

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

    // ====================================================================
    // ==                Phase 4.5 chunks-protocol verifier              ==
    // ====================================================================
    //
    // Mirrors `step_1`/`step_2`/`step_3`/`step_4` and `verify` but consumes
    // `StarkProofChunks`. Transcript replay differs in (a) round-2 receives
    // one append per chunk root (not one composition_poly_root), and (b)
    // round-3 receives `Q_c(z)` per chunk (not the composition-parts OOD
    // evaluation at `z^N`). Single-H path is untouched.

    /// Chunks-protocol analogue of
    /// [`Self::step_1_replay_rounds_and_recover_challenges`].
    fn step_1_replay_rounds_and_recover_challenges_chunks(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &StarkProofChunks<Field, FieldExtension, PI>,
        domain: &VerifierDomain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Challenges<FieldExtension>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        // === Round 1 ===
        transcript.append_bytes(&proof.lde_trace_main_merkle_root);
        let rap_challenges = air.build_rap_challenges(transcript);
        if let Some(root) = proof.lde_trace_aux_merkle_root {
            transcript.append_bytes(&root);
        }

        // === Round 2 ===
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

        // <<<< Per-chunk roots, in chunk-index order (matches prover).
        for root in &proof.quotient_chunk_roots {
            transcript.append_bytes(root);
        }

        // === Round 3 ===
        let z = transcript.sample_z_ood_with_domain_params(
            domain.trace_length,
            domain.lde_length,
            &domain.coset_offset,
        );
        let trace_ood_evaluations_columns = proof.trace_ood_evaluations.columns();
        for col in trace_ood_evaluations_columns.iter() {
            for elem in col.iter() {
                transcript.append_field_element(elem);
            }
        }
        // <<<< Q_c(z) per chunk (replaces composition_poly_parts_ood_evaluation).
        for element in proof.quotient_chunk_ood_evaluations.iter() {
            transcript.append_field_element(element);
        }

        // === Round 4 ===
        let num_chunks = proof.quotient_chunk_ood_evaluations.len();
        let num_terms_trace =
            air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;
        let gamma = transcript.sample_field_element();
        let mut deep_composition_coefficients: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
                .take(num_chunks + num_terms_trace)
                .collect();
        let trace_term_coeffs: Vec<_> = deep_composition_coefficients
            .drain(..num_terms_trace)
            .collect::<Vec<_>>()
            .chunks(air.context().transition_offsets.len() * air.step_size())
            .map(|chunk| chunk.to_vec())
            .collect();
        let gammas = deep_composition_coefficients;

        // FRI commit phase replay.
        let merkle_roots = &proof.fri_layers_merkle_roots;
        let mut zetas = merkle_roots
            .iter()
            .map(|root| {
                let element = transcript.sample_field_element();
                transcript.append_bytes(root);
                element
            })
            .collect::<Vec<FieldElement<FieldExtension>>>();
        zetas.push(transcript.sample_field_element());
        // pₙ: one element in the binary-fold path; the Vec carries the final
        // polynomial coefficients when early stopping is enabled.
        for coef in &proof.fri_last_value {
            transcript.append_field_element(coef);
        }

        // Grinding replay.
        let security_bits = air.context().proof_options.grinding_factor;
        let mut grinding_seed = [0u8; 32];
        if security_bits > 0
            && let Some(nonce_value) = proof.nonce
        {
            grinding_seed = transcript.state();
            transcript.append_bytes(&nonce_value.to_be_bytes());
        }

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

    /// Chunks-protocol analogue of
    /// [`Self::step_2_verify_claimed_composition_polynomial`].
    ///
    /// Computes the expected `H(z)` from boundary + transition constraint
    /// folds exactly like the single-H path. The "claimed" `H(z)` is then
    /// recomposed from `proof.quotient_chunk_ood_evaluations` via
    /// [`QuotientDomain::recompose_at`] (P3-style Lagrange over disjoint
    /// sub-cosets) instead of the single-H Horner fold of composition-poly
    /// parts at `z^num_parts`.
    fn step_2_verify_claimed_composition_polynomial_chunks(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &StarkProofChunks<Field, FieldExtension, PI>,
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
                    Err(_) => return false,
                }
            }
            None => FieldElement::zero(),
        };

        let ood_frame =
            (proof.trace_ood_evaluations.clone()).into_frame(num_main_trace_columns, air.step_size());
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

        // Verify the proof's chunk count matches the AIR's d_max. We compute
        // expected_num_chunks directly (without building a QuotientDomain)
        // because the full Domain construction allocates `roots_of_unity_coset`
        // of size N — wasted work in the num_chunks=1 fast path below, and
        // measurable on large traces (~5ms at log_rows=21 ≈ 18% of single-H's
        // verify_s on the EPYC bench server, 2026-05-15).
        let d_max = air.composition_poly_degree_bound(trace_length) / trace_length;
        let expected_num_chunks = d_max.next_power_of_two().max(1);
        if expected_num_chunks != proof.quotient_chunk_ood_evaluations.len() {
            return false;
        }

        let composition_poly_claimed_ood_evaluation = if expected_num_chunks == 1 {
            // For num_chunks=1, recompose_at is the identity: the empty Lagrange
            // product gives zps[0] = 1, so H(z) = 1 * Q_0(z) = chunk_ood[0].
            // Skip the QuotientDomain construction entirely.
            proof.quotient_chunk_ood_evaluations[0].clone()
        } else {
            let qd = QuotientDomain::<Field>::from_parts(
                trace_length,
                domain.coset_offset.clone(),
                d_max,
            );
            qd.recompose_at(&proof.quotient_chunk_ood_evaluations, &challenges.z)
        };

        composition_poly_claimed_ood_evaluation == composition_poly_ood_evaluation
    }

    /// Chunks-protocol analogue of [`Self::verify_trace_openings`].
    fn verify_trace_openings_chunks(
        proof: &StarkProofChunks<Field, FieldExtension, PI>,
        deep_poly_openings: &DeepPolynomialOpeningChunks<Field, FieldExtension>,
        iota: usize,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let index = iota * 2;
        let index_sym = iota * 2 + 1;
        let mut result = true;

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

        match (
            &proof.lde_trace_precomputed_merkle_root,
            &deep_poly_openings.precomputed_trace_polys,
        ) {
            (None, Some(_)) => result = false,
            (Some(_), None) => result = false,
            (Some(pre_root), Some(pre_open)) => {
                result &= Self::verify_opening::<Field>(
                    &pre_open.proof,
                    pre_root,
                    index,
                    &pre_open.evaluations,
                );
                result &= Self::verify_opening::<Field>(
                    &pre_open.proof_sym,
                    pre_root,
                    index_sym,
                    &pre_open.evaluations_sym,
                );
            }
            _ => {}
        }

        match (
            proof.lde_trace_aux_merkle_root,
            &deep_poly_openings.aux_trace_polys,
        ) {
            (None, Some(_)) => result = false,
            (Some(_), None) => result = false,
            (Some(aux_root), Some(aux_open)) => {
                result &= Self::verify_opening::<FieldExtension>(
                    &aux_open.proof,
                    &aux_root,
                    index,
                    &aux_open.evaluations,
                );
                result &= Self::verify_opening::<FieldExtension>(
                    &aux_open.proof_sym,
                    &aux_root,
                    index_sym,
                    &aux_open.evaluations_sym,
                );
            }
            _ => {}
        }

        result
    }

    /// Chunks-protocol analogue of
    /// [`Self::step_4_verify_trace_and_composition_openings`]: verifies
    /// trace Merkle paths (identical to single-H) plus per-chunk Merkle paths
    /// — one inclusion proof per chunk, against its own root in
    /// `proof.quotient_chunk_roots`.
    fn step_4_verify_trace_and_composition_openings_chunks(
        proof: &StarkProofChunks<Field, FieldExtension, PI>,
        challenges: &Challenges<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        challenges.iotas.iter().zip(&proof.deep_poly_openings).fold(
            true,
            |mut result, (iota_n, deep_poly_opening)| {
                // Verify each chunk's Merkle path independently against its root.
                if deep_poly_opening.quotient_chunks.len() != proof.quotient_chunk_roots.len() {
                    return false;
                }
                for (chunk_open, chunk_root) in deep_poly_opening
                    .quotient_chunks
                    .iter()
                    .zip(proof.quotient_chunk_roots.iter())
                {
                    let mut leaf = chunk_open.evaluations.clone();
                    leaf.extend_from_slice(&chunk_open.evaluations_sym);
                    result &=
                        chunk_open.proof.verify::<BatchedMerkleTreeBackend<FieldExtension>>(
                            chunk_root,
                            *iota_n,
                            &leaf,
                        );
                }
                result &= Self::verify_trace_openings_chunks(proof, deep_poly_opening, *iota_n);
                result
            },
        )
    }

    /// Reconstructs the DEEP composition polynomial value at a single
    /// evaluation point from the chunks-protocol query opening.
    ///
    /// Trace contribution is identical to the single-H reconstruction; the
    /// H contribution is replaced by the chunks formula
    /// `sum_c gamma_c * (Q_c(x) - Q_c(z)) / (x - z)`.
    fn reconstruct_deep_composition_poly_evaluation_chunks(
        proof: &StarkProofChunks<Field, FieldExtension, PI>,
        evaluation_point: &FieldElement<Field>,
        primitive_root: &FieldElement<Field>,
        challenges: &Challenges<FieldExtension>,
        lde_trace_evaluations: &[FieldElement<FieldExtension>],
        chunk_evaluations_at_point: &[FieldElement<FieldExtension>],
    ) -> FieldElement<FieldExtension> {
        let ood_evaluations_table_height = proof.trace_ood_evaluations.height;
        let ood_evaluations_table_width = proof.trace_ood_evaluations.width;

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

        // Chunks H contribution: sum_c gamma_c * (Q_c(x) - Q_c(z)) / (x - z).
        let denom_h = (evaluation_point - &challenges.z).inv().unwrap();
        let mut h_terms = FieldElement::zero();
        for (c, q_c_at_x) in chunk_evaluations_at_point.iter().enumerate() {
            let q_c_at_z = &proof.quotient_chunk_ood_evaluations[c];
            let term = (q_c_at_x - q_c_at_z) * &challenges.gammas[c];
            h_terms += term;
        }
        h_terms *= denom_h;

        trace_term + h_terms
    }

    /// Chunks-protocol analogue of
    /// `reconstruct_deep_composition_poly_evaluations_for_all_queries`.
    fn reconstruct_deep_composition_poly_evaluations_for_all_queries_chunks(
        challenges: &Challenges<FieldExtension>,
        domain: &VerifierDomain<Field>,
        proof: &StarkProofChunks<Field, FieldExtension, PI>,
    ) -> DeepPolynomialEvaluations<FieldExtension> {
        let num_queries = challenges.iotas.len();
        let mut deep_poly_evaluations = Vec::with_capacity(num_queries);
        let mut deep_poly_evaluations_sym = Vec::with_capacity(num_queries);
        for (i, iota) in challenges.iotas.iter().enumerate() {
            let primitive_root =
                &Field::get_primitive_root_of_unity(domain.root_order as u64).unwrap();

            // Trace evaluations at v.
            let mut evaluations: Vec<FieldElement<FieldExtension>> = Vec::new();
            if let Some(pre) = &proof.deep_poly_openings[i].precomputed_trace_polys {
                evaluations.extend(pre.evaluations.iter().cloned().map(|x| x.to_extension()));
            }
            evaluations.extend(
                proof.deep_poly_openings[i]
                    .main_trace_polys
                    .evaluations
                    .iter()
                    .cloned()
                    .map(|x| x.to_extension()),
            );
            if let Some(aux) = &proof.deep_poly_openings[i].aux_trace_polys {
                evaluations.extend_from_slice(&aux.evaluations);
            }

            // Q_c(v) — one per chunk, in chunk-index order.
            let chunk_evals_at_point: Vec<FieldElement<FieldExtension>> = proof
                .deep_poly_openings[i]
                .quotient_chunks
                .iter()
                .map(|co| co.evaluations[0].clone())
                .collect();

            let evaluation_point = Self::query_challenge_to_evaluation_point(*iota, domain);
            deep_poly_evaluations.push(
                Self::reconstruct_deep_composition_poly_evaluation_chunks(
                    proof,
                    &evaluation_point,
                    primitive_root,
                    challenges,
                    &evaluations,
                    &chunk_evals_at_point,
                ),
            );

            // Symmetric side.
            let mut evaluations_sym: Vec<FieldElement<FieldExtension>> = Vec::new();
            if let Some(pre) = &proof.deep_poly_openings[i].precomputed_trace_polys {
                evaluations_sym.extend(pre.evaluations_sym.iter().cloned().map(|x| x.to_extension()));
            }
            evaluations_sym.extend(
                proof.deep_poly_openings[i]
                    .main_trace_polys
                    .evaluations_sym
                    .iter()
                    .cloned()
                    .map(|x| x.to_extension()),
            );
            if let Some(aux) = &proof.deep_poly_openings[i].aux_trace_polys {
                evaluations_sym.extend_from_slice(&aux.evaluations_sym);
            }
            let chunk_evals_at_point_sym: Vec<FieldElement<FieldExtension>> = proof
                .deep_poly_openings[i]
                .quotient_chunks
                .iter()
                .map(|co| co.evaluations_sym[0].clone())
                .collect();

            let evaluation_point_sym =
                Self::query_challenge_to_evaluation_point_sym(*iota, domain);
            deep_poly_evaluations_sym.push(
                Self::reconstruct_deep_composition_poly_evaluation_chunks(
                    proof,
                    &evaluation_point_sym,
                    primitive_root,
                    challenges,
                    &evaluations_sym,
                    &chunk_evals_at_point_sym,
                ),
            );
        }
        (deep_poly_evaluations, deep_poly_evaluations_sym)
    }

    /// Chunks-protocol analogue of [`Self::step_3_verify_fri`].
    fn step_3_verify_fri_chunks(
        proof: &StarkProofChunks<Field, FieldExtension, PI>,
        domain: &VerifierDomain<Field>,
        challenges: &Challenges<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let (deep_poly_evaluations, deep_poly_evaluations_sym) =
            Self::reconstruct_deep_composition_poly_evaluations_for_all_queries_chunks(
                challenges, domain, proof,
            );

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
                // FRI fold verification at query `iota_s`. Replays the same
                // logic as `verify_query_and_sym_openings` but in-line here
                // because that helper is typed against `StarkProof` (single-H).
                let fri_layers_merkle_roots = &proof.fri_layers_merkle_roots;
                let evaluation_point_vec: Vec<FieldElement<Field>> = core::iter::successors(
                    Some(eval.square()),
                    |evaluation_point| Some(evaluation_point.square()),
                )
                .take(fri_layers_merkle_roots.len())
                .collect();

                let p0_eval = &deep_poly_evaluations[i];
                let p0_eval_sym = &deep_poly_evaluations_sym[i];
                // Save before the fold consumes `eval`; we need it again
                // for `check_final_fri_value`.
                let initial_eval_inv = eval.clone();
                let mut v = (p0_eval + p0_eval_sym)
                    + eval * &challenges.zetas[0] * (p0_eval - p0_eval_sym);
                let mut index = *iota_s;

                if fri_layers_merkle_roots.is_empty() {
                    // Only the initial fold has been performed (K = 1).
                    let last_ok = Self::check_final_fri_value(
                        &proof.fri_last_value,
                        &initial_eval_inv,
                        1,
                        &v,
                    );
                    return result & last_ok;
                }

                let total_folds = fri_layers_merkle_roots.len() + 1;
                let inner = fri_layers_merkle_roots
                    .iter()
                    .enumerate()
                    .zip(&proof_s.layers_auth_paths)
                    .zip(&proof_s.layers_evaluations_sym)
                    .zip(evaluation_point_vec)
                    .fold(
                        true,
                        |inner_ok,
                         ((((j, merkle_root), auth_path_sym), evaluation_sym_layer), evaluation_point_inv)| {
                            // Binary fold: each layer carries exactly one sibling.
                            let evaluation_sym = &evaluation_sym_layer[0];
                            let openings_ok = Self::verify_fri_layer_openings(
                                merkle_root,
                                auth_path_sym,
                                &v,
                                evaluation_sym,
                                index,
                            );
                            v = (&v + evaluation_sym)
                                + evaluation_point_inv
                                    * &challenges.zetas[j + 1]
                                    * (&v - evaluation_sym);
                            index >>= 1;

                            if j < proof_s.layers_evaluations_sym.len() - 1 {
                                inner_ok & openings_ok
                            } else {
                                let last_ok = Self::check_final_fri_value(
                                    &proof.fri_last_value,
                                    &initial_eval_inv,
                                    total_folds,
                                    &v,
                                );
                                inner_ok & last_ok & openings_ok
                            }
                        },
                    );
                result &= inner;
                result
            })
    }

    /// Top-level chunks-protocol verifier: replays the transcript, runs
    /// step_2/3/4 chunks variants, returns whether the proof is accepted.
    fn verify_chunks(
        proof: &StarkProofChunks<Field, FieldExtension, PI>,
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
        PI: Clone,
    {
        let domain = new_verifier_domain(air, proof.trace_length);
        let challenges = Self::step_1_replay_rounds_and_recover_challenges_chunks(
            air, proof, &domain, transcript,
        );

        // Grinding check.
        if air.context().proof_options.grinding_factor > 0 {
            let Some(nonce) = proof.nonce else {
                return false;
            };
            if !grinding::is_valid_nonce(
                &challenges.grinding_seed,
                nonce,
                air.context().proof_options.grinding_factor,
            ) {
                return false;
            }
        }

        if !Self::step_2_verify_claimed_composition_polynomial_chunks(
            air, proof, &domain, &challenges,
        ) {
            return false;
        }
        if !Self::step_3_verify_fri_chunks(proof, &domain, &challenges) {
            return false;
        }
        if !Self::step_4_verify_trace_and_composition_openings_chunks(proof, &challenges) {
            return false;
        }
        true
    }
}
