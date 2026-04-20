use super::{
    config::{BatchedMerkleTreeBackend, FriLayerQuadMerkleTreeBackend},
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
        let query_domain_size = domain_size >> 2;
        if query_domain_size == 0 {
            return vec![];
        }
        (0..number_of_queries)
            .map(|_| (transcript.sample_u64(query_domain_size)) as usize)
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

        // FRI commit phase — arity-4: 2 zetas per double-fold layer, 1 for odd-extra layer.
        // number_layers = domain.root_order (log2 of the LDE domain size).
        let number_layers = domain.root_order as usize;
        let num_double_rounds = number_layers.saturating_sub(1) / 2;
        let merkle_roots = &proof.fri_layers_merkle_roots;
        let mut zetas = Vec::with_capacity(number_layers);
        for (i, root) in merkle_roots.iter().enumerate() {
            // >>>> Send challenges: 2 for double-fold layers, 1 for the odd-extra layer.
            let z1 = transcript.sample_field_element();
            zetas.push(z1);
            if i < num_double_rounds {
                let z2 = transcript.sample_field_element();
                zetas.push(z2);
            }
            // <<<< Receive commitment: [pₖ]
            transcript.append_bytes(root);
        }

        // >>>> Send final challenge 𝜁ₙ₋₁ (for the last single fold that produces fri_last_value)
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
        let (evals_0, evals_1, evals_2, evals_3) =
            Self::reconstruct_deep_composition_poly_evaluations_for_all_queries(
                challenges, domain, proof,
            );

        // Arity-4: each query needs eval_inv at position 4*iota (eval_inv_a)
        // and at position 4*iota+2 (eval_inv_b) for the 2-step fold bootstrap.
        let mut eval_inv_a_vec: Vec<FieldElement<Field>> = challenges
            .iotas
            .iter()
            .map(|iota| Self::query_challenge_to_evaluation_point(*iota, domain))
            .collect();
        let mut eval_inv_b_vec: Vec<FieldElement<Field>> = challenges
            .iotas
            .iter()
            .map(|iota| Self::query_challenge_to_evaluation_point_2(*iota, domain))
            .collect();
        FieldElement::inplace_batch_inverse(&mut eval_inv_a_vec).unwrap();
        FieldElement::inplace_batch_inverse(&mut eval_inv_b_vec).unwrap();

        proof
            .query_list
            .iter()
            .zip(&challenges.iotas)
            .zip(eval_inv_a_vec)
            .zip(eval_inv_b_vec)
            .enumerate()
            .fold(
                true,
                |mut result, (i, (((proof_s, iota_s), eval_inv_a), eval_inv_b))| {
                    result &= Self::verify_query_and_sym_openings(
                        proof,
                        domain,
                        &challenges.zetas,
                        *iota_s,
                        proof_s,
                        eval_inv_a,
                        eval_inv_b,
                        &evals_0[i],
                        &evals_1[i],
                        &evals_2[i],
                        &evals_3[i],
                    );
                    result
                },
            )
    }

    /// Returns the coset element at bit-reversed position `4*iota` (orbit base for arity-4 FRI).
    fn query_challenge_to_evaluation_point(
        iota: usize,
        domain: &VerifierDomain<Field>,
    ) -> FieldElement<Field> {
        let index = reverse_index(iota * 4, domain.lde_length as u64);
        domain.lde_coset_element(index)
    }

    /// Returns the coset element at bit-reversed position `4*iota+2` (second fold partner in arity-4 FRI).
    fn query_challenge_to_evaluation_point_2(
        iota: usize,
        domain: &VerifierDomain<Field>,
    ) -> FieldElement<Field> {
        let index = reverse_index(iota * 4 + 2, domain.lde_length as u64);
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
        // Arity-4: 4 positions per query orbit.
        let i0 = iota * 4;
        let i1 = iota * 4 + 1;
        let i2 = iota * 4 + 2;
        let i3 = iota * 4 + 3;
        let mut result = true;

        // Verify main trace (multiplicities for preprocessed, full trace for normal)
        result &= Self::verify_opening::<Field>(
            &deep_poly_openings.main_trace_polys.proof,
            &proof.lde_trace_main_merkle_root,
            i0,
            &deep_poly_openings.main_trace_polys.evaluations,
        );
        result &= Self::verify_opening::<Field>(
            &deep_poly_openings.main_trace_polys.proof_sym,
            &proof.lde_trace_main_merkle_root,
            i1,
            &deep_poly_openings.main_trace_polys.evaluations_sym,
        );
        result &= Self::verify_opening::<Field>(
            &deep_poly_openings.main_trace_polys.proof_2,
            &proof.lde_trace_main_merkle_root,
            i2,
            &deep_poly_openings.main_trace_polys.evaluations_2,
        );
        result &= Self::verify_opening::<Field>(
            &deep_poly_openings.main_trace_polys.proof_3,
            &proof.lde_trace_main_merkle_root,
            i3,
            &deep_poly_openings.main_trace_polys.evaluations_3,
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
                    i0,
                    &precomputed_opening.evaluations,
                );
                result &= Self::verify_opening::<Field>(
                    &precomputed_opening.proof_sym,
                    precomputed_root,
                    i1,
                    &precomputed_opening.evaluations_sym,
                );
                result &= Self::verify_opening::<Field>(
                    &precomputed_opening.proof_2,
                    precomputed_root,
                    i2,
                    &precomputed_opening.evaluations_2,
                );
                result &= Self::verify_opening::<Field>(
                    &precomputed_opening.proof_3,
                    precomputed_root,
                    i3,
                    &precomputed_opening.evaluations_3,
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
                    i0,
                    &aux_trace_polys_opening.evaluations,
                );
                result &= Self::verify_opening::<FieldExtension>(
                    &aux_trace_polys_opening.proof_sym,
                    &aux_root,
                    i1,
                    &aux_trace_polys_opening.evaluations_sym,
                );
                result &= Self::verify_opening::<FieldExtension>(
                    &aux_trace_polys_opening.proof_2,
                    &aux_root,
                    i2,
                    &aux_trace_polys_opening.evaluations_2,
                );
                result &= Self::verify_opening::<FieldExtension>(
                    &aux_trace_polys_opening.proof_3,
                    &aux_root,
                    i3,
                    &aux_trace_polys_opening.evaluations_3,
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
        // Arity-4: composition poly tree has pair leaves (PairKeccak256Backend).
        // The 4-element orbit {4*iota, 4*iota+1, 4*iota+2, 4*iota+3} spans leaves
        // {2*iota, 2*iota+1}. Verify both leaves independently.
        let mut value_01 = deep_poly_openings.composition_poly.evaluations.clone();
        value_01.extend_from_slice(&deep_poly_openings.composition_poly.evaluations_sym);

        let mut value_23 = deep_poly_openings.composition_poly.evaluations_2.clone();
        value_23.extend_from_slice(&deep_poly_openings.composition_poly.evaluations_3);

        deep_poly_openings
            .composition_poly
            .proof
            .verify::<BatchedMerkleTreeBackend<FieldExtension>>(
                composition_poly_merkle_root,
                iota * 2,
                &value_01,
            )
            & deep_poly_openings
                .composition_poly
                .proof_2
                .verify::<BatchedMerkleTreeBackend<FieldExtension>>(
                    composition_poly_merkle_root,
                    iota * 2 + 1,
                    &value_23,
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

    /// Verifies a single FRI layer opening for arity-4 (quad Merkle tree, 4-element leaves).
    ///
    /// `v` is the computed fold value at position `index` within the committed layer.
    /// `siblings` are the prover's claimed values at `index^1`, `index^2`, `index^3`.
    /// `auth_path` proves the 4-element leaf at `index >> 2`.
    fn verify_fri_layer_openings_quad(
        merkle_root: &Commitment,
        auth_path: &Proof<Commitment>,
        v: &FieldElement<FieldExtension>,
        siblings: &[FieldElement<FieldExtension>; 3],
        index: usize,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        // Reconstruct the 4-element leaf from v (at position k = index & 3) and siblings.
        let k = index & 3;
        let mut leaf: [FieldElement<FieldExtension>; 4] = [
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
            FieldElement::zero(),
        ];
        leaf[k] = v.clone();
        leaf[k ^ 1] = siblings[0].clone();
        leaf[k ^ 2] = siblings[1].clone();
        leaf[k ^ 3] = siblings[2].clone();

        auth_path.verify::<FriLayerQuadMerkleTreeBackend<FieldExtension>>(
            merkle_root,
            index >> 2,
            &leaf,
        )
    }

    /// Verify a single arity-4 FRI query.
    ///
    /// - `domain`: the verifier domain, needed to recompute twiddles at each fold level.
    /// - `zetas`: all FRI folding challenges. Layout: [z0, z1] per double-fold layer,
    ///   [z_k] for the odd-extra layer (if any), [z_last] for the final fold.
    /// - `iota`: FRI query index in [0, LDE/4).
    /// - `eval_inv_a`: inverse of the coset element at bit-reversed position `4*iota`.
    /// - `eval_inv_b`: inverse of the coset element at bit-reversed position `4*iota+2`.
    /// - `deep_eval_{0..3}`: DEEP composition poly evaluations at the 4 orbit positions.
    #[allow(clippy::too_many_arguments)]
    fn verify_query_and_sym_openings(
        proof: &StarkProof<Field, FieldExtension, PI>,
        domain: &VerifierDomain<Field>,
        zetas: &[FieldElement<FieldExtension>],
        iota: usize,
        fri_decommitment: &FriDecommitment<FieldExtension>,
        eval_inv_a: FieldElement<Field>,
        eval_inv_b: FieldElement<Field>,
        deep_eval_0: &FieldElement<FieldExtension>,
        deep_eval_1: &FieldElement<FieldExtension>,
        deep_eval_2: &FieldElement<FieldExtension>,
        deep_eval_3: &FieldElement<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let fri_layers_merkle_roots = &proof.fri_layers_merkle_roots;

        // Derive bootstrap type from zeta count.
        // zetas.len() = number_layers = 2*num_double_rounds + has_odd_extra + 1
        let num_double_rounds = (zetas.len() - 1) / 2;

        // Bootstrap: reconstruct layer[0] value from DEEP poly evaluations.
        // Double bootstrap (num_double_rounds >= 1): two binary folds using zetas[0,1].
        //   val_a = (A+B) + inv_a  * z0 * (A-B)   [fold pair {pos0,pos1}]
        //   val_b = (C+D) + inv_b  * z0 * (C-D)   [fold pair {pos2,pos3}]
        //   v     = (val_a+val_b) + inv_a² * z1 * (val_a-val_b)
        // Single bootstrap (num_double_rounds == 0): one binary fold using zetas[0].
        //   v = (A+B) + inv_a * z0 * (A-B)
        //
        // `folds_done` counts binary folds completed so far; used to recompute twiddles.
        let (mut v, mut zeta_idx, mut index, mut layer_eval_inv_a, mut folds_done) =
            if num_double_rounds >= 1 {
                let eval_inv_a_sq = eval_inv_a.square();
                let val_a = (deep_eval_0 + deep_eval_1)
                    + &eval_inv_a * &zetas[0] * (deep_eval_0 - deep_eval_1);
                let val_b = (deep_eval_2 + deep_eval_3)
                    + &eval_inv_b * &zetas[0] * (deep_eval_2 - deep_eval_3);
                let v = (&val_a + &val_b) + &eval_inv_a_sq * &zetas[1] * (&val_a - &val_b);
                (
                    v,
                    2usize,
                    iota,
                    eval_inv_a_sq.square(), // eval_inv_a^4: twiddle for layer[0]→[1] fold
                    2usize,                 // 2 binary folds done in bootstrap
                )
            } else {
                let v = (deep_eval_0 + deep_eval_1)
                    + &eval_inv_a * &zetas[0] * (deep_eval_0 - deep_eval_1);
                (
                    v,
                    1usize,
                    iota * 2,
                    eval_inv_a.square(), // eval_inv_a^2: twiddle for layer[0] fold
                    1usize,              // 1 binary fold done in bootstrap
                )
            };

        if fri_layers_merkle_roots.is_empty() {
            return v == proof.fri_last_value;
        }

        let num_layers = fri_layers_merkle_roots.len();
        let mut result = true;

        for i in 0..num_layers {
            let siblings = &fri_decommitment.layers_evaluations_siblings[i];
            let auth_path = &fri_decommitment.layers_auth_paths[i];
            let merkle_root = &fri_layers_merkle_roots[i];

            result &= Self::verify_fri_layer_openings_quad(merkle_root, auth_path, &v, siblings, index);

            let is_last = i == num_layers - 1;

            // Round (i+1) is a double fold iff it is one of the prover's committed
            // double-fold rounds, i.e. (i+1) < num_double_rounds.
            let fold_is_double = (i + 1) < num_double_rounds;

            if fold_is_double {
                let z_a = &zetas[zeta_idx];
                let z_b = &zetas[zeta_idx + 1];
                zeta_idx += 2;

                let sib_a = &siblings[0]; // index^1
                let sib_b = &siblings[1]; // index^2
                let sib_c = &siblings[2]; // index^3

                let inner_val_a = (&v + sib_a) + &layer_eval_inv_a * z_a * (&v - sib_a);

                // Compute the "b" pair twiddle fresh at this fold level.
                // The "b" pair is {index^2, index^3}; its twiddle position is j_b = (index>>1)^1.
                // twiddle_d[j] = lde_coset_element(reverse_index(2^{d+1}·j, N))^{-2^d}
                // Sign: index^2 is lo (even) when index is even, hi (odd) when index is odd.
                let j_b = (index >> 1) ^ 1;
                let lde_idx_b =
                    reverse_index((1usize << (folds_done + 1)) * j_b, domain.lde_length as u64);
                let base_b = domain.lde_coset_element(lde_idx_b);
                let twiddle_b_unsigned = base_b.pow(1u64 << folds_done).inv().unwrap();
                let layer_eval_inv_b = if index & 1 == 0 {
                    twiddle_b_unsigned
                } else {
                    -twiddle_b_unsigned
                };

                let inner_val_b = (sib_b + sib_c) + layer_eval_inv_b * z_a * (sib_b - sib_c);

                let layer_inv_a_sq = layer_eval_inv_a.square();
                v = (&inner_val_a + &inner_val_b)
                    + &layer_inv_a_sq * z_b * (&inner_val_a - &inner_val_b);

                layer_eval_inv_a = layer_inv_a_sq.square();
                index >>= 2;
                folds_done += 2;
            } else {
                let z = &zetas[zeta_idx];
                zeta_idx += 1;

                let sib_a = &siblings[0];
                v = (&v + sib_a) + &layer_eval_inv_a * z * (&v - sib_a);

                layer_eval_inv_a = layer_eval_inv_a.square();
                index >>= 1;
                folds_done += 1;
            }

            if is_last {
                result &= v == proof.fri_last_value;
            }
        }

        result
    }

    /// Returns 4-tuple of DEEP composition polynomial evaluations for all queries.
    /// Each query's 4-element orbit gives evaluations at positions
    /// {4*iota, 4*iota+1, 4*iota+2, 4*iota+3} in the LDE domain.
    fn reconstruct_deep_composition_poly_evaluations_for_all_queries(
        challenges: &Challenges<FieldExtension>,
        domain: &VerifierDomain<Field>,
        proof: &StarkProof<Field, FieldExtension, PI>,
    ) -> (
        Vec<FieldElement<FieldExtension>>,
        Vec<FieldElement<FieldExtension>>,
        Vec<FieldElement<FieldExtension>>,
        Vec<FieldElement<FieldExtension>>,
    ) {
        let num_queries = challenges.iotas.len();
        let mut evals_0 = Vec::with_capacity(num_queries);
        let mut evals_1 = Vec::with_capacity(num_queries);
        let mut evals_2 = Vec::with_capacity(num_queries);
        let mut evals_3 = Vec::with_capacity(num_queries);

        for (i, iota) in challenges.iotas.iter().enumerate() {
            let primitive_root =
                &Field::get_primitive_root_of_unity(domain.root_order as u64).unwrap();
            let opening = &proof.deep_poly_openings[i];

            // Helper to gather trace evaluations for one orbit position.
            // For preprocessed tables: precomputed columns come FIRST, then multiplicities.
            let gather_trace_evals =
                |main_evals: &[FieldElement<Field>],
                 precomp_evals: Option<&[FieldElement<Field>]>,
                 aux_evals: Option<&[FieldElement<FieldExtension>]>|
                 -> Vec<FieldElement<FieldExtension>> {
                    let mut evals: Vec<FieldElement<FieldExtension>> = Vec::new();
                    if let Some(pe) = precomp_evals {
                        evals.extend(pe.iter().cloned().map(|x| x.to_extension()));
                    }
                    evals.extend(main_evals.iter().cloned().map(|x| x.to_extension()));
                    if let Some(ae) = aux_evals {
                        evals.extend_from_slice(ae);
                    }
                    evals
                };

            let precomp_0 = opening
                .precomputed_trace_polys
                .as_ref()
                .map(|p| p.evaluations.as_slice());
            let precomp_1 = opening
                .precomputed_trace_polys
                .as_ref()
                .map(|p| p.evaluations_sym.as_slice());
            let precomp_2 = opening
                .precomputed_trace_polys
                .as_ref()
                .map(|p| p.evaluations_2.as_slice());
            let precomp_3 = opening
                .precomputed_trace_polys
                .as_ref()
                .map(|p| p.evaluations_3.as_slice());

            let aux_0 = opening.aux_trace_polys.as_ref().map(|a| a.evaluations.as_slice());
            let aux_1 = opening
                .aux_trace_polys
                .as_ref()
                .map(|a| a.evaluations_sym.as_slice());
            let aux_2 = opening.aux_trace_polys.as_ref().map(|a| a.evaluations_2.as_slice());
            let aux_3 = opening.aux_trace_polys.as_ref().map(|a| a.evaluations_3.as_slice());

            let te0 = gather_trace_evals(&opening.main_trace_polys.evaluations, precomp_0, aux_0);
            let te1 =
                gather_trace_evals(&opening.main_trace_polys.evaluations_sym, precomp_1, aux_1);
            let te2 =
                gather_trace_evals(&opening.main_trace_polys.evaluations_2, precomp_2, aux_2);
            let te3 =
                gather_trace_evals(&opening.main_trace_polys.evaluations_3, precomp_3, aux_3);

            let ep0 = Self::query_challenge_to_evaluation_point(*iota, domain);
            let ep1 = {
                let idx = reverse_index(iota * 4 + 1, domain.lde_length as u64);
                domain.lde_coset_element(idx)
            };
            let ep2 = Self::query_challenge_to_evaluation_point_2(*iota, domain);
            let ep3 = {
                let idx = reverse_index(iota * 4 + 3, domain.lde_length as u64);
                domain.lde_coset_element(idx)
            };

            evals_0.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                &ep0,
                primitive_root,
                challenges,
                &te0,
                &opening.composition_poly.evaluations,
            ));
            evals_1.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                &ep1,
                primitive_root,
                challenges,
                &te1,
                &opening.composition_poly.evaluations_sym,
            ));
            evals_2.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                &ep2,
                primitive_root,
                challenges,
                &te2,
                &opening.composition_poly.evaluations_2,
            ));
            evals_3.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                &ep3,
                primitive_root,
                challenges,
                &te3,
                &opening.composition_poly.evaluations_3,
            ));
        }
        (evals_0, evals_1, evals_2, evals_3)
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

        // FRI commit phase — arity-4: 2 zetas per double-fold layer, 1 for odd-extra layer.
        let number_layers = domain.root_order as usize;
        let num_double_rounds = number_layers.saturating_sub(1) / 2;
        let merkle_roots = &proof.fri_layers_merkle_roots;
        let mut zetas = Vec::with_capacity(number_layers);
        for (i, root) in merkle_roots.iter().enumerate() {
            let z1 = transcript.sample_field_element();
            zetas.push(z1);
            if i < num_double_rounds {
                let z2 = transcript.sample_field_element();
                zetas.push(z2);
            }
            transcript.append_bytes(root);
        }

        // >>>> Send final challenge 𝜁ₙ₋₁
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

        // Verify there are enough queries. When LDE <= 4 the query domain is empty
        // and 0 queries is correct; otherwise require the full configured count.
        let query_domain_size = (domain.lde_length as u64) >> 2;
        let expected_queries = if query_domain_size == 0 {
            0
        } else {
            air.options().fri_number_of_queries
        };
        if proof.query_list.len() < expected_queries {
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
}
