use super::{
    config::BatchedMerkleTreeBackend, domain::Domain, fri::fri_decommit::FriDecommitment, grinding,
    proof::stark::StarkProof, traits::AIR,
};
use crate::{
    config::Commitment, domain::new_domain, frame::Frame, proof::stark::DeepPolynomialOpening,
    traits::TransitionEvaluationContext,
};
use crypto::{fiat_shamir::is_transcript::IsStarkTranscript, merkle_tree::proof::Proof};
#[cfg(not(feature = "test_fiat_shamir"))]
use log::error;
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

type AirAndProof<'a, Field, FieldExtension, PI> = (
    &'a dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
    &'a StarkProof<Field, FieldExtension>,
);

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
    FieldExtension: Send + Sync + IsFFTField,
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
        domain: &Domain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Vec<usize> {
        let domain_size = domain.lde_roots_of_unity_coset.len() as u64;
        (0..number_of_queries)
            .map(|_| (transcript.sample_u64(domain_size >> 1)) as usize)
            .collect::<Vec<usize>>()
    }

    /// Returns the list of challenges sent to the prover.
    fn step_1_replay_rounds_and_recover_challenges(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &StarkProof<Field, FieldExtension>,
        domain: &Domain<Field>,
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

        // Use new simplified constraint system for constraint count
        let constraints = air.constraints(&rap_challenges);
        let num_transition =
            constraints.degree_1.len() + constraints.degree_2.len() + constraints.degree_3.len();
        let num_boundary = constraints.boundary.len();
        let num_constraints = num_transition + num_boundary;

        let coefficients: Vec<_> = (0..num_constraints).map(|i| beta.pow(i)).collect();

        // For backward compatibility with Challenges struct, split into transition and boundary
        // New system orders: degree_1, degree_2, degree_3, then boundary
        let transition_coeffs: Vec<_> = coefficients[..num_transition].to_vec();
        let boundary_coeffs: Vec<_> = coefficients[num_transition..].to_vec();

        // <<<< Receive commitments: [H₁], [H₂]
        transcript.append_bytes(&proof.composition_poly_root);

        // ===================================
        // ==========|   Round 3   |==========
        // ===================================

        // >>>> Send challenge: z
        let z = transcript.sample_z_ood(
            &domain.lde_roots_of_unity_coset,
            &domain.trace_roots_of_unity,
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
        proof: &StarkProof<Field, FieldExtension>,
        domain: &Domain<Field>,
        challenges: &Challenges<FieldExtension>,
    ) -> bool {
        // Use legacy evaluation path for Stone prover compatibility
        if air.use_legacy_evaluator() {
            return Self::step_2_verify_legacy(air, proof, domain, challenges);
        }

        let constraints = air.constraints(&challenges.rap_challenges);
        let trace_length = air.trace_length();
        let z = &challenges.z;
        let g = &domain.trace_primitive_root;
        let n = trace_length;

        // Build frame from OOD evaluations
        let num_main_trace_columns =
            proof.trace_ood_evaluations.width - air.num_auxiliary_rap_columns();
        let ood_frame =
            (proof.trace_ood_evaluations).into_frame(num_main_trace_columns, air.step_size());

        // Compute values at z
        let zerofier_z = z.pow(n) - FieldElement::<FieldExtension>::one();
        let zerofier_inv_z = zerofier_z.inv().unwrap();
        let trace_deg = n - 1;
        let x_td = z.pow(trace_deg);
        let x_2td = z.pow(2 * trace_deg);

        let max_degree = constraints.max_degree();
        let legacy_eval = constraints.use_legacy_evaluation;
        let legacy_ordering = constraints.use_legacy_ordering;
        let mut acc = FieldElement::<FieldExtension>::zero();

        // Helper to compute exemption polynomial at z: Π_{i=0}^{k-1} (z - g^{n-1-i})
        let compute_exemption_at_z = |end_exemptions: usize| -> FieldElement<FieldExtension> {
            (0..end_exemptions).fold(FieldElement::one(), |prod, i| {
                let omega_exp = g.pow((n - 1 - i) as u64);
                prod * (z - &omega_exp.to_extension())
            })
        };

        // Helper to evaluate a transition constraint
        let eval_transition =
            |c: &crate::constraints::simple::TransitionConstraint<Field, FieldExtension>,
             beta: &FieldElement<FieldExtension>,
             degree: usize|
             -> FieldElement<FieldExtension> {
                let c_eval: FieldElement<FieldExtension> = (c.evaluate_ext)(&ood_frame);
                let exempt_z = compute_exemption_at_z(c.end_exemptions);
                let corrected = if legacy_eval {
                    &exempt_z * &c_eval
                } else {
                    match (max_degree, degree) {
                        (3, 1) => &exempt_z * &x_2td * &c_eval,
                        (2, 1) | (3, 2) => &exempt_z * &x_td * &c_eval,
                        _ => &exempt_z * &c_eval,
                    }
                };
                beta * &corrected
            };

        // Helper to evaluate a boundary constraint
        let n_fe = FieldElement::<FieldExtension>::from(n as u64);
        let eval_boundary =
            |bc: &crate::constraints::simple::BoundaryConstraint<FieldExtension>,
             beta: &FieldElement<FieldExtension>|
             -> FieldElement<FieldExtension> {
                let omega_j = g.pow(bc.row as u64);
                let omega_j_ext: FieldElement<FieldExtension> = omega_j.to_extension();

                // Get trace value at z from OOD evaluations
                let trace_val: FieldElement<FieldExtension> = if bc.is_aux {
                    let col_idx = num_main_trace_columns + bc.col;
                    proof.trace_ood_evaluations.get_row(0)[col_idx].clone()
                } else {
                    proof.trace_ood_evaluations.get_row(0)[bc.col].clone()
                };

                let c_eval = &trace_val - &bc.value;

                let corrected = if legacy_eval {
                    // Legacy: simple zerofier 1/(z - g^row)
                    let zerofier_inv = (z - &omega_j_ext).inv().unwrap();
                    &zerofier_inv * &c_eval
                } else {
                    // New: Lagrange basis with degree correction
                    let omega_neg_j = g.pow((n - bc.row) as u64);
                    let omega_neg_j_ext: FieldElement<FieldExtension> = omega_neg_j.to_extension();
                    let n_inv = n_fe.inv().unwrap();
                    let denom = (z - &omega_j_ext) * &omega_neg_j_ext * &n_inv;
                    let lagrange_z = &zerofier_z * denom.inv().unwrap();
                    match max_degree {
                        3 => &lagrange_z * &x_td * &c_eval,
                        _ => &lagrange_z * &c_eval,
                    }
                };
                beta * &corrected
            };

        if legacy_ordering {
            // Legacy ordering: transition first, then boundary
            let mut beta_idx = 0;

            for c in &constraints.degree_1 {
                acc = acc + eval_transition(c, &challenges.transition_coeffs[beta_idx], 1);
                beta_idx += 1;
            }
            for c in &constraints.degree_2 {
                acc = acc + eval_transition(c, &challenges.transition_coeffs[beta_idx], 2);
                beta_idx += 1;
            }
            for c in &constraints.degree_3 {
                acc = acc + eval_transition(c, &challenges.transition_coeffs[beta_idx], 3);
                beta_idx += 1;
            }
            for (bc_idx, bc) in constraints.boundary.iter().enumerate() {
                acc = acc + eval_boundary(bc, &challenges.boundary_coeffs[bc_idx]);
            }
        } else {
            // New ordering: degree_1, degree_2, degree_3, boundary
            let mut beta_idx = 0;

            for c in &constraints.degree_1 {
                acc = acc + eval_transition(c, &challenges.transition_coeffs[beta_idx], 1);
                beta_idx += 1;
            }
            for c in &constraints.degree_2 {
                acc = acc + eval_transition(c, &challenges.transition_coeffs[beta_idx], 2);
                beta_idx += 1;
            }
            for c in &constraints.degree_3 {
                acc = acc + eval_transition(c, &challenges.transition_coeffs[beta_idx], 3);
                beta_idx += 1;
            }
            for (bc_idx, bc) in constraints.boundary.iter().enumerate() {
                acc = acc + eval_boundary(bc, &challenges.boundary_coeffs[bc_idx]);
            }
        }

        // Divide by zerofier
        let composition_poly_ood_evaluation = &zerofier_inv_z * acc;

        let composition_poly_claimed_ood_evaluation = proof
            .composition_poly_parts_ood_evaluation
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, coeff| acc * z + coeff);

        composition_poly_claimed_ood_evaluation == composition_poly_ood_evaluation
    }

    /// Legacy verification path for Stone prover compatibility.
    /// Uses the old transition_constraints and boundary_constraints methods.
    fn step_2_verify_legacy(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &StarkProof<Field, FieldExtension>,
        domain: &Domain<Field>,
        challenges: &Challenges<FieldExtension>,
    ) -> bool {
        let trace_length = air.trace_length();
        let z = &challenges.z;
        let g = &domain.trace_primitive_root;

        // Build frame from OOD evaluations (all elements in extension field)
        let num_main_trace_columns =
            proof.trace_ood_evaluations.width - air.num_auxiliary_rap_columns();
        let ood_frame: Frame<FieldExtension, FieldExtension> = proof
            .trace_ood_evaluations
            .into_frame(num_main_trace_columns, air.step_size());

        // Compute transition constraint evaluations
        let evaluation_context = TransitionEvaluationContext::new_verifier(
            &ood_frame,
            &[], // No periodic values at OOD point
            &challenges.rap_challenges,
        );
        let transition_evaluations = air.compute_transition(&evaluation_context);

        // For each transition constraint, compute the zerofier quotient at z
        let mut transition_acc = FieldElement::<FieldExtension>::zero();
        for (idx, constraint) in air.transition_constraints().iter().enumerate() {
            let eval = &transition_evaluations[idx];
            let end_exemptions = constraint.end_exemptions();

            // Compute exemption polynomial at z: Π_{i=0}^{k-1} (z - g^{n-1-i})
            let mut exempt_z = FieldElement::<FieldExtension>::one();
            for i in 0..end_exemptions {
                let omega_exp = g.pow((trace_length - 1 - i) as u64);
                exempt_z = exempt_z * (z - &omega_exp.to_extension());
            }

            // Zerofier for transition: (z^n - 1) / exemption_poly
            // So quotient = eval * exempt_z / (z^n - 1)
            // But old system multiplies eval by exempt_z / zerofier directly
            let zerofier_z = z.pow(trace_length) - FieldElement::<FieldExtension>::one();
            let quotient = eval * &exempt_z * zerofier_z.inv().unwrap();

            transition_acc = transition_acc + &challenges.transition_coeffs[idx] * &quotient;
        }

        // Compute boundary constraint evaluations
        let boundary_constraints = air.boundary_constraints(&challenges.rap_challenges);
        let mut boundary_acc = FieldElement::<FieldExtension>::zero();
        for (idx, bc) in boundary_constraints.constraints.iter().enumerate() {
            let omega_j = g.pow(bc.step as u64);
            let omega_j_ext: FieldElement<FieldExtension> = omega_j.to_extension();

            // Get trace value at z from OOD evaluations
            let trace_val: FieldElement<FieldExtension> = if bc.is_aux {
                let col_idx = num_main_trace_columns + bc.col;
                proof.trace_ood_evaluations.get_row(0)[col_idx].clone()
            } else {
                proof.trace_ood_evaluations.get_row(0)[bc.col].clone()
            };

            let c_eval = &trace_val - &bc.value;
            // Zerofier inverse: 1 / (z - g^row)
            let zerofier_inv = (z - &omega_j_ext).inv().unwrap();
            let quotient = &zerofier_inv * &c_eval;

            boundary_acc = boundary_acc + &challenges.boundary_coeffs[idx] * &quotient;
        }

        // Total composition polynomial evaluation at z
        let composition_poly_ood_evaluation = transition_acc + boundary_acc;

        // Compare with claimed evaluation
        let composition_poly_claimed_ood_evaluation = proof
            .composition_poly_parts_ood_evaluation
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, coeff| acc * z + coeff);

        composition_poly_claimed_ood_evaluation == composition_poly_ood_evaluation
    }

    /// Reconstructs the Deep composition polynomial evaluations at the challenge indices values using the provided
    /// openings of the trace polynomials and the composition polynomial parts. It then uses these to verify that the
    /// FRI decommitments are valid and correspond to the Deep composition polynomial.
    fn step_3_verify_fri(
        proof: &StarkProof<Field, FieldExtension>,
        domain: &Domain<Field>,
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
        domain: &Domain<Field>,
    ) -> FieldElement<Field> {
        domain.lde_roots_of_unity_coset
            [reverse_index(iota * 2, domain.lde_roots_of_unity_coset.len() as u64)]
        .clone()
    }

    /// Returns the symmetric field element element of the domain `domain` corresponding to the given FRI query index challenge `iota`.
    fn query_challenge_to_evaluation_point_sym(
        iota: usize,
        domain: &Domain<Field>,
    ) -> FieldElement<Field> {
        domain.lde_roots_of_unity_coset
            [reverse_index(iota * 2 + 1, domain.lde_roots_of_unity_coset.len() as u64)]
        .clone()
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
        proof: &StarkProof<Field, FieldExtension>,
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
        proof: &StarkProof<Field, FieldExtension>,
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
        proof: &StarkProof<Field, FieldExtension>,
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
        domain: &Domain<Field>,
        proof: &StarkProof<Field, FieldExtension>,
    ) -> DeepPolynomialEvaluations<FieldExtension> {
        let mut deep_poly_evaluations = Vec::new();
        let mut deep_poly_evaluations_sym = Vec::new();
        for (i, iota) in challenges.iotas.iter().enumerate() {
            let primitive_root =
                &Field::get_primitive_root_of_unity(domain.root_order as u64).unwrap();

            let mut evaluations: Vec<FieldElement<FieldExtension>> = proof.deep_poly_openings[i]
                .main_trace_polys
                .evaluations
                .clone()
                .into_iter()
                .map(|x| x.to_extension())
                .collect();
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

            let mut evaluations_sym: Vec<FieldElement<FieldExtension>> = proof.deep_poly_openings
                [i]
                .main_trace_polys
                .evaluations_sym
                .clone()
                .into_iter()
                .map(|x| x.to_extension())
                .collect();
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
        proof: &StarkProof<Field, FieldExtension>,
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

        let mut denoms_trace = (0..ood_evaluations_table_height)
            .map(|row_idx| evaluation_point - primitive_root.pow(row_idx as u64) * &challenges.z)
            .collect::<Vec<FieldElement<FieldExtension>>>();
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
    /// The function replays Round 1 for all proofs first (to match the prover's transcript state),
    /// then verifies each proof sequentially.
    ///
    /// Warning: the transcript must be safely initialized before passing it to this method.
    fn multi_verify(
        airs_and_proofs: &[AirAndProof<'_, Field, FieldExtension, PI>],
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        // First, replay round 1 for all tables so the transcript state matches the prover,
        // which commits to all traces before sampling any further randomness.
        let mut rap_challenges_vec = Vec::new();
        for (air, proof) in airs_and_proofs {
            let rap_challenges = Self::replay_round_1(*air, proof, transcript);
            rap_challenges_vec.push(rap_challenges);
        }

        // Verify each proof
        for ((air, proof), rap_challenges) in airs_and_proofs.iter().zip(rap_challenges_vec) {
            if !Self::verify_rounds_2_to_4(*air, proof, transcript, rap_challenges) {
                return false;
            }
        }
        true
    }

    /// Verify a single STARK proof.
    /// This is equivalent to calling `multi_verify` with a single-element slice.
    fn verify(
        proof: &StarkProof<Field, FieldExtension>,
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        Self::multi_verify(&[(air, proof)], transcript)
    }

    /// Replays round 1 of the protocol for a given proof, appending the main and auxiliary
    /// trace commitments to the transcript and returning the RAP challenges.
    fn replay_round_1(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &StarkProof<Field, FieldExtension>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Vec<FieldElement<FieldExtension>>
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

        rap_challenges
    }

    /// Replays rounds 2, 3 and 4 of the protocol for a given proof, assuming round 1 has
    /// already been replayed and the RAP challenges are known.
    fn replay_rounds_after_round_1(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &StarkProof<Field, FieldExtension>,
        domain: &Domain<Field>,
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

        // Use new simplified constraint system for constraint count
        let constraints = air.constraints(&rap_challenges);
        let num_transition =
            constraints.degree_1.len() + constraints.degree_2.len() + constraints.degree_3.len();
        let num_boundary = constraints.boundary.len();
        let num_constraints = num_transition + num_boundary;

        let coefficients: Vec<_> = (0..num_constraints).map(|i| beta.pow(i)).collect();

        // New system orders: degree_1, degree_2, degree_3, then boundary
        let transition_coeffs: Vec<_> = coefficients[..num_transition].to_vec();
        let boundary_coeffs: Vec<_> = coefficients[num_transition..].to_vec();

        // <<<< Receive commitments: [H₁], [H₂]
        transcript.append_bytes(&proof.composition_poly_root);

        // ===================================
        // ==========|   Round 3   |==========
        // ===================================

        // >>>> Send challenge: z
        let z = transcript.sample_z_ood(
            &domain.lde_roots_of_unity_coset,
            &domain.trace_roots_of_unity,
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
        proof: &StarkProof<Field, FieldExtension>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        rap_challenges: Vec<FieldElement<FieldExtension>>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let domain = new_domain(air);

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
}
