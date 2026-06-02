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
    proof::stark::{DeepPolynomialOpening, MultiProof, PolynomialOpenings},
};
use crypto::{fiat_shamir::is_transcript::IsStarkTranscript, merkle_tree::proof::Proof};
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
use std::collections::HashMap;
use std::marker::PhantomData;

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
#[derive(Clone)]
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
        // Precompute g^step once per distinct step to avoid the prior O(B^2)
        // linear scan. A single pass populates a memo and resolves each
        // constraint's step to its point in O(1) amortized.
        let mut step_to_point: HashMap<usize, FieldElement<Field>> = HashMap::new();
        let boundary_points: Vec<FieldElement<Field>> = boundary_constraints
            .constraints
            .iter()
            .map(|c| {
                step_to_point
                    .entry(c.step)
                    .or_insert_with(|| domain.trace_primitive_root.pow(c.step as u64))
                    .clone()
            })
            .collect();

        let main_trace_width = air.trace_layout().0;
        let ood_row = proof.trace_ood_evaluations.get_row(0);

        let (boundary_c_i_evaluations_num, mut boundary_c_i_evaluations_den): (
            Vec<FieldElement<FieldExtension>>,
            Vec<FieldElement<FieldExtension>>,
        ) = boundary_constraints
            .constraints
            .iter()
            .zip(&boundary_points)
            .map(|(c, point)| {
                let column_idx = if c.is_aux {
                    main_trace_width + c.col
                } else {
                    c.col
                };
                let trace_evaluation = &ood_row[column_idx];
                let boundary_zerofier_challenges_z_den = -point + &challenges.z;
                let boundary_quotient_ood_evaluation_num = -&c.value + trace_evaluation;
                (
                    boundary_quotient_ood_evaluation_num,
                    boundary_zerofier_challenges_z_den,
                )
            })
            .unzip();

        // A malformed proof can land `z` on a boundary step, making a denominator zero.
        if FieldElement::inplace_batch_inverse(&mut boundary_c_i_evaluations_den).is_err() {
            return false;
        }

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


    /// Returns the field element element of the domain `domain` corresponding to the given FRI query index challenge `iota`.
    /// Returns the LDE-coset element for FRI query challenge `iota`. The
    /// `sym` flag picks the symmetric counterpart (`iota*2+1`) instead of the
    /// primary index (`iota*2`).
    fn query_challenge_to_evaluation_point(
        iota: usize,
        sym: bool,
        domain: &VerifierDomain<Field>,
    ) -> FieldElement<Field> {
        let raw = iota * 2 + if sym { 1 } else { 0 };
        domain.lde_coset_element(reverse_index(raw, domain.lde_length as u64))
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

    /// Verify both (proof, evaluations) and (proof_sym, evaluations_sym) openings
    /// of a `PolynomialOpenings` against the given `root` at iota positions
    /// `iota*2` and `iota*2 + 1`.
    fn verify_opening_pair<E>(
        opening: &PolynomialOpenings<E>,
        root: &Commitment,
        iota: usize,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<E>: AsBytes + Sync + Send,
        E: IsField,
        Field: IsSubFieldOf<E>,
    {
        Self::verify_opening::<E>(&opening.proof, root, iota * 2, &opening.evaluations)
            && Self::verify_opening::<E>(
                &opening.proof_sym,
                root,
                iota * 2 + 1,
                &opening.evaluations_sym,
            )
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
        // Main trace (multiplicities for preprocessed, full trace for normal).
        let mut ok = Self::verify_opening_pair::<Field>(
            &deep_poly_openings.main_trace_polys,
            &proof.lde_trace_main_merkle_root,
            iota,
        );

        // Precomputed trace (preprocessed tables only). Mismatched presence is
        // unreachable in practice (multi_verify rejects such proofs upstream),
        // but a defensive check keeps this function self-contained.
        ok &= match (
            &proof.lde_trace_precomputed_merkle_root,
            &deep_poly_openings.precomputed_trace_polys,
        ) {
            (Some(root), Some(opening)) => Self::verify_opening_pair::<Field>(opening, root, iota),
            (None, None) => true,
            _ => false,
        };

        // Auxiliary trace.
        ok &= match (
            proof.lde_trace_aux_merkle_root,
            &deep_poly_openings.aux_trace_polys,
        ) {
            (Some(root), Some(opening)) => {
                Self::verify_opening_pair::<FieldExtension>(opening, &root, iota)
            }
            (None, None) => true,
            _ => false,
        };

        ok
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
        challenges
            .iotas
            .iter()
            .zip(&proof.deep_poly_openings)
            .all(|(iota_n, deep_poly_opening)| {
                Self::verify_composition_poly_opening(
                    deep_poly_opening,
                    &proof.composition_poly_root,
                    iota_n,
                ) && Self::verify_trace_openings(proof, deep_poly_opening, *iota_n)
            })
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


    fn reconstruct_deep_composition_poly_evaluations_for_all_queries(
        challenges: &Challenges<FieldExtension>,
        domain: &VerifierDomain<Field>,
        proof: &StarkProof<Field, FieldExtension, PI>,
    ) -> Option<DeepPolynomialEvaluations<FieldExtension>> {
        let num_queries = challenges.iotas.len();
        let mut deep_poly_evaluations = Vec::with_capacity(num_queries);
        let mut deep_poly_evaluations_sym = Vec::with_capacity(num_queries);

        // Build the base-field LDE evaluations as concatenated slice (precomputed + main)
        // without lifting to the extension field. The helper now subtracts directly via
        // the F: IsSubFieldOf<E> Sub impl, so we avoid a per-query base->extension lift.
        let primitive_root = &Field::get_primitive_root_of_unity(domain.root_order as u64)
            .expect("verifier domain root_order is a valid power of two");

        for (i, iota) in challenges.iotas.iter().enumerate() {
            let opening = &proof.deep_poly_openings[i];

            // Base-field portion: precomputed columns FIRST, then main trace columns.
            let mut lde_base: Vec<FieldElement<Field>> = Vec::new();
            if let Some(p) = &opening.precomputed_trace_polys {
                lde_base.extend_from_slice(&p.evaluations);
            }
            lde_base.extend_from_slice(&opening.main_trace_polys.evaluations);

            let lde_aux: &[FieldElement<FieldExtension>] = opening
                .aux_trace_polys
                .as_ref()
                .map(|a| a.evaluations.as_slice())
                .unwrap_or(&[]);

            let evaluation_point = Self::query_challenge_to_evaluation_point(*iota, false, domain);
            deep_poly_evaluations.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                &evaluation_point,
                primitive_root,
                challenges,
                &lde_base,
                lde_aux,
                &opening.composition_poly.evaluations,
            )?);

            // Mirror for the symmetric query point.
            let mut lde_base_sym: Vec<FieldElement<Field>> = Vec::new();
            if let Some(p) = &opening.precomputed_trace_polys {
                lde_base_sym.extend_from_slice(&p.evaluations_sym);
            }
            lde_base_sym.extend_from_slice(&opening.main_trace_polys.evaluations_sym);

            let lde_aux_sym: &[FieldElement<FieldExtension>] = opening
                .aux_trace_polys
                .as_ref()
                .map(|a| a.evaluations_sym.as_slice())
                .unwrap_or(&[]);

            let evaluation_point = Self::query_challenge_to_evaluation_point(*iota, true, domain);
            deep_poly_evaluations_sym.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                &evaluation_point,
                primitive_root,
                challenges,
                &lde_base_sym,
                lde_aux_sym,
                &opening.composition_poly.evaluations_sym,
            )?);
        }
        Some((deep_poly_evaluations, deep_poly_evaluations_sym))
    }

    fn reconstruct_deep_composition_poly_evaluation(
        proof: &StarkProof<Field, FieldExtension, PI>,
        evaluation_point: &FieldElement<Field>,
        primitive_root: &FieldElement<Field>,
        challenges: &Challenges<FieldExtension>,
        lde_trace_base_evaluations: &[FieldElement<Field>],
        lde_trace_aux_evaluations: &[FieldElement<FieldExtension>],
        lde_composition_poly_parts_evaluation: &[FieldElement<FieldExtension>],
    ) -> Option<FieldElement<FieldExtension>> {
        let ood_evaluations_table_height = proof.trace_ood_evaluations.height;
        let ood_evaluations_table_width = proof.trace_ood_evaluations.width;
        let trace_term_coeffs = &challenges.trace_term_coeffs;

        // Runtime guard: a malformed proof may supply opening evaluations whose
        // column count does not match the OOD table width, or whose composition
        // poly parts count does not match the proof's `composition_poly_parts_ood_evaluation`.
        // Without these checks the indexing below would panic in release builds.
        if lde_trace_base_evaluations.len() + lde_trace_aux_evaluations.len()
            != ood_evaluations_table_width
        {
            return None;
        }
        if trace_term_coeffs.is_empty()
            || trace_term_coeffs.len() * trace_term_coeffs[0].len()
                != ood_evaluations_table_height * ood_evaluations_table_width
        {
            return None;
        }

        let mut denoms_trace = Vec::with_capacity(ood_evaluations_table_height);
        let mut current_z = challenges.z.clone();
        for _ in 0..ood_evaluations_table_height {
            denoms_trace.push(evaluation_point - &current_z);
            current_z = primitive_root * &current_z;
        }
        // A malformed proof can land an OOD evaluation point on the LDE coset, reject.
        FieldElement::inplace_batch_inverse(&mut denoms_trace).ok()?;

        let num_base = lde_trace_base_evaluations.len();
        let trace_term = (0..ood_evaluations_table_width)
            .zip(&challenges.trace_term_coeffs)
            .fold(FieldElement::zero(), |trace_terms, (col_idx, coeff_row)| {
                let trace_i = (0..ood_evaluations_table_height).zip(coeff_row).fold(
                    FieldElement::zero(),
                    |trace_t, (row_idx, coeff)| {
                        let ood_val = &proof.trace_ood_evaluations.get_row(row_idx)[col_idx];
                        // Stay in base when we can: F: IsSubFieldOf<E> gives F - E -> E.
                        let diff: FieldElement<FieldExtension> = if col_idx < num_base {
                            &lde_trace_base_evaluations[col_idx] - ood_val
                        } else {
                            &lde_trace_aux_evaluations[col_idx - num_base] - ood_val
                        };
                        let poly_evaluation = diff * &denoms_trace[row_idx];
                        trace_t + &poly_evaluation * coeff
                    },
                );
                trace_terms + trace_i
            });

        let number_of_parts = lde_composition_poly_parts_evaluation.len();
        let z_pow = &challenges.z.pow(number_of_parts);

        // A malformed proof can make evaluation_point == z^N, reject.
        let denom_composition = (evaluation_point - z_pow).inv().ok()?;
        let mut h_terms = FieldElement::zero();
        for (j, h_i_upsilon) in lde_composition_poly_parts_evaluation.iter().enumerate() {
            // Bounds-check via `.get(j)?`: a malformed opening may have more
            // parts than the proof header advertises.
            let h_i_zpower = proof.composition_poly_parts_ood_evaluation.get(j)?;
            let gamma = challenges.gammas.get(j)?;
            let h_i_term = (h_i_upsilon - h_i_zpower) * gamma;
            h_terms += h_i_term;
        }
        h_terms *= denom_composition;

        Some(trace_term + h_terms)
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

        // Phase C + Rounds 2-3 (no FRI) per table, then Phase D batched FRI.
        if !Self::verify_chunks_phase_c_d(
            airs,
            multi_proof,
            transcript,
            &lookup_challenges,
        ) {
            return false;
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

    /// Verify a single-table proof, supplied as a one-element [`MultiProof`]
    /// (the shape returned by `Prover::prove`). The batched-FRI bucket data
    /// lives at the multi-proof level, so single-table verification consumes the
    /// wrapper directly.
    fn verify(
        proof: &MultiProof<Field, FieldExtension, PI>,
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
        PI: Clone,
    {
        Self::multi_verify(&[air], proof, transcript, &FieldElement::zero())
    }

    /// Replays rounds 2 and 3 of the protocol for a given proof, assuming round 1
    /// has already been replayed and the RAP challenges are known. Stops right
    /// after sampling the DEEP gamma coefficients — FRI challenges (zetas, iotas,
    /// grinding) are NOT derived here; they come from the chunk-shared bucket
    /// seed in Phase D. The returned `Challenges` has empty `zetas`/`iotas` and a
    /// zeroed `grinding_seed`, filled in per bucket by the caller.
    fn replay_rounds_2_to_3(
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
        // ==========|   Round 4 (DEEP coeffs only)   |==========
        // ===================================
        // Sample the DEEP gamma coefficients. FRI commit/grinding/query sampling
        // does NOT happen on this per-table fork — it is done per bucket from the
        // chunk-shared seed in Phase D.

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

        Challenges {
            z,
            boundary_coeffs,
            transition_coeffs,
            trace_term_coeffs,
            gammas,
            zetas: Vec::new(),
            iotas: Vec::new(),
            rap_challenges,
            grinding_seed: [0u8; 32],
        }
    }

    fn verify_chunks_phase_c_d(
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        multi_proof: &MultiProof<Field, FieldExtension, PI>,
        transcript: &(impl IsStarkTranscript<FieldExtension, Field> + Clone),
        lookup_challenges: &[FieldElement<FieldExtension>],
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let num_tables = airs.len();
        let pre_fork_transcript = transcript.clone();

        // -- Phase C: per-table Rounds 2-3 (no FRI) --
        let mut domains: Vec<VerifierDomain<Field>> = Vec::with_capacity(num_tables);
        let mut table_challenges: Vec<Challenges<FieldExtension>> = Vec::with_capacity(num_tables);
        for (idx, (air, proof)) in airs.iter().zip(&multi_proof.proofs).enumerate() {
            let domain = new_verifier_domain(*air, proof.trace_length);
            let mut table_transcript = transcript.clone();
            if num_tables > 1 {
                table_transcript.append_bytes(&(idx as u64).to_le_bytes());
            }
            if let Some(root) = proof.lde_trace_aux_merkle_root {
                table_transcript.append_bytes(&root);
            }
            if let Some(ref bpi) = proof.bus_public_inputs {
                table_transcript.append_field_element(&bpi.table_contribution);
            }
            let challenges = Self::replay_rounds_2_to_3(
                *air,
                proof,
                &domain,
                &mut table_transcript,
                lookup_challenges.to_vec(),
            );
            domains.push(domain);
            table_challenges.push(challenges);
        }

        // -- Phase D: per-(chunk, lde_size) batched FRI --
        let k = (multi_proof.chunk_size as usize).max(1);
        let expected_num_chunks = num_tables.div_ceil(k);
        if multi_proof.fri_chunk_buckets.len() != expected_num_chunks {
            error!("fri_chunk_buckets chunk count mismatch");
            return false;
        }
        for (chunk_idx, chunk_start) in (0..num_tables).step_by(k).enumerate() {
            let chunk_end = (chunk_start + k).min(num_tables);
            let chunk_size = chunk_end - chunk_start;
            let bucket_seed =
                Self::build_bucket_seed(&pre_fork_transcript, multi_proof, chunk_start, chunk_size);

            let mut bucket_members: Vec<Vec<usize>> = Vec::new();
            let mut bucket_lde_sizes: Vec<usize> = Vec::new();
            for j in 0..chunk_size {
                let lde_size = domains[chunk_start + j].lde_length;
                match bucket_lde_sizes.iter().position(|&s| s == lde_size) {
                    Some(b) => bucket_members[b].push(j),
                    None => {
                        bucket_lde_sizes.push(lde_size);
                        bucket_members.push(vec![j]);
                    }
                }
            }

            let proof_buckets = &multi_proof.fri_chunk_buckets[chunk_idx];
            if proof_buckets.len() != bucket_members.len() {
                error!("chunk {chunk_idx}: bucket count mismatch");
                return false;
            }
            for (b, (members, &lde_size)) in
                bucket_members.iter().zip(bucket_lde_sizes.iter()).enumerate()
            {
                if !Self::verify_one_bucket(
                    airs,
                    multi_proof,
                    &domains,
                    &table_challenges,
                    &bucket_seed,
                    chunk_start,
                    chunk_idx,
                    b,
                    members,
                    lde_size,
                ) {
                    return false;
                }
            }
        }
        true
    }

    /// Build the chunk-shared `bucket_seed`. Byte order MUST match the prover:
    /// pre-fork shared state, then for each chunk-local index (ascending):
    /// table_contribution (if any), then composition_poly_root, then OOD evals
    /// (all trace_ood columns column-major, then composition_poly_parts_ood).
    fn build_bucket_seed<T>(
        pre_fork_transcript: &T,
        multi_proof: &MultiProof<Field, FieldExtension, PI>,
        chunk_start: usize,
        chunk_size: usize,
    ) -> T
    where
        T: IsStarkTranscript<FieldExtension, Field> + Clone,
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let mut bucket_seed = pre_fork_transcript.clone();
        for j in 0..chunk_size {
            let proof = &multi_proof.proofs[chunk_start + j];
            if let Some(ref bpi) = proof.bus_public_inputs {
                bucket_seed.append_field_element(&bpi.table_contribution);
            }
        }
        for j in 0..chunk_size {
            let proof = &multi_proof.proofs[chunk_start + j];
            bucket_seed.append_bytes(&proof.composition_poly_root);
        }
        for j in 0..chunk_size {
            let proof = &multi_proof.proofs[chunk_start + j];
            for col in proof.trace_ood_evaluations.columns().iter() {
                for elem in col.iter() {
                    bucket_seed.append_field_element(elem);
                }
            }
            for elem in proof.composition_poly_parts_ood_evaluation.iter() {
                bucket_seed.append_field_element(elem);
            }
        }
        bucket_seed
    }

    /// Verify one (chunk, lde_size) bucket: derive its FRI challenges from the
    /// chunk-shared `bucket_seed`, reconstruct each member's DEEP evaluations at
    /// the bucket-shared iotas, combine them with `delta_fri` powers, verify the
    /// batched FRI fold, and verify each member's per-table openings.
    #[allow(clippy::too_many_arguments)]
    fn verify_one_bucket(
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        multi_proof: &MultiProof<Field, FieldExtension, PI>,
        domains: &[VerifierDomain<Field>],
        table_challenges: &[Challenges<FieldExtension>],
        bucket_seed: &(impl IsStarkTranscript<FieldExtension, Field> + Clone),
        chunk_start: usize,
        chunk_idx: usize,
        b: usize,
        members: &[usize],
        lde_size: usize,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let bucket = &multi_proof.fri_chunk_buckets[chunk_idx][b];
        if bucket.members != *members || bucket.lde_size as usize != lde_size {
            error!("chunk {chunk_idx} bucket {b}: members/lde_size mismatch");
            return false;
        }

        let leader_idx = chunk_start + members[0];
        let leader_air = airs[leader_idx];
        let leader_domain = &domains[leader_idx];

        let mut bt = bucket_seed.clone();
        bt.append_bytes(&(lde_size as u64).to_le_bytes());
        let delta_fri: FieldElement<FieldExtension> = bt.sample_field_element();

        let mut zetas = bucket
            .layer_roots
            .iter()
            .map(|root| {
                let element = bt.sample_field_element();
                bt.append_bytes(root);
                element
            })
            .collect::<Vec<FieldElement<FieldExtension>>>();
        zetas.push(bt.sample_field_element());
        bt.append_field_element(&bucket.last_value);

        let security_bits = leader_air.context().proof_options.grinding_factor;
        if security_bits > 0 {
            let grinding_seed = bt.state();
            let nonce_is_valid = bucket.nonce.is_some_and(|nonce_value| {
                grinding::is_valid_nonce(&grinding_seed, nonce_value, security_bits)
            });
            if !nonce_is_valid {
                #[cfg(not(feature = "test_fiat_shamir"))]
                error!("chunk {chunk_idx} bucket {b}: grinding factor not satisfied");
                return false;
            }
            if let Some(nonce_value) = bucket.nonce {
                bt.append_bytes(&nonce_value.to_be_bytes());
            }
        }

        let number_of_queries = leader_air.options().fri_number_of_queries;
        let iotas = Self::sample_query_indexes(number_of_queries, leader_domain, &mut bt);
        if bucket.decommitments.len() < number_of_queries {
            error!("chunk {chunk_idx} bucket {b}: too few FRI decommitments");
            return false;
        }

        let mut evaluation_point_inverse = iotas
            .iter()
            .map(|iota| Self::query_challenge_to_evaluation_point(*iota, false, leader_domain))
            .collect::<Vec<FieldElement<Field>>>();
        if FieldElement::inplace_batch_inverse(&mut evaluation_point_inverse).is_err() {
            error!("chunk {chunk_idx} bucket {b}: zero FRI evaluation point");
            return false;
        }

        let num_queries = iotas.len();
        let mut combined_eval: Vec<FieldElement<FieldExtension>> =
            vec![FieldElement::zero(); num_queries];
        let mut combined_eval_sym: Vec<FieldElement<FieldExtension>> =
            vec![FieldElement::zero(); num_queries];
        let mut delta_power = FieldElement::<FieldExtension>::one();
        for (i_local, &j) in members.iter().enumerate() {
            let idx = chunk_start + j;
            let air = airs[idx];
            let proof = &multi_proof.proofs[idx];
            let domain = &domains[idx];

            let mut challenges = table_challenges[idx].clone();
            challenges.iotas = iotas.clone();
            challenges.zetas = zetas.clone();

            if !Self::step_2_verify_claimed_composition_polynomial(air, proof, domain, &challenges) {
                #[cfg(not(feature = "test_fiat_shamir"))]
                error!("chunk {chunk_idx} bucket {b}: table {idx} composition poly failed");
                return false;
            }

            let (member_eval, member_eval_sym) =
                match Self::reconstruct_deep_composition_poly_evaluations_for_all_queries(
                    &challenges,
                    domain,
                    proof,
                ) {
                    Some(pair) => pair,
                    None => {
                        error!("chunk {chunk_idx} bucket {b}: table {idx} DEEP reconstruct failed");
                        return false;
                    }
                };

            if !Self::step_4_verify_trace_and_composition_openings(proof, &challenges) {
                #[cfg(not(feature = "test_fiat_shamir"))]
                error!("chunk {chunk_idx} bucket {b}: table {idx} openings failed");
                return false;
            }

            if i_local == 0 {
                combined_eval = member_eval;
                combined_eval_sym = member_eval_sym;
            } else {
                for q in 0..num_queries {
                    combined_eval[q] = &combined_eval[q] + &delta_power * &member_eval[q];
                    combined_eval_sym[q] =
                        &combined_eval_sym[q] + &delta_power * &member_eval_sym[q];
                }
            }
            delta_power = &delta_power * &delta_fri;
        }

        for (q, ((iota, decommitment), eval_point_inv)) in iotas
            .iter()
            .zip(&bucket.decommitments)
            .zip(evaluation_point_inverse)
            .enumerate()
        {
            if !Self::verify_bucket_fri_query(
                &bucket.layer_roots,
                &bucket.last_value,
                &zetas,
                *iota,
                decommitment,
                eval_point_inv,
                &combined_eval[q],
                &combined_eval_sym[q],
            ) {
                #[cfg(not(feature = "test_fiat_shamir"))]
                error!("chunk {chunk_idx} bucket {b}: FRI query {q} failed");
                return false;
            }
        }
        true
    }

    /// Verify a single batched-FRI query for a bucket. The combined DEEP
    /// evaluations `D(𝜐)` / `D(-𝜐)` (already linearly combined across
    /// bucket-mates with `delta_fri` powers) are folded against the bucket's
    /// `layer_roots` / `last_value`. MMCS-free port of the per-table FRI query
    /// verification.
    #[allow(clippy::too_many_arguments)]
    fn verify_bucket_fri_query(
        layer_roots: &[Commitment],
        last_value: &FieldElement<FieldExtension>,
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
        let evaluation_point_vec: Vec<FieldElement<Field>> =
            core::iter::successors(Some(evaluation_point_inv.square()), |evaluation_point| {
                Some(evaluation_point.square())
            })
            .take(layer_roots.len())
            .collect();

        let p0_eval = deep_composition_evaluation;
        let p0_eval_sym = deep_composition_evaluation_sym;

        let mut v =
            (p0_eval + p0_eval_sym) + evaluation_point_inv * &zetas[0] * (p0_eval - p0_eval_sym);
        let mut index = iota;

        if layer_roots.is_empty() {
            return v == *last_value;
        }

        layer_roots
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
                    let openings_ok = Self::verify_fri_layer_openings(
                        merkle_root,
                        auth_path_sym,
                        &v,
                        evaluation_sym,
                        index,
                    );
                    v = (&v + evaluation_sym)
                        + evaluation_point_inv * &zetas[i + 1] * (&v - evaluation_sym);
                    index >>= 1;
                    if i < fri_decommitment.layers_evaluations_sym.len() - 1 {
                        result & openings_ok
                    } else {
                        result & (v == *last_value) & openings_ok
                    }
                },
            )
    }
}
