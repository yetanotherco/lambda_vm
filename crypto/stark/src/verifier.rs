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

    /// Reconstruct the per-table DEEP composition evaluations `D_i(iota)` and
    /// `D_i(-iota)` for ONE table at every query index. Used by the
    /// chunk-bucket FRI verification (Phase D) to combine bucket-mates
    /// into the polynomial actually committed by FRI.
    fn reconstruct_d_evaluations_for_table(
        proof: &StarkProof<Field, FieldExtension, PI>,
        domain: &VerifierDomain<Field>,
        challenges: &Challenges<FieldExtension>,
    ) -> Option<DeepPolynomialEvaluations<FieldExtension>> {
        Self::reconstruct_deep_composition_poly_evaluations_for_all_queries(
            challenges, domain, proof,
        )
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

    /// Verify the main MMCS opening + precomputed + aux openings at FRI
    /// challenge `iota`. `main_*` and `aux_*` come from the surrounding
    /// multi-proof. Aux is `None` when no AIR in the multi-proof has an
    /// aux trace.
    #[allow(clippy::too_many_arguments)]
    fn verify_trace_openings(
        proof: &StarkProof<Field, FieldExtension, PI>,
        deep_poly_openings: &DeepPolynomialOpening<Field, FieldExtension>,
        iota: usize,
        main_tag: crypto::merkle_tree::mmcs::MatrixTag,
        main_mmcs_root: Option<&Commitment>,
        main_mmcs_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)],
        aux_mmcs_root: Option<&Commitment>,
        aux_mmcs_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)],
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + Sync + Send + math::traits::ByteConversion,
    {
        use crate::proof::stark::MainTraceOpening;
        let main_ok = match &deep_poly_openings.main_trace_polys {
            MainTraceOpening::Mmcs { .. } => Self::verify_main_mmcs_pair(
                &deep_poly_openings.main_trace_polys,
                iota,
                main_tag,
                main_mmcs_root,
                main_mmcs_spec,
            ),
            MainTraceOpening::Tree(opening) => match &proof.lde_trace_main_merkle_root {
                Some(root) => Self::verify_opening_pair::<Field>(opening, root, iota),
                None => false,
            },
        };
        let mut ok = main_ok;

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

        // Auxiliary trace: shared MMCS opening for tables with aux, or
        // None when this AIR has no aux at all.
        ok &= match (&deep_poly_openings.aux_trace_polys, aux_mmcs_root) {
            (Some(opening), Some(root)) => verify_aux_mmcs_pair_inner::<FieldExtension>(
                opening, iota, main_tag, root, aux_mmcs_spec,
            ),
            (None, _) => true,
            (Some(_), None) => false,
        };

        ok
    }

    /// Authenticate the main-trace MMCS pair for one query.
    fn verify_main_mmcs_pair(
        main_opening: &crate::proof::stark::MainTraceOpening<Field>,
        iota: usize,
        main_tag: crypto::merkle_tree::mmcs::MatrixTag,
        main_mmcs_root: Option<&Commitment>,
        main_mmcs_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)],
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send + math::traits::ByteConversion,
    {
        verify_main_mmcs_pair_inner::<Field>(
            main_opening,
            iota,
            main_tag,
            main_mmcs_root,
            main_mmcs_spec,
        )
    }

    /// Verify opening Open(Hᵢ(D_LDE), 𝜐) and Open(Hᵢ(D_LDE), -𝜐) for all parts Hᵢof the composition
    /// polynomial, where 𝜐 and -𝜐 are the elements corresponding to the index challenge `iota`.
    /// Verify the composition-trace MMCS opening pair for one query.
    /// Rehashes the row-pair leaf using the COMPOSITION domain
    /// separator, checks it matches `matrix_leaves[table_idx]`, and
    /// authenticates against the chunk's composition root + spec.
    fn verify_composition_poly_opening(
        deep_poly_openings: &DeepPolynomialOpening<Field, FieldExtension>,
        comp_mmcs_root: Option<&Commitment>,
        comp_mmcs_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)],
        main_tag: crypto::merkle_tree::mmcs::MatrixTag,
        iota: usize,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send + math::traits::ByteConversion,
    {
        verify_comp_mmcs_pair_inner::<FieldExtension>(
            &deep_poly_openings.composition_poly,
            iota,
            main_tag,
            comp_mmcs_root,
            comp_mmcs_spec,
        )
    }

    /// Verifies the validity of the purported values of the trace polynomials and the composition polynomial
    /// parts at the domain elements and their symmetric counterparts corresponding to all the FRI query
    /// index challenges.
    #[allow(clippy::too_many_arguments)]
    fn step_4_verify_trace_and_composition_openings(
        proof: &StarkProof<Field, FieldExtension, PI>,
        challenges: &Challenges<FieldExtension>,
        main_tag: crypto::merkle_tree::mmcs::MatrixTag,
        main_mmcs_root: Option<&Commitment>,
        main_mmcs_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)],
        aux_mmcs_root: Option<&Commitment>,
        aux_mmcs_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)],
        comp_mmcs_root: Option<&Commitment>,
        comp_mmcs_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)],
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + Sync + Send + math::traits::ByteConversion,
    {
        challenges
            .iotas
            .iter()
            .zip(&proof.deep_poly_openings)
            .all(|(iota_n, deep_poly_opening)| {
                Self::verify_composition_poly_opening(
                    deep_poly_opening,
                    comp_mmcs_root,
                    comp_mmcs_spec,
                    main_tag,
                    *iota_n,
                ) && Self::verify_trace_openings(
                    proof,
                    deep_poly_opening,
                    *iota_n,
                    main_tag,
                    main_mmcs_root,
                    main_mmcs_spec,
                    aux_mmcs_root,
                    aux_mmcs_spec,
                )
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

    /// Verify a single bucket-FRI query.
    ///
    /// `fri_layers_merkle_roots` / `fri_last_value` come from the bucket
    /// (`ChunkBucketFri`), not from any per-table proof. `deep_composition_*`
    /// is `D_combined(±iota)` — the linear combination of bucket-mates'
    /// reconstructed D_i evaluations with successive powers of `delta_fri`.
    #[allow(clippy::too_many_arguments)]
    fn verify_bucket_fri_query(
        fri_layers_merkle_roots: &[Commitment],
        fri_last_value: &FieldElement<FieldExtension>,
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
            .take(fri_layers_merkle_roots.len())
            .collect();

        let p0_eval = deep_composition_evaluation;
        let p0_eval_sym = deep_composition_evaluation_sym;

        // Reconstruct p₁(𝜐²)
        let mut v =
            (p0_eval + p0_eval_sym) + evaluation_point_inv * &zetas[0] * (p0_eval - p0_eval_sym);
        let mut index = iota;

        // 0-layer FRI (trivially small LDE): folded p0 must equal the bucket's last_value.
        if fri_layers_merkle_roots.is_empty() {
            return v == *fri_last_value;
        }

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
                    let openings_ok = Self::verify_fri_layer_openings(
                        merkle_root,
                        auth_path_sym,
                        &v,
                        evaluation_sym,
                        index,
                    );

                    v = (&v + evaluation_sym) + evaluation_point_inv * &zetas[i + 1] * (&v - evaluation_sym);
                    index >>= 1;

                    if i < fri_decommitment.layers_evaluations_sym.len() - 1 {
                        result & openings_ok
                    } else {
                        result & (v == *fri_last_value) & openings_ok
                    }
                },
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
            lde_base.extend_from_slice(opening.main_trace_polys.evaluations());

            let lde_aux: &[FieldElement<FieldExtension>] = opening
                .aux_trace_polys
                .as_ref()
                .map(|a| a.evaluations())
                .unwrap_or(&[]);

            let evaluation_point = Self::query_challenge_to_evaluation_point(*iota, false, domain);
            deep_poly_evaluations.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                &evaluation_point,
                primitive_root,
                challenges,
                &lde_base,
                lde_aux,
                opening.composition_poly.evaluations(),
            )?);

            // Mirror for the symmetric query point.
            let mut lde_base_sym: Vec<FieldElement<Field>> = Vec::new();
            if let Some(p) = &opening.precomputed_trace_polys {
                lde_base_sym.extend_from_slice(&p.evaluations_sym);
            }
            lde_base_sym.extend_from_slice(opening.main_trace_polys.evaluations_sym());

            let lde_aux_sym: &[FieldElement<FieldExtension>] = opening
                .aux_trace_polys
                .as_ref()
                .map(|a| a.evaluations_sym())
                .unwrap_or(&[]);

            let evaluation_point = Self::query_challenge_to_evaluation_point(*iota, true, domain);
            deep_poly_evaluations_sym.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                &evaluation_point,
                primitive_root,
                challenges,
                &lde_base_sym,
                lde_aux_sym,
                opening.composition_poly.evaluations_sym(),
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
        main_tags: &[crypto::merkle_tree::mmcs::MatrixTag],
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
        expected_bus_balance: &FieldElement<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + Sync + Send + math::traits::ByteConversion,
    {
        if airs.len() != multi_proof.proofs.len() {
            error!(
                "AIR count ({}) does not match proof count ({})",
                airs.len(),
                multi_proof.proofs.len()
            );
            return false;
        }
        if main_tags.len() != airs.len() {
            error!(
                "main_tags count ({}) does not match AIR count ({})",
                main_tags.len(),
                airs.len()
            );
            return false;
        }

        // Check if any AIR has an auxiliary trace
        let needs_lookup_challenges = airs.iter().any(|air| air.has_aux_trace());

        // =====================================================================
        // Round 1, Phase A: Replay main trace commitments
        // =====================================================================
        // Per table: validate the optional precomputed commitment against
        // the hardcoded AIR value (the only one the verifier trusts), and
        // absorb it into the transcript. After every table, absorb the
        // single shared MMCS root that commits to every main trace. Also
        // cross-check `main_mmcs_spec` against the (tag, padded_height_lde)
        // pairs reproduced from the AIRs.

        // Per-chunk Phase A replay: chunk tables of size `chunk_size`. For
        // each table absorb its preprocessed root + per-table main root
        // (preprocessed only); at the end of each chunk, validate the
        // chunk's main MMCS spec and absorb the chunk's main MMCS root
        // (`Some`) or skip (`None` when the chunk has no non-preprocessed
        // tables). Must match `multi_prove` Phase A absorb order exactly.
        let chunk_size = multi_proof.chunk_size as usize;
        if chunk_size == 0 {
            error!("multi_proof.chunk_size is zero");
            return false;
        }
        let expected_num_chunks = (airs.len() + chunk_size - 1) / chunk_size;
        if multi_proof.main_mmcs_roots.len() != expected_num_chunks
            || multi_proof.main_mmcs_specs.len() != expected_num_chunks
            || multi_proof.aux_mmcs_roots.len() != expected_num_chunks
            || multi_proof.aux_mmcs_specs.len() != expected_num_chunks
            || multi_proof.comp_mmcs_roots.len() != expected_num_chunks
            || multi_proof.comp_mmcs_specs.len() != expected_num_chunks
        {
            error!(
                "per-chunk MMCS Vec lengths inconsistent with chunk_size={chunk_size}: \
                 expected {expected_num_chunks} chunks; got main_roots={}, main_specs={}, \
                 aux_roots={}, aux_specs={}, comp_roots={}, comp_specs={}",
                multi_proof.main_mmcs_roots.len(),
                multi_proof.main_mmcs_specs.len(),
                multi_proof.aux_mmcs_roots.len(),
                multi_proof.aux_mmcs_specs.len(),
                multi_proof.comp_mmcs_roots.len(),
                multi_proof.comp_mmcs_specs.len(),
            );
            return false;
        }

        for chunk_idx in 0..expected_num_chunks {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(airs.len());

            let mut expected_spec: Vec<(crypto::merkle_tree::mmcs::MatrixTag, usize)> =
                Vec::new();
            for idx in chunk_start..chunk_end {
                let (air, proof) = (airs[idx], &multi_proof.proofs[idx]);
                let lde_size = proof.trace_length * (air.options().blowup_factor as usize);
                if air.is_preprocessed() {
                    let expected_precomputed = air.precomputed_commitment();
                    match &proof.lde_trace_precomputed_merkle_root {
                        Some(actual) if *actual == expected_precomputed => {}
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
                    transcript.append_bytes(&expected_precomputed);
                    match &proof.lde_trace_main_merkle_root {
                        Some(root) => transcript.append_bytes(root),
                        None => {
                            error!(
                                "Preprocessed table {idx} proof missing multiplicities Merkle root"
                            );
                            return false;
                        }
                    }
                } else {
                    if proof.lde_trace_main_merkle_root.is_some() {
                        error!(
                            "Non-preprocessed table {idx} unexpectedly supplied a per-table main root"
                        );
                        return false;
                    }
                    expected_spec.push((main_tags[idx], lde_size));
                }
            }

            // Deterministic sort matches `MmcsBuilder::finalize`
            // (height desc, tag asc) — same as the streaming builder.
            expected_spec.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            if expected_spec != multi_proof.main_mmcs_specs[chunk_idx] {
                error!(
                    "chunk {chunk_idx} main_mmcs_spec mismatch: expected {:?}, got {:?}",
                    expected_spec, multi_proof.main_mmcs_specs[chunk_idx],
                );
                return false;
            }
            match (
                &multi_proof.main_mmcs_roots[chunk_idx],
                expected_spec.is_empty(),
            ) {
                (Some(root), false) => transcript.append_bytes(root),
                (None, true) => {}
                (Some(_), true) => {
                    error!("chunk {chunk_idx} main_mmcs_root present but no Shared tables");
                    return false;
                }
                (None, false) => {
                    error!("chunk {chunk_idx} main_mmcs_root missing but Shared tables exist");
                    return false;
                }
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
        // Phase C: validate + absorb the shared aux MMCS root (if any)
        // =====================================================================
        // The aux MMCS lives at multi-proof level: a single absorb into the
        // SHARED transcript replaces the per-table aux root absorb of the
        // pre-MMCS protocol. Verify the spec mirrors the prover-side
        // filtered-by-has_aux_trace order before binding.
        // Per-chunk Phase C replay (aux). Mirrors Phase A: for each chunk,
        // validate the aux spec + absorb the aux MMCS root (or skip when
        // the chunk has no aux-bearing tables). Must match `multi_prove`
        // Phase C absorb order exactly.
        for chunk_idx in 0..expected_num_chunks {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(airs.len());

            let mut expected_aux_spec: Vec<(crypto::merkle_tree::mmcs::MatrixTag, usize)> =
                Vec::new();
            for idx in chunk_start..chunk_end {
                let (air, proof) = (airs[idx], &multi_proof.proofs[idx]);
                if air.has_aux_trace() {
                    let lde_size = proof.trace_length * (air.options().blowup_factor as usize);
                    expected_aux_spec.push((main_tags[idx], lde_size));
                }
            }
            expected_aux_spec.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            if expected_aux_spec != multi_proof.aux_mmcs_specs[chunk_idx] {
                error!(
                    "chunk {chunk_idx} aux_mmcs_spec mismatch: expected {:?}, got {:?}",
                    expected_aux_spec, multi_proof.aux_mmcs_specs[chunk_idx],
                );
                return false;
            }
            match (
                &multi_proof.aux_mmcs_roots[chunk_idx],
                expected_aux_spec.is_empty(),
            ) {
                (Some(root), false) => transcript.append_bytes(root),
                (None, true) => {}
                (Some(_), true) => {
                    error!("chunk {chunk_idx} aux_mmcs_root present but no aux tables");
                    return false;
                }
                (None, false) => {
                    error!("chunk {chunk_idx} aux_mmcs_root missing but aux tables exist");
                    return false;
                }
            }
        }

        // Per-chunk composition MMCS spec validation. Every table has a
        // composition polynomial, so every chunk has Some(root). The
        // composition root is NOT absorbed here at the shared-transcript
        // level — it gets absorbed PER-TABLE inside `verify_rounds_2_to_4`
        // between sampling beta and sampling z (mirroring the prover,
        // which absorbs it into each chunk-mate's fork at that point).
        for chunk_idx in 0..expected_num_chunks {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(airs.len());

            let mut expected_comp_spec: Vec<(crypto::merkle_tree::mmcs::MatrixTag, usize)> =
                Vec::new();
            for idx in chunk_start..chunk_end {
                let proof = &multi_proof.proofs[idx];
                let lde_size =
                    proof.trace_length * (airs[idx].options().blowup_factor as usize);
                // Composition MMCS padded height = lde_size / 2 (row-pair leaves).
                expected_comp_spec.push((main_tags[idx], lde_size / 2));
            }
            expected_comp_spec.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            if expected_comp_spec != multi_proof.comp_mmcs_specs[chunk_idx] {
                error!(
                    "chunk {chunk_idx} comp_mmcs_spec mismatch: expected {:?}, got {:?}",
                    expected_comp_spec, multi_proof.comp_mmcs_specs[chunk_idx],
                );
                return false;
            }
            if multi_proof.comp_mmcs_roots[chunk_idx].is_none() {
                error!(
                    "chunk {chunk_idx} comp_mmcs_root missing (every chunk must commit at least one composition matrix)"
                );
                return false;
            }
        }

        // =====================================================================
        // Rounds 2 → 3.5 per-fork replay + per-chunk bucket FRI (Phase D)
        // =====================================================================
        // Per chunk-mate: build fork, replay through γ + step 2 verify.
        // Then per chunk: build the bucket-shared transcript, verify each
        // height bucket's batched FRI, and use the bucket-shared iotas to
        // authenticate every per-query trace / aux / composition opening.

        let num_tables = airs.len();
        let pre_fork_transcript = transcript.clone();
        let mut challenges_per_table: Vec<Option<Challenges<FieldExtension>>> =
            (0..num_tables).map(|_| None).collect();

        for (idx, (air, proof)) in airs.iter().zip(&multi_proof.proofs).enumerate() {
            let mut table_transcript = transcript.clone();
            if num_tables > 1 {
                table_transcript.append_bytes(&(idx as u64).to_le_bytes());
            }
            if let Some(ref bpi) = proof.bus_public_inputs {
                table_transcript.append_field_element(&bpi.table_contribution);
            }

            let table_chunk_idx = idx / chunk_size;
            let comp_root_for_chunk =
                multi_proof.comp_mmcs_roots[table_chunk_idx].as_ref();

            let chal = match Self::replay_and_verify_step_2(
                *air,
                proof,
                &mut table_transcript,
                lookup_challenges.clone(),
                comp_root_for_chunk,
            ) {
                Some(c) => c,
                None => {
                    error!(
                        "Table {} failed replay_and_verify_step_2 (num_constraints={}, trace_cols={})",
                        idx,
                        air.context().num_transition_constraints,
                        air.context().trace_columns
                    );
                    return false;
                }
            };
            challenges_per_table[idx] = Some(chal);
        }

        // Per-chunk: build bucket_seed (canonical replay on pre-fork state),
        // validate fri_chunk_buckets[chunk_idx] structure, verify each
        // bucket's batched FRI, then per chunk-mate verify step 4.
        if multi_proof.fri_chunk_buckets.len() != expected_num_chunks {
            error!(
                "fri_chunk_buckets outer length {} != expected_num_chunks {}",
                multi_proof.fri_chunk_buckets.len(),
                expected_num_chunks,
            );
            return false;
        }

        for chunk_idx in 0..expected_num_chunks {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(num_tables);

            // bucket_seed: clone pre-fork shared state + canonical replay.
            let mut bucket_seed = pre_fork_transcript.clone();
            for idx in chunk_start..chunk_end {
                if let Some(ref bpi) = multi_proof.proofs[idx].bus_public_inputs {
                    bucket_seed.append_field_element(&bpi.table_contribution);
                }
            }
            if let Some(ref root) = multi_proof.comp_mmcs_roots[chunk_idx] {
                bucket_seed.append_bytes(root);
            }
            for idx in chunk_start..chunk_end {
                let p = &multi_proof.proofs[idx];
                for col in p.trace_ood_evaluations.columns().iter() {
                    for elem in col.iter() {
                        bucket_seed.append_field_element(elem);
                    }
                }
                for elem in p.composition_poly_parts_ood_evaluation.iter() {
                    bucket_seed.append_field_element(elem);
                }
            }

            // Expected bucketing: first-encounter order by lde_size.
            let mut expected_bucket_indices: Vec<Vec<usize>> = Vec::new();
            let mut expected_bucket_lde_sizes: Vec<usize> = Vec::new();
            for j in 0..(chunk_end - chunk_start) {
                let idx = chunk_start + j;
                let lde_size = multi_proof.proofs[idx].trace_length
                    * airs[idx].options().blowup_factor as usize;
                match expected_bucket_lde_sizes.iter().position(|&s| s == lde_size) {
                    Some(b) => expected_bucket_indices[b].push(j),
                    None => {
                        expected_bucket_lde_sizes.push(lde_size);
                        expected_bucket_indices.push(vec![j]);
                    }
                }
            }

            let chunk_buckets = &multi_proof.fri_chunk_buckets[chunk_idx];
            if chunk_buckets.len() != expected_bucket_indices.len() {
                error!(
                    "chunk {chunk_idx}: bucket count {} != expected {}",
                    chunk_buckets.len(),
                    expected_bucket_indices.len(),
                );
                return false;
            }

            // map chunk-local-index → bucket index (for step 4 dispatch).
            let mut member_bucket_idx: Vec<usize> = vec![0; chunk_end - chunk_start];
            // Cache bucket iotas: derived once during FRI verification,
            // reused in step 4 without re-cloning the bucket transcript.
            let mut bucket_iotas_cache: Vec<Vec<usize>> =
                Vec::with_capacity(chunk_buckets.len());

            for (b, bucket) in chunk_buckets.iter().enumerate() {
                let expected_members = &expected_bucket_indices[b];
                let expected_lde_size = expected_bucket_lde_sizes[b];
                if bucket.lde_size as usize != expected_lde_size {
                    error!(
                        "chunk {chunk_idx} bucket {b}: lde_size {} != expected {}",
                        bucket.lde_size, expected_lde_size,
                    );
                    return false;
                }
                if bucket.members.len() != expected_members.len() {
                    error!(
                        "chunk {chunk_idx} bucket {b}: members.len {} != expected {}",
                        bucket.members.len(),
                        expected_members.len(),
                    );
                    return false;
                }
                for (mi, &j) in expected_members.iter().enumerate() {
                    let expected_tag = main_tags[chunk_start + j];
                    if bucket.members[mi] != expected_tag {
                        error!(
                            "chunk {chunk_idx} bucket {b} member {mi}: tag mismatch",
                        );
                        return false;
                    }
                    member_bucket_idx[j] = b;
                }

                // Verify the bucket FRI: replay layer-root absorbs, sample
                // zetas, absorb last_value, grinding, sample iotas, and run
                // per-iota combined-D fold check.
                let leader_idx = chunk_start + expected_members[0];
                let leader_air = airs[leader_idx];
                let leader_domain =
                    new_verifier_domain(leader_air, multi_proof.proofs[leader_idx].trace_length);

                let mut bt = bucket_seed.clone();
                bt.append_bytes(&(bucket.lde_size as u64).to_le_bytes());
                let delta_fri: FieldElement<FieldExtension> = bt.sample_field_element();

                let mut zetas: Vec<FieldElement<FieldExtension>> =
                    Vec::with_capacity(bucket.layer_roots.len() + 1);
                for root in &bucket.layer_roots {
                    let z = bt.sample_field_element();
                    bt.append_bytes(root);
                    zetas.push(z);
                }
                zetas.push(bt.sample_field_element());
                bt.append_field_element(&bucket.last_value);

                let security_bits = leader_air.context().proof_options.grinding_factor;
                if security_bits > 0 {
                    let nonce = match bucket.nonce {
                        Some(n) => n,
                        None => {
                            error!(
                                "chunk {chunk_idx} bucket {b}: grinding required but nonce missing",
                            );
                            return false;
                        }
                    };
                    let grinding_seed = bt.state();
                    if !grinding::is_valid_nonce(&grinding_seed, nonce, security_bits) {
                        #[cfg(not(feature = "test_fiat_shamir"))]
                        error!("chunk {chunk_idx} bucket {b}: grinding factor not satisfied");
                        return false;
                    }
                    bt.append_bytes(&nonce.to_be_bytes());
                } else if bucket.nonce.is_some() {
                    error!(
                        "chunk {chunk_idx} bucket {b}: nonce present but grinding disabled",
                    );
                    return false;
                }

                let number_of_queries = leader_air.options().fri_number_of_queries;
                let iotas =
                    Self::sample_query_indexes(number_of_queries, &leader_domain, &mut bt);

                if bucket.decommitments.len() != iotas.len() {
                    error!(
                        "chunk {chunk_idx} bucket {b}: decommitments {} != iotas {}",
                        bucket.decommitments.len(),
                        iotas.len(),
                    );
                    return false;
                }

                // Reconstruct per-bucket-mate D_i(iota±) for every iota.
                let mut per_member_d: Vec<DeepPolynomialEvaluations<FieldExtension>> =
                    Vec::with_capacity(expected_members.len());
                for &j in expected_members.iter() {
                    let idx = chunk_start + j;
                    let chal = challenges_per_table[idx]
                        .as_ref()
                        .expect("step-2 succeeded → challenges populated");
                    // Replace the challenge's empty iotas with bucket iotas.
                    let chal_with_iotas = Challenges {
                        z: chal.z.clone(),
                        boundary_coeffs: chal.boundary_coeffs.clone(),
                        transition_coeffs: chal.transition_coeffs.clone(),
                        trace_term_coeffs: chal.trace_term_coeffs.clone(),
                        gammas: chal.gammas.clone(),
                        zetas: zetas.clone(),
                        iotas: iotas.clone(),
                        rap_challenges: chal.rap_challenges.clone(),
                        grinding_seed: [0u8; 32],
                    };
                    let member_domain =
                        new_verifier_domain(airs[idx], multi_proof.proofs[idx].trace_length);
                    let pair = match Self::reconstruct_d_evaluations_for_table(
                        &multi_proof.proofs[idx],
                        &member_domain,
                        &chal_with_iotas,
                    ) {
                        Some(pair) => pair,
                        None => {
                            error!(
                                "chunk {chunk_idx} bucket {b} member {j}: D reconstruction failed",
                            );
                            return false;
                        }
                    };
                    // chal_with_iotas only needed inside the call.
                    let _ = chal_with_iotas;
                    per_member_d.push(pair);
                }

                // Per-iota: combine D_i with successive powers of δ_fri,
                // verify FRI fold authenticates and reaches bucket.last_value.
                let mut evaluation_point_inv = iotas
                    .iter()
                    .map(|iota| {
                        Self::query_challenge_to_evaluation_point(*iota, false, &leader_domain)
                    })
                    .collect::<Vec<FieldElement<Field>>>();
                if FieldElement::inplace_batch_inverse(&mut evaluation_point_inv).is_err() {
                    error!(
                        "chunk {chunk_idx} bucket {b}: query evaluation point not invertible",
                    );
                    return false;
                }

                for (q, &iota) in iotas.iter().enumerate() {
                    let mut d_iota = FieldElement::<FieldExtension>::zero();
                    let mut d_iota_sym = FieldElement::<FieldExtension>::zero();
                    let mut coeff = FieldElement::<FieldExtension>::one();
                    for (i_local, member_d) in per_member_d.iter().enumerate() {
                        d_iota = d_iota + &coeff * &member_d.0[q];
                        d_iota_sym = d_iota_sym + &coeff * &member_d.1[q];
                        if i_local + 1 < per_member_d.len() {
                            coeff = coeff * &delta_fri;
                        }
                    }

                    if !Self::verify_bucket_fri_query(
                        &bucket.layer_roots,
                        &bucket.last_value,
                        &zetas,
                        iota,
                        &bucket.decommitments[q],
                        evaluation_point_inv[q].clone(),
                        &d_iota,
                        &d_iota_sym,
                    ) {
                        #[cfg(not(feature = "test_fiat_shamir"))]
                        error!(
                            "chunk {chunk_idx} bucket {b} query {q}: FRI fold verification failed",
                        );
                        return false;
                    }
                }
                bucket_iotas_cache.push(iotas);
            }

            // Per chunk-mate: step 4 at its bucket's iotas (cached above,
            // no transcript replay needed).
            for j in 0..(chunk_end - chunk_start) {
                let idx = chunk_start + j;
                let b = member_bucket_idx[j];
                let iotas = &bucket_iotas_cache[b];

                let proof = &multi_proof.proofs[idx];
                let main_root = multi_proof.main_mmcs_roots[chunk_idx].as_ref();
                let main_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)] =
                    &multi_proof.main_mmcs_specs[chunk_idx];
                let aux_root = multi_proof.aux_mmcs_roots[chunk_idx].as_ref();
                let aux_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)] =
                    &multi_proof.aux_mmcs_specs[chunk_idx];
                let comp_root = multi_proof.comp_mmcs_roots[chunk_idx].as_ref();
                let comp_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)] =
                    &multi_proof.comp_mmcs_specs[chunk_idx];

                if !Self::verify_step_4_at_iotas(
                    proof,
                    iotas,
                    main_tags[idx],
                    main_root,
                    main_spec,
                    aux_root,
                    aux_spec,
                    comp_root,
                    comp_spec,
                ) {
                    #[cfg(not(feature = "test_fiat_shamir"))]
                    error!("Table {idx}: step 4 trace/comp openings failed at bucket iotas");
                    return false;
                }
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

    /// Verify a single-AIR STARK proof packaged as a one-element `MultiProof`.
    /// Equivalent to `multi_verify(&[air], proof, &[default_tag], ...)`.
    fn verify(
        proof: &MultiProof<Field, FieldExtension, PI>,
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + Sync + Send + math::traits::ByteConversion,
        PI: Clone,
    {
        let main_tags = [crypto::merkle_tree::mmcs::MatrixTag::new([0; 8])];
        Self::multi_verify(
            &[air],
            proof,
            &main_tags,
            transcript,
            &FieldElement::zero(),
        )
    }

    /// Replays rounds 2, 3 and 4 of the protocol for a given proof, assuming round 1 has
    /// already been replayed and the RAP challenges are known.
    ///
    /// `comp_mmcs_root` is this table's chunk composition MMCS root,
    /// absorbed between beta and z sampling. The prover absorbs the
    /// same root into each chunk-mate's fork.
    #[allow(clippy::too_many_arguments)]
    fn replay_rounds_after_round_1(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &StarkProof<Field, FieldExtension, PI>,
        domain: &VerifierDomain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        rap_challenges: Vec<FieldElement<FieldExtension>>,
        comp_mmcs_root: Option<&Commitment>,
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

        // <<<< Receive commitment: chunk composition MMCS root (one
        // absorb per chunk-mate's fork, mirroring `multi_prove`).
        if let Some(root) = comp_mmcs_root {
            transcript.append_bytes(root);
        }

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
        // ==========|  Round 3.5  |==========
        // ===================================
        // Sample γ from the per-fork transcript; build the per-table
        // DEEP composition coefficient layout. The FRI commit + iotas
        // happen at chunk-bucket level (verified separately) — this
        // replay stops at γ.

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

        // zetas / iotas / grinding_seed are populated by the chunk-bucket
        // FRI verification step in `multi_verify` (Phase D). The per-fork
        // transcript ends here.
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

    /// Verifies a single table after round 1 has been replayed.
    ///
    /// `main_*` / `aux_*` come from the shared multi-proof and authenticate
    /// the per-table trace openings in step 4.
    /// Replays per-fork rounds 2 → 3.5 for one table and runs step 2
    /// (composition-polynomial OOD consistency). Returns the per-fork
    /// Challenges populated up through γ — `zetas`, `iotas`, and
    /// `grinding_seed` remain empty and are filled in by the chunk-bucket
    /// FRI verification (Phase D).
    ///
    /// Step 4 (trace openings at iotas) is split into
    /// [`verify_step_4_at_iotas`] driven by `multi_verify` after the
    /// bucket FRI sets each chunk-mate's iota list.
    #[allow(clippy::too_many_arguments)]
    fn replay_and_verify_step_2(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: &StarkProof<Field, FieldExtension, PI>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        rap_challenges: Vec<FieldElement<FieldExtension>>,
        comp_mmcs_root: Option<&Commitment>,
    ) -> Option<Challenges<FieldExtension>>
    where
        FieldElement<Field>: AsBytes + Sync + Send + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + Sync + Send + math::traits::ByteConversion,
    {
        let domain = new_verifier_domain(air, proof.trace_length);

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
            comp_mmcs_root,
        );

        // Grinding + iotas + FRI verification moved to chunk-bucket level
        // in `multi_verify` (Phase D batched FRI).

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
            return None;
        }

        #[cfg(feature = "instruments")]
        let elapsed2 = timer2.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed2);
        #[cfg(feature = "instruments")]
        println!("- Started step 3: Verify FRI");
        #[cfg(feature = "instruments")]
        let timer3 = Instant::now();

        // FRI verification (Phase D) is driven from `multi_verify` per
        // chunk-bucket. This per-table replay stops here.

        #[cfg(feature = "instruments")]
        let elapsed3 = timer3.elapsed();
        #[cfg(feature = "instruments")]
        println!("  Time spent: {:?}", elapsed3);

        #[cfg(feature = "instruments")]
        println!("- Started step 4: Verify deep composition polynomial");
        #[cfg(feature = "instruments")]
        let timer4 = Instant::now();

        // Step 4 (per-iota openings) runs at chunk-bucket level (Phase D).

        #[cfg(feature = "instruments")]
        let _ = (elapsed1, timer2.elapsed(), timer3.elapsed(), timer4.elapsed());

        Some(challenges)
    }

    /// Step 4 for one table at the bucket-shared iotas: authenticate
    /// every per-query opening against the chunk's main / aux /
    /// composition MMCS roots.
    #[allow(clippy::too_many_arguments)]
    fn verify_step_4_at_iotas(
        proof: &StarkProof<Field, FieldExtension, PI>,
        iotas: &[usize],
        main_tag: crypto::merkle_tree::mmcs::MatrixTag,
        main_mmcs_root: Option<&Commitment>,
        main_mmcs_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)],
        aux_mmcs_root: Option<&Commitment>,
        aux_mmcs_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)],
        comp_mmcs_root: Option<&Commitment>,
        comp_mmcs_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)],
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send + math::traits::ByteConversion,
        FieldElement<FieldExtension>: AsBytes + Sync + Send + math::traits::ByteConversion,
    {
        if proof.deep_poly_openings.len() < iotas.len() {
            return false;
        }
        iotas
            .iter()
            .zip(proof.deep_poly_openings.iter())
            .all(|(iota_n, deep_poly_opening)| {
                Self::verify_composition_poly_opening(
                    deep_poly_opening,
                    comp_mmcs_root,
                    comp_mmcs_spec,
                    main_tag,
                    *iota_n,
                ) && Self::verify_trace_openings(
                    proof,
                    deep_poly_opening,
                    *iota_n,
                    main_tag,
                    main_mmcs_root,
                    main_mmcs_spec,
                    aux_mmcs_root,
                    aux_mmcs_spec,
                )
            })
    }
}

fn verify_main_mmcs_pair_inner<F>(
    main_opening: &crate::proof::stark::MainTraceOpening<F>,
    iota: usize,
    main_tag: crypto::merkle_tree::mmcs::MatrixTag,
    main_mmcs_root: Option<&Commitment>,
    main_mmcs_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)],
) -> bool
where
    F: IsField,
    FieldElement<F>: AsBytes + Sync + Send + math::traits::ByteConversion,
{
    use crate::mmcs_leaf::hash_tagged_row;
    use crate::proof::stark::MainTraceOpening;

    let (evaluations, evaluations_sym, mmcs_opening, mmcs_opening_sym) = match main_opening {
        MainTraceOpening::Mmcs {
            evaluations,
            evaluations_sym,
            mmcs_opening,
            mmcs_opening_sym,
        } => (evaluations, evaluations_sym, mmcs_opening, mmcs_opening_sym),
        MainTraceOpening::Tree(_) => return false,
    };

    // Shared opening requires a chunk MMCS root; if missing, reject.
    let main_mmcs_root = match main_mmcs_root {
        Some(r) => r,
        None => return false,
    };

    let table_idx = match main_mmcs_spec.iter().position(|(t, _)| *t == main_tag) {
        Some(i) => i,
        None => return false,
    };
    let table_height = main_mmcs_spec[table_idx].1;
    let max_height = match main_mmcs_spec.first().map(|(_, h)| *h) {
        Some(h) => h,
        None => return false,
    };
    if !table_height.is_power_of_two() || max_height < table_height {
        return false;
    }
    let shift = (max_height / table_height).trailing_zeros() as usize;
    let g_primary = (iota * 2) << shift;
    let g_sym = (iota * 2 + 1) << shift;
    let leaf_primary = hash_tagged_row::<F>(main_tag, evaluations);
    let leaf_sym = hash_tagged_row::<F>(main_tag, evaluations_sym);
    if mmcs_opening.global_index != g_primary || mmcs_opening_sym.global_index != g_sym {
        return false;
    }
    let leaves = &mmcs_opening.matrix_leaves;
    let leaves_sym = &mmcs_opening_sym.matrix_leaves;
    if table_idx >= leaves.len() || table_idx >= leaves_sym.len() {
        return false;
    }
    if leaves[table_idx].0 != main_tag || leaves[table_idx].1 != leaf_primary {
        return false;
    }
    if leaves_sym[table_idx].0 != main_tag || leaves_sym[table_idx].1 != leaf_sym {
        return false;
    }
    let ok = mmcs_opening.verify::<BatchedMerkleTreeBackend<F>>(main_mmcs_root, main_mmcs_spec);
    let ok_sym =
        mmcs_opening_sym.verify::<BatchedMerkleTreeBackend<F>>(main_mmcs_root, main_mmcs_spec);
    ok && ok_sym
}

/// Aux-trace counterpart of [`verify_main_mmcs_pair_inner`]. Same shape,
/// but rehashes the row using the AUX domain separator so an aux opening
/// cannot authenticate a main leaf (or vice versa).
fn verify_aux_mmcs_pair_inner<E>(
    aux_opening: &crate::proof::stark::AuxTraceOpening<E>,
    iota: usize,
    main_tag: crypto::merkle_tree::mmcs::MatrixTag,
    aux_mmcs_root: &Commitment,
    aux_mmcs_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)],
) -> bool
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send + math::traits::ByteConversion,
{
    use crate::mmcs_leaf::hash_tagged_row_aux;
    use crate::proof::stark::AuxTraceOpening;
    let AuxTraceOpening::Mmcs {
        evaluations,
        evaluations_sym,
        mmcs_opening,
        mmcs_opening_sym,
    } = aux_opening;

    let table_idx = match aux_mmcs_spec.iter().position(|(t, _)| *t == main_tag) {
        Some(i) => i,
        None => return false,
    };
    let table_height = aux_mmcs_spec[table_idx].1;
    let max_height = match aux_mmcs_spec.first().map(|(_, h)| *h) {
        Some(h) => h,
        None => return false,
    };
    if !table_height.is_power_of_two() || max_height < table_height {
        return false;
    }
    let shift = (max_height / table_height).trailing_zeros() as usize;
    let g_primary = (iota * 2) << shift;
    let g_sym = (iota * 2 + 1) << shift;
    let leaf_primary = hash_tagged_row_aux::<E>(main_tag, evaluations);
    let leaf_sym = hash_tagged_row_aux::<E>(main_tag, evaluations_sym);
    if mmcs_opening.global_index != g_primary || mmcs_opening_sym.global_index != g_sym {
        return false;
    }
    let leaves = &mmcs_opening.matrix_leaves;
    let leaves_sym = &mmcs_opening_sym.matrix_leaves;
    if table_idx >= leaves.len() || table_idx >= leaves_sym.len() {
        return false;
    }
    if leaves[table_idx].0 != main_tag || leaves[table_idx].1 != leaf_primary {
        return false;
    }
    if leaves_sym[table_idx].0 != main_tag || leaves_sym[table_idx].1 != leaf_sym {
        return false;
    }
    let ok = mmcs_opening.verify::<BatchedMerkleTreeBackend<E>>(aux_mmcs_root, aux_mmcs_spec);
    let ok_sym =
        mmcs_opening_sym.verify::<BatchedMerkleTreeBackend<E>>(aux_mmcs_root, aux_mmcs_spec);
    ok && ok_sym
}

/// Composition-trace counterpart of [`verify_main_mmcs_pair_inner`]. Uses
/// `LEAF_DOMAIN_TAG_COMPOSITION` for rehash; the leaf hashes a row-PAIR
/// rather than a single row, so the opening covers both `evaluations`
/// (row 0 / br_0) and `evaluations_sym` (row 1 / br_1) under one MMCS
/// opening — no separate `_sym` opening at this layer (the underlying
/// tree's leaves are already row-pairs).
fn verify_comp_mmcs_pair_inner<E>(
    comp_opening: &crate::proof::stark::CompositionTraceOpening<E>,
    iota: usize,
    main_tag: crypto::merkle_tree::mmcs::MatrixTag,
    comp_mmcs_root: Option<&Commitment>,
    comp_mmcs_spec: &[(crypto::merkle_tree::mmcs::MatrixTag, usize)],
) -> bool
where
    E: IsField,
    FieldElement<E>: AsBytes + Sync + Send + math::traits::ByteConversion,
{
    use crate::mmcs_leaf::hash_tagged_row_pair_composition;
    use crate::proof::stark::CompositionTraceOpening;

    let comp_mmcs_root = match comp_mmcs_root {
        Some(r) => r,
        None => return false,
    };
    let CompositionTraceOpening::Mmcs {
        evaluations,
        evaluations_sym,
        mmcs_opening,
    } = comp_opening;

    let table_idx = match comp_mmcs_spec.iter().position(|(t, _)| *t == main_tag) {
        Some(i) => i,
        None => return false,
    };
    let table_height = comp_mmcs_spec[table_idx].1;
    let max_height = match comp_mmcs_spec.first().map(|(_, h)| *h) {
        Some(h) => h,
        None => return false,
    };
    if !table_height.is_power_of_two() || max_height < table_height {
        return false;
    }
    let shift = (max_height / table_height).trailing_zeros() as usize;
    // Composition opens at row-pair index iota, so the global index in
    // the chunk MMCS is iota shifted up by the chunk-mate's depth diff.
    let g_index = iota << shift;
    if mmcs_opening.global_index != g_index {
        return false;
    }

    let leaf = hash_tagged_row_pair_composition::<E>(main_tag, evaluations, evaluations_sym);
    let leaves = &mmcs_opening.matrix_leaves;
    if table_idx >= leaves.len() {
        return false;
    }
    if leaves[table_idx].0 != main_tag || leaves[table_idx].1 != leaf {
        return false;
    }
    mmcs_opening.verify::<BatchedMerkleTreeBackend<E>>(comp_mmcs_root, comp_mmcs_spec)
}
