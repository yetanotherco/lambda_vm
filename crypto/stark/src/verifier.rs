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
        logup_alpha_powers: &[FieldElement<FieldExtension>],
    ) -> bool {
        let trace_length = proof.trace_length;
        let boundary_constraints = air.boundary_constraints(
            &proof.public_inputs,
            &challenges.rap_challenges,
            proof.bus_public_inputs.as_ref(),
            trace_length,
        );
        // Precompute g^step once per distinct step. A small `Vec` with a
        // linear scan beats `HashMap` here: boundary constraints typically
        // number in the low tens, the recursion guest pays no allocator/hash
        // overhead, and the AIR generally emits its constraints grouped by
        // step so the scan hits the hot entry first.
        let mut step_to_point: Vec<(usize, FieldElement<Field>)> = Vec::new();
        let boundary_points: Vec<FieldElement<Field>> = boundary_constraints
            .constraints
            .iter()
            .map(|c| {
                if let Some((_, point)) = step_to_point.iter().find(|(s, _)| *s == c.step) {
                    point.clone()
                } else {
                    let point = domain.trace_primitive_root.pow(c.step as u64);
                    step_to_point.push((c.step, point.clone()));
                    point
                }
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

        // Reuse a prefix slice of the globally-computed alpha powers instead of
        // recomputing the multiplication chain per table. The global vector is
        // sized to the maximum bus element count across all AIRs, so this
        // table's prefix is always available; `.min` is purely defensive.
        let logup_alpha_powers_slice: &[FieldElement<FieldExtension>] =
            if !logup_alpha_powers.is_empty() {
                &logup_alpha_powers[..air.max_bus_elements().min(logup_alpha_powers.len())]
            } else {
                &[]
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
            logup_alpha_powers_slice,
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
            match Self::reconstruct_deep_composition_poly_evaluations_for_all_queries(
                challenges, domain, proof,
            ) {
                Some(pair) => pair,
                None => return false,
            };

        // verify FRI
        let mut evaluation_point_inverse = challenges
            .iotas
            .iter()
            .map(|iota| Self::query_challenge_to_evaluation_point(*iota, false, domain))
            .collect::<Vec<FieldElement<Field>>>();
        // Any zero evaluation point means a malformed query index, reject.
        if FieldElement::inplace_batch_inverse(&mut evaluation_point_inverse).is_err() {
            return false;
        }

        proof
            .query_list
            .iter()
            .zip(&challenges.iotas)
            .zip(evaluation_point_inverse)
            .enumerate()
            .all(|(i, ((proof_s, iota_s), eval))| {
                Self::verify_query_and_sym_openings(
                    proof,
                    &challenges.zetas,
                    *iota_s,
                    proof_s,
                    eval,
                    &deep_poly_evaluations[i],
                    &deep_poly_evaluations_sym[i],
                )
            })
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
        let leaf = BatchedMerkleTreeBackend::<E>::hash_elements(value.iter());
        proof.verify_hashed::<BatchedMerkleTreeBackend<E>>(root, index, leaf)
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
        // Short-circuit on any failure: each opening pair is a Merkle-path
        // verification (~20 Keccak hashes against base-field leaves); in the
        // recursion guest this is non-trivial cycle cost worth skipping.
        if !Self::verify_opening_pair::<Field>(
            &deep_poly_openings.main_trace_polys,
            &proof.lde_trace_main_merkle_root,
            iota,
        ) {
            return false;
        }

        // Precomputed trace (preprocessed tables only). Mismatched presence is
        // unreachable in practice (multi_verify rejects such proofs upstream),
        // but a defensive check keeps this function self-contained.
        match (
            &proof.lde_trace_precomputed_merkle_root,
            &deep_poly_openings.precomputed_trace_polys,
        ) {
            (Some(root), Some(opening)) => {
                if !Self::verify_opening_pair::<Field>(opening, root, iota) {
                    return false;
                }
            }
            (None, None) => {}
            _ => return false,
        }

        // Auxiliary trace.
        match (
            proof.lde_trace_aux_merkle_root,
            &deep_poly_openings.aux_trace_polys,
        ) {
            (Some(root), Some(opening)) => {
                if !Self::verify_opening_pair::<FieldExtension>(opening, &root, iota) {
                    return false;
                }
            }
            (None, None) => {}
            _ => return false,
        }

        true
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
        let leaf = BatchedMerkleTreeBackend::<FieldExtension>::hash_elements(
            deep_poly_openings
                .composition_poly
                .evaluations
                .iter()
                .chain(deep_poly_openings.composition_poly.evaluations_sym.iter()),
        );

        deep_poly_openings
            .composition_poly
            .proof
            .verify_hashed::<BatchedMerkleTreeBackend<FieldExtension>>(
                composition_poly_merkle_root,
                *iota,
                leaf,
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
        let (a, b) = if iota % 2 == 1 {
            (evaluation_sym, evaluation)
        } else {
            (evaluation, evaluation_sym)
        };
        let leaf = BatchedMerkleTreeBackend::<FieldExtension>::hash_elements([a, b]);

        auth_path_sym.verify_hashed::<BatchedMerkleTreeBackend<FieldExtension>>(
            merkle_root,
            iota >> 1,
            leaf,
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
        let p0_eval = deep_composition_evaluation;
        let p0_eval_sym = deep_composition_evaluation_sym;

        // Reconstruct p₁(𝜐²)
        let mut v =
            (p0_eval + p0_eval_sym) + &evaluation_point_inv * &zetas[0] * (p0_eval - p0_eval_sym);
        let mut index = iota;

        // Handle case with 0 FRI layers (trace_length <= 2)
        // In this case, the fold loop below doesn't iterate, so we need to verify
        // the final value directly here.
        if fri_layers_merkle_roots.is_empty() {
            return v == proof.fri_last_value;
        }

        // Guard zip alignment: the three iterables MUST have equal lengths.
        // A malformed proof with mismatched lengths would otherwise silently
        // truncate the verification or panic on the `len() - 1` below.
        if fri_decommitment.layers_auth_paths.len() != fri_layers_merkle_roots.len()
            || fri_decommitment.layers_evaluations_sym.len() != fri_layers_merkle_roots.len()
        {
            return false;
        }

        // For each FRI layer, verify openings then fold to the next layer's
        // evaluation. `evaluation_point_squared` is stepped in-place instead
        // of pre-collecting a Vec, and a failed opening short-circuits the
        // remaining Merkle work (each call is ~log₂(N) Keccak hashes).
        let last_layer_idx = fri_layers_merkle_roots.len() - 1;
        let mut evaluation_point_squared = evaluation_point_inv.square();
        for (i, ((merkle_root, auth_path_sym), evaluation_sym)) in fri_layers_merkle_roots
            .iter()
            .zip(&fri_decommitment.layers_auth_paths)
            .zip(&fri_decommitment.layers_evaluations_sym)
            .enumerate()
        {
            // Verify opening Open(pᵢ(Dₖ), −𝜐^(2ⁱ)) and Open(pᵢ(Dₖ), 𝜐^(2ⁱ)).
            // `v` is pᵢ(𝜐^(2ⁱ)). `evaluation_sym` is pᵢ(−𝜐^(2ⁱ)).
            if !Self::verify_fri_layer_openings(
                merkle_root,
                auth_path_sym,
                &v,
                evaluation_sym,
                index,
            ) {
                return false;
            }

            // Update `v` with next value pᵢ₊₁(𝜐^(2ⁱ⁺¹)).
            v = (&v + evaluation_sym)
                + &evaluation_point_squared * &zetas[i + 1] * (&v - evaluation_sym);

            // Index of the squares in the next layer = current index halved
            // (bit-reverse ordering of the Merkle tree).
            index >>= 1;

            if i == last_layer_idx {
                return v == proof.fri_last_value;
            }
            evaluation_point_squared = evaluation_point_squared.square();
        }

        // Unreachable: the length guard above ensures the loop iterates at
        // least once (we passed the is_empty check) and hits the
        // `i == last_layer_idx` return.
        unreachable!("loop must hit the last_layer_idx return")
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

        let height = proof.trace_ood_evaluations.height;
        // Per-entry stride in the flat denominator buffer: `height` trace
        // denominators followed by one composition denominator.
        let stride = height + 1;

        // The composition denominator exponent is constant across all queries:
        // it is the number of composition poly parts the proof advertises (the
        // same array the consumer validates against). Hoist `z^N` once.
        let number_of_parts = proof.composition_poly_parts_ood_evaluation.len();
        let z_pow = challenges.z.pow(number_of_parts);

        // Per-entry data carried from Pass 1 to Pass 2. We borrow every slice
        // straight out of the proof opening: precomputed and main base-field
        // columns separately (avoiding a per-query concatenation allocation),
        // plus the aux and composition evaluation slices.
        struct DeepEntry<'a, Field: IsField, FieldExtension: IsField> {
            lde_precomputed: &'a [FieldElement<Field>],
            lde_main: &'a [FieldElement<Field>],
            lde_aux: &'a [FieldElement<FieldExtension>],
            comp_evals: &'a [FieldElement<FieldExtension>],
            is_sym: bool,
        }

        let mut entries: Vec<DeepEntry<Field, FieldExtension>> =
            Vec::with_capacity(num_queries * 2);
        // Flat buffer of all denominators across every (query, query-point).
        // A SINGLE batch inversion is performed over this whole buffer.
        let mut all_denoms: Vec<FieldElement<FieldExtension>> =
            Vec::with_capacity(num_queries * 2 * stride);

        // Pass 1: collect openings + denominators (no inversions yet).
        for (i, iota) in challenges.iotas.iter().enumerate() {
            let opening = &proof.deep_poly_openings[i];

            for is_sym in [false, true] {
                // Base-field portion: precomputed columns FIRST, then main trace
                // columns. Borrow both slices directly (empty slice when the
                // opening carries no precomputed trace).
                let lde_precomputed: &[FieldElement<Field>] = match &opening.precomputed_trace_polys
                {
                    Some(p) if is_sym => p.evaluations_sym.as_slice(),
                    Some(p) => p.evaluations.as_slice(),
                    None => &[],
                };
                let lde_main: &[FieldElement<Field>] = if is_sym {
                    opening.main_trace_polys.evaluations_sym.as_slice()
                } else {
                    opening.main_trace_polys.evaluations.as_slice()
                };

                let lde_aux: &[FieldElement<FieldExtension>] = match &opening.aux_trace_polys {
                    Some(a) if is_sym => a.evaluations_sym.as_slice(),
                    Some(a) => a.evaluations.as_slice(),
                    None => &[],
                };

                let comp_evals: &[FieldElement<FieldExtension>] = if is_sym {
                    &opening.composition_poly.evaluations_sym
                } else {
                    &opening.composition_poly.evaluations
                };

                let evaluation_point =
                    Self::query_challenge_to_evaluation_point(*iota, is_sym, domain);

                // `height` trace denominators: (upsilon - z*g^k) for k = 0..height.
                let mut current_z = challenges.z.clone();
                for _ in 0..height {
                    all_denoms.push(&evaluation_point - &current_z);
                    current_z = primitive_root * &current_z;
                }
                // One composition denominator: (upsilon - z^N).
                all_denoms.push(&evaluation_point - &z_pow);

                entries.push(DeepEntry {
                    lde_precomputed,
                    lde_main,
                    lde_aux,
                    comp_evals,
                    is_sym,
                });
            }
        }

        // Single global batch inversion. A malformed proof can land an OOD
        // evaluation point on the LDE coset (zero denominator); this rejects
        // the whole verify, matching the prior per-call semantics.
        FieldElement::inplace_batch_inverse(&mut all_denoms).ok()?;

        // Pass 2: reconstruct each DEEP evaluation using the pre-inverted denoms.
        for (e, entry) in entries.iter().enumerate() {
            let denoms_slice = &all_denoms[e * stride..e * stride + stride];
            let trace_denoms_inv = &denoms_slice[..height];
            let comp_denom_inv = &denoms_slice[height];

            let value = Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                challenges,
                entry.lde_precomputed,
                entry.lde_main,
                entry.lde_aux,
                entry.comp_evals,
                trace_denoms_inv,
                comp_denom_inv,
            )?;

            if entry.is_sym {
                deep_poly_evaluations_sym.push(value);
            } else {
                deep_poly_evaluations.push(value);
            }
        }

        Some((deep_poly_evaluations, deep_poly_evaluations_sym))
    }

    #[allow(clippy::too_many_arguments)]
    fn reconstruct_deep_composition_poly_evaluation(
        proof: &StarkProof<Field, FieldExtension, PI>,
        challenges: &Challenges<FieldExtension>,
        lde_precomputed: &[FieldElement<Field>],
        lde_main: &[FieldElement<Field>],
        lde_trace_aux_evaluations: &[FieldElement<FieldExtension>],
        lde_composition_poly_parts_evaluation: &[FieldElement<FieldExtension>],
        trace_denoms_inv: &[FieldElement<FieldExtension>],
        comp_denom_inv: &FieldElement<FieldExtension>,
    ) -> Option<FieldElement<FieldExtension>> {
        let ood_evaluations_table_height = proof.trace_ood_evaluations.height;
        let ood_evaluations_table_width = proof.trace_ood_evaluations.width;
        let trace_term_coeffs = &challenges.trace_term_coeffs;

        // Runtime guard: a malformed proof may supply opening evaluations whose
        // column count does not match the OOD table width, or whose composition
        // poly parts count does not match the proof's `composition_poly_parts_ood_evaluation`.
        // Without these checks the indexing below would panic in release builds.
        let num_precomp = lde_precomputed.len();
        let num_base = num_precomp + lde_main.len();
        if num_base + lde_trace_aux_evaluations.len() != ood_evaluations_table_width {
            return None;
        }
        if trace_term_coeffs.is_empty()
            || trace_term_coeffs.len() * trace_term_coeffs[0].len()
                != ood_evaluations_table_height * ood_evaluations_table_width
        {
            return None;
        }
        if trace_denoms_inv.len() != ood_evaluations_table_height {
            return None;
        }

        // Row-outer fold: the trace denominator `1/(x - z)` depends only on the
        // row (OOD step), so we factor it out of the per-column sum instead of
        // multiplying it into every cell. This turns the `W·H` inner products
        //   Σ_col Σ_row (diff·denom)·coeff
        // into the algebraically identical
        //   Σ_row denom·(Σ_col diff·coeff),
        // dropping one ext3 multiplication per cell (≈ 2× fewer ext3 muls in the
        // DEEP reconstruction, the dominant verifier arithmetic) and hoisting the
        // OOD `get_row` out of the column loop (W·H → H calls). `trace_term_coeffs`
        // is verifier-derived with shape `width × height`, so `[col][row]` indexing
        // is in bounds (the OOD-width guard above pins `width`).
        let trace_term =
            (0..ood_evaluations_table_height).fold(FieldElement::zero(), |trace_term, row_idx| {
                let ood_row = proof.trace_ood_evaluations.get_row(row_idx);
                let row_sum = (0..ood_evaluations_table_width).fold(
                    FieldElement::zero(),
                    |row_sum, col_idx| {
                        let ood_val = &ood_row[col_idx];
                        // Stay in base when we can: F: IsSubFieldOf<E> gives F - E -> E.
                        // Base columns are precomputed first, then main, then aux.
                        let diff: FieldElement<FieldExtension> = if col_idx < num_precomp {
                            &lde_precomputed[col_idx] - ood_val
                        } else if col_idx < num_base {
                            &lde_main[col_idx - num_precomp] - ood_val
                        } else {
                            &lde_trace_aux_evaluations[col_idx - num_base] - ood_val
                        };
                        let coeff = &trace_term_coeffs[col_idx][row_idx];
                        row_sum + &diff * coeff
                    },
                );
                trace_term + &row_sum * &trace_denoms_inv[row_idx]
            });

        let mut h_terms = FieldElement::zero();
        for (j, h_i_upsilon) in lde_composition_poly_parts_evaluation.iter().enumerate() {
            // Bounds-check via `.get(j)?`: a malformed opening may have more
            // parts than the proof header advertises.
            let h_i_zpower = proof.composition_poly_parts_ood_evaluation.get(j)?;
            let gamma = challenges.gammas.get(j)?;
            let h_i_term = (h_i_upsilon - h_i_zpower) * gamma;
            h_terms += h_i_term;
        }
        h_terms *= comp_denom_inv;

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
        Self::multi_verify_proofs(airs, &multi_proof.proofs, transcript, expected_bus_balance)
    }

    /// Slice-taking variant of [`Self::multi_verify`]. Callers that already
    /// hold a slice of proofs (or a single proof via [`core::slice::from_ref`])
    /// can call this directly without constructing a [`MultiProof`].
    fn multi_verify_proofs(
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        proofs: &[StarkProof<Field, FieldExtension, PI>],
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
        expected_bus_balance: &FieldElement<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        if airs.len() != proofs.len() {
            error!(
                "AIR count ({}) does not match proof count ({})",
                airs.len(),
                proofs.len()
            );
            return false;
        }

        // Check if any AIR has an auxiliary trace
        let needs_lookup_challenges = airs.iter().any(|air| air.has_aux_trace());

        // #####################################################################
        // ##### COMMON (shared, pre-fork) #####################################
        // #####################################################################
        // Everything below is computed ONCE on the shared transcript before any
        // per-table fork: main commitments are appended, the shared LogUp
        // challenges are sampled, the global alpha powers are derived, and the
        // bus_public_inputs layout is validated. Only after this section do we
        // fork the transcript per table. The exact sequence of transcript
        // operations here is soundness-critical (Fiat-Shamir) and must match
        // the prover byte-for-byte.

        // =====================================================================
        // Round 1, Phase A: Replay main trace commitments
        // =====================================================================
        // For preprocessed tables, use the hardcoded commitment (verifier cannot
        // trust the prover). For normal tables, use the commitment from the proof.

        for (idx, (air, proof)) in airs.iter().zip(proofs).enumerate() {
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

        // Compute the LogUp alpha powers ONCE, up to the global maximum bus
        // element count across all AIRs. `compute_alpha_powers` returns the
        // strict prefix sequence `[1, α, α², …]`, and the alpha challenge is
        // shared (identical) across all tables, so each table can reuse a
        // prefix slice of this global vector instead of recomputing the chain.
        let logup_alpha_powers_global: Vec<FieldElement<FieldExtension>> =
            if lookup_challenges.len() > LOGUP_CHALLENGE_ALPHA {
                let global_max_bus = airs.iter().map(|a| a.max_bus_elements()).max().unwrap_or(0);
                compute_alpha_powers(&lookup_challenges[LOGUP_CHALLENGE_ALPHA], global_max_bus)
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

        for (idx, (air, proof)) in airs.iter().zip(proofs).enumerate() {
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

        // #####################################################################
        // ##### PER-TABLE (forked transcript) #################################
        // #####################################################################
        // The shared/common section is finished. From here each table branches.
        //
        // Phase C + Rounds 2-4: Forked per table.
        // Each table gets an independent transcript fork (cloned from the shared
        // state after Phase B, domain-separated by table index). This matches
        // the prover's forking and makes per-table verification independent.

        for (idx, (air, proof)) in airs.iter().zip(proofs).enumerate() {
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
                &lookup_challenges,
                &logup_alpha_powers_global,
            ) {
                error!(
                    "Table {} failed verify_rounds_2_to_4 (num_constraints={}, trace_cols={})",
                    idx,
                    air.context().num_transition_constraints,
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
            for (air, proof) in airs.iter().zip(proofs) {
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
    {
        Self::multi_verify_proofs(
            &[air],
            core::slice::from_ref(proof),
            transcript,
            &FieldElement::zero(),
        )
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
        rap_challenges: &[FieldElement<FieldExtension>],
        logup_alpha_powers: &[FieldElement<FieldExtension>],
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

        // `replay_rounds_after_round_1` takes ownership of `rap_challenges`
        // (it is stored owned in the returned `Challenges`). Clone exactly once
        // here, where ownership is actually required — this removes the
        // per-table clone that previously lived at the `multi_verify` call site.
        let challenges = Self::replay_rounds_after_round_1(
            air,
            proof,
            &domain,
            transcript,
            rap_challenges.to_vec(),
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

        if !Self::step_2_verify_claimed_composition_polynomial(
            air,
            proof,
            &domain,
            &challenges,
            logup_alpha_powers,
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
