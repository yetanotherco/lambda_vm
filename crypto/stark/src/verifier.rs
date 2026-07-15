use super::{
    config::BatchedMerkleTreeBackend,
    domain::VerifierDomain,
    fri::{batched::derive_batched_fri_challenges, mmcs::MixedMmcs},
    grinding,
    proof::stark::StarkProof,
    traits::{AIR, TransitionEvaluationContext},
};
pub use crate::proof::view::PiDeserializer;
use crate::{
    config::Commitment,
    domain::new_verifier_domain,
    lookup::{BusPublicInputs, LOGUP_CHALLENGE_ALPHA, LOGUP_NUM_CHALLENGES, compute_alpha_powers},
    proof::stark::{ArchivedStarkProof, BatchedMultiProof, BatchedTableData, MultiProof},
    proof::view::{
        DeepPolynomialOpeningView, FriDecommitmentView, PolynomialOpeningsView, StarkProofView,
    },
};
use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use crypto::merkle_tree::proof::verify_merkle_path;
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
where
    Field::BaseType: math::field::element::NativeArchived,
    FieldExtension::BaseType: math::field::element::NativeArchived,
    PI: rkyv::Archive + Clone,
    <PI as rkyv::Archive>::Archived: rkyv::Deserialize<PI, PiDeserializer>,
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

/// Verifier state carried across the batched (unified-shard) round-4 seam:
/// everything `batched_verify_round_4` needs that rounds 1-3 derived from the
/// transcript (per-table domains/heights, the shared OOD point `z`, the round-2
/// constraint coefficients, and the shared LogUp challenges). Produced by
/// `batched_verify_rounds_1_to_3`; lets the continuation epoch verifier weave the
/// separate L2G lane in at the seam.
pub struct VmMidState<Field: IsFFTField, FieldExtension: IsField + Send + Sync> {
    pub(crate) domains: Vec<VerifierDomain<Field>>,
    pub(crate) heights: Vec<usize>,
    pub(crate) h_max: usize,
    pub(crate) tallest: usize,
    pub(crate) needs_lookup_challenges: bool,
    pub(crate) lookup_challenges: Vec<FieldElement<FieldExtension>>,
    pub(crate) boundary_coeffs_all: Vec<Vec<FieldElement<FieldExtension>>>,
    pub(crate) transition_coeffs_all: Vec<Vec<FieldElement<FieldExtension>>>,
    pub(crate) z: FieldElement<FieldExtension>,
}

pub type DeepPolynomialEvaluations<F> = (Vec<FieldElement<F>>, Vec<FieldElement<F>>);

// The verifier reads proofs in place from their rkyv archive; archived field
// elements are viewed as native ones, which is only valid on little-endian.
#[cfg(not(target_endian = "little"))]
compile_error!("the zero-copy STARK verifier requires a little-endian target");

/// The functionality of a STARK verifier providing methods to run the STARK Verify protocol
/// https://lambdaclass.github.io/lambdaworks/starks/protocol.html
///
/// Every method below takes proof data through a [`StarkProofView`] (and its
/// nested `*View` types), a borrowed view implemented once for a real owned
/// [`StarkProof`] and once for an rkyv-archived proof read in place. This is
/// the single verification implementation: [`Self::multi_verify`] (owned) and
/// [`Self::multi_verify_archived`] (archived, used by the recursion guest)
/// are thin entry points that build the matching view and share every
/// downstream check — no serialization, no duplicated logic.
pub trait IsStarkVerifier<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
    PI,
> where
    Field::BaseType: math::field::element::NativeArchived,
    FieldExtension::BaseType: math::field::element::NativeArchived,
    PI: rkyv::Archive + Clone,
    <PI as rkyv::Archive>::Archived: rkyv::Deserialize<PI, PiDeserializer>,
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
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
        public_inputs: &PI,
        domain: &VerifierDomain<Field>,
        challenges: &Challenges<FieldExtension>,
    ) -> bool {
        crate::profile_markers::step_marker::<
            { crate::profile_markers::STEP_VERIFY_CLAIMED_COMPOSITION_POLYNOMIAL },
        >();
        let trace_length = proof.trace_length();
        // Owned `BusPublicInputs` (just the table contribution L — one field
        // element) reconstructed for the AIR boundary call.
        let bus_public_inputs = proof
            .bus_table_contribution()
            .map(BusPublicInputs::from_contribution);
        let boundary_constraints = air.boundary_constraints(
            public_inputs,
            &challenges.rap_challenges,
            bus_public_inputs.as_ref(),
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
        let trace_ood_evaluations = proof.trace_ood_evaluations();
        let ood_row = trace_ood_evaluations.get_row(0);

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

        // A malformed archive can advertise fewer OOD columns than the AIR's
        // aux count; reject instead of underflowing.
        let num_main_trace_columns = match trace_ood_evaluations
            .width()
            .checked_sub(air.num_auxiliary_rap_columns())
        {
            Some(n) => n,
            None => return false,
        };

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
            Some(contribution) => {
                let n = FieldElement::<Field>::from(trace_length as u64);
                match n.inv() {
                    Ok(n_inv) => n_inv * &contribution,
                    Err(_) => return false, // trace_length == 0 is invalid
                }
            }
            None => FieldElement::zero(),
        };

        let ood_frame = trace_ood_evaluations.into_frame(num_main_trace_columns, air.step_size());
        let transition_evaluation_context = TransitionEvaluationContext::new_verifier(
            &ood_frame,
            &challenges.rap_challenges,
            &logup_alpha_powers,
            &logup_table_offset,
        );
        let transition_ood_frame_evaluations =
            air.compute_transition(&transition_evaluation_context);

        let mut denominators =
            vec![FieldElement::<FieldExtension>::zero(); air.num_transition_constraints()];
        air.constraints_meta().iter().for_each(|m| {
            denominators[m.constraint_idx] = crate::constraints::zerofier::evaluate_zerofier(
                m,
                &challenges.z,
                &domain.trace_primitive_root,
                trace_length,
            );
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
            .composition_poly_parts_ood_evaluation()
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, coeff| {
                acc * &challenges.z + coeff
            });

        composition_poly_claimed_ood_evaluation == composition_poly_ood_evaluation
    }

    /// The FRI fold layout for this proof, derived from options + domain.
    ///
    /// Delegates to the shared [`crate::fri::terminal::FriFoldLayout`] so the
    /// verifier's Fiat-Shamir replay and structural checks use exactly the same
    /// arithmetic as the CPU and GPU provers; drift between them would break all
    /// proofs. `VerifierDomain.lde_length` is the codeword size and
    /// `lde_length / trace_length` the blowup factor.
    // `FriFoldLayout` is a crate-internal helper type returned from a default method
    // of this public trait; the exposure is intentional (internal helper).
    #[allow(private_interfaces)]
    fn fri_termination_params(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        domain: &VerifierDomain<Field>,
    ) -> crate::fri::terminal::FriFoldLayout {
        let k = air.options().fri_final_poly_log_degree as u32;
        let blowup_log = (domain.lde_length / domain.trace_length).trailing_zeros();
        crate::fri::terminal::FriFoldLayout::new(domain.lde_length.trailing_zeros(), blowup_log, k)
    }

    /// Reconstructs the Deep composition polynomial evaluations at the challenge indices values using the provided
    /// openings of the trace polynomials and the composition polynomial parts. It then uses these to verify that the
    /// FRI decommitments are valid and correspond to the Deep composition polynomial.
    fn step_3_verify_fri(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
        domain: &VerifierDomain<Field>,
        challenges: &Challenges<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        crate::profile_markers::step_marker::<{ crate::profile_markers::STEP_VERIFY_FRI }>();
        let (deep_poly_evaluations, deep_poly_evaluations_sym) =
            match Self::reconstruct_deep_composition_poly_evaluations_for_all_queries(
                challenges, domain, proof,
            ) {
                Some(pair) => pair,
                None => return false,
            };

        // ---- Reconstruct the FRI terminal codeword from the final-poly coeffs ----
        // The prover folds the deep composition codeword down to a terminal
        // codeword of length `terminal_len = 2^(blowup_log + effective_k)` and sends
        // the `2^effective_k` coefficients of the low-degree polynomial it encodes.
        let layout = Self::fri_termination_params(air, domain);
        let num_committed = layout.num_committed;

        // Structural check: number of committed FRI layers must equal
        // `num_committed` (zero when no fold or a single final fold happened).
        if proof.fri_layers_merkle_roots().len() != num_committed {
            return false;
        }
        // Structural check: the final polynomial must have exactly `2^effective_k`
        // coefficients; otherwise the reconstruction below is ill-defined.
        if proof.fri_final_poly_coeffs().len() != (1usize << layout.effective_k) {
            return false;
        }
        // Structural check: every per-query FRI decommitment must carry exactly
        // `num_committed` layers. The fold loop in `verify_query_and_sym_openings`
        // zips these untrusted, variable-length vecs against the committed layer
        // roots, and they are NOT bound into the Fiat-Shamir transcript. Without
        // this check a prover could send them empty (making the fold run zero
        // iterations and accept the query vacuously) or padded (making the loop
        // skip the terminal low-degree check), bypassing FRI entirely. This length
        // check is the only thing that pins them, so it must run before the loop.
        if (0..proof.query_list_len()).any(|i| {
            let decommitment = proof.query(i);
            decommitment.layers_auth_paths_len() != num_committed
                || decommitment.layers_evaluations_sym().len() != num_committed
        }) {
            return false;
        }

        let terminal_offset = domain.coset_offset.pow(1u64 << layout.total_folds);
        let terminal_codeword =
            crate::fri::terminal::terminal_codeword_from_coeffs::<Field, FieldExtension>(
                proof.fri_final_poly_coeffs(),
                &terminal_offset,
                layout.terminal_len,
            );

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

        (0..challenges.iotas.len())
            .zip(evaluation_point_inverse)
            .all(|(i, eval)| {
                Self::verify_query_and_sym_openings(
                    proof,
                    &challenges.zetas,
                    challenges.iotas[i],
                    proof.query(i),
                    eval,
                    &deep_poly_evaluations[i],
                    &deep_poly_evaluations_sym[i],
                    &terminal_codeword,
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

    /// Verify a row-paired `PolynomialOpenings` against `root`. The row pair
    /// (`2·iota`, `2·iota+1`) is committed as the single leaf at position `iota`,
    /// so one Merkle path authenticates both `evaluations` (the row) and
    /// `evaluations_sym` (its symmetric). Same layout used for trace and composition.
    fn verify_opening_pair<E>(
        opening: PolynomialOpeningsView<'_, E>,
        root: &Commitment,
        iota: usize,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<E>: AsBytes + Sync + Send,
        E: IsField,
        E::BaseType: math::field::element::NativeArchived,
        Field: IsSubFieldOf<E>,
    {
        let mut value = opening.evaluations().to_vec();
        value.extend_from_slice(opening.evaluations_sym());
        verify_merkle_path::<BatchedMerkleTreeBackend<E>>(opening.merkle_path(), root, iota, &value)
    }

    /// Verify opening Open(tⱼ(D_LDE), 𝜐) and Open(tⱼ(D_LDE), -𝜐) for all trace polynomials tⱼ,
    /// where 𝜐 and -𝜐 are the elements corresponding to the index challenge `iota`.
    fn verify_trace_openings(
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
        deep_poly_openings: DeepPolynomialOpeningView<'_, Field, FieldExtension>,
        iota: usize,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        // Main trace (multiplicities for preprocessed, full trace for normal).
        let mut ok = Self::verify_opening_pair::<Field>(
            deep_poly_openings.main_trace_polys(),
            proof.lde_trace_main_merkle_root(),
            iota,
        );

        // Precomputed trace (preprocessed tables only). Mismatched presence is
        // unreachable in practice (multi_verify rejects such proofs upstream),
        // but a defensive check keeps this function self-contained.
        ok &= match (
            proof.lde_trace_precomputed_merkle_root(),
            deep_poly_openings.precomputed_trace_polys(),
        ) {
            (Some(root), Some(opening)) => Self::verify_opening_pair::<Field>(opening, root, iota),
            (None, None) => true,
            _ => false,
        };

        // Auxiliary trace.
        ok &= match (
            proof.lde_trace_aux_merkle_root(),
            deep_poly_openings.aux_trace_polys(),
        ) {
            (Some(root), Some(opening)) => {
                Self::verify_opening_pair::<FieldExtension>(opening, root, iota)
            }
            (None, None) => true,
            _ => false,
        };

        ok
    }

    /// Verify opening Open(Hᵢ(D_LDE), 𝜐) and Open(Hᵢ(D_LDE), -𝜐) for all parts Hᵢof the composition
    /// polynomial, where 𝜐 and -𝜐 are the elements corresponding to the index challenge `iota`.
    fn verify_composition_poly_opening(
        deep_poly_openings: DeepPolynomialOpeningView<'_, Field, FieldExtension>,
        composition_poly_merkle_root: &Commitment,
        iota: &usize,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let composition_poly = deep_poly_openings.composition_poly();
        let mut value = composition_poly.evaluations().to_vec();
        value.extend_from_slice(composition_poly.evaluations_sym());

        verify_merkle_path::<BatchedMerkleTreeBackend<FieldExtension>>(
            composition_poly.merkle_path(),
            composition_poly_merkle_root,
            *iota,
            &value,
        )
    }

    /// Verifies the validity of the purported values of the trace polynomials and the composition polynomial
    /// parts at the domain elements and their symmetric counterparts corresponding to all the FRI query
    /// index challenges.
    fn step_4_verify_trace_and_composition_openings(
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
        challenges: &Challenges<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        crate::profile_markers::step_marker::<
            { crate::profile_markers::STEP_VERIFY_TRACE_AND_COMPOSITION_OPENINGS },
        >();
        // `step_3_verify_fri` (which runs before this) already rejects proofs
        // whose `deep_poly_openings` is shorter than `challenges.iotas`.
        challenges.iotas.iter().enumerate().all(|(i, iota_n)| {
            let deep_poly_opening = proof.deep_poly_opening(i);
            Self::verify_composition_poly_opening(
                deep_poly_opening,
                proof.composition_poly_root(),
                iota_n,
            ) && Self::verify_trace_openings(proof, deep_poly_opening, *iota_n)
        })
    }

    /// Verifies the openings of a fold polynomial of an inner layer of FRI.
    fn verify_fri_layer_openings(
        merkle_root: &Commitment,
        auth_path_sym: &[Commitment],
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

        verify_merkle_path::<BatchedMerkleTreeBackend<FieldExtension>>(
            auth_path_sym,
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
    #[allow(clippy::too_many_arguments)]
    fn verify_query_and_sym_openings(
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
        zetas: &[FieldElement<FieldExtension>],
        iota: usize,
        fri_decommitment: FriDecommitmentView<'_, FieldExtension>,
        evaluation_point_inv: FieldElement<Field>,
        deep_composition_evaluation: &FieldElement<FieldExtension>,
        deep_composition_evaluation_sym: &FieldElement<FieldExtension>,
        terminal_codeword: &[FieldElement<FieldExtension>],
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let fri_layers_merkle_roots = proof.fri_layers_merkle_roots();

        let p0_eval = deep_composition_evaluation;
        let p0_eval_sym = deep_composition_evaluation_sym;

        // No-fold (clamp) case: the codeword never folds (`total_folds == 0`), so
        // no folding challenges were drawn and the terminal codeword *is* the deep
        // composition codeword p₀ itself. The query's two points 𝜐 and -𝜐 sit at
        // FRI-order positions `iota*2` and `iota*2 + 1` of the terminal codeword.
        if zetas.is_empty() {
            return terminal_codeword
                .get(iota * 2)
                .is_some_and(|t| p0_eval == t)
                && terminal_codeword
                    .get(iota * 2 + 1)
                    .is_some_and(|t| p0_eval_sym == t);
        }

        let evaluation_point_vec: Vec<FieldElement<Field>> =
            core::iter::successors(Some(evaluation_point_inv.square()), |evaluation_point| {
                Some(evaluation_point.square())
            })
            .take(fri_layers_merkle_roots.len())
            .collect();

        // Reconstruct p₁(𝜐²)
        let mut v =
            (p0_eval + p0_eval_sym) + evaluation_point_inv * &zetas[0] * (p0_eval - p0_eval_sym);
        let mut index = iota;

        // Fold through every committed layer: use the proof to verify the openings
        // of pᵢ(−𝜐^(2ⁱ)) (given by the prover) and pᵢ(𝜐^(2ⁱ)) (computed on the
        // previous iteration), then obtain pᵢ₊₁(𝜐^(2ⁱ⁺¹)). When there are no
        // committed layers (`total_folds == 1`, a single final fold) this fold is
        // empty and `v`/`index` already hold the terminal-layer value/position.
        let openings_ok = fri_layers_merkle_roots
            .iter()
            .zip(fri_decommitment.layers_evaluations_sym())
            .zip(evaluation_point_vec)
            .enumerate()
            .fold(
                true,
                |result, (i, ((merkle_root, evaluation_sym), evaluation_point_inv))| {
                    // Verify opening Open(pᵢ(Dₖ), −𝜐^(2ⁱ)) and Open(pᵢ(Dₖ), 𝜐^(2ⁱ)).
                    // `v` is pᵢ(𝜐^(2ⁱ)).
                    // `evaluation_sym` is pᵢ(−𝜐^(2ⁱ)).
                    let openings_ok = Self::verify_fri_layer_openings(
                        merkle_root,
                        fri_decommitment.layer_auth_path(i),
                        &v,
                        evaluation_sym,
                        index,
                    );

                    // Update `v` with next value pᵢ₊₁(𝜐^(2ⁱ⁺¹)).
                    v = (&v + evaluation_sym)
                        + evaluation_point_inv * &zetas[i + 1] * (&v - evaluation_sym);

                    // Update index for next iteration. The index of the squares in the next layer
                    // is obtained by halving the current index. This is due to the bit-reverse
                    // ordering of the elements in the Merkle tree.
                    index >>= 1;

                    result & openings_ok
                },
            );

        // After folding through all committed layers, `v` is the query's value at
        // the terminal layer and `index` its FRI-order position there. Check it
        // against the reconstructed terminal codeword. This single check covers
        // both the single-fold (`total_folds == 1`, empty fold above) and
        // multi-fold regimes; `.get()` fails closed on an out-of-range index.
        let terminal_ok = terminal_codeword.get(index).is_some_and(|t| &v == t);
        openings_ok & terminal_ok
    }

    fn reconstruct_deep_composition_poly_evaluations_for_all_queries(
        challenges: &Challenges<FieldExtension>,
        domain: &VerifierDomain<Field>,
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
    ) -> Option<DeepPolynomialEvaluations<FieldExtension>> {
        let num_queries = challenges.iotas.len();

        // `deep_poly_openings` comes straight from the untrusted proof and its
        // length is not otherwise pinned (the `query_list.len()` guard checks a
        // different field). The loop below indexes `deep_poly_openings[i]` for
        // every `i` in `0..num_queries`, so a truncated Vec would panic the
        // verifier with an out-of-bounds index on a malicious proof. Reject
        // instead. (Extra entries are harmless — they are never indexed —
        // matching the `<` convention of the `query_list` guard.)
        if proof.deep_poly_openings_len() < num_queries {
            return None;
        }

        let mut deep_poly_evaluations = Vec::with_capacity(num_queries);
        let mut deep_poly_evaluations_sym = Vec::with_capacity(num_queries);

        // Build the base-field LDE evaluations as concatenated slice (precomputed + main)
        // without lifting to the extension field. The helper now subtracts directly via
        // the F: IsSubFieldOf<E> Sub impl, so we avoid a per-query base->extension lift.
        let primitive_root = &Field::get_primitive_root_of_unity(domain.root_order as u64)
            .expect("verifier domain root_order is a valid power of two");

        for (i, iota) in challenges.iotas.iter().enumerate() {
            let opening = proof.deep_poly_opening(i);

            // Base-field portion: precomputed columns FIRST, then main trace columns.
            let mut lde_base: Vec<FieldElement<Field>> = Vec::new();
            if let Some(p) = opening.precomputed_trace_polys() {
                lde_base.extend_from_slice(p.evaluations());
            }
            lde_base.extend_from_slice(opening.main_trace_polys().evaluations());

            let lde_aux: &[FieldElement<FieldExtension>] = opening
                .aux_trace_polys()
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
                opening.composition_poly().evaluations(),
            )?);

            // Mirror for the symmetric query point.
            let mut lde_base_sym: Vec<FieldElement<Field>> = Vec::new();
            if let Some(p) = opening.precomputed_trace_polys() {
                lde_base_sym.extend_from_slice(p.evaluations_sym());
            }
            lde_base_sym.extend_from_slice(opening.main_trace_polys().evaluations_sym());

            let lde_aux_sym: &[FieldElement<FieldExtension>] = opening
                .aux_trace_polys()
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
                opening.composition_poly().evaluations_sym(),
            )?);
        }
        Some((deep_poly_evaluations, deep_poly_evaluations_sym))
    }

    fn reconstruct_deep_composition_poly_evaluation(
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
        evaluation_point: &FieldElement<Field>,
        primitive_root: &FieldElement<Field>,
        challenges: &Challenges<FieldExtension>,
        lde_trace_base_evaluations: &[FieldElement<Field>],
        lde_trace_aux_evaluations: &[FieldElement<FieldExtension>],
        lde_composition_poly_parts_evaluation: &[FieldElement<FieldExtension>],
    ) -> Option<FieldElement<FieldExtension>> {
        let trace_ood_evaluations = proof.trace_ood_evaluations();
        let ood_evaluations_table_height = trace_ood_evaluations.height();
        let ood_evaluations_table_width = trace_ood_evaluations.width();
        // Hot loop below: resolve the OOD data to one flat slice once instead
        // of re-deriving a row slice per element.
        let ood_data = trace_ood_evaluations.row_major_data();
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
                        let ood_val = &ood_data[row_idx * ood_evaluations_table_width + col_idx];
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

        let composition_parts_ood = proof.composition_poly_parts_ood_evaluation();
        let number_of_parts = lde_composition_poly_parts_evaluation.len();
        let z_pow = &challenges.z.pow(number_of_parts);

        // A malformed proof can make evaluation_point == z^N, reject.
        let denom_composition = (evaluation_point - z_pow).inv().ok()?;
        let mut h_terms = FieldElement::zero();
        for (j, h_i_upsilon) in lde_composition_poly_parts_evaluation.iter().enumerate() {
            // Bounds-check via `.get(j)?`: a malformed opening may have more
            // parts than the proof header advertises.
            let h_i_zpower = composition_parts_ood.get(j)?;
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
        let views: Vec<StarkProofView<Field, FieldExtension, PI>> = multi_proof
            .proofs
            .iter()
            .map(StarkProofView::Owned)
            .collect();
        Self::multi_verify_views(airs, &views, transcript, expected_bus_balance)
    }

    /// Verifies one or more rkyv-archived STARK proofs read **in place** from
    /// their archive buffer — no proof deserialization, no per-field allocation.
    fn multi_verify_archived(
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        proofs: &[ArchivedStarkProof<Field, FieldExtension, PI>],
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
        expected_bus_balance: &FieldElement<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let views: Vec<StarkProofView<Field, FieldExtension, PI>> =
            proofs.iter().map(StarkProofView::Archived).collect();
        Self::multi_verify_views(airs, &views, transcript, expected_bus_balance)
    }

    /// The single verification implementation, shared by [`Self::multi_verify`]
    /// (owned) and [`Self::multi_verify_archived`] (archived), operating on
    /// proof views rather than either's concrete type.
    fn multi_verify_views(
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        proofs: &[StarkProofView<Field, FieldExtension, PI>],
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

        // =====================================================================
        // Round 1, Phase A: Replay main trace commitments
        // =====================================================================
        // For preprocessed tables, use the hardcoded commitment (verifier cannot
        // trust the prover). For normal tables, use the commitment from the proof.

        for (idx, (air, proof)) in airs.iter().zip(proofs).enumerate() {
            let proof = *proof;
            // Soundness: the number of composition-poly parts is fixed by the AIR's
            // degree bound, NOT chosen by the prover. Deriving it from the proof would
            // let a malicious prover inflate the part count, widening the composition
            // polynomial's degree space and weakening the low-degree test. Reject any
            // proof whose advertised part count disagrees with the AIR.
            let trace_length = proof.trace_length();
            if trace_length == 0
                || proof.composition_poly_parts_ood_evaluation().len()
                    != air.composition_poly_degree_bound(trace_length) / trace_length
            {
                return false;
            }
            // The archive is read in place without validation; reject an OOD
            // table whose advertised dimensions disagree with its data length,
            // has no rows, or whose height isn't a whole number of AIR steps
            // (which `into_frame` below only `debug_assert!`s, not checks) —
            // all before any row access indexes into it.
            let trace_ood_evaluations = proof.trace_ood_evaluations();
            if !trace_ood_evaluations.dimensions_consistent()
                || trace_ood_evaluations.height() == 0
                || !trace_ood_evaluations
                    .height()
                    .is_multiple_of(air.step_size())
            {
                return false;
            }
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

        for (idx, (air, proof)) in airs.iter().zip(proofs).enumerate() {
            let proof = *proof;
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

        for (idx, (air, proof)) in airs.iter().zip(proofs).enumerate() {
            let proof = *proof;
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
            if let Some(contribution) = proof.bus_table_contribution() {
                table_transcript.append_field_element(&contribution);
            }

            // The AIR API takes owned public inputs; materialize the (tiny) PI.
            // For the VM verifier `PI = ()` and this is a no-op.
            let public_inputs: PI = match proof.public_inputs() {
                Some(pi) => pi,
                None => return false,
            };

            // Rounds 2-4: verify
            if !Self::verify_rounds_2_to_4(
                *air,
                proof,
                &public_inputs,
                &mut table_transcript,
                lookup_challenges.clone(),
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
                    && let Some(contribution) = proof.bus_table_contribution()
                {
                    total += contribution;
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
        Self::multi_verify_views(
            &[air],
            &[StarkProofView::Owned(proof)],
            transcript,
            &FieldElement::zero(),
        )
    }

    /// Replays rounds 2, 3 and 4 of the protocol for a given proof, assuming round 1 has
    /// already been replayed and the RAP challenges are known.
    fn replay_rounds_after_round_1(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
        public_inputs: &PI,
        domain: &VerifierDomain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        rap_challenges: Vec<FieldElement<FieldExtension>>,
    ) -> Challenges<FieldExtension>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        crate::profile_markers::step_marker::<
            { crate::profile_markers::STEP_REPLAY_ROUNDS_AFTER_ROUND_1 },
        >();
        // ===================================
        // ==========|   Round 2   |==========
        // ===================================

        // <<<< Receive challenge: 𝛽
        let beta = transcript.sample_field_element();
        let trace_length = proof.trace_length();
        let bus_public_inputs = proof
            .bus_table_contribution()
            .map(BusPublicInputs::from_contribution);
        let num_boundary_constraints = air
            .boundary_constraints(
                public_inputs,
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
        // Column-major append (matches `Table::columns()` order) reading the
        // rows in place, without materializing transposed columns.
        let ood = proof.trace_ood_evaluations();
        for col_idx in 0..ood.width() {
            for row_idx in 0..ood.height() {
                transcript.append_field_element(&ood.get_row(row_idx)[col_idx]);
            }
        }
        // <<<< Receive value: Hᵢ(z^N)
        for element in proof.composition_poly_parts_ood_evaluation().iter() {
            transcript.append_field_element(element);
        }

        // ===================================
        // ==========|   Round 4   |==========
        // ===================================

        let num_terms_composition_poly = proof.composition_poly_parts_ood_evaluation().len();
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

        // The prover only samples the final-fold challenge when the codeword
        // actually folds past the committed layers. For tiny traces (the clamp
        // case) no fold happens, so no challenge is drawn. This must mirror the
        // prover's `commit_phase_from_evaluations` exactly.
        let total_folds = Self::fri_termination_params(air, domain).total_folds;

        // >>>> Send final-fold challenge 𝜁_final (only when folding occurs)
        if total_folds > 0 {
            zetas.push(transcript.sample_field_element());
        }

        // <<<< Receive the FRI final-polynomial coefficients (same Vec, same
        // order the prover appended them in `commit_phase_from_evaluations`).
        for c in proof.fri_final_poly_coeffs() {
            transcript.append_field_element(c);
        }

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
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
        public_inputs: &PI,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
        rap_challenges: Vec<FieldElement<FieldExtension>>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
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

        let challenges = Self::replay_rounds_after_round_1(
            air,
            proof,
            public_inputs,
            &domain,
            transcript,
            rap_challenges,
        );

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
            public_inputs,
            &domain,
            &challenges,
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

        if !Self::step_3_verify_fri(air, proof, &domain, &challenges) {
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

    /// Build a lightweight per-table `StarkProof` carrying only the fields
    /// `step_2_verify_claimed_composition_polynomial` and
    /// `reconstruct_deep_composition_poly_evaluation` actually read
    /// (trace_length, OOD evaluations, precomputed root, bus/public inputs). All
    /// commitment/opening/FRI fields are placeholders those two helpers never
    /// inspect — this lets the batched verifier reuse them unchanged.
    fn batched_synthetic_table_proof(
        table: &BatchedTableData<FieldExtension, PI>,
    ) -> StarkProof<Field, FieldExtension, PI>
    where
        PI: Clone,
    {
        StarkProof {
            trace_length: table.trace_length,
            lde_trace_main_merkle_root: [0u8; 32],
            lde_trace_aux_merkle_root: None,
            lde_trace_precomputed_merkle_root: table.precomputed_root,
            trace_ood_evaluations: table.trace_ood_evaluations.clone(),
            composition_poly_root: [0u8; 32],
            composition_poly_parts_ood_evaluation: table
                .composition_poly_parts_ood_evaluation
                .clone(),
            fri_layers_merkle_roots: Vec::new(),
            fri_final_poly_coeffs: Vec::new(),
            query_list: Vec::new(),
            deep_poly_openings: Vec::new(),
            nonce: None,
            bus_public_inputs: table.bus_public_inputs.clone(),
            public_inputs: table.public_inputs.clone(),
        }
    }

    /// Verify a `BatchedMultiProof` (unified-shard): ONE linear transcript, ONE
    /// shared OOD point z, and ONE FRI over the height-combined per-table DEEP
    /// codewords, with all tables opened from three shared mixed-height MMCS
    /// trees per query. Mirrors `Prover::multi_prove_batched`.
    #[allow(clippy::too_many_lines)]
    fn batched_multi_verify(
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        proof: &BatchedMultiProof<Field, FieldExtension, PI>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
        expected_bus_balance: &FieldElement<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
        PI: Clone,
    {
        let mid = match Self::batched_verify_rounds_1_to_3(airs, proof, transcript) {
            Some(m) => m,
            None => return false,
        };
        Self::batched_verify_round_4(mid, airs, proof, transcript, expected_bus_balance)
    }

    /// Rounds 1-3 of the batched (unified-shard) verifier: replays the Fiat-Shamir
    /// transcript from Phase A (preprocessed roots + the single main MMCS root)
    /// through the OOD absorption, returning the derived `VmMidState` that round 4
    /// consumes. Split out of `batched_multi_verify` (behavior-preserving) so the
    /// continuation epoch verifier can weave the separate L2G lane in at the seam.
    /// Returns `None` on any structural rejection.
    fn batched_verify_rounds_1_to_3(
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        proof: &BatchedMultiProof<Field, FieldExtension, PI>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
    ) -> Option<VmMidState<Field, FieldExtension>>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
        PI: Clone,
    {
        let num_tables = airs.len();
        if num_tables == 0 || num_tables != proof.per_table.len() {
            return None;
        }

        // Per-table lightweight domains + FRI heights (= lde_log_height).
        let domains: Vec<VerifierDomain<Field>> = airs
            .iter()
            .zip(&proof.per_table)
            .map(|(air, t)| new_verifier_domain(*air, t.trace_length))
            .collect();
        let heights: Vec<usize> = domains
            .iter()
            .map(|d| d.lde_length.trailing_zeros() as usize)
            .collect();
        let h_max = *heights.iter().max().expect("num_tables > 0");
        // Any tallest table works: all tables at h_max share identical domain
        // params (global blowup + coset_offset), so z, the FRI point and the
        // query domain are the same whichever we pick. Mirrors the prover's
        // `max_by_key` choice (which value it lands on is immaterial here).
        let tallest = heights
            .iter()
            .position(|h| *h == h_max)
            .expect("h_max is present");

        let needs_lookup_challenges = airs.iter().any(|air| air.has_aux_trace());

        // ===== Round 1 replay =====
        // Phase A: per preprocessed table, append its hardcoded precomputed root
        // (checked against the AIR), then the SINGLE batched main-trace MMCS root.
        for (air, t) in airs.iter().zip(&proof.per_table) {
            // Soundness: composition part count is fixed by the AIR degree bound,
            // not chosen by the prover.
            if t.trace_length == 0
                || t.composition_poly_parts_ood_evaluation.len()
                    != air.composition_poly_degree_bound(t.trace_length) / t.trace_length
            {
                return None;
            }
            if air.is_preprocessed() {
                let expected = air.precomputed_commitment();
                match t.precomputed_root {
                    Some(actual) if actual == expected => {}
                    _ => return None,
                }
                transcript.append_bytes(&expected);
            } else if t.precomputed_root.is_some() {
                return None;
            }
        }
        transcript.append_bytes(&proof.main_root);

        // Bus-input presence must match the AIR layout (a dishonest prover could
        // omit bus_public_inputs to bypass the balance check).
        for (air, t) in airs.iter().zip(&proof.per_table) {
            if air.has_trace_interaction() != t.bus_public_inputs.is_some() {
                return None;
            }
        }

        // Phase B: shared LogUp challenges.
        let lookup_challenges: Vec<FieldElement<FieldExtension>> = if needs_lookup_challenges {
            (0..LOGUP_NUM_CHALLENGES)
                .map(|_| transcript.sample_field_element())
                .collect()
        } else {
            Vec::new()
        };

        // Phase C: single batched aux MMCS root (present iff any table has aux).
        if needs_lookup_challenges != proof.aux_root.is_some() {
            return None;
        }
        if let Some(root) = proof.aux_root {
            transcript.append_bytes(&root);
        }

        // Bus contributions bind before the round-2 challenges.
        for t in &proof.per_table {
            if let Some(bpi) = &t.bus_public_inputs {
                transcript.append_field_element(&bpi.table_contribution);
            }
        }

        // ===== Round 2: per-table beta -> boundary/transition coeffs, then the
        // single batched composition MMCS root. =====
        let mut boundary_coeffs_all: Vec<Vec<FieldElement<FieldExtension>>> =
            Vec::with_capacity(num_tables);
        let mut transition_coeffs_all: Vec<Vec<FieldElement<FieldExtension>>> =
            Vec::with_capacity(num_tables);
        for (air, t) in airs.iter().zip(&proof.per_table) {
            let beta = transcript.sample_field_element();
            let num_boundary = air
                .boundary_constraints(
                    &t.public_inputs,
                    &lookup_challenges,
                    t.bus_public_inputs.as_ref(),
                    t.trace_length,
                )
                .constraints
                .len();
            let num_transition = air.context().num_transition_constraints;
            let mut coeffs = compute_alpha_powers(&beta, num_boundary + num_transition);
            let transition_coeffs: Vec<_> = coeffs.drain(..num_transition).collect();
            transition_coeffs_all.push(transition_coeffs);
            boundary_coeffs_all.push(coeffs);
        }
        transcript.append_bytes(&proof.composition_root);

        // ===== Round 3: shared z (tallest domain), per-table OOD absorbed. =====
        let z = transcript.sample_z_ood_with_domain_params(
            domains[tallest].trace_length,
            domains[tallest].lde_length,
            &domains[tallest].coset_offset,
        );
        for t in &proof.per_table {
            for col in t.trace_ood_evaluations.columns().iter() {
                for elem in col.iter() {
                    transcript.append_field_element(elem);
                }
            }
            for elem in t.composition_poly_parts_ood_evaluation.iter() {
                transcript.append_field_element(elem);
            }
        }

        Some(VmMidState {
            domains,
            heights,
            h_max,
            tallest,
            needs_lookup_challenges,
            lookup_challenges,
            boundary_coeffs_all,
            transition_coeffs_all,
            z,
        })
    }

    /// Round 4 of the batched (unified-shard) verifier: the FRI + query phase over
    /// the height-combined per-table DEEP codewords, plus the bus-balance check.
    /// Split out of `batched_multi_verify` (behavior-preserving) so the
    /// continuation epoch verifier can run it AFTER the L2G lane at the seam.
    fn batched_verify_round_4(
        mid: VmMidState<Field, FieldExtension>,
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        proof: &BatchedMultiProof<Field, FieldExtension, PI>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
        expected_bus_balance: &FieldElement<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
        PI: Clone,
    {
        let VmMidState {
            domains,
            heights,
            h_max,
            tallest,
            needs_lookup_challenges,
            lookup_challenges,
            boundary_coeffs_all,
            transition_coeffs_all,
            z,
        } = mid;
        let num_tables = airs.len();

        // ===== Round 4: shared gamma, per-table DEEP coeffs, batched FRI challenges. =====
        let gamma = transcript.sample_field_element();
        let mut trace_term_coeffs_all: Vec<Vec<Vec<FieldElement<FieldExtension>>>> =
            Vec::with_capacity(num_tables);
        let mut gammas_all: Vec<Vec<FieldElement<FieldExtension>>> = Vec::with_capacity(num_tables);
        for (air, t) in airs.iter().zip(&proof.per_table) {
            let num_terms_comp = t.composition_poly_parts_ood_evaluation.len();
            let num_terms_trace = air.context().transition_offsets.len()
                * air.step_size()
                * air.context().trace_columns;
            let mut coeffs: Vec<_> =
                core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
                    .take(num_terms_comp + num_terms_trace)
                    .collect();
            let trace_term_coeffs: Vec<_> = coeffs
                .drain(..num_terms_trace)
                .collect::<Vec<_>>()
                .chunks(air.context().transition_offsets.len() * air.step_size())
                .map(|c| c.to_vec())
                .collect();
            trace_term_coeffs_all.push(trace_term_coeffs);
            gammas_all.push(coeffs);
        }

        let grinding_factor = airs[0].context().proof_options.grinding_factor;
        let num_queries = airs[0].options().fri_number_of_queries;
        let fri_domain_size = 1usize << h_max;
        let fri_challenges = derive_batched_fri_challenges(
            transcript,
            &heights,
            &proof.fri_layers_merkle_roots,
            &proof.fri_last_value,
            grinding_factor,
            proof.nonce,
            num_queries,
            fri_domain_size,
        );
        let alpha = fri_challenges.alpha;
        let betas_fri = fri_challenges.betas;
        let iotas = fri_challenges.iotas;

        // Grinding.
        if grinding_factor > 0 {
            let ok = proof.nonce.is_some_and(|n| {
                grinding::is_valid_nonce(&fri_challenges.grinding_seed, n, grinding_factor)
            });
            if !ok {
                return false;
            }
        }

        if proof.query_list.len() < num_queries || proof.deep_poly_openings.len() < num_queries {
            return false;
        }

        // Per-table synthetic proofs + Challenges (reused by step 2 and the query loop).
        let synth_proofs: Vec<StarkProof<Field, FieldExtension, PI>> = proof
            .per_table
            .iter()
            .map(Self::batched_synthetic_table_proof)
            .collect();
        let table_challenges: Vec<Challenges<FieldExtension>> = (0..num_tables)
            .map(|i| Challenges {
                z: z.clone(),
                boundary_coeffs: boundary_coeffs_all[i].clone(),
                transition_coeffs: transition_coeffs_all[i].clone(),
                trace_term_coeffs: trace_term_coeffs_all[i].clone(),
                gammas: gammas_all[i].clone(),
                zetas: Vec::new(),
                iotas: Vec::new(),
                rap_challenges: lookup_challenges.clone(),
                grinding_seed: [0u8; 32],
            })
            .collect();

        // ===== Step 2 (claimed composition polynomial) per table. =====
        for i in 0..num_tables {
            if !Self::step_2_verify_claimed_composition_polynomial(
                airs[i],
                StarkProofView::Owned(&synth_proofs[i]),
                &synth_proofs[i].public_inputs,
                &domains[i],
                &table_challenges[i],
            ) {
                return false;
            }
        }

        // MMCS binding data (all public / from the AIRs).
        // Committed main-split width per table = full main columns minus the
        // precomputed prefix. `context().trace_columns` counts every committed
        // trace column (main + aux), so subtracting the aux and precomputed
        // counts yields the main-split width. All three are AIR-intrinsic (not
        // proof-supplied), so this binds the MMCS leaf boundaries independently
        // of the prover. NB: `trace_layout().0` is NOT usable here — for
        // step-packed AIRs (e.g. BitFlags) it is a logical layout figure, not
        // the physical column count.
        let main_widths: Vec<usize> = airs
            .iter()
            .map(|a| {
                a.context().trace_columns
                    - a.num_auxiliary_rap_columns()
                    - a.num_precomputed_columns()
            })
            .collect();
        let comp_widths: Vec<usize> = proof
            .per_table
            .iter()
            .map(|t| t.composition_poly_parts_ood_evaluation.len())
            .collect();
        let aux_indices: Vec<usize> = (0..num_tables)
            .filter(|&i| airs[i].has_aux_trace())
            .collect();
        let aux_heights: Vec<usize> = aux_indices.iter().map(|&i| heights[i]).collect();
        let aux_widths: Vec<usize> = aux_indices
            .iter()
            .map(|&i| airs[i].num_auxiliary_rap_columns())
            .collect();
        let precomputed_indices: Vec<usize> = (0..num_tables)
            .filter(|&i| airs[i].is_preprocessed())
            .collect();

        // alpha^i powers for the cross-table combination.
        let mut alpha_pows: Vec<FieldElement<FieldExtension>> = Vec::with_capacity(num_tables);
        {
            let mut cur = FieldElement::<FieldExtension>::one();
            for _ in 0..num_tables {
                alpha_pows.push(cur.clone());
                cur = &cur * &alpha;
            }
        }
        let num_layers = proof.fri_layers_merkle_roots.len();

        // ===== Per query: MMCS openings, DEEP reconstruction, fold-and-inject FRI. =====
        for (q, &iota) in iotas.iter().enumerate() {
            let qo = &proof.deep_poly_openings[q];

            // 1) Authenticate the shared per-phase openings.
            if !MixedMmcs::<Field>::verify_batch(
                &proof.main_root,
                iota,
                &qo.main,
                &heights,
                &main_widths,
            ) {
                return false;
            }
            match (&proof.aux_root, &qo.aux) {
                (Some(root), Some(aux_op)) => {
                    if !MixedMmcs::<FieldExtension>::verify_batch(
                        root,
                        iota,
                        aux_op,
                        &aux_heights,
                        &aux_widths,
                    ) {
                        return false;
                    }
                }
                (None, None) => {}
                _ => return false,
            }
            if !MixedMmcs::<FieldExtension>::verify_batch(
                &proof.composition_root,
                iota,
                &qo.composition,
                &heights,
                &comp_widths,
            ) {
                return false;
            }

            // Precomputed openings (one per preprocessed table, in that order).
            if qo.precomputed.len() != precomputed_indices.len() {
                return false;
            }
            for (pc, &ti) in precomputed_indices.iter().enumerate() {
                let root = match proof.per_table[ti].precomputed_root {
                    Some(r) => r,
                    None => return false,
                };
                let local = iota >> (h_max - heights[ti]);
                if !Self::verify_opening_pair::<Field>(
                    PolynomialOpeningsView::Owned(&qo.precomputed[pc]),
                    &root,
                    local,
                ) {
                    return false;
                }
            }

            // 2) Reconstruct each table's DEEP value at its opened row pair.
            let mut deep_primary = vec![FieldElement::<FieldExtension>::zero(); num_tables];
            let mut deep_sym = vec![FieldElement::<FieldExtension>::zero(); num_tables];
            for i in 0..num_tables {
                let leaf = iota >> (h_max - heights[i]);
                let main_op = &qo.main.per_matrix[i];
                let comp_op = &qo.composition.per_matrix[i];
                let precomp_op = precomputed_indices
                    .iter()
                    .position(|&x| x == i)
                    .map(|pc| &qo.precomputed[pc]);
                let aux_op = aux_indices
                    .iter()
                    .position(|&x| x == i)
                    .and_then(|ai| qo.aux.as_ref().map(|a| &a.per_matrix[ai]));

                let mut base_p: Vec<FieldElement<Field>> = Vec::new();
                let mut base_s: Vec<FieldElement<Field>> = Vec::new();
                if let Some(p) = precomp_op {
                    base_p.extend_from_slice(&p.evaluations);
                    base_s.extend_from_slice(&p.evaluations_sym);
                }
                base_p.extend_from_slice(&main_op.evaluations);
                base_s.extend_from_slice(&main_op.evaluations_sym);
                let aux_p: &[FieldElement<FieldExtension>] =
                    aux_op.map(|a| a.evaluations.as_slice()).unwrap_or(&[]);
                let aux_s: &[FieldElement<FieldExtension>] =
                    aux_op.map(|a| a.evaluations_sym.as_slice()).unwrap_or(&[]);

                let prim_root = &domains[i].trace_primitive_root;
                let ep_p = domains[i]
                    .lde_coset_element(reverse_index(leaf * 2, domains[i].lde_length as u64));
                let ep_s = domains[i]
                    .lde_coset_element(reverse_index(leaf * 2 + 1, domains[i].lde_length as u64));

                deep_primary[i] = match Self::reconstruct_deep_composition_poly_evaluation(
                    StarkProofView::Owned(&synth_proofs[i]),
                    &ep_p,
                    prim_root,
                    &table_challenges[i],
                    &base_p,
                    aux_p,
                    &comp_op.evaluations,
                ) {
                    Some(v) => v,
                    None => return false,
                };
                deep_sym[i] = match Self::reconstruct_deep_composition_poly_evaluation(
                    StarkProofView::Owned(&synth_proofs[i]),
                    &ep_s,
                    prim_root,
                    &table_challenges[i],
                    &base_s,
                    aux_s,
                    &comp_op.evaluations_sym,
                ) {
                    Some(v) => v,
                    None => return false,
                };
            }

            // combined[h] at a codeword position selected by `bit` (0 -> primary
            // row, 1 -> symmetric row): Sum over tables at height h of alpha^i * deep_i.
            let combined_at = |h: usize, bit: usize| -> FieldElement<FieldExtension> {
                let mut acc = FieldElement::<FieldExtension>::zero();
                for i in 0..num_tables {
                    if heights[i] == h {
                        let d = if bit == 0 {
                            &deep_primary[i]
                        } else {
                            &deep_sym[i]
                        };
                        acc += &alpha_pows[i] * d;
                    }
                }
                acc
            };

            // 3) Fold-and-inject FRI (inverse of `batched_commit_phase`).
            let c_hmax = combined_at(h_max, 0);
            let c_hmax_sym = combined_at(h_max, 1);

            let ep0 = domains[tallest]
                .lde_coset_element(reverse_index(iota * 2, domains[tallest].lde_length as u64));
            let ep0_inv = match ep0.inv() {
                Ok(v) => v,
                Err(_) => return false,
            };

            // Initial fold of the (uncommitted) tallest layer with betas_fri[0].
            let mut v =
                (&c_hmax + &c_hmax_sym) + ep0_inv.clone() * &betas_fri[0] * (&c_hmax - &c_hmax_sym);
            let mut index = iota;
            let mut point_inv = ep0_inv.square();

            let fri_deco = &proof.query_list[q];
            if fri_deco.layers_auth_paths.len() != num_layers
                || fri_deco.layers_evaluations_sym.len() != num_layers
            {
                return false;
            }

            let mut fold_ok = true;
            for iter in 0..num_layers {
                let h = h_max - 1 - iter;
                // Inject the tables entering at this height (adds zero if none).
                let inj = combined_at(h, index & 1);
                v += betas_fri[iter].square() * inj;

                let eval_sym = &fri_deco.layers_evaluations_sym[iter];
                fold_ok &= Self::verify_fri_layer_openings(
                    &proof.fri_layers_merkle_roots[iter],
                    &fri_deco.layers_auth_paths[iter].merkle_path,
                    &v,
                    eval_sym,
                    index,
                );

                v = (&v + eval_sym) + point_inv.clone() * &betas_fri[iter + 1] * (&v - eval_sym);
                index >>= 1;
                point_inv = point_inv.square();
            }
            if !fold_ok || v != proof.fri_last_value {
                return false;
            }
        }

        // ===== Bus balance. =====
        if needs_lookup_challenges {
            let mut total = FieldElement::<FieldExtension>::zero();
            for (air, t) in airs.iter().zip(&proof.per_table) {
                if air.has_trace_interaction()
                    && let Some(bpi) = &t.bus_public_inputs
                {
                    total = total + &bpi.table_contribution;
                }
            }
            if total != *expected_bus_balance {
                return false;
            }
        }

        true
    }

    /// Continuation epoch verifier: verify the epoch's VM tables with the batched
    /// (unified-shard) FRI while verifying the single L2G sub-table as a SEPARATE
    /// commitment lane. Mirrors `IsStarkProver::multi_prove_batched_epoch`'s
    /// transcript order exactly:
    ///
    /// 1. Absorb the L2G main root FIRST.
    /// 2. `batched_verify_rounds_1_to_3` over the VM tables (to the round-4 seam).
    /// 3. At the seam, FORK the transcript (single lane -> no idx bytes; then absorb
    ///    the L2G aux root, then the L2G bus `table_contribution`) and verify the
    ///    L2G lane via `verify_rounds_2_to_4` with the shared LogUp challenges.
    /// 4. `batched_verify_round_4` for the VM tables on the main transcript.
    fn batched_verify_epoch(
        vm_refs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        l2g_ref: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        vm_proof: &BatchedMultiProof<Field, FieldExtension, PI>,
        l2g_proof: &StarkProof<Field, FieldExtension, PI>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
        expected_bus_balance: &FieldElement<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
        PI: Clone,
    {
        // (1) Mirror the prover: absorb the L2G main root FIRST (canonical order).
        transcript.append_bytes(&l2g_proof.lde_trace_main_merkle_root);

        // (2) VM rounds 1-3 to the round-4 seam.
        let mid = match Self::batched_verify_rounds_1_to_3(vm_refs, vm_proof, transcript) {
            Some(m) => m,
            None => return false,
        };

        // (3) L2G lane on a fork of the post-seam transcript. Single lane -> no idx
        // bytes; absorb the aux root then the bus table_contribution (matches the
        // prover's fork in `multi_prove_batched_epoch`).
        let mut l2g_fork = transcript.clone();
        if let Some(aux_root) = l2g_proof.lde_trace_aux_merkle_root {
            l2g_fork.append_bytes(&aux_root);
        }
        if let Some(bpi) = l2g_proof.bus_public_inputs.as_ref() {
            l2g_fork.append_field_element(&bpi.table_contribution);
        }
        let l2g_ok = Self::verify_rounds_2_to_4(
            l2g_ref,
            StarkProofView::Owned(l2g_proof),
            &l2g_proof.public_inputs,
            &mut l2g_fork,
            mid.lookup_challenges.clone(),
        );

        // (4) VM batched Round 4 continues on the main (un-cloned) transcript.
        //
        // Bus balance: L2G shares the in-trace Memory / range-check buses with the
        // VM tables. The monolithic check summed table_contribution over VM + L2G
        // against the COMMIT offset; batched_verify_round_4 sums only the VM lane,
        // so fold L2G's contribution into the target:
        //   Sum_VM table_contribution == expected - L2G_contribution
        // i.e. Sum_VM + L2G == expected. L2G's table_contribution is bound to its
        // committed trace by the L2G proof verified above, so this stays sound.
        let mut vm_expected = expected_bus_balance.clone();
        if l2g_ref.has_trace_interaction()
            && let Some(bpi) = l2g_proof.bus_public_inputs.as_ref()
        {
            vm_expected = &vm_expected - &bpi.table_contribution;
        }
        let vm_ok = Self::batched_verify_round_4(mid, vm_refs, vm_proof, transcript, &vm_expected);

        l2g_ok && vm_ok
    }
}
