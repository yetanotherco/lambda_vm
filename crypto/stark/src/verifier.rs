use super::{
    config::BatchedMerkleTreeBackend,
    domain::VerifierDomain,
    grinding,
    proof::stark::StarkProof,
    traits::{AIR, TransitionEvaluationContext},
};
pub use crate::proof::view::PiDeserializer;
use crate::{
    config::Commitment,
    domain::new_verifier_domain,
    lookup::{BusPublicInputs, LOGUP_CHALLENGE_ALPHA, LOGUP_NUM_CHALLENGES, compute_alpha_powers},
    proof::stark::{ArchivedStarkProof, MultiProof},
    proof::view::{
        DeepPolynomialOpeningView, FriDecommitmentView, PolynomialOpeningsView, StarkProofView,
        StarkTableView,
    },
    table::Table,
};
use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use crypto::merkle_tree::proof::verify_merkle_path_from_leaf_hash;
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

    /// Validate the two OOD trace-opening blocks' public shapes (soundness
    /// invariant I3) and reconstruct the full current+next-row OOD grid once, so
    /// steps 2 and 3 can share it instead of each rebuilding it. Returns the grid
    /// plus the per-column next-row mask, or `None` on any shape mismatch — which
    /// rejects the proof.
    ///
    /// Both block shapes are a public function of the AIR, never of the
    /// (prover-controlled) proof: the current-row block opens every column over
    /// `step_size` rows; the next-row block opens only the transition-window
    /// columns over the remaining rows (and is empty when there are none). Any
    /// mismatch (including an archive whose advertised dims disagree with its data
    /// length) is rejected before either block is used, so a malicious prover
    /// cannot reshape them to dodge a check or desync the frame reconstruction.
    fn reconstruct_ood_grid(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
    ) -> Option<(Table<FieldExtension>, Vec<bool>)> {
        let step_size = air.step_size();
        let num_eval_points = air.context().transition_offsets.len() * step_size;
        let next_row_cols = air.trace_ood_next_row_columns();
        let expected_next_width = next_row_cols.len();
        let expected_next_height = if expected_next_width == 0 {
            0
        } else {
            num_eval_points - step_size
        };
        let ood_current = proof.trace_ood_evaluations();
        let ood_next = proof.trace_ood_next_evaluations();
        if ood_current.height() != step_size
            || !ood_current.dimensions_consistent()
            || ood_next.width() != expected_next_width
            || ood_next.height() != expected_next_height
            || !ood_next.dimensions_consistent()
        {
            return None;
        }

        let ood_full = crate::ood::reconstruct_ood_full(
            ood_current.row_major_data(),
            ood_current.width(),
            ood_next.row_major_data(),
            num_eval_points,
            step_size,
            &next_row_cols,
        );
        let next_row_flags = crate::ood::next_row_col_flags(ood_full.width, &next_row_cols);
        Some((ood_full, next_row_flags))
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
        ood_full: &Table<FieldExtension>,
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

        // The full current+next-row OOD grid (surviving values placed, pruned
        // next-row entries zero — never read by any constraint) is reconstructed
        // once by the caller and shared with step 3; its shape was validated there.
        // `step_size` sizes the transition frame below.
        let step_size = air.step_size();

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

        // Fold the transition zerofier's shared denominator (zᴺ − 1) into the same
        // batch inverse as one trailing element: ~3 muls instead of a separate
        // extension inversion (~70 base muls). The boundary sum below zips against
        // the length-B num/coeff vectors, so this element only feeds
        // `inv_zerofier_denominator`, popped back off right after inversion.
        boundary_c_i_evaluations_den
            .push(-FieldElement::<Field>::one() + challenges.z.pow(trace_length));

        // A malformed proof can land `z` on a boundary step, or on the trace domain
        // (zᴺ = 1) — either makes a denominator zero.
        if FieldElement::inplace_batch_inverse(&mut boundary_c_i_evaluations_den).is_err() {
            return false;
        }

        // Trailing element is now 1/(zᴺ − 1), the transition zerofier factor shared
        // by every transition constraint.
        let inv_zerofier_denominator = match boundary_c_i_evaluations_den.pop() {
            Some(inv) => inv,
            None => return false,
        };

        let boundary_quotient_ood_evaluation: FieldElement<FieldExtension> =
            boundary_c_i_evaluations_num
                .iter()
                .zip(&boundary_c_i_evaluations_den)
                .zip(&challenges.boundary_coeffs)
                .map(|((num, den), beta)| num * den * beta)
                .fold(FieldElement::<FieldExtension>::zero(), |acc, x| acc + x);

        // A malformed archive can advertise fewer OOD columns than the AIR's
        // aux count; reject instead of underflowing. The current-row block keeps
        // the full trace width even under g·z pruning, so this still yields the
        // main width.
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

        // Frame from the reconstructed full grid: the next-row step reads only
        // its transition-window columns; the zero-filled remainder is never read.
        // `into_frame` lives on the borrowed table view, so wrap the owned grid.
        let ood_frame =
            StarkTableView::Owned(ood_full).into_frame(num_main_trace_columns, step_size);
        let transition_evaluation_context = TransitionEvaluationContext::new_verifier(
            &ood_frame,
            &challenges.rap_challenges,
            &logup_alpha_powers,
            &logup_table_offset,
        );
        let transition_ood_frame_evaluations =
            air.compute_transition(&transition_evaluation_context);

        // Every transition constraint's zerofier at z is 1/(zᴺ − 1) times an
        // end-exemptions correction ∏(z − rᵢ) that depends only on the constraint's
        // `end_exemptions`. 1/(zᴺ − 1) is shared by all of them (computed once,
        // folded into the boundary batch inverse above), so we group the constraints
        // by `end_exemptions` (a tiny set — almost always just {0}), accumulate
        // Σ βᵢ·evalᵢ per group, then factor the shared inverse and each group's
        // correction out of the sum.

        // Σ βᵢ·evalᵢ bucketed by end_exemptions. `constraints_meta` has one entry per
        // active transition constraint; a constraint absent from it contributed a
        // zero denominator before and is likewise skipped here.
        let mut grouped_numerator_sums: Vec<(usize, FieldElement<FieldExtension>)> = Vec::new();
        for m in air.constraints_meta() {
            let term = &challenges.transition_coeffs[m.constraint_idx]
                * &transition_ood_frame_evaluations[m.constraint_idx];
            match grouped_numerator_sums
                .iter_mut()
                .find(|(exemptions, _)| *exemptions == m.end_exemptions)
            {
                Some((_, acc)) => *acc += term,
                None => grouped_numerator_sums.push((m.end_exemptions, term)),
            }
        }

        let transition_c_i_evaluations_sum = grouped_numerator_sums
            .into_iter()
            .fold(
                FieldElement::zero(),
                |acc, (end_exemptions, numerator_sum)| {
                    let correction = crate::constraints::zerofier::end_exemptions_correction(
                        end_exemptions,
                        &challenges.z,
                        &domain.trace_primitive_root,
                        trace_length,
                    );
                    acc + correction * numerator_sum
                },
            )
            * inv_zerofier_denominator;

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
        ood_full: &Table<FieldExtension>,
        next_row_flags: &[bool],
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        crate::profile_markers::step_marker::<{ crate::profile_markers::STEP_VERIFY_FRI }>();
        // g·z pruning: the full OOD grid (reconstructed once by the caller) lets
        // the DEEP reconstruction skip pruned next-row openings; `step_size`
        // separates the current- and next-row blocks below.
        let step_size = air.step_size();

        // The Q primary FRI evaluation points, computed once (one `pow` each) and
        // shared between the DEEP reconstruction and the FRI eval-point inverses
        // below — previously each recomputed them, so the primary point was `pow`ed
        // twice per query and the symmetric point a third time.
        let query_points: Vec<FieldElement<Field>> = challenges
            .iotas
            .iter()
            .map(|iota| Self::query_challenge_to_evaluation_point(*iota, false, domain))
            .collect();

        let (deep_poly_evaluations, deep_poly_evaluations_sym) =
            match Self::reconstruct_deep_composition_poly_evaluations_for_all_queries(
                challenges,
                domain,
                proof,
                ood_full,
                next_row_flags,
                step_size,
                &query_points,
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

        // verify FRI. Reuse the primary evaluation points already computed for the
        // DEEP reconstruction (same `query_challenge_to_evaluation_point(iota, false)`)
        // rather than `pow`ing them a second time.
        let mut evaluation_point_inverse = query_points;
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
        // Two-slice leaf hash: the committed leaf is `evaluations ‖ evaluations_sym`,
        // hashed without allocating the concatenation (see `hash_data_from_slices`).
        let leaf_hash = BatchedMerkleTreeBackend::<E>::hash_data_from_slices(
            opening.evaluations(),
            opening.evaluations_sym(),
        );
        verify_merkle_path_from_leaf_hash::<BatchedMerkleTreeBackend<E>>(
            opening.merkle_path(),
            root,
            iota,
            leaf_hash,
        )
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
        // Two-slice leaf hash of `evaluations ‖ evaluations_sym`, no concat alloc.
        let leaf_hash = BatchedMerkleTreeBackend::<FieldExtension>::hash_data_from_slices(
            composition_poly.evaluations(),
            composition_poly.evaluations_sym(),
        );

        verify_merkle_path_from_leaf_hash::<BatchedMerkleTreeBackend<FieldExtension>>(
            composition_poly.merkle_path(),
            composition_poly_merkle_root,
            *iota,
            leaf_hash,
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
        // The committed leaf is the ordered pair `(a, b)` of this layer's two
        // folded evaluations. Hash it from the two element slices directly — no
        // `vec![a, b]` allocation (see `verify_opening_pair` / `hash_data_from_slices`).
        let (a, b) = if iota % 2 == 1 {
            (evaluation_sym, evaluation)
        } else {
            (evaluation, evaluation_sym)
        };
        let leaf_hash = BatchedMerkleTreeBackend::<FieldExtension>::hash_data_from_slices(
            core::slice::from_ref(a),
            core::slice::from_ref(b),
        );
        verify_merkle_path_from_leaf_hash::<BatchedMerkleTreeBackend<FieldExtension>>(
            auth_path_sym,
            merkle_root,
            iota >> 1,
            leaf_hash,
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

        // The per-layer inverse points 𝜐^(-2ⁱ) are produced lazily and zipped into
        // the fold below — no intermediate Vec. Take the first (𝜐⁻²) here, before
        // `evaluation_point_inv` is consumed reconstructing p₁.
        let first_layer_point_inv = evaluation_point_inv.square();

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
            .zip(core::iter::successors(
                Some(first_layer_point_inv),
                |evaluation_point| Some(evaluation_point.square()),
            ))
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

    #[allow(clippy::too_many_arguments)]
    fn reconstruct_deep_composition_poly_evaluations_for_all_queries(
        challenges: &Challenges<FieldExtension>,
        domain: &VerifierDomain<Field>,
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
        ood_full: &Table<FieldExtension>,
        next_row_flags: &[bool],
        step_size: usize,
        // The Q primary FRI evaluation points (`query_challenge_to_evaluation_point`
        // with `sym = false`), computed once by the caller and shared with the FRI
        // eval-point inverses. The symmetric point of each is its negation.
        query_points: &[FieldElement<Field>],
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

        // z·gᵏ for the OOD rows and z^parts are query-independent, so build them
        // once. g is already stored on the domain (`trace_primitive_root`); recomputing
        // it via `get_primitive_root_of_unity` would repeat ~log2(N) squarings per table.
        let primitive_root = &domain.trace_primitive_root;
        let ood_height = ood_full.height;
        let mut z_row_points = Vec::with_capacity(ood_height);
        let mut current_z = challenges.z.clone();
        for _ in 0..ood_height {
            z_row_points.push(current_z.clone());
            current_z = primitive_root * &current_z;
        }
        let number_of_parts = proof.composition_poly_parts_ood_evaluation().len();
        let z_pow_parts = challenges.z.pow(number_of_parts);

        // Collect every query point's denominators (primary + symmetric) for a SINGLE
        // batch inverse over the whole proof (was two tiny per-query inversions). The
        // symmetric point is the negation of the primary (`sym_index = primary_index +
        // N/2` and `lde_root^(N/2) = −1`), so it needs no extra `pow`. Layout per query
        // point: `ood_height` trace denominators then 1 composition denominator; points
        // ordered [q0 primary, q0 sym, q1 primary, q1 sym, …].
        let stride = ood_height + 1;
        let mut denominators = Vec::with_capacity(2 * num_queries * stride);
        for primary in query_points {
            let sym = -primary;
            for evaluation_point in [primary, &sym] {
                for z_row_point in &z_row_points {
                    denominators.push(evaluation_point - z_row_point);
                }
                denominators.push(evaluation_point - &z_pow_parts);
            }
        }
        // One inversion for the whole proof's DEEP denominators. Fails closed if any
        // evaluation point lands on an OOD point (malformed proof) — same rejection
        // the previous per-query inversions gave.
        FieldElement::inplace_batch_inverse(&mut denominators).ok()?;

        let mut deep_poly_evaluations = Vec::with_capacity(num_queries);
        let mut deep_poly_evaluations_sym = Vec::with_capacity(num_queries);

        for (i, _iota) in challenges.iotas.iter().enumerate() {
            let opening = proof.deep_poly_opening(i);
            let primary_base = 2 * i * stride;
            let sym_base = primary_base + stride;

            // Base-field portion as two borrowed slices in commit order —
            // precomputed columns FIRST, then main trace columns. The callee
            // resolves a base column via `base_at`, so there is no per-query
            // concat allocation.
            let lde_precomputed: &[FieldElement<Field>] = opening
                .precomputed_trace_polys()
                .map(|p| p.evaluations())
                .unwrap_or(&[]);
            let lde_main = opening.main_trace_polys().evaluations();

            let lde_aux: &[FieldElement<FieldExtension>] = opening
                .aux_trace_polys()
                .map(|a| a.evaluations())
                .unwrap_or(&[]);

            deep_poly_evaluations.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                challenges,
                lde_precomputed,
                lde_main,
                lde_aux,
                opening.composition_poly().evaluations(),
                ood_full,
                next_row_flags,
                step_size,
                &denominators[primary_base..primary_base + ood_height],
                &denominators[primary_base + ood_height],
            )?);

            // Mirror for the symmetric query point.
            let lde_precomputed_sym: &[FieldElement<Field>] = opening
                .precomputed_trace_polys()
                .map(|p| p.evaluations_sym())
                .unwrap_or(&[]);
            let lde_main_sym = opening.main_trace_polys().evaluations_sym();

            let lde_aux_sym: &[FieldElement<FieldExtension>] = opening
                .aux_trace_polys()
                .map(|a| a.evaluations_sym())
                .unwrap_or(&[]);

            deep_poly_evaluations_sym.push(Self::reconstruct_deep_composition_poly_evaluation(
                proof,
                challenges,
                lde_precomputed_sym,
                lde_main_sym,
                lde_aux_sym,
                opening.composition_poly().evaluations_sym(),
                ood_full,
                next_row_flags,
                step_size,
                &denominators[sym_base..sym_base + ood_height],
                &denominators[sym_base + ood_height],
            )?);
        }
        Some((deep_poly_evaluations, deep_poly_evaluations_sym))
    }

    #[allow(clippy::too_many_arguments)]
    fn reconstruct_deep_composition_poly_evaluation<'b>(
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
        challenges: &Challenges<FieldExtension>,
        lde_trace_precomputed_evaluations: &'b [FieldElement<Field>],
        lde_trace_main_evaluations: &'b [FieldElement<Field>],
        lde_trace_aux_evaluations: &[FieldElement<FieldExtension>],
        lde_composition_poly_parts_evaluation: &[FieldElement<FieldExtension>],
        ood_full: &Table<FieldExtension>,
        next_row_flags: &[bool],
        step_size: usize,
        // Pre-inverted 1/(evaluation_point − z·gᵏ) for each OOD row, and
        // 1/(evaluation_point − z^parts) for the composition part. The caller
        // computes these for every query point in one shared batch inverse.
        inv_trace_denominators: &[FieldElement<FieldExtension>],
        inv_composition_denominator: &FieldElement<FieldExtension>,
    ) -> Option<FieldElement<FieldExtension>> {
        // g·z pruning: read from the reconstructed full grid (current-row block
        // plus the scattered next-row window, zeros elsewhere), not from
        // `proof.trace_ood_evaluations()` which now carries only the current-row
        // block. Flatten once for the hot loop below.
        let ood_evaluations_table_height = ood_full.height;
        let ood_evaluations_table_width = ood_full.width;
        let ood_data = ood_full.row_major_data();
        let trace_term_coeffs = &challenges.trace_term_coeffs;

        // Base columns are supplied as two slices (precomputed ‖ main) that the
        // prover concatenated in this order; `num_base` is their combined width
        // and `base_at` indexes into them as if concatenated, without allocating.
        let num_precomputed = lde_trace_precomputed_evaluations.len();
        let num_base = num_precomputed + lde_trace_main_evaluations.len();
        let base_at = move |col: usize| -> &'b FieldElement<Field> {
            if col < num_precomputed {
                &lde_trace_precomputed_evaluations[col]
            } else {
                &lde_trace_main_evaluations[col - num_precomputed]
            }
        };

        // Runtime guard: a malformed proof may supply opening evaluations whose
        // column count does not match the OOD table width, or whose composition
        // poly parts count does not match the proof's `composition_poly_parts_ood_evaluation`.
        // Without these checks the indexing below would panic in release builds.
        if num_base + lde_trace_aux_evaluations.len() != ood_evaluations_table_width {
            return None;
        }
        if trace_term_coeffs.is_empty()
            || trace_term_coeffs.len() * trace_term_coeffs[0].len()
                != ood_evaluations_table_height * ood_evaluations_table_width
        {
            return None;
        }
        // One pre-inverted trace denominator per OOD row (built by the caller).
        if inv_trace_denominators.len() != ood_evaluations_table_height {
            return None;
        }

        let trace_term = (0..ood_evaluations_table_width)
            .zip(&challenges.trace_term_coeffs)
            .fold(FieldElement::zero(), |trace_terms, (col_idx, coeff_row)| {
                let opened_next_row = next_row_flags[col_idx];
                let trace_i = (0..ood_evaluations_table_height).zip(coeff_row).fold(
                    FieldElement::zero(),
                    |trace_t, (row_idx, coeff)| {
                        // g·z pruning: the next-row block opens only transition-
                        // window columns. Skip every other next-row opening — its
                        // coefficient is zero, so the term is vacuous, and skipping
                        // it is where the verifier/recursion cycle saving lands.
                        if row_idx >= step_size && !opened_next_row {
                            return trace_t;
                        }
                        let ood_val = &ood_data[row_idx * ood_evaluations_table_width + col_idx];
                        // Stay in base when we can: F: IsSubFieldOf<E> gives F - E -> E.
                        let diff: FieldElement<FieldExtension> = if col_idx < num_base {
                            base_at(col_idx) - ood_val
                        } else {
                            &lde_trace_aux_evaluations[col_idx - num_base] - ood_val
                        };
                        let poly_evaluation = diff * &inv_trace_denominators[row_idx];
                        trace_t + &poly_evaluation * coeff
                    },
                );
                trace_terms + trace_i
            });

        let composition_parts_ood = proof.composition_poly_parts_ood_evaluation();
        let mut h_terms = FieldElement::zero();
        for (j, h_i_upsilon) in lde_composition_poly_parts_evaluation.iter().enumerate() {
            // Bounds-check via `.get(j)?`: a malformed opening may have more
            // parts than the proof header advertises.
            let h_i_zpower = composition_parts_ood.get(j)?;
            let gamma = challenges.gammas.get(j)?;
            let h_i_term = (h_i_upsilon - h_i_zpower) * gamma;
            h_terms += h_i_term;
        }
        h_terms *= inv_composition_denominator;

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
            // has no rows, whose width doesn't match the AIR's column layout, or
            // whose height isn't a whole number of AIR steps (which `into_frame`
            // below only `debug_assert!`s, not checks) — all before any row
            // access indexes into it.
            //
            // The width check is load-bearing and prevents two distinct faults:
            // (a) the AIR-derived column index `main_trace_width + c.col` in
            //     `step_2_verify_claimed_composition_polynomial` indexing past a
            //     too-narrow OOD row (a release-mode out-of-bounds panic), and
            // (b) a width-0 table, whose `width * height == 0 == data.len()`
            //     satisfies `dimensions_consistent()` for an arbitrary advertised
            //     height and would otherwise slip through this guard entirely.
            // An honest proof always commits exactly `main_trace_width + num_aux`
            // OOD columns (the same quantities `column_idx` and the `checked_sub`
            // boundary use), so exact equality never rejects a valid proof.
            let trace_ood_evaluations = proof.trace_ood_evaluations();
            let expected_ood_width = air.trace_layout().0 + air.num_auxiliary_rap_columns();
            if !trace_ood_evaluations.dimensions_consistent()
                || trace_ood_evaluations.height() == 0
                || trace_ood_evaluations.width() != expected_ood_width
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

        // <<<< Receive values: tⱼ(zgᵏ). Absorb the two pruned OOD blocks in the
        // same order the prover sent them (current-row block, then next-row
        // block), each column-major (matching `Table::columns()` order) reading
        // rows in place, without materializing transposed columns.
        for ood in [
            proof.trace_ood_evaluations(),
            proof.trace_ood_next_evaluations(),
        ] {
            for col_idx in 0..ood.width() {
                for row_idx in 0..ood.height() {
                    transcript.append_field_element(&ood.get_row(row_idx)[col_idx]);
                }
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
        let num_eval_points = air.context().transition_offsets.len() * air.step_size();
        let next_row_cols = air.trace_ood_next_row_columns();
        // Must match the prover's g·z pruning exactly (same AIR metadata): the
        // current-row block opens every column, the next-row block only the
        // transition-window columns.
        let num_terms_trace = crate::ood::num_surviving_trace_openings(
            air.context().trace_columns,
            num_eval_points,
            air.step_size(),
            next_row_cols.len(),
        );
        let gamma = transcript.sample_field_element();

        // <<<< Receive challenges: 𝛾, 𝛾'
        let mut deep_composition_coefficients: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
                .take(num_terms_composition_poly + num_terms_trace)
                .collect();

        let trace_term_powers: Vec<_> = deep_composition_coefficients
            .drain(..num_terms_trace)
            .collect();
        let trace_term_coeffs = crate::ood::build_pruned_trace_term_coeffs(
            &trace_term_powers,
            air.context().trace_columns,
            num_eval_points,
            air.step_size(),
            &next_row_cols,
        );

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

        // Reconstruct the shared current+next-row OOD grid once (with soundness-I3
        // shape validation) and reuse it across steps 2 and 3 rather than rebuilding
        // it in each.
        let (ood_full, next_row_flags) = match Self::reconstruct_ood_grid(air, proof) {
            Some(pair) => pair,
            None => {
                #[cfg(not(feature = "test_fiat_shamir"))]
                error!("OOD trace opening shape mismatch");
                return false;
            }
        };

        if !Self::step_2_verify_claimed_composition_polynomial(
            air,
            proof,
            public_inputs,
            &domain,
            &challenges,
            &ood_full,
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

        if !Self::step_3_verify_fri(air, proof, &domain, &challenges, &ood_full, &next_row_flags) {
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
