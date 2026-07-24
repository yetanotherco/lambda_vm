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
    proof::stark::{ArchivedMultiProof, BatchedMultiProof, MultiProof},
    proof::view::{
        BatchedMultiProofView, BatchedTableDataView, DeepPolynomialOpeningView,
        FriDecommitmentView, MultiProofView, PolynomialOpeningsView, ProofViewSource,
        StarkProofView, StarkTableView,
    },
    table::Table,
};
use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use crypto::field_ext::Fp3Fma;
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
    FieldExtension: Send + Sync + IsField + Fp3Fma,
    PI,
> {
    phantom: PhantomData<(Field, FieldExtension, PI)>,
}

impl<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: IsField + Send + Sync + Fp3Fma,
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

/// Deep-composition sums that are identical across all FRI queries of a
/// single proof (see `compute_query_invariant_deep_terms`).
pub struct QueryInvariantDeepTerms<FieldExtension>
where
    FieldExtension: Send + Sync + IsField,
{
    /// `ood_row_sum[row] = sum_col trace_term_coeffs[col][row] * ood(row, col)`,
    /// over the reconstructed full OOD grid (g·z-pruned positions are zero).
    ood_row_sum: Vec<FieldElement<FieldExtension>>,
    /// The DEEP denominator base points `g^k·z` for `k in 0..ood_height`
    /// (`gz_powers[0] = z`, `gz_powers[k] = primitive_root^k · z`). These are
    /// query-invariant — the walk depends only on `z`, the domain generator, and
    /// the OOD height — so it is computed once per table here instead of being
    /// re-walked (via base×ext products / FEXT_BASE_MUL on the guest) inside every
    /// query's `reconstruct_deep_composition_poly_evaluation_pair`. Each query then
    /// forms its per-point denominators as `evaluation_point − gz_powers[k]`.
    gz_powers: Vec<FieldElement<FieldExtension>>,
    /// Width of the reconstructed full OOD grid (= full trace width).
    ood_width: usize,
    /// Derived from `proof.composition_poly_parts_ood_evaluation().len()`.
    number_of_parts: usize,
    /// `challenges.z.pow(number_of_parts)`.
    z_pow: FieldElement<FieldExtension>,
    /// `sum_j composition_poly_parts_ood_evaluation[j] * challenges.gammas[j]`.
    h_sum_zpow: FieldElement<FieldExtension>,
}

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
    FieldExtension: Send + Sync + IsField + Fp3Fma,
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

    /// The pruned-OOD layout for this AIR — the single place in the verifier that
    /// reads the shape metadata (`trace_columns`, `step_size`, the
    /// transition-offset count, and the next-row column set). Everything that used
    /// to recompute these values now derives them from the returned
    /// [`crate::ood::OodLayout`]. Pure AIR metadata, never a proof dimension.
    fn ood_layout(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
    ) -> crate::ood::OodLayout {
        crate::ood::OodLayout::new(
            air.context().trace_columns,
            air.context().transition_offsets.len() * air.step_size(),
            air.step_size(),
            air.trace_ood_next_row_columns(),
        )
    }

    /// Checks whether the purported evaluations of the composition polynomial parts and the trace
    /// polynomials at the out-of-domain challenge are consistent.
    /// See https://lambdaclass.github.io/lambdaworks/starks/protocol.html#step-2-verify-claimed-composition-polynomial
    /// Soundness (I3): both OOD blocks' shapes are a public function of the AIR,
    /// never of the (prover-controlled) proof. The current-row block opens every
    /// column over `step_size` rows; the next-row block opens only the
    /// transition-window columns over the remaining rows, and is empty when the
    /// AIR reads none.
    ///
    /// Must run before Round 3, which absorbs the next-row block through
    /// `get_row` — an unchecked `data[start..start + width]` slice. A hostile
    /// archive whose advertised dims disagree with its data length would panic
    /// there rather than be rejected as a false proof; `dimensions_consistent()`
    /// closes that gap, which rkyv's bytecheck leaves open.
    fn ood_blocks_well_formed(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
    ) -> bool {
        let step_size = air.step_size();
        let num_eval_points = air.context().transition_offsets.len() * step_size;
        let expected_next_width = air.trace_ood_next_row_columns().len();
        let expected_next_height = if expected_next_width == 0 {
            0
        } else {
            num_eval_points.saturating_sub(step_size)
        };
        let current = proof.trace_ood_evaluations();
        let next = proof.trace_ood_next_evaluations();

        // `height == step_size` also rejects a height-0 current block: every AIR
        // reports `step_size >= 1`.
        current.dimensions_consistent()
            && current.width() == air.trace_layout().0 + air.num_auxiliary_rap_columns()
            && current.height() == step_size
            && next.dimensions_consistent()
            && next.width() == expected_next_width
            && next.height() == expected_next_height
    }

    fn step_2_verify_claimed_composition_polynomial(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
        public_inputs: &PI,
        domain: &VerifierDomain<Field>,
        challenges: &Challenges<FieldExtension>,
        // The full current+next-row OOD grid, shape-checked and reconstructed once
        // by the caller (after `ood_blocks_well_formed`) and shared with
        // `step_3_verify_fri`. Its pruned next-row entries are zero — those are
        // never read by any constraint. `step_size` accompanies it for the frame
        // split below.
        ood_full: &Table<FieldExtension>,
        step_size: usize,
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

        // Once-per-proof OOD boundary fold `acc += num*den*beta`: the 3-operand
        // resident accumulate routes the two ext muls per term through the chip.
        let boundary_quotient_ood_evaluation: FieldElement<FieldExtension> = {
            let mut acc = FieldExtension::prod_acc_new();
            for ((num, den), beta) in boundary_c_i_evaluations_num
                .iter()
                .zip(&boundary_c_i_evaluations_den)
                .zip(&challenges.boundary_coeffs)
            {
                FieldExtension::prod_acc_add(&mut acc, num, den, beta);
            }
            FieldExtension::prod_acc_finish(acc)
        };

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
                // Resident-handle power sequence: keeps α loaded in the FEXT chip
                // across the whole [1, α, α², …] walk instead of the `*` operator
                // reloading it per element. Byte-identical to `compute_alpha_powers`.
                FieldExtension::geometric_powers(
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

        // The zerofier denominator `1/(zᴺ − 1)` and the end-exemption decrement
        // `g^(N−1)` are identical for every transition constraint of this table
        // (only `z` and `trace_length` feed them, both fixed here), yet the
        // per-constraint `evaluate_zerofier` recomputes that field power +
        // inverse on every call. Hoist them out of the loop and evaluate only
        // each constraint's end-exemptions product inside.
        let z_pow_n = challenges.z.pow(trace_length);
        let zerofier_denominator_inv = match (-FieldElement::<Field>::one() + z_pow_n).inv() {
            Ok(inv) => inv,
            // zᴺ == 1 ⇒ z lies on the trace domain (malformed proof/challenge).
            Err(_) => return false,
        };
        let end_exemption_decrement = domain.trace_primitive_root.pow(trace_length - 1);
        let mut denominators =
            vec![FieldElement::<FieldExtension>::zero(); air.num_transition_constraints()];
        air.constraints_meta().iter().for_each(|m| {
            denominators[m.constraint_idx] = crate::constraints::zerofier::evaluate_zerofier_with(
                m,
                &challenges.z,
                &end_exemption_decrement,
                &zerofier_denominator_inv,
            );
        });

        // Once-per-proof OOD transition fold `acc += beta*eval*denominator`.
        let transition_c_i_evaluations_sum = {
            let mut acc = FieldExtension::prod_acc_new();
            for (eval, beta, denominator) in itertools::izip!(
                transition_ood_frame_evaluations,
                &challenges.transition_coeffs,
                denominators
            ) {
                FieldExtension::prod_acc_add(&mut acc, beta, &eval, &denominator);
            }
            FieldExtension::prod_acc_finish(acc)
        };

        let composition_poly_ood_evaluation =
            &boundary_quotient_ood_evaluation + transition_c_i_evaluations_sum;

        // Once-per-proof Horner `acc = acc*z + coeff`: one chip FMA per part.
        let composition_poly_claimed_ood_evaluation = proof
            .composition_poly_parts_ood_evaluation()
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, coeff| {
                FieldExtension::fma(&acc, &challenges.z, coeff)
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
        // g·z pruning: the full OOD grid (reconstructed once by the caller and
        // shared with `step_2`) plus the transition-window column indices, so the
        // DEEP reconstruction can skip pruned next-row openings.
        ood_full: &Table<FieldExtension>,
        next_row_cols: &[usize],
        step_size: usize,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        crate::profile_markers::step_marker::<{ crate::profile_markers::STEP_VERIFY_FRI }>();
        // The primary FRI query points 𝜐ᵢ are needed twice — as the DEEP
        // denominator arguments (reconstruct, below) and, inverted, as the fold
        // start points (`evaluation_point_inverse`, further down). Compute the
        // `pow`-derived points ONCE here, lend them to the reconstruction, then
        // move the same Vec into the batch inverse — no second per-query `pow`.
        let evaluation_points: Vec<FieldElement<Field>> = challenges
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
                next_row_cols,
                step_size,
                &evaluation_points,
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

        // verify FRI. Reuse the primary points computed above (reconstruct only
        // borrowed them) and invert them in place for the fold start points.
        let mut evaluation_point_inverse = evaluation_points;
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
        // The committed leaf is the ordered pair (evaluation, evaluation_sym) for
        // an even index, (evaluation_sym, evaluation) for an odd one. Hash it
        // straight from the two borrowed elements as single-element slices: the
        // two-slice leaf hash streams `first ‖ second` byte-identically to
        // `hash_data(&vec![first, second])`, but without the per-layer heap Vec
        // and the two element clones that form required (a hot per-query×layer
        // allocation in the FRI fold).
        let (first, second) = if iota % 2 == 1 {
            (evaluation_sym, evaluation)
        } else {
            (evaluation, evaluation_sym)
        };
        let leaf_hash = BatchedMerkleTreeBackend::<FieldExtension>::hash_data_from_slices(
            core::slice::from_ref(first),
            core::slice::from_ref(second),
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

        // The per-layer squared evaluation points 𝜐^(-2ⁱ) are consumed exactly once
        // by the fold below, so keep them as a lazy iterator instead of collecting
        // into a Vec — the squares are computed on demand during the fold, saving a
        // per-(query×table) heap allocation, N stores/loads, and the drop. The
        // yielded owned values and their order are identical to the collected Vec,
        // so this is byte-identical (implementation-only).
        let evaluation_point_iter =
            core::iter::successors(Some(evaluation_point_inv.square()), |evaluation_point| {
                Some(evaluation_point.square())
            })
            .take(fri_layers_merkle_roots.len());

        // Reconstruct p₁(𝜐²). FRI commit-phase butterfly:
        // v = (p₀+p₀_sym) + 𝜐⁻¹·ζ·(p₀−p₀_sym). `c0 = 𝜐⁻¹·ζ` is a base×ext product
        // (FEXT_BASE_MUL on the guest, software elsewhere); the ext×ext product +
        // add is one chip FMA. CAVEAT: this site is eliminated if fold-by-4 lands.
        let c0 = FieldExtension::base_mul(&evaluation_point_inv, &zetas[0]);
        let mut v = FieldExtension::fma(&c0, &(p0_eval - p0_eval_sym), &(p0_eval + p0_eval_sym));
        let mut index = iota;

        // Fold through every committed layer: use the proof to verify the openings
        // of pᵢ(−𝜐^(2ⁱ)) (given by the prover) and pᵢ(𝜐^(2ⁱ)) (computed on the
        // previous iteration), then obtain pᵢ₊₁(𝜐^(2ⁱ⁺¹)). When there are no
        // committed layers (`total_folds == 1`, a single final fold) this fold is
        // empty and `v`/`index` already hold the terminal-layer value/position.
        let openings_ok = fri_layers_merkle_roots
            .iter()
            .zip(fri_decommitment.layers_evaluations_sym())
            .zip(evaluation_point_iter)
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

                    // Update `v` with next value pᵢ₊₁(𝜐^(2ⁱ⁺¹)). Same butterfly,
                    // one chip FMA (c = 𝜐^(-2ⁱ)·ζ is base×ext -> FEXT_BASE_MUL).
                    let c = FieldExtension::base_mul(&evaluation_point_inv, &zetas[i + 1]);
                    v = FieldExtension::fma(&c, &(&v - evaluation_sym), &(&v + evaluation_sym));

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

    /// Sums that depend only on `challenges` and proof-level OOD/gamma data —
    /// identical for every FRI query — computed once instead of once per
    /// query.
    ///
    /// g·z pruning: the trace OOD values come from the reconstructed full grid
    /// `ood_full` (current-row block plus the scattered next-row window, zeros
    /// elsewhere), not from `proof.trace_ood_evaluations()` which now carries
    /// only the current-row block. Pruned positions are zero in both the grid
    /// and `trace_term_coeffs`, so next rows sum only the window columns.
    fn compute_query_invariant_deep_terms(
        challenges: &Challenges<FieldExtension>,
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
        ood_full: &Table<FieldExtension>,
        next_row_cols: &[usize],
        step_size: usize,
        primitive_root: &FieldElement<Field>,
    ) -> Option<QueryInvariantDeepTerms<FieldExtension>> {
        let ood_evaluations_table_height = ood_full.height;
        let ood_evaluations_table_width = ood_full.width;
        let ood_data = ood_full.row_major_data();
        let trace_term_coeffs = &challenges.trace_term_coeffs;

        if trace_term_coeffs.is_empty()
            || trace_term_coeffs.len() * trace_term_coeffs[0].len()
                != ood_evaluations_table_height * ood_evaluations_table_width
        {
            return None;
        }

        let mut ood_row_sum = Vec::with_capacity(ood_evaluations_table_height);
        for row_idx in 0..ood_evaluations_table_height {
            let ood_row = &ood_data[row_idx * ood_evaluations_table_width
                ..(row_idx + 1) * ood_evaluations_table_width];
            let mut sum = FieldElement::<FieldExtension>::zero();
            if row_idx < step_size {
                for col_idx in 0..ood_evaluations_table_width {
                    sum += &trace_term_coeffs[col_idx][row_idx] * &ood_row[col_idx];
                }
            } else {
                // Next-row row: off-window columns contribute coeff·0 with a
                // zero coeff too, so the window-only sum is exact.
                for &col_idx in next_row_cols {
                    sum += &trace_term_coeffs[col_idx][row_idx] * &ood_row[col_idx];
                }
            }
            ood_row_sum.push(sum);
        }

        let composition_parts_ood = proof.composition_poly_parts_ood_evaluation();
        let number_of_parts = composition_parts_ood.len();
        let z_pow = challenges.z.pow(number_of_parts);

        // A malformed proof/challenge set can advertise more composition
        // parts than sampled gammas; reject rather than silently truncate
        // the sum below.
        if challenges.gammas.len() < number_of_parts {
            return None;
        }
        let mut h_sum_zpow = FieldElement::<FieldExtension>::zero();
        for (h_i_zpower, gamma) in composition_parts_ood.iter().zip(challenges.gammas.iter()) {
            h_sum_zpow += h_i_zpower * gamma;
        }

        // The query-invariant DEEP denominator base points g^k·z (walked once here
        // instead of once per query). `gz_powers[0] = z`, and each step multiplies
        // by the domain generator (base×ext -> FEXT_BASE_MUL on the guest).
        let mut gz_powers = Vec::with_capacity(ood_evaluations_table_height);
        let mut current_z = challenges.z.clone();
        for _ in 0..ood_evaluations_table_height {
            gz_powers.push(current_z.clone());
            current_z = FieldExtension::base_mul(primitive_root, &current_z);
        }

        Some(QueryInvariantDeepTerms {
            ood_row_sum,
            gz_powers,
            ood_width: ood_evaluations_table_width,
            number_of_parts,
            z_pow,
            h_sum_zpow,
        })
    }

    fn reconstruct_deep_composition_poly_evaluations_for_all_queries(
        challenges: &Challenges<FieldExtension>,
        domain: &VerifierDomain<Field>,
        proof: StarkProofView<'_, Field, FieldExtension, PI>,
        ood_full: &Table<FieldExtension>,
        next_row_cols: &[usize],
        step_size: usize,
        // The primary FRI query points 𝜐ᵢ, computed once by the caller (which also
        // batch-inverts them for the fold), so they are not re-derived (a full
        // `pow` each) here. `evaluation_points.len() == challenges.iotas.len()`.
        evaluation_points: &[FieldElement<Field>],
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

        let query_invariant_terms = Self::compute_query_invariant_deep_terms(
            challenges,
            proof,
            ood_full,
            next_row_cols,
            step_size,
            primitive_root,
        )?;

        // ROUND-2 increment C (MEASUREMENT-ONLY, never proven): the in-place
        // reduced-opening ABI. `trace_term_coeffs` is proof-constant, so build its
        // per-column pointer table ONCE here (not once per query as Level A does)
        // and register it — plus the transition window / OOD dims / constant
        // column counts — with the executor. Each per-row ecall below then passes
        // only the six per-query eval-slice base pointers. `sim_col_ptrs` must
        // outlive the whole query loop (the executor reads through its addresses).
        #[cfg(all(target_arch = "riscv64", feature = "sim-ro-inplace"))]
        let sim_col_ptrs: Vec<u64> = challenges
            .trace_term_coeffs
            .iter()
            .map(|c| c.as_ptr() as u64)
            .collect();

        for (i, evaluation_point) in evaluation_points.iter().enumerate() {
            let opening = proof.deep_poly_opening(i);

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

            let lde_precomputed_sym: &[FieldElement<Field>] = opening
                .precomputed_trace_polys()
                .map(|p| p.evaluations_sym())
                .unwrap_or(&[]);
            let lde_main_sym = opening.main_trace_polys().evaluations_sym();

            let lde_aux_sym: &[FieldElement<FieldExtension>] = opening
                .aux_trace_polys()
                .map(|a| a.evaluations_sym())
                .unwrap_or(&[]);

            // ROUND-2 increment C: register the proof-constant reduced-opening
            // layout once (on the first query — the column counts are identical
            // for every query). Thereafter the per-row ecalls read it from the
            // executor's cache.
            #[cfg(all(target_arch = "riscv64", feature = "sim-ro-inplace"))]
            if i == 0 {
                let layout = math::sim_ro::ReducedOpeningLayout {
                    coeff_col_ptrs_ptr: sim_col_ptrs.as_ptr() as u64,
                    next_row_cols_ptr: next_row_cols.as_ptr() as u64,
                    next_row_cols_len: next_row_cols.len() as u64,
                    ood_width: query_invariant_terms.ood_width as u64,
                    step_size: step_size as u64,
                    precomputed_len: lde_precomputed.len() as u64,
                    main_len: lde_main.len() as u64,
                    aux_len: lde_aux.len() as u64,
                    precomputed_sym_len: lde_precomputed_sym.len() as u64,
                    main_sym_len: lde_main_sym.len() as u64,
                    aux_sym_len: lde_aux_sym.len() as u64,
                };
                lambda_vm_syscalls::syscalls::register_ro_layout(&layout as *const _ as usize);
            }

            // The symmetric FRI query point is exactly -𝜐. The pair sits at
            // FRI-order positions 2·iota and 2·iota+1, whose bit-reversed LDE
            // indices differ by exactly lde_length/2 (`reverse_index(2i+1) =
            // reverse_index(2i) + N/2`), and `lde_primitive_root^(N/2) = -1`, so
            // `lde_coset_element(rev(2i+1)) = -lde_coset_element(rev(2i))`. Negate
            // the primary point (supplied by the caller) instead of a second full
            // `pow` (the same 𝜐 / -𝜐 pairing the FRI butterfly below relies on).
            let evaluation_point_sym = -evaluation_point;
            let (evaluation, evaluation_sym) =
                Self::reconstruct_deep_composition_poly_evaluation_pair(
                    evaluation_point,
                    &evaluation_point_sym,
                    primitive_root,
                    challenges,
                    &query_invariant_terms,
                    next_row_cols,
                    step_size,
                    lde_precomputed,
                    lde_main,
                    lde_aux,
                    opening.composition_poly().evaluations(),
                    lde_precomputed_sym,
                    lde_main_sym,
                    lde_aux_sym,
                    opening.composition_poly().evaluations_sym(),
                )?;
            deep_poly_evaluations.push(evaluation);
            deep_poly_evaluations_sym.push(evaluation_sym);
        }
        Some((deep_poly_evaluations, deep_poly_evaluations_sym))
    }

    /// Reconstructs the deep composition polynomial evaluation at a query's
    /// point and its symmetric counterpart together. Rewriting the per-element
    /// trace term `coeff*(base-ood)*denom` as `denom*(coeff*base - coeff*ood)`
    /// isolates `coeff*ood` (identical for both points, hoisted into
    /// `query_invariant_terms`) from `coeff*base` (per-point), so both points
    /// share the OOD walk and a single batch-inverse for their denominators.
    /// g·z pruning restricts next rows (`row_idx >= step_size`) to the
    /// transition-window columns `next_row_cols` — all other next-row
    /// coefficients are zero, so those terms vanish from both sums.
    #[allow(clippy::too_many_arguments)]
    fn reconstruct_deep_composition_poly_evaluation_pair<'b>(
        evaluation_point: &FieldElement<Field>,
        evaluation_point_sym: &FieldElement<Field>,
        // The g^k·z base points are now hoisted into `query_invariant_terms`, so
        // this is consumed only by the Level-B `sim-ro-query` measurement ecall
        // below; elsewhere (host, guest without that feature) it is unused.
        #[cfg_attr(
            not(all(target_arch = "riscv64", feature = "sim-ro-query")),
            allow(unused_variables)
        )]
        primitive_root: &FieldElement<Field>,
        challenges: &Challenges<FieldExtension>,
        query_invariant_terms: &QueryInvariantDeepTerms<FieldExtension>,
        next_row_cols: &[usize],
        step_size: usize,
        lde_trace_precomputed_evaluations: &'b [FieldElement<Field>],
        lde_trace_main_evaluations: &'b [FieldElement<Field>],
        lde_trace_aux_evaluations: &[FieldElement<FieldExtension>],
        lde_composition_poly_parts_evaluation: &[FieldElement<FieldExtension>],
        lde_trace_precomputed_evaluations_sym: &'b [FieldElement<Field>],
        lde_trace_main_evaluations_sym: &'b [FieldElement<Field>],
        lde_trace_aux_evaluations_sym: &[FieldElement<FieldExtension>],
        lde_composition_poly_parts_evaluation_sym: &[FieldElement<FieldExtension>],
    ) -> Option<(FieldElement<FieldExtension>, FieldElement<FieldExtension>)> {
        // MEASUREMENT-ONLY stubs (never prove a build with these features on).
        // Level B (`sim-ro-query`) hands the whole function to one trusted
        // ecall; Level A (`sim-ro-ecalls`) replaces only the per-row column
        // loop below. Both are no-ops on host / with the features off, leaving
        // the original path byte-identical.
        #[cfg(all(target_arch = "riscv64", feature = "sim-ro-query"))]
        return Self::sim_reduced_opening_query_ecall(
            evaluation_point,
            evaluation_point_sym,
            primitive_root,
            challenges,
            query_invariant_terms,
            next_row_cols,
            step_size,
            lde_trace_precomputed_evaluations,
            lde_trace_main_evaluations,
            lde_trace_aux_evaluations,
            lde_composition_poly_parts_evaluation,
            lde_trace_precomputed_evaluations_sym,
            lde_trace_main_evaluations_sym,
            lde_trace_aux_evaluations_sym,
            lde_composition_poly_parts_evaluation_sym,
        );

        #[cfg(not(all(target_arch = "riscv64", feature = "sim-ro-query")))]
        return {
            let ood_evaluations_table_height = query_invariant_terms.ood_row_sum.len();
            let ood_evaluations_table_width = query_invariant_terms.ood_width;
            let trace_term_coeffs = &challenges.trace_term_coeffs;

            // Base columns are supplied as two slices (precomputed ‖ main) that the
            // prover concatenated in this order; `num_base`/`base_at` index into
            // them as if concatenated, without allocating. (`base_at`/`base_at_sym`
            // feed only the software column loop, so they are compiled out when
            // the Level A ecall replaces it.)
            let num_precomputed = lde_trace_precomputed_evaluations.len();
            let num_base = num_precomputed + lde_trace_main_evaluations.len();
            #[cfg(not(all(
                target_arch = "riscv64",
                any(feature = "sim-ro-ecalls", feature = "sim-ro-inplace")
            )))]
            let base_at = move |col: usize| -> &'b FieldElement<Field> {
                if col < num_precomputed {
                    &lde_trace_precomputed_evaluations[col]
                } else {
                    &lde_trace_main_evaluations[col - num_precomputed]
                }
            };
            let num_precomputed_sym = lde_trace_precomputed_evaluations_sym.len();
            let num_base_sym = num_precomputed_sym + lde_trace_main_evaluations_sym.len();
            #[cfg(not(all(
                target_arch = "riscv64",
                any(feature = "sim-ro-ecalls", feature = "sim-ro-inplace")
            )))]
            let base_at_sym = move |col: usize| -> &'b FieldElement<Field> {
                if col < num_precomputed_sym {
                    &lde_trace_precomputed_evaluations_sym[col]
                } else {
                    &lde_trace_main_evaluations_sym[col - num_precomputed_sym]
                }
            };

            // Runtime guards: a malformed proof may supply opening evaluations
            // whose column count does not match the OOD table width, or whose
            // regular/symmetric base-column split disagree. Without these checks
            // the indexing below would panic in release builds.
            if num_base != num_base_sym {
                return None;
            }
            if num_base + lde_trace_aux_evaluations.len() != ood_evaluations_table_width
                || num_base + lde_trace_aux_evaluations_sym.len() != ood_evaluations_table_width
            {
                return None;
            }

            // Build both denominator sets (regular, then symmetric) from the
            // query-invariant g^k·z base points (walked ONCE per table in
            // `compute_query_invariant_deep_terms`, not re-walked here per query),
            // and invert them together in a single batch. Only the per-query
            // subtraction against `evaluation_point[_sym]` remains in this loop.
            let gz_powers = &query_invariant_terms.gz_powers;
            let mut denoms = Vec::with_capacity(2 * ood_evaluations_table_height);
            for gz in gz_powers.iter() {
                denoms.push(evaluation_point - gz);
            }
            for gz in gz_powers.iter() {
                denoms.push(evaluation_point_sym - gz);
            }
            // A malformed proof can land an OOD evaluation point on the LDE coset, reject.
            FieldElement::inplace_batch_inverse(&mut denoms).ok()?;
            let (denoms_trace, denoms_trace_sym) = denoms.split_at(ood_evaluations_table_height);

            // Level A (`sim-ro-ecalls`): build the per-query reduced-opening
            // ecall input once (constant across the row loop). `sim_col_ptrs`
            // is a per-column pointer table into the [col][row] coeff grid; it
            // must outlive the loop (the ecall reads through its addresses).
            #[cfg(all(target_arch = "riscv64", feature = "sim-ro-ecalls"))]
            {
                debug_assert_eq!(core::mem::size_of::<FieldElement<Field>>(), 8);
                debug_assert_eq!(core::mem::size_of::<FieldElement<FieldExtension>>(), 24);
            }
            #[cfg(all(target_arch = "riscv64", feature = "sim-ro-ecalls"))]
            let sim_col_ptrs: Vec<u64> = trace_term_coeffs
                .iter()
                .map(|c| c.as_ptr() as u64)
                .collect();
            #[cfg(all(target_arch = "riscv64", feature = "sim-ro-ecalls"))]
            let sim_row_input = math::sim_ro::ReducedOpeningRowInput {
                precomputed_ptr: lde_trace_precomputed_evaluations.as_ptr() as u64,
                precomputed_len: lde_trace_precomputed_evaluations.len() as u64,
                main_ptr: lde_trace_main_evaluations.as_ptr() as u64,
                main_len: lde_trace_main_evaluations.len() as u64,
                aux_ptr: lde_trace_aux_evaluations.as_ptr() as u64,
                aux_len: lde_trace_aux_evaluations.len() as u64,
                precomputed_sym_ptr: lde_trace_precomputed_evaluations_sym.as_ptr() as u64,
                precomputed_sym_len: lde_trace_precomputed_evaluations_sym.len() as u64,
                main_sym_ptr: lde_trace_main_evaluations_sym.as_ptr() as u64,
                main_sym_len: lde_trace_main_evaluations_sym.len() as u64,
                aux_sym_ptr: lde_trace_aux_evaluations_sym.as_ptr() as u64,
                aux_sym_len: lde_trace_aux_evaluations_sym.len() as u64,
                coeff_col_ptrs_ptr: sim_col_ptrs.as_ptr() as u64,
                next_row_cols_ptr: next_row_cols.as_ptr() as u64,
                next_row_cols_len: next_row_cols.len() as u64,
                ood_width: ood_evaluations_table_width as u64,
                step_size: step_size as u64,
            };

            // Increment C: the only per-query marshaling is the six eval-slice
            // base pointers (the slices live where the verifier already holds
            // them). The coeff grid / dims / column counts were registered once
            // per proof via REGISTER_RO_LAYOUT, so the per-query struct fill +
            // col-ptr gather that Level A repeats for every query are gone. When
            // sim-ro-inplace is on it supplies base_row_sum per row; the outer
            // trace-term accumulation below still runs in-guest (and routes
            // through the FEXT accelerator), so the two levers compose.
            #[cfg(all(target_arch = "riscv64", feature = "sim-ro-inplace"))]
            let sim_evals: [u64; math::sim_ro::REDUCED_OPENING_INPLACE_EVALS] = [
                lde_trace_precomputed_evaluations.as_ptr() as u64,
                lde_trace_main_evaluations.as_ptr() as u64,
                lde_trace_aux_evaluations.as_ptr() as u64,
                lde_trace_precomputed_evaluations_sym.as_ptr() as u64,
                lde_trace_main_evaluations_sym.as_ptr() as u64,
                lde_trace_aux_evaluations_sym.as_ptr() as u64,
            ];

            // Resident trace-term accumulators: on the guest they live in FEXT
            // field-storage across the whole row loop, so each row is a single
            // LOAD/LOAD/FMA (3 ecalls) instead of the stateless `fma`'s
            // LOAD×3/FMA/STORE (5). Created regular-then-sym; finished sym-first
            // to keep the backend's accumulator stack LIFO.
            let mut trace_acc = FieldExtension::prod_acc_new();
            let mut trace_acc_sym = FieldExtension::prod_acc_new();
            for row_idx in 0..ood_evaluations_table_height {
                let ood_row_sum = &query_invariant_terms.ood_row_sum[row_idx];

                #[cfg(all(target_arch = "riscv64", feature = "sim-ro-ecalls"))]
                let (base_row_sum, base_row_sum_sym) = {
                    // Two-element literal (not `[x; 2]`): the generic
                    // `FieldExtension` isn't `Copy`. The ecall (asm memory
                    // clobber) writes both elements; move them out afterward.
                    let mut out = [
                        FieldElement::<FieldExtension>::zero(),
                        FieldElement::<FieldExtension>::zero(),
                    ];
                    lambda_vm_syscalls::syscalls::reduced_opening_row(
                        &sim_row_input as *const _ as usize,
                        row_idx,
                        out.as_mut_ptr() as usize,
                    );
                    let [base_row_sum, base_row_sum_sym] = out;
                    (base_row_sum, base_row_sum_sym)
                };

                // Increment C: the in-place row ecall. Passes only row_idx + the
                // per-query eval-slice base pointers + the out scratch; the
                // executor reads the coeff grid / dims from the registered layout.
                #[cfg(all(target_arch = "riscv64", feature = "sim-ro-inplace"))]
                let (base_row_sum, base_row_sum_sym) = {
                    let mut out = [
                        FieldElement::<FieldExtension>::zero(),
                        FieldElement::<FieldExtension>::zero(),
                    ];
                    lambda_vm_syscalls::syscalls::reduced_opening_row_inplace(
                        row_idx,
                        sim_evals.as_ptr() as usize,
                        out.as_mut_ptr() as usize,
                    );
                    let [base_row_sum, base_row_sum_sym] = out;
                    (base_row_sum, base_row_sum_sym)
                };

                #[cfg(not(all(
                    target_arch = "riscv64",
                    any(feature = "sim-ro-ecalls", feature = "sim-ro-inplace")
                )))]
                let (base_row_sum, base_row_sum_sym) = {
                    // Base×ext products keep the cheap asymmetric software path;
                    // the ext×ext aux products accumulate into a resident FEXT
                    // accumulator (regular + sym) and are folded into the base
                    // partial at the end. Addition is commutative, so splitting
                    // base from aux is exact. Accumulators are created
                    // regular-then-sym, finished sym-first (backend stack LIFO).
                    let mut base_soft = FieldElement::<FieldExtension>::zero();
                    let mut base_soft_sym = FieldElement::<FieldExtension>::zero();
                    let mut aux_acc = FieldExtension::prod_acc_new();
                    let mut aux_acc_sym = FieldExtension::prod_acc_new();
                    if row_idx < step_size {
                        for (col_idx, coeff_col) in trace_term_coeffs.iter().enumerate() {
                            let coeff = &coeff_col[row_idx];
                            if col_idx < num_base {
                                // F: IsSubFieldOf<E> gives the cheap asymmetric F * E -> E product.
                                base_soft += base_at(col_idx) * coeff;
                                base_soft_sym += base_at_sym(col_idx) * coeff;
                            } else {
                                let aux_idx = col_idx - num_base;
                                // Ext × ext products route through the FEXT accelerator
                                // on the guest (resident `acc += aux * coeff`; plain
                                // software off the guest / for non-Fp3 fields).
                                FieldExtension::prod_acc_add2(
                                    &mut aux_acc,
                                    &lde_trace_aux_evaluations[aux_idx],
                                    coeff,
                                );
                                FieldExtension::prod_acc_add2(
                                    &mut aux_acc_sym,
                                    &lde_trace_aux_evaluations_sym[aux_idx],
                                    coeff,
                                );
                            }
                        }
                    } else {
                        // g·z pruning: the next-row block opens only transition-window
                        // columns; every other column's coefficient is zero
                        // (`build_pruned_trace_term_coeffs`), so summing the window
                        // alone is exact — and skipping the rest is where the
                        // verifier/recursion cycle saving lands.
                        for &col_idx in next_row_cols {
                            let coeff = &trace_term_coeffs[col_idx][row_idx];
                            if col_idx < num_base {
                                base_soft += base_at(col_idx) * coeff;
                                base_soft_sym += base_at_sym(col_idx) * coeff;
                            } else {
                                let aux_idx = col_idx - num_base;
                                // Ext × ext products route through the FEXT accelerator
                                // on the guest (resident `acc += aux * coeff`; plain
                                // software off the guest / for non-Fp3 fields).
                                FieldExtension::prod_acc_add2(
                                    &mut aux_acc,
                                    &lde_trace_aux_evaluations[aux_idx],
                                    coeff,
                                );
                                FieldExtension::prod_acc_add2(
                                    &mut aux_acc_sym,
                                    &lde_trace_aux_evaluations_sym[aux_idx],
                                    coeff,
                                );
                            }
                        }
                    }
                    // Finish sym-first (LIFO), then fold in the base partials.
                    let aux_sym = FieldExtension::prod_acc_finish(aux_acc_sym);
                    let aux = FieldExtension::prod_acc_finish(aux_acc);
                    (&base_soft + &aux, &base_soft_sym + &aux_sym)
                };

                // Trace-term accumulation routes through the FEXT accelerator on
                // the guest as a resident accumulate
                // `trace_acc += denom * (base_row_sum - ood_row_sum)`.
                FieldExtension::prod_acc_add2(
                    &mut trace_acc,
                    &denoms_trace[row_idx],
                    &(&base_row_sum - ood_row_sum),
                );
                FieldExtension::prod_acc_add2(
                    &mut trace_acc_sym,
                    &denoms_trace_sym[row_idx],
                    &(&base_row_sum_sym - ood_row_sum),
                );
            }
            // Finish sym-first (LIFO: it was created after the regular chain).
            let trace_term_sym = FieldExtension::prod_acc_finish(trace_acc_sym);
            let trace_term = FieldExtension::prod_acc_finish(trace_acc);

            let number_of_parts = query_invariant_terms.number_of_parts;
            // Also rejects a per-query opening length that disagrees with the
            // proof-level `number_of_parts`, not just a regular/symmetric mismatch.
            if lde_composition_poly_parts_evaluation.len() != number_of_parts
                || lde_composition_poly_parts_evaluation_sym.len() != number_of_parts
            {
                return None;
            }
            let z_pow = &query_invariant_terms.z_pow;

            // A malformed proof can make evaluation_point == z_pow, reject.
            let mut denom_composition_pair =
                [evaluation_point - z_pow, evaluation_point_sym - z_pow];
            FieldElement::inplace_batch_inverse(&mut denom_composition_pair).ok()?;
            let [denom_composition, denom_composition_sym] = denom_composition_pair;

            // Composition-part combination `h_sum += h_i * gamma`: an ext×ext
            // accumulate, resident on the FEXT accelerator (previously plain
            // software `*`, untouched by #831). Created regular-then-sym after
            // the trace-term chain freed its stack regions; finished sym-first.
            let mut h_acc = FieldExtension::prod_acc_new();
            let mut h_acc_sym = FieldExtension::prod_acc_new();
            for j in 0..number_of_parts {
                let gamma = &challenges.gammas[j];
                FieldExtension::prod_acc_add2(
                    &mut h_acc,
                    &lde_composition_poly_parts_evaluation[j],
                    gamma,
                );
                FieldExtension::prod_acc_add2(
                    &mut h_acc_sym,
                    &lde_composition_poly_parts_evaluation_sym[j],
                    gamma,
                );
            }
            let h_sum_sym = FieldExtension::prod_acc_finish(h_acc_sym);
            let h_sum = FieldExtension::prod_acc_finish(h_acc);
            // `(h_sum - h_sum_zpow) * denom_composition`: ext×ext product routed
            // through the accelerator.
            let h_terms = FieldExtension::ext_mul(
                &(&h_sum - &query_invariant_terms.h_sum_zpow),
                &denom_composition,
            );
            let h_terms_sym = FieldExtension::ext_mul(
                &(&h_sum_sym - &query_invariant_terms.h_sum_zpow),
                &denom_composition_sym,
            );

            Some((trace_term + h_terms, trace_term_sym + h_terms_sym))
        };
    }

    /// MEASUREMENT-ONLY (Level B, `sim-ro-query`). Marshals every input of
    /// [`Self::reconstruct_deep_composition_poly_evaluation_pair`] into the
    /// [`math::sim_ro::ReducedOpeningQueryInput`] ABI and delegates the whole
    /// `(deep_eval, deep_eval_sym)` reconstruction to the trusted
    /// `REDUCED_OPENING_QUERY` ecall (see `others/accelerator_noop_sim_spec.md`,
    /// Experiment 2, fusion upper bound). NEVER prove a build with this on — the
    /// unmatched ecall unbalances the LogUp bus.
    ///
    /// Field-element scalars are passed by pointer (this generic method cannot
    /// assume `Field::BaseType == u64`); `sim_col_ptrs` (the per-column pointer
    /// table into the coeff grid) must outlive the ecall.
    #[cfg(all(target_arch = "riscv64", feature = "sim-ro-query"))]
    #[allow(clippy::too_many_arguments)]
    fn sim_reduced_opening_query_ecall<'b>(
        evaluation_point: &FieldElement<Field>,
        evaluation_point_sym: &FieldElement<Field>,
        primitive_root: &FieldElement<Field>,
        challenges: &Challenges<FieldExtension>,
        query_invariant_terms: &QueryInvariantDeepTerms<FieldExtension>,
        next_row_cols: &[usize],
        step_size: usize,
        lde_trace_precomputed_evaluations: &'b [FieldElement<Field>],
        lde_trace_main_evaluations: &'b [FieldElement<Field>],
        lde_trace_aux_evaluations: &[FieldElement<FieldExtension>],
        lde_composition_poly_parts_evaluation: &[FieldElement<FieldExtension>],
        lde_trace_precomputed_evaluations_sym: &'b [FieldElement<Field>],
        lde_trace_main_evaluations_sym: &'b [FieldElement<Field>],
        lde_trace_aux_evaluations_sym: &[FieldElement<FieldExtension>],
        lde_composition_poly_parts_evaluation_sym: &[FieldElement<FieldExtension>],
    ) -> Option<(FieldElement<FieldExtension>, FieldElement<FieldExtension>)> {
        debug_assert_eq!(core::mem::size_of::<FieldElement<Field>>(), 8);
        debug_assert_eq!(core::mem::size_of::<FieldElement<FieldExtension>>(), 24);
        let sim_col_ptrs: Vec<u64> = challenges
            .trace_term_coeffs
            .iter()
            .map(|c| c.as_ptr() as u64)
            .collect();
        let input = math::sim_ro::ReducedOpeningQueryInput {
            evaluation_point_ptr: evaluation_point as *const _ as u64,
            evaluation_point_sym_ptr: evaluation_point_sym as *const _ as u64,
            primitive_root_ptr: primitive_root as *const _ as u64,
            z_ptr: &challenges.z as *const _ as u64,
            z_pow_ptr: &query_invariant_terms.z_pow as *const _ as u64,
            h_sum_zpow_ptr: &query_invariant_terms.h_sum_zpow as *const _ as u64,
            ood_height: query_invariant_terms.ood_row_sum.len() as u64,
            ood_width: query_invariant_terms.ood_width as u64,
            number_of_parts: query_invariant_terms.number_of_parts as u64,
            step_size: step_size as u64,
            precomputed_ptr: lde_trace_precomputed_evaluations.as_ptr() as u64,
            precomputed_len: lde_trace_precomputed_evaluations.len() as u64,
            main_ptr: lde_trace_main_evaluations.as_ptr() as u64,
            main_len: lde_trace_main_evaluations.len() as u64,
            aux_ptr: lde_trace_aux_evaluations.as_ptr() as u64,
            aux_len: lde_trace_aux_evaluations.len() as u64,
            precomputed_sym_ptr: lde_trace_precomputed_evaluations_sym.as_ptr() as u64,
            precomputed_sym_len: lde_trace_precomputed_evaluations_sym.len() as u64,
            main_sym_ptr: lde_trace_main_evaluations_sym.as_ptr() as u64,
            main_sym_len: lde_trace_main_evaluations_sym.len() as u64,
            aux_sym_ptr: lde_trace_aux_evaluations_sym.as_ptr() as u64,
            aux_sym_len: lde_trace_aux_evaluations_sym.len() as u64,
            composition_ptr: lde_composition_poly_parts_evaluation.as_ptr() as u64,
            composition_sym_ptr: lde_composition_poly_parts_evaluation_sym.as_ptr() as u64,
            coeff_col_ptrs_ptr: sim_col_ptrs.as_ptr() as u64,
            gammas_ptr: challenges.gammas.as_ptr() as u64,
            ood_row_sum_ptr: query_invariant_terms.ood_row_sum.as_ptr() as u64,
            next_row_cols_ptr: next_row_cols.as_ptr() as u64,
            next_row_cols_len: next_row_cols.len() as u64,
        };
        let mut out = [
            FieldElement::<FieldExtension>::zero(),
            FieldElement::<FieldExtension>::zero(),
        ];
        lambda_vm_syscalls::syscalls::reduced_opening_query(
            &input as *const _ as usize,
            out.as_mut_ptr() as usize,
        );
        let [deep_eval, deep_eval_sym] = out;
        Some((deep_eval, deep_eval_sym))
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
        Self::multi_verify_views(
            airs,
            MultiProofView::Owned(multi_proof),
            transcript,
            expected_bus_balance,
        )
    }

    /// Verifies one or more rkyv-archived STARK proofs read **in place** from
    /// their archive buffer — no proof deserialization, no per-field allocation.
    fn multi_verify_archived(
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        multi_proof: &ArchivedMultiProof<Field, FieldExtension, PI>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
        expected_bus_balance: &FieldElement<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        Self::multi_verify_views(
            airs,
            MultiProofView::Archived(multi_proof),
            transcript,
            expected_bus_balance,
        )
    }

    /// The single verification implementation, shared by [`Self::multi_verify`]
    /// (owned) and [`Self::multi_verify_archived`] (archived), operating on
    /// proof views rather than either's concrete type.
    fn multi_verify_views<'p>(
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        proofs: impl ProofViewSource<'p, Field, FieldExtension, PI>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
        expected_bus_balance: &FieldElement<FieldExtension>,
    ) -> bool
    where
        Field: 'p,
        FieldExtension: 'p,
        PI: 'p,
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        if airs.len() != proofs.view_len() {
            error!(
                "AIR count ({}) does not match proof count ({})",
                airs.len(),
                proofs.view_len()
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

        for (idx, (air, proof)) in airs.iter().zip(proofs.view_iter()).enumerate() {
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
            // The archive is read in place without validation, so both OOD blocks
            // must be shape-checked here — before Round 3 absorbs the next-row
            // block and before any row access indexes into either. The width check
            // is load-bearing: it stops the AIR-derived column index
            // `main_trace_width + c.col` in `step_2_verify_claimed_composition_polynomial`
            // from indexing past a too-narrow OOD row, and it rejects a width-0
            // table, whose `width * height == 0 == data.len()` would otherwise
            // satisfy `dimensions_consistent()` for any advertised height.
            if !Self::ood_blocks_well_formed(*air, proof) {
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

        for (idx, (air, proof)) in airs.iter().zip(proofs.view_iter()).enumerate() {
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

        for (idx, (air, proof)) in airs.iter().zip(proofs.view_iter()).enumerate() {
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
            for (air, proof) in airs.iter().zip(proofs.view_iter()) {
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
            &[StarkProofView::Owned(proof)][..],
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
        layout: &crate::ood::OodLayout,
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

        // Resident-handle power sequence (see `geometric_powers`): β stays loaded
        // in the FEXT chip across [1, β, β², …]. Byte-identical to the running
        // product `compute_alpha_powers` computed here before.
        let mut coefficients = FieldExtension::geometric_powers(
            &beta,
            num_boundary_constraints + num_transition_constraints,
        );

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
        // Must match the prover's g·z pruning exactly (same AIR metadata): the
        // current-row block opens every column, the next-row block only the
        // transition-window columns.
        let num_terms_trace = layout.num_surviving();
        let gamma = transcript.sample_field_element();

        // <<<< Receive challenges: 𝛾, 𝛾'
        // `Take<Successors>` reports a lower size-hint bound of 0, so a plain
        // `.collect()` starts from an empty Vec and reallocates through the whole
        // doubling schedule. The final length is known exactly here, so reserve it
        // once and extend — identical values, no realloc/copy churn.
        let num_deep_coeffs = num_terms_composition_poly + num_terms_trace;
        let mut deep_composition_coefficients: Vec<FieldElement<FieldExtension>> =
            Vec::with_capacity(num_deep_coeffs);
        deep_composition_coefficients.extend(
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
                .take(num_deep_coeffs),
        );

        let trace_term_powers: Vec<_> = deep_composition_coefficients
            .drain(..num_terms_trace)
            .collect();
        let trace_term_coeffs = layout.build_trace_term_coeffs(&trace_term_powers);

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

        // The pruned-OOD layout, read from the AIR once and shared by the round-4
        // challenge replay, the block-shape guard, the single grid reconstruction,
        // and both verify steps below — one reconstruction instead of the previous
        // two, and no chance of the sites drifting apart.
        let layout = Self::ood_layout(air);

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
            &layout,
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

        // Reject either OOD block whose shape disagrees with the AIR before
        // reconstructing or using it, so a malicious prover cannot reshape them
        // to dodge a check or desync the frame reconstruction. This guard used to
        // run at the top of `step_2`; `step_3` silently relied on it. Now it runs
        // once here, before both steps, and the full grid is reconstructed once
        // and shared with them (one reconstruction instead of two). The Phase A
        // loop in `multi_verify_views` runs the same guard even earlier, before
        // Round 3 absorbs the next-row block.
        if !Self::ood_blocks_well_formed(air, proof) {
            #[cfg(not(feature = "test_fiat_shamir"))]
            error!("Composition Polynomial verification failed");
            return false;
        }
        let ood_current = proof.trace_ood_evaluations();
        let ood_next = proof.trace_ood_next_evaluations();
        // Full current+next-row OOD grid (surviving values placed, pruned next-row
        // entries zero — those are never read by any constraint).
        let ood_full = layout.reconstruct_full(
            ood_current.row_major_data(),
            ood_current.width(),
            ood_next.row_major_data(),
        );

        if !Self::step_2_verify_claimed_composition_polynomial(
            air,
            proof,
            public_inputs,
            &domain,
            &challenges,
            &ood_full,
            layout.step_size(),
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

        if !Self::step_3_verify_fri(
            air,
            proof,
            &domain,
            &challenges,
            &ood_full,
            layout.next_row_cols(),
            layout.step_size(),
        ) {
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
    /// `reconstruct_deep_composition_poly_evaluation_pair` actually read
    /// (trace_length, OOD evaluations, precomputed root, bus/public inputs). All
    /// commitment/opening/FRI fields are placeholders those two helpers never
    /// inspect — this lets the batched verifier reuse them unchanged.
    fn batched_synthetic_table_proof(
        table: BatchedTableDataView<'_, FieldExtension, PI>,
    ) -> Option<StarkProof<Field, FieldExtension, PI>>
    where
        PI: Clone,
    {
        // Only the small per-table OOD blocks + tiny scalars are materialized here
        // (never a per-query opening): the two OOD tables are `step_size × width`,
        // so the synthetic proof stays a bounded, per-table copy. `bus_public_inputs`
        // is rebuilt from just the `table_contribution` the verifier reads (the
        // debug-only per-bus aggregation is not part of the archived proof).
        Some(StarkProof {
            trace_length: table.trace_length(),
            lde_trace_main_merkle_root: [0u8; 32],
            lde_trace_aux_merkle_root: None,
            lde_trace_precomputed_merkle_root: table.precomputed_root().copied(),
            // Split OOD (g·z pruning): current-row block + pruned next-row block,
            // the same shape the non-batched `StarkProof` carries. The batched
            // verifier reconstructs the full grid from these two exactly as
            // `verify_rounds_2_to_4` does.
            trace_ood_evaluations: table.trace_ood_evaluations().to_owned_table(),
            trace_ood_next_evaluations: table.trace_ood_next_evaluations().to_owned_table(),
            composition_poly_root: [0u8; 32],
            composition_poly_parts_ood_evaluation: table
                .composition_poly_parts_ood_evaluation()
                .to_vec(),
            fri_layers_merkle_roots: Vec::new(),
            fri_final_poly_coeffs: Vec::new(),
            query_list: Vec::new(),
            deep_poly_openings: Vec::new(),
            nonce: None,
            bus_public_inputs: table
                .bus_table_contribution()
                .map(BusPublicInputs::from_contribution),
            public_inputs: table.public_inputs()?,
        })
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
        let view = BatchedMultiProofView::Owned(proof);
        let mid = match Self::batched_verify_rounds_1_to_3(airs, view, transcript) {
            Some(m) => m,
            None => return false,
        };
        Self::batched_verify_round_4(mid, airs, view, transcript, expected_bus_balance)
    }

    /// Rounds 1-3 of the batched (unified-shard) verifier: replays the Fiat-Shamir
    /// transcript from Phase A (preprocessed roots + the single main MMCS root)
    /// through the OOD absorption, returning the derived `VmMidState` that round 4
    /// consumes. Split out of `batched_multi_verify` (behavior-preserving) so the
    /// continuation epoch verifier can weave the separate L2G lane in at the seam.
    /// Returns `None` on any structural rejection.
    fn batched_verify_rounds_1_to_3(
        airs: &[&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>],
        proof: BatchedMultiProofView<'_, Field, FieldExtension, PI>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
    ) -> Option<VmMidState<Field, FieldExtension>>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
        PI: Clone,
    {
        let num_tables = airs.len();
        if num_tables == 0 || num_tables != proof.per_table_len() {
            return None;
        }

        // Per-table lightweight domains + FRI heights (= lde_log_height).
        let domains: Vec<VerifierDomain<Field>> = airs
            .iter()
            .enumerate()
            .map(|(i, air)| new_verifier_domain(*air, proof.per_table(i).trace_length()))
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
        for (i, air) in airs.iter().enumerate() {
            let t = proof.per_table(i);
            let trace_length = t.trace_length();
            // Soundness: composition part count is fixed by the AIR degree bound,
            // not chosen by the prover.
            if trace_length == 0
                || t.composition_poly_parts_ood_evaluation().len()
                    != air.composition_poly_degree_bound(trace_length) / trace_length
            {
                return None;
            }
            // Both OOD blocks' shapes are a public function of the AIR (invariant
            // I3), never the prover's. Reject any table whose current/next-row
            // block disagrees BEFORE Round 3 absorbs them and before the round-4
            // frame/DEEP reconstruction indexes into them. Mirrors the non-batched
            // `ood_blocks_well_formed`, but the current-row width is checked against
            // `context().trace_columns` (the physical OOD width = main + aux, the
            // same figure `OodLayout.num_total_cols` and the DEEP trace-term grid
            // use) rather than `trace_layout().0 + num_aux`: the batched lane
            // includes step-packed AIRs (e.g. BitFlags) whose `trace_layout().0` is
            // a logical, not physical, column count. The width check is load-bearing
            // (a wrong width desyncs the frame/grid reconstruction and would make
            // `compute_query_invariant_deep_terms` reject a valid proof, or let a
            // too-narrow row index out of bounds). All owned tables here, but
            // `dimensions_consistent` still rejects a mis-deserialized one.
            let layout = Self::ood_layout(*air);
            let ood_current = t.trace_ood_evaluations();
            let ood_next = t.trace_ood_next_evaluations();
            if !ood_current.dimensions_consistent()
                || ood_current.width() != air.context().trace_columns
                || ood_current.height() != layout.step_size()
                || !ood_next.dimensions_consistent()
                || ood_next.width() != layout.expected_next_width()
                || ood_next.height() != layout.expected_next_height()
            {
                return None;
            }
            if air.is_preprocessed() {
                let expected = air.precomputed_commitment();
                match t.precomputed_root().copied() {
                    Some(actual) if actual == expected => {}
                    _ => return None,
                }
                transcript.append_bytes(&expected);
            } else if t.precomputed_root().is_some() {
                return None;
            }
        }
        transcript.append_bytes(proof.main_root());

        // Bus-input presence must match the AIR layout (a dishonest prover could
        // omit bus_public_inputs to bypass the balance check).
        for (i, air) in airs.iter().enumerate() {
            if air.has_trace_interaction() != proof.per_table(i).has_bus_public_inputs() {
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
        if needs_lookup_challenges != proof.aux_root().is_some() {
            return None;
        }
        if let Some(root) = proof.aux_root() {
            transcript.append_bytes(root);
        }

        // Bus contributions bind before the round-2 challenges.
        for i in 0..num_tables {
            if let Some(contribution) = proof.per_table(i).bus_table_contribution() {
                transcript.append_field_element(&contribution);
            }
        }

        // ===== Round 2: per-table beta -> boundary/transition coeffs, then the
        // single batched composition MMCS root. =====
        let mut boundary_coeffs_all: Vec<Vec<FieldElement<FieldExtension>>> =
            Vec::with_capacity(num_tables);
        let mut transition_coeffs_all: Vec<Vec<FieldElement<FieldExtension>>> =
            Vec::with_capacity(num_tables);
        for (i, air) in airs.iter().enumerate() {
            let t = proof.per_table(i);
            // Rebuild the tiny PI + bus contribution the boundary builder reads
            // (a per-table copy of one `PI` and one field element — no bulk data).
            let public_inputs = t.public_inputs()?;
            let bus_public_inputs = t
                .bus_table_contribution()
                .map(BusPublicInputs::from_contribution);
            let beta = transcript.sample_field_element();
            let num_boundary = air
                .boundary_constraints(
                    &public_inputs,
                    &lookup_challenges,
                    bus_public_inputs.as_ref(),
                    t.trace_length(),
                )
                .constraints
                .len();
            let num_transition = air.context().num_transition_constraints;
            let mut coeffs = compute_alpha_powers(&beta, num_boundary + num_transition);
            let transition_coeffs: Vec<_> = coeffs.drain(..num_transition).collect();
            transition_coeffs_all.push(transition_coeffs);
            boundary_coeffs_all.push(coeffs);
        }
        transcript.append_bytes(proof.composition_root());

        // ===== Round 3: shared z (tallest domain), per-table OOD absorbed. =====
        let z = transcript.sample_z_ood_with_domain_params(
            domains[tallest].trace_length,
            domains[tallest].lde_length,
            &domains[tallest].coset_offset,
        );
        for i in 0..num_tables {
            let t = proof.per_table(i);
            // g·z pruning: absorb the current-row block then the pruned next-row
            // block, each column-major, in the exact order the prover sent them.
            // Index `row_major_data` column-major directly (byte-identical to
            // `Table::columns()`) so the archived OOD block is read in place.
            for block in [t.trace_ood_evaluations(), t.trace_ood_next_evaluations()] {
                let width = block.width();
                let height = block.height();
                let data = block.row_major_data();
                for col in 0..width {
                    for row in 0..height {
                        transcript.append_field_element(&data[row * width + col]);
                    }
                }
            }
            for elem in t.composition_poly_parts_ood_evaluation().iter() {
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
        proof: BatchedMultiProofView<'_, Field, FieldExtension, PI>,
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

        // Per-table pruned-OOD layout (AIR-derived, never proof-derived — I3),
        // built once and shared by the round-4 challenge replay, the full-grid
        // reconstruction, step 2, the query-invariant hoist, and the per-query
        // DEEP reconstruction. Mirrors the single `layout` the non-batched
        // `verify_rounds_2_to_4` threads through those sites.
        let layouts: Vec<crate::ood::OodLayout> =
            airs.iter().map(|air| Self::ood_layout(*air)).collect();

        // ===== Round 4: shared gamma, per-table DEEP coeffs, batched FRI challenges. =====
        let gamma = transcript.sample_field_element();
        let mut trace_term_coeffs_all: Vec<Vec<Vec<FieldElement<FieldExtension>>>> =
            Vec::with_capacity(num_tables);
        let mut gammas_all: Vec<Vec<FieldElement<FieldExtension>>> = Vec::with_capacity(num_tables);
        for (i, layout) in layouts.iter().enumerate() {
            let num_terms_comp = proof
                .per_table(i)
                .composition_poly_parts_ood_evaluation()
                .len();
            // g·z pruning: draw only the surviving trace-term powers (current-row
            // block all columns + next-row block masked columns) and scatter them
            // into the rectangular W×num_eval_points grid with zeros at pruned
            // positions — identical to the prover's `build_trace_term_coeffs` and
            // the non-batched round-4 replay.
            let num_terms_trace = layout.num_surviving();
            let mut coeffs: Vec<_> =
                core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
                    .take(num_terms_comp + num_terms_trace)
                    .collect();
            let trace_term_powers: Vec<_> = coeffs.drain(..num_terms_trace).collect();
            let trace_term_coeffs = layout.build_trace_term_coeffs(&trace_term_powers);
            trace_term_coeffs_all.push(trace_term_coeffs);
            gammas_all.push(coeffs);
        }

        let grinding_factor = airs[0].context().proof_options.grinding_factor;
        let num_queries = airs[0].options().fri_number_of_queries;
        let fri_domain_size = 1usize << h_max;
        let fri_last_value = proof.fri_last_value();
        let nonce = proof.nonce();
        let fri_challenges = derive_batched_fri_challenges(
            transcript,
            &heights,
            proof.fri_layers_merkle_roots(),
            &fri_last_value,
            grinding_factor,
            nonce,
            num_queries,
            fri_domain_size,
        );
        let alpha = fri_challenges.alpha;
        let betas_fri = fri_challenges.betas;
        let iotas = fri_challenges.iotas;

        // Grinding.
        if grinding_factor > 0 {
            let ok = nonce.is_some_and(|n| {
                grinding::is_valid_nonce(&fri_challenges.grinding_seed, n, grinding_factor)
            });
            if !ok {
                return false;
            }
        }

        if proof.query_list_len() < num_queries || proof.deep_poly_openings_len() < num_queries {
            return false;
        }

        // Per-table synthetic proofs + Challenges (reused by step 2 and the query
        // loop). Built from borrowed per-table views: only the small OOD blocks are
        // materialized (never a per-query opening). A failed PI deserialize rejects.
        let synth_proofs: Vec<StarkProof<Field, FieldExtension, PI>> = match (0..num_tables)
            .map(|i| Self::batched_synthetic_table_proof(proof.per_table(i)))
            .collect::<Option<Vec<_>>>()
        {
            Some(v) => v,
            None => return false,
        };
        // Move (not clone) each table's per-table coefficient vectors into its
        // `Challenges`; they are never read again after this. Only `z` (one field
        // element) and the shared `rap_challenges` are cloned per table.
        let table_challenges: Vec<Challenges<FieldExtension>> = itertools::izip!(
            boundary_coeffs_all,
            transition_coeffs_all,
            trace_term_coeffs_all,
            gammas_all,
        )
        .map(
            |(boundary_coeffs, transition_coeffs, trace_term_coeffs, gammas)| Challenges {
                z: z.clone(),
                boundary_coeffs,
                transition_coeffs,
                trace_term_coeffs,
                gammas,
                zetas: Vec::new(),
                iotas: Vec::new(),
                rap_challenges: lookup_challenges.clone(),
                grinding_seed: [0u8; 32],
            },
        )
        .collect();

        // The full current+next-row OOD grid, reconstructed once per table from
        // the two pruned proof blocks and shared by step 2, the query-invariant
        // hoist, and the per-query DEEP reconstruction — exactly as
        // `verify_rounds_2_to_4` reconstructs it once and threads it through.
        // Pruned next-row entries are zero: no constraint reads them and DEEP
        // skips them.
        let ood_fulls: Vec<Table<FieldExtension>> = (0..num_tables)
            .map(|i| {
                let current = &synth_proofs[i].trace_ood_evaluations;
                let next = &synth_proofs[i].trace_ood_next_evaluations;
                layouts[i].reconstruct_full(
                    current.row_major_data(),
                    current.width,
                    next.row_major_data(),
                )
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
                &ood_fulls[i],
                layouts[i].step_size(),
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
        let comp_widths: Vec<usize> = (0..num_tables)
            .map(|i| {
                proof
                    .per_table(i)
                    .composition_poly_parts_ood_evaluation()
                    .len()
            })
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
        let num_layers = proof.fri_layers_merkle_roots().len();

        // Per-table DEEP query-invariant terms, hoisted out of the query loop
        // (#826): the OOD/gamma sums depend only on each table's challenges and
        // reconstructed OOD grid, not on the query point. g·z pruning: the
        // transition-window columns (`layouts[i].next_row_cols()`) are the only
        // next-row openings that survive, derived from the AIR (never the proof),
        // and the pruned-away next-row terms are skipped — the same reconstruction
        // the non-batched path performs and the batched prover's DEEP codeword
        // committed.
        let mut query_invariant_terms_all = Vec::with_capacity(num_tables);
        for i in 0..num_tables {
            let terms = match Self::compute_query_invariant_deep_terms(
                &table_challenges[i],
                StarkProofView::Owned(&synth_proofs[i]),
                &ood_fulls[i],
                layouts[i].next_row_cols(),
                layouts[i].step_size(),
                // Per-table trace generator `g` for the hoisted g^k·z DEEP
                // denominator walk (sim/22 change #2). Same value the query loop
                // hands `reconstruct_deep_composition_poly_evaluation_pair` below
                // as `prim_root`; `trace_primitive_root == get_primitive_root_of_unity(root_order)`.
                &domains[i].trace_primitive_root,
            ) {
                Some(t) => t,
                None => return false,
            };
            query_invariant_terms_all.push(terms);
        }

        // ===== Per query: MMCS openings, DEEP reconstruction, fold-and-inject FRI. =====
        for (q, &iota) in iotas.iter().enumerate() {
            let qo = proof.deep_poly_opening(q);

            // 1) Authenticate the shared per-phase openings.
            if !MixedMmcs::<Field>::verify_batch_view(
                proof.main_root(),
                iota,
                qo.main(),
                &heights,
                &main_widths,
            ) {
                return false;
            }
            match (proof.aux_root(), qo.aux()) {
                (Some(root), Some(aux_op)) => {
                    if !MixedMmcs::<FieldExtension>::verify_batch_view(
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
            if !MixedMmcs::<FieldExtension>::verify_batch_view(
                proof.composition_root(),
                iota,
                qo.composition(),
                &heights,
                &comp_widths,
            ) {
                return false;
            }

            // Precomputed openings (one per preprocessed table, in that order).
            if qo.precomputed_len() != precomputed_indices.len() {
                return false;
            }
            for (pc, &ti) in precomputed_indices.iter().enumerate() {
                let root = match proof.per_table(ti).precomputed_root().copied() {
                    Some(r) => r,
                    None => return false,
                };
                let local = iota >> (h_max - heights[ti]);
                if !Self::verify_opening_pair::<Field>(qo.precomputed(pc), &root, local) {
                    return false;
                }
            }

            // 2) Reconstruct each table's DEEP value at its opened row pair.
            let mut deep_primary = vec![FieldElement::<FieldExtension>::zero(); num_tables];
            let mut deep_sym = vec![FieldElement::<FieldExtension>::zero(); num_tables];
            for i in 0..num_tables {
                let leaf = iota >> (h_max - heights[i]);
                // Per-matrix openings as borrowed views (owned or archived-in-place):
                // their evaluation slices are read straight off the proof buffer.
                let main_op = qo.main().per_matrix(i);
                let comp_op = qo.composition().per_matrix(i);
                let precomp_op = precomputed_indices
                    .iter()
                    .position(|&x| x == i)
                    .map(|pc| qo.precomputed(pc));
                let aux_op = aux_indices
                    .iter()
                    .position(|&x| x == i)
                    .and_then(|ai| qo.aux().map(|a| a.per_matrix(ai)));

                // Base columns as two borrowed slices in commit order
                // (precomputed FIRST, then main) — the pair reconstruction
                // resolves a base column via its own `base_at`, so there is no
                // per-query concat allocation.
                let precomp_p: &[FieldElement<Field>] =
                    precomp_op.map(|p| p.evaluations()).unwrap_or(&[]);
                let precomp_s: &[FieldElement<Field>] =
                    precomp_op.map(|p| p.evaluations_sym()).unwrap_or(&[]);
                let aux_p: &[FieldElement<FieldExtension>] =
                    aux_op.map(|a| a.evaluations()).unwrap_or(&[]);
                let aux_s: &[FieldElement<FieldExtension>] =
                    aux_op.map(|a| a.evaluations_sym()).unwrap_or(&[]);

                let prim_root = &domains[i].trace_primitive_root;
                let ep_p = domains[i]
                    .lde_coset_element(reverse_index(leaf * 2, domains[i].lde_length as u64));
                let ep_s = domains[i]
                    .lde_coset_element(reverse_index(leaf * 2 + 1, domains[i].lde_length as u64));

                // Reconstruct the DEEP value at the query's row pair (regular +
                // symmetric) together, sharing the hoisted OOD/gamma sums (#826).
                let (dp, ds) = match Self::reconstruct_deep_composition_poly_evaluation_pair(
                    &ep_p,
                    &ep_s,
                    prim_root,
                    &table_challenges[i],
                    &query_invariant_terms_all[i],
                    layouts[i].next_row_cols(),
                    layouts[i].step_size(),
                    precomp_p,
                    main_op.evaluations(),
                    aux_p,
                    comp_op.evaluations(),
                    precomp_s,
                    main_op.evaluations_sym(),
                    aux_s,
                    comp_op.evaluations_sym(),
                ) {
                    Some(v) => v,
                    None => return false,
                };
                deep_primary[i] = dp;
                deep_sym[i] = ds;
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

            let fri_deco = proof.query(q);
            let layers_evaluations_sym = fri_deco.layers_evaluations_sym();
            if fri_deco.layers_auth_paths_len() != num_layers
                || layers_evaluations_sym.len() != num_layers
            {
                return false;
            }
            let fri_roots = proof.fri_layers_merkle_roots();

            let mut fold_ok = true;
            for iter in 0..num_layers {
                let h = h_max - 1 - iter;
                // Inject the tables entering at this height (adds zero if none).
                let inj = combined_at(h, index & 1);
                v += betas_fri[iter].square() * inj;

                let eval_sym = &layers_evaluations_sym[iter];
                fold_ok &= Self::verify_fri_layer_openings(
                    &fri_roots[iter],
                    fri_deco.layer_auth_path(iter),
                    &v,
                    eval_sym,
                    index,
                );

                v = (&v + eval_sym) + point_inv.clone() * &betas_fri[iter + 1] * (&v - eval_sym);
                index >>= 1;
                point_inv = point_inv.square();
            }
            if !fold_ok || v != fri_last_value {
                return false;
            }
        }

        // ===== Bus balance. =====
        if needs_lookup_challenges {
            let mut total = FieldElement::<FieldExtension>::zero();
            for (i, air) in airs.iter().enumerate() {
                if air.has_trace_interaction()
                    && let Some(contribution) = proof.per_table(i).bus_table_contribution()
                {
                    total = total + &contribution;
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
        vm_proof: BatchedMultiProofView<'_, Field, FieldExtension, PI>,
        l2g_proof: StarkProofView<'_, Field, FieldExtension, PI>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
        expected_bus_balance: &FieldElement<FieldExtension>,
    ) -> bool
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
        PI: Clone,
    {
        // (1) Mirror the prover: absorb the L2G main root FIRST (canonical order).
        transcript.append_bytes(l2g_proof.lde_trace_main_merkle_root());

        // (2) VM rounds 1-3 to the round-4 seam.
        let mid = match Self::batched_verify_rounds_1_to_3(vm_refs, vm_proof, transcript) {
            Some(m) => m,
            None => return false,
        };

        // (3) L2G lane on a fork of the post-seam transcript. Single lane -> no idx
        // bytes; absorb the aux root then the bus table_contribution (matches the
        // prover's fork in `multi_prove_batched_epoch`).
        let l2g_public_inputs = match l2g_proof.public_inputs() {
            Some(pi) => pi,
            None => return false,
        };
        let mut l2g_fork = transcript.clone();
        if let Some(aux_root) = l2g_proof.lde_trace_aux_merkle_root() {
            l2g_fork.append_bytes(aux_root);
        }
        if let Some(contribution) = l2g_proof.bus_table_contribution() {
            l2g_fork.append_field_element(&contribution);
        }
        let l2g_ok = Self::verify_rounds_2_to_4(
            l2g_ref,
            l2g_proof,
            &l2g_public_inputs,
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
            && let Some(contribution) = l2g_proof.bus_table_contribution()
        {
            vm_expected = &vm_expected - &contribution;
        }
        let vm_ok = Self::batched_verify_round_4(mid, vm_refs, vm_proof, transcript, &vm_expected);

        l2g_ok && vm_ok
    }
}
