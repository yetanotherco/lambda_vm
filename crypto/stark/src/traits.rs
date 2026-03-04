use std::collections::HashMap;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::{
    field::{
        element::FieldElement,
        fields::fft_friendly::{
            extensions_goldilocks::Degree3GoldilocksExtensionField,
            u64_goldilocks::GoldilocksField,
            u64_goldilocks_packed::{fp3::PackedFp3, PackedGoldilocks},
        },
        packed::PackedField,
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
    polynomial::Polynomial,
};

use crate::{
    constraints::transition::TransitionConstraint, domain::Domain, lookup::BusPublicInputs,
};

use super::{
    config::Commitment, constraints::boundary::BoundaryConstraints, context::AirContext,
    frame::Frame, proof::options::ProofOptions, trace::TraceTable,
};

/// Deduplicated zerofier evaluations: unique zerofier vectors indexed by constraint.
///
/// Multiple constraints often share the same zerofier (same period, offset, and exemptions).
/// Instead of cloning a `Vec<FieldElement<F>>` per constraint, this struct stores each unique
/// zerofier vector once and maps each constraint index to its group.
pub struct ZerofierEvaluations<F: IsField> {
    /// Unique zerofier evaluation vectors (deduplicated).
    pub groups: Vec<Vec<FieldElement<F>>>,
    /// constraint_idx → group index.
    pub constraint_to_group: Vec<usize>,
}

impl<F: IsField> ZerofierEvaluations<F> {
    #[inline]
    pub fn get(&self, constraint_idx: usize, lde_idx: usize) -> &FieldElement<F> {
        let group = &self.groups[self.constraint_to_group[constraint_idx]];
        &group[lde_idx % group.len()]
    }

    /// Returns true if all constraints share the same zerofier group.
    pub fn is_uniform(&self) -> bool {
        self.groups.len() == 1
    }

    /// Fast path for uniform case: all constraints share one zerofier.
    #[inline]
    pub fn get_uniform(&self, lde_idx: usize) -> &FieldElement<F> {
        let group = &self.groups[0];
        &group[lde_idx % group.len()]
    }
}

/// Key identifying a unique zerofier shape — constraints with the same key share
/// the same zerofier evaluations on the extended domain.
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct ZerofierGroupKey {
    period: usize,
    offset: usize,
    exemptions_period: Option<usize>,
    periodic_exemptions_offset: Option<usize>,
    end_exemptions: usize,
}

/// This enum is necessary because, while both the prover and verifier perform the same operations
///  to compute transition constraints, their frames differ.
///  The prover uses a frame containing elements from both the base field and its extension
/// (common when working with small fields and challengers in the extension).
/// In contrast, the verifier, lacking access to the trace and relying solely on evaluations at the challengers,
/// works with a frame that contains only elements from the extension.
pub enum TransitionEvaluationContext<'a, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    Prover {
        frame: &'a Frame<F, E>,
        periodic_values: &'a [FieldElement<F>],
        rap_challenges: &'a [FieldElement<E>],
        logup_alpha_powers: &'a [FieldElement<E>],
        logup_table_offset: &'a FieldElement<E>,
    },
    Verifier {
        frame: &'a Frame<E, E>,
        periodic_values: &'a [FieldElement<E>],
        rap_challenges: &'a [FieldElement<E>],
        logup_alpha_powers: &'a [FieldElement<E>],
        logup_table_offset: &'a FieldElement<E>,
    },
}

impl<'a, F, E> TransitionEvaluationContext<'a, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    pub fn new_prover(
        frame: &'a Frame<F, E>,
        periodic_values: &'a [FieldElement<F>],
        rap_challenges: &'a [FieldElement<E>],
        logup_alpha_powers: &'a [FieldElement<E>],
        logup_table_offset: &'a FieldElement<E>,
    ) -> Self {
        Self::Prover {
            frame,
            periodic_values,
            rap_challenges,
            logup_alpha_powers,
            logup_table_offset,
        }
    }

    pub fn new_verifier(
        frame: &'a Frame<E, E>,
        periodic_values: &'a [FieldElement<E>],
        rap_challenges: &'a [FieldElement<E>],
        logup_alpha_powers: &'a [FieldElement<E>],
        logup_table_offset: &'a FieldElement<E>,
    ) -> Self {
        Self::Verifier {
            frame,
            periodic_values,
            rap_challenges,
            logup_alpha_powers,
            logup_table_offset,
        }
    }
}

/// AIR is a representation of the Constraints
pub trait AIR: Send + Sync {
    type Field: IsFFTField + IsSubFieldOf<Self::FieldExtension> + Send + Sync;
    type FieldExtension: IsField + Send + Sync;
    type PublicInputs;

    fn step_size(&self) -> usize;

    /// Human-readable name for this AIR (used in profiling output).
    fn name(&self) -> &str {
        "unknown"
    }

    fn new(proof_options: &ProofOptions) -> Self
    where
        Self: Sized;

    fn build_auxiliary_trace(
        &self,
        _main_trace: &mut TraceTable<Self::Field, Self::FieldExtension>,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> Option<BusPublicInputs<Self::FieldExtension>> {
        None
    }

    fn build_rap_challenges(
        &self,
        _transcript: &mut dyn IsStarkTranscript<Self::FieldExtension, Self::Field>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        Vec::new()
    }

    /// Returns the amount main trace columns and auxiliary trace columns
    fn trace_layout(&self) -> (usize, usize);

    fn has_aux_trace(&self) -> bool {
        let (_main_trace_columns, aux_trace_columns) = self.trace_layout();
        aux_trace_columns != 0
    }

    /// Returns true if this AIR interacts with other traces (lookup), such is the case
    /// of `AirWithBuses` (override to return true).
    /// Generic RAP AIRs with auxiliary columns but no bus interactions must return false.
    fn has_trace_interaction(&self) -> bool {
        false
    }

    /// Returns the maximum number of bus elements across all interactions.
    /// Used to compute the correct number of alpha powers for LogUp fingerprints.
    fn max_bus_elements(&self) -> usize {
        0
    }

    /// Returns true if this AIR has preprocessed (precomputed) columns.
    ///
    /// Preprocessed tables have columns that are fully deterministic and known
    /// to both prover and verifier (e.g., bitwise lookup tables).
    fn is_preprocessed(&self) -> bool {
        false
    }

    /// Returns the number of precomputed columns (columns 0..n are precomputed).
    ///
    /// Only meaningful if `is_preprocessed()` returns true.
    /// The remaining columns (n..) are multiplicities.
    fn num_precomputed_columns(&self) -> usize {
        0
    }

    /// Returns the hardcoded commitment to the precomputed columns.
    ///
    /// Only meaningful if `is_preprocessed()` returns true.
    fn precomputed_commitment(&self) -> Commitment {
        [0u8; 32]
    }

    fn num_auxiliary_rap_columns(&self) -> usize {
        self.trace_layout().1
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize;

    /// The method called by the prover to evaluate the transitions corresponding to an evaluation frame.
    /// In the case of the prover, the main evaluation table of the frame takes values in
    /// `Self::Field`, since they are the evaluations of the main trace at the LDE domain.
    /// In the case of the verifier, the frame take elements of Self::FieldExtension.
    fn compute_transition(
        &self,
        evaluation_context: &TransitionEvaluationContext<Self::Field, Self::FieldExtension>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        let mut evaluations =
            vec![FieldElement::<Self::FieldExtension>::zero(); self.num_transition_constraints()];
        self.transition_constraints()
            .iter()
            .for_each(|c| c.evaluate(evaluation_context, &mut evaluations));

        evaluations
    }

    /// Evaluate all transition constraints into a caller-provided buffer.
    ///
    /// Same as `compute_transition` but reuses a pre-allocated buffer, avoiding
    /// a `Vec` allocation per LDE domain point in the prover's hot loop.
    fn compute_transition_into(
        &self,
        evaluation_context: &TransitionEvaluationContext<Self::Field, Self::FieldExtension>,
        evaluations: &mut [FieldElement<Self::FieldExtension>],
    ) {
        for e in evaluations.iter_mut() {
            *e = FieldElement::zero();
        }
        self.transition_constraints()
            .iter()
            .for_each(|c| c.evaluate(evaluation_context, evaluations));
    }

    fn boundary_constraints(
        &self,
        pub_inputs: &Self::PublicInputs,
        rap_challenges: &[FieldElement<Self::FieldExtension>],
        bus_public_inputs: Option<&BusPublicInputs<Self::FieldExtension>>,
        trace_length: usize,
    ) -> BoundaryConstraints<Self::FieldExtension>;

    fn context(&self) -> &AirContext;

    fn options(&self) -> &ProofOptions {
        &self.context().proof_options
    }

    fn blowup_factor(&self) -> u8 {
        self.options().blowup_factor
    }

    fn coset_offset(&self) -> FieldElement<Self::Field> {
        FieldElement::from(self.options().coset_offset)
    }

    fn trace_primitive_root(&self, trace_length: usize) -> FieldElement<Self::Field> {
        let root_of_unity_order = u64::from(trace_length.trailing_zeros());

        Self::Field::get_primitive_root_of_unity(root_of_unity_order).unwrap()
    }

    fn num_transition_constraints(&self) -> usize {
        self.context().num_transition_constraints
    }

    fn get_periodic_column_values(&self) -> Vec<Vec<FieldElement<Self::Field>>> {
        vec![]
    }

    fn get_periodic_column_polynomials(
        &self,
        trace_length: usize,
    ) -> Vec<Polynomial<FieldElement<Self::Field>>> {
        let mut result = Vec::new();
        for periodic_column in self.get_periodic_column_values() {
            let values: Vec<_> = periodic_column
                .iter()
                .cycle()
                .take(trace_length)
                .cloned()
                .collect();
            let poly =
                Polynomial::<FieldElement<Self::Field>>::interpolate_fft::<Self::Field>(&values)
                    .unwrap();
            result.push(poly);
        }
        result
    }

    /// Evaluate all transition constraints at WIDTH consecutive LDE points using packed SIMD.
    ///
    /// Default implementation: scalar fallback (calls compute_transition_into WIDTH times,
    /// scatters results into packed registers). Override in concrete AIR implementations
    /// to evaluate constraints directly on packed inputs for better performance.
    ///
    /// Only called from the Goldilocks packed path (gated by TypeId check in evaluator),
    /// so all parameters use concrete Goldilocks types.
    ///
    /// # Arguments
    /// * `lde_main_cols` - Column-major main trace LDE: `lde_main_cols[col][row]`
    /// * `lde_aux_cols` - Column-major aux trace LDE: `lde_aux_cols[col][row]`
    /// * `lde_periodic_cols` - Periodic column values on LDE domain
    /// * `base_row` - Starting LDE row index for this WIDTH-wide chunk
    /// * `lde_domain_size` - Total number of LDE domain points
    /// * `offsets` - Evaluation step offsets (typically [0, 1] for step_size=1)
    /// * `rap_challenges` - RAP challenges (z, alpha for LogUp)
    /// * `logup_alpha_powers` - Precomputed alpha powers for LogUp fingerprints
    /// * `logup_table_offset` - L/N offset for circular constraint
    /// * `blowup_factor` - LDE blowup factor
    /// * `lde_step_size` - LDE step size (trace step_size × blowup_factor)
    /// * `transition_buf` - Scratch buffer for scalar constraint evaluations
    /// * `periodic_buf` - Scratch buffer for periodic column values
    /// * `frame` - Pre-allocated frame for scalar constraint evaluation
    /// * `results` - Output buffer: `results[constraint_idx]` receives packed evaluation
    #[allow(clippy::too_many_arguments)]
    fn compute_transitions_packed(
        &self,
        lde_main_cols: &[Vec<FieldElement<GoldilocksField>>],
        lde_aux_cols: &[Vec<FieldElement<Degree3GoldilocksExtensionField>>],
        lde_periodic_cols: &[Vec<FieldElement<GoldilocksField>>],
        base_row: usize,
        lde_domain_size: usize,
        offsets: &[usize],
        rap_challenges: &[FieldElement<Degree3GoldilocksExtensionField>],
        logup_alpha_powers: &[FieldElement<Degree3GoldilocksExtensionField>],
        logup_table_offset: &FieldElement<Degree3GoldilocksExtensionField>,
        blowup_factor: usize,
        lde_step_size: usize,
        transition_buf: &mut [FieldElement<Degree3GoldilocksExtensionField>],
        periodic_buf: &mut [FieldElement<GoldilocksField>],
        frame: &mut Frame<GoldilocksField, Degree3GoldilocksExtensionField>,
        results: &mut [PackedFp3<PackedGoldilocks>],
    ) {
        let _ = (lde_main_cols, lde_aux_cols, blowup_factor);
        let chunk_len = PackedGoldilocks::WIDTH.min(lde_domain_size - base_row);

        // Default: scalar fallback. Fill frame from column-major LDE data,
        // evaluate constraints, scatter into packed lanes.
        for local_j in 0..chunk_len {
            let i = base_row + local_j;

            // Fill frame from column-major LDE
            let rows_per_step = lde_step_size / blowup_factor;
            for (step_idx, &offset) in offsets.iter().enumerate() {
                let initial_step_row = i + offset * lde_step_size;
                let step = &mut frame.steps[step_idx];
                for sub_row in 0..rows_per_step {
                    let step_row_idx =
                        (initial_step_row + sub_row * blowup_factor) % lde_domain_size;
                    for col in 0..lde_main_cols.len() {
                        step.data[sub_row][col] = lde_main_cols[col][step_row_idx].clone();
                    }
                    for col in 0..lde_aux_cols.len() {
                        step.aux_data[sub_row][col] = lde_aux_cols[col][step_row_idx].clone();
                    }
                }
            }

            for (k, col) in lde_periodic_cols.iter().enumerate() {
                periodic_buf[k] = col[i].clone();
            }

            // SAFETY: This method is only called from the Goldilocks packed path,
            // where TypeId checks guarantee Self::Field == GoldilocksField and
            // Self::FieldExtension == Degree3GoldilocksExtensionField.
            // FieldElement is #[repr(transparent)], so pointer casts are safe.
            let ctx = TransitionEvaluationContext::new_prover(
                frame,
                periodic_buf,
                rap_challenges,
                logup_alpha_powers,
                logup_table_offset,
            );
            let ctx_generic: &TransitionEvaluationContext<Self::Field, Self::FieldExtension> =
                unsafe { &*(&ctx as *const _ as *const _) };
            let buf_generic: &mut [FieldElement<Self::FieldExtension>] =
                unsafe { &mut *(transition_buf as *mut _ as *mut _) };
            self.compute_transition_into(ctx_generic, buf_generic);

            for c in 0..transition_buf.len() {
                results[c].set_lane(local_j, &transition_buf[c]);
            }
        }
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>>;

    fn transition_zerofier_evaluations(
        &self,
        domain: &Domain<Self::Field>,
    ) -> Vec<Vec<FieldElement<Self::Field>>> {
        let mut evals = vec![Vec::new(); self.num_transition_constraints()];

        let mut zerofier_groups: HashMap<ZerofierGroupKey, Vec<FieldElement<Self::Field>>> =
            HashMap::new();

        self.transition_constraints().iter().for_each(|c| {
            let period = c.period();
            let offset = c.offset();
            let exemptions_period = c.exemptions_period();
            let periodic_exemptions_offset = c.periodic_exemptions_offset();
            let end_exemptions = c.end_exemptions();

            // This hashmap is used to avoid recomputing with an fft the same zerofier evaluation
            // If there are multiple domain and subdomains it can be further optimized
            // as to share computation between them

            let zerofier_group_key = ZerofierGroupKey {
                period,
                offset,
                exemptions_period,
                periodic_exemptions_offset,
                end_exemptions,
            };
            zerofier_groups
                .entry(zerofier_group_key)
                .or_insert_with(|| c.zerofier_evaluations_on_extended_domain(domain));

            let zerofier_evaluations = zerofier_groups.get(&zerofier_group_key).unwrap();
            evals[c.constraint_idx()] = zerofier_evaluations.clone();
        });

        evals
    }

    /// Compute zerofier evaluations as deduplicated groups with index mapping.
    ///
    /// This replaces `transition_zerofier_evaluations` for the prover's constraint
    /// evaluation loop. Instead of cloning `Vec<FieldElement<F>>` per constraint,
    /// each unique zerofier is computed once and constraints map to group indices.
    fn transition_zerofier_evaluations_grouped(
        &self,
        domain: &Domain<Self::Field>,
    ) -> ZerofierEvaluations<Self::Field> {
        let num_constraints = self.num_transition_constraints();
        let mut constraint_to_group = vec![0usize; num_constraints];
        let mut zerofier_groups_map: HashMap<ZerofierGroupKey, usize> = HashMap::new();
        let mut groups: Vec<Vec<FieldElement<Self::Field>>> = Vec::new();

        self.transition_constraints().iter().for_each(|c| {
            let key = ZerofierGroupKey {
                period: c.period(),
                offset: c.offset(),
                exemptions_period: c.exemptions_period(),
                periodic_exemptions_offset: c.periodic_exemptions_offset(),
                end_exemptions: c.end_exemptions(),
            };
            let group_idx = *zerofier_groups_map.entry(key).or_insert_with(|| {
                let idx = groups.len();
                groups.push(c.zerofier_evaluations_on_extended_domain(domain));
                idx
            });
            constraint_to_group[c.constraint_idx()] = group_idx;
        });

        ZerofierEvaluations {
            groups,
            constraint_to_group,
        }
    }
}
