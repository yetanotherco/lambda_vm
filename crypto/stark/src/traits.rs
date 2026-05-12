use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use hashbrown::HashMap;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
    polynomial::Polynomial,
};

use crate::{
    constraints::transition::TransitionConstraintEvaluator,
    domain::Domain,
    lookup::{BusPublicInputs, PackingShifts},
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
        packing_shifts: &'a PackingShifts<F>,
    },
    Verifier {
        frame: &'a Frame<E, E>,
        periodic_values: &'a [FieldElement<E>],
        rap_challenges: &'a [FieldElement<E>],
        logup_alpha_powers: &'a [FieldElement<E>],
        logup_table_offset: &'a FieldElement<E>,
        packing_shifts: &'a PackingShifts<E>,
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
        packing_shifts: &'a PackingShifts<F>,
    ) -> Self {
        Self::Prover {
            frame,
            periodic_values,
            rap_challenges,
            logup_alpha_powers,
            logup_table_offset,
            packing_shifts,
        }
    }

    pub fn new_verifier(
        frame: &'a Frame<E, E>,
        periodic_values: &'a [FieldElement<E>],
        rap_challenges: &'a [FieldElement<E>],
        logup_alpha_powers: &'a [FieldElement<E>],
        logup_table_offset: &'a FieldElement<E>,
        packing_shifts: &'a PackingShifts<E>,
    ) -> Self {
        Self::Verifier {
            frame,
            periodic_values,
            rap_challenges,
            logup_alpha_powers,
            logup_table_offset,
            packing_shifts,
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
            .for_each(|c| c.evaluate_verifier(evaluation_context, &mut evaluations));

        evaluations
    }

    /// Number of constraints that evaluate in the base field F.
    ///
    /// These constraints use the cheaper F×E accumulation path (3 base-field muls
    /// per term) instead of E×E (9 muls). Domain constraints (ALU, memory, PC, etc.)
    /// produce base-field values; only LogUp constraints need extension arithmetic.
    ///
    /// The first `num_base_transition_constraints()` entries in the constraint list
    /// must be base-field constraints. Default is 0 (all E×E, no optimization).
    fn num_base_transition_constraints(&self) -> usize {
        0
    }

    /// Prover-optimized evaluation that writes base-field constraints to `base_evals`
    /// and extension-field constraints to `ext_evals[num_base..]`.
    ///
    /// `base_evals` has length `num_base_transition_constraints()`.
    /// `ext_evals` has length `num_transition_constraints()`; only indices
    /// `[num_base..]` are written/read for extension constraints.
    fn compute_transition_prover(
        &self,
        evaluation_context: &TransitionEvaluationContext<Self::Field, Self::FieldExtension>,
        base_evals: &mut [FieldElement<Self::Field>],
        ext_evals: &mut [FieldElement<Self::FieldExtension>],
    ) {
        for e in base_evals.iter_mut() {
            *e = FieldElement::zero();
        }
        let num_base = base_evals.len();
        for e in ext_evals[num_base..].iter_mut() {
            *e = FieldElement::zero();
        }
        self.transition_constraints()
            .iter()
            .for_each(|c| c.evaluate_prover(evaluation_context, base_evals, ext_evals));
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

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraintEvaluator<Self::Field, Self::FieldExtension>>>;

    /// Compute zerofier evaluations as deduplicated groups with index mapping.
    ///
    /// Each unique zerofier (keyed by period/offset/exemption parameters) is
    /// computed once and constraints map to group indices, avoiding the
    /// per-constraint Vec clone that an unindexed layout would require.
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
