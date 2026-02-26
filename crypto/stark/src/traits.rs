use std::collections::HashMap;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::{
    field::{
        element::FieldElement,
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

/// Custom split evaluator for enum-dispatch constraint evaluation.
///
/// Tables provide a concrete implementation that iterates over a per-table constraint
/// enum (with `match` dispatch → jump table) instead of `dyn TransitionConstraint`
/// (vtable dispatch → indirect call + potential cache miss).
///
/// The evaluator handles only the table-specific (base-field) constraints.
/// Lookup constraints (extension-field) are handled separately by the caller.
pub trait SplitEvaluator<F, E>: Send + Sync
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    /// Evaluate all table-specific constraints, writing base-field results to
    /// `base_evaluations` and extension-field results to `ext_evaluations`.
    ///
    /// Buffers are pre-zeroed by the caller.
    fn evaluate_split(
        &self,
        frame: &Frame<F, E>,
        periodic_values: &[FieldElement<F>],
        rap_challenges: &[FieldElement<E>],
        logup_alpha_powers: &[FieldElement<E>],
        base_evaluations: &mut [FieldElement<F>],
        ext_evaluations: &mut [FieldElement<E>],
    );
}

/// Generic split evaluator for tables where all constraints compute in base field.
///
/// Wraps a `Vec<C>` where `C: TransitionConstraint<F, E>` and iterates with
/// static dispatch. Works for any table-specific constraint type (concrete struct
/// or enum with match dispatch).
pub struct BaseSplitEvaluator<F, E, C>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    C: TransitionConstraint<F, E>,
{
    constraints: Vec<C>,
    _phantom: std::marker::PhantomData<(F, E)>,
}

impl<F, E, C> BaseSplitEvaluator<F, E, C>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    C: TransitionConstraint<F, E>,
{
    pub fn new(constraints: Vec<C>) -> Self {
        Self {
            constraints,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F, E, C> SplitEvaluator<F, E> for BaseSplitEvaluator<F, E, C>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    C: TransitionConstraint<F, E> + Send + Sync,
{
    fn evaluate_split(
        &self,
        frame: &Frame<F, E>,
        periodic_values: &[FieldElement<F>],
        _rap_challenges: &[FieldElement<E>],
        _logup_alpha_powers: &[FieldElement<E>],
        base_evaluations: &mut [FieldElement<F>],
        _ext_evaluations: &mut [FieldElement<E>],
    ) {
        for c in &self.constraints {
            c.evaluate_prover_base(frame, periodic_values, base_evaluations);
        }
    }
}

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

    /// Evaluate all transition constraints for the verifier (all values in extension field).
    fn compute_transition_verifier(
        &self,
        frame: &Frame<Self::FieldExtension, Self::FieldExtension>,
        periodic_values: &[FieldElement<Self::FieldExtension>],
        rap_challenges: &[FieldElement<Self::FieldExtension>],
        logup_alpha_powers: &[FieldElement<Self::FieldExtension>],
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        let mut evaluations =
            vec![FieldElement::<Self::FieldExtension>::zero(); self.num_transition_constraints()];
        self.transition_constraints().iter().for_each(|c| {
            c.evaluate_verifier(
                frame,
                periodic_values,
                rap_challenges,
                logup_alpha_powers,
                &mut evaluations,
            )
        });
        evaluations
    }

    /// Evaluate all transition constraints for the prover into a caller-provided buffer.
    ///
    /// Reuses a pre-allocated buffer, avoiding a `Vec` allocation per LDE domain point
    /// in the prover's hot loop.
    fn compute_transition_prover_into(
        &self,
        frame: &Frame<Self::Field, Self::FieldExtension>,
        periodic_values: &[FieldElement<Self::Field>],
        rap_challenges: &[FieldElement<Self::FieldExtension>],
        logup_alpha_powers: &[FieldElement<Self::FieldExtension>],
        evaluations: &mut [FieldElement<Self::FieldExtension>],
    ) {
        for e in evaluations.iter_mut() {
            *e = FieldElement::zero();
        }
        self.transition_constraints().iter().for_each(|c| {
            c.evaluate_prover(
                frame,
                periodic_values,
                rap_challenges,
                logup_alpha_powers,
                evaluations,
            )
        });
    }

    /// Evaluate transition constraints split into base-field and extension-field buffers.
    ///
    /// Base-field constraints write to `base_evaluations` (in F), extension-field constraints
    /// write to `ext_evaluations` (in E). The caller accumulates base-field results using
    /// F×E arithmetic (3 base muls) instead of E×E (9 base muls).
    fn compute_transition_prover_split(
        &self,
        frame: &Frame<Self::Field, Self::FieldExtension>,
        periodic_values: &[FieldElement<Self::Field>],
        rap_challenges: &[FieldElement<Self::FieldExtension>],
        logup_alpha_powers: &[FieldElement<Self::FieldExtension>],
        base_evaluations: &mut [FieldElement<Self::Field>],
        ext_evaluations: &mut [FieldElement<Self::FieldExtension>],
    ) {
        for e in base_evaluations.iter_mut() {
            *e = FieldElement::zero();
        }
        for e in ext_evaluations.iter_mut() {
            *e = FieldElement::zero();
        }
        for c in self.transition_constraints().iter() {
            if c.computes_in_base_field() {
                c.evaluate_prover_base(frame, periodic_values, base_evaluations);
            } else {
                c.evaluate_prover(
                    frame,
                    periodic_values,
                    rap_challenges,
                    logup_alpha_powers,
                    ext_evaluations,
                );
            }
        }
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
