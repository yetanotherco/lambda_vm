use std::marker::PhantomData;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

use crate::{
    constraints::{
        boundary::{BoundaryConstraint, BoundaryConstraints},
        transition::TransitionConstraint,
    },
    context::AirContext,
    table::TableView,
    trace::TraceTable,
    traits::TransitionEvaluationContext,
};

/// Struct representing and AIR with Lookup. Contains own implementation of boundary constraints and auxiliary trace building
pub struct AirWithLookup<
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    B: BoundaryConstraintBuilder<F, E>,
> {
    context: AirContext,
    trace_length: usize,
    pub_inputs: LookUpPublicInputs<F>,
    step_size: usize,
    trace_layout: (usize, usize),
    transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
    auxiliary_trace_build_data: AuxiliaryTraceBuildData,
    boundary_constraint_builder: PhantomData<B>,
}

impl<
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    B: BoundaryConstraintBuilder<F, E>,
> AirWithLookup<F, E, B>
{
    /// Creates a new AirWithLookup adding LookUp-specific transition constraints to existing constraints
    pub fn create(
        auxiliary_trace_build_data: AuxiliaryTraceBuildData,
        pub_inputs: LookUpPublicInputs<F>,
        mut context: AirContext,
        trace_length: usize,
        step_size: usize,
        trace_layout: (usize, usize),
        mut transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
    ) -> Self {
        // Add a transition constraint for each auxiliary column representing a table interaction
        for (i, interaction) in auxiliary_trace_build_data.interactions.iter().enumerate() {
            let constraint = LookupTransitionConstraint::new(
                interaction.clone(),
                i,
                transition_constraints.len(),
            );
            transition_constraints.push(Box::new(constraint));
        }
        // Add a transition constraint for the grand sum auxiliary constraint (sum of all previous aux columns)
        let grand_sum_constraint = LookupGrandSumTransitionConstraint::new(
            transition_constraints.len(),
            auxiliary_trace_build_data.interactions.len(),
        );
        transition_constraints.push(Box::new(grand_sum_constraint));
        // Update context
        context.num_transition_constraints = transition_constraints.len();
        Self {
            context,
            trace_length,
            pub_inputs,
            step_size,
            trace_layout,
            transition_constraints,
            auxiliary_trace_build_data,
            boundary_constraint_builder: PhantomData::<B>,
        }
    }
}

impl<F, E, B> crate::traits::AIR for AirWithLookup<F, E, B>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    B: BoundaryConstraintBuilder<F, E>,
{
    type Field = F;

    type FieldExtension = E;

    type PublicInputs = LookUpPublicInputs<F>;

    fn step_size(&self) -> usize {
        self.step_size
    }

    fn new(
        _trace_length: usize,
        _pub_inputs: &Self::PublicInputs,
        _proof_options: &crate::proof::options::ProofOptions,
    ) -> Self
    where
        Self: Sized,
    {
        // Each individual Air should implement their own constructor
        // ie: instead of using BitFlagAir::new we should use new_bitflag_air() -> Air
        unreachable!("THIS SHOULD NO LONGER BE USED")
    }

    fn trace_layout(&self) -> (usize, usize) {
        self.trace_layout
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.trace_length() * 2
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.pub_inputs
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> {
        &self.transition_constraints
    }

    fn build_auxiliary_trace(&self, trace: &mut TraceTable<F, E>, challenges: &[FieldElement<E>]) {
        // Build an auxiliary column for each table interaction
        // (FIXME) Uses the same challenges for all auxiliary columns (We should use the same challenges for auxiliary columns used for lookups between table pairs, the first solution was to use the same rap challenges for all auxilary columns across tables, we need to checkk if this is safe)
        for (aux_column_idx, aux_column_build_data) in self
            .auxiliary_trace_build_data
            .interactions
            .iter()
            .enumerate()
        {
            build_auxiliary_trace_column(aux_column_idx, aux_column_build_data, trace, challenges);
        }
        // Build grand sum auxiliary column
        let grand_sum_aux_idx = self.auxiliary_trace_build_data.interactions.len();
        for i in 0..trace.num_rows() {
            let grand_sum = trace.columns_aux().iter().map(|col| col[i].clone()).sum();
            trace.set_aux(i, grand_sum_aux_idx, grand_sum)
        }
    }

    // TODO: remove from trait and sample them in prove
    fn build_rap_challenges(
        &self,
        _transcript: &mut dyn IsStarkTranscript<E, F>,
    ) -> Vec<FieldElement<E>> {
        // TODO: rap challenges should be built beforehand for each interaction pair, not built here
        // Toy values used for intial testing
        vec![FieldElement::one(), FieldElement::one()]
    }
    fn boundary_constraints(&self, rap_challenges: &[FieldElement<E>]) -> BoundaryConstraints<E> {
        let mut boundary_constraints = vec![];
        for (pub_inputs, aux_column_build_data) in self
            .pub_inputs
            .columns
            .iter()
            .zip(&self.auxiliary_trace_build_data.interactions)
        {
            boundary_constraints
                .extend(build_boundary_constraint(pub_inputs, aux_column_build_data));
        }
        boundary_constraints.extend(B::boundary_constraints(&self.pub_inputs, rap_challenges));
        BoundaryConstraints::from_constraints(boundary_constraints)
    }
}

/// Struct representing how each lookup air should build its auxiliary column
/// The auxiliary column is built from data used by each table interaction
pub struct AuxiliaryTraceBuildData {
    pub interactions: Vec<TableInteraction>,
}

/// Struct representing how to build a given auxiliary column
#[derive(Clone)]
pub struct TableInteraction {
    pub flag_columns: Vec<usize>,
    pub value_columns: Vec<usize>,
}

/// Public inputs related to each lookup aux column
pub struct LookUpPublicInputs<F>
where
    F: IsFFTField + Send + Sync,
{
    pub columns: Vec<LookupPublicInputsPerInteraction<F>>,
}

// TODO: check this
pub struct LookupPublicInputsPerInteraction<F>
where
    F: IsFFTField + Send + Sync,
{
    pub flags: Vec<FieldElement<F>>,
    pub values: Vec<FieldElement<F>>,
}

pub trait BoundaryConstraintBuilder<
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
>: Send + Sync
{
    fn boundary_constraints(
        _pub_inputs: &LookUpPublicInputs<F>,
        _rap_challenges: &[FieldElement<E>],
    ) -> Vec<BoundaryConstraint<E>> {
        vec![]
    }
}

pub struct NullBoundaryConstraintBuilder {}
impl<F, E> BoundaryConstraintBuilder<F, E> for NullBoundaryConstraintBuilder
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
}

/// Helper method to build a single auxiliary trace column for lookups
fn build_auxiliary_trace_column<F, E>(
    aux_column_idx: usize,
    table_interaction: &TableInteraction,
    trace: &mut TraceTable<F, E>,
    challenges: &[FieldElement<E>],
) where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    // Main table
    let main_segment_cols = trace.columns_main();
    let values = table_interaction
        .value_columns
        .iter()
        .map(|i| &main_segment_cols[*i])
        .collect::<Vec<_>>();
    let flags = table_interaction
        .flag_columns
        .iter()
        .map(|i| &main_segment_cols[*i])
        .collect::<Vec<_>>();

    // Challenges
    let z = challenges[0].clone();
    let alpha = &challenges[1];
    // Coefficients for each value column
    let coeffs: Vec<FieldElement<E>> = (0..values.len()).map(|i| alpha.pow(i)).collect();

    let trace_len = trace.num_rows();
    let mut aux_col: Vec<FieldElement<E>> = Vec::new();

    // fingerprint = z - (v[0] * alpha^0 + v[1] * alpha^1 +...+ value[n] * alpha^n)
    // Where v are the values for each row and n the number of value columns
    // We calculate the first fingerprint separately using the values from the first row
    let fingerprint_inv: FieldElement<E> = (-(values
        .iter()
        .zip(coeffs.iter())
        .map(|(v, coeff)| v[0].clone() * coeff.clone())
        .sum::<FieldElement<E>>())
        + z.clone())
    .inv()
    .unwrap();
    // Sum of all flags
    let flag: FieldElement<F> = flags.iter().map(|flag_column| flag_column[0].clone()).sum();
    // Fill first aux column row (should be overwritten next)
    aux_col.push(flag * fingerprint_inv.clone());

    for i in 0..trace_len - 1 {
        // fingerprint = z - (v[0] * alpha^0 + v[1] * alpha^1 +...+ value[n] * alpha^n)
        // Where v are the values for each row and n the number of value columns
        let fingerprint_inv: FieldElement<E> = (-(values
            .iter()
            .zip(coeffs.iter())
            .map(|(v, coeff)| v[i + 1].clone() * coeff)
            .sum::<FieldElement<E>>())
            + z.clone())
        .inv()
        .unwrap();
        // Sum of all flags
        let flag: FieldElement<F> = flags
            .iter()
            .map(|flag_column| flag_column[i + 1].clone())
            .sum();
        // Fill the auxiliary column row
        aux_col.push(&aux_col[i] + flag * fingerprint_inv);
    }

    for (i, aux_elem) in aux_col.iter().enumerate().take(trace.num_rows()) {
        trace.set_aux(i, aux_column_idx, aux_elem.clone())
    }
}

fn build_boundary_constraint<'a, F, E>(
    pub_inputs: &LookupPublicInputsPerInteraction<F>,
    table_interaction: &TableInteraction,
) -> Vec<BoundaryConstraint<E>>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    // Add constraints for starting value of each flag & value column
    table_interaction
        .flag_columns
        .iter()
        .zip(pub_inputs.flags.iter())
        .chain(
            table_interaction
                .value_columns
                .iter()
                .zip(pub_inputs.values.iter()),
        )
        .map(|(column, starting_value)| {
            BoundaryConstraint::new_main(*column, 0, starting_value.clone().to_extension())
        })
        .collect()
}

// Constraint for each auxiliary column representing a table interaction
struct LookupTransitionConstraint {
    // Indicates columns with flags and values used to build the auxiliary column
    interaction: TableInteraction,
    // Index of the auxiliary column
    interaction_number: usize,
    // Index of the constraint
    constraint_idx: usize,
}

impl LookupTransitionConstraint {
    pub fn new(
        interaction: TableInteraction,
        interaction_number: usize,
        constraint_idx: usize,
    ) -> Self {
        Self {
            interaction,
            interaction_number,
            constraint_idx,
        }
    }
}

impl<F, E> TransitionConstraint<F, E> for LookupTransitionConstraint
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        1
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _periodic_values,
                rap_challenges,
            } => {
                let first_step = frame.get_evaluation_step(0);
                let second_step = frame.get_evaluation_step(1);

                // Auxiliary frame elements
                let s0 = first_step.get_aux_evaluation_element(0, self.interaction_number);
                let s1 = second_step.get_aux_evaluation_element(0, self.interaction_number);

                let z = &rap_challenges[0];
                let alpha = &rap_challenges[1];

                // Main frame elements
                let flag: FieldElement<F> = self
                    .interaction
                    .flag_columns
                    .iter()
                    .map(|c| second_step.get_main_evaluation_element(0, *c).clone())
                    .sum();
                let values = self
                    .interaction
                    .value_columns
                    .iter()
                    .map(|c| second_step.get_main_evaluation_element(0, *c))
                    .collect::<Vec<_>>();

                // Coefficients for each value column
                let coeffs: Vec<FieldElement<E>> =
                    (0..values.len()).map(|i| alpha.pow(i)).collect();

                // fingerprint = z - (v[0] * alpha^0 + v[1] * alpha^1 +...+ value[n] * alpha^n)
                // Where v are the values for each row and n the number of value columns
                let fingerprint: FieldElement<E> = (-values
                    .iter()
                    .zip(coeffs.iter())
                    .map(|(v, coeff)| *v * coeff)
                    .sum::<FieldElement<E>>())
                    + z.clone();

                // We are using the following LogUp equation:
                // s1 = s0 + flag / fingerprint
                // 0 =  s0 * fingerprint + flag - s1 * fingerprint
                // Since constraints must be expressed without division, we multiply each term by sorted_term * unsorted_term:
                let res = flag + s0 * &fingerprint - s1 * fingerprint;

                // The eval always exists, except if the constraint idx were incorrectly defined.
                if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx) {
                    *eval = res;
                }
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _periodic_values,
                rap_challenges,
            } => {
                let first_step = frame.get_evaluation_step(0);
                let second_step = frame.get_evaluation_step(1);

                // Auxiliary frame elements
                let s0 = first_step.get_aux_evaluation_element(0, self.interaction_number);
                let s1 = second_step.get_aux_evaluation_element(0, self.interaction_number);

                let z = &rap_challenges[0];
                let alpha = &rap_challenges[1];

                // Main frame elements
                let flag: FieldElement<E> = self
                    .interaction
                    .flag_columns
                    .iter()
                    .map(|c| second_step.get_main_evaluation_element(0, *c).clone())
                    .sum();
                let values = self
                    .interaction
                    .value_columns
                    .iter()
                    .map(|c| second_step.get_main_evaluation_element(0, *c))
                    .collect::<Vec<_>>();

                // Coefficients for each value column
                let coeffs: Vec<FieldElement<E>> =
                    (0..values.len()).map(|i| alpha.pow(i)).collect();

                // fingerprint = z - (v[0] * alpha^0 + v[1] * alpha^1 +...+ value[n] * alpha^n)
                // Where v are the values for each row and n the number of value columns
                let fingerprint: FieldElement<E> = (-values
                    .iter()
                    .zip(coeffs.iter())
                    .map(|(v, coeff)| *v * coeff)
                    .sum::<FieldElement<E>>())
                    + z.clone();

                // We are using the following LogUp equation:
                // s1 = s0 + flag / fingerprint
                // 0 =  s0 * fingerprint + flag - s1 * fingerprint
                // Since constraints must be expressed without division, we multiply each term by sorted_term * unsorted_term:
                let res = flag + s0 * &fingerprint - s1 * fingerprint;

                // The eval always exists, except if the constraint idx were incorrectly defined.
                if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx) {
                    *eval = res;
                }
            }
        }
    }
}

/// Constraint for the last auxiliary column
struct LookupGrandSumTransitionConstraint {
    // Index of the constraint
    constraint_idx: usize,
    // Amount of interactions -> we could infer this from the amount of aux columns
    interaction_amount: usize,
}

impl LookupGrandSumTransitionConstraint {
    pub fn new(constraint_idx: usize, interaction_amount: usize) -> Self {
        Self {
            constraint_idx,
            interaction_amount,
        }
    }
}

impl<F, E> TransitionConstraint<F, E> for LookupGrandSumTransitionConstraint
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        1
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        fn evaluate_grand_sum_constraint<'a, A: IsSubFieldOf<B>, B: IsField>(
            step: &TableView<'a, A, B>,
            aux_column_idx: usize,
        ) -> FieldElement<B> {
            // Auxiliary frame elements
            let grand_sum = step.get_aux_evaluation_element(0, aux_column_idx);

            let interaction_values_sum: FieldElement<B> = (0..aux_column_idx)
                .map(|i| step.get_aux_evaluation_element(0, i).clone())
                .sum();

            // Check that the grand sum is equal to the sum of all other auxiliary columns in the same row
            // Aka that we correctly built the grand sum auxiliary column
            grand_sum - interaction_values_sum
        }
        let res = match evaluation_context {
            TransitionEvaluationContext::Prover { frame, .. } => {
                evaluate_grand_sum_constraint(frame.get_evaluation_step(0), self.interaction_amount)
            }

            TransitionEvaluationContext::Verifier { frame, .. } => {
                evaluate_grand_sum_constraint(frame.get_evaluation_step(0), self.interaction_amount)
            }
        };
        // The eval always exists, except if the constraint idx were incorrectly defined.
        if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx) {
            *eval = res;
        }
    }
}
