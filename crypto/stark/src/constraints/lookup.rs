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

/// Struct representing AIR content with the generic L identifying individual Air behaviour
pub struct Air<
    L: AirLogic<F, E>,
    PI,
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
> {
    pub context: AirContext,
    pub trace_length: usize,
    pub pub_inputs: PI,
    pub step_size: usize,
    pub trace_layout: (usize, usize),
    pub transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
    pub logic: L,
}

/// Trait containing inidivial AIR behaviour
pub trait AirLogic<F: IsFFTField + IsSubFieldOf<E> + Send + Sync, E: IsField + Send + Sync> {
    fn build_auxiliary_trace(
        &self,
        _trace: &mut TraceTable<F, E>,
        _challenges: &[FieldElement<E>],
    ) {
    }

    fn build_rap_challenges(
        &self,
        _transcript: &mut dyn IsStarkTranscript<E, F>,
    ) -> Vec<FieldElement<E>> {
        vec![]
    }
    fn boundary_constraints(&self, _rap_challenges: &[FieldElement<E>]) -> BoundaryConstraints<E> {
        BoundaryConstraints::from_constraints(vec![])
    }
}

impl<L, PI, F, E> crate::traits::AIR for Air<L, PI, F, E>
where
    L: AirLogic<F, E> + Send + Sync,
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    PI: Send + Sync,
{
    type Field = F;

    type FieldExtension = E;

    type PublicInputs = PI;

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

    fn boundary_constraints(
        &self,
        rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> BoundaryConstraints<Self::FieldExtension> {
        self.logic.boundary_constraints(rap_challenges)
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

    fn build_auxiliary_trace(
        &self,
        main_trace: &mut TraceTable<Self::Field, Self::FieldExtension>,
        rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) {
        self.logic.build_auxiliary_trace(main_trace, rap_challenges);
    }

    fn build_rap_challenges(
        &self,
        transcript: &mut dyn IsStarkTranscript<Self::FieldExtension, Self::Field>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        self.logic.build_rap_challenges(transcript)
    }
}

impl<L, PI, F, E> Air<L, PI, F, E>
where
    L: AirLogic<F, E> + Send + Sync,
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    /// Transform a regular Air into a LookUpAir by adding the common LookUp behaviour to it.
    /// Logic added:
    /// - Transition constraints: based off of `columns` argument
    /// - Auxiliary trace build logic: based off of `auxiliary_trace_build_data`. This also asumes the AIR has no previous auxiliary trace build logic, this could be changed if needed
    /// - Boundary Constraints: ???
    pub fn into_lookup(
        mut self,
        columns: PermutationColumns, // TODO: Could we infer these columns from the aux build data?
        auxiliary_trace_build_data: AuxiliaryTraceBuildData,
        lookup_public_inputs: LookUpPublicInputs<F>,
    ) -> Air<LookUpAirLogicWrapper<L, F, E>, PI, F, E> {
        // PermutationConstraint being used as placeholder
        self.transition_constraints
            .push(Box::new(PermutationConstraint::<F, E>::new(columns)));
        Air {
            context: self.context,
            trace_length: self.trace_length,
            pub_inputs: self.pub_inputs,
            step_size: self.step_size,
            trace_layout: self.trace_layout,
            transition_constraints: self.transition_constraints,
            logic: LookUpAirLogicWrapper {
                auxiliary_trace_build_data,
                public_inputs: lookup_public_inputs,
                logic: self.logic,
                phantom: PhantomData::<(F, E)>,
            },
        }
    }
}

/// A wrapper for Air logic which extends the received AirLogic with generic LookUp behaviour
pub struct LookUpAirLogicWrapper<L, F, E>
where
    L: AirLogic<F, E> + Send + Sync,
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    auxiliary_trace_build_data: AuxiliaryTraceBuildData,
    public_inputs: LookUpPublicInputs<F>, // TODO: check if this is okay here
    logic: L,
    phantom: PhantomData<(F, E)>,
}

impl<L, F, E> AirLogic<F, E> for LookUpAirLogicWrapper<L, F, E>
where
    L: AirLogic<F, E> + Send + Sync,
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    fn build_auxiliary_trace(&self, trace: &mut TraceTable<F, E>, challenges: &[FieldElement<E>]) {
        // Ignores build_auxiliary_trace logic from wrapped air (assumption: only lookups use auxiliary columns)
        // Uses the same challenges for all auxiliary columns (We should use the same challenges for auxiliary columns used for lookups between table pairs, the first solution was to use the same rap challenges for all auxilary columns across tables, we need to checkk if this is safe)
        for (aux_column_idx, aux_column_build_data) in
            self.auxiliary_trace_build_data.columns.iter().enumerate()
        {
            build_auxiliary_trace_column(aux_column_idx, aux_column_build_data, trace, challenges);
        }
    }

    // TODO: remove from trait and sample them in prove
    fn build_rap_challenges(
        &self,
        transcript: &mut dyn IsStarkTranscript<E, F>,
    ) -> Vec<FieldElement<E>> {
        // Assumption: only lookups use aux trace, only lookups use rap challenges, logic from wrapped air is ignores
        // This method shall be removed and rap challenges shall be sampled only once for all airs in prove methdod after comitting
        vec![]
    }
    fn boundary_constraints(&self, rap_challenges: &[FieldElement<E>]) -> BoundaryConstraints<E> {
        let mut boundary_constraints = vec![];
        for (pub_inputs, aux_column_build_data) in self
            .public_inputs
            .columns
            .iter()
            .zip(&self.auxiliary_trace_build_data.columns)
        {
            boundary_constraints
                .extend(build_boundary_constraint(pub_inputs, aux_column_build_data));
        }
        BoundaryConstraints::from_constraints(boundary_constraints)
    }
}

/// Struct representing how each lookup air should build its auxiliary columns
pub struct AuxiliaryTraceBuildData {
    pub columns: Vec<AuxiliaryColumnBuildData>,
}

/// Struct representing how to build a given auxiliary column
pub struct AuxiliaryColumnBuildData {
    pub flag_columns: Vec<usize>,
    pub value_columns: Vec<usize>,
}

/// Public inputs related to each lookup aux column
pub struct LookUpPublicInputs<F>
where
    F: IsFFTField + Send + Sync,
{
    pub columns: Vec<LookupPublicInputsPerAuxColumn<F>>,
}

// TODO: check this
pub struct LookupPublicInputsPerAuxColumn<F>
where
    F: IsFFTField + Send + Sync,
{
    pub flags: Vec<FieldElement<F>>,
    pub values: Vec<FieldElement<F>>,
}

/// Helper method to build a single auxiliary trace column for lookups
fn build_auxiliary_trace_column<F, E>(
    aux_column_idx: usize,
    aux_column_build_data: &AuxiliaryColumnBuildData,
    trace: &mut TraceTable<F, E>,
    challenges: &[FieldElement<E>],
) where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    // Main table
    let main_segment_cols = trace.columns_main();
    let values = aux_column_build_data
        .value_columns
        .iter()
        .map(|i| &main_segment_cols[*i])
        .collect::<Vec<_>>();
    let a = values[0];
    let v = &main_segment_cols[1];
    let a_sorted = &main_segment_cols[2];
    let v_sorted = &main_segment_cols[3];
    let m = &main_segment_cols[4]; // flag
    // let flags = aux_column_build_data.flag_columns.iter().map(|i| &main_segment_cols[*i]);

    // Challenges
    // TODO: check how to obtain more challenges as needed
    let z = &challenges[0];
    let alpha = &challenges[1];

    let trace_len = trace.num_rows();
    let mut aux_col = Vec::new();

    // TODO: Check what the logic should be for more terms
    // s_0 = m_0/(z - (a'_0 + α * v'_0) - 1/(z - (a_0 + α * v_0)
    let unsorted_term = (-(&a[0] + &v[0] * alpha) + z).inv().unwrap();
    let sorted_term = (-(&a_sorted[0] + &v_sorted[0] * alpha) + z).inv().unwrap();
    aux_col.push(&m[0] * sorted_term - unsorted_term);

    // Apply the same equation given in the permutation transition contraint to the rest of the trace.
    // s_{i+1} = s_i + m_{i+1}/(z - (a'_{i+1} + α * v'_{i+1}) - 1/(z - (a_{i+1} + α * v_{i+1})
    for i in 0..trace_len - 1 {
        let unsorted_term = (-(&a[i + 1] + &v[i + 1] * alpha) + z).inv().unwrap();
        let sorted_term = (-(&a_sorted[i + 1] + &v_sorted[i + 1] * alpha) + z)
            .inv()
            .unwrap();
        aux_col.push(&aux_col[i] + &m[i + 1] * sorted_term - unsorted_term);
    }

    for (i, aux_elem) in aux_col.iter().enumerate().take(trace.num_rows()) {
        trace.set_aux(i, aux_column_idx, aux_elem.clone())
    }
}

fn build_boundary_constraint<'a, F, E>(
    pub_inputs: &LookupPublicInputsPerAuxColumn<F>,
    aux_column_build_data: &AuxiliaryColumnBuildData,
) -> Vec<BoundaryConstraint<E>>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    // Add constraints for starting value of each flag & value column
    aux_column_build_data
        .flag_columns
        .iter()
        .zip(pub_inputs.flags.iter())
        .chain(
            aux_column_build_data
                .value_columns
                .iter()
                .zip(pub_inputs.values.iter()),
        )
        .map(|(column, starting_value)| {
            BoundaryConstraint::new_main(*column, 0, starting_value.clone().to_extension())
        })
        .collect()
}

// Placeholder constraint, not the one used for lookups
//
// Transition constraint that ensures that the sorted columns are a permutation of the original ones.
/// We are using the LogUp construction described in:
/// <https://0xpolygonmiden.github.io/miden-vm/design/lookups/logup.html>.
/// See also our post of LogUp argument in blog.lambdaclass.com.
#[derive(Clone)]
pub struct PermutationConstraint<
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
> {
    phantom: PhantomData<(F, E)>,
    columns: PermutationColumns,
}
impl<F, E> PermutationConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    pub fn new(columns: PermutationColumns) -> Self {
        Self {
            columns,
            phantom: PhantomData::<(F, E)>,
        }
    }
}

#[derive(Clone)]
pub struct PermutationColumns {
    pub a: usize,
    pub v: usize,
    pub a_s: usize,
    pub v_s: usize,
    pub m: usize,
}

impl<F, E> TransitionConstraint<F, E> for PermutationConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        3
    }

    fn constraint_idx(&self) -> usize {
        2
    }

    fn end_exemptions(&self) -> usize {
        1
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        // In both evaluation contexts, Prover and Verfier will evaluate the transition polynomial in the same way.
        // The only difference is that the Prover's Frame has base field and field extension elements,
        // while the Verfier's Frame has only field extension elements.
        let res = match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                rap_challenges,
                ..
            } => compute_permutation_constraint(
                frame.get_evaluation_step(0),
                frame.get_evaluation_step(1),
                rap_challenges,
                self.columns.clone(),
            ),
            TransitionEvaluationContext::Verifier {
                frame,
                rap_challenges,
                ..
            } => compute_permutation_constraint(
                frame.get_evaluation_step(0),
                frame.get_evaluation_step(1),
                rap_challenges,
                self.columns.clone(),
            ),
        };

        // The eval always exists, except if the constraint idx were incorrectly defined.
        if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx()) {
            *eval = res;
        }
    }
}

fn compute_permutation_constraint<F, E>(
    first_step: &TableView<'_, F, E>,
    second_step: &TableView<'_, F, E>,
    rap_challenges: &[FieldElement<E>],
    columns: PermutationColumns,
) -> FieldElement<E>
where
    F: IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    // Auxiliary frame elements
    let s0 = first_step.get_aux_evaluation_element(0, 0);
    let s1 = second_step.get_aux_evaluation_element(0, 0);

    // Challenges
    let z = &rap_challenges[0];
    let alpha = &rap_challenges[1];

    // Main frame elements
    let a1 = second_step.get_main_evaluation_element(0, columns.a);
    let v1 = second_step.get_main_evaluation_element(0, columns.v);
    let a_sorted_1 = second_step.get_main_evaluation_element(0, columns.a_s);
    let v_sorted_1 = second_step.get_main_evaluation_element(0, columns.v_s);
    let m = second_step.get_main_evaluation_element(0, columns.m);

    let unsorted_term = -(a1 + v1 * alpha) + z;
    let sorted_term = -(a_sorted_1 + v_sorted_1 * alpha) + z;

    // We are using the following LogUp equation:
    // s1 = s0 + m / sorted_term - 1/unsorted_term.
    // Since constraints must be expressed without division, we multiply each term by sorted_term * unsorted_term:
    s0 * &unsorted_term * &sorted_term + m * &unsorted_term
        - &sorted_term
        - s1 * unsorted_term * sorted_term
}
