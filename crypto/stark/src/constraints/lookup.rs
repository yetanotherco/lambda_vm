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
    pub context: AirContext,
    pub trace_length: usize,
    pub pub_inputs: LookUpPublicInputs<F>,
    pub step_size: usize,
    pub trace_layout: (usize, usize),
    pub transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
    pub auxiliary_trace_build_data: AuxiliaryTraceBuildData,
    pub boundary_constraint_builder: PhantomData<B>,
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
        context: AirContext,
        trace_length: usize,
        step_size: usize,
        trace_layout: (usize, usize),
        mut transition_constraints: Vec<Box<dyn TransitionConstraint<F, E>>>,
    ) -> Self {
        // PermutationConstraint being used as placeholder
        transition_constraints.push(Box::new(PermutationConstraint::<F, E>::new(
            // TODO: should be infered from AuxiliaryTraceBuildData
            PermutationColumns {
                a: 0,
                v: 0,
                a_s: 0,
                v_s: 0,
                m: 0,
            },
        )));
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
        // Assumption: only lookups use aux trace, only lookups use rap challenges, logic from wrapped air is ignores
        // This method shall be removed and rap challenges shall be sampled only once for all airs in prove methdod after comitting
        vec![]
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

    // fingerprint = v[0] * alpha^0 + v[1] * alpha^1 +...+ value[n] * alpha^n + z
    // Where v are the values for each row and n the number of value columns
    // We calculate the first fingerprint separately using the values from the first row
    let fingerprint_inv: FieldElement<E> = (values
        .iter()
        .zip(coeffs.iter())
        .map(|(v, coeff)| v[0].clone() * coeff)
        .sum::<FieldElement<E>>()
        + z.clone())
    .inv()
    .unwrap();
    // Sum of all flags
    let flag: FieldElement<F> = flags.iter().map(|flag_column| flag_column[0].clone()).sum();
    // Fill first aux column row (should be overwritten next)
    aux_col.push(flag * fingerprint_inv.clone());

    for i in 0..trace_len - 1 {
        // fingerprint = v[0] * alpha^0 + v[1] * alpha^1 +...+ value[n] * alpha^n + z
        // Where v are the values for each row and n the number of value columns
        let fingerprint_inv: FieldElement<E> = (values
            .iter()
            .zip(coeffs.iter())
            .map(|(v, coeff)| v[i].clone() * coeff)
            .sum::<FieldElement<E>>()
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
