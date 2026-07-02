use std::marker::PhantomData;

use crate::{
    constraints::{
        boundary::{BoundaryConstraint, BoundaryConstraints},
        transition::TransitionConstraintEvaluator,
    },
    context::AirContext,
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::{AIR, TransitionEvaluationContext},
};
use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

/// Transition constraint for a single Fibonacci column.
/// Enforces: col[i+2] = col[i+1] + col[i]
#[derive(Clone)]
pub struct FibColumnConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    column_idx: usize,
    constraint_idx: usize,
    phantom_f: PhantomData<F>,
    phantom_e: PhantomData<E>,
}

impl<F, E> FibColumnConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    pub fn new(column_idx: usize, constraint_idx: usize) -> Self {
        Self {
            column_idx,
            constraint_idx,
            phantom_f: PhantomData,
            phantom_e: PhantomData,
        }
    }
}

impl<F, E> TransitionConstraintEvaluator<F, E> for FibColumnConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        2
    }

    fn evaluate_verifier(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _,
                rap_challenges: _,
                ..
            } => {
                let step_0 = frame.get_evaluation_step(0);
                let step_1 = frame.get_evaluation_step(1);
                let step_2 = frame.get_evaluation_step(2);

                // Get the values from the column at each step
                let a0 = step_0.get_main_evaluation_element(0, self.column_idx);
                let a1 = step_1.get_main_evaluation_element(0, self.column_idx);
                let a2 = step_2.get_main_evaluation_element(0, self.column_idx);

                // Constraint: a2 = a1 + a0  =>  a2 - a1 - a0 = 0
                let res = a2 - a1 - a0;

                transition_evaluations[self.constraint_idx] = res.to_extension();
            }
            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _,
                rap_challenges: _,
                ..
            } => {
                let step_0 = frame.get_evaluation_step(0);
                let step_1 = frame.get_evaluation_step(1);
                let step_2 = frame.get_evaluation_step(2);

                // Get the values from the column at each step
                let a0 = step_0.get_main_evaluation_element(0, self.column_idx);
                let a1 = step_1.get_main_evaluation_element(0, self.column_idx);
                let a2 = step_2.get_main_evaluation_element(0, self.column_idx);

                // Constraint: a2 = a1 + a0  =>  a2 - a1 - a0 = 0
                let res = a2 - a1 - a0;

                transition_evaluations[self.constraint_idx] = res;
            }
        }
    }
}

/// Public inputs for the multi-column Fibonacci AIR.
/// Contains the initial values (first two elements) for each column.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "FieldElement<F>: serde::Serialize + serde::de::DeserializeOwned")]
pub struct FibonacciMultiColumnPublicInputs<F: IsFFTField> {
    /// Initial values for each column: (a0, a1) pairs
    pub initial_values: Vec<(FieldElement<F>, FieldElement<F>)>,
}

/// Multi-column Fibonacci AIR.
/// Each column contains an independent Fibonacci sequence.
pub struct FibonacciMultiColumnAIR<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    context: AirContext,
    constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>>,
    num_columns: usize,
}

impl<F, E> AIR for FibonacciMultiColumnAIR<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = E;
    type PublicInputs = FibonacciMultiColumnPublicInputs<Self::Field>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(proof_options: &ProofOptions) -> Self {
        // Default to 2 columns if created via this method
        Self::with_num_columns(proof_options, 2)
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        trace_length
    }

    fn transition_constraints(&self) -> &Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> {
        &self.constraints
    }

    fn boundary_constraints(
        &self,
        pub_inputs: &Self::PublicInputs,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
        _bus_public_inputs: Option<&crate::lookup::BusPublicInputs<Self::FieldExtension>>,
        _trace_length: usize,
    ) -> BoundaryConstraints<Self::FieldExtension> {
        let mut constraints = Vec::new();

        // For each column, add boundary constraints for the first two rows
        for (col_idx, (a0, a1)) in pub_inputs.initial_values.iter().enumerate() {
            // First value (row 0)
            constraints.push(BoundaryConstraint::new_main(
                col_idx,
                0,
                a0.clone().to_extension(),
            ));
            // Second value (row 1)
            constraints.push(BoundaryConstraint::new_main(
                col_idx,
                1,
                a1.clone().to_extension(),
            ));
        }

        BoundaryConstraints::from_constraints(constraints)
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn trace_layout(&self) -> (usize, usize) {
        (self.num_columns, 0)
    }
}

impl<F, E> FibonacciMultiColumnAIR<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    /// Creates a new multi-column Fibonacci AIR with the specified number of columns.
    pub fn with_num_columns(proof_options: &ProofOptions, num_columns: usize) -> Self {
        // Create one constraint per column
        let constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> = (0..num_columns)
            .map(|col_idx| {
                Box::new(FibColumnConstraint::new(col_idx, col_idx))
                    as Box<dyn TransitionConstraintEvaluator<F, E>>
            })
            .collect();

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: num_columns,
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: num_columns,
        };

        Self {
            context,
            constraints,
            num_columns,
        }
    }
}

/// Computes the multi-column Fibonacci trace.
///
/// Each column starts with the provided initial values and computes a Fibonacci sequence.
///
/// # Arguments
/// * `initial_values` - Initial (a0, a1) pairs for each column
/// * `trace_length` - Number of rows in the trace (must be a power of 2)
///
/// # Returns
/// A TraceTable with `num_columns` columns and `trace_length` rows.
pub fn compute_trace<F, E>(
    initial_values: &[(FieldElement<F>, FieldElement<F>)],
    trace_length: usize,
) -> TraceTable<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    let num_columns = initial_values.len();
    let mut columns: Vec<Vec<FieldElement<F>>> = Vec::with_capacity(num_columns);

    for (a0, a1) in initial_values {
        let mut column = Vec::with_capacity(trace_length);
        column.push(a0.clone());
        column.push(a1.clone());

        for i in 2..trace_length {
            let next = column[i - 1].clone() + column[i - 2].clone();
            column.push(next);
        }

        columns.push(column);
    }

    TraceTable::from_columns_main(columns, 1)
}

/// Creates public inputs from initial values.
pub fn create_public_inputs<F: IsFFTField>(
    initial_values: Vec<(FieldElement<F>, FieldElement<F>)>,
) -> FibonacciMultiColumnPublicInputs<F> {
    FibonacciMultiColumnPublicInputs { initial_values }
}
