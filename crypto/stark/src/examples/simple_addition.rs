//! A minimal AIR with a simple addition constraint: col0 + col1 = col2
//! This is used to test STARK proving/verification with small traces (1-2 rows).

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
use math::field::{element::FieldElement, traits::IsFFTField};

/// Transition constraint: col0 + col1 = col2
/// This constraint is applied at every row (end_exemptions = 0).
#[derive(Clone)]
struct AdditionConstraint<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField> AdditionConstraint<F> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> TransitionConstraintEvaluator<F, F> for AdditionConstraint<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        0
    }

    fn evaluate_verifier(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, F>,
        transition_evaluations: &mut [FieldElement<F>],
    ) {
        let (frame, _periodic_values, _rap_challenges) = match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values,
                rap_challenges,
                ..
            }
            | TransitionEvaluationContext::Verifier {
                frame,
                periodic_values,
                rap_challenges,
                ..
            } => (frame, periodic_values, rap_challenges),
        };

        let current_step = frame.get_evaluation_step(0);

        let col0 = current_step.get_main_evaluation_element(0, 0);
        let col1 = current_step.get_main_evaluation_element(0, 1);
        let col2 = current_step.get_main_evaluation_element(0, 2);

        // Constraint: col0 + col1 - col2 = 0
        let res = col0 + col1 - col2;

        transition_evaluations[self.constraint_idx()] = res;
    }
}

pub struct SimpleAdditionAIR<F>
where
    F: IsFFTField,
{
    context: AirContext,
    constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, F>>>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "FieldElement<F>: serde::Serialize + serde::de::DeserializeOwned")]
pub struct SimpleAdditionPublicInputs<F>
where
    F: IsFFTField,
{
    /// First value (col0 at row 0)
    pub a: FieldElement<F>,
    /// Second value (col1 at row 0)
    pub b: FieldElement<F>,
}

impl<F> AIR for SimpleAdditionAIR<F>
where
    F: IsFFTField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = F;
    type PublicInputs = SimpleAdditionPublicInputs<Self::Field>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(proof_options: &ProofOptions) -> Self {
        let constraints: Vec<
            Box<dyn TransitionConstraintEvaluator<Self::Field, Self::FieldExtension>>,
        > = vec![Box::new(AdditionConstraint::new())];

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 3,            // col0, col1, col2
            transition_offsets: vec![0], // Only need current step
            num_transition_constraints: constraints.len(),
        };

        Self {
            context,
            constraints,
        }
    }

    fn boundary_constraints(
        &self,
        pub_inputs: &Self::PublicInputs,
        _rap_challenges: &[FieldElement<Self::Field>],
        _bus_public_inputs: Option<&crate::lookup::BusPublicInputs<Self::FieldExtension>>,
        _trace_length: usize,
    ) -> BoundaryConstraints<Self::Field> {
        // Boundary constraints: col0[0] = a, col1[0] = b
        // new_main(col, step, value)
        let a0 = BoundaryConstraint::new_main(0, 0, pub_inputs.a.clone()); // col0 at step 0
        let a1 = BoundaryConstraint::new_main(1, 0, pub_inputs.b.clone()); // col1 at step 0

        BoundaryConstraints::from_constraints(vec![a0, a1])
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraintEvaluator<Self::Field, Self::FieldExtension>>> {
        &self.constraints
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        // Degree 1 constraint
        trace_length
    }

    fn trace_layout(&self) -> (usize, usize) {
        (3, 0) // 3 main columns, 0 aux columns
    }
}

/// Creates a trace table with `num_rows` rows where each row satisfies col0 + col1 = col2.
/// The values are: row i has col0=i+1, col1=i+2, col2=2i+3
pub fn simple_addition_trace<F: IsFFTField>(num_rows: usize) -> TraceTable<F, F> {
    let mut col0 = Vec::with_capacity(num_rows);
    let mut col1 = Vec::with_capacity(num_rows);
    let mut col2 = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let a = FieldElement::<F>::from(i as u64 + 1);
        let b = FieldElement::<F>::from(i as u64 + 2);
        let c = &a + &b;

        col0.push(a);
        col1.push(b);
        col2.push(c);
    }

    TraceTable::from_columns_main(vec![col0, col1, col2], 1)
}
