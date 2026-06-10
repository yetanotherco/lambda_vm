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

pub struct PeriodicConstraint<F: IsFFTField> {
    phantom: PhantomData<F>,
}
impl<F: IsFFTField> PeriodicConstraint<F> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}
impl<F: IsFFTField> Default for PeriodicConstraint<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> TransitionConstraintEvaluator<F, F> for PeriodicConstraint<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        0
    }

    fn end_exemptions(&self) -> usize {
        2
    }

    fn evaluate_verifier(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, F>,
        transition_evaluations: &mut [FieldElement<F>],
    ) {
        let (frame, periodic_values, _rap_challenges) = match evaluation_context {
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

        let first_step = frame.get_evaluation_step(0);
        let second_step = frame.get_evaluation_step(1);
        let third_step = frame.get_evaluation_step(2);

        let a0 = first_step.get_main_evaluation_element(0, 0);
        let a1 = second_step.get_main_evaluation_element(0, 0);
        let a2 = third_step.get_main_evaluation_element(0, 0);

        let s = &periodic_values[0];

        transition_evaluations[self.constraint_idx()] = s * (a2 - a1 - a0);
    }
}

/// A sequence that uses periodic columns. It has two columns
/// - C1: at each step adds the last two values or does
///   nothing depending on C2.
/// - C2: it is a binary column that cycles around [0, 1]
///
///   C1   |   C2
///   1    |   0     Boundary col1 = 1
///   1    |   1     Boundary col1 = 1
///   1    |   0     Does nothing
///   2    |   1     Adds 1 + 1
///   2    |   0     Does nothing
///   4    |   1     Adds 2 + 2
///   4    |   0     ...
///   8    |   1
pub struct SimplePeriodicAIR<F>
where
    F: IsFFTField,
{
    context: AirContext,
    transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, F>>>,
}

#[derive(Clone, Debug)]
pub struct SimplePeriodicPublicInputs<F>
where
    F: IsFFTField,
{
    pub a0: FieldElement<F>,
    pub a1: FieldElement<F>,
}

impl<F> AIR for SimplePeriodicAIR<F>
where
    F: IsFFTField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = F;
    type PublicInputs = SimplePeriodicPublicInputs<Self::Field>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(proof_options: &ProofOptions) -> Self {
        let transition_constraints: Vec<
            Box<dyn TransitionConstraintEvaluator<Self::Field, Self::FieldExtension>>,
        > = vec![Box::new(PeriodicConstraint::new())];

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 1,
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: transition_constraints.len(),
        };

        Self {
            context,
            transition_constraints,
        }
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        trace_length
    }

    fn boundary_constraints(
        &self,
        pub_inputs: &Self::PublicInputs,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
        _bus_public_inputs: Option<&crate::lookup::BusPublicInputs<Self::FieldExtension>>,
        trace_length: usize,
    ) -> BoundaryConstraints<Self::Field> {
        let a0 = BoundaryConstraint::new_simple_main(0, pub_inputs.a0.clone());
        let a1 = BoundaryConstraint::new_simple_main(trace_length - 1, pub_inputs.a1.clone());

        BoundaryConstraints::from_constraints(vec![a0, a1])
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraintEvaluator<Self::Field, Self::FieldExtension>>> {
        &self.transition_constraints
    }

    fn get_periodic_column_values(&self) -> Vec<Vec<FieldElement<Self::Field>>> {
        vec![vec![FieldElement::zero(), FieldElement::one()]]
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn trace_layout(&self) -> (usize, usize) {
        (1, 0)
    }
}

pub fn simple_periodic_trace<F: IsFFTField>(trace_length: usize) -> TraceTable<F, F> {
    let mut ret: Vec<FieldElement<F>> = vec![];

    ret.push(FieldElement::one());
    ret.push(FieldElement::one());
    ret.push(FieldElement::one());

    let mut accum = FieldElement::from(2);
    while ret.len() < trace_length - 1 {
        ret.push(accum.clone());
        ret.push(accum.clone());
        accum = &accum + &accum;
    }
    ret.push(accum);

    TraceTable::from_columns_main(vec![ret], 1)
}
