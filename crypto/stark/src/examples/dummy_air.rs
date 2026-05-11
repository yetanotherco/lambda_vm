use core::marker::PhantomData;

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
use math::field::{element::FieldElement, goldilocks::GoldilocksField, traits::IsFFTField};

type StarkField = GoldilocksField;

#[derive(Clone)]
struct FibConstraint<F: IsFFTField> {
    phantom: PhantomData<F>,
}
impl<F: IsFFTField> FibConstraint<F> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> TransitionConstraintEvaluator<F, F> for FibConstraint<F>
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

        let first_step = frame.get_evaluation_step(0);
        let second_step = frame.get_evaluation_step(1);
        let third_step = frame.get_evaluation_step(2);

        let a0 = first_step.get_main_evaluation_element(0, 1);
        let a1 = second_step.get_main_evaluation_element(0, 1);
        let a2 = third_step.get_main_evaluation_element(0, 1);

        let res = a2 - a1 - a0;

        transition_evaluations[self.constraint_idx()] = res;
    }
}

#[derive(Clone)]
struct BitConstraint<F: IsFFTField> {
    phantom: PhantomData<F>,
}
impl<F: IsFFTField> BitConstraint<F> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> TransitionConstraintEvaluator<F, F> for BitConstraint<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        1
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

        let first_step = frame.get_evaluation_step(0);

        let bit = first_step.get_main_evaluation_element(0, 0);

        let res = bit * (bit - FieldElement::<F>::one());

        transition_evaluations[self.constraint_idx()] = res;
    }
}

pub struct DummyAIR {
    context: AirContext,
    transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<StarkField, StarkField>>>,
}

impl AIR for DummyAIR {
    type Field = StarkField;
    type FieldExtension = StarkField;
    type PublicInputs = ();

    fn step_size(&self) -> usize {
        1
    }

    fn new(proof_options: &ProofOptions) -> Self {
        let transition_constraints: Vec<
            Box<dyn TransitionConstraintEvaluator<Self::Field, Self::FieldExtension>>,
        > = vec![
            Box::new(FibConstraint::new()),
            Box::new(BitConstraint::new()),
        ];

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 2,
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: 2,
        };

        Self {
            context,
            transition_constraints,
        }
    }

    fn boundary_constraints(
        &self,
        _pub_inputs: &Self::PublicInputs,
        _rap_challenges: &[FieldElement<Self::Field>],
        _bus_public_inputs: Option<&crate::lookup::BusPublicInputs<Self::FieldExtension>>,
        _trace_length: usize,
    ) -> BoundaryConstraints<Self::Field> {
        let a0 = BoundaryConstraint::new_main(1, 0, FieldElement::<Self::Field>::one());
        let a1 = BoundaryConstraint::new_main(1, 1, FieldElement::<Self::Field>::one());

        BoundaryConstraints::from_constraints(vec![a0, a1])
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraintEvaluator<Self::Field, Self::FieldExtension>>> {
        &self.transition_constraints
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        trace_length * 2
    }

    fn trace_layout(&self) -> (usize, usize) {
        (2, 0)
    }
}

pub fn dummy_trace<F: IsFFTField>(trace_length: usize) -> TraceTable<F, F> {
    let mut ret: Vec<FieldElement<F>> = vec![];

    let a0 = FieldElement::one();
    let a1 = FieldElement::one();

    ret.push(a0);
    ret.push(a1);

    for i in 2..(trace_length) {
        ret.push(ret[i - 1].clone() + ret[i - 2].clone());
    }

    TraceTable::from_columns_main(vec![vec![FieldElement::<F>::one(); trace_length], ret], 1)
}
