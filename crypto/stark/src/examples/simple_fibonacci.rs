use crate::{
    constraints::{
        boundary::{BoundaryConstraint as OldBoundaryConstraint, BoundaryConstraints},
        simple::{
            BoundaryConstraint, Constraints, TransitionConstraint as SimpleTransitionConstraint,
        },
        transition::TransitionConstraint,
    },
    context::AirContext,
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::{AIR, TransitionEvaluationContext},
};
use math::field::{element::FieldElement, traits::IsFFTField};
use std::marker::PhantomData;

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

impl<F> TransitionConstraint<F, F> for FibConstraint<F>
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

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, F>,
        transition_evaluations: &mut [FieldElement<F>],
    ) {
        let (frame, _periodic_values, _rap_challenges) = match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values,
                rap_challenges,
            }
            | TransitionEvaluationContext::Verifier {
                frame,
                periodic_values,
                rap_challenges,
            } => (frame, periodic_values, rap_challenges),
        };

        let first_step = frame.get_evaluation_step(0);
        let second_step = frame.get_evaluation_step(1);
        let third_step = frame.get_evaluation_step(2);

        let a0 = first_step.get_main_evaluation_element(0, 0);
        let a1 = second_step.get_main_evaluation_element(0, 0);
        let a2 = third_step.get_main_evaluation_element(0, 0);

        let res = a2 - a1 - a0;

        transition_evaluations[self.constraint_idx()] = res;
    }
}

pub struct FibonacciAIR<F>
where
    F: IsFFTField,
{
    context: AirContext,
    trace_length: usize,
    pub_inputs: FibonacciPublicInputs<F>,
    old_constraints: Vec<Box<dyn TransitionConstraint<F, F>>>,
}

#[derive(Clone, Debug)]
pub struct FibonacciPublicInputs<F>
where
    F: IsFFTField,
{
    pub a0: FieldElement<F>,
    pub a1: FieldElement<F>,
}

impl<F> AIR for FibonacciAIR<F>
where
    F: IsFFTField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = F;
    type PublicInputs = FibonacciPublicInputs<Self::Field>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        let old_constraints: Vec<Box<dyn TransitionConstraint<F, F>>> =
            vec![Box::new(FibConstraint::new())];

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 1,
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: old_constraints.len(),
        };

        Self {
            pub_inputs: pub_inputs.clone(),
            context,
            trace_length,
            old_constraints,
        }
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.trace_length()
    }

    fn transition_constraints(&self) -> &Vec<Box<dyn TransitionConstraint<F, F>>> {
        &self.old_constraints
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &[FieldElement<Self::Field>],
    ) -> BoundaryConstraints<Self::Field> {
        let a0 = OldBoundaryConstraint::new_simple_main(0, self.pub_inputs.a0.clone());
        let a1 = OldBoundaryConstraint::new_simple_main(1, self.pub_inputs.a1.clone());

        BoundaryConstraints::from_constraints(vec![a0, a1])
    }

    /// New simplified constraints implementation
    fn constraints(
        &self,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> Constraints<Self::Field, Self::FieldExtension> {
        let a0 = self.pub_inputs.a0.clone();
        let a1 = self.pub_inputs.a1.clone();

        Constraints {
            degree_1: vec![SimpleTransitionConstraint {
                name: "fib_transition",
                evaluate: |frame| {
                    let step0 = frame.get_evaluation_step(0);
                    let step1 = frame.get_evaluation_step(1);
                    let step2 = frame.get_evaluation_step(2);

                    let a0 = step0.get_main_evaluation_element(0, 0);
                    let a1 = step1.get_main_evaluation_element(0, 0);
                    let a2 = step2.get_main_evaluation_element(0, 0);

                    a2 - a1 - a0
                },
                // Same function for verifier (F == E for fibonacci)
                evaluate_ext: |frame| {
                    let step0 = frame.get_evaluation_step(0);
                    let step1 = frame.get_evaluation_step(1);
                    let step2 = frame.get_evaluation_step(2);

                    let a0 = step0.get_main_evaluation_element(0, 0);
                    let a1 = step1.get_main_evaluation_element(0, 0);
                    let a2 = step2.get_main_evaluation_element(0, 0);

                    a2 - a1 - a0
                },
                end_exemptions: 2,
            }],
            degree_2: vec![],
            degree_3: vec![],
            boundary: vec![
                BoundaryConstraint::new_main("init_a0", 0, 0, a0), // col 0, row 0
                BoundaryConstraint::new_main("init_a1", 0, 1, a1), // col 0, row 1
            ],
            use_legacy_ordering: false,
            use_legacy_evaluation: false,
        }
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn trace_layout(&self) -> (usize, usize) {
        (1, 0)
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.pub_inputs
    }
}

pub fn fibonacci_trace<F: IsFFTField>(
    initial_values: [FieldElement<F>; 2],
    trace_length: usize,
) -> TraceTable<F, F> {
    let mut ret: Vec<FieldElement<F>> = vec![];

    ret.push(initial_values[0].clone());
    ret.push(initial_values[1].clone());

    for i in 2..(trace_length) {
        ret.push(ret[i - 1].clone() + ret[i - 2].clone());
    }

    TraceTable::from_columns_main(vec![ret], 1)
}
