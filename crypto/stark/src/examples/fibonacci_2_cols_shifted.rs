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
use math::{
    field::{element::FieldElement, traits::IsFFTField},
    traits::AsBytes,
};
use std::marker::PhantomData;

#[derive(Clone)]
struct ShiftedFibTransition1<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField> ShiftedFibTransition1<F> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> TransitionConstraint<F, F> for ShiftedFibTransition1<F>
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
        1
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

        let first_row = frame.get_evaluation_step(0);
        let second_row = frame.get_evaluation_step(1);

        let a0_1 = first_row.get_main_evaluation_element(0, 1);
        let a1_0 = second_row.get_main_evaluation_element(0, 0);

        let res = a1_0 - a0_1;

        transition_evaluations[self.constraint_idx()] = res;
    }
}

#[derive(Clone)]
struct ShiftedFibTransition2<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField> ShiftedFibTransition2<F> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> TransitionConstraint<F, F> for ShiftedFibTransition2<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        1
    }

    fn end_exemptions(&self) -> usize {
        1
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

        let first_row = frame.get_evaluation_step(0);
        let second_row = frame.get_evaluation_step(1);

        let a0_0 = first_row.get_main_evaluation_element(0, 0);
        let a0_1 = first_row.get_main_evaluation_element(0, 1);
        let a1_1 = second_row.get_main_evaluation_element(0, 1);

        let res = a1_1 - a0_0 - a0_1;

        transition_evaluations[self.constraint_idx()] = res;
    }
}

#[derive(Clone, Debug)]
pub struct PublicInputs<F>
where
    F: IsFFTField,
{
    pub claimed_value: FieldElement<F>,
    pub claimed_index: usize,
}

impl<F> AsBytes for PublicInputs<F>
where
    F: IsFFTField,
    FieldElement<F>: AsBytes,
{
    fn as_bytes(&self) -> Vec<u8> {
        let mut transcript_init_seed = self.claimed_index.to_be_bytes().to_vec();
        transcript_init_seed.extend_from_slice(&self.claimed_value.as_bytes());
        transcript_init_seed
    }
}

pub struct Fibonacci2ColsShifted<F>
where
    F: IsFFTField,
{
    context: AirContext,
    trace_length: usize,
    pub_inputs: PublicInputs<F>,
    transition_constraints: Vec<Box<dyn TransitionConstraint<F, F>>>,
    stone_compatible: bool,
}

impl<F: IsFFTField> Fibonacci2ColsShifted<F> {
    /// Create an AIR with Stone-compatible constraint evaluation
    pub fn new_stone_compatible(
        trace_length: usize,
        pub_inputs: &PublicInputs<F>,
        proof_options: &ProofOptions,
    ) -> Self
    where
        F: Send + Sync + 'static,
    {
        let mut air = Self::new(trace_length, pub_inputs, proof_options);
        air.stone_compatible = true;
        air
    }
}

/// The AIR for to a 2 column trace, where each column is a Fibonacci sequence and the
/// second column is constrained to be the shift of the first one. That is, if `Col0_i`
/// and `Col1_i` denote the i-th entry of each column, then `Col0_{i+1}` equals `Col1_{i}`
/// for all `i`. Also, `Col0_0` is constrained to be `1`.
impl<F> AIR for Fibonacci2ColsShifted<F>
where
    F: IsFFTField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = F;
    type PublicInputs = PublicInputs<Self::Field>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        let transition_constraints: Vec<
            Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>,
        > = vec![
            Box::new(ShiftedFibTransition1::new()),
            Box::new(ShiftedFibTransition2::new()),
        ];

        let context = AirContext {
            proof_options: proof_options.clone(),
            transition_offsets: vec![0, 1],
            num_transition_constraints: 2,
            trace_columns: 2,
        };

        Self {
            trace_length,
            context,
            pub_inputs: pub_inputs.clone(),
            transition_constraints,
            stone_compatible: false,
        }
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> BoundaryConstraints<Self::Field> {
        let initial_condition = OldBoundaryConstraint::new_main(0, 0, FieldElement::one());
        let claimed_value_constraint = OldBoundaryConstraint::new_main(
            0,
            self.pub_inputs.claimed_index,
            self.pub_inputs.claimed_value.clone(),
        );

        BoundaryConstraints::from_constraints(vec![initial_condition, claimed_value_constraint])
    }

    fn constraints(
        &self,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> Constraints<Self::Field, Self::FieldExtension> {
        let claimed_value = self.pub_inputs.claimed_value.clone();
        let claimed_index = self.pub_inputs.claimed_index;

        Constraints {
            degree_1: vec![
                // Constraint 1: a1[0] = a0[1] (next row's col 0 equals current row's col 1)
                SimpleTransitionConstraint {
                    name: "shifted_fib_1",
                    evaluate: |frame| {
                        let row0 = frame.get_evaluation_step(0);
                        let row1 = frame.get_evaluation_step(1);

                        let a0_1 = row0.get_main_evaluation_element(0, 1);
                        let a1_0 = row1.get_main_evaluation_element(0, 0);

                        a1_0 - a0_1
                    },
                    evaluate_ext: |frame| {
                        let row0 = frame.get_evaluation_step(0);
                        let row1 = frame.get_evaluation_step(1);

                        let a0_1 = row0.get_main_evaluation_element(0, 1);
                        let a1_0 = row1.get_main_evaluation_element(0, 0);

                        a1_0 - a0_1
                    },
                    end_exemptions: 1,
                },
                // Constraint 2: a1[1] = a0[0] + a0[1] (next row's col 1 equals sum of current row)
                SimpleTransitionConstraint {
                    name: "shifted_fib_2",
                    evaluate: |frame| {
                        let row0 = frame.get_evaluation_step(0);
                        let row1 = frame.get_evaluation_step(1);

                        let a0_0 = row0.get_main_evaluation_element(0, 0);
                        let a0_1 = row0.get_main_evaluation_element(0, 1);
                        let a1_1 = row1.get_main_evaluation_element(0, 1);

                        a1_1 - a0_0 - a0_1
                    },
                    evaluate_ext: |frame| {
                        let row0 = frame.get_evaluation_step(0);
                        let row1 = frame.get_evaluation_step(1);

                        let a0_0 = row0.get_main_evaluation_element(0, 0);
                        let a0_1 = row0.get_main_evaluation_element(0, 1);
                        let a1_1 = row1.get_main_evaluation_element(0, 1);

                        a1_1 - a0_0 - a0_1
                    },
                    end_exemptions: 1,
                },
            ],
            degree_2: vec![],
            degree_3: vec![],
            boundary: vec![
                // Initial condition: col 0, row 0 = 1
                BoundaryConstraint::new_main("initial", 0, 0, FieldElement::one()),
                // Claimed value: col 0, row claimed_index = claimed_value
                BoundaryConstraint::new_main("claimed", 0, claimed_index, claimed_value),
            ],
            use_legacy_ordering: self.stone_compatible,
            use_legacy_evaluation: self.stone_compatible,
        }
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>> {
        &self.transition_constraints
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.trace_length()
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn trace_layout(&self) -> (usize, usize) {
        (2, 0)
    }

    fn use_legacy_evaluator(&self) -> bool {
        self.stone_compatible
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.pub_inputs
    }
}

pub fn compute_trace<F: IsFFTField>(
    initial_value: FieldElement<F>,
    trace_length: usize,
) -> TraceTable<F, F> {
    let mut x = FieldElement::one();
    let mut y = initial_value;
    let mut col0 = vec![x.clone()];
    let mut col1 = vec![y.clone()];

    for _ in 1..trace_length {
        (x, y) = (y.clone(), &x + &y);
        col0.push(x.clone());
        col1.push(y.clone());
    }

    TraceTable::from_columns_main(vec![col0, col1], 1)
}

#[cfg(test)]
mod tests {
    use math::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    };

    use super::compute_trace;

    #[test]
    fn trace_has_expected_rows() {
        let trace = compute_trace(FieldElement::<Stark252PrimeField>::one(), 8);
        assert_eq!(trace.num_rows(), 8);

        let trace = compute_trace(FieldElement::<Stark252PrimeField>::one(), 64);
        assert_eq!(trace.num_rows(), 64);
    }
}
