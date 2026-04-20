use crate::{
    constraints::{boundary::BoundaryConstraints, transition::TransitionConstraintEvaluator},
    context::AirContext,
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::{AIR, TransitionEvaluationContext},
};
use math::field::{element::FieldElement, goldilocks::GoldilocksField};

type StarkField = GoldilocksField;
type Felt = FieldElement<GoldilocksField>;

#[derive(Clone)]
pub struct BitConstraint;
impl BitConstraint {
    fn new() -> Self {
        Self
    }
}

impl TransitionConstraintEvaluator<StarkField, StarkField> for BitConstraint {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        0
    }

    fn exemptions_period(&self) -> Option<usize> {
        Some(16)
    }

    fn periodic_exemptions_offset(&self) -> Option<usize> {
        Some(15)
    }

    fn evaluate_verifier(
        &self,
        evaluation_context: &TransitionEvaluationContext<StarkField, StarkField>,
        transition_evaluations: &mut [FieldElement<StarkField>],
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

        let step = frame.get_evaluation_step(0);

        let prefix_flag = step.get_main_evaluation_element(0, 0);
        let next_prefix_flag = step.get_main_evaluation_element(1, 0);

        let two = Felt::from(2);
        let one = Felt::one();
        let bit_flag = prefix_flag - two * next_prefix_flag;

        let bit_constraint = bit_flag * (bit_flag - one);

        transition_evaluations[self.constraint_idx()] = bit_constraint;
    }
}

#[derive(Clone)]
pub struct ZeroFlagConstraint;
impl ZeroFlagConstraint {
    fn new() -> Self {
        Self
    }
}

impl TransitionConstraintEvaluator<StarkField, StarkField> for ZeroFlagConstraint {
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        1
    }

    fn period(&self) -> usize {
        16
    }

    fn evaluate_verifier(
        &self,
        evaluation_context: &TransitionEvaluationContext<StarkField, StarkField>,
        transition_evaluations: &mut [FieldElement<StarkField>],
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

        let step = frame.get_evaluation_step(0);
        let zero_flag = step.get_main_evaluation_element(15, 0);

        transition_evaluations[self.constraint_idx()] = *zero_flag;
    }
}

pub struct BitFlagsAIR {
    context: AirContext,
    constraints: Vec<Box<dyn TransitionConstraintEvaluator<StarkField, StarkField>>>,
}

impl AIR for BitFlagsAIR {
    type Field = StarkField;
    type FieldExtension = StarkField;
    type PublicInputs = ();

    fn step_size(&self) -> usize {
        16
    }

    fn new(proof_options: &ProofOptions) -> Self {
        let bit_constraint = Box::new(BitConstraint::new());
        let flag_constraint = Box::new(ZeroFlagConstraint::new());
        let constraints: Vec<
            Box<dyn TransitionConstraintEvaluator<Self::Field, Self::FieldExtension>>,
        > = vec![bit_constraint, flag_constraint];

        let num_transition_constraints = constraints.len();

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 2,
            transition_offsets: vec![0],
            num_transition_constraints,
        };

        Self {
            context,
            constraints,
        }
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraintEvaluator<Self::Field, Self::FieldExtension>>> {
        &self.constraints
    }

    fn boundary_constraints(
        &self,
        _pub_inputs: &Self::PublicInputs,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
        _bus_public_inputs: Option<&crate::lookup::BusPublicInputs<Self::FieldExtension>>,
        _trace_length: usize,
    ) -> BoundaryConstraints<Self::FieldExtension> {
        BoundaryConstraints::from_constraints(vec![])
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        trace_length * 2
    }

    fn trace_layout(&self) -> (usize, usize) {
        (1, 0)
    }
}

pub fn bit_prefix_flag_trace(num_steps: usize) -> TraceTable<StarkField, StarkField> {
    debug_assert!(num_steps.is_power_of_two());
    let step: Vec<Felt> = [
        1031u64, 515, 257, 128, 64, 32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0,
    ]
    .iter()
    .map(|t| Felt::from(*t))
    .collect();

    let mut data: Vec<Felt> = std::iter::repeat_n(step, num_steps).flatten().collect();
    data[0] = Felt::from(1030);

    let mut dummy_column = (0..16).map(Felt::from).collect();
    dummy_column = std::iter::repeat_n(dummy_column, num_steps)
        .flatten()
        .collect();
    TraceTable::from_columns_main(vec![data, dummy_column], 16)
}
