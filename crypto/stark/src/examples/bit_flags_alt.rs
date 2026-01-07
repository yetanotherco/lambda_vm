use crate::{
    Felt252,
    constraints::{
        boundary::BoundaryConstraints,
        lookup::{Air, AirLogic},
        transition::TransitionConstraint,
    },
    context::AirContext,
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::{AIR, TransitionEvaluationContext},
};
use math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

type StarkField = Stark252PrimeField;

#[derive(Clone)]
pub struct BitConstraint;
impl BitConstraint {
    fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<StarkField, StarkField> for BitConstraint {
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

    fn end_exemptions(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<StarkField, StarkField>,
        transition_evaluations: &mut [FieldElement<StarkField>],
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

        let step = frame.get_evaluation_step(0);

        let prefix_flag = step.get_main_evaluation_element(0, 0);
        let next_prefix_flag = step.get_main_evaluation_element(1, 0);

        let two = Felt252::from(2);
        let one = Felt252::one();
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

impl TransitionConstraint<StarkField, StarkField> for ZeroFlagConstraint {
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        1
    }

    fn end_exemptions(&self) -> usize {
        0
    }

    fn period(&self) -> usize {
        16
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<StarkField, StarkField>,
        transition_evaluations: &mut [FieldElement<StarkField>],
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

        let step = frame.get_evaluation_step(0);
        let zero_flag = step.get_main_evaluation_element(15, 0);

        transition_evaluations[self.constraint_idx()] = *zero_flag;
    }
}

pub fn bit_flags_air(
    trace_length: usize,
    proof_options: &ProofOptions,
) -> crate::constraints::lookup::Air<BitFlagsAirLogic, &(), StarkField, StarkField> {
    let bit_constraint = Box::new(BitConstraint::new());
    let flag_constraint = Box::new(ZeroFlagConstraint::new());
    let transition_constraints: Vec<Box<dyn TransitionConstraint<StarkField, StarkField>>> =
        vec![bit_constraint, flag_constraint];

    let num_transition_constraints = transition_constraints.len();

    let context = AirContext {
        proof_options: proof_options.clone(),
        trace_columns: 2,
        transition_offsets: vec![0],
        num_transition_constraints,
    };
    Air {
        context,
        trace_length,
        pub_inputs: &(),
        step_size: 16,
        trace_layout: (1, 0),
        transition_constraints,
        logic: BitFlagsAirLogic {},
    }
}

pub struct BitFlagsAirLogic {}

impl AirLogic<StarkField, StarkField> for BitFlagsAirLogic {}
