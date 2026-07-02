use std::marker::PhantomData;

use crate::{
    constraints::{
        boundary::{BoundaryConstraint, BoundaryConstraints},
        builder::{
            ConstraintBuilder, ConstraintMeta, ConstraintSet, num_base_from_meta,
            run_transition_prover, run_transition_verifier,
        },
        transition::TransitionConstraintEvaluator,
    },
    context::AirContext,
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::{AIR, TransitionEvaluationContext},
};
use math::field::{element::FieldElement, traits::IsFFTField};

#[derive(Clone)]
struct QuadraticConstraint<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField> QuadraticConstraint<F> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> TransitionConstraintEvaluator<F, F> for QuadraticConstraint<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        0
    }

    fn end_exemptions(&self) -> usize {
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
        let second_step = frame.get_evaluation_step(1);

        let x = first_step.get_main_evaluation_element(0, 0);
        let x_squared = second_step.get_main_evaluation_element(0, 0);

        let res = x_squared - x * x;

        transition_evaluations[self.constraint_idx()] = res;
    }
}

/// Single-body [`ConstraintSet`] for [`QuadraticAIR`]: the same constraint
/// as `QuadraticConstraint`, written once against the [`ConstraintBuilder`].
pub struct QuadraticConstraints<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField> Default for QuadraticConstraints<F> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> ConstraintSet<F, F> for QuadraticConstraints<F>
where
    F: IsFFTField + Send + Sync,
{
    fn meta(&self) -> Vec<ConstraintMeta> {
        // idx 0: x_{i+1} = x_i²; reads the next row ⇒ 1 end exemption.
        vec![ConstraintMeta::base(0, 2).with_end_exemptions(1)]
    }

    fn eval<B: ConstraintBuilder<F, F>>(&self, b: &mut B) {
        let x = b.main(0, 0);
        let x_squared = b.main(1, 0);
        b.emit_base(0, x_squared - x.clone() * x);
    }
}

pub struct QuadraticAIR<F>
where
    F: IsFFTField,
{
    context: AirContext,
    constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, F>>>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "FieldElement<F>: serde::Serialize + serde::de::DeserializeOwned")]
pub struct QuadraticPublicInputs<F>
where
    F: IsFFTField,
{
    pub a0: FieldElement<F>,
}

impl<F> AIR for QuadraticAIR<F>
where
    F: IsFFTField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = F;
    type PublicInputs = QuadraticPublicInputs<Self::Field>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(proof_options: &ProofOptions) -> Self {
        let constraints: Vec<
            Box<dyn TransitionConstraintEvaluator<Self::Field, Self::FieldExtension>>,
        > = vec![Box::new(QuadraticConstraint::new())];

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 1,
            transition_offsets: vec![0, 1],
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
        let a0 = BoundaryConstraint::new_simple_main(0, pub_inputs.a0.clone());

        BoundaryConstraints::from_constraints(vec![a0])
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraintEvaluator<Self::Field, Self::FieldExtension>>> {
        &self.constraints
    }

    fn compute_transition_prover(
        &self,
        evaluation_context: &TransitionEvaluationContext<Self::Field, Self::FieldExtension>,
        base_evals: &mut [FieldElement<Self::Field>],
        ext_evals: &mut [FieldElement<Self::FieldExtension>],
    ) {
        run_transition_prover(
            &QuadraticConstraints::default(),
            evaluation_context,
            base_evals,
            ext_evals,
        );
    }

    fn compute_transition(
        &self,
        evaluation_context: &TransitionEvaluationContext<Self::Field, Self::FieldExtension>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        run_transition_verifier(
            &QuadraticConstraints::default(),
            evaluation_context,
            self.num_base_transition_constraints(),
            self.num_transition_constraints(),
        )
    }

    fn num_base_transition_constraints(&self) -> usize {
        num_base_from_meta(&QuadraticConstraints::<F>::default().meta())
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        2 * trace_length
    }

    fn trace_layout(&self) -> (usize, usize) {
        (1, 0)
    }
}

pub fn quadratic_trace<F: IsFFTField>(
    initial_value: FieldElement<F>,
    trace_length: usize,
) -> TraceTable<F, F> {
    let mut ret: Vec<FieldElement<F>> = vec![];

    ret.push(initial_value);

    for i in 1..(trace_length) {
        ret.push(ret[i - 1].clone() * ret[i - 1].clone());
    }

    TraceTable::from_columns_main(vec![ret], 1)
}
