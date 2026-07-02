use crate::{
    constraints::{
        boundary::{BoundaryConstraint, BoundaryConstraints},
        builder::{
            ConstraintBuilder, ConstraintMeta, ConstraintSet, num_base_from_meta,
            run_transition_prover, run_transition_verifier,
        },
    },
    context::AirContext,
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::{AIR, TransitionEvaluationContext},
};
use math::field::{element::FieldElement, traits::IsFFTField};
use std::marker::PhantomData;

/// Single-body [`ConstraintSet`] for [`FibonacciAIR`]: `a_{i+2} = a_{i+1} + a_i`,
/// written once against the [`ConstraintBuilder`].
pub struct SimpleFibonacciConstraints<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField> Default for SimpleFibonacciConstraints<F> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> ConstraintSet<F, F> for SimpleFibonacciConstraints<F>
where
    F: IsFFTField + Send + Sync,
{
    fn meta(&self) -> Vec<ConstraintMeta> {
        // idx 0: a_{i+2} = a_{i+1} + a_i; reads two next rows ⇒ 2 end exemptions.
        vec![ConstraintMeta::base(0, 1).with_end_exemptions(2)]
    }

    fn eval<B: ConstraintBuilder<F, F>>(&self, b: &mut B) {
        let a0 = b.main(0, 0);
        let a1 = b.main(1, 0);
        let a2 = b.main(2, 0);
        b.emit_base(0, a2 - a1 - a0);
    }
}

pub struct FibonacciAIR<F>
where
    F: IsFFTField,
{
    context: AirContext,
    meta: Vec<ConstraintMeta>,
    phantom: PhantomData<F>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "FieldElement<F>: serde::Serialize + serde::de::DeserializeOwned")]
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

    fn new(proof_options: &ProofOptions) -> Self {
        let meta = SimpleFibonacciConstraints::<F>::default().meta();

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 1,
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: meta.len(),
        };

        Self {
            context,
            meta,
            phantom: PhantomData,
        }
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        trace_length
    }

    fn constraints_meta(&self) -> &[ConstraintMeta] {
        &self.meta
    }

    fn compute_transition_prover(
        &self,
        evaluation_context: &TransitionEvaluationContext<Self::Field, Self::FieldExtension>,
        base_evals: &mut [FieldElement<Self::Field>],
        ext_evals: &mut [FieldElement<Self::FieldExtension>],
    ) {
        run_transition_prover(
            &SimpleFibonacciConstraints::default(),
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
            &SimpleFibonacciConstraints::default(),
            evaluation_context,
            self.num_base_transition_constraints(),
            self.num_transition_constraints(),
        )
    }

    fn num_base_transition_constraints(&self) -> usize {
        num_base_from_meta(&SimpleFibonacciConstraints::<F>::default().meta())
    }

    fn boundary_constraints(
        &self,
        pub_inputs: &Self::PublicInputs,
        _rap_challenges: &[FieldElement<Self::Field>],
        _bus_public_inputs: Option<&crate::lookup::BusPublicInputs<Self::FieldExtension>>,
        _trace_length: usize,
    ) -> BoundaryConstraints<Self::Field> {
        let a0 = BoundaryConstraint::new_simple_main(0, pub_inputs.a0.clone());
        let a1 = BoundaryConstraint::new_simple_main(1, pub_inputs.a1.clone());

        BoundaryConstraints::from_constraints(vec![a0, a1])
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn trace_layout(&self) -> (usize, usize) {
        (1, 0)
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
