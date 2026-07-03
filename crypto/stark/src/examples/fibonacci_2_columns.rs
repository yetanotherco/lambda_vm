use std::marker::PhantomData;

use super::simple_fibonacci::FibonacciPublicInputs;
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

/// Single-body [`ConstraintSet`] for [`Fibonacci2ColsAIR`]: the two row-major
/// Fibonacci recurrences, written once against the [`ConstraintBuilder`].
pub struct Fibonacci2ColsConstraints<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField> Default for Fibonacci2ColsConstraints<F> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> ConstraintSet<F, F> for Fibonacci2ColsConstraints<F>
where
    F: IsFFTField + Send + Sync,
{
    fn eval<B: ConstraintBuilder<F, F>>(&self, b: &mut B) {
        let s0_0 = b.main(0, 0);
        let s0_1 = b.main(0, 1);
        let s1_0 = b.main(1, 0);
        let s1_1 = b.main(1, 1);

        // idx 0: s_{0, i+1} = s_{0, i} + s_{1, i}; reads the next row ⇒ 1 end exemption.
        b.emit_base_exempt(0, 1, 1, s1_0.clone() - s0_0 - s0_1.clone());
        // idx 1: s_{1, i+1} = s_{1, i} + s_{0, i+1}; reads the next row ⇒ 1 end exemption.
        b.emit_base_exempt(1, 1, 1, s1_1 - s0_1 - s1_0);
    }
}

pub struct Fibonacci2ColsAIR<F>
where
    F: IsFFTField,
{
    context: AirContext,
    meta: Vec<ConstraintMeta>,
    phantom: PhantomData<F>,
}

/// The AIR for to a 2 column trace, where the columns form a Fibonacci sequence when
/// stacked in row-major order.
impl<F> AIR for Fibonacci2ColsAIR<F>
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
        let meta = Fibonacci2ColsConstraints::<F>::default().meta();

        let context = AirContext {
            proof_options: proof_options.clone(),
            transition_offsets: vec![0, 1],
            num_transition_constraints: meta.len(),
            trace_columns: 2,
        };

        Self {
            context,
            meta,
            phantom: PhantomData,
        }
    }

    fn boundary_constraints(
        &self,
        pub_inputs: &Self::PublicInputs,
        _rap_challenges: &[FieldElement<Self::Field>],
        _bus_public_inputs: Option<&crate::lookup::BusPublicInputs<Self::FieldExtension>>,
        _trace_length: usize,
    ) -> BoundaryConstraints<Self::Field> {
        let a0 = BoundaryConstraint::new_main(0, 0, pub_inputs.a0.clone());
        let a1 = BoundaryConstraint::new_main(1, 0, pub_inputs.a1.clone());

        BoundaryConstraints::from_constraints(vec![a0, a1])
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
            &Fibonacci2ColsConstraints::default(),
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
            &Fibonacci2ColsConstraints::default(),
            evaluation_context,
            self.num_base_transition_constraints(),
            self.num_transition_constraints(),
        )
    }

    fn num_base_transition_constraints(&self) -> usize {
        num_base_from_meta(&Fibonacci2ColsConstraints::<F>::default().meta())
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        trace_length
    }

    fn trace_layout(&self) -> (usize, usize) {
        (2, 0)
    }
}

pub fn compute_trace<F: IsFFTField>(
    initial_values: [FieldElement<F>; 2],
    trace_length: usize,
) -> TraceTable<F, F> {
    let mut ret1: Vec<FieldElement<F>> = vec![];
    let mut ret2: Vec<FieldElement<F>> = vec![];

    ret1.push(initial_values[0].clone());
    ret2.push(initial_values[1].clone());

    for i in 1..(trace_length) {
        let new_val = ret1[i - 1].clone() + ret2[i - 1].clone();
        ret1.push(new_val.clone());
        ret2.push(new_val + ret2[i - 1].clone());
    }

    TraceTable::from_columns_main(vec![ret1, ret2], 1)
}
