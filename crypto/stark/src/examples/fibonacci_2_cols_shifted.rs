use crate::{
    constraints::{
        boundary::{BoundaryConstraint, BoundaryConstraints},
        builder::{
            ConstraintBuilder, ConstraintMeta, ConstraintSet, RowDomain, num_base_from_meta,
            run_transition_prover, run_transition_verifier,
        },
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
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "FieldElement<F>: serde::Serialize + serde::de::DeserializeOwned")]
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

/// Single-body [`ConstraintSet`] for [`Fibonacci2ColsShifted`]: the two
/// shifted-Fibonacci recurrences, written once against the
/// [`ConstraintBuilder`].
pub struct Fibonacci2ColsShiftedConstraints<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField> Default for Fibonacci2ColsShiftedConstraints<F> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> ConstraintSet<F, F> for Fibonacci2ColsShiftedConstraints<F>
where
    F: IsFFTField + Send + Sync,
{
    fn eval<B: ConstraintBuilder<F, F>>(&self, b: &mut B) {
        let a0_0 = b.main(0, 0);
        let a0_1 = b.main(0, 1);
        let a1_0 = b.main(1, 0);
        let a1_1 = b.main(1, 1);

        // idx 0: Col0_{i+1} = Col1_i; reads the next row ⇒ 1 end exemption.
        b.emit_base_rows(0, RowDomain::except_last(1), a1_0 - a0_1.clone());
        // idx 1: Col1_{i+1} = Col0_i + Col1_i; reads the next row ⇒ 1 end exemption.
        b.emit_base_rows(1, RowDomain::except_last(1), a1_1 - a0_0 - a0_1);
    }
}

pub struct Fibonacci2ColsShifted<F>
where
    F: IsFFTField,
{
    context: AirContext,
    meta: Vec<ConstraintMeta>,
    phantom: PhantomData<F>,
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

    fn new(proof_options: &ProofOptions) -> Self {
        let meta = Fibonacci2ColsShiftedConstraints::<F>::default().meta();

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
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
        _bus_public_inputs: Option<&crate::lookup::BusPublicInputs<Self::FieldExtension>>,
        _trace_length: usize,
    ) -> BoundaryConstraints<Self::Field> {
        let initial_condition = BoundaryConstraint::new_main(0, 0, FieldElement::one());
        let claimed_value_constraint = BoundaryConstraint::new_main(
            0,
            pub_inputs.claimed_index,
            pub_inputs.claimed_value.clone(),
        );

        BoundaryConstraints::from_constraints(vec![initial_condition, claimed_value_constraint])
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
            &Fibonacci2ColsShiftedConstraints::default(),
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
            &Fibonacci2ColsShiftedConstraints::default(),
            evaluation_context,
            self.num_base_transition_constraints(),
            self.num_transition_constraints(),
        )
    }

    fn num_base_transition_constraints(&self) -> usize {
        num_base_from_meta(&Fibonacci2ColsShiftedConstraints::<F>::default().meta())
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
    use math::field::{element::FieldElement, goldilocks::GoldilocksField};

    use super::compute_trace;

    #[test]
    fn trace_has_expected_rows() {
        let trace = compute_trace(FieldElement::<GoldilocksField>::one(), 8);
        assert_eq!(trace.num_rows(), 8);

        let trace = compute_trace(FieldElement::<GoldilocksField>::one(), 64);
        assert_eq!(trace.num_rows(), 64);
    }
}
