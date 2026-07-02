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
use math::field::{element::FieldElement, goldilocks::GoldilocksField, traits::IsFFTField};

type StarkField = GoldilocksField;

/// Single-body [`ConstraintSet`] for [`DummyAIR`]: a fibonacci recurrence on
/// column 1 and an IS_BIT on column 0, written once against the
/// [`ConstraintBuilder`].
#[derive(Default)]
pub struct DummyConstraints;

impl ConstraintSet<StarkField, StarkField> for DummyConstraints {
    fn meta(&self) -> Vec<ConstraintMeta> {
        vec![
            // idx 0: fibonacci on column 1; reads two next rows ⇒ 2 end exemptions.
            ConstraintMeta::base(0, 1).with_end_exemptions(2),
            // idx 1: IS_BIT on column 0, every row.
            ConstraintMeta::base(1, 2),
        ]
    }

    fn eval<B: ConstraintBuilder<StarkField, StarkField>>(&self, b: &mut B) {
        // a_{i+2} = a_{i+1} + a_i on column 1.
        let a0 = b.main(0, 1);
        let a1 = b.main(1, 1);
        let a2 = b.main(2, 1);
        b.emit_base(0, a2 - a1 - a0);

        // bit * (bit - 1) = 0 on column 0.
        let bit = b.main(0, 0);
        let one = b.one();
        b.emit_base(1, bit.clone() * (bit - one));
    }
}

pub struct DummyAIR {
    context: AirContext,
    meta: Vec<ConstraintMeta>,
}

impl AIR for DummyAIR {
    type Field = StarkField;
    type FieldExtension = StarkField;
    type PublicInputs = ();

    fn step_size(&self) -> usize {
        1
    }

    fn new(proof_options: &ProofOptions) -> Self {
        let meta = DummyConstraints.meta();

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 2,
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: meta.len(),
        };

        Self { context, meta }
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

    fn constraints_meta(&self) -> &[ConstraintMeta] {
        &self.meta
    }

    fn compute_transition_prover(
        &self,
        evaluation_context: &TransitionEvaluationContext<Self::Field, Self::FieldExtension>,
        base_evals: &mut [FieldElement<Self::Field>],
        ext_evals: &mut [FieldElement<Self::FieldExtension>],
    ) {
        run_transition_prover(&DummyConstraints, evaluation_context, base_evals, ext_evals);
    }

    fn compute_transition(
        &self,
        evaluation_context: &TransitionEvaluationContext<Self::Field, Self::FieldExtension>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        run_transition_verifier(
            &DummyConstraints,
            evaluation_context,
            self.num_base_transition_constraints(),
            self.num_transition_constraints(),
        )
    }

    fn num_base_transition_constraints(&self) -> usize {
        num_base_from_meta(&DummyConstraints.meta())
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
