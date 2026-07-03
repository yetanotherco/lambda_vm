//! A minimal AIR with a simple addition constraint: col0 + col1 = col2
//! This is used to test STARK proving/verification with small traces (1-2 rows).

use std::marker::PhantomData;

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

/// Single-body [`ConstraintSet`] for [`SimpleAdditionAIR`]: `col0 + col1 = col2`
/// (applied at every row), written once against the [`ConstraintBuilder`].
pub struct SimpleAdditionConstraints<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField> Default for SimpleAdditionConstraints<F> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> ConstraintSet<F, F> for SimpleAdditionConstraints<F>
where
    F: IsFFTField + Send + Sync,
{
    fn eval<B: ConstraintBuilder<F, F>>(&self, b: &mut B) {
        let col0 = b.main(0, 0);
        let col1 = b.main(0, 1);
        let col2 = b.main(0, 2);
        // idx 0: col0 + col1 - col2 = 0, applied at every row (degree 1, no exemptions).
        b.emit_base(0, 1, col0 + col1 - col2);
    }
}

pub struct SimpleAdditionAIR<F>
where
    F: IsFFTField,
{
    context: AirContext,
    meta: Vec<ConstraintMeta>,
    phantom: PhantomData<F>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "FieldElement<F>: serde::Serialize + serde::de::DeserializeOwned")]
pub struct SimpleAdditionPublicInputs<F>
where
    F: IsFFTField,
{
    /// First value (col0 at row 0)
    pub a: FieldElement<F>,
    /// Second value (col1 at row 0)
    pub b: FieldElement<F>,
}

impl<F> AIR for SimpleAdditionAIR<F>
where
    F: IsFFTField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = F;
    type PublicInputs = SimpleAdditionPublicInputs<Self::Field>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(proof_options: &ProofOptions) -> Self {
        let meta = SimpleAdditionConstraints::<F>::default().meta();

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 3,            // col0, col1, col2
            transition_offsets: vec![0], // Only need current step
            num_transition_constraints: meta.len(),
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
        // Boundary constraints: col0[0] = a, col1[0] = b
        // new_main(col, step, value)
        let a0 = BoundaryConstraint::new_main(0, 0, pub_inputs.a.clone()); // col0 at step 0
        let a1 = BoundaryConstraint::new_main(1, 0, pub_inputs.b.clone()); // col1 at step 0

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
            &SimpleAdditionConstraints::default(),
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
            &SimpleAdditionConstraints::default(),
            evaluation_context,
            self.num_base_transition_constraints(),
            self.num_transition_constraints(),
        )
    }

    fn num_base_transition_constraints(&self) -> usize {
        num_base_from_meta(&SimpleAdditionConstraints::<F>::default().meta())
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        // Degree 1 constraint
        trace_length
    }

    fn trace_layout(&self) -> (usize, usize) {
        (3, 0) // 3 main columns, 0 aux columns
    }
}

/// Creates a trace table with `num_rows` rows where each row satisfies col0 + col1 = col2.
/// The values are: row i has col0=i+1, col1=i+2, col2=2i+3
pub fn simple_addition_trace<F: IsFFTField>(num_rows: usize) -> TraceTable<F, F> {
    let mut col0 = Vec::with_capacity(num_rows);
    let mut col1 = Vec::with_capacity(num_rows);
    let mut col2 = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let a = FieldElement::<F>::from(i as u64 + 1);
        let b = FieldElement::<F>::from(i as u64 + 2);
        let c = &a + &b;

        col0.push(a);
        col1.push(b);
        col2.push(c);
    }

    TraceTable::from_columns_main(vec![col0, col1, col2], 1)
}
