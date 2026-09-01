use std::marker::PhantomData;

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
use math::field::{element::FieldElement, traits::IsFFTField};

/// DEGREE-LANE EXPERIMENT HOOK (temporary, not for merge).
///
/// Number of composition-polynomial parts this AIR advertises. The prover and
/// verifier both derive the part count as
/// `composition_poly_degree_bound(N) / N`, so overriding it here inflates the
/// part count exactly as a higher-degree AIR would, while leaving the actual
/// constraint (degree 2, one multiplication) untouched. That isolates the
/// *structural* cost of extra quotient parts from the cost of evaluating
/// genuinely higher-degree constraint expressions.
///
/// Read once from `LVM_DEGREE_PARTS`; defaults to the AIR's natural 2.
fn parts_override() -> usize {
    // Read per call (not cached): the probe sweeps several part counts inside a
    // single process. Called O(1) times per prove/verify, never in a hot loop.
    std::env::var("LVM_DEGREE_PARTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2)
}

/// Single-body [`ConstraintSet`] for [`QuadraticAIR`]: `x_{i+1} = x_i²`,
/// written once against the [`ConstraintBuilder`].
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
    fn eval<B: ConstraintBuilder<F, F>>(&self, b: &mut B) {
        let x = b.main(0, 0);
        let x_squared = b.main(1, 0);
        // idx 0: x_{i+1} = x_i²; reads the next row ⇒ 1 end exemption.
        b.emit_base_rows(0, RowDomain::except_last(1), x_squared - x.clone() * x);
    }
}

pub struct QuadraticAIR<F>
where
    F: IsFFTField,
{
    context: AirContext,
    meta: Vec<ConstraintMeta>,
    phantom: PhantomData<F>,
}

#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
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
        let meta = QuadraticConstraints::<F>::default().meta();

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 1,
            transition_offsets: vec![0, 1],
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
        let a0 = BoundaryConstraint::new_simple_main(0, pub_inputs.a0.clone());

        BoundaryConstraints::from_constraints(vec![a0])
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
        parts_override() * trace_length
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
