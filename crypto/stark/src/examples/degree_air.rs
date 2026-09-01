//! DEGREE-LANE EXPERIMENT AIR (temporary, not for merge).
//!
//! A degree-parameterised AIR used to measure the cost of STARK constraint
//! degree. `W` columns each carry an independent chain `x_{i+1} = x_i^D`, so
//! the table has `W` transition constraints of multivariate degree exactly `D`.
//!
//! Two knobs, deliberately orthogonal:
//! - `D` sets the constraint degree, hence the composition part count `D - 1`
//!   (via [`AIR::composition_poly_degree_bound`]) *and* the per-row constraint
//!   evaluation work.
//! - `W` sets the trace width and constraint count, holding degree fixed.
//!
//! Sweeping `D` at fixed `W` gives the degree axis; sweeping `W` at fixed `D`
//! calibrates the per-constraint cost so the two can be separated.

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

/// `W` independent degree-`D` chains: `x_{i+1,c} = x_{i,c}^D`.
pub struct DegreeConstraints<F: IsFFTField, const D: usize, const W: usize> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField, const D: usize, const W: usize> Default for DegreeConstraints<F, D, W> {
    fn default() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F, const D: usize, const W: usize> ConstraintSet<F, F> for DegreeConstraints<F, D, W>
where
    F: IsFFTField + Send + Sync,
{
    fn eval<B: ConstraintBuilder<F, F>>(&self, b: &mut B) {
        for c in 0..W {
            let x = b.main(0, c);
            let next = b.main(1, c);
            // x^D by repeated multiplication: degree exactly D.
            let mut pow = x.clone();
            for _ in 1..D {
                pow = pow * x.clone();
            }
            // Reads the next row ⇒ 1 end exemption.
            b.emit_base_rows(c, RowDomain::except_last(1), next - pow);
        }
    }

    fn max_degree(&self) -> usize {
        D
    }
}

pub struct DegreeAir<F, const D: usize, const W: usize>
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
pub struct DegreePublicInputs<F>
where
    F: IsFFTField,
{
    pub seeds: Vec<FieldElement<F>>,
}

impl<F, const D: usize, const W: usize> AIR for DegreeAir<F, D, W>
where
    F: IsFFTField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = F;
    type PublicInputs = DegreePublicInputs<F>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(proof_options: &ProofOptions) -> Self {
        let meta = DegreeConstraints::<F, D, W>::default().meta();
        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: W,
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
        BoundaryConstraints::from_constraints(
            pub_inputs
                .seeds
                .iter()
                .enumerate()
                .map(|(c, v)| BoundaryConstraint::new_main(c, 0, v.clone()))
                .collect(),
        )
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
            &DegreeConstraints::<F, D, W>::default(),
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
            &DegreeConstraints::<F, D, W>::default(),
            evaluation_context,
            self.num_base_transition_constraints(),
            self.num_transition_constraints(),
        )
    }

    fn num_base_transition_constraints(&self) -> usize {
        num_base_from_meta(&DegreeConstraints::<F, D, W>::default().meta())
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    /// `parts = max_degree - 1`, matching the framework's rule in
    /// `LookupAir::composition_poly_degree_bound`.
    ///
    /// `LVM_DEGREE_DECLARED` overrides the *declared* degree independently of
    /// the true degree `D`. That decouples the two things raising a constraint
    /// degree does at once — more arithmetic per row, and more composition
    /// parts — so each can be priced on its own:
    ///
    ///   true D=3, declared 3  → 2 parts, degree-3 expressions
    ///   true D=5, declared 5  → 4 parts, degree-5 expressions   (the real d=5)
    ///   true D=3, declared 5  → 4 parts, degree-3 expressions   (parts only)
    ///
    /// The third minus the first is the cost of the parts; the second minus the
    /// third is the cost of the arithmetic. Over-declaring is always safe;
    /// UNDER-declaring is not (the bound would no longer cover H's true degree),
    /// so the override is clamped to at least the true degree.
    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        let declared = std::env::var("LVM_DEGREE_DECLARED")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(D)
            .max(D);
        trace_length * (declared - 1).max(1)
    }

    fn trace_layout(&self) -> (usize, usize) {
        (W, 0)
    }
}

/// Build the trace: each column `c` starts at `seeds[c]` and iterates `x → x^D`.
pub fn degree_trace<F: IsFFTField, const D: usize>(
    seeds: &[FieldElement<F>],
    trace_length: usize,
) -> TraceTable<F, F> {
    let width = seeds.len();
    let columns: Vec<Vec<FieldElement<F>>> = seeds
        .iter()
        .map(|seed| {
            let mut col = Vec::with_capacity(trace_length);
            col.push(seed.clone());
            for i in 1..trace_length {
                let prev = col[i - 1].clone();
                let mut v = prev.clone();
                for _ in 1..D {
                    v = v * prev.clone();
                }
                col.push(v);
            }
            col
        })
        .collect();
    TraceTable::from_columns_main(columns, width)
}
