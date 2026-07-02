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
use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

/// Public inputs for the multi-column Fibonacci AIR.
/// Contains the initial values (first two elements) for each column.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "FieldElement<F>: serde::Serialize + serde::de::DeserializeOwned")]
pub struct FibonacciMultiColumnPublicInputs<F: IsFFTField> {
    /// Initial values for each column: (a0, a1) pairs
    pub initial_values: Vec<(FieldElement<F>, FieldElement<F>)>,
}

/// Single-body [`ConstraintSet`] for [`FibonacciMultiColumnAIR`]: one
/// Fibonacci constraint per column, written once against the
/// [`ConstraintBuilder`].
pub struct FibonacciMultiColumnConstraints {
    pub num_columns: usize,
}

impl<F, E> ConstraintSet<F, E> for FibonacciMultiColumnConstraints
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    fn meta(&self) -> Vec<ConstraintMeta> {
        // idx i: column i's a_{j+2} = a_{j+1} + a_j; reads two next rows ⇒ 2
        // end exemptions.
        (0..self.num_columns)
            .map(|i| ConstraintMeta::base(i, 1).with_end_exemptions(2))
            .collect()
    }

    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        for col in 0..self.num_columns {
            let a0 = b.main(0, col);
            let a1 = b.main(1, col);
            let a2 = b.main(2, col);
            // Constraint: a2 = a1 + a0  =>  a2 - a1 - a0 = 0
            b.emit_base(col, a2 - a1 - a0);
        }
    }
}

/// Multi-column Fibonacci AIR.
/// Each column contains an independent Fibonacci sequence.
pub struct FibonacciMultiColumnAIR<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    context: AirContext,
    meta: Vec<ConstraintMeta>,
    num_columns: usize,
    phantom: PhantomData<(F, E)>,
}

impl<F, E> AIR for FibonacciMultiColumnAIR<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = E;
    type PublicInputs = FibonacciMultiColumnPublicInputs<Self::Field>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(proof_options: &ProofOptions) -> Self {
        // Default to 2 columns if created via this method
        Self::with_num_columns(proof_options, 2)
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
            &FibonacciMultiColumnConstraints {
                num_columns: self.num_columns,
            },
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
            &FibonacciMultiColumnConstraints {
                num_columns: self.num_columns,
            },
            evaluation_context,
            self.num_base_transition_constraints(),
            self.num_transition_constraints(),
        )
    }

    fn num_base_transition_constraints(&self) -> usize {
        num_base_from_meta(&ConstraintSet::<F, E>::meta(
            &FibonacciMultiColumnConstraints {
                num_columns: self.num_columns,
            },
        ))
    }

    fn boundary_constraints(
        &self,
        pub_inputs: &Self::PublicInputs,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
        _bus_public_inputs: Option<&crate::lookup::BusPublicInputs<Self::FieldExtension>>,
        _trace_length: usize,
    ) -> BoundaryConstraints<Self::FieldExtension> {
        let mut constraints = Vec::new();

        // For each column, add boundary constraints for the first two rows
        for (col_idx, (a0, a1)) in pub_inputs.initial_values.iter().enumerate() {
            // First value (row 0)
            constraints.push(BoundaryConstraint::new_main(
                col_idx,
                0,
                a0.clone().to_extension(),
            ));
            // Second value (row 1)
            constraints.push(BoundaryConstraint::new_main(
                col_idx,
                1,
                a1.clone().to_extension(),
            ));
        }

        BoundaryConstraints::from_constraints(constraints)
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn trace_layout(&self) -> (usize, usize) {
        (self.num_columns, 0)
    }
}

impl<F, E> FibonacciMultiColumnAIR<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    /// Creates a new multi-column Fibonacci AIR with the specified number of columns.
    pub fn with_num_columns(proof_options: &ProofOptions, num_columns: usize) -> Self {
        let meta = ConstraintSet::<F, E>::meta(&FibonacciMultiColumnConstraints { num_columns });

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: num_columns,
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: meta.len(),
        };

        Self {
            context,
            meta,
            num_columns,
            phantom: PhantomData,
        }
    }
}

/// Computes the multi-column Fibonacci trace.
///
/// Each column starts with the provided initial values and computes a Fibonacci sequence.
///
/// # Arguments
/// * `initial_values` - Initial (a0, a1) pairs for each column
/// * `trace_length` - Number of rows in the trace (must be a power of 2)
///
/// # Returns
/// A TraceTable with `num_columns` columns and `trace_length` rows.
pub fn compute_trace<F, E>(
    initial_values: &[(FieldElement<F>, FieldElement<F>)],
    trace_length: usize,
) -> TraceTable<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    let num_columns = initial_values.len();
    let mut columns: Vec<Vec<FieldElement<F>>> = Vec::with_capacity(num_columns);

    for (a0, a1) in initial_values {
        let mut column = Vec::with_capacity(trace_length);
        column.push(a0.clone());
        column.push(a1.clone());

        for i in 2..trace_length {
            let next = column[i - 1].clone() + column[i - 2].clone();
            column.push(next);
        }

        columns.push(column);
    }

    TraceTable::from_columns_main(columns, 1)
}

/// Creates public inputs from initial values.
pub fn create_public_inputs<F: IsFFTField>(
    initial_values: Vec<(FieldElement<F>, FieldElement<F>)>,
) -> FibonacciMultiColumnPublicInputs<F> {
    FibonacciMultiColumnPublicInputs { initial_values }
}
