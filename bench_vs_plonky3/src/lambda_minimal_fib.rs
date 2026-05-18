//! Lambda AIR with the SAME logical work as `lambda_fibonacci_pair` but
//! collapsed into a SINGLE transition constraint.
//!
//! Purpose: isolate the per-constraint vtable dispatch cost. `fib_pair`
//! registers `2 * num_sequences` `Box<dyn TransitionConstraintEvaluator>`
//! (one per shift + one per sum per sequence) and the prover hot loop
//! dispatches through each on every LDE row. `minimal_fib` registers ONE
//! constraint that internally loops over all sequences within a single
//! `evaluate_verifier` body — the framework dispatches once per row.
//!
//! Both AIRs:
//! - Same trace shape: `2 * num_sequences` columns, same step_size=1, same
//!   2-row window
//! - Same boundary constraints (pin `(a, b)` at row 0 per sequence)
//! - Same composition_poly_degree_bound = trace_length
//! - Same public inputs shape
//!
//! Differs only in:
//! - `num_transition_constraints`: `2 * num_sequences` (fib_pair) vs `1`
//!   (minimal_fib)
//! - Dispatch count per LDE row: 2N (fib_pair) vs 1 (minimal_fib)
//!
//! The composition polynomial in minimal_fib's case is α⁰ × (sum of the
//! individual constraint values), without per-constraint α-weighting. For a
//! HONEST prover on a valid trace every individual constraint is zero, so
//! the sum is zero — proof generation succeeds. The verifier replays the
//! same single-constraint computation. This is bench-only, not production:
//! the loss of per-constraint α-protection trades soundness for cleanly
//! isolating the dispatch cost.

use std::marker::PhantomData;

use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};
use stark::{
    constraints::{
        boundary::{BoundaryConstraint, BoundaryConstraints},
        transition::TransitionConstraintEvaluator,
    },
    context::AirContext,
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::{AIR, TransitionEvaluationContext},
};

/// One transition constraint that internally evaluates every Fibonacci pair
/// shift + sum constraint across every sequence and sums them into a single
/// output slot.
#[derive(Clone)]
pub struct MinimalFibConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    num_sequences: usize,
    constraint_idx: usize,
    phantom_f: PhantomData<F>,
    phantom_e: PhantomData<E>,
}

impl<F, E> MinimalFibConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    pub fn new(num_sequences: usize) -> Self {
        Self {
            num_sequences,
            constraint_idx: 0,
            phantom_f: PhantomData,
            phantom_e: PhantomData,
        }
    }
}

impl<F, E> TransitionConstraintEvaluator<F, E> for MinimalFibConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        1
    }

    fn evaluate_verifier(
        &self,
        eval_ctx: &TransitionEvaluationContext<F, E>,
        out: &mut [FieldElement<E>],
    ) {
        let mut acc = FieldElement::<E>::zero();
        match eval_ctx {
            TransitionEvaluationContext::Prover { frame, .. } => {
                let s0 = frame.get_evaluation_step(0);
                let s1 = frame.get_evaluation_step(1);
                for seq in 0..self.num_sequences {
                    let local_left = s0.get_main_evaluation_element(0, 2 * seq);
                    let local_right = s0.get_main_evaluation_element(0, 2 * seq + 1);
                    let next_left = s1.get_main_evaluation_element(0, 2 * seq);
                    let next_right = s1.get_main_evaluation_element(0, 2 * seq + 1);

                    // Shift: next.left = local.left + local.right
                    let shift = next_left - local_left - local_right;
                    // Sum: next.right = local.right + next.left
                    let sum = next_right - local_right - next_left;

                    acc = acc + shift.to_extension() + sum.to_extension();
                }
            }
            TransitionEvaluationContext::Verifier { frame, .. } => {
                let s0 = frame.get_evaluation_step(0);
                let s1 = frame.get_evaluation_step(1);
                for seq in 0..self.num_sequences {
                    let local_left = s0.get_main_evaluation_element(0, 2 * seq);
                    let local_right = s0.get_main_evaluation_element(0, 2 * seq + 1);
                    let next_left = s1.get_main_evaluation_element(0, 2 * seq);
                    let next_right = s1.get_main_evaluation_element(0, 2 * seq + 1);

                    let shift = next_left - local_left - local_right;
                    let sum = next_right - local_right - next_left;

                    acc = acc + shift + sum;
                }
            }
        }
        out[self.constraint_idx] = acc;
    }
}

/// Public inputs: initial `(a, b)` per sequence (identical shape to fib_pair).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct MinimalFibPublicInputs<F: IsFFTField> {
    pub initial_values: Vec<(FieldElement<F>, FieldElement<F>)>,
}

/// Minimal-dispatch Fibonacci AIR.
pub struct MinimalFibMultiColAIR<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    context: AirContext,
    constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>>,
    num_sequences: usize,
}

impl<F, E> AIR for MinimalFibMultiColAIR<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = E;
    type PublicInputs = MinimalFibPublicInputs<Self::Field>;

    fn step_size(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "minimal_fib"
    }

    fn new(proof_options: &ProofOptions) -> Self {
        Self::with_num_sequences(proof_options, 2)
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        trace_length
    }

    fn transition_constraints(&self) -> &Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> {
        &self.constraints
    }

    fn boundary_constraints(
        &self,
        pub_inputs: &Self::PublicInputs,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
        _bus_public_inputs: Option<&stark::lookup::BusPublicInputs<Self::FieldExtension>>,
        _trace_length: usize,
    ) -> BoundaryConstraints<Self::FieldExtension> {
        let mut constraints = Vec::with_capacity(2 * pub_inputs.initial_values.len());
        for (seq_idx, (a, b)) in pub_inputs.initial_values.iter().enumerate() {
            constraints.push(BoundaryConstraint::new_main(
                2 * seq_idx,
                0,
                a.clone().to_extension(),
            ));
            constraints.push(BoundaryConstraint::new_main(
                2 * seq_idx + 1,
                0,
                b.clone().to_extension(),
            ));
        }
        BoundaryConstraints::from_constraints(constraints)
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn trace_layout(&self) -> (usize, usize) {
        (2 * self.num_sequences, 0)
    }
}

impl<F, E> MinimalFibMultiColAIR<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    pub fn with_num_sequences(proof_options: &ProofOptions, num_sequences: usize) -> Self {
        let constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> =
            vec![Box::new(MinimalFibConstraint::new(num_sequences))];

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 2 * num_sequences,
            transition_offsets: vec![0, 1],
            num_transition_constraints: 1,
        };

        Self {
            context,
            constraints,
            num_sequences,
        }
    }
}

/// Trace generator — identical to `lambda_fibonacci_pair::compute_trace`.
pub fn compute_trace<F, E>(
    initial_values: &[(FieldElement<F>, FieldElement<F>)],
    trace_length: usize,
) -> TraceTable<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    let num_sequences = initial_values.len();
    let mut columns: Vec<Vec<FieldElement<F>>> = Vec::with_capacity(2 * num_sequences);

    for (a, b) in initial_values {
        let mut left_col = Vec::with_capacity(trace_length);
        let mut right_col = Vec::with_capacity(trace_length);

        let mut left = a.clone();
        let mut right = b.clone();

        for _ in 0..trace_length {
            left_col.push(left.clone());
            right_col.push(right.clone());
            let new_left = left.clone() + right.clone();
            let new_right = right.clone() + new_left.clone();
            left = new_left;
            right = new_right;
        }

        columns.push(left_col);
        columns.push(right_col);
    }

    TraceTable::from_columns_main(columns, 1)
}

pub fn create_public_inputs<F: IsFFTField>(
    initial_values: Vec<(FieldElement<F>, FieldElement<F>)>,
) -> MinimalFibPublicInputs<F> {
    MinimalFibPublicInputs { initial_values }
}
