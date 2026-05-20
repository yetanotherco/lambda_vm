//! Lambda AIR with the same Fibonacci-pair statement used for Plonky3.
//!
//! Each sequence uses two columns `(left, right)` and advances two Fibonacci
//! steps per row:
//!
//! `next.left = local.left + local.right`
//! `next.right = local.right + next.left`

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

#[derive(Clone)]
pub struct FibPairShiftConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    seq_idx: usize,
    constraint_idx: usize,
    phantom: PhantomData<(F, E)>,
}

impl<F, E> FibPairShiftConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    pub fn new(seq_idx: usize, constraint_idx: usize) -> Self {
        Self {
            seq_idx,
            constraint_idx,
            phantom: PhantomData,
        }
    }
}

impl<F, E> TransitionConstraintEvaluator<F, E> for FibPairShiftConstraint<F, E>
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
        match eval_ctx {
            TransitionEvaluationContext::Prover { frame, .. } => {
                let s0 = frame.get_evaluation_step(0);
                let s1 = frame.get_evaluation_step(1);
                let local_left = s0.get_main_evaluation_element(0, 2 * self.seq_idx);
                let local_right = s0.get_main_evaluation_element(0, 2 * self.seq_idx + 1);
                let next_left = s1.get_main_evaluation_element(0, 2 * self.seq_idx);
                out[self.constraint_idx] = (next_left - local_left - local_right).to_extension();
            }
            TransitionEvaluationContext::Verifier { frame, .. } => {
                let s0 = frame.get_evaluation_step(0);
                let s1 = frame.get_evaluation_step(1);
                let local_left = s0.get_main_evaluation_element(0, 2 * self.seq_idx);
                let local_right = s0.get_main_evaluation_element(0, 2 * self.seq_idx + 1);
                let next_left = s1.get_main_evaluation_element(0, 2 * self.seq_idx);
                out[self.constraint_idx] = next_left - local_left - local_right;
            }
        }
    }

    #[inline]
    fn evaluate_prover(
        &self,
        eval_ctx: &TransitionEvaluationContext<F, E>,
        base_evals: &mut [FieldElement<F>],
        _ext_evals: &mut [FieldElement<E>],
    ) {
        let TransitionEvaluationContext::Prover { frame, .. } = eval_ctx else {
            unreachable!("prover evaluation must receive a prover frame");
        };
        let s0 = frame.get_evaluation_step(0);
        let s1 = frame.get_evaluation_step(1);
        let local_left = s0.get_main_evaluation_element(0, 2 * self.seq_idx);
        let local_right = s0.get_main_evaluation_element(0, 2 * self.seq_idx + 1);
        let next_left = s1.get_main_evaluation_element(0, 2 * self.seq_idx);
        base_evals[self.constraint_idx] = next_left - local_left - local_right;
    }
}

#[derive(Clone)]
pub struct FibPairSumConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    seq_idx: usize,
    constraint_idx: usize,
    phantom: PhantomData<(F, E)>,
}

impl<F, E> FibPairSumConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    pub fn new(seq_idx: usize, constraint_idx: usize) -> Self {
        Self {
            seq_idx,
            constraint_idx,
            phantom: PhantomData,
        }
    }
}

impl<F, E> TransitionConstraintEvaluator<F, E> for FibPairSumConstraint<F, E>
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
        match eval_ctx {
            TransitionEvaluationContext::Prover { frame, .. } => {
                let s0 = frame.get_evaluation_step(0);
                let s1 = frame.get_evaluation_step(1);
                let local_right = s0.get_main_evaluation_element(0, 2 * self.seq_idx + 1);
                let next_left = s1.get_main_evaluation_element(0, 2 * self.seq_idx);
                let next_right = s1.get_main_evaluation_element(0, 2 * self.seq_idx + 1);
                out[self.constraint_idx] = (next_right - local_right - next_left).to_extension();
            }
            TransitionEvaluationContext::Verifier { frame, .. } => {
                let s0 = frame.get_evaluation_step(0);
                let s1 = frame.get_evaluation_step(1);
                let local_right = s0.get_main_evaluation_element(0, 2 * self.seq_idx + 1);
                let next_left = s1.get_main_evaluation_element(0, 2 * self.seq_idx);
                let next_right = s1.get_main_evaluation_element(0, 2 * self.seq_idx + 1);
                out[self.constraint_idx] = next_right - local_right - next_left;
            }
        }
    }

    #[inline]
    fn evaluate_prover(
        &self,
        eval_ctx: &TransitionEvaluationContext<F, E>,
        base_evals: &mut [FieldElement<F>],
        _ext_evals: &mut [FieldElement<E>],
    ) {
        let TransitionEvaluationContext::Prover { frame, .. } = eval_ctx else {
            unreachable!("prover evaluation must receive a prover frame");
        };
        let s0 = frame.get_evaluation_step(0);
        let s1 = frame.get_evaluation_step(1);
        let local_right = s0.get_main_evaluation_element(0, 2 * self.seq_idx + 1);
        let next_left = s1.get_main_evaluation_element(0, 2 * self.seq_idx);
        let next_right = s1.get_main_evaluation_element(0, 2 * self.seq_idx + 1);
        base_evals[self.constraint_idx] = next_right - local_right - next_left;
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct FibonacciPairPublicInputs<F: IsFFTField> {
    pub initial_values: Vec<(FieldElement<F>, FieldElement<F>)>,
}

pub struct FibonacciPairMultiColAIR<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    context: AirContext,
    constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>>,
    /// Concrete copies of the shift constraints, used by the monomorphic
    /// `compute_transition_prover` override below. Mirrors the boxed entries
    /// stored in `constraints` (which the framework still needs for
    /// `transition_constraints()`), but lets the compiler inline
    /// `evaluate_prover` directly in the hot loop. Pattern lifted from
    /// PR #593 (`crypto/stark/src/lookup.rs`'s LogUp dispatch), applied here
    /// to base (domain) constraints.
    shift_constraints_direct: Vec<FibPairShiftConstraint<F, E>>,
    sum_constraints_direct: Vec<FibPairSumConstraint<F, E>>,
    num_sequences: usize,
}

impl<F, E> AIR for FibonacciPairMultiColAIR<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = E;
    type PublicInputs = FibonacciPairPublicInputs<Self::Field>;

    fn step_size(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "fib_pair"
    }

    fn new(proof_options: &ProofOptions) -> Self {
        Self::with_num_sequences(proof_options, 1)
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        trace_length
    }

    fn transition_constraints(&self) -> &Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> {
        &self.constraints
    }

    fn num_base_transition_constraints(&self) -> usize {
        2 * self.num_sequences
    }

    /// Monomorphic prover dispatch: bypass the `Box<dyn TransitionConstraintEvaluator>`
    /// vtable for the hot path. Lambda's default `compute_transition_prover`
    /// iterates `self.transition_constraints()` (the boxed slice) and calls
    /// `evaluate_prover` through an indirect call per constraint per LDE
    /// point. For log21 n=64 that's ~512M indirect calls. Storing concrete
    /// constraint copies and dispatching directly lets LLVM inline the body,
    /// which on `fib_iterative_4M` was worth -11.4% in r2_evaluate per
    /// PR #593's measurement (LogUp constraints). Same technique, applied
    /// here to the base (domain) constraints of the bench AIR.
    fn compute_transition_prover(
        &self,
        evaluation_context: &TransitionEvaluationContext<Self::Field, Self::FieldExtension>,
        base_evals: &mut [FieldElement<Self::Field>],
        ext_evals: &mut [FieldElement<Self::FieldExtension>],
    ) {
        for c in &self.shift_constraints_direct {
            c.evaluate_prover(evaluation_context, base_evals, ext_evals);
        }
        for c in &self.sum_constraints_direct {
            c.evaluate_prover(evaluation_context, base_evals, ext_evals);
        }
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

impl<F, E> FibonacciPairMultiColAIR<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    pub fn with_num_sequences(proof_options: &ProofOptions, num_sequences: usize) -> Self {
        let mut constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> =
            Vec::with_capacity(2 * num_sequences);
        let mut shift_constraints_direct = Vec::with_capacity(num_sequences);
        let mut sum_constraints_direct = Vec::with_capacity(num_sequences);
        for seq in 0..num_sequences {
            let shift = FibPairShiftConstraint::new(seq, 2 * seq);
            let sum = FibPairSumConstraint::new(seq, 2 * seq + 1);
            shift_constraints_direct.push(shift.clone());
            sum_constraints_direct.push(sum.clone());
            constraints.push(Box::new(shift));
            constraints.push(Box::new(sum));
        }

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 2 * num_sequences,
            transition_offsets: vec![0, 1],
            num_transition_constraints: 2 * num_sequences,
        };

        Self {
            context,
            constraints,
            shift_constraints_direct,
            sum_constraints_direct,
            num_sequences,
        }
    }
}

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
            let next_left = left + &right;
            let next_right = right + &next_left;
            left = next_left;
            right = next_right;
        }

        columns.push(left_col);
        columns.push(right_col);
    }

    TraceTable::from_columns_main(columns, 1)
}

pub fn create_public_inputs<F: IsFFTField>(
    initial_values: Vec<(FieldElement<F>, FieldElement<F>)>,
) -> FibonacciPairPublicInputs<F> {
    FibonacciPairPublicInputs { initial_values }
}
