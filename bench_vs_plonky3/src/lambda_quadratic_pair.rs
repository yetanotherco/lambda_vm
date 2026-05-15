//! Quadratic-degree analogue of `lambda_fibonacci_pair` used to validate the
//! chunks-protocol payoff in the d_max=2 case.
//!
//! Same shape: `num_sequences` independent sequences, each with `(left,
//! right)` columns and a 2-row transition window — 2·num_sequences columns,
//! 2·num_sequences transition constraints, 2·num_sequences boundary
//! constraints.
//!
//! The transition is **quadratic** instead of linear:
//!
//! ```text
//!   next.left  = local.left * local.right          (degree 2)
//!   next.right = next.left  * local.right          (degree 2)
//! ```
//!
//! → `composition_poly_degree_bound = 2 * trace_length`
//! → `d_max = 2` → `num_chunks = 2` under the chunks protocol.
//!
//! This is the smallest non-degenerate exercise of the chunks-protocol
//! quotient split (num_chunks=2). Single-H pays `decompose_and_extend_d2`
//! (algebraic H(x) = H_0(x²) + x·H_1(x²) split); chunks pays the equivalent
//! per-chunk LDE+commit without `break_in_parts`. Expected: chunks wins by
//! ~10-20% at log_rows ≥ 19 on the EPYC bench server.
//!
//! Not present in the P3 side — this is bench-only, internal to Lambda.

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

/// `next.left = local.left * local.right` — degree 2 in trace polynomials.
#[derive(Clone)]
pub struct QuadPairLeftConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    seq_idx: usize,
    constraint_idx: usize,
    phantom_f: PhantomData<F>,
    phantom_e: PhantomData<E>,
}

impl<F, E> QuadPairLeftConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    pub fn new(seq_idx: usize, constraint_idx: usize) -> Self {
        Self {
            seq_idx,
            constraint_idx,
            phantom_f: PhantomData,
            phantom_e: PhantomData,
        }
    }
}

impl<F, E> TransitionConstraintEvaluator<F, E> for QuadPairLeftConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        2
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
                let res = next_left - local_left * local_right;
                out[self.constraint_idx] = res.to_extension();
            }
            TransitionEvaluationContext::Verifier { frame, .. } => {
                let s0 = frame.get_evaluation_step(0);
                let s1 = frame.get_evaluation_step(1);
                let local_left = s0.get_main_evaluation_element(0, 2 * self.seq_idx);
                let local_right = s0.get_main_evaluation_element(0, 2 * self.seq_idx + 1);
                let next_left = s1.get_main_evaluation_element(0, 2 * self.seq_idx);
                let res = next_left - local_left * local_right;
                out[self.constraint_idx] = res;
            }
        }
    }
}

/// `next.right = next.left * local.right` — degree 2 in trace polynomials.
#[derive(Clone)]
pub struct QuadPairRightConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    seq_idx: usize,
    constraint_idx: usize,
    phantom_f: PhantomData<F>,
    phantom_e: PhantomData<E>,
}

impl<F, E> QuadPairRightConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    pub fn new(seq_idx: usize, constraint_idx: usize) -> Self {
        Self {
            seq_idx,
            constraint_idx,
            phantom_f: PhantomData,
            phantom_e: PhantomData,
        }
    }
}

impl<F, E> TransitionConstraintEvaluator<F, E> for QuadPairRightConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        2
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
                let res = next_right - next_left * local_right;
                out[self.constraint_idx] = res.to_extension();
            }
            TransitionEvaluationContext::Verifier { frame, .. } => {
                let s0 = frame.get_evaluation_step(0);
                let s1 = frame.get_evaluation_step(1);
                let local_right = s0.get_main_evaluation_element(0, 2 * self.seq_idx + 1);
                let next_left = s1.get_main_evaluation_element(0, 2 * self.seq_idx);
                let next_right = s1.get_main_evaluation_element(0, 2 * self.seq_idx + 1);
                let res = next_right - next_left * local_right;
                out[self.constraint_idx] = res;
            }
        }
    }
}

/// Public inputs: initial `(a, b)` pair per sequence (same shape as fib_pair).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct QuadraticPairPublicInputs<F: IsFFTField> {
    pub initial_values: Vec<(FieldElement<F>, FieldElement<F>)>,
}

/// Multi-sequence quadratic AIR. d_max=2.
pub struct QuadraticPairMultiColAIR<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    context: AirContext,
    constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>>,
    num_sequences: usize,
}

impl<F, E> AIR for QuadraticPairMultiColAIR<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = E;
    type PublicInputs = QuadraticPairPublicInputs<Self::Field>;

    fn step_size(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "quadratic_pair"
    }

    fn new(proof_options: &ProofOptions) -> Self {
        Self::with_num_sequences(proof_options, 2)
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        2 * trace_length
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

impl<F, E> QuadraticPairMultiColAIR<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
{
    pub fn with_num_sequences(proof_options: &ProofOptions, num_sequences: usize) -> Self {
        let mut constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>> =
            Vec::with_capacity(2 * num_sequences);
        for seq in 0..num_sequences {
            constraints.push(Box::new(QuadPairLeftConstraint::new(seq, 2 * seq)));
            constraints.push(Box::new(QuadPairRightConstraint::new(seq, 2 * seq + 1)));
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
            num_sequences,
        }
    }
}

/// Compute the quadratic-pair trace.
///
/// For each sequence with initial `(a, b)`:
/// ```text
///   row 0: (a,       b)
///   row 1: (a*b,     a*b * b)
///   row 2: (a*b * a*b², a²*b³ * a*b²)
///   ...
/// ```
/// Values can be very large but stay in field (Goldilocks p ≈ 2^64).
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
            let new_left = left.clone() * right.clone();
            let new_right = new_left.clone() * right.clone();
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
) -> QuadraticPairPublicInputs<F> {
    QuadraticPairPublicInputs { initial_values }
}
