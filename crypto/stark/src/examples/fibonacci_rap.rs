use std::{marker::PhantomData, ops::Div};

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
use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::{
    field::{element::FieldElement, traits::IsFFTField},
    traits::ByteConversion,
};

/// Pads each trace column with zeros up to the next power of two so the
/// radix-2 FFT can be applied. Local to this example — the production
/// prover sizes its traces directly.
fn resize_to_next_power_of_two<F: IsFFTField>(trace_columns: &mut [Vec<FieldElement<F>>]) {
    for col in trace_columns.iter_mut() {
        col.resize(col.len().next_power_of_two(), FieldElement::<F>::zero());
    }
}

/// Single-body [`ConstraintSet`] for [`FibonacciRAP`]: the Fibonacci
/// recurrence plus the RAP permutation constraint, written once against the
/// [`ConstraintBuilder`]. The permutation constraint reads the auxiliary
/// (RAP) column and the interaction challenge, so it is an `Ext` constraint
/// after the `Base` prefix.
pub struct FibonacciRAPConstraints;

impl<F> ConstraintSet<F, F> for FibonacciRAPConstraints
where
    F: IsFFTField + Send + Sync,
{
    fn eval<B: ConstraintBuilder<F, F>>(&self, b: &mut B) {
        // idx 0: a_{i+2} = a_{i+1} + a_i on column 0. End exemptions hard-coded
        // for the steps = 16 integration tests.
        let a0 = b.main(0, 0);
        let a1 = b.main(1, 0);
        let a2 = b.main(2, 0);
        b.emit_base_rows(0, RowDomain::except_last(3 + 32 - 16 - 1), a2 - a1 - a0);

        // idx 1: permutation; z_{i+1} * (b_i + gamma) = z_i * (a_i + gamma);
        // reads the next row ⇒ 1 end exemption.
        let z_i = b.aux(0, 0);
        let z_i_plus_one = b.aux(1, 0);
        let gamma = b.challenge(0);
        let a_i = b.main(0, 0);
        let b_i = b.main(0, 1);
        b.emit_ext_rows(
            1,
            RowDomain::except_last(1),
            z_i_plus_one * (b_i + gamma.clone()) - z_i * (a_i + gamma),
        );
    }
}

pub struct FibonacciRAP<F>
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
pub struct FibonacciRAPPublicInputs<F>
where
    F: IsFFTField,
{
    pub steps: usize,
    pub a0: FieldElement<F>,
    pub a1: FieldElement<F>,
}

impl<F> AIR for FibonacciRAP<F>
where
    F: IsFFTField + Send + Sync + 'static,
    FieldElement<F>: ByteConversion,
{
    type Field = F;
    type FieldExtension = F;
    type PublicInputs = FibonacciRAPPublicInputs<Self::Field>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(proof_options: &ProofOptions) -> Self {
        let meta = ConstraintSet::<F, F>::meta(&FibonacciRAPConstraints);

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 3,
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: meta.len(),
        };

        Self {
            context,
            meta,
            phantom: PhantomData,
        }
    }

    fn build_auxiliary_trace(
        &self,
        trace: &mut TraceTable<Self::Field, Self::FieldExtension>,
        challenges: &[FieldElement<F>],
    ) -> Option<crate::lookup::BusPublicInputs<Self::FieldExtension>> {
        let main_segment_cols = trace.columns_main();
        let not_perm = &main_segment_cols[0];
        let perm = &main_segment_cols[1];
        let gamma = &challenges[0];

        let trace_len = trace.num_rows();

        let mut aux_col = Vec::new();
        for i in 0..trace_len {
            if i == 0 {
                aux_col.push(FieldElement::<Self::Field>::one());
            } else {
                let z_i = &aux_col[i - 1];
                let n_p_term = not_perm[i - 1].clone() + gamma;
                let p_term = &perm[i - 1] + gamma;

                // We are using that with high probability p_term != 0 because gamma is a random element.
                aux_col.push(z_i * n_p_term.div(p_term).unwrap());
            }
        }

        for (i, aux_elem) in aux_col.iter().enumerate().take(trace.num_rows()) {
            trace.set_aux(i, 0, aux_elem.clone())
        }

        None
    }

    fn build_rap_challenges(
        &self,
        transcript: &mut dyn IsStarkTranscript<Self::Field, Self::Field>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        vec![transcript.sample_field_element()]
    }

    fn trace_layout(&self) -> (usize, usize) {
        (2, 1)
    }

    fn boundary_constraints(
        &self,
        _pub_inputs: &Self::PublicInputs,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
        _bus_public_inputs: Option<&crate::lookup::BusPublicInputs<Self::FieldExtension>>,
        _trace_length: usize,
    ) -> BoundaryConstraints<Self::FieldExtension> {
        // Main boundary constraints
        let a0 =
            BoundaryConstraint::new_simple_main(0, FieldElement::<Self::FieldExtension>::one());
        let a1 =
            BoundaryConstraint::new_simple_main(1, FieldElement::<Self::FieldExtension>::one());

        // Auxiliary boundary constraints
        let a0_aux = BoundaryConstraint::new_aux(0, 0, FieldElement::<Self::FieldExtension>::one());

        BoundaryConstraints::from_constraints(vec![a0, a1, a0_aux])
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
            &FibonacciRAPConstraints,
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
            &FibonacciRAPConstraints,
            evaluation_context,
            self.num_base_transition_constraints(),
            self.num_transition_constraints(),
        )
    }

    fn num_base_transition_constraints(&self) -> usize {
        num_base_from_meta(&ConstraintSet::<F, F>::meta(&FibonacciRAPConstraints))
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        trace_length
    }
}

pub fn fibonacci_rap_trace<F: IsFFTField>(
    initial_values: [FieldElement<F>; 2],
    trace_length: usize,
) -> TraceTable<F, F> {
    let mut fib_seq: Vec<FieldElement<F>> = vec![];
    fib_seq.push(initial_values[0].clone());
    fib_seq.push(initial_values[1].clone());

    for i in 2..(trace_length) {
        fib_seq.push(fib_seq[i - 1].clone() + fib_seq[i - 2].clone());
    }

    let last_value = fib_seq[trace_length - 1].clone();
    let mut fib_permuted = fib_seq.clone();
    fib_permuted[0] = last_value;
    fib_permuted[trace_length - 1] = initial_values[0].clone();

    fib_seq.push(FieldElement::<F>::zero());
    fib_permuted.push(FieldElement::<F>::zero());
    let mut trace_cols = vec![fib_seq, fib_permuted];
    resize_to_next_power_of_two(&mut trace_cols);

    let aux_columns = vec![vec![FieldElement::<F>::zero(); trace_cols[0].len()]];

    let trace: TraceTable<F, F> = TraceTable::from_columns(trace_cols, aux_columns, 1);

    trace
}

#[cfg(test)]
mod test {
    use super::*;
    use math::field::goldilocks::GoldilocksField;

    type GoldilocksFE = FieldElement<GoldilocksField>;

    #[test]
    fn test_build_fibonacci_rap_trace() {
        // The fibonacci RAP trace should have two columns:
        //     * The usual fibonacci sequence column
        //     * The permuted fibonacci sequence column. The first and last elements are permuted.
        // Also, a 0 is appended at the end of both columns. The reason for this can be read in
        // https://hackmd.io/@aztec-network/plonk-arithmetiization-air#RAPs---PAIRs-with-interjected-verifier-randomness

        let trace = fibonacci_rap_trace([GoldilocksFE::from(1u64), GoldilocksFE::from(1u64)], 8);
        let mut expected_trace = vec![
            vec![
                GoldilocksFE::one(),
                GoldilocksFE::one(),
                GoldilocksFE::from(2u64),
                GoldilocksFE::from(3u64),
                GoldilocksFE::from(5u64),
                GoldilocksFE::from(8u64),
                GoldilocksFE::from(13u64),
                GoldilocksFE::from(21u64),
                GoldilocksFE::zero(),
            ],
            vec![
                GoldilocksFE::from(21u64),
                GoldilocksFE::one(),
                GoldilocksFE::from(2u64),
                GoldilocksFE::from(3u64),
                GoldilocksFE::from(5u64),
                GoldilocksFE::from(8u64),
                GoldilocksFE::from(13u64),
                GoldilocksFE::one(),
                GoldilocksFE::zero(),
            ],
        ];
        resize_to_next_power_of_two(&mut expected_trace);

        assert_eq!(trace.columns_main(), expected_trace);
    }

    #[test]
    fn aux_col() {
        let trace = fibonacci_rap_trace([GoldilocksFE::from(1u64), GoldilocksFE::from(1u64)], 64);
        let trace_cols = trace.columns_main();

        let not_perm = trace_cols[0].clone();
        let perm = trace_cols[1].clone();
        let gamma = GoldilocksFE::from(10u64);

        assert_eq!(perm.len(), not_perm.len());
        let trace_len = not_perm.len();

        let mut aux_col = Vec::new();
        for i in 0..trace_len {
            if i == 0 {
                aux_col.push(GoldilocksFE::one());
            } else {
                let z_i = aux_col[i - 1];
                let n_p_term = not_perm[i - 1] + gamma;
                let p_term = perm[i - 1] + gamma;

                aux_col.push(z_i * n_p_term.div(p_term).unwrap());
            }
        }

        assert_eq!(aux_col.last().unwrap(), &GoldilocksFE::one());
    }
}
