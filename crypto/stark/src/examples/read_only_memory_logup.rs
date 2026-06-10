//! Implementation of a LogUp Lookup Argument example.
//! See our blog post for detailed explanation.
//! <https://blog.lambdaclass.com/logup-lookup-argument-and-its-implementation-using-lambdaworks-for-continuous-read-only-memory/>

use std::marker::PhantomData;

use crate::{
    constraints::{
        boundary::{BoundaryConstraint, BoundaryConstraints},
        transition::TransitionConstraintEvaluator,
    },
    context::AirContext,
    proof::options::ProofOptions,
    trace::TraceTable,
    traits::{AIR, TransitionEvaluationContext},
};
use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use itertools::Itertools;
use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
    },
    traits::ByteConversion,
};

/// Transition Constraint that ensures the continuity of the sorted address column of a memory.
#[derive(Clone)]
struct ContinuityConstraint<F: IsSubFieldOf<E> + IsFFTField + Send + Sync, E: IsField + Send + Sync>
{
    phantom_f: PhantomData<F>,
    phantom_e: PhantomData<E>,
}

impl<F, E> ContinuityConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    pub fn new() -> Self {
        Self {
            phantom_f: PhantomData::<F>,
            phantom_e: PhantomData::<E>,
        }
    }
}

impl<F, E> TransitionConstraintEvaluator<F, E> for ContinuityConstraint<F, E>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        0
    }

    fn end_exemptions(&self) -> usize {
        // NOTE: We are assuming that the trace has as length a power of 2.
        1
    }

    fn evaluate_verifier(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        // In both evaluation contexts, Prover and Verfier will evaluate the transition polynomial in the same way.
        // The only difference is that the Prover's Frame has base field and field extension elements,
        // while the Verfier's Frame has only field extension elements.
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
                ..
            } => {
                let first_step = frame.get_evaluation_step(0);
                let second_step = frame.get_evaluation_step(1);

                let a_sorted_0 = first_step.get_main_evaluation_element(0, 2);
                let a_sorted_1 = second_step.get_main_evaluation_element(0, 2);
                // (a'_{i+1} - a'_i)(a'_{i+1} - a'_i - 1) = 0 where a' is the sorted address
                let res = (a_sorted_1 - a_sorted_0)
                    * (a_sorted_1 - a_sorted_0 - FieldElement::<F>::one());

                // The eval always exists, except if the constraint idx were incorrectly defined.
                if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx()) {
                    *eval = res.to_extension();
                }
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
                ..
            } => {
                let first_step = frame.get_evaluation_step(0);
                let second_step = frame.get_evaluation_step(1);

                let a_sorted_0 = first_step.get_main_evaluation_element(0, 2);
                let a_sorted_1 = second_step.get_main_evaluation_element(0, 2);
                // (a'_{i+1} - a'_i)(a'_{i+1} - a'_i - 1) = 0 where a' is the sorted address
                let res = (a_sorted_1 - a_sorted_0)
                    * (a_sorted_1 - a_sorted_0 - FieldElement::<E>::one());

                // The eval always exists, except if the constraint idx were incorrectly defined.
                if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx()) {
                    *eval = res;
                }
            }
        }
    }
}
/// Transition constraint that ensures that same addresses have same values, making the sorted memory read-only.
#[derive(Clone)]
struct SingleValueConstraint<
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
> {
    phantom_f: PhantomData<F>,
    phantom_e: PhantomData<E>,
}

impl<F, E> SingleValueConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    pub fn new() -> Self {
        Self {
            phantom_f: PhantomData::<F>,
            phantom_e: PhantomData::<E>,
        }
    }
}

impl<F, E> TransitionConstraintEvaluator<F, E> for SingleValueConstraint<F, E>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        1
    }

    fn end_exemptions(&self) -> usize {
        // NOTE: We are assuming that the trace has as length a power of 2.
        1
    }

    fn evaluate_verifier(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        // In both evaluation contexts, Prover and Verfier will evaluate the transition polynomial in the same way.
        // The only difference is that the Prover's Frame has base field and field extension elements,
        // while the Verfier's Frame has only field extension elements.
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
                ..
            } => {
                let first_step = frame.get_evaluation_step(0);
                let second_step = frame.get_evaluation_step(1);

                let a_sorted_0 = first_step.get_main_evaluation_element(0, 2);
                let a_sorted_1 = second_step.get_main_evaluation_element(0, 2);
                let v_sorted_0 = first_step.get_main_evaluation_element(0, 3);
                let v_sorted_1 = second_step.get_main_evaluation_element(0, 3);
                // (v'_{i+1} - v'_i) * (a'_{i+1} - a'_i - 1) = 0
                let res = (v_sorted_1 - v_sorted_0)
                    * (a_sorted_1 - a_sorted_0 - FieldElement::<F>::one());

                // The eval always exists, except if the constraint idx were incorrectly defined.
                if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx()) {
                    *eval = res.to_extension();
                }
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _periodic_values,
                rap_challenges: _rap_challenges,
                ..
            } => {
                let first_step = frame.get_evaluation_step(0);
                let second_step = frame.get_evaluation_step(1);

                let a_sorted_0 = first_step.get_main_evaluation_element(0, 2);
                let a_sorted_1 = second_step.get_main_evaluation_element(0, 2);
                let v_sorted_0 = first_step.get_main_evaluation_element(0, 3);
                let v_sorted_1 = second_step.get_main_evaluation_element(0, 3);
                // (v'_{i+1} - v'_i) * (a'_{i+1} - a'_i - 1) = 0
                let res = (v_sorted_1 - v_sorted_0)
                    * (a_sorted_1 - a_sorted_0 - FieldElement::<E>::one());

                // The eval always exists, except if the constraint idx were incorrectly defined.
                if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx()) {
                    *eval = res;
                }
            }
        }
    }
}
/// Transition constraint that ensures that the sorted columns are a permutation of the original ones.
/// We are using the LogUp construction described in:
/// <https://0xpolygonmiden.github.io/miden-vm/design/lookups/logup.html>.
/// See also our post of LogUp argument in blog.lambdaclass.com.
#[derive(Clone)]
struct PermutationConstraint<
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
> {
    phantom_f: PhantomData<F>,
    phantom_e: PhantomData<E>,
}

impl<F, E> PermutationConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    pub fn new() -> Self {
        Self {
            phantom_f: PhantomData::<F>,
            phantom_e: PhantomData::<E>,
        }
    }
}

impl<F, E> TransitionConstraintEvaluator<F, E> for PermutationConstraint<F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    fn degree(&self) -> usize {
        3
    }

    fn constraint_idx(&self) -> usize {
        2
    }

    fn end_exemptions(&self) -> usize {
        1
    }

    fn evaluate_verifier(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, E>,
        transition_evaluations: &mut [FieldElement<E>],
    ) {
        // In both evaluation contexts, Prover and Verfier will evaluate the transition polynomial in the same way.
        // The only difference is that the Prover's Frame has base field and field extension elements,
        // while the Verfier's Frame has only field extension elements.
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _periodic_values,
                rap_challenges,
                ..
            } => {
                let first_step = frame.get_evaluation_step(0);
                let second_step = frame.get_evaluation_step(1);

                // Auxiliary frame elements
                let s0 = first_step.get_aux_evaluation_element(0, 0);
                let s1 = second_step.get_aux_evaluation_element(0, 0);

                // Challenges
                let z = &rap_challenges[0];
                let alpha = &rap_challenges[1];

                // Main frame elements
                let a1 = second_step.get_main_evaluation_element(0, 0);
                let v1 = second_step.get_main_evaluation_element(0, 1);
                let a_sorted_1 = second_step.get_main_evaluation_element(0, 2);
                let v_sorted_1 = second_step.get_main_evaluation_element(0, 3);
                let m = second_step.get_main_evaluation_element(0, 4);

                let unsorted_term = -(a1 + v1 * alpha) + z;
                let sorted_term = -(a_sorted_1 + v_sorted_1 * alpha) + z;

                // We are using the following LogUp equation:
                // s1 = s0 + m / sorted_term - 1/unsorted_term.
                // Since constraints must be expressed without division, we multiply each term by sorted_term * unsorted_term:
                let res = s0 * &unsorted_term * &sorted_term + m * &unsorted_term
                    - &sorted_term
                    - s1 * unsorted_term * sorted_term;

                // The eval always exists, except if the constraint idx were incorrectly defined.
                if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx()) {
                    *eval = res;
                }
            }

            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _periodic_values,
                rap_challenges,
                ..
            } => {
                let first_step = frame.get_evaluation_step(0);
                let second_step = frame.get_evaluation_step(1);

                // Auxiliary frame elements
                let s0 = first_step.get_aux_evaluation_element(0, 0);
                let s1 = second_step.get_aux_evaluation_element(0, 0);

                // Challenges
                let z = &rap_challenges[0];
                let alpha = &rap_challenges[1];

                // Main frame elements
                let a1 = second_step.get_main_evaluation_element(0, 0);
                let v1 = second_step.get_main_evaluation_element(0, 1);
                let a_sorted_1 = second_step.get_main_evaluation_element(0, 2);
                let v_sorted_1 = second_step.get_main_evaluation_element(0, 3);
                let m = second_step.get_main_evaluation_element(0, 4);

                let unsorted_term = z - (a1 + alpha * v1);
                let sorted_term = z - (a_sorted_1 + alpha * v_sorted_1);

                // We are using the following LogUp equation:
                // s1 = s0 + m / sorted_term - 1/unsorted_term.
                // Since constraints must be expressed without division, we multiply each term by sorted_term * unsorted_term:
                let res = s0 * &unsorted_term * &sorted_term + m * &unsorted_term
                    - &sorted_term
                    - s1 * unsorted_term * sorted_term;

                // The eval always exists, except if the constraint idx were incorrectly defined.
                if let Some(eval) = transition_evaluations.get_mut(self.constraint_idx()) {
                    *eval = res;
                }
            }
        }
    }
}

/// AIR for a continuous read-only memory using the LogUp Lookup Argument.
/// To accompany the understanding of this code you can see corresponding post in blog.lambdaclass.com.
pub struct LogReadOnlyRAP<F, E>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    context: AirContext,
    transition_constraints: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>>,
}

#[derive(Clone, Debug)]
pub struct LogReadOnlyPublicInputs<F>
where
    F: IsFFTField + Send + Sync,
{
    pub a0: FieldElement<F>,
    pub v0: FieldElement<F>,
    pub a_sorted_0: FieldElement<F>,
    pub v_sorted_0: FieldElement<F>,
    // The multiplicity of (a_sorted_0, v_sorted_0)
    pub m0: FieldElement<F>,
}

impl<F, E> AIR for LogReadOnlyRAP<F, E>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: ByteConversion,
{
    type Field = F;
    type FieldExtension = E;
    type PublicInputs = LogReadOnlyPublicInputs<F>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(proof_options: &ProofOptions) -> Self {
        let transition_constraints: Vec<
            Box<dyn TransitionConstraintEvaluator<Self::Field, Self::FieldExtension>>,
        > = vec![
            Box::new(ContinuityConstraint::new()),
            Box::new(SingleValueConstraint::new()),
            Box::new(PermutationConstraint::new()),
        ];

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 6,
            transition_offsets: vec![0, 1],
            num_transition_constraints: transition_constraints.len(),
        };

        Self {
            context,
            transition_constraints,
        }
    }

    fn build_auxiliary_trace(
        &self,
        trace: &mut TraceTable<Self::Field, Self::FieldExtension>,
        challenges: &[FieldElement<E>],
    ) -> Option<crate::lookup::BusPublicInputs<Self::FieldExtension>> {
        // Main table
        let main_segment_cols = trace.columns_main();
        let a = &main_segment_cols[0];
        let v = &main_segment_cols[1];
        let a_sorted = &main_segment_cols[2];
        let v_sorted = &main_segment_cols[3];
        let m = &main_segment_cols[4];

        // Challenges
        let z = &challenges[0];
        let alpha = &challenges[1];

        let trace_len = trace.num_rows();
        let mut aux_col = Vec::new();

        // s_0 = m_0/(z - (a'_0 + α * v'_0) - 1/(z - (a_0 + α * v_0)
        let unsorted_term = (-(&a[0] + &v[0] * alpha) + z).inv().unwrap();
        let sorted_term = (-(&a_sorted[0] + &v_sorted[0] * alpha) + z).inv().unwrap();
        aux_col.push(&m[0] * sorted_term - unsorted_term);

        // Apply the same equation given in the permutation transition contraint to the rest of the trace.
        // s_{i+1} = s_i + m_{i+1}/(z - (a'_{i+1} + α * v'_{i+1}) - 1/(z - (a_{i+1} + α * v_{i+1})
        for i in 0..trace_len - 1 {
            let unsorted_term = (-(&a[i + 1] + &v[i + 1] * alpha) + z).inv().unwrap();
            let sorted_term = (-(&a_sorted[i + 1] + &v_sorted[i + 1] * alpha) + z)
                .inv()
                .unwrap();
            aux_col.push(&aux_col[i] + &m[i + 1] * sorted_term - unsorted_term);
        }

        for (i, aux_elem) in aux_col.iter().enumerate().take(trace.num_rows()) {
            trace.set_aux(i, 0, aux_elem.clone())
        }

        None
    }

    fn build_rap_challenges(
        &self,
        transcript: &mut dyn IsStarkTranscript<Self::FieldExtension, Self::Field>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        vec![
            transcript.sample_field_element(),
            transcript.sample_field_element(),
        ]
    }

    fn trace_layout(&self) -> (usize, usize) {
        (5, 1)
    }

    fn boundary_constraints(
        &self,
        pub_inputs: &Self::PublicInputs,
        rap_challenges: &[FieldElement<Self::FieldExtension>],
        _bus_public_inputs: Option<&crate::lookup::BusPublicInputs<Self::FieldExtension>>,
        trace_length: usize,
    ) -> BoundaryConstraints<Self::FieldExtension> {
        let a0 = &pub_inputs.a0;
        let v0 = &pub_inputs.v0;
        let a_sorted_0 = &pub_inputs.a_sorted_0;
        let v_sorted_0 = &pub_inputs.v_sorted_0;
        let m0 = &pub_inputs.m0;
        let z = &rap_challenges[0];
        let alpha = &rap_challenges[1];

        // Main boundary constraints
        let c1 = BoundaryConstraint::new_main(0, 0, a0.clone().to_extension());
        let c2 = BoundaryConstraint::new_main(1, 0, v0.clone().to_extension());
        let c3 = BoundaryConstraint::new_main(2, 0, a_sorted_0.clone().to_extension());
        let c4 = BoundaryConstraint::new_main(3, 0, v_sorted_0.clone().to_extension());
        let c5 = BoundaryConstraint::new_main(4, 0, m0.clone().to_extension());

        // Auxiliary boundary constraints
        let unsorted_term = (-(a0 + v0 * alpha) + z).inv().unwrap();
        let sorted_term = (-(a_sorted_0 + v_sorted_0 * alpha) + z).inv().unwrap();
        let p0_value = m0 * sorted_term - unsorted_term;

        let c_aux1 = BoundaryConstraint::new_aux(0, 0, p0_value);
        let c_aux2 = BoundaryConstraint::new_aux(
            0,
            trace_length - 1,
            FieldElement::<Self::FieldExtension>::zero(),
        );

        BoundaryConstraints::from_constraints(vec![c1, c2, c3, c4, c5, c_aux1, c_aux2])
    }

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraintEvaluator<Self::Field, Self::FieldExtension>>> {
        &self.transition_constraints
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    // The prover use this function to define the number of parts of the composition polynomial.
    // The number of parts will be: composition_poly_degree_bound() / trace_length().
    // Since we have a transition constraint of degree 3, we need the bound to be two times the trace length.
    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        trace_length * 2
    }
}

/// Return a trace table with an auxiliary column full of zeros (that will be then replaced
/// with the correct values by the air) and the following five main columns:
/// The original addresses and values, the sorted addresses and values without duplicates, and
/// the multiplicities of each sorted address and value in the original ones (i.e. how many times
/// they appear in the original address an value columns).
pub fn read_only_logup_trace<
    F: IsPrimeField + IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
>(
    addresses: Vec<FieldElement<F>>,
    values: Vec<FieldElement<F>>,
) -> TraceTable<F, E> {
    let mut address_value_pairs: Vec<_> = addresses.iter().zip(values.iter()).collect();
    address_value_pairs.sort_by_key(|(addr, _)| addr.canonical());

    let mut multiplicities = Vec::new();
    let mut sorted_addresses = Vec::new();
    let mut sorted_values = Vec::new();

    for (key, group) in &address_value_pairs.into_iter().group_by(|&(a, v)| (a, v)) {
        let group_vec: Vec<_> = group.collect();
        multiplicities.push(FieldElement::<F>::from(group_vec.len() as u64));
        sorted_addresses.push(key.0.clone());
        sorted_values.push(key.1.clone());
    }

    // We resize the sorted addresses and values with the last value of each one so they have the
    // same number of rows as the original addresses and values. However, their multiplicity should be zero.
    sorted_addresses.resize(addresses.len(), sorted_addresses.last().unwrap().clone());
    sorted_values.resize(addresses.len(), sorted_values.last().unwrap().clone());
    multiplicities.resize(addresses.len(), FieldElement::<F>::zero());

    let main_columns = vec![
        addresses.clone(),
        values.clone(),
        sorted_addresses,
        sorted_values,
        multiplicities,
    ];

    // create a vector with zeros of the same length as the main columns
    let zero_vec = vec![FieldElement::<E>::zero(); main_columns[0].len()];
    TraceTable::from_columns(main_columns, vec![zero_vec], 1)
}

#[cfg(test)]
mod test {
    use super::*;
    use math::field::{
        extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
    };

    type GoldilocksFE = FieldElement<GoldilocksField>;

    #[test]
    fn tes_logup_trace_construction() {
        let address_col = vec![
            GoldilocksFE::from(3u64),
            GoldilocksFE::from(7u64),
            GoldilocksFE::from(2u64),
            GoldilocksFE::from(8u64),
            GoldilocksFE::from(4u64),
            GoldilocksFE::from(5u64),
            GoldilocksFE::from(1u64),
            GoldilocksFE::from(6u64),
        ];
        let value_col = vec![
            GoldilocksFE::from(30u64),
            GoldilocksFE::from(70u64),
            GoldilocksFE::from(20u64),
            GoldilocksFE::from(80u64),
            GoldilocksFE::from(40u64),
            GoldilocksFE::from(50u64),
            GoldilocksFE::from(10u64),
            GoldilocksFE::from(60u64),
        ];

        let logup_trace: TraceTable<GoldilocksField, GoldilocksField> =
            read_only_logup_trace(address_col, value_col);

        let expected_sorted_addresses = vec![
            GoldilocksFE::from(1u64),
            GoldilocksFE::from(2u64),
            GoldilocksFE::from(3u64),
            GoldilocksFE::from(4u64),
            GoldilocksFE::from(5u64),
            GoldilocksFE::from(6u64),
            GoldilocksFE::from(7u64),
            GoldilocksFE::from(8u64),
        ];
        let expected_sorted_values = vec![
            GoldilocksFE::from(10u64),
            GoldilocksFE::from(20u64),
            GoldilocksFE::from(30u64),
            GoldilocksFE::from(40u64),
            GoldilocksFE::from(50u64),
            GoldilocksFE::from(60u64),
            GoldilocksFE::from(70u64),
            GoldilocksFE::from(80u64),
        ];
        let expected_multiplicities = vec![
            GoldilocksFE::one(),
            GoldilocksFE::one(),
            GoldilocksFE::one(),
            GoldilocksFE::one(),
            GoldilocksFE::one(),
            GoldilocksFE::one(),
            GoldilocksFE::one(),
            GoldilocksFE::one(),
        ];
        assert_eq!(logup_trace.columns_main()[2], expected_sorted_addresses);
        assert_eq!(logup_trace.columns_main()[3], expected_sorted_values);
        assert_eq!(logup_trace.columns_main()[4], expected_multiplicities);
    }

    #[test]
    fn test_logup_trace_construction_2() {
        let address_col = vec![
            FieldElement::<GoldilocksField>::from(3u64), // a0
            FieldElement::<GoldilocksField>::from(2u64), // a1
            FieldElement::<GoldilocksField>::from(2u64), // a2
            FieldElement::<GoldilocksField>::from(3u64), // a3
            FieldElement::<GoldilocksField>::from(4u64), // a4
            FieldElement::<GoldilocksField>::from(5u64), // a5
            FieldElement::<GoldilocksField>::from(1u64), // a6
            FieldElement::<GoldilocksField>::from(3u64), // a7
        ];
        let value_col = vec![
            FieldElement::<GoldilocksField>::from(30u64), // v0
            FieldElement::<GoldilocksField>::from(20u64), // v1
            FieldElement::<GoldilocksField>::from(20u64), // v2
            FieldElement::<GoldilocksField>::from(30u64), // v3
            FieldElement::<GoldilocksField>::from(40u64), // v4
            FieldElement::<GoldilocksField>::from(50u64), // v5
            FieldElement::<GoldilocksField>::from(10u64), // v6
            FieldElement::<GoldilocksField>::from(30u64), // v7
        ];

        let sorted_address_col = vec![
            FieldElement::<GoldilocksField>::from(1u64), // a0
            FieldElement::<GoldilocksField>::from(2u64), // a1
            FieldElement::<GoldilocksField>::from(3u64), // a2
            FieldElement::<GoldilocksField>::from(4u64), // a3
            FieldElement::<GoldilocksField>::from(5u64), // a4
            FieldElement::<GoldilocksField>::from(5u64), // a5
            FieldElement::<GoldilocksField>::from(5u64), // a6
            FieldElement::<GoldilocksField>::from(5u64), // a7
        ];
        let sorted_value_col = vec![
            FieldElement::<GoldilocksField>::from(10u64), // v0
            FieldElement::<GoldilocksField>::from(20u64), // v1
            FieldElement::<GoldilocksField>::from(30u64), // v2
            FieldElement::<GoldilocksField>::from(40u64), // v3
            FieldElement::<GoldilocksField>::from(50u64), // v4
            FieldElement::<GoldilocksField>::from(50u64), // v5
            FieldElement::<GoldilocksField>::from(50u64), // v6
            FieldElement::<GoldilocksField>::from(50u64), // v7
        ];

        let multiplicity_col = vec![
            FieldElement::<GoldilocksField>::from(1u64), // v0
            FieldElement::<GoldilocksField>::from(2u64), // v1
            FieldElement::<GoldilocksField>::from(3u64), // v2
            FieldElement::<GoldilocksField>::from(1u64), // v3
            FieldElement::<GoldilocksField>::from(1u64), // v4
            FieldElement::<GoldilocksField>::from(0u64), // v5
            FieldElement::<GoldilocksField>::from(0u64), // v6
            FieldElement::<GoldilocksField>::from(0u64), // v7
        ];
        let logup_trace: TraceTable<GoldilocksField, Degree3GoldilocksExtensionField> =
            read_only_logup_trace(address_col, value_col);

        assert_eq!(logup_trace.columns_main()[2], sorted_address_col);
        assert_eq!(logup_trace.columns_main()[3], sorted_value_col);
        assert_eq!(logup_trace.columns_main()[4], multiplicity_col);
    }
}
