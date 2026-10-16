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
use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::field::traits::IsPrimeField;
use math::{
    field::{element::FieldElement, traits::IsFFTField},
    traits::ByteConversion,
};

/// Single-body [`ConstraintSet`] for [`ReadOnlyRAP`]: the continuity,
/// single-value and permutation constraints, written once against the
/// [`ConstraintBuilder`]. The permutation constraint reads the auxiliary
/// (RAP) column and the interaction challenges, so it is an `Ext` constraint
/// after the `Base` prefix.
pub struct ReadOnlyRAPConstraints;

impl<F> ConstraintSet<F, F> for ReadOnlyRAPConstraints
where
    F: IsFFTField + Send + Sync,
{
    fn eval<B: ConstraintBuilder<F, F>>(&self, b: &mut B) {
        let a_sorted_0 = b.main(0, 2);
        let a_sorted_1 = b.main(1, 2);
        let v_sorted_0 = b.main(0, 3);
        let v_sorted_1 = b.main(1, 3);
        let one = b.one();
        let addr_diff = a_sorted_1 - a_sorted_0;

        // All three read the next row ⇒ degree 2, 1 end exemption each.
        // idx 0 — continuity: (a'_{i+1} - a'_i)(a'_{i+1} - a'_i - 1) = 0 where a' is the sorted address
        b.emit_base_rows(
            0,
            RowDomain::except_last(1),
            addr_diff.clone() * (addr_diff.clone() - one.clone()),
        );
        // idx 1 — single value: (v'_{i+1} - v'_i) * (a'_{i+1} - a'_i - 1) = 0
        b.emit_base_rows(
            1,
            RowDomain::except_last(1),
            (v_sorted_1 - v_sorted_0) * (addr_diff - one),
        );

        // (z - (a'_{i+1} + α * v'_{i+1})) * p_{i+1} = (z - (a_{i+1} + α * v_{i+1})) * p_i
        let p0 = b.aux(0, 0);
        let p1 = b.aux(1, 0);
        let z = b.challenge(0);
        let alpha = b.challenge(1);
        let a1 = b.main(1, 0);
        let v1 = b.main(1, 1);
        let a_sorted_1 = b.main(1, 2);
        let v_sorted_1 = b.main(1, 3);
        let sorted_fp = z.clone() - (a_sorted_1 + v_sorted_1 * alpha.clone());
        let unsorted_fp = z - (a1 + v1 * alpha);
        // idx 2 — permutation (degree 2, 1 end exemption).
        b.emit_ext_rows(
            2,
            RowDomain::except_last(1),
            sorted_fp * p1 - unsorted_fp * p0,
        );
    }
}

pub struct ReadOnlyRAP<F>
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
pub struct ReadOnlyPublicInputs<F>
where
    F: IsFFTField,
{
    pub a0: FieldElement<F>,
    pub v0: FieldElement<F>,
    pub a_sorted0: FieldElement<F>,
    pub v_sorted0: FieldElement<F>,
}

impl<F> AIR for ReadOnlyRAP<F>
where
    F: IsFFTField + Send + Sync + 'static,
    FieldElement<F>: ByteConversion,
{
    type Field = F;
    type FieldExtension = F;
    type PublicInputs = ReadOnlyPublicInputs<F>;

    fn step_size(&self) -> usize {
        1
    }

    fn new(proof_options: &ProofOptions) -> Self {
        let meta = ConstraintSet::<F, F>::meta(&ReadOnlyRAPConstraints);

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 5,
            transition_offsets: vec![0, 1],
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
        let a = &main_segment_cols[0];
        let v = &main_segment_cols[1];
        let a_sorted = &main_segment_cols[2];
        let v_sorted = &main_segment_cols[3];
        let z = &challenges[0];
        let alpha = &challenges[1];

        let trace_len = trace.num_rows();

        let mut aux_col = Vec::new();
        let num = z - (&a[0] + alpha * &v[0]);
        let den = z - (&a_sorted[0] + alpha * &v_sorted[0]);
        // We are using that den != 0 with high probability because alpha is a random element.
        aux_col.push((num / den).unwrap());
        // Apply the same equation given in the permutation case to the rest of the trace
        for i in 0..trace_len - 1 {
            let num = (z - (&a[i + 1] + alpha * &v[i + 1])) * &aux_col[i];
            let den = z - (&a_sorted[i + 1] + alpha * &v_sorted[i + 1]);
            // We are using that den != 0 with high probability because alpha is a random element.
            aux_col.push((num / den).unwrap());
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
        vec![
            transcript.sample_field_element(),
            transcript.sample_field_element(),
        ]
    }

    fn trace_layout(&self) -> (usize, usize) {
        (4, 1)
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
        let a_sorted0 = &pub_inputs.a_sorted0;
        let v_sorted0 = &pub_inputs.v_sorted0;
        let z = &rap_challenges[0];
        let alpha = &rap_challenges[1];

        // Main boundary constraints
        let c1 = BoundaryConstraint::new_main(0, 0, a0.clone());
        let c2 = BoundaryConstraint::new_main(1, 0, v0.clone());
        let c3 = BoundaryConstraint::new_main(2, 0, a_sorted0.clone());
        let c4 = BoundaryConstraint::new_main(3, 0, v_sorted0.clone());

        // Auxiliary boundary constraints
        let num = z - (a0 + alpha * v0);
        let den = z - (a_sorted0 + alpha * v_sorted0);
        let p0_value = num / den;

        let c_aux1 = BoundaryConstraint::new_aux(0, 0, p0_value.unwrap());
        let c_aux2 = BoundaryConstraint::new_aux(
            0,
            trace_length - 1,
            FieldElement::<Self::FieldExtension>::one(),
        );

        BoundaryConstraints::from_constraints(vec![c1, c2, c3, c4, c_aux1, c_aux2])
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
            &ReadOnlyRAPConstraints,
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
            &ReadOnlyRAPConstraints,
            evaluation_context,
            self.num_base_transition_constraints(),
            self.num_transition_constraints(),
        )
    }

    fn num_base_transition_constraints(&self) -> usize {
        num_base_from_meta(&ConstraintSet::<F, F>::meta(&ReadOnlyRAPConstraints))
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn composition_poly_degree_bound(&self, trace_length: usize) -> usize {
        trace_length
    }
}

/// Given the adress and value columns, it returns the trace table with 5 columns, which are:
/// Addres, Value, Adress Sorted, Value Sorted and a Column of Zeroes (where we'll insert the auxiliary colunn).
pub fn sort_rap_trace<F: IsFFTField + IsPrimeField>(
    address: Vec<FieldElement<F>>,
    value: Vec<FieldElement<F>>,
) -> TraceTable<F, F> {
    let mut address_value_pairs: Vec<_> = address.iter().zip(value.iter()).collect();

    address_value_pairs.sort_by_key(|(addr, _)| addr.canonical());

    let (sorted_address, sorted_value): (Vec<FieldElement<F>>, Vec<FieldElement<F>>) =
        address_value_pairs
            .into_iter()
            .map(|(addr, val)| (addr.clone(), val.clone()))
            .unzip();
    let main_columns = vec![address.clone(), value.clone(), sorted_address, sorted_value];
    // create a vector with zeros of the same length as the main columns
    let zero_vec = vec![FieldElement::<F>::zero(); main_columns[0].len()];
    TraceTable::from_columns(main_columns, vec![zero_vec], 1)
}

#[cfg(test)]
mod test {
    use super::*;
    use math::field::goldilocks::GoldilocksField;

    type GoldilocksFE = FieldElement<GoldilocksField>;

    #[test]
    fn test_sort_rap_trace() {
        let address_col = vec![
            GoldilocksFE::from(5u64),
            GoldilocksFE::from(2u64),
            GoldilocksFE::from(3u64),
            GoldilocksFE::from(4u64),
            GoldilocksFE::from(1u64),
            GoldilocksFE::from(6u64),
            GoldilocksFE::from(7u64),
            GoldilocksFE::from(8u64),
        ];
        let value_col = vec![
            GoldilocksFE::from(50u64),
            GoldilocksFE::from(20u64),
            GoldilocksFE::from(30u64),
            GoldilocksFE::from(40u64),
            GoldilocksFE::from(10u64),
            GoldilocksFE::from(60u64),
            GoldilocksFE::from(70u64),
            GoldilocksFE::from(80u64),
        ];

        let sorted_trace = sort_rap_trace(address_col.clone(), value_col.clone());

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

        assert_eq!(sorted_trace.columns_main()[2], expected_sorted_addresses);
        assert_eq!(sorted_trace.columns_main()[3], expected_sorted_values);
    }
}
