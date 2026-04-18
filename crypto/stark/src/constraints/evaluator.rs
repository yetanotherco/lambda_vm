use super::boundary::BoundaryConstraints;
#[cfg(all(debug_assertions, not(feature = "parallel")))]
use crate::debug::check_boundary_polys_divisibility;
use crate::domain::Domain;
use crate::lookup::{BusPublicInputs, LOGUP_CHALLENGE_ALPHA, compute_alpha_powers};
use crate::trace::LDETraceTable;
use crate::traits::{AIR, ZerofierEvaluations};
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
#[cfg(not(feature = "parallel"))]
use math::polynomial::Polynomial;
#[cfg(feature = "parallel")]
use rayon::{
    iter::IndexedParallelIterator,
    prelude::{IntoParallelIterator, ParallelIterator},
};

use std::marker::PhantomData;

pub struct ConstraintEvaluator<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
    PI,
> {
    boundary_constraints: BoundaryConstraints<FieldExtension>,
    logup_table_offset: FieldElement<FieldExtension>,
    phantom: PhantomData<(Field, PI)>,
}
impl<Field, FieldExtension, PI> ConstraintEvaluator<Field, FieldExtension, PI>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
{
    /// Evaluate transition + boundary constraints across the entire LDE domain.
    ///
    /// Uses the AirBuilder pattern: each AIR's `eval_constraints_with_builder()`
    /// accumulates constraint evaluations with alpha powers into a single sum,
    /// which is then multiplied by the uniform zerofier.
    fn evaluate_transitions(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        rap_challenges: &[FieldElement<FieldExtension>],
        zerofier_data: &ZerofierEvaluations<Field>,
        boundary_evaluation: Vec<FieldElement<FieldExtension>>,
        logup_table_offset: &FieldElement<FieldExtension>,
        composition_alpha: &FieldElement<FieldExtension>,
    ) -> Vec<FieldElement<FieldExtension>> {
        assert!(
            zerofier_data.is_uniform(),
            "AirBuilder requires uniform zerofiers"
        );

        // Pre-compute LogUp alpha powers once for all LDE domain points.
        let logup_alpha_powers: Vec<FieldElement<FieldExtension>> =
            if rap_challenges.len() > LOGUP_CHALLENGE_ALPHA {
                compute_alpha_powers(
                    &rap_challenges[LOGUP_CHALLENGE_ALPHA],
                    air.max_bus_elements(),
                )
            } else {
                Vec::new()
            };

        #[cfg(feature = "parallel")]
        {
            let evaluations_t: Vec<_> = boundary_evaluation
                .into_par_iter()
                .enumerate()
                .map(|(i, boundary)| {
                    let mut builder = crate::air_builder::ProverBuilder::new(
                        lde_trace,
                        i,
                        composition_alpha,
                        rap_challenges,
                        &logup_alpha_powers,
                        logup_table_offset,
                    );
                    air.eval_constraints_with_builder(&mut builder);
                    zerofier_data.get_uniform(i) * &builder.finish() + boundary
                })
                .collect();
            evaluations_t
        }

        #[cfg(not(feature = "parallel"))]
        {
            boundary_evaluation
                .into_iter()
                .enumerate()
                .map(|(i, boundary)| {
                    let mut builder = crate::air_builder::ProverBuilder::new(
                        lde_trace,
                        i,
                        composition_alpha,
                        rap_challenges,
                        &logup_alpha_powers,
                        logup_table_offset,
                    );
                    air.eval_constraints_with_builder(&mut builder);
                    zerofier_data.get_uniform(i) * &builder.finish() + boundary
                })
                .collect()
        }
    }

    pub fn new(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        pub_inputs: &PI,
        rap_challenges: &[FieldElement<FieldExtension>],
        bus_public_inputs: Option<&BusPublicInputs<FieldExtension>>,
        trace_length: usize,
    ) -> Self {
        let boundary_constraints =
            air.boundary_constraints(pub_inputs, rap_challenges, bus_public_inputs, trace_length);

        // Compute logup_table_offset = table_contribution / trace_length
        let logup_table_offset = match bus_public_inputs {
            Some(bpi) => {
                // trace_length is always a power of two >= 2, so inv() cannot fail
                let n_inv = FieldElement::<Field>::from(trace_length as u64)
                    .inv()
                    .unwrap();
                n_inv * &bpi.table_contribution
            }
            None => FieldElement::zero(),
        };

        Self {
            boundary_constraints,
            logup_table_offset,
            phantom: PhantomData::<(Field, PI)> {},
        }
    }

    pub(crate) fn evaluate(
        &self,
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        transition_coefficients: &[FieldElement<FieldExtension>],
        boundary_coefficients: &[FieldElement<FieldExtension>],
        rap_challenges: &[FieldElement<FieldExtension>],
    ) -> Vec<FieldElement<FieldExtension>> {
        let boundary_constraints = &self.boundary_constraints;
        let mut boundary_step_points: Vec<(usize, FieldElement<Field>)> = Vec::new();
        let boundary_zerofiers_inverse_evaluations: Vec<Vec<FieldElement<Field>>> =
            boundary_constraints
                .constraints
                .iter()
                .map(|bc| {
                    let point = match boundary_step_points.iter().find(|(s, _)| *s == bc.step) {
                        Some((_, p)) => p.clone(),
                        None => {
                            let p = domain.trace_primitive_root.pow(bc.step as u64);
                            boundary_step_points.push((bc.step, p.clone()));
                            p
                        }
                    };
                    let mut evals = domain
                        .lde_roots_of_unity_coset
                        .iter()
                        .map(|v| v - &point)
                        .collect::<Vec<FieldElement<Field>>>();
                    FieldElement::inplace_batch_inverse(&mut evals).unwrap();
                    evals
                })
                .collect::<Vec<Vec<FieldElement<Field>>>>();

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let boundary_polys: Vec<Polynomial<FieldElement<Field>>> = Vec::new();

        // Fused boundary evaluation: compute (trace[col] - value) on-the-fly
        // instead of pre-computing all boundary_polys_evaluations.
        // This eliminates N_constraints x LDE_size intermediate allocations.
        #[cfg(feature = "parallel")]
        let boundary_eval_iter = (0..domain.lde_roots_of_unity_coset.len()).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let boundary_eval_iter = 0..domain.lde_roots_of_unity_coset.len();

        let b_constraints = &boundary_constraints.constraints;
        let boundary_evaluation: Vec<_> = boundary_eval_iter
            .map(|domain_index| {
                b_constraints
                    .iter()
                    .zip(boundary_coefficients)
                    .zip(boundary_zerofiers_inverse_evaluations.iter())
                    .fold(
                        FieldElement::zero(),
                        |acc, ((constraint, beta), zerofier_inv)| {
                            let bp = if constraint.is_aux {
                                lde_trace.get_aux(domain_index, constraint.col) - &constraint.value
                            } else {
                                lde_trace.get_main(domain_index, constraint.col) - &constraint.value
                            };
                            acc + &zerofier_inv[domain_index] * beta * bp
                        },
                    )
            })
            .collect();

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let boundary_zerofiers = Vec::new();

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        check_boundary_polys_divisibility(boundary_polys, boundary_zerofiers);

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let _transition_evaluations: Vec<FieldElement<FieldExtension>> = Vec::new();

        let zerofier_data = air.transition_zerofier_evaluations_grouped(domain);

        // Extract the raw composition alpha from the pre-computed powers [1, beta, beta^2, ...].
        // The ProverBuilder generates its own powers internally via assert_zero().
        let composition_alpha = if transition_coefficients.len() >= 2 {
            transition_coefficients[1].clone()
        } else {
            FieldElement::one()
        };

        Self::evaluate_transitions(
            air,
            lde_trace,
            rap_challenges,
            &zerofier_data,
            boundary_evaluation,
            &self.logup_table_offset,
            &composition_alpha,
        )
    }
}
