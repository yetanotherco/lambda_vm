use super::boundary::BoundaryConstraints;
#[cfg(all(debug_assertions, not(feature = "parallel")))]
use crate::debug::check_boundary_polys_divisibility;
use crate::domain::Domain;
use crate::lookup::BusPublicInputs;
use crate::trace::LDETraceTable;
use crate::traits::{AIR, TransitionEvaluationContext};
use crate::{frame::Frame, prover::evaluate_polynomial_on_lde_domain};
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
#[cfg(not(feature = "parallel"))]
use math::polynomial::Polynomial;
use math::{fft::errors::FFTError, field::element::FieldElement};
#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use std::marker::PhantomData;
#[cfg(feature = "instruments")]
use std::time::Instant;

pub struct ConstraintEvaluator<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
    PI,
> {
    boundary_constraints: BoundaryConstraints<FieldExtension>,
    phantom: PhantomData<(Field, PI)>,
}
impl<Field, FieldExtension, PI> ConstraintEvaluator<Field, FieldExtension, PI>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
{
    pub fn new(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        pub_inputs: &PI,
        rap_challenges: &[FieldElement<FieldExtension>],
        bus_public_inputs: Option<&BusPublicInputs<FieldExtension>>,
        trace_length: usize,
    ) -> Self {
        let boundary_constraints =
            air.boundary_constraints(pub_inputs, rap_challenges, bus_public_inputs, trace_length);

        Self {
            boundary_constraints,
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
        let boundary_zerofiers_inverse_evaluations: Vec<Vec<FieldElement<Field>>> =
            boundary_constraints
                .constraints
                .iter()
                .map(|bc| {
                    let point = &domain.trace_primitive_root.pow(bc.step as u64);
                    let mut evals = domain
                        .lde_roots_of_unity_coset
                        .iter()
                        .map(|v| v - point)
                        .collect::<Vec<FieldElement<Field>>>();
                    FieldElement::inplace_batch_inverse(&mut evals).unwrap();
                    evals
                })
                .collect::<Vec<Vec<FieldElement<Field>>>>();

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let boundary_polys: Vec<Polynomial<FieldElement<Field>>> = Vec::new();

        #[cfg(feature = "instruments")]
        let timer = Instant::now();

        let trace_length = domain.interpolation_domain_size;
        let lde_periodic_columns = air
            .get_periodic_column_polynomials(trace_length)
            .iter()
            .map(|poly| {
                evaluate_polynomial_on_lde_domain(
                    poly,
                    domain.blowup_factor,
                    domain.interpolation_domain_size,
                    &domain.coset_offset,
                )
            })
            .collect::<Result<Vec<Vec<FieldElement<Field>>>, FFTError>>()
            .unwrap();

        #[cfg(feature = "instruments")]
        println!(
            "     Evaluating periodic columns on lde: {:#?}",
            timer.elapsed()
        );

        // NOTE: We intentionally skip precomputing boundary_polys_evaluations and
        // boundary_evaluation vectors. Instead, we compute boundary contributions
        // on-the-fly in the main loop below. This saves ~1GB of intermediate memory
        // for large traces.

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let boundary_zerofiers = Vec::new();

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        check_boundary_polys_divisibility(boundary_polys, boundary_zerofiers);

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let mut transition_evaluations = Vec::new();

        #[cfg(feature = "instruments")]
        let timer = Instant::now();
        let zerofiers_evals = air.transition_zerofier_evaluations(domain);
        #[cfg(feature = "instruments")]
        println!(
            "     Evaluated transition zerofiers: {:#?}",
            timer.elapsed()
        );

        // Iterate over all LDE domain and compute both boundary and transition
        // contributions to the composition polynomial in a single pass.
        // This fused approach avoids allocating large intermediate vectors.

        #[cfg(feature = "instruments")]
        let timer = Instant::now();
        let evaluations_t_iter = 0..domain.lde_roots_of_unity_coset.len();

        #[cfg(feature = "parallel")]
        let evaluations_t_iter = evaluations_t_iter.into_par_iter();

        let boundary_constraints_vec = &boundary_constraints.constraints;

        let evaluations_t = evaluations_t_iter
            .map(|i| {
                // Compute boundary contribution on-the-fly (fused from previous separate passes)
                let boundary = boundary_constraints_vec
                    .iter()
                    .zip(boundary_coefficients)
                    .enumerate()
                    .fold(
                        FieldElement::zero(),
                        |acc, (constraint_index, (constraint, beta))| {
                            // Get trace value at this position and compute (trace - boundary_value)
                            let poly_eval: FieldElement<FieldExtension> = if constraint.is_aux {
                                lde_trace.get_aux(i, constraint.col) - &constraint.value
                            } else {
                                lde_trace.get_main(i, constraint.col) - &constraint.value
                            };
                            // Multiply by zerofier_inverse and beta coefficient
                            acc + &boundary_zerofiers_inverse_evaluations[constraint_index][i]
                                * beta
                                * poly_eval
                        },
                    );

                let frame = Frame::read_from_lde(lde_trace, i, &air.context().transition_offsets);

                let periodic_values: Vec<_> = lde_periodic_columns
                    .iter()
                    .map(|col| col[i].clone())
                    .collect();

                // Compute all the transition constraints at this point of the LDE domain.
                let transition_evaluation_context = TransitionEvaluationContext::new_prover(
                    &frame,
                    &periodic_values,
                    rap_challenges,
                );
                let evaluations_transition = air.compute_transition(&transition_evaluation_context);

                #[cfg(all(debug_assertions, not(feature = "parallel")))]
                transition_evaluations.push(evaluations_transition.clone());

                // Add each term of the transition constraints to the composition polynomial, including the zerofier,
                // the challenge and the exemption polynomial if it is necessary.
                let acc_transition = itertools::izip!(
                    evaluations_transition,
                    &zerofiers_evals,
                    transition_coefficients
                )
                .fold(FieldElement::zero(), |acc, (eval, zerof_eval, beta)| {
                    // Zerofier evaluations are cyclical, so we only calculate one cycle.
                    // This means that here we have to wrap around
                    // Ex: Suppose the full zerofier vector is Z = [1,2,3,1,2,3]
                    // we will instead have calculated Z' = [1,2,3]
                    // Now if you need Z[4] this is equal to Z'[1]
                    let wrapped_idx = i % zerof_eval.len();
                    acc + &zerof_eval[wrapped_idx] * eval * beta
                });

                acc_transition + boundary
            })
            .collect();

        #[cfg(feature = "instruments")]
        println!(
            "     Evaluated transitions and accumulated results: {:#?}",
            timer.elapsed()
        );

        evaluations_t
    }
}
