use super::boundary::BoundaryConstraints;
#[cfg(all(debug_assertions, not(feature = "parallel")))]
use crate::debug::check_boundary_polys_divisibility;
use crate::domain::Domain;
use crate::trace::LDETraceTable;
use crate::traits::{AIR, TransitionEvaluationContext};
use crate::{frame::Frame, prover::evaluate_polynomial_on_lde_domain};
use itertools::Itertools;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
#[cfg(not(feature = "parallel"))]
use math::polynomial::Polynomial;
use math::{fft::errors::FFTError, field::element::FieldElement};
#[cfg(feature = "parallel")]
use rayon::{
    iter::IndexedParallelIterator,
    prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
};

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
        rap_challenges: &[FieldElement<FieldExtension>],
    ) -> Self {
        let boundary_constraints = air.boundary_constraints(rap_challenges);

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
        let number_of_b_constraints = boundary_constraints.constraints.len();

        // Parallelize boundary zerofier inverse evaluations
        #[cfg(feature = "parallel")]
        let boundary_constraints_iter = boundary_constraints.constraints.par_iter();
        #[cfg(not(feature = "parallel"))]
        let boundary_constraints_iter = boundary_constraints.constraints.iter();

        let boundary_zerofiers_inverse_evaluations: Vec<Vec<FieldElement<Field>>> =
            boundary_constraints_iter
                .map(|bc| {
                    let point = domain.trace_primitive_root.pow(bc.step as u64);
                    let mut evals: Vec<FieldElement<Field>> = domain
                        .lde_roots_of_unity_coset
                        .iter()
                        .map(|v| v - &point)
                        .collect();
                    FieldElement::inplace_batch_inverse(&mut evals).unwrap();
                    evals
                })
                .collect();

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let boundary_polys: Vec<Polynomial<FieldElement<Field>>> = Vec::new();

        #[cfg(feature = "instruments")]
        let timer = Instant::now();

        // Parallelize periodic column evaluations
        let periodic_polys = air.get_periodic_column_polynomials();
        #[cfg(feature = "parallel")]
        let periodic_polys_iter = periodic_polys.par_iter();
        #[cfg(not(feature = "parallel"))]
        let periodic_polys_iter = periodic_polys.iter();

        let lde_periodic_columns = periodic_polys_iter
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

        // Pre-transpose periodic columns to row-major format to avoid per-iteration allocation
        // lde_periodic_columns is column-major: Vec<Vec<FieldElement>> where outer = columns
        // lde_periodic_rows is row-major: Vec<Vec<FieldElement>> where outer = rows
        let num_periodic_cols = lde_periodic_columns.len();
        let lde_periodic_rows: Vec<Vec<FieldElement<Field>>> = if num_periodic_cols > 0 {
            let num_rows = lde_periodic_columns[0].len();
            (0..num_rows)
                .map(|row_idx| {
                    lde_periodic_columns
                        .iter()
                        .map(|col| col[row_idx].clone())
                        .collect()
                })
                .collect()
        } else {
            vec![vec![]; domain.lde_roots_of_unity_coset.len()]
        };

        #[cfg(feature = "instruments")]
        println!(
            "     Evaluating periodic columns on lde: {:#?}",
            timer.elapsed()
        );

        #[cfg(feature = "instruments")]
        let timer = Instant::now();

        // Parallelize boundary polynomial evaluations
        #[cfg(feature = "parallel")]
        let boundary_polys_iter = boundary_constraints.constraints.par_iter();
        #[cfg(not(feature = "parallel"))]
        let boundary_polys_iter = boundary_constraints.constraints.iter();

        let boundary_polys_evaluations: Vec<Vec<FieldElement<FieldExtension>>> = boundary_polys_iter
            .map(|constraint| {
                if constraint.is_aux {
                    (0..lde_trace.num_rows())
                        .map(|row| {
                            let v = lde_trace.get_aux(row, constraint.col);
                            v - &constraint.value
                        })
                        .collect_vec()
                } else {
                    (0..lde_trace.num_rows())
                        .map(|row| {
                            let v = lde_trace.get_main(row, constraint.col);
                            v - &constraint.value
                        })
                        .collect_vec()
                }
            })
            .collect();

        #[cfg(feature = "instruments")]
        println!("     Created boundary polynomials: {:#?}", timer.elapsed());
        #[cfg(feature = "instruments")]
        let timer = Instant::now();

        #[cfg(feature = "parallel")]
        let boundary_eval_iter = (0..domain.lde_roots_of_unity_coset.len()).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let boundary_eval_iter = 0..domain.lde_roots_of_unity_coset.len();

        let boundary_evaluation: Vec<_> = boundary_eval_iter
            .map(|domain_index| {
                (0..number_of_b_constraints)
                    .zip(boundary_coefficients)
                    .fold(FieldElement::zero(), |acc, (constraint_index, beta)| {
                        acc + &boundary_zerofiers_inverse_evaluations[constraint_index]
                            [domain_index]
                            * beta
                            * &boundary_polys_evaluations[constraint_index][domain_index]
                    })
            })
            .collect();

        #[cfg(feature = "instruments")]
        println!(
            "     Evaluated boundary polynomials on LDE: {:#?}",
            timer.elapsed()
        );

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

        // Iterate over all LDE domain and compute the part of the composition polynomial
        // related to the transition constraints and add it to the already computed part of the
        // boundary constraints.

        #[cfg(feature = "instruments")]
        let timer = Instant::now();
        let evaluations_t_iter = 0..domain.lde_roots_of_unity_coset.len();

        #[cfg(feature = "parallel")]
        let boundary_evaluation = boundary_evaluation.into_par_iter();
        #[cfg(feature = "parallel")]
        let evaluations_t_iter = evaluations_t_iter.into_par_iter();

        let evaluations_t = evaluations_t_iter
            .zip(boundary_evaluation)
            .map(|(i, boundary)| {
                let frame = Frame::read_from_lde(lde_trace, i, &air.context().transition_offsets);

                // Use pre-transposed row-major periodic values (no allocation per iteration)
                let periodic_values = &lde_periodic_rows[i];

                // Compute all the transition constraints at this point of the LDE domain.
                let transition_evaluation_context = TransitionEvaluationContext::new_prover(
                    &frame,
                    periodic_values,
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
