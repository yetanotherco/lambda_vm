use super::boundary::BoundaryConstraints;
#[cfg(all(debug_assertions, not(feature = "parallel")))]
use crate::debug::check_boundary_polys_divisibility;
use crate::domain::Domain;
use crate::lookup::{BusPublicInputs, LOGUP_CHALLENGE_ALPHA, compute_alpha_powers};
use crate::trace::LDETraceTable;
use crate::traits::{AIR, TransitionEvaluationContext, ZerofierEvaluations};
use crate::{frame::Frame, prover::evaluate_polynomial_on_lde_domain};
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
#[cfg(not(feature = "parallel"))]
use math::polynomial::Polynomial;
use math::{fft::errors::FFTError, field::element::FieldElement};
#[cfg(feature = "parallel")]
use rayon::{
    iter::IndexedParallelIterator,
    prelude::{IntoParallelIterator, ParallelIterator},
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
    /// Uses `map_init` for per-thread buffer reuse (transition evaluations + periodic values)
    /// and `ZerofierEvaluations` for deduplicated zerofier access.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_transitions(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        lde_periodic_columns: &[Vec<FieldElement<Field>>],
        rap_challenges: &[FieldElement<FieldExtension>],
        zerofier_data: &ZerofierEvaluations<Field>,
        transition_coefficients: &[FieldElement<FieldExtension>],
        boundary_evaluation: Vec<FieldElement<FieldExtension>>,
        num_transition: usize,
        num_periodic: usize,
        offsets: &[usize],
        logup_table_offset: &FieldElement<FieldExtension>,
    ) -> Vec<FieldElement<FieldExtension>> {
        let is_uniform = zerofier_data.is_uniform();

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

        // Per-thread buffers via map_init: each Rayon worker allocates once,
        // then reuses for all iterations assigned to that thread.
        //
        // When rows_per_step == 1 (all VM tables), the Frame uses LDE base pointers
        // for zero-copy access — only the row offset is updated per LDE point,
        // eliminating all element copies. For step_size > 1, we fall back to the
        // copy-based path.
        let blowup_factor = lde_trace.blowup_factor;
        let lde_step_size = lde_trace.lde_step_size;
        let rows_per_step = lde_step_size / blowup_factor;
        let num_rows = lde_trace.num_rows();
        let num_offsets = offsets.len();
        let use_zero_copy = rows_per_step == 1;

        #[cfg(feature = "parallel")]
        {
            let num_main_cols = lde_trace.num_main_cols();
            let num_aux_cols = lde_trace.num_aux_cols();

            let evaluations_t: Vec<_> = boundary_evaluation
                .into_par_iter()
                .enumerate()
                .map_init(
                    || {
                        let frame = if use_zero_copy {
                            Frame::preallocate_lde(lde_trace, num_offsets)
                        } else {
                            Frame::preallocate(
                                num_offsets,
                                rows_per_step,
                                num_main_cols,
                                num_aux_cols,
                            )
                        };
                        (
                            vec![FieldElement::<FieldExtension>::zero(); num_transition],
                            vec![FieldElement::<Field>::zero(); num_periodic],
                            frame,
                        )
                    },
                    |(transition_buf, periodic_buf, frame), (i, boundary)| {
                        if use_zero_copy {
                            frame.bind_to_lde(i, num_rows, lde_step_size, offsets);
                        } else {
                            frame.fill_from_lde(lde_trace, i, offsets);
                        }

                        for (j, col) in lde_periodic_columns.iter().enumerate() {
                            periodic_buf[j] = col[i].clone();
                        }

                        let ctx = TransitionEvaluationContext::new_prover(
                            frame,
                            periodic_buf,
                            rap_challenges,
                            &logup_alpha_powers,
                            logup_table_offset,
                        );
                        air.compute_transition_into(&ctx, transition_buf);

                        let acc_transition = if is_uniform {
                            // All constraints share one zerofier: factor it out of the sum.
                            let z = zerofier_data.get_uniform(i);
                            let sum = transition_buf
                                .iter()
                                .zip(transition_coefficients)
                                .fold(FieldElement::zero(), |acc, (eval, beta)| acc + eval * beta);
                            z * &sum
                        } else {
                            transition_buf
                                .iter()
                                .enumerate()
                                .zip(transition_coefficients)
                                .fold(FieldElement::zero(), |acc, ((c_idx, eval), beta)| {
                                    acc + zerofier_data.get(c_idx, i) * eval * beta
                                })
                        };

                        acc_transition + boundary
                    },
                )
                .collect();
            evaluations_t
        }

        #[cfg(not(feature = "parallel"))]
        {
            let num_main_cols = lde_trace.num_main_cols();
            let num_aux_cols = lde_trace.num_aux_cols();

            let mut transition_buf = vec![FieldElement::<FieldExtension>::zero(); num_transition];
            let mut periodic_buf = vec![FieldElement::<Field>::zero(); num_periodic];
            let mut frame = if use_zero_copy {
                Frame::preallocate_lde(lde_trace, num_offsets)
            } else {
                Frame::preallocate(num_offsets, rows_per_step, num_main_cols, num_aux_cols)
            };

            boundary_evaluation
                .into_iter()
                .enumerate()
                .map(|(i, boundary)| {
                    if use_zero_copy {
                        frame.bind_to_lde(i, num_rows, lde_step_size, offsets);
                    } else {
                        frame.fill_from_lde(lde_trace, i, offsets);
                    }

                    for (j, col) in lde_periodic_columns.iter().enumerate() {
                        periodic_buf[j] = col[i].clone();
                    }

                    let ctx = TransitionEvaluationContext::new_prover(
                        &frame,
                        &periodic_buf,
                        rap_challenges,
                        &logup_alpha_powers,
                        logup_table_offset,
                    );
                    air.compute_transition_into(&ctx, &mut transition_buf);

                    let acc_transition = if is_uniform {
                        let z = zerofier_data.get_uniform(i);
                        let sum = transition_buf
                            .iter()
                            .zip(transition_coefficients)
                            .fold(FieldElement::zero(), |acc, (eval, beta)| acc + eval * beta);
                        z * &sum
                    } else {
                        transition_buf
                            .iter()
                            .enumerate()
                            .zip(transition_coefficients)
                            .fold(FieldElement::zero(), |acc, ((c_idx, eval), beta)| {
                                acc + zerofier_data.get(c_idx, i) * eval * beta
                            })
                    };

                    acc_transition + boundary
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

        #[cfg(feature = "instruments")]
        let timer = Instant::now();

        // Fused boundary evaluation: compute (trace[col] - value) on-the-fly
        // instead of pre-computing all boundary_polys_evaluations.
        // This eliminates N_constraints × LDE_size intermediate allocations.
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
        let _transition_evaluations: Vec<FieldElement<FieldExtension>> = Vec::new();

        #[cfg(feature = "instruments")]
        let timer = Instant::now();
        let zerofier_data = air.transition_zerofier_evaluations_grouped(domain);
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

        let num_transition = air.num_transition_constraints();
        let num_periodic = lde_periodic_columns.len();
        let offsets = &air.context().transition_offsets;

        let evaluations_t = Self::evaluate_transitions(
            air,
            lde_trace,
            &lde_periodic_columns,
            rap_challenges,
            &zerofier_data,
            transition_coefficients,
            boundary_evaluation,
            num_transition,
            num_periodic,
            offsets,
            &self.logup_table_offset,
        );

        #[cfg(feature = "instruments")]
        println!(
            "     Evaluated transitions and accumulated results: {:#?}",
            timer.elapsed()
        );

        evaluations_t
    }
}
