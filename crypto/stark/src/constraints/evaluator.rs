use super::boundary::BoundaryConstraints;
#[cfg(all(debug_assertions, not(feature = "parallel")))]
use crate::debug::check_boundary_polys_divisibility;
use crate::domain::Domain;
use crate::lookup::{BusPublicInputs, LOGUP_CHALLENGE_ALPHA, PackingShifts, compute_alpha_powers};
use crate::trace::LDETraceTable;
use crate::traits::{AIR, TransitionEvaluationContext, ZerofierEvaluations};
use crate::{frame::Frame, prover::evaluate_polynomial_on_lde_domain};
use math::fft::errors::FFTError;
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
    /// Uses the AirBuilder pattern when the AIR has a builder, otherwise falls back
    /// to the old frame-based `compute_transition` path for test/example AIRs.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_transitions(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        rap_challenges: &[FieldElement<FieldExtension>],
        zerofier_data: &ZerofierEvaluations<Field>,
        boundary_evaluation: Vec<FieldElement<FieldExtension>>,
        logup_table_offset: &FieldElement<FieldExtension>,
        composition_alpha: &FieldElement<FieldExtension>,
        transition_coefficients: &[FieldElement<FieldExtension>],
        domain: &Domain<Field>,
    ) -> Vec<FieldElement<FieldExtension>> {
        let use_builder = air.has_any_builder();

        if use_builder {
            // AirBuilder path: uniform zerofiers required
            assert!(
                zerofier_data.is_uniform(),
                "AirBuilder requires uniform zerofiers"
            );
            Self::evaluate_transitions_builder(
                air,
                lde_trace,
                rap_challenges,
                zerofier_data,
                boundary_evaluation,
                logup_table_offset,
                composition_alpha,
            )
        } else {
            // Old frame-based path for test/example AIRs
            Self::evaluate_transitions_legacy(
                air,
                lde_trace,
                rap_challenges,
                zerofier_data,
                boundary_evaluation,
                logup_table_offset,
                transition_coefficients,
                domain,
            )
        }
    }

    /// AirBuilder-based transition evaluation (for VM tables with builders).
    #[allow(clippy::needless_range_loop)]
    fn evaluate_transitions_builder(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        rap_challenges: &[FieldElement<FieldExtension>],
        zerofier_data: &ZerofierEvaluations<Field>,
        boundary_evaluation: Vec<FieldElement<FieldExtension>>,
        logup_table_offset: &FieldElement<FieldExtension>,
        composition_alpha: &FieldElement<FieldExtension>,
    ) -> Vec<FieldElement<FieldExtension>> {
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

        // Pre-compute composition alpha powers [1, α, α², ...] for ALL constraints.
        // This avoids an E×E multiply per constraint per LDE point in the hot loop.
        // Use a generous count: main constraints + LogUp aux constraints.
        // Over-allocating is cheap (one-time cost); under-allocating would panic.
        let total_constraints = air.num_transition_constraints() * 2 + 64;
        let composition_alpha_powers = compute_alpha_powers(composition_alpha, total_constraints);

        let num_main_cols = lde_trace.num_main_cols();
        let use_dual_path = air.has_main_builder();

        // Buffer sizes for constraint evaluations
        let num_base_constraints = air.num_transition_constraints();
        let num_ext_constraints = total_constraints - num_base_constraints;

        #[cfg(feature = "parallel")]
        {
            let evaluations_t: Vec<_> = boundary_evaluation
                .into_par_iter()
                .enumerate()
                .map_init(
                    || {
                        (
                            vec![FieldElement::<Field>::zero(); num_main_cols],
                            vec![FieldElement::<Field>::zero(); num_base_constraints],
                            vec![FieldElement::<FieldExtension>::zero(); num_ext_constraints],
                        )
                    },
                    |(row_cache, base_buf, ext_buf), (i, boundary)| {
                        if use_dual_path {
                            // Fill row cache from LDE trace
                            for col in 0..num_main_cols {
                                row_cache[col] = lde_trace.get_main(i, col).clone();
                            }
                            // Direct closure: ONE dyn call, zero vtable inside
                            air.eval_main_constraints_direct(row_cache, base_buf);
                            // Create builder with base_count pre-set for LogUp only
                            let mut builder = crate::air_builder::ProverBuilder::new_with_buffers(
                                lde_trace,
                                i,
                                &composition_alpha_powers,
                                rap_challenges,
                                &logup_alpha_powers,
                                logup_table_offset,
                                row_cache,
                                base_buf,
                                ext_buf,
                            );
                            builder.set_base_count(num_base_constraints);
                            air.eval_logup_with_builder(&mut builder);
                            zerofier_data.get_uniform(i) * &builder.finish() + boundary
                        } else {
                            let mut builder = crate::air_builder::ProverBuilder::new_with_buffers(
                                lde_trace,
                                i,
                                &composition_alpha_powers,
                                rap_challenges,
                                &logup_alpha_powers,
                                logup_table_offset,
                                row_cache,
                                base_buf,
                                ext_buf,
                            );
                            air.eval_constraints_with_builder(&mut builder);
                            zerofier_data.get_uniform(i) * &builder.finish() + boundary
                        }
                    },
                )
                .collect();
            evaluations_t
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut row_cache = vec![FieldElement::<Field>::zero(); num_main_cols];
            let mut base_buf = vec![FieldElement::<Field>::zero(); num_base_constraints];
            let mut ext_buf = vec![FieldElement::<FieldExtension>::zero(); num_ext_constraints];
            boundary_evaluation
                .into_iter()
                .enumerate()
                .map(|(i, boundary)| {
                    if use_dual_path {
                        // Fill row cache from LDE trace
                        for col in 0..num_main_cols {
                            row_cache[col] = lde_trace.get_main(i, col).clone();
                        }
                        // Direct closure: ONE dyn call, zero vtable inside
                        air.eval_main_constraints_direct(&row_cache, &mut base_buf);
                        // Create builder with base_count pre-set for LogUp only
                        let mut builder = crate::air_builder::ProverBuilder::new_with_buffers(
                            lde_trace,
                            i,
                            &composition_alpha_powers,
                            rap_challenges,
                            &logup_alpha_powers,
                            logup_table_offset,
                            &mut row_cache,
                            &mut base_buf,
                            &mut ext_buf,
                        );
                        builder.set_base_count(num_base_constraints);
                        air.eval_logup_with_builder(&mut builder);
                        zerofier_data.get_uniform(i) * &builder.finish() + boundary
                    } else {
                        let mut builder = crate::air_builder::ProverBuilder::new_with_buffers(
                            lde_trace,
                            i,
                            &composition_alpha_powers,
                            rap_challenges,
                            &logup_alpha_powers,
                            logup_table_offset,
                            &mut row_cache,
                            &mut base_buf,
                            &mut ext_buf,
                        );
                        air.eval_constraints_with_builder(&mut builder);
                        zerofier_data.get_uniform(i) * &builder.finish() + boundary
                    }
                })
                .collect()
        }
    }

    /// Legacy frame-based transition evaluation for test/example AIRs.
    ///
    /// Handles both uniform and non-uniform zerofiers.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_transitions_legacy(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        rap_challenges: &[FieldElement<FieldExtension>],
        zerofier_data: &ZerofierEvaluations<Field>,
        boundary_evaluation: Vec<FieldElement<FieldExtension>>,
        logup_table_offset: &FieldElement<FieldExtension>,
        transition_coefficients: &[FieldElement<FieldExtension>],
        domain: &Domain<Field>,
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

        // Precompute packing shift constants once for all LDE domain points.
        let packing_shifts = PackingShifts::<Field>::new();

        // Pre-compute periodic column evaluations on LDE domain.
        let trace_length = domain.interpolation_domain_size;
        let lde_periodic_columns: Vec<Vec<FieldElement<Field>>> = air
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

        let num_transition = air.num_transition_constraints();
        let num_periodic = lde_periodic_columns.len();
        let offsets = &air.context().transition_offsets;

        let blowup_factor = lde_trace.blowup_factor;
        let lde_step_size = lde_trace.lde_step_size;
        let rows_per_step = lde_step_size / blowup_factor;
        let num_main_cols = lde_trace.num_main_cols();
        let num_aux_cols = lde_trace.num_aux_cols();
        let num_offsets = offsets.len();

        #[cfg(feature = "parallel")]
        {
            let evaluations_t: Vec<_> = boundary_evaluation
                .into_par_iter()
                .enumerate()
                .map_init(
                    || {
                        (
                            vec![FieldElement::<FieldExtension>::zero(); num_transition],
                            vec![FieldElement::<Field>::zero(); num_periodic],
                            Frame::preallocate(
                                num_offsets,
                                rows_per_step,
                                num_main_cols,
                                num_aux_cols,
                            ),
                        )
                    },
                    |(transition_buf, periodic_buf, frame), (i, boundary)| {
                        frame.fill_from_lde(lde_trace, i, offsets);

                        for (j, col) in lde_periodic_columns.iter().enumerate() {
                            periodic_buf[j] = col[i].clone();
                        }

                        let ctx = TransitionEvaluationContext::new_prover(
                            frame,
                            periodic_buf,
                            rap_challenges,
                            &logup_alpha_powers,
                            logup_table_offset,
                            &packing_shifts,
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
            let mut transition_buf = vec![FieldElement::<FieldExtension>::zero(); num_transition];
            let mut periodic_buf = vec![FieldElement::<Field>::zero(); num_periodic];
            let mut frame =
                Frame::preallocate(num_offsets, rows_per_step, num_main_cols, num_aux_cols);

            boundary_evaluation
                .into_iter()
                .enumerate()
                .map(|(i, boundary)| {
                    frame.fill_from_lde(lde_trace, i, offsets);

                    for (j, col) in lde_periodic_columns.iter().enumerate() {
                        periodic_buf[j] = col[i].clone();
                    }

                    let ctx = TransitionEvaluationContext::new_prover(
                        &frame,
                        &periodic_buf,
                        rap_challenges,
                        &logup_alpha_powers,
                        logup_table_offset,
                        &packing_shifts,
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
            transition_coefficients,
            domain,
        )
    }
}
