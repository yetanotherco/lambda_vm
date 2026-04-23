use std::collections::HashMap;

use super::boundary::BoundaryConstraints;
#[cfg(all(debug_assertions, not(feature = "parallel")))]
use crate::debug::check_boundary_polys_divisibility;
use crate::domain::Domain;
use crate::lookup::{BusPublicInputs, LOGUP_CHALLENGE_ALPHA, PackingShifts, compute_alpha_powers};
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
        let num_base = air.num_base_transition_constraints();

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

        // Per-thread buffers via map_init: each Rayon worker allocates once,
        // then reuses for all iterations assigned to that thread.
        // The Frame is pre-allocated and filled in-place to avoid Vec allocations
        // on every LDE point (a significant fraction of total CPU time).
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
                            vec![FieldElement::<Field>::zero(); num_base],
                            vec![FieldElement::<Field>::zero(); num_periodic],
                            Frame::preallocate(
                                num_offsets,
                                rows_per_step,
                                num_main_cols,
                                num_aux_cols,
                            ),
                        )
                    },
                    |(transition_buf, base_buf, periodic_buf, frame), (i, boundary)| {
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
                        air.compute_transition_prover(&ctx, base_buf, transition_buf);

                        let acc_transition = if is_uniform {
                            // All constraints share one zerofier: factor it out of the sum.
                            let z = zerofier_data.get_uniform(i);
                            // F×E inner product for base constraints (3 muls per term)
                            let mut sum = base_buf
                                .iter()
                                .zip(&transition_coefficients[..num_base])
                                .fold(FieldElement::zero(), |acc, (eval, beta)| acc + eval * beta);
                            // E×E for extension constraints (9 muls per term)
                            sum = transition_buf[num_base..]
                                .iter()
                                .zip(&transition_coefficients[num_base..])
                                .fold(sum, |acc, (eval, beta)| acc + eval * beta);
                            z * &sum
                        } else {
                            // Iterate group-by-group, amortizing each zerofier
                            // multiply across the constraints that share it.
                            // Base vs extension routing per constraint is still
                            // needed (F×E vs E×E arithmetic) but the zerofier
                            // multiply is hoisted outside the inner loop.
                            zerofier_data.group_to_constraints.iter().enumerate().fold(
                                FieldElement::zero(),
                                |acc, (group_idx, members)| {
                                    let z = zerofier_data.get_group(group_idx, i);
                                    let group_sum = members.iter().fold(
                                        FieldElement::<FieldExtension>::zero(),
                                        |s, &c_idx| {
                                            let beta = &transition_coefficients[c_idx];
                                            if c_idx < num_base {
                                                s + &base_buf[c_idx] * beta
                                            } else {
                                                s + &transition_buf[c_idx] * beta
                                            }
                                        },
                                    );
                                    acc + z * group_sum
                                },
                            )
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
            let mut base_buf = vec![FieldElement::<Field>::zero(); num_base];
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
                    air.compute_transition_prover(&ctx, &mut base_buf, &mut transition_buf);

                    let acc_transition = if is_uniform {
                        let z = zerofier_data.get_uniform(i);
                        // F×E inner product for base constraints (3 muls per term)
                        let mut sum = base_buf
                            .iter()
                            .zip(&transition_coefficients[..num_base])
                            .fold(FieldElement::zero(), |acc, (eval, beta)| acc + eval * beta);
                        // E×E for extension constraints (9 muls per term)
                        sum = transition_buf[num_base..]
                            .iter()
                            .zip(&transition_coefficients[num_base..])
                            .fold(sum, |acc, (eval, beta)| acc + eval * beta);
                        z * &sum
                    } else {
                        zerofier_data.group_to_constraints.iter().enumerate().fold(
                            FieldElement::zero(),
                            |acc, (group_idx, members)| {
                                let z = zerofier_data.get_group(group_idx, i);
                                let group_sum = members.iter().fold(
                                    FieldElement::<FieldExtension>::zero(),
                                    |s, &c_idx| {
                                        let beta = &transition_coefficients[c_idx];
                                        if c_idx < num_base {
                                            s + &base_buf[c_idx] * beta
                                        } else {
                                            s + &transition_buf[c_idx] * beta
                                        }
                                    },
                                );
                                acc + z * group_sum
                            },
                        )
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

        // Deduplicate boundary zerofier inverses by step.
        //
        // Two boundary constraints at the same step share the same zerofier
        // `v - g^step` on the LDE coset, so `inplace_batch_inverse` over the
        // full LDE (≥2M elements at typical sizes) would run redundantly if we
        // computed one vector per constraint.
        //
        // Build `unique_zerofier_invs`: one inverted zerofier vector per unique
        // step. `constraint_zerofier_idx[c]` indexes into it so the downstream
        // hot-loop keeps its per-constraint shape but reads from the shared
        // storage.
        //
        // Typical savings (e.g. circular LogUp pins `acc[N-1]=0` on every table
        // sharing step = N-1): the batch inversion runs ≤ 3 times per table
        // instead of `num_boundary_constraints` times, and peak memory drops
        // from `num_boundary_constraints × lde_size` to `unique_steps × lde_size`.
        let mut step_to_idx: HashMap<usize, usize> = HashMap::new();
        let mut unique_zerofier_invs: Vec<Vec<FieldElement<Field>>> = Vec::new();
        let mut constraint_zerofier_idx: Vec<usize> =
            Vec::with_capacity(boundary_constraints.constraints.len());
        for bc in boundary_constraints.constraints.iter() {
            let idx = *step_to_idx.entry(bc.step).or_insert_with(|| {
                let point = domain.trace_primitive_root.pow(bc.step as u64);
                let mut evals = domain
                    .lde_roots_of_unity_coset
                    .iter()
                    .map(|v| v - &point)
                    .collect::<Vec<FieldElement<Field>>>();
                FieldElement::inplace_batch_inverse(&mut evals)
                    .expect("boundary zerofier has no zero: g^step is unique in the trace coset");
                let idx = unique_zerofier_invs.len();
                unique_zerofier_invs.push(evals);
                idx
            });
            constraint_zerofier_idx.push(idx);
        }

        #[cfg(all(debug_assertions, not(feature = "parallel")))]
        let boundary_polys: Vec<Polynomial<FieldElement<Field>>> = Vec::new();

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
                    .zip(constraint_zerofier_idx.iter())
                    .fold(
                        FieldElement::zero(),
                        |acc, ((constraint, beta), &zerofier_idx)| {
                            let bp = if constraint.is_aux {
                                lde_trace.get_aux(domain_index, constraint.col) - &constraint.value
                            } else {
                                lde_trace.get_main(domain_index, constraint.col) - &constraint.value
                            };
                            let zerofier_inv = &unique_zerofier_invs[zerofier_idx][domain_index];
                            acc + zerofier_inv * beta * bp
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

        // Iterate over all LDE domain and compute the part of the composition polynomial
        // related to the transition constraints and add it to the already computed part of the
        // boundary constraints.

        let num_transition = air.num_transition_constraints();
        let num_periodic = lde_periodic_columns.len();
        let offsets = &air.context().transition_offsets;

        Self::evaluate_transitions(
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
        )
    }
}
