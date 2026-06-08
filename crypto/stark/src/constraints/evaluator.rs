use super::boundary::BoundaryConstraints;
use crate::domain::Domain;
use crate::lookup::{BusPublicInputs, LOGUP_CHALLENGE_ALPHA, PackingShifts, compute_alpha_powers};
use crate::trace::LDETraceTable;
use crate::traits::{AIR, TransitionEvaluationContext, ZerofierEvaluations};
use crate::{frame::Frame, prover::evaluate_polynomial_on_lde_domain};
use math::fft::bit_reversing::{in_place_bit_reverse_permute, reverse_index};
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
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

        // Bit-reversed evaluator: the LDE is read at physical row `i`, whose logical
        // row is `reverse_index(i)`. The zerofier groups stay in their compact cycled
        // form (`group[logical % len]`, cache-resident) — indexed with the bit-reversed
        // logical row, rather than materializing full `lde_size` copies per group (which
        // would stream hundreds of MB per `evaluate`). `reverse_index(i)` is computed
        // once per row and shared across all constraints in that row.
        let lde_size = boundary_evaluation.len();

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

        // Per-row evaluation, shared by the parallel and sequential paths below:
        // fill the frame, evaluate transition constraints, accumulate with zerofiers.
        let eval_row = |i: usize,
                        boundary: FieldElement<FieldExtension>,
                        transition_buf: &mut [FieldElement<FieldExtension>],
                        base_buf: &mut [FieldElement<Field>],
                        periodic_buf: &mut [FieldElement<Field>],
                        frame: &mut Frame<Field, FieldExtension>|
         -> FieldElement<FieldExtension> {
            frame.fill_from_lde_bitrev(lde_trace, i, offsets);

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

            // Logical row for the bit-reversed physical row `i`: zerofier groups are
            // stored in natural (cycled) order, so we look them up at `reverse_index(i)`.
            let rev_i = reverse_index(i, lde_size as u64);
            let acc_transition = if is_uniform {
                // All constraints share one zerofier: factor it out of the sum.
                let z = zerofier_data.get_uniform(rev_i);
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
                let mut sum = base_buf
                    .iter()
                    .enumerate()
                    .zip(&transition_coefficients[..num_base])
                    .fold(FieldElement::zero(), |acc, ((c_idx, eval), beta)| {
                        acc + zerofier_data.get(c_idx, rev_i) * eval * beta
                    });
                sum = transition_buf[num_base..]
                    .iter()
                    .enumerate()
                    .zip(&transition_coefficients[num_base..])
                    .fold(sum, |acc, ((j, eval), beta)| {
                        acc + zerofier_data.get(num_base + j, rev_i) * eval * beta
                    });
                sum
            };

            acc_transition + boundary
        };

        #[cfg(feature = "parallel")]
        {
            boundary_evaluation
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
                        eval_row(i, boundary, transition_buf, base_buf, periodic_buf, frame)
                    },
                )
                .collect()
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
                    eval_row(
                        i,
                        boundary,
                        &mut transition_buf,
                        &mut base_buf,
                        &mut periodic_buf,
                        &mut frame,
                    )
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
                    // Bit-reverse to align with the bit-reversed LDE: physical row
                    // p reads `evals[p]` (= the zerofier inv at logical row
                    // reverse_index(p)), matching `get_main(p, ..)`.
                    in_place_bit_reverse_permute(&mut evals);
                    evals
                })
                .collect::<Vec<Vec<FieldElement<Field>>>>();

        let trace_length = domain.interpolation_domain_size;
        let lde_periodic_columns = air
            .get_periodic_column_polynomials(trace_length)
            .iter()
            .map(|poly| {
                let mut col = evaluate_polynomial_on_lde_domain(
                    poly,
                    domain.blowup_factor,
                    domain.interpolation_domain_size,
                    &domain.coset_offset,
                )?;
                // Bit-reverse so physical row p reads `col[p]` (periodic value at
                // logical row reverse_index(p)), aligned with the bit-reversed LDE.
                in_place_bit_reverse_permute(&mut col);
                Ok(col)
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
