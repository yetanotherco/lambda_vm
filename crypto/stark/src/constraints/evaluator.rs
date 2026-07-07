use super::boundary::BoundaryConstraints;
use crate::domain::Domain;
use crate::frame::RowFrame;
use crate::lookup::{BusPublicInputs, LOGUP_CHALLENGE_ALPHA, compute_alpha_powers};
use crate::trace::LDETraceTable;
use crate::traits::{AIR, TransitionEvaluationContext, ZerofierEvaluations};
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
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
    /// Uses `map_init` for per-thread buffer reuse (transition evaluations)
    /// and `ZerofierEvaluations` for deduplicated zerofier access.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_transitions(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        rap_challenges: &[FieldElement<FieldExtension>],
        zerofier_data: &ZerofierEvaluations<Field>,
        transition_coefficients: &[FieldElement<FieldExtension>],
        boundary_evaluation: Vec<FieldElement<FieldExtension>>,
        num_transition: usize,
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

        // Per-thread output buffers via map_init: each Rayon worker allocates
        // once, then reuses for all iterations assigned to that thread. The
        // trace rows themselves are BORROWED in place per LDE point (the LDE
        // buffers are row-major) — no per-row gather copy.
        // Per-row evaluation, shared by the parallel and sequential paths below:
        // borrow the rows, evaluate transition constraints, accumulate with zerofiers.
        let eval_row = |i: usize,
                        boundary: FieldElement<FieldExtension>,
                        transition_buf: &mut [FieldElement<FieldExtension>],
                        base_buf: &mut [FieldElement<Field>]|
         -> FieldElement<FieldExtension> {
            let rows = RowFrame::from_lde(lde_trace, i, offsets);

            let ctx = TransitionEvaluationContext::new_prover(
                rows,
                rap_challenges,
                &logup_alpha_powers,
                logup_table_offset,
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
                let mut sum = base_buf
                    .iter()
                    .enumerate()
                    .zip(&transition_coefficients[..num_base])
                    .fold(FieldElement::zero(), |acc, ((c_idx, eval), beta)| {
                        acc + zerofier_data.get(c_idx, i) * eval * beta
                    });
                sum = transition_buf[num_base..]
                    .iter()
                    .enumerate()
                    .zip(&transition_coefficients[num_base..])
                    .fold(sum, |acc, ((j, eval), beta)| {
                        acc + zerofier_data.get(num_base + j, i) * eval * beta
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
                        )
                    },
                    |(transition_buf, base_buf), (i, boundary)| {
                        eval_row(i, boundary, transition_buf, base_buf)
                    },
                )
                .collect()
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut transition_buf = vec![FieldElement::<FieldExtension>::zero(); num_transition];
            let mut base_buf = vec![FieldElement::<Field>::zero(); num_base];

            boundary_evaluation
                .into_iter()
                .enumerate()
                .map(|(i, boundary)| eval_row(i, boundary, &mut transition_buf, &mut base_buf))
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
    ) -> Vec<FieldElement<FieldExtension>>
    where
        Field: 'static,
        FieldExtension: 'static,
    {
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

        let zerofier_data = air.transition_zerofier_evaluations_grouped(domain);

        // GPU composition path: fuse H(row) = z_inv·Σβᵢ·Cᵢ + boundary on-device
        // (no CPU trace read, no per-constraint matrix). Falls through to the CPU
        // path below when the GPU LDE is absent, the field is not Goldilocks, the
        // zerofier is non-uniform, or the transition offsets are non-contiguous.
        #[cfg(feature = "cuda")]
        {
            if let Some(h) = self.try_evaluate_composition_gpu(
                air,
                lde_trace,
                rap_challenges,
                transition_coefficients,
                boundary_coefficients,
                &zerofier_data,
                &boundary_zerofiers_inverse_evaluations,
            ) {
                return h;
            }
        }

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

        // Iterate over all LDE domain and compute the part of the composition polynomial
        // related to the transition constraints and add it to the already computed part of the
        // boundary constraints.

        let num_transition = air.num_transition_constraints();
        let offsets = &air.context().transition_offsets;

        Self::evaluate_transitions(
            air,
            lde_trace,
            rap_challenges,
            &zerofier_data,
            transition_coefficients,
            boundary_evaluation,
            num_transition,
            offsets,
            &self.logup_table_offset,
        )
    }

    /// GPU composition path: produce `H(row)` on-device (transition + boundary
    /// fused, no CPU trace read, no per-constraint matrix), returning `None` to
    /// fall back to the CPU path when the GPU LDE is absent, the tower is not
    /// Goldilocks/degree-3, the zerofier is non-uniform (end-exemptions), or the
    /// transition offsets are non-contiguous. The result feeds the existing
    /// decompose + composition commit exactly like the CPU `H` vector.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn try_evaluate_composition_gpu(
        &self,
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        rap_challenges: &[FieldElement<FieldExtension>],
        transition_coefficients: &[FieldElement<FieldExtension>],
        boundary_coefficients: &[FieldElement<FieldExtension>],
        zerofier_data: &ZerofierEvaluations<Field>,
        boundary_z_inv: &[Vec<FieldElement<Field>>],
    ) -> Option<Vec<FieldElement<FieldExtension>>>
    where
        Field: 'static,
        FieldExtension: 'static,
    {
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
        use math::field::goldilocks::GoldilocksField;
        use std::any::TypeId;

        if TypeId::of::<Field>() != TypeId::of::<GoldilocksField>()
            || TypeId::of::<FieldExtension>() != TypeId::of::<Degree3GoldilocksExtensionField>()
        {
            return None;
        }
        if !zerofier_data.is_uniform() {
            return None;
        }
        // The kernel's row math assumes Var offsets index a contiguous [0..n)
        // frame (offset·step). The VM uses [0, 1]; anything else → CPU.
        let offsets = &air.context().transition_offsets;
        if !offsets.iter().enumerate().all(|(i, &o)| o == i) {
            return None;
        }
        let main = lde_trace.gpu_main()?;
        let aux = lde_trace.gpu_aux()?;

        let prog = air.constraint_program();

        // LogUp alpha powers, exactly as `evaluate_transitions` derives them.
        let logup_alpha_powers: Vec<FieldElement<FieldExtension>> =
            if rap_challenges.len() > LOGUP_CHALLENGE_ALPHA {
                compute_alpha_powers(
                    &rap_challenges[LOGUP_CHALLENGE_ALPHA],
                    air.max_bus_elements(),
                )
            } else {
                Vec::new()
            };

        // Boundary spec (aligned with `boundary_coefficients` / `boundary_z_inv`).
        let bcs = &self.boundary_constraints.constraints;
        let b_col: Vec<usize> = bcs.iter().map(|c| c.col).collect();
        let b_is_aux: Vec<bool> = bcs.iter().map(|c| c.is_aux).collect();
        let b_value: Vec<FieldElement<FieldExtension>> =
            bcs.iter().map(|c| c.value.clone()).collect();
        let b_z_inv_flat: Vec<FieldElement<Field>> =
            boundary_z_inv.iter().flatten().cloned().collect();

        let inputs = crate::constraint_ir::gpu_interp::CompositionInputs {
            beta_trans: transition_coefficients,
            z_inv: &zerofier_data.groups[0],
            b_col: &b_col,
            b_is_aux: &b_is_aux,
            b_value: &b_value,
            b_beta: boundary_coefficients,
            b_z_inv: &b_z_inv_flat,
        };

        let next_step = lde_trace.lde_step_size; // == blowup_factor (single-row steps)
        let num_rows = lde_trace.num_rows();

        let raw = crate::constraint_ir::gpu_interp::try_eval_composition_gpu(
            prog,
            main,
            aux,
            rap_challenges,
            &logup_alpha_powers,
            &self.logup_table_offset,
            next_step,
            num_rows,
            &inputs,
        )?;

        // SAFETY: the TypeId gate established `FieldExtension ==
        // Degree3GoldilocksExtensionField`, which is `#[repr(transparent)]` over
        // `[u64; 3]`; `raw.len() == num_rows * 3`.
        let h: Vec<FieldElement<FieldExtension>> = unsafe {
            std::slice::from_raw_parts(
                raw.as_ptr() as *const FieldElement<FieldExtension>,
                raw.len() / 3,
            )
        }
        .to_vec();
        Some(h)
    }
}
