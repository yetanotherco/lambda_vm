use super::domain::Domain;
use super::lookup::BusPublicInputs;
use super::trace::TraceTable;
use super::traits::{AIR, TransitionEvaluationContext};
use crate::lookup::{LOGUP_CHALLENGE_ALPHA, PackingShifts, compute_alpha_powers};
use crate::{frame::Frame, trace::LDETraceTable};
use log::{error, info};
use math::field::traits::IsSubFieldOf;
use math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField},
};

/// Validates that the trace is valid with respect to the supplied AIR constraints.
///
/// Accepts a `TraceTable` directly (no coefficient-form polynomials needed).
/// The trace table contains the original trace values on the interpolation domain.
pub fn validate_trace<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
    PI,
>(
    air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
    pub_inputs: &PI,
    trace: &TraceTable<Field, FieldExtension>,
    domain: &Domain<Field>,
    rap_challenges: &[FieldElement<FieldExtension>],
    bus_public_inputs: Option<&BusPublicInputs<FieldExtension>>,
) -> bool {
    info!("Starting constraints validation over trace...");
    let mut ret = true;

    // Build an LDE trace with blowup=1 from the trace table columns.
    let main_trace_columns: Vec<Vec<FieldElement<Field>>> = (0..trace.num_main_columns)
        .map(|col| {
            (0..trace.num_rows())
                .map(|row| trace.main_table.get(row, col).clone())
                .collect()
        })
        .collect();

    let aux_trace_columns: Vec<Vec<FieldElement<FieldExtension>>> = (0..trace.num_aux_columns)
        .map(|col| {
            (0..trace.num_rows())
                .map(|row| trace.aux_table.get(row, col).clone())
                .collect()
        })
        .collect();

    let lde_trace =
        LDETraceTable::from_columns(main_trace_columns, aux_trace_columns, air.step_size(), 1);

    // --------- VALIDATE BOUNDARY CONSTRAINTS ------------
    let trace_length = domain.interpolation_domain_size;
    air.boundary_constraints(pub_inputs, rap_challenges, bus_public_inputs, trace_length)
        .constraints
        .iter()
        .for_each(|constraint| {
            let col = constraint.col;
            let step = constraint.step;
            let boundary_value = constraint.value.clone();

            let trace_value = if !constraint.is_aux {
                lde_trace.get_main(step, col).clone().to_extension()
            } else {
                lde_trace.get_aux(step,  col).clone()
            };

            if boundary_value.clone().to_extension() != trace_value {
                ret = false;
                error!("Boundary constraint inconsistency - Expected value {boundary_value:?} in step {step} and column {col}, found: {trace_value:?}");
            }
        });

    // --------- VALIDATE TRANSITION CONSTRAINTS -----------
    let exemption_steps: Vec<usize> = air
        .constraints_meta()
        .iter()
        .map(|m| lde_trace.num_steps() - m.end_exemptions)
        .collect();

    let logup_alpha_powers: Vec<FieldElement<FieldExtension>> =
        if rap_challenges.len() > LOGUP_CHALLENGE_ALPHA {
            compute_alpha_powers(
                &rap_challenges[LOGUP_CHALLENGE_ALPHA],
                air.max_bus_elements(),
            )
        } else {
            Vec::new()
        };

    let logup_table_offset = match bus_public_inputs {
        Some(bpi) => {
            let n_inv = FieldElement::<Field>::from(trace_length as u64)
                .inv()
                .unwrap();
            n_inv * &bpi.table_contribution
        }
        None => FieldElement::zero(),
    };

    // Iterate over trace and compute transitions
    let packing_shifts = PackingShifts::<Field>::new();
    for step in 0..lde_trace.num_steps() {
        let frame = Frame::read_step_from_lde(&lde_trace, step, &air.context().transition_offsets);
        let transition_evaluation_context = TransitionEvaluationContext::new_prover(
            &frame,
            rap_challenges,
            &logup_alpha_powers,
            &logup_table_offset,
            &packing_shifts,
        );
        let evaluations = air.compute_transition(&transition_evaluation_context);

        // Iterate over each transition evaluation. When the evaluated step is not from
        // the exemption steps corresponding to the transition, it should have zero as a
        // result
        evaluations.iter().enumerate().for_each(|(i, eval)| {
            // Check that all the transition constraint evaluations of the trace are zero.
            // We don't take into account the transition exemptions.
            if step < exemption_steps[i] && eval != &FieldElement::zero() {
                ret = false;
                error!(
                    "Inconsistent evaluation of transition {i} in step {step} - expected 0, got {eval:?}"
                );
            }
        })
    }
    info!("Constraints validation check ended");
    ret
}

/// Validates that the one-dimensional array `data` can be interpreted as two-dimensional
/// array, returning a true when valid and false when not.
pub fn validate_2d_structure<F>(data: &[FieldElement<F>], width: usize) -> bool
where
    F: IsField,
{
    let rows: Vec<Vec<FieldElement<F>>> = data.chunks(width).map(|c| c.to_vec()).collect();
    rows.iter().all(|r| r.len() == rows[0].len())
}
