use crate::{
    table::{LdeRowRef, TableView},
    trace::LDETraceTable,
};
use itertools::Itertools;
use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};

/// A frame represents a collection of trace steps.
/// The collected steps are all the necessary steps for
/// all transition constraints over a trace to be evaluated.
///
/// Owns its row data so it can be built from either row-major Tables
/// (verifier) or column-major LDE data (prover) without lifetime issues.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame<F: IsSubFieldOf<E>, E: IsField>
where
    E: IsField,
    F: IsSubFieldOf<E>,
{
    steps: Vec<TableView<F, E>>,
}

impl<F: IsSubFieldOf<E>, E: IsField> Frame<F, E> {
    pub fn new(steps: Vec<TableView<F, E>>) -> Self {
        Self { steps }
    }

    pub fn get_evaluation_step(&self, step: usize) -> &TableView<F, E> {
        &self.steps[step]
    }

    /// Build a Frame by gathering row data from a column-major LDETraceTable.
    ///
    /// Each step gathers elements from columns into owned Vecs. For the typical
    /// case (2 offsets, step_size=1), this gathers 2 rows of ~74 main + aux elements.
    pub fn read_from_lde(lde_trace: &LDETraceTable<F, E>, row: usize, offsets: &[usize]) -> Self {
        let blowup_factor = lde_trace.blowup_factor;
        let num_rows = lde_trace.num_rows();
        let step_size = lde_trace.lde_step_size;
        let num_main_cols = lde_trace.num_main_cols();
        let num_aux_cols = lde_trace.num_aux_cols();

        let lde_steps = offsets
            .iter()
            .map(|offset| {
                let initial_step_row = row + offset * step_size;
                let end_step_row = initial_step_row + step_size;
                let (table_view_main_data, table_view_aux_data): (Vec<_>, Vec<_>) =
                    (initial_step_row..end_step_row)
                        .step_by(blowup_factor)
                        .map(|step_row| {
                            let step_row_idx = step_row % num_rows;

                            // Gather main row from columns
                            let main_row: Vec<_> = (0..num_main_cols)
                                .map(|col| lde_trace.get_main(step_row_idx, col).clone())
                                .collect();

                            // Gather aux row from columns
                            let aux_row: Vec<_> = (0..num_aux_cols)
                                .map(|col| lde_trace.get_aux(step_row_idx, col).clone())
                                .collect();

                            (main_row, aux_row)
                        })
                        .unzip();

                TableView::new(table_view_main_data, table_view_aux_data)
            })
            .collect_vec();

        Frame::new(lde_steps)
    }

    pub fn read_step_from_lde(
        lde_trace: &LDETraceTable<F, E>,
        step: usize,
        offsets: &[usize],
    ) -> Self {
        let row = lde_trace.step_to_row(step);
        Self::read_from_lde(lde_trace, row, offsets)
    }

    /// Pre-allocate a Frame with the right dimensions for reuse in hot loops.
    ///
    /// The frame will have `offsets.len()` steps, each containing
    /// `step_size / blowup_factor` rows (typically 1) of main and aux columns.
    pub fn preallocate(
        num_offsets: usize,
        rows_per_step: usize,
        num_main_cols: usize,
        num_aux_cols: usize,
    ) -> Self {
        let steps = (0..num_offsets)
            .map(|_| {
                let main_data: Vec<Vec<FieldElement<F>>> = (0..rows_per_step)
                    .map(|_| vec![FieldElement::zero(); num_main_cols])
                    .collect();
                let aux_data: Vec<Vec<FieldElement<E>>> = (0..rows_per_step)
                    .map(|_| vec![FieldElement::zero(); num_aux_cols])
                    .collect();
                TableView::new(main_data, aux_data)
            })
            .collect();
        Frame { steps }
    }

    /// Fill a pre-allocated frame from LDE data, without allocating.
    ///
    /// The frame must have been created with `preallocate` with matching dimensions.
    pub fn fill_from_lde(
        &mut self,
        lde_trace: &LDETraceTable<F, E>,
        row: usize,
        offsets: &[usize],
    ) {
        let blowup_factor = lde_trace.blowup_factor;
        let num_rows = lde_trace.num_rows();
        let step_size = lde_trace.lde_step_size;
        let num_main_cols = lde_trace.num_main_cols();
        let num_aux_cols = lde_trace.num_aux_cols();

        for (step_idx, &offset) in offsets.iter().enumerate() {
            let initial_step_row = row + offset * step_size;
            let end_step_row = initial_step_row + step_size;
            let step = &mut self.steps[step_idx];

            let mut sub_row_idx = 0;
            let mut step_row = initial_step_row;
            while step_row < end_step_row {
                let step_row_idx = step_row % num_rows;

                // Overwrite main row elements
                for col in 0..num_main_cols {
                    step.data[sub_row_idx][col] = lde_trace.get_main(step_row_idx, col).clone();
                }

                // Overwrite aux row elements
                for col in 0..num_aux_cols {
                    step.aux_data[sub_row_idx][col] = lde_trace.get_aux(step_row_idx, col).clone();
                }

                sub_row_idx += 1;
                step_row += blowup_factor;
            }
        }
    }

    /// Pre-allocate a Frame for LDE-backed zero-copy access.
    ///
    /// Base pointers are captured from `lde_trace` columns and never change.
    /// Only `row_offset` is updated per LDE point via `bind_to_lde`.
    /// This eliminates all per-point element copies in the constraint evaluation loop.
    pub fn preallocate_lde(lde_trace: &LDETraceTable<F, E>, num_offsets: usize) -> Self {
        let main_base: Vec<*const FieldElement<F>> =
            lde_trace.main_columns.iter().map(|c| c.as_ptr()).collect();
        let aux_base: Vec<*const FieldElement<E>> =
            lde_trace.aux_columns.iter().map(|c| c.as_ptr()).collect();

        let steps = (0..num_offsets)
            .map(|_| TableView {
                data: Vec::new(),
                aux_data: Vec::new(),
                lde_main: Some(LdeRowRef {
                    base_ptrs: main_base.clone(),
                    row_offset: 0,
                }),
                lde_aux: Some(LdeRowRef {
                    base_ptrs: aux_base.clone(),
                    row_offset: 0,
                }),
            })
            .collect();
        Frame { steps }
    }

    /// Update row offsets for the current LDE domain point. No element copies.
    ///
    /// Each step's LDE references are rebased to point at the correct row in
    /// the LDE column buffers. This replaces the element-by-element copying
    /// that `fill_from_lde` performs.
    #[inline]
    pub fn bind_to_lde(
        &mut self,
        row: usize,
        num_rows: usize,
        step_size: usize,
        offsets: &[usize],
    ) {
        for (step_idx, &offset) in offsets.iter().enumerate() {
            let step_row_idx = (row + offset * step_size) % num_rows;
            let step = &mut self.steps[step_idx];
            if let Some(ref mut lde) = step.lde_main {
                lde.row_offset = step_row_idx;
            }
            if let Some(ref mut lde) = step.lde_aux {
                lde.row_offset = step_row_idx;
            }
        }
    }
}
