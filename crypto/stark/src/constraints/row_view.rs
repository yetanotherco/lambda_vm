//! Zero-copy row accessor for constraint evaluation.
//!
//! `RowView` replaces the per-LDE-row `Frame`/`TableView` gather (which cloned
//! every column cell into nested `Vec<Vec<FieldElement>>`). It borrows the
//! column-major LDE columns and resolves `(offset, sub_row, col)` to a cell by
//! reference, with no allocation and no clone.
//!
//! Indexing matches `Frame::fill_from_lde` exactly: for a base LDE row `row`,
//! offset index `offset`, and within-step row `sub_row`, the LDE row is
//! `(row + offset * lde_step_size + sub_row * blowup_factor) % num_rows`.

use crate::trace::LDETraceTable;
use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};

/// A borrowed, zero-copy view of one evaluation point's rows over a
/// row-major [`LDETraceTable`]. A row's cells are contiguous, so a constraint's
/// per-row column reads scan sequential memory (the row-major-LDE win).
pub struct RowView<'a, F, E>
where
    E: IsField,
    F: IsSubFieldOf<E>,
{
    main: &'a [FieldElement<F>],
    aux: &'a [FieldElement<E>],
    num_main_cols: usize,
    num_aux_cols: usize,
    row: usize,
    lde_step_size: usize,
    blowup_factor: usize,
    num_rows: usize,
}

impl<'a, F, E> RowView<'a, F, E>
where
    E: IsField,
    F: IsSubFieldOf<E>,
{
    /// Build a view anchored at LDE row `row`.
    pub fn new(lde_trace: &'a LDETraceTable<F, E>, row: usize) -> Self {
        Self {
            main: &lde_trace.main_data,
            aux: &lde_trace.aux_data,
            num_main_cols: lde_trace.num_main_cols,
            num_aux_cols: lde_trace.num_aux_cols,
            row,
            lde_step_size: lde_trace.lde_step_size,
            blowup_factor: lde_trace.blowup_factor,
            num_rows: lde_trace.num_rows,
        }
    }

    /// Resolve `(offset, sub_row)` to the absolute LDE row index, matching
    /// `Frame::fill_from_lde`.
    #[inline]
    fn lde_row(&self, offset: usize, sub_row: usize) -> usize {
        (self.row + offset * self.lde_step_size + sub_row * self.blowup_factor) % self.num_rows
    }

    /// Main-trace cell at offset/sub_row/col, by reference (row-major index).
    #[inline]
    pub fn get_main(&self, offset: usize, sub_row: usize, col: usize) -> &FieldElement<F> {
        &self.main[self.lde_row(offset, sub_row) * self.num_main_cols + col]
    }

    /// Auxiliary-trace cell at offset/sub_row/col, by reference (row-major index).
    #[inline]
    pub fn get_aux(&self, offset: usize, sub_row: usize, col: usize) -> &FieldElement<E> {
        &self.aux[self.lde_row(offset, sub_row) * self.num_aux_cols + col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use math::field::goldilocks::GoldilocksField;

    type F = GoldilocksField;

    fn fe(x: u64) -> FieldElement<F> {
        FieldElement::<F>::from(x)
    }

    /// `RowView` must return exactly what `Frame::fill_from_lde` would, for every
    /// (offset, sub_row, col) and a range of base rows including wraparound.
    #[test]
    fn row_view_matches_frame() {
        // trace_step_size=2, blowup=2 -> lde_step_size=4, rows_per_step=2.
        let num_rows = 8;
        let trace_step_size = 2;
        let blowup = 2;
        let num_main = 3;
        let num_aux = 2;

        let main_columns: Vec<Vec<FieldElement<F>>> = (0..num_main)
            .map(|c| (0..num_rows).map(|r| fe((r * 10 + c) as u64)).collect())
            .collect();
        let aux_columns: Vec<Vec<FieldElement<F>>> = (0..num_aux)
            .map(|c| (0..num_rows).map(|r| fe((1000 + r * 10 + c) as u64)).collect())
            .collect();

        let lde = LDETraceTable::<F, F>::from_columns(
            main_columns,
            aux_columns,
            trace_step_size,
            blowup,
        );

        let offsets = [0usize, 1];
        let rows_per_step = lde.lde_step_size / blowup; // 2

        // Base rows including ones that force the modulo wrap (6, 7).
        for &row in &[0usize, 1, 3, 6, 7] {
            let mut frame =
                Frame::<F, F>::preallocate(offsets.len(), rows_per_step, num_main, num_aux);
            frame.fill_from_lde(&lde, row, &offsets);

            let view = RowView::new(&lde, row);

            for (offset_idx, _) in offsets.iter().enumerate() {
                let step = frame.get_evaluation_step(offset_idx);
                for sub_row in 0..rows_per_step {
                    for col in 0..num_main {
                        assert_eq!(
                            view.get_main(offset_idx, sub_row, col),
                            step.get_main_evaluation_element(sub_row, col),
                            "main mismatch row={row} offset={offset_idx} sub_row={sub_row} col={col}"
                        );
                    }
                    for col in 0..num_aux {
                        assert_eq!(
                            view.get_aux(offset_idx, sub_row, col),
                            step.get_aux_evaluation_element(sub_row, col),
                            "aux mismatch row={row} offset={offset_idx} sub_row={sub_row} col={col}"
                        );
                    }
                }
            }
        }
    }
}
