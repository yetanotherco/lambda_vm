use crate::{table::TableView, trace::LDETraceTable};
use itertools::Itertools;
use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};

/// Maximum number of transition offsets a [`RowFrame`] can hold. Every
/// production table uses two (`[0, 1]`); the widest example AIR uses three.
pub const MAX_TRANSITION_OFFSETS: usize = 4;

/// Borrowed per-row view of the trace for prover-side transition
/// evaluation: one contiguous `(main, aux)` row-slice pair per transition
/// offset, taken IN PLACE from the row-major storage. Replaces the per-row
/// gather-copy into an owned [`Frame`] on the evaluator hot path — the LDE
/// buffers are row-major, so a step is just two borrowed slices.
///
/// Requires single-row steps (step_size 1) — the only shape since
/// virtual columns were removed.
pub struct RowFrame<'a, F: IsSubFieldOf<E>, E: IsField> {
    mains: [&'a [FieldElement<F>]; MAX_TRANSITION_OFFSETS],
    auxs: [&'a [FieldElement<E>]; MAX_TRANSITION_OFFSETS],
    num_offsets: usize,
}

// Manual impls: the derives would demand `F: Copy`/`E: Copy`, but every field
// is a shared reference (or usize), which is Copy for any field type.
impl<F: IsSubFieldOf<E>, E: IsField> Clone for RowFrame<'_, F, E> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<F: IsSubFieldOf<E>, E: IsField> Copy for RowFrame<'_, F, E> {}

impl<'a, F: IsSubFieldOf<E>, E: IsField> RowFrame<'a, F, E> {
    /// Borrow the rows for LDE point `row` at each transition offset,
    /// wrapping cyclically at the domain end (the same cyclic row arithmetic
    /// the owned-Frame gather used, with single-row steps).
    pub fn from_lde(lde_trace: &'a LDETraceTable<F, E>, row: usize, offsets: &[usize]) -> Self {
        debug_assert_eq!(
            lde_trace.lde_step_size, lde_trace.blowup_factor,
            "RowFrame requires single-row steps (step_size 1)"
        );
        assert!(
            offsets.len() <= MAX_TRANSITION_OFFSETS,
            "RowFrame supports at most {MAX_TRANSITION_OFFSETS} transition offsets"
        );
        let num_rows = lde_trace.num_rows();
        let mut mains: [&'a [FieldElement<F>]; MAX_TRANSITION_OFFSETS] =
            [&[]; MAX_TRANSITION_OFFSETS];
        let mut auxs: [&'a [FieldElement<E>]; MAX_TRANSITION_OFFSETS] =
            [&[]; MAX_TRANSITION_OFFSETS];
        for (k, &offset) in offsets.iter().enumerate() {
            let idx = (row + offset * lde_trace.lde_step_size) % num_rows;
            mains[k] = lde_trace.main_row(idx);
            auxs[k] = lde_trace.aux_row(idx);
        }
        Self {
            mains,
            auxs,
            num_offsets: offsets.len(),
        }
    }

    /// The main-trace element at (offset position, column).
    #[inline(always)]
    pub fn main(&self, offset: usize, col: usize) -> &FieldElement<F> {
        &self.mains[offset][col]
    }

    /// The aux-trace element at (offset position, column).
    #[inline(always)]
    pub fn aux(&self, offset: usize, col: usize) -> &FieldElement<E> {
        &self.auxs[offset][col]
    }

    pub fn num_offsets(&self) -> usize {
        self.num_offsets
    }
}

/// A frame represents a collection of trace steps.
/// The collected steps are all the necessary steps for
/// all transition constraints over a trace to be evaluated.
///
/// Owns its row data so it can be built from either row-major Tables
/// (verifier) or column-major LDE data (prover) without lifetime issues.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame<F: IsSubFieldOf<E>, E: IsField> {
    steps: Vec<TableView<F, E>>,
}

impl<F: IsSubFieldOf<E>, E: IsField> Frame<F, E> {
    pub fn new(steps: Vec<TableView<F, E>>) -> Self {
        Self { steps }
    }

    pub fn get_evaluation_step(&self, step: usize) -> &TableView<F, E> {
        &self.steps[step]
    }

    /// Borrow this frame's single-row steps as a [`RowFrame`] — the bridge
    /// for callers that own a `Frame` (debug validation, tests); the
    /// evaluator hot loop uses [`RowFrame::from_lde`] directly.
    pub fn as_row_frame(&self) -> RowFrame<'_, F, E> {
        assert!(
            self.steps.len() <= MAX_TRANSITION_OFFSETS,
            "RowFrame supports at most {MAX_TRANSITION_OFFSETS} transition offsets"
        );
        let mut mains: [&[FieldElement<F>]; MAX_TRANSITION_OFFSETS] = [&[]; MAX_TRANSITION_OFFSETS];
        let mut auxs: [&[FieldElement<E>]; MAX_TRANSITION_OFFSETS] = [&[]; MAX_TRANSITION_OFFSETS];
        for (k, step) in self.steps.iter().enumerate() {
            debug_assert!(
                step.data.len() <= 1 && step.aux_data.len() <= 1,
                "RowFrame requires single-row steps (step_size 1)"
            );
            mains[k] = step.data.first().map(|r| r.as_slice()).unwrap_or(&[]);
            auxs[k] = step.aux_data.first().map(|r| r.as_slice()).unwrap_or(&[]);
        }
        RowFrame {
            mains,
            auxs,
            num_offsets: self.steps.len(),
        }
    }

    /// Build a Frame by gathering row data from a column-major LDETraceTable.
    ///
    /// Each step gathers elements from columns into owned Vecs. For the typical
    /// case (2 offsets, step_size=1), this gathers 2 rows of ~74 main + aux elements.
    fn read_from_lde(lde_trace: &LDETraceTable<F, E>, row: usize, offsets: &[usize]) -> Self {
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
}

#[cfg(test)]
mod row_frame_tests {
    use super::*;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;

    type Fp = FieldElement<Gl>;
    type Fp3 = FieldElement<Ext3>;

    /// An 8-row, 2-main/1-aux LDE table (blowup 2) with distinct per-cell
    /// values, so any mis-indexed read is caught by value.
    fn table() -> LDETraceTable<Gl, Ext3> {
        let main: Vec<Vec<Fp>> = (0..2)
            .map(|c| (0..8).map(|r| Fp::from((100 * c + r) as u64)).collect())
            .collect();
        let aux: Vec<Vec<Fp3>> = vec![
            (0..8)
                .map(|r| Fp3::new([Fp::from(1000 + r as u64), Fp::zero(), Fp::zero()]))
                .collect(),
        ];
        LDETraceTable::from_columns(main, aux, 1, 2)
    }

    #[test]
    fn borrows_rows_at_each_offset() {
        let t = table();
        let rows = RowFrame::from_lde(&t, 3, &[0, 1]);
        // offset 0 -> row 3; offset 1 -> row 3 + lde_step_size (= blowup 2) = 5.
        assert_eq!(rows.main(0, 0), t.get_main(3, 0));
        assert_eq!(rows.main(0, 1), t.get_main(3, 1));
        assert_eq!(rows.main(1, 0), t.get_main(5, 0));
        assert_eq!(rows.aux(0, 0), t.get_aux(3, 0));
        assert_eq!(rows.aux(1, 0), t.get_aux(5, 0));
        assert_eq!(rows.num_offsets(), 2);
    }

    #[test]
    fn wraps_cyclically_at_the_domain_end() {
        let t = table();
        // Last LDE row: offset 1 reads (7 + 2) % 8 = row 1.
        let rows = RowFrame::from_lde(&t, 7, &[0, 1]);
        assert_eq!(rows.main(0, 0), t.get_main(7, 0));
        assert_eq!(rows.main(1, 0), t.get_main(1, 0));
        assert_eq!(rows.aux(1, 0), t.get_aux(1, 0));
    }

    #[test]
    #[should_panic(expected = "at most")]
    fn rejects_too_many_offsets() {
        let t = table();
        let _ = RowFrame::from_lde(&t, 0, &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn as_row_frame_matches_owned_frame() {
        let t = table();
        let frame = Frame::read_step_from_lde(&t, 2, &[0, 1]);
        let rows = frame.as_row_frame();
        let direct = RowFrame::from_lde(&t, t.step_to_row(2), &[0, 1]);
        for offset in 0..2 {
            for col in 0..2 {
                assert_eq!(rows.main(offset, col), direct.main(offset, col));
            }
            assert_eq!(rows.aux(offset, 0), direct.aux(offset, 0));
        }
    }
}
