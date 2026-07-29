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
