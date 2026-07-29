//! Shared, prover = verifier-identical helpers for out-of-domain (OOD) trace
//! opening pruning.
//!
//! The frame OOD table has `num_offsets * step_size` rows (offset-major: the
//! first `step_size` rows are offset 0's current-row block, and every later
//! offset contributes a `step_size`-row next-row block) and one column per
//! trace column. Only the columns a transition constraint actually reads at the
//! next row — the AIR's transition window,
//! [`crate::traits::AIR::trace_ood_next_row_columns`] — need to be
//! opened in the next-row block(s). Every other next-row entry is redundant and
//! is pruned from the proof.
//!
//! Everything here is a pure function of public AIR shape metadata (`step_size`,
//! the column count, and the next-row column set), so the prover and verifier
//! derive the identical layout without trusting proof dimensions (invariant I3).

use crate::table::Table;
use math::field::{element::FieldElement, traits::IsField};

/// Per-column flags: `flags[c] == true` iff column `c` is opened at the next
/// row. Indices outside `0..num_total_cols` are ignored.
pub fn next_row_col_flags(num_total_cols: usize, next_row_cols: &[usize]) -> Vec<bool> {
    let mut flags = vec![false; num_total_cols];
    for &c in next_row_cols {
        if c < num_total_cols {
            flags[c] = true;
        }
    }
    flags
}

/// Number of surviving trace openings: the current-row block opens every column
/// (`step_size * num_total_cols`), and each next-row row opens only the masked
/// columns (`(num_eval_points - step_size) * num_next_row_cols`).
pub fn num_surviving_trace_openings(
    num_total_cols: usize,
    num_eval_points: usize,
    step_size: usize,
    num_next_row_cols: usize,
) -> usize {
    let next_rows = num_eval_points.saturating_sub(step_size);
    step_size * num_total_cols + next_rows * num_next_row_cols
}

/// Build the rectangular `num_total_cols x num_eval_points` DEEP trace-term
/// coefficient grid from `powers` (the `num_surviving_trace_openings` gamma
/// powers drained for the trace terms). Surviving positions receive a power in a
/// fixed order; pruned next-row positions receive zero. A rectangular DEEP
/// evaluation over the full grid therefore yields the identical polynomial as
/// summing only the survivors — which is what lets the prover keep its
/// (GPU-friendly) rectangular DEEP unchanged.
///
/// Precondition: `powers.len() == num_surviving_trace_openings(num_total_cols,
/// num_eval_points, step_size, next_row_cols.len())` for the same layout args —
/// every power binds to exactly one surviving position and every surviving
/// position consumes exactly one power. Both operands are AIR-metadata-derived
/// (invariant I3), so this holds for every real AIR; a debug build checks it.
///
/// Assignment order (mirrored exactly by [`num_surviving_trace_openings`]):
///   1. current-row block — for every column `j`, rows `0..step_size`;
///   2. next-row block — for each masked column `j`, rows `step_size..num_eval_points`.
pub fn build_pruned_trace_term_coeffs<E: IsField>(
    powers: &[FieldElement<E>],
    num_total_cols: usize,
    num_eval_points: usize,
    step_size: usize,
    next_row_cols: &[usize],
) -> Vec<Vec<FieldElement<E>>> {
    let flags = next_row_col_flags(num_total_cols, next_row_cols);
    let mut coeffs = vec![vec![FieldElement::<E>::zero(); num_eval_points]; num_total_cols];
    let mut p = 0usize;
    // Current-row block: all columns, rows 0..step_size.
    for col in coeffs.iter_mut() {
        for slot in col.iter_mut().take(step_size) {
            if p < powers.len() {
                *slot = powers[p].clone();
                p += 1;
            }
        }
    }
    // Next-row block(s): masked columns only, rows step_size..num_eval_points.
    for (j, col) in coeffs.iter_mut().enumerate() {
        if flags[j] {
            for slot in col.iter_mut().take(num_eval_points).skip(step_size) {
                if p < powers.len() {
                    *slot = powers[p].clone();
                    p += 1;
                }
            }
        }
    }
    debug_assert_eq!(p, powers.len(), "power assignment must consume every power");
    coeffs
}

/// Split the full `num_eval_points x num_total_cols` OOD table (computed by the
/// prover) into the two blocks carried by the proof:
///   * block 0 — the current-row block, `step_size x num_total_cols` (all columns);
///   * block 1 — the next-row block, `next_rows x num_next_row_cols`, holding only
///     the masked columns in `next_row_cols` order.
///
/// Block 1 has width 0 (an empty table) when the AIR reads no next-row columns.
pub fn split_ood_blocks<E: IsField>(
    full: &Table<E>,
    step_size: usize,
    next_row_cols: &[usize],
) -> (Table<E>, Table<E>) {
    let w = full.width;

    let mut b0 = Vec::with_capacity(step_size * w);
    for r in 0..step_size {
        b0.extend_from_slice(full.get_row(r));
    }
    let block0 = Table::new(b0, w);

    let mut b1 = Vec::with_capacity((full.height.saturating_sub(step_size)) * next_row_cols.len());
    for r in step_size..full.height {
        let row = full.get_row(r);
        for &c in next_row_cols {
            b1.push(row[c].clone());
        }
    }
    let block1 = Table::new(b1, next_row_cols.len());

    (block0, block1)
}

/// Rebuild the full `num_eval_points x width` OOD table from the two pruned
/// proof blocks, given as row-major slices (a [`Table`]'s `row_major_data()` or a
/// [`crate::proof::view::StarkTableView`]'s, so this stays decoupled from owned
/// vs. rkyv-archived proofs). Current-row rows come straight from `current_block`;
/// each next-row row scatters the masked values from `next_block` into their
/// columns and leaves every other column zero. Those zero entries are never read
/// — no transition constraint references a pruned column at the next row, and
/// DEEP skips them — so the reconstruction is exact where it matters.
///
/// Reads are bounds-checked (`.get`): a malformed archive whose advertised
/// dimensions disagree with its data length yields a zero-filled grid rather than
/// a panic, and fails the downstream consistency checks instead.
pub fn reconstruct_ood_full<E: IsField>(
    current_block: &[FieldElement<E>],
    width: usize,
    next_block: &[FieldElement<E>],
    num_eval_points: usize,
    step_size: usize,
    next_row_cols: &[usize],
) -> Table<E> {
    let mask_width = next_row_cols.len();
    let mut data = Vec::with_capacity(num_eval_points * width);

    for r in 0..step_size {
        for c in 0..width {
            data.push(
                current_block
                    .get(r * width + c)
                    .cloned()
                    .unwrap_or_else(FieldElement::<E>::zero),
            );
        }
    }

    // Zero-fill the next-row rows, then scatter the surviving masked values
    // directly into their columns instead of scanning `next_row_cols` per
    // cell. `.max` keeps the current-row block intact even if
    // `num_eval_points < step_size` (defensive only: for a well-formed AIR
    // `num_eval_points` is always a positive multiple of `step_size`).
    data.resize(
        data.len().max(num_eval_points * width),
        FieldElement::<E>::zero(),
    );
    for next_row in 0..num_eval_points.saturating_sub(step_size) {
        let row_base = (step_size + next_row) * width;
        for (m, &mc) in next_row_cols.iter().enumerate() {
            if mc < width
                && let Some(v) = next_block.get(next_row * mask_width + m)
            {
                data[row_base + mc] = v.clone();
            }
        }
    }

    Table::new(data, width)
}

/// The pruned-OOD trace-opening layout, derived once from public AIR shape
/// metadata and shared by every site that used to recompute it. Every field is
/// a pure function of the AIR (`trace_columns`, `step_size`, the
/// transition-offset count, and the next-row column set), so the prover and the
/// verifier build the identical layout without trusting any proof dimension
/// (invariant I3). This struct only bundles those values and forwards to the
/// free functions above; it adds no new arithmetic.
///
/// It stays decoupled from the `AIR` trait: callers that have an AIR in scope
/// read the four raw values once (see the `ood_layout` helpers in the verifier
/// and prover) and pass them to [`OodLayout::new`].
#[derive(Clone, Debug)]
pub struct OodLayout {
    /// Total trace columns (`main + aux`), i.e. the full current-row block width.
    num_total_cols: usize,
    /// Rows in the full OOD grid: `num_transition_offsets * step_size`.
    num_eval_points: usize,
    /// Rows per offset block.
    step_size: usize,
    /// Full-width column indices opened at the next row (the transition window).
    next_row_cols: Vec<usize>,
}

impl OodLayout {
    /// Build from raw AIR-metadata values. `num_eval_points` is
    /// `num_transition_offsets * step_size`; keeping it a plain argument lets the
    /// single AIR-reading expression live in the verifier/prover, not here.
    pub fn new(
        num_total_cols: usize,
        num_eval_points: usize,
        step_size: usize,
        next_row_cols: Vec<usize>,
    ) -> Self {
        Self {
            num_total_cols,
            num_eval_points,
            step_size,
            next_row_cols,
        }
    }

    /// Rows per offset block.
    pub fn step_size(&self) -> usize {
        self.step_size
    }

    /// Full-width column indices opened at the next row (the transition window),
    /// in the order the DEEP reconstruction sums them.
    pub fn next_row_cols(&self) -> &[usize] {
        &self.next_row_cols
    }

    /// Width of the pruned next-row proof block: one column per transition-window
    /// column (the current-row block always keeps every column).
    pub fn expected_next_width(&self) -> usize {
        self.next_row_cols.len()
    }

    /// Height of the pruned next-row proof block: the non-current rows, or 0 when
    /// the AIR reads no next-row column (then the block is empty).
    pub fn expected_next_height(&self) -> usize {
        if self.next_row_cols.is_empty() {
            0
        } else {
            self.num_eval_points.saturating_sub(self.step_size)
        }
    }

    /// Number of surviving trace openings under g·z pruning; see
    /// [`num_surviving_trace_openings`].
    pub fn num_surviving(&self) -> usize {
        num_surviving_trace_openings(
            self.num_total_cols,
            self.num_eval_points,
            self.step_size,
            self.next_row_cols.len(),
        )
    }

    /// Per-column next-row open flags for a table of `grid_width` columns; see
    /// [`next_row_col_flags`]. The width is that of the table being indexed — the
    /// reconstructed OOD grid, whose width is the current-row block's width — and
    /// need not equal `num_total_cols`; the free function ignores any next-row
    /// index that falls outside `grid_width`.
    pub fn flags(&self, grid_width: usize) -> Vec<bool> {
        next_row_col_flags(grid_width, &self.next_row_cols)
    }

    /// Build the rectangular DEEP trace-term coefficient grid; see
    /// [`build_pruned_trace_term_coeffs`].
    pub fn build_trace_term_coeffs<E: IsField>(
        &self,
        powers: &[FieldElement<E>],
    ) -> Vec<Vec<FieldElement<E>>> {
        build_pruned_trace_term_coeffs(
            powers,
            self.num_total_cols,
            self.num_eval_points,
            self.step_size,
            &self.next_row_cols,
        )
    }

    /// Split a full prover OOD table into the two pruned proof blocks; see
    /// [`split_ood_blocks`].
    pub fn split_full<E: IsField>(&self, full: &Table<E>) -> (Table<E>, Table<E>) {
        split_ood_blocks(full, self.step_size, &self.next_row_cols)
    }

    /// Rebuild the full OOD grid from the two pruned proof blocks; see
    /// [`reconstruct_ood_full`]. `current_width` is the (proof-supplied)
    /// current-row block width and becomes the reconstructed grid's width.
    pub fn reconstruct_full<E: IsField>(
        &self,
        current_block: &[FieldElement<E>],
        current_width: usize,
        next_block: &[FieldElement<E>],
    ) -> Table<E> {
        reconstruct_ood_full(
            current_block,
            current_width,
            next_block,
            self.num_eval_points,
            self.step_size,
            &self.next_row_cols,
        )
    }
}
