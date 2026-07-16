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

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField as Gl;

    type Fe = FieldElement<Gl>;

    fn fe(x: u64) -> Fe {
        Fe::from(x)
    }

    #[test]
    fn surviving_count_matches_layout() {
        // 3 columns, 2 eval points (step_size 1), 1 next-row column:
        // current-row opens 3, next-row opens 1 => 4.
        assert_eq!(num_surviving_trace_openings(3, 2, 1, 1), 4);
        // No next-row columns => only the current-row block survives.
        assert_eq!(num_surviving_trace_openings(3, 2, 1, 0), 3);
        // Every column open at the next row => full 2*W grid.
        assert_eq!(num_surviving_trace_openings(3, 2, 1, 3), 6);
    }

    #[test]
    fn split_then_reconstruct_preserves_survivors_and_zeros_pruned() {
        // Full 2x3 OOD table: row 0 (current row), row 1 (next row).
        let full = Table::new(vec![fe(10), fe(11), fe(12), fe(20), fe(21), fe(22)], 3);
        let next_row_cols = [1usize]; // only column 1 opens at the next row
        let step_size = 1;

        let (b0, b1) = split_ood_blocks(&full, step_size, &next_row_cols);
        assert_eq!((b0.width, b0.height), (3, 1));
        assert_eq!((b1.width, b1.height), (1, 1));
        assert_eq!(b1.get_row(0)[0], fe(21)); // full[1][1]

        let recon = reconstruct_ood_full(
            b0.row_major_data(),
            b0.width,
            b1.row_major_data(),
            2,
            step_size,
            &next_row_cols,
        );
        assert_eq!(recon.get_row(0), full.get_row(0)); // current row is exact
        assert_eq!(recon.get_row(1)[1], fe(21)); // survivor placed
        assert_eq!(recon.get_row(1)[0], Fe::zero()); // pruned -> zero
        assert_eq!(recon.get_row(1)[2], Fe::zero()); // pruned -> zero
    }

    #[test]
    fn empty_next_row_block_reconstructs_to_zeros() {
        let full = Table::new(vec![fe(10), fe(11), fe(20), fe(21)], 2);
        let (b0, b1) = split_ood_blocks(&full, 1, &[]);
        assert_eq!(b1.width, 0);
        let recon = reconstruct_ood_full(
            b0.row_major_data(),
            b0.width,
            b1.row_major_data(),
            2,
            1,
            &[],
        );
        assert_eq!(recon.get_row(0), full.get_row(0));
        assert_eq!(recon.get_row(1), &[Fe::zero(), Fe::zero()]);
    }

    #[test]
    fn out_of_range_next_row_col_is_ignored_not_panicking() {
        // width = 3, but next_row_cols advertises column 5 -- out of range.
        let current_block = vec![fe(1), fe(2), fe(3)];
        let next_block = vec![fe(99)]; // would-be value for the bogus column
        let recon = reconstruct_ood_full(&current_block, 3, &next_block, 2, 1, &[5]);
        assert_eq!(recon.get_row(0), &[fe(1), fe(2), fe(3)]);
        assert_eq!(recon.get_row(1), &[Fe::zero(), Fe::zero(), Fe::zero()]);
    }

    #[test]
    fn short_next_block_leaves_missing_cells_zero_not_panicking() {
        // width = 3, 3 eval points (step_size 1) => 2 next rows, mask = {0, 2}
        // so the mask implies 4 next-row values, but next_block only has 1.
        let current_block = vec![fe(1), fe(2), fe(3)];
        let next_block = vec![fe(99)];
        let recon = reconstruct_ood_full(&current_block, 3, &next_block, 3, 1, &[0, 2]);
        assert_eq!(recon.get_row(0), &[fe(1), fe(2), fe(3)]);
        assert_eq!(recon.get_row(1), &[fe(99), Fe::zero(), Fe::zero()]); // only present value scattered
        assert_eq!(recon.get_row(2), &[Fe::zero(), Fe::zero(), Fe::zero()]); // fully missing -> zero
    }

    #[test]
    fn pruned_coeffs_are_zero_off_the_window() {
        // 4 surviving powers for W=3, num_eval_points=2, mask={1}.
        let powers: Vec<Fe> = (1..=4).map(fe).collect();
        let coeffs = build_pruned_trace_term_coeffs(&powers, 3, 2, 1, &[1]);
        // Current-row row (k=0) is fully populated; next-row row (k=1) only col 1.
        assert_ne!(coeffs[0][0], Fe::zero());
        assert_ne!(coeffs[1][0], Fe::zero());
        assert_ne!(coeffs[2][0], Fe::zero());
        assert_ne!(coeffs[1][1], Fe::zero()); // masked column, next row
        assert_eq!(coeffs[0][1], Fe::zero()); // pruned
        assert_eq!(coeffs[2][1], Fe::zero()); // pruned
    }

    #[test]
    fn ood_layout_delegates_to_free_functions() {
        // W=3 cols, num_eval_points=2 (step_size 1, 2 offsets), next-row mask {1}.
        let layout = OodLayout::new(3, 2, 1, vec![1]);

        assert_eq!(layout.step_size(), 1);
        assert_eq!(layout.expected_next_width(), 1);
        assert_eq!(layout.expected_next_height(), 1);
        assert_eq!(
            layout.num_surviving(),
            num_surviving_trace_openings(3, 2, 1, 1)
        );

        // Empty next-row mask => empty next-row block.
        let empty = OodLayout::new(3, 2, 1, vec![]);
        assert_eq!(empty.expected_next_width(), 0);
        assert_eq!(empty.expected_next_height(), 0);

        // flags(), build_trace_term_coeffs(), split_full() and reconstruct_full()
        // must be bit-identical to the free functions they forward to.
        assert_eq!(layout.flags(3), next_row_col_flags(3, &[1]));
        let powers: Vec<Fe> = (1..=4).map(fe).collect();
        assert_eq!(
            layout.build_trace_term_coeffs(&powers),
            build_pruned_trace_term_coeffs(&powers, 3, 2, 1, &[1])
        );

        let full = Table::new(vec![fe(10), fe(11), fe(12), fe(20), fe(21), fe(22)], 3);
        let (lb0, lb1) = layout.split_full(&full);
        let (fb0, fb1) = split_ood_blocks(&full, 1, &[1]);
        assert_eq!(lb0.row_major_data(), fb0.row_major_data());
        assert_eq!(lb1.row_major_data(), fb1.row_major_data());

        let recon = layout.reconstruct_full(lb0.row_major_data(), lb0.width, lb1.row_major_data());
        let free_recon = reconstruct_ood_full(
            fb0.row_major_data(),
            fb0.width,
            fb1.row_major_data(),
            2,
            1,
            &[1],
        );
        assert_eq!(recon.row_major_data(), free_recon.row_major_data());
    }
}
