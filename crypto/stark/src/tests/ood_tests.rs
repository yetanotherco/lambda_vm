use crate::ood::{
    OodLayout, build_pruned_trace_term_coeffs, next_row_col_flags, num_surviving_trace_openings,
    reconstruct_ood_full, split_ood_blocks,
};
use crate::table::Table;
use math::field::element::FieldElement;
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
