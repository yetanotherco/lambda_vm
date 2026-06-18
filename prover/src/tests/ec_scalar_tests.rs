//! Tests for the EC_SCALAR table — constraint satisfaction on generated traces,
//! the `last_limb` schedule, and the constraint count.

use crate::constraints::templates::IsBitConstraint;
use crate::tables::ec_scalar::{
    MulZeroConstraint, cols, create_constraints, generate_ec_scalar_trace, rows_for_scalar,
};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use stark::constraints::transition::TransitionConstraint;
use stark::table::TableView;
use stark::trace::TraceTable;

/// Builds a one-row `TableView` for `row` of the trace (constraints only read row 0).
fn row_view(
    trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
    row: usize,
) -> TableView<GoldilocksField, GoldilocksExtension> {
    let main: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *trace.main_table.get(row, c))
        .collect();
    TableView::new(vec![main], vec![])
}

#[test]
fn constraints_hold_on_generated_trace() {
    let mut k = [0u8; 32];
    // a scalar with assorted bit patterns across several bytes
    k[0] = 0b1010_0101;
    k[1] = 0xFF;
    k[15] = 0x80;
    k[31] = 0x01;
    let ops = rows_for_scalar(444, 0x3000, &k);
    let trace = generate_ec_scalar_trace(&ops);

    // IS_BIT columns
    let mut bit_cols = vec![cols::MU];
    bit_cols.extend((0..8).map(cols::limb_bit));
    bit_cols.push(cols::LAST_LIMB);

    for row in 0..trace.num_rows() {
        let view = row_view(&trace, row);
        for &col in &bit_cols {
            let v = IsBitConstraint::unconditional(col, 0).evaluate(&view);
            assert_eq!(v, FE::zero(), "IS_BIT col {col} row {row}");
        }
        // implication constraints
        for i in 0..8 {
            let c = MulZeroConstraint {
                a: cols::limb_bit(i),
                b: cols::MU,
                b_complement: true,
                constraint_idx: 0,
            };
            assert_eq!(c.evaluate(&view), FE::zero(), "limb_bit{i}=>mu row {row}");
        }
        let c = MulZeroConstraint {
            a: cols::LAST_LIMB,
            b: cols::MU,
            b_complement: true,
            constraint_idx: 0,
        };
        assert_eq!(c.evaluate(&view), FE::zero(), "last_limb=>mu row {row}");
        let c = MulZeroConstraint {
            a: cols::LAST_LIMB,
            b: cols::OFFSET,
            b_complement: false,
            constraint_idx: 0,
        };
        assert_eq!(c.evaluate(&view), FE::zero(), "last_limb=>offset row {row}");
    }
}

#[test]
fn last_limb_set_only_at_offset_zero() {
    let k = [7u8; 32];
    let ops = rows_for_scalar(4, 0x100, &k);
    assert_eq!(ops.len(), 32);
    for op in &ops {
        assert_eq!(op.last_limb, op.offset == 0);
    }
    // 32 distinct offsets 31..0
    assert_eq!(ops[0].offset, 31);
    assert_eq!(ops[31].offset, 0);
}

#[test]
fn create_constraints_count() {
    let (constraints, next) = create_constraints(0);
    assert_eq!(constraints.len(), 20);
    assert_eq!(next, 20);
}
