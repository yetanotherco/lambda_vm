//! Tests for the EC_SCALAR table — constraint satisfaction on generated traces,
//! the `last_limb` schedule, and the single-source constraint count.

use crate::tables::ec_scalar::{
    EcScalarConstraints, cols, generate_ec_scalar_trace, rows_for_scalar,
};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use math::field::element::FieldElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

/// Evaluate the EC_SCALAR [`ConstraintSet`] on one trace row (the compiled
/// prover folder path), returning every base-field constraint value.
fn eval_row(trace: &TraceTable<GoldilocksField, GoldilocksExtension>, row: usize) -> Vec<FE> {
    let main: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *trace.main_table.get(row, c))
        .collect();
    let n = EcScalarConstraints.meta().len();
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
        vec![main],
        vec![vec![]],
    )]);
    let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset_e = FieldElement::<GoldilocksExtension>::zero();
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
    let mut base = vec![FE::zero(); n];
    let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
    EcScalarConstraints.eval(&mut folder);
    base
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

    for row in 0..trace.num_rows() {
        for (i, v) in eval_row(&trace, row).iter().enumerate() {
            assert_eq!(*v, FE::zero(), "constraint {i} must hold at row {row}");
        }
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
fn constraint_set_count() {
    assert_eq!(EcScalarConstraints.meta().len(), 20);
}
