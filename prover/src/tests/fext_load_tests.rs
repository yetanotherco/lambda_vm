//! Tests for the FEXT_LOAD table: constraint count, bus count, trace layout,
//! and the μ bit constraint.

use crate::tables::fext_load::{
    FextLoadConstraints, FextLoadOperation, bus_interactions, cols, generate_fext_load_trace,
};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};
use math::field::element::FieldElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

fn eval_main_row(main: Vec<FE>) -> Vec<FE> {
    let n = FextLoadConstraints.meta().len();
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
    FextLoadConstraints.eval(&mut folder);
    base
}

fn eval_row(trace: &TraceTable<GoldilocksField, GoldilocksExtension>, row: usize) -> Vec<FE> {
    let main: Vec<FE> = (0..cols::NUM_COLUMNS)
        .map(|c| *trace.main_table.get(row, c))
        .collect();
    eval_main_row(main)
}

fn op(addr: u64, coeffs: [u64; 3]) -> FextLoadOperation {
    FextLoadOperation {
        timestamp: 100,
        addr,
        coeffs,
        old_ts: [0; 3],
        old_val: [0; 3],
    }
}

#[test]
fn fext_load_constraint_count_is_one() {
    // Only IS_BIT(μ); coefficient range checks are bus interactions.
    assert_eq!(FextLoadConstraints.meta().len(), 1);
}

#[test]
fn fext_load_bus_interaction_count() {
    // 1 Ecall receiver + 4 register reads + 3 range checks + 3 field-storage
    // writes × (consume-old, emit-new, old_ts<ts) = 1 + 4 + 3 + 9.
    assert_eq!(bus_interactions().len(), 17);
}

#[test]
fn fext_load_trace_decomposes_coeffs_into_words() {
    // Coefficient with both limbs non-zero.
    let coeff = 0x1234_5678_9ABC_DEF0u64;
    let trace = generate_fext_load_trace(&[op(0xAA, [coeff, 7, 0])]);
    let t = &trace.main_table;

    // addr and timestamp low/high limbs.
    assert_eq!(*t.get(0, cols::ADDR_0), FE::from(0xAAu64));
    assert_eq!(*t.get(0, cols::ADDR_1), FE::from(0u64));
    assert_eq!(*t.get(0, cols::TIMESTAMP_0), FE::from(100u64));

    // coeff 0 split into words.
    assert_eq!(*t.get(0, cols::C0_0), FE::from(0x9ABC_DEF0u64));
    assert_eq!(*t.get(0, cols::C0_1), FE::from(0x1234_5678u64));
    // coeff 1 = 7 (low limb only).
    assert_eq!(*t.get(0, cols::C1_0), FE::from(7u64));
    assert_eq!(*t.get(0, cols::C1_1), FE::from(0u64));

    assert_eq!(*t.get(0, cols::MU), FE::one());
}

#[test]
fn fext_load_trace_shape_and_padding() {
    let trace = generate_fext_load_trace(&[op(1, [1, 2, 3])]);
    assert_eq!(trace.num_rows(), 4);
    assert_eq!(*trace.main_table.get(0, cols::MU), FE::one());
    for row in 1..4 {
        assert_eq!(*trace.main_table.get(row, cols::MU), FE::zero());
    }
}

#[test]
fn fext_load_mu_bit_constraint_holds_and_detects_violation() {
    let mut trace = generate_fext_load_trace(&[op(1, [1, 2, 3])]);
    // Valid: every row satisfies IS_BIT(μ).
    for row in 0..trace.num_rows() {
        assert_eq!(eval_row(&trace, row)[0], FE::zero(), "row {row}");
    }
    // μ = 2 breaks it.
    trace.main_table.set_fe(0, cols::MU, FE::from(2u64));
    assert_ne!(eval_row(&trace, 0)[0], FE::zero());
}
