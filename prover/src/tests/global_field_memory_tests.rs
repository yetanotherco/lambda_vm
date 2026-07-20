//! Tests for the GLOBAL_FIELD_MEMORY cross-epoch field-storage aggregation table.

use crate::tables::global_field_memory::{
    FieldCellFinal, GlobalFieldMemoryConstraints, bus_interactions, cols,
    generate_global_field_trace,
};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};
use math::field::element::FieldElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

/// Evaluate every GLOBAL_FIELD_MEMORY constraint over the transition frame
/// `(row, row+1)`.
fn eval_transition(
    trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
    row: usize,
) -> Vec<FE> {
    let n = GlobalFieldMemoryConstraints.meta().len();
    let get_row = |r: usize| -> Vec<FE> {
        (0..cols::NUM_COLUMNS)
            .map(|c| *trace.main_table.get(r, c))
            .collect()
    };
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![
        TableView::new(vec![get_row(row)], vec![vec![]]),
        TableView::new(vec![get_row(row + 1)], vec![vec![]]),
    ]);
    let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset_e = FieldElement::<GoldilocksExtension>::zero();
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
    let mut base = vec![FE::zero(); n];
    let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
    GlobalFieldMemoryConstraints.eval(&mut folder);
    base
}

fn cell(domain: u64, addr: u64) -> FieldCellFinal {
    FieldCellFinal {
        domain,
        addr,
        value: 42,
        epoch: 2,
    }
}

#[test]
fn global_field_memory_constraint_and_bus_counts() {
    // Same sorted-keys shape as FEXT_PAGE: 11 constraints.
    assert_eq!(GlobalFieldMemoryConstraints.meta().len(), 11);
    // GFM-GENESIS + GFM-FINAL + addr LT + 4 IsHalfword = 7.
    assert_eq!(bus_interactions().len(), 7);
}

#[test]
fn global_field_memory_trace_layout_and_padding() {
    let cells = vec![
        FieldCellFinal {
            domain: 3,
            addr: 0x10,
            value: 42,
            epoch: 1,
        },
        FieldCellFinal {
            domain: 5,
            addr: 0x20,
            value: 7,
            epoch: 3,
        },
    ];
    let trace = generate_global_field_trace(&cells);
    assert_eq!(trace.num_rows(), 4); // 2 cells padded to 4

    let t = &trace.main_table;
    assert_eq!(*t.get(0, cols::DOMAIN), FE::from(3u64));
    assert_eq!(*t.get(0, cols::FINI_VAL), FE::from(42u64));
    assert_eq!(*t.get(0, cols::FINI_EPOCH), FE::from(1u64));
    assert_eq!(*t.get(0, cols::MU), FE::one());
    assert_eq!(*t.get(1, cols::DOMAIN), FE::from(5u64));

    for row in 2..4 {
        assert_eq!(*t.get(row, cols::MU), FE::zero());
        assert_eq!(*t.get(row, cols::DOMAIN), FE::from(3u64));
    }
}

#[test]
fn global_field_memory_sorts_by_domain_then_addr() {
    let cells = vec![cell(5, 0x30), cell(3, 0x40), cell(4, 0x10), cell(3, 0x20)];
    let trace = generate_global_field_trace(&cells);
    let t = &trace.main_table;
    let key = |row: usize| (*t.get(row, cols::DOMAIN), *t.get(row, cols::ADDR_0));
    assert_eq!(key(0), (FE::from(3u64), FE::from(0x20u64)));
    assert_eq!(key(1), (FE::from(3u64), FE::from(0x40u64)));
    assert_eq!(key(2), (FE::from(4u64), FE::from(0x10u64)));
    assert_eq!(key(3), (FE::from(5u64), FE::from(0x30u64)));
}

#[test]
fn global_field_memory_addr_limb_halfword_decomposition() {
    let addr = (0xABCDu64 << 48) | (0x1234u64 << 32) | (0x5678u64 << 16) | 0x9ABC;
    let trace = generate_global_field_trace(&[cell(3, addr)]);
    let t = &trace.main_table;
    assert_eq!(*t.get(0, cols::ADDR0_HW_LO), FE::from(0x9ABCu64));
    assert_eq!(*t.get(0, cols::ADDR0_HW_HI), FE::from(0x5678u64));
    assert_eq!(*t.get(0, cols::ADDR1_HW_LO), FE::from(0x1234u64));
    assert_eq!(*t.get(0, cols::ADDR1_HW_HI), FE::from(0xABCDu64));
}

#[test]
fn global_field_memory_constraints_hold_on_valid_trace() {
    let cells = vec![cell(3, 0x10), cell(3, 0x20), cell(4, 0x08)];
    let trace = generate_global_field_trace(&cells);
    for row in 0..trace.num_rows() - 1 {
        for (idx, v) in eval_transition(&trace, row).into_iter().enumerate() {
            assert_eq!(v, FE::zero(), "row {row}, constraint {idx} should be zero");
        }
    }
}

#[test]
fn global_field_memory_rejects_forged_domain() {
    let mut trace = generate_global_field_trace(&[cell(3, 0x10)]);
    trace.main_table.set_fe(0, cols::DOMAIN, FE::from(0u64)); // domain 0 = RAM
    assert_ne!(eval_transition(&trace, 0)[1], FE::zero());
}

#[test]
fn global_field_memory_rejects_domain_decrease() {
    let mut trace = generate_global_field_trace(&[cell(3, 0x10), cell(4, 0x20)]);
    trace.main_table.set_fe(0, cols::DOMAIN, FE::from(5u64));
    assert_ne!(eval_transition(&trace, 0)[8], FE::zero());
}

#[test]
fn global_field_memory_rejects_active_row_after_padding() {
    let mut trace = generate_global_field_trace(&[cell(3, 0x10)]);
    trace.main_table.set_fe(2, cols::MU, FE::one()); // row 1 padding, row 2 "active"
    assert_ne!(eval_transition(&trace, 1)[5], FE::zero());
}

#[test]
fn global_field_memory_rejects_mismatched_same_dom() {
    let mut trace = generate_global_field_trace(&[cell(3, 0x10), cell(3, 0x20)]);
    trace.main_table.set_fe(0, cols::SAME_DOM, FE::zero());
    trace.main_table.set_fe(0, cols::SEL_SAME, FE::zero());
    let base = eval_transition(&trace, 0);
    assert!(base[6] != FE::zero() || base[8] != FE::zero());
}

#[test]
fn global_field_memory_rejects_forged_next_addr() {
    let mut trace = generate_global_field_trace(&[cell(3, 0x10), cell(3, 0x20)]);
    trace
        .main_table
        .set_fe(0, cols::NEXT_ADDR_0, FE::from(0x999u64));
    assert_ne!(eval_transition(&trace, 0)[9], FE::zero());
}
