//! Tests for the FEXT_LOCAL_TO_GLOBAL per-epoch field-storage bookend table.

use crate::tables::fext_local_to_global::{
    FextLocalToGlobalConstraints, FieldCellBoundary, collect_bitwise_from_fext_l2g,
    collect_lt_from_touches, cols, generate_fext_local_to_global_trace, global_bus_interactions,
    memory_bus_interactions, range_check_interactions,
};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};
use math::field::element::FieldElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

fn eval_transition(
    trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
    row: usize,
) -> Vec<FE> {
    let n = FextLocalToGlobalConstraints.meta().len();
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
    FextLocalToGlobalConstraints.eval(&mut folder);
    base
}

fn boundary(domain: u64, addr: u64) -> FieldCellBoundary {
    FieldCellBoundary {
        domain,
        addr,
        init_val: 11,
        init_epoch: 1,
        final_val: 42,
        final_ts: 100,
    }
}

#[test]
fn fext_l2g_constraint_and_bus_counts() {
    assert_eq!(FextLocalToGlobalConstraints.meta().len(), 12);
    // cross-epoch init receiver + fini sender.
    assert_eq!(global_bus_interactions(2).len(), 2);
    // epoch-local Memory init receiver + fini sender.
    assert_eq!(memory_bus_interactions().len(), 2);
    // 4 addr IsHalfword + 2 init_epoch IsHalfword + 1 IsB20 + 1 addr LT = 8.
    assert_eq!(range_check_interactions(2).len(), 8);
}

#[test]
fn fext_l2g_trace_layout_and_padding() {
    let cells = vec![
        FieldCellBoundary {
            domain: 3,
            addr: 0x10,
            init_val: 5,
            init_epoch: 0,
            final_val: 42,
            final_ts: 100,
        },
        FieldCellBoundary {
            domain: 5,
            addr: 0x20,
            init_val: 9,
            init_epoch: 2,
            final_val: 7,
            final_ts: 200,
        },
    ];
    let trace = generate_fext_local_to_global_trace(&cells);
    assert_eq!(trace.num_rows(), 4);

    let t = &trace.main_table;
    assert_eq!(*t.get(0, cols::DOMAIN), FE::from(3u64));
    assert_eq!(*t.get(0, cols::INIT_VAL), FE::from(5u64));
    assert_eq!(*t.get(0, cols::INIT_EPOCH_0), FE::from(0u64));
    assert_eq!(*t.get(0, cols::FINAL_VAL), FE::from(42u64));
    assert_eq!(*t.get(0, cols::FINAL_TS_0), FE::from(100u64));
    assert_eq!(*t.get(0, cols::MU), FE::one());
    // Row 1: init_epoch 2 → halfword low = 2.
    assert_eq!(*t.get(1, cols::INIT_EPOCH_0), FE::from(2u64));

    for row in 2..4 {
        assert_eq!(*t.get(row, cols::MU), FE::zero());
        assert_eq!(*t.get(row, cols::DOMAIN), FE::from(3u64));
    }
}

#[test]
fn fext_l2g_bitwise_collector_count() {
    // 7 lookups per cell: 4 addr IsHalfword + 2 init_epoch IsHalfword + 1 IsB20.
    let cells = vec![boundary(3, 0x10), boundary(4, 0x20)];
    assert_eq!(collect_bitwise_from_fext_l2g(&cells, 5).len(), 2 * 7);
}

#[test]
fn fext_l2g_lt_collector_same_domain_windows() {
    // Same-domain consecutive cells yield one addr-LT each; the domain change does not.
    let touched = vec![
        (3u64, 0x10u64, 0u64, 0u64),
        (3, 0x20, 0, 0),
        (4, 0x08, 0, 0),
    ];
    // (3,0x10)<(3,0x20) is one LT; (3,0x20)->(4,0x08) crosses domains → none.
    assert_eq!(collect_lt_from_touches(&touched).len(), 1);
}

#[test]
fn fext_l2g_constraints_hold_on_valid_trace() {
    let cells = vec![boundary(3, 0x10), boundary(3, 0x20), boundary(4, 0x08)];
    let trace = generate_fext_local_to_global_trace(&cells);
    for row in 0..trace.num_rows() - 1 {
        for (idx, v) in eval_transition(&trace, row).into_iter().enumerate() {
            assert_eq!(v, FE::zero(), "row {row}, constraint {idx} should be zero");
        }
    }
}

#[test]
fn fext_l2g_rejects_forged_domain() {
    let mut trace = generate_fext_local_to_global_trace(&[boundary(3, 0x10)]);
    trace.main_table.set_fe(0, cols::DOMAIN, FE::from(0u64));
    assert_ne!(eval_transition(&trace, 0)[1], FE::zero());
}

#[test]
fn fext_l2g_rejects_domain_decrease() {
    let mut trace = generate_fext_local_to_global_trace(&[boundary(3, 0x10), boundary(4, 0x20)]);
    trace.main_table.set_fe(0, cols::DOMAIN, FE::from(5u64));
    assert_ne!(eval_transition(&trace, 0)[8], FE::zero());
}

#[test]
fn fext_l2g_rejects_active_row_after_padding() {
    let mut trace = generate_fext_local_to_global_trace(&[boundary(3, 0x10)]);
    trace.main_table.set_fe(2, cols::MU, FE::one());
    assert_ne!(eval_transition(&trace, 1)[5], FE::zero());
}

#[test]
fn fext_l2g_rejects_mismatched_same_dom() {
    let mut trace = generate_fext_local_to_global_trace(&[boundary(3, 0x10), boundary(3, 0x20)]);
    trace.main_table.set_fe(0, cols::SAME_DOM, FE::zero());
    trace.main_table.set_fe(0, cols::SEL_SAME, FE::zero());
    let base = eval_transition(&trace, 0);
    assert!(base[6] != FE::zero() || base[8] != FE::zero());
}

#[test]
fn fext_l2g_rejects_forged_next_addr() {
    let mut trace = generate_fext_local_to_global_trace(&[boundary(3, 0x10), boundary(3, 0x20)]);
    trace
        .main_table
        .set_fe(0, cols::NEXT_ADDR_0, FE::from(0x999u64));
    assert_ne!(eval_transition(&trace, 0)[9], FE::zero());
}
