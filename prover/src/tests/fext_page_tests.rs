//! Tests for the FEXT_PAGE bookend table.

use crate::tables::fext_page::{
    FextPageConstraints, FextPageOperation, bus_interactions, cols, generate_fext_page_trace,
};
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};
use math::field::element::FieldElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

/// Evaluate every FEXT_PAGE constraint over the transition frame `(row, row+1)`.
/// Per-row constraints see `row`; transition constraints see both.
fn eval_transition(
    trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
    row: usize,
) -> Vec<FE> {
    let n = FextPageConstraints.meta().len();
    let get_row = |r: usize| -> Vec<FE> {
        (0..cols::NUM_COLUMNS)
            .map(|c| *trace.main_table.get(r, c))
            .collect()
    };
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![
        TableView::new(vec![get_row(row)], vec![vec![]]),
        TableView::new(vec![get_row((row + 1) % trace.num_rows())], vec![vec![]]),
    ]);
    let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset_e = FieldElement::<GoldilocksExtension>::zero();
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
    let mut base = vec![FE::zero(); n];
    let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
    FextPageConstraints.eval(&mut folder);
    base
}

fn op(domain: u64, addr: u64) -> FextPageOperation {
    FextPageOperation {
        domain,
        addr,
        final_ts: 100,
        final_val: 42,
    }
}

#[test]
fn fext_page_constraint_and_bus_counts() {
    // IS_BIT(μ), domain ∈ {3,4,5}, IS_BIT(same_dom), 2 addr recompose,
    // μ non-increasing, sel_same def, same-domain⇒equal, domain-increase,
    // 2 next-addr copies, IS_BIT(sel_same) = 12.
    assert_eq!(FextPageConstraints.meta().len(), 12);
    // init receiver + fini sender + addr LT + 4 IsHalfword = 7.
    assert_eq!(bus_interactions().len(), 7);
}

#[test]
fn fext_page_trace_layout_and_padding() {
    let ops = vec![
        FextPageOperation {
            domain: 3,
            addr: 0x10,
            final_ts: 100,
            final_val: 42,
        },
        FextPageOperation {
            domain: 5,
            addr: 0x20,
            final_ts: 200,
            final_val: 7,
        },
    ];
    let trace = generate_fext_page_trace(&ops);
    assert_eq!(trace.num_rows(), 4); // 2 ops padded to 4

    let t = &trace.main_table;
    assert_eq!(*t.get(0, cols::DOMAIN), FE::from(3u64));
    assert_eq!(*t.get(0, cols::FINAL_VAL), FE::from(42u64));
    assert_eq!(*t.get(0, cols::FINAL_TS_0), FE::from(100u64));
    assert_eq!(*t.get(0, cols::MU), FE::one());
    assert_eq!(*t.get(1, cols::DOMAIN), FE::from(5u64));

    // Padding rows have μ = 0 and a valid domain (3) so the domain constraint holds.
    for row in 2..4 {
        assert_eq!(*t.get(row, cols::MU), FE::zero());
        assert_eq!(*t.get(row, cols::DOMAIN), FE::from(3u64));
    }
}

#[test]
fn fext_page_sorts_by_domain_then_addr() {
    // Deliberately unsorted; trace-gen must emit strictly ascending (domain, addr).
    let ops = vec![op(5, 0x30), op(3, 0x40), op(4, 0x10), op(3, 0x20)];
    let trace = generate_fext_page_trace(&ops);
    let t = &trace.main_table;

    let key = |row: usize| {
        (
            *t.get(row, cols::DOMAIN),
            *t.get(row, cols::ADDR_0),
            *t.get(row, cols::ADDR_1),
        )
    };
    // Expected order: (3,0x20), (3,0x40), (4,0x10), (5,0x30).
    assert_eq!(key(0), (FE::from(3u64), FE::from(0x20u64), FE::from(0u64)));
    assert_eq!(key(1), (FE::from(3u64), FE::from(0x40u64), FE::from(0u64)));
    assert_eq!(key(2), (FE::from(4u64), FE::from(0x10u64), FE::from(0u64)));
    assert_eq!(key(3), (FE::from(5u64), FE::from(0x30u64), FE::from(0u64)));
}

#[test]
fn fext_page_same_dom_and_selector_columns() {
    // Two domain-3 rows then one domain-4 row (all active), padded to 4.
    let ops = vec![op(3, 0x10), op(3, 0x20), op(4, 0x10)];
    let trace = generate_fext_page_trace(&ops);
    let t = &trace.main_table;

    // Row 0 → row 1: same domain (3,3), both active ⇒ same_dom=1, sel_same=1.
    assert_eq!(*t.get(0, cols::SAME_DOM), FE::one());
    assert_eq!(*t.get(0, cols::SEL_SAME), FE::one());
    // next_addr on row 0 is row 1's addr (0x20).
    assert_eq!(*t.get(0, cols::NEXT_ADDR_0), FE::from(0x20u64));

    // Row 1 → row 2: domains differ (3 vs 4) ⇒ same_dom=0, sel_same=0.
    assert_eq!(*t.get(1, cols::SAME_DOM), FE::zero());
    assert_eq!(*t.get(1, cols::SEL_SAME), FE::zero());

    // Row 2 → row 3: next row is padding (μ=0) ⇒ sel_same=0.
    assert_eq!(*t.get(2, cols::SEL_SAME), FE::zero());
}

#[test]
fn fext_page_addr_limb_halfword_decomposition() {
    // A 64-bit addr with bits set across both limbs and both halves.
    let addr = (0xABCDu64 << 48) | (0x1234u64 << 32) | (0x5678u64 << 16) | 0x9ABC;
    let trace = generate_fext_page_trace(&[op(3, addr)]);
    let t = &trace.main_table;
    assert_eq!(*t.get(0, cols::ADDR0_HW_LO), FE::from(0x9ABCu64));
    assert_eq!(*t.get(0, cols::ADDR0_HW_HI), FE::from(0x5678u64));
    assert_eq!(*t.get(0, cols::ADDR1_HW_LO), FE::from(0x1234u64));
    assert_eq!(*t.get(0, cols::ADDR1_HW_HI), FE::from(0xABCDu64));
}

#[test]
fn fext_page_constraints_hold_on_valid_trace() {
    // Two domain-3 cells and one domain-4 cell: exercises same-domain (addr
    // increase) and domain-change transitions plus padding.
    let ops = vec![op(3, 0x10), op(3, 0x20), op(4, 0x08)];
    let trace = generate_fext_page_trace(&ops);
    for row in 0..trace.num_rows() - 1 {
        for (idx, v) in eval_transition(&trace, row).into_iter().enumerate() {
            assert_eq!(v, FE::zero(), "row {row}, constraint {idx} should be zero");
        }
    }
}

#[test]
fn fext_page_rejects_forged_domain() {
    // Domain outside {3,4,5} must fail the domain constraint (idx 1).
    let mut trace = generate_fext_page_trace(&[op(3, 0x10)]);
    trace.main_table.set_fe(0, cols::DOMAIN, FE::from(0u64)); // domain 0 = RAM
    assert_ne!(eval_transition(&trace, 0)[1], FE::zero());
}

#[test]
fn fext_page_rejects_domain_decrease() {
    // Sorted output is (3,_),(4,_); forcing the first row's domain to 5 makes the
    // domain decrease across the transition, which idx 8 must reject.
    let mut trace = generate_fext_page_trace(&[op(3, 0x10), op(4, 0x20)]);
    trace.main_table.set_fe(0, cols::DOMAIN, FE::from(5u64));
    assert_ne!(eval_transition(&trace, 0)[8], FE::zero());
}

#[test]
fn fext_page_rejects_active_row_after_padding() {
    // An active row following a padding row breaks μ non-increasing (idx 5).
    let mut trace = generate_fext_page_trace(&[op(3, 0x10)]);
    trace.main_table.set_fe(2, cols::MU, FE::one()); // row 1 padding, row 2 "active"
    assert_ne!(eval_transition(&trace, 1)[5], FE::zero());
}

#[test]
fn fext_page_rejects_mismatched_same_dom() {
    // same_dom claims "different" on two equal-domain rows: the selector
    // definition (idx 6) or the domain-increase check (idx 8) must reject it.
    let mut trace = generate_fext_page_trace(&[op(3, 0x10), op(3, 0x20)]);
    trace.main_table.set_fe(0, cols::SAME_DOM, FE::zero());
    trace.main_table.set_fe(0, cols::SEL_SAME, FE::zero());
    let base = eval_transition(&trace, 0);
    assert!(base[6] != FE::zero() || base[8] != FE::zero());
}

#[test]
fn fext_page_rejects_forged_next_addr() {
    // The next-addr copy (idx 9) pins the cross-row LT operand to the real next
    // row, so tampering with it is caught.
    let mut trace = generate_fext_page_trace(&[op(3, 0x10), op(3, 0x20)]);
    trace
        .main_table
        .set_fe(0, cols::NEXT_ADDR_0, FE::from(0x999u64));
    assert_ne!(eval_transition(&trace, 0)[9], FE::zero());
}

#[test]
fn fext_page_rejects_free_last_row_sel_same() {
    // CRIT-001 regression: the `sel_same` definition (idx 6) is `except_last`-gated,
    // so it never pins the final row's `sel_same`. Since `sel_same` is the addr-LT
    // sender's multiplicity, a free last-row value of −1 would cancel a forced `+1`
    // LT claim of the same tuple and erase the strict-increase check (enabling a
    // duplicate cell). The ungated `IS_BIT(sel_same)` (idx 11) must reject any
    // non-{0,1} last-row multiplicity — here the −1 wildcard.
    let mut trace = generate_fext_page_trace(&[op(3, 0x10), op(3, 0x20)]);
    let last = trace.num_rows() - 1;
    trace
        .main_table
        .set_fe(last, cols::SEL_SAME, FE::zero() - FE::one());
    assert_ne!(eval_transition(&trace, last)[11], FE::zero());
}
