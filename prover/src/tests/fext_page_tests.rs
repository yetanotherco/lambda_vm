//! Tests for the FEXT_PAGE bookend table.

use crate::tables::fext_page::{
    FextPageConstraints, FextPageOperation, bus_interactions, cols, generate_fext_page_trace,
};
use crate::tables::types::FE;
use stark::constraints::builder::ConstraintSet;

#[test]
fn fext_page_constraint_and_bus_counts() {
    assert_eq!(FextPageConstraints.meta().len(), 1); // IS_BIT(μ)
    assert_eq!(bus_interactions().len(), 2); // init receiver + fini sender
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

    // Padding rows have μ = 0.
    for row in 2..4 {
        assert_eq!(*t.get(row, cols::MU), FE::zero());
    }
}
