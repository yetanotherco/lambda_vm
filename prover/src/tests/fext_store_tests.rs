//! Tests for the FEXT_STORE table: trace layout, padding, constraint/bus counts,
//! and the IS_BIT(μ) check.

use crate::tables::fext_store::{
    FextStoreConstraints, FextStoreOperation, bus_interactions, cols, generate_fext_store_trace,
};
use crate::tables::types::FE;
use stark::constraints::builder::ConstraintSet;

fn op(coeffs: [u64; 3]) -> FextStoreOperation {
    FextStoreOperation {
        timestamp: 100,
        src_addr: 0x40,
        coeffs,
        old_ts: [10, 20, 30],
    }
}

#[test]
fn fext_store_constraint_count() {
    // IS_BIT(μ) + 6 word-recompose constraints (one per coefficient word).
    assert_eq!(FextStoreConstraints.meta().len(), 7);
}

#[test]
fn fext_store_bus_interaction_count() {
    // 1 Ecall + 1 register read (x10) + 3 field reads (consume + emit + old_ts<ts
    // = 3 each) + 3 register writes + 12 IsHalfword + 3 coeff<p
    // = 1 + 1 + 9 + 3 + 12 + 3.
    assert_eq!(bus_interactions().len(), 29);
}

#[test]
fn fext_store_trace_layout_and_padding() {
    let ops = vec![op([11, 22, 33]), op([44, 55, 66])];
    let trace = generate_fext_store_trace(&ops);
    assert_eq!(trace.num_rows(), 4); // 2 ops padded to 4

    let t = &trace.main_table;
    assert_eq!(*t.get(0, cols::TIMESTAMP_0), FE::from(100u64));
    assert_eq!(*t.get(0, cols::SRC_ADDR_0), FE::from(0x40u64));
    assert_eq!(*t.get(0, cols::C0_LO), FE::from(11u64));
    assert_eq!(*t.get(0, cols::C1_LO), FE::from(22u64));
    assert_eq!(*t.get(0, cols::C2_LO), FE::from(33u64));
    assert_eq!(*t.get(0, cols::OLD_TS0_0), FE::from(10u64));
    assert_eq!(*t.get(0, cols::MU), FE::one());
    assert_eq!(*t.get(1, cols::C0_LO), FE::from(44u64));

    // Padding rows have μ = 0.
    for row in 2..4 {
        assert_eq!(*t.get(row, cols::MU), FE::zero());
    }
}

#[test]
fn fext_store_splits_high_word() {
    // A coefficient above 2^32 must land in both the lo and hi words.
    let val = (7u64 << 32) | 5;
    let trace = generate_fext_store_trace(&[op([val, 0, 0])]);
    let t = &trace.main_table;
    assert_eq!(*t.get(0, cols::C0_LO), FE::from(5u64));
    assert_eq!(*t.get(0, cols::C0_HI), FE::from(7u64));
}

#[test]
fn fext_store_trace_shape() {
    let trace = generate_fext_store_trace(&[op([1, 2, 3])]);
    // 1 op → padded to min 4 rows.
    assert_eq!(trace.num_rows(), 4);
    assert_eq!(*trace.main_table.get(0, cols::MU), FE::one());
    for row in 1..4 {
        assert_eq!(*trace.main_table.get(row, cols::MU), FE::zero());
    }
}
