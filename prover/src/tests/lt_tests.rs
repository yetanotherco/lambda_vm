//! Tests for the LT (Less-Than) table.

use crate::tables64::lt::{bus_interactions, cols, generate_lt_trace, LtOperation};
use crate::tables64::types::FE;

#[test]
fn test_lt_unsigned_basic() {
    let ops = vec![
        LtOperation::new(5, 10, false),       // 5 < 10 unsigned -> true
        LtOperation::new(10, 5, false),       // 10 < 5 unsigned -> false
        LtOperation::new(5, 5, false),        // 5 < 5 unsigned -> false
        LtOperation::new(0, 1, false),        // 0 < 1 unsigned -> true
        LtOperation::new(u64::MAX, 0, false), // MAX < 0 unsigned -> false
    ];

    assert!(ops[0].compute_lt());
    assert!(!ops[1].compute_lt());
    assert!(!ops[2].compute_lt());
    assert!(ops[3].compute_lt());
    assert!(!ops[4].compute_lt());
}

#[test]
fn test_lt_signed_basic() {
    let ops = vec![
        LtOperation::new(5, 10, true),             // 5 < 10 signed -> true
        LtOperation::new(10, 5, true),             // 10 < 5 signed -> false
        LtOperation::new((-5i64) as u64, 5, true), // -5 < 5 signed -> true
        LtOperation::new(5, (-5i64) as u64, true), // 5 < -5 signed -> false
        LtOperation::new((-10i64) as u64, (-5i64) as u64, true), // -10 < -5 signed -> true
    ];

    assert!(ops[0].compute_lt());
    assert!(!ops[1].compute_lt());
    assert!(ops[2].compute_lt());
    assert!(!ops[3].compute_lt());
    assert!(ops[4].compute_lt());
}

#[test]
fn test_trace_generation() {
    let ops = vec![
        LtOperation::new(100, 200, false),
        LtOperation::new(200, 100, true),
    ];

    let trace = generate_lt_trace(&ops);

    // Should be padded to power of 2
    assert_eq!(trace.main_table.height, 2);
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);

    // Check first row
    let row0 = trace.main_table.get_row(0);
    assert_eq!(row0[cols::LHS_0], FE::from(100u64));
    assert_eq!(row0[cols::RHS_0], FE::from(200u64));
    assert_eq!(row0[cols::SIGNED], FE::zero());
    assert_eq!(row0[cols::LT], FE::one()); // 100 < 200

    // Check second row
    let row1 = trace.main_table.get_row(1);
    assert_eq!(row1[cols::LHS_0], FE::from(200u64));
    assert_eq!(row1[cols::RHS_0], FE::from(100u64));
    assert_eq!(row1[cols::SIGNED], FE::one());
    assert_eq!(row1[cols::LT], FE::zero()); // 200 < 100 signed -> false
}

#[test]
fn test_bus_interactions_count() {
    let interactions = bus_interactions();
    // MSB16 x2 + IS_HALFWORD x4 + LT x1 = 7 interactions
    assert_eq!(interactions.len(), 7);
}
