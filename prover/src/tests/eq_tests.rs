//! Tests for the EQ (equality) table.

use crate::tables::eq::{EqOperation, bus_interactions, cols, generate_eq_trace};
use crate::tables::types::{BusId, FE};

#[test]
fn test_compute_eq_and_res() {
    assert!(EqOperation::new(5, 5, false).compute_eq());
    assert!(!EqOperation::new(5, 3, false).compute_eq());

    // res = eq XOR invert
    assert!(EqOperation::new(5, 5, false).compute_res()); // 1 XOR 0
    assert!(!EqOperation::new(5, 5, true).compute_res()); // 1 XOR 1
    assert!(!EqOperation::new(5, 3, false).compute_res()); // 0 XOR 0
    assert!(EqOperation::new(5, 3, true).compute_res()); // 0 XOR 1
}

#[test]
fn test_trace_equal_operands() {
    // a == b → diff = 0, eq = 1, res = 1 (invert = 0)
    let trace = generate_eq_trace(&[EqOperation::new(5, 5, false)]);
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);
    assert_eq!(trace.main_table.height, 4); // padded to min 4

    let row = trace.main_table.get_row(0);
    assert_eq!(row[cols::A_0], FE::from(5u64));
    assert_eq!(row[cols::B_0], FE::from(5u64));
    assert_eq!(row[cols::DIFF_0], FE::from(0u64));
    assert_eq!(row[cols::DIFF_1], FE::from(0u64));
    assert_eq!(row[cols::DIFF_2], FE::from(0u64));
    assert_eq!(row[cols::DIFF_3], FE::from(0u64));
    assert_eq!(row[cols::EQ], FE::from(1u64));
    assert_eq!(row[cols::RES], FE::from(1u64));
    assert_eq!(row[cols::INVERT], FE::from(0u64));
    assert_eq!(row[cols::MU], FE::from(1u64));
}

#[test]
fn test_trace_unequal_operands() {
    // a = 5, b = 3 → diff = 2, eq = 0, res = 0
    let trace = generate_eq_trace(&[EqOperation::new(5, 3, false)]);
    let row = trace.main_table.get_row(0);
    assert_eq!(row[cols::DIFF_0], FE::from(2u64));
    assert_eq!(row[cols::EQ], FE::from(0u64));
    assert_eq!(row[cols::RES], FE::from(0u64));
}

#[test]
fn test_trace_invert_and_wrapping() {
    // a == b with invert = 1 → eq = 1, res = 0
    let trace = generate_eq_trace(&[EqOperation::new(7, 7, true)]);
    let row = trace.main_table.get_row(0);
    assert_eq!(row[cols::EQ], FE::from(1u64));
    assert_eq!(row[cols::INVERT], FE::from(1u64));
    assert_eq!(row[cols::RES], FE::from(0u64));

    // a = 0, b = 1 → diff = 0 - 1 = 0xFFFF_FFFF_FFFF_FFFF (all halves 0xFFFF), eq = 0
    let trace = generate_eq_trace(&[EqOperation::new(0, 1, false)]);
    let row = trace.main_table.get_row(0);
    assert_eq!(row[cols::DIFF_0], FE::from(0xFFFFu64));
    assert_eq!(row[cols::DIFF_3], FE::from(0xFFFFu64));
    assert_eq!(row[cols::EQ], FE::from(0u64));
}

#[test]
fn test_trace_dword_split() {
    // a spanning both words: 0x1234_5678_9ABC_DEF0
    let a = 0x1234_5678_9ABC_DEF0u64;
    let trace = generate_eq_trace(&[EqOperation::new(a, 0, false)]);
    let row = trace.main_table.get_row(0);
    assert_eq!(row[cols::A_0], FE::from(0x9ABC_DEF0u64));
    assert_eq!(row[cols::A_1], FE::from(0x1234_5678u64));
}

#[test]
fn test_multiplicity_aggregation() {
    // Same op three times + one distinct → 2 unique rows, padded to 4.
    let ops = vec![
        EqOperation::new(5, 5, false),
        EqOperation::new(9, 8, false),
        EqOperation::new(5, 5, false),
        EqOperation::new(5, 5, false),
    ];
    let trace = generate_eq_trace(&ops);
    assert_eq!(trace.main_table.height, 4);

    let mut found = false;
    for row_idx in 0..4 {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::A_0] == FE::from(5u64) && row[cols::B_0] == FE::from(5u64) {
            assert_eq!(row[cols::MU], FE::from(3u64));
            found = true;
        }
    }
    assert!(found, "expected the (5,5) row with multiplicity 3");
}

#[test]
fn test_bus_interactions_shape() {
    let interactions = bus_interactions();
    // 4 IS_HALF senders + 1 ZERO sender + 1 ALU receiver
    assert_eq!(interactions.len(), 6);

    let is_half = interactions
        .iter()
        .filter(|i| i.bus_id == u64::from(BusId::IsHalfword) && i.is_sender)
        .count();
    assert_eq!(is_half, 4);

    let zero = interactions
        .iter()
        .filter(|i| i.bus_id == u64::from(BusId::Zero) && i.is_sender)
        .count();
    assert_eq!(zero, 1);

    // Exactly one ALU receiver carrying [a, b, flags, res].
    let alu: Vec<_> = interactions
        .iter()
        .filter(|i| i.bus_id == u64::from(BusId::Alu))
        .collect();
    assert_eq!(alu.len(), 1);
    assert!(!alu[0].is_sender, "ALU is a receiver for EQ");
    // [a, b, flags, res, 0] — the ALU output is DWordWL ([res, 0]).
    assert_eq!(alu[0].values.len(), 5);
}
