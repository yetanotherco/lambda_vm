//! Tests for the BYTEWISE ALU table.

use crate::tables::bytewise::{BytewiseOperation, bus_interactions, cols, generate_bytewise_trace};
use crate::tables::types::{BusId, FE, alu_op};

#[test]
fn test_compute_res() {
    let a = 0xFF00u64;
    let b = 0x0FF0u64;
    assert_eq!(
        BytewiseOperation::new(a, b, alu_op::AND).compute_res(),
        0x0F00
    );
    assert_eq!(
        BytewiseOperation::new(a, b, alu_op::OR).compute_res(),
        0xFFF0
    );
    assert_eq!(
        BytewiseOperation::new(a, b, alu_op::XOR).compute_res(),
        0xF0F0
    );
}

#[test]
fn test_trace_byte_decomposition() {
    // a XOR b across all 8 bytes.
    let a = 0x1122_3344_5566_7788u64;
    let b = 0x00FF_00FF_00FF_00FFu64;
    let trace = generate_bytewise_trace(&[BytewiseOperation::new(a, b, alu_op::XOR)]);
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);
    assert_eq!(trace.main_table.height, 4); // padded to min 4

    let row = trace.main_table.get_row(0);
    // Little-endian: byte 0 is the least significant.
    assert_eq!(row[cols::A[0]], FE::from(0x88u64));
    assert_eq!(row[cols::A[7]], FE::from(0x11u64));
    assert_eq!(row[cols::B[0]], FE::from(0xFFu64));
    assert_eq!(row[cols::OP], FE::from(alu_op::XOR as u64));
    // res byte 0 = 0x88 ^ 0xFF = 0x77
    assert_eq!(row[cols::RES[0]], FE::from(0x77u64));
    // res byte 7 = 0x11 ^ 0x00 = 0x11
    assert_eq!(row[cols::RES[7]], FE::from(0x11u64));
    assert_eq!(row[cols::MU], FE::from(1u64));
}

#[test]
fn test_multiplicity_aggregation() {
    let ops = vec![
        BytewiseOperation::new(1, 2, alu_op::AND),
        BytewiseOperation::new(3, 4, alu_op::OR),
        BytewiseOperation::new(1, 2, alu_op::AND),
    ];
    let trace = generate_bytewise_trace(&ops);
    assert_eq!(trace.main_table.height, 4);

    let mut found = false;
    for row_idx in 0..4 {
        let row = trace.main_table.get_row(row_idx);
        if row[cols::A[0]] == FE::from(1u64)
            && row[cols::B[0]] == FE::from(2u64)
            && row[cols::OP] == FE::from(alu_op::AND as u64)
        {
            assert_eq!(row[cols::MU], FE::from(2u64));
            found = true;
        }
    }
    assert!(found, "expected the (1, 2, AND) row with multiplicity 2");
}

#[test]
fn test_bus_interactions_shape() {
    let interactions = bus_interactions();
    // 8 BYTE_ALU senders + 1 ALU receiver.
    assert_eq!(interactions.len(), 9);

    let byte_alu_senders = interactions
        .iter()
        .filter(|i| i.bus_id == u64::from(BusId::ByteAlu) && i.is_sender)
        .count();
    assert_eq!(byte_alu_senders, 8);

    let alu: Vec<_> = interactions
        .iter()
        .filter(|i| i.bus_id == u64::from(BusId::Alu))
        .collect();
    assert_eq!(alu.len(), 1);
    assert!(!alu[0].is_sender, "ALU is a receiver for BYTEWISE");
    assert_eq!(alu[0].values.len(), 4); // [a, b, op, res]
}
