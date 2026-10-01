//! Tests for the STORE table.

use crate::tables::store::{StoreOperation, bus_interactions, cols, generate_store_trace};
use crate::tables::types::{BusId, FE};
use stark::lookup::{BusValue, LinearTerm};

#[test]
fn test_new_sets_width_flags() {
    let sb = StoreOperation::new(0, 0, 0, 1);
    assert!(!sb.write2 && !sb.write4 && !sb.write8); // 1 byte: none set
    assert!(StoreOperation::new(0, 0, 0, 2).write2);
    assert!(StoreOperation::new(0, 0, 0, 4).write4);
    assert!(StoreOperation::new(0, 0, 0, 8).write8);
}

#[test]
fn test_trace_layout() {
    let op = StoreOperation::new(0xDEAD_BEEF_0000_1000, 0x40, 0x1122_3344_5566_7788, 8);
    let trace = generate_store_trace(&[op]);
    assert_eq!(trace.main_table.width, cols::NUM_COLUMNS);
    assert_eq!(trace.main_table.height, 4); // padded to min 4

    let row = trace.main_table.get_row(0);
    assert_eq!(row[cols::BASE_ADDRESS_0], FE::from(0x0000_1000u64));
    assert_eq!(row[cols::BASE_ADDRESS_1], FE::from(0xDEAD_BEEFu64));
    assert_eq!(row[cols::TIMESTAMP_0], FE::from(0x40u64));
    assert_eq!(row[cols::WRITE8], FE::from(1u64));
    assert_eq!(row[cols::WRITE2], FE::from(0u64));
    // value little-endian byte split
    assert_eq!(row[cols::VALUE[0]], FE::from(0x88u64));
    assert_eq!(row[cols::VALUE[7]], FE::from(0x11u64));
    assert_eq!(row[cols::MU], FE::from(1u64));
}

#[test]
fn test_bus_interactions_shape() {
    let interactions = bus_interactions();
    // 1 MEMW write + 1 MEMORY receiver + 8 ARE_BYTES.
    assert_eq!(interactions.len(), 10);

    let memw = interactions
        .iter()
        .filter(|i| i.bus_id == u64::from(BusId::Memw) && i.is_sender)
        .count();
    assert_eq!(memw, 1);

    let are_bytes = interactions
        .iter()
        .filter(|i| i.bus_id == u64::from(BusId::AreBytes) && i.is_sender)
        .count();
    assert_eq!(are_bytes, 8);

    let memory: Vec<_> = interactions
        .iter()
        .filter(|i| i.bus_id == u64::from(BusId::MemoryOp))
        .collect();
    assert_eq!(memory.len(), 1);
    assert!(!memory[0].is_sender, "STORE receives MEMORY");
    // [timestamp, base_address, value, flags, out_lo, out_hi]
    assert_eq!(memory[0].values.len(), 6);
}

#[test]
fn test_memory_flags_include_memory_op_bit() {
    // Q7 fix: the MEMORY flags must carry the memory_op bit (constant 1).
    let interactions = bus_interactions();
    let memory = interactions
        .iter()
        .find(|i| i.bus_id == u64::from(BusId::MemoryOp))
        .expect("MEMORY receiver exists");

    // flags is the 4th value (index 3).
    match &memory.values[3] {
        BusValue::Linear(terms) => {
            let has_memory_op = terms.iter().any(|t| matches!(t, LinearTerm::Constant(1)));
            assert!(
                has_memory_op,
                "MEMORY flags must include the memory_op constant 1 (Q7 fix)"
            );
        }
        _ => panic!("expected a linear flags term"),
    }
}
