//! Tests for the REGISTER table.

use crate::tables::register::*;
use crate::tables::types::*;

#[test]
fn test_register_base_address() {
    assert_eq!(register_base_address(0), 0);
    assert_eq!(register_base_address(1), 2);
    assert_eq!(register_base_address(2), 4);
    assert_eq!(register_base_address(31), 62);
    assert_eq!(register_base_address(254), 508);
    assert_eq!(register_base_address(255), 510);
}

#[test]
fn test_generate_register_trace_empty() {
    let entry_point = 0x1000u64;
    let final_state = FinalRegisterStateMap::new();
    let trace = generate_register_trace(&final_state, &register_init_from_entry_point(entry_point));

    // Should have power-of-2 rows >= 67 (x0-x31, x254, x255)
    assert!(trace.num_rows() >= NUM_REGISTER_ADDRESSES);
    assert!(trace.num_rows().is_power_of_two());

    // Check first row (address 0, never accessed): timestamp defaults to 1
    // per spec/memory.typ so that REG-C1/REG-C2 cancel on the bus.
    assert_eq!(*trace.main_table.get(0, cols::OFFSET), FE::zero());
    assert_eq!(*trace.main_table.get(0, cols::INIT), FE::zero());
    assert_eq!(*trace.main_table.get(0, cols::FINI), FE::zero());
    assert_eq!(*trace.main_table.get(0, cols::TIMESTAMP_LO), FE::from(1u64));

    // Check x254 row (row 64 = addr 508)
    assert_eq!(*trace.main_table.get(64, cols::OFFSET), FE::from(508u64));
    assert_eq!(*trace.main_table.get(64, cols::INIT), FE::zero());
    assert_eq!(*trace.main_table.get(64, cols::FINI), FE::zero());

    // Check x255 rows (row 65 = addr 510, row 66 = addr 511)
    assert_eq!(*trace.main_table.get(65, cols::OFFSET), FE::from(510u64));
    assert_eq!(
        *trace.main_table.get(65, cols::INIT),
        FE::from(entry_point & 0xFFFF_FFFF)
    );
    assert_eq!(
        *trace.main_table.get(65, cols::FINI),
        FE::from(entry_point & 0xFFFF_FFFF)
    ); // fini=init when never accessed
    assert_eq!(*trace.main_table.get(66, cols::OFFSET), FE::from(511u64));
    assert_eq!(
        *trace.main_table.get(66, cols::INIT),
        FE::from(entry_point >> 32)
    );
}

#[test]
fn test_generate_register_trace_with_access() {
    let entry_point = 0x1000u64;
    let mut final_state = FinalRegisterStateMap::new();
    // Register x5 low Word was written with value 0x42 at timestamp 100
    let addr = register_base_address(5); // = 10
    final_state.insert(
        addr,
        FinalRegisterWordState {
            timestamp: 100,
            value: 0x42,
        },
    );

    let trace = generate_register_trace(&final_state, &register_init_from_entry_point(entry_point));

    // Row 10 (address 10) should have the final state
    assert_eq!(*trace.main_table.get(10, cols::OFFSET), FE::from(10u64));
    assert_eq!(*trace.main_table.get(10, cols::INIT), FE::zero()); // init is always 0
    assert_eq!(*trace.main_table.get(10, cols::FINI), FE::from(0x42u64));
    assert_eq!(
        *trace.main_table.get(10, cols::TIMESTAMP_LO),
        FE::from(100u64)
    );
}

#[test]
fn test_bus_interactions() {
    let interactions = bus_interactions();
    assert_eq!(interactions.len(), 2); // C1, C2
}
