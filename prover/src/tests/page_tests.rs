//! Tests for the PAGE table.

use crate::tables::page::*;
use crate::tables::types::*;

#[test]
fn test_page_base_for_address() {
    let page_size = 4096;
    assert_eq!(page_base_for_address(0x1000, page_size), 0x1000);
    assert_eq!(page_base_for_address(0x1001, page_size), 0x1000);
    assert_eq!(page_base_for_address(0x1FFF, page_size), 0x1000);
    assert_eq!(page_base_for_address(0x2000, page_size), 0x2000);
}

#[test]
fn test_offset_in_page() {
    let page_size = 4096;
    assert_eq!(offset_in_page(0x1000, page_size), 0);
    assert_eq!(offset_in_page(0x1001, page_size), 1);
    assert_eq!(offset_in_page(0x1FFF, page_size), 4095);
    assert_eq!(offset_in_page(0x2000, page_size), 0);
}

#[test]
fn test_generate_page_trace_zero_init() {
    let config = PageConfig::zero_init(0x1000, 16); // Small page for testing
    let final_state = FinalStateMap::new();

    let trace = generate_page_trace(&config, &final_state);

    assert_eq!(trace.num_rows(), 16);

    // Check first row (address is virtual: 0x1000 + offset)
    assert_eq!(*trace.main_table.get(0, cols::OFFSET), FE::zero());
    assert_eq!(*trace.main_table.get(0, cols::INIT), FE::zero());
    assert_eq!(*trace.main_table.get(0, cols::FINI), FE::zero());
    assert_eq!(*trace.main_table.get(0, cols::TIMESTAMP_LO), FE::zero());

    // Check last row (address is virtual: 0x1000 + 15 = 0x100F)
    assert_eq!(*trace.main_table.get(15, cols::OFFSET), FE::from(15u64));
    assert_eq!(*trace.main_table.get(15, cols::INIT), FE::zero());
}

#[test]
fn test_generate_page_trace_with_data() {
    let data = vec![0x01, 0x02, 0x03, 0x04];
    let config = PageConfig::with_data(0x2000, 16, data);
    let final_state = FinalStateMap::new();

    let trace = generate_page_trace(&config, &final_state);

    // Check initial values from data
    assert_eq!(*trace.main_table.get(0, cols::INIT), FE::from(0x01u64));
    assert_eq!(*trace.main_table.get(1, cols::INIT), FE::from(0x02u64));
    assert_eq!(*trace.main_table.get(2, cols::INIT), FE::from(0x03u64));
    assert_eq!(*trace.main_table.get(3, cols::INIT), FE::from(0x04u64));
    // Rest should be zero (padding)
    assert_eq!(*trace.main_table.get(4, cols::INIT), FE::zero());

    // Without accesses, fini should equal init
    assert_eq!(*trace.main_table.get(0, cols::FINI), FE::from(0x01u64));
}

#[test]
fn test_generate_page_trace_with_accesses() {
    let data = vec![0xAA, 0xBB];
    let config = PageConfig::with_data(0x3000, 16, data);

    let mut final_state = FinalStateMap::new();
    // Address 0x3000 was written with value 0xFF at timestamp 100
    final_state.insert(
        0x3000,
        FinalByteState {
            timestamp: 100,
            value: 0xFF,
        },
    );

    let trace = generate_page_trace(&config, &final_state);

    // Row 0: address 0x3000 - was accessed
    assert_eq!(*trace.main_table.get(0, cols::INIT), FE::from(0xAAu64));
    assert_eq!(*trace.main_table.get(0, cols::FINI), FE::from(0xFFu64));
    assert_eq!(
        *trace.main_table.get(0, cols::TIMESTAMP_LO),
        FE::from(100u64)
    );

    // Row 1: address 0x3001 - not accessed, fini = init
    assert_eq!(*trace.main_table.get(1, cols::INIT), FE::from(0xBBu64));
    assert_eq!(*trace.main_table.get(1, cols::FINI), FE::from(0xBBu64));
    assert_eq!(*trace.main_table.get(1, cols::TIMESTAMP_LO), FE::zero());
}

#[test]
fn test_bus_interactions() {
    let interactions = bus_interactions(0x1000); // page_base
    assert_eq!(interactions.len(), 3); // C1+C2 (batched ARE_BYTES), C3, C4
}

#[test]
fn test_bus_interactions_high_address() {
    // Test with high address like stack region
    let stack_page = STACK_TOP & !(DEFAULT_PAGE_SIZE as u64 - 1);
    let interactions = bus_interactions(stack_page);
    assert_eq!(interactions.len(), 3);
}
