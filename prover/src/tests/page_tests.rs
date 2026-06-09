//! Tests for the PAGE table.

use crate::tables::page::*;
use crate::tables::types::*;

#[test]
fn test_page_base_for_address() {
    // DEFAULT_PAGE_SIZE = 1 << 18 = 0x40000
    assert_eq!(page_base_for_address(0x00000), 0x00000);
    assert_eq!(page_base_for_address(0x1000), 0x00000); // 0x1000 < 0x40000
    assert_eq!(page_base_for_address(0x3FFFF), 0x00000); // last byte of first page
    assert_eq!(page_base_for_address(0x40000), 0x40000); // start of second page
    assert_eq!(page_base_for_address(0x40001), 0x40000); // one byte into second page
}

#[test]
fn test_offset_in_page() {
    // DEFAULT_PAGE_SIZE = 0x40000
    assert_eq!(offset_in_page(0x00000), 0);
    assert_eq!(offset_in_page(0x1000), 0x1000); // 4096
    assert_eq!(offset_in_page(0x3FFFF), 0x3FFFF); // last offset in first page
    assert_eq!(offset_in_page(0x40000), 0); // start of second page → offset 0
    assert_eq!(offset_in_page(0x40001), 1); // one byte into second page
}

#[test]
fn test_generate_page_trace_zero_init() {
    // Use 0 as page_base (aligned to DEFAULT_PAGE_SIZE = 256KB)
    let config = PageConfig::zero_init(0);
    let final_state = FinalStateMap::new();

    let trace = generate_page_trace(&config, &final_state);

    // Trace must have DEFAULT_PAGE_SIZE rows
    assert_eq!(trace.num_rows(), DEFAULT_PAGE_SIZE);

    // Sample: first row
    assert_eq!(*trace.main_table.get(0, cols::OFFSET), FE::zero());
    assert_eq!(*trace.main_table.get(0, cols::INIT), FE::zero());
    assert_eq!(*trace.main_table.get(0, cols::FINI), FE::zero());
    assert_eq!(*trace.main_table.get(0, cols::TIMESTAMP_LO), FE::zero());

    // Sample: some middle row (offset 42)
    assert_eq!(*trace.main_table.get(42, cols::OFFSET), FE::from(42u64));
    assert_eq!(*trace.main_table.get(42, cols::INIT), FE::zero());
    assert_eq!(*trace.main_table.get(42, cols::FINI), FE::zero());

    // Sample: last row
    let last = DEFAULT_PAGE_SIZE - 1;
    assert_eq!(
        *trace.main_table.get(last, cols::OFFSET),
        FE::from(last as u64)
    );
    assert_eq!(*trace.main_table.get(last, cols::INIT), FE::zero());
    assert_eq!(*trace.main_table.get(last, cols::FINI), FE::zero());
}

#[test]
fn test_generate_page_trace_with_data() {
    // Use 0 as page_base (aligned to DEFAULT_PAGE_SIZE = 256KB)
    let data = vec![0x01, 0x02, 0x03, 0x04];
    let config = PageConfig::with_data(0, data);
    let final_state = FinalStateMap::new();

    let trace = generate_page_trace(&config, &final_state);

    // Trace must have DEFAULT_PAGE_SIZE rows
    assert_eq!(trace.num_rows(), DEFAULT_PAGE_SIZE);

    // Check initial values from data (first 4 bytes)
    assert_eq!(*trace.main_table.get(0, cols::INIT), FE::from(0x01u64));
    assert_eq!(*trace.main_table.get(1, cols::INIT), FE::from(0x02u64));
    assert_eq!(*trace.main_table.get(2, cols::INIT), FE::from(0x03u64));
    assert_eq!(*trace.main_table.get(3, cols::INIT), FE::from(0x04u64));
    // Bytes past the supplied data should be zero (trailing zeros)
    assert_eq!(*trace.main_table.get(4, cols::INIT), FE::zero());
    assert_eq!(*trace.main_table.get(100, cols::INIT), FE::zero());

    // Without accesses, fini should equal init
    assert_eq!(*trace.main_table.get(0, cols::FINI), FE::from(0x01u64));
    assert_eq!(*trace.main_table.get(3, cols::FINI), FE::from(0x04u64));
    assert_eq!(*trace.main_table.get(4, cols::FINI), FE::zero());

    // OFFSET column must equal row index
    assert_eq!(*trace.main_table.get(0, cols::OFFSET), FE::from(0u64));
    assert_eq!(*trace.main_table.get(3, cols::OFFSET), FE::from(3u64));
}

#[test]
fn test_generate_page_trace_with_accesses() {
    // Use 0 as page_base (aligned to DEFAULT_PAGE_SIZE = 256KB)
    let data = vec![0xAA, 0xBB];
    let config = PageConfig::with_data(0, data);

    let mut final_state = FinalStateMap::new();
    // Address 0 (= page_base + offset 0) was written with value 0xFF at timestamp 100
    final_state.insert(
        0,
        FinalByteState {
            timestamp: 100,
            value: 0xFF,
        },
    );

    let trace = generate_page_trace(&config, &final_state);

    // Row 0: address 0 (offset 0) - was accessed
    assert_eq!(*trace.main_table.get(0, cols::INIT), FE::from(0xAAu64));
    assert_eq!(*trace.main_table.get(0, cols::FINI), FE::from(0xFFu64));
    assert_eq!(
        *trace.main_table.get(0, cols::TIMESTAMP_LO),
        FE::from(100u64)
    );

    // Row 1: address 1 (offset 1) - not accessed, fini = init = 0xBB
    assert_eq!(*trace.main_table.get(1, cols::INIT), FE::from(0xBBu64));
    assert_eq!(*trace.main_table.get(1, cols::FINI), FE::from(0xBBu64));
    assert_eq!(*trace.main_table.get(1, cols::TIMESTAMP_LO), FE::zero());

    // Row 2: not in data, not accessed — init=0, fini=0
    assert_eq!(*trace.main_table.get(2, cols::INIT), FE::zero());
    assert_eq!(*trace.main_table.get(2, cols::FINI), FE::zero());
    assert_eq!(*trace.main_table.get(2, cols::TIMESTAMP_LO), FE::zero());

    // OFFSET column must equal row index
    assert_eq!(*trace.main_table.get(0, cols::OFFSET), FE::from(0u64));
    assert_eq!(*trace.main_table.get(1, cols::OFFSET), FE::from(1u64));
}

#[test]
fn test_bus_interactions() {
    let interactions = bus_interactions(0); // page_base = 0
    assert_eq!(interactions.len(), 3); // C1+C2 (batched ARE_BYTES), C3, C4
}

#[test]
fn test_bus_interactions_high_address() {
    // Test with high address like stack region
    let stack_page = STACK_TOP & !(DEFAULT_PAGE_SIZE as u64 - 1);
    let interactions = bus_interactions(stack_page);
    assert_eq!(interactions.len(), 3);
}
