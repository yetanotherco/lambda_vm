//! Tests for the MEMW_A (aligned memory word) table.

use crate::tables::memw::MemwOperation;
use crate::tables::memw_aligned::*;
use crate::tables::types::*;

#[test]
fn test_memw_aligned_trace_generation() {
    let ops = vec![
        MemwOperation::new(true, 4, [42, 0, 0, 0, 0, 0, 0, 0], 100, 2, true)
            .with_old([42, 0, 0, 0, 0, 0, 0, 0], [50, 50, 0, 0, 0, 0, 0, 0]),
        MemwOperation::new(false, 0x1000, [1, 2, 3, 4, 0, 0, 0, 0], 200, 4, false)
            .with_old([0; 8], [100; 8]),
    ];

    let trace = generate_memw_aligned_trace(&ops);
    assert_eq!(trace.num_cols(), cols::NUM_COLUMNS);
    assert!(trace.num_rows() >= 2);

    // Check address decomposition for op[1]: addr = 0x1000
    // base_address[0] (low half)  = 0x1000
    // base_address[1] (mid half)  = 0
    // base_address[2] (high word) = 0
    assert_eq!(
        *trace.get_main(1, cols::BASE_ADDRESS[0]),
        FE::from(0x1000u64)
    );
    assert_eq!(*trace.get_main(1, cols::BASE_ADDRESS[1]), FE::from(0u64));
    assert_eq!(*trace.get_main(1, cols::BASE_ADDRESS[2]), FE::from(0u64));
}
