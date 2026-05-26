//! Tests for the MEMW_R (register memory word) table.

use crate::tables::memw::MemwOperation;
use crate::tables::memw_register::*;
use crate::tables::types::*;

#[test]
fn test_memw_register_trace_generation() {
    // Create a simple register op (reg x1 = address 1, so base_address = 2)
    let ops = vec![
        MemwOperation::new(
            true, // is_register
            2,    // base_address = 2 * register_index (reg x1)
            [42, 7, 0, 0, 0, 0, 0, 0],
            100,
            2, // width = 2 words (registers are DWordWL)
            true,
        )
        .with_old([10, 3, 0, 0, 0, 0, 0, 0], [50, 50, 0, 0, 0, 0, 0, 0]),
    ];

    let trace = generate_memw_register_trace(&ops);
    assert_eq!(trace.num_cols(), cols::NUM_COLUMNS);
    assert!(trace.num_rows() >= 4); // minimum 4 rows

    // ADDRESS = base_address / 2 = 2 / 2 = 1
    assert_eq!(*trace.get_main(0, cols::ADDRESS), FE::from(1u64));

    // TIMESTAMP split
    assert_eq!(*trace.get_main(0, cols::TIMESTAMP_0), FE::from(100u64));
    assert_eq!(*trace.get_main(0, cols::TIMESTAMP_1), FE::from(0u64));

    // Values
    assert_eq!(*trace.get_main(0, cols::VAL_0), FE::from(42u64));
    assert_eq!(*trace.get_main(0, cols::VAL_1), FE::from(7u64));

    // Old values
    assert_eq!(*trace.get_main(0, cols::OLD_0), FE::from(10u64));
    assert_eq!(*trace.get_main(0, cols::OLD_1), FE::from(3u64));

    // Old timestamp lo
    assert_eq!(*trace.get_main(0, cols::OLD_TIMESTAMP_LO), FE::from(50u64));

    // Multiplicity: is_read = true => MU_READ=1, MU_WRITE=0
    assert_eq!(*trace.get_main(0, cols::MU_READ), FE::from(1u64));
    assert_eq!(*trace.get_main(0, cols::MU_WRITE), FE::from(0u64));
}

#[test]
fn test_memw_register_trace_generation_write_op() {
    // Write op: is_read = false => MU_WRITE=1, MU_READ=0
    let ops = vec![
        MemwOperation::new(
            true, // is_register
            4,    // base_address = 2 * register_index (reg x2)
            [99, 55, 0, 0, 0, 0, 0, 0],
            200,
            2,     // width = 2 words
            false, // is_read = false (write)
        )
        .with_old([11, 22, 0, 0, 0, 0, 0, 0], [180, 180, 0, 0, 0, 0, 0, 0]),
    ];

    let trace = generate_memw_register_trace(&ops);

    // ADDRESS = base_address / 2 = 4 / 2 = 2
    assert_eq!(*trace.get_main(0, cols::ADDRESS), FE::from(2u64));

    // Values
    assert_eq!(*trace.get_main(0, cols::VAL_0), FE::from(99u64));
    assert_eq!(*trace.get_main(0, cols::VAL_1), FE::from(55u64));

    // Old values
    assert_eq!(*trace.get_main(0, cols::OLD_0), FE::from(11u64));
    assert_eq!(*trace.get_main(0, cols::OLD_1), FE::from(22u64));

    // Old timestamp lo
    assert_eq!(*trace.get_main(0, cols::OLD_TIMESTAMP_LO), FE::from(180u64));

    // Multiplicity: is_read = false => MU_WRITE=1, MU_READ=0
    assert_eq!(*trace.get_main(0, cols::MU_READ), FE::from(0u64));
    assert_eq!(*trace.get_main(0, cols::MU_WRITE), FE::from(1u64));
}
