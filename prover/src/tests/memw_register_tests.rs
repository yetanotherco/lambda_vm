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

    // TIMESTAMP (single Word)
    assert_eq!(*trace.get_main(0, cols::TIMESTAMP), FE::from(100u64));

    // Values
    assert_eq!(*trace.get_main(0, cols::VAL_0), FE::from(42u64));
    assert_eq!(*trace.get_main(0, cols::VAL_1), FE::from(7u64));

    // Old values
    assert_eq!(*trace.get_main(0, cols::OLD_0), FE::from(10u64));
    assert_eq!(*trace.get_main(0, cols::OLD_1), FE::from(3u64));

    // Old timestamp
    assert_eq!(*trace.get_main(0, cols::OLD_TIMESTAMP), FE::from(50u64));

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

    // Old timestamp
    assert_eq!(*trace.get_main(0, cols::OLD_TIMESTAMP), FE::from(180u64));

    // Multiplicity: is_read = false => MU_WRITE=1, MU_READ=0
    assert_eq!(*trace.get_main(0, cols::MU_READ), FE::from(0u64));
    assert_eq!(*trace.get_main(0, cols::MU_WRITE), FE::from(1u64));
}

// MEMW_R register-ordering soundness rests on the IS_HALFWORD lookup over
// `timestamp - old_timestamp - 1`, valid iff the delta `timestamp - old_timestamp`
// is in `[1, 2^16]` (i.e. the IS_HALF input is in `[0, 2^16-1]`). The trace stores
// `timestamp` and `old_timestamp` directly, and the bus feeds their difference into
// the lookup, so these tests pin the two ACCEPT boundaries by checking the stored
// columns reproduce IS_HALF inputs of exactly 0 and 0xFFFF.
//
// The REJECT boundaries (delta = 0 and delta > 2^16) never reach MEMW_R in an honest
// trace: the routing predicate `is_register_op` diverts them to MEMW_A, and its
// boundary cases are covered in `trace_builder_tests::routing_tests`
// (`test_is_register_op_delta_{zero,one,at_boundary,above_boundary}_*`). A forged
// out-of-range row would then fail the IS_HALFWORD lookup at prove time.

fn register_read_op(timestamp: u64, old_timestamp: u64) -> MemwOperation {
    MemwOperation::new(true, 2, [42, 7, 0, 0, 0, 0, 0, 0], timestamp, 2, true).with_old(
        [10, 3, 0, 0, 0, 0, 0, 0],
        [old_timestamp, old_timestamp, 0, 0, 0, 0, 0, 0],
    )
}

#[test]
fn test_memw_register_is_half_accept_delta_one() {
    // delta = 1 (minimum): IS_HALF input = timestamp - old_timestamp - 1 = 0.
    let (old_ts, ts) = (100u64, 101u64);
    let trace = generate_memw_register_trace(&[register_read_op(ts, old_ts)]);

    assert_eq!(*trace.get_main(0, cols::TIMESTAMP), FE::from(ts));
    assert_eq!(*trace.get_main(0, cols::OLD_TIMESTAMP), FE::from(old_ts));
    assert_eq!(
        ts - old_ts - 1,
        0,
        "IS_HALF input at the low accept boundary"
    );
}

#[test]
fn test_memw_register_is_half_accept_delta_max() {
    // delta = 2^16 (maximum): IS_HALF input = 2^16 - 1 = 0xFFFF (top of the Half range).
    let (old_ts, ts) = (100u64, 100u64 + (1 << 16));
    let trace = generate_memw_register_trace(&[register_read_op(ts, old_ts)]);

    assert_eq!(*trace.get_main(0, cols::TIMESTAMP), FE::from(ts));
    assert_eq!(*trace.get_main(0, cols::OLD_TIMESTAMP), FE::from(old_ts));
    assert_eq!(
        ts - old_ts - 1,
        0xFFFF,
        "IS_HALF input at the high accept boundary"
    );
}
