//! Tests for the MEMW table.

use crate::tables::memw::*;
use crate::tables::types::*;

#[test]
fn test_memw_trace_generation() {
    let ops = vec![
        MemwOperation::new(false, 0x1000, [1, 2, 3, 4, 5, 6, 7, 8], 100, 8, false)
            .with_old([0; 8], [50; 8]),
        MemwOperation::new(true, 5, [42, 0, 0, 0, 0, 0, 0, 0], 200, 1, true)
            .with_old([10, 0, 0, 0, 0, 0, 0, 0], [150, 0, 0, 0, 0, 0, 0, 0]),
    ];

    let trace = generate_memw_trace(&ops);
    assert_eq!(trace.num_cols(), cols::NUM_COLUMNS);
    assert!(trace.num_rows() >= 2);
}

#[test]
fn test_write_flags() {
    let op1 = MemwOperation::new(false, 0, [0; 8], 0, 1, false);
    assert_eq!(op1.write_flags(), (false, false, false));

    let op2 = MemwOperation::new(false, 0, [0; 8], 0, 2, false);
    assert_eq!(op2.write_flags(), (true, false, false));

    let op4 = MemwOperation::new(false, 0, [0; 8], 0, 4, false);
    assert_eq!(op4.write_flags(), (false, true, false));

    let op8 = MemwOperation::new(false, 0, [0; 8], 0, 8, false);
    assert_eq!(op8.write_flags(), (false, false, true));
}

#[test]
fn test_carry_flags() {
    // Address 0xFFFF_FFFF should carry when adding 1
    let op =
        MemwOperation::new(false, 0xFFFF_FFFF, [0; 8], 100, 8, false).with_old([0; 8], [50; 8]);
    let trace = generate_memw_trace(&[op]);

    // All 7 carry flags should be 1 since 0xFFFF_FFFF + i >= 2^32 for i >= 1
    for i in 0..7 {
        let val = trace.get_main(0, cols::CARRY[i]);
        assert_eq!(*val, FE::one(), "carry[{i}] should be 1");
    }

    // Address 0x0000_0000 should not carry
    let op2 =
        MemwOperation::new(false, 0x0000_0000, [0; 8], 100, 8, false).with_old([0; 8], [50; 8]);
    let trace2 = generate_memw_trace(&[op2]);
    for i in 0..7 {
        let val = trace2.get_main(0, cols::CARRY[i]);
        assert_eq!(*val, FE::zero(), "carry[{i}] should be 0");
    }

    // Address 0xFFFF_FFFE with width=8 exercises mixed per-byte carry bits:
    // carry[0]=0 (0xFFFF_FFFE+1 = 0xFFFF_FFFF < 2^32)
    // carry[1..6]=1 (0xFFFF_FFFE+2..8 >= 2^32)
    let op3 =
        MemwOperation::new(false, 0xFFFF_FFFE, [0; 8], 100, 8, false).with_old([0; 8], [50; 8]);
    let trace3 = generate_memw_trace(&[op3]);
    let val0 = trace3.get_main(0, cols::CARRY[0]);
    assert_eq!(
        *val0,
        FE::zero(),
        "carry[0] should be 0 for base 0xFFFF_FFFE"
    );
    for i in 1..7 {
        let val = trace3.get_main(0, cols::CARRY[i]);
        assert_eq!(
            *val,
            FE::one(),
            "carry[{i}] should be 1 for base 0xFFFF_FFFE"
        );
    }
}
