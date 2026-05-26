//! Tests for the LOAD table.

use crate::tables::load::*;

#[test]
fn test_load_trace_generation() {
    // Load 4 bytes, sign-extend
    let ops = vec![
        LoadOperation::new(
            0x1000,
            100,
            4,
            true,
            [0x12, 0x34, 0x56, 0x78, 0xFF, 0xFF, 0xFF, 0xFF],
        ),
        LoadOperation::new(
            0x2000,
            200,
            1,
            false,
            [0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        ),
    ];

    let trace = generate_load_trace(&ops);
    assert_eq!(trace.num_cols(), cols::NUM_COLUMNS);
    assert!(trace.num_rows() >= 2);
}

#[test]
fn test_read_flags() {
    // "Exactly N" semantics per spec
    let op1 = LoadOperation::new(0, 0, 1, false, [0; 8]);
    assert_eq!(op1.read_flags(), (false, false, false)); // no flags for 1 byte

    let op2 = LoadOperation::new(0, 0, 2, false, [0; 8]);
    assert_eq!(op2.read_flags(), (true, false, false)); // read2 only

    let op4 = LoadOperation::new(0, 0, 4, false, [0; 8]);
    assert_eq!(op4.read_flags(), (false, true, false)); // read4 only

    let op8 = LoadOperation::new(0, 0, 8, false, [0; 8]);
    assert_eq!(op8.read_flags(), (false, false, true)); // read8 only
}

#[test]
fn test_sign_bit_extraction() {
    // Byte with MSB set
    let op1 = LoadOperation::new(0, 0, 1, true, [0x80, 0, 0, 0, 0, 0, 0, 0]);
    assert!(op1.compute_sign_bit());

    // Byte without MSB set
    let op2 = LoadOperation::new(0, 0, 1, true, [0x7F, 0, 0, 0, 0, 0, 0, 0]);
    assert!(!op2.compute_sign_bit());

    // Halfword with MSB set
    let op3 = LoadOperation::new(0, 0, 2, true, [0x00, 0x80, 0, 0, 0, 0, 0, 0]);
    assert!(op3.compute_sign_bit());

    // Word with MSB set
    let op4 = LoadOperation::new(0, 0, 4, true, [0, 0, 0, 0x80, 0, 0, 0, 0]);
    assert!(op4.compute_sign_bit());
}
