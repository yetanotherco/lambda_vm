//! Tests for the BITWISE precomputed table.

use crate::tables64::bitwise::{
    NUM_ROWS, bus_interactions, cols, generate_bitwise_trace, row_index,
};
use crate::tables64::types::FE;

#[test]
fn test_row_index() {
    // First row: x=0, y=0, z=0
    assert_eq!(row_index(0, 0, 0), 0);

    // x=1, y=0, z=0
    assert_eq!(row_index(1, 0, 0), 1);

    // x=0, y=1, z=0
    assert_eq!(row_index(0, 1, 0), 256);

    // x=0, y=0, z=1
    assert_eq!(row_index(0, 0, 1), 256 * 256);

    // Last row: x=255, y=255, z=15
    assert_eq!(row_index(255, 255, 15), NUM_ROWS - 1);
}

#[test]
fn test_generate_bitwise_trace() {
    let trace = generate_bitwise_trace();

    // Check dimensions
    assert_eq!(trace.num_rows(), NUM_ROWS);

    // Check a few specific values
    // Row for x=5, y=3, z=0
    let row = row_index(5, 3, 0);
    let row_data = trace.main_table.get_row(row);

    assert_eq!(row_data[cols::X], FE::from(5u64));
    assert_eq!(row_data[cols::Y], FE::from(3u64));
    assert_eq!(row_data[cols::Z], FE::from(0u64));
    assert_eq!(row_data[cols::AND], FE::from(1u64)); // 5 & 3 = 1
    assert_eq!(row_data[cols::OR], FE::from(7u64)); // 5 | 3 = 7
    assert_eq!(row_data[cols::XOR], FE::from(6u64)); // 5 ^ 3 = 6

    // Check MSB8 for x=128 (MSB set)
    let row = row_index(128, 0, 0);
    let row_data = trace.main_table.get_row(row);
    assert_eq!(row_data[cols::MSB8], FE::from(1u64));

    // Check MSB8 for x=127 (MSB not set)
    let row = row_index(127, 0, 0);
    let row_data = trace.main_table.get_row(row);
    assert_eq!(row_data[cols::MSB8], FE::from(0u64));

    // Check MSB16 for halfword = 32768 (0x8000)
    // 32768 = 0 + 256 * 128
    let row = row_index(0, 128, 0);
    let row_data = trace.main_table.get_row(row);
    assert_eq!(row_data[cols::MSB16], FE::from(1u64));

    // Check shift: x=1, y=0, z=4 -> SLL = 16, SLLC = 0
    let row = row_index(1, 0, 4);
    let row_data = trace.main_table.get_row(row);
    assert_eq!(row_data[cols::SLL], FE::from(16u64)); // 1 << 4 = 16
    assert_eq!(row_data[cols::SLLC], FE::from(0u64)); // no carry

    // Check shift with carry: x=0, y=128, z=1 -> halfword=32768, SLL=0, SLLC=1
    let row = row_index(0, 128, 1);
    let row_data = trace.main_table.get_row(row);
    // 32768 << 1 = 65536 & 0xFFFF = 0
    assert_eq!(row_data[cols::SLL], FE::from(0u64));
    // 32768 >> (16-1) = 32768 >> 15 = 1
    assert_eq!(row_data[cols::SLLC], FE::from(1u64));
}

#[test]
fn test_zero_check() {
    let trace = generate_bitwise_trace();

    // ZERO should be 1 only when both X and Y are 0
    let row = row_index(0, 0, 0);
    assert_eq!(trace.main_table.get_row(row)[cols::ZERO], FE::from(1u64));

    let row = row_index(1, 0, 0);
    assert_eq!(trace.main_table.get_row(row)[cols::ZERO], FE::from(0u64));

    let row = row_index(0, 1, 0);
    assert_eq!(trace.main_table.get_row(row)[cols::ZERO], FE::from(0u64));
}

#[test]
fn test_bus_interactions_count() {
    let interactions = bus_interactions();
    // Should have 11 interactions (one for each lookup type)
    assert_eq!(interactions.len(), 11);
}

#[test]
fn test_first_row() {
    // First row: x=0, y=0, z=0
    let trace = generate_bitwise_trace();
    let row = row_index(0, 0, 0);
    let row_data = trace.main_table.get_row(row);

    assert_eq!(row_data[cols::X], FE::from(0u64));
    assert_eq!(row_data[cols::Y], FE::from(0u64));
    assert_eq!(row_data[cols::Z], FE::from(0u64));
    assert_eq!(row_data[cols::AND], FE::from(0u64)); // 0 & 0 = 0
    assert_eq!(row_data[cols::OR], FE::from(0u64)); // 0 | 0 = 0
    assert_eq!(row_data[cols::XOR], FE::from(0u64)); // 0 ^ 0 = 0
    assert_eq!(row_data[cols::MSB8], FE::from(0u64)); // MSB of 0 = 0
    assert_eq!(row_data[cols::MSB16], FE::from(0u64)); // MSB of 0 = 0
    assert_eq!(row_data[cols::ZERO], FE::from(1u64)); // 0 and 0 are both zero
    assert_eq!(row_data[cols::SLL], FE::from(0u64)); // 0 << 0 = 0
    assert_eq!(row_data[cols::SLLC], FE::from(0u64)); // 0 >> 16 = 0
}

#[test]
fn test_last_row() {
    // Last row: x=255, y=255, z=15
    let trace = generate_bitwise_trace();
    let row = row_index(255, 255, 15);
    let row_data = trace.main_table.get_row(row);

    assert_eq!(row_data[cols::X], FE::from(255u64));
    assert_eq!(row_data[cols::Y], FE::from(255u64));
    assert_eq!(row_data[cols::Z], FE::from(15u64));
    assert_eq!(row_data[cols::AND], FE::from(255u64)); // 255 & 255 = 255
    assert_eq!(row_data[cols::OR], FE::from(255u64)); // 255 | 255 = 255
    assert_eq!(row_data[cols::XOR], FE::from(0u64)); // 255 ^ 255 = 0
    assert_eq!(row_data[cols::MSB8], FE::from(1u64)); // MSB of 255 = 1
    // halfword = 255 + 256*255 = 65535 = 0xFFFF, MSB is bit 15 = 1
    assert_eq!(row_data[cols::MSB16], FE::from(1u64));
    assert_eq!(row_data[cols::ZERO], FE::from(0u64)); // not zero
    // SLL: (65535 << 15) & 0xFFFF = 0x8000 = 32768
    assert_eq!(row_data[cols::SLL], FE::from(32768u64));
    // SLLC: 65535 >> (16 - 15) = 65535 >> 1 = 32767
    assert_eq!(row_data[cols::SLLC], FE::from(32767u64));
}

#[test]
fn test_boundary_msb16() {
    let trace = generate_bitwise_trace();

    // halfword = 32767 (0x7FFF): MSB16 should be 0
    // 32767 = 255 + 256*127, so x=255, y=127
    let row = row_index(255, 127, 0);
    assert_eq!(trace.main_table.get_row(row)[cols::MSB16], FE::from(0u64));

    // halfword = 32768 (0x8000): MSB16 should be 1
    // 32768 = 0 + 256*128, so x=0, y=128
    let row = row_index(0, 128, 0);
    assert_eq!(trace.main_table.get_row(row)[cols::MSB16], FE::from(1u64));
}

#[test]
fn test_shift_boundaries() {
    let trace = generate_bitwise_trace();

    // Test SLL and SLLC at z=0 (no shift)
    // halfword = 1 (x=1, y=0)
    let row = row_index(1, 0, 0);
    let row_data = trace.main_table.get_row(row);
    assert_eq!(row_data[cols::SLL], FE::from(1u64)); // 1 << 0 = 1
    assert_eq!(row_data[cols::SLLC], FE::from(0u64)); // 1 >> 16 = 0

    // Test SLL and SLLC at z=15 (max shift)
    // halfword = 1 (x=1, y=0)
    let row = row_index(1, 0, 15);
    let row_data = trace.main_table.get_row(row);
    // SLL: (1 << 15) & 0xFFFF = 0x8000 = 32768
    assert_eq!(row_data[cols::SLL], FE::from(32768u64));
    // SLLC: 1 >> (16 - 15) = 1 >> 1 = 0
    assert_eq!(row_data[cols::SLLC], FE::from(0u64));

    // Test with halfword = 0x8001 = 32769 (x=1, y=128)
    let row = row_index(1, 128, 1);
    let row_data = trace.main_table.get_row(row);
    // halfword = 1 + 256*128 = 32769 = 0x8001
    // SLL: (32769 << 1) & 0xFFFF = 65538 & 0xFFFF = 2
    assert_eq!(row_data[cols::SLL], FE::from(2u64));
    // SLLC: 32769 >> (16 - 1) = 32769 >> 15 = 1
    assert_eq!(row_data[cols::SLLC], FE::from(1u64));
}

#[test]
fn test_all_bitwise_operations() {
    let trace = generate_bitwise_trace();

    // Test with x=0xAA, y=0x55 (alternating bits)
    let row = row_index(0xAA, 0x55, 0);
    let row_data = trace.main_table.get_row(row);

    assert_eq!(row_data[cols::AND], FE::from(0u64)); // 0xAA & 0x55 = 0
    assert_eq!(row_data[cols::OR], FE::from(0xFFu64)); // 0xAA | 0x55 = 0xFF
    assert_eq!(row_data[cols::XOR], FE::from(0xFFu64)); // 0xAA ^ 0x55 = 0xFF
    assert_eq!(row_data[cols::MSB8], FE::from(1u64)); // MSB of 0xAA = 1

    // Test with x=0x55, y=0xAA
    let row = row_index(0x55, 0xAA, 0);
    let row_data = trace.main_table.get_row(row);

    assert_eq!(row_data[cols::AND], FE::from(0u64)); // 0x55 & 0xAA = 0
    assert_eq!(row_data[cols::OR], FE::from(0xFFu64)); // 0x55 | 0xAA = 0xFF
    assert_eq!(row_data[cols::XOR], FE::from(0xFFu64)); // 0x55 ^ 0xAA = 0xFF
    assert_eq!(row_data[cols::MSB8], FE::from(0u64)); // MSB of 0x55 = 0
}

#[test]
fn test_row_count() {
    let trace = generate_bitwise_trace();
    // 256 * 256 * 16 = 2^20 = 1048576
    assert_eq!(trace.num_rows(), 256 * 256 * 16);
    assert_eq!(trace.num_rows(), 1048576);
}
