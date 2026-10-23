//! Tests for the MEMW_A (aligned memory word) table.

use crate::tables::memw::MemwOperation;
use crate::tables::memw_aligned::*;
use crate::tables::types::*;

#[test]
fn test_memw_aligned_trace_generation() {
    let ops = [
        MemwOperation::new(true, 4, [42, 0, 0, 0, 0, 0, 0, 0], 100, 2, true)
            .with_old([42, 0, 0, 0, 0, 0, 0, 0], [50, 50, 0, 0, 0, 0, 0, 0]),
        MemwOperation::new(false, 0x1000, [1, 2, 3, 4, 0, 0, 0, 0], 200, 4, false)
            .with_old([0; 8], [100; 8]),
    ];

    let rows: Vec<AlignedRow> = ops.iter().map(AlignedRow::from_memw).collect();
    let trace = generate_memw_aligned_trace(&rows);
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

/// An aligned row keeps every element of the op it packs: eight bytes of a
/// memory access at each width, the two 32-bit halves of a register access,
/// and the shared old timestamp; its MEMW_A row is the op's.
#[test]
fn an_aligned_row_keeps_its_ops_elements() {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut ops = Vec::new();
    for width in [1u8, 2, 4, 8] {
        for is_read in [false, true] {
            let mut value = [0u32; 8];
            let mut old = [0u32; 8];
            for i in 0..8 {
                value[i] = (next() & 0xFF) as u32;
                old[i] = (next() & 0xFF) as u32;
            }
            let old_ts = next() >> 8;
            let base = (next() >> 4) & !(width as u64 - 1);
            ops.push(
                MemwOperation::new(false, base, value, old_ts + 4, width, is_read)
                    .with_old(old, [old_ts; 8]),
            );
        }
    }
    for is_read in [false, true] {
        let value = [next() as u32, next() as u32, 0, 0, 0, 0, 0, 0];
        let old = [next() as u32, next() as u32, 0, 0, 0, 0, 0, 0];
        let old_ts = next() >> 8;
        ops.push(
            MemwOperation::new(true, 2 * 17, value, old_ts + (1 << 20), 2, is_read)
                .with_old(old, [old_ts, old_ts, 0, 0, 0, 0, 0, 0]),
        );
    }
    for op in &ops {
        let row = AlignedRow::from_memw(op);
        assert_eq!(row.value(), op.value, "{op:?}");
        assert_eq!(row.old(), op.old, "{op:?}");
        assert_eq!(row.old_timestamp(), op.old_timestamp[0]);
        assert_eq!(
            (
                row.base_address(),
                row.timestamp(),
                row.width(),
                row.is_read(),
                row.is_register()
            ),
            (
                op.base_address,
                op.timestamp,
                op.width,
                op.is_read,
                op.is_register
            )
        );
        assert_eq!(row.write_flags(), op.write_flags());
    }
}
