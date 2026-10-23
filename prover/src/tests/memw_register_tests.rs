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

/// A register read's Memw message as the CPU sends it and as MEMW_R receives
/// it, at timestamp `ts`: the CPU's clock is one column, sent with a zero high
/// word ("CPU timestamps fit in 32 bits"); MEMW_R takes the timestamp as its
/// two 32-bit words.
fn register_read_messages(ts: u64) -> (Vec<FE>, Vec<FE>) {
    use crate::tables::cpu;
    let (reg, lo, hi) = (1u64, 42u32, 7u32);

    let cpu_read = cpu::bus_interactions()
        .into_iter()
        .find(|i| {
            let read = i.columns_read();
            i.bus_id == BusId::Memw as u64
                && i.is_sender
                && read.contains(&cpu::cols::READ_REGISTER1)
                && read.contains(&cpu::cols::RS1)
        })
        .expect("the CPU sends its rs1 read on Memw");
    let cpu_row = |c: usize| -> FE {
        match c {
            cpu::cols::TIMESTAMP => FE::from(ts),
            cpu::cols::RS1 => FE::from(reg),
            cpu::cols::RV1_0 => FE::from(lo as u64),
            cpu::cols::RV1_1 => FE::from(hi as u64),
            _ => FE::zero(),
        }
    };
    let sent: Vec<FE> = cpu_read
        .values
        .iter()
        .flat_map(|v| v.combine_from::<GoldilocksField, _>(cpu_row))
        .collect();

    let ops = vec![
        MemwOperation::new(true, 2 * reg, [lo, hi, 0, 0, 0, 0, 0, 0], ts, 2, true).with_old(
            [lo, hi, 0, 0, 0, 0, 0, 0],
            [ts - 4, ts - 4, 0, 0, 0, 0, 0, 0],
        ),
    ];
    let trace = generate_memw_register_trace(&ops);
    let memw_r_read = bus_interactions()
        .into_iter()
        .find(|i| {
            i.bus_id == BusId::Memw as u64
                && !i.is_sender
                && i.columns_read().contains(&cols::MU_READ)
        })
        .expect("MEMW_R receives register reads on Memw");
    let received: Vec<FE> = memw_r_read
        .values
        .iter()
        .flat_map(|v| v.combine_from::<GoldilocksField, _>(|c| *trace.get_main(0, c)))
        .collect();
    (sent, received)
}

/// The block's clock is `4·cycle + 4` over the whole execution, so it passes
/// 2^32 at cycle 2^30 − 1. Below 2^32 the CPU's register-read message and
/// MEMW_R's are one; from 2^32 on they differ, and an honest block past 2^30
/// cycles cannot balance the Memw bus (T200-R, 1.83 G cycles: BusImbalance).
#[test]
fn the_cpu_and_memw_r_register_read_messages_part_at_2_32() {
    let below = (1u64 << 32) - 4;
    let (sent, received) = register_read_messages(below);
    assert_eq!(sent.len(), received.len());
    assert_eq!(sent, received, "one message below 2^32");

    // Past 2^32 only the timestamp's two slots (after old 8, is_register,
    // base 2, value 8) differ: the CPU's whole clock and a zero against
    // MEMW_R's low and high words.
    let past = (1u64 << 32) + 4;
    let (sent, received) = register_read_messages(past);
    assert_eq!(sent.len(), received.len());
    let differ: Vec<usize> = (0..sent.len())
        .filter(|&k| sent[k] != received[k])
        .collect();
    assert_eq!(differ, vec![19, 20]);
    assert_eq!((sent[19], sent[20]), (FE::from(past), FE::zero()));
    assert_eq!(
        (received[19], received[20]),
        (FE::from(past & 0xFFFF_FFFF), FE::from(past >> 32))
    );
}
