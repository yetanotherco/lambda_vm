//! Parity: GPU `generate_memw_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/memw.rs::generate_memw_trace`.

use math_cuda::memw_trace::generate_memw_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 49;

#[allow(clippy::too_many_arguments)]
fn cpu_row(
    out: &mut [u64],
    base: usize,
    addr: u64,
    ts: u64,
    values: &[u64],
    olds: &[u64],
    old_timestamps: &[u64],
    flags: u64,
) {
    let active = (flags >> 5) & 1 == 1;
    if !active {
        return;
    }
    let is_register = (flags >> 0) & 1;
    let is_read = (flags >> 1) & 1;
    let w2 = (flags >> 2) & 1;
    let w4 = (flags >> 3) & 1;
    let w8 = (flags >> 4) & 1;
    let addr_lo = addr & 0xFFFF_FFFF;

    out[base + 0] = is_register;
    out[base + 1] = addr_lo;
    out[base + 2] = addr >> 32;
    for i in 0..8 {
        out[base + 3 + i] = values[i];
    }
    out[base + 11] = ts & 0xFFFF_FFFF;
    out[base + 12] = ts >> 32;
    out[base + 13] = w2;
    out[base + 14] = w4;
    out[base + 15] = w8;
    for i in 0..8 {
        out[base + 16 + i] = olds[i];
    }
    for i in 0..7 {
        let overflows = (addr_lo + (i as u64 + 1) >= (1u64 << 32)) as u64;
        out[base + 24 + i] = overflows;
    }
    for i in 0..8 {
        let ots = old_timestamps[i];
        out[base + 31 + 2 * i] = ots & 0xFFFF_FFFF;
        out[base + 32 + 2 * i] = ots >> 32;
    }
    out[base + 47] = is_read;
    out[base + 48] = 1 - is_read;
}

fn run_parity(num_rows: usize, num_active: usize, seed: u64) {
    assert!(num_active <= num_rows);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut base_addresses = vec![0u64; num_rows];
    let mut timestamps = vec![0u64; num_rows];
    let mut values = vec![0u64; num_rows * 8];
    let mut olds = vec![0u64; num_rows * 8];
    let mut old_timestamps = vec![0u64; num_rows * 8];
    let mut flags = vec![0u64; num_rows];

    for i in 0..num_active {
        base_addresses[i] = rng.r#gen::<u64>();
        timestamps[i] = rng.r#gen::<u64>();
        for j in 0..8 {
            values[i * 8 + j] = (rng.r#gen::<u32>() & 0xFF) as u64;
            olds[i * 8 + j] = (rng.r#gen::<u32>() & 0xFF) as u64;
            old_timestamps[i * 8 + j] = rng.r#gen::<u64>();
        }
        let is_register = (rng.r#gen::<u32>() & 1) as u64;
        let is_read = (rng.r#gen::<u32>() & 1) as u64;
        let widthsel = rng.r#gen::<u32>() % 4;
        let (w2, w4, w8) = match widthsel {
            0 => (0u64, 0, 0),
            1 => (1, 0, 0),
            2 => (0, 1, 0),
            _ => (0, 0, 1),
        };
        flags[i] = is_register | (is_read << 1) | (w2 << 2) | (w4 << 3) | (w8 << 4) | (1 << 5);
    }

    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_row(
            &mut cpu,
            row * NUM_COLS,
            base_addresses[row],
            timestamps[row],
            &values[row * 8..row * 8 + 8],
            &olds[row * 8..row * 8 + 8],
            &old_timestamps[row * 8..row * 8 + 8],
            flags[row],
        );
    }
    let gpu = generate_memw_trace_dev(
        num_rows,
        &base_addresses,
        &timestamps,
        &values,
        &olds,
        &old_timestamps,
        &flags,
        NUM_COLS,
    )
    .unwrap();
    assert_eq!(cpu, gpu);
}

#[test]
fn memw_parity_small() {
    run_parity(4, 3, 1);
    run_parity(16, 9, 2);
}

#[test]
fn memw_parity_realistic() {
    run_parity(1 << 14, 12_000, 100);
    run_parity(1 << 16, 50_000, 101);
}

#[test]
fn memw_parity_carry_edges() {
    // Exercise base_address_lo values near the 2^32 boundary to test carry[i]
    // edge transitions.
    let edge_addr_los: &[u64] = &[
        0xFFFF_FFF0,
        0xFFFF_FFF8,
        0xFFFF_FFFB,
        0xFFFF_FFFE,
        0xFFFF_FFFF,
        0xFFFF_FF00,
    ];
    let num_rows = edge_addr_los.len().next_power_of_two().max(4);
    let mut base_addresses = vec![0u64; num_rows];
    let mut timestamps = vec![0u64; num_rows];
    let mut values = vec![0u64; num_rows * 8];
    let mut olds = vec![0u64; num_rows * 8];
    let mut old_timestamps = vec![0u64; num_rows * 8];
    let mut flags = vec![0u64; num_rows];

    for (i, &lo) in edge_addr_los.iter().enumerate() {
        base_addresses[i] = 0x12_3456_0000_0000 | lo;
        timestamps[i] = 0xCAFE_BEEF_DEAD_F00D;
        for j in 0..8 {
            values[i * 8 + j] = j as u64;
            olds[i * 8 + j] = (j * 17 + 3) as u64;
            old_timestamps[i * 8 + j] = (j as u64) * 0x10_0000_0000 + 1;
        }
        flags[i] = 1 | (1 << 1) | (1 << 4) | (1 << 5); // memory write8, read, active
    }

    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_row(
            &mut cpu,
            row * NUM_COLS,
            base_addresses[row],
            timestamps[row],
            &values[row * 8..row * 8 + 8],
            &olds[row * 8..row * 8 + 8],
            &old_timestamps[row * 8..row * 8 + 8],
            flags[row],
        );
    }
    let gpu = generate_memw_trace_dev(
        num_rows,
        &base_addresses,
        &timestamps,
        &values,
        &olds,
        &old_timestamps,
        &flags,
        NUM_COLS,
    )
    .unwrap();
    assert_eq!(cpu, gpu);
}
