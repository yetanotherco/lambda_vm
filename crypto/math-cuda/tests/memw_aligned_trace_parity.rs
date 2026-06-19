//! Parity: GPU `generate_memw_aligned_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/memw_aligned.rs::generate_memw_aligned_trace`.

use math_cuda::memw_aligned_trace::generate_memw_aligned_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 29;

#[allow(clippy::too_many_arguments)]
fn cpu_row(
    out: &mut [u64],
    base: usize,
    addr: u64,
    ts: u64,
    old_ts: u64,
    values: &[u64],
    olds: &[u64],
    flags: u64,
) {
    let active = (flags >> 5) & 1 == 1;
    if !active {
        return; // out already zeroed
    }
    let is_register = (flags >> 0) & 1;
    let is_read = (flags >> 1) & 1;
    let w2 = (flags >> 2) & 1;
    let w4 = (flags >> 3) & 1;
    let w8 = (flags >> 4) & 1;

    out[base + 0] = is_register;
    out[base + 1] = addr & 0xFFFF;
    out[base + 2] = (addr >> 16) & 0xFFFF;
    out[base + 3] = addr >> 32;
    for i in 0..8 {
        out[base + 4 + i] = values[i];
    }
    out[base + 12] = ts & 0xFFFF_FFFF;
    out[base + 13] = ts >> 32;
    out[base + 14] = w2;
    out[base + 15] = w4;
    out[base + 16] = w8;
    for i in 0..8 {
        out[base + 17 + i] = olds[i];
    }
    out[base + 25] = old_ts & 0xFFFF_FFFF;
    out[base + 26] = old_ts >> 32;
    out[base + 27] = is_read;
    out[base + 28] = 1 - is_read;
}

fn run_parity(num_rows: usize, num_active: usize, seed: u64) {
    assert!(num_active <= num_rows);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut base_addresses = vec![0u64; num_rows];
    let mut timestamps = vec![0u64; num_rows];
    let mut old_timestamps = vec![0u64; num_rows];
    let mut values = vec![0u64; num_rows * 8];
    let mut olds = vec![0u64; num_rows * 8];
    let mut flags = vec![0u64; num_rows];

    for i in 0..num_active {
        base_addresses[i] = rng.r#gen::<u64>();
        timestamps[i] = rng.r#gen::<u64>();
        old_timestamps[i] = rng.r#gen::<u64>();
        for j in 0..8 {
            values[i * 8 + j] = (rng.r#gen::<u32>() & 0xFF) as u64;
            olds[i * 8 + j] = (rng.r#gen::<u32>() & 0xFF) as u64;
        }
        let is_register = (rng.r#gen::<u32>() & 1) as u64;
        let is_read = (rng.r#gen::<u32>() & 1) as u64;
        // Pick one of {1,2,4,8} → flags pattern
        let widthsel = rng.r#gen::<u32>() % 4;
        let (w2, w4, w8) = match widthsel {
            0 => (0u64, 0, 0), // byte: no flag set
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
            old_timestamps[row],
            &values[row * 8..row * 8 + 8],
            &olds[row * 8..row * 8 + 8],
            flags[row],
        );
    }
    let gpu = generate_memw_aligned_trace_dev(
        num_rows,
        &base_addresses,
        &timestamps,
        &old_timestamps,
        &values,
        &olds,
        &flags,
        NUM_COLS,
    )
    .unwrap();
    assert_eq!(cpu, gpu);
}

#[test]
fn memw_aligned_parity_small() {
    run_parity(4, 3, 1);
    run_parity(16, 9, 2);
}

#[test]
fn memw_aligned_parity_realistic() {
    run_parity(1 << 14, 12_000, 100);
    run_parity(1 << 16, 50_000, 101);
}
