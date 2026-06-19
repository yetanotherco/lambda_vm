//! Parity: GPU `generate_memw_register_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/memw_register.rs::generate_memw_register_trace`.

use math_cuda::memw_register_trace::generate_memw_register_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 10;

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
    let active = (flags >> 1) & 1 == 1;
    if !active {
        return;
    }
    let is_read = (flags >> 0) & 1;
    out[base + 0] = addr / 2;
    out[base + 1] = ts & 0xFFFF_FFFF;
    out[base + 2] = ts >> 32;
    out[base + 3] = values[0];
    out[base + 4] = values[1];
    out[base + 5] = olds[0];
    out[base + 6] = olds[1];
    out[base + 7] = old_ts & 0xFFFF_FFFF;
    out[base + 8] = is_read;
    out[base + 9] = 1 - is_read;
}

fn run_parity(num_rows: usize, num_active: usize, seed: u64) {
    assert!(num_active <= num_rows);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut base_addresses = vec![0u64; num_rows];
    let mut timestamps = vec![0u64; num_rows];
    let mut old_timestamps = vec![0u64; num_rows];
    let mut values = vec![0u64; num_rows * 2];
    let mut olds = vec![0u64; num_rows * 2];
    let mut flags = vec![0u64; num_rows];

    for i in 0..num_active {
        // CPU sends 2 * register_index (0..31).
        base_addresses[i] = 2 * ((rng.r#gen::<u32>() % 32) as u64);
        timestamps[i] = rng.r#gen::<u64>();
        old_timestamps[i] = rng.r#gen::<u64>();
        values[i * 2 + 0] = rng.r#gen::<u32>() as u64;
        values[i * 2 + 1] = rng.r#gen::<u32>() as u64;
        olds[i * 2 + 0] = rng.r#gen::<u32>() as u64;
        olds[i * 2 + 1] = rng.r#gen::<u32>() as u64;
        let is_read = (rng.r#gen::<u32>() & 1) as u64;
        flags[i] = is_read | (1 << 1); // active
    }

    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_row(
            &mut cpu,
            row * NUM_COLS,
            base_addresses[row],
            timestamps[row],
            old_timestamps[row],
            &values[row * 2..row * 2 + 2],
            &olds[row * 2..row * 2 + 2],
            flags[row],
        );
    }
    let gpu = generate_memw_register_trace_dev(
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
fn memw_register_parity_small() {
    run_parity(4, 3, 1);
    run_parity(16, 9, 2);
}

#[test]
fn memw_register_parity_realistic() {
    run_parity(1 << 16, 50_000, 100);
    run_parity(1 << 18, 200_000, 101);
}
