//! Parity: GPU `generate_store_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/store.rs::generate_store_trace`'s row layout.

use math_cuda::store_trace::generate_store_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 16;

fn cpu_store_trace(
    num_rows: usize,
    base_addresses: &[u64],
    timestamps: &[u64],
    values: &[u64],
    flags: &[u64],
) -> Vec<u64> {
    let mut data = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        let base = row * NUM_COLS;
        let addr = base_addresses[row];
        let ts = timestamps[row];
        let v = values[row];
        let f = flags[row];
        data[base] = addr & 0xFFFF_FFFF;
        data[base + 1] = addr >> 32;
        data[base + 2] = ts & 0xFFFF_FFFF;
        data[base + 3] = ts >> 32;
        data[base + 4] = (f >> 0) & 1;
        data[base + 5] = (f >> 1) & 1;
        data[base + 6] = (f >> 2) & 1;
        for i in 0..8 {
            data[base + 7 + i] = (v >> (8 * i)) & 0xFF;
        }
        data[base + 15] = (f >> 3) & 1;
    }
    data
}

fn run_parity(num_rows: usize, num_active: usize, seed: u64) {
    assert!(num_active <= num_rows);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut base_addresses = vec![0u64; num_rows];
    let mut timestamps = vec![0u64; num_rows];
    let mut values = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];
    for i in 0..num_active {
        base_addresses[i] = rng.r#gen::<u64>();
        timestamps[i] = rng.r#gen::<u64>();
        values[i] = rng.r#gen::<u64>();
        flags[i] = (rng.r#gen::<u64>() & 0x7) | (1 << 3); // 3 random write flags + mu=1
    }
    let cpu = cpu_store_trace(num_rows, &base_addresses, &timestamps, &values, &flags);
    let gpu =
        generate_store_trace_dev(num_rows, &base_addresses, &timestamps, &values, &flags, NUM_COLS)
            .unwrap();
    assert_eq!(cpu, gpu, "store trace mismatch num_rows={num_rows}");
}

#[test]
fn store_trace_parity_small() {
    run_parity(4, 3, 1);
    run_parity(8, 5, 2);
}

#[test]
fn store_trace_parity_realistic() {
    run_parity(1 << 14, 10_000, 100);
    run_parity(1 << 20, 1_000_000, 101);
}
