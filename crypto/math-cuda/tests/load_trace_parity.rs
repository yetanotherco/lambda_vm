//! Parity: GPU `generate_load_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/load.rs::generate_load_trace`'s row layout.

use math_cuda::load_trace::generate_load_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 18;

fn cpu_load_trace(
    num_rows: usize,
    base_addresses: &[u64],
    timestamps: &[u64],
    flags: &[u64],
    res_bytes: &[u64],
) -> Vec<u64> {
    let mut data = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        let base = row * NUM_COLS;
        let addr = base_addresses[row];
        let ts = timestamps[row];
        let f = flags[row];
        data[base] = addr & 0xFFFF_FFFF;
        data[base + 1] = addr >> 32;
        data[base + 2] = ts & 0xFFFF_FFFF;
        data[base + 3] = ts >> 32;
        data[base + 4] = (f >> 0) & 1;
        data[base + 5] = (f >> 1) & 1;
        data[base + 6] = (f >> 2) & 1;
        data[base + 7] = (f >> 3) & 1;
        for i in 0..8 {
            data[base + 8 + i] = res_bytes[row * 8 + i];
        }
        data[base + 16] = (f >> 4) & 1;
        data[base + 17] = (f >> 5) & 1;
    }
    data
}

fn run_parity(num_rows: usize, num_active: usize, seed: u64) {
    assert!(num_active <= num_rows);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut base_addresses = vec![0u64; num_rows];
    let mut timestamps = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];
    let mut res_bytes = vec![0u64; 8 * num_rows];
    for i in 0..num_active {
        base_addresses[i] = rng.r#gen::<u64>();
        timestamps[i] = rng.r#gen::<u64>();
        // 6 random flag bits + mu=1 for active rows.
        flags[i] = (rng.r#gen::<u64>() & 0x1F) | (1 << 5);
        for b in 0..8 {
            res_bytes[i * 8 + b] = rng.r#gen::<u8>() as u64;
        }
    }
    let cpu = cpu_load_trace(num_rows, &base_addresses, &timestamps, &flags, &res_bytes);
    let gpu =
        generate_load_trace_dev(num_rows, &base_addresses, &timestamps, &flags, &res_bytes, NUM_COLS)
            .unwrap();
    assert_eq!(cpu, gpu, "load trace mismatch num_rows={num_rows}");
}

#[test]
fn load_trace_parity_small() {
    run_parity(4, 3, 1);
    run_parity(8, 5, 2);
}

#[test]
fn load_trace_parity_realistic() {
    run_parity(1 << 14, 12_000, 100);
    run_parity(1 << 20, 1_000_000, 101);
}
