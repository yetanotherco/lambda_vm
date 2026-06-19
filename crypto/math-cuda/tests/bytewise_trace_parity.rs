//! Parity: GPU `generate_bytewise_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/bytewise.rs::generate_bytewise_trace`'s
//! row layout (CPU side does dedup; GPU does byte decomp + layout).

use math_cuda::bytewise_trace::generate_bytewise_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 26;

fn cpu_bytewise_trace(
    num_rows: usize,
    a_values: &[u64],
    b_values: &[u64],
    res_values: &[u64],
    ops: &[u64],
    multiplicities: &[u64],
) -> Vec<u64> {
    let mut data = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        let base = row * NUM_COLS;
        let a = a_values[row];
        let b = b_values[row];
        let r = res_values[row];
        for i in 0..8 {
            data[base + i] = (a >> (8 * i)) & 0xFF;
            data[base + 8 + i] = (b >> (8 * i)) & 0xFF;
            data[base + 17 + i] = (r >> (8 * i)) & 0xFF;
        }
        data[base + 16] = ops[row];
        data[base + 25] = multiplicities[row];
    }
    data
}

fn run_parity(num_rows: usize, num_active: usize, seed: u64) {
    assert!(num_active <= num_rows);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut a_values = vec![0u64; num_rows];
    let mut b_values = vec![0u64; num_rows];
    let mut res_values = vec![0u64; num_rows];
    let mut ops = vec![0u64; num_rows];
    let mut multiplicities = vec![0u64; num_rows];
    for i in 0..num_active {
        let a = rng.r#gen::<u64>();
        let b = rng.r#gen::<u64>();
        let op = rng.r#gen::<u32>() % 3;
        let res = match op {
            0 => a & b,
            1 => a | b,
            _ => a ^ b,
        };
        a_values[i] = a;
        b_values[i] = b;
        ops[i] = op as u64;
        res_values[i] = res;
        multiplicities[i] = (rng.r#gen::<u32>() % 256) as u64 + 1;
    }
    let cpu = cpu_bytewise_trace(num_rows, &a_values, &b_values, &res_values, &ops, &multiplicities);
    let gpu = generate_bytewise_trace_dev(
        num_rows,
        &a_values,
        &b_values,
        &res_values,
        &ops,
        &multiplicities,
        NUM_COLS,
    )
    .unwrap();
    assert_eq!(cpu, gpu, "bytewise trace mismatch num_rows={num_rows}");
}

#[test]
fn bytewise_trace_parity_small() {
    run_parity(4, 3, 1);
    run_parity(8, 5, 2);
}

#[test]
fn bytewise_trace_parity_realistic() {
    run_parity(1 << 14, 12_000, 100);
    run_parity(1 << 18, 200_000, 101);
}
