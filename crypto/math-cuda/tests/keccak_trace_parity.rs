//! Parity: GPU `generate_keccak_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/keccak.rs::generate_keccak_trace`.

use math_cuda::keccak_trace::generate_keccak_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 511;

fn cpu_row(
    out: &mut [u64],
    base: usize,
    ts: u64,
    addr: u64,
    input: &[u64],
    output: &[u64],
    flags: u64,
) {
    let active = (flags >> 0) & 1 == 1;
    if !active {
        // Padding row: state_ptr[lane][0] = 8 * lane_idx.
        for lane in 0..25 {
            out[base + 410 + lane * 4] = (lane as u64) * 8;
        }
        return;
    }

    out[base + 0] = ts & 0xFFFF_FFFF;
    out[base + 1] = ts >> 32;
    for b in 0..8 {
        out[base + 2 + b] = (addr >> (b * 8)) & 0xFF;
    }
    for lane in 0..25 {
        let in_l = input[lane];
        let out_l = output[lane];
        for b in 0..8 {
            out[base + 10 + lane * 8 + b] = (in_l >> (b * 8)) & 0xFF;
            out[base + 210 + lane * 8 + b] = (out_l >> (b * 8)) & 0xFF;
        }
    }
    for lane in 0..25 {
        let ptr = addr.checked_add(lane as u64 * 8).expect("range");
        out[base + 410 + lane * 4 + 0] = ptr & 0xFFFF;
        out[base + 410 + lane * 4 + 1] = (ptr >> 16) & 0xFFFF;
        out[base + 410 + lane * 4 + 2] = (ptr >> 32) & 0xFFFF;
        out[base + 410 + lane * 4 + 3] = (ptr >> 48) & 0xFFFF;
    }
    out[base + 510] = 1;
}

fn run_parity(num_rows: usize, num_active: usize, seed: u64) {
    assert!(num_active <= num_rows);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut timestamps = vec![0u64; num_rows];
    let mut state_addrs = vec![0u64; num_rows];
    let mut inputs = vec![0u64; num_rows * 25];
    let mut outputs = vec![0u64; num_rows * 25];
    let mut flags = vec![0u64; num_rows];

    for i in 0..num_active {
        timestamps[i] = rng.r#gen::<u64>();
        // Keep addr below 2^60 so addr + 8*24 doesn't overflow (matches
        // executor's address-range validation contract).
        state_addrs[i] = rng.r#gen::<u64>() & ((1u64 << 60) - 1);
        for lane in 0..25 {
            inputs[i * 25 + lane] = rng.r#gen::<u64>();
            outputs[i * 25 + lane] = rng.r#gen::<u64>();
        }
        flags[i] = 1; // active
    }

    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_row(
            &mut cpu,
            row * NUM_COLS,
            timestamps[row],
            state_addrs[row],
            &inputs[row * 25..row * 25 + 25],
            &outputs[row * 25..row * 25 + 25],
            flags[row],
        );
    }
    let gpu = generate_keccak_trace_dev(
        num_rows,
        &timestamps,
        &state_addrs,
        &inputs,
        &outputs,
        &flags,
        NUM_COLS,
    )
    .unwrap();
    assert_eq!(cpu, gpu);
}

#[test]
fn keccak_trace_parity_small() {
    run_parity(4, 3, 1);
    run_parity(8, 5, 2);
}

#[test]
fn keccak_trace_parity_realistic() {
    run_parity(64, 50, 100);
    run_parity(256, 200, 101);
}
