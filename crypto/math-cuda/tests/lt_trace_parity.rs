//! Parity: GPU `generate_lt_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/lt.rs::generate_lt_trace`'s row layout.

use math_cuda::lt_trace::generate_lt_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 17;

fn cpu_row(out: &mut [u64], base: usize, lhs: u64, rhs: u64, flags: u64, mu: u64) {
    let active = (flags >> 2) & 1 == 1;
    if !active {
        return;
    }
    let is_signed = (flags >> 0) & 1 == 1;
    let invert = (flags >> 1) & 1 == 1;
    let lt = if is_signed {
        ((lhs as i64) < (rhs as i64)) as u64
    } else {
        (lhs < rhs) as u64
    };
    let out_val = lt ^ (invert as u64);
    let sub = lhs.wrapping_sub(rhs);

    out[base + 0] = lhs & 0xFFFF_FFFF;
    out[base + 1] = (lhs >> 32) & 0xFFFF;
    out[base + 2] = (lhs >> 48) & 0xFFFF;
    out[base + 3] = rhs & 0xFFFF_FFFF;
    out[base + 4] = (rhs >> 32) & 0xFFFF;
    out[base + 5] = (rhs >> 48) & 0xFFFF;
    out[base + 6] = is_signed as u64;
    out[base + 7] = lt;
    out[base + 8] = sub & 0xFFFF;
    out[base + 9] = (sub >> 16) & 0xFFFF;
    out[base + 10] = (sub >> 32) & 0xFFFF;
    out[base + 11] = (sub >> 48) & 0xFFFF;
    out[base + 12] = (lhs >> 63) & 1;
    out[base + 13] = (rhs >> 63) & 1;
    out[base + 14] = invert as u64;
    out[base + 15] = out_val;
    out[base + 16] = mu;
}

fn run_parity(num_rows: usize, num_active: usize, seed: u64) {
    assert!(num_active <= num_rows);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut lhs_values = vec![0u64; num_rows];
    let mut rhs_values = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];
    let mut multiplicities = vec![0u64; num_rows];
    for i in 0..num_active {
        lhs_values[i] = rng.r#gen::<u64>();
        rhs_values[i] = rng.r#gen::<u64>();
        let is_signed = (rng.r#gen::<u32>() & 1) as u64;
        let invert = (rng.r#gen::<u32>() & 1) as u64;
        flags[i] = is_signed | (invert << 1) | (1 << 2);
        multiplicities[i] = (rng.r#gen::<u32>() % 256) as u64 + 1;
    }
    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_row(
            &mut cpu,
            row * NUM_COLS,
            lhs_values[row],
            rhs_values[row],
            flags[row],
            multiplicities[row],
        );
    }
    let gpu = generate_lt_trace_dev(
        num_rows,
        &lhs_values,
        &rhs_values,
        &flags,
        &multiplicities,
        NUM_COLS,
    )
    .unwrap();
    assert_eq!(cpu, gpu);
}

#[test]
fn lt_trace_parity_small() {
    run_parity(4, 3, 1);
    run_parity(16, 9, 2);
}

#[test]
fn lt_trace_parity_realistic() {
    run_parity(1 << 14, 12_000, 100);
    run_parity(1 << 18, 200_000, 101);
}

#[test]
fn lt_trace_parity_signed_edges() {
    // Exercise signed/unsigned edges around 0 and i64::MIN.
    let cases: &[(u64, u64)] = &[
        (0, 1),
        (1, 0),
        (i64::MIN as u64, 0),
        (0, i64::MIN as u64),
        (i64::MAX as u64, i64::MIN as u64),
        (u64::MAX, 0),
        (u64::MAX, u64::MAX - 1),
        (1u64 << 63, (1u64 << 63) - 1),
    ];
    let num_rows = (cases.len() * 4).next_power_of_two();
    let mut lhs_values = vec![0u64; num_rows];
    let mut rhs_values = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];
    let mut multiplicities = vec![0u64; num_rows];
    let mut i = 0;
    for &(a, b) in cases {
        for (is_signed, invert) in [(0u64, 0u64), (0, 1), (1, 0), (1, 1)] {
            lhs_values[i] = a;
            rhs_values[i] = b;
            flags[i] = is_signed | (invert << 1) | (1 << 2);
            multiplicities[i] = 1;
            i += 1;
        }
    }
    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_row(
            &mut cpu,
            row * NUM_COLS,
            lhs_values[row],
            rhs_values[row],
            flags[row],
            multiplicities[row],
        );
    }
    let gpu = generate_lt_trace_dev(
        num_rows,
        &lhs_values,
        &rhs_values,
        &flags,
        &multiplicities,
        NUM_COLS,
    )
    .unwrap();
    assert_eq!(cpu, gpu);
}
