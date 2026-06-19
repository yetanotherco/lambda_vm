//! Parity: GPU `generate_mul_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/mul.rs::generate_mul_trace`'s row layout.

use math_cuda::mul_trace::generate_mul_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 26;
const SIGN_FILL: u64 = 0xFFFF;

#[allow(clippy::too_many_arguments)]
fn cpu_row(out: &mut [u64], base: usize, lhs: u64, rhs: u64, flags: u64, mu_lo: u64, mu_hi: u64) {
    let active = (flags >> 2) & 1 == 1;
    if !active {
        return;
    }
    let lhs_signed = (flags >> 0) & 1 == 1;
    let rhs_signed = (flags >> 1) & 1 == 1;
    let lhs_is_neg = lhs_signed && (lhs as i64) < 0;
    let rhs_is_neg = rhs_signed && (rhs as i64) < 0;

    let a: i128 = if lhs_signed {
        lhs as i64 as i128
    } else {
        lhs as u128 as i128
    };
    let b: i128 = if rhs_signed {
        rhs as i64 as i128
    } else {
        rhs as u128 as i128
    };
    let product = a.wrapping_mul(b);
    let lo = product as u64;
    let hi = (product >> 64) as u64;

    let mut lhs_ext = [0u64; 8];
    let mut rhs_ext = [0u64; 8];
    lhs_ext[0] = lhs & 0xFFFF;
    lhs_ext[1] = (lhs >> 16) & 0xFFFF;
    lhs_ext[2] = (lhs >> 32) & 0xFFFF;
    lhs_ext[3] = (lhs >> 48) & 0xFFFF;
    rhs_ext[0] = rhs & 0xFFFF;
    rhs_ext[1] = (rhs >> 16) & 0xFFFF;
    rhs_ext[2] = (rhs >> 32) & 0xFFFF;
    rhs_ext[3] = (rhs >> 48) & 0xFFFF;
    let lhs_fill = if lhs_is_neg { SIGN_FILL } else { 0 };
    let rhs_fill = if rhs_is_neg { SIGN_FILL } else { 0 };
    for j in 4..8 {
        lhs_ext[j] = lhs_fill;
        rhs_ext[j] = rhs_fill;
    }

    let mut raw = [0u64; 4];
    for i in 0..4 {
        let mut sum: u128 = 0;
        for k in 0..=1 {
            let idx = 2 * i + k;
            if idx < 8 {
                let mut inner: u128 = 0;
                for j in 0..=idx {
                    if j < 8 && (idx - j) < 8 {
                        inner += (lhs_ext[j] as u128) * (rhs_ext[idx - j] as u128);
                    }
                }
                sum += inner << (16 * k);
            }
        }
        raw[i] = sum as u64;
    }

    out[base + 0] = lhs & 0xFFFF;
    out[base + 1] = (lhs >> 16) & 0xFFFF;
    out[base + 2] = (lhs >> 32) & 0xFFFF;
    out[base + 3] = (lhs >> 48) & 0xFFFF;
    out[base + 4] = lhs_signed as u64;
    out[base + 5] = rhs & 0xFFFF;
    out[base + 6] = (rhs >> 16) & 0xFFFF;
    out[base + 7] = (rhs >> 32) & 0xFFFF;
    out[base + 8] = (rhs >> 48) & 0xFFFF;
    out[base + 9] = rhs_signed as u64;
    out[base + 10] = lo & 0xFFFF;
    out[base + 11] = (lo >> 16) & 0xFFFF;
    out[base + 12] = (lo >> 32) & 0xFFFF;
    out[base + 13] = (lo >> 48) & 0xFFFF;
    out[base + 14] = hi & 0xFFFF;
    out[base + 15] = (hi >> 16) & 0xFFFF;
    out[base + 16] = (hi >> 32) & 0xFFFF;
    out[base + 17] = (hi >> 48) & 0xFFFF;
    out[base + 18] = lhs_is_neg as u64;
    out[base + 19] = rhs_is_neg as u64;
    out[base + 20] = raw[0];
    out[base + 21] = raw[1];
    out[base + 22] = raw[2];
    out[base + 23] = raw[3];
    out[base + 24] = mu_lo;
    out[base + 25] = mu_hi;
}

fn run_parity(num_rows: usize, num_active: usize, seed: u64) {
    assert!(num_active <= num_rows);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut lhs_values = vec![0u64; num_rows];
    let mut rhs_values = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];
    let mut mu_lo = vec![0u64; num_rows];
    let mut mu_hi = vec![0u64; num_rows];
    for i in 0..num_active {
        lhs_values[i] = rng.r#gen::<u64>();
        rhs_values[i] = rng.r#gen::<u64>();
        let lhs_signed = (rng.r#gen::<u32>() & 1) as u64;
        let rhs_signed = (rng.r#gen::<u32>() & 1) as u64;
        flags[i] = lhs_signed | (rhs_signed << 1) | (1 << 2);
        mu_lo[i] = (rng.r#gen::<u32>() % 8) as u64;
        mu_hi[i] = (rng.r#gen::<u32>() % 8) as u64;
    }
    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_row(
            &mut cpu,
            row * NUM_COLS,
            lhs_values[row],
            rhs_values[row],
            flags[row],
            mu_lo[row],
            mu_hi[row],
        );
    }
    let gpu = generate_mul_trace_dev(
        num_rows,
        &lhs_values,
        &rhs_values,
        &flags,
        &mu_lo,
        &mu_hi,
        NUM_COLS,
    )
    .unwrap();
    assert_eq!(cpu, gpu);
}

#[test]
fn mul_trace_parity_small() {
    run_parity(4, 3, 1);
    run_parity(16, 9, 2);
}

#[test]
fn mul_trace_parity_realistic() {
    run_parity(1 << 14, 12_000, 100);
    run_parity(1 << 16, 50_000, 101);
}

#[test]
fn mul_trace_parity_signed_edges() {
    let cases: &[(u64, u64)] = &[
        (0, 0),
        (1, 1),
        (u64::MAX, u64::MAX),
        (i64::MIN as u64, i64::MIN as u64),
        (i64::MIN as u64, u64::MAX),
        (1, u64::MAX),
        ((-1i64) as u64, 2),
        (2, (-1i64) as u64),
    ];
    let num_rows = (cases.len() * 4).next_power_of_two();
    let mut lhs_values = vec![0u64; num_rows];
    let mut rhs_values = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];
    let mut mu_lo = vec![0u64; num_rows];
    let mut mu_hi = vec![0u64; num_rows];
    let mut i = 0;
    for &(a, b) in cases {
        for (ls, rs) in [(0u64, 0u64), (0, 1), (1, 0), (1, 1)] {
            lhs_values[i] = a;
            rhs_values[i] = b;
            flags[i] = ls | (rs << 1) | (1 << 2);
            mu_lo[i] = 1;
            mu_hi[i] = 1;
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
            mu_lo[row],
            mu_hi[row],
        );
    }
    let gpu = generate_mul_trace_dev(
        num_rows,
        &lhs_values,
        &rhs_values,
        &flags,
        &mu_lo,
        &mu_hi,
        NUM_COLS,
    )
    .unwrap();
    assert_eq!(cpu, gpu);
}
