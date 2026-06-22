//! Parity: GPU `generate_dvrm_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/dvrm.rs::generate_dvrm_trace`.

use math_cuda::dvrm_trace::generate_dvrm_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 34;

fn abs_value(v: u64, is_negative: bool) -> u64 {
    if is_negative {
        (v as i64).unsigned_abs()
    } else {
        v
    }
}

fn cpu_row(out: &mut [u64], base: usize, n: u64, d: u64, flags: u64, mu_q: u64, mu_r: u64) {
    let active = (flags >> 1) & 1 == 1;
    if !active {
        return;
    }
    let is_signed = (flags >> 0) & 1 == 1;
    let div_by_zero = d == 0;
    let overflow = is_signed && n == (i64::MIN as u64) && d == u64::MAX;
    let (q, r) = if div_by_zero {
        (u64::MAX, n)
    } else if overflow {
        (n, 0u64)
    } else if is_signed {
        (
            (n as i64).wrapping_div(d as i64) as u64,
            (n as i64).wrapping_rem(d as i64) as u64,
        )
    } else {
        (n / d, n % d)
    };
    let n_sub_r = n.wrapping_sub(r);
    let sign_n = is_signed && (n >> 63) & 1 == 1;
    let sign_d = is_signed && (d >> 63) & 1 == 1;
    let sign_r = is_signed && (r >> 63) & 1 == 1;
    let sign_q = is_signed && !overflow;
    let sign_n_sub_r = is_signed && (n_sub_r >> 63) & 1 == 1;
    let abs_r = abs_value(r, sign_r);
    let abs_d = abs_value(d, sign_d);

    out[base + 0] = n & 0xFFFF;
    out[base + 1] = (n >> 16) & 0xFFFF;
    out[base + 2] = (n >> 32) & 0xFFFF;
    out[base + 3] = (n >> 48) & 0xFFFF;
    out[base + 4] = d & 0xFFFF;
    out[base + 5] = (d >> 16) & 0xFFFF;
    out[base + 6] = (d >> 32) & 0xFFFF;
    out[base + 7] = (d >> 48) & 0xFFFF;
    out[base + 8] = is_signed as u64;
    out[base + 9] = q & 0xFFFF;
    out[base + 10] = (q >> 16) & 0xFFFF;
    out[base + 11] = (q >> 32) & 0xFFFF;
    out[base + 12] = (q >> 48) & 0xFFFF;
    out[base + 13] = r & 0xFFFF;
    out[base + 14] = (r >> 16) & 0xFFFF;
    out[base + 15] = (r >> 32) & 0xFFFF;
    out[base + 16] = (r >> 48) & 0xFFFF;
    out[base + 17] = div_by_zero as u64;
    out[base + 18] = overflow as u64;
    out[base + 19] = abs_r & 0xFFFF_FFFF;
    out[base + 20] = abs_r >> 32;
    out[base + 21] = abs_d & 0xFFFF_FFFF;
    out[base + 22] = abs_d >> 32;
    out[base + 23] = n_sub_r & 0xFFFF;
    out[base + 24] = (n_sub_r >> 16) & 0xFFFF;
    out[base + 25] = (n_sub_r >> 32) & 0xFFFF;
    out[base + 26] = (n_sub_r >> 48) & 0xFFFF;
    out[base + 27] = sign_n_sub_r as u64;
    out[base + 28] = sign_n as u64;
    out[base + 29] = sign_d as u64;
    out[base + 30] = sign_q as u64;
    out[base + 31] = sign_r as u64;
    out[base + 32] = mu_q;
    out[base + 33] = mu_r;
}

fn run_parity(num_rows: usize, num_active: usize, seed: u64) {
    assert!(num_active <= num_rows);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut ns = vec![0u64; num_rows];
    let mut ds = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];
    let mut mu_qs = vec![0u64; num_rows];
    let mut mu_rs = vec![0u64; num_rows];
    for i in 0..num_active {
        ns[i] = rng.r#gen::<u64>();
        // Avoid d=0 here (covered explicitly in edges test) — keeps the random
        // run exercising the common divider path.
        let d = loop {
            let v = rng.r#gen::<u64>();
            if v != 0 {
                break v;
            }
        };
        ds[i] = d;
        let is_signed = (rng.r#gen::<u32>() & 1) as u64;
        flags[i] = is_signed | (1 << 1);
        mu_qs[i] = (rng.r#gen::<u32>() % 8) as u64;
        mu_rs[i] = (rng.r#gen::<u32>() % 8) as u64;
    }
    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_row(
            &mut cpu,
            row * NUM_COLS,
            ns[row],
            ds[row],
            flags[row],
            mu_qs[row],
            mu_rs[row],
        );
    }
    let gpu = generate_dvrm_trace_dev(num_rows, &ns, &ds, &flags, &mu_qs, &mu_rs, NUM_COLS)
        .unwrap();
    assert_eq!(cpu, gpu);
}

#[test]
fn dvrm_trace_parity_small() {
    run_parity(4, 3, 1);
    run_parity(16, 9, 2);
}

#[test]
fn dvrm_trace_parity_realistic() {
    run_parity(1 << 14, 12_000, 100);
    run_parity(1 << 16, 50_000, 101);
}

#[test]
fn dvrm_trace_parity_edges() {
    // Cover div_by_zero, signed overflow (i64::MIN / -1), -1/1, and standard
    // signed mixes.
    let cases: &[(u64, u64)] = &[
        (0, 0),                       // 0/0 div-by-zero
        (5, 0),                       // div-by-zero w/ nonzero n
        (i64::MIN as u64, u64::MAX),  // signed overflow
        (i64::MIN as u64, 1),         // signed MIN / 1
        ((-1i64) as u64, 1),
        (1, (-1i64) as u64),
        (10, 3),
        (10, (-3i64) as u64),
        ((-10i64) as u64, 3),
        ((-10i64) as u64, (-3i64) as u64),
        (u64::MAX, u64::MAX),
    ];
    // Exercise each case both signed and unsigned.
    let num_rows = (cases.len() * 2).next_power_of_two().max(4);
    let mut ns = vec![0u64; num_rows];
    let mut ds = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];
    let mut mu_qs = vec![0u64; num_rows];
    let mut mu_rs = vec![0u64; num_rows];
    let mut i = 0;
    for &(n, d) in cases {
        for is_signed in [0u64, 1u64] {
            ns[i] = n;
            ds[i] = d;
            flags[i] = is_signed | (1 << 1);
            mu_qs[i] = 1;
            mu_rs[i] = 1;
            i += 1;
        }
    }
    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_row(
            &mut cpu,
            row * NUM_COLS,
            ns[row],
            ds[row],
            flags[row],
            mu_qs[row],
            mu_rs[row],
        );
    }
    let gpu = generate_dvrm_trace_dev(num_rows, &ns, &ds, &flags, &mu_qs, &mu_rs, NUM_COLS)
        .unwrap();
    assert_eq!(cpu, gpu);
}
