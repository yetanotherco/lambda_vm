//! Parity: GPU `generate_cpu32_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/cpu32.rs::generate_cpu32_trace`.

use math_cuda::cpu32_trace::generate_cpu32_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 38;
const HI_FILL: u64 = 0xFFFF_FFFF;
const SIGNED_BIT: u32 = 5;

#[allow(clippy::too_many_arguments)]
fn cpu_row(
    out: &mut [u64],
    base: usize,
    ts: u64,
    pc: u64,
    rv1: u64,
    rv2: u64,
    imm: u64,
    res: u64,
    f: u64,
) {
    let active = (f >> 46) & 1 == 1;
    if !active {
        return;
    }
    let alu_flags = (f >> 32) & 0xFF;
    let is_signed = (alu_flags >> SIGNED_BIT) & 1 == 1;
    let rv1_sign = is_signed && (rv1 >> 31) & 1 == 1;
    let rv2_sign = is_signed && (rv2 >> 31) & 1 == 1;
    let res_sign = (res >> 31) & 1 == 1;

    let arg1_hi = if rv1_sign { HI_FILL } else { 0 };
    let arg1 = (rv1 & 0xFFFF_FFFF) | (arg1_hi << 32);

    let arg2_lo = (rv2 & 0xFFFF_FFFF) + (imm & 0xFFFF_FFFF);
    let arg2_hi_raw = if rv2_sign { HI_FILL } else { 0 } + (imm >> 32);
    let arg2 = (arg2_lo & 0xFFFF_FFFF) | ((arg2_hi_raw & 0xFFFF_FFFF) << 32);

    let rvd_hi = if res_sign { HI_FILL } else { 0 };
    let rvd = (res & 0xFFFF_FFFF) | (rvd_hi << 32);

    out[base + 0] = ts & 0xFFFF_FFFF;
    out[base + 1] = ts >> 32;
    out[base + 2] = pc & 0xFFFF_FFFF;
    out[base + 3] = pc >> 32;
    out[base + 4] = (f >> 0) & 0xFF;
    out[base + 5] = (f >> 40) & 1;
    out[base + 6] = rv1 & 0xFFFF;
    out[base + 7] = (rv1 >> 16) & 0xFFFF;
    out[base + 8] = rv1 >> 32;
    out[base + 9] = rv1_sign as u64;
    out[base + 10] = arg1 & 0xFFFF_FFFF;
    out[base + 11] = arg1 >> 32;
    out[base + 12] = (f >> 8) & 0xFF;
    out[base + 13] = (f >> 41) & 1;
    out[base + 14] = rv2 & 0xFFFF;
    out[base + 15] = (rv2 >> 16) & 0xFFFF;
    out[base + 16] = rv2 >> 32;
    out[base + 17] = rv2_sign as u64;
    out[base + 18] = imm & 0xFFFF_FFFF;
    out[base + 19] = imm >> 32;
    out[base + 20] = arg2 & 0xFFFF_FFFF;
    out[base + 21] = arg2 >> 32;
    out[base + 22] = res & 0xFFFF;
    out[base + 23] = (res >> 16) & 0xFFFF;
    out[base + 24] = (res >> 32) & 0xFFFF;
    out[base + 25] = (res >> 48) & 0xFFFF;
    out[base + 26] = res_sign as u64;
    out[base + 27] = (f >> 16) & 0xFF;
    out[base + 28] = (f >> 42) & 1;
    out[base + 29] = rvd & 0xFFFF_FFFF;
    out[base + 30] = rvd >> 32;
    out[base + 31] = (f >> 43) & 1;
    out[base + 32] = alu_flags;
    out[base + 33] = (f >> 44) & 1;
    out[base + 34] = (f >> 45) & 1;
    out[base + 35] = (f >> 24) & 0xFF;
    out[base + 36] = is_signed as u64;
    out[base + 37] = 1;
}

fn run_parity(num_rows: usize, num_active: usize, seed: u64) {
    assert!(num_active <= num_rows);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut timestamps = vec![0u64; num_rows];
    let mut pcs = vec![0u64; num_rows];
    let mut rv1s = vec![0u64; num_rows];
    let mut rv2s = vec![0u64; num_rows];
    let mut imms = vec![0u64; num_rows];
    let mut ress = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];

    for i in 0..num_active {
        timestamps[i] = rng.r#gen::<u64>();
        pcs[i] = rng.r#gen::<u64>();
        rv1s[i] = rng.r#gen::<u64>();
        // The decoding assumption is "exactly one of rv2/imm is non-zero per word".
        // Pick uniformly which one feeds arg2, so the lo/hi sums stay below 2^33.
        let (rv2, imm) = if rng.r#gen::<u32>() & 1 == 0 {
            (rng.r#gen::<u64>(), 0u64)
        } else {
            (0u64, rng.r#gen::<u64>())
        };
        rv2s[i] = rv2;
        imms[i] = imm;
        ress[i] = rng.r#gen::<u64>();
        let rs1 = (rng.r#gen::<u32>() & 0xFF) as u64;
        let rs2 = (rng.r#gen::<u32>() & 0xFF) as u64;
        let rd = (rng.r#gen::<u32>() & 0xFF) as u64;
        let half_il = (1 + (rng.r#gen::<u32>() & 1)) as u64;
        let alu_flags = (rng.r#gen::<u32>() & 0xFF) as u64;
        let rr1 = (rng.r#gen::<u32>() & 1) as u64;
        let rr2 = (rng.r#gen::<u32>() & 1) as u64;
        let wr = (rng.r#gen::<u32>() & 1) as u64;
        let alu = (rng.r#gen::<u32>() & 1) as u64;
        let add = (rng.r#gen::<u32>() & 1) as u64;
        let sub = (rng.r#gen::<u32>() & 1) as u64;
        let mut f: u64 = 0;
        f |= rs1;
        f |= rs2 << 8;
        f |= rd << 16;
        f |= half_il << 24;
        f |= alu_flags << 32;
        f |= rr1 << 40;
        f |= rr2 << 41;
        f |= wr << 42;
        f |= alu << 43;
        f |= add << 44;
        f |= sub << 45;
        f |= 1 << 46;
        flags[i] = f;
    }

    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_row(
            &mut cpu,
            row * NUM_COLS,
            timestamps[row],
            pcs[row],
            rv1s[row],
            rv2s[row],
            imms[row],
            ress[row],
            flags[row],
        );
    }
    let gpu = generate_cpu32_trace_dev(
        num_rows, &timestamps, &pcs, &rv1s, &rv2s, &imms, &ress, &flags, NUM_COLS,
    )
    .unwrap();
    assert_eq!(cpu, gpu);
}

#[test]
fn cpu32_trace_parity_small() {
    run_parity(4, 3, 1);
    run_parity(16, 9, 2);
}

#[test]
fn cpu32_trace_parity_realistic() {
    run_parity(1 << 14, 12_000, 100);
    run_parity(1 << 16, 50_000, 101);
}

#[test]
fn cpu32_trace_parity_sign_edges() {
    // Hand-picked rows exercising signed/unsigned, MSB-set rv1/rv2/res,
    // and imm-fed arg2.
    let cases: &[(u64, u64, u64, u64, u64, u64)] = &[
        // (rv1, rv2, imm, res, alu_flags, alu_flags-signed-bit-set)
        (0, 0, 0, 0, 0, 0),
        (0x8000_0000, 0, 0, 0, 1 << SIGNED_BIT, 1 << SIGNED_BIT),
        (0, 0x8000_0000, 0, 0, 1 << SIGNED_BIT, 1 << SIGNED_BIT),
        (0, 0, 0x8000_0000_8000_0000, 0, 0, 0),
        (0, 0, 0, 0x8000_0000, 0, 0),
        (0xFFFF_FFFF, 0, 0, 0xFFFF_FFFF, 0, 0),
        (0xFFFF_FFFF, 0, 0, 0xFFFF_FFFF, 1 << SIGNED_BIT, 1 << SIGNED_BIT),
    ];
    let num_rows = cases.len().next_power_of_two().max(4);
    let timestamps = vec![0u64; num_rows];
    let pcs = vec![0u64; num_rows];
    let mut rv1s = vec![0u64; num_rows];
    let mut rv2s = vec![0u64; num_rows];
    let mut imms = vec![0u64; num_rows];
    let mut ress = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];
    for (i, &(rv1, rv2, imm, res, alu_flags, _)) in cases.iter().enumerate() {
        rv1s[i] = rv1;
        rv2s[i] = rv2;
        imms[i] = imm;
        ress[i] = res;
        flags[i] = (alu_flags << 32) | (1u64 << 46);
    }
    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_row(
            &mut cpu,
            row * NUM_COLS,
            timestamps[row],
            pcs[row],
            rv1s[row],
            rv2s[row],
            imms[row],
            ress[row],
            flags[row],
        );
    }
    let gpu = generate_cpu32_trace_dev(
        num_rows, &timestamps, &pcs, &rv1s, &rv2s, &imms, &ress, &flags, NUM_COLS,
    )
    .unwrap();
    assert_eq!(cpu, gpu);
}
