//! Parity: GPU `generate_cpu_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/cpu.rs::generate_cpu_trace`'s row layout.
//! Caller does all word/padding masking upfront, so the kernel and the CPU
//! reference are pure column splits.

use math_cuda::cpu_trace::generate_cpu_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 38;

#[allow(clippy::too_many_arguments)]
fn cpu_row(
    out: &mut [u64],
    base: usize,
    ts: u64,
    pc: u64,
    npc: u64,
    imm: u64,
    rvd: u64,
    rv1: u64,
    rv2: u64,
    arg2: u64,
    res: u64,
    f: u64,
) {
    out[base + 0] = ts;
    out[base + 1] = pc & 0xFFFF_FFFF;
    out[base + 2] = pc >> 32;
    out[base + 3] = (f >> 0) & 0xFF;
    out[base + 4] = (f >> 8) & 0xFF;
    out[base + 5] = (f >> 16) & 0xFF;
    out[base + 6] = (f >> 55) & 1;
    out[base + 7] = (f >> 56) & 1;
    out[base + 8] = (f >> 57) & 1;
    out[base + 9] = imm & 0xFFFF_FFFF;
    out[base + 10] = imm >> 32;
    out[base + 11] = (f >> 24) & 0xFF;
    out[base + 12] = (f >> 48) & 1;
    out[base + 13] = (f >> 49) & 1;
    out[base + 14] = (f >> 32) & 0xFF;
    out[base + 15] = (f >> 50) & 1;
    out[base + 16] = (f >> 51) & 1;
    out[base + 17] = (f >> 52) & 1;
    out[base + 18] = (f >> 40) & 0xFF;
    out[base + 19] = (f >> 53) & 1;
    out[base + 20] = (f >> 54) & 1;
    out[base + 21] = npc & 0xFFFF_FFFF;
    out[base + 22] = npc >> 32;
    out[base + 23] = rvd & 0xFFFF_FFFF;
    out[base + 24] = rvd >> 32;
    out[base + 25] = (f >> 60) & 1;
    out[base + 26] = (f >> 59) & 1;
    out[base + 27] = rv1 & 0xFFFF_FFFF;
    out[base + 28] = rv1 >> 32;
    out[base + 29] = rv2 & 0xFFFF_FFFF;
    out[base + 30] = rv2 >> 32;
    out[base + 31] = arg2 & 0xFFFF_FFFF;
    out[base + 32] = arg2 >> 32;
    out[base + 33] = (res >> 0) & 0xFFFF;
    out[base + 34] = (res >> 16) & 0xFFFF;
    out[base + 35] = (res >> 32) & 0xFFFF;
    out[base + 36] = (res >> 48) & 0xFFFF;
    out[base + 37] = (f >> 58) & 1;
}

fn run_parity(num_rows: usize, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut timestamps = vec![0u64; num_rows];
    let mut pcs = vec![0u64; num_rows];
    let mut next_pcs = vec![0u64; num_rows];
    let mut imms = vec![0u64; num_rows];
    let mut rvds = vec![0u64; num_rows];
    let mut rv1s = vec![0u64; num_rows];
    let mut rv2s = vec![0u64; num_rows];
    let mut arg2s = vec![0u64; num_rows];
    let mut ress = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];

    for i in 0..num_rows {
        timestamps[i] = rng.r#gen::<u64>();
        pcs[i] = rng.r#gen::<u64>();
        next_pcs[i] = rng.r#gen::<u64>();
        imms[i] = rng.r#gen::<u64>();
        rvds[i] = rng.r#gen::<u64>();
        rv1s[i] = rng.r#gen::<u64>();
        rv2s[i] = rng.r#gen::<u64>();
        arg2s[i] = rng.r#gen::<u64>();
        ress[i] = rng.r#gen::<u64>();
        flags[i] = rng.r#gen::<u64>();
    }

    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_row(
            &mut cpu,
            row * NUM_COLS,
            timestamps[row],
            pcs[row],
            next_pcs[row],
            imms[row],
            rvds[row],
            rv1s[row],
            rv2s[row],
            arg2s[row],
            ress[row],
            flags[row],
        );
    }

    let gpu = generate_cpu_trace_dev(
        num_rows,
        &timestamps,
        &pcs,
        &next_pcs,
        &imms,
        &rvds,
        &rv1s,
        &rv2s,
        &arg2s,
        &ress,
        &flags,
        NUM_COLS,
    )
    .unwrap();
    assert_eq!(cpu, gpu);
}

#[test]
fn cpu_trace_parity_small() {
    run_parity(4, 1);
    run_parity(16, 2);
}

#[test]
fn cpu_trace_parity_realistic() {
    run_parity(1 << 16, 100);
    run_parity(1 << 18, 101);
}
