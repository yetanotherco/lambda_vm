//! Parity: GPU `generate_branch_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/branch.rs::generate_branch_trace`.

use math_cuda::branch_trace::generate_branch_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 14;

fn cpu_row(out: &mut [u64], base: usize, pc: u64, off: u64, reg: u64, flags: u64, mu: u64) {
    let active = (flags >> 1) & 1 == 1;
    if !active {
        return;
    }
    let jalr = (flags >> 0) & 1 == 1;
    let base_v = if jalr { reg } else { pc };
    let unmasked = base_v.wrapping_add(off);
    let next_pc = unmasked & !1u64;

    out[base + 0] = pc & 0xFFFF_FFFF;
    out[base + 1] = pc >> 32;
    out[base + 2] = off & 0xFFFF_FFFF;
    out[base + 3] = off >> 32;
    out[base + 4] = reg & 0xFFFF_FFFF;
    out[base + 5] = reg >> 32;
    out[base + 6] = jalr as u64;
    out[base + 7] = (next_pc >> 16) & 0xFFFF;
    out[base + 8] = (next_pc >> 32) & 0xFFFF;
    out[base + 9] = (next_pc >> 48) & 0xFFFF;
    out[base + 10] = next_pc & 0xFF;
    out[base + 11] = (next_pc >> 8) & 0xFF;
    out[base + 12] = unmasked & 0xFF;
    out[base + 13] = mu;
}

fn run_parity(num_rows: usize, num_active: usize, seed: u64) {
    assert!(num_active <= num_rows);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut pcs = vec![0u64; num_rows];
    let mut offsets = vec![0u64; num_rows];
    let mut registers = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];
    let mut multiplicities = vec![0u64; num_rows];
    for i in 0..num_active {
        pcs[i] = rng.r#gen::<u64>();
        offsets[i] = rng.r#gen::<u64>();
        registers[i] = rng.r#gen::<u64>();
        let jalr = (rng.r#gen::<u32>() & 1) as u64;
        flags[i] = jalr | (1 << 1);
        multiplicities[i] = (rng.r#gen::<u32>() % 256) as u64 + 1;
    }
    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_row(
            &mut cpu,
            row * NUM_COLS,
            pcs[row],
            offsets[row],
            registers[row],
            flags[row],
            multiplicities[row],
        );
    }
    let gpu = generate_branch_trace_dev(
        num_rows,
        &pcs,
        &offsets,
        &registers,
        &flags,
        &multiplicities,
        NUM_COLS,
    )
    .unwrap();
    assert_eq!(cpu, gpu);
}

#[test]
fn branch_trace_parity_small() {
    run_parity(4, 3, 1);
    run_parity(16, 9, 2);
}

#[test]
fn branch_trace_parity_realistic() {
    run_parity(1 << 14, 12_000, 100);
    run_parity(1 << 16, 50_000, 101);
}

#[test]
fn branch_trace_parity_wrapping_edges() {
    // Exercise wraparound (base + offset overflows 64 bits) and odd unmasked LSBs.
    let cases: &[(u64, u64, u64, bool)] = &[
        (0, 0, 0, false),
        (0, 1, 0, false),                       // odd unmasked
        (0, u64::MAX, 0, false),                // wraps; LSB clears
        (u64::MAX, 1, 0, false),                // wraps to 0
        (0x100, 0x80, 0, false),
        (0, 0x123, 0xFFFF_FFFF_FFFF_FFFE, true), // jalr path
        (0xDEAD_BEEF, 0xCAFE, 0x1000, true),
    ];
    let num_rows = cases.len().next_power_of_two().max(4);
    let mut pcs = vec![0u64; num_rows];
    let mut offsets = vec![0u64; num_rows];
    let mut registers = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];
    let mut multiplicities = vec![0u64; num_rows];
    for (i, &(pc, off, reg, jalr)) in cases.iter().enumerate() {
        pcs[i] = pc;
        offsets[i] = off;
        registers[i] = reg;
        flags[i] = (jalr as u64) | (1 << 1);
        multiplicities[i] = 1;
    }
    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_row(
            &mut cpu,
            row * NUM_COLS,
            pcs[row],
            offsets[row],
            registers[row],
            flags[row],
            multiplicities[row],
        );
    }
    let gpu = generate_branch_trace_dev(
        num_rows,
        &pcs,
        &offsets,
        &registers,
        &flags,
        &multiplicities,
        NUM_COLS,
    )
    .unwrap();
    assert_eq!(cpu, gpu);
}
