//! Parity: GPU `generate_shift_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/shift.rs::generate_shift_trace`'s row layout
//! and `compute_aux` semantics.

use math_cuda::shift_trace::generate_shift_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 29;

fn hwsl(h: u16, z: u8) -> u16 {
    if z == 0 { h } else { ((h as u32) << z as u32) as u16 }
}
fn hwslc(h: u16, z: u8) -> u16 {
    if z == 0 { 0 } else { h >> (16 - z as u16) }
}

#[allow(clippy::too_many_arguments)]
fn cpu_shift_row(out: &mut [u64], base: usize, v: u64, sa: u64, flags: u64) {
    let active = (flags >> 3) & 1 == 1;
    if !active {
        // Padding row: only ZBS=1, rest 0 (out is already zeroed).
        out[base + 12] = 1;
        return;
    }
    let in_h = [
        (v & 0xFFFF) as u16,
        ((v >> 16) & 0xFFFF) as u16,
        ((v >> 32) & 0xFFFF) as u16,
        ((v >> 48) & 0xFFFF) as u16,
    ];
    let direction = (flags >> 0) & 1 == 1;
    let signed = (flags >> 1) & 1 == 1;
    let word_instr = (flags >> 2) & 1 == 1;
    let left = !direction;
    let right = direction;

    let shift = (sa & 0xFF) as u8;
    let is_negative = signed && (in_h[3] >> 15) & 1 == 1;
    let extension: u16 = if is_negative { 0xFFFF } else { 0 };

    let bit_shift = if left {
        shift & 15
    } else {
        (256u16.wrapping_sub(shift as u16) & 15) as u8
    };
    let zbs = bit_shift == 0;

    let mut x = [0u16; 5];
    let mut y = [0u16; 4];
    if zbs {
        for i in 0..4 {
            if left {
                x[i] = in_h[i];
            } else {
                y[i] = in_h[i];
            }
        }
        x[4] = 0;
    } else {
        for i in 0..4 {
            x[i] = hwsl(in_h[i], bit_shift);
            y[i] = hwslc(in_h[i], bit_shift);
        }
        x[4] = hwsl(extension, bit_shift);
    }

    let limb_idx = if word_instr {
        ((shift >> 4) & 1) as usize
    } else {
        ((shift >> 4) & 3) as usize
    };
    let mut ls = [false; 4];
    ls[limb_idx] = true;

    let intra_left = |i: usize, x: &[u16; 5], y: &[u16; 4]| -> u16 {
        if i == 0 { x[0] } else { x[i].wrapping_add(y[i - 1]) }
    };
    let intra_right = |i: usize, x: &[u16; 5], y: &[u16; 4]| -> u16 {
        y[i].wrapping_add(x[i + 1])
    };

    let mut shifted = [0u16; 4];
    for i in 0..4 {
        let mut val: u16 = 0;
        if left {
            for j in 0..=i {
                if ls[j] {
                    val = val.wrapping_add(intra_left(i - j, &x, &y));
                }
            }
        }
        if right {
            for j in 0..=(3 - i) {
                if ls[j] {
                    val = val.wrapping_add(intra_right(i + j, &x, &y));
                }
            }
            for j in (4 - i)..4 {
                if ls[j] {
                    val = val.wrapping_add(extension);
                }
            }
        }
        shifted[i] = val;
    }

    let out_0 = shifted[0] as u32 | (shifted[1] as u32) << 16;
    let out_1 = shifted[2] as u32 | (shifted[3] as u32) << 16;

    out[base + 0] = in_h[0] as u64;
    out[base + 1] = in_h[1] as u64;
    out[base + 2] = in_h[2] as u64;
    out[base + 3] = in_h[3] as u64;
    out[base + 4] = shift as u64;
    out[base + 5] = direction as u64;
    out[base + 6] = signed as u64;
    out[base + 7] = word_instr as u64;
    out[base + 8] = out_0 as u64;
    out[base + 9] = out_1 as u64;
    out[base + 10] = is_negative as u64;
    out[base + 11] = bit_shift as u64;
    out[base + 12] = zbs as u64;
    for i in 0..5 {
        out[base + 13 + i] = x[i] as u64;
    }
    for i in 0..4 {
        out[base + 18 + i] = y[i] as u64;
    }
    for i in 0..3 {
        out[base + 22 + i] = ls[i] as u64;
    }
    out[base + 25] = 1;
    out[base + 26] = (sa >> 8) & 0xFF;
    out[base + 27] = (sa >> 16) & 0xFFFF;
    out[base + 28] = sa >> 32;
}

fn run_parity(num_rows: usize, num_active: usize, seed: u64) {
    assert!(num_active <= num_rows);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut in_values = vec![0u64; num_rows];
    let mut shift_amounts = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];
    for i in 0..num_active {
        in_values[i] = rng.r#gen::<u64>();
        // Cover the full byte/half/word ranges for shift_amount range checks.
        shift_amounts[i] = rng.r#gen::<u64>();
        let direction = (rng.r#gen::<u32>() & 1) as u64;
        let signed = (rng.r#gen::<u32>() & 1) as u64;
        let word_instr = (rng.r#gen::<u32>() & 1) as u64;
        flags[i] = direction | (signed << 1) | (word_instr << 2) | (1 << 3);
    }
    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_shift_row(
            &mut cpu,
            row * NUM_COLS,
            in_values[row],
            shift_amounts[row],
            flags[row],
        );
    }
    let gpu = generate_shift_trace_dev(num_rows, &in_values, &shift_amounts, &flags, NUM_COLS)
        .unwrap();
    assert_eq!(cpu.len(), gpu.len());
    for row in 0..num_rows {
        let b = row * NUM_COLS;
        let c = &cpu[b..b + NUM_COLS];
        let g = &gpu[b..b + NUM_COLS];
        if c != g {
            panic!(
                "shift trace mismatch at row {row} (active={}): cpu={c:?} gpu={g:?}",
                (flags[row] >> 3) & 1
            );
        }
    }
}

#[test]
fn shift_trace_parity_small() {
    run_parity(4, 3, 1);
    run_parity(8, 5, 2);
}

#[test]
fn shift_trace_parity_realistic() {
    run_parity(1 << 14, 12_000, 100);
    run_parity(1 << 16, 50_000, 101);
}

#[test]
fn shift_trace_parity_corner_shifts() {
    // Exercise specific edge shift amounts that hit zbs=true (shift mod 16 == 0)
    // and the limb_shift boundaries.
    let num_rows = 64usize;
    let mut in_values = vec![0u64; num_rows];
    let mut shift_amounts = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let edge_shifts = [
        0u64, 1, 7, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 79, 80, 127,
    ];
    for (i, &s) in edge_shifts.iter().enumerate() {
        let row = i;
        in_values[row] = rng.r#gen::<u64>();
        shift_amounts[row] = s;
        let direction = (i & 1) as u64;
        let signed = ((i >> 1) & 1) as u64;
        let word_instr = ((i >> 2) & 1) as u64;
        flags[row] = direction | (signed << 1) | (word_instr << 2) | (1 << 3);
    }
    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_shift_row(
            &mut cpu,
            row * NUM_COLS,
            in_values[row],
            shift_amounts[row],
            flags[row],
        );
    }
    let gpu = generate_shift_trace_dev(num_rows, &in_values, &shift_amounts, &flags, NUM_COLS)
        .unwrap();
    assert_eq!(cpu, gpu);
}
