//! Parity: GPU `generate_ecsm_trace_dev` vs an inlined CPU reference
//! that performs the same column splay the prover-side helper does.
//! Uses synthetic data (not a real ECSM witness) — the kernel is a pure
//! layout splay, so any consistent bytes/halfwords/carry blobs validate it.

use math_cuda::ecsm_trace::generate_ecsm_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 427;

#[allow(clippy::too_many_arguments)]
fn cpu_row(
    out: &mut [u64],
    base: usize,
    ts: u64,
    a_xg: u64,
    a_k: u64,
    a_xr: u64,
    fk: u64,
    bb: &[u64], // 257
    hw: &[u64], // 32
    cb: &[u64], // 128
) {
    out[base + 0] = ts & 0xFFFF_FFFF;
    out[base + 1] = ts >> 32;
    out[base + 2] = a_xg & 0xFFFF_FFFF;
    out[base + 3] = a_xg >> 32;
    out[base + 4] = a_k & 0xFFFF_FFFF;
    out[base + 5] = a_k >> 32;
    out[base + 6] = a_xr & 0xFFFF_FFFF;
    out[base + 7] = a_xr >> 32;
    for i in 0..32 {
        out[base + 8 + i] = bb[i];
        out[base + 40 + i] = bb[32 + i];
        out[base + 72 + i] = bb[64 + i];
    }
    out[base + 104] = fk & 0xFF;
    for i in 0..32 {
        out[base + 105 + i] = bb[96 + i];
        out[base + 137 + i] = bb[128 + i];
        out[base + 169 + i] = bb[160 + i];
        out[base + 201 + i] = bb[192 + i];
    }
    for i in 0..64 {
        out[base + 233 + i] = cb[i];
    }
    for i in 0..33 {
        out[base + 297 + i] = bb[224 + i];
    }
    for i in 0..64 {
        out[base + 330 + i] = cb[64 + i];
    }
    for i in 0..16 {
        out[base + 394 + i] = hw[i];
        out[base + 410 + i] = hw[16 + i];
    }
    out[base + 426] = (fk >> 8) & 1;
}

fn run_parity(num_rows: usize, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut timestamps = vec![0u64; num_rows];
    let mut addr_xgs = vec![0u64; num_rows];
    let mut addr_ks = vec![0u64; num_rows];
    let mut addr_xrs = vec![0u64; num_rows];
    let mut flags_len_k = vec![0u64; num_rows];
    let mut byte_blob = vec![0u64; num_rows * 257];
    let mut hw_blob = vec![0u64; num_rows * 32];
    let mut c_blob = vec![0u64; num_rows * 128];

    for i in 0..num_rows {
        timestamps[i] = rng.r#gen::<u64>();
        addr_xgs[i] = rng.r#gen::<u64>();
        addr_ks[i] = rng.r#gen::<u64>();
        addr_xrs[i] = rng.r#gen::<u64>();
        let len_k = (rng.r#gen::<u32>() & 0x1F) as u64; // 0..32
        let active = (rng.r#gen::<u32>() & 1) as u64;
        flags_len_k[i] = len_k | (active << 8);
        for j in 0..257 {
            byte_blob[i * 257 + j] = (rng.r#gen::<u32>() & 0xFF) as u64;
        }
        for j in 0..32 {
            hw_blob[i * 32 + j] = (rng.r#gen::<u32>() & 0xFFFF) as u64;
        }
        for j in 0..128 {
            c_blob[i * 128 + j] = rng.r#gen::<u64>();
        }
    }

    let mut cpu = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        cpu_row(
            &mut cpu,
            row * NUM_COLS,
            timestamps[row],
            addr_xgs[row],
            addr_ks[row],
            addr_xrs[row],
            flags_len_k[row],
            &byte_blob[row * 257..row * 257 + 257],
            &hw_blob[row * 32..row * 32 + 32],
            &c_blob[row * 128..row * 128 + 128],
        );
    }
    let gpu = generate_ecsm_trace_dev(
        num_rows,
        &timestamps,
        &addr_xgs,
        &addr_ks,
        &addr_xrs,
        &flags_len_k,
        &byte_blob,
        &hw_blob,
        &c_blob,
        NUM_COLS,
    )
    .unwrap();
    assert_eq!(cpu, gpu);
}

#[test]
fn ecsm_trace_parity_small() {
    run_parity(4, 1);
    run_parity(16, 2);
}

#[test]
fn ecsm_trace_parity_realistic() {
    run_parity(64, 100);
    run_parity(256, 101);
}
