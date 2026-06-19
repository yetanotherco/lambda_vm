//! Parity: GPU `generate_decode_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/decode.rs:generate_decode_trace`'s row
//! layout. Multiplicity column stays at 0 in both (matches the contract:
//! `update_multiplicities` runs on host after this kernel).

use math_cuda::decode_trace::generate_decode_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 6;

fn cpu_decode_trace(
    num_rows: usize,
    pcs: &[u64],
    packed_decodes: &[u64],
    imms: &[u64],
) -> Vec<u64> {
    assert_eq!(pcs.len(), num_rows);
    let mut data = vec![0u64; num_rows * NUM_COLS];
    for row in 0..num_rows {
        let base = row * NUM_COLS;
        data[base] = pcs[row] & 0xFFFF_FFFF;
        data[base + 1] = pcs[row] >> 32;
        data[base + 2] = packed_decodes[row];
        data[base + 3] = imms[row] & 0xFFFF_FFFF;
        data[base + 4] = imms[row] >> 32;
        // MU stays at 0.
    }
    data
}

fn run_parity(num_entries: usize, num_rows: usize, seed: u64) {
    assert!(num_rows >= num_entries + 1);
    assert!(num_rows.is_power_of_two());

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut pcs = vec![0u64; num_rows];
    let mut packed = vec![0u64; num_rows];
    let mut imms = vec![0u64; num_rows];

    // Actual entries: random pcs in a wide 64-bit space, random packed/imm.
    for i in 0..num_entries {
        pcs[i] = rng.r#gen::<u64>();
        packed[i] = rng.r#gen::<u64>() & 0x00FF_FFFF; // packed_decode is < 2^24 in practice
        imms[i] = rng.r#gen::<u64>();
    }
    // CPU padding row at index num_entries: pc=1, packed=0, imm=0.
    pcs[num_entries] = 1;
    // Trailing padding rows: pc=1, packed=0, imm=0.
    for i in (num_entries + 1)..num_rows {
        pcs[i] = 1;
    }

    let cpu = cpu_decode_trace(num_rows, &pcs, &packed, &imms);
    let gpu = generate_decode_trace_dev(num_rows, &pcs, &packed, &imms, NUM_COLS).unwrap();
    assert_eq!(cpu, gpu, "decode trace mismatch num_rows={num_rows}");
}

#[test]
fn decode_trace_parity_tiny() {
    run_parity(3, 4, 1);
    run_parity(7, 8, 2);
}

#[test]
fn decode_trace_parity_realistic() {
    // Roughly fib_iterative_1M-scale program: ~100k actual entries, padded
    // to the next power of two.
    run_parity(60_000, 65_536, 100);
    run_parity(100_000, 131_072, 101);
}
