//! Parity: GPU `generate_bitwise_trace_dev` vs an inlined CPU reference
//! mirroring `prover/src/tables/bitwise.rs::generate_bitwise_trace`'s
//! row layout (precomputed columns only — multiplicity columns are 0).

use math_cuda::bitwise_trace::generate_bitwise_trace_dev;

const NUM_COLS: usize = 21;
const NUM_ROWS: usize = 256 * 256 * 16;

fn cpu_bitwise_trace() -> Vec<u64> {
    let mut data = vec![0u64; NUM_ROWS * NUM_COLS];
    for x in 0u64..256 {
        for y in 0u64..256 {
            for z in 0u64..16 {
                let row = (x as usize) + (y as usize) * 256 + (z as usize) * 256 * 256;
                let base = row * NUM_COLS;
                let halfword = x + y * 256;
                let msb8 = (x >> 7) & 1;
                let msb16 = (halfword >> 15) & 1;
                let is_zero = if x == 0 && y == 0 && z == 0 { 1 } else { 0 };
                let sll = if z == 0 { halfword } else { (halfword << z) & 0xFFFF };
                let sllc = if z == 0 { 0 } else { halfword >> (16 - z) };

                data[base] = x;
                data[base + 1] = y;
                data[base + 2] = z;
                data[base + 3] = x & y;
                data[base + 4] = x | y;
                data[base + 5] = x ^ y;
                data[base + 6] = msb8;
                data[base + 7] = msb16;
                data[base + 8] = is_zero;
                data[base + 9] = sll;
                data[base + 10] = sllc;
            }
        }
    }
    data
}

#[test]
fn bitwise_trace_full_parity() {
    let cpu = cpu_bitwise_trace();
    let gpu = generate_bitwise_trace_dev(NUM_ROWS, NUM_COLS).unwrap();
    assert_eq!(cpu.len(), gpu.len());
    // Element-by-element check; print first mismatch to keep output small.
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        if c != g {
            panic!(
                "bitwise mismatch at i={i} (row={}, col={}): cpu={c} gpu={g}",
                i / NUM_COLS,
                i % NUM_COLS,
            );
        }
    }
}
