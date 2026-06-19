//! Parity: GPU `generate_page_trace_dev` vs the CPU `generate_page_trace`
//! in `prover/src/tables/page.rs`. Compares the produced row-major u64
//! buffer element-by-element across a few page configs.

use math_cuda::page_trace::generate_page_trace_dev;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_COLS: usize = 5;

/// CPU reference, inlined from `prover/src/tables/page.rs:generate_page_trace`.
/// Produces row-major u64s in canonical Goldilocks form. Skips the
/// `FieldElement` wrapping since the parity check is on raw u64.
fn cpu_page_trace(
    page_size: usize,
    init_values: &[u64],
    final_values: &[u64],
    final_timestamps: &[u64],
) -> Vec<u64> {
    assert_eq!(init_values.len(), page_size);
    assert_eq!(final_values.len(), page_size);
    assert_eq!(final_timestamps.len(), page_size);
    let mut data = vec![0u64; page_size * NUM_COLS];
    for offset in 0..page_size {
        let base = offset * NUM_COLS;
        let ts = final_timestamps[offset];
        data[base] = offset as u64;
        data[base + 1] = init_values[offset];
        data[base + 2] = final_values[offset];
        data[base + 3] = ts & 0xFFFF_FFFF;
        data[base + 4] = ts >> 32;
    }
    data
}

fn run_parity(page_size: usize, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Init values are bytes (0..255 as u64).
    let init_values: Vec<u64> = (0..page_size).map(|_| rng.r#gen::<u8>() as u64).collect();
    // Final values: simulate ~25% of bytes were written, the rest stayed at init.
    let mut final_values = init_values.clone();
    let mut final_timestamps = vec![0u64; page_size];
    for offset in 0..page_size {
        if rng.r#gen::<u32>() % 4 == 0 {
            final_values[offset] = rng.r#gen::<u8>() as u64;
            final_timestamps[offset] = rng.r#gen::<u64>();
        }
    }

    let cpu = cpu_page_trace(page_size, &init_values, &final_values, &final_timestamps);
    let gpu =
        generate_page_trace_dev(page_size, &init_values, &final_values, &final_timestamps, NUM_COLS)
            .unwrap();
    assert_eq!(cpu, gpu, "page trace mismatch at page_size={page_size}");
}

#[test]
fn page_trace_parity_tiny() {
    run_parity(64, 100);
    run_parity(256, 101);
}

#[test]
fn page_trace_parity_realistic_small_page() {
    // 2^14 — useful for tests, also matches some preprocessed tables.
    run_parity(1 << 14, 200);
}

#[test]
fn page_trace_parity_default_page_size() {
    // DEFAULT_PAGE_SIZE = 2^18, the production size.
    run_parity(1 << 18, 9001);
}
