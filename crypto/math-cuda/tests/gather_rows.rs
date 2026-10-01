//! Parity: the device row-gather (`gather_rows_base` / `gather_rows_ext3`,
//! used by R4 query openings to read opened column values from the resident LDE
//! instead of the host trace) returns exactly the column values obtained by
//! directly indexing the column-major device buffer at each requested row.

use std::sync::Arc;

use math_cuda::barycentric::{gather_rows_base_on_device, gather_rows_ext3_on_device};
use math_cuda::device::backend;
use math_cuda::lde::{GpuLdeBase, GpuLdeExt3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn run_base(lde_size: usize, num_cols: usize, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Column-major base LDE: buf[col*lde_size + row].
    let buf: Vec<u64> = (0..num_cols * lde_size)
        .map(|_| rng.r#gen::<u64>())
        .collect();

    let be = backend().unwrap();
    let stream = be.next_stream();
    let dev = stream.clone_htod(&buf).unwrap();
    stream.synchronize().unwrap();
    let handle = GpuLdeBase {
        ready: None,
        buf: Arc::new(dev),
        m: num_cols,
        lde_size,
        tree: None,
        trace_dev: None,
        trace_rows: 0,
    };

    let rows: Vec<u32> = (0..9).map(|_| rng.gen_range(0..lde_size) as u32).collect();
    let got = gather_rows_base_on_device(&handle, &rows, &stream).unwrap();
    assert_eq!(got.len(), rows.len() * num_cols, "base gather shape");
    for (q, &row) in rows.iter().enumerate() {
        for col in 0..num_cols {
            assert_eq!(
                got[q * num_cols + col],
                buf[col * lde_size + row as usize],
                "base gather mismatch: row {row}, col {col}"
            );
        }
    }
}

fn run_ext3(lde_size: usize, num_cols: usize, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // De-interleaved ext3 LDE: buf[(col*3 + k)*lde_size + row].
    let buf: Vec<u64> = (0..num_cols * 3 * lde_size)
        .map(|_| rng.r#gen::<u64>())
        .collect();

    let be = backend().unwrap();
    let stream = be.next_stream();
    let dev = stream.clone_htod(&buf).unwrap();
    stream.synchronize().unwrap();
    let handle = GpuLdeExt3 {
        ready: None,
        buf: Arc::new(dev),
        m: num_cols,
        lde_size,
        tree: None,
    };

    let rows: Vec<u32> = (0..9).map(|_| rng.gen_range(0..lde_size) as u32).collect();
    let got = gather_rows_ext3_on_device(&handle, &rows, &stream).unwrap();
    assert_eq!(got.len(), rows.len() * num_cols * 3, "ext3 gather shape");
    for (q, &row) in rows.iter().enumerate() {
        for col in 0..num_cols {
            let o = (q * num_cols + col) * 3;
            for k in 0..3 {
                assert_eq!(
                    got[o + k],
                    buf[(col * 3 + k) * lde_size + row as usize],
                    "ext3 gather mismatch: row {row}, col {col}, comp {k}"
                );
            }
        }
    }
}

#[test]
fn gather_rows_base_matches_direct_indexing() {
    for (log_size, cols) in [(6u32, 3usize), (12, 20), (16, 8)] {
        run_base(1usize << log_size, cols, 100 + log_size as u64);
    }
}

#[test]
fn gather_rows_ext3_matches_direct_indexing() {
    for (log_size, cols) in [(6u32, 2usize), (12, 5), (16, 3)] {
        run_ext3(1usize << log_size, cols, 200 + log_size as u64);
    }
}
