//! Parity: the multi-eval-point chunked barycentric kernels match K separate
//! single-point strided calls over the same device LDE handle.

use std::sync::Arc;

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math_cuda::barycentric::{
    barycentric_base_multi_on_device, barycentric_base_on_device, barycentric_ext3_multi_on_device,
    barycentric_ext3_on_device,
};
use math_cuda::device::backend;
use math_cuda::lde::{GpuLdeBase, GpuLdeExt3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;

fn rand_fp(rng: &mut ChaCha8Rng) -> Fp {
    Fp::from_raw(rng.r#gen::<u64>())
}

fn run_base(log_trace: u32, blowup: usize, num_cols: usize, k_points: usize, seed: u64) {
    let n = 1usize << log_trace;
    let lde_size = n * blowup;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut lde_flat = vec![0u64; num_cols * lde_size];
    for v in lde_flat.iter_mut() {
        *v = *rand_fp(&mut rng).value();
    }
    let coset_points: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>()).collect();
    // K contiguous inv_denom blocks of 3n, the R3DevContext layout.
    let inv_denoms_all: Vec<u64> = (0..(k_points * n * 3))
        .map(|_| rng.r#gen::<u64>())
        .collect();

    let be = backend().unwrap();
    let stream = be.next_stream();
    let lde_dev = stream.clone_htod(&lde_flat).unwrap();
    let points_dev = stream.clone_htod(&coset_points).unwrap();
    let inv_dev = stream.clone_htod(&inv_denoms_all).unwrap();
    stream.synchronize().unwrap();
    let handle = GpuLdeBase {
        ready: None,
        buf: Arc::new(lde_dev),
        m: num_cols,
        lde_size,
        tree: None,
        trace_dev: None,
        trace_rows: 0,
    };

    let multi = barycentric_base_multi_on_device(
        &stream,
        &handle,
        blowup,
        &points_dev,
        &inv_dev,
        n,
        k_points,
    )
    .unwrap();
    assert_eq!(multi.len(), 3 * k_points * num_cols);

    for k in 0..k_points {
        let single = barycentric_base_on_device(
            &handle,
            blowup,
            &coset_points,
            &inv_denoms_all[k * 3 * n..(k + 1) * 3 * n],
            n,
        )
        .unwrap();
        assert_eq!(
            &multi[k * 3 * num_cols..(k + 1) * 3 * num_cols],
            &single[..],
            "base multi mismatch at k={k} (log_trace={log_trace}, blowup={blowup}, \
             cols={num_cols}, k_points={k_points})"
        );
    }
}

fn run_ext3(log_trace: u32, blowup: usize, num_cols: usize, k_points: usize, seed: u64) {
    let n = 1usize << log_trace;
    let lde_size = n * blowup;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut lde_flat = vec![0u64; num_cols * 3 * lde_size];
    for v in lde_flat.iter_mut() {
        *v = *rand_fp(&mut rng).value();
    }
    let coset_points: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>()).collect();
    let inv_denoms_all: Vec<u64> = (0..(k_points * n * 3))
        .map(|_| rng.r#gen::<u64>())
        .collect();

    let be = backend().unwrap();
    let stream = be.next_stream();
    let lde_dev = stream.clone_htod(&lde_flat).unwrap();
    let points_dev = stream.clone_htod(&coset_points).unwrap();
    let inv_dev = stream.clone_htod(&inv_denoms_all).unwrap();
    stream.synchronize().unwrap();
    let handle = GpuLdeExt3 {
        ready: None,
        buf: Arc::new(lde_dev),
        m: num_cols,
        lde_size,
        tree: None,
    };

    let multi = barycentric_ext3_multi_on_device(
        &stream,
        &handle,
        blowup,
        &points_dev,
        &inv_dev,
        n,
        k_points,
    )
    .unwrap();
    assert_eq!(multi.len(), 3 * k_points * num_cols);

    for k in 0..k_points {
        let single = barycentric_ext3_on_device(
            &handle,
            blowup,
            &coset_points,
            &inv_denoms_all[k * 3 * n..(k + 1) * 3 * n],
            n,
        )
        .unwrap();
        assert_eq!(
            &multi[k * 3 * num_cols..(k + 1) * 3 * num_cols],
            &single[..],
            "ext3 multi mismatch at k={k} (log_trace={log_trace}, blowup={blowup}, \
             cols={num_cols}, k_points={k_points})"
        );
    }
}

#[test]
fn bary_base_multi_matches_single_point() {
    // Covers: k=1 degenerate, the production k=2, the kernel cap k=8, a
    // single-chunk tiny n, and a column count that forces the chunk heuristic
    // to its occupancy branch.
    for (log_t, blowup, cols, k) in [
        (4u32, 2usize, 3usize, 1usize),
        (8, 4, 10, 2),
        (12, 2, 5, 3),
        (14, 2, 100, 2),
        (10, 2, 4, 8),
    ] {
        run_base(log_t, blowup, cols, k, 3000 + log_t as u64 + k as u64);
    }
}

#[test]
fn bary_ext3_multi_matches_single_point() {
    for (log_t, blowup, cols, k) in [
        (4u32, 2usize, 2usize, 1usize),
        (8, 4, 5, 2),
        (10, 2, 3, 3),
        (14, 2, 40, 2),
        (10, 2, 4, 8),
    ] {
        run_ext3(log_t, blowup, cols, k, 4000 + log_t as u64 + k as u64);
    }
}
