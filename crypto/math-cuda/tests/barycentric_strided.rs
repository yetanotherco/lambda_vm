//! Parity: strided barycentric kernels (used by R3 OOD on device LDE handles)
//! match the non-strided kernels fed a pre-strided column buffer.

use std::sync::Arc;

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math_cuda::barycentric::{
    barycentric_base, barycentric_base_on_device, barycentric_ext3, barycentric_ext3_on_device,
};
use math_cuda::device::backend;
use math_cuda::lde::{GpuLdeBase, GpuLdeExt3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

fn rand_fp(rng: &mut ChaCha8Rng) -> Fp {
    Fp::from_raw(rng.r#gen::<u64>())
}
fn rand_fp3(rng: &mut ChaCha8Rng) -> Fp3 {
    Fp3::new([rand_fp(rng), rand_fp(rng), rand_fp(rng)])
}

fn run_base(log_trace: u32, blowup: usize, num_cols: usize, seed: u64) {
    let n = 1usize << log_trace;
    let lde_size = n * blowup;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let lde_data: Vec<Vec<Fp>> = (0..num_cols)
        .map(|_| (0..lde_size).map(|_| rand_fp(&mut rng)).collect())
        .collect();
    let coset_points: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>()).collect();
    let inv_denoms_ext3: Vec<u64> = (0..(n * 3)).map(|_| rng.r#gen::<u64>()).collect();

    // Pack full LDE column-major for device.
    let mut lde_flat = vec![0u64; num_cols * lde_size];
    for (c, col) in lde_data.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            lde_flat[c * lde_size + r] = *v.value();
        }
    }
    let be = backend().unwrap();
    let stream = be.next_stream();
    let lde_dev = stream.clone_htod(&lde_flat).unwrap();
    stream.synchronize().unwrap();
    let handle = GpuLdeBase {
        buf: Arc::new(lde_dev),
        m: num_cols,
        lde_size,
        tree: None,
        trace_dev: None,
        trace_rows: 0,
    };

    // Pre-strided buffer for non-strided reference: trace-size picks of each col.
    let mut pre_strided = vec![0u64; num_cols * n];
    for c in 0..num_cols {
        for i in 0..n {
            pre_strided[c * n + i] = lde_flat[c * lde_size + i * blowup];
        }
    }

    let reference = barycentric_base(
        &pre_strided,
        n,
        &coset_points,
        &inv_denoms_ext3,
        n,
        num_cols,
    )
    .unwrap();

    let strided =
        barycentric_base_on_device(&handle, blowup, &coset_points, &inv_denoms_ext3, n).unwrap();

    assert_eq!(
        reference, strided,
        "base strided mismatch (log_trace={log_trace}, blowup={blowup}, cols={num_cols})"
    );
}

fn run_ext3(log_trace: u32, blowup: usize, num_cols: usize, seed: u64) {
    let n = 1usize << log_trace;
    let lde_size = n * blowup;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let lde_data: Vec<Vec<Fp3>> = (0..num_cols)
        .map(|_| (0..lde_size).map(|_| rand_fp3(&mut rng)).collect())
        .collect();
    let coset_points: Vec<u64> = (0..n).map(|_| rng.r#gen::<u64>()).collect();
    let inv_denoms_ext3: Vec<u64> = (0..(n * 3)).map(|_| rng.r#gen::<u64>()).collect();

    // Pack LDE de-interleaved: (m*3) by lde_size.
    let mut lde_flat = vec![0u64; num_cols * 3 * lde_size];
    for (c, col) in lde_data.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            lde_flat[(c * 3) * lde_size + r] = *v.value()[0].value();
            lde_flat[(c * 3 + 1) * lde_size + r] = *v.value()[1].value();
            lde_flat[(c * 3 + 2) * lde_size + r] = *v.value()[2].value();
        }
    }
    let be = backend().unwrap();
    let stream = be.next_stream();
    let lde_dev = stream.clone_htod(&lde_flat).unwrap();
    stream.synchronize().unwrap();
    let handle = GpuLdeExt3 {
        buf: Arc::new(lde_dev),
        m: num_cols,
        lde_size,
        tree: None,
        ready: None,
    };

    // Pre-strided buffer for non-strided reference.
    let mut pre_strided = vec![0u64; num_cols * 3 * n];
    for c in 0..num_cols {
        for i in 0..n {
            pre_strided[(c * 3) * n + i] = lde_flat[(c * 3) * lde_size + i * blowup];
            pre_strided[(c * 3 + 1) * n + i] = lde_flat[(c * 3 + 1) * lde_size + i * blowup];
            pre_strided[(c * 3 + 2) * n + i] = lde_flat[(c * 3 + 2) * lde_size + i * blowup];
        }
    }
    let reference = barycentric_ext3(
        &pre_strided,
        n,
        &coset_points,
        &inv_denoms_ext3,
        n,
        num_cols,
    )
    .unwrap();

    let strided =
        barycentric_ext3_on_device(&handle, blowup, &coset_points, &inv_denoms_ext3, n).unwrap();

    assert_eq!(reference, strided, "ext3 strided mismatch");
}

#[test]
fn bary_base_strided_small() {
    for (log_t, blowup, cols) in [(4u32, 2usize, 3usize), (8, 4, 10), (12, 2, 5)] {
        run_base(log_t, blowup, cols, 1000 + log_t as u64);
    }
}

#[test]
fn bary_ext3_strided_small() {
    for (log_t, blowup, cols) in [(4u32, 2usize, 2usize), (8, 4, 5), (10, 2, 3)] {
        run_ext3(log_t, blowup, cols, 2000 + log_t as u64);
    }
}
