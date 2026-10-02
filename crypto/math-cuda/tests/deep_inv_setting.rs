//! The DEEP/OOD denominators follow the process setting: inverted row-wise and
//! summed row-chunked by default, the legacy scan and one block per column
//! under `LAMBDA_VM_DEEP_INV_LEGACY=1` (`math_cuda::deep_inv::LEGACY_ENV`).
//!
//! ```text
//! cargo test --release -p math-cuda --test deep_inv_setting -- --nocapture
//! LAMBDA_VM_DEEP_INV_LEGACY=1 cargo test --release -p math-cuda --test deep_inv_setting -- --nocapture
//! ```
//!
//! The parity tests name both paths per call (`tests/deep_inv_parity.rs`), so
//! none of them reads the setting; this one does. It is its own binary with one
//! test, so the process-wide counters move for its two calls alone, and the
//! once-per-process `[gpu] DEEP/OOD denominators: …` banner is printed by its
//! first call (visible under `--nocapture`).

use std::sync::Arc;

use math_cuda::barycentric::barycentric_ext3_on_device_with_dev_inv_denoms;
use math_cuda::deep_inv::{LEGACY_ENV, chunked_ood_sums, rowwise_inversions};
use math_cuda::device::backend;
use math_cuda::inverse::{DenomSign, compute_and_invert_denoms_ext3_dev};
use math_cuda::lde::GpuLdeExt3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const P: u64 = 0xFFFF_FFFF_0000_0001;

#[test]
fn the_denominators_follow_the_process_setting() {
    let setting = std::env::var(LEGACY_ENV).unwrap_or_else(|_| "<unset>".into());
    let legacy = setting == "1";
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let mut elem = || rng.r#gen::<u64>() % P;
    let be = backend().expect("cuda backend");
    let stream = be.next_stream();

    // One inversion: two OOD points over a 2^12-point domain.
    let n = 1usize << 12;
    let points: Vec<u64> = (0..n).map(|_| elem()).collect();
    let z: Vec<u64> = (0..6).map(|_| elem()).collect();
    let points_dev = stream.clone_htod(&points).expect("upload the points");
    let before = rowwise_inversions();
    let inv =
        compute_and_invert_denoms_ext3_dev(&points_dev, &z, n, 2, DenomSign::ZMinusX, &stream)
            .expect("the inversion");
    let rowwise = rowwise_inversions() - before;

    // One single-point OOD sum: the second point's inverses over one ext3
    // column at blowup 2.
    let blowup = 2;
    let column: Vec<u64> = (0..3 * n * blowup).map(|_| elem()).collect();
    let aux = GpuLdeExt3 {
        ready: None,
        buf: Arc::new(stream.clone_htod(&column).expect("upload the column")),
        m: 1,
        lde_size: n * blowup,
        tree: None,
    };
    let before = chunked_ood_sums();
    barycentric_ext3_on_device_with_dev_inv_denoms(
        &stream,
        &aux,
        blowup,
        &points_dev,
        &inv,
        3 * n,
        n,
    )
    .expect("the OOD sum");
    let chunked = chunked_ood_sums() - before;
    stream.synchronize().expect("sync");

    println!(
        "deep_inv_setting: {LEGACY_ENV}={setting}: the inversion took the {}, the OOD sum took {}",
        if rowwise == 1 {
            "row-wise kernel"
        } else {
            "scan"
        },
        if chunked == 1 {
            "the chunked kernel"
        } else {
            "one block per column"
        },
    );
    assert_eq!(
        rowwise,
        u64::from(!legacy),
        "{LEGACY_ENV}={setting}: the inversion did not follow the process setting"
    );
    assert_eq!(
        chunked,
        u64::from(!legacy),
        "{LEGACY_ENV}={setting}: the OOD sum did not follow the process setting"
    );
}
