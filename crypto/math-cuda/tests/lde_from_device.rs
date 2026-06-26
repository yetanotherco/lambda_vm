//! Parity: the device-resident LDE seam
//! (`coset_lde_from_device_with_merkle_tree_keep`) vs the host-input path
//! (`coset_lde_batch_base_into_with_merkle_tree_keep`). Same input columns and
//! weights through both paths must yield byte-identical LDE outputs AND
//! Merkle node buffers — proving the device-to-device input load matches the
//! host pack + H2D.
//!
//! `#[ignore]`'d (needs a GPU). Run with:
//!   cargo test -p math-cuda --release --test lde_from_device -- --ignored --nocapture

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Fp = FieldElement<GoldilocksField>;

/// A random canonical Goldilocks value (< p), as raw u64.
fn rand_fp(rng: &mut ChaCha8Rng) -> u64 {
    *Fp::from(rng.r#gen::<u64>()).value()
}

fn run(log_n: u32, blowup: usize, m: usize, seed: u64) {
    let n = 1usize << log_n;
    let lde_size = n * blowup;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Random input columns (column-major flat) and weights (both paths identical).
    let mut flat = vec![0u64; m * n];
    for v in flat.iter_mut() {
        *v = rand_fp(&mut rng);
    }
    let weights: Vec<u64> = (0..n).map(|_| rand_fp(&mut rng)).collect();

    // --- host-input path ---
    let columns: Vec<&[u64]> = (0..m).map(|c| &flat[c * n..c * n + n]).collect();
    let mut host_out_raw: Vec<Vec<u64>> = (0..m).map(|_| vec![0u64; lde_size]).collect();
    let mut host_out: Vec<&mut [u64]> = host_out_raw.iter_mut().map(|v| v.as_mut_slice()).collect();
    let mut host_nodes = vec![0u8; (2 * lde_size - 1) * 32];
    math_cuda::lde::coset_lde_batch_base_into_with_merkle_tree_keep(
        &columns,
        blowup,
        &weights,
        &mut host_out,
        &mut host_nodes,
    )
    .unwrap();

    // --- device-resident path (same input uploaded as DeviceMainCols) ---
    let dev_cols = math_cuda::trace::DeviceMainCols::upload(&flat, m, n).unwrap();
    let mut dev_out_raw: Vec<Vec<u64>> = (0..m).map(|_| vec![0u64; lde_size]).collect();
    let mut dev_out: Vec<&mut [u64]> = dev_out_raw.iter_mut().map(|v| v.as_mut_slice()).collect();
    let mut dev_nodes = vec![0u8; (2 * lde_size - 1) * 32];
    math_cuda::lde::coset_lde_from_device_with_merkle_tree_keep(
        &dev_cols,
        blowup,
        &weights,
        &mut dev_out,
        &mut dev_nodes,
    )
    .unwrap();

    // LDE outputs must match.
    for c in 0..m {
        assert_eq!(
            host_out_raw[c], dev_out_raw[c],
            "LDE output mismatch col {c} at log_n={log_n} blowup={blowup} m={m}"
        );
    }
    // Merkle node buffers (hence roots) must match.
    assert_eq!(
        host_nodes, dev_nodes,
        "Merkle node mismatch at log_n={log_n} blowup={blowup} m={m}"
    );
    println!("from-device LDE OK: log_n={log_n} blowup={blowup} m={m} (lde_size={lde_size})");
}

#[test]
#[ignore = "requires GPU; run with --ignored --nocapture"]
fn lde_from_device_matches_host() {
    for &(log_n, blowup, m) in &[
        (4u32, 2usize, 3usize),
        (6, 2, 8),
        (8, 4, 5),
        (10, 2, 38), // CPU-table width at a realistic size
        (14, 2, 16),
    ] {
        run(log_n, blowup, m, 100 + log_n as u64 * 7 + m as u64);
    }
}
