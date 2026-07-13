//! Parity for the on-GPU trace generator's LDE seam: the device-input keep path
//! [`coset_lde_row_major_with_merkle_tree_keep_dev`] must produce a
//! byte-identical Merkle root and row-major LDE as the host-input keep path
//! [`coset_lde_row_major_with_merkle_tree_keep`] for the same matrix. This is
//! the "Step 1a" isolation test: it validates the seam without any trace-fill
//! kernel — upload a host matrix, run both paths, compare.
//!
//! Requires a GPU (skips cleanly if the CUDA backend is unavailable).

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsPrimeField};
use math_cuda::device::backend;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Coset weights `[1/N, g/N, ...]` — the layout `crypto/stark/src/prover.rs` uses.
fn coset_weights(n: usize, coset_offset: u64) -> Vec<u64> {
    let inv_n = FieldElement::<GoldilocksField>::from(n as u64)
        .inv()
        .expect("n non-zero");
    let mut w = Vec::with_capacity(n);
    let mut cur = *inv_n.value();
    for _ in 0..n {
        w.push(cur);
        cur = GoldilocksField::mul(&cur, &coset_offset);
    }
    w
}

fn assert_dev_matches_host(log_n: u64, m: usize, blowup: usize, seed: u64) {
    // Skip cleanly when no GPU/CUDA backend is present.
    if backend().is_err() {
        eprintln!("skipping lde_dev_parity: no CUDA backend");
        return;
    }

    let n = 1usize << log_n;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Row-major [row*m + col], the layout the trace generator emits.
    let row_major: Vec<u64> = (0..n * m).map(|_| rng.r#gen::<u64>()).collect();

    let coset_offset = 7u64;
    let weights = coset_weights(n, coset_offset);

    // Host path: uploads `row_major` internally.
    let (h_handle, h_lde) =
        math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep(&row_major, n, m, blowup, &weights)
            .expect("host keep");

    // Device path: pre-upload the SAME matrix, then run the device-input LDE.
    let be = backend().unwrap();
    let stream = be.next_stream();
    let dev = stream.clone_htod(&row_major).expect("upload matrix");
    stream.synchronize().expect("sync upload");
    let (d_handle, d_lde) = math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep_dev(
        &dev, n, m, blowup, &weights,
    )
    .expect("dev keep");

    assert_eq!(
        h_handle.tree.as_ref().unwrap().root,
        d_handle.tree.as_ref().unwrap().root,
        "Merkle root mismatch (n={n}, m={m}, blowup={blowup})"
    );
    let hc: Vec<u64> = h_lde.iter().map(GoldilocksField::canonical).collect();
    let dc: Vec<u64> = d_lde.iter().map(GoldilocksField::canonical).collect();
    assert_eq!(
        hc, dc,
        "row-major LDE mismatch (n={n}, m={m}, blowup={blowup})"
    );
}

#[test]
fn dev_matches_host_cpu_table_width() {
    // CPU table is 38 columns — the P1 target.
    assert_dev_matches_host(10, 38, 2, 0xC0DE);
    assert_dev_matches_host(12, 38, 4, 0xBEEF);
}

#[test]
fn dev_matches_host_various_widths() {
    assert_dev_matches_host(12, 10, 2, 0x11); // memw_register width
    assert_dev_matches_host(14, 16, 2, 0x22); // store width
    assert_dev_matches_host(13, 49, 2, 0x33); // memw width
}
