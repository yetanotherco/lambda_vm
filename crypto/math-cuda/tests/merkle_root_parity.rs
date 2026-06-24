//! Parity: GPU LDE + GPU Keccak leaf hash + GPU Merkle tree must produce the
//! same root as the CPU row-major LDE path introduced in PR #650
//! (`coset_lde_full_expand_row_major` + `commit_rows_bit_reversed`).
//!
//! This is the end-to-end checkpoint that closes the CPU/GPU commitment parity
//! gap: the GPU fused pipeline commits on-device (before `columns_to_row_major`
//! is called), so its root must agree with what the CPU path would commit for
//! the same input columns.

use math::fft::two_half_fft::TwoHalfTwiddles;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use math::polynomial::Polynomial;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use stark::prover::{IsStarkProver, Prover};

type Fp = FieldElement<GoldilocksField>;

fn coset_weights(n: usize, g: u64) -> Vec<Fp> {
    let inv_n = Fp::from(n as u64).inv().unwrap();
    let g_fp = Fp::from_raw(g);
    let mut w = Vec::with_capacity(n);
    let mut cur = inv_n;
    for _ in 0..n {
        w.push(cur.clone());
        cur = &cur * &g_fp;
    }
    w
}

fn coset_weights_u64(n: usize, g: u64) -> Vec<u64> {
    coset_weights(n, g)
        .iter()
        .map(|w| *w.value())
        .collect()
}

/// Run GPU batch LDE + GPU Keccak leaf hashing + GPU Merkle tree build.
/// Returns the 32-byte root extracted from the node array.
fn gpu_merkle_root(columns: &[Vec<u64>], blowup: usize, weights: &[u64]) -> [u8; 32] {
    let col_slices: Vec<&[u64]> = columns.iter().map(|c| c.as_slice()).collect();
    let lde_columns = math_cuda::lde::coset_lde_batch_base(&col_slices, blowup, weights)
        .expect("GPU batch LDE");

    let n_lde = lde_columns[0].len();
    let num_cols = lde_columns.len();

    // Pack into column-major flat layout: [col * stride + row].
    let mut flat = vec![0u64; num_cols * n_lde];
    for (c, col) in lde_columns.iter().enumerate() {
        for (r, &v) in col.iter().enumerate() {
            flat[c * n_lde + r] = v;
        }
    }

    let gpu_leaves =
        math_cuda::merkle::keccak_leaves_base(&flat, n_lde, num_cols, n_lde)
            .expect("GPU keccak leaves");
    let nodes = math_cuda::merkle::build_merkle_tree_on_device(&gpu_leaves)
        .expect("GPU Merkle tree");

    // Root is the last 32 bytes of the node array.
    let mut root = [0u8; 32];
    root.copy_from_slice(&nodes[nodes.len() - 32..]);
    root
}

/// Run the new CPU row-major LDE (`coset_lde_full_expand_row_major`) +
/// `commit_rows_bit_reversed` and return the Merkle root.
fn cpu_row_major_merkle_root(
    columns: &[Vec<u64>],
    blowup: usize,
    weights: &[Fp],
    inv_tw: &TwoHalfTwiddles<GoldilocksField>,
    fwd_tw: &TwoHalfTwiddles<GoldilocksField>,
) -> [u8; 32] {
    let n = columns[0].len();
    let num_cols = columns.len();

    // Build row-major buffer: data[row * num_cols + col] = columns[col][row].
    let mut buf: Vec<Fp> = vec![Fp::from(0u64); n * num_cols];
    for (c, col) in columns.iter().enumerate() {
        for (r, &v) in col.iter().enumerate() {
            buf[r * num_cols + c] = Fp::from_raw(v);
        }
    }

    Polynomial::<Fp>::coset_lde_full_expand_row_major::<GoldilocksField>(
        &mut buf,
        num_cols,
        blowup,
        weights,
        inv_tw,
        fwd_tw,
    )
    .expect("CPU row-major LDE");

    let (_, root) = Prover::<GoldilocksField, GoldilocksField, ()>::commit_rows_bit_reversed(
        &buf, num_cols,
    )
    .expect("CPU commit");

    root
}

#[test]
fn gpu_and_cpu_row_major_merkle_roots_match() {
    const COSET_OFFSET: u64 = 7;

    for log_n in [4usize, 6, 8, 10] {
        for blowup in [2usize, 4] {
            for num_cols in [1usize, 3, 8] {
                let n = 1usize << log_n;
                let log_lde = (n * blowup).trailing_zeros() as usize;
                let mut rng = ChaCha8Rng::seed_from_u64(
                    (log_n * 1000 + blowup * 100 + num_cols) as u64,
                );

                let columns: Vec<Vec<u64>> = (0..num_cols)
                    .map(|_| (0..n).map(|_| rng.r#gen::<u64>()).collect())
                    .collect();

                let weights_u64 = coset_weights_u64(n, COSET_OFFSET);
                let weights_fp = coset_weights(n, COSET_OFFSET);
                let inv_tw = TwoHalfTwiddles::<GoldilocksField>::new(log_n, true)
                    .expect("inv twiddles");
                let fwd_tw = TwoHalfTwiddles::<GoldilocksField>::new(log_lde, false)
                    .expect("fwd twiddles");

                let gpu_root = gpu_merkle_root(&columns, blowup, &weights_u64);
                let cpu_root = cpu_row_major_merkle_root(
                    &columns,
                    blowup,
                    &weights_fp,
                    &inv_tw,
                    &fwd_tw,
                );

                assert_eq!(
                    gpu_root,
                    cpu_root,
                    "root mismatch: log_n={log_n} blowup={blowup} num_cols={num_cols}"
                );
            }
        }
    }
}
