//! GPU LDE + GPU Keccak leaf hash + GPU Merkle tree must produce the same root
//! as the CPU row-major LDE path (`coset_lde_full_expand_row_major` +
//! `commit_rows_bit_reversed`). Covers base field (main trace) and ext3 (aux trace).
//!
//! Two non-obvious layout details caught while writing these tests:
//! - `build_merkle_tree_on_device` stores the tree top-down: root at `nodes[0..32]`,
//!   leaves in the tail (not the end).
//! - `keccak_leaves_ext3` expects component-major layout `[all-a, all-b, all-c]`,
//!   not the interleaved `[a,b,c per element]` that `coset_lde_batch_ext3_into` produces.

use math::fft::two_half_fft::TwoHalfTwiddles;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::polynomial::Polynomial;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use stark::config::KeccakStarkHash;
use stark::prover::{GenericProver, IsStarkProver};

/// The keccak prover, named: these tests compare against the CUDA keccak
/// kernels, so the CPU side must say keccak rather than follow the default
/// alias (BLAKE3 since the P-a flip).
type Prover<F, E, PI> = GenericProver<F, E, PI, KeccakStarkHash>;

type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

type Fp = FieldElement<GoldilocksField>;

fn coset_weights(n: usize, g: u64) -> Vec<Fp> {
    let inv_n = Fp::from(n as u64).inv().unwrap();
    let g_fp = Fp::from_raw(g);
    let mut w = Vec::with_capacity(n);
    let mut cur = inv_n;
    for _ in 0..n {
        w.push(cur);
        cur = &cur * &g_fp;
    }
    w
}

fn coset_weights_u64(n: usize, g: u64) -> Vec<u64> {
    coset_weights(n, g).iter().map(|w| *w.value()).collect()
}

/// Run GPU batch LDE + GPU Keccak leaf hashing + GPU Merkle tree build.
/// Returns the 32-byte root extracted from the node array.
fn gpu_merkle_root(columns: &[Vec<u64>], blowup: usize, weights: &[u64]) -> [u8; 32] {
    let col_slices: Vec<&[u64]> = columns.iter().map(|c| c.as_slice()).collect();
    let lde_columns =
        math_cuda::lde::coset_lde_batch_base(&col_slices, blowup, weights).expect("GPU batch LDE");

    let n_lde = lde_columns[0].len();
    let num_cols = lde_columns.len();

    // Pack into column-major flat layout: [col * stride + row].
    let mut flat = vec![0u64; num_cols * n_lde];
    for (c, col) in lde_columns.iter().enumerate() {
        for (r, &v) in col.iter().enumerate() {
            flat[c * n_lde + r] = v;
        }
    }

    // Row-pair leaves (rows_per_leaf = 2, matching `ROWS_PER_LEAF`): the CPU
    // reference is `commit_rows_bit_reversed`, which hashes bit-reversed row
    // pairs into each leaf, so the generic GPU keccak-leaves + Merkle path must
    // use the same row-pair layout to produce a matching root.
    let gpu_leaves = math_cuda::merkle::keccak_leaves_base(&flat, n_lde, num_cols, n_lde, 2)
        .expect("GPU keccak leaves");
    let nodes =
        math_cuda::merkle::build_merkle_tree_on_device(&gpu_leaves).expect("GPU Merkle tree");

    // `build_merkle_tree_on_device` places the root at index 0 (the leaves
    // live in the tail), so the root is the first 32 bytes of the node array.
    let mut root = [0u8; 32];
    root.copy_from_slice(&nodes[0..32]);
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
        &mut buf, num_cols, blowup, weights, inv_tw, fwd_tw,
    )
    .expect("CPU row-major LDE");

    let (_, root) =
        Prover::<GoldilocksField, GoldilocksField, ()>::commit_rows_bit_reversed(&buf, num_cols)
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
                let mut rng =
                    ChaCha8Rng::seed_from_u64((log_n * 1000 + blowup * 100 + num_cols) as u64);

                let columns: Vec<Vec<u64>> = (0..num_cols)
                    .map(|_| (0..n).map(|_| rng.r#gen::<u64>()).collect())
                    .collect();

                let weights_u64 = coset_weights_u64(n, COSET_OFFSET);
                let weights_fp = coset_weights(n, COSET_OFFSET);
                let inv_tw =
                    TwoHalfTwiddles::<GoldilocksField>::new(log_n, true).expect("inv twiddles");
                let fwd_tw =
                    TwoHalfTwiddles::<GoldilocksField>::new(log_lde, false).expect("fwd twiddles");

                let gpu_root = gpu_merkle_root(&columns, blowup, &weights_u64);
                let cpu_root =
                    cpu_row_major_merkle_root(&columns, blowup, &weights_fp, &inv_tw, &fwd_tw);

                assert_eq!(
                    gpu_root, cpu_root,
                    "root mismatch: log_n={log_n} blowup={blowup} num_cols={num_cols}"
                );
            }
        }
    }
}

// ── Ext3 helpers ─────────────────────────────────────────────────────────────

fn rand_ext3(rng: &mut ChaCha8Rng) -> Fp3 {
    Fp3::new([
        FieldElement::<GoldilocksField>::from_raw(rng.r#gen::<u64>()),
        FieldElement::<GoldilocksField>::from_raw(rng.r#gen::<u64>()),
        FieldElement::<GoldilocksField>::from_raw(rng.r#gen::<u64>()),
    ])
}

fn ext3_to_u64s(col: &[Fp3]) -> Vec<u64> {
    let mut out = Vec::with_capacity(col.len() * 3);
    for e in col {
        out.push(*e.value()[0].value());
        out.push(*e.value()[1].value());
        out.push(*e.value()[2].value());
    }
    out
}

/// GPU ext3 LDE + Keccak leaf hash + Merkle tree → root.
fn gpu_ext3_merkle_root(columns: &[Vec<Fp3>], blowup: usize, weights: &[u64]) -> [u8; 32] {
    let n = columns[0].len();
    let lde_size = n * blowup;
    let num_cols = columns.len();

    let flat_inputs: Vec<Vec<u64>> = columns.iter().map(|c| ext3_to_u64s(c)).collect();
    let input_slices: Vec<&[u64]> = flat_inputs.iter().map(|v| v.as_slice()).collect();

    let mut flat_outputs: Vec<Vec<u64>> = (0..num_cols).map(|_| vec![0u64; 3 * lde_size]).collect();
    {
        let mut out_slices: Vec<&mut [u64]> =
            flat_outputs.iter_mut().map(|v| v.as_mut_slice()).collect();
        math_cuda::lde::coset_lde_batch_ext3_into(
            &input_slices,
            n,
            blowup,
            weights,
            &mut out_slices,
        )
        .expect("GPU ext3 LDE");
    }

    // Repack from interleaved [a,b,c per element] to component-major
    // [all-a, all-b, all-c] as keccak_leaves_ext3 expects.
    let mut flat_for_keccak = vec![0u64; num_cols * 3 * lde_size];
    for (c, out) in flat_outputs.iter().enumerate() {
        for r in 0..lde_size {
            flat_for_keccak[(c * 3) * lde_size + r] = out[r * 3];
            flat_for_keccak[(c * 3 + 1) * lde_size + r] = out[r * 3 + 1];
            flat_for_keccak[(c * 3 + 2) * lde_size + r] = out[r * 3 + 2];
        }
    }

    // Row-pair leaves (rows_per_leaf = 2, matching `ROWS_PER_LEAF`) to match the
    // row-pair `commit_rows_bit_reversed` CPU reference below.
    let gpu_leaves =
        math_cuda::merkle::keccak_leaves_ext3(&flat_for_keccak, lde_size, num_cols, lde_size, 2)
            .expect("GPU ext3 keccak leaves");
    let nodes =
        math_cuda::merkle::build_merkle_tree_on_device(&gpu_leaves).expect("GPU Merkle tree");

    let mut root = [0u8; 32];
    root.copy_from_slice(&nodes[0..32]);
    root
}

/// CPU row-major ext3 LDE + `commit_rows_bit_reversed` → root.
fn cpu_ext3_row_major_merkle_root(
    columns: &[Vec<Fp3>],
    blowup: usize,
    weights: &[FieldElement<GoldilocksField>],
    inv_tw: &TwoHalfTwiddles<GoldilocksField>,
    fwd_tw: &TwoHalfTwiddles<GoldilocksField>,
) -> [u8; 32] {
    let n = columns[0].len();
    let num_cols = columns.len();

    let mut buf: Vec<Fp3> = vec![Fp3::from(0u64); n * num_cols];
    for (c, col) in columns.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            buf[r * num_cols + c] = *v;
        }
    }

    Polynomial::<Fp3>::coset_lde_full_expand_row_major::<GoldilocksField>(
        &mut buf, num_cols, blowup, weights, inv_tw, fwd_tw,
    )
    .expect("CPU ext3 row-major LDE");

    let (_, root) =
        Prover::<GoldilocksField, Degree3GoldilocksExtensionField, ()>::commit_rows_bit_reversed(
            &buf, num_cols,
        )
        .expect("CPU ext3 commit");

    root
}

#[test]
fn gpu_and_cpu_ext3_merkle_roots_match() {
    const COSET_OFFSET: u64 = 7;

    for log_n in [4usize, 6, 8] {
        for blowup in [2usize, 4] {
            for num_cols in [1usize, 3, 5] {
                let n = 1usize << log_n;
                let log_lde = (n * blowup).trailing_zeros() as usize;
                let mut rng = ChaCha8Rng::seed_from_u64(
                    (log_n * 1000 + blowup * 100 + num_cols) as u64 + 9999,
                );

                let columns: Vec<Vec<Fp3>> = (0..num_cols)
                    .map(|_| (0..n).map(|_| rand_ext3(&mut rng)).collect())
                    .collect();

                let weights_u64 = coset_weights_u64(n, COSET_OFFSET);
                let weights_fp = coset_weights(n, COSET_OFFSET);
                let inv_tw =
                    TwoHalfTwiddles::<GoldilocksField>::new(log_n, true).expect("inv twiddles");
                let fwd_tw =
                    TwoHalfTwiddles::<GoldilocksField>::new(log_lde, false).expect("fwd twiddles");

                let gpu_root = gpu_ext3_merkle_root(&columns, blowup, &weights_u64);
                let cpu_root =
                    cpu_ext3_row_major_merkle_root(&columns, blowup, &weights_fp, &inv_tw, &fwd_tw);

                assert_eq!(
                    gpu_root, cpu_root,
                    "ext3 root mismatch: log_n={log_n} blowup={blowup} num_cols={num_cols}"
                );
            }
        }
    }
}

// ── New row-major pipeline tests ─────────────────────────────────────────────

#[test]
fn new_row_major_pipeline_base_root_matches_cpu() {
    const COSET_OFFSET: u64 = 7;

    for log_n in [4usize, 6, 8, 10] {
        for blowup in [2usize, 4] {
            for num_cols in [1usize, 3, 8] {
                let n = 1usize << log_n;
                let log_lde = (n * blowup).trailing_zeros() as usize;
                let mut rng = ChaCha8Rng::seed_from_u64(
                    (log_n * 1000 + blowup * 100 + num_cols) as u64 + 10000,
                );

                let row_major: Vec<u64> = (0..n * num_cols).map(|_| rng.r#gen::<u64>()).collect();

                let weights_u64 = coset_weights_u64(n, COSET_OFFSET);
                let weights_fp = coset_weights(n, COSET_OFFSET);
                let inv_tw =
                    TwoHalfTwiddles::<GoldilocksField>::new(log_n, true).expect("inv twiddles");
                let fwd_tw =
                    TwoHalfTwiddles::<GoldilocksField>::new(log_lde, false).expect("fwd twiddles");

                let (handle, _lde) = math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep(
                    &row_major,
                    n,
                    num_cols,
                    blowup,
                    &weights_u64,
                    true,
                )
                .expect("new row-major GPU pipeline");
                let gpu_root = handle.tree.as_ref().expect("resident merkle tree").root;

                let cpu_root = cpu_row_major_merkle_root(
                    &(0..num_cols)
                        .map(|c| (0..n).map(|r| row_major[r * num_cols + c]).collect())
                        .collect::<Vec<Vec<u64>>>(),
                    blowup,
                    &weights_fp,
                    &inv_tw,
                    &fwd_tw,
                );

                assert_eq!(
                    gpu_root, cpu_root,
                    "new row-major pipeline root mismatch: log_n={log_n} blowup={blowup} num_cols={num_cols}"
                );
            }
        }
    }
}

#[test]
fn new_row_major_pipeline_ext3_root_matches_cpu() {
    const COSET_OFFSET: u64 = 7;

    for log_n in [4usize, 6, 8] {
        for blowup in [2usize, 4] {
            for num_cols in [1usize, 3, 5] {
                let n = 1usize << log_n;
                let log_lde = (n * blowup).trailing_zeros() as usize;
                let mut rng = ChaCha8Rng::seed_from_u64(
                    (log_n * 1000 + blowup * 100 + num_cols) as u64 + 20000,
                );

                let columns: Vec<Vec<Fp3>> = (0..num_cols)
                    .map(|_| (0..n).map(|_| rand_ext3(&mut rng)).collect())
                    .collect();

                let mut row_major: Vec<u64> = Vec::with_capacity(n * num_cols * 3);
                for r in 0..n {
                    for col in &columns {
                        row_major.push(*col[r].value()[0].value());
                        row_major.push(*col[r].value()[1].value());
                        row_major.push(*col[r].value()[2].value());
                    }
                }

                let weights_u64 = coset_weights_u64(n, COSET_OFFSET);
                let weights_fp = coset_weights(n, COSET_OFFSET);
                let inv_tw =
                    TwoHalfTwiddles::<GoldilocksField>::new(log_n, true).expect("inv twiddles");
                let fwd_tw =
                    TwoHalfTwiddles::<GoldilocksField>::new(log_lde, false).expect("fwd twiddles");

                let (handle, _lde) =
                    math_cuda::lde::coset_lde_ext3_row_major_with_merkle_tree_keep(
                        &row_major,
                        n,
                        num_cols,
                        blowup,
                        &weights_u64,
                        true,
                    )
                    .expect("new ext3 row-major GPU pipeline");
                let gpu_root = handle.tree.as_ref().expect("resident merkle tree").root;

                let cpu_root =
                    cpu_ext3_row_major_merkle_root(&columns, blowup, &weights_fp, &inv_tw, &fwd_tw);

                assert_eq!(
                    gpu_root, cpu_root,
                    "new ext3 row-major pipeline root mismatch: log_n={log_n} blowup={blowup} num_cols={num_cols}"
                );
            }
        }
    }
}
