//! The fused GPU pipelines under BLAKE3 must produce the same Merkle root as
//! the CPU BLAKE3 path — the fused twin of `merkle_root_parity`, which pins
//! the same pipelines under keccak.
//!
//! `merkle_root_parity` covers the leaf/tree kernels through the generic
//! entry points; this file drives the FUSED entries (`coset_lde_row_major_
//! with_merkle_tree_keep` and the ext3 variant) with `DeviceHash::Blake3`,
//! i.e. exactly the dispatch the production commit path takes on a cuda
//! build, and compares against the CPU row-major LDE + `Blake3StarkHash`
//! commit byte for byte. A tamper arm proves the comparison is not vacuous.

use math::fft::two_half_fft::TwoHalfTwiddles;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::polynomial::Polynomial;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use stark::config::Blake3StarkHash;
use stark::prover::{GenericProver, IsStarkProver};

/// The BLAKE3 prover, named: these tests compare against the CUDA BLAKE3
/// kernels, so the CPU side must say BLAKE3 explicitly rather than follow
/// the default alias — the comparison stays meaningful even if the default
/// ever moves.
type Prover<F, E, PI> = GenericProver<F, E, PI, Blake3StarkHash>;

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

fn cpu_row_major_blake3_root(
    columns: &[Vec<u64>],
    blowup: usize,
    weights: &[Fp],
    inv_tw: &TwoHalfTwiddles<GoldilocksField>,
    fwd_tw: &TwoHalfTwiddles<GoldilocksField>,
) -> [u8; 32] {
    let n = columns[0].len();
    let num_cols = columns.len();

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
            .expect("CPU BLAKE3 commit");

    root
}

/// Device fused row-major LDE + BLAKE3 leaves + Merkle, root only.
fn gpu_fused_blake3_root(columns: &[Vec<u64>], blowup: usize, weights_u64: &[u64]) -> [u8; 32] {
    let n = columns[0].len();
    let num_cols = columns.len();

    // Row-major input: data[row * num_cols + col].
    let mut row_major = vec![0u64; n * num_cols];
    for (c, col) in columns.iter().enumerate() {
        for (r, &v) in col.iter().enumerate() {
            row_major[r * num_cols + c] = v;
        }
    }

    let (handle, _lde) = math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep(
        &row_major,
        None,
        math_cuda::DeviceHash::Blake3,
        n,
        num_cols,
        blowup,
        weights_u64,
        true,
    )
    .expect("fused BLAKE3 GPU pipeline");
    handle.tree.as_ref().expect("resident merkle tree").root
}

#[test]
fn blake3_fused_base_root_matches_cpu() {
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

                let gpu_root = gpu_fused_blake3_root(&columns, blowup, &weights_u64);
                let cpu_root =
                    cpu_row_major_blake3_root(&columns, blowup, &weights_fp, &inv_tw, &fwd_tw);

                assert_eq!(
                    gpu_root, cpu_root,
                    "BLAKE3 fused root mismatch: log_n={log_n} blowup={blowup} num_cols={num_cols}"
                );
            }
        }
    }
}

fn rand_ext3(rng: &mut ChaCha8Rng) -> Fp3 {
    Fp3::new([
        FieldElement::<GoldilocksField>::from_raw(rng.r#gen::<u64>()),
        FieldElement::<GoldilocksField>::from_raw(rng.r#gen::<u64>()),
        FieldElement::<GoldilocksField>::from_raw(rng.r#gen::<u64>()),
    ])
}

fn cpu_ext3_row_major_blake3_root(
    columns: &[Vec<Fp3>],
    blowup: usize,
    weights: &[Fp],
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
        .expect("CPU ext3 BLAKE3 commit");

    root
}

#[test]
fn blake3_fused_ext3_root_matches_cpu() {
    const COSET_OFFSET: u64 = 7;

    for log_n in [4usize, 6, 8] {
        for blowup in [2usize, 4] {
            for num_cols in [1usize, 3, 5] {
                let n = 1usize << log_n;
                let log_lde = (n * blowup).trailing_zeros() as usize;
                let mut rng = ChaCha8Rng::seed_from_u64(
                    (log_n * 1000 + blowup * 100 + num_cols) as u64 + 4242,
                );

                let columns: Vec<Vec<Fp3>> = (0..num_cols)
                    .map(|_| (0..n).map(|_| rand_ext3(&mut rng)).collect())
                    .collect();

                // Row-major ext3 = row-major base with 3 * num_cols lanes.
                let mut row_major = vec![0u64; n * num_cols * 3];
                for (c, col) in columns.iter().enumerate() {
                    for (r, v) in col.iter().enumerate() {
                        for k in 0..3 {
                            row_major[(r * num_cols + c) * 3 + k] = *v.value()[k].value();
                        }
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
                        math_cuda::DeviceHash::Blake3,
                        n,
                        num_cols,
                        blowup,
                        &weights_u64,
                        true,
                    )
                    .expect("fused ext3 BLAKE3 GPU pipeline");
                let gpu_root = handle.tree.as_ref().expect("resident merkle tree").root;

                let cpu_root =
                    cpu_ext3_row_major_blake3_root(&columns, blowup, &weights_fp, &inv_tw, &fwd_tw);

                assert_eq!(
                    gpu_root, cpu_root,
                    "BLAKE3 fused ext3 root mismatch: log_n={log_n} blowup={blowup} num_cols={num_cols}"
                );
            }
        }
    }
}

/// Negative control: one corrupted input element must move the device root.
/// Proves the equality assertions above compare live data, not fixed points.
#[test]
fn blake3_fused_tamper_diverges() {
    const COSET_OFFSET: u64 = 7;
    let n = 1usize << 6;
    let num_cols = 3usize;
    let mut rng = ChaCha8Rng::seed_from_u64(777);

    let columns: Vec<Vec<u64>> = (0..num_cols)
        .map(|_| (0..n).map(|_| rng.r#gen::<u64>()).collect())
        .collect();
    let weights_u64 = coset_weights_u64(n, COSET_OFFSET);

    let honest = gpu_fused_blake3_root(&columns, 2, &weights_u64);

    let mut tampered = columns.clone();
    tampered[1][n / 2] ^= 1;
    let forged = gpu_fused_blake3_root(&tampered, 2, &weights_u64);

    assert_ne!(
        honest, forged,
        "a corrupted input element must move the root"
    );
}
