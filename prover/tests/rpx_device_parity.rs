//! The RPX device kernels must produce the host's bytes — the phase-2 gate for
//! lane K, the twin of `crypto/math-cuda/tests/blake3_fused_parity.rs` (Batch
//! backend through the fused LDE+commit pipelines) plus the FRI-layer tree
//! (Pair backend), the comp-poly tree and the bare permutation.
//!
//! Lives in the prover crate rather than `math-cuda` because the host side —
//! `RpxStarkHash`, `AlgebraicBatchBackend`, `AlgebraicPairBackend`, `Rpx256` —
//! lives here; `math-cuda` is a dev-dependency of this crate, not the reverse.
//!
//! Every comparison is byte for byte or lane for lane against the production
//! host path; a tamper arm proves the equalities are not vacuous. Needs a GPU:
//!
//!   cargo test -p lambda-vm-prover --release --features cuda --test rpx_device_parity -- --nocapture
#![cfg(feature = "cuda")]

use crypto::merkle_tree::merkle::MerkleTree;
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use lambda_vm_prover::lfm::algebraic_commit::{
    AlgebraicBatchBackend, AlgebraicPairBackend, RpxCommit, RpxStarkHash,
};
use lambda_vm_prover::lfm::hash::LfmHasher;
use lambda_vm_prover::lfm::rpx::Rpx256;
use lambda_vm_prover::tables::types::FE;
use math::fft::two_half_fft::TwoHalfTwiddles;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::polynomial::Polynomial;
use stark::prover::{GenericProver, IsStarkProver};

/// The RPX prover, named: these tests compare against the CUDA RPX kernels, so
/// the CPU side must say RPX explicitly.
type Prover<F, E, PI> = GenericProver<F, E, PI, RpxStarkHash>;

type Ext3 = Degree3GoldilocksExtensionField;
type Fp3 = FieldElement<Ext3>;
type Fp = FieldElement<GoldilocksField>;

const P: u64 = 0xFFFF_FFFF_0000_0001;

/// splitmix64 — deterministic inputs from a seed, with no `rand` dependency
/// (the prover crate carries none for tests).
struct SplitMix(u64);

impl SplitMix {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}
const COSET_OFFSET: u64 = 7;

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

fn canonical(v: u64) -> u64 {
    if v >= P { v - P } else { v }
}

// ===========================================================================
// The bare permutation: device vs the host oracle, lane for lane.
// ===========================================================================

/// Random states plus the two edge states the host-KAT names; raw (`≥ p`)
/// lanes included, since that is what an LDE buffer holds.
fn probe_states(seed: u64, n: usize) -> Vec<[u64; 12]> {
    let mut rng = SplitMix::new(seed);
    let mut states = vec![[0u64; 12], [P - 1; 12]];
    for k in 0..n {
        let mut s = [0u64; 12];
        for (i, lane) in s.iter_mut().enumerate() {
            // Every fifth lane is a raw twin `c + p` of a small canonical `c`.
            *lane = if (k + i) % 5 == 0 {
                rng.next() % 0xFFFF_FFFF + P
            } else {
                rng.next() % P
            };
        }
        states.push(s);
    }
    states
}

#[test]
fn rpx_device_permutation_matches_the_host_oracle() {
    let states = probe_states(0x0052_5058, 256);
    let got = math_cuda::rpx::permute_probe(&states).expect("device permute probe");
    assert_eq!(got.len(), states.len());
    for (n, (input, out)) in states.iter().zip(got.iter()).enumerate() {
        let want = Rpx256.permute(core::array::from_fn(|i| FE::from(input[i])));
        for (i, (o, w)) in out.iter().zip(want.iter()).enumerate() {
            let w = canonical(*w.value());
            // RAW comparison: the device canonicalises its output, and that
            // loop is part of what is pinned (R-952).
            assert_eq!(*o, w, "state {n} lane {i}: device {o} vs host {w}");
        }
    }
}

// ===========================================================================
// The Batch backend through the fused row-major LDE + commit pipelines.
// ===========================================================================

fn cpu_row_major_rpx_root(
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
            .expect("CPU RPX commit");
    root
}

/// Device fused row-major LDE + RPX leaves + Merkle, root only.
fn gpu_fused_rpx_root(columns: &[Vec<u64>], blowup: usize, weights_u64: &[u64]) -> [u8; 32] {
    let n = columns[0].len();
    let num_cols = columns.len();
    let mut row_major = vec![0u64; n * num_cols];
    for (c, col) in columns.iter().enumerate() {
        for (r, &v) in col.iter().enumerate() {
            row_major[r * num_cols + c] = v;
        }
    }
    let (handle, _lde) = math_cuda::lde::coset_lde_row_major_with_merkle_tree_keep(
        &row_major,
        None,
        math_cuda::DeviceHash::Rpx256,
        n,
        num_cols,
        blowup,
        weights_u64,
        true,
        true,
    )
    .expect("fused RPX GPU pipeline");
    handle.tree.as_ref().expect("resident merkle tree").root
}

#[test]
fn rpx_fused_base_root_matches_cpu() {
    for log_n in [4usize, 6, 8, 10] {
        for blowup in [2usize, 4] {
            for num_cols in [1usize, 3, 8, 9] {
                let n = 1usize << log_n;
                let log_lde = (n * blowup).trailing_zeros() as usize;
                let mut rng = SplitMix::new((log_n * 1000 + blowup * 100 + num_cols) as u64);
                let columns: Vec<Vec<u64>> = (0..num_cols)
                    .map(|_| (0..n).map(|_| rng.next()).collect())
                    .collect();
                let weights_u64 = coset_weights_u64(n, COSET_OFFSET);
                let weights_fp = coset_weights(n, COSET_OFFSET);
                let inv_tw =
                    TwoHalfTwiddles::<GoldilocksField>::new(log_n, true).expect("inv twiddles");
                let fwd_tw =
                    TwoHalfTwiddles::<GoldilocksField>::new(log_lde, false).expect("fwd twiddles");

                let gpu_root = gpu_fused_rpx_root(&columns, blowup, &weights_u64);
                let cpu_root =
                    cpu_row_major_rpx_root(&columns, blowup, &weights_fp, &inv_tw, &fwd_tw);
                assert_eq!(
                    gpu_root, cpu_root,
                    "RPX fused root mismatch: log_n={log_n} blowup={blowup} num_cols={num_cols}"
                );
            }
        }
    }
}

fn rand_ext3(rng: &mut SplitMix) -> Fp3 {
    Fp3::new([
        Fp::from_raw(rng.next()),
        Fp::from_raw(rng.next()),
        Fp::from_raw(rng.next()),
    ])
}

fn cpu_ext3_row_major_rpx_root(
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
    let (_, root) = Prover::<GoldilocksField, Ext3, ()>::commit_rows_bit_reversed(&buf, num_cols)
        .expect("CPU ext3 RPX commit");
    root
}

#[test]
fn rpx_fused_ext3_root_matches_cpu() {
    for log_n in [4usize, 6, 8] {
        for blowup in [2usize, 4] {
            for num_cols in [1usize, 3, 5] {
                let n = 1usize << log_n;
                let log_lde = (n * blowup).trailing_zeros() as usize;
                let mut rng = SplitMix::new((log_n * 1000 + blowup * 100 + num_cols) as u64 + 4242);
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
                        math_cuda::DeviceHash::Rpx256,
                        n,
                        num_cols,
                        blowup,
                        &weights_u64,
                        true,
                    )
                    .expect("fused ext3 RPX GPU pipeline");
                let gpu_root = handle.tree.as_ref().expect("resident merkle tree").root;
                let cpu_root =
                    cpu_ext3_row_major_rpx_root(&columns, blowup, &weights_fp, &inv_tw, &fwd_tw);
                assert_eq!(
                    gpu_root, cpu_root,
                    "RPX fused ext3 root mismatch: log_n={log_n} blowup={blowup} num_cols={num_cols}"
                );
            }
        }
    }
}

// ===========================================================================
// The comp-poly tree from interleaved ext3 parts (the `gpu_lde` site), against
// the same row-pair leaf layout committed on the CPU.
// ===========================================================================

#[test]
fn rpx_comp_poly_tree_root_matches_cpu() {
    for (log_lde, m) in [(4usize, 1usize), (6, 2), (10, 3), (12, 4)] {
        let lde_size = 1usize << log_lde;
        let mut rng = SplitMix::new((log_lde * 10 + m) as u64 + 99);
        let parts: Vec<Vec<Fp3>> = (0..m)
            .map(|_| (0..lde_size).map(|_| rand_ext3(&mut rng)).collect())
            .collect();

        // Interleaved `[a0,a1,a2,b0,b1,b2,…]` per part for the device.
        let parts_u64: Vec<Vec<u64>> = parts
            .iter()
            .map(|p| {
                p.iter()
                    .flat_map(|e| e.value().iter().map(|c| *c.value()))
                    .collect()
            })
            .collect();
        let raw_parts: Vec<&[u64]> = parts_u64.iter().map(|p| p.as_slice()).collect();
        let dev_tree = math_cuda::rpx::build_comp_poly_tree_from_evals_ext3_keep(&raw_parts)
            .expect("device comp-poly tree");
        assert_eq!(dev_tree.leaves_len, lde_size / 2);

        // Row-major with `m` ext3 columns: row r = [part_0[r], …, part_{m-1}[r]].
        let mut buf: Vec<Fp3> = vec![Fp3::from(0u64); lde_size * m];
        for (c, p) in parts.iter().enumerate() {
            for (r, v) in p.iter().enumerate() {
                buf[r * m + c] = *v;
            }
        }
        let (_, cpu_root) = Prover::<GoldilocksField, Ext3, ()>::commit_rows_bit_reversed(&buf, m)
            .expect("CPU comp-poly commit");
        assert_eq!(
            dev_tree.root, cpu_root,
            "RPX comp-poly root mismatch: log_lde={log_lde} m={m}"
        );
    }
}

// ===========================================================================
// The Pair backend: the FRI-layer tree, node for node.
// ===========================================================================

fn fri_layer_parity(log_num_leaves: u32, seed: u64) {
    let num_leaves = 1usize << log_num_leaves;
    let mut rng = SplitMix::new(seed);
    let evals: Vec<Fp3> = (0..num_leaves * 2).map(|_| rand_ext3(&mut rng)).collect();

    let mut evals_u64 = Vec::with_capacity(evals.len() * 3);
    for e in &evals {
        for c in e.value().iter() {
            evals_u64.push(*c.value());
        }
    }
    let leaves: Vec<[Fp3; 2]> = evals.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
    let cpu_tree = MerkleTree::<AlgebraicPairBackend<Ext3, RpxCommit>>::build(&leaves).unwrap();
    let cpu_nodes = cpu_tree.nodes();

    let gpu_bytes = math_cuda::rpx::build_fri_layer_tree_from_evals_ext3(&evals_u64).unwrap();
    assert_eq!(cpu_nodes.len() * 32, gpu_bytes.len(), "node count");
    for (i, expected) in cpu_nodes.iter().enumerate() {
        assert_eq!(
            &gpu_bytes[i * 32..(i + 1) * 32],
            &expected[..],
            "node {i} mismatch at log_num_leaves={log_num_leaves}"
        );
    }
}

/// Small trees: every level fits the block, so the tail kernel builds the
/// whole tree in one launch.
#[test]
fn rpx_fri_layer_tree_small() {
    for log in 1u32..=6 {
        fri_layer_parity(log, 100 + log as u64);
    }
}

/// Deep enough that the per-level kernel runs first and hands over to the tail
/// partway up — the launch path a real commit takes.
#[test]
fn rpx_fri_layer_tree_medium() {
    for log in [10u32, 12, 14] {
        fri_layer_parity(log, 500 + log as u64);
    }
}

// ===========================================================================
// The column-range leaves (preprocessed tables), leaf for leaf against the
// Batch backend's `hash_data` over the same felts.
// ===========================================================================

#[test]
fn rpx_row_major_range_leaves_match_cpu() {
    let log_n = 6u32;
    let n = 1usize << log_n;
    let m = 7usize;
    let mut rng = SplitMix::new(0xBEEF);
    let data: Vec<u64> = (0..n * m).map(|_| rng.next()).collect();
    let reverse_index = |i: usize| -> usize { (i as u64).reverse_bits() as usize >> (64 - log_n) };

    for (cs, ce) in [(0usize, m), (0, 3), (3, m), (2, 5)] {
        let gpu = math_cuda::rpx::leaves_base_row_major_row_pair_range(&data, m, cs, ce, n)
            .expect("device ranged leaves");
        assert_eq!(gpu.len(), (n / 2) * 32);
        for leaf in 0..n / 2 {
            let mut felts: Vec<Fp> = Vec::with_capacity(2 * (ce - cs));
            for k in 0..2 {
                let br = reverse_index(2 * leaf + k);
                for c in cs..ce {
                    felts.push(Fp::from_raw(data[br * m + c]));
                }
            }
            let want =
                <AlgebraicBatchBackend<GoldilocksField, RpxCommit> as IsMerkleTreeBackend>::hash_data(
                    &felts,
                );
            assert_eq!(
                &gpu[leaf * 32..(leaf + 1) * 32],
                &want[..],
                "ranged leaf {leaf} mismatch for columns [{cs}, {ce})"
            );
        }
    }
}

// ===========================================================================
// Negative control: one corrupted input element must move the device root.
// ===========================================================================

#[test]
fn rpx_fused_tamper_diverges() {
    let n = 1usize << 6;
    let num_cols = 3usize;
    let mut rng = SplitMix::new(777);
    let columns: Vec<Vec<u64>> = (0..num_cols)
        .map(|_| (0..n).map(|_| rng.next()).collect())
        .collect();
    let weights_u64 = coset_weights_u64(n, COSET_OFFSET);

    let honest = gpu_fused_rpx_root(&columns, 2, &weights_u64);
    let mut tampered = columns.clone();
    tampered[1][n / 2] ^= 1;
    let forged = gpu_fused_rpx_root(&tampered, 2, &weights_u64);
    assert_ne!(
        honest, forged,
        "a corrupted input element must move the root"
    );
}
